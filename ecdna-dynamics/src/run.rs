use crate::dynamics::{
    Dynamic, Dynamics, GillespieT, MeanDyn, Moments, NMinus, NPlus,
};
use anyhow::ensure;
use ecdna_data::data::{
    Data, EcDNADistribution, EcDNASummary, Entropy, Frequency, Mean,
};
use ecdna_sim::event::{AdvanceRun, Event, GillespieTime};
use ecdna_sim::process::BirthDeathProcess;
use ecdna_sim::rate::{GetRates, Range, Rates};
use ecdna_sim::{NbIndividuals, Seed};
use enum_dispatch::enum_dispatch;
use rand::{Rng, SeedableRng};
use rand_distr::{Binomial, Distribution};
use rand_pcg::Pcg64Mcg;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Number of ecDNA copies within a cell. We assume that a cell cannot have more
/// than 65535 copies (`u16` is 2^16 - 1 = 65535 copies).
pub type DNACopy = u16;

/// Simulation of an exponentially growing tumour, that is one realization of
/// the stochastic birth-death process.
///
/// The `Run` uses the [typestate pattern]. The possible states are [`Started`]
/// and [`Ended`].
///
/// [typestate pattern]: https://github.com/cbiffle/m4vga-rs/blob/a1e2ba47eaeb4864f0d8b97637611d9460ce5c4d/notes/20190131-typestate.md
#[derive(Clone, Debug)]
pub struct Run<S: RunState> {
    state: S,

    /// Index of the run
    pub idx: usize,
    /// Birth-Death process created from some rates.
    bd_process: BirthDeathProcess,
    /// Initial state of the system
    pub init_state: InitialState,
    /// How many times the run has been restarted (longitudinal analyses)
    pub restarted: u8,
    seed: Seed,
    rng: Pcg64Mcg,
}

impl<S> Run<S>
where
    S: RunState,
{
    pub fn get_seed(&self) -> &Seed {
        &self.seed
    }
}

/// The simulation of the run has started, the stochastic birth-death process
/// has started looping over the iterations.
#[derive(Debug)]
pub struct Started {
    /// State of the system at one particular iteration
    system: System,
}

/// The simulation of the run has ended, which is ready to be saved.
#[derive(Clone, Debug)]
pub struct Ended {
    /// Gillespie time at the end of the run
    gillespie_time: GillespieTime,
    /// The iteration number at which the run has stopped
    last_iter: usize,
    /// Data of interest for the last iteration
    data: Data,
    /// The run idx from which the sample was taken.
    sampled_run: Option<usize>,
}

pub trait RunState {}
impl RunState for Started {}
impl RunState for Ended {}

/// Create a new `Run` continuing from the last simulated event: the run will
/// start with the data of the ended `Run`.
impl From<Run<Ended>> for Run<Started> {
    fn from(run: Run<Ended>) -> Self {
        let last_iter = run.state.last_iter;
        let ecdna = run.state.data.ecdna.clone();
        let restarted = run.restarted + 1;
        let init_state = InitialState::new(ecdna, last_iter);
        let state = run.state.into();

        Run {
            idx: run.idx,
            bd_process: run.bd_process,
            state,
            init_state,
            restarted,
            seed: run.seed,
            rng: run.rng,
        }
    }
}

impl From<Ended> for Started {
    fn from(state: Ended) -> Self {
        let nminus = *state.data.ecdna.get_nminus();
        let ecdna_distr = EcDNADistributionNPlus::from(state.data.ecdna);
        let event =
            Event { kind: AdvanceRun::Init, time: state.gillespie_time };
        let system = System { nminus, ecdna_distr, event };

        Started { system }
    }
}

impl Run<Started> {
    pub fn new(
        idx: usize,
        time: f32,
        init_state: InitialState,
        rates: Rates,
        seed: Seed,
        rng: &mut Pcg64Mcg,
    ) -> Self {
        //! Initialize a stoachastic realization of a birth-death process.

        Run {
            idx,
            bd_process: BirthDeathProcess::new(rates, rng),
            init_state: init_state.clone(),
            state: Started {
                system: System {
                    nminus: *init_state.init_distribution.get_nminus(),
                    ecdna_distr: EcDNADistributionNPlus::from(
                        init_state.init_distribution,
                    ),
                    event: Event { kind: AdvanceRun::Init, time },
                },
            },
            restarted: 0,
            seed,
            rng: Pcg64Mcg::seed_from_u64(*seed.get_seed()),
        }
    }

    pub fn get_nminus(&self) -> &NbIndividuals {
        //! Number of cells w/o any ecDNA copies for the current iteration.
        &self.state.system.nminus
    }

    pub fn get_nplus(&self) -> NbIndividuals {
        //! Number of cells w at least one ecDNA copy for the current iteration.
        self.state.system.nplus_cells()
    }

    pub fn get_init_state(&self) -> &InitialState {
        &self.init_state
    }

    pub fn get_init_iter(&self) -> &usize {
        &self.init_state.init_iter
    }

    pub fn mean_ecdna(&self) -> f32 {
        //! Average of ecDNA copy number distribution within the population for
        //! the current iteration.
        self.state
            .system
            .ecdna_distr
            .compute_mean(&(self.get_nplus() + self.get_nminus()))
    }

    pub fn variance_ecdna(&self, mean: &f32) -> f32 {
        //! Variance of ecDNA copy number distribution within the population for
        //! the current iteration.
        if self.state.system.ecdna_distr.is_empty() {
            panic!(
                "Cannot compute the variance of an empty ecDNA distribution"
            )
        } else {
            let nb_nplus = self.get_nplus();
            let nb_nminus = self.get_nminus();

            self.state.system.ecdna_distr[..]
                .iter()
                .chain(std::iter::repeat(&0u16).take(*nb_nminus as usize))
                .map(|value| {
                    let diff = mean - (*value as f32);
                    diff * diff
                })
                .sum::<f32>()
                / ((nb_nminus + nb_nplus) as f32)
        }
    }

    pub fn get_gillespie_event(&self) -> &Event {
        &self.state.system.event
    }

    pub fn simulate(
        mut self,
        dynamics: &mut Option<&mut Dynamics>,
        max_cells: &NbIndividuals,
        max_iter: usize,
    ) -> Run<Ended> {
        //! Simulate one realisation of the birth-death stochastic process.
        //!
        //! If the some `dynamics` are given, those quantities will be
        //! calculated using the [`Update`] method.
        let mut iter = self.init_state.init_iter;

        /*
        // if we start from an initial state having more than one cell, must prepare
        // the dynamics by filling them with const values, unless we are restarting
        // the run
        if initial_fill_dynamics {
            if let Some(dynamics) = dynamics {
                // -1 because the init has already been carried out previously
                for _ in 0..self.state.system.ntot() - 1 {
                    for d in dynamics.iter_mut() {
                        d.update(&self);
                    }
                }
            }
        }
        */

        let (time, condition, ntot) = {
            loop {
                let ntot = self.get_nplus() + *self.get_nminus();

                // StopIteration appears when there are no cells anymore (due to
                // cell death), when the iteration has reached the max number of
                // iterations nb_iter >= self.max_iter or maximal number of cells
                // i.e. when the iteration has generated a tumor of max_cells size
                if ntot == 0u64 {
                    break (
                        self.state.system.event.time,
                        EndRun::NoIndividualsLeft,
                        ntot,
                    );
                };
                if iter >= max_iter {
                    break (
                        self.state.system.event.time,
                        EndRun::MaxItersReached,
                        ntot,
                    );
                };
                if &ntot >= max_cells {
                    break (
                        self.state.system.event.time,
                        EndRun::MaxIndividualsReached,
                        ntot,
                    );
                }

                // Compute the next event using Gillespie algorithm based on the
                // stochastic process
                let event = self.bd_process.gillespie(
                    self.get_nplus(),
                    *self.get_nminus(),
                    &mut self.rng,
                );
                self.update(event, dynamics);
                iter += 1;
            }
        };

        if let BirthDeathProcess::PureBirth(_) = &self.bd_process {
            assert!(ntot > 0, "No cells found with PureBirth process")
        }

        let (idx, process, init_state, seed, rng, restarted) = (
            self.idx,
            self.bd_process.clone(),
            self.init_state.clone(),
            self.seed,
            Pcg64Mcg::from_rng(self.rng.clone())
                .expect("Cannot create rng from the run"),
            self.restarted,
        );

        let data = self.create_data(&condition);

        Run {
            idx,
            bd_process: process,
            init_state,
            state: Ended {
                data,
                gillespie_time: time,
                sampled_run: None,
                last_iter: iter,
            },
            restarted,
            seed,
            rng,
        }
    }

    fn create_data(self, stop_condition: &EndRun) -> Data {
        let ntot = self.state.system.ntot();
        let ecdna_distr = self.state.system.ecdna_distr.0;
        let nplus = ecdna_distr.len() as NbIndividuals;
        let mut counts: HashMap<DNACopy, NbIndividuals> = HashMap::new();

        if ecdna_distr.is_empty() {
            if ntot == 0u64 {
                match stop_condition {
                    EndRun::NoIndividualsLeft => {
                        // no cells left, return NAN
                        return Data {
                            ecdna: EcDNADistribution::from(counts),
                            summary: EcDNASummary {
                                mean: Mean(f32::NAN),
                                frequency: Frequency(f32::NAN),
                                entropy: Entropy(f32::NAN),
                            },
                        };
                    }
                    _ => {
                        panic!(
                                "Found wrong value of ntot: found ntot = 0 but stop condition {:#?} instead of NoIndividualsLeft",
                                stop_condition
                                );
                    }
                }
            } else {
                // no nplus cells left, return 0
                counts.insert(0u16, ntot);
                return Data {
                    ecdna: EcDNADistribution::from(counts),
                    summary: EcDNASummary {
                        mean: Mean(0f32),
                        frequency: Frequency(0f32),
                        entropy: Entropy(0f32),
                    },
                };
            }
        } else if ntot == 0 {
            panic!("Found wrong value of ntot: cannot create data when ntot is 0 and the ecDNA distribution is not empty")
        } else if ntot < (ecdna_distr.len() as NbIndividuals) {
            panic!("Found wrong value of ntot: should be equal or greater than the number of cells w/ ecDNA, found smaller")
        }

        let mut sum = 0u64;
        for ecdna in ecdna_distr.into_iter() {
            sum += ecdna as u64;
            *counts.entry(ecdna).or_default() += 1;
        }
        // add nminus cells
        counts.insert(0u16, ntot - nplus);

        let ecdna = EcDNADistribution::from(counts);
        assert_eq!(ecdna.nb_cells(), ntot);
        let entropy = Entropy::try_from(&ecdna).unwrap();
        // do not try_from to avoid traversing the distribution vector
        let mean = Mean(sum as f32 / ecdna.nb_cells() as f32);
        let frequency = Frequency::try_from(&ecdna).unwrap();

        Data { ecdna, summary: EcDNASummary { mean, frequency, entropy } }
    }

    fn update(
        &mut self,
        next_event: Event,
        dynamics: &mut Option<&mut Dynamics>,
    ) {
        //! Update the run for the next iteration, according to the `next_event`
        //! sampled from Gillespie algorithm. This updates also the `dynamics`
        //! if present.
        if let AdvanceRun::Init = next_event.kind {
            panic!("Init state can only be used to initialize simulation!")
        }
        self.state.system.update(next_event, &mut self.rng);
        if let Some(dynamics) = dynamics {
            for d in dynamics.iter_mut() {
                // updates all dynamics according to the new state
                d.update(self);
            }
        }
    }
}

impl Run<Ended> {
    pub fn undersample_ecdna(
        mut self,
        nb_cells: &NbIndividuals,
        idx: usize,
    ) -> Self {
        //! Returns a copy of the run with subsampled ecDNA distribution
        let data = self
            .state
            .data
            .ecdna
            .undersample_data(nb_cells, &mut self.rng)
            .unwrap();

        assert_eq!(
            &data.ecdna.nb_cells(),
            nb_cells,
            "Wrong undersampling of the ecDNA distribution: {} cells expected after sampling, found {}, {:#?}", nb_cells, data.ecdna.nb_cells(), data.ecdna
        );

        Run {
            idx,
            bd_process: self.bd_process.clone(),
            init_state: self.init_state,
            state: Ended {
                data,
                gillespie_time: self.state.gillespie_time,
                last_iter: self.state.last_iter,
                sampled_run: Some(self.idx),
            },
            restarted: self.restarted,
            seed: self.seed,
            rng: self.rng,
        }
    }

    pub fn set_iter(&mut self, iter: usize) {
        self.state.last_iter = iter
    }

    pub fn save_ecdna_statistics(&self, path: &Path) {
        //! Save the ecDNA distribution, its entropy, its mean and the frequency
        //! of cells with ecDNA.
        self.state.data.save(path, &PathBuf::from(format!("{}", self.idx)));
    }

    pub fn nb_cells(&self) -> NbIndividuals {
        self.state.data.ecdna.nb_cells()
    }

    pub fn get_ecdna(&self) -> &EcDNADistribution {
        //! The ecDNA distribution for the last iteration.
        &self.state.data.ecdna
    }

    pub fn get_mean(&self) -> &f32 {
        //! The mean of ecDNA distribution for the last iteration.
        &self.state.data.summary.mean
    }

    pub fn get_frequency(&self) -> &f32 {
        //! The frequency of cells w/ ecDNA for the last iteration.
        &self.state.data.summary.frequency
    }

    pub fn get_entropy(&self) -> &f32 {
        //! The entropy of the ecDNA distribution for the last iteration.
        &self.state.data.summary.entropy
    }

    pub fn get_parental_run(&self) -> &Option<usize> {
        //! The idx for the sampled run
        &self.state.sampled_run
    }

    pub fn rates(&self) -> [f32; 4] {
        self.bd_process.get_rates()
    }

    pub fn filename(&self) -> PathBuf {
        //! File path for the current run (used to save data)
        PathBuf::from(self.idx.to_string())
    }
}

// impl PartialEq for Range {
//     fn eq(&self, other: &Self) -> bool {
//         self.isbn == other.isbn
//     }
// }

#[derive(Debug, Clone)]
pub struct InitialState {
    init_iter: usize,
    init_distribution: EcDNADistribution,
}

impl InitialState {
    pub fn random(min: u16, max: u16, rng: &mut Pcg64Mcg) -> Self {
        //! Create an `InitialState` with one cell with a random number of ecDNA
        //! copies, sampled uniformly from a range defined by `min` and `max`.
        InitialState {
            init_iter: 0usize,
            init_distribution: EcDNADistribution::from(vec![Range::new(
                min, max,
            )
            .sample_uniformly(rng)]),
        }
    }

    pub fn new_from_one_copy(init_copy: DNACopy, init_iter: usize) -> Self {
        //! Create an `InitialState` with one cell with `init_copy` number of
        //! ecDNA copies.
        InitialState {
            init_iter,
            init_distribution: EcDNADistribution::from(vec![init_copy]),
        }
    }

    pub fn new(
        init_distribution: EcDNADistribution,
        init_iter: usize,
    ) -> Self {
        InitialState { init_iter, init_distribution }
    }

    pub fn get_distribution(&self) -> &EcDNADistribution {
        &self.init_distribution
    }
}

#[derive(Debug)]
struct System {
    nminus: NbIndividuals,
    ecdna_distr: EcDNADistributionNPlus,
    event: Event,
}

impl System {
    fn update(&mut self, event: Event, rng: &mut Pcg64Mcg) {
        //! Update the state of the system based on event sampled from Gillespie
        match event.kind {
            AdvanceRun::Proliferate1 => {
                // Generate randomly a `NPlus` cell that will proliferate
                // next and update the ecdna distribution as well.
                // Match the cell that has been generated by the
                // proliferation of the nplus cell
                match self.ecdna_distr.proliferate_nplus(rng) {
                    // Updates of nplus cells is done by ecdna_distr
                    Cell::NPlus => {}
                    Cell::NMinus => {
                        self.nminus += 1;
                    }
                }
            }
            AdvanceRun::Proliferate2 => {
                self.nminus += 1;
            }
            // Updates of nplus cells is done by ecdna_distr
            AdvanceRun::Die1 => {
                self.ecdna_distr.kill_nplus(rng);
            }
            AdvanceRun::Die2 => {
                self.nminus -= 1;
            }
            AdvanceRun::Init => {
                panic!("Cannot update with event init");
            }
        }
        self.event = event;
    }

    fn nplus_cells(&self) -> NbIndividuals {
        self.ecdna_distr.get_nplus_cells()
    }

    pub fn ntot(&self) -> NbIndividuals {
        self.nplus_cells() + self.nminus
    }
}

/// The distribution of ecDNA copies without considering the cells w/o any ecDNA
/// copy.
#[derive(Clone, Debug, Default)]
struct EcDNADistributionNPlus(Vec<DNACopy>);

impl<Idx> std::ops::Index<Idx> for EcDNADistributionNPlus
where
    Idx: std::slice::SliceIndex<[DNACopy]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

/// Convert the histogram representation of the ecDNA distribution into a single-cell
/// representantion, where each entry is a cell with its ecDNA content. Remove
/// `NMinus` cells.
impl From<EcDNADistribution> for EcDNADistributionNPlus {
    fn from(ecdna: EcDNADistribution) -> Self {
        EcDNADistributionNPlus(ecdna.into_vec_no_minus())
    }
}

impl EcDNADistributionNPlus {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get_nplus_cells(&self) -> NbIndividuals {
        self.0.len() as NbIndividuals
    }

    fn pick_nplus_cell(&self, rng: &mut Pcg64Mcg) -> usize {
        //! Pick a nplus cell at random
        assert!(!self.is_empty());
        rng.gen_range(0..self.get_nplus_cells() as usize)
    }

    fn kill_nplus(&mut self, rng: &mut Pcg64Mcg) {
        //! A `NPlus` dies and thus we remove its ecdna contribution from the
        //! state vector `ecdna_distribution`. Remember that we assume
        //! the order is not important for `ecdna_distribution`,
        //! since we only need to compute the mean and the variance from this
        //! vector.
        // generate randomly a `NPlus` cell that will die next
        let idx: usize = self.pick_nplus_cell(rng);

        // Update the distribution of copies of ecdna in the nplus population.
        // Note that `die_and_update_distribution` will returns the next cell
        // but also update the ecdna distribution `self.ecdna_distribution`
        self.0.swap_remove(idx);
    }

    fn proliferate_nplus(&mut self, rng: &mut Pcg64Mcg) -> Cell {
        //! Update the distribution of ec_dna per cell and flag if a new
        //! `NMinus` cell appeared after division. This happened if
        //! `daughter1` or `daughter2` is 0. Uneven segregation means that one
        //! `NMinus` cell is born, and the other daughter cell got all the
        //! available ec_dna.
        let idx = self.pick_nplus_cell(rng);
        let (daughter1, daughter2, uneven_segregation) =
            self.dna_segregation(idx, rng);

        if uneven_segregation {
            self.0[idx] = daughter1 + daughter2; // n + 0 = n = 0 + n
            Cell::NMinus
        } else {
            self.0[idx] = daughter1; // update old cell ecDNA copies
            self.0.push(daughter2); // create new NPlus cell
            Cell::NPlus
        }
    }

    fn dna_segregation(
        &self,
        idx: usize,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy, bool) {
        //! Simulate the proliferation of one `NPlus` cell and returns whether a
        //! uneven segregation occured (i.e. one daughter cell inherited all
        //! ecDNA material from mother cells). When a `idx` `NPlus` cell
        //! proliferates, we multiply the ec_dna present in that cell by the
        //! `FITNESS` of the `NPlus` cells compared to the `NMinus` cells (e.g.
        //! 1 for the neutral case). Then, we distribute the multiplied ec_dna
        //! material among two daughter cells according to a Binomial
        //! distribution with number of samples equal to the multiplied ec_dna
        //! material and the probability of 1/2.

        // Double the number of `NPlus` from the idx cell before proliferation
        // because a parental cell gives rise to 2 daughter cells. We also
        // multiply by `FITNESS` for models where `NPlus` cells have an
        // advantage over `NMinus` cells
        let available_dna: DNACopy = self.0[idx]
            .checked_add(self.0[idx])
            .expect("Overflow while segregating DNA into two daughter cells");
        assert_ne!(available_dna, 0);

        // draw the ec_dna that will be given to the daughter cells
        let bin = Binomial::new(available_dna as u64, 0.5).unwrap();
        let d_1: DNACopy = bin.sample(rng).try_into().unwrap();
        let d_2: DNACopy = available_dna - d_1;
        assert_ne!(d_1 + d_2, 0);
        assert!(
            d_1.checked_add(d_2).is_some(),
            "Overflow error while segregating DNA copies during proliferation"
        );

        // uneven_segregation happens if there is at least one zero and
        // `daughter_1` and `daughter2` cannot be both equal to 0
        let uneven = d_1.saturating_mul(d_2) == 0;

        (d_1, d_2, uneven)
    }

    fn compute_mean(&self, ntot: &NbIndividuals) -> f32 {
        if self.is_empty() {
            if ntot == &0u64 {
                f32::NAN
            } else {
                0f32
            }
        } else if ntot == &0 {
            panic!("Cannot compute the mean of an when ntot is 0")
        } else if ntot < &self.get_nplus_cells() {
            panic!("Found wrong value of ntot: should be equal or greater than the number of cells w/ ecDNA, found smaller")
        } else {
            let sum = self[..]
                .iter()
                .fold(0u64, |accum, &item| accum + (item as u64))
                as f32;
            assert!(sum > 0.0);
            let mean = sum / (*ntot as f32);
            assert!(
                !(mean.is_nan()),
                "Compute mean NaN from ecdna distribution vector"
            );
            mean
        }
    }
}

/// Type of the individuals simulated.
#[derive(PartialEq, Debug)]
enum Cell {
    /// An individual with at least one copy of ecDNA
    NPlus,
    /// An individual without any copy of ecDNA
    NMinus,
}

/// An event defines the outcome that the simulation must simulate for an
/// iteration
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum StartRun {
    // The initialization event, should be only used in the constructor
    Init,
    /// Restart appears when we have two measurements for the same quantity, for
    /// the same patient at more than one timepoint.
    Restart,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum EndRun {
    /// The event sampled stop the simulation because there are no individual
    /// left
    NoIndividualsLeft,
    /// The event sampled stop the simulation because there the maximal number
    /// of individual has been reached
    MaxIndividualsReached,
    /// The event sampled stop the simulation because there the maximal number
    /// of iterations has been reached
    MaxItersReached,
}

#[enum_dispatch]
/// When taking a subsample of the whole population, specify how to continue the
/// cell growth: either from a sample or from the whole tumour population.
pub trait ContinueGrowth {
    fn restart_growth(
        &self,
        run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>>;
}

/// Specify how to growth the population after a subsample has been taken (biospy
/// or cell culture).
#[enum_dispatch(ContinueGrowth)]
#[derive(Debug)]
pub enum Growth {
    /// Cell culture: growth restart from the subsample of the whole population
    /// since the subsample is a new cell culture.
    CellCulture,
    /// Patient tumour: growth continues from the whole population since the
    /// subsample is just a biopsy.
    PatientStudy,
}

#[derive(Debug, Default)]
pub struct CellCulture;

impl CellCulture {
    pub fn new() -> Self {
        CellCulture
    }
}

impl ContinueGrowth for CellCulture {
    fn restart_growth(
        &self,
        run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>> {
        //! In cell culture experiments, growth restart from subsample of the
        //! whole population.
        let idx = run.idx;
        let mut run = run.undersample_ecdna(sample_size, idx);
        run.set_iter(0);
        ensure!(&run.nb_cells() == sample_size);
        Ok(run.into())
    }
}

#[derive(Debug, Default)]
pub struct PatientStudy;

impl PatientStudy {
    pub fn new() -> Self {
        PatientStudy
    }
}

impl ContinueGrowth for PatientStudy {
    fn restart_growth(
        &self,
        run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>> {
        //! In patient studies, growth restart from the whole population.
        ensure!(&run.nb_cells() >= sample_size);
        Ok(run.into())
    }
}

/// The main trait for the `Dynamic` which updates the dynamical measurement
/// based on the state of the `Run` for each iteration. It allows the
/// communication between the `Dynamic` and the `Run`.
///
/// # How can I implement `Update`?
/// Types that are `Dynamic` must implement `Update` which defines how to update
/// the dynamical measurement based on the state of the `Run` at each iteration.
///
/// An example of `Dynamic`al measurement keeping track of the number of cells
/// with any copy of ecDNA per iteration:
///
/// ```no_run
/// use ecdna_dynamics::run::{Update, Run, Started};
/// use ecdna_sim::NbIndividuals;
///
/// pub struct NPlus {
///     /// Record the number of cells w/ ecDNA for each iteration.
///     nplus_dynamics: Vec<NbIndividuals>,
/// }
///
/// impl Update for NPlus {
///     fn update(&mut self, run: &Run<Started>) {
///         self.nplus_dynamics.push(run.get_nplus());
///     }
/// }
/// ```

#[enum_dispatch]
pub trait Update {
    /// Update the measurement based on the `run` for each iteration, i.e.
    /// defines how to interact with `Run` to update the quantity of interest
    /// for each iteration.
    fn update(&mut self, run: &Run<Started>);
}

impl Update for NPlus {
    fn update(&mut self, run: &Run<Started>) {
        self.store_nplus(run.get_nplus());
    }
}

impl Update for NMinus {
    fn update(&mut self, run: &Run<Started>) {
        self.store_nminus(*run.get_nminus());
    }
}

impl Update for MeanDyn {
    fn update(&mut self, run: &Run<Started>) {
        let mean = run.mean_ecdna();
        self.store_mean(mean);
    }
}

impl Update for Moments {
    fn update(&mut self, run: &Run<Started>) {
        let mean = run.mean_ecdna();
        self.store_moments(mean, run.variance_ecdna(&mean));
    }
}

impl Update for GillespieT {
    fn update(&mut self, run: &Run<Started>) {
        let gillespie_time = run.get_gillespie_event().time;
        self.store_time(
            self.get_previous_time()
                .map_or_else(|_| gillespie_time, |time| time + gillespie_time),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean_low_frequency() {
        let ntot = 2u64;
        let ecdna = EcDNADistributionNPlus(vec![60u16]);
        assert_eq!(30f32, ecdna.compute_mean(&ntot));
    }
}
