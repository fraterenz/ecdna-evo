use crate::abc::ABCRejection;
use crate::data::{Data, EcDNADistribution, Entropy, Frequency, Mean, ToFile};
use crate::dynamics::{Dynamics, Name, Update};
use crate::gillespie::{
    AdvanceRun, BirthDeathProcess, Event, GetRates, GillespieTime,
};
use crate::{NbIndividuals, Patient, Rates};
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng};
use rand_distr::{Binomial, Distribution};
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
pub struct Run<S: RunState> {
    state: S,

    /// Index of the run
    pub idx: usize,
    /// Parameters to configure the run
    pub parameters: Parameters,
}

/// The simulation of the run has started, the stochastic birth-death process
/// has started looping over the iterations.
pub struct Started {
    /// Stochastic process simulating tumour growth
    process: BirthDeathProcess,
    /// State of the system at one particular iteration
    system: System,
}

/// The simulation of the run has ended, which is ready to be saved.
pub struct Ended {
    #[allow(dead_code)] // TODO restart run with two timepoints
    /// Gillespie time at the end of the run
    gillespie_time: GillespieTime,
    #[allow(dead_code)] // TODO restart run with two timepoints
    /// Stopping condition
    stop: EndRun,
    /// Data of interest for the last iteration
    data: Data,
    /// Rates: the proliferation and death rates of the cells `NPlus` and
    /// `NMninus` cells respectively.
    rates: [f32; 4],
    /// The run idx from which the sample was taken.
    sampled_run: Option<usize>,
}

pub trait RunState {}
impl RunState for Started {}
impl RunState for Ended {}

impl Run<Started> {
    pub fn new(idx: usize, parameters: Parameters, rates: &Rates) -> Self {
        //! Use the `parameters` and the `rates` to initialize a realization of
        //! a birth-death stochastic process.

        Run {
            idx,
            parameters,
            state: Started {
                process: BirthDeathProcess::new(rates),
                system: System {
                    nminus: parameters.init_nminus,
                    ecdna_distr: EcDNADistributionNPlus::new(&parameters),
                    event: Event {
                        kind: AdvanceRun::Init,
                        time: parameters.init_time,
                    },
                },
            },
        }
    }

    pub fn get_nminus(&self) -> &NbIndividuals {
        //! Number of cells w/o any ecDNA copies for the current iteration.
        &self.state.system.nminus
    }

    pub fn get_nplus(&self) -> NbIndividuals {
        //! Number of cells w at least one ecDNA copy for the current iteration.
        self.state.system.get_nplus_cells()
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

    pub fn simulate(mut self, mut dynamics: Option<Dynamics>) -> Run<Ended> {
        //! Simulate one realisation of the birth-death stochastic process.
        //!
        //! If the some `dynamics` are given, those quantities will be
        //! calculated using the [`Update`] method.
        let mut iter = self.parameters.init_iter;
        let mut nplus = self.parameters.init_nplus as u64;
        let mut nminus = self.parameters.init_nminus;

        let (time, condition) = {
            loop {
                // Compute the next event using Gillespie algorithm based on the
                // stochastic process defined by `process`
                let event = self.state.process.gillespie(nplus, nminus);
                self.update(event, &mut dynamics);
                nplus = self.get_nplus();
                nminus = *self.get_nminus();
                iter += 1;

                // StopIteration appears when there are no cells anymore
                // (due to cell death), when the iteration has reached
                // the maximal number of iterations nb_iter >=
                // self.max_iter or maximal number of cells
                // (self.max_cells), i.e. when the iteration has
                // generated a tumor of self.max_cells size
                if nplus + nminus == 0u64 {
                    break (
                        self.state.system.event.time,
                        EndRun::NoIndividualsLeft,
                    );
                };
                if iter >= self.parameters.max_iter - 1usize {
                    break (
                        self.state.system.event.time,
                        EndRun::MaxItersReached,
                    );
                };
                if nplus + nminus >= self.parameters.max_cells {
                    break (
                        self.state.system.event.time,
                        EndRun::MaxIndividualsReached,
                    );
                }
            }
        };

        let ntot = nminus + nplus;

        if let BirthDeathProcess::PureBirth(_) = &self.state.process {
            assert!(ntot > 0, "No cells found with PureBirth process")
        }

        let (idx, parameters, rates) =
            (self.idx, self.parameters, self.state.process.get_rates());

        let data = self.create_data(&ntot, &condition);

        Run {
            idx,
            parameters,
            state: Ended {
                data,
                rates,
                gillespie_time: time,
                stop: condition,
                sampled_run: None,
            },
        }
    }

    fn create_data(
        self,
        ntot: &NbIndividuals,
        stop_condition: &EndRun,
    ) -> Data {
        let ecdna_distr = self.state.system.ecdna_distr.0;
        let nplus = ecdna_distr.len() as NbIndividuals;
        let mut counts: HashMap<DNACopy, NbIndividuals> = HashMap::new();

        if ecdna_distr.is_empty() {
            if ntot == &0u64 {
                match stop_condition {
                    EndRun::NoIndividualsLeft => {
                        // no cells left, return NAN
                        return Data {
                            ecdna: EcDNADistribution::from(counts),
                            mean: Mean(f32::NAN),
                            frequency: Frequency(f32::NAN),
                            entropy: Entropy(f32::NAN),
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
                return Data {
                    ecdna: EcDNADistribution::from(counts),
                    mean: Mean(0f32),
                    frequency: Frequency(0f32),
                    entropy: Entropy(0f32),
                };
            }
        } else if ntot == &0 {
            panic!("Found wrong value of ntot: cannot create data when ntot is 0 and the ecDNA distribution is not empty")
        } else if ntot < &(ecdna_distr.len() as NbIndividuals) {
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
        assert_eq!(ecdna.nb_cells(), *ntot);
        let entropy = Entropy::try_from(&ecdna).unwrap();
        // do not try_from to avoid traversing the distribution vector
        let mean = Mean(sum as f32 / ecdna.nb_cells() as f32);
        let frequency = Frequency::try_from(&ecdna).unwrap();

        Data { ecdna, mean, frequency, entropy }
    }

    fn update(&mut self, next_event: Event, dynamics: &mut Option<Dynamics>) {
        //! Update the run for the next iteration, according to the `next_event`
        //! sampled from Gillespie algorithm. This updates also the `dynamics`
        //! if present.
        if let AdvanceRun::Init = next_event.kind {
            panic!("Init state can only be used to initialize simulation!")
        }
        self.state.system.update(next_event);
        if let Some(dynamics) = dynamics {
            for d in dynamics.iter_mut() {
                // updates all dynamics according to the new state
                d.update(self);
            }
        }
    }
}

impl Run<Ended> {
    pub fn save(
        self,
        abspath: &Path,
        dynamics: &Option<Dynamics>,
        patient: &Option<Patient>,
    ) -> anyhow::Result<()> {
        //! Save the run the dynamics (updated for each iter) if present and the
        //! other quantities, computed at the end of the run such as the mean,
        //! frequency.
        //!
        //! Do not save the full ecDNA distribution when ABC.
        // idx of the run
        let abspath_d = abspath.join("dynamics");

        // 1. dynamics
        if let Some(ref dynamics) = dynamics {
            assert!(patient.is_none(), "Cannot have patient and dynamics");
            for d in dynamics.iter() {
                let file2path =
                    abspath_d.join(d.get_name()).join(self.filename());
                d.save(&file2path).unwrap();
            }
        }

        let abspath = {
            if let Some(patient) = &patient {
                abspath.join(patient.name.clone())
            } else {
                abspath.to_owned()
            }
        };

        if let Some(patient) = &patient {
            let results = ABCRejection::run(&self, patient);
            results.save(&abspath, &self.parameters.subsample)
        } else {
            self.state.data.save(
                &abspath,
                &self.filename(),
                &self.parameters.subsample,
            );
            Ok(())
        }
    }

    pub fn undersample_ecdna(
        &self,
        nb_cells: &NbIndividuals,
        rng: &mut SmallRng,
        idx: usize,
    ) -> Self {
        //! Returns a copy of the run with subsampled ecDNA distribution
        let data =
            self.state.data.ecdna.undersample_data(nb_cells, rng).unwrap();

        assert_eq!(
            &data.ecdna.nb_cells(),
            nb_cells,
            "Wrong undersampling of the ecDNA distribution: {} cells expected after sampling, found {}, {:#?}", nb_cells, data.ecdna.nb_cells(), data.ecdna
        );

        Run {
            idx,
            parameters: self.parameters,
            state: Ended {
                data,
                rates: self.state.rates,
                gillespie_time: self.state.gillespie_time,
                stop: self.state.stop,
                sampled_run: Some(self.idx),
            },
        }
    }

    pub fn get_rates(&self) -> [f32; 4] {
        //! f1, f2, d1, d2
        self.state.rates
    }

    pub fn get_nb_cells(&self) -> NbIndividuals {
        self.state.data.ecdna.nb_cells()
    }

    pub fn get_ecdna(&self) -> &EcDNADistribution {
        //! The ecDNA distribution for the last iteration.
        &self.state.data.ecdna
    }

    pub fn get_mean(&self) -> &f32 {
        //! The mean of ecDNA distribution for the last iteration.
        &self.state.data.mean
    }

    pub fn get_frequency(&self) -> &f32 {
        //! The frequency of cells w/ ecDNA for the last iteration.
        &self.state.data.frequency
    }

    pub fn get_entropy(&self) -> &f32 {
        //! The entropy of the ecDNA distribution for the last iteration.
        &self.state.data.entropy
    }

    pub fn get_parental_run(&self) -> &Option<usize> {
        //! The idx for the sampled run
        &self.state.sampled_run
    }

    pub fn restart(self) -> Run<Started> {
        //! Restart the simulation of the run setting the initial state to the
        //! last state of `self`.
        todo!()
    }

    fn filename(&self) -> PathBuf {
        //! File path for the current run (used to save data)
        PathBuf::from(self.idx.to_string())
    }
}

#[derive(Debug)]
struct System {
    nminus: NbIndividuals,
    ecdna_distr: EcDNADistributionNPlus,
    event: Event,
}

impl System {
    fn update(&mut self, event: Event) {
        //! Update the state of the system based on event sampled from Gillespie
        match event.kind {
            AdvanceRun::Proliferate1 => {
                // Generate randomly a `NPlus` cell that will proliferate
                // next and update the ecdna distribution as well.
                // Match the cell that has been generated by the
                // proliferation of the nplus cell
                match self.ecdna_distr.proliferate_nplus() {
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
                self.ecdna_distr.kill_nplus();
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

    fn get_nplus_cells(&self) -> NbIndividuals {
        self.ecdna_distr.get_nplus_cells()
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

impl EcDNADistributionNPlus {
    pub fn new(parameters: &Parameters) -> Self {
        //! Initialize the ecDNA distribution not considering any cell w/o any
        //! ecDNA copy. The distribution will be empty if the
        //! `parameters.init_copies` is 0.
        let mut ecdna = Vec::with_capacity(parameters.max_cells as usize);
        if parameters.init_copies > 0 {
            ecdna.push(parameters.init_copies);
        }
        EcDNADistributionNPlus(ecdna)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get_nplus_cells(&self) -> NbIndividuals {
        self.0.len() as NbIndividuals
    }

    fn pick_nplus_cell(&self) -> usize {
        //! Pick a nplus cell at random
        assert!(!self.is_empty());
        thread_rng().gen_range(0..self.get_nplus_cells() as usize)
    }

    fn kill_nplus(&mut self) {
        //! A `NPlus` dies and thus we remove its ecdna contribution from the
        //! state vector `ecdna_distribution`. Remember that we assume
        //! the order is not important for `ecdna_distribution`,
        //! since we only need to compute the mean and the variance from this
        //! vector.
        // generate randomly a `NPlus` cell that will die next
        let idx: usize = self.pick_nplus_cell();

        // Update the distribution of copies of ecdna in the nplus population.
        // Note that `die_and_update_distribution` will returns the next cell
        // but also update the ecdna distribution `self.ecdna_distribution`
        self.0.swap_remove(idx);
    }

    fn proliferate_nplus(&mut self) -> Cell {
        //! Update the distribution of ec_dna per cell and flag if a new
        //! `NMinus` cell appeared after division. This happened if
        //! `daughter1` or `daughter2` is 0. Uneven segregation means that one
        //! `NMinus` cell is born, and the other daughter cell got all the
        //! available ec_dna.
        let idx = self.pick_nplus_cell();
        let (daughter1, daughter2, uneven_segregation) =
            self.dna_segregation(idx);

        if uneven_segregation {
            self.0[idx] = daughter1 + daughter2; // n + 0 = n = 0 + n
            Cell::NMinus
        } else {
            self.0[idx] = daughter1; // update old cell ecDNA copies
            self.0.push(daughter2); // create new NPlus cell
            Cell::NPlus
        }
    }

    fn dna_segregation(&self, idx: usize) -> (DNACopy, DNACopy, bool) {
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
        let d_1: DNACopy =
            bin.sample(&mut rand::thread_rng()).try_into().unwrap();
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

#[derive(Debug, Clone, Copy)]
/// Parameters used to simulate tumour growth, they are shared across runs.
pub struct Parameters {
    /// The number of runs of the simulation
    pub nb_runs: usize,
    /// The maximal number of iterations after which one run is stopped, the
    /// same for each run
    pub max_cells: NbIndividuals,
    /// Max time is required when we use cell death in case there is strong cell
    /// death
    pub max_iter: usize,
    /// Initial copies of ecdna in `NPlus` cell for the run
    /// run
    pub init_copies: DNACopy,
    /// Whether to start with one `NPlus` cell for the run
    pub init_nplus: bool,
    /// Initial number of `NMinus` cells for the run
    pub init_nminus: NbIndividuals,
    /// Initial iteration from which start the simulation (always 0 except some
    /// data with two measurements at two different timepoints)
    pub init_iter: usize,
    /// Initial gillespie time from which start the simulation (always 0 except
    /// some data with two measurements at two different timepoints)
    pub init_time: f32,
    /// How many nplus cells to take while subsampling the ecDNA distribution
    pub subsample: Option<NbIndividuals>,
    pub verbosity: u8,
}

/// Start the simulation with one cells w/ one copy of ecDNA and no cells w/o
/// any ecDNA
impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            nb_runs: 1usize,
            max_cells: 10000,
            max_iter: 3 * 10000,
            init_copies: 1u16,
            init_nplus: true,
            init_nminus: 0u64,
            verbosity: 0u8,
            init_iter: 0usize,
            init_time: 0f32,
            subsample: None,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdna_distribution_is_empty_with_nminus_cells() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert!(ecdna.is_empty());
    }

    #[test]
    fn ecdna_distribution_getter_when_empty() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 0u64);
        assert!(ecdna.is_empty());
    }

    #[test]
    fn ecdna_distribution_from_default() {
        let p = Parameters::default();
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1u64);
        assert!(!ecdna.is_empty());
    }

    #[test]
    fn ecdna_distribution_from() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1);
        assert!(!ecdna.is_empty());
    }

    #[test]
    #[should_panic]
    fn ecdna_distribution_kill_nplus_empty() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
    }

    #[test]
    fn ecdna_distribution_kill_nplus_one_cell() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
        assert!(ecdna.is_empty());
        assert_eq!(ecdna.get_nplus_cells(), 0);
    }

    #[test]
    fn ecdna_distribution_kill_nplus() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
        assert!(ecdna.is_empty());
        assert_eq!(ecdna.get_nplus_cells(), 0);
    }

    #[test]
    #[should_panic]
    fn ecdna_distribution_proliferate_nplus_empty() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.proliferate_nplus();
    }

    #[test]
    fn ecdna_distribution_proliferate() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.proliferate_nplus();
        assert!(ecdna.get_nplus_cells() > 0);
    }

    #[test]
    #[should_panic]
    fn ecdna_distribution_dna_segregation_overflow() {
        let p = Parameters { init_copies: u16::MAX, ..Default::default() };
        let ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.dna_segregation(0usize);
    }

    #[test]
    fn ecdna_distribution_dna_segregation() {
        let init_copies = 2u16;
        let p = Parameters { init_copies, ..Default::default() };
        let ecdna = EcDNADistributionNPlus::new(&p);
        let (ecdna1, ecdna2, uneven) = ecdna.dna_segregation(0usize);
        assert_eq!(
            ecdna1 + ecdna2,
            2 * init_copies,
            "Sum does not match {}",
            ecdna1 + ecdna2
        );
        if ecdna1 * ecdna2 == 0 {
            assert!(uneven);
        } else {
            assert!(!uneven);
        }
    }

    #[test]
    #[should_panic]
    fn compute_mean_wrong_ntot() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        EcDNADistributionNPlus::new(&p).compute_mean(&0);
    }

    #[test]
    fn compute_mean_empty() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        assert!(
            (EcDNADistributionNPlus::new(&p).compute_mean(&1) - 0f32).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn compute_mean_empty_no_cells() {
        let p = Parameters { init_copies: 0u16, ..Default::default() };
        assert!(EcDNADistributionNPlus::new(&p).compute_mean(&0).is_nan());
    }

    #[test]
    #[should_panic]
    fn compute_mean_no_cells() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        assert!(EcDNADistributionNPlus::new(&p).compute_mean(&0).is_nan());
    }

    #[test]
    fn compute_mean_with_1s() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        assert!(
            (EcDNADistributionNPlus::new(&p).compute_mean(&1) - 1f32).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn compute_mean_with_default() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        assert!(
            (EcDNADistributionNPlus::new(&p).compute_mean(&1) - 1f32).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn compute_mean_with_1s_and_nminus() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        assert!(
            (EcDNADistributionNPlus::new(&p).compute_mean(&2) - 0.5f32).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn compute_mean_with_no_nminus() {
        let p = Parameters { init_copies: 2u16, ..Default::default() };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(4);
        assert!((distr.compute_mean(&2) - 3f32).abs() < f32::EPSILON)
    }

    #[test]
    fn compute_mean_with_nminus() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(2);
        assert!((distr.compute_mean(&3) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_mean_overflow() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(u16::MAX);
        assert!(distr.get_nplus_cells() == 2u64);
        let exp = ((1u64 + (u16::MAX as u64)) as f32) / 2f32;
        assert!((distr.compute_mean(&2) - exp).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_mean_overflow_nminus() {
        let p = Parameters { init_copies: 1u16, ..Default::default() };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(u16::MAX);
        assert!(distr.get_nplus_cells() == 2u64);
        let exp = ((1u64 + (u16::MAX as u64)) as f32) / 3f32;
        assert!((distr.compute_mean(&3) - exp).abs() < f32::EPSILON);
    }
}
