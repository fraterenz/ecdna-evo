use crate::abc::ABCRejection;
use crate::data::{Data, EcDNA, Entropy, Frequency, Mean};
use crate::dynamics::{Dynamics, Name, Update};
use crate::gillespie::{AdvanceRun, BirthDeathProcess, Event, GetRates, GillespieTime};
use crate::{NbIndividuals, Patient, Rates, ToFile};
use rand::thread_rng;
use rand::Rng;
use rand_distr::{Binomial, Distribution};
use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};

/// Number of ecDNA copies within a cell. We assume that a cell cannot have more
/// than 65535 copies (`u16` is 2^16 - 1 = 65535 copies).
pub type DNACopy = u16;

/// Simulation of an exponentially growing tumour, that is one realization of
/// the stochastic birth-death process.
pub struct Run {
    /// Index of the run
    pub idx: usize,
    /// Data of interest
    pub data: Data,
    /// Rates: the proliferation and death rates of the cells `NPlus` and
    /// `NMninus` cells respectively.
    rates: [f32; 4],
    /// Parameters to configure the run
    parameters: Parameters,
    /// Gillespie time at the end of the run
    gillespie_time: GillespieTime,
    /// Stopping condition
    stop: EndRun,
}

impl Run {
    pub fn new(idx: usize, parameters: Parameters, rates: &Rates) -> InitializedRun {
        //! Initialize the run with the `parameters` and the proliferation and
        //! death rates.

        InitializedRun {
            idx,
            process: BirthDeathProcess::new(rates),
            state: State {
                nminus: parameters.init_nminus,
                ecdna_distr: EcDNADistributionNPlus::new(&parameters),
                event: Event {
                    kind: AdvanceRun::Init,
                    time: parameters.init_time,
                },
            },
            parameters,
        }
    }

    pub fn save(
        self,
        abspath: &Path,
        dynamics: &Option<Dynamics>,
        patient: &Option<Patient>,
    ) -> SavedRun {
        //! Save the run the dynamics (updated for each iter) if present and the
        //! other quantities, computed at the end of the run such as the mean,
        //! frequency.
        let (idx, filename) = (self.idx, self.filename());
        let (abspath_d, abspath_abc) = (abspath.join("dynamics"), abspath.join("abc"));

        // 1. dynamics
        if let Some(ref dynamics) = dynamics {
            assert!(patient.is_none(), "Cannot have patient and dynamics");
            for d in dynamics.iter() {
                let file2path = abspath_d.join(d.get_name()).join(filename.clone());
                d.save(&file2path).unwrap();
            }
        }

        // save the ecDNA distribution, mean, entropy and the frequency for the
        // last iteration. Do not save ecdna with abc program
        self.data.save(abspath, &filename, patient.is_none());
        let saved_succeeded = {
            // save but handle the case of abc where we save only
            // if runs are similar
            if let Some(patient) = &patient {
                // ABC save the rates i.e. the proliferation and
                // death rates
                let (rates, filename) = (self.rates, self.filename());
                let results = ABCRejection::run(self, patient);
                results.save(&filename, &abspath_abc, &rates)
            } else {
                true
            }
        };

        SavedRun {
            saved: saved_succeeded,
            idx,
        }
    }

    pub fn restart(self) -> InitializedRun {
        //! Restart the simulation of the run setting the initial state to the
        //! last state of `self`.
        todo!()
    }

    fn filename(&self) -> PathBuf {
        //! File path for the current run (used to save data)
        PathBuf::from(format!("{}.csv", self.idx))
    }
}

pub struct InitializedRun {
    /// Index of the run
    pub idx: usize,
    /// Stochastic process simulating tumour growth
    process: BirthDeathProcess,
    /// State of the system at one particular iteration
    state: State,
    /// Parameters to configure the run
    parameters: Parameters,
}

impl InitializedRun {
    pub fn new(idx: usize, parameters: Parameters, rates: &Rates) -> Self {
        let process: BirthDeathProcess = BirthDeathProcess::new(rates);
        todo!()
    }

    pub fn get_nminus(&self) -> &NbIndividuals {
        &self.state.nminus
    }

    pub fn get_nplus(&self) -> NbIndividuals {
        self.state.get_nplus_cells()
    }

    pub fn mean_ecdna(&self) -> f32 {
        self.state
            .ecdna_distr
            .compute_mean(&(self.get_nplus() + self.get_nminus()))
    }

    pub fn variance_ecdna(&self, mean: &f32) -> f32 {
        if self.state.ecdna_distr.is_empty() {
            panic!("Cannot compute the variance of an empty ecDNA distribution")
        } else {
            let nb_nplus = self.get_nplus();
            let nb_nminus = self.get_nminus();

            self.state.ecdna_distr[..]
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
        &self.state.event
    }

    pub fn simulate(mut self, mut dynamics: Option<Dynamics>) -> Run {
        //! Simulate the run (tumour growth) looping until the stop conditions
        //! are reached. If the `measurement` overrides the `update` method in
        //! `Measurable`, then it gets updated during the loop for each
        //! iteration based on the state of the system at the current
        //! iteration.
        let mut iter = self.parameters.init_iter;
        let mut nplus = self.parameters.init_nplus as u64;
        let mut nminus = self.parameters.init_nminus;

        let (time, condition) = {
            loop {
                // Compute the next event using Gillespie algorithm based on the
                // stochastic process defined by `process`
                let event = self.process.gillespie(nplus, nminus);
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
                    break (self.state.event.time, EndRun::NoIndividualsLeft);
                };
                if iter >= self.parameters.max_iter - 1usize {
                    break (self.state.event.time, EndRun::MaxItersReached);
                };
                if nplus + nminus >= self.parameters.max_cells {
                    break (self.state.event.time, EndRun::MaxIndividualsReached);
                }
            }
        };
        let ntot = nminus + nplus;
        if let BirthDeathProcess::PureBirth(_) = &self.process {
            assert!(ntot > 0, "No cells found with PureBirth process")
        }

        let mean = Mean(self.state.ecdna_distr.compute_mean(&ntot));

        if nplus > 0 {
            assert!(mean.0 > 0f32)
        }

        let entropy = Entropy(self.state.ecdna_distr.compute_entropy(&ntot));
        let (ecdna, frequency) = (
            EcDNA(
                self // add nminus cells to ecDNA distribution
                    .state
                    .ecdna_with_nminus_cells(),
            ),
            Frequency((nplus as f32) / (ntot as f32)),
        );

        Run {
            idx: self.idx,
            data: Data::new(ecdna, mean, frequency, entropy),
            rates: self.process.get_rates(),
            parameters: self.parameters,
            gillespie_time: time,
            stop: condition,
        }
    }

    fn update(&mut self, next_event: Event, dynamics: &mut Option<Dynamics>) {
        //! Update the run according to the `next_event` sampled from Gillespie
        //! algorithm. This updates also the measurements.
        if let AdvanceRun::Init = next_event.kind {
            panic!("Init state can only be used to initialize simulation!")
        }
        self.state.update(next_event);
        if let Some(dynamics) = dynamics {
            for d in dynamics.iter_mut() {
                // updates all dynamics according to the new state
                d.update(self);
            }
        }
    }
}

pub struct SavedRun {
    /// Index of the run
    pub idx: usize,
    saved: bool,
}

impl SavedRun {
    pub fn is_saved(self) -> bool {
        self.saved
    }
}

#[derive(Debug)]
struct State {
    nminus: NbIndividuals,
    ecdna_distr: EcDNADistributionNPlus,
    event: Event,
}

impl State {
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

    pub fn ecdna_with_nminus_cells(self) -> EcDNADistribution {
        self.ecdna_distr
            .0
            .into_iter()
            .chain(std::iter::repeat(0u16).take(self.nminus as usize))
            .collect::<Vec<DNACopy>>()
            .into()
    }
}

/// The distribution of ecDNA copies without considering the cells w/o any ecDNA
/// copy.
#[derive(Clone, Debug, Default)]
pub struct EcDNADistributionNPlus(Vec<DNACopy>);

impl From<EcDNADistribution> for EcDNADistributionNPlus {
    fn from(distr: EcDNADistribution) -> Self {
        EcDNADistributionNPlus(
            distr
                .0
                .into_iter()
                .filter(|&copy| copy > 0u16)
                .collect::<Vec<DNACopy>>(),
        )
    }
}

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

    fn compute_entropy(&self, ntot: &NbIndividuals) -> f32 {
        //! Compute entropy with a `ntot` number of individuals
        -compute_counts(&self.0)
            .values()
            .map(|&count| {
                let prob: f32 = (count as f32) / (*ntot as f32);
                prob * prob.log2()
            })
            .sum::<f32>()
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
                .fold(0u64, |accum, &item| accum + (item as u64)) as f32;
            assert!(sum > 0.0);
            let mean = sum / (*ntot as f32);
            assert!(
                !(mean.is_nan()),
                "Compute mean NaN from ecdna distribution vector"
            );
            mean
        }
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
        let (daughter1, daughter2, uneven_segregation) = self.dna_segregation(idx);

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
        let d_1: DNACopy = bin.sample(&mut rand::thread_rng()).try_into().unwrap();
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
}

/// The distribution of ecDNA copies considering the cells w/o any ecDNA copy.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct EcDNADistribution(Vec<DNACopy>);

impl From<Vec<DNACopy>> for EcDNADistribution {
    fn from(distr: Vec<DNACopy>) -> Self {
        EcDNADistribution(distr)
    }
}

impl Deref for EcDNADistribution {
    type Target = Vec<DNACopy>;

    fn deref(&self) -> &Self::Target {
        &self.0
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

fn compute_counts(data: &[u16]) -> HashMap<&u16, u64> {
    //! Compute how many times the elements of data appear in data. From
    //! `<https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.counts>`
    let mut counts = HashMap::new();
    data.iter()
        .for_each(|item| *counts.entry(item).or_default() += 1);
    counts
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
    fn test_ecdna_distribution_is_empty_with_nminus_cells() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert!(ecdna.is_empty());
    }

    #[test]
    fn test_ecdna_distribution_getter_when_empty() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 0u64);
        assert!(ecdna.is_empty());
    }

    #[test]
    fn test_ecdna_distribution_from_default() {
        let p = Parameters::default();
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1u64);
        assert!(!ecdna.is_empty());
    }

    #[test]
    fn test_ecdna_distribution_from() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let ecdna = EcDNADistributionNPlus::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1);
        assert!(!ecdna.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_ecdna_distribution_kill_nplus_empty() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
    }

    #[test]
    fn test_ecdna_distribution_kill_nplus_one_cell() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
        assert!(ecdna.is_empty());
        assert_eq!(ecdna.get_nplus_cells(), 0);
    }

    #[test]
    fn test_ecdna_distribution_kill_nplus() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.kill_nplus();
        assert!(ecdna.is_empty());
        assert_eq!(ecdna.get_nplus_cells(), 0);
    }

    #[test]
    #[should_panic]
    fn test_ecdna_distribution_proliferate_nplus_empty() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.proliferate_nplus();
    }

    #[test]
    fn test_ecdna_distribution_proliferate() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.proliferate_nplus();
        assert!(ecdna.get_nplus_cells() > 0);
    }

    #[test]
    #[should_panic]
    fn test_ecdna_distribution_dna_segregation_overflow() {
        let p = Parameters {
            init_copies: u16::MAX,
            ..Default::default()
        };
        let ecdna = EcDNADistributionNPlus::new(&p);
        ecdna.dna_segregation(0usize);
    }

    #[test]
    fn test_ecdna_distribution_dna_segregation() {
        let init_copies = 2u16;
        let p = Parameters {
            init_copies,
            ..Default::default()
        };
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
    fn test_compute_mean_wrong_ntot() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        EcDNADistributionNPlus::new(&p).compute_mean(&0);
    }

    #[test]
    fn test_compute_mean_empty() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        assert!((EcDNADistributionNPlus::new(&p).compute_mean(&1) - 0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_empty_no_cells() {
        let p = Parameters {
            init_copies: 0u16,
            ..Default::default()
        };
        assert!(EcDNADistributionNPlus::new(&p).compute_mean(&0).is_nan());
    }

    #[test]
    #[should_panic]
    fn test_compute_mean_no_cells() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        assert_eq!(EcDNADistributionNPlus::new(&p).compute_mean(&0), f32::NAN);
    }

    #[test]
    fn test_compute_mean_with_1s() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        assert!((EcDNADistributionNPlus::new(&p).compute_mean(&1) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_with_default() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        assert!((EcDNADistributionNPlus::new(&p).compute_mean(&1) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_with_1s_and_nminus() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        assert!((EcDNADistributionNPlus::new(&p).compute_mean(&2) - 0.5f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_with_no_nminus() {
        let p = Parameters {
            init_copies: 2u16,
            ..Default::default()
        };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(4);
        assert!((distr.compute_mean(&2) - 3f32).abs() < f32::EPSILON)
    }

    #[test]
    fn test_compute_mean_with_nminus() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(2);
        assert!((distr.compute_mean(&3) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_overflow() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(u16::MAX);
        assert!(distr.get_nplus_cells() == 2u64);
        let exp = ((1u64 + (u16::MAX as u64)) as f32) / 2f32;
        assert!((distr.compute_mean(&2) - exp).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_overflow_nminus() {
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };
        let mut distr = EcDNADistributionNPlus::new(&p);
        distr.0.push(u16::MAX);
        assert!(distr.get_nplus_cells() == 2u64);
        let exp = ((1u64 + (u16::MAX as u64)) as f32) / 3f32;
        assert!((distr.compute_mean(&3) - exp).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_vector() {
        let original_data = vec![0u16, 2u16, 10u16];
        let ecdna = EcDNADistribution::from(original_data.clone());
        assert_eq!(ecdna.0, original_data);
    }

    #[test]
    fn test_entropy_higher_than_1() {
        let original_data = EcDNADistribution::from(vec![1u16, 2u16, 10u16]);
        let distr = EcDNADistributionNPlus::from(original_data);
        assert!(distr.compute_entropy(&3) > 0f32);
    }

    #[test]
    fn test_entropy_0() {
        let original_data = EcDNADistribution::from(vec![1u16, 1u16, 1u16, 1u16]);
        let distr = EcDNADistributionNPlus::from(original_data);
        assert!((distr.compute_entropy(&4) - 0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_entropy_05() {
        let original_data: EcDNADistribution = vec![1u16, 1u16, 2u16, 2u16].into();
        let distr = EcDNADistributionNPlus::from(original_data);
        assert!((distr.compute_entropy(&4) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution_empty() {
        let my_vec = vec![];
        let dna = EcDNADistribution::from(my_vec);
        assert!(dna.is_empty());
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution() {
        let my_vec: EcDNADistribution = vec![1, 1, 2].into();
        let dna = EcDNADistributionNPlus::from(my_vec);
        assert_eq!(dna.0, vec![1, 1, 2]);
        assert_eq!(dna.get_nplus_cells(), 3);
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution_zeros() {
        let my_vec: EcDNADistribution = vec![1, 1, 2, 0, 0].into();
        let dna = EcDNADistributionNPlus::from(my_vec);
        assert_eq!(dna.0, vec![1, 1, 2]);
        assert_eq!(dna.get_nplus_cells(), 3);
    }

    #[test]
    fn test_compute_counts_empty() {
        let empty = Vec::new();
        assert!(compute_counts(&empty).is_empty())
    }

    #[test]
    fn test_compute_counts_1() {
        let ones = vec![1u16, 1u16];
        let result = HashMap::from([(&1u16, 2u64)]);
        assert_eq!(compute_counts(&ones), result);
    }

    #[test]
    fn test_compute_counts_1_2() {
        let data = vec![1u16, 2u16];
        let result = HashMap::from([(&1u16, 1u64), (&2u16, 1u64)]);
        assert_eq!(compute_counts(&data), result);
    }

    #[test]
    fn test_compute_counts() {
        let data = vec![1u16, 2u16, 10u16];
        let result = HashMap::from([(&1u16, 1u64), (&2u16, 1u64), (&10u16, 1u64)]);
        assert_eq!(compute_counts(&data), result);
    }

    #[test]
    fn test_state_from_ecdnanplus_to_ecdna_no_nminus() {
        let expected = EcDNADistribution::from(vec![1, 0, 0]);
        let p = Parameters {
            init_copies: 1u16,
            ..Default::default()
        };

        let ecdna_distr = EcDNADistributionNPlus::new(&p);
        let state = State {
            nminus: 2u64,
            ecdna_distr,
            event: Event {
                kind: AdvanceRun::Proliferate1,
                time: 2f32,
            },
        };
        assert_eq!(state.ecdna_with_nminus_cells(), expected);
    }

    #[test]
    fn test_state_from_ecdnanplus_to_ecdna_nminus() {
        let expected = EcDNADistribution::from(vec![2, 0, 0]);
        let p = Parameters {
            init_copies: 2u16,
            init_nminus: 1u64,
            ..Default::default()
        };

        let ecdna_distr = EcDNADistributionNPlus::new(&p);
        let state = State {
            nminus: 2u64,
            ecdna_distr,
            event: Event {
                kind: AdvanceRun::Proliferate1,
                time: 2f32,
            },
        };
        assert_eq!(state.ecdna_with_nminus_cells(), expected);
    }
}
