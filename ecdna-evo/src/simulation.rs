//! Implementation of the individual-based stochastic computer simulations of
//! the ecDNA population dynamics, assuming an exponential growing population of
//! tumour cells in a well-mixed environment. Simulation are carried out using
//! the Gillespie algorithm.
use crate::abc;
use crate::dynamics::{Dynamic, Update};
use crate::gillespie::{AdvanceRun, BirthDeathProcess, Event, GetRates, GillespieEvent};
use crate::timepoints::{Compute, Timepoints};
use crate::{GillespieTime, NbIndividuals, Patient, Rates};
use anyhow::{anyhow, Context};
use chrono::Utc;
use enum_dispatch::enum_dispatch;
use flate2::write::GzEncoder;
use flate2::Compression;
use indicatif::ParallelProgressIterator;
use rand::thread_rng;
use rand::Rng;
use rand_distr::{Binomial, Distribution};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Quantities of interest that will be simulated using the Gillespie algorithm
#[derive(Builder, Debug)]
pub struct Quantities {
    #[builder(setter(into), default)]
    timepoints: Option<Timepoints>,
    #[builder(setter(into), default)]
    dynamics: Option<Vec<Dynamic>>,
}

impl Quantities {
    pub fn timepoints_names(&self) -> anyhow::Result<String> {
        match &self.timepoints {
            None => Err(anyhow!("No timepoints found")),
            Some(timepoints) => timepoints.names(),
        }
    }
}

/// Perform multiple independent simulations of tumour growth in parallel using
/// `rayon` parallel iter API.
pub struct Simulation;

impl<'sim, 'run> Simulation {
    pub fn run(
        parameters: Parameters,
        rates: Rates,
        quantities: Option<Quantities>,
        patient: Option<Patient>,
    ) -> anyhow::Result<()> {
        //! Run in parallel `nb_runs` independent simulations of tumour growth.
        //! The arguments `patient` and `quantities` define whether to simulate
        //! tumour growth or ABC:
        //!
        //! 1. simulate tumour growth: `patient` must be `None` and `quantities`
        //! must be `Some`
        //!
        //! 2. ABC: `patient` must be `Some` and
        //! quantities must be `None`.
        // If we are in abc, `quantities` will be None (because created from the
        // data) and `data` will be Some
        let quantities = match quantities {
            None => {
                if let Some(ref patient) = patient {
                    patient.create_quantities(&parameters)
                } else {
                    panic!("Cannot run simulation without any patient nor quantities");
                }
            }
            Some(quantities) => quantities,
        };
        assert!(quantities.timepoints.is_some() || quantities.dynamics.is_some());

        println!(
            "{} Simulating with {} cores {} runs, each of them with {} cells and max iterations {}",
            Utc::now(),
            rayon::current_num_threads(),
            parameters.nb_runs,
            parameters.max_cells,
            parameters.max_iter
        );

        if parameters.verbosity > 1 {
            println!("{} {:#?}", Utc::now(), parameters);
        }

        if parameters.verbosity > 0 {
            println!(
                "{} Launching {} runs in parallel",
                Utc::now(),
                parameters.nb_runs
            );
        }

        // create path: relative path (used to creating the tarball) and
        // absolute path, for dynamics, timepoints and abc
        let (relapath_d, abspath_d) = Simulation::create_path(&parameters, &rates, "dynamics");
        let (relapath_t, abspath_t) = Simulation::create_path(&parameters, &rates, "timepoints");
        // add timepoints names abc/ecdna_mean/rates.gz
        let mut relative_abc = String::from("abc/");
        relative_abc.push_str(&quantities.timepoints_names().unwrap());
        let (relapath_abc, abspath_abc) =
            Simulation::create_path(&parameters, &rates, &relative_abc);

        let saved = (0..parameters.nb_runs)
            .into_par_iter()
            .progress_count(parameters.nb_runs as u64)
            .filter_map(|idx| {
                // Create one result for each run
                let mut dynamics = quantities.dynamics.clone();
                let mut timepoints = quantities.timepoints.clone();

                // Create new run and simulate tumour growth updating dynamics
                // for each iteration
                let mut r = Run::new(idx, &parameters, &rates);
                r.simulate(&parameters, &mut dynamics).unwrap();

                // Save results: 1. dynamics (updated for each iter) and
                // 2. timepoints (computed at the end of the run)
                //
                // 1. dynamics
                if let Some(dynamics) = dynamics {
                    for d in dynamics {
                        let file2path = abspath_d.join(d.get_name()).join(r.filename());
                        d.save(&file2path).unwrap();
                    }
                }
                // 2. timepoints
                // timepoints are computed now, as opposed to dynamics that are
                // updated during the simulation i.e. `run.simulate`
                let saved_succeeded = {
                    if let Some(timepoints) = &mut timepoints {
                        // save but handle the case of abc where we save only
                        // if runs are similar
                        match &patient {
                            Some(patient) => {
                                let results = abc::ABCRejection::run(&r, timepoints, patient);
                                results.save(&r, &abspath_abc)
                            }

                            None => {
                                for timepoint in timepoints.iter_mut() {
                                    timepoint.compute(&r);
                                    let file2path =
                                        abspath_t.join(timepoint.get_name()).join(r.filename());
                                    timepoint.save(&file2path).unwrap();
                                }
                                true
                            }
                        }
                    } else {
                        true
                    }
                };
                // do not consider dynamics which are always saved (i.e. never
                // in abc mode)
                Some(saved_succeeded)
            })
            .collect::<Vec<bool>>();

        println!("{} End simulating {} runs", Utc::now(), parameters.nb_runs);

        println!(
            "{} Start compressing {} runs",
            Utc::now(),
            parameters.nb_runs
        );

        // Save tarballs from abc even if there aren't any saved runs for abc
        if patient.is_some() {
            if parameters.verbosity > 0 {
                println!("{} Creating tarball for abc", Utc::now());
            }
            Simulation::compress_results("all_rates", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress all rates");

            Simulation::compress_results("metadata", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress metadata");

            Simulation::compress_results("values", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress metadata values");
        }

        // compress runs into tarball only if there is at least one saved run
        if saved.iter().any(|&saved_run| saved_run) {
            if let Some(dynamics) = &quantities.dynamics {
                for d in dynamics.iter() {
                    if parameters.verbosity > 0 {
                        println!("{} Creating tarball for dynamics", Utc::now());
                    }
                    Simulation::compress_results(d.get_name(), &relapath_d, parameters.verbosity)
                        .expect("Cannot compress dynamics");
                }
            }

            if let Some(timepoints) = &quantities.timepoints {
                // data is some means abc subcomand that is only the rates must
                // be saved and not the timepoints
                if patient.is_some() {
                    if parameters.verbosity > 0 {
                        println!("{} Creating tarball for abc", Utc::now());
                    }
                    Simulation::compress_results("rates", &relapath_abc, parameters.verbosity)
                        .expect("Cannot compress rates");
                } else {
                    for t in timepoints.iter() {
                        if parameters.verbosity > 0 {
                            println!("{} Creating tarball for timepoints", Utc::now());
                        }
                        Simulation::compress_results(
                            t.get_name(),
                            &relapath_t,
                            parameters.verbosity,
                        )
                        .expect("Cannot compress timepoints");
                    }
                }
            }
        }

        println!("{} End compressing {} runs", Utc::now(), parameters.nb_runs);

        println!(
            "{} {} runs over total of {} were saved",
            Utc::now(),
            saved.iter().filter(|&saved_run| *saved_run).count(),
            parameters.nb_runs
        );
        Ok(())
    }

    fn create_path(parameters: &Parameters, rates: &Rates, data_type: &str) -> (PathBuf, PathBuf) {
        //! Resturns the paths where to store the data. The hashmap has keys as
        //! the quantities stored, and values as the dest and source of where to
        //! compress the data.
        // create path where to store the results
        let relative_path = PathBuf::from("results")
            .join(Simulation::create_path_helper(parameters, rates))
            .join(data_type);
        let abspath = env::current_dir().unwrap().join(&relative_path);

        (relative_path, abspath)
    }

    fn create_path_helper(parameters: &Parameters, rates: &Rates) -> PathBuf {
        format!(
            "{}runs_{}cells_{}_{}1_{}2",
            parameters.nb_runs, parameters.max_cells, rates.fitness1, rates.death1, rates.death2
        )
        .into()
    }

    fn compress_results(kind: &str, basepath: &Path, verbosity: u8) -> anyhow::Result<()> {
        let dest = basepath.join(kind);
        let src = env::current_dir().unwrap().join(&dest);
        Simulation::compress_dir(&dest, &src, verbosity)
            .with_context(|| format!("Cannot compress {:#?} into {:#?}", &src, &dest))?;
        Ok(())
    }

    fn compress_dir(dest_path: &Path, src_path_dir: &Path, verbosity: u8) -> anyhow::Result<()> {
        //! Compress the directory where all runs are saved into tarball at the
        //! same level of the saved runs.
        let mut dest_path_archive = dest_path.to_owned();
        dest_path_archive.set_extension("tar.gz");

        // open stream, create encoder to compress and create tar builder to
        // create tarball
        let tar_gz = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest_path_archive)
            .with_context(|| format!("Error while opening the stream {:#?}", &dest_path_archive))?;
        let enc = GzEncoder::new(tar_gz, Compression::default());
        let mut tar = tar::Builder::new(enc);
        // append recursively all the runs into the archive that is first created in the
        // working and then moved to the location of the runs `src_path_dir`
        tar.append_dir_all(&dest_path_archive, src_path_dir)
            .with_context(|| {
                format!(
                    "Cannot append files to tar archive {:#?} from source {:#?} ",
                    &dest_path_archive, &src_path_dir
                )
            })
            .and_then(|()| {
                if verbosity > 1 {
                    println!(
                        "{} Gzip {:#?} into {:#?}",
                        Utc::now(),
                        src_path_dir,
                        &dest_path_archive,
                    );
                }
                fs::remove_dir_all(src_path_dir)
                    .with_context(|| format!("Cannot remove directory {:#?}", &src_path_dir))
            })?;
        Ok(())
    }
}

/// Simulation of an exponentially growing tumour, that is one realization of
/// the stochastic birth-death process.
pub struct Run {
    /// Index of the run
    idx: usize,
    /// Stochastic process simulating tumour growth
    process: BirthDeathProcess,
    /// State of the system at one particular iteration
    state: State,
}

#[enum_dispatch]
pub trait Name {
    fn get_name(&self) -> &String;
}

#[derive(Debug)]
struct State {
    nminus: NbIndividuals,
    ecdna_distr: EcDNADistribution,
    event: Event,
}

impl State {
    fn update(&mut self, event: Event) {
        //! Update the state of the system based on event sampled from Gillespie
        match event.kind {
            GillespieEvent::Advance(e) => {
                match e {
                    AdvanceRun::Proliferate1 => {
                        // Generate randomly a `NPlus` cell that will proliferate
                        // next and update the ecdna distribution as well.
                        // Match the cell that has been generated by the
                        // proliferation of the nplus cell
                        match self.ecdna_distr.distribution.proliferate_nplus() {
                            // Updates of nplus cells is done by ecdna_distr
                            Cell::NPlus => self.ecdna_distr.ntot += 1,
                            Cell::NMinus => {
                                self.nminus += 1;
                                self.ecdna_distr.ntot += 1;
                            }
                        }
                    }
                    AdvanceRun::Proliferate2 => {
                        self.nminus += 1;
                        self.ecdna_distr.ntot += 1;
                    }
                    // Updates of nplus cells is done by ecdna_distr
                    AdvanceRun::Die1 => {
                        self.ecdna_distr.distribution.kill_nplus();
                        self.ecdna_distr.ntot -= 1;
                    }
                    AdvanceRun::Die2 => {
                        self.nminus -= 1;
                        self.ecdna_distr.ntot -= 1;
                    }
                }
            }
            GillespieEvent::End(_) => {}
            GillespieEvent::Init => panic!("Cannot initialize run during simulation loop"),
        }
        self.event = Event::new(event.kind, event.time);
    }

    fn get_nplus_cells(&self) -> NbIndividuals {
        self.ecdna_distr.get_nplus_cells()
    }
}

impl Run {
    fn new(idx: usize, parameters: &Parameters, rates: &Rates) -> Self {
        //! Initialize the run with the `parameters`.
        let process: BirthDeathProcess = BirthDeathProcess::new(rates);

        let state = State {
            nminus: parameters.init_nminus,
            ecdna_distr: EcDNADistribution::new(parameters),
            event: Event {
                kind: GillespieEvent::Init,
                time: 0 as GillespieTime,
            },
        };

        Run {
            idx,
            process,
            state,
        }
    }

    pub fn get_nminus(&self) -> &NbIndividuals {
        //! Get the number of cells w/o any ecDNA copy for this iteration
        &self.state.nminus
    }

    pub fn get_nplus(&self) -> NbIndividuals {
        //! Get the number of cells w/ any ecDNA copy for this iteration
        self.state.get_nplus_cells()
    }

    pub fn get_gillespie_event(&self) -> &Event {
        //! Get the event associated to this iteration according to the
        //! Gillespie algorithm
        &self.state.event
    }

    pub fn get_rates(&self) -> [f32; 4] {
        //! Get the proliferation rates (fitness) and the death coefficients
        self.process.get_rates()
    }

    pub fn get_ecdna_distr(&self) -> &EcDNADistribution {
        //! Get the ecDNA distribution for the current iteration
        &self.state.ecdna_distr
    }

    pub fn filename(&self) -> PathBuf {
        //! File path for the current run (used to save data)
        PathBuf::from(format!("{}.csv", self.idx))
    }

    pub fn simulate(
        &mut self,
        parameters: &Parameters,
        dynamics: &mut Option<Vec<Dynamic>>,
    ) -> anyhow::Result<()> {
        //! Simulate the run (tumour growth) looping until the stop conditions
        //! are reached. If the `measurement` overrides the `update` method in
        //! `Measurable`, then it gets updated during the loop for each
        //! iteration based on the state of the system at the current
        //! iteration.
        let mut iter = 0_usize;
        let mut nplus = parameters.init_nplus;
        let mut nminus = parameters.init_nminus;

        loop {
            // Compute the next event using Gillespie algorithm based on the
            // stochastic process defined by `process`
            match self.process.gillespie(
                iter,
                nplus,
                nminus,
                parameters.max_iter,
                parameters.max_cells,
            ) {
                // Decide next step based on the output generated by Gillespie
                // `next_event`
                Event {
                    kind: GillespieEvent::End(_),
                    ..
                } => {
                    // StopIteration appears when there are no cells anymore
                    // (due to cell death), when the iteration has reached the
                    // maximal number of iterations nb_iter >= self.max_iter or
                    // maximal number of cells (self.max_cells), i.e. when the
                    // iteration has generated a tumor of self.max_cells size
                    break;
                }

                e @ Event {
                    kind: GillespieEvent::Advance(_),
                    ..
                } => {
                    self.update(e, dynamics);
                    // Retrieve the nplus and nminus cell present in the system
                    // after the update
                    nplus = self.get_nplus();
                    nminus = *self.get_nminus();
                    iter += 1;
                }
                Event {
                    kind: GillespieEvent::Init,
                    ..
                } => panic!("Cannot init the run in the middle of the simulate loop"),
            }
        }

        Ok(())
    }

    fn update(&mut self, next_event: Event, dynamics: &mut Option<Vec<Dynamic>>) {
        //! Update the run according to the `next_event` sampled from Gillespie
        //! algorithm. This updates also the measurements.
        match next_event {
            e @ Event {
                kind: GillespieEvent::End(_),
                ..
            } => {
                self.state.update(e);
                if let Some(dynamics) = dynamics {
                    for d in dynamics {
                        // updates all dynamics according to the new state
                        d.update(self);
                    }
                }
            }

            Event {
                kind: GillespieEvent::Init,
                ..
            } => {
                panic!("Init state can only be used to initliaze the simulation!")
            }

            e @ Event {
                kind: GillespieEvent::Advance(_),
                ..
            } => {
                self.state.update(e);
                if let Some(dynamics) = dynamics {
                    for d in dynamics {
                        // updates all dynamics according to the new state
                        d.update(self);
                    }
                }
            }
        }
    }
}

/// The distribution of ecDNA copies without considering the cells w/o any ecDNA
/// copy.
#[derive(Clone, Debug, Default)]
struct EcDNADistributionNPlus(Vec<DNACopy>);

impl From<Vec<DNACopy>> for EcDNADistributionNPlus {
    fn from(distr: Vec<DNACopy>) -> Self {
        assert!(
            !distr.iter().any(|&x| x < 1),
            "Cannot convert into EcDNADistribution with 0"
        );
        EcDNADistributionNPlus(distr)
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
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn compute_entropy(&self, ntot: NbIndividuals) -> f32 {
        -compute_counts(&self.0)
            .values()
            .map(|&count| {
                let prob: f32 = (count as f32) / (ntot as f32);
                prob * prob.log2()
            })
            .sum::<f32>()
    }

    fn compute_mean(&self, ntot: NbIndividuals) -> f32 {
        if self.is_empty() {
            if ntot == 0u64 {
                f32::NAN
            } else {
                0f32
            }
        } else if ntot == 0 {
            panic!("Cannot compute the mean of an when ntot is 0")
        } else if ntot < self.get_nplus_cells() {
            panic!("Found wrong value of ntot: should be equal or greater than the number of cells w/ ecDNA, found smaller")
        } else {
            let sum = self[..].iter().sum::<DNACopy>() as f32;
            let mean = sum / (ntot as f32);
            if mean.is_nan() {
                panic!("Compute mean NaN from ecdna distribution vector");
            }
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
        let (daughter1, daughter2) = self.dna_segregation(idx);

        // uneven_segregation happens if there is at least one zero and `daughter_1` and
        // `daughter2` cannot be both equal to 0
        let uneven_segregation = (daughter1 * daughter2) == 0;

        if uneven_segregation {
            self.0[idx] = daughter1 + daughter2; // n + 0 = n = 0 + n
            Cell::NMinus
        } else {
            self.0[idx] = daughter1; // update old cell ecDNA copies
            self.0.push(daughter2); // create new NPlus cell
            Cell::NPlus
        }
    }

    fn dna_segregation(&self, idx: usize) -> (DNACopy, DNACopy) {
        //! Simulate the proliferation of one `NPlus` cell.
        //! When a `idx` `NPlus` cell proliferates, we multiply the ec_dna
        //! present in that cell by the `FITNESS` of the `NPlus` cells
        //! compared to the `NMinus` cells (e.g. 1 for the neutral
        //! case). Then, we distribute the multiplied ec_dna material among two
        //! daughter cells according to a Binomial distribution with number of
        //! samples equal to the multiplied ec_dna material and the
        //! probability of 1/2.
        // Double the number of `NPlus` from the idx cell before proliferation because a
        // parental cell gives rise to 2 daughter cells. We also multiply by
        // `FITNESS` for models where `NPlus` cells have an advantage over
        // `NMinus` cells
        let available_dna: DNACopy = SPECIES as u64 * self.0[idx];
        assert_ne!(available_dna, 0);

        // draw the ec_dna that will be given to the daughter cells
        let bin = Binomial::new(available_dna, 0.5).unwrap();
        let d_1: DNACopy = bin.sample(&mut rand::thread_rng());
        let d_2: DNACopy = available_dna - d_1;
        assert_ne!(d_1 + d_2, 0);

        (d_1, d_2)
    }
}

/// The distribution of ecDNA copies not considering the cells w/o any ecDNA
/// copy.
#[derive(Clone, Debug, Default)]
pub struct EcDNADistribution {
    /// Distribution w/o `NMinus` cells
    distribution: EcDNADistributionNPlus,
    /// The total number of cells considering also `NMinus` cells
    ntot: NbIndividuals,
}

impl From<Vec<DNACopy>> for EcDNADistribution {
    fn from(distr: Vec<DNACopy>) -> Self {
        let ntot = distr.len() as NbIndividuals;
        let distribution = distr
            .into_iter()
            .filter(|&copy| copy > 0u64)
            .collect::<Vec<DNACopy>>();

        EcDNADistribution {
            distribution: EcDNADistributionNPlus::from(distribution),
            ntot,
        }
    }
}

impl IntoIterator for EcDNADistribution {
    type Item = DNACopy;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.distribution.0.into_iter()
    }
}

impl<Idx> std::ops::Index<Idx> for EcDNADistribution
where
    Idx: std::slice::SliceIndex<[DNACopy]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.distribution.0[index]
    }
}

impl EcDNADistribution {
    pub fn new(parameters: &Parameters) -> Self {
        let mut ecdna = Vec::with_capacity(parameters.max_cells as usize);
        ecdna.push(parameters.init_copies);
        EcDNADistribution {
            distribution: EcDNADistributionNPlus::from(ecdna),
            ntot: parameters.init_nplus + parameters.init_nminus,
        }
    }

    pub fn get_nplus_cells(&self) -> NbIndividuals {
        self.distribution.0.len() as u64
    }

    pub fn get_nminus_cells(&self) -> NbIndividuals {
        self.ntot - (self.distribution.0.len() as NbIndividuals)
    }

    pub fn create_vector_with_nminus_cells(&self) -> Vec<DNACopy> {
        let mut distr_with_nminus = self.distribution.0.clone();
        distr_with_nminus.append(&mut vec![0u64; self.get_nminus_cells() as usize]);
        distr_with_nminus
    }

    pub fn ks_distance(&self, ecdna_distribution: &EcDNADistribution) -> f32 {
        //! Compute the Kolmogorov-Smirnov distance defined as the maximal
        //! distance between the two ecdna distributions (two-sample ks test)
        //! considering cells w/o any ecDNA copy, `NMinus` cells.
        //!
        //! Memory intensive since we need to create two new vectors of ecDNA
        //! distributions adding `NMinus` cells.
        let distr_with_nminus = self.create_vector_with_nminus_cells();

        let other_with_nminus = ecdna_distribution.create_vector_with_nminus_cells();
        abc::ks_distance(&distr_with_nminus, &other_with_nminus)
    }

    pub fn compute_mean(&self) -> f32 {
        //! Compute the mean of the `ecdna_distribution` considering all cells
        //! that do not have any ecDNA copy (the `NMinus` cells), that are by
        //! definition not in the `ecdna_distribution` vec. Computationally
        //! intesive. When the ecDNA distribution is empty (as in the case of
        //! strong death coefficiens) it returns 0 if there are still some
        //! cells, else NAN
        self.distribution.compute_mean(self.ntot)
    }

    pub fn compute_entropy(&self) -> f32 {
        //! Compute the entropy of the ecDNA distribution without considering
        //! `NMinus` cells
        self.distribution.compute_entropy(self.ntot)
    }

    pub fn is_empty(&self) -> bool {
        self.distribution.0.is_empty()
    }
}

/// Number of ecDNA copies within a cell.
pub type DNACopy = u64;

const SPECIES: usize = 2;

/// Type of the individuals simulated.
#[derive(PartialEq, Debug)]
enum Cell {
    /// An individual with at least one copy of ecDNA
    NPlus,
    /// An individual without any copy of ecDNA
    NMinus,
}

/// Generate anyhow error if `val` is `f32::NAN`
pub fn find_nan(val: f32) -> anyhow::Result<f32> {
    if val.is_nan() {
        return Err(anyhow!("Found NaN value!"));
    }
    Ok(val)
}

#[derive(Debug, Clone, Copy)]
/// Parameters used to simulate tumour growth.
pub struct Parameters {
    /// The number of runs of the simulation
    pub nb_runs: usize,
    /// The maximal number of iterations after which one run is stopped, the
    /// same for each run
    pub max_cells: NbIndividuals,
    /// Max time is required when we use cell death in case there is too cell
    /// death TODO
    pub max_iter: usize,
    /// Initial copies of ecdna in `NPlus` cell for the run, the same for each
    /// run
    pub init_copies: DNACopy,
    /// Initial number of `NPlus` cells for the run, the same for each run
    pub init_nplus: NbIndividuals,
    /// Initial number of `NMinus` cells for the run, the same for each run
    pub init_nminus: NbIndividuals,
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
            init_copies: 1u64,
            init_nplus: 1u64,
            init_nminus: 0u64,
            verbosity: 0u8,
        }
    }
}

/// Trait to write the data to file
#[enum_dispatch]
pub trait ToFile {
    fn save(&self, path2file: &Path) -> anyhow::Result<()>;
}

pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
) -> anyhow::Result<()> {
    //! Write vector of float into new file with a precision of 4 decimals.
    //! Write NAN if the slice to write to file is empty.
    fs::create_dir_all(path.parent().unwrap()).expect("Cannot create dir");
    let f = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let mut buffer = BufWriter::new(f);
    if !data.is_empty() {
        if let Some(h) = header {
            writeln!(buffer, "{}", h)?;
        }
        write!(buffer, "{:.4}", data.first().unwrap())?;
        for ele in data.iter().skip(1) {
            write!(buffer, ",{:.4}", ele)?;
        }
    } else {
        write!(buffer, "{}", f32::NAN)?;
    }
    Ok(())
}

fn compute_counts(data: &[u64]) -> HashMap<&u64, u64> {
    //! Compute how many times the elements of data appear in data. From
    //! `<https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.counts>`
    let mut counts = HashMap::new();
    data.iter()
        .for_each(|item| *counts.entry(item).or_default() += 1);
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nan() {
        match find_nan(0f32) {
            Ok(v) => assert!((v - 0f32).abs() < f32::EPSILON),
            Err(_) => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn test_find_nan_panics() {
        match find_nan(f32::NAN) {
            Ok(v) => println!("{}", v),

            Err(_) => panic!(),
        }
    }

    #[test]
    fn test_ecdna_distribution_from_default() {
        let p = Parameters::default();
        let ecdna = EcDNADistribution::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1u64);
        assert!(!ecdna.is_empty());
    }

    #[test]
    fn test_ecdna_distribution_from() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 1u64,
            ..Default::default()
        };
        let ecdna = EcDNADistribution::new(&p);
        assert_eq!(ecdna.get_nplus_cells(), 1);
        assert!(!ecdna.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_compute_mean_with_default() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 1u64,
            ..Default::default()
        };
        assert!((EcDNADistribution::new(&p).compute_mean() - 0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_mean_with_1s() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 1u64,
            ..Default::default()
        };
        assert!(
            (EcDNADistribution::new(&p).distribution.compute_mean(1) - 1f32).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_compute_mean_with_1s_and_nminus() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 1u64,
            ..Default::default()
        };
        assert!(
            (EcDNADistribution::new(&p).distribution.compute_mean(2) - 0.5f32).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_compute_mean_with_no_nminus() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 2u64,
            ..Default::default()
        };
        let mut distr = EcDNADistribution::new(&p).distribution;
        distr.0.push(4);
        assert!((distr.compute_mean(2) - 3f32).abs() < f32::EPSILON)
    }

    #[test]
    fn test_compute_mean_with_nminus() {
        let p = Parameters {
            max_cells: 10u64,
            init_copies: 1u64,
            ..Default::default()
        };
        let mut distr = EcDNADistribution::new(&p).distribution;
        distr.0.push(2);
        assert!((distr.compute_mean(3) - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_small_simulation() {
        let p = Parameters::default();
        let rates = Rates::new(&[1f32], &[1f32], &[0f32], &[0f32]);
        let mut r = Run::new(0usize, &p, &rates);
        r.simulate(&p, &mut None).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_from_vector() {
        let original_data = vec![0u64, 2u64, 10u64];
        let _ = EcDNADistributionNPlus::from(original_data);
    }

    #[test]
    fn test_entropy_higher_than_1() {
        let original_data = vec![1u64, 2u64, 10u64];
        let distr = EcDNADistribution::from(original_data);
        assert!(distr.compute_entropy() > 0f32);
    }

    #[test]
    fn test_entropy_0() {
        let original_data = vec![1u64, 1u64, 1u64, 1u64];
        let distr = EcDNADistribution::from(original_data);
        assert!((distr.compute_entropy() - 0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_entropy_05() {
        let original_data = vec![1u64, 1u64, 2u64, 2u64];
        let distr = EcDNADistribution::from(original_data);
        assert!((distr.compute_entropy() - 1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_counts_empty() {
        let empty = Vec::new();
        assert!(compute_counts(&empty).is_empty())
    }

    #[test]
    fn test_compute_counts_1() {
        let ones = vec![1u64, 1u64];
        let result = HashMap::from([(&1u64, 2u64)]);
        assert_eq!(compute_counts(&ones), result);
    }

    #[test]
    fn test_compute_counts_1_2() {
        let data = vec![1u64, 2u64];
        let result = HashMap::from([(&1u64, 1u64), (&2u64, 1u64)]);
        assert_eq!(compute_counts(&data), result);
    }

    #[test]
    fn test_compute_counts() {
        let data = vec![1u64, 2u64, 10u64];
        let result = HashMap::from([(&1u64, 1u64), (&2u64, 1u64), (&10u64, 1u64)]);
        assert_eq!(compute_counts(&data), result);
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution_empty() {
        let my_vec = vec![];
        let dna = EcDNADistribution::from(my_vec);
        assert!(dna.is_empty());
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution_no_zeros() {
        let my_vec = vec![1, 1, 2];
        let dna = EcDNADistribution::from(my_vec);
        assert_eq!(dna.distribution.0, vec![1, 1, 2]);
        assert_eq!(dna.get_nplus_cells(), 3);
        assert_eq!(dna.get_nminus_cells(), 0);
        assert_eq!(dna.create_vector_with_nminus_cells(), vec![1, 1, 2]);
    }

    #[test]
    fn test_from_vec_for_ecdna_distribution_zeros() {
        let my_vec = vec![1, 1, 2, 0, 0];
        let dna = EcDNADistribution::from(my_vec);
        assert_eq!(dna.distribution.0, vec![1, 1, 2]);
        assert_eq!(dna.get_nplus_cells(), 3);
        assert_eq!(dna.get_nminus_cells(), 2);
        assert_eq!(dna.create_vector_with_nminus_cells(), vec![1, 1, 2, 0, 0]);
    }
}
