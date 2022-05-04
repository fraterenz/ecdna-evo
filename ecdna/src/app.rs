use crate::{abc::ABCResults, clap_app::clap_app};
use anyhow::{ensure, Context};
use chrono::Utc;
use clap::ArgMatches;
use ecdna_data::{
    data::{EcDNADistribution, EcDNASummary, Entropy, Frequency, Mean},
    patient::{Patient, SequencingData},
};
use ecdna_dynamics::dynamics::{Dynamic, Dynamics, Name, Save};
use ecdna_dynamics::run::{
    CellCulture, ContinueGrowth, DNACopy, Ended, Growth, InitialState,
    PatientStudy, Run, Started, Update,
};
use ecdna_sim::{rate::Rates, NbIndividuals, Seed};
use enum_dispatch::enum_dispatch;
use flate2::{write::GzEncoder, Compression};
use indicatif::ParallelProgressIterator;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::HashSet,
    fs,
    ops::Deref,
    path::{Path, PathBuf},
};

pub fn build_config() -> Config {
    //! Build the app by parsing CL arguments with `clap` to build the structs
    //! required by `Simulation` to run the stochastic simulation.
    let matches = clap_app().get_matches();
    return match matches.subcommand() {
        Some(("abc", abc_matches)) => {
            Config::Bayesian(Bayesian::new(abc_matches))
        }
        Some(("simulate", dynamical_matches)) => {
            Config::Dynamical(Dynamical::new(dynamical_matches))
        }
        _ => unreachable!("Expect subcomands `simulate` or `abc`"),
    };
}

fn match_verbosity(matches: &ArgMatches) -> u8 {
    match matches.occurrences_of("verbosity") {
        0 => 0_u8,
        1 => 1_u8,
        _ => 2_u8,
    }
}

fn match_shared_args(matches: &ArgMatches) -> (u8, Option<PathBuf>, usize) {
    let verbosity = match_verbosity(matches);
    let distribution: Option<PathBuf> =
        matches.value_of("distribution").map(PathBuf::from);
    let runs: usize = matches.value_of_t("runs").unwrap_or_else(|e| e.exit());

    (verbosity, distribution, runs)
}

#[enum_dispatch]
pub trait Perform {
    fn run(&mut self) -> anyhow::Result<()>;
}

#[enum_dispatch]
pub trait Tarball {
    fn compress(self) -> anyhow::Result<()>;
}

#[enum_dispatch(Perform, Tarball)]
pub enum App {
    /// Bayesian inference
    Bayesian(BayesianApp),
    /// Simulate the dynamics of the ecDNA distribution over time
    Dynamical(DynamicalApp),
}

/// Infer the most probable coefficients from the data from one sequencing experiment.
#[derive(Debug)]
pub struct BayesianApp {
    pub patient: Patient,
    pub runs: usize,
    pub rho1: Vec<f32>,
    pub rho2: Vec<f32>,
    pub delta1: Vec<f32>,
    pub delta2: Vec<f32>,
    pub init_copies: Vec<DNACopy>,
    pub mean: Mean,
    pub frequency: Frequency,
    pub entropy: Entropy,
    pub growth: Growth,
    pub seed: u64,
    relative_path: PathBuf,
    absolute_path: PathBuf,
    pub verbosity: u8,
}

impl BayesianApp {
    pub fn new(config: Bayesian) -> anyhow::Result<Self> {
        let patient =
            Patient::load_from_file(&config.patient, config.verbosity)
                .with_context(|| "Cannot load patient")?;

        if config.verbosity > 0 {
            println!("Loaded patient {:#?}", patient);
        }

        let (ecdna, summary) = config
            .load()
            .with_context(|| "Cannot load initial state for Bayesian app")?;

        if config.verbosity > 0 {
            println!(
                "Initial ecDNA distribution {:#?} with {:#?} for the simulations",
                ecdna, summary
            );
        }

        let relative_path = PathBuf::from("results")
            .join(PathBuf::from(patient.name.clone()))
            .join("abc");
        let absolute_path =
            std::env::current_dir()?.join(relative_path.clone());

        let growth: Growth = match config.growth.as_str() {
            "patient" => PatientStudy::new().into(),
            "culture" => CellCulture::new().into(),
            _ => unreachable!("Possible vaules are patient or culture"),
        };

        let app = BayesianApp {
            patient,
            runs: config.runs,
            rho1: config.rho1,
            rho2: config.rho2,
            delta1: config.delta1,
            delta2: config.delta2,
            init_copies: config.init_copies,
            mean: summary.mean,
            frequency: summary.frequency,
            entropy: summary.entropy,
            growth,
            seed: config.seed,
            relative_path,
            absolute_path,
            verbosity: config.verbosity,
        };

        if config.verbosity > 1 {
            println!("{:#?}", app);
        }

        Ok(app)
    }

    fn save(
        &self,
        run: Run<Ended>,
        abspath: &Path,
        sequecing_sample: &SequencingData,
        sample_size: &Option<NbIndividuals>,
    ) -> anyhow::Result<Run<Started>> {
        let (results, run) =
            if let Some(true) = sequecing_sample.is_undersampled() {
                let sample_size = sample_size.expect(
                    "Sample is undersample but the sample size is not known",
                );
                // trade-off between computational time and accuracy
                let nb_samples = 10usize;
                let mut results = ABCResults::with_capacity(nb_samples);

                // create multiple subsamples of the same run and save the results
                // in the same file `path`. It's ok as long as cells is not too
                // big because deep copies of the ecDNA distribution for each
                // subsample
                for i in 0usize..nb_samples {
                    // returns new ecDNA distribution with cells NPlus cells (clone)
                    let simulated_sample =
                        run.clone().undersample_ecdna(&sample_size, i);
                    results.test(&simulated_sample, sequecing_sample);
                }
                (results, self.growth.restart_growth(run, &sample_size)?)
            } else {
                let mut results = ABCResults::with_capacity(1usize);
                results.test(&run, sequecing_sample);
                (results, run.into())
            };
        results.save(abspath)?;
        Ok(run)
    }
}

impl Perform for BayesianApp {
    fn run(&mut self) -> anyhow::Result<()> {
        println!(
            "{} Start ABC with {} runs for patient {}",
            Utc::now(),
            self.runs,
            self.patient.name
        );

        // we want to infer the parameters for this sample
        let sample2infer = self.patient.samples.first().unwrap();

        if self.verbosity > 0 {
            println!(
                "Is the sample to infer subsampled? {:#?}",
                sample2infer.is_undersampled()
            );
        }

        ensure!(self.patient.is_sorted());

        (0..self.runs)
            .into_par_iter()
            .progress_count(self.runs as u64)
            .for_each(|idx| {
                let seed_run = idx as u64 + self.seed;
                let mut rng = Pcg64Mcg::seed_from_u64(seed_run);

                let initial_state = match *self.init_copies {
                    [initial_copy] => {
                        InitialState::new_from_one_copy(initial_copy, 0usize)
                    }
                    [min, max] => InitialState::random(min, max, &mut rng),
                    _ => panic!("Max 2 values, found more than 2 or empty"),
                };

                // create rates from ranges: sample parameters from [min, max]
                let rates = Rates::new(
                    &self.rho1,
                    &self.rho2,
                    &self.delta1,
                    &self.delta2,
                );

                let mut run = Run::new(
                    idx,
                    0f32,
                    initial_state,
                    rates.clone(),
                    Seed::new(seed_run),
                    &mut rng,
                );

                // for all sequecing experiments
                for sample in self.patient.samples.iter() {
                    let run_ended = run.simulate(
                        &mut None,
                        &sample.tumour_size,
                        rates.estimate_max_iter(&sample.tumour_size),
                    );
                    run = self
                        .save(
                            run_ended,
                            &self.absolute_path,
                            sample,
                            &sample.sample_size(),
                        )
                        .with_context(|| format!("Cannot save run {}", idx))
                        .unwrap();
                }
            });

        println!("{} End using ABC with {} runs", Utc::now(), self.runs);

        // save the app's seed (each run has a seed = app_seed + run_idx)
        Seed::new(self.seed).save(&self.absolute_path.join("seed.csv"))?;

        Ok(())
    }
}

impl Tarball for BayesianApp {
    fn compress(self) -> anyhow::Result<()> {
        if self.verbosity > 0 {
            println!(
                "{} Saving inferences results in {:#?}",
                Utc::now(),
                self.absolute_path
            );
        }

        compress_dir(&self.relative_path, &self.absolute_path, self.verbosity)
            .with_context(|| {
                format!(
                    "Cannot compress {:#?} into {:#?}",
                    &self.absolute_path, &self.relative_path
                )
            })?;

        Ok(())
    }
}

/// Simulate exponential tumour growth with ecDNAs and track the dynamics of the
/// ecDNA distribution over time.
#[derive(Debug)]
pub struct DynamicalApp {
    pub patient_name: String,
    pub runs: usize,
    pub rates: Rates,
    pub dynamics: Dynamics,
    pub size: NbIndividuals,
    pub ecdna: EcDNADistribution,
    pub experiments: Experiments,
    pub growth: Growth,
    pub seed: u64,
    relative_path: PathBuf,
    absolute_path: PathBuf,
    pub verbosity: u8,
}

impl DynamicalApp {
    pub fn new(config: Dynamical) -> anyhow::Result<Self> {
        let (ecdna, _) = config.load().with_context(|| {
            "Cannot load initial state for DynamicalApp app"
        })?;

        let rates = Rates::new(
            &[config.rho1],
            &[config.rho2],
            &[config.delta1],
            &[config.delta2],
        );
        let max_iter = rates.estimate_max_iter(&config.cells);

        let mut dynamics = Vec::with_capacity(config.kind.len());
        for k in config.kind.iter() {
            dynamics.push(
                Dynamic::new(max_iter, ecdna.clone(), k).with_context(
                    || {
                        format!(
                            "Cannot create dynamical quantity {} for dynamics",
                            k
                        )
                    },
                )?,
            );
        }

        let relative_path = PathBuf::from("results")
            .join(PathBuf::from(config.patient_name.clone()));
        let absolute_path =
            std::env::current_dir()?.join(relative_path.clone());

        let experiments = Experiments::new(
            config.tumour_sizes.unwrap_or_else(|| vec![config.cells]),
            config.sample_sizes.unwrap_or_else(|| vec![config.cells]),
        )?;

        let growth: Growth = match config.growth.as_str() {
            "patient" => PatientStudy::new().into(),
            "culture" => CellCulture::new().into(),
            _ => unreachable!("Possible vaules are patient or culture"),
        };

        let app = DynamicalApp {
            patient_name: config.patient_name,
            runs: config.runs,
            rates,
            dynamics: dynamics.into(),
            size: config.cells,
            ecdna,
            growth,
            seed: config.seed,
            relative_path,
            absolute_path,
            experiments,
            verbosity: config.verbosity,
        };

        if config.verbosity > 1 {
            println!("{:#?}", app);
        }
        Ok(app)
    }

    fn save(
        &self,
        run: Run<Ended>,
        abspath: &Path,
        mut dynamics: Dynamics,
        tumour_size: &NbIndividuals,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<(Run<Started>, Dynamics)> {
        ensure!(tumour_size >= sample_size);

        let filename = run.filename();
        let abspath_with_undersampling = abspath.to_owned().join(format!(
            "{}sample{}cells",
            sample_size,
            run.nb_cells()
        ));

        // undersample dynamics and restart the growth for the next timepoint
        let (dynamics, run) = if tumour_size > sample_size {
            let idx = run.idx;
            run.clone()
                .undersample_ecdna(sample_size, idx)
                .save_ecdna(&abspath_with_undersampling);
            let run = self.growth.restart_growth(run, sample_size)?;
            // this will result in dynamics having different values for the
            // same gillesipe time: is it a problem?
            for d in dynamics.iter_mut() {
                d.update(&run);
            }
            (dynamics, run)
        } else {
            run.save_ecdna(&abspath_with_undersampling);
            (dynamics, run.into())
        };

        for d in dynamics.iter() {
            let mut file2path = abspath_with_undersampling
                .join(d.get_name())
                .join(filename.clone());
            file2path.set_extension("csv");
            d.save(&file2path).unwrap();
        }

        Ok((run, dynamics))
    }
}

impl Perform for DynamicalApp {
    fn run(&mut self) -> anyhow::Result<()> {
        if self.verbosity > 0 {
            println!("{} Starting simulating dynamics", Utc::now());
        }

        (0..self.runs)
            .into_par_iter()
            .progress_count(self.runs as u64)
            .for_each(|idx| {
                let initial_state = InitialState::new(
                    self.ecdna.clone(),
                    self.ecdna.nb_cells() as usize,
                );
                let seed_run = idx as u64 + self.seed;
                let mut rng = Pcg64Mcg::seed_from_u64(seed_run);

                let mut run = Run::new(
                    idx,
                    0f32,
                    initial_state,
                    self.rates.clone(),
                    Seed::new(seed_run),
                    &mut rng,
                );

                let mut dynamics = self.dynamics.clone();

                // for all sequecing experiemnts
                for experiment in self.experiments.iter() {
                    let (tumour_cells, sample_cells) =
                        (experiment.0, experiment.1);

                    // simulate and update both the run and the dynamics
                    let run_ended = run.simulate(
                        &mut Some(&mut dynamics),
                        &tumour_cells,
                        self.rates.estimate_max_iter(&tumour_cells),
                    );
                    (run, dynamics) = self
                        .save(
                            run_ended,
                            &self.absolute_path,
                            dynamics,
                            &tumour_cells,
                            &sample_cells,
                        )
                        .with_context(|| format!("Cannot save run {}", idx))
                        .unwrap();
                }
            });

        println!(
            "{} End simulating dynamics with {} runs",
            Utc::now(),
            &self.runs
        );

        // save the app's seed (each run has a seed = app_seed + run_idx)
        Seed::new(self.seed).save(&self.absolute_path.join("seed.csv"))?;

        Ok(())
    }
}

impl Tarball for DynamicalApp {
    fn compress(self) -> anyhow::Result<()> {
        for (tumour_size, sample_size) in self.experiments.iter() {
            let src_sample = self
                .absolute_path
                .clone()
                .join(format!("{}sample{}cells", sample_size, tumour_size));
            let dest_sample = self
                .relative_path
                .clone()
                .join(format!("{}sample{}cells", sample_size, tumour_size));

            for d in self.dynamics.iter() {
                if self.verbosity > 0 {
                    println!(
                        "{} Creating tarball for dynamics {}",
                        Utc::now(),
                        d.get_name()
                    );
                }
                let src = src_sample.clone().join(d.get_name());
                let dest = dest_sample.clone().join(d.get_name());
                if self.verbosity > 0 {
                    println!("src {:#?} dest {:#?} ", src, dest);
                }
                compress_dir(&dest, &src, self.verbosity)
                    .expect("Cannot compress dynamics");
            }

            // specific case with moments dynamics (saving the mean)
            for path in fs::read_dir(&src_sample)? {
                match path {
                    Ok(p) => {
                        if p.path().ends_with("mean_dynamics") {
                            let src = src_sample.clone().join("mean_dynamics");
                            let dest =
                                dest_sample.clone().join("mean_dynamics");
                            if self.verbosity > 0 {
                                println!("src {:#?} dest {:#?} ", src, dest);
                            }
                            compress_dir(&dest, &src, self.verbosity)
                                .expect("Cannot compress mean_dynamics");
                        }
                    }
                    _ => {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

/// Load the initial state, i.e. the ecDNA distribution from a file. If the file
/// is not specified, create an initial state with one single-cell with one ecDNA copy.
#[enum_dispatch]
trait InitialStateLoader {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)>;
}

#[enum_dispatch(InitialStateLoader)]
pub enum Config {
    Bayesian(Bayesian),
    // Longitudinal(Longitudinal),
    // Subsampled(Subsampled),
    Dynamical(Dynamical),
}

pub struct Bayesian {
    pub patient: PathBuf,
    pub runs: usize,
    pub distribution: Option<PathBuf>,
    pub rho1: Vec<f32>,
    pub rho2: Vec<f32>,
    pub delta1: Vec<f32>,
    pub delta2: Vec<f32>,
    pub init_copies: Vec<DNACopy>,
    pub tumour_sizes: Option<Vec<NbIndividuals>>,
    pub sample_sizes: Option<Vec<NbIndividuals>>,
    pub seed: u64,
    pub growth: String,
    pub verbosity: u8,
}

impl Bayesian {
    pub fn new(matches: &ArgMatches) -> Bayesian {
        let (verbosity, distribution, runs) = match_shared_args(matches);

        let patient: PathBuf =
            matches.value_of_t("patient").unwrap_or_else(|e| e.exit());

        // Rates of the two-type stochastic birth-death process
        // Proliferation rate of the cells w/ ecDNA
        let rho1: Vec<f32> =
            matches.values_of_t("rho1").unwrap_or_else(|e| e.exit());
        // Proliferation rate of the cells w/o ecDNA
        let rho2: Vec<f32> =
            matches.values_of_t("rho2").unwrap_or_else(|e| e.exit());

        // Death rate of cells w/ ecDNA
        let delta1: Vec<f32> =
            matches.values_of_t("delta1").unwrap_or_else(|e| e.exit());
        let delta2: Vec<f32> =
            matches.values_of_t("delta2").unwrap_or_else(|e| e.exit());

        let init_copies: Vec<DNACopy> =
            matches.values_of_t("init_copies").unwrap_or_else(|e| e.exit());

        // population and sample sizes
        let tumour_sizes = match matches.values_of_t("tumour_size") {
            Ok(sizes) => Some(sizes),
            _ => None,
        };
        let sample_sizes = match matches.values_of_t("sample_sizes") {
            Ok(sizes) => Some(sizes),
            _ => None,
        };

        let growth = if matches.is_present("culture") {
            String::from("culture")
        } else {
            String::from("patient")
        };

        let seed: u64 = matches.value_of_t("seed").unwrap_or(26u64);

        Bayesian {
            patient,
            runs,
            distribution,
            rho1,
            rho2,
            delta1,
            delta2,
            init_copies,
            seed,
            tumour_sizes,
            sample_sizes,
            growth,
            verbosity,
        }
    }
}

impl InitialStateLoader for Bayesian {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
        let ecdna = if let Some(distribution) = &self.distribution {
            EcDNADistribution::load_from_file(distribution).with_context(
                || "Cannot load the initial ecDNA distribution to init app",
            )?
        } else {
            EcDNADistribution::from(vec![1])
        };
        let summary = ecdna.summarize();

        Ok((ecdna, summary))
    }
}

// pub struct Longitudinal(pub Bayesian);
//
// impl Longitudinal {
//     pub fn new(matches: &ArgMatches) -> Longitudinal {
//         Longitudinal(Bayesian::new(matches))
//     }
// }
//
// impl InitialStateLoader for Longitudinal {
//     fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
//         self.0.load()
//     }
// }
//
// pub struct Subsampled(pub Bayesian);
//
// impl Subsampled {
//     pub fn new(matches: &ArgMatches) -> Subsampled {
//         Subsampled(Bayesian::new(matches))
//     }
// }
//
// impl InitialStateLoader for Subsampled {
//     fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
//         self.0.load()
//     }
// }

pub struct Dynamical {
    pub patient_name: String,
    pub kind: Vec<String>,
    pub runs: usize,
    pub cells: NbIndividuals,
    pub distribution: Option<PathBuf>,
    pub rho1: f32,
    pub rho2: f32,
    pub delta1: f32,
    pub delta2: f32,
    pub tumour_sizes: Option<Vec<NbIndividuals>>,
    pub sample_sizes: Option<Vec<NbIndividuals>>,
    pub seed: u64,
    pub growth: String,
    pub verbosity: u8,
}

impl Dynamical {
    pub fn new(matches: &ArgMatches) -> Dynamical {
        let (verbosity, distribution, runs) = match_shared_args(matches);

        let cells: NbIndividuals =
            matches.value_of_t("cells").unwrap_or_else(|e| e.exit());
        let patient_name: String =
            matches.value_of_t("patient").unwrap_or_else(|e| e.exit());

        // Quantities of interest that changes for each iteration
        let mut dynamics: HashSet<String> =
            match matches.values_of_t("dynamics") {
                Ok(dynamics) => dynamics.into_iter().collect(),
                _ => {
                    let mut dynamics = HashSet::with_capacity(4);
                    dynamics.insert("nplus".to_owned());
                    dynamics.insert("nminus".to_owned());
                    dynamics.insert("moments".to_owned());
                    dynamics.insert("time".to_owned());
                    dynamics
                }
            };

        // mean and variance or just mean
        if dynamics.contains("moments") {
            dynamics.remove("mean");
        }

        // Rates of the two-type stochastic birth-death process
        // Proliferation rate of the cells w/ ecDNA
        let rho1: f32 =
            matches.value_of_t("rho1").unwrap_or_else(|e| e.exit());
        // Proliferation rate of the cells w/o ecDNA
        let rho2: f32 =
            matches.value_of_t("rho2").unwrap_or_else(|e| e.exit());

        // Death rate of cells w/ ecDNA
        let delta1: f32 =
            matches.value_of_t("delta1").unwrap_or_else(|e| e.exit());
        let delta2: f32 =
            matches.value_of_t("delta2").unwrap_or_else(|e| e.exit());

        // population and sample sizes
        let tumour_sizes = match matches.values_of_t("sizes") {
            Ok(sizes) => {
                assert!(sizes.last().unwrap() == &cells,
                    "The last value of `sizes` must have the same number of cells as provided by `cells` argument");
                Some(sizes)
            }
            _ => None,
        };
        let sample_sizes = match matches.values_of_t("samples") {
            Ok(sizes) => {
                assert!(tumour_sizes
                    .as_ref()
                    .map_or(true, |v| v.len() == sizes.len()),
                    "Must supply the same number of values for arguments `sizes` and `samples`");
                Some(sizes)
            }
            _ => None,
        };

        let growth = if matches.is_present("culture") {
            String::from("culture")
        } else {
            String::from("patient")
        };

        let seed: u64 = matches.value_of_t("seed").unwrap_or(26u64);

        if verbosity > 1 {
            println!("dynamics: {:#?}", dynamics);
        }

        Dynamical {
            kind: dynamics.into_iter().collect(),
            patient_name,
            runs,
            cells,
            distribution,
            rho1,
            rho2,
            delta1,
            delta2,
            growth,
            seed,
            tumour_sizes,
            sample_sizes,
            verbosity,
        }
    }
}

impl InitialStateLoader for Dynamical {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
        let ecdna = if let Some(distribution) = &self.distribution {
            EcDNADistribution::load_from_file(distribution).with_context(
                || "Cannot load the initial ecDNA distribution to init app",
            )?
        } else {
            EcDNADistribution::from(vec![1])
        };

        let summary = ecdna.summarize();

        Ok((ecdna, summary))
    }
}

/// Collection of population and sample sizes specifying the longitudinal timepoints.
/// The population size indicates the number of cells in the tumour population
/// when the timepoint has been collected, and the sample size the number of cells
/// sequenced from the subsampling of the whole tumour.
#[derive(Debug)]
pub struct Experiments(Vec<(NbIndividuals, NbIndividuals)>);

impl Experiments {
    pub fn new(
        tumour_sizes: Vec<NbIndividuals>,
        sample_sizes: Vec<NbIndividuals>,
    ) -> anyhow::Result<Self> {
        ensure!(
            tumour_sizes.len() == sample_sizes.len(),
            "Found incosistent number of tumour and sample sizes"
        );
        ensure!(!tumour_sizes.is_empty());

        let experiments: Vec<(NbIndividuals, NbIndividuals)> =
            tumour_sizes.into_iter().zip(sample_sizes.into_iter()).collect();

        Ok(Experiments(experiments))
    }
}

impl Deref for Experiments {
    type Target = Vec<(NbIndividuals, NbIndividuals)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn compress_dir(
    dest_path: &Path,
    src_path_dir: &Path,
    verbosity: u8,
) -> anyhow::Result<()> {
    //! Compress the directory into tarball. `dest_path` is a relative path and `src_path_dir`
    //! is absolute path.
    let mut dest_path_archive = dest_path.to_owned();
    dest_path_archive.set_extension("tar.gz");

    ensure!(dest_path_archive.is_relative());
    ensure!(src_path_dir.is_absolute());

    // open stream, create encoder to compress and create tar builder to
    // create tarball
    let tar_gz = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest_path_archive)
        .with_context(|| {
            format!("Error while opening the stream {:#?}", &dest_path_archive)
        })?;

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

            fs::remove_dir_all(src_path_dir).with_context(|| {
                format!("Cannot remove directory {:#?}", &src_path_dir)
            })
        })?;

    Ok(())
}
