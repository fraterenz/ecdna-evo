use crate::{abc::ABCResults, clap_app::clap_app};
use anyhow::{bail, ensure, Context};
use chrono::Utc;
use clap::ArgMatches;
use ecdna_data::{
    data::{EcDNADistribution, EcDNASummary, Entropy, Frequency, Mean},
    patient::{Patient, SequencingData, SequencingDataBuilder},
};
use ecdna_dynamics::{
    dynamics::{Clear, Dynamic, Dynamics, Name, Save},
    segregation::Segregation,
};
use ecdna_dynamics::{
    run::{
        CellCulture, ContinueGrowth, DNACopy, Ended, Growth, InitialState,
        PatientStudy, Run, Started, Update,
    },
    segregation::{BinomialNoUneven, BinomialSegregation},
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
    fmt, fs,
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
        Some(("preprocess", preprocess_matches)) => {
            Config::Preprocess(Preprocess::new(preprocess_matches))
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
    fn run(self) -> anyhow::Result<()>;
}

#[enum_dispatch(Perform)]
pub enum App {
    /// Bayesian inference
    Bayesian(BayesianApp),
    /// Simulate the dynamics of the ecDNA distribution over time
    Dynamical(DynamicalApp),
    /// Preprocess the data for the abc inference
    Preprocess(PreprocessApp),
}

/// Prepare the data for the abc inference
pub struct PreprocessApp {
    pub patient_name: String,
    pub sample_name: String,
    pub size: NbIndividuals,
    pub ecdna: Option<EcDNADistribution>,
    pub mean: Option<Mean>,
    pub frequency: Option<Frequency>,
    pub entropy: Option<Entropy>,
    pub savedir: PathBuf,
    pub verbosity: u8,
}

impl PreprocessApp {
    pub fn new(config: Preprocess) -> anyhow::Result<PreprocessApp> {
        //! Parse the CL arguments with `clap` to preprocess the data and create the
        //! input required for the bayesian inferences.
        // load the ecdna distribution
        let ecdna = if config.distribution.is_some() {
            Some(
                config
                    .load()
                    .with_context(|| "Cannot load the ecDNA distribution")?
                    .0,
            )
        } else {
            None
        };

        let patient_name = config.patient_name;
        let sample_name = config.sample_name;
        let size = config.size;
        let verbosity = config.verbosity;
        let savedir = config.savedir;

        let mean = config.mean.map(Mean);
        let frequency = config.frequency.map(Frequency);
        let entropy = config.entropy.map(Entropy);

        Ok(PreprocessApp {
            patient_name,
            sample_name,
            size,
            ecdna,
            mean,
            frequency,
            entropy,
            savedir,
            verbosity,
        })
    }

    fn new_sample(self) -> SequencingData {
        let mut builder = SequencingDataBuilder::default();
        builder.tumour_size(self.size).name(self.sample_name);
        if let Some(distribution) = self.ecdna {
            let summary = distribution.summarize();
            builder
                .ecdna(distribution)
                .mean(summary.mean)
                .frequency(summary.frequency)
                .entropy(summary.entropy);
        } else {
            if let Some(mean) = self.mean {
                builder.mean(mean);
            }
            if let Some(freq) = self.frequency {
                builder.frequency(freq);
            }

            if let Some(entropy) = self.entropy {
                builder.entropy(entropy);
            }
        }

        builder.build().unwrap()
    }
}

impl Perform for PreprocessApp {
    fn run(self) -> anyhow::Result<()> {
        println!(
            "Adding sample {} to patient {}",
            self.sample_name, self.patient_name
        );

        let mut patient = Patient::new(&self.patient_name, self.verbosity);
        let path2patient =
            self.savedir.join(format!("preprocessed/{}.json", patient.name));

        // if patient exists load it, else create new file
        if path2patient.exists() {
            if self.verbosity > 0 {
                println!(
                    "Adding sample {} to an existing patient {}",
                    self.sample_name, self.patient_name
                )
            }
            if self.verbosity > 1 {
                println!(
                    "Loading patient {} from {:#?}",
                    self.patient_name, path2patient
                );
            }
            patient
                .load(&path2patient)
                .with_context(|| {
                    format!("Cannot load patient from {:#?}", path2patient)
                })
                .unwrap();
        } else {
            if self.verbosity > 0 {
                println!("Saving new patient");
            }
            fs::create_dir_all(path2patient.parent().unwrap()).unwrap();
        }

        let new_sample = self.new_sample();
        let name = patient.name.clone();

        if let Err(e) = patient.add_sample(new_sample) {
            bail!("Error, cannot add sample to patient due to:\n{}", e)
        }

        if let Err(e) = patient.save(&path2patient) {
            bail!(
                "Cannot save patient {} in {:#?} due to:\n{}",
                name,
                path2patient,
                e
            )
        } else {
            println!(
                "The new/updated data for patient {} is in {:#?}\n\
                Use this file as input for the bayesian inferences.",
                name, path2patient
            );
        }

        Ok(())
    }
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
    pub segregation: Segregation,
    pub growth: Growth,
    pub seed: u64,
    relative_path: PathBuf,
    absolute_path: PathBuf,
    pub verbosity: u8,
}

impl BayesianApp {
    pub fn new(config: Bayesian) -> anyhow::Result<Self> {
        if config.verbosity > 1 {
            println!("{:#?}", config);
        }

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

        let relative_path = PathBuf::from(patient.name.clone()).join("abc");
        let absolute_path = config.savedir.join(relative_path.clone());

        let growth: Growth = match config.growth.as_str() {
            "patient" => PatientStudy::new().into(),
            "culture" => CellCulture::new().into(),
            _ => unreachable!("Possible vaules are patient or culture"),
        };

        let segregation = match config.segregation.as_ref() {
            "deterministic" => Segregation::Deterministic,
            "binomial" => Segregation::Random(BinomialSegregation.into()),
            "nouneven" => Segregation::Random(BinomialNoUneven(BinomialSegregation).into()),
            _ => unreachable!("Possible values are `deterministic`, `binomial` or `nouneven`")
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
            segregation,
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
        timepoint: usize,
    ) -> anyhow::Result<Run<Started>> {
        let run = if let Some(true) = sequecing_sample.is_undersampled() {
            let sample_size = sample_size.expect(
                "Sample is undersample but the sample size is not known",
            );
            // trade-off between computational time and accuracy
            let nb_samples = 10usize;
            let mut results = ABCResults::with_capacity(nb_samples);

            // create multiple subsamples of the same run and save the results
            // in the same file `path`. It's ok as long as sample_size is not
            // too big because deep copies of the ecDNA distribution for each
            // subsample.
            //
            // Run a first test outside the loop such that that subsample can
            // be reused later on to restart the tumour growth (see below)
            let simulated_sample =
                run.clone().undersample_ecdna(&sample_size, 0usize);
            results.test(&simulated_sample, sequecing_sample, timepoint);
            for i in 1usize..nb_samples {
                // returns new ecDNA distribution with cells NPlus cells (clone)
                let simulated_sample =
                    run.clone().undersample_ecdna(&sample_size, i);
                results.test(&simulated_sample, sequecing_sample, timepoint);
            }
            results.save(abspath)?;
            // restart tumour growth from the first sample
            self.growth.restart_growth(simulated_sample, &sample_size)?
        } else {
            let mut results = ABCResults::with_capacity(1usize);
            results.test(&run, sequecing_sample, timepoint);
            results.save(abspath)?;
            run.into()
        };
        Ok(run)
    }

    pub fn compress(self) -> anyhow::Result<()> {
        println!(
            "{} Saving inferences results in {:#?}",
            Utc::now(),
            self.absolute_path
        );

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

impl Perform for BayesianApp {
    fn run(self) -> anyhow::Result<()> {
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
                    self.patient.samples[0].tumour_size, //assume same size
                    &mut rng,
                );

                // for all sequecing experiments
                for (timepoint, sample) in
                    self.patient.samples.iter().enumerate()
                {
                    let run_ended = run.simulate(
                        &self.segregation,
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
                            timepoint,
                        )
                        .with_context(|| format!("Cannot save run {}", idx))
                        .unwrap();
                }
            });

        println!("{} End using ABC with {} runs", Utc::now(), self.runs);

        self.compress()?;

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
    /// Final size of the tumour, i.e. number of cells
    pub size: NbIndividuals,
    pub ecdna: EcDNADistribution,
    pub experiments: Experiments,
    pub growth: Growth,
    pub segregation: Segregation,
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

        let relative_path = PathBuf::from(config.patient_name.clone());
        let absolute_path = config.savedir.join(relative_path.clone());

        let experiments = Experiments::new(
            config.tumour_sizes.unwrap_or_else(|| vec![config.cells]),
            config.sample_sizes.unwrap_or_else(|| vec![config.cells]),
        )?;

        let growth: Growth = match config.growth.as_str() {
            "patient" => PatientStudy::new().into(),
            "culture" => CellCulture::new().into(),
            _ => unreachable!("Possible vaules are patient or culture"),
        };

        let segregation = match config.segregation.as_ref() {
            "deterministic" => Segregation::Deterministic,
            "binomial" => Segregation::Random(BinomialSegregation.into()),
            "nouneven" => Segregation::Random(BinomialNoUneven(BinomialSegregation).into()),
            _ => unreachable!("Possible values are `deterministic`, `binomial` or `nouneven`")
            };

        let app = DynamicalApp {
            patient_name: config.patient_name,
            runs: config.runs,
            rates,
            dynamics: dynamics.into(),
            size: config.cells,
            ecdna,
            growth,
            segregation,
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
        timepoint: usize,
    ) -> anyhow::Result<(Run<Started>, Dynamics)> {
        ensure!(tumour_size >= sample_size);

        let filename = run.filename();
        let abspath_with_undersampling = abspath.to_owned().join(format!(
            "{}",
            Experiment::new(run.population_size, *sample_size)
        ));
        let is_undersampled = tumour_size > sample_size;

        let (mut dynamics, run) =
            if is_undersampled {
                let idx = run.idx;
                run.save_ecdna_statistics(&abspath_with_undersampling.join(
                    format!("timepoint_{}_before_subsampling", timepoint),
                ));
                let run = run.undersample_ecdna(sample_size, idx);
                run.save_ecdna_statistics(
                    &abspath_with_undersampling.join(format!("{}", timepoint)),
                );
                let run = self.growth.restart_growth(run, sample_size)?;
                if let Growth::CellCulture(_) = self.growth {
                    // remove all dynamics except the last entry
                    for d in dynamics.iter_mut() {
                        d.update(&run);
                    }
                }
                (dynamics, run)
            } else {
                run.save_ecdna_statistics(
                    &abspath_with_undersampling.join(format!("{}", timepoint)),
                );
                (dynamics, run.into())
            };

        for d in dynamics.iter() {
            let mut file2path = abspath_with_undersampling
                .join(d.get_name())
                .join(filename.clone());
            file2path.set_extension("csv");
            d.save(&file2path).unwrap();
        }

        let dynamics = if tumour_size > sample_size {
            for d in dynamics.iter_mut() {
                if let Growth::CellCulture(_) = self.growth {
                    // remove all dynamics except the last entry
                    d.clear();
                }
            }
            dynamics
        } else {
            dynamics
        };

        Ok((run, dynamics))
    }

    pub fn compress(self) -> anyhow::Result<()> {
        let mut visited: HashSet<Experiment> =
            HashSet::with_capacity(self.experiments.len());

        for experiment in self.experiments.iter() {
            if !visited.contains(experiment) {
                let src_sample =
                    self.absolute_path.clone().join(format!("{}", experiment));
                let dest_sample =
                    self.relative_path.clone().join(format!("{}", experiment));

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
                                let src =
                                    src_sample.clone().join("mean_dynamics");
                                let dest =
                                    dest_sample.clone().join("mean_dynamics");
                                if self.verbosity > 0 {
                                    println!(
                                        "src {:#?} dest {:#?} ",
                                        src, dest
                                    );
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
                visited.insert((*experiment).clone());
            }
        }
        Ok(())
    }
}

impl Perform for DynamicalApp {
    fn run(self) -> anyhow::Result<()> {
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
                    self.experiments[0].tumour_cells, //assume same size
                    &mut rng,
                );

                let mut dynamics = self.dynamics.clone();

                // for all sequecing experiemnts
                for (i, experiment) in self.experiments.iter().enumerate() {
                    // simulate and update both the run and the dynamics
                    let run_ended = run.simulate(
                        &self.segregation,
                        &mut Some(&mut dynamics),
                        &experiment.tumour_cells,
                        self.rates.estimate_max_iter(&experiment.tumour_cells),
                    );
                    (run, dynamics) = self
                        .save(
                            run_ended,
                            &self.absolute_path,
                            dynamics,
                            &experiment.tumour_cells,
                            &experiment.sample_cells,
                            i,
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

        self.compress()?;

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
    Dynamical(Dynamical),
    Preprocess(Preprocess),
}

#[derive(Debug)]
pub struct Preprocess {
    patient_name: String,
    sample_name: String,
    size: NbIndividuals,
    distribution: Option<PathBuf>,
    mean: Option<f32>,
    frequency: Option<f32>,
    entropy: Option<f32>,
    pub savedir: PathBuf,
    verbosity: u8,
}

impl Preprocess {
    pub fn new(matches: &ArgMatches) -> Self {
        let patient_name: String =
            matches.value_of_t("patient").unwrap_or_else(|e| e.exit());
        let sample_name: String =
            matches.value_of_t("sample").unwrap_or_else(|e| e.exit());
        let size: NbIndividuals =
            matches.value_of_t("size").unwrap_or_else(|e| e.exit());

        let distribution = matches
            .value_of_t::<String>("distribution")
            .ok()
            .map(PathBuf::from);
        let mean = matches.value_of_t("mean").ok();
        let frequency = matches.value_of_t("frequency").ok();
        let entropy = matches.value_of_t("entropy").ok();

        let verbosity = matches.occurrences_of("verbosity") as u8;

        let savedir: PathBuf = matches
            .value_of_t("savedir")
            .unwrap_or_else(|_| std::env::current_dir().unwrap());

        Preprocess {
            patient_name,
            sample_name,
            size,
            distribution,
            mean,
            frequency,
            entropy,
            savedir,
            verbosity,
        }
    }
}

impl InitialStateLoader for Preprocess {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
        if let Some(distribution) = &self.distribution {
            let ecdna = EcDNADistribution::try_from(distribution.as_ref())
                .with_context(|| {
                    format!(
                        "Cannot load the ecDNA distribution from file {:#?}",
                        &distribution
                    )
                })?;

            let summary = ecdna.summarize();
            return Ok((ecdna, summary));
        }
        anyhow::bail!("Cannot load the ecDNA distribution from empty pathbuf")
    }
}

#[derive(Debug)]
pub struct Bayesian {
    pub patient: PathBuf,
    pub runs: usize,
    pub distribution: Option<PathBuf>,
    pub rho1: Vec<f32>,
    pub rho2: Vec<f32>,
    pub delta1: Vec<f32>,
    pub delta2: Vec<f32>,
    pub init_copies: Vec<DNACopy>,
    pub segregation: String,
    pub seed: u64,
    pub growth: String,
    pub savedir: PathBuf,
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

        let growth = if matches.is_present("culture") {
            String::from("culture")
        } else {
            String::from("patient")
        };

        let seed: u64 = matches.value_of_t("seed").unwrap_or(26u64);

        let savedir: PathBuf = matches
            .value_of_t("savedir")
            .unwrap_or_else(|_| std::env::current_dir().unwrap());
        assert!(
            savedir.is_absolute(),
            "Argument --savedir must be used with an absolute path, found {:#?} instead",
            savedir
        );

        let segregation: String =
            matches.value_of_t("segregation").unwrap_or_else(|e| e.exit());

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
            segregation,
            growth,
            savedir,
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
    pub segregation: String,
    pub savedir: PathBuf,
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

        let savedir: PathBuf = matches
            .value_of_t("savedir")
            .unwrap_or_else(|_| std::env::current_dir().unwrap());
        assert!(
            savedir.is_absolute(),
            "Argument --savedir must be used with an absolute path, found {:#?} instead",
            savedir
        );

        if verbosity > 1 {
            println!("dynamics: {:#?}", dynamics);
        }

        let segregation: String =
            matches.value_of_t("segregation").unwrap_or_else(|e| e.exit());

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
            segregation,
            seed,
            tumour_sizes,
            sample_sizes,
            savedir,
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Experiment {
    tumour_cells: NbIndividuals,
    sample_cells: NbIndividuals,
}

impl Experiment {
    pub fn new(
        tumour_cells: NbIndividuals,
        sample_cells: NbIndividuals,
    ) -> Self {
        Experiment { tumour_cells, sample_cells }
    }
}

impl fmt::Display for Experiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}samples{}cells", self.sample_cells, self.tumour_cells)
    }
}

/// Collection of population and sample sizes specifying the longitudinal timepoints.
/// The population size indicates the number of cells in the tumour population
/// when the timepoint has been collected, and the sample size the number of cells
/// sequenced from the subsampling of the whole tumour.
#[derive(Debug)]
pub struct Experiments(Vec<Experiment>);

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
        let mut experiments: Vec<Experiment> =
            Vec::with_capacity(tumour_sizes.len());

        for (tumour_size, sample_size) in
            tumour_sizes.into_iter().zip(sample_sizes.into_iter())
        {
            experiments.push(Experiment {
                tumour_cells: tumour_size,
                sample_cells: sample_size,
            })
        }

        Ok(Experiments(experiments))
    }
}

impl Deref for Experiments {
    type Target = Vec<Experiment>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn compress_dir(
    dest_path: &Path,
    src_path_dir: &Path,
    verbosity: u8,
) -> anyhow::Result<()> {
    //! Compress the directory into tarball. `dest_path` is a relative path of
    //! the dir name within the archive, wherease `src_path_dir` is absolute path.
    let mut tarball = src_path_dir.to_owned();
    tarball.set_extension("tar.gz");

    ensure!(dest_path.is_relative());
    ensure!(tarball.is_absolute());

    // open stream, create encoder to compress and create tar builder to
    // create tarball
    let tar_gz = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tarball)
        .with_context(|| {
            format!(
                "Error while opening the stream {:#?} to save the tarball",
                &tarball
            )
        })?;

    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = tar::Builder::new(enc);

    tar.append_dir_all(dest_path, src_path_dir)
        .with_context(|| {
            format!(
                "Cannot append files to tar archive {:#?} from source {:#?} ",
                &dest_path, &src_path_dir
            )
        })
        .and_then(|()| {
            if verbosity > 1 {
                println!(
                    "{} Gzip {:#?} into {:#?}",
                    Utc::now(),
                    src_path_dir,
                    &dest_path,
                );
            }

            fs::remove_dir_all(src_path_dir).with_context(|| {
                format!("Cannot remove directory {:#?}", &src_path_dir)
            })
        })?;

    Ok(())
}
