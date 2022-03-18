use crate::clap_app::clap_app;
use anyhow::{anyhow, ensure, Context};
use chrono::Utc;
use clap::ArgMatches;
use ecdna_evo::{
    data::{EcDNADistribution, EcDNASummary, Entropy, Frequency, Mean},
    dynamics::{Dynamic, Dynamics, Name},
    patient::{Patient, SequencingData, SequencingDataBuilder},
    run::Run,
    NbIndividuals, Rates,
};
use enum_dispatch::enum_dispatch;
use flate2::{write::GzEncoder, Compression};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn build_config() -> Config {
    //! Build the app by parsing CL arguments with `clap` to build the structs
    //! required by `Simulation` to run the stochastic simulation.
    let matches = clap_app().get_matches();
    return match matches.subcommand() {
        Some(("abc", abc_matches)) => {
            if abc_matches.is_present("longitudinal") {
                Config::Longitudinal(Longitudinal::new(abc_matches))
            } else if abc_matches.is_present("subsampled") {
                Config::Subsampled(Subsampled::new(abc_matches))
            } else {
                Config::Bayesian(Bayesian::new(abc_matches))
            }
        }
        Some(("simulate", dynamical_matches)) => {
            Config::Dynamical(Dynamical::new(dynamical_matches))
        }
        _ => unreachable!("Expect subcomands `simulate` or `abc`"),
    };
}

fn rates_from_args(matches: &ArgMatches) -> Rates {
    // Proliferation rate of the cells w/ ecDNA
    let fitness1: Vec<f32> =
        matches.values_of_t("rho1").unwrap_or_else(|e| e.exit());
    // Proliferation rate of the cells w/o ecDNA
    let fitness2: Vec<f32> =
        matches.values_of_t("rho2").unwrap_or_else(|e| e.exit());

    // Death rate of cells w/ ecDNA
    let death1: Vec<f32> =
        matches.values_of_t("delta1").unwrap_or_else(|e| e.exit());
    let death2: Vec<f32> =
        matches.values_of_t("delta2").unwrap_or_else(|e| e.exit());

    Rates::new(&fitness1, &fitness2, &death1, &death2)
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
    Bayesian(BayesianApp),
    Londitudinal(LonditudinalApp),
    Subsampled(SubsampledApp),
    Dynamical(DynamicalApp),
}

/// Infer the most probable coefficients from the data from one sequencing experiment.
#[derive(Debug)]
pub struct BayesianApp {
    pub patient: Patient,
    pub runs: usize,
    pub rates: Rates,
    pub ecdna: EcDNADistribution,
    pub mean: Mean,
    pub frequency: Frequency,
    pub entropy: Entropy,
    pub to_infer: String,
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
                "Initial ecDNA distribution {:#?} with {:#?}",
                ecdna, summary
            );
        }

        let relative_path = PathBuf::from("results")
            .join(PathBuf::from(patient.name.clone()))
            .join("abc");
        let absolute_path =
            std::env::current_dir()?.join(relative_path.clone());

        let app = BayesianApp {
            patient,
            runs: config.runs,
            rates: config.rates,
            ecdna,
            mean: summary.mean,
            frequency: summary.frequency,
            entropy: summary.entropy,
            to_infer: config.to_infer,
            relative_path,
            absolute_path,
            verbosity: config.verbosity,
        };

        if config.verbosity > 1 {
            println!("{:#?}", app);
        }

        Ok(app)
    }
}

impl Perform for BayesianApp {
    fn run(&mut self) -> anyhow::Result<()> {
        ensure!(
            self.patient.samples.len() == 1,
            "Cannot run Bayesian app with multiple patient's samples"
        );

        println!(
            "{} Start inferring {} using ABC with {} runs for patient {}",
            Utc::now(),
            self.to_infer,
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

        (0..self.runs)
            .into_par_iter()
            .progress_count(self.runs as u64)
            .for_each(|idx| {
                let cells = sample2infer.get_tumour_size();
                Run::new(idx, 0f32, 0usize, &self.ecdna, &self.rates)
                    .simulate(None, cells, self.rates.estimate_max_iter(cells))
                    .save(&self.absolute_path, None, &Some(sample2infer))
                    .with_context(|| format!("Cannot save run {}", idx))
                    .unwrap();
            });

        println!(
            "{} End inferring {} using ABC with {} runs",
            Utc::now(),
            self.to_infer,
            self.runs
        );
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

/// Infer the most probable coefficients from the data using multiple timepoints
/// (longitudinal sequencing experiment).
pub struct LonditudinalApp(BayesianApp);

impl LonditudinalApp {
    pub fn new(config: Longitudinal) -> anyhow::Result<Self> {
        Ok(LonditudinalApp(BayesianApp::new(config.0)?))
    }
}

impl Perform for LonditudinalApp {
    fn run(&mut self) -> anyhow::Result<()> {
        ensure!(
            self.0.patient.samples.len() > 1,
            "Cannot run Longitudinal app with one single patient's sample"
        );

        println!(
            "{} Start inferring {} using ABC with {} runs for patient {}",
            Utc::now(),
            self.0.to_infer,
            self.0.runs,
            self.0.patient.name
        );

        // we start the parameters' inference for the first sample
        let sample2infer = self.0.patient.samples.first().unwrap();

        if self.0.verbosity > 0 {
            println!("{} Starting running the inference", Utc::now());
        }

        (0..self.0.runs)
            .into_iter()
            // .into_par_iter()
            // .progress_count(self.0.runs as u64)
            .for_each(|idx| {
                let mut cells = sample2infer.get_tumour_size();
                // first timepoint
                let mut run =
                    Run::new(idx, 0f32, 0usize, &self.0.ecdna, &self.0.rates)
                        .simulate(
                            None,
                            cells,
                            self.0.rates.estimate_max_iter(cells),
                        )
                        .save(&self.0.absolute_path, None, &Some(sample2infer))
                        .with_context(|| format!("Cannot save run {}", idx))
                        .unwrap();

                // remaining timepoints
                for t in self.0.patient.samples.iter().skip(1) {
                    // continue from where we left at `cells` inidividuals
                    cells = t.get_tumour_size();
                    run = run
                        .continue_simulation(
                            cells,
                            self.0.rates.estimate_max_iter(cells),
                        )
                        .save(&self.0.absolute_path, None, &Some(t))
                        .unwrap();
                }
            });

        println!(
            "{} End inferring {} using ABC with {} runs",
            Utc::now(),
            self.0.to_infer,
            self.0.runs
        );
        Ok(())
    }
}

impl Tarball for LonditudinalApp {
    fn compress(self) -> anyhow::Result<()> {
        self.0.compress()
    }
}

/// Infer the most probable coefficients from the data subsampling the final population.
pub struct SubsampledApp(BayesianApp);

impl SubsampledApp {
    pub fn new(config: Subsampled) -> anyhow::Result<Self> {
        Ok(SubsampledApp(BayesianApp::new(config.0)?))
    }
}

impl Perform for SubsampledApp {
    fn run(&mut self) -> anyhow::Result<()> {
        ensure!(
            self.0.patient.samples.len() == 1,
            "Cannot run Subsampled app with multiple patient's samples"
        );

        let is_undersampled =
            self.0.patient.samples.first().unwrap().is_undersampled();
        ensure!(is_undersampled.is_some(), "Cannot perform subsampled app bayesian inference without any ecDNA distribution as input");

        ensure!(is_undersampled.unwrap(), "Flag --subsampled passed without any subsampling in patient's data: ntot and the distribution have the same number of cells");
        self.0.run()
    }
}

impl Tarball for SubsampledApp {
    fn compress(self) -> anyhow::Result<()> {
        self.0.compress()
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
    pub max_iter: usize,
    relative_path: PathBuf,
    absolute_path: PathBuf,
    pub verbosity: u8,
}

impl DynamicalApp {
    pub fn new(config: Dynamical) -> anyhow::Result<Self> {
        let (ecdna, summary) = config.load().with_context(|| {
            "Cannot load initial state for DynamicalApp app"
        })?;
        let max_iter = config.rates.estimate_max_iter(&config.cells);

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

        let app = DynamicalApp {
            patient_name: config.patient_name,
            runs: config.runs,
            rates: config.rates,
            dynamics: dynamics.into(),
            size: config.cells,
            ecdna,
            max_iter,
            relative_path,
            absolute_path,
            verbosity: config.verbosity,
        };

        if config.verbosity > 1 {
            println!("{:#?}", app);
        }
        Ok(app)
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
                Run::new(idx, 0f32, 0usize, &self.ecdna, &self.rates)
                    .simulate(
                        Some(&mut self.dynamics.clone()),
                        &self.size,
                        self.max_iter,
                    )
                    .save(&self.absolute_path, Some(&self.dynamics), &None)
                    .with_context(|| format!("Cannot save run {}", idx))
                    .unwrap();
            });

        println!(
            "{} End simulating dynamics with {} runs",
            Utc::now(),
            &self.runs
        );
        Ok(())
    }
}

impl Tarball for DynamicalApp {
    fn compress(self) -> anyhow::Result<()> {
        for d in self.dynamics.iter() {
            if self.verbosity > 0 {
                println!(
                    "{} Creating tarball for dynamics {}",
                    Utc::now(),
                    d.get_name()
                );
            }
            let src = self.absolute_path.clone().join(d.get_name());
            let dest = self.relative_path.clone().join(d.get_name());
            if self.verbosity > 0 {
                println!("src {:#?} dest {:#?} ", src, dest);
            }
            compress_dir(&dest, &src, self.verbosity)
                .expect("Cannot compress dynamics");
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
    Longitudinal(Longitudinal),
    Subsampled(Subsampled),
    Dynamical(Dynamical),
}

pub struct Bayesian {
    pub patient: PathBuf,
    pub runs: usize,
    pub distribution: Option<PathBuf>,
    pub rates: Rates,
    pub to_infer: String,
    pub verbosity: u8,
}

impl Bayesian {
    pub fn new(matches: &ArgMatches) -> Bayesian {
        let (verbosity, distribution, runs) = match_shared_args(matches);

        let to_infer: String =
            matches.value_of_t("infer").unwrap_or_else(|e| e.exit());
        let patient: PathBuf =
            matches.value_of_t("patient").unwrap_or_else(|e| e.exit());
        // Rates of the two-type stochastic birth-death process
        let rates = rates_from_args(matches);

        Bayesian { patient, runs, distribution, rates, verbosity, to_infer }
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

pub struct Longitudinal(pub Bayesian);

impl Longitudinal {
    pub fn new(matches: &ArgMatches) -> Longitudinal {
        Longitudinal(Bayesian::new(matches))
    }
}

impl InitialStateLoader for Longitudinal {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
        self.0.load()
    }
}

pub struct Subsampled(pub Bayesian);

impl Subsampled {
    pub fn new(matches: &ArgMatches) -> Subsampled {
        Subsampled(Bayesian::new(matches))
    }
}

impl InitialStateLoader for Subsampled {
    fn load(&self) -> anyhow::Result<(EcDNADistribution, EcDNASummary)> {
        self.0.load()
    }
}

pub struct Dynamical {
    pub patient_name: String,
    pub kind: Vec<String>,
    pub runs: usize,
    pub cells: NbIndividuals,
    pub distribution: Option<PathBuf>,
    pub rates: Rates,
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
        let dynamics: Vec<String> =
            matches.values_of_t("dynamics").unwrap_or_else(|e| e.exit());

        // Rates of the two-type stochastic birth-death process
        let rates = rates_from_args(matches);

        if verbosity > 1 {
            println!("dynamics: {:#?}", dynamics);
        }

        Dynamical {
            kind: dynamics,
            patient_name,
            runs,
            cells,
            distribution,
            rates,
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

// /// Defaults to tracking the cells w/ ecDNA in the neutral case with 100 runs, 10000 individual
// /// to simulate and starting with a single cell with one ecDNA copy.
// impl Default for Config {
//     fn default() -> Self {
//         Config (Dynamical {
//             runs: 100usize,
//             cells: 10000 as NbIndividuals,
//             distribution: PathBuf::from("results/one_initial_nplus.json"),
//             command: Commands::Dynamical(Dynamical::default()),
//             verbosity: 0u8,
//         }
//     }
// }

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

/// Generate anyhow error if `val` is `f32::NAN`
pub fn find_nan(val: f32) -> anyhow::Result<f32> {
    if val.is_nan() {
        return Err(anyhow!("Found NaN value!"));
    }
    Ok(val)
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
}
