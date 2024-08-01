use anyhow::Context;
use clap::{ArgAction, Parser, ValueEnum};
use ecdna_evo::{
    distribution::EcDNADistribution,
    segregation::{
        Binomial, BinomialNoNminus, BinomialNoUneven, Deterministic, Segregate,
    },
    SnapshotCells,
};
use rand::Rng;
use sosa::{IterTime, NbIndividuals, Options};
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
};

use crate::{SimulationOptions, MAX_CELLS, MAX_ITER};

#[derive(Debug)]
pub enum Parallel {
    False,
    True,
    Debug,
}

#[derive(Debug, Parser)]
#[command(name = "Dynamics")]
#[command(
    version,
    about = "Agent-based modelling of the ecDNA dynamics",
    long_about = "Study the effect of the random segregation and positive selection on the ecDNA dynamics using a stochastic simulation algorithm (SSA) aka Gillespie algorithm"
)]
pub struct Cli {
    /// The ecDNA segregation type
    #[arg(long, default_value_t = SegregationOptions::Binomial)]
    segregation: SegregationOptions,
    /// The tumour growth model
    #[arg(long, default_value_t = GrowthOptions::Exponential )]
    growth: GrowthOptions,
    /// Proliferation rate of the cells without ecDNAs (wild-type)
    #[arg(long, value_name = "RATE", default_value_t = 1.)]
    b0: f32,
    /// Proliferation rate of the cells with ecDNAs
    #[arg(long, value_name = "RATE", default_value_t = 1.)]
    b1: f32,
    /// Death rate of the cells without ecDNAs (wild-type).
    /// If both `d0` and `d1` are empty or set to zero, run a pure-birth
    /// process.
    #[arg(long, value_name = "RATE")]
    d0: Option<f32>,
    /// Death rate of the cells with ecDNAs.
    /// If both `d0` and `d1` are empty or set to zero, run a pure-birth
    /// process.
    #[arg(long, value_name = "RATE")]
    d1: Option<f32>,
    /// Number of years to simulate before stopping the simulation
    #[arg(short, long, group = "stop", conflicts_with = "debug")]
    years: Option<NbIndividuals>,
    /// Number of cells to simulate before stopping the simulation
    #[arg(short, long, group = "stop", conflicts_with = "debug")]
    cells: Option<NbIndividuals>,
    /// Seed for reproducibility
    #[arg(long, default_value_t = 26)]
    seed: u64,
    /// Triggers debug mode: max verbosity, 1 sequential simulation and 10 cells
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    debug: bool,
    /// Run sequentially each run instead of using rayon for parallelisation
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false, conflicts_with = "debug")]
    sequential: bool,
    /// Path to store the results of the simulations
    #[arg(value_name = "DIR")]
    path: PathBuf,
    /// The JSON file used as an initial starting distribution
    #[arg(
        long,
        value_name = "FILE",
        value_parser = |path: &str| {
            let path_b = PathBuf::from(path);
            if path_b.extension() == Some(std::ffi::OsStr::new("json")) {
                Ok(path_b)
            } else {
                Err("Must be JSON file: extension must be .json)")
            }
        }
        )
    ]
    initial: Option<PathBuf>,
    #[arg(short, long, default_value_t = 12, conflicts_with = "debug")]
    /// Number of independent realisations to simulate of the same stochastic process
    runs: usize,
    #[arg(long, value_delimiter = ',', require_equals = true, num_args = 0..)]
    /// Subsample the ecDNA distribution at the end of the simulation
    subsamples: Option<Vec<u64>>,
    #[arg(long, value_delimiter = ',', require_equals = true, num_args = 0..)]
    /// Number of cells that will trigger the saving of the ecDNA distribution.
    snapshots: Option<Vec<u64>>,
    #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
    verbosity: u8,
}

fn build_snapshots(
    cells: NbIndividuals,
    snapshots: Option<Vec<NbIndividuals>>,
) -> anyhow::Result<VecDeque<SnapshotCells>> {
    let mut snapshots = match snapshots {
        Some(s) => VecDeque::from_iter(
            s.into_iter().map(|cells| SnapshotCells { cells }),
        ),
        None => VecDeque::from(build_snapshots_from_cells(11, cells)),
    };

    snapshots.make_contiguous();
    snapshots
        .as_mut_slices()
        .0
        .sort_by(|s1, s2| s1.cells.partial_cmp(&s2.cells).unwrap());
    Ok(snapshots)
}

fn build_snapshots_from_cells(
    n_snapshots: usize,
    cells: NbIndividuals,
) -> Vec<SnapshotCells> {
    let dx = cells / (n_snapshots as NbIndividuals - 1);
    let mut x = vec![1; n_snapshots];
    for i in 1..n_snapshots - 1 {
        x[i] = x[i - 1] + dx;
    }

    x.shrink_to_fit();
    x[n_snapshots - 1] = cells;
    x.into_iter().map(|v| SnapshotCells { cells: v }).collect()
}

impl Cli {
    pub fn build() -> anyhow::Result<SimulationOptions> {
        let cli = Cli::parse();

        let (cells, years, verbosity, parallel, runs) = if cli.debug {
            (300, 2, u8::MAX, Parallel::Debug, 1)
        } else if let Some(years) = cli.years {
            if cli.sequential {
                (MAX_CELLS, years, cli.verbosity, Parallel::False, cli.runs)
            } else {
                (MAX_CELLS, years, cli.verbosity, Parallel::True, cli.runs)
            }
        } else {
            let cells = cli.cells.unwrap_or(1_000);
            // approx. is it ok?
            let years = (f32::log2(cells as f32) + 4f32) as u64;
            if cli.sequential {
                (cells, years, cli.verbosity, Parallel::False, cli.runs)
            } else {
                (cells, years, cli.verbosity, Parallel::True, cli.runs)
            }
        };

        let snapshots = build_snapshots(cells, cli.snapshots).unwrap();
        let segregation = cli.segregation.into();
        let subsamples = cli.subsamples;

        // if both d0, d1 are either unset or equal to 0, pure birth,
        // else birthdeath
        let (is_birth_death_d0, d0) = match cli.d0 {
            Some(rate) => (rate > 0f32, rate),
            None => (false, 0f32),
        };

        let (is_birth_death_d1, d1) = match cli.d1 {
            Some(rate) => (rate > 0f32, rate),
            None => (false, 0f32),
        };
        let is_birth_death = is_birth_death_d0 | is_birth_death_d1;

        // if no initial distribution, start with 1 cell with 1 ecDNA copy
        let distribution = match cli.initial {
            Some(path) => EcDNADistribution::load(
                &path, MAX_ITER, // TODO double check this
            )
            .with_context(|| {
                format!(
                    "Cannot load the ecDNA distribution from {:#?}",
                    cli.path
                )
            })
            .unwrap(),
            None => EcDNADistribution::new(
                HashMap::<u16, NbIndividuals>::from([(1, 1)]),
                MAX_ITER, // TODO double check this
            ),
        };

        let process_type = {
            if is_birth_death {
                ProcessType::BirthDeath
            } else {
                ProcessType::PureBirth
            }
        };

        let options = SimulationOptions {
            seed: cli.seed,
            options: Options {
                max_iter_time: IterTime { iter: MAX_ITER, time: years as f32 },
                init_iter: 0,
                max_cells: cells,
                verbosity,
            },
            path2dir: cli.path,
            process_type,
            b0: cli.b0,
            b1: cli.b1,
            d0,
            d1,
            distribution,
            parallel,
            segregation,
            snapshots,
            growth: cli.growth,
            subsamples,
            runs,
        };
        if verbosity > 0 {
            println!("running sims with {:#?}", options);
        }

        Ok(options)
    }
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum SegregationOptions {
    Deterministic,
    BinomialNoUneven,
    Binomial,
    BinomialNoNminus,
}

#[derive(Debug, Clone, Copy)]
pub enum Segregation {
    Deterministic(Deterministic),
    BinomialNoUneven(BinomialNoUneven),
    BinomialNoNminus(BinomialNoNminus),
    Binomial(Binomial),
}

impl Segregate for Segregation {
    fn ecdna_segregation(
        &self,
        copies: ecdna_evo::segregation::DNACopySegregating,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, ecdna_evo::segregation::IsUneven) {
        match self {
            Self::Deterministic(s) => {
                s.ecdna_segregation(copies, rng, verbosity)
            }
            Self::BinomialNoUneven(s) => {
                s.ecdna_segregation(copies, rng, verbosity)
            }
            Self::BinomialNoNminus(s) => {
                s.ecdna_segregation(copies, rng, verbosity)
            }
            Self::Binomial(s) => s.ecdna_segregation(copies, rng, verbosity),
        }
    }
}

impl std::fmt::Display for SegregationOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum GrowthOptions {
    Exponential,
    Constant,
}

impl From<SegregationOptions> for Segregation {
    fn from(segregation: SegregationOptions) -> Self {
        match segregation {
            SegregationOptions::Deterministic => {
                Segregation::Deterministic(Deterministic)
            }
            SegregationOptions::BinomialNoUneven => {
                Segregation::BinomialNoUneven(BinomialNoUneven(Binomial))
            }
            SegregationOptions::BinomialNoNminus => {
                Segregation::BinomialNoNminus(BinomialNoNminus(Binomial))
            }
            SegregationOptions::Binomial => Segregation::Binomial(Binomial),
        }
    }
}

impl std::fmt::Display for GrowthOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ProcessType {
    PureBirth,
    BirthDeath,
}

#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, ValueEnum,
)]
enum SamplingStrategyArg {
    #[default]
    Uniform,
    Gaussian,
}
