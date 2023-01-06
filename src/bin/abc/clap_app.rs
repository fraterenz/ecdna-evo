use clap::{ArgAction, ArgGroup, Parser, ValueEnum};
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64Mcg;
use ssa::{
    ecdna::{
        data::EcDNADistribution,
        process::ABC,
        proliferation::Exponential,
        segregation::{
            BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
            RandomSegregation, Segregation,
        },
    },
    iteration::Iteration,
    NbIndividuals,
};
use std::{collections::HashMap, path::PathBuf};

use crate::{
    abc::{Data, Input, Summaries},
    app::Abc,
    SimulationOptions,
};

pub enum Parallel {
    False,
    True,
    Debug,
}

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "Abc")]
#[command(
    about = "Infer the birth-rate (fitness coefficient) from the data",
    group(ArgGroup::new("input")
        .required(true)
        .args(["mean", "data", "frequency", "entropy"])
        )
        )
]
pub struct Cli {
    /// Infer the fitness coefficient (birth-rate of cells with ecDNAs) in a
    /// pure-birth stochastic process.
    /// Proliferation rate of the cells without ecDNAs (wild-type)
    #[arg(long, value_name = "RATE", default_value_t = 1.)]
    b0: f32,
    /// proliferation rate of the cells with ecdnas.
    /// when a range is specified, abc samples a rate randomly from this range
    #[arg(long, required = true, value_name = "rate", num_args = 2)]
    b1: Vec<f32>,
    #[arg(short, long, default_value_t = 100)]
    /// number of independent runs used to recover the posterior distribution
    /// of the fitness coefficient
    runs: usize,
    /// number of cells to simulate
    #[arg(long, short, default_value_t = 100000)]
    cells: NbIndividuals,
    /// seed for reproducibility
    #[arg(long, default_value_t = 26)]
    seed: u64,
    /// triggers sequential mode
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false, conflicts_with = "debug")]
    sequential: bool,
    /// triggers debug mode: max verbosity and 1 sequential simulation
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    debug: bool,
    #[arg(value_name = "DIR", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.is_dir() { Ok(path_b) } else { Err("Cannot find dir") }} ) ]
    path: PathBuf,
    /// use the mean to infer the posterior distribution
    #[arg(long, conflicts_with = "data")]
    mean: Option<f32>,
    /// use the ferquency to infer the posterior distribution
    #[arg(long, conflicts_with = "data")]
    frequency: Option<f32>,
    /// use the entropy to infer the posterior distribution
    #[arg(long, conflicts_with = "data")]
    entropy: Option<f32>,
    /// The tumour size at which we believe the subsample of the tumour has
    /// been taken
    #[arg(long)]
    subsample: Option<NbIndividuals>,
    /// the ecdna distribution used to infer the posterior distribution.
    /// when present, abc will compute from this file the mean, the entropy as well as the frequency.
    #[arg(long, value_name = "FILE", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.extension() == Some(std::ffi::OsStr::new("json")) { Ok(path_b) } else { Err("must be json file: extension must be .json)") }} ) ]
    data: Option<PathBuf>,
    #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
    verbose: u8,
}

impl Cli {
    pub fn build() -> anyhow::Result<SimulationOptions> {
        let cli = Cli::parse();

        let (parallel, runs) = if cli.debug {
            (Parallel::Debug, 1)
        } else if cli.sequential {
            (Parallel::False, cli.runs)
        } else {
            (Parallel::True, cli.runs)
        };

        // we assume fixed initial pop for now, starting with one cell
        // with 1 ecDNA
        let initial_population = [0, 1];
        let distribution = EcDNADistribution::new(
            HashMap::from([(0, 0), (1, 1)]),
            cli.cells as usize,
        );

        // load data
        let input = match cli.data {
            Some(path2file) => Input::Distribution(EcDNADistribution::load(
                &path2file,
                cli.cells as usize,
            )?),
            None => Input::Summaries(Summaries::new(
                cli.mean,
                cli.frequency,
                cli.entropy,
            )),
        };
        let data = Data::from_input(input);

        // create proceses to simulate
        let mut rng = Pcg64Mcg::seed_from_u64(cli.seed);
        // we assume pure birth only, hence 2
        let iterations = (0..runs).map(|_| {
            // sample at random a fitness, assuming b1[0] > b1[1]
            let b1 = Uniform::new(cli.b1[0], cli.b1[1]).sample(&mut rng);
            Iteration::new(
                [cli.b0, b1],
                initial_population,
                cli.cells,
                cli.cells as usize,
            )
        });
        let processes = iterations
            .into_iter()
            .map(|iteration| {
                ABC::new(
                    Exponential {
                        segregation: Segregation::Random(
                            RandomSegregation::BinomialSegregation(
                                BinomialSegregation,
                            ),
                        ),
                    },
                    iteration,
                    distribution.clone(),
                    cli.verbose,
                )
                .expect("Cannot create the ABC simulation")
            })
            .collect();

        Ok(SimulationOptions {
            simulation: Abc {
                seed: cli.seed,
                path2dir: cli.path,
                verbose: cli.verbose,
                target: data,
            },
            parallel,
            processes,
            subsample: cli.subsample,
        })
    }
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum SegregationOptions {
    Deterministic,
    BinomialNoUneven,
    BinomialSegregation,
    BinomialNoNminus,
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
enum GrowthOptions {
    Exponential,
    Constant,
}

impl From<SegregationOptions> for Segregation {
    fn from(segregation: SegregationOptions) -> Self {
        match segregation {
            SegregationOptions::Deterministic => Self::Deterministic,
            SegregationOptions::BinomialNoUneven => {
                Self::Random(BinomialNoUneven(BinomialSegregation).into())
            }
            SegregationOptions::BinomialNoNminus => {
                Self::Random(BinomialNoNminus(BinomialSegregation).into())
            }
            SegregationOptions::BinomialSegregation => {
                Self::Random(BinomialSegregation.into())
            }
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
