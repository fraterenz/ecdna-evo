use crate::{dynamics::app::Dynamics, Simulate, MAX_ITER};
use anyhow::Context;
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use ssa::{
    ecdna::{
        data::EcDNADistribution,
        process::{
            BirthDeathMeanTimeEcDNA, BirthDeathTimeEcDNA,
            PureBirthMeanTimeEcDNA, PureBirthTimeEcDNA,
        },
        proliferation::{EcDNAGrowth, Exponential},
        segregation::{
            BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
            Segregation,
        },
    },
    iteration::Iteration,
    NbIndividuals, Process,
};
use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "ecdna")]
#[command(
    about = "Mathematical modelling of the ecDNA dynamics",  // TODO
    long_about = "Study the effect of the random segregation on the ecDNA dynamics using a stochastic simulation algorithm (SSA) aka Gillespie algorithm"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

impl Cli {
    pub fn build() -> anyhow::Result<Box<dyn Simulate>> {
        let args = Cli::parse();

        match args.command {
            Commands::Dynamics {
                segregation,
                growth,
                b0,
                b1,
                d0,
                d1,
                cells,
                mean,
                path,
                subsampling,
                save,
                initial,
                seed,
                runs,
                debug,
                sequential,
                verbose,
            } => {
                if save.is_some() {
                    todo!()
                }
                // if both d0, d1 are either unset or equal to 0, pure birth,
                // else birthdeath
                let (is_birth_death_d0, d0) = match d0 {
                    Some(rate) => (rate > 0f32, rate),
                    None => (false, 0f32),
                };

                let (is_birth_death_d1, d1) = match d1 {
                    Some(rate) => (rate > 0f32, rate),
                    None => (false, 0f32),
                };
                let segregation: Segregation = segregation.into();

                let is_birth_death = is_birth_death_d0 | is_birth_death_d1;
                let iterations = if is_birth_death {
                    MAX_ITER * cells as usize
                } else {
                    cells as usize
                };

                // if no initial distribution, start with 1 cell with 1 ecDNA copy
                let distribution = match initial {
                    Some(path) => EcDNADistribution::load(&path).with_context(|| format!("Cannot load the ecDNA distribution from {:#?}", path)).unwrap(),
                    None => EcDNADistribution::new(HashMap::<u16, NbIndividuals>::from([(1, 1)]), iterations)

                };
                let (nminus, nplus) =
                    (*distribution.get_nminus(), distribution.compute_nplus());

                let growth = match growth {
                    GrowthOptions::Constant => todo!(),
                    GrowthOptions::Exponential => {
                        EcDNAGrowth::Exponential(Exponential { segregation })
                    }
                };

                let with_mean = mean.is_some();
                let cells = if debug { 100 } else { cells };
                let verbose = if debug { u8::MAX } else { verbose };

                let process: Process = match is_birth_death {
                    true => {
                        let initial_population =
                            [nminus, nplus, nminus, nplus];
                        let iteration = Iteration::new(
                            [b0, b1, d0, d1],
                            initial_population,
                            cells,
                            iterations,
                        );
                        match with_mean {
                            true => Process::EcDNAProcess(
                                BirthDeathMeanTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            ),
                            false => Process::EcDNAProcess(
                                BirthDeathTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            ),
                        }
                    }
                    false => {
                        let initial_population = [nminus, nplus];
                        let iteration = Iteration::new(
                            [b0, b1],
                            initial_population,
                            cells,
                            iterations,
                        );

                        match with_mean {
                            true => Process::EcDNAProcess(
                                PureBirthMeanTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            ),
                            false => Process::EcDNAProcess(
                                PureBirthTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            ),
                        }
                    }
                };

                Ok(Box::new(Dynamics {
                    subsampling,
                    process,
                    seed,
                    path2dir: path,
                    runs,
                    save,
                    debug,
                    sequential,
                    verbose,
                }))
            }
            Commands::Abc { .. } => todo!(),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(arg_required_else_help = true)]
    /// Simulate the ecDNA dynamics
    Dynamics {
        /// The ecDNA segregation type
        #[arg(long, default_value_t = SegregationOptions::BinomialSegregation )]
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
        /// Number of cells to simulate
        #[arg(
            long,
            short,
            default_value_t = 100000,
            conflicts_with = "debug"
        )]
        cells: NbIndividuals,
        /// Seed for reproducibility
        #[arg(long, default_value_t = 26)]
        seed: u64,
        /// Triggers debug mode: max verbosity and 1 sequential simulation
        #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
        debug: bool,
        /// Run sequentially each run instead of using rayon for parallelisation
        #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false, conflicts_with = "debug")]
        sequential: bool,
        /// Whether to track over simulations the evolution of the mean ecDNA
        /// copies in the tumour population
        #[arg(short, long, action = ArgAction::SetTrue)]
        mean: Option<bool>,
        /// Path to store the results of the simulations
        #[arg(value_name = "DIR", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.is_dir() { Ok(path_b) } else { Err("Cannot find dir") }} ) ]
        path: PathBuf,
        /// The JSON file used as an initial starting distribution
        #[arg(long, value_name = "FILE", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.extension() == Some(std::ffi::OsStr::new("json")) { Ok(path_b) } else { Err("Must be JSON file: extension must be .json)") }} ) ]
        initial: Option<PathBuf>,
        /// The number of cells kept for each subsampling
        #[arg(
            long,
            value_name = "CELLS",
            num_args = 0..,
            value_enum,
            conflicts_with = "save",
            long_help = "The number of cells kept for each subsampling. Samples are taken as soon as the tumour reaches `cells` cells. Then, the tumour restarts growing from these subsample."
        )]
        subsampling: Option<Vec<NbIndividuals>>,
        #[arg(short, long, default_value_t = 12, conflicts_with = "debug")]
        /// Number of independent realisations to simulate of the same stochastic process
        runs: usize,
        #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
        verbose: u8,
        /// Timepoints at which we save the ecDNA distribution
        #[arg(
            long,
            value_name = "CELLS",
            num_args = 0..,
            value_enum
        )]
        save: Option<Vec<NbIndividuals>>,
    },
    Abc(Abc),
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

#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]

struct Abc {
    #[command(subcommand)]
    command: Option<StashCommands>,

    #[command(flatten)]
    push: StashPush,
}

#[derive(Debug, Subcommand)]

enum StashCommands {
    Push(StashPush),
    Pop { stash: Option<String> },
    Apply { stash: Option<String> },
}

#[derive(Debug, Args)]

struct StashPush {
    #[arg(short, long)]
    message: Option<String>,
}
