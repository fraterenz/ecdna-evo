use anyhow::Context;
use clap::{ArgAction, Parser, ValueEnum};
use ssa::{
    ecdna::{
        data::EcDNADistribution,
        process::{
            BirthDeathMeanTimeEcDNA, BirthDeathTimeEcDNA, PureBirthEcDNA,
            PureBirthMeanTimeEcDNA, PureBirthTimeEcDNA,
        },
        proliferation::{EcDNAGrowth, Exponential},
        segregation::{
            BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
            Segregation,
        },
    },
    iteration::Iteration,
    NbIndividuals, Process, RestartGrowth,
};
use std::{collections::HashMap, path::PathBuf};

use crate::{app::Dynamics, SamplingOptions, SimulationOptions, MAX_ITER};

pub enum Parallel {
    False,
    True,
    Debug,
}

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "Dynamics")]
#[command(
    about = "Mathematical modelling of the ecDNA dynamics",  // TODO
    long_about = "Study the effect of the random segregation on the ecDNA dynamics using a stochastic simulation algorithm (SSA) aka Gillespie algorithm"
)]
pub struct Cli {
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
    #[arg(long, short, default_value_t = 100000, conflicts_with = "debug")]
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
    /// Whether to track over simulations the evolution of the gillespie
    /// time, that is the waiting time for each reaction.
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    time: bool,
    /// Whether to track over simulations the evolution of the mean ecDNA
    /// copies in the tumour population
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    mean: bool,
    /// The number of cells kept after subsampling.
    #[arg(long, num_args = 0.., value_name = "CELLS", requires = "restart_growth")]
    subsample: Option<Vec<NbIndividuals>>,
    /// Whether to restart tumour growth from the sample (cell-lines) or from
    /// the whole population
    #[arg(long, requires = "subsample")]
    restart_growth: Option<RestartGrowthOptions>,
    /// Path to store the results of the simulations
    #[arg(
        value_name = "DIR",
        value_parser = |path: &str| {
            let path_b = PathBuf::from(path);
            if path_b.is_dir() { Ok(path_b) } else { Err("Cannot find dir") }
        }
        )
    ]
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
    #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
    verbose: u8,
}

impl Cli {
    pub fn build() -> anyhow::Result<SimulationOptions> {
        let cli = Cli::parse();

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
        let segregation: Segregation = cli.segregation.into();

        let is_birth_death = is_birth_death_d0 | is_birth_death_d1;
        let iterations = if is_birth_death {
            MAX_ITER * cli.cells as usize
        } else {
            cli.cells as usize
        };

        // if no initial distribution, start with 1 cell with 1 ecDNA copy
        let distribution = match cli.initial {
            Some(path) => EcDNADistribution::load(&path, iterations)
                .with_context(|| {
                    format!(
                        "Cannot load the ecDNA distribution from {:#?}",
                        cli.path
                    )
                })
                .unwrap(),
            None => EcDNADistribution::new(
                HashMap::<u16, NbIndividuals>::from([(1, 1)]),
                iterations,
            ),
        };
        let (nminus, nplus) =
            (*distribution.get_nminus(), distribution.compute_nplus());

        let growth = match cli.growth {
            GrowthOptions::Constant => todo!(),
            GrowthOptions::Exponential => {
                EcDNAGrowth::Exponential(Exponential { segregation })
            }
        };

        let cells = if cli.debug { 100 } else { cli.cells };
        let verbose = if cli.debug { u8::MAX } else { cli.verbose };

        let process: Process = match is_birth_death {
            true => {
                let initial_population = [nminus, nplus, nminus, nplus];
                let iteration = Iteration::new(
                    [cli.b0, cli.b1, d0, d1],
                    initial_population,
                    cells,
                    iterations,
                );
                match cli.mean {
                    true => {
                        if cli.time {
                            Process::EcDNAProcess(
                                BirthDeathMeanTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            )
                        } else {
                            todo!();
                            // Process::EcDNAProcess(
                            //     BirthDeathMeanEcDNA::new(
                            //         0f32,
                            //         growth,
                            //         iteration,
                            //         distribution,
                            //         verbose,
                            //     )?
                            //     .into(),
                            // )
                        }
                    }
                    false => {
                        if cli.time {
                            Process::EcDNAProcess(
                                BirthDeathTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            )
                        } else {
                            todo!();
                            // Process::EcDNAProcess(
                            //     BirthDeathEcDNA::new(
                            //         0f32,
                            //         growth,
                            //         iteration,
                            //         distribution,
                            //         verbose,
                            //     )?
                            //     .into(),
                            // )
                        }
                    }
                }
            }
            false => {
                let initial_population = [nminus, nplus];
                let iteration = Iteration::new(
                    [cli.b0, cli.b1],
                    initial_population,
                    cells,
                    iterations,
                );

                match cli.mean {
                    true => {
                        if cli.time {
                            Process::EcDNAProcess(
                                PureBirthMeanTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            )
                        } else {
                            todo!();
                            //  Process::EcDNAProcess(
                            //      PureBirthMeanEcDNA::new(
                            //          growth,
                            //          iteration,
                            //          distribution,
                            //          verbose,
                            //      )?
                            //      .into(),
                            //  )
                        }
                    }
                    false => {
                        if cli.time {
                            Process::EcDNAProcess(
                                PureBirthTimeEcDNA::new(
                                    0f32,
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            )
                        } else {
                            Process::EcDNAProcess(
                                PureBirthEcDNA::new(
                                    growth,
                                    iteration,
                                    distribution,
                                    verbose,
                                )?
                                .into(),
                            )
                        }
                    }
                }
            }
        };

        let (parallel, runs) = if cli.debug {
            (Parallel::Debug, 1)
        } else if cli.sequential {
            (Parallel::False, cli.runs)
        } else {
            (Parallel::True, cli.runs)
        };

        let path2dir = match cli.subsample.as_ref() {
            Some(samples) => {
                let samples_str: Vec<String> =
                    samples.iter().map(|ele| ele.to_string()).collect();
                cli.path.join(format!(
                    "{}samples{}population",
                    samples_str.join("_"),
                    cli.cells
                ))
            }
            None => cli
                .path
                .join(format!("{}samples{}population", cli.cells, cli.cells)),
        };

        let sampling_options = cli.restart_growth.map(|sampling_options| {
            let restart_growth = match sampling_options {
                RestartGrowthOptions::ContinueFromSubsample => {
                    RestartGrowth::ContinueFromSubsample
                }
                RestartGrowthOptions::ContinueFromPopulation => {
                    RestartGrowth::ContinueFromPopulation
                }
            };
            SamplingOptions {
                sample_at: cli
                    .subsample
                    .expect("Clap has checked due to requires"),
                restart_growth,
            }
        });

        Ok(SimulationOptions {
            simulation: Dynamics { seed: cli.seed, path2dir, verbose },
            parallel,
            processes: vec![process; runs],
            sampling_options,
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

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum RestartGrowthOptions {
    ContinueFromSubsample,
    ContinueFromPopulation,
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
