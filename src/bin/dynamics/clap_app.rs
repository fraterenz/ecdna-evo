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
    NbIndividuals, Process,
};
use std::{collections::HashMap, path::PathBuf};

use crate::{app::Dynamics, SimulationOptions, MAX_ITER};

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
    /// The number of cells kept for each subsampling
    #[arg(
        long, value_name = "CELLS",
         num_args = 0..,
         value_enum,
         conflicts_with = "save",
         long_help = "The number of cells kept for each subsampling. Samples are taken as soon as the tumour reaches `cells` cells. Then, the tumour restarts growing from these subsample."
        )
    ]
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
}

impl Cli {
    pub fn build() -> anyhow::Result<SimulationOptions> {
        let cli = Cli::parse();

        if cli.save.is_some() {
            todo!()
        }
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

        let parallel = if cli.debug {
            Parallel::Debug
        } else if cli.sequential {
            Parallel::False
        } else {
            Parallel::True
        };

        Ok(SimulationOptions {
            simulation: Box::new(Dynamics {
                subsampling: cli.subsampling,
                seed: cli.seed,
                path2dir: cli.path,
                save: cli.save,
                verbose,
            }),
            parallel,
            processes: vec![process; cli.runs],
        })
    }
}
//
//     #[command(
//         arg_required_else_help = true,
//         group(
//             ArgGroup::new("input")
//             .required(true)
//             .args(["mean", "data", "frequency", "entropy"])
//             )
//         )
//     ]
//             Commands::Abc {
//                 b0,
//                 b1,
//                 runs,
//                 cells,
//                 seed,
//                 debug,
//                 path,
//                 mean,
//                 frequency,
//                 entropy,
//                 data,
//                 verbose,
//             } => {
//                 let parallel =
//                     if debug { Parallel::Debug } else { Parallel::True };
//
//                 // we assume fixed initial pop for now, starting with one cell
//                 // with 1 ecDNA
//                 let initial_population = [0, 1];
//                 let distribution = EcDNADistribution::new(
//                     HashMap::from([(0, 0), (1, 1)]),
//                     cells as usize,
//                 );
//
//                 // load data
//                 let input = match data {
//                     Some(path2file) => Input::Distribution(
//                         EcDNADistribution::load(&path2file)?,
//                     ),
//                     None => Input::Summaries(Summaries::new(
//                         mean, frequency, entropy,
//                     )),
//                 };
//                 let data = Data::from_input(input);
//
//                 // create proceses to simulate
//                 let mut rng = Pcg64Mcg::seed_from_u64(seed);
//                 // we assume pure birth only, hence 2
//                 let iterations: Vec<Iteration<2>> = (0..runs)
//                     .map(|_| {
//                         // sample at random a fitness, assuming b1[0] > b1[1]
//                         let b1 = Uniform::new(&b1[0], &b1[1]).sample(&mut rng);
//                         Iteration::new(
//                             [b0, b1],
//                             initial_population,
//                             cells,
//                             cells as usize,
//                         )
//                     })
//                     .collect();
//                 let processes = iterations
//                     .into_iter()
//                     .map(|iteration| {
//                         Process::EcDNAProcess(
//                             ABC::new(
//                                 EcDNAGrowth::Exponential(Exponential {
//                                     segregation: Segregation::Random(
//                                         RandomSegregation::BinomialSegregation(
//                                             BinomialSegregation,
//                                         ),
//                                     ),
//                                 })
//                                 .into(),
//                                 iteration,
//                                 distribution.clone(),
//                                 &data,
//                                 verbose,
//                             )
//                             .expect("Cannot create the ABC simulation")
//                             .into(),
//                         )
//                     })
//                     .collect();
//                 Ok(SimulationOptions {
//                     simulation: Box::new(Abc {
//                         seed,
//                         path2dir: path,
//                         verbose,
//                         target: data,
//                     }),
//                     parallel,
//                     processes,
//                 })
//             }
//         }
//     }
// }
//
// #[derive(Debug, Subcommand)]
// enum Commands {
//     #[command(arg_required_else_help = true)]
//     /// Infer the fitness coefficient (birth-rate of cells with ecDNAs) in a
//     /// pure-birth stochastic process.
//     Abc {
//         /// Proliferation rate of the cells without ecDNAs (wild-type)
//         #[arg(long, value_name = "RATE", default_value_t = 1.)]
//         b0: f32,
//         /// proliferation rate of the cells with ecdnas.
//         /// when a range is specified, abc samples a rate randomly from this range
//         #[arg(long, value_name = "rate", num_args=0..=2)]
//         b1: Vec<f32>,
//         #[arg(short, long, default_value_t = 100, conflicts_with = "debug")]
//         /// number of independent runs used to recover the posterior distribution
//         /// of the fitness coefficient
//         runs: usize,
//         /// number of cells to simulate
//         #[arg(
//             long,
//             short,
//             default_value_t = 100000,
//             conflicts_with = "debug"
//         )]
//         cells: NbIndividuals,
//         /// seed for reproducibility
//         #[arg(long, default_value_t = 26)]
//         seed: u64,
//         /// triggers debug mode: max verbosity and 1 sequential simulation
//         #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
//         debug: bool,
//         #[arg(value_name = "DIR", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.is_dir() { Ok(path_b) } else { Err("Cannot find dir") }} ) ]
//         path: PathBuf,
//         /// use the mean to infer the posterior distribution
//         #[arg(long, conflicts_with = "data")]
//         mean: Option<f32>,
//         /// use the ferquency to infer the posterior distribution
//         #[arg(long, conflicts_with = "data")]
//         frequency: Option<f32>,
//         /// use the entropy to infer the posterior distribution
//         #[arg(long, conflicts_with = "data")]
//         entropy: Option<f32>,
//         /// the ecdna distribution used to infer the posterior distribution.
//         /// when present, abc will compute from this file the mean, the entropy as well as the frequency.
//         #[arg(long, value_name = "FILE", value_parser = |path: &str| { let path_b = PathBuf::from(path); if path_b.extension() == Some(std::ffi::OsStr::new("json")) { Ok(path_b) } else { Err("must be json file: extension must be .json)") }} ) ]
//         data: Option<PathBuf>,
//         #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
//         verbose: u8,
//     },
// }

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
