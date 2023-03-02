use anyhow::Context;
use clap::{ArgAction, Parser, ValueEnum};
use ecdna_evo::{
    distribution::EcDNADistribution,
    segregation::{
        BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
        Deterministic, Segregate,
    },
    IterationsToSimulate,
};
use rand::Rng;
use ssa::NbIndividuals;
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
    /// Triggers debug mode: max verbosity, 1 sequential simulation and 10 cells
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    debug: bool,
    /// Run sequentially each run instead of using rayon for parallelisation
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false, conflicts_with = "debug")]
    sequential: bool,
    /// Whether to track over simulations the evolution of the cells with and
    /// without any ecDNAs.
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    nplus_nminus: bool,
    /// Whether to track over simulations the evolution of the gillespie
    /// time, that is the waiting time for each reaction.
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    time: bool,
    /// Whether to track over simulations the evolution of the mean ecDNA
    /// copies in the tumour population
    #[arg(short, long, action = ArgAction::SetTrue, default_value_t = false)]
    mean: bool,
    /// The number of cells kept after subsampling.
    #[arg(long, num_args = 0.., value_name = "CELLS")]
    subsample: Option<Vec<NbIndividuals>>,
    /// Path to store the results of the simulations
    #[arg(
        value_name = "DIR",
        // value_parser = |path: &str| {
        //     let path_b = PathBuf::from(path);
        //     if path_b.is_dir() { Ok(path_b) } else { Err("Cannot find dir") }
        // }
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
    pub fn build() -> SimulationOptions {
        let cli = Cli::parse();

        let segregation = cli.segregation.into();

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
        let cells = if cli.debug { 10 } else { cli.cells };
        let iterations = if is_birth_death {
            MAX_ITER * cells as usize
        } else {
            cells as usize
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
        let verbose = if cli.debug { u8::MAX } else { cli.verbose };

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
                    "{}samples{}population_{}b0_{}b1_{}d0_{}d1",
                    samples_str.join("_"),
                    cells,
                    cli.b0.to_string().replace('.', ""),
                    cli.b1.to_string().replace('.', ""),
                    d0.to_string().replace('.', ""),
                    d1.to_string().replace('.', ""),
                ))
            }
            None => cli.path.join(format!(
                "{}samples{}population_{}b0_{}b1_{}d0_{}d1",
                cells,
                cells,
                cli.b0.to_string().replace('.', ""),
                cli.b1.to_string().replace('.', ""),
                d0.to_string().replace('.', ""),
                d1.to_string().replace('.', ""),
            )),
        };

        let process_type = {
            match is_birth_death {
                true => match cli.mean {
                    true => {
                        if cli.time {
                            todo!();
                        } else {
                            todo!();
                        }
                    }
                    false => {
                        if cli.time {
                            todo!();
                        } else if cli.nplus_nminus {
                            ProcessType::BirthDeath(
                                BirthDeathType::BirthDeathNMinusNPlus,
                            )
                        } else {
                            ProcessType::BirthDeath(BirthDeathType::BirthDeath)
                        }
                    }
                },
                false => match cli.mean {
                    true => {
                        if cli.time {
                            todo!();
                        } else {
                            todo!();
                        }
                    }
                    false => {
                        if cli.time {
                            todo!();
                        } else if cli.nplus_nminus {
                            ProcessType::PureBirth(
                                PureBirthType::PureBirthNMinusNPlus,
                            )
                        } else {
                            ProcessType::PureBirth(PureBirthType::PureBirth)
                        }
                    }
                },
            }
        };
        SimulationOptions {
            simulation: Dynamics {
                seed: cli.seed,
                max_cells: cells,
                iterations: IterationsToSimulate {
                    max_iter: iterations,
                    init_iter: 0,
                },
                path2dir,
                verbose,
            },
            process_type,
            b0: cli.b0,
            b1: cli.b1,
            d0,
            d1,
            distribution,
            parallel,
            segregation,
            sampling_at: cli.subsample,
            growth: cli.growth,
            runs,
        }
    }
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum SegregationOptions {
    Deterministic,
    BinomialNoUneven,
    BinomialSegregation,
    BinomialNoNminus,
}

#[derive(Debug, Clone, Copy)]
pub enum Segregation {
    Deterministic(Deterministic),
    BinomialNoUneven(BinomialNoUneven),
    BinomialNoNminus(BinomialNoNminus),
    BinomialSegregation(BinomialSegregation),
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
            Self::BinomialSegregation(s) => {
                s.ecdna_segregation(copies, rng, verbosity)
            }
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
                Segregation::BinomialNoUneven(BinomialNoUneven(
                    BinomialSegregation,
                ))
            }
            SegregationOptions::BinomialNoNminus => {
                Segregation::BinomialNoNminus(BinomialNoNminus(
                    BinomialSegregation,
                ))
            }
            SegregationOptions::BinomialSegregation => {
                Segregation::BinomialSegregation(BinomialSegregation)
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

#[derive(Clone, Copy)]
pub enum ProcessType {
    PureBirth(PureBirthType),
    BirthDeath(BirthDeathType),
}

#[derive(Clone, Copy)]
pub enum PureBirthType {
    PureBirth,
    PureBirthNMinusNPlus,
}

#[derive(Clone, Copy)]
pub enum BirthDeathType {
    BirthDeath,
    BirthDeathNMinusNPlus,
}
