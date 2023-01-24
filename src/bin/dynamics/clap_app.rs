use anyhow::Context;
use clap::{ArgAction, Parser, ValueEnum};
use ecdna_evo::{
    process::PureBirth,
    proliferation::Exponential,
    segregation::{
        BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
        Deterministic, Segregate,
    },
};
use ssa::{
    distribution::EcDNADistribution,
    iteration::{Iterate, Iteration},
    NbIndividuals, Process, RandomSampling, ToFile,
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

        let proliferation = match cli.growth {
            GrowthOptions::Constant => todo!(),
            GrowthOptions::Exponential => Exponential {},
        };

        let cells = if cli.debug { 100 } else { cli.cells };
        let verbose = if cli.debug { u8::MAX } else { cli.verbose };

        let process = match is_birth_death {
            true => {
                todo!();
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
                            todo!();
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
                            todo!();
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
                            todo!();
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
                            todo!();
                        } else {
                            EcDNAProcess::PureBirthExp(
                                PureBirth::new(
                                    proliferation,
                                    segregation,
                                    iteration,
                                    distribution,
                                    verbose,
                                )
                                .unwrap(),
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

        Ok(SimulationOptions {
            simulation: Dynamics { seed: cli.seed, path2dir, verbose },
            parallel,
            processes: vec![process; runs],
            sampling_at: cli.subsample,
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

#[derive(Debug, Clone)]
pub enum EcDNAProcess {
    PureBirthExp(PureBirth<Exponential, Segregation>),
}

impl Process for EcDNAProcess {}
impl ToFile for EcDNAProcess {
    fn save(
        &self,
        path2dir: &std::path::Path,
        id: usize,
    ) -> anyhow::Result<()> {
        match self {
            Self::PureBirthExp(p) => p.save(path2dir, id),
        }
    }
}
impl RandomSampling for EcDNAProcess {
    fn random_sample(
        &mut self,
        nb_individuals: NbIndividuals,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) {
        match self {
            Self::PureBirthExp(p) => p.random_sample(nb_individuals, rng),
        }
    }
}

impl Iterate for EcDNAProcess {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> (ssa::iteration::SimState, Option<ssa::iteration::NextReaction>) {
        match self {
            Self::PureBirthExp(p) => p.next_reaction(iter, rng),
        }
    }
    fn update_process(
        &mut self,
        reaction: ssa::iteration::NextReaction,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) {
        match self {
            Self::PureBirthExp(p) => p.update_process(reaction, rng),
        }
    }
    fn update_iteration(&mut self) {
        match self {
            Self::PureBirthExp(p) => p.update_iteration(),
        }
    }
}

#[derive(Debug, Clone)]
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
        rng: &mut rand_chacha::ChaCha8Rng,
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
enum GrowthOptions {
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
