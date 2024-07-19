use anyhow::{ensure, Context};
use clap::{ArgAction, Parser, ValueEnum};
use ecdna_evo::{
    distribution::EcDNADistribution,
    segregation::{
        Binomial, BinomialNoNminus, BinomialNoUneven, Deterministic, Segregate,
    },
    Snapshot,
};
use rand::Rng;
use sosa::{IterTime, NbIndividuals, Options};
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
};

use crate::{SimulationOptions, MAX_ITER};

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
    /// Number of years to simulate
    #[arg(long, short, default_value_t = 5, conflicts_with = "debug")]
    years: NbIndividuals,
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
    #[arg(long, requires = "subsamples", value_delimiter = ',', require_equals = true, num_args = 0..)]
    /// Snapshots to take to save the simulation, requires `subsamples`.
    ///
    /// The combination of `snapshots` with `subsamples` gives four different
    /// behaviours:
    ///
    /// 1. when `snapshots.len() = 1` and `subsamples.len() = 1`: subsample once with the number of cells corresponding to `snapshots[0]`
    ///
    /// 2. when `snapshots.len() > 1` and `subsamples.len() = 1`: for every `s` in `snapshots`, subsample with the number of cells corresponding to `snapshots[0]`
    ///
    /// 3. when `snapshots.len() = 1` and `subsamples.len() > 1`: for every `c` in `subsamples`, subsample once with the number of cells corresponding to `c`
    ///
    /// 4. when `snapshots.len() > 1` and `subsamples.len() > 1`: for every pair `(s, c)` in `snapshots.zip(subsamples)`, subsample at time `s` with `c` cells
    snapshots: Option<Vec<f32>>,
    /// Number of cells to subsample before saving the measurements, leave
    /// empty when no subsample is needed.
    /// If subsampling is performed, the measurements of the whole population
    /// will also be saved.
    ///
    /// See help for `snapshots` for more details.
    #[arg(long, requires = "snapshots", num_args = 0.., value_delimiter = ',', require_equals = true)]
    subsamples: Option<Vec<usize>>,
    #[arg(short, long, action = clap::ArgAction::Count, conflicts_with = "debug", default_value_t = 0)]
    verbosity: u8,
}

fn build_snapshots_from_time(n_snapshots: usize, time: f32) -> Vec<f32> {
    let dx = time / ((n_snapshots - 1) as f32);
    let mut x = vec![0.; n_snapshots];
    for i in 1..n_snapshots - 1 {
        x[i] = x[i - 1] + dx;
    }

    x.shrink_to_fit();
    x[n_snapshots - 1] = time;
    x
}

impl Cli {
    pub fn build() -> anyhow::Result<SimulationOptions> {
        let cli = Cli::parse();

        let (years, verbosity, parallel, runs) = if cli.debug {
            (2, u8::MAX, Parallel::Debug, 1)
        } else if cli.sequential {
            (cli.years, cli.verbosity, Parallel::False, cli.runs)
        } else {
            (cli.years, cli.verbosity, Parallel::True, cli.runs)
        };

        let mut snapshots = match (cli.subsamples, cli.snapshots) {
            (Some(sub), Some(snap)) => {
                match (&sub[..], &snap[..]) {
                    // subsample `unique_sub` once at `unique_snap` time
                    ([unique_sub], [unique_snap]) => VecDeque::from_iter(
                        [(unique_sub, unique_snap)].into_iter().map(
                            |(&cells2sample, &time)| Snapshot {
                                cells2sample,
                                time,
                            },
                        ),
                    ),
                    // subsample `unique_sub` at several `snaps` times
                    ([unique_sub], snaps) => VecDeque::from_iter(
                        snaps.iter().zip(std::iter::repeat(unique_sub)).map(
                            |(&time, &cells2sample)| Snapshot {
                                cells2sample,
                                time,
                            },
                        ),
                    ),
                    // subsample with different cells `unique_sub` once at `unique_snap` time
                    (subs, [unique_snap]) => VecDeque::from_iter(
                        subs.iter().zip(std::iter::repeat(unique_snap)).map(
                            |(&cells2sample, &time)| Snapshot {
                                cells2sample,
                                time,
                            },
                        ),
                    ),
                    // subsample with many cells `subs` at specific `snaps`
                    (subs, snaps) => {
                        ensure!(
                            subs.len() == snaps.len(),
                            "the lenght of snapshots do not match the lenght of subsamples"
                        );
                        VecDeque::from_iter(subs.iter().zip(snaps).map(
                            |(&cells2sample, &time)| Snapshot {
                                cells2sample,
                                time,
                            },
                        ))
                    }
                }
            }
            (None, None) => VecDeque::from_iter(
                build_snapshots_from_time(
                    10usize,
                    if years > 1 { years as f32 - 1. } else { 0.9 },
                )
                .into_iter()
                .map(|t| Snapshot { cells2sample: 10, time: t }),
            ),
            _ => unreachable!(),
        };

        snapshots.make_contiguous();
        snapshots
            .as_mut_slices()
            .0
            .sort_by(|s1, s2| s1.time.partial_cmp(&s2.time).unwrap());

        ensure!(
            snapshots.iter().all(|s| s.time < cli.years as f32),
            "times to take `snapshots` must be smaller than total `years`"
        );

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
                max_cells: 100_000_000,
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
