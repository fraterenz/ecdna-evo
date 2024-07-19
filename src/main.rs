use std::{collections::VecDeque, path::PathBuf};

use chrono::Utc;
use clap_app::{GrowthOptions, Segregation};
use ecdna_evo::{
    create_filename_birth_death, create_filename_pure_birth,
    distribution::EcDNADistribution,
    process::{save, BirthDeath, EcDNAEvent, PureBirth},
    proliferation::{CellDeath, Exponential},
    SavingOptions, Snapshot,
};
use indicatif::ParallelProgressIterator;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use sosa::{simulate, CurrentState, Options, ReactionRates};

use crate::clap_app::{Cli, Parallel, ProcessType};

mod clap_app;

/// The number of max iterations to simulate.
pub const MAX_ITER: usize = 1_000_000_000;
/// The number of cells to simulate.
pub const MAX_CELLS: u64 = 1_000_000_000;

#[derive(Debug)]
pub struct SimulationOptions {
    parallel: Parallel,
    process_type: ProcessType,
    b0: f32,
    b1: f32,
    d0: f32,
    d1: f32,
    distribution: EcDNADistribution,
    segregation: Segregation,
    growth: GrowthOptions,
    runs: usize,
    seed: u64,
    path2dir: PathBuf,
    options: Options,
    snapshots: VecDeque<Snapshot>,
}

fn main() {
    let app = Cli::build().expect("cannot construct the app");
    let proliferation = match app.growth {
        GrowthOptions::Constant => todo!(),
        GrowthOptions::Exponential => Exponential {},
    };

    println!("{} Starting the simulation", Utc::now());

    let run_simulations = |idx| {
        let stream = idx as u64;
        let mut rng = ChaCha8Rng::seed_from_u64(app.seed);
        rng.set_stream(stream);
        let (stop_reason, final_state, final_time) = match app.process_type {
            ProcessType::PureBirth => {
                let mut initial_state = CurrentState {
                    population: [
                        *app.distribution.get_nminus(),
                        app.distribution.compute_nplus(),
                    ],
                };
                let rates = ReactionRates([app.b0, app.b1]);
                let reactions = [
                    EcDNAEvent::ProliferateNMinus,
                    EcDNAEvent::ProliferateNPlus,
                ];
                let filename = create_filename_pure_birth(&rates.0, idx);

                let mut process = PureBirth::new(
                    app.distribution.clone(),
                    proliferation,
                    app.segregation,
                    0.,
                    SavingOptions {
                        snapshots: app.snapshots.clone(),
                        path2dir: app.path2dir.clone(),
                        filename,
                    },
                    app.options.verbosity,
                )
                .unwrap();

                if app.options.verbosity > 1 {
                    println!("{:#?}", process);
                }

                let stop_reason = simulate(
                    &mut initial_state,
                    &rates,
                    &reactions,
                    &mut process,
                    &app.options,
                    &mut rng,
                );
                save(
                    &process.path2dir,
                    &process.filename,
                    process.time,
                    process.get_ecdna_distribution(),
                    process.verbosity,
                )
                .expect(
                    "cannot save the ecDNA distribution at the end of the sim",
                );
                (
                    stop_reason,
                    [initial_state.population[0], initial_state.population[1]],
                    process.time,
                )
            }
            ProcessType::BirthDeath => {
                let mut initial_state = CurrentState {
                    population: [
                        *app.distribution.get_nminus(),
                        app.distribution.compute_nplus(),
                        *app.distribution.get_nminus(),
                        app.distribution.compute_nplus(),
                    ],
                };
                let rates = ReactionRates([app.b0, app.b1, app.d0, app.d1]);
                let reactions = [
                    EcDNAEvent::ProliferateNMinus,
                    EcDNAEvent::ProliferateNPlus,
                    EcDNAEvent::DeathNMinus,
                    EcDNAEvent::DeathNPlus,
                ];
                let filename = create_filename_birth_death(&rates.0, idx);

                let mut process = BirthDeath::new(
                    app.distribution.clone(),
                    proliferation,
                    app.segregation,
                    CellDeath,
                    0.,
                    SavingOptions {
                        snapshots: app.snapshots.clone(),
                        path2dir: app.path2dir.clone(),
                        filename,
                    },
                    app.options.verbosity,
                )
                .unwrap();
                if app.options.verbosity > 1 {
                    println!("{:#?}", process);
                }

                let stop_reason = simulate(
                    &mut initial_state,
                    &rates,
                    &reactions,
                    &mut process,
                    &app.options,
                    &mut rng,
                );
                save(
                    &process.path2dir,
                    &process.filename,
                    process.time,
                    process.get_ecdna_distribution(),
                    process.verbosity,
                )
                .expect(
                    "cannot save the ecDNA distribution at the end of the sim",
                );
                (
                    stop_reason,
                    [initial_state.population[0], initial_state.population[1]],
                    process.time,
                )
            }
        };
        if app.options.verbosity > 0 {
            println!(
                "stop reason: {:#?}\nnminus, nplus: {:#?}\ntime: {}",
                stop_reason, final_state, final_time
            );
        }
    };
    std::process::exit({
        // start from seed for the array job
        let start = (app.seed * 10) as usize;
        let start_end = start..app.runs + start;

        match app.parallel {
            Parallel::Debug | Parallel::False => {
                (start_end).for_each(run_simulations)
            }
            Parallel::True => (start_end)
                .into_par_iter()
                .progress_count(app.runs as u64)
                .for_each(run_simulations),
        }
        println!("{} End simulation", Utc::now());
        0
    });
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;

    Cli::command().debug_assert()
}
