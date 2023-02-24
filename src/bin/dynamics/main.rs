use app::Dynamics;
use chrono::Utc;
use clap_app::{GrowthOptions, Segregation};
use ecdna_evo::{
    distribution::EcDNADistribution, process::PureBirth,
    proliferation::Exponential,
};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ssa::{iteration::CurrentState, rates::ReactionRates, NbIndividuals};

use crate::clap_app::{Cli, Parallel};

mod app;
mod clap_app;

/// The number of max iterations that we max simulate compared to the cells.
/// The max number of iterations will be MAX_ITER * cells.
pub const MAX_ITER: usize = 3;
/// The number of time a run restarts when there are no individuals left because
/// of the high cell death
const NB_RESTARTS: u64 = 30;

pub struct SimulationOptions {
    simulation: Dynamics,
    parallel: Parallel,
    sampling_at: Option<Vec<NbIndividuals>>,
    is_birth_death: bool,
    b0: f32,
    b1: f32,
    d0: f32,
    d1: f32,
    time: bool,
    mean: bool,
    distribution: EcDNADistribution,
    segregation: Segregation,
    growth: GrowthOptions,
    runs: usize,
}

fn main() {
    let app = Cli::build();
    let proliferation = match app.growth {
        GrowthOptions::Constant => todo!(),
        GrowthOptions::Exponential => Exponential {},
    };
    let (process, initial_state, rates) = match app.is_birth_death {
        true => match app.mean {
            true => {
                if app.time {
                    todo!();
                } else {
                    todo!();
                }
            }
            false => {
                if app.time {
                    todo!();
                } else {
                    todo!();
                }
            }
        },
        false => {
            let initial_population = CurrentState {
                population: [
                    *app.distribution.get_nminus(),
                    app.distribution.compute_nplus(),
                ],
            };
            let rates = ReactionRates([app.b0, app.b1]);
            match app.mean {
                true => {
                    if app.time {
                        todo!();
                    } else {
                        todo!();
                    }
                }
                false => {
                    if app.time {
                        todo!();
                    } else {
                        (
                            PureBirth::with_distribution(
                                proliferation,
                                app.segregation,
                                app.distribution,
                                app.simulation.max_iterations,
                                app.simulation.verbose,
                            )
                            .unwrap(),
                            initial_population,
                            rates,
                        )
                    }
                }
            }
        }
    };

    println!("{} Starting the simulation", Utc::now());
    let my_closure = |idx| {
        app.simulation
            .run(
                idx,
                process.clone(),
                initial_state.clone(),
                &rates,
                &app.sampling_at,
            )
            .unwrap();
    };
    std::process::exit({
        match app.parallel {
            Parallel::Debug | Parallel::False => {
                (0..app.runs).into_iter().for_each(|idx| my_closure(idx))
            }
            Parallel::True => (0..app.runs)
                .into_par_iter()
                .progress_count(app.runs as u64)
                .for_each(|idx| my_closure(idx)),
        }
        println!("{} End simulation", Utc::now(),);
        0
    });
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;

    Cli::command().debug_assert()
}
