use chrono::Utc;
use enum_dispatch::enum_dispatch;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ssa::run::Growth;
use ssa::Process;

use crate::clap_app::{Cli, Parallel};
use crate::dynamics::app::Dynamics;

mod clap_app;
mod dynamics;

/// The number of max iterations that we max simulate compared to the cells.
/// The max number of iterations will be MAX_ITER * cells.
pub const MAX_ITER: usize = 3;
/// The number of time a run restarts when there are no individuals left because
/// of the high cell death
const NB_RESTARTS: u64 = 30;

/// Run the simulations
#[enum_dispatch]
pub trait Simulate {
    fn run(
        &self,
        idx: usize,
        process: Process,
        growth: Growth,
    ) -> anyhow::Result<()>;
}

pub struct SimulationOptions {
    simulation: Simulation,
    parallel: Parallel,
    runs: usize,
    growth: Growth,
    process: Process,
}

#[enum_dispatch(Simulate)]
enum Simulation {
    Dynamics,
    // Abc,
}

fn main() {
    let simulation = Cli::build();
    println!("{} Starting the simulation", Utc::now());
    match simulation {
        Ok(cli) => {
            std::process::exit({
                match cli.parallel {
                    Parallel::Debug => {
                        cli.simulation.run(0, cli.process, cli.growth).unwrap()
                    }
                    Parallel::False => {
                        (0..cli.runs).into_iter().for_each(|idx| {
                            cli.simulation
                                .run(idx, cli.process.clone(), cli.growth)
                                .unwrap()
                        })
                    }
                    Parallel::True => (0..cli.runs)
                        .into_par_iter()
                        .progress_count(cli.runs as u64)
                        .for_each(|idx| {
                            cli.simulation
                                .run(idx, cli.process.clone(), cli.growth)
                                .unwrap()
                        }),
                }
                println!("{} End simulation", Utc::now(),);
                0
            });
        }
        Err(err) => {
            eprintln!(
                "{} Error while building the cli: {:?}",
                Utc::now(),
                err
            );
            std::process::exit(1);
        }
    }
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;

    Cli::command().debug_assert()
}
