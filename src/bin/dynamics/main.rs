use chrono::Utc;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use ssa::Process;

use crate::clap_app::{Cli, Parallel};

mod app;
mod clap_app;

/// The number of max iterations that we max simulate compared to the cells.
/// The max number of iterations will be MAX_ITER * cells.
pub const MAX_ITER: usize = 3;
/// The number of time a run restarts when there are no individuals left because
/// of the high cell death
const NB_RESTARTS: u64 = 30;

/// Run the simulations
pub trait Simulate {
    fn run(&self, idx: usize, process: Process) -> anyhow::Result<()>;
}

pub struct SimulationOptions {
    simulation: Box<dyn Simulate + Sync>,
    parallel: Parallel,
    processes: Vec<Process>,
}

fn main() {
    let simulation = Cli::build();
    println!("{} Starting the simulation", Utc::now());
    match simulation {
        Ok(cli) => {
            std::process::exit({
                let runs = cli.processes.len() as u64;
                match cli.parallel {
                    Parallel::Debug | Parallel::False => {
                        cli.processes.into_iter().enumerate().for_each(
                            |(idx, process)| {
                                cli.simulation.run(idx, process).unwrap()
                            },
                        )
                    }
                    Parallel::True => cli
                        .processes
                        .into_par_iter()
                        .enumerate()
                        .progress_count(runs)
                        .for_each(|(idx, process)| {
                            cli.simulation.run(idx, process).unwrap()
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
