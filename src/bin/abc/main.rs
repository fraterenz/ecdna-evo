use app::Abc;
use chrono::Utc;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use ssa::{ecdna::process::PureBirthNoDynamics, NbIndividuals};

use crate::clap_app::{Cli, Parallel};

/// Perform the ABC to infer the fitness coefficient from the data.
mod abc;
mod app;
mod clap_app;

pub struct SimulationOptions {
    simulation: Abc,
    parallel: Parallel,
    processes: Vec<PureBirthNoDynamics>,
    /// subsample tumour when it has reached this size
    subsample: Option<NbIndividuals>,
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
                                cli.simulation
                                    .run(
                                        idx,
                                        process,
                                        &cli.simulation.target,
                                        cli.subsample,
                                    )
                                    .unwrap()
                            },
                        )
                    }
                    Parallel::True => cli
                        .processes
                        .into_par_iter()
                        .enumerate()
                        .progress_count(runs)
                        .for_each(|(idx, process)| {
                            cli.simulation
                                .run(
                                    idx,
                                    process,
                                    &cli.simulation.target,
                                    cli.subsample,
                                )
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
