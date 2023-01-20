use app::Abc;
use chrono::Utc;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use ssa::{ecdna::process::PureBirthNoDynamics, NbIndividuals};

use crate::{
    app::{save, ABCResultFitness},
    clap_app::{Cli, Parallel},
};

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
                let results: Vec<ABCResultFitness> = match cli.parallel {
                    Parallel::Debug | Parallel::False => cli
                        .processes
                        .into_iter()
                        .enumerate()
                        .map(|(idx, process)| {
                            cli.simulation
                                .run(
                                    idx,
                                    process,
                                    &cli.simulation.target,
                                    cli.subsample,
                                )
                                .unwrap()
                        })
                        .collect(),
                    Parallel::True => cli
                        .processes
                        .into_par_iter()
                        .enumerate()
                        .progress_count(runs)
                        .map(|(idx, process)| {
                            cli.simulation
                                .run(
                                    idx,
                                    process,
                                    &cli.simulation.target,
                                    cli.subsample,
                                )
                                .unwrap()
                        })
                        .collect(),
                };
                if let Err(err) = save(
                    results,
                    &cli.simulation.path2dir,
                    cli.simulation.verbose,
                ) {
                    eprintln!(
                        "{} Error while saving the results into {:#?}, {}",
                        Utc::now(),
                        &cli.simulation.path2dir,
                        err
                    );
                    std::process::exit(1);
                };
                println!(
                    "{} Saving the results into {:#?}",
                    Utc::now(),
                    &cli.simulation.path2dir
                );
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
