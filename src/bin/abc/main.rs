use app::Abc;
use chrono::Utc;
use ecdna_evo::{
    distribution::EcDNADistribution,
    process::{EcDNAEvent, PureBirth},
    proliferation::Exponential,
    segregation::BinomialSegregation,
};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use ssa::{CurrentState, ReactionRates};

use crate::{
    app::{save, ABCResultFitness},
    clap_app::{Cli, Parallel},
};

mod app;
mod clap_app;

pub struct SimulationOptions {
    simulation: Abc,
    parallel: Parallel,
    /// subsample tumour when it has reached this size
    fitness_coefficients: Vec<f32>,
    initial_distribution: EcDNADistribution,
    verbose: u8,
}

fn main() {
    let simulation = Cli::build();
    println!("{} Starting the simulation", Utc::now());
    match simulation {
        Ok(cli) => {
            let runs = cli.fitness_coefficients.len();
            std::process::exit({
                let my_closure = |(idx, b1)| -> ABCResultFitness {
                    {
                        let rates = ReactionRates([1f32, b1]);
                        let initial_state = CurrentState {
                            population: [
                                *cli.initial_distribution.get_nminus(),
                                cli.initial_distribution.compute_nplus(),
                            ],
                        };

                        let process = PureBirth::new(
                            cli.initial_distribution.clone(),
                            Exponential {},
                            BinomialSegregation,
                            cli.verbose,
                        )
                        .expect("Cannot create the process");
                        cli.simulation
                            .run(
                                idx,
                                process,
                                &cli.simulation.target,
                                initial_state,
                                &rates,
                                &[
                                    EcDNAEvent::ProliferateNMinus,
                                    EcDNAEvent::ProliferateNPlus,
                                ],
                            )
                            .unwrap()
                    }
                };

                let results: Vec<ABCResultFitness> = match cli.parallel {
                    Parallel::Debug | Parallel::False => cli
                        .fitness_coefficients
                        .into_iter()
                        .enumerate()
                        .map(|(idx, b1)| my_closure((idx, b1)))
                        .collect(),
                    Parallel::True => cli
                        .fitness_coefficients
                        .into_par_iter()
                        .enumerate()
                        .progress_count(runs as u64)
                        .map(|(idx, b1)| my_closure((idx, b1)))
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
