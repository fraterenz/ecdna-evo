use app::Abc;
use chrono::Utc;
use ecdna_evo::{
    abc::{
        ABCRejection, ABCResultBuilder, ABCResultFitness, ABCResultsFitness,
    },
    distribution::EcDNADistribution,
    process::{BirthDeath, EcDNAEvent, PureBirth},
    proliferation::{EcDNADeath, Exponential},
    segregation::Binomial,
};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use sosa::{CurrentState, ReactionRates};

use crate::clap_app::{Cli, Parallel};

mod app;
mod clap_app;

pub struct SimulationOptions {
    simulation: Abc,
    parallel: Parallel,
    rates: Vec<(f32, Option<f32>)>,
    initial_distribution: EcDNADistribution,
    drop_nminus: bool,
    verbose: u8,
}

fn main() {
    let simulation = Cli::build();
    println!("{} Starting the simulation", Utc::now());
    match simulation {
        Ok(cli) => {
            let runs = cli.rates.len();
            std::process::exit({
                let my_closure = |(idx, b1, d)| -> ABCResultFitness {
                    {
                        if let Some(d) = d {
                            let rates = ReactionRates([1f32, b1, d, d]);
                            let initial_state = CurrentState {
                                population: [
                                    *cli.initial_distribution.get_nminus(),
                                    cli.initial_distribution.compute_nplus(),
                                    *cli.initial_distribution.get_nminus(),
                                    cli.initial_distribution.compute_nplus(),
                                ],
                            };

                            let process = BirthDeath::new(
                                cli.initial_distribution.clone(),
                                Exponential {},
                                Binomial,
                                EcDNADeath,
                                cli.verbose,
                            )
                            .expect("Cannot create the process");

                            let mut process: BirthDeath<
                                Exponential,
                                Binomial,
                            > = cli
                                .simulation
                                .run::<BirthDeath<
                                    ecdna_evo::proliferation::Exponential,
                                    ecdna_evo::segregation::Binomial,
                                >, EcDNAEvent, 4>(
                                    idx,
                                    process,
                                    initial_state,
                                    &rates,
                                    &[
                                        EcDNAEvent::ProliferateNMinus,
                                        EcDNAEvent::ProliferateNPlus,
                                        EcDNAEvent::DeathNMinus,
                                        EcDNAEvent::DeathNPlus,
                                    ],
                                )
                                .unwrap();

                            let mut builder = ABCResultBuilder::default();
                            builder.idx(idx);

                            if cli.drop_nminus {
                                process
                                    .get_mut_ecdna_distribution()
                                    .drop_nminus();

                                // the frequency is the number of cells w/o ecDNAs
                                builder.frequency_stat(None);
                            }

                            ABCResultFitness {
                                result: ABCRejection::run(
                                    builder,
                                    process.get_ecdna_distribution(),
                                    &cli.simulation.target,
                                    cli.drop_nminus,
                                    cli.verbose,
                                ),
                                rates: rates.0.to_vec(),
                            }
                        } else {
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
                                Binomial,
                                cli.verbose,
                            )
                            .expect("Cannot create the process");

                            let mut process = cli
                                .simulation
                                .run::<PureBirth<
                                    ecdna_evo::proliferation::Exponential,
                                    ecdna_evo::segregation::Binomial,
                                >, EcDNAEvent, 2>(
                                    idx,
                                    process,
                                    initial_state,
                                    &rates,
                                    &[
                                        EcDNAEvent::ProliferateNMinus,
                                        EcDNAEvent::ProliferateNPlus,
                                    ],
                                )
                                .unwrap();

                            let mut builder = ABCResultBuilder::default();
                            builder.idx(idx);

                            if cli.drop_nminus {
                                process
                                    .get_mut_ecdna_distribution()
                                    .drop_nminus();
                                builder.frequency_stat(None);
                            }

                            ABCResultFitness {
                                result: ABCRejection::run(
                                    builder,
                                    process.get_ecdna_distribution(),
                                    &cli.simulation.target,
                                    cli.drop_nminus,
                                    cli.verbose,
                                ),
                                rates: rates.0.to_vec(),
                            }
                        }
                    }
                };

                let results: Vec<ABCResultFitness> = match cli.parallel {
                    Parallel::Debug | Parallel::False => cli
                        .rates
                        .into_iter()
                        .enumerate()
                        .map(|(idx, (b1, d))| my_closure((idx, b1, d)))
                        .collect(),
                    Parallel::True => cli
                        .rates
                        .into_par_iter()
                        .enumerate()
                        .progress_count(runs as u64)
                        .map(|(idx, (b1, d))| my_closure((idx, b1, d)))
                        .collect(),
                };
                let results = ABCResultsFitness(results);

                if let Err(err) = results.save(
                    &cli.simulation.path2dir,
                    cli.simulation.options.verbosity,
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
