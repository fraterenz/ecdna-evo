use app::Dynamics;
use chrono::Utc;
use clap_app::{GrowthOptions, Segregation};
use ecdna_evo::{
    distribution::EcDNADistribution,
    process::{
        BirthDeath, BirthDeathMean, BirthDeathMeanTime,
        BirthDeathMeanTimeVariance, BirthDeathMeanTimeVarianceEntropy,
        BirthDeathNMinusNPlus, BirthDeathNMinusNPlusTime, EcDNAEvent,
        PureBirth, PureBirthMean, PureBirthMeanTime, PureBirthNMinusNPlus,
        PureBirthNMinusNPlusTime,
    },
    proliferation::{EcDNADeath, Exponential},
};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use sosa::{CurrentState, ReactionRates};

use crate::{
    app::Sampling,
    clap_app::{BirthDeathType, Cli, Parallel, ProcessType, PureBirthType},
};

mod app;
mod clap_app;

/// The number of max iterations that we max simulate compared to the cells.
/// The max number of iterations will be MAX_ITER * cells.
pub const MAX_ITER: usize = 10;
/// The number of time a run restarts when there are no individuals left because
/// of the high cell death
const NB_RESTARTS: u64 = 30;

pub struct SimulationOptions {
    simulation: Dynamics,
    parallel: Parallel,
    sampling: Option<Sampling>,
    process_type: ProcessType,
    b0: f32,
    b1: f32,
    d0: f32,
    d1: f32,
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

    println!("{} Starting the simulation", Utc::now());
    let my_closure = |idx| match app.process_type {
        ProcessType::PureBirth(p_type) => {
            let mut initial_state = CurrentState {
                population: [
                    *app.distribution.get_nminus(),
                    app.distribution.compute_nplus(),
                ],
            };
            let rates = ReactionRates([app.b0, app.b1]);
            let reactions =
                [EcDNAEvent::ProliferateNMinus, EcDNAEvent::ProliferateNPlus];
            match p_type {
                PureBirthType::NMinusNPlus => app
                    .simulation
                    .run(
                        idx,
                        PureBirthNMinusNPlus::with_distribution(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                PureBirthType::NMinusNPlusTime => app
                    .simulation
                    .run(
                        idx,
                        PureBirthNMinusNPlusTime::with_distribution(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            0.,
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                PureBirthType::PureBirth => app
                    .simulation
                    .run(
                        idx,
                        PureBirth::new(
                            app.distribution.clone(),
                            proliferation,
                            app.segregation,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                PureBirthType::Mean => app
                    .simulation
                    .run(
                        idx,
                        PureBirthMean::new(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                PureBirthType::MeanTime => app
                    .simulation
                    .run(
                        idx,
                        PureBirthMeanTime::new(
                            0.,
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
            };
        }
        ProcessType::BirthDeath(p_type) => {
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
            match p_type {
                BirthDeathType::BirthDeath => app
                    .simulation
                    .run(
                        idx,
                        BirthDeath::new(
                            app.distribution.clone(),
                            proliferation,
                            app.segregation,
                            EcDNADeath,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::NMinusNPlus => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathNMinusNPlus::with_distribution(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::NMinusNPlusTime => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathNMinusNPlusTime::with_distribution(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            0.,
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::Mean => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathMean::new(
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::MeanTime => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathMeanTime::new(
                            0.,
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::MeanTimeVariance => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathMeanTimeVariance::new(
                            0.,
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
                BirthDeathType::MeanTimeVarianceEntropy => app
                    .simulation
                    .run(
                        idx,
                        BirthDeathMeanTimeVarianceEntropy::new(
                            0.,
                            proliferation,
                            app.segregation,
                            app.distribution.clone(),
                            app.simulation.options.max_iter,
                            app.simulation.options.verbosity,
                        )
                        .unwrap(),
                        &mut initial_state,
                        &rates,
                        &reactions,
                        &app.sampling,
                    )
                    .unwrap(),
            }
        }
    };
    std::process::exit({
        match app.parallel {
            Parallel::Debug | Parallel::False => {
                (0..app.runs).for_each(my_closure)
            }
            Parallel::True => (0..app.runs)
                .into_par_iter()
                .progress_count(app.runs as u64)
                .for_each(my_closure),
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
