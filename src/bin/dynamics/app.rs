use anyhow::Context;
use ecdna_evo::{distribution::SamplingStrategy, RandomSampling, ToFile};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sosa::{
    simulate, AdvanceStep, CurrentState, NbIndividuals, Options, ReactionRates,
};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

use crate::NB_RESTARTS;

pub struct Dynamics {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub max_cells: NbIndividuals,
    pub options: Options,
}

pub struct Sampling {
    pub at: Vec<NbIndividuals>,
    pub strategy: SamplingStrategy,
}

impl Dynamics {
    pub fn run<P, REACTION, const NB_REACTIONS: usize>(
        &self,
        idx: usize,
        mut process: P,
        initial_state: &mut CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<NB_REACTIONS>,
        possible_reactions: &[REACTION; NB_REACTIONS],
        sampling: &Option<Sampling>,
    ) -> anyhow::Result<()>
    where
        P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>
            + Clone
            + Debug
            + ToFile
            + RandomSampling,
        REACTION: std::fmt::Debug,
    {
        let run_helper =
            |sampling: Option<(NbIndividuals, SamplingStrategy)>,
             path2dir: &Path,
             state: &mut CurrentState<NB_REACTIONS>,
             process: &mut P| {
                let stream = idx as u64 * NB_RESTARTS;
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                rng.set_stream(stream);

                // clone the initial state in case we restart
                let stop_reason = simulate(
                    state,
                    rates,
                    possible_reactions,
                    process,
                    &self.options,
                    &mut rng,
                );

                // restart, this can happen when high death rates
                // TODO: bug, see issue #104 and also biases the results since
                // we are kind of conditionning upon survival
                //
                // while (stop == StopReason::NoIndividualsLeft
                //     || stop == StopReason::AbsorbingStateReached)
                //     && j <= NB_RESTARTS
                // {
                //     // restart process as well
                //     process_copy = process.clone();
                //     let stream = idx as u64 * NB_RESTARTS + self.seed + j;
                //     if self.options.verbosity > 1 {
                //         println!(
                //             "Restarting with stream {} because {:#?}",
                //             stream, stop
                //         );
                //     }
                //     rng.set_stream(stream);

                //     // clone the initial state in case we restart
                //     stop = simulate(
                //         &mut initial_state.clone(),
                //         rates,
                //         possible_reactions,
                //         &mut process_copy,
                //         &self.options,
                //         &mut rng,
                //     );
                //     j += 1;
                // }

                if self.options.verbosity > 1 {
                    println!("stop reason: {:#?}", stop_reason);
                }

                if let Some(sampling) = sampling {
                    if self.options.verbosity > 1 {
                        println!("subsample with {} cells", sampling.0);
                    }
                    process
                        .save(&path2dir.join("before_subsampling"), idx)
                        .with_context(|| {
                            format!(
                                "Cannot save run {} before subsampling",
                                idx
                            )
                        })
                        .unwrap();

                    if self.options.verbosity > 0 {
                        println!("Subsampling");
                    }
                    process.random_sample(&sampling.1, sampling.0, rng);
                    // after subsampling we need to update the state
                    process.update_state(state);
                }
            };

        if let Some(sampling) = sampling {
            let path2dir =
                &self.path2dir.join((sampling.at.len() - 1).to_string());
            for (i, sample_at) in sampling.at.iter().enumerate() {
                // save only at last sample
                let last_sample = i == sampling.at.len() - 1;
                run_helper(
                    Some((*sample_at, sampling.strategy)),
                    path2dir,
                    initial_state,
                    &mut process,
                );
                // happens with birth-death processes
                let no_individuals_left =
                    initial_state.population.iter().sum::<u64>() == 0u64;
                if last_sample || no_individuals_left {
                    process
                        .save(&path2dir.join("after_subsampling"), idx)
                        .with_context(|| {
                            format!(
                                "Cannot save run {} for timepoint {}",
                                idx, i
                            )
                        })
                        .unwrap();
                }
                if no_individuals_left {
                    if self.options.verbosity > 0 {
                        println!("do not subsample, early stopping, due to no cells left");
                    }
                    break;
                }
            }
        } else {
            if self.options.verbosity > 1 {
                println!("Nosubsampling");
            }
            run_helper(None, &self.path2dir, initial_state, &mut process);
            process
                .save(&self.path2dir, idx)
                .with_context(|| format!("Cannot save run {}", idx))
                .unwrap();
        };
        Ok(())
    }
}
