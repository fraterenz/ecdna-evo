use anyhow::Context;
use ecdna_evo::{simulate, IterationsToSimulate, RandomSampling, ToFile};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ssa::{
    AdvanceStep, CurrentState, NbIndividuals, ReactionRates, StopReason,
};
use std::{fmt::Debug, path::PathBuf};

use crate::NB_RESTARTS;

pub struct Dynamics {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub max_cells: NbIndividuals,
    pub iterations: IterationsToSimulate,
    pub verbose: u8,
}

impl Dynamics {
    pub fn run<P, REACTION, const NB_REACTIONS: usize>(
        &self,
        idx: usize,
        process: P,
        initial_state: CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<NB_REACTIONS>,
        possible_reactions: &[REACTION; NB_REACTIONS],
        sampling_at: &Option<Vec<NbIndividuals>>,
    ) -> anyhow::Result<()>
    where
        P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>
            + Clone
            + Debug
            + ToFile
            + RandomSampling,
        REACTION: std::fmt::Debug,
    {
        let run_helper = |sample_at: Option<u64>| -> P {
            let mut process_copy = process.clone();

            let stream = idx as u64 * NB_RESTARTS;
            let mut j = 1;
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
            rng.set_stream(stream);

            // clone the initial state in case we restart
            let mut stop = simulate(
                &mut initial_state.clone(),
                rates,
                possible_reactions,
                &mut process_copy,
                &self.iterations,
                self.verbose,
                &mut rng,
            );

            // restart, this can happen when high death rates
            while (stop == StopReason::NoIndividualsLeft
                || stop == StopReason::AbsorbingStateReached)
                && j <= NB_RESTARTS
            {
                // restart process as well
                process_copy = process.clone();
                let stream = idx as u64 * NB_RESTARTS + self.seed + j;
                if self.verbose > 1 {
                    println!(
                        "Restarting with stream {} because {:#?}",
                        stream, stop
                    );
                }
                rng.set_stream(stream);

                // clone the initial state in case we restart
                stop = simulate(
                    &mut initial_state.clone(),
                    rates,
                    possible_reactions,
                    &mut process_copy,
                    &self.iterations,
                    self.verbose,
                    &mut rng,
                );
                j += 1;
            }

            if self.verbose > 0 {
                println!("{} restarts", j);
            }

            if let Some(sample_at) = sample_at {
                process_copy
                    .save(&self.path2dir.join("before_subsampling"), idx)
                    .with_context(|| {
                        format!("Cannot save run {} before subsampling", idx)
                    })
                    .unwrap();
                process_copy.random_sample(sample_at, &mut rng);
            }
            process_copy
        };

        if let Some(sampling_at) = sampling_at {
            if self.verbose > 0 {
                println!("Subsampling");
            }
            let path2dir =
                &self.path2dir.join((sampling_at.len() - 1).to_string());
            for (i, sample_at) in sampling_at.iter().enumerate() {
                if self.verbose > 1 {
                    println!("{}-th subsample with {} cells", i, sample_at);
                }
                // save only at last sample
                let last_sample = i == sampling_at.len() - 1;
                let process = run_helper(Some(*sample_at));
                if last_sample {
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
                todo!();
                // run = run_ended.into();
            }
        } else {
            if self.verbose > 1 {
                println!("Nosubsampling");
            }
            // no subsampling no saving
            let process = run_helper(None);
            process
                .save(&self.path2dir, idx)
                .with_context(|| format!("Cannot save run {}", idx))
                .unwrap();
        };
        Ok(())
    }
}
