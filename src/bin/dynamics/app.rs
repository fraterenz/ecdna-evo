use anyhow::Context;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ssa::{
    iteration::{CurrentState, StopReason},
    rates::ReactionRates,
    run::{Ended, Run, Started},
    NbIndividuals, Process,
};
use std::path::PathBuf;

use crate::NB_RESTARTS;

pub struct Dynamics {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub max_cells: NbIndividuals,
    pub max_iterations: usize,
    pub verbose: u8,
}

impl Dynamics {
    pub fn run<const NB_REACTIONS: usize, P>(
        &self,
        idx: usize,
        process: P,
        initial_state: CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<NB_REACTIONS>,
        sampling_at: &Option<Vec<NbIndividuals>>,
    ) -> anyhow::Result<()>
    where
        P: Process<NB_REACTIONS>,
    {
        let run_helper = |run: Run<Started>,
                          sample_at: Option<u64>|
         -> (Run<Ended>, P) {
            let mut process_copy = process.clone();
            if run.verbosity > 0 {
                println!("{:#?}", run);
            }

            let stream = idx as u64 * NB_RESTARTS;
            let mut j = 0;
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
            rng.set_stream(stream);

            // clone the initial state in case we restart
            let (mut run_ended, mut stop) = run.simulate(
                0,
                &mut initial_state.clone(),
                rates,
                &mut process_copy,
                &mut rng,
            );

            // restart, this can happen when high death rates
            while stop == StopReason::NoIndividualsLeft && j < NB_RESTARTS {
                // restart process as well
                process_copy = process.clone();
                let seed_run = idx as u64 * NB_RESTARTS + self.seed + j;
                if self.verbose > 1 {
                    println!("Restarting with seed {} because all lineages have died out", seed_run);
                }

                let mut rng = ChaCha8Rng::seed_from_u64(seed_run);

                let run = Run::new(
                    idx,
                    self.max_cells,
                    self.max_iterations,
                    self.verbose,
                );
                // clone the initial state in case we restart
                (run_ended, stop) = run.simulate(
                    0,
                    &mut initial_state.clone(),
                    rates,
                    &mut process_copy,
                    &mut rng,
                );
                j += 1;
            }

            if self.verbose > 0 {
                println!("{} restarts", j);
            }

            if let Some(sample_at) = sample_at {
                run_ended
                    .save(
                        &process_copy,
                        &self.path2dir.join("before_subsampling"),
                    )
                    .with_context(|| {
                        format!("Cannot save run {} before subsampling", idx)
                    })
                    .unwrap();
                process_copy.random_sample(sample_at, &mut rng);
            }
            (run_ended, process_copy)
        };
        let mut run =
            Run::new(idx, self.max_cells, self.max_iterations, self.verbose);

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
                let (run_ended, process) = run_helper(run, Some(*sample_at));
                if last_sample {
                    run_ended
                        .save(&process, &path2dir.join("after_subsampling"))
                        .with_context(|| {
                            format!(
                                "Cannot save run {} for timepoint {}",
                                idx, i
                            )
                        })
                        .unwrap();
                }
                run = run_ended.into();
            }
        } else {
            if self.verbose > 1 {
                println!("Nosubsampling");
            }
            // no subsampling no saving
            let (run_ended, process) = run_helper(run, None);
            run_ended
                .save(&process, &self.path2dir)
                .with_context(|| format!("Cannot save run {}", idx))
                .unwrap();
        };
        Ok(())
    }
}
