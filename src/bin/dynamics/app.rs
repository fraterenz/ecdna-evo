use anyhow::Context;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ssa::{
    iteration::StopReason,
    run::{Ended, Run, Started},
    NbIndividuals, Process, RandomSampling,
};
use std::path::{Path, PathBuf};

use crate::NB_RESTARTS;

pub struct Dynamics {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub verbose: u8,
}

impl Dynamics {
    fn run_helper(
        &self,
        run: Run<Started>,
        idx: usize,
        mut j: u64,
        stream: u64,
        sampling_at: Option<NbIndividuals>,
        path2dir: Option<&Path>,
    ) -> Run<Ended> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        rng.set_stream(stream);
        let process = run.bd_process.clone();
        let (mut run_ended, mut stop) = run.simulate(0, &mut rng);
        // restart, this can happen when high death rates
        while stop == StopReason::NoIndividualsLeft && j < NB_RESTARTS {
            let seed_run = idx as u64 * NB_RESTARTS + self.seed + j;
            if self.verbose > 1 {
                println!("Restarting with seed {} because all lineages have died out", seed_run);
            }
            let mut rng = ChaCha8Rng::seed_from_u64(seed_run);

            // clone the process because we restart with new simulation with
            // fresh data
            let run = Run::new(idx, process.clone(), self.verbose);
            (run_ended, stop) = run.simulate(0, &mut rng);
            j += 1;
        }
        if self.verbose > 0 {
            println!("{} restarts", j);
        }

        if let Some(sample_at) = sampling_at {
            if let Some(path2dir) = path2dir {
                run_ended
                    .save(&path2dir.join("before_subsampling"))
                    .with_context(|| {
                        format!("Cannot save run {} before subsampling", idx)
                    })
                    .unwrap();
                run_ended.bd_process.random_sample(sample_at, &mut rng);
            }
        }
        run_ended
    }

    pub fn run(
        &self,
        idx: usize,
        process: Process,
        sampling_at: &Option<Vec<NbIndividuals>>,
    ) -> anyhow::Result<()> {
        let stream = idx as u64 * NB_RESTARTS;
        if self.verbose > 0 {
            println!("Stream: {:#?}", stream);
        }
        let mut run = Run::new(idx, process, self.verbose);

        if run.verbosity > 0 {
            println!("{:#?}", run);
        }

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
                let run_ended = self.run_helper(
                    run,
                    idx,
                    0,
                    stream,
                    Some(*sample_at),
                    if last_sample { Some(path2dir) } else { None },
                );
                if last_sample {
                    run_ended
                        .save(&path2dir.join("after_subsampling"))
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
            self.run_helper(run, idx, 0, stream, None, None)
                .save(&self.path2dir)
                .with_context(|| format!("Cannot save run {}", idx))
                .unwrap();
        };
        Ok(())
    }
}
