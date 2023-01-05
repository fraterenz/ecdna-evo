use anyhow::Context;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use ssa::{
    iteration::StopReason,
    run::{Ended, Run, Started},
    NbIndividuals, Process,
};
use std::path::PathBuf;

use crate::{Simulate, NB_RESTARTS};

pub struct Dynamics {
    pub subsampling: Option<Vec<NbIndividuals>>,
    pub save: Option<Vec<NbIndividuals>>,
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
    ) -> Run<Ended> {
        let process = run.bd_process.clone();
        let (mut run_ended, mut stop) = run.simulate(0);
        // restart, this can happen when high death rates
        while stop == StopReason::NoIndividualsLeft && j < NB_RESTARTS {
            let seed_run = idx as u64 * NB_RESTARTS + self.seed + j;
            if self.verbose > 1 {
                println!("Restarting with seed {} because all lineages have died out", seed_run);
            }
            let rng = Pcg64Mcg::seed_from_u64(seed_run);

            // clone the process because we restart with new simulation with
            // fresh data
            let run = Run::new(idx, process.clone(), rng, self.verbose);
            (run_ended, stop) = run.simulate(0);
            j += 1;
        }
        if self.verbose > 0 {
            println!("{} restarts", j);
        }
        run_ended
    }
}

impl Simulate for Dynamics {
    fn run(&self, idx: usize, process: Process) -> anyhow::Result<()> {
        let seed_run = idx as u64 * NB_RESTARTS + self.seed;
        if self.verbose > 0 {
            println!("Seed: {:#?}", seed_run);
        }
        let rng = Pcg64Mcg::seed_from_u64(seed_run);
        let mut run = Run::new(idx, process, rng, self.verbose);

        if run.verbosity > 0 {
            println!("{:#?}", run);
        }

        if let Some(experiments) = self.subsampling.as_ref() {
            for (i, nb_cells) in experiments.iter().enumerate() {
                if self.verbose > 1 {
                    println!("Subsampling {} with {} cells", i, nb_cells);
                }
                let run_ended =
                    self.run_helper(run, idx, 0).undersample(nb_cells, i);
                run_ended
                    .save(&self.path2dir)
                    .with_context(|| {
                        format!("Cannot save run {} for timepoint {}", idx, i)
                    })
                    .unwrap();
                run = Run::<Started>::from(run_ended);
            }
        } else {
            if self.verbose > 1 {
                println!("Nosubsampling");
            }
            // no subsampling no saving
            self.run_helper(run, idx, 0)
                .save(&self.path2dir)
                .with_context(|| format!("Cannot save run {}", idx))
                .unwrap();
        };
        Ok(())
    }
}
