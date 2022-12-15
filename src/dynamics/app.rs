use anyhow::Context;
use chrono::Utc;
use indicatif::ParallelProgressIterator;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ssa::{
    process::{Process, StopReason},
    NbIndividuals,
};
use std::path::PathBuf;

use crate::{
    run::{CellCulture, Ended, Growth, PatientStudy, Run, Started},
    Simulate, NB_RESTARTS,
};

pub struct Dynamics {
    pub subsampling: Option<Vec<NbIndividuals>>,
    pub save: Option<Vec<NbIndividuals>>,
    pub process: Process,
    pub seed: u64,
    pub path2dir: PathBuf,
    pub runs: usize,
    pub debug: bool,
    pub verbose: u8,
}

impl Dynamics {
    fn run_helper(
        &self,
        run: Run<Started>,
        growth: Growth,
        idx: usize,
        mut j: u64,
    ) -> Run<Ended> {
        let process = self.process.clone();
        let (mut run_ended, mut stop) = run.simulate(0);
        // restart, this can happen when high death rates
        while stop == StopReason::NoIndividualsLeft && j < NB_RESTARTS {
            let seed_run = idx as u64 * NB_RESTARTS + self.seed + j;
            if self.verbose > 1 {
                println!("Restarting with seed {} because all lineages have died out", seed_run);
            }
            let rng = Pcg64Mcg::seed_from_u64(seed_run);

            let run = Run::new(
                idx,
                process.clone(),
                growth.clone(),
                0,
                rng,
                self.verbose,
            );
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
    fn run(self: Box<Self>) -> anyhow::Result<()> {
        println!("{} Starting the simulation", Utc::now());

        let growth = if self.subsampling.is_some() {
            CellCulture.into()
        } else {
            PatientStudy.into()
        };
        let experiments = match growth {
            Growth::CellCulture(_) => self.subsampling.as_ref(),
            Growth::PatientStudy(_) => self.save.as_ref(),
        };

        let simulate_run = |idx| {
            let seed_run = idx as u64 * NB_RESTARTS + self.seed;
            if self.verbose > 0 {
                println!("Seed: {:#?}", seed_run);
            }
            let rng = Pcg64Mcg::seed_from_u64(seed_run);
            let mut run = Run::new(
                idx,
                self.process.clone(),
                growth.clone(),
                0,
                rng,
                self.verbose,
            );

            if run.verbosity > 0 {
                println!("{:#?}", run);
            }

            if let Some(experiments) = experiments {
                for (i, nb_cells) in experiments.iter().enumerate() {
                    if self.verbose > 1 {
                        println!("Subsampling {} with {} cells", i, nb_cells);
                    }
                    let run_ended = self
                        .run_helper(run, growth.clone(), idx, 0)
                        .undersample(nb_cells, i);
                    run_ended
                        .save(&self.path2dir)
                        .with_context(|| {
                            format!(
                                "Cannot save run {} for timepoint {}",
                                idx, i
                            )
                        })
                        .unwrap();
                    run = Run::<Started>::from(run_ended);
                }
            } else {
                if self.verbose > 1 {
                    println!("Nosubsampling");
                }
                // no subsampling no saving
                self.run_helper(run, growth.clone(), idx, 0)
                    .save(&self.path2dir)
                    .with_context(|| format!("Cannot save run {}", idx))
                    .unwrap();
            };
        };

        if self.debug {
            (0..1).into_iter().for_each(simulate_run)
        } else {
            (0..self.runs)
                .into_par_iter()
                .progress_count(self.runs as u64)
                .for_each(simulate_run)
        };

        println!("{} End simulating dynamics", Utc::now(),);

        Ok(())
    }
}
