use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use ssa::{ecdna::process::ABC, run::Run, Process};
use std::path::PathBuf;

use crate::abc::{ABCRejection, Data};

pub struct Abc {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub verbose: u8,
    pub target: Data,
}

impl Abc {
    pub fn run(
        &self,
        idx: usize,
        process: ABC,
        data: &Data,
    ) -> anyhow::Result<()> {
        //! Find the posterior distribution of the fitness coefficient using
        //! ABC on taking `data` as input.
        //! The fitness coefficient is the birth-rate of cells with ecDNAs.
        let seed_run = idx as u64 + self.seed;
        if self.verbose > 0 {
            println!("Seed: {:#?}", seed_run);
        }
        let rng = Pcg64Mcg::seed_from_u64(seed_run);
        let run = Run::new(
            idx,
            Process::EcDNAProcess(process.into()),
            rng,
            self.verbose,
        );

        if run.verbosity > 0 {
            println!("{:#?}", run);
        }
        let (run_ended, _) = run.simulate(0);
        match run_ended.bd_process {
            Process::EcDNAProcess(process) => match process {
                ssa::ecdna::process::EcDNAProcess::ABC(run) => {
                    ABCRejection::run(&run, data, idx).save(
                        &self.path2dir,
                        idx,
                        self.verbose,
                    )
                }
                _ => unreachable!("Cannot perform ABC without abc process"),
            },
        }?;
        Ok(())
    }
}
