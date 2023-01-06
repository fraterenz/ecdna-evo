use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use ssa::{
    ecdna::process::ABC, run::Run, NbIndividuals, Process, RandomSampling,
};
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
        sample_at: Option<NbIndividuals>,
    ) -> anyhow::Result<()> {
        //! Find the posterior distribution of the fitness coefficient using
        //! ABC on taking `data` as input.
        //! The fitness coefficient is the birth-rate of cells with ecDNAs.
        //!
        //! Perform subsampling when `sample_at` is some.
        let seed_run = idx as u64 + self.seed;
        if self.verbose > 0 {
            println!("Seed: {:#?}", seed_run);
        }
        let mut rng = Pcg64Mcg::seed_from_u64(seed_run);
        let run =
            Run::new(idx, Process::EcDNAProcess(process.into()), self.verbose);
        if run.verbosity > 0 {
            println!("{:#?}", run);
        }

        let (mut run_ended, _) = run.simulate(0, &mut rng);

        if let Some(sample_at) = sample_at {
            run_ended.bd_process.random_sample(sample_at, &mut rng);
        }

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
