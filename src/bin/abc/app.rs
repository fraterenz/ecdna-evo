use ecdna_lib::abc::{ABCRejection, ABCResultBuilder, Data};
use rand::SeedableRng;
use rand_chacha::{self, ChaCha8Rng};
use ssa::{
    ecdna::process::{EcDNAProcess, PureBirthNoDynamics},
    run::Run,
    NbIndividuals, Process, RandomSampling,
};
use std::path::PathBuf;

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
        process: PureBirthNoDynamics,
        data: &Data,
        sample_at: Option<NbIndividuals>,
    ) -> anyhow::Result<()> {
        //! Find the posterior distribution of the fitness coefficient using
        //! ABC on taking `data` as input.
        //! The fitness coefficient is the birth-rate of cells with ecDNAs.
        //!
        //! Perform subsampling when `sample_at` is some.
        if self.verbose > 0 {
            println!("Stream: {:#?}", idx);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        rng.set_stream(idx as u64);
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
                EcDNAProcess::PureBirthNoDynamics(run) => {
                    let mut builder = ABCResultBuilder::default();
                    // run data
                    let rates = *run.get_rates();
                    builder.b0(rates[0]);
                    builder.b1(rates[1]);
                    builder.idx(idx);

                    ABCRejection::run(
                        builder,
                        run.get_ecdna_distribution(),
                        data,
                        self.verbose,
                    )
                    .save(&self.path2dir, self.verbose)
                }
                _ => unreachable!("Cannot perform ABC without abc process"),
            },
        }?;
        Ok(())
    }
}
