use anyhow::Context;
use ecdna_lib::abc::{ABCRejection, ABCResult, ABCResultBuilder, Data};
use rand::SeedableRng;
use rand_chacha::{self, ChaCha8Rng};
use serde::{ser::SerializeStruct, Serialize, Serializer};
use ssa::{
    ecdna::process::{EcDNAProcess, PureBirthNoDynamics},
    run::Run,
    NbIndividuals, Process, RandomSampling,
};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ABCResultFitness {
    pub result: ABCResult,
    pub b0: f32,
    pub b1: f32,
}

impl Serialize for ABCResultFitness {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ABCResultFitness", 12)?;
        state.serialize_field("idx", &self.result.idx)?;
        state.serialize_field("mean", &self.result.mean)?;
        state.serialize_field("mean_stat", &self.result.mean_stat)?;
        state.serialize_field("frequency", &self.result.frequency)?;
        state
            .serialize_field("frequency_stat", &self.result.frequency_stat)?;
        state.serialize_field("entropy", &self.result.entropy)?;
        state.serialize_field("entropy_stat", &self.result.entropy_stat)?;
        state.serialize_field("ecdna_stat", &self.result.ecdna_stat)?;
        state.serialize_field("pop_size", &self.result.pop_size)?;
        state.serialize_field("b0", &self.b0)?;
        state.serialize_field("b1", &self.b1)?;
        state.end()
    }
}

impl ABCResultFitness {
    pub fn save(
        &self,
        path2folder: &Path,
        verbosity: u8,
    ) -> anyhow::Result<()> {
        if verbosity > 0 {
            println!(
                "Statistics of the run: Mean: {}, Freq: {}, Entropy: {}",
                self.result.mean, self.result.frequency, self.result.entropy
            );
        }
        fs::create_dir_all(path2folder.join("abc"))
            .with_context(|| "Cannot create dir abc".to_string())?;
        let mut abc =
            path2folder.join("abc").join(self.result.idx.to_string());
        abc.set_extension("csv");
        if verbosity > 1 {
            println!("Saving ABC results to {:#?}", abc);
        }
        let mut wtr = csv::Writer::from_path(abc)?;
        wtr.serialize(&self).with_context(|| {
            "Cannot serialize the results from ABC inference".to_string()
        })?;
        wtr.flush()?;
        Ok(())
    }
}

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
                    builder.idx(idx);
                    // run data
                    let rates = *run.get_rates();

                    ABCResultFitness {
                        result: ABCRejection::run(
                            builder,
                            run.get_ecdna_distribution(),
                            data,
                            self.verbose,
                        ),
                        b0: rates[0],
                        b1: rates[1],
                    }
                    .save(&self.path2dir, self.verbose)
                }
                _ => unreachable!("Cannot perform ABC without abc process"),
            },
        }?;
        Ok(())
    }
}
