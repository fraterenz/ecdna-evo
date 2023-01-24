use anyhow::Context;
use ecdna_evo::process::PureBirthNoDynamics;
use ecdna_evo::proliferation::Exponential;
use ecdna_evo::segregation::BinomialSegregation;
use rand::SeedableRng;
use rand_chacha::{self, ChaCha8Rng};
use serde::{ser::SerializeStruct, Serialize, Serializer};
use ssa::abc::{ABCRejection, ABCResult, ABCResultBuilder, Data};
use ssa::{run::Run, NbIndividuals, RandomSampling};
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
        process: PureBirthNoDynamics<Exponential, BinomialSegregation>,
        data: &Data,
        sample_at: Option<NbIndividuals>,
    ) -> anyhow::Result<ABCResultFitness> {
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
        let run = Run::new(idx, process, self.verbose);
        if run.verbosity > 0 {
            println!("{:#?}", run);
        }

        let (mut run_ended, _) = run.simulate(0, &mut rng);

        if let Some(sample_at) = sample_at {
            run_ended.bd_process.random_sample(sample_at, &mut rng);
        }

        let mut builder = ABCResultBuilder::default();
        builder.idx(idx);
        // run data
        let rates = *run_ended.bd_process.get_rates();

        Ok(ABCResultFitness {
            result: ABCRejection::run(
                builder,
                run_ended.bd_process.get_ecdna_distribution(),
                data,
                self.verbose,
            ),
            b0: rates[0],
            b1: rates[1],
        })
    }
}

pub fn save(
    results: Vec<ABCResultFitness>,
    path2folder: &Path,
    verbosity: u8,
) -> anyhow::Result<()> {
    fs::create_dir_all(path2folder)
        .with_context(|| "Cannot create dir".to_string())?;

    let mut abc = path2folder.join("abc");
    abc.set_extension("csv");
    if verbosity > 1 {
        println!("Saving ABC results to {:#?}", abc);
    }
    let mut wtr = csv::Writer::from_path(abc)?;

    for res in results {
        if verbosity > 0 {
            println!(
                "Statistics of run {}: Mean: {}, Freq: {}, Entropy: {}",
                res.result.idx,
                res.result.mean,
                res.result.frequency,
                res.result.entropy
            );
        }
        wtr.serialize(&res).with_context(|| {
            "Cannot serialize the results from ABC inference".to_string()
        })?;
    }
    wtr.flush()?;
    Ok(())
}
