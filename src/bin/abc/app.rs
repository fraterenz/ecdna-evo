use anyhow::Context;
use ecdna_evo::abc::{ABCResult, Data};
use ecdna_evo::distribution::SamplingStrategy;
use ecdna_evo::proliferation::EcDNAProliferation;
use ecdna_evo::segregation::Segregate;
use ecdna_evo::{RandomSampling, ToFile};
use rand::SeedableRng;
use rand_chacha::{self, ChaCha8Rng};
use serde::{ser::SerializeStruct, Serialize, Serializer};
use sosa::{
    simulate, AdvanceStep, CurrentState, NbIndividuals, Options, ReactionRates,
};
use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ABCResultFitness {
    pub result: ABCResult,
    pub rates: Vec<f32>,
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
        state.serialize_field("b0", &self.rates[0])?;
        state.serialize_field("b1", &self.rates[1])?;
        if self.rates.len() > 2 {
            state.serialize_field("d0", &self.rates[2])?;
            state.serialize_field("d1", &self.rates[3])?;
            if self.rates.len() > 4 {
                unreachable!()
            }
        } else {
            state.serialize_field("d0", &0f32)?;
            state.serialize_field("d1", &0f32)?;
        }

        state.end()
    }
}

pub struct Abc {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub options: Options,
    pub target: Data,
    pub sample_at: Option<NbIndividuals>,
}

impl Abc {
    pub fn run<P, REACTION, const NB_REACTIONS: usize, Proliferation, S>(
        &self,
        idx: usize,
        mut process: P,
        mut initial_state: CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<NB_REACTIONS>,
        possible_reactions: &[REACTION; NB_REACTIONS],
    ) -> anyhow::Result<P>
    where
        P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>
            + Clone
            + Debug
            + ToFile
            + RandomSampling,
        Proliferation: EcDNAProliferation,
        S: Segregate,
        REACTION: std::fmt::Debug,
    {
        //! Find the posterior distribution of the fitness coefficient and
        //! optionally the posterior of the death-rate using ABC on `data`.
        //!
        //! We can run abc on both pure-birth or birth-death process, for the
        //! latter we assume that cells with and without ecDNAs have the same
        //! death-rate.
        //!
        //! Perform subsampling when `sample_at` is some and then run abc on
        //! this subsample.
        if self.options.verbosity > 0 {
            println!("Stream: {:#?}", idx);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        rng.set_stream(idx as u64);

        let stop_reason = simulate(
            &mut initial_state,
            rates,
            possible_reactions,
            &mut process,
            &self.options,
            &mut rng,
        );

        if self.options.verbosity > 1 {
            println!("Stopped simulation because {:#?}", stop_reason);
        }

        if let Some(sample_at) = self.sample_at {
            process.random_sample(&SamplingStrategy::Uniform, sample_at, rng);
        };

        Ok(process)
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
