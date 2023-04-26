use ecdna_evo::abc::Data;
use ecdna_evo::distribution::SamplingStrategy;
use ecdna_evo::proliferation::EcDNAProliferation;
use ecdna_evo::segregation::Segregate;
use ecdna_evo::{RandomSampling, ToFile};
use rand::SeedableRng;
use rand_chacha::{self, ChaCha8Rng};
use sosa::{
    simulate, AdvanceStep, CurrentState, NbIndividuals, Options, ReactionRates,
};
use std::fmt::Debug;
use std::path::PathBuf;

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
