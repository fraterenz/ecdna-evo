//! The two-type/k-type ecDNA simulation problem modelling the effect of the
//! random segregation on the ecDNA dynamics.
/// The ecDNA dynamics such as the mean in function of time.
pub mod dynamics;
/// The available ecDNA processes.
pub mod process;
/// EcDNA growth models
pub mod proliferation;
/// EcDNA segregation models.
pub mod segregation;

use std::path::Path;

use ecdna_lib::distribution::SamplingStrategy;
pub use ecdna_lib::{abc, distribution, DNACopy};
use rand::Rng;
use ssa::{
    AdvanceStep, CurrentState, NbIndividuals, ReactionRates, SimState,
    StopReason,
};

pub struct IterationsToSimulate {
    pub max_iter: usize,
    pub init_iter: usize,
}

/// The main loop running one realisation of a stochastic process with
/// `NB_REACTIONS` possible `REACTION`s.
pub fn simulate<P, REACTION, const NB_REACTIONS: usize>(
    state: &mut CurrentState<NB_REACTIONS>,
    rates: &ReactionRates<{ NB_REACTIONS }>,
    possible_reactions: &[REACTION; NB_REACTIONS],
    bd_process: &mut P,
    iterations: &IterationsToSimulate,
    verbosity: u8,
    rng: &mut impl Rng,
) -> StopReason
where
    P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>,
    REACTION: std::fmt::Debug,
{
    let mut iter = iterations.init_iter;
    loop {
        let (sim_state, reaction) = bd_process.next_reaction(
            state,
            rates,
            possible_reactions,
            iter,
            iterations.max_iter - 1,
            rng,
        );

        if verbosity > 1 {
            println!("State: {:#?}, reaction: {:#?}", state, reaction);
        }

        match sim_state {
            SimState::Continue => {
                // unwrap is safe since SimState::Continue returns always
                // something (i.e. not None).
                let reaction = reaction.unwrap();

                // update the process according to the reaction
                bd_process.advance_step(reaction, rng);

                // update the state according to the process
                bd_process.update_state(state);
                iter += 1;
            }
            SimState::Stop(reason) => return reason,
        }
        // the absorbing state is when there are no NPlus cells and only NMinus
        // cells.
        if state.population[1] == 0 {
            return StopReason::AbsorbingStateReached;
        }
    }
}

/// Save the results of the simulations.
pub trait ToFile {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()>;
}

/// Undersample a process by randomly reducing the number of individuals.
pub trait RandomSampling {
    fn random_sample(
        &mut self,
        strategy: &SamplingStrategy,
        nb_individuals: NbIndividuals,
        rng: impl Rng,
    );
}

#[cfg(test)]
pub mod test_util {
    use std::{
        collections::HashMap,
        num::{NonZeroU16, NonZeroU8},
    };

    use super::segregation::DNACopySegregating;
    use ecdna_lib::{distribution::EcDNADistribution, DNACopy};
    use quickcheck::{Arbitrary, Gen};

    #[derive(Clone, Debug)]
    struct DNACopyGreaterOne(DNACopy);

    impl Arbitrary for DNACopyGreaterOne {
        fn arbitrary(g: &mut Gen) -> DNACopyGreaterOne {
            let mut copy = NonZeroU16::arbitrary(g);
            if copy == NonZeroU16::new(1).unwrap() {
                copy = NonZeroU16::new(2).unwrap();
            }
            DNACopyGreaterOne(copy)
        }
    }

    #[derive(Clone, Debug)]
    pub struct NonEmptyDistribtionWithNPlusCells(pub EcDNADistribution);

    impl Arbitrary for NonEmptyDistribtionWithNPlusCells {
        fn arbitrary(g: &mut Gen) -> NonEmptyDistribtionWithNPlusCells {
            const MAX_ENTRIES: usize = 500;
            let mut distr = HashMap::with_capacity(MAX_ENTRIES);
            for _ in 0..MAX_ENTRIES {
                let copy = DNACopySegregatingGreatherThanOne::arbitrary(g);
                let cells = NonZeroU8::arbitrary(g).get() as u64;
                distr.insert(u16::from(DNACopy::from(copy.0)), cells);
            }
            let cells = NonZeroU8::arbitrary(g).get() as u64;
            distr.insert(0, cells);
            let distr = EcDNADistribution::new(distr, 1000);
            NonEmptyDistribtionWithNPlusCells(distr)
        }
    }
    #[derive(Clone, Debug)]
    pub struct DNACopySegregatingGreatherThanOne(pub DNACopySegregating);

    impl Arbitrary for DNACopySegregatingGreatherThanOne {
        fn arbitrary(g: &mut Gen) -> DNACopySegregatingGreatherThanOne {
            let mut copy =
                DNACopy::new(NonZeroU8::arbitrary(g).get() as u16).unwrap();
            if copy == DNACopy::new(1).unwrap() || copy.get() % 2 == 1 {
                copy = DNACopy::new(2).unwrap();
            }
            DNACopySegregatingGreatherThanOne(
                DNACopySegregating::try_from(copy).unwrap(),
            )
        }
    }

    #[derive(Debug, Clone)]
    pub struct TestSegregation(pub SegregationTypes);

    #[derive(Debug, Clone)]
    pub enum SegregationTypes {
        Deterministic,
        BinomialNoUneven,
        BinomialNoNminus,
        BinomialSegregation,
    }

    impl Arbitrary for TestSegregation {
        fn arbitrary(g: &mut Gen) -> Self {
            let t = g.choose(&[0, 1, 2, 3]).unwrap();
            match t {
                0 => TestSegregation(SegregationTypes::Deterministic),
                1 => TestSegregation(SegregationTypes::BinomialNoUneven),
                2 => TestSegregation(SegregationTypes::BinomialNoNminus),
                3 => TestSegregation(SegregationTypes::BinomialSegregation),
                _ => unreachable!(),
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct NPlusVec(pub Vec<DNACopy>);

    impl Arbitrary for NPlusVec {
        fn arbitrary(g: &mut Gen) -> Self {
            NPlusVec(
                (0..4u16 * (u8::MAX as u16))
                    .map(|_| NonZeroU16::arbitrary(g))
                    .collect(),
            )
        }
    }
}
