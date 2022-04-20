//! Events simulated by the birth-death process, also known as reactions in the
//! chemical litterature (see Gillespie 2007).

use crate::{
    process::{BirthDeath, BirthDeath1, BirthDeath2, PureBirth},
    rate::{exprand, GillespieRate},
    NbIndividuals,
};
use enum_dispatch::enum_dispatch;
use rand_pcg::Pcg64Mcg;

/// The time sampled from Gillespie for the next `Event`. This time does not
/// represent actual time but it is relative to the previous simulated `Event`.
pub type GillespieTime = f32;

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum AdvanceRun {
    /// Init the run
    Init,
    /// The event sampled create a new individual from the first population
    Proliferate1,
    /// The event sampled create a new individual from the second population
    Proliferate2,
    /// The event sampled kill an individual from the first population
    Die1,
    /// The event sampled kill an individual from the second population
    Die2,
}

/// The action that will be simulated (reaction in the chemical literature)
/// determined by Gillespie sampling algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// The type of action simulated
    pub kind: AdvanceRun,
    /// The time of appearance of this event (Gillespie time) relative to the
    /// previous simulated `Event`
    pub time: GillespieTime,
}

#[enum_dispatch(BirthDeathProcess)]
pub trait ComputeTimesEvents {
    //! Compute the next reaction times with the associated reactions
    fn compute_times_events(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]);
}

impl ComputeTimesEvents for PureBirth {
    fn compute_times_events(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32, rng), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32, rng), /* proliferation rate for cell
                                                  * w/o ecDNA */
            f32::INFINITY,
            f32::INFINITY,
        ];
        let events = [
            AdvanceRun::Proliferate1,
            AdvanceRun::Proliferate2,
            AdvanceRun::Die1,
            AdvanceRun::Die2,
        ];
        (rates, events)
    }
}

impl ComputeTimesEvents for BirthDeath1 {
    fn compute_times_events(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32, rng), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32, rng), /* proliferation rate for cell
                                                  * w/o ecDNA */
            exprand(self.d1 * pop1 as f32, rng), // death rate for pop1
            f32::INFINITY,                       // death rate for pop2
        ];
        let events = [
            AdvanceRun::Proliferate1,
            AdvanceRun::Proliferate2,
            AdvanceRun::Die1,
            AdvanceRun::Die2,
        ];
        (rates, events)
    }
}

impl ComputeTimesEvents for BirthDeath2 {
    fn compute_times_events(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32, rng), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32, rng), /* proliferation rate for cell
                                                  * w/o ecDNA */
            f32::INFINITY, // death rate for pop1
            exprand(self.d2 * pop2 as f32, rng), // death rate for pop2
        ];
        let events = [
            AdvanceRun::Proliferate1,
            AdvanceRun::Proliferate2,
            AdvanceRun::Die1,
            AdvanceRun::Die2,
        ];
        (rates, events)
    }
}

impl ComputeTimesEvents for BirthDeath {
    fn compute_times_events(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32, rng), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32, rng), /* proliferation rate for cell
                                                  * w/o ecDNA */
            exprand(self.d1 * pop1 as f32, rng), // death rate for pop1
            exprand(self.d2 * pop2 as f32, rng), // death rate for pop2
        ];
        let events = [
            AdvanceRun::Proliferate1,
            AdvanceRun::Proliferate2,
            AdvanceRun::Die1,
            AdvanceRun::Die2,
        ];
        (rates, events)
    }
}

pub fn fast_mean_computation(
    previous_mean: f32,
    event: &AdvanceRun,
    ntot: NbIndividuals,
) -> Option<f32> {
    //! Function allowing the computation of the ecDNA distirbution mean without
    //! traversing the whole vector of the ecDNA distribution.
    //!
    //! There are four cases when the mean can be computed for the next
    //! iteration without traversing the whole vector of state of nplus
    //! cells (n means the total number of cells of the previous iteration):
    //!     1. when the next cell to proliferate is `NMinus`. For this event
    //! the mean of the next iteration can be computed as: previous mean * n
    //! / (n + 1)     2. when the next cell to die is `NMinus` and n
    //! > 1. For this event the mean of the next iteration can be computed
    //! as: previous mean * n / (n - 1)     3. when the next cell to
    //! die is `NMinus` and n == 1. For this event the mean of the next
    //! iteration is simply 0, because no cell will remain since they all
    //! died out     4. when the next cell to die is `NPlus` and n
    //! == 1. For this event the mean of the next iteration is simply 0,
    //! because no cell will remain since they all died out
    //!
    //! To find n of the previous iteration, we can sum the number of cells
    //! in state from the current iteration, ie self.state.len() - SPECIES
    //! to the end.
    match event {
        // when dynamics are present and we start with some initial state with more
        // than one cells, we init the dynamics (and thus the mean if needed) with
        // constant values
        AdvanceRun::Init => Some(previous_mean),
        // The mean will be n_old * mean_old / (n_old + 1)
        AdvanceRun::Proliferate2 => {
            Some(previous_mean * ntot as f32 / (ntot as f32 + 1_f32))
        }

        // If the next event is the death of a NMinus cell there are two possible outcomes based
        // on the number of n cells in the previous iteration: 1. in the previous there was only
        // 1 cell, then put 0 since the next event is death
        AdvanceRun::Die2 => {
            if ntot == 1_u64 {
                Some(0_f32)
            } else {
                // 2. else, precompute the mean
                // The mean will be n_old * mean_old / (n_old - 1)
                Some(previous_mean * ntot as f32 / (ntot as f32 - 1_f32))
            }
        }

        // Cannot easily compute the mean; can do that only when the event is
        // Event::{NMinusProliferates,NMinusDies}.
        AdvanceRun::Proliferate1 => None,

        // If the next event is the death of a NPlus cell there are two possible outcomes based
        // on the number of n cells in the previous iteration: 1. in the previous there was only
        // 1 cell, then put 0 since the next event is death
        AdvanceRun::Die1 => {
            if ntot == 1_u64 {
                Some(0_f32)
            } else {
                // 2. else, we cannot easily precompute the mean
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{process::BirthDeathProcess, rate::Rates, Seed};
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for BirthDeathProcess {
        fn arbitrary(g: &mut Gen) -> BirthDeathProcess {
            BirthDeathProcess::new(Rates::new(
                &[PositiveRate::arbitrary(g).0],
                &[PositiveRate::arbitrary(g).0],
                &[PositiveRate::arbitrary(g).0],
                &[PositiveRate::arbitrary(g).0],
                Seed::new(u64::arbitrary(g)),
            ))
        }
    }

    #[derive(Clone)]
    struct PositiveRate(pub f32);

    impl Arbitrary for PositiveRate {
        fn arbitrary(g: &mut Gen) -> Self {
            let rate = f32::arbitrary(g).abs();
            if rate.is_nan() {
                return PositiveRate::arbitrary(g);
            }
            PositiveRate(rate)
        }
    }

    #[test]
    fn test_fast_mean_computation() {
        let previous_mean = 1f32;
        let nb_individuals = 1f32;
        let next_event = AdvanceRun::Proliferate2;
        assert_eq!(
            fast_mean_computation(
                previous_mean,
                &next_event,
                nb_individuals as u64
            ),
            Some(0.5f32)
        );
        let next_event = AdvanceRun::Proliferate1;
        assert_eq!(
            fast_mean_computation(
                previous_mean,
                &next_event,
                nb_individuals as u64
            ),
            None
        );
    }
}
