//! Type of processes that can be simulated.
//!
//! `PureBirth` etc TODO
use crate::event::{AdvanceRun, ComputeTimesEvents, Event, GillespieTime};
use crate::rate::{GillespieRate, Rate, Rates};
use crate::NbIndividuals;
use enum_dispatch::enum_dispatch;
use rand_pcg::Pcg64Mcg;
use std::ops::BitXor;

/// Process simulating only birth of the individuals for both populations
#[derive(PartialEq, Clone, Debug)]
pub struct PureBirth {
    pub r1: f32,
    pub r2: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the first population
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath1 {
    pub r1: f32,
    pub r2: f32,
    pub d1: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the second population
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath2 {
    pub r1: f32,
    pub r2: f32,
    pub d2: f32,
}

/// Process simulating birth and death of the individuals for both populations
/// (the real birth- death process)
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath {
    pub r1: f32,
    pub r2: f32,
    pub d1: f32,
    pub d2: f32,
}

/// Two-type stochastic birth-death process.
#[enum_dispatch]
#[derive(PartialEq, Clone, Debug)]
pub enum BirthDeathProcess {
    /// Cells cannot die but only proliferate.
    PureBirth,
    /// Only cells of the first type can die, cells of both types can
    /// proliferate.
    BirthDeath1,
    /// Only cells of the second type can die, cells of both types can
    /// proliferate.
    BirthDeath2,
    /// Cells of both types can die and proliferate.
    BirthDeath,
}

impl BirthDeathProcess {
    pub fn new(rates: Rates) -> BirthDeathProcess {
        //! Creates a new stochastic process. This will sample the rates (for
        //! abc).
        //!
        //! Depending on the rates, the stochastic process can either be:
        //!
        //! 1. `PureBirth`, that is `d1` and `d2` are equal to zero: individuals
        //! of both populations can only proliferate (there is no death)
        //!
        //! 2. `BirthDeath1`, that is `d1` is not zero and `d2` is zero:
        //! individuals of the first population can die but individuals of the
        //! second population cannot
        //!
        //! 3. `BirthDeath1`, that is `d1` is zero and `d2` is not zero:
        //! individuals of the first population cannot die but
        //! individuals of the second population can
        //!
        //! 4. `BirthDeath1`, that is `d1` is not zero and `d2` is not zero:
        //! individuals of both populations can proliferate and die

        let f1 = match rates.fitness1.0 {
            Rate::Range(mut range) => range.sample_uniformly(),
            Rate::Scalar(v) => v,
        };

        let f2 = match rates.fitness2.0 {
            Rate::Range(mut range) => range.sample_uniformly(),
            Rate::Scalar(v) => v,
        };

        let d1 = match rates.death1.0 {
            Rate::Range(mut range) => range.sample_uniformly(),
            Rate::Scalar(v) => v,
        };

        let d2 = match rates.death2.0 {
            Rate::Range(mut range) => range.sample_uniformly(),
            Rate::Scalar(v) => v,
        };

        assert!(
            d1 >= 0f32,
            "Found negative death_rate_nplus should be positive!"
        );
        assert!(
            d2 >= 0f32,
            "Found negative death_rate_nminus should be positive!"
        );
        let death_rate1_found = d1 > 0f32;
        let exactly_one_death_rate_found =
            (death_rate1_found).bitxor(d2 > 0_f32);

        if exactly_one_death_rate_found {
            if death_rate1_found {
                let process = BirthDeath1 { r1: f1, r2: f2, d1 };
                process.into()
            } else {
                let process = BirthDeath2 { r1: f1, r2: f2, d2 };
                process.into()
            }
        } else if death_rate1_found {
            let process = BirthDeath { r1: f1, r2: f2, d1, d2 };
            process.into()
        } else {
            let process = PureBirth { r1: f1, r2: f2 };
            process.into()
        }
    }

    pub fn gillespie(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> Event {
        //! Determine the next `Event` using the Gillespie algorithm.
        assert!((pop1 + pop2) > 0u64);
        let (kind, time) = self.next_event(pop1, pop2, rng);
        Event { kind, time }
    }

    fn next_event(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> (AdvanceRun, GillespieTime) {
        let (times, events) = self.compute_times_events(pop1, pop2, rng);

        // Find the event that will occur next, corresponding to the smaller
        // waiting time
        let mut selected_event = 0_usize;
        let mut smaller_waiting_time = times[selected_event];
        for (idx, &waiting_time) in times.iter().enumerate() {
            if waiting_time <= smaller_waiting_time {
                smaller_waiting_time = waiting_time;
                selected_event = idx;
            }
        }
        (events[selected_event], smaller_waiting_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::BirthDeathProcess;
    use rand::SeedableRng;

    #[quickcheck]
    fn next_event_same_seed_same_event(
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        bd: BirthDeathProcess,
        seed: u64,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let (event1, _) = bd.next_event(pop1, pop2, &mut rng);

        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let (event2, _) = bd.next_event(pop1, pop2, &mut rng);

        event1 == event2
    }

    #[quickcheck]
    fn next_event_same_seed_same_time(
        pop1: NbIndividuals,
        pop2: NbIndividuals,
        bd: BirthDeathProcess,
        seed: u64,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let (_, time1) = bd.next_event(pop1, pop2, &mut rng);

        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let (_, time2) = bd.next_event(pop1, pop2, &mut rng);

        time1 == time2
    }
}
