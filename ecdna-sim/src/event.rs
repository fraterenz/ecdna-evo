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

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
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
