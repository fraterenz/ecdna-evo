use crate::run::{Range, Seed};
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rand_distr::Open01;
use rand_pcg::Pcg64Mcg;
use std::convert::From;
use std::fmt;
use std::ops::{BitXor, Mul};

/// Gillespie rate
type GillespieRate = f32;

/// Rates of the two-type stochastic birth-death process (units [1/N])
#[derive(Debug, Clone)]
pub struct Rates {
    /// Proliferation rate of the cells w/ ecDNA of a stochastic birth-death
    /// process
    pub fitness1: ProliferationRate,
    /// Proliferation rate of the cells w/o ecDNA of a stochastic birth-death
    /// process
    pub fitness2: ProliferationRate,
    /// Death rate of the cells w/ ecDNA of a stochastic birth-death process
    pub death1: DeathRate,
    /// Death rate of the cells w/o ecDNA of a stochastic birth-death process
    pub death2: DeathRate,
}

impl Rates {
    pub fn new(
        f1: &[f32],
        f2: &[f32],
        d1: &[f32],
        d2: &[f32],
        seed: Seed,
    ) -> Self {
        Rates {
            fitness1: ProliferationRate::new(f1, seed),
            fitness2: ProliferationRate::new(f2, seed),
            death1: DeathRate::new(d1, seed),
            death2: DeathRate::new(d2, seed),
        }
    }

    pub fn estimate_max_iter(&self, max_cells: &NbIndividuals) -> usize {
        //! Returns the maximal number of iterations.
        if self.death1.is_zero() || self.death2.is_zero() {
            *max_cells as usize
        } else {
            *max_cells as usize * 3usize
        }
    }
}

/// The case with no selection (fitness coefficients both 1) and cells cannot
/// die
impl Default for Rates {
    fn default() -> Self {
        Rates::new(&[1f32], &[1f32], &[0f32], &[0f32], Seed::default())
    }
}

/// Gillespie rate units 1/N
#[derive(Clone, Debug)]
enum Rate {
    Range(Range<f32>),
    Scalar(f32),
}

impl Rate {
    pub fn new(rates: &[f32], seed: Seed) -> Self {
        match *rates {
            [rate] => Rate::Scalar(rate),
            [min, max] => Rate::Range(Range::new(min, max, seed)),
            _ => {
                panic!(
                    "Cannot create Rate with more than two rates {:#?}",
                    rates
                )
            }
        }
    }
}

impl Default for Rate {
    fn default() -> Self {
        Rate::Scalar(0f32)
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = {
            match self {
                Rate::Range(range) => {
                    format!("{}", range)
                }
                Rate::Scalar(v) => {
                    format!("{}", v)
                }
            }
        };
        write!(f, "{}", s.replace('.', ""))
    }
}

impl Mul for Rate {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match rhs {
            Rate::Scalar(v1) => match self {
                Rate::Scalar(v2) => Rate::Scalar(v1 * v2),
                Rate::Range(_) => panic!("Do not know how to mulitply ranges"),
            },
            Rate::Range(_) => panic!("Do not know how to mulitply ranges"),
        }
    }
}

#[derive(Clone, Debug)]
/// How many times on average there will be proliferation for a stochastic
/// birth-death process
pub struct ProliferationRate(Rate);

/// By default a proliferation rate is 1 (neutral case)
impl Default for ProliferationRate {
    fn default() -> Self {
        ProliferationRate::new(&[1f32], Seed::default())
    }
}

impl ProliferationRate {
    pub fn new(rates: &[f32], seed: Seed) -> Self {
        assert!(
            !rates.iter().any(|val| val < &0f32),
            "ProliferationRate cannot be negative"
        );
        ProliferationRate(Rate::new(rates, seed))
    }
}

#[derive(Clone, Debug, Default)]
/// How many times on average there will be death for a stochastic birth-death
/// process
pub struct DeathRate(Rate);

impl DeathRate {
    pub fn new(rates: &[f32], seed: Seed) -> Self {
        assert!(
            !rates.iter().any(|val| val < &0f32),
            "DeathRate cannot be negative"
        );
        DeathRate(Rate::new(rates, seed))
    }

    pub fn is_zero(&self) -> bool {
        match self.0 {
            Rate::Scalar(rate) => rate != 0f32,
            Rate::Range(_) => panic!("Cannot say if rate is 0 with range"),
        }
    }
}

impl std::str::FromStr for ProliferationRate {
    type Err = std::num::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f32::from_str(s).map(|val| ProliferationRate(Rate::Scalar(val)))
    }
}

impl From<f32> for ProliferationRate {
    fn from(val: f32) -> Self {
        ProliferationRate(Rate::Scalar(val))
    }
}

impl From<ProliferationRate> for GillespieRate {
    fn from(rate: ProliferationRate) -> Self {
        match rate.0 {
            Rate::Scalar(v) => v,
            Rate::Range(mut range) => range.sample_uniformly(),
        }
    }
}

impl From<u64> for ProliferationRate {
    fn from(val: u64) -> Self {
        ProliferationRate::from(val as f32)
    }
}

impl fmt::Display for ProliferationRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ProliferationRate", self.0)
    }
}

impl Mul for ProliferationRate {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl std::str::FromStr for DeathRate {
    type Err = std::num::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f32::from_str(s).map(|val| DeathRate(Rate::Scalar(val)))
    }
}

impl From<f32> for DeathRate {
    fn from(val: f32) -> Self {
        DeathRate(Rate::Scalar(val))
    }
}

impl From<DeathRate> for GillespieRate {
    fn from(rate: DeathRate) -> Self {
        match rate.0 {
            Rate::Scalar(v) => v,
            Rate::Range(mut range) => range.sample_uniformly(),
        }
    }
}

impl From<u64> for DeathRate {
    fn from(val: u64) -> Self {
        DeathRate::from(val as f32)
    }
}

impl fmt::Display for DeathRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}DeathRate", self.0)
    }
}

impl Mul for DeathRate {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

/// Number of individual cells present in the system.
pub type NbIndividuals = u64;
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

#[enum_dispatch(BirthDeathProcess)]
pub trait GetRates {
    //! Get the proliferation and death rates
    fn get_rates(&self) -> [f32; 4];
}

/// Process simulating only birth of the individuals for both populations
#[derive(PartialEq, Clone, Debug)]
pub struct PureBirth {
    r1: f32,
    r2: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the first population
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath1 {
    r1: f32,
    r2: f32,
    d1: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the second population
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath2 {
    r1: f32,
    r2: f32,
    d2: f32,
}

/// Process simulating birth and death of the individuals for both populations
/// (the real birth- death process)
#[derive(PartialEq, Clone, Debug)]
pub struct BirthDeath {
    r1: f32,
    r2: f32,
    d1: f32,
    d2: f32,
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

impl GetRates for PureBirth {
    fn get_rates(&self) -> [f32; 4] {
        //! For `PureBirth` the death coefficients are 0
        [self.r1, self.r2, 0f32, 0f32]
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

impl GetRates for BirthDeath1 {
    fn get_rates(&self) -> [f32; 4] {
        //! For `BirthDeath1` the death coefficient of the cells w/o ecDNA is 0
        [self.r1, self.r2, self.d1, 0f32]
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

impl GetRates for BirthDeath2 {
    fn get_rates(&self) -> [f32; 4] {
        //! For `BirthDeath2` the death coefficient of the cells w/ ecDNA is 0
        [self.r1, self.r2, 0f32, self.d2]
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

impl GetRates for BirthDeath {
    fn get_rates(&self) -> [f32; 4] {
        [self.r1, self.r2, self.d1, self.d2]
    }
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

fn exprand(lambda: GillespieRate, rng: &mut Pcg64Mcg) -> f32 {
    //! Generates a random waiting time using the exponential waiting time with
    //! parameter `lambda` of Poisson StochasticProcess.
    if (lambda - 0_f32).abs() < f32::EPSILON {
        f32::INFINITY
    } else {
        // random number between (0, 1)
        let val: f32 = rng.sample(Open01);
        -(1. - val).ln() / lambda
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
extern crate quickcheck;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[quickcheck]
    fn exprand_same_seed(lambda: f32, seed: u64) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        if lambda == 0f32 {
            exprand(lambda, &mut rng).is_infinite()
        } else if lambda.is_nan() {
            exprand(lambda, &mut rng).is_nan()
        } else {
            let exp1 = exprand(lambda, &mut rng);
            let mut rng = Pcg64Mcg::seed_from_u64(seed);
            let exp2 = exprand(lambda, &mut rng);
            (exp1 - exp2).abs() < f32::EPSILON
        }
    }

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

    #[test]
    fn test_exprand() {
        let mut rng = Pcg64Mcg::seed_from_u64(1u64);
        let lambda: GillespieRate = 0_f32;
        let first = exprand(lambda, &mut rng);
        assert!(first.is_infinite());

        let lambda: GillespieRate = f32::INFINITY;
        let first = exprand(lambda, &mut rng);
        assert!((0f32 - first).abs() < f32::EPSILON);
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
