use enum_dispatch::enum_dispatch;
use rand::distributions::{Open01, Uniform};
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::Distribution;
use std::convert::From;
use std::fmt;
use std::ops::{BitXor, Mul};

/// Gillespie rate
type GillespieRate = f32;

/// Rates of the two-type stochastic birth-death process (units [1/N])
#[derive(Debug, PartialEq)]
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
    pub fn new(f1: &[f32], f2: &[f32], d1: &[f32], d2: &[f32]) -> Self {
        Rates {
            fitness1: ProliferationRate::new(f1),
            fitness2: ProliferationRate::new(f2),
            death1: DeathRate::new(d1),
            death2: DeathRate::new(d2),
        }
    }
}

/// The case with no selection (fitness coefficients both 1) and cells cannot
/// die
impl Default for Rates {
    fn default() -> Self {
        Rates::new(&[1f32], &[1f32], &[0f32], &[0f32])
    }
}

/// Gillespie rate units 1/N
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
enum Rate {
    Range(Range),
    Scalar(f32),
}

impl Rate {
    pub fn new(rates: &[f32]) -> Self {
        match *rates {
            [rate] => Rate::Scalar(rate),
            [min, max] => Rate::Range(Range::new(min, max)),
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
                    format!("{}_{}", range.min, range.max)
                }
                Rate::Scalar(v) => {
                    format!("{}", v)
                }
            }
        };
        write!(f, "{}", s.replace(".", ""))
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

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
struct Range {
    min: f32,
    max: f32,
}

impl Default for Range {
    fn default() -> Self {
        Range { min: 0f32, max: 1f32 }
    }
}

impl Range {
    pub fn new(min: f32, max: f32) -> Range {
        if min >= max {
            panic!("Found min {} greater than max {}", min, max)
        }
        Range { min, max }
    }

    fn sample_uniformly(&self, rng: &mut SmallRng) -> f32 {
        Uniform::new(self.min, self.max).sample(rng)
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
/// How many times on average there will be proliferation for a stochastic
/// birth-death process
pub struct ProliferationRate(Rate);

/// By default a proliferation rate is 1 (neutral case)
impl Default for ProliferationRate {
    fn default() -> Self {
        ProliferationRate::new(&[1f32])
    }
}

impl ProliferationRate {
    pub fn new(rates: &[f32]) -> Self {
        assert!(
            !rates.iter().any(|val| val < &0f32),
            "ProliferationRate cannot be negative"
        );
        ProliferationRate(Rate::new(rates))
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Default)]
/// How many times on average there will be death for a stochastic birth-death
/// process
pub struct DeathRate(Rate);

impl DeathRate {
    pub fn new(rates: &[f32]) -> Self {
        assert!(
            !rates.iter().any(|val| val < &0f32),
            "DeathRate cannot be negative"
        );
        DeathRate(Rate::new(rates))
    }
}

impl std::str::FromStr for ProliferationRate {
    type Err = std::num::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f32::from_str(s).map(|val| ProliferationRate { 0: Rate::Scalar(val) })
    }
}

impl From<f32> for ProliferationRate {
    fn from(val: f32) -> Self {
        ProliferationRate(Rate::Scalar(val))
    }
}

impl From<ProliferationRate> for GillespieRate {
    fn from(rate: ProliferationRate) -> Self {
        let mut rng = SmallRng::from_entropy();
        match rate.0 {
            Rate::Scalar(v) => v,
            Rate::Range(range) => range.sample_uniformly(&mut rng),
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
        Self { 0: self.0 * rhs.0 }
    }
}

impl std::str::FromStr for DeathRate {
    type Err = std::num::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f32::from_str(s).map(|val| DeathRate { 0: Rate::Scalar(val) })
    }
}

impl From<f32> for DeathRate {
    fn from(val: f32) -> Self {
        DeathRate(Rate::Scalar(val))
    }
}

impl From<DeathRate> for GillespieRate {
    fn from(rate: DeathRate) -> Self {
        let mut rng = SmallRng::from_entropy();
        match rate.0 {
            Rate::Scalar(v) => v,
            Rate::Range(range) => range.sample_uniformly(&mut rng),
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
        Self { 0: self.0 * rhs.0 }
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
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]);
}

#[enum_dispatch(BirthDeathProcess)]
pub trait GetRates {
    //! Get the proliferation and death rates
    fn get_rates(&self) -> [f32; 4];
}

/// Process simulating only birth of the individuals for both populations
#[derive(PartialEq, Debug)]
pub struct PureBirth {
    r1: f32,
    r2: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the first population
#[derive(PartialEq, Debug)]
pub struct BirthDeath1 {
    r1: f32,
    r2: f32,
    d1: f32,
}

/// Process simulating birth of the individuals for both populations and death
/// only for the second population
#[derive(PartialEq, Debug)]
pub struct BirthDeath2 {
    r1: f32,
    r2: f32,
    d2: f32,
}

/// Process simulating birth and death of the individuals for both populations
/// (the real birth- death process)
#[derive(PartialEq, Debug)]
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
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32), /* proliferation rate for cell
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
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32), /* proliferation rate for cell
                                             * w/o ecDNA */
            exprand(self.d1 * pop1 as f32), // death rate for pop1
            f32::INFINITY,                  // death rate for pop2
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
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32), /* proliferation rate for cell
                                             * w/o ecDNA */
            f32::INFINITY,                  // death rate for pop1
            exprand(self.d2 * pop2 as f32), // death rate for pop2
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
    ) -> ([GillespieRate; 4], [AdvanceRun; 4]) {
        let rates = [
            exprand(self.r1 * pop1 as f32), /* proliferation rate for pop1 */
            exprand(self.r2 * pop2 as f32), /* proliferation rate for cell
                                             * w/o ecDNA */
            exprand(self.d1 * pop1 as f32), // death rate for pop1
            exprand(self.d2 * pop2 as f32), // death rate for pop2
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
#[derive(PartialEq, Debug)]
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
    pub fn new(rates: &Rates) -> BirthDeathProcess {
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
        let mut rng = SmallRng::from_entropy();

        let f1 = match rates.fitness1.0 {
            Rate::Range(range) => range.sample_uniformly(&mut rng),
            Rate::Scalar(v) => v,
        };

        let f2 = match rates.fitness2.0 {
            Rate::Range(range) => range.sample_uniformly(&mut rng),
            Rate::Scalar(v) => v,
        };

        let d1 = match rates.death1.0 {
            Rate::Range(range) => range.sample_uniformly(&mut rng),
            Rate::Scalar(v) => v,
        };

        let d2 = match rates.death2.0 {
            Rate::Range(range) => range.sample_uniformly(&mut rng),
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
    ) -> Event {
        //! Determine the next `Event` using the Gillespie algorithm.
        assert!((pop1 + pop2) > 0u64);
        let (kind, time) = self.next_event(pop1, pop2);
        Event { kind, time }
    }

    fn next_event(
        &self,
        pop1: NbIndividuals,
        pop2: NbIndividuals,
    ) -> (AdvanceRun, GillespieTime) {
        let (times, events) = self.compute_times_events(pop1, pop2);

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

fn exprand(lambda: GillespieRate) -> f32 {
    //! Generates a random waiting time using the exponential waiting time with
    //! parameter `lambda` of Poisson StochasticProcess.
    if (lambda - 0_f32).abs() < f32::EPSILON {
        f32::INFINITY
    } else {
        // random number between (0, 1)
        let val: f32 = thread_rng().sample(Open01);
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
        AdvanceRun::Init => panic!("Cannot compute mean with Init event"),
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

    #[test]
    fn test_exprand() {
        let lambda: GillespieRate = 0.3;
        let first = exprand(lambda);
        let second = exprand(lambda);
        assert!((first - second).abs() > f32::EPSILON);

        let lambda: GillespieRate = 0_f32;
        let first = exprand(lambda);
        assert!(first.is_infinite());

        let lambda: GillespieRate = f32::INFINITY;
        let first = exprand(lambda);
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

    // -------------------------------------------------------------------------
    // TEST ONE ITERATION OF GILLESPIE ALGORITHM
    #[test]
    #[should_panic]
    fn test_gillespie_panics_with_no_individuals_lefth() {
        let my_process =
            BirthDeathProcess::new(&Rates::new(&[1.], &[1.], &[0.], &[0.]));
        my_process.gillespie(0, 0);
    }

    #[test]
    fn test_gillespie_returns_proliferate1() {
        let my_process =
            BirthDeathProcess::new(&Rates::new(&[1.], &[0.], &[0.], &[0.]));
        assert_eq!(AdvanceRun::Proliferate1, my_process.gillespie(1, 2).kind);
    }

    #[test]
    fn test_gillespie_returns_highly_probably_proliferate1() {
        let my_process = BirthDeathProcess::new(&Rates::new(
            &[10000.],
            &[0.],
            &[0.01],
            &[0.],
        ));
        assert_eq!(AdvanceRun::Proliferate1, my_process.gillespie(1, 2).kind);
    }

    #[test]
    fn test_gillespie_returns_highly_probably_proliferate2() {
        let my_process =
            BirthDeathProcess::new(&Rates::new(&[0.], &[1.], &[0.0], &[0.]));
        assert_eq!(AdvanceRun::Proliferate2, my_process.gillespie(1, 2).kind);
    }

    #[test]
    fn test_gillespie_returns_die1() {
        let my_process =
            BirthDeathProcess::new(&Rates::new(&[0.], &[0.], &[1.], &[0.]));
        assert_eq!(my_process.gillespie(1, 2).kind, AdvanceRun::Die1);
    }

    #[test]
    fn test_gillespie_returns_die2() {
        let my_process =
            BirthDeathProcess::new(&Rates::new(&[0.], &[0.], &[0.], &[1.]));
        assert_eq!(AdvanceRun::Die2, my_process.gillespie(1, 2).kind);
    }

    #[test]
    fn test_default_rates() {
        let rates = Rates::new(&[1f32], &[1f32], &[0f32], &[0f32]);
        assert_eq!(rates, Rates::default());
        assert_eq!(rates.fitness1, ProliferationRate::from(1f32));
        assert_eq!(rates.fitness2, ProliferationRate::from(1f32));
        assert_eq!(rates.death1, DeathRate::from(0f32));
        assert_eq!(rates.death2, DeathRate::from(0f32));
    }

    #[test]
    #[should_panic]
    fn test_panic_proliferation_rate_smaller_than_0() {
        ProliferationRate::new(&[-1f32]);
    }

    #[test]
    #[should_panic]
    fn test_negative_death_rate_panics() {
        DeathRate::new(&[-2f32]);
    }
}
