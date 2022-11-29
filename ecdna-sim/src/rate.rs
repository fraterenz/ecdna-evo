//! Rates of the birth-death process.
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rand_distr::{Distribution, Open01, Uniform};
use rand_pcg::Pcg64Mcg;
use std::convert::From;
use std::fmt::{self, Display};
use std::ops::Mul;

use crate::process::{
    BirthDeath, BirthDeath1, BirthDeath2, BirthDeathProcess, PureBirth,
};
use crate::NbIndividuals;

#[derive(Clone, Debug)]
pub struct Range<T> {
    min: T,
    max: T,
}

impl<T> Default for Range<T>
where
    T: num_traits::Zero + num_traits::One,
{
    fn default() -> Self {
        Range { min: T::zero(), max: T::one() }
    }
}

impl<T: std::fmt::Display> Display for Range<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.min, self.max)
    }
}

impl<T> Range<T>
where
    T: std::fmt::Display
        + rand_distr::uniform::SampleUniform
        + std::cmp::PartialOrd,
{
    pub fn new(min: T, max: T) -> Range<T> {
        if min >= max {
            panic!("Found min {} greater than max {}", min, max)
        }
        Range { min, max }
    }

    pub fn sample_uniformly(&mut self, rng: &mut Pcg64Mcg) -> T {
        Uniform::new(&self.min, &self.max).sample(rng)
    }
}

/// Gillespie rate
pub type GillespieRate = f32;

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
    pub fn new(f1: &[f32], f2: &[f32], d1: &[f32], d2: &[f32]) -> Self {
        //! Create new `Rates`.
        //!
        //! Arguments must have len 1 or 2:
        //!     1. with len 1: the rate will be create with that value
        //!     2. with len 2: the rate will be sampled uniformly between an
        //!     interval specified by the two values
        Rates {
            fitness1: ProliferationRate::new(f1),
            fitness2: ProliferationRate::new(f2),
            death1: DeathRate::new(d1),
            death2: DeathRate::new(d2),
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
        Rates::new(&[1f32], &[1f32], &[0f32], &[0f32])
    }
}

/// Gillespie rate units 1/N
#[derive(Clone, Debug)]
pub enum Rate {
    Range(Range<f32>),
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
pub struct ProliferationRate(pub Rate);

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
        assert!(!rates.is_empty());
        assert!(rates.len() <= 2);
        ProliferationRate(Rate::new(rates))
    }
}

#[derive(Clone, Debug, Default)]
/// How many times on average there will be death for a stochastic birth-death
/// process
pub struct DeathRate(pub Rate);

impl DeathRate {
    pub fn new(rates: &[f32]) -> Self {
        assert!(
            !rates.iter().any(|val| val < &0f32),
            "DeathRate cannot be negative"
        );
        assert!(!rates.is_empty());
        assert!(rates.len() <= 2);
        DeathRate(Rate::new(rates))
    }

    pub fn is_zero(&self) -> bool {
        match self.0 {
            Rate::Scalar(rate) => (rate - 0f32).abs() < f32::EPSILON,
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

#[enum_dispatch(BirthDeathProcess)]
pub trait GetRates {
    //! Get the proliferation and death rates
    fn get_rates(&self) -> [f32; 4];
}

impl GetRates for PureBirth {
    fn get_rates(&self) -> [f32; 4] {
        //! For `PureBirth` the death coefficients are 0
        [self.r1, self.r2, 0f32, 0f32]
    }
}

impl GetRates for BirthDeath1 {
    fn get_rates(&self) -> [f32; 4] {
        //! For `BirthDeath1` the death coefficient of the cells w/o ecDNA is 0
        [self.r1, self.r2, self.d1, 0f32]
    }
}

impl GetRates for BirthDeath2 {
    fn get_rates(&self) -> [f32; 4] {
        //! For `BirthDeath2` the death coefficient of the cells w/ ecDNA is 0
        [self.r1, self.r2, 0f32, self.d2]
    }
}

impl GetRates for BirthDeath {
    fn get_rates(&self) -> [f32; 4] {
        [self.r1, self.r2, self.d1, self.d2]
    }
}

pub fn exprand(lambda: GillespieRate, rng: &mut Pcg64Mcg) -> f32 {
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
}
