//! The ecDNA segregation rules applied upon proliferation of a cell with
//! ecDNAs.
use crate::DNACopy;
use rand::Rng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU16;

impl From<DNACopySegregating> for DNACopy {
    fn from(copies: DNACopySegregating) -> Self {
        copies.0
    }
}

/// A [`DNACopy`] that is known to be greater 1.
///
/// Represents the ecDNA copies of a proliferating cell that will be divided
/// into two partitions `k1` and `k2` via ecDNA segregation.
#[derive(Debug, Clone, Copy)]
pub struct DNACopySegregating(DNACopy);

impl From<DNACopySegregating> for u16 {
    fn from(value: DNACopySegregating) -> Self {
        value.0.get()
    }
}

impl TryFrom<DNACopy> for DNACopySegregating {
    type Error = &'static str;

    fn try_from(value: DNACopy) -> Result<Self, Self::Error> {
        if value <= unsafe { NonZeroU16::new_unchecked(1) } {
            Err("DNACopySegregating only accepts value superior than one")
        } else if value.get() % 2 == 1 {
            Err("DNACopySegregating only accepts multiple of two values")
        } else {
            Ok(DNACopySegregating(value))
        }
    }
}

/// `IsUneven` indicates whether upon cell proliferation and ecDNA segregation
/// a complete uneven segregation has occurred.
///
/// A complete uneven segregation occurs when one daughter cell inherits all
/// the ecDNA material from the mother cell.
/// In this case, upon proliferation of a cell with ecDNAs, a cell without
/// ecDNA is generated.

#[derive(Debug, PartialEq, Eq)]
pub enum IsUneven {
    True,
    False,
    /// This special case if an uneven segregation occurs, but we do not want
    /// to increase the count of the cells without any ecDNAs.
    TrueWithoutNMinusIncrease,
}

/// `Segregate` runs the ecDNA segregation by splitting the total ecDNA copies
/// into two partitions.
///
/// Assuming a dividing cell has `k` copies, then the ecDNA segregation is a
/// three-step process implying:
/// 1. duplicating the ecDNA copies (from `k` to `2k`),
/// 2. segregating the `2k` copies into two partitions, one with `k1` and the
/// other with `k2=2k-k1`,
/// 3. updating the ecDNA distribution with `k1` and `k2` copies and removing
/// the cell with `k` copies that has just divided.
///
/// Note that `Segregate` only takes care of the second point, that is the
/// splitting of the `2k` ecDNA copies.
///
/// The segregation rule of the ecDNA copies that creates two partitions either
/// randomly via a [`RandomSegregation`] or deterministically.
pub trait Segregate {
    /// Run the segregation and returns `k1`, `k2` and whether an uneven
    /// segregation occured.
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, IsUneven);
}

/// The deterministic model always return the same number of copies in the two
/// daughter cells
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Deterministic;
/// The Binomial segregation model splits the ecDNA copies into `2k`
/// indipendent [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_trial)
/// trials with probability of 1/2.
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Binomial;
/// The Binomial segregation model without the possibility to generate a
/// complete uneven segregation.
///
/// Segregates ecDNA copies according to a Binomial segregation as in
/// [`Binomial`] but without the option `k1=0` and `k2=2k` or viceversa.
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct BinomialNoUneven(pub Binomial);
/// The Binomial segregation model which does not increase the number of cells
/// without ecDNAs upon complete uneven segregation.
///
/// Every time a complete uneven segregation is simulated, we do not increase
/// the number of cells without ecDNAs.
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct BinomialNoNminus(pub Binomial);

impl Segregate for Binomial {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, IsUneven) {
        // unwrap because p=0.5 will never raise a ProbabilityTooSmall or
        // ProbabilityTooLarge error
        // downcast u64 to u16 will never fail because `copies` is u16
        let copies = u16::from(copies) as u64;

        let k1: u64 =
            rand_distr::Binomial::new(copies, 0.5).unwrap().sample(rng);
        let k2 = copies - k1;

        // uneven_segregation happens if there is at least one zero
        let is_uneven = {
            if k1 == 0 || k2 == 0 {
                IsUneven::True
            } else {
                IsUneven::False
            }
        };
        if verbosity > 1 {
            println!("{} duplicated copies have been segregated into {}, {} generating an complete uneven seg {:#?}", copies, k1, k2, is_uneven);
        }

        (k1, k2, is_uneven)
    }
}

impl Segregate for Deterministic {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        _rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, IsUneven) {
        let k1 = copies.0.get() as u64 / 2u64;
        if verbosity > 1 {
            println!("Returning {}, {} and IsUneven::False", k1, k1);
        }
        (k1, k1, IsUneven::False)
    }
}

impl Segregate for BinomialNoUneven {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, IsUneven) {
        let (mut k1, mut k2, mut is_uneven) =
            self.0.ecdna_segregation(copies, rng, verbosity);

        while is_uneven == IsUneven::True {
            (k1, k2, is_uneven) =
                self.0.ecdna_segregation(copies, rng, verbosity);
        }

        (k1, k2, IsUneven::False)
    }
}

impl Segregate for BinomialNoNminus {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> (u64, u64, IsUneven) {
        let (k1, k2, is_uneven) =
            self.0.ecdna_segregation(copies, rng, verbosity);

        match is_uneven {
            IsUneven::True => (k1, k2, IsUneven::TrueWithoutNMinusIncrease),
            IsUneven::False => (k1, k2, IsUneven::False),
            IsUneven::TrueWithoutNMinusIncrease => {
                unreachable!("Should never returns that")
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseSegregationError;

/// The ecDNA random segregation rule applied upon cell division.
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum RandomSegregation {
    /// Perform the random segregation using a [`Binomial`] model.
    BinomialSegregation,
    /// Perform the random segregation using a Binomial segregation model but
    /// without the possibility of generating a complete uneven segregation,
    /// see [`BinomialNoUneven`].
    BinomialNoUneven,
    /// Perform the random segregation using a Binomial segregation model but
    /// every time a complete uneven segregation is simulated, do not increase
    /// the number of cells without ecDNAs.
    BinomialNoNminus,
}

#[cfg(test)]
mod tests {
    use crate::test_util::DNACopySegregatingGreatherThanOne;

    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    #[should_panic]
    fn try_from_dna_copy_0_test() {
        DNACopySegregating::try_from(NonZeroU16::new(0).unwrap()).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_from_dna_copy_1_test() {
        DNACopySegregating::try_from(NonZeroU16::new(1).unwrap()).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_from_dna_copy_3_test() {
        DNACopySegregating::try_from(NonZeroU16::new(1).unwrap()).unwrap();
    }

    #[quickcheck]
    fn try_from_dna_copy_test(copy: DNACopySegregatingGreatherThanOne) {
        let copy_segregating: DNACopy = (copy.0).into();

        assert_eq!(u16::from(copy.0), copy_segregating.get());
    }

    #[derive(Clone, Debug)]
    struct AvoidOverflowDNACopy(DNACopy);

    impl Arbitrary for AvoidOverflowDNACopy {
        fn arbitrary(g: &mut Gen) -> AvoidOverflowDNACopy {
            let mut copy = DNACopy::arbitrary(g);
            while copy >= NonZeroU16::new(u16::MAX / 2).unwrap() {
                copy = DNACopy::arbitrary(g);
            }
            AvoidOverflowDNACopy(copy)
        }
    }

    #[quickcheck]
    fn segregate_deterministic_test(
        copies: DNACopySegregatingGreatherThanOne,
        seed: u64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (k1, k2, is_uneven) =
            Deterministic {}.ecdna_segregation(copies.0, &mut rng, 0);
        // uneven numbers
        assert_eq!(k1, k2);
        assert_eq!(u16::from(copies.0) as u64, 2 * k1);
        assert_eq!(is_uneven, IsUneven::False);
    }

    #[quickcheck]
    fn segregate_random_binomial_test(
        copies: DNACopySegregatingGreatherThanOne,
        seed: u64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let ecdna_copies = u16::from(DNACopy::from(copies.0));
        let (k1, k2, is_uneven) =
            Binomial.ecdna_segregation(copies.0, &mut rng, 0);
        assert_eq!(k1 + k2, ecdna_copies as u64);
        match is_uneven {
            IsUneven::True | IsUneven::TrueWithoutNMinusIncrease => {
                assert!(k1 == 0 || k2 == 0)
            }
            IsUneven::False => assert!(k1 != 0 && k2 != 0),
        }
    }

    #[quickcheck]
    fn segregate_random_binomial_no_nminus_test(
        copies: DNACopySegregatingGreatherThanOne,
        seed: u64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let ecdna_copies = copies.0;
        let (k1, k2, is_uneven) = BinomialNoUneven(Binomial)
            .ecdna_segregation(ecdna_copies, &mut rng, 0);
        assert_eq!(k1 + k2, u16::from(DNACopy::from(ecdna_copies)) as u64);
        assert_eq!(is_uneven, IsUneven::False);
    }
}
