//! The ecDNA segregation rules applied upon proliferation of a cell with
//! ecDNAs.
use ecdna_data::{DNACopy, DNACopySegregating};
use enum_dispatch::enum_dispatch;
use rand_distr::{Binomial, Distribution};
use rand_pcg::Pcg64Mcg;

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
#[enum_dispatch]
pub trait Segregate {
    /// Run the segregation and returns only `k1`, since `k2` is by definition
    /// `k2=2k-k1`.
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> DNACopy;
}

/// The Binomial segregation model splits the ecDNA copies into `2k`
/// indipendent [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_trial)
/// trials with probability of 1/2.
#[derive(Clone, Debug, Copy)]
pub struct BinomialSegregation;

/// The Binomial segregation model without the possibility to generate a
/// complete uneven segregation, that is perform Binomial segregation as in
/// [`BinomialSegregation`] but without the option `k1=0` and `k2=2k` or
/// viceversa.
#[derive(Clone, Debug, Copy)]
pub struct BinomialNoNMinusSegregation(pub BinomialSegregation);

impl Segregate for BinomialSegregation {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> DNACopy {
        // unwrap because p=0.5 will never raise a ProbabilityTooSmall or
        // ProbabilityTooLarge error
        // downcast u64 to u16 will never fail because `copies` is u16
        let copies: DNACopy = DNACopy::from(copies);
        Binomial::new(copies as u64, 0.5)
            .unwrap()
            .sample(rng)
            .try_into()
            .unwrap()
    }
}

impl Segregate for BinomialNoNMinusSegregation {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> DNACopy {
        let mut k1: DNACopy = 0;
        while k1 == 0 || k1 == DNACopy::from(copies) {
            k1 = self.0.ecdna_segregation(copies, rng);
        }
        k1
    }
}

/// The segregation rule of the ecDNA copies that creates two partitions either
/// randomly via a [`RandomSegregation`] or deterministically.
#[derive(Clone, Debug, Copy)]
pub enum Segregation {
    /// The Random segregation model split the ecDNA copies randomly.
    Random(RandomSegregation),
    /// The Deterministic segregation model is non-random and always halves the
    /// ecDNA copies.
    Deterministic,
}

impl Segregation {
    pub fn segregate(
        &self,
        copies: DNACopy,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy) {
        //! Double `copies` ecDNAs and split them into `k1` and `k2`.
        //! # Panics
        //! Panics when `copies` whether is zero or one.
        //!
        //! Panics also when overflow error, that is when `DNACopy >= u16::MAX / 2`.
        // Double the number of `NPlus` from the idx cell before proliferation
        // because a parental cell gives rise to 2 daughter cells.
        let doubled_copies: DNACopy = copies
            .checked_add(copies)
            .expect("Overflow while segregating DNA into two daughter cells");

        // check if doubled_copies is greater than 1 (also for the
        // deterministic case)
        let segregating_copies =
            DNACopySegregating::try_from(doubled_copies).expect("Cannot cast");
        match self {
            Segregation::Random(seg) => {
                let k1 = seg.ecdna_segregation(segregating_copies, rng);
                (k1, doubled_copies - k1)
            }
            Segregation::Deterministic => (copies, copies),
        }
    }
}

/// The ecDNA random segregation rule applied upon cell division.
#[enum_dispatch[Segregate]]
#[derive(Clone, Debug, Copy)]
pub enum RandomSegregation {
    /// Perform the random segregation using a [`BinomialSegregation`] model.
    BinomialSegregation,
    /// Perform the random segregation using a Binomial segregation model but
    /// without the possibility of generating a complete uneven segregation,
    /// see [`BinomialNoNMinusSegregation`].
    BinomialNoNMinusSegregation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use rand::SeedableRng;

    #[derive(Clone, Debug)]
    struct AvoidOverflowDNACopy(DNACopy);

    #[derive(Clone, Debug)]
    struct DNACopySegregatingGreatherThanOne(DNACopySegregating);

    impl Arbitrary for DNACopySegregatingGreatherThanOne {
        fn arbitrary(g: &mut Gen) -> DNACopySegregatingGreatherThanOne {
            let mut copy = DNACopy::arbitrary(g);
            while copy <= 1 {
                copy = DNACopy::arbitrary(g);
            }
            DNACopySegregatingGreatherThanOne(
                DNACopySegregating::try_from(copy).unwrap(),
            )
        }
    }

    impl Arbitrary for AvoidOverflowDNACopy {
        fn arbitrary(g: &mut Gen) -> AvoidOverflowDNACopy {
            let mut copy = DNACopy::arbitrary(g);
            while copy >= (u16::MAX / 2) || copy == 0 {
                copy = DNACopy::arbitrary(g);
            }
            AvoidOverflowDNACopy(copy)
        }
    }

    #[quickcheck]
    fn segregate_deterministic_test(copies: AvoidOverflowDNACopy, seed: u64) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies: DNACopy = copies.0;
        let (k1, k2) =
            Segregation::Deterministic.segregate(ecdna_copies, &mut rng);
        // uneven numbers
        assert!(k1 == k2);
        assert!(ecdna_copies == k1);
    }

    #[test]
    #[should_panic]
    fn segregate_deterministic_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        Segregation::Deterministic.segregate(0, &mut rng);
    }

    #[quickcheck]
    fn segregate_random_binomial_test(
        copies: DNACopySegregatingGreatherThanOne,
        seed: u64,
    ) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies = copies.0;
        BinomialSegregation.ecdna_segregation(ecdna_copies, &mut rng);
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        Segregation::Random(BinomialSegregation.into()).segregate(0, &mut rng);
    }

    #[quickcheck]
    fn segregate_random_binomial_no_nminus_test(
        copies: DNACopySegregatingGreatherThanOne,
        seed: u64,
    ) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies = copies.0;
        BinomialNoNMinusSegregation(BinomialSegregation)
            .ecdna_segregation(ecdna_copies, &mut rng);
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_no_nminus_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        Segregation::Random(
            BinomialNoNMinusSegregation(BinomialSegregation).into(),
        )
        .segregate(0, &mut rng);
    }
}
