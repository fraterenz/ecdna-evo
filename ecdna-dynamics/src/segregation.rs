//! The ecDNA segregation rules applied upon proliferation of a cell with
//! ecDNAs.
use ecdna_data::DNACopy;
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
/// splitting of the `2k` ecDNA copies, which is defined by [`Segregation`].
///
#[enum_dispatch]
pub trait Segregate {
    /// Run the segregation and returns only `k1`, since `k2` is by definition
    /// `k2=2k-k1`.
    fn ecdna_segregation(
        &self,
        copies: DNACopy,
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
        copies: DNACopy,
        rng: &mut Pcg64Mcg,
    ) -> DNACopy {
        // unwrap because p=0.5 will never raise a ProbabilityTooSmall or
        // ProbabilityTooLarge error
        // downcast u64 to u16 will never fail because `copies` is u16
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
        copies: DNACopy,
        rng: &mut Pcg64Mcg,
    ) -> DNACopy {
        //! # Panics
        //! Panics if `copies` is 1 or 0.
        assert!(copies > 1);
        let mut k1: DNACopy = 0;
        while k1 == 0 || k1 == copies {
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
    //! Segregate the ecDNA `copies` into two partitions.
    pub fn segregate(
        &self,
        copies: DNACopy,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy) {
        //! Take `copies` ecDNAs and split them into `k1` and `k2` ecDNA
        //! copies.
        //! # Panics
        //! Panics when `copies` is zero.
        assert_ne!(copies, 0);
        let k1 = match self {
            Segregation::Random(seg) => seg.ecdna_segregation(copies, rng),
            Segregation::Deterministic => copies / 2,
        };
        let k2 = copies - k1;
        (k1, k2)
    }
}

/// The random segregation rule will be applied upon ecDNA segregation.
#[enum_dispatch[Segregate]]
#[derive(Clone, Debug, Copy)]
pub enum RandomSegregation {
    /// Perform the random segregation using the [`BinomialSegregation`].
    BinomialSegregation,
    BinomialNoNMinusSegregation,
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use quickcheck::{Arbitrary, Gen};

    #[derive(Clone, Debug)]
    struct NonZeroDNACopy(DNACopy);

    impl Arbitrary for NonZeroDNACopy {
        fn arbitrary(g: &mut Gen) -> NonZeroDNACopy {
            let mut copy = DNACopy::arbitrary(g);
            while copy == 0 {
                copy = DNACopy::arbitrary(g);
            }
            NonZeroDNACopy(copy)
        }
    }

    #[derive(Clone, Debug)]
    struct NonZeroGreaterThanOneDNACopy(NonZeroDNACopy);

    impl Arbitrary for NonZeroGreaterThanOneDNACopy {
        fn arbitrary(g: &mut Gen) -> NonZeroGreaterThanOneDNACopy {
            let mut copy = NonZeroDNACopy::arbitrary(g);
            while copy.0 == 1 {
                copy = NonZeroDNACopy::arbitrary(g);
            }
            NonZeroGreaterThanOneDNACopy(copy)
        }
    }

    #[quickcheck]
    fn segregate_deterministic_test(copies: NonZeroDNACopy, seed: u64) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies = copies.0;
        let (k1, k2) =
            Segregation::Deterministic.segregate(ecdna_copies, &mut rng);
        // uneven numbers
        assert!(k1 == k2 || k1 + 1 == k2 || k1 == k2 + 1);
        assert!(k1 == ecdna_copies / 2 || k2 == ecdna_copies / 2);
    }

    #[test]
    #[should_panic]
    fn segregate_deterministic_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        let ecdna_copies = 0;
        Segregation::Deterministic.segregate(ecdna_copies, &mut rng);
    }

    #[quickcheck]
    fn segregate_random_binomial_test(copies: NonZeroDNACopy, seed: u64) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies = copies.0;
        let (k1, k2) = Segregation::Random(BinomialSegregation.into())
            .segregate(ecdna_copies, &mut rng);
        assert_eq!(k1 + k2, ecdna_copies)
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        let ecdna_copies = 0;
        Segregation::Random(BinomialSegregation.into())
            .segregate(ecdna_copies, &mut rng);
    }

    #[quickcheck]
    fn segregate_random_binomial_no_nminus_test(
        copies: NonZeroGreaterThanOneDNACopy,
        seed: u64,
    ) {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let ecdna_copies = copies.0 .0;
        let (k1, k2) = Segregation::Random(
            BinomialNoNMinusSegregation(BinomialSegregation).into(),
        )
        .segregate(ecdna_copies, &mut rng);
        assert_eq!(k1 + k2, ecdna_copies);
        assert_ne!(k1, 0);
        assert_ne!(k2, 0);
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_no_nminus_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        let ecdna_copies = 0;
        Segregation::Random(
            BinomialNoNMinusSegregation(BinomialSegregation).into(),
        )
        .segregate(ecdna_copies, &mut rng);
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_no_nminus_1_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        let ecdna_copies = 1;
        Segregation::Random(
            BinomialNoNMinusSegregation(BinomialSegregation).into(),
        )
        .segregate(ecdna_copies, &mut rng);
    }
}
