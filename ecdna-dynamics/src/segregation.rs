//! The ecDNA segregation rules applied upon proliferation of a cell with
//! ecDNAs.
use ecdna_data::{DNACopy, DNACopySegregating};
use enum_dispatch::enum_dispatch;
use rand_distr::{Binomial, Distribution};
use rand_pcg::Pcg64Mcg;

/// `IsUneven` indicates whether upon cell proliferation and ecDNA segregation
/// a complete uneven segregation has occurred.
///
/// A complete uneven segregation occurs when one daughter cell inherits all
/// the ecDNA material from the mother cell.
/// In this case, upon proliferation of a cell with ecDNAs, a cell without
/// ecDNA is generated.
#[derive(Debug, PartialEq)]
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
#[enum_dispatch]
pub trait Segregate {
    /// Run the segregation and returns `k1`, `k2` and whether an uneven
    /// segregation occured.
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy, IsUneven);
}

/// The Binomial segregation model splits the ecDNA copies into `2k`
/// indipendent [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_trial)
/// trials with probability of 1/2.
#[derive(Clone, Debug, Copy)]
pub struct BinomialSegregation;

/// The Binomial segregation model without the possibility to generate a
/// complete uneven segregation.
///
/// Segregates ecDNA copies according to a Binomial segregation as in
/// [`BinomialSegregation`] but without the option `k1=0` and `k2=2k` or
/// viceversa.
#[derive(Clone, Debug, Copy)]
pub struct BinomialNoUneven(pub BinomialSegregation);

/// The Binomial segregation model which does not increase the number of cells
/// without ecDNAs upon complete uneven segregation.
///
/// Every time a complete uneven segregation is simulated, we do not increase
/// the number of cells without ecDNAs.
#[derive(Clone, Debug, Copy)]
pub struct BinomialNoNminus(pub BinomialSegregation);

impl Segregate for BinomialSegregation {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy, IsUneven) {
        // unwrap because p=0.5 will never raise a ProbabilityTooSmall or
        // ProbabilityTooLarge error
        // downcast u64 to u16 will never fail because `copies` is u16
        let copies: DNACopy = DNACopy::from(copies);
        let k1 = Binomial::new(copies as u64, 0.5)
            .unwrap()
            .sample(rng)
            .try_into()
            .unwrap();
        let k2 = copies - k1;
        // uneven_segregation happens if there is at least one zero
        let is_uneven = {
            if k1 == 0 || k2 == 0 {
                IsUneven::True
            } else {
                IsUneven::False
            }
        };
        (k1, k2, is_uneven)
    }
}

impl Segregate for BinomialNoUneven {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy, IsUneven) {
        let mut k1: DNACopy = 0;
        let mut k2: DNACopy = 0;
        let ecdna_copies = DNACopy::from(copies);
        while k1 == 0 || k1 == ecdna_copies {
            (k1, k2, _) = self.0.ecdna_segregation(copies, rng);
        }
        (k1, k2, IsUneven::False)
    }
}

impl Segregate for BinomialNoNminus {
    fn ecdna_segregation(
        &self,
        copies: DNACopySegregating,
        rng: &mut Pcg64Mcg,
    ) -> (DNACopy, DNACopy, IsUneven) {
        let (k1, k2, is_uneven) = self.0.ecdna_segregation(copies, rng);
        match is_uneven {
            IsUneven::True => (k1, k2, IsUneven::TrueWithoutNMinusIncrease),
            IsUneven::False => (k1, k2, IsUneven::False),
            IsUneven::TrueWithoutNMinusIncrease => {
                unreachable!("Should never returns that")
            }
        }
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
    ) -> (DNACopy, DNACopy, IsUneven) {
        //! Double `copies` ecDNAs and split them into `k1` and `k2`.
        //! Returns also whether a uneven random segregation occurred, that is
        //! if `k1=2*copies` and `k2=0` or viceversa.
        //!
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
                seg.ecdna_segregation(segregating_copies, rng)
            }
            Segregation::Deterministic => (copies, copies, IsUneven::False),
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
    /// see [`BinomialNoUneven`].
    BinomialNoUneven,
    /// Perform the random segregation using a Binomial segregation model but
    /// every time a complete uneven segregation is simulated, do not increase
    /// the number of cells without ecDNAs.
    BinomialNoNminus,
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
        let (k1, k2, is_uneven) =
            Segregation::Deterministic.segregate(ecdna_copies, &mut rng);
        // uneven numbers
        assert_eq!(k1, k2);
        assert_eq!(ecdna_copies, k1);
        assert_eq!(is_uneven, IsUneven::False);
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
        let (k1, k2, is_uneven) =
            BinomialSegregation.ecdna_segregation(ecdna_copies, &mut rng);
        assert_eq!(k1 + k2, DNACopy::from(ecdna_copies));
        match is_uneven {
            IsUneven::True | IsUneven::TrueWithoutNMinusIncrease => {
                assert!(k1 == 0 || k2 == 0)
            }
            IsUneven::False => assert!(k1 != 0 && k2 != 0),
        }
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
        let (k1, k2, is_uneven) = BinomialNoUneven(BinomialSegregation)
            .ecdna_segregation(ecdna_copies, &mut rng);
        assert_eq!(k1 + k2, DNACopy::from(ecdna_copies));
        assert_eq!(is_uneven, IsUneven::False);
    }

    #[test]
    #[should_panic]
    fn segregate_binomial_no_nminus_0_copies_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(26);
        Segregation::Random(BinomialNoUneven(BinomialSegregation).into())
            .segregate(0, &mut rng);
    }
}
