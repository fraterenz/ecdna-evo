use rand::Rng;
use std::num::NonZeroU16;

use crate::distribution::EcDNADistribution;
use crate::segregation::{DNACopySegregating, IsUneven, Segregate};

/// Update the [`EcDNADistribution`] according to the random segregation.
pub trait EcDNAProliferation {
    fn increase_nplus(
        &self,
        distribution: &mut EcDNADistribution,
        segregation: &impl Segregate,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> anyhow::Result<IsUneven>;
    fn increase_nminus(&self, distribution: &mut EcDNADistribution);
}

/// The ecDNA dynamics according to an exponential growth model with random
/// segregation.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {}

impl EcDNAProliferation for Exponential {
    fn increase_nplus(
        &self,
        distribution: &mut EcDNADistribution,
        segregation: &impl Segregate,
        rng: &mut impl Rng,
        verbosity: u8,
    ) -> anyhow::Result<IsUneven> {
        //! Can overflow. Updates the ecDNA distribution.
        //!
        //! Simulate a proliferative event of a cell with ecDNAs in an
        //! exponentially growing tumour.
        //!
        //! We pick the next proliferative cells with ecDNAs at random from the
        //! population.
        //! Then, we randomly distribute its ecDNA copies between two daughter
        //! cells, and update the ecDNA distribution according to the random
        //! segregation (see [`Segregate`]) simulated upon cell-proliferation.
        //! A proliferation of a cell with ecDNAs generates one of the three
        //! following outcomes:
        //!
        //! - a [`IsUneven::False`]: we increase the number of cells with
        //! ecDNAs
        //!
        //! - a [`IsUneven::True`]: we increase the number of cells without
        //! ecDNA, we don't change the number of cells with ecDNAs,
        //!
        //! - a [`IsUneven::TrueWithoutNMinusIncrease`]: we do not increase the
        //! number of cells without ecDNAs even though a complete uneven
        //! segregation event occured.
        //!
        //! ## Fails
        //! Fails when there are no cells with ecDNA in the population.
        let ecdna = distribution.pick_remove_random_nplus(rng)?;
        if verbosity > 1 {
            println!("Picked {} copies", ecdna);
        }
        // Double the number of `NPlus` from the idx cell before proliferation
        // because a parental cell gives rise to 2 daughter cells.
        let doubled_copies = unsafe {
            NonZeroU16::new_unchecked(ecdna.get().checked_mul(2).expect(
                "Overflow while segregating DNA into two daughter cells",
            ))
        };

        if verbosity > 1 {
            println!("Duplicating {} copies into {}", ecdna, doubled_copies);
        }

        // check if doubled_copies is greater than 1 (also for the
        // deterministic case)
        let segregating_copies = DNACopySegregating::try_from(doubled_copies)
            .unwrap_or_else(|_| panic!("Cannot cast {}", doubled_copies));

        let (k1, k2, uneven_segregation) =
            segregation.ecdna_segregation(segregating_copies, rng, verbosity);
        let (k1, k2) = (k1 as u16, k2 as u16);
        let copies = match uneven_segregation {
            IsUneven::False => {
                // is not complete uneven, then no 0 copies should have appeared
                unsafe {
                    vec![
                        NonZeroU16::new_unchecked(k1),
                        NonZeroU16::new_unchecked(k2),
                    ]
                }
            }
            IsUneven::True => {
                distribution.increase_nminus();
                // n + 0 == n == 0 + n, hence the sum cannot be zero
                unsafe { vec![NonZeroU16::new_unchecked(k1 + k2)] }
            }
            IsUneven::TrueWithoutNMinusIncrease => {
                // n + 0 == n == 0 + n, hence the sum cannot be zero
                unsafe { vec![NonZeroU16::new_unchecked(k1 + k2)] }
            }
        };
        if verbosity > 1 {
            println!(
                "A cell with {} copies duplicates its copies and distributes them as {:#?} resulting in {:#?} complete uneven",
                ecdna,
                copies,
                uneven_segregation
            );
        }
        distribution.increase_nplus(copies, verbosity);
        Ok(uneven_segregation)
    }

    fn increase_nminus(&self, distribution: &mut EcDNADistribution) {
        // we unwrap because a EcDNADistribution has always an entry for the
        // nminus cells.
        distribution.increase_nminus();
    }
}

/// Add cell death as an event in the model to simulate a birth-death process.
#[derive(Debug, Clone)]
pub struct CellDeath;

/// Update the ecDNA distribution upon cell death.
impl CellDeath {
    pub fn decrease_nplus(
        &self,
        distribution: &mut EcDNADistribution,
        rng: &mut impl Rng,
        verbosity: u8,
    ) {
        distribution.decrease_nplus(rng, verbosity);
    }

    pub fn decrease_nminus(&self, distribution: &mut EcDNADistribution) {
        // we unwrap because a EcDNADistribution has always an entry for the
        // nminus cells.
        distribution.decrease_nminus();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        segregation::{
            Binomial, BinomialNoNminus, BinomialNoUneven, Deterministic,
        },
        test_util::{
            NonEmptyDistribtionWithNPlusCells, SegregationTypes,
            TestSegregation,
        },
    };

    use super::*;
    use quickcheck_macros::quickcheck;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[quickcheck]
    fn increase_nplus_test(
        seed: u64,
        distribution: NonEmptyDistribtionWithNPlusCells,
        segregation: TestSegregation,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let ecdna = Exponential {};
        let mut distribution = distribution.0;
        let nplus_before_update = distribution.compute_nplus();
        let nminus_before_update = *distribution.get_nminus();

        let is_uneven = match segregation.0 {
            SegregationTypes::Deterministic => {
                let is_uneven = ecdna
                    .increase_nplus(
                        &mut distribution,
                        &Deterministic,
                        &mut rng,
                        0,
                    )
                    .unwrap();

                assert_eq!(IsUneven::False, is_uneven);
                assert_eq!(
                    distribution.compute_nplus(),
                    nplus_before_update + 1
                );
                assert_eq!(*distribution.get_nminus(), nminus_before_update);
                is_uneven
            }
            SegregationTypes::BinomialNoUneven => {
                let is_uneven = ecdna
                    .increase_nplus(
                        &mut distribution,
                        &BinomialNoUneven(Binomial),
                        &mut rng,
                        0,
                    )
                    .unwrap();
                assert_eq!(IsUneven::False, is_uneven);
                is_uneven
            }
            SegregationTypes::BinomialNoNminus => {
                let is_uneven = ecdna.increase_nplus(
                    &mut distribution,
                    &BinomialNoNminus(Binomial),
                    &mut rng,
                    0,
                );
                is_uneven.unwrap()
            }
            SegregationTypes::BinomialSegregation => {
                let is_uneven = ecdna.increase_nplus(
                    &mut distribution,
                    &Binomial,
                    &mut rng,
                    0,
                );
                is_uneven.unwrap()
            }
        };
        match is_uneven {
            IsUneven::False => {
                assert_eq!(
                    distribution.compute_nplus(),
                    nplus_before_update + 1
                );
                assert_eq!(*distribution.get_nminus(), nminus_before_update);
            }
            IsUneven::True => {
                assert_eq!(distribution.compute_nplus(), nplus_before_update);
                assert_eq!(
                    *distribution.get_nminus(),
                    nminus_before_update + 1
                );
            }
            IsUneven::TrueWithoutNMinusIncrease => {
                assert_eq!(distribution.compute_nplus(), nplus_before_update,);
                assert_eq!(*distribution.get_nminus(), nminus_before_update);
            }
        }
    }

    #[quickcheck]
    fn increase_nminus_test(
        distribution: NonEmptyDistribtionWithNPlusCells,
    ) -> bool {
        let ecdna = Exponential {};
        let mut distribution = distribution.0;
        let expected_nminus = *distribution.get_nminus() + 1;
        let expected_nplus = distribution.compute_nplus();

        ecdna.increase_nminus(&mut distribution);
        distribution.get_nminus() == &expected_nminus
            && distribution.compute_nplus() == expected_nplus
    }

    #[quickcheck]
    fn decrease_nplus_test(
        seed: u64,
        distribution: NonEmptyDistribtionWithNPlusCells,
    ) -> bool {
        let ecdna = CellDeath;
        let mut distribution = distribution.0;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let expected_nminus = *distribution.get_nminus();
        let expected_nplus = distribution.compute_nplus() - 1;

        ecdna.decrease_nplus(&mut distribution, &mut rng, 0);
        *distribution.get_nminus() == expected_nminus
            && distribution.compute_nplus() == expected_nplus
    }

    #[quickcheck]
    fn decrease_nminus_test(
        distribution: NonEmptyDistribtionWithNPlusCells,
    ) -> bool {
        let ecdna = CellDeath;
        let mut distribution = distribution.0;
        let expected_nminus = *distribution.get_nminus() - 1;
        let expected_nplus = distribution.compute_nplus();

        ecdna.decrease_nminus(&mut distribution);
        *distribution.get_nminus() == expected_nminus
            && distribution.compute_nplus() == expected_nplus
    }
}
