//! The two-type/k-type ecDNA simulation problem modelling the effect of the
//! random segregation on the ecDNA dynamics.
/// The ecDNA dynamics such as the mean in function of time.
pub mod dynamics;
/// The available ecDNA processes.
pub mod process;
/// EcDNA growth models
pub mod proliferation;
/// EcDNA segregation models.
pub mod segregation;

#[cfg(test)]
pub mod test_util {
    use std::{
        collections::HashMap,
        num::{NonZeroU16, NonZeroU8},
    };

    use super::segregation::{
        BinomialNoNminus, BinomialNoUneven, BinomialSegregation,
        DNACopySegregating, Segregation,
    };
    use quickcheck::{Arbitrary, Gen};
    use ssa::{distribution::EcDNADistribution, DNACopy};

    #[derive(Clone, Debug)]
    struct DNACopyGreaterOne(DNACopy);

    impl Arbitrary for DNACopyGreaterOne {
        fn arbitrary(g: &mut Gen) -> DNACopyGreaterOne {
            let mut copy = NonZeroU16::arbitrary(g);
            if copy == NonZeroU16::new(1).unwrap() {
                copy = NonZeroU16::new(2).unwrap();
            }
            DNACopyGreaterOne(copy)
        }
    }

    #[derive(Clone, Debug)]
    pub struct NonEmptyDistribtionWithNPlusCells(pub EcDNADistribution);

    impl Arbitrary for NonEmptyDistribtionWithNPlusCells {
        fn arbitrary(g: &mut Gen) -> NonEmptyDistribtionWithNPlusCells {
            const MAX_ENTRIES: usize = 500;
            let mut distr = HashMap::with_capacity(MAX_ENTRIES);
            for _ in 0..MAX_ENTRIES {
                let copy = DNACopySegregatingGreatherThanOne::arbitrary(g);
                let cells = NonZeroU8::arbitrary(g).get() as u64;
                distr.insert(u16::from(DNACopy::from(copy.0)), cells);
            }
            let cells = NonZeroU8::arbitrary(g).get() as u64;
            distr.insert(0, cells);
            let distr = EcDNADistribution::new(distr, 1000);
            NonEmptyDistribtionWithNPlusCells(distr)
        }
    }
    #[derive(Clone, Debug)]
    pub struct DNACopySegregatingGreatherThanOne(pub DNACopySegregating);

    impl Arbitrary for DNACopySegregatingGreatherThanOne {
        fn arbitrary(g: &mut Gen) -> DNACopySegregatingGreatherThanOne {
            let mut copy =
                DNACopy::new(NonZeroU8::arbitrary(g).get() as u16).unwrap();
            if copy == DNACopy::new(1).unwrap() {
                copy = DNACopy::new(2).unwrap();
            }
            DNACopySegregatingGreatherThanOne(
                DNACopySegregating::try_from(copy).unwrap(),
            )
        }
    }

    impl Arbitrary for Segregation {
        fn arbitrary(g: &mut Gen) -> Self {
            let y = g.choose(&[0u8, 1u8, 2u8, 3u8, 4u8]).unwrap();
            let bin = BinomialSegregation;
            if y % 2 == 0 {
                match y {
                    0 => return Self::Random(bin.into()),
                    2 => return Self::Random(BinomialNoUneven(bin).into()),
                    4 => return Self::Random(BinomialNoNminus(bin).into()),
                    _ => unreachable!(),
                };
            }
            Self::Deterministic
        }
    }

    #[derive(Clone, Debug)]
    pub struct NPlusVec(pub Vec<DNACopy>);

    impl Arbitrary for NPlusVec {
        fn arbitrary(g: &mut Gen) -> Self {
            NPlusVec(
                (0..4u16 * (u8::MAX as u16))
                    .map(|_| NonZeroU16::arbitrary(g))
                    .collect(),
            )
        }
    }
}
