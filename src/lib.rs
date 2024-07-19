//! A k-type population model to simulate the effect of the random segregation
//! and positive selection on the ecDNA dynamics
/// The pure-birth and birth-death processes simulating the ecDNA dynamics.
pub mod process;
/// EcDNA growth models.
pub mod proliferation;
pub mod segregation;

use std::{collections::VecDeque, path::PathBuf};

pub use ecdna_lib::{distribution, DNACopy};

#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The number of cells to subsample
    pub cells2sample: usize,
    /// The time at which we subsample
    pub time: f32,
}

#[derive(Debug, Clone)]
pub struct SavingOptions {
    pub snapshots: VecDeque<Snapshot>,
    pub path2dir: PathBuf,
    pub filename: String,
}

pub fn create_filename_birth_death(rates: &[f32; 4], id: usize) -> String {
    format!(
        "{}b0_{}b1_{}d0_{}d1_{}idx",
        rates[0].to_string().replace('.', "dot"),
        rates[1].to_string().replace('.', "dot"),
        rates[2].to_string().replace('.', "dot"),
        rates[3].to_string().replace('.', "dot"),
        id,
    )
}

pub fn create_filename_pure_birth(rates: &[f32; 2], id: usize) -> String {
    format!(
        "{}b0_{}b1_0d0_0d1_{}idx",
        rates[0].to_string().replace('.', "dot"),
        rates[1].to_string().replace('.', "dot"),
        id,
    )
}

#[cfg(test)]
pub mod test_util {
    use std::{
        collections::HashMap,
        num::{NonZeroU16, NonZeroU8},
    };

    use super::segregation::DNACopySegregating;
    use ecdna_lib::{distribution::EcDNADistribution, DNACopy};
    use quickcheck::{Arbitrary, Gen};

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
            if copy == DNACopy::new(1).unwrap() || copy.get() % 2 == 1 {
                copy = DNACopy::new(2).unwrap();
            }
            DNACopySegregatingGreatherThanOne(
                DNACopySegregating::try_from(copy).unwrap(),
            )
        }
    }

    #[derive(Debug, Clone)]
    pub struct TestSegregation(pub SegregationTypes);

    #[derive(Debug, Clone)]
    pub enum SegregationTypes {
        Deterministic,
        BinomialNoUneven,
        BinomialNoNminus,
        BinomialSegregation,
    }

    impl Arbitrary for TestSegregation {
        fn arbitrary(g: &mut Gen) -> Self {
            let t = g.choose(&[0, 1, 2, 3]).unwrap();
            match t {
                0 => TestSegregation(SegregationTypes::Deterministic),
                1 => TestSegregation(SegregationTypes::BinomialNoUneven),
                2 => TestSegregation(SegregationTypes::BinomialNoNminus),
                3 => TestSegregation(SegregationTypes::BinomialSegregation),
                _ => unreachable!(),
            }
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
