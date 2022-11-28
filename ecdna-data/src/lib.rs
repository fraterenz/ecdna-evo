pub mod data;
pub mod patient;

#[macro_use]
extern crate derive_builder;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// Number of ecDNA copies within a cell. We assume that a cell cannot have more
/// than 65535 copies (`u16` is 2^16 - 1 = 65535 copies).
pub type DNACopy = u16;

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

impl TryFrom<DNACopy> for DNACopySegregating {
    type Error = &'static str;

    fn try_from(value: DNACopy) -> Result<Self, Self::Error> {
        if value <= 1 {
            Err("DNACopySegregating only accepts value superior than one!")
        } else {
            Ok(DNACopySegregating(value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use quickcheck::{Arbitrary, Gen};

    #[derive(Clone, Debug)]
    struct NonZeroNonOneDNACopy(DNACopy);

    impl Arbitrary for NonZeroNonOneDNACopy {
        fn arbitrary(g: &mut Gen) -> NonZeroNonOneDNACopy {
            let mut copy = DNACopy::arbitrary(g);
            while copy == 0 || copy == 1 {
                copy = DNACopy::arbitrary(g);
            }
            NonZeroNonOneDNACopy(copy)
        }
    }

    #[test]
    #[should_panic]
    fn try_from_dna_copy_0_test() {
        DNACopySegregating::try_from(0).unwrap();
    }

    #[test]
    #[should_panic]
    fn try_from_dna_copy_1_test() {
        DNACopySegregating::try_from(1).unwrap();
    }

    #[quickcheck]
    fn try_from_dna_copy_test(copy: NonZeroNonOneDNACopy) {
        let copy_segregating: DNACopy =
            DNACopySegregating::try_from(copy.0).unwrap().try_into().unwrap();
        assert_eq!(copy.0, copy_segregating);
    }
}
