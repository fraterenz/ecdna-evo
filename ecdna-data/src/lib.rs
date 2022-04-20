pub mod data;
pub mod patient;

#[macro_use]
extern crate derive_builder;

/// Number of ecDNA copies within a cell. We assume that a cell cannot have more
/// than 65535 copies (`u16` is 2^16 - 1 = 65535 copies).
pub type DNACopy = u16;
