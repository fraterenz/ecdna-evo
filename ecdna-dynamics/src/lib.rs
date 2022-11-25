pub mod dynamics;
pub mod run;
pub mod segregation;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
