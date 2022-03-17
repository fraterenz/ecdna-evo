//! Evolutionary models of extrachromosomal DNA (ecDNA).
//!
//! There are two different ways of using this library:
//!
//! 1. simulate exponential tumour growth using birth-death process, where the
//! process is driven by ecDNA, and store some quantities of interest
//!
//! 2. perform approximate Bayesian computation (ABC) to infer from the data the
//! most probable values of the fitness and the deaths coefficients of cells w/
//! and w/o ecDNA
//!
//! # Simulation example
//! The simulation of tumour growth using Gillespie algorithm.
//! ```no_run
//! use ecdna_evo::dynamics::{Dynamic, Dynamics};
//! use ecdna_evo::{Parameters, Rates, Simulation};
//!
//! // Configure the simulation with parameters and rates
//! // 1. parameters such as the number of iterations, number of cells
//! let params = Parameters::default();
//! // 2. rates of the process: the neutral dynamics (no selection) by default
//! let rates = Rates::default();
//!
//! // Define the quantities to be simulated (this is optional)
//! let dynamics = Dynamics::from(vec![Dynamic::new(&params, "nplus")]);
//!
//! // Run the simulation
//! Simulation::run(params, rates, Some(dynamics), None);
//! ```
//!
//! # ABC example
//! The Bayesian framework to infer the parameters from the patient's data.
//! ```no_run
//! use ecdna_evo::abc::PatientPathsBuilder;
//! use ecdna_evo::{Parameters, Patient, Rates, Simulation};
//! use std::path::PathBuf;
//!
//! // Configure the simulation with parameters and rates
//! // 1. parameters such as the number of iterations, number of cells
//! let params = Parameters::default();
//! // 2. rates of the process: the neutral dynamics (no selection) by default
//! let rates = Rates::default();
//!
//! // Load the patient's data used to run ABC, in this case only the ecDNA
//! // distribution
//! let patient = Patient::load(
//!     PatientPathsBuilder::default()
//!         .distribution("path2ecdna_distribution")
//!         .build()
//!         .unwrap(),
//!     "PatientName",
//!     params.verbosity,
//! );
//!
//! // Run the ABC inference to determine the parameters for the `patient`
//! Simulation::run(params, rates, None, Some(patient));
//! ```
// pub mod abc;
pub mod abc;
pub mod data;
pub mod dynamics;
mod gillespie;
pub mod patient;
pub mod run;

#[doc(inline)]
pub use crate::gillespie::{GillespieTime, NbIndividuals, Rates};
#[doc(inline)]
pub use crate::patient::Patient;
#[doc(inline)]
pub use crate::run::{DNACopy, Run};
// #[doc(inline)]
// pub use crate::simulation::Simulation;

#[macro_use]
extern crate derive_builder;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
