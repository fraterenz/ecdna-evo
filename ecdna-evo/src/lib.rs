//! Evolutionary models of extrachromosomal DNA (ecDNA).
//!
//! There are two different ways of using this library:
//!
//! 1. simulate exponential tumour growth using birth-death process, where the
//! process is driven by ecDNA, and store some quantities of interest
//! `timepoints` and `dynamics`
//!
//! 2. perform approximate Bayesian computation (ABC) to infer from the data the
//! most probable values of the fitness and the deaths coefficients of cells w/
//! and w/o ecDNA
//!
//! # Simulation example
//! The simulation of tumour growth using Gillespie algorithm.
//! ```no_run
//! use ecdna_evo::{
//!     Dynamic, Parameters, Quantities, QuantitiesBuilder, Rates, Simulation, Timepoint,
//!     Timepoints,
//! };
//!
//! // Configure the simulation with parameters and rates
//! // 1. parameters such as the number of iterations, number of cells
//! let params = Parameters::default();
//! // 2. rates of the process: the neutral dynamics (no selection) by default
//! let rates = Rates::default();
//!
//! // Define the quantities to be simulated
//! let quantities: Quantities = QuantitiesBuilder::default()
//!     .timepoints(Timepoints::from(vec![Timepoint::new(&params, "ecdna")]))
//!     .dynamics(Some(vec![Dynamic::new(&params, "nplus")]))
//!     .build()
//!     .unwrap();
//!
//! // Run the simulation
//! Simulation::run(params, rates, Some(quantities), None);
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
//! );
//!
//! // Run the ABC inference to determine the parameters for the `patient`
//! Simulation::run(params, rates, None, Some(patient));
//! ```
pub mod abc;
pub mod dynamics;
mod gillespie;
pub mod simulation;
pub mod timepoints;

#[doc(inline)]
pub use crate::abc::Patient;
#[doc(inline)]
pub use crate::dynamics::{Dynamic, Update};
#[doc(inline)]
pub use crate::gillespie::{GillespieTime, NbIndividuals, Rates};
#[doc(inline)]
pub use crate::simulation::{
    DNACopy, EcDNADistribution, Parameters, Quantities, QuantitiesBuilder, Run, Simulation, ToFile,
};
#[doc(inline)]
pub use crate::timepoints::{Compute, Frequency, Mean, Timepoint, Timepoints};

#[macro_use]
extern crate approx;
#[macro_use]
extern crate derive_builder;
