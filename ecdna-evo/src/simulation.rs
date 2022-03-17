//! Implementation of the individual-based stochastic computer simulations of
//! the ecDNA population dynamics, assuming an exponential growing population of
//! tumour cells in a well-mixed environment. Simulation are carried out using
//! the Gillespie algorithm.
// use crate::config::{Bayesian, Config};
// use crate::dynamics::{Dynamics, Name};
// use crate::run::Run;
// use crate::{Parameters, Patient, Rates};
use anyhow::{anyhow, Context};
// use chrono::Utc;
// use flate2::write::GzEncoder;
// use flate2::Compression;
// use indicatif::ParallelProgressIterator;
// use rayon::prelude::{IntoParallelIterator, ParallelIterator};
// use std::env;
// use std::fs;
// use std::path::{Path, PathBuf};
