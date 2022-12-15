use std::path::PathBuf;

use chrono::Utc;
use indicatif::ParallelProgressIterator;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ssa::process::Process;

use crate::NB_RESTARTS;

pub trait Simulate {
    fn run(self: Box<Self>) -> anyhow::Result<()>;
}
