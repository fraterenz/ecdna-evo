//! A timepoint is a quantity computed at the end of the simulation, that is
//! at a single timepoint. Examples are the final frequency of cells with ecDNA,
//! the final mean of ecDNA copies per cell, the final ecDNA distribution.
//!
//! To add more timepoints create new timepoint by implementing the traits
//! `Compute` and `Name`, and modifying `Timepoint::save`.

use crate::simulation::{write2file, Name, ToFile};
use crate::{EcDNADistribution, NbIndividuals, Parameters, Run};
use enum_dispatch::enum_dispatch;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// The main trait for the `Timepoint` which computes the quantity based on the
/// state of the `Run` at the end of the simulation. It allows the communication
/// between the `Timepoint` and the `Run`.
///
/// # How can I implement `Compute`?
/// Types that are `Timepoint` must implement `Compute` which defines how to
/// compute the quantity based on the state of the `Run` at the end of the
/// simulation.
///
/// An example of `Timepoint` computing the mean ecDNA copy number at the end of
/// the `Run`:
///
/// ```no_run
/// use ecdna_evo::{Compute, Run};
///
/// pub struct Mean {
///     mean: f32,
/// }
///
/// impl Compute for Mean {
///     fn compute(&mut self, run: &Run) {
///         self.mean = run.get_ecdna_distr().compute_mean();
///     }
/// }
/// ```

#[derive(Clone, Default, Debug)]
pub struct Timepoints(Vec<Timepoint>);

impl Timepoints {
    pub fn new() -> Timepoints {
        Timepoints(Vec::with_capacity(6))
    }

    pub fn names(&self) -> anyhow::Result<String> {
        //! Create the string "{}_{}" where {} is a timepoint name
        let mut path = String::new();
        for t in self.iter() {
            path.push_str(&format!("{}_", t.get_name()));
        }
        // remove last trailing _
        path.pop().unwrap();
        Ok(path)
    }
}

// impl Iterator for Timepoints {
//     type Item = Timepoint;
//
//     fn next(&mut self) -> Option<Self::Item> {
//
//
//     }
// }

impl Deref for Timepoints {
    type Target = Vec<Timepoint>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Timepoints {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<Timepoint>> for Timepoints {
    fn from(timepoints: Vec<Timepoint>) -> Self {
        Timepoints(timepoints)
    }
}

#[enum_dispatch]
pub trait Compute {
    /// Compute the quantity based on the state of the `Run` at last iteration
    fn compute(&mut self, run: &Run);
}

/// The quantities of interest that we are estimating through simulations. These
/// quantities are computed only at the end of the simulation.
#[enum_dispatch(Compute, Name, ToFile)]
#[derive(Clone, Debug)]
pub enum Timepoint {
    /// ecDNA distribution considering also the cells w/o any ecDNA copy
    EcDNA,
    /// Mean of the ecDNA copies over all cells (w/ and w/o ecDNA) within the
    /// tumour
    Mean,
    /// Frequency of cells w/ ecDNA at the end of the run
    Frequency,
    /// Entropy of the ecDNA distribution computed at end of the run (not
    /// considering cells w/o ecDNA)
    Entropy,
}

/// The ecDNA distribution computed at last iteration by default.
impl Default for Timepoint {
    fn default() -> Self {
        EcDNA::default().into()
    }
}

impl Timepoint {
    pub fn new(params: &Parameters, kind: &str) -> Self {
        match kind {
            "ecdna" | "distribution_input" => Timepoint::EcDNA(EcDNA::new(params)),
            "mean" | "mean_input" => Timepoint::Mean(Mean::new(params)),
            "frequency" | "frequency_input" => Timepoint::Frequency(Frequency::new(params)),
            "entropy" | "entropy_input" => Timepoint::Entropy(Entropy::new(params)),
            _ => panic!("Cannot create time from {}", kind),
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct EcDNA {
    distribution: EcDNADistribution,
    name: String,
}

impl Compute for EcDNA {
    fn compute(&mut self, run: &Run) {
        self.distribution = run.get_ecdna_distr().clone()
    }
}

impl Name for EcDNA {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl From<&EcDNADistribution> for EcDNA {
    fn from(distr: &EcDNADistribution) -> Self {
        EcDNA {
            distribution: distr.clone(),
            name: "ecdna".to_string(),
        }
    }
}

impl From<EcDNADistribution> for EcDNA {
    fn from(distr: EcDNADistribution) -> Self {
        EcDNA {
            distribution: distr,
            name: "ecdna".to_string(),
        }
    }
}

impl EcDNA {
    pub fn new(parameters: &Parameters) -> Self {
        EcDNA::from(EcDNADistribution::new(parameters))
    }

    pub fn get_nplus_cells(&self) -> NbIndividuals {
        self.distribution.get_nplus_cells()
    }

    pub fn get_ecdna_distr(&self) -> &EcDNADistribution {
        &self.distribution
    }
}

impl ToFile for EcDNA {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(
            &self.distribution.create_vector_with_nminus_cells(),
            path2file,
            None,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Mean {
    mean: f32,
    name: String,
}

impl Compute for Mean {
    fn compute(&mut self, run: &Run) {
        self.mean = run.get_ecdna_distr().compute_mean()
    }
}

impl Name for Mean {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl From<f32> for Mean {
    fn from(mean: f32) -> Self {
        Mean {
            mean,
            name: "mean".to_string(),
        }
    }
}

impl Mean {
    pub fn new(parameters: &Parameters) -> Self {
        Mean {
            mean: parameters.init_copies as f32,
            name: "mean".to_string(),
        }
    }

    pub fn get_mean(&self) -> &f32 {
        &self.mean
    }
}

impl ToFile for Mean {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.mean], path2file, None)?;
        Ok(())
    }
}

/// The frequency of cells with ecDNA at last iteration
#[derive(Clone, Debug)]
pub struct Frequency {
    frequency: f32,
    name: String,
}

impl Compute for Frequency {
    fn compute(&mut self, run: &Run) {
        let nplus = run.get_nplus() as f32;
        let nminus = *run.get_nminus() as f32;
        self.frequency = nplus / (nplus + nminus);
    }
}

impl Name for Frequency {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Frequency {
    pub fn new(parameters: &Parameters) -> Self {
        Frequency {
            frequency: parameters.init_nplus as f32
                / ((parameters.init_nplus + parameters.init_nminus) as f32),
            name: "frequency".to_string(),
        }
    }

    pub fn get_frequency(&self) -> &f32 {
        &self.frequency
    }
}

impl ToFile for Frequency {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.frequency], path2file, None)?;
        Ok(())
    }
}

impl From<f32> for Frequency {
    fn from(frequency: f32) -> Self {
        Frequency {
            frequency,
            name: "frequency".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entropy {
    entropy: f32,
    name: String,
}

impl Compute for Entropy {
    fn compute(&mut self, run: &Run) {
        //! Compute the entropy of the ecDNA distribution w/o `NMinus` cells at
        //! the end of the run
        self.entropy = run.get_ecdna_distr().compute_entropy();
    }
}

impl Name for Entropy {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl From<f32> for Entropy {
    fn from(entropy: f32) -> Self {
        Entropy {
            entropy,
            name: "entropy".to_string(),
        }
    }
}

impl Entropy {
    pub fn new(parameters: &Parameters) -> Self {
        Entropy::from(parameters.init_copies as f32)
    }

    pub fn get_entropy(&self) -> &f32 {
        &self.entropy
    }
}

impl ToFile for Entropy {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.entropy], path2file, None)?;
        Ok(())
    }
}
