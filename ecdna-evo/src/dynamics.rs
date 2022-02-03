//! The population dynamics of ecDNAs, i.e. keep track of the population status
//! (cells w/ and w/o ecDNA as well as other quantities such as the mean of the
//! ecDNA distribution) over time for each iteration. To add more measurements
//! create new dynamics by implementing the trait `Update`.
//!
//! To add a new dynamical quantity, register it in the enum `Dynamic` and
//! implement the trait `Name` and `Update` and modify `Dynamic::save`.

use crate::data::{write2file, ToFile};
use crate::gillespie;
use crate::run::{Run, Started};
use crate::{GillespieTime, NbIndividuals, Parameters};
use enum_dispatch::enum_dispatch;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// The main trait for the `Dynamic` which updates the dynamical measurement
/// based on the state of the `Run` for each iteration. It allows the
/// communication between the `Dynamic` and the `Run`.
///
/// # How can I implement `Update`?
/// Types that are `Dynamic` must implement `Update` which defines how to update
/// the dynamical measurement based on the state of the `Run` at each iteration.
///
/// An example of `Dynamic`al measurement keeping track of the number of cells
/// with any copy of ecDNA per iteration:
///
/// ```no_run
/// use ecdna_evo::dynamics::Update;
/// use ecdna_evo::run::{Run, Started};
/// use ecdna_evo::NbIndividuals;
///
/// pub struct NPlus {
///     /// Record the number of cells w/ ecDNA for each iteration.
///     nplus_dynamics: Vec<NbIndividuals>,
/// }
///
/// impl Update for NPlus {
///     fn update(&mut self, run: &Run<Started>) {
///         self.nplus_dynamics.push(run.get_nplus());
///     }
/// }
/// ```

#[enum_dispatch]
pub trait Update {
    /// Update the measurement based on the `run` for each iteration, i.e.
    /// defines how to interact with `Run` to update the quantity of interest
    /// for each iteration.
    fn update(&mut self, run: &Run<Started>);
}

#[derive(Clone, Default, Debug)]
pub struct Dynamics(Vec<Dynamic>);

impl Dynamics {
    pub fn new() -> Dynamics {
        Dynamics(Vec::with_capacity(6))
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

impl Deref for Dynamics {
    type Target = Vec<Dynamic>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Dynamics {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<Dynamic>> for Dynamics {
    fn from(dynamics: Vec<Dynamic>) -> Self {
        Dynamics(dynamics)
    }
}

/// The quantities of interest estimated from the state of the `Run` for each
/// iteration. Add new dynamical quantities here.
#[enum_dispatch(Update, Name, ToFile)]
#[derive(Clone, Debug)]
pub enum Dynamic {
    /// Number of cells w/ any ecDNA copy per iteration
    NPlus,
    /// Number of cells w/o any ecDNA copy per iteration
    NMinus,
    /// The mean ecDNA copy number per iteration
    MeanDyn,
    /// The variance ecDNA copy number per iteration
    Variance,
    /// The Gillespie time for each iteration, that is how
    GillespieT,
}

/// Tracks the cells w/ ecDNA per iteration per default.
impl Default for Dynamic {
    fn default() -> Self {
        NPlus::default().into()
    }
}

impl Dynamic {
    pub fn new(params: &Parameters, kind: &str) -> Self {
        match kind {
            "nplus" => Dynamic::NPlus(NPlus::new(params)),
            "nminus" => Dynamic::NMinus(NMinus::new(params)),
            "mean" => Dynamic::MeanDyn(MeanDyn::new(params)),
            "variance" => Dynamic::Variance(Variance::new(params)),
            "time" => Dynamic::GillespieT(GillespieT::new(params)),
            _ => panic!("Cannot create time from {}", kind),
        }
    }
}

#[enum_dispatch]
pub trait Name {
    fn get_name(&self) -> &String;
}

/// Compute the population dynamics of cells w/ ecDNAs.
#[derive(Clone, Default, Debug)]
pub struct NPlus {
    /// Record the number of cells w/ ecDNA for each iteration.
    nplus_dynamics: Vec<NbIndividuals>,
    pub name: String,
}

impl ToFile for NPlus {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.nplus_dynamics, path2file, None, false)?;
        Ok(())
    }
}

impl Update for NPlus {
    fn update(&mut self, run: &Run<Started>) {
        self.nplus_dynamics.push(run.get_nplus());
    }
}

impl Name for NPlus {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl NPlus {
    pub fn new(parameters: &Parameters) -> Self {
        let mut nplus_dynamics =
            Vec::with_capacity(parameters.max_cells as usize);
        nplus_dynamics.push(parameters.init_nplus as u64);
        NPlus { nplus_dynamics, name: "nplus".to_string() }
    }
}

/// Compute the population dynamics of cells w/o ecDNAs.
#[derive(Clone, Debug)]
pub struct NMinus {
    /// Record the number of cells w/o ecDNA for each iteration.
    nminus_dynamics: Vec<NbIndividuals>,
    pub name: String,
}

impl ToFile for NMinus {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.nminus_dynamics, path2file, None, false)?;
        Ok(())
    }
}

impl Update for NMinus {
    fn update(&mut self, run: &Run<Started>) {
        self.nminus_dynamics.push(*run.get_nminus());
    }
}

impl Name for NMinus {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl NMinus {
    pub fn new(parameters: &Parameters) -> Self {
        let mut nminus_dynamics =
            Vec::with_capacity(parameters.max_cells as usize);
        nminus_dynamics.push(parameters.init_nminus);
        NMinus { nminus_dynamics, name: "nminus".to_string() }
    }
}

#[derive(Clone, Debug)]
pub struct MeanDyn {
    /// Record the mean of the ecDNA distribution for each iteration.
    mean: Vec<f32>,
    pub name: String,
}

impl MeanDyn {
    pub fn new(parameters: &Parameters) -> Self {
        let mut mean = Vec::with_capacity(parameters.max_iter);
        mean.push((parameters.init_nplus as u16) as f32);

        MeanDyn { mean, name: "mean_dynamics".to_string() }
    }
    pub fn ecdna_distr_mean(&self, run: &Run<Started>) -> f32 {
        //! The mean of the ecDNA distribution for the current iteration.
        let ntot = run.get_nminus() + run.get_nplus();
        match gillespie::fast_mean_computation(
            *self.mean.last().unwrap(),
            &run.get_gillespie_event().kind,
            ntot,
        ) {
            Some(mean) => mean,
            // slow version: traverse the whole vector of the ecDNA distribution
            None => run.mean_ecdna(),
        }
    }
}

impl ToFile for MeanDyn {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.mean, path2file, None, false)?;
        Ok(())
    }
}

impl Update for MeanDyn {
    fn update(&mut self, run: &Run<Started>) {
        self.mean.push(self.ecdna_distr_mean(run));
    }
}

impl Name for MeanDyn {
    fn get_name(&self) -> &String {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct Variance {
    /// Record the variance of the ecDNA distribution for each iteration.
    variance: Vec<f32>,
    /// Object to compute the mean in a clever way
    mean: MeanDyn,
    pub name: String,
}

impl Update for Variance {
    fn update(&mut self, run: &Run<Started>) {
        let mean = self.mean.ecdna_distr_mean(run);
        self.mean.update(run);
        self.variance.push(run.variance_ecdna(&mean));
    }
}

impl Name for Variance {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Variance {
    pub fn new(parameters: &Parameters) -> Self {
        let mut variance = Vec::with_capacity(parameters.max_iter);
        variance.push(0f32);
        let mean = MeanDyn::new(parameters);

        Variance { variance, mean, name: "var_dynamics".to_string() }
    }
}

impl ToFile for Variance {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.variance, path2file, None, false)?;
        Ok(())
    }
}

/// Compute the population dynamics of cells w/ and w/o ecDNAs.
#[derive(Clone, Default, Debug)]
pub struct GillespieT {
    /// Record the number of cells w/ ecDNA for each iteration.
    time: Vec<GillespieTime>,
    pub name: String,
}

impl Update for GillespieT {
    fn update(&mut self, run: &Run<Started>) {
        self.time
            .push(self.time.last().unwrap() + run.get_gillespie_event().time);
    }
}

impl Name for GillespieT {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl GillespieT {
    pub fn new(parameters: &Parameters) -> Self {
        let mut time = Vec::with_capacity(parameters.max_cells as usize);
        time.push(0f32);
        GillespieT { time, name: "time".to_string() }
    }
}

impl ToFile for GillespieT {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.time, path2file, None, false)?;
        Ok(())
    }
}

/// Initialize numerical data reading from file. Panics if the conversion from
/// string fails or there is an IO error
pub trait FromFile {
    fn from_file(path2file: &Path, capacity: usize) -> Self;
}
