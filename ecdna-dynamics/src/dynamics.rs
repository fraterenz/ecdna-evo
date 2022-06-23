//! The population dynamics of ecDNAs, i.e. keep track of the population status
//! (cells w/ and w/o ecDNA as well as other quantities such as the mean of the
//! ecDNA distribution) over time for each iteration. To add more measurements
//! create new dynamics by implementing the trait `Update`.
//!
//! To add a new dynamical quantity, register it in the enum `Dynamic` and
//! implement the trait `Name` and `Update` and modify `Dynamic::save`.

use anyhow::{bail, Context};
use ecdna_data::data::EcDNADistribution;
use ecdna_sim::event::GillespieTime;
use ecdna_sim::{write2file, NbIndividuals};
use enum_dispatch::enum_dispatch;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

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

/// Trait to write the dynamics to file
#[enum_dispatch]
pub trait Save {
    fn save(&self, path2file: &Path) -> anyhow::Result<()>;
}

/// Remove all rec
#[enum_dispatch]
pub trait Clear {
    fn clear(&mut self);
}

/// The quantities of interest estimated from the state of the `Run` for each
/// iteration. Add new dynamical quantities here.
#[enum_dispatch(Clear, Name, Save, Update)]
#[derive(Clone, Debug)]
pub enum Dynamic {
    /// Number of cells w/ any ecDNA copy per iteration
    NPlus,
    /// Number of cells w/o any ecDNA copy per iteration
    NMinus,
    /// The mean ecDNA copy number per iteration
    MeanDyn,
    /// The variance ecDNA copy number per iteration
    Moments,
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
    pub fn new(
        max_iter: usize,
        initial_ecdna: EcDNADistribution,
        kind: &str,
    ) -> anyhow::Result<Self> {
        match kind {
            "nplus" => Ok(Dynamic::NPlus(NPlus::new(max_iter, initial_ecdna))),
            "nminus" => {
                Ok(Dynamic::NMinus(NMinus::new(max_iter, initial_ecdna)))
            }
            "mean" => {
                MeanDyn::new(max_iter, &initial_ecdna).map(Dynamic::MeanDyn)
            }
            "moments" => {
                Moments::new(max_iter, initial_ecdna).map(Dynamic::Moments)
            }
            "time" => Ok(Dynamic::GillespieT(GillespieT::new(max_iter))),
            _ => panic!("Cannot create dynamic from {}", kind),
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
    name: String,
}

impl Save for NPlus {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.nplus_dynamics, path2file, None, false)?;
        Ok(())
    }
}

impl Name for NPlus {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Clear for NPlus {
    fn clear(&mut self) {
        self.nplus_dynamics.clear()
    }
}

impl NPlus {
    pub fn new(max_iter: usize, initial_ecdna: EcDNADistribution) -> Self {
        let mut nplus_dynamics = Vec::with_capacity(max_iter);
        nplus_dynamics
            .push(initial_ecdna.into_vec_no_minus().len() as NbIndividuals);
        NPlus { nplus_dynamics, name: "nplus".to_string() }
    }

    pub fn store_nplus(&mut self, nplus: NbIndividuals) {
        self.nplus_dynamics.push(nplus)
    }
}

/// Compute the population dynamics of cells w/o ecDNAs.
#[derive(Clone, Debug)]
pub struct NMinus {
    /// Record the number of cells w/o ecDNA for each iteration.
    nminus_dynamics: Vec<NbIndividuals>,
    name: String,
}

impl Save for NMinus {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.nminus_dynamics, path2file, None, false)?;
        Ok(())
    }
}

impl Name for NMinus {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Clear for NMinus {
    fn clear(&mut self) {
        self.nminus_dynamics.clear()
    }
}

impl NMinus {
    pub fn new(max_iter: usize, initial_ecdna: EcDNADistribution) -> Self {
        let mut nminus_dynamics = Vec::with_capacity(max_iter);
        nminus_dynamics.push(*initial_ecdna.get_nminus());
        NMinus { nminus_dynamics, name: "nminus".to_string() }
    }

    pub fn store_nminus(&mut self, nminus: NbIndividuals) {
        self.nminus_dynamics.push(nminus)
    }
}

#[derive(Clone, Debug)]
/// Record the mean of the ecDNA distribution for each iteration.
pub struct MeanDyn {
    mean: Vec<f32>,
    name: String,
}

impl MeanDyn {
    pub fn new(
        max_iter: usize,
        initial_ecdna: &EcDNADistribution,
    ) -> anyhow::Result<Self> {
        let mut mean = Vec::with_capacity(max_iter);
        let initial_mean = initial_ecdna.mean().with_context(|| {
            "Cannot initialize the mean from the initial ecDNA distribution"
        })?;
        mean.push(initial_mean.0);

        Ok(MeanDyn { mean, name: "mean_dynamics".to_string() })
    }

    pub fn store_mean(&mut self, mean: f32) {
        self.mean.push(mean)
    }
}

impl Save for MeanDyn {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.mean, path2file, None, false)?;
        Ok(())
    }
}

impl Name for MeanDyn {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Clear for MeanDyn {
    fn clear(&mut self) {
        self.mean.clear()
    }
}

#[derive(Debug, Clone)]
pub struct Moments {
    /// Record the variance of the ecDNA distribution for each iteration.
    variance: Vec<f32>,
    /// Object to compute the mean in a clever way
    mean: MeanDyn,
    name: String,
}

impl Name for Moments {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Clear for Moments {
    fn clear(&mut self) {
        self.mean.clear();
        self.variance.clear();
    }
}

impl Moments {
    pub fn new(
        max_iter: usize,
        initial_ecdna: EcDNADistribution,
    ) -> anyhow::Result<Self> {
        let mean = MeanDyn::new(max_iter, &initial_ecdna).with_context(||
            "Cannot compute the mean while initializing the variance from the initial ecDNA distribution")?;
        let mut variance = Vec::with_capacity(max_iter);
        let initial_variance = initial_ecdna.variance(Some(*mean.mean.first().unwrap())).with_context(|| {
            "Cannot initialize the variance from the initial ecDNA distribution"
        })?;
        variance.push(initial_variance);

        Ok(Moments { variance, mean, name: "var_dynamics".to_string() })
    }

    pub fn store_moments(&mut self, mean: f32, variance: f32) {
        self.mean.store_mean(mean);
        self.variance.push(variance);
    }
}

impl Save for Moments {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        let file = path2file.file_name().unwrap().to_owned();
        let path2mean =
            PathBuf::from(path2file.parent().unwrap().parent().unwrap())
                .join(self.mean.get_name())
                .join(file);

        self.mean.save(&path2mean)?;
        write2file(&self.variance, path2file, None, false)?;
        Ok(())
    }
}

/// Compute the population dynamics of cells w/ and w/o ecDNAs.
#[derive(Clone, Default, Debug)]
pub struct GillespieT {
    /// Record the number of cells w/ ecDNA for each iteration.
    time: Vec<GillespieTime>,
    name: String,
}

impl Name for GillespieT {
    fn get_name(&self) -> &String {
        &self.name
    }
}

impl Clear for GillespieT {
    fn clear(&mut self) {
        self.time.clear()
    }
}

impl GillespieT {
    pub fn new(max_iter: usize) -> Self {
        let mut time = Vec::with_capacity(max_iter);
        time.push(0f32);
        GillespieT { time, name: "time".to_string() }
    }

    pub fn store_time(&mut self, time: f32) {
        self.time.push(time)
    }

    pub fn get_previous_time(&self) -> anyhow::Result<f32> {
        match self.time.last() {
            None => {
                bail!("Cannot retrieve the gillespie time from the previous iteration")
            }
            Some(&time) => Ok(time),
        }
    }
}

impl Save for GillespieT {
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
