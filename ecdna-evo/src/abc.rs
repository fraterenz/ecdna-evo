//! Perform the approximate Bayesian computation to infer the most probable
//! fitness and death coefficients from the data.
//!
//! To add a new `Datum` create new datum by implementing the traits
//! `Differs` and `FromFile`, and modifying `Patient::load`.
use crate::simulation::{write2file, EcDNADistribution, Name, Run};
use crate::timepoints::{Compute, EcDNA, Entropy, Frequency, Mean, Timepoint, Timepoints};
use crate::{DNACopy, Parameters, Quantities, QuantitiesBuilder};
use anyhow::Context;
use enum_dispatch::enum_dispatch;
use kolmogorov_smirnov as ks;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::ops::Index;
use std::path::{Path, PathBuf};

/// The main trait for the `Datum` which defines how to compare the patient's
/// data against the `Timepoint` of the stochastic run. It allows the
/// communication between the `Datum` and the `Timepoint`.
///
/// # How can I implement `Differs`?
/// Types that are `Datum` must implement `Differs` which defines how to compare
/// the patient's data against the `Timepoint` of the stochastic run.
///
/// An example of `Datum` comparing the ecDNA copy number mean of the patient
/// against the one simulated by the run:
///
/// ```no_run
/// use ecdna_evo::abc::{euclidean_distance, Differs};
/// use ecdna_evo::timepoints::{Mean, Timepoint};
///
/// // Datum is a wrapper of `Timepoint`
/// pub struct MeanPatient(Mean);
///
/// impl Differs for MeanPatient {
///     fn is_different(&self, timepoint: &Timepoint) -> (bool, f32) {
///         if let Timepoint::Mean(mean) = timepoint {
///             let distance = euclidean_distance(*self.0.get_mean(), *mean.get_mean());
///             return (distance >= 2f32, distance);
///         }
///         panic!("Timepoint must be a mean");
///     }
/// }
/// ```

#[enum_dispatch]
pub trait Differs {
    /// The data differs from the run when the distance between a `Datum`
    /// according to a certain metric is higher than a threshold
    fn is_different(&self, timepoint: &Timepoint) -> (bool, f32);
}

/// A collection of `Datum` representing the data available for a cancer
/// patient.
#[derive(Debug)]
pub struct Patient(HashMap<String, Datum>);

impl<'a> Index<&'a str> for Patient {
    type Output = Datum;

    fn index(&self, key: &'a str) -> &Datum {
        &self.0[key]
    }
}

impl Patient {
    pub fn load(paths: PatientPaths) -> Self {
        //! Load patient's data from paths
        let mut data = HashMap::new();
        if let Some(path) = &paths.distribution {
            data.insert(
                "ecdna".to_string(),
                Datum::EcDNAPatient(EcDNAPatient::from_file(path)),
            );
        }

        if let Some(path) = &paths.mean {
            data.insert("mean".to_string(), MeanPatient::from_file(path).into());
        }

        if let Some(path) = &paths.frequency {
            data.insert(
                "frequency".to_string(),
                FrequencyPatient::from_file(path).into(),
            );
        }

        if let Some(path) = &paths.entropy {
            data.insert(
                "entropy".to_string(),
                EntropyPatient::from_file(path).into(),
            );
        }

        assert!(
            !data.is_empty(),
            "Must provide at least one path to load data"
        );
        Patient(data)
    }

    pub fn create_quantities(&self, params: &Parameters) -> Quantities {
        //! Create `Quantities` with only `Timepoints` from the patient's data.
        let mut timepoints: Timepoints = Timepoints::new();
        for datum in self.0.values() {
            match datum {
                Datum::EcDNAPatient(_) => timepoints.push(EcDNA::new(params).into()),
                Datum::MeanPatient(_) => timepoints.push(Mean::new(params).into()),
                Datum::FrequencyPatient(_) => timepoints.push(Frequency::new(params).into()),
                Datum::EntropyPatient(_) => timepoints.push(Entropy::new(params).into()),
            }
        }

        // A `Patient` can only have some `Timepoints` not `Dynamics` for now
        QuantitiesBuilder::default()
            .timepoints(Some(timepoints))
            .build()
            .unwrap()
    }
}

/// Patient's paths to ABC input data
#[derive(Builder, Default)]
#[builder(derive(Debug))]
pub struct PatientPaths {
    #[builder(setter(into, strip_option), default)]
    distribution: Option<PathBuf>,
    #[builder(setter(into, strip_option), default)]
    mean: Option<PathBuf>,
    #[builder(setter(into, strip_option), default)]
    frequency: Option<PathBuf>,
    #[builder(setter(into, strip_option), default)]
    entropy: Option<PathBuf>,
    // #[builder(setter(into, strip_option), default)]
    // exponential: Option<PathBuf>,
}

/// Patient's data
#[enum_dispatch(Differs)]
#[derive(Debug)]
pub enum Datum {
    EcDNAPatient,
    MeanPatient,
    FrequencyPatient,
    EntropyPatient,
}

/// Perform the ABC rejection algorithm for one run to infer the most probable
/// values of the rates based on the patient's data.
///
/// ABC infer the most probable values of the birth-death rates (proliferation
/// rates and death rates) by comparing the summary statistics of the run
/// generated by the birth-death process against the summary statistics of the
/// patient's data.
///
/// When testing multiple statistics with ABC, save runs only if all statistics
/// pass the tests
pub struct ABCRejection;

impl ABCRejection {
    pub fn run(r: &Run, timepoints: &mut Timepoints, patient: &Patient) -> ABCResults {
        ABCResults(
            timepoints
                .iter_mut()
                .map(|timepoint| {
                    timepoint.compute(r);
                    let datatype = timepoint.get_name();
                    let (different, distance) = patient[datatype].is_different(timepoint);
                    (datatype.clone(), (!different as u8, distance))
                })
                .collect::<HashMap<String, (u8, f32)>>(),
        )
    }
}

/// Results of the ABC rejection algorithm, i.e. the posterior distributions of
/// the rates. There is one `ABCResults` for each run.
pub struct ABCResults(HashMap<String, (u8, f32)>);

impl ABCResults {
    pub fn save(&self, run: &Run, path2folder: &Path) -> bool {
        //! Save the results of the abc inference for a run generating the
        //! following files:
        //!
        //! 1. metadata: stores whether the statistics that are similar (1)
        //! or different (0) for ABC (ground truth vs simulation) for each run
        //!
        //! 2. values: the values of the summary statistics for each run
        //!
        //! 3. all_rates: the rates of all runs, also those that have been
        //! rejected and thus do not contribute to the posterior distribution
        //!
        //! 4. rates: the posterior distribution of the rates, here the rates
        //! are only those that contribute to the posterior distribution as
        //! opposed to file all_rates
        let header: String = self
            .0
            .keys()
            .fold(String::new(), |accum, stat| accum + stat + ",");

        // write metadata
        write2file(
            &self
                .0
                .values()
                .map(|(is_different, _)| is_different)
                .collect::<Vec<&u8>>(),
            &path2folder.join("metadata").join(run.filename()),
            Some(&header),
        )
        .unwrap();

        write2file(
            &self
                .0
                .values()
                .map(|(_, distance)| distance)
                .collect::<Vec<&f32>>(),
            &path2folder.join("values").join(run.filename()),
            Some(&header),
        )
        .unwrap();

        // save all runs to plot the histograms for each individual statistic
        // Save the proliferation and death rates of the cells w/ ecDNA and
        // w/o ecDNA respectively
        write2file(
            &run.get_rates(),
            &path2folder.join("all_rates").join(run.filename()),
            Some("f1,f2,d1,d2"),
        )
        .unwrap();

        // write rates only if all stats pass the test
        if self.positive() {
            write2file(
                &run.get_rates(),
                &path2folder.join("rates").join(run.filename()),
                None,
            )
            .unwrap();
            true
        } else {
            false
        }
    }

    pub fn positive(&self) -> bool {
        //! A positive result means that the run is similar to the patient's
        //! data
        self.0.values().all(|&result| result.0 > 0u8)
    }
}

/// Initialize numerical data reading from file. Panics if the conversion from
/// string fails or there is an IO error
pub trait FromFile {
    fn from_file(path2file: &Path) -> Self;
}

#[derive(Debug)]
pub struct EcDNAPatient(EcDNA);

impl EcDNAPatient {
    pub fn get_ecdna_distr(&self) -> &EcDNADistribution {
        self.0.get_ecdna_distr()
    }
}

impl FromFile for EcDNAPatient {
    fn from_file(path2file: &Path) -> Self {
        EcDNAPatient(EcDNA::from(EcDNADistribution::from(
            read_csv::<DNACopy>(path2file)
                .with_context(|| format!("Cannot load the distribution from {:#?}", path2file))
                .unwrap(),
        )))
    }
}

impl Differs for EcDNAPatient {
    fn is_different(&self, timepoint: &Timepoint) -> (bool, f32) {
        //! The run and the patient's data ecDNA distributions (considering
        //! cells w/o ecDNA) are different if the Kolmogorov-Smirnov statistic
        //! is greater than a certain threshold or if there are less than 10
        //! cells
        if let Timepoint::EcDNA(distr) = timepoint {
            // do not compute the KS statistics with less than 10 datapoints
            let distance = {
                if self.0.get_nplus_cells() <= 10 || distr.get_nplus_cells() <= 10 {
                    f32::INFINITY
                } else {
                    self.get_ecdna_distr().ks_distance(distr.get_ecdna_distr())
                }
            };
            return (distance >= 0.02f32, distance);
        }
        panic!("Timepoint must be a ecdna distribution");
    }
}

#[derive(Debug)]
pub struct MeanPatient(Mean);

impl FromFile for MeanPatient {
    fn from_file(path2file: &Path) -> Self {
        MeanPatient(Mean::from(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the mean from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        ))
    }
}

impl Differs for MeanPatient {
    fn is_different(&self, timepoint: &Timepoint) -> (bool, f32) {
        if let Timepoint::Mean(mean) = timepoint {
            let change = relative_change(*self.0.get_mean(), *mean.get_mean());
            return (change > 0.25, change);
        }
        panic!("Timepoint must be a mean");
    }
}

#[derive(Debug)]
pub struct FrequencyPatient(Frequency);

impl FromFile for FrequencyPatient {
    fn from_file(path2file: &Path) -> Self {
        FrequencyPatient(Frequency::from(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the mean from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        ))
    }
}

impl Differs for FrequencyPatient {
    fn is_different(&self, timepoint: &Timepoint) -> (bool, f32) {
        // does not make sense to compute the KS statistics with less than 10 datapoints
        if let Timepoint::Frequency(freq) = timepoint {
            let change = relative_change(*self.0.get_frequency(), *freq.get_frequency());
            return (change > 0.25, change);
        }
        panic!("Timepoint must be a mean");
    }
}

#[derive(Debug)]
pub struct EntropyPatient(Entropy);

impl FromFile for EntropyPatient {
    fn from_file(path2file: &Path) -> Self {
        EntropyPatient(Entropy::from(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the entropy from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        ))
    }
}

impl Differs for EntropyPatient {
    fn is_different(&self, timepoint: &Timepoint) -> (bool, f32) {
        if let Timepoint::Entropy(entropy) = timepoint {
            let change = relative_change(*self.0.get_entropy(), *entropy.get_entropy());
            return (change > 0.25, change);
        }
        panic!("Timepoint must be entropy");
    }
}

fn read_csv<T>(path2data: &Path) -> anyhow::Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let csv: String = read_to_string(path2data)?
        .parse()
        .with_context(|| format!("Cannot read csv from {:#?}", path2data))?;
    if csv.is_empty() {
        panic!("Found empty csv file {:#?}", path2data)
    }
    parse_data(csv)
}

fn parse_data<T>(str_data: String) -> anyhow::Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let data: Vec<T> = Vec::new();
    parse_csv_string(&str_data, data)
}

fn parse_csv_string<T>(str_data: &str, mut data: Vec<T>) -> anyhow::Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    for result in str_data.split(',') {
        let val = result.parse::<T>().unwrap();
        if result == f32::NAN.to_string() {
            panic!("Found {} in file", result);
        }
        data.push(val);
    }
    Ok(data)
}

/// Relative change between two scalars
pub fn relative_change(x1: f32, x2: f32) -> f32 {
    (x1 - x2).abs() / x1
}

/// L2 norm between two scalars
pub fn euclidean_distance(val1: f32, val2: f32) -> f32 {
    f32::abs(val1 - val2)
}

/// Kolmogorov-Smirnov distance between two discrete distributions
pub fn ks_distance(dist1: &[DNACopy], dist2: &[DNACopy]) -> f32 {
    ks::test(dist1, dist2, 0.95).statistic as f32
}
