//! The data of interest, such as the ecDNA distribution, its mean, its entropy,
//! the frequency of cells w/ ecDNA etc...
use crate::run::Run;
use crate::simulation::{write2file, ToFile};
use crate::{DNACopy, EcDNADistribution, NbIndividuals, Parameters};
use anyhow::Context;
use kolmogorov_smirnov as ks;
use std::fs::read_to_string;
use std::ops::Deref;
use std::path::Path;

/// The main trait for the `Measurement` which defines how to compare the `Run`
/// against the `Measurement`. Types that are `Measurement` must implement
/// `Differs`
///
/// # How can I implement `Differs`?
/// An example of `Measurement` comparing the ecDNA copy number mean of the
/// patient against the one simulated by the run:
///
/// ```no_run
/// use ecdna_evo::data::{euclidean_distance, relative_change, Differs};
/// use ecdna_evo::Run;
///
/// pub struct Mean(pub f32);
///
/// impl Differs for Mean {
///     fn is_different(&self, run: &Run) -> (bool, f32) {
///         //! The run and the patient's data differ when absolute difference
///         //! between the means considering `NMinus` cells is greater than a
///         //! threshold.
///         let change = relative_change(&self.0, &run.data.mean);
///         return (change > 0.25, change);
///     }
/// }
/// ```

pub trait Differs {
    /// The data differs from the run when the distance between a `Measurement`
    /// according to a certain metric is higher than a threshold
    fn is_different(&self, run: &Run) -> (bool, f32);
}

/// Initialize numerical data reading from file. Panics if the conversion from
/// string fails or there is an IO error
pub trait FromFile {
    fn from_file(path2file: &Path) -> Self;
}

impl FromFile for EcDNA {
    fn from_file(path2file: &Path) -> Self {
        EcDNA(EcDNADistribution::from(
            read_csv::<DNACopy>(path2file)
                .with_context(|| format!("Cannot load the distribution from {:#?}", path2file))
                .unwrap(),
        ))
    }
}

impl Differs for EcDNA {
    fn is_different(&self, run: &Run) -> (bool, f32) {
        //! The run and the patient's data ecDNA distributions (considering
        //! cells w/o ecDNA) are different if the Kolmogorov-Smirnov statistic
        //! is greater than a certain threshold or if there are less than 10
        //! cells
        // do not compute the KS statistics with less than 10 datapoints
        let distance = {
            if self.len() <= 10usize || run.data.ecdna.len() <= 10usize {
                f32::INFINITY
            } else {
                ks::test(self, &run.data.ecdna, 0.95).statistic as f32
            }
        };
        (distance >= 0.02f32, distance)
    }
}

impl FromFile for Mean {
    fn from_file(path2file: &Path) -> Self {
        Mean(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the mean from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        )
    }
}

impl Differs for Mean {
    fn is_different(&self, run: &Run) -> (bool, f32) {
        //! The run and the patient's data differ when the absolute difference
        //! between the means considering `NMinus` cells is greater than a
        //! threshold.
        let change = relative_change(self, &run.data.mean);
        (change > 0.25, change)
    }
}

impl FromFile for Frequency {
    fn from_file(path2file: &Path) -> Self {
        Frequency(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the mean from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        )
    }
}

impl Differs for Frequency {
    fn is_different(&self, run: &Run) -> (bool, f32) {
        //! The run and the patient's data differ when the absolute difference
        //! between the frequencies considering `NMinus` cells is greater than a
        //! threshold.
        let change = relative_change(self, &run.data.frequency);
        (change > 0.25, change)
    }
}

impl FromFile for Entropy {
    fn from_file(path2file: &Path) -> Self {
        Entropy(
            *read_csv::<f32>(path2file)
                .with_context(|| format!("Cannot load the entropy from {:#?}", path2file))
                .unwrap()
                .first()
                .unwrap(),
        )
    }
}

impl Differs for Entropy {
    fn is_different(&self, run: &Run) -> (bool, f32) {
        //! The run and the patient's data differ when the absolute difference
        //! between the entropies considering `NMinus` cells is greater than a
        //! threshold.
        let change = relative_change(self, &run.data.entropy);
        (change > 0.25, change)
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
        let val = result.parse::<T>().expect("Error while parsing csv");
        if result == f32::NAN.to_string() {
            panic!("Found {} in file", result);
        }
        data.push(val);
    }
    Ok(data)
}

/// Relative change between two scalars
pub fn relative_change(x1: &f32, &x2: &f32) -> f32 {
    (x1 - x2).abs() / x1
}

/// L2 norm between two scalars
pub fn euclidean_distance(val1: f32, val2: f32) -> f32 {
    f32::abs(val1 - val2)
}

pub struct Data {
    pub ecdna: EcDNA,
    pub mean: Mean,
    pub frequency: Frequency,
    pub entropy: Entropy,
}

impl Data {
    pub fn new(ecdna: EcDNA, mean: Mean, frequency: Frequency, entropy: Entropy) -> Self {
        Data {
            ecdna,
            mean,
            frequency,
            entropy,
        }
    }

    pub fn save(&self, file2path: &Path, filename: &Path, save_ecdna: bool) {
        if save_ecdna {
            let ecdna = file2path.join("ecdna").join(filename);
            write2file(&self.ecdna, &ecdna, None).expect("Cannot save the ecDNA distribution");
        }

        let mean = file2path.join("mean").join(filename);
        write2file(&[*self.mean], &mean, None).expect("Cannot save the ecDNA distribution mean");

        let frequency = file2path.join("frequency").join(filename);
        write2file(&[*self.frequency], &frequency, None).expect("Cannot save the frequency");

        let entropy = file2path.join("entropy").join(filename);
        write2file(&[*self.entropy], &entropy, None)
            .expect("Cannot save the ecDNA distribution entropy");
    }
}

#[derive(Clone, Default, Debug)]
pub struct EcDNA(pub EcDNADistribution);

impl EcDNA {
    pub fn get_ecdna_distr(&self) -> &EcDNADistribution {
        &self.0
    }

    pub fn ntot(&self) -> NbIndividuals {
        self.0.len().try_into().unwrap()
    }
}

impl ToFile for EcDNA {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&self.0, path2file, None)?;
        Ok(())
    }
}

impl Deref for EcDNA {
    type Target = EcDNADistribution;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct Mean(pub f32);

impl Mean {
    pub fn new(parameters: &Parameters) -> Self {
        Mean(parameters.init_copies as f32)
    }

    pub fn get_mean(&self) -> &f32 {
        &self.0
    }
}

impl ToFile for Mean {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None)?;
        Ok(())
    }
}

macro_rules! impl_deref {
    ($t:ty) => {
        impl Deref for $t {
            type Target = f32;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

impl_deref!(Mean);
impl_deref!(Frequency);
impl_deref!(Entropy);

/// The frequency of cells with ecDNA at last iteration
#[derive(Clone, Debug)]
pub struct Frequency(pub f32);

impl Frequency {
    pub fn new(parameters: &Parameters) -> Self {
        let init_nplus = parameters.init_nplus as u64;
        Frequency(init_nplus as f32 / ((init_nplus + parameters.init_nminus) as f32))
    }

    pub fn get_frequency(&self) -> &f32 {
        &self.0
    }
}

impl ToFile for Frequency {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Entropy(pub f32);

impl Entropy {
    pub fn new(parameters: &Parameters) -> Self {
        Entropy(parameters.init_copies as f32)
    }

    pub fn get_entropy(&self) -> &f32 {
        &self.0
    }
}

impl ToFile for Entropy {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None)?;
        Ok(())
    }
}
