//! The data of interest, such as the ecDNA distribution, its mean, its entropy,
//! the frequency of cells w/ ecDNA etc...
use crate::run::{Ended, Run};
use crate::{DNACopy, NbIndividuals};
use anyhow::{anyhow, bail, ensure, Context};
use enum_dispatch::enum_dispatch;
use rand::distributions::WeightedIndex;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fs;
use std::fs::read_to_string;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};

/// The main trait for the `Measurement` which defines how to compare the `Run`
/// against the `Measurement`. Types that are `Measurement` must implement
/// `Distance`
///
/// # How can I implement `Distance`?
/// An example of `Measurement` comparing the ecDNA copy number mean of the
/// patient against the one simulated by the run:
///
/// ```no_run
/// use ecdna_evo::data::{euclidean_distance, relative_change, Distance};
/// use ecdna_evo::run::{Ended, Run};
///
/// pub struct Mean(pub f32);
///
/// impl Distance for Mean {
///     fn distance(&self, run: &Run<Ended>) -> f32 {
///         //! The run and the patient's data differ when absolute difference
///         //! between the means considering `NMinus` cells is greater than a
///         //! threshold.
///         relative_change(&self.0, &run.get_mean())
///     }
/// }
/// ```

pub trait Distance {
    /// The data differs from the run when the distance between a `Measurement`
    /// according to a certain metric is higher than a threshold
    fn distance(&self, run: &Run<Ended>) -> f32;
}

/// Trait to write the data to file
#[enum_dispatch]
pub trait ToFile {
    fn save(&self, path2file: &Path) -> anyhow::Result<()>;
}

pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
    endline: bool,
) -> anyhow::Result<()> {
    //! Write vector of float into new file with a precision of 4 decimals.
    //! Write NAN if the slice to write to file is empty.
    fs::create_dir_all(path.parent().unwrap()).expect("Cannot create dir");
    let f = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let mut buffer = BufWriter::new(f);
    if !data.is_empty() {
        if let Some(h) = header {
            writeln!(buffer, "{}", h)?;
        }
        write!(buffer, "{:.4}", data.first().unwrap())?;
        for ele in data.iter().skip(1) {
            write!(buffer, ",{:.4}", ele)?;
        }
        if endline {
            writeln!(buffer)?;
        }
    } else {
        write!(buffer, "{}", f32::NAN)?;
    }
    Ok(())
}

/// Load from file the the ecDNA distribution. The file can be either a json (histogram) or a csv (single-cell ecDNA distribution)
impl TryFrom<&Path> for EcDNADistribution {
    type Error = anyhow::Error;

    fn try_from(path2file: &Path) -> anyhow::Result<Self, Self::Error> {
        let extension = path2file
            .extension()
            .with_context(|| {
                format!("Do not recognize extension of file {:#?}", path2file)
            })
            .unwrap();

        match extension.to_str() {
            Some("csv") => {
                read_csv::<DNACopy>(path2file).map(EcDNADistribution::from)
            }
            Some("json") => {
                serde_json::from_str(&fs::read_to_string(path2file).unwrap())
                    .map_err(|e| anyhow::anyhow!(e))
            }
            _ => panic!("Extension not recognized!"),
        }
    }
}

impl Distance for EcDNADistribution {
    fn distance(&self, run: &Run<Ended>) -> f32 {
        //! The run and the patient's data ecDNA distributions (considering
        //! cells w/o ecDNA) are different if the Kolmogorov-Smirnov statistic
        //! is greater than a certain threshold or if there are less than 10
        //! cells
        // do not compute the KS statistics with less than 10 datapoints
        let too_few_cells =
            self.nb_cells() <= 10u64 || run.get_nb_cells() <= 10u64;
        if too_few_cells {
            f32::INFINITY
        } else {
            self.ks_distance(run.get_ecdna()).0
        }
    }
}

impl TryFrom<&EcDNADistribution> for Mean {
    type Error = &'static str;

    fn try_from(ecdna: &EcDNADistribution) -> Result<Self, Self::Error> {
        if ecdna.is_empty() {
            Err("Mean only accepts non empty ecDNA distributions")
        } else {
            ecdna.mean().map_err(|_| stringify!(e))
        }
    }
}

impl Distance for Mean {
    fn distance(&self, run: &Run<Ended>) -> f32 {
        //! The run and the patient's data differ when the absolute difference
        //! between the means considering `NMinus` cells is greater than a
        //! threshold.
        relative_change(self, run.get_mean())
    }
}

impl TryFrom<&EcDNADistribution> for Frequency {
    type Error = &'static str;

    fn try_from(ecdna: &EcDNADistribution) -> Result<Self, Self::Error> {
        if ecdna.is_empty() {
            Err("Frequency only accepts non empty ecDNA distributions")
        } else {
            ecdna.frequency().map_err(|_| stringify!(e))
        }
    }
}

impl Distance for Frequency {
    fn distance(&self, run: &Run<Ended>) -> f32 {
        relative_change(self, run.get_frequency())
    }
}

impl TryFrom<&EcDNADistribution> for Entropy {
    type Error = &'static str;

    fn try_from(ecdna: &EcDNADistribution) -> Result<Self, Self::Error> {
        if ecdna.is_empty() {
            Err("Entropy only accepts non empty ecDNA distributions")
        } else {
            ecdna.entropy().map_err(|_| stringify!(e))
        }
    }
}

impl Distance for Entropy {
    fn distance(&self, run: &Run<Ended>) -> f32 {
        //! The run and the patient's data differ when the absolute difference
        //! between the entropies considering `NMinus` cells is greater than a
        //! threshold.
        relative_change(self, run.get_entropy())
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

fn parse_csv_string<T>(
    str_data: &str,
    mut data: Vec<T>,
) -> anyhow::Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    for result in str_data.split(',') {
        match result.parse::<T>() {
            Ok(val) => {
                if result == f32::NAN.to_string() {
                    panic!("Found {} in file", result);
                }
                data.push(val);
            }
            _ => bail!("Cannot parse csv: {}", result),
        }
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

#[derive(Debug)]
pub struct EcDNASummary {
    pub mean: Mean,
    pub frequency: Frequency,
    pub entropy: Entropy,
}

pub struct Data {
    pub ecdna: EcDNADistribution,
    pub summary: EcDNASummary,
}

impl TryFrom<&EcDNADistribution> for Data {
    type Error = anyhow::Error;

    fn try_from(ecdna: &EcDNADistribution) -> anyhow::Result<Self> {
        if ecdna.is_empty() {
            Err(anyhow::anyhow!(
                "Data only accepts non empty ecDNA distributions"
            ))
        } else {
            Ok(Data { ecdna: ecdna.clone(), summary: ecdna.summarize() })
        }
    }
}

impl Data {
    pub fn save(
        &self,
        abspath: &Path,
        run_idx: &Path,
        subsample: &Option<NbIndividuals>,
    ) {
        if let Some(cells) = subsample {
            let mut rng = SmallRng::from_entropy();
            let abspath_sampling = abspath.join("samples");
            let data = self.undersample_ecdna(cells, &mut rng);
            data.save_data(&abspath_sampling, run_idx, true);
        }
        self.save_data(abspath, run_idx, true);
    }

    fn save_data(&self, file2path: &Path, filename: &Path, save_ecdna: bool) {
        if save_ecdna {
            let mut ecdna = file2path.join("ecdna").join(filename);
            ecdna.set_extension("json");

            self.ecdna
                .save(&ecdna)
                .expect("Cannot save the ecDNA distribution");
        }

        let mut mean = file2path.join("mean").join(filename);
        mean.set_extension("csv");
        self.summary
            .mean
            .save(&mean)
            .expect("Cannot save the ecDNA distribution mean");

        let mut frequency = file2path.join("frequency").join(filename);
        frequency.set_extension("csv");
        self.summary
            .frequency
            .save(&frequency)
            .expect("Cannot save the frequency");

        let mut entropy = file2path.join("entropy").join(filename);
        entropy.set_extension("csv");
        self.summary
            .entropy
            .save(&entropy)
            .expect("Cannot save the ecDNA distribution entropy");
    }

    fn undersample_ecdna(
        &self,
        nb_cells: &NbIndividuals,
        rng: &mut SmallRng,
    ) -> Self {
        //! Returns a copy of the run with subsampled ecDNA distribution
        let data = self
            .ecdna
            .undersample_data(nb_cells, rng)
            .with_context(|| {
                "Cannot undersample_data from empty ecDNA distribution"
            })
            .unwrap();

        assert_eq!(
                &data.ecdna.nb_cells(),
                nb_cells,
                "Wrong undersampling of the ecDNA distirbution: {} cells expected after sampling, found {}, {:#?}", nb_cells, data.ecdna.nb_cells(), data.ecdna
            );

        data
    }
}

impl ToFile for EcDNADistribution {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        let data = serde_json::to_string(self)
            .expect("Cannot serialize the ecDNA distribution");
        fs::create_dir_all(path2file.parent().unwrap())
            .expect("Cannot create dir");
        fs::write(path2file, data)?;
        Ok(())
    }
}

#[derive(Clone, PartialEq, Debug, Default, Deserialize, Serialize)]
pub struct Mean(pub f32);

impl ToFile for Mean {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None, false)?;
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
#[derive(Clone, PartialEq, Debug, Default, Deserialize, Serialize)]
pub struct Frequency(pub f32);

impl Frequency {
    pub fn new(initial_ecdna: &EcDNADistribution) -> Self {
        let init_nplus = initial_ecdna.nb_nplus() as f32;
        let ntot = initial_ecdna.ntot as f32;
        Frequency(init_nplus / ntot)
    }
}

impl ToFile for Frequency {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None, false)?;
        Ok(())
    }
}

#[derive(Clone, PartialEq, Debug, Default, Deserialize, Serialize)]
pub struct Entropy(pub f32);

impl Entropy {
    pub fn new(initial_ecdna: &EcDNADistribution) -> anyhow::Result<Self> {
        initial_ecdna.entropy().map(|entropy| Entropy(entropy.0))
    }
}

impl ToFile for Entropy {
    fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None, false)?;
        Ok(())
    }
}

/// The distribution of ecDNA copies considering the cells w/o any ecDNA copy
/// represented as an histogram.
#[derive(Clone, PartialEq, Debug, Deserialize, Serialize)]
pub struct EcDNADistribution {
    distribution: HashMap<DNACopy, NbIndividuals>,
    /// Number of total (`NPlus` and `NMinus`) cells
    ntot: NbIndividuals,
}

impl From<HashMap<DNACopy, NbIndividuals>> for EcDNADistribution {
    fn from(distribution: HashMap<DNACopy, NbIndividuals>) -> Self {
        let ntot = distribution.values().sum();
        EcDNADistribution { distribution, ntot }
    }
}

impl From<EcDNADistribution> for Vec<DNACopy> {
    fn from(distr: EcDNADistribution) -> Self {
        let mut distribution = Vec::with_capacity(distr.nb_cells() as usize);
        for (dna, cells) in distr.distribution.into_iter() {
            for _ in 0..cells {
                distribution.push(dna);
            }
        }
        distribution
    }
}

impl From<Vec<DNACopy>> for EcDNADistribution {
    fn from(distr: Vec<DNACopy>) -> Self {
        let mut mapping: HashMap<DNACopy, NbIndividuals> = HashMap::new();
        let ntot = distr.len() as NbIndividuals;
        distr
            .into_iter()
            .for_each(|ecdna| *mapping.entry(ecdna).or_default() += 1);
        EcDNADistribution { distribution: mapping, ntot }
    }
}

/// Defaults to a distribution with a single cell with 1 ecDNA copy.
impl Default for EcDNADistribution {
    fn default() -> Self {
        EcDNADistribution::from(vec![1])
    }
}

impl EcDNADistribution {
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        EcDNADistribution::try_from(path)
    }

    pub fn ks_distance(&self, ecdna: &EcDNADistribution) -> (f32, bool) {
        //! The ks distance represents the maximal absolute distance between the
        //! empirical cumulative distributions of two `EcDNADistribution`s.
        //!
        //! Compute ks distance with `NPlus` and `NMinus` cells, and returns the
        //! distance as well whether the loop stopped before reaching the maximal
        //! allowed copy number `u16::MAX`, which is the case when the distance is
        //! 1 ie maximal, or one of the cumulative distributions have reached 1
        //! and thus the distance can only decrease monotonically.
        //!
        //! Does **not** panic if empty distributions.
        // Start from one which is the max value of the distance
        let mut distance = 0f32;
        // Compare the two empirical cumulative distributions (self) and ecdna
        let mut ecdf = 0f32;
        let mut ecdf_other = 0f32;

        // do not test small samples because ks is not reliable (ecdf)
        if self.ntot < 10u64 || ecdna.ntot < 10u64 {
            return (1f32, false);
        }

        // iter over all ecDNA copies present in both data (self) and ecdna
        for copy in 0u16..u16::MAX {
            if let Some(ecdna_copy) = self.distribution.get(&copy) {
                ecdf += (*ecdna_copy as f32) / (self.ntot as f32);
            }

            if let Some(ecdna_copy) = ecdna.distribution.get(&copy) {
                ecdf_other += (*ecdna_copy as f32) / (ecdna.ntot as f32);
            }

            // store the maximal distance between the two ecdfs
            let diff = (ecdf - ecdf_other).abs();
            if diff - distance > f32::EPSILON {
                distance = diff;
            }

            // check if any of the ecdf have reached 1. If it's the case
            // the difference will decrease monotonically and we can stop
            let max_dist = (ecdf - 1.0).abs() <= f32::EPSILON
                || (ecdf_other - 1.0).abs() <= f32::EPSILON
                || (distance - 1.0).abs() <= f32::EPSILON;
            if max_dist {
                return (distance, true);
            }
        }
        (distance, false)
    }

    pub fn summarize(&self) -> EcDNASummary {
        //! The summary is the mean, the frequency and the entropy.
        //!
        //! Panics if `self` is `empty`.
        let mean =
            Mean::try_from(self).map_err(|e| anyhow::anyhow!(e)).unwrap();
        let frequency =
            Frequency::try_from(self).map_err(|e| anyhow::anyhow!(e)).unwrap();
        let entropy =
            Entropy::try_from(self).map_err(|e| anyhow::anyhow!(e)).unwrap();

        EcDNASummary { mean, frequency, entropy }
    }

    pub fn mean(&self) -> anyhow::Result<Mean> {
        ensure!(
            !self.distribution.is_empty(),
            "Cannot compute mean from empty distribution"
        );
        let sum =
            self.distribution.iter().fold(0u64, |accum, (copy, count)| {
                accum + (*copy as u64) * *count
            }) as f32;
        Ok(Mean(sum / (self.ntot as f32)))
    }

    pub fn variance(self, mean: Option<f32>) -> anyhow::Result<f32> {
        let mean = match mean {
            Some(mean) => mean,
            _ => {
                if let Ok(mean) = self.mean() {
                    mean.0
                } else {
                    panic!(
                        "Cannot compute the mean while computing the variance"
                    );
                }
            }
        };

        let ntot = self.ntot as f32;
        let ecdna: Vec<DNACopy> = self.into();
        Ok(ecdna
            .into_iter()
            .map(|value| {
                let diff = mean - (value as f32);
                diff * diff
            })
            .sum::<f32>()
            / ntot)
    }

    fn frequency(&self) -> anyhow::Result<Frequency> {
        ensure!(
            !self.distribution.is_empty(),
            "Cannot compute frequency from empty distribution"
        );
        let nminus =
            self.distribution.get(&0u16).cloned().unwrap_or_default() as f32;
        Ok(Frequency(1f32 - nminus / (self.ntot as f32)))
    }

    fn entropy(&self) -> anyhow::Result<Entropy> {
        ensure!(
            !self.distribution.is_empty(),
            "Cannot compute entropy from empty distribution"
        );
        Ok(Entropy(
            -self
                .distribution
                .values()
                .map(|&count| {
                    let prob = (count as f32) / (self.ntot as f32);
                    prob * prob.log2()
                })
                .sum::<f32>(),
        ))
    }

    pub fn is_empty(&self) -> bool {
        self.distribution.is_empty()
    }

    pub fn undersample_data(
        &self,
        nb_cells: &NbIndividuals,
        rng: &mut SmallRng,
    ) -> anyhow::Result<Data> {
        let ecdna = self.undersample(nb_cells, rng);
        Data::try_from(&ecdna)
    }

    fn undersample(
        &self,
        nb_cells: &NbIndividuals,
        rng: &mut SmallRng,
    ) -> EcDNADistribution {
        //! Undersample the ecDNA distribution taking roughly sample the proportion of ecDNA copies in cells found in the tumour
        assert!(
            !self.is_empty(),
            "Cannot undersample from empty ecDNA distribution"
        );

        assert!(nb_cells <= &self.nb_cells(), "Cannot undersample with `nb_cells` greater than the cells found in the ecDNA distribution");

        let ecdna = self.distribution.clone();
        // ecdna.remove(&0u16); // remove NMinus cells
        if ecdna.is_empty() {
            return EcDNADistribution::from(ecdna);
        }
        let ecdna =
            ecdna.into_iter().collect::<Vec<(DNACopy, NbIndividuals)>>();

        let mut distribution: HashMap<DNACopy, NbIndividuals> = HashMap::new();
        let sampler =
            WeightedIndex::new(ecdna.iter().map(|item| item.1)).unwrap();

        for _ in 0..*nb_cells {
            *distribution.entry(ecdna[sampler.sample(rng)].0).or_default() +=
                1;
        }

        EcDNADistribution::from(distribution)
    }

    pub fn copies(&self) -> HashSet<DNACopy> {
        //! Get all the ecDNA copies available in the population (the x values
        //! in the histogram).
        self.distribution.keys().copied().collect()
    }

    pub fn nb_cells(&self) -> NbIndividuals {
        self.distribution.values().sum::<NbIndividuals>()
    }

    pub fn nb_nplus(&self) -> NbIndividuals {
        self.nb_cells() - *self.get_nminus()
    }

    pub fn get_nminus(&self) -> &NbIndividuals {
        self.distribution.get(&0u16).unwrap_or(&0u64)
    }

    pub fn into_vec_no_minus(mut self) -> Vec<DNACopy> {
        //! Convert the `EcDNADistribution` into a vec of ecDNA copies where each entry
        //! represents a cell discarding all the cells w/o any ecDNA copy.
        // remove nminus cells
        self.distribution.remove(&0u16);

        let mut distribution = Vec::with_capacity(self.nb_cells() as usize);
        for (dna, cells) in self.distribution.into_iter() {
            for _ in 0..cells {
                distribution.push(dna);
            }
        }
        distribution
    }
}

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
mod tests {
    use super::*;
    use fake::Fake;
    use quickcheck::{quickcheck, Gen};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_case::test_case;

    #[test]
    fn from_map() {
        let original_data =
            HashMap::from([(0u16, 12u64), (2u16, 1u64), (10u16, 3u64)]);
        let ecdna = EcDNADistribution::from(original_data.clone());
        assert_eq!(ecdna.distribution, original_data);
        assert_eq!(ecdna.ntot, 16u64);
    }

    #[test]
    fn from_vector() {
        let original_data = vec![0u16, 2u16, 10u16];
        let ntot = original_data.len();
        let ecdna = EcDNADistribution::from(original_data);
        let hist = HashMap::from([(0u16, 1u64), (2u16, 1u64), (10u16, 1u64)]);
        assert_eq!(ecdna.distribution, hist);
        assert_eq!(ecdna.ntot, ntot as NbIndividuals);
    }

    #[test]
    fn from_vec_for_ecdna_distribution_empty() {
        let my_vec = vec![];
        let dna = EcDNADistribution::from(my_vec);
        assert!(dna.is_empty());
    }

    #[test]
    fn from_vector_multiple_values() {
        let original_data = vec![0u16, 2u16, 2u16, 10u16];
        let ecdna = EcDNADistribution::from(original_data);
        let hist = HashMap::from([(0u16, 1u64), (2u16, 2u64), (10u16, 1u64)]);
        assert_eq!(ecdna.distribution, hist);
    }

    #[test_case(vec![0u16, 1u16, 2u16] => Mean(1f32)  ; "Balanced input")]
    #[test_case(vec![1u16, 1u16, 1u16] => Mean(1f32)  ; "Constant input")]
    #[test_case(vec![0u16, 2u16] => Mean(1f32)  ; "Unbalanced input")]
    fn ecdna_mean_1(ecdna: Vec<u16>) -> Mean {
        EcDNADistribution::from(ecdna).mean().unwrap()
    }

    #[test]
    #[should_panic]
    fn ecdna_mean_empty() {
        EcDNADistribution::from(vec![]).mean().unwrap();
    }

    #[test_case(vec![1u16, 1u16, 1u16] => Frequency(1f32)  ; "NPlus")]
    #[test_case(vec![0u16] => Frequency(0f32)  ; "Nminus")]
    #[test_case(vec![0u16, 2u16] => Frequency(0.5f32)  ; "Mixed")]
    fn ecdna_frequency(ecdna: Vec<u16>) -> Frequency {
        EcDNADistribution::from(ecdna).frequency().unwrap()
    }

    #[test]
    #[should_panic]
    fn ecdna_frequency_empty() {
        EcDNADistribution::from(vec![]).frequency().unwrap();
    }

    // #[test_case(vec![0u16, 1u16, 2u16] => Mean(1f32)  ; "Balanced input")]
    // #[test_case(vec![1u16, 1u16, 1u16] => Mean(1f32)  ; "Constant input")]
    // #[test_case(vec![0u16, 2u16] => Mean(1f32)  ; "Unbalanced input")]
    // fn ecdna_mean_1(ecdna: Vec<u16>) -> Mean {
    //     EcDNADistribution::from(ecdna).mean()
    // }

    #[test]
    #[should_panic]
    fn ecdna_entropy_empty() {
        EcDNADistribution::from(vec![]).entropy().unwrap();
    }

    #[test]
    #[should_panic]
    fn data_try_from_empty_ecdna() {
        Data::try_from(&EcDNADistribution::from(vec![])).unwrap();
    }

    #[test]
    #[should_panic]
    fn ecdna_distribution_undersampling_empty() {
        let mut rng = SmallRng::from_entropy();
        let ecdna = EcDNADistribution::from(vec![]);
        ecdna.undersample(&3u64, &mut rng);
    }

    #[test]
    #[should_panic]
    fn ecdna_distribution_undersampling_more_cells_than_ecdna_distribution() {
        let mut rng = SmallRng::from_entropy();
        let ecdna = EcDNADistribution::from(vec![10u16]);
        ecdna.undersample(&3u64, &mut rng);
    }

    impl fake::Dummy<EcDNADistribution> for EcDNADistribution {
        fn dummy_with_rng<R: Rng + ?Sized>(
            _: &EcDNADistribution,
            rng: &mut R,
        ) -> Self {
            let range = rand::distributions::Uniform::new(0, 20);
            let mut distribution =
                (0..19).map(|_| rng.sample(&range)).collect::<Vec<DNACopy>>();
            distribution.push(0);
            distribution.into()
        }
    }

    // Both `Clone` and `Debug` are required by `quickcheck`
    #[derive(Debug, Clone)]
    struct ValidEcDNADistributionFixture(pub EcDNADistribution);

    impl quickcheck::Arbitrary for ValidEcDNADistributionFixture {
        fn arbitrary(_g: &mut Gen) -> Self {
            // TODO should use g instead of rng but Gen does not seem to impl Rng trait??
            let mut rng = SmallRng::from_entropy();
            let ecdna =
                EcDNADistribution::from(vec![]).fake_with_rng(&mut rng);
            Self(ecdna)
        }
    }

    #[quickcheck]
    fn ecdna_distribution_undersampling_check_cells_tot(
        valid_ecdna: ValidEcDNADistributionFixture,
    ) -> bool {
        let mut rng = SmallRng::from_entropy();
        valid_ecdna.0.undersample(&8u64, &mut rng).nb_cells() == 8u64
    }

    #[quickcheck]
    fn ecdna_distribution_undersampling_check_copies(
        valid_ecdna: ValidEcDNADistributionFixture,
    ) -> bool {
        let mut rng = SmallRng::from_entropy();
        valid_ecdna
            .0
            .undersample(&8u64, &mut rng)
            .copies()
            .is_subset(&valid_ecdna.0.copies())
    }

    #[test]
    fn ecdna_distribution_undersampling_check_copies_one_copy() {
        let mut rng = SmallRng::from_entropy();
        let distribution = HashMap::from([(10u16, 2u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };

        assert_eq!(
            ecdna.undersample(&1u64, &mut rng).copies(),
            HashSet::from([10u16]),
        );

        assert_eq!(
            ecdna.undersample(&1u64, &mut rng),
            EcDNADistribution::from(vec![10u16])
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_with_nminus_empty() {
        let mut rng = SmallRng::from_entropy();
        let distribution = HashMap::from([(0u16, 10u64)]);
        let ecdna = EcDNADistribution::from(distribution);
        assert_eq!(
            ecdna.undersample(&1u64, &mut rng),
            EcDNADistribution::from(vec![0u16])
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_copies_one_copy_two_samples() {
        let mut rng = SmallRng::from_entropy();
        let distribution = HashMap::from([(10u16, 2u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };
        assert_eq!(
            ecdna.undersample(&2u64, &mut rng).copies(),
            HashSet::from([10u16])
        );
        assert_eq!(
            ecdna.undersample(&2u64, &mut rng),
            EcDNADistribution::from(vec![10u16, 10u16])
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_likelihood() {
        let mut rng = SmallRng::from_entropy();
        let distribution = HashMap::from([(10u16, 10000u64), (1u16, 1u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };
        assert_eq!(
            ecdna.undersample(&1u64, &mut rng).copies(),
            HashSet::from([10u16])
        );

        assert_eq!(
            ecdna.undersample(&1u64, &mut rng),
            EcDNADistribution::from(vec![10u16])
        );
    }

    #[test]
    fn ecdna_ks_distance_empty() {
        let x = EcDNADistribution::from(vec![]);
        let y = EcDNADistribution::from(vec![]);
        assert_eq!(x.ks_distance(&y), (1f32, false));
        assert_eq!(y.ks_distance(&x), (1f32, false));
        assert_eq!(x.ks_distance(&x), (1f32, false));
    }

    #[test]
    fn ecdna_ks_distance_small_samples() {
        let x = EcDNADistribution::from(vec![1, 2]);
        let y = EcDNADistribution::from(vec![1, 2]);
        assert_eq!(x.ks_distance(&y), (1f32, false));
        assert_eq!(y.ks_distance(&x), (1f32, false));
        assert_eq!(x.ks_distance(&x), (1f32, false));
    }

    #[quickcheck]
    fn ecdna_ks_distance_same_data(x: ValidEcDNADistributionFixture) -> bool {
        let (distance, _) = x.0.ks_distance(&x.0);
        (distance - 0f32).abs() <= f32::EPSILON
    }

    #[quickcheck]
    fn ecdna_ks_distance_max_copy_number_u16(
        mut x: ValidEcDNADistributionFixture,
        y: ValidEcDNADistributionFixture,
    ) -> bool {
        x.0.distribution.insert(u16::MAX, 3u64);
        let (_, convergence) = x.0.ks_distance(&y.0);
        convergence
    }

    #[quickcheck]
    fn ecdna_ks_distance_same_data_shifted_by_1(
        x: ValidEcDNADistributionFixture,
    ) -> bool {
        // https://github.com/daithiocrualaoich/kolmogorov_smirnov/blob/master/src/test.rs#L474
        let y_min = *x.0.distribution.keys().max().unwrap() + 1;
        let y: EcDNADistribution =
            x.0.distribution
                .iter()
                .map(|(k, val)| (y_min + *k, *val))
                .collect::<HashMap<DNACopy, NbIndividuals>>()
                .into();

        assert_eq!(y.ntot, 20);
        let (distance, convergence) = x.0.ks_distance(&y);
        println!("{}", distance);
        (distance - 1f32).abs() <= f32::EPSILON && convergence
    }

    #[quickcheck]
    fn ecdna_ks_distance_same_data_shifted_by_1_added_support(
        x: ValidEcDNADistributionFixture,
    ) -> bool {
        // https://github.com/daithiocrualaoich/kolmogorov_smirnov/blob/master/src/test.rs#L474
        let y_min = *x.0.distribution.keys().max().unwrap() + 1;
        // Add all the original items back too.
        let mut y: HashMap<DNACopy, NbIndividuals> =
            HashMap::with_capacity(2usize * x.0.ntot as usize);

        for (k, v) in x.0.distribution.iter() {
            y.insert(*k + y_min, *v);
            y.insert(*k, *v);
        }

        let y: EcDNADistribution = y.into();

        assert_eq!(y.ntot, x.0.ntot * 2);
        let (distance, convergence) = x.0.ks_distance(&y);
        (distance - 0.5f32).abs() <= f32::EPSILON && convergence
    }

    #[quickcheck]
    fn ecdna_ks_distance_is_one_div_length_for_sample_with_additional_low_value(
        x: ValidEcDNADistributionFixture,
    ) -> bool {
        // Add a extra sample of early weight to ys.
        let mut ys = x.clone();
        let nminus = ys.0.distribution.entry(0u16).or_insert(0u64);
        *nminus += 1;
        ys.0.ntot += 1;
        assert_eq!(ys.0.ntot, x.0.ntot + 1);

        let (distance, convergence) = x.0.ks_distance(&ys.0);

        let expected = match x.0.distribution.get(&0u16) {
            Some(&copy) => {
                (copy as f32 + 1.0) / ys.0.ntot as f32
                    - (copy as f32) / (x.0.ntot as f32)
            }
            None => 1.0 / ys.0.ntot as f32,
        };

        (distance - expected).abs() <= f32::EPSILON && convergence
    }
}
