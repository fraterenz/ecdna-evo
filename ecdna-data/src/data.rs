//! The data of interest, such as the ecDNA distribution, its mean, its entropy,
//! the frequency of cells w/ ecDNA etc...
use anyhow::{bail, ensure, Context};
use ecdna_sim::{write2file, NbIndividuals};
use enum_dispatch::enum_dispatch;
use rand::prelude::SliceRandom;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fs;
use std::fs::read_to_string;
use std::iter::repeat;
use std::ops::Deref;
use std::path::Path;

use crate::DNACopy;

const EPSILON_KS_STAT: f32 = 0.00001;

/// Trait to write the data to file
#[enum_dispatch]
pub trait ToFile {
    fn save(&self, path2file: &Path) -> anyhow::Result<()>;
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
                let path2read =
                    fs::read_to_string(path2file).with_context(|| {
                        format!("Cannot read {:#?}", path2file)
                    })?;
                serde_json::from_str(&path2read)
                    .map_err(|e| anyhow::anyhow!(e))
                    .with_context(|| "Cannot load ecDNA distribution")
            }
            _ => panic!("Extension not recognized!"),
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

/// L2 norm between two scalars
pub fn euclidean_distance(val1: f32, val2: f32) -> f32 {
    f32::abs(val1 - val2)
}

#[derive(Debug, Clone)]
pub struct EcDNASummary {
    pub mean: Mean,
    pub frequency: Frequency,
    pub entropy: Entropy,
}

#[derive(Clone, Debug)]
pub struct Data {
    /// Histogram representation of the ecDNA distribution (with `NMinus` cells)
    pub ecdna: EcDNADistribution,
    /// Summary statistics of the ecDNA distribution (mean, frequency and entropy)
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
    pub fn save(&self, path2dir: &Path, filename: &Path) {
        let mut ecdna = path2dir.join("ecdna").join(filename);
        ecdna.set_extension("json");

        self.ecdna.save(&ecdna).expect("Cannot save the ecDNA distribution");

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        self.summary
            .mean
            .save(&mean)
            .expect("Cannot save the ecDNA distribution mean");

        let mut frequency = path2dir.join("frequency").join(filename);
        frequency.set_extension("csv");
        self.summary
            .frequency
            .save(&frequency)
            .expect("Cannot save the frequency");

        let mut entropy = path2dir.join("entropy").join(filename);
        entropy.set_extension("csv");
        self.summary
            .entropy
            .save(&entropy)
            .expect("Cannot save the ecDNA distribution entropy");
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

macro_rules! impl_to_file {
    ($t:ty) => {
        impl ToFile for $t {
            fn save(&self, path2file: &Path) -> anyhow::Result<()> {
                write2file(&[self.0], path2file, None, false)?;
                Ok(())
            }
        }
    };
}

impl_to_file!(Mean);
impl_to_file!(Frequency);
impl_to_file!(Entropy);

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

#[derive(Clone, PartialEq, Debug, Default, Deserialize, Serialize)]
pub struct Entropy(pub f32);

impl Entropy {
    pub fn new(initial_ecdna: &EcDNADistribution) -> anyhow::Result<Self> {
        initial_ecdna.entropy().map(|entropy| Entropy(entropy.0))
    }
}

/// The distribution of ecDNA copies considering the cells w/o any ecDNA copy
/// represented as an histogram.
#[derive(Clone, PartialEq, Eq, Debug, Deserialize, Serialize)]
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
        // do not test small samples because ks is not reliable (ecdf)
        if self.ntot < 10u64 || ecdna.ntot < 10u64 {
            return (1f32, false);
        }

        let mut distance = 0f32;
        // Compare the two empirical cumulative distributions (self) and ecdna
        let mut ecdf = 0f32;
        let mut ecdf_other = 0f32;

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
            if diff - distance >= EPSILON_KS_STAT {
                distance = diff;
            }

            // check if any of the ecdf have reached 1. If it's the case
            // the difference will decrease monotonically and we can stop
            let max_dist = (ecdf - 1.0).abs() <= EPSILON_KS_STAT
                || (ecdf_other - 1.0).abs() <= EPSILON_KS_STAT
                || (distance - 1.0).abs() <= EPSILON_KS_STAT;
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
        self.ntot == 0
    }

    pub fn undersample_data(
        self,
        nb_cells: &NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> anyhow::Result<Data> {
        //! Undersample the ecDNA distribution taking `nb_cells` cells and
        //! roughly respecting the frequencies of the ecDNA copies in the ecDNA
        //! distribution using [`rand::distributions::WeightedIndex`]. So we
        //! sample in a multitype fashion (k classes) and not in a 2 type
        //! fashion `NMinus` vs `NPlus`.
        let ecdna = self.undersample(nb_cells, rng);
        Data::try_from(&ecdna)
    }

    fn undersample(
        self,
        nb_cells: &NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> EcDNADistribution {
        //! Undersample the ecDNA distribution taking roughly sample the
        //! proportion of ecDNA copies in cells found in the tumour, i.e.
        //! sample per ecDNA copies (not `NMinus` vs `NPlus`).
        //!
        //! Sampling is performed without replacement, e.g. the distribution is
        //! updated each time a cell is sampled since the sampled cell is
        //! removed from the population.
        assert!(
            !self.is_empty(),
            "Cannot undersample from empty ecDNA distribution"
        );

        assert!(nb_cells <= &self.nb_cells(), "Cannot undersample with `nb_cells` greater than the cells found in the ecDNA distribution");

        if !self.is_empty() {
            self.sample_distribution(*nb_cells, rng)
        } else {
            self.distribution.into()
        }
    }

    fn sample_distribution(
        self,
        nb_cells: NbIndividuals,
        rng: &mut Pcg64Mcg,
    ) -> Self {
        //! Draw a random sample without replacement from the
        //! `EcDNADistribution` by storing all cells into a `vec`, shuffling it
        //! and taking `nb_cells`.
        // convert the histogram into a vec of cells
        let mut distribution: Vec<DNACopy> =
            Vec::with_capacity(self.nb_cells() as usize);
        distribution.extend(repeat(0u16).take(*self.get_nminus() as usize));
        distribution.extend(self.into_vec_no_minus());

        // shuffle and take the first `nb_cells`
        distribution.shuffle(rng);
        distribution.truncate(nb_cells as usize);
        distribution.into()
    }

    pub fn copies(&self) -> HashSet<DNACopy> {
        //! Get all the ecDNA copies available in the population (the x values
        //! in the histogram).
        self.distribution.keys().copied().collect()
    }

    pub fn get_unique_nplus_copies(&self) -> Option<&DNACopy> {
        //! Get the ecDNA copy number of assuming there is only one cell with ecDNA
        //! (and any number of cells w/o any ecDNA copies). If there are more than
        //! one cell w/ ecDNA, returns `None`.
        assert!(self.ntot == self.nb_nplus() + *self.get_nminus());
        if self.nb_nplus() == 1 {
            return Some(
                self.distribution
                    .keys()
                    .collect::<Vec<&DNACopy>>()
                    .first()
                    .unwrap(),
            );
        }
        None
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
mod tests {
    use super::*;
    use crate::DNACopy;
    use ecdna_sim::NbIndividuals;
    use fake::Fake;
    use quickcheck::Gen;
    use quickcheck_macros::quickcheck;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;
    use test_case::test_case;

    #[test]
    fn ecdna_from_map() {
        let original_data =
            HashMap::from([(0u16, 12u64), (2u16, 1u64), (10u16, 3u64)]);
        let ecdna = EcDNADistribution::from(original_data.clone());
        assert_eq!(ecdna.distribution, original_data);
        assert_eq!(ecdna.ntot, 16u64);
    }

    #[quickcheck]
    fn ecdna_from_vector_twice(original_data: Vec<u16>) -> bool {
        let ecdna = EcDNADistribution::from(original_data.clone());
        ecdna == EcDNADistribution::from(original_data)
    }

    #[test]
    fn ecdna_from_vector() {
        let original_data = vec![0u16, 2u16, 10u16];
        let ntot = original_data.len();
        let ecdna = EcDNADistribution::from(original_data);
        let hist = HashMap::from([(0u16, 1u64), (2u16, 1u64), (10u16, 1u64)]);
        assert_eq!(ecdna.distribution, hist);
        assert_eq!(ecdna.ntot, ntot as NbIndividuals);
    }

    #[test]
    fn ecdna_from_emtpy_vec() {
        let my_vec: Vec<DNACopy> = vec![];
        let dna = EcDNADistribution::from(my_vec);
        assert!(dna.is_empty());
    }

    #[test]
    fn ecdna_from_vector_multiple_values() {
        let original_data = vec![0u16, 2u16, 2u16, 10u16];
        let ecdna = EcDNADistribution::from(original_data);
        let hist = HashMap::from([(0u16, 1u64), (2u16, 2u64), (10u16, 1u64)]);
        assert_eq!(ecdna.distribution, hist);
    }

    #[quickcheck]
    fn ecdna_into_vec_no_nminus(distribution: Vec<u16>) -> bool {
        let mut expected = distribution
            .clone()
            .into_iter()
            .filter(|x| x > &0u16)
            .collect::<Vec<u16>>();
        expected.sort_unstable();

        let mut got =
            EcDNADistribution::from(distribution).into_vec_no_minus();
        got.sort_unstable();
        got == expected
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
    fn ecdna_undersampling_empty() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let ecdna = EcDNADistribution::from(vec![]);
        ecdna.undersample(&3u64, &mut rng);
    }

    #[test]
    #[should_panic]
    fn ecdna_undersampling_more_cells_than_ecdna_distribution() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let ecdna = EcDNADistribution::from(vec![10u16]);
        ecdna.undersample(&3u64, &mut rng);
    }

    #[test]
    fn ecdna_undersampling_nminus() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let ecdna = EcDNADistribution::from(vec![0u16, 0u16]);
        assert_eq!(
            ecdna.undersample(&1u64, &mut rng),
            EcDNADistribution::from(vec![0u16])
        );
    }

    #[ignore]
    #[quickcheck]
    fn ecdna_undersample_sample_reproducible_different_trials(
        seed: u64,
        distribution: ValidEcDNADistributionFixture,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let nb_cells: NbIndividuals = distribution.0.nb_cells() - 1;

        let first =
            distribution.clone().0.sample_distribution(nb_cells, &mut rng);

        let second = distribution.0.sample_distribution(nb_cells, &mut rng);

        first != second
    }

    #[quickcheck]
    fn ecdna_undersample_reproducible(
        seed: u64,
        distribution: ValidEcDNADistributionFixture,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let nb_cells: NbIndividuals = 1;

        let ecdna = distribution.0;
        let first = ecdna.clone().undersample(&nb_cells, &mut rng);

        let mut rng = Pcg64Mcg::seed_from_u64(seed);

        ecdna.undersample(&nb_cells, &mut rng) == first
    }

    #[quickcheck]
    #[ignore]
    fn ecdna_undersample_reproducible_different_trials(
        seed: u64,
        distribution: ValidEcDNADistributionFixture,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let nb_cells: NbIndividuals = distribution.0.nb_cells() - 1;

        let ecdna = distribution.0;
        let first = ecdna.clone().undersample(&nb_cells, &mut rng);

        ecdna.undersample(&nb_cells, &mut rng) != first
    }

    impl fake::Dummy<EcDNADistribution> for EcDNADistribution {
        fn dummy_with_rng<R: Rng + ?Sized>(
            _: &EcDNADistribution,
            rng: &mut R,
        ) -> Self {
            let nb_cells = 1000u16;
            let range = rand::distributions::Uniform::new(0, nb_cells);
            let mut distribution = (0..nb_cells)
                .map(|_| rng.sample(&range))
                .collect::<Vec<DNACopy>>();
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
        seed: u64,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        valid_ecdna.0.undersample(&8u64, &mut rng).nb_cells() == 8u64
    }

    #[quickcheck]
    fn ecdna_distribution_undersampling_check_copies(
        valid_ecdna: ValidEcDNADistributionFixture,
        seed: u64,
    ) -> bool {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        valid_ecdna
            .0
            .clone()
            .undersample(&8u64, &mut rng)
            .copies()
            .is_subset(&valid_ecdna.0.copies())
    }

    #[test]
    fn ecdna_distribution_undersampling_check_copies_one_copy() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let distribution = HashMap::from([(10u16, 2u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };

        assert_eq!(
            ecdna.undersample(&1u64, &mut rng).copies(),
            HashSet::from([10u16]),
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_with_nminus_empty() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let distribution = HashMap::from([(0u16, 10u64)]);
        let ecdna = EcDNADistribution::from(distribution);
        assert_eq!(
            ecdna.undersample(&1u64, &mut rng),
            EcDNADistribution::from(vec![0u16])
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_copies_one_copy_two_samples() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let distribution = HashMap::from([(10u16, 2u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };
        assert_eq!(
            ecdna.undersample(&2u64, &mut rng).copies(),
            HashSet::from([10u16])
        );
    }

    #[test]
    fn ecdna_distribution_undersampling_check_likelihood() {
        let mut rng = Pcg64Mcg::seed_from_u64(26u64);
        let distribution = HashMap::from([(10u16, 10000u64), (1u16, 1u64)]);
        let ntot = distribution.values().sum();
        let ecdna = EcDNADistribution { distribution, ntot };
        assert_eq!(
            ecdna.undersample(&1u64, &mut rng).copies(),
            HashSet::from([10u16])
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
    fn ecdna_ks_distance_is_one_when_no_overlap_in_support(
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

        let (distance, convergence) = x.0.ks_distance(&y);
        (distance - 1f32).abs() <= EPSILON_KS_STAT && convergence
    }

    #[quickcheck]
    fn ecdna_ks_distance_is_point_five_when_semi_overlap_in_support(
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
        (distance - 0.5f32).abs() <= EPSILON_KS_STAT && convergence
    }

    #[quickcheck]
    fn ecdna_ks_distance_fast_convergence(
        x: ValidEcDNADistributionFixture,
    ) -> bool {
        let y = EcDNADistribution::from(vec![1u16; 100]);
        let (_, convergence) = x.0.ks_distance(&y);
        let (_, convergence1) = y.ks_distance(&x.0);
        convergence && convergence1
    }
}
