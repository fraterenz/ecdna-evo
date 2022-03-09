//! Perform the approximate Bayesian computation to infer the most probable
//! fitness and death coefficients from the data.
use crate::data::{Distance, EcDNADistribution, Entropy, Frequency, Mean};
use crate::run::{Ended, Run};
use crate::NbIndividuals;
use anyhow::Context;
use csv;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;
use std::path::{Path, PathBuf};

/// A patient is a collection of data associated to one tumour sequenced one or
/// several times. The timing of the sequencing experiment is referred to by the
/// number of malignant cells present in the whole tumour at time of sequencing.
#[derive(Debug)]
pub struct Patient {
    name: String,
    /// Patient's sequencing data, where the keys are the number of tumour cells
    /// at sequencing.
    pub data: HashMap<NbIndividuals, SequencingData>,
}

impl Patient {
    pub fn load(
        name: &str,
        paths: Vec<SamplePaths>,
        nb_cells: Vec<NbIndividuals>,
        verbosity: u8,
    ) -> Self {
        //! Create a new patient by loading its data for each sequencing experiment
        assert_eq!(paths.len(), nb_cells.len());
        let mut data = HashMap::with_capacity(paths.len());

        for (cells, path) in nb_cells.into_iter().zip(paths.into_iter()) {
            data.insert(cells, SequencingData::load(path, verbosity));
        }

        Patient { name: name.to_owned(), data }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}

/// A collection of `Measurement`s representing the data available for a cancer
/// sample.
#[derive(Debug)]
pub struct SequencingData {
    ecdna: Option<EcDNADistribution>,
    mean: Option<Mean>,
    frequency: Option<Frequency>,
    entropy: Option<Entropy>,
}

impl SequencingData {
    fn load(paths: SamplePaths, verbosity: u8) -> Self {
        //! Load patient's data from paths
        let mut found_one = false;

        let ecdna = {
            match paths.distribution {
                Some(path) => {
                    found_one = true;
                    if verbosity > 0u8 {
                        println!("Loading ecdna distribution");
                    }
                    Some(EcDNADistribution::try_from(&path)
                            .with_context(|| {
                                format!(
                                    "Cannot load the ecDNA distribution from {:#?}",
                                    path
                                )
                            })
                            .unwrap(),
                    )
                }

                None => None,
            }
        };

        let mean = {
            match paths.mean {
                Some(path) => {
                    found_one = true;
                    if verbosity > 0u8 {
                        println!("Loading mean");
                    }
                    Some(
                        Mean::try_from(&path)
                            .with_context(|| {
                                format!(
                                    "Cannot load the mean from {:#?}",
                                    path
                                )
                            })
                            .unwrap(),
                    )
                }
                None => None,
            }
        };

        let frequency = {
            match paths.frequency {
                Some(path) => {
                    found_one = true;
                    if verbosity > 0u8 {
                        println!("Loading frequency");
                    }
                    Some(
                        Frequency::try_from(&path)
                            .with_context(|| {
                                format!(
                                    "Cannot load the frequency from {:#?}",
                                    path
                                )
                            })
                            .unwrap(),
                    )
                }
                None => None,
            }
        };

        let entropy = {
            match paths.entropy {
                Some(path) => {
                    found_one = true;
                    if verbosity > 0u8 {
                        println!("Loading entropy");
                    }
                    Some(
                        Entropy::try_from(&path)
                            .with_context(|| {
                                format!(
                                    "Cannot load the entropy from {:#?}",
                                    path
                                )
                            })
                            .unwrap(),
                    )
                }
                None => None,
            }
        };

        assert!(found_one, "Must provide at least one path to load data");

        SequencingData { ecdna, mean, frequency, entropy }
    }
}

/// Sample's paths to ABC input data
#[derive(Builder, Default)]
#[builder(derive(Debug))]
pub struct SamplePaths {
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
    pub fn run(
        run: &Run<Ended>,
        sequencing_sample: &SequencingData,
        subsample: &Option<NbIndividuals>,
    ) -> ABCResults {
        //! Run the ABC rejection method by comparing the run against the
        //! patient's data
        let nb_samples = 100usize;
        let mut results = ABCResults(Vec::with_capacity(nb_samples));
        if let Some(cells) = subsample {
            // create multiple subsamples of the same run and save the results
            // in the same file `path`. It's ok as long as cells is not too
            // big because deep copies of the ecDNA distribution for each
            // subsample
            let mut rng = SmallRng::from_entropy();
            for i in 0usize..nb_samples {
                // returns new ecDNA distribution with cells NPlus cells (clone)
                let sampled = run.undersample_ecdna(cells, &mut rng, i);
                results
                    .0
                    .push(ABCRejection::run_it(&sampled, sequencing_sample));
            }
        } else {
            results.0.push(ABCRejection::run_it(run, sequencing_sample));
        }
        results
    }

    fn run_it(
        run: &Run<Ended>,
        sequencing_sample: &SequencingData,
    ) -> ABCResult {
        let mut builder = ABCResultBuilder::default();
        if let Some(ecdna) = &sequencing_sample.ecdna {
            builder.ecdna(ecdna.distance(run));
        }

        if let Some(mean) = &sequencing_sample.mean {
            builder.mean(mean.distance(run));
        }

        if let Some(frequency) = &sequencing_sample.frequency {
            builder.frequency(frequency.distance(run));
        }

        if let Some(entropy) = &sequencing_sample.entropy {
            builder.entropy(entropy.distance(run));
        }

        let idx = run.idx.to_string();
        if let Some(run) = run.get_parental_run() {
            builder.parental_idx(*run);
        }

        let mut result =
            builder.build().expect("Cannot run ABC without any data");
        let rates = run.get_rates();
        result.f1 = rates[0];
        result.f2 = rates[1];
        result.d1 = rates[2];
        result.d2 = rates[3];

        result.idx = idx;
        result
    }
}

/// Results of the ABC rejection algorithm, i.e. the posterior distributions of
/// the rates. There is one `ABCResults` for each run.
pub struct ABCResults(Vec<ABCResult>);

#[derive(Builder, Debug, Serialize)]
struct ABCResult {
    #[builder(setter(strip_option), default)]
    parental_idx: Option<usize>,
    #[builder(setter(skip))]
    idx: String,
    #[builder(setter(strip_option), default)]
    ecdna: Option<f32>,
    #[builder(setter(strip_option), default)]
    mean: Option<f32>,
    #[builder(setter(strip_option), default)]
    frequency: Option<f32>,
    #[builder(setter(strip_option), default)]
    entropy: Option<f32>,
    #[builder(setter(skip))]
    f1: f32,
    #[builder(setter(skip))]
    f2: f32,
    #[builder(setter(skip))]
    d1: f32,
    #[builder(setter(skip))]
    d2: f32,
}

impl ABCResults {
    pub fn save(
        &self,
        path2folder: &Path,
        subsamples: &Option<NbIndividuals>,
    ) -> anyhow::Result<()> {
        //! Save the results of the abc inference for a run in a folder abc, where there is one file for each (parental) run and the name of the file corresponds to the parental run.
        let path2save = if subsamples.is_some() {
            path2folder.to_owned().join("samples")
        } else {
            path2folder.to_owned()
        };
        let filename = match self.0[0].parental_idx {
            Some(run) => run.to_string(),
            None => self.0[0].idx.clone(),
        };
        let mut path2file = path2save.join("abc").join(filename);
        path2file.set_extension("csv");
        fs::create_dir_all(path2file.parent().unwrap())
            .expect("Cannot create dir");

        let mut wtr = csv::Writer::from_path(path2file)?;
        for record in self.0.iter() {
            wtr.serialize(record)?;
            wtr.flush()?;
        }
        Ok(())
    }
}
