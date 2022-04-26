//! A `Patient` is a collection of `SequencingData` that have been collected and sequenced from the same tumour mass but at different timepoints.
use crate::data::{EcDNADistribution, Entropy, Frequency, Mean};
use anyhow::{anyhow, ensure, Context};
use ecdna_sim::NbIndividuals;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SequencingData {
    #[builder(setter(strip_option), default)]
    pub ecdna: Option<EcDNADistribution>,
    #[builder(setter(into, strip_option), default)]
    pub mean: Option<Mean>,
    #[builder(setter(into, strip_option), default)]
    pub frequency: Option<Frequency>,
    #[builder(setter(into, strip_option), default)]
    pub entropy: Option<Entropy>,
    /// The number of malignant cells in the whole tumour mass when the sample has been collected.
    pub tumour_size: NbIndividuals,
    pub name: String,
}

impl Eq for SequencingData {}

impl Ord for SequencingData {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tumour_size.cmp(&other.tumour_size)
    }
}

impl PartialOrd for SequencingData {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SequencingData {
    fn eq(&self, other: &Self) -> bool {
        self.tumour_size == other.tumour_size
    }
}

impl SequencingDataBuilder {
    /// Check that the `EcDNADistribution` has less cell than the whole tumour mass `tumour_size`.
    fn validate(&self) -> Result<(), String> {
        if let Some(Some(ecdna)) = &self.ecdna {
            if ecdna.nb_cells() > self.tumour_size.unwrap() {
                return Err(format!(
                "Cannot have a sample size {} bigger than the whole tumour mass {}!",
				ecdna.nb_cells(),
				self.tumour_size.unwrap(),
            ));
            }
        } else {
            assert!(
                self.mean.is_some()
                    || self.frequency.is_some()
                    || self.entropy.is_some(),
                     "At least one field among `ecdna` `mean` `frequency` or `entropy` must be present"
            )
        }
        Ok(())
    }
}

impl SequencingData {
    pub fn sample_size(&self) -> Option<NbIndividuals> {
        self.ecdna.as_ref().map(|ecdna| ecdna.nb_cells())
    }

    pub fn get_tumour_size(&self) -> &NbIndividuals {
        &self.tumour_size
    }

    pub fn is_undersampled(&self) -> Option<bool> {
        //! Compute the number of cells
        if let Some(cells) = self.sample_size() {
            let tumour_size = *self.get_tumour_size();
            match cells.cmp(&tumour_size) {
                std::cmp::Ordering::Less => return Some(true),
                std::cmp::Ordering::Equal =>return  Some(false),
                _ => unreachable!("Cannot have more cells in the sample compared to the whole tumour mass!"),
            }
        }
        None
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Patient {
    pub name: String,
    pub samples: Vec<SequencingData>,
    verbosity: u8,
}

impl Patient {
    pub fn new(name: &str, verbosity: u8) -> Self {
        if verbosity > 0 {
            println!(
                "Creating new patient {} with verbosity {}",
                name, verbosity
            );
        }
        Patient { name: name.to_owned(), samples: vec![], verbosity }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn is_sorted(&self) -> bool {
        self.samples
            .windows(2)
            .all(|sample| sample[0].tumour_size <= sample[1].tumour_size)
    }

    pub fn add_sample(
        &mut self,
        sample: SequencingData,
    ) -> anyhow::Result<()> {
        if self.verbosity > 0 {
            println!("Adding sample {} to patient {}", sample.name, self.name);
        }
        ensure!(!self.hash_set().contains(&sample.name));
        self.samples.push(sample);
        self.samples.sort();
        Ok(())
    }

    pub fn save(self) -> anyhow::Result<()> {
        //! Save patient into ./results/preprocessed/self.name.json
        if self.verbosity > 0 {
            println!("Saving patient to {:#?}", self.path2json());
        }
        let patient = serde_json::to_string(&self).with_context(|| {
            format!("Cannot serialize the patient {}", self.name)
        })?;

        fs::write(self.path2json(), &patient)
            .with_context(|| {
                format!("Cannot write to file {:#?}", self.path2json())
            })
            .map_err(|e| anyhow!(e))
    }

    pub fn load(&mut self) -> anyhow::Result<()> {
        if self.verbosity > 0 {
            println!("Loading patient from {:#?}", self.path2json());
        }
        let patient: Patient = serde_json::from_str(
            &fs::read_to_string(&self.path2json()).with_context(|| {
                format!("Cannot load patient from {:#?}", self.path2json())
            })?,
        )
        .with_context(|| {
            format!("Cannot deserialize patient from {:#?}", self.path2json())
        })?;

        if self.verbosity > 1 {
            println!("Samples for patient before loading {:#?}", self.samples);
        }

        for other_sample in patient.samples.into_iter() {
            self.add_sample(other_sample)?;
        }

        if self.verbosity > 1 {
            println!("Samples for patient after loading {:#?}", self.samples);
        }

        if self.verbosity > 0 {
            println!("Patient loaded from {:#?}", self.path2json());
        }
        Ok(())
    }

    pub fn load_from_file(path: &Path, verbosity: u8) -> anyhow::Result<Self> {
        if verbosity > 0 {
            println!("Loading patient from {:#?}", path);
        }
        let mut patient: Patient =
            serde_json::from_str(&fs::read_to_string(path).with_context(
                || format!("Cannot load patient from {:#?}", path),
            )?)
            .with_context(|| {
                format!("Cannot deserialize patient from {:#?}", path)
            })?;

        patient.samples.sort();
        Ok(patient)
    }

    pub fn path2json(&self) -> PathBuf {
        PathBuf::from(format!("results/preprocessed/{}.json", self.name))
    }

    fn hash_set(&self) -> HashSet<String> {
        self.samples.iter().map(|sample| sample.name.clone()).collect()
    }
}
