//! A `Patient` is a collection of `SequencingData` that have been collected and sequenced from the same tumour mass but at different timepoints.
use crate::data::{EcDNADistribution, Entropy, Frequency, Mean};
use crate::NbIndividuals;
use anyhow::{anyhow, ensure, Context};
use serde::{Deserialize, Serialize};
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
            if cells == tumour_size {
                return Some(false);
            } else if cells < tumour_size {
                return Some(true);
            } else {
                unreachable!("Cannot have more cells in the sample compared to the whole tumour mass!")
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

    pub fn add_sample(&mut self, sample: SequencingData) {
        if self.verbosity > 0 {
            println!("Adding sample {} to patient {}", sample.name, self.name);
        }
        self.samples.push(sample)
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
        let mut patient: Patient = serde_json::from_str(
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

        self.samples.append(&mut patient.samples);

        if self.verbosity > 1 {
            println!("Samples for patient after loading {:#?}", self.samples);
        }

        if self.verbosity > 0 {
            println!("Patient loaded from {:#?}", self.path2json());
        }
        Ok(())
    }

    pub fn path2json(&self) -> PathBuf {
        PathBuf::from(format!("results/preprocessed/{}.json", self.name))
    }
}
