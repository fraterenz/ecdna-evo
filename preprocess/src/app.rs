use crate::clap_app::clap_app;
use anyhow::Context;
use clap::ArgMatches;
use ecdna_evo::data::{EcDNADistribution, Entropy, Frequency, Mean};
use ecdna_evo::patient::{Patient, SequencingData};
use ecdna_evo::{DNACopy, NbIndividuals, Rates};
use std::env;
use std::path::PathBuf;

pub struct App {
    pub patient_name: String,
    pub sample_name: String,
    pub size: NbIndividuals,
    pub ecdna: Option<EcDNADistribution>,
    pub mean: Option<Mean>,
    pub frequency: Option<Frequency>,
    pub entropy: Option<Entropy>,
    pub verbosity: u8,
}

pub fn build() -> App {
    //! Parse the CL arguments with `clap` to preprocess the data and create the
    //! input required for the bayesian inferences.
    let matches = clap_app().get_matches();
    let patient_name: String =
        matches.value_of_t("patient").unwrap_or_else(|e| e.exit());
    let sample_name: String =
        matches.value_of_t("sample").unwrap_or_else(|e| e.exit());
    let size: NbIndividuals =
        matches.value_of_t("size").unwrap_or_else(|e| e.exit());
    let verbosity = matches.occurrences_of("verbosity") as u8;

    // load the ecdna distribution
    let ecdna: Option<EcDNADistribution> = matches
        .value_of_t::<String>("distribution")
        .ok()
        .map(PathBuf::from)
        .map(|path| {
            EcDNADistribution::try_from(path.as_ref())
                .with_context(|| {
                    format!(
                        "Cannot load the ecDNA distribution from file {:#?}",
                        path
                    )
                })
                .unwrap()
        });

    let mean: Option<Mean> = matches.value_of_t("mean").ok().map(Mean);
    let frequency: Option<Frequency> =
        matches.value_of_t("frequency").ok().map(Frequency);
    let entropy: Option<Entropy> =
        matches.value_of_t("entropy").ok().map(Entropy);

    App {
        patient_name,
        sample_name,
        size,
        ecdna,
        mean,
        frequency,
        entropy,
        verbosity,
    }
}
