mod app;
mod clap_app;

use crate::app::build;
use anyhow::Context;
use app::App;
use ecdna_evo::patient::{Patient, SequencingData, SequencingDataBuilder};
use std::fs;
use std::process;

fn new_sample(app: App) -> SequencingData {
    let mut builder = SequencingDataBuilder::default();
    builder.tumour_size(app.size).name(app.sample_name);
    if let Some(distribution) = app.ecdna {
        let summary = distribution.summarize();
        builder
            .ecdna(distribution)
            .mean(summary.mean)
            .frequency(summary.frequency)
            .entropy(summary.entropy);
    } else {
        if let Some(mean) = app.mean {
            builder.mean(mean);
        }
        if let Some(freq) = app.frequency {
            builder.frequency(freq);
        }

        if let Some(entropy) = app.entropy {
            builder.entropy(entropy);
        }
    }

    builder.build().unwrap()
}

fn main() {
    let app = build();

    println!(
        "Adding sample {} to patient {}",
        app.sample_name, app.patient_name
    );

    let mut patient = Patient::new(&app.patient_name, app.verbosity);

    // if patient exists load it, else create new file
    if patient.path2json().exists() {
        if app.verbosity > 0 {
            println!(
                "Adding sample {} to an existing patient {}",
                app.sample_name, app.patient_name
            )
        }
        if app.verbosity > 1 {
            println!(
                "Loading patient {} from {:#?}",
                app.patient_name,
                patient.path2json()
            );
        }
        patient
            .load()
            .with_context(|| {
                format!("Cannot load patient from {:#?}", patient.path2json())
            })
            .unwrap();
    } else {
        if app.verbosity > 0 {
            println!("Saving new patient");
        }
        fs::create_dir_all(patient.path2json().parent().unwrap()).unwrap();
    }

    let new_sample = new_sample(app);
    let (name, path) = (patient.name.clone(), patient.path2json());

    patient.add_sample(new_sample);

    process::exit(match patient.save() {
        Ok(_) => {
            println!(
                "The new/updated data for patient {} is in {:#?}\n\
                Use this file as input for the bayesian inferences.",
                name, path
            );
            0
        }
        Err(e) => {
            eprintln!(
                "Error, cannot save patient to {:#?} due to:\n{}",
                path, e
            );
            1
        }
    })
}
