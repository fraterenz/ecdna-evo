mod abc;
mod app;
mod clap_app;

#[macro_use]
extern crate derive_builder;
#[cfg(test)]
extern crate quickcheck_macros;

use anyhow::Context;
use app::{build_config, App, BayesianApp, Config, DynamicalApp, Perform};
use chrono::Utc;

use crate::app::PreprocessApp;

fn main() {
    let config = build_config();
    let app: App = match config {
        Config::Bayesian(bayesian) => BayesianApp::new(bayesian)
            .with_context(|| "Cannot create new bayesian app")
            .unwrap()
            .into(),
        Config::Dynamical(dynamical) => DynamicalApp::new(dynamical)
            .with_context(|| "Cannot create new dynamical app")
            .unwrap()
            .into(),
        Config::Preprocess(preprocess) => PreprocessApp::new(preprocess)
            .with_context(|| "Cannot create new preprocessing app")
            .unwrap()
            .into(),
    };

    println!("{} Starting the simulation", Utc::now());

    std::process::exit(match app.run() {
        Ok(_) => {
            println!("{} End simulation", Utc::now(),);
            0
        }
        Err(err) => {
            eprintln!("{} Error: {:?}", Utc::now(), err);
            1
        }
    });
}
