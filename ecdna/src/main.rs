mod app;
mod clap_app;

use anyhow::Context;
use app::{
    build_config, App, BayesianApp, Config, DynamicalApp, LonditudinalApp,
    Perform, SubsampledApp, Tarball,
};

fn main() {
    let config = build_config();
    let mut app: App = match config {
        Config::Bayesian(bayesian) => BayesianApp::new(bayesian)
            .with_context(|| "Cannot create new bayesian app")
            .unwrap()
            .into(),
        Config::Longitudinal(longitudinal) => {
            LonditudinalApp::new(longitudinal)
                .with_context(|| "Cannot create new longitudinal app")
                .unwrap()
                .into()
        }
        Config::Subsampled(subsampled) => SubsampledApp::new(subsampled)
            .with_context(|| "Cannot create new subsampled app")
            .unwrap()
            .into(),
        Config::Dynamical(dynamical) => DynamicalApp::new(dynamical)
            .with_context(|| "Cannot create new dynamical app")
            .unwrap()
            .into(),
    };

    app.run().unwrap(); // TODO

    std::process::exit(match app.compress() {
        Ok(_) => {
            println!("End simulation");
            0
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            1
        }
    });
}
