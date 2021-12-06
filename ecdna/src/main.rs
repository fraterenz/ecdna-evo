mod app;
mod clap_app;

use crate::app::build_app;
use ecdna_evo::simulation::Simulation;

fn main() {
    // parameters defines how to run the simulation, the dynamics are the
    // quantities of interest that change for each iteration (saved in results),
    // the timepoints are the quantities of interest that do not change for each
    // iteration and are computed at last iteration (saved in results)
    let (parameters, rates, quantities, patient_data) = build_app();
    std::process::exit(
        match Simulation::run(parameters, rates, quantities, patient_data) {
            Ok(_) => {
                println!("End simulation");
                0
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                1
            }
        },
    );
}
