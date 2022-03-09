use crate::clap_app::clap_app;
use clap::ArgMatches;
use ecdna_evo::abc::SamplePathsBuilder;
use ecdna_evo::dynamics::{Dynamic, Dynamics};
use ecdna_evo::{DNACopy, NbIndividuals, Parameters, Patient, Rates};

pub fn build_app() -> (Parameters, Rates, Option<Dynamics>, Option<Patient>) {
    //! Build the app by parsing CL arguments with `clap` to build the structs
    //! required by `Simulation` to run the stochastic simulation.
    let matches = clap_app().get_matches();

    // Global arguments first (valid for all subcomands)
    let nb_runs: usize =
        matches.value_of_t("runs").unwrap_or_else(|e| e.exit());
    let init_copies: DNACopy =
        matches.value_of_t("init_copies").unwrap_or_else(|e| e.exit());
    let init_nplus: NbIndividuals =
        matches.value_of_t("init_nplus").unwrap_or_else(|e| e.exit());
    let init_nminus: NbIndividuals =
        matches.value_of_t("init_nminus").unwrap_or_else(|e| e.exit());
    let max_cells: NbIndividuals =
        matches.value_of_t("max_cells").unwrap_or_else(|e| e.exit());
    let subsample: Option<NbIndividuals> =
        matches.value_of_t("undersample").ok();
    let verbosity = {
        match matches.occurrences_of("v") {
            0 => 0_u8,
            1 => 1_u8,
            _ => 2_u8,
        }
    };

    // Then other arguments based on the subcomand used
    let (d, rates, patient_data, parameters) = match matches.subcommand() {
        Some(("simulate", simulate_matches)) => {
            let parameters = Parameters {
                nb_runs,
                max_iter,
                init_copies,
                init_nplus: init_nplus != 0,
                init_nminus,
                verbosity,
                init_iter: 0usize,
                init_time: 0f32,
                subsample,
                tumour_sizes,
            };

            // Quantities of interest that changes for each iteration
            let d = create_dynamics(simulate_matches, &parameters);
            // Rates of the two-type stochastic birth-death process
            let r = rates_from_args(simulate_matches);

            if verbosity > 0 {
                println!("dynamics is some {}", d.is_some());
                if verbosity > 1 {
                    println!("dynamics: {:#?}", d);
                }
            }

            (d, r, None, parameters)
        }
        Some(("abc", abc_matches)) => {
            let parameters = Parameters {
                nb_runs,
                init_copies,
                init_nplus: init_nplus != 0,
                init_nminus,
                verbosity,
                init_iter: 0usize,
                init_time: 0f32,
                subsample,
                tumour_sizes,
            };

            // Rates of the two-type stochastic birth-death process
            let r = rates_from_args(abc_matches);

            let patient = create_patient(abc_matches, verbosity);

            // cannot have quantities with abc
            (None, r, patient, parameters)
        }
        _ => unreachable!(),
    };

    println!("Successfully created App");
    (parameters, rates, d, patient_data)
}

pub fn create_dynamics(
    matches: &ArgMatches,
    parameters: &Parameters,
) -> Option<Dynamics> {
    //! Create dynamics parsing the CL arguments. Modify this function when
    //! adding new type of `Dynamic`.
    if parameters.verbosity > 1 {
        println!("{:#?}", matches)
    }
    let mut dynamics = Vec::with_capacity(5);
    match matches.values_of("dynamics") {
        None => return None,
        Some(kind) => {
            for k in kind {
                if parameters.verbosity > 0 {
                    println!("Adding {} to dynamics", k);
                }
                dynamics.push(Dynamic::new(parameters, k))
            }
        }
    }

    Some(dynamics.into())
}

pub fn create_patient(matches: &ArgMatches, verbosity: u8) -> Option<Patient> {
    let mut paths = PatientPathsBuilder::default();

    if verbosity > 0 {
        println!("Paths to patient's input data: {:#?}", paths);
    }

    if let Some(d) = matches.value_of("distribution_input") {
        if verbosity > 1 {
            println!("Add distribution to paths");
        }
        paths.distribution(d);
    }

    if let Some(m) = matches.value_of("mean_input") {
        if verbosity > 1 {
            println!("Add mean to paths");
        }
        paths.mean(m);
    }

    if let Some(f) = matches.value_of("frequency_input") {
        if verbosity > 1 {
            println!("Add frequency to paths");
        }
        paths.frequency(f);
    }

    if let Some(e) = matches.value_of("entropy_input") {
        if verbosity > 1 {
            println!("Add entropy to paths");
        }
        paths.entropy(e);
    }

    // if let Some(_exp) = matches.value_of("exp_input") {
    //     if verbosity > 1 {
    //         println!("Add exponential to paths");
    //     }
    //     todo!()
    // }

    if verbosity > 0 {
        println!("Paths to patient's input data: {:#?}", paths);
    }

    let name: String = matches.value_of_t("name").unwrap_or_else(|e| e.exit());
    let patient = Patient::load(paths.build().unwrap(), &name, verbosity);
    if verbosity > 0 {
        println!("Patient: {:#?}", patient);
    }

    Some(patient)
}

fn rates_from_args(matches: &ArgMatches) -> Rates {
    // Proliferation rate of the cells w/ ecDNA
    let fitness1: Vec<f32> =
        matches.values_of_t("selection").unwrap_or_else(|e| e.exit());
    // Proliferation rate of the cells w/o ecDNA
    let fitness2 = vec![1f32];

    // Death rate of cells w/ ecDNA
    let death1: Vec<f32> =
        matches.values_of_t("death1").unwrap_or_else(|e| e.exit());
    let death2: Vec<f32> =
        matches.values_of_t("death2").unwrap_or_else(|e| e.exit());

    Rates::new(&fitness1, &fitness2, &death1, &death2)
}
