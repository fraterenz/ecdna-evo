//! Implementation of the individual-based stochastic computer simulations of
//! the ecDNA population dynamics, assuming an exponential growing population of
//! tumour cells in a well-mixed environment. Simulation are carried out using
//! the Gillespie algorithm.
use crate::dynamics::{Dynamics, Name};
use crate::run::Run;
use crate::{Config, Patient, Rates};
use anyhow::{anyhow, Context};
use chrono::Utc;
use flate2::write::GzEncoder;
use flate2::Compression;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Perform multiple independent simulations of tumour growth in parallel using
/// `rayon` parallel iter API.
pub struct Simulation;

impl<'sim, 'run> Simulation {
    pub fn run(
        parameters: Config,
        rates: Rates,
        dynamics: Option<Dynamics>,
        patient: Option<Patient>,
    ) -> anyhow::Result<()> {
        //! Run in parallel `nb_runs` independent simulations of tumour growth.
        //! The arguments `patient` and `quantities` define whether to simulate
        //! tumour growth or ABC:
        //!
        //! 1. simulate tumour growth: `patient` must be `None` and `quantities`
        //! must be `Some`
        //!
        //! 2. ABC: `patient` must be `Some` and quantities must be `None`.
        Simulation::info(&parameters);

        let (experiment, patient_name) = (
            Simulation::create_path(&parameters, &rates),
            &patient.as_ref().map(|p| p.get_name()),
        );

        let (abspath, relapath) = (
            env::current_dir()
                .unwrap()
                .join("results")
                .join(experiment.clone()),
            PathBuf::from("results").join(experiment),
        );

        (0..parameters.nb_runs)
            .into_par_iter()
            .progress_count(parameters.nb_runs as u64)
            .for_each(|idx| {
                // simulate the first timepoint
                let nb_cells =
                    parameters.tumour_sizes.all_sizes.first().unwrap();
                let seq_data =
                    patient.as_ref().map(|p| p.data.get(nb_cells).unwrap());
                let mut run = Run::new(idx, &parameters, &rates)
                    .simulate(dynamics.clone(), nb_cells)
                    .save(
                        &abspath,
                        &dynamics,
                        &seq_data,
                        &parameters.subsample,
                        patient_name,
                    )
                    .unwrap();

                // simulate other timepoints if present
                for t in parameters.tumour_sizes.all_sizes.iter().skip(1) {
                    run = run
                        .continue_simulation(t)
                        .save(
                            &abspath,
                            &dynamics,
                            &patient
                                .as_ref()
                                .map(|p| p.samples.get(t).unwrap()),
                            &parameters.subsample,
                            patient_name,
                        )
                        .unwrap();
                }
            });

        println!("{} End simulating {} runs", Utc::now(), &parameters.nb_runs);
        println!("{} Results saved in {:#?}", Utc::now(), abspath);

        Simulation::tarballs(&parameters, patient_name, relapath, dynamics);
        Ok(())
    }

    fn info(parameters: &Config) {
        println!(
            "{} Simulating with {} cores {} runs, each of them with {} cells and max iterations {} and {:#?} timpoints",
            Utc::now(),
            rayon::current_num_threads(),
            parameters.nb_runs,
            parameters.max_cells,
            parameters.max_iter,
            parameters.final_tumour_size(),
            parameters.max_iter,
            parameters.tumour_sizes.all_sizes,
        );

        if parameters.verbosity > 1 {
            println!("{} {:#?}", Utc::now(), parameters);
        }

        if parameters.verbosity > 0 {
            println!(
                "{} Launching {} runs in parallel",
                Utc::now(),
                parameters.nb_runs
            );
        }
    }

    fn tarballs(
        parameters: &Config,
        patient_name: &Option<&str>,
        relapath: PathBuf,
        dynamics: Option<Dynamics>,
    ) {
        let relapath2save = {
            if let Some(name) = patient_name {
                if parameters.verbosity > 0 {
                    println!("{} Creating tarball for abc", Utc::now());
                }
                relapath.join(name)
            } else if dynamics.is_some() {
                assert!(
                    patient_name.is_none(),
                    "Cannot have patient and dynamics at the same time"
                );
                relapath.join("dynamics")
            } else {
                relapath
            }
        };

        // subsamples: save the rates and values
        if parameters.subsample.is_some() {
            Simulation::tarball_samples(
                parameters,
                patient_name,
                &relapath2save,
            )
        } else {
            Simulation::tarball(
                parameters,
                patient_name,
                &relapath2save,
                dynamics,
            )
        }

        Simulation::tarball(parameters, &None, &relapath2save, None);
    }

    fn tarball_samples(
        parameters: &Config,
        patient_name: &Option<&str>,
        relapath: &Path,
    ) {
        // subsamples extras
        let path2multiple_samples = relapath.join("samples");
        Simulation::tarball(
            parameters,
            patient_name,
            &path2multiple_samples,
            None,
        );
    }

    fn tarball(
        parameters: &Config,
        patient_name: &Option<&str>,
        parent_dir: &Path,
        dynamics: Option<Dynamics>,
    ) {
        if let Some(dynamics) = &dynamics {
            for d in dynamics.iter() {
                if parameters.verbosity > 0 {
                    println!("{} Creating tarball for dynamics", Utc::now());
                }
                Simulation::compress_results(
                    d.get_name(),
                    parent_dir,
                    parameters.verbosity,
                )
                .expect("Cannot compress dynamics");
            }
        }

        if patient_name.is_some() {
            Simulation::compress_results(
                "abc",
                parent_dir,
                parameters.verbosity,
            )
            .expect("Cannot compress abc results");
        } else {
            for name in ["ecdna", "mean", "frequency", "entropy"] {
                if parameters.verbosity > 0 {
                    println!("{} Creating tarball", Utc::now());
                }

                Simulation::compress_results(
                    name,
                    parent_dir,
                    parameters.verbosity,
                )
                .unwrap();
            }
        }
    }

    fn create_path(parameters: &Config, rates: &Rates) -> PathBuf {
        //! Resturns the paths where to store the data. The hashmap has keys as
        //! the quantities stored, and values as the dest and source of where to
        //! compress the data.
        // longitudinal studies are carried out when there are several sequencing
        // data available for the same tumour at different timepoints.
        let mut my_path = format!(
            "{}runs_{}cells_{}_{}1_{}2_{}undersample",
            parameters.nb_runs,
            parameters.max_cells,
            rates.fitness1,
            rates.death1,
            rates.death2,
            parameters.subsample.unwrap_or_default()
        );
        for sizes in parameters.tumour_sizes.all_sizes.iter() {
            my_path.push_str(&format!("_{}timepoint", sizes));
        }
        my_path.into()
    }

    fn compress_results(
        kind: &str,
        basepath: &Path,
        verbosity: u8,
    ) -> anyhow::Result<()> {
        let dest = basepath.join(kind);
        let src = env::current_dir().unwrap().join(&dest);
        Simulation::compress_dir(&dest, &src, verbosity).with_context(
            || format!("Cannot compress {:#?} into {:#?}", &src, &dest),
        )?;
        Ok(())
    }

    fn compress_dir(
        dest_path: &Path,
        src_path_dir: &Path,
        verbosity: u8,
    ) -> anyhow::Result<()> {
        //! Compress the directory where all runs are saved into tarball at the
        //! same level of the saved runs.
        let mut dest_path_archive = dest_path.to_owned();
        dest_path_archive.set_extension("tar.gz");

        // open stream, create encoder to compress and create tar builder to
        // create tarball
        let tar_gz = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest_path_archive)
            .with_context(|| {
                format!(
                    "Error while opening the stream {:#?}",
                    &dest_path_archive
                )
            })?;
        let enc = GzEncoder::new(tar_gz, Compression::default());
        let mut tar = tar::Builder::new(enc);
        // append recursively all the runs into the archive that is first created in the
        // working and then moved to the location of the runs `src_path_dir`
        tar.append_dir_all(&dest_path_archive, src_path_dir)
            .with_context(|| {
                format!(
                    "Cannot append files to tar archive {:#?} from source {:#?} ",
                    &dest_path_archive, &src_path_dir
                )
            })
            .and_then(|()| {
                if verbosity > 1 {
                    println!(
                        "{} Gzip {:#?} into {:#?}",
                        Utc::now(),
                        src_path_dir,
                        &dest_path_archive,
                    );
                }
                fs::remove_dir_all(src_path_dir)
                    .with_context(|| format!("Cannot remove directory {:#?}", &src_path_dir))
            })?;
        Ok(())
    }
}

/// Generate anyhow error if `val` is `f32::NAN`
pub fn find_nan(val: f32) -> anyhow::Result<f32> {
    if val.is_nan() {
        return Err(anyhow!("Found NaN value!"));
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nan() {
        match find_nan(0f32) {
            Ok(v) => assert!((v - 0f32).abs() < f32::EPSILON),
            Err(_) => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn test_find_nan_panics() {
        match find_nan(f32::NAN) {
            Ok(v) => println!("{}", v),

            Err(_) => panic!(),
        }
    }
}
