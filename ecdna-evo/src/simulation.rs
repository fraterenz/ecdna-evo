//! Implementation of the individual-based stochastic computer simulations of
//! the ecDNA population dynamics, assuming an exponential growing population of
//! tumour cells in a well-mixed environment. Simulation are carried out using
//! the Gillespie algorithm.
use crate::dynamics::{Dynamics, Name};
use crate::run::Run;
use crate::{Parameters, Patient, Rates};
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
        parameters: Parameters,
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

        // create path: relative path (used to creating the tarball) and
        // absolute path, for dynamics and abc
        let experiment = Simulation::create_path(&parameters, &rates);
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
                Run::new(idx, parameters, &rates)
                    .simulate(dynamics.clone())
                    .save(&abspath, &dynamics, &patient)
                    .unwrap()
            });

        println!("{} End simulating {} runs", Utc::now(), parameters.nb_runs);
        println!("{} Results saved in {:#?}", Utc::now(), abspath);

        Simulation::tarballs(&parameters, patient, relapath, dynamics);
        Ok(())
    }

    fn info(parameters: &Parameters) {
        println!(
            "{} Simulating with {} cores {} runs, each of them with {} cells and max iterations {}",
            Utc::now(),
            rayon::current_num_threads(),
            parameters.nb_runs,
            parameters.max_cells,
            parameters.max_iter
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
        parameters: &Parameters,
        patient: Option<Patient>,
        relapath: PathBuf,
        dynamics: Option<Dynamics>,
    ) {
        // ABC
        if let Some(ref p) = patient {
            let relapath_abc = relapath.join(p.name.clone());
            if parameters.verbosity > 0 {
                println!("{} Creating tarball for abc", Utc::now());
            }

            // subsamples: save the rates and values
            if parameters.subsample.is_some() {
                Simulation::tarball_samples(
                    parameters,
                    &patient,
                    &relapath_abc,
                )
            } else {
                Simulation::tarball(parameters, &patient, &relapath_abc, None)
            }
        } else {
            // dynamics
            if dynamics.is_some() {
                assert!(
                    patient.is_none(),
                    "Cannot have patient and dynamics at the same time"
                );
                let relapath_d = relapath.join("dynamics");
                return Simulation::tarball(
                    parameters,
                    &None,
                    &relapath_d,
                    dynamics,
                );
            }

            //subsamples: save the ecdna, mean, frequency and entropy for
            //each subsample
            if parameters.subsample.is_some() {
                Simulation::tarball_samples(parameters, &patient, &relapath);
            }

            Simulation::tarball(parameters, &None, &relapath, None);
        }
    }

    fn tarball_samples(
        parameters: &Parameters,
        patient: &Option<Patient>,
        relapath: &Path,
    ) {
        // subsamples extras
        let path2multiple_samples = relapath.join("samples");
        Simulation::tarball(parameters, patient, &path2multiple_samples, None);
    }

    fn tarball(
        parameters: &Parameters,
        patient: &Option<Patient>,
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

        if patient.is_some() {
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

    fn create_path(parameters: &Parameters, rates: &Rates) -> PathBuf {
        //! Resturns the paths where to store the data. The hashmap has keys as
        //! the quantities stored, and values as the dest and source of where to
        //! compress the data.
        format!(
            "{}runs_{}cells_{}_{}1_{}2_{}undersample",
            parameters.nb_runs,
            parameters.max_cells,
            rates.fitness1,
            rates.death1,
            rates.death2,
            parameters.subsample.unwrap_or_default()
        )
        .into()
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
