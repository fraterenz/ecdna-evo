//! Implementation of the individual-based stochastic computer simulations of
//! the ecDNA population dynamics, assuming an exponential growing population of
//! tumour cells in a well-mixed environment. Simulation are carried out using
//! the Gillespie algorithm.
use crate::dynamics::{Dynamic, Dynamics, Name};
use crate::run::Run;
use crate::{Parameters, Patient, Rates};
use anyhow::{anyhow, Context};
use chrono::Utc;
use enum_dispatch::enum_dispatch;
use flate2::write::GzEncoder;
use flate2::Compression;
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
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
        // If we are in abc, `quantities` will be None (because created from
        // the data) and `data` will be Some
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

        // create path: relative path (used to creating the tarball) and
        // absolute path, for dynamics and abc
        let (relapath, abspath) = Simulation::create_path(&parameters, &rates);
        let (relapath_d, relapath_abc) = (relapath.join("dynamics"), relapath.join("abc"));

        let saved = (0..parameters.nb_runs)
            .into_par_iter()
            .progress_count(parameters.nb_runs as u64)
            .map(|idx| {
                Run::new(idx, parameters, &rates)
                    .simulate(dynamics.clone())
                    .save(&abspath, &dynamics, &patient)
                    .is_saved()
            })
            .collect::<Vec<bool>>();

        println!("{} End simulating {} runs", Utc::now(), parameters.nb_runs);

        println!(
            "{} Start compressing {} runs",
            Utc::now(),
            parameters.nb_runs
        );

        // Save tarballs from abc even if there aren't any saved runs for abc
        if patient.is_some() {
            if parameters.verbosity > 0 {
                println!("{} Creating tarball for abc", Utc::now());
            }
            Simulation::compress_results("all_rates", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress all rates");

            Simulation::compress_results("metadata", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress metadata");

            Simulation::compress_results("values", &relapath_abc, parameters.verbosity)
                .expect("Cannot compress metadata values");
        }

        // compress runs into tarball only if there is at least one saved run
        if saved.iter().any(|&saved_run| saved_run) {
            if let Some(dynamics) = &dynamics {
                for d in dynamics.iter() {
                    if parameters.verbosity > 0 {
                        println!("{} Creating tarball for dynamics", Utc::now());
                    }
                    Simulation::compress_results(d.get_name(), &relapath_d, parameters.verbosity)
                        .expect("Cannot compress dynamics");
                }
            }

            // data is some means abc subcomand, that is only the rates must
            // be saved
            if patient.is_some() {
                if parameters.verbosity > 0 {
                    println!("{} Creating tarball for abc", Utc::now());
                }
                Simulation::compress_results("rates", &relapath_abc, parameters.verbosity)
                    .expect("Cannot compress rates");
            }
            for name in ["ecdna", "mean", "frequency", "entropy"] {
                if parameters.verbosity > 0 {
                    println!("{} Creating tarball", Utc::now());
                }

                // do not save the ecdna distr when there is a patient (abc)
                match Simulation::compress_results(name, &relapath, parameters.verbosity) {
                    Ok(_) => {}
                    Err(_) => {
                        if !((name == "ecdna") && patient.is_some()) {
                            panic!("Cannot compress {}", name);
                        }
                    }
                }
            }
        }

        println!("{} End compressing {} runs", Utc::now(), parameters.nb_runs);

        println!(
            "{} {} runs over total of {} were saved",
            Utc::now(),
            saved.iter().filter(|&saved_run| *saved_run).count(),
            parameters.nb_runs
        );
        Ok(())
    }

    fn create_path(parameters: &Parameters, rates: &Rates) -> (PathBuf, PathBuf) {
        //! Resturns the paths where to store the data. The hashmap has keys as
        //! the quantities stored, and values as the dest and source of where to
        //! compress the data.
        // create path where to store the results
        let relative_path =
            PathBuf::from("results").join(Simulation::create_path_helper(parameters, rates));
        let abspath = env::current_dir().unwrap().join(&relative_path);

        (relative_path, abspath)
    }

    fn create_path_helper(parameters: &Parameters, rates: &Rates) -> PathBuf {
        format!(
            "{}runs_{}cells_{}_{}1_{}2",
            parameters.nb_runs, parameters.max_cells, rates.fitness1, rates.death1, rates.death2
        )
        .into()
    }

    fn compress_results(kind: &str, basepath: &Path, verbosity: u8) -> anyhow::Result<()> {
        let dest = basepath.join(kind);
        let src = env::current_dir().unwrap().join(&dest);
        Simulation::compress_dir(&dest, &src, verbosity)
            .with_context(|| format!("Cannot compress {:#?} into {:#?}", &src, &dest))?;
        Ok(())
    }

    fn compress_dir(dest_path: &Path, src_path_dir: &Path, verbosity: u8) -> anyhow::Result<()> {
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
            .with_context(|| format!("Error while opening the stream {:#?}", &dest_path_archive))?;
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

/// Trait to write the data to file
#[enum_dispatch]
pub trait ToFile {
    fn save(&self, path2file: &Path) -> anyhow::Result<()>;
}

pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
) -> anyhow::Result<()> {
    //! Write vector of float into new file with a precision of 4 decimals.
    //! Write NAN if the slice to write to file is empty.
    fs::create_dir_all(path.parent().unwrap()).expect("Cannot create dir");
    let f = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let mut buffer = BufWriter::new(f);
    if !data.is_empty() {
        if let Some(h) = header {
            writeln!(buffer, "{}", h)?;
        }
        write!(buffer, "{:.4}", data.first().unwrap())?;
        for ele in data.iter().skip(1) {
            write!(buffer, ",{:.4}", ele)?;
        }
    } else {
        write!(buffer, "{}", f32::NAN)?;
    }
    Ok(())
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
