//! Gillespie simulator of a two-population birth-death process.
pub mod event;
pub mod process;
pub mod rate;

use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

use serde::Serialize;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// Number of individual cells present in the system.
pub type NbIndividuals = u64;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Seed(u64);

impl Seed {
    pub fn new(seed: u64) -> Self {
        Seed(seed)
    }

    pub fn get_seed(&self) -> &u64 {
        &self.0
    }

    pub fn save(&self, path2file: &Path) -> anyhow::Result<()> {
        write2file(&[self.0], path2file, None, false)?;
        Ok(())
    }
}

impl Default for Seed {
    fn default() -> Self {
        Seed(26u64)
    }
}

pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
    endline: bool,
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
        for ele in data.iter() {
            write!(buffer, "{:.4},", ele)?;
        }
        if endline {
            writeln!(buffer)?;
        }
    } else {
        write!(buffer, "{},", f32::NAN)?;
    }
    Ok(())
}
