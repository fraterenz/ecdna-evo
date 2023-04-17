//! The ecDNA data model.
use std::path::Path;

use crate::distribution::EcDNADistribution;
use anyhow::Context;
use sosa::{write2file, NbIndividuals};

/// Some summary statistics for the [`EcDNADistribution`].
#[derive(Debug, Clone)]
pub struct EcDNASummary {
    pub mean: f32,
    pub frequency: f32,
    pub entropy: f32,
}

/// The quantities of interest that evolve over time for the ecDNA problem.
#[derive(Debug, Clone)]
pub struct EcDNADynamics {
    /// The number of cells with ecDNAs for each iteration.
    pub nplus: Vec<NbIndividuals>,
    /// The number of cells without ecDNAs for each iteration.
    pub nminus: Vec<NbIndividuals>,
    /// The ecDNA distribution for the **current** iteration.
    pub distribution: EcDNADistribution,
}

impl EcDNADynamics {
    pub fn new(distribution: EcDNADistribution, iterations: usize) -> Self {
        // an ecDNA distribution as always an entry for the nminus cells
        let mut nminus = Vec::with_capacity(iterations);
        nminus.push(*distribution.get_nminus());
        let mut nplus = Vec::with_capacity(iterations);
        nplus.push(distribution.compute_nplus());

        Self { nplus, nminus, distribution }
    }

    pub fn save(
        &self,
        path2dir: &Path,
        id: &str,
        verbosity: u8,
    ) -> anyhow::Result<()> {
        let mut nplus = path2dir.join("nplus").join(id);
        nplus.set_extension("csv");
        if verbosity > 1 {
            println!("Saving nplus to {:#?}", nplus);
        }
        write2file(&self.nplus, &nplus, None, false)
            .with_context(|| "Cannot save nplus".to_string())?;

        let mut nminus = path2dir.join("nminus").join(id);
        nminus.set_extension("csv");
        if verbosity > 1 {
            println!("Saving nminus to {:#?}", nminus);
        }
        write2file(&self.nminus, &nminus, None, false)
            .with_context(|| "Cannot save nminus".to_string())?;

        let mut ecdna = path2dir.join("ecdna").join(id);
        ecdna.set_extension("json");
        self.distribution.save(&ecdna, verbosity)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EcDNADynamicsTime {
    pub ecdna_dynamics: EcDNADynamics,
    /// The Gillespie time evolving for each iteration.
    pub time: Vec<f32>,
}

impl EcDNADynamicsTime {
    pub fn new(
        time: f32,
        distribution: EcDNADistribution,
        iterations: usize,
    ) -> Self {
        // an ecDNA distribution as always an entry for the nminus cells
        let ecdna_dynamics = EcDNADynamics::new(distribution, iterations);
        let mut times = Vec::with_capacity(iterations);
        times.push(time);

        Self { ecdna_dynamics, time: times }
    }

    pub fn save(
        &self,
        path2dir: &Path,
        id: &str,
        verbosity: u8,
    ) -> anyhow::Result<()> {
        let mut times = path2dir.join("times").join(id);
        times.set_extension("csv");
        if verbosity > 1 {
            println!("Saving times to {:#?}", times);
        }
        write2file(&self.time, &times, None, false)
            .with_context(|| "Cannot save times".to_string())?;

        self.ecdna_dynamics.save(path2dir, id, verbosity)
    }
}

#[cfg(test)]
mod tests {

    use quickcheck_macros::quickcheck;

    use crate::test_util::NonEmptyDistribtionWithNPlusCells;

    use super::*;

    #[quickcheck]
    fn ecdna_new_test(
        time: f32,
        distribution: NonEmptyDistribtionWithNPlusCells,
    ) -> bool {
        let ecdna = EcDNADynamicsTime::new(time, distribution.clone().0, 1);
        ecdna.ecdna_dynamics.nplus.last().unwrap()
            == &distribution.0.compute_nplus()
            && ecdna.ecdna_dynamics.nminus.last().unwrap()
                == distribution.0.get_nminus()
    }
}
