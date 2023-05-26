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
///
/// Those quantities are not save for every new reaction (i.e. every new
/// birth/death event), instead for snapshots taken at `timepoints`.
#[derive(Debug, Clone)]
pub struct EcDNADynamics {
    /// The number of cells with ecDNAs for each iteration.
    nplus: Vec<NbIndividuals>,
    /// The number of cells without ecDNAs for each iteration.
    nminus: Vec<NbIndividuals>,
    /// The times at which the dynamics will be registered, used as a stack.
    timepoints: Vec<f32>,
    /// The timepoints that will be saved, not used as a stack but const from
    /// the start to the end of the simulation
    timepoints2save: Vec<f32>,
    /// The time for the **current** iteration.
    pub time: f32,
    /// The ecDNA distribution for the **current** iteration.
    pub distribution: EcDNADistribution,
}

/// When saving the dynamics, we look at some fixed timepoints such that all
/// simulations will share the same time.
/// To do this, we bin the continous time variable into fixed-sized bins and
/// look whether the new reaction's time is smaller or equal to the timepoint.
/// It's equal, then we save the new dynamic with `SaveNewOne`.
/// If it's greater than the current timepoints (timepoints get removed once
/// used), then we save the dynamic using the one of the previous iteration with
/// `SavePreviousOne`.
/// Else, we dont save `DoNotSave`.
pub enum SaveDynamic {
    /// Save the previous value for the dynamic `times` times.
    SavePreviousOne {
        times: usize,
    },
    SaveNewOne,
    DoNotSave,
}

impl EcDNADynamics {
    pub fn new(
        distribution: EcDNADistribution,
        iterations: usize,
        timepoints: &[f32],
        initial_time: f32,
    ) -> Self {
        // an ecDNA distribution as always an entry for the nminus cells
        let mut nminus = Vec::with_capacity(iterations);
        nminus.push(*distribution.get_nminus());
        let mut nplus = Vec::with_capacity(iterations);
        nplus.push(distribution.compute_nplus());

        let mut timepoints2save = timepoints.to_owned();
        // sort them to use vec as stack
        timepoints2save.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let timepoints = timepoints2save.clone().into_iter().rev().collect();
        // sort them decreasing order such that FILO (first the smallest)

        Self {
            nplus,
            nminus,
            distribution,
            timepoints,
            time: initial_time,
            timepoints2save: timepoints2save.to_vec(),
        }
    }

    pub fn update_nplus_nminus(
        &mut self,
        nplus: NbIndividuals,
        nminus: NbIndividuals,
    ) {
        self.nplus.push(nplus);
        self.nminus.push(nminus);
    }

    pub fn is_time_to_save(
        &mut self,
        time: f32,
        verbosity: u8,
    ) -> SaveDynamic {
        //! Is it time to save based on `time` and on the timepoints?
        //!
        //! This is true when the time is smaller than the timepoint and close
        //! enough to it (due to float precision).
        if let Some(current_t) = self.get_current_timepoint() {
            let time2save = (current_t - time).abs() < f32::EPSILON;
            if time2save {
                if verbosity > 1 {
                    println!(
                        "saving the new dynamic at timepoint {} for time {}",
                        current_t, time
                    );
                }
                self.register_timepoint();
                return SaveDynamic::SaveNewOne;
            }
            if &time >= current_t {
                if verbosity > 1 {
                    println!(
                    "saving the previous dynamic at timepoint {} for time {}",
                    current_t, time
                );
                }
                // nb of times to repeat the dynamic
                let tot_nb_bins = self.timepoints.len();
                self.timepoints.retain_mut(|&mut t| t >= time);
                let nb_skipped_bins = tot_nb_bins - self.timepoints.len();
                if verbosity > 1 {
                    println!(
                        "saving the previous dynamic {} times",
                        nb_skipped_bins
                    );
                }
                return SaveDynamic::SavePreviousOne {
                    times: nb_skipped_bins,
                };
            }

            if verbosity > 1 {
                println!(
                    "do not save at timepoint {} at time {}",
                    current_t, time
                );
            }
        };
        SaveDynamic::DoNotSave
    }

    fn get_current_timepoint(&self) -> Option<&f32> {
        self.timepoints.last()
    }

    fn register_timepoint(&mut self) {
        self.timepoints.pop().expect("timepoint already registered");
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

        let mut times = path2dir.join("times").join(id);
        times.set_extension("csv");
        if verbosity > 1 {
            println!("Saving times to {:#?}", times);
        }
        write2file(&self.timepoints2save, &times, None, false)
            .with_context(|| "Cannot save times".to_string())?;

        // the absolute gillespie time at the last iteration
        let mut gillespie_time = path2dir.join("gillespie_time").join(id);
        gillespie_time.set_extension("csv");
        if verbosity > 1 {
            println!("Saving gillespie time to {:#?}", gillespie_time);
        }
        write2file(&[self.time], &gillespie_time, None, false)
            .with_context(|| "Cannot save times".to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use quickcheck_macros::quickcheck;

    use crate::test_util::NonEmptyDistribtionWithNPlusCells;

    use super::*;

    #[quickcheck]
    fn ecdna_new_test(
        time: u8,
        distribution: NonEmptyDistribtionWithNPlusCells,
    ) -> bool {
        let ecdna = EcDNADynamics::new(
            distribution.clone().0,
            1,
            &[time as f32, time as f32 + 0.4],
            0.,
        );
        ecdna.nplus.last().unwrap() == &distribution.0.compute_nplus()
            && ecdna.nminus.last().unwrap() == distribution.0.get_nminus()
            && ecdna.timepoints == vec![time as f32 + 0.4, time as f32]
    }
}
