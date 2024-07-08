use anyhow::ensure;
use rand::Rng;
use sosa::{AdvanceStep, CurrentState, NextReaction};
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    distribution::EcDNADistribution, segregation::Segregate, SavingOptions,
    Snapshot,
};

use super::{
    proliferation::{CellDeath, EcDNAProliferation},
    segregation::IsUneven,
};

#[derive(Debug, Clone, Copy)]
pub enum EcDNAEvent {
    ProliferateNPlus,
    ProliferateNMinus,
    DeathNPlus,
    DeathNMinus,
    SymmetricDivision,
    AsymmetricDivision,
    SymmetricDifferentiation,
}

fn save(
    path2dir: &Path,
    filename: &str,
    time: f32,
    distribution: &EcDNADistribution,
    verbosity: u8,
) -> anyhow::Result<()> {
    //! Helper fn to save the ecDNA distribution
    let cells = distribution.compute_nplus() + *distribution.get_nminus();
    let path2file = &path2dir.join(format!("{}cells/ecdna/", cells));
    let mut timepoint = format!("{:.1}", time).replace('.', "dot");
    timepoint.push_str("years");
    let mut path2file = path2file.join(timepoint).join(filename);
    path2file = path2file.with_extension("json");
    fs::create_dir_all(path2file.parent().unwrap())
        .expect("Cannot create dir");
    if verbosity > 0 {
        println!(
            "saving state at time {} with {} cells in {:#?}",
            time, cells, path2file
        );
    }
    distribution.save(&path2file, verbosity)?;
    Ok(())
}

/// The simulation of the [`EcDNADistribution`] according to a pure-birth
/// stochastic process.
#[derive(Debug, Clone)]
pub struct PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    distribution: EcDNADistribution,
    proliferation: P,
    segregation: S,
    time: f32,
    snapshots: VecDeque<Snapshot>,
    path2dir: PathBuf,
    filename: String,
    verbosity: u8,
}

impl<P, S> PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    pub fn new(
        distribution: EcDNADistribution,
        proliferation: P,
        segregation: S,
        time: f32,
        saving_options: SavingOptions,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            distribution,
            proliferation,
            segregation,
            time,
            path2dir: saving_options.path2dir,
            filename: saving_options.filename,
            snapshots: saving_options.snapshots,
            verbosity,
        })
    }

    pub fn get_mut_ecdna_distribution(&mut self) -> &mut EcDNADistribution {
        &mut self.distribution
    }

    pub fn get_ecdna_distribution(&self) -> &EcDNADistribution {
        &self.distribution
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2> for PureBirth<P, S> {
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        while !self.snapshots.is_empty()
            && self.snapshots.iter().any(|s| self.time >= s.time)
        {
            let snapshot = self.snapshots.pop_front().unwrap();
            if self.verbosity > 0 {
                println!(
                    "saving state for timepoint at time {:#?} at simulation's time {} with {} cells",
                    snapshot.time, self.time, snapshot.cells2sample
                );
            }
            let cells = (*self.distribution.get_nminus()
                + self.distribution.compute_nplus())
                as usize;

            if snapshot.cells2sample == cells || cells < snapshot.cells2sample
            {
                save(
                    &self.path2dir,
                    &self.filename,
                    snapshot.time,
                    &self.distribution,
                    self.verbosity,
                )
                .expect("cannot save snapshot");
            } else {
                save(
                    &self.path2dir,
                    &self.filename,
                    snapshot.time,
                    &self
                        .distribution
                        .into_subsampled(snapshot.cells2sample as u64, rng),
                    self.verbosity,
                )
                .expect("cannot save snapshot");
            }
        }

        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.distribution,
                    &self.segregation,
                    rng,
                    self.verbosity,
                ) {
                    match is_uneven {
                        IsUneven::False => {
                            if self.verbosity > 1 {
                                println!("IsUneven is false");
                            }
                        }
                        IsUneven::True => {
                            if self.verbosity > 1 {
                                println!("IsUneven is true");
                            }
                        }
                        IsUneven::TrueWithoutNMinusIncrease => {
                            if self.verbosity > 1 {
                                println!(
                                    "IsUneven is true but w/o nminus increase"
                                );
                            }
                        }
                    }
                }
            }
            EcDNAEvent::ProliferateNMinus => {
                self.proliferation.increase_nminus(&mut self.distribution);
            }
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.distribution);
        }
        self.time += reaction.time;
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        state.population[0] = *self.distribution.get_nminus();
        state.population[1] = self.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                state.population
            );
        }
    }
}

/// The simulation of the [`EcDNADistribution`] according to a birth-death
/// stochastic process.
#[derive(Debug, Clone)]
pub struct BirthDeath<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    distribution: EcDNADistribution,
    proliferation: P,
    segregation: S,
    death: CellDeath,
    time: f32,
    snapshots: VecDeque<Snapshot>,
    path2dir: PathBuf,
    filename: String,
    verbosity: u8,
}

impl<P, S> BirthDeath<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    pub fn new(
        distribution: EcDNADistribution,
        proliferation: P,
        segregation: S,
        death: CellDeath,
        time: f32,
        saving_options: SavingOptions,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            distribution,
            proliferation,
            segregation,
            death,
            time,
            path2dir: saving_options.path2dir,
            filename: saving_options.filename,
            snapshots: saving_options.snapshots,
            verbosity,
        })
    }

    pub fn get_mut_ecdna_distribution(&mut self) -> &mut EcDNADistribution {
        &mut self.distribution
    }

    pub fn get_ecdna_distribution(&self) -> &EcDNADistribution {
        &self.distribution
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4> for BirthDeath<P, S> {
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        while !self.snapshots.is_empty()
            && self.snapshots.iter().any(|s| self.time >= s.time)
        {
            let snapshot = self.snapshots.pop_front().unwrap();
            if self.verbosity > 0 {
                println!(
                    "saving state for timepoint at time {:#?} at simulation's time {} with {} cells",
                    snapshot.time, self.time, snapshot.cells2sample
                );
            }
            let cells = (*self.distribution.get_nminus()
                + self.distribution.compute_nplus())
                as usize;

            if snapshot.cells2sample == cells {
                save(
                    &self.path2dir,
                    &self.filename,
                    snapshot.time,
                    &self.distribution,
                    self.verbosity,
                )
                .expect("cannot save snapshot");
            } else {
                save(
                    &self.path2dir,
                    &self.filename,
                    snapshot.time,
                    &self
                        .distribution
                        .into_subsampled(snapshot.cells2sample as u64, rng),
                    self.verbosity,
                )
                .expect("cannot save snapshot");
            }
        }
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.distribution,
                    &self.segregation,
                    rng,
                    self.verbosity,
                ) {
                    match is_uneven {
                        IsUneven::False => {
                            if self.verbosity > 1 {
                                println!("IsUneven is false");
                            }
                        }
                        IsUneven::True => {
                            if self.verbosity > 1 {
                                println!("IsUneven is true");
                            }
                        }
                        IsUneven::TrueWithoutNMinusIncrease => {
                            if self.verbosity > 1 {
                                println!(
                                    "IsUneven is true but w/o nminus increase"
                                );
                            }
                        }
                    }
                }
            }
            EcDNAEvent::ProliferateNMinus => {
                self.proliferation.increase_nminus(&mut self.distribution);
            }
            EcDNAEvent::DeathNMinus => {
                self.death.decrease_nminus(&mut self.distribution)
            }
            EcDNAEvent::DeathNPlus => self.death.decrease_nplus(
                &mut self.distribution,
                rng,
                self.verbosity,
            ),
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.distribution);
        }
        self.time += reaction.time;
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.distribution.get_nminus();
        state.population[1] = self.distribution.compute_nplus();
        state.population[2] = *self.distribution.get_nminus();
        state.population[3] = self.distribution.compute_nplus();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        proliferation::Exponential, segregation::Binomial,
        test_util::NonEmptyDistribtionWithNPlusCells,
    };
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn create_birth_death_process_test(
        distr: NonEmptyDistribtionWithNPlusCells,
        verbosity: u8,
        time: u8,
    ) -> bool {
        let time = time as f32 / 10.;
        let nplus = distr.0.compute_nplus();
        let nminus = *distr.0.get_nminus();
        let mean = distr.0.compute_mean();
        let p = BirthDeath::new(
            distr.0,
            Exponential {},
            Binomial,
            CellDeath,
            time,
            SavingOptions {
                snapshots: VecDeque::from([Snapshot {
                    cells2sample: 10,
                    time: 2.,
                }]),
                path2dir: PathBuf::default(),
                filename: String::from("filename"),
            },
            verbosity,
        )
        .unwrap();
        p.distribution.compute_nplus() == nplus
            && p.distribution.get_nminus() == &nminus
            && p.time == time
            && p.distribution.compute_mean() == mean
    }
}
