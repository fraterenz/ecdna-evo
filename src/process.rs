use std::{fs, path::Path};

use anyhow::ensure;
use rand_chacha::ChaCha8Rng;
use ssa::{
    iteration::{AdvanceStep, CurrentState, NextReaction},
    write2file, NbIndividuals, Process, RandomSampling, ToFile,
};

use crate::{
    distribution::EcDNADistribution, dynamics::EcDNADynamicsTime,
    segregation::Segregate,
};

use super::{
    dynamics::EcDNADynamics,
    proliferation::{EcDNADeath, EcDNAProliferation},
    segregation::IsUneven,
};

#[derive(Debug, Clone)]
pub enum Event {
    ProliferateNPlus,
    ProliferateNMinus,
    DeathNPlus,
    DeathNMinus,
    SymmetricDivision,
    AsymmetricDivision,
    SymmetricDifferentiation,
}

pub const PUREBIRTH_EVENTS: [Event; 2] =
    [Event::ProliferateNMinus, Event::ProliferateNPlus];

pub const BIRTHDEATH_EVENTS: [Event; 4] = [
    Event::ProliferateNMinus,
    Event::ProliferateNPlus,
    Event::DeathNMinus,
    Event::DeathNPlus,
];

pub enum Dynamic {
    EcDNA,
    EcDNAMean,
}

#[derive(Debug, Clone)]
pub struct PureBirthNoDynamics<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    distribution: EcDNADistribution,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P, S> PureBirthNoDynamics<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    pub fn new(
        distribution: EcDNADistribution,
        proliferation: P,
        segregation: S,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self { distribution, proliferation, segregation, verbosity })
    }

    pub fn get_ecdna_distribution(&self) -> &EcDNADistribution {
        &self.distribution
    }
}

impl<P: EcDNAProliferation + Clone, S: Segregate + Clone> Process<2>
    for PureBirthNoDynamics<P, S>
{
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2>
    for PureBirthNoDynamics<P, S>
{
    fn advance_step(&mut self, reaction: NextReaction, rng: &mut ChaCha8Rng) {
        match PUREBIRTH_EVENTS[reaction.event] {
            Event::ProliferateNPlus => {
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
            Event::ProliferateNMinus => {
                self.proliferation.increase_nminus(&mut self.distribution);
            }
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.distribution);
        }
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

impl<P: EcDNAProliferation, S: Segregate> ToFile
    for PureBirthNoDynamics<P, S>
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        //! Save the ecDNA distribution
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let path2file = path2dir.join(id.to_string());
        self.distribution.save(&path2file, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling
    for PureBirthNoDynamics<P, S>
{
    fn random_sample(
        &mut self,
        nb_individuals: NbIndividuals,
        rng: &mut ChaCha8Rng,
    ) {
        self.distribution.undersample(nb_individuals, rng);
    }
}

#[derive(Debug, Clone)]
pub struct PureBirth<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirth<P, S> {
    pub fn with_distribution(
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        //! Create a pure-birth process from a non-empty `distribution`
        //! tracking the nminus and nplus cells over time.
        //!
        //! ## Error
        //! Returns error when the distribution is empty.
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamics::new(distribution, max_iter),
            proliferation,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation + Clone, S: Segregate + Clone> Process<2>
    for PureBirth<P, S>
{
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2> for PureBirth<P, S> {
    fn advance_step(&mut self, reaction: NextReaction, rng: &mut ChaCha8Rng) {
        match PUREBIRTH_EVENTS[reaction.event] {
            Event::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.data.distribution,
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
            Event::ProliferateNMinus => {
                self.proliferation
                    .increase_nminus(&mut self.data.distribution);
            }
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.data.distribution);
        }

        self.data.nplus.push(self.data.distribution.compute_nplus());
        self.data.nminus.push(*self.data.distribution.get_nminus());
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                state.population
            );
        }
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for PureBirth<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = id.to_string();
        self.data.save(path2dir, &filename, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling for PureBirth<P, S> {
    fn random_sample(
        &mut self,
        nb_individuals: NbIndividuals,
        rng: &mut ChaCha8Rng,
    ) {
        if self.verbosity > 0 {
            println!("Subsampling the ecDNA distribution");
        }
        if self.verbosity > 1 {
            println!("Before {:#?}", self.data.distribution);
        }
        self.data.distribution.undersample(nb_individuals, rng);
        if self.verbosity > 1 {
            println!("After {:#?}", self.data.distribution);
        }
        let (nplus, nminus) = (
            self.data.distribution.compute_nplus(),
            *self.data.distribution.get_nminus(),
        );
        self.data.nplus.push(nplus);
        self.data.nminus.push(nminus);
        if self.verbosity > 1 {
            println!(
                "NPlus/NMinus after subsampling {}, {}",
                self.data.nplus.last().unwrap(),
                self.data.nminus.last().unwrap()
            );
        }
        todo!();
        // self.iteration.population[0] = nplus;
        // self.iteration.population[1] = nminus;
    }
}

#[derive(Debug, Clone)]
pub struct PureBirthMean<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    mean: Vec<f32>,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthMean<P, S> {
    pub fn new(
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamics::new(distribution, max_iter),
            mean,
            proliferation,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2>
    for PureBirthMean<P, S>
{
    fn advance_step(&mut self, reaction: NextReaction, rng: &mut ChaCha8Rng) {
        todo!();
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        todo!();
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for PureBirthMean<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = id.to_string();
        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        write2file(&self.mean, &mean, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling
    for PureBirthMean<P, S>
{
    fn random_sample(
        &mut self,
        nb_individuals: NbIndividuals,
        rng: &mut ChaCha8Rng,
    ) {
        todo!();
    }
}

#[derive(Debug, Clone)]
pub struct BirthDeathMeanTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    mean: Vec<f32>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathMeanTime<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamicsTime::new(time, distribution, max_iter),
            mean,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathMeanTime<P, S>
{
    fn advance_step(&mut self, reaction: NextReaction, rng: &mut ChaCha8Rng) {
        todo!()
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        todo!();
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for BirthDeathMeanTime<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", path2dir)
        }
        write2file(&self.mean, &mean, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling
    for BirthDeathMeanTime<P, S>
{
    fn random_sample(
        &mut self,
        nb_individuals: NbIndividuals,
        rng: &mut ChaCha8Rng,
    ) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "todo"]
    fn random_sample_test() {
        todo!()
    }
}
