use std::{fs, path::Path};

use anyhow::ensure;
use rand_chacha::ChaCha8Rng;
use ssa::{
    distribution::EcDNADistribution,
    iteration::{Iterate, Iteration, NextReaction, SimState},
    write2file, NbIndividuals, Process, RandomSampling, ToFile,
};

use crate::{dynamics::EcDNADynamicsTime, segregation::Segregate};

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
    iteration: Iteration<2>,
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
        iteration: Iteration<2>,
        distribution: EcDNADistribution,
        proliferation: P,
        segregation: S,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            distribution,
            iteration,
            proliferation,
            segregation,
            verbosity,
        })
    }

    pub fn get_ecdna_distribution(&self) -> &EcDNADistribution {
        &self.distribution
    }

    pub fn get_rates(&self) -> &[f32; 2] {
        &self.iteration.rates.rates
    }
}

impl<P: EcDNAProliferation, S: Segregate> Process
    for PureBirthNoDynamics<P, S>
{
}

impl<P: EcDNAProliferation, S: Segregate> Iterate
    for PureBirthNoDynamics<P, S>
{
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.distribution.compute_nplus()
            + *self.distribution.get_nminus();
        if self.verbosity > 1 {
            println!("Population: {:#?}", population);
        }
        self.iteration.next_reaction(population, iter, rng)
    }

    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
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

    fn update_iteration(&mut self) {
        // Iteration's update is delegated to data
        self.iteration.population[0] = *self.distribution.get_nminus();
        self.iteration.population[1] = self.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                self.iteration.population
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
        self.iteration.population[0] = *self.distribution.get_nminus();
        self.iteration.population[1] = self.distribution.compute_nplus();
    }
}

#[derive(Debug, Clone)]
pub struct PureBirth<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    iteration: Iteration<2>,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirth<P, S> {
    pub fn new(
        proliferation: P,
        segregation: S,
        iteration: Iteration<2>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamics::new(distribution, iteration.max_iter),
            iteration,
            proliferation,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate for PureBirth<P, S> {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.data.distribution.compute_nplus()
            + *self.data.distribution.get_nminus();
        if self.verbosity > 1 {
            println!("Population: {:#?}", population);
        }
        self.iteration.next_reaction(population, iter, rng)
    }

    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
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
        // Phe update of the dynamics is delegated to the ecDNA distribution
        self.data.nminus.push(*self.data.distribution.get_nminus());
        self.data.nplus.push(self.data.distribution.compute_nplus());
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.data.distribution);
        }
        if self.verbosity > 1 {
            println!("NMinus new {:#?}", self.data.nminus.last().unwrap());
        }
        if self.verbosity > 1 {
            println!("NPlus new {:#?}", self.data.nplus.last().unwrap());
        }
    }

    fn update_iteration(&mut self) {
        // Iteration's update is delegated to the ecDNA distribution
        self.iteration.population[0] =
            *self.data.nminus.last().expect("Empty NMinus population");
        self.iteration.population[1] =
            *self.data.nplus.last().expect("Empty NPlus population");
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                self.iteration.population
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
        self.iteration.population[0] = nplus;
        self.iteration.population[1] = nminus;
    }
}

#[derive(Debug, Clone)]
pub struct PureBirthMean<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    mean: Vec<f32>,
    iteration: Iteration<2>,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthMean<P, S> {
    pub fn new(
        proliferation: P,
        segregation: S,
        iteration: Iteration<2>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(iteration.max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamics::new(distribution, iteration.max_iter),
            iteration,
            mean,
            proliferation,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate for PureBirthMean<P, S> {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population =
            self.data.nplus.last().unwrap() + self.data.nminus.last().unwrap();
        self.iteration.next_reaction(population, iter, rng)
    }

    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
        todo!()
    }

    fn update_iteration(&mut self) {
        todo!()
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
pub struct PureBirthTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    iteration: Iteration<2>,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthTime<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        iteration: Iteration<2>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamicsTime::new(
                time,
                distribution,
                iteration.max_iter,
            ),
            iteration,
            segregation,
            proliferation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate for PureBirthTime<P, S> {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.data.ecdna_dynamics.nplus.last().unwrap()
            + self.data.ecdna_dynamics.nminus.last().unwrap();
        self.iteration.next_reaction(population, iter, rng)
    }

    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
        todo!()
    }

    fn update_iteration(&mut self) {
        todo!()
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for PureBirthTime<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = id.to_string();
        self.data.save(path2dir, &filename, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling
    for PureBirthTime<P, S>
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
pub struct BirthDeathTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    iteration: Iteration<4>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathTime<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        iteration: Iteration<4>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamicsTime::new(
                time,
                distribution,
                iteration.max_iter,
            ),
            iteration,
            proliferation,
            segregation,
            death: EcDNADeath,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate for BirthDeathTime<P, S> {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.data.ecdna_dynamics.nplus.last().unwrap()
            + self.data.ecdna_dynamics.nminus.last().unwrap();
        self.iteration.next_reaction(population, iter, rng)
    }
    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
        todo!()
    }

    fn update_iteration(&mut self) {
        todo!()
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for BirthDeathTime<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = id.to_string();
        self.data.save(path2dir, &filename, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling
    for BirthDeathTime<P, S>
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
pub struct PureBirthMeanTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    mean: Vec<f32>,
    iteration: Iteration<2>,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthMeanTime<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        iteration: Iteration<2>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(iteration.max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamicsTime::new(
                time,
                distribution,
                iteration.max_iter,
            ),
            iteration,
            mean,
            proliferation,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate for PureBirthMeanTime<P, S> {
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.data.ecdna_dynamics.nplus.last().unwrap()
            + self.data.ecdna_dynamics.nminus.last().unwrap();
        self.iteration.next_reaction(population, iter, rng)
    }
    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
        todo!()
    }

    fn update_iteration(&mut self) {
        todo!()
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for PureBirthMeanTime<P, S> {
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
    for PureBirthMeanTime<P, S>
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
    iteration: Iteration<4>,
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
        iteration: Iteration<4>,
        distribution: EcDNADistribution,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(iteration.max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamicsTime::new(
                time,
                distribution,
                iteration.max_iter,
            ),
            mean,
            iteration,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P: EcDNAProliferation, S: Segregate> Iterate
    for BirthDeathMeanTime<P, S>
{
    fn next_reaction(
        &mut self,
        iter: usize,
        rng: &mut ChaCha8Rng,
    ) -> (SimState, Option<NextReaction>) {
        let population = self.data.ecdna_dynamics.nplus.last().unwrap()
            + self.data.ecdna_dynamics.nminus.last().unwrap();
        if self.verbosity > 1 {
            println!("Population: {:#?}", population);
        }
        self.iteration.next_reaction(population, iter, rng)
    }

    fn update_process(
        &mut self,
        reaction: NextReaction,
        rng: &mut ChaCha8Rng,
    ) {
        todo!();
        // self.data
        //     .time
        //     .push(reaction.time + self.data.time.last().unwrap());
        // let segregation = match BIRTHDEATH_EVENTS[reaction.event] {
        //     Event::ProliferateNPlus => {
        //         self.proliferation
        //             .increase_nplus(&mut self.data.ecdna_dynamics, rng, self.verbosity)
        //     }
        //     Event::ProliferateNMinus => {
        //         self.proliferation.increase_nminus(&mut self.data.ecdna_dynamics);
        //         Ok(IsUneven::False)
        //     }
        //     Event::DeathNPlus => {
        //         self.death
        //             .decrease_nplus(&mut self.data.ecdna_dynamics, rng, self.verbosity);
        //         Ok(IsUneven::False)
        //     }
        //     Event::DeathNMinus => {
        //         self.death.decrease_nminus(&mut self.data.ecdna_dynamics);
        //         Ok(IsUneven::False)
        //     }
        //     _ => unreachable!(),
        // };
        // if self.verbosity > 1 {
        //     println!("Segregation is uneven {:#?}", segregation);
        //     println!("Distribution {:#?}", self.data.ecdna_dynamics.distribution);
        //     println!(
        //         "Nplus {:#?}",
        //         self.data.ecdna_dynamics.nplus.last().unwrap()
        //     );
        //     println!(
        //         "NMinus {:#?}",
        //         self.data.ecdna_dynamics.nminus.last().unwrap()
        //     );
        // }
        // self.mean
        //     .push(self.data.ecdna_dynamics.distribution.compute_mean());
        // if self.verbosity > 1 {
        //     println!("New mean: {:#?}", self.mean.last().unwrap());
        // }
    }

    fn update_iteration(&mut self) {
        // Iteration's update is delegated to data
        self.iteration.population[0] = *self
            .data
            .ecdna_dynamics
            .nminus
            .last()
            .expect("Empty NMinus population");
        self.iteration.population[1] = *self
            .data
            .ecdna_dynamics
            .nplus
            .last()
            .expect("Empty NPlus population");
        // in ecDNA with death, there are only two-types but four reactions
        // which are linked
        self.iteration.population[2] = self.iteration.population[0];
        self.iteration.population[3] = self.iteration.population[1];
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                self.iteration.population
            );
        }
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
