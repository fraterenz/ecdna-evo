use std::{fs, path::Path};

use anyhow::ensure;
use ecdna_lib::distribution::SamplingStrategy;
use rand::Rng;
use sosa::{
    write2file, AdvanceStep, CurrentState, NbIndividuals, NextReaction,
};

use crate::{
    distribution::EcDNADistribution, dynamics::EcDNADynamicsTime,
    segregation::Segregate, RandomSampling, ToFile,
};

use super::{
    dynamics::EcDNADynamics,
    proliferation::{EcDNADeath, EcDNAProliferation},
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

/// The simulation of the [`EcDNADistribution`] according to a pure-birth
/// stochastic process.
///
/// The distribution can be saved during or at the end of the simulation using
/// [`Self::save`].
#[derive(Debug, Clone)]
pub struct PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    distribution: EcDNADistribution,
    proliferation: P,
    segregation: S,
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
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self { distribution, proliferation, segregation, verbosity })
    }

    pub fn get_mut_ecdna_distribution(&mut self) -> &mut EcDNADistribution {
        &mut self.distribution
    }

    pub fn get_ecdna_distribution(&self) -> &EcDNADistribution {
        &self.distribution
    }
}

impl<P, S> RandomSampling for PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        strategy: &SamplingStrategy,
        nb_individuals: NbIndividuals,
        rng: impl Rng + std::clone::Clone,
    ) {
        self.distribution.sample(nb_individuals, strategy, rng);
    }
}

impl<P, S> ToFile for PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        //! Save the ecDNA distribution
        let path2ecdna = path2dir.join("ecdna");
        fs::create_dir_all(&path2ecdna).expect("Cannot create dir");
        let path2file = path2ecdna.join(id.to_string()).with_extension("json");
        self.distribution.save(&path2file, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2> for PureBirth<P, S> {
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
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

/// A pure-birth stochastic process tracking the evolution of the
/// [`EcDNADynamics`] over time.
#[derive(Debug, Clone)]
pub struct PureBirthNMinusNPlus<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthNMinusNPlus<P, S> {
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

impl<P, S> RandomSampling for PureBirthNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        strategy: &SamplingStrategy,
        nb_individuals: NbIndividuals,
        rng: impl Rng + std::clone::Clone,
    ) {
        if self.verbosity > 0 {
            println!("Subsampling the ecDNA distribution");
        }
        if self.verbosity > 1 {
            println!("Before {:#?}", self.data.distribution);
        }
        self.data.distribution.sample(nb_individuals, strategy, rng);

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

impl<P, S> ToFile for PureBirthNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = id.to_string();
        self.data.save(path2dir, &filename, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2>
    for PureBirthNMinusNPlus<P, S>
{
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
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
            EcDNAEvent::ProliferateNMinus => {
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

/// A pure-birth stochastic process tracking the evolution of the
/// [`EcDNADynamics`] per time.
#[derive(Debug, Clone)]
pub struct PureBirthNMinusNPlusTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    proliferation: P,
    segregation: S,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> PureBirthNMinusNPlusTime<P, S> {
    pub fn with_distribution(
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        time: f32,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamicsTime::new(time, distribution, max_iter),
            proliferation,
            segregation,
            verbosity,
        })
    }
}
impl<P, S> RandomSampling for PureBirthNMinusNPlusTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for PureBirthNMinusNPlusTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2>
    for PureBirthNMinusNPlusTime<P, S>
{
    type Reaction = EcDNAEvent;
    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.data.ecdna_dynamics.distribution,
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
                self.proliferation.increase_nminus(
                    &mut self.data.ecdna_dynamics.distribution,
                );
            }
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!(
                "Distribution {:#?}",
                self.data.ecdna_dynamics.distribution
            );
        }

        self.data
            .ecdna_dynamics
            .nplus
            .push(self.data.ecdna_dynamics.distribution.compute_nplus());
        self.data
            .ecdna_dynamics
            .nminus
            .push(*self.data.ecdna_dynamics.distribution.get_nminus());
        self.data.time.push(reaction.time);
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        state.population[0] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[1] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                state.population
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct PureBirthMean<P: EcDNAProliferation, S: Segregate> {
    distribution: EcDNADistribution,
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
        Ok(Self { distribution, mean, proliferation, segregation, verbosity })
    }
}

impl<P, S> RandomSampling for PureBirthMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for PureBirthMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.distribution.save(
            &path2dir.join("ecdna").join(&filename).with_extension("json"),
            self.verbosity,
        )?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", path2dir)
        }
        write2file(&self.mean, &mean, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<2>
    for PureBirthMean<P, S>
{
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
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
        self.mean.push(self.distribution.compute_mean());
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        state.population[0] = *self.distribution.get_nminus();
        state.population[1] = self.distribution.compute_nplus();
    }
}

/// The simulation of the [`EcDNADistribution`] according to a birth-death
/// stochastic process.
///
/// The distribution can be saved during or at the end of the simulation using
/// [`Self::save`].
#[derive(Debug, Clone)]
pub struct BirthDeath<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    distribution: EcDNADistribution,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
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
        death: EcDNADeath,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self { distribution, proliferation, segregation, death, verbosity })
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
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.distribution.get_nminus();
        state.population[1] = self.distribution.compute_nplus();
        state.population[2] = *self.distribution.get_nminus();
        state.population[3] = self.distribution.compute_nplus();
    }
}

impl<P: EcDNAProliferation, S: Segregate> ToFile for BirthDeath<P, S> {
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        //! Save the ecDNA distribution
        let path2ecdna = path2dir.join("ecdna");
        fs::create_dir_all(&path2ecdna).expect("Cannot create dir");
        let path2file = path2ecdna.join(id.to_string()).with_extension("json");
        self.distribution.save(&path2file, self.verbosity)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> RandomSampling for BirthDeath<P, S> {
    fn random_sample(
        &mut self,
        strategy: &SamplingStrategy,
        nb_individuals: NbIndividuals,
        rng: impl Rng + std::clone::Clone,
    ) {
        self.distribution.sample(nb_individuals, strategy, rng)
    }
}

/// A birth-death stochastic process tracking the evolution of the
/// [`EcDNADynamics`] per iterations.
#[derive(Debug, Clone)]
pub struct BirthDeathNMinusNPlus<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathNMinusNPlus<P, S> {
    pub fn with_distribution(
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamics::new(distribution, max_iter),
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}
impl<P, S> RandomSampling for BirthDeathNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for BirthDeathNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathNMinusNPlus<P, S>
{
    type Reaction = EcDNAEvent;
    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
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
            EcDNAEvent::ProliferateNMinus => {
                self.proliferation
                    .increase_nminus(&mut self.data.distribution);
            }
            EcDNAEvent::DeathNMinus => {
                self.death.decrease_nminus(&mut self.data.distribution)
            }

            EcDNAEvent::DeathNPlus => self.death.decrease_nplus(
                &mut self.data.distribution,
                rng,
                self.verbosity,
            ),

            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!("Distribution {:#?}", self.data.distribution);
        }

        self.data.nplus.push(self.data.distribution.compute_nplus());
        self.data.nminus.push(*self.data.distribution.get_nminus());
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
        state.population[2] = *self.data.distribution.get_nminus();
        state.population[3] = self.data.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                state.population
            );
        }
    }
}

/// A birth-death stochastic process tracking the evolution of the
/// [`EcDNADynamics`] per time.
#[derive(Debug, Clone)]
pub struct BirthDeathNMinusNPlusTime<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathNMinusNPlusTime<P, S> {
    pub fn with_distribution(
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        time: f32,
        max_iter: usize,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamicsTime::new(time, distribution, max_iter),
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}
impl<P, S> RandomSampling for BirthDeathNMinusNPlusTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for BirthDeathNMinusNPlusTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathNMinusNPlusTime<P, S>
{
    type Reaction = EcDNAEvent;
    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.data.ecdna_dynamics.distribution,
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
                self.proliferation.increase_nminus(
                    &mut self.data.ecdna_dynamics.distribution,
                );
            }
            EcDNAEvent::DeathNMinus => self
                .death
                .decrease_nminus(&mut self.data.ecdna_dynamics.distribution),

            EcDNAEvent::DeathNPlus => self.death.decrease_nplus(
                &mut self.data.ecdna_dynamics.distribution,
                rng,
                self.verbosity,
            ),

            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!(
                "Distribution {:#?}",
                self.data.ecdna_dynamics.distribution
            );
        }

        self.data
            .ecdna_dynamics
            .nplus
            .push(self.data.ecdna_dynamics.distribution.compute_nplus());
        self.data
            .ecdna_dynamics
            .nminus
            .push(*self.data.ecdna_dynamics.distribution.get_nminus());
        self.data.time.push(reaction.time);
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[1] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
        state.population[2] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[3] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
        if self.verbosity > 1 {
            println!(
                "Update iteration after process update, population: {:#?}",
                state.population
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct BirthDeathMean<P: EcDNAProliferation, S: Segregate> {
    distribution: EcDNADistribution,
    mean: Vec<f32>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathMean<P, S> {
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
            distribution,
            mean,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P, S> RandomSampling for BirthDeathMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for BirthDeathMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.distribution.save(
            &path2dir.join("ecdna").join(&filename).with_extension("json"),
            self.verbosity,
        )?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", path2dir)
        }
        write2file(&self.mean, &mean, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathMean<P, S>
{
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
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
        self.mean.push(self.distribution.compute_mean());
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.distribution.get_nminus();
        state.population[1] = self.distribution.compute_nplus();
        state.population[2] = *self.distribution.get_nminus();
        state.population[3] = self.distribution.compute_nplus();
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

impl<P, S> RandomSampling for BirthDeathMeanTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for BirthDeathMeanTime<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
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

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathMeanTime<P, S>
{
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        if self.verbosity > 1 {
            println!("dynamics {:#?}", self.data);
        }

        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.data.ecdna_dynamics.distribution,
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
                self.proliferation.increase_nminus(
                    &mut self.data.ecdna_dynamics.distribution,
                );
            }
            EcDNAEvent::DeathNMinus => self
                .death
                .decrease_nminus(&mut self.data.ecdna_dynamics.distribution),
            EcDNAEvent::DeathNPlus => self.death.decrease_nplus(
                &mut self.data.ecdna_dynamics.distribution,
                rng,
                self.verbosity,
            ),
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!(
                "Distribution {:#?}",
                self.data.ecdna_dynamics.distribution
            );
        }
        self.data.time.push(reaction.time);
        self.mean.push(self.data.ecdna_dynamics.distribution.compute_mean());
        self.data
            .ecdna_dynamics
            .nplus
            .push(self.data.ecdna_dynamics.distribution.compute_nplus());
        self.data
            .ecdna_dynamics
            .nminus
            .push(*self.data.ecdna_dynamics.distribution.get_nminus());
        if self.verbosity > 1 {
            println!("updated dynamics {:#?}", self.data);
        }
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[1] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
        state.population[2] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[3] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
    }
}

#[derive(Debug, Clone)]
pub struct BirthDeathMeanTimeVariance<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamicsTime,
    mean: Vec<f32>,
    variance: Vec<f32>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathMeanTimeVariance<P, S> {
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
        let mut variance = Vec::with_capacity(max_iter);
        variance.push(distribution.compute_variance());
        Ok(Self {
            data: EcDNADynamicsTime::new(time, distribution, max_iter),
            mean,
            variance,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P, S> RandomSampling for BirthDeathMeanTimeVariance<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn random_sample(
        &mut self,
        _strategy: &SamplingStrategy,
        _nb_individuals: NbIndividuals,
        _rng: impl Rng,
    ) {
        todo!()
    }
}

impl<P, S> ToFile for BirthDeathMeanTimeVariance<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(&self, path2dir: &Path, id: usize) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = id.to_string();
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename.clone());
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", path2dir)
        }
        write2file(&self.mean, &mean, None, false)?;

        let mut variance = path2dir.join("variance").join(filename);
        variance.set_extension("csv");
        if self.verbosity > 1 {
            println!("Variance data in {:#?}", path2dir)
        }
        write2file(&self.variance, &variance, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathMeanTimeVariance<P, S>
{
    type Reaction = EcDNAEvent;

    fn advance_step(
        &mut self,
        reaction: NextReaction<Self::Reaction>,
        rng: &mut impl Rng,
    ) {
        match reaction.event {
            EcDNAEvent::ProliferateNPlus => {
                if let Ok(is_uneven) = self.proliferation.increase_nplus(
                    &mut self.data.ecdna_dynamics.distribution,
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
                self.proliferation.increase_nminus(
                    &mut self.data.ecdna_dynamics.distribution,
                );
            }
            EcDNAEvent::DeathNMinus => self
                .death
                .decrease_nminus(&mut self.data.ecdna_dynamics.distribution),
            EcDNAEvent::DeathNPlus => self.death.decrease_nplus(
                &mut self.data.ecdna_dynamics.distribution,
                rng,
                self.verbosity,
            ),
            _ => unreachable!(),
        };
        if self.verbosity > 1 {
            println!(
                "Distribution {:#?}",
                self.data.ecdna_dynamics.distribution
            );
        }
        self.data.time.push(reaction.time);
        self.mean.push(self.data.ecdna_dynamics.distribution.compute_mean());
        self.variance
            .push(self.data.ecdna_dynamics.distribution.compute_variance());
        self.data
            .ecdna_dynamics
            .nplus
            .push(self.data.ecdna_dynamics.distribution.compute_nplus());
        self.data
            .ecdna_dynamics
            .nminus
            .push(*self.data.ecdna_dynamics.distribution.get_nminus());
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[1] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
        state.population[2] =
            *self.data.ecdna_dynamics.distribution.get_nminus();
        state.population[3] =
            self.data.ecdna_dynamics.distribution.compute_nplus();
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
    fn create_process_nplus_nminus_mean_time_test(
        distr: NonEmptyDistribtionWithNPlusCells,
        max_iter: u8,
        verbosity: u8,
        time: u8,
    ) -> bool {
        let time = time as f32 / 10.;
        let nplus = distr.0.compute_nplus();
        let nminus = *distr.0.get_nminus();
        let mean = distr.0.compute_mean();
        let p = BirthDeathMeanTime::new(
            time,
            Exponential {},
            Binomial,
            distr.0,
            max_iter as usize,
            verbosity,
        )
        .unwrap();
        p.data.ecdna_dynamics.distribution.compute_nplus() == nplus
            && p.data.ecdna_dynamics.distribution.get_nminus() == &nminus
            && p.data.time[0] == time
            && p.data.ecdna_dynamics.distribution.compute_mean() == mean
    }
}
