use std::{fs, path::Path};

use anyhow::ensure;
use ecdna_lib::distribution::SamplingStrategy;
use rand::Rng;
use sosa::{
    write2file, AdvanceStep, CurrentState, NbIndividuals, NextReaction,
    ReactionRates,
};

use crate::{
    distribution::EcDNADistribution, dynamics::SaveDynamic,
    segregation::Segregate, RandomSampling, ToFile,
};

use super::{
    dynamics::EcDNADynamics,
    proliferation::{EcDNADeath, EcDNAProliferation},
    segregation::IsUneven,
};

fn create_filename_birth_death(rates: &[f32; 4], id: usize) -> String {
    format!(
        "{}b0_{}b1_{}d0_{}d1_{}idx",
        rates[0].to_string().replace('.', ""),
        rates[1].to_string().replace('.', ""),
        rates[2].to_string().replace('.', ""),
        rates[3].to_string().replace('.', ""),
        id,
    )
}

fn create_filename_pure_birth(rates: &[f32; 2], id: usize) -> String {
    format!(
        "{}b0_{}b1_0d0_0d1_{}idx",
        rates[0].to_string().replace('.', "dot"),
        rates[1].to_string().replace('.', "dot"),
        id,
    )
}

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

impl<P, S> ToFile<2> for PureBirth<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<2>,
        id: usize,
    ) -> anyhow::Result<()> {
        //! Save the ecDNA distribution
        let path2ecdna = path2dir.join("ecdna");
        fs::create_dir_all(&path2ecdna).expect("Cannot create dir");
        let filename = create_filename_pure_birth(&rates.0, id);
        let path2file = path2ecdna.join(filename).with_extension("json");
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
        timepoints: &[f32],
        initial_time: f32,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        //! Create a pure-birth process from a non-empty `distribution`
        //! tracking the nminus and nplus cells over time.
        //!
        //! ## Error
        //! Returns error when the distribution is empty.
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamics::new(
                distribution,
                max_iter,
                timepoints,
                initial_time,
            ),
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
        todo!();
        // let (nplus, nminus) = (
        //     self.data.distribution.compute_nplus(),
        //     *self.data.distribution.get_nminus(),
        // );
        // self.data.nplus.push(nplus);
        // self.data.nminus.push(nminus);
        // if self.verbosity > 1 {
        //     println!(
        //         "NPlus/NMinus after subsampling {}, {}",
        //         self.data.nplus.last().unwrap(),
        //         self.data.nminus.last().unwrap()
        //     );
        // }
    }
}

impl<P, S> ToFile<2> for PureBirthNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<2>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");
        let filename = create_filename_pure_birth(&rates.0, id);
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                for _ in 0..times {
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
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
        timepoints: &[f32],
        initial_time: f32,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamics::new(
                distribution,
                max_iter,
                timepoints,
                initial_time,
            ),
            mean,
            proliferation,
            segregation,
            verbosity,
        })
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

impl<P, S> ToFile<2> for PureBirthMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<2>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = create_filename_pure_birth(&rates.0, id);
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", mean)
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.mean.push(self.data.distribution.compute_mean());
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                let mean = self.data.distribution.compute_mean();
                for _ in 0..times {
                    self.mean.push(mean);
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
    }

    fn update_state(&self, state: &mut CurrentState<2>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
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

impl<P: EcDNAProliferation, S: Segregate> ToFile<4> for BirthDeath<P, S> {
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<4>,
        id: usize,
    ) -> anyhow::Result<()> {
        //! Save the ecDNA distribution
        let path2ecdna = path2dir.join("ecdna");
        let filename = create_filename_birth_death(&rates.0, id);
        fs::create_dir_all(&path2ecdna).expect("Cannot create dir");
        let path2file = path2ecdna.join(filename).with_extension("json");
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
        if nb_individuals
            < self.distribution.get_nminus()
                + self.distribution.compute_nplus()
        {
            self.distribution.sample(nb_individuals, strategy, rng)
        }
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
        initial_time: f32,
        timepoints: &[f32],
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        Ok(Self {
            data: EcDNADynamics::new(
                distribution,
                max_iter,
                timepoints,
                initial_time,
            ),
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

impl<P, S> ToFile<4> for BirthDeathNMinusNPlus<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<4>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = create_filename_birth_death(&rates.0, id);
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                for _ in 0..times {
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
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

#[derive(Debug, Clone)]
pub struct BirthDeathMean<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
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
        timepoints: &[f32],
        initial_time: f32,
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        Ok(Self {
            data: EcDNADynamics::new(
                distribution,
                max_iter,
                timepoints,
                initial_time,
            ),
            proliferation,
            death: EcDNADeath,
            mean,
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

impl<P, S> ToFile<4> for BirthDeathMean<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<4>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = create_filename_birth_death(&rates.0, id);
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename);
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", mean)
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.mean.push(self.data.distribution.compute_mean());
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                let mean = self.data.distribution.compute_mean();
                for _ in 0..times {
                    self.mean.push(mean);
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
        state.population[2] = *self.data.distribution.get_nminus();
        state.population[3] = self.data.distribution.compute_nplus();
    }
}

#[derive(Debug, Clone)]
pub struct BirthDeathMeanVariance<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    mean: Vec<f32>,
    variance: Vec<f32>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathMeanVariance<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        timepoints: &[f32],
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        let mut variance = Vec::with_capacity(max_iter);
        variance.push(distribution.compute_variance());
        Ok(Self {
            data: EcDNADynamics::new(distribution, max_iter, timepoints, time),
            mean,
            variance,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P, S> RandomSampling for BirthDeathMeanVariance<P, S>
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
            println!(
                "Subsampling the ecDNA distribution with {} cells",
                nb_individuals
            );
        }
        if self.verbosity > 1 {
            println!("Before {:#?}", self.data.distribution);
        }
        if nb_individuals
            < self.data.distribution.get_nminus()
                + self.data.distribution.compute_nplus()
        {
            self.data.distribution.sample(nb_individuals, strategy, rng);
        }

        if self.verbosity > 1 {
            println!("After {:#?}", self.data.distribution);
        }
        todo!();
        // let (nplus, nminus) = (
        //     self.data.distribution.compute_nplus(),
        //     *self.data.distribution.get_nminus(),
        // );
        // self.data.nplus.push(nplus);
        // self.data.nminus.push(nminus);
        // self.data.time.push(0f32);
        // self.mean.push(self.data.distribution.compute_mean());
        // self.variance.push(self.data.distribution.compute_variance());

        // if self.verbosity > 1 {
        //     println!(
        //         "NPlus/NMinus after subsampling {}, {}",
        //         self.data.nplus.last().unwrap(),
        //         self.data.nminus.last().unwrap()
        //     );
        // }
    }
}

impl<P, S> ToFile<4> for BirthDeathMeanVariance<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<4>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = create_filename_birth_death(&rates.0, id);
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename.clone());
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", mean)
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
    for BirthDeathMeanVariance<P, S>
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.mean.push(self.data.distribution.compute_mean());
                self.variance.push(self.data.distribution.compute_variance());
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                let mean = self.data.distribution.compute_mean();
                let variance = self.data.distribution.compute_variance();
                for _ in 0..times {
                    self.mean.push(mean);
                    self.variance.push(variance);
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
        state.population[2] = *self.data.distribution.get_nminus();
        state.population[3] = self.data.distribution.compute_nplus();
    }
}

#[derive(Debug, Clone)]
pub struct BirthDeathMeanVarianceEntropy<P: EcDNAProliferation, S: Segregate> {
    data: EcDNADynamics,
    mean: Vec<f32>,
    variance: Vec<f32>,
    entropy: Vec<f32>,
    proliferation: P,
    segregation: S,
    death: EcDNADeath,
    verbosity: u8,
}

impl<P: EcDNAProliferation, S: Segregate> BirthDeathMeanVarianceEntropy<P, S> {
    pub fn new(
        time: f32,
        proliferation: P,
        segregation: S,
        distribution: EcDNADistribution,
        max_iter: usize,
        timepoints: &[f32],
        verbosity: u8,
    ) -> anyhow::Result<Self> {
        ensure!(!distribution.is_empty());
        let mut mean = Vec::with_capacity(max_iter);
        mean.push(distribution.compute_mean());
        let mut variance = Vec::with_capacity(max_iter);
        variance.push(distribution.compute_variance());
        let mut entropy = Vec::with_capacity(max_iter);
        entropy.push(distribution.compute_entropy());
        Ok(Self {
            data: EcDNADynamics::new(distribution, max_iter, timepoints, time),
            mean,
            variance,
            entropy,
            proliferation,
            death: EcDNADeath,
            segregation,
            verbosity,
        })
    }
}

impl<P, S> RandomSampling for BirthDeathMeanVarianceEntropy<P, S>
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
            println!(
                "Subsampling the ecDNA distribution with {} cells",
                nb_individuals
            );
        }
        if self.verbosity > 1 {
            println!("Before {:#?}", self.data.distribution);
        }
        if nb_individuals
            < self.data.distribution.get_nminus()
                + self.data.distribution.compute_nplus()
        {
            self.data.distribution.sample(nb_individuals, strategy, rng);
        }

        if self.verbosity > 1 {
            println!("After {:#?}", self.data.distribution);
        }
        todo!();
        // let (nplus, nminus) = (
        //     self.data.distribution.compute_nplus(),
        //     *self.data.distribution.get_nminus(),
        // );
        // self.data.nplus.push(nplus);
        // self.data.nminus.push(nminus);
        // self.data.time.push(0f32);
        // self.mean.push(self.data.distribution.compute_mean());
        // self.variance.push(self.data.distribution.compute_variance());
        // self.entropy.push(self.data.distribution.compute_entropy());

        // if self.verbosity > 1 {
        //     println!(
        //         "NPlus/NMinus after subsampling {}, {}",
        //         self.data.nplus.last().unwrap(),
        //         self.data.nminus.last().unwrap()
        //     );
        // }
    }
}

impl<P, S> ToFile<4> for BirthDeathMeanVarianceEntropy<P, S>
where
    P: EcDNAProliferation,
    S: Segregate,
{
    fn save(
        &self,
        path2dir: &Path,
        rates: &ReactionRates<4>,
        id: usize,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(path2dir).expect("Cannot create dir");

        let filename = create_filename_birth_death(&rates.0, id);
        if self.verbosity > 1 {
            println!("Saving data in {:#?}", path2dir)
        }

        self.data.save(path2dir, &filename, self.verbosity)?;

        let mut mean = path2dir.join("mean").join(filename.clone());
        mean.set_extension("csv");
        if self.verbosity > 1 {
            println!("Mean data in {:#?}", mean)
        }
        write2file(&self.mean, &mean, None, false)?;

        let mut variance = path2dir.join("variance").join(filename.clone());
        variance.set_extension("csv");
        if self.verbosity > 1 {
            println!("Variance data in {:#?}", path2dir)
        }
        write2file(&self.variance, &variance, None, false)?;

        let mut entropy = path2dir.join("entropy").join(filename);
        entropy.set_extension("csv");
        if self.verbosity > 1 {
            println!("entropy data in {:#?}", path2dir)
        }
        write2file(&self.entropy, &entropy, None, false)?;
        Ok(())
    }
}

impl<P: EcDNAProliferation, S: Segregate> AdvanceStep<4>
    for BirthDeathMeanVarianceEntropy<P, S>
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

        // store absolute time
        self.data.time += reaction.time;

        match self.data.is_time_to_save(self.data.time, self.verbosity) {
            SaveDynamic::DoNotSave => {}
            SaveDynamic::SaveNewOne => {
                self.mean.push(self.data.distribution.compute_mean());
                self.variance.push(self.data.distribution.compute_variance());
                self.entropy.push(self.data.distribution.compute_entropy());
                self.data.update_nplus_nminus(
                    self.data.distribution.compute_nplus(),
                    *self.data.distribution.get_nminus(),
                );
            }
            SaveDynamic::SavePreviousOne { times } => {
                let mean = self.data.distribution.compute_mean();
                let variance = self.data.distribution.compute_variance();
                let entropy = self.data.distribution.compute_entropy();
                for _ in 0..times {
                    self.mean.push(mean);
                    self.variance.push(variance);
                    self.entropy.push(entropy);
                    self.data.update_nplus_nminus(
                        self.data.distribution.compute_nplus(),
                        *self.data.distribution.get_nminus(),
                    );
                }
            }
        };
    }

    fn update_state(&self, state: &mut CurrentState<4>) {
        state.population[0] = *self.data.distribution.get_nminus();
        state.population[1] = self.data.distribution.compute_nplus();
        state.population[2] = *self.data.distribution.get_nminus();
        state.population[3] = self.data.distribution.compute_nplus();
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
        let p = BirthDeathMean::new(
            Exponential {},
            Binomial,
            distr.0,
            max_iter as usize,
            &[1., 2.],
            time,
            verbosity,
        )
        .unwrap();
        p.data.distribution.compute_nplus() == nplus
            && p.data.distribution.get_nminus() == &nminus
            && p.data.time == time
            && p.data.distribution.compute_mean() == mean
    }
}
