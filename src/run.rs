use enum_dispatch::enum_dispatch;
use rand_pcg::Pcg64Mcg;
use ssa::process::{
    Iterate, Process, RandomSampling, SimState, StopReason, ToFile,
};
use ssa::NbIndividuals;
use std::path::Path;

/// Simulation of an exponentially growing tumour, that is one realization of
/// the stochastic birth-death process.
///
/// The `Run` uses the [typestate pattern]. The possible states are [`Started`]
/// and [`Ended`].
///
/// [typestate pattern]: https://github.com/cbiffle/m4vga-rs/blob/a1e2ba47eaeb4864f0d8b97637611d9460ce5c4d/notes/20190131-typestate.md
#[derive(Clone, Debug)]
pub struct Run<S: RunState> {
    state: S,

    idx: usize,
    bd_process: Process,
    /// Number of times the run has been restarted (longitudinal analyses).
    restarted: u8,
    growth: Growth,
    rng: Pcg64Mcg,
    pub verbosity: u8,
}

/// The simulation of the run has started, the stochastic birth-death process
/// has started looping over the iterations.
#[derive(Debug)]
pub struct Started;

/// The simulation of the run has ended, which is ready to be saved.
#[derive(Clone, Debug)]
pub struct Ended {
    /// Gillespie time at the end of the run
    gillespie_time: f32,
    /// The iteration number at which the run has stopped
    last_iter: usize,
    /// The run idx from which the sample was taken.
    sampled_run: Option<usize>,
}

pub trait RunState {}
impl RunState for Started {}
impl RunState for Ended {}

/// Create a new `Run` continuing from the last simulated event: the run will
/// start with the data of the ended `Run`.
impl From<Run<Ended>> for Run<Started> {
    fn from(run: Run<Ended>) -> Self {
        let restarted = run.restarted + 1;
        let state = run.state.into();

        Run {
            idx: run.idx,
            bd_process: run.bd_process,
            state,
            restarted,
            growth: run.growth,
            rng: run.rng,
            verbosity: run.verbosity,
        }
    }
}

impl From<Ended> for Started {
    fn from(_: Ended) -> Self {
        todo!("Pass last_iter!");
        Started
    }
}

impl Run<Started> {
    pub fn new(
        idx: usize,
        bd_process: Process,
        growth: Growth,
        iter: usize,
        rng: Pcg64Mcg,
        verbosity: u8,
    ) -> Self {
        //! Initialize a stoachastic realization of a birth-death process.
        Run {
            idx,
            bd_process,
            state: Started,
            restarted: 0,
            growth,
            rng,
            verbosity,
        }
    }

    pub fn simulate(mut self, init_iter: usize) -> (Run<Ended>, StopReason) {
        //! Simulate one realisation of the birth-death stochastic [`Process`].
        let mut iter = init_iter;
        let mut previous_time = 0f32;
        if self.verbosity > 0 {
            println!(
                "Starting simulation with iter: {}, time: {}",
                iter, previous_time
            );
        }

        let (condition, time) = {
            loop {
                let (state, reaction) =
                    self.bd_process.next_reaction(iter, &mut self.rng);
                if self.verbosity > 1 {
                    println!("State: {:#?}, reaction: {:#?}", state, reaction);
                }
                match state {
                    SimState::Continue => {
                        // unwrap is safe since SimState::Continue returns always
                        // something (i.e. not None).
                        let reaction = reaction.unwrap();
                        previous_time = reaction.time;
                        self.bd_process
                            .store_iteration(reaction, &mut self.rng);
                        self.bd_process.update_iteration();
                        iter += 1;
                    }
                    SimState::Stop(condition) => {
                        break (condition, previous_time)
                    }
                }
            }
        };

        (
            Run {
                idx: self.idx,
                bd_process: self.bd_process,
                state: Ended {
                    gillespie_time: time,
                    sampled_run: None,
                    last_iter: iter,
                },
                restarted: self.restarted,
                growth: self.growth,
                rng: self.rng,
                verbosity: self.verbosity,
            },
            condition,
        )
    }
}

impl Run<Ended> {
    pub fn get_parental_run(&self) -> &Option<usize> {
        //! The idx for the sampled run
        &self.state.sampled_run
    }

    pub fn undersample(self, nb_cells: &NbIndividuals, idx: usize) -> Self {
        //! Returns a copy of the run with a subsampled process.
        match self.bd_process {
            Process::EcDNAProcess(p) => Run {
                idx,
                bd_process: p.random_sample(nb_cells).into(),
                state: Ended {
                    gillespie_time: self.state.gillespie_time,
                    last_iter: self.state.last_iter,
                    sampled_run: Some(self.idx),
                },
                restarted: self.restarted,
                growth: self.growth,
                rng: self.rng,
                verbosity: self.verbosity,
            },
            _ => todo!(),
        }
    }

    pub fn set_iter(&mut self, iter: usize) {
        self.state.last_iter = iter
    }

    pub fn save(&self, path2dir: &Path) -> anyhow::Result<()> {
        //! Save the process to file.
        self.bd_process.save(path2dir, self.idx)
    }
}

#[enum_dispatch]
/// When taking a subsample of the whole population, specify how to continue the
/// cell growth: either from a sample or from the whole tumour population.
pub trait ContinueGrowth {
    fn restart_growth(
        &self,
        run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>>;
}

/// Specify how to growth the population after a subsample has been taken (biospy
/// or cell culture).
#[enum_dispatch(ContinueGrowth)]
#[derive(Debug, Clone)]
pub enum Growth {
    /// Cell culture: growth restart from the subsample of the whole population
    /// since the subsample is a new cell culture.
    CellCulture,
    /// Patient tumour: growth continues from the whole population since the
    /// subsample is just a biopsy.
    PatientStudy,
}

#[derive(Debug, Default, Clone)]
pub struct CellCulture;

impl CellCulture {
    pub fn new() -> Self {
        CellCulture
    }
}

impl ContinueGrowth for CellCulture {
    fn restart_growth(
        &self,
        mut run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>> {
        //! In cell culture experiments, growth restart from subsample of the
        //! whole population.
        todo!();
        run.idx = run.get_parental_run().unwrap();
        run.set_iter(0);
        Ok(run.into())
    }
}

#[derive(Debug, Default, Clone)]
pub struct PatientStudy;

impl ContinueGrowth for PatientStudy {
    fn restart_growth(
        &self,
        run: Run<Ended>,
        sample_size: &NbIndividuals,
    ) -> anyhow::Result<Run<Started>> {
        //! In patient studies, growth restart from the whole population.
        Ok(run.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_growth_test() {
        todo!()
    }
}
