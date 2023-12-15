use anyhow::Context;
use ecdna_evo::{distribution::SamplingStrategy, RandomSampling, ToFile};
use ecdna_lib::distribution::EcDNADistribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sosa::{
    simulate, AdvanceStep, CurrentState, NbIndividuals, Options, ReactionRates,
};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

pub fn save_distribution(
    path2dir: &Path,
    distribution: &EcDNADistribution,
    rates: &[f32],
    id: usize,
    verbosity: u8,
) -> anyhow::Result<()> {
    let mut filename = match rates {
        [b0, b1] => format!("{}b0_{}b1_{}d0_{}d1_{}idx", b0, b1, 0, 0, id),
        [b0, b1, d0, d1] => {
            format!("{}b0_{}b1_{}d0_{}d1_{}idx", b0, b1, d0, d1, id)
        }
        _ => unreachable!(),
    };
    filename = filename.replace('.', "dot");
    distribution.save(
        &path2dir
            .join(format!(
                "{}cells/ecdna/{}",
                distribution.compute_nplus() + distribution.get_nminus(),
                filename
            ))
            .with_extension("json"),
        verbosity,
    )
}

pub struct Dynamics {
    pub seed: u64,
    pub path2dir: PathBuf,
    pub max_cells: NbIndividuals,
    pub options: Options,
    pub save_before_subsampling: bool,
}

pub struct Sampling {
    pub at: Vec<NbIndividuals>,
    pub strategy: SamplingStrategy,
}

impl Dynamics {
    pub fn run<P, REACTION, const NB_REACTIONS: usize>(
        &self,
        idx: usize,
        mut process: P,
        initial_state: &mut CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<NB_REACTIONS>,
        possible_reactions: &[REACTION; NB_REACTIONS],
        sampling: &Option<Sampling>,
    ) -> anyhow::Result<()>
    where
        P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>
            + Clone
            + Debug
            + ToFile<NB_REACTIONS>
            + RandomSampling,
        REACTION: std::fmt::Debug,
    {
        if self.options.verbosity > 1 {
            println!("{:#?}", process);
        }

        let stream = idx as u64;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        rng.set_stream(stream);

        let stop_reason = simulate(
            initial_state,
            rates,
            possible_reactions,
            &mut process,
            &self.options,
            &mut rng,
        );

        if self.options.verbosity > 0 {
            println!("stop reason: {:#?}", stop_reason);
        }

        if let Some(sampling) = sampling {
            for (i, sample_at) in sampling.at.iter().enumerate() {
                // happens with birth-death processes
                let no_individuals_left =
                    initial_state.population.iter().sum::<u64>() == 0u64;
                if no_individuals_left {
                    if self.options.verbosity > 0 {
                        println!("do not subsample, early stopping, due to no cells left");
                    }
                    break;
                }
                if self.options.verbosity > 0 {
                    println!(
                        "subsample {}th timepoint with {} cells from the ecDNA distribution with {:#?} cells",
                        i, sample_at, process
                    );
                }
                if self.save_before_subsampling {
                    process
                        .save(
                            &self.path2dir.join(format!(
                                "{}cells/before_subsampling",
                                sample_at
                            )),
                            rates,
                            idx,
                        )
                        .with_context(|| {
                            format!(
                                "Cannot save run {} before subsampling",
                                idx
                            )
                        })
                        .unwrap();
                }

                if self.options.verbosity > 0 {
                    println!("Subsampling");
                }
                let distribution = process
                    .random_sample(*sample_at, &mut rng)
                    .with_context(|| {
                        format!(
                            "wrong number of cells to subsample? {}",
                            sample_at
                        )
                    })
                    .unwrap();
                save_distribution(
                    &self.path2dir,
                    &distribution,
                    rates.0.as_ref(),
                    idx,
                    self.options.verbosity,
                )
                .with_context(|| "cannot save the ecDNA distribution")
                .unwrap();
            }
        } else if self.options.verbosity > 1 {
            println!("Nosubsampling");
        };
        Ok(())
    }
}
