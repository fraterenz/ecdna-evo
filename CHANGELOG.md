# Changelog

## 0.13.9
The pure-birth process with time, nplus and nminus cells.

## 0.13.8
### Fixed
Fix the birth-death process with time, nplus and nminus cells.

## 0.13.7
The birth-death process with time, nplus and nminus cells.

## 0.13.6
### Fixed
Saving `times` only once.

## 0.13.5
The birth-death process with time, mean and variance.

**`ecdna_lib` version**: 0.5.4

## 0.13.4
The birth-death process with time and mean.

## 0.13.3
Remove `simulate`.

**`ecdna_lib` version**: 0.5.3

**`ssa` version**: 0.6.0

## 0.13.2
Using git `tag` attribute in `Cargo.toml`.

## 0.13.1
**`ecdna_lib` version**: 0.3.2

When `subsample` arg of clap is 0, take the full ecDNA distribution as a sample, which makes sense only if `--sampling-strategy` is *not* `uniform`.

## 0.13.0
**`ecdna_lib` version**: 0.3.0

New clap arg `sampling_strategy` and update `ecdna-lib` version to sample the ecDNA distributions according to different sampling strategies.

## 0.12.0
**`ssa` version**: 0.5.0

Refactor: rename `PureBirthNoDynamics` to `PureBirth` and `PureBirth` to `PureBirthNMinusNPlus`. The default process simulated by the `dynamics` is now `PureBirth` not `PureBirthNoDynamics`.

New flag `nplus-nminus` to generate `PureBirthNMinusNPlus`.

## 0.11.0
**`ssa` version**: 0.4.0

## 0.10.1
Update `ecdna-evo` version in `Cargo.toml`


## 0.10.0
This crate has now also its library, since in `ssa` v0.3.0 `enum_dispatch`ed heterogenous collection of processes has been removed.

**`ssa` version**: 0.3.0

## 0.9.0
Save a single `abc.csv` file for all runs instead of the `abc` folder with inside all runs.

## 0.8.1
Update `ssa` version

**`ssa` version:** 0.2.1

## 0.8.0
There are two main changes:
1. refactor: create a new lib `ecdna-lib` and moved `abc.rs` and the ecDNA distribution there.
2. Use `ChaCha8` as rng instead of `Pcg`, so we can set the stream in independent threads, see [here](https://rust-random.github.io/book/guide-parallel.html).

**`ssa` version:** 0.2.0

## 0.7.2
Update `ssa` version.

**`ssa` version:** 0.1.2
## 0.7.1
Some many bug fixes.
The previous version (v.0.7.0) contains so much bugs that it should not be used.

**`ssa` version:** 0.1.1

## 0.7.0
The code uses now the `ssa` lib.

Perform ABC with subsampling (one timepoint only for now) with only pure-birth.
ABC infers only the fitness coefficient, that is the birth-rate of cells with
ecDNAs `b1`.

**Advantages:**

- reduced by approximatively one half the code base
- more general, should be directly applicable the other problems/processes (see `ssa::hsc`)

## 0.4.4
### Fixed
- Fix path to save data with cell death
- Fix computation of the variance

### Changes
- Save ecDNA statistics also before subsampling `dynamics`
- Add segregation mode to perform the nonminus experiment
## 0.4.3
### Changes
- Ignore the test `data::tests::ecdna_undersample_reproducible_different_trials`.
- Add new dynamics `Uneven` which tracks the number of complete uneven segregation
events.
- Simplify `any_individual_left` by removing overflow checks.

## 0.4.2
### Fixed
- Sampling was performed with replacement, now without replacement (might
need more memory)
- Remove `results` folder from preprocess app
- The computation of the `ks_distance` has changed by taking into account the
loss of precision in converting `u16` to `f16`
- Dont panic when computing the variance when there are no N+ cells, but return 0

## 0.4.1
Remove data from the project: folder `results` is now in `~/ecdna-results`.
### Added
- Option `--savedir` for `preprocess`

### Fixed
- Avoid sample twice when multiple timepoints before saving the statistics (
both `abc` and `simulate`).
- Bug in the cell culture experiment with subsampling: restarting from one
subsample from the whole distribution with correct `idx`, issue #65

## 0.4.0
Removed plotting library into another git folder
## 0.3.3
### Added
- When plotting abc results, the flag `--export` can save the runs that been filtered based on the thresholds (to for instance make the plots in Mathematica)

### Fixed
- Remove the `CONFIG` options from `preprocess` subcommand.

## 0.3.0
- Merged `preprocess` into `ecdna` whihc has now three commands: `simulate`, `abc` and `preprocess`.
### Added
- New file `timepoint` in output of abc inference `abc.tar.gz`.

## 0.2.0
### Added
- Subsample option.

- New binary `preprocess`.

- Inference of multiple timepoints (longitudinal analysis) with patient and cell
lines.

- The patient growth strategy implies that after the first timepoint, tumour
growth is restarted from the whole population. Whereas cell line growth
strategy implies that the tumour growth restart from the subsample.

- New python plots.

- Codecov

### Fixed
- Tarballs with relative paths

## 0.1.1
### Added

- The ecDNA distribution is a vector of `u16` The ecDNA distribution is a vector
where each entries is a cell and the content is the number of ecDNA copies the
cell has.

- The ecDNA distribution is saved as a JSON, i.e. as a mapping where the keys are
the ecDNA copies and the entries are the number of cells for each key
(histogram).

- Implement the [state pattern](https://doc.rust-lang.org/book/ch17-03-oo-design-patterns.html)
to pass the ecDNA distribution by consuming the `Run` instead of deep copy.

- Introduce subsampling: run the simulations for a number of cells, then run ABC
with 100 subsamples for each run. In the end, the number of tested simulations
will be 100 * the number of runs.

### Fixed
- Fix bug in the computation of the ks distance.
