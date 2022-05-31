# Changelog
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
