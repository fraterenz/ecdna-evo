# Changelog
## 0.1.1
### Added
The ecDNA distribution is a vector of `u16`
The ecDNA distribution is a vector where each entries is a cell and the content is the number of ecDNA copies the cell has.
The ecDNA distribution is saved as a JSON, i.e. as a mapping where the keys are the ecDNA copies and the entries are the number of cells for each key (histogram).
Implement the [state pattern](https://doc.rust-lang.org/book/ch17-03-oo-design-patterns.html) to pass the ecDNA distribution by consuming the `Run` instead of deep copy.
Introduce subsampling: run the simulations for a number of cells, then run ABC with 100 subsamples for each run. In the end, the number of tested simulations will be 100 * the number of runs.

### Fixed
Fix bug in the computation of the ks distance.
