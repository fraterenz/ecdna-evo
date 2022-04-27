# Approximate Bayesian computation (ABC)
1. [Examples](#exmples)
2. [Input](#input)
3. [Ouput](#output)
4. [Implementation](#implementation)

The goal is to infer the fitness coefficient experienced by the cells with ecDNA from the data, using a Bayesian framework (`/path/to/ecdna simulate`).

## Examples
Running ABC with the mean and the ecDNA distribution, assuming no cell-death: `/path/to/ecdna abc -r 1000 --maxcells 1000000 --name sample_name --selection-range 1 3 --death1-range 0 --death2-range 0 --distribution /path/to/ecdna/distribution.json --mean /path/to/ecdna/mean.csv`, see `/path/to/ecdna abc --help`.

## Input
The user specifies the quantities available for the sample of interest to perform ABC.

The quantities given as input can be any combination of the following:
1. the **ecDNA distribution**, `csv` file or `json` file: if `csv` file, then each entry represent the copy number of each cell. If `json` file, then the file represents an histogram with keys as the ecDNA copies and entries the number of cells in sample for each copy number
2. the **mean** of the ecDNA distribution: a single entry csv file with the sample's mean of the ecdna distribution
3. the **entropy** of the ecDNA distribution: a single entry csv file with the sample's entropy of the ecdna distribution
4. the **frequency** of the cells w/ ecDNA: a single entry csv file with the sample's frequency of cells w/ any ecDNA copy

Adding information about the data (i.e. providing the program with several input quantities) will give a more precise inference of the fitness coefficient.

**ecDNA distribution file example:** `csv` file `1,0,10,1,20,0,2` or similarly `json` file `{"0": 2, "1": 2, "10": 1, "20":1}`

## Output
`abc.tar.gz` which is a archived folder, where a csv file for each run is stored.
This file has the following structure:

| Field        | Type         | Description                                                                                 |
|--------------|--------------|---------------------------------------------------------------------------------------------|
| parental_idx | int          | The idx of the subsampled run, if not present there was no subsampling                      |
| idx          | int          | Index of the run                                                                            |
| ecdna        | float        | One of the metric for ABC: the Kolmogorov-Smirnov distance between the ecDNA distributions  |
| mean         | float        | One of the metric for ABC: the relative difference between the ecDNA distribution means     |
| entropy      | float        | One of the metric for ABC: the relative difference between the ecDNA distribution entropies |
| f1           | float        | Selection coefficient of the cells w/ ecDNA                                                 |
| f2           | float        | Selection coefficient of the cells w/o ecDNA                                                |
| d1           | float        | Death coefficient of the cells w/ ecDNA                                                     |
| cells        | NbIndividuals| Number of cells in the sample when the abc rejection algorithm was ran                      |
| tumour_cells | NbIndividuals| Number of cells in the tumour population when the abc rejection algorithm was ran           |
| init_mean    | float        | Mean of the ecDNA distribution used to initialize the system                                |
| init_cells   | NbIndividuals| Number of cells used to initialize the system                                               |
| init_copies  | int          | Copies of ecDNA used to initialize the system                                               |

## Implementation
### All simulations are first saved, then filtered out while plotting
To infer the most probable fitness coefficient from the data, we compare the data provided by the user to stochastic realization of a birth-death process with exponential growth.
ABC compares the runs from the simulation (for which the fitness coefficient is known) against the real data to recover the posterior distribution of the fitness coefficient.
Instead of keeping only the appropriate runs, i.e. those similar to the patient's data, during the simulations implemented in rust, we save all the runs at first and then filter out those that have an associated distance bigger than a certain threshold during the plotting phase (python code).
Doing so we can change the threshold after the expensive simulations, i.e. without the need of rerunning all ABC simulations.

### Simulations run in parallel and can be easily merged
Since runs in ABC are independent, we can run the same experiment twice and then merge the runs from the two experiments, if they were ran using the same parameters and input.
Since runs in ABC are independent, we can run them in parallel intuitively with rust ([`rayon iter`](https://docs.rs/rayon/latest/rayon/iter/index.html) API).

