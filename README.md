# Evolutionary Models of extrachromosomal DNAs
![example workflow](https://github.com/fraterenz/ecdna-evo/actions/workflows/clippy-fmt.yml/badge.svg)
![workflow](https://github.com/fraterenz/ecdna-evo/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/fraterenz/ecdna-evo/branch/master/graph/badge.svg?token=0ZLN5UWXQQ)](https://codecov.io/gh/fraterenz/ecdna-evo)

1. [Introduction](#introduction)
2. [Code organization](#code-organization)
3. [Installation](#installation): [Simulations](#simulations) and [Plots](#plots)
4. [Usage](#usage)
5. [Input and output](#input-and-output)

## Introduction
Extrachromosomal DNA (ecDNA) seems to play an important role in fostering
cancer progression. Recently, [Lange et al. 2021](https://www.biorxiv.org/content/10.1101/2021.06.11.447968v1)
showed indirect arguments in favor of strong selection on ecDNA in both cell
lines and cancer patients: cells with ecDNA seem to be selected and thus to
have a proliferative advantage compared to cells without any ecDNA copy.

The goal of this project is to two-fold:

1. to study the evolution of ecDNA dynamics in cancer
2. to infer the proliferative advantage (i.e. selection strength) present in
   tumours with ecDNA.

To study the impact of the ecDNAs in tumour progression we use agent-based
stochastic simulations. We use a Gillespie algorithm to simulate the
proliferation and death of cancer cells at each time step proportional to
pre-defined proliferation and death rates.

## Code organization
The most important packages are:

1. the binary `ecnda` implementing a command line interface used to configure
   the simulations
2. the python plotting API `ecdna-plot`

## Installation
The simulations can be ran in Linux, Windows and macOS, however the plots are
only available in Linux and macOS(?).

### Simulations
Install the rust package either from the pre-built binaries or from source, see
below.

#### From binaries (recommended)
Pre-built binaries for macOS, Windows and Linux can also be downloaded from the
[GitHub releases page](https://github.com/fraterenz/ecdna-evo/releases).

For older version of Linux or is something does not work, install
[from source](#from-source).

#### From source
Need rust to be [installed](https://www.rust-lang.org/tools/install). Download
the source code and compile with cargo with the **`--release` flag**;
1. `git clone git@github.com:fraterenz/ecdna-evo.git`
2. `cargo build --release -- --help`

When building from source, the `/path/to/ecdna` (see [usage](#Usage)) will be
`/path/to/ecdna-evo/target/release/ecdna`.

### Plots
Additional optional step is to install the python package to be plot the
results (works in Linux, might work in macOS not sure?):

1. create an environment with
   [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):
   `python3 -m venv /path/to/my/env/ecdna-evo`.
2. activate the environment: `source /path/to/my/env/bin/activate`.
3. update pip: `python3 -m pip install --upgrade pip`
4. install the python package in the environment: `pip install -r
   requirements.txt`.

## Usage
There are two main usages:

1. study the dynamics of ecDNA by simulating an exponentially growing tumor
   population carrying ecDNA copies: `/path/to/ecdna simulate --help`, see also
   [here](./dynamics.md).
2. infer the proliferation advantage of cells with ecDNA copies ($\rho_1$) from
   data using approximate Bayesian computation (ABC): `/path/to/ecdna abc --help`,
   see also [here](./abc.md).

#### Example
When prebuilt binaries are used, replace in the example
`./target/release/ecdna` by `/path/to/ecdna`, where `/path/to/ecdna` is the
path to the ecdna binaries.

1. Simulate 10 tumour growths (10000 cells each) with $\rho_1$ equals to 1
   (neutral case) using the code compiled from source (see
   `./target/release/ecdna simulate --help`):
```shell
# simulate tumour growth
./target/release/ecdna simulate --cells 10000 --runs 10 --rho1 1 --dynamics nplus nminus --patient example
```

2. Optional step is to plot the dynamics
```shell
# activate your python env
source /path/to/my/env/bin/activate
python3 -m ecdna-plot.dynamics --nplus results/example/10000samples10000cells/nplus.tar.gz --nminus results/example/10000samples10000cells/nminus.tar.gz --save results/example/10000samples10000cells
```

3. Prepare the data for the abc inference, add to the patient `example` one sample
`sample1` defined by the ecdna distribution
`results/example/10000samples10000cells/0/ecdna/0.json`, this sample having an
estimated population of 10000 tumour cells (see
`./target/release/preproces --help`):
```shell
./target/release/ecdna preprocess example sample1 10000 --distribution results/example/10000samples10000cells/0/ecdna/0.json
```

4. Now perform the bayesian inference.
Performing the bayesian inference with more the runs will generate more
accurate results, but will also take more time.
Infer the proliferation advantage and the initial copy number using 1000 runs
for the patient `example` (see `./target/release/ecdna abc --help`):
```shell
./target/release/ecdna abc --runs 1000 --rho1-range 1 3 --rho2-range 1 --delta1-range 0 --delta2-range 0 --copies-range 1 20 --patient results/preprocessed/example.json
 ```

5. Finally, plot the posterior distributions by keeping runs with distance metrics
smaller than 0.1 (see `ecdna-plot.abc --help`)
```shell
# activate your python env
source /path/to/my/env/bin/activate
# plot with thresholds 0.1 0.1 0.1 0.1
python3 -m ecdna-plot.abc --theta f1 copies --abc results/example/abc.tar.gz 10 10 10 0.1
```

## Input and output
See [here](./dynamics.md) for the first usage and [here](./abc.md) for the
bayesian inference framework.

