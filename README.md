# Evolutionary Models of extrachromosomal DNAs
![example workflow](https://github.com/fraterenz/ecdna-evo/actions/workflows/clippy-fmt.yml/badge.svg)
![workflow](https://github.com/fraterenz/ecdna-evo/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/fraterenz/ecdna-evo/branch/master/graph/badge.svg?token=0ZLN5UWXQQ)](https://codecov.io/gh/fraterenz/ecdna-evo)

1. [Introduction](#introduction)
2. [Code organization](#code-organization)
3. [Installation](#installation): [Simulations](#simulations) and [Plots](#plots)
4. [Usage](#usage)
5. [Input and output](#input-and-output)
6. [Troubleshooting](#troubleshooting)

## Introduction
Extrachromosomal DNA (ecDNA) seems to play an important role in fostering cancer progression.
Recently, [Lange et al. 2021](https://www.biorxiv.org/content/10.1101/2021.06.11.447968v1) showed indirect arguments in favor of strong selection on ecDNA in both cell lines and cancer patients: cells with ecDNA seem to be selected and thus to have a proliferative advantage compared to cells without any ecDNA copy.

The goal of this project is to two-fold:

1. to study the evolution of ecDNA dynamics in cancer
2. to infer the selection strength present in tumours with ecDNA.

To study the impact of the ecDNAs in tumour progression we use agent-based stochastic simulations, where cells are considered as an entities that represent the whole system.
We use a Gillespie algorithm to simulate the proliferation and death of cancer cells at each time step proportional to pre-defined proliferation and death rates.

## Code organization
This package is composed by three parts:
1. small binary `ecnda` implementing a command line interface used to configure the simulations
2. a library `ecdna-evo` running the simulations
3. the plotting API `ecdnaevo`

## Installation
The simulations can be ran in Linux, Windows and macOS, however the plots are only available in Linux and macOS(?).

### Simulations
Install the rust package either from the pre-built binaries or from source, see below.
#### From binaries (recommended)
Pre-built binaries for macOS, Windows and Linux can also be downloaded from the [GitHub releases page](https://github.com/fraterenz/ecdna-evo/releases).

For older version of Linux, see [troubleshooting](#Troubleshooting).

#### From source
Need rust to be [installed](https://www.rust-lang.org/tools/install).
Download the source code and compile with cargo with the **`--release` flag**;
1. `git clone git@github.com:fraterenz/ecdna-evo.git`
2. `cargo build --release -- --help`

When building from source, the `/path/to/ecdna` (see [usage](#Usage)) will be `/path/to/cloned/ecdna-evo/target/release/ecdna`.

For older version of Linux, see [troubleshooting](#Troubleshooting).

### Plots
Additional optional step is to install the python package to be plot the results (works in Linux, might work in macOS not sure?):

1. create an environment with venv: `python3 -m venv /path/to/my/env/ecdna-evo`.
2. activate the environment: `TODO`.
3. update pip: `python3 -m pip install --upgrade pip`
4. install the python package in the environment: `pip install -r requirements.txt`.

## Usage
There are two main usages:

1. study the dynamics of ecDNA by simulating an exponentially growing tumor population carrying ecDNA copies: `/path/to/ecdna simulate --help`, see also [here](./dynamics.md).
2. infer the selection coefficients from data using approximate Bayesian computation (ABC): `/path/to/ecdna abc --help`, see also [here](./abc.md).

## Input and output
See [here](./dynamics.md) for the first usage and [here](./abc.md) for the bayesian inference framework.

## Troubleshooting
The error `/lib64/libm.so.6: version 'GLIBC_2.27' not found` in Linux usually means that the OS is too old. A solution to this problem is to download and compile the source code (need rust to be [installed](https://www.rust-lang.org/tools/install)), specifying the rust flag to [statically link C runtimes](https://doc.rust-lang.org/reference/linkage.html#static-and-dynamic-c-runtimes): `git clone` the repository and then `RUSTFLAGS='-C target-feature=+crt-static' cargo b --release --target x86_64-unknown-linux-gnu`.

