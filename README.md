# Introduction
Extrachromosomal DNA (ecDNA) seems to play an important role in fostering cancer progression. 
Recently, [Lange et al. 2021](https://www.biorxiv.org/content/10.1101/2021.06.11.447968v1) showed indirect arguments in favor of strong selection on ecDNA in both cell lines and cancer patients: cells with ecDNA seem to be selected and thus to have a proliferative advantage compared to cells without any ecDNA copy.

The goal of the project is to two-fold: 

1. to study the evolution of ecDNA dynamics in cancer 
2. to infer the selection strength present in tumours with ecDNA. 

To study the impact of the ecDNAs in tumour progression we use agent-based stochastic simulations, where we represent each cell as an entity and update the state of the system based on the number of cells present in the system at each iteration. 
We use a Gillespie algorithm to simulate the proliferation and death of cancer cells at each time step proportional to pre-defined proliferation and death rates.

# Code organization
This package is composed by three main parts: 
1. small binary implementing a command line interface used to configure the simulations `ecdna-cfg`
2. a library `ecdna-evo` performing the simulations (add link to crates.io doc)
3. the plotting API `ecdnaevo`

# Installation
https://rust-cli.github.io/book/tutorial/packaging.html 
To be able to plot the results, install also the python package. 

1. create an environment with venv: `python3 -m venv /path/to/my/env/ecdna-evo`. 
2. activate the environment: `TODO`.
3. update pip: `python3 -m pip install --upgrade pip`
4. install the python package in the environment: `pip install -r requirements.txt`.

# Usage
TODO
There are two main usages of this code:

1. study the dynamics of ecDNA by simulating an exponentially growing tumor population carrying ecDNA copies: `cargo run --release -- simulate --help` and then `python3 path/to/json.json` TODO maybe do a sh script. This generates the figures as well as 5 files (see Output section).

2. infer the selection coefficients from data using approximate Bayesian computation (ABC): `cargo run --release -- abc --help` if real patient data are present TODO a sh script, else `./abc/abc.sh 1.1 0 0.2` to generate sythentic data using fitness coefficient of 1.1, death of the nplus cells of `0` and death of nminus cells of `0.2`.

# Input and output
TODO

# Troubleshoot
If `/lib64/libm.so.6: version 'GLIBC_2.27' not found`, use the release `x86_64-unknown-linux-musl`.
