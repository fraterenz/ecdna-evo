/// CLI to build a configuration used to run the simulations.
use clap::{
    arg, command, AppSettings, Arg, ArgEnum, ArgGroup, Args, Command, Parser,
    Subcommand,
};
use std::path::PathBuf;

pub fn clap_app() -> Command<'static> {
    command!()
        .propagate_version(true)
        .global_setting(AppSettings::DeriveDisplayOrder)
        .subcommand_required(true)
        .subcommand(command!("abc")
            .about("Infer the most probable parameter from the patient's data using ABC")
            .arg_required_else_help(true)
            .arg(
                arg!(--infer <VALUE> "Infer the parameter given some data")
                    .possible_values(["fitness"]))
            .arg(arg!(--longitudinal "Infer the parameters using two or more sequencing experiment from the same patient. \
                                    The inference will be performed assuming same parameters during the all sequencing experiments"
            ))
            .arg(
                arg!(-p --patient <FILE> "Path to the json patient file created with `ecdna add`. See `ecdna add --help`"))
        	.arg(Arg::new("rho1")
                .long("rho1-range")
                .max_values(2)
                .required(true)
                .help(
						"The range of possible values for the proliferation rate of the cells w/ ecDNA. \
                        If one value given, do not optimize over the proliferation rate which will be fixed for all simulations. \
						If two values are given, they represent the minimal and maximal value of proliferation rate to test")
                .takes_value(true),
        	)
        	.arg(Arg::new("rho2")
                .long("rho2-range")
                .max_values(2)
                .required(true)
                .help(
						"The range of possible values for the proliferation rate of the cells w/o ecDNA. \
                        If one value given, do not optimize over the proliferation rate which will be fixed for all simulations. \
						If two values are given, they represent the minimal and maximal value of proliferation rate to test")
                .takes_value(true),
        	)
        	.arg(Arg::new("delta1")
                .long("delta1-range")
                .max_values(2)
                .required(true)
                .help(
						"The range of possible values for death coefficient of the cells w/ ecDNA. \
						If one value given, do not optimize over death1 hence simulations will have fixed dealta1 parameter. \
						If two values are given, they represent the minimal and maximal value of death rate for the NPlus cells w/ ecdna")
                .takes_value(true),
        	)
        	.arg(Arg::new("delta2")
                .long("delta2-range")
                .max_values(2)
                .required(true)
                .help(
						"The range of possible values for death coefficient of the cells w/o ecDNA. \
						If one value given, do not optimize over death2 hence simulations will have fixed delta2 parameter. \
						If two values are given, they represent the minimal and maximal value of death rate for the NPlus cells w/o ecdna")
                .takes_value(true),
        	)
        )
        .subcommand(command!("simulate")
            .about("Simulate the dynamics of the ecDNA distribution assuming exponential growth and random ecDNA segregation.")
            .arg_required_else_help(true)
            .arg(arg!(--dynamics)
                .takes_value(true)
				.required(true)
                .possible_values(
                    &["nplus", "nminus", "mean", "variance", "time"]
                )
                .multiple_values(true)
                .help(
						"Quantities computed for each iteration (dynamical).\n \
						\t- nplus: track the number of cells w/ ecDNA for each iteration.\n \
						\t- nminus: track the number of cells w/o ecDNA for each iteration.\n \
						\t- mean: track the mean of the ecDNA distribution for each iteration (computationally intensive).\n \
						\t- variance: track the variance of the ecDNA distribution for each iteration (computationally intensive).\n \
                        \t- time: track the Gillespie time for each iteration.\n"
					)
            )
            .arg(
                arg!(--patient <name>"Patient name used to save the results."))
            .arg(
                arg!(--rho1 [value] "Proliferation rate of the cells w/ ecDNA")
                    .default_value("1."))
            .arg(
                arg!(--rho2 [value] "Proliferation rate of the cells w/o ecDNA")
                    .default_value("1."))
            .arg(
                arg!(--delta1 [value] "Death rate of the cells w/ ecDNA")
                    .default_value("0."))
            .arg(
                arg!(--delta2 [value] "Death rate of the cells w/o ecDNA")
                    .default_value("0."))
            .arg(arg!(--cells [value] "Maximal number of cells to simulate")
                    .default_value("10000")
                    .help_heading("CONFIG"))
        )
        .arg(arg!(-v --verbosity ... "Verbosity")
            .global(true)
        )
        .arg(
            arg!(-d --distribution [FILE] "ecDNA distribution specifying the initial state of system from which the simulations will be started. \
                                            If not specified, start simulations with one single cell with one ecDNA copy.")
            .help_heading("CONFIG")
            .global(true)
        )
        .arg(arg!(-r --runs [value]"The number of runs run in parallel to infer the parameter.")
            .global(true)
            .default_value("100")
            .help_heading("CONFIG"))
}
