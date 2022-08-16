use clap::{arg, command, AppSettings, Arg, ArgGroup, Command};

/// CLI to build a configuration used to run the simulations.

pub fn clap_app() -> Command<'static> {
    command!()
        .propagate_version(true)
        .global_setting(AppSettings::DeriveDisplayOrder)
        .subcommand_required(true)
        .subcommand(command!("abc")
            .about("Infer the most probable set of parameters from the patient's data using ABC")
            .arg_required_else_help(true)
            .arg(
                arg!(-p --patient <FILE> "Path to the json patient file created with `preprocess` command. See `preprocess --help`"))
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
        	.arg(Arg::new("init_copies")
                .long("copies-range")
                .max_values(2)
                .required(false)
                .conflicts_with("distribution")
                .default_value("1")
                .help(
						"The range of possible ecDNA copies to initialized the first cells carrying ecDNA copies. \
                        If one value given, do not optimize over the initial ecDNA copies which will be fixed for all simulations. \
						If two values are given, they represent the minimal and maximal value of ecDNA copies to test")
                .takes_value(true),
            )
            .arg(
                arg!(-d --distribution [FILE] "ecDNA distribution specifying the initial state of system from which the simulations will be started. \
                                                If not specified, start simulations with one single cell with one ecDNA copy.")
                .help_heading("CONFIG")
            )
            .arg(arg!(-r --runs [value] "The number of runs run in parallel to infer the parameter.")
                .default_value("100")
                .help_heading("CONFIG"))
            .arg(arg!(-s --seed [value] "The seed to reproduce the same results over different experiments")
                .help_heading("CONFIG"))
            .arg(arg!(--culture "Whether to run cell culture experiment, i.e. after one subsample is taken, tumour growth restart from that sample")
                .help_heading("CONFIG"))
        )
        .subcommand(command!("preprocess")
            .about("Create the patient json file required for ABC inference. Add to `patient` a `sample`.")
			.arg_required_else_help(true)
        	.arg(
        	    arg!(<patient> "Name of the patient for whom we add a new sample"))
        	.arg(
        	    arg!(<sample> "Name of the sample to add")
        	        .help_heading("SAMPLE"))
        	.arg(
        	    arg!(<size> "Estimation of the number of tumour cells size at sample collection")
        	        .help_heading("SAMPLE"))
        	.arg(
        	    arg!(--distribution [FILE] "ecDNA distribution of the sample")
        	        .required_unless_present("summary")
        	        .help_heading("SAMPLE"))
        	.arg(
        	    arg!(--mean [VALUE] "Mean of the ecDNA distribution of the sample")
        	        .help_heading("SAMPLE"))
        	.arg(
        	    arg!(--frequency [VALUE] "Frequency of cells any w/ ecDNA copies within the sample")
        	        .help_heading("SAMPLE"))
        	.arg(
        	    arg!(--entropy [VALUE] "Entropy of the ecDNA distribution of the sample")
        	        .help_heading("SAMPLE"))
        	.group(
        	    ArgGroup::new("summary")
        	        .required(false)
					.multiple(true)
        	        .args(&["mean", "frequency", "entropy"])
					.conflicts_with("distribution")
        	)
        )
        .subcommand(command!("simulate")
            .about("Simulate the dynamics of the ecDNA distribution assuming exponential growth and random ecDNA segregation.")
            .arg_required_else_help(true)
            .arg(arg!(--dynamics ...)
                .takes_value(true)
                .possible_values(
                    &["nplus", "nminus", "mean", "moments", "time", "uneven"]
                )
                .multiple_values(true)
                .help(
						"Quantities computed for each iteration (dynamical).\n\
						\t- nplus: track the number of cells w/ ecDNA for each iteration.\n\
						\t- nminus: track the number of cells w/o ecDNA for each iteration.\n\
						\t- mean: track the mean of the ecDNA distribution for each iteration (computationally intensive).\n\
						\t- moments: track the variance and the mean of the ecDNA distribution for each iteration (computationally intensive).\n\
                        \t- time: track the Gillespie time for each iteration.\n\
                        \t- uneven: track the number of complete uneven segregations.\n\
                        If none is specified, simulate nplus and nminus.\n"
					)
            )
            .arg(Arg::new("sizes")
                .long("sizes")
                .multiple_values(true)
                .takes_value(true)
                .requires("samples")
                .required(false)
                .help("Number of cells present in the tumour when samples have been collected. The number of occurences of this argument must match those of `samples`."))
            .arg(Arg::new("samples")
                .long("samples")
                .multiple_values(true)
                .takes_value(true)
                .requires("sizes")
                .required(false)
                .help("Number of cells present in the samples. When this argument is present, must also specify the argument `sizes` for each `samples` value."))
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
            .arg(
                arg!(-d --distribution [FILE] "ecDNA distribution specifying the initial state of system from which the simulations will be started. \
                                                If not specified, start simulations with one single cell with one ecDNA copy.")
                .help_heading("CONFIG")
            )
            .arg(arg!(-r --runs [value] "The number of runs run in parallel to infer the parameter.")
                .default_value("100")
                .help_heading("CONFIG"))
            .arg(arg!(-s --seed [value] "The seed to reproduce the same results over different experiments")
                .help_heading("CONFIG"))
            .arg(arg!(--culture "Whether to run cell culture experiment, i.e. after one subsample is taken, tumour growth restart from that sample")
                .help_heading("CONFIG"))
        )
        .arg(arg!(-v --verbosity ... "Verbosity")
            .global(true)
        )
        .arg(
            Arg::new("savedir")
            .long("savedir")
            .required(false)
            .takes_value(true)
            .help_heading("CONFIG")
            .help("Full path to the directory used to store the results. If not specified, use the current dir")
            .global(true)
        )
}
