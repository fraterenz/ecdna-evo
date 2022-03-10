/// CLI to build a configuration used to run the simulations.
use clap::{arg, command, App, ArgGroup, Command};

pub fn clap_app() -> Command<'static> {
    command!()
        .propagate_version(true)
        .arg_required_else_help(true)
        .about("Create the patient json file required for ABC inference. Add to `patient` a `sample`.")
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
        .arg(arg!(-v --verbosity ... "Verbosity"))
        .group(
            ArgGroup::new("summary")
                .required(false)
				.multiple(true)
                .args(&["mean", "frequency", "entropy"])
				.conflicts_with("distribution")
        )
}
