use chrono::Utc;

use crate::clap_app::Cli;

mod clap_app;
mod dynamics;
mod run;

/// The number of max iterations that we max simulate compared to the cells.
/// The max number of iterations will be MAX_ITER * cells.
pub const MAX_ITER: usize = 3;
/// The number of time a run restarts when there are no individuals left because
/// of the high cell death
const NB_RESTARTS: u64 = 30;

/// Run the simulations
pub trait Simulate {
    fn run(self: Box<Self>) -> anyhow::Result<()>;
}

fn main() {
    let cli = Cli::build();
    match cli {
        Ok(cli) => {
            std::process::exit(match cli.run() {
                Ok(_) => {
                    println!("{} End simulation", Utc::now(),);
                    0
                }
                Err(err) => {
                    eprintln!("{} Error: {:?}", Utc::now(), err);
                    1
                }
            });
        }
        Err(err) => {
            eprintln!(
                "{} Error while building the cli: {:?}",
                Utc::now(),
                err
            );
            std::process::exit(1);
        }
    }
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;

    Cli::command().debug_assert()
}
