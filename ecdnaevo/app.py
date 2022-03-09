#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
from collections import UserDict
from pathlib import Path
from dataclasses import dataclass
from ecdnaevo import plots


@dataclass
class App:
    """
    paths: MyPaths
    thresholds: dict with the statistics as keys ("ecdna", "mean", "frequency", "entropy")
    and the values are the thresholds. A threshold indicate the minimal required
    difference between the run and the patient's data to accept the run in ABC. The
    thresholds for "mean", "frequency" and "entropy" are relative differences
    (abs(x - x_sim) / x), whereas the ks distance of the ecDNA distribution is absolute.
    stats: plot several subplots for combinations of statistics
    """

    abc: Path
    thresholds: plots.Thresholds
    path2save: Path
    stats: bool
    nb_timepoints: int
    verbosity: bool


def load(path2abc: Path, verbosity: bool) -> pd.DataFrame:
    abc: pd.DataFrame = pd.read_csv(path2abc, header=0, low_memory=False)
    abc.drop(abc[abc.idx == "idx"].index, inplace=True)
    abc.dropna(how="all", inplace=True)
    abc.loc[:, "idx"] = abc.idx.astype("uint32")
    # print(abc.loc[abc[abc.columns[0]].isna().index, :])
    # abc.loc[:, abc.columns[0]] = abc[abc.columns[0]].astype("uint32")
    for col in abc.columns[2:]:
        abc[col] = abc[col].astype("float")
    if verbosity:
        print(abc.head())
    return abc


def create_path2save(path2dir: Path, filename: Path) -> Path:
    assert path2dir.is_dir()
    path = path2dir / filename
    path.touch(exist_ok=False)
    return path


def run(app: App):
    """Plot posterior distribution of the fitness coefficient"""
    abc = load(app.abc, app.verbosity)

    if app.stats:
        path2save = create_path2save(
            app.path2save,
            Path(
                "{}mean_{}frequency_{}entropy_{}ecdna_fitness_raw_subplots.pdf".format(
                    app.thresholds["mean"],
                    app.thresholds["frequency"],
                    app.thresholds["entropy"],
                    app.thresholds["ecdna"],
                )
            ),
        )
        plots.plot_posterior_per_stats(app.thresholds, abc, path2save, app.nb_timepoints)
    else:
        path2save = create_path2save(
            app.path2save,
            Path(
                "{}relative_{}ecdna_fitness_raw.pdf".format(
                    app.thresholds["mean"], app.thresholds["ecdna"]
                )
            ),
        )
        plots.plot_post(app.thresholds, abc, path2save, app.nb_timepoints)


def build_app() -> App:
    """Parse and returns the app"""
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Plot posterior distribution (histograms) for ABC inference."
    )

    parser.add_argument(
        "--stats",
        dest="stats",
        required=False,
        action="store_true",
        default=False,
        help="Plot posterior for combinations of statistics",
    )

    parser.add_argument(
        "mean",
        type=int,
        help="Integer specifying the relative difference percentage in the mean threshold for which the posterior will be plotted.",
    )

    parser.add_argument(
        "frequency",
        type=int,
        help="Integer specifying the relative difference percentage in the frequency threshold for which the posterior will be plotted.",
    )

    parser.add_argument(
        "entropy",
        type=int,
        help="Integer specifying the relative difference percentage threshold in the entropy for which the posterior will be plotted.",
    )

    parser.add_argument(
        "ecdna",
        type=float,
        help="Float specifying the distance used to accept runs based on the ks distance of the ecDNA distribution",
    )

    parser.add_argument(
        "--timepoints",
        required=False,
        default=1,
        type=int,
        help="Optional int specifying the number of sequencing timepoints available for the patient's sample. Default 1."
    )

    parser.add_argument(
        "--abc",
        metavar="FILE",
        dest="path2abc",
        required=True,
        type=str,
        help="Path to tarball file abc.tar.gz, where the output of the ABC inference can be found (csv files for each run)",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        default=False,
        help="increase output verbosity",
    )

    args = vars(parser.parse_args())

    thresholds = UserDict(
        {
            k: float(val / 100)
            for (k, val) in zip(
                ["mean", "frequency", "entropy"],
                [int(args["mean"]), int(args["frequency"]), int(args["entropy"])],
            )
        }
    )
    thresholds["ecdna"] = float(args["ecdna"])
    abc = Path(args["path2abc"])
    stats = args["stats"]
    assert abc.parent.is_dir()

    return App(abc, plots.Thresholds(thresholds), abc.parent, stats, args["timepoints"], args["verbosity"])
