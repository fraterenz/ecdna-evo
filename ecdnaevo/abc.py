#!/usr/bn/env python
# coding: utf-8
"""Plot the bayesian inferences."""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from collections import UserDict
from pathlib import Path
from dataclasses import dataclass
from collections import UserDict
from typing import NewType, Tuple
from itertools import combinations
from ecdnaevo import commons

PLOTTING_STATS = {"ecdna": "D", "mean": "M", "frequency": "F", "entropy": "E"}
MYSHAPE = (len(PLOTTING_STATS), len(PLOTTING_STATS))
Thresholds = NewType("Thresholds", UserDict[str, float])


@dataclass
class App:
    """
    path: path to abc.tar.gz file.
    thresholds: dict with the statistics as keys ("ecdna", "mean", "frequency", "entropy")
    and the values are the thresholds. A threshold indicate the minimal required
    difference between the run and the patient's data to accept the run in ABC. The
    thresholds for "mean", "frequency" and "entropy" are relative differences
    (abs(x - x_sim) / x), whereas the ks distance of the ecDNA distribution is absolute.
    stats: plot several subplots for combinations of statistics
    """

    abc: Path
    thresholds: Thresholds
    theta: str
    path2save: Path
    stats: bool
    verbosity: bool


def infer_nb_timepoints(data: pd.DataFrame, verbosity: bool) -> int:
    nb_timepoints = data.idx.value_counts().unique()
    assert (
        nb_timepoints.shape[0] == 1
    ), "Found runs with different nb of timepoints {}".format(nb_timepoints)

    nb_timepoints = int(nb_timepoints[0])
    if verbosity:
        print("Found {} timpoints".format(nb_timepoints))
    return nb_timepoints


def load(path2abc: Path, verbosity: bool) -> Tuple[pd.DataFrame, int]:
    abc: pd.DataFrame = pd.read_csv(path2abc, header=0, low_memory=False)
    abc.drop(abc[abc.idx == "idx"].index, inplace=True)
    abc.dropna(how="all", inplace=True)
    abc.loc[:, "idx"] = abc.idx.astype("uint32")
    # abc.loc[:, abc.columns[0]] = abc[abc.columns[0]].astype("uint32")
    for col in abc.columns[2:-2]:
        abc[col] = abc[col].astype("float")
    for col in abc.columns[-2:]:
        abc[col] = abc[col].astype("uint")

    if verbosity:
        print(abc.head())
        print(abc.dtypes)
    return (abc, infer_nb_timepoints(abc, verbosity))


def run(app: App):
    """Plot posterior distribution of the fitness coefficient"""
    (abc, nb_timepoints) = load(app.abc, app.verbosity)

    if app.stats:
        path2save = commons.create_path2save(
            app.path2save,
            Path(
                "{}mean_{}frequency_{}entropy_{}ecdna_{}_subplots.pdf".format(
                    app.thresholds["mean"],
                    app.thresholds["frequency"],
                    app.thresholds["entropy"],
                    app.thresholds["ecdna"],
                    app.theta,
                )
            ),
        )
        plot_posterior_per_stats(
            app.thresholds, abc, path2save, nb_timepoints, app.theta, app.verbosity
        )
    else:
        path2save = commons.create_path2save(
            app.path2save,
            Path(
                "{}relative_{}ecdna_{}.pdf".format(
                    app.thresholds["mean"], app.thresholds["ecdna"], app.theta
                )
            ),
        )
        plot_post(
            app.thresholds, abc, path2save, nb_timepoints, app.theta, app.verbosity
        )


def build_app() -> App:
    """Parse and returns the app"""
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Plot posterior distribution (histograms) for the parameters theta inferred by ABC."
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
        "--theta",
        dest="theta",
        required=True,
        choices=["f1", "d1", "d2", "init_copies"],
        help="Parameter to infer",
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

    return App(
        abc,
        Thresholds(thresholds),
        args["theta"],
        abc.parent,
        stats,
        args["verbosity"],
    )


def accuracy(df, stats: list):
    """Compute the accuracy by statistic `stats` and grouped.
    Accuracy is computed in by counting the nb of rows with at
    least a 1 in a column of stats.
    """
    # rows for which all columns have a 1
    all_row = df[stats].all(axis=1)
    # counts
    acc_grouped = df[stats].all(axis=1).sum()
    acc_by_stat = df.loc[all_row, stats].sum(axis=0)
    # divide to get accuracy from counts
    return acc_grouped / df.shape[0], acc_by_stat / df.shape[0]


def plot_fitness(fitness, ax, title=None):
    plot_rates(fitness, (0.9, 3.1), 120, ax, title)


def plot_death(death, ax, title=None):
    plot_rates(death, (0, 1.1), 120, ax, title)


def plot_copies(copies, ax, title=None):
    plot_rates(copies, (0, copies.max()), 120, ax, title)


def plot_rates(rates, range_hist: Tuple[float, float], bins: int, ax, title=None):
    """Modify inplace the ax by plotting the hist of the posterior distribution
    `rates`. Here `title` is the title of the axis `ax`."""
    ax.hist(rates, bins=bins, range=range_hist, align="mid")
    ax.set_title(title)


def plot_posterior_per_stats(
    thresholds: Thresholds, abc, path2save, nb_timepoints, theta, verbosity
):
    """Plot the posterior distribution for the fitness coefficient
    for each combination of statistics.

    abc: DataFrame
        The entries are the values of the distance between the patient's data
        and the run, for each run. Distance is the (for now) relative difference
        for mean, frequency, and entropy, and for the ecdna is the ks distance.
    """
    fig_tot, axes = plt.subplots(*MYSHAPE, sharex=True)
    # axes[0, 0].set_xlim([0.9, 3.1])
    i = 0

    # combinations of statistics
    for r in range(1, 5):
        # iter over the combinations
        for the_thresholds in combinations(thresholds.items(), r):
            # iter over the (stats, threshold) for each combination
            ax = axes[np.unravel_index(i, MYSHAPE)]
            plot_posterior(the_thresholds, abc, ax, nb_timepoints, theta, verbosity)
            clean = ",".join(
                [
                    PLOTTING_STATS[stat]
                    for stat in "".join(
                        [e for e in ax.get_title() if e not in {"[", "]", "'", " "}]
                    ).split(",")
                ]
            )
            ax.set_title(clean, {"fontsize": 10})
            ax.tick_params(axis="both", size=2, which="major", labelsize=4)
            ax.tick_params(
                axis="both",
                size=2,
                which="major",
                labelsize=5,
            )
            fig_tot.axes.append(ax)
            i += 1

    # last axis
    ax = axes[np.unravel_index(i, MYSHAPE)]
    ax.axis("off")
    fig_tot.axes.append(ax)
    fig_tot.subplots_adjust(hspace=0.5)
    print("Saving figure to", path2save)
    fig_tot.savefig(fname=path2save, bbox_inches="tight")
    plt.close(fig_tot)


def plot_posterior(thresholds, abc, ax, nb_timepoints, theta, verbosity):
    my_query = ""
    stats = []
    for threshold in thresholds:
        my_query += "(`{}` < {}) & ".format(*threshold)
        stats.append(threshold[0])
    my_query = my_query.rstrip(" & ")
    print(my_query)
    if nb_timepoints == 1:
        to_plot = abc.loc[abc[stats].query(my_query).index, theta]
    else:
        # when multiple timepoints are present, runs can have the idx. In this
        # case, # we want to plot runs for which the query is satisfied for
        # **all** their # timepoints: groupby idx (same run with different
        # timepoints), then apply query on those timepoints, then keep run only
        # if all timepoints match query, i.e. shape[0] == timepoints. Finally,
        # take only once the fitness # coefficient to avoid saving it multiple
        # times
        to_plot = (
            abc.groupby("idx")
            .filter(lambda x: (x.query(my_query)).shape[0] == nb_timepoints)
            .drop_duplicates(["idx"])[theta]
        )
    if verbosity:
        print(to_plot)
    if theta == "f1":
        plot_fitness(to_plot, ax, list(stats))
    elif theta in {"d1", "d2"}:
        plot_death(to_plot, ax, list(stats))
    elif theta == "init_copies":
        plot_copies(to_plot, ax, list(stats))
    else:
        raise ValueError(
            "{} not valid: theta must be either `f1`, `d1`, `d2` or `init_copies`".format(
                theta
            )
        )


def plot_post(thresholds: Thresholds, abc, path2save, nb_timepoints, theta, verbosity):
    fig, ax = plt.subplots(1, 1)
    plot_posterior(thresholds.items(), abc, ax, nb_timepoints, theta, verbosity)
    # ax.tick_params(axis="both", size=2, which="major", labelsize=4)
    fig.axes.append(ax)
    fig.subplots_adjust(hspace=0.5)
    print("Saving figure to", path2save)
    fig.savefig(fname=path2save, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run(build_app())
