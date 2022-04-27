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
from typing import NewType, Tuple, List
from itertools import combinations
from . import commons

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
    theta: List[str]
    path2save: Path
    stats: bool
    verbosity: bool


def abc_is_subsampled(data: pd.DataFrame) -> bool:
    """When the runs have been subsampled before running ABC"""
    return not data["parental_idx"].isna().any()


def infer_nb_timepoints(data: pd.DataFrame, verbosity: bool) -> int:
    if abc_is_subsampled(data):
        nb_timepoints = (
            data[["parental_idx", "idx", "cells"]]
            .groupby(["parental_idx", "idx"])
            .count()
            .cells.unique()
        )
    else:
        nb_timepoints = data[["idx", "cells"]].groupby("idx").count().cells.unique()
    assert (
        nb_timepoints.shape[0] == 1
    ), "Found runs with different nb of timepoints {}".format(nb_timepoints)

    nb_timepoints = int(nb_timepoints[0])
    if verbosity:
        print("Found {} timepoints".format(nb_timepoints))
    return nb_timepoints


def abc_longitudinal(
    data: pd.DataFrame, my_query: str, nb_timepoints: int, verbosity: bool
) -> pd.DataFrame:
    """when multiple timepoints are present, runs can have the idx. In this
    case, we want to plot runs for which the query is satisfied for **all**
    their timepoints: groupby idx (same run with different nb timepoints),
    then apply query on those timepoints, then keep run only if all timepoints
    match query, i.e. shape[0] == timepoints. Finally, take only once the
    fitness coefficient to avoid saving it multiple times"""
    return (
        data.groupby(["parental_idx", "idx"]).filter(
            lambda x: (x.query(my_query)).shape[0] == nb_timepoints
        )
        # .drop_duplicates(["parental_idx", "idx"])
    )


def load(path2abc: Path, verbosity: bool) -> Tuple[pd.DataFrame, int]:
    abc: pd.DataFrame = pd.read_csv(path2abc, header=0, low_memory=False)
    abc.drop(abc[abc.idx == "idx"].index, inplace=True)
    abc.dropna(how="all", inplace=True)
    if verbosity:
        print(abc.head())
    abc.rename(columns={abc.columns[0]: "parental_idx"}, inplace=True)
    try:  # can be nan when no subsampling
        abc["parental_idx"] = abc["parental_idx"].astype("uint")
    except ValueError:
        assert abc["parental_idx"].isna().any()
        abc["parental_idx"] = abc["parental_idx"].astype("float")

    abc["idx"] = abc["idx"].astype("uint")
    abc["seed"] = abc["seed"].astype("uint")
    abc["cells"] = abc["cells"].astype("uint")
    abc["tumour_cells"] = abc["tumour_cells"].astype("uint")
    abc["init_cells"] = abc["init_cells"].astype("uint")

    # init_copies can be NaN when the run started with custom ecDNA distribution
    try:
        abc["init_copies"] = abc["init_copies"].astype("uint")
    except ValueError:
        abc["init_copies"] = abc["init_copies"].astype("float")

    abc["ecdna"] = abc["ecdna"].astype("float")
    abc["mean"] = abc["mean"].astype("float")
    abc["frequency"] = abc["frequency"].astype("float")
    abc["entropy"] = abc["entropy"].astype("float")
    abc["f1"] = abc["f1"].astype("float")
    abc["f2"] = abc["f2"].astype("float")
    abc["d1"] = abc["d1"].astype("float")
    abc["d2"] = abc["d2"].astype("float")
    abc["init_mean"] = abc["init_mean"].astype("float")

    if verbosity:
        print(abc.head())
        print(abc.tail())
        print(abc.dtypes)
        print(abc.shape)
        print(abc.f1.describe())

    return (abc, infer_nb_timepoints(abc, verbosity))


def run(app: App):
    """Plot posterior distribution of the fitness coefficient"""
    for theta in app.theta:
        print("Generating posterior for", theta)
        if app.stats:
            path2save = commons.create_path2save(
                app.path2save,
                Path(
                    "{}mean_{}frequency_{}entropy_{}ecdna_{}_subplots.pdf".format(
                        app.thresholds["mean"],
                        app.thresholds["frequency"],
                        app.thresholds["entropy"],
                        app.thresholds["ecdna"],
                        theta,
                    )
                ),
            )
            (abc, nb_timepoints) = load(app.abc, app.verbosity)
            plot_posterior_per_stats(
                app.thresholds, abc, path2save, nb_timepoints, theta, app.verbosity
            )
        else:
            path2save = commons.create_path2save(
                app.path2save,
                Path(
                    "{}relative_{}ecdna_{}.pdf".format(
                        app.thresholds["mean"], app.thresholds["ecdna"], theta
                    )
                ),
            )
            (abc, nb_timepoints) = load(app.abc, app.verbosity)
            plot_post(
                app.thresholds, abc, path2save, nb_timepoints, theta, app.verbosity
            )

    # plot distances
    path2save = commons.create_path2save(
        app.path2save,
        Path("hisograms.pdf"),
    )
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    n_bins = 100
    axs[0, 0].hist(abc.ecdna, bins=n_bins)
    axs[0, 0].set_title("KS distance distribution")
    axs[0, 1].hist(abc["mean"], bins=n_bins)  # ylabel=)
    axs[0, 1].set_title("Mean rel distance")
    axs[1, 0].hist(abc["frequency"], bins=n_bins)
    axs[1, 0].set_title("Frequency rel distance")
    axs[1, 1].hist(abc["entropy"], bins=n_bins)
    axs[1, 1].set_title("Entropy rel distance")
    plt.savefig(path2save)


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
        nargs="+",
        choices=["f1", "d1", "d2", "copies"],
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

    # list of quantities for which we want to approximate the posterior distribution
    theta_list = [
        theta if theta != "copies" else "init_copies" for theta in args["theta"]
    ]

    return App(
        abc,
        Thresholds(thresholds),
        theta_list,
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
    plot_rates(copies, (0, copies.max().iloc[0] + 1), 120, ax, title)


def plot_rates(rates, range_hist: Tuple[float, float], bins: int, ax, title=None):
    """Modify inplace the ax by plotting the hist of the posterior distribution
    `rates`. Here `title` is the title of the axis `ax`."""
    ax.hist(rates, bins=bins, range=range_hist, align="mid")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=20)


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
            plot_posterior(
                the_thresholds,
                abc,
                ax,
                nb_timepoints,
                theta,
                verbosity,
                stats_mode=True,
            )
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


def plot_posterior(
    thresholds, abc, ax, nb_timepoints, theta, verbosity, stats_mode=False
):
    assert theta in set(abc.columns), "Cannot find column '{}' in data".format(theta)
    my_query = ""
    stats = []
    for threshold in thresholds:
        my_query += "(`{}` < {}) & ".format(*threshold)
        stats.append(threshold[0])
    my_query = my_query.rstrip(" & ")
    print(my_query)
    to_plot = abc.loc[abc[stats].query(my_query).index, :]
    if nb_timepoints > 1:
        if verbosity:
            print("Running longitudinal ABC")
        abc_longitudinal(to_plot, my_query, nb_timepoints, verbosity)
        # to_plot.drop(index=to_plot[to_plot["cells"] == 10000].index, inplace=True)
        # assert to_plot.cells.unique() == 1000000, to_plot.cells.unique()
        # print(to_plot)
    to_plot.drop(columns=set(to_plot.columns) - set([theta]), inplace=True)
    if verbosity:
        print(to_plot)
    if theta == "f1":
        plot_fitness(to_plot, ax, title=list(stats) if stats_mode else None)
        if not stats_mode:
            ax.set_xlabel(
                r"Posterior distribution for $\rho_1^*$", fontsize=24, usetex=True
            )
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
