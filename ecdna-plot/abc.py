#!/usr/bn/env python
# coding: utf-8
"""Plot the bayesian inferences."""
from types import UnionType
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import pandas as pd
import math
import seaborn as sns
from matplotlib.ticker import MaxNLocator, MultipleLocator
from collections import UserDict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, unique
from collections import UserDict
from typing import NewType, Tuple, List, Union
from itertools import combinations
from . import commons

PLOTTING_STATS = {"ecdna": "D", "mean": "M", "frequency": "F", "entropy": "E"}
THETA_MAPPING = {
    "f1": "$\\rho_1^*$",
    "init_copies": "$\\gamma^*$",
    "d1": "$\\delta_1^*$",
    "d2": "$\\delta_2^*$",
}
MYSHAPE = (len(PLOTTING_STATS), len(PLOTTING_STATS))
Thresholds = NewType("Thresholds", UserDict[str, float])
plt.style.use("./ecdna-plot/paper.mplstyle")


@unique
class Plot(Enum):
    # one graph with the posterior distribution
    SIMPLE = 0
    STATS = 1
    TIMEPOINTS = 2


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
    plot: Plot
    png: bool
    verbosity: bool


def abc_is_subsampled(data: pd.DataFrame) -> bool:
    """When the runs have been subsampled before running ABC"""
    return not data["parental_idx"].isna().any()


def query_from_thresholds(thresholds) -> Tuple[str, List[str]]:
    """Create a query and stats from the thresholds"""
    my_query = ""
    stats = []
    for threshold in thresholds:
        my_query += "(`{}` < {}) & ".format(*threshold)
        stats.append(threshold[0])
    my_query = my_query.rstrip(" & ")
    print(my_query)
    return my_query, stats


def infer_timepoints(data: pd.DataFrame, verbosity: bool) -> int:
    # tumour cells of the timepoint found for each run idx2analyze
    cells = data[["timepoint", "tumour_cells"]].drop_duplicates()

    assert (
        cells.groupby("timepoint").count().tumour_cells.unique().shape[0] == 1
    ), "Found runs with different nb of timepoints"

    if verbosity:
        print(
            "Found {} timepoints with cells {}".format(
                cells.shape[0], cells.tumour_cells
            )
        )
    return cells.timepoint.shape[0]


def abc_longitudinal(
    df: pd.DataFrame, my_query: str, nb_timepoints: int, verbosity: bool
) -> pd.DataFrame:
    if verbosity:
        print("Running longitudinal ABC")

    # Create a new column for each row, indicating whether the condition is met.
    # Groupby and reduce the timepoint dimensionality: selection is true when the
    # all the timepoints match the query.
    grouped = (
        df.eval("selection = {}".format(my_query))
        .loc[:, ["selection", "parental_idx", "idx"]]
        .groupby(["parental_idx", "idx"])
        .sum()
        == nb_timepoints
    )

    if verbosity:
        print("selected runs: ", grouped)

    merged = pd.merge(
        left=df,
        right=grouped,
        how="left",
        left_on=["parental_idx", "idx"],
        right_index=True,
    )

    return merged.loc[merged.selection, :].drop_duplicates("parental_idx")


def load(path2abc: Path, verbosity: bool) -> Tuple[pd.DataFrame, int]:
    if verbosity > 0:
        print("Loading data")
    abc: pd.DataFrame = pd.read_csv(path2abc, header=0, low_memory=False)
    abc.drop(abc[abc.idx == "idx"].index, inplace=True)
    abc.dropna(how="all", inplace=True)
    if verbosity:
        print(abc.head())
    abc.rename(columns={str(abc.columns[0]): "parental_idx"}, inplace=True)
    try:  # can be nan when no subsampling
        abc["parental_idx"] = abc["parental_idx"].astype("uint")
    except ValueError:
        assert abc["parental_idx"].isna().any()
        abc["parental_idx"] = abc["parental_idx"].astype("float")

    abc["idx"] = abc["idx"].astype("uint")
    abc["timepoint"] = abc["timepoint"].astype("uint")
    abc["seed"] = abc["seed"].astype("uint")
    abc["cells"] = abc["cells"].astype("uint")
    abc["tumour_cells"] = abc["tumour_cells"].astype("uint")
    abc["init_cells"] = abc["init_cells"].astype("uint")

    # init_copies can be NaN when the run started with custom ecDNA distribution
    try:
        abc["init_copies"] = abc["init_copies"].astype("uint")
    except ValueError:
        abc["init_copies"] = abc["init_copies"].astype("float")
        # when restart (longitudinal analysis) the init_copies are the same of
        # the previous subsampled run
        abc["init_copies"].fillna(method="ffill", inplace=True)
        abc["init_copies"] = abc["init_copies"].astype("uint")

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

    return (abc, infer_timepoints(abc, verbosity))


def run(app: App):
    """Plot posterior distribution of the fitness coefficient"""
    (abc, timepoints) = load(app.abc, app.verbosity)

    print("Checking priors first")

    try:
        path2save = commons.create_path2save(app.path2save, Path("priors.pdf"))
    except FileExistsError:
        pass
    else:
        with mpl.rc_context(
            {
                "xtick.top": False,
                "xtick.bottom": False,
                "ytick.left": False,
                "ytick.right": False,
                "xtick.minor.visible": False,
                "ytick.minor.visible": False,
            }
        ):
            g = sns.pairplot(
                abc[app.theta].rename(columns=THETA_MAPPING), diag_kws={"bins": 100}
            )
            savefig(path2save, g, app.png, app.verbosity)

    if app.verbosity:
        print("distributions\n", abc[app.theta].describe())

    correlation = abc[app.theta].corr()
    if app.verbosity:
        print("correlation: ", correlation)
    # process to find if any high values are found
    # 1. remove diag elements which are 1 by def
    np.fill_diagonal(correlation.to_numpy(), 0.0)
    # 2. find if any high values
    highly_correlated = correlation[correlation >= 0.1].any().any()
    if highly_correlated:
        print("\t--WARNING: high correlation between the priors ", app.theta)

    try:
        path2save = commons.create_path2save(app.path2save, Path("posteriors.pdf"))
    except FileExistsError:
        pass
    else:
        my_query, stats = query_from_thresholds(app.thresholds.items())
        assert my_query
        to_plot = abc.loc[abc[stats].query(my_query).index, :]
        to_plot.rename(columns=THETA_MAPPING, inplace=True)
        with mpl.rc_context(
            {
                "axes.grid": True,
                "xtick.top": False,
                "xtick.bottom": False,
                "ytick.left": False,
                "ytick.right": False,
                "xtick.minor.visible": False,
                "ytick.minor.visible": False,
            }
        ):
            try:
                g = sns.pairplot(
                    to_plot[[THETA_MAPPING[theta] for theta in app.theta]],
                    kind="kde",
                )
            except np.linalg.LinAlgError:
                pass
            g = sns.pairplot(
                to_plot[[THETA_MAPPING[theta] for theta in app.theta]],
                diag_kws={"bins": 100},
            )
            # g.axes[-1, 0].xaxis.set_major_locator(MultipleLocator(2))
            g.axes[0, 0].yaxis.set_major_locator(MultipleLocator(2))
            # g.fig.suptitle("Inferrences", y=1.08)  # y= some height>1

            savefig(path2save, g, app.png, app.verbosity)

    for theta in app.theta:
        print("Generating posterior for", theta)
        if app.plot is Plot.STATS:
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
            plot_posterior_per_stats(
                app.thresholds,
                abc,
                path2save,
                app.png,
                timepoints,
                theta,
                app.verbosity,
            )
        elif app.plot in {Plot.SIMPLE, Plot.TIMEPOINTS}:
            try:
                path2save = commons.create_path2save(
                    app.path2save,
                    Path(
                        "{}relative_{}ecdna_{}.pdf".format(
                            app.thresholds["mean"], app.thresholds["ecdna"], theta
                        )
                    ),
                )
            except FileExistsError as e:
                if not app.plot is Plot.TIMEPOINTS:
                    raise e
            else:
                plot_post(
                    app.thresholds,
                    abc,
                    path2save,
                    app.png,
                    timepoints,
                    theta,
                    app.verbosity,
                )

            if app.plot is Plot.TIMEPOINTS:
                assert timepoints > 1, "Found app timepoints but only one timepoint"
                path2save = commons.create_path2save(
                    app.path2save,
                    Path(
                        "{}relative_{}ecdna_{}_timepoints.pdf".format(
                            app.thresholds["mean"],
                            app.thresholds["ecdna"],
                            theta,
                        )
                    ),
                )
                plot_posterior_per_timepoints(
                    app.thresholds,
                    abc,
                    path2save,
                    timepoints,
                    theta,
                    app.png,
                    app.verbosity,
                )
        else:
            raise ValueError(
                "Wrong value of app.plot {}, must be one of {}".format(app.plot, Plot)
            )

    # plot distances
    try:
        path2save = commons.create_path2save(
            app.path2save,
            Path("distances.pdf"),
        )
    except FileExistsError as e:
        if app.verbosity > 0:
            print("Skipping generation of distances because file already present")
    else:
        fig, axs = plt.subplots(2, 2, tight_layout=True)
        n_bins = 100
        axs[0, 0].hist(abc.ecdna, bins=n_bins)
        axs[0, 0].set_xlabel("KS distance")
        axs[0, 1].hist(abc["mean"], bins=n_bins)  # ylabel=)
        axs[0, 1].set_xlabel("Mean distance")
        axs[1, 0].hist(abc["frequency"], bins=n_bins)
        axs[1, 0].set_xlabel("Frequency distance")
        axs[1, 1].hist(abc["entropy"], bins=n_bins)
        axs[1, 1].set_xlabel("Entropy distance")
        savefig(path2save, fig, app.png, app.verbosity)


def build_app() -> App:
    """Parse and returns the app"""
    parser = argparse.ArgumentParser(
        description="Plot posterior distribution (histograms) for the parameters theta inferred by ABC."
    )

    plot = parser.add_mutually_exclusive_group()

    plot.add_argument(
        "--stats",
        dest="stats",
        required=False,
        action="store_true",
        default=False,
        help="Plot posterior for combinations of statistics",
    )

    plot.add_argument(
        "--timepoints",
        dest="timepoints",
        required=False,
        action="store_true",
        default=False,
        help="Plot posterior for combinations of timepoints",
    )

    parser.add_argument(
        "--theta",
        dest="theta",
        nargs="+",
        choices=["f1", "d1", "d2", "copies"],
        help="Parameter to infer",
    )

    parser.add_argument(
        "--png",
        dest="png",
        required=False,
        default=False,
        action="store_true",
        help="Use flag to generate png copies of the pdf plots",
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
    assert abc.parent.is_dir()

    if args["stats"]:
        plot = Plot.STATS
    elif args["timepoints"]:
        plot = Plot.TIMEPOINTS
    else:
        plot = Plot.SIMPLE

    # list of quantities used to approximate the posterior distribution
    theta_list = [
        theta if theta != "copies" else "init_copies" for theta in args["theta"]
    ]

    return App(
        abc,
        Thresholds(thresholds),
        theta_list,
        abc.parent,
        plot,
        args["png"],
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


def savefig(path2save: Path, fig, png: bool, verbosity: bool):
    if verbosity:
        print("Saving figure to", path2save)
    fig.savefig(fname=path2save, bbox_inches="tight")
    if png:
        fig.savefig(fname=path2save.with_suffix(".png"), bbox_inches="tight", dpi=600)
        if verbosity:
            print("Saved also a png copy")
    try:
        plt.close(fig)
    except TypeError:
        assert isinstance(fig, sns.axisgrid.PairGrid)
        pass


def plot_fitness(fitness, ax, title=None, xlabel=None):
    plot_rates(fitness, (0.9, 3.1), 120, ax, title, xlabel)


def plot_death(death, ax, title=None, xlabel=None):
    plot_rates(death, (0, 1.1), 120, ax, title, xlabel)


def plot_copies(copies, ax, title=None, xlabel=None):
    if copies.empty:
        max_copies = 1
    else:
        max_copies = copies.max().iloc[0] + 1
    plot_rates(copies, (0, max_copies), 120, ax, title, xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_rates(rates, range_hist: Tuple[float, float], bins: int, ax, title, xlabel):
    """Modify inplace the ax by plotting the hist of the posterior distribution
    `rates`. Here `title` is the title of the axis `ax`."""
    rates.plot(
        kind="hist",
        ax=ax,
        bins=bins,
        range=range_hist,
        align="mid",
        legend=False,
    )
    ax.set_title(title)
    # ax.tick_params(axis="both", labelsize=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_posterior_per_timepoints(
    thresholds: Thresholds,
    abc,
    path2save,
    timepoints: int,
    theta,
    png: bool,
    verbosity,
):
    # all ok if two or more rows (i.e. timepoints > 3) else squeeze to get shape
    # compatible with axes
    shape = np.empty((math.ceil(timepoints / 3), 3)).squeeze().shape
    fig_tot, axes = plt.subplots(*shape, sharex=False)
    my_query, stats = query_from_thresholds(thresholds.items())
    axis2empty = False

    for i, ax in enumerate(axes.ravel()):
        if verbosity > 0:
            print("Generating the posterior distribution for timepoint", i)
        if i <= timepoints - 1:
            ax = axes[np.unravel_index(i, shape)]
            plot_posterior(
                abc=abc[abc["timepoint"] == i],
                ax=ax,
                timepoints=1,
                theta=theta,
                my_query=my_query,
                stats=stats,
                xlabel=None,
                title="Timepoint {}".format(i),
                verbosity=verbosity,
            )
            ax.set_title(ax.get_title(), {"fontsize": 10})
            ax.tick_params(axis="both", size=2, which="major", labelsize=4)
            ax.tick_params(
                axis="both",
                size=2,
                which="major",
                labelsize=5,
            )
        else:
            axis2empty = True
            ax.axis("off")
            fig_tot.axes.append(ax)

    if not axis2empty:
        fig_tot.subplots_adjust(hspace=0.5)
    savefig(path2save, fig_tot, png, verbosity)


def plot_posterior_per_stats(
    thresholds: Thresholds,
    abc,
    path2save,
    png: bool,
    timepoints: int,
    theta,
    verbosity,
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
            my_query, stats = query_from_thresholds(the_thresholds)
            plot_posterior(
                abc=abc,
                ax=ax,
                timepoints=timepoints,
                theta=theta,
                my_query=my_query,
                stats=stats,
                title=None,
                xlabel=None,
                verbosity=verbosity,
            )
            clean = ",".join([PLOTTING_STATS[stat] for stat in stats])
            ax.set_title(clean, {"fontsize": 10})
            ax.tick_params(axis="both", size=2, which="major", labelsize=4)
            ax.tick_params(axis="both", size=0, which="minor", labelsize=4)
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
    savefig(path2save, fig_tot, png, verbosity)


def plot_posterior(
    abc,
    ax,
    timepoints: int,
    theta,
    my_query: str,
    stats: List[str],
    xlabel: None | str,
    title: None | str,
    verbosity,
):
    assert theta in set(abc.columns), "Cannot find column '{}' in data".format(theta)
    if timepoints > 1:
        to_plot = abc_longitudinal(abc, my_query, timepoints, verbosity)
    else:
        to_plot = abc.loc[abc[stats].query(my_query).index, :]
    if to_plot.empty:
        print(
            "\t--WARNING: no run is similar to the ground truth; try to change the thresholds values"
        )
    to_plot.drop(columns=set(to_plot.columns) - set([theta]), inplace=True)
    if verbosity:
        print(to_plot)
    if theta == "f1":
        plot_fitness(to_plot, ax, title=title, xlabel=xlabel)
    elif theta in {"d1", "d2"}:
        plot_death(to_plot, ax, list(stats), xlabel)
    elif theta == "init_copies":
        plot_copies(to_plot, ax, title=title, xlabel=xlabel)
    else:
        raise ValueError(
            "{} not valid: theta must be either `f1`, `d1`, `d2` or `init_copies`".format(
                theta
            )
        )


def plot_post(
    thresholds: Thresholds,
    abc,
    path2save,
    png: bool,
    timepoints: List[int],
    theta,
    verbosity,
):
    fig, ax = plt.subplots(1, 1)
    xlabel = r"Inferred {}".format(THETA_MAPPING[theta])
    title = None
    my_query, stats = query_from_thresholds(thresholds.items())

    plot_posterior(
        abc,
        ax,
        timepoints,
        theta,
        my_query=my_query,
        stats=stats,
        xlabel=xlabel,
        title=title,
        verbosity=verbosity,
    )
    fig.axes.append(ax)
    fig.subplots_adjust(hspace=0.5)
    savefig(path2save, fig, png, verbosity)


if __name__ == "__main__":
    run(build_app())
