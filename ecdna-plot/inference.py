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
from typing import NewType, Tuple, List, Union
from itertools import combinations
from enum import Enum, unique
from .abc import (
    Thresholds,
    savefig,
    THETA_MAPPING,
    PLOTTING_STATS,
    load,
    filter_abc_runs,
    query_from_thresholds,
)
from . import commons

MYSHAPE = (len(PLOTTING_STATS), len(PLOTTING_STATS))


@unique
class Plot(Enum):
    # one graph with the posterior distribution
    SIMPLE = 0
    STATS = 1
    TIMEPOINTS = 2


@dataclass
class HistogramsApp:
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


def build_app() -> HistogramsApp:
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

    return HistogramsApp(
        abc,
        Thresholds(thresholds),
        theta_list,
        abc.parent,
        plot,
        args["png"],
        args["verbosity"],
    )


def run(app: HistogramsApp):
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

    for i, theta in enumerate(app.theta):
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
            my_query, stats = query_from_thresholds(app.thresholds.items())
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
                path2save = app.path2save / Path(
                    "{}mean_{}frequency_{}entropy_{}ecdna".format(
                        app.thresholds["mean"],
                        app.thresholds["frequency"],
                        app.thresholds["entropy"],
                        app.thresholds["ecdna"],
                        theta,
                    )
                )
                plot_posterior_per_timepoints(
                    app.thresholds,
                    abc,
                    path2save,
                    timepoints,
                    theta,
                    app.png,
                    i > 0,
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
    skip_distances: bool,
    verbosity,
):
    # all ok if two or more rows (i.e. timepoints > 3) else squeeze to get shape
    # compatible with axes
    shape = np.empty((math.ceil(timepoints / 3), 3)).squeeze().shape
    fig_posterior, axes_posterior = plt.subplots(*shape, sharex=False)
    fig_ks, axes_ks = plt.subplots(*shape, sharex=False, sharey=True)
    fig_mean, axes_mean = plt.subplots(*shape, sharex=False, sharey=True)
    fig_frequency, axes_frequency = plt.subplots(*shape, sharey=True)
    fig_entropy, axes_entropy = plt.subplots(*shape, sharey=True)
    axis2empty = False

    for i, (ax, ax1, ax2, ax3, ax4) in enumerate(
        zip(
            axes_posterior.ravel(),
            axes_ks.ravel(),
            axes_mean.ravel(),
            axes_frequency.ravel(),
            axes_entropy.ravel(),
        )
    ):
        if verbosity > 0:
            print("Generating the posterior distribution for timepoint", i)
        if i <= timepoints - 1:
            abc2plot = abc[abc["timepoint"] == i]

            ax = axes_posterior[np.unravel_index(i, shape)]
            to_plot, stats = filter_abc_runs(abc2plot, 1, thresholds, theta, verbosity)
            to_plot.drop(columns=set(to_plot.columns) - set([theta]), inplace=True)
            plot_posterior(
                to_plot=to_plot,
                ax=ax,
                theta=theta,
                title="Timepoint {}".format(i),
                xlabel=None,
                verbosity=verbosity,
            )

            ax.set_title(ax.get_title(), {"fontsize": 10})
            ax.set_xlabel("Inferred {}".format(THETA_MAPPING[theta]), {"fontsize": 10})
            ax.tick_params(axis="both", size=2, which="major", labelsize=4)
            ax.tick_params(
                axis="both",
                size=2,
                which="major",
                labelsize=5,
            )

            if not skip_distances:
                # now distances
                n_bins = 500
                ax1.hist(abc2plot.ecdna, bins=n_bins)
                ax1.set_title("Timepoint {}".format(i), {"fontsize": 10})
                ax1.set_xlabel("KS distance", {"fontsize": 10})
                ax1.tick_params(
                    axis="both",
                    size=2,
                    which="major",
                    labelsize=5,
                )

                ax2.hist(abc2plot["mean"], bins=n_bins)
                ax2.set_title("Timepoint {}".format(i), {"fontsize": 10})
                ax2.set_xlabel("Mean distance", {"fontsize": 10})
                ax2.tick_params(
                    axis="both",
                    size=2,
                    which="major",
                    labelsize=5,
                )

                ax3.hist(abc2plot["frequency"], bins=n_bins)
                ax3.set_title("Timepoint {}".format(i), {"fontsize": 10})
                ax3.set_xlabel("Frequency distance", {"fontsize": 10})
                ax3.tick_params(
                    axis="both",
                    size=2,
                    which="major",
                    labelsize=5,
                )
                ax3.set_xlim([0, 0.3])

                ax4.hist(abc2plot.entropy, bins=n_bins)
                ax4.set_title("Timepoint {}".format(i), {"fontsize": 10})
                ax4.set_xlabel("Entropy distance", {"fontsize": 10})
                ax4.tick_params(
                    axis="both",
                    size=2,
                    which="major",
                    labelsize=5,
                )

        else:
            axis2empty = True
            ax.axis("off")
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            ax4.axis("off")
            fig_posterior.axes.append(ax)
            fig_ks.axes.append(ax1)
            fig_mean.axes.append(ax2)
            fig_frequency.axes.append(ax3)
            fig_entropy.axes.append(ax4)

    fig_posterior.set_tight_layout(True)
    fig_ks.set_tight_layout(True)
    fig_mean.set_tight_layout(True)
    fig_frequency.set_tight_layout(True)
    fig_entropy.set_tight_layout(True)

    savefig(
        path2save.with_name(path2save.name + "_{}_timepoints.pdf".format(theta)),
        fig_posterior,
        png,
        verbosity,
    )
    if not skip_distances:
        savefig(
            path2save.with_name(path2save.name + "_ecdna.pdf"), fig_ks, png, verbosity
        )
        savefig(
            path2save.with_name(path2save.name + "_mean.pdf"), fig_mean, png, verbosity
        )
        savefig(
            path2save.with_name(path2save.name + "_frequency.pdf"),
            fig_frequency,
            png,
            verbosity,
        )
        savefig(
            path2save.with_name(path2save.name + "_entropy.pdf"),
            fig_entropy,
            png,
            verbosity,
        )


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
            to_plot, stats = filter_abc_runs(
                abc, timepoints, dict(the_thresholds), theta, verbosity
            )
            to_plot = to_plot.drop(columns=set(to_plot.columns) - set([theta]))
            plot_posterior(
                to_plot=to_plot,
                ax=ax,
                theta=theta,
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
    to_plot,
    ax,
    theta,
    xlabel: None | str,
    title: None | str,
    verbosity,
):
    if theta == "f1":
        plot_fitness(to_plot, ax, title=title, xlabel=xlabel)
    elif theta in {"d1", "d2"}:
        plot_death(to_plot, ax, title, xlabel)
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
    to_plot,
    path2save,
    png: bool,
    timepoints: int,
    theta,
    verbosity,
):
    fig, ax = plt.subplots(1, 1)
    xlabel = r"Inferred {}".format(THETA_MAPPING[theta])
    title = None
    to_plot, stats = filter_abc_runs(to_plot, 1, thresholds, theta, verbosity)
    to_plot = to_plot.drop(columns=set(to_plot.columns) - set([theta]))
    plot_posterior(
        to_plot,
        ax,
        theta,
        xlabel=xlabel,
        title=title,
        verbosity=verbosity,
    )
    fig.axes.append(ax)
    fig.subplots_adjust(hspace=0.5)
    savefig(path2save, fig, png, verbosity)


if __name__ == "__main__":
    run(build_app())
