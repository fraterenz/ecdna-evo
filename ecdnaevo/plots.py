#!/usr/bin/env python
# coding: utf-8
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import UserDict
from typing import NewType
from itertools import combinations

PLOTTING_STATS = {"ecdna": "D", "mean": "M", "frequency": "F", "entropy": "E"}
MYSHAPE = (len(PLOTTING_STATS), len(PLOTTING_STATS))
Thresholds = NewType("Thresholds", UserDict[str, float])


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


def plot_death1(death1, ax, title=None):
    plot_rates(death1, (0, 1.1), 40, ax, title)


def plot_death2(death2, ax, title=None):
    plot_rates(death2, (0, 1.1), 40, ax, title)


def plot_rates(rates, range_hist: Tuple[float, float], bins: int, ax, title=None):
    """Modify inplace the ax by plotting the hist of the posterior distribution
    `rates`. Here `title` is the title of the axis `ax`."""
    ax.hist(rates, bins=bins, range=range_hist, align="mid")
    ax.set_title(title)


def plot_posterior_per_stats(thresholds: Thresholds, abc, path2save, nb_timepoints):
    """Plot the posterior distribution for the fitness coefficient
    for each combination of statistics.

    abc: DataFrame
        The entries are the values of the distance between the patient's data
        and the run, for each run. Distance is the (for now) relative difference
        for mean, frequency, and entropy, and for the ecdna is the ks distance.
    """
    fig_tot, axes = plt.subplots(*MYSHAPE, sharex=True)
    axes[0, 0].set_xlim([0.9, 3.1])
    i = 0

    # combinations of statistics
    for r in range(1, 5):
        # iter over the combinations
        for the_thresholds in combinations(thresholds.items(), r):
            # iter over the (stats, threshold) for each combination
            ax = axes[np.unravel_index(i, MYSHAPE)]
            plot_posterior(the_thresholds, abc, ax, nb_timepoints)
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


def plot_posterior(thresholds, abc, ax, nb_timepoints):
    my_query = ""
    stats = []
    for threshold in thresholds:
        my_query += "(`{}` < {}) & ".format(*threshold)
        stats.append(threshold[0])
    my_query = my_query.rstrip(" & ")
    print(my_query)
    if nb_timepoints == 1:
        to_plot = abc.loc[abc[stats].query(my_query).index].f1
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
            .drop_duplicates(["idx"])
            .f1
        )
    plot_fitness(to_plot, ax, list(stats))


def plot_post(thresholds: Thresholds, abc, path2save, nb_timepoints):
    fig, ax = plt.subplots(1, 1)
    plot_posterior(thresholds.items(), abc, ax, nb_timepoints)
    # ax.tick_params(axis="both", size=2, which="major", labelsize=4)
    fig.axes.append(ax)
    fig.subplots_adjust(hspace=0.5)
    print("Saving figure to", path2save)
    fig.savefig(fname=path2save, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from ecdnaevo import app

    app.run(app.build_app())
