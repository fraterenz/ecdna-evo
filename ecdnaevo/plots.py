#!/usr/bin/env python
# coding: utf-8
from ecdnaevo import load
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from pathlib import Path


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
    plot_rates(fitness, (1, 3), 20, ax, title)


def plot_death1(death1, ax, title=None):
    plot_rates(death1, (0, 1), 20, ax, title)


def plot_death2(death2, ax, title=None):
    # path2save = load.path2save_from_rates(path2rates, "death2", title)
    plot_rates(death2, (0, 1), 20, ax, title)


def plot_rates(rates, range_hist: (float, float), bins: int, ax, title=None):
    """Modify inplace the ax by plotting the hist of the posterior distribution `rates`. Here `title` is the title of the axis `ax`."""
    ax.hist(rates, bins=bins, range=range_hist, align="mid")
    ax.set_title(title)


def plot_posterior_per_stats(values, all_rates, thresholds, path2save_dir: Path):
    """Plot the posterior distribution for the fitness coefficient
    for each combination of statistics.

    values: DataFrame with the runs idx as idx and the statistics as columns.
        The entries are the values of the distance between the patient's data
        and the run, for each run. Distance is the (for now) relative difference
        for mean, frequency, and entropy (abs(x - x_sim) / x), and for the
        ecdna is the ks distance.
    all_rates: DataFrame with runs idx as idx and the rates (f1, f2, d1, d2)
        as columns. The entries are the values of the rates for each run.
    thresholds: Dict with statistics as keys and values thresholds. The keys
        must be the same as the columns of values

    """
    assert not set(values.index) - set(
        all_rates.index
    ), "Found different idx in values and all_rates"
    assert not set(thresholds.keys()) - set(
        values.columns
    ), "Thresholds keys do not match values columns"
    assert path2save_dir.is_dir(), "`path2save` must be a directory"

    # assume all thresholds are the same except for ecdna distribution
    relative_threshold = thresholds["mean"]
    my_query = ""
    stats = []

    path2save_subplots = path2save_dir / "statistics_subplots"
    if not path2save_subplots.exists():
        path2save_subplots.mkdir()

    myshape = (4, 4)
    fig_tot, axes = plt.subplots(*myshape)
    path2save_tot = path2save_dir / str(
        "{}relative_fitness_raw_subplots".format(relative_threshold) + ".pdf"
    )
    i = 0
    plotting_stats = {"ecdna": "D", "mean": "M", "frequency": "F", "entropy": "E"}

    # combinations of statistics
    for r in range(1, 5):
        # iter over the combinations
        for the_thresholds in combinations(thresholds.items(), r):
            # iter over the (stats, threshold) for each combination
            for threshold in the_thresholds:
                my_query += "({} < {}) & ".format(*threshold)
                stats.append(threshold[0])
            my_query = my_query.rstrip(" & ")
            print(my_query)
            to_plot = all_rates.loc[values[stats].query(my_query).index].f1
            path2save = path2save_subplots / str(
                "{}relative_fitness_raw_".format(relative_threshold)
                + "_".join(list(stats))
                + ".pdf"
            )

            fig, ax = plt.subplots(1, 1)
            plot_fitness(to_plot, ax, list(stats))
            fig.axes.append(ax)
            fig.savefig(fname=path2save)

            ax = axes[np.unravel_index(i, myshape)]
            plot_fitness(to_plot, ax, list(stats))
            clean = ",".join(
                [
                    plotting_stats[stat]
                    for stat in "".join(
                        [e for e in ax.get_title() if e not in {"[", "]", "'", " "}]
                    ).split(",")
                ]
            )

            ax.set_title(clean, {"fontsize": 10})
            ax.tick_params(
                axis="both",
                size=2,
                which="major",
                labelsize=5,
            )
            fig_tot.axes.append(ax)
            i += 1

            my_query = ""
            stats = []
    # last axis
    ax = axes[np.unravel_index(i, myshape)]
    ax.tick_params(axis="both", size=2, which="major", labelsize=4)
    fig_tot.axes.append(ax)
    fig_tot.subplots_adjust(hspace=0.5)
    fig_tot.savefig(fname=path2save_tot, bbox_inches="tight")


if __name__ == "__main__":
    from ecdnaevo import app

    path2rates, path2all_rates, path2values, threshold = app.build_app()

    # load ABC results
    (rates, _) = load.extract_tarball(path2rates, float, False)
    try:
        (all_rates, _) = load.extract_tarball(path2all_rates, float, True)
    except ValueError:
        assert path2all_rates is None
    try:
        (values, _) = load.extract_tarball(path2values, float, True)
    except ValueError:
        assert path2all_rates is None

    # Plots
    if path2values is not None:
        if path2all_rates is not None:
            relative_thr = threshold / 100
            thresholds = {
                "ecdna": 0.02,
                "entropy": relative_thr,
                "mean": relative_thr,
                "frequency": relative_thr,
            }
            plot_posterior_per_stats(values, all_rates, thresholds, path2rates.parent)
        else:
            raise ValueError(
                "When path2values is set, path2all_rates should be set as well, found None"
            )
    else:
        path = Path(path2rates.parent / "fitness_raw.pdf")
        print("Plotting fitness_raw and saving in", path)
        fig, ax = plt.subplots(1, 1)
        plt.savefig(fname=path)
        plot_fitness(rates[0], ax, "fitness_raw")
        fig.axes.append(ax)
        fig.savefig(fname=path)

        path = path2rates.parent / "death1.pdf"
        print("Plotting death1 and save in", path)
        fig, ax = plt.subplots(1, 1)
        plot_death1(rates[2], ax, "death1")
        fig.savefig(fname=path)

        path = path2rates.parent / "death2.pdf"
        print("Plotting death2 and saving in", path)
        fig, ax = plt.subplots(1, 1)
        plot_death2(rates[3], ax, "death2")
        fig.savefig(fname=path)
