#!/usr/bn/env python
# coding: utf-8
"""Plot the bayesian inferences."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import UserDict
from pathlib import Path
from typing import NewType, Tuple, List, Union
from . import commons

PLOTTING_STATS = {"ecdna": "D", "mean": "M", "frequency": "F", "entropy": "E"}
THETA_MAPPING = {
    "f1": "$\\rho_1^*$",
    "init_copies": "$\\gamma^*$",
    "d1": "$\\delta_1^*$",
    "d2": "$\\delta_2^*$",
}
Thresholds = NewType("Thresholds", UserDict[str, float])
plt.style.use("./ecdna-plot/paper.mplstyle")


def filter_abc_runs(
    abc: pd.DataFrame, timepoints: int, thresholds, theta, verbosity
) -> Tuple[pd.DataFrame, List[str]]:
    my_query, stats = query_from_thresholds(thresholds.items())
    # assert theta in set(abc.columns), "Cannot find column '{}' in data".format(theta)
    if timepoints > 1:
        to_plot = abc_longitudinal(abc, my_query, timepoints, verbosity)
    else:
        to_plot = abc.loc[abc[stats].query(my_query).index, :]
    if to_plot.empty:
        print(
            "\t--WARNING: no run is similar to the ground truth; try to change the thresholds values"
        )
    if verbosity:
        print(to_plot)
    return to_plot, stats


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
