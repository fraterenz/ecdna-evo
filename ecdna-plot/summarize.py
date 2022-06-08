#!/usr/bn/env python
# coding: utf-8
#
# Plot the posterior distributions for experiments with different ground truth
# on the same graph using seaborn.boxenplot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from . import commons
from .abc import Thresholds, load, savefig, filter_abc_runs

plt.style.use("./ecdna-plot/paper.mplstyle")


@dataclass
class SummarizeApp:
    """Plot the posterior distributions for gamma and rho1 for experiemnts with
    different ground truth parameters (synthetic data).
    """

    paths2abc: List[Path]  # one path for each experiment
    thresholds: List[Thresholds]
    gr_rhos1: List[float]
    gr_gammas: List[int]
    path2save: Path
    png: bool
    verbosity: bool


def run(app: SummarizeApp):
    dfs = []
    for path2data, rho1, gamma, threshold in zip(
        app.paths2abc,
        app.gr_rhos1,
        app.gr_gammas,
        app.thresholds,
    ):
        df, timepoints = load(path2data, app.verbosity)
        to_plot, _ = filter_abc_runs(
            df, timepoints, threshold, {"f1", "init_copies"}, app.verbosity
        )
        if app.verbosity:
            print("Found {} timepoints".format(timepoints))
        to_plot["rho_1"] = rho1
        to_plot["gamma"] = gamma
        dfs.append(to_plot)
    toplot = pd.concat(dfs)

    for theta in {"f1", "init_copies"}:
        fig, ax = plt.subplots(1, 1, sharex=False)
        if theta == "f1":
            x2plot, hue2plot, hue_order, my_title = (
                "rho_1",
                "gamma",
                sorted([float(ele) for ele in (set(app.gr_rhos1))]),
                r"""Copy number $\gamma^*$""",
            )
        else:
            x2plot, hue2plot, hue_order, my_title = (
                "gamma",
                "rho_1",
                sorted([int(ele) for ele in (set(app.gr_gammas))]),
                r"""Proliferative advantage $\rho_1^*$""",
            )

        print(hue_order)

        sns.boxenplot(
            ax=ax,
            x=x2plot,
            y=theta,
            hue=hue2plot,
            k_depth="full",
            data=toplot,
            palette="Set2",
        )
        ax.legend(title=my_title)
        ax.set_xticks([], minor=True)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        if theta == "f1":
            ax.set_xlabel(r"Real proliferative advantage $\rho_1^*$")
            ax.set_ylabel(r"Inferred proliferative advantage $\rho_1$")
        else:
            ax.set_xlabel(r"Real initial copy number $\gamma^*$")
            ax.set_ylabel(r"Inferred initial copy number $\gamma$")

        path2save = commons.create_path2save(
            app.path2save, Path("{}.pdf".format(theta))
        )

        savefig(
            path2save,
            fig,
            app.png,
            app.verbosity,
        )


def build_app() -> SummarizeApp:
    """Parse and returns the app"""
    parser = argparse.ArgumentParser(
        description="Plot posterior distributions for syntehtic data with different rho1 and gamma in two plots using seaborn.boxnplot."
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
        "--ecdna",
        dest="ecdna",
        metavar="FLOAT",
        required=True,
        nargs="+",
        help="Floats specifying the distance used to accept runs based on the ks distance of the ecDNA distribution for all experiments",
    )

    parser.add_argument(
        "--mean",
        dest="mean",
        metavar="INT",
        required=True,
        nargs="+",
        help="Integer specifying the relative difference percentage in the mean threshold for which the posterior will be plottedfor all experiments.",
    )

    parser.add_argument(
        "--frequency",
        dest="frequency",
        metavar="INT",
        required=True,
        nargs="+",
        help="Integer specifying the relative difference percentage in the frequency threshold for which the posterior will be plottedfor all experiments.",
    )

    parser.add_argument(
        "--entropy",
        dest="entropy",
        metavar="INT",
        required=True,
        nargs="+",
        help="Integer specifying the relative difference percentage in the entropy threshold for which the posterior will be plottedfor all experiments.",
    )

    parser.add_argument(
        "--abc",
        metavar="FILE",
        dest="abc",
        required=True,
        nargs="+",
        help="Paths to tarball file abc.tar.gz, where the output of the ABC inference can be found (csv files for each run)",
    )

    parser.add_argument(
        "--rho1",
        dest="rho1",
        metavar="FLOAT",
        required=True,
        nargs="+",
        help="Values for the ground truth rho1 for each abc.tar.gz passed",
    )

    parser.add_argument(
        "--gamma",
        dest="gamma",
        metavar="INT",
        required=True,
        nargs="+",
        help="Values for the ground truth gamma for each abc.tar.gz passed",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        default=False,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--path2save",
        metavar="DIR",
        dest="path2save",
        required=True,
        type=str,
        help="Path to dir where to save the figures",
    )

    args = vars(parser.parse_args())
    abc, rho1, gamma = args["abc"], args["rho1"], args["gamma"]
    abc = [Path(path) for path in abc]
    assert (
        len(abc) == len(rho1) == len(gamma)
    ), "ERROR: arguments --abc --gamma --rho1 and --ecdna must have the same number of values"

    thresholds = []
    i = 0
    for ecdna, mean, frequency, entropy in zip(
        args["ecdna"], args["mean"], args["frequency"], args["entropy"], strict=True
    ):
        thresholds.append(
            Thresholds(
                UserDict(
                    {
                        "ecdna": float(ecdna),
                        "mean": float(mean) / 100,
                        "frequency": float(frequency) / 100,
                        "entropy": float(entropy) / 100,
                    }
                )
            )
        )
        i += 1

    assert i == len(
        abc
    ), "Error: the number of thresholds do not match the number of input file abc.tar.gz. Found {} vs {}".format(
        i, len(abc)
    )

    return SummarizeApp(
        abc,
        thresholds,
        rho1,
        gamma,
        Path(args["path2save"]),
        args["png"],
        args["verbosity"],
    )


if __name__ == "__main__":
    run(build_app())
