"""Plot the ecDNA dynamics"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json
from typing import List
from pathlib import Path
from dataclasses import dataclass
from . import abc, commons


@dataclass
class App:
    """
    nplus: Optional path to nplus dynamics nplus.tar.gz
    nminus: Option path to nminus dynamics nminus.tar.gz
    mean: Option path to mean dynamics mean_dynamics.tar.gz
    variance: Option path to var dynamics var_dynamics.tar.gz
    """

    nplus: List[Path]
    nminus: List[Path]
    mean: List[Path]
    variance: List[Path]
    time: List[Path]
    ecdna: List[Path]
    path2dir: Path
    verbosity: bool

    def run(self):
        found_nplus, found_nminus = False, False
        if self.nplus:
            nplus = []
            path2save = commons.create_path2save(self.path2dir, Path("nplus.pdf"))
            for plus in self.nplus:
                df = commons.load_unformatted_csv(plus)
                nplus.append(df)
                if self.verbosity > 0:
                    print(df.head())
            self.plot(
                nplus,
                path2save,
                xlabel="Iterations",
                ylabel="Cells w/ ecDNAs",
                legend=False,
                loglog=True,
            )
            found_nplus = True
        if self.nminus:
            path2save = commons.create_path2save(self.path2dir, Path("nminus.pdf"))
            nminus = []
            for minus in self.nminus:
                df = commons.load_unformatted_csv(minus)
                nminus.append(df)
                if self.verbosity > 0:
                    print(df.head())
            self.plot(
                nminus,
                path2save,
                xlabel="Iterations",
                ylabel="Cells w/o any ecDNAs",
                legend=False,
                loglog=True,
                fontsize=18,
            )
            found_nminus = True
        if found_nplus and found_nminus:
            path2save = commons.create_path2save(
                self.path2dir, Path("population_iterations.pdf")
            )
            population = []
            for plus, minus in zip(nplus, nminus):
                population.append(plus.add(minus, axis="columns"))
            self.plot(
                population,
                path2save,
                xlabel="Iterations",
                ylabel="Total number of cells",
                legend=False,
                loglog=True,
                fontsize=18,
            )
        if self.time:
            if not (found_nplus and found_nminus):
                raise ValueError(
                    "Cannot plot the gillepsie time without nplus or nminus"
                )
            else:
                loaded = []
                for time in self.time:
                    df = commons.load_unformatted_csv(time)
                    loaded.append(df)
                    if self.verbosity > 0:
                        print(df.head())

                path2save = commons.create_path2save(self.path2dir, Path("time.pdf"))
                self.plot(loaded, path2save, legend=False)

                if found_nplus:
                    path2save = commons.create_path2save(
                        self.path2dir, Path("nplus_growth.pdf")
                    )
                    self.plot_xy(loaded, nplus, path2save)
                if found_nminus:
                    path2save = commons.create_path2save(
                        self.path2dir, Path("nminus_growth.pdf")
                    )
                    self.plot_xy(loaded, nminus, path2save)
                if found_nplus and found_nminus:
                    path2save = commons.create_path2save(
                        self.path2dir, Path("population.pdf")
                    )
                    self.plot_xy(loaded, population, path2save)

        if self.mean:
            loaded = []
            path2save = commons.create_path2save(self.path2dir, Path("mean.pdf"))
            for mean in self.mean:
                df = commons.load_unformatted_csv(mean)
                loaded.append(df)
                if self.verbosity > 0:
                    print(df.head())
            self.plot(
                loaded,
                path2save,
                xlabel="Iterations",
                ylabel="Average ecDNA copies per cell",
                legend=False,
            )
        if self.variance:
            loaded = []
            path2save = commons.create_path2save(self.path2dir, Path("variance.pdf"))
            for variance in self.variance:
                df = commons.load_unformatted_csv(variance)
                loaded.append(df)
                if self.verbosity > 0:
                    print(df.head())
            self.plot(
                loaded,
                path2save,
                xlabel="Iterations",
                ylabel="Average ecDNA copies per cell",
                legend=False,
            )
        if found_nplus and found_nminus:
            loaded = []
            path2save = commons.create_path2save(self.path2dir, Path("frequency.pdf"))
            # assume order matches
            for (plus, minus) in zip(nplus, nminus):
                # plus = commons.load_unformatted_csv(nplus)
                # minus = commons.load_unformatted_csv(nminus)
                frequency = plus.div(plus.add(minus, axis="columns"), axis="columns")
                loaded.append(frequency)

                if self.verbosity > 0:
                    print(frequency.head())
            self.plot(
                loaded,
                path2save,
                xlabel="Iterations",
                ylabel="Frequency of cells w/ ecDNA copies",
                legend=False,
                logx=True,
            )
        if self.ecdna:
            fig, ax = plt.subplots(1, 1)
            loaded = []
            path2save = commons.create_path2save(self.path2dir, Path("ecdna.pdf"))
            # load data
            for ecdna in self.ecdna:
                assert ecdna.is_file(), "Expected file"
                assert ecdna.suffix == ".json", "Expected json file"

                with open(ecdna) as file:
                    data = {
                        int(k): val
                        for k, val in json.load(file)["distribution"].items()
                    }
                loaded.append(data)

            # standarize ecDNA distributions: create the same bins for all distributions
            # find max copy number k_max present in all the data
            distributions = []
            k_max = max(
                [int(copy) for distribution in loaded for copy in distribution.keys()]
            )
            # cutoff for viz
            k_max = 50
            for ecdna, c in zip(loaded, commons.PALETTE.colors):
                for k in range(k_max):
                    ecdna.setdefault(k, 0)
                data = pd.Series(ecdna, dtype="uint").sort_index()
                if self.verbosity > 0:
                    print(data.head())
                # skip nminus cells
                data.drop(index=0, inplace=True)
                data /= data.sum()
                distributions.append(data)

                data.loc[:k_max].plot(
                    kind="bar",
                    color=c,
                    alpha=0.25,
                    ax=ax,
                )
            ax.set_xlabel("ecDNA copies per cell k", fontsize=18)
            ax.set_ylabel("Fraction of cells", fontsize=18)
            xticks = np.linspace(0, k_max, 11, dtype="uint")
            ax.tick_params(axis="y", labelsize=18, which="both", width=1)
            ax.set_xticks(
                ticks=xticks,
                labelsize=18,
                rotation=90,
                width=1,
            )
            ax.set_xlim([0, k_max])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation=0)
            ax.tick_params(axis="x", labelsize=18, which="both", width=1)
            print("Saving figure to", path2save)
            fig.savefig(fname=path2save, bbox_inches="tight")

            # compute scaling w/o nminus cells
            path2save = commons.create_path2save(self.path2dir, Path("scaling.pdf"))
            fig, ax = plt.subplots(1, 1)
            # take high copies, from Lange et al. 2021 preprint fig3e
            high_copy = 7
            for ecdna, c in zip(distributions, commons.PALETTE.colors):
                high_ecdnas = pd.DataFrame(
                    ecdna.loc[high_copy:].copy(deep=True), columns=["distribution"]
                )
                ntot = high_ecdnas.distribution.sum()
                high_ecdnas["1/k"] = 1.0 / high_ecdnas.index
                high_ecdnas["fraction"] = high_ecdnas.loc[:, "distribution"] / ntot
                if self.verbosity > 0:
                    print(high_ecdnas.head())
                high_ecdnas.plot(ax=ax, x="1/k", y="fraction", color=c, legend=False)
            ax.set_xlabel("1/(ecDNA copies per cell)", fontsize=18)
            ax.set_ylabel("Fraction of cells", fontsize=18)
            ax.tick_params(axis="y", labelsize=18, which="both", width=1)
            ax.tick_params(axis="x", labelsize=18, which="both", width=1)
            print("Saving figure to", path2save)
            fig.savefig(fname=path2save, bbox_inches="tight")

    def plot(
        self,
        data: List[pd.DataFrame],
        path2save: Path,
        **kwargs,
    ):
        # TODO legend with labels
        fig, ax = plt.subplots(1, 1)
        # df has shape (iterations x runs)
        for df, c in zip(data, commons.PALETTE.colors):
            if self.verbosity > 0:
                print(df)
            df.plot(ax=ax, alpha=0.2, color=c, **kwargs)
            df.mean(axis=1).plot(ax=ax, alpha=1, color=c, linestyle="--", **kwargs)
        ax.tick_params(axis="y", labelsize=18, which="both", width=1)
        ax.tick_params(axis="x", labelsize=18, which="both", width=1)
        ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=18)
        ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=18)
        print("Saving figure to", path2save)
        fig.savefig(fname=path2save, bbox_inches="tight")

    def plot_xy(
        self,
        data_x: List[pd.DataFrame],
        data_y: List[pd.DataFrame],
        path2save: Path,
        **kwargs,
    ):
        # TODO legend with labels
        fig, ax = plt.subplots(1, 1)
        for df_x, df_y, c in zip(data_x, data_y, commons.PALETTE.colors):
            ax.plot(df_x, df_y, alpha=0.2, color=c, **kwargs)
            # ax.plot(df_x, df_y.mean(axis=1), alpha=1, color=c, linestyle="--", **kwargs)
        ax.tick_params(axis="y", labelsize=18, which="both", width=1)
        ax.tick_params(axis="x", labelsize=18, which="both", width=1)
        ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=18)
        ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=18)
        print("Saving figure to", path2save)
        fig.savefig(fname=path2save, bbox_inches="tight")


def build_app() -> App:
    """Parse and returns the app"""
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="The ecDNAs dynamics simulated with the program `ecdna`."
    )

    parser.add_argument(
        "--nplus",
        metavar="FILE",
        dest="nplus",
        action="extend",
        nargs="*",
        required=False,
        type=str,
        help="Optional path list to nplus dynamics nplus.tar.gz",
    )

    parser.add_argument(
        "--nminus",
        metavar="FILE",
        dest="nminus",
        action="extend",
        nargs="*",
        required=False,
        type=str,
        help="Optional path list to nminus dynamics nminus.tar.gz",
    )

    parser.add_argument(
        "--mean",
        metavar="FILE",
        action="extend",
        nargs="*",
        dest="mean",
        required=False,
        type=str,
        help="Optional path list to mean dynamics mean_dynamics.tar.gz",
    )

    parser.add_argument(
        "--variance",
        metavar="FILE",
        action="extend",
        nargs="*",
        dest="variance",
        required=False,
        type=str,
        help="Optional path list to variance dynamics var_dynamics.tar.gz",
    )

    parser.add_argument(
        "--time",
        metavar="FILE",
        action="extend",
        nargs="*",
        dest="time",
        required=False,
        type=str,
        help="Optional path list to gillespie time dynamics time.tar.gz",
    )

    parser.add_argument(
        "--ecdna",
        metavar="FILE",
        action="extend",
        nargs="*",
        dest="ecdna",
        required=False,
        type=str,
        help="Optional path list to json ecDNA distribution",
    )

    parser.add_argument(
        "--save",
        metavar="FILE",
        dest="path2dir",
        required=True,
        type=str,
        help="Path to directory where to store the results",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        default=False,
        help="increase output verbosity",
    )

    args = vars(parser.parse_args())
    path2dir = Path(args["path2dir"])

    assert path2dir.is_dir(), "{} is not a path to a valid directory".format(path2dir)

    return App(
        [Path(path) for path in args["nplus"]] if args["nplus"] else [],
        [Path(path) for path in args["nminus"]] if args["nminus"] else [],
        [Path(path) for path in args["mean"]] if args["mean"] else [],
        [Path(path) for path in args["variance"]] if args["variance"] else [],
        [Path(path) for path in args["time"]] if args["time"] else [],
        [Path(path) for path in args["ecdna"]] if args["ecdna"] else [],
        path2dir,
        args["verbosity"],
    )


if __name__ == "__main__":
    build_app().run()
