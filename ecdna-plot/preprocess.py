import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from . import commons

plt.style.use("./ecdna-plot/paper.mplstyle")

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Plot the preprocessed ecDNA distributions"
    )

    parser.add_argument(
        "preprocessed",
        metavar="FILE",
        type=str,
        help="Input json distribution file to plot",
    )

    parser.add_argument(
        "save", metavar="DIR", type=str, help="Path2dir to save the plots"
    )

    parser.add_argument(
        "--png",
        dest="png",
        required=False,
        default=False,
        action="store_true",
        help="Use flag to save plots in the png format",
    )

    args = vars(parser.parse_args())
    distr = Path(args["preprocessed"])
    png = args["png"]

    with open(distr) as file:
        for sample in json.load(file)["samples"]:
            fig, ax = plt.subplots(1, 1)
            data = pd.Series(sample["ecdna"]["distribution"], dtype="uint")
            data.index = data.index.astype("uint")
            data.sort_index(inplace=True)
            ax.bar(x=data.index, height=data.tolist())
            ax.tick_params(axis="x", rotation=90)
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.set_xlim(left=0, right=ax.get_xlim()[-1])

            ax.axvline(x=sample["mean"], c="r")
            ax.set_title(
                "m={}, f={:.3}, e={:.3}, s={}".format(
                    sample["mean"],
                    sample["frequency"],
                    sample["entropy"],
                    sample["tumour_size"],
                ),
                fontsize=18,
            )
            ax.set_xlabel("ecDNA copies", fontsize=18)
            ax.set_ylabel("Cells", fontsize=18)

            path2save = commons.create_path2save(
                Path(args["save"]),
                Path("{}.{}".format(sample["name"], "png" if png else "pdf")),
            )
            print("Saving figure to", path2save)
            fig.savefig(fname=path2save, bbox_inches="tight", dpi=600)

            # compute scaling w/o nminus cells
            path2save = commons.create_path2save(
                Path(args["save"]),
                Path("{}_scaling.{}".format(sample["name"], "png" if png else "pdf")),
            )
            fig, ax = plt.subplots(1, 1)
            # take high copies, from Lange et al. 2021 preprint fig3e
            high_copy = 7
            high_ecdnas = pd.DataFrame(
                data.loc[high_copy:].copy(deep=True), columns=["distribution"]
            )
            ntot = high_ecdnas.distribution.sum()
            high_ecdnas["1/k"] = 1.0 / high_ecdnas.index
            high_ecdnas["fraction"] = high_ecdnas.loc[:, "distribution"] / ntot
            high_ecdnas.plot(ax=ax, x="1/k", y="fraction", legend=False)

            ax.set_xlabel("1/(ecDNA copies per cell)", fontsize=18)
            ax.set_ylabel("Fraction of cells", fontsize=18)
            print("Saving figure to", path2save)
            fig.savefig(fname=path2save, bbox_inches="tight", dpi=600)
