import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from . import commons

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

    args = vars(parser.parse_args())
    distr = Path(args["preprocessed"])

    with open(distr) as file:
        for sample in json.load(file)["samples"]:
            fig, ax = plt.subplots(1, 1)
            pd.Series(sample["ecdna"]["distribution"], dtype="uint").sort_index().plot(
                kind="bar",
                ax=ax,
            )
            ax.axvline(x=sample["mean"], c="r")
            ax.text(
                x=0,
                y=0,
                s="f={}, e={}, s={}".format(
                    sample["frequency"],
                    sample["entropy"],
                    sample["tumour_size"],
                ),
            )
            ax.set_xlabel("ecDNA copies", fontsize=18)
            ax.set_ylabel("Cells", fontsize=18)

            path2save = commons.create_path2save(
                Path(args["save"]), Path("{}.pdf".format(sample["name"]))
            )
            print("Saving figure to", path2save)
            fig.savefig(fname=path2save, bbox_inches="tight")
