import pandas as pd
import tarfile
from pathlib import Path
from matplotlib import colors


PALETTE = colors.ListedColormap(["#2b8cbe", "#636363", "red", "yellow"])


def create_path2save(path2dir: Path, filename: Path, exist_ok=False) -> Path:
    assert path2dir.is_dir()
    path = path2dir / filename
    path.touch(exist_ok=exist_ok)
    return path


def load_unformatted_csv(path: Path) -> pd.DataFrame:
    """Load unformatted csv files in a tar ball.
    Files can have different number of entries, e.g.
    [1, 2, 3] and [1, 2] in different files compress in `path`
    """

    assert path.is_file(), "Exptected file found {}".format(path)
    assert path.suffix == ".gz", "Exptected tarball found {}".format(path)

    data = dict()
    max_entries = 0

    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                col = member.name.split("/")[-1].rstrip(".csv")
                run = f.read().decode().rstrip("\n").split(",")
                data[col] = pd.Series(run, index=range(len(run)))
                if len(data[col]) > max_entries:
                    max_entries = len(data[col])
    # transpose for plotting API with pandas that considers columns as
    # different experiments (labels)
    df = pd.DataFrame(data, index=range(max_entries), dtype=float)
    df.index = df.index.astype(int)
    df.sort_index(inplace=True)
    return df


def prettify(
    ax,
    xlabel: str,
    ylabel: str,
    legend: bool,
    log_scale: str,
    xlim: list = [],
    ylim: list = [],
):
    # log scale is a proxy for fig2, no log scale is proxy for fig3
    if log_scale == "both":
        ax.set_yscale("log")
        ax.set_xscale("log")
    elif log_scale == "xscale":
        ax.set_xscale("log")
    elif log_scale == "yscale":
        ax.set_yscale("log")
    elif log_scale == "none":
        pass
    else:
        ValueError(
            "Invalid value for argument `log_scale`.  \
            Must be `both`, `xscale`, `yscale` or `none`, found `{}` instead".format(
                log_scale
            )
        )
    ax.tick_params(axis="y", labelsize=12, which="both", width=1)
    ax.tick_params(axis="x", labelsize=12, which="both", width=1)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if legend:
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        leg = ax.legend(
            bbox_to_anchor=(-0.2, 1.02, 1.3, 0.102),
            loc="lower left",
            prop={"size": 18},
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            handlelength=1.1,
            handletextpad=0.3,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    if ylabel:
        ax.set_ylabel(ylabel=ylabel, fontsize=18)
    return ax
