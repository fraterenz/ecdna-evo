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
