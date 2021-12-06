import re
import tarfile
from pathlib import Path, PurePath
import pandas as pd


def find_metadata_files(path: Path) -> [Path]:
    """From parent directory `path`, find all metdata tarballs in all subdirectories"""
    try:
        return find_files(path, "metadata.tar.gz")
    except AssertionError as e:
        raise ValueError(e)


def find_all_rates_files(path: Path) -> [Path]:
    """From parent directory `path`, find all all_rates tarballs in all subdirectories"""
    try:
        return find_files(path, "all_rates.tar.gz")
    except AssertionError as e:
        raise ValueError(e)


def find_files(path: Path, filename: str) -> [Path]:
    assert path.is_dir(), "{} is not a dir".format(path)
    return [Path(adir) / filename for adir in path.iterdir()]


def infer_runs(path2file: str) -> int:
    pattern_runs = re.compile(r"(\d+)runs_")
    try:
        return re.search(pattern_runs, path2file).group(1)
    except AttributeError:
        raise ValueError("Cannot infer the nb of runs from file '{}'".format(path2file))


def infer_ground_truth_idx(path2file: str) -> int:
    """From path2file extract the integer representing the idx of the ground truth run"""
    pattern_runs = re.compile(r"_(\d+)idx")
    processed_path = PurePath(path2file).parent.stem
    try:
        return int(re.search(pattern_runs, processed_path).group(1))
    except AttributeError:
        raise ValueError(
            "Cannot infer the idx of the ground truth from file '{}'".format(path2file)
        )


def load(path2data: Path, dtype: str):
    """Create a dataframe with idx representing the runs idx of the ground truth used and the entries can be anything. Returns also the ground truth idx that is found in `path2data`.
    `path2data` must end with _Xidx where X is any int
    """
    return extract_tarball(path2data, dtype, True)


def extract_tarball(path2data: Path, dtype: str, header: bool = False):
    """Create dataframe from tarball in `path2data`, where each member is a file with or without any `header`. All files (members) must have the same format (nb columns) and must only have at max 2 rows (in this case `header` must be true)."""
    idx, data = [], []
    with tarfile.open(path2data, "r:gz") as tar:
        for i, member in enumerate(tar.getmembers()):
            # name of the member (filename)

            if member.isfile():
                run_idx = int(Path(member.name).stem)
                idx.append(run_idx)

                f = tar.extractfile(member)
                lines = f.readlines()
                if header:
                    data.append(
                        {
                            k: float(val)
                            for (k, val) in zip(
                                lines[0].decode().strip("\n").split(","),
                                lines[1].decode().strip("\n").split(","),
                            )
                        }
                    )
                else:
                    data.append(lines[0].decode().strip("\n").split(","))

    print("Loaded {} files".format(i))
    idx = pd.Series(idx, dtype=int)
    data = pd.DataFrame.from_records(data, index=idx).astype(dtype=dtype)
    data.index.rename("run", inplace=True)
    try:
        gr_truth = infer_ground_truth_idx(path2data)
    except ValueError:
        gr_truth = None

    return data, gr_truth
