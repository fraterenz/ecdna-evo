#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path


def build_app() -> (Path, Path, Path, int):
    """Parse and returns the paths to:
    1. rates.tar.gz: the tarball where the proliferation and death rates are
        stored only for the ABC runs that match the patient's data
    2. all_rates.tar.gz: same as above, but for all runs (not only the
        matching ones)
    3. values.tar.gz: the tarball where the distances between the patient's
        data and the runs are stored, for all runs
    Returns also an integer representing the percentage for the threshold for
    the runs accepted (e.g. 5 means plot rates for runs that have
    a relative distance, stored in values.tar.gz, smaller or equal than 5%)
    """
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="Plot posterior distribution (histograms) for ABC inference."
    )
    parser.add_argument(
        "path2rates",
        action="store",
        metavar="FILE",
        type=str,
        help="Path to tarball file rates.tar.gz",
    )

    subparsers = parser.add_subparsers(title="subcomand")

    # create the parser for the "stats" command
    parser_stats = subparsers.add_parser(
        "stats",
        help="Plot posterior distribution of the fitness coefficient for all combinations of statistics given a relative threshold",
    )

    parser_stats.add_argument(
        "threshold",
        type=int,
        help="integer specifying the relative difference percentage threshold for which the posterior will be plotted",
    )

    parser_stats.add_argument(
        "--all-rates",
        metavar="FILE",
        dest="path2all_rates",
        required=True,
        type=str,
        help="Path to tarball file all_rates.tar.gz",
    )

    parser_stats.add_argument(
        "--values",
        dest="path2values",
        metavar="FILE",
        type=str,
        required=True,
        help="Path to tarball file values.tar.gz",
    )

    # TODO remove?
    # parser_stats.add_argument(
    #     "--metadata",
    #     dest="path2metadata",
    #     metavar="FILE",
    #     type=str,
    #     required=False,
    #     help="Path to tarball file metadata.tar.gz",
    # )

    args = vars(parser.parse_args())

    try:
        path2all_rates = Path(args["path2all_rates"])
    except KeyError:
        return Path(args["path2rates"]), None, None, None

    try:
        path2values = Path(args["path2values"])
    except KeyError:
        raise parser.error("When `stats` is present, `--values` is required")

    try:
        threshold = int(args["threshold"])
    except KeyError:
        raise parser.error("When `stats` is present, the threshold must be specified")

    return Path(args["path2rates"]), path2all_rates, path2values, threshold
