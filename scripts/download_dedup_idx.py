from argparse import ArgumentParser
from io import StringIO

import pandas as pd
import pyarrow.fs
import logging
import os

logger = logging.getLogger(__name__)

SPLITS = {
    1: {"start_n_files": 0, "end_n_files": 10_000},
    2: {"start_n_files": 10_000, "end_n_files": 20_000},
    3: {"start_n_files": 20_000, "end_n_files": 30_000},
    4: {"start_n_files": 30_000, "end_n_files": 40_000},
    5: {"start_n_files": 40_000, "end_n_files": 50_000},
    6: {"start_n_files": 50_000, "end_n_files": 60_000},
    7: {"start_n_files": 60_000, "end_n_files": 70_000},
    8: {"start_n_files": 70_000, "end_n_files": 80_000},
    9: {"start_n_files": 80_000, "end_n_files": 90_000},
    10: {"start_n_files": 90_000, "end_n_files": 100_000},
    11: { "start_n_files": 100_000, "end_n_files": 110_000},
    12: { "start_n_files": 110_000, "end_n_files": 120_000},
    13: { "start_n_files": 120_000, "end_n_files": 130_000},
    14: { "start_n_files": 130_000, "end_n_files": 140_000},
    15: { "start_n_files": 140_000, "end_n_files": 150_000},
    16: { "start_n_files": 150_000, "end_n_files": 160_000},
    17: { "start_n_files": 160_000, "end_n_files": 170_000},
    18: { "start_n_files": 170_000, "end_n_files": 175_575},
}


def main(args):
    dedup_idx = args.dedup_idx
    start_n_files = SPLITS[dedup_idx]["start_n_files"]
    end_n_files = SPLITS[dedup_idx]["end_n_files"]

    string_buffer = StringIO()

    for file_n in range(start_n_files, end_n_files):
        print(file_n)

        file_n_str = "{:0>{}}".format(file_n, 5)

        local_dedup_metadata_shard_file = (
            f"./laion-coyo-dedup-metadata/{file_n_str}.parquet"
        )

        try:
            df = pd.read_parquet(local_dedup_metadata_shard_file)
        except pyarrow.lib.ArrowInvalid:
            logger.warning(f"invalid parquet file {local_dedup_metadata_shard_file}")
            df = None

        if "url" not in df:
            continue

        for url in df.url:
            string_buffer.write(url)
            string_buffer.write(" ")
            string_buffer.write(str(file_n))
            string_buffer.write("\n")

    string = string_buffer.getvalue()

    with open(f"dedup-url-{dedup_idx}.txt", "w") as idx_file:
        idx_file.write(string)


def args():
    args = ArgumentParser()

    args.add_argument("--dedup_idx", required=False, type=int)
    args.add_argument("--slurm", action="store_true")

    args = args.parse_args()

    if args.slurm:
        args.dedup_idx = int(os.environ["SLURM_PROCID"]) + 1
        logger.warning(f"running in slurm mode, processing index {args.dedup_idx}")
    else:
        if args.dedup_idx is None:
            raise ValueError("`--dedup_idx` must be set if `--slurm` is not set")

    return args


if __name__ == "__main__":
    main(args())
