import json
from argparse import ArgumentParser

import pandas as pd
import pyarrow.fs

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
}


def main(args):
    dedup_idx = args.dedup_idx
    start_n_files = SPLITS[dedup_idx]["start_n_files"]
    end_n_files = SPLITS[dedup_idx]["end_n_files"]

    s3 = pyarrow.fs.S3FileSystem()
    idx = {}

    for file_n in range(start_n_files, end_n_files):
        print(file_n)

        file_n_str = "{:0>{}}".format(file_n, 5)

        with s3.open_input_file(f"muse-datasets/laion-coyo-dedup-metadata/{file_n_str}.parquet") as f:
            df = pd.read_parquet(f)

            if "sha256" not in df:
                continue

            for sha256 in df.sha256:
                idx[sha256] = file_n

    with open(f"dedup-{dedup_idx}.json", "w") as idx_file:
        json.dump(idx, idx_file)


def args():
    args = ArgumentParser()
    args.add_argument("--dedup_idx", required=True, type=int)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    main()
