import time
import msgspec
import logging
from typing import Dict
from argparse import ArgumentParser
import os
from io import StringIO

logger = logging.getLogger(__name__)

def get_dedup_metadata_index(idx):
    t0 = time.perf_counter()
    filename = f"dedup-{idx}.json"

    logger.warning(f"loading {filename}")

    with open(filename) as f:
        f = f.read()
        a_dedup_metadata_index = msgspec.json.decode(f, type=Dict[str, int])

    logger.warning(f"loaded {filename} in {time.perf_counter() - t0}")

    return a_dedup_metadata_index

def create_string(a_dedup_metadata_index):
    t0 = time.perf_counter()

    string_buffer = StringIO()

    for shasum, shard in a_dedup_metadata_index.items():
        string_buffer.write(shasum)
        string_buffer.write(" ")
        string_buffer.write(str(shard))
        string_buffer.write("\n")

    rv = string_buffer.getvalue()

    logger.warning(f"converted to string in {time.perf_counter() - t0}")

    return rv


def write_txt_file(string, idx):
    t0 = time.perf_counter()

    with open(f"dedup-{idx}.txt", "w") as f:
            f.write(string)

    logger.warning(f"written to file in {time.perf_counter() - t0}")

def main(args):
    idx = args.idx

    a_dedup_metadata_index = get_dedup_metadata_index(idx)

    string = create_string(a_dedup_metadata_index)

    write_txt_file(string, idx)

def args():
    args = ArgumentParser()

    args.add_argument("--idx", type=int, required=False, default=None)
    args.add_argument("--slurm", action="store_true")

    args = args.parse_args()

    if args.slurm:
        args.idx = int(os.environ["SLURM_PROCID"]) + 1
        logger.warning(f"running in slurm mode, processing index {args.idx}")
    else:
        if args.idx is None:
            raise ValueError("`--idx` must be set if `--slurm` is not set")

    return args

if __name__ == "__main__":
    main(args())
