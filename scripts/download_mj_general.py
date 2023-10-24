import argparse
import logging
import os
import time
from io import BytesIO

import pandas as pd
import requests
import webdataset as wds
from PIL import Image

logger = logging.getLogger(__name__)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--slurm", action="store_true")
    args = args.parse_args()

    # download parquet from https://huggingface.co/datasets/ptx0/mj-general/tree/main

    data = pd.read_parquet("/fsx/william/tmp/mj-general-0002.parquet")

    write_to = "pipe:aws s3 cp - s3://muse-datasets/mj-general/0002"

    checkpoint_dir = "/fsx/william/tmp/download_mj_general_0002"

    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])
        n_rows_per_job = len(data) // slurm_ntasks
        start_idx = slurm_procid * n_rows_per_job
        data = data[start_idx : start_idx + n_rows_per_job]
        write_to_subfolder = slurm_procid
    else:
        write_to_subfolder = 0

    data = data.iterrows()
    data = take_second(data)
    data = take_up_to(data, 500)

    subsequent_errors = 0
    total_errors = 0

    for shard_n, data_ in enumerate(data):
        writer = wds.TarWriter(f"{write_to}/{write_to_subfolder}/{format_shard_number(shard_n)}.tar")

        key = 0

        t0 = time.perf_counter()

        for row in data_:
            url = row.url
            prompt = row.caption

            response = requests.get(url)

            if response.ok:
                subsequent_errors = 0
            else:
                subsequent_errors += 1
                total_errors += 1

                logger.error(f"error in request subsequent_errors {subsequent_errors} {total_errors}")
                logger.error("**********")
                logger.error(response.text)
                logger.error("**********")

                if subsequent_errors < 10:
                    # Not obvious to me any of these are rate limiting, but just in case sleep
                    time.sleep(5)
                    continue
                else:
                    raise ValueError(f"Too many errors, killing job")

            image = Image.open(BytesIO(response.content))

            writer.write(
                {
                    "__key__": format_shard_number(key),
                    "png": image,
                    "txt": prompt,
                }
            )

            key += 1

            logger.warning(f"{write_to_subfolder} {shard_n} {key}")

        writer.close()

        logger.warning(f"{write_to_subfolder} {shard_n} {time.perf_counter() - t0}")

        with open(f"{checkpoint_dir}/{write_to_subfolder}", "w") as checkpoint:
            checkpoint.write(str(shard_n))


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


def take_second(iterator):
    for _, it in iterator:
        yield it


def take_up_to(iterator, n):
    iterator = iter(iterator)

    iterator_has_elements = True

    while iterator_has_elements:
        items = []

        for _ in range(n):
            try:
                items.append(next(iterator))
            except StopIteration:
                iterator_has_elements = False

        yield items


if __name__ == "__main__":
    main()
