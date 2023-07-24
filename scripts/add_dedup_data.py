import logging
import os
import time
from argparse import ArgumentParser

import boto3
import pandas as pd
import pyarrow
import webdataset as wds


logger = logging.getLogger(__name__)

LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED = "s3://muse-datasets/hf-datasets-laion-aesthetic6plus-data-pre-encoded"
LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED_WITH_DEDUP_METADATA = (
    "s3://muse-datasets/hf-datasets-laion-aesthetic6plus-data-pre-encoded-with-dedup-metadata"
)

LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED = "s3://muse-datasets/hf-datasets-laion-aesthetics-v2-5-plus-data-pre-encoded"
LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED_WITH_DEDUP_METADATA = "s3://muse-datasets/hf-datasets-laion-aesthetics-v2-5-plus-data-pre-encoded-with-dedup-metadata"

COYO_PRE_ENCODED = "s3://muse-datasets/hf-datasets-coyo-700m-pre-encoded"
COYO_PRE_ENCODED_WITH_DEDUP_METADATA = "s3://muse-datasets/hf-datasets-coyo-700m-pre-encoded-with-dedup-metadata"


def get_dedup_metadata_index():
    dedup_metadata_index = {}

    for idx in range(1, 19):
        t0 = time.perf_counter()
        filename = f"dedup-url-{idx}.txt"

        logger.warning(f"loading {filename}")

        with open(filename) as f:
            for line in f.readlines():
                space = line.rfind(" ")
                shasum = line[:space]
                shard = line[space+1:]
                shard = int(shard)
                dedup_metadata_index[shasum] = shard
        
        logger.warning(f"loaded {filename} in {time.perf_counter() - t0}")

    return dedup_metadata_index


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


def get_dedup_metadata_df(dedup_metadata_shard_number, s3, can_download=False):
    local_dedup_metadata_shard_file = (
        f"./laion-coyo-dedup-metadata/{format_shard_number(dedup_metadata_shard_number)}.parquet"
    )

    # exists = os.path.exists(local_dedup_metadata_shard_file)

    # if can_download:
    #     if not exists:
    #         logger.warning(f"{local_dedup_metadata_shard_file} does not exist, downloading")

    #         with open(local_dedup_metadata_shard_file, "wb") as f:
    #             s3.download_fileobj(
    #                 "muse-datasets",
    #                 f"laion-coyo-dedup-metadata/{format_shard_number(dedup_metadata_shard_number)}.parquet",
    #                 f,
    #             )
    # else:
    #     assert exists

    try:
        df = pd.read_parquet(local_dedup_metadata_shard_file)
    except pyarrow.lib.ArrowInvalid:
        logger.warning(f"invalid parquet file {local_dedup_metadata_shard_file}")
        df = None

    return df


def add_metadata_to_shard(dataset, upload_to, shard, dedup_metadata_index, s3):
    t0 = time.perf_counter()

    download_shards = f"pipe:aws s3 cp {dataset}/{format_shard_number(shard)}.tar -"

    src = wds.WebDataset(
        download_shards,
    ).decode()

    dest = wds.TarWriter(f"pipe:aws s3 cp - {upload_to}/{format_shard_number(shard)}.tar")

    num_not_found = 0
    num_found = 0
    hash_lookup_time = 0
    dedup_read_time = 0
    dedup_update_time = 0
    write_time = 0

    for it in src:
        lookup_value = it["json"]["url"]

        t1 = time.perf_counter()
        dedup_metadata_shard_number = dedup_metadata_index.get(lookup_value, None)
        hash_lookup_time += time.perf_counter() - t1

        if dedup_metadata_shard_number is None:
            # NOTE: helpful for debugging but makes actual logs noisy
            # logger.warning(f"no dedup metadata found for {lookup_value}")
            num_not_found += 1
        else:
            num_found += 1

            t2 = time.perf_counter()
            df = get_dedup_metadata_df(dedup_metadata_shard_number, s3)
            dedup_read_time += time.perf_counter() - t2

            t3 = time.perf_counter()
            if df is not None:
                dedup_metadata = df[df.url == lookup_value].iloc[0].to_json()

                # NOTE: mutation
                it["json"]["dedup_metadata"] = dedup_metadata
            dedup_update_time += time.perf_counter() - t3

        t4 = time.perf_counter()
        dest.write(it)
        write_time += time.perf_counter() - t4

    dest.close()

    logger.warning(f"finished adding metadata to shard: {shard} in {time.perf_counter() - t0}")
    logger.warning(f"found metadata for {num_found}. couldn't find metadata for {num_not_found}")
    logger.warning(f"hash_lookup_time: {hash_lookup_time}")
    logger.warning(f"dedup_read_time: {dedup_read_time}")
    logger.warning(f"dedup_update_time: {dedup_update_time}")
    logger.warning(f"write_time: {write_time}")


def distribute_shards(start_shard_all, end_shard_all, slurm_ntasks):
    total_shards = end_shard_all - start_shard_all + 1
    shards_per_task = total_shards // slurm_ntasks
    shards_per_task = [shards_per_task] * slurm_ntasks

    # to distribute the remainder of tasks for non-evenly divisible number of shards
    left_over_shards = total_shards % slurm_ntasks

    for slurm_procid in range(left_over_shards):
        shards_per_task[slurm_procid] += 1

    assert sum(shards_per_task) == total_shards

    distributed_shards = []

    for slurm_procid in range(len(shards_per_task)):
        if slurm_procid == 0:
            start_shard = start_shard_all
        else:
            start_shard = distributed_shards[slurm_procid - 1][1] + 1

        end_shard = start_shard + shards_per_task[slurm_procid] - 1
        distributed_shards.append((start_shard, end_shard))

    assert sum([end_shard - start_shard + 1 for start_shard, end_shard in distributed_shards]) == total_shards

    return distributed_shards


def main(args):
    s3 = boto3.client("s3")

    if args.dataset == "laion_6":
        dataset = LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED
        upload_to = LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED_WITH_DEDUP_METADATA
    elif args.dataset == "laion_5":
        dataset = LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED
        upload_to = LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED_WITH_DEDUP_METADATA
    elif args.dataset == "coyo":
        dataset = COYO_PRE_ENCODED
        upload_to = COYO_PRE_ENCODED_WITH_DEDUP_METADATA
    else:
        assert False

    if args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])

        distributed_shards = distribute_shards(args.start_shard, args.end_shard, slurm_ntasks)

        start_shard_task, end_shard_task = distributed_shards[slurm_procid]

        args.start_shard = start_shard_task
        args.end_shard = end_shard_task

        logger.warning("************")
        logger.warning("Running as slurm task")
        logger.warning(f"SLURM_NTASKS: {slurm_ntasks}")
        logger.warning(f"SLURM_PROCID: {slurm_procid}")
        logger.warning(f"start_shard: {start_shard_task}, end_shard: {end_shard_task}")
        logger.warning("************")
        logger.warning(f"all slurm processes")
        for slurm_proc_id_, (start_shard, end_shard) in enumerate(distributed_shards):
            logger.warning(f"slurm process: {slurm_proc_id_}, start_shard: {start_shard}, end_shard: {end_shard}")
        logger.warning("************")

    dedup_metadata_index = get_dedup_metadata_index()

    for shard in range(args.start_shard, args.end_shard + 1):
        add_metadata_to_shard(dataset, upload_to, shard, dedup_metadata_index, s3)


def args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to augment",
        choices=["laion_5", "laion_6", "coyo"],
        required=True,
    )
    parser.add_argument(
        "--start_shard",
        type=int,
        help="The starting shard to update with dedup data.",
        required=True,
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        help="The ending shard to update with dedup data, inclusive. If not given, defaults to `--start_shard`.",
        required=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "If set, this process is running under a batch of slurm tasks."
            "`--start_shard` and `--end_shard` must be set for the entirety of shards over all slurm tasks."
            " The shards that will be encoded in each instance of the task will be determined via"
            " the env vars `$SLURM_NTASKS` and `$SLURM_PROCID`."
        ),
    )

    args = parser.parse_args()

    if args.slurm and args.end_shard is None:
        raise ValueError("`--end_shard` must be set when `--slurm` is set")

    if args.end_shard is None:
        args.end_shard = args.start_shard

    if args.end_shard < args.start_shard:
        raise ValueError("`--end_shard` must be >= `--start_shard`")

    return args


if __name__ == "__main__":
    main(args())
