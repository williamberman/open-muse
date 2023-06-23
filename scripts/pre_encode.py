# This script is used for pre-encoding both laion and coyo.
#
# We pre-encode both captions and images. The captions are all encoded with the CLIP
# checkpoint https://huggingface.co/openMUSE/CLIP-ViT-L-14-DataComp.XL-s13B-b90K-penultimate
#
# The images are pre-encoded with two different vae models,
# https://huggingface.co/openMUSE/paellavq-f8-8192-laion and
# https://huggingface.co/openMUSE/vqgan-f16-8192-laion
#
# We re-upload a single new set of tar archive containing all embeddings:
# <n>.tar
# - <n><x>.pth
# - <n><x>.pth
# - <n><x>.pth
# - ...
#
# Re-uploading a single new tar archive simplifies the pre-embedding and training code.
# The only downside is the downloading of an un-used embedding at train time, but this
# should be fine if we're not bottlenecked by shard download speed.

import argparse
import logging
import webdataset as wds
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import time
import torch
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue
import numpy as np
from PIL.Image import Image

from muse import (
    PaellaVQModel,
    VQGANModel,
)

from transformers import CLIPTokenizerFast, CLIPTextModel

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

PAELLA_F8_VQVAE = "openMUSE/paellavq-f8-8192-laion"
VQGAN_F16_VQVAE = "openMUSE/vqgan-f16-8192-laion"
CLIP = "openMUSE/CLIP-ViT-L-14-DataComp.XL-s13B-b90K-penultimate"

PAELLA_F8_VQVAE_EXT = f"{'.'.join(PAELLA_F8_VQVAE.split('/'))}.pth"
VQGAN_F16_VQVAE_EXT = f"{'.'.join(VQGAN_F16_VQVAE.split('/'))}.pth"
CLIP_EXT = f"{'.'.join(CLIP.split('/'))}.pth"

LAION_AESTHETICS_V2_5_PLUS = "s3://hf-datasets-laion-5b-us-west-2/glacier/laion-data/laion-aesthetics-v2-5-plus-data"
LAION_AESTHETICS_V2_6_PLUS = "s3://muse-datasets/laion-aesthetic6plus-data"
COYO = "s3://hf-datasets-coyo-700m-us-west-2/data"

LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED = "s3://muse-datasets/hf-datasets-laion-aesthetics-v2-5-plus-data-pre-encoded"
LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED = "s3://muse-datasets/hf-datasets-laion-aesthetic6plus-data-pre-encoded"
COYO_PRE_ENCODED = "s3://muse-datasets/hf-datasets-coyo-700m-pre-encoded"

logger = logging.getLogger(__name__)


class TimingWrapper:
    def __init__(self, wrap):
        self.wrap = wrap
        self.timing = None

    def __iter__(self):
        self.timing = 0

        iterator = iter(self.wrap)

        while True:
            t0 = time.perf_counter()

            try:
                next_ = next(iterator)
            except StopIteration:
                break

            self.timing += time.perf_counter() - t0

            yield next_


def upload_process_body(fileobj, upload_queue, skip_upload):
    logger.warning(f"upload process: starting {fileobj}")

    if not skip_upload:
        with wds.TarWriter(fileobj) as dst:
            while True:
                sample = upload_queue.get(block=True)

                if sample is None:
                    break

                dst.write(sample)
    else:
        logger.warning("upload process: skipping uploads")
        while True:
            sample = upload_queue.get(block=True)

            if sample is None:
                break

    upload_queue.close()
    logger.warning(f"upload process: finishing {fileobj}")


def main():
    t0_setup = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to pre-encode",
        choices=["laion_5", "laion_6", "coyo"],
        required=True,
    )
    parser.add_argument(
        "--start_shard",
        type=int,
        help="The starting shard to pre-encode.",
        required=True,
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        help="The ending shard to pre-encode, inclusive. If not given, defaults to `--start_shard`.",
        required=False,
    )
    parser.add_argument(
        "--batch_size", type=int, help="The batch size to encode at a time", required=False, default=160
    )
    parser.add_argument(
        "--resolution", type=int, help="The resolution to convert the image to.", required=False, default=256
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug. Will syncronize cuda between model calls for better timing measurement.",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Set to not actually upload results, helpful for only testing encoding.",
    )

    args = parser.parse_args()

    if args.end_shard is None:
        args.end_shard = args.start_shard

    if args.end_shard < args.start_shard:
        raise ValueError("end_shard must be >= start_shard")

    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if args.resolution < 1:
        raise ValueError("resolution must be >= 1")

    if args.dataset == "laion_5":
        args.dataset = LAION_AESTHETICS_V2_5_PLUS
    elif args.dataset == "laion_6":
        args.dataset = LAION_AESTHETICS_V2_6_PLUS
    elif args.dataset == "coyo":
        args.dataset = COYO
    else:
        assert False

    if args.dataset == LAION_AESTHETICS_V2_5_PLUS:
        upload_to = LAION_AESTHETICS_V2_5_PLUS_PRE_ENCODED
    elif args.dataset == LAION_AESTHETICS_V2_6_PLUS:
        upload_to = LAION_AESTHETICS_V2_6_PLUS_PRE_ENCODED
    elif args.dataset == COYO:
        upload_to = COYO_PRE_ENCODED
    else:
        assert False

    logger.warning("********************")
    logger.warning("Pre-encoding dataset")
    logger.warning(f"dataset: {args.dataset}")
    logger.warning(f"start_shard: {args.start_shard}")
    logger.warning(f"end_shard: {args.end_shard}")
    logger.warning(f"upload_to: {upload_to}")
    logger.warning(f"batch_size: {args.batch_size}")
    logger.warning(f"debug: {args.debug}")
    logger.warning("********************")

    vae_f8 = PaellaVQModel.from_pretrained(PAELLA_F8_VQVAE)
    vae_f8.to("cuda")
    vae_f8.requires_grad_(False)

    vae_f16 = VQGANModel.from_pretrained(VQGAN_F16_VQVAE)
    vae_f16.to("cuda")
    vae_f16.requires_grad_(False)

    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP)
    text_encoder = CLIPTextModel.from_pretrained(CLIP)
    text_encoder.to_bettertransformer()
    text_encoder.to("cuda")

    def image_transforms(image: Image):
        t0 = time.perf_counter()

        mode = image.mode

        height = image.height
        width = image.width

        if hasattr(image, "getbands"):
            channels = len(image.getbands())
        else:
            channels = image.channels

        if mode == "I":
            nptype = np.int32
        elif mode == "I;16":
            nptype = np.int16
        elif mode == "F":
            nptype = np.float32
        else:
            nptype = np.uint8

        image = np.array(image, nptype)
        image = torch.from_numpy(image)

        t1 = time.perf_counter()
        return image, mode, height, width, channels, t1 - t0

    def tokenize(text):
        t0 = time.perf_counter()
        input_ids = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids[0]
        t = time.perf_counter() - t0
        return input_ids, t

    def collate_with_timings(datapoints):
        num_timings = len(datapoints[0]) - 1

        all_t = [0] * num_timings
        all = []

        for datapoint, *t in datapoints:
            for i in range(num_timings):
                all_t[i] += t[i]

            all.append(datapoint)

        all = torch.stack(all)

        return all, *all_t

    def collate_images_with_timings(images):
        num_timings = len(images[0]) - 5

        all_t = [0] * num_timings
        images_ = []

        for image, mode, height, width, channels, *t in images:
            for i in range(num_timings):
                all_t[i] += t[i]

            images_.append((image, mode, height, width, channels))

        return images_, *all_t

    def collate_fn(args):
        __key__, images, input_ids = args
        images = collate_images_with_timings(images)
        input_ids = collate_with_timings(input_ids)

        return __key__, images, input_ids

    time_setup = time.perf_counter() - t0_setup

    for shard in range(args.start_shard, args.end_shard + 1):
        shard = "{:0>{}}".format(shard, 5)
        download_shard_url = f"{args.dataset}/{shard}.tar"
        logger.warning(f"Downloading shard {download_shard_url}")

        src = (
            wds.WebDataset(f"pipe:aws s3 cp {download_shard_url} -")
            .decode("pil")
            .rename(image="jpg;png;jpeg;webp", input_ids="text;txt;caption")
            .map(lambda dict: {"__key__": dict["__key__"], "image": dict["image"], "input_ids": dict["input_ids"]})
            .map_dict(image=image_transforms, input_ids=tokenize)
            .to_tuple("__key__", "image", "input_ids")
            .batched(args.batch_size)
        )
        src = DataLoader(
            src,
            batch_size=None,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=5,
        )
        src = TimingWrapper(src)

        upload_shard_url = f"{upload_to}/{shard}.tar"
        logger.warning(f"Uploading shard {upload_shard_url}")

        upload_queue = Queue()
        upload_process = Process(
            target=upload_process_body,
            args=(f"pipe:aws s3 cp - {upload_shard_url}", upload_queue, args.skip_upload),
        )
        upload_process.start()

        batch_ctr = 0
        img_ctr = 0
        time_to_cuda = 0
        time_postprocess = 0
        time_write = 0
        time_encoding_f8 = 0
        time_encoding_f16 = 0
        time_encoding_text_encoder = 0
        time_image_dataloader_to_tensor = 0
        time_image_dataloader_convert = 0
        time_image_dataloader_resize = 0
        time_image_dataloader_center_crop = 0
        time_tokenize_dataloader = 0

        for __key__, image, input_ids in src:
            batch_ctr += 1
            img_ctr += len(__key__)
            logger.warning(f"Encoding {len(__key__)} examples: {__key__[0]} to {__key__[-1]}.")

            (
                image,
                time_image_dataloader_to_tensor_,
            ) = image
            time_image_dataloader_to_tensor += time_image_dataloader_to_tensor_

            input_ids, time_tokenize_dataloader_ = input_ids
            time_tokenize_dataloader += time_tokenize_dataloader_

            t0 = time.perf_counter()
            input_ids = input_ids.to("cuda")
            time_to_cuda += time.perf_counter() - t0

            all_images = []

            for image_, mode, height, width, channels in image:
                t0 = time.perf_counter()
                image_ = image_.to("cuda")
                if args.debug:
                    torch.cuda.synchronize()
                time_to_cuda += time.perf_counter() - t0

                t0 = time.perf_counter()
                image_ = image_.view(height, width, channels)
                image_ = image_.permute((2, 0, 1)).contiguous()

                if mode != "1" and image_.dtype == torch.uint8:
                    image_ = image_.to(dtype=torch.float32).div(255)
                if args.debug:
                    torch.cuda.synchronize()
                time_image_dataloader_convert += time.perf_counter() - t0

                t0 = time.perf_counter()
                image_ = TF.resize(image_, size=args.resolution, interpolation=InterpolationMode.BILINEAR)
                if args.debug:
                    torch.cuda.synchronize()
                time_image_dataloader_resize += time.perf_counter() - t0

                t0 = time.perf_counter()
                image_ = TF.center_crop(image_, args.resolution)
                if args.debug:
                    torch.cuda.synchronize()
                time_image_dataloader_center_crop += time.perf_counter() - t0

                all_images.append(image_)

            image = torch.stack(all_images)

            t0 = time.perf_counter()
            with torch.cuda.amp.autocast():
                encoded_image_f8 = vae_f8.get_code(image)
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_f8 += time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.cuda.amp.autocast():
                encoded_image_f16 = vae_f16.get_code(image)
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_f16 += time.perf_counter() - t0

            t0 = time.perf_counter()
            encoder_hidden_states = text_encoder(input_ids)[0]
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_text_encoder += time.perf_counter() - t0

            # when saving a view of a tensor, pytorch will save the entirety of the original tensor.
            # cloning the view, will save just the subset of the original tensor.
            t0 = time.perf_counter()
            encoded_image_f8 = [x.clone().to("cpu") for x in torch.unbind(encoded_image_f8)]
            encoded_image_f16 = [x.clone().to("cpu") for x in torch.unbind(encoded_image_f16)]
            encoder_hidden_states = [x.clone().to("cpu") for x in torch.unbind(encoder_hidden_states)]
            time_postprocess += time.perf_counter() - t0

            logger.warning("Writing examples")

            t0 = time.perf_counter()
            for __key__, encoded_image_f8, encoded_image_f16, encoder_hidden_states in zip(
                __key__, encoded_image_f8, encoded_image_f16, encoder_hidden_states
            ):
                sample = {
                    "__key__": __key__,
                    PAELLA_F8_VQVAE_EXT: encoded_image_f8,
                    VQGAN_F16_VQVAE_EXT: encoded_image_f16,
                    CLIP_EXT: encoder_hidden_states,
                }

                upload_queue.put(sample, block=False)
            time_write += time.perf_counter() - t0

        upload_queue.put(None, block=True)
        upload_queue.close()
        upload_process.join()

        def safe_div_batch_ctr(n):
            if batch_ctr == 0:
                return n
            else:
                return n / batch_ctr

        def log_batched(name, n):
            logger.warning(f"{name}: total: {n}, per batch: {safe_div_batch_ctr(n)}")

        logger.warning("************")
        logger.warning(f"num batches: {batch_ctr}")
        logger.warning(f"num images: {img_ctr}")
        logger.warning("************")
        logger.warning("timing")
        logger.warning(f"time_setup: {time_setup}")
        log_batched("time_dataloader", src.timing)
        log_batched("time_image_dataloader_to_tensor", time_image_dataloader_to_tensor)
        log_batched("time_image_dataloader_convert", time_image_dataloader_convert)
        log_batched("time_image_dataloader_resize", time_image_dataloader_resize)
        log_batched("time_image_dataloader_center_crop", time_image_dataloader_center_crop)
        time_image_dataloader = (
            time_image_dataloader_to_tensor
            + time_image_dataloader_convert
            + time_image_dataloader_resize
            + time_image_dataloader_center_crop
        )
        log_batched("time_image_dataloader", time_image_dataloader)
        log_batched("time_tokenize_dataloader", time_tokenize_dataloader)
        log_batched("time_to_cuda", time_to_cuda)
        log_batched("time_encoding_f8", time_encoding_f8)
        log_batched("time_encoding_f16", time_encoding_f16)
        log_batched("time_encoding_text_encoder", time_encoding_text_encoder)
        log_batched("time_postprocess", time_postprocess)
        log_batched("time_write", time_write)
        logger.warning("************")


if __name__ == "__main__":
    main()
