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
from torchvision import transforms
import time
import torch
from multiprocessing import Process, Queue

from muse import (
    PaellaVQModel,
    VQGANModel,
)

from transformers import CLIPTokenizer, CLIPTextModel

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


def upload_process_body(fileobj, upload_queue):
    logger.warning(f"upload process: starting {fileobj}")

    with wds.TarWriter(fileobj) as dst:
        while True:
            sample = upload_queue.get(block=True)

            if sample is None:
                break

            dst.write(sample)

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
    vae_f8 = torch.compile(vae_f8, mode="reduce-overhead")  # inputs are cropped to fixed size, so will not re-compile
    vae_f8(torch.rand((args.batch_size, 3, args.resolution, args.resolution), device="cuda"))

    vae_f16 = VQGANModel.from_pretrained(VQGAN_F16_VQVAE)
    vae_f16.to("cuda")
    vae_f16.requires_grad_(False)
    vae_f16 = torch.compile(vae_f16, mode="reduce-overhead")
    # TODO - throws illegal access error
    # vae_f16(torch.rand((args.batch_size, 3, args.resolution, args.resolution), device='cuda'))

    tokenizer = CLIPTokenizer.from_pretrained(CLIP)
    text_encoder = CLIPTextModel.from_pretrained(CLIP)
    text_encoder.to("cuda")
    text_encoder.requires_grad_(False)
    text_encoder = torch.compile(text_encoder, mode="reduce-overhead")
    text_encoder(torch.randint(0, 8000, (args.batch_size, tokenizer.model_max_length), device="cuda"))

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def tokenize(text):
        input_ids = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return input_ids[0]

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
        src = wds.WebLoader(src, batch_size=None, shuffle=False, num_workers=1, pin_memory=True)
        src = TimingWrapper(src)

        upload_shard_url = f"{upload_to}/{shard}.tar"
        logger.warning(f"Uploading shard {upload_shard_url}")

        upload_queue = Queue()
        upload_process = Process(
            target=upload_process_body,
            args=(
                f"pipe:aws s3 cp - {upload_shard_url}",
                upload_queue,
            ),
        )
        upload_process.start()

        batch_ctr = 0
        img_ctr = 0
        time_to_cuda = 0
        time_to_cpu = 0
        time_postprocess = 0
        time_write = 0
        time_encoding_f8 = 0
        time_encoding_f16 = 0
        time_encoding_text_encoder = 0

        for __key__, image, input_ids in src:
            batch_ctr += 1
            img_ctr += len(__key__)
            logger.warning(f"Encoding examples {__key__}")

            t0 = time.perf_counter()
            image = image.to("cuda")
            input_ids = input_ids.to("cuda")
            time_to_cuda += time.perf_counter() - t0

            t0 = time.perf_counter()
            encoded_image_f8 = vae_f8.get_code(image)
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_f8 += time.perf_counter() - t0

            t0 = time.perf_counter()
            encoded_image_f16 = vae_f16.get_code(image)
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_f16 += time.perf_counter() - t0

            t0 = time.perf_counter()
            encoder_hidden_states = text_encoder(input_ids)[0]
            if args.debug:
                torch.cuda.synchronize()
            time_encoding_text_encoder += time.perf_counter() - t0

            t0 = time.perf_counter()
            encoded_image_f8 = encoded_image_f8.to("cpu")
            encoded_image_f16 = encoded_image_f16.to("cpu")
            encoder_hidden_states = encoder_hidden_states.to("cpu")
            time_to_cpu += time.perf_counter() - t0

            # when saving a view of a tensor, pytorch will save the entirety of the original tensor.
            # cloning the view, will save just the subset of the original tensor.
            t0 = time.perf_counter()
            encoded_image_f8 = [x.clone() for x in torch.unbind(encoded_image_f8)]
            encoded_image_f16 = [x.clone() for x in torch.unbind(encoded_image_f16)]
            encoder_hidden_states = [x.clone() for x in torch.unbind(encoder_hidden_states)]
            time_postprocess += time.perf_counter() - t0

            logger.warning("Writing examples")

            t0 = time.perf_counter()
            for __key__, encoded_image_f8, encoded_image_f16, encoder_hidden_states in zip(
                __key__, encoded_image_f8, encoded_image_f16, encoder_hidden_states
            ):
                sample = {
                    "__key__": __key__,
                    f"{'.'.join(PAELLA_F8_VQVAE.split('/'))}.pth": encoded_image_f8,
                    f"{'.'.join(VQGAN_F16_VQVAE.split('/'))}.pth": encoded_image_f16,
                    f"{'.'.join(CLIP.split('/'))}.pth": encoder_hidden_states,
                }

                upload_queue.put(sample, block=False)
            time_write += time.perf_counter() - t0

        upload_queue.put(None, block=True)
        upload_queue.close()
        upload_process.join()

        logger.warning("************")
        logger.warning(f"num batches: {batch_ctr}")
        logger.warning(f"num images: {img_ctr}")
        logger.warning("************")
        logger.warning("timing")
        logger.warning(f"time_setup: {time_setup}")
        logger.warning(f"time_dataset: {src.timing}")
        logger.warning(
            f"time_to_cuda: total: {time_to_cuda}, per batch: {time_to_cuda / batch_ctr if batch_ctr != 0 else time_to_cuda}"
        )
        logger.warning(
            f"time_encoding_f8: total: {time_encoding_f8}, per batch: {time_encoding_f8 / batch_ctr if batch_ctr != 0 else time_encoding_f8}"
        )
        logger.warning(
            f"time_encoding_f16: total: {time_encoding_f16}, per batch: {time_encoding_f16 / batch_ctr if batch_ctr != 0 else time_encoding_f16}"
        )
        logger.warning(
            f"time_encoding_text_encoder: total: {time_encoding_text_encoder}, per batch: {time_encoding_text_encoder / batch_ctr if batch_ctr != 0 else time_encoding_text_encoder}"
        )
        logger.warning(
            f"time_to_cpu: total: {time_to_cpu}, per batch: {time_to_cpu / batch_ctr if batch_ctr != 0 else time_to_cpu}"
        )
        logger.warning(
            f"time_postprocess: total: {time_postprocess}, per batch: {time_postprocess / batch_ctr if batch_ctr != 0 else time_postprocess}"
        )
        logger.warning(
            f"time_write: total: {time_write}, per_batch: {time_write / batch_ctr if batch_ctr != 0 else time_write}"
        )
        logger.warning("************")


if __name__ == "__main__":
    main()
