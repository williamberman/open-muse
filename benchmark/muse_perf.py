import csv
from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.benchmark import Compare, Timer
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from muse import MaskGiTUViT, PipelineMuse, VQGANModel

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"


def main():
    args = ArgumentParser()
    args.add_argument("--device", choices=["4090", "a100"], required=True)

    args = args.parse_args()
    csv_data = []

    for batch_size in [1, 8]:
        for timesteps in [12, 20]:
            for resolution in [256, 512]:
                for use_xformers in [False, True]:
                    out, mem_bytes = sd_benchmark(
                        resolution=resolution, batch_size=batch_size, timesteps=timesteps, use_xformers=use_xformers
                    )

                    Compare([out]).print()
                    print("*******")

                    csv_data.append(
                        [
                            batch_size,
                            "muse",
                            out.median * 1000,
                            args.device,
                            timesteps,
                            mem_bytes,
                            resolution,
                            use_xformers,
                            None,
                            None,
                        ]
                    )

                    for use_fused_mlp in [False, True]:
                        for use_fused_residual_norm in [False, True]:
                            out, mem_bytes = muse_benchmark(
                                resolution=resolution,
                                batch_size=batch_size,
                                timesteps=timesteps,
                                use_xformers=use_xformers,
                                use_fused_mlp=use_fused_mlp,
                                use_fused_residual_norm=use_fused_residual_norm,
                            )

                            Compare([out]).print()
                            print("*******")

                            csv_data.append(
                                [
                                    batch_size,
                                    "stable_diffusion_1_5",
                                    out.median * 1000,
                                    args.device,
                                    timesteps,
                                    mem_bytes,
                                    resolution,
                                    use_xformers,
                                    use_fused_mlp,
                                    use_fused_residual_norm,
                                ]
                            )

    with open("benchmark/artifacts/all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark(resolution, batch_size, timesteps, use_xformers, use_fused_mlp, use_fused_residual_norm):
    model = "williamberman/muse_research_run_benchmarking_512_output"
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    text_encoder = CLIPTextModelWithProjection.from_pretrained(model, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype)

    vae = VQGANModel.from_pretrained(model, subfolder="vae")
    vae.to(device=device, dtype=dtype)

    if use_fused_mlp:
        ffn_type = "vanilla"
    else:
        ffn_type = "glu"

    transformer = MaskGiTUViT(
        **research_run_transformer_config,
        use_fused_mlp=use_fused_mlp,
        use_fused_residual_norm=use_fused_residual_norm,
        ffn_type=ffn_type,
    )
    transformer = transformer.to(device=device, dtype=dtype)

    if use_xformers:
        transformer.enable_xformers_memory_efficient_attention()

    pipe = PipelineMuse(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
    )
    pipe.device = device
    pipe.dtype = dtype

    seq_len = (resolution // 16) ** 2

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=timesteps, transformer_seq_len=seq_len)

    pipe(prompt, num_images_per_prompt=batch_size, timesteps=2, transformer_seq_len=seq_len)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def sd_benchmark(resolution, batch_size, timesteps, use_xformers):
    model = "runwayml/stable-diffusion-v1-5"
    device = "cuda"
    dtype = torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")
    unet = unet.to(device=device, dtype=dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
    )

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    def benchmark_fn():
        pipe(
            prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=timesteps,
            height=resolution,
            width=resolution,
        )

    pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=2, height=resolution, width=resolution)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def measure_max_memory_allocated(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    rv = fn()

    mem_bytes = torch.cuda.max_memory_allocated()

    return rv, mem_bytes


research_run_transformer_config = {
    "_class_name": "MaskGiTUViT",
    "_version": "0.0.1",
    "add_cond_embeds": True,
    "add_cross_attention": True,
    "add_micro_cond_embeds": True,
    "attention_dropout": 0.0,
    "block_has_attention": [True],
    "block_num_heads": [12],
    "block_out_channels": [768],
    "codebook_size": 8192,
    "cond_embed_dim": 768,
    "encoder_hidden_size": 768,
    "ffn_type": "glu",
    "hidden_dropout": 0.0,
    "hidden_size": 1024,
    "in_channels": 768,
    "initializer_range": 0.02,
    "intermediate_size": 2816,
    "layer_norm_before_mlm": False,
    "layer_norm_embedddings": False,
    "layer_norm_eps": 0.000001,
    "learn_uncond_embeds": False,
    "ln_elementwise_affine": True,
    "mask_token_id": 8255,
    "max_position_embeddings": 256,
    "micro_cond_embed_dim": 1280,
    "micro_cond_encode_dim": 256,
    "norm_type": "rmsnorm",
    "num_attention_heads": 16,
    "num_classes": null,
    "num_hidden_layers": 22,
    "num_res_blocks": 3,
    "num_vq_tokens": 256,
    "patch_size": 1,
    "project_encoder_hidden_states": True,
    "res_ffn_factor": 4,
    "use_bias": False,
    "use_codebook_size_for_output": True,
    "use_empty_embeds_for_uncond": True,
    "use_encoder_layernorm": False,
    "use_normformer": False,
    "use_position_embeddings": False,
    "use_vannilla_resblock": False,
    "vocab_size": 8256,
    "xavier_init_embed": True,
}

if __name__ == "__main__":
    main()
