import torch
from torch import nn
from torch.utils.benchmark import Compare, Timer

from muse.transformer_profiling import LayerNormOptionalBias
from muse.transformer_profiling import MaskGitUVit as MaskGitUVitProfile
from muse.transformer_profiling import SingleProjTransformerFFN

# RMSNorm -> Layer norm
# Remove additional projection in transformer ffn
# Replace Global Norm in Res block and/or just replace res blocks with additional transformer blocks

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


class UnfusedRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, use_bias=False):
        super().__init__()
        assert not use_bias

        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, input):
        input_dtype = input.dtype
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)

        if self.elementwise_affine:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                input = input.to(self.weight.dtype)
            input = input * self.weight
        else:
            input = input.to(input_dtype)

        return input


results = []

for config_name, model_args in [
    ("unfused_rms_norm", {"norm_cls": UnfusedRMSNorm}),
    ("base", {}),
    ("ln", {"norm_cls": LayerNormOptionalBias}),
    ("single_proj_ffn", {"ffn_cls": SingleProjTransformerFFN}),
    ("replace res with transformer", {"num_res_blocks": 0, "num_transformer_layers": 25}),
    (
        "all",
        {
            "norm_cls": LayerNormOptionalBias,
            "ffn_cls": SingleProjTransformerFFN,
            "num_res_blocks": 0,
            "num_transformer_layers": 25,
        },
    ),
]:
    model = MaskGitUVitProfile(**model_args)
    model.to("cuda", dtype=torch.float16)

    for batch_size in [1, 2, 4, 8, 16, 32]:
        input_ids = torch.randint(0, 1000, (batch_size, 576), device="cuda")
        encoder_hidden_states = torch.randn((batch_size, 77, 768), device="cuda", dtype=torch.float16)

        def benchmark_fn():
            model(input_ids, encoder_hidden_states)

        timer = Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=1,
            description=config_name,
            label=str(batch_size),
        ).blocked_autorange(min_run_time=1)

        results.append(timer)

    del model

Compare(results).print()
