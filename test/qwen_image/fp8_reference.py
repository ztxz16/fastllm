"""Development-only FP8 oracle using PyTorch scaled_mm, independent of native GEMM."""

import json
import torch
from safetensors import safe_open


class FP8Linear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.register_buffer("weight", weight.cuda())
        self.register_buffer("scale", scale.float().reshape(-1, 1).cuda())

    def forward(self, x):
        assert x.dtype == torch.bfloat16
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        # TorchAO _choose_scale_float8 divides in the input dtype, then casts
        # to FP32. Preserve that BF16 rounding before quantizing the payload.
        # https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
        scale = (x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448).float()
        quantized = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        result = torch._scaled_mm(
            quantized, self.weight.t(), scale_a=scale, scale_b=self.scale.t(),
            out_dtype=torch.bfloat16, use_fast_accum=True,
        )
        return result.reshape(*shape[:-1], self.weight.shape[0])


def install_fp8(model, checkpoint):
    """Replace the released TorchAO row-scaled projections; keep all other layers."""
    count = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
        for name in weights.keys():
            if not name.endswith("._weight_qdata"):
                continue
            prefix = name.removesuffix("._weight_qdata")
            original = model.get_submodule(prefix)
            assert isinstance(original, torch.nn.Linear) and original.bias is None, prefix
            weight = weights.get_tensor(name)
            scale = weights.get_tensor(prefix + "._weight_scale")
            description = json.loads(metadata[prefix + ".weight"])
            assert weight.dtype == torch.float8_e4m3fn and weight.shape == original.weight.shape
            assert description["_data"]["block_size"] == [1, weight.shape[1]]
            assert description["_data"]["act_quant_kwargs"]["_data"]["granularity"]["_type"] == "PerRow"
            assert scale.numel() == weight.shape[0] and torch.isfinite(scale).all() and (scale > 0).all()
            parent, _, child = prefix.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, FP8Linear(weight, scale))
            count += 1
    assert count, "checkpoint has no FP8 projections"
    torch.cuda.empty_cache()
    return count
