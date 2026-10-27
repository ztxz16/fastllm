#!/usr/bin/env python3
"""Export Naive FP8 routed experts as weight-only NVFP4 safetensors.

Requires torch with CUDA and safetensors. Non-expert tensors are copied exactly.
Uses round-to-nearest-even E2M1, E4M3 scales per 16 columns, and an FP32
per-tensor multiplier, following NVIDIA Model Optimizer's NVFP4 convention:
https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
This is weight-only round-to-nearest quantization, without activation calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantize(weight):
    """Return packed E2M1, E4M3 block scales and the FP32 dequant multiplier."""
    rows, cols = weight.shape
    if cols % 16:
        raise ValueError("NVFP4 requires a multiple of 16 columns")
    blocks = weight.float().reshape(rows, cols // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    global_scale = (amax.amax() / (6 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    scales = (amax / (6 * global_scale)).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    normalized = blocks / (scales.float().unsqueeze(-1) * global_scale)
    absolute = normalized.abs()
    bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], device=weight.device)
    codes = torch.bucketize(absolute, bounds, out_int32=True).to(torch.uint8)
    codes += ((absolute == .75) | (absolute == 1.75) | (absolute == 3.5)).to(torch.uint8)
    codes |= (normalized < 0).to(torch.uint8) << 3
    codes = codes.reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.cpu(), scales.cpu(), global_scale.reshape(1).cpu()


def dequantize(packed, scales, global_scale):
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    return (values[codes].reshape(*scales.shape, 16) *
            scales.float().unsqueeze(-1) * global_scale).flatten(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source, output = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "naive_n05_flash":
        raise ValueError("Expected a Naive-N0.5 checkpoint")
    if config.get("quantization_config", {}).get("weight_block_size") != [128, 128]:
        raise ValueError("Expected source FP8 block size [128, 128]")
    shards = sorted(source.glob("*.safetensors"))
    if not shards or output.exists():
        raise ValueError("Source must contain safetensors; output must be a new directory")
    torch.set_num_threads(8)
    output.mkdir(parents=True)
    started = time.monotonic()
    index, dense_hashes, samples = {}, {}, []
    count = total_bytes = source_bytes = 0
    for shard in shards:
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            experts = {key for key in keys if ".mlp.experts." in key and key.endswith(".weight")}
            for name in keys:
                if name.endswith(".weight_scale_inv") and name.removesuffix("_scale_inv") in experts:
                    continue
                tensor = reader.get_tensor(name)
                if name in experts:
                    if tensor.dtype != torch.float8_e4m3fn or tensor.ndim != 2:
                        raise ValueError(f"Expected FP8 expert matrix: {name}")
                    rows, cols = tensor.shape
                    fp8_scales = reader.get_tensor(name + "_scale_inv").to(args.device).float()
                    if rows % 128 or cols % 128 or tuple(fp8_scales.shape) != (rows // 128, cols // 128):
                        raise ValueError(f"Unexpected FP8 scale shape: {name}")
                    weight = tensor.to(args.device).float().reshape(rows // 128, 128, cols // 128, 128)
                    weight = (weight * fp8_scales[:, None, :, None]).reshape(rows, cols)
                    packed, scales, scale2 = quantize(weight)
                    tensors[name] = packed
                    tensors[name + "_scale"] = scales
                    tensors[name + "_scale_2"] = scale2
                    if count % 256 == 0:
                        original = weight.cpu()
                        reconstructed = dequantize(packed, scales, scale2)
                        error = (original - reconstructed).square().sum().item()
                        energy = original.square().sum().item()
                        samples.append({"name": name, "relative_rmse": (error / energy)**.5 if energy else 0.0})
                    count += 1
                else:
                    tensors[name] = tensor
                    dense_hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy()).hexdigest()
            destination = output / shard.name
            temporary = destination.with_suffix(".partial")
            save_file(tensors, temporary, metadata={"format": "pt"})
            temporary.rename(destination)
        for name, tensor in tensors.items():
            index[name] = shard.name
            total_bytes += tensor.numel() * tensor.element_size()
        source_bytes += shard.stat().st_size
        del tensors
        print(json.dumps({"shard": shard.name, "experts": count,
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    for path in source.iterdir():
        if path.is_file() and (path.suffix in {".json", ".jinja", ".txt", ".py"} or
                               path.name in {"LICENSE", "LICENSE.md"}) and path.name not in {
                "config.json", "model.safetensors.index.json"}:
            shutil.copy2(path, output / path.name)
    original_quantization = config.pop("quantization_config")
    config["quantization_config"] = {
        "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
        "exclude_modules": ["lm_head", "model.embed_tokens", "model.layers.0.mlp", "*.self_attn.*", "*.mlp.gate"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes}, "weight_map": index}, indent=2) + "\n")
    report = {"source": str(source), "source_quantization": original_quantization,
              "format": "NVFP4 E2M1 / E4M3 block16 / FP32 tensor scale; weight-only RTN",
              "expert_matrices": count, "source_shard_bytes": source_bytes,
              "tensor_bytes": total_bytes, "seconds": time.monotonic() - started,
              "nonexpert_sha256": dense_hashes, "weight_error_samples": samples}
    (output / "nvfp4_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {count} expert matrices to {output}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
