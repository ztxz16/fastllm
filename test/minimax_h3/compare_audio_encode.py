"""Compare the native FP32 audio encoder with the official Diffusers implementation."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model/audio_vae")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument("--samples", type=int, default=6401)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(101)
vae = (
    AutoencoderKLMiniMaxH3Audio.from_pretrained(a.model, torch_dtype=torch.float32)
    .cuda()
    .eval()
)
values = {}


def hook(name):
    def capture(module, inputs, result):
        values[name] = result.transpose(1, 2).detach().float().cpu().numpy().reshape(-1)

    return capture


for i in range(6):
    vae.encoder.block[i].register_forward_hook(hook(f"encoder.block.{i}"))
vae.encoder.register_forward_hook(hook("encoder"))
vae.pre_block.register_forward_hook(
    lambda m, args, result: values.update(
        {"pre_block": result.detach().float().cpu().numpy().reshape(-1)}
    )
)
vae.mean_proj.register_forward_hook(hook("mean_proj"))
inputs = torch.randn(1, 1, a.samples, device="cuda") * 0.2
with torch.inference_mode():
    result = vae.encode(inputs).latent_dist.mode()
values["output"] = result.transpose(1, 2).contiguous().cpu().numpy().reshape(-1)
host = inputs.contiguous().cpu().numpy()
del vae, inputs, result
torch.cuda.empty_cache()
lib = ctypes.CDLL(a.library)
fp = ctypes.POINTER(ctypes.c_float)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_audio_encode.argtypes = [ctypes.c_char_p, ctypes.c_int, fp, ctypes.c_char_p, fp]
native = np.empty(((a.samples + 799) // 800) * 32, dtype=np.float32)
if lib.h3_audio_encode(
    a.model.encode(),
    a.samples,
    host.ctypes.data_as(fp),
    str(out).encode(),
    native.ctypes.data_as(fp),
):
    raise RuntimeError(lib.h3_error().decode())
report = {}
for name, reference in values.items():
    actual = (
        native
        if name == "output"
        else np.fromfile(out / f"{name}.f32", dtype=np.float32)
    )
    delta = actual.astype(np.float64) - reference
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(delta) / np.linalg.norm(reference.astype(np.float64))
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }
    reference.tofile(out / f"{name}.reference.f32")
native.tofile(out / "output.f32")
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert all(v["relative_l2"] < 1e-4 for v in report.values()), report
