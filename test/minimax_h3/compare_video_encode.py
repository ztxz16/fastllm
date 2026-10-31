"""Compare native FP32 video/image VAE encoding to the released Diffusers model."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"
p = argparse.ArgumentParser()
p.add_argument("--model", default="/model/vae")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument("--frames", type=int, default=1)
p.add_argument("--height", type=int, default=64)
p.add_argument("--width", type=int, default=64)
p.add_argument("--tiled", action="store_true")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(205)
vae = AutoencoderKLMiniMaxH3.from_pretrained(a.model, torch_dtype=torch.float32).eval()
# Only the encoder is exercised; keep the large decoder off the GPU.
vae.encoder.cuda()
vae.quant_conv.cuda()
x = torch.randn(1, 3, a.frames, a.height, a.width, device="cuda")
values = {}


def capture(name):
    def hook(module, args, result):
        values[name] = (
            result.float().permute(0, 2, 3, 4, 1).contiguous().cpu().numpy().reshape(-1)
        )

    return hook


if not a.tiled:
    for name, module in vae.named_modules():
        if name in ("encoder.conv_in", "encoder.conv_out", "quant_conv") or (
            name.startswith("encoder.down_blocks.")
            and name.count(".") in (2, 4)
            and ".downsamplers" not in name
        ):
            module.register_forward_hook(capture(name))
with torch.inference_mode():
    y = vae._encode(x) if a.tiled else vae.quant_conv(vae.encoder(x))
reference = y.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy().reshape(-1)
host = x.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
del vae, x, y
torch.cuda.empty_cache()
lib = ctypes.CDLL(a.library)
fp = ctypes.POINTER(ctypes.c_float)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_video_encode.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    fp,
    ctypes.c_char_p,
    fp,
    ctypes.c_int,
]
actual = np.empty_like(reference)
if lib.h3_video_encode(
    a.model.encode(),
    a.frames,
    a.height,
    a.width,
    host.ctypes.data_as(fp),
    str(out).encode(),
    actual.ctypes.data_as(fp),
    a.tiled,
):
    raise RuntimeError(lib.h3_error().decode())
values["output"] = reference
report = {}
for name, expected in values.items():
    got = (
        actual
        if name == "output"
        else np.fromfile(out / f"{name}.f32", dtype=np.float32)
    )
    delta = got.astype(np.float64) - expected
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }
    expected.tofile(out / f"{name}.reference.f32")
actual.tofile(out / "output.f32")
(out / "report.json").write_text(json.dumps(report, indent=2))
(out / "config.json").write_text(json.dumps(vars(a), indent=2))
print(json.dumps(report, indent=2), flush=True)
assert np.isfinite(actual).all()
assert all(v["relative_l2"] < 1e-4 for v in report.values()), report
