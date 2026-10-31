"""Full 50-layer H3 differential test with a streamed reference in either precision."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import MiniMaxH3Transformer3DModel
from safetensors import safe_open

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model/transformer")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
p.add_argument(
    "--isolated-blocks",
    action="store_true",
    help="Validate each block on its reference input; report accumulated full-DiT error separately",
)
a = p.parse_args()
fp32 = a.precision == "fp32"
root = Path(a.model)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((root / "config.json").read_text())
index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
dtype = torch.float32 if fp32 else torch.bfloat16
with torch.device("meta"):
    model = MiniMaxH3Transformer3DModel.from_config(cfg)


def load(module, prefix):
    state = {}
    for name in module.state_dict():
        key = prefix + name
        with safe_open(str(root / index[key]), framework="pt", device="cpu") as f:
            target = (
                torch.float32
                if any(s in key for s in MiniMaxH3Transformer3DModel._keep_in_fp32_modules)
                else dtype
            )
            state[name] = f.get_tensor(key).to(device="cuda", dtype=target)
    module.load_state_dict(state, assign=True)


values = {}
block_inputs = {}
for name, child in model.named_children():
    if name in ("rope", "time_proj"):
        continue
    if name == "transformer_blocks":
        for i, block in enumerate(child):
            block.register_forward_pre_hook(lambda m, args, i=i: load(m, f"transformer_blocks.{i}."))

            def hook(m, ins, result, i=i):
                values[f"block.{i}"] = result.detach().float().cpu().numpy().reshape(-1)
                if a.isolated_blocks:
                    block_inputs[i] = ins[0].detach().float().cpu().numpy().reshape(-1)
                m.to_empty(device="meta")

            block.register_forward_hook(hook)
    else:
        load(child, name + ".")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

model.rope = MiniMaxH3RotaryPosEmbed(cfg["rope_freq_dim"], cfg["rope_theta"]).cuda()
model.eval()
torch.manual_seed(888)
nt, na, nv = 11, 16, 40
n = nt + na + nv
text = torch.randn(1, nt, 5120, device="cuda").to(dtype) * 0.1
audio = torch.randn(1, na, 32, device="cuda")
video = torch.randn(1, nv, 96, device="cuda")
positions = torch.randn(n, 3, device="cuda") * 20
times = torch.tensor([0.0, 0.4, 0.8], device="cuda")
indices = torch.arange(n, device="cuda") % 3
tags = torch.tensor([1] * nt + [2] * na + [0] * nv, device="cuda")
with torch.inference_mode():
    result = model(
        video,
        audio,
        text,
        times,
        indices,
        tags,
        positions,
        torch.arange(nt + na, n, device="cuda"),
        torch.arange(nt, nt + na, device="cuda"),
        torch.arange(nt, device="cuda"),
    )
refs = {
    "video": result.sample.float().cpu().numpy().reshape(-1),
    "audio": result.audio_sample.float().cpu().numpy().reshape(-1),
}


def array(t):
    return t.float().cpu().numpy().astype(np.float32).reshape(-1)


inputs = [
    array(text),
    array(audio),
    array(video),
    array(times),
    indices.int().cpu().numpy(),
    array(positions),
]
del model, text, audio, video, result, child, block
import gc

gc.collect()
torch.cuda.empty_cache()
lib = ctypes.CDLL(a.library)
fp = ctypes.POINTER(ctypes.c_float)
ip = ctypes.POINTER(ctypes.c_int)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_transformer.argtypes = (
    [ctypes.c_char_p] + [ctypes.c_int] * 4 + [fp] * 4 + [ip, fp, ctypes.c_char_p, fp, fp, ctypes.c_int]
)
v = np.empty_like(refs["video"])
audio = np.empty_like(refs["audio"])
args = [t.ctypes.data_as(ip if i == 4 else fp) for i, t in enumerate(inputs)]
if lib.h3_transformer(
    str(root).encode(),
    nt,
    na,
    nv,
    3,
    *args,
    str(out).encode(),
    v.ctypes.data_as(fp),
    audio.ctypes.data_as(fp),
    fp32,
):
    raise RuntimeError(lib.h3_error().decode())
report = {}
for name, ref in {**values, **refs}.items():
    actual = (
        v
        if name == "video"
        else audio
        if name == "audio"
        else np.fromfile(out / f"{name}.f32", dtype=np.float32)
    )
    diff = actual.astype(np.float64) - ref
    report[name] = {
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref.astype(np.float64))),
        "max_abs": float(np.max(np.abs(diff))),
    }
    ref.tofile(out / f"{name}.reference.f32")
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
limit = 1e-4 if fp32 else 1e-2
if a.isolated_blocks:
    lib.h3_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.h3_create.restype = ctypes.c_void_p
    lib.h3_destroy.argtypes = [ctypes.c_void_p]
    lib.h3_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        fp,
        fp,
        ip,
        fp,
        ctypes.c_char_p,
        fp,
    ]
    handle = lib.h3_create(str(root).encode(), int(fp32))
    if not handle:
        raise RuntimeError(lib.h3_error().decode())
    mod_indices = inputs[4] * 3 + np.array([1] * nt + [2] * na + [0] * nv, dtype=np.int32)
    isolated = {}
    try:
        for i, x in block_inputs.items():
            result = np.empty_like(x)
            code = lib.h3_block(
                handle,
                i,
                n,
                3,
                x.ctypes.data_as(fp),
                inputs[3].ctypes.data_as(fp),
                mod_indices.ctypes.data_as(ip),
                inputs[5].ctypes.data_as(fp),
                None,
                result.ctypes.data_as(fp),
            )
            if code:
                raise RuntimeError(lib.h3_error().decode())
            expected = values[f"block.{i}"]
            isolated[f"block.{i}"] = float(
                np.linalg.norm(result.astype(float) - expected) / np.linalg.norm(expected.astype(float))
            )
    finally:
        lib.h3_destroy(handle)
    (out / "isolated-blocks.json").write_text(json.dumps(isolated, indent=2))
    print("Isolated block relative L2:", isolated)
    assert all(x < limit for x in isolated.values()), isolated
    print(
        "PASS: isolated blocks; accumulated full-DiT errors above are diagnostic, not a full-DiT precision pass"
    )
else:
    assert all(report[name]["relative_l2"] < limit for name in ("video", "audio")), report
