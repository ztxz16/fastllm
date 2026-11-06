"""Differential checks for the shared Qwen3-VL visual tower using H3 weights."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from transformers import AutoImageProcessor, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
p = argparse.ArgumentParser()
p.add_argument("--image")
p.add_argument("--video", action="store_true")
p.add_argument("--isolated", action="store_true")
p.add_argument("--model", default="/model")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument("--height", type=int, default=256)
p.add_argument("--width", type=int, default=256)
p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
a = p.parse_args()
root, out = Path(a.model), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
dtype = torch.float32 if a.precision == "fp32" else torch.bfloat16
config = json.loads((root / "text_encoder/config.json").read_text())["vision_config"]
config["_attn_implementation"] = "sdpa"
with torch.device("meta"):
    vision = Qwen3VLVisionModel(Qwen3VLVisionConfig(**config))
index = json.loads((root / "text_encoder/model.safetensors.index.json").read_text())[
    "weight_map"
]
state = {}
for name in vision.state_dict():
    key = "model.visual." + name
    with safe_open(str(root / "text_encoder" / index[key]), framework="pt") as f:
        state[name] = f.get_tensor(key).cuda().to(dtype)
vision.load_state_dict(state, assign=True)
del state
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding

vision.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(vision.config).cuda()
vision.eval()
rgb = np.random.default_rng(42).integers(0, 256, (a.height, a.width, 3), dtype=np.uint8)
if a.image:
    rgb = np.asarray(
        Image.open(a.image)
        .convert("RGB")
        .resize((a.width, a.height), Image.Resampling.BICUBIC)
    )
rgb.tofile(out / "input.rgb")
rgba = np.concatenate(
    (rgb, np.full((a.height, a.width, 1), 255, dtype=np.uint8)), axis=2
)
if a.video:
    from transformers import AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained(root / "processor")
    second = np.roll(rgba, 1, axis=1).copy()
    inputs = processor(
        videos=[np.stack((rgb, second[:, :, :3]))],
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    pixels, grid = inputs.pixel_values_videos, inputs.video_grid_thw
else:
    processor = AutoImageProcessor.from_pretrained(root / "processor")
    inputs = processor(
        images=[Image.fromarray(rgb)], return_tensors="pt", do_resize=False
    )
    pixels, grid = inputs.pixel_values, inputs.image_grid_thw
values = {}
vision.patch_embed.register_forward_hook(
    lambda m, args, result: values.update(
        {"vision.patch": result.float().cpu().numpy().reshape(-1)}
    )
)
vision.blocks[0].register_forward_pre_hook(
    lambda m, args: values.update(
        {"vision.embedding": args[0].float().cpu().numpy().reshape(-1)}
    )
)
vision.blocks[0].norm1.register_forward_hook(
    lambda m, args, result: values.update(
        {"vision.norm1": result.float().cpu().numpy().reshape(-1)}
    )
)
vision.blocks[0].attn.qkv.register_forward_hook(
    lambda m, args, result: values.update(
        {"vision.qkv": result.float().cpu().numpy().reshape(-1)}
    )
)
for label, module in [
    ("vision.projected", vision.blocks[0].attn.proj),
    ("vision.norm2", vision.blocks[0].norm2),
    ("vision.fc1", vision.blocks[0].mlp.linear_fc1),
    ("vision.fc2", vision.blocks[0].mlp.linear_fc2),
]:
    module.register_forward_hook(
        lambda m, args, result, label=label: values.update(
            {label: result.float().cpu().numpy().reshape(-1)}
        )
    )
vision.blocks[0].attn.proj.register_forward_pre_hook(
    lambda m, args: values.update(
        {"vision.attention": args[0].float().cpu().numpy().reshape(-1)}
    )
)
vision.blocks[0].norm2.register_forward_pre_hook(
    lambda m, args: values.update(
        {"vision.residual": args[0].float().cpu().numpy().reshape(-1)}
    )
)
vision.blocks[0].mlp.linear_fc2.register_forward_pre_hook(
    lambda m, args: values.update(
        {"vision.gelu": args[0].float().cpu().numpy().reshape(-1)}
    )
)
for i, block in enumerate(vision.blocks):
    block.register_forward_pre_hook(
        lambda m, args, i=i: values.update(
            {f"vision.input.{i}": args[0].float().cpu().numpy().reshape(-1)}
        )
    )
    block.register_forward_hook(
        lambda m, args, result, i=i: values.update(
            {f"vision.block.{i}": result.float().cpu().numpy().reshape(-1)}
        )
    )
with torch.inference_mode():
    result = vision(pixels.cuda().to(dtype), grid_thw=grid.cuda())
for i, tensor in enumerate([result.pooler_output, *result.deepstack_features]):
    values[f"vision.output.{i}"] = tensor.float().cpu().numpy().reshape(-1)
del vision, result, tensor
torch.cuda.empty_cache()
for name, expected in values.items():
    expected.tofile(out / f"{name}.reference.f32")
lib = ctypes.CDLL(a.library)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_vision_encode.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
]
if lib.h3_vision_encode(
    str(root / "text_encoder").encode(),
    a.height,
    a.width,
    rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    str(out).encode(),
    a.precision == "fp32",
    a.isolated,
    second.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if a.video else None,
):
    raise RuntimeError(lib.h3_error().decode())
report = {}
for name, expected in values.items():
    actual = np.fromfile(out / f"{name}.f32", dtype=np.float32)
    delta = actual.astype(np.float64) - expected
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }
    expected.tofile(out / f"{name}.reference.f32")
    assert np.isfinite(actual).all()
(out / "report.json").write_text(json.dumps(report, indent=2))
(out / "config.json").write_text(json.dumps(vars(a), indent=2))
print(json.dumps(report, indent=2), flush=True)
tolerance = 1e-4 if a.precision == "fp32" else 2e-2
assert all(v["relative_l2"] < tolerance for v in report.values()), report
