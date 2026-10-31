"""Compare image-conditioned Qwen3-VL features, including MRoPE and DeepStack."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3VLTextConfig, Qwen3VLTextModel

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"

p = argparse.ArgumentParser()
p.add_argument("--copies", type=int, choices=(1, 2), default=1)
p.add_argument("--vision-fixture", required=True)
p.add_argument("--model", default="/model")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--prompt", default="A cat plays piano on a wooden stage. The audience applauds."
)
p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
a = p.parse_args()
fp32 = a.precision == "fp32"
root = Path(a.model)
vision_dir = Path(a.vision_fixture)
vision_config = json.loads((vision_dir / "config.json").read_text())
assert vision_config["precision"] == a.precision
height, width = vision_config["height"], vision_config["width"]
video_input = vision_config.get("video", False)
pad_token = 151656 if video_input else 151655
features = [
    torch.from_numpy(
        np.fromfile(
            vision_dir / f"vision.output.{i}.reference.f32", dtype=np.float32
        ).reshape(-1, 5120)
    )
    .repeat(a.copies, 1)
    .cuda()
    .to(dtype=torch.float32 if fp32 else torch.bfloat16)
    for i in range(4)
]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((root / "text_encoder/config.json").read_text())["text_config"]
cfg["num_hidden_layers"] = 50
cfg["_attn_implementation"] = "sdpa"
dtype = torch.float32 if fp32 else torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
with torch.device("meta"):
    encoder = Qwen3VLTextModel(Qwen3VLTextConfig(**cfg))
index = json.loads((root / "text_encoder/model.safetensors.index.json").read_text())[
    "weight_map"
]


# Stream the reference one layer at a time for both precisions.
def load(module, prefix):
    state = {}
    for name in module.state_dict():
        key = "model.language_model." + prefix + name
        with safe_open(
            str(root / "text_encoder" / index[key]), framework="pt", device="cpu"
        ) as f:
            state[name] = f.get_tensor(key).to(device="cuda", dtype=dtype)
    module.load_state_dict(state, assign=True)


load(encoder.embed_tokens, "embed_tokens.")
load(encoder.norm, "norm.")
values = {}
for i, layer in enumerate(encoder.layers):
    layer.register_forward_pre_hook(lambda m, args, i=i: load(m, f"layers.{i}."))

    def capture(m, args, result, i=i):
        value = result.detach().clone()
        if i < 3:
            value[visual_mask] += features[i + 1]
        values[f"layer.{i}"] = value.float().cpu().numpy().reshape(-1)
        m.to_empty(device="meta")

    layer.register_forward_hook(capture)
# RoPE frequencies are derived buffers, not checkpoint state.
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

encoder.rotary_emb = Qwen3VLTextRotaryEmbedding(encoder.config).cuda()
encoder.eval()
tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
tokens = []
for i in range(a.copies):
    tokens += (
        tokenizer.encode(
            f"<Video {i + 1}>: <0.2 seconds>"
            if video_input
            else f"<Picture {i + 1}>: ",
            add_special_tokens=False,
        )
        + [151652]
        + [pad_token] * (height // 32 * (width // 32))
        + [151653]
    )
tokens += tokenizer.encode(a.prompt, add_special_tokens=False)
input_ids = torch.tensor([tokens], device="cuda")
visual_mask = input_ids == pad_token
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from types import SimpleNamespace, MethodType

stub = SimpleNamespace(config=Qwen3VLConfig.from_pretrained(root / "text_encoder"))
stub.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, stub)
pos, _ = Qwen3VLModel.get_rope_index(
    stub,
    input_ids,
    visual_mask.int() * (2 if video_input else 1),
    **{
        ("video_grid_thw" if video_input else "image_grid_thw"): torch.tensor(
            [[1, height // 16, width // 16]] * a.copies, device="cuda"
        )
    },
)
with torch.inference_mode():
    embeds = encoder.embed_tokens(input_ids)
    embeds[visual_mask] = features[0]
    encoder(
        inputs_embeds=embeds,
        position_ids=pos,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=features[1:],
        use_cache=False,
    )

reference = values["layer.49"]
del encoder
torch.cuda.empty_cache()
lib = ctypes.CDLL(a.library)
fp = ctypes.POINTER(ctypes.c_float)
ip = ctypes.POINTER(ctypes.c_int)
lib.h3_error.restype = ctypes.c_char_p
bp = ctypes.POINTER(ctypes.c_uint8)
lib.h3_image_text_encode.argtypes = [
    ctypes.c_char_p,
    ip,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    bp,
    ctypes.c_char_p,
    fp,
    ctypes.c_int,
    ctypes.c_int,
    bp,
]
rgb = (
    np.fromfile(vision_dir / "input.rgb", dtype=np.uint8).reshape(height, width, 3)
    if (vision_dir / "input.rgb").exists()
    else np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
)
rgba = np.concatenate((rgb, np.full((height, width, 1), 255, dtype=np.uint8)), axis=2)
second = np.roll(rgba, 1, axis=1).copy()
host = np.array(tokens, dtype=np.int32)
native = np.empty_like(reference)
if lib.h3_image_text_encode(
    str(root / "text_encoder").encode(),
    host.ctypes.data_as(ip),
    len(tokens),
    height,
    width,
    rgba.ctypes.data_as(bp),
    str(out).encode(),
    native.ctypes.data_as(fp),
    fp32,
    a.copies,
    second.ctypes.data_as(bp) if video_input else None,
):
    raise RuntimeError(lib.h3_error().decode())
report = {}
for name, ref in values.items():
    actual = np.fromfile(out / f"{name}.f32", dtype=np.float32)
    diff = actual.astype(np.float64) - ref
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(diff) / np.linalg.norm(ref.astype(np.float64))
        ),
        "max_abs": float(np.max(np.abs(diff))),
    }
    ref.tofile(out / f"{name}.reference.f32")
(out / "tokens.json").write_text(json.dumps(tokens))
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report["layer.49"]["relative_l2"] < (1e-4 if fp32 else 1e-2), report
