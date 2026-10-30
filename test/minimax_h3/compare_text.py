"""Load only the official 50-layer text prefix, capture the residual before final norm."""

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
p.add_argument("--model", default="/model")
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
p.add_argument("--prompt", default="A cat plays piano on a wooden stage. The audience applauds.")
a = p.parse_args()
root = Path(a.model)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((root / "text_encoder/config.json").read_text())["text_config"]
cfg["num_hidden_layers"] = 50
cfg["_attn_implementation"] = "sdpa"
dtype = torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
with torch.device("meta"):
    encoder = Qwen3VLTextModel(Qwen3VLTextConfig(**cfg))
index = json.loads((root / "text_encoder/model.safetensors.index.json").read_text())["weight_map"]


# FP32 diagnostics stream one layer at a time to stay within the single GPU.
def load(module, prefix):
    state = {}
    for name in module.state_dict():
        key = "model.language_model." + prefix + name
        with safe_open(str(root / "text_encoder" / index[key]), framework="pt", device="cpu") as f:
            state[name] = f.get_tensor(key).to(device="cuda", dtype=dtype)
    module.load_state_dict(state, assign=True)


load(encoder.embed_tokens, "embed_tokens.")
load(encoder.norm, "norm.")
values = {}
for i, layer in enumerate(encoder.layers):
    layer.register_forward_pre_hook(lambda m, args, i=i: load(m, f"layers.{i}."))

    def capture(m, args, result, i=i):
        values[f"layer.{i}"] = result.detach().float().cpu().numpy().reshape(-1)
        m.to_empty(device="meta")

    layer.register_forward_hook(capture)
# RoPE frequencies are derived buffers, not checkpoint state.
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

encoder.rotary_emb = Qwen3VLTextRotaryEmbedding(encoder.config).cuda()
encoder.eval()
tokens = AutoTokenizer.from_pretrained(root / "tokenizer").encode(a.prompt, add_special_tokens=False)
with torch.inference_mode():
    encoder(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False)
reference = values["layer.49"]
del encoder
torch.cuda.empty_cache()
lib = ctypes.CDLL(a.library)
fp = ctypes.POINTER(ctypes.c_float)
ip = ctypes.POINTER(ctypes.c_int)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_text_encode.argtypes = [ctypes.c_char_p, ip, ctypes.c_int, ctypes.c_char_p, fp]
host = np.array(tokens, dtype=np.int32)
native = np.empty_like(reference)
if lib.h3_text_encode(
    str(root / "text_encoder").encode(),
    host.ctypes.data_as(ip),
    len(tokens),
    str(out).encode(),
    native.ctypes.data_as(fp),
):
    raise RuntimeError(lib.h3_error().decode())
report = {}
for name, ref in values.items():
    actual = np.fromfile(out / f"{name}.f32", dtype=np.float32)
    diff = actual.astype(np.float64) - ref
    report[name] = {
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref.astype(np.float64))),
        "max_abs": float(np.max(np.abs(diff))),
    }
    ref.tofile(out / f"{name}.reference.f32")
(out / "tokens.json").write_text(json.dumps(tokens))
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report["layer.49"]["relative_l2"] < 1e-5, report
