"""GPU-only reference test; torch is intentionally confined to test/.

Executes the official MMDoubleStreamBlock.forward_with_cache source unchanged.
Only the distributed attention adapter is replaced with PyTorch SDPA (FLASH
without a mask, FP32 math with a mask, which FLASH does not support).
No diffusers, magi-attention or project installation is needed. The official
checkout and its commit hash are recorded in the output for reproducibility.
"""
import argparse
import ast
import ctypes as C
import functools
import json
import os
from pathlib import Path
import time
import typing

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from safetensors import safe_open


def official_symbols(root):
    env = dict(torch=torch, nn=nn, F=F, rearrange=rearrange,
               partial=functools.partial, to_2tuple=lambda x: (x, x), **vars(typing))
    modules = {
        "dit/modules/activation_layers.py": {"get_activation_layer"},
        "dit/modules/norm_layers.py": {"RMSNorm", "get_norm_layer"},
        "dit/modules/modulate_layers.py": {"ModulateDiT", "modulate", "apply_gate"},
        "dit/modules/mlp_layers.py": {"MLP"},
        "dit/modules/posemb_layers.py": {"reshape_for_broadcast", "rotate_half", "apply_rotary_emb"},
        "dit/nomas_cpt.py": {"MMDoubleStreamBlock"},
    }
    for path, names in modules.items():
        file = root / "hymm/models" / path
        parsed = ast.parse(file.read_text())
        nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        # Disable evaluated annotations without altering function bodies.
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(file), "exec"), env)
    return env


def metrics(actual, expected):
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = a - b
    return dict(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta ** 2))),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
                cosine=float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
                finite=bool(np.isfinite(a).all()))


def run(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    env = official_symbols(args.official)
    os.environ.setdefault("FASTLLM_NOMAS_PRECISION", "bf16")
    lib = C.CDLL(str(args.library))
    fp = C.POINTER(C.c_float)
    bp = C.POINTER(C.c_uint8)
    lib.fastllm_nomas_create.argtypes = [C.c_char_p, C.c_int]
    lib.fastllm_nomas_create.restype = C.c_void_p
    lib.fastllm_nomas_destroy.argtypes = [C.c_void_p]
    lib.fastllm_nomas_error.restype = C.c_char_p
    lib.fastllm_nomas_block.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, fp, fp, fp, fp, fp, C.c_int, fp, fp, bp, fp, fp, fp, fp]
    lib.fastllm_nomas_block.restype = C.c_int
    ptr = lambda x: None if x is None else x.ctypes.data_as(fp)
    handle = lib.fastllm_nomas_create(str(args.checkpoint).encode(), 0)
    if not handle:
        raise RuntimeError(lib.fastllm_nomas_error().decode())
    git_head = (args.official / ".git/HEAD").read_text().strip()
    if git_head.startswith("ref: "):
        git_head = (args.official / ".git" / git_head[5:]).read_text().strip()
    report = {"official_commit": git_head,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "cases": []}
    try:
        with safe_open(str(args.checkpoint), framework="pt", device="cpu") as weights:
            for layer, ni, nt, nh, masked in [(0, 32, 11, 0, False), (27, 65, 17, 23, True), (53, 8, 7, 13, True)]:
                prefix = f"double_blocks.{layer}."
                model = env["MMDoubleStreamBlock"](2048, 16, 4.0, qkv_bias=True, dtype=torch.bfloat16, device="meta")
                state = {k[len(prefix):]: weights.get_tensor(k).to(device="cuda", dtype=torch.bfloat16)
                         for k in weights.keys() if k.startswith(prefix)}
                model.load_state_dict(state, strict=True, assign=True)
                model.eval()
                img = torch.randn(1, ni, 2048, device="cuda", dtype=torch.bfloat16)
                txt = torch.randn(1, nt, 2048, device="cuda", dtype=torch.bfloat16)
                vec = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
                # Real 3D RoPE widths and theta, including a temporal offset.
                positions = torch.arange(ni, device="cuda", dtype=torch.float32)
                freqs = torch.cat([((positions // 16 + 5)[:, None] * 256 ** (-torch.arange(0, 16, 2, device="cuda").float() / 16)),
                                   ((positions // 4 % 4)[:, None] * 256 ** (-torch.arange(0, 56, 2, device="cuda").float() / 56)),
                                   ((positions % 4)[:, None] * 256 ** (-torch.arange(0, 56, 2, device="cuda").float() / 56))], -1).repeat_interleave(2, -1)
                cs, sn = freqs.cos(), freqs.sin()
                hk = torch.randn(1, nh, 16, 128, device="cuda", dtype=torch.bfloat16) if nh else None
                hv = torch.randn_like(hk) if nh else None
                mask = torch.ones(ni+nt, nh+ni+nt, device="cuda", dtype=torch.bool)
                if masked:
                    mask[:, :3] = False                 # cached hidden text / padding
                    mask[:ni//2, nh+ni+nt//2:] = False # time-dependent action span
                    mask[-1, :] = False               # all-masked row must be finite zero
                capture = {}

                def reference_attention(qs, ks, vs, **kwargs):
                    q, k, v = [torch.cat(xs, dim=1) for xs in (qs, ks, vs)]
                    capture.update(key=k, value=v)
                    if nh:
                        k, v = torch.cat([hk, k], 1), torch.cat([hv, v], 1)
                    backend = torch.nn.attention.SDPBackend.MATH if masked else torch.nn.attention.SDPBackend.FLASH_ATTENTION
                    with torch.nn.attention.sdpa_kernel(backend):
                        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                             attn_mask=mask[None, None] if masked else None)
                    out = out.transpose(1, 2).flatten(2)
                    return out[:, :ni], out[:, ni:]

                env["causal_attention_with_cache_mp"] = reference_attention
                with torch.no_grad():
                    ref_i, ref_t = model.forward_with_cache(img, txt, vec, freqs_cis=(cs, sn), kv_cache={})
                to_np = lambda x: None if x is None else x.detach().float().cpu().contiguous().numpy()
                inputs = list(map(to_np, [img, txt, vec, cs, sn, hk, hv]))
                mask_np = mask.cpu().numpy().astype(np.uint8) if masked else None
                outputs = [np.empty((ni, 2048), np.float32), np.empty((nt, 2048), np.float32),
                           np.empty((ni+nt, 2048), np.float32), np.empty((ni+nt, 2048), np.float32)]
                start = time.monotonic()
                rc = lib.fastllm_nomas_block(handle, layer, ni, nt, *map(ptr, inputs[:5]), nh, *map(ptr, inputs[5:]),
                                             None if mask_np is None else mask_np.ctypes.data_as(bp), *map(ptr, outputs))
                if rc:
                    raise RuntimeError(lib.fastllm_nomas_error().decode())
                case = dict(layer=layer, image_tokens=ni, text_tokens=nt, history_tokens=nh,
                            masked=masked, native_seconds_including_first_weight_load=time.monotonic()-start)
                for name, actual, expected in zip(["image", "text", "key", "value"], outputs, [ref_i, ref_t, capture["key"], capture["value"]]):
                    case[name] = metrics(actual, to_np(expected))
                case["passed"] = all(case[name]["finite"] and case[name]["relative_l2"] <= 0.01
                                     for name in ["image", "text", "key", "value"])
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                del state, model
        report["passed"] = all(c["passed"] for c in report["cases"])
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise SystemExit("Block comparison failed; see report")
    finally:
        lib.fastllm_nomas_destroy(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
