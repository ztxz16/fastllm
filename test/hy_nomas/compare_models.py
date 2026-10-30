"""Compare complete native DiT/PixelUMM forwards with official modules.

Reference-only dependencies: torch, numpy, safetensors, einops. Production does
not import this file. Small grids deliberately keep validation cheap; this is
not a 720p quality, speed, streaming or camera-control acceptance test.
"""
import argparse
import ast
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from compare_block import official_symbols, metrics

DTYPE = torch.float32 if os.environ.get("FASTLLM_NOMAS_PRECISION") == "fp32" else torch.bfloat16


def selected(env, path, names):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)


def load(module, weights, prefix):
    state = {key[len(prefix):]: weights.get_tensor(key).to(device="cuda", dtype=DTYPE)
             for key in weights.keys() if key.startswith(prefix)}
    module.load_state_dict(state, strict=True, assign=True)
    return module.eval()


@torch.no_grad()
def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    os.environ.setdefault("FASTLLM_NOMAS_PRECISION", "bf16")
    lib = C.CDLL(str(args.library))
    fp = C.POINTER(C.c_float)
    lib.fastllm_nomas_error.restype = C.c_char_p
    for name in ["fastllm_nomas_create", "fastllm_nomas_decoder_create"]:
        getattr(lib, name).argtypes = [C.c_char_p, C.c_int]
        getattr(lib, name).restype = C.c_void_p
    lib.fastllm_nomas_destroy.argtypes = [C.c_void_p]
    lib.fastllm_nomas_decoder_destroy.argtypes = [C.c_void_p]
    lib.fastllm_nomas_dit.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, fp, fp, C.c_float, C.c_int, C.c_int, fp]
    lib.fastllm_nomas_decode.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, fp, fp, fp]
    lib.fastllm_nomas_decoder_block.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, fp, fp]
    ptr = lambda a: a.ctypes.data_as(fp)
    array = lambda x: x.float().cpu().contiguous().numpy()
    report = {"torch": torch.__version__, "precision": str(DTYPE), "attention_reference": "PyTorch SDPA math" if DTYPE == torch.float32 else "PyTorch SDPA", "cases": []}

    def record(name, output, expected, **metadata):
        case = dict(name=name, **metrics(output, array(expected)), **metadata)
        case["passed"] = case["finite"] and case["relative_l2"] <= (0.001 if DTYPE == torch.float32 else 0.02)
        report["cases"].append(case)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(case), flush=True)

    if args.component in ("all", "dit"):
        env = official_symbols(args.official)
        env.update(math=math, linear_dtype=lambda m: m.weight.dtype)
        base = args.official / "hymm/models"
        selected(env, base / "dit/modules/embed_layers.py", {"timestep_embedding", "TimestepEmbedder", "TextProjection"})
        selected(env, base / "text_encoder/token_refiner.py", {"IndividualTokenRefinerBlock", "IndividualTokenRefiner", "SingleTokenRefiner"})
        selected(env, base / "dit/modules/mlp_layers.py", {"FinalLayer"})

        def attention(q, k, v, **kwargs):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)

        def joint(q, k, v, **kwargs):
            ni = q[0].shape[1]
            a = attention(*[torch.cat(xs, 1) for xs in (q, k, v)])
            return a[:, :ni], a[:, ni:]

        env.update(attention=attention, causal_attention_with_cache_mp=joint)
        path = args.weights / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
        t, h, w, nt = 2, 2, 2, 17
        latent = torch.randn(1, t*h*w, 32, device="cuda", dtype=DTYPE)
        features = torch.randn(1, nt, 3584, device="cuda", dtype=DTYPE)
        timestep = torch.tensor([750.0], device="cuda")
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            refiner = load(env["SingleTokenRefiner"](3584, 2048, 16, 2, device="meta", dtype=DTYPE), weights, "txt_in.")
            txt = refiner(features, timestep, torch.ones(1, nt, device="cuda", dtype=torch.bool))
            time_in = load(env["TimestepEmbedder"](2048, torch.nn.SiLU, device="meta", dtype=DTYPE), weights, "time_in.")
            vec = time_in(timestep)
            x = F.linear(latent, weights.get_tensor("img_in.proj.weight").flatten(1).to("cuda", DTYPE), weights.get_tensor("img_in.proj.bias").to("cuda", DTYPE))
            positions = torch.meshgrid(*[torch.arange(n, device="cuda") for n in (t, h, w)], indexing="ij")
            freq = torch.cat([p.flatten()[:, None] * 256 ** (-torch.arange(0, d, 2, device="cuda").float()/d)
                              for p, d in zip(positions, [16, 56, 56])], -1).repeat_interleave(2, -1)
            for i in range(54):
                block = load(env["MMDoubleStreamBlock"](2048, 16, 4, qkv_bias=True, device="meta", dtype=DTYPE), weights, f"double_blocks.{i}.")
                x, txt = block.forward_with_cache(x, txt, vec, freqs_cis=(freq.cos(), freq.sin()), kv_cache={})
                del block
                if i % 18 == 17:
                    print(f"official DiT {i+1}/54", flush=True)
            final = load(env["FinalLayer"](2048, [1, 1, 1], 32, torch.nn.SiLU, device="meta", dtype=DTYPE), weights, "final_layer.")
            expected = final(x, vec)
        handle = lib.fastllm_nomas_create(str(path).encode(), 0)
        if not handle:
            raise RuntimeError(lib.fastllm_nomas_error().decode())
        try:
            output = np.empty((t*h*w, 32), np.float32)
            a, b = array(latent), array(features)
            start = time.monotonic()
            rc = lib.fastllm_nomas_dit(handle, t, h, w, nt, ptr(a), ptr(b), 750.0, 0, 0, ptr(output))
            if rc:
                raise RuntimeError(lib.fastllm_nomas_error().decode())
            record("dit_54_layers", output, expected, native_seconds_including_load=time.monotonic()-start)
        finally:
            lib.fastllm_nomas_destroy(handle)
        del refiner, time_in, final
        torch.cuda.empty_cache()

    if args.component in ("all", "decoder"):
        file = args.official / "hymm/models/vae/pixelumm_decoder/model.py"
        spec = importlib.util.spec_from_file_location("pixel_reference", file)
        pixel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pixel
        spec.loader.exec_module(pixel)
        path = args.weights / "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
        handle = lib.fastllm_nomas_decoder_create(str(path).encode(), 0)
        if not handle:
            raise RuntimeError(lib.fastllm_nomas_error().decode())
        try:
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                # A >3-tile plane exercises the shifted-border local attention.
                for layer, t, h, w, tile in [(0, 1, 4, 4, 0), (35, 1, 5, 7, 1)]:
                    with torch.device("meta"):
                        block = pixel.GenLayer(pixel.QWEN3_8B_GEN)
                    block = load(block, weights, f"decoder.layers.{layer}.")
                    inp = torch.randn(1, t*h*w, 4096, device="cuda", dtype=DTYPE)
                    cs, sn = pixel.mrope_tables(pixel.QWEN3_8B_GEN, (t, h, w), "cuda")
                    if tile:
                        ids = torch.arange(t*h*w, device="cuda")
                        qh, qw = (ids//w%h//tile).clamp(1, max((h+tile-1)//tile-2, 1)), (ids%w//tile).clamp(1, max((w+tile-1)//tile-2, 1))
                        mask = ((qh[:, None]-ids[None, :]//w%h//tile).abs()<=1) & ((qw[:, None]-ids[None, :]%w//tile).abs()<=1)
                        pixel._flex_attention = lambda q, k, v, _: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                    expected = block(inp, cs, sn, object() if tile else None)
                    a, output = array(inp), np.empty((t*h*w, 4096), np.float32)
                    rc = lib.fastllm_nomas_decoder_block(handle, layer, t, h, w, tile, ptr(a), ptr(output))
                    if rc:
                        raise RuntimeError(lib.fastllm_nomas_error().decode())
                    record(f"decoder_block_{layer}", output, expected, grid=[t, h, w], tile=tile)
                    del block
                with torch.device("meta"):
                    model = pixel.DCAEVPixelUMMDecoder(local_attention_tile=8)
                model = load(model, weights, "decoder.")
                t, h, w = 1, 2, 2
                latent = torch.randn(1, 32, t, h, w, device="cuda", dtype=DTYPE)
                noise = torch.randn(1, 3, t*4, h*32, w*32, device="cuda", dtype=torch.float32)
                expected = pixel.patchify(model(latent, noise))
                a = array(latent.permute(0, 2, 3, 4, 1))
                b = array(pixel.patchify(noise))
                output = np.empty((t*h*w*4, 3072), np.float32)
                rc = lib.fastllm_nomas_decode(handle, t, h, w, ptr(a), ptr(b), ptr(output))
                if rc:
                    raise RuntimeError(lib.fastllm_nomas_error().decode())
                record("decoder_36_layers", output, expected)
        finally:
            lib.fastllm_nomas_decoder_destroy(handle)
    report["passed"] = all(c["passed"] for c in report["cases"])
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("Model comparison failed; see report")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--component", choices=["all", "dit", "decoder"], default="all")
    main(p.parse_args())
