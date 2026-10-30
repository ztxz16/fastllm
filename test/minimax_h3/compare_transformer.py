"""Developer-only differential check; PyTorch is never a native inference dependency."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)
from safetensors import safe_open


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
    assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model/transformer")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=67)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attention-backend", choices=["auto", "math"], default="auto")
    args = parser.parse_args()
    fp32 = args.precision == "fp32"
    if args.attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    dtype = torch.float32 if fp32 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(1701)

    def load(module, prefix, target_dtype):
        state = {}
        for name in module.state_dict():
            key = prefix + name
            with safe_open(str(root / index[key]), framework="pt", device="cpu") as handle:
                state[name] = handle.get_tensor(key).to(device="cuda", dtype=target_dtype)
        module.load_state_dict(state, assign=True)
        return module.eval()

    with torch.device("meta"):
        block = MiniMaxH3TransformerBlock(
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["attention_head_dim"],
            cfg["ffn_dim"],
            cfg["time_embed_dim"],
            cfg["norm_eps"],
            cfg["qk_norm_eps"],
        )
        time_mlp = TimestepEmbedding(
            cfg["freq_dim"], cfg["time_embed_hidden_dim"], out_dim=cfg["time_embed_dim"]
        )
    block = load(block, f"transformer_blocks.{args.layer}.", dtype)
    time_mlp = load(time_mlp, "time_embedder.", torch.float32)
    time_proj = Timesteps(cfg["freq_dim"], True, 0)
    positions = torch.randn(args.rows, 3, device="cuda") * 15
    times = torch.tensor([0.0, 0.43, 1.0], device="cuda")
    indices = torch.arange(args.rows, device="cuda") % 9
    x = (torch.randn(1, args.rows, cfg["hidden_size"], device="cuda") * 0.4).to(dtype)
    values = {}

    def keep(name, value):
        values[name] = value.detach().float().cpu().numpy()

    with torch.inference_mode():
        temb = time_mlp(time_proj(times))
        rope = MiniMaxH3RotaryPosEmbed(cfg["rope_freq_dim"], cfg["rope_theta"]).cuda()(positions)
        mods = block.adaln_proj(temb)
        keep("temb", temb)
        keep("mod", torch.cat(mods, dim=-1))
        keep("cos", rope[0])
        keep("sin", rope[1])
        norm1 = block.norm1(x)
        keep("norm1", norm1)
        mod1 = norm1 * (1 + mods[1][indices]) + mods[0][indices]
        keep("modulated1", mod1)
        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        q = block.attn.norm_q(block.attn.to_q(mod1).unflatten(-1, (cfg["num_attention_heads"], -1)))
        k = block.attn.norm_k(block.attn.to_k(mod1).unflatten(-1, (cfg["num_attention_heads"], -1)))
        v = block.attn.to_v(mod1).unflatten(-1, (cfg["num_attention_heads"], -1))
        q, k = _apply_rotary_emb(q, *rope), _apply_rotary_emb(k, *rope)
        keep("q", q)
        keep("k", k)
        keep("v", v)
        a = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            .transpose(1, 2)
            .contiguous()
            .flatten(2)
        )
        keep("attention", a)
        a = block.attn.to_out[0](a)
        keep("attn_out", a)
        res = x + mods[2][indices] * a
        keep("residual1", res)
        mod2 = block.norm2(res) * (1 + mods[4][indices]) + mods[3][indices]
        keep("modulated2", mod2)
        ff = block.ff(mod2)
        keep("ff_out", ff)
        keep("output", res + mods[5][indices] * ff)
        actual = block(x, temb, indices, rope)
        # This guards the hand-captured intermediate path against a changed upstream block.
        np.testing.assert_allclose(actual.float().cpu().numpy(), values["output"], rtol=0, atol=0)
    host_x = x.float().cpu().numpy().reshape(-1)
    host_times = times.cpu().numpy()
    host_positions = positions.cpu().numpy().reshape(-1)
    host_indices = indices.int().cpu().numpy()
    del block, time_mlp, x, actual, q, k, v, a, res, ff, mod1, mod2, norm1, mods, temb, rope
    torch.cuda.empty_cache()
    lib = ctypes.CDLL(args.library)
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.h3_error.restype = ctypes.c_char_p
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
    result = np.empty_like(host_x)
    try:
        code = lib.h3_block(
            handle,
            args.layer,
            args.rows,
            3,
            host_x.ctypes.data_as(fp),
            host_times.ctypes.data_as(fp),
            host_indices.ctypes.data_as(ip),
            host_positions.ctypes.data_as(fp),
            str(output).encode(),
            result.ctypes.data_as(fp),
        )
        if code:
            raise RuntimeError(lib.h3_error().decode())
    finally:
        lib.h3_destroy(handle)
    report = {}
    for name, reference in values.items():
        native = result if name == "output" else np.fromfile(output / f"{name}.f32", dtype=np.float32)
        reference = reference.reshape(-1)
        delta = native.astype(np.float64) - reference
        report[name] = {
            "relative_l2": float(
                np.linalg.norm(delta) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30)
            ),
            "max_abs": float(np.max(np.abs(delta))),
            "equal_fraction": float(np.mean(native == reference)),
        }
        reference.tofile(output / f"{name}.reference.f32")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["output"]["relative_l2"] < (1e-4 if fp32 else 1e-2), report


if __name__ == "__main__":
    main()
