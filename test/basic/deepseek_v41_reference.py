#!/usr/bin/env python3
"""
DeepSeek-V4.1 端到端数值对齐测试。

流程：
  1. 用官方 ``inference/model.py`` 的模块定义构造一个随机初始化的迷你 V4.1 模型
     （BF16 线性层、FP8 + UE8M0 的 Engram 表），保存为 HF 命名的 safetensors，
     并写出 fastllm 需要的 HF 风格 config.json 与 engram_meta.json；
  2. 用官方 model.py（把 tilelang kernel 换成纯 torch 实现）做 prefill + 贪心 decode，
     记录每步 logits；
  3. 用 fastllm 加载同一个目录，对同一段 token 做 prefill + decode，逐步比较 logits。

用法（在装有 torch / transformers / safetensors 的环境里）：
  PYTHONPATH=<fastllm>/build/tools python test/basic/deepseek_v41_reference.py \
      --work-dir /tmp/v41-tiny --tokenizer-dir /path/with/tokenizer.json \
      --reference-dir /path/to/DeepSeek-V4.1-Flash/inference [--prefill 60 --decode 6]
"""

import argparse
import ctypes
import importlib
import json
import math
import os
import shutil
import sys
import types

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="迷你模型输出目录")
    parser.add_argument("--tokenizer-dir", required=True, help="包含 tokenizer.json 的目录")
    parser.add_argument("--reference-dir", required=True, help="官方 inference/ 目录（model.py, engram.py）")
    parser.add_argument("--prefill", type=int, default=60)
    parser.add_argument("--decode", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16", help="fastllm 线性层 dtype")
    parser.add_argument("--moe-device", default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-cache-dtype", default="", help="fastllm KV 缓存存储类型（如 fp8_e4m3；默认 BF16）")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-fastllm", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--fastllm-only-load", action="store_true")
    parser.add_argument("--dump-dir", default="", help="逐层张量对比输出目录")
    parser.add_argument("--index-topk", type=int, default=-1, help="覆盖 index_topk（不重新生成权重）")
    parser.add_argument("--no-fake-quant", action="store_true", help="两侧都关闭 FP8/FP4 伪量化，用于隔离实现误差")
    parser.add_argument("--experts", type=int, default=-1, help="生成时覆盖 n_routed_experts")
    parser.add_argument("--perf-config", action="store_true",
                        help="生成接近真实注意力尺寸的配置（64 头、窗口 128、top-k 512）用于测速")
    parser.add_argument("--chunked-prefill", type=int, default=-1, help="fastllm 分块 prefill 大小（测试 startPos>0 的多 token 前向）")
    parser.add_argument("--quant-format", default="bf16", choices=["bf16", "real"],
                        help="real: 按真实 checkpoint 的格式保存（稠密 FP8 32x32 + 路由专家 FP4 + 共享专家 FP8）")
    parser.add_argument("--activated", type=int, default=-1, help="生成时覆盖 n_activated_experts")
    parser.add_argument("--dim", type=int, default=-1, help="生成时覆盖 hidden_size（MoE 测速用）")
    parser.add_argument("--moe-inter-dim", type=int, default=-1,
                        help="生成时覆盖 moe_intermediate_size（MoE 测速用）")
    parser.add_argument("--candidate-topk-blocks", type=int, default=-1, help="覆盖 candidate_topk_blocks")
    parser.add_argument("--layers-config", default="",
                        help="覆盖层布局，格式 n_layers:compress_ratios:kv_source:index_source:engram_layers:candidate_layer，"
                             "例如 22:0,0,2,...:2,8,14,20:2,8,14,20:2:14（用于复现真实 config 的 KV 几何）")
    return parser.parse_args()


# ---------------- 迷你配置 ----------------

TINY = dict(
    vocab_size=129280, dim=512, moe_inter_dim=256, n_layers=6, n_mtp_layers=0,
    n_heads=32, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", gate_temp=1.0, norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0,
    q_lora_rank=128, head_dim=512, rope_head_dim=64, norm_eps=1e-20, o_groups=4, o_lora_rank=64,
    window_size=8, compress_ratios=(0, 2, 2, 1, 1, 0), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
    compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0, rope_factor=16, beta_fast=32, beta_slow=1,
    index_n_heads=4, index_head_dim=128, index_topk=4,
    candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 4), engram_max_ngram_size=3, engram_vocab_size=1000, engram_n_heads=2,
    engram_head_dim=64, engram_pad_id=2, engram_compressed_vocab_size=99092,
    vision_n_layers=0, image_token_id=129264,
    dspark_block_size=0, dspark_target_layer_ids=(), dtype="bf16", expert_dtype=None,
    max_batch_size=1, max_seq_len=4096, temperature=0,
)


def hf_config_from_tiny(t, engram_num_embeddings, compressed_vocab):
    text = {
        "model_type": "deepseek_v41_text",
        "vocab_size": t["vocab_size"], "hidden_size": t["dim"], "moe_intermediate_size": t["moe_inter_dim"],
        "num_hidden_layers": t["n_layers"], "num_attention_heads": t["n_heads"], "num_key_value_heads": 1,
        "head_dim": t["head_dim"], "qk_rope_head_dim": t["rope_head_dim"], "q_lora_rank": t["q_lora_rank"],
        "o_lora_rank": t["o_lora_rank"], "o_groups": t["o_groups"], "hidden_act": "silu",
        "swiglu_limit": t["swiglu_limit"], "rms_norm_eps": t["norm_eps"], "attention_bias": False,
        "max_position_embeddings": 1048576, "rope_theta": t["rope_theta"],
        "rope_scaling": {"rope_type": "yarn", "factor": t["rope_factor"], "beta_fast": t["beta_fast"],
                         "beta_slow": t["beta_slow"], "original_max_position_embeddings": t["original_seq_len"]},
        "n_routed_experts": t["n_routed_experts"], "n_shared_experts": t["n_shared_experts"],
        "num_experts_per_tok": t["n_activated_experts"], "scoring_func": t["score_func"], "topk_method": "noaux_tc",
        "norm_topk_prob": t["norm_topk_prob"], "routed_scaling_factor": t["route_scale"],
        "sliding_window": t["window_size"], "compress_ratios": list(t["compress_ratios"]),
        "compress_rope_theta": t["compress_rope_theta"], "kv_source_layer_ids": list(t["kv_source_layers"]),
        "index_source_layer_ids": list(t["index_source_layers"]), "index_n_heads": t["index_n_heads"],
        "index_head_dim": t["index_head_dim"], "index_topk": t["index_topk"],
        "candidate_source_layer_id": t["candidate_source_layer"], "candidate_topk_blocks": t["candidate_topk_blocks"],
        "candidate_block_size": t["candidate_block_size"], "hc_mult": t["hc_mult"],
        "hc_sinkhorn_iters": t["hc_sinkhorn_iters"], "hc_eps": t["hc_eps"],
        "engram_layer_ids": list(t["engram_layer_ids"]), "engram_num_embeddings": engram_num_embeddings,
        "engram_max_ngram_size": t["engram_max_ngram_size"], "engram_vocab_size": t["engram_vocab_size"],
        "engram_n_heads": t["engram_n_heads"], "engram_head_dim": t["engram_head_dim"],
        "engram_pad_token_id": t["engram_pad_id"], "engram_compressed_vocab_size": compressed_vocab,
        "num_nextn_predict_layers": 0, "tie_word_embeddings": False,
    }
    return {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": t["image_token_id"],
        "text_config": text,
    }


# ---------------- 纯 torch kernel shim ----------------

def install_shims(reference_dir):
    import torch
    import torch.nn.functional as F

    kernel = types.ModuleType("kernel")

    def _pow2_ceil(x):
        # fast_round_scale：2^ceil(log2(x))，按 IEEE 位运算
        bits = x.float().contiguous().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exp)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        N = x.size(-1)
        z = x.float().contiguous().view(-1, N // block_size, block_size)
        amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        if scale_fmt is not None:
            s = _pow2_ceil(amax * (1.0 / 448.0))
        else:
            s = amax * (1.0 / 448.0)
        q = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
        y = q.view(x.shape).to(x.dtype)
        if inplace:
            x.copy_(y)
            return x
        return y, s.view(*x.shape[:-1], N // block_size)

    FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def _fp4_round(a):
        # a >= 0, RNE 到 e2m1 网格
        grid = FP4_GRID.to(a.device)
        idx = torch.searchsorted(grid, a.contiguous(), right=True) - 1
        idx = idx.clamp(0, 7)
        lo = grid[idx]
        hi = grid[(idx + 1).clamp(max=7)]
        mid = 0.5 * (lo + hi)
        up = (a > mid) | ((a == mid) & ((idx & 1) == 1))
        r = torch.where(up, hi, lo)
        return torch.where(a >= 6.0, torch.full_like(a, 6.0), r)

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        N = x.size(-1)
        z = x.float().contiguous().view(-1, N // block_size, block_size)
        amax = z.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            amax = amax.clamp_min(6 * 2 ** -9)
            s = (amax / 6.0).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp_min(6 * 2 ** -126)
            s = _pow2_ceil(amax * (1.0 / 6.0))
        q = (z / s).clamp(-6.0, 6.0)
        q = torch.copysign(_fp4_round(q.abs()), q) * s
        y = q.view(x.shape).to(x.dtype)
        if inplace:
            x.copy_(y)
            return x
        return y, s

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        b, s_, _ = mixes.size()
        m = mixes.float()
        pre = torch.sigmoid(m[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
        post = 2 * torch.sigmoid(m[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
        comb = m[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
        comb = comb.view(b, s_, hc_mult, hc_mult)
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        # q [b,s,h,d], kv [b,n,d], topk_idxs [b,s,k] (-1 无效), attn_sink [h]
        b, s_, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        rows = torch.gather(kv.float(), 1, idx.clamp(min=0).view(b, -1, 1).expand(-1, -1, d)).view(b, s_, -1, d)
        scores = torch.einsum("bshd,bskd->bshk", q.float(), rows) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, s_, h, 1)
        logits = torch.cat([scores, sink], dim=-1)
        probs = logits.softmax(-1)[..., :-1]
        out = torch.einsum("bshk,bskd->bshd", probs, rows)
        return out.to(q.dtype)

    def fp8_gemm(*args, **kwargs):
        raise RuntimeError("fp8_gemm is not available in the torch shim; use dtype=bf16")

    fp4_gemm = fp8_gemm
    kernel.CALLS = []

    def sparse_attn_recording(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        kernel.CALLS.append((q.detach().clone(), kv.detach().clone(), topk_idxs.detach().clone(), out.detach().clone()))
        return out

    if os.environ.get("V41_REF_NO_FAKE_QUANT") == "1":
        def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
            if inplace:
                return x
            return x, torch.ones(*x.shape[:-1], x.size(-1) // block_size, device=x.device)

        def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
            if inplace:
                return x
            return x, torch.ones(*x.shape[:-1], x.size(-1) // block_size, device=x.device)
    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_act_quant
    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    kernel.sparse_attn = sparse_attn_recording
    kernel.fp8_gemm = fp8_gemm
    kernel.fp4_gemm = fp4_gemm
    sys.modules["kernel"] = kernel

    vision = types.ModuleType("vision")

    class _Unused(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    vision.Aligner = _Unused
    vision.ViT = _Unused
    sys.modules["vision"] = vision

    image_processor = types.ModuleType("image_processor")
    image_processor.IMAGE = 0
    image_processor.IMAGE_END = 1
    image_processor.IMAGE_NEW_LINE = 2
    image_processor.IMAGE_START = 3
    sys.modules["image_processor"] = image_processor

    sys.path.insert(0, reference_dir)
    for name in ("engram", "model"):
        if name in sys.modules:
            del sys.modules[name]
    engram = importlib.import_module("engram")
    model = importlib.import_module("model")

    # 记录 indexer 内部张量（与官方 Indexer.forward 逐行一致，仅加了记录）
    kernel.INDEXER = []
    shared_attn = model.shared_attn
    apply_rotary_emb = model.apply_rotary_emb
    select_candidate_blocks = model.select_candidate_blocks
    fp4_act_quant_fn = kernel.fp4_act_quant

    def indexer_forward(self, x, qr, latent, start_pos, offset):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        if self.owns_k and latent is not None:
            freqs = (self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0
                     else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant_fn(k, 32, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            shared_attn.index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant_fn(q, 32, True)
        index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        kernel.INDEXER.append((q.detach().clone(), index_k.detach().clone(), weights.detach().clone(),
                               index_score.detach().clone()))
        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    model.Indexer.forward = indexer_forward
    return engram, model


# ---------------- 随机权重生成 ----------------

def e8m0_quantize(x, block=32):
    """x [rows, dim] float -> (fp8 e4m3 [rows, dim], e8m0 scale [rows, dim/block])"""
    import torch
    rows, dim = x.shape
    z = x.float().view(rows, dim // block, block)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0))
    s = torch.pow(2.0, exp)
    q = (z / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(rows, dim)
    scale = (exp.view(rows, dim // block) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q, scale


def _pow2_ceil_np(x):
    import numpy as np
    bits = x.astype(np.float32).view(np.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).astype(np.int32)
    return np.ldexp(np.ones_like(x, dtype=np.float32), exp), exp


def quantize_fp8_block32(w):
    """w [n, m] -> (fp8 [n, m], e8m0 scale [n/32, m/32], dequantized float)"""
    import torch
    n, m = w.shape
    z = w.float().view(n // 32, 32, m // 32, 32).permute(0, 2, 1, 3).contiguous()
    amax = z.abs().amax(dim=(2, 3)).clamp_min(1e-4)
    s, exp = _pow2_ceil_np((amax / 448.0).numpy())
    s = torch.from_numpy(s)
    q = (z / s[:, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    deq = q.float() * s[:, :, None, None]
    q = q.permute(0, 2, 1, 3).reshape(n, m).contiguous()
    deq = deq.permute(0, 2, 1, 3).reshape(n, m).contiguous()
    scale = torch.from_numpy((exp + 127).astype(np.uint8)).view(torch.float8_e8m0fnu)
    return q, scale, deq


FP4_TABLE_NP = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def quantize_fp4_group32(w):
    """w [n, m] -> (packed int8 [n, m/2], e8m0 scale [n, m/32], dequantized float)"""
    import torch
    n, m = w.shape
    z = w.float().view(n, m // 32, 32)
    amax = z.abs().amax(-1).clamp_min(6 * 2 ** -126)
    s, exp = _pow2_ceil_np((amax / 6.0).numpy())
    s = torch.from_numpy(s)
    q = (z / s[:, :, None]).clamp(-6, 6)
    grid = torch.tensor(FP4_TABLE_NP)
    a = q.abs()
    idx = (torch.searchsorted(grid, a.contiguous(), right=True) - 1).clamp(0, 7)
    lo, hi = grid[idx], grid[(idx + 1).clamp(max=7)]
    mid = 0.5 * (lo + hi)
    up = (a > mid) | ((a == mid) & ((idx & 1) == 1))
    code = torch.where(up, (idx + 1).clamp(max=7), idx)
    code = torch.where(a >= 6.0, torch.full_like(code, 7), code)
    code = torch.where(q < 0, code + 8, code).to(torch.uint8)      # 符号位在 bit3
    deq = (grid[(code & 7).long()] * torch.where(q < 0, -1.0, 1.0)) * s[:, :, None]
    code = code.view(n, m)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).view(torch.int8).contiguous()
    scale = torch.from_numpy((exp + 127).astype(np.uint8)).view(torch.float8_e8m0fnu)
    return packed, scale, deq.view(n, m).contiguous()


def apply_real_quant_format(state):
    """把 bf16 state 转成真实 checkpoint 的存储格式；返回 (quantized_state, reference_state)"""
    import torch
    quantized, reference = {}, {}
    for name, value in state.items():
        dense_fp8 = (any(k in name for k in (".attn.wq_a.", ".attn.wq_b.", ".attn.wkv.", ".attn.wo_a.", ".attn.wo_b.",
                                              ".attn.indexer.wq_b.", ".engram.wkv.", ".ffn.shared_experts."))
                     and name.endswith(".weight") and value.dim() == 2)
        routed_fp4 = ".ffn.experts." in name and name.endswith(".weight") and value.dim() == 2
        if dense_fp8:
            q, scale, deq = quantize_fp8_block32(value)
            quantized[name] = q
            quantized[name.replace(".weight", ".scale")] = scale
            reference[name] = deq.to(torch.bfloat16)
        elif routed_fp4:
            packed, scale, deq = quantize_fp4_group32(value)
            quantized[name] = packed
            quantized[name.replace(".weight", ".scale")] = scale
            reference[name] = deq.to(torch.bfloat16)
        else:
            quantized[name] = value
            reference[name] = value
    return quantized, reference


def generate_checkpoint(args, engram_mod, model_mod, tokenizer):
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(args.seed)
    margs = model_mod.ModelArgs(**TINY)
    layout = engram_mod.EngramLayout.from_args(margs)
    num_embeddings = [int(sum(sum(per) for per in layer)) for layer in layout.primes]
    margs.engram_num_embeddings = tuple(num_embeddings)
    with torch.device("cpu"):
        torch.set_default_dtype(torch.bfloat16)
        model = model_mod.Transformer(margs, tokenizer)
        torch.set_default_dtype(torch.float32)

    state = {}
    for name, param in model.state_dict().items():
        if name.startswith("engram_hash."):
            continue
        if name.startswith("mtp.") or ".k_cache" in name or "window_kv_cache" in name or "compress_kv_cache" in name \
                or name.endswith(".kv_state") or name.endswith(".score_state") or name.endswith(".freqs_cis"):
            continue
        shape = tuple(param.shape)
        if param.dtype == torch.float8_e4m3fn:
            raw = torch.randn(shape, dtype=torch.float32) * 0.5
            q, scale = e8m0_quantize(raw)
            state[name] = q
            state[name.replace(".weight", ".scale")] = scale
            continue
        if param.dtype == torch.float8_e8m0fnu:
            continue
        if name.endswith(".attn_sink"):
            value = torch.randn(shape) * 0.5
        elif ".hc_attn_fn" in name or ".hc_ffn_fn" in name:
            value = torch.randn(shape) * 0.02
        elif name.endswith("_scale") and ".hc_" in name:
            value = torch.full(shape, 0.5)
        elif name.endswith("_base") and ".hc_" in name:
            value = torch.randn(shape) * 0.1
        elif name.endswith("norm.weight") or name == "norm.weight":
            value = 1.0 + torch.randn(shape) * 0.1
        elif ".engram.q_weight" in name or ".engram.k_weight" in name:
            value = 1.0 + torch.randn(shape) * 0.1
        elif name.endswith(".gate.bias") or name.endswith(".gate.bias_vl"):
            value = torch.randn(shape) * 0.1
        elif name == "embed.weight":
            value = torch.randn(shape) * 0.5
        elif name == "head.weight":
            value = torch.randn(shape) * (1.0 / math.sqrt(shape[1]))
        elif len(shape) == 2:
            value = torch.randn(shape) * (1.0 / math.sqrt(shape[1]))
        else:
            value = torch.randn(shape) * 0.1
        state[name] = value.to(param.dtype).contiguous()

    os.makedirs(args.work_dir, exist_ok=True)
    ref_path = os.path.join(args.work_dir, "reference.safetensors")
    if os.path.exists(ref_path):
        os.remove(ref_path)
    if args.quant_format == "real":
        state, reference = apply_real_quant_format(state)
        save_file(reference, ref_path)
    save_file(state, os.path.join(args.work_dir, "model.safetensors"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(args.tokenizer_dir, f), os.path.join(args.work_dir, f))
    hash_state = engram_mod.NgramHashState(margs, layout, tokenizer)
    compressed_vocab = int(hash_state.token_map.max().item()) + 1
    with open(os.path.join(args.work_dir, "config.json"), "w") as f:
        json.dump(hf_config_from_tiny(TINY, num_embeddings, compressed_vocab), f, indent=2)
    with open(os.path.join(args.work_dir, "reference_config.json"), "w") as f:
        cfg = dict(TINY)
        cfg["engram_num_embeddings"] = num_embeddings
        json.dump(cfg, f, indent=2)
    # engram meta
    from ftllm.deepseek_v41_engram import build_engram_meta
    meta = build_engram_meta(args.work_dir)
    with open(os.path.join(args.work_dir, "engram_meta.json"), "w") as f:
        json.dump(meta, f)
    print("checkpoint written to", args.work_dir, "engram rows:", num_embeddings, "compressed vocab:", compressed_vocab)
    return state


# ---------------- 参考前向 ----------------

def run_reference(args, engram_mod, model_mod, tokenizer, prompt):
    import torch
    from safetensors.torch import load_file

    with open(os.path.join(args.work_dir, "reference_config.json")) as f:
        cfg = json.load(f)
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids",
              "engram_num_embeddings", "dspark_target_layer_ids"):
        cfg[k] = tuple(cfg[k])
    margs = model_mod.ModelArgs(**cfg)
    for fn in (model_mod.precompute_freqs_cis, model_mod.get_window_topk_idxs, model_mod.get_dspark_topk_idxs):
        fn.cache_clear()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    model = model_mod.Transformer(margs, tokenizer)
    torch.set_default_dtype(torch.float32)
    ref_path = os.path.join(args.work_dir, "reference.safetensors")
    state = load_file(ref_path if os.path.exists(ref_path) else os.path.join(args.work_dir, "model.safetensors"),
                      device="cuda")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not (m.startswith("engram_hash.") or "freqs_cis" in m or "_cache" in m
                                          or m.endswith("kv_state") or m.endswith("score_state"))]
    assert not missing and not unexpected, (missing, unexpected)

    # ParallelEngramEmbedding 用 F.embedding 索引 fp8 表；某些 torch 版本不支持，改成手动 gather
    def engram_forward(self, indices):
        flat = indices.reshape(-1)
        values = self.weight.view(torch.uint8)[flat].view(torch.float8_e4m3fn).float()
        scales = self.scale.view(torch.uint8)[flat].float() - 127.0
        scales = torch.pow(2.0, scales)
        values = values.unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, self.dim)
    model_mod.ParallelEngramEmbedding.forward = engram_forward

    ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    logits_list = []
    tokens = []
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        import numpy as np

        call_counter = {}

        def suffix_for(name):
            n = call_counter.get(name, 0)
            call_counter[name] = n + 1
            return "" if n == 0 else "_p%d" % (len(prompt) + n - 1)

        def dump(name):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t.detach().float().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(args.dump_dir, name + suffix_for(name) + ".bin"))
            return hook

        def dump_input(name):
            def hook(module, inp):
                inp[0].detach().float().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(args.dump_dir, name + suffix_for(name) + ".bin"))
            return hook

        model.embed.register_forward_hook(dump("ref_embed"))

        def dump_gate(name):
            def hook(module, inp, out):
                out[1].detach().float().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(args.dump_dir, name + suffix_for(name) + ".bin"))
            return hook

        for i, layer in enumerate(model.layers):
            layer.ffn.gate.register_forward_hook(dump_gate("ref_layer%d_expert_idx" % i))
            layer.register_forward_hook(dump("ref_layer%d" % i))
            layer.attn.register_forward_pre_hook(dump_input("ref_layer%d_attn_in" % i))
            layer.attn.register_forward_hook(dump("ref_layer%d_attn" % i))
            layer.ffn.register_forward_pre_hook(dump_input("ref_layer%d_ffn_in" % i))
            layer.ffn.register_forward_hook(dump("ref_layer%d_ffn" % i))
    with torch.inference_mode():
        sys.modules["kernel"].CALLS.clear()
        sys.modules["kernel"].INDEXER.clear()
        _, logits, _ = model(ids, 0)
        if args.dump_dir:
            import numpy as np
            index_layers = [i for i in range(len(model.layers)) if model.layers[i].attn.indexer is not None
                            and model.layers[i].attn.compress_ratio]
            for li, (q_, k_, w_, sc_) in zip(index_layers, sys.modules["kernel"].INDEXER):
                q_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_idxq.bin" % li))
                k_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_idxk.bin" % li))
                w_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_idxw.bin" % li))
                sc_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_score.bin" % li))
            for i, (q_, kv_, idx_, o_) in enumerate(sys.modules["kernel"].CALLS):
                q_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_q.bin" % i))
                kv_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_kvcat.bin" % i))
                idx_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_topk.bin" % i))
                o_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_attn_o_raw.bin" % i))
        logits_list.append(logits[0].float().cpu())
        nxt = int(logits[0].argmax().item())
        tokens.append(nxt)
        pos = ids.size(1)
        for _ in range(args.decode):
            sys.modules["kernel"].CALLS.clear()
            _, logits, _ = model(torch.tensor([[nxt]], device="cuda"), pos)
            if args.dump_dir:
                import numpy as np
                for i, (q_, kv_, idx_, o_) in enumerate(sys.modules["kernel"].CALLS):
                    sfx = "_p%d" % pos
                    q_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_q%s.bin" % (i, sfx)))
                    kv_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_kvcat%s.bin" % (i, sfx)))
                    idx_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_topk%s.bin" % (i, sfx)))
                    o_.float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_layer%d_attn_o_raw%s.bin" % (i, sfx)))
            logits_list.append(logits[0].float().cpu())
            nxt = int(logits[0].argmax().item())
            tokens.append(nxt)
            pos += 1
    torch.set_default_device("cpu")
    return logits_list, tokens


# ---------------- fastllm 前向 ----------------

def run_fastllm(args, prompt, vocab_size):
    import numpy as np
    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_ENGRAM_META"] = os.path.join(args.work_dir, "engram_meta.json")
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        os.environ["FASTLLM_DSV41_DUMP_DIR"] = args.dump_dir
    parser = make_normal_parser("v41 reference")
    fastllm_argv = ["--path", args.work_dir, "--dtype", args.dtype, "--device", args.device,
                    "--moe_device", args.moe_device, "-t", str(args.threads)]
    if args.chunked_prefill > 0:
        fastllm_argv += ["--chunked_prefill_size", str(args.chunked_prefill)]
    if args.kv_cache_dtype:
        fastllm_argv += ["--kv_cache_dtype", args.kv_cache_dtype]
    fargs = parser.parse_args(fastllm_argv)
    if fargs.max_batch <= 0:
        fargs.max_batch = 1
    if fargs.tokens <= 0:
        fargs.tokens = 65536
    model = make_normal_llm_model(fargs)
    if args.fastllm_only_load:
        print("fastllm model loaded")
        return [], []
    import time
    t0 = time.time()
    input_array = (ctypes.c_int * len(prompt))(*prompt)
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model, len(prompt), input_array, ctypes.c_int(args.decode + 1), ctypes.c_int(0),
        ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0), ctypes.c_float(1.0),
        ctypes.c_bool(True), ctypes.c_int(0), None)
    logits_list, tokens = [], []
    buf = (ctypes.c_float * vocab_size)()
    first = None
    while True:
        token_id = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
        if token_id < 0:
            break
        if first is None:
            first = time.time()
        tokens.append(int(token_id))
        logits_list.append(np.ctypeslib.as_array(buf).copy())
    end = time.time()
    if first is not None:
        print("fastllm timing: prefill %d tokens in %.2fs, decode %d tokens in %.2fs (%.2f tok/s)" % (
            len(prompt), first - t0, max(0, len(tokens) - 1), end - first,
            (len(tokens) - 1) / max(1e-6, end - first)))
    return logits_list, tokens


def main():
    args = parse_args()
    if args.perf_config:
        TINY.update(dict(n_layers=4, n_heads=64, o_groups=8, window_size=128, index_topk=512,
                         compress_ratios=(0, 2, 1, 1), kv_source_layers=(1, 2), index_source_layers=(1, 2, 3),
                         candidate_source_layer=2, candidate_topk_blocks=2048, candidate_block_size=8,
                         index_n_heads=32, engram_layer_ids=(1,)))
    if args.layers_config:
        parts = args.layers_config.split(":")
        TINY["n_layers"] = int(parts[0])
        TINY["compress_ratios"] = tuple(int(x) for x in parts[1].split(","))
        TINY["kv_source_layers"] = tuple(int(x) for x in parts[2].split(","))
        TINY["index_source_layers"] = tuple(int(x) for x in parts[3].split(","))
        TINY["engram_layer_ids"] = tuple(int(x) for x in parts[4].split(",")) if parts[4] else ()
        TINY["candidate_source_layer"] = int(parts[5])
        assert len(TINY["compress_ratios"]) == TINY["n_layers"], "compress_ratios 长度要等于 n_layers"
    if args.dim > 0:
        TINY["dim"] = args.dim
    if args.moe_inter_dim > 0:
        TINY["moe_inter_dim"] = args.moe_inter_dim
    if args.experts > 0:
        TINY["n_routed_experts"] = args.experts
    if args.activated > 0:
        TINY["n_activated_experts"] = args.activated
    if args.no_fake_quant:
        os.environ["V41_REF_NO_FAKE_QUANT"] = "1"
        os.environ["FASTLLM_DSV41_DISABLE_FAKE_QUANT"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    engram_mod, model_mod = install_shims(args.reference_dir)

    if args.regenerate or not os.path.exists(os.path.join(args.work_dir, "model.safetensors")):
        generate_checkpoint(args, engram_mod, model_mod, tokenizer)
    overrides = {}
    if args.index_topk > 0:
        overrides["index_topk"] = args.index_topk
    if args.candidate_topk_blocks > 0:
        overrides["candidate_topk_blocks"] = args.candidate_topk_blocks
    if overrides:
        hf_path = os.path.join(args.work_dir, "config.json")
        ref_path = os.path.join(args.work_dir, "reference_config.json")
        hf = json.load(open(hf_path))
        ref = json.load(open(ref_path))
        for k, v in overrides.items():
            ref[k] = v
            TINY[k] = v
        hf["text_config"]["index_topk"] = ref["index_topk"]
        hf["text_config"]["candidate_topk_blocks"] = ref["candidate_topk_blocks"]
        json.dump(hf, open(hf_path, "w"), indent=2)
        json.dump(ref, open(ref_path, "w"), indent=2)
        print("config overrides:", overrides)

    text = ("DeepSeek-V4.1-Flash is the first model of a new architecture family. It combines sliding window "
            "attention, cross-layer compressed KV sharing, a two-level sparse indexer, engram n-gram memory and "
            "hyper-connections. 这是一个用于对齐测试的中英文混合提示词，包含数字 12345 与符号 !@#。") * max(4, args.prefill // 60 + 1)
    prompt = tokenizer.encode(text)[: args.prefill]
    print("prompt tokens:", len(prompt))

    ref_logits, ref_tokens = ([], [])
    if not args.skip_reference:
        ref_logits, ref_tokens = run_reference(args, engram_mod, model_mod, tokenizer, prompt)
        print("reference tokens:", ref_tokens)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    if args.skip_fastllm:
        return
    fl_logits, fl_tokens = run_fastllm(args, prompt, TINY["vocab_size"])
    print("fastllm tokens:  ", fl_tokens)
    if not ref_logits:
        if args.dump_dir:
            verify_selection_kernels(args.dump_dir, len(prompt))
        return
    import numpy as np
    ok = True
    for step in range(min(len(ref_logits), len(fl_logits))):
        a = ref_logits[step].numpy()
        b = fl_logits[step]
        diff = np.abs(a - b)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        same = int(a.argmax()) == int(b.argmax())
        print("step %d: max|diff|=%.4f mean|diff|=%.5f cos=%.6f ref_argmax=%d fl_argmax=%d %s" % (
            step, diff.max(), diff.mean(), cos, a.argmax(), b.argmax(), "OK" if same else "MISMATCH"))
        ok = ok and same and cos > 0.999
        if not same:
            break
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.dump_dir:
        compare_dumps(args.dump_dir, TINY["n_layers"], TINY["hc_mult"], TINY["dim"], len(prompt))
        verify_selection_kernels(args.dump_dir, len(prompt))
        compare_decode_dumps(args.dump_dir, TINY["n_layers"], len(prompt), args.decode)


def verify_selection_kernels(dump_dir, seqlen):
    """用 numpy 复算候选块与 top-k（同样的分数输入），验证 CUDA / CPU 选择算子。"""
    import numpy as np
    ratios = TINY["compress_ratios"]
    bs = TINY["candidate_block_size"]
    cand_layer = TINY["candidate_source_layer"]
    topk_blocks = TINY["candidate_topk_blocks"]
    topk = TINY["index_topk"]

    def load(name):
        path = os.path.join(dump_dir, name + ".bin")
        return np.fromfile(path, dtype=np.float32) if os.path.exists(path) else None

    cand_mask = None
    for layer in TINY["index_source_layers"]:
        ratio = ratios[layer]
        score = load("fl_layer%d_score" % layer)
        if score is None:
            continue
        score = score.reshape(seqlen, -1)
        m = score.shape[1]
        visible = np.minimum(m, (np.arange(seqlen) + 1) // ratio)
        masked = score.copy()
        for t in range(seqlen):
            masked[t, visible[t]:] = -np.inf
        if layer == cand_layer:
            nb = (m + bs - 1) // bs
            padded = np.full((seqlen, nb * bs), -np.inf, dtype=np.float32)
            padded[:, :m] = masked
            block = padded.reshape(seqlen, nb, bs).max(-1)
            for t in range(seqlen):
                if visible[t] > 0:
                    block[t, (visible[t] - 1) // bs] = np.inf
            keep = min(topk_blocks, nb)
            expect = np.zeros((seqlen, nb), dtype=np.uint8)
            for t in range(seqlen):
                order = np.argsort(-block[t], kind="stable")[:keep]
                for k in order:
                    if block[t, k] > -np.inf:
                        expect[t, k] = 1
            got = load("fl_layer%d_cand" % layer)
            if got is not None:
                got = got.reshape(seqlen, nb).astype(np.uint8)
                bad = int((got != expect).any(1).sum())
                print("%-22s numpy vs kernel candidate masks differ on %d / %d tokens" % ("layer%d cand" % layer, bad, seqlen))
                cand_mask = got
        use_cand = cand_mask is not None and cand_layer >= 0 and cand_layer < layer
        if use_cand:
            for t in range(seqlen):
                for j in range(m):
                    if cand_mask[t, j // bs] == 0:
                        masked[t, j] = -np.inf
        got = load("fl_layer%d_topk" % layer)
        if got is None:
            continue
        got = got.reshape(seqlen, -1)
        bad = 0
        for t in range(seqlen):
            eligible = np.where(np.isfinite(masked[t]))[0]
            k = min(topk, len(eligible))
            order = eligible[np.argsort(-masked[t, eligible], kind="stable")[:k]]
            expect_set = set(int(v) for v in order)
            got_set = set(int(v) for v in got[t] if v >= 0)
            if expect_set != got_set:
                # 允许并列分数造成的差异
                thr = masked[t, order].min() if k > 0 else -np.inf
                ties_ok = all(masked[t, v] >= thr for v in got_set) and len(got_set) == k
                if not ties_ok:
                    bad += 1
        print("%-22s numpy vs kernel top-k sets differ on %d / %d tokens" % ("layer%d topk" % layer, bad, seqlen))


def compare_decode_dumps(dump_dir, n_layers, seqlen, decode_steps):
    import numpy as np

    def load(name):
        path = os.path.join(dump_dir, name + ".bin")
        return np.fromfile(path, dtype=np.float32) if os.path.exists(path) else None

    def rel(a, b):
        if a is None or b is None or a.size != b.size:
            return "missing(%s,%s)" % (None if a is None else a.size, None if b is None else b.size)
        d = np.abs(a - b)
        return "rel=%.4f max=%.3f" % (d.mean() / (np.abs(a).mean() + 1e-9), d.max())

    for step in range(decode_steps):
        pos = seqlen + step
        sfx = "_p%d" % pos
        line = ["pos %d:" % pos]
        for i in range(n_layers):
            kvcat = load("ref_layer%d_kvcat" % i + sfx)
            ring = load("fl_layer%d_ring" % i + sfx)
            rt = load("ref_layer%d_topk" % i + sfx)
            ft = load("fl_layer%d_topk" % i + sfx)
            parts = ["L%d attn_o %s" % (i, rel(load("ref_layer%d_attn_o_raw" % i + sfx), load("fl_layer%d_attn_o_raw" % i + sfx)))]
            if kvcat is not None and ring is not None:
                # 参考的 kvcat 前 windowSize 行是环形缓存（按 slot 排列，但被写入的是本 token 后的状态）
                w = ring.size // 512
                parts.append("ring %s" % rel(kvcat.reshape(-1, 512)[:w].reshape(-1), ring))
                fl_ckv = load("fl_layer%d_ckv" % i + sfx)
                if fl_ckv is not None:
                    n = kvcat.size // 512 - w
                    parts.append("ckv %s" % rel(kvcat.reshape(-1, 512)[w:].reshape(-1), fl_ckv.reshape(-1, 512)[:n].reshape(-1)))
            if rt is not None and ft is not None:
                w = 8
                rset = set(int(v) - w for v in rt.reshape(-1)[w:] if v >= 0)
                fset = set(int(v) for v in ft.reshape(-1) if v >= 0)
                parts.append("topk %s" % ("same" if rset == fset else "DIFF %s vs %s" % (sorted(rset)[:8], sorted(fset)[:8])))
            re_ = load("ref_layer%d_expert_idx" % i + sfx)
            fe_ = load("fl_layer%d_expert_idx" % i + sfx)
            if re_ is not None and fe_ is not None:
                parts.append("experts %s" % ("same" if set(re_.astype(int)) == set(fe_.astype(int)) else "FLIP %s vs %s" % (sorted(re_.astype(int)), sorted(fe_.astype(int)))))
            parts.append("hidden %s" % rel(load("ref_layer%d" % i + sfx), load("fl_layer%d" % i + sfx)))
            line.append("  " + " | ".join(parts))
        print("\n".join(line))


def compare_dumps(dump_dir, n_layers, hc, dim, seqlen):
    import numpy as np

    def load(name):
        path = os.path.join(dump_dir, name + ".bin")
        if not os.path.exists(path):
            return None
        return np.fromfile(path, dtype=np.float32)

    def report(tag, a, b):
        if a is None or b is None or a.size != b.size:
            print("%-22s missing or size mismatch (%s vs %s)" % (tag, None if a is None else a.size, None if b is None else b.size))
            return
        diff = np.abs(a - b)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print("%-22s max|diff|=%.4f mean|diff|=%.5f rel=%.4f cos=%.6f" % (
            tag, diff.max(), diff.mean(), diff.mean() / (np.abs(a).mean() + 1e-9), cos))

    e = load("ref_embed")
    if e is not None:
        report("embed", np.repeat(e.reshape(seqlen, 1, dim), hc, axis=1).reshape(-1), load("fl_embed"))
    ratios = TINY["compress_ratios"]
    for i in range(n_layers):
        report("layer%d attn_in" % i, load("ref_layer%d_attn_in" % i), load("fl_layer%d_attn_in" % i))
        report("layer%d q" % i, load("ref_layer%d_q" % i), load("fl_layer%d_q" % i))
        kvcat = load("ref_layer%d_kvcat" % i)
        if kvcat is not None:
            kvcat = kvcat.reshape(-1, 512)
            report("layer%d window kv" % i, kvcat[:seqlen].reshape(-1), load("fl_layer%d_kv" % i))
            if ratios[i] > 0:
                fl_ckv = load("fl_layer%d_ckv" % i)
                nblocks = kvcat.shape[0] - seqlen
                if fl_ckv is not None:
                    report("layer%d compressed kv" % i, kvcat[seqlen:].reshape(-1), fl_ckv.reshape(-1, 512)[:nblocks].reshape(-1))
        report("layer%d idxq" % i, load("ref_layer%d_idxq" % i), load("fl_layer%d_idxq" % i))
        rk = load("ref_layer%d_idxk" % i)
        fk = load("fl_layer%d_idxk" % i)
        if rk is not None and fk is not None:
            report("layer%d idxk" % i, rk, fk.reshape(-1, 128)[: rk.size // 128].reshape(-1))
        rs = load("ref_layer%d_score" % i)
        fs = load("fl_layer%d_score" % i)
        if rs is not None and fs is not None:
            rs = rs.reshape(seqlen, -1)
            fs = fs.reshape(seqlen, -1)
            finite = np.isfinite(rs)
            report("layer%d score" % i, rs[finite], fs[finite])
            print("   layer%d score row 9 ref=%s fl=%s" % (i, np.round(rs[9][:6], 3), np.round(fs[9][:6], 3)))
        rt = load("ref_layer%d_topk" % i)
        ft = load("fl_layer%d_topk" % i)
        if rt is not None and ft is not None and ratios[i] > 0:
            rt = rt.reshape(seqlen, -1)
            ft = ft.reshape(seqlen, -1)
            width = ft.shape[1]
            mism = 0
            for t in range(seqlen):
                rset = set(int(v) - seqlen for v in rt[t, rt.shape[1] - width:] if v >= 0)
                fset = set(int(v) for v in ft[t] if v >= 0)
                if rset != fset:
                    mism += 1
                    if mism <= 3:
                        print("   layer%d token %d topk ref=%s fl=%s" % (i, t, sorted(rset), sorted(fset)))
            print("%-22s mismatching tokens: %d / %d" % ("layer%d topk" % i, mism, seqlen))
        report("layer%d attn_o_raw" % i, load("ref_layer%d_attn_o_raw" % i), load("fl_layer%d_attn_o_raw" % i))
        report("layer%d attn" % i, load("ref_layer%d_attn" % i), load("fl_layer%d_attn" % i))
        report("layer%d ffn_in" % i, load("ref_layer%d_ffn_in" % i), load("fl_layer%d_ffn_in" % i))
        re_ = load("ref_layer%d_expert_idx" % i)
        fe_ = load("fl_layer%d_expert_idx" % i)
        if re_ is not None and fe_ is not None:
            k = re_.size // seqlen
            flips = sum(1 for t in range(seqlen) if set(re_.reshape(seqlen, k)[t].astype(int)) != set(fe_.reshape(seqlen, k)[t].astype(int)))
            print("%-22s expert-set flips: %d / %d" % ("layer%d routing" % i, flips, seqlen))
        report("layer%d ffn" % i, load("ref_layer%d_ffn" % i), load("fl_layer%d_ffn" % i))
        report("layer%d hidden" % i, load("ref_layer%d" % i), load("fl_layer%d" % i))


if __name__ == "__main__":
    main()
