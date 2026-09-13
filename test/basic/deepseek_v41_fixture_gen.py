#!/usr/bin/env python3
"""
生成 ``test/basic/test_deepseek_v41_cpu_fixture.py`` 使用的固定 fixture。

思路：``deepseek_v41_reference.py`` 的端到端对齐测试需要 torch + 官方 inference 代码 + GPU，
进不了 CI。这里复用它的迷你模型生成与官方参考前向，把「一个足够小的 V4.1 模型 + 官方实现算出的
参考 logits」一次性固化成一个 npz，之后的回归测试只需要 numpy 与 fastllm 的 CPU 路径。

微型配置（MICRO）在保持 V4.1 全部结构特性的前提下把规模压到最小：

  * ``compress_ratios = (0, 2, 2, 1, 1)``：0 / 1 / 2 三种压缩层都在；
  * ``kv_source_layers = (1, 3)``：层 2 复用层 1 的压缩 KV，层 4 复用层 3 的（跨层共享）；
  * ``index_source_layers = (1, 3, 4)`` + ``candidate_source_layer = 3``：两级 top-k
    （层 3 先按块选候选，层 4 只在候选块内选 top-k）；
  * ``engram_layer_ids = (1, 4)``：两个 Engram 层；
  * ``hc_mult = 2``：Hyper-Connections；
  * ``window_size = 8`` 而 prefill 远大于 8：滑窗环形缓冲要绕圈。

用法（在 21 这类装有 torch / 官方 inference 代码的机器上跑一次）：

    PYTHONPATH=build/tools python test/basic/deepseek_v41_fixture_gen.py \
        --reference-dir /mnt/shared2/models/DeepSeek-V4.1-Flash/inference \
        --work-dir /root/v41-micro --out test/basic/deepseek_v41_fixture.npz
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "deepseek_v41_fixture.npz")

# 特意选得很小：整包 fixture 目标 < 5 MB。
MICRO = dict(
    vocab_size=128, dim=128, moe_inter_dim=128, n_layers=5, n_mtp_layers=0,
    n_heads=4, n_routed_experts=2, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", gate_temp=1.0, norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0,
    q_lora_rank=32, head_dim=128, rope_head_dim=32, norm_eps=1e-20, o_groups=2, o_lora_rank=32,
    window_size=8, compress_ratios=(0, 2, 2, 1, 1), kv_source_layers=(1, 3),
    index_source_layers=(1, 3, 4),
    compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0, rope_factor=16,
    beta_fast=32, beta_slow=1,
    index_n_heads=2, index_head_dim=64, index_topk=64,
    candidate_source_layer=3, candidate_topk_blocks=32, candidate_block_size=2,
    hc_mult=2, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 4), engram_max_ngram_size=3, engram_vocab_size=64, engram_n_heads=2,
    engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=128,  # 每个 token 归一化后互不相同，等于 vocab_size
    vision_n_layers=0, image_token_id=127,
    dspark_block_size=0, dspark_target_layer_ids=(), dtype="bf16", expert_dtype=None,
    max_batch_size=1, max_seq_len=4096, temperature=0,
)

FIXTURE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "engram_meta.json", "model.safetensors")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", required=True, help="官方 inference/ 目录")
    p.add_argument("--work-dir", default="/tmp/v41-micro", help="迷你模型中间目录")
    p.add_argument("--out", default=DEFAULT_OUT, help="输出的 npz")
    p.add_argument("--prefill", type=int, default=48)
    p.add_argument("--decode", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-seed", type=int, default=1234)
    p.add_argument("--min-margin", type=float, default=0.6,
                   help="每一步参考 logits 的 top1-top2 最小间距；不满足就换一个提示词种子重试")
    p.add_argument("--seed-tries", type=int, default=96)
    p.add_argument("--index-topk", type=int, default=-1, help="覆盖 index_topk")
    p.add_argument("--candidate-topk-blocks", type=int, default=-1)
    p.add_argument("--keep-work-dir", action="store_true")
    return p.parse_args()


# ---------------- 迷你 tokenizer ----------------

def write_tiny_tokenizer(directory, vocab_size):
    """写一个只有 vocab_size 个 token 的 ByteLevel BPE tokenizer。

    fastllm 只读 ``model.vocab`` 与 ``added_tokens``；Engram 元数据生成需要 ``tokenizers``
    能 decode 每个 id（用于计算归一化后的压缩 id）。这里让每个 token 都是互不相同的字面串，
    压缩词表大小恰好等于 vocab_size。
    """
    os.makedirs(directory, exist_ok=True)
    specials = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"]
    vocab = {}
    for i, tok in enumerate(specials):
        vocab[tok] = i
    # ByteLevel 字母表：'Ġ' = 空格。用 "Ġt<id>" 保证 decode 出来互不相同且归一化后仍不同。
    for i in range(len(specials), vocab_size):
        vocab["Ġt%d" % i] = i
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": i, "content": tok, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for i, tok in enumerate(specials)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                          "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                    "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": []},
    }
    with open(os.path.join(directory, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)
    config = {
        "add_bos_token": True, "add_eos_token": False, "clean_up_tokenization_spaces": False,
        "bos_token": specials[0], "eos_token": specials[1], "pad_token": specials[2],
        "legacy": True, "model_max_length": 4096, "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    }
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    return directory


def load_reference_module():
    path = os.path.join(HERE, "deepseek_v41_reference.py")
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["deepseek_v41_reference"] = module
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    if args.index_topk > 0:
        MICRO["index_topk"] = args.index_topk
    if args.candidate_topk_blocks > 0:
        MICRO["candidate_topk_blocks"] = args.candidate_topk_blocks

    os.environ["V41_REF_NO_FAKE_QUANT"] = "1"
    os.environ["FASTLLM_DSV41_DISABLE_FAKE_QUANT"] = "1"

    ref = load_reference_module()
    ref.TINY.clear()
    ref.TINY.update(MICRO)

    tokenizer_dir = os.path.join(args.work_dir, "tokenizer")
    write_tiny_tokenizer(tokenizer_dir, MICRO["vocab_size"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    gen_args = types.SimpleNamespace(
        work_dir=args.work_dir, tokenizer_dir=tokenizer_dir, reference_dir=args.reference_dir,
        seed=args.seed, quant_format="bf16", decode=args.decode, dump_dir="",
    )
    engram_mod, model_mod = ref.install_shims(args.reference_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    ref.generate_checkpoint(gen_args, engram_mod, model_mod, tokenizer)

    # 随机权重下贪心 token 对微小数值差敏感，挑一个每步 top1/top2 间距都够大的提示词，
    # 让 fixture 不会因为 bf16 参考与 fp32 CPU 的舍入差异翻转 argmax。
    best = None
    for attempt in range(args.seed_tries):
        rng = np.random.default_rng(args.prompt_seed + attempt)
        # 3 以下是 bos / eos / pad，避开它们，让提示词都是普通 token
        prompt = [int(v) for v in rng.integers(3, MICRO["vocab_size"], size=args.prefill)]
        logits, tokens = ref.run_reference(gen_args, engram_mod, model_mod, tokenizer, prompt)
        logits = np.stack([t.numpy().astype(np.float32) for t in logits])
        top2 = np.sort(logits, axis=1)[:, -2:]
        margin = float((top2[:, 1] - top2[:, 0]).min())
        print("prompt seed %d: min top1-top2 margin %.4f" % (args.prompt_seed + attempt, margin))
        if best is None or margin > best[0]:
            best = (margin, prompt, logits, tokens)
        if margin >= args.min_margin:
            break
    margin, prompt, ref_logits, ref_tokens = best
    if margin < args.min_margin:
        print("WARNING: best min margin %.4f < %.4f" % (margin, args.min_margin))
    print("reference tokens:", ref_tokens)

    # 贪心解码不能撞上 eos，否则 fastllm 会提前停；必要时把 eos 换成模型不会产出的 id
    cfg_path = os.path.join(args.work_dir, "config.json")
    cfg = json.load(open(cfg_path))
    produced = set(int(t) for t in ref_tokens)
    if cfg["eos_token_id"] in produced:
        free = [i for i in range(MICRO["vocab_size"]) if i not in produced]
        assert free, "no free token id for eos"
        cfg["eos_token_id"] = free[-1]
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("eos_token_id remapped to", cfg["eos_token_id"])

    # 记录参考侧 top-1 与 top-2 的间距，测试用它判断"并列"
    top2 = np.sort(ref_logits, axis=1)[:, -2:]
    margins = (top2[:, 1] - top2[:, 0]).tolist()
    print("top1-top2 margins:", ["%.4f" % m for m in margins])

    meta = {
        "format": 1,
        "description": "DeepSeek-V4.1 micro model + official reference logits (CPU alignment fixture)",
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in MICRO.items()},
        "prompt": prompt,
        "decode_steps": args.decode,
        "ref_tokens": [int(t) for t in ref_tokens],
        "margins": margins,
        "files": {name: "f%d" % i for i, name in enumerate(FIXTURE_FILES)},
        "fastllm": {"dtype": "float32", "device": "cpu", "moe_device": "cpu",
                    "env": {"FASTLLM_DSV41_DISABLE_FAKE_QUANT": "1"}},
        "generator": "test/basic/deepseek_v41_fixture_gen.py",
    }
    arrays = {"meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
              "ref_logits": ref_logits,
              "ref_tokens": np.array(ref_tokens, dtype=np.int32)}
    for name, key in meta["files"].items():
        with open(os.path.join(args.work_dir, name), "rb") as f:
            arrays[key] = np.frombuffer(f.read(), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("fixture written to %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
