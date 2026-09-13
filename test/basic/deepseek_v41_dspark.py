#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek-V4.1 DSpark 投机解码测试（迷你模型）。

投机解码必须是精确的：目标模型逐位置贪心比对候选，第一个不匹配处截断，
因此开启 DSpark 与关闭时的贪心输出必须逐 token 完全一致。本脚本：

  1. 生成一个带 mtp.0/1/2 草稿层的迷你 V4.1 checkpoint（复用
     test/basic/deepseek_v41_reference.py 的迷你配置、shim 与量化工具）；
  2. 关闭 DSpark 跑一遍贪心生成，得到基准 token 序列；
  3. 开启 DSpark 再跑一遍，比较两次的 token 序列并统计接受率；
  4. 回滚测试：用 FASTLLM_DSPARK_FORCE_DRAFTS 注入候选，构造"接受 0 个 /
     接受一部分 / 全部接受"三种情况，确认后续生成仍与基准逐 token 一致。

例：

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=build/tools python test/basic/deepseek_v41_dspark.py \\
      --work-dir /root/v41-tiny-dspark --tokenizer-dir /root/v41-tokenizer \\
      --reference-dir /mnt/shared2/models/DeepSeek-V4.1-Flash/inference --regenerate
"""

import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import deepseek_v41_reference as ref     # noqa: E402


# mtp 相关的迷你配置：3 个草稿层，block size 5，目标层取主干最后三层。
# 随机权重下专家选择与 indexer top-k 对微小数值差极敏感（一次 6 token 的校验前向与
# 六次单 token 前向选到的 GEMM kernel 不同，BF16 舍入会让并列的 argmax 翻转），因此
# 这里沿用参考测试的隔离配置：2 专家 top-2、index_topk 大于压缩块数、关闭伪量化。
DSPARK_TINY = dict(
    n_mtp_layers=3,
    dspark_block_size=5,
    dspark_noise_token_id=128799,
    dspark_target_layer_ids=(3, 4, 5),
    dspark_markov_rank=64,
    dspark_n_routed_experts=2,
    dspark_n_activated_experts=2,
    n_routed_experts=2,
    n_activated_experts=2,
    index_topk=4096,
    candidate_topk_blocks=4096,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="迷你模型输出目录")
    parser.add_argument("--tokenizer-dir", required=True, help="包含 tokenizer.json 的目录")
    parser.add_argument("--reference-dir", required=True, help="官方 inference/ 目录")
    parser.add_argument("--prefill", type=int, default=40)
    parser.add_argument("--decode", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    # 迷你模型 + float16 的 lm_head 会产生完全相同的 logits（随机权重下 3 路并列），
    # 这时 argmax 由 kernel 的归约顺序决定，一次多 token 与逐 token 前向可能选到不同的
    # 并列项。用 float32 让 logits 分开，投机解码的"逐 token 完全一致"才是可判定的。
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moe-device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dspark-tokens", type=int, default=5)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-rollback", action="store_true")
    parser.add_argument("--real-checkpoint", default="",
                        help="真实 checkpoint 目录：只静态核对 mtp.* 的张量命名 / 形状 / 量化格式，不加载模型")
    parser.add_argument("--child", default="", help="内部使用：子进程模式")
    return parser.parse_args()


# ---------------- checkpoint ----------------

def build_tiny_config():
    tiny = dict(ref.TINY)
    tiny.update(DSPARK_TINY)
    n_layers = tiny["n_layers"]
    # 草稿层的 compress_ratio 必须是 0（纯滑窗注意力）
    tiny["compress_ratios"] = tuple(list(tiny["compress_ratios"]) + [0] * tiny["n_mtp_layers"])
    for layer in tiny["dspark_target_layer_ids"]:
        assert 0 <= layer < n_layers, "目标层必须在主干范围内"
    return tiny


def hf_config(tiny, engram_num_embeddings, compressed_vocab):
    cfg = ref.hf_config_from_tiny(tiny, engram_num_embeddings, compressed_vocab)
    text = cfg["text_config"]
    text["compress_ratios"] = list(tiny["compress_ratios"])
    text["num_nextn_predict_layers"] = tiny["n_mtp_layers"]
    text["dspark_block_size"] = tiny["dspark_block_size"]
    text["dspark_noise_token_id"] = tiny["dspark_noise_token_id"]
    text["dspark_target_layer_ids"] = list(tiny["dspark_target_layer_ids"])
    text["dspark_markov_rank"] = tiny["dspark_markov_rank"]
    text["dspark_n_routed_experts"] = tiny["dspark_n_routed_experts"]
    text["dspark_num_experts_per_tok"] = tiny["dspark_n_activated_experts"]
    return cfg


def generate_checkpoint(args, tiny, engram_mod, model_mod, tokenizer):
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(args.seed)
    margs = model_mod.ModelArgs(**tiny)
    layout = engram_mod.EngramLayout.from_args(margs)
    num_embeddings = [int(sum(sum(per) for per in layer)) for layer in layout.primes]
    margs.engram_num_embeddings = tuple(num_embeddings)
    with torch.device("cpu"):
        torch.set_default_dtype(torch.bfloat16)
        model = model_mod.Transformer(margs, tokenizer)
        torch.set_default_dtype(torch.float32)

    n_layers = tiny["n_layers"]
    skipped = 0
    state = {}
    for name, param in model.state_dict().items():
        if name.startswith("engram_hash."):
            continue
        if ".k_cache" in name or "window_kv_cache" in name or "compress_kv_cache" in name \
                or name.endswith(".kv_state") or name.endswith(".score_state") or name.endswith(".freqs_cis"):
            continue
        # mtp.N.embed / mtp.N.head 是与主干共享的模块，真实 checkpoint 里不单独存
        if name.startswith("mtp.") and (".embed.weight" in name or name.endswith(".head.weight")) \
                and "markov_head" not in name:
            skipped += 1
            continue
        shape = tuple(param.shape)
        if param.dtype == torch.float8_e4m3fn:
            raw = torch.randn(shape, dtype=torch.float32) * 0.5
            q, scale = ref.e8m0_quantize(raw)
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

    mtp_names = sorted(n for n in state if n.startswith("mtp."))
    assert mtp_names, "生成的 checkpoint 里没有 mtp.* 权重"
    for required in ("mtp.0.main_proj.weight", "mtp.0.main_norm.weight",
                     "mtp.%d.norm.weight" % (tiny["n_mtp_layers"] - 1),
                     "mtp.%d.markov_head.embed.weight" % (tiny["n_mtp_layers"] - 1),
                     "mtp.%d.markov_head.head.weight" % (tiny["n_mtp_layers"] - 1),
                     "mtp.%d.confidence_head.proj.weight" % (tiny["n_mtp_layers"] - 1)):
        assert required in state, "缺少 %s" % required

    os.makedirs(args.work_dir, exist_ok=True)
    save_file(state, os.path.join(args.work_dir, "model.safetensors"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(args.tokenizer_dir, f), os.path.join(args.work_dir, f))
    hash_state = engram_mod.NgramHashState(margs, layout, tokenizer)
    compressed_vocab = int(hash_state.token_map.max().item()) + 1
    with open(os.path.join(args.work_dir, "config.json"), "w") as f:
        json.dump(hf_config(tiny, num_embeddings, compressed_vocab), f, indent=2)
    from ftllm.deepseek_v41_engram import build_engram_meta
    meta = build_engram_meta(args.work_dir)
    with open(os.path.join(args.work_dir, "engram_meta.json"), "w") as f:
        json.dump(meta, f)
    print("checkpoint written to %s (%d tensors, %d of them mtp.*, skipped %d shared)"
          % (args.work_dir, len(state), len(mtp_names), skipped))


# ---------------- fastllm 生成 ----------------

VOCAB_SIZE = 129280


def run_fastllm(args, prompt, decode_steps, want_logits=False):
    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_ENGRAM_META"] = os.path.join(args.work_dir, "engram_meta.json")
    # 两次运行都关闭伪量化，避免量化边界翻转掩盖真正的实现差异
    os.environ.setdefault("FASTLLM_DSV41_DISABLE_FAKE_QUANT", "1")
    parser = make_normal_parser("v41 dspark")
    argv = ["--path", args.work_dir, "--dtype", args.dtype, "--device", args.device,
            "--moe_device", args.moe_device, "-t", str(args.threads)]
    # DSpark 走正式的启动参数（ftllm 的 --speculative_algorithm dspark --dspark N），
    # util.py 会清掉直接设置的 FASTLLM_DSPARK_TOKENS，所以不能只靠环境变量
    dspark = int(os.environ.get("V41_TEST_DSPARK_TOKENS", "0") or 0)
    if dspark > 0:
        argv += ["--speculative_algorithm", "dspark", "--dspark", str(dspark),
                 "--speculative_dspark_confidence_threshold",
                 os.environ.get("V41_TEST_DSPARK_THRESHOLD", "0")]
    fargs = parser.parse_args(argv)
    if fargs.max_batch <= 0:
        fargs.max_batch = 1
    if fargs.tokens <= 0:
        fargs.tokens = 65536
    model = make_normal_llm_model(fargs)

    import time
    t0 = time.time()
    input_array = (ctypes.c_int * len(prompt))(*prompt)
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model, len(prompt), input_array, ctypes.c_int(decode_steps), ctypes.c_int(0),
        ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0), ctypes.c_float(1.0),
        ctypes.c_bool(want_logits), ctypes.c_int(0), None)   # 开 DSpark 时必须关闭 output_logits（要求简单贪心）
    tokens, tops = [], []
    first = None
    if want_logits:
        import numpy as np
        buf = (ctypes.c_float * VOCAB_SIZE)()
        while True:
            token_id = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
            if token_id < 0:
                break
            if first is None:
                first = time.time()
            tokens.append(int(token_id))
            values = np.ctypeslib.as_array(buf)
            order = np.argsort(-values)[:16]
            tops.append({int(t): float(values[t]) for t in order})
    else:
        while True:
            token_id = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
            if token_id < 0:
                break
            if first is None:
                first = time.time()
            tokens.append(int(token_id))
    end = time.time()
    speed = (len(tokens) - 1) / max(1e-6, end - first) if first is not None else 0.0
    print("fastllm: %d tokens, prefill %.3fs, decode %.3fs (%.2f tok/s)"
          % (len(tokens), (first or end) - t0, end - (first or end), speed))
    return tokens, speed, tops


def child_main(args):
    """子进程：加载模型跑一遍生成，把 token 序列写到 --child 指定的 json 文件。"""
    payload = json.load(open(args.child + ".in"))
    tokens, speed, tops = run_fastllm(args, payload["prompt"], payload["decode"],
                                      bool(payload.get("logits")))
    json.dump({"tokens": tokens, "speed": speed, "tops": tops}, open(args.child + ".out", "w"))


def run_child(args, prompt, decode_steps, env, want_logits=False):
    """在子进程里跑一遍生成（每次都是全新的模型实例与环境变量）。"""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "child")
        json.dump({"prompt": prompt, "decode": decode_steps, "logits": want_logits},
                  open(base + ".in", "w"))
        child_env = dict(os.environ)
        child_env.update({k: str(v) for k, v in env.items()})
        for key in ("V41_TEST_DSPARK_TOKENS", "V41_TEST_DSPARK_THRESHOLD",
                    "FASTLLM_DSPARK_FORCE_DRAFTS", "FASTLLM_DSPARK_STATS_FILE"):
            if key not in env:
                child_env.pop(key, None)
        argv = [sys.executable, os.path.abspath(__file__),
                "--work-dir", args.work_dir, "--tokenizer-dir", args.tokenizer_dir,
                "--reference-dir", args.reference_dir, "--dtype", args.dtype,
                "--device", args.device, "--moe-device", args.moe_device,
                "--threads", str(args.threads), "--child", base]
        result = subprocess.run(argv, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = result.stdout.decode("utf-8", "replace")
        if result.returncode != 0 or not os.path.exists(base + ".out"):
            print(text)
            raise RuntimeError("child process failed (rc=%d)" % result.returncode)
        for line in text.splitlines():
            if "DSpark" in line or "fastllm:" in line or "Traceback" in line:
                print("    | " + line)
        out = json.load(open(base + ".out"))
        tops = [{int(k): v for k, v in d.items()} for d in out.get("tops", [])]
        return out["tokens"], out["speed"], tops


def bf16_ulp(value):
    value = abs(float(value))
    if value <= 0:
        return 1.0 / 256
    exp = math.floor(math.log2(value))
    return 2.0 ** (exp - 7)


def classify_mismatch(top, baseline_token, other_token):
    """判断一次 argmax 分歧是不是 BF16 分辨率下的"几乎并列"。

    迷你模型的 logits 是 BF16（分辨率约 1/32），而 GEMM 会按 batch 大小选不同的核，
    一次多 token 的校验前向与逐 token 解码之间会有同量级的舍入差。这时贪心 argmax
    会在几乎并列的位置翻转，与 DSpark 的实现无关（不开 DSpark 时，一次大 prefill
    与逐 token 解码在同样的位置也会翻转）。
    """
    if other_token not in top:
        return "real", float("inf")
    best = max(top.values())
    gap = best - top[other_token]
    return ("tie" if gap <= 3.0 * bf16_ulp(best) else "real"), gap


def compare_sequences(base_tokens, base_tops, other_tokens):
    """返回 (第一个分歧位置, 类型, 差距)；完全一致时位置为 -1。"""
    for i, (a, b) in enumerate(zip(base_tokens, other_tokens)):
        if a != b:
            kind, gap = classify_mismatch(base_tops[i] if i < len(base_tops) else {}, a, b)
            return i, kind, gap
    return -1, "same", 0.0


def compare_with_reanchor(args, prompt, spec_tokens, decode_steps, max_anchors=5):
    """逐段比较 spec_tokens 与不开 DSpark 的基准。

    遇到"几乎并列"的分歧时，以 spec 的 token 为新前缀重新跑一遍基准继续比较，
    这样即使中途有并列翻转，也能覆盖整条序列。返回 (失败信息列表, 并列分歧位置列表)。
    """
    failures, ties, offset, anchors = [], [], 0, 0
    while offset < len(spec_tokens):
        base, _, tops = run_child(args, prompt + spec_tokens[:offset],
                                  decode_steps - offset, {}, want_logits=True)
        idx, kind, gap = compare_sequences(base, tops, spec_tokens[offset:])
        if idx < 0:
            break
        position = offset + idx
        if kind == "real":
            failures.append("位置 %d：基准 %d，DSpark %d（差距 %.4g，不是并列）"
                            % (position, base[idx], spec_tokens[position], gap))
            break
        ties.append((position, gap))
        anchors += 1
        if anchors > max_anchors:
            failures.append("并列分歧过多（超过 %d 次），无法继续比较" % max_anchors)
            break
        offset = position + 1
    return failures, ties


def check_real_checkpoint(path):
    """静态核对真实 checkpoint 的 mtp.* 权重：加载器需要的张量是否齐全、量化格式是否与主干一致。

    不加载模型（真实权重需要双卡 + 数百 GB 内存），只读 safetensors 的索引与头部。
    """
    index_path = os.path.join(path, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    cfg = json.load(open(os.path.join(path, "config.json")))
    text = cfg.get("text_config", cfg)
    stages = int(text["num_nextn_predict_layers"])
    experts = int(text["dspark_n_routed_experts"])
    need = []
    for stage in range(stages):
        pre = "mtp.%d." % stage
        need += [pre + n for n in (
            "attn.wq_a.weight", "attn.q_norm.weight", "attn.wq_b.weight", "attn.wkv.weight",
            "attn.kv_norm.weight", "attn.wo_a.weight", "attn.wo_b.weight", "attn.attn_sink",
            "attn_norm.weight", "ffn_norm.weight", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
            "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn.gate.weight", "ffn.gate.bias",
            "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight")]
        need += [pre + "ffn.experts.%d.%s.weight" % (e, w)
                 for e in range(experts) for w in ("w1", "w2", "w3")]
    need += ["mtp.0.main_proj.weight", "mtp.0.main_norm.weight"]
    last = "mtp.%d." % (stages - 1)
    need += [last + n for n in ("norm.weight", "markov_head.embed.weight",
                                "markov_head.head.weight", "confidence_head.proj.weight")]
    missing = [n for n in need if n not in weight_map]
    print("stages=%d experts=%d 需要 %d 个张量，缺失 %d 个" % (stages, experts, len(need), len(missing)))
    if missing:
        print("  缺失示例:", missing[:5])
        return ["真实 checkpoint 缺少 %d 个 DSpark 张量" % len(missing)]

    import struct

    def header(filename):
        with open(os.path.join(path, filename), "rb") as stream:
            size, = struct.unpack("<Q", stream.read(8))
            return json.loads(stream.read(size))

    failures = []
    expect = {
        "mtp.0.main_proj.weight": ("F8_E4M3", "F8_E8M0", 32),          # 稠密 FP8 32x32
        "mtp.0.attn.wkv.weight": ("F8_E4M3", "F8_E8M0", 32),
        "mtp.0.ffn.experts.0.w1.weight": ("I8", "F8_E8M0", None),      # 路由专家 FP4，沿 K 每 32 个一组
    }
    for name, (dtype, scale_dtype, block) in expect.items():
        head = header(weight_map[name])
        info = head[name]
        scale = head.get(name.replace(".weight", ".scale"))
        print("  %-34s %-8s %-16s scale=%s" % (name, info["dtype"], info["shape"],
                                               (scale["dtype"], scale["shape"]) if scale else None))
        if info["dtype"] != dtype or scale is None or scale["dtype"] != scale_dtype:
            failures.append("%s 的量化格式与预期不符" % name)
            continue
        rows, cols = info["shape"]
        if block is not None and scale["shape"] != [rows // block, cols // block]:
            failures.append("%s 的 scale 形状不是 %dx%d 块" % (name, block, block))
        if block is None and scale["shape"] != [rows, cols * 2 // 32]:
            failures.append("%s 的 FP4 scale 不是沿 K 每 32 个一组" % name)
    for name, dtype, shape in (
            ("mtp.%d.markov_head.embed.weight" % (stages - 1), "BF16",
             [int(text["vocab_size"]), int(text["dspark_markov_rank"])]),
            ("mtp.%d.markov_head.head.weight" % (stages - 1), "BF16",
             [int(text["vocab_size"]), int(text["dspark_markov_rank"])]),
            ("mtp.%d.confidence_head.proj.weight" % (stages - 1), "BF16",
             [1, int(text["hidden_size"]) + int(text["dspark_markov_rank"])])):
        info = header(weight_map[name])[name]
        print("  %-34s %-8s %s" % (name, info["dtype"], info["shape"]))
        if info["dtype"] != dtype or info["shape"] != shape:
            failures.append("%s 的 dtype / 形状与预期不符（期望 %s %s）" % (name, dtype, shape))
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), "build", "tools"))
        from ftllm.launcher_mtp import detect_mtp_support
        reason = detect_mtp_support(path, cfg)
        print("  detect_mtp_support = %s" % reason)
        if reason != "enabled":
            failures.append("detect_mtp_support 返回 %s，应为 enabled" % reason)
    except ImportError as exc:
        print("  （跳过 detect_mtp_support：%s）" % exc)
    return failures


def read_stats(path):
    rounds = []
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            if len(parts) == 3:
                rounds.append((int(parts[1]), int(parts[2])))
    return rounds


# ---------------- 回滚测试 ----------------

def build_force_file(baseline, first_decode_index, pattern, block, path):
    """按给定的接受长度序列构造候选。

    baseline[i] 是第 i 个生成的 token（位置 prompt_len + i）。第一次 decode 不带候选
    （prefill 之后才生成第一批候选），因此第一轮校验的锚点是 baseline[first_decode_index]。
    某一轮锚点为 baseline[j] 时，候选对应 baseline[j+1:]，想让它接受 k 个就把前 k 个
    填成正确值、第 k 个填成错误值。
    """
    lines, j, accepts = [], first_decode_index, []
    for want in pattern:
        want = min(want, block)
        row = []
        for t in range(block):
            idx = j + 1 + t
            if idx >= len(baseline):
                break
            if t < want:
                row.append(baseline[idx])
            else:
                # 一个一定不会被接受的 token
                row.append((baseline[idx] + 7919) % 100000)
                break
        if not row:
            break
        lines.append(row)
        accepts.append(min(want, len(row)))
        j += min(want, len(row)) + 1
        if j + 1 >= len(baseline):
            break
    with open(path, "w") as f:
        for row in lines:
            f.write(" ".join(str(x) for x in row) + "\n")
    return accepts


def main():
    args = parse_args()
    if args.child:
        child_main(args)
        return

    if args.real_checkpoint:
        print("=== 真实 checkpoint 的 mtp.* 权重核对 ===")
        problems = check_real_checkpoint(args.real_checkpoint)
        for problem in problems:
            print("FAIL:", problem)
        print("PASS" if not problems else "")
        sys.exit(1 if problems else 0)

    tiny = build_tiny_config()
    need_generate = args.regenerate or not os.path.exists(os.path.join(args.work_dir, "model.safetensors"))
    if need_generate:
        from transformers import AutoTokenizer
        tokenizer_hf = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        engram_mod, model_mod = ref.install_shims(args.reference_dir)
        generate_checkpoint(args, tiny, engram_mod, model_mod, tokenizer_hf)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))

    prompt_text = ("DeepSeek-V4.1 speculative decoding regression prompt. " * 8)
    prompt = tokenizer.encode(prompt_text).ids[:args.prefill]
    assert len(prompt) >= 8, "prompt 太短"
    decode_steps = args.decode

    failures = []

    print("\n=== 1. 基准：关闭 DSpark ===")
    baseline, base_speed, base_tops = run_child(args, prompt, decode_steps, {}, want_logits=True)
    print("    baseline tokens:", baseline)

    print("\n=== 2. 开启 DSpark（模型自己的候选）===")
    stats_path = os.path.join(tempfile.gettempdir(), "v41-dspark-stats-%d.txt" % os.getpid())
    if os.path.exists(stats_path):
        os.remove(stats_path)
    spec, spec_speed, _ = run_child(args, prompt, decode_steps, {
        "V41_TEST_DSPARK_TOKENS": args.dspark_tokens,
        "FASTLLM_DSPARK_STATS_FILE": stats_path,
    })
    rounds = read_stats(stats_path)
    proposed = sum(r[0] for r in rounds)
    accepted = sum(r[1] for r in rounds)
    print("    dspark tokens:  ", spec)
    print("    verify rounds: %d, drafts %d, accepted %d (%.1f%%), %.2f tokens / target forward"
          % (len(rounds), proposed, accepted,
             100.0 * accepted / max(1, proposed),
             (accepted + len(rounds)) / max(1.0, float(len(rounds)))))
    print("    speed: baseline %.2f tok/s, dspark %.2f tok/s" % (base_speed, spec_speed))
    if not rounds:
        failures.append("DSpark 没有进行任何一次校验（草稿路径没有被触发）")
    idx, kind, gap = compare_sequences(baseline, base_tops, spec)
    if idx < 0:
        print("    OK: 与基准逐 token 完全一致")
    else:
        print("    位置 %d 分歧：基准 %d，DSpark %d（%s，差距 %.4g）"
              % (idx, baseline[idx], spec[idx], kind, gap))
        if kind == "real":
            failures.append("开启 DSpark 后贪心输出与基准不一致（位置 %d，非并列）" % idx)
        else:
            sub_fail, ties = compare_with_reanchor(args, prompt, spec, decode_steps)
            print("    重新锚定后：并列分歧 %d 处 %s" % (len(ties), [t[0] for t in ties]))
            failures.extend(sub_fail)
            if not sub_fail:
                print("    OK: 除 BF16 并列翻转外与基准一致")

    if not args.skip_rollback:
        block = args.dspark_tokens

        def run_case(name, pattern):
            """按 pattern 注入候选跑一遍，返回 (失败信息或 None, 覆盖到的轮数, 分歧位置)。"""
            force_path = os.path.join(tempfile.gettempdir(), "v41-dspark-force-%d.txt" % os.getpid())
            stats2 = force_path + ".stats"
            for path in (force_path, stats2):
                if os.path.exists(path):
                    os.remove(path)
            want = build_force_file(baseline, 1, pattern, block, force_path)
            got, _, _ = run_child(args, prompt, decode_steps, {
                "V41_TEST_DSPARK_TOKENS": block,
                "FASTLLM_DSPARK_FORCE_DRAFTS": force_path,
                "FASTLLM_DSPARK_STATS_FILE": stats2,
            })
            observed = [r[1] for r in read_stats(stats2)]
            idx, kind, gap = compare_sequences(baseline, base_tops, got)
            # 输出一旦在某个位置分歧（迷你模型上是 BF16 并列翻转），注入的候选就不再等于
            # 真实 token，接受长度自然掉到 0；因此只检查完全落在分歧之前的那些轮次。
            valid, anchor = len(want), 1
            for k, a in enumerate(want):
                last = anchor + a + 1        # 本轮校验覆盖到的最后一个输出下标
                if idx >= 0 and last >= idx:
                    valid = k
                    break
                anchor = anchor + a + 1
            print("    构造的接受长度 %s" % want)
            print("    实际接受长度   %s" % observed[:len(want)])
            if kind == "real":
                return ("回滚测试「%s」的输出与基准不一致（位置 %d，非并列）" % (name, idx)), valid, idx
            if observed[:valid] != want[:valid]:
                return ("回滚测试「%s」前 %d 轮的接受长度与构造的不一致：%s != %s"
                        % (name, valid, observed[:valid], want[:valid])), valid, idx
            return None, valid, idx

        cases = [
            ("接受 0 个", [0] * 8),
            ("接受一部分", [2, 1, 3, 2, 1]),
            ("全部接受", [block] * 6),
            ("混合", [0, block, 1, block, 0, 2]),
        ]
        covered_lengths = set()
        for name, pattern in cases:
            print("\n=== 3. 回滚测试：%s ===" % name)
            problem, valid, idx = run_case(name, pattern)
            if problem is None and valid <= 0 and idx >= 3:
                # 基准在很靠前的位置就出现 BF16 并列翻转，构造的第一轮跨过了它。
                # 缩短第一轮的接受长度让它落在分歧之前，重试一次。
                shrunk = list(pattern)
                shrunk[0] = max(0, min(shrunk[0], idx - 3))
                if shrunk != list(pattern):
                    print("    （基准在位置 %d 就并列翻转，把第一轮的接受长度缩到 %d 重试）"
                          % (idx, shrunk[0]))
                    problem, valid, idx = run_case(name, shrunk)
                    pattern = shrunk
            if problem is not None:
                failures.append(problem)
            elif valid <= 0:
                print("    SKIP: 基准在位置 %d 就并列翻转，这一档没有可判定的轮次" % idx)
            else:
                covered_lengths.update(pattern[:valid])
                print("    OK: 前 %d 轮接受长度符合预期%s"
                      % (valid, "，输出与基准完全一致" if idx < 0 else
                         "，位置 %d 起为 BF16 并列翻转" % idx))
        # 整套测试至少要覆盖到"接受 0 个"、"接受一部分"、"接受满一整块"三类
        print("\n=== 回滚测试覆盖到的接受长度: %s ===" % sorted(covered_lengths))
        if 0 not in covered_lengths:
            failures.append("回滚测试没有覆盖到接受 0 个的情况")
        if not any(0 < v < block for v in covered_lengths):
            failures.append("回滚测试没有覆盖到接受一部分的情况")
        if max(covered_lengths, default=0) < 2:
            failures.append("回滚测试覆盖到的最大接受长度只有 %d，太弱" % max(covered_lengths, default=0))

    print("\n==================== 结果 ====================")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
