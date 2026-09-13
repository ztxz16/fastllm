#!/usr/bin/env python3
"""
DeepSeek-V4.1 的 CPU-only 对齐回归测试（可进 CI）。

与 ``deepseek_v41_reference.py`` 不同，本测试不需要 torch、transformers 和官方 inference 代码：
参考 logits 已经由官方实现算过一次并连同一个微型 checkpoint 一起固化在
``test/basic/deepseek_v41_fixture.npz`` 里（见 ``deepseek_v41_fixture_gen.py``）。
测试只依赖 numpy 与 fastllm 的 Python 包，跑纯 CPU 路径。

fixture 里的微型模型覆盖 V4.1 的全部结构特性：
  compress_ratios 0/1/2 三种层、跨层共享压缩 KV、indexer 两级 top-k（候选块 + top-k）、
  两个 Engram 层、Hyper-Connections、滑窗环形缓冲绕圈（window_size 远小于 prefill 长度）。

用法：
    PYTHONPATH=<fastllm>/build/tools python test/basic/test_deepseek_v41_cpu_fixture.py

退出码：0 = 通过（或 fixture 缺失而跳过），1 = 失败。
"""

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "deepseek_v41_fixture.npz")
GEN = "test/basic/deepseek_v41_fixture_gen.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cos", type=float, default=0.9975, help="每步 logits 余弦相似度下限")
    p.add_argument("--rel", type=float, default=0.04,
                   help="max|diff| 相对于参考 logits 值域（max-min）的上限")
    p.add_argument("--keep", action="store_true", help="保留解包出来的模型目录")
    p.add_argument("--device", default="", help="覆盖 fixture 记录的 device（调试用，默认 cpu）")
    p.add_argument("--moe-device", default="", help="覆盖 moe_device（调试用）")
    p.add_argument("--dtype", default="", help="覆盖 dtype（调试用）")
    p.add_argument("--save-logits", default="", help="把 fastllm 的 logits 存成 npy（调试用）")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def unpack(fixture_path, dst):
    data = np.load(fixture_path)
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    for name, key in meta["files"].items():
        with open(os.path.join(dst, name), "wb") as f:
            f.write(bytes(data[key]))
    return meta, data["ref_logits"], data["ref_tokens"]


def run_fastllm(model_dir, meta, threads, device="", moe_device="", dtype=""):
    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_ENGRAM_META"] = os.path.join(model_dir, "engram_meta.json")
    for k, v in meta["fastllm"].get("env", {}).items():
        os.environ[k] = v

    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    parser = make_normal_parser("v41 cpu fixture")
    argv = ["--path", model_dir, "--dtype", dtype or meta["fastllm"]["dtype"],
            "--device", device or meta["fastllm"]["device"],
            "--moe_device", moe_device or meta["fastllm"]["moe_device"],
            "-t", str(threads)]
    fargs = parser.parse_args(argv)
    if fargs.max_batch <= 0:
        fargs.max_batch = 1
    if fargs.tokens <= 0:
        fargs.tokens = 4096
    model = make_normal_llm_model(fargs)

    prompt = meta["prompt"]
    vocab = meta["config"]["vocab_size"]
    steps = meta["decode_steps"] + 1
    array = (ctypes.c_int * len(prompt))(*prompt)
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model, len(prompt), array, ctypes.c_int(steps), ctypes.c_int(0),
        ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0),
        ctypes.c_float(1.0), ctypes.c_bool(True), ctypes.c_int(0), None)
    buf = (ctypes.c_float * vocab)()
    logits, tokens = [], []
    while True:
        token_id = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
        if token_id < 0:
            break
        tokens.append(int(token_id))
        logits.append(np.ctypeslib.as_array(buf).copy())
    return np.stack(logits) if logits else np.zeros((0, vocab), np.float32), tokens


def compare(ref_logits, ref_tokens, fl_logits, fl_tokens, args, margins):
    failures = []
    steps = min(len(ref_logits), len(fl_logits))
    if len(fl_logits) != len(ref_logits):
        failures.append("step count: reference %d, fastllm %d" % (len(ref_logits), len(fl_logits)))
    # 参考侧是官方实现的 bf16 前向，fastllm 侧是 fp32 CPU 路径，绝对误差随 logits 的量级走，
    # 所以容差按每一步 logits 的值域（max - min）折算，跟词表大小与权重初始化无关。
    for step in range(steps):
        a = ref_logits[step].astype(np.float64)
        b = fl_logits[step].astype(np.float64)
        diff = np.abs(a - b)
        span = float(a.max() - a.min())
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        rel = float(diff.max()) / max(span, 1e-9)
        tol = args.rel * span
        stage = "prefill" if step == 0 else "decode%d" % step
        ok = True
        if cos < args.cos:
            failures.append("%s: cos %.6f < %.6f" % (stage, cos, args.cos))
            ok = False
        if rel > args.rel:
            failures.append("%s: max|diff| %.5f = %.2f%% of logits span %.4f (limit %.2f%%)" % (
                stage, diff.max(), 100 * rel, span, 100 * args.rel))
            ok = False
        ref_tok, fl_tok = int(ref_tokens[step]), int(fl_tokens[step])
        note = ""
        if ref_tok != fl_tok:
            # 并列处理：两个 token 在参考侧的 logits 差已落在数值容差内时，argmax 翻转不算错误。
            # fixture 生成时挑过提示词，保证每一步的 top1-top2 间距远大于容差，所以这里通常不会触发。
            gap = float(a[ref_tok] - a[fl_tok])
            if gap <= max(tol, 2.0 * float(diff.max())):
                note = " (tie flip, reference gap %.5f)" % gap
            else:
                failures.append("%s: token %d != reference %d (reference gap %.5f)" % (
                    stage, fl_tok, ref_tok, gap))
                ok = False
        print("%-9s cos=%.6f max|diff|=%.5f (%.2f%% of span %.3f, limit %.2f%%) "
              "ref_top1=%d fl_top1=%d margin=%.4f %s%s" % (
                  stage, cos, diff.max(), 100 * rel, span, 100 * args.rel,
                  ref_tok, fl_tok, margins[step], "OK" if ok else "FAIL", note))
        if not ok:
            break
    return failures


def main():
    args = parse_args()
    if not os.path.exists(args.fixture):
        print("SKIP: fixture %s not found." % args.fixture)
        print("      它是一个约 2.3 MB 的 npz（微型 V4.1 模型 + 官方参考 logits）。")
        print("      用装有 torch 与官方 inference 代码的机器重新生成：")
        print("        PYTHONPATH=build/tools python %s \\" % GEN)
        print("          --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \\")
        print("          --work-dir /tmp/v41-micro --out %s" % args.fixture)
        return 0

    work = tempfile.mkdtemp(prefix="v41-fixture-")
    try:
        meta, ref_logits, ref_tokens = unpack(args.fixture, work)
        cfg = meta["config"]
        print("fixture: %s (%.2f MB)" % (args.fixture, os.path.getsize(args.fixture) / 1e6))
        print("model: %d layers, dim %d, ratios %s, kv sources %s, index sources %s, "
              "candidate layer %d, engram layers %s, hc %d, window %d" % (
                  cfg["n_layers"], cfg["dim"], cfg["compress_ratios"], cfg["kv_source_layers"],
                  cfg["index_source_layers"], cfg["candidate_source_layer"],
                  cfg["engram_layer_ids"], cfg["hc_mult"], cfg["window_size"]))
        print("prompt %d tokens, %d decode steps, fastllm %s/%s" % (
            len(meta["prompt"]), meta["decode_steps"], meta["fastllm"]["device"],
            meta["fastllm"]["dtype"]))
        fl_logits, fl_tokens = run_fastllm(work, meta, args.threads,
                                           args.device, args.moe_device, args.dtype)
        if args.save_logits:
            np.save(args.save_logits, fl_logits)
        failures = compare(ref_logits, ref_tokens, fl_logits, fl_tokens, args, meta["margins"])
    finally:
        if args.keep:
            print("model dir kept at", work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
