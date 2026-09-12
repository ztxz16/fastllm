#!/usr/bin/env python3
# coding: utf-8
"""DeepSeek-V4.1 的 --kv_cache_dtype 三档存储（bf16 / fp8_e4m3 / fp4_e2m1）对照测试。

要点：压缩 KV 与 indexer key 在写入 cache 之前已经过伪量化（压缩 KV 是 FP4 E2M1 + 每 16 个
一组的 E4M3 scale，indexer key 是 FP4 E2M1 + 每 32 个一组的 UE8M0 scale），值本来就落在 FP4
网格上，因此按 FP4 存储应当逐位无损 —— 本脚本用「与 BF16 存储的 logits 逐 bit 相同」来证明。

每一档都在独立子进程里跑（kvCacheDataType 在建模型时固定），logits 存成 .npy 后逐 bit 比较。
同时打印每一档实际占用的长期 KV 字节数（由 FASTLLM_DSV41_KV_STATS=1 输出）。

用法：
    python test/basic/deepseek_v41_kv_cache_dtype.py \
        --work-dir /root/v41-tiny2 --tokenizer-dir /root/v41-tokenizer
"""

import argparse
import ctypes
import os
import re
import subprocess
import sys

VOCAB_SIZE = 129280
DEFAULT_DTYPES = "bf16,fp8_e4m3,fp4_e2m1"

PROMPT_TEXT = ("DeepSeek-V4.1-Flash is the first model of a new architecture family. It combines sliding window "
               "attention, cross-layer compressed KV sharing, a two-level sparse indexer, engram n-gram memory and "
               "hyper-connections. 这是一个用于对齐测试的中英文混合提示词，包含数字 12345 与符号 !@#。")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--dtypes", default=DEFAULT_DTYPES, help="逗号分隔的 kv_cache_dtype 列表，第一个作为基准")
    p.add_argument("--prefill", type=int, default=200)
    p.add_argument("--decode", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--moe-device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--chunked-prefill", type=int, default=-1)
    p.add_argument("--no-fake-quant", action="store_true",
                   help="关闭伪量化。此时 FP4 存储是有损的，只检查 argmax 是否一致")
    p.add_argument("--index-topk", type=int, default=-1)
    p.add_argument("--out-dir", default="")
    # 子进程用
    p.add_argument("--single", default="", help="内部使用：只跑一档并把 logits 写到 --out")
    p.add_argument("--out", default="")
    return p.parse_args()


def build_prompt(tokenizer_dir, prefill):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    text = PROMPT_TEXT * max(4, prefill // 60 + 1)
    return tok.encode(text).ids[:prefill]


def run_single(args, prompt):
    import numpy as np
    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_ENGRAM_META"] = os.path.join(args.work_dir, "engram_meta.json")
    if args.no_fake_quant:
        os.environ["FASTLLM_DSV41_DISABLE_FAKE_QUANT"] = "1"
    parser = make_normal_parser("v41 kv cache dtype")
    argv = ["--path", args.work_dir, "--dtype", args.dtype, "--device", args.device,
            "--moe_device", args.moe_device, "-t", str(args.threads)]
    if args.chunked_prefill > 0:
        argv += ["--chunked_prefill_size", str(args.chunked_prefill)]
    if args.single != "bf16":
        argv += ["--kv_cache_dtype", args.single]
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
        model.model, len(prompt), input_array, ctypes.c_int(args.decode + 1), ctypes.c_int(0),
        ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0), ctypes.c_float(1.0),
        ctypes.c_bool(True), ctypes.c_int(0), None)
    buf = (ctypes.c_float * VOCAB_SIZE)()
    logits, tokens = [], []
    first = None
    while True:
        token_id = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
        if token_id < 0:
            break
        if first is None:
            first = time.time()
        tokens.append(int(token_id))
        logits.append(np.ctypeslib.as_array(buf).copy())
    end = time.time()
    np.save(args.out, np.stack(logits))
    print("TOKENS: " + " ".join(str(t) for t in tokens))
    if first is not None and len(tokens) > 1:
        print("TIMING: prefill %.3f s, decode %.2f ms/token" % (
            first - t0, 1000.0 * (end - first) / (len(tokens) - 1)))
    return 0


def main():
    args = parse_args()
    if args.single:
        return run_single(args, build_prompt(args.tokenizer_dir, args.prefill))

    import numpy as np
    out_dir = args.out_dir or os.path.join("/tmp", "v41_kv_dtype")
    os.makedirs(out_dir, exist_ok=True)
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    prompt = build_prompt(args.tokenizer_dir, args.prefill)
    print("prompt tokens:", len(prompt))

    results = {}
    for dt in dtypes:
        out = os.path.join(out_dir, "logits_%s.npy" % dt)
        cmd = [sys.executable, os.path.abspath(__file__), "--work-dir", args.work_dir,
               "--tokenizer-dir", args.tokenizer_dir, "--prefill", str(args.prefill),
               "--decode", str(args.decode), "--device", args.device, "--moe-device", args.moe_device,
               "--dtype", args.dtype, "--threads", str(args.threads),
               "--single", dt, "--out", out]
        if args.chunked_prefill > 0:
            cmd += ["--chunked-prefill", str(args.chunked_prefill)]
        if args.no_fake_quant:
            cmd += ["--no-fake-quant"]
        env = dict(os.environ, FASTLLM_DSV41_KV_STATS="1")
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = proc.stdout.decode("utf-8", "replace")
        if proc.returncode != 0:
            print(text)
            print("RESULT: FAIL (%s exited %d)" % (dt, proc.returncode))
            return 1
        tokens, timing = [], ""
        bytes_per_token, longterm = None, None
        for line in text.splitlines():
            if line.startswith("TOKENS:"):
                tokens = [int(x) for x in line.split()[1:]]
            if line.startswith("TIMING:"):
                timing = line[len("TIMING:"):].strip()
            m = re.search(r"tokens=(\d+) compressedKV=(\d+) B indexK=(\d+) B \(long-term (\d+) B, ([0-9.]+) B/token\)", line)
            if m:
                longterm = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(1)))
                bytes_per_token = float(m.group(5))
        results[dt] = dict(logits=np.load(out), tokens=tokens, bpt=bytes_per_token, lt=longterm, timing=timing)
        print("[%s] tokens=%s bytes/token=%s  %s" % (dt, tokens, bytes_per_token, timing))
        if longterm:
            print("        compressedKV=%d B  indexK=%d B  long-term=%d B  over %d tokens" % longterm)

    base = dtypes[0]
    ok = True
    for dt in dtypes[1:]:
        a, b = results[base]["logits"], results[dt]["logits"]
        n = min(a.shape[0], b.shape[0])
        identical = a.shape == b.shape and np.array_equal(a[:n].view(np.int32), b[:n].view(np.int32))
        maxdiff = float(np.abs(a[:n] - b[:n]).max())
        same_tokens = results[base]["tokens"] == results[dt]["tokens"]
        argmax_same = bool((a[:n].argmax(axis=1) == b[:n].argmax(axis=1)).all())
        cos = float(np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0]) + 1e-9))
        print("%-10s vs %-10s: bit-identical=%s max|diff|=%.6g step0-cos=%.6f same_tokens=%s argmax=%s" % (
            dt, base, identical, maxdiff, cos, same_tokens, argmax_same))
        # FP4 存储：压缩 KV 与 indexer key 写入前已伪量化到同一张 FP4 网格上，因此必须逐 bit 相同。
        # FP8 存储：压缩 KV 的 FP4 网格落不到 FP8 的 pow2 块 scale 上，本来就是有损的（见 docs），
        #           迷你模型是随机权重，argmax 对微小差极敏感，这里只作参考不作判据。
        if dt.startswith("fp4"):
            if args.no_fake_quant:
                ok = ok and cos > 0.99          # 关伪量化时 FP4 存储确实有损
            else:
                ok = ok and identical and same_tokens
        # FP8 只作参考：随机权重的迷你模型在多步 decode 后误差会放大，不作判据

    if results.get(base, {}).get("bpt"):
        print("\n每 token 长期 KV 字节数 / 速度：")
        for dt in dtypes:
            bpt = results[dt]["bpt"] or -1
            per_gb = int((1 << 30) / bpt) if bpt > 0 else 0
            print("  %-10s %7.1f B/token  每 GB 约 %d token  %s" % (dt, bpt, per_gb, results[dt]["timing"]))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
