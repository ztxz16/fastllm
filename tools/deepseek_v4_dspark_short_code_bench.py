#!/usr/bin/env python3

import ctypes
import hashlib
import json
import statistics
import struct
import time

from ftllm.llm import encode_hf_prompt, fastllm_lib
from ftllm.util import make_normal_llm_model, make_normal_parser


TASKS = [
    (
        "merge_intervals",
        "Implement this Python function. Return only the completed function "
        "and a few concise asserts.\n\n"
        "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
        "    # Merge every overlapping closed interval. Input order is arbitrary.\n"
        "    pass\n",
    ),
    (
        "fix_binary_search",
        "Fix the bug in the Python function below. It must return the first "
        "index whose value is greater than or equal to target, or len(nums). "
        "Return only corrected code and three concise asserts.\n\n"
        "def lower_bound(nums, target):\n"
        "    lo, hi = 0, len(nums) - 1\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if nums[mid] < target:\n"
        "            lo = mid\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return lo\n",
    ),
    (
        "ttl_cache",
        "Write a small, thread-safe Python TTLCache class with get(key), "
        "put(key, value, ttl_seconds), and delete(key). Use time.monotonic, "
        "lazily discard expired entries, and distinguish a missing key from "
        "a stored None. Return only code plus a short usage example.",
    ),
    (
        "topological_order",
        "Implement the Python function below using Kahn's algorithm. It "
        "should return a deterministic lexicographically smallest valid "
        "order, or [] when the graph has a cycle. Return only code and a few "
        "concise asserts.\n\n"
        "def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:\n"
        "    pass\n",
    ),
]


def encode_query(model, query):
    prompt = model.get_prompt(query, [])
    if getattr(model, "hf_tokenizer", None) is not None:
        return encode_hf_prompt(model.hf_tokenizer, prompt)
    return model.encode(prompt)


def decode_tokens(model, tokens):
    tokenizer = getattr(model, "hf_tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(tokens)
    decoder = getattr(model, "_decode_fastllm_token", None)
    if decoder is None:
        return ""
    return b"".join(decoder(token) for token in tokens).decode(
        "utf-8", errors="replace"
    )


def token_hash(tokens):
    packed = b"".join(struct.pack("<i", token) for token in tokens)
    return hashlib.sha256(packed).hexdigest()


def run_request(model, label, query, output_tokens):
    input_tokens = encode_query(model, query)
    stop_token_len, stop_token_list = model.stop_token_ctypes(None)
    input_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
    start = time.perf_counter()
    handle = fastllm_lib.launch_response_llm_model(
        model.model,
        len(input_tokens),
        input_buffer,
        ctypes.c_int(output_tokens),
        ctypes.c_int(0),
        ctypes.c_bool(False),
        ctypes.c_float(1.0),
        ctypes.c_int(1),
        ctypes.c_float(1.0),
        ctypes.c_float(1.0),
        ctypes.c_bool(False),
        stop_token_len,
        stop_token_list,
    )
    generated = []
    first_token_time = None
    finish_code = None
    while True:
        if not fastllm_lib.can_fetch_response_llm_model(model.model, handle):
            time.sleep(0.0005)
            continue
        token = fastllm_lib.fetch_response_llm_model(model.model, handle)
        now = time.perf_counter()
        if token <= -1:
            finish_code = token
            end = now
            break
        if first_token_time is None:
            first_token_time = now
        generated.append(token)

    if first_token_time is None:
        ttft = None
        tpop = None
        decode_speed = 0.0
    else:
        ttft = first_token_time - start
        decode_count = max(len(generated) - 1, 0)
        decode_time = end - first_token_time
        tpop = decode_time / decode_count if decode_count else None
        decode_speed = decode_count / decode_time if decode_time > 0 else 0.0

    result = {
        "label": label,
        "input_tokens": len(input_tokens),
        "output_tokens": len(generated),
        "finish_code": finish_code,
        "total_time_s": end - start,
        "ttft_s": ttft,
        "tpop_s_per_token": tpop,
        "decode_tokens_per_s": decode_speed,
        "token_sha256": token_hash(generated),
        "text": decode_tokens(model, generated),
    }
    print("@@RESULT@@ " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = make_normal_parser("DeepSeek-V4 DSpark short coding benchmark")
    parser.add_argument("--bench-output-tokens", type=int, default=160)
    parser.add_argument("--bench-warmup-output-tokens", type=int, default=32)
    parser.add_argument("--bench-task-limit", type=int, default=len(TASKS))
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be greater than 0")
    if args.bench_output_tokens <= 1:
        parser.error("--bench-output-tokens must be greater than 1")
    if args.bench_warmup_output_tokens <= 1:
        parser.error("--bench-warmup-output-tokens must be greater than 1")
    if not 1 <= args.bench_task_limit <= len(TASKS):
        parser.error("--bench-task-limit is out of range")

    print(
        "@@CONFIG@@ "
        + json.dumps(
            {
                "model": args.path or args.model,
                "threads": args.threads,
                "device": args.device,
                "moe_device": args.moe_device,
                "dspark": args.dspark,
                "dspark_confidence_threshold": (
                    args.speculative_dspark_confidence_threshold
                ),
                "output_tokens": args.bench_output_tokens,
                "task_limit": args.bench_task_limit,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("@@PHASE@@ load_model", flush=True)
    model = make_normal_llm_model(args)
    try:
        print("@@PHASE@@ warmup", flush=True)
        run_request(
            model,
            "warmup",
            "Write a Python function add(a, b) that returns the sum. Return only code.",
            args.bench_warmup_output_tokens,
        )
        measured = []
        for label, query in TASKS[: args.bench_task_limit]:
            print("@@PHASE@@ " + label, flush=True)
            measured.append(
                run_request(model, label, query, args.bench_output_tokens)
            )
        speeds = [item["decode_tokens_per_s"] for item in measured]
        tpops = [
            item["tpop_s_per_token"]
            for item in measured
            if item["tpop_s_per_token"] is not None
        ]
        summary = {
            "tasks": len(measured),
            "total_input_tokens": sum(item["input_tokens"] for item in measured),
            "total_output_tokens": sum(item["output_tokens"] for item in measured),
            "mean_decode_tokens_per_s": statistics.mean(speeds),
            "median_decode_tokens_per_s": statistics.median(speeds),
            "mean_tpop_s_per_token": statistics.mean(tpops) if tpops else None,
        }
        print("@@SUMMARY@@ " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        model.release_memory()
        print("@@PHASE@@ released", flush=True)


if __name__ == "__main__":
    main()
