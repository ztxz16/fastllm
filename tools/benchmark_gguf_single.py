#!/usr/bin/env python3
"""Benchmark one GGUF model with fixed single-request prefill/decode shapes."""

import ctypes
import hashlib
import json
import os
import statistics
import time

from ftllm.benchmark import _build_input_tokens
from ftllm.llm import fastllm_lib
from ftllm.util import make_normal_llm_model, make_normal_parser


def _launch(model, input_tokens, output_tokens, exact_output):
    stop_len, stop_ids = model.stop_token_ctypes(None)
    input_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
    started = time.perf_counter()
    handle = fastllm_lib.launch_response_llm_model(
        model.model,
        len(input_tokens),
        input_buffer,
        ctypes.c_int(output_tokens),
        ctypes.c_int(output_tokens if exact_output else 0),
        ctypes.c_bool(False),
        ctypes.c_float(1.0),
        ctypes.c_int(1),
        ctypes.c_float(1.0),
        ctypes.c_float(1.0),
        ctypes.c_bool(False),
        stop_len,
        stop_ids,
    )
    generated = []
    first = None
    while True:
        if not fastllm_lib.can_fetch_response_llm_model(model.model, handle):
            time.sleep(0.00005)
            continue
        token = fastllm_lib.fetch_response_llm_model(model.model, handle)
        now = time.perf_counter()
        if token <= -1:
            ended = now
            break
        if first is None:
            first = now
        generated.append(token)
    return started, first, ended, generated


def _token_hash(tokens):
    payload = b"".join(
        int(token).to_bytes(4, "little", signed=True) for token in tokens)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = make_normal_parser(
        "single-concurrency GGUF prefill/decode benchmark")
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-context-tokens", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-runs", type=int, default=3)
    parser.add_argument("--decode-runs", type=int, default=3)
    parser.add_argument("--decode-warmup-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt-unit",
        default="请只继续输出这个正整数序列，不要解释，不要停止：1 2 3 4 5 6 7 8 9 10 ",
    )
    args = parser.parse_args()
    args.max_batch = 1
    args.tokens = max(
        args.tokens,
        args.prefill_tokens + 8,
        args.decode_context_tokens + args.decode_tokens + 8,
    )
    os.environ.setdefault("FASTLLM_PREFIX_CACHE", "0")

    model = make_normal_llm_model(args)
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    try:
        prefill_input = _build_input_tokens(
            model, args.prefill_tokens, args.prompt_unit)
        _launch(model, prefill_input, 1, False)
        if cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError("prefill warmup cudaDeviceSynchronize failed")
        prefill_seconds = []
        for _ in range(args.prefill_runs):
            started = time.perf_counter()
            _launch(model, prefill_input, 1, False)
            if cudart.cudaDeviceSynchronize() != 0:
                raise RuntimeError("prefill cudaDeviceSynchronize failed")
            prefill_seconds.append(time.perf_counter() - started)

        decode_input = _build_input_tokens(
            model, args.decode_context_tokens, args.prompt_unit)
        if args.decode_warmup_tokens > 0:
            _launch(model, decode_input, args.decode_warmup_tokens, True)
        decode_results = []
        for _ in range(args.decode_runs):
            started, first, ended, generated = _launch(
                model, decode_input, args.decode_tokens, True)
            decode_results.append({
                "tokens": len(generated),
                "ttft_ms": ((first - started) * 1000.0
                            if first is not None else None),
                "decode_tps": (
                    (len(generated) - 1) / (ended - first)
                    if first is not None and len(generated) > 1 else 0.0),
                "sha256": _token_hash(generated),
                "first_tokens": generated[:8],
            })

        prefill_median = statistics.median(prefill_seconds)
        result = {
            "model": args.path,
            "prefill_tokens": len(prefill_input),
            "prefill_runs_ms": [value * 1000.0
                                for value in prefill_seconds],
            "prefill_median_ms": prefill_median * 1000.0,
            "prefill_tps": len(prefill_input) / prefill_median,
            "decode_context_tokens": len(decode_input),
            "decode_tokens": args.decode_tokens,
            "decode_results": decode_results,
            "decode_median_tps": statistics.median(
                item["decode_tps"] for item in decode_results),
        }
        print("GGUF_BENCH " + json.dumps(
            result, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        model.release_memory()


if __name__ == "__main__":
    main()
