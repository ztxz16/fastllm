#!/usr/bin/env python3
"""Profile one warmed FastLLM DeepSeek-V4 request with cudaProfilerApi."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

from ftllm.util import make_normal_llm_model, make_normal_parser


def parse_args():
    parser = make_normal_parser("FastLLM DeepSeek-V4 Nsight Systems workload")
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def read_input_ids(path: str) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def cuda_profiler_runtime():
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart.argtypes = []
    cudart.cudaProfilerStart.restype = ctypes.c_int
    cudart.cudaProfilerStop.argtypes = []
    cudart.cudaProfilerStop.restype = ctypes.c_int
    return cudart


def check_cuda(state: int, operation: str) -> None:
    if state != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {state}")


def run_request(model, input_ids: list[int], output_tokens: int) -> dict[str, object]:
    from ftllm import llm

    stop_token_len, stop_token_list = model.stop_token_ctypes(None)
    input_buffer = (ctypes.c_int * len(input_ids))(*input_ids)
    started = time.perf_counter()
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model,
        len(input_ids),
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
    token_ids: list[int] = []
    first_token_time: float | None = None
    while True:
        if not llm.fastllm_lib.can_fetch_response_llm_model(model.model, handle):
            time.sleep(0.0002)
            continue
        token_id = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
        if token_id < 0:
            finish_code = int(token_id)
            break
        if first_token_time is None:
            first_token_time = time.perf_counter()
        token_ids.append(int(token_id))
    finished = time.perf_counter()
    decode_seconds = (
        finished - first_token_time
        if first_token_time is not None and len(token_ids) > 1
        else 0.0
    )
    return {
        "token_ids": token_ids,
        "finish_code": finish_code,
        "total_seconds": finished - started,
        "ttft_seconds": (
            first_token_time - started if first_token_time is not None else None
        ),
        "decode_tokens_per_second": (
            (len(token_ids) - 1) / decode_seconds if decode_seconds > 0 else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_tokens <= 0 or args.warmup_tokens <= 0:
        raise ValueError("output and warmup token counts must be positive")
    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    args.max_batch = 1
    if args.tokens <= 0:
        args.tokens = 512
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"

    input_ids = read_input_ids(args.input_ids)
    model = make_normal_llm_model(args)
    try:
        warmup = run_request(model, input_ids, args.warmup_tokens)
        cudart = cuda_profiler_runtime()
        check_cuda(cudart.cudaProfilerStart(), "cudaProfilerStart")
        profiled = run_request(model, input_ids, args.output_tokens)
        check_cuda(cudart.cudaProfilerStop(), "cudaProfilerStop")

        result = {
            "engine": "fastllm",
            "model": args.path or args.model,
            "input_ids": input_ids,
            "input_tokens": len(input_ids),
            "target_output_tokens": args.output_tokens,
            "warmup": warmup,
            "profiled": profiled,
            "mode": {
                "tp": args.tp,
                "device": args.device,
                "kv_cache_dtype": args.kv_cache_dtype,
                "dspark": int(args.dspark or 0),
            },
        }
        Path(args.result).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        model.release_memory()


if __name__ == "__main__":
    main()
