#!/usr/bin/env python3
"""Run Dots3-Note in FastLLM and save processor-free raw logits to NPZ."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from ftllm.util import make_normal_llm_model, make_normal_parser


def parse_args():
    parser = make_normal_parser("Dots3-Note FastLLM raw-logits capture")
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--repeat-to",
        type=int,
        default=0,
        help="repeat the supplied input-id pattern to this prompt length",
    )
    return parser.parse_args()


def read_input_ids(path: str) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.repeat_to < 0:
        raise ValueError("--repeat-to must be non-negative")
    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    if args.max_batch <= 0:
        args.max_batch = 1
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "bfloat16"
    if args.moe_atype in ("", "auto"):
        args.moe_atype = "bfloat16"

    input_ids = read_input_ids(args.input_ids)
    if args.repeat_to > 0:
        input_ids = [
            input_ids[index % len(input_ids)]
            for index in range(args.repeat_to)
        ]
    if args.tokens <= 0:
        args.tokens = len(input_ids) + args.steps + 16
    else:
        args.tokens = max(args.tokens, len(input_ids) + args.steps + 1)
    load_started = time.perf_counter()
    model = make_normal_llm_model(args)
    load_seconds = time.perf_counter() - load_started
    cuda_profiler = None
    if os.environ.get("FASTLLM_CUDA_PROFILER_CAPTURE", "") not in (
        "", "0", "false", "off"
    ):
        cuda_profiler = ctypes.CDLL("libcudart.so")
    try:
        from ftllm import llm

        with (Path(args.path) / "config.json").open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        vocab_size = int(config["vocab_size"])
        input_array = (ctypes.c_int * len(input_ids))(*input_ids)
        if cuda_profiler is not None:
            status = cuda_profiler.cudaProfilerStart()
            if status != 0:
                raise RuntimeError(f"cudaProfilerStart failed: {status}")
        request_started = time.perf_counter()
        response_handle = llm.fastllm_lib.launch_response_llm_model(
            model.model,
            len(input_ids),
            input_array,
            ctypes.c_int(args.steps),
            ctypes.c_int(0),
            ctypes.c_bool(False),
            ctypes.c_float(1.0),
            ctypes.c_int(1),
            ctypes.c_float(1.0),
            ctypes.c_float(1.0),
            ctypes.c_bool(True),
            ctypes.c_int(0),
            None,
        )

        output_ids: list[int] = []
        logits: list[np.ndarray] = []
        token_timestamps: list[float] = []
        logits_buffer = (ctypes.c_float * vocab_size)()
        while True:
            token_id = llm.fastllm_lib.fetch_response_logits_llm_model(
                model.model, response_handle, logits_buffer
            )
            if token_id < 0:
                break
            output_ids.append(int(token_id))
            logits.append(np.ctypeslib.as_array(logits_buffer).copy())
            token_timestamps.append(time.perf_counter())
        if cuda_profiler is not None:
            status = cuda_profiler.cudaProfilerStop()
            if status != 0:
                raise RuntimeError(f"cudaProfilerStop failed: {status}")
        if len(logits) != args.steps:
            raise RuntimeError(
                f"FastLLM returned {len(logits)} logits steps, expected "
                f"{args.steps}"
            )

        stacked_logits = np.stack(logits).astype(np.float32, copy=False)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_seconds = token_timestamps[-1] - request_started
        ttft_seconds = token_timestamps[0] - request_started
        decode_seconds = (
            token_timestamps[-1] - token_timestamps[0]
            if len(token_timestamps) > 1
            else 0.0
        )
        decode_tokens_per_second = (
            (len(token_timestamps) - 1) / decode_seconds
            if decode_seconds > 0.0
            else 0.0
        )
        np.savez(
            output_path,
            engine=np.asarray("fastllm"),
            model=np.asarray(args.path),
            input_ids=np.asarray(input_ids, dtype=np.int32),
            output_ids=np.asarray(output_ids, dtype=np.int32),
            logits=stacked_logits,
            vocab_size=np.asarray([vocab_size], dtype=np.int32),
            load_seconds=np.asarray([load_seconds], dtype=np.float64),
            request_seconds=np.asarray([request_seconds], dtype=np.float64),
            ttft_seconds=np.asarray([ttft_seconds], dtype=np.float64),
            decode_seconds=np.asarray([decode_seconds], dtype=np.float64),
            decode_tokens_per_second=np.asarray(
                [decode_tokens_per_second], dtype=np.float64
            ),
            mode=np.asarray(
                json.dumps(
                    {
                        "tp": args.tp,
                        "device": args.device,
                        "moe_device": args.moe_device,
                        "dtype": args.dtype,
                        "atype": args.atype,
                        "moe_atype": args.moe_atype,
                        "kv_cache_dtype": args.kv_cache_dtype,
                        "cuda_shared_expert": args.cuda_shared_expert,
                        "steps": args.steps,
                        "use_dsa": True,
                        "prompt_tokens": len(input_ids),
                        "chunked_prefill_size": args.chunked_prefill_size,
                    }
                )
            ),
        )
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "output_ids": output_ids,
                    "shape": list(stacked_logits.shape),
                    "load_seconds": load_seconds,
                    "request_seconds": request_seconds,
                    "ttft_seconds": ttft_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": decode_tokens_per_second,
                },
                indent=2,
            )
        )
    finally:
        model.release_memory()


if __name__ == "__main__":
    main()
