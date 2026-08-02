#!/usr/bin/env python3
"""Run one FastLLM request and save prefill/decode raw logits to NPZ."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np

from ftllm.util import make_normal_llm_model, make_normal_parser


def parse_args():
    parser = make_normal_parser("DeepSeek-V4 FastLLM raw-logits capture")
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
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
    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    if args.max_batch <= 0:
        args.max_batch = 1
    if args.tokens <= 0:
        args.tokens = 256
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"

    input_ids = read_input_ids(args.input_ids)
    model = make_normal_llm_model(args)
    try:
        from ftllm import llm

        with (Path(args.path) / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        vocab_size = int(config["vocab_size"])
        input_array = (ctypes.c_int * len(input_ids))(*input_ids)
        handle = llm.fastllm_lib.launch_response_llm_model(
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
        logits_buffer = (ctypes.c_float * vocab_size)()
        while True:
            token_id = llm.fastllm_lib.fetch_response_logits_llm_model(
                model.model, handle, logits_buffer
            )
            if token_id < 0:
                break
            output_ids.append(int(token_id))
            logits.append(np.ctypeslib.as_array(logits_buffer).copy())
        if len(logits) != args.steps:
            raise RuntimeError(
                f"FastLLM returned {len(logits)} logits steps, expected {args.steps}"
            )

        stacked_logits = np.stack(logits).astype(np.float32, copy=False)
        np.savez(
            args.output,
            engine=np.asarray("fastllm"),
            model=np.asarray(args.path),
            input_ids=np.asarray(input_ids, dtype=np.int32),
            output_ids=np.asarray(output_ids, dtype=np.int32),
            logits=stacked_logits,
            vocab_size=np.asarray([vocab_size], dtype=np.int32),
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
                        "steps": args.steps,
                        "dspark": False,
                    }
                )
            ),
        )
        print(
            json.dumps(
                {
                    "output": args.output,
                    "output_ids": output_ids,
                    "shape": list(stacked_logits.shape),
                },
                indent=2,
            )
        )
    finally:
        model.release_memory()


if __name__ == "__main__":
    main()
