#!/usr/bin/env python3
"""Profile one warmed vLLM DeepSeek-V4 request with cudaProfilerApi."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--result", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--post-profile-wait", type=float, default=5.0,
        help="leave worker processes alive briefly so Nsight can flush CUPTI data",
    )
    parser.add_argument(
        "--dspark", type=int, default=0,
        help="enable embedded DSpark with this many draft tokens",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="disable CUDA graphs (useful for one-off logits alignment probes)",
    )
    return parser.parse_args()


def read_input_ids(path: str) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def generate(llm, input_ids: list[int], output_tokens: int) -> dict[str, object]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    started = time.perf_counter()
    request = llm.generate(
        [{"prompt_token_ids": input_ids}], sampling, use_tqdm=False
    )[0]
    finished = time.perf_counter()
    token_ids = [int(token_id) for token_id in request.outputs[0].token_ids]
    return {
        "token_ids": token_ids,
        "total_seconds": finished - started,
        "tokens_per_second": len(token_ids) / max(finished - started, 1.0e-9),
    }


def main() -> None:
    args = parse_args()
    if args.output_tokens <= 0 or args.warmup_tokens <= 0:
        raise ValueError("output and warmup token counts must be positive")

    from vllm import LLM

    input_ids = read_input_ids(args.input_ids)
    max_model_len = max(512, len(input_ids) + args.output_tokens + 1)
    graph_sizes = [1]
    speculative_config = None
    if args.dspark > 0:
        graph_sizes.append(args.dspark + 1)
        speculative_config = {
            "method": "dspark",
            "num_speculative_tokens": args.dspark,
            "draft_sample_method": "greedy",
        }
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=0.90,
        kv_cache_dtype="fp8",
        block_size=256,
        enforce_eager=args.enforce_eager,
        cudagraph_capture_sizes=graph_sizes,
        max_cudagraph_capture_size=max(graph_sizes),
        enable_prefix_caching=False,
        enable_flashinfer_autotune=False,
        speculative_config=speculative_config,
        profiler_config={"profiler": "cuda"},
        seed=0,
    )

    warmup = generate(llm, input_ids, args.warmup_tokens)
    profiling = False
    try:
        llm.start_profile()
        profiling = True
        profiled = generate(llm, input_ids, args.output_tokens)
    finally:
        if profiling:
            llm.stop_profile()
            # vLLM owns CUDA contexts in spawned worker processes.  Returning
            # immediately lets LLM teardown terminate those workers while
            # Nsight Systems is still draining CUPTI buffers, which can yield
            # an empty report even though profiling did run successfully.
            if args.post_profile_wait > 0:
                time.sleep(args.post_profile_wait)

    result = {
        "engine": "vllm",
        "model": args.model,
        "input_ids": input_ids,
        "input_tokens": len(input_ids),
        "target_output_tokens": args.output_tokens,
        "warmup": warmup,
        "profiled": profiled,
        "mode": {
            "tp": args.tp,
            "kv_cache_dtype": "fp8",
            "cuda_graph_sizes": graph_sizes,
            "prefix_caching": False,
            "expert_parallel": False,
            "dspark": args.dspark,
        },
    }
    Path(args.result).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
