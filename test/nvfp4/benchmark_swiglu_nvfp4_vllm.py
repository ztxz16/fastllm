#!/usr/bin/env python3
"""单case运行vLLM SwiGLU+NVFP4融合量化，供通用算子对比框架调用。"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM SwiGLU NVFP4算子基准")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rows <= 0 or args.hidden <= 0 or args.hidden % 32 != 0:
        raise ValueError("rows必须大于0，hidden必须是32的倍数")

    import torch
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    from vllm.scalar_type import scalar_types
    from vllm.v1.worker.workspace import init_workspace_manager

    from operator_performance_compare import make_swiglu_runner, wall_time_benchmark

    if not current_platform.has_device_capability(100):
        raise RuntimeError("vLLM NVFP4算子需要SM100或更新GPU")
    torch.cuda.set_device(args.device)
    init_workspace_manager(torch.device(f"cuda:{args.device}"))
    case = {
        "kind": "swiglu_quant",
        "m": args.rows,
        "n": 0,
        "k": args.hidden,
        "dtype": args.dtype,
    }
    runner = make_swiglu_runner(
        case, torch, ops, scalar_types.float4_e2m1f.max(),
        torch.finfo(torch.float8_e4m3fn).max)
    latency_ms = wall_time_benchmark(torch, runner, args.warmup, args.iters)
    result = {
        "implementation": "vllm",
        "rows": args.rows,
        "hidden": args.hidden,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms": latency_ms,
    }
    print(json.dumps(result, ensure_ascii=False))
    print(f"latency: avg_ms={latency_ms:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
