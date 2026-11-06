#!/usr/bin/env python3
"""W8A8顺序执行的vLLM/FastLLM前向正确性检查入口。"""

import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import forward_check_vllm as shared_forward  # noqa: E402


def argument_value(name):
    """取得命令行参数的下一个值，未提供时返回None。"""
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def target_trace_pattern():
    """根据compressed-tensors模型配置选择精确的W8A8目标路径。"""
    model = argument_value("--model")
    if model is None:
        return "[fastllm][w8a8] path=w8a8-cutlass"

    try:
        config = json.loads((Path(model) / "config.json").read_text(
            encoding="utf-8"))
        groups = config["quantization_config"]["config_groups"]
        for group in groups.values():
            weight = group.get("weights") or {}
            activation = group.get("input_activations") or {}
            if weight.get("type") != "int" or weight.get("num_bits") != 8:
                continue
            if activation.get("type") != "int" or activation.get("num_bits") != 8:
                continue
            return (
                "[fastllm][w8a8] path=w8a8-int8-cutlass"
                if activation.get("symmetric", True)
                else "[fastllm][w8a8] path=w8a8-int8-azp-cutlass"
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    return "[fastllm][w8a8] path=w8a8-cutlass"


if __name__ == "__main__":
    if "--label" not in sys.argv:
        sys.argv.extend(("--label", "W8A8"))
    if "--target-trace-pattern" not in sys.argv:
        sys.argv.extend((
            "--target-trace-pattern",
            target_trace_pattern(),
        ))
    raise SystemExit(shared_forward.main())
