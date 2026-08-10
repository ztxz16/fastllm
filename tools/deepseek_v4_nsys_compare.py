#!/usr/bin/env python3
"""Compare steady DeepSeek-V4 CUDA Graph decode traces from Nsight Systems.

The node-granularity reports are used for kernel attribution.  Only kernels
with a non-null graphId are included, which excludes prefill and FastLLM's
warm/capture pass.  The graph-granularity reports are used only for replay
span statistics because graph tracing perturbs execution much less than node
tracing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


FAST_ATTENTION_PATTERNS = (
    "%sparse_mla_decode_dsv4_kernel%",
    "%fastllm_deepseek_v4_sparse_decode%split_kernel%",
    "%fastllm_deepseek_v4_sparse_decode_kernel%",
)
VLLM_ATTENTION_PATTERN = "%sparse_mla_decode_dsv4_kernel%"


@dataclass
class Operator:
    engine: str
    family: str
    name: str
    calls: int
    total_gpu_ms: float
    average_kernel_us: float
    calls_per_rank_token: float
    gpu_us_per_rank_token: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-node", required=True, type=Path)
    parser.add_argument("--vllm-node", required=True, type=Path)
    parser.add_argument("--fast-graph", required=True, type=Path)
    parser.add_argument("--vllm-graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def fast_family(name: str) -> str:
    lower_name = name.lower()
    if (
        "fastllm_deepseek_v4_sparse_decode" in name
        or "sparse_mla_decode_dsv4" in name
    ):
        return "sparse_attention_core"
    if "CustomAllReduce" in name or "ncclDevKernel" in name:
        return "communication"
    if (
        "SelectExpert" in name
        or "RouteScoreTransform" in name
        or "HashRouteScore" in name
        or "SqrtSoftplusTop6" in name
        or "sqrtsoftplus_router" in lower_name
    ):
        return "moe_router"
    if "marlin_moe_wna16::Marlin" in name:
        return "expert_gemm"
    if any(token in name for token in ("BuildEpMetadata", "SwigluRows", "ReduceEpRows")):
        return "expert_plumbing"
    if any(
        token in name
        for token in ("GemvBF16FP8", "GemvFp32Fp16", "GemvBf16Fp16", "WoAPairBlockReduce")
    ):
        return "linear_projection"
    if "Hc" in name and "GraphCompressor" not in name:
        return "mhc"
    if any(
        token in name
        for token in (
            "FusedQKVRopeCache",
            "UpdateCompressedKVGraph",
            "SparseDecodeRotaryCast",
            "StoreGraphCompressorRaw",
            "RMSNormKernelInner1<(int)512>",
        )
    ):
        return "attention_prepare_indexer"
    if any(token in name for token in ("GreedySampling", "EmbeddingVector", "RMSNormKernelInner1<(int)1024>")):
        return "output_head_sampling"
    return "other"


def vllm_family(name: str) -> str:
    if "sparse_mla_decode_dsv4" in name:
        return "sparse_attention_core"
    if "ncclDevKernel" in name:
        return "communication"
    if "_dsv4_topk_kernel" in name or "dsv4HashTopk" in name:
        return "moe_router"
    if (
        "deep_gemm::sm120_fp8_fp4_gemm" in name
        and "(deep_gemm::GemmType)1" in name
    ):
        return "expert_gemm"
    if any(
        token in name
        for token in (
            "_fwd_kernel_ep_scatter",
            "_fwd_kernel_ep_gather",
            "_count_expert_num_tokens",
            "_silu_mul_quant_fp8",
        )
    ):
        return "expert_plumbing"
    if any(
        token in name
        for token in (
            "deep_gemm::sm120_fp8_fp4_gemm",
            "sm120_split_k_reduce",
            "gemvx::kernel",
            "per_token_group_quant_8bit",
            "_save_partial_states_kernel",
        )
    ):
        return "linear_projection"
    if "mhc" in name or "hc_" in name or "tf32_hc_" in name:
        return "mhc"
    if any(
        token in name
        for token in (
            "fusedDeepseekV4QNormRope",
            "sparse_attn_compress",
            "sm120_fp8_paged_mqa_logits",
            "_fused_kv_compress",
            "IndexerQFp8",
            "_compute_global_topk",
            "persistent_topk",
            "SparseAttnCompress",
            "SparseAttnNorm",
            "_fused_inv_rope",
            "_fused_q_kv_rmsnorm",
        )
    ):
        return "attention_prepare_indexer"
    if "rms_norm_kernel" in name:
        return "output_head_sampling"
    return "other"


def resolve_name_pattern(path: Path, patterns: Iterable[str]) -> str:
    connection = sqlite3.connect(path)
    try:
        for pattern in patterns:
            calls = connection.execute(
                """
                SELECT COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                JOIN StringIds AS strings ON strings.id = kernels.demangledName
                WHERE kernels.graphId IS NOT NULL AND strings.value LIKE ?
                """,
                (pattern,),
            ).fetchone()[0]
            if calls > 0:
                return pattern
    finally:
        connection.close()
    raise RuntimeError(
        f"none of the attention anchor patterns matched {path}: "
        + ", ".join(patterns)
    )


def load_operators(
    path: Path,
    engine: str,
    family_fn: Callable[[str], str],
    attention_pattern: str,
) -> tuple[list[Operator], dict[str, int]]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        SELECT strings.value, COUNT(*), SUM(kernels.end - kernels.start) / 1.0e6,
               AVG(kernels.end - kernels.start) / 1.0e3
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.graphId IS NOT NULL
        GROUP BY kernels.demangledName
        ORDER BY SUM(kernels.end - kernels.start) DESC
        """
    ).fetchall()
    ranks = connection.execute(
        "SELECT COUNT(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()[0]
    attention_calls = connection.execute(
        """
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.graphId IS NOT NULL AND strings.value LIKE ?
        """,
        (attention_pattern,),
    ).fetchone()[0]
    connection.close()

    layers = 43
    replay_tokens = attention_calls // (ranks * layers)
    if replay_tokens <= 0 or attention_calls != replay_tokens * ranks * layers:
        raise RuntimeError(
            f"cannot infer complete replay count for {engine}: "
            f"attention_calls={attention_calls}, ranks={ranks}"
        )
    denominator = ranks * replay_tokens
    operators = [
        Operator(
            engine=engine,
            family=family_fn(name),
            name=name,
            calls=int(calls),
            total_gpu_ms=float(total_ms),
            average_kernel_us=float(average_us),
            calls_per_rank_token=calls / denominator,
            gpu_us_per_rank_token=total_ms * 1000.0 / denominator,
        )
        for name, calls, total_ms, average_us in rows
    ]
    return operators, {
        "ranks": ranks,
        "layers": layers,
        "replay_tokens": replay_tokens,
        "rank_tokens": denominator,
        "kernel_calls": sum(operator.calls for operator in operators),
    }


def aggregate_families(operators: Iterable[Operator]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls_per_rank_token": 0.0, "gpu_us_per_rank_token": 0.0}
    )
    for operator in operators:
        family = result[operator.family]
        family["calls_per_rank_token"] += operator.calls_per_rank_token
        family["gpu_us_per_rank_token"] += operator.gpu_us_per_rank_token
    return dict(result)


def family_value(
    families: dict[str, dict[str, float]], family: str, field: str = "gpu_us_per_rank_token"
) -> float:
    return families.get(family, {}).get(field, 0.0)


def comparison(
    name: str,
    fast_us: float,
    vllm_us: float,
    unit: str,
    logical_instances_per_token: int | None = None,
) -> dict[str, float | str | int]:
    output: dict[str, float | str | int] = {
        "name": name,
        "unit": unit,
        "fastllm": fast_us,
        "vllm": vllm_us,
        "fastllm_over_vllm": fast_us / vllm_us if vllm_us else math.inf,
        "fastllm_minus_vllm": fast_us - vllm_us,
    }
    if logical_instances_per_token is not None:
        output["logical_instances_per_token"] = logical_instances_per_token
        output["delta_us_per_token"] = (
            fast_us - vllm_us
        ) * logical_instances_per_token
    return output


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def collective_stats(path: Path, name_pattern: str) -> dict[str, object]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        SELECT kernels.deviceId, (kernels.end - kernels.start) / 1.0e3
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.graphId IS NOT NULL AND strings.value LIKE ?
        """,
        (name_pattern,),
    ).fetchall()
    connection.close()
    all_values = [float(duration) for _, duration in rows]
    per_device: dict[str, dict[str, float | int]] = {}
    for device in sorted({device for device, _ in rows}):
        values = [float(duration) for row_device, duration in rows if row_device == device]
        per_device[str(device)] = {
            "calls": len(values),
            "mean_us": statistics.mean(values),
            "p50_us": percentile(values, 0.50),
            "p90_us": percentile(values, 0.90),
            "p99_us": percentile(values, 0.99),
            "max_us": max(values),
            "calls_over_100us": sum(value > 100.0 for value in values),
        }
    return {
        "calls": len(all_values),
        "mean_us": statistics.mean(all_values),
        "p50_us": percentile(all_values, 0.50),
        "p90_us": percentile(all_values, 0.90),
        "p99_us": percentile(all_values, 0.99),
        "max_us": max(all_values),
        "calls_over_100us": sum(value > 100.0 for value in all_values),
        "per_device": per_device,
    }


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    total = 0
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def layer_window_stats(path: Path, attention_pattern: str) -> dict[str, object]:
    """Measure overlap in the 42 intra-token sparse-attention anchor windows."""
    connection = sqlite3.connect(path)
    devices = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
        )
    ]
    all_windows: list[tuple[int, int, int, int, int, int]] = []
    per_device: dict[str, dict[str, float | int]] = {}
    for device in devices:
        anchors = [
            row[0]
            for row in connection.execute(
                """
                SELECT kernels.start
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                JOIN StringIds AS strings ON strings.id = kernels.demangledName
                WHERE kernels.graphId IS NOT NULL AND kernels.deviceId = ?
                      AND strings.value LIKE ?
                ORDER BY kernels.start
                """,
                (device, attention_pattern),
            )
        ]
        kernels = connection.execute(
            """
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE graphId IS NOT NULL AND deviceId = ?
            ORDER BY start
            """,
            (device,),
        ).fetchall()
        token_count = len(anchors) // 43
        bounds = [
            (anchors[token * 43 + layer], anchors[token * 43 + layer + 1])
            for token in range(token_count)
            for layer in range(42)
        ]
        active: list[tuple[int, int]] = []
        kernel_index = 0
        windows: list[tuple[int, int, int, int, int, int]] = []
        for start, end in bounds:
            while kernel_index < len(kernels) and kernels[kernel_index][0] < end:
                active.append(kernels[kernel_index])
                kernel_index += 1
            active = [interval for interval in active if interval[1] > start]
            clipped = [
                (max(kernel_start, start), min(kernel_end, end))
                for kernel_start, kernel_end in active
                if kernel_start < end and kernel_end > start
            ]
            span = end - start
            duration_sum = sum(kernel_end - kernel_start for kernel_start, kernel_end in clipped)
            union = interval_union_ns(clipped)
            windows.append(
                (span, duration_sum, union, span - union, duration_sum - union, len(clipped))
            )
        all_windows.extend(windows)
        per_device[str(device)] = {
            "windows": len(windows),
            "span_mean_us": statistics.mean(row[0] for row in windows) / 1.0e3,
            "span_p50_us": percentile([row[0] / 1.0e3 for row in windows], 0.50),
            "duration_sum_mean_us": statistics.mean(row[1] for row in windows) / 1.0e3,
            "union_busy_mean_us": statistics.mean(row[2] for row in windows) / 1.0e3,
            "idle_mean_us": statistics.mean(row[3] for row in windows) / 1.0e3,
            "overlap_mean_us": statistics.mean(row[4] for row in windows) / 1.0e3,
            "kernels_mean": statistics.mean(row[5] for row in windows),
        }
    connection.close()
    duration_sum = sum(row[1] for row in all_windows)
    overlap_sum = sum(row[4] for row in all_windows)
    return {
        "windows": len(all_windows),
        "span_mean_us": statistics.mean(row[0] for row in all_windows) / 1.0e3,
        "span_p50_us": percentile([row[0] / 1.0e3 for row in all_windows], 0.50),
        "span_p90_us": percentile([row[0] / 1.0e3 for row in all_windows], 0.90),
        "duration_sum_mean_us": statistics.mean(row[1] for row in all_windows) / 1.0e3,
        "union_busy_mean_us": statistics.mean(row[2] for row in all_windows) / 1.0e3,
        "idle_mean_us": statistics.mean(row[3] for row in all_windows) / 1.0e3,
        "overlap_mean_us": statistics.mean(row[4] for row in all_windows) / 1.0e3,
        "overlap_fraction_of_duration_sum": overlap_sum / duration_sum,
        "kernels_mean": statistics.mean(row[5] for row in all_windows),
        "per_device": per_device,
    }


def graph_replay_stats(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        SELECT start, end, deviceId, graphExecId
        FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
        ORDER BY start
        """
    ).fetchall()
    connection.close()

    per_device: dict[str, dict[str, float | int]] = {}
    primary_rows = []
    for device in sorted({row[2] for row in rows}):
        device_rows = [row for row in rows if row[2] == device]
        by_exec: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for row in device_rows:
            by_exec[row[3]].append(row)
        primary_exec, selected = max(
            by_exec.items(),
            key=lambda item: (
                len(item[1]),
                statistics.median((row[1] - row[0]) for row in item[1]),
            ),
        )
        primary_rows.extend(selected)
        durations = [(row[1] - row[0]) / 1.0e6 for row in selected]
        starts = [row[0] for row in selected]
        intervals = [
            (right - left) / 1.0e6 for left, right in zip(starts, starts[1:])
        ]
        per_device[str(device)] = {
            "graph_exec_id": primary_exec,
            "replays": len(selected),
            "duration_mean_ms": statistics.mean(durations),
            "duration_p50_ms": percentile(durations, 0.50),
            "duration_p90_ms": percentile(durations, 0.90),
            "duration_min_ms": min(durations),
            "duration_max_ms": max(durations),
            "start_interval_p50_ms": percentile(intervals, 0.50),
        }
    durations = [(row[1] - row[0]) / 1.0e6 for row in primary_rows]
    return {
        "primary_replays": len(primary_rows),
        "duration_p50_ms_across_ranks": percentile(durations, 0.50),
        "duration_p90_ms_across_ranks": percentile(durations, 0.90),
        "per_device": per_device,
    }


def runtime_api_stats(path: Path, patterns: Iterable[str]) -> dict[str, dict[str, float | int | None]]:
    connection = sqlite3.connect(path)
    output: dict[str, dict[str, float | int | None]] = {}
    for pattern in patterns:
        count, first, last, total_ms = connection.execute(
            """
            SELECT COUNT(*), MIN(runtime.start), MAX(runtime.end),
                   SUM(runtime.end - runtime.start) / 1.0e6
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            WHERE strings.value LIKE ?
            """,
            (pattern,),
        ).fetchone()
        output[pattern] = {
            "calls": count,
            "first_ms": first / 1.0e6 if first is not None else None,
            "last_ms": last / 1.0e6 if last is not None else None,
            "total_api_ms": total_ms,
        }
    connection.close()
    return output


def write_operator_csv(path: Path, operators: list[Operator]) -> None:
    fields = list(asdict(operators[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(operator) for operator in operators)


def write_category_csv(
    path: Path,
    fast: dict[str, dict[str, float]],
    vllm: dict[str, dict[str, float]],
) -> None:
    families = sorted(set(fast) | set(vllm))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "fastllm_calls_per_rank_token",
                "fastllm_gpu_us_per_rank_token",
                "vllm_calls_per_rank_token",
                "vllm_gpu_us_per_rank_token",
                "fastllm_minus_vllm_gpu_us_per_rank_token",
            ]
        )
        for family in families:
            fast_calls = family_value(fast, family, "calls_per_rank_token")
            fast_us = family_value(fast, family)
            vllm_calls = family_value(vllm, family, "calls_per_rank_token")
            vllm_us = family_value(vllm, family)
            writer.writerow([family, fast_calls, fast_us, vllm_calls, vllm_us, fast_us - vllm_us])


def report_markdown(summary: dict[str, object]) -> str:
    fast_meta = summary["trace_metadata"]["fastllm"]
    vllm_meta = summary["trace_metadata"]["vllm"]
    direct = summary["direct_comparisons"]
    fast_families = summary["families"]["fastllm"]
    vllm_families = summary["families"]["vllm"]
    fast_collective = summary["collectives"]["fastllm_custom_allreduce"]
    vllm_collective = summary["collectives"]["vllm_nccl_allreduce"]
    fast_windows = summary["intra_token_layer_windows"]["fastllm"]
    vllm_windows = summary["intra_token_layer_windows"]["vllm"]
    fast_graph = summary["graph_replays"]["fastllm"]
    vllm_graph = summary["graph_replays"]["vllm"]
    fast_lifecycle = summary["graph_lifecycle_node_trace"]["fastllm"]
    vllm_lifecycle = summary["graph_lifecycle_node_trace"]["vllm"]

    lines = [
        "# DeepSeek-V4 FastLLM / vLLM Nsight Systems 对比",
        "",
        "## 口径",
        "",
        f"- FastLLM：TP{fast_meta['ranks']}，{fast_meta['replay_tokens']} 个完整 CUDA Graph replay。",
        f"- vLLM：TP{vllm_meta['ranks']}，{vllm_meta['replay_tokens']} 个完整 CUDA Graph replay。",
        "- node trace 只统计 `graphId IS NOT NULL` 的稳态 decode kernel；prefill、FastLLM warm/capture 和采样尾部不计入逐算子表。",
        "- `GPU us/token` 是跨 rank 的平均 GPU-resident 时间，不等价于关键路径；NCCL 等待尤其不能按八卡总和解释成 wall time。",
        "",
        "## 直接可比算子",
        "",
        "| 算子族 | FastLLM | vLLM | Fast/vLLM | 每 token 差值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in direct:
        delta = item.get("delta_us_per_token", item["fastllm_minus_vllm"])
        lines.append(
            f"| {item['name']} | {item['fastllm']:.3f} {item['unit']} | "
            f"{item['vllm']:.3f} {item['unit']} | {item['fastllm_over_vllm']:.2f}× | "
            f"{delta:.1f} µs/token |"
        )

    lines.extend(
        [
            "",
            "## 稳态算子族 GPU-resident 时间",
            "",
            "| 算子族 | FastLLM µs/token | vLLM µs/token | Fast-vLLM |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in sorted(set(fast_families) | set(vllm_families)):
        fast_us = family_value(fast_families, family)
        vllm_us = family_value(vllm_families, family)
        lines.append(f"| {family} | {fast_us:.1f} | {vllm_us:.1f} | {fast_us - vllm_us:+.1f} |")

    lines.extend(
        [
            "",
            "## 通信分布",
            "",
            "| 实现 | p50 | p90 | p99 | mean | >100 µs 次数 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| FastLLM custom AR | {fast_collective['p50_us']:.2f} | {fast_collective['p90_us']:.2f} | {fast_collective['p99_us']:.2f} | {fast_collective['mean_us']:.2f} | {fast_collective['calls_over_100us']} |",
            f"| vLLM NCCL AR | {vllm_collective['p50_us']:.2f} | {vllm_collective['p90_us']:.2f} | {vllm_collective['p99_us']:.2f} | {vllm_collective['mean_us']:.2f} | {vllm_collective['calls_over_100us']} |",
            "",
            "通信 kernel 的 GPU-resident 时长包含等待；不能把跨 rank 总和直接解释成 wall time。",
            "",
            "## 单层波形与 stream overlap",
            "",
            "以相邻 sparse-attention 主核的起点作为层窗口，排除每 token 最后一层到下一 token 的边界：",
            "",
            "| 引擎 | 平均层窗口 | kernel 时长和 | union busy | overlap | overlap/时长和 | kernel 数/层 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| FastLLM | {fast_windows['span_mean_us']:.2f} µs | {fast_windows['duration_sum_mean_us']:.2f} µs | {fast_windows['union_busy_mean_us']:.2f} µs | {fast_windows['overlap_mean_us']:.2f} µs | {fast_windows['overlap_fraction_of_duration_sum'] * 100.0:.1f}% | {fast_windows['kernels_mean']:.2f} |",
            f"| vLLM | {vllm_windows['span_mean_us']:.2f} µs | {vllm_windows['duration_sum_mean_us']:.2f} µs | {vllm_windows['union_busy_mean_us']:.2f} µs | {vllm_windows['overlap_mean_us']:.2f} µs | {vllm_windows['overlap_fraction_of_duration_sum'] * 100.0:.1f}% | {vllm_windows['kernels_mean']:.2f} |",
            "",
            f"FastLLM overlap/时长和为 {fast_windows['overlap_fraction_of_duration_sum'] * 100.0:.1f}%，"
            f"vLLM 为 {vllm_windows['overlap_fraction_of_duration_sum'] * 100.0:.1f}%；"
            "层窗口比单个 kernel 更接近 decode 关键路径，但仍会受到 node tracing 扰动。",
            "",
            "## CUDA Graph 波形与生命周期",
            "",
            "| 引擎 | 主 graph replay | graph-trace replay p50 | 正式区间 capture | instantiate |",
            "|---|---:|---:|---:|---:|",
            f"| FastLLM | {fast_graph['primary_replays']} | {fast_graph['duration_p50_ms_across_ranks']:.2f} ms | {fast_lifecycle['cudaStreamBeginCapture%']['calls']} | {fast_lifecycle['cudaGraphInstantiate%']['calls']} |",
            f"| vLLM | {vllm_graph['primary_replays']} | {vllm_graph['duration_p50_ms_across_ranks']:.2f} ms | {vllm_lifecycle['cudaStreamBeginCapture%']['calls']} | {vllm_lifecycle['cudaGraphInstantiate%']['calls']} |",
            "",
            f"FastLLM node trace 中 capture/instantiate 分别为 "
            f"{fast_lifecycle['cudaStreamBeginCapture%']['calls']}/"
            f"{fast_lifecycle['cudaGraphInstantiate%']['calls']} 次；vLLM 为 "
            f"{vllm_lifecycle['cudaStreamBeginCapture%']['calls']}/"
            f"{vllm_lifecycle['cudaGraphInstantiate%']['calls']} 次。",
            "",
            "graph-trace replay 时长同样有 profiler 扰动，只用于观察 rank/graph 结构，不代替无 profiler 的吞吐数据。",
            "",
            "## 结论",
            "",
            "1. 优化优先级应以本轮表格中 FastLLM-vLLM 的正差值排序，不能沿用旧波形的瓶颈结论。",
            f"2. sparse attention core 当前 Fast/vLLM 为 {direct[0]['fastllm_over_vllm']:.2f}×；"
            "split 与 merge 都计入该算子族。",
            f"3. 层窗口 Fast/vLLM 为 {fast_windows['span_mean_us'] / vllm_windows['span_mean_us']:.2f}×，"
            "它同时反映单核时长、launch 空洞和 stream overlap。",
            f"4. custom AR 与 vLLM NCCL 的 p50 分别为 {fast_collective['p50_us']:.2f}/"
            f"{vllm_collective['p50_us']:.2f} us；通信长尾需结合时间线判断，不能只看总 GPU 时间。",
            "5. `category_summary.csv` 用于排优先级，`operator_summary.csv` 用于下钻到具体 kernel。",
            "",
            "完整精确 kernel 名称与数值见 `operator_summary.csv`，互斥算子族见 `category_summary.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fast_attention_pattern = resolve_name_pattern(
        args.fast_node, FAST_ATTENTION_PATTERNS
    )

    fast_operators, fast_meta = load_operators(
        args.fast_node,
        "fastllm",
        fast_family,
        fast_attention_pattern,
    )
    vllm_operators, vllm_meta = load_operators(
        args.vllm_node,
        "vllm",
        vllm_family,
        VLLM_ATTENTION_PATTERN,
    )
    fast_families = aggregate_families(fast_operators)
    vllm_families = aggregate_families(vllm_operators)

    direct = [
        comparison(
            "sparse attention core",
            family_value(fast_families, "sparse_attention_core") / 43.0,
            family_value(vllm_families, "sparse_attention_core") / 43.0,
            "µs/layer",
            43,
        ),
        comparison(
            "MoE router core",
            family_value(fast_families, "moe_router") / 40.0,
            family_value(vllm_families, "moe_router") / 40.0,
            "µs/routed-layer",
            40,
        ),
        comparison(
            "MHC helper chain",
            family_value(fast_families, "mhc"),
            family_value(vllm_families, "mhc"),
            "µs/token",
        ),
        comparison(
            "expert GEMM core",
            family_value(fast_families, "expert_gemm") / 43.0,
            family_value(vllm_families, "expert_gemm") / 43.0,
            "µs/layer",
            43,
        ),
    ]

    summary: dict[str, object] = {
        "trace_metadata": {"fastllm": fast_meta, "vllm": vllm_meta},
        "attention_anchor_patterns": {
            "fastllm": fast_attention_pattern,
            "vllm": VLLM_ATTENTION_PATTERN,
        },
        "direct_comparisons": direct,
        "families": {"fastllm": fast_families, "vllm": vllm_families},
        "collectives": {
            "fastllm_custom_allreduce": collective_stats(
                args.fast_node, "%FastllmCustomAllReduceKernel%"
            ),
            "vllm_nccl_allreduce": collective_stats(
                args.vllm_node, "%ncclDevKernel_AllReduce_Sum_bf16%"
            ),
        },
        "intra_token_layer_windows": {
            "fastllm": layer_window_stats(
                args.fast_node, fast_attention_pattern
            ),
            "vllm": layer_window_stats(
                args.vllm_node, VLLM_ATTENTION_PATTERN
            ),
        },
        "graph_replays": {
            "fastllm": graph_replay_stats(args.fast_graph),
            "vllm": graph_replay_stats(args.vllm_graph),
        },
        "graph_lifecycle_node_trace": {
            "fastllm": runtime_api_stats(
                args.fast_node,
                (
                    "cudaStreamBeginCapture%",
                    "cudaStreamEndCapture%",
                    "cudaGraphInstantiate%",
                    "cudaGraphLaunch%",
                ),
            ),
            "vllm": runtime_api_stats(
                args.vllm_node,
                (
                    "cudaStreamBeginCapture%",
                    "cudaStreamEndCapture%",
                    "cudaGraphInstantiate%",
                    "cudaGraphLaunch%",
                ),
            ),
        },
        "caveats": [
            "Nsight node tracing strongly perturbs multi-process rank scheduling and CUDA graph APIs.",
            "Kernel GPU-resident sums are attribution metrics, not wall-clock critical paths.",
            "The linear/projection family is approximate because the two engines fuse quantization and reductions differently.",
        ],
    }

    write_operator_csv(
        args.output_dir / "operator_summary.csv", fast_operators + vllm_operators
    )
    write_category_csv(
        args.output_dir / "category_summary.csv", fast_families, vllm_families
    )
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison_report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
