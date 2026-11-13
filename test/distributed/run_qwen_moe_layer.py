#!/usr/bin/env python3
"""阶段 2：通过正式进程入口验收 MoE 权重与单层；只调用已有产物，不触发编译。"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

from run_process_tp import ROOT, capture, case_process, parse_report, save_report, wait_case
from run_qwen3_shards import decode, save_fixture, encode
from qwen_moe_reference import fixture, forward, weights_for_rank

BACKEND = "qwen2_moe_attention_dp_ep_fp16_nccl"
ATOL, RTOL = 0.008, 0.008


def require(condition, reason):
    """/** 拒绝不完整验收证据，失败消息定位到 case/task/rank/张量。 */"""
    if not condition:
        raise AssertionError(reason)


def compare(actual, expected, metrics, label, exact=False):
    """/** 累计真实输出误差和失败元素数；形状或非有限值也必须失败。 */"""
    require(len(actual) == len(expected), f"{label}: shape {len(actual)} != {len(expected)}")
    failures = 0
    for a, b in zip(actual, expected):
        error = abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("inf")
        mismatch = not (math.isfinite(a) and math.isfinite(b)) or error > (0 if exact else ATOL + RTOL * abs(b))
        metrics["max"] = max(metrics["max"], error)
        metrics["sum"] += error
        metrics["count"] += 1
        metrics["failed"] += int(mismatch)
        failures += int(mismatch)
    require(not failures, f"{label}: {failures} elements exceed tolerance")


def tasks_for(world, hotspot=False):
    """/** 每 rank 独立序列；包含变长、空闲、单 Token、不同层与重复运行。 */"""
    tasks = []
    for name, counts, stage, layer in [
        ("full_mixed", [3, 0, 1, 5], 0, 0), ("attention", [3, 0, 1, 5], 1, 0),
        ("moe", [3, 0, 1, 5], 2, 0), ("decode_mixed", [1, 0, 1, 1], 0, 0),
        ("decode_all", [1, 1, 1, 1], 2, 0), ("idle", [0, 0, 0, 0], 0, 0),
        ("second_layer", [2, 4, 0, 1], 0, 1)]:
        inputs = []
        for rank, count in enumerate(counts[:world]):
            values = [0.4 + math.sin((i + 1) * 0.7 + rank * 1.3) * 0.3 for i in range(count * 8)]
            if hotspot:
                values = [1.0 if i % 8 == 0 else 0.0 for i in range(count * 8)]
            inputs.append(dict(position_begin=rank * 11, hidden=values))
        tasks.append(dict(name=name, layer=layer, hidden=8, stage=stage, inputs=inputs))
    causal = copy.deepcopy(tasks[0])
    causal["name"] = "causal_future"
    for input_ in causal["inputs"]:
        if input_["hidden"]:
            input_["hidden"][-8:] = [x + 3 for x in input_["hidden"][-8:]]
    tasks.append(causal)
    repeat = copy.deepcopy(tasks[0])
    repeat["name"] = "repeat_full"
    tasks.append(repeat)
    return tasks


def cases():
    """/** 支持布局与拒绝路径矩阵；阈值固定，不因 GPU 失败自动放宽。 */"""
    result = [("f16_bias", {}), ("bf16_no_bias", dict(dtype="BF16", bias=False)),
              ("f32_normalized", dict(dtype="F32", normalize=True)),
              ("indexed_tied", dict(indexed=True, tied=True)),
              ("zero_local_experts", dict(experts=3, topk=1)),
              ("hotspot", dict(hotspot=True, topk=1)),
              ("topk_all", dict(experts=5, topk=5, normalize=True)),
              ("official_expert_counts", dict(experts=60, topk=4)),
              ("context_boundary", dict(boundary=True))]
    for name, patch, reason in [
        ("quantized", {"quantization_config": {}}, "quantized checkpoints"),
        ("sparse_step", {"decoder_sparse_step": 2}, "decoder_sparse_step"),
        ("topk", {"num_experts_per_tok": 6}, "expert dimensions"),
        ("norm_topk", {"norm_topk_prob": "false"}, "norm_topk_prob"),
        ("rope", {"rope_scaling": {"type": "linear", "factor": 2}}, "unsupported Qwen3 forward"),
        ("activation", {"hidden_act": "gelu"}, "unsupported Qwen3 forward"),
        ("model_type", {"model_type": "qwen3"}, "expected model_type=qwen2_moe")]:
        result.append(("reject_" + name, dict(config=patch, reason=reason, group_failed=True)))
    for scenario, reason in [("unloaded", "MoE weights not loaded"), ("double_load", "already loaded"),
                             ("dense_api", "dense layer requires"), ("source_limit", "source tensor limit"),
                             ("device_limit", "device weight limit"), ("graph", "requires Eager"),
                             ("prefix", "requires Eager")]:
        result.append(("reject_" + scenario, dict(scenario=scenario, reason=reason, group_failed=True)))
    result += [("reject_foreign_missing", dict(missing=True, reason="missing tensor", group_failed=True)),
               ("reject_nonfinite_weight", dict(nan_weight=True, reason="nonfinite/out-of-range", group_failed=True)),
               ("reject_router_overflow", dict(router_overflow=True, reason="nonfinite MoE", group_failed=True)),
               ("reject_partial_bias", dict(partial_bias=True, reason="incomplete QKV", group_failed=True)),
               ("reject_dtype", dict(bad_dtype=True, reason="unsupported dtype", group_failed=True)),
               ("reject_input", dict(bad_input=True, reason="outside finite FP16", group_failed=False)),
               ("reject_shape", dict(bad_shape=True, reason="input/protocol size", group_failed=False)),
               ("reject_stage", dict(bad_stage=True, reason="shape/stage/timeout", group_failed=False)),
               ("reject_context", dict(over_context=True, reason="input/protocol size", group_failed=False)),
               ("reject_layer", dict(bad_layer=True, reason="invalid/unsupported", group_failed=True))]
    return result


def make_case(directory, spec, world):
    """/** 保存固定输入和模型，故障 case 仅改变被验证的约束，不修改执行后端。 */"""
    options = {k: v for k, v in spec.items() if k in {"dtype", "bias", "normalize", "experts", "topk", "tied", "indexed", "hotspot"}}
    config, tensors = fixture(directory / "model", **options)
    config.update(spec.get("config", {}))
    tasks = tasks_for(world, spec.get("hotspot", False))
    if spec.get("missing"):
        del tensors["model.layers.0.mlp.experts.4.down_proj.weight"]
    if spec.get("nan_weight"):
        tensor = tensors["model.layers.0.mlp.experts.4.down_proj.weight"]
        values = decode(tensor["data"], tensor["dtype"])
        values[0] = float("nan")
        tensor["data"] = encode(values, tensor["dtype"])
    if spec.get("router_overflow"):
        tensor = tensors["model.layers.0.mlp.gate.weight"]
        tensor["data"] = encode([65504.0]*math.prod(tensor["shape"]), tensor["dtype"])
        tasks = [dict(name="router_overflow", layer=0, hidden=8, stage=2,
                      inputs=[dict(position_begin=0, hidden=[1.0]*8 if rank == 0 else []) for rank in range(world)])]
    if spec.get("partial_bias"):
        del tensors["model.layers.0.self_attn.v_proj.bias"]
    if spec.get("bad_dtype"):
        tensors["model.norm.weight"]["dtype"] = "I16"
    if spec.get("bad_input"):
        tasks[0]["inputs"][0]["hidden"][0] = 70000.0
    if spec.get("bad_shape"):
        tasks[0]["inputs"][0]["hidden"].pop()
    if spec.get("bad_stage"):
        tasks[0]["stage"] = 3
    if spec.get("bad_layer"):
        tasks[0]["layer"] = config["num_hidden_layers"]
    if spec.get("boundary") or spec.get("over_context"):
        count = 2049 if spec.get("over_context") else 2048
        tasks = [dict(name="context_boundary", layer=0, hidden=8, stage=2,
                      inputs=[dict(position_begin=0, hidden=[0.25]* (count*8) if rank == 0 else []) for rank in range(world)])]
    save_fixture(directory / "model", config, tensors, spec.get("indexed", False))
    (directory / "tasks.json").write_text(json.dumps(tasks))
    return config, tensors, tasks


def verify(report, directory, devices, config, tensors, tasks, metrics):
    """/** 检查真实 GPU 权重、各 rank 数值、通信路径与专家工作量；缺任一证据即失败。 */"""
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, str(report))
    workers = report["workers"]
    world = len(devices)
    require(len(workers) == world and len({w["pid"] for w in workers}) == world, "worker identities")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank] and worker["pid"] > 0, "worker mapping")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing CUDA/NCCL evidence")
    require(report.get("workers_reaped") and report.get("reload_passed"), "missing lifecycle evidence")
    require(len(report["weights"]) == world, "missing rank weights")
    for rank, plan in enumerate(report["weights"]):
        expected = weights_for_rank(config, tensors, rank, world)
        actual = {x["name"]: x for x in plan["tensors"]}
        require(len(actual) == len(plan["tensors"]) and actual.keys() == expected.keys(), f"rank {rank}: wrong expert ownership/tensors")
        require(plan["source_bytes_read"] == sum(x["source_bytes"] for x in expected.values()), "foreign expert payload read")
        require(plan["device_bytes"] == sum(len(x["values"])*2 for x in expected.values()), "weight memory accounting")
        for name, tensor in actual.items():
            reference = expected[name]
            require(tensor["shape"] == reference["shape"], f"rank {rank}: shape {name}")
            values = decode((directory / "gpu" / tensor["file"]).read_bytes(), "F16")
            compare(values, reference["values"], metrics, f"rank {rank}: GPU weight {name}", exact=True)
    results = report["results"]
    require(len(results) == len(tasks), "missing task results")
    references = []
    for task, result in zip(tasks, results):
        require(result["task"] == task["name"] and len(result["ranks"]) == world, "wrong task/rank count")
        refs = [forward(config, tensors, task, item) for item in task["inputs"]]
        references.append(dict(task=task["name"], ranks=refs))
        expected_counts = [0]*config["num_experts"]
        for ref in refs:
            for expert in ref["selected_experts"]:
                expected_counts[expert] += 1
        capacity = max([1] + [len(x["hidden"])//config["hidden_size"] for x in task["inputs"]])
        for rank, (actual, ref) in enumerate(zip(result["ranks"], refs)):
            label = f"{task['name']} rank={rank} shape=({len(task['inputs'][rank]['hidden'])//8},8)"
            for key, values in ref.items():
                compare(actual[key], values, metrics, label + " " + key, exact=key == "selected_experts")
            is_moe = task["stage"] != 1
            require(actual["allgather_calls"] == (3 if is_moe else 0) and actual["reduce_scatter_calls"] == int(is_moe), label + " wrong NCCL path")
            require(actual["indexed_decode"] == (is_moe and capacity == 1), label + " wrong expert kernel path")
            expected = [count if e % world == rank else 0 for e, count in enumerate(expected_counts)] if is_moe else []
            require(actual["expert_tokens"] == expected, label + " wrong local expert work/padding")
    (directory / "reference.json").write_text(json.dumps(references))
    by_name = {result["task"]: result for result in results}
    if "repeat_full" in by_name:
        for rank in range(world):
            before = by_name["full_mixed"]["ranks"][rank]
            require(before == by_name["repeat_full"]["ranks"][rank], f"rank {rank}: repeated execution differs")
            after = by_name["causal_future"]["ranks"][rank]
            compare(after["output"][:-8], before["output"][:-8], metrics, f"rank {rank}: future-token leakage")


def invoke(args, directory, devices, scenario):
    """/** 运行已有 C++ 检查器；保留完整命令与原始日志，超时回收整个进程会话。 */"""
    command = [str(args.check.resolve()), "--worker", str(args.worker.resolve()),
               "--model", str(directory / "model"), "--tasks", str(directory / "tasks.json"),
               "--output", str(directory / "gpu"), "--devices", ",".join(map(str, devices)), "--case", scenario]
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        with case_process(command, stdout=stdout, stderr=stderr) as process:
            code = wait_case(process, 600)
    return code, parse_report((directory / "stdout.log").read_text())


def main():
    """/** 执行 1/2/4 rank 矩阵，任何失败返回非零，提前失败也保存四种格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--build-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if len(devices) not in (1, 2, 4) or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("devices must contain 1, 2 or 4 distinct logical IDs")
    output = args.output_root.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_moe_layer_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    sizes = [size for size in (1, 2, 4) if size <= len(devices)]
    matrix = [(f"ep{size}_{name}", spec, devices[:size]) for size in sizes for name, spec in cases()]
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
                    git_status=capture(["git", "status", "--short"]), seed=20261113, warmup=0, iterations=1, repeats=1,
                    gpu=capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), atol=ATOL, rtol=RTOL,
                    expected_case_count=len(matrix), report_title="Qwen MoE EP weight and layer correctness",
                    report_description="真实 GPU 权重逐元素读回，单层与独立 CPU 参考比较；不代表完整模型生成或性能。",
                    cuda_runtime_versions=[], nccl_versions=[])
    rows = []
    try:
        cache = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in cache.splitlines() if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        for name, spec, selected in matrix:
            directory = output / name
            directory.mkdir()
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            row = dict(case=name, backend=BACKEND, status="FAIL", reason="", spec=spec, ranks=len(selected),
                       max_abs_error=None, mean_abs_error=None, failed_elements=None)
            try:
                config, tensors, tasks = make_case(directory, spec, len(selected))
                row["input_shapes"] = [[len(item["hidden"])//8, 8] for item in tasks[0]["inputs"]]
                code, report = invoke(args, directory, selected, spec.get("scenario", "forward"))
                row["workers"] = report.get("workers", [])
                metadata["cuda_runtime_versions"] = sorted(set(metadata["cuda_runtime_versions"]) | {w["cuda_runtime_version"] for w in row["workers"]})
                metadata["nccl_versions"] = sorted(set(metadata["nccl_versions"]) | {w["nccl_version"] for w in row["workers"]})
                if "reason" in spec:
                    require(code == 1 and report.get("status") == "FAIL" and report.get("backend") == BACKEND, str(report))
                    require(spec["reason"] in report.get("error", "") and report.get("group_failed") is spec["group_failed"], str(report))
                    require(report.get("shutdown_complete") and not report.get("cleanup_error"), str(report))
                    if spec["group_failed"]:
                        require(report.get("workers_reaped"), "failed worker group not reaped")
                    row["observed_error"] = report["error"]
                else:
                    require(code == 0, str(report))
                    verify(report, directory, selected, config, tensors, tasks, metrics)
                    row["paths"] = sorted({"indexed_decode" if r["indexed_decode"] else "prefill_cublas" if r["allgather_calls"] else "attention"
                                           for result in report["results"] for r in result["ranks"]})
                row["status"] = "PASS"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            if metrics["count"]:
                row.update(max_abs_error=metrics["max"], mean_abs_error=metrics["sum"]/metrics["count"], failed_elements=metrics["failed"])
            rows.append(row)
            print(name, row["status"], row["reason"], flush=True)
            save_report(output, rows, metadata)
    except Exception as error:
        rows.append(dict(case="runner_setup", backend=BACKEND, status="FAIL", reason=str(error)))
    finally:
        summary = save_report(output, rows, metadata)
        print(summary, output, flush=True)
    return 0 if summary == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
