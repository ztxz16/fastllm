#!/usr/bin/env python3
"""单层真实 GPU TP 验收：单卡、全部 TP rank、独立 CPU 数学参考三方比较。"""

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
from run_qwen3_shards import fixture, encode, decode, save_fixture
from qwen3_layer_reference import forward

BACKEND = "qwen3_process_layer_cublas_nccl"
ATOL, RTOL = 0.004, 0.004


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected, metrics, label, exact=False):
    require(len(actual) == len(expected), label + ": output size mismatch")
    failed = 0
    for a, b in zip(actual, expected):
        finite = math.isfinite(a) and math.isfinite(b)
        error = abs(a - b) if finite else float("inf")
        mismatch = not finite or error > (0 if exact else ATOL + RTOL * abs(b))
        metrics["max"] = max(metrics["max"], error)
        metrics["sum"] += error
        metrics["count"] += 1
        metrics["failed"] += int(mismatch)
        failed += int(mismatch)
    require(failed == 0, f"{label}: {failed} elements exceed tolerance")


def make_fixture(directory, spec):
    config, tensors = fixture(directory / "model", dtype=spec.get("dtype", "F16"),
        bias=spec.get("bias", False), hidden_size=16, head_dim=spec.get("head_dim", 4), num_attention_heads=6,
        num_key_value_heads=spec.get("kv", 3), intermediate_size=19, vocab_size=17)
    config.update(rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="silu")
    config.update(spec.get("config", {}))
    for name, tensor in tensors.items():
        values = decode(tensor["data"], tensor["dtype"])
        values = [1 + value * 0.01 if "norm.weight" in name else value * 0.02 for value in values]
        tensor["data"] = encode(values, tensor["dtype"])
    save_fixture(directory / "model", config, tensors)
    tasks = []
    tokens = spec.get("tokens", 5)
    for stage in (1, 2, 0):
        tasks.append(dict(name=f"stage-{stage}", layer=spec.get("layer", 1), tokens=tokens,
                          hidden=config["hidden_size"], position_begin=spec.get("position", 7), stage=stage,
                          input=[((i * 17) % 97 - 48) / 32 for i in range(tokens * config["hidden_size"])]))
    if spec.get("bad_shape"):
        tasks[0]["hidden"] += 1
    if spec.get("bad_layer"):
        tasks[0]["layer"] = config["num_hidden_layers"]
    if spec.get("bad_stage"):
        tasks[0]["stage"] = 3
    # 后缀变化不能影响前缀；第二次任务也检查临时缓冲与输入状态没有残留。
    changed = copy.deepcopy(tasks[-1])
    changed["name"] = "changed-tail"
    changed["input"][-config["hidden_size"]:] = [0.25] * config["hidden_size"]
    tasks.append(changed)
    (directory / "tasks.json").write_text(json.dumps(tasks, allow_nan=False))
    return config, tensors, tasks


def invoke(binary, worker, directory, devices, label, scenario="forward"):
    command = [str(binary), "--worker", str(worker), "--model", str(directory / "model"),
               "--tasks", str(directory / "tasks.json"), "--devices", ",".join(map(str, devices)), "--case", scenario]
    (directory / f"{label}.command.txt").write_text(shlex.join(command) + "\n")
    with (directory / f"{label}.stdout.log").open("wb") as stdout, (directory / f"{label}.stderr.log").open("wb") as stderr:
        try:
            with case_process(command, stdout=stdout, stderr=stderr) as process:
                code = wait_case(process, 600)
        except BaseException as error:
            stderr.write(f"\n{type(error).__name__}: {error}\n".encode())
            raise
    return code, parse_report((directory / f"{label}.stdout.log").read_text())


def verify(report, devices, tasks):
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, str(report))
    workers = report["workers"]
    require(len(workers) == len(devices) and len({w["pid"] for w in workers}) == len(devices), "worker count/PIDs")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank] and worker["pid"] > 0, "worker identity")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing CUDA/NCCL")
    require(len(report["results"]) == len(tasks), "missing task results")
    for task, result in zip(tasks, report["results"]):
        require(result["task"] == task["name"] and len(result["outputs"]) == len(devices), "task identity/ranks")
        for output in result["outputs"]:
            require(len(output) == len(task["input"]), "wrong output shape")


def verify_failure(code, report, spec):
    require(code == 1 and report.get("status") == "FAIL" and report.get("backend") == BACKEND, str(report))
    require(spec["reason"] in report.get("error", "") and report.get("group_failed") is spec["group_failed"], str(report))
    require("cleanup_error" not in report, str(report))
    require(report.get("shutdown_complete") is True, "missing successful shutdown confirmation")
    if spec["group_failed"]:
        require(report.get("workers_reaped") is True, "failed group left workers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--build-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(part) for part in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error("this acceptance suite requires exactly two distinct logical GPUs")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_layer_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    cases = [(dtype, dict(dtype=dtype)) for dtype in ("F16", "BF16", "F32")]
    cases += [("bias", dict(bias=True)), ("kv_replication", dict(kv=1)), ("single_token", dict(tokens=1)),
              ("max_tokens", dict(tokens=128)), ("layer_zero", dict(layer=0, position=0)),
              ("head_dim_128", dict(head_dim=128)), ("max_head_dim", dict(head_dim=256)),
              ("large_position", dict(position=65536)),
              ("max_position", dict(position=1048576)),
              ("reject_zero_tokens", dict(tokens=0, reason="invalid Qwen3 layer request", group_failed=False)),
              ("reject_tokens", dict(tokens=129, reason="invalid Qwen3 layer request", group_failed=False)),
              ("reject_position", dict(position=1048577, reason="invalid Qwen3 layer request", group_failed=False)),
              ("reject_stage", dict(bad_stage=True, reason="invalid Qwen3 layer request", group_failed=False)),
              ("reject_odd_head_dim", dict(head_dim=3, reason="invalid/unsupported Qwen3 layer", group_failed=True)),
              ("reject_head_dim", dict(head_dim=258, reason="invalid/unsupported Qwen3 layer", group_failed=True)),
              ("reject_layer", dict(bad_layer=True, reason="invalid/unsupported Qwen3 layer", group_failed=True)),
              ("reject_shape", dict(bad_shape=True, reason="invalid Qwen3 layer request", group_failed=False)),
              ("reject_rope", dict(config=dict(rope_scaling={"type": "linear", "factor": 2}),
                                   reason="unsupported Qwen3 forward configuration", group_failed=True)),
              ("reject_activation", dict(config=dict(hidden_act="gelu"), reason="unsupported Qwen3 forward configuration", group_failed=True)),
              ("reject_sliding", dict(config=dict(use_sliding_window=True), reason="unsupported Qwen3 forward configuration", group_failed=True)),
              ("reject_epsilon", dict(config=dict(rms_norm_eps=-1), reason="unsupported Qwen3 forward configuration", group_failed=True)),
              ("reject_unloaded", dict(scenario="unloaded", reason="weights not loaded", group_failed=True))]
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices, atol=ATOL, rtol=RTOL,
        cuda_runtime_versions=[], nccl_versions=[],
        expected_case_count=len(cases), report_title="Qwen3 single-layer TP correctness",
        report_description="完整层、Attention、MLP 均经同一公开入口；比较 CPU 数学参考/单卡/双卡，检查因果掩码。非性能测试。")
    rows = []
    try:
        metadata["build_cache"] = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in metadata["build_cache"].splitlines()
            if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        for name, spec in cases:
            directory = output / name
            directory.mkdir()
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            row = dict(case=name, backend=BACKEND, status="FAIL", reason="", input=spec,
                       max_abs_error=None, mean_abs_error=None, failed_elements=None)
            try:
                config, tensors, tasks = make_fixture(directory, spec)
                row["shape"] = [tasks[0]["tokens"], tasks[0]["hidden"]]
                code, tp = invoke(args.check.resolve(), args.worker.resolve(), directory, devices, "tp", spec.get("scenario", "forward"))
                row["workers"] = tp.get("workers", [])
                if row["workers"]:
                    metadata["cuda_runtime_versions"] = sorted({w["cuda_runtime_version"] for w in row["workers"]})
                    metadata["nccl_versions"] = sorted({w["nccl_version"] for w in row["workers"]})
                if "reason" in spec:
                    verify_failure(code, tp, spec)
                else:
                    require(code == 0, str(tp))
                    verify(tp, devices, tasks)
                    code, single = invoke(args.check.resolve(), args.worker.resolve(), directory, devices[:1], "single")
                    require(code == 0, str(single))
                    verify(single, devices[:1], tasks)
                    refs = [forward(config, tensors, task) for task in tasks]
                    (directory / "reference.json").write_text(json.dumps(refs))
                    for i, task in enumerate(tasks):
                        baseline = single["results"][i]["outputs"][0]
                        compare(baseline, refs[i], metrics, f"single vs CPU {task['name']}")
                        for rank, actual in enumerate(tp["results"][i]["outputs"]):
                            compare(actual, refs[i], metrics, f"rank {rank} vs CPU {task['name']}")
                            compare(actual, baseline, metrics, f"rank {rank} vs single {task['name']}")
                            compare(actual, tp["results"][i]["outputs"][0], metrics, "rank consistency", exact=True)
                    for rank in range(len(devices)):
                        before = tp["results"][2]["outputs"][rank][:-config["hidden_size"]]
                        after = tp["results"][3]["outputs"][rank][:-config["hidden_size"]]
                        compare(after, before, metrics, "causal prefix", exact=True)
                row["status"] = "PASS"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            finally:
                if metrics["count"]:
                    mean = metrics["sum"] / metrics["count"]
                    row.update(max_abs_error=metrics["max"] if math.isfinite(metrics["max"]) else "inf",
                               mean_abs_error=mean if math.isfinite(mean) else "inf",
                               failed_elements=metrics["failed"], compared_elements=metrics["count"])
                rows.append(row)
                save_report(output, rows, metadata)
    except BaseException as error:
        rows.append(dict(case="runner", backend="python", status="FAIL", reason=f"{type(error).__name__}: {error}"))
        raise
    finally:
        status = save_report(output, rows, metadata)
    print(f"{status}: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
