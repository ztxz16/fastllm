#!/usr/bin/env python3
"""通过真实多进程公开入口检查 GPU 权重；不编译，不执行模型前向或性能测试。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import struct
import sys
import uuid

from run_process_tp import ROOT, capture, case_process, parse_report, save_report, wait_case
from run_qwen3_shards import decode, encode, fixture, reference, save_fixture

BACKEND = "qwen3_process_cuda_fp16"


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def fp16_reference(target):
    values = decode(target["data"], target["dtype"])
    if any(not math.isfinite(value) or abs(value) > 65504 for value in values):
        raise ValueError("source outside finite FP16 range")
    # Python struct 是独立于 CUDA host intrinsic 的 IEEE round-to-nearest-even 参考。
    return struct.pack("<" + "e" * len(values), *values)


def compare_payload(actual, desired, metrics, label):
    require(len(actual) == len(desired), f"{label}: byte length mismatch")
    for index, (value, expected) in enumerate(zip(decode(actual, "F16"), decode(desired, "F16"))):
        mismatch = actual[index * 2:index * 2 + 2] != desired[index * 2:index * 2 + 2]
        error = abs(value - expected) if math.isfinite(value) else float("inf")
        metrics["max"] = max(metrics["max"], error)
        metrics["sum"] += error
        metrics["count"] += 1
        metrics["failed"] += int(mismatch)
    require(actual == desired, f"{label}: GPU payload differs from independent FP16 reference")


def verify(report, output, config, tensors, devices, repeats, metrics, reload_reference=None):
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, str(report))
    workers = report["workers"]
    require(len(workers) == len(devices), "worker count mismatch")
    require(len({worker["pid"] for worker in workers}) == len(devices), "workers share a PID")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank], "worker identity mismatch")
        require(isinstance(worker["pid"], int) and worker["pid"] > 0, "invalid worker PID")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing CUDA/NCCL")
    require(len(report["cycles"]) == repeats, "missing load/reload cycle")
    for iteration, cycle in enumerate(report["cycles"]):
        active_config, active_tensors = (config, tensors) if iteration == 0 or reload_reference is None else reload_reference
        require(cycle["directory"] == f"cycle-{iteration}", "wrong cycle directory")
        require(len(cycle["ranks"]) == len(devices), "missing rank")
        for rank, result in enumerate(cycle["ranks"]):
            require(result["rank"] == rank, "wrong rank")
            expected, _ = reference(active_config, active_tensors, rank, len(devices))
            entries = {item["name"]: item for item in result["tensors"]}
            require(len(entries) == len(result["tensors"]) and entries.keys() == expected.keys(), "tensor list mismatch")
            total = 0
            for name, target in expected.items():
                entry = entries[name]
                label = f"rank={rank} cycle={iteration} tensor={name} shape={target['shape']} backend={BACKEND}"
                require(entry["dtype"] == "F16" and entry["source_dtype"] == target["dtype"], label + ": dtype")
                require(entry["shape"] == target["shape"], label + ": shape")
                require(Path(entry["file"]).name == entry["file"], label + ": unsafe filename")
                desired = fp16_reference(target)
                actual = (output / cycle["directory"] / entry["file"]).read_bytes()
                require(entry["bytes"] == len(desired), label + ": size")
                compare_payload(actual, desired, metrics, label)
                total += len(desired)
            require(result["device_bytes"] == total, "GPU byte accounting mismatch")
            require(result["source_bytes_read"] == sum(len(t["data"]) for t in expected.values() if not t["zero"]),
                    "source byte accounting mismatch")


def verify_rejection(code, report, reason):
    require(code == 1 and report.get("status") == "FAIL" and report.get("backend") == BACKEND,
            f"expected explicit checker failure, got {code}: {report}")
    require(reason in report.get("error", ""), f"wrong rejection reason: {report}")
    require(report.get("group_failed") is True and report.get("workers_reaped") is True,
            "failed model load did not terminate/reap the whole group")
    require("cleanup_error" not in report, "cleanup failed")


def prepare(directory, spec):
    options = dict(spec.get("fixture", {}))
    config, tensors = fixture(directory / "model", **options)
    mode = spec.get("values")
    if mode == "finite_half_exhaustive":
        name = "model.embed_tokens.weight"
        values = [struct.unpack("<e", struct.pack("<H", bits))[0] if bits & 0x7c00 != 0x7c00 else 0.0
                  for bits in range(65536)]
        require(math.prod(tensors[name]["shape"]) == len(values), "exhaustive fixture size")
        tensors[name]["data"] = encode(values, "F16")
    elif mode == "rounding":
        values = [0.0, -0.0, 2**-25, -2**-25, 2**-24, 3 * 2**-25,
                  2**-14 - 2**-25, 2**-14, 1 - 2**-12, 1 + 2**-11,
                  1 + 3 * 2**-11, -1 - 2**-11, 65504.0, -65504.0]
        for tensor in tensors.values():
            tensor["data"] = encode([values[i % len(values)] for i in range(math.prod(tensor["shape"]))], "F32")
    elif mode in ("nan", "inf", "overflow"):
        # 最后一词表行只属于最后 rank；验证其他 rank 已成功或正在加载时整组失败。
        name = "lm_head.weight"
        tensor = tensors[name]
        values = decode(tensor["data"], tensor["dtype"])
        values[-1] = {"nan": float("nan"), "inf": float("inf"), "overflow": 65536.0}[mode]
        tensor["data"] = encode(values, tensor["dtype"])
    if mode:
        save_fixture(directory / "model", config, tensors, options.get("indexed", False))
    return config, tensors


def execute(checker, worker, directory, devices, spec):
    command = [str(checker), "--worker", str(worker), "--model", str(directory / "model"),
               "--devices", ",".join(map(str, devices)), "--output", str(directory / "gpu"),
               "--case", spec.get("case", "load"), *spec.get("extra", [])]
    if spec.get("case") == "reload":
        command += ["--reload-model", str(directory / "reload-model")]
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            with case_process(command, stdout=stdout, stderr=stderr) as process:
                code = wait_case(process, 420)
        except BaseException as error:
            stderr.write(f"\nchecker invocation failed: {type(error).__name__}: {error}\n".encode())
            raise
    return code, parse_report((directory / "stdout.log").read_text())


def prepare_reload(directory, config, tensors, indexed=False):
    # 第二轮使用同形状、不同数值的模型，防止残留 GPU 权重被同一份输入掩盖。
    updated = {name: dict(tensor, data=encode([value + 0.25 for value in decode(tensor["data"], tensor["dtype"])],
                                             tensor["dtype"])) for name, tensor in tensors.items()}
    path = directory / "reload-model"
    path.mkdir()
    save_fixture(path, config, updated, indexed)
    return config, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--build-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(part) for part in args.devices.split(",")]
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0 or len(devices) > 8:
        parser.error("provide 1..8 distinct nonnegative logical devices")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_device_weights_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    cases = [(dtype, dict(fixture=dict(dtype=dtype))) for dtype in ("F16", "BF16", "F32")]
    cases += [
        ("reload_tied_bias_indexed", dict(case="reload", fixture=dict(dtype="BF16", tied=True, bias=True, indexed=True))),
        ("kv_replication", dict(fixture=dict(num_key_value_heads=1))),
        ("stop_loaded", dict(case="stop_loaded")),
        ("rounding", dict(values="rounding", fixture=dict(dtype="F32"))),
        ("all_finite_fp16", dict(values="finite_half_exhaustive", fixture=dict(hidden_size=512, vocab_size=128))),
        ("chunk_boundary", dict(fixture=dict(hidden_size=1024, vocab_size=513))),
        ("exact_limits", dict(limits_delta=0)),
        ("above_limits", dict(limits_delta=1)),
        ("reject_source_cap", dict(extra=["--max-tensor-bytes", "2"], reason="Qwen3 source tensor limit")),
        ("reject_device_cap", dict(extra=["--max-device-bytes", "2"], reason="Qwen3 device weight limit")),
        ("reject_duplicate", dict(case="duplicate", reason="already loaded")),
        ("reject_unloaded_read", dict(case="unloaded_read", reason="weights not loaded")),
        ("reject_index", dict(case="invalid_index", reason="tensor index out of range")),
        ("reject_range", dict(case="invalid_range", reason="readback out of range")),
    ]
    for dtype in ("F16", "BF16", "F32"):
        for mode in ("nan", "inf", "overflow"):
            if dtype == "F16" and mode == "overflow":
                continue
            cases.append((f"reject_{dtype}_{mode}", dict(fixture=dict(dtype=dtype), values=mode,
                                                       reason="nonfinite/out-of-range FP16 value")))
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
                    git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
                    test_kind="GPU weight residency correctness; no inference or performance claims",
                    gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices,
                    cuda_runtime_versions=[], nccl_versions=[],
                    expected_case_count=len(cases), report_title="Qwen3 process GPU weight correctness",
                    report_description="对实际 GPU 读回的全部 FP16 字节与独立 Python 参考进行比较；含卸载重载及整组失败清理。",
                    checker=str(args.check.resolve()), worker=str(args.worker.resolve()))
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
                config, tensors = prepare(directory, spec)
                reload_reference = None
                if spec.get("case") == "reload":
                    reload_reference = prepare_reload(directory, config, tensors, spec.get("fixture", {}).get("indexed", False))
                if "limits_delta" in spec:
                    targets = [reference(config, tensors, rank, len(devices))[0] for rank in range(len(devices))]
                    source_limit = max(len(t["data"]) for rank in targets for t in rank.values())
                    device_limit = max(sum(len(fp16_reference(t)) for t in rank.values()) for rank in targets)
                    spec = dict(spec, extra=["--max-tensor-bytes", str(source_limit + spec["limits_delta"]),
                                             "--max-device-bytes", str(device_limit + spec["limits_delta"])])
                code, report = execute(args.check.resolve(), args.worker.resolve(), directory, devices, spec)
                row["worker_report"] = report
                if report.get("workers"):
                    metadata["cuda_runtime_versions"] = sorted({w["cuda_runtime_version"] for w in report["workers"]})
                    metadata["nccl_versions"] = sorted({w["nccl_version"] for w in report["workers"]})
                if "reason" in spec:
                    verify_rejection(code, report, spec["reason"])
                else:
                    require(code == 0, f"checker exit {code}: {report}")
                    verify(report, directory / "gpu", config, tensors, devices,
                           2 if spec.get("case") == "reload" else 1, metrics, reload_reference)
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
