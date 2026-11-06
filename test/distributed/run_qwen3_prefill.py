#!/usr/bin/env python3
"""完整 Qwen3 prefill 验收；只运行已有产物，不编译。--model 指定真实模型做单卡/TP 对比。"""

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
from qwen3_layer_reference import prefill

BACKEND = "qwen3_process_prefill_cublas_nccl"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_logits(actual, expected, metrics, label, atol, rtol):
    require(len(actual) == len(expected) and len(actual) > 0, label + ": logits shape mismatch")
    failed, first = 0, None
    for index, (a, b) in enumerate(zip(actual, expected)):
        finite = math.isfinite(a) and math.isfinite(b)
        error = abs(a - b) if finite else float("inf")
        mismatch = not finite or error > atol + rtol * abs(b)
        metrics["max"] = max(metrics["max"], error)
        metrics["sum"] += error
        metrics["count"] += 1
        metrics["failed"] += int(mismatch)
        failed += int(mismatch)
        if mismatch and first is None:
            first = dict(index=index, actual=a, expected=b)
    require(failed == 0, f"{label}: {failed} mismatches, first={first}")


def make_model(directory, spec):
    config, tensors = fixture(directory, dtype=spec.get("dtype", "F16"), bias=spec.get("bias", False),
        tied=spec.get("tied", False), hidden_size=spec.get("hidden", 16), head_dim=spec.get("head_dim", 4),
        num_hidden_layers=spec.get("layers", 3), num_attention_heads=6,
        num_key_value_heads=spec.get("kv", 3), intermediate_size=19, vocab_size=17)
    config.update(rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="silu")
    config.update(spec.get("config", {}))
    for name, tensor in tensors.items():
        values = decode(tensor["data"], tensor["dtype"])
        values = [1 + value * 0.01 if "norm.weight" in name else value * 0.02 for value in values]
        tensor["data"] = encode(values, tensor["dtype"])
    save_fixture(directory, config, tensors)
    return config, tensors


def make_tasks(config, spec):
    vocab = config["vocab_size"]
    count = spec.get("tokens", 8)
    ids = [((i * 37 + 1) % vocab) for i in range(count)]
    if spec.get("bad_id"):
        ids[-1] = vocab
    position = spec.get("position", 0)
    task = dict(name="prompt", token_ids=ids, position_begin=position)
    return [task, dict(name="single-token", token_ids=[vocab - 1], position_begin=position),
            dict(task, name="repeat")]


def prepare_reload(directory, config, tensors, tasks):
    # 改变最终 Norm，保证重载后 logits 必须变化；词表、请求和模型结构保持相同。
    updated = copy.deepcopy(tensors)
    tensor = updated["model.norm.weight"]
    tensor["data"] = encode([-value for value in decode(tensor["data"], tensor["dtype"])], tensor["dtype"])
    reload_dir = directory / "reloaded-model"
    reload_dir.mkdir()
    save_fixture(reload_dir, config, updated)
    tasks[:] = [tasks[0], dict(tasks[0], name="after-reload")]
    return reload_dir, updated


def invoke(binary, worker, model, directory, devices, label, scenario="forward", reload_model=None):
    command = [str(binary), "--worker", str(worker), "--model", str(model),
        "--tasks", str(directory / "tasks.json"), "--devices", ",".join(map(str, devices)), "--case", scenario]
    if reload_model is not None:
        command += ["--reload-model", str(reload_model)]
    (directory / f"{label}.command.txt").write_text(shlex.join(command) + "\n")
    with (directory / f"{label}.stdout.log").open("wb") as stdout, (directory / f"{label}.stderr.log").open("wb") as stderr:
        try:
            with case_process(command, stdout=stdout, stderr=stderr) as process:
                code = wait_case(process, 1500)
        except BaseException as error:
            stderr.write(f"\n{type(error).__name__}: {error}\n".encode())
            raise
    return code, parse_report((directory / f"{label}.stdout.log").read_text())


def verify(report, devices, tasks, vocab):
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, str(report))
    workers = report["workers"]
    require(len(workers) == len(devices) and len({w["pid"] for w in workers}) == len(devices), "worker count/PIDs")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank] and worker["pid"] > 0, "worker identity")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing CUDA/NCCL version")
    require(len(report["results"]) == len(tasks), "missing prefill results")
    for task, result in zip(tasks, report["results"]):
        require(result["task"] == task["name"], "wrong task identity")
        logits = result["logits"]
        require(len(logits) == vocab and all(math.isfinite(value) for value in logits), "invalid/nonfinite vocabulary logits")
        require(result["argmax"] == max(range(vocab), key=logits.__getitem__), "incorrect argmax report")


def verify_failure(code, report, spec):
    require(code == 1 and report.get("status") == "FAIL" and report.get("backend") == BACKEND, str(report))
    require(spec["reason"] in report.get("error", "") and report.get("group_failed") is spec["group_failed"], str(report))
    require(report.get("shutdown_complete") is True and "cleanup_error" not in report, "prefill cleanup failed")
    if spec["group_failed"]:
        require(report.get("workers_reaped") is True, "failed group left workers")


def cases():
    return [(dtype, dict(dtype=dtype)) for dtype in ("F16", "BF16", "F32")] + [
        ("bias", dict(bias=True)), ("tied_embedding", dict(tied=True)),
        ("kv_replication", dict(kv=1)), ("one_layer", dict(layers=1)), ("four_layers", dict(layers=4)),
        ("wide_norm", dict(hidden=257)), ("head_dim_128", dict(head_dim=128)),
        ("max_tokens", dict(tokens=128)), ("position_offset", dict(position=123)),
        ("max_position", dict(position=1048576)),
        ("reload", dict(scenario="reload")),
        ("reject_empty", dict(tokens=0, reason="invalid Qwen3 prefill request", group_failed=False)),
        ("reject_tokens", dict(tokens=129, reason="invalid Qwen3 prefill request", group_failed=False)),
        ("reject_position", dict(position=1048577, reason="invalid Qwen3 prefill request", group_failed=False)),
        ("reject_id", dict(bad_id=True, reason="token ID outside vocabulary", group_failed=True)),
        ("reject_activation", dict(config=dict(hidden_act="gelu"), reason="unsupported Qwen3 forward configuration", group_failed=True)),
        ("reject_quantized", dict(config=dict(quantization_config={"quant_method": "unsupported"}),
                                  reason="quantized checkpoints", group_failed=True)),
        ("reject_unloaded", dict(scenario="unloaded", reason="weights not loaded", group_failed=True))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--build-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="可选真实模型目录；仅验证单卡/双卡一致性，无独立 HF 基准")
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(part) for part in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error("requires exactly two distinct logical GPUs")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_prefill_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    selected = [("real_model", {})] if args.model else cases()
    atol, rtol = (0.05, 0.01) if args.model else (0.008, 0.008)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices, atol=atol, rtol=rtol,
        model=str(args.model.resolve()) if args.model else "generated", kv_cache="none", prefix_cache=False,
        execution="eager", output_tokens=0, sampling="none", eos="no stopping", logits="last input token, full vocabulary",
        reference_kind="single GPU only" if args.model else "independent CPU math and single GPU",
        expected_case_count=len(selected), report_title="Qwen3 full prefill correctness",
        report_description="完整模型末 token logits 数值对比；无 KV、采样或性能计时。argmax 仅记录，不作为通过条件；真实模型模式不是 HF/原线程后端一致性验收。")
    rows = []
    try:
        metadata["build_cache"] = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in metadata["build_cache"].splitlines()
            if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        for name, spec in selected:
            directory = output / name
            directory.mkdir()
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            row = dict(case=name, backend=BACKEND, status="FAIL", reason="", input=spec,
                       max_abs_error=None, mean_abs_error=None, failed_elements=None)
            try:
                model = args.model.resolve() if args.model else directory / "model"
                config, tensors = (json.loads((model / "config.json").read_text()), None) if args.model else make_model(model, spec)
                tasks = make_tasks(config, spec)
                reload_model, updated = None, None
                if spec.get("scenario") == "reload":
                    reload_model, updated = prepare_reload(directory, config, tensors, tasks)
                (directory / "tasks.json").write_text(json.dumps(tasks, allow_nan=False))
                row.update(tasks=tasks, shape=[len(tasks[0]["token_ids"]), config["hidden_size"]],
                           layers=config["num_hidden_layers"], vocabulary=config["vocab_size"])
                code, tp = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices, "tp",
                                  spec.get("scenario", "forward"), reload_model)
                row["workers"] = tp.get("workers", [])
                if "reason" in spec:
                    verify_failure(code, tp, spec)
                else:
                    require(code == 0, str(tp))
                    verify(tp, devices, tasks, config["vocab_size"])
                    code, single = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices[:1], "single",
                                          spec.get("scenario", "forward"), reload_model)
                    require(code == 0, str(single))
                    verify(single, devices[:1], tasks, config["vocab_size"])
                    refs = None
                    if tensors is not None:
                        refs = [prefill(config, updated if updated is not None and i == 1 else tensors,
                                        task["token_ids"], task["position_begin"]) for i, task in enumerate(tasks)]
                        (directory / "reference.json").write_text(json.dumps(refs, allow_nan=False))
                    row["argmax"] = []
                    for i, task in enumerate(tasks):
                        actual, baseline = tp["results"][i]["logits"], single["results"][i]["logits"]
                        compare_logits(actual, baseline, metrics, f"TP vs single {task['name']}", atol, rtol)
                        if refs is not None:
                            compare_logits(actual, refs[i], metrics, f"TP vs CPU {task['name']}", atol, rtol)
                            compare_logits(baseline, refs[i], metrics, f"single vs CPU {task['name']}", atol, rtol)
                        row["argmax"].append(dict(task=task["name"], tp=tp["results"][i]["argmax"], single=single["results"][i]["argmax"]))
                    for label, report in (("TP", tp), ("single", single)):
                        if updated is None:
                            compare_logits(report["results"][2]["logits"], report["results"][0]["logits"],
                                           metrics, label + " repeat", 0, 0)
                        else:
                            require(report["results"][0]["logits"] != report["results"][1]["logits"], label + " stale logits after reload")
                row["status"] = "PASS"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            finally:
                if metrics["count"]:
                    row.update(max_abs_error=metrics["max"] if math.isfinite(metrics["max"]) else "inf",
                        mean_abs_error=metrics["sum"] / metrics["count"] if math.isfinite(metrics["sum"]) else "inf",
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
