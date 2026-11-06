#!/usr/bin/env python3
"""运行正式生成入口，重放相同 ID 验证数值；只调用已有产物，不编译。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

from run_process_tp import ROOT, capture, case_process, parse_report, save_report, wait_case
from run_qwen3_prefill import compare_logits, invoke, make_model, report_error, require
from run_qwen3_decode import verify as verify_decode
from qwen3_layer_reference import prefill

BACKEND = "qwen3_process_generate_cublas_nccl"


def candidates(logits, request):
    """独立计算允许的采样集合；具体概率边界另由原生采样黄金用例验证。"""
    require(logits and all(math.isfinite(x) for x in logits), "invalid generation logits")
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    if request["temperature"] == 0:
        return order[:1]
    if request["top_k"]:
        order = order[:request["top_k"]]
    weights = [math.exp((logits[i] - logits[order[0]]) / request["temperature"]) for i in order]
    cutoff, acc, kept = request["top_p"] * sum(weights), 0, []
    for i, weight in zip(order, weights):
        if weight > 0:
            kept.append(i)
        acc += weight
        if request["top_p"] < 1 and acc >= cutoff:
            break
    return kept


def verify_generation(report, request, devices, vocab):
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, report_error(report))
    require(report.get("request") == request and report.get("shutdown_complete") is True, "request/shutdown mismatch")
    workers = report["workers"]
    require(len(workers) == len(devices) and len({w["pid"] for w in workers}) == len(devices), "worker count/PIDs")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank] and worker["pid"] > 0, "worker identity")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing runtime metadata")
    for key in ("cache_after_generate", "cache_after_unload"):
        require(report[key] == [dict(active=False, tokens=0, capacity=0, device_bytes=0) for _ in devices], "KV not released")
    ids, steps = report["token_ids"], report["steps"]
    require(1 <= len(ids) <= request["max_new_tokens"] and len(steps) == len(ids), "generation count mismatch")
    eos = request["eos_token_ids"]
    for index, (token, step) in enumerate(zip(ids, steps)):
        step_token = step.get("token_id")
        require(type(token) is int and 0 <= token < vocab and type(step_token) is int and step_token == token,
                f"invalid generated ID at step={index}: token={token!r} ({type(token).__name__}), "
                f"step_token={step_token!r} ({type(step_token).__name__}), vocab={vocab}")
        require(len(step["logits"]) == vocab and token in candidates(step["logits"], request), "incorrect sampled token")
        require(index == len(ids)-1 or token not in eos, "generation continued after EOS")
    expected_reason = "eos" if ids[-1] in eos else "length"
    require(report["finish_reason"] == expected_reason, "wrong finish reason")
    require(expected_reason == "eos" or len(ids) == request["max_new_tokens"], "stopped before length without EOS")


def replay_tasks(prompt, generated):
    """按生成入口真正消费的前缀重放；最后输出不进入 KV，避免多算一步。"""
    tasks = [dict(name="step-0", op="begin", token_ids=prompt,
                  capacity=len(prompt)+len(generated)-1, position_begin=0)]
    tasks += [dict(name=f"step-{i+1}", op="decode", token_id=token) for i, token in enumerate(generated[:-1])]
    tasks.append(dict(name="end", op="end"))
    return tasks


def run_binary(command, directory, label):
    (directory / f"{label}.command.txt").write_text(shlex.join(command) + "\n")
    with (directory / f"{label}.stdout.log").open("wb") as stdout, (directory / f"{label}.stderr.log").open("wb") as stderr:
        with case_process(command, stdout=stdout, stderr=stderr) as process:
            code = wait_case(process, 1500)
    return code, parse_report((directory / f"{label}.stdout.log").read_text())


def generate(args, directory, model, request, label="generate"):
    path = directory / (label + ".request.json")
    path.write_text(json.dumps(request, allow_nan=False))
    return run_binary([str(args.generate.resolve()), "--worker", str(args.worker.resolve()), "--model", str(model),
        "--request", str(path), "--devices", ",".join(map(str, args.devices)), "--trace-logits", "1"], directory, label)


def cases():
    return [(dtype, dict(dtype=dtype)) for dtype in ("F16", "BF16", "F32")] + [
        ("bias", dict(bias=True)), ("tied", dict(tied=True)), ("kv_replication", dict(kv=1)),
        ("limit_one", dict(max_new_tokens=1)), ("max_prompt", dict(tokens=128, max_new_tokens=2)),
        ("eos_first", dict(eos=True)), ("sample_seeded", dict(temperature=0.8, top_k=5, top_p=0.9)),
        ("sample_top_k_one", dict(temperature=0.8, top_k=1)),
        ("reject_zero_length", dict(reject={"max_new_tokens": 0}, error="generation length")),
        ("reject_capacity", dict(reject={"max_new_tokens": 2048}, error="generation length")),
        ("reject_prompt", dict(tokens=129, error="generation length")),
        ("reject_temperature", dict(reject={"temperature": -1}, error="sampling parameters")),
        ("reject_top_p", dict(reject={"top_p": 0}, error="sampling parameters")),
        ("reject_eos", dict(reject={"eos_token_ids": [17]}, error="EOS token outside vocabulary"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generate", "sampling-check", "decode-check", "worker", "build-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    args.devices = [int(x) for x in args.devices.split(",")]
    if len(args.devices) != 2 or min(args.devices) < 0 or len(set(args.devices)) != 2:
        parser.error("requires two distinct logical GPUs")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_generation_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    selected = [("real_greedy", {})] if args.model else cases()
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=123, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=args.devices,
        expected_case_count=len(selected)+1, execution="eager", prefix_cache=False, kv_cache="FP16 request-local",
        report_title="Qwen3 generation correctness", report_description="正式生成入口；相同生成前缀在单卡重放，小模型另对照CPU。非性能测试。")
    rows = []
    try:
        metadata["build_cache"] = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in metadata["build_cache"].splitlines()
            if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        row = dict(case="sampling_golden", backend="coordinator_cpu_sampler", status="FAIL", reason="")
        try:
            code, sampler = run_binary([str(args.sampling_check.resolve())], output, "sampling")
            require(code == 0 and sampler.get("status") == "PASS" and len(sampler.get("cases", [])) == 24,
                    "native sampling golden cases failed; see sampling.stdout.log")
            require(all(item.get("status") == "PASS" and item.get("failed_elements") == 0 for item in sampler["cases"]),
                    "native sampling partial failure")
            row.update(status="PASS", max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
        except Exception as error:
            row["reason"] = str(error)
        rows.append(row)
        save_report(output, rows, metadata)

        for name, spec in selected:
            directory = output / name
            directory.mkdir()
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            refs = []
            row = dict(case=name, backend=BACKEND, status="FAIL", reason="", input=spec)
            try:
                model = args.model.resolve() if args.model else directory / "model"
                config, tensors = (json.loads((model / "config.json").read_text()), None) if args.model else make_model(model, spec)
                vocab = config["vocab_size"]
                prompt = [(i*3+1) % vocab for i in range(spec.get("tokens", 5))]
                request = dict(token_ids=prompt, max_new_tokens=spec.get("max_new_tokens", 4), seed=123,
                    temperature=spec.get("temperature", 0), top_k=spec.get("top_k", 0), top_p=spec.get("top_p", 1), eos_token_ids=[])
                request.update(spec.get("reject", {}))
                row.update(prompt_tokens=len(prompt), max_new_tokens=request["max_new_tokens"],
                           layers=config["num_hidden_layers"], hidden=config["hidden_size"], vocabulary=vocab)
                if spec.get("eos"):
                    code, probe = generate(args, directory, model, dict(request, max_new_tokens=1), "eos_probe")
                    require(code == 0, report_error(probe))
                    verify_generation(probe, dict(request, max_new_tokens=1), args.devices, vocab)
                    first = probe["token_ids"][0]
                    request["eos_token_ids"] = [(first+1) % vocab, first]
                code, report = generate(args, directory, model, request)
                row["workers"] = report.get("workers", [])
                if "error" in spec:
                    require(code == 1 and report.get("status") == "FAIL" and spec["error"] in report.get("error", ""), report_error(report))
                    require(report.get("shutdown_complete") is True and "cleanup_error" not in report, "rejection cleanup failed")
                    require(report.get("cache_after_error") == [dict(active=False, tokens=0, capacity=0, device_bytes=0)
                            for _ in args.devices], "rejected generation left active KV before shutdown")
                else:
                    require(code == 0, report_error(report))
                    verify_generation(report, request, args.devices, vocab)
                    ids = report["token_ids"]
                    row.update(generated_tokens=len(ids), finish_reason=report["finish_reason"])
                    if spec.get("eos"):
                        require(len(ids) == 1 and report["finish_reason"] == "eos", "first EOS did not stop")
                    if request["temperature"] > 0:
                        code, repeated = generate(args, directory, model, request, "same_seed")
                        require(code == 0, report_error(repeated))
                        verify_generation(repeated, request, args.devices, vocab)
                        require(repeated["token_ids"] == ids, "same seed generated different IDs")
                    tasks = replay_tasks(prompt, ids)
                    (directory / "tasks.json").write_text(json.dumps(tasks))
                    code, replay = invoke(args.decode_check.resolve(), args.worker.resolve(), model, directory, args.devices[:1], "single_replay")
                    require(code == 0, report_error(replay))
                    verify_decode(replay, args.devices[:1], tasks, config)
                    atol, rtol = (0.05, 0.01) if args.model else (0.008, 0.008)
                    prefix = prompt[:]
                    for index, step in enumerate(report["steps"]):
                        reference = prefill(config, tensors, prefix) if tensors is not None else None
                        if reference is not None:
                            refs.append(dict(token_ids=prefix[:], logits=reference))
                        compare_logits(step["logits"], replay["results"][index]["logits"], metrics, f"step-{index} TP vs single", atol, rtol)
                        if reference is not None:
                            compare_logits(step["logits"], reference, metrics, f"step-{index} TP vs CPU", atol, rtol)
                        prefix.append(ids[index])
                row["status"] = "PASS"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            finally:
                (directory / "reference.json").write_text(json.dumps(refs, allow_nan=False))
                if metrics["count"]:
                    row.update(max_abs_error=metrics["max"] if math.isfinite(metrics["max"]) else "inf",
                        mean_abs_error=metrics["sum"]/metrics["count"] if math.isfinite(metrics["sum"]) else "inf", failed_elements=metrics["failed"])
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
