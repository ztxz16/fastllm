#!/usr/bin/env python3
"""请求 KV/Decode 数值验收；只运行已有产物，不编译。真实模型模式使用固定 ID 比较单卡/TP。"""

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

from run_process_tp import ROOT, capture, save_report
from run_qwen3_prefill import compare_logits, invoke, make_model, report_error, require
from run_qwen3_shards import decode, encode, save_fixture
from qwen3_layer_reference import prefill

BACKEND = "qwen3_process_decode_cublas_nccl"
EMPTY = dict(active=False, tokens=0, capacity=0, position_begin=0, device_bytes=0)


def cases():
    positive = [(dtype, dict(dtype=dtype)) for dtype in ("F16", "BF16", "F32")]
    positive += [("bias", dict(bias=True)), ("tied", dict(tied=True)), ("kv_replication", dict(kv=1)),
        ("head_dim_128", dict(head_dim=128)), ("position", dict(position=123)),
        ("position_limit", dict(position=1048573, tokens=1, steps=3, capacity=4)),
        ("capacity_one", dict(tokens=1, steps=0, capacity=1)),
        ("cross_128", dict(tokens=128, steps=2, capacity=130, cycles=1, layers=1)),
        ("new_prompt", dict(new_prompt=True)), ("reload", dict(reload=True)),
        ("active_unload", dict(active_unload=True)),
        ("max_context", dict(tokens=128, steps=1920, capacity=2048, cycles=1,
                             layers=1, hidden=4, head_dim=2, kv=1, prompt_id=14, decode_id=16,
                             reference_lengths=[128, 129, 256, 2048]))]
    negative = [
        ("no_request", "Qwen3 decode requires active request", True),
        ("after_end", "Qwen3 decode requires active request", True),
        ("after_unload", "weights not loaded", True),
        ("duplicate", "Qwen3 request already active", True),
        ("exhausted", "Qwen3 request capacity exhausted", True),
        ("bad_id", "token ID outside vocabulary", True),
        ("bad_prompt", "token ID outside vocabulary", True),
        ("zero_capacity", "invalid Qwen3 request capacity/position/timeout", False),
        ("large_capacity", "invalid Qwen3 request capacity/position/timeout", False),
        ("small_capacity", "invalid Qwen3 request capacity/position/timeout", False),
        ("position_overflow", "invalid Qwen3 request capacity/position/timeout", False),
        ("empty", "invalid Qwen3 request capacity/position/timeout", False),
        ("too_many_prompt", "invalid Qwen3 request capacity/position/timeout", False),
        ("kv_limit", "Qwen3 request KV exceeds 256 MiB", True),
        ("report_limit", "decode report logit limit exceeded", False)]
    return positive + [("reject_" + name, dict(reject=name, reason=reason, group_failed=failed,
        **(dict(layers=129, head_dim=128) if name == "kv_limit" else
           dict(max_logit_elements=17) if name == "report_limit" else {}))) for name, reason, failed in negative]


def make_tasks(config, spec):
    vocab = config["vocab_size"]
    count, steps = spec.get("tokens", 5), spec.get("steps", 3)
    ids = [(i * 37 + 1) % vocab for i in range(count)]
    if "prompt_id" in spec:
        ids = [spec["prompt_id"] % vocab] * count
    capacity, position = spec.get("capacity", count + steps), spec.get("position", 0)
    begin = dict(op="begin", token_ids=ids, capacity=capacity, position_begin=position)
    reject = spec.get("reject")
    if reject:
        actions = [copy.deepcopy(begin)]
        if reject == "no_request":
            actions = [dict(op="decode", token_id=1)]
        elif reject in ("after_end", "after_unload"):
            actions += [dict(op="end" if reject == "after_end" else "unload"), dict(op="decode", token_id=1)]
        elif reject == "duplicate":
            actions += [copy.deepcopy(begin)]
        elif reject == "exhausted":
            actions[0]["capacity"] = len(ids)
            actions += [dict(op="decode", token_id=1)]
        elif reject == "bad_id":
            actions += [dict(op="decode", token_id=vocab)]
        elif reject == "report_limit":
            actions += [dict(op="decode", token_id=1)]
        elif reject == "bad_prompt":
            actions[0]["token_ids"] = [vocab]
        elif reject in ("zero_capacity", "large_capacity", "small_capacity", "kv_limit"):
            actions[0]["capacity"] = dict(zero_capacity=0, large_capacity=2049,
                                         small_capacity=len(ids)-1, kv_limit=2048)[reject]
        elif reject == "position_overflow":
            actions[0]["position_begin"] = 1048576
        elif reject == "empty":
            actions[0]["token_ids"] = []
        elif reject == "too_many_prompt":
            actions[0].update(token_ids=[1] * 129, capacity=256)
        return [dict(action, name=f"reject-{i}-{action['op']}") for i, action in enumerate(actions)]

    tasks = [dict(name="initial", op="query")]
    for cycle in range(spec.get("cycles", 2)):
        prompt = [(value + 3) % vocab for value in ids] if cycle and spec.get("new_prompt") else ids[:]
        prefix = prompt[:]
        for step in range(steps + 1):
            name = f"cycle-{cycle}-step-{step}"
            action = dict(begin, token_ids=prompt[:]) if step == 0 else dict(
                op="decode", token_id=spec.get("decode_id", step * 13 + 2) % vocab)
            if step:
                prefix.append(action["token_id"])
            if cycle and not spec.get("new_prompt") and not spec.get("reload"):
                action["repeat_of"] = f"cycle-0-step-{step}"
            tasks.append(dict(action, name=name))
            if len(prefix) <= 128:
                tasks.append(dict(name=name + "-full", op="prefill", token_ids=prefix[:],
                                  position_begin=position, compare_to=name))
        if spec.get("active_unload") or spec.get("reload"):
            tasks += [dict(name=f"unload-{cycle}", op="unload"),
                      dict(name=f"load-{cycle}", op="load", reload=bool(spec.get("reload")))]
        else:
            tasks += [dict(name=f"end-{cycle}", op="end"), dict(name=f"end-again-{cycle}", op="end")]
    return tasks


def expectations(tasks):
    """根据请求操作推导缓存有效范围和每步完整前缀，供独立重算使用。"""
    prefix, capacity, position, model = [], 0, 0, "original"
    expected = []
    for task in tasks:
        op = task["op"]
        if op == "begin":
            require(not prefix, "test plan starts an already active request")
            prefix, capacity, position = task["token_ids"][:], task["capacity"], task["position_begin"]
        elif op == "decode":
            require(prefix and len(prefix) < capacity, "test plan decodes without capacity")
            prefix.append(task["token_id"])
        elif op in ("end", "unload"):
            prefix, capacity, position = [], 0, 0
        elif op == "load":
            require(not prefix, "test plan loads with active cache")
            model = "reloaded" if task.get("reload") else "original"
        item = dict(active=bool(prefix), tokens=len(prefix), capacity=capacity, position_begin=position, model=model)
        if op in ("begin", "decode", "prefill"):
            item["ids"] = task["token_ids"][:] if op == "prefill" else prefix[:]
            item["rope_position"] = task["position_begin"] if op == "prefill" else position
        expected.append(item)
    return expected


def expected_cache(config, state, rank, world):
    if not state["active"]:
        return EMPTY.copy()
    heads = config["num_key_value_heads"]
    local_heads = 1 if world > heads else heads // world + int(rank < heads % world)
    dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    size = config["num_hidden_layers"] * state["capacity"] * 2 * local_heads * dim * 2
    return dict(active=True, tokens=state["tokens"], capacity=state["capacity"],
                position_begin=state["position_begin"], device_bytes=size)


def verify(report, devices, tasks, config):
    require(report.get("status") == "PASS" and report.get("backend") == BACKEND, report_error(report))
    workers = report["workers"]
    require(len(workers) == len(devices) and len({w["pid"] for w in workers}) == len(devices), "worker count/PIDs")
    for rank, worker in enumerate(workers):
        require(worker["rank"] == rank and worker["device"] == devices[rank] and worker["pid"] > 0, "worker identity")
        require(worker["cuda_runtime_version"] > 0 and worker["nccl_version"] > 0, "missing CUDA/NCCL version")
    require(len(report["results"]) == len(tasks), "missing decode results")
    for task, state, result in zip(tasks, expectations(tasks), report["results"]):
        label = task["name"]
        require(result["task"] == label and result["op"] == task["op"], label + ": wrong task identity")
        cache = [expected_cache(config, state, rank, len(devices)) for rank in range(len(devices))]
        require(result["cache"] == cache, label + ": incorrect KV length/allocation/lifetime")
        if "ids" in state:
            logits = result["logits"]
            require(len(logits) == config["vocab_size"] and all(math.isfinite(v) for v in logits), label + ": invalid logits")
            require(result["argmax"] == max(range(len(logits)), key=logits.__getitem__), label + ": incorrect argmax")
        else:
            require("logits" not in result, label + ": unexpected logits")
    require(report["after_unload"] == [EMPTY] * len(devices), "unload left KV allocated")


def verify_failure(code, report, spec, tasks):
    require(code == 1 and report.get("status") == "FAIL" and report.get("backend") == BACKEND,
            f"exit={code}; {report_error(report)}")
    require(spec["reason"] in report.get("error", "") and report.get("group_failed") is spec["group_failed"], report_error(report))
    require(report.get("current_task") == tasks[-1]["name"], "failed before intended rejection")
    require(report.get("shutdown_complete") is True and "cleanup_error" not in report, "decode cleanup failed")
    if spec["group_failed"]:
        require(report.get("workers_reaped") is True, "failed group left workers")
    if spec.get("reject") == "report_limit":
        require(report.get("logit_limit") == spec["max_logit_elements"] and
                report.get("recorded_logits") == 17 and len(report.get("results", [])) == 1,
                "logit limit discarded or miscounted completed results")


def compare_results(tp, single, tasks, config, tensors, updated, metrics, atol, rtol, spec, reference_path=None):
    """逐步比较输出；在数值失败或中断时仍保存已经计算的独立 CPU 参考。"""
    expected, refs, outputs = expectations(tasks), {}, {"TP": {}, "single": {}}
    try:
        for index, (task, state) in enumerate(zip(tasks, expected)):
            if "ids" not in state:
                continue
            name = task["name"]
            reference = None
            lengths = spec.get("reference_lengths")
            if tensors is not None and (lengths is None or len(state["ids"]) in lengths):
                key = (state["model"], state["rope_position"], tuple(state["ids"]))
                if key not in refs:
                    source = updated if state["model"] == "reloaded" else tensors
                    refs[key] = prefill(config, source, state["ids"], state["rope_position"])
                reference = refs[key]

            actual, baseline = tp["results"][index]["logits"], single["results"][index]["logits"]
            compare_logits(actual, baseline, metrics, name + " TP vs single", atol, rtol)
            for label, values in (("TP", actual), ("single", baseline)):
                if "compare_to" in task:
                    compare_logits(values, outputs[label][task["compare_to"]], metrics, label + " cached vs full " + name, atol, rtol)
                if "repeat_of" in task:
                    compare_logits(values, outputs[label][task["repeat_of"]], metrics, label + " repeat " + name, 0, 0)
                outputs[label][name] = values
            if reference is not None:
                for label, values in (("TP", actual), ("single", baseline)):
                    compare_logits(values, reference, metrics, label + " vs CPU " + name, atol, rtol)
        if spec.get("reload"):
            for label in outputs:
                require(outputs[label]["cycle-0-step-0"] != outputs[label]["cycle-1-step-0"], label + " stale reload logits")
    finally:
        records = [dict(model=key[0], position=key[1], token_ids=list(key[2]), logits=value) for key, value in refs.items()]
        if reference_path is not None:
            reference_path.write_text(json.dumps(records, allow_nan=False))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--build-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(part) for part in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error("requires exactly two distinct logical GPUs")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_decode_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    selected = [("real_model", {})] if args.model else cases()
    atol, rtol = (0.05, 0.01) if args.model else (0.008, 0.008)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices, atol=atol, rtol=rtol,
        model=str(args.model.resolve()) if args.model else "generated", kv_cache="FP16 request-local contiguous",
        prefix_cache=False, execution="eager", sampling="fixed input IDs; no sampling", eos="no stopping",
        expected_case_count=len(selected), report_title="Qwen3 request KV and decode correctness",
        report_description="逐步 TP/单卡对比，前缀不超过128时对照完整重算；小模型含独立CPU参考。max_context仅在128/129/256/2048检查CPU，其余步仍逐元素比较TP与单卡。非性能或HF端到端验收。")
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
                if spec.get("reload"):
                    updated = copy.deepcopy(tensors)
                    tensor = updated["model.norm.weight"]
                    tensor["data"] = encode([-v for v in decode(tensor["data"], tensor["dtype"])], tensor["dtype"])
                    reload_model = directory / "reloaded-model"
                    reload_model.mkdir()
                    save_fixture(reload_model, config, updated)
                (directory / "tasks.json").write_text(json.dumps(tasks, allow_nan=False))
                row.update(task_count=len(tasks), hidden=config["hidden_size"], layers=config["num_hidden_layers"],
                           vocabulary=config["vocab_size"], prompt_tokens=spec.get("tokens", 5),
                           decode_steps=spec.get("steps", 3), cycles=spec.get("cycles", 2),
                           output_tokens=0, input_tasks="tasks.json")
                extra_args = ["--max-logit-elements", str(spec["max_logit_elements"])] if "max_logit_elements" in spec else []
                code, tp = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices, "tp",
                                  reload_model=reload_model, extra_args=extra_args)
                row["workers"] = tp.get("workers", [])
                if "reason" in spec:
                    verify_failure(code, tp, spec, tasks)
                else:
                    require(code == 0, f"TP exit={code}; {report_error(tp)}")
                    verify(tp, devices, tasks, config)
                    code, single = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices[:1], "single", reload_model=reload_model)
                    require(code == 0, f"single exit={code}; {report_error(single)}")
                    verify(single, devices[:1], tasks, config)
                    compare_results(tp, single, tasks, config, tensors, updated, metrics, atol, rtol, spec,
                                    reference_path=directory / "reference.json")
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
