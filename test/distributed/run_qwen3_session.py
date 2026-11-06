#!/usr/bin/env python3
"""服务器常驻会话验收：同一正式入口多次生成、恢复拒绝请求并验证实际 logits。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import threading
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen3_prefill import compare_logits, make_model, require, report_error
from run_qwen3_generation import candidates, replay_tasks
from run_qwen3_decode import invoke, verify as verify_decode
from qwen3_layer_reference import prefill
from qwen3_process_session import BACKEND, GenerationCancelled, GenerationRejected, Qwen3ProcessSession


def verify_output(report, request, vocab):
    ids, steps = report["token_ids"], report["steps"]
    require(report["request"] == request, "request mismatch")
    require(1 <= len(ids) <= request["max_new_tokens"] and len(ids) == len(steps), "invalid output count")
    eos = request["eos_token_ids"]
    for index, (token, step) in enumerate(zip(ids, steps)):
        require(type(token) is int and 0 <= token < vocab and type(step["token_id"]) is int and
                step["token_id"] == token, f"invalid token at step {index}")
        require(len(step["logits"]) == vocab and token in candidates(step["logits"], request),
                f"incorrect sampled token at step {index}")
        require(index == len(ids)-1 or token not in eos, "generation continued after EOS")
    finish = "eos" if ids[-1] in eos else "length"
    require(report["finish_reason"] == finish and (finish == "eos" or len(ids) == request["max_new_tokens"]),
            "incorrect finish reason/count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generate", "worker", "decode-check", "build-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 2 or min(devices) < 0 or len(set(devices)) != 2:
        parser.error("requires two distinct GPUs")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_session_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    specs = [("real", {})] if args.model else [(dtype, dict(dtype=dtype)) for dtype in ("F16", "BF16", "F32")]
    if not args.model:
        specs += [("bias", dict(bias=True)), ("tied", dict(tied=True)), ("kv_replication", dict(kv=1))]
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=123, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices,
        expected_case_count=len(specs)*12, execution="eager", prefix_cache=False, kv_cache="FP16 request-local",
        report_title="Qwen3 persistent session correctness",
        report_description="同一常驻进程组串行处理请求；小模型对照CPU，真实模型逐步单卡重放。非性能测试。")
    rows = []
    try:
        metadata["build_cache"] = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in metadata["build_cache"].splitlines()
            if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        for name, spec in specs:
            directory = output / name
            directory.mkdir()
            model = args.model.resolve() if args.model else directory / "model"
            config, tensors = (json.loads((model / "config.json").read_text()), None) if args.model else make_model(model, spec)
            vocab = config["vocab_size"]
            base = dict(token_ids=[1, 4, 7], max_new_tokens=3, temperature=0, top_k=0, top_p=1, seed=123, eos_token_ids=[])
            tasks = [("A", base), ("B", dict(base, token_ids=[2, 3])), ("A_repeat", base),
                     ("reject_length", dict(base, max_new_tokens=0)), ("reject_empty", dict(base, token_ids=[])),
                     ("A_after_reject", base),
                     ("eos_first", None), ("seeded", dict(base, temperature=0.8, top_k=5, top_p=0.9)),
                     ("seeded_repeat", dict(base, temperature=0.8, top_k=5, top_p=0.9)),
                     ("cancel_before_begin", base), ("A_after_cancel", base)]
            previous = {}
            with Qwen3ProcessSession(args.generate, args.worker, model, args.devices, directory / "session", trace=True) as session:
                for task_name, request in tasks:
                    path = directory / task_name
                    path.mkdir()
                    metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
                    refs = []
                    row = dict(case=f"{name}/{task_name}", backend=BACKEND, status="FAIL", reason="",
                               dtype=spec.get("dtype", "model"), layers=config["num_hidden_layers"],
                               hidden=config["hidden_size"], vocabulary=vocab, workers=session.ready["workers"])
                    try:
                        if task_name == "eos_first":
                            require("A" in previous, "first request failed; cannot determine EOS")
                            request = dict(base, eos_token_ids=[previous["A"]["token_ids"][0]])
                        row.update(prompt_tokens=len(request["token_ids"]), max_new_tokens=request["max_new_tokens"])
                        if task_name == "cancel_before_begin":
                            event = threading.Event()
                            event.set()
                            try:
                                session.generate(request, cancel_event=event)
                            except GenerationCancelled as error:
                                require(not error.report.get("steps"), "pre-cancelled request executed tokens")
                                row.update(status="PASS", reason="cancelled before prefill; KV empty and workers reusable")
                            else:
                                raise AssertionError("pre-cancelled request was executed")
                        elif task_name in ("reject_length", "reject_empty"):
                            try:
                                session.generate(request)
                            except GenerationRejected as error:
                                require("generation length" in str(error), "wrong rejection: " + str(error))
                                row.update(status="PASS", reason="invalid length rejected; session reusable")
                            else:
                                raise AssertionError("invalid request accepted")
                        else:
                            report = session.generate(request)
                            verify_output(report, request, vocab)
                            if task_name == "eos_first":
                                require(len(report["token_ids"]) == 1 and report["finish_reason"] == "eos",
                                        "first EOS did not stop generation")
                            if task_name in ("A_repeat", "A_after_reject", "A_after_cancel", "seeded_repeat"):
                                original = "seeded" if task_name == "seeded_repeat" else "A"
                                require(original in previous and report["token_ids"] == previous[original]["token_ids"],
                                        "request state/seed changed across session reuse")
                            previous[task_name] = report
                            replay = None
                            if tensors is None:
                                tasks_file = replay_tasks(request["token_ids"], report["token_ids"])
                                (path / "tasks.json").write_text(json.dumps(tasks_file))
                                code, replay = invoke(args.decode_check.resolve(), args.worker.resolve(), model, path, devices[:1], "single_replay")
                                require(code == 0, report_error(replay))
                                verify_decode(replay, devices[:1], tasks_file, config)
                            prefix = request["token_ids"][:]
                            for index, step in enumerate(report["steps"]):
                                reference = prefill(config, tensors, prefix) if tensors is not None else replay["results"][index]["logits"]
                                refs.append(dict(token_ids=prefix[:], logits=reference))
                                atol, rtol = (0.008, 0.008) if tensors is not None else (0.05, 0.01)
                                compare_logits(step["logits"], reference, metrics, f"{task_name}/step-{index}", atol, rtol)
                                prefix.append(report["token_ids"][index])
                            row.update(status="PASS", generated_tokens=len(report["token_ids"]), finish_reason=report["finish_reason"])
                    except Exception as error:
                        row["reason"] = f"{type(error).__name__}: {error}"
                    finally:
                        (path / "reference.json").write_text(json.dumps(refs, allow_nan=False))
                        if metrics["count"]:
                            row.update(max_abs_error=metrics["max"] if math.isfinite(metrics["max"]) else "inf",
                                mean_abs_error=metrics["sum"]/metrics["count"] if math.isfinite(metrics["sum"]) else "inf",
                                failed_elements=metrics["failed"])
                        rows.append(row)
                        save_report(output, rows, metadata)
            closed = session.state == "closed" and session.shutdown is not None and session.shutdown.get("shutdown_complete") is True
            rows.append(dict(case=f"{name}/shutdown", backend=BACKEND, status="PASS" if closed else "FAIL",
                             reason="worker group closed; KV released" if closed else f"session shutdown incomplete: state={session.state}"))
            save_report(output, rows, metadata)
    except BaseException as error:
        rows.append(dict(case="runner", backend="python", status="FAIL", reason=f"{type(error).__name__}: {error}"))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        status = save_report(output, rows, metadata)
    print(f"{status}: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
