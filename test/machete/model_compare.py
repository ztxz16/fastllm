#!/usr/bin/env python3
"""原生/Machete模型矩阵，复用正式Forward、HTTP性能和Decode Nsight流程。"""
import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from report import REPO, ChartSpec, metadata, save, timing, write_dashboard
from common.performance_report import model_performance_chart_specs


def confirmed(text, backend):
    actual = set(re.findall(r"\[linear-backend\].*?actual=(\S+)", text))
    if actual != {backend}:
        raise RuntimeError(f"Linear路径未严格确认：requested={backend}, actual={sorted(actual)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtypes", default="int4g128,int8")
    parser.add_argument("--atypes", default="float16,bfloat16")
    parser.add_argument("--stages", default="forward,performance,nsight")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--modes", default="eager,best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-cache-layout", default="auto", choices=("auto", "continuous", "paged"))
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--topk-logprob-tolerance", type=float, default=0.2)
    args = parser.parse_args()
    if args.repeats < 5 or args.warmup < 1 or args.output_tokens < 2 or args.input_tokens < 1:
        parser.error("要求repeats>=5、warmup>=1、input-tokens>=1、output-tokens>=2")
    if any(x not in ("forward", "performance", "nsight") for x in args.stages.split(",")):
        parser.error("未知stage")
    if any(x not in ("float16", "bfloat16") for x in args.atypes.split(",")):
        parser.error("atypes为float16,bfloat16")
    if any(x not in ("int4g64", "int4g128", "int4", "int8") for x in args.dtypes.split(",")):
        parser.error("dtypes支持int4、int4g64、int4g128、int8；不接受NVFP4/FP8/W8A8")
    if any(x not in ("eager", "best") for x in args.modes.split(",")):
        parser.error("modes为eager,best")
    import model_performance_compare as performance
    import forward_check_vllm as forward
    root = Path(args.result_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    meta = metadata(shlex.join(sys.argv))
    meta.update(model=args.model, repeats=args.repeats, warmup=args.warmup,
                input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                accuracy_scope="first generated token and common first-step TopK logprobs; not full logits")
    rows, images = [], []
    original_env = {key: os.environ.get(key) for key in ("FASTLLM_LINEAR_BACKEND", "FASTLLM_LINEAR_BACKEND_TRACE")}

    def persist():
        return save(root, rows, meta, images)

    def run(command, path, env):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write("COMMAND: " + shlex.join(command) + "\n")
            handle.flush()
            status = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        if status:
            raise RuntimeError(f"exit={status}, log={path}")
        return path.read_text(errors="replace")

    try:
        for dtype in args.dtypes.split(","):
            for atype in args.atypes.split(","):
                pair = {}
                for backend in ("native", "machete"):
                    name = f"{dtype}-{atype}-{backend}"
                    directory = root / name
                    directory.mkdir()
                    # 原生缺失的精确BF16组合不能改成FP16后假装是公平对照。
                    if backend == "native" and atype == "bfloat16":
                        rows.append(dict(case=name, status="UNSUPPORTED", backend=backend,
                                         reason="native BF16 dispatch lacks these INT8/INT4_GROUP formats"))
                        persist()
                        continue
                    os.environ["FASTLLM_LINEAR_BACKEND"] = backend
                    os.environ["FASTLLM_LINEAR_BACKEND_TRACE"] = "1"
                    env = os.environ.copy()
                    common = ["--model", args.model, "--flm-dtype", dtype, "--flm-atype", atype,
                              "--flm-device", args.device, "--flm-kv-cache-layout", args.kv_cache_layout,
                              "--flm-attention-backend", args.attention_backend]
                    for stage in args.stages.split(","):
                        try:
                            if stage == "forward":
                                command = [args.python, str(REPO / "test/nvfp4/forward_check_vllm.py"),
                                           *common, "--stage", "fastllm", "--tokens", "8", "--top-logprobs", "10",
                                           "--output", str(directory / "forward.json")]
                                text = run(command, directory / "forward.log", env)
                                confirmed(text, backend)
                                result = json.loads((directory / "forward.json").read_text())
                                production, bounded = forward.production_trace(text)
                                if not bounded:
                                    raise RuntimeError("缺少正式请求日志边界")
                                traces = forward.parse_sampling_topk_traces(production)
                                if not traces:
                                    raise RuntimeError("缺少正式请求的TopK输出")
                                result["topk"] = traces[0]["top_logprobs"]
                                pair[backend] = result
                                rows.append(dict(case=name + "-forward", backend=backend, dtype=dtype, atype=atype,
                                                 status="UNPAIRED", stage=stage, generated_token_ids=result["generated_token_ids"],
                                                 reason="awaiting same-configuration native reference"))
                            elif stage == "performance":
                                config = SimpleNamespace(
                                    model=args.model, fastllm_python=args.python, quantization="a16",
                                    flm_dtype=dtype, flm_atype=atype, flm_device=args.device,
                                    flm_kv_cache_layout=args.kv_cache_layout, flm_attention_backend=args.attention_backend,
                                    flm_attention_backend_strict=args.attention_backend != "auto",
                                    batch_sizes=args.batch_sizes, input_tokens=args.input_tokens,
                                    output_tokens=args.output_tokens, repeats=args.repeats, warmup=args.warmup,
                                    port=args.port, startup_timeout=1200, request_timeout=3600)
                                # 同一dtype/atype的两个后端读取同一份Prompt Token文件。
                                prompt_dir = root / f"{dtype}-{atype}-prompts"
                                prompt_dir.mkdir(exist_ok=True)
                                prompt_path = prompt_dir / "shared-prompt-token-ids.json"
                                if not prompt_path.exists():
                                    performance.build_shared_prompts(config, prompt_dir)
                                prompts = json.loads(prompt_path.read_text())
                                for mode in args.modes.split(","):
                                    with (directory / f"performance-{mode}.log").open("w") as log:
                                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                            results = performance.run_fastllm(config, prompts, directory, mode)
                                    confirmed((directory / f"fastllm-{mode}-server.log").read_text(errors="replace"), backend)
                                    for result in results:
                                        result.update(case=name + "-performance-" + mode, backend=backend,
                                                      dtype=dtype, atype=atype, stage=stage, status="PASS")
                                        for metric in ("ttft", "tpot", "itl", "e2el"):
                                            result[metric + "_ms"] = result[metric + "_s"] * 1000
                                            samples = [trial[metric + "_s"] * 1000 for trial in result["trials"]]
                                            stats = timing(samples)
                                            result[metric + "_p95_ms"] = stats["p95_ms"]
                                            result[metric + "_cv"] = stats["cv"]
                                        rows.append(result)
                            else:
                                command = [args.python, str(REPO / "test/nvfp4/decode_nsys_compare.py"), *common,
                                           "--quantization", "a16", "--backends", "fastllm", "--cpu-trace",
                                           "--batch-sizes", args.batch_sizes, "--prompt-tokens", str(args.input_tokens),
                                           "--output-tokens", str(args.output_tokens), "--max-model-len", str(args.max_model_len),
                                           "--fastllm-python", args.python, "--nsys", args.nsys,
                                           "--result-dir", str(directory / "nsight")]
                                run(command, directory / "nsight.log", env)
                                logs = "\n".join(p.read_text(errors="replace") for p in (directory / "nsight").rglob("*.log"))
                                confirmed(logs, backend)
                                results = json.loads((directory / "nsight/decode-nsys-compare.json").read_text())
                                for result in results:
                                    ns = result["nsys"]
                                    result.update(case=name + "-nsight", backend=backend, dtype=dtype, atype=atype,
                                                  stage=stage, status="PASS", kernel_ms=ns["kernel_ms_per_decode_step"],
                                                  kernel_count=ns["kernel_instances_per_decode_step"],
                                                  idle_pct=ns["timeline"]["idle_ratio"] * 100,
                                                  submit_ms=(ns.get("cpu_trace") or {}).get("launch_api_union_ms_per_decode_step"))
                                    rows.append(result)
                        except Exception as exc:
                            rows.append(dict(case=name + "-" + stage, backend=backend, stage=stage, status="FAIL", failure=str(exc)))
                        persist()
                if "native" in pair and "machete" in pair:
                    a, b = pair["native"], pair["machete"]
                    common_tokens = set(a["topk"]) & set(b["topk"])
                    errors = [abs(a["topk"][token] - b["topk"][token]) for token in common_tokens]
                    failed = sum(e > args.topk_logprob_tolerance for e in errors)
                    same_prompt = a["prompt_token_ids"] == b["prompt_token_ids"]
                    first_match = bool(a["generated_token_ids"] and b["generated_token_ids"]) and a["generated_token_ids"][0] == b["generated_token_ids"][0]
                    passed = same_prompt and first_match and len(common_tokens) >= 8 and not failed
                    for row in rows:
                        if row.get("stage") == "forward" and row.get("dtype") == dtype and row.get("atype") == atype:
                            row.update(status="PASS" if passed else "FAIL", prompt_match=same_prompt, first_token_match=first_match,
                                       common_topk_max_abs_error=max(errors) if errors else None,
                                       common_topk_mean_abs_error=sum(errors) / len(errors) if errors else None,
                                       failed_elements=failed, common_topk_count=len(common_tokens),
                                       generated_tokens_match=a["generated_token_ids"] == b["generated_token_ids"], reason="")
                for mode in args.modes.split(","):
                    for scenario in ("cold", "cache_hit"):
                        selected = [r for r in rows if (r.get("dtype"), r.get("atype"), r.get("mode"), r.get("scenario"), r.get("stage")) ==
                                    (dtype, atype, mode, scenario, "performance") and r["status"] == "PASS"]
                        for row in selected:
                            peer = next((p for p in selected if p["backend"] == "native" and p["batch"] == row["batch"]), None)
                            if peer and row["backend"] == "machete":
                                row["machete_tpot_speedup_x"] = peer["tpot_s"] / row["tpot_s"]
                        if selected:
                            for workload in ("prefill", "decode"):
                                images.append(write_dashboard(root / "images" / f"{dtype}-{atype}-{mode}-{scenario}-{workload}.png",
                                    selected, model_performance_chart_specs(workload), f"{dtype} {atype} {mode} {scenario} {workload}".upper()))
                selected = [r for r in rows if r.get("stage") == "nsight" and r.get("dtype") == dtype and r.get("atype") == atype and r["status"] == "PASS"]
                if selected:
                    images.append(write_dashboard(root / "images" / f"{dtype}-{atype}-nsight.png", selected,
                        [ChartSpec("KERNEL TIME", "kernel_ms", "ms/step"), ChartSpec("KERNEL COUNT", "kernel_count", "count/step"),
                         ChartSpec("GPU IDLE", "idle_pct", "%"), ChartSpec("SUBMIT API", "submit_ms", "ms/step")],
                        f"{dtype} {atype} NSIGHT".upper()))
                persist()
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        status = persist()
    print(f"Summary: {status}\nReport: {root / 'summary.xlsx'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
