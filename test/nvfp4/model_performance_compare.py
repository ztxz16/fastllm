#!/usr/bin/env python3
"""通过相同HTTP接口和Token ID对比FastLLM与vLLM整体模型性能。"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import requests

TEST_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_DIR))

from common.performance_report import (  # noqa: E402
    markdown_images, model_performance_chart_specs, slugify, write_dashboard)
from common.attention_backend_report import (  # noqa: E402
    actual_attention_backends, actual_kv_cache_layouts,
    attention_backend_confirmed, kv_cache_layout_confirmed)
from xlsx_report import write_xlsx


REPO_DIR = Path(__file__).resolve().parents[2]
MODES = ("eager", "best")
SCENARIOS = ("cold", "cache_hit")


def parse_args():
    parser = argparse.ArgumentParser(description="NVFP4 vLLM/FastLLM严格整体性能对比")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--vllm-python", default=os.environ.get(
        "NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--flm-kv-cache-layout", default="auto",
                        choices=("auto", "continuous", "paged"))
    parser.add_argument("--flm-attention-backend", default="auto")
    parser.add_argument("--flm-attention-backend-strict", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8,16,32",
        help="完整请求的并发数量矩阵；同一次测量生成Prefill和Decode图")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--quantization", choices=("nvfp4", "w8a8"),
                        default="nvfp4", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name}必须大于0")


def batch_cases(args):
    values = [int(value.strip()) for value in args.batch_sizes.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError("batch-sizes必须是逗号分隔的正整数")
    return list(dict.fromkeys(values))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_prompt_tokens(tokenizer, target_tokens, label):
    """生成指定长度Prompt；label位于开头，确保Cold用例不能命中同一前缀页。"""
    messages = [{
        "role": "user",
        "content": f"{label}。请基于后续上下文继续回答。",
    }]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(base_ids) > target_tokens:
        raise ValueError(f"目标长度{target_tokens}小于基础Prompt长度{len(base_ids)}")
    filler = tokenizer.encode(
        " FastLLM vLLM shared benchmark context block.",
        add_special_tokens=False)
    if not filler:
        raise RuntimeError("上下文填充文本未产生token")
    missing = target_tokens - len(base_ids)
    fill_ids = (filler * ((missing + len(filler) - 1) // len(filler)))[:missing]
    token_ids = base_ids[:-1] + fill_ids + base_ids[-1:] if base_ids else fill_ids
    if len(token_ids) != target_tokens:
        raise AssertionError("固定Prompt Token长度构造失败")
    return token_ids


def case_specs(args):
    return [
        ("request", batch, args.input_tokens, args.output_tokens)
        for batch in batch_cases(args)
    ]


def build_shared_prompts(args, result_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = {}
    for workload, batch, input_length, output_length in case_specs(args):
        case_id = f"{workload}_b{batch}"
        cold_warmup = render_prompt_tokens(
            tokenizer, input_length, f"{case_id} cold warmup")
        cold_trials = [
            render_prompt_tokens(
                tokenizer, input_length, f"{case_id} cold measured trial {index}")
            for index in range(args.repeats)
        ]
        cache_tokens = render_prompt_tokens(
            tokenizer, input_length, f"{case_id} cache hit")
        first_differences = []
        for trial in cold_trials:
            difference = next(
                (index for index, pair in enumerate(zip(cold_warmup, trial))
                 if pair[0] != pair[1]), input_length)
            first_differences.append(difference)
        # 差异必须落在Prompt开头，不能让Cold用例复用完整的Paged KV页。
        if any(index >= 128 for index in first_differences):
            raise RuntimeError(f"{case_id}的Cold Prompt公共前缀过长")
        cases[case_id] = {
            "workload": workload,
            "batch": batch,
            "input_length": input_length,
            "output_length": output_length,
            "cold": {
                "warmup_token_ids": cold_warmup,
                "trial_token_ids": cold_trials,
                "first_difference_from_warmup": first_differences,
            },
            "cache_hit": {
                "warmup_token_ids": cache_tokens,
                "trial_token_ids": [cache_tokens for _ in range(args.repeats)],
            },
        }
    payload = {
        "model": args.model,
        "tokenizer": type(tokenizer).__name__,
        "repeats": args.repeats,
        "cases": cases,
    }
    path = result_dir / "shared-prompt-token-ids.json"
    write_json(path, payload)
    return path


def mean_or_none(values):
    return statistics.mean(values) if values else None


def summarize_requests(backend, mode, scenario, workload, batch, input_tokens,
                       target_output_tokens, requests_data, batch_start, batch_end):
    ttfts = []
    tpots = []
    itls = []
    e2els = []
    for item in requests_data:
        token_times = item["token_times"]
        e2els.append(item["end_time"] - item["start_time"])
        if token_times:
            ttfts.append(token_times[0] - item["start_time"])
        if len(token_times) > 1:
            tpots.append(
                (item["end_time"] - token_times[0]) / (len(token_times) - 1))
            itls.extend(
                token_times[index] - token_times[index - 1]
                for index in range(1, len(token_times)))
    total_output_tokens = sum(item["output_tokens"] for item in requests_data)
    cached_values = [item["cached_input_tokens"] for item in requests_data
                     if item["cached_input_tokens"] is not None]
    total_cached_tokens = sum(cached_values) if cached_values else None
    total_prompt_tokens = len(input_tokens) * batch
    wall = batch_end - batch_start
    ttft_avg = mean_or_none(ttfts)
    # 客户端观测的有效Prompt吞吐，包括排队、缓存恢复和首Token返回。
    # Cache Hit时计入复用Token，不能解释为实际Prefill kernel计算吞吐。
    first_tokens = [item["token_times"][0] for item in requests_data
                    if item["token_times"]]
    prefill_wall = (max(first_tokens) - batch_start
                    if len(first_tokens) == batch else None)
    return {
        "mode": mode,
        "scenario": scenario,
        "workload": workload,
        "backend": backend,
        "batch": batch,
        "prompt_tokens_per_request": len(input_tokens),
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_prompt_tokens": total_cached_tokens,
        "cache_hit_ratio": (
            total_cached_tokens / total_prompt_tokens
            if total_cached_tokens is not None and total_prompt_tokens else None),
        "target_output_tokens_per_request": target_output_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_s": ttft_avg,
        "tpot_s": mean_or_none(tpots),
        "itl_s": mean_or_none(itls),
        "e2el_s": mean_or_none(e2els),
        "wall_s": wall,
        "prefill_tok_s": (
            total_prompt_tokens / prefill_wall
            if prefill_wall and prefill_wall > 0 else None),
        "output_tok_s": total_output_tokens / wall if wall > 0 else 0.0,
        "requests": requests_data,
    }


def run_http_request(base_url, model_name, input_tokens, output_tokens,
                     request_timeout, request_id):
    expected_prompt_ids = list(input_tokens)
    payload = {
        "model": model_name,
        "prompt": expected_prompt_ids,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    start_time = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/completions", json=payload, stream=True,
        timeout=request_timeout)
    response.raise_for_status()
    token_times = []
    returned_prompt_ids = None
    usage = {}
    # 使用最小流式读取粒度，避免FastLLM/vLLM的SSE chunk大小不同导致客户端
    # 把多个Token攒到一起后再记录时间。
    for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            error = chunk["error"]
            message = error.get("message", str(error)) if isinstance(
                error, dict) else str(error)
            raise RuntimeError(f"请求{request_id}后端执行失败：{message}")
        usage = chunk.get("usage") or usage
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("prompt_token_ids") is not None:
            returned_prompt_ids = choice["prompt_token_ids"]
        delta_ids = choice.get("token_ids") or []
        now = time.perf_counter()
        token_times.extend([now] * len(delta_ids))
    end_time = time.perf_counter()
    if returned_prompt_ids is None:
        raise RuntimeError(
            f"请求{request_id}未返回Prompt Token ID；"
            f"流式收到{len(token_times)}个输出Token，"
            f"usage={usage or 'n/a'}")
    if not isinstance(returned_prompt_ids, list):
        raise RuntimeError(
            f"请求{request_id}返回的Prompt Token ID格式错误；"
            f"期望list，实际{type(returned_prompt_ids).__name__}")
    if returned_prompt_ids != expected_prompt_ids:
        common_length = min(
            len(expected_prompt_ids), len(returned_prompt_ids))
        mismatch_index = next(
            (index for index in range(common_length)
             if expected_prompt_ids[index] != returned_prompt_ids[index]),
            common_length)
        expected_token = (
            expected_prompt_ids[mismatch_index]
            if mismatch_index < len(expected_prompt_ids) else "<end>")
        returned_token = (
            returned_prompt_ids[mismatch_index]
            if mismatch_index < len(returned_prompt_ids) else "<end>")
        raise RuntimeError(
            f"请求{request_id}返回的Prompt Token ID不一致；"
            f"期望长度={len(expected_prompt_ids)}，"
            f"实际长度={len(returned_prompt_ids)}，"
            f"首个差异位置={mismatch_index}，"
            f"期望Token={expected_token}，实际Token={returned_token}")
    output_count = int(usage.get("completion_tokens") or len(token_times))
    if output_count != len(token_times) or output_count != output_tokens:
        raise RuntimeError(
            f"请求{request_id}应输出{output_tokens}个Token，流式收到"
            f"{len(token_times)}个，usage报告{output_count}个")
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_input_tokens = prompt_details.get("cached_tokens")
    if cached_input_tokens is not None:
        cached_input_tokens = int(cached_input_tokens)
    return {
        "request_id": request_id,
        "start_time": start_time,
        "end_time": end_time,
        "token_times": token_times,
        "output_tokens": output_count,
        "cached_input_tokens": cached_input_tokens,
    }


def run_http_batch(base_url, model_name, backend, mode, scenario, workload,
                   input_tokens, output_tokens, batch, request_timeout):
    batch_start = time.perf_counter()
    requests_data = []
    with ThreadPoolExecutor(max_workers=batch) as executor:
        futures = [
            executor.submit(
                run_http_request, base_url, model_name, input_tokens,
                output_tokens, request_timeout, request_id)
            for request_id in range(batch)
        ]
        for future in as_completed(futures):
            requests_data.append(future.result())
    requests_data.sort(key=lambda item: item["request_id"])
    batch_end = max(item["end_time"] for item in requests_data)
    return summarize_requests(
        backend, mode, scenario, workload, batch, input_tokens, output_tokens,
        requests_data, batch_start, batch_end)


def median_or_none(values):
    filtered = [value for value in values if value is not None]
    return statistics.median(filtered) if filtered else None


def aggregate_trials(trials):
    result = {key: value for key, value in trials[0].items()
              if key not in ("requests", "total_output_tokens")}
    metric_keys = (
        "ttft_s", "tpot_s", "itl_s", "e2el_s", "wall_s",
        "prefill_tok_s", "output_tok_s", "cache_hit_ratio")
    for key in metric_keys:
        result[key] = median_or_none([trial[key] for trial in trials])
    result["total_output_tokens"] = int(statistics.median(
        trial["total_output_tokens"] for trial in trials))
    result["total_cached_prompt_tokens"] = median_or_none(
        [trial["total_cached_prompt_tokens"] for trial in trials])
    result["repeat_count"] = len(trials)
    result["aggregate"] = "median"
    result["trials"] = trials
    return result


def run_backend_cases(base_url, model_name, backend, mode, prompts, args):
    results = []
    for scenario in SCENARIOS:
        for case in prompts["cases"].values():
            scenario_data = case[scenario]
            for _ in range(args.warmup):
                run_http_batch(
                    base_url, model_name, backend, mode, scenario, "warmup",
                    scenario_data["warmup_token_ids"],
                    min(case["output_length"], 8), case["batch"],
                    args.request_timeout)
            trials = []
            for input_tokens in scenario_data["trial_token_ids"]:
                trials.append(run_http_batch(
                    base_url, model_name, backend, mode, scenario,
                    case["workload"], input_tokens, case["output_length"],
                    case["batch"], args.request_timeout))
            result = aggregate_trials(trials)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
    return results


def wait_server(base_url, process, timeout, server_log, backend):
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            tail = Path(server_log).read_text(
                encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"{backend}服务提前退出，code={process.returncode}\n{tail}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"等待{backend}启动超时: {last_error}")


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_server(command, env, server_log, backend, base_url, prompts, args, mode):
    print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
    with Path(server_log).open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {shlex.join(command)}\n\n")
        for name in sorted(name for name in env if name.startswith(
                ("FASTLLM_CUDA_NVFP4", "FASTLLM_CUDA_GRAPH",
                 "FASTLLM_CUDA_W8A8", "FASTLLM_CUDA_PAGED_DEBUG",
                 "VLLM_USE_FLASHINFER"))):
            log_handle.write(f"ENV: {name}={env[name]}\n")
        log_handle.write("\n")
        log_handle.flush()
        process = subprocess.Popen(
            command, cwd=REPO_DIR, env=env, stdout=log_handle,
            stderr=subprocess.STDOUT, text=True)
        try:
            wait_server(base_url, process, args.startup_timeout, server_log, backend)
            return run_backend_cases(
                base_url, "nvfp4-performance", backend, mode, prompts, args)
        finally:
            stop_process(process)


def require_cache_hit(rows, backend):
    """确认标记为cache_hit的用例确实复用了至少一个Prompt Token。"""
    missed = [
        row for row in rows
        if row["scenario"] == "cache_hit" and
        (row.get("cache_hit_ratio") is None or row["cache_hit_ratio"] <= 0)
    ]
    if missed:
        cases = ", ".join(
            f"{row['workload']}/batch-{row['batch']}" for row in missed)
        raise RuntimeError(f"{backend}的cache_hit场景未实际命中Prefix Cache：{cases}")


def run_fastllm(args, prompts, result_dir, mode):
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        args.fastllm_python,
        str(REPO_DIR / "test/nvfp4/fastllm_http_benchmark_server.py"),
        "--model", args.model, "--host", "127.0.0.1",
        "--port", str(args.port), "--dtype", args.flm_dtype,
        "--atype", args.flm_atype, "--device", args.flm_device,
        "--max-batch", str(max(batch_cases(args))),
        "--kv-cache-layout", args.flm_kv_cache_layout,
        "--attention-backend", args.flm_attention_backend,
        "--attention-backend-trace", "--enable-prefix-cache",
    ]
    if args.flm_attention_backend_strict:
        command.append("--attention-backend-strict")
    env = os.environ.copy()
    env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
    env.pop("FASTLLM_CUDA_GRAPH_TRACE", None)
    if args.quantization == "nvfp4":
        env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
        env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
        env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
        env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
    elif args.quantization == "w8a8":
        env.pop("FASTLLM_CUDA_NVFP4_W4A4", None)
        env.pop("FASTLLM_CUDA_NVFP4_W4A4_STRICT", None)
        env.pop("FASTLLM_CUDA_MOE_NVFP4_W4A4", None)
        env.pop("FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT", None)
        env["FASTLLM_CUDA_W8A8"] = "1"
        env["FASTLLM_CUDA_W8A8_STRICT"] = "1"
    elif args.quantization == "a16":
        # 供W4A16/W8A16原生/Machete矩阵复用正式HTTP测量流程。
        for key in ("FASTLLM_CUDA_NVFP4_W4A4", "FASTLLM_CUDA_NVFP4_W4A4_STRICT",
                    "FASTLLM_CUDA_MOE_NVFP4_W4A4", "FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT",
                    "FASTLLM_CUDA_W8A8", "FASTLLM_CUDA_W8A8_STRICT"):
            env.pop(key, None)
    env["FASTLLM_CUDA_GRAPH"] = "0" if mode == "eager" else "1"
    server_log = result_dir / f"fastllm-{mode}-server.log"
    rows = run_server(
        command, env, server_log, "FastLLM", base_url, prompts, args, mode)
    require_cache_hit(rows, "FastLLM")
    server_output = server_log.read_text(encoding="utf-8", errors="replace")
    actual_backends = actual_attention_backends(server_output)
    actual_layouts = actual_kv_cache_layouts(server_output)
    confirmed = attention_backend_confirmed(
        args.flm_attention_backend, actual_backends)
    layout_confirmed = kv_cache_layout_confirmed(
        args.flm_kv_cache_layout, actual_layouts)
    if args.flm_attention_backend_strict and not confirmed:
        raise RuntimeError(
            "FastLLM请求的Attention Backend未在服务日志中确认："
            f"requested={args.flm_attention_backend}, "
            f"actual={actual_backends}")
    if not layout_confirmed:
        raise RuntimeError(
            "FastLLM请求的KV Cache布局未在服务日志中确认："
            f"requested={args.flm_kv_cache_layout}, actual={actual_layouts}")
    for row in rows:
        row["requested_attention_backend"] = args.flm_attention_backend
        row["actual_attention_backends"] = actual_backends
        row["attention_backend_confirmed"] = confirmed
        row["requested_kv_cache_layout"] = args.flm_kv_cache_layout
        row["actual_kv_cache_layouts"] = actual_layouts
        row["kv_cache_layout_confirmed"] = layout_confirmed
    return rows


def run_vllm(args, prompts, result_dir, mode):
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", "nvfp4-performance",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max(batch_cases(args))),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code", "--linear-backend", "cutlass",
        # Dense Linear与NVFP4 MoE分别受linear_backend和moe_backend控制。
        # 固定vLLM原生CUTLASS，避免auto选择FLASHINFER_CUTLASS触发运行时JIT。
        "--moe-backend", "cutlass",
        "--enable-prefix-caching", "--enable-force-include-usage",
        "--enable-prompt-tokens-details",
    ]
    if mode == "eager":
        command.append("--enforce-eager")
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
    rows = run_server(
        command, env, result_dir / f"vllm-{mode}-server.log", "vLLM",
        base_url, prompts, args, mode)
    require_cache_hit(rows, "vLLM")
    return rows


def fmt_ms(value):
    return "n/a" if value is None else f"{value * 1000:.3f}"


def fmt_number(value):
    return "n/a" if value is None else f"{value:.2f}"


def fmt_percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def ratio(fastllm_value, vllm_value):
    if fastllm_value is None or vllm_value in (None, 0):
        return "n/a"
    return f"{fastllm_value / vllm_value:.4f}x"


def ordered_results(fastllm_results, vllm_results, mode, scenario):
    def select(items):
        return {(item["workload"], item["batch"]): item for item in items
                if item["mode"] == mode and item["scenario"] == scenario}

    fastllm_map = select(fastllm_results)
    vllm_map = select(vllm_results)
    keys = list(dict.fromkeys([*fastllm_map, *vllm_map]))
    rows = []
    for key in keys:
        if key not in fastllm_map or key not in vllm_map:
            raise RuntimeError(f"FastLLM/vLLM缺少对应测试项: {mode}/{scenario}/{key}")
        rows.extend((fastllm_map[key], vllm_map[key]))
    return rows


def make_report(result_dir, fastllm_results, vllm_results, quantization):
    lines = [
        f"# {quantization.upper()}严格整体性能对比", "",
        "> 两个后端通过HTTP `/v1/completions`接收同一组Token ID；忽略EOS；",
        "> 每个Case测试5轮并取中位数。Cold与Cache Hit分别统计。", "",
        "> 每个Batch运行一组完整请求；Prefill/Decode两张图共享测量数据。",
        "> Prefill吞吐为总Prompt Token数/从批次开始到最后一个首Token到达的时间，",
        "> 包含缓存复用Token、排队及传输时间，不代表纯Prefill计算吞吐。", "",
    ]
    csv_rows = []
    comparison_rows = []
    for mode in MODES:
        for scenario in SCENARIOS:
            rows = ordered_results(
                fastllm_results, vllm_results, mode, scenario)
            lines.extend([
                f"## {mode.upper()} / {scenario}", "",
                "| 状态 | Workload | 后端 | KV Cache布局 | Attention Backend | Batch | Prompt/请求 | Prompt总数 | "
                "Output总数 | Cache命中 | TTFT(ms) | TPOT(ms) | ITL(ms) | E2EL(ms) | "
                "Prefill(tok/s) | Output(tok/s) | Wall(s) |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for item in rows:
                attention_backends = item.get("actual_attention_backends", [])
                attention_label = ",".join(attention_backends) or "n/a"
                kv_layouts = item.get("actual_kv_cache_layouts", [])
                kv_layout_label = ",".join(kv_layouts) or "n/a"
                lines.append(
                    f"| PASS | {item['workload']} | {item['backend']} | "
                    f"{kv_layout_label} | {attention_label} | {item['batch']} | "
                    f"{item['prompt_tokens_per_request']} | "
                    f"{item['total_prompt_tokens']} | {item['total_output_tokens']} | "
                    f"{fmt_percent(item['cache_hit_ratio'])} | "
                    f"{fmt_ms(item['ttft_s'])} | {fmt_ms(item['tpot_s'])} | "
                    f"{fmt_ms(item['itl_s'])} | {fmt_ms(item['e2el_s'])} | "
                    f"{fmt_number(item['prefill_tok_s'])} | "
                    f"{fmt_number(item['output_tok_s'])} | {item['wall_s']:.6f} |")
                csv_rows.append({
                    "result": "PASS",
                    "mode": mode,
                    "scenario": scenario,
                    "workload": item["workload"],
                    "backend": item["backend"],
                    "requested_attention_backend": item.get(
                        "requested_attention_backend", "n/a"),
                    "actual_attention_backends": json.dumps(
                        attention_backends, ensure_ascii=False),
                    "attention_backend_confirmed": item.get(
                        "attention_backend_confirmed", "n/a"),
                    "requested_kv_cache_layout": item.get(
                        "requested_kv_cache_layout", "n/a"),
                    "actual_kv_cache_layouts": json.dumps(
                        kv_layouts, ensure_ascii=False),
                    "kv_cache_layout_confirmed": item.get(
                        "kv_cache_layout_confirmed", "n/a"),
                    "batch": item["batch"],
                    "prompt_tokens_per_request": item["prompt_tokens_per_request"],
                    "total_prompt_tokens": item["total_prompt_tokens"],
                    "output_tokens": item["total_output_tokens"],
                    "cache_hit_ratio": item["cache_hit_ratio"],
                    "repeat_count": item["repeat_count"],
                    "ttft_ms": None if item["ttft_s"] is None else item["ttft_s"] * 1000,
                    "tpot_ms": None if item["tpot_s"] is None else item["tpot_s"] * 1000,
                    "itl_ms": None if item["itl_s"] is None else item["itl_s"] * 1000,
                    "e2el_ms": None if item["e2el_s"] is None else item["e2el_s"] * 1000,
                    "prefill_tok_s": item["prefill_tok_s"],
                    "output_tok_s": item["output_tok_s"],
                    "wall_s": item["wall_s"],
                })
            lines.extend([
                "", "### FastLLM / vLLM", "",
                "| Workload | Batch | TTFT | TPOT | ITL | E2EL | Output吞吐 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for index in range(0, len(rows), 2):
                fastllm_item, vllm_item = rows[index:index + 2]
                lines.append(
                    f"| {fastllm_item['workload']} | {fastllm_item['batch']} | "
                    f"{ratio(fastllm_item['ttft_s'], vllm_item['ttft_s'])} | "
                    f"{ratio(fastllm_item['tpot_s'], vllm_item['tpot_s'])} | "
                    f"{ratio(fastllm_item['itl_s'], vllm_item['itl_s'])} | "
                    f"{ratio(fastllm_item['e2el_s'], vllm_item['e2el_s'])} | "
                    f"{ratio(fastllm_item['output_tok_s'], vllm_item['output_tok_s'])} |")
                comparison_rows.append([
                    "PASS", mode, scenario, fastllm_item["workload"],
                    fastllm_item["batch"],
                    (None if fastllm_item["ttft_s"] is None or
                     vllm_item["ttft_s"] in (None, 0) else
                     fastllm_item["ttft_s"] / vllm_item["ttft_s"]),
                    (None if fastllm_item["tpot_s"] is None or
                     vllm_item["tpot_s"] in (None, 0) else
                     fastllm_item["tpot_s"] / vllm_item["tpot_s"]),
                    (None if fastllm_item["itl_s"] is None or
                     vllm_item["itl_s"] in (None, 0) else
                     fastllm_item["itl_s"] / vllm_item["itl_s"]),
                    (None if fastllm_item["e2el_s"] is None or
                     vllm_item["e2el_s"] in (None, 0) else
                     fastllm_item["e2el_s"] / vllm_item["e2el_s"]),
                    (None if fastllm_item["output_tok_s"] is None or
                     vllm_item["output_tok_s"] in (None, 0) else
                     fastllm_item["output_tok_s"] /
                     vllm_item["output_tok_s"]),
                ])
            lines.append("")
    chart_paths = []
    image_dir = result_dir / "images"
    for mode in MODES:
        for scenario in SCENARIOS:
            for workload in ("prefill", "decode"):
                chart_rows = [
                    row for row in csv_rows
                    if row["mode"] == mode and row["scenario"] == scenario
                ]
                filename = "model-performance-{}-{}-{}.png".format(
                    slugify(mode), slugify(scenario), slugify(workload))
                chart_paths.append(write_dashboard(
                    image_dir / filename, chart_rows,
                    model_performance_chart_specs(workload),
                    f"{quantization.upper()} {mode} {scenario} {workload}"))
    lines.extend(markdown_images(chart_paths))

    report = "\n".join(lines)
    markdown_path = result_dir / "model-performance-compare.md"
    csv_path = result_dir / "model-performance-compare.csv"
    markdown_path.write_text(report, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(result_dir / "model-performance-compare.json", {
        "summary": {"result": "PASS", "case_records": len(csv_rows)},
        "fastllm": fastllm_results,
        "vllm": vllm_results,
    })
    detail_headers = list(csv_rows[0])
    write_xlsx(result_dir / "model-performance-compare.xlsx", [
        ("性能明细", detail_headers,
         [[row[key] for key in detail_headers] for row in csv_rows]),
        ("FastLLM-vLLM比值", [
            "状态", "模式", "场景", "Workload", "Batch", "TTFT比", "TPOT比",
            "ITL比", "E2EL比", "Output吞吐比"], comparison_rows),
    ], image_sheets=[("Batch趋势图", chart_paths)])
    print(report)
    print(f"Summary: PASS ({len(csv_rows)} backend case records)")
    print(f"Markdown: {markdown_path}")
    print(f"CSV: {csv_path}")
    print(f"Excel: {result_dir / 'model-performance-compare.xlsx'}")
    print(f"Images: {image_dir}")


def orchestrate(args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = build_shared_prompts(args, result_dir)
    prompts = read_json(prompt_path)
    all_fastllm_results = []
    all_vllm_results = []
    for mode in MODES:
        fastllm_results = run_fastllm(args, prompts, result_dir, mode)
        write_json(result_dir / f"fastllm-{mode}-results.json", fastllm_results)
        all_fastllm_results.extend(fastllm_results)
        vllm_results = run_vllm(args, prompts, result_dir, mode)
        write_json(result_dir / f"vllm-{mode}-results.json", vllm_results)
        all_vllm_results.extend(vllm_results)
    make_report(result_dir, all_fastllm_results, all_vllm_results,
                args.quantization)


def main():
    args = parse_args()
    require_positive("input-tokens", args.input_tokens)
    require_positive("output-tokens", args.output_tokens)
    batch_cases(args)
    require_positive("repeats", args.repeats)
    if args.warmup < 0:
        raise ValueError("warmup不能小于0")
    orchestrate(args)


if __name__ == "__main__":
    main()
