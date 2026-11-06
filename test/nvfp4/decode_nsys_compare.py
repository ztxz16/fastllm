#!/usr/bin/env python3
"""按Decode Batch采集并对比FastLLM与vLLM的Nsight Systems报告。"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import requests


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))
sys.path.insert(0, str(REPO_DIR / "test"))

import model_performance_compare as model_compare  # noqa: E402
from common.performance_report import (  # noqa: E402
    ChartSpec, markdown_images, write_dashboard)
from common.attention_backend_report import (  # noqa: E402
    actual_attention_backends, actual_kv_cache_layouts,
    attention_backend_confirmed, kv_cache_layout_confirmed)
from xlsx_report import write_xlsx  # noqa: E402


BACKENDS = ("fastllm", "vllm")

# 报告中的类别顺序按一次Transformer Decode的主要数据流排列。这里描述的
# 是Kernel语义，而不是模型源码中的精确调用栈；同名GEMM无法仅凭Nsight
# 符号区分o_proj和MLP，融合Kernel也只归到可确认的最小语义范围。
KERNEL_CATEGORY_ORDER = (
    "Embedding",
    "Type Conversion",
    "Quantize",
    "GEMM/Linear",
    "QKV Postprocess",
    "KV Cache",
    "Attention",
    "Norm",
    "Activation",
    "Residual/Elementwise",
    "Layout/Index",
    "Sampling",
    "Fused/Unresolved",
    "Other",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="按Batch对比FastLLM/vLLM稳定Decode阶段的Nsight Systems数据")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--quantization", choices=("nvfp4", "w8a8", "a16"),
                        default="w8a8")
    parser.add_argument("--backends", default="fastllm,vllm",
                        help="逗号分隔：fastllm,vllm")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument(
        "--prefix-cache-padding-tokens", type=int, default=128,
        help="Warmup Prompt在正式Prompt之后追加的Token数，用于缓存正式Prompt的最后一页")
    parser.add_argument(
        "--collection-ready-timeout", type=float, default=30.0,
        help="等待Nsight会话进入Collection状态的超时时间（秒）")
    parser.add_argument(
        "--collection-ready-delay", type=float, default=1.0,
        help="Nsight进入Collection状态后、发送正式请求前额外等待的时间（秒）")
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
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--top-kernels", type=int, default=20)
    parser.add_argument(
        "--keep-nsys-reports", action="store_true",
        help=("保留每个Batch的nsys-rep、SQLite及导出CSV；默认在统计完成后"
              "立即删除这些大体积中间文件"))
    parser.add_argument(
        "--cpu-trace", action="store_true",
        help=("在同一次Nsight采集中增加OS Runtime，并汇总稳定Decode区间的"
              "CUDA API、Kernel提交、GPU排队及TPOT未解释时间"))
    parser.add_argument(
        "--vllm-extra-arg", action="append", default=[],
        help="追加一个vLLM服务参数；需要多个参数时重复使用")
    return parser.parse_args()


def parse_positive_csv(value, label):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label}必须是逗号分隔的正整数")
    return list(dict.fromkeys(values))


def selected_backends(value):
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in BACKENDS for item in values):
        raise ValueError("backends只能包含fastllm和vllm")
    return list(dict.fromkeys(values))


def run_command(command, *, env=None, check=True, capture=True):
    print(f"COMMAND: {shlex.join([str(item) for item in command])}", flush=True)
    return subprocess.run(
        [str(item) for item in command], cwd=REPO_DIR, env=env, check=check,
        text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None)


def nsys_control_env():
    # FastLLM Python可能需要预加载系统libstdc++，但该库不能注入Nsight自身的
    # importer，否则部分Nsight版本只能留下qdstrm而无法生成nsys-rep。
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    return env


def target_command(command):
    preload = os.environ.get("LD_PRELOAD", "")
    if not preload:
        return command
    return ["/usr/bin/env", f"LD_PRELOAD={preload}", *command]


def make_prompt_tokens(tokenizer, length, batch):
    return model_compare.render_prompt_tokens(
        tokenizer, length, f"decode nsys batch {batch}")


def wait_server(base_url, launcher, timeout, log_path, backend):
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if launcher.poll() is not None:
            tail = log_path.read_text(
                encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(
                f"{backend} Nsight服务提前退出，code={launcher.returncode}\n{tail}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"等待{backend}服务启动超时：{last_error}")


def server_spec(args, backend, max_batch):
    if backend == "fastllm":
        command = [
            args.fastllm_python,
            str(REPO_DIR / "test/nvfp4/fastllm_http_benchmark_server.py"),
            "--model", args.model, "--host", "127.0.0.1",
            "--port", str(args.port), "--dtype", args.flm_dtype,
            "--atype", args.flm_atype, "--device", args.flm_device,
            "--max-batch", str(max_batch), "--enable-prefix-cache",
            "--kv-cache-layout", args.flm_kv_cache_layout,
            "--attention-backend", args.flm_attention_backend,
            "--attention-backend-trace",
        ]
        if args.flm_attention_backend_strict:
            command.append("--attention-backend-strict")
        env = os.environ.copy()
        env["FASTLLM_CUDA_GRAPH"] = "1"
        env.pop("FASTLLM_CUDA_GRAPH_TRACE", None)
        env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
        if args.quantization == "w8a8":
            env["FASTLLM_CUDA_W8A8"] = "1"
            env["FASTLLM_CUDA_W8A8_STRICT"] = "1"
        elif args.quantization == "nvfp4":
            env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
            env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
            env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
            env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
        elif args.quantization == "a16":
            for key in ("FASTLLM_CUDA_NVFP4_W4A4", "FASTLLM_CUDA_NVFP4_W4A4_STRICT",
                        "FASTLLM_CUDA_MOE_NVFP4_W4A4", "FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT",
                        "FASTLLM_CUDA_W8A8", "FASTLLM_CUDA_W8A8_STRICT"):
                env.pop(key, None)
        return command, env, "nvfp4-performance"

    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", "decode-nsys",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max_batch), "--gpu-memory-utilization",
        str(args.gpu_memory_utilization), "--trust-remote-code",
        "--linear-backend", "cutlass", "--enable-prefix-caching",
        "--enable-force-include-usage",
    ]
    if args.quantization == "nvfp4":
        command.extend(("--moe-backend", "cutlass"))
    command.extend(args.vllm_extra_arg)
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
    return command, env, "decode-nsys"


def read_attention_backend_selection(log_path):
    """从FastLLM服务日志读取框架实际选择的Attention后端。"""
    return actual_attention_backends(
        log_path.read_text(encoding="utf-8", errors="replace"))


def read_kv_cache_layout_selection(log_path):
    """从FastLLM服务日志读取模型实际使用的KV Cache布局。"""
    return actual_kv_cache_layouts(
        log_path.read_text(encoding="utf-8", errors="replace"))


def launch_profiled_server(args, backend, max_batch, backend_dir):
    command, target_env, served_name = server_spec(args, backend, max_batch)
    session = re.sub(
        r"[^A-Za-z0-9_-]", "_",
        f"decode_{backend}_{os.getpid()}_{int(time.time())}")
    trace_targets = "cuda,nvtx,osrt" if args.cpu_trace else "cuda,nvtx"
    launch = [
        args.nsys, "launch", f"--session-new={session}",
        f"--trace={trace_targets}", "--cuda-graph-trace=node",
        *target_command(command),
    ]
    log_path = backend_dir / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"COMMAND: {shlex.join(launch)}\n\n")
    log_handle.flush()
    control_env = nsys_control_env()
    # 除LD_PRELOAD外，目标服务与Nsight控制进程共享显式测试环境。
    control_env.update({key: value for key, value in target_env.items()
                        if key != "LD_PRELOAD"})
    launcher = subprocess.Popen(
        launch, cwd=REPO_DIR, env=control_env, stdout=log_handle,
        stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        wait_server(base_url, launcher, args.startup_timeout, log_path, backend)
    except Exception:
        log_handle.close()
        shutdown_session(args, session, launcher)
        raise
    return {
        "backend": backend,
        "session": session,
        "launcher": launcher,
        "log_handle": log_handle,
        "log_path": log_path,
        "base_url": base_url,
        "served_name": served_name,
    }


def shutdown_session(args, session, launcher):
    run_command(
        [args.nsys, "shutdown", f"--session={session}", "--kill=sigterm"],
        env=nsys_control_env(), check=False)
    if launcher.poll() is None:
        try:
            launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            launcher.terminate()
            try:
                launcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                launcher.kill()


def resolve_report(output_base):
    direct = output_base.with_suffix(".nsys-rep")
    if direct.exists():
        return direct
    matches = sorted(output_base.parent.glob(output_base.name + "*.nsys-rep"))
    if matches:
        return matches[-1]
    qdstrm = output_base.with_suffix(".qdstrm")
    detail = f"；仅生成了{qdstrm}" if qdstrm.exists() else ""
    raise RuntimeError(
        "Nsight未生成nsys-rep" + detail +
        "。请确认Nsight importer可运行，并确保LD_PRELOAD未注入nsys进程。")


def start_collection(args, session, output_base):
    result = run_command([
        args.nsys, "start", f"--session={session}", "--sample=none",
        "--cpuctxsw=none", f"--output={output_base}",
        "--force-overwrite=true",
    ], env=nsys_control_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nsys start失败：\n{result.stdout}")


def wait_collection_ready(args, session):
    """等待Nsight真正进入采集状态，避免遗漏短暂的Prefill开头。"""
    deadline = time.time() + args.collection_ready_timeout
    last_output = ""
    while time.time() < deadline:
        result = subprocess.run(
            [str(args.nsys), "sessions", "list"], cwd=REPO_DIR,
            env=nsys_control_env(), check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_output = result.stdout or ""
        if (result.returncode == 0 and session in last_output and
                re.search(r"\bCollection\b", last_output)):
            # Collection状态表示控制面已切换；再留出少量时间，让目标进程的
            # CUDA追踪完成挂接，然后才发送本次需要统计的HTTP请求。
            time.sleep(args.collection_ready_delay)
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"等待Nsight会话{session}进入Collection状态超时：\n{last_output}")


def stop_collection(args, session, output_base):
    result = run_command(
        [args.nsys, "stop", f"--session={session}"],
        env=nsys_control_env(), check=False)
    if result.stdout:
        (output_base.parent / f"{output_base.name}-stop.log").write_text(
            result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"nsys stop失败：\n{result.stdout}")
    return resolve_report(output_base)


def run_decode_batch(server, prompt, output_tokens, batch, args, workload):
    return model_compare.run_http_batch(
        server["base_url"], server["served_name"], server["backend"],
        "best", "cache_hit", workload, prompt, output_tokens, batch,
        args.request_timeout)


def make_cache_warmup_prompt(prompt, padding_tokens):
    """构造以正式Prompt为完整前缀的加长Prompt，使最后一个Prefix页可缓存。"""
    if not prompt:
        raise ValueError("Decode Nsight Prompt不能为空")
    if padding_tokens <= 0:
        return list(prompt)
    # 复用Prompt中已有的有效Token ID，避免依赖不同Tokenizer的特殊Token。
    padding_source = prompt[-min(len(prompt), padding_tokens):]
    repeats = (padding_tokens + len(padding_source) - 1) // len(padding_source)
    return list(prompt) + (padding_source * repeats)[:padding_tokens]


def export_stats(args, report, report_name, output_base):
    command = [
        args.nsys, "stats", "--force-export=true", "--force-overwrite=true",
        "--format=csv", f"--report={report_name}",
        f"--output={output_base}", str(report),
    ]
    result = run_command(command, env=nsys_control_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"生成Nsight统计{report_name}失败：\n{result.stdout}")
    candidates = sorted(output_base.parent.glob(
        output_base.name + "*" + report_name.split(":", 1)[0] + "*.csv"))
    if not candidates:
        candidates = sorted(output_base.parent.glob(output_base.name + "*.csv"))
    if not candidates:
        raise RuntimeError(f"Nsight没有输出{report_name} CSV")
    return candidates[-1]


def export_optional_stats(args, report, report_name, output_base):
    """导出可能为空的Nsight报告；无对应事件时返回None。"""
    try:
        return export_stats(args, report, report_name, output_base)
    except RuntimeError as error:
        print(f"WARNING: 跳过可选Nsight报告{report_name}: {error}",
              file=sys.stderr, flush=True)
        return None


def numeric(row, names):
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        value = value.replace(",", "").strip()
        try:
            return float(value)
        except ValueError:
            continue
    return 0.0


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_intervals(intervals):
    """汇总可能来自多个CPU线程的时间区间，分别保留累计与去重墙钟时间。"""
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return {"sum_ms": 0.0, "union_ms": 0.0}
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return {
        "sum_ms": sum(end - start for start, end in intervals) / 1e6,
        "union_ms": sum(end - start for start, end in merged) / 1e6,
    }


def text_value(row, names):
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return "unknown"


def read_stats_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_named_rows(rows):
    result = []
    for row in rows:
        total_ns = numeric(row, ("Total Time (ns)", "Total Time"))
        instances = int(numeric(row, (
            "Instances", "Count", "Calls", "Num Calls")))
        average_ns = numeric(row, ("Avg (ns)", "Average (ns)", "Avg"))
        if not average_ns and instances:
            average_ns = total_ns / instances
        result.append({
            "name": text_value(row, ("Name", "Operation", "API Name")),
            "total_ms": total_ns / 1e6,
            "instances": instances,
            "average_us": average_ns / 1e3,
        })
    result.sort(key=lambda item: item["total_ms"], reverse=True)
    return result


def trace_name(row):
    return text_value(row, ("Name", "Kernel Name", "Operation"))


def is_memory_trace_row(row):
    name = trace_name(row).lower()
    if any(row.get(key) for key in (
            "SrcMemKind", "DstMemKind", "Source Memory Kind",
            "Destination Memory Kind")):
        return True
    return "memcpy" in name or "memset" in name or "memory operation" in name


def summarize_trace_kernels(rows):
    """从逐事件GPU Trace汇总Kernel，保留裁剪稳定Decode区间的能力。"""
    grouped = {}
    for row in rows:
        if is_memory_trace_row(row):
            continue
        duration_ns = numeric(row, ("Duration (ns)", "Duration"))
        if duration_ns <= 0:
            continue
        name = trace_name(row)
        entry = grouped.setdefault(name, {"total_ns": 0.0, "instances": 0})
        entry["total_ns"] += duration_ns
        entry["instances"] += 1
    result = []
    for name, value in grouped.items():
        total_ns = value["total_ns"]
        instances = value["instances"]
        result.append({
            "name": name,
            "total_ms": total_ns / 1e6,
            "instances": instances,
            "average_us": total_ns / instances / 1e3,
        })
    result.sort(key=lambda item: item["total_ms"], reverse=True)
    return result


def kernel_category(name):
    lowered = name.lower()
    # 分类必须先识别语义明确的kernel，不能看到字符串"cutlass"就认为是
    # GEMM：FlashAttention模板参数也包含cutlass类型，Triton融合kernel名称
    # 则可能同时出现scaled_mm、RMSNorm、SwiGLU和量化。
    if any(pattern in lowered for pattern in (
            "qkvrmsnormropesplit", "ropeencoding", "rotary_embedding",
            "rotaryembedding", "apply_rotary", "fused_rope")):
        return "QKV Postprocess"
    if any(pattern in lowered for pattern in (
            "reshape_and_cache", "pagedcachecopy", "appendpagedcache",
            "kv_cache", "cache_kernel", "gather_block_tables")):
        return "KV Cache"
    if any(pattern in lowered for pattern in (
            "flash_fwd", "flashinfer::batchprefill",
            "batchprefillwithpagedkvcache", "attention", "fmha",
            "fused_attn", "persistentvariablelengthmergestates",
            "flashmla", "mla::")):
        return "Attention"
    if any(pattern in lowered for pattern in (
            "gumbel", "sampling", "sample_kernel", "topk", "top_k", "topp",
            "top_p", "argmax", "layernormkerneltop1", "resetlogitsofeos")):
        return "Sampling"
    if any(pattern in lowered for pattern in (
            "embedding", "indexselectlargeindex",
            "indexselectsmallindex", "embedding_lookup")):
        return "Embedding"
    if any(pattern in lowered for pattern in (
            "float2bf16", "bf162float", "half2float", "float2half",
            "bf16tofloat", "floattobf16", "cast_kernel", "typeconvert",
            "convert_kernel")):
        return "Type Conversion"
    if any(pattern in lowered for pattern in (
            "transpos", "permute", "reorder", "indexselect", "index_select",
            "gatherkernel", "scatterkernel")):
        return "Layout/Index"
    if any(pattern in lowered for pattern in (
            "addtokernel", "residual", "elementwise", "pointwise",
            "add_kernel", "addkernel")):
        return "Residual/Elementwise"
    semantic_markers = sum(pattern in lowered for pattern in (
        "scaled_mm", "rms_norm", "rmsnorm", "silu", "swiglu", "quant",
        "embedding", "rope"))
    if "triton" in lowered and semantic_markers >= 2:
        return "Fused/Unresolved"
    if any(pattern in lowered for pattern in (
            "quantize", "scaled_fp8_quant", "nvfp4quant", "float4quant")):
        return "Quantize"
    if any(pattern in lowered for pattern in (
            "fused_add_rms_norm", "rmsnorm", "rms_norm", "layernorm",
            "layer_norm")):
        return "Norm"
    if any(pattern in lowered for pattern in (
            "swiglu", "silu", "gelu", "activation")):
        return "Activation"
    if any(pattern in lowered for pattern in (
            "gemmkernel", "gemm::kernel", "gemv", "scaled_mm",
            "matmul", "marlin", "kernel2<cutlass")):
        return "GEMM/Linear"
    return "Other"


def ordered_categories(categories):
    """按模型数据流排序算子类别，并在末尾保留未知扩展类别。"""
    known = [name for name in KERNEL_CATEGORY_ORDER if name in categories]
    return known + sorted(set(categories) - set(KERNEL_CATEGORY_ORDER))


def kernel_match_signature(name):
    """生成跨FastLLM/vLLM名称空间的保守Kernel匹配签名。"""
    is_fastllm_cutlass = "fastllm_w8a8_dense_detail::DenseKernel" in name
    is_vllm_cutlass = "vllm::cutlass_3x_gemm" in name
    if is_fastllm_cutlass or is_vllm_cutlass:
        shapes = re.findall(r"cute::C<\(int\)(\d+)>", name)
        architecture = re.search(r"cutlass::arch::Sm(\d+)", name)
        mainloop = re.search(r"cutlass::gemm::(Kernel[\w]+)", name)
        epilogue = re.search(r"cutlass::epilogue::([\w:]+)", name)
        if len(shapes) >= 6 and architecture and mainloop and epilogue:
            input_type = (
                "int8" if "signed char" in name else
                "fp8-e4m3" if "e4m3" in name.lower() else "unknown")
            output_type = (
                "bf16" if "bfloat16_t" in name else
                "fp16" if "half_t" in name else "unknown")
            tile = "x".join(shapes[:3])
            cluster = "x".join(shapes[3:6])
            arch = f"SM{architecture.group(1)}"
            label = f"{arch} {input_type.upper()} {tile} cluster={cluster}"
            signature = "|".join((
                "cutlass-gemm", arch, input_type, output_type, tile, cluster,
                mainloop.group(1), epilogue.group(1)))
            return signature, "structural", label

    lowered = name.lower()
    if ("quantizepertoken" in lowered and "signed char" in lowered) or \
            "dynamic_scaled_int8_quant_kernel" in lowered:
        return ("dynamic-int8-per-token-quant", "semantic",
                "Dynamic INT8 per-token quantize")

    # 相同的Nsight符号只证明进入同名实现，不能替代SASS一致性校验。
    return f"symbol|{name}", "symbol", name.split("(", 1)[0].strip()


def build_kernel_matches(rows):
    """基于全部汇总Kernel建立跨后端配对，并返回Top页注释映射。"""
    grouped = {}
    for row in rows:
        kernels = row["nsys"].get("kernels", row["nsys"]["top_kernels"])
        for item in kernels:
            signature, level, label = kernel_match_signature(item["name"])
            entry = grouped.setdefault(
                (row["batch"], signature),
                {"level": level, "label": label,
                 "fastllm": [], "vllm": []})
            entry[row["backend"]].append(item)

    matches = []
    annotations = {}
    match_index = 1
    for (batch, signature), entry in sorted(grouped.items()):
        if not entry["fastllm"] or not entry["vllm"]:
            continue
        match_id = f"K{match_index:03d}"
        match_index += 1

        def aggregate(items):
            instances = sum(item["instances"] for item in items)
            total_ms = sum(item["total_ms"] for item in items)
            return {
                "name": " | ".join(item["name"] for item in items),
                "instances": instances,
                "total_ms": total_ms,
                "average_us": total_ms * 1000.0 / instances,
            }

        fast = aggregate(entry["fastllm"])
        vllm = aggregate(entry["vllm"])
        common_instances = min(fast["instances"], vllm["instances"])
        match = {
            "batch": batch,
            "match_id": match_id,
            "signature": signature,
            "label": entry["label"],
            "level": entry["level"],
            "category": kernel_category(fast["name"]),
            "fastllm": fast,
            "vllm": vllm,
            "common_instances": common_instances,
            "fastllm_extra_instances": fast["instances"] - common_instances,
            "vllm_extra_instances": vllm["instances"] - common_instances,
            "instance_ratio": fast["instances"] / vllm["instances"],
            "average_us_ratio": fast["average_us"] / vllm["average_us"],
            "total_ms_ratio": fast["total_ms"] / vllm["total_ms"],
        }
        matches.append(match)
        for backend in BACKENDS:
            for item in entry[backend]:
                annotations[(batch, backend, item["name"])] = (
                    match_id, entry["level"], entry["label"])
    return matches, annotations


def stable_decode_trace(rows):
    """裁掉首Token以前的Prefill与采样，只保留后续稳定Decode GPU事件。"""
    events = []
    for row in rows:
        start = numeric(row, ("Start (ns)", "Start"))
        duration = numeric(row, ("Duration (ns)", "Duration"))
        if duration > 0:
            events.append((start, start + duration, row))
    events.sort(key=lambda item: (item[0], item[1]))

    first_sampling = None
    for index, (_, _, row) in enumerate(events):
        if (not is_memory_trace_row(row) and
                kernel_category(trace_name(row)) == "Sampling"):
            first_sampling = index
            break
    if first_sampling is None:
        raise RuntimeError(
            "Nsight GPU Trace中未找到首Token采样Kernel，无法排除Prefill。"
            "请检查Sampling分类规则或确认请求确实生成了Token。")

    # FastLLM与vLLM的Sampling可能分别由一个或多个Kernel组成。首个采样
    # Kernel之后，遇到下一轮Embedding/Norm/量化/GEMM/Attention中的任一
    # 模型计算，即认为稳定Decode开始；采样内部的数据转换不会误作边界。
    model_categories = {
        "Embedding", "Quantize", "GEMM/Linear", "QKV Postprocess",
        "KV Cache", "Attention", "Norm", "Activation",
        "Residual/Elementwise",
    }
    boundary_index = None
    for index in range(first_sampling + 1, len(events)):
        row = events[index][2]
        if is_memory_trace_row(row):
            continue
        if kernel_category(trace_name(row)) in model_categories:
            boundary_index = index
            break
    if boundary_index is None:
        raise RuntimeError(
            "首Token采样后未找到稳定Decode模型Kernel，无法确定采集边界。")

    boundary_ns = events[boundary_index][0]
    stable_rows = [row for start, _, row in events if start >= boundary_ns]
    stable_end_ns = max(
        start + numeric(row, ("Duration (ns)", "Duration"))
        for start, _, row in events if start >= boundary_ns)
    return stable_rows, {
        "boundary_ns": boundary_ns,
        "stable_end_ns": stable_end_ns,
        "excluded_event_count": boundary_index,
        "stable_event_count": len(stable_rows),
        "first_sampling_kernel": trace_name(events[first_sampling][2]),
        "first_stable_kernel": trace_name(events[boundary_index][2]),
    }


def summarize_gpu_timeline(rows):
    intervals = []
    for row in rows:
        start = numeric(row, ("Start (ns)", "Start"))
        duration = numeric(row, ("Duration (ns)", "Duration"))
        if duration > 0:
            intervals.append((start, start + duration))
    if not intervals:
        return {
            "event_count": 0, "span_ms": 0.0, "active_ms": 0.0,
            "idle_ms": 0.0, "idle_ratio": None,
        }
    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    span_ns = max(end for _, end in intervals) - min(
        start for start, _ in intervals)
    active_ns = sum(end - start for start, end in merged)
    idle_ns = max(0.0, span_ns - active_ns)
    return {
        "event_count": len(intervals),
        "span_ms": span_ns / 1e6,
        "active_ms": active_ns / 1e6,
        "idle_ms": idle_ns / 1e6,
        "idle_ratio": idle_ns / span_ns if span_ns > 0 else 0.0,
    }


def cuda_api_category(name):
    """把CUDA Runtime/Driver API归并为少量CPU提交阶段。"""
    lowered = name.lower()
    if "launch" in lowered:
        return "Kernel Launch"
    if "synchronize" in lowered or "eventquery" in lowered:
        return "Synchronization"
    if "memcpy" in lowered or "memset" in lowered:
        return "Memory"
    if "malloc" in lowered or "free" in lowered or "alloc" in lowered:
        return "Allocation"
    return "Other CUDA API"


def summarize_cpu_trace(kernel_exec_rows, api_rows, decode_filter):
    """汇总稳定Decode窗口内CPU侧CUDA提交和GPU排队时间。"""
    begin_ns = decode_filter["boundary_ns"]
    end_ns = decode_filter["stable_end_ns"]
    kernel_executions = []
    unique_launches = {}
    launcher_threads = set()
    for row in kernel_exec_rows:
        kernel_start = numeric(row, ("Kernel Start (ns)", "Kernel Start"))
        if kernel_start < begin_ns or kernel_start > end_ns:
            continue
        api_start = numeric(row, ("API Start (ns)", "API Start"))
        api_duration = numeric(row, ("API Dur (ns)", "API Dur"))
        queue_duration = numeric(row, ("Queue Dur (ns)", "Queue Dur"))
        thread_id = text_value(row, ("TID", "Tid", "Thread ID"))
        if thread_id != "unknown":
            launcher_threads.add(thread_id)
        api_name = text_value(row, ("API Function", "API Name", "Name"))
        if api_start > 0:
            unique_launches.setdefault(
                (api_start, api_duration, thread_id, api_name), {
                    "start": api_start, "end": api_start + api_duration,
                    "api_us": api_duration / 1e3,
                })
        kernel_executions.append({
            "queue_us": queue_duration / 1e3,
            "launch_to_kernel_us": (
                max(0.0, kernel_start - api_start) / 1e3
                if api_start > 0 else 0.0),
        })

    stable_apis = []
    api_intervals = []
    api_groups = {}
    for row in api_rows:
        start = numeric(row, ("Start (ns)", "Start"))
        duration = numeric(row, ("Duration (ns)", "Duration"))
        finish = start + duration
        if duration <= 0 or finish < begin_ns or start > end_ns:
            continue
        name = text_value(row, ("Name", "API Name"))
        stable_apis.append((name, duration))
        api_intervals.append((start, finish))
        group = api_groups.setdefault(
            cuda_api_category(name), {"total_ms": 0.0, "calls": 0})
        group["total_ms"] += duration / 1e6
        group["calls"] += 1

    named_apis = {}
    for name, duration in stable_apis:
        entry = named_apis.setdefault(name, {"total_ns": 0.0, "calls": 0})
        entry["total_ns"] += duration
        entry["calls"] += 1
    top_apis = []
    for name, value in named_apis.items():
        top_apis.append({
            "name": name,
            "total_ms": value["total_ns"] / 1e6,
            "calls": value["calls"],
            "average_us": value["total_ns"] / value["calls"] / 1e3,
        })
    top_apis.sort(key=lambda item: item["total_ms"], reverse=True)

    launch_intervals = [
        (item["start"], item["end"]) for item in unique_launches.values()]
    api_us = [item["api_us"] for item in unique_launches.values()]
    queue_us = [item["queue_us"] for item in kernel_executions]
    launch_to_kernel_us = [
        item["launch_to_kernel_us"] for item in kernel_executions]
    return {
        "scope": "stable_decode_gpu_window",
        "window_start_ns": begin_ns,
        "window_end_ns": end_ns,
        "kernel_execution_count": len(kernel_executions),
        "launch_api_count": len(unique_launches),
        "launcher_threads": len(launcher_threads),
        "launch_api": {
            **summarize_intervals(launch_intervals),
            "median_us": percentile(api_us, 50),
            "p95_us": percentile(api_us, 95),
        },
        "gpu_queue": {
            "sum_ms": sum(queue_us) / 1e3,
            "median_us": percentile(queue_us, 50),
            "p95_us": percentile(queue_us, 95),
        },
        "launch_to_kernel": {
            "median_us": percentile(launch_to_kernel_us, 50),
            "p95_us": percentile(launch_to_kernel_us, 95),
        },
        "cuda_api": {
            **summarize_intervals(api_intervals),
            "calls": len(stable_apis),
            "categories": api_groups,
            "top": top_apis,
        },
    }


def summarize_report(args, report, output_base):
    kernel_csv = export_stats(
        args, report, "cuda_gpu_kern_sum", output_base.with_name(
            output_base.name + "-kernels"))
    api_csv = export_stats(
        args, report, "cuda_api_sum", output_base.with_name(
            output_base.name + "-cuda-api"))
    memory_csv = export_stats(
        args, report, "cuda_gpu_mem_time_sum", output_base.with_name(
            output_base.name + "-memory"))
    timeline_csv = export_stats(
        args, report, "cuda_gpu_trace", output_base.with_name(
            output_base.name + "-gpu-trace"))
    kernel_exec_csv = None
    api_trace_csv = None
    osrt_csv = None
    nvtx_csv = None
    if args.cpu_trace:
        kernel_exec_csv = export_stats(
            args, report, "cuda_kern_exec_trace", output_base.with_name(
                output_base.name + "-kernel-exec"))
        api_trace_csv = export_stats(
            args, report, "cuda_api_trace", output_base.with_name(
                output_base.name + "-cuda-api-trace"))
        osrt_csv = export_optional_stats(
            args, report, "osrt_sum", output_base.with_name(
                output_base.name + "-os-runtime"))
        nvtx_csv = export_optional_stats(
            args, report, "nvtx_sum", output_base.with_name(
                output_base.name + "-nvtx"))
    # cuda_gpu_kern_sum保留完整请求，便于人工核对；正式Decode汇总必须从
    # 逐事件Trace裁掉首Token以前的Prefill与采样，避免Cache最后一页重算
    # 污染M形状、Kernel次数和GPU总时间。
    trace_rows, decode_filter = stable_decode_trace(
        read_stats_csv(timeline_csv))
    kernels = summarize_trace_kernels(trace_rows)
    apis = summarize_named_rows(read_stats_csv(api_csv))
    memory = summarize_named_rows(read_stats_csv(memory_csv))
    timeline = summarize_gpu_timeline(trace_rows)
    cpu_trace = None
    if args.cpu_trace:
        cpu_trace = summarize_cpu_trace(
            read_stats_csv(kernel_exec_csv), read_stats_csv(api_trace_csv),
            decode_filter)
        cpu_trace["kernel_exec_csv"] = str(kernel_exec_csv)
        cpu_trace["api_trace_csv"] = str(api_trace_csv)
        cpu_trace["osrt_csv"] = None if osrt_csv is None else str(osrt_csv)
        cpu_trace["nvtx_csv"] = None if nvtx_csv is None else str(nvtx_csv)
        cpu_trace["os_runtime_scope"] = "full_collected_request"
        cpu_trace["top_os_runtime"] = (
            [] if osrt_csv is None else
            summarize_named_rows(read_stats_csv(osrt_csv))[:10])
        cpu_trace["top_nvtx"] = (
            [] if nvtx_csv is None else
            summarize_named_rows(read_stats_csv(nvtx_csv))[:10])
    stable_memory_total_ms = sum(
        numeric(row, ("Duration (ns)", "Duration"))
        for row in trace_rows if is_memory_trace_row(row)) / 1e6
    categories = {}
    for item in kernels:
        category = kernel_category(item["name"])
        entry = categories.setdefault(
            category, {"total_ms": 0.0, "instances": 0})
        entry["total_ms"] += item["total_ms"]
        entry["instances"] += item["instances"]
    return {
        "report": str(report),
        "kernel_csv": str(kernel_csv),
        "cuda_api_csv": str(api_csv),
        "memory_csv": str(memory_csv),
        "timeline_csv": str(timeline_csv),
        "kernel_total_ms": sum(item["total_ms"] for item in kernels),
        "kernel_instances": sum(item["instances"] for item in kernels),
        "memory_total_ms": stable_memory_total_ms,
        "timeline": timeline,
        "stable_decode_filter": decode_filter,
        "categories": categories,
        "kernels": kernels,
        "top_kernels": kernels[:args.top_kernels],
        "top_cuda_apis": apis[:10],
        "memory_operations": memory,
        "cpu_trace": cpu_trace,
    }


def cleanup_nsys_artifacts(case_dir, output_base, summary=None):
    """删除单个Batch已经解析完成的Nsight原始报告和导出明细。"""
    patterns = (
        output_base.name + "*.nsys-rep",
        output_base.name + "*.qdstrm",
        output_base.name + "*.sqlite",
        output_base.name + "-*.csv",
    )
    deleted = []
    for pattern in patterns:
        for path in sorted(case_dir.glob(pattern)):
            if not path.is_file():
                continue
            path.unlink()
            deleted.append(str(path))

    if summary is not None:
        for key in ("report", "kernel_csv", "cuda_api_csv", "memory_csv",
                    "timeline_csv"):
            summary[key] = None
        cpu_trace = summary.get("cpu_trace")
        if cpu_trace is not None:
            for key in ("kernel_exec_csv", "api_trace_csv", "osrt_csv",
                        "nvtx_csv"):
                cpu_trace[key] = None
        summary["nsys_artifacts_retained"] = False
        summary["deleted_nsys_artifact_count"] = len(deleted)
    return deleted


def profile_backend(args, backend, batches, prompts, result_dir):
    backend_dir = result_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    server = None
    rows = []
    try:
        for batch in batches:
            case_dir = backend_dir / f"batch-{batch}"
            case_dir.mkdir(parents=True, exist_ok=True)
            # 每个Batch使用独立服务进程，避免其他Batch保存的Prefix Cache
            # 占用GPU内存或改变当前case的缓存淘汰状态。服务启动、模型加载
            # 和关闭均位于Nsight采集区间之外。
            server = launch_profiled_server(
                args, backend, batch, case_dir)
            prompt = prompts[batch]
            cache_warmup_prompt = make_cache_warmup_prompt(
                prompt, args.prefix_cache_padding_tokens)
            if (len(cache_warmup_prompt) + args.warmup_output_tokens >
                    args.max_model_len):
                raise ValueError(
                    "加长后的Prefix Cache Warmup超过max-model-len："
                    f"{len(cache_warmup_prompt)} + "
                    f"{args.warmup_output_tokens} > {args.max_model_len}")
            # 先用单请求把加长Prompt写入Prefix Cache，避免目标Batch
            # 同时提交多个尚未缓存的长Prefill。目标Batch预热继续
            # 使用同一加长Prompt，只更新同一条历史缓存，不保留第二份
            # 正式Prompt KV Cache。该预热覆盖目标Batch、CUDA Graph和缓存
            # 命中路径，且在Nsight采集前完成。正式Prompt是加长Prompt
            # 的完整前缀；FastLLM会按主流程退掉最后一个Cache页以重新
            # 产生首Token logits，该Prefill尾部随后由GPU Trace边界裁掉。
            run_decode_batch(
                server, cache_warmup_prompt,
                args.warmup_output_tokens, 1, args,
                "prefix_cache_seed")
            run_decode_batch(
                server, cache_warmup_prompt,
                args.warmup_output_tokens, batch, args,
                "warmup")
            output_base = case_dir / "decode"
            start_collection(args, server["session"], output_base)
            wait_collection_ready(args, server["session"])
            request_result = run_decode_batch(
                server, prompt, args.output_tokens, batch, args,
                "decode")
            report = stop_collection(args, server["session"], output_base)
            cache_ratio = request_result["cache_hit_ratio"]
            minimum_cache_ratio = max(
                0.0, (len(prompt) - args.prefix_cache_padding_tokens) /
                len(prompt))
            if (backend == "fastllm" and (
                    cache_ratio is None or
                    cache_ratio + 1e-9 < minimum_cache_ratio)):
                actual = "未报告" if cache_ratio is None else f"{cache_ratio:.1%}"
                raise RuntimeError(
                    "FastLLM Decode Nsight采集的Prefix Cache命中不足一个"
                    "预期尾页，实际为"
                    f"{actual}，最低要求{minimum_cache_ratio:.1%}；"
                    "请检查Cache页大小和Warmup Prompt。")
            summary = summarize_report(args, report, output_base)
            if args.keep_nsys_reports:
                summary["nsys_artifacts_retained"] = True
                summary["deleted_nsys_artifact_count"] = 0
            else:
                cleanup_nsys_artifacts(case_dir, output_base, summary)
            # 第一个输出Token属于Prefill结束点；之后每个Token才对应一次稳定
            # Decode迭代。按decode步数归一化后，才能比较不同batch下每步的
            # kernel数量与GPU时间，而不会把请求长度误认为算子变慢。
            decode_steps = max(args.output_tokens - 1, 1)
            request_tpots_ms = []
            request_e2els_ms = []
            for request in request_result["requests"]:
                token_times = request["token_times"]
                request_e2els_ms.append(
                    (request["end_time"] - request["start_time"]) * 1000)
                if len(token_times) > 1:
                    request_tpots_ms.append(
                        (request["end_time"] - token_times[0]) * 1000 /
                        (len(token_times) - 1))
            summary["decode_steps"] = decode_steps
            summary["kernel_ms_per_decode_step"] = (
                summary["kernel_total_ms"] / decode_steps)
            summary["kernel_instances_per_decode_step"] = (
                summary["kernel_instances"] / decode_steps)
            summary["memory_ms_per_decode_step"] = (
                summary["memory_total_ms"] / decode_steps)
            for value in summary["categories"].values():
                value["total_ms_per_decode_step"] = (
                    value["total_ms"] / decode_steps)
                value["instances_per_decode_step"] = (
                    value["instances"] / decode_steps)
            timeline = summary["timeline"]
            timeline["span_ms_per_decode_step"] = (
                timeline["span_ms"] / decode_steps)
            timeline["active_ms_per_decode_step"] = (
                timeline["active_ms"] / decode_steps)
            timeline["idle_ms_per_decode_step"] = (
                timeline["idle_ms"] / decode_steps)
            row = {
                "result": "PASS",
                "backend": backend,
                "batch": batch,
                "prompt_tokens": len(prompt),
                "warmup_prompt_tokens": len(cache_warmup_prompt),
                "output_tokens_per_request": args.output_tokens,
                "request_wall_ms": request_result["wall_s"] * 1000,
                "tpot_ms": (None if request_result["tpot_s"] is None else
                            request_result["tpot_s"] * 1000),
                "itl_ms": (None if request_result["itl_s"] is None else
                           request_result["itl_s"] * 1000),
                "request_tpot_p50_ms": percentile(request_tpots_ms, 50),
                "request_tpot_p95_ms": percentile(request_tpots_ms, 95),
                "request_e2el_p50_ms": percentile(request_e2els_ms, 50),
                "request_e2el_p95_ms": percentile(request_e2els_ms, 95),
                "output_tok_s": request_result["output_tok_s"],
                "cache_hit_ratio": cache_ratio,
                "minimum_cache_hit_ratio": minimum_cache_ratio,
                "cache_status": ("reported" if cache_ratio is not None else
                                 "unreported"),
                "nsys": summary,
            }
            if backend == "fastllm":
                selected_attention_backends = read_attention_backend_selection(
                    server["log_path"])
                backend_confirmed = attention_backend_confirmed(
                    args.flm_attention_backend, selected_attention_backends)
                if (args.flm_attention_backend_strict and
                        not backend_confirmed):
                    raise RuntimeError(
                        "FastLLM请求的Attention Backend未在Nsight服务日志中确认："
                        f"requested={args.flm_attention_backend}, "
                        f"actual={selected_attention_backends}")
                selected_kv_layouts = read_kv_cache_layout_selection(
                    server["log_path"])
                layout_confirmed = kv_cache_layout_confirmed(
                    args.flm_kv_cache_layout, selected_kv_layouts)
                if not layout_confirmed:
                    raise RuntimeError(
                        "FastLLM请求的KV Cache布局未在Nsight服务日志中确认："
                        f"requested={args.flm_kv_cache_layout}, "
                        f"actual={selected_kv_layouts}")
                row["requested_attention_backend"] = (
                    args.flm_attention_backend)
                row["actual_attention_backends"] = selected_attention_backends
                row["attention_backend_confirmed"] = (
                    backend_confirmed)
                row["requested_kv_cache_layout"] = args.flm_kv_cache_layout
                row["actual_kv_cache_layouts"] = selected_kv_layouts
                row["kv_cache_layout_confirmed"] = layout_confirmed
            cpu_trace = summary["cpu_trace"]
            if cpu_trace is not None:
                cpu_trace["launch_api_sum_ms_per_decode_step"] = (
                    cpu_trace["launch_api"]["sum_ms"] / decode_steps)
                cpu_trace["launch_api_union_ms_per_decode_step"] = (
                    cpu_trace["launch_api"]["union_ms"] / decode_steps)
                cpu_trace["cuda_api_sum_ms_per_decode_step"] = (
                    cpu_trace["cuda_api"]["sum_ms"] / decode_steps)
                cpu_trace["cuda_api_union_ms_per_decode_step"] = (
                    cpu_trace["cuda_api"]["union_ms"] / decode_steps)
                # 这是端到端TPOT与GPU事件跨度之差，不等同于纯CPU执行时间；
                # 它还包含请求排队、同步、线程唤醒和HTTP返回等未被GPU覆盖的时间。
                cpu_trace["tpot_minus_gpu_span_ms_per_step"] = (
                    None if row["tpot_ms"] is None else
                    row["tpot_ms"] - timeline["span_ms_per_decode_step"])
                cpu_trace["non_cuda_host_ms_per_step"] = (
                    None if cpu_trace["tpot_minus_gpu_span_ms_per_step"] is None
                    else cpu_trace["tpot_minus_gpu_span_ms_per_step"] -
                    cpu_trace["cuda_api_union_ms_per_decode_step"])
            (case_dir / "result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "backend": backend, "batch": batch,
                "tpot_ms": row["tpot_ms"],
                "output_tok_s": row["output_tok_s"],
                "kernel_total_ms": summary["kernel_total_ms"],
                "kernel_instances": summary["kernel_instances"],
            }, ensure_ascii=False), flush=True)
            shutdown_session(args, server["session"], server["launcher"])
            server["log_handle"].close()
            server = None
    finally:
        if server is not None:
            shutdown_session(args, server["session"], server["launcher"])
            server["log_handle"].close()
        if not args.keep_nsys_reports:
            for case_dir in backend_dir.glob("batch-*"):
                cleanup_nsys_artifacts(case_dir, case_dir / "decode")
    return rows


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def make_excel_report(result_dir, rows, top_count, chart_paths,
                      kernel_matches, kernel_annotations):
    """把Decode Nsight结构化结果写入便于筛选分析的Excel工作簿。"""
    sheets = []
    sheets.append(("说明", ["项目", "内容"], [
        ["报告", "FastLLM / vLLM Decode Nsight Systems对比"],
        ["计时范围", "GPU Trace中首Token之后的稳定Decode阶段"],
        ["Decode步数", "第一个输出Token属于Prefill，按output_tokens - 1归一化"],
        ["注意", "Nsight会扰动绝对延迟，正式TPOT以非Profiler性能脚本为准"],
        ["原始文件", "默认按Batch汇总后删除；使用--keep-nsys-reports显式保留"],
        ["类别限制", "同名GEMM无法仅凭符号区分o_proj与MLP Linear"],
    ]))

    engine_headers = [
        "状态", "Batch", "后端", "请求KV Cache布局", "实际KV Cache布局",
        "KV Cache布局确认", "请求Attention后端", "实际Attention后端",
        "Attention后端确认", "平均TPOT(ms)", "请求TPOT P50(ms)",
        "请求TPOT P95(ms)", "ITL(ms)", "请求E2EL P50(ms)",
        "请求E2EL P95(ms)", "Batch Wall(ms)", "Output(tok/s)",
        "Cache命中率",
    ]
    engine_rows = []
    for row in rows:
        engine_rows.append([
            row["result"], row["batch"], row["backend"],
            row.get("requested_kv_cache_layout", "n/a"),
            ",".join(row.get("actual_kv_cache_layouts", [])) or "n/a",
            row.get("kv_cache_layout_confirmed", "n/a"),
            row.get("requested_attention_backend", "n/a"),
            ",".join(row.get("actual_attention_backends", [])) or "n/a",
            row.get("attention_backend_confirmed", "n/a"), row["tpot_ms"],
            row["request_tpot_p50_ms"], row["request_tpot_p95_ms"],
            row["itl_ms"], row["request_e2el_p50_ms"],
            row["request_e2el_p95_ms"], row["request_wall_ms"],
            row["output_tok_s"],
            None if row["cache_status"] == "unreported" else row["cache_hit_ratio"],
        ])
    sheets.append(("引擎指标", engine_headers, engine_rows))

    timeline_headers = [
        "Batch", "后端", "Decode步数", "边界前排除事件", "Kernel(ms/步)",
        "Kernel数/步", "MemOp(ms/步)", "GPU跨度(ms/步)",
        "GPU活跃(ms/步)", "GPU内部空闲(ms/步)", "空闲占比",
    ]
    timeline_rows = []
    for row in rows:
        nsys = row["nsys"]
        timeline = nsys["timeline"]
        timeline_rows.append([
            row["batch"], row["backend"], nsys["decode_steps"],
            nsys["stable_decode_filter"]["excluded_event_count"],
            nsys["kernel_ms_per_decode_step"],
            nsys["kernel_instances_per_decode_step"],
            nsys["memory_ms_per_decode_step"],
            timeline["span_ms_per_decode_step"],
            timeline["active_ms_per_decode_step"],
            timeline["idle_ms_per_decode_step"], timeline["idle_ratio"],
        ])
    sheets.append(("GPU时间线", timeline_headers, timeline_rows))

    cpu_rows = [row for row in rows if row["nsys"].get("cpu_trace")]
    if cpu_rows:
        cpu_headers = [
            "Batch", "后端", "TPOT(ms)", "GPU跨度(ms/步)",
            "TPOT-GPU跨度(ms/步)", "CUDA API累计(ms/步)",
            "CUDA API去重墙钟(ms/步)", "非CUDA主机剩余(ms/步)",
            "Kernel提交API累计(ms/步)",
            "Kernel提交API去重墙钟(ms/步)", "Kernel执行数/步",
            "Kernel提交API数/步", "提交CPU线程数", "提交API P50(us)",
            "提交API P95(us)",
            "GPU排队P50(us)", "GPU排队P95(us)",
            "提交到Kernel P50(us)", "提交到Kernel P95(us)",
        ]
        cpu_sheet_rows = []
        for row in cpu_rows:
            cpu = row["nsys"]["cpu_trace"]
            steps = row["nsys"]["decode_steps"]
            cpu_sheet_rows.append([
                row["batch"], row["backend"], row["tpot_ms"],
                row["nsys"]["timeline"]["span_ms_per_decode_step"],
                cpu["tpot_minus_gpu_span_ms_per_step"],
                cpu["cuda_api_sum_ms_per_decode_step"],
                cpu["cuda_api_union_ms_per_decode_step"],
                cpu["non_cuda_host_ms_per_step"],
                cpu["launch_api_sum_ms_per_decode_step"],
                cpu["launch_api_union_ms_per_decode_step"],
                cpu["kernel_execution_count"] / steps,
                cpu["launch_api_count"] / steps, cpu["launcher_threads"],
                cpu["launch_api"]["median_us"],
                cpu["launch_api"]["p95_us"],
                cpu["gpu_queue"]["median_us"],
                cpu["gpu_queue"]["p95_us"],
                cpu["launch_to_kernel"]["median_us"],
                cpu["launch_to_kernel"]["p95_us"],
            ])
        sheets.append(("CPU调度汇总", cpu_headers, cpu_sheet_rows))
        api_rows = []
        for row in cpu_rows:
            cpu = row["nsys"]["cpu_trace"]
            steps = row["nsys"]["decode_steps"]
            for category, value in cpu["cuda_api"]["categories"].items():
                api_rows.append([
                    row["batch"], row["backend"], category,
                    value["total_ms"], value["total_ms"] / steps,
                    value["calls"], value["calls"] / steps,
                ])
        sheets.append(("CUDA API分类", [
            "Batch", "后端", "类别", "CPU线程累计时间(ms)",
            "累计时间(ms/步)", "调用数", "调用数/步"], api_rows))
        top_api_rows = []
        for row in cpu_rows:
            for rank, item in enumerate(
                    row["nsys"]["cpu_trace"]["cuda_api"]["top"][:20], 1):
                top_api_rows.append([
                    row["batch"], row["backend"], rank, item["name"],
                    item["calls"], item["total_ms"], item["average_us"],
                ])
        sheets.append(("Top CUDA APIs", [
            "Batch", "后端", "排名", "CUDA API", "调用数",
            "CPU线程累计时间(ms)", "平均(us)"], top_api_rows))
        osrt_rows = []
        for row in cpu_rows:
            for rank, item in enumerate(
                    row["nsys"]["cpu_trace"]["top_os_runtime"], 1):
                osrt_rows.append([
                    row["batch"], row["backend"], rank, item["name"],
                    item["instances"], item["total_ms"], item["average_us"],
                    "完整采集请求，未裁剪到稳定Decode窗口",
                ])
        if osrt_rows:
            sheets.append(("OS Runtime热点", [
                "Batch", "后端", "排名", "OS Runtime", "调用数",
                "CPU线程累计时间(ms)", "平均(us)", "统计范围"], osrt_rows))

    by_case = {(row["batch"], row["backend"]): row for row in rows}
    paired_batches = sorted({
        row["batch"] for row in rows
        if (row["batch"], "fastllm") in by_case and
        (row["batch"], "vllm") in by_case
    })
    ratio_rows = []
    for batch in paired_batches:
        fast = by_case[(batch, "fastllm")]
        vllm = by_case[(batch, "vllm")]

        def ratio(left, right):
            return None if left is None or right in (None, 0) else left / right

        ratio_rows.append([
            batch, ratio(fast["tpot_ms"], vllm["tpot_ms"]),
            ratio(fast["request_wall_ms"], vllm["request_wall_ms"]),
            ratio(fast["output_tok_s"], vllm["output_tok_s"]),
            ratio(fast["nsys"]["kernel_ms_per_decode_step"],
                  vllm["nsys"]["kernel_ms_per_decode_step"]),
            ratio(fast["nsys"]["timeline"]["idle_ms_per_decode_step"],
                  vllm["nsys"]["timeline"]["idle_ms_per_decode_step"]),
        ])
    if ratio_rows:
        sheets.append(("FastLLM-vLLM比值", [
            "Batch", "TPOT比", "Batch Wall比", "Output吞吐比",
            "Kernel时间/步比", "GPU内部空闲/步比"], ratio_rows))

    cpu_joint_rows = []
    for batch in paired_batches:
        fast = by_case[(batch, "fastllm")]
        vllm = by_case[(batch, "vllm")]
        fast_cpu = fast["nsys"].get("cpu_trace")
        vllm_cpu = vllm["nsys"].get("cpu_trace")
        if fast_cpu is None or vllm_cpu is None:
            continue
        fast_gpu = fast["nsys"]["timeline"]["span_ms_per_decode_step"]
        vllm_gpu = vllm["nsys"]["timeline"]["span_ms_per_decode_step"]
        cpu_joint_rows.append([
            batch,
            fast["tpot_ms"], vllm["tpot_ms"],
            fast["tpot_ms"] - vllm["tpot_ms"],
            fast_gpu, vllm_gpu, fast_gpu - vllm_gpu,
            fast_cpu["tpot_minus_gpu_span_ms_per_step"],
            vllm_cpu["tpot_minus_gpu_span_ms_per_step"],
            fast_cpu["tpot_minus_gpu_span_ms_per_step"] -
            vllm_cpu["tpot_minus_gpu_span_ms_per_step"],
            fast_cpu["non_cuda_host_ms_per_step"],
            vllm_cpu["non_cuda_host_ms_per_step"],
            fast_cpu["non_cuda_host_ms_per_step"] -
            vllm_cpu["non_cuda_host_ms_per_step"],
            fast_cpu["launch_api_union_ms_per_decode_step"],
            vllm_cpu["launch_api_union_ms_per_decode_step"],
            fast_cpu["launch_api_union_ms_per_decode_step"] -
            vllm_cpu["launch_api_union_ms_per_decode_step"],
            fast_cpu["gpu_queue"]["p95_us"],
            vllm_cpu["gpu_queue"]["p95_us"],
        ])
    if cpu_joint_rows:
        sheets.append(("CPU-GPU联合分析", [
            "Batch", "FastLLM TPOT(ms)", "vLLM TPOT(ms)", "TPOT差(ms)",
            "FastLLM GPU跨度(ms/步)", "vLLM GPU跨度(ms/步)",
            "GPU跨度差(ms/步)", "FastLLM未解释(ms/步)",
            "vLLM未解释(ms/步)", "未解释时间差(ms/步)",
            "FastLLM非CUDA主机剩余(ms/步)",
            "vLLM非CUDA主机剩余(ms/步)", "非CUDA主机剩余差(ms/步)",
            "FastLLM提交墙钟(ms/步)", "vLLM提交墙钟(ms/步)",
            "提交墙钟差(ms/步)", "FastLLM GPU排队P95(us)",
            "vLLM GPU排队P95(us)"], cpu_joint_rows))

    category_rows = []
    for row in rows:
        for category, value in sorted(
                row["nsys"]["categories"].items(),
                key=lambda item: item[1]["total_ms"], reverse=True):
            category_rows.append([
                row["batch"], row["backend"], category, value["total_ms"],
                value["total_ms_per_decode_step"], value["instances"],
                value["instances_per_decode_step"],
            ])
    sheets.append(("算子类别", [
        "Batch", "后端", "类别", "GPU总时间(ms)", "GPU时间(ms/步)",
        "Kernel总数", "Kernel数/步"], category_rows))

    compare_rows = []
    for batch in paired_batches:
        fast = by_case[(batch, "fastllm")]
        vllm = by_case[(batch, "vllm")]
        fast_categories = fast["nsys"]["categories"]
        vllm_categories = vllm["nsys"]["categories"]
        for category in ordered_categories(set(fast_categories) | set(vllm_categories)):
            fast_value = fast_categories.get(
                category, {"total_ms": 0.0, "instances": 0,
                           "total_ms_per_decode_step": 0.0})
            vllm_value = vllm_categories.get(
                category, {"total_ms": 0.0, "instances": 0,
                           "total_ms_per_decode_step": 0.0})
            compare_rows.append([
                batch, category, fast_value["total_ms"], vllm_value["total_ms"],
                fast_value["total_ms"] - vllm_value["total_ms"],
                fast_value["total_ms_per_decode_step"],
                vllm_value["total_ms_per_decode_step"],
                fast_value["instances"], vllm_value["instances"],
                fast_value["instances"] - vllm_value["instances"],
            ])
        fast_timeline = fast["nsys"]["timeline"]
        vllm_timeline = vllm["nsys"]["timeline"]
        compare_rows.append([
            batch, "GPU Internal Idle", fast_timeline["idle_ms"],
            vllm_timeline["idle_ms"],
            fast_timeline["idle_ms"] - vllm_timeline["idle_ms"],
            fast_timeline["idle_ms_per_decode_step"],
            vllm_timeline["idle_ms_per_decode_step"], None, None, None,
        ])
    if compare_rows:
        sheets.append(("算子并排对比", [
            "Batch", "类别", "FastLLM总耗时(ms)", "vLLM总耗时(ms)",
            "时间差(ms)", "FastLLM(ms/步)", "vLLM(ms/步)",
            "FastLLM启动数", "vLLM启动数", "启动数差"], compare_rows))

    match_rows = []
    for match in kernel_matches:
        fast = match["fastllm"]
        vllm = match["vllm"]
        match_rows.append([
            match["batch"], match["match_id"], match["level"],
            match["category"], match["label"], match["signature"],
            fast["name"], vllm["name"], fast["instances"],
            vllm["instances"], match["common_instances"],
            match["fastllm_extra_instances"], match["vllm_extra_instances"],
            match["instance_ratio"], fast["average_us"],
            vllm["average_us"], match["average_us_ratio"],
            fast["total_ms"], vllm["total_ms"], match["total_ms_ratio"],
        ])
    sheets.append(("Kernel配对对比", [
        "Batch", "匹配ID", "匹配等级", "类别", "标准Kernel", "标准签名",
        "FastLLM Kernel", "vLLM Kernel", "FastLLM调用数", "vLLM调用数",
        "共同调用数", "FastLLM额外调用", "vLLM额外调用", "调用数比",
        "FastLLM平均(us)", "vLLM平均(us)", "平均耗时比",
        "FastLLM总时间(ms)", "vLLM总时间(ms)", "总时间比"], match_rows))

    top_rows = []
    for row in rows:
        for rank, item in enumerate(row["nsys"]["top_kernels"][:top_count], 1):
            annotation = kernel_annotations.get(
                (row["batch"], row["backend"], item["name"]))
            top_rows.append([
                row["batch"], row["backend"], rank,
                "" if annotation is None else annotation[0],
                "unmatched" if annotation is None else annotation[1],
                "" if annotation is None else annotation[2],
                kernel_category(item["name"]), item["name"], item["instances"],
                item["total_ms"], item["average_us"],
            ])
    sheets.append(("Top Kernels", [
        "Batch", "后端", "排名", "匹配ID", "匹配等级", "标准Kernel", "类别",
        "Kernel", "调用次数", "总时间(ms)", "平均(us)"], top_rows))

    path = result_dir / "decode-nsys-compare.xlsx"
    write_xlsx(path, sheets, image_sheets=[("Batch趋势图", chart_paths)])
    return path


def make_report(result_dir, rows, top_count):
    lines = [
        "# FastLLM / vLLM Decode Nsight Systems对比", "",
        "> 每个Batch先使用加长Prompt完成warmup，使正式Prompt成为完整缓存前缀；",
        "> Nsight进入Collection状态并稳定后才发送BEST模式HTTP请求。FastLLM会",
        "> 按主流程退掉最后一个Cache页以重新产生首Token logits；报告依据GPU Trace",
        "> 裁掉首Token以前的Prefill及采样，仅汇总后续稳定Decode事件。Nsight",
        "> 会扰动绝对延迟，正式TPOT仍应以非Profiler性能脚本为准。",
        "> GPU时间线空闲只统计首个与最后一个GPU事件之间的内部空隙；",
        "> 第一个输出Token属于Prefill，逐步指标按`output_tokens - 1`归一化。", "",
        "> `.nsys-rep`、SQLite和导出CSV默认在每个Batch汇总后删除；",
        "> 只有指定`--keep-nsys-reports`才会保留。", "",
        "## 引擎指标", "",
        "| 状态 | Batch | 后端 | 请求KV Cache布局 | 实际KV Cache布局 | 布局确认 | "
        "请求Attention后端 | 实际Attention后端 | 后端确认 | "
        "平均TPOT(ms) | 请求TPOT P50(ms) | 请求TPOT P95(ms) | "
        "ITL(ms) | 请求E2EL P50(ms) | 请求E2EL P95(ms) | Batch Wall(ms) | "
        "Output(tok/s) | Cache命中 |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |",
    ]
    for row in rows:
        cache = row["cache_hit_ratio"]
        cache_text = ("未报告" if row["cache_status"] == "unreported" else
                      f"{cache * 100:.1f}%")
        lines.append(
            f"| {row['result']} | {row['batch']} | {row['backend']} | "
            f"{row.get('requested_kv_cache_layout', 'n/a')} | "
            f"{','.join(row.get('actual_kv_cache_layouts', [])) or 'n/a'} | "
            f"{row.get('kv_cache_layout_confirmed', 'n/a')} | "
            f"{row.get('requested_attention_backend', 'n/a')} | "
            f"{','.join(row.get('actual_attention_backends', [])) or 'n/a'} | "
            f"{row.get('attention_backend_confirmed', 'n/a')} | "
            f"{fmt(row['tpot_ms'])} | "
            f"{fmt(row['request_tpot_p50_ms'])} | "
            f"{fmt(row['request_tpot_p95_ms'])} | {fmt(row['itl_ms'])} | "
            f"{fmt(row['request_e2el_p50_ms'])} | "
            f"{fmt(row['request_e2el_p95_ms'])} | "
            f"{fmt(row['request_wall_ms'])} | {fmt(row['output_tok_s'], 2)} | "
            f"{cache_text} |")

    lines.extend([
        "", "## GPU时间线", "",
        "| Batch | 后端 | Decode步数 | 边界前排除事件 | Kernel(ms/步) | Kernel数/步 | "
        "MemOp(ms/步) | GPU跨度(ms/步) | GPU活跃(ms/步) | "
        "GPU内部空闲(ms/步) | 空闲占比 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        nsys = row["nsys"]
        timeline = nsys["timeline"]
        idle_ratio = timeline["idle_ratio"]
        lines.append(
            f"| {row['batch']} | {row['backend']} | {nsys['decode_steps']} | "
            f"{nsys['stable_decode_filter']['excluded_event_count']} | "
            f"{fmt(nsys['kernel_ms_per_decode_step'])} | "
            f"{fmt(nsys['kernel_instances_per_decode_step'], 2)} | "
            f"{fmt(nsys['memory_ms_per_decode_step'])} | "
            f"{fmt(timeline['span_ms_per_decode_step'])} | "
            f"{fmt(timeline['active_ms_per_decode_step'])} | "
            f"{fmt(timeline['idle_ms_per_decode_step'])} | "
            f"{'n/a' if idle_ratio is None else f'{idle_ratio * 100:.1f}%'} |")

    cpu_rows = [row for row in rows if row["nsys"].get("cpu_trace")]
    if cpu_rows:
        lines.extend([
            "", "## CPU调度汇总", "",
            "> `TPOT-GPU跨度`是CPU调度、同步、请求排队和响应等未被GPU事件",
            "> 覆盖的综合差值，不等同于纯CPU执行时间。CUDA API累计时间会叠加",
            "> 多线程重叠，去重墙钟时间才表示时间轴覆盖长度。", "",
            "| Batch | 后端 | TPOT(ms) | GPU跨度(ms/步) | 未解释(ms/步) | "
            "CUDA API墙钟(ms/步) | 非CUDA主机剩余(ms/步) | "
            "提交墙钟(ms/步) | Kernel执行数/步 | "
            "提交API数/步 | "
            "提交API P95(us) | GPU排队P95(us) | 提交到Kernel P95(us) |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |",
        ])
        for row in cpu_rows:
            cpu = row["nsys"]["cpu_trace"]
            steps = row["nsys"]["decode_steps"]
            lines.append(
                f"| {row['batch']} | {row['backend']} | {fmt(row['tpot_ms'])} | "
                f"{fmt(row['nsys']['timeline']['span_ms_per_decode_step'])} | "
                f"{fmt(cpu['tpot_minus_gpu_span_ms_per_step'])} | "
                f"{fmt(cpu['cuda_api_union_ms_per_decode_step'])} | "
                f"{fmt(cpu['non_cuda_host_ms_per_step'])} | "
                f"{fmt(cpu['launch_api_union_ms_per_decode_step'])} | "
                f"{cpu['kernel_execution_count'] / steps:.2f} | "
                f"{cpu['launch_api_count'] / steps:.2f} | "
                f"{fmt(cpu['launch_api']['p95_us'])} | "
                f"{fmt(cpu['gpu_queue']['p95_us'])} | "
                f"{fmt(cpu['launch_to_kernel']['p95_us'])} |")

    by_case = {(row["batch"], row["backend"]): row for row in rows}
    paired_batches = sorted({
        row["batch"] for row in rows
        if (row["batch"], "fastllm") in by_case and
        (row["batch"], "vllm") in by_case
    })
    kernel_matches, kernel_annotations = build_kernel_matches(rows)
    if paired_batches:
        lines.extend([
            "", "## FastLLM / vLLM比值", "",
            "> 延迟比小于1表示FastLLM更快；吞吐比大于1表示FastLLM更快。", "",
            "| Batch | TPOT比 | Batch Wall比 | Output吞吐比 | "
            "Kernel时间/步比 | GPU内部空闲/步比 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for batch in paired_batches:
            fast = by_case[(batch, "fastllm")]
            vllm = by_case[(batch, "vllm")]

            def ratio(left, right):
                if left is None or right is None or right == 0:
                    return None
                return left / right

            def ratio_text(left, right):
                value = ratio(left, right)
                return "n/a" if value is None else f"{value:.3f}x"

            lines.append(
                f"| {batch} | {ratio_text(fast['tpot_ms'], vllm['tpot_ms'])} | "
                f"{ratio_text(fast['request_wall_ms'], vllm['request_wall_ms'])} | "
                f"{ratio_text(fast['output_tok_s'], vllm['output_tok_s'])} | "
                f"{ratio_text(fast['nsys']['kernel_ms_per_decode_step'], vllm['nsys']['kernel_ms_per_decode_step'])} | "
                f"{ratio_text(fast['nsys']['timeline']['idle_ms_per_decode_step'], vllm['nsys']['timeline']['idle_ms_per_decode_step'])} |")

        if all(by_case[(batch, backend)]["nsys"].get("cpu_trace")
               for batch in paired_batches for backend in BACKENDS):
            lines.extend([
                "", "## CPU-GPU联合分析", "",
                "> 差值均为`FastLLM-vLLM`；负数表示FastLLM耗时更少。", "",
                "| Batch | TPOT差(ms) | GPU跨度差(ms/步) | "
                "未解释时间差(ms/步) | 非CUDA主机剩余差(ms/步) | "
                "提交墙钟差(ms/步) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for batch in paired_batches:
                fast = by_case[(batch, "fastllm")]
                vllm = by_case[(batch, "vllm")]
                fast_cpu = fast["nsys"]["cpu_trace"]
                vllm_cpu = vllm["nsys"]["cpu_trace"]
                lines.append(
                    f"| {batch} | {fast['tpot_ms'] - vllm['tpot_ms']:+.3f} | "
                    f"{fast['nsys']['timeline']['span_ms_per_decode_step'] - vllm['nsys']['timeline']['span_ms_per_decode_step']:+.3f} | "
                    f"{fast_cpu['tpot_minus_gpu_span_ms_per_step'] - vllm_cpu['tpot_minus_gpu_span_ms_per_step']:+.3f} | "
                    f"{fast_cpu['non_cuda_host_ms_per_step'] - vllm_cpu['non_cuda_host_ms_per_step']:+.3f} | "
                    f"{fast_cpu['launch_api_union_ms_per_decode_step'] - vllm_cpu['launch_api_union_ms_per_decode_step']:+.3f} |")

    lines.extend(["", "## 算子类别", "",
                  "> 类别根据Nsight Kernel符号归纳；同名GEMM不能仅凭符号可靠区分",
                  "> Attention `o_proj`与MLP Linear，无法确认的融合Kernel归入",
                  "> `Fused/Unresolved`，不把推测结果伪装成精确模型阶段。", "",
                  "| Batch | 后端 | 类别 | GPU总时间(ms) | GPU时间(ms/步) | "
                  "Kernel总数 | Kernel数/步 |",
                  "| ---: | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        for category, value in sorted(
                row["nsys"]["categories"].items(),
                key=lambda item: item[1]["total_ms"], reverse=True):
            lines.append(
                f"| {row['batch']} | {row['backend']} | {category} | "
                f"{value['total_ms']:.3f} | "
                f"{value['total_ms_per_decode_step']:.3f} | "
                f"{value['instances']} | "
                f"{value['instances_per_decode_step']:.2f} |")

    if paired_batches:
        lines.extend([
            "", "## 算子类别并排对比", "",
            "> 时间差=`FastLLM-vLLM`，负数表示FastLLM在该类别耗时更少；",
            "> GPU内部空闲不是Kernel，启动次数以`—`表示。绝对热点看总时间，",
            "> 引擎差异来源看时间差，二者不能混为一谈。", "",
            "| Batch | 类别 | FastLLM总耗时(ms) | vLLM总耗时(ms) | "
            "时间差(ms) | FastLLM(ms/步) | vLLM(ms/步) | "
            "FastLLM启动数 | vLLM启动数 | 启动数差 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for batch in paired_batches:
            fast = by_case[(batch, "fastllm")]
            vllm = by_case[(batch, "vllm")]
            fast_categories = fast["nsys"]["categories"]
            vllm_categories = vllm["nsys"]["categories"]
            categories = ordered_categories(
                set(fast_categories) | set(vllm_categories))
            for category in categories:
                fast_value = fast_categories.get(
                    category, {"total_ms": 0.0, "instances": 0,
                               "total_ms_per_decode_step": 0.0})
                vllm_value = vllm_categories.get(
                    category, {"total_ms": 0.0, "instances": 0,
                               "total_ms_per_decode_step": 0.0})
                lines.append(
                    f"| {batch} | {category} | "
                    f"{fast_value['total_ms']:.3f} | "
                    f"{vllm_value['total_ms']:.3f} | "
                    f"{fast_value['total_ms'] - vllm_value['total_ms']:+.3f} | "
                    f"{fast_value['total_ms_per_decode_step']:.3f} | "
                    f"{vllm_value['total_ms_per_decode_step']:.3f} | "
                    f"{fast_value['instances']} | {vllm_value['instances']} | "
                    f"{fast_value['instances'] - vllm_value['instances']:+d} |")

            fast_idle = fast["nsys"]["timeline"]["idle_ms"]
            vllm_idle = vllm["nsys"]["timeline"]["idle_ms"]
            fast_idle_step = fast["nsys"]["timeline"][
                "idle_ms_per_decode_step"]
            vllm_idle_step = vllm["nsys"]["timeline"][
                "idle_ms_per_decode_step"]
            lines.append(
                f"| {batch} | GPU Internal Idle | {fast_idle:.3f} | "
                f"{vllm_idle:.3f} | {fast_idle - vllm_idle:+.3f} | "
                f"{fast_idle_step:.3f} | {vllm_idle_step:.3f} | — | — | — |")

    if kernel_matches:
        lines.extend([
            "", "## Kernel配对对比", "",
            "> structural表示CUTLASS数据类型、Tile、Cluster和调度结构一致；",
            "> semantic表示功能相同但实现边界不同；symbol只表示Nsight符号相同，",
            "> 不能替代SASS一致性验证。", "",
            "| Batch | ID | 等级 | 标准Kernel | FT调用 | vLLM调用 | 调用比 | "
            "FT平均(us) | vLLM平均(us) | 平均耗时比 |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for match in kernel_matches:
            fast = match["fastllm"]
            vllm = match["vllm"]
            lines.append(
                f"| {match['batch']} | {match['match_id']} | {match['level']} | "
                f"{match['label'].replace('|', '/')} | {fast['instances']} | "
                f"{vllm['instances']} | {match['instance_ratio']:.3f} | "
                f"{fast['average_us']:.3f} | {vllm['average_us']:.3f} | "
                f"{match['average_us_ratio']:.3f} |")

    for row in rows:
        lines.extend([
            "", f"## Batch={row['batch']} / {row['backend']} Top Kernels", "",
            "| 匹配ID | 匹配等级 | 类别 | Kernel | 调用次数 | 总时间(ms) | 平均(us) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ])
        for item in row["nsys"]["top_kernels"][:top_count]:
            name = item["name"].replace("|", "\\|")
            annotation = kernel_annotations.get(
                (row["batch"], row["backend"], item["name"]))
            lines.append(
                f"| {'' if annotation is None else annotation[0]} | "
                f"{'unmatched' if annotation is None else annotation[1]} | "
                f"{kernel_category(item['name'])} | `{name}` | {item['instances']} | "
                f"{item['total_ms']:.3f} | {item['average_us']:.3f} |")

    chart_rows = []
    for row in rows:
        nsys = row["nsys"]
        timeline = nsys["timeline"]
        cpu = nsys.get("cpu_trace") or {}
        launch_count = cpu.get("launch_api_count")
        decode_steps = nsys["decode_steps"]
        submit_count_per_step = (
            None if launch_count is None else launch_count / decode_steps)
        kernels_per_submission = (
            None if not launch_count else
            cpu["kernel_execution_count"] / launch_count)
        chart_rows.append({
            "batch": row["batch"],
            "backend": row["backend"],
            "tpot_ms": row["tpot_ms"],
            "output_tok_s": row["output_tok_s"],
            "kernel_ms_per_step": nsys["kernel_ms_per_decode_step"],
            "kernel_count_per_step": nsys["kernel_instances_per_decode_step"],
            "gpu_active_ms_per_step": timeline["active_ms_per_decode_step"],
            "gpu_idle_ms_per_step": timeline["idle_ms_per_decode_step"],
            "gpu_idle_pct": (None if timeline["idle_ratio"] is None else
                             timeline["idle_ratio"] * 100.0),
            "submit_ms_per_step": cpu.get("launch_api_union_ms_per_decode_step"),
            "submit_count_per_step": submit_count_per_step,
            "kernels_per_submission": kernels_per_submission,
        })

    def chart_ratio(left, right):
        return None if left is None or right in (None, 0) else left / right

    ratio_rows = []
    for batch in paired_batches:
        fast = by_case[(batch, "fastllm")]
        vllm = by_case[(batch, "vllm")]
        fast_cpu = fast["nsys"].get("cpu_trace")
        vllm_cpu = vllm["nsys"].get("cpu_trace")
        if fast_cpu is None or vllm_cpu is None:
            continue
        fast_steps = fast["nsys"]["decode_steps"]
        vllm_steps = vllm["nsys"]["decode_steps"]
        ratio_rows.append({
            "batch": batch,
            "backend": "ratio",
            "kernel_execution_ratio": chart_ratio(
                fast_cpu["kernel_execution_count"] / fast_steps,
                vllm_cpu["kernel_execution_count"] / vllm_steps),
            "kernel_time_ratio": chart_ratio(
                fast["nsys"]["kernel_ms_per_decode_step"],
                vllm["nsys"]["kernel_ms_per_decode_step"]),
            "submit_time_ratio": chart_ratio(
                fast_cpu["launch_api_union_ms_per_decode_step"],
                vllm_cpu["launch_api_union_ms_per_decode_step"]),
            "tpot_ratio": chart_ratio(fast["tpot_ms"], vllm["tpot_ms"]),
        })

    matched_kernel_rows = []
    for batch in paired_batches:
        batch_matches = [
            match for match in kernel_matches if match["batch"] == batch
        ]
        if not batch_matches:
            continue
        common_instances = sum(
            match["common_instances"] for match in batch_matches)
        fast_common_us = sum(
            match["fastllm"]["average_us"] * match["common_instances"]
            for match in batch_matches)
        vllm_common_us = sum(
            match["vllm"]["average_us"] * match["common_instances"]
            for match in batch_matches)
        fast_instances = sum(
            match["fastllm"]["instances"] for match in batch_matches)
        vllm_instances = sum(
            match["vllm"]["instances"] for match in batch_matches)
        fast_total_ms = sum(
            match["fastllm"]["total_ms"] for match in batch_matches)
        vllm_total_ms = sum(
            match["vllm"]["total_ms"] for match in batch_matches)
        steps = by_case[(batch, "fastllm")]["nsys"]["decode_steps"]
        matched_kernel_rows.append({
            "batch": batch,
            "backend": "ratio",
            "matched_average_us_ratio": chart_ratio(
                fast_common_us / common_instances,
                vllm_common_us / common_instances),
            "matched_instance_ratio": chart_ratio(
                fast_instances, vllm_instances),
            "matched_total_ms_ratio": chart_ratio(
                fast_total_ms, vllm_total_ms),
            "fastllm_extra_per_step": sum(
                match["fastllm_extra_instances"] for match in batch_matches
            ) / steps,
        })
    image_dir = result_dir / "images"
    chart_paths = [
        write_dashboard(
            image_dir / "decode-nsys-engine-trends.png", chart_rows, (
                ChartSpec("TPOT", "tpot_ms", "ms"),
                ChartSpec("OUTPUT THROUGHPUT", "output_tok_s", "tok/s"),
                ChartSpec("GPU ACTIVE", "gpu_active_ms_per_step", "ms/step"),
                ChartSpec("GPU IDLE", "gpu_idle_ms_per_step", "ms/step"),
            ), "DECODE NSYS ENGINE TRENDS"),
        write_dashboard(
            image_dir / "decode-nsys-bottleneck-trends.png", chart_rows, (
                ChartSpec("KERNEL TIME", "kernel_ms_per_step", "ms/step"),
                ChartSpec("KERNEL COUNT", "kernel_count_per_step", "count/step"),
                ChartSpec("GPU IDLE RATIO", "gpu_idle_pct", "%"),
                ChartSpec("SUBMIT API", "submit_ms_per_step", "ms/step"),
            ), "DECODE NSYS BOTTLENECK TRENDS"),
        write_dashboard(
            image_dir / "decode-nsys-submission-trends.png", chart_rows, (
                ChartSpec("KERNEL SUBMISSIONS", "submit_count_per_step", "count/step"),
                ChartSpec("KERNELS PER SUBMISSION", "kernels_per_submission", "ratio"),
                ChartSpec("SUBMIT API TIME", "submit_ms_per_step", "ms/step"),
                ChartSpec("GPU IDLE RATIO", "gpu_idle_pct", "%"),
            ), "DECODE NSYS CPU SUBMISSION TRENDS"),
    ]
    if ratio_rows:
        chart_paths.append(write_dashboard(
            image_dir / "decode-nsys-relative-ratio-trends.png", ratio_rows, (
                ChartSpec("KERNEL EXECUTION RATIO", "kernel_execution_ratio", "x", 1.0),
                ChartSpec("KERNEL TIME RATIO", "kernel_time_ratio", "x", 1.0),
                ChartSpec("SUBMIT API TIME RATIO", "submit_time_ratio", "x", 1.0),
                ChartSpec("TPOT RATIO", "tpot_ratio", "x", 1.0),
            ), "FASTLLM / VLLM RELATIVE TRENDS"))
    if matched_kernel_rows:
        chart_paths.append(write_dashboard(
            image_dir / "decode-nsys-matched-kernel-trends.png",
            matched_kernel_rows, (
                ChartSpec("MATCHED AVG LATENCY RATIO", "matched_average_us_ratio", "x", 1.0),
                ChartSpec("MATCHED CALL COUNT RATIO", "matched_instance_ratio", "x", 1.0),
                ChartSpec("MATCHED TOTAL TIME RATIO", "matched_total_ms_ratio", "x", 1.0),
                ChartSpec("FASTLLM EXTRA CALLS", "fastllm_extra_per_step", "count/step"),
            ), "MATCHED KERNEL TRENDS"))
    lines.extend(markdown_images(chart_paths))

    report = "\n".join(lines) + "\n"
    path = result_dir / "decode-nsys-compare.md"
    path.write_text(report, encoding="utf-8")
    (result_dir / "decode-nsys-compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "decode-nsys-kernel-matches.json").write_text(
        json.dumps(kernel_matches, ensure_ascii=False, indent=2),
        encoding="utf-8")
    excel_path = make_excel_report(
        result_dir, rows, top_count, chart_paths,
        kernel_matches, kernel_annotations)
    print(report)
    print(f"Summary: PASS ({len(rows)} Decode Nsight cases)")
    print(f"Markdown: {path}")
    print(f"Excel: {excel_path}")
    print(f"Kernel matches: {result_dir / 'decode-nsys-kernel-matches.json'}")
    print(f"Images: {image_dir}")


def main():
    args = parse_args()
    batches = parse_positive_csv(args.batch_sizes, "batch-sizes")
    backends = selected_backends(args.backends)
    if args.quantization == "a16" and "vllm" in backends:
        raise ValueError("a16用于FastLLM native/Machete矩阵，请指定--backends fastllm")
    for name in ("prompt_tokens", "output_tokens", "warmup_output_tokens",
                 "max_model_len", "startup_timeout", "request_timeout"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')}必须大于0")
    if args.prefix_cache_padding_tokens <= 0:
        raise ValueError("prefix-cache-padding-tokens必须大于0")
    if args.collection_ready_timeout <= 0:
        raise ValueError("collection-ready-timeout必须大于0")
    if args.collection_ready_delay < 0:
        raise ValueError("collection-ready-delay不能小于0")
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = {
        batch: make_prompt_tokens(tokenizer, args.prompt_tokens, batch)
        for batch in batches
    }
    (result_dir / "prompt-token-ids.json").write_text(json.dumps(
        {str(batch): value for batch, value in prompts.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for backend in backends:
        rows.extend(profile_backend(
            args, backend, batches, prompts, result_dir))
    rows.sort(key=lambda item: (item["batch"], BACKENDS.index(item["backend"])))
    make_report(result_dir, rows, args.top_kernels)


if __name__ == "__main__":
    main()
