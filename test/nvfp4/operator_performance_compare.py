#!/usr/bin/env python3
"""分进程运行vLLM NVFP4算子，并与FastLLM算子日志生成对比表。"""

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


REPO_DIR = Path(__file__).resolve().parents[2]
from xlsx_report import write_xlsx  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="NVFP4算子级FastLLM/vLLM性能对比")
    parser.add_argument("--fastllm-log-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--vllm-python", default=os.environ.get("NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--stage", choices=("orchestrate", "vllm"),
                        default="orchestrate", help=argparse.SUPPRESS)
    parser.add_argument("--cases", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def command_tokens(text):
    for line in text.splitlines():
        if line.startswith("COMMAND:"):
            return shlex.split(line.removeprefix("COMMAND:").strip())
    return []


def option(tokens, name, default="-"):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError):
        return default


def parameters(tokens):
    result = {}
    for index, token in enumerate(tokens[:-1]):
        if token == "--param" and "=" in tokens[index + 1]:
            key, value = tokens[index + 1].split("=", 1)
            result[key] = value
    return result


def parse_fastllm_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = command_tokens(text)
    params = parameters(tokens)
    op = option(tokens, "--op")
    latency_match = re.search(r"latency:\s+avg_ms=([0-9.eE+-]+)", text)
    if not latency_match:
        raise RuntimeError(f"FastLLM日志缺少latency: {path}")
    if op == "linear_nvfp4":
        m, n, k = int(params["batch"]), int(params["out"]), int(params["in"])
        kind = "dense"
    elif op == "nvfp4_swiglu_quant":
        m, n, k = int(params["rows"]), 0, int(params["hidden"])
        kind = "swiglu_quant"
    elif op == "mergemoe_fp8" and params.get("weight_type") == "nvfp4":
        m, n, k = int(params["batch"]), int(params["inter"]), int(params["hidden"])
        kind = "grouped_moe"
    else:
        return None
    return {
        "case": path.stem,
        "kind": kind,
        "op": op,
        "m": m,
        "n": n,
        "k": k,
        "topk": int(params.get("topk", "0")),
        "experts": int(params.get("experts", "0")),
        "dtype": params.get("input_type", "bf16"),
        "warmup": int(option(tokens, "--warmup", "0")),
        "iters": int(option(tokens, "--iters", "1")),
        "fastllm_latency_ms": float(latency_match.group(1)),
    }


def load_fastllm_cases(log_dir):
    cases = []
    for path in sorted(Path(log_dir).glob("op_*perf*.log")):
        case = parse_fastllm_log(path)
        if case is not None:
            cases.append(case)
    if not cases:
        raise RuntimeError(f"没有可对比的NVFP4算子日志: {log_dir}/op_*perf*.log")
    return cases


def theoretical_flops(case):
    if case["kind"] == "dense":
        return 2.0 * case["m"] * case["n"] * case["k"]
    if case["kind"] == "grouped_moe":
        return 6.0 * case["m"] * case["topk"] * case["n"] * case["k"]
    return 4.5 * case["m"] * case["k"]


def compute_tflops(case, latency_ms):
    return theoretical_flops(case) / (latency_ms / 1000.0) / 1e12


def torch_dtype(name, torch):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"不支持的输入类型: {name}")


def wall_time_benchmark(torch, runner, warmup, iters):
    """与FastLLM optest一致，在框架调用边界计时并在末尾同步。"""
    for _ in range(warmup):
        runner()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        runner()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - begin) * 1000.0 / max(iters, 1)


def make_dense_runner(case, torch, ops, float4_max, float8_max):
    dtype = torch_dtype(case["dtype"], torch)
    m, n, k = case["m"], case["n"], case["k"]
    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)
    a_global_scale = (
        float8_max * float4_max / torch.abs(a).max().to(torch.float32))
    b_global_scale = (
        float8_max * float4_max / torch.abs(b).max().to(torch.float32))
    b_fp4, b_scale = ops.scaled_fp4_quant(b, b_global_scale)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    def runner():
        a_fp4, a_scale = ops.scaled_fp4_quant(a, a_global_scale)
        return ops.cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale, b_scale, alpha, dtype)

    return runner


def make_swiglu_runner(case, torch, ops, float4_max, float8_max):
    dtype = torch_dtype(case["dtype"], torch)
    rows, hidden = case["m"], case["k"]
    torch.manual_seed(7)
    source = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)
    # 等价于vLLM SiluAndMul.forward_native，但不实例化CustomOp；后者要求
    # set_current_vllm_config上下文，而这里只需要计算固定的量化global scale。
    reference = torch.nn.functional.silu(source[..., :hidden]) * source[..., hidden:]
    global_scale = (
        float8_max * float4_max / torch.abs(reference).max().to(torch.float32))
    packed, scales = ops.scaled_fp4_quant(reference, global_scale)
    packed = torch.empty_like(packed)
    scales = torch.empty_like(scales)

    def runner():
        torch.ops._C.silu_and_mul_nvfp4_quant(
            packed, scales, source, global_scale)
        return packed

    return runner


def make_moe_config(case, torch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    experts = case["experts"]
    return FusedMoEConfig(
        num_experts=experts,
        experts_per_token=case["topk"],
        hidden_dim=case["k"],
        intermediate_size=case["n"],
        num_local_experts=experts,
        num_logical_experts=experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch_dtype(case["dtype"], torch),
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max(512, case["m"]),
    )


def quantize_moe_weight(weight, torch, ops, float4_max, float8_max):
    experts = weight.shape[0]
    quantized = []
    scales = []
    global_scales = []
    for expert in range(experts):
        global_scale = (
            float8_max * float4_max /
            torch.abs(weight[expert]).max().to(torch.float32))
        packed, block_scale = ops.scaled_fp4_quant(
            weight[expert], global_scale)
        quantized.append(packed)
        scales.append(block_scale)
        global_scales.append(global_scale)
    return (torch.stack(quantized), torch.stack(scales),
            torch.stack(global_scales))


def make_moe_runner(case, torch, ops, float4_max, float8_max):
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsFp4,
    )

    dtype = torch_dtype(case["dtype"], torch)
    m, n, k = case["m"], case["n"], case["k"]
    experts, topk = case["experts"], case["topk"]
    torch.manual_seed(7)
    hidden = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((experts, k, n), device="cuda", dtype=dtype) / 10
    w1_fp4, w1_scale, w1_global = quantize_moe_weight(
        w1, torch, ops, float4_max, float8_max)
    w2_fp4, w2_scale, w2_global = quantize_moe_weight(
        w2, torch, ops, float4_max, float8_max)
    topk_ids = (
        torch.arange(m * topk, device="cuda", dtype=torch.int32)
        .reshape(m, topk) % experts)
    topk_weights = torch.full(
        (m, topk), 1.0 / topk, device="cuda", dtype=torch.float32)
    activation_scales = torch.ones(
        (experts,), device="cuda", dtype=torch.float32)
    quant_config = nvfp4_moe_quant_config(
        a1_gscale=activation_scales,
        a2_gscale=activation_scales,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=1.0 / w1_global,
        g2_alphas=1.0 / w2_global,
    )
    moe_config = make_moe_config(case, torch)
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(pipeline_parallel_size=1))
    config_context = set_current_vllm_config(vllm_config)
    config_context.__enter__()
    kernel = mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        CutlassExpertsFp4(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    def runner():
        return kernel.apply(
            hidden_states=hidden,
            w1=w1_fp4,
            w2=w2_fp4,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=experts,
            activation=moe_config.activation,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

    # 保持vLLM配置上下文存活到runner测速结束。
    runner.config_context = config_context
    return runner


def run_vllm_stage(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    from vllm.scalar_type import scalar_types
    from vllm.v1.worker.workspace import init_workspace_manager

    if not current_platform.has_device_capability(100):
        raise RuntimeError("vLLM NVFP4算子需要SM100或更新GPU")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    init_workspace_manager(torch.device("cuda:0"))
    float4_max = scalar_types.float4_e2m1f.max()
    float8_max = torch.finfo(torch.float8_e4m3fn).max
    results = []
    for case in cases:
        runner = None
        try:
            if case["kind"] == "dense":
                runner = make_dense_runner(case, torch, ops, float4_max, float8_max)
            elif case["kind"] == "swiglu_quant":
                runner = make_swiglu_runner(case, torch, ops, float4_max, float8_max)
            elif case["kind"] == "grouped_moe":
                runner = make_moe_runner(case, torch, ops, float4_max, float8_max)
            else:
                raise ValueError(f"未知算子类型: {case['kind']}")
            latency_ms = wall_time_benchmark(
                torch, runner, case["warmup"], case["iters"])
            result = {
                "case": case["case"],
                "vllm_latency_ms": latency_ms,
                "vllm_compute_tflops": compute_tflops(case, latency_ms),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
        finally:
            context = getattr(runner, "config_context", None)
            if context is not None:
                context.__exit__(None, None, None)
            del runner
            torch.cuda.empty_cache()
    write_json(args.output, results)


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def make_report(prefix, cases, vllm_results):
    vllm_map = {item["case"]: item for item in vllm_results}
    long_rows = []
    summary_rows = []
    for case in cases:
        if case["case"] not in vllm_map:
            raise RuntimeError(f"vLLM缺少case: {case['case']}")
        vllm = vllm_map[case["case"]]
        ft_latency = case["fastllm_latency_ms"]
        ft_tflops = compute_tflops(case, ft_latency)
        speedup = vllm["vllm_latency_ms"] / ft_latency
        common = {
            "case": case["case"],
            "op": case["op"],
            "M": case["m"],
            "N/inter": case["n"] or "-",
            "K/hidden": case["k"],
            "topk": case["topk"] or "-",
            "experts": case["experts"] or "-",
            "dtype": case["dtype"],
            "warmup": case["warmup"],
            "iters": case["iters"],
        }
        long_rows.append({
            **common,
            "backend": "FastLLM",
            "latency_ms": ft_latency,
            "compute_tflops": ft_tflops,
            "fastllm_speedup_vs_vllm": speedup,
        })
        long_rows.append({
            **common,
            "backend": "vLLM",
            "latency_ms": vllm["vllm_latency_ms"],
            "compute_tflops": vllm["vllm_compute_tflops"],
            "fastllm_speedup_vs_vllm": speedup,
        })
        summary_rows.append({
            **common,
            "fastllm_latency_ms": ft_latency,
            "vllm_latency_ms": vllm["vllm_latency_ms"],
            "fastllm_speedup_vs_vllm": speedup,
        })
    columns = list(long_rows[0])
    lines = [
        "# NVFP4算子级FastLLM/vLLM性能对比", "",
        "> 同一shape、dtype、warmup、iters和框架调用边界计时；延迟越低越好。"
        "Speedup = vLLM延迟 / FastLLM延迟。", "",
        "## 明细", "",
        "| case | op | 后端 | M | N/inter | K/hidden | topk | experts | dtype | "
        "warmup | iters | latency_ms | compute_TFLOPS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | "
        "---: | ---: | ---: |",
    ]
    for row in long_rows:
        lines.append(
            f"| {row['case']} | {row['op']} | {row['backend']} | {row['M']} | "
            f"{row['N/inter']} | {row['K/hidden']} | {row['topk']} | "
            f"{row['experts']} | {row['dtype']} | {row['warmup']} | {row['iters']} | "
            f"{fmt(row['latency_ms'])} | {fmt(row['compute_tflops'])} |")
    lines.extend([
        "", "## FastLLM相对vLLM", "",
        "| case | FastLLM latency_ms | vLLM latency_ms | FastLLM speedup |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['case']} | {fmt(row['fastllm_latency_ms'])} | "
            f"{fmt(row['vllm_latency_ms'])} | "
            f"{fmt(row['fastllm_speedup_vs_vllm'])}x |")
    lines.append("")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(long_rows)
    write_json(prefix.with_suffix(".json"), {
        "rows": long_rows,
        "summary": summary_rows,
    })
    write_xlsx(prefix.with_suffix(".xlsx"), [
        ("性能明细", columns,
         [[row.get(column) for column in columns] for row in long_rows]),
        ("性能对比", list(summary_rows[0]) if summary_rows else ["case"],
         [[row.get(column) for column in list(summary_rows[0])]
          for row in summary_rows] if summary_rows else []),
    ])
    print("\n".join(lines))
    print(f"Markdown: {prefix.with_suffix('.md')}")
    print(f"CSV: {prefix.with_suffix('.csv')}")
    print(f"JSON: {prefix.with_suffix('.json')}")
    print(f"Excel: {prefix.with_suffix('.xlsx')}")


def orchestrate(args):
    cases = load_fastllm_cases(args.fastllm_log_dir)
    prefix = Path(args.output_prefix)
    cases_path = prefix.with_name(prefix.name + "-cases.json")
    vllm_path = prefix.with_name(prefix.name + "-vllm.json")
    vllm_log = prefix.with_name(prefix.name + "-vllm.log")
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cases_path, cases)
    command = [
        args.vllm_python, str(Path(__file__).resolve()),
        "--stage", "vllm",
        "--fastllm-log-dir", args.fastllm_log_dir,
        "--output-prefix", args.output_prefix,
        "--cases", str(cases_path),
        "--output", str(vllm_path),
    ]
    print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
    with vllm_log.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n\n")
        process = subprocess.Popen(
            command, cwd=REPO_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status != 0:
        raise subprocess.CalledProcessError(status, command)
    vllm_results = json.loads(vllm_path.read_text(encoding="utf-8"))
    make_report(prefix, cases, vllm_results)


def main():
    args = parse_args()
    if args.stage == "vllm":
        if not args.cases or not args.output:
            raise ValueError("vLLM阶段缺少cases或output")
        run_vllm_stage(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
