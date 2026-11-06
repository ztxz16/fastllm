#!/usr/bin/env python3
"""比较FastLLM与vLLM的Dense W8A8线性算子性能。"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))
from xlsx_report import write_xlsx  # noqa: E402

# 前几项覆盖FastLLM SM90/SM120的分派边界；其余形状来自vLLM
# tests/kernels/quantization/test_cutlass_scaled_mm.py::MNK_FACTORS。
CASES = [
    ("branch_m1", 1, 4096, 4096),
    ("branch_m16", 16, 4096, 4096),
    ("branch_m17", 17, 4096, 4096),
    ("branch_m32", 32, 4096, 4096),
    ("branch_m33", 33, 4096, 4096),
    ("branch_m256", 256, 4096, 4096),
    ("branch_m257", 257, 4096, 4096),
    ("sm90_swap_n1280", 17, 1280, 4096),
    ("sm90_swap_n1288", 17, 1288, 4096),
    ("sm90_cooperative", 8192, 128, 6144),
    ("vllm_m1_n256_k128", 1, 256, 128),
    ("vllm_m1_n16384_k1024", 1, 16384, 1024),
    ("vllm_m1_n24576_k496", 1, 24576, 496),
    ("vllm_m16_n256_k496", 16, 256, 496),
    ("vllm_m16_n16384_k128", 16, 16384, 128),
    ("vllm_m16_n24576_k4096", 16, 24576, 4096),
    ("vllm_m32_n8192_k4096", 32, 8192, 4096),
    ("vllm_m32_n16384_k4096", 32, 16384, 4096),
    ("vllm_m33_n1024_k1024", 33, 1024, 1024),
    ("vllm_m33_n8192_k128", 33, 8192, 128),
    ("vllm_m64_n2048_k496", 64, 2048, 496),
    ("vllm_m64_n16384_k1024", 64, 16384, 1024),
    ("vllm_m100_n8192_k496", 100, 8192, 496),
    ("vllm_m128_n32768_k4096", 128, 32768, 4096),
    ("vllm_m256_n4096_k4096", 256, 4096, 4096),
    ("vllm_m512_n256_k1024", 512, 256, 1024),
    ("vllm_m512_n8192_k4096", 512, 8192, 4096),
    ("vllm_m512_n16384_k128", 512, 16384, 128),
    ("vllm_m512_n24576_k128", 512, 24576, 128),
]

# Block128只接受K、N均按128对齐。前六项覆盖vLLM SM120分派条件：
# M<=64或M不能被4整除时走SwapAB，其余M<=256走Pingpong，之后走默认配置。
BLOCK128_CASES = [
    ("block_branch_m1", 1, 4096, 4096),
    ("block_branch_m64", 64, 4096, 4096),
    ("block_branch_m65", 65, 4096, 4096),
    ("block_branch_m68", 68, 4096, 4096),
    ("block_branch_m256", 256, 4096, 4096),
    ("block_branch_m257", 257, 4096, 4096),
] + [
    (f"block_{name}", m, n, k)
    for name, m, n, k in CASES
    if name.startswith("vllm_") and n % 128 == 0 and k % 128 == 0
]

# vLLM SM90 INT8入口要求B的K、N均按16对齐。只保留双方能直接
# 调用CUTLASS的共同形状，不将padding开销混入INT8对比。
INT8_CASES = [
    case for case in CASES if case[2] % 16 == 0 and case[3] % 16 == 0
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SM90/SM120 W8A8 FastLLM/vLLM算子性能对比")
    parser.add_argument("--optest", default=str(REPO_DIR / "optest"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--outer-repeats", type=int, default=5,
        help="每个形状的独立重复次数；用于计算Median、P95和CV")
    parser.add_argument(
        "--scale-layout", choices=("perchannel", "block128"),
        default="perchannel",
        help="双方共同使用的权重/激活scale语义")
    parser.add_argument(
        "--quantization",
        choices=("fp8", "int8-symmetric", "int8-azp"), default="fp8",
        help=("fp8为现有Dense/Block128路径；int8-symmetric为SM90动态"
              "对称INT8；int8-azp为SM90动态非对称INT8"))
    return parser.parse_args()


def selected_cases(args):
    """按scale布局返回合法且覆盖正式分派边界的测试形状。"""
    if args.quantization.startswith("int8-"):
        return INT8_CASES
    return BLOCK128_CASES if args.scale_layout == "block128" else CASES


def percentile(samples, ratio):
    """使用线性插值计算少量独立延迟样本的分位数。"""
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(prefix, samples, m, n, k, backend_path):
    """汇总独立测试样本，并计算延迟、稳定性和有效计算吞吐。"""
    median_ms = statistics.median(samples)
    mean_ms = statistics.mean(samples)
    cv_pct = (statistics.pstdev(samples) / mean_ms * 100.0
              if len(samples) > 1 and mean_ms > 0.0 else 0.0)
    return {
        f"{prefix}_backend_path": backend_path,
        f"{prefix}_latency_median_ms": median_ms,
        f"{prefix}_latency_p95_ms": percentile(samples, 0.95),
        f"{prefix}_latency_cv_pct": cv_pct,
        f"{prefix}_effective_tops": 2.0 * m * n * k / median_ms / 1e9,
        f"{prefix}_samples_ms": samples,
    }


def display_value(value):
    """将报告展示层的浮点值统一格式化为小数点后3位。"""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return json.dumps(
            [f"{item:.3f}" if isinstance(item, float) else item
             for item in value], ensure_ascii=False)
    return value


def runtime_metadata(args):
    """采集性能报告的硬件、构建和完整执行参数。"""
    import torch

    major, minor = torch.cuda.get_device_capability()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True,
        stdout=subprocess.PIPE, check=True).stdout.strip()
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{major}.{minor}",
        "cuda_version": torch.version.cuda,
        "compile_arch": "90a" if (major, minor) == (9, 0)
                        else f"{major}{minor}",
        "git_commit": commit,
        "command": shlex.join([sys.executable, *sys.argv]),
        "warmup": args.warmup,
        "iters": args.iters,
        "outer_repeats": args.outer_repeats,
    }


def run_fastllm(args, log_dir):
    pattern = re.compile(r"latency:\s+avg_ms=([0-9.eE+-]+)")
    rows = []
    env = os.environ.copy()
    env.update({
        "FASTLLM_CUDA_W8A8": "1",
        "FASTLLM_CUDA_W8A8_STRICT": "1",
        "FASTLLM_CUDA_W8A8_TRACE": "0",
    })
    if args.quantization.startswith("int8-"):
        weight_layout = None
        op_name = "linear_int8_w8a8"
        backend_path = (
            "w8a8-int8-azp-cutlass (strict)"
            if args.quantization == "int8-azp"
            else "w8a8-int8-cutlass (strict)")
    else:
        weight_layout = (
            "separate" if args.scale_layout == "block128" else "perchannel")
        op_name = (
            "linear_fp8_block128"
            if args.scale_layout == "block128" else "linear_fp8_w8a8")
        backend_path = (
            "w8a8-block128-cutlass (strict)"
            if args.scale_layout == "block128" else "w8a8-cutlass (strict)")

    for name, m, n, k in selected_cases(args):
        samples = []
        for repeat in range(args.outer_repeats):
            command = [
                args.optest, "--op", op_name, "--device", "cuda:0",
                "--param", f"batch={m}", "--param", f"in={k}",
                "--param", f"out={n}",
                "--param", "input_type=bf16", "--param", "has_bias=0",
                "--warmup", str(args.warmup), "--iters", str(args.iters),
            ]
            if args.quantization.startswith("int8-"):
                activation_scheme = (
                    "azp_dynamic" if args.quantization == "int8-azp"
                    else "symmetric")
                command.extend([
                    "--param", f"activation_scheme={activation_scheme}"])
            else:
                command.extend(["--param", f"weight_layout={weight_layout}"])
            result = subprocess.run(
                command, cwd=REPO_DIR, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            (log_dir / f"fastllm-{name}-r{repeat}.log").write_text(
                result.stdout, encoding="utf-8")
            matches = pattern.findall(result.stdout)
            if not matches:
                raise RuntimeError(f"FastLLM未输出{name}的latency")
            samples.append(float(matches[-1]))
        rows.append({
            "case": name, "m": m, "n": n, "k": k,
            "scale_layout": (
                "per-token AZP/per-channel"
                if args.quantization == "int8-azp"
                else ("per-token symmetric/per-channel"
                      if args.quantization == "int8-symmetric"
                      else args.scale_layout)),
            **summarize_samples(
                "fastllm", samples, m, n, k, backend_path),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def run_vllm(args):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    rows = []
    fp8 = torch.float8_e4m3fn
    for name, m, n, k in selected_cases(args):
        samples = []
        if args.quantization.startswith("int8-"):
            asymmetric = args.quantization == "int8-azp"
            for _ in range(args.outer_repeats):
                torch.cuda.empty_cache()
                gc.collect()
                indices = torch.arange(
                    m * k, device="cuda", dtype=torch.int32)
                source = (((indices * 37) % 501 - 100) * 0.001).to(
                    torch.bfloat16).view(m, k)
                k_indices = torch.arange(
                    k, device="cuda", dtype=torch.int32).view(k, 1)
                n_indices = torch.arange(
                    n, device="cuda", dtype=torch.int32).view(1, n)
                weight = ((n_indices * 17 + k_indices * 13) % 255 - 127).to(
                    torch.int8)
                # vLLM CUTLASS接收逻辑[K,N]、物理Column-Major的B矩阵。
                weight = weight.t().contiguous().t()
                weight_scale = (
                    0.0025 + 0.00001 * (n_indices % 31)).to(torch.float32)
                del indices, k_indices, n_indices
                azp_adj = (weight.sum(dim=0, keepdim=True, dtype=torch.int32)
                           if asymmetric else None)

                def run_int8():
                    quantized, token_scale, token_azp = ops.scaled_int8_quant(
                        source, None, None, symmetric=not asymmetric)
                    if asymmetric:
                        return ops.cutlass_scaled_mm_azp(
                            quantized, weight, token_scale, weight_scale,
                            torch.bfloat16, azp_adj, token_azp, None)
                    return ops.cutlass_scaled_mm(
                        quantized, weight, token_scale, weight_scale,
                        torch.bfloat16, None)

                for _ in range(args.warmup):
                    output = run_int8()
                torch.cuda.synchronize()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iters):
                    output = run_int8()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) / args.iters)
                del source, weight, weight_scale, output
                if azp_adj is not None:
                    del azp_adj
            rows.append({
                "case": name, "m": m, "n": n, "k": k,
                "scale_layout": (
                    "per-token AZP/per-channel" if asymmetric
                    else "per-token symmetric/per-channel"),
                **summarize_samples(
                    "vllm", samples, m, n, k,
                    ("scaled_int8_quant(asymmetric) + "
                     "cutlass_scaled_mm_azp" if asymmetric else
                     "scaled_int8_quant(symmetric) + cutlass_scaled_mm")),
            })
            print(json.dumps(rows[-1]), flush=True)
            continue

        pad_k = 0 if args.scale_layout == "block128" else (16 - k % 16) % 16
        pad_n = 0 if args.scale_layout == "block128" else (16 - n % 16) % 16
        dense_kernel = None
        if args.scale_layout == "perchannel":
            # 复用vLLM正式Linear运行入口；静态权重padding在模型加载阶段
            # 完成，动态激活padding和输出slice由apply_scaled_mm计入延迟。
            dense_kernel = object.__new__(CutlassFP8ScaledMMLinearKernel)
            dense_kernel.logical_output_size = n

        for _ in range(args.outer_repeats):
            torch.cuda.empty_cache()
            gc.collect()
            source = torch.empty(
                (m, k), device="cuda", dtype=torch.bfloat16).normal_()
            weight = torch.empty(
                (n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
            if args.scale_layout == "block128":
                weight_scale = torch.empty(
                    (k // 128, n // 128), device="cuda",
                    dtype=torch.float32).uniform_(0.001, 0.01)
                # 与vLLM功能测试相同：B scale在逻辑上K-major，且末维连续。
                weight_scale = weight_scale.t().contiguous().t()
            else:
                weight_scale = torch.empty(
                    (1, n), device="cuda",
                    dtype=torch.float32).uniform_(0.001, 0.01)
                if pad_k or pad_n:
                    weight = torch.nn.functional.pad(
                        weight.t().contiguous(),
                        (0, pad_k, 0, pad_n)).t()
                    if pad_n:
                        weight_scale = torch.nn.functional.pad(
                            weight_scale, (0, pad_n), value=1.0)

            def run():
                if args.scale_layout == "block128":
                    quantized, token_scale = per_token_group_quant_fp8(
                        source, 128, use_ue8m0=False)
                    return ops.cutlass_scaled_mm(
                        quantized, weight, token_scale, weight_scale,
                        torch.bfloat16, None)
                else:
                    quantized, token_scale = ops.scaled_fp8_quant(
                        source, scale=None, use_per_token_if_dynamic=True)
                    return dense_kernel.apply_scaled_mm(
                        A=quantized, B=weight, out_dtype=torch.bfloat16,
                        As=token_scale, Bs=weight_scale, bias=None,
                        output_shape=[m, n])

            for _ in range(args.warmup):
                output = run()
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iters):
                output = run()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.iters)
            del source, weight, weight_scale, output
        rows.append({
            "case": name, "m": m, "n": n, "k": k,
            "scale_layout": args.scale_layout,
            **summarize_samples(
                "vllm", samples, m, n, k,
                ("per_token_group_quant_fp8(128) + cutlass_scaled_mm"
                 if args.scale_layout == "block128"
                 else (
                     "scaled_fp8_quant + pad + cutlass_scaled_mm + slice"
                     if pad_k or pad_n
                     else "scaled_fp8_quant + cutlass_scaled_mm"))),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def write_reports(output_dir, fastllm_rows, vllm_rows, scale_layout,
                  quantization, metadata):
    vllm = {row["case"]: row for row in vllm_rows}
    combined = []
    for row in fastllm_rows:
        other = vllm[row["case"]]
        ft_ms = row["fastllm_latency_median_ms"]
        vl_ms = other["vllm_latency_median_ms"]
        combined.append({
            "result": "PASS",
            **metadata,
            **row,
            **{key: value for key, value in other.items()
               if key.startswith("vllm_")},
            "fastllm_speedup_vs_vllm_x": vl_ms / ft_ms,
        })
    (output_dir / "operator-performance-compare.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8")
    with (output_dir / "operator-performance-compare.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(
            {key: display_value(value) for key, value in row.items()}
            for row in combined)
    headers = list(combined[0])
    write_xlsx(
        output_dir / "operator-performance-compare.xlsx",
        [("算子性能", headers,
          [[display_value(row[key]) for key in headers] for row in combined])])
    lines = [
        f"# SM90/SM120 W8A8算子性能对比（{quantization}/{scale_layout}）", "",
        f"> GPU: {metadata['gpu']}；CC: {metadata['compute_capability']}；"
        f"CUDA: {metadata['cuda_version']}；编译架构: {metadata['compile_arch']}；"
        f"Git: {metadata['git_commit']}。", "",
        f"> 命令: `{metadata['command']}`", "",
        ("> 双方都计入BF16激活动态per-token非对称INT8量化、"
         "per-channel对称INT8权重、AZP融合CUTLASS GEMM及输出；无bias。"
         if quantization == "int8-azp" else
         ("> 双方都计入BF16激活动态per-token对称INT8量化、"
          "per-channel对称INT8权重、CUTLASS GEMM及输出；无bias。"
          if quantization == "int8-symmetric" else
         ("> 双方都计入BF16激活动态per-group(1,128) FP8量化、"
          "block(128,128)权重"
          if scale_layout == "block128"
          else "> 双方都计入BF16激活动态per-token FP8量化、per-channel权重"))),
        "> 每项使用独立重复样本计算统计值。", "",
        "> 加速比 = vLLM耗时 / FastLLM耗时；大于1表示FastLLM更快，",
        "> 小于1表示FastLLM更慢。", "",
        "> Effective TOPS由`2*M*N*K/Median延迟`推导；CV越小表示重复测试越稳定。", "",
        "| 状态 | Case | M | N | K | FastLLM路径 | FastLLM Median(ms) | FastLLM P95(ms) | FastLLM CV(%) | FastLLM TOPS | vLLM路径 | vLLM Median(ms) | vLLM P95(ms) | vLLM CV(%) | vLLM TOPS | FastLLM相对vLLM加速比 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in combined:
        lines.append(
            f"| {row['result']} | {row['case']} | {row['m']} | {row['n']} | {row['k']} | "
            f"{row['fastllm_backend_path']} | "
            f"{row['fastllm_latency_median_ms']:.3f} | "
            f"{row['fastllm_latency_p95_ms']:.3f} | "
            f"{row['fastllm_latency_cv_pct']:.3f} | "
            f"{row['fastllm_effective_tops']:.3f} | "
            f"{row['vllm_backend_path']} | "
            f"{row['vllm_latency_median_ms']:.3f} | "
            f"{row['vllm_latency_p95_ms']:.3f} | "
            f"{row['vllm_latency_cv_pct']:.3f} | "
            f"{row['vllm_effective_tops']:.3f} | "
            f"{row['fastllm_speedup_vs_vllm_x']:.3f}x |")
    report = "\n".join(lines) + "\n"
    (output_dir / "operator-performance-compare.md").write_text(
        report, encoding="utf-8")
    print(report)
    print(f"Summary: PASS ({len(combined)} operator performance cases)")
    print(f"Excel: {output_dir / 'operator-performance-compare.xlsx'}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fastllm_rows = run_fastllm(args, output_dir)
    # FastLLM均在子进程中运行并退出，父进程此时才加载vLLM，避免显存共存。
    vllm_rows = run_vllm(args)
    metadata = runtime_metadata(args)
    write_reports(
        output_dir, fastllm_rows, vllm_rows,
        args.scale_layout, args.quantization, metadata)


if __name__ == "__main__":
    main()
