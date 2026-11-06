#!/usr/bin/env python3
"""覆盖vLLM CUTLASS FP8 scaled-mm官方形状和缩放组合。"""

import argparse
import gc

import torch
from vllm import _custom_ops as ops


ALIGNED = [
    (1, 256, 128), (1, 16384, 1024), (1, 24576, 496),
    (16, 256, 496), (16, 16384, 128), (16, 24576, 4096),
    (32, 8192, 4096), (32, 16384, 4096), (33, 1024, 1024),
    (33, 8192, 128), (64, 2048, 496), (64, 16384, 1024),
    (100, 8192, 496), (128, 32768, 4096), (256, 4096, 4096),
    (512, 256, 1024), (512, 8192, 4096), (512, 16384, 128),
    (512, 24576, 128),
]
UNALIGNED = [
    (32, 3420, 1280), (32, 1280, 6840), (1, 3420, 1280),
    (64, 6840, 1280), (16, 100, 200), (33, 255, 513),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证vLLM Dense或Block128 FP8 scaled-mm")
    parser.add_argument(
        "--scale-layout", choices=("all", "dense", "block128"),
        default="all", help="选择需要验证的scale布局")
    return parser.parse_args()


def expanded_scale(scale, rows, columns, side):
    if scale.numel() == 1:
        return scale
    if side == "a":
        return scale.view(rows, 1)
    return scale.view(1, columns)


def check_case(m, n, k, per_token, per_channel, use_bias,
               out_dtype=torch.bfloat16):
    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
    scale_a = torch.empty(
        (m, 1) if per_token else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = torch.empty(
        (1, n) if per_channel else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    bias = (torch.empty((n,), device="cuda", dtype=out_dtype).uniform_(-0.1, 0.1)
            if use_bias else None)
    actual = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, bias)
    expected = (
        a.float() * expanded_scale(scale_a, m, k, "a")) @ (
        b.float() * expanded_scale(scale_b, k, n, "b"))
    if bias is not None:
        expected += bias.float()
    torch.testing.assert_close(
        actual.float(), expected, rtol=5e-1, atol=1.5e-1)


def check_block128_case(m, n, k, out_dtype=torch.bfloat16):
    """验证vLLM正式(1,128)x(128,128) Blockwise scaled-mm。"""
    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
    scale_a = torch.empty(
        (m, k // 128), device="cuda", dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = torch.empty(
        (k // 128, n // 128), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = scale_b.t().contiguous().t()

    actual = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, None)
    expected = (
        a.float() * scale_a.repeat_interleave(128, dim=1)) @ (
        b.float() * scale_b.repeat_interleave(
            128, dim=0).repeat_interleave(128, dim=1))
    torch.testing.assert_close(
        actual.float(), expected, rtol=5e-1, atol=1.5e-1)


def check_unaligned_case(m, n, k, per_token, per_channel, use_bias):
    """按vLLM高层Linear kernel的真实padding路径检查非对齐形状。"""
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )

    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
    scale_a = torch.empty(
        (m, 1) if per_token else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = torch.empty(
        (1, n) if per_channel else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    bias = (torch.empty((n,), device="cuda", dtype=torch.bfloat16)
            .uniform_(-0.1, 0.1) if use_bias else None)
    expected = (
        a.float() * expanded_scale(scale_a, m, k, "a")) @ (
        b.float() * expanded_scale(scale_b, k, n, "b"))
    if bias is not None:
        expected += bias.float()

    # 与vLLM process_weights_after_loading保持一致：权重和per-channel
    # scale先补齐，apply_scaled_mm再负责激活K维及输出N维的padding/slice。
    pad_k = (16 - k % 16) % 16
    pad_n = (16 - n % 16) % 16
    if pad_k or pad_n:
        b = torch.nn.functional.pad(
            b.t().contiguous(), (0, pad_k, 0, pad_n)).t()
        if pad_n and scale_b.numel() > 1:
            scale_b = torch.nn.functional.pad(scale_b, (0, pad_n), value=1.0)
    kernel = object.__new__(CutlassFP8ScaledMMLinearKernel)
    kernel.logical_output_size = n
    actual = kernel.apply_scaled_mm(
        A=a, B=b, out_dtype=torch.bfloat16, As=scale_a, Bs=scale_b,
        bias=bias, output_shape=[m, n])
    torch.testing.assert_close(
        actual.float(), expected, rtol=5e-1, atol=1.5e-1)


def main():
    args = parse_args()

    capability = torch.cuda.get_device_capability()
    if capability not in ((9, 0), (12, 0)):
        major, minor = capability
        raise RuntimeError(
            f"本测试要求SM90或SM120 GPU，当前为SM{major}{minor}")

    total = 0

    if args.scale_layout in ("all", "dense"):
        # 与vLLM官方test_cutlass_fp8_gemm一致，覆盖两种A scale、
        # 两种B scale和bias。
        for m, n, k in ALIGNED:
            for per_token in (True, False):
                for per_channel in (True, False):
                    for use_bias in (False, True):
                        check_case(
                            m, n, k, per_token, per_channel, use_bias)
                        total += 1
                        print(
                            f"PASS m={m} n={n} k={k} "
                            f"per_token={per_token} "
                            f"per_channel={per_channel} bias={use_bias}",
                            flush=True)
                        torch.cuda.empty_cache()
                        gc.collect()

        for m, n, k in UNALIGNED:
            for per_token in (True, False):
                for per_channel in (True, False):
                    for use_bias in (False, True):
                        check_unaligned_case(
                            m, n, k, per_token, per_channel, use_bias)
                        total += 1
                        print(
                            f"PASS padded m={m} n={n} k={k} "
                            f"per_token={per_token} "
                            f"per_channel={per_channel} "
                            f"bias={use_bias}", flush=True)
                        torch.cuda.empty_cache()
                        gc.collect()

        # 对应vLLM output-dtype case；FastLLM侧也独立覆盖
        # FP16/BF16输出。
        for out_dtype in (torch.bfloat16, torch.float16):
            check_case(512, 512, 512, True, True, True, out_dtype)
            total += 1
            print(f"PASS output_dtype={out_dtype}", flush=True)

    if args.scale_layout in ("all", "block128"):
        # 对应vLLM test_cutlass_fp8_blockwise_scale_gemm：只执行
        # 满足Block128布局约束的官方MNK。
        for m, n, k in ALIGNED:
            if n % 128 != 0 or k % 128 != 0:
                continue
            for out_dtype in (torch.bfloat16, torch.float16):
                check_block128_case(m, n, k, out_dtype)
                total += 1
                print(
                    f"PASS block128 m={m} n={n} k={k} "
                    f"output_dtype={out_dtype}", flush=True)
                torch.cuda.empty_cache()
                gc.collect()

    print(
        f"Summary: PASS ({total} vLLM FP8 scaled-mm cases; "
        f"scale_layout={args.scale_layout})")


if __name__ == "__main__":
    main()
