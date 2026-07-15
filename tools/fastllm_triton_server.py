#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


_compile_lock = Lock()
_triton_error = None

try:
    import triton
    import triton.language as tl
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
except Exception as exc:  # pragma: no cover - this is reported through /compile.
    triton = None
    tl = None
    ASTSource = None
    GPUTarget = None
    _triton_error = exc


if triton is not None:
    @triton.jit
    def _fastllm_fp8e4m3_to_float(x):
        x = x.to(tl.uint32)
        bits = ((x & 0x80) << 8) | ((x & 0x7F) << 7)
        return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * 256.0


    @triton.jit
    def fastllm_linear_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * K + k_idxs[:, None],
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="tf32")

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def fastllm_merge_moe_fp8_init_count_kernel(
        indices_ptr,
        expert_counts,
        expert_offsets,
        expert_cursors,
        expert_block_offsets,
        total_blocks_ptr,
        total_tasks,
        experts,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        expert_offs = tl.arange(0, BLOCK_E)
        expert_mask = expert_offs < experts
        tl.store(expert_counts + expert_offs, 0, mask=expert_mask)
        tl.store(expert_offsets + expert_offs, 0, mask=expert_mask)
        tl.store(expert_cursors + expert_offs, 0, mask=expert_mask)
        tl.store(expert_block_offsets + expert_offs, 0, mask=expert_mask)
        tl.store(expert_offsets + experts, 0)
        tl.store(total_blocks_ptr, 0)
        tl.debug_barrier()

        task_offs = tl.arange(0, BLOCK_T)
        task_mask = task_offs < total_tasks
        expert = tl.load(indices_ptr + task_offs, mask=task_mask, other=-1)
        valid = task_mask & (expert >= 0) & (expert < experts)
        tl.atomic_add(expert_counts + expert, 1, sem="relaxed", mask=valid)


    @triton.jit
    def fastllm_merge_moe_fp8_zero_route_kernel(
        expert_counts,
        expert_offsets,
        expert_cursors,
        expert_block_offsets,
        total_blocks_ptr,
        experts,
        BLOCK_E: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_E)
        mask = offs < experts
        tl.store(expert_counts + offs, 0, mask=mask)
        tl.store(expert_offsets + offs, 0, mask=mask)
        tl.store(expert_cursors + offs, 0, mask=mask)
        tl.store(expert_block_offsets + offs, 0, mask=mask)
        tl.store(expert_offsets + experts, 0)
        tl.store(total_blocks_ptr, 0)


    @triton.jit
    def fastllm_merge_moe_fp8_count_kernel(
        indices_ptr,
        expert_counts,
        total_tasks,
        experts,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs < total_tasks
        expert = tl.load(indices_ptr + offs, mask=mask, other=-1)
        valid = mask & (expert >= 0) & (expert < experts)
        tl.atomic_add(expert_counts + expert, 1, sem="relaxed", mask=valid)


    @triton.jit
    def fastllm_merge_moe_fp8_prefix_kernel(
        expert_counts,
        expert_offsets,
        expert_cursors,
        expert_block_offsets,
        total_blocks_ptr,
        experts,
        BLOCK_E: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_E)
        mask = offs < experts
        counts = tl.load(expert_counts + offs, mask=mask, other=0)
        block_counts = tl.cdiv(counts, BLOCK_M)
        block_cumsum = tl.cumsum(block_counts, 0)
        block_starts = block_cumsum - block_counts
        padded_counts = block_counts * BLOCK_M
        padded_cumsum = tl.cumsum(padded_counts, 0)
        starts = padded_cumsum - padded_counts

        tl.store(expert_offsets + offs, starts, mask=mask)
        tl.store(expert_offsets + experts, tl.sum(padded_counts, axis=0))
        tl.store(expert_cursors + offs, 0, mask=mask)
        tl.store(expert_block_offsets + offs, block_starts, mask=mask)
        tl.store(total_blocks_ptr, tl.sum(block_counts, axis=0))


    @triton.jit
    def fastllm_merge_moe_fp8_fill_sorted_kernel(
        sorted_tasks,
        expert_offsets,
        total_tasks,
        experts,
        BLOCK_T: tl.constexpr,
    ):
        total_padded = tl.load(expert_offsets + experts)
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs < total_padded
        tl.store(sorted_tasks + offs, total_tasks, mask=mask)


    @triton.jit
    def fastllm_merge_moe_fp8_scatter_blocks_kernel(
        indices_ptr,
        expert_offsets,
        expert_cursors,
        expert_block_offsets,
        sorted_tasks,
        block_experts,
        block_starts,
        total_tasks,
        experts,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs < total_tasks
        expert = tl.load(indices_ptr + offs, mask=mask, other=-1)
        valid = mask & (expert >= 0) & (expert < experts)
        local = tl.atomic_add(expert_cursors + expert, 1, sem="relaxed", mask=valid)
        start = tl.load(expert_offsets + expert, mask=valid, other=0)
        pos = start + local
        tl.store(sorted_tasks + pos, offs, mask=valid)

        block_local = local // BLOCK_M
        block_start_task = (local % BLOCK_M) == 0
        dst = tl.load(expert_block_offsets + expert, mask=valid, other=0) + block_local
        tl.store(block_experts + dst, expert, mask=valid & block_start_task)
        tl.store(block_starts + dst, pos, mask=valid & block_start_task)


    @triton.jit
    def fastllm_merge_moe_fp8_zero_output_kernel(
        output_accum,
        elements,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs < elements
        tl.store(output_accum + offs, tl.zeros((BLOCK_T,), dtype=tl.float32), mask=mask)


    @triton.jit
    def fastllm_merge_moe_fp8_cast_output_kernel(
        output_accum,
        output,
        elements,
        COMPUTE_TYPE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs < elements
        values = tl.load(output_accum + offs, mask=mask, other=0.0)
        tl.store(output + offs, values.to(COMPUTE_TYPE), mask=mask)


    @triton.jit
    def fastllm_merge_moe_fp8_quant_input_kernel(
        input_ptr,
        q_ptr,
        scale_ptr,
        batch,
        hidden,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        group = tl.program_id(1)
        offs = group * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (token < batch) & (offs < hidden)
        x = tl.load(input_ptr + token * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x)), 1.0e-10)
        scale = absmax * (1.0 / 448.0)
        q = tl.clamp(x / scale, -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + token * hidden + offs, q, mask=mask)
        tl.store(scale_ptr + token * tl.cdiv(hidden, BLOCK_K) + group, scale, mask=token < batch)


    @triton.jit
    def fastllm_linear_fp8_block128_matmul_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        PER_ROW,
        SCALE_COLS,
        HAS_BIAS: tl.constexpr,
        PACKED_WEIGHT: tl.constexpr,
        COMPUTE_TYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WEIGHT_BLOCK_N: tl.constexpr,
        WEIGHT_BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        group_size_m = tl.maximum(group_size_m, 1)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_scale_cols = tl.cdiv(K, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_idxs = k * BLOCK_K + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            if PACKED_WEIGHT:
                b = tl.load(
                    b_ptr
                    + offs_n[None, :] * PER_ROW
                    + k * (BLOCK_K + 4)
                    + offs_k[:, None],
                    mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                    other=0.0,
                )
                b_scale_ptrs = (
                    b_ptr + offs_n * PER_ROW + k * (BLOCK_K + 4) + BLOCK_K
                ).to(tl.pointer_type(tl.float32))
                b_scale = tl.load(b_scale_ptrs, mask=offs_n < N, other=0.0)
            else:
                b = tl.load(
                    b_ptr + offs_n[None, :] * K + k_idxs[:, None],
                    mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                    other=0.0,
                )
                b_scale = tl.load(
                    b_scale_ptr
                    + (offs_n // WEIGHT_BLOCK_N) * SCALE_COLS
                    + ((k * BLOCK_K) // WEIGHT_BLOCK_K),
                    mask=offs_n < N,
                    other=0.0,
                )
            a_scale = tl.load(
                a_scale_ptr + offs_m * a_scale_cols + k,
                mask=offs_m < M,
                other=0.0,
            )
            acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(COMPUTE_TYPE),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def fastllm_linear_fp8_block128_strided_matmul_kernel(
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        group_n,
        group_k,
        stride_am,
        stride_bn,
        stride_cm,
        stride_As_m,
        stride_Bs_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :])
        b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * stride_bn)

        As_ptrs = As + offs_am * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * stride_Bs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks)
            b_s = tl.load(Bs_ptrs + offs_ks)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


    @triton.jit
    def fastllm_deepseek_v4_fp8_woa_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        out_ptr,
        NUM_TOKENS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        OUT_RANK: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        UPCAST_FP8: tl.constexpr,
    ):
        """DeepSeek-V4 ``bhr,hdr->bhd`` block-scaled FP8 output projection.

        The decode input is contiguous as [token, group, hidden].  Weight and
        scale layouts match the checkpoint directly: [group, out, hidden] and
        [group, out/128, hidden/128].  This is the SM89/SM12x vLLM einsum
        schedule, specialized here so the AOT launcher only passes pointers.
        """
        token_block = tl.program_id(0)
        out_block = tl.program_id(1)
        group = tl.program_id(2)

        token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
        accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

        hidden_blocks: tl.constexpr = HIDDEN_SIZE // BLOCK_HIDDEN
        out_blocks: tl.constexpr = OUT_RANK // BLOCK_OUT
        for hidden_block in range(0, hidden_blocks):
            hidden = hidden_block * BLOCK_HIDDEN + hidden_offsets
            a = tl.load(
                a_ptr
                + (token_offsets[:, None] * NUM_GROUPS + group) * HIDDEN_SIZE
                + hidden[None, :],
                mask=token_offsets[:, None] < NUM_TOKENS,
                other=0.0,
            )
            b = tl.load(
                b_ptr
                + (group * OUT_RANK + out_offsets[None, :]) * HIDDEN_SIZE
                + hidden[:, None],
                mask=out_offsets[None, :] < OUT_RANK,
                other=0.0,
            )
            if UPCAST_FP8:
                # Ada does not expose the native FP8 dot used by SM12x.
                a = a.to(tl.bfloat16)
                b = b.to(tl.bfloat16)
            raw = tl.dot(a, b, out_dtype=tl.float32)
            a_scale = tl.load(
                a_scale_ptr
                + (token_offsets * NUM_GROUPS + group) * hidden_blocks
                + hidden_block,
                mask=token_offsets < NUM_TOKENS,
                other=0.0,
            )
            b_scale = tl.load(
                b_scale_ptr
                + (group * out_blocks + out_offsets // BLOCK_OUT) * hidden_blocks
                + hidden_block,
                mask=out_offsets < OUT_RANK,
                other=0.0,
            )
            accum += raw * a_scale[:, None] * b_scale[None, :]

        tl.store(
            out_ptr
            + (token_offsets[:, None] * NUM_GROUPS + group) * OUT_RANK
            + out_offsets[None, :],
            accum,
            mask=(token_offsets[:, None] < NUM_TOKENS)
            & (out_offsets[None, :] < OUT_RANK),
        )


    @triton.jit
    def fastllm_deepseek_v4_sparse_decode_kernel(
        q_ptr,
        window_kv_ptr,
        compressed_kv_ptr,
        sink_ptr,
        decode_meta_ptr,
        output_ptr,
        softmax_scale,
        BATCH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Graph-safe DeepSeek-V4 sparse MLA single-token decode.

        This follows vLLM's FP8DS online-softmax schedule, while consuming
        FastLLM's existing FP32 sliding-window cache and BF16 compressed cache.
        A Triton program owns one (batch, head-block) tile, keeps the query and
        online-softmax state in registers, and avoids the CUDA fallback's
        block-wide synchronization for every candidate key.
        """
        batch_idx = tl.program_id(0)
        head_block_idx = tl.program_id(1)
        head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < NUM_HEADS
        dim_mask = dim_offsets < HEAD_DIM
        matrix_mask = head_mask[:, None] & dim_mask[None, :]

        q = tl.load(
            q_ptr
            + (batch_idx * NUM_HEADS + head_offsets[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
        running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
        running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

        start_pos = tl.load(decode_meta_ptr)
        live_window = tl.minimum(start_pos + 1, WINDOW_SIZE)
        ring_pos = start_pos % WINDOW_SIZE
        ring_full = start_pos >= WINDOW_SIZE - 1
        for candidate_idx in range(0, live_window):
            window_idx = tl.where(
                ring_full,
                (ring_pos + 1 + candidate_idx) % WINDOW_SIZE,
                candidate_idx,
            )
            kv = tl.load(
                window_kv_ptr
                + (batch_idx * WINDOW_SIZE + window_idx) * HEAD_DIM
                + dim_offsets,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv[None, :], axis=1) * softmax_scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

        if COMPRESS_RATIO > 0:
            compressed_count = (start_pos + 1) // COMPRESS_RATIO
            for candidate_idx in range(0, compressed_count):
                kv = tl.load(
                    compressed_kv_ptr
                    + batch_idx * compressed_count * HEAD_DIM
                    + candidate_idx * HEAD_DIM
                    + dim_offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(q * kv[None, :], axis=1) * softmax_scale
                next_max = tl.maximum(running_max, score)
                previous_weight = tl.exp(running_max - next_max)
                candidate_weight = tl.exp(score - next_max)
                running_acc = (
                    running_acc * previous_weight[:, None]
                    + kv[None, :] * candidate_weight[:, None]
                )
                running_denom = running_denom * previous_weight + candidate_weight
                running_max = next_max

        sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=-float("inf"))
        has_tokens = running_denom > 0.0
        has_sink = sink > -float("inf")
        valid_max = tl.where(has_tokens, running_max, -float("inf"))
        valid_sink = tl.where(has_sink, sink, -float("inf"))
        merge_max = tl.maximum(valid_max, valid_sink)
        has_any = has_tokens | has_sink
        safe_merge_max = tl.where(has_any, merge_max, 0.0)
        safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
        safe_sink = tl.where(has_sink, sink, safe_merge_max)
        subset_scale = tl.where(
            has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0
        )
        sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
        total_weight = running_denom * subset_scale + sink_weight
        inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)
        final = running_acc * subset_scale[:, None] * inv_total[:, None]
        output_offsets = (
            (batch_idx * NUM_HEADS + head_offsets[:, None]) * HEAD_DIM
            + dim_offsets[None, :]
        )
        tl.store(output_ptr + output_offsets, final, mask=matrix_mask)


    @triton.jit
    def fastllm_merge_moe_fp8_swiglu_quant_kernel(
        gateup_ptr,
        c_ptr,
        c_scale_ptr,
        total_tasks,
        inter,
        COMPUTE_TYPE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        task = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (task < total_tasks) & (offs_n < inter)

        gate = tl.load(
            gateup_ptr + task * (inter * 2) + offs_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gateup_ptr + task * (inter * 2) + inter + offs_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated = (gate / (1.0 + tl.exp(-gate))) * up
        act_absmax = tl.maximum(tl.max(tl.abs(activated)), 1.0e-10)
        act_scale = act_absmax * (1.0 / 448.0)
        activated_q = tl.clamp(activated / act_scale, -448.0, 448.0).to(c_ptr.dtype.element_ty)
        scale_cols = tl.cdiv(inter, BLOCK_N)
        tl.store(c_ptr + task * inter + offs_n, activated_q, mask=mask)
        tl.store(c_scale_ptr + task * scale_cols + pid_n, act_scale, mask=task < total_tasks)


    @triton.jit
    def fastllm_merge_moe_fp8_fused_gateup_matmul_kernel(
        a_ptr,
        gate_ptr,
        up_ptr,
        c_ptr,
        a_scale_ptr,
        gate_scale_ptr,
        up_scale_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        INTER: tl.constexpr,
        EM,
        num_valid_tokens,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_be: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_ask: tl.constexpr,
        stride_bse: tl.constexpr,
        stride_bsk: tl.constexpr,
        stride_bsn: tl.constexpr,
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts == -1:
            fastllm_merge_moe_fp8_write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
            return

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
        is_up = offs_bn >= INTER
        local_bn = offs_bn - tl.where(is_up, INTER, 0)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        gate_base = gate_ptr + off_experts * stride_be
        up_base = up_ptr + off_experts * stride_be
        gate_scale_base = gate_scale_ptr + off_experts * stride_bse
        up_scale_base = up_scale_ptr + off_experts * stride_bse
        gate_ptrs = (
            gate_base
            + offs_k[:, None] * stride_bk
            + local_bn[None, :] * stride_bn
        )
        up_ptrs = (
            up_base
            + offs_k[:, None] * stride_bk
            + local_bn[None, :] * stride_bn
        )
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        offs_bsn = local_bn // group_n
        gate_scale_ptrs = gate_scale_base + offs_bsn * stride_bsn
        up_scale_ptrs = up_scale_base + offs_bsn * stride_bsn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_limit = K - k * BLOCK_SIZE_K
            b_mask = (offs_k[:, None] < k_limit) & (local_bn[None, :] < INTER)
            gate_b = tl.load(gate_ptrs, mask=b_mask, other=0.0)
            up_b = tl.load(up_ptrs, mask=b_mask, other=0.0)
            b = tl.where(is_up[None, :], up_b, gate_b)
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_limit),
                other=0.0,
            )
            offs_ks = (k * BLOCK_SIZE_K) // group_k
            a_scale = tl.load(
                a_scale_ptrs + offs_ks * stride_ask,
                mask=token_mask,
                other=0.0,
            )
            gate_b_scale = tl.load(
                gate_scale_ptrs + offs_ks * stride_bsk,
                mask=local_bn < INTER,
                other=0.0,
            )
            up_b_scale = tl.load(
                up_scale_ptrs + offs_ks * stride_bsk,
                mask=local_bn < INTER,
                other=0.0,
            )
            b_scale = tl.where(is_up, up_b_scale, gate_b_scale)
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_ak
            gate_ptrs += BLOCK_SIZE_K * stride_bk
            up_ptrs += BLOCK_SIZE_K * stride_bk

        accumulator = accumulator.to(compute_type)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


    @triton.jit
    def fastllm_merge_moe_fp8_write_zeros_to_output(
        c_ptr,
        stride_cm,
        stride_cn,
        pid_n,
        N,
        offs_token,
        token_mask,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


    @triton.jit
    def fastllm_merge_moe_fp8_fused_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_bias_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_be: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_ask: tl.constexpr,
        stride_bse: tl.constexpr,
        stride_bsk: tl.constexpr,
        stride_bsn: tl.constexpr,
        stride_bbe: tl.constexpr,
        stride_bbn: tl.constexpr,
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        naive_block_assignment: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        per_channel_quant: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        if not naive_block_assignment:
            offs_token_id = pid_m * BLOCK_SIZE_M + offs
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        else:
            offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
        offs_token = offs_token.to(tl.int64)

        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts == -1:
            fastllm_merge_moe_fp8_write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
            return

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_base = b_ptr + off_experts * stride_be
        b_scale_base = b_scale_ptr + off_experts * stride_bse
        b_ptrs = (
            b_base
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

        if use_int8_w8a16:
            b_scale_ptrs = b_scale_base + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = b_scale_base + offs_bsn * stride_bsn
            elif per_channel_quant:
                b_scale_ptrs = b_scale_base + offs_bn[None, :] * stride_bsn
                b_scale = tl.load(b_scale_ptrs)
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:
                a_scale = tl.load(a_scale_ptr)
                b_scale = tl.load(b_scale_base)

        if HAS_BIAS:
            bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
            bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask,
                        mask=token_mask,
                        other=0.0,
                    )
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    if use_fp8_w8a8:
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
            else:
                accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_int8_w8a16:
            accumulator = accumulator * b_scale
        elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
            accumulator = accumulator * a_scale * b_scale

        if HAS_BIAS:
            accumulator += bias[None, :]

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
            accumulator *= moe_weight[:, None]

        accumulator = accumulator.to(compute_type)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


    @triton.jit
    def fastllm_merge_moe_fp8_sum_output_kernel(
        output_cache,
        output,
        batch,
        topk,
        hidden,
        COMPUTE_TYPE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        token_ids = offs // hidden
        hidden_ids = offs - token_ids * hidden
        mask = token_ids < batch
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for slot in range(0, topk):
            task_ids = token_ids * topk + slot
            values = tl.load(
                output_cache + task_ids * hidden + hidden_ids,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += values
        tl.store(output + offs, acc.to(COMPUTE_TYPE), mask=mask)


def default_cache_dir():
    value = os.environ.get("FASTLLM_CUDA_TRITON_CACHE_DIR")
    if value:
        return Path(value).expanduser()
    value = os.environ.get("XDG_CACHE_HOME")
    if value:
        return Path(value).expanduser() / "fastllm" / "triton"
    value = os.environ.get("HOME")
    if value:
        return Path(value).expanduser() / ".cache" / "fastllm" / "triton"
    return Path("/tmp") / "fastllm-triton"


def require_int(payload, name, fallback=None):
    value = payload.get(name, fallback)
    if value is None:
        raise ValueError(f"missing required field: {name}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_nonnegative_int(payload, name, fallback=0):
    value = int(payload.get(name, fallback))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def require_dtype(payload, name):
    value = str(payload.get(name, ""))
    if value not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"{name} must be fp16, bf16, or fp32")
    return value


def linear_cache_paths(payload):
    arch = require_int(payload, "arch")
    input_dtype = require_dtype(payload, "input_dtype")
    weight_dtype = require_dtype(payload, "weight_dtype")
    output_dtype = require_dtype(payload, "output_dtype")
    has_bias = 1 if bool(payload.get("has_bias", False)) else 0
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 64)
    block_k = require_int(payload, "block_k", 64)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"linear_{input_dtype}_{weight_dtype}_{output_dtype}_bias{has_bias}"
        f"_sm{arch}_bm{block_m}_bn{block_n}_bk{block_k}"
        f"_nw{num_warps}_ns{num_stages}"
    )
    return cache_dir / f"{name}.cubin", cache_dir / f"{name}.json"


MERGE_MOE_FP8_KERNEL_ORDER = (
    "init_count",
    "zero_route",
    "count",
    "prefix",
    "fill_sorted",
    "scatter_blocks",
    "quant_input",
    "gateup",
    "gateup_fused",
    "swiglu_quant",
    "down",
    "sum_output",
)


LINEAR_FP8_BLOCK128_KERNEL_ORDER = ("quant_input", "matmul")


def linear_fp8_block128_matmul_variant(payload):
    variant = str(payload.get("matmul_variant") or "fastllm").strip().lower()
    if variant not in {"fastllm", "strided"}:
        raise ValueError("matmul_variant must be fastllm or strided")
    return variant


def linear_fp8_block128_cache_paths(payload):
    arch = require_int(payload, "arch")
    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    weight_layout = str(payload.get("weight_layout") or "packed")
    if weight_layout not in {"packed", "separate"}:
        raise ValueError("weight_layout must be packed or separate")
    has_bias = 1 if bool(payload.get("has_bias", False)) else 0
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 128)
    block_k = require_int(payload, "block_k", 128)
    group_size_m = require_int(payload, "group_size_m", 32)
    quant_num_warps = require_int(payload, "quant_num_warps", 4)
    matmul_num_warps = require_int(payload, "matmul_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    if block_k != 128:
        raise ValueError("FP8 block128 linear requires block_k=128")
    matmul_variant = linear_fp8_block128_matmul_variant(payload)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    if matmul_variant == "fastllm":
        name = (
            f"linear_fp8_block128_v5_{weight_layout}_{input_dtype}_bias{has_bias}_sm{arch}"
            f"_bm{block_m}_bn{block_n}_bk{block_k}_gsm{group_size_m}"
            f"_qnw{quant_num_warps}_mnw{matmul_num_warps}_ns{num_stages}"
        )
    else:
        name = (
            f"linear_fp8_block128_strided_v4_{weight_layout}_{input_dtype}_bias{has_bias}_sm{arch}"
            f"_bm{block_m}_bn{block_n}_bk{block_k}_gsm{group_size_m}"
            f"_qnw{quant_num_warps}_mnw{matmul_num_warps}_ns{num_stages}"
        )
    cubins = {
        key: cache_dir / f"{name}_{key}.cubin"
        for key in LINEAR_FP8_BLOCK128_KERNEL_ORDER
    }
    return cubins, cache_dir / f"{name}.json"


def deepseek_v4_fp8_woa_cache_paths(payload):
    arch = require_int(payload, "arch")
    num_tokens = require_int(payload, "num_tokens", 1)
    num_groups = require_int(payload, "num_groups", 8)
    out_rank = require_int(payload, "out_rank", 1024)
    hidden_size = require_int(payload, "hidden_size", 4096)
    block_tokens = require_int(payload, "block_tokens", 16)
    block_out = require_int(payload, "block_out", 128)
    block_hidden = require_int(payload, "block_hidden", 128)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"deepseek_v4_fp8_woa_v1_sm{arch}"
        f"_t{num_tokens}_g{num_groups}_r{out_rank}_h{hidden_size}"
        f"_bt{block_tokens}_bo{block_out}_bh{block_hidden}"
        f"_nw{num_warps}_ns{num_stages}"
    )
    return cache_dir / f"{name}.cubin", cache_dir / f"{name}.json"


def deepseek_v4_sparse_decode_cache_paths(payload):
    arch = require_int(payload, "arch")
    batch = require_int(payload, "batch", 1)
    num_heads = require_int(payload, "num_heads", 64)
    head_dim = require_int(payload, "head_dim", 512)
    window_size = require_int(payload, "window_size", 128)
    compress_ratio = require_nonnegative_int(payload, "compress_ratio", 0)
    head_block = require_int(payload, "head_block", 1)
    block_d = require_int(payload, "block_d", 512)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"deepseek_v4_sparse_decode_v1_sm{arch}"
        f"_b{batch}_h{num_heads}_d{head_dim}_w{window_size}_cr{compress_ratio}"
        f"_hb{head_block}_bd{block_d}_nw{num_warps}_ns{num_stages}"
    )
    return cache_dir / f"{name}.cubin", cache_dir / f"{name}.json"


def merge_moe_fp8_cache_paths(payload):
    arch = require_int(payload, "arch")
    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    route_block_t = require_int(payload, "route_block_t", 1024)
    max_experts = require_int(payload, "max_experts", 256)
    topk = require_int(payload, "topk", 8)
    group_block_m = require_int(payload, "group_block_m", 16)
    group_block_n = require_int(payload, "group_block_n", 128)
    group_block_k = require_int(payload, "group_block_k", 128)
    group_size_m = require_nonnegative_int(payload, "group_size_m", 8)
    hidden = int(payload.get("hidden", 0) or 0)
    inter = int(payload.get("inter", 0) or 0)
    if group_block_n != group_block_k:
        raise ValueError("group_block_n and group_block_k must match for W8A8 FP8 MoE")
    route_num_warps = require_int(payload, "route_num_warps", 4)
    group_num_warps = require_int(payload, "group_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"merge_moe_fp8_v34_{input_dtype}_sm{arch}"
        f"_rt{route_block_t}_me{max_experts}_tk{topk}"
        f"_h{hidden}_i{inter}"
        f"_gm{group_block_m}_gn{group_block_n}_gk{group_block_k}"
        f"_gsm{group_size_m}"
        f"_rnw{route_num_warps}_gnw{group_num_warps}"
        f"_ns{num_stages}"
    )
    cubins = {key: cache_dir / f"{name}_{key}.cubin" for key in MERGE_MOE_FP8_KERNEL_ORDER}
    return cubins, cache_dir / f"{name}.json"


def compile_linear(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    input_dtype = require_dtype(payload, "input_dtype")
    weight_dtype = require_dtype(payload, "weight_dtype")
    output_dtype = require_dtype(payload, "output_dtype")
    if input_dtype != weight_dtype or input_dtype != output_dtype:
        raise ValueError("this prototype only supports matching input, weight, and output dtypes")

    arch = require_int(payload, "arch")
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 64)
    block_k = require_int(payload, "block_k", 64)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    has_bias = bool(payload.get("has_bias", False))

    cubin_path, meta_path = linear_cache_paths(payload)
    if cubin_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    cubin_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "a_ptr": f"*{input_dtype}",
        "b_ptr": f"*{weight_dtype}",
        "bias_ptr": "*fp32",
        "c_ptr": f"*{output_dtype}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "HAS_BIAS": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }
    constexprs = {
        "HAS_BIAS": has_bias,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
    }

    attrs = {}
    src = ASTSource(fn=fastllm_linear_kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    cubin_path.write_bytes(ccinfo.asm[backend.binary_ext])

    meta = {
        "ok": True,
        "op": "linear",
        "cubin": str(cubin_path),
        "kernel": ccinfo.metadata.name,
        "shared": int(ccinfo.metadata.shared),
        "num_warps": int(ccinfo.metadata.num_warps),
        "num_stages": int(ccinfo.metadata.num_stages),
        "warp_size": int(ccinfo.metadata.warp_size),
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "arch": arch,
        "input_dtype": input_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
        "has_bias": has_bias,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def _compile_cubin(
    fn, signature, constexprs, arch, num_warps, num_stages, cubin_path,
    extra_divisible_by_16=None,
):
    extra_divisible_by_16 = set(extra_divisible_by_16 or ())
    attrs = {
        (i,): [["tt.divisibility", 16]]
        for i, name in enumerate(fn.arg_names)
        if str(signature.get(name, "")).startswith("*") or name in extra_divisible_by_16
    }
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    cubin_path.write_bytes(ccinfo.asm[backend.binary_ext])
    return ccinfo


def compile_linear_fp8_block128(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    arch = require_int(payload, "arch")
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 128)
    block_k = require_int(payload, "block_k", 128)
    group_size_m = require_int(payload, "group_size_m", 32)
    quant_num_warps = require_int(payload, "quant_num_warps", 4)
    matmul_num_warps = require_int(payload, "matmul_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    has_bias = bool(payload.get("has_bias", False))
    weight_layout = str(payload.get("weight_layout") or "packed")
    if weight_layout not in {"packed", "separate"}:
        raise ValueError("weight_layout must be packed or separate")
    packed_weight = weight_layout == "packed"
    matmul_variant = linear_fp8_block128_matmul_variant(payload)
    if matmul_variant == "strided" and packed_weight:
        raise ValueError("strided FP8 block128 linear variant requires separate weight scales")
    if matmul_variant == "strided" and has_bias:
        raise ValueError("strided FP8 block128 linear variant does not support bias")
    if block_k != 128:
        raise ValueError("FP8 block128 linear requires block_k=128")

    cubin_paths, meta_path = linear_fp8_block128_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    compute_type = tl.float16 if input_dtype == "fp16" else tl.bfloat16

    quant_signature = {
        "input_ptr": f"*{input_dtype}",
        "q_ptr": "*fp8e4nv",
        "scale_ptr": "*fp32",
        "batch": "i32",
        "hidden": "i32",
        "BLOCK_K": "constexpr",
    }
    quant_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_quant_input_kernel,
        quant_signature,
        {"BLOCK_K": block_k},
        arch,
        quant_num_warps,
        num_stages,
        cubin_paths["quant_input"],
    )

    if matmul_variant == "strided":
        matmul_fn = fastllm_linear_fp8_block128_strided_matmul_kernel
        matmul_signature = {
            "A": "*fp8e4nv",
            "B": "*fp8e4nv",
            "C": f"*{input_dtype}",
            "As": "*fp32",
            "Bs": "*fp32",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "group_n": "i32",
            "group_k": "i32",
            "stride_am": "i32",
            "stride_bn": "i32",
            "stride_cm": "i32",
            "stride_As_m": "i32",
            "stride_Bs_n": "i32",
            "BLOCK_SIZE_M": "constexpr",
            "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr",
            "GROUP_SIZE_M": "constexpr",
        }
        matmul_constexprs = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_size_m,
        }
        matmul_extra_divisible_by_16 = {
            "K",
            "group_n",
            "group_k",
            "stride_am",
            "stride_bn",
        }
    else:
        matmul_fn = fastllm_linear_fp8_block128_matmul_kernel
        matmul_signature = {
            "a_ptr": "*fp8e4nv",
            "a_scale_ptr": "*fp32",
            "b_ptr": "*fp8e4nv",
            "b_scale_ptr": "*fp32",
            "bias_ptr": "*fp32",
            "c_ptr": f"*{input_dtype}",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "PER_ROW": "i32",
            "SCALE_COLS": "i32",
            "HAS_BIAS": "constexpr",
            "PACKED_WEIGHT": "constexpr",
            "COMPUTE_TYPE": "constexpr",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
            "WEIGHT_BLOCK_N": "constexpr",
            "WEIGHT_BLOCK_K": "constexpr",
            "GROUP_SIZE_M": "constexpr",
        }
        matmul_constexprs = {
            "HAS_BIAS": has_bias,
            "PACKED_WEIGHT": packed_weight,
            "COMPUTE_TYPE": compute_type,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "WEIGHT_BLOCK_N": 128,
            "WEIGHT_BLOCK_K": 128,
            "GROUP_SIZE_M": group_size_m,
        }
        matmul_extra_divisible_by_16 = None
    matmul_ccinfo = _compile_cubin(
        matmul_fn,
        matmul_signature,
        matmul_constexprs,
        arch,
        matmul_num_warps,
        num_stages,
        cubin_paths["matmul"],
        extra_divisible_by_16=matmul_extra_divisible_by_16,
    )

    ccinfos = {
        "quant_input": quant_ccinfo,
        "matmul": matmul_ccinfo,
    }
    kernels = {}
    for key in LINEAR_FP8_BLOCK128_KERNEL_ORDER:
        ccinfo = ccinfos[key]
        kernels[key] = {
            "cubin": str(cubin_paths[key]),
            "kernel": ccinfo.metadata.name,
            "shared": int(ccinfo.metadata.shared),
            "num_warps": int(ccinfo.metadata.num_warps),
        }

    meta = {
        "ok": True,
        "op": "linear_fp8_block128",
        "kernels": kernels,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "weight_block_n": 128,
        "weight_block_k": 128,
        "group_size_m": group_size_m,
        "quant_num_warps": quant_num_warps,
        "matmul_num_warps": matmul_num_warps,
        "num_stages": num_stages,
        "arch": arch,
        "input_dtype": input_dtype,
        "weight_layout": weight_layout,
        "packed_weight": packed_weight,
        "has_bias": has_bias,
        "matmul_variant": matmul_variant,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_deepseek_v4_fp8_woa(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    if arch != 89 and arch not in {120, 121}:
        raise ValueError("DeepSeek-V4 FP8 WoA supports SM89 and SM12x")
    num_tokens = require_int(payload, "num_tokens", 1)
    num_groups = require_int(payload, "num_groups", 8)
    out_rank = require_int(payload, "out_rank", 1024)
    hidden_size = require_int(payload, "hidden_size", 4096)
    block_tokens = require_int(payload, "block_tokens", 16)
    block_out = require_int(payload, "block_out", 128)
    block_hidden = require_int(payload, "block_hidden", 128)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    if num_tokens > block_tokens:
        raise ValueError("DeepSeek-V4 FP8 WoA currently requires num_tokens <= block_tokens")
    if block_out != 128 or block_hidden != 128:
        raise ValueError("DeepSeek-V4 FP8 WoA requires 128x128 weight scales")
    if out_rank % block_out != 0 or hidden_size % block_hidden != 0:
        raise ValueError("DeepSeek-V4 FP8 WoA shape must be divisible by its block sizes")

    cubin_path, meta_path = deepseek_v4_fp8_woa_cache_paths(payload)
    if cubin_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "a_ptr": "*fp8e4nv",
        "a_scale_ptr": "*fp32",
        "b_ptr": "*fp8e4nv",
        "b_scale_ptr": "*fp32",
        "out_ptr": "*bf16",
        "NUM_TOKENS": "constexpr",
        "NUM_GROUPS": "constexpr",
        "OUT_RANK": "constexpr",
        "HIDDEN_SIZE": "constexpr",
        "BLOCK_TOKENS": "constexpr",
        "BLOCK_OUT": "constexpr",
        "BLOCK_HIDDEN": "constexpr",
        "UPCAST_FP8": "constexpr",
    }
    constexprs = {
        "NUM_TOKENS": num_tokens,
        "NUM_GROUPS": num_groups,
        "OUT_RANK": out_rank,
        "HIDDEN_SIZE": hidden_size,
        "BLOCK_TOKENS": block_tokens,
        "BLOCK_OUT": block_out,
        "BLOCK_HIDDEN": block_hidden,
        "UPCAST_FP8": arch == 89,
    }
    ccinfo = _compile_cubin(
        fastllm_deepseek_v4_fp8_woa_kernel,
        signature,
        constexprs,
        arch,
        num_warps,
        num_stages,
        cubin_path,
    )
    meta = {
        "ok": True,
        "op": "deepseek_v4_fp8_woa",
        "cubin": str(cubin_path),
        "kernel": ccinfo.metadata.name,
        "shared": int(ccinfo.metadata.shared),
        "num_warps": int(ccinfo.metadata.num_warps),
        "num_stages": num_stages,
        "arch": arch,
        "num_tokens": num_tokens,
        "num_groups": num_groups,
        "out_rank": out_rank,
        "hidden_size": hidden_size,
        "block_tokens": block_tokens,
        "block_out": block_out,
        "block_hidden": block_hidden,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_deepseek_v4_sparse_decode(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    if arch != 89 and arch not in {120, 121}:
        raise ValueError("DeepSeek-V4 sparse decode supports SM89 and SM12x")
    batch = require_int(payload, "batch", 1)
    num_heads = require_int(payload, "num_heads", 64)
    head_dim = require_int(payload, "head_dim", 512)
    window_size = require_int(payload, "window_size", 128)
    compress_ratio = require_nonnegative_int(payload, "compress_ratio", 0)
    head_block = require_int(payload, "head_block", 1)
    block_d = require_int(payload, "block_d", 512)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    if batch != 1:
        raise ValueError("DeepSeek-V4 sparse decode currently requires batch=1")
    if head_block not in {1, 2, 4}:
        raise ValueError("DeepSeek-V4 sparse decode head_block must be 1, 2, or 4")
    if block_d < head_dim or block_d > 1024 or (block_d & (block_d - 1)) != 0:
        raise ValueError("DeepSeek-V4 sparse decode block_d must be a power of two covering head_dim")
    if num_heads <= 0 or head_dim <= 0 or window_size <= 0:
        raise ValueError("DeepSeek-V4 sparse decode dimensions must be positive")
    cubin_path, meta_path = deepseek_v4_sparse_decode_cache_paths(payload)
    if cubin_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "q_ptr": "*bf16",
        "window_kv_ptr": "*fp32",
        "compressed_kv_ptr": "*bf16",
        "sink_ptr": "*fp32",
        "decode_meta_ptr": "*i32",
        "output_ptr": "*fp32",
        "softmax_scale": "fp32",
        "BATCH": "constexpr",
        "NUM_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "WINDOW_SIZE": "constexpr",
        "COMPRESS_RATIO": "constexpr",
        "HEAD_BLOCK": "constexpr",
        "BLOCK_D": "constexpr",
    }
    constexprs = {
        "BATCH": batch,
        "NUM_HEADS": num_heads,
        "HEAD_DIM": head_dim,
        "WINDOW_SIZE": window_size,
        "COMPRESS_RATIO": compress_ratio,
        "HEAD_BLOCK": head_block,
        "BLOCK_D": block_d,
    }
    ccinfo = _compile_cubin(
        fastllm_deepseek_v4_sparse_decode_kernel,
        signature,
        constexprs,
        arch,
        num_warps,
        num_stages,
        cubin_path,
    )
    meta = {
        "ok": True,
        "op": "deepseek_v4_sparse_decode",
        "cubin": str(cubin_path),
        "kernel": ccinfo.metadata.name,
        "shared": int(ccinfo.metadata.shared),
        "num_warps": int(ccinfo.metadata.num_warps),
        "num_stages": num_stages,
        "arch": arch,
        "batch": batch,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "compress_ratio": compress_ratio,
        "head_block": head_block,
        "block_d": block_d,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_merge_moe_fp8(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    arch = require_int(payload, "arch")
    route_block_t = require_int(payload, "route_block_t", 1024)
    max_experts = require_int(payload, "max_experts", 256)
    topk = require_int(payload, "topk", 8)
    group_block_m = require_int(payload, "group_block_m", 16)
    group_block_n = require_int(payload, "group_block_n", 128)
    group_block_k = require_int(payload, "group_block_k", 128)
    group_size_m = require_nonnegative_int(payload, "group_size_m", 8)
    hidden = int(payload.get("hidden", 0) or 0)
    inter = int(payload.get("inter", 0) or 0)
    if group_block_n != group_block_k:
        raise ValueError("group_block_n and group_block_k must match for W8A8 FP8 MoE")
    if hidden <= 0 or inter <= 0:
        raise ValueError("hidden and inter are required for merge_moe_fp8 kernels")
    route_num_warps = require_int(payload, "route_num_warps", 4)
    group_num_warps = require_int(payload, "group_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)

    cubin_paths, meta_path = merge_moe_fp8_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    compute_type = tl.float16 if input_dtype == "fp16" else tl.bfloat16

    init_count_signature = {
        "indices_ptr": "*i32",
        "expert_counts": "*i32",
        "expert_offsets": "*i32",
        "expert_cursors": "*i32",
        "expert_block_offsets": "*i32",
        "total_blocks_ptr": "*i32",
        "total_tasks": "i32",
        "experts": "i32",
        "BLOCK_T": "constexpr",
        "BLOCK_E": "constexpr",
    }
    init_count_constexprs = {"BLOCK_T": route_block_t, "BLOCK_E": max_experts}
    init_count_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_init_count_kernel,
        init_count_signature,
        init_count_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["init_count"],
    )

    route_block_signature = {
        "indices_ptr": "*i32",
        "expert_counts": "*i32",
        "total_tasks": "i32",
        "experts": "i32",
        "BLOCK_T": "constexpr",
    }
    route_block_constexprs = {"BLOCK_T": route_block_t}

    zero_signature = {
        "expert_counts": "*i32",
        "expert_offsets": "*i32",
        "expert_cursors": "*i32",
        "expert_block_offsets": "*i32",
        "total_blocks_ptr": "*i32",
        "experts": "i32",
        "BLOCK_E": "constexpr",
    }
    zero_constexprs = {"BLOCK_E": max_experts}
    zero_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_zero_route_kernel,
        zero_signature,
        zero_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["zero_route"],
    )

    count_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_count_kernel,
        route_block_signature,
        route_block_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["count"],
    )

    prefix_signature = {
        "expert_counts": "*i32",
        "expert_offsets": "*i32",
        "expert_cursors": "*i32",
        "expert_block_offsets": "*i32",
        "total_blocks_ptr": "*i32",
        "experts": "i32",
        "BLOCK_E": "constexpr",
        "BLOCK_M": "constexpr",
    }
    prefix_constexprs = {"BLOCK_E": max_experts, "BLOCK_M": group_block_m}
    prefix_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_prefix_kernel,
        prefix_signature,
        prefix_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["prefix"],
    )

    fill_sorted_signature = {
        "sorted_tasks": "*i32",
        "expert_offsets": "*i32",
        "total_tasks": "i32",
        "experts": "i32",
        "BLOCK_T": "constexpr",
    }
    fill_sorted_constexprs = {"BLOCK_T": route_block_t}
    fill_sorted_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_fill_sorted_kernel,
        fill_sorted_signature,
        fill_sorted_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["fill_sorted"],
    )

    scatter_signature = {
        "indices_ptr": "*i32",
        "expert_offsets": "*i32",
        "expert_cursors": "*i32",
        "expert_block_offsets": "*i32",
        "sorted_tasks": "*i32",
        "block_experts": "*i32",
        "block_starts": "*i32",
        "total_tasks": "i32",
        "experts": "i32",
        "BLOCK_T": "constexpr",
        "BLOCK_M": "constexpr",
    }
    scatter_constexprs = {"BLOCK_T": route_block_t, "BLOCK_M": group_block_m}
    scatter_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_scatter_blocks_kernel,
        scatter_signature,
        scatter_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["scatter_blocks"],
    )

    quant_input_signature = {
        "input_ptr": f"*{input_dtype}",
        "q_ptr": "*fp8e4nv",
        "scale_ptr": "*fp32",
        "batch": "i32",
        "hidden": "i32",
        "BLOCK_K": "constexpr",
    }
    quant_input_constexprs = {"BLOCK_K": group_block_k}
    quant_input_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_quant_input_kernel,
        quant_input_signature,
        quant_input_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["quant_input"],
    )

    input_scale_cols = (hidden + group_block_k - 1) // group_block_k if hidden > 0 else 0
    activation_scale_cols = (inter + group_block_k - 1) // group_block_k if inter > 0 else 0
    gate_scale_rows = (inter * 2 + group_block_n - 1) // group_block_n if inter > 0 else 0
    down_scale_rows = (hidden + group_block_n - 1) // group_block_n if hidden > 0 else 0

    gateup_signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": f"*{input_dtype}",
        "b_bias_ptr": "*fp32",
        "a_scale_ptr": "*fp32",
        "b_scale_ptr": "*fp32",
        "topk_weights_ptr": "*fp32",
        "sorted_token_ids_ptr": "*i32",
        "expert_ids_ptr": "*i32",
        "num_tokens_post_padded_ptr": "*i32",
        "N": "constexpr",
        "K": "constexpr",
        "EM": "i32",
        "num_valid_tokens": "i32",
        "stride_am": "constexpr",
        "stride_ak": "constexpr",
        "stride_be": "constexpr",
        "stride_bk": "constexpr",
        "stride_bn": "constexpr",
        "stride_cm": "constexpr",
        "stride_cn": "constexpr",
        "stride_asm": "constexpr",
        "stride_ask": "constexpr",
        "stride_bse": "constexpr",
        "stride_bsk": "constexpr",
        "stride_bsn": "constexpr",
        "stride_bbe": "constexpr",
        "stride_bbn": "constexpr",
        "group_n": "constexpr",
        "group_k": "constexpr",
        "naive_block_assignment": "constexpr",
        "BLOCK_SIZE_M": "constexpr",
        "BLOCK_SIZE_N": "constexpr",
        "BLOCK_SIZE_K": "constexpr",
        "GROUP_SIZE_M": "constexpr",
        "SPLIT_K": "constexpr",
        "MUL_ROUTED_WEIGHT": "constexpr",
        "top_k": "constexpr",
        "compute_type": "constexpr",
        "use_fp8_w8a8": "constexpr",
        "use_int8_w8a8": "constexpr",
        "use_int8_w8a16": "constexpr",
        "per_channel_quant": "constexpr",
        "HAS_BIAS": "constexpr",
    }
    gateup_constexprs = {
        "N": inter * 2,
        "K": hidden,
        "stride_am": hidden,
        "stride_ak": 1,
        "stride_be": inter * 2 * hidden,
        "stride_bk": 1,
        "stride_bn": hidden,
        "stride_cm": inter * 2,
        "stride_cn": 1,
        "stride_asm": input_scale_cols,
        "stride_ask": 1,
        "stride_bse": gate_scale_rows * input_scale_cols,
        "stride_bsk": 1,
        "stride_bsn": input_scale_cols,
        "stride_bbe": 0,
        "stride_bbn": 0,
        "BLOCK_SIZE_M": group_block_m,
        "BLOCK_SIZE_N": group_block_n,
        "BLOCK_SIZE_K": group_block_k,
        "GROUP_SIZE_M": group_size_m,
        "group_n": group_block_n,
        "group_k": group_block_k,
        "naive_block_assignment": False,
        "SPLIT_K": 1,
        "MUL_ROUTED_WEIGHT": False,
        "top_k": topk,
        "compute_type": compute_type,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "per_channel_quant": False,
        "HAS_BIAS": False,
    }
    gateup_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_fused_matmul_kernel,
        gateup_signature,
        gateup_constexprs,
        arch,
        group_num_warps,
        num_stages,
        cubin_paths["gateup"],
    )

    fused_gateup_signature = {
        "a_ptr": "*fp8e4nv",
        "gate_ptr": "*fp8e4nv",
        "up_ptr": "*fp8e4nv",
        "c_ptr": f"*{input_dtype}",
        "a_scale_ptr": "*fp32",
        "gate_scale_ptr": "*fp32",
        "up_scale_ptr": "*fp32",
        "sorted_token_ids_ptr": "*i32",
        "expert_ids_ptr": "*i32",
        "num_tokens_post_padded_ptr": "*i32",
        "N": "constexpr",
        "K": "constexpr",
        "INTER": "constexpr",
        "EM": "i32",
        "num_valid_tokens": "i32",
        "stride_am": "constexpr",
        "stride_ak": "constexpr",
        "stride_be": "constexpr",
        "stride_bk": "constexpr",
        "stride_bn": "constexpr",
        "stride_cm": "constexpr",
        "stride_cn": "constexpr",
        "stride_asm": "constexpr",
        "stride_ask": "constexpr",
        "stride_bse": "constexpr",
        "stride_bsk": "constexpr",
        "stride_bsn": "constexpr",
        "group_n": "constexpr",
        "group_k": "constexpr",
        "BLOCK_SIZE_M": "constexpr",
        "BLOCK_SIZE_N": "constexpr",
        "BLOCK_SIZE_K": "constexpr",
        "GROUP_SIZE_M": "constexpr",
        "top_k": "constexpr",
        "compute_type": "constexpr",
    }
    fused_gateup_constexprs = {
        "N": inter * 2,
        "K": hidden,
        "INTER": inter,
        "stride_am": hidden,
        "stride_ak": 1,
        "stride_be": inter * hidden,
        "stride_bk": 1,
        "stride_bn": hidden,
        "stride_cm": inter * 2,
        "stride_cn": 1,
        "stride_asm": input_scale_cols,
        "stride_ask": 1,
        "stride_bse": ((inter + group_block_n - 1) // group_block_n) * input_scale_cols,
        "stride_bsk": 1,
        "stride_bsn": input_scale_cols,
        "BLOCK_SIZE_M": group_block_m,
        "BLOCK_SIZE_N": group_block_n,
        "BLOCK_SIZE_K": group_block_k,
        "GROUP_SIZE_M": group_size_m,
        "group_n": group_block_n,
        "group_k": group_block_k,
        "top_k": topk,
        "compute_type": compute_type,
    }
    fused_gateup_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_fused_gateup_matmul_kernel,
        fused_gateup_signature,
        fused_gateup_constexprs,
        arch,
        group_num_warps,
        num_stages,
        cubin_paths["gateup_fused"],
    )

    swiglu_quant_signature = {
        "gateup_ptr": f"*{input_dtype}",
        "c_ptr": "*fp8e4nv",
        "c_scale_ptr": "*fp32",
        "total_tasks": "i32",
        "inter": "i32",
        "COMPUTE_TYPE": "constexpr",
        "BLOCK_N": "constexpr",
    }
    swiglu_quant_constexprs = {
        "COMPUTE_TYPE": compute_type,
        "BLOCK_N": group_block_n,
    }
    swiglu_quant_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_swiglu_quant_kernel,
        swiglu_quant_signature,
        swiglu_quant_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["swiglu_quant"],
    )

    down_signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": f"*{input_dtype}",
        "b_bias_ptr": "*fp32",
        "a_scale_ptr": "*fp32",
        "b_scale_ptr": "*fp32",
        "topk_weights_ptr": "*fp32",
        "sorted_token_ids_ptr": "*i32",
        "expert_ids_ptr": "*i32",
        "num_tokens_post_padded_ptr": "*i32",
        "N": "constexpr",
        "K": "constexpr",
        "EM": "i32",
        "num_valid_tokens": "i32",
        "stride_am": "constexpr",
        "stride_ak": "constexpr",
        "stride_be": "constexpr",
        "stride_bk": "constexpr",
        "stride_bn": "constexpr",
        "stride_cm": "constexpr",
        "stride_cn": "constexpr",
        "stride_asm": "constexpr",
        "stride_ask": "constexpr",
        "stride_bse": "constexpr",
        "stride_bsk": "constexpr",
        "stride_bsn": "constexpr",
        "stride_bbe": "constexpr",
        "stride_bbn": "constexpr",
        "group_n": "constexpr",
        "group_k": "constexpr",
        "naive_block_assignment": "constexpr",
        "BLOCK_SIZE_M": "constexpr",
        "BLOCK_SIZE_N": "constexpr",
        "BLOCK_SIZE_K": "constexpr",
        "GROUP_SIZE_M": "constexpr",
        "SPLIT_K": "constexpr",
        "MUL_ROUTED_WEIGHT": "constexpr",
        "top_k": "constexpr",
        "compute_type": "constexpr",
        "use_fp8_w8a8": "constexpr",
        "use_int8_w8a8": "constexpr",
        "use_int8_w8a16": "constexpr",
        "per_channel_quant": "constexpr",
        "HAS_BIAS": "constexpr",
    }
    down_constexprs = {
        "N": hidden,
        "K": inter,
        "stride_am": inter,
        "stride_ak": 1,
        "stride_be": hidden * inter,
        "stride_bk": 1,
        "stride_bn": inter,
        "stride_cm": hidden,
        "stride_cn": 1,
        "stride_asm": activation_scale_cols,
        "stride_ask": 1,
        "stride_bse": down_scale_rows * activation_scale_cols,
        "stride_bsk": 1,
        "stride_bsn": activation_scale_cols,
        "stride_bbe": 0,
        "stride_bbn": 0,
        "BLOCK_SIZE_M": group_block_m,
        "BLOCK_SIZE_N": group_block_n,
        "BLOCK_SIZE_K": group_block_k,
        "GROUP_SIZE_M": group_size_m,
        "group_n": group_block_n,
        "group_k": group_block_k,
        "naive_block_assignment": False,
        "SPLIT_K": 1,
        "MUL_ROUTED_WEIGHT": True,
        "top_k": 1,
        "compute_type": compute_type,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "per_channel_quant": False,
        "HAS_BIAS": False,
    }
    down_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_fused_matmul_kernel,
        down_signature,
        down_constexprs,
        arch,
        group_num_warps,
        num_stages,
        cubin_paths["down"],
    )

    sum_output_signature = {
        "output_cache": f"*{input_dtype}",
        "output": f"*{input_dtype}",
        "batch": "i32",
        "topk": "i32",
        "hidden": "i32",
        "COMPUTE_TYPE": "constexpr",
        "BLOCK_T": "constexpr",
    }
    sum_output_constexprs = {"COMPUTE_TYPE": compute_type, "BLOCK_T": route_block_t}
    sum_output_ccinfo = _compile_cubin(
        fastllm_merge_moe_fp8_sum_output_kernel,
        sum_output_signature,
        sum_output_constexprs,
        arch,
        route_num_warps,
        num_stages,
        cubin_paths["sum_output"],
    )

    ccinfos = {
        "init_count": init_count_ccinfo,
        "zero_route": zero_ccinfo,
        "count": count_ccinfo,
        "prefix": prefix_ccinfo,
        "fill_sorted": fill_sorted_ccinfo,
        "scatter_blocks": scatter_ccinfo,
        "quant_input": quant_input_ccinfo,
        "gateup": gateup_ccinfo,
        "gateup_fused": fused_gateup_ccinfo,
        "swiglu_quant": swiglu_quant_ccinfo,
        "down": down_ccinfo,
        "sum_output": sum_output_ccinfo,
    }
    kernels = {}
    for key in MERGE_MOE_FP8_KERNEL_ORDER:
        ccinfo = ccinfos[key]
        kernels[key] = {
            "cubin": str(cubin_paths[key]),
            "kernel": ccinfo.metadata.name,
            "shared": int(ccinfo.metadata.shared),
            "num_warps": int(ccinfo.metadata.num_warps),
        }

    meta = {
        "ok": True,
        "op": "merge_moe_fp8",
        "kernels": kernels,
        "route_block_t": route_block_t,
        "max_experts": max_experts,
        "group_block_m": group_block_m,
        "group_block_n": group_block_n,
        "group_block_k": group_block_k,
        "group_size_m": group_size_m,
        "route_num_warps": route_num_warps,
        "group_num_warps": group_num_warps,
        "num_stages": num_stages,
        "arch": arch,
        "input_dtype": input_dtype,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def handle_compile(payload):
    op = payload.get("op")
    with _compile_lock:
        if op == "linear":
            return compile_linear(payload)
        if op == "linear_fp8_block128":
            return compile_linear_fp8_block128(payload)
        if op == "deepseek_v4_fp8_woa":
            return compile_deepseek_v4_fp8_woa(payload)
        if op == "deepseek_v4_sparse_decode":
            return compile_deepseek_v4_sparse_decode(payload)
        if op == "merge_moe_fp8":
            return compile_merge_moe_fp8(payload)
        raise ValueError(f"unsupported op: {op}")


class Handler(BaseHTTPRequestHandler):
    server_version = "fastllm-triton/0.1"

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def write_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.write_json(200, {"ok": True, "triton": triton is not None})
            return
        self.write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/compile":
            self.write_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = handle_compile(payload)
            self.write_json(200, response)
        except Exception as exc:
            self.write_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48989)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.verbose = args.verbose
    print(f"fastllm triton server listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
