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
    from triton.language.extra import libdevice
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
except Exception as exc:  # pragma: no cover - this is reported through /compile.
    triton = None
    tl = None
    libdevice = None
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
    def fastllm_chunk_gdn_prefill_h_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        k_cumdecay_ptr,
        state_ptr,
        next_state_ptr,
        h_ptr,
        v_new_ptr,
        row_scale_ptr,
        state_scale_ptr,
        CHUNKS: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        USE_PRECOMPUTED_SCALE: tl.constexpr,
    ):
        """Build chunk states and updated values for gated-delta prefill.

        FastLLM stores each recurrent state as [K, V].  One Triton program owns
        a V tile for one batch/head pair, keeps the state tile in FP32 across
        chunks, and uses tensor-core dot products for both recurrence GEMMs.
        """
        v_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        t_offsets = tl.arange(0, CHUNK_SIZE)
        k_offsets = tl.arange(0, 64)
        v_mask = v_offsets < V_DIM

        state_base = batch_head * K_DIM * V_DIM
        state_offsets_0 = (
            state_base
            + k_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        state_0 = tl.load(
            state_ptr + state_offsets_0,
            mask=v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        state_offsets_1 = state_offsets_0 + 64 * V_DIM
        state_1 = tl.load(
            state_ptr + state_offsets_1,
            mask=v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for chunk in range(0, CHUNKS):
            chunk_index = batch_head * CHUNKS + chunk
            k_base = chunk_index * CHUNK_SIZE * K_DIM
            v_base = chunk_index * CHUNK_SIZE * V_DIM
            g_base = chunk_index * CHUNK_SIZE
            h_base = chunk_index * K_DIM * V_DIM

            h_offsets_0 = (
                h_base
                + k_offsets[:, None] * V_DIM
                + v_offsets[None, :]
            )
            tl.store(
                h_ptr + h_offsets_0,
                state_0.to(h_ptr.dtype.element_ty),
                mask=v_mask[None, :],
            )
            tl.store(
                h_ptr + h_offsets_0 + 64 * V_DIM,
                state_1.to(h_ptr.dtype.element_ty),
                mask=v_mask[None, :],
            )

            k_cum_offsets_0 = (
                k_base
                + t_offsets[:, None] * K_DIM
                + k_offsets[None, :]
            )
            k_cum_0 = tl.load(k_cumdecay_ptr + k_cum_offsets_0)
            k_cum_1 = tl.load(k_cumdecay_ptr + k_cum_offsets_0 + 64)
            v_offsets_2d = (
                v_base
                + t_offsets[:, None] * V_DIM
                + v_offsets[None, :]
            )
            v_value = tl.load(
                v_ptr + v_offsets_2d,
                mask=v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v_new = v_value - tl.dot(
                k_cum_0, state_0.to(k_cum_0.dtype)
            )
            v_new -= tl.dot(
                k_cum_1, state_1.to(k_cum_1.dtype)
            )
            v_new_half = v_new.to(v_new_ptr.dtype.element_ty)
            tl.store(
                v_new_ptr + v_offsets_2d,
                v_new_half,
                mask=v_mask[None, :],
            )

            if USE_PRECOMPUTED_SCALE:
                row_scale = tl.load(
                    row_scale_ptr + g_base + t_offsets
                )
                state_scale = tl.load(
                    state_scale_ptr + chunk_index
                )
            else:
                g = tl.load(
                    g_ptr + g_base + t_offsets
                ).to(tl.float32)
                g_last = tl.load(
                    g_ptr + g_base + CHUNK_SIZE - 1
                ).to(tl.float32)
                row_scale = tl.exp(g_last - g)
                state_scale = tl.exp(g_last)
            state_0 *= state_scale
            state_1 *= state_scale

            k_0 = tl.load(k_ptr + k_cum_offsets_0)
            k_1 = tl.load(k_ptr + k_cum_offsets_0 + 64)
            k_scaled_0 = (
                k_0.to(tl.float32) * row_scale[:, None]
            ).to(k_0.dtype)
            k_scaled_1 = (
                k_1.to(tl.float32) * row_scale[:, None]
            ).to(k_1.dtype)
            state_0 += tl.dot(
                tl.trans(k_scaled_0), v_new_half
            )
            state_1 += tl.dot(
                tl.trans(k_scaled_1), v_new_half
            )

        tl.store(
            next_state_ptr + state_offsets_0,
            state_0.to(next_state_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )
        tl.store(
            next_state_ptr + state_offsets_1,
            state_1.to(next_state_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )


    @triton.jit
    def fastllm_chunk_gdn_prefill_o_kernel(
        q_ptr,
        g_ptr,
        attn_ptr,
        decay_mask_ptr,
        h_ptr,
        v_new_ptr,
        output_ptr,
        CHUNKS: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        APPLY_DECAY_MASK: tl.constexpr,
    ):
        """Compute chunk outputs from saved states and updated values."""
        v_block = tl.program_id(0)
        chunk = tl.program_id(1)
        batch_head = tl.program_id(2)
        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        t_offsets = tl.arange(0, CHUNK_SIZE)
        k_offsets = tl.arange(0, 64)
        v_mask = v_offsets < V_DIM

        chunk_index = batch_head * CHUNKS + chunk
        q_base = chunk_index * CHUNK_SIZE * K_DIM
        v_base = chunk_index * CHUNK_SIZE * V_DIM
        g_base = chunk_index * CHUNK_SIZE
        attn_base = chunk_index * CHUNK_SIZE * CHUNK_SIZE
        h_base = chunk_index * K_DIM * V_DIM

        q_offsets_0 = (
            q_base
            + t_offsets[:, None] * K_DIM
            + k_offsets[None, :]
        )
        q_0 = tl.load(q_ptr + q_offsets_0)
        q_1 = tl.load(q_ptr + q_offsets_0 + 64)
        g = tl.load(g_ptr + g_base + t_offsets).to(tl.float32)
        q_scale = tl.exp(g)
        q_scaled_0 = (
            q_0.to(tl.float32) * q_scale[:, None]
        ).to(q_0.dtype)
        q_scaled_1 = (
            q_1.to(tl.float32) * q_scale[:, None]
        ).to(q_1.dtype)

        h_offsets_0 = (
            h_base
            + k_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        h_0 = tl.load(
            h_ptr + h_offsets_0,
            mask=v_mask[None, :],
            other=0.0,
        )
        h_1 = tl.load(
            h_ptr + h_offsets_0 + 64 * V_DIM,
            mask=v_mask[None, :],
            other=0.0,
        )
        output = tl.dot(q_scaled_0, h_0)
        output += tl.dot(q_scaled_1, h_1)

        attn_offsets = (
            attn_base
            + t_offsets[:, None] * CHUNK_SIZE
            + t_offsets[None, :]
        )
        attn = tl.load(attn_ptr + attn_offsets)
        if APPLY_DECAY_MASK:
            decay_mask = tl.load(decay_mask_ptr + attn_offsets)
            causal_mask = (
                t_offsets[None, :] <= t_offsets[:, None]
            )
            attn = tl.where(
                causal_mask,
                (attn * decay_mask).to(
                    attn_ptr.dtype.element_ty
                ),
                0.0,
            )
        v_offsets_2d = (
            v_base
            + t_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        v_new = tl.load(
            v_new_ptr + v_offsets_2d,
            mask=v_mask[None, :],
            other=0.0,
        )
        output += tl.dot(attn, v_new)
        tl.store(
            output_ptr + v_offsets_2d,
            output.to(output_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )


    @triton.jit
    def fastllm_chunk_gdn_varlen_prefill_h_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        k_cumdecay_ptr,
        state_ptr,
        next_state_ptr,
        chunk_offsets_ptr,
        h_ptr,
        v_new_ptr,
        row_scale_ptr,
        state_scale_ptr,
        total_chunks,
        key_heads,
        heads,
        MAX_CHUNKS: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        USE_PRECOMPUTED_SCALE: tl.constexpr,
    ):
        """Build recurrent states for packed variable-length GDN chunks.

        Packed tensors have layout [1, heads, total_chunks, 64, dim].  Chunk
        offsets delimit each request in the shared chunk axis.  One program
        owns one request/head/V tile, preserving the required sequential
        recurrence without padding every request to the global maximum.
        """
        v_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_index = batch_head // heads
        head = batch_head - batch_index * heads
        key_head = head * key_heads // heads
        chunk_begin = tl.load(chunk_offsets_ptr + batch_index)
        chunk_end = tl.load(chunk_offsets_ptr + batch_index + 1)
        chunk_count = chunk_end - chunk_begin

        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        t_offsets = tl.arange(0, CHUNK_SIZE)
        k_offsets = tl.arange(0, 64)
        v_mask = v_offsets < V_DIM

        state_base = batch_head * K_DIM * V_DIM
        state_offsets_0 = (
            state_base
            + k_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        state_0 = tl.load(
            state_ptr + state_offsets_0,
            mask=v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        state_offsets_1 = state_offsets_0 + 64 * V_DIM
        state_1 = tl.load(
            state_ptr + state_offsets_1,
            mask=v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for relative_chunk in range(0, MAX_CHUNKS):
            if relative_chunk < chunk_count:
                global_chunk = chunk_begin + relative_chunk
                chunk_index = head * total_chunks + global_chunk
                key_chunk_index = key_head * total_chunks + global_chunk
                k_base = key_chunk_index * CHUNK_SIZE * K_DIM
                k_cum_base = chunk_index * CHUNK_SIZE * K_DIM
                v_base = chunk_index * CHUNK_SIZE * V_DIM
                g_base = chunk_index * CHUNK_SIZE
                h_base = chunk_index * K_DIM * V_DIM

                h_offsets_0 = (
                    h_base
                    + k_offsets[:, None] * V_DIM
                    + v_offsets[None, :]
                )
                tl.store(
                    h_ptr + h_offsets_0,
                    state_0.to(h_ptr.dtype.element_ty),
                    mask=v_mask[None, :],
                )
                tl.store(
                    h_ptr + h_offsets_0 + 64 * V_DIM,
                    state_1.to(h_ptr.dtype.element_ty),
                    mask=v_mask[None, :],
                )

                k_cum_offsets_0 = (
                    k_cum_base
                    + t_offsets[:, None] * K_DIM
                    + k_offsets[None, :]
                )
                k_cum_0 = tl.load(k_cumdecay_ptr + k_cum_offsets_0)
                k_cum_1 = tl.load(k_cumdecay_ptr + k_cum_offsets_0 + 64)
                v_offsets_2d = (
                    v_base
                    + t_offsets[:, None] * V_DIM
                    + v_offsets[None, :]
                )
                v_value = tl.load(
                    v_ptr + v_offsets_2d,
                    mask=v_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                v_new = v_value - tl.dot(
                    k_cum_0, state_0.to(k_cum_0.dtype)
                )
                v_new -= tl.dot(
                    k_cum_1, state_1.to(k_cum_1.dtype)
                )
                v_new_half = v_new.to(v_new_ptr.dtype.element_ty)
                tl.store(
                    v_new_ptr + v_offsets_2d,
                    v_new_half,
                    mask=v_mask[None, :],
                )

                if USE_PRECOMPUTED_SCALE:
                    row_scale = tl.load(
                        row_scale_ptr + g_base + t_offsets
                    )
                    state_scale = tl.load(
                        state_scale_ptr + chunk_index
                    )
                else:
                    g = tl.load(
                        g_ptr + g_base + t_offsets
                    ).to(tl.float32)
                    g_last = tl.load(
                        g_ptr + g_base + CHUNK_SIZE - 1
                    ).to(tl.float32)
                    row_scale = tl.exp(g_last - g)
                    state_scale = tl.exp(g_last)
                state_0 *= state_scale
                state_1 *= state_scale

                k_offsets_2d = (
                    k_base
                    + t_offsets[:, None] * K_DIM
                    + k_offsets[None, :]
                )
                k_0 = tl.load(k_ptr + k_offsets_2d)
                k_1 = tl.load(k_ptr + k_offsets_2d + 64)
                k_scaled_0 = (
                    k_0.to(tl.float32) * row_scale[:, None]
                ).to(k_0.dtype)
                k_scaled_1 = (
                    k_1.to(tl.float32) * row_scale[:, None]
                ).to(k_1.dtype)
                state_0 += tl.dot(tl.trans(k_scaled_0), v_new_half)
                state_1 += tl.dot(tl.trans(k_scaled_1), v_new_half)

        tl.store(
            next_state_ptr + state_offsets_0,
            state_0.to(next_state_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )
        tl.store(
            next_state_ptr + state_offsets_1,
            state_1.to(next_state_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )


    @triton.jit
    def fastllm_chunk_gdn_varlen_prefill_o_kernel(
        q_ptr,
        k_ptr,
        g_ptr,
        attn_ptr,
        decay_mask_ptr,
        h_ptr,
        v_new_ptr,
        chunk_token_bases_ptr,
        chunk_valid_tokens_ptr,
        output_ptr,
        total_chunks,
        total_tokens,
        key_heads,
        heads,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        APPLY_DECAY_MASK: tl.constexpr,
        DIRECT_QK: tl.constexpr,
    ):
        """Compute outputs for packed variable-length GDN chunks."""
        v_block = tl.program_id(0)
        chunk = tl.program_id(1)
        head = tl.program_id(2)
        key_head = head * key_heads // heads
        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        t_offsets = tl.arange(0, CHUNK_SIZE)
        k_offsets = tl.arange(0, 64)
        v_mask = v_offsets < V_DIM

        chunk_index = head * total_chunks + chunk
        key_chunk_index = key_head * total_chunks + chunk
        q_base = key_chunk_index * CHUNK_SIZE * K_DIM
        v_base = chunk_index * CHUNK_SIZE * V_DIM
        g_base = chunk_index * CHUNK_SIZE
        attn_base = key_chunk_index * CHUNK_SIZE * CHUNK_SIZE
        decay_mask_base = chunk_index * CHUNK_SIZE * CHUNK_SIZE
        h_base = chunk_index * K_DIM * V_DIM

        q_offsets_0 = (
            q_base
            + t_offsets[:, None] * K_DIM
            + k_offsets[None, :]
        )
        q_0 = tl.load(q_ptr + q_offsets_0)
        q_1 = tl.load(q_ptr + q_offsets_0 + 64)
        g = tl.load(g_ptr + g_base + t_offsets).to(tl.float32)
        q_scale = tl.exp(g)
        q_scaled_0 = (
            q_0.to(tl.float32) * q_scale[:, None]
        ).to(q_0.dtype)
        q_scaled_1 = (
            q_1.to(tl.float32) * q_scale[:, None]
        ).to(q_1.dtype)

        h_offsets_0 = (
            h_base
            + k_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        h_0 = tl.load(
            h_ptr + h_offsets_0,
            mask=v_mask[None, :],
            other=0.0,
        )
        h_1 = tl.load(
            h_ptr + h_offsets_0 + 64 * V_DIM,
            mask=v_mask[None, :],
            other=0.0,
        )
        output = tl.dot(q_scaled_0, h_0)
        output += tl.dot(q_scaled_1, h_1)

        if DIRECT_QK:
            k_base = key_chunk_index * CHUNK_SIZE * K_DIM
            k_offsets_2d = (
                k_base
                + t_offsets[:, None] * K_DIM
                + k_offsets[None, :]
            )
            k_0 = tl.load(k_ptr + k_offsets_2d)
            k_1 = tl.load(k_ptr + k_offsets_2d + 64)
            attn = tl.dot(q_0, tl.trans(k_0))
            attn += tl.dot(q_1, tl.trans(k_1))
            # Match the legacy materialized FP16 QK tensor exactly at the
            # numerical boundary before applying the FP16 decay mask.
            attn = attn.to(q_0.dtype)
            decay_mask_offsets = (
                decay_mask_base
                + t_offsets[:, None] * CHUNK_SIZE
                + t_offsets[None, :]
            )
            decay = tl.load(
                decay_mask_ptr + decay_mask_offsets
            )
            causal_mask = (
                t_offsets[None, :] <= t_offsets[:, None]
            )
            attn = tl.where(
                causal_mask,
                (attn * decay).to(v_new_ptr.dtype.element_ty),
                0.0,
            )
        else:
            attn_offsets = (
                attn_base
                + t_offsets[:, None] * CHUNK_SIZE
                + t_offsets[None, :]
            )
            attn = tl.load(attn_ptr + attn_offsets)
            if APPLY_DECAY_MASK:
                decay_mask_offsets = (
                    decay_mask_base
                    + t_offsets[:, None] * CHUNK_SIZE
                    + t_offsets[None, :]
                )
                decay_mask = tl.load(
                    decay_mask_ptr + decay_mask_offsets
                )
                causal_mask = (
                    t_offsets[None, :] <= t_offsets[:, None]
                )
                attn = tl.where(
                    causal_mask,
                    (attn * decay_mask).to(
                        attn_ptr.dtype.element_ty
                    ),
                    0.0,
                )
        v_offsets_2d = (
            v_base
            + t_offsets[:, None] * V_DIM
            + v_offsets[None, :]
        )
        v_new = tl.load(
            v_new_ptr + v_offsets_2d,
            mask=v_mask[None, :],
            other=0.0,
        )
        output += tl.dot(attn, v_new)
        token_base = tl.load(chunk_token_bases_ptr + chunk)
        valid_tokens = tl.load(chunk_valid_tokens_ptr + chunk)
        output_offsets = (
            ((token_base + t_offsets[:, None]) * heads + head)
            * V_DIM
            + v_offsets[None, :]
        )
        output_mask = (
            (t_offsets[:, None] < valid_tokens)
            & v_mask[None, :]
            & (token_base + t_offsets[:, None] < total_tokens)
        )
        tl.store(
            output_ptr + output_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )


    @triton.jit
    def fastllm_chunk_gdn_kkt_kernel(
        k_beta_ptr,
        k_ptr,
        output_ptr,
        total_chunks,
        key_heads,
        value_heads,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
    ):
        """Compute mapped (K * beta) @ K^T without repeating K heads."""
        chunk = tl.program_id(0)
        value_head = tl.program_id(1)
        key_head = value_head * key_heads // value_heads
        t_offsets = tl.arange(0, CHUNK_SIZE)
        d_offsets = tl.arange(0, 64)
        value_chunk = value_head * total_chunks + chunk
        key_chunk = key_head * total_chunks + chunk
        k_beta_base = value_chunk * CHUNK_SIZE * K_DIM
        k_base = key_chunk * CHUNK_SIZE * K_DIM
        output_base = value_chunk * CHUNK_SIZE * CHUNK_SIZE
        k_beta_offsets = (
            k_beta_base
            + t_offsets[:, None] * K_DIM
            + d_offsets[None, :]
        )
        k_offsets = (
            k_base
            + t_offsets[:, None] * K_DIM
            + d_offsets[None, :]
        )
        k_beta_0 = tl.load(k_beta_ptr + k_beta_offsets)
        k_beta_1 = tl.load(k_beta_ptr + k_beta_offsets + 64)
        k_0 = tl.load(k_ptr + k_offsets)
        k_1 = tl.load(k_ptr + k_offsets + 64)
        output = tl.dot(k_beta_0, tl.trans(k_0))
        output += tl.dot(k_beta_1, tl.trans(k_1))
        output_offsets = (
            output_base
            + t_offsets[:, None] * CHUNK_SIZE
            + t_offsets[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            output.to(output_ptr.dtype.element_ty),
        )


    @triton.jit
    def fastllm_chunk_gdn_postconv_kernel(
        q_input_ptr,
        k_input_ptr,
        qkv_input_ptr,
        g_input_ptr,
        beta_input_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        k_beta_ptr,
        v_beta_ptr,
        seq_len,
        chunks,
        q_scale,
        KEY_HEADS: tl.constexpr,
        VALUE_HEADS: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Fuse the bit-stable layout preparation for uniform GDN prefill.

        Q/K have already gone through FastLLM's exact combined-layout
        RMSNorm kernel. V remains in the token-major combined-QKV convolution
        output. This launch repeats key heads, reads V without a split,
        transposes to head-major, pads, scales Q with fp16 arithmetic, and
        materializes beta-scaled K/V.
        """
        token_block = tl.program_id(0)
        batch = tl.program_id(1)
        head = tl.program_id(2)
        token_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
        valid_tokens = token_offsets < seq_len
        flat_tokens = batch * seq_len + token_offsets

        if head < KEY_HEADS:
            dim_offsets = tl.arange(0, K_DIM)
            token_dim_mask = valid_tokens[:, None]
            q_values = tl.load(
                q_input_ptr
                + (flat_tokens[:, None] * KEY_HEADS + head) * K_DIM
                + dim_offsets[None, :],
                mask=token_dim_mask,
                other=0.0,
            )
            k_values = tl.load(
                k_input_ptr
                + (flat_tokens[:, None] * KEY_HEADS + head) * K_DIM
                + dim_offsets[None, :],
                mask=token_dim_mask,
                other=0.0,
            )
            q_values = (
                q_values
                * q_scale.to(tl.float16)
            ).to(tl.float16)
            for group_offset in range(0, HEAD_GROUP):
                value_head = head * HEAD_GROUP + group_offset
                beta_values = tl.load(
                    beta_input_ptr
                    + flat_tokens * VALUE_HEADS
                    + value_head,
                    mask=valid_tokens,
                    other=0.0,
                )
                output_offsets = (
                    (batch * VALUE_HEADS + value_head)
                    * chunks
                    * 64
                    * K_DIM
                    + token_offsets[:, None] * K_DIM
                    + dim_offsets[None, :]
                )
                tl.store(q_ptr + output_offsets, q_values)
                tl.store(k_ptr + output_offsets, k_values)
                tl.store(
                    k_beta_ptr + output_offsets,
                    (k_values * beta_values[:, None]).to(tl.float16),
                )
        else:
            value_head = head - KEY_HEADS
            dim_offsets = tl.arange(0, V_DIM)
            token_dim_mask = valid_tokens[:, None]
            v_values = tl.load(
                qkv_input_ptr
                + flat_tokens[:, None]
                * (KEY_HEADS * K_DIM * 2 + VALUE_HEADS * V_DIM)
                + KEY_HEADS * K_DIM * 2
                + value_head * V_DIM
                + dim_offsets[None, :],
                mask=token_dim_mask,
                other=0.0,
            )
            v_offsets = (
                (batch * VALUE_HEADS + value_head)
                * chunks
                * 64
                * V_DIM
                + token_offsets[:, None] * V_DIM
                + dim_offsets[None, :]
            )
            tl.store(v_ptr + v_offsets, v_values)
            beta_values = tl.load(
                beta_input_ptr
                + flat_tokens * VALUE_HEADS
                + value_head,
                mask=valid_tokens,
                other=0.0,
            )
            g_values = tl.load(
                g_input_ptr
                + flat_tokens * VALUE_HEADS
                + value_head,
                mask=valid_tokens,
                other=0.0,
            )
            tl.store(
                v_beta_ptr + v_offsets,
                (v_values * beta_values[:, None]).to(tl.float16),
            )
            scalar_offsets = (
                (batch * VALUE_HEADS + value_head) * chunks * 64
                + token_offsets
            )
            tl.store(g_ptr + scalar_offsets, g_values)
            tl.store(beta_ptr + scalar_offsets, beta_values)


    @triton.jit
    def fastllm_chunk_gdn_recompute_kernel(
        attn_ptr,
        v_beta_ptr,
        k_beta_ptr,
        g_exp_ptr,
        g_ptr,
        v_output_ptr,
        k_output_ptr,
        row_scale_ptr,
        state_scale_ptr,
        chunks,
        CHUNK_SIZE: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        WRITE_SCALE: tl.constexpr,
        COMPUTE_G_EXP: tl.constexpr,
    ):
        """Fuse the two WY recompute GEMMs for uniform GDN prefill.

        beta has already been rounded into v_beta/k_beta.  Either load the
        separately materialized exp(g), or compute it with libdevice expf and
        explicitly round it to FP16.  The scaled key is rounded back to FP16
        before the dot, preserving the original Exp/MulTo/GEMM boundaries.
        """
        chunk = tl.program_id(0)
        batch_head = tl.program_id(1)
        chunk_index = batch_head * chunks + chunk
        token_offsets = tl.arange(0, CHUNK_SIZE)
        dim_offsets = tl.arange(0, BLOCK_D)

        attn_base = chunk_index * CHUNK_SIZE * CHUNK_SIZE
        attn_offsets = (
            attn_base
            + token_offsets[:, None] * CHUNK_SIZE
            + token_offsets[None, :]
        )
        attn = tl.load(attn_ptr + attn_offsets)

        value_base = chunk_index * CHUNK_SIZE * V_DIM
        for block in range(0, tl.cdiv(V_DIM, BLOCK_D)):
            value_offsets = (
                value_base
                + token_offsets[:, None] * V_DIM
                + block * BLOCK_D
                + dim_offsets[None, :]
            )
            value_beta = tl.load(v_beta_ptr + value_offsets)
            value_output = tl.dot(
                attn, value_beta, out_dtype=tl.float16
            )
            tl.store(
                v_output_ptr + value_offsets,
                value_output.to(v_output_ptr.dtype.element_ty),
            )

        g_base = chunk_index * CHUNK_SIZE
        if COMPUTE_G_EXP or WRITE_SCALE:
            g_values = tl.load(
                g_ptr + g_base + token_offsets
            ).to(tl.float32)
        if COMPUTE_G_EXP:
            g_exp = libdevice.exp(g_values).to(tl.float16)
        else:
            g_exp = tl.load(g_exp_ptr + g_base + token_offsets)
        key_base = chunk_index * CHUNK_SIZE * K_DIM
        for block in range(0, tl.cdiv(K_DIM, BLOCK_D)):
            key_offsets = (
                key_base
                + token_offsets[:, None] * K_DIM
                + block * BLOCK_D
                + dim_offsets[None, :]
            )
            key_beta = tl.load(k_beta_ptr + key_offsets)
            key_scaled = (
                key_beta * g_exp[:, None]
            ).to(k_beta_ptr.dtype.element_ty)
            key_output = tl.dot(
                attn, key_scaled, out_dtype=tl.float16
            )
            tl.store(
                k_output_ptr + key_offsets,
                key_output.to(k_output_ptr.dtype.element_ty),
            )

        if WRITE_SCALE:
            g_last = tl.load(
                g_ptr + g_base + CHUNK_SIZE - 1
            ).to(tl.float32)
            tl.store(
                row_scale_ptr + g_base + token_offsets,
                tl.exp(g_last - g_values),
            )
            tl.store(
                state_scale_ptr + chunk_index,
                tl.exp(g_last),
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
    def fastllm_deepseek_v4_sqrtsoftplus_router_sm120_kernel(
        logits_ptr,
        bias_ptr,
        index_ptr,
        score_ptr,
        route_scale,
        NUM_EXPERTS: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Single-warp DeepSeek-V4 sqrt-softplus top-k router for SM120."""
        row = tl.program_id(0)
        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS
        raw = tl.load(
            logits_ptr + row * NUM_EXPERTS + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)

        # Match DeepSeekV4Softplus's stable branches. The selected score uses
        # the unbiased transformed weight; correction bias only affects rank.
        softplus = tl.where(
            raw > 20.0,
            raw,
            tl.where(
                raw < -20.0,
                tl.exp(raw),
                tl.log(1.0 + tl.exp(raw)),
            ),
        )
        weights = tl.sqrt(softplus)
        weight_finite = (
            (weights == weights)
            & (weights < float("inf"))
            & (weights > -float("inf"))
        )
        score_weights = tl.where(weight_finite, weights, 0.0)
        current = weights + bias
        finite = (
            (current == current)
            & (current < float("inf"))
            & (current > -float("inf"))
        )
        current = tl.where(expert_mask & finite, current, -float("inf"))

        output_offsets = tl.arange(0, 8)
        selected_weights = tl.zeros((8,), dtype=tl.float32)
        selected_ids = tl.zeros((8,), dtype=tl.int32)
        for slot in tl.static_range(0, TOPK):
            max_value = tl.max(current, axis=0)
            candidate = tl.where(
                current == max_value, expert_offsets, NUM_EXPERTS
            )
            expert_id = tl.min(candidate, axis=0).to(tl.int32)
            selected_weight = tl.sum(
                tl.where(
                    expert_offsets == expert_id, score_weights, 0.0
                ),
                axis=0,
            )
            is_slot = output_offsets == slot
            selected_weights = tl.where(
                is_slot, selected_weight, selected_weights
            )
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            current = tl.where(
                expert_offsets == expert_id, -float("inf"), current
            )

        weight_sum = tl.sum(selected_weights, axis=0)
        valid_sum = (
            (weight_sum == weight_sum)
            & (weight_sum < float("inf"))
            & (weight_sum > -float("inf"))
            & (tl.abs(weight_sum) >= 1.0e-20)
        )
        denominator = tl.where(valid_sum, weight_sum, 1.0)
        selected_weights *= route_scale / denominator
        output_mask = output_offsets < TOPK
        row_offsets = row * TOPK + output_offsets
        tl.store(index_ptr + row_offsets, selected_ids, mask=output_mask)
        tl.store(score_ptr + row_offsets, selected_weights, mask=output_mask)


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
    def fastllm_deepseek_v4_sparse_decode_split_kernel(
        q_ptr,
        window_kv_ptr,
        compressed_kv_ptr,
        decode_meta_ptr,
        partial_output_ptr,
        partial_max_ptr,
        partial_denom_ptr,
        softmax_scale,
        BATCH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        COMPRESSED_CAPACITY: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Build one online-softmax partial for a candidate-key split.

        Unlike the original graph-safe kernel, the candidate dimension is
        represented in the launch grid.  Each program still uses FastLLM's
        general FP32-window/BF16-compressed cache ABI, but only scans a small
        fixed-size split.  The second kernel combines these partials exactly in
        FP32, so unsupported shapes can continue to fall back to the original
        CUDA implementation without changing cache ownership or layout.
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        split_idx = tl.program_id(2)
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < HEAD_DIM

        start_pos = tl.load(decode_meta_ptr)
        live_window = tl.minimum(start_pos + 1, WINDOW_SIZE)
        if COMPRESS_RATIO > 0:
            compressed_count = tl.minimum(
                (start_pos + 1) // COMPRESS_RATIO, COMPRESSED_CAPACITY
            )
        else:
            compressed_count = 0
        total_count = live_window + compressed_count
        candidate_base = split_idx * SPLIT_SIZE
        # The launch grid covers graph capacity, while the live candidate
        # count grows with decode_meta.  Empty capacity splits are excluded by
        # the merge kernel, so they can return without loading Q/KV or writing
        # the large FP32 partial buffer.
        if candidate_base >= total_count:
            return

        q = tl.load(
            q_ptr
            + (batch_idx * NUM_HEADS + head_idx) * HEAD_DIM
            + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.full((), -float("inf"), tl.float32)
        running_denom = tl.zeros((), tl.float32)
        running_acc = tl.zeros((BLOCK_D,), tl.float32)

        for split_offset in range(0, SPLIT_SIZE):
            candidate_idx = candidate_base + split_offset
            valid = candidate_idx < total_count
            use_window = valid & (candidate_idx < live_window)

            ring_pos = start_pos % WINDOW_SIZE
            ring_full = start_pos >= WINDOW_SIZE - 1
            window_idx = tl.where(
                ring_full,
                (ring_pos + 1 + candidate_idx) % WINDOW_SIZE,
                candidate_idx,
            )
            window_kv = tl.load(
                window_kv_ptr
                + (batch_idx * WINDOW_SIZE + window_idx) * HEAD_DIM
                + dim_offsets,
                mask=use_window & dim_mask,
                other=0.0,
            ).to(tl.float32)

            compressed_idx = candidate_idx - live_window
            use_compressed = valid & ~use_window
            compressed_kv = tl.load(
                compressed_kv_ptr
                + (batch_idx * COMPRESSED_CAPACITY + compressed_idx) * HEAD_DIM
                + dim_offsets,
                mask=use_compressed & dim_mask,
                other=0.0,
            ).to(tl.float32)
            kv = tl.where(use_window, window_kv, compressed_kv)
            score = tl.sum(q * kv, axis=0) * softmax_scale

            # Empty graph-capacity splits are common early in decode.  Keep
            # every exponent finite even when both maxima are -inf so those
            # splits deterministically write a zero partial.
            had_previous = running_denom > 0.0
            next_max = tl.where(
                valid,
                tl.where(had_previous, tl.maximum(running_max, score), score),
                running_max,
            )
            has_any = had_previous | valid
            safe_next_max = tl.where(has_any, next_max, 0.0)
            previous_weight = tl.where(
                had_previous, tl.exp(running_max - safe_next_max), 0.0
            )
            candidate_weight = tl.where(
                valid, tl.exp(score - safe_next_max), 0.0
            )
            running_acc = (
                running_acc * previous_weight + kv * candidate_weight
            )
            running_denom = (
                running_denom * previous_weight + candidate_weight
            )
            running_max = next_max

        partial_row = (
            (batch_idx * NUM_HEADS + head_idx) * NUM_SPLITS + split_idx
        )
        tl.store(
            partial_output_ptr + partial_row * HEAD_DIM + dim_offsets,
            running_acc,
            mask=dim_mask,
        )
        tl.store(partial_max_ptr + partial_row, running_max)
        tl.store(partial_denom_ptr + partial_row, running_denom)


    @triton.jit
    def fastllm_deepseek_v4_sparse_decode_sm120_split_kernel(
        q_ptr,
        window_kv_ptr,
        compressed_kv_ptr,
        decode_meta_ptr,
        partial_output_ptr,
        partial_max_ptr,
        partial_denom_ptr,
        softmax_scale,
        BATCH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        COMPRESSED_CAPACITY: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
    ):
        """SM12x tensor-core sparse decode split.

        A program owns a 16-head x candidate tile, matching FlashInfer's
        head-padded sparse-MLA schedule.  QK and PV use BF16 tensor-core dot
        products while the graph-capacity ABI and FP32 split/merge state stay
        identical to the generic kernel.  The generic scalar-FP32 path remains
        available for other architectures, dtypes, dimensions, and strict
        compatibility runs.
        """
        batch_idx = tl.program_id(0)
        head_block_idx = tl.program_id(1)
        split_idx = tl.program_id(2)
        head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        candidate_offsets = tl.arange(0, SPLIT_SIZE)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < NUM_HEADS
        dim_mask = dim_offsets < HEAD_DIM

        start_pos = tl.load(decode_meta_ptr)
        live_window = tl.minimum(start_pos + 1, WINDOW_SIZE)
        if COMPRESS_RATIO > 0:
            compressed_count = tl.minimum(
                (start_pos + 1) // COMPRESS_RATIO, COMPRESSED_CAPACITY
            )
        else:
            compressed_count = 0
        total_count = live_window + compressed_count
        candidate_indices = split_idx * SPLIT_SIZE + candidate_offsets
        candidate_mask = candidate_indices < total_count
        if split_idx * SPLIT_SIZE >= total_count:
            return

        ring_pos = start_pos % WINDOW_SIZE
        ring_full = start_pos >= WINDOW_SIZE - 1
        use_window = candidate_mask & (candidate_indices < live_window)
        window_indices = tl.where(
            ring_full,
            (ring_pos + 1 + candidate_indices) % WINDOW_SIZE,
            candidate_indices,
        )
        window_kv = tl.load(
            window_kv_ptr
            + (batch_idx * WINDOW_SIZE + window_indices[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            mask=use_window[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        compressed_indices = candidate_indices - live_window
        use_compressed = candidate_mask & ~use_window
        compressed_kv = tl.load(
            compressed_kv_ptr
            + (batch_idx * COMPRESSED_CAPACITY + compressed_indices[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            mask=use_compressed[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        kv = tl.where(use_window[:, None], window_kv, compressed_kv)
        q = tl.load(
            q_ptr
            + (batch_idx * NUM_HEADS + head_offsets[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(kv), out_dtype=tl.float32)
        scores *= softmax_scale
        score_mask = head_mask[:, None] & candidate_mask[None, :]
        scores = tl.where(score_mask, scores, -float("inf"))
        partial_max = tl.max(scores, axis=1)
        safe_max = tl.where(head_mask, partial_max, 0.0)
        weights = tl.where(
            score_mask,
            tl.exp(scores - safe_max[:, None]),
            0.0,
        )
        partial_denom = tl.sum(weights, axis=1)
        partial_output = tl.dot(
            weights.to(tl.bfloat16), kv, out_dtype=tl.float32
        )

        partial_rows = (
            (batch_idx * NUM_HEADS + head_offsets) * NUM_SPLITS + split_idx
        )
        tl.store(
            partial_output_ptr
            + partial_rows[:, None] * HEAD_DIM
            + dim_offsets[None, :],
            partial_output,
            mask=head_mask[:, None] & dim_mask[None, :],
        )
        tl.store(partial_max_ptr + partial_rows, partial_max, mask=head_mask)
        tl.store(
            partial_denom_ptr + partial_rows, partial_denom, mask=head_mask
        )


    @triton.jit
    def fastllm_deepseek_v4_sparse_decode_merge_kernel(
        partial_output_ptr,
        partial_max_ptr,
        partial_denom_ptr,
        sink_ptr,
        decode_meta_ptr,
        output_ptr,
        BATCH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        COMPRESSED_CAPACITY: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Merge split online-softmax states and the learned attention sink."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        dim_block = tl.program_id(2)
        split_offsets = tl.arange(0, BLOCK_SPLITS)
        dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        start_pos = tl.load(decode_meta_ptr)
        live_window = tl.minimum(start_pos + 1, WINDOW_SIZE)
        if COMPRESS_RATIO > 0:
            compressed_count = tl.minimum(
                (start_pos + 1) // COMPRESS_RATIO, COMPRESSED_CAPACITY
            )
        else:
            compressed_count = 0
        live_splits = tl.cdiv(live_window + compressed_count, SPLIT_SIZE)
        split_mask = (split_offsets < NUM_SPLITS) & (split_offsets < live_splits)
        dim_mask = dim_offsets < HEAD_DIM
        partial_base = (batch_idx * NUM_HEADS + head_idx) * NUM_SPLITS

        partial_max = tl.load(
            partial_max_ptr + partial_base + split_offsets,
            mask=split_mask,
            other=-float("inf"),
        )
        partial_denom = tl.load(
            partial_denom_ptr + partial_base + split_offsets,
            mask=split_mask,
            other=0.0,
        )
        sink = tl.load(sink_ptr + head_idx)
        token_max = tl.max(partial_max, axis=0)
        merge_max = tl.maximum(token_max, sink)
        has_tokens = tl.sum(partial_denom, axis=0) > 0.0
        has_sink = sink > -float("inf")
        has_any = has_tokens | has_sink
        safe_merge_max = tl.where(has_any, merge_max, 0.0)
        partial_scale = tl.where(
            partial_denom > 0.0,
            tl.exp(partial_max - safe_merge_max),
            0.0,
        )
        sink_weight = tl.where(
            has_sink, tl.exp(sink - safe_merge_max), 0.0
        )
        total_denom = (
            tl.sum(partial_denom * partial_scale, axis=0) + sink_weight
        )

        partial = tl.load(
            partial_output_ptr
            + (partial_base + split_offsets[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            mask=split_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        numerator = tl.sum(partial * partial_scale[:, None], axis=0)
        result = tl.where(total_denom > 0.0, numerator / total_denom, 0.0)
        output_row = (batch_idx * NUM_HEADS + head_idx) * HEAD_DIM
        tl.store(output_ptr + output_row + dim_offsets, result, mask=dim_mask)


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
        packed_block128: tl.constexpr,
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
        if packed_block128:
            packed_row = K + tl.cdiv(K, group_k) * 4
            gate_base = gate_ptr + off_experts * INTER * packed_row
            up_base = up_ptr + off_experts * INTER * packed_row
            gate_ptrs = gate_base + local_bn[None, :] * packed_row + offs_k[:, None]
            up_ptrs = up_base + local_bn[None, :] * packed_row + offs_k[:, None]
        else:
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
        if not packed_block128:
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
            if packed_block128:
                gate_scale_ptrs = (
                    gate_base
                    + local_bn * packed_row
                    + offs_ks * (group_k + 4)
                    + group_k
                ).to(tl.pointer_type(tl.float32))
                up_scale_ptrs = (
                    up_base
                    + local_bn * packed_row
                    + offs_ks * (group_k + 4)
                    + group_k
                ).to(tl.pointer_type(tl.float32))
                gate_b_scale = tl.load(gate_scale_ptrs, mask=local_bn < INTER, other=0.0)
                up_b_scale = tl.load(up_scale_ptrs, mask=local_bn < INTER, other=0.0)
            else:
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
            if packed_block128:
                gate_ptrs += BLOCK_SIZE_K + 4
                up_ptrs += BLOCK_SIZE_K + 4
            else:
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
        packed_block128: tl.constexpr,
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
        if packed_block128:
            packed_row = K + tl.cdiv(K, group_k) * 4
            b_base = b_ptr + off_experts * N * packed_row
            b_ptrs = b_base + offs_bn[None, :] * packed_row + offs_k[:, None]
        else:
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
                if not packed_block128:
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
                    if packed_block128:
                        packed_scale_ptrs = (
                            b_base
                            + offs_bn * packed_row
                            + offs_ks * (group_k + 4)
                            + group_k
                        ).to(tl.pointer_type(tl.float32))
                        b_scale = tl.load(packed_scale_ptrs)
                    else:
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
            if packed_block128:
                b_ptrs += BLOCK_SIZE_K + 4
            else:
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


CHUNK_GDN_PREFILL_KERNEL_ORDER = (
    "h",
    "o",
    "h_precomputed_scale",
    "o_fused_decay_mask",
)
CHUNK_GDN_VARLEN_PREFILL_KERNEL_ORDER = (
    "h",
    "o",
    "h_precomputed_scale",
    "o_fused_decay_mask",
    "o_direct_qk",
)
CHUNK_GDN_RECOMPUTE_KERNEL_ORDER = (
    "recompute",
    "recompute_precomputed_scale",
    "recompute_internal_exp",
    "recompute_precomputed_scale_internal_exp",
    "kkt",
)


def chunk_gdn_prefill_cache_paths(payload):
    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_prefill currently requires fp16")
    chunks = require_int(payload, "chunks")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_v = require_int(payload, "block_v", 32)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    if chunk_size != 64 or k_dim != 128 or v_dim != 128:
        raise ValueError(
            "chunk_gdn_prefill currently requires chunk_size=64, k_dim=128, v_dim=128"
        )
    if block_v not in {32, 64}:
        raise ValueError("chunk_gdn_prefill block_v must be 32 or 64")
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"chunk_gdn_prefill_v6_{dtype}_sm{arch}"
        f"_c{chunks}_t{chunk_size}_k{k_dim}_v{v_dim}_bv{block_v}"
        f"_nw{num_warps}_ns{num_stages}"
    )
    cubins = {
        key: cache_dir / f"{name}_{key}.cubin"
        for key in CHUNK_GDN_PREFILL_KERNEL_ORDER
    }
    return cubins, cache_dir / f"{name}.json"


def chunk_gdn_varlen_prefill_cache_paths(payload):
    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_varlen_prefill currently requires fp16")
    max_chunks = require_int(payload, "max_chunks")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    h_block_v = require_int(payload, "h_block_v", 32)
    o_block_v = require_int(payload, "o_block_v", 64)
    num_warps = require_int(payload, "num_warps", 4)
    h_num_stages = require_int(payload, "h_num_stages", 2)
    o_num_stages = require_int(payload, "o_num_stages", 3)
    if chunk_size != 64 or k_dim != 128 or v_dim != 128:
        raise ValueError(
            "chunk_gdn_varlen_prefill requires chunk_size=64, k_dim=128, v_dim=128"
        )
    if h_block_v not in {32, 64} or o_block_v not in {32, 64}:
        raise ValueError("chunk_gdn_varlen_prefill block_v must be 32 or 64")
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"chunk_gdn_varlen_prefill_v7_{dtype}_sm{arch}"
        f"_mc{max_chunks}_t{chunk_size}_k{k_dim}_v{v_dim}"
        f"_hbv{h_block_v}_obv{o_block_v}_nw{num_warps}"
        f"_hns{h_num_stages}_ons{o_num_stages}"
    )
    cubins = {
        key: cache_dir / f"{name}_{key}.cubin"
        for key in CHUNK_GDN_VARLEN_PREFILL_KERNEL_ORDER
    }
    return cubins, cache_dir / f"{name}.json"


def chunk_gdn_postconv_cache_paths(payload):
    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_postconv currently requires fp16")
    key_heads = require_int(payload, "key_heads")
    value_heads = require_int(payload, "value_heads")
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_t = require_int(payload, "block_t", 16)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    if value_heads % key_heads != 0:
        raise ValueError("value_heads must be divisible by key_heads")
    if k_dim != 128 or v_dim != 128 or block_t != 16:
        raise ValueError(
            "chunk_gdn_postconv currently requires k_dim=v_dim=128 and block_t=16"
        )
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"chunk_gdn_postconv_v5_{dtype}_sm{arch}"
        f"_hk{key_heads}_hv{value_heads}_k{k_dim}_v{v_dim}"
        f"_bt{block_t}_nw{num_warps}_ns{num_stages}"
    )
    return cache_dir / f"{name}.cubin", cache_dir / f"{name}.json"


def chunk_gdn_recompute_cache_paths(payload):
    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_recompute currently requires fp16")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_d = require_int(payload, "block_d", 64)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    if chunk_size != 64 or k_dim != 128 or v_dim != 128 or block_d != 64:
        raise ValueError(
            "chunk_gdn_recompute currently requires "
            "chunk_size=64, k_dim=v_dim=128, and block_d=64"
        )
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"chunk_gdn_recompute_v7_{dtype}_sm{arch}"
        f"_t{chunk_size}_k{k_dim}_v{v_dim}_bd{block_d}"
        f"_nw{num_warps}_ns{num_stages}"
    )
    cubins = {
        key: cache_dir / f"{name}_{key}.cubin"
        for key in CHUNK_GDN_RECOMPUTE_KERNEL_ORDER
    }
    return cubins, cache_dir / f"{name}.json"


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


def deepseek_v4_sqrtsoftplus_router_cache_paths(payload):
    arch = require_int(payload, "arch")
    num_experts = require_int(payload, "num_experts", 256)
    topk = require_int(payload, "topk", 6)
    block_n = require_int(payload, "block_n", 256)
    num_warps = require_int(payload, "num_warps", 1)
    num_stages = require_int(payload, "num_stages", 1)
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"deepseek_v4_sqrtsoftplus_router_v1_sm{arch}"
        f"_e{num_experts}_k{topk}_bn{block_n}"
        f"_nw{num_warps}_ns{num_stages}"
    )
    return cache_dir / f"{name}.cubin", cache_dir / f"{name}.json"


DEEPSEEK_V4_SPARSE_DECODE_KERNEL_ORDER = ("split", "merge")


def deepseek_v4_sparse_decode_cache_paths(payload):
    arch = require_int(payload, "arch")
    batch = require_int(payload, "batch", 1)
    num_heads = require_int(payload, "num_heads", 64)
    head_dim = require_int(payload, "head_dim", 512)
    window_size = require_int(payload, "window_size", 128)
    compress_ratio = require_nonnegative_int(payload, "compress_ratio", 0)
    compressed_capacity = require_int(payload, "compressed_capacity", 1)
    split_size = require_int(payload, "split_size", 16)
    block_d = require_int(payload, "block_d", 512)
    merge_block_d = require_int(payload, "merge_block_d", 32)
    split_num_warps = require_int(payload, "split_num_warps", 4)
    merge_num_warps = require_int(payload, "merge_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    variant = str(payload.get("variant") or "generic").strip().lower()
    if variant not in {"generic", "sm120_tensorcore"}:
        raise ValueError(
            "DeepSeek-V4 sparse decode variant must be generic or sm120_tensorcore"
        )
    num_splits = (window_size + compressed_capacity + split_size - 1) // split_size
    block_splits = 1 << (num_splits - 1).bit_length()
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    variant_tag = "" if variant == "generic" else f"_{variant}"
    name = (
        f"deepseek_v4_sparse_decode_v3{variant_tag}_sm{arch}"
        f"_b{batch}_h{num_heads}_d{head_dim}_w{window_size}_cr{compress_ratio}"
        f"_cc{compressed_capacity}_ss{split_size}_bd{block_d}"
        f"_mbd{merge_block_d}_bs{block_splits}"
        f"_snw{split_num_warps}_mnw{merge_num_warps}_ns{num_stages}"
    )
    cubins = {
        key: cache_dir / f"{name}_{key}.cubin"
        for key in DEEPSEEK_V4_SPARSE_DECODE_KERNEL_ORDER
    }
    return cubins, cache_dir / f"{name}.json"


def merge_moe_fp8_cache_paths(payload):
    arch = require_int(payload, "arch")
    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    weight_layout = str(payload.get("weight_layout") or "separate").strip().lower()
    if weight_layout not in {"separate", "block128"}:
        raise ValueError("merge_moe_fp8 weight_layout must be separate or block128")
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
        f"merge_moe_fp8_v35_{weight_layout}_{input_dtype}_sm{arch}"
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


def compile_chunk_gdn_prefill(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_prefill currently requires fp16")
    chunks = require_int(payload, "chunks")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_v = require_int(payload, "block_v", 32)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    cubin_paths, meta_path = chunk_gdn_prefill_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())

    for path in cubin_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    constexprs = {
        "CHUNKS": chunks,
        "CHUNK_SIZE": chunk_size,
        "K_DIM": k_dim,
        "V_DIM": v_dim,
        "BLOCK_V": block_v,
    }
    h_constexprs = dict(constexprs)
    h_constexprs["USE_PRECOMPUTED_SCALE"] = False
    h_precomputed_scale_constexprs = dict(constexprs)
    h_precomputed_scale_constexprs["USE_PRECOMPUTED_SCALE"] = True
    o_constexprs = dict(constexprs)
    o_constexprs["APPLY_DECAY_MASK"] = False
    o_fused_decay_mask_constexprs = dict(constexprs)
    o_fused_decay_mask_constexprs["APPLY_DECAY_MASK"] = True
    h_signature = {
        "k_ptr": f"*{dtype}",
        "v_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "k_cumdecay_ptr": f"*{dtype}",
        "state_ptr": f"*{dtype}",
        "next_state_ptr": f"*{dtype}",
        "h_ptr": f"*{dtype}",
        "v_new_ptr": f"*{dtype}",
        "row_scale_ptr": "*fp32",
        "state_scale_ptr": "*fp32",
        "CHUNKS": "constexpr",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "BLOCK_V": "constexpr",
        "USE_PRECOMPUTED_SCALE": "constexpr",
    }
    o_signature = {
        "q_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "attn_ptr": f"*{dtype}",
        "decay_mask_ptr": f"*{dtype}",
        "h_ptr": f"*{dtype}",
        "v_new_ptr": f"*{dtype}",
        "output_ptr": f"*{dtype}",
        "CHUNKS": "constexpr",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "BLOCK_V": "constexpr",
        "APPLY_DECAY_MASK": "constexpr",
    }
    ccinfos = {
        "h": _compile_cubin(
            fastllm_chunk_gdn_prefill_h_kernel,
            h_signature,
            h_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["h"],
        ),
        "o": _compile_cubin(
            fastllm_chunk_gdn_prefill_o_kernel,
            o_signature,
            o_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["o"],
        ),
        "h_precomputed_scale": _compile_cubin(
            fastllm_chunk_gdn_prefill_h_kernel,
            h_signature,
            h_precomputed_scale_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["h_precomputed_scale"],
        ),
        "o_fused_decay_mask": _compile_cubin(
            fastllm_chunk_gdn_prefill_o_kernel,
            o_signature,
            o_fused_decay_mask_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["o_fused_decay_mask"],
        ),
    }
    kernels = {
        key: {
            "cubin": str(cubin_paths[key]),
            "kernel": ccinfos[key].metadata.name,
            "shared": int(ccinfos[key].metadata.shared),
            "num_warps": int(ccinfos[key].metadata.num_warps),
        }
        for key in CHUNK_GDN_PREFILL_KERNEL_ORDER
    }
    meta = {
        "ok": True,
        "op": "chunk_gdn_prefill",
        "kernels": kernels,
        "arch": arch,
        "dtype": dtype,
        "chunks": chunks,
        "chunk_size": chunk_size,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "block_v": block_v,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_chunk_gdn_varlen_prefill(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_varlen_prefill currently requires fp16")
    max_chunks = require_int(payload, "max_chunks")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    h_block_v = require_int(payload, "h_block_v", 32)
    o_block_v = require_int(payload, "o_block_v", 64)
    num_warps = require_int(payload, "num_warps", 4)
    h_num_stages = require_int(payload, "h_num_stages", 2)
    o_num_stages = require_int(payload, "o_num_stages", 3)
    cubin_paths, meta_path = chunk_gdn_varlen_prefill_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())

    for path in cubin_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    h_constexprs = {
        "MAX_CHUNKS": max_chunks,
        "CHUNK_SIZE": chunk_size,
        "K_DIM": k_dim,
        "V_DIM": v_dim,
        "BLOCK_V": h_block_v,
        "USE_PRECOMPUTED_SCALE": False,
    }
    h_precomputed_scale_constexprs = dict(h_constexprs)
    h_precomputed_scale_constexprs["USE_PRECOMPUTED_SCALE"] = True
    o_constexprs = {
        "CHUNK_SIZE": chunk_size,
        "K_DIM": k_dim,
        "V_DIM": v_dim,
        "BLOCK_V": o_block_v,
        "APPLY_DECAY_MASK": False,
        "DIRECT_QK": False,
    }
    o_fused_decay_mask_constexprs = dict(o_constexprs)
    o_fused_decay_mask_constexprs["APPLY_DECAY_MASK"] = True
    o_direct_qk_constexprs = dict(o_constexprs)
    o_direct_qk_constexprs["DIRECT_QK"] = True
    h_signature = {
        "k_ptr": f"*{dtype}",
        "v_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "k_cumdecay_ptr": f"*{dtype}",
        "state_ptr": f"*{dtype}",
        "next_state_ptr": f"*{dtype}",
        "chunk_offsets_ptr": "*i32",
        "h_ptr": f"*{dtype}",
        "v_new_ptr": f"*{dtype}",
        "row_scale_ptr": "*fp32",
        "state_scale_ptr": "*fp32",
        "total_chunks": "i32",
        "key_heads": "i32",
        "heads": "i32",
        "MAX_CHUNKS": "constexpr",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "BLOCK_V": "constexpr",
        "USE_PRECOMPUTED_SCALE": "constexpr",
    }
    o_signature = {
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "attn_ptr": f"*{dtype}",
        "decay_mask_ptr": f"*{dtype}",
        "h_ptr": f"*{dtype}",
        "v_new_ptr": f"*{dtype}",
        "chunk_token_bases_ptr": "*i32",
        "chunk_valid_tokens_ptr": "*i32",
        "output_ptr": f"*{dtype}",
        "total_chunks": "i32",
        "total_tokens": "i32",
        "key_heads": "i32",
        "heads": "i32",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "BLOCK_V": "constexpr",
        "APPLY_DECAY_MASK": "constexpr",
        "DIRECT_QK": "constexpr",
    }
    ccinfos = {
        "h": _compile_cubin(
            fastllm_chunk_gdn_varlen_prefill_h_kernel,
            h_signature, h_constexprs, arch, num_warps,
            h_num_stages, cubin_paths["h"],
        ),
        "o": _compile_cubin(
            fastllm_chunk_gdn_varlen_prefill_o_kernel,
            o_signature, o_constexprs, arch, num_warps,
            o_num_stages, cubin_paths["o"],
        ),
        "h_precomputed_scale": _compile_cubin(
            fastllm_chunk_gdn_varlen_prefill_h_kernel,
            h_signature, h_precomputed_scale_constexprs,
            arch, num_warps, h_num_stages,
            cubin_paths["h_precomputed_scale"],
        ),
        "o_fused_decay_mask": _compile_cubin(
            fastllm_chunk_gdn_varlen_prefill_o_kernel,
            o_signature, o_fused_decay_mask_constexprs,
            arch, num_warps, o_num_stages,
            cubin_paths["o_fused_decay_mask"],
        ),
        "o_direct_qk": _compile_cubin(
            fastllm_chunk_gdn_varlen_prefill_o_kernel,
            o_signature, o_direct_qk_constexprs,
            arch, num_warps, o_num_stages,
            cubin_paths["o_direct_qk"],
        ),
    }
    kernels = {
        key: {
            "cubin": str(cubin_paths[key]),
            "kernel": ccinfos[key].metadata.name,
            "shared": int(ccinfos[key].metadata.shared),
            "num_warps": int(ccinfos[key].metadata.num_warps),
        }
        for key in CHUNK_GDN_VARLEN_PREFILL_KERNEL_ORDER
    }
    meta = {
        "ok": True,
        "op": "chunk_gdn_varlen_prefill",
        "kernels": kernels,
        "arch": arch,
        "dtype": dtype,
        "max_chunks": max_chunks,
        "chunk_size": chunk_size,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "h_block_v": h_block_v,
        "o_block_v": o_block_v,
        "num_warps": num_warps,
        "h_num_stages": h_num_stages,
        "o_num_stages": o_num_stages,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_chunk_gdn_postconv(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_postconv currently requires fp16")
    key_heads = require_int(payload, "key_heads")
    value_heads = require_int(payload, "value_heads")
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_t = require_int(payload, "block_t", 16)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    cubin_path, meta_path = chunk_gdn_postconv_cache_paths(payload)
    if cubin_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    cubin_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "q_input_ptr": f"*{dtype}",
        "k_input_ptr": f"*{dtype}",
        "qkv_input_ptr": f"*{dtype}",
        "g_input_ptr": f"*{dtype}",
        "beta_input_ptr": f"*{dtype}",
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{dtype}",
        "v_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "beta_ptr": f"*{dtype}",
        "k_beta_ptr": f"*{dtype}",
        "v_beta_ptr": f"*{dtype}",
        "seq_len": "i32",
        "chunks": "i32",
        "q_scale": "fp32",
        "KEY_HEADS": "constexpr",
        "VALUE_HEADS": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "HEAD_GROUP": "constexpr",
        "BLOCK_T": "constexpr",
    }
    constexprs = {
        "KEY_HEADS": key_heads,
        "VALUE_HEADS": value_heads,
        "K_DIM": k_dim,
        "V_DIM": v_dim,
        "HEAD_GROUP": value_heads // key_heads,
        "BLOCK_T": block_t,
    }
    ccinfo = _compile_cubin(
        fastllm_chunk_gdn_postconv_kernel,
        signature,
        constexprs,
        arch,
        num_warps,
        num_stages,
        cubin_path,
    )
    meta = {
        "ok": True,
        "op": "chunk_gdn_postconv",
        "cubin": str(cubin_path),
        "kernel": ccinfo.metadata.name,
        "shared": int(ccinfo.metadata.shared),
        "num_warps": int(ccinfo.metadata.num_warps),
        "num_stages": int(ccinfo.metadata.num_stages),
        "arch": arch,
        "dtype": dtype,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "block_t": block_t,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_chunk_gdn_recompute(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    dtype = require_dtype(payload, "dtype")
    if dtype != "fp16":
        raise ValueError("chunk_gdn_recompute currently requires fp16")
    chunk_size = require_int(payload, "chunk_size", 64)
    k_dim = require_int(payload, "k_dim", 128)
    v_dim = require_int(payload, "v_dim", 128)
    block_d = require_int(payload, "block_d", 64)
    num_warps = require_int(payload, "num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    cubin_paths, meta_path = chunk_gdn_recompute_cache_paths(payload)
    if (
        all(path.exists() for path in cubin_paths.values())
        and meta_path.exists()
    ):
        return json.loads(meta_path.read_text())

    for path in cubin_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "attn_ptr": f"*{dtype}",
        "v_beta_ptr": f"*{dtype}",
        "k_beta_ptr": f"*{dtype}",
        "g_exp_ptr": f"*{dtype}",
        "g_ptr": f"*{dtype}",
        "v_output_ptr": f"*{dtype}",
        "k_output_ptr": f"*{dtype}",
        "row_scale_ptr": "*fp32",
        "state_scale_ptr": "*fp32",
        "chunks": "i32",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
        "V_DIM": "constexpr",
        "BLOCK_D": "constexpr",
        "WRITE_SCALE": "constexpr",
        "COMPUTE_G_EXP": "constexpr",
    }
    constexprs = {
        "CHUNK_SIZE": chunk_size,
        "K_DIM": k_dim,
        "V_DIM": v_dim,
        "BLOCK_D": block_d,
    }
    baseline_constexprs = dict(constexprs)
    baseline_constexprs["WRITE_SCALE"] = False
    baseline_constexprs["COMPUTE_G_EXP"] = False
    precomputed_scale_constexprs = dict(constexprs)
    precomputed_scale_constexprs["WRITE_SCALE"] = True
    precomputed_scale_constexprs["COMPUTE_G_EXP"] = False
    internal_exp_constexprs = dict(constexprs)
    internal_exp_constexprs["WRITE_SCALE"] = False
    internal_exp_constexprs["COMPUTE_G_EXP"] = True
    precomputed_scale_internal_exp_constexprs = dict(constexprs)
    precomputed_scale_internal_exp_constexprs["WRITE_SCALE"] = True
    precomputed_scale_internal_exp_constexprs["COMPUTE_G_EXP"] = True
    kkt_signature = {
        "k_beta_ptr": f"*{dtype}",
        "k_ptr": f"*{dtype}",
        "output_ptr": f"*{dtype}",
        "total_chunks": "i32",
        "key_heads": "i32",
        "value_heads": "i32",
        "CHUNK_SIZE": "constexpr",
        "K_DIM": "constexpr",
    }
    kkt_constexprs = {
        "CHUNK_SIZE": chunk_size,
        "K_DIM": k_dim,
    }
    ccinfos = {
        "recompute": _compile_cubin(
            fastllm_chunk_gdn_recompute_kernel,
            signature,
            baseline_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["recompute"],
        ),
        "recompute_precomputed_scale": _compile_cubin(
            fastllm_chunk_gdn_recompute_kernel,
            signature,
            precomputed_scale_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["recompute_precomputed_scale"],
        ),
        "recompute_internal_exp": _compile_cubin(
            fastllm_chunk_gdn_recompute_kernel,
            signature,
            internal_exp_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["recompute_internal_exp"],
        ),
        "recompute_precomputed_scale_internal_exp": _compile_cubin(
            fastllm_chunk_gdn_recompute_kernel,
            signature,
            precomputed_scale_internal_exp_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths[
                "recompute_precomputed_scale_internal_exp"
            ],
        ),
        "kkt": _compile_cubin(
            fastllm_chunk_gdn_kkt_kernel,
            kkt_signature,
            kkt_constexprs,
            arch,
            num_warps,
            num_stages,
            cubin_paths["kkt"],
        ),
    }
    kernels = {
        key: {
            "cubin": str(cubin_paths[key]),
            "kernel": ccinfos[key].metadata.name,
            "shared": int(ccinfos[key].metadata.shared),
            "num_warps": int(ccinfos[key].metadata.num_warps),
        }
        for key in CHUNK_GDN_RECOMPUTE_KERNEL_ORDER
    }
    meta = {
        "ok": True,
        "op": "chunk_gdn_recompute",
        "kernels": kernels,
        "num_stages": int(
            ccinfos["recompute"].metadata.num_stages
        ),
        "arch": arch,
        "dtype": dtype,
        "chunk_size": chunk_size,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "block_d": block_d,
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
    compressed_capacity = require_int(payload, "compressed_capacity", 1)
    split_size = require_int(payload, "split_size", 16)
    block_d = require_int(payload, "block_d", 512)
    merge_block_d = require_int(payload, "merge_block_d", 32)
    split_num_warps = require_int(payload, "split_num_warps", 4)
    merge_num_warps = require_int(payload, "merge_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 2)
    variant = str(payload.get("variant") or "generic").strip().lower()
    if variant not in {"generic", "sm120_tensorcore"}:
        raise ValueError(
            "DeepSeek-V4 sparse decode variant must be generic or sm120_tensorcore"
        )
    if batch != 1:
        raise ValueError("DeepSeek-V4 sparse decode currently requires batch=1")
    if split_size not in {8, 16, 32, 64}:
        raise ValueError("DeepSeek-V4 sparse decode split_size must be 8, 16, 32, or 64")
    if block_d < head_dim or block_d > 1024 or (block_d & (block_d - 1)) != 0:
        raise ValueError("DeepSeek-V4 sparse decode block_d must be a power of two covering head_dim")
    if merge_block_d not in {16, 32, 64, 128}:
        raise ValueError("DeepSeek-V4 sparse decode merge_block_d must be 16, 32, 64, or 128")
    if num_heads <= 0 or head_dim <= 0 or window_size <= 0:
        raise ValueError("DeepSeek-V4 sparse decode dimensions must be positive")
    if variant == "sm120_tensorcore":
        if arch not in {120, 121}:
            raise ValueError("SM120 tensor-core sparse decode requires SM12x")
        if head_dim != 512 or block_d != 512 or num_heads > 16:
            raise ValueError(
                "SM120 tensor-core sparse decode requires head_dim=block_d=512 "
                "and at most 16 local heads"
            )
        if split_size not in {16, 32, 64}:
            raise ValueError(
                "SM120 tensor-core sparse decode requires split_size 16, 32, or 64"
            )
    num_splits = (window_size + compressed_capacity + split_size - 1) // split_size
    block_splits = 1 << (num_splits - 1).bit_length()
    if num_splits > 256:
        raise ValueError(
            "DeepSeek-V4 sparse decode optimized path supports at most 256 splits"
        )

    cubin_paths, meta_path = deepseek_v4_sparse_decode_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    split_signature = {
        "q_ptr": "*bf16",
        "window_kv_ptr": "*fp32",
        "compressed_kv_ptr": "*bf16",
        "decode_meta_ptr": "*i32",
        "partial_output_ptr": "*fp32",
        "partial_max_ptr": "*fp32",
        "partial_denom_ptr": "*fp32",
        "softmax_scale": "fp32",
        "BATCH": "constexpr",
        "NUM_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "WINDOW_SIZE": "constexpr",
        "COMPRESS_RATIO": "constexpr",
        "COMPRESSED_CAPACITY": "constexpr",
        "NUM_SPLITS": "constexpr",
        "SPLIT_SIZE": "constexpr",
        "BLOCK_D": "constexpr",
    }
    split_constexprs = {
        "BATCH": batch,
        "NUM_HEADS": num_heads,
        "HEAD_DIM": head_dim,
        "WINDOW_SIZE": window_size,
        "COMPRESS_RATIO": compress_ratio,
        "COMPRESSED_CAPACITY": compressed_capacity,
        "NUM_SPLITS": num_splits,
        "SPLIT_SIZE": split_size,
        "BLOCK_D": block_d,
    }
    split_kernel = fastllm_deepseek_v4_sparse_decode_split_kernel
    split_head_block = 1
    if variant == "sm120_tensorcore":
        split_signature["HEAD_BLOCK"] = "constexpr"
        split_constexprs["HEAD_BLOCK"] = 16
        split_kernel = fastllm_deepseek_v4_sparse_decode_sm120_split_kernel
        split_head_block = 16
    split_ccinfo = _compile_cubin(
        split_kernel,
        split_signature,
        split_constexprs,
        arch,
        split_num_warps,
        num_stages,
        cubin_paths["split"],
    )

    merge_signature = {
        "partial_output_ptr": "*fp32",
        "partial_max_ptr": "*fp32",
        "partial_denom_ptr": "*fp32",
        "sink_ptr": "*fp32",
        "decode_meta_ptr": "*i32",
        "output_ptr": "*fp32",
        "BATCH": "constexpr",
        "NUM_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "WINDOW_SIZE": "constexpr",
        "COMPRESS_RATIO": "constexpr",
        "COMPRESSED_CAPACITY": "constexpr",
        "NUM_SPLITS": "constexpr",
        "SPLIT_SIZE": "constexpr",
        "BLOCK_SPLITS": "constexpr",
        "BLOCK_D": "constexpr",
    }
    merge_constexprs = {
        "BATCH": batch,
        "NUM_HEADS": num_heads,
        "HEAD_DIM": head_dim,
        "WINDOW_SIZE": window_size,
        "COMPRESS_RATIO": compress_ratio,
        "COMPRESSED_CAPACITY": compressed_capacity,
        "NUM_SPLITS": num_splits,
        "SPLIT_SIZE": split_size,
        "BLOCK_SPLITS": block_splits,
        "BLOCK_D": merge_block_d,
    }
    merge_ccinfo = _compile_cubin(
        fastllm_deepseek_v4_sparse_decode_merge_kernel,
        merge_signature,
        merge_constexprs,
        arch,
        merge_num_warps,
        num_stages,
        cubin_paths["merge"],
    )

    kernel_infos = {
        "split": split_ccinfo,
        "merge": merge_ccinfo,
    }
    meta = {
        "ok": True,
        "op": "deepseek_v4_sparse_decode",
        "version": 3,
        "variant": variant,
        "split_head_block": split_head_block,
        "kernels": {
            key: {
                "cubin": str(cubin_paths[key]),
                "kernel": kernel_infos[key].metadata.name,
                "shared": int(kernel_infos[key].metadata.shared),
                "num_warps": int(kernel_infos[key].metadata.num_warps),
            }
            for key in DEEPSEEK_V4_SPARSE_DECODE_KERNEL_ORDER
        },
        "num_stages": num_stages,
        "arch": arch,
        "batch": batch,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "compress_ratio": compress_ratio,
        "compressed_capacity": compressed_capacity,
        "num_splits": num_splits,
        "block_splits": block_splits,
        "split_size": split_size,
        "block_d": block_d,
        "merge_block_d": merge_block_d,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_deepseek_v4_sqrtsoftplus_router(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    arch = require_int(payload, "arch")
    num_experts = require_int(payload, "num_experts", 256)
    topk = require_int(payload, "topk", 6)
    block_n = require_int(payload, "block_n", 256)
    num_warps = require_int(payload, "num_warps", 1)
    num_stages = require_int(payload, "num_stages", 1)
    if arch not in {120, 121}:
        raise ValueError("DeepSeek-V4 high-efficiency router requires SM12x")
    if num_experts != 256 or topk != 6 or block_n != 256:
        raise ValueError(
            "DeepSeek-V4 high-efficiency router requires 256 experts and top-6"
        )
    if num_warps != 1 or num_stages != 1:
        raise ValueError(
            "DeepSeek-V4 high-efficiency router requires one warp and one stage"
        )

    cubin_path, meta_path = deepseek_v4_sqrtsoftplus_router_cache_paths(payload)
    if cubin_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "logits_ptr": "*fp32",
        "bias_ptr": "*fp32",
        "index_ptr": "*i32",
        "score_ptr": "*fp32",
        "route_scale": "fp32",
        "NUM_EXPERTS": "constexpr",
        "TOPK": "constexpr",
        "BLOCK_N": "constexpr",
    }
    constexprs = {
        "NUM_EXPERTS": num_experts,
        "TOPK": topk,
        "BLOCK_N": block_n,
    }
    ccinfo = _compile_cubin(
        fastllm_deepseek_v4_sqrtsoftplus_router_sm120_kernel,
        signature,
        constexprs,
        arch,
        num_warps,
        num_stages,
        cubin_path,
    )
    meta = {
        "ok": True,
        "op": "deepseek_v4_sqrtsoftplus_router",
        "version": 1,
        "variant": "sm120",
        "cubin": str(cubin_path),
        "kernel": ccinfo.metadata.name,
        "shared": int(ccinfo.metadata.shared),
        "num_warps": int(ccinfo.metadata.num_warps),
        "num_stages": int(ccinfo.metadata.num_stages),
        "arch": arch,
        "num_experts": num_experts,
        "topk": topk,
        "block_n": block_n,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def compile_merge_moe_fp8(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")

    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    weight_layout = str(payload.get("weight_layout") or "separate").strip().lower()
    if weight_layout not in {"separate", "block128"}:
        raise ValueError("merge_moe_fp8 weight_layout must be separate or block128")
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
    if weight_layout == "block128" and (
        group_block_n != 128 or group_block_k != 128
    ):
        raise ValueError("Block128 FP8 MoE requires 128x128 scale groups")
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
        "packed_block128": "constexpr",
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
        "packed_block128": weight_layout == "block128",
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
        "packed_block128": "constexpr",
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
        "packed_block128": weight_layout == "block128",
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
        "packed_block128": "constexpr",
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
        "packed_block128": weight_layout == "block128",
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
        "weight_layout": weight_layout,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def handle_compile(payload):
    op = payload.get("op")
    with _compile_lock:
        if op == "linear":
            return compile_linear(payload)
        if op == "chunk_gdn_prefill":
            return compile_chunk_gdn_prefill(payload)
        if op in (
            "chunk_gdn_varlen_prefill",
            "chunk_gdn_varlen_prefill_v3",
            "chunk_gdn_varlen_prefill_v4",
            "chunk_gdn_varlen_prefill_v5",
            "chunk_gdn_varlen_prefill_v6",
            "chunk_gdn_varlen_prefill_v7",
        ):
            return compile_chunk_gdn_varlen_prefill(payload)
        if op == "chunk_gdn_postconv":
            return compile_chunk_gdn_postconv(payload)
        if op in (
            "chunk_gdn_recompute",
            "chunk_gdn_recompute_v5",
            "chunk_gdn_recompute_v6",
            "chunk_gdn_recompute_v7",
        ):
            return compile_chunk_gdn_recompute(payload)
        if op == "linear_fp8_block128":
            return compile_linear_fp8_block128(payload)
        if op == "deepseek_v4_fp8_woa":
            return compile_deepseek_v4_fp8_woa(payload)
        if op == "deepseek_v4_sparse_decode":
            return compile_deepseek_v4_sparse_decode(payload)
        if op == "deepseek_v4_sqrtsoftplus_router":
            return compile_deepseek_v4_sqrtsoftplus_router(payload)
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
