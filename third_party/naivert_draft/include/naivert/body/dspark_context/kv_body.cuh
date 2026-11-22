#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/body/dspark_context/args.hpp"
#include "naivert/body/dspark_context/geometry.cuh"
#include "naivert/common/resident_grid_barrier.cuh"
// D2: shared BF16 rows MMA for all five layers' context K/V projections,
// Keep the original 128-wide head norm, RoPE and cache scatter.
// One grid barrier separates projection from finishing; the fixed resident
// grid and caller ABI are unchanged.
#include "naivert/body/attn_common/strided_rows_mma.cuh"
#include "naivert/common/bulk_input_copy.cuh"

#include "naivert/common/tp8_mailbox.cuh"

#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace fastllm_draft_ops {

constexpr int kHeadDim = 128;                 // D_HEAD_DIM
constexpr int kLayers = 5;                    // D_LAYERS
constexpr int kKvRowsR = 128;                 // KV_HEADS_R * D_HEAD_DIM, one kv head per rank
constexpr int kColsPerLayer = 2 * kKvRowsR;   // k columns then v columns
constexpr int kTotalCols = kLayers * kColsPerLayer;   // 1280

constexpr int kCtasD2 = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kFinishCtas = kLayers * kWindow;


static_assert(kTotalCols % 16 == 0, "whole MMA tiles");
static_assert(kFinishCtas <= kCtasD2, "phase 2 must be a subset of the launched grid");
// Same residency requirement as D1: a CTA that has not launched cannot arrive at the barrier, and
// the ones that did would spin forever -- inside a captured graph, an unkillable hang.
static_assert(kCtasD2 < NAIVERT_SM_COUNT,
              "dspark_ctx_kv needs the whole grid resident; kCtasD2 must stay below the SM count");



// Sense-reversing arrive-and-wait, identical to `dspark_fc_ctx.cu`'s. `generation` is never reset,
// which is what makes it safe across graph replays: a replay just continues counting.

struct CtxKvBody {
  template <bool kPreparedInput = false>
  __device__ static void run(unsigned char* smem_raw, const CtxKvArgs& args) {
    constexpr int stride = kHidden + (kPreparedInput ? 32 : 0);
    __nv_bfloat16* ctx_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [8][4096]
    float* reduce_s = reinterpret_cast<float*>(smem_raw + kWindow * stride * 2);  // [kWarps]
    __nv_bfloat16* head_s = reinterpret_cast<__nv_bfloat16*>(reduce_s + kWarps); // [128]

    auto* partials = reinterpret_cast<float*>(head_s + kHeadDim);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if constexpr (!kPreparedInput)
      naivert::bulk_input_copy<kWindow * kHidden * 2>(ctx_s, args.ctx);

    naivert::attn_rows::project_strided_rows_mma<kWindow, kHidden, 8, 1, kWarps, stride>(
        0, kTotalCols / 16,
        [&](int column) {
          const int layer = column / kColsPerLayer;
          const int kv = column % kColsPerLayer;
          return ((kv >= kKvRowsR) ? args.v_w[layer] : args.k_w[layer]) +
                 std::size_t(kv % kKvRowsR) * kHidden;
        }, ctx_s, partials,
        [&](int column, int row, float value) {
          const int layer = column / kColsPerLayer;
          const int kv = column % kColsPerLayer;
          const std::size_t offset = ((std::size_t(layer) * 2 + kv / kKvRowsR) * kWindow + row) *
                                     kKvRowsR + kv % kKvRowsR;
          args.kv_raw[offset] = __float2bfloat16_rn(value);
        });

    grid_barrier(args.barrier);

    // ---------------------------------------------------------------- phase 2: k_norm, RoPE, scatter
    if (blockIdx.x >= kFinishCtas) return;
    const int L = blockIdx.x / kWindow;
    const int m = blockIdx.x - L * kWindow;
    const std::size_t krow = ((std::size_t(L) * 2 + 0) * kWindow + m) * kKvRowsR;
    const std::size_t vrow = ((std::size_t(L) * 2 + 1) * kWindow + m) * kKvRowsR;
    const std::int64_t dst = args.kv_dst[m];
    const std::size_t out = (std::size_t(L) * args.cache_rows + std::size_t(dst)) * kKvRowsR;

    // Threads 0..127 carry k (norm + rope), 128..255 carry v (straight through). In the original body that
    // is warps 0-3 for k, 4-7 for v, and the helper warps holding neither -- but ALL of them run to the end, and
    // the two `__syncthreads()` below sit OUTSIDE the branch: a barrier only part of a CTA reaches
    // is undefined behaviour. The early `return` above is safe for the same reason inverted -- it is
    // uniform over the CTA, so a returning CTA reaches neither barrier.
    const bool is_k = tid < kKvRowsR;
    const bool is_v = tid >= kKvRowsR && tid < 2 * kKvRowsR;

    // v needs neither norm nor rope.
    if (is_v) args.kv_v[out + (tid - kKvRowsR)] = args.kv_raw[vrow + (tid - kKvRowsR)];

    const float x = is_k ? __bfloat162float(args.kv_raw[krow + tid]) : 0.0f;
    float s = x * x;
  #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, 32);
    if (lane == 0) reduce_s[warp] = s;
    __syncthreads();
    // The four warp sums in ascending warp order -- stated, because the order is the contract. Only
    // warps 0..3 hold k, so only their sums are read.
    float total = 0.0f;
  #pragma unroll
    for (int w2 = 0; w2 < kKvRowsR / 32; ++w2) total += reduce_s[w2];

    const float scale = rsqrtf(total / float(kHeadDim) + args.rms_epsilon);
    // Round once here: the stand-in's `_head_rmsnorm` writes bf16 into `k_ctx` and `_rope_dev` reads
    // that bf16 back, so the rope input is the ROUNDED normed value, not the fp32 one.
    if (is_k) head_s[tid] = __float2bfloat16((x * scale) * args.k_norm[L][tid]);
    __syncthreads();
    if (!is_k) return;

    const float f = __bfloat162float(head_s[tid]);
    const float rot = (tid < kHeadDim / 2) ? -__bfloat162float(head_s[tid + kHeadDim / 2])
                                           : __bfloat162float(head_s[tid - kHeadDim / 2]);
    const std::size_t rp = std::size_t(args.ctx_pos[m]) * kHeadDim + tid;
    // Not fmaf: the stand-in multiplies twice and adds once, and contraction would move a ulp.
    const float roped = __fadd_rn(__fmul_rn(f, args.rope_cos[rp]), __fmul_rn(rot, args.rope_sin[rp]));
    args.kv_k[out + tid] = __float2bfloat16(roped);
  }
};

}  // namespace fastllm_draft_ops
