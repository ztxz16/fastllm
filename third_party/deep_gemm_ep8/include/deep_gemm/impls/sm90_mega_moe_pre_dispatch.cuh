#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

namespace fastllm_ep8_deep_gemm {

struct CopySelectedRoutes {
    template<uint32_t Stride>
    __device__ void write(uint32_t row, uint32_t k, const int32_t *ids, const float *weights,
                          int64_t *outIds, float *outWeights, float scale) const {
        if (threadIdx.x < k) {
            const uint32_t slot = row * k + threadIdx.x;
            outIds[slot] = ids[slot];
            outWeights[slot * Stride] = weights[slot] * scale;
        }
    }
};

template<uint32_t kGroupSize, bool kPowerOfTwoScale, uint32_t kSfStride, uint32_t kWeightStride>
__device__ __forceinline__ void quantize_input_vector(const __nv_bfloat16 *x, __nv_fp8_e4m3 *buf_x,
        float *buf_x_sf, uint32_t token_id, uint32_t tid, uint32_t hidden, uint32_t num_groups) {
    static_assert(kGroupSize == 128, "SM90 mega-moe pre-dispatch requires per-128 SF");
    constexpr uint32_t kVecElems = 8;  // 16-byte BF16 load per thread
    static_assert(kGroupSize % kVecElems == 0, "kGroupSize must be a multiple of 8");
    constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;

    const auto* token_in = x + static_cast<uint64_t>(token_id) * hidden;

    uint4 in_bits;
    if (kPowerOfTwoScale && (reinterpret_cast<uintptr_t>(token_in) % alignof(uint4)) != 0) {
        // The MXFP4 adapter also accepts BF16 views with an unaligned first row.
        auto* values = reinterpret_cast<__nv_bfloat16*>(&in_bits);
        #pragma unroll
        for (uint32_t i = 0; i < kVecElems; ++i)
            values[i] = token_in[tid * kVecElems + i];
    } else {
        in_bits = reinterpret_cast<const uint4*>(token_in)[tid];
    }
    const auto* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(&in_bits);

    float vals[kVecElems];
    float local_max = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
        float2 fp = __bfloat1622float2(bf16_pairs[i]);
        vals[2 * i + 0] = fp.x;
        vals[2 * i + 1] = fp.y;
        local_max = fmaxf(local_max, fmaxf(fabsf(fp.x), fabsf(fp.y)));
    }

    local_max = math::warp_reduce<kThreadsPerGroup, false>(
        local_max, math::ReduceMax<float>{});

    const float raw_scale = kPowerOfTwoScale
        ? exp2f(ceilf(log2f(fmaxf(local_max / 448.0f, 0x1p-126f))))
        : fmaxf(local_max, 1e-10f) / 448.0f;
    const float inv_scale = 1.0f / raw_scale;

    uint64_t packed = 0;
    #pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
        const __nv_fp8x2_storage_t fp8x2 = __nv_cvt_float2_to_fp8x2(
            make_float2(vals[2 * i + 0] * inv_scale,
                        vals[2 * i + 1] * inv_scale),
            __NV_SATFINITE, __NV_E4M3);
        packed |= static_cast<uint64_t>(fp8x2) << (16u * i);
    }
    auto* row_out = reinterpret_cast<uint64_t*>(buf_x) +
                    static_cast<uint64_t>(token_id) * (hidden / 8u);
    row_out[tid] = packed;

    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group_id = tid % kThreadsPerGroup;
    if (within_group_id == 0u && group_id < num_groups) {
        // TP-local descriptors read scales in [K group, token] order.
        const uint32_t off = kSfStride == 1 ? token_id * num_groups + group_id
                                            : group_id * kSfStride + token_id * kWeightStride;
        buf_x_sf[off] = raw_scale;
    }
}

template <uint32_t kGroupSize, bool kUsePDL, bool kPowerOfTwoScale = false,
          uint32_t kSfStride = 1, uint32_t kWeightStride = 1, class Routes = CopySelectedRoutes>
__launch_bounds__(1024, 2)
__global__ void sm90_mega_moe_pre_dispatch_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t*       __restrict__ topk_idx,
    const float*         __restrict__ topk_weights,
    __nv_fp8_e4m3*       __restrict__ buf_x,
    float*               __restrict__ buf_x_sf,
    int64_t*             __restrict__ buf_topk_idx,
    float*               __restrict__ buf_topk_weights,
    const uint32_t num_tokens,
    const uint32_t padded_max,
    const uint32_t hidden,
    const uint32_t num_groups,
    const uint32_t top_k,
    const float routed_scaling_factor,
    const int* __restrict__ live_lengths = nullptr, Routes routes = {}) {
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if constexpr (kUsePDL) {
        cudaGridDependencySynchronize();
    }

    if (bid < num_tokens) {
        const uint32_t token_id = bid;
        // Whole-block decision before reading activations. Keep the source
        // shape/collectives unchanged but send no experts for inactive rows.
        if (live_lengths && live_lengths[token_id] <= 0) {
            if (tid < top_k) {
                const uint32_t off = token_id * top_k + tid;
                buf_topk_idx[off] = -1;
                buf_topk_weights[off * kWeightStride] = 0.f;
            }
            if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
            return;
        }
        quantize_input_vector<kGroupSize,kPowerOfTwoScale,kSfStride,kWeightStride>(
            x,buf_x,buf_x_sf,token_id,tid,hidden,num_groups);

        routes.template write<kWeightStride>(token_id, top_k, topk_idx, topk_weights,
                buf_topk_idx, buf_topk_weights, routed_scaling_factor);
    } else {
        const uint32_t copy_bid = bid - num_tokens;
        const uint32_t pad_base = num_tokens * top_k;
        const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
        const uint32_t total = padded_max * top_k;
        if (slot < total) {
            buf_topk_idx[slot] = static_cast<int64_t>(-1);
            buf_topk_weights[slot * kWeightStride] = 0.0f;
        }
    }

    if constexpr (kUsePDL) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace deep_gemm
