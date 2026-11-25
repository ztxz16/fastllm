// Eight BF16 verification rows, 256 FP32 router weights and 64 Marlin routes.
#pragma once
#include "verify_expert_tile.cuh"
#include "devices/cuda/fastllm-norm-rows-router.cuh"
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace fastllm_marlin_moe {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void ProjectVerifyRouterDots(const __nv_bfloat16 *input, const float *weight,
                                                        float *probabilities) {
    const int thread = threadIdx.x, lane = thread % 32;
    const int pair = thread % 128, firstRow = thread / 128 * 4, expert = blockIdx.x;
    float2 sums[4]{};
    // Two adjacent virtual lanes per thread. Keep sixteen FP32 FMAs per
    // virtual lane before the first pairwise addition of the reference tree.
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int col = k * 256 + pair * 2;
        const auto w = *reinterpret_cast<const float2 *>(weight + expert * 4096 + col);
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const auto x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                input + (firstRow + row) * 4096 + col));
            sums[row].x = __fmaf_rn(x.x, w.x, sums[row].x);
            sums[row].y = __fmaf_rn(x.y, w.y, sums[row].y);
        }
    }
    __shared__ float warpSums[8][4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        float sum = sums[row].x + sums[row].y;
#pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0) warpSums[firstRow + row][pair / 32] = sum;
    }
    __syncthreads();
    const int row = thread / 32;
    float sum = lane < 4 ? warpSums[row][lane] : 0.f;
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (lane == 0) probabilities[row * 256 + expert] = 1.0 / (1.0 + expf(-(sum + 0.f)));
}

__device__ __forceinline__ int VerifyRouteByte(uint64_t list, int position) {
    return int((list >> (position * 8)) & 255);
}

// Reproduce the legacy 64-list merge on exact ties. Each lane first combines
// its two four-entry lists without truncation. Later merges take the right
// prefix first at a tied cutoff and place retained left entries first on ties.
__device__ int SelectTiedVerifyRouter(const float *probabilities, const float *bias) {
    const int lane = threadIdx.x % 32;
    __shared__ float rowKeys[8][256];
    float *keys = rowKeys[threadIdx.x / 32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int id = lane + 32 * i;
        const float key = probabilities[id] + bias[id];
        keys[id] = isfinite(key) ? key : -FLT_MAX;
    }
    __syncwarp();
    uint64_t list = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned id = lane + (i / 4) * 32 + (i % 4) * 64;
        int pos = i;
        while (pos > 0 && keys[id] > keys[VerifyRouteByte(list, pos - 1)]) --pos;
        const uint64_t low = (uint64_t(1) << (pos * 8)) - 1;
        list = (list & low) | ((list & ~low) << 8) | (uint64_t(id) << (pos * 8));
    }
    for (int stride = 16; stride; stride /= 2) {
        const uint64_t right = __shfl_down_sync(0xffffffff, list, stride);
        if (lane >= stride) continue;
        int lo = 0, hi = 8;
        while (lo < hi) {
            const int middle = (lo + hi) / 2;
            if (keys[VerifyRouteByte(list, middle)] > keys[VerifyRouteByte(right, 7 - middle)]) lo = middle + 1;
            else hi = middle;
        }
        int a = 0, b = 0;
        uint64_t result = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const bool left = b == 8 - lo ||
                (a < lo && keys[VerifyRouteByte(list, a)] >= keys[VerifyRouteByte(right, b)]);
            const int id = left ? VerifyRouteByte(list, a++) : VerifyRouteByte(right, b++);
            result |= uint64_t(id) << (k * 8);
        }
        list = result;
    }
    list = __shfl_sync(0xffffffff, list, 0);
    return VerifyRouteByte(list, lane % 8);
}

__device__ void SelectVerifyRouterRow(const float *probabilities, const float *bias,
                                      int *ids, float *scores, bool normalize, float routeScale) {
    const int lane = threadIdx.x % 32;
    unsigned keys[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int id = i * 32 + lane;
        float key = probabilities[id] + bias[id];
        if (!isfinite(key)) key = -FLT_MAX;
        const unsigned bits = __float_as_uint(key == 0.f ? 0.f : key);
        keys[i] = bits ^ (bits >> 31 ? ~0u : 0x80000000u);
    }
    int selected = 0;
    bool tied = false;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        unsigned best = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) best = max(best, keys[i]);
        const unsigned winner = __reduce_max_sync(0xffffffff, best);
        unsigned matches = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (keys[i] != winner) continue;
            matches |= 1u << i;
            keys[i] = 0;
        }
        const unsigned owners = __ballot_sync(0xffffffff, matches != 0);
        tied |= (owners & (owners - 1)) != 0 || (matches & (matches - 1)) != 0;
        const int id = __shfl_sync(0xffffffff, (__ffs(matches) - 1) * 32 + lane, __ffs(owners) - 1);
        if (lane == k) selected = id;
    }
    if (__any_sync(0xffffffff, tied)) selected = SelectTiedVerifyRouter(probabilities, bias);
    float probability = lane < 8 ? probabilities[selected] : 0.f;
    if (!isfinite(probability)) probability = 0.f;
    float sum = 1.f;
    if (normalize) {
        sum = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) sum += __shfl_sync(0xffffffff, probability, k);
        if (!isfinite(sum) || fabsf(sum) < 1.e-20f) sum = 1.f;
    }
    if (lane < 8) {
        ids[lane] = selected;
        scores[lane] = probability / sum * routeScale;
    }
}

// The eight choices in each row are distinct, so an expert has at most
// eight routes. Its two warp masks give both the tile leader and slot order.
__device__ void GroupVerifyRouterRows(const int *ids, int *sorted, int *experts, int *padded) {
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    __shared__ unsigned members[2][256], leaders[2];
    members[0][t] = members[1][t] = 0;
    __syncthreads();
    const int id = t < 64 ? ids[t] : 0;
    if (t < 64) {
        const unsigned mask = __match_any_sync(0xffffffff, id);
        if (lane == __ffs(mask) - 1) members[warp][id] = mask;
    }
    __syncthreads();
    const unsigned low = t < 64 ? members[0][id] : 0, high = t < 64 ? members[1][id] : 0;
    const int leader = low ? __ffs(low) - 1 : 32 + __ffs(high) - 1;
    if (t < 64) {
        const unsigned bits = __ballot_sync(0xffffffff, t == leader);
        if (lane == 0) leaders[warp] = bits;
    }
    __syncthreads();
    if (t == 0) *padded = (__popc(leaders[0]) + __popc(leaders[1])) * 8;
    if (t >= 64) return;
    const int w = leader / 32, l = leader % 32;
    const int tile = (w ? __popc(leaders[0]) : 0) + __popc(leaders[w] & ((1u << l) - 1));
    const int slot = (warp ? __popc(low) : 0) + __popc((warp ? high : low) & ((1u << lane) - 1));
    if (t == leader) {
        experts[tile] = id;
        for (int i = __popc(low) + __popc(high); i < 8; ++i) sorted[tile * 8 + i] = 64;
    }
    sorted[tile * 8 + slot] = t;
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void FinishVerifyRoutes(float *probs, const float *bias, int32_t *ids,
        float *scores, int32_t *sortedIds, int32_t *experts, int32_t *padded, unsigned *finished,
        bool normalize, float scale) {
    const int t = threadIdx.x;
    __syncthreads();
    __shared__ bool last;
    if (t == 0) {
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device> ticket(*finished);
        last = ticket.fetch_add(1, ::cuda::memory_order_acq_rel) == gridDim.x * gridDim.y - 1;
        if (last) ticket.store(0, ::cuda::memory_order_relaxed);
    }
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (!last) return;
    const int row = t / 32;
    SelectVerifyRouterRow(probs + row * 256, bias, ids + row * 8, scores + row * 8, normalize, scale);
    __syncthreads();
    GroupVerifyRouterRows(ids, sortedIds, experts, padded);
}
#endif

constexpr int VerifyRouterExperts = 8, VerifyRouterRows = 4;

template <bool WaitForSwa>
__global__ void NormRowsMarlinVerifyRoutes(const __nv_bfloat16 *hidden, const float *gamma,
        __nv_bfloat16 *normed, float eps, const float *weight, float *probs, const float *bias,
        int32_t *ids, float *scores, int32_t *sortedIds, int32_t *experts, int32_t *padded,
        unsigned *finished, bool normalize, float scale, int *verifyReady) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < fastllm_marlin_verify::kRoutes)
        verifyReady[threadIdx.x] = 0;
    fastllm_norm_rows::NormRowsRouterBody<VerifyRouterExperts, VerifyRouterRows, false, 8, false, WaitForSwa, true>(
        hidden, weight, probs, gamma, normed, eps, 8, {});
    FinishVerifyRoutes(probs, bias, ids, scores, sortedIds, experts, padded, finished, normalize, scale);
#endif
}

__global__ void ProjectMarlinVerifyRoutesSm120(const __nv_bfloat16 *input, const float *weight,
        float *probs, const float *bias, int32_t *ids, float *scores, int32_t *sortedIds,
        int32_t *experts, int32_t *padded, unsigned *finished, bool normalize, float scale, int *verifyReady) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int t = threadIdx.x;
    if (verifyReady && blockIdx.x == 0 && t < fastllm_marlin_verify::kRoutes) verifyReady[t] = 0;
    ProjectVerifyRouterDots(input, weight, probs);
    FinishVerifyRoutes(probs, bias, ids, scores, sortedIds, experts, padded, finished, normalize, scale);
#endif
}
}  // namespace fastllm_marlin_moe
