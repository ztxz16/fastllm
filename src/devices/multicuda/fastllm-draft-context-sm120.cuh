#pragma once
#include "fastllm-draft-layer-sm120.cuh"

namespace sm120_draft {
struct ContextWork {
    alignas(16) B feature[8][4096];
    float incoming[8][32][128];
    alignas(16) B projected[8][4096], normalized[8][4096];
    float kvPartial[2][1280][8];
    unsigned ready[2][8];
};
struct ContextArgs {
    ContextWork *peers[8];
    const uint4 *fc, *kv;
    const float *gamma, *kNorm[5];
    const B *feature;
    B *cache[10];
    uint64_t *nextBindings;
    int rank, count, old, position, anchor;
    unsigned epoch;
    FastllmDraftSamplingRequest request;
    float eps, theta;
};
// The rank consumes columns belonging to one of the eight captured target layers.
__global__ void PackContextFc(const B *weight, uint4 *packed, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = i % 32;
    int tile = i / 8192, k = i / 32 % 256;
    const B *p = weight + size_t(tile * 16 + lane / 4) * 32768 + rank * 4096 + k * 16 + lane % 4 * 2;
    packed[i] = make_uint4(Pair(p), Pair(p + 8 * 32768), Pair(p + 8), Pair(p + 8 * 32768 + 8));
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
__device__ __forceinline__ void NormalizeContextRow(const B *src, const float *gamma, B *out, B *copy,
                                                    float eps, float *tmp) {
    const int t = threadIdx.x;
    float values[16], sum = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        B x = src[j * Threads + t];
        if (copy)
            copy[j * Threads + t] = x;
        values[j] = float(x);
        sum = fmaf(values[j], values[j], sum);
    }
    float inv = rsqrtf(fastllm::cuda_norm::WarpFirstBlockSum256(sum, tmp) / 4096 + eps);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        out[j * Threads + t] =
            __float2bfloat16_rn(float(__float2bfloat16_rn(values[j] * inv)) * gamma[j * Threads + t]);
}
// The feature barrier separates all sliding-window reads from all writes. Each
// CTA moves contiguous packets, even when the source and destination overlap.
template <bool Store>
__device__ __forceinline__ void MoveContextTile(const ContextArgs &a, int keep, uint4 (&tile)[4]) {
    const int b = blockIdx.x, drop = a.old - keep;
    if (drop <= 0 || b >= 160)
        return;
    auto *cache = reinterpret_cast<uint4 *>(a.cache[b / 16]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int packet = (b % 16) * 1024 + j * Threads + threadIdx.x;
        if (packet < keep * 16) {
            if constexpr (Store)
                cache[packet] = tile[j];
            else
                tile[j] = cache[packet + drop * 16];
        }
    }
}
template <int Splits>
__device__ __forceinline__ void ContextProjection(const B *x, const uint4 *w, int tile, int split,
                                                  float (&parts)[8][128]) {
    int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const B *row = x + lane / 4 * 4096 + lane % 4 * 2;
    const uint4 *panel = w + tile * 8192 + lane;
    float even[4]{}, odd[4]{};
#pragma unroll 4
    for (int k = split * (256 / Splits) + warp; k < (split + 1) * (256 / Splits); k += 16) {
        Mma(even, panel[k * 32], Pair(row + k * 16), Pair(row + k * 16 + 8));
        Mma(odd, panel[(k + 8) * 32], Pair(row + (k + 8) * 16), Pair(row + (k + 8) * 16 + 8));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        even[i] += odd[i];
    ReduceWarps(even, parts);
}
__device__ __forceinline__ void PublishContext(const ContextArgs &a, int phase,
                                               cooperative_groups::grid_group grid) {
    if (threadIdx.x < 128 && (!phase || blockIdx.x < 32))
        __threadfence_system();
    grid.sync();
    if (!blockIdx.x && threadIdx.x < 8) {
        int dest = threadIdx.x;
        CustomArStoreRelease(&a.peers[dest]->ready[phase][a.rank], a.epoch);
        while (CustomArLoadAcquire(&a.peers[a.rank]->ready[phase][dest]) != a.epoch) {
        }
    }
    grid.sync();
}
__device__ __forceinline__ void WriteContextKV(const ContextArgs &a, ContextWork &w, int keep, float *tmp,
                                               B *normed) {
    int row = blockIdx.x % 8, layer = blockIdx.x / 8, t = threadIdx.x;
    if (layer >= 5 || row >= a.count)
        return;
    int col = layer * 256 + t;
    B raw = __float2bfloat16_rn(w.kvPartial[0][col][row] + w.kvPartial[1][col][row]);
    float x = t < 128 ? float(raw) : 0.f;
    float inv = rsqrtf(fastllm::cuda_norm::WarpFirstBlockSum256(x * x, tmp) / 128 + a.eps);
    int base = (keep + row) * 128;
    if (t < 128)
        normed[t] = __float2bfloat16_rn(float(__float2bfloat16_rn(x * inv)) * a.kNorm[layer][t]);
    else
        a.cache[2 * layer + 1][base + t - 128] = raw;
    __syncthreads();
    if (t < 64) {
        float angle = float(a.position + row) * powf(a.theta, -float(t) / 64);
        B c = __float2bfloat16_rn(cosf(angle)), s = __float2bfloat16_rn(sinf(angle));
        B lo = normed[t], hi = normed[t + 64];
        B *key = a.cache[2 * layer] + base + t;
        key[0] = __hsub(__hmul_rn(lo, c), __hmul_rn(hi, s));
        key[64] = __hadd(__hmul_rn(hi, c), __hmul_rn(lo, s));
    }
}
#endif
__global__ __launch_bounds__(Threads, 3) void FusedDraftContextSm120(
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
    __grid_constant__
#endif
        const ContextArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    auto grid = cooperative_groups::this_grid();
    auto &w = *a.peers[a.rank];
    int b = blockIdx.x, t = threadIdx.x;
    int keep = min(a.old, 1023 - a.count);
    __shared__ float parts[8][128];
    __shared__ B normed[128];
    // Hand the next graph its KV bindings, accepted token and new absolute length.
    if (a.nextBindings && !b) {
        if (t == 11 && !a.rank && a.request.probabilities)
            *reinterpret_cast<FastllmDraftSamplingRequest *>(a.nextBindings + 11) = a.request;
        if (t < 10)
            a.nextBindings[t] = reinterpret_cast<uint64_t>(a.cache[t]);
        if (t == 10)
            a.nextBindings[10] = uint64_t(__float_as_uint(float(a.anchor))) |
                                 (uint64_t(unsigned(a.position + a.count + 1)) << 32);
    }
    if (!t)
        PrefetchWeight(a.fc + b * 8192, 131072);
    int packet = b * Threads + t;
    if (packet < 4096)
        reinterpret_cast<uint4 *>(w.feature)[packet] =
            packet < a.count * 512 ? reinterpret_cast<const uint4 *>(a.feature)[packet]
                                   : make_uint4(0, 0, 0, 0);
    uint4 retained[4];
    MoveContextTile<false>(a, keep, retained);
    grid.sync();
    MoveContextTile<true>(a, keep, retained);
    ContextProjection<1>(&w.feature[0][0], a.fc, b, 0, parts);
    if (t < 128)
        a.peers[b / 32]->incoming[a.rank][b % 32][t] = ColumnSum(parts, t);
    PublishContext(a, 0, grid);
    B *values = reinterpret_cast<B *>(parts);
    if (b < 32 && t < 128) {
        float sum = 0;
#pragma unroll
        for (int r = 0; r < 8; ++r)
            sum += w.incoming[r][b][t];
        values[t % 8 * 16 + t / 8] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    if (b < 32) {
        if (t < 8 * 16) {
            int dest = t / 16, packet = t % 16;
            reinterpret_cast<uint4 *>(a.peers[dest]->projected[packet / 2] + a.rank * 512 +
                                      b * 16)[packet % 2] = reinterpret_cast<uint4 *>(values)[packet];
        }
    }
    PublishContext(a, 1, grid);
    if (b < 8)
        NormalizeContextRow(w.projected[b], a.gamma, w.normalized[b], nullptr, a.eps, parts[0]);
    grid.sync();
    if (b < 160) {
        ContextProjection<2>(&w.normalized[0][0], a.kv, b / 2, b % 2, parts);
        if (t < 128)
            w.kvPartial[b % 2][b / 2 * 16 + t / 8][t % 8] = ColumnSum(parts, t);
    }
    grid.sync();
    WriteContextKV(a, w, keep, parts[0], normed);
#else
    asm("trap;");
#endif
}
__global__ void PrepareDraftEmbeddingSm120(const float *id, const int *live, const B *embed, const B *mask,
                                           const float *gamma, B *hidden, B *normed, float *positions,
                                           float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    cudaTriggerProgrammaticLaunchCompletion();
    int row = blockIdx.x;
    __shared__ float tmp[8];
    if (!threadIdx.x)
        positions[row] = float(*live - 1 + row);
    NormalizeContextRow(row ? mask : embed + size_t(int(*id)) * 4096, gamma, normed + row * 4096,
                        hidden + row * 4096, eps, tmp);
#else
    asm("trap;");
#endif
}
} // namespace sm120_draft
