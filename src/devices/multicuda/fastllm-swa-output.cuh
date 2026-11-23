#pragma once

// Single-row attention output projection, TP reduction and norm.
// Row-major BF16 weights are borrowed; no packed or quantized copy is made.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-bf16-ordered-dot.cuh"
#include <cstdint>

namespace fastllm_swa_output {
using BF16 = __nv_bfloat16;
constexpr int Width = 4096, Inner = 1024, Ranks = 8;
constexpr int Frames = 3, MaxBlocks = 128, MaxRows = 8;
constexpr uint64_t Period = 0xffffffffULL;
constexpr size_t SlotBytes = Width / 2 * sizeof(uint64_t);
constexpr size_t FrameBytes = MaxRows * Ranks * SlotBytes;
struct Peers { char *data[Ranks]{}; };
__device__ __forceinline__ void Store(char *p, uint64_t word) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(p), "l"(word) : "memory");
#endif
}
__device__ __forceinline__ uint64_t Load(const char *p) {
    uint64_t word = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(word) : "l"(p) : "memory");
#endif
    return word;
}

// All ranks use one ordered stream, and every invocation consumes all peer
// publications before completing. Three tagged frames protect a slower reader
// from a producer one invocation ahead, including repeated Graph replay.
template<int Tile = 32>
__global__ __launch_bounds__(256) void OutputNorm(
        const BF16 *input, const BF16 *weight, BF16 *projected,
        BF16 *hidden, const float *normWeight, BF16 *normed, float eps,
        Peers peers, uint64_t *epochs, int rank) {
    __shared__ BF16 partial[Tile];
    __shared__ uint64_t step;
    __shared__ float sums[8];
    __shared__ bool last;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int first = blockIdx.x * Tile;
    if (t == 0) {
        // Each epoch fits in 32 bits. Reserve the unused high word of
        // epochs[0] for the local completion counter; initialize via the
        // existing zeroed epoch allocation and never update it here.
        auto *generation = reinterpret_cast<uint32_t *>(epochs + blockIdx.x);
        step = *generation;
        *generation = step + 1 == Period ? 0 : step + 1;
    }
#pragma unroll
    for (int j = warp; j < Tile; j += 8) {
        BF16 value = fastllm::cuda::Bf16OrderedWarpDot<Inner>(input, weight + size_t(first + j) * Inner);
        if (lane == 0) partial[j] = value;
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    const size_t base = (step % Frames) * FrameBytes;
    __shared__ uint32_t received[Ranks][Tile / 2];
    // Each lane owns one (peer, BF16 pair). Publish every peer before
    // polling, then acquire the independent sources concurrently.
    if (t < Ranks * (Tile / 2)) {
        const int peer = t / (Tile / 2), index = t % (Tile / 2);
        const int wordIndex = first / 2 + index;
        const uint32_t bits = reinterpret_cast<const uint32_t *>(partial)[index];
        Store(peers.data[peer] + base + rank * SlotBytes + wordIndex * 8,
            (uint64_t(bits) << 32) | tag);
    }
    __syncthreads();
    if (t < Ranks * (Tile / 2)) {
        const int peer = t / (Tile / 2), index = t % (Tile / 2);
        const int wordIndex = first / 2 + index;
        uint64_t word;
        do { word = Load(peers.data[rank] + base + peer * SlotBytes + wordIndex * 8); }
        while (uint32_t(word) != tag);
        received[peer][index] = uint32_t(word >> 32);
    }
    __syncthreads();
    if (t < Tile / 2) {
        const int wordIndex = first / 2 + t;
        float a = 0, b = 0;
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            const uint32_t data = received[peer][t];
            const float x = float(__ushort_as_bfloat16(data & 65535u));
            const float y = float(__ushort_as_bfloat16(data >> 16));
            // Fetch order changes, but accumulation remains rank 0..7 with
            // the original BF16 projection and reduction boundaries.
            if (peer == 0) { a = x; b = y; } else { a += x; b += y; }
        }
        const BF16 x = __float2bfloat16_rn(a), y = __float2bfloat16_rn(b);
        projected[wordIndex * 2] = x; projected[wordIndex * 2 + 1] = y;
        hidden[wordIndex * 2] = __float2bfloat16_rn(float(hidden[wordIndex * 2]) + float(x));
        hidden[wordIndex * 2 + 1] = __float2bfloat16_rn(float(hidden[wordIndex * 2 + 1]) + float(y));
    }
    // Every writer makes its hidden stores globally visible before the
    // CTA contributes to the counter. The final CTA then reads through
    // L2 below, avoiding stale L1 lines from another CTA. Stream ordering
    // prevents a new invocation until this invocation's norm is complete.
    if (t < Tile / 2) __threadfence();
    __syncthreads();
    if (t == 0)
        last = atomicInc(reinterpret_cast<unsigned *>(epochs) + 1,
            gridDim.x - 1) == gridDim.x - 1;
    __syncthreads();
    if (last) {
        float values[16], affine[16], sum = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned short bits;
            asm volatile("ld.global.cg.u16 %0, [%1];"
                : "=h"(bits) : "l"(hidden + t + j * 256) : "memory");
            values[j] = float(__ushort_as_bfloat16(bits));
            affine[j] = normWeight[t + j * 256];
            sum += values[j] * values[j];
        }
        const float inv = rsqrtf(fastllm::cuda::WarpFirstBlockSum256(sum, sums) / Width + eps);
#pragma unroll
        for (int j = 0; j < 16; ++j)
            normed[t + j * 256] = __float2bfloat16_rn(float(__float2bfloat16_rn(values[j] * inv)) * affine[j]);
    }
}
} // namespace fastllm_swa_output
