// Single-device routed NVFP4 projections. Each 16x64 weight tile is packed
// once in SM120 MMA register order. Decode and grouped prefill share the layout.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#if CUDART_VERSION >= 12080 && !defined(CUDA_NO_TENSOR_CORE) && __has_include(<cute/arch/mma_sm120.hpp>)
#include <cute/arch/mma_sm120.hpp>
#endif
namespace fastllm::cuda::routed_fp4 {
static __device__ const int nativeImage =
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    1;
#else
    0;
#endif
struct Block16 {
    uint32_t code[2], scale;
    float outer;
};
struct Weights {
    const uint4 *codes;
    const uint32_t *scales;
    const float *outer;
};
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
using BlockMma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cute::float_e2m1_t, cute::float_e2m1_t, float,
                                                               cute::float_ue4m3_t, 16>;
#endif

static __global__ void pack(const uint32_t *raw, const uint32_t *scale, uint4 *dst, uint32_t *sf, int n, int k,
                            bool interleaved) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n / 16) * (k / 64) * 32) return;
    const int tile = i / 32, lane = i % 32, band = tile % (k / 64);
    const int row = tile / (k / 64) * 16 + lane / 4;
    const int r0 = interleaved ? (row % (n / 2)) * 2 + row / (n / 2) : row;
    const int r1 = interleaved ? ((row + 8) % (n / 2)) * 2 + (row + 8) / (n / 2) : row + 8;
    const int col = band * 8 + lane % 4;
    const uint32_t s0 = scale[size_t(r0) * (k / 64) + band], s1 = scale[size_t(r1) * (k / 64) + band];
    // MMA accepts unsigned scales. Move any E4M3 sign to its sixteen codes.
    const int shift = 8 * ((lane % 4) / 2) + 7;
    auto signMask = [](uint32_t s, int bit) { return ((s >> bit) & 1u) * 0x88888888u; };
    dst[i] = make_uint4(raw[size_t(r0) * (k / 8) + col] ^ signMask(s0, shift),
                        raw[size_t(r1) * (k / 8) + col] ^ signMask(s1, shift),
                        raw[size_t(r0) * (k / 8) + col + 4] ^ signMask(s0, shift + 16),
                        raw[size_t(r1) * (k / 8) + col + 4] ^ signMask(s1, shift + 16));
    if (lane % 4 < 2) sf[tile * 16 + lane / 4 * 2 + lane % 4] = (lane % 4 ? s1 : s0) & 0x7f7f7f7fu;
}
constexpr int RouteTile = 32;
struct Tile {
    int expert, count, routes[RouteTile];
};
// One CTA per expert gives each route a deterministic slot without host reads.
static __global__ void groupRoutes(const int *ids, int routes, Tile *tiles, int *used) {
    __shared__ int prefix[8], total, base;
    const int tid = threadIdx.x, expert = blockIdx.x;
    int count = 0;
    for (int i = tid; i < routes; i += blockDim.x)
        count += ids[i] == expert;
    for (int d = 16; d; d >>= 1)
        count += __shfl_down_sync(0xffffffff, count, d);
    if (!(tid % 32)) prefix[tid / 32] = count;
    __syncthreads();
    if (!tid) {
        int n = 0;
        for (int w = 0; w < 8; ++w)
            n += prefix[w];
        total = n;
        base = atomicAdd(used, (n + RouteTile - 1) / RouteTile);
    }
    __syncthreads();
    if (!total) return;
    for (int i = tid; i < (total + RouteTile - 1) / RouteTile; i += blockDim.x) {
        tiles[base + i].expert = expert;
        tiles[base + i].count = min(RouteTile, total - RouteTile * i);
    }
    int offset = 0;
    for (int start = 0; start < routes; start += blockDim.x) {
        const int r = start + tid;
        const unsigned mask = __ballot_sync(0xffffffff, r < routes && ids[r] == expert);
        if (!(tid % 32)) prefix[tid / 32] = __popc(mask);
        __syncthreads();
        int pos = offset;
        for (int i = 0; i < 8; ++i) {
            if (i < tid / 32) pos += prefix[i];
            offset += prefix[i];
        }
        if ((mask >> (tid % 32)) & 1u) {
            pos += __popc(mask & ((1u << (tid % 32)) - 1));
            tiles[base + pos / RouteTile].routes[pos % RouteTile] = r;
        }
        __syncthreads();
    }
}
// Grouped prefill reuses each activation fragment across output-channel tiles.
struct InputTile {
    uint2 words[32];
    uint32_t scales[8];
};
static __global__ void packInput(const Tile *tasks, const int *used, const Block16 *input, InputTile *packed,
                                 int k, int divisor) {
    const int tile = blockIdx.y;
    if (tile >= *used) return;
    const int lane = threadIdx.x % 32;
    const int band = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (band >= k / 64) return;
    const Tile &task = tasks[tile];
    const int group = blockIdx.z;
    if (8 * group >= task.count) return;
    const int route = task.routes[min(int(8 * group + lane / 4), task.count - 1)];
    const auto *src = input + size_t(route / divisor) * (k / 16) + band * 4;
    auto &dst = packed[(size_t(tile) * (k / 64) + band) * (RouteTile / 8) + group];
    dst.words[lane] = make_uint2(src[(lane % 4) / 2].code[lane % 2], src[2 + (lane % 4) / 2].code[lane % 2]);
    if (!(lane % 4))
        dst.scales[lane / 4] = src[0].scale | (src[1].scale << 8) | (src[2].scale << 16) | (src[3].scale << 24);
}
template <int Warps>
static __global__ void groupedProject(Weights w, const Block16 *input, const InputTile *packed, const Tile *tasks,
                                      const int *used, __nv_bfloat16 *out, int n, int k, int divisor) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    const int tile = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int outTile = blockIdx.x * Warps + warp;
    if (tile >= *used || outTile >= n / 16) return;
    const Tile &task = tasks[tile];
    const int expert = task.expert;
    const size_t base = (size_t(expert) * (n / 16) + outTile) * (k / 64);
    constexpr int groups = RouteTile / 8;
    float v[groups][4]{};
    for (int t = 0; t < k / 64; ++t) {
        uint4 a = w.codes[(base + t) * 32 + lane];
        const unsigned sa = w.scales[(base + t) * 16 + lane / 4 * 2 + (lane & 1)];
#pragma unroll
        for (int g = 0; g < groups; ++g) {
            if (g * 8 >= task.count) continue;
            const InputTile &x = packed[(size_t(tile) * (k / 64) + t) * groups + g];
            const uint2 b = x.words[lane];
            auto &c = v[g];
            BlockMma::fma(c[0], c[1], c[2], c[3], a.x, a.y, a.z, a.w, b.x, b.y, c[0], c[1], c[2], c[3], sa,
                          x.scales[lane / 4]);
        }
    }
#pragma unroll
    for (int g = 0; g < groups; ++g) {
        for (int j = 0; j < 2; ++j) {
            const int col = g * 8 + 2 * (lane % 4) + j;
            if (col < task.count) {
                const int dest = task.routes[col], ch = outTile * 16 + lane / 4;
                const float rowScale = input[size_t(dest / divisor) * (k / 16)].outer;
                out[size_t(dest) * n + ch] = __float2bfloat16_rn(v[g][j] * w.outer[expert] * rowScale);
                out[size_t(dest) * n + ch + 8] = __float2bfloat16_rn(v[g][j + 2] * w.outer[expert] * rowScale);
            }
        }
    }
#endif
}
// One warp computes sixteen output channels with a native 16x8x64 MMA.
// Decode has one routed vector per expert. Split K between warps in a CTA.
template <int Warps>
static __global__ void decodeProject(Weights w, const Block16 *input, const int *ids, __nv_bfloat16 *out, int n,
                                     int k, int divisor) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    const int route = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int expert = ids[route];
    const Block16 *x = input + size_t(route / divisor) * (k / 16);
    const size_t base = (size_t(expert) * (n / 16) + blockIdx.x) * (k / 64);
    float v[4] = {};
    for (int t = warp; t < k / 64; t += Warps) {
        const uint4 a = w.codes[(base + t) * 32 + lane];
        const unsigned sa = w.scales[(base + t) * 16 + lane / 4 * 2 + (lane & 1)];
        const unsigned b0 = x[t * 4 + (lane % 4) / 2].code[lane & 1],
                       b1 = x[t * 4 + 2 + (lane % 4) / 2].code[lane & 1];
        const unsigned sb =
            x[t * 4].scale | (x[t * 4 + 1].scale << 8) | (x[t * 4 + 2].scale << 16) | (x[t * 4 + 3].scale << 24);
        BlockMma::fma(v[0], v[1], v[2], v[3], a.x, a.y, a.z, a.w, b0, b1, v[0], v[1], v[2], v[3], sa, sb);
    }
    __shared__ float partial[Warps][16];
    if (!(lane % 4)) {
        partial[warp][lane / 4] = v[0];
        partial[warp][lane / 4 + 8] = v[2];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0;
#pragma unroll
        for (int i = 0; i < Warps; ++i)
            sum += partial[i][lane];
        out[size_t(route) * n + blockIdx.x * 16 + lane] = __float2bfloat16_rn(sum * w.outer[expert] * x[0].outer);
    }
#endif
}
} // namespace fastllm::cuda::routed_fp4
