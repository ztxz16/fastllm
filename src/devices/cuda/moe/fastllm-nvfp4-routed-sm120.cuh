// Routed NVFP4 projections. Each 16x64 weight tile is packed
// once in SM120 MMA register order. Decode and grouped prefill share the layout.
#pragma once
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include <cuda_bf16.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp8.h>
#endif
#include <cstdint>
#include <algorithm>
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
template <int Rows> struct RouteGroup {
    int expert, count, first, routes[Rows];
};
// Expert rows are compact. Only the MMA input tiles include tail padding.
struct PlanSize { int tiles, rows; };
template <int Rows> struct CompactRouteTiles {
    static constexpr int tileRows = Rows;
    const RouteGroup<Rows> *tasks;
    const PlanSize *used;
    __device__ bool active(int tile) const { return tile < used->tiles; }
    __device__ const RouteGroup<Rows> &operator[](int tile) const { return tasks[tile]; }
};

// Borrow a padded expert-major route table, e.g. from a preceding grouped GEMM.
template <int Rows> struct PaddedRouteTiles {
    static constexpr int tileRows = Rows;
    const int *counts, *offsets, *tileExperts, *routes;
    int experts;
    struct Tile { int expert, count, first; const int *routes; };
    __device__ bool active(int tile) const { return tile * Rows < offsets[experts]; }
    __device__ Tile operator[](int tile) const {
        const int first = tile * Rows, e = tileExperts[tile];
        return {e, min(Rows, counts[e] - (first - offsets[e])), first, routes + first};
    }
};
// One CTA per expert gives each route a deterministic slot without host reads.
template <int TileRows>
static __global__ void groupRoutes(const int *ids, int routes, RouteGroup<TileRows> *tiles, PlanSize *used) {
    __shared__ int prefix[8], total, base, first;
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
        first = atomicAdd(&used->rows, n);
        base = atomicAdd(&used->tiles, (n + TileRows - 1) / TileRows);
    }
    __syncthreads();
    if (!total) return;
    for (int i = tid; i < (total + TileRows - 1) / TileRows; i += blockDim.x) {
        tiles[base + i].expert = expert;
        tiles[base + i].first = first + i * TileRows;
        tiles[base + i].count = min(TileRows, total - TileRows * i);
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
            tiles[base + pos / TileRows].routes[pos % TileRows] = r;
        }
        __syncthreads();
    }
}
// Grouped prefill reuses each activation fragment across output-channel tiles.
struct InputTile {
    uint2 words[32];
    uint32_t scales[8];
};
template <class Plan>
static __global__ void packInput(Plan plan, const Block16 *input,
                                 InputTile *packed, int k, int divisor) {
    constexpr int TileRows = Plan::tileRows;
    const int tile = blockIdx.y;
    if (!plan.active(tile)) return;
    const int lane = threadIdx.x % 32;
    const int band = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (band >= k / 64) return;
    const auto &task = plan[tile];
    const int group = blockIdx.z;
    if (8 * group >= task.count) return;
    const int route = task.routes[min(int(8 * group + lane / 4), task.count - 1)];
    const auto *src = input + size_t(route / divisor) * (k / 16) + band * 4;
    auto &dst = packed[(size_t(tile) * (k / 64) + band) * (TileRows / 8) + group];
    dst.words[lane] = make_uint2(src[(lane % 4) / 2].code[lane % 2], src[2 + (lane % 4) / 2].code[lane % 2]);
    if (!(lane % 4))
        dst.scales[lane / 4] = src[0].scale | (src[1].scale << 8) | (src[2].scale << 16) | (src[3].scale << 24);
}
// A lane handles columns i and i+8 of a 16-value scale block. The first
// addition below and the following 8-lane butterfly preserve the error
// reduction order used by the scalar four/six scale selector.
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
__device__ inline unsigned quantizePair(float2 v, float scale, float &error) {
    float lo = scale ? v.x / scale : 0.f, hi = scale ? v.y / scale : 0.f;
    unsigned byte, halves;
    asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
        "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
        : "=r"(byte), "=r"(halves) : "f"(lo), "f"(hi));
    const float2 q = make_float2(__half2float(__ushort_as_half(halves)),
                                 __half2float(__ushort_as_half(halves >> 16)));
    const float2 reconstructed = __fmul2_rn(q, make_float2(scale, scale));
    const float2 difference = __fadd2_rn(v, make_float2(-reconstructed.x, -reconstructed.y));
    error = fabsf(difference.x) + fabsf(difference.y);
    return byte;
}
__device__ inline unsigned roundPair(float2 v, float scale, float &error) {
    const unsigned byte = quantizePair(v, scale, error);
    for (int delta = 4; delta; delta >>= 1)
        error += __shfl_xor_sync(0xffffffff, error, delta, 8);
    return byte;
}
#endif

// Keep activations in expert order between the two GEMMs. Each row writes
// the four scale bytes and eight FP4 words in exactly the down-MMA layout.
template <int TileRows, class Activation>
static __global__ void activatePacked(Activation act, const RouteGroup<TileRows> *tasks, const PlanSize *used, int width,
                                      InputTile *packed, float *rowScales, bool cache) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    const int tile = blockIdx.x / TileRows, local = blockIdx.x % TileRows;
    if (tile >= used->tiles) return;
    const auto &task = tasks[tile];
    if (local / 8 * 8 >= task.count) return;
    const int index = min(local, task.count - 1), row = task.first + index, route = task.routes[index];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int bands = width / 64, part = lane / 8;
    extern __shared__ float2 values[];
    __shared__ float partial[4];
    float peak = 0;
    for (int b = warp; b < bands; b += 4) {
        int col = b * 64 + part * 16 + lane % 8;
        float2 v = make_float2(act.get(row, route, col, width), act.get(row, route, col + 8, width));
        if (cache) values[b * 32 + lane] = v;
        peak = fmaxf(peak, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    for (int d = 16; d; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, d));
    if (lane == 0) partial[warp] = peak;
    __syncthreads();
    float outer = fmaxf(fmaxf(partial[0], partial[1]), fmaxf(partial[2], partial[3])) / 1536.f;
    if (!threadIdx.x && local == index) rowScales[row] = outer;
    for (int b = warp; b < bands; b += 4) {
        int col = b * 64 + part * 16 + lane % 8;
        float2 v = cache ? values[b * 32 + lane] :
                   make_float2(act.get(row, route, col, width), act.get(row, route, col + 8, width));
        float m = fmaxf(fabsf(v.x), fabsf(v.y));
        for (int d = 4; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d, 8));
        float ratio = outer > 0 ? m / outer : 0;
        __nv_fp8_e4m3 s4(fminf(448.f, ratio / 4)), s6(fminf(448.f, ratio / 6));
        float err4, err6;
        unsigned q4 = roundPair(v, float(s4) * outer, err4);
        unsigned q6 = roundPair(v, float(s6) * outer, err6);
        unsigned q = err4 < err6 ? q4 : q6;
        unsigned scale = err4 < err6 ? s4.__x : s6.__x;
        unsigned low = (q & 15) << (4 * (lane % 8)), high = (q >> 4) << (4 * (lane % 8));
        for (int d = 1; d < 8; d <<= 1) {
            low |= __shfl_xor_sync(0xffffffff, low, d, 8);
            high |= __shfl_xor_sync(0xffffffff, high, d, 8);
        }
        InputTile &dst = packed[(size_t(tile) * bands + b) * (TileRows / 8) + local / 8];
        const unsigned half = (lane & 1) ? high : low;
        const int src = (lane & 2) * 4 + (lane & 1);
        const uint2 words = make_uint2(__shfl_sync(0xffffffff, half, src),
                                       __shfl_sync(0xffffffff, half, src + 16));
        if (lane < 4) dst.words[(local % 8) * 4 + lane] = words;
        scale <<= 8 * part;
        scale |= __shfl_xor_sync(0xffffffff, scale, 8);
        scale |= __shfl_xor_sync(0xffffffff, scale, 16);
        if (!lane) dst.scales[local % 8] = scale;
    }
#endif
}
template <int TileRows, class Activation>
static void launchPackedActivation(Activation act, const RouteGroup<TileRows> *tasks, const PlanSize *used, int tiles,
                                   int width, InputTile *packed, float *rowScales) {
    const size_t bytes = size_t(width) * sizeof(float);
    const bool cache = bytes + 4 * sizeof(float) <= 48 * 1024;
    activatePacked<<<tiles * TileRows, 128, cache ? bytes : 0, cudaStreamPerThread>>>(
        act, tasks, used, width, packed, rowScales, cache);
}

// Register-resident quantization for a row that fits in a CTA. Each lane
// owns whole 16-value scale blocks, including their two candidate encodings.
constexpr int QuantThreads = 128;
struct EncodedBlock { uint2 words; unsigned scale; };

#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
__device__ inline uint2 encodeCandidate(const float (&v)[16], float scale, float &error) {
    float errors[8];
    uint2 words{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned byte = quantizePair(make_float2(v[i], v[i + 8]), scale, errors[i]);
        words.x |= (byte & 15) << (4 * i);
        words.y |= (byte >> 4) << (4 * i);
    }
    for (int step = 4; step; step /= 2) {
#pragma unroll
        for (int i = 0; i < step; ++i) errors[i] += errors[i + step];
    }
    error = errors[0];
    return words;
}
__device__ inline EncodedBlock encodeBlock(const float (&values)[16], float peak, float outer) {
    const float ratio = outer ? peak / outer : 0.f;
    __nv_fp8_e4m3 s4(fminf(448.f, ratio / 4)), s6(fminf(448.f, ratio / 6));
    float e4, e6;
    uint2 q4 = encodeCandidate(values, float(s4) * outer, e4);
    uint2 q6 = encodeCandidate(values, float(s6) * outer, e6);
    return e4 < e6 ? EncodedBlock{q4, s4.__x} : EncodedBlock{q6, s6.__x};
}
// Four lanes form one 64-column MMA band. Shuffle once before replication to experts.
__device__ inline EncodedBlock packBand(uint2 words, unsigned scale) {
    const int lane = threadIdx.x % 4;
    const uint2 pair = make_uint2(__shfl_sync(__activemask(), words.x, lane / 2, 4),
                            __shfl_sync(__activemask(), words.x, 2 + lane / 2, 4));
    const uint2 hi = make_uint2(__shfl_sync(__activemask(), words.y, lane / 2, 4),
                          __shfl_sync(__activemask(), words.y, 2 + lane / 2, 4));
    scale <<= (8 * lane);
    scale |= __shfl_xor_sync(__activemask(), scale, 1, 4);
    scale |= __shfl_xor_sync(__activemask(), scale, 2, 4);
    return {lane & 1 ? hi : pair, scale};
}
__device__ inline void storeBand(InputTile &dst, int local, EncodedBlock band) {
    const int lane = threadIdx.x % 4;
    dst.words[local * 4 + lane] = band.words;
    if (!lane) dst.scales[local] = band.scale;
}
#endif

// A lane owns one complete FP4 block. Keep the BF16-rounded activation in
// registers across the row scale reduction, then write the MMA layout directly.
template <int TileRows, class Activation>
static __global__ void activateBlocks(Activation act, const RouteGroup<TileRows> *tasks, const PlanSize *used,
                                      int width, InputTile *packed, float *rowScales) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    const int tile = blockIdx.x / TileRows, local = blockIdx.x % TileRows;
    if (tile >= used->tiles) return;
    const auto &task = tasks[tile];
    if (local / 8 * 8 >= task.count) return;
    const int idx = min(local, task.count - 1), row = task.first + idx, group = threadIdx.x;
    float values[16], peak = 0;
    if (group < width / 16) {
        act.loadBlock(row, task.routes[idx], group * 16, width, values);
#pragma unroll
        for (float value : values) peak = fmaxf(peak, fabsf(value));
    }
    float maxValue = peak;
    for (int d = 16; d; d /= 2) maxValue = fmaxf(maxValue, __shfl_xor_sync(0xffffffff, maxValue, d));
    __shared__ float partial[QuantThreads / 32];
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = maxValue;
    __syncthreads();
    for (int w = 0; w < blockDim.x / 32; ++w) maxValue = fmaxf(maxValue, partial[w]);
    const float outer = maxValue / 1536.f;
    if (!group && idx == local) rowScales[row] = outer;
    if (group < width / 16) {
        const auto enc = encodeBlock(values, peak, outer);
        auto &dst = packed[(size_t(tile) * (width / 64) + group / 4) * (TileRows / 8) + local / 8];
        storeBand(dst, local % 8, packBand(enc.words, enc.scale));
    }
#endif
}

// Prefixes over experts suffice once every selected route has its own local rank.
template <int TileRows>
static __global__ void expertOffsets(const int *counts, int2 *starts, int experts, PlanSize *size) {
    int2 base = make_int2(0, 0);
    const int lane = threadIdx.x;
    for (int first = 0; first < experts; first += 32) {
        int e = first + lane, rows = e < experts ? counts[e] : 0, tiles = (rows + TileRows - 1) / TileRows;
        int r = rows, t = tiles;
        for (int d = 1; d < 32; d *= 2) {
            int a = __shfl_up_sync(0xffffffff, r, d), b = __shfl_up_sync(0xffffffff, t, d);
            if (lane >= d) { r += a; t += b; }
        }
        if (e < experts) starts[e] = make_int2(base.x + r - rows, base.y + t - tiles);
        base.x += __shfl_sync(0xffffffff, r, 31);
        base.y += __shfl_sync(0xffffffff, t, 31);
    }
    if (!lane) *size = {base.y, base.x};
}
// A token is quantized once and its packed bands are written at their
// final addresses in all selected experts, including the last partial band of rows.
template <int TileRows, bool Local = false>
static __global__ void scatterInput(const float *x, int width, const int *ids, const int *ranks,
                                    const int *counts, const int2 *starts, int top,
                                    RouteGroup<TileRows> *tasks, InputTile *packed, float *rowScales) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    constexpr int Threads = QuantThreads, Blocks = 2;
    __shared__ int slots[64], repeats[64];
    __shared__ float maxima[Threads / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int r = row * top + tid;
    const int e = tid < top ? ids[r] : -1;
    if constexpr (Local) {
        if (!__syncthreads_or(e >= 0)) return;
        if (tid < top && e < 0) slots[tid] = -1;
    }
    if (tid < top && (!Local || e >= 0)) {
        const int rank = ranks[r], count = counts[e];
        const int tile = starts[e].y + rank / TileRows, local = rank % TileRows;
        auto &task = tasks[tile];
        if (!local) {
            task.expert = e;
            task.first = starts[e].x + rank;
            task.count = min(TileRows, count - rank);
        }
        task.routes[local] = r;
        slots[tid] = tile * TileRows + local;
        repeats[tid] = rank == count - 1 ? 8 - local % 8 : 1;
    }
    float values[Blocks][16], peaks[Blocks], peak = 0;
#pragma unroll
    for (int b = 0; b < Blocks; ++b) {
        const int group = tid + b * Threads;
        if (group >= width / 16) continue;
        const float4 *src = reinterpret_cast<const float4 *>(x + size_t(row) * width + group * 16);
        float m = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = src[i];
            values[b][4*i] = v.x; values[b][4*i+1] = v.y;
            values[b][4*i+2] = v.z; values[b][4*i+3] = v.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
        peaks[b] = m;
        peak = fmaxf(peak, m);
    }
    for (int d = 16; d; d /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, d));
    if (!(tid % 32)) maxima[tid / 32] = peak;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < Threads / 32; ++w) peak = fmaxf(peak, maxima[w]);
    const float outer = peak / 1536.f;
    if (!tid) rowScales[row] = outer;
#pragma unroll
    for (int b = 0; b < Blocks; ++b) {
        int group = tid + b * Threads;
        if (group >= width / 16) continue;
        const auto enc = encodeBlock(values[b], peaks[b], outer);
        const auto band = packBand(enc.words, enc.scale);
        for (int i = 0; i < top; ++i) {
            const int slot = slots[i];
            if constexpr (Local) if (slot < 0) continue;
            const int tile = slot / TileRows, local = slot % TileRows;
            auto &dst = packed[(size_t(tile) * (width / 64) + group / 4) * (TileRows / 8) + local / 8];
            for (int j = 0; j < repeats[i]; ++j)
                storeBand(dst, local % 8 + j, band);
        }
    }
#endif
}

constexpr int WarpTileRows = 64;
// A warp reuses each activation fragment across ColumnTiles adjacent 16-column
// weight tiles. Up writes compact expert rows; down restores the original routes.
template <int Warps, int ColumnTiles, bool Up, bool Compact = false, class Plan>
static __global__ void groupedProject(Weights w, const Block16 *input, const float *rowScales,
                                      const InputTile *packed, Plan plan,
                                      __nv_bfloat16 *out, int n, int k, int divisor, int outputStride = 0) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    constexpr int TileRows = Plan::tileRows;
    static_assert(TileRows % 8 == 0 && (TileRows <= WarpTileRows || TileRows % WarpTileRows == 0));
    constexpr int groups = (TileRows < WarpTileRows ? TileRows : WarpTileRows) / 8;
    const int tile = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int rowGroup = (warp / Warps) * groups;
    const int first = (blockIdx.x * Warps + warp % Warps) * ColumnTiles;
    const int columns = n / 16, bands = k / 64;
    if (!plan.active(tile)) return;
    const auto &task = plan[tile];
    const int count = task.count;
    if (rowGroup * 8 >= count || first >= columns) return;
    const size_t base = (size_t(task.expert) * columns + first) * bands;
    float sums[groups][ColumnTiles][4]{};
    for (int band = 0; band < bands; ++band) {
        uint4 weight[ColumnTiles];
        unsigned scale[ColumnTiles];
#pragma unroll
        for (int c = 0; c < ColumnTiles; ++c) {
            const size_t addr = base + min(c, columns - first - 1) * bands + band;
            weight[c] = w.codes[addr * 32 + lane];
            scale[c] = w.scales[addr * 16 + lane / 4 * 2 + (lane & 1)];
        }
#pragma unroll
        for (int g = 0; g < groups; ++g) {
            if ((rowGroup + g) * 8 >= count) continue;
            const InputTile &x = packed[(size_t(tile) * bands + band) * (TileRows / 8) + rowGroup + g];
            const uint2 b = x.words[lane];
            const unsigned sb = x.scales[lane / 4];
#pragma unroll
            for (int c = 0; c < ColumnTiles; ++c) {
                auto &v = sums[g][c];
                const auto a = weight[c];
                BlockMma::fma(v[0], v[1], v[2], v[3], a.x, a.y, a.z, a.w, b.x, b.y,
                              v[0], v[1], v[2], v[3], scale[c], sb);
            }
        }
    }
#pragma unroll
    for (int g = 0; g < groups; ++g) {
        for (int j = 0; j < 2; ++j) {
            const int col = (rowGroup + g) * 8 + 2 * (lane % 4) + j;
            if (col >= count) continue;
            const int route = task.routes[col], row = task.first + col;
            float outer;
            if constexpr (!Up) outer = rowScales ? rowScales[row] : input[size_t(route) * (k / 16)].outer;
            else if constexpr (Compact) outer = rowScales[route / divisor];
            else outer = input[size_t(route / divisor) * (k / 16)].outer;
#pragma unroll
            for (int c = 0; c < ColumnTiles; ++c) {
                if (first + c < columns) {
                    auto *dst = out + size_t(Up ? row : route) * (outputStride ? outputStride : n) + (first + c) * 16 + lane / 4;
                    const auto &v = sums[g][c];
                    dst[0] = __float2bfloat16_rn(v[j] * w.outer[task.expert] * outer);
                    dst[8] = __float2bfloat16_rn(v[j + 2] * w.outer[task.expert] * outer);
                }
            }
        }
    }
#endif
}
// One MMA has eight input columns. Two warp ballots can match up to 64 routes.
// A single ballot of routes shows little reuse in small-batch verification.
inline int decodeRouteTileRows(int rows, int topK) {
    const int tile = std::min({rows, 8, 64 / topK});
    return tile > 1 && tile * topK > 32 ? tile : 0;
}

// The bits encode route indices within one tile. Slots follow their original routing order.
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
__device__ __forceinline__ int matchingRoute(unsigned low, unsigned high, int slot) {
    const int lowerCount = __popc(low);
    return slot < lowerCount ? __fns(low, 0, slot + 1) : 32 + __fns(high, 0, slot - lowerCount + 1);
}
#endif
// One warp computes sixteen output channels with a native 16x8x64 MMA.
// Matching routes fill its input columns. Both paths split K between warps in a CTA.
enum class DecodeDependency { Serial, Gate, Down };
template <int Warps, bool MatchRoutes = false, DecodeDependency Edge = DecodeDependency::Serial, bool LocalExperts = false>
static __global__ void decodeProject(Weights w, const Block16 *input, const int *ids, __nv_bfloat16 *out, int n,
                                     int k, int divisor, int routes = 0, int capacity = 0, int outputStride = 0) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    static_assert(!MatchRoutes || Edge != DecodeDependency::Down);
    if constexpr (Edge == DecodeDependency::Gate) cuda_launch::DependencyStart();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first = MatchRoutes ? blockIdx.z * capacity : 0;
    const int local = blockIdx.y, route = first + local;
    if constexpr (MatchRoutes) {
        if (route >= routes) return;
    }
    const int expert = ids[route];
    if constexpr (LocalExperts) {
        if (expert < 0) {
            if constexpr (Edge == DecodeDependency::Down) cuda_launch::DependencyStart();
            if (threadIdx.x < 16)
                out[size_t(route) * n + blockIdx.x * 16 + threadIdx.x] = __float2bfloat16_rn(0.f);
            return;
        }
    }
    int src = route, count = 1;
    unsigned low = 0, high = 0;
    if constexpr (MatchRoutes) {
        static_assert(Warps == 4);
        const int live = min(capacity, routes - first);
        if (live >= 32) {
            low = __ballot_sync(0xffffffff, lane < live && ids[first + lane] == expert);
            high = __ballot_sync(0xffffffff, lane + 32 < live && ids[first + 32 + lane] == expert);
            const int leader = low ? __ffs(low) - 1 : 32 + __ffs(high) - 1;
            if (local != leader) return;
            count = __popc(low) + __popc(high);
            if (count > 1) src = first + matchingRoute(low, high, min(lane / 4, count - 1));
        }
    }
    const Block16 *x = input + size_t(src / divisor) * (k / 16);
    const size_t base = (size_t(expert) * (n / 16) + blockIdx.x) * (k / 64);
    // The gate waited for the router and the activation waited for the gate.
    // Only the down input is outstanding here. Stage the selected expert's
    // first weight tile while that activation is still being produced.
    uint4 firstWeight{};
    unsigned firstScale = 0;
    if constexpr (Edge == DecodeDependency::Down) {
        if (warp < k / 64) {
            firstWeight = w.codes[(base + warp) * 32 + lane];
            firstScale = w.scales[(base + warp) * 16 + lane / 4 * 2 + (lane & 1)];
        }
        cuda_launch::DependencyStart();
    }
    float v[4] = {};
    for (int t = warp; t < k / 64; t += Warps) {
        const uint4 a = Edge == DecodeDependency::Down && t == warp ? firstWeight : w.codes[(base + t) * 32 + lane];
        const unsigned sa = Edge == DecodeDependency::Down && t == warp ? firstScale : w.scales[(base + t) * 16 + lane / 4 * 2 + (lane & 1)];
        const unsigned b0 = x[t * 4 + (lane % 4) / 2].code[lane & 1],
                       b1 = x[t * 4 + 2 + (lane % 4) / 2].code[lane & 1];
        const unsigned sb =
            x[t * 4].scale | (x[t * 4 + 1].scale << 8) | (x[t * 4 + 2].scale << 16) | (x[t * 4 + 3].scale << 24);
        BlockMma::fma(v[0], v[1], v[2], v[3], a.x, a.y, a.z, a.w, b0, b1, v[0], v[1], v[2], v[3], sa, sb);
    }
    if constexpr (MatchRoutes) {
        if (count > 1) {
            // MMA columns hold different routes. Preserve the singleton's order of split-K reduction.
            __shared__ float partials[Warps][4][32];
#pragma unroll
            for (int j = 0; j < 4; ++j) partials[warp][j][lane] = v[j];
            __syncthreads();
            const int slot = 2 * (lane % 4) + (warp & 1);
            if (slot < count) {
                float sum = 0;
#pragma unroll
                for (int part = 0; part < Warps; ++part) sum += partials[part][warp][lane];
                const int dst = first + matchingRoute(low, high, slot);
                const float outer = input[size_t(dst / divisor) * (k / 16)].outer;
                out[size_t(dst) * (outputStride ? outputStride : n) + blockIdx.x * 16 + lane / 4 + 8 * (warp / 2)] =
                    __float2bfloat16_rn(sum * w.outer[expert] * outer);
            }
            return;
        }
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
        out[size_t(route) * (outputStride ? outputStride : n) + blockIdx.x * 16 + lane] = __float2bfloat16_rn(sum * w.outer[expert] * x[0].outer);
    }
#endif
}
} // namespace fastllm::cuda::routed_fp4
