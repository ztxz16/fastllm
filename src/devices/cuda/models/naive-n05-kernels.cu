#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <algorithm>
#include <cmath>
#include <vector>

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define FASTLLM_NAIVE_SWA_FLASHINFER
#include "naive-n05-swa-flashinfer.cuh"
#define FASTLLM_NAIVE_DSA_MMA
#include "naive-n05-dsa-mma.cuh"
#endif

namespace {
using BF16 = __nv_bfloat16;
__device__ float RoundBF16(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ float WarpSum(float x) {
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}
void Output(fastllm::Data &out, fastllm::DataType type, const std::vector<int> &dims) {
    out.dataType = type;
    out.Resize(dims);
    out.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    out.Allocate(false);
}
void CheckLaunch() {
    auto status = cudaGetLastError();
    fastllm::AssertInFastLLM(status == cudaSuccess,
        std::string("Naive-N0.5 CUDA: ") + cudaGetErrorString(status));
}

// Each thread owns a disjoint vector/scalar column. Moving rows in increasing order
// is overlap-safe even when dropping just one row; no other thread touches
// that column. K and V share a launch and retain their reserved capacity.
template <typename T>
__global__ void TrimCachePair(T *key, T *value, int keyColumns,
                              int valueColumns, int drop, int keep) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    T *data = column < keyColumns ? key : value;
    int columns = column < keyColumns ? keyColumns : valueColumns;
    if (column >= keyColumns) column -= keyColumns;
    if (column >= columns) return;
    for (int row = 0; row < keep; ++row)
        data[(size_t)row * columns + column] = data[(size_t)(row + drop) * columns + column];
}

constexpr int kTrimTileBytes = 64;
constexpr int kTrimMaxRows = 128;

// A CTA owns a disjoint 64-byte column tile and stages all retained rows
// before writing. The barrier makes overlapping suffix moves safe while rows
// are copied in parallel; other CTAs never read or write these columns.
template <typename T>
__global__ void TrimCachePairTiled(T *key, T *value, int keyColumns,
                                 int valueColumns, int drop, int keep) {
    constexpr int columns = kTrimTileBytes / sizeof(T);
    __shared__ T rows[kTrimMaxRows * columns];
    int keyBlocks = (keyColumns + columns - 1) / columns;
    bool isKey = blockIdx.x < keyBlocks;
    T *data = isKey ? key : value;
    int stride = isKey ? keyColumns : valueColumns;
    int firstColumn = (isKey ? blockIdx.x : blockIdx.x - keyBlocks) * columns;
    for (int i = threadIdx.x; i < keep * columns; i += blockDim.x) {
        int row = i / columns, col = firstColumn + i % columns;
        if (col < stride) rows[i] = data[(size_t)(row + drop) * stride + col];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < keep * columns; i += blockDim.x) {
        int row = i / columns, col = firstColumn + i % columns;
        if (col < stride) data[(size_t)row * stride + col] = rows[i];
    }
}


template <typename T>
void LaunchTrimCachePair(T *key, T *value, int keyColumns,
                        int valueColumns, int drop, int keep) {
    if (keep <= kTrimMaxRows) {
        constexpr int columns = kTrimTileBytes / sizeof(T);
        int blocks = (keyColumns + columns - 1) / columns +
                     (valueColumns + columns - 1) / columns;
        TrimCachePairTiled<<<blocks, 256>>>(key, value, keyColumns, valueColumns, drop, keep);
    } else {
        TrimCachePair<<<(keyColumns + valueColumns + 255) / 256, 256>>>(
            key, value, keyColumns, valueColumns, drop, keep);
    }
}

__device__ unsigned OrderedScoreBits(float score) {
    unsigned bits = score == 0.0f ? 0u : __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ unsigned long long TopKOrder(unsigned scoreBits, int index) {
    return ((unsigned long long)scoreBits << 32) | (0xffffffffu - (unsigned)index);
}

// Sorting the original score bits plus the inverse position preserves the CPU
// comparator, including +/-0 ties. Separate causal segment ends exclude future
// keys even when a valid score is -infinity; no score matrix leaves the GPU.
__global__ void EncodeTopK(const float *scores, unsigned long long *order,
                           int *offsets, int queries, int keys, int queryStart) {
    int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    int count = queryStart + row + 1;
    if (offsets && col == 0) {
        offsets[row] = row * keys;
        offsets[queries + row] = row * keys + count;
    }
    if (col >= count) return;
    order[(size_t)row * keys + col] =
        TopKOrder(OrderedScoreBits(scores[(size_t)row * keys + col]), col);
}
__global__ void DecodeTopK(const unsigned long long *order, int *indices,
                           int keys, int queryStart, int topK) {
    int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= topK) return;
    indices[(size_t)row * topK + col] = col <= queryStart + row
        ? (int)(0xffffffffu - (unsigned)order[(size_t)row * keys + col]) : -1;
}

// CUB radix sort is stable: ascending input positions resolve equal score
// bits without sorting an additional 32-bit position suffix.
__global__ void EncodeTopKPairs(const float *scores, unsigned *bits, int *positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        bits[i] = OrderedScoreBits(scores[i]);
        positions[i] = i;
    }
}

__global__ void DecodeTopKPairs(const int *positions, int *indices, int count, int topK) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < topK) indices[i] = i < count ? positions[i] : -1;
}

// Select a bounded superset of the top K before sorting it. A radix prefix can
// stop as soon as at most capacity entries remain; no possible winner is lost.
// If equal scores straddle a full candidate buffer, retain the lowest indices.
__global__ void SelectTopKCompact(const float *scores, unsigned long long *selected, int *offsets,
                                  int queries, int keys, int queryStart, int topK, int capacity) {
    constexpr int threads = 256, warps = threads / 32;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int histogram[warps][256];
    __shared__ unsigned prefix;
    __shared__ int rank, candidateCount, selectedCount, tiesSeen;
    int row = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    int count = queryStart + row + 1, keep = min(count, topK);
    const float *input = scores + (size_t)row * keys;
    unsigned long long *output = selected + (size_t)row * capacity;
    if (t == 0) {
        offsets[row] = row * capacity;
        prefix = 0; rank = keep; selectedCount = 0; tiesSeen = 0;
    }
    __syncthreads();
    if (count <= capacity) {
        if (t == 0) offsets[queries + row] = row * capacity + count;
        for (int col = t; col < count; col += threads)
            output[col] = TopKOrder(OrderedScoreBits(input[col]), col);
        return;
    }
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        unsigned wanted = prefix;
        int target = rank;
        for (int i = t; i < warps * 256; i += threads) ((int *)histogram)[i] = 0;
        __syncthreads();
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            int bucket = col < count && (bits & mask) == wanted ? int((bits >> shift) & 255) : 256;
            // Aggregate identical buckets within each warp before its histogram
            // update, including the common case where all score exponents match.
            #if __CUDA_ARCH__ >= 700
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < 256 && lane == __ffs(peers) - 1)
                atomicAdd(&histogram[warp][bucket], __popc(peers));
            #else
            if (bucket < 256) atomicAdd(&histogram[warp][bucket], 1);
            #endif
        }
        __syncthreads();
        int bucket = 255 - t, n = 0;
        #pragma unroll
        for (int w = 0; w < warps; ++w) n += histogram[w][bucket];
        int before;
        Scan(scan).ExclusiveSum(n, before);
        if (before < target && before + n >= target) {
            prefix = wanted | ((unsigned)bucket << shift);
            rank = target - before;
            candidateCount = keep - rank + n;
        }
        __syncthreads();
        mask |= 255u << shift;
        if (candidateCount <= capacity) break;
    }
    unsigned threshold = prefix;
    int tiesWanted = rank;
    if (candidateCount <= capacity) {
        if (t == 0) offsets[queries + row] = row * capacity + candidateCount;
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            bool take = col < count && bits >= threshold;
            unsigned ballot = __ballot_sync(0xffffffff, take);
            if (ballot) {
                int outputBase = 0, leader = __ffs(ballot) - 1;
                if (lane == leader) outputBase = atomicAdd(&selectedCount, __popc(ballot));
                outputBase = __shfl_sync(0xffffffff, outputBase, leader);
                if (take)
                    output[outputBase + __popc(ballot & ((1u << lane) - 1))] =
                        TopKOrder(bits, col);
            }
        }
    } else {
        if (t == 0) offsets[queries + row] = row * capacity + keep;
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            int equal = col < count && bits == threshold, equalBefore, equalTotal;
            Scan(scan).ExclusiveSum(equal, equalBefore, equalTotal);
            __syncthreads();
            int take = col < count &&
                (bits > threshold || (equal && tiesSeen + equalBefore < tiesWanted));
            int before, total;
            Scan(scan).ExclusiveSum(take, before, total);
            __syncthreads();
            if (take)
                output[selectedCount + before] = TopKOrder(bits, col);
            __syncthreads();
            if (t == 0) { selectedCount += total; tiesSeen += equalTotal; }
            __syncthreads();
        }
    }
}

__global__ void Rope(BF16 *data, const float *positions, int heads, int dim,
                     int rotaryDim, float theta) {
    int row = blockIdx.x;
    BF16 *x = data + (size_t)row * dim;
    for (int d = threadIdx.x; d < rotaryDim / 2; d += blockDim.x) {
        float angle = positions[row / heads] * powf(theta, -2.0f * d / rotaryDim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)x[d], b = (float)x[d + rotaryDim / 2];
        // Match eager GPT-NeoX RoPE, including each BF16 multiplication.
        x[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        x[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
}

__global__ void RoundIndexer(const BF16 *input, float *output, int stride,
                             int offset, bool fp8) {
    __shared__ float maximum[128];
    int d = threadIdx.x, row = blockIdx.x;
    float x = (float)input[(size_t)row * stride + offset + d];
    maximum[d] = fabsf(x);
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (d < step) maximum[d] = fmaxf(maximum[d], maximum[d + step]);
        __syncthreads();
    }
    if (fp8) {
        float scale = fmaxf(maximum[0], 1e-4f) / 448.0f;
        x = (float)__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / scale))) * scale;
    }
    output[(size_t)row * 128 + d] = x;
}

__global__ void IndexScores(const float *q, const float *k, const BF16 *weights,
                            float *scores, int heads, int keys, int queryStart) {
    int query = blockIdx.y;
    int key = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (key >= keys) return;
    float score = 0;
    for (int h = 0; h < heads; h++) {
        float dot = 0;
        for (int d = lane; d < 128; d += 32)
            dot += q[((size_t)query * heads + h) * 128 + d] * k[(size_t)key * 128 + d];
        dot = WarpSum(dot);
        score += fmaxf(dot, 0.0f) * (float)weights[query * heads + h];
    }
    if (lane == 0)
        scores[(size_t)query * keys + key] = key <= queryStart + query ? score : -INFINITY;
}

// Decode consumes each packed K row once. Keep its original E4M3-rounded
// FP32 operands in registers across the 16 heads, avoiding a full temporary K.
__global__ void IndexScoresDecode(const float *q, const BF16 *packedKeys,
        const BF16 *weights, float *scores, int stride, int keys, int queryStart) {
    int key = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (key >= keys) return;
    float k[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        k[i] = (float)packedKeys[(size_t)key * stride + stride - 128 + lane + i * 32];
    float maximum = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[2])),
                         fmaxf(fabsf(k[1]), fabsf(k[3])));
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    float scale = fmaxf(__shfl_sync(0xffffffff, maximum, 0), 1e-4f) / 448.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        k[i] = (float)__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, k[i] / scale))) * scale;
    float score = 0;
    for (int h = 0; h < 16; ++h) {
        float dot = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            dot += q[h * 128 + lane + i * 32] * k[i];
        dot = WarpSum(dot);
        score += fmaxf(dot, 0.0f) * (float)weights[h];
    }
    if (lane == 0) scores[key] = key <= queryStart ? score : -INFINITY;
}

// Two 16-lane subgroups evaluate adjacent heads in parallel. Each lane owns
// the original lane and lane+16 partial sums, so the first add reproduces the
// old shuffle-by-16 stage. The remaining shuffle tree and head accumulation
// order are unchanged. Key fragments stay in registers across all 16 heads.
__global__ void IndexScoresPrefill(const float *__restrict__ q, const float *__restrict__ k,
                                   const BF16 *__restrict__ weights, float *__restrict__ scores,
                                   int keys, int queryStart) {
    int query = blockIdx.y, key = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32, column = lane % 16, headOffset = lane / 16;
    if (key >= keys) return;
    if (key > queryStart + query) {
        if (lane == 0) scores[(size_t)query * keys + key] = -INFINITY;
        return;
    }
    float keyLow[4], keyHigh[4];
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        keyLow[d] = k[(size_t)key * 128 + column + d * 32];
        keyHigh[d] = k[(size_t)key * 128 + column + 16 + d * 32];
    }
    float score = 0;
    #pragma unroll
    for (int head = 0; head < 16; head += 2) {
        const float *row = q + ((size_t)query * 16 + head + headOffset) * 128;
        float low = 0, high = 0;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            low += row[column + d * 32] * keyLow[d];
            high += row[column + 16 + d * 32] * keyHigh[d];
        }
        float dot = low + high;
        #pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset, 16);
        float first = __shfl_sync(0xffffffff, dot, 0);
        float second = __shfl_sync(0xffffffff, dot, 16);
        score += fmaxf(first, 0.0f) * (float)weights[query * 16 + head];
        score += fmaxf(second, 0.0f) * (float)weights[query * 16 + head + 1];
    }
    if (lane == 0) scores[(size_t)query * keys + key] = score;
}

constexpr int kIndexerKeysPerWarp = 16;
constexpr int kIndexerQueryTile = 2;
constexpr int kIndexerWarps = 4;

// Reuse each key fragment across two query rows, and each query-head fragment
// across 16 keys. Keep the original per-lane sums, shuffle tree and head order.
__global__ void IndexScoresPrefillTiled(const float *__restrict__ q, const float *__restrict__ k,
                                        const BF16 *__restrict__ weights, float *__restrict__ scores,
                                        int queries, int keys, int queryStart) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int column = lane % 16, headOffset = lane / 16;
    int firstKey = (blockIdx.x * kIndexerWarps + warp) * kIndexerKeysPerWarp;
    if (firstKey >= keys) return;
    float keyLow[kIndexerKeysPerWarp][4], keyHigh[kIndexerKeysPerWarp][4];
    #pragma unroll
    for (int j = 0; j < kIndexerKeysPerWarp; ++j) {
        int key = firstKey + j;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            keyLow[j][d] = key < keys ? k[(size_t)key * 128 + column + d * 32] : 0;
            keyHigh[j][d] = key < keys ? k[(size_t)key * 128 + column + 16 + d * 32] : 0;
        }
    }
    #pragma unroll
    for (int qi = 0; qi < kIndexerQueryTile; ++qi) {
        int query = blockIdx.y * kIndexerQueryTile + qi;
        if (query >= queries) break;
        if (firstKey > queryStart + query) {
            if (lane == 0) {
                #pragma unroll
                for (int j = 0; j < kIndexerKeysPerWarp; ++j)
                    if (firstKey + j < keys) scores[(size_t)query * keys + firstKey + j] = -INFINITY;
            }
            continue;
        }
        float sums[kIndexerKeysPerWarp] = {};
        #pragma unroll
        for (int head = 0; head < 16; head += 2) {
            const float *row = q + ((size_t)query * 16 + head + headOffset) * 128;
            float queryLow[4], queryHigh[4];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                queryLow[d] = row[column + d * 32];
                queryHigh[d] = row[column + 16 + d * 32];
            }
            float firstWeight = (float)weights[query * 16 + head];
            float secondWeight = (float)weights[query * 16 + head + 1];
            #pragma unroll
            for (int j = 0; j < kIndexerKeysPerWarp; ++j) {
                float low = 0, high = 0;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    low += queryLow[d] * keyLow[j][d];
                    high += queryHigh[d] * keyHigh[j][d];
                }
                float dot = low + high;
                #pragma unroll
                for (int offset = 8; offset; offset >>= 1)
                    dot += __shfl_down_sync(0xffffffff, dot, offset, 16);
                float first = __shfl_sync(0xffffffff, dot, 0);
                float second = __shfl_sync(0xffffffff, dot, 16);
                sums[j] += fmaxf(first, 0.0f) * firstWeight;
                sums[j] += fmaxf(second, 0.0f) * secondWeight;
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < kIndexerKeysPerWarp; ++j)
                if (firstKey + j < keys)
                    scores[(size_t)query * keys + firstKey + j] =
                        firstKey + j <= queryStart + query ? sums[j] : -INFINITY;
        }
    }
}

__device__ int KeyIndex(const int *indices, int query, int slot, int count,
                        int past, int window) {
    if (indices) return indices[(size_t)query * count + slot];
    return window ? max(0, past + query - window + 1) + slot : slot;
}

__global__ void AttentionScores(const BF16 *q, const BF16 *k, const int *indices,
                                float *scores, int heads, int kvHeads, int dim,
                                int keyStride, int keys, int count, int past, int window, bool causal) {
    int query = blockIdx.y, h = blockIdx.x;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int kvHead = h / (heads / kvHeads);
    // Decode spreads key tiles over SMs while preserving each dot's reduction.
    int first = gridDim.z > 1 ? blockIdx.z * 64 : 0;
    int end = gridDim.z > 1 ? min(count, first + 64) : count;
    for (int slot = first + warp; slot < end; slot += 8) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        float dot = 0;
        bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
        if (valid) {
            for (int d = lane; d < dim; d += 32)
                dot += (float)q[((size_t)query * heads + h) * dim + d] *
                       (float)k[(size_t)key * keyStride + kvHead * dim + d];
        }
        dot = WarpSum(dot);
        if (lane == 0)
            scores[((size_t)query * heads + h) * count + slot] =
                valid ? RoundBF16(RoundBF16(dot) * rsqrtf((float)dim)) : -INFINITY;
    }
}

// Keep each warp's original lane-strided dot product and reduction order,
// but retain Q in registers and interleave four independent selected keys.
// The bounded query register tile leaves larger head dimensions on the
// general kernel; this path is only used for multiple-query prefill.
template <int MaxDim>
__global__ void AttentionScoresPrefill(const BF16 *q, const BF16 *k, const int *indices,
                                      float *scores, int heads, int kvHeads, int dim,
                                      int keyStride, int keys, int count, int past,
                                      int window, bool causal) {
    constexpr int keysPerWarp = 4, warps = 4;
    int query = blockIdx.y, h = blockIdx.x;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int kvHead = h / (heads / kvHeads);
    float queryValues[MaxDim / 32];
    #pragma unroll
    for (int j = 0; j < MaxDim / 32; ++j)
        if (lane + j * 32 < dim)
            queryValues[j] = (float)q[((size_t)query * heads + h) * dim + lane + j * 32];
    for (int first = warp * keysPerWarp; first < count; first += warps * keysPerWarp) {
        int key[keysPerWarp];
        bool valid[keysPerWarp];
        float dot[keysPerWarp];
        #pragma unroll
        for (int i = 0; i < keysPerWarp; ++i) {
            key[i] = first + i < count ? KeyIndex(indices, query, first + i, count, past, window) : -1;
            valid[i] = key[i] >= 0 && key[i] < keys && (!causal || key[i] <= past + query);
            dot[i] = 0;
        }
        #pragma unroll
        for (int j = 0; j < MaxDim / 32; ++j) if (lane + j * 32 < dim) {
            #pragma unroll
            for (int i = 0; i < keysPerWarp; ++i) if (valid[i])
                dot[i] += queryValues[j] * (float)k[(size_t)key[i] * keyStride + kvHead * dim + lane + j * 32];
        }
        #pragma unroll
        for (int i = 0; i < keysPerWarp; ++i) {
            dot[i] = WarpSum(dot[i]);
            if (lane == 0 && first + i < count)
                scores[((size_t)query * heads + h) * count + first + i] =
                    valid[i] ? RoundBF16(RoundBF16(dot[i]) * rsqrtf((float)dim)) : -INFINITY;
        }
    }
}

// Reuse each K load across two Q heads in the same KV group. One key per
// warp keeps the register tile small; each dot retains the original lane
// accumulation, warp reduction and BF16 rounding order.
__global__ void AttentionScoresPrefillGrouped(const BF16 *q, const BF16 *k, const int *indices,
                                             float *scores, int heads, int kvHeads, int dim,
                                             int keyStride, int keys, int count, int past,
                                             int window, bool causal) {
    constexpr int headTile = 2, warps = 4, maxDim = 192;
    int query = blockIdx.y, kvHead = blockIdx.x, headsPerKv = heads / kvHeads;
    int firstHead = kvHead * headsPerKv + blockIdx.z * headTile;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float queryValues[headTile][maxDim / 32];
    #pragma unroll
    for (int h = 0; h < headTile; ++h) {
        #pragma unroll
        for (int j = 0; j < maxDim / 32; ++j)
            queryValues[h][j] = firstHead + h < (kvHead + 1) * headsPerKv && lane + j * 32 < dim
                ? (float)q[((size_t)query * heads + firstHead + h) * dim + lane + j * 32] : 0;
    }
    for (int slot = warp; slot < count; slot += warps) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
        float dot[headTile] = {};
        #pragma unroll
        for (int j = 0; j < maxDim / 32; ++j) if (lane + j * 32 < dim && valid) {
            float keyValue = (float)k[(size_t)key * keyStride + kvHead * dim + lane + j * 32];
            #pragma unroll
            for (int h = 0; h < headTile; ++h)
                dot[h] += queryValues[h][j] * keyValue;
        }
        #pragma unroll
        for (int h = 0; h < headTile; ++h) {
            dot[h] = WarpSum(dot[h]);
            if (lane == 0 && firstHead + h < (kvHead + 1) * headsPerKv)
                scores[((size_t)query * heads + firstHead + h) * count + slot] =
                    valid ? RoundBF16(RoundBF16(dot[h]) * rsqrtf((float)dim)) : -INFINITY;
        }
    }
}

// Four output columns per lane amortize index/probability reads and permit
// aligned 64-bit V loads. Each column keeps its original slot-ordered FP32
// accumulation. Four independent heads per CTA avoid one-warp block limits.
__global__ void AttentionValuesPrefill(const float *prob, const BF16 *v, const int *indices,
                                      BF16 *out, int heads, int kvHeads, int dim,
                                      int keys, int count, int past, int window, bool causal) {
    constexpr int columns = 4, headGroup = 4;
    int query = blockIdx.y, lane = threadIdx.x % 32;
    int h = blockIdx.x * headGroup + threadIdx.x / 32;
    if (h >= heads) return;
    int kvHead = h / (heads / kvHeads);
    const float *p = prob + ((size_t)query * heads + h) * count;
    for (int d = lane * columns; d < dim; d += 32 * columns) {
        float sum[columns] = {};
        for (int slot = 0; slot < count; ++slot) {
            int key = KeyIndex(indices, query, slot, count, past, window);
            if (key >= 0 && key < keys && (!causal || key <= past + query)) {
                uint2 bits = *reinterpret_cast<const uint2*>(v + ((size_t)key * kvHeads + kvHead) * dim + d);
                unsigned words[2] = {bits.x, bits.y};
                float weight = p[slot];
                #pragma unroll
                for (int j = 0; j < columns; ++j)
                    sum[j] += weight * __bfloat162float(__ushort_as_bfloat16(
                        (unsigned short)(words[j / 2] >> (16 * (j % 2)))));
            }
        }
        #pragma unroll
        for (int j = 0; j < columns; ++j)
            out[((size_t)query * heads + h) * dim + d + j] = __float2bfloat16(sum[j]);
    }
}

__global__ void AttentionSoftmax(float *scores, const float *sink, int heads, int count) {
    __shared__ float scratch[256];
    int row = blockIdx.x, t = threadIdx.x;
    float *values = scores + (size_t)row * count;
    float maximum = sink ? sink[row % heads] : -INFINITY;
    for (int i = t; i < count; i += 256) maximum = fmaxf(maximum, values[i]);
    scratch[t] = maximum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] = fmaxf(scratch[t], scratch[t + step]);
        __syncthreads();
    }
    maximum = scratch[0];
    float sum = 0;
    for (int i = t; i < count; i += 256) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    if (t == 0 && sink) sum += expf(sink[row % heads] - maximum);
    // All warps must consume scratch[0] (the maximum) before warp zero
    // overwrites it with the denominator. Racecheck catches this on the
    // unfused long/sparse path even when ordinary runs happen to agree.
    __syncthreads();
    scratch[t] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] += scratch[t + step];
        __syncthreads();
    }
    sum = scratch[0];
    for (int i = t; i < count; i += 256)
        values[i] = sum > 0 ? RoundBF16(values[i] / sum) : 0;
}

// Short full attention and the 128-token sliding window fit in shared memory.
// Keep eager's BF16 score/probability rounding, sink and reduction order while
// removing the score tensor round-trip and two launches per layer.
__global__ void AttentionShort(const BF16 *q, const BF16 *k, const BF16 *v,
                               const int *indices, const float *sink, BF16 *out,
                               int heads, int kvHeads, int dim, int valueDim,
                               int keyStride, int keys, int count, int past, int window, bool causal) {
    __shared__ float scores[256], scratch[256];
    int query = blockIdx.y, h = blockIdx.x, t = threadIdx.x;
    int lane = t % 32, warp = t / 32, kvHead = h / (heads / kvHeads);
    for (int slot = warp; slot < count; slot += 8) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        float dot = 0;
        bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
        if (valid) for (int d = lane; d < dim; d += 32)
            dot += (float)q[((size_t)query * heads + h) * dim + d] *
                   (float)k[(size_t)key * keyStride + kvHead * dim + d];
        dot = WarpSum(dot);
        if (lane == 0) scores[slot] = valid ? RoundBF16(RoundBF16(dot) * rsqrtf((float)dim)) : -INFINITY;
    }
    __syncthreads();
    float maximum = sink ? sink[h] : -INFINITY;
    if (t < count) maximum = fmaxf(maximum, scores[t]);
    scratch[t] = maximum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] = fmaxf(scratch[t], scratch[t + step]);
        __syncthreads();
    }
    maximum = scratch[0];
    float probability = t < count ? expf(scores[t] - maximum) : 0;
    float sum = probability;
    if (t == 0 && sink) sum += expf(sink[h] - maximum);
    __syncthreads();
    scratch[t] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] += scratch[t + step];
        __syncthreads();
    }
    if (t < count) scores[t] = scratch[0] > 0 ? RoundBF16(probability / scratch[0]) : 0;
    __syncthreads();
    for (int d = t; d < valueDim; d += 256) {
        float value = 0;
        for (int slot = 0; slot < count; ++slot) {
            int key = KeyIndex(indices, query, slot, count, past, window);
            if (key >= 0 && key < keys && (!causal || key <= past + query))
                value += scores[slot] * (float)v[((size_t)key * kvHeads + kvHead) * valueDim + d];
        }
        out[((size_t)query * heads + h) * valueDim + d] = __float2bfloat16(value);
    }
}

constexpr int kSwaWindow = 128;
constexpr int kSwaQkDim = 192;
constexpr int kSwaValueDim = 128;
constexpr int kSwaOutputTile = 32;
constexpr int kSwaThreads = 256;

// Single-query sliding window with head dimensions 192/128. Four
// output slices spread the work across SMs; cooperative V loads avoid the
// reference kernel's dependent global load for every output/slot pair.
// The softmax tree, BF16 rounding and slot-ordered FP32 FMAs are unchanged.
__global__ void AttentionSwaDecode(const BF16 *q, const BF16 *k, const BF16 *v,
        const float *sink, BF16 *out, int heads, int kvHeads, int keys) {
    __shared__ float scores[kSwaWindow], scratch[kSwaThreads], maximum, denominator;
    __shared__ BF16 values[kSwaWindow][kSwaOutputTile];
    int h = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    int kvHead = h / (heads / kvHeads), firstDim = blockIdx.z * kSwaOutputTile;
    for (int i = t; i < keys * kSwaOutputTile; i += kSwaThreads) {
        int row = i / kSwaOutputTile, col = i % kSwaOutputTile;
        values[row][col] = v[((size_t)row * kvHeads + kvHead) * kSwaValueDim + firstDim + col];
    }
    float query[kSwaQkDim / 32];
    #pragma unroll
    for (int i = 0; i < kSwaQkDim / 32; ++i) query[i] = (float)q[(size_t)h * kSwaQkDim + lane + i * 32];
    for (int slot = warp; slot < keys; slot += 8) {
        float dot = 0;
        #pragma unroll
        for (int i = 0; i < kSwaQkDim / 32; ++i)
            dot += query[i] * (float)k[((size_t)slot * kvHeads + kvHead) * kSwaQkDim + lane + i * 32];
        dot = WarpSum(dot);
        if (lane == 0) scores[slot] = RoundBF16(RoundBF16(dot) * rsqrtf((float)kSwaQkDim));
    }
    __syncthreads();
    float bias = sink ? sink[h] : -INFINITY;
    scratch[t] = t < keys ? fmaxf(bias, scores[t]) : bias;
    __syncthreads();
    // Fold the first three stages of the original 256-lane tree into
    // warp-local work; separate scalars keep scratch reuse race-free.
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]);
        float b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]);
        float d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float m = fmaxf(fmaxf(a, b), fmaxf(c, d));
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
        if (t == 0) maximum = m;
    }
    __syncthreads();
    float probability = t < keys ? expf(scores[t] - maximum) : 0;
    float sum = probability;
    if (t == 0 && sink) sum += expf(bias - maximum);
    scratch[t] = sum;
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128];
        float b = scratch[t + 64] + scratch[t + 192];
        float c = scratch[t + 32] + scratch[t + 160];
        float d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (t == 0) denominator = total;
    }
    __syncthreads();
    if (t < keys) scores[t] = denominator > 0 ? RoundBF16(probability / denominator) : 0;
    __syncthreads();
    if (t < kSwaOutputTile) {
        float result = 0;
        for (int first = 0; first < keys; first += 8) {
            float p[8], value[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (first + i < keys) {
                    p[i] = scores[first + i];
                    value[i] = (float)values[first + i][t];
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (first + i < keys) result += p[i] * value[i];
        }
        out[(size_t)h * kSwaValueDim + firstDim + t] = __float2bfloat16(result);
    }
}

__global__ void AttentionValues(const float *prob, const BF16 *v, const int *indices,
                                BF16 *out, int heads, int kvHeads, int dim,
                                int keys, int count, int past, int window, bool causal) {
    int query = blockIdx.y, h = blockIdx.x;
    int kvHead = h / (heads / kvHeads);
    const float *p = prob + ((size_t)query * heads + h) * count;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0;
        for (int slot = 0; slot < count; slot++) {
            int key = KeyIndex(indices, query, slot, count, past, window);
            if (key >= 0 && key < keys && (!causal || key <= past + query))
                sum += p[slot] * (float)v[((size_t)key * kvHeads + kvHead) * dim + d];
        }
        out[((size_t)query * heads + h) * dim + d] = __float2bfloat16(sum);
    }
}

constexpr int kDecodePVKeys = 2048;
constexpr int kDecodePVValueDim = 128;
constexpr int kDecodePVParts = 32;

// Decode has just one query. Split its 2048 selected keys across CTAs so
// every thread accumulates one value column, then combine FP32 partial sums.
// Inputs/probability rounding and BF16 output are unchanged; the FP32 sum order
// differs from the serial fallback. Prefill and other shapes keep their paths.
__global__ void AttentionValuesDecodePartial(const float *prob, const BF16 *v,
        const int *indices, float *partial, int heads, int kvHeads,
        int keys, int past, bool causal) {
    constexpr int count = kDecodePVKeys, dim = kDecodePVValueDim;
    constexpr int parts = kDecodePVParts, slots = count / parts;
    int h = blockIdx.x, part = blockIdx.y, d = threadIdx.x;
    int kvHead = h / (heads / kvHeads);
    float sum = 0;
    for (int base = part * slots; base < (part + 1) * slots; base += 8) {
        float probabilities[8], values[8];
        bool valid[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int slot = base + i, key = indices ? indices[slot] : slot;
            valid[i] = key >= 0 && key < keys && (!causal || key <= past);
            probabilities[i] = prob[h * count + slot];
            values[i] = valid[i] ? (float)v[((size_t)key * kvHeads + kvHead) * dim + d] : 0;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (valid[i]) sum = fmaf(probabilities[i], values[i], sum);
    }
    partial[(h * parts + part) * dim + d] = sum;
}

__global__ void AttentionValuesDecodeReduce(const float *partial, BF16 *out, int heads) {
    constexpr int dim = kDecodePVValueDim, parts = kDecodePVParts;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim) return;
    int h = i / dim, d = i % dim;
    float sum = 0;
    #pragma unroll
    for (int part = 0; part < parts; ++part)
        sum += partial[(h * parts + part) * dim + d];
    out[i] = __float2bfloat16(sum);
}

// Cooperatively stage a tile of V instead of issuing one dependent global
// load for each of 2048 slots. Output dimensions form separate CTAs; each
// output keeps exactly the original slot order and FP32 FMA accumulation.
__global__ void AttentionValuesTiled(const float *prob, const BF16 *v,
        const int *indices, BF16 *out, int heads, int kvHeads, int dim,
        int keys, int count, int past, int window, bool causal) {
    __shared__ BF16 values[64][32];
    __shared__ float probabilities[64];
    __shared__ int selected[64];
    int h = blockIdx.x, query = blockIdx.y, t = threadIdx.x;
    int d = blockIdx.z * 32 + t, kvHead = h / (heads / kvHeads);
    float sum = 0;
    const float *p = prob + ((size_t)query * heads + h) * count;
    for (int first = 0; first < count; first += 64) {
        if (t < 64) {
            int slot = first + t;
            int key = slot < count ? KeyIndex(indices, query, slot, count, past, window) : -1;
            bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
            selected[t] = valid ? key : -1;
            probabilities[t] = slot < count ? p[slot] : 0;
        }
        __syncthreads();
        for (int i = t; i < 64 * 32; i += 256) {
            int row = i / 32, col = blockIdx.z * 32 + i % 32;
            int key = selected[row];
            values[row][i % 32] = key >= 0 && col < dim
                ? v[((size_t)key * kvHeads + kvHead) * dim + col] : __float2bfloat16(0);
        }
        __syncthreads();
        if (t < 32 && d < dim) {
            for (int row = 0; row < min(64, count - first); ++row)
                if (selected[row] >= 0) sum += probabilities[row] * (float)values[row][t];
        }
        __syncthreads();
    }
    if (t < 32 && d < dim) out[((size_t)query * heads + h) * dim + d] = __float2bfloat16(sum);
}
}

void FastllmCudaNaiveTrimCache(fastllm::Data &key, fastllm::Data &value, int keep) {
    using namespace fastllm;
    AssertInFastLLM(keep >= 0 && key.dims.size() == 3 && value.dims.size() == 3 &&
        key.dims[0] == 1 && value.dims[0] == 1 && key.dims[1] == value.dims[1] &&
        key.dataType == BFLOAT16 && value.dataType == BFLOAT16 &&
        key.dataDevice == DataDevice::CUDA && value.dataDevice == DataDevice::CUDA &&
        key.dataDeviceIds == value.dataDeviceIds && key.dims[2] > 0 && value.dims[2] > 0,
        "Invalid Naive-N0.5 sliding cache layout.");
    if (key.dims[1] <= keep) return;
    int drop = key.dims[1] - keep;
    if (key.dims[2] % 8 == 0 && value.dims[2] % 8 == 0) {
        LaunchTrimCachePair((uint4 *)key.cudaData, (uint4 *)value.cudaData,
            key.dims[2] / 8, value.dims[2] / 8, drop, keep);
    } else {
        // The vector path requires 16-byte row alignment. Other BF16 layouts
        // use the same overlap-safe column ownership at scalar granularity.
        LaunchTrimCachePair((uint16_t *)key.cudaData, (uint16_t *)value.cudaData,
            key.dims[2], value.dims[2], drop, keep);
    }
    CheckLaunch();
    key.Resize({1, keep, key.dims[2]});
    value.Resize({1, keep, value.dims[2]});
}

void FastllmCudaNaiveTopK(const fastllm::Data &scores, int queryStart, int topK,
                         fastllm::Data &indices) {
    using namespace fastllm;
    AssertInFastLLM(scores.dataDevice == DataDevice::CUDA && scores.dataType == FLOAT32 &&
        scores.dims.size() == 2 && scores.dims[0] > 0 && queryStart >= 0 &&
        (int64_t)queryStart + scores.dims[0] <= scores.dims[1] && topK > 0,
        "Invalid Naive-N0.5 TopK.");
    int queries = scores.dims[0], keys = scores.dims[1];
    if (queries > 1) {
        int64_t total = (int64_t)queries * keys;
        AssertInFastLLM(total <= INT_MAX && (int64_t)queries * topK <= INT_MAX,
            "Naive-N0.5 batched TopK is too large.");
        // Full sorting remains cheaper for short rows or large requested K.
        bool compact = (int64_t)keys >= (int64_t)topK * 4;
        int sortStride = compact ? topK * 2 : keys;
        Data encoded, sorted, offsets, workspace;
        Output(encoded, INT32, {queries, sortStride, 2});
        Output(sorted, INT32, {queries, sortStride, 2});
        Output(offsets, INT32, {2, queries});
        Output(indices, INT32, {queries, topK});
        auto *input = (unsigned long long *)encoded.cudaData;
        auto *output = (unsigned long long *)sorted.cudaData;
        auto *begin = (int *)offsets.cudaData;
        size_t bytes = 0;
        int items = compact ? queries * sortStride : (queries - 1) * keys + queryStart + queries;
        auto status = cub::DeviceSegmentedRadixSort::SortKeysDescending(
            nullptr, bytes, input, output, items, queries, begin, begin + queries,
            0, 64, cudaStreamPerThread);
        AssertInFastLLM(status == cudaSuccess && (bytes + 3) / 4 <= INT_MAX,
            "Naive-N0.5 batched TopK workspace query failed.");
        Output(workspace, INT32, {(int)((bytes + 3) / 4)});
        if (compact) {
            SelectTopKCompact<<<queries, 256>>>((const float *)scores.cudaData,
                input, begin, queries, keys, queryStart, topK, sortStride);
        } else {
            EncodeTopK<<<dim3((keys + 255) / 256, queries), 256>>>(
                (const float *)scores.cudaData, input, begin, queries, keys, queryStart);
        }
        status = cub::DeviceSegmentedRadixSort::SortKeysDescending(
            workspace.cudaData, bytes, input, output, items, queries, begin, begin + queries,
            0, 64, cudaStreamPerThread);
        AssertInFastLLM(status == cudaSuccess, "Naive-N0.5 batched GPU TopK failed.");
        DecodeTopK<<<dim3((topK + 255) / 256, queries), 256>>>(
            output, (int *)indices.cudaData, sortStride, queryStart, topK);
        CheckLaunch();
        return;
    }
    int count = queryStart + 1;
    Data encoded, sorted, workspace;
    Output(encoded, INT32, {count, 2});
    Output(sorted, INT32, {count, 2});
    Output(indices, INT32, {1, topK});
    auto *input = (unsigned *)encoded.cudaData;
    auto *output = (unsigned *)sorted.cudaData;
    auto *inputPositions = (int *)encoded.cudaData + count;
    auto *outputPositions = (int *)sorted.cudaData + count;
    size_t bytes = 0;
    // Stability retains ascending position order for equal score bits. This
    // gives the same total order as the batched 64-bit key with half the bits.
    // CCCL's driver launcher needs the per-thread stream explicitly.
    auto status = cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
        input, output, inputPositions, outputPositions, count, 0, 32, cudaStreamPerThread);
    AssertInFastLLM(status == cudaSuccess && (bytes + 3) / 4 <= INT_MAX,
        "Naive-N0.5 TopK workspace query failed.");
    Output(workspace, INT32, {(int)((bytes + 3) / 4)});
    EncodeTopKPairs<<<(count + 255) / 256, 256>>>(
        (const float *)scores.cudaData, input, inputPositions, count);
    status = cub::DeviceRadixSort::SortPairsDescending(workspace.cudaData, bytes,
        input, output, inputPositions, outputPositions, count, 0, 32, cudaStreamPerThread);
    AssertInFastLLM(status == cudaSuccess, "Naive-N0.5 GPU TopK failed.");
    DecodeTopKPairs<<<(topK + 255) / 256, 256>>>(outputPositions, (int *)indices.cudaData, count, topK);
    CheckLaunch();
}

void FastllmCudaNaiveRope(fastllm::Data &input, const fastllm::Data &positions,
                         int heads, int dim, int rotaryDim, float theta) {
    fastllm::AssertInFastLLM(input.dataType == fastllm::DataType::BFLOAT16 &&
                            rotaryDim > 0 && rotaryDim % 2 == 0 && rotaryDim <= dim,
                            "Invalid Naive-N0.5 RoPE input.");
    Rope<<<input.Count(0) / dim, 128>>>((BF16 *)input.cudaData,
        (const float *)positions.cudaData, heads, dim, rotaryDim, theta);
    CheckLaunch();
}

void FastllmCudaNaiveIndexer(const fastllm::Data &query, const fastllm::Data &weights,
                            const fastllm::Data &packedKeys, int heads, int dim,
                            int queryStart, int topK, bool fp8, fastllm::Data &indices) {
    using namespace fastllm;
    // Indexer quantization is defined on 128-element blocks. The score kernel
    // has a general head-count path, but a different block width is unsupported.
    AssertInFastLLM(dim == 128 && heads > 0 && query.dims.size() == 3 &&
        query.dims[0] == 1 && query.dims[1] > 0 && query.dims[2] == (int64_t)heads * dim &&
        packedKeys.dims.size() == 3 && packedKeys.dims[0] == 1 && packedKeys.dims[2] >= dim &&
        queryStart >= 0 && (int64_t)queryStart + query.dims[1] <= packedKeys.dims[1] && topK > 0 &&
        query.dataType == BFLOAT16 && packedKeys.dataType == BFLOAT16 && weights.dataType == BFLOAT16 &&
        query.dataDevice == DataDevice::CUDA && packedKeys.dataDevice == DataDevice::CUDA &&
        weights.dataDevice == DataDevice::CUDA && query.dataDeviceIds == packedKeys.dataDeviceIds &&
        query.dataDeviceIds == weights.dataDeviceIds && query.cudaData && packedKeys.cudaData &&
        weights.cudaData && weights.Count(0) == (uint64_t)query.dims[1] * heads,
        "Invalid Naive-N0.5 Indexer layout (requires 128-element BF16 blocks).");
    int queries = query.dims[1], keys = packedKeys.dims[1];
    int stride = packedKeys.dims[2];
    Data q, k, scores;
    Output(scores, DataType::FLOAT32, {queries, keys});
#ifdef FASTLLM_NAIVE_DSA_MMA
    // Keep the E4M3 values exact in BF16 and apply their FP32 scales after MMA.
    // Small prefill blocks and decode retain the original FP32 reduction path.
    if (fp8 && heads == 16 && queries >= 32 &&
        (int64_t)queries * keys >= 1024 * 1024 &&
        FastllmCudaFlashInferDataTypeSupported(DataType::BFLOAT16)) {
        Data qScale, kScale;
        Output(q, DataType::BFLOAT16, {queries, heads, dim});
        Output(k, DataType::BFLOAT16, {keys, dim});
        Output(qScale, DataType::FLOAT32, {queries, heads});
        Output(kScale, DataType::FLOAT32, {keys});
        naive_dsa_mma::QuantizeIndexer<<<queries * heads, 128>>>(
            (const BF16 *)query.cudaData, (BF16 *)q.cudaData,
            (float *)qScale.cudaData, dim, 0);
        naive_dsa_mma::QuantizeIndexer<<<keys, 128>>>(
            (const BF16 *)packedKeys.cudaData, (BF16 *)k.cudaData,
            (float *)kScale.cudaData, stride, stride - dim);
        naive_dsa_mma::IndexerScores<<<dim3((keys + 63) / 64, (queries + 63) / 64), 256>>>(
            (const BF16 *)q.cudaData, (const BF16 *)k.cudaData,
            (const float *)qScale.cudaData, (const float *)kScale.cudaData,
            (const BF16 *)weights.cudaData, (float *)scores.cudaData, queries, keys, queryStart);
    } else
#endif
    {
        Output(q, DataType::FLOAT32, {queries, heads, dim});
        RoundIndexer<<<queries * heads, 128>>>((const BF16 *)query.cudaData,
            (float *)q.cudaData, dim, 0, fp8);
        if (queries == 1 && heads == 16 && fp8) {
            IndexScoresDecode<<<(keys + 7) / 8, 256>>>((const float *)q.cudaData,
                (const BF16 *)packedKeys.cudaData, (const BF16 *)weights.cudaData,
                (float *)scores.cudaData, stride, keys, queryStart);
        } else {
            Output(k, DataType::FLOAT32, {keys, dim});
            RoundIndexer<<<keys, 128>>>((const BF16 *)packedKeys.cudaData,
                (float *)k.cudaData, stride, stride - dim, fp8);
            if (queries > 1 && heads == 16) {
                // A larger tile amortizes operand loads once there are enough tiles
                // to fill the GPU. Keep the smaller tile for short/underfilled work.
                if ((int64_t)queries * keys >= 128 * 1024) {
                    constexpr int keysPerBlock = kIndexerKeysPerWarp * kIndexerWarps;
                    IndexScoresPrefillTiled<<<dim3((keys + keysPerBlock - 1) / keysPerBlock,
                        (queries + kIndexerQueryTile - 1) / kIndexerQueryTile), kIndexerWarps * 32>>>(
                        (const float *)q.cudaData, (const float *)k.cudaData,
                        (const BF16 *)weights.cudaData, (float *)scores.cudaData, queries, keys, queryStart);
                } else {
                    IndexScoresPrefill<<<dim3((keys + 7) / 8, queries), 256>>>((const float *)q.cudaData,
                        (const float *)k.cudaData, (const BF16 *)weights.cudaData,
                        (float *)scores.cudaData, keys, queryStart);
                }
            } else {
                IndexScores<<<dim3((keys + 7) / 8, queries), 256>>>((const float *)q.cudaData,
                    (const float *)k.cudaData, (const BF16 *)weights.cudaData,
                    (float *)scores.cudaData, heads, keys, queryStart);
            }
        }
    }
    CheckLaunch();
    // Stable GPU selection for decode and every row of a prefill chunk.
    FastllmCudaNaiveTopK(scores, queryStart, topK, indices);
}

void FastllmCudaNaiveAttention(const fastllm::Data &query, const fastllm::Data &key,
                              const fastllm::Data &value, const fastllm::Data &indices,
                              const fastllm::Data &sink, int heads, int kvHeads,
                              int dim, int valueDim, int pastLength, int window,
                              fastllm::Data &output, bool causal) {
    using namespace fastllm;
    int queries = query.dims[1], keys = key.dims[1];
    int count = indices.dims.empty() ? (window ? std::min(window + (causal ? 0 : queries - 1), keys) : keys) : indices.dims[1];
    const int *selected = indices.dims.empty() ? nullptr : (const int *)indices.cudaData;
    Output(output, DataType::BFLOAT16, {1, queries, heads * valueDim});
#ifdef FASTLLM_NAIVE_SWA_FLASHINFER
    // Match FlashInfer's bottom-right causal alignment. Full-window decode
    // benefits with at least 64 Q heads and GQA 8; shorter windows stay on
    // the dedicated decode kernel. BF16 MMA requires aligned input rows.
    if (((queries >= 32 && (size_t)queries * heads >= 2048) ||
         (queries == 1 && keys == kSwaWindow && heads >= 64 && heads / kvHeads == 8)) &&
        window == kSwaWindow && causal && !selected && pastLength == keys - queries &&
        dim == kSwaQkDim && valueDim == kSwaValueDim && key.dims[2] % 8 == 0 &&
        (size_t)query.cudaData % 16 == 0 && (size_t)key.cudaData % 16 == 0 &&
        (size_t)value.cudaData % 16 == 0 && FastllmCudaFlashInferDataTypeSupported(DataType::BFLOAT16)) {
        auto status = naive_swa_flashinfer::Run((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, queries, keys, heads, kvHeads, key.dims[2]);
        AssertInFastLLM(status == cudaSuccess,
            std::string("Naive SWA FlashInfer: ") + cudaGetErrorString(status));
        CheckLaunch();
        return;
    }
#endif
    if (queries == 1 && window == kSwaWindow && causal && !selected &&
        keys > 0 && keys <= kSwaWindow && pastLength == keys - 1 &&
        dim == kSwaQkDim && valueDim == kSwaValueDim && key.dims[2] == kvHeads * dim) {
        AttentionSwaDecode<<<dim3(heads, 1, kSwaValueDim / kSwaOutputTile), kSwaThreads>>>(
            (const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, heads, kvHeads, keys);
        CheckLaunch();
        return;
    }
    if (count <= 256) {
        AttentionShort<<<dim3(heads, queries), 256>>>((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData, selected,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData, (BF16 *)output.cudaData,
            heads, kvHeads, dim, valueDim, key.dims[2], keys, count, pastLength, window, causal);
        CheckLaunch();
        return;
    }
    Data scores;
    Output(scores, DataType::FLOAT32, {queries, heads, count});
#ifdef FASTLLM_NAIVE_DSA_MMA
    // Sixteen Q heads share each gathered K/V tile. Preserve materialized
    // BF16-rounded logits/probabilities; MMA changes FP32 reduction order.
    const bool useMma = queries >= 32 && (size_t)queries * heads >= 2048 && window == 0 &&
        heads % kvHeads == 0 && (heads / kvHeads) % naive_dsa_mma::kHeads == 0 &&
        dim == naive_dsa_mma::kQkDim && valueDim == naive_dsa_mma::kValueDim &&
        key.dims[2] % 8 == 0 && (size_t)query.cudaData % 16 == 0 &&
        (size_t)key.cudaData % 16 == 0 && (size_t)value.cudaData % 16 == 0 &&
        FastllmCudaFlashInferDataTypeSupported(DataType::BFLOAT16);
    if (useMma) {
        naive_dsa_mma::Scores<<<dim3(queries, heads / naive_dsa_mma::kHeads,
            (count + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys), naive_dsa_mma::kThreads>>>(
            (const BF16 *)query.cudaData, (const BF16 *)key.cudaData, selected, (float *)scores.cudaData,
            heads, kvHeads, key.dims[2], keys, count, pastLength, causal);
    } else
#endif
    // Small query blocks need the original per-head CTA count for occupancy.
    if (queries >= 32 && dim <= 192 && heads / kvHeads >= 2) {
        AttentionScoresPrefillGrouped<<<dim3(kvHeads, queries, (heads / kvHeads + 1) / 2), 128>>>(
            (const BF16 *)query.cudaData, (const BF16 *)key.cudaData, selected, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    } else if (queries > 1 && dim <= 256) {
        auto kernel = dim <= 192 ? AttentionScoresPrefill<192> : AttentionScoresPrefill<256>;
        kernel<<<dim3(heads, queries), 128>>>((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, selected, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    } else {
        AttentionScores<<<dim3(heads, queries, queries == 1 ? (count + 63) / 64 : 1), 256>>>((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, selected, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    }
    AttentionSoftmax<<<queries * heads, 256>>>((float *)scores.cudaData,
        sink.dims.empty() ? nullptr : (const float *)sink.cudaData, heads, count);
#ifdef FASTLLM_NAIVE_DSA_MMA
    if (useMma) {
        naive_dsa_mma::Values<<<dim3(queries, heads / naive_dsa_mma::kHeads), naive_dsa_mma::kThreads>>>(
            (const float *)scores.cudaData, (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
            heads, kvHeads, keys, count, pastLength, causal);
    } else
#endif
    if (queries == 1 && window == 0 && count == kDecodePVKeys &&
        valueDim == kDecodePVValueDim && heads >= 32) {
        Data partial;
        Output(partial, FLOAT32, {heads, kDecodePVParts, kDecodePVValueDim});
        AttentionValuesDecodePartial<<<dim3(heads, kDecodePVParts), kDecodePVValueDim>>>((const float *)scores.cudaData,
            (const BF16 *)value.cudaData, selected, (float *)partial.cudaData,
            heads, kvHeads, keys, pastLength, causal);
        AttentionValuesDecodeReduce<<<(heads * kDecodePVValueDim + 255) / 256, 256>>>(
            (const float *)partial.cudaData, (BF16 *)output.cudaData, heads);
    } else if (queries == 1) {
        AttentionValuesTiled<<<dim3(heads, queries, (valueDim + 31) / 32), 256>>>((const float *)scores.cudaData,
            (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
            heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    } else if (valueDim % 4 == 0 && (size_t)value.cudaData % alignof(uint2) == 0) {
        AttentionValuesPrefill<<<dim3((heads + 3) / 4, queries), 128>>>((const float *)scores.cudaData,
            (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
            heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    } else {
        AttentionValues<<<dim3(heads, queries), 256>>>((const float *)scores.cudaData,
            (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
            heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    }
    CheckLaunch();
}
