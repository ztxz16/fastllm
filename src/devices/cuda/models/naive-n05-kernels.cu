#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
#include "naive-n05-topk.cuh"
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
#endif

#if !defined(FASTLLM_CUDA_LEGACY_ONLY)
// Keep host launch stubs and device definitions in every architecture pass.
// MMA bodies and runtime dispatch already guard the SM80-only instructions.
#define FASTLLM_NAIVE_DSA_MMA
#include "naive-n05-dsa-mma.cuh"
#endif

namespace {
using BF16 = __nv_bfloat16;
// Decode graphs keep launch geometry fixed while the live KV length changes.
// Ordinary launches retain the int specialization, with no device-side branch.
struct DecodeKeys {
    const int *length;
    int window;
    __device__ operator int() const { return window ? min(*length, window) : *length; }
};
__device__ int ShortCount(int, int count) { return count; }
__device__ int ShortCount(DecodeKeys keys, int count) { return min((int)keys, count); }
__device__ int VerifyCount(int first, int row) { return first + row; }
__device__ int VerifyCount(DecodeKeys keys, int row) {
    keys.length += row;
    return keys;
}

// Draft graphs append a noncausal block to a bounded sliding prefix.
struct DraftLength {
    const int *length;
    int window, extra;
    __device__ operator int() const { return min(*length - 1, window - 1) + extra; }
};
__device__ int ShortCount(DraftLength keys, int count) { return min((int)keys, count); }

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
    if (status != cudaSuccess && FastllmCudaGraphIsCapturingFast()) {
        FastllmCudaSetThreadError();
        return;
    }
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
template <typename T, bool Dynamic = false>
__global__ void TrimCachePairTiled(T *key, T *value, int keyColumns,
                                 int valueColumns, int drop, int keep, const int *decodeKeys = nullptr) {
    if constexpr (Dynamic) { if (*decodeKeys <= keep) return; }
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

using naive_topk::OrderedScoreBits;

// Emit ordered scores during decode scoring. Only the CUB fallback needs
// an explicit position array; cooperative selection derives positions itself.
struct IndexerTopKOutput {
    unsigned *scoreBits;
    int *positions;
    __device__ void Store(int index, float score) const {
        scoreBits[index] = OrderedScoreBits(score);
        if (positions) positions[index] = index;
    }
};

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

// Cache device capability once per worker/device, before repeated Graph captures.
// The selector specializes K=2048; other shapes retain the CUB fallback.
int CooperativeTopKBlocks(int count, int topK) {
    using namespace naive_topk;
    if (topK != kTopK || count < kMinKeys || count > kMaxKeys) return 0;
    int device = FastllmCudaGetDevice();
    static thread_local std::vector<int> blockCounts;
    if (device >= (int)blockCounts.size()) blockCounts.resize(device + 1, -1);
    if (blockCounts[device] < 0) {
        cudaDeviceProp prop;
        auto status = cudaGetDeviceProperties(&prop, device);
        fastllm::AssertInFastLLM(status == cudaSuccess, "Naive TopK device query failed.");
        blockCounts[device] = 0;
        if (prop.cooperativeLaunch) {
            int floatBlocks, bitBlocks;
            status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&floatBlocks,
                Select<float>, kThreads, 0);
            fastllm::AssertInFastLLM(status == cudaSuccess, "Naive TopK occupancy query failed.");
            status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bitBlocks,
                Select<unsigned>, kThreads, 0);
            fastllm::AssertInFastLLM(status == cudaSuccess, "Naive TopK occupancy query failed.");
            blockCounts[device] = std::min(kMaxBlocks,
                std::min(floatBlocks, bitBlocks) * prop.multiProcessorCount);
        }
    }
    return blockCounts[device];
}

template <typename Score>
void CooperativeTopK(const Score *scores, int count, int blocks, fastllm::Data &indices) {
    using namespace fastllm;
    using namespace naive_topk;
    // Each invocation owns its scratch through the existing graph-aware allocator.
    Data workspace;
    Output(workspace, INT32, {(int)(sizeof(Workspace) / sizeof(int))});
    Output(indices, INT32, {1, kTopK});
    auto *scratch = (Workspace *)workspace.cudaData;
    auto *candidates = &scratch->candidates;
    auto *histograms = scratch->partials;
    auto *ties = scratch->ties;
    auto *state = &scratch->state;
    void *args[] = {&scores, &count, &histograms, &ties, &state, &candidates};
    auto status = cudaLaunchCooperativeKernel((void *)naive_topk::Select<Score>,
        blocks, kThreads, args, 0, cudaStreamPerThread);
    AssertInFastLLM(status == cudaSuccess, "Naive cooperative TopK selection failed.");
    naive_topk::Sort<<<1, kThreads, 0, cudaStreamPerThread>>>(candidates, state, (int *)indices.cudaData);
    CheckLaunch();
}

// Owning outputs hold all sorted positions and expose only the first topK.
// A borrowed row can hold only topK indices: sort into separate storage when
// count exceeds that view, then copy its selected prefix without overrunning it.
void SortTopKPairs(const unsigned *input, const int *positions, int count,
                   int topK, fastllm::Data &indices) {
    using namespace fastllm;
    Data sorted, workspace, sortedPositions;
    const bool copyPrefix = indices.isFake && count > topK;
    Data &positionsOutput = copyPrefix ? sortedPositions : indices;
    Output(sorted, INT32, {count});
    Output(positionsOutput, INT32, {1, std::max(count, topK)});
    auto *output = (unsigned *)sorted.cudaData;
    auto *outputPositions = (int *)positionsOutput.cudaData;
    size_t bytes = 0;
    // Stable sorting resolves equal score bits by ascending input position.
    auto status = cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
        input, output, positions, outputPositions, count, 0, 32, cudaStreamPerThread);
    AssertInFastLLM(status == cudaSuccess && (bytes + 3) / 4 <= INT_MAX,
        "Naive-N0.5 TopK workspace query failed.");
    Output(workspace, INT32, {(int)((bytes + 3) / 4)});
    status = cub::DeviceRadixSort::SortPairsDescending(workspace.cudaData, bytes,
        input, output, positions, outputPositions, count, 0, 32, cudaStreamPerThread);
    AssertInFastLLM(status == cudaSuccess, "Naive-N0.5 GPU TopK failed.");
    if (copyPrefix || count < topK)
        DecodeTopKPairs<<<(topK + 255) / 256, 256>>>(
            outputPositions, (int *)indices.cudaData, count, topK);
    indices.Resize({1, topK});
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

__device__ void RopeHead(BF16 *x, float position, int rotaryDim, float theta) {
    for (int d = threadIdx.x; d < rotaryDim / 2; d += blockDim.x) {
        float angle = position * powf(theta, -2.0f * d / rotaryDim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)x[d], b = (float)x[d + rotaryDim / 2];
        // Match eager GPT-NeoX RoPE, including each BF16 multiplication.
        x[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        x[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
}

__global__ void Rope(BF16 *data, const float *positions, int heads, int dim,
                     int rotaryDim, float theta) {
    int row = blockIdx.x;
    RopeHead(data + (size_t)row * dim, positions[row / heads], rotaryDim, theta);
}

// Q/K rotate in place; V keeps the eager Mul's BF16 coefficient rounding.
// Each CTA owns one head, so all three outputs are independent.
__global__ void RopeQKScaleV(BF16 *q, BF16 *k, BF16 *v, const float *positions,
                            int heads, int kvHeads, int dim, int valueDim,
                            int rotaryDim, float theta, BF16 valueScale) {
    int totalHeads = heads + 2 * kvHeads;
    int token = blockIdx.x / totalHeads, head = blockIdx.x % totalHeads;
    if (head < heads + kvHeads) {
        BF16 *x = head < heads ? q + ((size_t)token * heads + head) * dim
            : k + ((size_t)token * kvHeads + head - heads) * dim;
        RopeHead(x, positions[token], rotaryDim, theta);
    } else {
        BF16 *x = v + ((size_t)token * kvHeads + head - heads - kvHeads) * valueDim;
        for (int d = threadIdx.x; d < valueDim; d += blockDim.x)
            x[d] = __float2bfloat16_rn((float)x[d] * (float)valueScale);
    }
}

__global__ void RoundIndexer(const BF16 *input, float *output, int stride,
                             int offset, bool fp8) {
    __shared__ float maximum[128];
    int d = threadIdx.x, row = blockIdx.x;
    float x = (float)input[(size_t)row * stride + offset + d];
    if (!fp8) {
        output[(size_t)row * 128 + d] = x;
        return;
    }
    maximum[d] = fabsf(x);
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (d < step) maximum[d] = fmaxf(maximum[d], maximum[d + step]);
        __syncthreads();
    }
    float scale = fmaxf(maximum[0], 1e-4f) / 448.0f;
    x = (float)__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / scale))) * scale;
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

// Decode consumes each packed BF16 K row once. Keep its FP32 values in
// registers across the 16 heads, avoiding a full temporary K.
template <typename ScoreOutput, typename Length = int, bool Verify = false>
__global__ void IndexScoresDecode(const float *q, const BF16 *packedKeys,
        const BF16 *weights, ScoreOutput scores, int stride, Length liveKeys, int queryStart) {
    if constexpr (Verify) {
        const int row = blockIdx.y;
        q += (size_t)row * 16 * 128;
        weights += row * 16;
        liveKeys.length += row;
        scores.scoreBits += (size_t)row * (queryStart + 1);
    }
    int keys = liveKeys;
    int key = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (key >= keys) return;
    float k[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        k[i] = (float)packedKeys[(size_t)key * stride + stride - 128 + lane + i * 32];
    float score = 0;
    for (int h = 0; h < 16; ++h) {
        float dot = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            dot += q[h * 128 + lane + i * 32] * k[i];
        dot = WarpSum(dot);
        score += fmaxf(dot, 0.0f) * (float)weights[h];
    }
    if (lane == 0) scores.Store(key, key <= queryStart ? score : -INFINITY);
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

template <typename Length = int, typename Count = int>
__global__ void AttentionScores(const BF16 *q, const BF16 *k, const int *indices,
                                float *scores, int heads, int kvHeads, int dim,
                                int keyStride, Length liveKeys, Count liveCount, int past, int window, bool causal) {
    int count = liveCount;
    int keys = liveKeys;
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

// Shared by the single-query score, softmax and split-PV paths.
constexpr int kDecodePVKeys = 2048;
constexpr int kDecodePVValueDim = 128;
constexpr int kDecodePVParts = 32;
constexpr int kDecodeQkDim = 192;
constexpr int kDecodeSharedHeads = 16;
constexpr int kDecodeSharedThreads = 256;
constexpr int kDecodeQkKeyTile = 8;
constexpr int kDecodeValueTile = 64;

// Preserve each head's arithmetic order while sharing gathered K loads.
template<int HeadTile, int MaxDim, int KeyTile = 32, typename Length = int, bool Verify = false>
__global__ void AttentionScoresDecodeGrouped(const BF16 *q, const BF16 *k,
        const int *indices, float *scores, int heads, int kvHeads, int dim,
        int keyStride, Length liveKeys, int count, int past, bool causal) {
    if constexpr (Verify) {
        const int row = blockIdx.z;
        liveKeys.length += row;
        q += (size_t)row * heads * dim;
        indices += (size_t)row * count;
        scores += (size_t)row * heads * count;
        past = (int)liveKeys - 1;
    }
    int keys = liveKeys;
    constexpr int warps = 4;
    int firstHead = blockIdx.x * HeadTile;
    int kvHead = firstHead / (heads / kvHeads);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float query[HeadTile][MaxDim / 32];
    #pragma unroll
    for (int h = 0; h < HeadTile; ++h) {
        #pragma unroll
        for (int j = 0; j < MaxDim / 32; ++j)
            query[h][j] = lane + j * 32 < dim
                ? (float)q[(size_t)(firstHead + h) * dim + lane + j * 32] : 0;
    }
    int first = blockIdx.y * KeyTile, end = min(count, first + KeyTile);
    for (int slot = first + warp; slot < end; slot += warps) {
        int key = indices ? indices[slot] : slot;
        bool valid = key >= 0 && key < keys && (!causal || key <= past);
        float dot[HeadTile] = {};
        #pragma unroll
        for (int j = 0; j < MaxDim / 32; ++j) {
            if (valid && lane + j * 32 < dim) {
                float value = (float)k[(size_t)key * keyStride + kvHead * dim + lane + j * 32];
                #pragma unroll
                for (int h = 0; h < HeadTile; ++h)
                    dot[h] += query[h][j] * value;
            }
        }
        #pragma unroll
        for (int h = 0; h < HeadTile; ++h) {
            dot[h] = WarpSum(dot[h]);
            if (lane == 0)
                scores[(size_t)(firstHead + h) * count + slot] = valid
                    ? RoundBF16(RoundBF16(dot[h]) * rsqrtf((float)dim)) : -INFINITY;
        }
    }
}

// Stage eight selected K rows once for sixteen Q heads. The aligned load
// path keeps each head's original lane sum, warp tree and BF16 rounding.
template <typename Length = int>
__global__ void AttentionScoresDecodeShared(const BF16 *q, const BF16 *k,
        const int *indices, float *scores, int heads, int kvHeads, int dim,
        int keyStride, Length liveKeys, int count, int past, bool causal) {
    int keys = liveKeys;
    constexpr int HeadTile = kDecodeSharedHeads, KeyTile = kDecodeQkKeyTile, HeadsPerWarp = 2;
    constexpr int MaxDim = kDecodeQkDim, threads = kDecodeSharedThreads;
    __shared__ __align__(16) BF16 values[KeyTile][MaxDim];
    __shared__ int selected[KeyTile];
    int t = threadIdx.x, lane = t % 32, warp = t / 32, firstHead = blockIdx.x * HeadTile;
    int kvHead = firstHead / (heads / kvHeads), first = blockIdx.y * KeyTile;
    if (t < KeyTile) {
        int slot = first + t, key = slot < count ? (indices ? indices[slot] : slot) : -1;
        selected[t] = key >= 0 && key < keys && (!causal || key <= past) ? key : -1;
    }
    __syncthreads();
    // Dispatch guarantees 16-byte alignment for every selected K row.
    #pragma unroll
    for (int i = t; i < KeyTile * (MaxDim / 8); i += threads) {
        int row = i / (MaxDim / 8), d = i % (MaxDim / 8) * 8, key = selected[row];
        *reinterpret_cast<uint4 *>(&values[row][d]) = key >= 0
            ? *reinterpret_cast<const uint4 *>(k + (size_t)key * keyStride + kvHead * dim + d)
            : make_uint4(0, 0, 0, 0);
    }
    int head = firstHead + warp * HeadsPerWarp;
    float query[HeadsPerWarp][MaxDim / 32];
    #pragma unroll
    for (int h = 0; h < HeadsPerWarp; ++h) {
        #pragma unroll
        for (int j = 0; j < MaxDim / 32; ++j)
            query[h][j] = lane + j * 32 < dim ? (float)q[(head + h) * dim + lane + j * 32] : 0;
    }
    __syncthreads();
    for (int slot = 0; slot < KeyTile && first + slot < count; ++slot) {
        float dot[HeadsPerWarp] = {};
        bool valid = selected[slot] >= 0;
        #pragma unroll
        for (int j = 0; j < MaxDim / 32; ++j) {
            if (lane + j * 32 < dim && valid) {
                float value = (float)values[slot][lane + j * 32];
                #pragma unroll
                for (int h = 0; h < HeadsPerWarp; ++h)
                    dot[h] += query[h][j] * value;
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            #pragma unroll
            for (int h = 0; h < HeadsPerWarp; ++h)
                dot[h] += __shfl_down_sync(0xffffffff, dot[h], offset);
        }
        if (lane == 0) {
            #pragma unroll
            for (int h = 0; h < HeadsPerWarp; ++h)
                scores[(head + h) * count + first + slot] = valid
                    ? RoundBF16(RoundBF16(dot[h]) * rsqrtf((float)dim)) : -INFINITY;
        }
    }
}

// Keep each warp's original lane-strided dot product and reduction order,
// but retain Q in registers and interleave four independent selected keys.
// The bounded query register tile leaves larger head dimensions on the
// general kernel; this path is only used for multiple-query prefill.
template <int MaxDim, typename Length = int, typename Past = int>
__global__ void AttentionScoresPrefill(const BF16 *q, const BF16 *k, const int *indices,
                                      float *scores, int heads, int kvHeads, int dim,
                                      int keyStride, Length liveKeys, Length liveCount, Past livePast,
                                      int window, bool causal) {
    int keys = liveKeys, count = liveCount, past = livePast;
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
template <typename Length = int, typename Past = int>
__global__ void AttentionValuesPrefill(const float *prob, const BF16 *v, const int *indices,
                                      BF16 *out, int heads, int kvHeads, int dim,
                                      Length liveKeys, Length liveCount, Past livePast, int window, bool causal) {
    int keys = liveKeys, count = liveCount, past = livePast;
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

template <typename Length = int>
__global__ void AttentionSoftmax(float *scores, const float *sink, int heads, Length liveCount) {
    int count = liveCount;
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

// Preserve the 256-lane softmax tree and sequential per-thread sum, while
// retaining eight exponentials in registers and folding the tree within a warp.
__global__ void AttentionSoftmaxDecode(float *scores, const float *sink) {
    constexpr int threads = kDecodeSharedThreads, count = kDecodePVKeys;
    constexpr int items = count / threads;
    __shared__ float scratch[threads], maximum, denominator;
    int h = blockIdx.x, t = threadIdx.x;
    float logits[items], bias = sink ? sink[h] : -INFINITY;
    float top = bias;
    #pragma unroll
    for (int i = 0; i < items; ++i) {
        logits[i] = scores[h * count + t + i * threads];
        top = fmaxf(top, logits[i]);
    }
    scratch[t] = top;
    __syncthreads();
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]), b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]), d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float mx = fmaxf(fmaxf(a, b), fmaxf(c, d));
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
        if (t == 0) maximum = mx;
    }
    __syncthreads();
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < items; ++i) {
        logits[i] = expf(logits[i] - maximum);
        sum += logits[i];
    }
    if (t == 0 && sink) sum += expf(bias - maximum);
    scratch[t] = sum;
    __syncthreads();
    if (t < 32) {
        // Fold strides 128, 64 and 32 in the original block-reduction order.
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192];
        float c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (t == 0) denominator = total;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < items; ++i)
        scores[h * count + t + i * threads] = denominator > 0 ? RoundBF16(logits[i] / denominator) : 0;
}

// Short full attention and the 128-token sliding window fit in shared memory.
// Keep eager's BF16 score/probability rounding, sink and reduction order while
// removing the score tensor round-trip and two launches per layer.
template <typename Length = int, typename Past = int>
__global__ void AttentionShort(const BF16 *q, const BF16 *k, const BF16 *v,
                               const int *indices, const float *sink, BF16 *out,
                               int heads, int kvHeads, int dim, int valueDim,
                               int keyStride, Length liveKeys, int count, Past livePast, int window, bool causal) {
    int keys = liveKeys, past = livePast;
    count = ShortCount(liveKeys, count);
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

// Single-query attention with head dimensions 192/128 and up to 256 keys. Four
// output slices spread the work across SMs; cooperative V loads avoid the
// reference kernel's dependent global load for every output/slot pair.
// The softmax tree, BF16 rounding and slot-ordered FP32 FMAs are unchanged.
template <int MaxKeys, typename Length = int, bool Verify = false>
__global__ void AttentionShortDecode(const BF16 *q, const BF16 *k, const BF16 *v,
        const float *sink, BF16 *out, int heads, int kvHeads, int keyStride, Length liveKeys) {
    if constexpr (Verify) {
        const int row = blockIdx.y;
        q += (size_t)row * heads * kSwaQkDim;
        out += (size_t)row * heads * kSwaValueDim;
        if constexpr (MaxKeys == kSwaWindow) {
            // Verification appends all rows before attention. Read each
            // causal window directly instead of copying it into scratch.
            const int end = (int)liveKeys + row;
            const int begin = max(0, end - kSwaWindow);
            k += (size_t)begin * keyStride;
            v += (size_t)begin * kvHeads * kSwaValueDim;
        }
    }
    int keys = Verify ? min(VerifyCount(liveKeys, blockIdx.y), MaxKeys) : (int)liveKeys;
    __shared__ float scores[MaxKeys], scratch[kSwaThreads], maximum, denominator;
    __shared__ BF16 values[MaxKeys][kSwaOutputTile];
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
            dot += query[i] * (float)k[(size_t)slot * keyStride + kvHead * kSwaQkDim + lane + i * 32];
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

// Decode has just one query. Split its 2048 selected keys across CTAs so
// every thread accumulates one value column, then combine FP32 partial sums.
// Inputs/probability rounding and BF16 output are unchanged; the FP32 sum order
// differs from the serial fallback. Prefill and other shapes keep their paths.
template <typename Length = int>
__global__ void AttentionValuesDecodePartial(const float *prob, const BF16 *v,
        const int *indices, float *partial, int heads, int kvHeads,
        Length liveKeys, int past, bool causal) {
    int keys = liveKeys;
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

template<int HeadTile, typename Length = int, bool Verify = false>
__global__ void AttentionValuesDecodeGrouped(const float *prob, const BF16 *v,
        const int *indices, float *partial, int heads, int kvHeads,
        Length liveKeys, int past, bool causal) {
    constexpr int count = kDecodePVKeys, dim = kDecodePVValueDim;
    constexpr int parts = kDecodePVParts, slots = count / parts;
    if constexpr (Verify) {
        const int row = blockIdx.z;
        liveKeys.length += row;
        prob += (size_t)row * heads * count;
        indices += (size_t)row * count;
        partial += (size_t)row * heads * parts * dim;
        past = (int)liveKeys - 1;
    }
    int keys = liveKeys;
    int firstHead = blockIdx.x * HeadTile, part = blockIdx.y, d = threadIdx.x;
    int kvHead = firstHead / (heads / kvHeads);
    float sum[HeadTile] = {};
    for (int base = part * slots; base < (part + 1) * slots; base += 8) {
        float values[8];
        bool valid[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int key = indices ? indices[base + i] : base + i;
            valid[i] = key >= 0 && key < keys && (!causal || key <= past);
            values[i] = valid[i] ? (float)v[((size_t)key * kvHeads + kvHead) * dim + d] : 0;
        }
        #pragma unroll
        for (int h = 0; h < HeadTile; ++h) {
            float probabilities[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                probabilities[i] = prob[(firstHead + h) * count + base + i];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (valid[i]) sum[h] = fmaf(probabilities[i], values[i], sum[h]);
        }
    }
    #pragma unroll
    for (int h = 0; h < HeadTile; ++h)
        partial[((firstHead + h) * parts + part) * dim + d] = sum[h];
}

// Sixteen Q heads share a 64-key by 64-column V tile. Each warp computes
// two heads; the existing 64-key partial sums and final reduction stay ordered.
template <typename Length = int>
__global__ void AttentionValuesDecodeShared(const float *prob, const BF16 *v,
        const int *indices, float *partial, int heads, int kvHeads,
        Length liveKeys, int past, bool causal) {
    int keys = liveKeys;
    constexpr int HeadTile = kDecodeSharedHeads, ValueTile = kDecodeValueTile, HeadsPerWarp = 2;
    constexpr int count = kDecodePVKeys, parts = kDecodePVParts, slots = count / parts;
    constexpr int dim = kDecodePVValueDim, columns = ValueTile / 32, threads = kDecodeSharedThreads;
    __shared__ __align__(16) BF16 values[slots][ValueTile];
    __shared__ int selected[slots];
    __shared__ float probabilities[HeadTile][slots];
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    int firstHead = blockIdx.x * HeadTile, part = blockIdx.y;
    int kvHead = firstHead / (heads / kvHeads), firstColumn = blockIdx.z * ValueTile;
    if (t < slots) {
        int key = indices ? indices[part * slots + t] : part * slots + t;
        selected[t] = key >= 0 && key < keys && (!causal || key <= past) ? key : -1;
    }
    for (int i = t; i < HeadTile * slots; i += threads)
        probabilities[i / slots][i % slots] = prob[(firstHead + i / slots) * count + part * slots + i % slots];
    __syncthreads();
    #pragma unroll
    for (int i = t; i < slots * (ValueTile / 8); i += threads) {
        int row = i / (ValueTile / 8), d = i % (ValueTile / 8) * 8, key = selected[row];
        *reinterpret_cast<uint4 *>(&values[row][d]) = key >= 0
            ? *reinterpret_cast<const uint4 *>(v + ((size_t)key * kvHeads + kvHead) * dim + firstColumn + d)
            : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    float sums[HeadsPerWarp][columns] = {};
    for (int base = 0; base < slots; base += 8) {
        float valueTile[8][columns], probTile[HeadsPerWarp][8];
        bool valid[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            valid[i] = selected[base + i] >= 0;
            #pragma unroll
            for (int d = 0; d < columns; ++d)
                valueTile[i][d] = (float)values[base + i][lane * columns + d];
            #pragma unroll
            for (int h = 0; h < HeadsPerWarp; ++h) {
                int head = warp * HeadsPerWarp + h;
                probTile[h][i] = probabilities[head][base + i];
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (valid[i]) {
                #pragma unroll
                for (int h = 0; h < HeadsPerWarp; ++h) {
                    #pragma unroll
                    for (int d = 0; d < columns; ++d)
                        sums[h][d] = fmaf(probTile[h][i], valueTile[i][d], sums[h][d]);
                }
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < HeadsPerWarp; ++h) {
        #pragma unroll
        for (int d = 0; d < columns; ++d)
            partial[((firstHead + warp * HeadsPerWarp + h) * parts + part) * dim + firstColumn + lane * columns + d] = sums[h][d];
    }
}

__global__ void AttentionValuesDecodeReduce(const float *partial, BF16 *out, int heads) {
    constexpr int dim = kDecodePVValueDim, parts = kDecodePVParts;
    partial += (size_t)blockIdx.y * heads * parts * dim;
    out += (size_t)blockIdx.y * heads * dim;
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
template <typename Length = int, typename Count = int>
__global__ void AttentionValuesTiled(const float *prob, const BF16 *v,
        const int *indices, BF16 *out, int heads, int kvHeads, int dim,
        Length liveKeys, Count liveCount, int past, int window, bool causal) {
    int count = liveCount;
    int keys = liveKeys;
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
    int blocks = CooperativeTopKBlocks(count, topK);
    if (blocks) {
        CooperativeTopK((const float *)scores.cudaData, count, blocks, indices);
        return;
    }
    Data encoded;
    Output(encoded, INT32, {count, 2});
    auto *input = (unsigned *)encoded.cudaData;
    auto *inputPositions = (int *)encoded.cudaData + count;
    EncodeTopKPairs<<<(count + 255) / 256, 256>>>(
        (const float *)scores.cudaData, input, inputPositions, count);
    SortTopKPairs(input, inputPositions, count, topK, indices);
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

void FastllmCudaNaiveRopeQKScaleV(fastllm::Data &q, fastllm::Data &k,
        fastllm::Data &v, const fastllm::Data &positions,
        int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, float valueScale) {
    using namespace fastllm;
    AssertInFastLLM(heads > 0 && kvHeads > 0 && dim > 0 && valueDim > 0 &&
        rotaryDim > 0 && rotaryDim % 2 == 0 && rotaryDim <= dim &&
        q.dims.size() == 3 && q.dims[0] == 1 && q.dims[1] > 0 &&
        q.dims[2] == (int64_t)heads * dim &&
        k.dims.size() == 3 && k.dims[0] == 1 && k.dims[1] == q.dims[1] &&
        k.dims[2] == (int64_t)kvHeads * dim &&
        v.dims.size() == 3 && v.dims[0] == 1 && v.dims[1] == q.dims[1] &&
        v.dims[2] == (int64_t)kvHeads * valueDim &&
        q.dataType == BFLOAT16 && k.dataType == BFLOAT16 && v.dataType == BFLOAT16 &&
        positions.dataType == FLOAT32 && positions.Count(0) >= (uint64_t)q.dims[1] &&
        q.dataDevice == DataDevice::CUDA && k.dataDevice == DataDevice::CUDA &&
        v.dataDevice == DataDevice::CUDA && positions.dataDevice == DataDevice::CUDA &&
        q.dataDeviceIds == k.dataDeviceIds && q.dataDeviceIds == v.dataDeviceIds &&
        q.dataDeviceIds == positions.dataDeviceIds &&
        q.cudaData && k.cudaData && v.cudaData && positions.cudaData,
        "Invalid Naive-N0.5 fused Q/K RoPE and V scale input.");
    RopeQKScaleV<<<(uint64_t)q.dims[1] * (heads + 2 * kvHeads), 128>>>(
        (BF16 *)q.cudaData, (BF16 *)k.cudaData, (BF16 *)v.cudaData,
        (const float *)positions.cudaData, heads, kvHeads, dim, valueDim,
        rotaryDim, theta, __float2bfloat16_rn(valueScale));
    CheckLaunch();
}

bool FastllmCudaNaiveQuantizeIndexer(const fastllm::Data &input,
        fastllm::Data &values, fastllm::Data &scales, bool roundScale) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    using namespace fastllm;
    if (input.dataDevice != DataDevice::CUDA || input.dataType != BFLOAT16 ||
        input.cudaData == nullptr || input.dims.empty() || input.dims.back() != 128 ||
        input.Count(0) == 0 || input.Count(0) / 128 > INT_MAX ||
        input.strides.back() != 1) return false;
    const int rows = input.Count(0) / 128;
    Output(values, BFLOAT16, input.dims);
    auto dims = input.dims; dims.pop_back();
    if (dims.empty()) dims.push_back(1);
    Output(scales, FLOAT32, dims);
    naive_dsa_mma::QuantizeIndexer<<<rows, 128>>>(
        (const BF16 *)input.cudaData, (BF16 *)values.cudaData,
        (float *)scales.cudaData, 128, 0, roundScale);
    return cudaGetLastError() == cudaSuccess;
#else
    return false;
#endif
}

void FastllmCudaNaiveIndexer(const fastllm::Data &query, const fastllm::Data &weights,
                            const fastllm::Data &packedKeys, int heads, int dim,
                            int queryStart, int topK, bool fp8Query, fastllm::Data &indices) {
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
    bool decodeTopK = queries == 1 && heads == 16;
    int selectBlocks = decodeTopK ? CooperativeTopKBlocks(queryStart + 1, topK) : 0;
    Data q, k, scores, positions;
    Output(scores, decodeTopK ? INT32 : FLOAT32, {queries, keys});
    if (decodeTopK && !selectBlocks) Output(positions, INT32, {keys});
    IndexerTopKOutput topKOutput{(unsigned *)scores.cudaData, (int *)positions.cudaData};
#ifdef FASTLLM_NAIVE_DSA_MMA
    using naive_dsa_mma::kKeys;
    using naive_dsa_mma::kThreads;
    // Q keeps its E4M3 values and FP32 scales; K retains the input BF16 values.
    // Decode tiles over keys; larger prefill chunks also share K across queries.
    if (fp8Query && heads == 16 && (queries == 1 ||
        (queries >= 32 && (int64_t)queries * keys >= 1024 * 1024)) &&
        FastllmCudaFlashInferDataTypeSupported(DataType::BFLOAT16)) {
        Data qScale;
        Output(q, DataType::BFLOAT16, {queries, heads, dim});
        Output(qScale, DataType::FLOAT32, {queries, heads});
        naive_dsa_mma::QuantizeIndexer<<<queries * heads, 128>>>(
            (const BF16 *)query.cudaData, (BF16 *)q.cudaData,
            (float *)qScale.cudaData, dim, 0);
        if (queries == 1) {
            naive_dsa_mma::IndexerDecodeScores<<<(keys + kKeys - 1) / kKeys, kThreads>>>(
                (const BF16 *)q.cudaData, (const BF16 *)packedKeys.cudaData,
                (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                topKOutput, stride, keys, queryStart);
        } else {
            naive_dsa_mma::IndexerScores<<<dim3((keys + kKeys - 1) / kKeys,
                (queries + kKeys - 1) / kKeys), kThreads>>>(
                (const BF16 *)q.cudaData, (const BF16 *)packedKeys.cudaData,
                (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                (float *)scores.cudaData, queries, keys, queryStart, stride);
        }
    } else
#endif
    {
        Output(q, DataType::FLOAT32, {queries, heads, dim});
        RoundIndexer<<<queries * heads, 128>>>((const BF16 *)query.cudaData,
            (float *)q.cudaData, dim, 0, fp8Query);
        if (decodeTopK) {
            IndexScoresDecode<<<(keys + 7) / 8, 256>>>((const float *)q.cudaData,
                (const BF16 *)packedKeys.cudaData, (const BF16 *)weights.cudaData,
                topKOutput, stride, keys, queryStart);
        } else {
            Output(k, DataType::FLOAT32, {keys, dim});
            RoundIndexer<<<keys, 128>>>((const BF16 *)packedKeys.cudaData,
                (float *)k.cudaData, stride, stride - dim, false);
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
    // The selector derives positions directly; CUB's fallback uses emitted positions.
    if (selectBlocks) {
        CooperativeTopK((const unsigned *)scores.cudaData, queryStart + 1, selectBlocks, indices);
        return;
    }
    if (decodeTopK) {
        SortTopKPairs((const unsigned *)scores.cudaData, (const int *)positions.cudaData,
            queryStart + 1, topK, indices);
        CheckLaunch();
    } else {
        FastllmCudaNaiveTopK(scores, queryStart, topK, indices);
    }
}

void FastllmCudaNaiveVerifyIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &packedKeys,
        int heads, int dim, int queryStart, int topK, bool fp8Query,
        fastllm::Data &indices) {
    using namespace fastllm;
    const int rows = query.dims[1];
    AssertInFastLLM(rows > 0 && rows <= 8 && queryStart >= 0 &&
        queryStart + rows <= packedKeys.dims[1], "Invalid Naive verification block.");
    Output(indices, INT32, {rows, topK});
    for (int row = 0; row < rows; ++row) {
        Data q(BFLOAT16, {1, 1, heads * dim}), w(BFLOAT16, {1, 1, heads});
        Data keys(BFLOAT16, {1, queryStart + row + 1, packedKeys.dims[2]});
        Data out(INT32, {1, topK});
        q.FakeFrom(query, (size_t)row * heads * dim * sizeof(BF16));
        w.FakeFrom(weights, (size_t)row * heads * sizeof(BF16));
        keys.FakeFrom(packedKeys, 0);
        out.FakeFrom(indices, (size_t)row * topK * sizeof(int));
        FastllmCudaNaiveIndexer(q, w, keys, heads, dim, queryStart + row, topK, fp8Query, out);
    }
}

void FastllmCudaNaiveVerifyAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        int heads, int kvHeads, int dim, int valueDim, int pastLength,
        int window, fastllm::Data &output) {
    using namespace fastllm;
    const int rows = query.dims[1];
    AssertInFastLLM(rows > 0 && rows <= 8 && pastLength >= 0 &&
        pastLength + rows <= key.dims[1], "Invalid Naive verification attention block.");
    Output(output, BFLOAT16, {1, rows, heads * valueDim});
    // Full-window, large-head eager attention may use FlashInfer. Preserve
    // that arithmetic; the compact decode kernel is exact for its own path.
    const bool compactWindow = window == kSwaWindow &&
        (heads < 64 || heads / kvHeads != 8 || pastLength + rows < kSwaWindow);
    if (dim == kSwaQkDim && valueDim == kSwaValueDim && indices.dims.empty() &&
        (compactWindow || (!window && pastLength + rows <= 256))) {
        const int first = window ? std::min(pastLength + 1, window) : pastLength + 1;
        auto *q = (const BF16 *)query.cudaData;
        auto *k = (const BF16 *)key.cudaData;
        auto *v = (const BF16 *)value.cudaData;
        auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        auto *out = (BF16 *)output.cudaData;
        // An eager caller can retain more than window-1 old rows.
        const int begin = window ? std::max(0, pastLength + 1 - window) : 0;
        k += (size_t)begin * key.dims[2];
        v += (size_t)begin * kvHeads * valueDim;
        dim3 grid(heads, rows, kSwaValueDim / kSwaOutputTile);
        if (window)
            AttentionShortDecode<kSwaWindow, int, true><<<grid, kSwaThreads>>>(
                q, k, v, bias, out, heads, kvHeads, key.dims[2], first);
        else
            AttentionShortDecode<256, int, true><<<grid, kSwaThreads>>>(
                q, k, v, bias, out, heads, kvHeads, key.dims[2], first);
        CheckLaunch();
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const int end = pastLength + row + 1;
        const int begin = window ? std::max(0, end - window) : 0;
        const int length = end - begin;
        Data q(BFLOAT16, {1, 1, heads * dim});
        Data k(BFLOAT16, {1, length, key.dims[2]});
        Data v(BFLOAT16, {1, length, value.dims[2]});
        Data out(BFLOAT16, {1, 1, heads * valueDim}), selected;
        q.FakeFrom(query, (size_t)row * heads * dim * sizeof(BF16));
        k.FakeFrom(key, (size_t)begin * key.dims[2] * sizeof(BF16));
        v.FakeFrom(value, (size_t)begin * value.dims[2] * sizeof(BF16));
        out.FakeFrom(output, (size_t)row * heads * valueDim * sizeof(BF16));
        // Crossing the dense/sparse boundary inside a block must not change
        // the PV reduction tree of the earlier, still-dense positions.
        if (!indices.dims.empty() && end > indices.dims[1]) {
            selected.Resize({1, indices.dims[1]});
            selected.FakeFrom(indices, (size_t)row * indices.dims[1] * sizeof(int));
        }
        FastllmCudaNaiveAttention(q, k, v, selected, sink, heads, kvHeads, dim,
                                 valueDim, length - 1, window, out);
    }
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
        AttentionShortDecode<kSwaWindow><<<dim3(heads, 1, kSwaValueDim / kSwaOutputTile), kSwaThreads>>>(
            (const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, heads, kvHeads, key.dims[2], keys);
        CheckLaunch();
        return;
    }
    if (queries == 1 && window == 0 && causal && !selected &&
        keys > 0 && keys <= 256 && pastLength == keys - 1 &&
        dim == kSwaQkDim && valueDim == kSwaValueDim) {
        AttentionShortDecode<256><<<dim3(heads, 1, kSwaValueDim / kSwaOutputTile), kSwaThreads>>>(
            (const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, heads, kvHeads, key.dims[2], keys);
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
    // Sequence-split PV also matters for TP shards: a handful of local
    // heads otherwise leaves the long serial PV loop almost un-parallelized.
    const bool splitDecode = queries == 1 && window == 0 && count == kDecodePVKeys && heads >= 4;
    const bool groupedDecode = splitDecode && heads % kvHeads == 0 && (heads / kvHeads) % 4 == 0;
    const bool sharedDecode = groupedDecode && (heads / kvHeads) % kDecodeSharedHeads == 0;
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
    // Each shared tile must stay within one KV head and have aligned vector loads.
    if (sharedDecode && dim == kDecodeQkDim && key.dims[2] % 8 == 0 &&
        (size_t)key.cudaData % 16 == 0) {
        AttentionScoresDecodeShared<<<dim3(heads / kDecodeSharedHeads, count / kDecodeQkKeyTile), kDecodeSharedThreads>>>(
            (const BF16 *)query.cudaData, (const BF16 *)key.cudaData, selected,
            (float *)scores.cudaData, heads, kvHeads, dim, key.dims[2],
            keys, count, pastLength, causal);
    } else if (groupedDecode && dim <= kDecodeQkDim) {
        AttentionScoresDecodeGrouped<4, kDecodeQkDim><<<dim3(heads / 4, count / 32), 128>>>(
            (const BF16 *)query.cudaData, (const BF16 *)key.cudaData, selected,
            (float *)scores.cudaData, heads, kvHeads, dim, key.dims[2],
            keys, count, pastLength, causal);
    } else
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
    if (sharedDecode) {
        AttentionSoftmaxDecode<<<heads, kDecodeSharedThreads>>>((float *)scores.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData);
    } else {
        AttentionSoftmax<<<queries * heads, 256>>>((float *)scores.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData, heads, count);
    }
#ifdef FASTLLM_NAIVE_DSA_MMA
    if (useMma) {
        naive_dsa_mma::Values<<<dim3(queries, heads / naive_dsa_mma::kHeads), naive_dsa_mma::kThreads>>>(
            (const float *)scores.cudaData, (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
            heads, kvHeads, keys, count, pastLength, causal);
    } else
#endif
    if (splitDecode && valueDim == kDecodePVValueDim) {
        Data partial;
        Output(partial, FLOAT32, {heads, kDecodePVParts, kDecodePVValueDim});
        if (sharedDecode && (size_t)value.cudaData % 16 == 0) {
            AttentionValuesDecodeShared<<<dim3(heads / kDecodeSharedHeads, kDecodePVParts,
                kDecodePVValueDim / kDecodeValueTile), kDecodeSharedThreads>>>(
                (const float *)scores.cudaData, (const BF16 *)value.cudaData,
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        } else if (groupedDecode) {
            AttentionValuesDecodeGrouped<4><<<dim3(heads / 4, kDecodePVParts), kDecodePVValueDim>>>(
                (const float *)scores.cudaData, (const BF16 *)value.cudaData,
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        } else {
            AttentionValuesDecodePartial<<<dim3(heads, kDecodePVParts), kDecodePVValueDim>>>(
                (const float *)scores.cudaData, (const BF16 *)value.cudaData,
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        }
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

namespace {
// Preserve AddTo's BF16 residual rounding and KimiK3RMSNorm's two BF16
// rounding steps, including the same 256-thread reduction tree.
template <int Channels = 0>
__global__ void AddDecodeRMSNorm(BF16 *hidden, const BF16 *branch,
        const float *weight, BF16 *output, int channels, float eps) {
    if constexpr (Channels) channels = Channels;
    hidden += (size_t)blockIdx.x * channels;
    branch += (size_t)blockIdx.x * channels;
    output += (size_t)blockIdx.x * channels;
    __shared__ float sums[8];
    float cached[Channels ? Channels / 256 : 1];
    float affine[Channels ? Channels / 256 : 1];
    float partial = 0;
    if constexpr (Channels) {
        // Load the entire row before reduction, as the eager 4096-wide kernel
        // does. Loading affine weights after the reduction serializes memory
        // latency on this one-CTA decode workload.
        #pragma unroll
        for (int part = 0; part < Channels / 256; ++part) {
            int c = threadIdx.x + part * 256;
            cached[part] = RoundBF16((float)hidden[c] + (float)branch[c]);
            affine[part] = weight[c];
        }
        #pragma unroll
        for (int part = 0; part < Channels / 256; ++part) {
            hidden[threadIdx.x + part * 256] = __float2bfloat16(cached[part]);
            partial += cached[part] * cached[part];
        }
    } else {
        for (int c = threadIdx.x; c < channels; c += 256) {
            float value = RoundBF16((float)hidden[c] + (float)branch[c]);
            hidden[c] = __float2bfloat16(value);
            partial += value * value;
        }
    }
    partial = WarpSum(partial);
    if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = partial;
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = WarpSum(threadIdx.x < 8 ? sums[threadIdx.x] : 0);
        if (threadIdx.x == 0) sums[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(sums[0] / channels + eps);
    if constexpr (Channels) {
        #pragma unroll
        for (int part = 0; part < Channels / 256; ++part)
            output[threadIdx.x + part * 256] =
                __float2bfloat16(RoundBF16(cached[part] * scale) * affine[part]);
    } else {
        for (int c = threadIdx.x; c < channels; c += 256)
            output[c] = __float2bfloat16(RoundBF16((float)hidden[c] * scale) * weight[c]);
    }
}

__global__ void DraftInput(const float *id, const BF16 *embedding, const BF16 *mask,
        const int *length, BF16 *hidden, float *positions, int channels) {
    int row = blockIdx.y, column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < channels)
        hidden[(size_t)row * channels + column] = row ? mask[column]
            : embedding[(size_t)(int)*id * channels + column];
    if (column == 0) positions[row] = *length - 1 + row;
}

__global__ void AppendDecodeKV(BF16 *key, BF16 *value, const BF16 *newKey,
        const BF16 *newValue, const int *length, int keyColumns, int valueColumns,
        int window) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = *length - 1;
    if (window) row = min(row, window - 1);
    if (column < keyColumns) key[(size_t)row * keyColumns + column] = newKey[column];
    else if ((column -= keyColumns) < valueColumns)
        value[(size_t)row * valueColumns + column] = newValue[column];
}

__global__ void AppendVerifyKV(BF16 *key, BF16 *value, const BF16 *newKey,
        const BF16 *newValue, const int *length, int kc, int vc, int window) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y;
    const int past = window ? min(*length - 1, window - 1) : *length - 1;
    const int row = past + query;
    if (column < kc) key[(size_t)row * kc + column] = newKey[(size_t)query * kc + column];
    else if ((column -= kc) < vc)
        value[(size_t)row * vc + column] = newValue[(size_t)query * vc + column];
}

__global__ void CopyVerifyWindow(const BF16 *key, const BF16 *value,
        BF16 *outKey, BF16 *outValue, const int *length, int query,
        int kc, int vc, int window) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int end = min(*length - 1, window - 1) + query + 1;
    const int begin = max(0, end - window), rows = min(end, window);
    if (index < rows * kc) outKey[index] = key[(size_t)begin * kc + index];
    else if ((index -= rows * kc) < rows * vc)
        outValue[index] = value[(size_t)begin * vc + index];
}

int DecodeTopKBlocks() {
    static thread_local std::map<int, int> cache;
    int device = FastllmCudaGetDevice();
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    cudaDeviceProp prop;
    int resident = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
            naive_topk::Select<unsigned, DecodeKeys>, naive_topk::kThreads, 0) != cudaSuccess) {
        cudaGetLastError();
        return cache[device] = 0;
    }
    return cache[device] = std::min(naive_topk::kMaxBlocks, resident * prop.multiProcessorCount);
}
}

bool FastllmCudaNaiveDecodeGraphSupported() { return DecodeTopKBlocks() > 0; }

void FastllmCudaNaiveAddDecodeRMSNorm(fastllm::Data &hidden,
        const fastllm::Data &branch, const fastllm::Data &weight,
        float eps, fastllm::Data &output) {
    Output(output, fastllm::BFLOAT16, hidden.dims);
    int channels = hidden.dims.back();
    auto kernel = channels == 4096 ? AddDecodeRMSNorm<4096> : AddDecodeRMSNorm<>;
    kernel<<<hidden.Count(0) / channels, 256>>>((BF16 *)hidden.cudaData, (const BF16 *)branch.cudaData,
        (const float *)weight.cudaData, (BF16 *)output.cudaData, channels, eps);
    CheckLaunch();
}

void FastllmCudaNaiveAppendDecodeCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &newKey, const fastllm::Data &newValue,
        const fastllm::Data &liveKeys, int window) {
    int kc = key.dims[2], vc = value.dims[2];
    AppendDecodeKV<<<(kc + vc + 255) / 256, 256>>>((BF16 *)key.cudaData,
        (BF16 *)value.cudaData, (const BF16 *)newKey.cudaData,
        (const BF16 *)newValue.cudaData, (const int *)liveKeys.cudaData, kc, vc, window);
    CheckLaunch();
}

void FastllmCudaNaiveTrimDecodeCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &liveKeys, int window) {
    constexpr int columns = kTrimTileBytes / sizeof(BF16);
    int kc = key.dims[2], vc = value.dims[2];
    int blocks = (kc + columns - 1) / columns + (vc + columns - 1) / columns;
    TrimCachePairTiled<BF16, true><<<blocks, 256>>>((BF16 *)key.cudaData,
        (BF16 *)value.cudaData, kc, vc, 1, window - 1, (const int *)liveKeys.cudaData);
    CheckLaunch();
}

void FastllmCudaNaiveDecodeIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &packedKeys,
        const fastllm::Data &liveKeys, int capacity, bool fp8Query,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &indices) {
    using namespace fastllm;
    DecodeKeys keys{(const int *)liveKeys.cudaData, 0};
    const int stride = packedKeys.dims[2];
    Output(scratch.indexScores, INT32, {capacity});
    IndexerTopKOutput scoreOutput{(unsigned *)scratch.indexScores.cudaData, nullptr};
#ifdef FASTLLM_NAIVE_DSA_MMA
    if (fp8Query && FastllmCudaFlashInferDataTypeSupported(BFLOAT16)) {
        Output(scratch.indexQuery, BFLOAT16, {16, 128});
        Output(scratch.indexScale, FLOAT32, {16});
        naive_dsa_mma::QuantizeIndexer<<<16, 128>>>((const BF16 *)query.cudaData,
            (BF16 *)scratch.indexQuery.cudaData, (float *)scratch.indexScale.cudaData, 128, 0);
        naive_dsa_mma::IndexerDecodeScores<<<(capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys,
            naive_dsa_mma::kThreads>>>((const BF16 *)scratch.indexQuery.cudaData,
            (const BF16 *)packedKeys.cudaData, (const float *)scratch.indexScale.cudaData,
            (const BF16 *)weights.cudaData, scoreOutput, stride, keys, capacity - 1);
    } else
#endif
    {
        Output(scratch.indexQuery, FLOAT32, {16, 128});
        RoundIndexer<<<16, 128>>>((const BF16 *)query.cudaData,
            (float *)scratch.indexQuery.cudaData, 128, 0, fp8Query);
        IndexScoresDecode<<<(capacity + 7) / 8, 256>>>((const float *)scratch.indexQuery.cudaData,
            (const BF16 *)packedKeys.cudaData, (const BF16 *)weights.cudaData,
            scoreOutput, stride, keys, capacity - 1);
    }
    Output(scratch.topk, INT32, {(int)(sizeof(naive_topk::Workspace) / sizeof(int))});
    Output(indices, INT32, {1, naive_topk::kTopK});
    auto *workspace = (naive_topk::Workspace *)scratch.topk.cudaData;
    auto *scores = (const unsigned *)scratch.indexScores.cudaData;
    auto *histograms = workspace->partials;
    auto *ties = workspace->ties;
    auto *state = &workspace->state;
    auto *candidates = &workspace->candidates;
    void *args[] = {&scores, &keys, &histograms, &ties, &state, &candidates};
    auto status = cudaLaunchCooperativeKernel((void *)naive_topk::Select<unsigned, DecodeKeys>,
        DecodeTopKBlocks(), naive_topk::kThreads, args, 0, cudaStreamPerThread);
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        if (!FastllmCudaGraphIsCapturingFast())
            AssertInFastLLM(false, "Naive decode graph TopK launch failed.");
    }
    naive_topk::Sort<<<1, naive_topk::kThreads>>>(candidates, state, (int *)indices.cudaData);
    CheckLaunch();
}

void FastllmCudaNaiveDecodeAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        const fastllm::Data &liveKeys, int capacity, int heads, int kvHeads,
        int dim, int valueDim, int window,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output) {
    using namespace fastllm;
    DecodeKeys keys{(const int *)liveKeys.cudaData, window};
    auto *q = (const BF16 *)query.cudaData;
    auto *k = (const BF16 *)key.cudaData;
    auto *v = (const BF16 *)value.cudaData;
    auto *selected = indices.dims.empty() ? nullptr : (const int *)indices.cudaData;
    auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    Output(output, BFLOAT16, {1, 1, heads * valueDim});
    auto *out = (BF16 *)output.cudaData;
    if (window == kSwaWindow && dim == kSwaQkDim && valueDim == kSwaValueDim) {
        AttentionShortDecode<kSwaWindow><<<dim3(heads, 1, kSwaValueDim / kSwaOutputTile), kSwaThreads>>>(
            q, k, v, bias, out, heads, kvHeads, key.dims[2], keys);
    } else if (!window && !selected && capacity > 0 && capacity <= 256 &&
               dim == kSwaQkDim && valueDim == kSwaValueDim) {
        // Bound the live length to the capacity captured for this graph.
        DecodeKeys shortKeys{(const int *)liveKeys.cudaData, capacity};
        AttentionShortDecode<256><<<dim3(heads, 1, kSwaValueDim / kSwaOutputTile), kSwaThreads>>>(
            q, k, v, bias, out, heads, kvHeads, key.dims[2], shortKeys);
    } else if (window || capacity <= 256) {
        int count = window ? window : capacity;
        AttentionShort<<<heads, 256>>>(q, k, v, selected, bias, out, heads, kvHeads,
            dim, valueDim, key.dims[2], keys, count, count - 1, window, true);
    } else {
        int count = selected ? kDecodePVKeys : capacity;
        Output(scratch.attentionScores, FLOAT32, {heads, count});
        auto *scores = (float *)scratch.attentionScores.cudaData;
        if (!selected) {
            // Keep the live row stride and exact eager softmax reduction tree.
            AttentionScores<<<dim3(heads, 1, (capacity + 63) / 64), 256>>>(
                q, k, selected, scores, heads, kvHeads, dim, key.dims[2],
                keys, keys, capacity - 1, 0, true);
            AttentionSoftmax<<<heads, 256>>>(scores, bias, heads, keys);
            AttentionValuesTiled<<<dim3(heads, 1, (valueDim + 31) / 32), 256>>>(
                scores, v, selected, out, heads, kvHeads, valueDim,
                keys, keys, capacity - 1, 0, true);
        } else {
            bool split = heads >= 4 && valueDim == kDecodePVValueDim;
            bool grouped = heads % kvHeads == 0 && (heads / kvHeads) % 4 == 0;
            bool shared = grouped && (heads / kvHeads) % kDecodeSharedHeads == 0;
            if (shared && dim == kDecodeQkDim && key.dims[2] % 8 == 0 && (size_t)k % 16 == 0) {
                AttentionScoresDecodeShared<<<dim3(heads / kDecodeSharedHeads, count / kDecodeQkKeyTile), kDecodeSharedThreads>>>(
                    q, k, selected, scores, heads, kvHeads, dim, key.dims[2], keys, count, capacity - 1, true);
            } else if (grouped && dim <= kDecodeQkDim) {
                AttentionScoresDecodeGrouped<4, kDecodeQkDim><<<dim3(heads / 4, count / 32), 128>>>(
                    q, k, selected, scores, heads, kvHeads, dim, key.dims[2], keys, count, capacity - 1, true);
            } else {
                AttentionScores<<<dim3(heads, 1, count / 64), 256>>>(q, k, selected, scores,
                    heads, kvHeads, dim, key.dims[2], keys, count, capacity - 1, 0, true);
            }
            if (shared) AttentionSoftmaxDecode<<<heads, kDecodeSharedThreads>>>(scores, bias);
            else AttentionSoftmax<<<heads, 256>>>(scores, bias, heads, count);
            if (split) {
                Output(scratch.attentionPartial, FLOAT32, {heads, kDecodePVParts, valueDim});
                auto *partial = (float *)scratch.attentionPartial.cudaData;
                if (shared && (size_t)v % 16 == 0) {
                    AttentionValuesDecodeShared<<<dim3(heads / kDecodeSharedHeads, kDecodePVParts, 2), kDecodeSharedThreads>>>(
                        scores, v, selected, partial, heads, kvHeads, keys, capacity - 1, true);
                } else if (grouped) {
                    AttentionValuesDecodeGrouped<4><<<dim3(heads / 4, kDecodePVParts), kDecodePVValueDim>>>(
                        scores, v, selected, partial, heads, kvHeads, keys, capacity - 1, true);
                } else {
                    AttentionValuesDecodePartial<<<dim3(heads, kDecodePVParts), kDecodePVValueDim>>>(
                        scores, v, selected, partial, heads, kvHeads, keys, capacity - 1, true);
                }
                AttentionValuesDecodeReduce<<<(heads * valueDim + 255) / 256, 256>>>(partial, out, heads);
            } else {
                AttentionValuesTiled<<<dim3(heads, 1, (valueDim + 31) / 32), 256>>>(scores, v,
                    selected, out, heads, kvHeads, valueDim, keys, count, capacity - 1, 0, true);
            }
        }
    }
    CheckLaunch();
}

void FastllmCudaNaiveAppendVerifyCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &newKey, const fastllm::Data &newValue,
        const fastllm::Data &liveKeys, int window) {
    const int kc = key.dims[2], vc = value.dims[2], rows = newKey.dims[1];
    AppendVerifyKV<<<dim3((kc + vc + 255) / 256, rows), 256>>>((BF16 *)key.cudaData,
        (BF16 *)value.cudaData, (const BF16 *)newKey.cudaData,
        (const BF16 *)newValue.cudaData, (const int *)liveKeys.cudaData, kc, vc, window);
    CheckLaunch();
}

namespace {
__global__ void DraftEmbedding(const float *ids, int step, const BF16 *weight,
                               BF16 *latent, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) latent[col] = weight[(size_t)(int)ids[step] * width + col];
}

// The original Top1 folds lanes at offsets 128,64,...,1, retaining its left
// operand on equality. Thus ties prefer bit-reversed lane order, then the
// first vocabulary entry visited by that lane (stride 256).
__device__ bool DraftBetter(float score, int id, float best, int bestId) {
    if (score != best) return score > best;
    unsigned rank = __brev((unsigned)id & 255u), bestRank = __brev((unsigned)bestId & 255u);
    return rank < bestRank || (rank == bestRank && id < bestId);
}
__device__ void DraftWarpMax(float &score, int &id) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, score, offset);
        int otherId = __shfl_down_sync(0xffffffffu, id, offset);
        if ((threadIdx.x & 31) + offset < 32 && DraftBetter(other, otherId, score, id)) {
            score = other; id = otherId;
        }
    }
}
__device__ void DraftBlockMax(float &score, int &id) {
    __shared__ float scores[8];
    __shared__ int ids[8];
    DraftWarpMax(score, id);
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (!lane) { scores[warp] = score; ids[warp] = id; }
    __syncthreads();
    if (!warp) {
        score = lane < 8 ? scores[lane] : -INFINITY;
        id = lane < 8 ? ids[lane] : INT_MAX;
        DraftWarpMax(score, id);
    }
}
__global__ void DraftArgmaxPartial(const BF16 *base, const BF16 *bias,
                                  float2 *partial, int vocab) {
    float best = -INFINITY;
    int bestId = INT_MAX;
    for (int i = blockIdx.x * 1024 + threadIdx.x;
         i < min((int)(blockIdx.x + 1) * 1024, vocab); i += 256) {
        float score = RoundBF16(__bfloat162float(base[i]) + __bfloat162float(bias[i]));
        // Like the original per-lane scan, ignore NaN and -infinity.
        if (score > -INFINITY && DraftBetter(score, i, best, bestId)) {
            best = score; bestId = i;
        }
    }
    DraftBlockMax(best, bestId);
    if (!threadIdx.x) partial[blockIdx.x] = make_float2((float)bestId, best);
}
__global__ void DraftArgmaxFinish(const float2 *partial, int count,
                                 float *ids, int step) {
    float best = -INFINITY;
    int bestId = INT_MAX;
    for (int i = threadIdx.x; i < count; i += 256) {
        float2 item = partial[i];
        // INT_MAX is not exactly representable as float; empty tiles use
        // their score to avoid converting that sentinel back to int.
        if (item.y > -INFINITY && DraftBetter(item.y, (int)item.x, best, bestId)) {
            best = item.y; bestId = (int)item.x;
        }
    }
    DraftBlockMax(best, bestId);
    if (!threadIdx.x) ids[step + 1] = bestId == INT_MAX ? 0.0f : (float)bestId;
}
struct DraftHiddenInputs { const BF16 *ptr[32]; };
__global__ void DraftConcat(DraftHiddenInputs inputs, BF16 *output,
                            int rows, int width, int count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)rows * width * count;
    if (index < total) {
        int column = index % width;
        int layer = (index / width) % count;
        int row = index / ((size_t)width * count);
        output[index] = inputs.ptr[layer][(size_t)row * width + column];
    }
}
}

void FastllmCudaNaiveDraftEmbedding(const fastllm::Data &ids, int step,
    const fastllm::Data &weight, fastllm::Data &latent) {
    const int width = weight.dims[1];
    Output(latent, fastllm::BFLOAT16, {1, 1, width});
    DraftEmbedding<<<(width + 255) / 256, 256>>>((const float *)ids.cudaData,
        step, (const BF16 *)weight.cudaData, (BF16 *)latent.cudaData, width);
    CheckLaunch();
}
void FastllmCudaNaiveDraftArgmax(const fastllm::Data &base,
    const fastllm::Data &bias, int step, fastllm::Data &partial, fastllm::Data &ids) {
    const int vocab = base.dims.back(), blocks = (vocab + 1023) / 1024;
    Output(partial, fastllm::FLOAT32, {blocks, 2});
    DraftArgmaxPartial<<<blocks, 256>>>((const BF16 *)base.cudaData + (size_t)step * vocab,
        (const BF16 *)bias.cudaData, (float2 *)partial.cudaData, vocab);
    DraftArgmaxFinish<<<1, 256>>>((const float2 *)partial.cudaData, blocks,
        (float *)ids.cudaData, step);
    CheckLaunch();
}
bool FastllmCudaNaiveDraftConcat(const std::vector<const fastllm::Data *> &inputs,
    int rows, fastllm::Data &output) {
    if (inputs.empty() || inputs.size() > 32 || rows <= 0) return false;
    int width = inputs[0]->dims.empty() ? 0 : inputs[0]->dims.back();
    if (width <= 0) return false;
    const std::vector<int> deviceIds{FastllmCudaGetDevice()};
    DraftHiddenInputs pointers{};
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &x = *inputs[i];
        if (x.dataType != fastllm::BFLOAT16 || x.dims.size() != 3 || x.dims[0] != 1 ||
            x.dims[1] < rows || x.dims[2] != width || x.dataDevice != fastllm::DataDevice::CUDA ||
            x.dataDeviceIds != deviceIds || !x.cudaData || x.strides.size() != 3 ||
            x.strides[1] != width || x.strides[2] != 1) return false;
        pointers.ptr[i] = (const BF16 *)x.cudaData;
    }
    Output(output, fastllm::BFLOAT16, {1, rows, width * (int)inputs.size()});
    size_t total = (size_t)rows * width * inputs.size();
    DraftConcat<<<(total + 255) / 256, 256>>>(pointers, (BF16 *)output.cudaData,
        rows, width, inputs.size());
    CheckLaunch();
    return true;
}

void FastllmCudaNaiveDraftInput(const fastllm::Data &id, const fastllm::Data &embedding,
        const fastllm::Data &mask, const fastllm::Data &liveKeys, int rows,
        fastllm::Data &hidden, fastllm::Data &positions) {
    using namespace fastllm;
    int channels = embedding.dims.back();
    Output(hidden, BFLOAT16, {1, rows, channels});
    Output(positions, FLOAT32, {1, rows});
    DraftInput<<<dim3((channels + 255) / 256, rows), 256>>>((const float *)id.cudaData,
        (const BF16 *)embedding.cudaData, (const BF16 *)mask.cudaData,
        (const int *)liveKeys.cudaData, (BF16 *)hidden.cudaData,
        (float *)positions.cudaData, channels);
    CheckLaunch();
}

void FastllmCudaNaiveDraftAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int window,
        bool shortAttention, fastllm::Data &scores, fastllm::Data &output) {
    using namespace fastllm;
    int rows = query.dims[1];
    AssertInFastLLM(rows > 1 && rows < 32 && dim > 0 && dim <= 256 && dim % 4 == 0,
                    "Unsupported Naive draft graph attention shape.");
    Output(output, BFLOAT16, {1, rows, heads * dim});
    DraftLength keys{(const int *)liveKeys.cudaData, window, rows};
    DraftLength past{(const int *)liveKeys.cudaData, window, 0};
    auto *q = (const BF16 *)query.cudaData, *k = (const BF16 *)key.cudaData;
    auto *v = (const BF16 *)value.cudaData;
    auto *out = (BF16 *)output.cudaData;
    if (shortAttention) {
        AttentionShort<<<dim3(heads, rows), 256>>>(q, k, v, nullptr, nullptr, out,
            heads, kvHeads, dim, dim, key.dims[2], keys, 256, past, window, false);
    } else {
        Output(scores, FLOAT32, {rows, heads, window + rows - 1});
        auto kernel = dim <= 192 ? AttentionScoresPrefill<192, DraftLength, DraftLength>
                                : AttentionScoresPrefill<256, DraftLength, DraftLength>;
        kernel<<<dim3(heads, rows), 128>>>(q, k, nullptr, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, keys, past, window, false);
        AttentionSoftmax<<<rows * heads, 256>>>((float *)scores.cudaData, nullptr, heads, keys);
        AttentionValuesPrefill<<<dim3((heads + 3) / 4, rows), 128>>>((float *)scores.cudaData,
            v, nullptr, out, heads, kvHeads, dim, keys, keys, past, window, false);
    }
    CheckLaunch();
}

void FastllmCudaNaiveGraphVerifyIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &packedKeys,
        const fastllm::Data &liveKeys, int capacity, bool fp8Query,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &indices) {
    using namespace fastllm;
    const int rows = query.dims[1];
    if (rows == 1) {
        FastllmCudaNaiveDecodeIndexer(query, weights, packedKeys, liveKeys,
            capacity, fp8Query, scratch, indices);
        return;
    }
    // Score every causal row together, retaining each row's exact decode MMA
    // and head accumulation. Cooperative selection stays within its original
    // resident grid; independent workspaces let the final sorts run together.
    Output(indices, INT32, {rows, naive_topk::kTopK});
    Output(scratch.indexScores, INT32, {rows, capacity});
    Output(scratch.topk, INT32, {rows, (int)(sizeof(naive_topk::Workspace) / sizeof(int))});
    DecodeKeys keys{(const int *)liveKeys.cudaData, 0};
    const int stride = packedKeys.dims[2];
    IndexerTopKOutput scoreOutput{(unsigned *)scratch.indexScores.cudaData, nullptr};
#ifdef FASTLLM_NAIVE_DSA_MMA
    if (fp8Query && FastllmCudaFlashInferDataTypeSupported(BFLOAT16)) {
        Output(scratch.indexQuery, BFLOAT16, {rows * 16, 128});
        Output(scratch.indexScale, FLOAT32, {rows * 16});
        naive_dsa_mma::QuantizeIndexer<<<rows * 16, 128>>>((const BF16 *)query.cudaData,
            (BF16 *)scratch.indexQuery.cudaData, (float *)scratch.indexScale.cudaData, 128, 0);
        naive_dsa_mma::IndexerDecodeScores<IndexerTopKOutput, DecodeKeys, true>
            <<<dim3((capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys, rows), naive_dsa_mma::kThreads>>>(
                (const BF16 *)scratch.indexQuery.cudaData, (const BF16 *)packedKeys.cudaData,
                (const float *)scratch.indexScale.cudaData, (const BF16 *)weights.cudaData,
                scoreOutput, stride, keys, capacity - 1);
    } else
#endif
    {
        Output(scratch.indexQuery, FLOAT32, {rows * 16, 128});
        RoundIndexer<<<rows * 16, 128>>>((const BF16 *)query.cudaData,
            (float *)scratch.indexQuery.cudaData, 128, 0, fp8Query);
        IndexScoresDecode<IndexerTopKOutput, DecodeKeys, true><<<dim3((capacity + 7) / 8, rows), 256>>>(
            (const float *)scratch.indexQuery.cudaData, (const BF16 *)packedKeys.cudaData,
            (const BF16 *)weights.cudaData, scoreOutput, stride, keys, capacity - 1);
    }
    auto *workspaces = (naive_topk::Workspace *)scratch.topk.cudaData;
    for (int row = 0; row < rows; ++row) {
        auto *workspace = workspaces + row;
        auto *scores = (const unsigned *)scratch.indexScores.cudaData + (size_t)row * capacity;
        DecodeKeys count{(const int *)liveKeys.cudaData + row, 0};
        auto *histograms = workspace->partials;
        auto *ties = workspace->ties;
        auto *state = &workspace->state;
        auto *candidates = &workspace->candidates;
        void *args[] = {&scores, &count, &histograms, &ties, &state, &candidates};
        auto status = cudaLaunchCooperativeKernel((void *)naive_topk::Select<unsigned, DecodeKeys>,
            DecodeTopKBlocks(), naive_topk::kThreads, args, 0, cudaStreamPerThread);
        if (status != cudaSuccess) {
            FastllmCudaSetThreadError();
            if (!FastllmCudaGraphIsCapturingFast())
                AssertInFastLLM(false, "Naive verify TopK launch failed.");
        }
    }
    naive_topk::Sort<<<rows, naive_topk::kThreads>>>(&workspaces->candidates, &workspaces->state,
        (int *)indices.cudaData, sizeof(naive_topk::Workspace));
    CheckLaunch();
}

void FastllmCudaNaiveGraphVerifyAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        const fastllm::Data &liveKeys, int capacity, int heads, int kvHeads,
        int dim, int valueDim, int window,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output) {
    using namespace fastllm;
    const int rows = query.dims[1];
    Output(output, BFLOAT16, {1, rows, heads * valueDim});
    if (dim == kSwaQkDim && valueDim == kSwaValueDim &&
        (window == kSwaWindow || (!window && indices.dims.empty() && capacity > 0 && capacity <= 256))) {
        DecodeKeys keys{(const int *)liveKeys.cudaData, window ? window : capacity};
        auto *q = (const BF16 *)query.cudaData;
        auto *k = (const BF16 *)key.cudaData;
        auto *v = (const BF16 *)value.cudaData;
        auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        auto *out = (BF16 *)output.cudaData;
        dim3 grid(heads, rows, kSwaValueDim / kSwaOutputTile);
        if (window)
            AttentionShortDecode<kSwaWindow, DecodeKeys, true><<<grid, kSwaThreads>>>(
                q, k, v, bias, out, heads, kvHeads, key.dims[2], keys);
        else
            AttentionShortDecode<256, DecodeKeys, true><<<grid, kSwaThreads>>>(
                q, k, v, bias, out, heads, kvHeads, key.dims[2], keys);
        CheckLaunch();
        return;
    }
    const bool grouped = heads % kvHeads == 0 && (heads / kvHeads) % 4 == 0;
    const bool shared = grouped && (heads / kvHeads) % kDecodeSharedHeads == 0;
    if (rows > 1 && !window && !indices.dims.empty() && grouped && !shared &&
        dim <= kDecodeQkDim && valueDim == kDecodePVValueDim) {
        // The grouped decode kernels keep their per-row arithmetic and causal
        // lengths, with independent score/partial slices indexed by grid.z.
        constexpr int count = kDecodePVKeys;
        Output(scratch.attentionScores, FLOAT32, {rows, heads, count});
        Output(scratch.attentionPartial, FLOAT32, {rows, heads, kDecodePVParts, valueDim});
        auto *scores = (float *)scratch.attentionScores.cudaData;
        auto *partial = (float *)scratch.attentionPartial.cudaData;
        auto *selected = (const int *)indices.cudaData;
        auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        DecodeKeys keys{(const int *)liveKeys.cudaData, 0};
        AttentionScoresDecodeGrouped<4, kDecodeQkDim, 32, DecodeKeys, true>
            <<<dim3(heads / 4, count / 32, rows), 128>>>((const BF16 *)query.cudaData,
                (const BF16 *)key.cudaData, selected, scores, heads, kvHeads, dim,
                key.dims[2], keys, count, capacity - 1, true);
        AttentionSoftmax<<<rows * heads, 256>>>(scores, bias, heads, count);
        AttentionValuesDecodeGrouped<4, DecodeKeys, true>
            <<<dim3(heads / 4, kDecodePVParts, rows), kDecodePVValueDim>>>(scores,
                (const BF16 *)value.cudaData, selected, partial, heads, kvHeads, keys, capacity - 1, true);
        AttentionValuesDecodeReduce<<<dim3((heads * valueDim + 255) / 256, rows), 256>>>(
            partial, (BF16 *)output.cudaData, heads);
        CheckLaunch();
        return;
    }
    if (window) {
        Output(scratch.windowKey, BFLOAT16, {1, window, key.dims[2]});
        Output(scratch.windowValue, BFLOAT16, {1, window, value.dims[2]});
    }
    for (int row = 0; row < rows; ++row) {
        Data q(BFLOAT16, {1, 1, heads * dim}), live(INT32, {1});
        Data out(BFLOAT16, {1, 1, heads * valueDim}), selected;
        q.FakeFrom(query, (size_t)row * heads * dim * sizeof(BF16));
        live.FakeFrom(liveKeys, row * sizeof(int));
        out.FakeFrom(output, (size_t)row * heads * valueDim * sizeof(BF16));
        if (!indices.dims.empty()) {
            selected.Resize({1, 2048});
            selected.FakeFrom(indices, (size_t)row * 2048 * sizeof(int));
        }
        if (window) {
            int kc = key.dims[2], vc = value.dims[2];
            CopyVerifyWindow<<<(window * (kc + vc) + 255) / 256, 256>>>(
                (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
                (BF16 *)scratch.windowKey.cudaData, (BF16 *)scratch.windowValue.cudaData,
                (const int *)liveKeys.cudaData, row, kc, vc, window);
        }
        FastllmCudaNaiveDecodeAttention(q, window ? scratch.windowKey : key,
            window ? scratch.windowValue : value, selected, sink, live, capacity,
            heads, kvHeads, dim, valueDim, window, scratch, out);
    }
}
