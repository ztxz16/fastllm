#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
#include "naive-n05-topk.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-dsa-decode.cuh"
#include "naive-n05-attention-hopper.cuh"
#include "naive-n05-prefill.cuh"
#include "naive-n05-prefill-hopper.cuh"
#ifdef FASTLLM_ENABLE_NAIVE_QKV_SM90
#include "naive-n05-qkv.cuh"
#endif
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM90
#include "naive-n05-indexer.cuh"
#endif
#include <cuda_bf16.h>
#include <nccl.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <algorithm>
#include <cmath>
#include <type_traits>
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
#include "naive-n05-indexer-batch.cuh"
#endif

namespace {
using BF16 = __nv_bfloat16;
template <class Kernel> bool BulkKernelSupported(Kernel kernel, size_t shared, int minArch);
constexpr int kSwaWindow = 128;
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

// Preserve the 256-thread FP32 GEMV accumulation and ascending reduction
// tree. Loading all 16 terms before arithmetic hides their memory latency.
__global__ void RouterSigmoid4096(const BF16 *input, const float *weight,
                                  float *output) {
    int t = threadIdx.x, expert = blockIdx.x;
    __shared__ float sums[8];
    float values[16], weights[16], partial = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = (float)input[t + i * 256];
        weights[i] = weight[(size_t)expert * 4096 + t + i * 256];
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) partial += values[i] * weights[i];
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2)
        partial += __shfl_down_sync(0xffffffffu, partial, stride);
    if (t % 32 == 0) sums[t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        partial = t < 8 ? sums[t] : 0;
        #pragma unroll
        for (int stride = 1; stride < 8; stride *= 2)
            partial += __shfl_down_sync(0xffffffffu, partial, stride);
        if (t == 0) {
            float logit = partial + 0.0f;
            // Match standalone FP32 sigmoid's double-precision division.
            output[expert] = 1.0 / (1.0 + expf(-logit));
        }
    }
}

// Reuse each expert's weights across decode/verify rows with exact GEMV
// accumulation. Each expert applies sigmoid before the batched TopK launch.
template <int Rows>
__global__ void BatchRouterSigmoid4096(const BF16 *input, const float *weight,
        float *output, int rows, int experts) {
    const int t = threadIdx.x, first = blockIdx.y * Rows, expert = blockIdx.x;
    __shared__ float sums[Rows][8];
    float weights[16], values[Rows][16], partial[Rows]{};
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        weights[i] = weight[(size_t)expert * 4096 + t + i * 256];
        #pragma unroll
        for (int row = 0; row < Rows; ++row)
            values[row][i] = first + row < rows
                ? (float)input[(size_t)(first + row) * 4096 + t + i * 256] : 0;
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        #pragma unroll
        for (int row = 0; row < Rows; ++row) partial[row] += values[row][i] * weights[i];
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        #pragma unroll
        for (int stride = 1; stride < 32; stride *= 2)
            partial[row] += __shfl_down_sync(0xffffffffu, partial[row], stride);
        if (t % 32 == 0) sums[row][t / 32] = partial[row];
    }
    __syncthreads();
    // Independent warps finish different rows, including the expensive sigmoid.
    // Keep the reference GEMV's reduction and zero-bias addition exactly.
    const int row = t / 32, lane = t % 32;
    if (row < Rows) {
        float value = lane < 8 ? sums[row][lane] : 0;
        #pragma unroll
        for (int stride = 1; stride < 8; stride *= 2)
            value += __shfl_down_sync(0xffffffffu, value, stride);
        if (lane == 0 && first + row < rows)
            output[(size_t)(first + row) * experts + expert] =
                1.0 / (1.0 + expf(-(value + 0.0f)));
    }
}

// Vector loads pair adjacent lanes of the original 256-thread GEMV. Retain
// each lane's 16-term accumulation, then combine the first shuffle stage
// locally. Four rows share each weight; the remaining tree and sigmoid are exact.
__global__ void BatchRouterSigmoid4096Vector(const BF16 *input, const float *weight,
        float *output, int rows, int experts) {
    const int t = threadIdx.x, first = blockIdx.y * 4, expert = blockIdx.x;
    __shared__ float sums[4][8];
    float2 weights[16];
    float values[4][16][2], partial[4][2]{};
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int column = t * 2 + i * 256;
        weights[i] = *(const float2 *)(weight + (size_t)expert * 4096 + column);
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            __align__(4) BF16 fragment[2];
            *(unsigned *)fragment = first + row < rows
                ? *(const unsigned *)(input + (size_t)(first + row) * 4096 + column) : 0;
            values[row][i][0] = (float)fragment[0];
            values[row][i][1] = (float)fragment[1];
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            partial[row][0] += values[row][i][0] * weights[i].x;
            partial[row][1] += values[row][i][1] * weights[i].y;
        }
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        float total = partial[row][0] + partial[row][1];
        #pragma unroll
        for (int stride = 1; stride < 16; stride *= 2)
            total += __shfl_down_sync(0xffffffffu, total, stride);
        if (t % 16 == 0) sums[row][t / 16] = total;
    }
    __syncthreads();
    const int row = t / 32, lane = t % 32;
    float total = lane < 8 ? sums[row][lane] : 0;
    #pragma unroll
    for (int stride = 1; stride < 8; stride *= 2)
        total += __shfl_down_sync(0xffffffffu, total, stride);
    if (lane == 0 && first + row < rows)
        output[(size_t)(first + row) * experts + expert] =
            1.0 / (1.0 + expf(-(total + 0.0f)));
}

constexpr int kDraftKVLayersPerLaunch = 16;
struct DraftKVWeights { const void *data[kDraftKVLayersPerLaunch]; };
__global__ void DraftKVPointerTable(DraftKVWeights weights, const void *input,
        BF16 *output, void **pointers, int first, int count, int layers, size_t stride) {
    int i = threadIdx.x;
    if (i < count) {
        const int layer = first + i;
        pointers[layer] = const_cast<void *>(weights.data[i]);
        pointers[layers + layer] = const_cast<void *>(input);
        pointers[2 * layers + layer] = output + (size_t)layer * stride;
    }
}
struct DraftKVDestinations {
    const BF16 *raw[kDraftKVLayersPerLaunch];
    const float *norm[kDraftKVLayersPerLaunch];
    BF16 *key[kDraftKVLayersPerLaunch], *value[kDraftKVLayersPerLaunch];
    int length[kDraftKVLayersPerLaunch];
};
__global__ void DraftNormRopeWrite(DraftKVDestinations dst, int rows,
        int heads, int dim, int start, float eps, float theta) {
    int head = blockIdx.x % heads, row = (blockIdx.x / heads) % rows;
    int slot = blockIdx.x / (heads * rows);
    int t = threadIdx.x, width = heads * dim;
    const BF16 *key = dst.raw[slot] + (size_t)row * 2 * width + head * dim;
    __shared__ float sums[8];
    __shared__ BF16 normalized[256];
    float value = t < dim ? (float)key[t] : 0;
    float partial = WarpSum(value * value);
    if ((t & 31) == 0) sums[t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        float total = WarpSum(t < 8 ? sums[t] : 0);
        if (t == 0) sums[0] = total;
    }
    __syncthreads();
    // Match KimiK3RMSNorm: BF16 normalization, FP32 weight, then BF16.
    if (t < dim)
        normalized[t] = __float2bfloat16(RoundBF16(value * rsqrtf(sums[0] / dim + eps)) * dst.norm[slot][t]);
    __syncthreads();
    size_t offset = (size_t)(dst.length[slot] + row) * width + head * dim;
    if (t < dim / 2) {
        float angle = (float)(start + row) * powf(theta, -2.0f * t / dim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)normalized[t], b = (float)normalized[t + dim / 2];
        dst.key[slot][offset + t] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        dst.key[slot][offset + t + dim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
    if (t < dim) dst.value[slot][offset + t] = key[width + t];
}
void Output(fastllm::Data &out, fastllm::DataType type, const std::vector<int> &dims) {
    out.dataType = type;
    out.Resize(dims);
    out.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    out.Allocate(false);
}
// Return zero without touching output when the router fast path is unsupported.
int RouterRows(const fastllm::Data &input, const fastllm::Data &weight,
               const fastllm::Data &output) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    auto onDevice = [&](const Data &x) {
        return x.dataDevice == DataDevice::CUDA && x.dataDeviceIds.size() == 1 &&
            x.dataDeviceIds[0] == device && x.cudaData;
    };
    if (&input == &output || &weight == &output ||
        (output.cudaData && (output.cudaData == input.cudaData || output.cudaData == weight.cudaData)) ||
        !onDevice(input) || !onDevice(weight) || input.dataType != BFLOAT16 ||
        weight.dataType != FLOAT32 || input.dims.empty() || input.dims.back() != 4096 ||
        weight.dims.size() != 2 || weight.dims[1] != 4096 || weight.dims[0] <= 0 ||
        weight.strides.size() != 2 || weight.strides[0] != 4096 || weight.strides[1] != 1 ||
        input.strides.size() != input.dims.size()) return 0;
    int64_t count = 1;
    for (int axis = (int)input.dims.size() - 1; axis >= 0; --axis) {
        if (input.strides[axis] != count || input.dims[axis] <= 0) return 0;
        count *= input.dims[axis];
        if (count > 8 * 4096) return 0;
    }
    return count / 4096;
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
template <bool ToRing, typename T>
__global__ void SwaCacheLayout(T *key, T *value, int kc, int vc, int count, int first) {
    constexpr int columns = kTrimTileBytes / sizeof(T);
    __shared__ T rows[kSwaWindow * columns];
    int keyBlocks = (kc + columns - 1) / columns;
    bool isKey = blockIdx.x < keyBlocks;
    T *data = isKey ? key : value;
    int stride = isKey ? kc : vc;
    int column = (isKey ? blockIdx.x : blockIdx.x - keyBlocks) * columns;
    for (int i = threadIdx.x; i < count * columns; i += blockDim.x) {
        int row = i / columns, col = column + i % columns;
        int source = ToRing ? row : ((first + row) & (kSwaWindow - 1));
        if (col < stride) rows[i] = data[(size_t)source * stride + col];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count * columns; i += blockDim.x) {
        int row = i / columns, col = column + i % columns;
        int dest = ToRing ? ((first + row) & (kSwaWindow - 1)) : row;
        if (col < stride) data[(size_t)dest * stride + col] = rows[i];
    }
}

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

// AllGather writes [front_0, back_0, front_1, back_1, ...]. Each
// thread owns one column across all tiles, so the permutation is in-place.
template<int Ranks>
__global__ void RestoreBalancedRows(int *indices, int tileElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tileElements) return;
    int values[2 * Ranks];
    #pragma unroll
    for (int tile = 0; tile < 2 * Ranks; ++tile)
        values[tile] = indices[(size_t)tile * tileElements + i];
    #pragma unroll
    for (int rank = 0; rank < Ranks; ++rank) {
        indices[(size_t)rank * tileElements + i] = values[2 * rank];
        indices[(size_t)(2 * Ranks - 1 - rank) * tileElements + i] = values[2 * rank + 1];
    }
}

// Hopper benefits from fewer radix passes; retain CUB's original setting
// on other architectures where the larger digit's resource cost is unmeasured.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
constexpr int kCompactTopKRadixBits = 6;
#else
constexpr int kCompactTopKRadixBits = 4;
#endif
template<int Items>
using CompactTopKSort = cub::BlockRadixSort<unsigned long long, 256, Items,
    cub::NullType, kCompactTopKRadixBits>;

// Keep all score bits but compress the inverse index to its actual width.
// Skipping constant index bits reduces radix passes without changing score
// order or lower-index tie breaking, including signed zero and NaNs.
template<int Items>
__device__ void SortTopKCompactRow(const unsigned long long *input, const int *ends,
        int *indices, int queries, int stride, int topK, int keyCount,
        typename CompactTopKSort<Items>::TempStorage &storage) {
    constexpr int Threads = 256;
    using Sort = CompactTopKSort<Items>;
    const int row = blockIdx.x, count = ends[queries + row] - row * stride;
    const int indexBits = 32 - __clz((unsigned)max(1, keyCount - 1));
    const unsigned mask = 0xffffffffu >> (32 - indexBits);
    unsigned long long keys[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int col = threadIdx.x + i * Threads;
        unsigned long long key = col < count ? input[(size_t)row * stride + col] : 0;
        unsigned index = 0xffffffffu - (unsigned)key;
        keys[i] = col < count ? ((key >> 32) << indexBits) | (mask - index) : 0;
    }
    Sort(storage).SortDescendingBlockedToStriped(keys, 0, 32 + indexBits);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int col = threadIdx.x + i * Threads;
        if (col < topK)
            indices[(size_t)row * topK + col] = col < count
                ? (int)(mask - ((unsigned)keys[i] & mask)) : -1;
    }
}

// Candidate selection still has 4096 slots. Sort only the populated range;
// shrinking selection itself would add radix passes on long or tied rows.
__global__ void SortTopKCompact(const unsigned long long *input, const int *ends,
        int *indices, int queries, int stride, int topK, int keyCount) {
    union Storage {
        CompactTopKSort<8>::TempStorage small;
        CompactTopKSort<12>::TempStorage medium;
        CompactTopKSort<16>::TempStorage large;
    };
    __shared__ Storage storage;
    const int count = ends[queries + blockIdx.x] - blockIdx.x * stride;
    if (count <= 2048)
        SortTopKCompactRow<8>(input, ends, indices, queries, stride, topK, keyCount, storage.small);
    else if (count <= 3072)
        SortTopKCompactRow<12>(input, ends, indices, queries, stride, topK, keyCount, storage.medium);
    else
        SortTopKCompactRow<16>(input, ends, indices, queries, stride, topK, keyCount, storage.large);
}

template <typename Output>
__device__ void RopeHeadTo(const BF16 *x, Output out, float position,
                           int rotaryDim, float theta) {
    for (int d = threadIdx.x; d < rotaryDim / 2; d += blockDim.x) {
        float angle = position * powf(theta, -2.0f * d / rotaryDim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)x[d], b = (float)x[d + rotaryDim / 2];
        // Match eager GPT-NeoX RoPE, including each BF16 multiplication.
        out[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        out[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
}

__device__ void RopeHead(BF16 *x, float position, int rotaryDim, float theta) {
    RopeHeadTo(x, x, position, rotaryDim, theta);
}

__global__ void Rope(BF16 *data, const float *positions, int heads, int dim,
                     int rotaryDim, float theta) {
    int row = blockIdx.x;
    RopeHead(data + (size_t)row * dim, positions[row / heads], rotaryDim, theta);
}

// Share RoPE frequencies across heads when prefill has enough independent tokens.
__global__ void RopeToken(BF16 *data, const float *positions, int heads, int dim,
                          int rotaryDim, float theta) {
    __shared__ float cosine[128], sine[128];
    const int token = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    if (t < rotaryDim / 2) {
        float angle = positions[token] * powf(theta, -2.0f * t / rotaryDim);
        cosine[t] = RoundBF16(cosf(angle));
        sine[t] = RoundBF16(sinf(angle));
    }
    __syncthreads();
    for (int head = warp; head < heads; head += 8) {
        BF16 *x = data + ((size_t)token * heads + head) * dim;
        for (int d = lane; d < rotaryDim / 2; d += 32) {
            float a = (float)x[d], b = (float)x[d + rotaryDim / 2];
            float c = cosine[d], s = sine[d];
            x[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
            x[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
        }
    }
}

// Each CTA owns one head. Packed projection input is unpacked while rotating
// Q/K and scaling V, without intermediate Split kernels.
template <bool Packed>
__global__ void RopeQKScaleV(BF16 *q, BF16 *k, BF16 *v, const float *positions,
                            int heads, int kvHeads, int dim, int valueDim,
                            int rotaryDim, float theta, BF16 valueScale, const BF16 *qkv) {
    int totalHeads = heads + 2 * kvHeads;
    int token = blockIdx.x / totalHeads, head = blockIdx.x % totalHeads;
    const size_t base = (size_t)token * ((heads + kvHeads) * dim + kvHeads * valueDim);
    if (head < heads + kvHeads) {
        BF16 *x = head < heads ? q + ((size_t)token * heads + head) * dim
            : k + ((size_t)token * kvHeads + head - heads) * dim;
        const BF16 *src = Packed ? qkv + base + head * dim : x;
        RopeHeadTo(src, x, positions[token], rotaryDim, theta);
        if constexpr (Packed)
            for (int d = rotaryDim + threadIdx.x; d < dim; d += blockDim.x) x[d] = src[d];
    } else {
        BF16 *x = v + ((size_t)token * kvHeads + head - heads - kvHeads) * valueDim;
        const BF16 *src = Packed ? qkv + base + (heads + kvHeads) * dim
            + (head - heads - kvHeads) * valueDim : x;
        for (int d = threadIdx.x; d < valueDim; d += blockDim.x)
            x[d] = __float2bfloat16_rn((float)src[d] * (float)valueScale);
    }
}

// Share the position-dependent rotation across Q/K heads of a token.
// Each warp owns whole pairs, including for in-place non-packed input.
template <bool Packed>
__global__ void RopeQKScaleVToken(BF16 *q, BF16 *k, BF16 *v, const float *positions,
        int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, BF16 valueScale, const BF16 *qkv) {
    __shared__ float cosine[128], sine[128];
    int token = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    if (t < rotaryDim / 2) {
        float angle = positions[token] * powf(theta, -2.0f * t / rotaryDim);
        cosine[t] = RoundBF16(cosf(angle));
        sine[t] = RoundBF16(sinf(angle));
    }
    __syncthreads();
    const size_t base = (size_t)token * ((heads + kvHeads) * dim + kvHeads * valueDim);
    for (int head = warp; head < heads + 2 * kvHeads; head += 8) {
        if (head < heads + kvHeads) {
            BF16 *x = head < heads ? q + ((size_t)token * heads + head) * dim
                : k + ((size_t)token * kvHeads + head - heads) * dim;
            const BF16 *src = Packed ? qkv + base + head * dim : x;
            for (int d = lane; d < rotaryDim / 2; d += 32) {
                float a = (float)src[d], b = (float)src[d + rotaryDim / 2];
                float c = cosine[d], s = sine[d];
                x[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
                x[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
            }
            if constexpr (Packed)
                for (int d = rotaryDim / 2 + lane; d < dim / 2; d += 32)
                    ((__nv_bfloat162 *)x)[d] = ((const __nv_bfloat162 *)src)[d];
        } else {
            BF16 *x = v + ((size_t)token * kvHeads + head - heads - kvHeads) * valueDim;
            const BF16 *src = Packed ? qkv + base + (heads + kvHeads) * dim
                + (head - heads - kvHeads) * valueDim : x;
            for (int d = lane; d < valueDim / 2; d += 32)
                ((__nv_bfloat162 *)x)[d] = __hmul2(((const __nv_bfloat162 *)src)[d],
                    __halves2bfloat162(valueScale, valueScale));
        }
    }
}

struct RopeCacheRow {
    BF16 *q, *key, *value;
    const BF16 *k, *v, *indexKey, *qkv;
    const float *positions;
    const int *length;
    int keyRows, valueRows, cachePast;
    const void *const *cache = nullptr;
    const int *keyPages = nullptr, *valuePages = nullptr;
    int pageLen = 0;
    bool fp8 = false, scaledIndexer = false;
};
constexpr int kRopeCacheBatch = 32;
struct RopeCacheBatch { RopeCacheRow rows[kRopeCacheBatch]; };

template <bool Packed>
__device__ void AppendRopeCacheRow(RopeCacheRow r, int token, int head,
        int heads, int kvHeads, int dim, int valueDim, int indexDim, int rotaryDim,
        float theta, BF16 valueScale, int window, bool ring) {
    BF16 *q = r.q;
    FastllmNaiveKVWriteRow key{r.cache ? (void *)r.cache[0] : r.key, r.fp8},
        value{r.cache ? (void *)r.cache[1] : r.value, r.fp8};
    const BF16 *k = r.k, *v = r.v, *indexKey = r.indexKey, *qkv = r.qkv;
    const float *positions = r.positions;
    const int *length = r.length;
    const int keyRows = r.keyRows, valueRows = r.valueRows, cachePast = r.cachePast;
    const int past = length ? (window ? min(*length - 1, window - 1) : *length - 1) : cachePast;
    const int row = ring ? ((*length - 1) & (kSwaWindow - 1)) : past + token;
    if (past < 0) {
        if (head < heads)
            for (int d = threadIdx.x; d < dim; d += blockDim.x)
                q[((size_t)token * heads + head) * dim + d] = __float2bfloat16(0.f);
        return;
    }
    if (row >= keyRows || row >= valueRows) return;
    const size_t base = (size_t)token * ((heads + kvHeads) * dim + kvHeads * valueDim);
    if (head < heads) {
        BF16 *dst = q + ((size_t)token * heads + head) * dim;
        const BF16 *src = Packed ? qkv + base + head * dim : dst;
        RopeHeadTo(src, dst, positions[token], rotaryDim, theta);
        if constexpr (Packed)
            for (int d = rotaryDim + threadIdx.x; d < dim; d += blockDim.x) dst[d] = src[d];
    } else if (head < heads + kvHeads || head == heads + 2 * kvHeads) {
        const bool index = head == heads + 2 * kvHeads;
        const int width = index ? indexDim : dim;
        const BF16 *src = index ? indexKey + (size_t)token * indexDim
            : (Packed ? qkv + base + head * dim
                      : k + ((size_t)token * kvHeads + head - heads) * dim);
        const int *pages = r.pageLen && r.cache ? (const int *)r.cache[2] : r.keyPages;
        int physical = pages ? pages[row / r.pageLen] * r.pageLen + row % r.pageLen : row;
        const int tail = r.scaledIndexer ? FastllmNaiveIndexerBytes / (r.fp8 ? 1 : 2) : indexDim;
        auto dst = key + (size_t)physical * (kvHeads * dim + tail)
            + (index ? kvHeads * dim : (head - heads) * dim);
        if (index && r.scaledIndexer) {
            __shared__ BF16 rotated[128];
            RopeHeadTo(src, rotated, positions[token], rotaryDim, theta);
            for (int d = rotaryDim + threadIdx.x; d < width; d += blockDim.x) rotated[d] = src[d];
            __syncthreads();
            FastllmNaiveStoreIndexer((float)rotated[threadIdx.x], dst.data);
        } else {
            RopeHeadTo(src, dst, positions[token], rotaryDim, theta);
            for (int d = rotaryDim + threadIdx.x; d < width; d += blockDim.x) dst[d] = src[d];
        }
    } else {
        const int h = head - heads - kvHeads;
        const BF16 *src = Packed ? qkv + base + (heads + kvHeads) * dim + h * valueDim
                                : v + ((size_t)token * kvHeads + h) * valueDim;
        const int *pages = r.pageLen && r.cache ? (const int *)r.cache[3] : r.valuePages;
        int physical = pages ? pages[row / r.pageLen] * r.pageLen + row % r.pageLen : row;
        auto dst = value + ((size_t)physical * kvHeads + h) * valueDim;
        for (int d = threadIdx.x; d < valueDim; d += blockDim.x)
            dst[d] = __float2bfloat16_rn((float)src[d] * (float)valueScale);
    }
}

template <bool Packed>
__global__ void RopeAppendCache(RopeCacheRow row, int heads, int kvHeads, int dim,
        int valueDim, int indexDim, int rotaryDim, float theta, BF16 valueScale,
        int window, bool ring) {
    const int totalHeads = heads + 2 * kvHeads + (indexDim != 0);
    AppendRopeCacheRow<Packed>(row, blockIdx.x / totalHeads, blockIdx.x % totalHeads,
        heads, kvHeads, dim, valueDim, indexDim, rotaryDim, theta, valueScale, window, ring);
}

template <bool Packed>
__global__ void BatchRopeAppendCache(RopeCacheBatch batch, int heads, int kvHeads, int dim,
        int valueDim, int indexDim, int rotaryDim, float theta, BF16 valueScale,
        int window, bool ring) {
    AppendRopeCacheRow<Packed>(batch.rows[blockIdx.y], 0, blockIdx.x, heads, kvHeads,
        dim, valueDim, indexDim, rotaryDim, theta, valueScale, window, ring);
}

// Same 64-lane FP32 tree and epsilon as LayerNorm(FP32), with BF16 I/O.
__global__ void IndexerKeyNorm(BF16 *input, const float *weight, const float *bias) {
    const int t = threadIdx.x;
    input += blockIdx.x * 128;
    __shared__ float sum[64], square[64], mean, deviation;
    float total = 0, totalSquare = 0;
    for (int i = t; i < 128; i += 64) {
        float x = (float)input[i];
        total += x;
        totalSquare += x * x;
    }
    sum[t] = total; square[t] = totalSquare;
    __syncthreads();
    for (int stride = 32; stride; stride >>= 1) {
        if (t < stride) {
            sum[t] += sum[t + stride];
            square[t] += square[t + stride];
        }
        __syncthreads();
    }
    if (!t) {
        mean = sum[0] / 128;
        deviation = square[0] + mean * mean * 128 - 2 * mean * 128 * mean;
        deviation = sqrt(deviation / 128 + 1e-10);
    }
    __syncthreads();
    for (int i = t; i < 128; i += 64)
        input[i] = __float2bfloat16_rn(((float)input[i] - mean) / deviation * weight[i] + bias[i]);
}

__global__ void RoundIndexer(FastllmNaiveKV input, float *output, int stride,
                             int offset, bool fp8) {
    __shared__ float maximum[128];
    int d = threadIdx.x, row = blockIdx.x;
    float x = input.scaledIndexer
        ? input.IndexRow(row, stride).LoadFloat(d) * input.IndexScale(row, stride)
        : (float)input.Row(row, stride)[offset + d];
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
__global__ void IndexScoresDecode(const float *q, FastllmNaiveKV packedKeys,
        const BF16 *weights, ScoreOutput scores, int stride, Length liveKeys, int queryStart,
        const void *const *cache = nullptr) {
    packedKeys.Bind(cache);
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
        k[i] = packedKeys.IndexRow(key, stride).LoadFloat(lane + i * 32) * packedKeys.IndexScale(key, stride);
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
__global__ void AttentionScores(const BF16 *q, FastllmNaiveKV k, const int *indices,
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
                       (float)k.Row(key, keyStride)[kvHead * dim + d];
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
__global__ void AttentionScoresDecodeGrouped(const BF16 *q, FastllmNaiveKV k,
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
                float value = (float)k.Row(key, keyStride)[kvHead * dim + lane + j * 32];
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
__global__ void AttentionScoresDecodeShared(const BF16 *q, FastllmNaiveKV k,
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
            ? (k.Row(key, keyStride) + kvHead * dim + d).Load8()
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
__global__ void AttentionScoresPrefill(const BF16 *q, FastllmNaiveKV k, const int *indices,
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
                dot[i] += queryValues[j] * (float)k.Row(key[i], keyStride)[kvHead * dim + lane + j * 32];
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
__global__ void AttentionScoresPrefillGrouped(const BF16 *q, FastllmNaiveKV k, const int *indices,
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
            float keyValue = (float)k.Row(key, keyStride)[kvHead * dim + lane + j * 32];
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
template <typename Length = int, typename Past = int, bool Paged = false>
__global__ void AttentionValuesPrefill(const float *prob, FastllmNaiveKV v, const int *indices,
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
        int mappedKey = -1;
        for (int slot = 0; slot < count; ++slot) {
            int key;
            bool valid;
            if constexpr (Paged) {
                // Resolve 32 page lookups together, then broadcast them in
                // original slot order. Avoid a dependent page-table load
                // before each serial V load without adding global scratch.
                if ((slot & 31) == 0) {
                    int logical = slot + lane < count
                        ? KeyIndex(indices, query, slot + lane, count, past, window) : -1;
                    mappedKey = logical >= 0 && logical < keys && (!causal || logical <= past + query)
                        ? v.PhysicalRow(logical) : -1;
                }
                key = __shfl_sync(0xffffffff, mappedKey, slot & 31);
                valid = key >= 0;
            } else {
                key = KeyIndex(indices, query, slot, count, past, window);
                valid = key >= 0 && key < keys && (!causal || key <= past + query);
            }
            if (valid) {
                FastllmNaiveKVRow row;
                if constexpr (Paged) row = FastllmNaiveKVRow(v.base, v.fp8) + (size_t)key * kvHeads * dim + v.offset;
                else row = v.Row(key, kvHeads * dim);
                uint2 bits = (row + kvHead * dim + d).Load4();
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
__global__ void AttentionShort(const BF16 *q, FastllmNaiveKV k, FastllmNaiveKV v,
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
                   (float)k.Row(key, keyStride)[kvHead * dim + d];
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
                value += scores[slot] * (float)v.Row(key, kvHeads * valueDim)[kvHead * valueDim + d];
        }
        out[((size_t)query * heads + h) * valueDim + d] = __float2bfloat16(value);
    }
}

constexpr int kSwaQkDim = 192;
constexpr int kSwaValueDim = 128;
constexpr int kSwaOutputTile = 32;
constexpr int kSwaThreads = 256;
constexpr int kSwaBatchLimit = 16;
struct BatchSwaKeys {
    FastllmNaiveSWARow rows[kSwaBatchLimit];
    const int *length;
    __device__ operator int() const { return min(length[blockIdx.y], kSwaWindow); }
};

// Single-query attention with head dimensions 192/128 and up to 256 keys. Four
// output slices spread the work across SMs; cooperative V loads avoid the
// reference kernel's dependent global load for every output/slot pair.
// The softmax tree, BF16 rounding and slot-ordered FP32 FMAs are unchanged.
template <int MaxKeys, typename Length = int, bool Verify = false, int Threads = kSwaThreads, int OutputTile = kSwaOutputTile, bool Ring = false>
__global__ void AttentionShortDecode(const BF16 *q, FastllmNaiveKV k, FastllmNaiveKV v,
        const float *sink, BF16 *out, int heads, int kvHeads, int keyStride, const Length liveKeys) {
    int keys = liveKeys;
    if constexpr (Verify) keys = min(VerifyCount(liveKeys, blockIdx.y), MaxKeys);
    // Handle inactive rows before binding KV or calculating the active-row
    // offsets, so padding does not extend their register lifetimes.
    if (keys <= 0) {
        const int row = (Verify || std::is_same<Length, BatchSwaKeys>::value) ? blockIdx.y : 0;
        if (threadIdx.x < OutputTile)
            out[((size_t)row * heads + blockIdx.x) * kSwaValueDim +
                blockIdx.z * OutputTile + threadIdx.x] = __float2bfloat16(0.f);
        return;
    }
    if constexpr (std::is_same<Length, BatchSwaKeys>::value) {
        const int row = blockIdx.y;
        q += (size_t)row * heads * kSwaQkDim;
        out += (size_t)row * heads * kSwaValueDim;
        const auto &cache = liveKeys.rows[row];
        k = FastllmNaiveKV(cache.cache ? cache.cache[0] : cache.key, nullptr, 0, cache.fp8);
        v = FastllmNaiveKV(cache.cache ? cache.cache[1] : cache.value, nullptr, 0, cache.fp8);
    }
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
    int firstKey = 0;
    if constexpr (Ring) {
        if constexpr (std::is_same<Length, BatchSwaKeys>::value)
            firstKey = max(0, liveKeys.length[blockIdx.y] - kSwaWindow);
        else firstKey = max(0, *liveKeys.length - kSwaWindow);
    }
    __shared__ float scores[MaxKeys], scratch[kSwaThreads], maximum, denominator;
    __shared__ __align__(16) BF16 values[MaxKeys][OutputTile];
    int h = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    int kvHead = h / (heads / kvHeads), firstDim = blockIdx.z * OutputTile;
    if (v.fp8) {
        for (int i = t; i < keys * (OutputTile / 8); i += Threads) {
            int row = i / (OutputTile / 8), col = i % (OutputTile / 8) * 8;
            const int physical = Ring ? ((firstKey + row) & (kSwaWindow - 1)) : row;
            *(uint4 *)&values[row][col] = (v.Row(physical, kvHeads * kSwaValueDim) +
                kvHead * kSwaValueDim + firstDim + col).Load8();
        }
    } else {
        for (int i = t; i < keys * OutputTile; i += Threads) {
            int row = i / OutputTile, col = i % OutputTile;
            const int physical = Ring ? ((firstKey + row) & (kSwaWindow - 1)) : row;
            values[row][col] = v.Row(physical, kvHeads * kSwaValueDim)[kvHead * kSwaValueDim + firstDim + col];
        }
    }
    float query[kSwaQkDim / 32];
    #pragma unroll
    for (int i = 0; i < kSwaQkDim / 32; ++i) query[i] = (float)q[(size_t)h * kSwaQkDim + lane + i * 32];
    for (int slot = warp; slot < keys; slot += Threads / 32) {
        const int physical = Ring ? ((firstKey + slot) & (kSwaWindow - 1)) : slot;
        float dot = 0;
        #pragma unroll
        for (int i = 0; i < kSwaQkDim / 32; ++i)
            dot += query[i] * k.Row(physical, keyStride).LoadFloat(kvHead * kSwaQkDim + lane + i * 32);
        dot = WarpSum(dot);
        if (lane == 0) scores[slot] = RoundBF16(RoundBF16(dot) * rsqrtf((float)kSwaQkDim));
    }
    __syncthreads();
    float bias = sink ? sink[h] : -INFINITY;
    if (t < kSwaThreads) scratch[t] = t < keys ? fmaxf(bias, scores[t]) : bias;
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
    if (t < kSwaThreads) scratch[t] = sum;
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
    if (t < OutputTile) {
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

// Dispatch by workload shape, independent of TP rank count. All variants retain
// the same QK/softmax/PV arithmetic; only the work assigned to each CTA changes.
template <bool Verify>
void LaunchShortAttention(const BF16 *q, FastllmNaiveKV k, FastllmNaiveKV v,
        const float *bias, BF16 *out, int rows, int heads, int kvHeads,
        int keyStride, DecodeKeys keys, bool sliding, bool ring = false) {
    int threads = kSwaThreads, tile = kSwaOutputTile;
    auto kernel = sliding ? AttentionShortDecode<kSwaWindow, DecodeKeys, Verify>
                          : AttentionShortDecode<256, DecodeKeys, Verify>;
    if (heads == 8 || heads == 16) {
        if constexpr (Verify) {
            if (rows == 8) {
                tile = 64;
                if (heads == 8) {
                    threads = 1024;
                    kernel = sliding ? AttentionShortDecode<kSwaWindow, DecodeKeys, true, 1024, 64>
                                     : AttentionShortDecode<256, DecodeKeys, true, 1024, 64>;
                } else {
                    threads = 512;
                    kernel = sliding ? AttentionShortDecode<kSwaWindow, DecodeKeys, true, 512, 64>
                                     : AttentionShortDecode<256, DecodeKeys, true, 512, 64>;
                }
            }
        } else {
            threads = 1024;
            kernel = sliding ? AttentionShortDecode<kSwaWindow, DecodeKeys, false, 1024>
                             : AttentionShortDecode<256, DecodeKeys, false, 1024>;
        }
    }
    if constexpr (!Verify) {
        if (ring) kernel = threads == 1024
            ? AttentionShortDecode<kSwaWindow, DecodeKeys, false, 1024, kSwaOutputTile, true>
            : AttentionShortDecode<kSwaWindow, DecodeKeys, false, kSwaThreads, kSwaOutputTile, true>;
    }
    kernel<<<dim3(heads, rows, kSwaValueDim / tile), threads>>>(
        q, k, v, bias, out, heads, kvHeads, keyStride, keys);
}

__global__ void AttentionValues(const float *prob, FastllmNaiveKV v, const int *indices,
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
                sum += p[slot] * (float)v.Row(key, kvHeads * dim)[kvHead * dim + d];
        }
        out[((size_t)query * heads + h) * dim + d] = __float2bfloat16(sum);
    }
}

// Decode has just one query. Split its 2048 selected keys across CTAs so
// every thread accumulates one value column, then combine FP32 partial sums.
// Inputs/probability rounding and BF16 output are unchanged; the FP32 sum order
// differs from the serial fallback. Prefill and other shapes keep their paths.
template <typename Length = int>
__global__ void AttentionValuesDecodePartial(const float *prob, FastllmNaiveKV v,
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
            values[i] = valid[i] ? (float)v.Row(key, kvHeads * dim)[kvHead * dim + d] : 0;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (valid[i]) sum = fmaf(probabilities[i], values[i], sum);
    }
    partial[(h * parts + part) * dim + d] = sum;
}

template<int HeadTile, typename Length = int, bool Verify = false>
__global__ void AttentionValuesDecodeGrouped(const float *prob, FastllmNaiveKV v,
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
            values[i] = valid[i] ? (float)v.Row(key, kvHeads * dim)[kvHead * dim + d] : 0;
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
__global__ void AttentionValuesDecodeShared(const float *prob, FastllmNaiveKV v,
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
            ? (v.Row(key, kvHeads * dim) + kvHead * dim + firstColumn + d).Load8()
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
__global__ void AttentionValuesTiled(const float *prob, FastllmNaiveKV v,
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
                ? v.Row(key, kvHeads * dim)[kvHead * dim + col] : __float2bfloat16(0);
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

// Contiguous single-query KV needs neither an index tile nor a causal test
// per FMA. Narrow output tiles give small TP head counts more independent
// CTAs. Keep the original ascending-key FP32 accumulation and final BF16
// rounding; all staging is shared memory, with no global partial buffer.
template <int Columns>
__global__ void AttentionValuesDecodeDense(const float *prob, FastllmNaiveKV value,
        BF16 *output, const int *liveKeys, int capacity, int heads, int kvHeads) {
    constexpr int Dim = 128, Keys = 64;
    const int head = blockIdx.x, firstColumn = blockIdx.y * Columns;
    const int t = threadIdx.x, count = *liveKeys;
    const int validCount = min(count, capacity), kvHead = head / (heads / kvHeads);
    const float *p = prob + (size_t)head * count;
    __shared__ float probabilities[Keys];
    __shared__ BF16 values[Keys][Columns];
    float sum = 0.0f;
    for (int first = 0; first < validCount; first += Keys) {
        const int rows = min(Keys, validCount - first);
        if (t < rows) probabilities[t] = p[first + t];
        for (int i = t; i < rows * Columns; i += blockDim.x) {
            const int row = i / Columns, column = i % Columns;
            values[row][column] = value.Row(first + row, kvHeads * Dim)[kvHead * Dim + firstColumn + column];
        }
        __syncthreads();
        if (t < Columns) {
            for (int row = 0; row < rows; ++row)
                sum += probabilities[row] * (float)values[row][t];
        }
        __syncthreads();
    }
    if (t < Columns) output[(size_t)head * Dim + firstColumn + t] = __float2bfloat16(sum);
}
}

namespace {
// Q and K share the original 256-lane RMSNorm reduction and BF16 rounding.
// Each K CTA also writes its V head; separate CTAs own disjoint cache columns.
__global__ void DraftQKVNormRope(const BF16 *raw, const float *qNorm,
        const float *kNorm, const float *positions, const int *live,
        BF16 *key, BF16 *value, BF16 *query, int heads, int kvHeads,
        int dim, int window, int past, float eps, float theta) {
    const int head = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const bool isKey = head >= heads;
    const int h = isKey ? head - heads : head;
    const int qw = heads * dim, kw = kvHeads * dim, width = qw + 2 * kw;
    const BF16 *src = raw + (size_t)row * width + (isKey ? qw : 0) + h * dim;
    const float *norm = isKey ? kNorm : qNorm;
    __shared__ float sums[8];
    __shared__ BF16 normalized[256];
    float v = t < dim ? (float)src[t] : 0;
    float partial = WarpSum(v * v);
    if ((t & 31) == 0) sums[t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        float total = WarpSum(t < 8 ? sums[t] : 0);
        if (t == 0) sums[0] = total;
    }
    __syncthreads();
    if (t < dim) normalized[t] = __float2bfloat16(RoundBF16(v * rsqrtf(sums[0] / dim + eps)) * norm[t]);
    __syncthreads();
    int base = live ? min(*live - 1, window - 1) : past;
    BF16 *dst = isKey ? key + (size_t)(base + row) * kw + h * dim
                     : query + (size_t)row * qw + h * dim;
    if (t < dim / 2) {
        float angle = positions[row] * powf(theta, -2.0f * t / dim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)normalized[t], b = (float)normalized[t + dim / 2];
        dst[t] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        dst[t + dim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
    if (isKey && t < dim)
        value[(size_t)(base + row) * kw + h * dim + t] = src[kw + t];
}
__global__ void DraftRoundedSwiGLU(const BF16 *input, BF16 *output, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int at = (i / width) * 2 * width + i % width;
    float gate = (float)input[at], up = (float)input[at + width];
    output[i] = __float2bfloat16(RoundBF16(gate / (1.0f + expf(-gate))) * up);
}
bool DraftDense(const fastllm::Data &x, fastllm::DataType type) {
    if (x.dataType != type || x.dataDevice != fastllm::DataDevice::CUDA ||
        x.multiDeviceData || x.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
        !x.cudaData || x.dims.empty() || x.strides.size() != x.dims.size()) return false;
    uint64_t stride = 1;
    for (int i = (int)x.dims.size() - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || x.strides[i] != stride) return false;
        stride *= x.dims[i];
    }
    return true;
}
}

bool FastllmCudaNaiveDraftKVProject(const fastllm::Data &input,
        const std::vector<const fastllm::Data *> &weights,
        fastllm::Data &output, fastllm::Data &pointers) {
    using namespace fastllm;
    if (!DraftDense(input, BFLOAT16) || input.dims.size() != 3 || input.dims[0] != 1 ||
        weights.empty() || weights.size() > INT_MAX / (3 * sizeof(void *) / sizeof(int)) ||
        &output == &pointers || output.isFake || pointers.isFake ||
        output.multiDeviceData || pointers.multiDeviceData ||
        (output.cudaData && output.cudaData == pointers.cudaData)) return false;
    auto aliases = [&](const Data &x) {
        return &x == &output || &x == &pointers || (x.cudaData &&
            (x.cudaData == output.cudaData || x.cudaData == pointers.cudaData));
    };
    if (aliases(input)) return false;
    const int rows = input.dims[1], inner = input.dims[2], layers = weights.size();
    int columns = 0;
    for (const Data *weight : weights) {
        if (!weight || !DraftDense(*weight, BFLOAT16) || aliases(*weight) ||
            weight->dims.size() != 2 || weight->dims[1] != inner ||
            (columns && weight->dims[0] != columns)) return false;
        columns = weight->dims[0];
    }
    Output(output, BFLOAT16, {layers, rows, columns});
    Output(pointers, INT32, {layers * 3 * (int)(sizeof(void *) / sizeof(int))});
    auto table = (void **)pointers.cudaData;
    // Tiny pointer tables are constructed on the current stream. No pageable
    // H2D copy, global pointer cache, or weight-lifetime assumption is needed.
    for (int first = 0; first < layers; first += kDraftKVLayersPerLaunch) {
        DraftKVWeights addresses{};
        int count = std::min(kDraftKVLayersPerLaunch, layers - first);
        for (int i = 0; i < count; ++i) addresses.data[i] = weights[first + i]->cudaData;
        DraftKVPointerTable<<<1, 32, 0, cudaStreamPerThread>>>(addresses, input.cudaData,
            (BF16 *)output.cudaData, table, first, count, layers, (size_t)rows * columns);
    }
    CheckLaunch();
    const float alpha = 1, beta = 0;
    return cublasGemmBatchedEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
        columns, rows, inner, &alpha, (const void *const *)table, CUDA_R_16BF, inner,
        (const void *const *)(table + layers), CUDA_R_16BF, inner, &beta,
        (void *const *)(table + 2 * layers), CUDA_R_16BF, columns, layers,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
}

bool FastllmCudaNaiveDraftQKV(const fastllm::Data &raw,
        const fastllm::Data &qNorm, const fastllm::Data &kNorm,
        const fastllm::Data &positions, const fastllm::Data &liveKeys,
        fastllm::Data &key, fastllm::Data &value, fastllm::Data &query,
        int heads, int kvHeads, int dim, int window, float eps, float theta) {
    using namespace fastllm;
    if (heads <= 0 || kvHeads <= 0 || heads % kvHeads || dim <= 0 || dim > 256 || dim % 2 ||
        window <= 1 || !std::isfinite(eps) || eps < 0 || !std::isfinite(theta) || theta <= 0 ||
        (int64_t)(heads + 2LL * kvHeads) * dim > INT_MAX ||
        !DraftDense(raw, BFLOAT16) || raw.dims.size() != 3 || raw.dims[0] != 1 ||
        raw.dims[2] != (heads + 2 * kvHeads) * dim ||
        !DraftDense(qNorm, FLOAT32) || qNorm.dims != std::vector<int>{dim} ||
        !DraftDense(kNorm, FLOAT32) || kNorm.dims != std::vector<int>{dim} ||
        !DraftDense(positions, FLOAT32) || positions.Count(0) != raw.dims[1]) return false;
    const int rows = raw.dims[1], width = kvHeads * dim;
    const bool dynamic = !liveKeys.dims.empty();
    if (dynamic && (!DraftDense(liveKeys, INT32) || liveKeys.Count(0) != 1)) return false;
    if (key.dims != value.dims || key.dims.size() != 3 || key.dims[0] != 1 ||
        key.dims[1] < 0 || key.dims[2] != width) return false;
    int64_t needed = (dynamic ? window - 1 : key.dims[1]) + (int64_t)rows;
    for (const Data *cache : {&key, &value}) {
        if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
            cache->multiDeviceData || cache->dataDeviceIds != raw.dataDeviceIds || !cache->cudaData ||
            cache->expansionDims.size() != 3 || cache->expansionDims[0] != 1 ||
            cache->expansionDims[1] < needed || cache->expansionDims[2] != width ||
            cache->strides.size() != 3 || cache->strides[1] != width || cache->strides[2] != 1) return false;
    }
    for (const Data *src : {&raw, &qNorm, &kNorm, &positions, &liveKeys, (const Data *)&key, (const Data *)&value})
        if (&query == src || (query.cudaData && query.cudaData == src->cudaData)) return false;
    if (raw.cudaData == key.cudaData || raw.cudaData == value.cudaData || key.cudaData == value.cudaData) return false;
    Output(query, BFLOAT16, {1, rows, heads * dim});
    DraftQKVNormRope<<<dim3(heads + kvHeads, rows), 256, 0, cudaStreamPerThread>>>(
        (const BF16 *)raw.cudaData, (const float *)qNorm.cudaData, (const float *)kNorm.cudaData,
        (const float *)positions.cudaData, dynamic ? (const int *)liveKeys.cudaData : nullptr,
        (BF16 *)key.cudaData, (BF16 *)value.cudaData, (BF16 *)query.cudaData,
        heads, kvHeads, dim, window, key.dims[1], eps, theta);
    CheckLaunch();
    return true;
}

void FastllmCudaNaiveDraftSwiGLU(const fastllm::Data &gateUp, fastllm::Data &output) {
    using namespace fastllm;
    AssertInFastLLM(DraftDense(gateUp, BFLOAT16) && gateUp.dims.back() % 2 == 0 &&
        gateUp.Count(0) <= INT_MAX && &gateUp != &output &&
        (!output.cudaData || output.cudaData != gateUp.cudaData), "Invalid draft GateUp layout.");
    auto dims = gateUp.dims;
    dims.back() /= 2;
    Output(output, BFLOAT16, dims);
    const int count = gateUp.Count(0) / 2;
    DraftRoundedSwiGLU<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        (const BF16 *)gateUp.cudaData, (BF16 *)output.cudaData, dims.back(), count);
    CheckLaunch();
}

bool FastllmCudaNaiveDraftKV(const std::vector<const fastllm::Data *> &raw,
        const std::vector<const fastllm::Data *> &norm, int start,
        std::vector<std::pair<fastllm::Data, fastllm::Data>> &kv,
        int heads, int dim, int window, int reserve, float eps, float theta) {
    using namespace fastllm;
    if (raw.empty() || raw.size() > INT_MAX || raw.size() != norm.size() ||
        heads <= 0 || dim <= 0 || dim > 256 || dim % 2 || window <= 1 ||
        reserve < 0 || start < 0 || (int64_t)heads * dim > INT_MAX / 2 ||
        !std::isfinite(eps) || eps < 0 || !std::isfinite(theta) || theta <= 0) return false;
    const int layers = (int)raw.size(), columns = 2 * heads * dim;
    int rows = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (!raw[i] || !norm[i] || !DraftDense(*raw[i], BFLOAT16) ||
            !DraftDense(*norm[i], FLOAT32) || norm[i]->dims != std::vector<int>{dim} ||
            raw[i]->dims.size() != 3 || raw[i]->dims[0] != 1 || raw[i]->dims[2] != columns ||
            (i && raw[i]->dims[1] != rows)) return false;
        rows = raw[i]->dims[1];
    }
    const std::vector<int> devices{FastllmCudaGetDevice()};
    auto resident = [&](const Data &data, DataType type) {
        return data.dataType == type && data.dataDevice == DataDevice::CUDA &&
            data.dataDeviceIds == devices && data.cudaData && !data.multiDeviceData;
    };
    if ((int64_t)start + rows > INT_MAX ||
        (int64_t)std::min(layers, kDraftKVLayersPerLaunch) * rows * heads > INT_MAX ||
        (!kv.empty() && kv.size() != raw.size())) return false;
    const int width = heads * dim;
    // Validate every layer before resizing any cache; rejected calls are safe to
    // fall back to the separate projections and ordinary append operations.
    for (const auto &pair : kv) {
        if (pair.first.dims != pair.second.dims) return false;
        for (const Data *cache : {&pair.first, &pair.second}) {
            if (cache->dims.empty()) {
                if (!cache->expansionDims.empty()) return false;
                continue;
            }
            if (!resident(*cache, BFLOAT16) || cache->dims.size() != 3 ||
                cache->dims[0] != 1 || cache->dims[2] != width || cache->dims[1] < 0 ||
                (int64_t)cache->dims[1] + rows > INT_MAX ||
                cache->strides.size() != 3 || cache->strides[2] != 1 || cache->strides[1] != width ||
                (!cache->expansionDims.empty() && (cache->expansionDims.size() != 3 ||
                    cache->expansionDims[0] != 1 || cache->expansionDims[2] != width))) return false;
        }
    }
    kv.resize(layers);
    // Pass a bounded pointer table by value; models with more layers use multiple
    // launches, without a per-call host-to-device pointer upload or fixed layer cap.
    for (int first = 0; first < layers; first += kDraftKVLayersPerLaunch) {
        DraftKVDestinations dst{};
        int count = std::min(kDraftKVLayersPerLaunch, layers - first);
        for (int i = 0; i < count; ++i) {
            auto &pair = kv[first + i];
            int old = pair.first.dims.empty() ? 0 : pair.first.dims[1];
            dst.raw[i] = (const BF16 *)raw[first + i]->cudaData;
            dst.norm[i] = (const float *)norm[first + i]->cudaData;
            dst.length[i] = old;
            for (Data *cache : {&pair.first, &pair.second}) {
                cache->dataType = BFLOAT16;
                cache->UpdateUnitSize();
                cache->ToDevice(DataDevice::CUDA, devices);
                int64_t wanted = std::max(old + rows, reserve);
                if (cache->expansionDims.empty() || cache->expansionDims[1] < wanted)
                    cache->Expansion({1, (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128), width});
                cache->Resize({1, old + rows, width});
                cache->isKVCache = true;
            }
            dst.key[i] = (BF16 *)pair.first.cudaData;
            dst.value[i] = (BF16 *)pair.second.cudaData;
        }
        DraftNormRopeWrite<<<count * rows * heads, 256, 0, cudaStreamPerThread>>>(
            dst, rows, heads, dim, start, eps, theta);
        CheckLaunch();
    }
    for (auto &pair : kv) FastllmCudaNaiveTrimCache(pair.first, pair.second, window - 1);
    return true;
}

void FastllmCudaNaiveTrimCache(fastllm::Data &key, fastllm::Data &value, int keep) {
    using namespace fastllm;
    AssertInFastLLM(keep >= 0 && key.dims.size() == 3 && value.dims.size() == 3 &&
        key.dims[0] == 1 && value.dims[0] == 1 && key.dims[1] == value.dims[1] &&
        FastllmNaiveKVType(key.dataType) && value.dataType == key.dataType &&
        key.dataDevice == DataDevice::CUDA && value.dataDevice == DataDevice::CUDA &&
        key.dataDeviceIds == value.dataDeviceIds && key.dims[2] > 0 && value.dims[2] > 0,
        "Invalid Naive-N0.5 sliding cache layout.");
    if (key.dims[1] <= keep) return;
    int drop = key.dims[1] - keep;
    const int kb = key.dims[2] * key.unitSize, vb = value.dims[2] * value.unitSize;
    if (kb % 16 == 0 && vb % 16 == 0) {
        LaunchTrimCachePair((uint4 *)key.cudaData, (uint4 *)value.cudaData,
            kb / 16, vb / 16, drop, keep);
    } else {
        LaunchTrimCachePair((uint8_t *)key.cudaData, (uint8_t *)value.cudaData,
            kb, vb, drop, keep);
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
        Data encoded, offsets;
        Output(encoded, INT32, {queries, sortStride, 2});
        const bool blockSort = topK == 2048 && sortStride <= 4096;
        Output(offsets, INT32, {2, queries});
        Output(indices, INT32, {queries, topK});
        auto *input = (unsigned long long *)encoded.cudaData;
        auto *begin = (int *)offsets.cudaData;
        if (compact) {
            SelectTopKCompact<<<queries, 256>>>((const float *)scores.cudaData,
                input, begin, queries, keys, queryStart, topK, sortStride);
        } else {
            EncodeTopK<<<dim3((keys + 255) / 256, queries), 256>>>(
                (const float *)scores.cudaData, input, begin, queries, keys, queryStart);
        }
        if (blockSort) {
            SortTopKCompact<<<queries, 256>>>(input, begin, (int *)indices.cudaData,
                queries, sortStride, topK, keys);
            CheckLaunch();
            return;
        }
        Data sorted, workspace;
        Output(sorted, INT32, {queries, sortStride, 2});
        auto *output = (unsigned long long *)sorted.cudaData;
        size_t bytes = 0;
        int items = compact ? queries * sortStride : (queries - 1) * keys + queryStart + queries;
        auto status = cub::DeviceSegmentedRadixSort::SortKeysDescending(
            nullptr, bytes, input, output, items, queries, begin, begin + queries,
            0, 64, cudaStreamPerThread);
        AssertInFastLLM(status == cudaSuccess && (bytes + 3) / 4 <= INT_MAX,
            "Naive-N0.5 batched TopK workspace query failed.");
        Output(workspace, INT32, {(int)((bytes + 3) / 4)});
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
                            heads > 0 && rotaryDim > 0 && rotaryDim % 2 == 0 && rotaryDim <= dim,
                            "Invalid Naive-N0.5 RoPE input.");
    const uint64_t rows = input.Count(0) / dim / heads;
    const bool tokenRope = rows >= 512 && heads >= 8 && rotaryDim <= 256;
    auto kernel = tokenRope ? RopeToken : Rope;
    kernel<<<tokenRope ? rows : rows * heads, tokenRope ? 256 : 128>>>((BF16 *)input.cudaData,
        (const float *)positions.cudaData, heads, dim, rotaryDim, theta);
    CheckLaunch();
}

void FastllmCudaNaiveRopeQKScaleV(fastllm::Data &q, fastllm::Data &k,
        fastllm::Data &v, const fastllm::Data &positions,
        int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, float valueScale, const fastllm::Data *packedQkv) {
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
    if (packedQkv) {
        const auto &x = *packedQkv;
        AssertInFastLLM(x.dims == std::vector<int>({1, q.dims[1], (heads + kvHeads) * dim + kvHeads * valueDim}) &&
            x.dataType == BFLOAT16 && x.dataDevice == DataDevice::CUDA && x.dataDeviceIds == q.dataDeviceIds &&
            x.cudaData && x.strides.size() == 3 && x.strides[2] == 1 && x.strides[1] == x.dims[2],
            "Invalid Naive-N0.5 packed QKV input.");
    }
    const uintptr_t addresses = (uintptr_t)q.cudaData | (uintptr_t)k.cudaData |
        (uintptr_t)v.cudaData | (packedQkv ? (uintptr_t)packedQkv->cudaData : 0);
    const bool tokenRope = q.dims[1] >= 1024 && rotaryDim <= 256 &&
        dim % 2 == 0 && valueDim % 2 == 0 && addresses % alignof(__nv_bfloat162) == 0;
    auto kernel = tokenRope ? (packedQkv ? RopeQKScaleVToken<true> : RopeQKScaleVToken<false>)
        : (packedQkv ? RopeQKScaleV<true> : RopeQKScaleV<false>);
    kernel<<<tokenRope ? (uint64_t)q.dims[1] : (uint64_t)q.dims[1] * (heads + 2 * kvHeads),
        tokenRope ? 256 : 128>>>(
        (BF16 *)q.cudaData, (BF16 *)k.cudaData, (BF16 *)v.cudaData,
        (const float *)positions.cudaData, heads, kvHeads, dim, valueDim,
        rotaryDim, theta, __float2bfloat16_rn(valueScale), packedQkv ? (const BF16 *)packedQkv->cudaData : nullptr);
    CheckLaunch();
}

static bool PrepareRopeCache(fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indexKey,
        const fastllm::Data &positions, fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, float valueScale, int window, const fastllm::Data *packedQkv, bool ring, RopeCacheRow &row, int &indexDim) {
    using namespace fastllm;
    const std::vector<int> device{FastllmCudaGetDevice()};
    const bool dynamicLength = !liveKeys.dims.empty();
    auto resident = [&](const Data &x, DataType type) {
        return x.dataType == type && x.dataDevice == DataDevice::CUDA &&
            x.dataDeviceIds == device && x.cudaData;
    };
    auto matrix = [&](const Data &x, int64_t columns, bool cache = false) {
        return resident(x, cache ? x.dataType : BFLOAT16) && (!cache || FastllmNaiveKVType(x.dataType)) && x.dims.size() == 3 && x.dims[0] == 1 &&
            x.dims[1] >= 0 && x.dims[2] == columns && x.strides.size() == 3 &&
            x.strides[2] == 1 && x.strides[1] == columns;
    };
    if (heads <= 0 || kvHeads <= 0 || dim <= 0 || valueDim <= 0 ||
        rotaryDim <= 0 || rotaryDim % 2 || rotaryDim > dim || window < 0 ||
        !std::isfinite(theta) || theta <= 0 || !std::isfinite(valueScale) ||
        !matrix(q, (int64_t)heads * dim) || q.dims[1] <= 0 ||
        !matrix(k, (int64_t)kvHeads * dim) || k.dims[1] != q.dims[1] ||
        !matrix(v, (int64_t)kvHeads * valueDim) || v.dims[1] != q.dims[1] ||
        !resident(positions, FLOAT32) || positions.Count(0) < (uint64_t)q.dims[1] ||
        (dynamicLength && (!resident(liveKeys, INT32) || liveKeys.Count(0) < 1))) return false;
    const int rows = q.dims[1];
    if (ring && (!dynamicLength || rows != 1 || window != kSwaWindow)) return false;
    if (packedQkv && (!matrix(*packedQkv, (int64_t)(heads + kvHeads) * dim + (int64_t)kvHeads * valueDim) ||
                     packedQkv->dims[1] != rows)) return false;
    indexDim = 0;
    if (!indexKey.dims.empty()) {
        if (indexKey.dims.size() != 3 || indexKey.dims[2] < rotaryDim ||
            !matrix(indexKey, indexKey.dims[2]) || indexKey.dims[1] != rows) return false;
        indexDim = indexKey.dims[2];
    }
    if (key.kvCacheFp8Indexer && indexDim != 128) return false;
    const int64_t kc = (int64_t)kvHeads * dim + (key.kvCacheFp8Indexer
        ? FastllmNaiveIndexerBytes / key.unitSize : indexDim), vc = (int64_t)kvHeads * valueDim;
    if (kc > INT_MAX || vc > INT_MAX ||
        (int64_t)rows * ((int64_t)heads + 2LL * kvHeads + (indexDim != 0)) > INT_MAX ||
        !matrix(key, kc, true) || !matrix(value, vc, true) || key.dataType != value.dataType) return false;
    auto reservedRows = [](const Data &x) {
        if (x.expansionDims.empty()) return x.dims[1];
        return x.expansionDims.size() == 3 && x.expansionDims[0] == 1 &&
            x.expansionDims[2] == x.dims[2] ? x.expansionDims[1] : 0;
    };
    int keyRows = reservedRows(key), valueRows = reservedRows(value);
    if (keyRows < rows || valueRows < rows ||
        (window && (int64_t)window - 1 + rows > std::min(keyRows, valueRows))) return false;
    const int cachePast = key.dims[1];
    if (!dynamicLength && (cachePast != value.dims[1] ||
        (int64_t)cachePast + rows > std::min(keyRows, valueRows))) return false;
    row = {(BF16 *)q.cudaData, (BF16 *)key.cudaData, (BF16 *)value.cudaData,
        (const BF16 *)k.cudaData, (const BF16 *)v.cudaData,
        indexDim ? (const BF16 *)indexKey.cudaData : nullptr,
        packedQkv ? (const BF16 *)packedQkv->cudaData : nullptr,
        (const float *)positions.cudaData, dynamicLength ? (const int *)liveKeys.cudaData : nullptr,
        keyRows, valueRows, cachePast};
    row.fp8 = key.dataType == FP8_E4M3;
    row.scaledIndexer = key.kvCacheFp8Indexer;
    if (key.isPagedKVCache) {
        if (!value.isPagedKVCache || key.pageLen != value.pageLen || window) return false;
        row.key = (BF16 *)key.pagedKVCacheData->cudaData;
        row.value = (BF16 *)value.pagedKVCacheData->cudaData;
        row.keyPages = (const int *)key.cudaData;
        row.valuePages = (const int *)value.cudaData;
        row.pageLen = key.pageLen;
    }
    return true;
}

bool FastllmCudaNaiveRopeAppendCache(fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indexKey,
        const fastllm::Data &positions, fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, float valueScale, int window, const fastllm::Data *packedQkv, bool ring) {
    RopeCacheRow row{};
    int indexDim = 0;
    if (!PrepareRopeCache(q, k, v, indexKey, positions, key, value, liveKeys,
            heads, kvHeads, dim, valueDim, rotaryDim, theta, valueScale, window,
            packedQkv, ring, row, indexDim)) return false;
    auto kernel = packedQkv ? RopeAppendCache<true> : RopeAppendCache<false>;
    kernel<<<q.dims[1] * (heads + 2 * kvHeads + (indexDim != 0)), 128>>>(row,
        heads, kvHeads, dim, valueDim, indexDim, rotaryDim, theta,
        __float2bfloat16_rn(valueScale), window, ring);
    CheckLaunch();
    return true;
}

static bool PrepareBatchRopeCache(fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &positions,
        const std::vector<FastllmNaiveRopeCacheRow> &rows, const fastllm::Data &liveKeys,
        int heads, int kvHeads, int dim, int valueDim, int rotaryDim, float theta,
        float valueScale, int window, const fastllm::Data *packedQkv, bool ring,
        std::vector<RopeCacheRow> &descriptors, int &indexDim) {
    using namespace fastllm;
    const int count = rows.size();
    if (!count || q.dims.size() != 3 || q.dims[1] != count || k.dims.size() != 3 ||
        k.dims[1] != count || v.dims.size() != 3 || v.dims[1] != count ||
        positions.Count(0) < (uint64_t)count || liveKeys.Count(0) < (uint64_t)count ||
        (packedQkv && (packedQkv->dims.size() != 3 || packedQkv->dims[1] != count))) return false;
    descriptors.resize(count);
    indexDim = 0;
    for (int b = 0; b < count; ++b) {
        if (!rows[b].key || !rows[b].value) return false;
        auto view = [b](Data &dst, const Data &src) {
            auto dims = src.dims;
            dims[1] = 1;
            dst.Resize(dims);
            dst.strides = src.strides;
            dst.FakeFrom(src, (size_t)b * src.dims.back() * src.unitSize);
        };
        Data rq, rk, rv, rp, rl, packed, empty;
        view(rq, q); view(rk, k); view(rv, v);
        rp.Resize({1, 1}); rp.FakeFrom(positions, b * sizeof(float));
        rl.Resize({1}); rl.FakeFrom(liveKeys, b * sizeof(int));
        if (packedQkv) view(packed, *packedQkv);
        int width = 0;
        if (!PrepareRopeCache(rq, rk, rv, rows[b].indexKey ? *rows[b].indexKey : empty,
                rp, *rows[b].key, *rows[b].value, rl, heads, kvHeads, dim, valueDim,
                rotaryDim, theta, valueScale, window, packedQkv ? &packed : nullptr,
                ring, descriptors[b], width) || (b && width != indexDim)) return false;
        indexDim = width;
        descriptors[b].cache = rows[b].cache;
    }
    return true;
}

bool FastllmCudaNaiveBatchRopeAppendCache(fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &positions,
        const std::vector<FastllmNaiveRopeCacheRow> &rows, const fastllm::Data &liveKeys,
        int heads, int kvHeads, int dim, int valueDim, int rotaryDim, float theta,
        float valueScale, int window, const fastllm::Data *packedQkv, bool ring) {
    const int count = rows.size();
    std::vector<RopeCacheRow> descriptors;
    int indexDim = 0;
    if (!PrepareBatchRopeCache(q, k, v, positions, rows, liveKeys, heads, kvHeads,
            dim, valueDim, rotaryDim, theta, valueScale, window, packedQkv, ring,
            descriptors, indexDim)) return false;
    // Validate every cache before launching, so a fallback cannot double-rotate Q.
    for (int first = 0; first < count; first += kRopeCacheBatch) {
        const int batchSize = std::min(kRopeCacheBatch, count - first);
        RopeCacheBatch batch{};
        std::copy_n(descriptors.begin() + first, batchSize, batch.rows);
        auto kernel = packedQkv ? BatchRopeAppendCache<true> : BatchRopeAppendCache<false>;
        kernel<<<dim3(heads + 2 * kvHeads + (indexDim != 0), batchSize), 128>>>(batch,
            heads, kvHeads, dim, valueDim, indexDim, rotaryDim, theta,
            __float2bfloat16_rn(valueScale), window, ring);
    }
    CheckLaunch();
    return true;
}

bool FastllmCudaNaiveBatchQkvRopeAppendCache(const fastllm::Data &input,
        const fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
        const fastllm::Data &positions, const std::vector<FastllmNaiveRopeCacheRow> &rows,
        const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int valueDim,
        int rotaryDim, float theta, float valueScale, int window, bool ring,
        fastllm::Data &scratch) {
#ifdef FASTLLM_ENABLE_NAIVE_QKV_SM90
    using namespace fastllm;
    const int count = rows.size();
    if (count < 1 || count > 256 || heads != 8 || kvHeads != 1 || dim != 192 ||
        valueDim != 128 || rotaryDim != 64 || !bias.dims.empty() ||
        (window && (!ring || window != 128)) || FastllmCudaRuntimeArch() != 90) return false;
    auto matrix = [&](const Data &x, const std::vector<int> &shape) {
        if (x.dataType != BFLOAT16 || x.dataDevice != DataDevice::CUDA ||
            x.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
            !x.cudaData || (reinterpret_cast<uintptr_t>(x.cudaData) & 15) ||
            x.dims != shape || x.strides.size() != shape.size()) return false;
        size_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            if (x.strides[i] != stride) return false;
            stride *= shape[i];
        }
        return true;
    };
    if (!matrix(input, {1, count, 4096}) || !matrix(weight, {1856, 4096})) return false;
    std::vector<RopeCacheRow> descriptors;
    int indexDim = 0;
    if (!PrepareBatchRopeCache(q, k, v, positions, rows, liveKeys, heads, kvHeads,
            dim, valueDim, rotaryDim, theta, valueScale, window, nullptr, ring,
            descriptors, indexDim) || (indexDim && indexDim != 128)) return false;
    const auto &first = descriptors.front();
    if (!first.cache) return false;
    for (int b = 0; b < count; ++b) {
        const auto &r = descriptors[b];
        if (r.cache != first.cache + b * 4 || r.keyRows != first.keyRows ||
            r.valueRows != first.valueRows || r.pageLen != first.pageLen || r.fp8 != first.fp8 || r.scaledIndexer != first.scaledIndexer ||
            (indexDim && r.indexKey != first.indexKey + b * 128)) return false;
    }
    if (!naive_qkv::Prepare()) return false;
    if (count > 1) {
        // Reuse the O-projection buffer after Indexer consumes its temporary.
        // Reserve the monotonic envelope: capturing a smaller batch later must
        // never relocate a shared graph allocation at split-K boundaries.
        int units = count <= 16 ? count * 8 : std::max(256, count * 2);
        scratch.dataType = BFLOAT16;
        scratch.Resize({1, units, 1856 * 2}); // FP32 partials in BF16-sized storage.
        scratch.ToDevice(input.dataDevice, input.dataDeviceIds, false);
        scratch.Allocate(false);
    }
    naive_qkv::Params params{(BF16 *)q.cudaData, first.indexKey,
        (const float *)positions.cudaData, (const int *)liveKeys.cudaData, first.cache,
        theta, __float2bfloat16_rn(valueScale), first.pageLen, first.keyRows, first.valueRows, ring, first.fp8, first.scaledIndexer};
    naive_qkv::Launch((const BF16 *)input.cudaData, (const BF16 *)weight.cudaData,
        params, count, count > 1 ? (float *)scratch.cudaData : nullptr);
    CheckLaunch();
    return true;
#else
    return false;
#endif
}

bool FastllmCudaNaiveIndexerKeyNorm(fastllm::Data &input,
        const fastllm::Data &weight, const fastllm::Data &bias) {
    using namespace fastllm;
    const std::vector<int> device{FastllmCudaGetDevice()};
    auto resident = [&](const Data &x, DataType type) {
        return x.dataType == type && x.dataDevice == DataDevice::CUDA &&
            x.dataDeviceIds == device && x.cudaData;
    };
    if (!resident(input, BFLOAT16) || !resident(weight, FLOAT32) || !resident(bias, FLOAT32) ||
        input.dims.size() != 3 || input.dims[0] != 1 || input.dims[1] <= 0 ||
        input.dims[2] != 128 || input.strides.size() != 3 || input.strides[1] != 128 ||
        input.strides[2] != 1 || weight.Count(0) != 128 || bias.Count(0) != 128) return false;
    IndexerKeyNorm<<<input.dims[1], 64>>>((BF16 *)input.cudaData,
        (const float *)weight.cudaData, (const float *)bias.cudaData);
    CheckLaunch();
    return true;
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

void FastllmCudaNaiveRestoreIndexerRows(fastllm::Data &indices, int ranks) {
    using namespace fastllm;
    AssertInFastLLM(indices.dataType == INT32 && indices.dataDevice == DataDevice::CUDA &&
        indices.cudaData && indices.dataDeviceIds.size() == 1 &&
        indices.dims.size() == 2 && indices.dims[0] > 0 &&
        indices.dims[1] > 0 && (ranks == 2 || ranks == 4 || ranks == 8) &&
        indices.dims[0] % (2 * ranks) == 0 && indices.Count(0) / (2 * ranks) <= INT_MAX,
        "Invalid balanced Indexer row layout.");
    const int count = indices.Count(0) / (2 * ranks);
    const int blocks = (count - 1) / 256 + 1;
    FastllmCudaSetDevice(indices.dataDeviceIds[0]);
    if (ranks == 2) RestoreBalancedRows<2><<<blocks, 256>>>((int *)indices.cudaData, count);
    else if (ranks == 4) RestoreBalancedRows<4><<<blocks, 256>>>((int *)indices.cudaData, count);
    else RestoreBalancedRows<8><<<blocks, 256>>>((int *)indices.cudaData, count);
    CheckLaunch();
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
        query.dataType == BFLOAT16 && FastllmNaiveKVType(packedKeys.dataType) && weights.dataType == BFLOAT16 &&
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
    // Q retains E4M3 values and FP32 scales; the cache descriptor selects K storage.
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
                (const BF16 *)q.cudaData, FastllmNaiveKV(packedKeys),
                (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                topKOutput, stride, keys, queryStart);
        }
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM90
        else if (naive_indexer::Prepare()) {
            naive_indexer::Launch((const BF16 *)q.cudaData, FastllmNaiveKV(packedKeys),
                (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                (float *)scores.cudaData, queries, keys, queryStart, stride);
        }
#endif
        else {
            // TopK consumes only each row's causal prefix. Blocks beyond
            // the final query's prefix cannot contribute any selected key.
            const bool pipeline = FastllmCudaRuntimeArch() == 90;
            const int rows = pipeline ? 32 : 64;
            auto kernel = pipeline ? naive_dsa_mma::IndexerScores<true> : naive_dsa_mma::IndexerScores<false>;
            kernel<<<dim3((queryStart + queries + kKeys - 1) / kKeys,
                (queries + rows - 1) / rows), rows * 4>>>(
                (const BF16 *)q.cudaData, FastllmNaiveKV(packedKeys),
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
                FastllmNaiveKV(packedKeys), (const BF16 *)weights.cudaData,
                topKOutput, stride, keys, queryStart);
        } else {
            Output(k, DataType::FLOAT32, {keys, dim});
            RoundIndexer<<<keys, 128>>>(FastllmNaiveKV(packedKeys),
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
        FastllmNaiveKV k(key), v(value);
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
        Data k(key.dataType, {1, length, key.dims[2]});
        Data v(value.dataType, {1, length, value.dims[2]});
        Data out(BFLOAT16, {1, 1, heads * valueDim}), selected;
        q.FakeFrom(query, (size_t)row * heads * dim * sizeof(BF16));
        k.FakeFrom(key, (size_t)begin * key.dims[2] * key.unitSize);
        v.FakeFrom(value, (size_t)begin * value.dims[2] * value.unitSize);
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

static bool LaunchNaivePrefillAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        int heads, int kvHeads, int pastLength, bool causal, int indexTile,
        fastllm::Data &output) {
    using namespace fastllm;
    namespace mma = naive_prefill_mma;
    const int queries = query.dims[1], keys = key.dims[1];
    const int count = indices.dims.empty() ? keys : indices.dims[1];
    if (count <= 0 || count > 2048 || (size_t)query.cudaData % 16 != 0) return false;
    auto kernel = mma::Generic<2048>;
    size_t shared = sizeof(mma::Storage<2048>);
    if (FastllmCudaRuntimeArch() == 90 &&
        BulkKernelSupported(mma::Hopper, sizeof(mma::HopperStorage), 90)) {
        kernel = mma::Hopper;
        shared = sizeof(mma::HopperStorage);
    }
    if (!BulkKernelSupported(kernel, shared, 80)) return false;
    Output(output, BFLOAT16, {1, queries, heads * 128});
    kernel<<<dim3(queries, heads / 8), 256, shared>>>(
        (const BF16 *)query.cudaData, FastllmNaiveKV(key), FastllmNaiveKV(value),
        indices.dims.empty() ? nullptr : (const int *)indices.cudaData,
        sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
        (BF16 *)output.cudaData, heads, kvHeads, key.dims[2], keys, count,
        pastLength, causal, indexTile);
    CheckLaunch();
    return true;
}

bool FastllmCudaNaiveBalancedAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        int heads, int kvHeads, int pastLength, int ranks, fastllm::Data &output) {
    using namespace fastllm;
    if ((ranks != 2 && ranks != 4 && ranks != 8) || heads <= 0 || kvHeads <= 0 ||
        heads != kvHeads * 8 || query.dims.size() != 3 || query.dims[0] != 1 ||
        query.dims[1] < 32 || query.dims[1] % (2 * ranks) != 0 || query.dims[2] != heads * 192 ||
        key.dims.size() != 3 || value.dims.size() != 3 || key.dims[0] != 1 || value.dims[0] != 1 ||
        key.dims[1] <= 0 || value.dims[1] != key.dims[1] || key.dims[2] < kvHeads * 192 ||
        key.dims[2] % 8 != 0 || value.dims[2] != kvHeads * 128 ||
        indices.dims.size() != 2 || indices.dims[0] != query.dims[1] ||
        indices.dataType != INT32 || query.dataType != BFLOAT16 ||
        !FastllmNaiveKVType(key.dataType) || !FastllmNaiveKVType(value.dataType) ||
        (size_t)key.cudaData % 16 != 0 || (size_t)value.cudaData % 16 != 0) return false;
    return LaunchNaivePrefillAttention(query, key, value, indices, sink,
        heads, kvHeads, pastLength, true, query.dims[1] / (2 * ranks), output);
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
        key.dataType == BFLOAT16 && value.dataType == BFLOAT16 &&
        dim == kSwaQkDim && valueDim == kSwaValueDim && key.dims[2] % 8 == 0 &&
        (size_t)query.cudaData % 16 == 0 && (size_t)key.cudaData % 16 == 0 &&
        (size_t)value.cudaData % 16 == 0 && FastllmCudaFlashInferDataTypeSupported(DataType::BFLOAT16)) {
        auto status = naive_swa_flashinfer::Run((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, queries, keys, heads, kvHeads, key.dims[2],
            cudaStreamPerThread, FastllmCudaRuntimeArch() == 90);
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
            FastllmNaiveKV(key), FastllmNaiveKV(value),
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
            FastllmNaiveKV(key), FastllmNaiveKV(value),
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
            (BF16 *)output.cudaData, heads, kvHeads, key.dims[2], keys);
        CheckLaunch();
        return;
    }
    if (count <= 256) {
        AttentionShort<<<dim3(heads, queries), 256>>>((const BF16 *)query.cudaData,
            FastllmNaiveKV(key), FastllmNaiveKV(value), selected,
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
    const bool tiledPrefill = queries >= 32 && window == 0 && heads % kvHeads == 0 &&
        heads / kvHeads == naive_prefill::Heads && dim == naive_prefill::QkDim &&
        valueDim == naive_prefill::ValueDim && key.dims[2] % 8 == 0 &&
        (size_t)key.cudaData % 16 == 0 && (size_t)value.cudaData % 16 == 0;
    if (tiledPrefill && LaunchNaivePrefillAttention(query, key, value, indices, sink,
            heads, kvHeads, pastLength, causal, 0, output)) return;
    Data scores;
    Output(scores, DataType::FLOAT32, {queries, heads, count});
    if (tiledPrefill) {
        naive_prefill::Scores<<<dim3(queries, heads / naive_prefill::ScoreHeads, (count + 63) / 64), 128>>>(
            (const BF16 *)query.cudaData, FastllmNaiveKV(key), selected, (float *)scores.cudaData,
            heads, kvHeads, key.dims[2], keys, count, pastLength, causal);
        AttentionSoftmax<<<queries * heads, 256>>>((float *)scores.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData, heads, count);
        naive_prefill::Values<<<dim3(queries, heads / naive_prefill::Heads), 256>>>(
            (const float *)scores.cudaData, FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
            heads, kvHeads, keys, count, pastLength, causal);
        CheckLaunch();
        return;
    }
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
            (const BF16 *)query.cudaData, FastllmNaiveKV(key), selected, (float *)scores.cudaData,
            heads, kvHeads, key.dims[2], keys, count, pastLength, causal);
    } else
#endif
    // Each shared tile must stay within one KV head and have aligned vector loads.
    if (sharedDecode && dim == kDecodeQkDim && key.dims[2] % 8 == 0 &&
        (size_t)key.cudaData % 16 == 0) {
        AttentionScoresDecodeShared<<<dim3(heads / kDecodeSharedHeads, count / kDecodeQkKeyTile), kDecodeSharedThreads>>>(
            (const BF16 *)query.cudaData, FastllmNaiveKV(key), selected,
            (float *)scores.cudaData, heads, kvHeads, dim, key.dims[2],
            keys, count, pastLength, causal);
    } else if (groupedDecode && dim <= kDecodeQkDim) {
        AttentionScoresDecodeGrouped<4, kDecodeQkDim><<<dim3(heads / 4, count / 32), 128>>>(
            (const BF16 *)query.cudaData, FastllmNaiveKV(key), selected,
            (float *)scores.cudaData, heads, kvHeads, dim, key.dims[2],
            keys, count, pastLength, causal);
    } else
    // Small query blocks need the original per-head CTA count for occupancy.
    if (queries >= 32 && dim <= 192 && heads / kvHeads >= 2) {
        AttentionScoresPrefillGrouped<<<dim3(kvHeads, queries, (heads / kvHeads + 1) / 2), 128>>>(
            (const BF16 *)query.cudaData, FastllmNaiveKV(key), selected, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    } else if (queries > 1 && dim <= 256) {
        auto kernel = dim <= 192 ? AttentionScoresPrefill<192> : AttentionScoresPrefill<256>;
        kernel<<<dim3(heads, queries), 128>>>((const BF16 *)query.cudaData,
            FastllmNaiveKV(key), selected, (float *)scores.cudaData,
            heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    } else {
        AttentionScores<<<dim3(heads, queries, queries == 1 ? (count + 63) / 64 : 1), 256>>>((const BF16 *)query.cudaData,
            FastllmNaiveKV(key), selected, (float *)scores.cudaData,
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
            (const float *)scores.cudaData, FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
            heads, kvHeads, keys, count, pastLength, causal);
    } else
#endif
    if (splitDecode && valueDim == kDecodePVValueDim) {
        Data partial;
        Output(partial, FLOAT32, {heads, kDecodePVParts, kDecodePVValueDim});
        if (sharedDecode && (size_t)value.cudaData % 16 == 0) {
            AttentionValuesDecodeShared<<<dim3(heads / kDecodeSharedHeads, kDecodePVParts,
                kDecodePVValueDim / kDecodeValueTile), kDecodeSharedThreads>>>(
                (const float *)scores.cudaData, FastllmNaiveKV(value),
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        } else if (groupedDecode) {
            AttentionValuesDecodeGrouped<4><<<dim3(heads / 4, kDecodePVParts), kDecodePVValueDim>>>(
                (const float *)scores.cudaData, FastllmNaiveKV(value),
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        } else {
            AttentionValuesDecodePartial<<<dim3(heads, kDecodePVParts), kDecodePVValueDim>>>(
                (const float *)scores.cudaData, FastllmNaiveKV(value),
                selected, (float *)partial.cudaData, heads, kvHeads, keys, pastLength, causal);
        }
        AttentionValuesDecodeReduce<<<(heads * kDecodePVValueDim + 255) / 256, 256>>>(
            (const float *)partial.cudaData, (BF16 *)output.cudaData, heads);
    } else if (queries == 1) {
        AttentionValuesTiled<<<dim3(heads, queries, (valueDim + 31) / 32), 256>>>((const float *)scores.cudaData,
            FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
            heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    } else if (valueDim % 4 == 0 && (size_t)value.cudaData % alignof(uint2) == 0) {
        // The cooperative page lookup requires all 32 lanes to own columns.
        if (value.isPagedKVCache && valueDim % 128 == 0)
            AttentionValuesPrefill<int, int, true><<<dim3((heads + 3) / 4, queries), 128>>>((const float *)scores.cudaData,
                FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
                heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
        else AttentionValuesPrefill<<<dim3((heads + 3) / 4, queries), 128>>>((const float *)scores.cudaData,
            FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
            heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    } else {
        AttentionValues<<<dim3(heads, queries), 256>>>((const float *)scores.cudaData,
            FastllmNaiveKV(value), selected, (BF16 *)output.cudaData,
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

__global__ void AppendDecodeKV(FastllmNaiveKVWriteRow key, FastllmNaiveKVWriteRow value, const BF16 *newKey,
        const BF16 *newValue, const int *length, int keyColumns, int valueColumns,
        int window, bool ring, const int *keyPages, const int *valuePages, int pageLen) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = *length - 1;
    if (window) row = ring ? (row & (kSwaWindow - 1)) : min(row, window - 1);
    int kr = keyPages ? keyPages[row / pageLen] * pageLen + row % pageLen : row;
    int vr = valuePages ? valuePages[row / pageLen] * pageLen + row % pageLen : row;
    if (column < keyColumns) key[(size_t)kr * keyColumns + column] = newKey[column];
    else if ((column -= keyColumns) < valueColumns)
        value[(size_t)vr * valueColumns + column] = newValue[column];
}

__global__ void AppendVerifyKV(FastllmNaiveKVWriteRow key, FastllmNaiveKVWriteRow value, const BF16 *newKey,
        const BF16 *newValue, const int *length, int kc, int vc, int window) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y;
    const int past = window ? min(*length - 1, window - 1) : *length - 1;
    const int row = past + query;
    if (column < kc) key[(size_t)row * kc + column] = newKey[(size_t)query * kc + column];
    else if ((column -= kc) < vc)
        value[(size_t)row * vc + column] = newValue[(size_t)query * vc + column];
}

__global__ void CopyVerifyWindow(FastllmNaiveKV key, FastllmNaiveKV value,
        BF16 *outKey, BF16 *outValue, const int *length, int query,
        int kc, int vc, int window) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int end = min(*length - 1, window - 1) + query + 1;
    const int begin = max(0, end - window), rows = min(end, window);
    if (index < rows * kc) outKey[index] = key.Row(begin + index / kc, kc)[index % kc];
    else if ((index -= rows * kc) < rows * vc)
        outValue[index] = value.Row(begin + index / vc, vc)[index % vc];
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

bool FastllmCudaNaiveRouterSigmoid(const fastllm::Data &input,
        const fastllm::Data &weight, fastllm::Data &output) {
    using namespace fastllm;
    const int rows = RouterRows(input, weight, output);
    if (!rows || (rows > 1 && rows >= FastllmCudaGetLinearExactBatchThreshold())) return false;
    auto dims = input.dims;
    dims.back() = weight.dims[0];
    Output(output, FLOAT32, dims);
    const int experts = weight.dims[0];
    if (rows == 1)
        RouterSigmoid4096<<<experts, 256>>>((const BF16 *)input.cudaData,
            (const float *)weight.cudaData, (float *)output.cudaData);
    else if (rows >= 3 && FastllmCudaRuntimeArch() == 90 &&
             (uintptr_t)input.cudaData % 4 == 0 && (uintptr_t)weight.cudaData % 8 == 0)
        BatchRouterSigmoid4096Vector<<<dim3(experts, (rows + 3) / 4), 128>>>(
            (const BF16 *)input.cudaData, (const float *)weight.cudaData,
            (float *)output.cudaData, rows, experts);
    else if (rows >= 4)
        BatchRouterSigmoid4096<4><<<dim3(experts, (rows + 3) / 4), 256>>>(
            (const BF16 *)input.cudaData, (const float *)weight.cudaData,
            (float *)output.cudaData, rows, experts);
    else
        BatchRouterSigmoid4096<2><<<dim3(experts, (rows + 1) / 2), 256>>>(
            (const BF16 *)input.cudaData, (const float *)weight.cudaData,
            (float *)output.cudaData, rows, experts);
    CheckLaunch();
    return true;
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

namespace {
struct KVReadBatch {
    struct Row { FastllmNaiveKV source; BF16 *target; int first, count, width, outWidth; } rows[32];
};
__global__ void ReadCache(KVReadBatch batch) {
    const auto &r = batch.rows[blockIdx.y];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.count) return;
    const int row = r.first + i / r.outWidth, col = i % r.outWidth;
    const int main = r.outWidth - (r.source.scaledIndexer ? FastllmNaiveIndexerBytes / 2 : 0);
    if (col < main) r.target[i] = r.source.Row(row, r.width)[col];
    else ((unsigned short *)r.target)[i] =
        ((const unsigned short *)r.source.IndexRow(row, r.width).data)[col - main];
}
__global__ void PackIndexer(const BF16 *input, BF16 *output, int width) {
    const int row = blockIdx.x, main = width - 128, outWidth = main + FastllmNaiveIndexerBytes / 2;
    for (int d = threadIdx.x; d < main; d += blockDim.x) output[(size_t)row * outWidth + d] = input[(size_t)row * width + d];
    FastllmNaiveStoreIndexer((float)input[(size_t)row * width + main + threadIdx.x], output + (size_t)row * outWidth + main);
}
__global__ void WriteIndexerCache(FastllmNaiveKV key, const BF16 *input, int width,
        int start, const int *live = nullptr, FastllmNaiveKV value = {},
        const BF16 *newValue = nullptr, int valueWidth = 0) {
    const int token = blockIdx.x;
    const int row = (live ? *live - 1 : start) + token;
    const int main = width - FastllmNaiveIndexerBytes / (key.fp8 ? 1 : 2);
    const int inputWidth = main + FastllmNaiveIndexerBytes / 2;
    auto destination = key.Row(row, width);
    FastllmNaiveKVWriteRow dst{(void *)destination.data, key.fp8};
    for (int d = threadIdx.x; d < main; d += blockDim.x) dst[d] = input[(size_t)token * inputWidth + d];
    for (int d = threadIdx.x; d < FastllmNaiveIndexerBytes / 4; d += blockDim.x)
        ((unsigned *)((char *)destination.data + main * (key.fp8 ? 1 : 2)))[d] =
            ((const unsigned *)(input + (size_t)token * inputWidth + main))[d];
    if (newValue) {
        FastllmNaiveKVWriteRow v{(void *)value.Row(row, valueWidth).data, value.fp8};
        for (int d = threadIdx.x; d < valueWidth; d += blockDim.x) v[d] = newValue[(size_t)token * valueWidth + d];
    }
}
__global__ void PagedWrite(FastllmNaiveKVWriteRow pool, const int *pages, const BF16 *input, int start, int count, int width, int pageLen) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)count * width) return;
    int row = start + i / width;
    size_t physical = pages ? (size_t)pages[row / pageLen] * pageLen + row % pageLen : row;
    pool[physical * width + i % width] = input[i];
}
}
void FastllmCudaNaiveReadCache(const std::vector<FastllmNaiveKVRead> &slices) {
    for (size_t first = 0; first < slices.size(); first += 32) {
        KVReadBatch batch{};
        int count = std::min<size_t>(32, slices.size() - first), maximum = 0;
        for (int i = 0; i < count; ++i) {
            const auto &s = slices[first + i];
            const auto &data = *s.source;
            fastllm::AssertInFastLLM(FastllmNaiveKVType(data.dataType) &&
                data.dataDevice == fastllm::DataDevice::CUDA && data.dims.size() == 3 &&
                s.target && s.first >= 0 && s.rows > 0, "Invalid Naive KV read.");
            int outWidth = data.dims[2];
            if (data.kvCacheFp8Indexer) outWidth += FastllmNaiveIndexerBytes / 2 - FastllmNaiveIndexerBytes / data.unitSize;
            int elements = s.rows * outWidth;
            batch.rows[i] = {FastllmNaiveKV(data), (BF16 *)s.target, s.first, elements, data.dims[2], outWidth};
            maximum = std::max(maximum, elements);
        }
        ReadCache<<<dim3((maximum + 255) / 256, count), 256>>>(batch);
    }
    CheckLaunch();
}
void FastllmCudaNaivePackIndexer(const fastllm::Data &input, fastllm::Data &output) {
    fastllm::AssertInFastLLM(input.dataType == fastllm::BFLOAT16 && input.dims.size() == 3 &&
        input.dims[0] == 1 && input.dims[1] > 0 && input.dims[2] >= 128 &&
        input.dataDevice == fastllm::DataDevice::CUDA && input.cudaData &&
        !input.kvCacheFp8Indexer, "Invalid Indexer packing input.");
    Output(output, fastllm::BFLOAT16, {1, input.dims[1], input.dims[2] - 128 + FastllmNaiveIndexerBytes / 2});
    output.kvCacheFp8Indexer = true;
    PackIndexer<<<input.dims[1], 128>>>((const BF16 *)input.cudaData, (BF16 *)output.cudaData, input.dims[2]);
    CheckLaunch();
}
void FastllmCudaNaivePagedWrite(fastllm::Data &cache, const fastllm::Data &input, int start) {
    using namespace fastllm;
    AssertInFastLLM(FastllmNaiveKVType(cache.dataType) && input.dataType == BFLOAT16 &&
        cache.dataDevice == DataDevice::CUDA && input.dataDevice == DataDevice::CUDA &&
        cache.dims.size() == 3 && input.dims.size() == 3 &&
        cache.kvCacheFp8Indexer == input.kvCacheFp8Indexer &&
        cache.dims[2] == input.dims[2] + (cache.kvCacheFp8Indexer
            ? FastllmNaiveIndexerBytes / cache.unitSize - FastllmNaiveIndexerBytes / 2 : 0) && start >= 0 &&
        start + input.dims[1] <= (cache.isPagedKVCache
            ? (int)cache.pageIndex.size() * cache.pageLen
            : (cache.expansionDims.empty() ? cache.dims[1] : cache.expansionDims[1])),
        "Invalid Naive KV append.");
    if (cache.kvCacheFp8Indexer) {
        WriteIndexerCache<<<input.dims[1], 128>>>(FastllmNaiveKV(cache), (const BF16 *)input.cudaData,
            cache.dims[2], start);
        CheckLaunch();
        return;
    }
    FastllmNaiveKVWriteRow destination{cache.isPagedKVCache ? cache.pagedKVCacheData->cudaData : cache.cudaData,
                                     cache.dataType == FP8_E4M3};
    PagedWrite<<<(input.Count(0) + 255) / 256, 256>>>(destination,
        cache.isPagedKVCache ? (const int *)cache.cudaData : nullptr, (const BF16 *)input.cudaData,
        start, input.dims[1], input.dims[2], cache.pageLen);
    CheckLaunch();
}

void FastllmCudaNaiveAppendDecodeCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &newKey, const fastllm::Data &newValue,
        const fastllm::Data &liveKeys, int window, bool ring) {
    fastllm::AssertInFastLLM(!ring || window == kSwaWindow, "Invalid SWA ring window.");
    if (key.kvCacheFp8Indexer) {
        WriteIndexerCache<<<1, 128>>>(FastllmNaiveKV(key), (const BF16 *)newKey.cudaData,
            key.dims[2], 0, (const int *)liveKeys.cudaData, FastllmNaiveKV(value),
            (const BF16 *)newValue.cudaData, value.dims[2]);
        CheckLaunch();
        return;
    }
    int kc = key.dims[2], vc = value.dims[2];
    AppendDecodeKV<<<(kc + vc + 255) / 256, 256>>>({key.isPagedKVCache ? key.pagedKVCacheData->cudaData : key.cudaData, key.dataType == fastllm::FP8_E4M3},
        {value.isPagedKVCache ? value.pagedKVCacheData->cudaData : value.cudaData, value.dataType == fastllm::FP8_E4M3}, (const BF16 *)newKey.cudaData,
        (const BF16 *)newValue.cudaData, (const int *)liveKeys.cudaData, kc, vc, window, ring,
        key.isPagedKVCache ? (const int *)key.cudaData : nullptr,
        value.isPagedKVCache ? (const int *)value.cudaData : nullptr, key.pageLen);
    CheckLaunch();
}

void FastllmCudaNaiveSwaCacheLayout(fastllm::Data &key, fastllm::Data &value,
        int past, bool toRing) {
    using namespace fastllm;
    const int count = std::min(past, kSwaWindow - 1);
    const int first = (past - count) & (kSwaWindow - 1);
    if (!count || !first) return;
    AssertInFastLLM(FastllmNaiveKVType(key.dataType) && value.dataType == key.dataType &&
        key.dataDevice == DataDevice::CUDA && value.dataDevice == DataDevice::CUDA &&
        key.dims.size() == 3 && value.dims.size() == 3 && key.dims[0] == 1 && value.dims[0] == 1 &&
        key.dims[1] == count && value.dims[1] == count &&
        key.expansionDims.size() == 3 && value.expansionDims.size() == 3 &&
        key.expansionDims[1] >= kSwaWindow && value.expansionDims[1] >= kSwaWindow,
        "Invalid SWA ring cache layout.");
    auto launch = [&](auto *storage) {
        using T = std::remove_pointer_t<decltype(storage)>;
        constexpr int columns = kTrimTileBytes / sizeof(T);
        int kc = key.dims[2], vc = value.dims[2];
        int blocks = (kc + columns - 1) / columns + (vc + columns - 1) / columns;
        auto kernel = toRing ? SwaCacheLayout<true, T> : SwaCacheLayout<false, T>;
        kernel<<<blocks, 256>>>((T *)key.cudaData, (T *)value.cudaData, kc, vc, count, first);
    };
    if (key.dataType == FP8_E4M3) launch((uint8_t *)nullptr);
    else launch((BF16 *)nullptr);
    CheckLaunch();
}

void FastllmCudaNaiveTrimDecodeCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &liveKeys, int window) {
    auto launch = [&](auto *storage) {
        using T = std::remove_pointer_t<decltype(storage)>;
        constexpr int columns = kTrimTileBytes / sizeof(T);
        int kc = key.dims[2], vc = value.dims[2];
        int blocks = (kc + columns - 1) / columns + (vc + columns - 1) / columns;
        TrimCachePairTiled<T, true><<<blocks, 256>>>((T *)key.cudaData,
            (T *)value.cudaData, kc, vc, 1, window - 1, (const int *)liveKeys.cudaData);
    };
    if (key.dataType == fastllm::FP8_E4M3) launch((uint8_t *)nullptr);
    else launch((BF16 *)nullptr);
    CheckLaunch();
}

namespace {
void LaunchDecodeIndexerTopK(const unsigned *scores, DecodeKeys keys, naive_topk::Workspace *workspace) {
    using namespace fastllm;
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
}
}

void FastllmCudaNaiveDecodeIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &packedKeys,
        const fastllm::Data &liveKeys, int capacity, bool fp8Query,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &indices, const void *const *cache) {
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
            FastllmNaiveKV(packedKeys), (const float *)scratch.indexScale.cudaData,
            (const BF16 *)weights.cudaData, scoreOutput, stride, keys, capacity - 1, cache);
    } else
#endif
    {
        Output(scratch.indexQuery, FLOAT32, {16, 128});
        RoundIndexer<<<16, 128>>>((const BF16 *)query.cudaData,
            (float *)scratch.indexQuery.cudaData, 128, 0, fp8Query);
        IndexScoresDecode<<<(capacity + 7) / 8, 256>>>((const float *)scratch.indexQuery.cudaData,
            FastllmNaiveKV(packedKeys), (const BF16 *)weights.cudaData,
            scoreOutput, stride, keys, capacity - 1, cache);
    }
    Output(scratch.topk, INT32, {(int)(sizeof(naive_topk::Workspace) / sizeof(int))});
    Output(indices, INT32, {1, naive_topk::kTopK});
    auto *workspace = (naive_topk::Workspace *)scratch.topk.cudaData;
    auto *scores = (const unsigned *)scratch.indexScores.cudaData;
    LaunchDecodeIndexerTopK(scores, keys, workspace);
    naive_topk::Sort<<<1, naive_topk::kThreads>>>(&workspace->candidates, &workspace->state,
        (int *)indices.cudaData);
    CheckLaunch();
}

bool FastllmCudaNaiveBatchDecodeIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &liveKeys,
        const std::vector<FastllmNaiveIndexerRow> &rows, bool fp8Query,
    int contextRank, int contextRanks, fastllm::Data *exchange) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    using namespace fastllm;
    namespace batch = naive_indexer_batch;
    constexpr int heads = naive_dsa_mma::kHeads, dim = naive_dsa_mma::kIndexerDim;
    if (rows.size() < 4 || !fp8Query || query.dataType != BFLOAT16 ||
        weights.dataType != BFLOAT16 || liveKeys.dataType != INT32 ||
        query.dims.empty() || query.dims.back() != heads * dim ||
        weights.dims.empty() || weights.dims.back() != heads ||
        !FastllmCudaFlashInferDataTypeSupported(BFLOAT16)) return false;
    const bool cp = contextRanks > 1;
    if (contextRanks < 1 || contextRank < 0 || contextRank >= contextRanks ||
        (cp && (!exchange || exchange->dataType != INT32 ||
         exchange->Count(0) < (size_t)contextRanks * rows.size() * naive_topk::kTopK * 2))) return false;
    std::vector<batch::Batch> packedBatches;
    if (cp) packedBatches.reserve((rows.size() + batch::BatchLimit - 1) / batch::BatchLimit);
    bool hopper = FastllmCudaRuntimeArch() == 90;
    const auto cacheType = rows.front().key ? rows.front().key->dataType : BFLOAT16;
    for (const auto &row : rows) {
        if (!row.key || !row.scratch || !row.indices || !FastllmNaiveKVType(row.key->dataType) ||
            row.key->dims.size() != 3 || row.key->dims[2] < dim ||
            row.capacity <= 0 || row.capacity > naive_topk::kMaxKeys ||
            row.sequence < 0 || row.sequence >= liveKeys.Count(0) ||
            (size_t)(row.sequence + 1) * heads * dim > query.Count(0) ||
            (size_t)(row.sequence + 1) * heads > weights.Count(0)) return false;
        // cp.async copies 16 bytes; FP8 therefore needs twice as many elements.
        hopper = hopper && row.key->dataType == cacheType &&
            row.key->kvCacheFp8Indexer == rows.front().key->kvCacheFp8Indexer &&
            row.key->dims[2] % (cacheType == FP8_E4M3 ? 16 : 8) == 0 &&
            (uintptr_t)row.key->cudaData % 16 == 0;
    }

    for (int first = 0; first < (int)rows.size(); first += batch::BatchLimit) {
        const int count = std::min(batch::BatchLimit, (int)rows.size() - first);
        batch::Batch descriptors{};
        int capacity = 0;
        for (int i = 0; i < count; ++i) {
            const auto &row = rows[first + i];
            auto &scratch = *row.scratch;
            Output(scratch.indexQuery, BFLOAT16, {heads, dim});
            Output(scratch.indexScale, FLOAT32, {heads});
            Output(scratch.indexScores, INT32, {row.capacity});
            Output(scratch.topk, INT32, {(int)(sizeof(naive_topk::Workspace) / sizeof(int))});
            Output(*row.indices, INT32, {1, naive_topk::kTopK});
            descriptors.rows[i] = {FastllmNaiveKV(*row.key),
                (const int *)liveKeys.cudaData + row.sequence, (BF16 *)scratch.indexQuery.cudaData,
                (float *)scratch.indexScale.cudaData, (unsigned *)scratch.indexScores.cudaData,
                (naive_topk::Workspace *)scratch.topk.cudaData, (int *)row.indices->cudaData,
                row.sequence, row.key->dims[2], row.cache};
            descriptors.rows[i].cpRank = contextRank;
            descriptors.rows[i].cpRanks = contextRanks;
            capacity = std::max(capacity, row.capacity);
        }
        if (cp) capacity = ((capacity + contextRanks * 64 - 1) / (contextRanks * 64)) * 64;
        batch::Quantize<<<dim3(heads, count), dim>>>((const BF16 *)query.cudaData, descriptors);
        const dim3 grid((capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys, count);
        if (hopper) {
            auto kernel = (cacheType == FP8_E4M3 || rows.front().key->kvCacheFp8Indexer) ? batch::HopperScores<true> : batch::HopperScores<false>;
            kernel<<<grid, 128>>>((const BF16 *)weights.cudaData, descriptors);
        }
        else batch::Scores<<<grid, naive_dsa_mma::kThreads>>>((const BF16 *)weights.cudaData, descriptors);
        if (capacity > 16384) {
            const int blocks = std::min(naive_topk::kMaxBlocks, (capacity + 4095) / 4096);
            for (int shift = 24; shift >= 0; shift -= 8) {
                batch::SplitHistogram<><<<dim3(blocks, count), naive_topk::kThreads>>>(descriptors, shift);
                batch::SplitThreshold<><<<count, naive_topk::kThreads>>>(descriptors, blocks, shift);
            }
            batch::SplitCompact<><<<dim3(blocks, count), naive_topk::kThreads>>>(descriptors);
        } else batch::Select<><<<count, naive_topk::kThreads>>>(descriptors);
        batch::Sort<><<<count, naive_topk::kThreads>>>(descriptors);
        if (cp) {
            batch::PackLocal<<<count, naive_topk::kThreads>>>(descriptors);
            packedBatches.push_back(descriptors);
        }
    }
    // Scores may alias the exchange arena. Finish packing every row before
    // any copy can overwrite another row's score slice (CUDA 12.0 batches).
    if (cp) for (int b = 0; b < (int)packedBatches.size(); ++b) {
        int first = b * batch::BatchLimit;
        int count = std::min(batch::BatchLimit, (int)rows.size() - first);
        batch::CopyPacked<<<count, naive_topk::kThreads>>>(packedBatches[b],
            (unsigned *)exchange->cudaData, rows.size(), first);
    }
    CheckLaunch();
    return true;
#else
    return false;
#endif
}

bool FastllmCudaNaiveIndexerContextSupported(const std::vector<int> &devices) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    static thread_local std::map<std::vector<int>, bool> supported;
    auto it = supported.find(devices);
    if (it != supported.end()) return it->second;
    bool ok = devices.size() > 1;
    for (int device : devices) {
        int major = 0;
        ok = ok && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess
            && major >= 8;
    }
    return supported[devices] = ok;
#else
    return false;
#endif
}

void FastllmCudaNaiveMergeDecodeIndexer(const fastllm::Data &liveKeys,
        const std::vector<FastllmNaiveIndexerRow> &rows,
        const fastllm::Data &exchange, int contextRanks) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    namespace batch = naive_indexer_batch;
    for (int first = 0; first < (int)rows.size(); first += batch::BatchLimit) {
        int count = std::min(batch::BatchLimit, (int)rows.size() - first);
        batch::Batch descriptors{};
        for (int i = 0; i < count; ++i) {
            const auto &source = rows[first + i];
            auto &r = descriptors.rows[i];
            r.live = (const int *)liveKeys.cudaData + source.sequence;
            r.workspace = (naive_topk::Workspace *)source.scratch->topk.cudaData;
            r.indices = (int *)source.indices->cudaData;
            r.cpRanks = contextRanks;
            r.merged = (const unsigned *)exchange.cudaData;
            r.mergedRows = rows.size(); r.mergedRow = first + i;
        }
        batch::Select<true><<<count, naive_topk::kThreads>>>(descriptors);
        batch::Sort<true><<<count, naive_topk::kThreads>>>(descriptors);
    }
    CheckLaunch();
#endif
}

namespace {
template <class Kernel> bool BulkKernelSupported(Kernel kernel, size_t shared, int minArch) {
    // Set attributes before capture and cache both success and unsupported images.
    static thread_local std::map<std::pair<int, const void *>, bool> cache;
    auto key = std::make_pair(FastllmCudaGetDevice(), (const void *)kernel);
    auto found = cache.find(key);
    if (found == cache.end()) {
        cudaFuncAttributes attr{};
        bool ok = FastllmCudaRuntimeArch() >= minArch &&
            cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.ptxVersion >= minArch &&
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) == cudaSuccess;
        if (!ok) cudaGetLastError();
        found = cache.emplace(key, ok).first;
    }
    return found->second;
}

template <int Max, bool Ring>
bool BulkAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads, fastllm::Data &output) {
    namespace bulk = naive_attention_bulk;
    const bool fp8 = rows.front().fp8;
    bool asyncFP8 = fp8;
    for (const auto &row : rows) {
        if (row.fp8 != fp8) return false;
        asyncFP8 = asyncFP8 && row.keyStride % 16 == 0;
    }
    auto kernel = fp8 ? (asyncFP8 ? bulk::Decode<Max, Ring, true> : bulk::Decode<Max, Ring, true, false>)
                      : bulk::Decode<Max, Ring, false>;
    size_t shared = asyncFP8 ? sizeof(bulk::Storage<Max, true>) : sizeof(bulk::Storage<Max, false>);
    if (!BulkKernelSupported(kernel, shared, 80)) {
        // GPUs with a smaller shared-memory limit retain the vector-load path.
        if (!asyncFP8) return false;
        kernel = bulk::Decode<Max, Ring, true, false>;
        shared = sizeof(bulk::Storage<Max, false>);
        if (!BulkKernelSupported(kernel, shared, 80)) return false;
    }
    for (int first = 0; first < (int)rows.size(); first += bulk::BatchLimit) {
        int n = std::min(bulk::BatchLimit, (int)rows.size() - first);
        bulk::Batch batch{};
        std::copy_n(rows.begin() + first, n, batch.rows);
        kernel<<<dim3(n, heads / 8), 256, shared>>>(
            (const BF16 *)query.cudaData + (size_t)first * heads * 192, batch, bias,
            (BF16 *)output.cudaData + (size_t)first * heads * 128, heads, kvHeads);
    }
    return true;
}

#if CUDART_VERSION >= 12010
template <int Max>
bool HopperBulkAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, fastllm::Data &output) {
    namespace bulk = naive_attention_bulk;
    if (FastllmCudaRuntimeArch() != 90 ||
        !BulkKernelSupported(bulk::HopperDecode<Max>, sizeof(bulk::TmaStorage<Max>), 90)) return false;
    // Encode every row before launching anything: unsupported TMA layouts can
    // fall back without partially writing output. No H2D copy or global buffer.
    std::vector<bulk::TmaBatch> batches((rows.size() + bulk::TmaBatchLimit - 1) / bulk::TmaBatchLimit);
    for (int first = 0; first < (int)rows.size(); first += bulk::TmaBatchLimit) {
        int n = std::min(bulk::TmaBatchLimit, (int)rows.size() - first);
        if (!bulk::PrepareTmaBatch(batches[first / bulk::TmaBatchLimit], rows.data() + first, n))
            return false;
    }
    for (int first = 0; first < (int)rows.size(); first += bulk::TmaBatchLimit) {
        int n = std::min(bulk::TmaBatchLimit, (int)rows.size() - first);
        bulk::HopperDecode<Max><<<dim3(n, heads / 8), 256, sizeof(bulk::TmaStorage<Max>)>>>(
            (const BF16 *)query.cudaData + (size_t)first * heads * 192,
            batches[first / bulk::TmaBatchLimit], bias,
            (BF16 *)output.cudaData + (size_t)first * heads * 128, heads);
    }
    return true;
}
#endif
} // namespace

bool FastllmCudaNaiveDSAMapsSupported() {
#if CUDART_VERSION >= 12010
    return FastllmCudaRuntimeArch() == 90;
#else
    return false;
#endif
}
bool FastllmCudaNaiveEncodeDSAMaps(void *host, const void *key, const void *value,
                                 int capacity, int keyStride) {
#if CUDART_VERSION >= 12010
    CUtensorMap maps[2];
    if (!naive_attention_bulk::EncodeMaps(maps, key, value, capacity, keyStride)) return false;
    std::memcpy(host, maps, sizeof(maps));
    return true;
#else
    return false;
#endif
}

void FastllmCudaNaiveBatchDSADecode(const fastllm::Data &query,
        const fastllm::Data &sink, const std::vector<FastllmNaiveDSARow> &rows,
        int heads, int kvHeads, int dim, int valueDim,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output) {
    using namespace fastllm;
    namespace dsa = naive_dsa_decode;
    AssertInFastLLM(!rows.empty() && heads > 0 && kvHeads > 0 && heads % kvHeads == 0 &&
        dim > 0 && dim <= 256 && (valueDim == 128 || valueDim == 256) &&
        query.dataType == BFLOAT16, "Unsupported batched DSA decode shape.");
    int count = 0;
    bool sparse = heads >= 4, dense = true;
    bool hopper = FastllmCudaRuntimeArch() == 90 && dim == 192 && valueDim == 128;
    for (const auto &row : rows) {
        AssertInFastLLM(row.key && row.value && row.capacity >= 0 &&
            row.keyStride >= kvHeads * dim && row.selectedCount >= 0,
            "Invalid batched DSA cache.");
        count = std::max(count, row.indices ? row.selectedCount : row.capacity);
        sparse = sparse && row.indices && row.selectedCount == 2048;
        dense = dense && !row.indices;
        hopper = hopper && (uintptr_t)row.value % 16 == 0;
    }
    AssertInFastLLM(count <= 2048, "Batched DSA supports at most 2048 selected keys.");
    // Keep the established small-batch kernels. Eight heads share KV only
    // when the complete GQA group and cache vectors satisfy the MMA layout.
    bool bulk = rows.size() >= 32 && dim == 192 && valueDim == 128 &&
        (heads / kvHeads) % 8 == 0 && count > 0;
    for (const auto &row : rows)
        bulk = bulk && row.keyStride % 8 == 0 && (uintptr_t)row.key % 16 == 0 &&
            (uintptr_t)row.value % 16 == 0;
    if (bulk) {
        Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        bool launched = false;
#if CUDART_VERSION >= 12010
        if (dense && kvHeads == 1)
            launched = count <= 1024 ? HopperBulkAttention<1024>(query, bias, rows, heads, output)
                                     : HopperBulkAttention<2048>(query, bias, rows, heads, output);
#endif
        if (!launched)
            launched = count <= 1024 ? BulkAttention<1024, false>(query, bias, rows, heads, kvHeads, output)
                                     : BulkAttention<2048, false>(query, bias, rows, heads, kvHeads, output);
        if (launched) { CheckLaunch(); return; }
    }
    hopper = hopper && count >= 512;
    if (hopper) {
        // Configure opt-in shared memory once per device, before graph capture.
        static thread_local std::map<int, bool> supported;
        const int device = FastllmCudaGetDevice();
        auto found = supported.find(device);
        if (found == supported.end()) {
            auto small = cudaFuncSetAttribute(dsa::SoftmaxSparseValues<2048, 16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(dsa::SparseStorage<2048, 16>));
            auto large = cudaFuncSetAttribute(dsa::SoftmaxSparseValues<2048, 32>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(dsa::SparseStorage<2048, 32>));
            auto grouped = cudaFuncSetAttribute(dsa::GroupedValues<2048, 2, 8, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(dsa::GroupStorage<2048, 2, 8>));
            auto denseSmall = cudaFuncSetAttribute(dsa::GroupedValues<1024, 4, 16, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(dsa::GroupStorage<1024, 4, 16>));
            auto denseLarge = cudaFuncSetAttribute(dsa::GroupedValues<2048, 4, 16, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(dsa::GroupStorage<2048, 4, 16>));
            bool available = small == cudaSuccess && large == cudaSuccess && grouped == cudaSuccess &&
                denseSmall == cudaSuccess && denseLarge == cudaSuccess;
            if (!available) cudaGetLastError();
            found = supported.emplace(device, available).first;
        }
        hopper = found->second;
    }
    Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
    const int chunkSize = std::min((int)rows.size(), FastllmNaiveDSABatchLimit);
    // Reuse the score workspace. Fusion and sparse partials only use shared memory.
    Output(scratch.attentionScores, FLOAT32, {chunkSize, heads, std::max(1, count)});
    const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    for (int first = 0; first < (int)rows.size(); first += FastllmNaiveDSABatchLimit) {
        int batch = std::min(FastllmNaiveDSABatchLimit, (int)rows.size() - first);
        FastllmNaiveDSABatch descriptors{};
        std::copy_n(rows.begin() + first, batch, descriptors.rows);
        auto *q = (const BF16 *)query.cudaData + (size_t)first * heads * dim;
        auto *out = (BF16 *)output.cudaData + (size_t)first * heads * valueDim;
        auto *scores = (float *)scratch.attentionScores.cudaData;
        if (hopper) {
            const bool sharedHeads = heads == 8 && kvHeads == 1;
            // Larger sparse batches have enough CTAs to amortize a descriptor
            // and page lookup over more keys without changing each dot product.
            if (sharedHeads && sparse && batch >= 8)
                dsa::DirectScores<16, 4><<<dim3(heads, (count + 63) / 64, batch), 128>>>(
                    q, descriptors, scores, heads, kvHeads, count);
            else if (sharedHeads && sparse && batch >= 2)
                dsa::DirectScores<8, 4><<<dim3(heads, (count + 31) / 32, batch), 128>>>(
                    q, descriptors, scores, heads, kvHeads, count);
            else
                dsa::DirectScores<4, 4><<<dim3(heads, (count + 15) / 16, batch), 128>>>(
                    q, descriptors, scores, heads, kvHeads, count);
            // Share V across heads, with wider output tiles for larger dense
            // batches. Each head retains the original softmax/PV arithmetic.
            if (sharedHeads && dense && batch >= 3 && count <= 1024)
                dsa::GroupedValues<1024, 4, 16, false><<<dim3(heads / 4, 8, batch), 1024,
                    sizeof(dsa::GroupStorage<1024, 4, 16>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sharedHeads && dense && batch >= 3)
                dsa::GroupedValues<2048, 4, 16, false><<<dim3(heads / 4, 8, batch), 1024,
                    sizeof(dsa::GroupStorage<2048, 4, 16>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sharedHeads && sparse && (batch == 1 || batch >= 8))
                dsa::GroupedValues<2048, 2, 8, true><<<dim3(heads / 2, 16, batch), 512,
                    sizeof(dsa::GroupStorage<2048, 2, 8>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sharedHeads && dense && count <= 1024 && batch == 1)
                dsa::GroupedValues<1024, 2, 8, false><<<dim3(heads / 2, 16, batch), 512,
                    sizeof(dsa::GroupStorage<1024, 2, 8>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sharedHeads && dense && count <= 1024 && batch == 2)
                dsa::GroupedValues<1024, 4, 8, false><<<dim3(heads / 4, 16, batch), 1024,
                    sizeof(dsa::GroupStorage<1024, 4, 8>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sparse && batch == 1)
                dsa::SoftmaxSparseValues<2048, 16><<<dim3(heads, 8, batch), 512,
                    sizeof(dsa::SparseStorage<2048, 16>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sparse)
                dsa::SoftmaxSparseValues<2048, 32><<<dim3(heads, 4, batch), 1024,
                    sizeof(dsa::SparseStorage<2048, 32>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (count <= 1024)
                dsa::SoftmaxTransposedValues<1024, 16><<<dim3(heads, 8, batch), 256,
                    sizeof(dsa::TransposedStorage<1024, 16>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else
                dsa::SoftmaxTransposedValues<2048, 8><<<dim3(heads, 16, batch), 256,
                    sizeof(dsa::TransposedStorage<2048, 8>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
        } else {
            dim3 grid(heads, (std::max(1, count) + 63) / 64, batch);
            if (dim == 192)
                dsa::Scores<192><<<grid, 256>>>(q, descriptors, scores, heads, kvHeads, dim, count);
            else
                dsa::Scores<><<<grid, 256>>>(q, descriptors, scores, heads, kvHeads, dim, count);
            dsa::Softmax<<<batch * heads, 256>>>(scores, bias, descriptors, heads, count);
            if (valueDim == 128)
                dsa::Values<128><<<dim3(heads, 16, batch), 128>>>(scores, descriptors, out,
                    heads, kvHeads, count);
            else
                dsa::Values<256><<<dim3(heads, 32, batch), 128>>>(scores, descriptors, out,
                    heads, kvHeads, count);
        }
    }
    CheckLaunch();
}

void FastllmCudaNaiveBatchSWADecode(const fastllm::Data &query,
        const fastllm::Data &sink, const std::vector<FastllmNaiveSWARow> &rows,
        const fastllm::Data &liveKeys, int heads, int kvHeads,
        fastllm::Data &output) {
    using namespace fastllm;
    AssertInFastLLM(!rows.empty() && heads > 0 && kvHeads > 0 && heads % kvHeads == 0 &&
        query.dataType == BFLOAT16 && query.dims == std::vector<int>({1, (int)rows.size(), heads * kSwaQkDim}) &&
        liveKeys.dataType == INT32 && liveKeys.Count(0) == rows.size(), "Invalid batched SWA decode shape.");
    Output(output, BFLOAT16, {1, (int)rows.size(), heads * kSwaValueDim});
    if (rows.size() >= 32 && (heads / kvHeads) % 8 == 0) {
        std::vector<FastllmNaiveDSARow> descriptors(rows.size());
        bool aligned = true;
        for (int i = 0; i < (int)rows.size(); ++i) {
            auto &r = descriptors[i];
            r.key = rows[i].key; r.value = rows[i].value; r.cache = rows[i].cache; r.fp8 = rows[i].fp8;
            r.liveKeys = (const int *)liveKeys.cudaData + i;
            r.capacity = kSwaWindow; r.keyStride = kvHeads * kSwaQkDim;
            aligned = aligned && r.key && r.value && (uintptr_t)r.key % 16 == 0 && (uintptr_t)r.value % 16 == 0;
        }
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        if (aligned && BulkAttention<128, true>(query, bias, descriptors, heads, kvHeads, output)) {
            CheckLaunch(); return;
        }
    }
    const int threads = heads == 8 || heads == 16 ? 1024 : kSwaThreads;
    auto kernel = threads == 1024
        ? AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, 1024, kSwaOutputTile, true>
        : AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, kSwaThreads, kSwaOutputTile, true>;
    const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    for (int first = 0; first < (int)rows.size(); first += kSwaBatchLimit) {
        const int count = std::min(kSwaBatchLimit, (int)rows.size() - first);
        BatchSwaKeys keys{};
        std::copy_n(rows.begin() + first, count, keys.rows);
        keys.length = (const int *)liveKeys.cudaData + first;
        kernel<<<dim3(heads, count, kSwaValueDim / kSwaOutputTile), threads>>>(
            (const BF16 *)query.cudaData + (size_t)first * heads * kSwaQkDim,
            nullptr, nullptr, bias, (BF16 *)output.cudaData + (size_t)first * heads * kSwaValueDim,
            heads, kvHeads, kvHeads * kSwaQkDim, keys);
    }
    CheckLaunch();
}

void FastllmCudaNaiveDecodeAttention(const fastllm::Data &query,
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const fastllm::Data &sink,
        const fastllm::Data &liveKeys, int capacity, int heads, int kvHeads,
        int dim, int valueDim, int window,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output, bool ring) {
    using namespace fastllm;
    AssertInFastLLM(!ring || (window == kSwaWindow && dim == kSwaQkDim && valueDim == kSwaValueDim),
                   "Unsupported SWA ring attention shape.");
    DecodeKeys keys{(const int *)liveKeys.cudaData, window};
    auto *q = (const BF16 *)query.cudaData;
    FastllmNaiveKV k(key);
    FastllmNaiveKV v(value);
    auto *selected = indices.dims.empty() ? nullptr : (const int *)indices.cudaData;
    auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    Output(output, BFLOAT16, {1, 1, heads * valueDim});
    auto *out = (BF16 *)output.cudaData;
    if (dim == kSwaQkDim && valueDim == kSwaValueDim &&
        (window == kSwaWindow || (!window && !selected && capacity > 0 && capacity <= 256))) {
        // Bound the live length to the capacity captured for this graph.
        DecodeKeys shortKeys{(const int *)liveKeys.cudaData, window ? window : capacity};
        LaunchShortAttention<false>(q, k, v, bias, out, 1, heads, kvHeads,
            key.dims[2], shortKeys, window != 0, ring);
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
            if (valueDim == 128 && heads <= 16 && capacity <= 2048 &&
                FastllmCudaRuntimeArch() == 90) {
                AttentionValuesDecodeDense<8><<<dim3(heads, 16), 128>>>(
                    scores, v, out, (const int *)liveKeys.cudaData, capacity, heads, kvHeads);
            } else {
                AttentionValuesTiled<<<dim3(heads, 1, (valueDim + 31) / 32), 256>>>(
                    scores, v, selected, out, heads, kvHeads, valueDim,
                    keys, keys, capacity - 1, 0, true);
            }
        } else {
            bool split = heads >= 4 && valueDim == kDecodePVValueDim;
            bool grouped = heads % kvHeads == 0 && (heads / kvHeads) % 4 == 0;
            bool shared = grouped && (heads / kvHeads) % kDecodeSharedHeads == 0;
            if (shared && dim == kDecodeQkDim && key.dims[2] % 8 == 0 && (size_t)k.base % 16 == 0) {
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
                if (shared && (size_t)v.base % 16 == 0) {
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
    if (key.kvCacheFp8Indexer) {
        WriteIndexerCache<<<rows, 128>>>(FastllmNaiveKV(key), (const BF16 *)newKey.cudaData,
            kc, 0, (const int *)liveKeys.cudaData, FastllmNaiveKV(value), (const BF16 *)newValue.cudaData, vc);
        CheckLaunch();
        return;
    }
    AppendVerifyKV<<<dim3((kc + vc + 255) / 256, rows), 256>>>({key.cudaData, key.dataType == fastllm::FP8_E4M3},
        {value.cudaData, value.dataType == fastllm::FP8_E4M3}, (const BF16 *)newKey.cudaData,
        (const BF16 *)newValue.cudaData, (const int *)liveKeys.cudaData, kc, vc, window);
    CheckLaunch();
}

namespace {
__global__ void DraftEmbedding(const float *ids, int step, const BF16 *weight,
                               BF16 *latent, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) latent[col] = weight[(size_t)(int)ids[step] * width + col];
}

__device__ void Top1WarpMax(float &score, int &id) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, score, offset);
        int otherId = __shfl_down_sync(0xffffffffu, id, offset);
        if ((threadIdx.x & 31) + offset < 32 && FastllmNaiveTop1Better(other, otherId, score, id)) {
            score = other; id = otherId;
        }
    }
}
__device__ void Top1BlockMax(float &score, int &id) {
    __shared__ float scores[8];
    __shared__ int ids[8];
    Top1WarpMax(score, id);
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (!lane) { scores[warp] = score; ids[warp] = id; }
    __syncthreads();
    if (!warp) {
        score = lane < 8 ? scores[lane] : -INFINITY;
        id = lane < 8 ? ids[lane] : INT_MAX;
        Top1WarpMax(score, id);
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
        if (score > -INFINITY && FastllmNaiveTop1Better(score, i, best, bestId)) {
            best = score; bestId = i;
        }
    }
    Top1BlockMax(best, bestId);
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
        if (item.y > -INFINITY && FastllmNaiveTop1Better(item.y, (int)item.x, best, bestId)) {
            best = item.y; bestId = (int)item.x;
        }
    }
    Top1BlockMax(best, bestId);
    if (!threadIdx.x) ids[step + 1] = bestId == INT_MAX ? 0.0f : (float)bestId;
}

__global__ void LogitsTop1Partial(const float *logits, float2 *partial, int vocab, int offset) {
    float best = -INFINITY;
    int bestId = INT_MAX;
    for (int i = blockIdx.x * 1024 + threadIdx.x;
         i < min((int)(blockIdx.x + 1) * 1024, vocab); i += 256) {
        float score = logits[(size_t)blockIdx.y * vocab + i];
        int id = offset + i;
        if (score > -INFINITY && FastllmNaiveTop1Better(score, id, best, bestId)) {
            best = score; bestId = id;
        }
    }
    Top1BlockMax(best, bestId);
    if (!threadIdx.x)
        partial[(size_t)blockIdx.y * gridDim.x + blockIdx.x] =
            make_float2(bestId == INT_MAX ? -1.0f : (float)bestId, best);
}
__global__ void LogitsTop1Finish(const float2 *partial, float2 *output, int count) {
    float best = -INFINITY;
    int bestId = INT_MAX;
    for (int i = threadIdx.x; i < count; i += 256) {
        float2 item = partial[(size_t)blockIdx.x * count + i];
        if (item.y > -INFINITY && FastllmNaiveTop1Better(item.y, (int)item.x, best, bestId)) {
            best = item.y; bestId = (int)item.x;
        }
    }
    Top1BlockMax(best, bestId);
    if (!threadIdx.x)
        output[blockIdx.x] = make_float2(bestId == INT_MAX ? -1.0f : (float)bestId, best);
}
// A tile sorts 1024 scores, retaining only its first K. The second stage
// merges sorted tile heads; no vocabulary-sized host transfer or allocation.
__global__ void LogitsTopKPartial(const float *logits, float2 *partial,
                                  int vocab, int offset, int count, float scale) {
    using Sort = cub::BlockRadixSort<unsigned long long, 256, 4>;
    __shared__ typename Sort::TempStorage storage;
    unsigned long long keys[4];
    for (int j = 0; j < 4; ++j) {
        int column = blockIdx.x * 1024 + threadIdx.x * 4 + j;
        keys[j] = 0;
        if (column < vocab) {
            float value = logits[(size_t)blockIdx.y * vocab + column] * scale;
            // Canonicalize signed zero for the CPU sampler's equality rule.
            unsigned bits = __float_as_uint(value == 0 ? 0.0f : value);
            unsigned ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
            if (!isnan(value))
                keys[j] = ((unsigned long long)ordered << 32) | (0xffffffffu - (offset + column));
        }
    }
    Sort(storage).SortDescending(keys);
    for (int j = 0; j < 4; ++j) {
        int index = threadIdx.x * 4 + j;
        if (index < count) {
            auto key = keys[j];
            unsigned ordered = key >> 32;
            unsigned bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
            partial[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * count + index] =
                key ? make_float2((float)(0xffffffffu - (unsigned)key), __uint_as_float(bits)) :
                      make_float2(-1.0f, -INFINITY);
        }
    }
}
__device__ bool SamplingBetter(float score, int id, float best, int bestId) {
    return score > best || (score == best && id < bestId);
}
__global__ void LogitsTopKFinish(const float2 *partial, float2 *output, int blocks, int count) {
    // Each lane owns one sorted tile. 256 tiles cover vocabularies up to 262144.
    __shared__ float scores[8];
    __shared__ int ids[8];
    __shared__ int winner;
    int head = 0, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const float2 *tile = partial + ((size_t)blockIdx.x * blocks + threadIdx.x) * count;
    for (int k = 0; k < count; ++k) {
        float2 value = threadIdx.x < blocks && head < count ? tile[head] : make_float2(-1, -INFINITY);
        float score = value.y;
        int id = value.x >= 0 ? (int)value.x : INT_MAX;
        for (int delta = 16; delta; delta >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, score, delta);
            int otherId = __shfl_down_sync(0xffffffffu, id, delta);
            if (lane + delta < 32 && SamplingBetter(other, otherId, score, id)) { score = other; id = otherId; }
        }
        if (!lane) { scores[warp] = score; ids[warp] = id; }
        __syncthreads();
        if (!warp) {
            score = lane < 8 ? scores[lane] : -INFINITY;
            id = lane < 8 ? ids[lane] : INT_MAX;
            for (int delta = 16; delta; delta >>= 1) {
                float other = __shfl_down_sync(0xffffffffu, score, delta);
                int otherId = __shfl_down_sync(0xffffffffu, id, delta);
                if (lane + delta < 32 && SamplingBetter(other, otherId, score, id)) { score = other; id = otherId; }
            }
            if (!lane) {
                winner = id;
                output[(size_t)blockIdx.x * count + k] = make_float2(id == INT_MAX ? -1.0f : (float)id, score);
            }
        }
        __syncthreads();
        if (value.x >= 0 && (int)value.x == winner) ++head;
        // No lane may overwrite winner before every lane consumes it.
        __syncthreads();
    }
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

void FastllmCudaNaiveLogitsSelect(const fastllm::Data &logits, int vocabOffset,
    int count, bool greedy, float invTemperature,
    fastllm::Data &partial, fastllm::Data &output) {
    using namespace fastllm;
    AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
        !logits.dims.empty() && logits.cudaData &&
        FastllmNaiveCanSelectLogits(logits.dims.back(), vocabOffset, count, greedy) &&
        (greedy || (std::isfinite(invTemperature) && invTemperature > 0)),
        "Invalid Naive vocabulary shard for logits selection.");
    const int vocab = logits.dims.back(), rows = logits.Count(0) / vocab;
    const int blocks = (vocab + 1023) / 1024;
    Output(partial, FLOAT32, {rows, blocks, count, 2});
    Output(output, FLOAT32, {rows, count * 2});
    if (greedy) {
        LogitsTop1Partial<<<dim3(blocks, rows), 256>>>((const float *)logits.cudaData,
            (float2 *)partial.cudaData, vocab, vocabOffset);
        LogitsTop1Finish<<<rows, 256>>>((const float2 *)partial.cudaData,
            (float2 *)output.cudaData, blocks);
    } else {
        LogitsTopKPartial<<<dim3(blocks, rows), 256>>>((const float *)logits.cudaData,
            (float2 *)partial.cudaData, vocab, vocabOffset, count, invTemperature);
        LogitsTopKFinish<<<rows, 256>>>((const float2 *)partial.cudaData,
            (float2 *)output.cudaData, blocks, count);
    }
    CheckLaunch();
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
    auto *q = (const BF16 *)query.cudaData;
    FastllmNaiveKV k(key), v(value);
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
                (const BF16 *)scratch.indexQuery.cudaData, FastllmNaiveKV(packedKeys),
                (const float *)scratch.indexScale.cudaData, (const BF16 *)weights.cudaData,
                scoreOutput, stride, keys, capacity - 1);
    } else
#endif
    {
        Output(scratch.indexQuery, FLOAT32, {rows * 16, 128});
        RoundIndexer<<<rows * 16, 128>>>((const BF16 *)query.cudaData,
            (float *)scratch.indexQuery.cudaData, 128, 0, fp8Query);
        IndexScoresDecode<IndexerTopKOutput, DecodeKeys, true><<<dim3((capacity + 7) / 8, rows), 256>>>(
            (const float *)scratch.indexQuery.cudaData, FastllmNaiveKV(packedKeys),
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
        FastllmNaiveKV k(key), v(value);
        auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        auto *out = (BF16 *)output.cudaData;
        LaunchShortAttention<true>(q, k, v, bias, out, rows, heads, kvHeads,
            key.dims[2], keys, window != 0);
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
                FastllmNaiveKV(key), selected, scores, heads, kvHeads, dim,
                key.dims[2], keys, count, capacity - 1, true);
        AttentionSoftmax<<<rows * heads, 256>>>(scores, bias, heads, count);
        AttentionValuesDecodeGrouped<4, DecodeKeys, true>
            <<<dim3(heads / 4, kDecodePVParts, rows), kDecodePVValueDim>>>(scores,
                FastllmNaiveKV(value), selected, partial, heads, kvHeads, keys, capacity - 1, true);
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
                FastllmNaiveKV(key), FastllmNaiveKV(value),
                (BF16 *)scratch.windowKey.cudaData, (BF16 *)scratch.windowValue.cudaData,
                (const int *)liveKeys.cudaData, row, kc, vc, window);
        }
        FastllmCudaNaiveDecodeAttention(q, window ? scratch.windowKey : key,
            window ? scratch.windowValue : value, selected, sink, live, capacity,
            heads, kvHeads, dim, valueDim, window, scratch, out);
    }
}

namespace {
struct DraftTPGroup { std::vector<int> devices; std::vector<ncclComm_t> comms; };
}
void *FastllmCudaNaiveDraftTPCreate(const std::vector<int> &devices) {
    if (devices.size() < 2 || devices.size() > (size_t)FastllmCudaGetDeviceCount()) return nullptr;
    for (size_t i = 0; i < devices.size(); ++i)
        if (devices[i] < 0 || std::find(devices.begin(), devices.begin() + i, devices[i]) != devices.begin() + i)
            return nullptr;
    auto *group = new DraftTPGroup{devices, std::vector<ncclComm_t>(devices.size())};
    int previous = FastllmCudaGetDevice();
    if (ncclCommInitAll(group->comms.data(), devices.size(), devices.data()) != ncclSuccess) {
        for (auto comm : group->comms) if (comm) ncclCommAbort(comm);
        delete group; group = nullptr;
    }
    FastllmCudaSetDevice(previous);
    return group;
}
void FastllmCudaNaiveDraftTPDestroy(void *opaque) {
    if (!opaque) return;
    auto *group = (DraftTPGroup *)opaque;
    int previous = FastllmCudaGetDevice();
    for (size_t i = 0; i < group->comms.size(); ++i) {
        FastllmCudaSetDevice(group->devices[i]);
        cudaDeviceSynchronize();
        ncclCommDestroy(group->comms[i]);
    }
    FastllmCudaSetDevice(previous);
    delete group;
}
void FastllmCudaNaiveDraftTPReduce(void *opaque, int rank, fastllm::Data &data) {
    auto *group = (DraftTPGroup *)opaque;
    fastllm::AssertInFastLLM(group && rank >= 0 && rank < (int)group->devices.size() &&
        data.dataType == fastllm::BFLOAT16 && data.dataDeviceIds == std::vector<int>{group->devices[rank]},
        "Invalid draft TP reduction.");
    fastllm::AssertInFastLLM(ncclAllReduce(data.cudaData, data.cudaData, data.Count(0), ncclBfloat16,
        ncclSum, group->comms[rank], cudaStreamPerThread) == ncclSuccess, "Draft TP reduction failed.");
}
