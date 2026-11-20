#include "devices/cuda/fastllm-cuda-norm.cuh"
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
#include "naive-n05-topk.cuh"
#include "naive-n05-topk-compact-sort.cuh"
#include "naive-n05-topk-compact-select.cuh"
#include "../linear/fastllm-native-lowbit-prefill.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-dsa-decode.cuh"
#include "naive-n05-attention-small.cuh"
#include "naive-n05-verify-dense.cuh"
#include "naive-n05-attention-prefetch.cuh"
#include "naive-n05-attention-hopper.cuh"
#include "naive-n05-decode-blackwell.cuh"
#include "naive-n05-prefill.cuh"
#include "naive-n05-prefill-hopper.cuh"
#include "naive-n05-prefill-split.cuh"
#include "naive-n05-swa-query.cuh"
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

// FlashInfer synchronization headers require SM70+. Keep the host pass
// and SM70+ launch stubs; runtime dispatch excludes BF16 attention below SM80.
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && \
    (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
#define FASTLLM_NAIVE_SWA_FLASHINFER
#include "naive-n05-swa-flashinfer.cuh"
#endif

// Scalar batching and TopK do not require Tensor Cores. Keep them in legacy
// builds as well; the descriptor kernels use the portable SM70 argument ABI.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define FASTLLM_NAIVE_BATCH_INDEXER
#include "naive-n05-indexer-batch.cuh"
#endif

#if !defined(FASTLLM_CUDA_LEGACY_ONLY)
// Keep host launch stubs and device definitions in every architecture pass.
// MMA bodies and runtime dispatch already guard the SM80-only instructions.
#define FASTLLM_NAIVE_DSA_MMA
#include "naive-n05-dsa-mma.cuh"
#ifdef FASTLLM_NAIVE_BATCH_INDEXER
#include "naive-n05-indexer-staged.cuh"
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM100
#include "naive-n05-indexer-sm100.cuh"
#endif
#if CUDART_VERSION >= 12040
#include "naive-n05-indexer-compact.cuh"
#endif
#include "naive-n05-indexer-sortpack.cuh"
#include "naive-n05-indexer-wide-select.cuh"
#include "naive-n05-context-merge.cuh"
#include "naive-n05-context-merge-chunk.cuh"
#endif
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

// Each lane owns two adjacent partials of the reference GEMV. Combining the
// pair before the warp reduction preserves every FP32 accumulation and tie.
template<int Rows> __global__ void RouterPairs4096(const BF16 *input, const float *weight,
                                                    float *output, int rows, int experts) {
    const int t = threadIdx.x, first = blockIdx.y * Rows, expert = blockIdx.x;
    __shared__ float sums[Rows][8];
    float partial[Rows][2]{};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = t * 2 + i * 256;
        const float2 w = *reinterpret_cast<const float2 *>(weight + size_t(expert) * 4096 + col);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const auto *x = reinterpret_cast<const __nv_bfloat162 *>(
                input + size_t(min(first + r, rows - 1)) * 4096 + col);
            const float2 v = __bfloat1622float2(*x);
            partial[r][0] = fmaf(v.x, w.x, partial[r][0]);
            partial[r][1] = fmaf(v.y, w.y, partial[r][1]);
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float value = partial[r][0] + partial[r][1];
#pragma unroll
        for (int d = 1; d < 16; d *= 2) value += __shfl_down_sync(~0u, value, d);
        if (t % 16 == 0) sums[r][t / 16] = value;
    }
    __syncthreads();
    const int r = t / 32, lane = t % 32;
    if (r >= Rows) return;
    float value = lane < 8 ? sums[r][lane] : 0.f;
#pragma unroll
    for (int d = 1; d < 8; d *= 2) value += __shfl_down_sync(~0u, value, d);
    if (lane == 0 && first + r < rows)
        output[size_t(first + r) * experts + expert] = 1.0 / (1.0 + expf(-(value + 0.0f)));
}

// Keep the full argument block below the legacy 4 KiB CUDA limit.
constexpr int kDraftKVLayersPerLaunch = 80;
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
    int length[kDraftKVLayersPerLaunch], rows[kDraftKVLayersPerLaunch], start[kDraftKVLayersPerLaunch];
};
static_assert(sizeof(DraftKVDestinations) + 64 <= 4096, "Draft KV argument limit");
__device__ void DraftWriteHead(const BF16 *key, const float *norm, BF16 *outK, BF16 *outV,
        int position, int width, int dim, float eps, float theta) {
    const int t = threadIdx.x;
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
        normalized[t] = __float2bfloat16(RoundBF16(value * rsqrtf(sums[0] / dim + eps)) * norm[t]);
    __syncthreads();
    if (t < dim / 2) {
        float angle = (float)(position) * powf(theta, -2.0f * t / dim);
        float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
        float a = (float)normalized[t], b = (float)normalized[t + dim / 2];
        outK[t] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
        outK[t + dim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
    }
    if (t < dim) outV[t] = key[width + t];
}
__global__ void DraftNormRopeWrite(DraftKVDestinations dst, int rows,
        int heads, int dim, float eps, float theta) {
    int head = blockIdx.x % heads, row = (blockIdx.x / heads) % rows;
    int slot = blockIdx.x / (heads * rows);
    if (row >= dst.rows[slot]) return;
    int width = heads * dim;
    auto key = dst.raw[slot] + (size_t)row * 2 * width + head * dim;
    size_t out = (size_t)(dst.length[slot] + row) * width + head * dim;
    DraftWriteHead(key, dst.norm[slot], dst.key[slot] + out, dst.value[slot] + out,
                   dst.start[slot] + row, width, dim, eps, theta);
}
__global__ void DraftNormRopeWriteAccepted(const BF16 *raw, const FastllmNaiveDraftKVSlot *slots,
        const int *emitted, int batch, int rows, int heads, int dim, float eps, float theta) {
    int head = blockIdx.x % heads, token = blockIdx.x / heads;
    int request = token / rows, row = token % rows, layer = blockIdx.y;
    // The anchor is committed even when every draft token was rejected.
    if (row > emitted[request]) return;
    const auto slot = slots[(size_t)layer * batch + request];
    if (slot.targetPair && !layer && !row && !head && !threadIdx.x) {
        slot.targetPair[0] = slot.position;
        slot.targetPair[1] = emitted[request] + 1;
    }
    const int width = heads * dim;
    const auto key = raw + ((size_t)layer * batch * rows + token) * 2 * width + head * dim;
    size_t out = (size_t)row * width + head * dim;
    DraftWriteHead(key, slot.norm, (BF16 *)slot.key + out, (BF16 *)slot.value + out,
                   slot.position + row, width, dim, eps, theta);
}
void Output(fastllm::Data &out, fastllm::DataType type, const std::vector<int> &dims) {
    out.dataType = type;
    out.Resize(dims);
    out.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    out.Allocate(false);
}
// Return zero without touching output when the router fast path is unsupported.
int RouterRows(const fastllm::Data &input, const fastllm::Data &weight,
               const fastllm::Data &output, fastllm::DataType type = fastllm::BFLOAT16) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    auto onDevice = [&](const Data &x) {
        return x.dataDevice == DataDevice::CUDA && x.dataDeviceIds.size() == 1 &&
            x.dataDeviceIds[0] == device && x.cudaData;
    };
    if (&input == &output || &weight == &output ||
        (output.cudaData && (output.cudaData == input.cudaData || output.cudaData == weight.cudaData)) ||
        !onDevice(input) || !onDevice(weight) || input.dataType != type ||
        weight.dataType != FLOAT32 || input.dims.empty() || input.dims.back() != 4096 ||
        weight.dims.size() != 2 || weight.dims[1] != 4096 || weight.dims[0] <= 0 ||
        weight.strides.size() != 2 || weight.strides[0] != 4096 || weight.strides[1] != 1 ||
        input.strides.size() != input.dims.size()) return 0;
    int64_t count = 1;
    for (int axis = (int)input.dims.size() - 1; axis >= 0; --axis) {
        if (input.strides[axis] != count || input.dims[axis] <= 0) return 0;
        count *= input.dims[axis];
        // Larger batches use four rows per CTA. Bound by grid.y,
        // not a particular speculative block size or request concurrency.
        if (count > (int64_t)4 * 65535 * 4096) return 0;
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

template <typename T>
__global__ void CommitSwaVerify(T *key, T *value, int kc, int vc, int past) {
    int column = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    const int slot = (past + row) & (kSwaWindow - 1);
    if (column < kc) key[(size_t)slot * kc + column] = key[(size_t)(kSwaWindow + row) * kc + column];
    else if ((column -= kc) < vc)
        value[(size_t)slot * vc + column] = value[(size_t)(kSwaWindow + row) * vc + column];
}

template <typename T, bool Dynamic = false, int MaxRows = kTrimMaxRows, int TileBytes = kTrimTileBytes,
          bool DynamicShared = false>
__device__ void TrimCachePairTiledBody(T *key, T *value, int keyColumns,
                                 int valueColumns, int drop, int keep, const int *decodeKeys = nullptr) {
    if constexpr (Dynamic) { if (*decodeKeys <= keep) return; }
    constexpr int columns = TileBytes / sizeof(T);
    __shared__ T localRows[DynamicShared ? 1 : MaxRows * columns];
    extern __shared__ __align__(16) unsigned char sharedRows[];
    T *rows = DynamicShared ? reinterpret_cast<T *>(sharedRows) : localRows;
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

template <typename T, bool Dynamic = false, int MaxRows = kTrimMaxRows, int TileBytes = kTrimTileBytes>
__global__ void TrimCachePairTiled(T *key, T *value, int kc, int vc, int drop, int keep, const int *live = nullptr) {
    TrimCachePairTiledBody<T, Dynamic, MaxRows, TileBytes>(key, value, kc, vc, drop, keep, live);
}
struct DraftTrimBatch {
    BF16 *key[kDraftKVLayersPerLaunch], *value[kDraftKVLayersPerLaunch];
    int width[kDraftKVLayersPerLaunch], drop[kDraftKVLayersPerLaunch], keep[kDraftKVLayersPerLaunch];
};
template<int MaxRows, int TileBytes>
__global__ void BatchDraftTrim(DraftTrimBatch batch) {
    const int i = blockIdx.y;
    if (!batch.drop[i]) return;
    TrimCachePairTiledBody<BF16, false, MaxRows, TileBytes>(batch.key[i], batch.value[i],
        batch.width[i], batch.width[i], batch.drop[i], batch.keep[i]);
}
// Wide vector transactions amortize the long-window suffix move. Each CTA
// still owns complete columns, so even a one-row overlapping move is safe.
template<int TileBytes>
__global__ void BatchDraftTrimVector(DraftTrimBatch batch) {
    const int i = blockIdx.y;
    if (!batch.drop[i]) return;
    TrimCachePairTiledBody<uint4, false, 1024, TileBytes, true>(
        (uint4 *)batch.key[i], (uint4 *)batch.value[i], batch.width[i] / 8,
        batch.width[i] / 8, batch.drop[i], batch.keep[i]);
}
__global__ void BatchDraftTrimColumns(DraftTrimBatch batch) {
    const int i = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (!batch.drop[i] || col >= 2 * batch.width[i]) return;
    BF16 *data = col < batch.width[i] ? batch.key[i] : batch.value[i];
    const int column = col % batch.width[i];
    for (int row = 0; row < batch.keep[i]; ++row)
        data[(size_t)row * batch.width[i] + column] = data[(size_t)(row + batch.drop[i]) * batch.width[i] + column];
}
// One CTA commits the accepted rows of one request/layer. The verifier
// already installed the stable table on this stream. Unused rows have count 0.
__global__ void CommitBoundSwa(const void *const *bindings, const int2 *accepted,
        uint64_t slidingMask, int tableStride, int kc, int vc) {
    const int request = blockIdx.y, layer = blockIdx.z;
    const int2 item = accepted[request];
    if (!item.y || !(slidingMask & (uint64_t(1) << layer))) return;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= kc + vc) return;
    const auto *entry = bindings + size_t(layer) * tableStride + request * 4;
    auto *ptr = (uint4 *)entry[column >= kc];
    const int width = column < kc ? kc : vc, local = column < kc ? column : column - kc;
    for (int row = 0; row < item.y; ++row)
        ptr[size_t((item.x + row) & (kSwaWindow - 1)) * width + local] =
            ptr[size_t(kSwaWindow + row) * width + local];
}

constexpr int kSwaCommitsPerLaunch = 112;
struct SwaCommitBatch {
    void *key[kSwaCommitsPerLaunch], *value[kSwaCommitsPerLaunch];
    int kc[kSwaCommitsPerLaunch], vc[kSwaCommitsPerLaunch], past[kSwaCommitsPerLaunch],
        count[kSwaCommitsPerLaunch], fp8[kSwaCommitsPerLaunch];
};
static_assert(sizeof(SwaCommitBatch) <= 4096, "SWA commit argument limit");
template<typename T>
__device__ void CommitSwaElement(SwaCommitBatch batch, int i, int row, int col) {
    T *key = (T *)batch.key[i], *value = (T *)batch.value[i];
    const int slot = (batch.past[i] + row) & (kSwaWindow - 1);
    if (col < batch.kc[i]) key[(size_t)slot * batch.kc[i] + col] = key[(size_t)(kSwaWindow + row) * batch.kc[i] + col];
    else if ((col -= batch.kc[i]) < batch.vc[i])
        value[(size_t)slot * batch.vc[i] + col] = value[(size_t)(kSwaWindow + row) * batch.vc[i] + col];
}
__global__ void BatchCommitSwaVerify(SwaCommitBatch batch) {
    int i = blockIdx.z, row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch.count[i]) return;
    if (batch.fp8[i]) CommitSwaElement<uint8_t>(batch, i, row, col);
    else CommitSwaElement<BF16>(batch, i, row, col);
}

template <typename T>
void LaunchTrimCachePair(T *key, T *value, int keyColumns,
                        int valueColumns, int drop, int keep) {
    if (keep <= kTrimMaxRows) {
        constexpr int columns = kTrimTileBytes / sizeof(T);
        int blocks = (keyColumns + columns - 1) / columns +
                     (valueColumns + columns - 1) / columns;
        TrimCachePairTiled<<<blocks, 256>>>(key, value, keyColumns, valueColumns, drop, keep);
    } else if (keep <= 1024) {
        // Stage each disjoint column tile before overwriting its overlapping
        // source. A 32-byte tile keeps the draft window within 32 KiB of shared
        // memory and exposes parallelism across columns on every architecture.
        constexpr int columns = 32 / sizeof(T);
        int blocks = (keyColumns + columns - 1) / columns +
                     (valueColumns + columns - 1) / columns;
        TrimCachePairTiled<T, false, 1024, 32><<<blocks, 256>>>(
            key, value, keyColumns, valueColumns, drop, keep);
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

using naive_topk::TopKOrder;

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

using naive_topk::SelectTopKCompact;
using naive_topk::SelectTopKCompactRow;

// Keep long/short contexts on the original compiled kernel. The sample is
// useful when its upper tail has enough observations to bracket K reliably.
void LaunchSelectTopKCompact(const float *scores, unsigned long long *selected, int *offsets,
                            int queries, int keys, int queryStart, int topK, int capacity,
                            int partRows = 0, int gap = 0) {
    if ((int64_t)keys >= 4LL * capacity && (int64_t)keys <= 32LL * capacity && topK >= 256) {
        // A wider vectorized scan amortizes its histogram over a long live
        // prefix. The sample only proposes a cutoff; all candidates are checked.
        const int arch = FastllmCudaRuntimeArch();
        const bool wide = queries >= 128 && ((topK > capacity / 2 && keys >= 32768 && arch == 90) ||
            (topK == 2048 && capacity == 4096 && queryStart >= 32768 && (arch == 100 || arch == 103)));
        if (wide)
            SelectTopKCompact<true, 2, 1024><<<queries, 1024>>>(scores, selected, offsets, queries,
                keys, queryStart, topK, capacity, partRows, gap);
        else if (topK > capacity / 2)
            SelectTopKCompact<true, 4, 512><<<queries, 512>>>(scores, selected, offsets, queries,
                keys, queryStart, topK, capacity, partRows, gap);
        else
            SelectTopKCompact<true><<<queries, 256>>>(scores, selected, offsets, queries,
                keys, queryStart, topK, capacity, partRows, gap);
    } else {
        SelectTopKCompact<false><<<queries, 256>>>(scores, selected, offsets, queries,
            keys, queryStart, topK, capacity, partRows, gap);
    }
}
#ifdef FASTLLM_NAIVE_DSA_MMA
__global__ void SelectBatchTopK(naive_dsa_mma::PrefillIndexerBatch batch, const float *scores,
                                unsigned long long *selected, int *offsets) {
    int row = blockIdx.x;
    const auto &e = batch.Find(row);
    int local = row - e.rowOffset;
    int count = e.past + (!e.part || local < e.part ? e.first + local : e.second + local - e.part) + 1;
    SelectTopKCompactRow<false>(scores + e.scoreOffset + (size_t)local * e.keys, selected, offsets, batch.rows, count, 2048,
                         4096);
}
#endif
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

using naive_topk::CompactTopKSort;
using naive_topk::SortTopKCompactRow;
using naive_topk::SortTopKCompact;

#ifdef FASTLLM_NAIVE_DSA_MMA
__global__ void SortBatchTopK(naive_dsa_mma::PrefillIndexerBatch batch, const unsigned long long *input,
                              const int *ends) {
    union Storage {
        CompactTopKSort<8>::TempStorage small;
        CompactTopKSort<12>::TempStorage medium;
        CompactTopKSort<16>::TempStorage large;
    };
    __shared__ Storage storage;
    const auto &e = batch.Find(blockIdx.x);
    const int count = ends[batch.rows + blockIdx.x] - blockIdx.x * 4096;
    const int row = blockIdx.x - e.rowOffset;
    if (count <= 2048)
        SortTopKCompactRow<8>(input, ends, e.output, batch.rows, 4096, 2048, e.keys, storage.small, row);
    else if (count <= 3072)
        SortTopKCompactRow<12>(input, ends, e.output, batch.rows, 4096, 2048, e.keys, storage.medium, row);
    else
        SortTopKCompactRow<16>(input, ends, e.output, batch.rows, 4096, 2048, e.keys, storage.large, row);
}
#endif
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

#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
// Packed 192/128 projection with a 64-column split-half RoPE. Eight
// consecutive vector lanes cover both rotary halves of each Q/K head.
__global__ void RopePackedPrefill(BF16 *q, BF16 *k, BF16 *v, const float *positions,
        int heads, int kvHeads, float theta, BF16 scale, const BF16 *packed) {
    __shared__ __align__(16) BF16 cosine[32], sine[32];
    int token = blockIdx.x, t = threadIdx.x;
    if (t < 32) {
        float angle = positions[token] * powf(theta, -2.f * t / 64);
        cosine[t] = __float2bfloat16_rn(cosf(angle));
        sine[t] = __float2bfloat16_rn(sinf(angle));
    }
    __syncthreads();
    const int qWidth = heads * 24, kWidth = kvHeads * 24, vWidth = kvHeads * 16;
    const int width = qWidth + kWidth + vWidth;
    int col = blockIdx.y * blockDim.x + t;
    if (col >= width) return;
    uint4 data = ((const uint4 *)packed)[size_t(token) * width + col];
    auto *a = reinterpret_cast<__nv_bfloat162 *>(&data);
    uint4 *dst;
    if (col < qWidth + kWidth) {
        int inHead = col % 24;
        if (inHead < 8) {
            // A 24-vector head starts on an eight-lane boundary.
            unsigned active = 0xffu << (t % 32 / 8 * 8);
            uint4 other = make_uint4(__shfl_xor_sync(active, data.x, 4, 8), __shfl_xor_sync(active, data.y, 4, 8),
                                      __shfl_xor_sync(active, data.z, 4, 8), __shfl_xor_sync(active, data.w, 4, 8));
            const auto *b = reinterpret_cast<const __nv_bfloat162 *>(&other);
            uint4 cv = ((const uint4 *)cosine)[inHead % 4], sv = ((const uint4 *)sine)[inHead % 4];
            const auto *c = reinterpret_cast<const __nv_bfloat162 *>(&cv);
            const auto *s = reinterpret_cast<const __nv_bfloat162 *>(&sv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                auto x = __hmul2_rn(a[j], c[j]), y = __hmul2_rn(b[j], s[j]);
                a[j] = inHead < 4 ? __hsub2_rn(x, y) : __hadd2_rn(x, y);
            }
        }
        dst = col < qWidth ? (uint4 *)q + size_t(token) * qWidth + col
                           : (uint4 *)k + size_t(token) * kWidth + col - qWidth;
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) a[j] = __hmul2_rn(a[j], __halves2bfloat162(scale, scale));
        dst = (uint4 *)v + size_t(token) * vWidth + col - qWidth - kWidth;
    }
    *dst = data;
}
#endif

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
    void *indexCache = nullptr;
    bool separateIndex = false;
    int indexPageLen = 0, indexContextRank = 0;
    bool fp4 = false, indexFp8 = false;
    bool ringVerify = false;
    int verifyOffset = 0;
};
constexpr int kRopeArgumentBytes = CUDART_VERSION >= 12010 ? 32764 : 4096;
constexpr int kRopeCacheBatch = std::min(192, (kRopeArgumentBytes - 128) / int(sizeof(RopeCacheRow)));
struct RopeCacheBatch { RopeCacheRow rows[kRopeCacheBatch]; };
#include "naive-n05-rope-compact.cuh"
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include "naive-n05-rope-request.cuh"
#endif

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
    const int row = r.ringVerify ? kSwaWindow + r.verifyOffset + token :
        ring ? ((*length - 1) & (kSwaWindow - 1)) : past + token;
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
        const int tail = r.scaledIndexer ? FastllmNaiveIndexerBytes / (r.indexFp8 ? 1 : 2) : indexDim;
        if (r.fp4 && !index) {
            __shared__ BF16 rotatedFP4[256];
            RopeHeadTo(src, rotatedFP4, positions[token], rotaryDim, theta);
            for (int d = rotaryDim + threadIdx.x; d < width; d += blockDim.x) rotatedFP4[d] = src[d];
            __syncthreads();
            const int columns = kvHeads * dim;
            auto *packedRow = (uint8_t *)key.data + (size_t)physical * (columns / 16 * 9);
            for (int block = threadIdx.x; block < dim / 16; block += blockDim.x)
                FastllmWriteFP4KVBlock(packedRow, rotatedFP4 + block * 16,
                    (head - heads) * dim + block * 16, columns);
            return;
        }
        auto dst = key + (size_t)physical * (kvHeads * dim + (r.separateIndex ? 0 : tail))
            + (index ? kvHeads * dim : (head - heads) * dim);
        if (index && r.separateIndex) {
            if (r.indexPageLen && r.indexPageLen != r.pageLen) {
                if (row % r.pageLen / r.indexPageLen != r.indexContextRank) return;
                physical = (physical / r.pageLen) * r.indexPageLen + row % r.indexPageLen;
            }
            dst = FastllmNaiveKVWriteRow{r.cache ? (void *)r.cache[r.pageLen ? 4 : 2] : r.indexCache, r.indexFp8} + (size_t)physical * tail;
        }
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
        if (r.fp4) {
            __shared__ BF16 scaledFP4[256];
            for (int d = threadIdx.x; d < valueDim; d += blockDim.x)
                scaledFP4[d] = __float2bfloat16_rn((float)src[d] * (float)valueScale);
            __syncthreads();
            const int columns = kvHeads * valueDim;
            auto *packedRow = (uint8_t *)value.data + (size_t)physical * (columns / 16 * 9);
            for (int block = threadIdx.x; block < valueDim / 16; block += blockDim.x)
                FastllmWriteFP4KVBlock(packedRow, scaledFP4 + block * 16,
                    h * valueDim + block * 16, columns);
            return;
        }
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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
template <bool Packed>
__global__ void BatchRopeAppendCache(RopeCacheBatch batch, int heads, int kvHeads, int dim,
        int valueDim, int indexDim, int rotaryDim, float theta, BF16 valueScale,
        int window, bool ring) {
    AppendRopeCacheRow<Packed>(batch.rows[blockIdx.y], 0, blockIdx.x, heads, kvHeads,
        dim, valueDim, indexDim, rotaryDim, theta, valueScale, window, ring);
}

#if CUDART_VERSION >= 12010
template <bool Packed>
__global__ void VerifyRopeAppendCache(const __grid_constant__ VerifyRopeBatch batch,
        int heads, int kvHeads, int dim, int valueDim, int rotaryDim, float theta,
        BF16 valueScale) {
    AppendRopeCacheRow<Packed>(VerifyRopeRow(batch, blockIdx.y, heads, kvHeads, dim, valueDim),
        0, blockIdx.x, heads, kvHeads, dim, valueDim, 0, rotaryDim, theta,
        valueScale, kSwaWindow, false);
}
#endif

#endif

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

__device__ void RoundIndexerRow(FastllmNaiveKV input, float *output, int stride,
                                int offset, bool fp8, int row) {
    __shared__ float maximum[128];
    int d = threadIdx.x;
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

__global__ void RoundIndexer(FastllmNaiveKV input, float *output, int stride,
                             int offset, bool fp8) {
    RoundIndexerRow(input, output, stride, offset, fp8, blockIdx.x);
}

__global__ void VerifyPositionLengths(const float *positions, int *lengths, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) lengths[row] = (int)positions[row] + 1;
}
struct VerifyLengthBatch { int offset[32], rows[32], past[32]; };
__global__ void VerifyIndexerLengths(VerifyLengthBatch batch, int *lengths) {
    const int item = blockIdx.x;
    for (int row = threadIdx.x; row < batch.rows[item]; row += blockDim.x)
        lengths[batch.offset[item] + row] = batch.past[item] + row + 1;
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
template <typename ScoreOutput>
__device__ void IndexScoresDecodeRow(const float *q, FastllmNaiveKV packedKeys,
        const BF16 *weights, ScoreOutput scores, int stride, int keys, int queryStart) {
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
    IndexScoresDecodeRow(q, packedKeys, weights, scores, stride, (int)liveKeys, queryStart);
}

#ifdef FASTLLM_NAIVE_BATCH_INDEXER
// Scalar batched fallback shares exactly the single-query rounding and score
// reduction. Only the request/query descriptor changes with grid.y.
__global__ void RoundBatchIndexer(const BF16 *query, naive_indexer_batch::Batch batch, bool fp8) {
    const auto &r = batch.rows[blockIdx.y];
    RoundIndexerRow(FastllmNaiveKV(query + (size_t)r.sequence * 16 * 128),
        (float *)r.query, 128, 0, fp8, blockIdx.x);
}
__global__ void BatchIndexScoresScalar(const BF16 *weights, naive_indexer_batch::Batch batch) {
    const auto &r = batch.rows[blockIdx.y];
    auto key = r.key;
    key.Bind(r.cache);
    naive_indexer_batch::ScoreOutput output{r.scores};
    IndexScoresDecodeRow((const float *)r.query, key, weights + (size_t)r.sequence * 16,
        output, r.stride, *r.live, *r.live - 1);
}
#endif

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
template <int MaxDim, typename Length = int, typename Past = int, int KeyTile = 0>
__device__ void AttentionScoresPrefillBody(const BF16 *q, FastllmNaiveKV k, const int *indices,
                                      float *scores, int heads, int kvHeads, int dim,
                                      int keyStride, Length liveKeys, Length liveCount, Past livePast,
                                      int window, bool causal, int bx, int by, int bz, int gz) {
    int keys = liveKeys, count = liveCount, past = livePast;
    constexpr int keysPerWarp = 4, warps = 4;
    int query = by, h = bx;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int kvHead = h / (heads / kvHeads);
    float queryValues[MaxDim / 32];
    #pragma unroll
    for (int j = 0; j < MaxDim / 32; ++j)
        if (lane + j * 32 < dim)
            queryValues[j] = (float)q[((size_t)query * heads + h) * dim + lane + j * 32];
    const int begin = KeyTile ? bz * KeyTile : 0;
    const int end = KeyTile ? min(count, begin + KeyTile) : count;
    for (int first = begin + warp * keysPerWarp; first < end; first += warps * keysPerWarp) {
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
template <int MaxDim, typename Length = int, typename Past = int, int KeyTile = 0>
__global__ void AttentionScoresPrefill(const BF16 *q, FastllmNaiveKV k, const int *indices,
                                      float *scores, int heads, int kvHeads, int dim,
                                      int keyStride, Length liveKeys, Length liveCount, Past livePast,
                                      int window, bool causal) {
    AttentionScoresPrefillBody<MaxDim, Length, Past, KeyTile>(q, k, indices, scores, heads, kvHeads, dim, keyStride, liveKeys, liveCount, livePast, window, causal, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}

// Reuse K across neighboring Q heads with the original dot-product order.
// A bounded grid strides over live keys, including during graph replay.
__device__ void DraftAttentionScoresBody(const BF16 *q, FastllmNaiveKV k, float *scores,
        int heads, int kvHeads, DraftLength liveKeys, DraftLength livePast, int window, int bx, int by, int bz, int gz) {
    constexpr int Dim = 128, KeysPerWarp = 4, HeadTile = 4;
    const int keys = liveKeys, past = livePast, query = by;
    const int firstHead = bx * HeadTile, kh = firstHead / (heads / kvHeads);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float queries[HeadTile][4];
    #pragma unroll
    for (int h = 0; h < HeadTile; ++h)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            queries[h][j] = (float)q[((size_t)query * heads + firstHead + h) * Dim + lane + j * 32];
    for (int first = bz * 16 + warp * KeysPerWarp; first < keys; first += gz * 16) {
        int key[KeysPerWarp];
        bool valid[KeysPerWarp];
        float dot[HeadTile][KeysPerWarp] = {};
        #pragma unroll
        for (int i = 0; i < KeysPerWarp; ++i) {
            key[i] = KeyIndex(nullptr, query, first + i, keys, past, window);
            valid[i] = first + i < keys && key[i] < keys;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int i = 0; i < KeysPerWarp; ++i) if (valid[i]) {
                float value = (float)k.Row(key[i], kvHeads * Dim)[kh * Dim + lane + j * 32];
                #pragma unroll
                for (int h = 0; h < HeadTile; ++h) dot[h][i] += queries[h][j] * value;
            }
        }
        #pragma unroll
        for (int h = 0; h < HeadTile; ++h)
            #pragma unroll
            for (int i = 0; i < KeysPerWarp; ++i) {
                float sum = WarpSum(dot[h][i]);
                if (lane == 0 && first + i < keys)
                    scores[((size_t)query * heads + firstHead + h) * keys + first + i] =
                        valid[i] ? RoundBF16(RoundBF16(sum) * rsqrtf((float)Dim)) : -INFINITY;
            }
    }
}
__global__ void DraftAttentionScores(const BF16 *q, FastllmNaiveKV k, float *scores,
        int heads, int kvHeads, DraftLength liveKeys, DraftLength livePast, int window) {
    DraftAttentionScoresBody(q, k, scores, heads, kvHeads, liveKeys, livePast, window, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
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
__device__ void AttentionValuesPrefillBody(const float *prob, FastllmNaiveKV v, const int *indices,
                                      BF16 *out, int heads, int kvHeads, int dim,
                                      Length liveKeys, Length liveCount, Past livePast, int window, bool causal, int bx, int by, int bz, int gz) {
    int keys = liveKeys, count = liveCount, past = livePast;
    constexpr int columns = 4, headGroup = 4;
    int query = by, lane = threadIdx.x % 32;
    int h = bx * headGroup + threadIdx.x / 32;
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
                if constexpr (Paged) row = v.RowAtPhysicalIndex(key, kvHeads * dim);
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
template <typename Length = int, typename Past = int, bool Paged = false>
__global__ void AttentionValuesPrefill(const float *prob, FastllmNaiveKV v, const int *indices,
                                      BF16 *out, int heads, int kvHeads, int dim,
                                      Length liveKeys, Length liveCount, Past livePast, int window, bool causal) {
    AttentionValuesPrefillBody<Length, Past, Paged>(prob, v, indices, out, heads, kvHeads, dim, liveKeys, liveCount, livePast, window, causal, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}

// Stage a small V column tile cooperatively, then accumulate each output in
// the original key order. Small draft blocks need more CTAs than prefill's
// four-head kernel; splitting output columns does not change the reduction.
// Larger batches share each V tile across a GQA head group; each output still
// accumulates keys in the original order. Small batches keep more independent CTAs.
template <int Columns = 16, int Padding = 8, int HeadTile = 1>
__device__ void DraftAttentionValuesBody(const float *prob, FastllmNaiveKV v, BF16 *out,
        int heads, int kvHeads, int dim, DraftLength liveKeys, DraftLength livePast, int window, int bx, int by, int bz, int gz) {
    constexpr int Tile = 128;
    const int query = by, h = bx * HeadTile, column = bz * Columns;
    const int t = threadIdx.x, keys = liveKeys, past = livePast, kh = h / (heads / kvHeads);
    const float *p = prob + ((size_t)query * heads + h) * keys;
    __shared__ __align__(16) BF16 values[Columns][Tile + Padding];
    __shared__ float weights[HeadTile][Tile];
    float sum = 0;
    for (int first = 0; first < keys; first += Tile) {
        const int count = min(Tile, keys - first);
        for (int i = t; i < HeadTile * count; i += blockDim.x)
            weights[i / count][i % count] = p[(size_t)(i / count) * keys + first + i % count];
        for (int i = t; i < count * (Columns / 8); i += blockDim.x) {
            int slot = i / (Columns / 8), d = i % (Columns / 8) * 8;
            int key = KeyIndex(nullptr, query, first + slot, keys, past, window);
            uint4 packed = key < keys
                ? (v.Row(key, kvHeads * dim) + kh * dim + column + d).Load8()
                : make_uint4(0, 0, 0, 0);
            const BF16 *bf = (const BF16 *)&packed;
            #pragma unroll
            for (int j = 0; j < 8; ++j) values[d + j][slot] = bf[j];
        }
        __syncthreads();
        if (t < Columns * HeadTile) {
            for (int slot = 0; slot < count; ++slot)
                sum = fmaf(weights[t / Columns][slot], (float)values[t % Columns][slot], sum);
        }
        __syncthreads();
    }
    if (t < Columns * HeadTile)
        out[((size_t)query * heads + h + t / Columns) * dim + column + t % Columns] = __float2bfloat16(sum);
}
template <int Columns = 16, int Padding = 8, int HeadTile = 1>
__global__ void DraftAttentionValues(const float *prob, FastllmNaiveKV v, BF16 *out,
        int heads, int kvHeads, int dim, DraftLength liveKeys, DraftLength livePast, int window) {
    DraftAttentionValuesBody<Columns, Padding, HeadTile>(prob, v, out, heads, kvHeads, dim, liveKeys, livePast, window, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}

template <typename Length = int>
__device__ void AttentionSoftmaxBody(float *scores, const float *sink, int heads, Length liveCount, int bx, int by, int bz, int gz) {
    int count = liveCount;
    __shared__ float scratch[256];
    int row = bx, t = threadIdx.x;
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
template <typename Length = int>
__global__ void AttentionSoftmax(float *scores, const float *sink, int heads, Length liveCount) {
    AttentionSoftmaxBody<Length>(scores, sink, heads, liveCount, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
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
__device__ void AttentionShortBody(const BF16 *q, FastllmNaiveKV k, FastllmNaiveKV v,
                               const int *indices, const float *sink, BF16 *out,
                               int heads, int kvHeads, int dim, int valueDim,
                               int keyStride, Length liveKeys, int count, Past livePast, int window, bool causal, int bx, int by, int bz, int gz) {
    int keys = liveKeys, past = livePast;
    count = ShortCount(liveKeys, count);
    __shared__ float scores[256], scratch[256];
    int query = by, h = bx, t = threadIdx.x;
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
template <typename Length = int, typename Past = int>
__global__ void AttentionShort(const BF16 *q, FastllmNaiveKV k, FastllmNaiveKV v,
                               const int *indices, const float *sink, BF16 *out,
                               int heads, int kvHeads, int dim, int valueDim,
                               int keyStride, Length liveKeys, int count, Past livePast, int window, bool causal) {
    AttentionShortBody<Length, Past>(q, k, v, indices, sink, out, heads, kvHeads, dim, valueDim, keyStride, liveKeys, count, livePast, window, causal, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}

constexpr int kSwaQkDim = 192;
constexpr int kSwaValueDim = 128;
constexpr int kSwaOutputTile = 32;
constexpr int kSwaThreads = 256;
constexpr int kSwaBatchLimit = 64;
struct BatchSwaKeys {
    FastllmNaiveSWARow rows[kSwaBatchLimit];
    const int *length;
    __device__ int Live() const { return length ? length[blockIdx.y] : rows[blockIdx.y].length; }
    __device__ operator int() const { return min(Live(), kSwaWindow); }
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
        if constexpr (MaxKeys == kSwaWindow && !Ring) {
            // Verification appends all rows before attention. Read each
            // causal window directly instead of copying it into scratch.
            const int end = (int)liveKeys + row;
            const int begin = max(0, end - kSwaWindow);
            k += (size_t)begin * keyStride;
            v += (size_t)begin * kvHeads * kSwaValueDim;
        }
    }
    int firstKey = 0, prefixEnd = 0;
    if constexpr (Ring) {
        if constexpr (Verify) {
            prefixEnd = *liveKeys.length - 1;
            firstKey = max(0, prefixEnd + (int)blockIdx.y + 1 - kSwaWindow);
        } else if constexpr (std::is_same<Length, BatchSwaKeys>::value)
            firstKey = max(0, liveKeys.Live() - kSwaWindow);
        else firstKey = max(0, *liveKeys.length - kSwaWindow);
    }
    __shared__ float scores[MaxKeys], scratch[kSwaThreads], maximum, denominator;
    __shared__ __align__(16) BF16 values[MaxKeys][OutputTile];
    int h = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    int kvHead = h / (heads / kvHeads), firstDim = blockIdx.z * OutputTile;
    auto cacheRow = [&](int slot) {
        const int row = firstKey + slot;
        if constexpr (std::is_same<Length, BatchSwaKeys>::value) {
            const auto &cache = liveKeys.rows[blockIdx.y];
            if (!cache.ring) return row;
            const int prefix = liveKeys.Live() - cache.verifyOffset - 1;
            return cache.verifyOffset >= 0 && row >= prefix
                ? kSwaWindow + row - prefix : row & (kSwaWindow - 1);
        }
        return Ring ? (Verify && row >= prefixEnd ? kSwaWindow + row - prefixEnd
                                                  : row & (kSwaWindow - 1)) : slot;
    };
    if (v.fp8) {
        for (int i = t; i < keys * (OutputTile / 8); i += Threads) {
            int row = i / (OutputTile / 8), col = i % (OutputTile / 8) * 8;
            const int physical = cacheRow(row);
            *(uint4 *)&values[row][col] = (v.Row(physical, kvHeads * kSwaValueDim) +
                kvHead * kSwaValueDim + firstDim + col).Load8();
        }
    } else {
        for (int i = t; i < keys * OutputTile; i += Threads) {
            int row = i / OutputTile, col = i % OutputTile;
            const int physical = cacheRow(row);
            values[row][col] = v.Row(physical, kvHeads * kSwaValueDim)[kvHead * kSwaValueDim + firstDim + col];
        }
    }
    float query[kSwaQkDim / 32];
    #pragma unroll
    for (int i = 0; i < kSwaQkDim / 32; ++i) query[i] = (float)q[(size_t)h * kSwaQkDim + lane + i * 32];
    for (int slot = warp; slot < keys; slot += Threads / 32) {
        const int physical = cacheRow(slot);
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

// Reuse a KV tile across four complete GQA heads without changing the
// scalar QK reduction, softmax tree or ordered PV accumulation. Small batches
// and devices without enough shared memory retain AttentionShortDecode.
template<int Groups> struct GroupedSwaStorage {
    __align__(16) BF16 key[128][200], value[128][128];
    float scores[Groups][128], scratch[Groups / 2][256];
    float maximum[Groups / 2], denominator[Groups / 2];
    int physical[128];
};
template<int Groups>
__global__ void GroupedSwaDecode(const BF16 *q, const float *sink, BF16 *out,
        int heads, int kvHeads, const BatchSwaKeys batch) {
    constexpr int Threads=Groups*128;
    const int t=threadIdx.x, row=blockIdx.y, firstHead=blockIdx.x*Groups;
    const auto &cache=batch.rows[row];
    const int length=batch.length?batch.length[row]:cache.length, count=min(length,128);
    if(count<=0) {
        for(int i=t;i<Groups*128;i+=Threads)out[((size_t)row*heads+firstHead)*128+i]=__float2bfloat16(0.f);
        return;
    }
    extern __shared__ __align__(16) unsigned char memory[];
    auto &s=*(GroupedSwaStorage<Groups>*)memory;
    const int kh=firstHead/(heads/kvHeads), first=max(0,length-128), prefix=length-cache.verifyOffset-1;
    if(t<128) {
        int logical=first+t;
        s.physical[t]=t<count?(cache.verifyOffset>=0&&logical>=prefix?128+logical-prefix:logical&127):-1;
    }
    __syncthreads();
    FastllmNaiveKV key(cache.cache?cache.cache[0]:cache.key,nullptr,0,cache.fp8);
    FastllmNaiveKV value(cache.cache?cache.cache[1]:cache.value,nullptr,0,cache.fp8);
    for(int i=t;i<128*24;i+=Threads) {
        int r=s.physical[i/24],d=i%24*8;
        naive_dsa_decode::Copy<false>(&s.key[i/24][d],key.Row(max(0,r),kvHeads*192)+kh*192+d,r>=0);
    }
    for(int i=t;i<128*16;i+=Threads) {
        int r=s.physical[i/16],d=i%16*8;
        naive_dsa_decode::Copy<false>(&s.value[i/16][d],value.Row(max(0,r),kvHeads*128)+kh*128+d,r>=0);
    }
    const int h=t/128,lane=t%32,warp=(t%128)/32;
    float qr[6];
    #pragma unroll
    for(int d=0;d<6;++d)qr[d]=float(q[((size_t)row*heads+firstHead+h)*192+lane+d*32]);
    __syncthreads();
    for(int i=warp;i<count;i+=4) {
        float dot=0;
        #pragma unroll
        for(int d=0;d<6;++d)dot=fmaf(qr[d],float(s.key[i][lane+d*32]),dot);
        dot=naive_dsa_decode::WarpSum(dot);
        if(!lane)s.scores[h][i]=RoundBF16(RoundBF16(dot)*rsqrtf(192.f));
    }
    __syncthreads();
    for(int wave=0;wave<2;++wave) {
        const int group=t/256,head=group+wave*(Groups/2);
        naive_dsa_decode::Normalize(s.scores[head],s.scores[head],sink,firstHead+head,count,
            s.scratch[group],s.maximum[group],s.denominator[group],t%256);
        __syncthreads();
    }
    float sum=0;
    for(int first=0;first<count;first+=8) {
        float p[8],v[8];
        #pragma unroll
        for(int i=0;i<8;++i)if(first+i<count) {
            p[i]=s.scores[h][first+i];v[i]=float(s.value[first+i][t%128]);
        }
        #pragma unroll
        for(int i=0;i<8;++i)if(first+i<count)sum=fmaf(p[i],v[i],sum);
    }
    out[((size_t)row*heads+firstHead+h)*128+t%128]=__float2bfloat16(sum);
}

// Dispatch by workload shape, independent of TP rank count. All variants retain
// the same QK/softmax/PV arithmetic; only the work assigned to each CTA changes.
static bool WideSwaOutputTile(int64_t work) {
    const int arch = FastllmCudaRuntimeArch();
    return (arch == 90 || arch == 120) && work >= 48;
}

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
    if constexpr (Verify) {
        if (ring) {
            if (threads == 1024)
                kernel = AttentionShortDecode<kSwaWindow, DecodeKeys, true, 1024, 64, true>;
            else if (threads == 512)
                kernel = AttentionShortDecode<kSwaWindow, DecodeKeys, true, 512, 64, true>;
            else kernel = AttentionShortDecode<kSwaWindow, DecodeKeys, true, kSwaThreads, kSwaOutputTile, true>;
        }
    } else {
        if (ring) kernel = threads == 1024
            ? AttentionShortDecode<kSwaWindow, DecodeKeys, false, 1024, kSwaOutputTile, true>
            : AttentionShortDecode<kSwaWindow, DecodeKeys, false, kSwaThreads, kSwaOutputTile, true>;
    }
    if (sliding && WideSwaOutputTile((int64_t)rows * heads)) {
        // One CTA computes all V columns, avoiding four copies of QK and
        // softmax. Keep the same live-length and staged verification mapping.
        tile = kSwaValueDim;
        threads = (int64_t)rows * heads > 128 ? 512 : 1024;
        if (threads == 512)
            kernel = ring ? AttentionShortDecode<kSwaWindow, DecodeKeys, Verify, 512, 128, true>
                          : AttentionShortDecode<kSwaWindow, DecodeKeys, Verify, 512, 128>;
        else
            kernel = ring ? AttentionShortDecode<kSwaWindow, DecodeKeys, Verify, 1024, 128, true>
                          : AttentionShortDecode<kSwaWindow, DecodeKeys, Verify, 1024, 128>;
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
__device__ void DraftQKVNormRopeBody(const BF16 *raw, const float *qNorm,
        const float *kNorm, const float *positions, const int *live,
        BF16 *key, BF16 *value, BF16 *query, int heads, int kvHeads,
        int dim, int window, int past, float eps, float theta, int bx, int by, int bz, int gz) {
    const int head = bx, row = by, t = threadIdx.x;
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
__global__ void DraftQKVNormRope(const BF16 *raw, const float *qNorm,
        const float *kNorm, const float *positions, const int *live,
        BF16 *key, BF16 *value, BF16 *query, int heads, int kvHeads,
        int dim, int window, int past, float eps, float theta) {
    DraftQKVNormRopeBody(raw, qNorm, kNorm, positions, live, key, value, query, heads, kvHeads, dim, window, past, eps, theta, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}

struct DraftBatchRow {
    const BF16 *input;
    BF16 *key, *value, *query;
    const float *positions;
    const int *live;
    float *scores;
    BF16 *output;
    bool shortAttention;
};
constexpr int kDraftBatchLimit = 32;
struct DraftBatch { DraftBatchRow rows[kDraftBatchLimit]; };
static_assert(sizeof(DraftBatch) < 3900, "Draft descriptors exceed generic CUDA argument limit");
// Capture descriptors by value in a small upload node. Unlike a captured H2D
// copy this never retains a pointer to a temporary host vector. The consuming
// kernel sees one device array, independent of the kernel argument limit.
template<class Row, int Bytes> struct DescriptorChunk {
    static constexpr int count = std::min(256, Bytes / int(sizeof(Row)));
    Row rows[count];
};
template<class Row, int Bytes>
__global__ void UploadDescriptors(DescriptorChunk<Row, Bytes> chunk, Row *out, int count) {
    if (threadIdx.x < count) out[threadIdx.x] = chunk.rows[threadIdx.x];
}
template<class Row, int Bytes>
void UploadDescriptorChunks(const std::vector<Row> &rows, Row *out) {
    using Chunk = DescriptorChunk<Row, Bytes>;
    for (size_t first = 0; first < rows.size(); first += Chunk::count) {
        const int count = std::min<size_t>(Chunk::count, rows.size() - first);
        Chunk chunk{};
        std::copy_n(rows.begin() + first, count, chunk.rows);
        UploadDescriptors<<<1, 256, 0, cudaStreamPerThread>>>(chunk, out + first, count);
    }
}
template<class Row>
const Row *DeviceDescriptors(const std::vector<Row> &rows, fastllm::Data &storage) {
    Output(storage, fastllm::INT32, {int((rows.size() * sizeof(Row) + 3) / 4)});
    auto *out = (Row *)storage.cudaData;
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
    if (FastllmCudaRuntimeArch() >= 70) {
        UploadDescriptorChunks<Row, 32000>(rows, out);
        return out;
    }
#endif
    UploadDescriptorChunks<Row, 3900>(rows, out);
    return out;
}

bool DescriptorWorkspaceAliases(const fastllm::Data *workspace, const fastllm::Data &data) {
    return workspace && (workspace->isFake || workspace->multiDeviceData ||
        workspace == &data || (workspace->cudaData && workspace->cudaData == data.cudaData));
}
struct DeviceDraftBatch {
    const DraftBatchRow *rows;
    const uint64_t *cacheBindings = nullptr;
};
__device__ __forceinline__ DraftBatchRow DraftRow(const DraftBatch &batch, int index) {
    return batch.rows[index];
}
__device__ __forceinline__ DraftBatchRow DraftRow(const DeviceDraftBatch &batch, int index) {
    auto row = batch.rows[index];
    if (batch.cacheBindings) {
        row.key = (BF16 *)batch.cacheBindings[index * 2];
        row.value = (BF16 *)batch.cacheBindings[index * 2 + 1];
    }
    return row;
}
#ifdef FASTLLM_NAIVE_SWA_FLASHINFER
// Reuse the existing FlashInfer device primitive with independent KV pointers.
// Request lengths are read on device so a captured graph remains reusable.
struct DraftFlashAttention : flashinfer::DefaultAttention<false, true, false, false> {
    template<class Params>
    __host__ __device__ DraftFlashAttention(const Params &p, unsigned batch, uint8_t *smem)
        : flashinfer::DefaultAttention<false, true, false, false>(p, batch, smem) {
        sm_scale_log2 = flashinfer::math::log2e;
    }
    REGISTER_LOGITS_TRANSFORM(p, logits, batch, qi, ki, qh, kh, {
        // Match the model's BF16 QK and scaling boundaries before softmax.
        return float(__float2bfloat16(float(__float2bfloat16(logits)) * p.sm_scale));
    })
};
using DraftFlashParams = flashinfer::SinglePrefillParams<BF16, BF16, BF16>;
using DraftFlashTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kNone,
    64, 1, 2, 8, 8, 4, 1, flashinfer::PosEncodingMode::kNone,
    BF16, BF16, BF16, float, int, DraftFlashAttention, false>;
__global__ __launch_bounds__(128) void BatchDraftFlash(DeviceDraftBatch batch, DraftFlashParams p) {
    const auto r = DraftRow(batch, blockIdx.y);
    if (!r.key || !r.value) return; // Padding has no KV and no published output.
    p.q = const_cast<BF16 *>(r.input); p.k = r.key; p.v = r.value; p.o = r.output;
    p.kv_len = min(*r.live - 1, p.window_left) + p.qo_len;
    extern __shared__ uint8_t smem[];
    auto &storage = *reinterpret_cast<DraftFlashTraits::SharedStorageSingle *>(smem);
    flashinfer::SinglePrefillWithKVCacheDevice<DraftFlashTraits>(p, storage,
        threadIdx, blockIdx.x, 0, blockIdx.z, 1, p.num_kv_heads);
}
#endif

template<class Batch>
__global__ void BatchDraftQKV(Batch batch, const float *qNorm, const float *kNorm,
        int rows, int heads, int kvHeads, int dim, int window, float eps, float theta) {
    const auto r = DraftRow(batch, blockIdx.y / rows);
    if (!r.key || !r.value) return;
    DraftQKVNormRopeBody(r.input, qNorm, kNorm, r.positions, r.live,
        r.key, r.value, r.query, heads, kvHeads, dim, window, 0, eps, theta,
        blockIdx.x, blockIdx.y % rows, 0, 1);
}
__global__ void BatchDraftShort(DraftBatch batch, int rows, int heads, int kvHeads, int dim, int window) {
    const auto &r = batch.rows[blockIdx.y / rows];
    if (!r.shortAttention) return;
    DraftLength keys{r.live, window, rows}, past{r.live, window, 0};
    AttentionShortBody(r.input, FastllmNaiveKV(r.key), FastllmNaiveKV(r.value), nullptr, nullptr,
        r.output, heads, kvHeads, dim, dim, kvHeads * dim, keys, 256, past, window, false,
        blockIdx.x, blockIdx.y % rows, 0, 1);
}
template <int MaxDim, int KeyTile = 0, bool Hopper = false>
__global__ void BatchDraftScores(DraftBatch batch, int rows, int heads, int kvHeads, int dim, int window) {
    const auto &r = batch.rows[blockIdx.y / rows];
    if (r.shortAttention) return;
    DraftLength keys{r.live, window, rows}, past{r.live, window, 0};
    if constexpr (Hopper)
        DraftAttentionScoresBody(r.input, FastllmNaiveKV(r.key), r.scores, heads, kvHeads,
            keys, past, window, blockIdx.x, blockIdx.y % rows, blockIdx.z, gridDim.z);
    else
        AttentionScoresPrefillBody<MaxDim, DraftLength, DraftLength, KeyTile>(r.input,
            FastllmNaiveKV(r.key), nullptr, r.scores, heads, kvHeads, dim, kvHeads * dim,
            keys, keys, past, window, false, blockIdx.x, blockIdx.y % rows, blockIdx.z, gridDim.z);
}
__global__ void BatchDraftSoftmax(DraftBatch batch, int rows, int heads, int window) {
    const auto &r = batch.rows[blockIdx.y];
    if (r.shortAttention) return;
    AttentionSoftmaxBody(r.scores, nullptr, heads, DraftLength{r.live, window, rows}, blockIdx.x, 0, 0, 1);
}
template <int Columns = 16, int Padding = 8, bool Generic = false, int HeadTile = 1>
__global__ void BatchDraftValues(DraftBatch batch, int rows, int heads, int kvHeads, int dim, int window) {
    const auto &r = batch.rows[blockIdx.y / rows];
    if (r.shortAttention) return;
    DraftLength keys{r.live, window, rows}, past{r.live, window, 0};
    if constexpr (Generic)
        AttentionValuesPrefillBody(r.scores, FastllmNaiveKV(r.value), nullptr, r.output,
            heads, kvHeads, dim, keys, keys, past, window, false,
            blockIdx.x, blockIdx.y % rows, blockIdx.z, gridDim.z);
    else
        DraftAttentionValuesBody<Columns, Padding, HeadTile>(r.scores, FastllmNaiveKV(r.value), r.output,
            heads, kvHeads, dim, keys, past, window, blockIdx.x, blockIdx.y % rows, blockIdx.z, gridDim.z);
}
__global__ void DraftRoundedSwiGLU(const BF16 *input, BF16 *output, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int at = (i / width) * 2 * width + i % width;
    float gate = (float)input[at], up = (float)input[at + width];
    output[i] = __float2bfloat16(RoundBF16(gate / (1.0f + expf(-gate))) * up);
}
bool DraftDense(const fastllm::Data &x, fastllm::DataType type, int device = -1) {
    if (x.dataType != type || x.dataDevice != fastllm::DataDevice::CUDA ||
        x.multiDeviceData || x.dataDeviceIds.size() != 1 ||
        x.dataDeviceIds[0] != (device < 0 ? FastllmCudaGetDevice() : device) ||
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
        DraftKVPointerTable<<<1, 128, 0, cudaStreamPerThread>>>(addresses, input.cudaData,
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

static bool PrepareDraftQKV(const fastllm::Data &raw,
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
    return true;
}

bool FastllmCudaNaiveDraftQKV(const fastllm::Data &raw,
        const fastllm::Data &qNorm, const fastllm::Data &kNorm,
        const fastllm::Data &positions, const fastllm::Data &liveKeys,
        fastllm::Data &key, fastllm::Data &value, fastllm::Data &query,
        int heads, int kvHeads, int dim, int window, float eps, float theta) {
    if (!PrepareDraftQKV(raw, qNorm, kNorm, positions, liveKeys, key, value, query,
            heads, kvHeads, dim, window, eps, theta)) return false;
    using namespace fastllm;
    const int rows = raw.dims[1];
    const bool dynamic = !liveKeys.dims.empty();
    Output(query, BFLOAT16, {1, rows, heads * dim});
    DraftQKVNormRope<<<dim3(heads + kvHeads, rows), 256, 0, cudaStreamPerThread>>>(
        (const BF16 *)raw.cudaData, (const float *)qNorm.cudaData, (const float *)kNorm.cudaData,
        (const float *)positions.cudaData, dynamic ? (const int *)liveKeys.cudaData : nullptr,
        (BF16 *)key.cudaData, (BF16 *)value.cudaData, (BF16 *)query.cudaData,
        heads, kvHeads, dim, window, key.dims[1], eps, theta);
    CheckLaunch();
    return true;
}

bool FastllmCudaNaiveDraftCacheBindingsSupported(int count, int rows,
        int heads, int kvHeads, int dim, int window) {
#ifdef FASTLLM_NAIVE_SWA_FLASHINFER
    return count > 0 && rows > 1 && rows < 32 && heads > 0 && kvHeads > 0 &&
        heads % kvHeads == 0 && heads / kvHeads <= 32 && dim == 128 && window >= rows &&
        BulkKernelSupported(BatchDraftFlash, sizeof(DraftFlashTraits::SharedStorageSingle), 80);
#else
    return false;
#endif
}

bool FastllmCudaNaiveBatchDraftQKV(const fastllm::Data &raw,
        const fastllm::Data &qNorm, const fastllm::Data &kNorm,
        const std::vector<FastllmNaiveDraftRow> &requests, fastllm::Data &query,
        int heads, int kvHeads, int dim, int window, float eps, float theta,
        fastllm::Data *descriptorsWorkspace, const fastllm::Data *cacheBindings) {
    using namespace fastllm;
    const int count = requests.size();
    if (!count || !DraftDense(raw, BFLOAT16) || raw.dims.size() != 3 ||
        raw.dims[0] != 1 || raw.dims[1] % count) return false;
    const int rows = raw.dims[1] / count;
    if (cacheBindings && (!descriptorsWorkspace || !DraftDense(*cacheBindings, INT32) ||
        cacheBindings->dataDeviceIds != raw.dataDeviceIds ||
        cacheBindings->dims != std::vector<int>({count, 4}) ||
        DescriptorWorkspaceAliases(descriptorsWorkspace, *cacheBindings))) return false;
    for (const auto *data : {&raw, &qNorm, &kNorm, (const Data *)&query})
        if (DescriptorWorkspaceAliases(descriptorsWorkspace, *data)) return false;
    std::vector<DraftBatchRow> descriptors;
    for (int i = 0; i < count; ++i) {
        const auto &r = requests[i];
        if (!r.key || !r.value || !r.positions || !r.liveKeys || !DraftDense(*r.liveKeys, INT32)) return false;
        for (const auto *data : {(const Data *)r.key, (const Data *)r.value, r.positions, r.liveKeys})
            if (DescriptorWorkspaceAliases(descriptorsWorkspace, *data)) return false;
        Data part; part.Resize({1, rows, raw.dims[2]});
        part.FakeFrom(raw, (size_t)i * rows * raw.dims[2] * raw.unitSize);
        if (!PrepareDraftQKV(part, qNorm, kNorm, *r.positions, *r.liveKeys, *r.key, *r.value,
                query, heads, kvHeads, dim, window, eps, theta)) return false;
        descriptors.push_back({(const BF16 *)part.cudaData, (BF16 *)r.key->cudaData,
            (BF16 *)r.value->cudaData, nullptr, (const float *)r.positions->cudaData,
            (const int *)r.liveKeys->cudaData, nullptr, nullptr, r.shortAttention});
    }
    Output(query, BFLOAT16, {1, raw.dims[1], heads * dim});
    for (int i = 0; i < count; ++i) descriptors[i].query = (BF16 *)query.cudaData + (size_t)i * rows * heads * dim;
    if (descriptorsWorkspace && (count > kDraftBatchLimit || cacheBindings)) {
        DeviceDraftBatch batch{DeviceDescriptors(descriptors, *descriptorsWorkspace),
            cacheBindings ? (const uint64_t *)cacheBindings->cudaData : nullptr};
        BatchDraftQKV<<<dim3(heads + kvHeads, rows * count), 256, 0, cudaStreamPerThread>>>(batch,
            (const float *)qNorm.cudaData, (const float *)kNorm.cudaData,
            rows, heads, kvHeads, dim, window, eps, theta);
        CheckLaunch();
        return true;
    }
    for (int first = 0; first < count; first += kDraftBatchLimit) {
        const int n = std::min(kDraftBatchLimit, count - first);
        DraftBatch batch{}; std::copy_n(descriptors.begin() + first, n, batch.rows);
        BatchDraftQKV<<<dim3(heads + kvHeads, rows * n), 256, 0, cudaStreamPerThread>>>(batch,
            (const float *)qNorm.cudaData, (const float *)kNorm.cudaData,
            rows, heads, kvHeads, dim, window, eps, theta);
    }
    CheckLaunch();
    return true;
}

bool FastllmCudaNaiveBatchDraftAttention(const fastllm::Data &query,
        const std::vector<FastllmNaiveDraftRow> &requests,
        int heads, int kvHeads, int dim, int window, fastllm::Data &scores, fastllm::Data &output,
        const fastllm::Data *cacheBindings) {
    using namespace fastllm;
    const int count = requests.size();
    if (!count || heads <= 0 || kvHeads <= 0 || heads % kvHeads || dim <= 0 || dim > 256 || dim % 4 || window < 2 ||
        !DraftDense(query, BFLOAT16) || query.dims.size() != 3 || query.dims[0] != 1 ||
        query.dims[1] % count || query.dims[2] != heads * dim) return false;
    const int rows = query.dims[1] / count, capacity = window + rows - 1;
    if (rows <= 1 || rows >= 32) return false;
    const bool fused = (cacheBindings || count >= 8) && FastllmCudaNaiveDraftCacheBindingsSupported(count, rows, heads, kvHeads, dim, window);
    if (cacheBindings && (!fused || !DraftDense(*cacheBindings, INT32) ||
        cacheBindings->dataDeviceIds != query.dataDeviceIds ||
        cacheBindings->dims != std::vector<int>({count, 4}) ||
        DescriptorWorkspaceAliases(&scores, *cacheBindings) ||
        DescriptorWorkspaceAliases(&output, *cacheBindings))) return false;
    std::vector<DraftBatchRow> descriptors;
    for (int i = 0; i < count; ++i) {
        const auto &r = requests[i];
        if (!r.key || !r.value || !r.liveKeys || !DraftDense(*r.liveKeys, INT32) || r.liveKeys->Count(0) != 1) return false;
        for (const Data *cache : {r.key, r.value}) {
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != query.dataDeviceIds || !cache->cudaData || cache->isPagedKVCache ||
                cache->multiDeviceData || cache->dims.size() != 3 || cache->dims[0] != 1 ||
                cache->dims[2] != kvHeads * dim || cache->expansionDims.size() != 3 ||
                cache->expansionDims[1] < capacity || cache->strides[1] != kvHeads * dim || cache->strides[2] != 1)
                return false;
        }
        descriptors.push_back({(const BF16 *)query.cudaData + (size_t)i * rows * heads * dim,
            (BF16 *)r.key->cudaData, (BF16 *)r.value->cudaData, nullptr, nullptr,
            (const int *)r.liveKeys->cudaData, nullptr, nullptr, r.shortAttention});
    }
    Output(output, BFLOAT16, query.dims);

#ifdef FASTLLM_NAIVE_SWA_FLASHINFER
    if (fused) {
        constexpr size_t shared = sizeof(DraftFlashTraits::SharedStorageSingle);
        static_assert(shared <= 49152 && !DraftFlashTraits::IsInvalid());
        DraftFlashParams params(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            heads, kvHeads, rows, capacity, heads * dim, dim, kvHeads * dim, dim,
            dim, window - 1, 0.f, 1.f / sqrtf(float(dim)), 1.f, 10000.f);
        for (int i = 0; i < count; ++i)
            descriptors[i].output = (BF16 *)output.cudaData + (size_t)i * rows * heads * dim;
        DeviceDraftBatch batch{DeviceDescriptors(descriptors, scores),
            cacheBindings ? (const uint64_t *)cacheBindings->cudaData : nullptr};
        BatchDraftFlash<<<dim3((rows * (heads / kvHeads) + 63) / 64, count, kvHeads),
            dim3(32, 4, 1), shared, cudaStreamPerThread>>>(batch, params);
        CheckLaunch(); return true;
    }
#endif
    Output(scores, FLOAT32, {query.dims[1], heads, capacity});
    for (int i = 0; i < count; ++i) {
        descriptors[i].output = (BF16 *)output.cudaData + (size_t)i * rows * heads * dim;
        descriptors[i].scores = (float *)scores.cudaData + (size_t)i * rows * heads * capacity;
    }
    const bool hopper = FastllmCudaRuntimeArch() == 90;
    for (int first = 0; first < count; first += kDraftBatchLimit) {
        const int n = std::min(kDraftBatchLimit, count - first);
        DraftBatch batch{}; std::copy_n(descriptors.begin() + first, n, batch.rows);
        bool shortRows = false, longRows = false;
        for (int i = 0; i < n; ++i) (batch.rows[i].shortAttention ? shortRows : longRows) = true;
        if (shortRows) BatchDraftShort<<<dim3(heads, rows * n), 256>>>(batch, rows, heads, kvHeads, dim, window);
        if (!longRows) continue;
        if (hopper && dim == 128 && window <= 1024 && (heads / kvHeads) % 4 == 0)
            BatchDraftScores<128, 0, true><<<dim3(heads / 4, rows * n, 16), 128>>>(batch, rows, heads, kvHeads, dim, window);
        else {
            auto kernel = dim == 128 ? BatchDraftScores<128, 64>
                : dim <= 192 ? BatchDraftScores<192> : BatchDraftScores<256>;
            kernel<<<dim3(heads, rows * n, dim == 128 ? (capacity + 63) / 64 : 1), 128>>>(batch, rows, heads, kvHeads, dim, window);
        }
        BatchDraftSoftmax<<<dim3(rows * heads, n), 256>>>(batch, rows, heads, window);
        const int headTile = dim == 128 && hopper && (int64_t)heads * rows * n >= 256
            ? ((heads / kvHeads) % 8 == 0 ? 8 : (heads / kvHeads) % 4 == 0 ? 4 : 1) : 1;
        if (headTile == 8)
            BatchDraftValues<32, 2, false, 8><<<dim3(heads / 8, rows * n, dim / 32), 256>>>(batch, rows, heads, kvHeads, dim, window);
        else if (headTile == 4)
            BatchDraftValues<32, 2, false, 4><<<dim3(heads / 4, rows * n, dim / 32), 128>>>(batch, rows, heads, kvHeads, dim, window);
        else if (dim == 128 && hopper)
            BatchDraftValues<32, 2><<<dim3(heads, rows * n, dim / 32), 128>>>(batch, rows, heads, kvHeads, dim, window);
        else if (dim == 128)
            BatchDraftValues<><<<dim3(heads, rows * n, dim / 16), 256>>>(batch, rows, heads, kvHeads, dim, window);
        else
            BatchDraftValues<16, 8, true><<<dim3((heads + 3) / 4, rows * n), 128>>>(batch, rows, heads, kvHeads, dim, window);
    }
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

bool FastllmCudaNaivePrepareDraftKVSlots(const std::vector<FastllmNaiveDraftKVSlot> &slots,
        fastllm::Data &scratch) {
    const size_t bytes = slots.size() * sizeof(FastllmNaiveDraftKVSlot);
    if (!bytes || !scratch.cudaData || scratch.GetBytes() < bytes) return false;
    return cudaMemcpyAsync(scratch.cudaData, slots.data(), bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess;
}

void FastllmCudaNaiveWriteAcceptedDraftKV(const fastllm::Data &raw, const void *slots, const int *emitted,
        int batch, int rows, int layers, int heads, int dim, float eps, float theta) {
    DraftNormRopeWriteAccepted<<<dim3(batch * rows * heads, layers), 256, 0, cudaStreamPerThread>>>(
        (const BF16 *)raw.cudaData, (const FastllmNaiveDraftKVSlot *)slots, emitted,
        batch, rows, heads, dim, eps, theta);
    CheckLaunch();
}

bool FastllmCudaNaiveBatchDraftKV(const std::vector<FastllmNaiveDraftKVUpdate> &updates,
        int heads, int dim, int window, int reserve, float eps, float theta) {
    using namespace fastllm;
    if (updates.empty() || updates.size() > INT_MAX || heads <= 0 || dim <= 0 || dim > 256 || dim % 2 ||
        window <= 1 || reserve < 0 || (int64_t)heads * dim > INT_MAX / 2 ||
        !std::isfinite(eps) || eps < 0 || !std::isfinite(theta) || theta <= 0) return false;
    const int width = heads * dim;
    const int device = FastllmCudaGetDevice();
    const std::vector<int> devices{device};
    for (const auto &u : updates) {
        if (!u.raw || !u.norm || !u.key || !u.value || u.start < 0 ||
            !DraftDense(*u.raw, BFLOAT16, device) || !DraftDense(*u.norm, FLOAT32, device) ||
            u.norm->dims.size() != 1 || u.norm->dims[0] != dim || u.raw->dims.size() != 3 ||
            u.raw->dims[0] != 1 || u.raw->dims[2] != 2 * width ||
            u.firstRow < 0 || u.firstRow >= u.raw->dims[1] || u.rowCount < -1) return false;
        const int available = u.raw->dims[1] - u.firstRow;
        const int rows = u.rowCount == -1 ? available : u.rowCount;
        if (rows <= 0 || rows > available || (int64_t)u.start + rows > INT_MAX ||
            (int64_t)16 * rows * heads > INT_MAX || u.key->dims != u.value->dims) return false;
        for (const Data *cache : {u.key, u.value}) {
            if (cache->dims.empty()) { if (!cache->expansionDims.empty()) return false; continue; }
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != devices || !cache->cudaData || cache->multiDeviceData ||
                cache->dims.size() != 3 || cache->dims[0] != 1 || cache->dims[2] != width || cache->dims[1] < 0 ||
                (int64_t)cache->dims[1] + rows > INT_MAX || cache->strides.size() != 3 ||
                cache->strides[2] != 1 || cache->strides[1] != width ||
                (!cache->expansionDims.empty() && (cache->expansionDims.size() != 3 ||
                    cache->expansionDims[0] != 1 || cache->expansionDims[2] != width))) return false;
        }
    }
    const bool vectorTrim = updates.size() >= 8 && window > kTrimMaxRows &&
        window - 1 <= 1024 && width % 8 == 0 &&
        BulkKernelSupported(BatchDraftTrimVector<64>, 1024 * 64, 80);
    // A descriptor can be a layer of any request. Chunking bounds the kernel
    // parameter area, without imposing a request/layer limit or sharing KV state.
    for (size_t first = 0; first < updates.size(); first += kDraftKVLayersPerLaunch) {
        DraftKVDestinations dst{}; DraftTrimBatch trim{};
        int count = std::min<size_t>(kDraftKVLayersPerLaunch, updates.size() - first), rows = 0;
        bool needsTrim = false;
        for (int i = 0; i < count; ++i) {
            const auto &u = updates[first + i];
            const int old = u.key->dims.empty() ? 0 : u.key->dims[1];
            const int sourceRows = u.rowCount == -1 ? u.raw->dims[1] - u.firstRow : u.rowCount;
            const int n = std::min(sourceRows, window - 1);
            const int skip = sourceRows - n;
            const int keep = std::min(old, window - 1 - n);
            rows = std::max(rows, n);
            dst.raw[i] = (const BF16 *)u.raw->cudaData + size_t(u.firstRow + skip) * 2 * width;
            dst.norm[i] = (const float *)u.norm->cudaData;
            dst.length[i] = keep; dst.rows[i] = n; dst.start[i] = u.start + skip;
            for (Data *cache : {u.key, u.value}) {
                int64_t wanted = std::max(keep + n, reserve);
                // Owned, initialized caches were preflighted above. Their
                // placement and capacity stay fixed through warm decode;
                // views, initialization and growth retain ToDevice's checks.
                if (!cache->isKVCache || cache->isFake ||
                    !FastllmNaiveCanResizeKVLength(*cache, (int)wanted, width)) {
                    cache->dataType = BFLOAT16; cache->UpdateUnitSize();
                    cache->ToDevice(DataDevice::CUDA, devices);
                    if (cache->expansionDims.empty() || cache->expansionDims[1] < wanted)
                        cache->Expansion({1, (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128), width});
                }
                FastllmNaiveResizeKVLength(*cache, keep + n, width); cache->isKVCache = true;
            }
            dst.key[i] = trim.key[i] = (BF16 *)u.key->cudaData;
            dst.value[i] = trim.value[i] = (BF16 *)u.value->cudaData;
            trim.width[i] = width; trim.keep[i] = keep;
            trim.drop[i] = old - keep; needsTrim |= keep > 0 && trim.drop[i] > 0;
        }
        // Evict before appending: old+new can approach two windows during
        // prefix restore/chunked prefill, but only the retained suffix is live.
        // Skipped raw rows keep their absolute RoPE positions via dst.start.
        if (needsTrim) {
            if (vectorTrim) {
                BatchDraftTrimVector<64><<<dim3(2 * ((width + 31) / 32), count), 256, 1024 * 64>>>(trim);
            } else if (window - 1 <= kTrimMaxRows) {
                constexpr int columns = kTrimTileBytes / sizeof(BF16);
                BatchDraftTrim<kTrimMaxRows, kTrimTileBytes><<<dim3(2 * ((width + columns - 1) / columns), count), 256>>>(trim);
            } else if (window - 1 <= 1024) {
                BatchDraftTrim<1024, 32><<<dim3(2 * ((width + 15) / 16), count), 256>>>(trim);
            } else BatchDraftTrimColumns<<<dim3((2 * width + 255) / 256, count), 256>>>(trim);
            CheckLaunch();
        }
        DraftNormRopeWrite<<<count * rows * heads, 256, 0, cudaStreamPerThread>>>(dst, rows, heads, dim, eps, theta);
        CheckLaunch();
    }
    return true;
}

bool FastllmCudaNaiveDraftKV(const std::vector<const fastllm::Data *> &raw,
        const std::vector<const fastllm::Data *> &norm, int start,
        std::vector<std::pair<fastllm::Data, fastllm::Data>> &kv,
        int heads, int dim, int window, int reserve, float eps, float theta) {
    if (raw.empty() || raw.size() != norm.size() || (!kv.empty() && kv.size() != raw.size())) return false;
    std::vector<std::pair<fastllm::Data, fastllm::Data>> empty;
    if (kv.empty()) empty.resize(raw.size());
    auto &target = kv.empty() ? empty : kv;
    std::vector<FastllmNaiveDraftKVUpdate> updates;
    for (size_t i = 0; i < raw.size(); ++i) updates.push_back({raw[i], norm[i], &target[i].first, &target[i].second, start});
    if (!FastllmCudaNaiveBatchDraftKV(updates, heads, dim, window, reserve, eps, theta)) return false;
    if (!empty.empty()) kv.swap(empty);
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
            LaunchSelectTopKCompact((const float *)scores.cudaData,
                input, begin, queries, keys, queryStart, topK, sortStride);
        } else {
            EncodeTopK<<<dim3((keys + 255) / 256, queries), 256>>>(
                (const float *)scores.cudaData, input, begin, queries, keys, queryStart);
        }
        if (blockSort) {
            auto sort = SortTopKCompact<>;
            if (queries >= 128 && keys >= 16384) {
                const int arch = FastllmCudaRuntimeArch();
                if (arch == 100 || arch == 103) sort = SortTopKCompact<false, 256, true>;
            }
            sort<<<queries, 256>>>(input, begin, (int *)indices.cudaData, queries, sortStride, topK, keys);
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
#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
    if (packedQkv && dim == 192 && valueDim == 128 && rotaryDim == 64 && q.dims[1] >= 1024 &&
        addresses % 16 == 0 && (FastllmCudaRuntimeArch() == 100 || FastllmCudaRuntimeArch() == 103)) {
        const int vectors = (heads + kvHeads) * 24 + kvHeads * 16;
        RopePackedPrefill<<<dim3(q.dims[1], (vectors + 127) / 128), 128>>>(
            (BF16 *)q.cudaData, (BF16 *)k.cudaData, (BF16 *)v.cudaData, (const float *)positions.cudaData,
            heads, kvHeads, theta, __float2bfloat16_rn(valueScale), (const BF16 *)packedQkv->cudaData);
        CheckLaunch();
        return;
    }
#endif
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
        int rotaryDim, float theta, float valueScale, int window, const fastllm::Data *packedQkv, bool ring, RopeCacheRow &row, int &indexDim, const fastllm::Data *indexCache = nullptr, int indexContextRank = 0) {
    using namespace fastllm;
    const std::vector<int> device{FastllmCudaGetDevice()};
    const bool dynamicLength = !liveKeys.dims.empty();
    auto resident = [&](const Data &x, DataType type) {
        return x.dataType == type && x.dataDevice == DataDevice::CUDA &&
            x.dataDeviceIds == device && x.cudaData;
    };
    auto matrix = [&](const Data &x, int64_t columns, bool cache = false) {
        return resident(x, cache ? x.dataType : BFLOAT16) && (!cache || FastllmNaiveAttentionKVType(x.dataType)) && x.dims.size() == 3 && x.dims[0] == 1 &&
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
    const int64_t kc = (int64_t)kvHeads * dim + (indexCache ? 0 : (key.kvCacheFp8Indexer
        ? FastllmNaiveIndexerBytes / key.unitSize : indexDim)), vc = (int64_t)kvHeads * valueDim;
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
    row.fp4 = key.dataType == FP4_E2M1;
    row.indexFp8 = indexCache ? indexCache->dataType == FP8_E4M3 : row.fp8;
    if (row.fp4 && (window || ring || dim % 16 || valueDim % 16 || dim > 256 || valueDim > 256 ||
                    key.kvCacheFp8Indexer || (indexDim && !indexCache))) return false;
    row.scaledIndexer = indexCache ? indexCache->kvCacheFp8Indexer : key.kvCacheFp8Indexer;
    if (indexCache) {
        const int width = row.scaledIndexer ? FastllmNaiveIndexerBytes / indexCache->unitSize : indexDim;
        if (indexDim != 128 || !matrix(*indexCache, width, true) ||
            indexCache->dataType != (row.fp4 ? FP8_E4M3 : key.dataType) ||
            indexCache->expansionDims.size() != 3 || indexCache->isPagedKVCache != key.isPagedKVCache) return false;
        if (key.isPagedKVCache) {
            // The Indexer borrows exactly K's page IDs. Serving can extend the
            // table without changing this graph's address or physical pool.
            if (indexCache->cudaData != key.cudaData || indexCache->pageLen <= 0 ||
                key.pageLen % indexCache->pageLen || indexContextRank < 0 ||
                indexContextRank >= key.pageLen / indexCache->pageLen ||
                indexCache->pagedKVCacheData->maxPages < key.pagedKVCacheData->maxPages) return false;
        } else if (indexCache->expansionDims[1] < keyRows) return false;
        row.indexCache = indexCache->isPagedKVCache ? indexCache->pagedKVCacheData->cudaData : indexCache->cudaData;
        row.separateIndex = true;
        row.indexPageLen = indexCache->pageLen;
        row.indexContextRank = indexContextRank;
    }
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
        int rotaryDim, float theta, float valueScale, int window, const fastllm::Data *packedQkv, bool ring, bool ringVerify) {
    RopeCacheRow row{};
    int indexDim = 0;
    if (!PrepareRopeCache(q, k, v, indexKey, positions, key, value, liveKeys,
            heads, kvHeads, dim, valueDim, rotaryDim, theta, valueScale, window,
            packedQkv, ring, row, indexDim)) return false;
    if (ringVerify && (ring || window != kSwaWindow || !row.length || q.dims[1] > 8 ||
        std::min(row.keyRows, row.valueRows) < kSwaWindow + q.dims[1])) return false;
    row.ringVerify = ringVerify;
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
                ring, descriptors[b], width, rows[b].indexCache, rows[b].indexerContextRank) || (b && width != indexDim)) return false;
        indexDim = width;
        auto &descriptor = descriptors[b];
        descriptor.cache = rows[b].cache;
        if (rows[b].verifyOffset >= 0) {
            if (ring || window != kSwaWindow || rows[b].verifyOffset >= 8 ||
                std::min(descriptor.keyRows, descriptor.valueRows) <= kSwaWindow + rows[b].verifyOffset)
                return false;
            descriptor.ringVerify = true;
            descriptor.verifyOffset = rows[b].verifyOffset;
        } else if (rows[b].cacheOffset >= 0) {
            if (ring || rows[b].cacheOffset >= std::min(descriptor.keyRows, descriptor.valueRows)) return false;
            descriptor.length = nullptr;
            descriptor.cachePast = rows[b].cacheOffset;
        }
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
    // SM60 retains the same per-row math without the SM70 large-argument ABI.
    if (FastllmCudaRuntimeArch() < 70) {
        auto kernel = packedQkv ? RopeAppendCache<true> : RopeAppendCache<false>;
        for (const auto &descriptor : descriptors)
            kernel<<<heads + 2 * kvHeads + (indexDim != 0), 128>>>(descriptor,
                heads, kvHeads, dim, valueDim, indexDim, rotaryDim, theta,
                __float2bfloat16_rn(valueScale), window, ring);
        CheckLaunch();
        return true;
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const bool staged = std::any_of(rows.begin(), rows.end(),
        [](const FastllmNaiveRopeCacheRow &r) { return r.verifyOffset >= 0 || r.cacheOffset >= 0; });
#if CUDART_VERSION >= 12010
    if (!ring && window == kSwaWindow && indexDim == 0 && count > kRopeCacheBatch &&
        CanPackVerifyRope(descriptors, heads, kvHeads, dim, valueDim)) {
        for (int first = 0; first < count; first += kVerifyRopeRows) {
            const int n = std::min(kVerifyRopeRows, count - first);
            VerifyRopeBatch batch{};
            batch.geometry = descriptors[first];
            for (int i = 0; i < n; ++i)
                batch.rows[i] = {descriptors[first + i].cache, descriptors[first + i].verifyOffset};
            auto kernel = packedQkv ? VerifyRopeAppendCache<true> : VerifyRopeAppendCache<false>;
            kernel<<<dim3(heads + 2 * kvHeads, n), 128>>>(batch, heads, kvHeads, dim,
                valueDim, rotaryDim, theta, __float2bfloat16_rn(valueScale));
        }
        CheckLaunch();
        return true;
    }
#endif
    // Compact only graph-bound decode requests with the screened geometry.
    // Validate the complete batch before any kernel can modify Q or KV.
#if CUDART_VERSION >= 12010
    const bool compactGeometry = !staged && count >= 80 && count <= kCompactRopeRequests &&
        heads == 16 && kvHeads == (ring ? 2 : 1) && dim == 192 && valueDim == 128 &&
        rotaryDim == 64 && indexDim == (ring ? 0 : 128) && window == (ring ? 128 : 0);
    if (compactGeometry && FastllmCudaRuntimeArch() == 90) {
        CompactRopeRequests packed{};
        if (PackRope(descriptors.data(), count, packed)) {
            auto kernel = packedQkv ? RequestRope<true, CompactRopeRequests, 1>
                                    : RequestRope<false, CompactRopeRequests, 1>;
            if (ring)
                kernel = packedQkv ? RequestRope<true, CompactRopeRequests, 2>
                                   : RequestRope<false, CompactRopeRequests, 2>;
            else if (packed.fp4)
                kernel = packedQkv ? RequestRope<true, CompactRopeRequests, 1, true>
                                   : RequestRope<false, CompactRopeRequests, 1, true>;
            kernel<<<count, 128>>>(packed, heads, kvHeads, dim, valueDim, indexDim,
                rotaryDim, theta, __float2bfloat16_rn(valueScale), window, ring);
            CheckLaunch();
            return true;
        }
    }
#endif
    // Validate every cache before launching, so a fallback cannot double-rotate Q.
    for (int first = 0; first < count; first += kRopeCacheBatch) {
        const int batchSize = std::min(kRopeCacheBatch, count - first);
        RopeCacheBatch batch{};
        std::copy_n(descriptors.begin() + first, batchSize, batch.rows);
        // One CTA per request amortizes angle calculation for sufficiently
        // large batches. Keep the original launch for small or wider shapes.
        const bool requestFusion = !staged && batchSize >= 80 && rotaryDim > 0 &&
            std::none_of(batch.rows, batch.rows + batchSize, [](const auto &r) { return r.fp4; }) &&
            rotaryDim <= 192 && rotaryDim % 2 == 0 && rotaryDim <= dim &&
            (!indexDim || (indexDim == 128 && rotaryDim <= indexDim));
        auto kernel = requestFusion ? (packedQkv ? RequestRope<true> : RequestRope<false>)
                                    : (packedQkv ? BatchRopeAppendCache<true> : BatchRopeAppendCache<false>);
        const dim3 grid = requestFusion ? dim3(batchSize)
            : dim3(heads + 2 * kvHeads + (indexDim != 0), batchSize);
        kernel<<<grid, 128>>>(batch,
            heads, kvHeads, dim, valueDim, indexDim, rotaryDim, theta,
            __float2bfloat16_rn(valueScale), window, ring);
    }
    CheckLaunch();
    return true;
#else
    return false;
#endif
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
    if (std::any_of(rows.begin(), rows.end(), [](const FastllmNaiveRopeCacheRow &r) {
            return r.verifyOffset >= 0 || r.cacheOffset >= 0; })) return false;
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
    if (!first.cache || first.separateIndex || first.fp4) return false;
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

#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM90
static unsigned LaunchFP8ShortlistIndexer(const naive_indexer::FP8Entry *entries, int count,
                                          int *const *outputs) {
    using namespace fastllm;
    naive_indexer::FP8Entry eligible[8];
    if (count < 1 || count > 8) return 0;
    for (int i = 0; i < count; ++i) {
        eligible[i] = entries[i];
        if (entries[i].keys <= 4096) eligible[i].rows = 0;
    }
    Data stagedStorage(FP8_E4M3);
    naive_indexer::FP8Entry staged[8];
    unsigned mask = naive_indexer::LaunchFP8(eligible, count, stagedStorage, staged);
    for (int i = 0; i < count; ++i) if (mask & (1u << i)) {
        const auto &e = staged[i];
        const BF16 *expanded = nullptr;
        Data candidates, offsets;
        const int parts = e.part ? 2 : 1;
        for (int part = 0; part < parts; ++part) {
            const int partRows = e.part ? e.part : e.rows;
            for (int offset = 0; offset < partRows; offset += 512) {
                const int rows = std::min(512, partRows - offset), first = part * partRows + offset;
                Output(candidates, INT32, {rows, 4096, 2});
                Output(offsets, INT32, {2, rows});
                auto *selected = (unsigned long long *)candidates.cudaData;
                auto *ends = (int *)offsets.cudaData;
                // Keep a bounded superset of the FP8 top 3072 before exact
                // candidate scoring. The original selector proves its size.
                LaunchSelectTopKCompact(e.scores + (size_t)first * e.keys, selected, ends,
                    rows, e.keys, e.past + first + (part ? e.gap : 0), 3072, 4096);
                if (!first && naive_indexer::StageRefinementKeys(e, rows))
                    expanded = (const BF16 *)e.scores;
                naive_indexer::FP8Entry tile = e;
                tile.q += (size_t)first * 2048; tile.scales += first * 16; tile.weights += first * 16;
                tile.rows = rows; tile.past += first + (part ? e.gap : 0); tile.part = tile.gap = 0;
                naive_indexer::RefineFP8(tile, selected, ends, 4096, expanded);
                // LaunchFP8 has already selected Hopper. Other architectures
                // retain the original BF16 scorer and generic compact sort.
                SortTopKCompact<true, 128><<<rows, 128>>>(selected, ends, outputs[i] + (size_t)first * 2048,
                    rows, 4096, 2048, e.keys);
            }
        }
    }
    return mask;
}
#endif

#ifdef FASTLLM_ENABLE_BLACKWELL_DSA
bool FastllmCudaTryBlackwellScoreAttention(const FastllmNaivePrefillAttentionItem &, const fastllm::Data &,
    int, int, int, int, const fastllm::Data &, const fastllm::Data &);
#endif
namespace {
struct PairedIndexerAttention {
    const FastllmNaivePrefillAttentionItem &item;
    const fastllm::Data &sink, &scratch;
    int heads, kvHeads, dim, valueDim;
};
}

static bool NaiveIndexer(const fastllm::Data &query, const fastllm::Data &weights,
                            const fastllm::Data &packedKeys, int heads, int dim,
                            int queryStart, int topK, bool fp8Query, fastllm::Data &indices, bool peerKeys, fastllm::Data *scoreScratch, bool fp8Compute,
        const PairedIndexerAttention *paired) {
    using namespace fastllm;
    // Indexer quantization is defined on 128-element blocks. The score kernel
    // has a general head-count path, but a different block width is unsupported.
    AssertInFastLLM(dim == 128 && heads > 0 && query.dims.size() == 3 &&
        query.dims[0] == 1 && query.dims[1] > 0 && query.dims[2] == (int64_t)heads * dim &&
        packedKeys.dims.size() == 3 && packedKeys.dims[0] == 1 &&
        packedKeys.dims[2] >= (packedKeys.kvCacheFp8Indexer ? FastllmNaiveIndexerBytes / packedKeys.unitSize : dim) &&
        queryStart >= 0 && (int64_t)queryStart + query.dims[1] <= packedKeys.dims[1] && topK > 0 &&
        query.dataType == BFLOAT16 && FastllmNaiveKVType(packedKeys.dataType) && weights.dataType == BFLOAT16 &&
        query.dataDevice == DataDevice::CUDA && packedKeys.dataDevice == DataDevice::CUDA &&
        weights.dataDevice == DataDevice::CUDA && (peerKeys || query.dataDeviceIds == packedKeys.dataDeviceIds) &&
        query.dataDeviceIds == weights.dataDeviceIds && query.cudaData && packedKeys.cudaData &&
        weights.cudaData && weights.Count(0) == (uint64_t)query.dims[1] * heads,
        "Invalid Naive-N0.5 Indexer layout (requires 128-element BF16 blocks).");
    int queries = query.dims[1], keys = packedKeys.dims[1];
    // Bound prefill score storage even when a ragged tail cannot use TP
    // query sharding. Keep 64-row boundaries and merge a short final tile
    // so splitting does not switch its score arithmetic to the decode or
    // small-query fallback. The final tile adds at most 63 score rows.
    constexpr size_t scoreBudget = size_t(256) << 20;
    const int scoreRows = std::max<size_t>(64, scoreBudget / (size_t(keys) * sizeof(float)) / 64 * 64);
    if (queries > scoreRows + 63) {
        Output(indices, INT32, {queries, topK});
        for (int first = 0; first < queries;) {
            int count = std::min(scoreRows, queries - first);
            if (queries - first - count < 64) count = queries - first;
            Data tileQuery(BFLOAT16, {1, count, heads * dim});
            Data tileWeights(BFLOAT16, {1, count, heads});
            Data tileIndices(INT32, {count, topK});
            tileQuery.FakeFrom(query, size_t(first) * heads * dim * query.unitSize);
            tileWeights.FakeFrom(weights, size_t(first) * heads * weights.unitSize);
            tileIndices.FakeFrom(indices, size_t(first) * topK * sizeof(int));
            FastllmCudaNaiveIndexer(tileQuery, tileWeights, packedKeys, heads, dim,
                queryStart + first, topK, fp8Query, tileIndices, peerKeys, scoreScratch, fp8Compute);
            first += count;
        }
        return false;
    }
    int stride = packedKeys.dims[2];
    bool fp8IndicesReady = false;
    bool decodeTopK = queries == 1 && heads == 16;
    int selectBlocks = decodeTopK ? CooperativeTopKBlocks(queryStart + 1, topK) : 0;
    Data q, k, localScores, positions;
    Data &scores = scoreScratch ? *scoreScratch : localScores;
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
        const int arch = queries > 1 ? FastllmCudaRuntimeArch() : 0;
        bool scored = false;
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM100
        if (!peerKeys && (arch == 100 || arch == 103))
            scored = FastllmCudaNaivePrefillIndexerSm100(static_cast<const BF16 *>(query.cudaData),
                static_cast<const BF16 *>(weights.cudaData), FastllmNaiveKV(packedKeys), stride, queries, keys,
                queryStart, q.cudaData, q.GetBytes(), static_cast<float *>(qScale.cudaData),
                static_cast<float *>(scores.cudaData), cudaStreamPerThread);
#endif
        if (!scored) {
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
            // WGMMA's smaller query tile repeats more K reads. For shared peer KV,
            // use the larger MMA tile below to reduce traffic to the owning GPU.
            else if ([&]() {
                naive_indexer::FP8Entry entry{FastllmNaiveKV(packedKeys),
                    (const BF16 *)q.cudaData, (const BF16 *)weights.cudaData, (const float *)qScale.cudaData,
                    (float *)scores.cudaData, queries, keys, queryStart, stride};
                if (!fp8Compute || topK != 2048) return false;
                Output(indices, INT32, {queries, topK});
                int *output = (int *)indices.cudaData;
                return fp8IndicesReady = LaunchFP8ShortlistIndexer(&entry, 1, &output) != 0;
            }()) {}
            else if (!peerKeys && naive_indexer::Prepare()) {
                naive_indexer::Launch((const BF16 *)q.cudaData, FastllmNaiveKV(packedKeys),
                    (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                    (float *)scores.cudaData, queries, keys, queryStart, stride);
            }
#endif
            else {
                // TopK consumes only each row's causal prefix. Blocks beyond
                // the final query's prefix cannot contribute any selected key.
                const bool pipeline = arch == 90 || (!peerKeys && (arch == 100 || arch == 103) &&
                    queries >= 4096 && keys >= 4096 && packedKeys.dataType == FP8_E4M3);
                const int rows = pipeline ? 32 : 64;
                auto kernel = pipeline ? naive_dsa_mma::IndexerScores<true> : naive_dsa_mma::IndexerScores<false>;
                kernel<<<dim3((queryStart + queries + kKeys - 1) / kKeys,
                    (queries + rows - 1) / rows), rows * 4>>>(
                    (const BF16 *)q.cudaData, FastllmNaiveKV(packedKeys),
                    (const float *)qScale.cudaData, (const BF16 *)weights.cudaData,
                    (float *)scores.cudaData, queries, keys, queryStart, stride, 0, 0);
            }
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
    if (fp8IndicesReady) return false;
#ifdef FASTLLM_ENABLE_BLACKWELL_DSA
    if (paired && FastllmCudaTryBlackwellScoreAttention(paired->item, paired->sink, paired->heads,
            paired->kvHeads, paired->dim, paired->valueDim, paired->scratch, scores)) return true;
#endif
    // The selector derives positions directly; CUB's fallback uses emitted positions.
    if (selectBlocks) {
        CooperativeTopK((const unsigned *)scores.cudaData, queryStart + 1, selectBlocks, indices);
        return false;
    }
    if (decodeTopK) {
        SortTopKPairs((const unsigned *)scores.cudaData, (const int *)positions.cudaData,
            queryStart + 1, topK, indices);
        CheckLaunch();
    } else {
        FastllmCudaNaiveTopK(scores, queryStart, topK, indices);
    }
    return false;
}

void FastllmCudaNaiveIndexer(const fastllm::Data &query, const fastllm::Data &weights,
        const fastllm::Data &keys, int heads, int dim, int past, int topK, bool fp8Query,
        fastllm::Data &indices, bool peer, fastllm::Data *scratch, bool fp8Compute) {
    NaiveIndexer(query, weights, keys, heads, dim, past, topK, fp8Query, indices, peer, scratch, fp8Compute, nullptr);
}

void FastllmCudaNaivePrefillIndexedAttention(const FastllmNaivePrefillIndexerItem &index,
        const FastllmNaivePrefillAttentionItem &attention, const fastllm::Data &sink,
        int indexHeads, int indexDim, int topK, bool fp8Query, bool fp8Compute,
        int heads, int kvHeads, int dim, int valueDim, fastllm::Data *scoreScratch,
        const fastllm::Data &attentionScratch) {
    const PairedIndexerAttention paired{attention, sink, attentionScratch, heads, kvHeads, dim, valueDim};
    const bool consumed = NaiveIndexer(*index.query, *index.weights, *index.keys, indexHeads, indexDim,
        index.past, topK, fp8Query, *index.indices, false, scoreScratch, fp8Compute, &paired);
    if (!consumed) FastllmCudaNaiveBatchIndexerAttention({attention}, sink, heads, kvHeads, dim, valueDim, attentionScratch);
}


#ifdef FASTLLM_NAIVE_DSA_MMA
static bool BalancedIndexerSupported(const fastllm::Data &query, const fastllm::Data &weights,
                                     const fastllm::Data &keys, int past, int rank, int ranks, bool fp8Query,
                                     bool peerKeys) {
    using namespace fastllm;
    if (!peerKeys || !fp8Query || FastllmCudaRuntimeArch() != 90 || (ranks != 2 && ranks != 4 && ranks != 8) ||
        rank < 0 || rank >= ranks || query.dataType != BFLOAT16 || weights.dataType != BFLOAT16 ||
        !FastllmNaiveKVType(keys.dataType) || query.dataDevice != DataDevice::CUDA ||
        weights.dataDevice != DataDevice::CUDA || keys.dataDevice != DataDevice::CUDA ||
        query.dataDeviceIds != weights.dataDeviceIds || !query.cudaData || !weights.cudaData || !keys.cudaData ||
        query.dims.size() != 3 || query.dims[0] != 1 || query.dims[2] != 2048 || query.dims[1] <= 0 ||
        query.dims[1] % (ranks * 128) || weights.Count(0) != (size_t)query.dims[1] * 16 || keys.dims.size() != 3 ||
        keys.dims[0] != 1 || keys.dims[2] < (keys.kvCacheFp8Indexer ? FastllmNaiveIndexerBytes / keys.unitSize : 128) ||
        past < 0 || (int64_t)past + query.dims[1] > keys.dims[1] || keys.dims[1] < 8192)
        return false;
    const int part = query.dims[1] / (2 * ranks), rows = part * 2, count = keys.dims[1];
    if ((int64_t)part * count < 1024 * 1024 || (uint64_t)rows * count * sizeof(float) > (size_t(512) << 20))
        return false;
    return true;
}
#endif
// Combine only layouts that previously selected the same peer-KV MMA path.
// Return false before allocating/writing output for all other shapes.
bool FastllmCudaNaiveBalancedIndexer(const fastllm::Data &query, const fastllm::Data &weights,
                                     const fastllm::Data &keys, int past, int rank, int ranks, bool fp8Query,
                                     bool peerKeys, fastllm::Data &indices, fastllm::Data *scoreScratch, bool fp8Compute) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    using namespace fastllm;
    if (!BalancedIndexerSupported(query, weights, keys, past, rank, ranks, fp8Query, peerKeys) ||
        !FastllmCudaFlashInferDataTypeSupported(BFLOAT16)) return false;
    const int part = query.dims[1] / (2 * ranks), rows = part * 2, count = keys.dims[1];
    const int first = rank * part, second = query.dims[1] - (rank + 1) * part;
    const int start = past + first, gap = second - first - part;
    Data q, scale, w, localScores, encoded, offsets;
    Data &scores = scoreScratch ? *scoreScratch : localScores;
    Output(q, BFLOAT16, {rows, 16, 128});
    Output(scale, FLOAT32, {rows, 16});
    Output(w, BFLOAT16, {rows, 16});
    Output(scores, FLOAT32, {rows, count});
    Output(encoded, INT32, {rows, 4096, 2});
    Output(offsets, INT32, {2, rows});
    Output(indices, INT32, {rows, 2048});
    naive_dsa_mma::QuantizeBalancedIndexer<<<rows * 16, 128>>>(
        (const BF16 *)query.cudaData, (const BF16 *)weights.cudaData, (BF16 *)q.cudaData, (float *)scale.cudaData,
        (BF16 *)w.cudaData, part, first, second);
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM90
    naive_indexer::FP8Entry entry{FastllmNaiveKV(keys), (const BF16 *)q.cudaData,
        (const BF16 *)w.cudaData, (const float *)scale.cudaData, (float *)scores.cudaData,
        rows, count, start, keys.dims[2], part, gap};
    int *output = (int *)indices.cudaData;
    if (fp8Compute && LaunchFP8ShortlistIndexer(&entry, 1, &output)) {
        CheckLaunch();
        return true;
    }
#endif
    naive_dsa_mma::IndexerScores<true><<<dim3((past + second + part + 63) / 64, (rows + 31) / 32), 128>>>(
        (const BF16 *)q.cudaData, FastllmNaiveKV(keys), (const float *)scale.cudaData, (const BF16 *)w.cudaData,
        (float *)scores.cudaData, rows, count, start, keys.dims[2], part, gap);
    LaunchSelectTopKCompact((const float *)scores.cudaData, (unsigned long long *)encoded.cudaData,
                                     (int *)offsets.cudaData, rows, count, start, 2048, 4096, part, gap);
    SortTopKCompact<><<<rows, 256>>>((const unsigned long long *)encoded.cudaData, (int *)offsets.cudaData,
                                   (int *)indices.cudaData, rows, 4096, 2048, count);
    CheckLaunch();
    return true;
#else
    return false;
#endif
}

#ifdef FASTLLM_NAIVE_DSA_MMA
static bool LocalPrefillIndexerSupported(const fastllm::Data &q, const fastllm::Data &w,
                                         const fastllm::Data &k, int past, bool fp8Query) {
    using namespace fastllm;
    return fp8Query && FastllmCudaRuntimeArch() == 90 && q.dataType == BFLOAT16 && w.dataType == BFLOAT16 &&
        FastllmNaiveKVType(k.dataType) && q.dataDevice == DataDevice::CUDA &&
        k.dataDevice == DataDevice::CUDA && w.dataDevice == DataDevice::CUDA &&
        q.dataDeviceIds == k.dataDeviceIds && q.dataDeviceIds == w.dataDeviceIds &&
        q.cudaData && w.cudaData && k.cudaData && q.dims.size() == 3 && q.dims[0] == 1 &&
        q.dims[1] >= 32 && q.dims[2] == 2048 && w.Count(0) == (size_t)q.dims[1] * 16 &&
        k.dims.size() == 3 && k.dims[0] == 1 &&
        k.dims[2] >= (k.kvCacheFp8Indexer ? FastllmNaiveIndexerBytes / k.unitSize : 128) &&
        past >= 0 && (int64_t)past + q.dims[1] <= k.dims[1] &&
        (int64_t)q.dims[1] * k.dims[1] >= 1024 * 1024;
}
#endif

static bool BatchPrefillIndexer(const std::vector<FastllmNaivePrefillIndexerItem> &items, int rank, int ranks,
                                          bool fp8Query, bool peerKeys, fastllm::Data *scoreScratch, bool fp8Compute) {
#ifdef FASTLLM_NAIVE_DSA_MMA
    using namespace fastllm;
    if (items.empty() || items.size() > 8 ||
        !FastllmCudaFlashInferDataTypeSupported(BFLOAT16))
        return false;
    naive_dsa_mma::PrefillIndexerBatch batch{};
    size_t scoreCount = 0;
    int maxRows = 0, maxKeys = 0;
    // Validate the entire batch before allocating or writing any output.
    for (const auto &item : items) {
        if (!item.query || !item.weights || !item.keys || !item.indices ||
            !(ranks == 1
                ? LocalPrefillIndexerSupported(*item.query, *item.weights, *item.keys, item.past, fp8Query)
                : BalancedIndexerSupported(*item.query, *item.weights, *item.keys, item.past, rank, ranks, fp8Query, peerKeys)) ||
            item.query->dataDeviceIds != items[0].query->dataDeviceIds)
            return false;
        const auto &q = *item.query, &key = *item.keys;
        int rows = q.dims[1] / ranks, part = ranks == 1 ? 0 : rows / 2;
        auto &e = batch.entries[batch.count++];
        e.query = (const BF16 *)q.cudaData;
        e.weights = (const BF16 *)item.weights->cudaData;
        e.key = FastllmNaiveKV(key);
        // FP8 refinement reuses consumed scores for 16-byte BF16 vector loads/stores.
        // Ragged request lengths must preserve that alignment between entries.
        scoreCount = (scoreCount + 3) & ~size_t(3);
        e.scoreOffset = scoreCount;
        e.rowOffset = batch.rows;
        e.rows = rows;
        e.part = part;
        e.first = rank * part;
        e.second = ranks == 1 ? 0 : q.dims[1] - (rank + 1) * part;
        e.past = item.past;
        e.keys = key.dims[1];
        e.stride = key.dims[2];
        scoreCount += (size_t)rows * e.keys;
        if (scoreCount * sizeof(float) > (size_t(512) << 20))
            return false;
        batch.rows += rows;
        maxRows = std::max(maxRows, rows);
        maxKeys = std::max(maxKeys, e.past + (part ? e.second + part : rows));
    }
    for (int i = 0; i < batch.count; ++i) {
        Output(*items[i].indices, INT32, {batch.entries[i].rows, 2048});
        batch.entries[i].output = (int *)items[i].indices->cudaData;
    }
    Data q, scale, w, localScores, encoded, offsets;
    Data &scores = scoreScratch ? *scoreScratch : localScores;
    Output(q, BFLOAT16, {batch.rows, 16, 128});
    Output(scale, FLOAT32, {batch.rows, 16});
    Output(w, BFLOAT16, {batch.rows, 16});
    Output(scores, FLOAT32, {(int)scoreCount});
    Output(encoded, INT32, {batch.rows, 4096, 2});
    Output(offsets, INT32, {2, batch.rows});
    naive_dsa_mma::QuantizeBatchIndexer<<<batch.rows * 16, 128>>>(batch, (BF16 *)q.cudaData, (float *)scale.cudaData,
                                                                  (BF16 *)w.cudaData);
    unsigned fp8Mask = 0;
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM90
    naive_indexer::FP8Entry fp8Entries[8];
    for (int i = 0; i < batch.count; ++i) {
        const auto &e = batch.entries[i];
        fp8Entries[i] = {e.key, (const BF16 *)q.cudaData + (size_t)e.rowOffset * 2048,
            (const BF16 *)w.cudaData + e.rowOffset * 16, (const float *)scale.cudaData + e.rowOffset * 16,
            (float *)scores.cudaData + e.scoreOffset, e.rows, e.keys, e.past + e.first, e.stride,
            e.part, e.second - e.first - e.part};
    }
    int *outputs[8];
    for (int i = 0; i < batch.count; ++i) outputs[i] = batch.entries[i].output;
    if (fp8Compute) fp8Mask = LaunchFP8ShortlistIndexer(fp8Entries, batch.count, outputs);
#endif
    if (fp8Mask || ranks == 1) {
        for (int i = 0; i < batch.count; ++i) if (!(fp8Mask & (1u << i))) {
            const auto &item = items[i];
            if (ranks == 1) {
                FastllmCudaNaiveIndexer(*item.query, *item.weights, *item.keys, 16, 128,
                    item.past, 2048, fp8Query, *item.indices, false, scoreScratch, fp8Compute);
                continue;
            }
            AssertInFastLLM(FastllmCudaNaiveBalancedIndexer(*item.query, *item.weights, *item.keys,
                item.past, rank, ranks, fp8Query, peerKeys, *item.indices, scoreScratch, false),
                "Validated Indexer batch fallback failed.");
        }
        CheckLaunch();
        return true;
    }
    naive_dsa_mma::BatchIndexerScores<<<dim3((maxKeys + 63) / 64, (maxRows + 31) / 32, batch.count), 128>>>(
        batch, (const BF16 *)q.cudaData, (const float *)scale.cudaData, (const BF16 *)w.cudaData,
        (float *)scores.cudaData);
    SelectBatchTopK<<<batch.rows, 256>>>(batch, (const float *)scores.cudaData, (unsigned long long *)encoded.cudaData,
                                         (int *)offsets.cudaData);
    SortBatchTopK<<<batch.rows, 256>>>(batch, (const unsigned long long *)encoded.cudaData,
                                       (const int *)offsets.cudaData);
    CheckLaunch();
    return true;
#else
    return false;
#endif
}
bool FastllmCudaNaiveBatchBalancedIndexer(const std::vector<FastllmNaivePrefillIndexerItem> &items,
        int rank, int ranks, bool fp8Query, bool peerKeys, fastllm::Data *scratch, bool fp8Compute) {
    if (ranks != 2 && ranks != 4 && ranks != 8) return false;
    return BatchPrefillIndexer(items, rank, ranks, fp8Query, peerKeys, scratch, fp8Compute);
}

void FastllmCudaNaiveBatchIndexer(const std::vector<FastllmNaivePrefillIndexerItem> &items,
        int heads, int dim, int topK, bool fp8Query, fastllm::Data *scratch, bool fp8Compute) {
    using namespace fastllm;
    std::vector<FastllmNaivePrefillIndexerItem> tiles;
    std::vector<std::unique_ptr<Data>> views;
    size_t bytes = 0;
    auto flush = [&] {
        if (tiles.empty()) return;
        if (tiles.size() == 1 || !BatchPrefillIndexer(tiles, 0, 1, fp8Query, false, scratch, fp8Compute))
            for (const auto &item : tiles)
                FastllmCudaNaiveIndexer(*item.query, *item.weights, *item.keys, heads, dim,
                    item.past, topK, fp8Query, *item.indices, false, scratch, fp8Compute);
        tiles.clear(); views.clear(); bytes = 0;
    };
    for (const auto &item : items) {
        bool supported = false;
#ifdef FASTLLM_NAIVE_DSA_MMA
        supported = heads == 16 && dim == 128 && topK == 2048 && fp8Compute &&
            (item.keys->dataType == FP8_E4M3 || item.keys->kvCacheFp8Indexer) &&
            LocalPrefillIndexerSupported(*item.query, *item.weights, *item.keys, item.past, fp8Query);
#endif
        if (!supported) {
            flush();
            FastllmCudaNaiveIndexer(*item.query, *item.weights, *item.keys, heads, dim,
                item.past, topK, fp8Query, *item.indices, false, scratch, fp8Compute);
            continue;
        }
        const int rows = item.query->dims[1], keys = item.keys->dims[1];
        // Match the ordinary Indexer's tile boundaries and its merged short tail.
        const int step = std::max<size_t>(64, (size_t(256) << 20) / (size_t(keys) * 4) / 64 * 64);
        Output(*item.indices, INT32, {rows, topK});
        for (int first = 0; first < rows;) {
            int count = std::min(step, rows - first);
            if (rows - first - count < 64) count = rows - first;
            const size_t tileBytes = size_t(count) * keys * 4;
            if (tiles.size() == 8 || bytes + tileBytes > (size_t(512) << 20)) flush();
            auto q = std::make_unique<Data>(BFLOAT16, std::vector<int>{1, count, heads * dim});
            auto w = std::make_unique<Data>(BFLOAT16, std::vector<int>{1, count, heads});
            auto out = std::make_unique<Data>(INT32, std::vector<int>{count, topK});
            q->FakeFrom(*item.query, size_t(first) * heads * dim * item.query->unitSize);
            w->FakeFrom(*item.weights, size_t(first) * heads * item.weights->unitSize);
            out->FakeFrom(*item.indices, size_t(first) * topK * sizeof(int));
            tiles.push_back({q.get(), w.get(), item.keys, out.get(), item.past + first});
            views.push_back(std::move(q)); views.push_back(std::move(w)); views.push_back(std::move(out));
            bytes += tileBytes; first += count;
        }
    }
    flush();
}

void FastllmCudaNaiveVerifyIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &packedKeys,
        int heads, int dim, int queryStart, int topK, bool fp8Query,
        fastllm::Data &indices) {
    using namespace fastllm;
    const int rows = query.dims[1];
    AssertInFastLLM(rows > 0 && rows <= 8 && queryStart >= 0 &&
        queryStart + rows <= packedKeys.dims[1], "Invalid Naive verification block.");
    if (heads == 16 && dim == 128 && topK == 2048) {
        FastllmNaiveDecodeScratch scratch;
        Data live;
        if (FastllmCudaNaiveBatchVerifyIndexer(query, weights,
                {{&packedKeys, &indices, 0, rows, queryStart}}, fp8Query, scratch, live)) return;
    }
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
    const int arch = FastllmCudaRuntimeArch();
    const bool blackwellBF16 = arch == 120 && key.dataType == BFLOAT16;
    if (heads == kvHeads * 16 && (queries < (blackwellBF16 ? 32 : (key.dataType == FP8_E4M3 ? 2048 : 1024)) ||
        count != 2048 || (arch != 90 && !blackwellBF16) ||
        key.dataType != value.dataType || !FastllmNaiveAttentionKVType(key.dataType) ||
        key.kvCacheFp8Indexer || value.kvCacheFp8Indexer)) return false;
    auto kernel = mma::Generic<2048>;
    size_t shared = sizeof(mma::Storage<2048>);
    int groupedHeads = 8;
    if (heads == kvHeads * 16) {
        // Keep the original normalized BF16 probabilities and 256-lane
        // softmax reduction. Online probability rounding changes model logits.
        kernel = mma::Generic<2048, 0, 2>;
        shared = sizeof(mma::Storage<2048, 2>);
        groupedHeads = 16;
    } else if (FastllmCudaRuntimeArch() == 90 &&
               BulkKernelSupported(mma::Hopper, sizeof(mma::HopperStorage), 90)) {
        kernel = mma::Hopper;
        shared = sizeof(mma::HopperStorage);
    }
    const size_t probabilityRowBytes = size_t(heads) * 2056 * sizeof(BF16);
    const int tileRows = std::min<size_t>(queries, (size_t(512) << 20) / probabilityRowBytes);
    const bool split = groupedHeads == 16 && count == 2048 && tileRows > 0;
    // Check the kernels actually launched. Split QK/PV need 32/39 KiB,
    // rather than the fused kernel's 96 KiB shared-memory allocation.
    if (split) {
        if (!BulkKernelSupported(mma::SplitQK<2048,0,2,mma::SplitBF16KV>, sizeof(mma::QKStorage<2>), 80) ||
            !BulkKernelSupported(mma::SplitNormalize<2048>, 0, 80) ||
            !BulkKernelSupported(mma::SplitPV<2048,0,2,mma::SplitBF16KV>, sizeof(mma::PVStorage<2>), 80)) return false;
    } else if (!BulkKernelSupported(kernel, shared, 80)) return false;
    Data stagedKey, stagedValue;
    const Data *readKey = &key, *readValue = &value;
    if (groupedHeads == 16 && key.dataType != BFLOAT16) {
        // Expand each cached value once, instead of repeating packed KV
        // conversion for every query's gathered tile. Storage stays quantized.
        if ((uint64_t)keys * (key.dims[2] + value.dims[2]) * sizeof(BF16) > (size_t(256) << 20)) return false;
        Output(stagedKey, BFLOAT16, key.dims);
        Output(stagedValue, BFLOAT16, value.dims);
        FastllmCudaNaiveReadCache({{&key, stagedKey.cudaData, 0, keys}, {&value, stagedValue.cudaData, 0, keys}});
        readKey = &stagedKey; readValue = &stagedValue;
    }
    Output(output, BFLOAT16, {1, queries, heads * 128});
    // Split the long sparse case to keep full probabilities out of CTA
    // shared memory. Preserve the exact normalized BF16 values and PV order.
    if (split) {
        Data probabilities;
        Output(probabilities, BFLOAT16, {tileRows, heads, 2056});
        auto *p = (BF16 *)probabilities.cudaData;
        const auto *ids = indices.dims.empty() ? nullptr : (const int *)indices.cudaData;
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        const mma::SplitBF16KV splitKey(*readKey), splitValue(*readValue);
        // KV is staged once above. Keep absolute query/index positions while
        // reusing a bounded probability tile on the same CUDA stream.
        for (int first = 0; first < queries; first += tileRows) {
            const int rows = std::min(tileRows, queries - first);
            mma::SplitQK<2048,0,2><<<dim3(rows,heads/16),256,sizeof(mma::QKStorage<2>)>>>(
                (const BF16 *)query.cudaData, splitKey,
                ids, p, heads, kvHeads, key.dims[2], keys, count, pastLength, causal, indexTile, first, queries);
            mma::SplitNormalize<2048><<<dim3(rows,heads),256>>>(p,bias,heads,count);
            mma::SplitPV<2048,0,2><<<dim3(rows,heads/16),256,sizeof(mma::PVStorage<2>)>>>(
                p, splitValue, ids,
                (BF16 *)output.cudaData, heads, kvHeads, keys, count,
                pastLength, causal, indexTile, first, queries);
        }
        CheckLaunch();
        return true;
    }
    kernel<<<dim3(queries, heads / groupedHeads), 256, shared>>>(
        (const BF16 *)query.cudaData, FastllmNaiveKV(*readKey), FastllmNaiveKV(*readValue),
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
        (heads != kvHeads * 8 && heads != kvHeads * 16) || query.dims.size() != 3 || query.dims[0] != 1 ||
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

#ifndef FASTLLM_ENABLE_BLACKWELL_DSA
bool FastllmCudaHasBlackwellIndexerAttention() { return false; }
bool FastllmCudaTryNaiveIndexerAttention(const FastllmNaivePrefillAttentionItem &,
        const fastllm::Data &, int, int, int, int, const fastllm::Data &) {
    return false;
}
#endif

static void NaiveBatchAttention(const std::vector<FastllmNaivePrefillAttentionItem> &items,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim, const fastllm::Data *indexerScratch) {
    using namespace fastllm;
#ifdef FASTLLM_NAIVE_DSA_MMA
    naive_dsa_mma::PrefillAttentionBatch batch{};
    Data scores;
    const int tileRows = heads > 0 ? std::max<size_t>(1, (size_t(1) << 30) / (size_t(heads) * 2048 * sizeof(BF16))) : 1;
    auto flush = [&] {
        if (!batch.count) return;
        Output(scores, BFLOAT16, {batch.rows, heads, 2048});
        if (batch.count == 1) {
            const auto &e = batch.entries[0];
            naive_dsa_mma::ScoresBF16<<<dim3(batch.rows, heads / 16, 32), 256>>>(e.query, e.key, e.indices,
                (BF16 *)scores.cudaData, heads, kvHeads, e.stride, e.keys, e.past);
        } else naive_dsa_mma::BatchScores<<<dim3(batch.rows, heads / 16, 32), 256>>>(
            batch, (BF16 *)scores.cudaData, heads, kvHeads);
        naive_dsa_mma::BatchSoftmax<<<batch.rows * heads, 256>>>((BF16 *)scores.cudaData,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData, heads);
        if (batch.count == 1) {
            const auto &e = batch.entries[0];
            naive_dsa_mma::ValuesBF16<<<dim3(batch.rows, heads / 16), 256>>>((const BF16 *)scores.cudaData,
                e.value, e.indices, e.output, heads, kvHeads, e.keys, e.past);
        } else naive_dsa_mma::BatchValues<<<dim3(batch.rows, heads / 16), 256>>>(
            batch, (const BF16 *)scores.cudaData, heads, kvHeads);
        CheckLaunch();
        batch.count = batch.rows = 0;
    };
#endif
    for (const auto &item : items) {
        if (indexerScratch && FastllmCudaTryNaiveIndexerAttention(item, sink, heads,
                kvHeads, dim, valueDim, *indexerScratch)) continue;
#ifdef FASTLLM_NAIVE_DSA_MMA
        const auto &q = *item.query, &k = *item.key, &v = *item.value, &ids = *item.indices;
        const int rows = q.dims[1], keys = k.dims[1];
        bool supported = rows >= 32 && (size_t)rows * heads >= 2048 && heads > 0 && kvHeads > 0 &&
            heads % kvHeads == 0 && (heads / kvHeads) % 16 == 0 && dim == 192 && valueDim == 128 &&
            q.dataType == BFLOAT16 && FastllmNaiveKVType(k.dataType) && FastllmNaiveKVType(v.dataType) &&
            q.dataDevice == DataDevice::CUDA && k.dataDevice == DataDevice::CUDA && v.dataDevice == DataDevice::CUDA &&
            q.dataDeviceIds == k.dataDeviceIds && q.dataDeviceIds == v.dataDeviceIds &&
            q.cudaData && k.cudaData && v.cudaData && (size_t)q.cudaData % 16 == 0 &&
            (size_t)k.cudaData % 16 == 0 && (size_t)v.cudaData % 16 == 0 && k.dims[2] % 8 == 0 &&
            ids.dataType == INT32 && ids.dims == std::vector<int>({rows, 2048}) &&
            ids.dataDevice == DataDevice::CUDA && ids.dataDeviceIds == q.dataDeviceIds && ids.cudaData &&
            FastllmCudaFlashInferDataTypeSupported(BFLOAT16);
        // Do not replace the faster fused/split dispatch or change its arithmetic.
        const bool canStage = k.dataType == BFLOAT16 ||
            (uint64_t)keys * (k.dims[2] + v.dims[2]) * sizeof(BF16) <= (size_t(256) << 20);
        if (heads == kvHeads * 16 && rows >= (k.dataType == FP8_E4M3 ? 2048 : 1024) &&
            FastllmCudaRuntimeArch() == 90 && k.dataType == v.dataType &&
            FastllmNaiveAttentionKVType(k.dataType) && !k.kvCacheFp8Indexer && !v.kvCacheFp8Indexer && canStage)
            supported = false;
        if (supported) {
            Output(*item.output, BFLOAT16, {1, rows, heads * valueDim});
            for (int first = 0; first < rows;) {
                if (batch.count == 8 || batch.rows == tileRows) flush();
                const int count = std::min(rows - first, tileRows - batch.rows);
                auto &e = batch.entries[batch.count++];
                e.query = (const BF16 *)q.cudaData + size_t(first) * heads * dim;
                e.key = FastllmNaiveKV(k); e.value = FastllmNaiveKV(v);
                e.indices = (const int *)ids.cudaData + size_t(first) * 2048;
                e.output = (BF16 *)item.output->cudaData + size_t(first) * heads * valueDim;
                e.rowOffset = batch.rows; e.keys = keys;
                e.past = item.past + first; e.stride = k.dims[2];
                batch.rows += count; first += count;
            }
            continue;
        }
        flush();
#endif
        FastllmCudaNaiveAttention(*item.query, *item.key, *item.value, *item.indices, sink,
            heads, kvHeads, dim, valueDim, item.past, 0, *item.output);
    }
#ifdef FASTLLM_NAIVE_DSA_MMA
    flush();
#endif
}

void FastllmCudaNaiveBatchAttention(const std::vector<FastllmNaivePrefillAttentionItem> &items,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim) {
    NaiveBatchAttention(items, sink, heads, kvHeads, dim, valueDim, nullptr);
}

void FastllmCudaNaiveBatchIndexerAttention(const std::vector<FastllmNaivePrefillAttentionItem> &items,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim, const fastllm::Data &scratch) {
    NaiveBatchAttention(items, sink, heads, kvHeads, dim, valueDim, &scratch);
}

void FastllmCudaNaiveAttention(const fastllm::Data &query, const fastllm::Data &key,
                              const fastllm::Data &value, const fastllm::Data &indices,
                              const fastllm::Data &sink, int heads, int kvHeads,
                              int dim, int valueDim, int pastLength, int window,
                              fastllm::Data &output, bool causal) {
    FastllmCudaNaiveAttention(query, key, value, indices, sink, heads, kvHeads,
        dim, valueDim, pastLength, window, output, causal, nullptr);
}

void FastllmCudaNaiveAttention(const fastllm::Data &query, const fastllm::Data &key,
                              const fastllm::Data &value, const fastllm::Data &indices,
                              const fastllm::Data &sink, int heads, int kvHeads,
                              int dim, int valueDim, int pastLength, int window,
                              fastllm::Data &output, bool causal, const fastllm::Data *prefillScratch) {
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
            cudaStreamPerThread, FastllmCudaRuntimeArch());
        AssertInFastLLM(status == cudaSuccess,
            std::string("Naive SWA FlashInfer: ") + cudaGetErrorString(status));
        CheckLaunch();
        return;
    }
#endif
    // Share each FP8 window across eight GQA heads. Keep the native BF16
    // score/probability rounding instead of changing the softmax algorithm.
    if (queries >= 32 && window == kSwaWindow && causal && !selected &&
        heads > 0 && kvHeads > 0 && heads == kvHeads * 8 &&
        dim == kSwaQkDim && valueDim == kSwaValueDim &&
        query.dataType == BFLOAT16 && key.dataType == FP8_E4M3 && value.dataType == FP8_E4M3 &&
        key.dims[2] % 8 == 0 && (size_t)query.cudaData % 16 == 0 &&
        (size_t)key.cudaData % 16 == 0 && (size_t)value.cudaData % 16 == 0) {
        // Neighboring causal windows overlap. Share their cache reads while
        // retaining each query's original MMA order and BF16 rounding.
        if (queries >= 1024 && pastLength >= 0 && count == 128 &&
            (int64_t)pastLength + queries <= keys) {
            auto grouped = naive_prefill_mma::SwaQueryTile<4>;
            size_t bytes = sizeof(naive_prefill_mma::SwaQueryStorage<4>);
            int queryTile = 4;
            if (FastllmCudaRuntimeArch() == 90 && (int64_t)queries * heads >= 65536) {
                auto registers = naive_prefill_mma::SwaRegisterTile<8>;
                const size_t registerBytes = sizeof(naive_prefill_mma::SwaRegisterStorage<8>);
                if (BulkKernelSupported(registers, registerBytes, 90)) {
                    grouped = registers; bytes = registerBytes; queryTile = 8;
                }
            }
            // Keep incomplete initial windows on the original kernel so the
            // softmax tree and BF16 rounding do not shift with left padding.
            const int initialQueries = std::min(queries, std::max(0, kSwaWindow - 1 - pastLength));
            auto initial = naive_prefill_mma::Generic<128, 128>;
            const size_t initialBytes = sizeof(naive_prefill_mma::Storage<128>);
            if (BulkKernelSupported(grouped, bytes, 80) &&
                (!initialQueries || BulkKernelSupported(initial, initialBytes, 80))) {
                FastllmNaiveKV readKey(key), readValue(value);
                int readKeys = keys, readPast = pastLength;
                const int first = std::max(0, pastLength - (kSwaWindow - 1));
                const int rows = pastLength + queries - first;
                const uint64_t stagingBytes = (uint64_t)rows * (key.dims[2] + value.dims[2]) * sizeof(BF16);
                // The caller can lend a dead activation or reserved scratch.
                // Its logical size bounds the write; no allocation or resize.
                if (prefillScratch && !FastllmCudaGraphIsCapturingFast() &&
                    prefillScratch->dataDevice == DataDevice::CUDA &&
                    prefillScratch->dataDeviceIds == query.dataDeviceIds &&
                    (prefillScratch->dataType == FLOAT32 || prefillScratch->dataType == BFLOAT16) &&
                    prefillScratch->cudaData &&
                    (uintptr_t)prefillScratch->cudaData % 16 == 0 &&
                    prefillScratch->cudaData != query.cudaData && prefillScratch->cudaData != key.cudaData &&
                    prefillScratch->cudaData != value.cudaData && prefillScratch->cudaData != output.cudaData &&
                    stagingBytes <= (size_t(256) << 20) && stagingBytes <= prefillScratch->GetBytes() &&
                    !key.kvCacheFp8Indexer && !value.kvCacheFp8Indexer) {
                    auto *stagedKey = (BF16 *)prefillScratch->cudaData;
                    auto *stagedValue = stagedKey + (size_t)rows * key.dims[2];
                    FastllmCudaNaiveReadCache({{&key, stagedKey, first, rows},
                                             {&value, stagedValue, first, rows}});
#ifdef FASTLLM_NAIVE_SWA_FLASHINFER
                    const int arch = FastllmCudaRuntimeArch();
                    if ((arch == 100 || arch == 103) && queries >= 8192 && value.dims[2] == kvHeads * valueDim &&
                        FastllmCudaFlashInferDataTypeSupported(BFLOAT16)) {
                        // The staged view starts at the oldest visible key. Its
                        // right-aligned causal window is identical to the cache's.
                        auto status = naive_swa_flashinfer::Run((const BF16 *)query.cudaData,
                            stagedKey, stagedValue, sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
                            (BF16 *)output.cudaData, queries, rows, heads, kvHeads, key.dims[2],
                            cudaStreamPerThread, arch);
                        AssertInFastLLM(status == cudaSuccess,
                            std::string("Naive staged SWA prefill: ") + cudaGetErrorString(status));
                        CheckLaunch();
                        return;
                    }
#endif
                    readKey = FastllmNaiveKV(stagedKey);
                    readValue = FastllmNaiveKV(stagedValue);
                    readKeys = rows; readPast = pastLength - first;
                }
                const auto *q = (const BF16 *)query.cudaData;
                auto *out = (BF16 *)output.cudaData;
                const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
                if (initialQueries)
                    initial<<<dim3(initialQueries, heads / 8), 256, initialBytes>>>(
                        q, readKey, readValue, nullptr, bias, out, heads, kvHeads,
                        key.dims[2], readKeys, count, readPast, true, 0);
                const int fullQueries = queries - initialQueries;
                grouped<<<dim3((fullQueries + queryTile - 1) / queryTile, heads / 8), 256, bytes>>>(
                    q + (size_t)initialQueries * heads * kSwaQkDim, readKey, readValue, bias,
                    out + (size_t)initialQueries * heads * kSwaValueDim,
                    fullQueries, heads, kvHeads, key.dims[2], readKeys, readPast + initialQueries);
                CheckLaunch();
                return;
            }
        }
        auto kernel = naive_prefill_mma::Generic<128, 128>;
        const size_t shared = sizeof(naive_prefill_mma::Storage<128>);
        if (BulkKernelSupported(kernel, shared, 80)) {
            Data stagedKey, stagedValue;
            const Data *readKey = &key, *readValue = &value;
            int readKeys = keys, readPast = pastLength;
            if (queries >= 256 &&
                !key.kvCacheFp8Indexer && !value.kvCacheFp8Indexer && pastLength >= 0 &&
                (int64_t)pastLength + queries <= keys) {
                // Only the trailing window and new rows can be attended to.
                // Expand FP8 once, preserving exact values and the existing
                // normalized BF16 probability/PV arithmetic in this kernel.
                const int first = std::max(0, pastLength - (kSwaWindow - 1));
                const int rows = pastLength + queries - first;
                if ((uint64_t)rows * (key.dims[2] + value.dims[2]) * sizeof(BF16) <= (size_t(256) << 20)) {
                    Output(stagedKey, BFLOAT16, {1, rows, key.dims[2]});
                    Output(stagedValue, BFLOAT16, {1, rows, value.dims[2]});
                    FastllmCudaNaiveReadCache({{&key, stagedKey.cudaData, first, rows},
                                             {&value, stagedValue.cudaData, first, rows}});
                    readKey = &stagedKey; readValue = &stagedValue;
                    readKeys = rows; readPast = pastLength - first;
                }
            }
            kernel<<<dim3(queries, heads / 8), 256, shared>>>(
                (const BF16 *)query.cudaData, FastllmNaiveKV(*readKey), FastllmNaiveKV(*readValue), nullptr,
                sink.dims.empty() ? nullptr : (const float *)sink.cudaData,
                (BF16 *)output.cudaData, heads, kvHeads, key.dims[2], readKeys, count, readPast, true, 0);
            CheckLaunch();
            return;
        }
    }
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
    const bool prefillShape = queries >= 32 && window == 0 && heads % kvHeads == 0 &&
        dim == naive_prefill::QkDim &&
        valueDim == naive_prefill::ValueDim && key.dims[2] % 8 == 0 &&
        (size_t)key.cudaData % 16 == 0 && (size_t)value.cudaData % 16 == 0;
    // Sixteen Q heads share the gathered KV tile. Retain the existing
    // fallback and all normalized BF16 score/probability rounding points.
    const bool tiledPrefill = prefillShape && heads / kvHeads == naive_prefill::Heads;
    if (prefillShape && (tiledPrefill || heads == kvHeads * 16) &&
        LaunchNaivePrefillAttention(query, key, value, indices, sink,
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
__global__ void DraftInput(const float *id, const BF16 *embedding, const BF16 *mask,
        const int *length, BF16 *hidden, float *positions, int channels, int rows) {
    id += blockIdx.z;
    length += blockIdx.z;
    hidden += (size_t)blockIdx.z * rows * channels;
    positions += blockIdx.z * rows;

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
        const BF16 *newValue, const int *length, int kc, int vc, int window, bool ringVerify) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y;
    const int past = window ? min(*length - 1, window - 1) : *length - 1;
    const int row = (ringVerify ? kSwaWindow : past) + query;
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

int DeviceSMCount() {
    static thread_local std::map<int, int> smCounts;
    const int device = FastllmCudaGetDevice();
    auto it = smCounts.find(device);
    if (it == smCounts.end()) {
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            cudaGetLastError();
            sms = 1;
        }
        it = smCounts.emplace(device, std::max(1, sms)).first;
    }
    return it->second;
}

int DecodeIndexerScoreBlocks(int rows, int tiles, int blocksPerSM) {
    // Spread work over several resident waves without launching the padded
    // history grid for every request. Small batches retain per-row parallelism.
    return std::min(tiles, std::min(256, std::max(1, DeviceSMCount() * blocksPerSM / rows)));
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

namespace {
__global__ void SplitRouterWeight(const float *weight, BF16 *terms, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float remaining = weight[i];
    for (int term = 0; term < 3; ++term) {
        const BF16 value = __float2bfloat16_rn(remaining);
        terms[term * count + i] = value;
        remaining -= __bfloat162float(value);
    }
}
__global__ void SumRouterTerms(const float *products, float *out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (products[i] + products[count + i]) + products[2 * count + i];
}
}

bool FastllmCudaNaivePrefillRouter(const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &scratch, fastllm::Data &output) {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
    using namespace fastllm;
    const int arch = FastllmCudaRuntimeArch();
    if ((arch != 100 && arch != 103) || FastllmCudaGraphIsCapturingFast()) return false;
    const int rows = RouterRows(input, weight, output);
    if (rows < 128 || scratch.isFake || scratch.cudaDataBorrowed || !scratch.cudaData ||
        scratch.dataDevice != CUDA || scratch.dataDeviceIds != input.dataDeviceIds) return false;
    const int k = weight.dims[1], n = weight.dims[0];
    if (int64_t(k) * n > INT_MAX / 3 || int64_t(rows) * n > INT_MAX / 3) return false;
    const size_t termsBytes = size_t(3) * k * n * sizeof(BF16);
    if (termsBytes + size_t(3) * rows * n * sizeof(float) > scratch.expansionBytes ||
        scratch.cudaData == input.cudaData || scratch.cudaData == weight.cudaData ||
        scratch.cudaData == output.cudaData) return false;
    // FP32 products and FP32 accumulation can use the existing device handle.
    auto handle = getFastllmCublasHandle();
    if (!handle) return false;
    auto *terms = (BF16 *)scratch.cudaData;
    auto *products = (float *)((char *)scratch.cudaData + termsBytes);
    // Transient decomposition keeps the original weight as the only resident copy.
    SplitRouterWeight<<<(k * n + 255) / 256, 256>>>((const float *)weight.cudaData, terms, k * n);
    const float alpha = 1.f, beta = 0.f;
    auto status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
        &alpha, terms, CUDA_R_16BF, k, int64_t(n) * k, input.cudaData, CUDA_R_16BF, k, 0,
        &beta, products, CUDA_R_32F, n, int64_t(rows) * n, 3, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) return false;
    AssertInFastLLM(status == CUBLAS_STATUS_SUCCESS, "Blackwell router GEMM failed.");
    auto dims = input.dims; dims.back() = n;
    Output(output, FLOAT32, dims);
    SumRouterTerms<<<(rows * n + 255) / 256, 256>>>(products, (float *)output.cudaData, rows * n);
    CheckLaunch();
    return true;
#else
    return false;
#endif
}

bool FastllmCudaNaiveRouterFP32(const fastllm::Data &input,
        const fastllm::Data &weight, fastllm::Data &output) {
    const int rows = RouterRows(input, weight, output, fastllm::FLOAT32);
    if (rows < 128 || FastllmCudaGraphIsCapturingFast()) return false;
    auto dims = input.dims; dims.back() = weight.dims[0];
    Output(output, fastllm::FLOAT32, dims);
    return fastllm_native_prefill::Fp32((const float *)input.cudaData,
        (const float *)weight.cudaData, (float *)output.cudaData, rows, weight.dims[0], 4096);
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
    const int arch = FastllmCudaRuntimeArch();
    const bool blackwell = arch == 100 || arch == 103;
    if (blackwell && rows >= 2 && ((uintptr_t(input.cudaData) | uintptr_t(weight.cudaData)) & 7) == 0) {
        auto kernel = rows <= 4 ? RouterPairs4096<1> : RouterPairs4096<2>;
        const int tile = rows <= 4 ? 1 : 2;
        kernel<<<dim3(experts, (rows + tile - 1) / tile), 128>>>(
            (const BF16 *)input.cudaData, (const float *)weight.cudaData,
            (float *)output.cudaData, rows, experts);
    }
    else if (rows == 1)
        RouterSigmoid4096<<<experts, 256>>>((const BF16 *)input.cudaData,
            (const float *)weight.cudaData, (float *)output.cudaData);
    else if (rows >= 3 && arch == 90 &&
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
    const int rows = hidden.Count(0) / channels;
    if (channels == 4096 && rows >= 128 &&
        ((uintptr_t)hidden.cudaData | (uintptr_t)branch.cudaData | (uintptr_t)output.cudaData) % 4 == 0 &&
        (uintptr_t)weight.cudaData % 8 == 0) {
        fastllm::cuda_norm::AddRMSNormPairs<BF16, 4096><<<rows, 128>>>(
            (BF16 *)hidden.cudaData, (const BF16 *)branch.cudaData,
            (const float *)weight.cudaData, (BF16 *)output.cudaData, eps);
        CheckLaunch();
        return;
    }
    auto kernel = channels == 4096
        ? fastllm::cuda_norm::AddRMSNorm<BF16, 4096, true, false>
        : fastllm::cuda_norm::AddRMSNorm<BF16, 0, true, false>;
    kernel<<<hidden.Count(0) / channels, 256>>>((BF16 *)hidden.cudaData, (const BF16 *)branch.cudaData,
        (const float *)weight.cudaData, (BF16 *)output.cudaData, channels, eps, false);
    CheckLaunch();
}

namespace {
struct KVReadBatch {
    struct Row { FastllmNaiveKV source; BF16 *target; int first, count, width, outWidth; } rows[32];
};
template<bool Vectorized = false>
__global__ void ReadCache(KVReadBatch batch) {
    const auto &r = batch.rows[blockIdx.y];
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * (Vectorized ? 8 : 1);
    if (i >= r.count) return;
    const int row = r.first + i / r.outWidth, col = i % r.outWidth;
    const int main = r.outWidth - (r.source.scaledIndexer ? FastllmNaiveIndexerBytes / 2 : 0);
    if constexpr (Vectorized) {
        *(uint4 *)(r.target + i) = (r.source.Row(row, r.width) + col).Load8();
    } else if (col < main) r.target[i] = r.source.Row(row, r.width)[col];
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
__global__ void PagedWrite(FastllmNaiveKVWriteRow pool, const int *pages, const BF16 *input, int start, int count, int width, int pageLen, const int *liveKeys, int capacity) {
    if (liveKeys) { start = *liveKeys - count; if (start < 0 || *liveKeys > capacity) return; }
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
        bool vectorized = true;
        for (int i = 0; i < count; ++i) {
            const auto &s = slices[first + i];
            const auto &data = *s.source;
            fastllm::AssertInFastLLM(FastllmNaiveAttentionKVType(data.dataType) &&
                data.dataDevice == fastllm::DataDevice::CUDA && data.dims.size() == 3 &&
                s.target && s.first >= 0 && s.rows > 0, "Invalid Naive KV read.");
            int outWidth = data.dims[2];
            if (data.kvCacheFp8Indexer) outWidth += FastllmNaiveIndexerBytes / 2 - FastllmNaiveIndexerBytes / data.unitSize;
            int elements = s.rows * outWidth;
            vectorized = vectorized && (data.dataType == fastllm::FP4_E2M1 ||
                (data.dataType == fastllm::FP8_E4M3 && (uintptr_t)data.cudaData % 8 == 0 &&
                 FastllmNaiveKV(data).offset % 8 == 0)) &&
                !data.kvCacheFp8Indexer && outWidth % 8 == 0 && (uintptr_t)s.target % 16 == 0;
            batch.rows[i] = {FastllmNaiveKV(data), (BF16 *)s.target, s.first, elements, data.dims[2], outWidth};
            maximum = std::max(maximum, elements);
        }
        if (vectorized)
            ReadCache<true><<<dim3((maximum / 8 + 255) / 256, count), 256>>>(batch);
        else
            ReadCache<false><<<dim3((maximum + 255) / 256, count), 256>>>(batch);
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
namespace {
__global__ void PagedWriteFP4(uint8_t *pool, const int *pages, const BF16 *input,
        int start, int count, int width, int pageLen, const int *liveKeys, int capacity) {
    if (liveKeys) { start = *liveKeys - count; if (start < 0 || *liveKeys > capacity) return; }
    const size_t block = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int blocksPerRow = width / 16;
    if (block >= (size_t)count * blocksPerRow) return;
    const int token = block / blocksPerRow, column = (block % blocksPerRow) * 16;
    const int row = start + token;
    const size_t physical = pages ? (size_t)pages[row / pageLen] * pageLen + row % pageLen : row;
    FastllmWriteFP4KVBlock(pool + physical * (blocksPerRow * 9),
        input + (size_t)token * width + column, column, width);
}
}
// Write eight contiguous FP8 values per lane and reuse the converted bytes
// when cold-window attention needs BF16 K/V in the original activation.
template<bool Expand>
__global__ void WriteFP8Rows(uint2 *cache, const int *pages, const uint4 *input,
        uint4 *expanded, int first, int rows, int width, int pageLen) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    int row = first + i / width;
    size_t physical = pages ? size_t(pages[row / pageLen]) * pageLen + row % pageLen : row;
    uint4 values = input[i];
    unsigned short pairs[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        pairs[j] = __nv_cvt_bfloat16raw2_to_fp8x2(reinterpret_cast<__nv_bfloat162 *>(&values)[j],
            __NV_SATFINITE, __NV_E4M3);
    cache[physical * width + i % width] = make_uint2(
        unsigned(pairs[0]) | (unsigned(pairs[1]) << 16),
        unsigned(pairs[2]) | (unsigned(pairs[3]) << 16));
    if constexpr (Expand)
        expanded[i] = make_uint4(FastllmNaiveKVRow::ConvertPair(pairs[0]), FastllmNaiveKVRow::ConvertPair(pairs[1]),
                                 FastllmNaiveKVRow::ConvertPair(pairs[2]), FastllmNaiveKVRow::ConvertPair(pairs[3]));
}
static void NaivePagedWrite(fastllm::Data &cache, const fastllm::Data &input, int start,
        const fastllm::Data *liveKeys, void *expanded) {
    using namespace fastllm;
    AssertInFastLLM(FastllmNaiveAttentionKVType(cache.dataType) && input.dataType == BFLOAT16 &&
        cache.dataDevice == DataDevice::CUDA && input.dataDevice == DataDevice::CUDA &&
        cache.dims.size() == 3 && input.dims.size() == 3 &&
        cache.kvCacheFp8Indexer == input.kvCacheFp8Indexer &&
        cache.dims[2] == input.dims[2] + (cache.kvCacheFp8Indexer
            ? FastllmNaiveIndexerBytes / cache.unitSize - FastllmNaiveIndexerBytes / 2 : 0) && start >= 0 &&
        start + input.dims[1] <= (cache.isPagedKVCache && !cache.isFake
            ? (int)cache.pageIndex.size() * cache.pageLen
            : (cache.expansionDims.empty() ? cache.dims[1] : cache.expansionDims[1])),
        "Invalid Naive KV append.");
    if (cache.dataType == FP4_E2M1) {
        AssertInFastLLM(!cache.kvCacheFp8Indexer && !input.kvCacheFp8Indexer && cache.dims[2] % 16 == 0,
                        "FP4 DSA KV stores block16-aligned K/V with a separate Indexer cache.");
        const size_t blocks = (size_t)input.dims[1] * (input.dims[2] / 16);
        if (blocks) PagedWriteFP4<<<(blocks + 127) / 128, 128>>>(
            (uint8_t *)(cache.isPagedKVCache ? cache.pagedKVCacheData->cudaData : cache.cudaData),
            cache.isPagedKVCache ? (const int *)cache.cudaData : nullptr,
            (const BF16 *)input.cudaData, start, input.dims[1], input.dims[2], cache.pageLen,
            liveKeys ? (const int *)liveKeys->cudaData : nullptr,
            cache.expansionDims.empty() ? cache.dims[1] : cache.expansionDims[1]);
        CheckLaunch();
        return;
    }
    if (cache.kvCacheFp8Indexer) {
        WriteIndexerCache<<<input.dims[1], 128>>>(FastllmNaiveKV(cache), (const BF16 *)input.cudaData,
            cache.dims[2], start, liveKeys ? (const int *)liveKeys->cudaData : nullptr);
        CheckLaunch();
        return;
    }
    FastllmNaiveKVWriteRow destination{cache.isPagedKVCache ? cache.pagedKVCacheData->cudaData : cache.cudaData,
                                     cache.dataType == FP8_E4M3};
    if (destination.fp8 && !liveKeys && input.dims[1] >= 128 && input.dims[2] % 8 == 0 &&
        (uintptr_t(destination.data) % 8) == 0 && (uintptr_t(input.cudaData) % 16) == 0) {
        auto kernel = expanded ? WriteFP8Rows<true> : WriteFP8Rows<false>;
        kernel<<<(input.Count(0) / 8 + 255) / 256, 256>>>((uint2 *)destination.data,
            cache.isPagedKVCache ? (const int *)cache.cudaData : nullptr, (const uint4 *)input.cudaData,
            (uint4 *)expanded, start, input.dims[1], input.dims[2] / 8, cache.pageLen);
        CheckLaunch();
        return;
    }
    PagedWrite<<<(input.Count(0) + 255) / 256, 256>>>(destination,
        cache.isPagedKVCache ? (const int *)cache.cudaData : nullptr, (const BF16 *)input.cudaData,
        start, input.dims[1], input.dims[2], cache.pageLen,
        liveKeys ? (const int *)liveKeys->cudaData : nullptr,
        cache.expansionDims.empty() ? cache.dims[1] : cache.expansionDims[1]);
    CheckLaunch();
    if (expanded) FastllmCudaNaiveReadCache({{&cache, expanded, start, input.dims[1]}});
}

void FastllmCudaNaivePagedWrite(fastllm::Data &cache, const fastllm::Data &input, int start,
        const fastllm::Data *liveKeys) {
    NaivePagedWrite(cache, input, start, liveKeys, nullptr);
}

void FastllmCudaNaiveWriteAndExpandFP8(fastllm::Data &cache, fastllm::Data &input, int start) {
    fastllm::AssertInFastLLM(cache.dataType == fastllm::FP8_E4M3 && !cache.kvCacheFp8Indexer,
                            "Expected an unscaled FP8 attention cache.");
    NaivePagedWrite(cache, input, start, nullptr, input.cudaData);

}

void FastllmCudaNaiveAppendDecodeCache(fastllm::Data &key, fastllm::Data &value,
        const fastllm::Data &newKey, const fastllm::Data &newValue,
        const fastllm::Data &liveKeys, int window, bool ring) {
    fastllm::AssertInFastLLM(!ring || window == kSwaWindow, "Invalid SWA ring window.");
    if (key.dataType == fastllm::FP4_E2M1) {
        fastllm::AssertInFastLLM(value.dataType == key.dataType && !window && !ring,
                               "FP4 KV is only supported for DSA, not SWA.");
        FastllmCudaNaivePagedWrite(key, newKey, 0, &liveKeys);
        FastllmCudaNaivePagedWrite(value, newValue, 0, &liveKeys);
        return;
    }
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

void FastllmCudaNaiveCommitSwaVerify(fastllm::Data &key, fastllm::Data &value,
        int past, int count) {
    using namespace fastllm;
    AssertInFastLLM(past >= 0 && count >= 0 && count <= 8 &&
        FastllmNaiveKVType(key.dataType) && key.dataType == value.dataType &&
        key.expansionDims.size() == 3 && value.expansionDims.size() == 3 &&
        std::min(key.expansionDims[1], value.expansionDims[1]) >= kSwaWindow + count,
        "Invalid staged SWA verification commit.");
    if (count) {
        const int kc = key.dims[2], vc = value.dims[2];
        const dim3 grid((kc + vc + 255) / 256, count);
        if (key.dataType == FP8_E4M3)
            CommitSwaVerify<<<grid, 256>>>((uint8_t *)key.cudaData, (uint8_t *)value.cudaData, kc, vc, past);
        else CommitSwaVerify<<<grid, 256>>>((BF16 *)key.cudaData, (BF16 *)value.cudaData, kc, vc, past);
        CheckLaunch();
    }
    const int kept = std::min(past + count, kSwaWindow - 1);
    key.Resize({1, kept, key.dims[2]});
    value.Resize({1, kept, value.dims[2]});
}

void FastllmCudaNaiveCommitBoundSwa(const void *const *bindings, const int *accepted,
        int rows, int layers, uint64_t slidingMask, int tableStride, int keyBytes, int valueBytes) {
    const int kc = keyBytes / sizeof(uint4), vc = valueBytes / sizeof(uint4);
    CommitBoundSwa<<<dim3((kc + vc + 63) / 64, rows, layers), 64, 0, cudaStreamPerThread>>>(
        bindings, (const int2 *)accepted, slidingMask, tableStride, kc, vc);
    CheckLaunch();
}

void FastllmCudaNaiveBatchCommitSwaVerify(const std::vector<FastllmNaiveCacheCommit> &caches) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    for (const auto &c : caches) {
        const bool valid = c.key && c.value && c.past >= 0 && c.count >= 0 && c.count <= 8 &&
            c.past <= INT_MAX - c.count && FastllmNaiveKVType(c.key->dataType) && c.key->dataType == c.value->dataType &&
            c.key->dims.size() == 3 && c.value->dims.size() == 3 && c.key->dims[0] == 1 && c.value->dims[0] == 1 &&
            c.key->dataDevice == DataDevice::CUDA && c.value->dataDevice == DataDevice::CUDA &&
            c.key->dataDeviceIds.size() == 1 && c.key->dataDeviceIds[0] == device &&
            c.value->dataDeviceIds.size() == 1 && c.value->dataDeviceIds[0] == device &&
            c.key->expansionDims.size() == 3 && c.value->expansionDims.size() == 3 &&
            std::min(c.key->expansionDims[1], c.value->expansionDims[1]) >= kSwaWindow + c.count;
        if (!valid) AssertInFastLLM(false, "Invalid batched SWA commit.");
    }
    for (size_t first = 0; first < caches.size(); first += kSwaCommitsPerLaunch) {
        SwaCommitBatch batch{}; int count = std::min<size_t>(kSwaCommitsPerLaunch, caches.size() - first), rows = 0, columns = 0;
        for (int i = 0; i < count; ++i) {
            const auto &c = caches[first + i];
            batch.key[i] = c.key->cudaData; batch.value[i] = c.value->cudaData;
            batch.kc[i] = c.key->dims[2]; batch.vc[i] = c.value->dims[2];
            batch.past[i] = c.past; batch.count[i] = c.count; batch.fp8[i] = c.key->dataType == FP8_E4M3;
            rows = std::max(rows, c.count); columns = std::max(columns, batch.kc[i] + batch.vc[i]);
        }
        if (rows) { BatchCommitSwaVerify<<<dim3((columns + 255) / 256, rows, count), 256>>>(batch); CheckLaunch(); }
        for (int i = 0; i < count; ++i) {
            const auto &c = caches[first + i]; const int kept = std::min(c.past + c.count, kSwaWindow - 1);
            FastllmNaiveResizeKVLength(*c.key, kept, c.key->dims[2]);
            FastllmNaiveResizeKVLength(*c.value, kept, c.value->dims[2]);
        }
    }
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

#ifdef FASTLLM_NAIVE_BATCH_INDEXER
static void SelectBatchDecodeIndexerTopK(const naive_indexer_batch::Batch &descriptors,
                                         int count, int capacity) {
    namespace batch = naive_indexer_batch;
    if (capacity > 16384) {
        const int blocks = std::min(naive_topk::kMaxBlocks, (capacity + 4095) / 4096);
        for (int shift = 24; shift >= 0; shift -= 8) {
            batch::SplitHistogram<><<<dim3(blocks, count), naive_topk::kThreads>>>(descriptors, shift);
            batch::SplitThreshold<><<<count, naive_topk::kThreads>>>(descriptors, blocks, shift);
        }
        batch::SplitCompact<><<<dim3(blocks, count), naive_topk::kThreads>>>(descriptors);
    } else batch::Select<><<<count, naive_topk::kThreads>>>(descriptors);
}
static void LaunchBatchDecodeIndexerTopK(const naive_indexer_batch::Batch &descriptors,
                                         int count, int capacity) {
    SelectBatchDecodeIndexerTopK(descriptors, count, capacity);
    naive_indexer_batch::Sort<><<<count, naive_topk::kThreads>>>(descriptors);
}
static bool ScalarBatchDecodeIndexer(const fastllm::Data &query, const fastllm::Data &weights,
        const fastllm::Data &liveKeys, const std::vector<FastllmNaiveIndexerRow> &rows, bool fp8Query) {
    using namespace fastllm;
    namespace batch = naive_indexer_batch;
    if (rows.empty() || query.dataType != BFLOAT16 || weights.dataType != BFLOAT16 ||
        liveKeys.dataType != INT32 || query.dims.empty() || query.dims.back() != 2048 ||
        weights.dims.empty() || weights.dims.back() != 16) return false;
    for (const auto &r : rows)
        if (!r.key || !r.scratch || !r.indices || !FastllmNaiveKVType(r.key->dataType) ||
            r.key->dims.size() != 3 || r.key->dims[2] < 128 || r.capacity <= 0 ||
            r.capacity > naive_topk::kMaxKeys || r.sequence < 0 || r.sequence >= liveKeys.Count(0) ||
            (size_t)(r.sequence + 1) * 2048 > query.Count(0) ||
            (size_t)(r.sequence + 1) * 16 > weights.Count(0)) return false;
    for (int first = 0; first < (int)rows.size(); first += batch::BatchLimit) {
        const int count = std::min(batch::BatchLimit, (int)rows.size() - first);
        batch::Batch descriptors{}; int capacity = 0;
        for (int i = 0; i < count; ++i) {
            const auto &r = rows[first + i]; auto &scratch = *r.scratch;
            Output(scratch.indexQuery, FLOAT32, {16, 128});
            Output(scratch.indexScores, INT32, {r.capacity});
            Output(scratch.topk, INT32, {(int)(sizeof(naive_topk::Workspace) / sizeof(int))});
            Output(*r.indices, INT32, {1, naive_topk::kTopK});
            descriptors.rows[i] = {FastllmNaiveKV(*r.key), (const int *)liveKeys.cudaData + r.sequence,
                (BF16 *)scratch.indexQuery.cudaData, nullptr, (unsigned *)scratch.indexScores.cudaData,
                (naive_topk::Workspace *)scratch.topk.cudaData, (int *)r.indices->cudaData,
                r.sequence, r.key->dims[2], r.cache};
            capacity = std::max(capacity, r.capacity);
        }
        RoundBatchIndexer<<<dim3(16, count), 128>>>((const BF16 *)query.cudaData, descriptors, fp8Query);
        BatchIndexScoresScalar<<<dim3((capacity + 7) / 8, count), 256>>>((const BF16 *)weights.cudaData, descriptors);
        LaunchBatchDecodeIndexerTopK(descriptors, count, capacity);
    }
    CheckLaunch();
    return true;
}

#endif

bool FastllmCudaNaiveBatchDecodeIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &liveKeys,
        const std::vector<FastllmNaiveIndexerRow> &rows, bool fp8Query,
        int contextRank, int contextRanks, fastllm::Data *exchange, int contextPageLen) {
    return FastllmCudaNaiveBatchDecodeIndexer(query, weights, liveKeys, rows, fp8Query,
        contextRank, contextRanks, exchange, contextPageLen, 0);
}

bool FastllmCudaNaiveBatchDecodeIndexer(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &liveKeys,
        const std::vector<FastllmNaiveIndexerRow> &rows, bool fp8Query,
        int contextRank, int contextRanks, fastllm::Data *exchange, int contextPageLen, int directWidth) {
#ifdef FASTLLM_NAIVE_BATCH_INDEXER
#ifdef FASTLLM_NAIVE_DSA_MMA
    if (!fp8Query || !FastllmCudaFlashInferDataTypeSupported(fastllm::BFLOAT16))
#endif
    {
        if (contextRanks != 1 || contextRank != 0 || contextPageLen) return false;
        return ScalarBatchDecodeIndexer(query, weights, liveKeys, rows, fp8Query);
    }
#endif
#if defined(FASTLLM_NAIVE_DSA_MMA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    using namespace fastllm;
    namespace batch = naive_indexer_batch;
    constexpr int heads = naive_dsa_mma::kHeads, dim = naive_dsa_mma::kIndexerDim;
    if (rows.empty() || query.dataType != BFLOAT16 ||
        weights.dataType != BFLOAT16 || liveKeys.dataType != INT32 ||
        query.dims.empty() || query.dims.back() != heads * dim ||
        weights.dims.empty() || weights.dims.back() != heads) return false;
    const bool cp = contextRanks > 1;
    if (contextPageLen && (!cp || contextPageLen < contextRanks || contextPageLen % contextRanks)) return false;
    const bool direct = directWidth != 0;
    const int width = direct ? directWidth : naive_topk::kTopK;
    if (direct && (!contextPageLen || rows.size() >= 16 || width < 1 || width > naive_topk::kTopK)) return false;
    if (contextRanks < 1 || contextRank < 0 || contextRank >= contextRanks ||
        (cp && (!exchange || exchange->dataType != INT32 ||
         exchange->Count(0) < (size_t)contextRanks * rows.size() * width * 2))) return false;
    std::vector<batch::Batch> packedBatches;
    std::vector<int> packedStarts;
    if (cp && !direct) packedBatches.reserve((rows.size() + batch::BatchLimit - 1) / batch::BatchLimit);
    auto pagedShardCapacity = [&](int capacity) {
        return ((capacity + contextPageLen - 1) / contextPageLen) * (contextPageLen / contextRanks);
    };
    const int arch = FastllmCudaRuntimeArch();
    const bool blackwell = !cp && (arch == 100 || arch == 103);
    const auto cacheType = rows.front().key ? rows.front().key->dataType : BFLOAT16;
    // Replicated decode scans live keys with a bounded score grid and split radix
    // passes. Only their tile-rounded int offsets need headroom. CP also has a
    // separate candidate exchange and retains its existing dispatch limit.
    const int maxCapacity = cp ? naive_topk::kMaxKeys : INT_MAX / 2;
    bool nativeScores = arch == 90 || (blackwell && rows.size() >= 8 &&
        (cacheType == BFLOAT16 || cacheType == FP8_E4M3) &&
        BulkKernelSupported(batch::HopperScores<false>, 0, 80));
    for (const auto &row : rows) {
        if (!row.key || !row.scratch || !row.indices || !FastllmNaiveKVType(row.key->dataType) ||
            row.key->dims.size() != 3 || row.key->dims[2] <
                (contextPageLen && row.key->kvCacheFp8Indexer ? FastllmNaiveIndexerBytes / row.key->unitSize : dim) ||
            (contextPageLen && row.key->pageLen != contextPageLen / contextRanks) ||
            row.capacity <= 0 || row.capacity > maxCapacity ||
            row.sequence < 0 || row.sequence >= liveKeys.Count(0) ||
            (size_t)(row.sequence + 1) * heads * dim > query.Count(0) ||
            (size_t)(row.sequence + 1) * heads > weights.Count(0)) return false;
        if (direct && width < std::min(naive_topk::kTopK, pagedShardCapacity(row.capacity))) return false;
        // cp.async copies 16 bytes; FP8 therefore needs twice as many elements.
        // Above the former batch limit the single-row path used BF16 MMA on
        // quantized keys. Keep its accumulation order, including TopK ties.
        nativeScores = nativeScores && row.capacity <= naive_topk::kMaxKeys && row.key->dataType == cacheType &&
            row.key->kvCacheFp8Indexer == rows.front().key->kvCacheFp8Indexer &&
            row.key->dims[2] % (cacheType == FP8_E4M3 ? 16 : 8) == 0 &&
            (uintptr_t)FastllmNaiveKV(*row.key).base % 16 == 0;
    }

    auto prepareRow = [&](int index) {
        const auto &row = rows[index];
        auto &scratch = *row.scratch;
        Output(scratch.indexQuery, BFLOAT16, {heads, dim});
        Output(scratch.indexScale, FLOAT32, {heads});
        Output(scratch.indexScores, INT32, {row.capacity});
        Output(scratch.topk, INT32, {(int)(sizeof(naive_topk::Workspace) / sizeof(int))});
        Output(*row.indices, INT32, {1, naive_topk::kTopK});
        batch::Row descriptor = {FastllmNaiveKV(*row.key),
            (const int *)liveKeys.cudaData + row.sequence, (BF16 *)scratch.indexQuery.cudaData,
            (float *)scratch.indexScale.cudaData, (unsigned *)scratch.indexScores.cudaData,
            (naive_topk::Workspace *)scratch.topk.cudaData, (int *)row.indices->cudaData,
            row.sequence, row.key->dims[2], row.cache};
        descriptor.cpRank = contextRank;
        descriptor.cpRanks = contextRanks;
        descriptor.contextPageLen = contextPageLen;
        return descriptor;
    };
    bool compactScores = false;
    std::vector<batch::Row> compactRows;
#if CUDART_VERSION >= 12040
    // Only combine descriptor chunks for the validated large Hopper CP4
    // paged-FP8 layout. Small batches and other layouts keep their old path.
    compactScores = nativeScores && cacheType == FP8_E4M3 && cp && contextRanks == 4 &&
        contextPageLen == 128 && rows.size() > batch::BatchLimit &&
        rows.size() <= naive_indexer_compact::CompactLimit;
    for (const auto &row : rows)
        compactScores = compactScores && row.key->dims[2] == rows.front().key->dims[2];
    compactScores = compactScores &&
        BulkKernelSupported(naive_indexer_compact::Scores<true>, 0, 90);
    if (compactScores) {
        namespace compact = naive_indexer_compact;
        compact::CompactBatch descriptors{};
        compactRows.reserve(rows.size());
        int capacity = 0;
        for (int i = 0; i < (int)rows.size(); ++i) {
            compactRows.push_back(prepareRow(i));
            const auto &r = compactRows.back();
            descriptors.rows[i] = {r.key.base, r.key.pages, r.cache, r.key.offset,
                r.live, r.query, r.scale, r.scores, r.sequence};
            if (!i) {
                descriptors.stride = r.stride;
                descriptors.pageLen = r.key.pageLen;
                descriptors.pageShift = r.key.pageShift;
                descriptors.cpRank = contextRank;
                descriptors.cpRanks = contextRanks;
                descriptors.contextPageLen = contextPageLen;
                descriptors.scaledIndexer = r.key.scaledIndexer;
            }
            capacity = std::max(capacity, rows[i].capacity);
        }
        capacity = pagedShardCapacity(capacity);
        compact::QuantizeCompact<<<dim3(heads, rows.size()), dim>>>(
            (const BF16 *)query.cudaData, descriptors);
        const int blocks = DecodeIndexerScoreBlocks(rows.size(),
            (capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys, 128);
        // Flat caches retain the original score kernel; only native32-key
        // pages reuse an address across the two16-key halves.
        if (descriptors.pageLen == 32 &&
            BulkKernelSupported(compact::Scores<true, true>, 0, 90)) {
            compact::Scores<true, true><<<dim3(blocks, rows.size()), 128>>>(
                (const BF16 *)weights.cudaData, descriptors);
        } else {
            compact::Scores<true><<<dim3(blocks, rows.size()), 128>>>(
                (const BF16 *)weights.cudaData, descriptors);
        }
    }

    // Score descriptors contain KV geometry; selection needs only four
    // pointers per row. Keep its batch size independent of score chunking.
    int localCapacity = 0;
    for (const auto &row : rows) localCapacity = std::max(localCapacity, row.capacity);
    if (contextPageLen) localCapacity = pagedShardCapacity(localCapacity);
    constexpr size_t localSortBytes = naive_indexer_sortpack::SortBytes<256, 5, true>();
    const bool compactTopK = nativeScores && contextRanks == 4 && contextPageLen && rows.size() >= 128 &&
        localCapacity > 16384 && localCapacity <= 65536 &&
        BulkKernelSupported(naive_indexer_wide_select::SelectWide<batch::LocalTopKBatch>, 0, 90) &&
        BulkKernelSupported(naive_indexer_sortpack::SortPack<256, 5, true, true, batch::LocalTopKBatch>,
                            localSortBytes, 90) &&
        BulkKernelSupported(batch::CopyPacked<batch::LocalTopKBatch>, 0, 90);
    if (compactTopK && !compactScores) compactRows.reserve(rows.size());
#endif

    std::vector<int> rowOrder;
    int pairedRows = 0;
    if (!cp && arch == 90 && nativeScores && cacheType == BFLOAT16 &&
            !rows.front().key->kvCacheFp8Indexer && rows.size() >= 16) {
        std::vector<int> singles;
        rowOrder.reserve(rows.size());
        for (int first = 0; first < (int)rows.size();) {
            int end = first + 1;
            while (end < (int)rows.size() && rows[end].key == rows[first].key &&
                    rows[end].cache == rows[first].cache && rows[end].capacity == rows[first].capacity) ++end;
            int next = first;
            for (; next + 1 < end; next += 2) {
                rowOrder.push_back(next); rowOrder.push_back(next + 1);
            }
            if (next < end) singles.push_back(next);
            first = end;
        }
        pairedRows = rowOrder.size();
        rowOrder.insert(rowOrder.end(), singles.begin(), singles.end());
    }
    for (int first = 0; first < (int)rows.size();) {
        const bool pairChunk = first < pairedRows;
        int count = std::min(pairChunk ? (batch::BatchLimit / 2) * 2 : batch::BatchLimit,
            (pairChunk ? pairedRows : (int)rows.size()) - first);
        if (blackwell) count = std::min(count, 128);
        int sharedQueries = 1;
#if CUDART_VERSION >= 12040
        // Reuse one K tile across adjacent verification queries. Keep row
        // order intact for CP exchange; ragged groups and other layouts fall
        // back to the single-query kernel. Paged CP has a common local origin;
        // contiguous CP can change Begin() with each device length.
        if (!compactScores && nativeScores && (cacheType == FP8_E4M3 || (blackwell && rows.front().key->kvCacheFp8Indexer)) && rows.size() >= 16 &&
                (!cp || contextPageLen) && rowOrder.empty()) {
            for (int queries : {4, 2}) {
                const int aligned = count / queries * queries;
                bool compatible = aligned > 0;
                for (int i = 0; compatible && i < aligned; ++i) {
                    const auto &a = rows[first + i], &b = rows[first + i / queries * queries];
                    compatible = a.key == b.key && a.cache == b.cache && a.capacity == b.capacity;
                }
                auto kernel = queries == 4 ? batch::HopperScoresSharedHead<4> : batch::HopperScoresSharedHead<2>;
                if (compatible && BulkKernelSupported(kernel, 0, 90)) {
                    sharedQueries = queries;
                    count = aligned;
                    break;
                }
            }
        }
#endif
        batch::Batch descriptors{};
        int capacity = 0;
        for (int i = 0; i < count; ++i) {
            const int index = rowOrder.empty() ? first + i : rowOrder[first + i];
            const auto &row = rows[index];
            descriptors.rows[i] = compactScores ? compactRows[first + i] : prepareRow(index);
#if CUDART_VERSION >= 12040
            if (compactTopK && !compactScores) compactRows.push_back(descriptors.rows[i]);
#endif
            capacity = std::max(capacity, row.capacity);
        }
        if (contextPageLen) capacity = pagedShardCapacity(capacity);
        else if (cp) capacity = ((capacity + contextRanks * 64 - 1) / (contextRanks * 64)) * 64;
        if (!compactScores) {
            batch::Quantize<<<dim3(heads, count), dim>>>((const BF16 *)query.cudaData, descriptors);
            const bool fp8Keys = cacheType == FP8_E4M3 || rows.front().key->kvCacheFp8Indexer;
            // Native FP8 tiles use less shared memory and admit more resident CTAs.
            auto pairedKernel = batch::HopperScores<false, false, 2>;
            bool paired = pairChunk && count >= 16;
            paired = paired && BulkKernelSupported(pairedKernel, 0, 90);
            // A long graph can have just one long live row. Give that row a
            // full grid budget; short rows retire their unused tiles.
            const int scoreRows = capacity > naive_topk::kMaxKeys ? 1 :
                (paired ? count / 2 : count / sharedQueries);
            const int scoreBlocks = DecodeIndexerScoreBlocks(scoreRows,
                (capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys, blackwell && nativeScores ? 7 :
                    nativeScores && fp8Keys && CUDART_VERSION >= 12040 ? 64 : 32);
            const dim3 grid(scoreBlocks, paired ? count / 2 : count / sharedQueries);
            bool blackwellScores = false;
#ifdef FASTLLM_ENABLE_NAIVE_INDEXER_SM100
            if (blackwell && nativeScores && (sharedQueries == 1 || count >= 32)) {
                naive_indexer_sm100::Row entries[naive_indexer_sm100::BatchLimit];
                for (int i = 0; i < count; ++i) {
                    const auto &r = descriptors.rows[i];
                    entries[i] = {r.key, r.live, r.query, r.scale, r.scores, r.sequence, r.stride, r.cache};
                }
                blackwellScores = FastllmCudaNaiveIndexerSm100(entries, count, capacity,
                    (const BF16 *)weights.cudaData, cudaStreamPerThread);
            }
#endif
            if (!blackwellScores) {
                if (paired) {
                    pairedKernel<<<grid, 128>>>((const BF16 *)weights.cudaData, descriptors);
                } else
#if CUDART_VERSION >= 12040
                // Check the compiled PTX image as well as the runtime GPU: an SM80
                // image JITed on Hopper must retain the original compatible path.
                if (nativeScores && (cacheType == FP8_E4M3 || (blackwell && fp8Keys)) && (rows.size() >= 16 || blackwell) &&
                    BulkKernelSupported(batch::HopperScoresSharedHead<>, 0, 90)) {
                    auto kernel = sharedQueries == 4 ? batch::HopperScoresSharedHead<4> :
                        sharedQueries == 2 ? batch::HopperScoresSharedHead<2> : batch::HopperScoresSharedHead<>;
                    kernel<<<grid, 128>>>((const BF16 *)weights.cudaData, descriptors);
                } else
#endif
                if (nativeScores) {
                    auto kernel = fp8Keys ? batch::HopperScores<true, (CUDART_VERSION >= 12040)> : batch::HopperScores<false>;
                    kernel<<<grid, 128>>>((const BF16 *)weights.cudaData, descriptors);
                } else {
                    batch::Scores<<<grid, naive_dsa_mma::kThreads>>>((const BF16 *)weights.cudaData, descriptors);
                }
            }
        }
#if CUDART_VERSION >= 12040
        if (compactTopK) {
            first += count;
            continue;
        }
#endif
        uint2 *pairs = direct ? (uint2 *)exchange->cudaData + ((size_t)contextRank * rows.size() + first) * width : nullptr;
        if (direct && capacity <= naive_topk::kTopK) {
            auto kernel = capacity <= 512 ? batch::SortContextPairs<4> :
                          capacity <= 1024 ? batch::SortContextPairs<8> : batch::SortContextPairs<16>;
            kernel<<<count, 128>>>(descriptors, pairs, width);
            first += count;
            continue;
        }
        // Sort independent 1K tiles in parallel, then keep the first K across
        // the tiles. The idle histogram storage holds all intermediate pairs.
        constexpr int tile = 1024, maxTiles = 8;
        static_assert(sizeof(naive_topk::Workspace::partials) >= tile * maxTiles * sizeof(uint2));
        if (direct && capacity <= tile * maxTiles) {
            const int tiles = (capacity + tile - 1) / tile;
            batch::SortContextPairs<8, true><<<dim3(tiles, count), 128>>>(descriptors, nullptr, tile);
            const int lanes = tiles <= 4 ? 4 : 8;
            auto kernel = tiles <= 4 ? batch::MergeContextPairs<4, true> : batch::MergeContextPairs<8, true>;
            kernel<<<dim3((tiles * tile * lanes + 255) / 256, count), 256>>>(descriptors, tile, pairs);
            first += count;
            continue;
        }
        // Fuse exact threshold selection and stable sorting. Refining to 2048
        // candidates pays off for short contexts; longer rows keep 4096 slots.
        // CP layouts, long contexts and unsupported images retain their paths.
        if (!cp && (blackwell || (nativeScores && rows.size() >= 16)) && capacity <= 16384) {
            namespace sortpack = naive_indexer_sortpack;
            auto kernel = capacity <= 4096 ? sortpack::SelectSort<2048> :
                blackwell ? sortpack::SelectSort<4096, true> : sortpack::SelectSort<4096>;
            const size_t bytes = capacity <= 4096 ? sortpack::SortBytes<256, 5, true, false, 2048>()
                                                  : sortpack::SortBytes<256, 5, true, false, 4096>();
            if (BulkKernelSupported(kernel, bytes, 90)) {
                kernel<<<count, 256, bytes>>>(descriptors);
                first += count;
                continue;
            }
        }
        // Bound the single-CTA row work; retain the existing paths for
        // short rows, small batches, other CP layouts and architectures.
        const bool wideSelect = ((cp && contextRanks == 4 && contextPageLen && arch == 90) ||
            (blackwell && nativeScores && sharedQueries > 1 && count >= 32)) &&
            rows.size() >= 16 && capacity > 16384 && capacity <= 65536 &&
            BulkKernelSupported(naive_indexer_wide_select::SelectWide<>, 0, 90);
        if (wideSelect) {
            naive_indexer_wide_select::SelectWide<<<count, 1024>>>(descriptors);
        } else {
            SelectBatchDecodeIndexerTopK(descriptors, count, capacity);
        }
        if (direct) {
            batch::SortContextPairs<0><<<count, naive_topk::kThreads>>>(descriptors, pairs, width);
            first += count;
            continue;
        }
        bool packed = false;
        if (cp && contextRanks == 4 && contextPageLen && nativeScores && rows.size() >= 16) {
            namespace sortpack = naive_indexer_sortpack;
            if (rows.size() >= 128) {
                constexpr size_t bytes = sortpack::SortBytes<256, 5, true>();
                if (BulkKernelSupported(sortpack::SortPack<256, 5, true>, bytes, 90)) {
                    sortpack::SortPack<256, 5, true><<<count, 256, bytes>>>(descriptors);
                    packed = true;
                }
            } else {
                constexpr size_t bytes = sortpack::SortBytes<512, 6, true>();
                if (BulkKernelSupported(sortpack::SortPack<512, 6, true>, bytes, 90)) {
                    sortpack::SortPack<512, 6, true><<<count, 512, bytes>>>(descriptors);
                    packed = true;
                }
            }
        }
        // Independent verification rows need only the sorted indices. Reuse
        // the stable cooperative sort without CP packing; unsupported images
        // and small batches keep the existing generic implementation.
        if (!cp && nativeScores && rows.size() >= 16) {
            namespace sortpack = naive_indexer_sortpack;
            constexpr size_t bytes = sortpack::SortBytes<256, 5, true, false>();
            if (BulkKernelSupported(sortpack::SortPack<256, 5, true, false>, bytes, 90)) {
                sortpack::SortPack<256, 5, true, false><<<count, 256, bytes>>>(descriptors);
                packed = true;
            }
        }
        if (!packed) {
            batch::Sort<><<<count, naive_topk::kThreads>>>(descriptors);
            if (cp) batch::PackLocal<<<count, naive_topk::kThreads>>>(descriptors);
        }
        if (cp) {
            packedBatches.push_back(descriptors);
            packedStarts.push_back(first);
        }
        first += count;
    }
#if CUDART_VERSION >= 12040
    if (compactTopK) {
        std::vector<batch::LocalTopKBatch> batches((rows.size() + batch::kLocalTopKLimit - 1) /
                                                  batch::kLocalTopKLimit);
        for (int first = 0; first < (int)rows.size(); first += batch::kLocalTopKLimit) {
            const int count = std::min(batch::kLocalTopKLimit, (int)rows.size() - first);
            auto &descriptors = batches[first / batch::kLocalTopKLimit];
            descriptors.rank = contextRank; descriptors.ranks = contextRanks; descriptors.page = contextPageLen;
            for (int i = 0; i < count; ++i) {
                const auto &r = compactRows[first + i];
                descriptors.rows[i] = {r.live, r.scores, r.workspace, r.indices};
            }
            naive_indexer_wide_select::SelectWide<<<count, 1024>>>(descriptors);
            naive_indexer_sortpack::SortPack<256, 5, true, true><<<count, 256, localSortBytes>>>(descriptors);
        }
        // Exchange can alias any row's scores. Complete every selection/sort
        // before copying even the first chunk, including ragged final chunks.
        for (int first = 0; first < (int)rows.size(); first += batch::kLocalTopKLimit) {
            const int count = std::min(batch::kLocalTopKLimit, (int)rows.size() - first);
            batch::CopyPacked<<<count, naive_topk::kThreads>>>(batches[first / batch::kLocalTopKLimit],
                (unsigned *)exchange->cudaData, rows.size(), first);
        }
        CheckLaunch();
        return true;
    }
#endif
    // Scores may alias the exchange arena. Finish packing every row before
    // any copy can overwrite another row's score slice (CUDA 12.0 batches).
    if (cp) for (int b = 0; b < (int)packedBatches.size(); ++b) {
        const int first = packedStarts[b];
        const int count = (b + 1 < (int)packedStarts.size() ? packedStarts[b + 1] : (int)rows.size()) - first;
        batch::CopyPacked<<<count, naive_topk::kThreads>>>(packedBatches[b],
            (unsigned *)exchange->cudaData, rows.size(), first);
    }
    CheckLaunch();
    return true;
#else
    return false;
#endif
}

void FastllmCudaNaivePrepareVerifyIndexerScratch(int count, int capacity, bool fp8Query,
        FastllmNaiveDecodeScratch &scratch) {
#ifdef FASTLLM_NAIVE_BATCH_INDEXER
    using namespace fastllm;
#ifdef FASTLLM_NAIVE_DSA_MMA
    const bool mma = fp8Query && FastllmCudaFlashInferDataTypeSupported(BFLOAT16);
#else
    const bool mma = false;
#endif
    Output(scratch.indexQuery, mma ? BFLOAT16 : FLOAT32, {count, 16, 128});
    Output(scratch.indexScale, FLOAT32, {count, 16});
    Output(scratch.indexScores, INT32, {count, capacity});
    Output(scratch.topk, INT32, {count, (int)(sizeof(naive_topk::Workspace) / sizeof(int))});
#endif
}

bool FastllmCudaNaiveBatchVerifyIndexer(const fastllm::Data &query,
        const fastllm::Data &weights,
        const std::vector<FastllmNaiveVerifyIndexerItem> &items, bool fp8Query,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &liveKeys,
        int contextRank, int contextRanks, fastllm::Data *exchange, int contextPageLen) {
#ifdef FASTLLM_NAIVE_BATCH_INDEXER
    using namespace fastllm;
    const int count = query.Count(0) / (16 * 128);
    if (items.empty() || query.dataType != BFLOAT16 || weights.dataType != BFLOAT16 ||
        query.dims.size() != 3 || query.dims.back() != 16 * 128 ||
        weights.Count(0) != (size_t)count * 16) return false;
    const bool sharded = contextPageLen > 0;
    if (sharded && (contextRanks <= 1 || contextRank < 0 || contextRank >= contextRanks ||
        contextPageLen % contextRanks || !exchange || !fp8Query)) return false;
    auto localLength = [&](int length) {
        if (!sharded) return length;
        const int page = contextPageLen / contextRanks;
        return length / contextPageLen * page +
            std::clamp(length % contextPageLen - contextRank * page, 0, page);
    };
    auto limit = [&](const FastllmNaiveVerifyIndexerItem &item) {
        return item.capacity > 0 ? item.capacity : sharded ? item.past + item.rows : item.key->dims[1];
    };
    int capacity = 0;
    const bool dynamic = items.front().capacity > 0;
    for (const auto &item : items) {
        if (!item.key || !item.indices || item.offset < 0 || item.rows < 1 ||
            item.offset + item.rows > count || item.past < 0 || item.key->dims.size() != 3 ||
            (dynamic != (item.capacity > 0)) ||
            (!dynamic && localLength(item.past + item.rows) > item.key->dims[1]) ||
            (dynamic && (item.key->expansionDims.size() != 3 ||
                localLength(item.capacity) > item.key->expansionDims[1])) ||
            !FastllmNaiveKVType(item.key->dataType) || item.key->dims[2] <
                (sharded && item.key->kvCacheFp8Indexer ? FastllmNaiveIndexerBytes / item.key->unitSize : 128) ||
            (!sharded && item.key->dims[1] <= 0) || limit(item) > naive_topk::kMaxKeys) return false;
        capacity = std::max(capacity, limit(item));
    }
    if (dynamic && (liveKeys.dataType != INT32 || liveKeys.Count(0) != count || !liveKeys.cudaData)) return false;
    if (!dynamic) Output(liveKeys, INT32, {count});
    for (size_t first = 0; !dynamic && first < items.size(); first += 32) {
        const int size = std::min<size_t>(32, items.size() - first);
        VerifyLengthBatch lengths{};
        for (int i = 0; i < size; ++i) {
            lengths.offset[i] = items[first + i].offset;
            lengths.rows[i] = items[first + i].rows;
            lengths.past[i] = items[first + i].past;
        }
        VerifyIndexerLengths<<<size, 32>>>(lengths, (int *)liveKeys.cudaData);
    }
    FastllmCudaNaivePrepareVerifyIndexerScratch(count, capacity, fp8Query, scratch);
    const int words = sizeof(naive_topk::Workspace) / sizeof(int);
    std::vector<FastllmNaiveDecodeScratch> views(count);
    std::vector<Data> indices(count);
    std::vector<FastllmNaiveIndexerRow> rows;
    rows.reserve(count);
    for (const auto &item : items) {
        Output(*item.indices, INT32, {item.rows, naive_topk::kTopK});
        for (int row = 0; row < item.rows; ++row) {
            const int offset = item.offset + row;
            auto &v = views[offset];
            auto view = [](Data &dst, Data &src, size_t offset, std::vector<int> shape) {
                dst.Resize(shape); dst.FakeFrom(src, offset * src.unitSize);
            };
            view(v.indexQuery, scratch.indexQuery, (size_t)offset * 2048, {16, 128});
            view(v.indexScale, scratch.indexScale, (size_t)offset * 16, {16});
            view(v.indexScores, scratch.indexScores, (size_t)offset * capacity, {limit(item)});
            view(v.topk, scratch.topk, (size_t)offset * words, {words});
            view(indices[offset], *item.indices, (size_t)row * naive_topk::kTopK, {1, naive_topk::kTopK});
            rows.push_back({item.key, &v, &indices[offset], offset, limit(item), item.cache});
        }
    }
    return FastllmCudaNaiveBatchDecodeIndexer(query, weights, liveKeys, rows, fp8Query,
        contextRank, contextRanks, exchange, contextPageLen);
#else
    return false;
#endif
}

void FastllmCudaNaiveMergeVerifyIndexer(
        const std::vector<FastllmNaiveVerifyIndexerItem> &items,
        FastllmNaiveDecodeScratch &scratch, const fastllm::Data &liveKeys,
        const fastllm::Data &exchange, int contextRanks, int contextPageLen) {
#ifdef FASTLLM_NAIVE_BATCH_INDEXER
    using namespace fastllm;
    const int count = liveKeys.Count(0), words = sizeof(naive_topk::Workspace) / sizeof(int);
    std::vector<FastllmNaiveDecodeScratch> views(count);
    std::vector<Data> indices(count);
    std::vector<FastllmNaiveIndexerRow> rows;
    for (const auto &item : items) for (int row = 0; row < item.rows; ++row) {
        const int offset = item.offset + row;
        auto &workspace = views[offset].topk;
        workspace.Resize({words}); workspace.FakeFrom(scratch.topk, size_t(offset) * words * sizeof(int));
        indices[offset].Resize({1, naive_topk::kTopK});
        indices[offset].FakeFrom(*item.indices, size_t(row) * naive_topk::kTopK * sizeof(int));
        rows.push_back({item.key, &views[offset], &indices[offset], offset, item.capacity, item.cache});
    }
    FastllmCudaNaiveMergeDecodeIndexer(liveKeys, rows, exchange, contextRanks, contextPageLen);
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
        const fastllm::Data &exchange, int contextRanks, int contextPageLen) {
    FastllmCudaNaiveMergeDecodeIndexer(liveKeys, rows, exchange, contextRanks, contextPageLen, 0);
}

void FastllmCudaNaiveMergeDecodeIndexer(const fastllm::Data &liveKeys,
        const std::vector<FastllmNaiveIndexerRow> &rows,
        const fastllm::Data &exchange, int contextRanks, int contextPageLen, int directWidth) {
#if defined(FASTLLM_NAIVE_DSA_MMA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    namespace batch = naive_indexer_batch;
    // Merge uses only live lengths and output pointers. Compact those fields
    // so arbitrary verification widths do not strand a small generic tail.
    if (contextPageLen && (contextRanks == 4 || contextRanks == 8) && rows.size() >= 64) {
        namespace merge = merge_chunk_screen;
        const size_t bytes = (contextRanks + contextRanks / 2) * naive_topk::kTopK * sizeof(unsigned long long);
        auto kernel = contextRanks == 8 ? merge::MergeChunked<1024, 4, merge::CompactBatch, 8>
                                        : merge::MergeChunked<1024, 4, merge::CompactBatch>;
        if (BulkKernelSupported(kernel, bytes, 90)) {
            for (int first = 0; first < (int)rows.size(); first += merge::CompactLimit) {
                const int count = std::min(merge::CompactLimit, (int)rows.size() - first);
                merge::CompactBatch descriptors{};
                descriptors.merged = (const unsigned *)exchange.cudaData;
                descriptors.total = rows.size(); descriptors.first = first; descriptors.page = contextPageLen;
                descriptors.ranks = contextRanks;
                for (int i = 0; i < count; ++i) {
                    const auto &source = rows[first + i];
                    descriptors.rows[i] = {(const int *)liveKeys.cudaData + source.sequence,
                                           (int *)source.indices->cudaData};
                }
                kernel<<<count, 1024, bytes>>>(descriptors);
            }
            CheckLaunch();
            return;
        }
    }
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
            r.contextPageLen = contextPageLen;
            r.merged = (const unsigned *)exchange.cudaData;
            r.mergedRows = rows.size(); r.mergedRow = first + i;
        }
        if (contextPageLen) {
            const int width = directWidth ? directWidth : naive_topk::kTopK;
            constexpr size_t mergeSharedBytes = 6 * naive_topk::kTopK * sizeof(unsigned long long);
            bool merged = false;
            if (directWidth && (contextRanks == 2 || contextRanks == 4 || contextRanks == 8)) {
                auto kernel = contextRanks == 2 ? batch::MergeContextPairs<2> :
                              contextRanks == 4 ? batch::MergeContextPairs<4> : batch::MergeContextPairs<8>;
                const int blocks = (contextRanks * width * contextRanks + 255) / 256;
                kernel<<<dim3(blocks, count), 256>>>(descriptors, width, nullptr);
                merged = true;
            } else if (contextRanks == 4 && count >= 64) {
                // Locate one interval per thread, then merge four adjacent
                // keys in the same score/position order. Larger descriptor
                // batches benefit from more independently active warps.
                auto kernel = count >= 128
                    ? merge_chunk_screen::MergeChunked<1024, 4>
                    : merge_chunk_screen::MergeChunked<512, 4>;
                if (BulkKernelSupported(kernel, mergeSharedBytes, 90)) {
                    kernel<<<count, count >= 128 ? 1024 : 512, mergeSharedBytes>>>(descriptors);
                    merged = true;
                } else if (BulkKernelSupported(merge_screen::MergeFour<512, true>, mergeSharedBytes, 90)) {
                    merge_screen::MergeFour<512, true><<<count, 512, mergeSharedBytes>>>(descriptors);
                    merged = true;
                }
            }
            if (!merged) {
                batch::ClearContextIndices<<<count, 256>>>(descriptors);
                batch::MergeContextCandidates<<<dim3((contextRanks * width + 255) / 256, count), 256>>>(descriptors, width);
            }
        } else {
            batch::Select<true><<<count, naive_topk::kThreads>>>(descriptors);
            batch::Sort<true><<<count, naive_topk::kThreads>>>(descriptors);
        }
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

// Overlap the next packed FP8 tile load with current BF16 MMA. The kernel
// retains the original QK/PV reduction order and global softmax tree.
bool PrefetchDSAAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads,
        fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    namespace bulk = pipeline_bulk;
    // Each CTA must remain within one KV group. The same pipeline works for
    // any number of groups, including unsharded DP attention heads.
    if (FastllmCudaRuntimeArch() != 90 || kvHeads <= 0 || heads % kvHeads ||
        heads / kvHeads < 16 || (heads / kvHeads) % 16 || rows.size() < 32)
        return false;
    for (const auto &row : rows)
        if (!row.fp8 || !row.indices || row.selectedCount != 2048 || row.keyStride % 16 ||
            (uintptr_t)row.key % 16 || (uintptr_t)row.value % 16)
            return false;
    // Share KV loads across two eight-warp groups without changing per-head
    // accumulation. Keep small batches and unsupported resources on the old path.
    if (rows.size() >= 64) {
        auto sharedHeads = bulk::Decode<2048, false, true, true, true, true, true,
            bulk::MaxBatchLimit, true>;
        constexpr size_t shared = sizeof(bulk::Storage<2048, true, 16>);
        if (BulkKernelSupported(sharedHeads, shared, 90)) {
            // End full descriptor batches on a resident-wave boundary. A
            // parameter-sized tail in every launch wastes SMs even when the
            // complete query batch has enough independent work to fill them.
            int batchLimit = bulk::MaxBatchLimit;
            if (rows.size() > bulk::MaxBatchLimit) {
                static thread_local std::map<int, int> batchLimits;
                const int device = FastllmCudaGetDevice();
                auto it = batchLimits.find(device);
                if (it == batchLimits.end()) {
                    int sms = 0, resident = 0;
                    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
                            sharedHeads, 512, shared) == cudaSuccess && sms > 0 && resident > 0) {
                        const int wave = sms * resident;
                        if (wave <= batchLimit) batchLimit = batchLimit / wave * wave;
                    } else {
                        cudaGetLastError(); // Retain the descriptor-sized fallback.
                    }
                    it = batchLimits.emplace(device, batchLimit).first;
                }
                batchLimit = it->second;
            }
            for (int first = 0; first < (int)rows.size();) {
                const int remaining = rows.size() - first;
                // Keep a fitting final batch intact, including a partial wave.
                const int count = remaining <= bulk::MaxBatchLimit ? remaining : batchLimit;
                bulk::BatchT<bulk::MaxBatchLimit> descriptors{};
                std::copy_n(rows.begin() + first, count, descriptors.rows);
                sharedHeads<<<dim3(heads / 16, count), 512, shared>>>(
                    (const BF16 *)query.cudaData + (size_t)first * heads * 192, descriptors, bias,
                    (BF16 *)output.cudaData + (size_t)first * heads * 128, heads, kvHeads);
                first += count;
            }
            return true;
        }
    }
    // A larger descriptor only helps here when it removes the second
    // launch. Keep the original specialization for small and larger batches.
    if (rows.size() > bulk::BatchLimit && rows.size() <= bulk::MaxBatchLimit) {
        auto expanded = bulk::Decode<2048, false, true, true, true, true, true,
            bulk::MaxBatchLimit>;
        constexpr size_t shared = sizeof(bulk::Storage<2048, true>);
        if (BulkKernelSupported(expanded, shared, 90)) {
            bulk::BatchT<bulk::MaxBatchLimit> descriptors{};
            std::copy(rows.begin(), rows.end(), descriptors.rows);
            expanded<<<dim3(heads / 8, rows.size()), 256, shared>>>(
                (const BF16 *)query.cudaData, descriptors, bias,
                (BF16 *)output.cudaData, heads, kvHeads);
            return true;
        }
    }
    const bool adjacentHeads = rows.size() >= 128;
    auto kernel = adjacentHeads
        ? bulk::Decode<2048, false, true, true, true, true, true>
        : bulk::Decode<2048, false, true, true, true, true, false>;
    constexpr size_t shared = sizeof(bulk::Storage<2048, true>);
    if (!BulkKernelSupported(kernel, shared, 90)) return false;
    for (int first = 0; first < (int)rows.size(); first += bulk::BatchLimit) {
        const int count = std::min(bulk::BatchLimit, (int)rows.size() - first);
        bulk::Batch batch{};
        std::copy_n(rows.begin() + first, count, batch.rows);
        const dim3 grid = adjacentHeads ? dim3(heads / 8, count) : dim3(count, heads / 8);
        kernel<<<grid, 256, shared>>>(
            (const BF16 *)query.cudaData + (size_t)first * heads * 192, batch, bias,
            (BF16 *)output.cudaData + (size_t)first * heads * 128, heads, kvHeads);
    }
    return true;
#else
    return false;
#endif
}

#if CUDART_VERSION >= 12010
bool CompactAttentionSupported(const std::vector<FastllmNaiveDSARow> &rows) {
    const auto &first = rows.front();
    for (const auto &row : rows) {
        if (!row.cache || !row.liveKeys || bool(row.indices) != bool(first.indices) ||
            row.capacity != first.capacity || row.keyStride != first.keyStride ||
            row.selectedCount != first.selectedCount || row.pageLen != first.pageLen ||
            row.poolTokens != first.poolTokens || row.fp8 != first.fp8 || row.verifyOffset != first.verifyOffset ||
            (!row.pageLen && (row.keyPages || row.valuePages))) return false;
    }
    return true;
}
#endif

bool VerifySwaQueryReuse(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads, fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    namespace swa = naive_prefill_mma;
    if (rows.empty() || rows.front().verifyOffset < 0) return false;
    const bool fp8 = rows.front().fp8;
    auto kernel = fp8 ? swa::SwaVerifyQueries<4, true> : swa::SwaVerifyQueries<4, false>;
    const size_t shared = std::max(sizeof(swa::SwaQueryStorage<4>),
        fp8 ? sizeof(naive_attention_bulk::Storage<128, true>) : sizeof(naive_attention_bulk::Storage<128>));
    if (!BulkKernelSupported(kernel, shared, 80)) return false;
    std::vector<swa::SwaVerifyGroup> groups;
    groups.reserve(rows.size());
    bool reused = false;
    for (int first = 0; first < (int)rows.size();) {
        const auto &base = rows[first];
        if (base.fp8 != fp8 || base.verifyOffset < 0) return false;
        int count = 1;
        while (count < 4 && first + count < (int)rows.size()) {
            const auto &next = rows[first + count];
            if (next.key != base.key || next.value != base.value || next.cache != base.cache ||
                next.fp8 != base.fp8 || next.verifyOffset != base.verifyOffset + count ||
                next.keyStride != base.keyStride ||
                (base.liveKeys ? next.liveKeys != base.liveKeys + count :
                                next.liveKeys || next.length != base.length + count)) break;
            ++count;
        }
        groups.push_back({base, first, count});
        reused = reused || count > 1;
        first += count;
    }
    if (!reused) return false;
    for (int first = 0; first < (int)groups.size(); first += swa::SwaVerifyLimit) {
        const int count = std::min(swa::SwaVerifyLimit, (int)groups.size() - first);
        swa::SwaVerifyBatch descriptors{};
        std::copy_n(groups.begin() + first, count, descriptors.groups);
        kernel<<<dim3(count, heads / 8), 256, shared>>>(
            (const BF16 *)query.cudaData, descriptors, bias, (BF16 *)output.cudaData, heads, kvHeads);
    }
    return true;
#else
    return false;
#endif
}

// FP4 uses the same BF16 MMA and global softmax as the FP8 fused path.
// Dequantize only the current shared-memory tile; retain FP4 in the KV pool.
template<bool Share16Heads>
bool PrefetchFP4DSAAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    namespace bulk = pipeline_bulk;
    constexpr int Heads = Share16Heads ? 16 : 8, Threads = Share16Heads ? 512 : 256;
    constexpr int Limit = bulk::FP4BatchLimit;
    constexpr size_t shared = sizeof(bulk::Storage<2048, true, Heads, true>);
    auto kernel = bulk::Decode<2048, false, false, true, true, true, true,
        Limit, Share16Heads, true>;
    if (!BulkKernelSupported(kernel, shared, 90)) return false;
    for (int first = 0; first < (int)rows.size(); first += Limit) {
        const int count = std::min(Limit, (int)rows.size() - first);
        bulk::BatchT<Limit, bulk::FP4Row> descriptors{};
        std::copy_n(rows.begin() + first, count, descriptors.rows);
        kernel<<<dim3(16 / Heads, count), Threads, shared>>>(
            (const BF16 *)query.cudaData + (size_t)first * 16 * 192, descriptors, bias,
            (BF16 *)output.cudaData + (size_t)first * 16 * 128, 16, 1);
    }
    return true;
#else
    return false;
#endif
}

// Small batches keep the scalar reference's rounding while sharing KV across
// complete GQA groups. Other layouts retain the existing attention paths.
bool SmallDSAAttention(const fastllm::Data &query, const fastllm::Data &sink,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads,
        int dim, int valueDim, FastllmNaiveDecodeScratch &scratch, fastllm::Data &output) {
    namespace small = naive_attention_small;
    using namespace fastllm;
    if (rows.empty() || rows.size() >= 32 || dim != 192 || valueDim != 128 ||
        heads < 16 || (heads / kvHeads) % 16 != 0) return false;
    for (const auto &row : rows)
        if (!row.indices || row.selectedCount != 2048 || row.fp4 || row.keyStride % 8 ||
            (uintptr_t)row.key % 16 || (uintptr_t)row.value % 16) return false;
    if (!BulkKernelSupported(small::Scores<16>, sizeof(small::SparseScoreStorage<16>), 90) ||
        !BulkKernelSupported(small::Values, sizeof(small::ValueStorage), 90)) return false;
    const int count = rows.size();
    FastllmNaiveDSABatch batch{};
    std::copy(rows.begin(), rows.end(), batch.rows);
    Output(output, BFLOAT16, {1, count, heads * 128});
    Output(scratch.attentionScores, FLOAT32, {count, heads, 2048});
    Output(scratch.attentionPartial, FLOAT32, {count, heads, 32, 128});
    auto *scores = (float *)scratch.attentionScores.cudaData;
    auto *parts = (float *)scratch.attentionPartial.cudaData;
    const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    small::Scores<16><<<dim3(heads / 16, 16, count), 512, sizeof(small::SparseScoreStorage<16>)>>>(
        (const BF16 *)query.cudaData, batch, scores, heads, kvHeads, 2048);
    naive_dsa_decode::Softmax<<<count * heads, 256>>>(scores, bias, batch, heads, 2048);
    small::Values<<<dim3(heads / 16, 32, count), 256, sizeof(small::ValueStorage)>>>(
        scores, batch, parts, heads, kvHeads);
    small::MergeValues<<<dim3(heads, count), 128>>>(parts, batch, (BF16 *)output.cudaData, heads);
    return true;
}

// Only pair queries whose KV bindings stay shared on every graph replay.
template<int Max>
bool DensePairAttention(const fastllm::Data &query, const float *sink,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads, fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    namespace bulk = naive_attention_bulk;
    if (rows.size() % 2 || FastllmCudaRuntimeArch() != 120) return false;
    // Leave small grids to the per-row kernel so they can fill the device.
    if (rows.size() / 2 * (heads / 8) < size_t(2 * DeviceSMCount())) return false;
    for (size_t i = 0; i < rows.size(); i += 2) {
        const auto &a = rows[i], &b = rows[i + 1];
        if (!a.fp8 || !b.fp8 || a.keyStride != b.keyStride || a.pageLen != b.pageLen ||
            a.cache != b.cache || a.keyPages != b.keyPages || a.valuePages != b.valuePages ||
            (!a.cache && (a.key != b.key || a.value != b.value))) return false;
    }
    constexpr size_t shared = sizeof(bulk::DensePairStorage<Max>);
    auto kernel = bulk::DensePair<Max>;
    if (!BulkKernelSupported(kernel, shared, 80)) return false;
    constexpr int limit = bulk::BatchLimit / 2 * 2;
    for (int first = 0; first < int(rows.size()); first += limit) {
        int n = std::min(limit, int(rows.size()) - first);
        bulk::Batch batch{};
        std::copy_n(rows.begin() + first, n, batch.rows);
        kernel<<<dim3(n / 2, heads / 8), 256, shared>>>(
            static_cast<const BF16 *>(query.cudaData) + size_t(first) * heads * 192, batch, sink,
            static_cast<BF16 *>(output.cudaData) + size_t(first) * heads * 128, heads, kvHeads);
    }
    return true;
#else
    return false;
#endif
}

template <int Max, bool Ring>
bool BulkAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads, fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    namespace bulk = naive_attention_bulk;
    const bool fp8 = rows.front().fp8;
    bool asyncFP8 = fp8;
    for (const auto &row : rows) {
        if (row.fp8 != fp8 || row.fp4) return false;
        asyncFP8 = asyncFP8 && row.keyStride % 16 == 0;
    }
    auto kernel = fp8 ? (asyncFP8 ? bulk::Decode<Max, Ring, true> : bulk::Decode<Max, Ring, true, false>)
                      : bulk::Decode<Max, Ring, false>;
    size_t shared = asyncFP8 ? sizeof(bulk::Storage<Max, true>) : sizeof(bulk::Storage<Max, false>);
    if (!BulkKernelSupported(kernel, shared, 80)) {
        // GPUs with a smaller shared-memory limit retain the vector-load path.
        if (!asyncFP8) return false;
        asyncFP8 = false;
        kernel = bulk::Decode<Max, Ring, true, false>;
        shared = sizeof(bulk::Storage<Max, false>);
        if (!BulkKernelSupported(kernel, shared, 80)) return false;
    }
#if CUDART_VERSION >= 12010
    if (rows.size() > bulk::BatchLimit && CompactAttentionSupported(rows)) {
        auto compact = fp8 ? (asyncFP8 ? bulk::Decode<Max, Ring, true, true, bulk::CompactBatch>
                                       : bulk::Decode<Max, Ring, true, false, bulk::CompactBatch>)
                            : bulk::Decode<Max, Ring, false, true, bulk::CompactBatch>;
        if (BulkKernelSupported(compact, shared, 80)) {
            for (int first = 0; first < (int)rows.size(); first += bulk::CompactLimit) {
                const int n = std::min(bulk::CompactLimit, int(rows.size()) - first);
                bulk::CompactBatch batch{};
                batch.geometry = rows[first];
                for (int i = 0; i < n; ++i) {
                    const auto &r = rows[first + i];
                    batch.rows[i] = {r.cache, r.liveKeys, r.indices};
                }
                compact<<<dim3(n, heads / 8), 256, shared>>>(
                    (const BF16 *)query.cudaData + (size_t)first * heads * 192, batch, bias,
                    (BF16 *)output.cudaData + (size_t)first * heads * 128, heads, kvHeads);
            }
            return true;
        }
    }
#endif
    for (int first = 0; first < (int)rows.size(); first += bulk::BatchLimit) {
        int n = std::min(bulk::BatchLimit, (int)rows.size() - first);
        bulk::Batch batch{};
        std::copy_n(rows.begin() + first, n, batch.rows);
        kernel<<<dim3(n, heads / 8), 256, shared>>>(
            (const BF16 *)query.cudaData + (size_t)first * heads * 192, batch, bias,
            (BF16 *)output.cudaData + (size_t)first * heads * 128, heads, kvHeads);
    }
    return true;
#else
    return false;
#endif
}

#if CUDART_VERSION >= 12010
template <int Max>
bool HopperBulkAttention(const fastllm::Data &query, const float *bias,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, fastllm::Data &output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    namespace bulk = naive_attention_bulk;
    if (std::any_of(rows.begin(), rows.end(), [](const auto &r) { return r.fp4; }) ||
        FastllmCudaRuntimeArch() != 90 ||
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
#else
    return false;
#endif
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

bool FastllmCudaNaiveOnlineDSADecode(const fastllm::Data &query, const fastllm::Data &sink,
        const std::vector<FastllmNaiveDSARow> &rows, int heads, int kvHeads, int dim, int valueDim,
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output) {
#ifdef FASTLLM_NAIVE_BLACKWELL_DECODE
    using namespace fastllm;
    namespace bw = blackwell_decode;
    const int arch = FastllmCudaRuntimeArch();
    if ((arch != 100 && arch != 103) || rows.empty() || kvHeads <= 0 ||
        heads != kvHeads * 16 || dim != 192 || valueDim != 128 || query.dataType != BFLOAT16)
        return false;
    for (const auto &row : rows) {
        if (!row.fp8 || row.fp4 || !row.indices || row.selectedCount != 2048 ||
            !row.key || !row.value || row.capacity < 0 || row.keyStride < kvHeads * dim ||
            row.keyStride % 16 || ((uintptr_t(row.key) | uintptr_t(row.value)) & 15))
            return false;
    }
    const auto *bias = sink.dims.empty() ? nullptr : static_cast<const float *>(sink.cudaData);
    const int total = static_cast<int>(rows.size());
    const bool split = total < 32;
    auto kernel = split ? bw::Decode<true> : bw::Decode<false>;
    if (!BulkKernelSupported(kernel, bw::SMEM_BYTES, 100)) return false;
    Output(output, BFLOAT16, {1, total, heads * 128});
    const int groups = total * kvHeads;
    const int splits = split ? (groups <= 16 ? 16 : groups <= 32 ? 8 : 4) : 1;
    float *parts = nullptr;
    float2 *stats = nullptr;
    if (split) {
        Output(scratch.attentionPartial, FLOAT32, {total, heads, splits, 128});
        Output(scratch.attentionScores, FLOAT32, {total, heads, splits, 2});
        parts = static_cast<float *>(scratch.attentionPartial.cudaData);
        stats = static_cast<float2 *>(scratch.attentionScores.cudaData);
    }
    for (int first = 0; first < total; first += naive_attention_bulk::BatchLimit) {
        const int count = std::min(total - first, int(naive_attention_bulk::BatchLimit));
        naive_attention_bulk::Batch batch{};
        std::copy_n(rows.begin() + first, count, batch.rows);
        auto *out = static_cast<BF16 *>(output.cudaData) + size_t(first) * heads * 128;
        kernel<<<dim3(count, kvHeads, splits), (bw::NCW + bw::NP) * 32, bw::SMEM_BYTES>>>(
            static_cast<const BF16 *>(query.cudaData) + size_t(first) * heads * 192, batch, bias,
            out, heads, parts, stats);
        // All split rows fit a single descriptor batch.
        if (split) bw::Merge<<<dim3(heads, count), 32>>>(parts, stats, bias, out, heads, splits);
    }
    CheckLaunch();
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
    if (SmallDSAAttention(query, sink, rows, heads, kvHeads, dim, valueDim, scratch, output)) {
        CheckLaunch(); return;
    }
    // Keep the established small-batch kernels. Eight heads share KV only
    // when the complete GQA group and cache vectors satisfy the MMA layout.
    bool bulk = rows.size() >= 32 && dim == 192 && valueDim == 128 &&
        (heads / kvHeads) % 8 == 0 && count > 0;
    for (const auto &row : rows)
        bulk = bulk && !row.fp4 && row.keyStride % 8 == 0 && (uintptr_t)row.key % 16 == 0 &&
            (uintptr_t)row.value % 16 == 0;
    if (bulk) {
        Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        bool launched = sparse &&
            PrefetchDSAAttention(query, bias, rows, heads, kvHeads, output);
#if CUDART_VERSION >= 12010
        if (!launched && dense && kvHeads == 1)
            launched = count <= 1024 ? HopperBulkAttention<1024>(query, bias, rows, heads, output)
                                     : HopperBulkAttention<2048>(query, bias, rows, heads, output);
#endif
        if (!launched && dense)
            launched = count <= 1024 ? DensePairAttention<1024>(query, bias, rows, heads, kvHeads, output)
                                     : DensePairAttention<2048>(query, bias, rows, heads, kvHeads, output);
        if (!launched)
            launched = count <= 1024 ? BulkAttention<1024, false>(query, bias, rows, heads, kvHeads, output)
                                     : BulkAttention<2048, false>(query, bias, rows, heads, kvHeads, output);
        if (launched) { CheckLaunch(); return; }
    }
    bool groupedFP4 = hopper && sparse && heads == 16 && kvHeads == 1 && rows.size() >= 32;
    bool fusedFP4 = groupedFP4;
    for (const auto &row : rows) {
        groupedFP4 = groupedFP4 && row.fp4 && row.keyStride % 16 == 0;
        fusedFP4 = fusedFP4 && row.fp4 && !row.fp8 && row.keyStride == 192 &&
            (uintptr_t)row.key % 16 == 0 && (uintptr_t)row.value % 16 == 0;
    }
    if (fusedFP4) {
        Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        if (PrefetchFP4DSAAttention<true>(query, bias, rows, output) ||
            PrefetchFP4DSAAttention<false>(query, bias, rows, output)) {
            CheckLaunch();
            return;
        }
    }
    // Decode a sparse FP4 KV tile once for all 16 GQA heads. Keep the
    // original scalar QK reduction, BF16 rounding, softmax and ordered PV.
    // Larger descriptors reduce launches without introducing device-side
    // descriptor storage. Lower-shared-memory kernels remain as fallbacks.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    constexpr int fp4BatchLimit = pipeline_bulk::BatchLimit;
    using FP4Batch = pipeline_bulk::BatchT<fp4BatchLimit>;
    constexpr size_t fp4ValueShared = sizeof(dsa::TiledValueStorage<8>);
    if (groupedFP4 && BulkKernelSupported(dsa::TiledSparseValues<8, FP4Batch>, fp4ValueShared, 90)) {
        Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
        Output(scratch.attentionScores, FLOAT32, {fp4BatchLimit, heads, 2048});
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        for (int first = 0; first < (int)rows.size(); first += fp4BatchLimit) {
            const int count = std::min(fp4BatchLimit, (int)rows.size() - first);
            FP4Batch descriptors{};
            std::copy_n(rows.begin() + first, count, descriptors.rows);
            const auto *q = (const BF16 *)query.cudaData + (size_t)first * heads * dim;
            auto *out = (BF16 *)output.cudaData + (size_t)first * heads * valueDim;
            auto *scores = (float *)scratch.attentionScores.cudaData;
            dsa::GroupedScores<16, 64><<<dim3(1, 32, count), 512>>>(
                q, descriptors, scores, heads, kvHeads, 2048);
            dsa::TiledSparseValues<8, FP4Batch><<<dim3(count, 2), 512, fp4ValueShared>>>(
                scores, descriptors, bias, out, heads);
        }
        CheckLaunch();
        return;
    }
#endif
    if (groupedFP4 && BulkKernelSupported(dsa::GroupedValues<2048, 4, 8, true>,
            sizeof(dsa::GroupStorage<2048, 4, 8>), 90)) {
        Output(output, BFLOAT16, {1, (int)rows.size(), heads * valueDim});
        Output(scratch.attentionScores, FLOAT32, {FastllmNaiveDSABatchLimit, heads, 2048});
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        for (int first = 0; first < (int)rows.size(); first += FastllmNaiveDSABatchLimit) {
            int batch = std::min(FastllmNaiveDSABatchLimit, (int)rows.size() - first);
            FastllmNaiveDSABatch descriptors{};
            std::copy_n(rows.begin() + first, batch, descriptors.rows);
            auto *q = (const BF16 *)query.cudaData + (size_t)first * heads * dim;
            auto *out = (BF16 *)output.cudaData + (size_t)first * heads * valueDim;
            auto *scores = (float *)scratch.attentionScores.cudaData;
            dsa::GroupedScores<16, 16><<<dim3(1, 128, batch), 512>>>(
                q, descriptors, scores, heads, kvHeads, 2048);
            dsa::GroupedValues<2048, 4, 8, true><<<dim3(4, 16, batch), 1024,
                sizeof(dsa::GroupStorage<2048, 4, 8>)>>>(scores, bias, descriptors, out,
                heads, kvHeads, 2048);
        }
        CheckLaunch();
        return;
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
    // Share the sparse V tile across complete groups of four query heads.
    // Each head keeps the existing softmax tree and ordered PV accumulation.
    // Other shapes and unsupported compiled images retain their original path.
    const bool groupedSparse = hopper && sparse && heads >= 16 && (heads / kvHeads) % 4 == 0 &&
        rows.size() >= 2 && rows.size() < 32 &&
        BulkKernelSupported(dsa::GroupedValues<2048, 4, 8, true>,
            sizeof(dsa::GroupStorage<2048, 4, 8>), 90);
    const bool groupedScores = groupedSparse && (heads / kvHeads) % 16 == 0 &&
        std::all_of(rows.begin(), rows.end(), [](const auto &r) {
            return r.keyStride % 8 == 0 && (uintptr_t)r.key % 16 == 0;
        });
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
            if (groupedScores)
                dsa::GroupedScores<16, 64><<<dim3(heads / 16, (count + 63) / 64, batch), 512>>>(
                    q, descriptors, scores, heads, kvHeads, count);
            else if (groupedSparse || (sharedHeads && sparse && batch >= 8))
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
            if (groupedSparse)
                dsa::GroupedValues<2048, 4, 8, true><<<dim3(heads / 4, 16, batch), 1024,
                    sizeof(dsa::GroupStorage<2048, 4, 8>)>>>(scores, bias, descriptors, out,
                    heads, kvHeads, count);
            else if (sharedHeads && dense && batch >= 3 && count <= 1024)
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

void FastllmCudaNaiveVerifyLengths(const fastllm::Data &positions, fastllm::Data &lengths) {
    fastllm::AssertInFastLLM(positions.dataType == fastllm::FLOAT32 &&
        positions.dataDevice == fastllm::DataDevice::CUDA, "Invalid verification positions.");
    const int count = positions.Count(0);
    Output(lengths, fastllm::INT32, {count});
    VerifyPositionLengths<<<(count + 255) / 256, 256>>>(
        (const float *)positions.cudaData, (int *)lengths.cudaData, count);
    CheckLaunch();
}

void FastllmCudaNaiveBatchSWADecode(const fastllm::Data &query,
        const fastllm::Data &sink, const std::vector<FastllmNaiveSWARow> &rows,
        const fastllm::Data &liveKeys, int heads, int kvHeads,
        fastllm::Data &output) {
    using namespace fastllm;
    AssertInFastLLM(!rows.empty() && heads > 0 && kvHeads > 0 && heads % kvHeads == 0 &&
        query.dataType == BFLOAT16 && query.dims == std::vector<int>({1, (int)rows.size(), heads * kSwaQkDim}) &&
        (liveKeys.dims.empty() || (liveKeys.dataType == INT32 && liveKeys.Count(0) == rows.size())),
        "Invalid batched SWA decode shape.");
    Output(output, BFLOAT16, {1, (int)rows.size(), heads * kSwaValueDim});
    const bool ringRows = std::all_of(rows.begin(), rows.end(),
        [](const FastllmNaiveSWARow &r) { return r.ring; });
    if (ringRows && rows.size() >= 32 && (heads / kvHeads) % 8 == 0) {
        std::vector<FastllmNaiveDSARow> descriptors(rows.size());
        bool aligned = true;
        for (int i = 0; i < (int)rows.size(); ++i) {
            auto &r = descriptors[i];
            r.key = rows[i].key; r.value = rows[i].value; r.cache = rows[i].cache; r.fp8 = rows[i].fp8;
            r.liveKeys = liveKeys.dims.empty() ? nullptr : (const int *)liveKeys.cudaData + i;
            r.length = rows[i].length; r.verifyOffset = rows[i].verifyOffset;
            r.capacity = kSwaWindow; r.keyStride = kvHeads * kSwaQkDim;
            aligned = aligned && r.key && r.value && (uintptr_t)r.key % 16 == 0 && (uintptr_t)r.value % 16 == 0;
        }
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        if (aligned && (VerifySwaQueryReuse(query, bias, descriptors, heads, kvHeads, output) ||
                        BulkAttention<128, true>(query, bias, descriptors, heads, kvHeads, output))) {
            CheckLaunch(); return;
        }
    }
    if (ringRows && rows.size() < 32 && rows.size() * heads >= 512 && (heads / kvHeads) % 8 == 0 &&
        std::all_of(rows.begin(), rows.end(), [](const auto &r) {
            return r.key && r.value && (uintptr_t)r.key % 16 == 0 && (uintptr_t)r.value % 16 == 0;
        }) && BulkKernelSupported(naive_attention_small::Swa, sizeof(naive_attention_small::SwaStorage), 90)) {
        FastllmNaiveDSABatch descriptors{};
        for (int i = 0; i < (int)rows.size(); ++i) {
            auto &r = descriptors.rows[i];
            r.key = rows[i].key; r.value = rows[i].value; r.cache = rows[i].cache; r.fp8 = rows[i].fp8;
            r.liveKeys = liveKeys.dims.empty() ? nullptr : (const int *)liveKeys.cudaData + i;
            r.length = rows[i].length; r.verifyOffset = rows[i].verifyOffset;
            r.capacity = kSwaWindow; r.keyStride = kvHeads * kSwaQkDim;
        }
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        naive_attention_small::Swa<<<dim3(rows.size(), heads / 8), 256,
                                    sizeof(naive_attention_small::SwaStorage)>>>(
            (const BF16 *)query.cudaData, descriptors, bias, (BF16 *)output.cudaData, heads, kvHeads);
        CheckLaunch(); return;
    }
    if (ringRows && rows.size() * heads >= 256 && (heads / kvHeads) % 4 == 0 &&
        std::all_of(rows.begin(), rows.end(), [](const auto &r) {
            return r.key && r.value && (uintptr_t)r.key % 16 == 0 && (uintptr_t)r.value % 16 == 0;
        }) && BulkKernelSupported(GroupedSwaDecode<4>, sizeof(GroupedSwaStorage<4>), 70)) {
        const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
        for (int first = 0; first < (int)rows.size(); first += kSwaBatchLimit) {
            const int count = std::min(kSwaBatchLimit, (int)rows.size() - first);
            BatchSwaKeys descriptors{};
            std::copy_n(rows.begin() + first, count, descriptors.rows);
            descriptors.length = liveKeys.dims.empty() ? nullptr : (const int *)liveKeys.cudaData + first;
            GroupedSwaDecode<4><<<dim3(heads / 4, count), 512, sizeof(GroupedSwaStorage<4>)>>>(
                (const BF16 *)query.cudaData + (size_t)first * heads * kSwaQkDim, bias,
                (BF16 *)output.cudaData + (size_t)first * heads * kSwaValueDim,
                heads, kvHeads, descriptors);
        }
        CheckLaunch(); return;
    }
    // Compute all output columns in one CTA instead of repeating QK and
    // softmax for four column tiles. Arithmetic within each output is unchanged.
    // Select by the work in one descriptor chunk, not request count or draft
    // length. Keep narrow tiles when too few query/head pairs can fill the GPU.
    const auto *bias = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    for (int first = 0; first < (int)rows.size(); first += kSwaBatchLimit) {
        const int count = std::min(kSwaBatchLimit, (int)rows.size() - first);
        const int64_t work = (int64_t)heads * count;
        const bool wideTile = WideSwaOutputTile(work);
        const int outputTile = wideTile ? 128 : kSwaOutputTile;
        const int threads = wideTile ? (work > 128 ? 512 : 1024)
            : (heads == 8 || heads == 16 ? 1024 : kSwaThreads);
        auto kernel = wideTile
            ? (threads == 512
                ? AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, 512, 128, true>
                : AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, 1024, 128, true>)
            : (threads == 1024
                ? AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, 1024, kSwaOutputTile, true>
                : AttentionShortDecode<kSwaWindow, BatchSwaKeys, false, kSwaThreads, kSwaOutputTile, true>);
        BatchSwaKeys keys{};
        std::copy_n(rows.begin() + first, count, keys.rows);
        keys.length = liveKeys.dims.empty() ? nullptr : (const int *)liveKeys.cudaData + first;
        kernel<<<dim3(heads, count, kSwaValueDim / outputTile), threads>>>(
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
        const fastllm::Data &liveKeys, int window, bool ringVerify) {
    const int kc = key.dims[2], vc = value.dims[2], rows = newKey.dims[1];
    if (key.kvCacheFp8Indexer) {
        WriteIndexerCache<<<rows, 128>>>(FastllmNaiveKV(key), (const BF16 *)newKey.cudaData,
            kc, 0, (const int *)liveKeys.cudaData, FastllmNaiveKV(value), (const BF16 *)newValue.cudaData, vc);
        CheckLaunch();
        return;
    }
    AppendVerifyKV<<<dim3((kc + vc + 255) / 256, rows), 256>>>({key.cudaData, key.dataType == fastllm::FP8_E4M3},
        {value.cudaData, value.dataType == fastllm::FP8_E4M3}, (const BF16 *)newKey.cudaData,
        (const BF16 *)newValue.cudaData, (const int *)liveKeys.cudaData, kc, vc, window, ringVerify);
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
__global__ void DraftSamplingLogits(const BF16 *base, const BF16 *bias,
        const uint32_t *mask, float *output, int vocab, int step = 0, int steps = 1) {
    base += ((size_t)blockIdx.y * steps + step) * vocab;
    bias += (size_t)blockIdx.y * vocab;
    mask += (size_t)blockIdx.y * ((vocab + 31) / 32);
    output += (size_t)blockIdx.y * vocab;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab)
        output[i] = (mask[i / 32] & (uint32_t(1) << (i % 32)))
            ? RoundBF16((float)base[i] + (float)bias[i]) : -INFINITY;
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
struct DraftRowIndices { int row[256]; };
template <typename Unit>
__global__ void DraftConcatRows(DraftHiddenInputs inputs, DraftRowIndices indices,
        Unit *output, int width, int layers, int first) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const int row = blockIdx.z, layer = blockIdx.y;
    const auto *source = reinterpret_cast<const Unit *>(inputs.ptr[layer]);
    output[(size_t(row) * layers + layer) * width + column] =
        source[size_t(first < 0 ? indices.row[row] : first + row) * width + column];
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
void FastllmCudaNaiveBatchDraftSamplingLogits(const fastllm::Data &base,
        const fastllm::Data &bias, const fastllm::Data &mask, fastllm::Data &output,
        int step, int steps) {
    using namespace fastllm;
    AssertInFastLLM(DraftDense(base, BFLOAT16) && DraftDense(bias, BFLOAT16) &&
        base.dims.size() == 3 && base.dims[0] == 1 && steps > 0 && step >= 0 && step < steps &&
        base.dims[1] % steps == 0 && bias.dims == std::vector<int>({1, base.dims[1] / steps, base.dims[2]}) &&
        DraftDense(mask, INT32) && mask.Count(0) == (size_t)(base.dims[1] / steps) * ((base.dims[2] + 31) / 32),
        "Invalid batched draft sampling logits.");
    const int rows = base.dims[1] / steps, vocab = base.dims[2];
    Output(output, FLOAT32, bias.dims);
    DraftSamplingLogits<<<dim3((vocab + 255) / 256, rows), 256>>>(
        (const BF16 *)base.cudaData, (const BF16 *)bias.cudaData,
        (const uint32_t *)mask.cudaData, (float *)output.cudaData, vocab, step, steps);
    CheckLaunch();
}

void FastllmCudaNaiveDraftSamplingLogits(const fastllm::Data &base,
        const fastllm::Data &bias, const fastllm::Data &mask, int step, fastllm::Data &output) {
    int vocab = base.dims.back();
    Output(output, fastllm::FLOAT32, {1, 1, vocab});
    DraftSamplingLogits<<<(vocab + 255) / 256, 256>>>(
        (const BF16 *)base.cudaData + (size_t)step * vocab, (const BF16 *)bias.cudaData,
        (const uint32_t *)mask.cudaData, (float *)output.cudaData, vocab);
    CheckLaunch();
}
static bool ConcatDraftRows(const std::vector<const fastllm::Data *> &inputs,
    int rows, const std::vector<int> &selected, fastllm::Data &output, const std::vector<int> *peerDevices) {
    if (inputs.empty() || inputs.size() > 32 || !inputs.front() || rows <= 0 || selected.size() > INT_MAX) return false;
    int width = inputs[0]->dims.empty() ? 0 : inputs[0]->dims.back();
    if (width <= 0 || width > INT_MAX / int(inputs.size())) return false;
    const int device = FastllmCudaGetDevice();
    if (peerDevices && peerDevices->size() != inputs.size()) return false;
    DraftHiddenInputs pointers{};
    bool vectorized = width % 8 == 0;
    for (int row : selected) if (row < 0 || row >= rows) return false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) return false;
        const auto &x = *inputs[i];
        if (x.dataType != fastllm::BFLOAT16 || x.dims.size() != 3 || x.dims[0] != 1 ||
            x.dims[1] < rows || x.dims[2] != width || x.dataDevice != fastllm::DataDevice::CUDA ||
            x.dataDeviceIds.size() != 1 || x.dataDeviceIds.front() != (peerDevices ? (*peerDevices)[i] : device) ||
            !x.cudaData || x.strides.size() != 3 ||
            x.strides[1] != width || x.strides[2] != 1) return false;
        vectorized &= reinterpret_cast<uintptr_t>(x.cudaData) % 16 == 0;
        pointers.ptr[i] = (const BF16 *)x.cudaData;
    }
    const int count = selected.empty() ? rows : static_cast<int>(selected.size());
    const int columnsOut = width * int(inputs.size());
    if (peerDevices) {
        if (!DraftDense(output, fastllm::BFLOAT16, device) || output.dims.size() != 3 ||
            output.dims[0] != 1 || output.dims[1] != count || output.dims[2] != columnsOut) return false;
    } else Output(output, fastllm::BFLOAT16, {1, count, columnsOut});
    vectorized &= reinterpret_cast<uintptr_t>(output.cudaData) % 16 == 0;
    const int columns = vectorized ? width / 8 : width;
    for (int first = 0; first < count;) {
        DraftRowIndices indices{};
        const int n = std::min(count - first, selected.empty() ? 65535 : int(sizeof(indices.row) / sizeof(int)));
        if (!selected.empty()) std::copy_n(selected.data() + first, n, indices.row);
        auto *dst = static_cast<BF16 *>(output.cudaData) + size_t(first) * width * inputs.size();
        dim3 grid((unsigned(columns) + 255) / 256, inputs.size(), n);
        if (vectorized)
            DraftConcatRows<<<grid, 256, 0, cudaStreamPerThread>>>(pointers, indices,
                reinterpret_cast<uint4 *>(dst), columns, int(inputs.size()), selected.empty() ? first : -1);
        else
            DraftConcatRows<<<grid, 256, 0, cudaStreamPerThread>>>(pointers, indices,
                dst, columns, int(inputs.size()), selected.empty() ? first : -1);
        CheckLaunch();
        first += n;
    }
    return true;
}

bool FastllmCudaNaiveDraftConcatRows(const std::vector<const fastllm::Data *> &inputs,
        int rows, const std::vector<int> &selected, fastllm::Data &output) {
    return ConcatDraftRows(inputs, rows, selected, output, nullptr);
}

bool FastllmCudaNaiveGatherDraftLogits(const std::vector<const fastllm::Data *> &shards,
        const std::vector<int> &devices, int rows, const std::vector<int> &selected, fastllm::Data &output) {
    return ConcatDraftRows(shards, rows, selected, output, &devices);
}

bool FastllmCudaNaiveDraftConcat(const std::vector<const fastllm::Data *> &inputs,
        int rows, fastllm::Data &output) {
    return FastllmCudaNaiveDraftConcatRows(inputs, rows, {}, output);
}

void FastllmCudaNaiveDraftInput(const fastllm::Data &id, const fastllm::Data &embedding,
        const fastllm::Data &mask, const fastllm::Data &liveKeys, int rows,
        fastllm::Data &hidden, fastllm::Data &positions) {
    using namespace fastllm;
    int channels = embedding.dims.back(), batch = id.Count(0);
    AssertInFastLLM(batch > 0 && liveKeys.Count(0) == batch, "Draft input length count mismatch.");
    Output(hidden, BFLOAT16, {1, rows * batch, channels});
    Output(positions, FLOAT32, {1, rows * batch});
    DraftInput<<<dim3((channels + 255) / 256, rows, batch), 256>>>((const float *)id.cudaData,
        (const BF16 *)embedding.cudaData, (const BF16 *)mask.cudaData,
        (const int *)liveKeys.cudaData, (BF16 *)hidden.cudaData,
        (float *)positions.cudaData, channels, rows);
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
        if (dim == 128 && window <= 1024 && (heads / kvHeads) % 4 == 0 &&
            FastllmCudaRuntimeArch() == 90) {
            DraftAttentionScores<<<dim3(heads / 4, rows, 16), 128>>>(q, k,
                (float *)scores.cudaData, heads, kvHeads, keys, past, window);
        } else {
            auto kernel = dim == 128 ? AttentionScoresPrefill<128, DraftLength, DraftLength, 64>
                        : dim <= 192 ? AttentionScoresPrefill<192, DraftLength, DraftLength>
                                     : AttentionScoresPrefill<256, DraftLength, DraftLength>;
            const int tiles = dim == 128 ? (scores.dims.back() + 63) / 64 : 1;
            kernel<<<dim3(heads, rows, tiles), 128>>>(q, k, nullptr, (float *)scores.cudaData,
                heads, kvHeads, dim, key.dims[2], keys, keys, past, window, false);
        }
        AttentionSoftmax<<<rows * heads, 256>>>((float *)scores.cudaData, nullptr, heads, keys);
        const int headTile = dim == 128 && FastllmCudaRuntimeArch() == 90 && (int64_t)heads * rows >= 256
            ? ((heads / kvHeads) % 8 == 0 ? 8 : (heads / kvHeads) % 4 == 0 ? 4 : 1) : 1;
        if (headTile == 8)
            DraftAttentionValues<32, 2, 8><<<dim3(heads / 8, rows, dim / 32), 256>>>(
                (float *)scores.cudaData, v, out, heads, kvHeads, dim, keys, past, window);
        else if (headTile == 4)
            DraftAttentionValues<32, 2, 4><<<dim3(heads / 4, rows, dim / 32), 128>>>(
                (float *)scores.cudaData, v, out, heads, kvHeads, dim, keys, past, window);
        else if (dim == 128 && FastllmCudaRuntimeArch() == 90)
            // Two BF16 padding elements spread column reads over all banks.
            // A full warp computes outputs; the remaining warps stage V.
            DraftAttentionValues<32, 2><<<dim3(heads, rows, dim / 32), 128>>>(
                (float *)scores.cudaData, v, out, heads, kvHeads, dim, keys, past, window);
        else if (dim == 128)
            DraftAttentionValues<><<<dim3(heads, rows, dim / 16), 256>>>(
                (float *)scores.cudaData, v, out, heads, kvHeads, dim, keys, past, window);
        else
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
#if defined(FASTLLM_NAIVE_DSA_MMA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    namespace batch = naive_indexer_batch;
    if (FastllmCudaRuntimeArch() == 90 && rows >= 4 && rows <= batch::BatchLimit &&
        capacity <= naive_topk::kMaxKeys) {
        // Reuse the batched decode selector on independent causal rows. The
        // score arithmetic and stable tie order stay unchanged; other devices
        // and shapes retain the cooperative per-row implementation below.
        batch::Batch descriptors{};
        for (int row = 0; row < rows; ++row) {
            auto &r = descriptors.rows[row];
            r.live = (const int *)liveKeys.cudaData + row;
            r.scores = (unsigned *)scratch.indexScores.cudaData + (size_t)row * capacity;
            r.workspace = workspaces + row;
            r.indices = (int *)indices.cudaData + row * naive_topk::kTopK;
        }
        LaunchBatchDecodeIndexerTopK(descriptors, rows, capacity);
        CheckLaunch();
        return;
    }
#endif
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
        FastllmNaiveDecodeScratch &scratch, fastllm::Data &output, bool ringVerify) {
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
            key.dims[2], keys, window != 0, ringVerify);
        CheckLaunch();
        return;
    }
    if (rows > 1 && !window && dim == 192 && valueDim == 128 &&
        heads % kvHeads == 0 && (capacity <= 2048 || !indices.dims.empty())) {
        // Each verification row shares the cache but has its own causal length
        // and sparse selection. Device length pointers also stay valid on replay.
        FastllmNaiveKV k(key), v(value);
        std::vector<FastllmNaiveDSARow> batch(rows);
        for (int row = 0; row < rows; ++row) {
            auto &r = batch[row];
            r.key = k.base;
            r.value = v.base;
            r.keyPages = k.pages;
            r.valuePages = v.pages;
            r.pageLen = k.pageLen;
            // TMA descriptors cover physical pool addresses, not the logical
            // request length: a restored prefix can reference any pool page.
            r.poolTokens = key.isPagedKVCache ? key.pagedKVCacheData->maxPages * key.pageLen : 0;
            r.fp8 = k.fp8;
            r.capacity = capacity;
            r.keyStride = key.dims[2];
            r.liveKeys = (const int *)liveKeys.cudaData + row;
            if (!indices.dims.empty()) {
                r.indices = (const int *)indices.cudaData + (size_t)row * 2048;
                r.selectedCount = 2048;
            }
        }
        FastllmCudaNaiveBatchDSADecode(query, sink, batch, heads, kvHeads,
                                      dim, valueDim, scratch, output);
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
        (data.dataType == fastllm::BFLOAT16 || data.dataType == fastllm::FLOAT32) &&
        data.dataDeviceIds == std::vector<int>{group->devices[rank]},
        "Invalid draft TP reduction.");
    const auto status = ncclAllReduce(data.cudaData, data.cudaData, data.Count(0),
        data.dataType == fastllm::FLOAT32 ? ncclFloat : ncclBfloat16,
        ncclSum, group->comms[rank], cudaStreamPerThread);
    if (status != ncclSuccess)
        fastllm::ErrorInFastLLM("Draft TP reduction failed on rank " + std::to_string(rank) +
            ": " + ncclGetErrorString(status));
}

#include "naive-n05-indexer-context.cuh"
