#include "fastllm-cuda.cuh"
#include "libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cuh"

#include "cute/tensor.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

namespace {

static constexpr int W4A8_PACKED_WEIGHT_IDX = 64;
static constexpr uint64_t W4A8_CACHE_MAGIC = 0x5734413843414348ULL; // "W4A8CACH"

struct W4A8WeightCacheMeta {
    uint64_t magic = W4A8_CACHE_MAGIC;
    fastllm::DataType sourceType = fastllm::DataType::FLOAT32;
    int inChannels = 0;
    int outChannels = 0;
    int groupCnt = 0;
    int group = 0;
    size_t packedWeightBytes = 0;
    const void *sourceCudaData = nullptr;
};

struct W4A8ActivationScratch {
    cutlass::float_e4m3_t *fp8 = nullptr;
    float *tokenScales = nullptr;
    int tokens = 0;
    int hidden = 0;
    size_t fp8Bytes = 0;
    size_t scaleBytes = 0;
};

static std::unordered_map<const fastllm::Data*, W4A8WeightCacheMeta> g_w4a8WeightCacheMetas;

static bool FastllmCudaW4A8PrepareCacheEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_CACHE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8PrepareActivationEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_ACTIVATION");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8BasicShapeSupported(int m, int k) {
    return m > 0 && k > 0 && (m % 128) == 0 && (k % 128) == 0;
}

static bool FastllmCudaW4A8CanQuantizeActivation(const fastllm::Data &input,
                                                 int n,
                                                 int m) {
    return n > 0 &&
           m > 0 &&
           input.cudaData != nullptr &&
           (input.dataType == fastllm::DataType::FLOAT16 ||
            input.dataType == fastllm::DataType::BFLOAT16);
}

static bool FastllmCudaW4A8CanUseInt4GroupSource(const fastllm::Data &weight, int m, int k) {
    if (weight.dataType != fastllm::DataType::INT4_GROUP ||
        weight.cudaData == nullptr ||
        weight.dims.size() != 2 ||
        weight.dims[0] != k ||
        weight.dims[1] != m ||
        weight.groupCnt != 128 ||
        weight.group != m / 128 ||
        weight.scales.size() != (size_t)k * weight.group ||
        weight.mins.size() != (size_t)k * weight.group ||
        !FastllmCudaW4A8BasicShapeSupported(m, k)) {
        return false;
    }

    // vLLM cutlass_w4a8 consumes signed INT4 weights with scale-only group
    // quantization. FastLLM INT4_GROUP can be repacked without changing
    // quantization math only when min + uint4 * scale == (uint4 - 8) * scale.
    for (size_t i = 0; i < weight.mins.size(); i++) {
        float expectedMin = -8.0f * weight.scales[i];
        float tol = 1e-6f * (std::fabs(weight.scales[i]) > 1.0f ? std::fabs(weight.scales[i]) : 1.0f);
        if (std::fabs(weight.mins[i] - expectedMin) > tol) {
            return false;
        }
    }
    return true;
}

__global__ void FastllmCudaW4A8PackInt4GroupToVllmBKernel(const uint8_t *src,
                                                          uint32_t *dst,
                                                          int inChannels,
                                                          int outChannels) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)(inChannels / 8) * outChannels;
    if (idx >= total) {
        return;
    }

    int packRow = (int)(idx / outChannels);
    int out = (int)(idx - (size_t)packRow * outChannels);
    int inBase = packRow * 8;
    const uint8_t *row = src + (size_t)out * (inChannels / 2);

    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int in = inBase + i;
        uint8_t byte = row[in / 2];
        uint32_t q = (in & 1) ? (byte & 0xF) : (byte >> 4);
        uint32_t signedQ = (q - 8) & 0xF;
        packed |= signedQ << (i * 4);
    }
    dst[idx] = packed;
}

__device__ inline float FastllmCudaW4A8ToFloat(half v) {
    return __half2float(v);
}

__device__ inline float FastllmCudaW4A8ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__global__ void FastllmCudaW4A8QuantizeActivationPerTokenKernel(
    const T *input,
    cutlass::float_e4m3_t *fp8,
    float *tokenScales,
    int tokens,
    int hidden) {
    int row = blockIdx.x;
    if (row >= tokens) {
        return;
    }

    __shared__ float reduce[256];
    float localMax = 0.0f;
    size_t rowOffset = (size_t)row * hidden;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = FastllmCudaW4A8ToFloat(input[rowOffset + col]);
        localMax = fmaxf(localMax, fabsf(v));
    }
    reduce[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float scale = fmaxf(reduce[0], 1.0e-10f) * (1.0f / 448.0f);
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = FastllmCudaW4A8ToFloat(input[rowOffset + col]) / scale;
        v = fminf(448.0f, fmaxf(-448.0f, v));
        fp8[rowOffset + col] = cutlass::float_e4m3_t(v);
    }
    if (threadIdx.x == 0) {
        tokenScales[row] = scale;
    }
}

static bool FastllmCudaW4A8EncodeAndReorderInt4B(uint8_t *rawPackedWeight,
                                                 uint8_t *cutlassPackedWeight,
                                                 int inChannels,
                                                 int outChannels) {
    using MmaType = cutlass::float_e4m3_t;
    using QuantType = cutlass::int4b_t;

    auto rawPtr = reinterpret_cast<QuantType const*>(rawPackedWeight);
    auto packedPtr = reinterpret_cast<QuantType*>(cutlassPackedWeight);
    size_t numInt4Elems = (size_t)inChannels * outChannels;

    if (!vllm::cutlass_w4a8_utils::unified_encode_int4b(rawPtr, packedPtr, numInt4Elems)) {
        return false;
    }

    auto shapeB = cute::make_shape(outChannels, inChannels, 1);
    auto layoutB = cute::make_layout(shapeB, cute::LayoutRight{});
    auto layoutAtomQuant = cutlass::compute_memory_reordering_atom<MmaType>();
    auto layoutBReordered = cute::tile_to_shape(layoutAtomQuant, shapeB);
    cutlass::reorder_tensor(packedPtr, layoutB, layoutBReordered);
    return true;
}

static void FastllmCudaW4A8ReleaseActivationScratch(W4A8ActivationScratch &scratch) {
    if (scratch.fp8 != nullptr) {
        FastllmCudaFree(scratch.fp8);
    }
    if (scratch.tokenScales != nullptr) {
        FastllmCudaFree(scratch.tokenScales);
    }
    scratch = W4A8ActivationScratch{};
}

static bool FastllmCudaW4A8QuantizeActivation(const fastllm::Data &input,
                                              int n,
                                              int m,
                                              W4A8ActivationScratch &scratch) {
    if (!FastllmCudaW4A8CanQuantizeActivation(input, n, m)) {
        return false;
    }

    FastllmCudaW4A8ReleaseActivationScratch(scratch);
    scratch.tokens = n;
    scratch.hidden = m;
    scratch.fp8Bytes = (size_t)n * m * sizeof(cutlass::float_e4m3_t);
    scratch.scaleBytes = (size_t)n * sizeof(float);
    scratch.fp8 = (cutlass::float_e4m3_t*)FastllmCudaMalloc(scratch.fp8Bytes);
    scratch.tokenScales = (float*)FastllmCudaMalloc(scratch.scaleBytes);
    if (scratch.fp8 == nullptr || scratch.tokenScales == nullptr) {
        FastllmCudaW4A8ReleaseActivationScratch(scratch);
        return false;
    }

    dim3 grid(n);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<grid, 256>>>(
            (const half*)input.cudaData, scratch.fp8, scratch.tokenScales, n, m);
    } else {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<grid, 256>>>(
            (const __nv_bfloat16*)input.cudaData, scratch.fp8, scratch.tokenScales, n, m);
    }
    if (cudaGetLastError() != cudaSuccess) {
        FastllmCudaW4A8ReleaseActivationScratch(scratch);
        return false;
    }
    return true;
}

static void FastllmCudaW4A8ReleasePackedWeightCache(fastllm::Data &weight) {
    if ((int)weight.extraCudaData.size() > W4A8_PACKED_WEIGHT_IDX &&
        weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] != nullptr) {
        FastllmCudaFree(weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX]);
        weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] = nullptr;
    }
    g_w4a8WeightCacheMetas.erase(&weight);
}

static bool FastllmCudaW4A8HasPackedWeightCache(const fastllm::Data &weight,
                                                int m,
                                                int k) {
    if ((int)weight.extraCudaData.size() <= W4A8_PACKED_WEIGHT_IDX ||
        weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] == nullptr) {
        return false;
    }
    auto it = g_w4a8WeightCacheMetas.find(&weight);
    if (it == g_w4a8WeightCacheMetas.end()) {
        return false;
    }
    const W4A8WeightCacheMeta &meta = it->second;
    return meta.magic == W4A8_CACHE_MAGIC &&
           meta.sourceType == weight.dataType &&
           meta.inChannels == m &&
           meta.outChannels == k &&
           meta.groupCnt == weight.groupCnt &&
           meta.group == weight.group &&
           meta.sourceCudaData == weight.cudaData &&
           meta.packedWeightBytes == (size_t)m * k / 2;
}

static bool FastllmCudaW4A8EnsurePackedWeightCache(fastllm::Data &weight,
                                                   int m,
                                                   int k) {
    if (!FastllmCudaW4A8CanUseInt4GroupSource(weight, m, k)) {
        return false;
    }
    if (FastllmCudaW4A8HasPackedWeightCache(weight, m, k)) {
        return true;
    }

    FastllmCudaW4A8ReleasePackedWeightCache(weight);

    size_t packedBytes = (size_t)m * k / 2;
    uint8_t *rawPackedWeight = (uint8_t*)FastllmCudaMalloc(packedBytes);
    uint8_t *cutlassPackedWeight = (uint8_t*)FastllmCudaMalloc(packedBytes);
    if (rawPackedWeight == nullptr || cutlassPackedWeight == nullptr) {
        if (rawPackedWeight != nullptr) {
            FastllmCudaFree(rawPackedWeight);
        }
        if (cutlassPackedWeight != nullptr) {
            FastllmCudaFree(cutlassPackedWeight);
        }
        return false;
    }

    int threads = 256;
    size_t words = (size_t)(m / 8) * k;
    FastllmCudaW4A8PackInt4GroupToVllmBKernel<<<(words + threads - 1) / threads, threads>>>(
        (const uint8_t*)weight.cudaData, (uint32_t*)rawPackedWeight, m, k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ||
        !FastllmCudaW4A8EncodeAndReorderInt4B(rawPackedWeight, cutlassPackedWeight, m, k)) {
        FastllmCudaFree(rawPackedWeight);
        FastllmCudaFree(cutlassPackedWeight);
        return false;
    }

    FastllmCudaFree(rawPackedWeight);
    if ((int)weight.extraCudaData.size() <= W4A8_PACKED_WEIGHT_IDX) {
        weight.extraCudaData.resize(W4A8_PACKED_WEIGHT_IDX + 1, nullptr);
    }
    weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] = cutlassPackedWeight;
    g_w4a8WeightCacheMetas[&weight] = W4A8WeightCacheMeta{
        W4A8_CACHE_MAGIC,
        weight.dataType,
        m,
        k,
        weight.groupCnt,
        weight.group,
        packedBytes,
        weight.cudaData,
    };
    return true;
}

} // namespace

bool TryCudaCutlassW4A8(const fastllm::Data &input, fastllm::Data &weight,
                        const fastllm::Data &bias, fastllm::Data &output,
                        int n, int m, int k) {
    (void)bias;
    (void)output;

    if (FastllmCudaW4A8PrepareCacheEnabled()) {
        (void)FastllmCudaW4A8EnsurePackedWeightCache(weight, m, k);
    }
    if (FastllmCudaW4A8PrepareActivationEnabled()) {
        W4A8ActivationScratch activation;
        (void)FastllmCudaW4A8QuantizeActivation(input, n, m, activation);
        FastllmCudaW4A8ReleaseActivationScratch(activation);
    }

    // Stage 4 only prepares independent weight/activation intermediates. The
    // actual W4A8 GEMM dispatch is wired in later stages after scale packing
    // and output handling are available.
    return false;
}
