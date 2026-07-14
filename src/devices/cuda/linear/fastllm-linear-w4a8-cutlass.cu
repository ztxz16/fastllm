#include "fastllm-cuda.cuh"
#include "libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cuh"

#include "cute/tensor.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

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

static std::unordered_map<const fastllm::Data*, W4A8WeightCacheMeta> g_w4a8WeightCacheMetas;

static bool FastllmCudaW4A8PrepareCacheEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_CACHE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8BasicShapeSupported(int m, int k) {
    return m > 0 && k > 0 && (m % 128) == 0 && (k % 128) == 0;
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
    (void)input;
    (void)bias;
    (void)output;
    (void)n;

    if (FastllmCudaW4A8PrepareCacheEnabled()) {
        (void)FastllmCudaW4A8EnsurePackedWeightCache(weight, m, k);
    }

    // Stage 3 only prepares an independent packed weight cache. The actual
    // W4A8 GEMM dispatch is wired in later stages after activation quantization
    // and scale packing are available.
    return false;
}
