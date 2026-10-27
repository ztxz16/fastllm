#include "devices/cuda/fastllm-cuda-hybrid-dsa.h"
#ifndef USE_ROCM
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>
#include <climits>
#include <cmath>

namespace {
using fastllm::Data;
using fastllm::DataType;
bool Dense(const Data &x, DataType type, const std::vector<int> &device) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || x.dataType != type ||
        x.dataDeviceIds != device || !x.cudaData || x.multiDeviceData || x.lockInCPU ||
        x.dims.empty() || x.dims.size() != x.strides.size()) return false;
    uint64_t stride = 1;
    for (int i = (int)x.dims.size()-1; i >= 0; --i) {
        if (x.dims[i] <= 0 || (x.dims[i] > 1 && x.strides[i] != stride)) return false;
        stride *= x.dims[i];
    }
    return true;
}
bool Position(const Data &x) {
    return x.dataDeviceIds == std::vector<int>{FastllmCudaGetDevice()} &&
        Dense(x, DataType::INT32, x.dataDeviceIds) && x.Count(0) == 1;
}

// Each logical lane accumulates columns t, t+256, ... just like the FP32
// Linear fallback. Adjacent-pair shuffles preserve its increasing-stride
// tree; the usual descending shuffle tree would change its rounded sum.
template<int Warps>
__device__ float Projection(const __nv_bfloat16 *input, const float *weight,
        int columns, float *partial) {
    constexpr int parts = 8 / Warps;
    const int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) % Warps;
    float sums[parts] = {};
    for (int base = 0; base < columns; base += 256) {
#pragma unroll
        for (int p = 0; p < parts; ++p) {
            const int col = base + (warp + p * Warps) * 32 + lane;
            if (col < columns)
                sums[p] = __fmaf_rn(__bfloat162float(input[col]), weight[col], sums[p]);
        }
    }
#pragma unroll
    for (int p = 0; p < parts; ++p) {
#pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            const float peer = __shfl_down_sync(0xffffffff, sums[p], step);
            if ((lane & (2 * step - 1)) == 0) sums[p] += peer;
        }
        if (lane == 0) partial[warp + p * Warps] = sums[p];
    }
    __syncthreads();
    float value = 0;
    if (warp == 0) {
        value = lane < 8 ? partial[lane] : 0;
#pragma unroll
        for (int step = 1; step < 8; step *= 2) {
            const float peer = __shfl_down_sync(0xffffffff, value, step);
            if (lane < 8 && (lane & (2 * step - 1)) == 0) value += peer;
        }
    }
    return __fadd_rn(value, 0.0f); // Preserve the original zero-bias addition.
}

__global__ void HybridDsaInputProjections(const __nv_bfloat16 *input,
        const float *keyWeight, const float *gateWeight, const float *headWeight,
        __nv_bfloat16 *key, __nv_bfloat16 *gate, float *head, int columns,
        float headScale, float channelScale) {
    const int row = blockIdx.x;
    const float *weight = row < 128 ? keyWeight + size_t(row) * columns :
        row < 256 ? gateWeight + size_t(row - 128) * columns : headWeight + size_t(row - 256) * columns;
    __shared__ float partial[8];
    const float value = Projection<8>(input, weight, columns, partial);
    if (threadIdx.x == 0) {
        if (row < 128) key[row] = __float2bfloat16_rn(value);
        else if (row < 256) gate[row - 128] = __float2bfloat16_rn(value);
        else head[row - 256] = __fmul_rn(__fmul_rn(value, headScale), channelScale);
    }
}

__global__ void HybridDsaQueryProjection(const __nv_bfloat16 *input,
        const float *weight, __nv_bfloat16 *query, int columns) {
    const int localRow = threadIdx.x / 128, row = blockIdx.x * 2 + localRow;
    __shared__ float partial[2][8];
    const float value = Projection<4>(input, weight + size_t(row) * columns, columns, partial[localRow]);
    if (threadIdx.x % 128 == 0) query[row] = __float2bfloat16_rn(value);
}

__global__ void HybridDsaKeyNorm(__nv_bfloat16 *key, const float *gamma, const float *beta) {
    // Original 64-thread LayerNorm, including its FP32 variance expression
    // and double epsilon/sqrt. The projection's BF16 boundary is retained.
    __shared__ float sums[64], squares[64], mean, deviation;
    const int tid = threadIdx.x;
    float sum = 0, sum2 = 0;
    for (int i = tid; i < 128; i += 64) {
        const float x = __bfloat162float(key[i]);
        sum += x;
        sum2 += x * x;
    }
    sums[tid] = sum; squares[tid] = sum2;
    __syncthreads();
    for (int step = 32; step; step >>= 1) {
        if (tid < step) { sums[tid] += sums[tid + step]; squares[tid] += squares[tid + step]; }
        __syncthreads();
    }
    if (!tid) {
        mean = sums[0] / 128;
        deviation = squares[0] + mean * mean * 128 - 2 * mean * 128 * mean;
        deviation = sqrt(deviation / 128 + double(1e-6f));
    }
    __syncthreads();
    for (int i = tid; i < 128; i += 64)
        key[i] = __float2bfloat16_rn((__bfloat162float(key[i]) - mean) / deviation * gamma[i] + beta[i]);
}
__global__ void QuantizeRotated(const float *input, __nv_bfloat16 *output) {
    // Match NaiveQuantizeIndexer and its subsequent BF16 scale conversion
    // and multiplication. The Hadamard GEMM and all rounding remain intact.
    __shared__ float maximum[128];
    const int d = threadIdx.x, row = blockIdx.x;
    const float x = __bfloat162float(__float2bfloat16_rn(input[size_t(row) * 128 + d]));
    maximum[d] = fabsf(x);
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (d < step) maximum[d] = fmaxf(maximum[d], maximum[d + step]);
        __syncthreads();
    }
    const float scale = exp2f(ceilf(log2f(fmaxf(maximum[0], 1e-4f) / 448.0f)));
    const float value = float(__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / scale))));
    const float roundedValue = __bfloat162float(__float2bfloat16_rn(value));
    const float roundedScale = __bfloat162float(__float2bfloat16_rn(scale));
    output[size_t(row) * 128 + d] = __float2bfloat16_rn(roundedValue * roundedScale);
}
__global__ void AppendPool(const __nv_bfloat16 *key, const __nv_bfloat16 *gate,
        const float *ape, const int32_t *position, __nv_bfloat16 *keys,
        __nv_bfloat16 *gates, __nv_bfloat16 *pooled) {
    const int d = threadIdx.x, slot = *position % 4;
    keys[slot * 128 + d] = key[d];
    gates[slot * 128 + d] = gate[d];
    // Pooling runs on the same four-row view each replay. Unfinished rows
    // are initialized, and their provisional pooled result is never stored.
    for (int row = slot + 1; row < 4; ++row) {
        keys[row * 128 + d] = __float2bfloat16_rn(0.f);
        gates[row * 128 + d] = __float2bfloat16_rn(0.f);
    }
    // Each thread owns one channel throughout append and four-row pooling.
    // Keep BuildCompressedKV's sequential softmax/FMA and final BF16 cast.
    float scores[4], values[4], maximum = -1e30f;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        scores[row] = __bfloat162float(gates[row * 128 + d]) + ape[row * 128 + d];
        values[row] = __bfloat162float(keys[row * 128 + d]);
        maximum = fmaxf(maximum, scores[row]);
    }
    float sum = 0, value = 0;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        const float e = expf(scores[row] - maximum);
        sum += e;
        value += e * values[row];
    }
    pooled[d] = __float2bfloat16_rn(value / fmaxf(sum, 1e-30f));
}
__global__ void StorePool(const __nv_bfloat16 *pooled, const int32_t *position,
        __nv_bfloat16 *keys) {
    if (*position % 4 == 3) keys[size_t(*position / 4) * 128 + threadIdx.x] = pooled[threadIdx.x];
}
__global__ void Advance(int32_t *position) { ++*position; }
}

bool FastllmCudaHybridDsaProjections(const Data &input, const Data &normalized,
        const Data &keyWeight, const Data &gateWeight, const Data &queryWeight, const Data &headWeight,
        const Data &gamma, const Data &beta, Data &key, Data &gate, Data &query, Data &head) {
    if (!FastllmCudaHybridDsaSupported() || input.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
        !Dense(input, DataType::BFLOAT16, input.dataDeviceIds) ||
        !Dense(normalized, DataType::BFLOAT16, input.dataDeviceIds) ||
        input.dims.size() != 3 || normalized.dims.size() != 3 ||
        input.dims[0] != 1 || input.dims[1] != 1 || normalized.dims[0] != 1 || normalized.dims[1] != 1)
        return false;
    const int columns = input.dims[2], rank = normalized.dims[2];
    for (const auto *x : {&keyWeight, &gateWeight, &queryWeight, &headWeight, &gamma, &beta})
        if (!Dense(*x, DataType::FLOAT32, input.dataDeviceIds)) return false;
    if (keyWeight.dims != std::vector<int>({128, columns}) || gateWeight.dims != keyWeight.dims ||
        queryWeight.dims != std::vector<int>({4096, rank}) || headWeight.dims != std::vector<int>({32, columns}) ||
        gamma.Count(0) != 128 || beta.Count(0) != 128) return false;
    auto allocate = [&](Data &x, DataType type, const std::vector<int> &dims) {
        x.dataType = type; x.UpdateUnitSize(); x.Resize(dims);
        x.ToDevice(fastllm::DataDevice::CUDA, input.dataDeviceIds, false); x.Allocate(false);
    };
    allocate(key, DataType::BFLOAT16, {1, 1, 128});
    allocate(gate, DataType::BFLOAT16, {1, 1, 128});
    allocate(query, DataType::BFLOAT16, {1, 1, 32, 128});
    allocate(head, DataType::FLOAT32, {1, 1, 32});
    HybridDsaInputProjections<<<288, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData, (const float*)keyWeight.cudaData, (const float*)gateWeight.cudaData,
        (const float*)headWeight.cudaData, (__nv_bfloat16*)key.cudaData, (__nv_bfloat16*)gate.cudaData,
        (float*)head.cudaData, columns, 1.0f / std::sqrt(32.0f), 1.0f / std::sqrt(128.0f));
    HybridDsaQueryProjection<<<2048, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)normalized.cudaData, (const float*)queryWeight.cudaData,
        (__nv_bfloat16*)query.cudaData, rank);
    HybridDsaKeyNorm<<<1, 64, 0, cudaStreamPerThread>>>(
        (__nv_bfloat16*)key.cudaData, (const float*)gamma.cudaData, (const float*)beta.cudaData);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaHybridDsaQuantizeRotated(const Data &input, Data &output) {
    const auto device = std::vector<int>{FastllmCudaGetDevice()};
    if (!FastllmCudaHybridDsaSupported() || !Dense(input, DataType::FLOAT32, device) ||
        input.dims.back() != 128 || input.Count(0) / 128 > INT_MAX ||
        &input == &output || input.cudaData == output.cudaData) return false;
    output.dataType = DataType::BFLOAT16; output.UpdateUnitSize(); output.Resize(input.dims);
    output.ToDevice(fastllm::DataDevice::CUDA, device, false); output.Allocate(false);
    QuantizeRotated<<<input.Count(0) / 128, 128, 0, cudaStreamPerThread>>>(
        (const float*)input.cudaData, (__nv_bfloat16*)output.cudaData);
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaHybridDsaAppendPool(const Data &key, const Data &gate, const Data &ape,
        const Data &position, Data &tailKeys, Data &tailGates, Data &pooled) {
    if (!Position(position) || !Dense(key, DataType::BFLOAT16, position.dataDeviceIds) ||
        !Dense(gate, DataType::BFLOAT16, position.dataDeviceIds) || key.Count(0) != 128 || gate.Count(0) != 128 ||
        !Dense(tailKeys, DataType::BFLOAT16, position.dataDeviceIds) || tailKeys.Count(0) != 512 ||
        !Dense(tailGates, DataType::BFLOAT16, position.dataDeviceIds) || tailGates.Count(0) != 512 ||
        !Dense(ape, DataType::FLOAT32, position.dataDeviceIds) || ape.Count(0) != 512) return false;
    pooled.dataType = DataType::BFLOAT16; pooled.UpdateUnitSize(); pooled.Resize({1, 1, 128});
    pooled.ToDevice(fastllm::DataDevice::CUDA, position.dataDeviceIds, false); pooled.Allocate(false);
    AppendPool<<<1, 128, 0, cudaStreamPerThread>>>((const __nv_bfloat16*)key.cudaData,
        (const __nv_bfloat16*)gate.cudaData, (const float*)ape.cudaData, (const int32_t*)position.cudaData,
        (__nv_bfloat16*)tailKeys.cudaData, (__nv_bfloat16*)tailGates.cudaData, (__nv_bfloat16*)pooled.cudaData);
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaHybridDsaStorePool(const Data &pooled, const Data &position, Data &keys) {
    if (!Position(position) || !Dense(pooled, DataType::BFLOAT16, position.dataDeviceIds) || pooled.Count(0) != 128 ||
        !Dense(keys, DataType::BFLOAT16, position.dataDeviceIds) || keys.dims.size() != 3 ||
        keys.dims[0] != 1 || keys.dims[2] != 128) return false;
    StorePool<<<1, 128, 0, cudaStreamPerThread>>>((const __nv_bfloat16*)pooled.cudaData,
        (const int32_t*)position.cudaData, (__nv_bfloat16*)keys.cudaData);
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaHybridDsaAdvance(Data &position) {
    if (!Position(position)) return false;
    Advance<<<1, 1, 0, cudaStreamPerThread>>>((int32_t*)position.cudaData);
    return cudaGetLastError() == cudaSuccess;
}
#endif
