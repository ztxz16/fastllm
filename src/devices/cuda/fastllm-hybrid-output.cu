#include "devices/cuda/fastllm-cuda-hybrid-output.h"
#ifndef USE_ROCM
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <math_constants.h>
#include <cstdint>

namespace {
using namespace fastllm;
bool Dense(const Data &x, DataType type) {
    if (x.dataType != type || x.dataDevice != CUDA || !x.cudaData || x.multiDeviceData ||
        x.lockInCPU || x.dataDeviceIds.size() != 1 || x.dataDeviceIds[0] != FastllmCudaGetDevice() ||
        x.strides.size() != x.dims.size() || x.dims.empty()) return false;
    size_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || x.strides[i] != stride) return false;
        stride *= x.dims[i];
    }
    return true;
}
void Allocate(Data &x, DataType type, const std::vector<int> &dims, const Data &source) {
    x.dataType = type; x.UpdateUnitSize(); x.Resize(dims);
    x.ToDevice(CUDA, source.dataDeviceIds, false); x.Allocate(false);
}

__global__ void HybridMoeHcPost(const __nv_bfloat16 *shared, const __nv_bfloat16 *routed,
        const __nv_bfloat16 *residual, const float *post, const float *comb,
        __nv_bfloat16 *output, int rows, int hc, int dim) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows) * hc * dim) return;
    const int d = index % dim, target = index / dim % hc, row = index / (size_t(dim) * hc);
    const size_t channel = size_t(row) * dim + d;
    // AddTo rounds the shared+routed result to BF16 before HcPost sees it.
    const float sum = __bfloat162float(__float2bfloat16_rn(__fmaf_rn(
        __bfloat162float(routed[channel]), 1.0f, __bfloat162float(shared[channel]))));
    double value = double(post[size_t(row) * hc + target]) * sum;
    for (int source = 0; source < hc; ++source)
        value += double(comb[(size_t(row) * hc + source) * hc + target]) *
            __bfloat162float(residual[(size_t(row) * hc + source) * dim + d]);
    output[index] = __float2bfloat16_rn(float(value));
}

__global__ void HybridHcMeanNorm(const __nv_bfloat16 *input, const float *weight,
        __nv_bfloat16 *output, int hc, float eps) {
    // Match HcMean's ordered FP32 Mul/AddTo and its BF16 boundary. Keep the
    // original 256-thread RMSNorm reduction and both normalization roundings.
    float values[16];
    float partial = 0;
    const float factor = 1.0f / hc;
    #pragma unroll
    for (int part = 0; part < 16; ++part) {
        const int channel = threadIdx.x + part * 256;
        float value = __bfloat162float(input[channel]) * factor;
        for (int i = 1; i < hc; ++i)
            value = fmaf(__bfloat162float(input[i * 4096 + channel]), factor, value);
        values[part] = __bfloat162float(__float2bfloat16_rn(value));
        partial += values[part] * values[part];
    }
    __shared__ float sums[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    if (!lane) sums[warp] = partial;
    __syncthreads();
    if (!warp) {
        float sum = lane < 8 ? sums[lane] : 0;
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (!lane) sums[0] = sum;
    }
    __syncthreads();
    const float scale = rsqrtf(sums[0] / 4096 + eps);
    #pragma unroll
    for (int part = 0; part < 16; ++part) {
        const int channel = threadIdx.x + part * 256;
        float normalized = __bfloat162float(__float2bfloat16_rn(values[part] * scale));
        output[channel] = __float2bfloat16_rn(normalized * weight[channel]);
    }
}

struct Maximum { float value; int index; };
__device__ Maximum Merge(Maximum a, Maximum b) {
    // Top1's halving shared-memory tree prefers bit-reversed lane order on
    // ties, then the earliest element scanned by that lane. Preserve both.
    const unsigned ap = __brev(unsigned(a.index & 255));
    const unsigned bp = __brev(unsigned(b.index & 255));
    if (b.value > a.value || (b.value == a.value &&
        (bp < ap || (bp == ap && b.index < a.index)))) return b;
    return a;
}
__device__ Maximum Reduce(Maximum value, Maximum *warps) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset >>= 1) {
        Maximum other{__shfl_down_sync(0xffffffffu, value.value, offset),
                      __shfl_down_sync(0xffffffffu, value.index, offset)};
        if (lane + offset < 32) value = Merge(value, other);
    }
    if (!lane) warps[warp] = value;
    __syncthreads();
    value = lane < 8 ? warps[lane] : Maximum{-CUDART_INF_F, 0};
    if (!warp) {
        for (int offset = 16; offset; offset >>= 1) {
            Maximum other{__shfl_down_sync(0xffffffffu, value.value, offset),
                          __shfl_down_sync(0xffffffffu, value.index, offset)};
            if (lane + offset < 32) value = Merge(value, other);
        }
    }
    return value;
}
__global__ void HybridTop1Part(const __nv_bfloat16 *input, Maximum *partials, int size) {
    const int begin = blockIdx.x * 4096;
    Maximum value{-CUDART_INF_F, 0};
    for (int i = begin + threadIdx.x; i < min(begin + 4096, size); i += 256) {
        float x = __bfloat162float(input[i]);
        if (x > value.value) value = {x, i};
    }
    __shared__ Maximum warps[8];
    value = Reduce(value, warps);
    if (!threadIdx.x) partials[blockIdx.x] = value;
}
__global__ void HybridTop1Finish(const Maximum *partials, float *output, int count) {
    Maximum value{-CUDART_INF_F, 0};
    for (int i = threadIdx.x; i < count; i += 256) value = Merge(value, partials[i]);
    __shared__ Maximum warps[8];
    value = Reduce(value, warps);
    if (!threadIdx.x) { output[0] = value.index; output[1] = value.value; }
}
} // namespace

bool FastllmCudaHybridHcMeanNorm(const fastllm::Data &input,
        const fastllm::Data &weight, float eps, fastllm::Data &output) {
    if (!Dense(input, BFLOAT16) || !Dense(weight, FLOAT32) ||
        input.dims.size() != 4 || input.dims[0] != 1 || input.dims[1] != 1 ||
        input.dims[2] > 16 || input.dims[3] != 4096 || weight.Count(0) != 4096 ||
        input.dataDeviceIds != weight.dataDeviceIds) return false;
    Allocate(output, BFLOAT16, {1, 1, 4096}, input);
    HybridHcMeanNorm<<<1, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData, (const float*)weight.cudaData,
        (__nv_bfloat16*)output.cudaData, input.dims[2], eps);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "Hybrid HC output normalization failed.");
    return true;
}

bool FastllmCudaHybridMoeHcPost(const fastllm::Data &shared, const fastllm::Data &routed,
        const fastllm::Data &residual, const fastllm::Data &post,
        const fastllm::Data &comb, fastllm::Data &output) {
    if (FastllmCudaRuntimeArch() != 75 || !Dense(shared, BFLOAT16) || !Dense(routed, BFLOAT16) ||
        !Dense(residual, BFLOAT16) || !Dense(post, FLOAT32) || !Dense(comb, FLOAT32) ||
        residual.dims.size() != 4) return false;
    const size_t rows = size_t(residual.dims[0]) * residual.dims[1];
    const int hc = residual.dims[2], dim = residual.dims[3];
    if (rows > 32 || shared.Count(0) != size_t(rows) * dim || routed.Count(0) != shared.Count(0) ||
        post.Count(0) != size_t(rows) * hc || comb.Count(0) != size_t(rows) * hc * hc) return false;
    for (const auto *x : {&shared, &routed, &residual, &post, &comb})
        if (output.cudaData && output.cudaData == x->cudaData) return false;
    Allocate(output, BFLOAT16, residual.dims, residual);
    const size_t count = size_t(rows) * hc * dim;
    HybridMoeHcPost<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)shared.cudaData, (const __nv_bfloat16*)routed.cudaData,
        (const __nv_bfloat16*)residual.cudaData, (const float*)post.cudaData, (const float*)comb.cudaData,
        (__nv_bfloat16*)output.cudaData, rows, hc, dim);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "Hybrid expert mHC finish failed.");
    return true;
}

bool FastllmCudaHybridTop1(const fastllm::Data &input, fastllm::Data &output) {
    if (!Dense(input, BFLOAT16) || input.Count(0) != input.dims.back() ||
        input.Count(0) >= (1 << 24)) return false;
    const int size = input.Count(0), count = (size + 4095) / 4096;
    Data partials;
    Allocate(partials, INT32, {count, 2}, input);
    Allocate(output, FLOAT32, {1, 2}, input);
    HybridTop1Part<<<count, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData, (Maximum*)partials.cudaData, size);
    HybridTop1Finish<<<1, 256, 0, cudaStreamPerThread>>>(
        (const Maximum*)partials.cudaData, (float*)output.cudaData, count);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "Hybrid greedy selection failed.");
    return true;
}
#endif
