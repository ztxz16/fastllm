#include "devices/cuda/fastllm-cuda-hybrid-kda.h"
#ifndef USE_ROCM
#include "fastllm-cuda.cuh"
#include "utils.h"
#include "gguf.h"
#include "fastllm-kda-prefill.cuh"
#include <cstdint>

namespace {
using fastllm::Data;
using fastllm::DataType;
using fastllm::DataDevice;

// Keep the original 256-thread GEMV's eight-product partial sums and
// compensated reduction tree. The grid selects a weight/output pair without
// concatenating weights or allocating a persistent copy.
__global__ void HybridKdaInputProjections(const __nv_bfloat16 *input,
        const __nv_bfloat16 *fw, const __nv_bfloat16 *bw, const __nv_bfloat16 *gw,
        __nv_bfloat16 *f, __nv_bfloat16 *b, __nv_bfloat16 *g,
        int columns, int fRows, int bRows) {
    int row = blockIdx.x;
    const __nv_bfloat16 *weight = fw;
    __nv_bfloat16 *output = f;
    if (row >= fRows) {
        row -= fRows; weight = bw; output = b;
        if (row >= bRows) { row -= bRows; weight = gw; output = g; }
    }
    const int tid = threadIdx.x;
    weight += (size_t)row * columns;
    float value = 0;
    for (int i = tid * 8; i < columns; i += 256 * 8) {
        union { uint4 packed; __nv_bfloat16 values[8]; } x, w;
        x.packed = *reinterpret_cast<const uint4 *>(input + i);
        w.packed = *reinterpret_cast<const uint4 *>(weight + i);
        float partial = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            partial += __bfloat162float(x.values[j]) * __bfloat162float(w.values[j]);
        value += partial;
    }
    __shared__ float sums[256];
    sums[tid] = value;
    __syncthreads();
    float correction = 0;
    #pragma unroll
    for (int step = 128; step >= 32; step >>= 1) {
        if (tid < step) {
            float other = sums[tid + step] - correction;
            float sum = value + other;
            correction = (sum - value) - other;
            sums[tid] = value = sum;
        }
        __syncthreads();
    }
    if (tid < 32) {
        #pragma unroll
        for (int step = 16; step; step >>= 1) {
            float peer = __shfl_down_sync(0xffffffffu, value, step);
            if (tid < step) {
                float other = peer - correction;
                float sum = value + other;
                correction = (sum - value) - other;
                value = sum;
            }
        }
        if (!tid) output[row] = __float2bfloat16_rn(value);
    }
}

// Eight independent output rows share a CTA. Keep the short BF16 GEMV's
// eight products per lane and compensated binary reduction, including zeros
// in inactive lanes for ranks below 256.
__global__ void HybridKdaLowRank(const __nv_bfloat16 *f, const __nv_bfloat16 *g,
        const __nv_bfloat16 *fw, const __nv_bfloat16 *gw,
        __nv_bfloat16 *out, int rank, int channels) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= channels) return;
    const auto *input = blockIdx.y ? g : f;
    const auto *weight = (blockIdx.y ? gw : fw) + (size_t)row * rank;
    float value = 0;
    if (lane * 8 < rank) {
        union { uint4 packed; __nv_bfloat16 values[8]; } x, w;
        x.packed = *reinterpret_cast<const uint4 *>(input + lane * 8);
        w.packed = *reinterpret_cast<const uint4 *>(weight + lane * 8);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            value += __bfloat162float(x.values[i]) * __bfloat162float(w.values[i]);
    }
    float correction = 0;
    #pragma unroll
    for (int step = 16; step; step >>= 1) {
        float peer = __shfl_down_sync(0xffffffffu, value, step);
        if (lane < step) {
            float other = peer - correction;
            float sum = value + other;
            correction = (sum - value) - other;
            value = sum;
        }
    }
    if (!lane) out[(size_t)blockIdx.y * channels + row] = __float2bfloat16_rn(value);
}

__device__ __forceinline__ __nv_bfloat16 HybridKdaConv(
        const __nv_bfloat16 *input, const float *weight, __nv_bfloat16 *cache,
        int channel, int channels, int kernelSize) {
    float value = 0;
    const int history = kernelSize - 1;
    for (int tap = 0; tap < history; ++tap) {
        value += __bfloat162float(cache[(size_t)tap * channels + channel]) *
            weight[(size_t)channel * kernelSize + tap];
    }
    const auto current = input[channel];
    value += __bfloat162float(current) * weight[(size_t)channel * kernelSize + history];
    // One thread owns the channel, including all history taps. No reader can
    // race with these writes; shifts run in the same order as the old kernel.
    for (int tap = 0; tap + 1 < history; ++tap)
        cache[(size_t)tap * channels + channel] = cache[(size_t)(tap + 1) * channels + channel];
    if (history) cache[(size_t)(history - 1) * channels + channel] = current;
    return __float2bfloat16_rn(value * (1.0f / (1.0f + expf(-value))));
}

// Fold the three one-token convolutions and their cache updates into the
// existing KDA preparation. Retain the convolution's BF16 boundary and the
// sequential 128-channel FP32 norm, rather than changing the reduction tree.
__global__ void HybridKdaConvPrepare(
        const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
        const float *qw, const float *kw, const float *vw, __nv_bfloat16 *cache,
        const __nv_bfloat16 *gate, const __nv_bfloat16 *rawBeta,
        const float *aLog, const float *dtBias,
        float *query, float *key, float *retention, float *beta, __nv_bfloat16 *value,
        int heads, int kernelSize, float lowerBound) {
    constexpr int dimension = KIMI_K3_KDA_DIMENSION;
    const int head = blockIdx.x, d = threadIdx.x;
    const int channels = heads * dimension, channel = head * dimension + d;
    const size_t historyStride = (size_t)(kernelSize - 1) * channels;
    __shared__ float rawQ[dimension], rawK[dimension], qScale, kScale;
    rawQ[d] = __bfloat162float(HybridKdaConv(q, qw, cache, channel, channels, kernelSize));
    rawK[d] = __bfloat162float(HybridKdaConv(k, kw, cache + historyStride, channel, channels, kernelSize));
    value[channel] = HybridKdaConv(v, vw, cache + 2 * historyStride, channel, channels, kernelSize);
    __syncthreads();
    if (!d) {
        float qs = 0, ks = 0;
        for (int i = 0; i < dimension; ++i) {
            qs = __fmaf_rn(rawQ[i], rawQ[i], qs);
            ks = __fmaf_rn(rawK[i], rawK[i], ks);
        }
        qScale = rsqrtf(qs + 1e-6f);
        kScale = rsqrtf(ks + 1e-6f);
        float activated = 1.0f / (1.0f + expf(-__bfloat162float(rawBeta[head])));
        beta[head] = __bfloat162float(__float2bfloat16_rn(activated));
    }
    __syncthreads();
    query[channel] = rawQ[d] * qScale;
    key[channel] = rawK[d] * kScale;
    float raw = __bfloat162float(gate[channel]);
    float activated = expf(aLog[head]) * (raw + dtBias[channel]);
    float decay = __fmul_rn(lowerBound, 1.0f / (1.0f + expf(-activated)));
    retention[channel] = expf(decay);
}
// Four warps prepare the convolutions and sequential Q/K norms while the
// other 28 warps compute the low-rank gates. They meet only for retention,
// which consumes the rounded F gate. The named barrier belongs exclusively
// to the 128 convolution threads; the final CTA barrier publishes rawF.
__global__ void HybridKdaPrepare(
        const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
        const __nv_bfloat16 *f, const __nv_bfloat16 *g,
        const __nv_bfloat16 *fw, const __nv_bfloat16 *gw,
        const float *qw, const float *kw, const float *vw, __nv_bfloat16 *cache,
        __nv_bfloat16 *gates, const __nv_bfloat16 *rawBeta,
        const float *aLog, const float *dtBias,
        float *query, float *key, float *retention, float *beta, __nv_bfloat16 *value,
        int rank, int heads, int kernelSize, float lowerBound) {
    constexpr int dimension = KIMI_K3_KDA_DIMENSION, warps = (1024 - 128) / 32;
    const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int channels = heads * dimension;
    __shared__ float rawQ[dimension], rawK[dimension], rawF[dimension], qScale, kScale;
    if (tid >= 128) for (int d = (tid - 128) / 32; d < dimension; d += warps) {
        const int channel = head * dimension + d;
        float fv = 0, gv = 0;
        if (lane * 8 < rank) {
            union { uint4 packed; __nv_bfloat16 values[8]; } xf, xg, wf, wg;
            xf.packed = *reinterpret_cast<const uint4 *>(f + lane * 8);
            xg.packed = *reinterpret_cast<const uint4 *>(g + lane * 8);
            wf.packed = *reinterpret_cast<const uint4 *>(fw + size_t(channel) * rank + lane * 8);
            wg.packed = *reinterpret_cast<const uint4 *>(gw + size_t(channel) * rank + lane * 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                fv += __bfloat162float(xf.values[i]) * __bfloat162float(wf.values[i]);
                gv += __bfloat162float(xg.values[i]) * __bfloat162float(wg.values[i]);
            }
        }
        float fc = 0, gc = 0;
#pragma unroll
        for (int step = 16; step; step >>= 1) {
            const float fp = __shfl_down_sync(0xffffffffu, fv, step);
            const float gp = __shfl_down_sync(0xffffffffu, gv, step);
            if (lane < step) {
                float other = fp - fc, sum = fv + other;
                fc = (sum - fv) - other;
                fv = sum;
                other = gp - gc;
                sum = gv + other;
                gc = (sum - gv) - other;
                gv = sum;
            }
        }
        if (!lane) {
            const auto gate = __float2bfloat16_rn(fv);
            gates[channel] = gate;
            gates[channels + channel] = __float2bfloat16_rn(gv);
            rawF[d] = __bfloat162float(gate);
        }
    }
    if (tid < dimension) {
        const int channel = head * dimension + tid;
        const size_t stride = size_t(kernelSize - 1) * channels;
        rawQ[tid] = __bfloat162float(HybridKdaConv(q, qw, cache, channel, channels, kernelSize));
        rawK[tid] = __bfloat162float(HybridKdaConv(k, kw, cache + stride, channel, channels, kernelSize));
        value[channel] = HybridKdaConv(v, vw, cache + 2 * stride, channel, channels, kernelSize);
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (!tid) {
            float qs = 0, ks = 0;
            for (int i = 0; i < dimension; ++i) {
                qs = __fmaf_rn(rawQ[i], rawQ[i], qs);
                ks = __fmaf_rn(rawK[i], rawK[i], ks);
            }
            qScale = rsqrtf(qs + 1e-6f);
            kScale = rsqrtf(ks + 1e-6f);
            float activated = 1.0f / (1.0f + expf(-__bfloat162float(rawBeta[head])));
            beta[head] = __bfloat162float(__float2bfloat16_rn(activated));
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        query[channel] = rawQ[tid] * qScale;
        key[channel] = rawK[tid] * kScale;
    }
    __syncthreads();
    if (tid < dimension) {
        const int channel = head * dimension + tid;
        float activated = expf(aLog[head]) * (rawF[tid] + dtBias[channel]);
        float decay = __fmul_rn(lowerBound, 1.0f / (1.0f + expf(-activated)));
        retention[channel] = expf(decay);
    }
}

// One head per CTA helps a single wave of SM75 heads; larger grids and
// smaller projections retain the original two-kernel preparation.
bool UseFusedPreparation(int device, int heads, int rank) {
    if (heads < 32 || heads > 64 || rank < 128) return false;
    static thread_local int cachedDevice = -1;
    static thread_local bool supported = false;
    if (cachedDevice != device) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
            return false;
        supported = major == 7 && minor == 5;
        cachedDevice = device;
    }
    return supported;
}

}

bool FastllmCudaHybridKdaInputProjections(const Data &input,
        const Data &fw, const Data &bw, const Data &gw, Data &f, Data &b, Data &g) {
    if (input.dims.size() != 3 || input.dims[0] != 1 || input.dims[1] != 1 ||
        input.dims[2] <= 256 || input.dims[2] % 8 ||
        input.dataDeviceIds.size() != 1 || input.dataDeviceIds[0] != FastllmCudaGetDevice()) return false;
    const int columns = input.dims[2];
    auto resident = [&](const Data &x) {
        return x.dataDevice == DataDevice::CUDA && x.cudaData && !x.lockInCPU &&
            !x.multiDeviceData && x.dataDeviceIds == input.dataDeviceIds &&
            reinterpret_cast<uintptr_t>(x.cudaData) % 16 == 0;
    };
    auto dense = [&](const Data &x) {
        if (x.strides.size() != x.dims.size()) return false;
        uint64_t stride = 1;
        for (int i = (int)x.dims.size() - 1; i >= 0; --i) {
            if (x.strides[i] != stride) return false;
            stride *= x.dims[i];
        }
        return true;
    };
    if (!resident(input) || input.dataType != DataType::BFLOAT16 || !dense(input)) return false;
    for (const Data *w : {&fw, &bw, &gw}) {
        if (!resident(*w) || w->dims.size() != 2 || w->dims[0] <= 0 ||
            w->dims[0] > 1024 || w->dims[1] != columns) return false;
        if (w->dataType == DataType::BFLOAT16 && dense(*w)) continue;
        const auto *tensor = static_cast<const ggml_tensor*>(w->ggmlTensor);
        if (w->dataType != DataType::DATA_GGUF_FORMAT || w->ggmlType != GGML_TYPE_BF16 ||
            !tensor || tensor->type != GGML_TYPE_BF16 || tensor->ne[0] != columns ||
            tensor->ne[1] != w->dims[0] || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            tensor->nb[0] != 2 || tensor->nb[1] != (size_t)columns * 2) return false;
    }
    if (&f == &b || &f == &g || &b == &g) return false;
    for (Data *out : {&f, &b, &g}) {
        for (const Data *in : {&input, &fw, &bw, &gw})
            if (out == in || (out->cudaData && out->cudaData == in->cudaData)) return false;
    }
    Data *outputs[] = {&f, &b, &g};
    const Data *weights[] = {&fw, &bw, &gw};
    for (int i = 0; i < 3; ++i) {
        auto &out = *outputs[i];
        out.dataType = DataType::BFLOAT16; out.UpdateUnitSize();
        out.Resize({1, 1, weights[i]->dims[0]});
        out.ToDevice(DataDevice::CUDA, input.dataDeviceIds, false);
        out.Allocate(false);
    }
    HybridKdaInputProjections<<<fw.dims[0] + bw.dims[0] + gw.dims[0], 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData,
        (const __nv_bfloat16*)fw.cudaData, (const __nv_bfloat16*)bw.cudaData, (const __nv_bfloat16*)gw.cudaData,
        (__nv_bfloat16*)f.cudaData, (__nv_bfloat16*)b.cudaData, (__nv_bfloat16*)g.cudaData,
        columns, fw.dims[0], bw.dims[0]);
    fastllm::AssertInFastLLM(cudaGetLastError() == cudaSuccess, "Hybrid KDA input projections failed.");
    return true;
}

bool FastllmCudaHybridKdaDecode(
        const Data &q, const Data &k, const Data &v,
        const Data &f, const Data &g, const Data &rawBeta,
        const Data &fw, const Data &gw, const Data &qw, const Data &kw, const Data &vw,
        const Data &aLog, const Data &dtBias, const Data &norm,
        Data &convCache, Data &state, int heads, int dimension,
        int kernelSize, float lowerBound, float eps, Data &output) {
    if (heads <= 0 || heads > 256 || dimension != KIMI_K3_KDA_DIMENSION ||
        kernelSize < 2 || kernelSize > 8 || f.dims.size() != 3 ||
        f.dims[0] != 1 || f.dims[1] != 1 || f.dims != g.dims) return false;
    const int rank = f.dims.back(), channels = heads * dimension;
    if (rank <= 0 || rank > 256 || rank % 8 ||
        q.dims != std::vector<int>({1, 1, channels}) || q.dims != k.dims || q.dims != v.dims ||
        rawBeta.dims != std::vector<int>({1, 1, heads}) ||
        fw.dims != std::vector<int>({channels, rank}) || fw.dims != gw.dims ||
        qw.dims != std::vector<int>({channels, 1, kernelSize}) || qw.dims != kw.dims || qw.dims != vw.dims ||
        convCache.dims != std::vector<int>({3, kernelSize - 1, channels}) ||
        state.dims != std::vector<int>({1, heads, dimension, dimension}) ||
        aLog.Count(0) != (uint64_t)heads || dtBias.Count(0) != (uint64_t)channels ||
        norm.Count(0) != (uint64_t)dimension || q.dataDeviceIds.size() != 1 ||
        q.dataDeviceIds[0] != FastllmCudaGetDevice()) return false;
    auto ready = [&](const Data &data, DataType type) {
        if (data.dataDevice != DataDevice::CUDA || data.dataType != type || !data.cudaData ||
            data.lockInCPU || data.dataDeviceIds != q.dataDeviceIds || data.strides.size() != data.dims.size()) return false;
        uint64_t stride = 1;
        for (int i = (int)data.dims.size()-1; i >= 0; --i) {
            if (data.strides[i] != stride) return false;
            stride *= data.dims[i];
        }
        return true;
    };
    for (const auto *data : std::initializer_list<const Data*>{&q, &k, &v, &f, &g, &rawBeta, &convCache})
        if (!ready(*data, DataType::BFLOAT16)) return false;
    for (const auto *data : std::initializer_list<const Data*>{&qw, &kw, &vw, &aLog, &dtBias, &norm, &state})
        if (!ready(*data, DataType::FLOAT32)) return false;
    for (const auto *data : {&f, &g, &fw, &gw})
        if (reinterpret_cast<uintptr_t>(data->cudaData) % 16) return false;
    for (const auto *weight : {&fw, &gw}) {
        if (ready(*weight, DataType::BFLOAT16)) continue;
        // GGUF BF16 uses the same scalar payload but carries its strides in
        // ggml metadata, rather than the native Data storage descriptor.
        const auto *tensor = static_cast<const ggml_tensor*>(weight->ggmlTensor);
        if (weight->dataType != DataType::DATA_GGUF_FORMAT || weight->ggmlType != GGML_TYPE_BF16 ||
            weight->dataDevice != DataDevice::CUDA || !weight->cudaData || weight->multiDeviceData ||
            weight->lockInCPU || weight->dataDeviceIds != q.dataDeviceIds || !tensor ||
            tensor->type != GGML_TYPE_BF16 || tensor->ne[0] != rank || tensor->ne[1] != channels ||
            tensor->nb[0] != 2 || tensor->nb[1] != (size_t)rank * 2) return false;
    }
    auto allocate = [&](Data &data, DataType type, const std::vector<int> &dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, q.dataDeviceIds, false);
        // Every element is written by the following kernels before it is read.
        // Avoid capturing redundant memset nodes for these temporary buffers.
        data.Allocate(false);
    };
    Data gates, query, key, retention, beta, value, attention, gate;
    allocate(gates, DataType::BFLOAT16, {2, channels});
    for (auto *data : {&query, &key, &retention}) allocate(*data, DataType::FLOAT32, {heads, dimension});
    allocate(beta, DataType::FLOAT32, {heads});
    allocate(value, DataType::BFLOAT16, {1, 1, heads, dimension});
    allocate(attention, DataType::BFLOAT16, value.dims);
    allocate(output, DataType::BFLOAT16, value.dims);
    if (UseFusedPreparation(q.dataDeviceIds[0], heads, rank)) {
        HybridKdaPrepare<<<heads, 1024, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)k.cudaData, (const __nv_bfloat16*)v.cudaData,
            (const __nv_bfloat16*)f.cudaData, (const __nv_bfloat16*)g.cudaData,
            (const __nv_bfloat16*)fw.cudaData, (const __nv_bfloat16*)gw.cudaData,
            (const float*)qw.cudaData, (const float*)kw.cudaData, (const float*)vw.cudaData,
            (__nv_bfloat16*)convCache.cudaData, (__nv_bfloat16*)gates.cudaData,
            (const __nv_bfloat16*)rawBeta.cudaData, (const float*)aLog.cudaData, (const float*)dtBias.cudaData,
            (float*)query.cudaData, (float*)key.cudaData, (float*)retention.cudaData, (float*)beta.cudaData,
            (__nv_bfloat16*)value.cudaData, rank, heads, kernelSize, lowerBound);
    } else {
        HybridKdaLowRank<<<dim3((channels + 7)/8, 2), 256, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)f.cudaData, (const __nv_bfloat16*)g.cudaData,
            (const __nv_bfloat16*)fw.cudaData, (const __nv_bfloat16*)gw.cudaData,
            (__nv_bfloat16*)gates.cudaData, rank, channels);
        HybridKdaConvPrepare<<<heads, 128, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)k.cudaData, (const __nv_bfloat16*)v.cudaData,
            (const float*)qw.cudaData, (const float*)kw.cudaData, (const float*)vw.cudaData,
            (__nv_bfloat16*)convCache.cudaData, (const __nv_bfloat16*)gates.cudaData,
            (const __nv_bfloat16*)rawBeta.cudaData, (const float*)aLog.cudaData, (const float*)dtBias.cudaData,
            (float*)query.cudaData, (float*)key.cudaData, (float*)retention.cudaData, (float*)beta.cudaData,
            (__nv_bfloat16*)value.cudaData, heads, kernelSize, lowerBound);
    }
    KimiK3KdaRegisterScanKernel<<<heads * (dimension / KIMI_K3_KDA_VALUE_TILE), 32, 0, cudaStreamPerThread>>>(
        (const float*)query.cudaData, (const float*)key.cudaData, (const float*)retention.cudaData,
        (const float*)beta.cudaData, (const __nv_bfloat16*)value.cudaData, (float*)state.cudaData,
        (__nv_bfloat16*)attention.cudaData, 1, heads, dimension, nullptr);
    gate.FakeFrom(gates, (size_t)channels * sizeof(__nv_bfloat16));
    gate.Resize(value.dims);
    // Reuse the original output normalization and gate, including its FP32
    // reduction and final BF16 rounding.
    fastllm::AssertInFastLLM(FastllmCudaKimiK3RMSNormSigmoidGate(attention, gate, norm, output, eps),
        "Hybrid KDA output normalization failed.");
    fastllm::AssertInFastLLM(cudaGetLastError() == cudaSuccess, "Hybrid KDA decode failed.");
    return true;
}
#endif
