// Resident V4.1 GGUF experts: one launch per projection for all GPU routes.
#include "../fastllm-gguf-mmvq-common.cuh"
#include "../fastllm-gguf-q3-k-gemv.cuh"
#include "../fastllm-gguf-iq1-dot.cuh"
#include "fastllm-moe-gguf-q8.cuh"
#include "fastllm-cuda-v41-gguf.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>

namespace {
// The same FP8 block-32 -> BF16 -> Q8_1 boundaries as the ordinary CUDA
// Linear path. Input conversion is shared by every selected expert.
template<bool Fp8Boundary>
__global__ void V41GGUFQuantize(const __nv_bfloat16 *input, block_q8_1 *output, int columns) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= columns) return;
    float x = __bfloat162float(input[size_t(blockIdx.y)*columns+c]);
    if constexpr (Fp8Boundary) {
        const float maximum = fmaxf(warp_reduce_max(fabsf(x)), 1e-4f);
        const unsigned bits = __float_as_uint(maximum * (1.0f/448.0f));
        const int exponent = int((bits>>23)&255)-127+int((bits&0x7fffff)!=0);
        const float scale = exp2f(float(exponent));
        x = __bfloat162float(__float2bfloat16_rn(
            float(__nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x/scale))))*scale));
    }
    const float maximum = warp_reduce_max(fabsf(x));
    const float sum = warp_reduce_sum(x), scale = maximum/127.f;
    auto &q = output[size_t(blockIdx.y)*(columns/32)+c/32];
    q.qs[c%32] = maximum == 0 ? 0 : int8_t(roundf(x/scale));
    if (c%32 == 0) q.ds = __floats2half2_rn(scale, sum);
}

template<ggml_type Type>
__device__ __forceinline__ float V41GGUFDot(const void *weight, const block_q8_1 *x, int columns, const uint32_t *iqGrid) {
    if constexpr (Type == GGML_TYPE_Q2_0) {
        return gguf_cache_q8::RowDot<Type>(weight, x, columns, nullptr);
    } else {
        constexpr int qi = [] {
            if constexpr (Type == GGML_TYPE_IQ1_S) return QI1_S;
            else if constexpr (Type == GGML_TYPE_IQ1_M) return QI1_M;
            else return ggml_cuda_type_traits<Type>::qi;
        }();
        constexpr int vdr = get_vdr_mmvq(Type);
        constexpr int step = vdr*128/qi;
        const int lane = threadIdx.x%32;
        float partial[4];
#pragma unroll
        for (int warp = 0; warp < 4; ++warp) {
            const int tid = lane+32*warp;
            float value = 0;
            for (int b = tid/(qi/vdr); b < columns/256; b += step) {
                if constexpr (Type == GGML_TYPE_IQ1_S)
                    value += vec_dot_iq1_s_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_IQ1_M)
                    value += vec_dot_iq1_m_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_Q3_K)
                    value += fastllm_gguf_q3_k::Dot(weight, x+b*8, b, vdr*(tid%(qi/vdr)));
                else value += get_vec_dot_q_cuda(Type)(weight, x+b*8, b, vdr*(tid%(qi/vdr)));
            }
            partial[warp] = value;
        }
        // Preserve the legacy four-warp addition order before the lane sum.
        float sum = partial[0];
#pragma unroll
        for (int warp = 1; warp < 4; ++warp) sum += partial[warp];
        return warp_reduce_sum(sum);
    }
}

__device__ const uint8_t *V41Weight(const uint8_t *const *weights,
        const int32_t *indices, int route, int experts, bool gate) {
    const int expert = indices[route];
    return expert >= 0 && expert < experts ? weights[2*expert+(gate ? 0 : 1)] : nullptr;
}
__device__ const uint8_t *V41Weight(const FastllmCudaMoeGGUFCacheView &view,
        const int32_t *, int route, int, bool gate) {
    const int slot = view.routeSlots[route];
    if (slot < 0) return nullptr;
    const size_t offset = view.slotOffsets ? view.slotOffsets[slot] : size_t(slot)*view.recordStride;
    return view.records + offset + (gate ? 0 : view.downOffset);
}

template<ggml_type Type, bool Gate, class Weights>
__global__ void V41GGUFProjection(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int columns, int rows, size_t rowBytes) {
    const int route = blockIdx.y;
    const auto *x = input + size_t(Gate ? route/topk : route)*(columns/32);
    extern __shared__ uint32_t activation[];
    // Stage the canonical codebook in the packed layout used by extended GEMV.
    // No host-side lazy initialization or synchronization is needed in capture.
    __shared__ uint32_t iqGrid[(Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) ? 2048 : 1];
    if constexpr (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
            const uint64_t values = iq1s_grid[i];
            uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                packed |= uint32_t(int(int8_t(values >> (8*j))) + 1) << (8*(j%4) + 4*(j/4));
            iqGrid[i] = packed;
        }
    }
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    const int row = blockIdx.x*8 + threadIdx.x/32;
    if (row >= rows) return;
    const uint8_t *weight = V41Weight(weights, indices, route, experts, Gate);
    float value = 0;
    if (weight)
        value = V41GGUFDot<Type>(weight+size_t(row)*rowBytes,
            reinterpret_cast<const block_q8_1 *>(activation), columns, iqGrid);
    if (threadIdx.x%32 == 0) output[size_t(route)*rows+row] = __float2bfloat16_rn(value);
}

__global__ void V41GGUFReduce(const __nv_bfloat16 *partial, float *output,
        const int32_t *indices, int hidden, int topk) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    const int base = blockIdx.y*topk;
    int order[32];
    for (int k = 0; k < topk; ++k) {
        int pos = k;
        while (pos && indices[base+order[pos-1]] > indices[base+k]) {
            order[pos] = order[pos-1]; --pos;
        }
        order[pos] = k;
    }
    float sum = 0;
    for (int k = 0; k < topk; ++k)
        sum = __fadd_rn(sum, __bfloat162float(partial[size_t(base+order[k])*hidden+c]));
    // The caller adds the shared expert in FP32 before the BF16 HC boundary.
    output[size_t(blockIdx.y)*hidden+c] = sum;
}

template<bool Gate, class Weights>
void Project(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int routes, int columns, int rows) {
    const dim3 grid((rows+7)/8, routes);
    const size_t shared = size_t(columns/32)*sizeof(block_q8_1);
    switch (type) {
#define PROJECT(Type) case GGML_TYPE_##Type: \
        V41GGUFProjection<GGML_TYPE_##Type, Gate, Weights><<<grid, 256, shared, cudaStreamPerThread>>>( \
            input, output, weights, indices, experts, topk, columns, rows, \
            ggml_row_size(GGML_TYPE_##Type, columns)); break;
        PROJECT(Q2_0) PROJECT(Q2_K) PROJECT(Q3_K) PROJECT(Q4_K) PROJECT(IQ1_S) PROJECT(IQ1_M)
#undef PROJECT
    }
}
}

bool FastllmCudaRunV41GGUF(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
        fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
        const int32_t *indices, const float *scores, int gateType, int downType,
        int hidden, int inter, int experts, int topk, float swigluLimit) {
    const int rows = input.dims[0], routes = rows*topk;
    auto *qInput = static_cast<block_q8_1 *>(workspace.cudaData);
    auto *qGate = qInput + size_t(rows)*(hidden/32);
    auto *partial = reinterpret_cast<__nv_bfloat16 *>(qGate+size_t(routes)*(inter/32));
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    const size_t baseBytes = (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*hidden*sizeof(__nv_bfloat16);
    const size_t offset = (baseBytes+255)&~size_t(255);
    const bool grouped = workspace.GetBytes() > offset;
    void *groupedScratch = static_cast<char *>(workspace.cudaData)+offset;
    V41GGUFQuantize<true><<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, hidden);
    if (grouped) {
        if (!FastllmCudaV41GGUFGroupedProjection(true, gateType, qInput, gate.cudaData,
            weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
    } else Project<true>(gateType, qInput, static_cast<__nv_bfloat16 *>(gate.cudaData),
        weights, indices, experts, topk, routes, hidden, 2*inter);
    if (!FastllmCudaDeepSeekV41SharedSwigluQuantized(gate, swigluLimit, activation, scores)) return false;
    V41GGUFQuantize<false><<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, inter);
    if (grouped) {
        if (!FastllmCudaV41GGUFGroupedProjection(false, downType, qGate, partial,
            weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
    } else Project<false>(downType, qGate, partial, weights, indices, experts, topk, routes, inter, hidden);
    V41GGUFReduce<<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        partial, static_cast<float *>(output.cudaData), indices, hidden, topk);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
size_t V41CacheWorkspaceBytes(int rows, int routes, int hidden, int inter) {
    return (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*(2*size_t(inter)+hidden)*sizeof(__nv_bfloat16);
}

__global__ void V41CachePartials(const __nv_bfloat16 *partial, float *output, int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) output[i] = __bfloat162float(partial[i]);
}
}

bool FastllmCudaV41GGUFSupported(int gateType, int downType) {
    const auto supported = [](int type) {
        return type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q2_K ||
            type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
            type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M;
    };
    return supported(gateType) && supported(downType);
}
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter) {
    if (hidden <= 0 || inter <= 0 || hidden%256 || inter%256) return 0;
    // Decode and DSpark verification, at most eight rows and sixteen routes/row.
    return V41CacheWorkspaceBytes(8, 8*16, hidden, inter);
}
bool FastllmCudaMoeV41CanonicalGGUFCacheCompute(
        const fastllm::Data &input, fastllm::Data &activation,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores,
        int topk, float swigluLimit, float *perExpert) {
    if (input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::CUDA ||
        !input.cudaData || input.dims.size() != 2 || input.dims[0] < 1 || input.dims[0] > 8 ||
        input.dims[1] != view.hidden || view.hidden <= 0 || view.inter <= 0 ||
        view.hidden%256 || view.inter%256 || topk < 1 || topk > 16 ||
        !FastllmCudaV41GGUFSupported(view.gateType, view.downType) ||
        !view.records || !view.routeSlots || !view.workspace || !scores || !perExpert ||
        view.routeMap || view.activeRouteCount || view.numaGateType >= 0 || view.numaDownType >= 0)
        return false;
    const int rows = input.dims[0], routes = rows*topk, hidden = view.hidden, inter = view.inter;
    const size_t bytes = V41CacheWorkspaceBytes(rows, routes, hidden, inter);
    if (view.workspaceBytes < bytes) return false;
    auto *qInput = static_cast<block_q8_1 *>(view.workspace);
    auto *qGate = qInput + size_t(rows)*(hidden/32);
    auto *gateData = reinterpret_cast<__nv_bfloat16 *>(qGate + size_t(routes)*(inter/32));
    auto *partial = gateData + size_t(routes)*2*inter;
    fastllm::Data gate(fastllm::BFLOAT16, {routes, 2*inter}, fastllm::CUDA, gateData);
    gate.dataDeviceIds = input.dataDeviceIds;
    V41GGUFQuantize<true><<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, hidden);
    Project<true>(view.gateType, qInput, gateData, view, nullptr, 0, topk, routes, hidden, 2*inter);
    if (!FastllmCudaDeepSeekV41SharedSwigluQuantized(gate, swigluLimit, activation, scores)) return false;
    V41GGUFQuantize<false><<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, inter);
    Project<false>(view.downType, qGate, partial, view, nullptr, 0, topk, routes, inter, hidden);
    V41CachePartials<<<(routes*hidden+255)/256, 256, 0, cudaStreamPerThread>>>(
        partial, perExpert, routes*hidden);
    return cudaGetLastError() == cudaSuccess;
}
