#include "devices/cuda/fastllm-clover.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <math_constants.h>

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750)
#include "attention_impl.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"
#endif

namespace {
// NHD, bounded scratch and FP32 online softmax. This path needs neither
// Tensor Cores nor a quadratic attention matrix on pre-Turing GPUs.
__global__ void CloverHalfAttentionFallback(const __half *q, const __half *k,
        const __half *v, __half *out, int tokens, int cached, int heads, int kvHeads, bool preview) {
    constexpr int Tile = 32;
    int query = blockIdx.x, head = blockIdx.y, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int keyHead = head / (heads / kvHeads), length = cached + (preview ? tokens : query + 1);
    __shared__ float probabilities[Tile], scale, denominator;
    float queryValues[8];
    for (int i = 0; i < 8; ++i) queryValues[i] = __half2float(q[((size_t)query * heads + head) * 256 + lane + i * 32]);
    float maximum = -CUDART_INF_F, sum = 0.f, result[2] = {};
    for (int start = 0; start < length; start += Tile) {
        for (int row = warp; row < Tile; row += 4) {
            int key = start + row;
            float dot = 0.f;
            if (key < length) for (int i = 0; i < 8; ++i)
                dot += queryValues[i] * __half2float(k[((size_t)key * kvHeads + keyHead) * 256 + lane + i * 32]);
            for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
            if (lane == 0) probabilities[row] = key < length ? dot / 16.f : -CUDART_INF_F;
        }
        __syncthreads();
        if (tid == 0) {
            float next = maximum;
            for (int row = 0; row < Tile; ++row) next = fmaxf(next, probabilities[row]);
            scale = expf(maximum - next);
            sum *= scale;
            for (int row = 0; row < Tile; ++row) {
                probabilities[row] = expf(probabilities[row] - next);
                sum += probabilities[row];
            }
            maximum = next; denominator = sum;
        }
        __syncthreads();
        for (int i = 0; i < 2; ++i) {
            result[i] *= scale;
            for (int row = 0; row < Tile && start + row < length; ++row)
                result[i] += probabilities[row] * __half2float(v[((size_t)(start + row) * kvHeads + keyHead) * 256 + tid + i * 128]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        out[((size_t)query * heads + head) * 256 + tid + i * 128] = __float2half_rn(result[i] / denominator);
}

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750)
template<typename T>
bool CloverFlashAttention(const void *query, const void *key, const void *value, void *output,
        int tokens, int cachedTokens, int heads, int kvHeads, bool preview, cudaStream_t stream, void *workspace) {
    using namespace flashinfer;
    using Params = SinglePrefillParams<T, T, T>;
    Params p((T *)query, (T *)key, (T *)value, nullptr, (T *)output, nullptr, nullptr,
        heads, kvHeads, tokens, cachedTokens + tokens,
        heads * 256, 256, kvHeads * 256, 256, 256, -1, 0.f, 1.f / 16.f, 1.f, 10000.f);
    cudaError_t result;
    if (preview)
        result = SinglePrefillWithKVCacheDispatched<256, 256, PosEncodingMode::kNone,
            false, MaskMode::kNone, DefaultAttention<false, false, false, false>, Params>(p, (T *)workspace, stream);
    else
        result = SinglePrefillWithKVCacheDispatched<256, 256, PosEncodingMode::kNone,
            false, MaskMode::kCausal, DefaultAttention<false, false, false, false>, Params>(p, (T *)workspace, stream);
    return result == cudaSuccess;
}
#endif
}

bool FastllmCudaCloverAttention(const void *query, const void *key,
        const void *value, void *output, int tokens, int cachedTokens,
        int heads, int kvHeads, bool preview, void *stream, void *workspace, bool fp16) {
    if (!query || !key || !value || !output || tokens < 1 || cachedTokens < 0 ||
        heads < 1 || kvHeads < 1 || heads % kvHeads) return false;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    if (fp16) {
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750)
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, CloverHalfAttentionFallback) != cudaSuccess) return false;
        if (attr.binaryVersion >= 75 && attr.ptxVersion >= 75)
            return CloverFlashAttention<__half>(query, key, value, output, tokens, cachedTokens, heads, kvHeads, preview, s, workspace);
#endif
        CloverHalfAttentionFallback<<<dim3(tokens, heads), 128, 0, s>>>((const __half *)query,
            (const __half *)key, (const __half *)value, (__half *)output, tokens, cachedTokens, heads, kvHeads, preview);
        return cudaGetLastError() == cudaSuccess;
    }
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    return CloverFlashAttention<__nv_bfloat16>(query, key, value, output, tokens, cachedTokens, heads, kvHeads, preview, s, workspace);
#else
    return false;
#endif
}
