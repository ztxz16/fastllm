#include "devices/cuda/fastllm-clover.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#include "attention_impl.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"
#endif

// NHD everywhere: contiguous appends preserve the clean prefix; preview only
// appends into spare capacity and leaves the committed length unchanged.
bool FastllmCudaCloverAttention(const void *query, const void *key,
        const void *value, void *output, int tokens, int cachedTokens,
        int heads, int kvHeads, bool preview, void *stream, void *workspace) {
    if (!query || !key || !value || !output || tokens < 1 || cachedTokens < 0 ||
        heads < 1 || kvHeads < 1 || heads % kvHeads) return false;
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    using namespace flashinfer;
    using Params = SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    Params p((__nv_bfloat16 *)query, (__nv_bfloat16 *)key, (__nv_bfloat16 *)value,
        nullptr, (__nv_bfloat16 *)output, nullptr, nullptr,
        heads, kvHeads, tokens, cachedTokens + tokens,
        heads * 256, 256, kvHeads * 256, 256, 256, -1,
        0.f, 1.f / 16.f, 1.f, 10000.f);
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    cudaError_t result;
    if (preview)
        result = SinglePrefillWithKVCacheDispatched<256, 256, PosEncodingMode::kNone,
            false, MaskMode::kNone, DefaultAttention<false, false, false, false>, Params>(p, (__nv_bfloat16 *)workspace, s);
    else
        result = SinglePrefillWithKVCacheDispatched<256, 256, PosEncodingMode::kNone,
            false, MaskMode::kCausal, DefaultAttention<false, false, false, false>, Params>(p, (__nv_bfloat16 *)workspace, s);
    return result == cudaSuccess;
#else
    return false;
#endif
}
