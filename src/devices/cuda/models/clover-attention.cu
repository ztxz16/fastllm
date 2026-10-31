#include "devices/cuda/fastllm-clover.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <math_constants.h>

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750)
#include "attention_impl.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"
#endif

namespace {
bool BatchAttentionGeometry(int batch, int tokens, int heads, int kvHeads,
        const CloverCudaAttentionCaches &caches, int &maximum) {
    if (batch < 1 || batch > CloverCudaAttentionCaches::Capacity || tokens < 1 || tokens > 32 ||
        heads < 1 || kvHeads < 1 || heads % kvHeads) return false;
    maximum = 0;
    for (int b = 0; b < batch; ++b) {
        if (!caches.key[b] || !caches.value[b] || caches.position[b] < 0 ||
            caches.position[b] > INT_MAX - tokens - 255) return false;
        maximum = std::max(maximum, caches.position[b] + tokens);
    }
    return true;
}
// Keep the generic serial implementation on other architectures. Hopper's
// established CTA64/KV64 geometry can be combined without changing either
// the split-KV boundaries or the softmax reduction tree.
bool BatchAttentionHopper(int tokens, int heads, int kvHeads, int &sms) {
    int device, major, minor;
    return tokens * (heads / kvHeads) > 16 &&
        cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
        major == 9 && minor == 0 &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess;
}
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
struct CloverAttentionPlan {
    CloverCudaAttentionCaches caches;
    int chunks[CloverCudaAttentionCaches::Capacity] = {};
};

template<typename Traits, typename Params>
__global__ void CloverBatchAttention(Params p, CloverAttentionPlan plan,
        typename Params::DTypeO *output, int chunkStride) {
    using T = typename Params::DTypeO;
    int request = blockIdx.z / p.num_kv_heads, head = blockIdx.z % p.num_kv_heads;
    int chunks = plan.chunks[request];
    if (blockIdx.y >= chunks) return;
    size_t rows = size_t(p.qo_len) * p.num_qo_heads;
    p.q += request * rows * 256;
    p.k = (T *)plan.caches.key[request]; p.v = (T *)plan.caches.value[request];
    p.kv_len = plan.caches.position[request] + p.qo_len;
    p.partition_kv = chunks > 1;
    p.o = chunks > 1 ? p.o + request * chunkStride * rows * 256 : output + request * rows * 256;
    p.lse = chunks > 1 ? p.lse + request * chunkStride * rows : nullptr;
    extern __shared__ uint8_t storage[];
    auto &smem = *reinterpret_cast<typename Traits::SharedStorageSingle *>(storage);
    flashinfer::SinglePrefillWithKVCacheDevice<Traits>(p, smem, threadIdx, blockIdx.x,
        blockIdx.y, head, chunks, p.num_kv_heads);
}

// Reuse FlashInfer's FP32 state merge, including the BF16/FP16 intermediate
// rounding of its four-warp large-index-set path. Only request addressing and
// loading are batched; each request retains the single-prefill arithmetic.
template<typename T>
__global__ void CloverBatchAttentionMerge(const T *partial, const float *lse, T *output,
        CloverAttentionPlan plan, int tokens, int heads, int chunkStride) {
    using namespace flashinfer;
    int request = blockIdx.z, token = blockIdx.x, head = blockIdx.y;
    int chunks = plan.chunks[request], tx = threadIdx.x, ty = threadIdx.y;
    if (chunks <= 1) return;
    size_t rows = size_t(tokens) * heads;
    partial += request * chunkStride * rows * 256;
    lse += request * chunkStride * rows;
    state_t<8> st;
    bool parallel = chunks >= tokens;
    if (!parallel && ty != 0) return;
    for (int c = parallel ? ty : 0; c < chunks; c += parallel ? 4 : 1) {
        vec_t<float, 8> v;
        size_t index = (size_t(token) * chunks + c) * heads + head;
        v.cast_load(partial + index * 256 + tx * 8);
        st.merge(v, lse[index], 1);
    }
    if (parallel) {
        __shared__ T values[4 * 256];
        __shared__ float scores[4];
        st.normalize();
        threadblock_sync_state<32, 4, 8>(st, values, scores);
    }
    st.normalize();
    if (ty == 0) st.o.cast_store(output + (request * rows + token * heads + head) * 256 + tx * 8);
}

template<typename T, flashinfer::MaskMode Mask>
bool CloverFlashAttentionBatch(const void *query, void *output, int batch, int tokens,
        int heads, int kvHeads, const CloverCudaAttentionCaches &caches,
        void *workspace, cudaStream_t stream, int sms) {
    using namespace flashinfer;
    using Params = SinglePrefillParams<T, T, T>;
    using Traits = KernelTraits<Mask, 64, 1, 4, 16, 16, 4, 1, PosEncodingMode::kNone,
        T, T, T, float, int32_t, DefaultAttention<false, false, false, false>>;
    constexpr size_t shared = sizeof(typename Traits::SharedStorageSingle);
    auto kernel = CloverBatchAttention<Traits, Params>;
    auto single = SinglePrefillWithKVCacheKernel<Traits, Params>;
    int blocks = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess ||
        cudaFuncSetAttribute(single, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, single, 128, shared) != cudaSuccess) return false;
    int tiles = (tokens * (heads / kvHeads) + 63) / 64;
    int limit = std::max(1, blocks * sms / (kvHeads * tiles));
    CloverAttentionPlan plan; plan.caches = caches;
    int maximum = 1;
    for (int b = 0; b < batch; ++b) {
        int length = caches.position[b] + tokens;
        int chunkSize = std::max(256, (length + limit - 1) / limit);
        plan.chunks[b] = (length + chunkSize - 1) / chunkSize;
        maximum = std::max(maximum, plan.chunks[b]);
    }
    Params p((T *)query, nullptr, nullptr, nullptr, (T *)workspace,
        (float *)((T *)workspace + size_t(batch) * maximum * tokens * heads * 256), nullptr,
        heads, kvHeads, tokens, tokens, heads * 256, 256, kvHeads * 256, 256, 256,
        -1, 0.f, 1.f / 16.f, 1.f, 10000.f);
    kernel<<<dim3(tiles, maximum, batch * kvHeads), dim3(32, 4), shared, stream>>>(p, plan, (T *)output, maximum);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (maximum > 1)
        CloverBatchAttentionMerge<<<dim3(tokens, heads, batch), dim3(32, 4), 0, stream>>>(
            p.o, p.lse, (T *)output, plan, tokens, heads, maximum);
    return cudaGetLastError() == cudaSuccess;
}

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

std::size_t FastllmCudaCloverAttentionBatchWorkspaceBytes(int batch, int tokens,
        int heads, int kvHeads, const CloverCudaAttentionCaches &caches) {
    int maximum = 0, sms = 0;
    if (!BatchAttentionGeometry(batch, tokens, heads, kvHeads, caches, maximum)) return 0;
    int chunks = (maximum + 255) / 256, copies = 1;
    if (BatchAttentionHopper(tokens, heads, kvHeads, sms)) {
        int tiles = (tokens * (heads / kvHeads) + 63) / 64;
        // CTA64/KV64 consumes 96 KiB, at most two resident CTAs per SM90 SM.
        chunks = std::min(chunks, std::max(1, 2 * sms / (kvHeads * tiles)));
        copies = batch;
    }
    return size_t(copies) * chunks * tokens * heads * (256 * sizeof(uint16_t) + sizeof(float));
}

bool FastllmCudaCloverAttentionBatch(const void *query, void *output, int batch,
        int tokens, int heads, int kvHeads, const CloverCudaAttentionCaches &caches,
        bool preview, void *workspace, void *stream, bool fp16) {
    int maximum = 0, sms = 0;
    if (!query || !output || !workspace ||
        !BatchAttentionGeometry(batch, tokens, heads, kvHeads, caches, maximum)) return false;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    if (BatchAttentionHopper(tokens, heads, kvHeads, sms)) {
        using flashinfer::MaskMode;
        if (fp16) return preview
            ? CloverFlashAttentionBatch<__half, MaskMode::kNone>(query, output, batch, tokens, heads, kvHeads, caches, workspace, s, sms)
            : CloverFlashAttentionBatch<__half, MaskMode::kCausal>(query, output, batch, tokens, heads, kvHeads, caches, workspace, s, sms);
        return preview
            ? CloverFlashAttentionBatch<__nv_bfloat16, MaskMode::kNone>(query, output, batch, tokens, heads, kvHeads, caches, workspace, s, sms)
            : CloverFlashAttentionBatch<__nv_bfloat16, MaskMode::kCausal>(query, output, batch, tokens, heads, kvHeads, caches, workspace, s, sms);
    }
#endif
    size_t stride = size_t(tokens) * heads * 256;
    for (int b = 0; b < batch; ++b)
        if (!FastllmCudaCloverAttention((const uint16_t *)query + b * stride, caches.key[b],
            caches.value[b], (uint16_t *)output + b * stride, tokens, caches.position[b],
            heads, kvHeads, preview, stream, workspace, fp16)) return false;
    return true;
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
