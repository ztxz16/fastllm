// C++ adapter for FlashInfer's DSV4.1 BF16-compute mixed FP8/FP4 attention.
// Third-party kernel provenance and the pointer-indirection patch are recorded
// in third_party/flashinfer_dsv41/README.md. No Python runtime is required.
#include "deepseekv41-flashinfer.cuh"
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_bf16/decode.cuh>
#include <cmath>
#include <memory>

namespace {
using namespace flashinfer::sparse_mla_sm120;
using namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16;
struct Plan {
    int width = 0, splits = 0, ratio = 0, heads = 0, batch = 0, scratchHeads = 0;
    float scale = 1.f;
    void *storage = nullptr;
    bf16 *mid = nullptr;
    float *midLse = nullptr;
    ~Plan() {
        if (storage)
            cudaFree(storage);
    }
};

// Merge old-KV splits, sink, and the unquantized current token in one pass.
// Only the final result is rounded to BF16; QK and the weighted sum use FP32.
__global__ void MergeCurrent(const bf16 *q, const bf16 *current, const bf16 *partial, const float *lse,
                             const float *sink, bf16 *output, float scale, int heads, int scratchHeads,
                             int splits) {
    const int h = blockIdx.x, t = threadIdx.x, token = blockIdx.y;
    q += (size_t(token) * heads + h) * 512;
    current += size_t(token) * 512;
    partial += (size_t(token) * scratchHeads + h) * splits * 512;
    lse += (size_t(token) * scratchHeads + h) * splits;
    output += (size_t(token) * heads + h) * 512;
    float dot = 0.f;
#pragma unroll
    for (int d = 0; d < 4; ++d)
        dot = fmaf(float(q[t * 4 + d]), float(current[t * 4 + d]), dot);
    for (int s = 16; s; s >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, s);
    __shared__ float sums[4], weights[34], currentWeight;
    if ((t & 31) == 0)
        sums[t / 32] = dot;
    __syncthreads();
    if (t < 32) {
        const float score = (sums[0] + sums[1] + sums[2] + sums[3]) * scale * 1.4426950408889634f;
        const float sinkScore = sink[h] * 1.4426950408889634f;
        float maximum = fmaxf(score, sinkScore);
        for (int i = t; i < splits; i += 32)
            maximum = fmaxf(maximum, lse[i]);
        for (int s = 16; s; s >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
        float denominator = 0.f;
        for (int i = t; i < splits; i += 32) {
            float w = lse[i] > -1e29f ? exp2f(lse[i] - maximum) : 0.f;
            weights[i] = w;
            denominator += w;
        }
        for (int s = 16; s; s >>= 1)
            denominator += __shfl_xor_sync(0xffffffff, denominator, s);
        const float cw = exp2f(score - maximum);
        const float inv = 1.f / (denominator + cw + exp2f(sinkScore - maximum));
        for (int i = t; i < splits; i += 32)
            weights[i] *= inv;
        if (!t)
            currentWeight = cw * inv;
    }
    __syncthreads();
    float acc[4];
#pragma unroll
    for (int d = 0; d < 4; ++d)
        acc[d] = currentWeight * float(current[t * 4 + d]);
    for (int i = 0; i < splits; ++i) {
        if (weights[i] == 0.f)
            continue;
        const uint2 packed = *reinterpret_cast<const uint2 *>(partial + size_t(i) * 512 + t * 4);
        const auto *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            const float2 v = __bfloat1622float2(pairs[d]);
            acc[d * 2] += weights[i] * v.x;
            acc[d * 2 + 1] += weights[i] * v.y;
        }
    }
    uint2 result;
    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
    pairs[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    pairs[1] = __floats2bfloat162_rn(acc[2], acc[3]);
    *reinterpret_cast<uint2 *>(output + t * 4) = result;
}
} // namespace

void *FastllmV41FlashInferCreate(const FastllmV41FullDecodeConfig &c) {
    // Packed full-decode is currently the single-GPU hybrid path. Verification,
    // BF16/FP8 compressed cache and other windows keep native math.
    if (c.verifying || c.batch < 1 || c.batch > 128 || c.heads < 1 || c.heads > 128 || c.window != 128 ||
        c.ringMode != 1 || c.ringRowBytes != 528 || c.cmpMode != 3 || c.cmpRowBytes != 288 || c.ratio < 0 ||
        c.ratio > 2 || c.topK < 1 || c.topK > 1024 || !std::isfinite(c.softmaxScale) || c.softmaxScale <= 0.f)
        return nullptr;
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        major != 12 || minor < 0 || minor > 1)
        return nullptr;
    auto kernel = c.heads == 64 ? (c.ratio ? sparse_mla_decode_dsv41_bf16_kernel<64, true, true>
                                           : sparse_mla_decode_dsv41_bf16_kernel<64, false, true>)
                                : (c.ratio ? sparse_mla_decode_dsv41_bf16_kernel<0, true, true>
                                           : sparse_mla_decode_dsv41_bf16_kernel<0, false, true>);
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Dsv41Bf16Smem)) !=
        cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    auto p = std::make_unique<Plan>();
    p->width = c.ratio ? c.topK : 0;
    p->splits = 2 + (p->width + 63) / 64;
    p->ratio = c.ratio;
    p->scale = c.softmaxScale;
    p->heads = c.heads;
    p->batch = c.batch;
    p->scratchHeads = (c.heads + 15) / 16 * 16;
    // cudaMalloc aligns the BF16 split tensor; each split occupies 1024 bytes.
    const size_t midCount = size_t(c.batch) * p->scratchHeads * p->splits;
    const size_t bytes = midCount * (512 * sizeof(bf16) + sizeof(float));
    if (cudaMalloc(&p->storage, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    p->mid = static_cast<bf16 *>(p->storage);
    p->midLse = reinterpret_cast<float *>(p->mid + midCount * 512);
    return p.release();
}

void FastllmV41FlashInferDestroy(void *plan) {
    delete static_cast<Plan *>(plan);
}

cudaError_t FastllmV41FlashInferRun(void *handle, const FastllmV41FullDecodeBuffers &b,
                                    const FastllmV41FullDecodeRequest *request, cudaStream_t stream) {
    auto *p = static_cast<Plan *>(handle);
    if (!p || !request || !b.q || !b.kv || !b.output || !b.sink || (p->width && !b.topk))
        return cudaErrorInvalidValue;
    auto kernel = p->heads == 64 ? (p->width ? sparse_mla_decode_dsv41_bf16_kernel<64, true, true>
                                             : sparse_mla_decode_dsv41_bf16_kernel<64, false, true>)
                                 : (p->width ? sparse_mla_decode_dsv41_bf16_kernel<0, true, true>
                                             : sparse_mla_decode_dsv41_bf16_kernel<0, false, true>);
    kernel<<<dim3(p->batch, p->scratchHeads / 16, p->splits), 256, sizeof(Dsv41Bf16Smem), stream>>>(
        static_cast<const bf16 *>(b.q), nullptr, nullptr, p->mid, p->midLse, nullptr, nullptr,
        p->width ? static_cast<const int32_t *>(b.topk) : nullptr, nullptr, p->width, 1, 288, p->batch,
        p->heads, 128, p->splits, 1, p->scale, 528, 128, p->width, Dsv41Pages{1, false}, &request->windowKV,
        p->width ? &request->compressedKV : nullptr, sizeof(*request), &request->position, p->ratio);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    MergeCurrent<<<dim3(p->heads, p->batch), 128, 0, stream>>>(
        static_cast<const bf16 *>(b.q), static_cast<const bf16 *>(b.kv), p->mid, p->midLse,
        static_cast<const float *>(b.sink), static_cast<bf16 *>(b.output), p->scale, p->heads,
        p->scratchHeads, p->splits);
    return cudaGetLastError();
}
