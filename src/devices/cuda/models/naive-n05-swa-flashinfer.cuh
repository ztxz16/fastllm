#pragma once
#include "attention/default_prefill_params.cuh"
#include "attention/prefill.cuh"
#include "attention/variants.cuh"

namespace naive_swa_flashinfer {
using BF16 = __nv_bfloat16;
struct Params : flashinfer::SinglePrefillParams<BF16, BF16, BF16> {
    using Base = flashinfer::SinglePrefillParams<BF16, BF16, BF16>;
    using Base::Base;
    const float *sink = nullptr;
};
struct Attention : flashinfer::DefaultAttention<false, true, false, false> {
    using Base = flashinfer::DefaultAttention<false, true, false, false>;
    using Base::Base;
    REGISTER_OUTPUT_TRANSFORM(p, output, batch, query, head, m, d, scale, {
        if (!p.sink) return output * flashinfer::math::ptx_rcp(d);
        // m is in log2 units. Rescale both terms to avoid overflow for a
        // large positive sink. Do not mutate m/d: this hook runs per column.
        float bias = p.sink[head] * flashinfer::math::log2e;
        float maximum = fmaxf(float(m), bias);
        float weight = exp2f(float(m) - maximum);
        float denominator = d * weight + exp2f(bias - maximum);
        return output * weight * flashinfer::math::ptx_rcp(denominator);
    })
};
inline cudaError_t Run(const BF16 *q, const BF16 *k, const BF16 *v, const float *sink,
                       BF16 *out, int queries, int keys, int heads, int kvHeads,
                       int keyStride, cudaStream_t stream = cudaStreamPerThread) {
    Params p(const_cast<BF16 *>(q), const_cast<BF16 *>(k), const_cast<BF16 *>(v), nullptr,
             out, nullptr, nullptr, heads, kvHeads, queries, keys,
             heads * 192, 192, keyStride, 192, 192, 127, 0.0f, 1.0f / sqrtf(192.0f), 1.0f, 10000.0f);
    p.v_stride_n = kvHeads * 128;
    p.v_stride_h = 128;
    p.sink = sink;
    // A null scratch pointer disables split-KV so the sink is counted once.
    return flashinfer::SinglePrefillWithKVCacheDispatched<192, 128,
        flashinfer::PosEncodingMode::kNone, false, flashinfer::MaskMode::kCausal, Attention>(p, nullptr, stream);
}
}
