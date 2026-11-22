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
// A 32-key tile limits wasted work at both edges of the 128-key window.
// Use smaller query tiles until there are enough independent CTAs; retain
// the general dispatcher for other architectures and head group sizes.
template<int QueryTile>
using WindowTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal,
    QueryTile, flashinfer::get_num_mma_q(QueryTile), 2, 12, 8, 4, 1,
    flashinfer::PosEncodingMode::kNone, BF16, BF16, BF16, float, int, Attention, false>;

template<int QueryTile>
inline cudaError_t RunWindow(Params p, cudaStream_t stream) {
    using Traits = WindowTraits<QueryTile>;
    auto kernel = flashinfer::SinglePrefillWithKVCacheKernel<Traits, Params>;
    constexpr size_t shared = sizeof(typename Traits::SharedStorageSingle);
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    p.partition_kv = false;
    void *args[] = {&p};
    dim3 blocks((p.qo_len * (p.num_qo_heads / p.num_kv_heads) + QueryTile - 1) / QueryTile,
                1, p.num_kv_heads);
    return cudaLaunchKernel((void *)kernel, blocks, dim3(32, 4, 1), args, shared, stream);
}
// Requests already own disjoint staged windows. Index the request's query tiles
// directly so different lengths can share one launch without padding or split-KV.
template<int QueryTile, int Limit>
__global__ void BatchWindow(Params common, const
#if CUDART_VERSION >= 12010
        __grid_constant__
#endif
        naive_prefill_mma::StagedSwaBatch<Limit> batch) {
    int i = 0;
    while (int(blockIdx.x) >= batch.entries[i].endTile) ++i;
    const auto &entry = batch.entries[i];
    const int tile = int(blockIdx.x) - (i ? batch.entries[i - 1].endTile : 0);
    common.q = (BF16 *)entry.query; common.k = (BF16 *)entry.key;
    common.v = (BF16 *)entry.value; common.o = (BF16 *)entry.output;
    common.qo_len = entry.rows; common.kv_len = entry.keys;
    common.k_stride_n = entry.stride;
    using Traits = WindowTraits<QueryTile>;
    extern __shared__ __align__(16) unsigned char memory[];
    flashinfer::SinglePrefillWithKVCacheDevice<Traits>(common,
        *reinterpret_cast<typename Traits::SharedStorageSingle *>(memory),
        threadIdx, tile, 0, blockIdx.y, 1, gridDim.y);
}

template<int QueryTile, int Limit>
inline cudaError_t RunBatch(naive_prefill_mma::StagedSwaBatch<Limit> &batch, int n,
        const float *sink, int heads, int kvHeads) {
    Params p(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
             heads, kvHeads, 0, 0, heads * 192, 192, 0, 192, 192, 127, 0,
             1.f / sqrtf(192.f), 1.f, 10000.f);
    p.v_stride_n = kvHeads * 128; p.v_stride_h = 128; p.sink = sink;
    int tiles = 0;
    for (int i = 0; i < n; ++i) {
        tiles += (batch.entries[i].rows * (heads / kvHeads) + QueryTile - 1) / QueryTile;
        batch.entries[i].endTile = tiles;
    }
    auto kernel = BatchWindow<QueryTile, Limit>;
    constexpr size_t shared = sizeof(typename WindowTraits<QueryTile>::SharedStorageSingle);
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    void *args[] = {&p, &batch};
    return cudaLaunchKernel((void *)kernel, dim3(tiles, kvHeads), dim3(32,4), args, shared, cudaStreamPerThread);
}

inline cudaError_t Run(const BF16 *q, const BF16 *k, const BF16 *v, const float *sink,
                       BF16 *out, int queries, int keys, int heads, int kvHeads,
                       int keyStride, cudaStream_t stream = cudaStreamPerThread, int arch = 0) {
    Params p(const_cast<BF16 *>(q), const_cast<BF16 *>(k), const_cast<BF16 *>(v), nullptr,
             out, nullptr, nullptr, heads, kvHeads, queries, keys,
             heads * 192, 192, keyStride, 192, 192, 127, 0.0f, 1.0f / sqrtf(192.0f), 1.0f, 10000.0f);
    p.v_stride_n = kvHeads * 128;
    p.v_stride_h = 128;
    p.sink = sink;
    if ((arch == 100 || arch == 103) && queries >= 32 && heads == kvHeads * 8)
        return queries >= 8192 ? RunWindow<128>(p, stream) : RunWindow<64>(p, stream);
    if ((arch == 90 || arch == 120) && queries >= 32 && heads == kvHeads * 8)
        return arch == 90 && queries >= 8192 ? RunWindow<128>(p, stream) : RunWindow<64>(p, stream);
    // A null scratch pointer disables split-KV so the sink is counted once.
    return flashinfer::SinglePrefillWithKVCacheDispatched<192, 128,
        flashinfer::PosEncodingMode::kNone, false, flashinfer::MaskMode::kCausal, Attention, Params, false>(p, nullptr, stream);
}
}
