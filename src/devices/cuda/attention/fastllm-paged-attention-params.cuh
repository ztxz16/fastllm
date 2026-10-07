#pragma once
#include "fastllm-attention-common.cuh"

#ifdef FASTLLM_ENABLE_FLASHINFER
template <typename QType, typename KVType>
struct FastllmFP4PagedParams : flashinfer::BatchPrefillPagedParams<QType, KVType, QType, uint32_t> {
    using Base = flashinfer::BatchPrefillPagedParams<QType, KVType, QType, uint32_t>;
    using Base::Base;
    uint8_t *maybe_k_cache_sf = nullptr;
    uint8_t *maybe_v_cache_sf = nullptr;
};

template <typename QType, typename KVType>
using FastllmPagedPrefillParams = std::conditional_t<flashinfer::is_fp4_type_v<KVType>,
    FastllmFP4PagedParams<QType, KVType>,
    flashinfer::BatchPrefillPagedParams<QType, KVType, QType, uint32_t>>;

template <typename Params>
static void FastllmConfigureFP4PagedParams(Params &params, int pageLen, int numHeads, int headDim) {
    if constexpr (flashinfer::is_fp4_type_v<typename Params::DTypeKV>) {
        auto &kv = params.paged_kv;
        const uint32_t pageElements = pageLen * numHeads * headDim;
        kv.head_dim = headDim / 2;
        kv.stride_page = pageElements / 16 * 9;
        kv.stride_n = numHeads * headDim / 2;
        kv.stride_h = headDim / 2;
        params.maybe_k_cache_sf = (uint8_t*)kv.k_data + pageElements / 2;
        params.maybe_v_cache_sf = (uint8_t*)kv.v_data + pageElements / 2;
        params.k_sf_stride_page = params.v_sf_stride_page = kv.stride_page;
        params.k_sf_stride_n = params.v_sf_stride_n = numHeads * headDim / 16;
        params.k_sf_stride_h = params.v_sf_stride_h = headDim / 16;
    }
}

template <typename DType, typename Params>
cudaError_t FastllmDispatchPagedPrefillByHeadDim(
    uint32_t head_dim, long cta_tile_q, Params &prefill_params, DType *tmp_v,
    float *tmp_s, bool enable_pdl, cudaStream_t stream, const char *op_name);
#endif
