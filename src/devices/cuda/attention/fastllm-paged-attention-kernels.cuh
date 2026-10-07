#pragma once
#include "fastllm-paged-attention-params.cuh"

#ifdef FASTLLM_ENABLE_FLASHINFER
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, typename DType, typename Params>
static cudaError_t FastllmDispatchPagedPrefillKernel(Params &prefill_params, DType *tmp_v, float *tmp_s,
                                                     bool enable_pdl, cudaStream_t stream) {
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
        flashinfer::PosEncodingMode::kNone, /*USE_FP16_QK_REDUCTION=*/false,
        flashinfer::MaskMode::kCausal, flashinfer::DefaultAttention<false, false, false, false>,
        Params>(prefill_params, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t HEAD_DIM, typename DType, typename Params>
static cudaError_t FastllmDispatchPagedPrefillByCtaTile(long cta_tile_q, Params &prefill_params, DType *tmp_v,
                                                        float *tmp_s, bool enable_pdl, cudaStream_t stream,
                                                        const char *op_name) {
    switch (cta_tile_q) {
        case 16:
            return FastllmDispatchPagedPrefillKernel<16, HEAD_DIM>(prefill_params, tmp_v, tmp_s, enable_pdl, stream);
        case 64:
            return FastllmDispatchPagedPrefillKernel<64, HEAD_DIM>(prefill_params, tmp_v, tmp_s, enable_pdl, stream);
        case 128:
            return FastllmDispatchPagedPrefillKernel<128, HEAD_DIM>(prefill_params, tmp_v, tmp_s, enable_pdl, stream);
        default:
            printf("%s: Unsupported cta_tile_q: %ld\n", op_name, cta_tile_q);
            return cudaErrorNotSupported;
    }
}

template <typename DType, typename Params>
cudaError_t FastllmDispatchPagedPrefillByHeadDim(uint32_t head_dim, long cta_tile_q, Params &prefill_params,
                                                        DType *tmp_v, float *tmp_s, bool enable_pdl,
                                                        cudaStream_t stream, const char *op_name) {
    switch (head_dim) {
        case 128:
            return FastllmDispatchPagedPrefillByCtaTile<128>(cta_tile_q, prefill_params, tmp_v, tmp_s, enable_pdl, stream, op_name);
        case 256:
            return FastllmDispatchPagedPrefillByCtaTile<256>(cta_tile_q, prefill_params, tmp_v, tmp_s, enable_pdl, stream, op_name);
        default:
            printf("%s: Unsupported head_dim %u\n", op_name, head_dim);
            return cudaErrorNotSupported;
    }
}

// One definition per query/KV pair, shared by single-request and batched paths.
#define FASTLLM_INSTANTIATE_PAGED_ATTENTION(Q, KV) \
    template cudaError_t FastllmDispatchPagedPrefillByHeadDim<Q, FastllmPagedPrefillParams<Q, KV>>( \
        uint32_t, long, FastllmPagedPrefillParams<Q, KV> &, Q *, float *, bool, cudaStream_t, const char *);
#endif
