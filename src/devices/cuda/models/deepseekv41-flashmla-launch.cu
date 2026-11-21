#include "deepseekv41-flashmla.cuh"
#include <cuda.h>
#include <cstring>
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"

namespace mla = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;

namespace {
template <SparseAttnFwdMode Mode>
mla::ParamT<Mode> MakeParams(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                             const FastllmV41FullDecodeBuffers &b, cudaStream_t stream) {
    mla::ParamT<Mode> p{};
    p.s_q = c.batch;
    p.h_q = 64;
    p.h_kv = 1;
    p.d_qk = p.d_v = 512;
    p.sm_scale = c.softmaxScale;
    p.sm_scale_div_log2 = c.softmaxScale * 1.4426950408889634f;
    p.q = static_cast<bf16 *>(w.query);
    p.kv = static_cast<bf16 *>(w.keys);
    p.indices = w.indices;
    p.attn_sink = const_cast<float *>(static_cast<const float *>(b.sink));
    p.lse = w.lse;
    p.stride_q_s_q = 64 * 512;
    p.stride_q_h_q = 512;
    p.stride_indices_s_q = w.pitch;
    p.num_sm = 148;
    p.stream = stream;

    // This checkpoint uses the non-normalizing query specialization.
    p.enable_q_norm = false;
    p.rms_norm_eps = c.normEps;
    p.token_positions = reinterpret_cast<uint32_t *>(w.positions);
    p.is_rope_neox_style = false;
    p.rope_dim = 64;
    p.cos_sin_cache = w.angles;
    p.n_wv_group = p.wv_group_size = 8;
    p.num_per_channels = 32;
    p.use_tma_aligned_col_major_sf = p.round_sf = p.use_packed_ue8m0 = true;
    p.out_fp8 = static_cast<fp8_e4m3 *>(w.codes);
    p.out_sf = w.scales;
    p.stride_out_sf_wv_group = 32 * w.scalePitch;
    p.stride_out_sf_head_dim = w.scalePitch;
#ifdef FASTLLM_FLASHMLA_BF16_EPILOGUE
    p.fastllm_bf16_rounding = w.bf16Rounding;
    p.fastllm_head_major = w.headMajor;
    if (w.nativeQuery) {
        p.fastllm_native_query = true;
        p.q = static_cast<bf16 *>(b.q);
        if (w.stageNative) {
            // The descriptor reorders the three innermost coordinates into FlashMLA's staged view.
            const cuuint64_t shape[] = {16, 64, 32, static_cast<cuuint64_t>(c.batch)};
            const cuuint64_t strides[] = {512 * 2, 16 * 2, 64 * 512 * 2};
            const cuuint32_t box[] = {16, 64, 8, 1}, step[] = {1, 1, 1, 1};
            CUtensorMap map{};
            static_assert(sizeof(map) == sizeof(p.fastllm_query_map));
            if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, b.q, shape,
                    strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS) {
                std::memcpy(p.fastllm_query_map, &map, sizeof(map));
                p.fastllm_stage_native = true;
            }
        }
    }
#endif
    return p;
}
} // namespace

void FastllmV41MlaCore(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                       const FastllmV41FullDecodeBuffers &b, cudaStream_t stream) {
    if (w.paged) {
        auto p = MakeParams<SparseAttnFwdMode::Decode>(w, c, b, stream);
        // Each query selects only from its own disjoint, packed page records.
        p.b = 1;
        p.num_blocks = c.batch;
        p.page_block_size = p.topk = 128;
        p.model_type = ModelType::V41;
        p.stride_kv_block = 128 * 528;
        p.stride_kv_row = 528;
        p.stride_lse_s_q = 64;
        if (w.width > 0) {
            const int extraPitch = w.pitch - 128;
            p.extra_model_type = ModelType::V41_FP4;
            p.extra_num_blocks = c.batch;
            p.extra_page_block_size = extraPitch;
            p.extra_topk = w.width;
            p.extra_kv = reinterpret_cast<bf16 *>(static_cast<uint8_t *>(w.keys) + size_t(c.batch) * 128 * 528);
            p.extra_indices = w.indices + 128;
            p.extra_topk_length = w.lengths;
            p.stride_extra_kv_block = extraPitch * 288;
            p.stride_extra_kv_row = 288;
            p.stride_extra_indices_s_q = w.pitch;
            mla::run_fused_norm_rope_attn_rope_cast_fwd_kernel<mla::Config{
                SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, 64, false}>(p);
        } else {
            p.extra_model_type = ModelType::V41;
            mla::run_fused_norm_rope_attn_rope_cast_fwd_kernel<mla::Config{
                SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41, 64, false}>(p);
        }
        return;
    }
    auto p = MakeParams<SparseAttnFwdMode::Prefill>(w, c, b, stream);
    p.s_kv = c.batch * w.pitch;
    p.topk = w.pitch;
    p.topk_length = w.lengths;
    p.stride_kv_s_kv = p.stride_kv_h_kv = 512;
    p.stride_indices_h_kv = w.pitch;
    p.max_logits = w.maxima;
    p.mode = SparseAttnFwdMode::Prefill;
    mla::run_fused_norm_rope_attn_rope_cast_fwd_kernel<mla::Config{
        SparseAttnFwdMode::Prefill, ModelType::V41, ModelType::V41, 64, false}>(p);
}
