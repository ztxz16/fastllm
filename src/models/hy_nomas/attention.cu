// Reuse the same vendored FlashInfer prefill kernel as FastllmCudaHalfAttention.
// This thin adapter preserves the HY-Nomas NHD layout and FP16/BF16 element types;
// it does not install or link another attention runtime.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "attention/prefill.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"

namespace fastllm { namespace nomas {
namespace {
template<class T> using Base = flashinfer::SinglePrefillParams<T, T, T>;
template<class T> struct Params : Base<T> { int tile, height, width; };
struct BF16Attention : flashinfer::DefaultAttention<false, false, false, false> {
    using DefaultAttention::DefaultAttention;
    // Preserve HY-Nomas' FP32 denominator without changing the FP16 path.
    static constexpr bool use_fp32_softmax_sum = true;
    REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
        return output * ((m != -flashinfer::math::inf && d > 0.f) ? 1.f / d : 0.f);
    })
};
static_assert(!flashinfer::prefill_fp32_softmax_sum<flashinfer::DefaultAttention<false,false,false,false>>::value,
              "HY-Nomas must not change the default attention accumulation");
template<class Attention> struct LocalAttention : Attention {
    template<class P> __host__ __device__ LocalAttention(const P &p, unsigned batch, uint8_t *smem)
        : Attention(p, batch, smem) {}
    REGISTER_LOGITS_MASK(p, batch, qi, ki, qh, kh, {
        int cy = max(1, min(int(qi) / p.width % p.height / p.tile, max((p.height+p.tile-1)/p.tile-2,1)));
        int cx = max(1, min(int(qi) % p.width / p.tile, max((p.width+p.tile-1)/p.tile-2,1)));
        return abs(int(ki)/p.width%p.height/p.tile-cy)<=1 && abs(int(ki)%p.width/p.tile-cx)<=1;
    })
};
template<class T,int D> cudaError_t dispatchAttention(const void *q,const void *k,const void *v,void *out,
                           int queries,int keys,int heads,int kvHeads,bool causal,
                           int tile,int height,int width,float scoreScale,cudaStream_t stream) {
    Params<T> p;
    static_cast<Base<T> &>(p)=Base<T>((T*)q,(T*)k,(T*)v,nullptr,(T*)out,nullptr,nullptr,
        heads,kvHeads,queries,keys,heads*D,D,kvHeads*D,D,D,-1,0,
        scoreScale,1,10000);
    p.tile=tile; p.height=height; p.width=width;
    using namespace flashinfer;
    using Attention = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, BF16Attention,
                                         DefaultAttention<false,false,false,false>>;
    if(tile && (height>tile*3 || width>tile*3))
        return SinglePrefillWithKVCacheDispatched<D,D,PosEncodingMode::kNone,false,
            MaskMode::kCustom,LocalAttention<Attention>,Params<T>,false>(p,nullptr,stream);
    if(causal)
        return SinglePrefillWithKVCacheDispatched<D,D,PosEncodingMode::kNone,false,
            MaskMode::kCausal,Attention,Params<T>,false>(p,nullptr,stream);
    return SinglePrefillWithKVCacheDispatched<D,D,PosEncodingMode::kNone,false,
        MaskMode::kNone,Attention,Params<T>,false>(p,nullptr,stream);
}
} // namespace
cudaError_t fusedAttention(const void *q,const void *k,const void *v,void *out,
                           int queries,int keys,int heads,int kvHeads,int dim,bool fp16,bool causal,
                           int tile,int height,int width,float scoreScale,cudaStream_t stream) {
#define ATTENTION(T,D) dispatchAttention<T,D>(q,k,v,out,queries,keys,heads,kvHeads,causal,tile,height,width,scoreScale,stream)
    if(dim==64) return fp16 ? ATTENTION(__half,64) : ATTENTION(__nv_bfloat16,64);
    if(dim==128) return fp16 ? ATTENTION(__half,128) : ATTENTION(__nv_bfloat16,128);
#undef ATTENTION
    return cudaErrorInvalidValue;
}
}} // namespace fastllm::nomas
