// Reuse the same vendored FlashInfer prefill kernel as FastllmCudaHalfAttention.
// This thin adapter preserves the HY-Nomas NHD layout and BF16 element type;
// it does not install or link another attention runtime.
#include <cuda_bf16.h>
#include "attention/prefill.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"

namespace fastllm { namespace nomas {
namespace {
using B = __nv_bfloat16;
using Base = flashinfer::SinglePrefillParams<B, B, B>;
struct Params : Base { int tile, height, width; };
struct LocalAttention : flashinfer::DefaultAttention<false, true, false, false> {
    template<class P> __host__ __device__ LocalAttention(const P &p, unsigned batch, uint8_t *smem)
        : flashinfer::DefaultAttention<false, true, false, false>(p, batch, smem) {}
    REGISTER_LOGITS_MASK(p, batch, qi, ki, qh, kh, {
        int cy = max(1, min(int(qi) / p.width % p.height / p.tile, max((p.height+p.tile-1)/p.tile-2,1)));
        int cx = max(1, min(int(qi) % p.width / p.tile, max((p.width+p.tile-1)/p.tile-2,1)));
        return abs(int(ki)/p.width%p.height/p.tile-cy)<=1 && abs(int(ki)%p.width/p.tile-cx)<=1;
    })
};
}
cudaError_t fusedAttention(const void *q,const void *k,const void *v,void *out,
                           int queries,int keys,int heads,int kvHeads,bool causal,
                           int tile,int height,int width,cudaStream_t stream) {
    Params p;
    static_cast<Base &>(p)=Base((B*)q,(B*)k,(B*)v,nullptr,(B*)out,nullptr,nullptr,
        heads,kvHeads,queries,keys,heads*128,128,kvHeads*128,128,128,-1,0,
        1.f/std::sqrt(128.f),1,10000);
    p.tile=tile; p.height=height; p.width=width;
    using namespace flashinfer;
    if(tile && (height>tile*3 || width>tile*3))
        return SinglePrefillWithKVCacheDispatched<128,128,PosEncodingMode::kNone,false,
            MaskMode::kCustom,LocalAttention,Params,false>(p,nullptr,stream);
    if(causal)
        return SinglePrefillWithKVCacheDispatched<128,128,PosEncodingMode::kNone,false,
            MaskMode::kCausal,DefaultAttention<false,false,false,false>,Params,false>(p,nullptr,stream);
    return SinglePrefillWithKVCacheDispatched<128,128,PosEncodingMode::kNone,false,
        MaskMode::kNone,DefaultAttention<false,false,false,false>,Params,false>(p,nullptr,stream);
}
}} // namespace fastllm::nomas
