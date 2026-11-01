#include "models/hy_nomas/parallel_decoder.h"

namespace fastllm { namespace nomas {
struct ParallelPixelDecoder::Rank {
    Runtime rt;
    PixelDecoder decoder;
    Rank(const std::string &directory,int device,DType dtype,const ParallelContext &ctx)
        :rt(device,dtype),decoder(rt,directory+"/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",ctx) {}
};
ParallelPixelDecoder::ParallelPixelDecoder(const std::string &directory,const ParallelOptions &options,DType dtype)
    :group_(options.devices),ranks_(options.devices.size()) {
    group_.run([&](int rank) {
        ranks_[rank]=std::make_unique<Rank>(directory,options.devices[rank],dtype,
            ParallelContext{&group_,rank,options.rowReduction});
    });
}
ParallelPixelDecoder::~ParallelPixelDecoder() {
    try {group_.run([&](int rank) {ranks_[rank].reset();});} catch(...) {}
}
Tensor ParallelPixelDecoder::decode(const Tensor &latent,const Tensor &noise,int t,int h,int w) {
    Tensor result;
    group_.run([&](int rank) {
        auto x=group_.broadcast(latent,rank),z=group_.broadcast(noise,rank);
        auto y=ranks_[rank]->decoder.forward(x,z,t,h,w);
        if(rank==0) result=std::move(y);
    });
    return result;
}
void ParallelPixelDecoder::unload() {
    group_.run([&](int rank) {ranks_[rank]->decoder.unload();});
}
}} // namespace fastllm::nomas
