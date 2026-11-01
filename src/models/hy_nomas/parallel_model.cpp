#include "models/hy_nomas/parallel_model.h"
#include <stdexcept>

namespace fastllm { namespace nomas {
struct ParallelModel::Rank {
    Runtime rt;
    // Reuse short-lived activations without a CUDA allocation/free per op.
    TensorPool pool;
    DiT dit;
    PixelDecoder decoder;
    std::vector<Tensor> camera;
    Rank(const std::string &directory,int device,DType dtype,const ParallelContext &ctx)
        :rt(device,dtype),pool(device,256u<<20),dit(rt,directory+"/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",ctx),
         decoder(rt,directory+"/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",ctx) {}
};
ParallelModel::ParallelModel(const std::string &directory,const ParallelOptions &options,DType dtype)
    :group_(options.devices),ranks_(options.devices.size()) {
    group_.run([&](int rank) {
        ranks_[rank]=std::make_unique<Rank>(directory,options.devices[rank],dtype,
            ParallelContext{&group_,rank,options.rowReduction});
    });
}
ParallelModel::~ParallelModel() {
    try {group_.run([&](int rank) {ranks_[rank].reset();});} catch(...) {}
}
void ParallelModel::prepareCamera(const Tensor &poses,int spatial) {
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        TensorPool::Scope allocations(r.pool);
        r.camera=r.dit.cameraProjections(r.dit.cameraEmbedding(group_.broadcast(poses,rank)),spatial);
    });
}
Tensor ParallelModel::forward(const Tensor &input,const Tensor &features,float time,int t,int h,int w,int offset,
                               std::vector<DiTCache> *cache,bool commit,bool camera,std::vector<DiTCache> *workspace) {
    if((commit && !cache) || (cache && !cache->empty() && cache->size()!=ranks_.size()) ||
       (workspace && !workspace->empty() && workspace->size()!=ranks_.size()) || (cache && cache==workspace))
        throw std::invalid_argument("invalid parallel DiT cache/workspace");
    auto next=cache?*cache:std::vector<DiTCache>{}; next.resize(ranks_.size());
    if(workspace) workspace->resize(ranks_.size());
    Tensor result;
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        TensorPool::Scope allocations(r.pool);
        auto x=group_.broadcast(input,rank),text=group_.broadcast(features,rank);
        auto y=r.dit.forward(x,text,time,t,h,w,offset,cache?&next[rank]:nullptr,commit,
                            camera?&r.camera:nullptr,workspace?&(*workspace)[rank]:nullptr);
        if(rank==0) result=std::move(y);
    });
    if(commit) *cache=std::move(next);
    return result;
}
Tensor ParallelModel::decode(const Tensor &latent,const Tensor &noise,int t,int h,int w) {
    Tensor result;
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        TensorPool::Scope allocations(r.pool);
        auto x=group_.broadcast(latent,rank),z=group_.broadcast(noise,rank);
        auto y=r.decoder.forward(x,z,t,h,w);
        if(rank==0) result=std::move(y);
    });
    return result;
}
void ParallelModel::unload() {
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        TensorPool::Scope allocations(r.pool);
        r.dit.unload();r.decoder.unload();r.camera.clear();
        r.rt.synchronize();r.pool.trim();
    });
}
}} // namespace fastllm::nomas
