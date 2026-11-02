#include "models/minimax_h3/parallel.h"

namespace fastllm { namespace minimax_h3 {
struct ParallelModel::Rank {
    Runtime rt,full;
    TextEncoder text;
    std::unique_ptr<Transformer> transformer;
    nomas::ParallelContext context;
    std::string directory;
    bool reference=false;
    std::unique_ptr<Runtime> videoRuntime;
    std::unique_ptr<VideoVAE> ownedVideo;
    VideoVAE *video;
    Rank(const std::string &dir,int device,bool fp32,nomas::ParallelContext ctx,DType videoPrecision,VideoVAE &rootVideo)
        :rt(device,fp32),full(device,true),text(rt,dir+"/text_encoder",ctx),
         context(ctx),directory(dir),video(&rootVideo) {
        if(ctx.rank) {
            videoRuntime=std::make_unique<Runtime>(device,videoPrecision);
            ownedVideo=std::make_unique<VideoVAE>(*videoRuntime,full,dir+"/vae");
            video=ownedVideo.get();
        }
    }
};
ParallelModel::ParallelModel(const std::string &dir,const std::vector<int> &devices,bool fp32,DType videoPrecision,VideoVAE &video,
                             nomas::RowReduction ditReduction)
    :group_(devices),ranks_(devices.size()) {
    // Construct and use each runtime on its forward thread, including cuBLAS
    // workspace and per-thread CUDA stream ownership.
    group_.run([&](int rank) {
        ranks_[rank]=std::make_unique<Rank>(dir,devices[rank],fp32,nomas::ParallelContext{&group_,rank},videoPrecision,video);
        // The text encoder keeps FP32 reduction; this context is used by DiT.
        ranks_[rank]->context.rowReduction=ditReduction;
    });
}
ParallelModel::~ParallelModel() {
    // Release device caches on their owning streams before stopping workers.
    try { group_.run([&](int rank) { ranks_[rank].reset(); }); } catch(...) {}
}
Tensor ParallelModel::encode(const std::vector<int> &tokens,const std::vector<VisionInput> &visual,Trace trace) {
    Tensor result;
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        auto x=r.text.encodeVisual(tokens,visual,rank==0?trace:Trace{});
        r.text.unload(true);
        if(rank==0) result=std::move(x);
    });
    return result;
}
void ParallelModel::prepare(const std::vector<std::vector<float>> &plans,bool reference,
                            std::function<void(int,int)> progress) {
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        if(r.transformer && r.reference!=reference) r.transformer.reset();
        if(!r.transformer) {
            r.transformer=std::make_unique<Transformer>(r.rt,r.full,
                r.directory+(reference?"/transformer_ref":"/transformer"),r.context);
            r.reference=reference;
        }
        r.transformer->prepare(plans,rank==0?progress:std::function<void(int,int)>{});
    });
}
std::pair<Tensor,Tensor> ParallelModel::forward(const Tensor &video,const Tensor &audio,const Tensor &text,
                                              const PackedLayout &layout,Trace trace) {
    std::pair<Tensor,Tensor> result;
    group_.run([&](int rank) {
        auto v=group_.broadcast(video,rank),a=group_.broadcast(audio,rank),t=group_.broadcast(text,rank);
        auto x=ranks_[rank]->transformer->forward(v,a,t,layout,rank==0?trace:Trace{});
        if(rank==0) result=std::move(x);
    });
    return result;
}
void ParallelModel::unload() {
    group_.run([&](int rank) {
        auto &r=*ranks_[rank];
        r.text.unload();
        if(r.transformer) r.transformer->unload();
        r.video->unload();
    });
}
void ParallelModel::unloadVideo() {
    group_.run([&](int rank) { ranks_[rank]->video->unload(); });
}
std::vector<Tensor> ParallelModel::videoTiles(size_t count,const std::function<Tensor(VideoVAE &,size_t)> &task) {
    return group_.mapToRoot(count,[&](int rank,size_t job) { return task(*ranks_[rank]->video,job); });
}
}} // namespace fastllm::minimax_h3
