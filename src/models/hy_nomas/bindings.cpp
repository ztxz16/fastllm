#include "models/hy_nomas/runtime.h"
#include "models/hy_nomas/parallel.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace fastllm { namespace nomas {
namespace { void require(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); } }

DiT::DiT(Runtime &rt,const std::string &path):DiT(rt,path,ParallelContext{}) {}
DiT::DiT(Runtime &rt,const std::string &path,const ParallelContext &parallel)
    :rt_(rt),weights_(rt,path),parallel_(new ParallelContext(parallel)) {}
DiT::~DiT()=default;
int DiT::sequencePartitions(int rows) const {
    // Tiny shards change the GEMM algorithm/rounding and cannot amortize NCCL.
    return rows/parallel_->size()>=64?parallel_->size():1;
}
void DiT::unload() { weights_.clear(); modulation_.clear(); }
BlockResult DiT::block(int layer,const Tensor &image,const Tensor &text,const Tensor &time,
    const float *cos,const float *sin,const Tensor *historyK,const Tensor *historyV,const uint8_t *mask) {
    return block(layer,image,text,time,rt_.prepareRope(cos,sin,image.rows,128),historyK,historyV,mask);
}
BlockResult DiT::block(int layer,const Tensor &image,const Tensor &text,const Tensor &time,
    const RopeTable &rope,const Tensor *historyK,const Tensor *historyV,const uint8_t *mask,
    const Tensor *imageMod,const Tensor *textMod,Tensor *keyWorkspace,Tensor *valueWorkspace,int globalImageRows) {
    require(layer>=0 && layer<54 && image.cols==2048 && text.cols==2048 && time.rows==1 && time.cols==2048,"invalid DiT block input");
    require(bool(historyK)==bool(historyV),"both K and V history required");
    require(bool(keyWorkspace)==bool(valueWorkspace),"both K and V workspaces required");
    require(globalImageRows>=0,"invalid distributed image rows");
    const int total=globalImageRows?globalImageRows:image.rows;
    const int ranks=sequencePartitions(total),rank=ranks>1?parallel_->rank:0;
    const bool split=ranks>1;
    const int chunk=(total+ranks-1)/ranks,first=rank*chunk,rows=std::min(chunk,total-first);
    require(!globalImageRows || image.rows==rows,"invalid distributed image rows");
    auto local=globalImageRows?image:image.rowSlice(first,rows);
    RopeTable localRope=globalImageRows?rope:RopeTable{rope.cos.rowSlice(first,rows),rope.sin.rowSlice(first,rows)};
    auto p="double_blocks."+std::to_string(layer)+".";
    auto act=imageMod && textMod?Tensor{}:rt_.activation(time);
    auto im=imageMod?*imageMod:weights_.linear(act,p+"img_mod.linear");
    auto tm=textMod?*textMod:weights_.linear(act,p+"txt_mod.linear");
    auto inputI=rt_.modulate(rt_.norm(local,false),im,0,1),inputT=rt_.modulate(rt_.norm(text,false),tm,0,1);
    auto project=[&](const Tensor &x,const std::string &s) {
        auto q=weights_.linear(x,p+s+"_attn_q"),k=weights_.linear(x,p+s+"_attn_k"),v=weights_.linear(x,p+s+"_attn_v");
        q=rt_.norm(q,true,&weights_.get(p+s+"_attn_q_norm.weight"),nullptr,1e-6f,128);
        k=rt_.norm(k,true,&weights_.get(p+s+"_attn_k_norm.weight"),nullptr,1e-6f,128);
        if(s=="img") {q=rt_.rope(q,localRope);k=rt_.rope(k,localRope);}
        return std::array<Tensor,3>{q,k,v};
    };
    auto i=project(inputI,"img");
    auto imageK=i[1],imageV=i[2];
    if(split) {
        auto full=parallel_->gatherRows(rt_,rt_.concat({imageK,imageV}),total,2,true);
        imageK=full.rowSlice(0,total);imageV=full.rowSlice(total,total);
    }
    auto t=project(inputT,"txt");
    if(split) parallel_->group->wait(parallel_->rank);
    auto append=[&](const Tensor &image,const Tensor &text,const Tensor *history,Tensor *workspace) {
        if(!workspace || !workspace->data) {
            auto joined=history?rt_.concat({*history,image,text}):rt_.concat({image,text});
            if(workspace) *workspace=joined;
            return joined;
        }
        int offset=history?history->rows:0;
        require(workspace->rows==offset+image.rows+text.rows && workspace->cols==image.cols &&
                workspace->dtype==image.dtype && (!history || workspace->storage!=history->storage),
                "invalid or aliased DiT workspace");
        auto current=workspace->rowSlice(offset,image.rows+text.rows);
        rt_.concatInto({image,text},current);
        return *workspace;
    };
    auto q=rt_.concat({i[0],t[0]});
    auto allK=append(imageK,t[1],historyK,keyWorkspace),allV=append(imageV,t[2],historyV,valueWorkspace);
    auto k=allK.rowSlice(historyK?historyK->rows:0,total+text.rows);
    auto v=allV.rowSlice(historyV?historyV->rows:0,total+text.rows);
    std::vector<uint8_t> localMask;
    if(mask && split) {
        localMask.resize(size_t(rows+text.rows)*allK.rows);
        std::copy_n(mask+size_t(first)*allK.rows,size_t(rows)*allK.rows,localMask.data());
        std::copy_n(mask+size_t(total)*allK.rows,size_t(text.rows)*allK.rows,localMask.data()+size_t(rows)*allK.rows);
        mask=localMask.data();
    }
    auto attn=rt_.attention(q,allK,allV,16,mask);
    auto finish=[&](const Tensor &x,const Tensor &m,const Tensor &a,const std::string &s) {
        auto h=rt_.residual(x,weights_.linear(a,p+s+"_attn_proj"),m,2);
        auto mlp=weights_.linear(rt_.activation(weights_.linear(rt_.modulate(rt_.norm(h,false),m,3,4),p+s+"_mlp.fc1"),true),p+s+"_mlp.fc2");
        return rt_.residual(h,mlp,m,5);
    };
    auto imageOut=finish(local,im,attn.rowSlice(0,rows),"img");
    auto textOut=finish(text,tm,attn.rowSlice(rows,text.rows),"txt");
    if(split && !globalImageRows) imageOut=parallel_->gatherRows(rt_,imageOut,total);
    return {imageOut,textOut,k,v};
}

}} // namespace fastllm::nomas

namespace {
thread_local std::string lastError;
struct NomasHandle {
    fastllm::nomas::Runtime runtime;
    fastllm::nomas::DiT dit;
    fastllm::nomas::DiTCache cache;
    NomasHandle(const char *path,int device):runtime(device),dit(runtime,path) {}
};
struct DecoderHandle {
    fastllm::nomas::Runtime runtime;
    fastllm::nomas::PixelDecoder decoder;
    DecoderHandle(const char *path,int device):runtime(device),decoder(runtime,path) {}
};
struct TextHandle {
    fastllm::nomas::Runtime runtime;
    fastllm::nomas::QwenTextEncoder encoder;
    fastllm::nomas::QwenTokenizer tokenizer;
    TextHandle(const char *path,int device):runtime(device),encoder(runtime,path),tokenizer(path) {}
};
struct PipelineHandle {
    std::mutex mutex;
    fastllm::nomas::Pipeline pipeline;
    PipelineHandle(const char *model,const char *text,int device):pipeline(model,text,device) {}
    PipelineHandle(const char *model,const char *text,const char *devices)
        :pipeline(model,text,fastllm::nomas::ParallelOptions{fastllm::nomas::parseParallelDevices(devices),
                                                          fastllm::nomas::RowReduction::Activation}) {}
};
int checkedTokens(int t,int h,int w) {
    if(t<1 || h<1 || w<1 || t>64 || h>128 || w>128 || int64_t(t)*h*w>16384)
        throw std::invalid_argument("invalid or oversized HY-Nomas tensor grid");
    return t*h*w;
}
void copyOutput(fastllm::nomas::Runtime &r,const fastllm::nomas::Tensor &x,float *dest) {
    auto values=r.download(x); std::memcpy(dest,values.data(),values.size()*4);
}
}
extern "C" {
void *fastllm_nomas_create(const char *path,int device) {
    try { lastError.clear(); if(!path) throw std::invalid_argument("null checkpoint"); return new NomasHandle(path,device); }
    catch(const std::exception &e) { lastError=e.what(); return nullptr; }
}
void fastllm_nomas_destroy(void *handle) { delete static_cast<NomasHandle *>(handle); }
const char *fastllm_nomas_error() { return lastError.c_str(); }
int fastllm_nomas_block(void *handle,int layer,int ni,int nt,const float *img,const float *txt,const float *time,
    const float *cs,const float *sn,int nh,const float *hk,const float *hv,const uint8_t *mask,float *oi,float *ot,float *ok,float *ov) {
    try {
        lastError.clear(); if(!handle || !oi || !ot || !ok || !ov || nh<0) throw std::invalid_argument("invalid block arguments");
        auto &h=*static_cast<NomasHandle *>(handle); auto &r=h.runtime;
        auto image=r.upload(img,ni,2048),text=r.upload(txt,nt,2048),vec=r.upload(time,1,2048);
        fastllm::nomas::Tensor k,v; if(nh) { k=r.upload(hk,nh,2048); v=r.upload(hv,nh,2048); }
        auto out=h.dit.block(layer,image,text,vec,cs,sn,nh?&k:nullptr,nh?&v:nullptr,mask);
        auto copy=[&](const fastllm::nomas::Tensor &x,float *dest) { auto values=r.download(x); std::memcpy(dest,values.data(),values.size()*4); };
        copy(out.image,oi); copy(out.text,ot); copy(out.key,ok); copy(out.value,ov); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_dit(void *handle,int t,int h,int w,int nt,const float *latent,const float *features,
                      float timestep,int offset,int commit,float *output) {
    try {
        lastError.clear(); if(!handle || !output || nt<1 || nt>1280 || (commit!=0 && commit!=1)) throw std::invalid_argument("invalid DiT arguments");
        int n=checkedTokens(t,h,w); auto &ctx=*static_cast<NomasHandle *>(handle); auto &r=ctx.runtime;
        auto out=ctx.dit.forward(r.upload(latent,n,32),r.upload(features,nt,3584),timestep,t,h,w,offset,&ctx.cache,commit);
        copyOutput(r,out,output); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
void *fastllm_nomas_decoder_create(const char *path,int device) {
    try { lastError.clear(); if(!path) throw std::invalid_argument("null decoder checkpoint"); return new DecoderHandle(path,device); }
    catch(const std::exception &e) { lastError=e.what(); return nullptr; }
}
void fastllm_nomas_decoder_destroy(void *handle) { delete static_cast<DecoderHandle *>(handle); }
int fastllm_nomas_decoder_block(void *handle,int layer,int t,int h,int w,int tile,const float *input,float *output) {
    try {
        lastError.clear(); if(!handle || !output) throw std::invalid_argument("invalid decoder arguments");
        int n=checkedTokens(t,h,w); auto &ctx=*static_cast<DecoderHandle *>(handle); auto &r=ctx.runtime;
        copyOutput(r,ctx.decoder.block(layer,r.upload(input,n,4096),t,h,w,tile),output); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_decode(void *handle,int t,int h,int w,const float *latent,const float *noise,float *output) {
    try {
        lastError.clear(); if(!handle || !output) throw std::invalid_argument("invalid decoder arguments");
        int n=checkedTokens(t,h,w); checkedTokens(t,h*2,w*2); auto &ctx=*static_cast<DecoderHandle *>(handle); auto &r=ctx.runtime;
        copyOutput(r,ctx.decoder.forward(r.upload(latent,n,32),r.upload(noise,n*4,3072),t,h,w),output); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
void *fastllm_nomas_text_create(const char *path,int device) {
    try { lastError.clear(); if(!path) throw std::invalid_argument("null text checkpoint"); return new TextHandle(path,device); }
    catch(const std::exception &e) { lastError=e.what(); return nullptr; }
}
void fastllm_nomas_text_destroy(void *handle) { delete static_cast<TextHandle *>(handle); }
int fastllm_nomas_tokenize(void *handle,const char *text,int video,int *tokens,int capacity,int *crop) {
    try {
        lastError.clear(); if(!handle || !text || !crop || capacity<0) throw std::invalid_argument("invalid tokenization arguments");
        auto &ctx=*static_cast<TextHandle *>(handle); *crop=0;
        auto ids=video?ctx.tokenizer.videoPrompt(text,*crop):ctx.tokenizer.encode(text);
        if(tokens && capacity>=int(ids.size())) std::copy(ids.begin(),ids.end(),tokens);
        else if(tokens) throw std::invalid_argument("token output buffer too small");
        return ids.size();
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_encode(void *handle,const int *tokens,int count,float *features) {
    try {
        lastError.clear(); if(!handle || !tokens || !features || count<1 || count>1536) throw std::invalid_argument("invalid text encoder arguments");
        auto &ctx=*static_cast<TextHandle *>(handle);
        copyOutput(ctx.runtime,ctx.encoder.encode(std::vector<int>(tokens,tokens+count)),features); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
void *fastllm_nomas_pipeline_create(const char *model,const char *text,int device) {
    try { lastError.clear(); if(!model || !text) throw std::invalid_argument("null model directory"); return new PipelineHandle(model,text,device); }
    catch(const std::exception &e) { lastError=e.what(); return nullptr; }
}
void *fastllm_nomas_pipeline_create_parallel(const char *model,const char *text,const char *devices) {
    try {
        lastError.clear();
        if(!model || !text || !devices) throw std::invalid_argument("null model directory or parallel devices");
        return new PipelineHandle(model,text,devices);
    } catch(const std::exception &e) {lastError=e.what();return nullptr;}
}
void fastllm_nomas_pipeline_destroy(void *handle) { delete static_cast<PipelineHandle *>(handle); }
int fastllm_nomas_generate(void *handle,const char *prompt,int width,int height,int frames,uint64_t seed,int steps,
                          uint8_t *rgb,size_t capacity,fastllm::nomas::Progress progress,void *user) {
    try {
        lastError.clear(); if(!handle || !prompt || !rgb || width<1 || width>512 || height<1 || height>512 || frames<1 || frames>40 || capacity<size_t(width)*height*frames*3)
            throw std::invalid_argument("invalid generation arguments or RGB buffer capacity");
        auto &ctx=*static_cast<PipelineHandle *>(handle); std::lock_guard<std::mutex> lock(ctx.mutex);
        auto result=ctx.pipeline.generate(prompt,width,height,frames,seed,steps,progress,user);
        std::memcpy(rgb,result.data(),result.size()); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
void *fastllm_nomas_world_create(void *pipeline,const char *prompt,int width,int height,uint64_t seed,int steps) {
    try {
        lastError.clear(); if(!pipeline || !prompt) throw std::invalid_argument("invalid world arguments");
        auto &ctx=*static_cast<PipelineHandle *>(pipeline); std::lock_guard<std::mutex> lock(ctx.mutex);
        return ctx.pipeline.createWorld(prompt,width,height,seed,steps).release();
    } catch(const std::exception &e) { lastError=e.what(); return nullptr; }
}
void fastllm_nomas_world_destroy(void *pipeline,void *world) {
    if(!pipeline) return;
    auto &ctx=*static_cast<PipelineHandle *>(pipeline); std::lock_guard<std::mutex> lock(ctx.mutex);
    delete static_cast<fastllm::nomas::WorldState *>(world);
}
int fastllm_nomas_world_step(void *pipeline,void *world,int action,uint8_t *rgb,size_t capacity,
                            fastllm::nomas::Progress progress,void *user) {
    return fastllm_nomas_world_step_prompt(pipeline,world,nullptr,action,rgb,capacity,progress,user);
}
int fastllm_nomas_world_step_prompt(void *pipeline,void *world,const char *prompt,int action,uint8_t *rgb,size_t capacity,
                                   fastllm::nomas::Progress progress,void *user) {
    try {
        lastError.clear(); if(!pipeline || !world || !rgb) throw std::invalid_argument("invalid world step arguments");
        auto &ctx=*static_cast<PipelineHandle *>(pipeline); std::lock_guard<std::mutex> lock(ctx.mutex);
        auto &state=*static_cast<fastllm::nomas::WorldState *>(world);
        if(capacity<size_t(state.width)*state.height*20*3) throw std::invalid_argument("world RGB buffer too small");
        std::string replacement=prompt?prompt:"";
        auto result=ctx.pipeline.stepWorld(state,action,progress,user,prompt?&replacement:nullptr);
        std::memcpy(rgb,result.data(),result.size()); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_camera(void *handle,int layer,int t,int h,int w,const float *input,const float *extrinsics,float *output) {
    try {
        lastError.clear(); if(!handle || !output) throw std::invalid_argument("invalid camera arguments");
        int n=checkedTokens(t,h,w); auto &ctx=*static_cast<NomasHandle *>(handle); auto &r=ctx.runtime;
        auto camera=ctx.dit.cameraEmbedding(r.upload(extrinsics,t,64));
        copyOutput(r,ctx.dit.cameraBlock(layer,r.upload(input,n,2048),camera,h*w),output); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_dit_camera(void *handle,int t,int h,int w,int nt,const float *latent,const float *features,float timestep,
                            int offset,int commit,const float *extrinsics,float *output) {
    try {
        lastError.clear(); if(!handle || !output || nt<1 || nt>1280 || (commit!=0 && commit!=1)) throw std::invalid_argument("invalid camera DiT arguments");
        int n=checkedTokens(t,h,w); auto &ctx=*static_cast<NomasHandle *>(handle); auto &r=ctx.runtime;
        auto camera=ctx.dit.cameraEmbedding(r.upload(extrinsics,t,64));
        auto projections=ctx.dit.cameraProjections(camera,h*w);
        copyOutput(r,ctx.dit.forward(r.upload(latent,n,32),r.upload(features,nt,3584),timestep,t,h,w,offset,&ctx.cache,commit,&projections),output);
        return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}
int fastllm_nomas_camera_trajectory(int frames,int action,float *output) {
    try {
        lastError.clear(); if(!output) throw std::invalid_argument("null camera output");
        auto values=fastllm::nomas::cameraTrajectory(frames,action);
        std::memcpy(output,values.data(),values.size()*sizeof(float)); return 0;
    } catch(const std::exception &e) { lastError=e.what(); return -1; }
}

}
