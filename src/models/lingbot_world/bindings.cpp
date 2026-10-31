#include "models/lingbot_world/runtime.h"
#include "models/lingbot_world/vae.h"
#include "models/lingbot_world/pipeline.h"
#include <cstring>
#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace {
thread_local std::string error;
struct Handle {
    std::mutex mutex;
    fastllm::lingbot::Runtime runtime;
    fastllm::lingbot::DiT model;
    fastllm::lingbot::Cache cache;
    Handle(const char *path,int device):runtime(device),model(runtime,path) {}
};
struct TextHandle {
    std::mutex mutex;
    fastllm::lingbot::Runtime runtime;
    fastllm::lingbot::TextEncoder model;
    TextHandle(const char *path,int device):runtime(device),model(runtime,path) {}
};
struct VAEHandle {
    std::mutex mutex;
    fastllm::lingbot::VAERuntime runtime;
    fastllm::lingbot::VAE model;
    fastllm::lingbot::VAEState encoder,decoder;
    VAEHandle(const char *path,int device,bool bf16=false):runtime(device,bf16),model(runtime,path) {}
};
struct PipelineHandle {
    std::mutex mutex;
    fastllm::lingbot::Pipeline model;
    std::unordered_map<void *,std::unique_ptr<fastllm::lingbot::WorldState>> worlds;
    PipelineHandle(const char *path,int device,bool bf16):model(path,device,bf16) {}
};
struct ResultHandle { std::vector<std::vector<uint8_t>> frames; };
fastllm::lingbot::Progress progressCallback(LingBotProgress callback,void *user) {
    if(!callback) return {};
    return [=](const std::string &stage,int current,int total) {
        if(callback(stage.c_str(),current,total,user)) throw std::runtime_error("generation cancelled");
    };
}
}
extern "C" {
const char *fastllm_lingbot_error() { return error.c_str(); }
void *fastllm_lingbot_pipeline_create(const char *path,int device) {
    return fastllm_lingbot_pipeline_create_ex(path,device,1);
}
void *fastllm_lingbot_pipeline_create_ex(const char *path,int device,int bf16Decode) {
    try {
        error.clear();
        if(!path) throw std::invalid_argument("null LingBot model directory");
        if(bf16Decode!=0 && bf16Decode!=1) throw std::invalid_argument("invalid LingBot precision");
        return new PipelineHandle(path,device,bf16Decode!=0);
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void fastllm_lingbot_pipeline_destroy(void *p) { delete static_cast<PipelineHandle *>(p); }
void *fastllm_lingbot_world_create_png(void *p,const char *prompt,const uint8_t *png,size_t length,
    int h,int w,uint64_t seed,const float *intrinsics,LingBotProgress progress,void *user) {
    try {
        error.clear();
        if(!p || !prompt || !png || !length || length>20*1024*1024 || h<64 || w<64 || h>832 || w>832 || h%16 || w%16)
            throw std::invalid_argument("invalid LingBot world image or dimensions");
        auto &ctx=*static_cast<PipelineHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        auto image=fastllm::qwen_image::decodePNG(std::string(reinterpret_cast<const char *>(png),length));
        auto pixels=fastllm::lingbot::prepareImage(image,h,w);
        fastllm::lingbot::Intrinsics k{};
        if(intrinsics) std::copy(intrinsics,intrinsics+4,k.begin());
        auto world=ctx.model.create(prompt,pixels,h,w,seed,intrinsics?&k:nullptr,progressCallback(progress,user));
        void *handle=world.get(); ctx.worlds.emplace(handle,std::move(world)); return handle;
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
int fastllm_lingbot_world_destroy(void *p,void *world) {
    try {
        error.clear(); if(!p || !world) throw std::invalid_argument("null LingBot world");
        auto &ctx=*static_cast<PipelineHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        if(!ctx.worlds.erase(world)) throw std::invalid_argument("unknown LingBot world");
        return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
void *fastllm_lingbot_world_step(void *p,void *world,const char *action,const char *prompt,
    LingBotProgress progress,void *user) {
    try {
        error.clear(); if(!p || !world || !action) throw std::invalid_argument("invalid LingBot world step");
        auto &ctx=*static_cast<PipelineHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        auto found=ctx.worlds.find(world);
        if(found==ctx.worlds.end()) throw std::invalid_argument("unknown LingBot world");
        std::string text=prompt?prompt:"";
        auto next=*found->second;
        auto segment=ctx.model.step(next,action,prompt?&text:nullptr,progressCallback(progress,user));
        auto result=std::make_unique<ResultHandle>();
        for(int f=0;f<segment.frames;++f)
            result->frames.push_back(fastllm::lingbot::encodeFramePNG(segment,f));
        *found->second=std::move(next);
        return result.release();
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
int fastllm_lingbot_result_frames(void *p) { return p?int(static_cast<ResultHandle *>(p)->frames.size()):0; }
const uint8_t *fastllm_lingbot_result_png(void *p,int frame,size_t *length) {
    if(!p || !length || frame<0 || frame>=int(static_cast<ResultHandle *>(p)->frames.size())) return nullptr;
    const auto &data=static_cast<ResultHandle *>(p)->frames[frame]; *length=data.size(); return data.data();
}
void fastllm_lingbot_result_destroy(void *p) { delete static_cast<ResultHandle *>(p); }
void *fastllm_lingbot_vae_create(const char *path,int device) {
    try {
        error.clear(); if(!path) throw std::invalid_argument("null LingBot VAE checkpoint");
        return new VAEHandle(path,device);
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void *fastllm_lingbot_vae_create_bf16(const char *path,int device) {
    try {
        error.clear(); if(!path) throw std::invalid_argument("null LingBot VAE checkpoint");
        return new VAEHandle(path,device,true);
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void fastllm_lingbot_vae_destroy(void *p) { delete static_cast<VAEHandle *>(p); }
int fastllm_lingbot_vae_reset(void *p) {
    try {
        error.clear(); if(!p) throw std::invalid_argument("null LingBot VAE");
        auto &ctx=*static_cast<VAEHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.encoder={}; ctx.decoder={}; return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
int fastllm_lingbot_vae_encode(void *p,int frames,int h,int w,const float *pixels,float *out,size_t capacity) {
    try {
        error.clear();
        if(!p || !pixels || !out || frames<1 || frames>81 || h<8 || w<8 || h>1024 || w>1024 || h%8 || w%8)
            throw std::invalid_argument("invalid LingBot VAE encoder arguments");
        auto &ctx=*static_cast<VAEHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        bool first=ctx.encoder.features.empty();
        if((frames-(first?1:0))%4) throw std::invalid_argument("Wan encoder needs first frame then groups of four");
        int count=(frames+(first?3:0))/4;
        if(capacity<size_t(count)*(h/8)*(w/8)*16) throw std::invalid_argument("VAE latent output buffer too small");
        ctx.runtime.activate(); auto next=ctx.encoder;
        auto value=ctx.model.encode(ctx.runtime.upload(pixels,frames*h*w,3),frames,h,w,next);
        auto host=ctx.runtime.download(value); std::memcpy(out,host.data(),host.size()*sizeof(float));
        ctx.encoder=std::move(next); return count;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
int fastllm_lingbot_vae_decode(void *p,int frames,int h,int w,const float *latent,float *out,size_t capacity) {
    try {
        error.clear();
        if(!p || !latent || !out || frames<1 || frames>21 || h<1 || w<1 || h>128 || w>128)
            throw std::invalid_argument("invalid LingBot VAE decoder arguments");
        auto &ctx=*static_cast<VAEHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        int count=frames*4-(ctx.decoder.features.empty()?3:0);
        if(capacity<size_t(count)*h*w*64*3) throw std::invalid_argument("VAE pixel output buffer too small");
        ctx.runtime.activate(); auto next=ctx.decoder;
        auto value=ctx.model.decode(ctx.runtime.upload(latent,frames*h*w,16),frames,h,w,next);
        auto host=ctx.runtime.download(value); std::memcpy(out,host.data(),host.size()*sizeof(float));
        ctx.decoder=std::move(next); return count;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
void *fastllm_lingbot_tokenizer_create(const char *path) {
    try { error.clear(); if(!path) throw std::invalid_argument("null tokenizer directory"); return new fastllm::lingbot::Tokenizer(path); }
    catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void fastllm_lingbot_tokenizer_destroy(void *p) { delete static_cast<fastllm::lingbot::Tokenizer *>(p); }
int fastllm_lingbot_tokenize(void *p,const char *text,int *output,int capacity) {
    try {
        error.clear(); if(!p || !text || capacity<0) throw std::invalid_argument("invalid tokenizer input");
        auto ids=static_cast<fastllm::lingbot::Tokenizer *>(p)->encode(text);
        if(output) {
            if(capacity<int(ids.size())) throw std::invalid_argument("token output buffer too small");
            std::copy(ids.begin(),ids.end(),output);
        }
        return ids.size();
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
void *fastllm_lingbot_dit_create(const char *path,int device) {
    try {
        error.clear(); if(!path) throw std::invalid_argument("null LingBot weights directory");
        return new Handle(path,device);
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void fastllm_lingbot_dit_destroy(void *p) { delete static_cast<Handle *>(p); }
void *fastllm_lingbot_text_create(const char *path,int device) {
    try {
        error.clear(); if(!path) throw std::invalid_argument("null LingBot T5 checkpoint");
        return new TextHandle(path,device);
    } catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void fastllm_lingbot_text_destroy(void *p) { delete static_cast<TextHandle *>(p); }
int fastllm_lingbot_text_encode(void *p,const int *tokens,int count,float *output) {
    try {
        error.clear(); if(!p || !tokens || !output || count<1 || count>512) throw std::invalid_argument("invalid LingBot text input");
        auto &ctx=*static_cast<TextHandle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.runtime.activate();
        auto out=ctx.model.encode(std::vector<int>(tokens,tokens+count));
        auto values=ctx.runtime.download(out); std::memcpy(output,values.data(),values.size()*sizeof(float));
        return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
int fastllm_lingbot_dit_reset(void *p,int window,int sink) {
    try {
        error.clear(); if(!p || window<1 || window>120 || sink<0 || sink>=window) throw std::invalid_argument("invalid LingBot cache window");
        auto &ctx=*static_cast<Handle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.cache={}; ctx.cache.window=window; ctx.cache.sink=sink; return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
int fastllm_lingbot_dit_forward(void *p,int frames,int h,int w,int nt,const float *latent,
    const float *condition,const float *text,const float *camera,float timestep,int commit,float *output) {
    try {
        error.clear();
        if(!p || !latent || !condition || !text || !output || frames<1 || frames>6 || h<2 || w<2 ||
           h>128 || w>128 || h%2 || w%2 || nt<1 || nt>512 || (commit!=0 && commit!=1))
            throw std::invalid_argument("invalid LingBot DiT arguments");
        auto &ctx=*static_cast<Handle *>(p); std::lock_guard<std::mutex> lock(ctx.mutex);
        auto &rt=ctx.runtime; rt.activate(); int n=frames*h*w;
        auto x=rt.fp32.upload(latent,n,16),y=rt.fp32.upload(condition,n,20),c=rt.upload(text,nt,4096);
        fastllm::lingbot::Tensor cam;
        if(camera) cam=rt.upload(camera,n,384);
        // Commit is transactional: failed inference leaves the previous world intact.
        auto next=ctx.cache;
        auto out=ctx.model.forward(x,y,c,camera?&cam:nullptr,timestep,frames,h,w,next,commit);
        auto values=rt.download(out);
        std::memcpy(output,values.data(),values.size()*sizeof(float));
        ctx.cache=std::move(next); return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
}
