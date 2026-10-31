#include "models/lingbot_world/pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
struct HostTensor {
    int rows=0,cols=0;
    bool fp32=true;
    mutable Tensor device;
    mutable std::vector<uint8_t> bytes;
};
// Immutable snapshots share device storage with out-of-place inference.
// Host bytes are materialized only under memory pressure or explicit offload.
struct HostVAEState {
    std::unordered_map<std::string,HostTensor> features;
    int height=0,width=0,conditionFrames=-1;
};
struct HostDiTState {
    std::vector<HostTensor> keys,values,textKeys,textValues;
    int frames=0,window=18,sink=6;
};
namespace {
void copyBytes(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind) {
    if(!bytes) return;
    auto status=cudaMemcpyAsync(destination,source,bytes,kind,cudaStreamPerThread);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
Tensor restoreTensor(const HostTensor &host) {
    if(host.device.data) return host.device;
    auto value=host.rows?Tensor(host.rows,host.cols,host.fp32):Tensor{};
    copyBytes(value.data,host.bytes.data(),host.bytes.size(),cudaMemcpyHostToDevice);
    return value;
}
HostTensor snapshotTensor(const Tensor &value) {
    return {value.rows,value.cols,value.isFloat32(),value,{}};
}
void offloadTensor(const HostTensor &host) {
    if(!host.device.data) return;
    if(host.bytes.empty()) {
        // Keep the device snapshot until its complete host copy is valid.
        std::vector<uint8_t> bytes(host.device.bytes());
        copyBytes(bytes.data(),host.device.data,bytes.size(),cudaMemcpyDeviceToHost);
        auto status=cudaStreamSynchronize(cudaStreamPerThread);
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        host.bytes=std::move(bytes);
    }
    host.device={};
}
Cache restore(Runtime &rt,const std::shared_ptr<const HostDiTState> &host) {
    Cache cache;
    if(!host) return cache;
    cache.frames=host->frames; cache.window=host->window; cache.sink=host->sink;
    auto upload=[](const std::vector<HostTensor> &source,std::vector<Tensor> &target) {
        target.reserve(source.size());
        for(const auto &feature:source) target.push_back(restoreTensor(feature));
    };
    try {
        upload(host->keys,cache.keys); upload(host->values,cache.values);
        upload(host->textKeys,cache.textKeys); upload(host->textValues,cache.textValues);
        rt.synchronize();
    } catch(...) { try { rt.synchronize(); } catch(...) {} throw; }
    return cache;
}
std::shared_ptr<const HostDiTState> snapshot(const Cache &cache) {
    auto host=std::make_shared<HostDiTState>();
    host->frames=cache.frames; host->window=cache.window; host->sink=cache.sink;
    auto retain=[](const std::vector<Tensor> &source,std::vector<HostTensor> &target) {
        target.reserve(source.size());
        for(const auto &value:source) target.push_back(snapshotTensor(value));
    };
    retain(cache.keys,host->keys); retain(cache.values,host->values);
    retain(cache.textKeys,host->textKeys); retain(cache.textValues,host->textValues);
    return host;
}
VAEState restore(VAERuntime &rt,const std::shared_ptr<const HostVAEState> &host) {
    VAEState state;
    if(!host) return state;
    state.height=host->height; state.width=host->width; state.conditionFrames=host->conditionFrames;
    try {
        for(const auto &entry:host->features) state.features[entry.first]=restoreTensor(entry.second);
        rt.synchronize();
    } catch(...) { try { rt.synchronize(); } catch(...) {} throw; }
    return state;
}
std::shared_ptr<const HostVAEState> snapshot(const VAEState &state) {
    auto host=std::make_shared<HostVAEState>();
    host->height=state.height; host->width=state.width; host->conditionFrames=state.conditionFrames;
    for(const auto &entry:state.features) host->features.emplace(entry.first,snapshotTensor(entry.second));
    return host;
}
template<class T> void prune(std::vector<std::weak_ptr<const T>> &states) {
    states.erase(std::remove_if(states.begin(),states.end(),[](const auto &s){return s.expired();}),states.end());
}
}
Pipeline::Pipeline(const std::string &directory,int device,bool bf16Decode)
    :tensorPool_(device),rt_(device),vaeRt_(device),decoderRt_(device,bf16Decode),
    tokenizer_(directory+"/google/umt5-xxl"),text_(rt_,directory+"/native/t5.safetensors"),
    dit_(rt_,directory+"/transformers"),vae_(vaeRt_,directory+"/native/vae.safetensors"),
    decoder_(decoderRt_,directory+"/native/vae.safetensors"),schedule_(fastSchedule()) {}
void Pipeline::offloadState() {
    rt_.activate();
    nomas::TensorPool::Scope allocations(tensorPool_);
    prune(cacheStates_); prune(vaeStates_);
    for(const auto &weak:cacheStates_) if(auto state=weak.lock()) {
        for(const auto *values:{&state->keys,&state->values,&state->textKeys,&state->textValues})
            for(const auto &value:*values) offloadTensor(value);
    }
    for(const auto &weak:vaeStates_) if(auto state=weak.lock())
        for(const auto &entry:state->features) offloadTensor(entry.second);
    rt_.synchronize();
    tensorPool_.trim();
}
void Pipeline::reserveWorkingMemory(size_t bytes) {
    prune(cacheStates_); prune(vaeStates_);
    auto available=[&] {
        rt_.synchronize();
        size_t free=0,total=0;
        auto status=cudaMemGetInfo(&free,&total);
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return free+tensorPool_.reusableBytes();
    };
    if(available()>=bytes) return;
    offloadState();
    if(available()<bytes) dit_.unload();
}
Tensor Pipeline::encodePrompt(const std::string &prompt) {
    if(prompt.size()>16384) throw std::invalid_argument("LingBot prompt exceeds 16384 bytes");
    offloadState();
    dit_.unload();
    try {
        auto features=text_.encode(tokenizer_.encode(prompt));
        text_.unload(); return features;
    } catch(...) { text_.unload(); throw; }
}
std::unique_ptr<WorldState> Pipeline::create(const std::string &prompt,const std::vector<float> &pixels,
    int h,int w,uint64_t seed,const Intrinsics *intrinsics,Progress progress) {
    if(h<64 || w<64 || h>832 || w>832 || h%16 || w%16 || pixels.size()!=size_t(h)*w*3)
        throw std::invalid_argument("LingBot image must be 64..832 pixels per side, in multiples of 16");
    for(float value:pixels) if(!std::isfinite(value)) throw std::invalid_argument("nonfinite input image");
    rt_.activate();
    nomas::TensorPool::Scope allocations(tensorPool_);
    auto world=std::make_unique<WorldState>();
    world->height=h; world->width=w; world->seed=seed; world->prompt=prompt;
    float focal=float(w)*.5f/std::tan(float(3.14159265358979323846/6.));
    world->intrinsics=intrinsics?*intrinsics:Intrinsics{focal,focal,float(w)*.5f,float(h)*.5f};
    for(float value:world->intrinsics) if(!std::isfinite(value)) throw std::invalid_argument("invalid camera intrinsics");
    if(world->intrinsics[0]<=0 || world->intrinsics[1]<=0) throw std::invalid_argument("camera focal lengths must be positive");
    if(progress) progress("text",0,1);
    world->text=encodePrompt(prompt);
    if(progress) progress("text",1,1);
    world->firstPixels=rt_.fp32.upload(pixels.data(),h*w,3);
    return world;
}
Segment Pipeline::step(WorldState &world,const std::string &action,const std::string *prompt,Progress progress,Trace trace) {
    rt_.activate();
    nomas::TensorPool::Scope allocations(tensorPool_);
    WorldState next=world;
    // Validate the action before expensive prompt/image processing.
    movePose(next.pose,action,0,0);
    bool changedPrompt=prompt && *prompt!=next.prompt;
    if(changedPrompt) {
        if(progress) progress("text",0,1);
        next.text=encodePrompt(*prompt); next.prompt=*prompt;
        if(progress) progress("text",1,1);
    }
    auto emit=[&](const std::string &name,const Tensor &x) { if(trace) trace(name,x); };
    int h=next.height,w=next.width,lh=h/8,lw=w/8,n=3*lh*lw;
    if(progress) progress("image",0,1);
    // Bound retained memory before the spatial encoder's working set grows.
    reserveWorkingMemory(size_t(h)*w*16384 + (size_t(512)<<20));
    auto encoder=restore(vaeRt_,next.encoder);
    auto condition=vae_.encodeCondition(next.firstPixels,3,h,w,encoder);
    next.encoder=snapshot(encoder);
    vaeStates_.push_back(next.encoder);
    encoder={};
    std::vector<float> mask(size_t(n)*4,0.f);
    if(next.segments==0) std::fill(mask.begin(),mask.begin()+size_t(lh)*lw*4,1.f);
    condition=rt_.join(rt_.fp32.upload(mask.data(),n,4),condition);
    if(progress) progress("image",1,1);
    auto oldPose=next.pose;
    std::vector<Pose> poses;
    for(int i=0;i<3;++i) {
        if(next.segments>0 || i>0) next.pose=movePose(next.pose,action,.1f,4.f);
        poses.push_back(next.pose);
    }
    auto relative=relativePoses(poses,next.segments?&oldPose:nullptr);
    auto camera=cameraEmbedding(rt_,relative,next.intrinsics,h,w);
    auto latent=rt_.normal(n,16,next.seed,next.noiseSequence++);
    emit("text",next.text); emit("condition",condition); emit("camera",camera); emit("latent.0",latent);
    // The image/decoder histories may still be resident after encoding.
    // Leave room for DiT weights, KV restoration and camera modulation before
    // retaining active cache references that eviction cannot release.
    reserveWorkingMemory(size_t(h)*w*8192 + (size_t(4)<<30));
    auto cache=restore(rt_,next.cache);
    if(changedPrompt) { cache.textKeys.clear(); cache.textValues.clear(); }
    for(size_t step=0;step<schedule_.size();++step) {
        if(progress) progress("denoise",step,schedule_.size());
        if(step==0) dit_.prepareCamera(camera,3,lh,lw);
        auto flow=dit_.forward(latent,condition,next.text,&camera,float(schedule_[step].timestep),3,lh,lw,cache,false);
        emit("flow."+std::to_string(step),flow);
        latent=rt_.flowToClean(latent,flow,schedule_[step].flowSigma);
        emit("clean."+std::to_string(step),latent);
        if(step+1<schedule_.size()) {
            auto noise=rt_.normal(n,16,next.seed,next.noiseSequence++);
            emit("noise."+std::to_string(step),noise);
            latent=rt_.addNoise(latent,noise,schedule_[step+1].noiseSigma);
            emit("latent."+std::to_string(step+1),latent);
        }
    }
    if(progress) progress("denoise",schedule_.size(),schedule_.size());
    if(progress) progress("history",0,1);
    dit_.forward(latent,condition,next.text,&camera,0,3,lh,lw,cache,true);
    dit_.clearCamera();
    // Cache updates replace tensors; the previous world remains immutable.
    next.cache=snapshot(cache);
    cacheStates_.push_back(next.cache);
    cache={};
    if(progress) progress("history",1,1);
    // Prefer keeping DiT weights over KV snapshots: reloading the FP32
    // checkpoint also converts and uploads several GB on every move. Retain
    // them only when the decoder has a conservative spatial memory reserve.
    reserveWorkingMemory(size_t(h)*w*32768 + (size_t(512)<<20));
    rt_.synchronize();
    if(progress) progress("decode",0,1);
    auto decoder=restore(decoderRt_,next.decoder);
    auto pixels=decoder_.decode(latent,3,lh,lw,decoder);
    next.decoder=snapshot(decoder);
    vaeStates_.push_back(next.decoder);
    decoder={};
    emit("pixels",pixels);
    Segment result{decoderRt_.download(pixels),next.segments?12:9,h,w};
    for(float value:result.pixels) if(!std::isfinite(value)) throw std::runtime_error("nonfinite LingBot output");
    next.segments++; next.firstPixels={};
    if(progress) progress("decode",1,1);
    world=std::move(next);
    return result;
}
}} // namespace fastllm::lingbot
