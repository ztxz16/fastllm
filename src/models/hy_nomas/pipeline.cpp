#include "models/hy_nomas/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>

namespace fastllm { namespace nomas {
namespace {
// One opt-in trace for identical-input reference tests. Disabled tracing must
// not download GPU tensors. The caller supplies an existing private directory.
class PipelineTrace {
    Runtime &rt_;
    std::string directory_,prefix_;
public:
    explicit PipelineTrace(Runtime &rt,const std::string &prefix={}):rt_(rt),prefix_(prefix) {
        if(const char *directory=std::getenv("FASTLLM_NOMAS_TRACE")) directory_=directory;
    }
    void operator()(const std::string &name,const std::vector<float> &values) const {
        if(directory_.empty()) return;
        std::ofstream file(directory_+"/"+prefix_+name+".f32",std::ios::binary);
        file.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
        if(!file) throw std::runtime_error("cannot write HY-Nomas pipeline trace");
    }
    void operator()(const std::string &name,const Tensor &values) const {
        if(!directory_.empty()) (*this)(name,rt_.download(values));
    }
};
}

Pipeline::Pipeline(const std::string &model,const std::string &text,int device):
    rt_(device),tokenizer_(text),text_(rt_,text),
    dit_(rt_,model+"/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"),
    decoder_(rt_,model+"/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors") {}

std::vector<uint8_t> Pipeline::generate(const std::string &prompt,int width,int height,int frames,
                                       uint64_t seed,int steps,Progress progress,void *user) {
    rt_.activate();
    if(prompt.empty() || prompt.size()>16384 || width<64 || height<64 || width>512 || height>512 ||
       width%32 || height%32 || frames<4 || frames>40 || frames%4 || steps<1 || steps>32 ||
       int64_t(frames/4)*(height/16)*(width/16)>16384)
        throw std::invalid_argument("baseline accepts a nonempty prompt, 64..512 pixel multiples of 32, 4..40 frames in multiples of 4, and 1..32 steps");
    auto update=[&](int stage,int current,int total) {
        if(progress && progress(stage,current,total,user)) throw std::runtime_error("generation cancelled");
    };
    PipelineTrace trace(rt_);
    try {
        // World sessions keep both networks resident between actions. Release
        // those weights before encoding an independent video prompt as well.
        dit_.unload(); decoder_.unload();
        update(1,0,1);
        int crop=0; auto ids=tokenizer_.videoPrompt(prompt,crop);
        auto features=text_.encode(ids).rowSlice(crop,int(ids.size())-crop);
        trace("features",features);
        text_.unload(); update(1,1,1);
        int t=frames/4,h=height/32,w=width/32,n=t*h*w;
        std::mt19937_64 generator(seed); std::normal_distribution<float> normal;
        std::vector<float> latent(size_t(n)*32);
        // DiT RNG is reproducible within this implementation. Precision
        // tests inject identical tensors, rather than assuming torch seed parity.
        for(auto &v:latent) v=normal(generator);
        trace("latent_0",latent);
        auto sigma=[steps](int i) { float s=1.f-float(i)/steps; return 5.f*s/(1.f+4.f*s); };
        for(int step=0;step<steps;++step) {
            update(2,step,steps);
            auto velocity=rt_.download(dit_.forward(rt_.upload(latent.data(),n,32),features,sigma(step)*1000.f,t,h,w));
            float dt=sigma(step+1)-sigma(step);
            for(size_t i=0;i<latent.size();++i) latent[i]+=velocity[i]*dt;
            trace("latent_"+std::to_string(step+1),latent);
        }
        dit_.unload(); update(2,steps,steps); update(3,0,1);
        auto noise=rt_.pixelNoise(seed,0,t,height,width);
        trace("pixel_noise",noise);
        auto pixels=decoder_.forward(rt_.upload(latent.data(),n,32),noise,t,h,w);
        trace("pixels",pixels);
        auto rgb=rt_.downloadPixelRGB(pixels,width,height,0,frames);
        decoder_.unload(); update(3,1,1);
        update(4,1,1); return rgb;
    } catch(...) {
        text_.unload(); dit_.unload(); decoder_.unload(); throw;
    }
}

std::vector<float> cameraTrajectory(int frames,int action) {
    if(frames<4 || frames>40 || frames%4 || action<0 || action>8)
        throw std::invalid_argument("invalid camera action or frame count");
    std::vector<float> poses(size_t(frames)*16,0.f);
    // Match prepare_camera_inputs -> _preprocess_camera, including the VAE's
    // first-frame-aware timeline. N output frames use N-3 camera poses, then
    // repeat pose 0 three times before grouping four matrices per latent.
    // Resampling preserves the full source duration (N-1 steps). The official
    // inference preset applies a 2.5x local rotation gain after normalization.
    const double angle=frames>4 ? 25.0/24.0*double(frames-1)/double(frames-4)*2.5*
                                  3.14159265358979323846/180.0 : 0.0;
    std::vector<double> displacement;
    double translationScale=1.0;
    if(frames>4 && action>=1 && action<=4) {
        // Upstream exports float32 absolute positions before interpolation,
        // then normalizes float64 framewise differences by their p99. Preserve
        // those rounding points instead of assuming every increment is one.
        std::vector<float> source(frames),resampled(frames-3);
        double position=0.0;
        for(int i=1;i<frames;++i) { position+=2.8/24.0; source[i]=float(position); }
        for(int i=0;i<frames-3;++i) {
            double sample=double(i)*double(frames-1)/double(frames-4);
            int left=std::min(int(sample),frames-2);
            resampled[i]=float(double(source[left])+(double(source[left+1])-source[left])*(sample-left));
        }
        for(int i=1;i<frames-3;++i)
            displacement.push_back(double(resampled[i])-resampled[i-1]);
        auto sorted=displacement; std::sort(sorted.begin(),sorted.end());
        double index=.99*double(sorted.size()-1); size_t lower=size_t(index);
        translationScale=sorted[lower]+(sorted[std::min(lower+1,sorted.size()-1)]-sorted[lower])*(index-lower);
    }
    for(int i=0;i<frames;++i) {
        float *p=poses.data()+i*16;
        p[0]=p[5]=p[10]=p[15]=1.f;
        if(i<4) continue;
        // These are normalized local increments, not absolute c2w poses.
        float distance=displacement.empty()?0.f:float(displacement[i-4]/translationScale);
        if(action==1 || action==2) p[11]=action==1?distance:-distance;
        if(action==3 || action==4) p[3]=action==4?distance:-distance;
        if(action==5 || action==6) {
            double a=action==6?angle:-angle;
            p[0]=p[10]=float(std::cos(a)); p[2]=float(std::sin(a)); p[8]=-p[2];
        }
        if(action==7 || action==8) {
            double a=action==7?angle:-angle;
            p[5]=p[10]=float(std::cos(a)); p[6]=-float(std::sin(a)); p[9]=-p[6];
        }
    }
    return poses;
}

std::unique_ptr<WorldState> Pipeline::createWorld(const std::string &prompt,int width,int height,uint64_t seed,int steps) {
    rt_.activate();
    if(prompt.empty() || prompt.size()>16384 || width<64 || width>512 || height<64 || height>512 || width%32 || height%32 || width*height>512*288 || steps<1 || steps>32)
        throw std::invalid_argument("world accepts a prompt, 64..512 pixel multiples of 32 (at most 147456 pixels), and 1..32 steps");
    auto state=std::make_unique<WorldState>();
    state->width=width; state->height=height; state->steps=steps; state->seed=seed;
    state->generator.seed(seed);
    state->features=encodeWorldPrompt(prompt); state->prompt=prompt;
    return state;
}

Tensor Pipeline::encodeWorldPrompt(const std::string &prompt) {
    if(prompt.empty() || prompt.size()>16384)
        throw std::invalid_argument("world prompt must be nonempty and at most 16384 bytes");
    // PE emits structured scene JSON. Allow it within the released 1280-token
    // text budget, checking before videoPrompt can silently truncate input.
    if(tokenizer_.encode(prompt).size()>1200)
        throw std::invalid_argument("world prompt exceeds 1200 encoded tokens; shorten the scene description");
    int crop=0; auto ids=tokenizer_.videoPrompt(prompt,crop);
    // Reuse the native encoder. Keep world KV/decoder history while switching
    // resident weights so that a live edit fits the same GPU memory budget.
    dit_.unload(); decoder_.unload();
    try {
        auto features=text_.encode(ids).rowSlice(crop,int(ids.size())-crop);
        // Encode the official camera-minimal continuation while Qwen is already
        // resident. Reuse it across worlds and edits, avoiding a weight reload
        // when switching between exploration and explicitly following text.
        if(!cameraContinuationFeatures_.data) {
            int continuationCrop=0;
            auto continuationIds=tokenizer_.videoPrompt("{\"segments\": [{}]}",continuationCrop);
            cameraContinuationFeatures_=text_.encode(continuationIds).rowSlice(
                continuationCrop,int(continuationIds.size())-continuationCrop);
        }
        text_.unload();
        rt_.synchronize(); // The returned world may next be used by another thread.
        return features;
    } catch(...) { text_.unload(); throw; }
}

std::vector<uint8_t> Pipeline::stepWorld(WorldState &world,int action,Progress progress,void *user,const std::string *prompt) {
    rt_.activate();
    if(action<0 || action>9 || world.frameOffset>=720)
        throw std::invalid_argument("invalid action or world reached its 120-second horizon; reset it");
    auto update=[&](int stage,int current,int total) {
        if(progress && progress(stage,current,total,user)) throw std::runtime_error("generation cancelled");
    };
    PipelineTrace trace(rt_,"step_"+std::to_string(world.frameOffset/5)+"_");
    // All state changes are transactional, including RNG and the KV commit.
    // A cancelled/failed step can be retried without advancing the world.
    WorldState next=world;
    const bool promptChanged=prompt && *prompt!=world.prompt;
    if(promptChanged) {
        update(1,0,1);
        next.features=encodeWorldPrompt(*prompt); next.prompt=*prompt;
        update(1,1,1);
    }
    const int t=5,h=world.height/32,w=world.width/32,n=t*h*w;
    std::normal_distribution<float> normal;
    std::vector<float> latent(size_t(n)*32);
    for(auto &v:latent) v=normal(next.generator);
    // "Follow the prompt" omits camera injection entirely. An identity
    // trajectory is a stationary-camera condition, not text-only generation.
    const bool hasCamera=action!=9;
    const bool cameraContinuation=hasCamera && world.frameOffset>0 && !promptChanged;
    const Tensor &segmentFeatures=cameraContinuation?cameraContinuationFeatures_:next.features;
    auto poses=hasCamera?cameraTrajectory(20,action):std::vector<float>();
    Tensor camera;
    if(hasCamera) camera=dit_.cameraEmbedding(rt_.upload(poses.data(),t,64));
    trace("camera",poses); trace("latent_0",latent);
    trace("features",segmentFeatures);
    auto sigma=[&](int i) { float s=1.f-float(i)/world.steps; return 5.f*s/(1.f+4.f*s); };
    for(int step=0;step<world.steps;++step) {
        update(2,step,world.steps+1);
        auto velocity=rt_.download(dit_.forward(rt_.upload(latent.data(),n,32),segmentFeatures,sigma(step)*1000.f,t,h,w,
                                              world.frameOffset,&next.cache,false,hasCamera?&camera:nullptr));
        float dt=sigma(step+1)-sigma(step);
        for(size_t i=0;i<latent.size();++i) latent[i]+=velocity[i]*dt;
        trace("latent_"+std::to_string(step+1),latent);
    }
    update(2,world.steps,world.steps+1);
    std::vector<float> cacheInput=latent;
    const float cacheSigma=.206897f;
    std::vector<float> cacheNoise(cacheInput.size());
    for(size_t i=0;i<cacheInput.size();++i) {
        cacheNoise[i]=normal(next.generator);
        cacheInput[i]=(1.f-cacheSigma)*cacheInput[i]+cacheSigma*cacheNoise[i];
    }
    trace("cache_noise",cacheNoise);
    trace("cache_input",cacheInput);
    dit_.forward(rt_.upload(cacheInput.data(),n,32),segmentFeatures,cacheSigma*1000.f,t,h,w,
                 world.frameOffset,&next.cache,true,hasCamera?&camera:nullptr);
    next.segmentTokens.push_back(n+segmentFeatures.rows);
    int remove=0;
    while(int(next.segmentTokens.size())>next.maxHistorySegments) {
        remove+=next.segmentTokens.front(); next.segmentTokens.erase(next.segmentTokens.begin());
    }
    if(remove) for(int i=0;i<54;++i) {
        int kept=next.cache.key[i].rows-remove;
        next.cache.key[i]=next.cache.key[i].rowSlice(remove,kept);
        next.cache.value[i]=next.cache.value[i].rowSlice(remove,kept);
    }
    update(2,world.steps+1,world.steps+1); update(3,0,1);
    std::vector<float> context=world.decoderLatents;
    int previous=int(context.size()/(size_t(h)*w*32));
    context.insert(context.end(),latent.begin(),latent.end());
    int decodeT=previous+t,begin=world.frameOffset-previous;
    auto noise=rt_.pixelNoise(world.seed,begin,decodeT,world.height,world.width);
    trace("decoder_latents",context);
    trace("pixel_noise",noise);
    auto pixels=decoder_.forward(rt_.upload(context.data(),decodeT*h*w,32),noise,decodeT,h,w);
    trace("pixels",pixels);
    auto rgb=rt_.downloadPixelRGB(pixels,world.width,world.height,previous*4,20);
    next.decoderLatents=std::move(latent); next.frameOffset+=t;
    update(3,1,1); update(4,1,1);
    world=std::move(next);
    return rgb;
}

}} // namespace fastllm::nomas
