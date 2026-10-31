#pragma once
#include "models/lingbot_world/runtime.h"
#include "models/lingbot_world/vae.h"
#include <functional>

namespace fastllm { namespace lingbot {

using Pose = std::array<float,16>;
using Intrinsics = std::array<float,4>; // fx, fy, cx, cy at output resolution
Pose identityPose();
// OpenCV camera axes: right, down, forward. A local action updates c2w.
Pose movePose(const Pose &, const std::string &action, float distance, float angleDegrees);
std::vector<Pose> relativePoses(const std::vector<Pose> &, const Pose *previous = nullptr);
Tensor cameraEmbedding(Runtime &, const std::vector<Pose> &, const Intrinsics &, int height, int width);

struct SamplingStep {
    int timestep;
    float flowSigma, noiseSigma;
};
std::vector<SamplingStep> fastSchedule(float shift = 5.f,
    const std::vector<int> &indices = {0,250,500,750});
std::vector<float> prepareImage(const qwen_image::Image &,int height,int width);

struct HostVAEState;
struct HostDiTState;
struct WorldState {
    Tensor text, firstPixels;
    std::string prompt;
    std::shared_ptr<const HostDiTState> cache;
    std::shared_ptr<const HostVAEState> encoder,decoder;
    Pose pose=identityPose();
    Intrinsics intrinsics{};
    int height=0,width=0,segments=0;
    uint64_t seed=0,noiseSequence=0;
};
struct Segment {
    std::vector<float> pixels; // THWC RGB in [-1,1]
    int frames=0,height=0,width=0;
};
std::vector<uint8_t> encodeFramePNG(const Segment &, int frame);
using Progress = std::function<void(const std::string &,int,int)>;
// Optional sampling snapshots used by the official differential tests.
using Trace = std::function<void(const std::string &,const Tensor &)>;
class Pipeline {
public:
    Pipeline(const std::string &directory,int device,bool bf16Decode=true);
    std::unique_ptr<WorldState> create(const std::string &prompt,const std::vector<float> &firstPixels,
        int height,int width,uint64_t seed,const Intrinsics *intrinsics=nullptr,Progress progress={});
    // Calls are transactional. The caller serializes access to this pipeline.
    Segment step(WorldState &,const std::string &action="stay",const std::string *prompt=nullptr,
        Progress progress={},Trace trace={});
    Runtime &runtime() { return rt_; }
    // Release saved GPU state losslessly; all worlds and rollback snapshots
    // remain usable. Useful when yielding device memory to another model.
    void offloadState();
private:
    Tensor encodePrompt(const std::string &);
    void reserveWorkingMemory(size_t bytes);
    // Weak references permit pressure-driven eviction without extending the
    // lifetime of deleted worlds or snapshots from completed transactions.
    std::vector<std::weak_ptr<const HostDiTState>> cacheStates_;
    std::vector<std::weak_ptr<const HostVAEState>> vaeStates_;
    nomas::TensorPool tensorPool_;
    Runtime rt_;
    VAERuntime vaeRt_, decoderRt_;
    Tokenizer tokenizer_;
    TextEncoder text_;
    DiT dit_;
    VAE vae_, decoder_;
    std::vector<SamplingStep> schedule_;
};

}} // namespace fastllm::lingbot

extern "C" {
using LingBotProgress = int (*)(const char *stage,int current,int total,void *user);
void *fastllm_lingbot_pipeline_create(const char *directory,int device);
void *fastllm_lingbot_pipeline_create_ex(const char *directory,int device,int bf16Decode);
void fastllm_lingbot_pipeline_destroy(void *);
void *fastllm_lingbot_world_create_png(void *,const char *prompt,const uint8_t *png,size_t length,
    int height,int width,uint64_t seed,const float *intrinsics,LingBotProgress,void *user);
int fastllm_lingbot_world_destroy(void *pipeline,void *world);
void *fastllm_lingbot_world_step(void *pipeline,void *world,const char *action,const char *prompt,
    LingBotProgress,void *user);
int fastllm_lingbot_result_frames(void *result);
const uint8_t *fastllm_lingbot_result_png(void *result,int frame,size_t *length);
void fastllm_lingbot_result_destroy(void *result);
}
