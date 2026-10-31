#pragma once
#include "models/qwen_image/runtime.h"
#include <unordered_map>

namespace fastllm { namespace lingbot {

// FP32 encoding preserves image conditioning. BF16 decoding follows CUDA
// autocast: BF16 convolutions/attention, FP32 normalization and latent scaling.
class VAERuntime : public qwen_image::Runtime {
public:
    explicit VAERuntime(int device, bool bf16 = false)
        : qwen_image::Runtime(device), low_(bf16 ? new nomas::Runtime(device, false) : nullptr) {}
    nomas::Runtime &convolutionRuntime() { return low_ ? *low_ : *this; }
    nomas::Tensor cast(const nomas::Tensor &, bool full);
    nomas::Tensor resize2x(const nomas::Tensor &, int height, int width);
    nomas::Tensor channelNorm(const nomas::Tensor &, const nomas::Tensor &weight);
    nomas::Tensor spatialAttention(const nomas::Tensor &queries, const nomas::Tensor &keys,
                                  const nomas::Tensor &values);
    nomas::Tensor conv2d(const nomas::Tensor &, const nomas::Tensor &weight,
        const nomas::Tensor *bias, int height, int width, int kernel,
        int stride = 1, int pad = 0, int padEnd = 0);
    nomas::Tensor conv3d(const nomas::Tensor &, const nomas::Tensor &weight,
        const nomas::Tensor *bias, int frames, int height, int width,
        int kt, int kernel, int strideT = 1, int padT = 0);
    nomas::Tensor temporalShuffle(const nomas::Tensor &, int frames, int height, int width);
    nomas::Tensor latentScale(const nomas::Tensor &, bool decode);
    nomas::Tensor clampPixels(const nomas::Tensor &);
private:
    std::unique_ptr<nomas::Runtime> low_;
    std::unordered_map<const void *, std::pair<nomas::Tensor,nomas::Tensor>> packedWeights_;
    nomas::Tensor convolutionBF16(const nomas::Tensor &, const nomas::Tensor &,
        const nomas::Tensor *, int frames, int height, int width, int kt, int kernel,
        int strideT, int padT, int strideS, int padS, int outHeight, int outWidth);
};

struct VAEState {
    std::unordered_map<std::string, nomas::Tensor> features;
    int height = 0, width = 0;
    // -1 for arbitrary video; otherwise the processed image/zero-tail prefix,
    // saturated after the encoder's 113-frame temporal receptive field.
    int conditionFrames = -1;
};

class VAE {
public:
    using Tensor = nomas::Tensor;
    VAE(VAERuntime &rt, const std::string &checkpoint)
        : rt_(rt), weights_(rt, checkpoint), convWeights_(rt.convolutionRuntime(), checkpoint) {}
    // THWC rows: first encoder call consumes 1+4k frames, continuations 4k.
    // Decoder consumes any positive number of latent frames. Initial output
    // contains 1+4(k-1) frames, subsequent outputs contain 4k frames.
    Tensor encode(const Tensor &, int frames, int height, int width, VAEState &);
    // Initial image followed exclusively by zero frames. The image is required
    // only for an empty state. Stable encoder prefixes are reused exactly.
    Tensor encodeCondition(const Tensor &image, int latentFrames, int height, int width, VAEState &);
    Tensor decode(const Tensor &, int frames, int height, int width, VAEState &);
private:
    Tensor pointwise(const Tensor &, const std::string &);
    Tensor causal(const Tensor &, const std::string &, int frames, int h, int w, VAEState &);
    Tensor residual(const Tensor &, const std::string &, int frames, int h, int w, VAEState &);
    Tensor middle(const Tensor &, const std::string &, int frames, int h, int w, VAEState &);
    Tensor resample(const Tensor &, const std::string &, int &frames, int &h, int &w,
                    bool up, bool temporal, VAEState &);
    Tensor encodeChunk(const Tensor &, int frames, int h, int w, VAEState &);
    Tensor decodeChunk(const Tensor &, int h, int w, VAEState &);
    VAERuntime &rt_;
    nomas::Weights weights_, convWeights_;
};

}} // namespace fastllm::lingbot

extern "C" {
void *fastllm_lingbot_vae_create(const char *checkpoint, int device);
// Explicit autocast decoder entry used by precision comparisons.
void *fastllm_lingbot_vae_create_bf16(const char *checkpoint, int device);
void fastllm_lingbot_vae_destroy(void *);
int fastllm_lingbot_vae_reset(void *);
// Returns the number of output frames, or -1 on error. Capacities count floats.
int fastllm_lingbot_vae_encode(void *, int frames, int height, int width,
    const float *pixels, float *latents, size_t capacity);
int fastllm_lingbot_vae_decode(void *, int frames, int height, int width,
    const float *latents, float *pixels, size_t capacity);
}
