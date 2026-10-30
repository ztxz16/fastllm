#pragma once

#include "models/hy_nomas/runtime.h"
#include <array>

namespace fastllm { namespace lingbot {

using Tensor = nomas::Tensor;
using Weights = nomas::Weights;

// Reuse continuous-tensor storage, BF16 projections and common activations.
// Wan keeps residuals and timestep/head projections in FP32 while its main
// projections use BF16. Tensor precision is explicit, independent of Nomas.
class Runtime : public nomas::Runtime {
public:
    explicit Runtime(int device);
    ~Runtime();
    Tensor cast(const Tensor &, bool full);
    Tensor join(const Tensor &, const Tensor &);
    Tensor affine(const Tensor &, const Tensor &scale, const Tensor &shift,
                  bool roundOnePlus = false);
    Tensor gated(const Tensor &, const Tensor &branch, const Tensor &gate);
    Tensor patch(const Tensor &, int frames, int height, int width, bool unpack = false);
    Tensor patchConv2d(const Tensor &, const Tensor &weight, const Tensor &bias,
                       int frames, int height, int width);
    Tensor rope3d(const Tensor &, int frames, int height, int width, int frameOffset);
    Tensor relativeAttention(const Tensor &, const Tensor &, const Tensor &, const Tensor &bias, int validKeys);
    Tensor t5Gelu(const Tensor &);
    Tensor t5Norm(const Tensor &, const Tensor &weight);
    Tensor wanRms(const Tensor &, const Tensor &weight);
    Tensor wanLayerNorm(const Tensor &, const Tensor *weight = nullptr, const Tensor *bias = nullptr);
    Tensor wanAttention(const Tensor &, const Tensor &, const Tensor &);
    Tensor wanLinearFP32(const Tensor &, const Tensor &weight, const Tensor &bias);
    Tensor normal(int rows, int cols, uint64_t seed, uint64_t sequence);
    Tensor flowToClean(const Tensor &latent, const Tensor &prediction, float sigma);
    Tensor addNoise(const Tensor &clean, const Tensor &noise, float sigma);
    nomas::Runtime fp32;
private:
    struct FullLinear;
    std::unique_ptr<FullLinear> fullLinear_;
    void *blas_ = nullptr;
};

class TextEncoder {
public:
    TextEncoder(Runtime &rt, const std::string &path) : rt_(rt), weights_(rt, path) {}
    Tensor encode(const std::vector<int> &tokens);
    void unload() { weights_.clear(); }
private:
    Runtime &rt_;
    Weights weights_;
};
class Tokenizer {
public:
    explicit Tokenizer(const std::string &directory);
    std::vector<int> encode(const std::string &text) const;
private:
    std::vector<int> word(const std::string &) const;
    std::unordered_map<std::string,std::pair<int,float>> vocab_;
    std::vector<std::pair<std::string,int>> special_;
    size_t maxPiece_ = 0;
    float unknownScore_ = -100.f;
    int unknown_ = 3;
};

struct Cache {
    std::vector<Tensor> keys, values, textKeys, textValues;
    int frames = 0;
    int window = 18;
    int sink = 6;
};

class DiT {
public:
    DiT(Runtime &, const std::string &directory);
    // Inputs are [F*H*W,C], where H/W are VAE latent dimensions.
    // camera, when provided, contains 6 Plucker channels folded over 8x8 pixels.
    Tensor forward(const Tensor &latent, const Tensor &condition, const Tensor &text,
                   const Tensor *camera, float timestep, int frames, int height,
                   int width, Cache &cache, bool commit);
    void unload() { weights_.clear(); full_.clear(); }
private:
    Tensor block(int layer, const Tensor &input, const Tensor &time,
                 const Tensor &context, const Tensor *camera, int frames,
                 int gridHeight, int gridWidth, Cache &cache, bool commit);
    Tensor linear(const Tensor &, const std::string &);
    Tensor fullLinear(const Tensor &, const std::string &);
    Tensor rms(const Tensor &, const std::string &);
    Runtime &rt_;
    Weights weights_, full_;
};

}} // namespace fastllm::lingbot

extern "C" {
void *fastllm_lingbot_dit_create(const char *directory, int device);
void fastllm_lingbot_dit_destroy(void *);
const char *fastllm_lingbot_error();
int fastllm_lingbot_dit_forward(void *, int frames, int height, int width,
    int textTokens, const float *latent, const float *condition, const float *text,
    const float *camera, float timestep, int commit, float *output);
int fastllm_lingbot_dit_reset(void *, int window, int sink);
void *fastllm_lingbot_text_create(const char *checkpoint, int device);
void fastllm_lingbot_text_destroy(void *);
int fastllm_lingbot_text_encode(void *, const int *tokens, int count, float *output);
void *fastllm_lingbot_tokenizer_create(const char *directory);
void fastllm_lingbot_tokenizer_destroy(void *);
int fastllm_lingbot_tokenize(void *, const char *text, int *output, int capacity);
}
