#pragma once
#include "models/minimax_h3/runtime.h"

namespace fastllm::minimax_h3_image {
using minimax_h3::DType;
using minimax_h3::Image;
using minimax_h3::Progress;
using minimax_h3::Runtime;
using minimax_h3::Tensor;
using minimax_h3::Trace;
using minimax_h3::Weights;

struct Conditioning {
    std::vector<int> textTags;
    std::vector<std::pair<int, int>> grids; // Reference latent height/width, in upload order.
};

Image readImage(const std::string &path);
Image decodeImage(const std::string &bytes);

// Shared by every stage, including the standalone VAE diagnostics. Explicit
// budgets use the same 1 GiB device / 2 GiB host allowances as the video model.
struct MemoryPolicy {
    MemoryPolicy(Runtime &, double gpuGiB = 0, double hostGiB = 0);
    bool layerStreaming = false;
    std::shared_ptr<nomas::TensorPool> pool;
    std::shared_ptr<nomas::HostMemoryBudget> host;
};

// Native MiniMax-H3-Image checkpoint: no Diffusers weight conversion required.
class Transformer {
  public:
    Transformer(Runtime &, Runtime &fp32, const std::string &model);
    void prepare(Progress = {});
    Tensor forward(const Tensor &packedLatents, const Tensor &text, int height, int width, int step,
                   Trace = {}, const Conditioning * = nullptr);
    const std::vector<float> &sigmas() const { return sigmas_; }
    void setMemoryPolicy(const MemoryPolicy &);
    // keepPrepared retains timestep tables and budgeted host weights, releasing GPU weights.
    void unload(bool keepPrepared = false);

  private:
    Tensor attention(const Tensor &, const std::string &, const Tensor * = nullptr, const Tensor * = nullptr);
    Tensor feedForward(const Tensor &, const std::string &);
    Tensor refine(const Tensor &);
    Runtime &rt_, &full_;
    Weights weights_, fullWeights_, heads_;
    std::vector<float> times_, sigmas_;
    std::vector<Tensor> modulation_;
    Tensor finalModulation_;
    bool lowMemory_ = false;
};

class VAE {
  public:
    VAE(Runtime &compute, Runtime &fp32, const std::string &model);
    // Unnormalized latent pixels in HWC order; returns patch rows [H*W,3*16*16].
    Tensor decodeTile(const Tensor &, int height, int width, Trace = {});
    // Independent tile sequences sharing the row-wise projections.
    Tensor decodeTiles(const Tensor &, int height, int width, int batch, Trace = {});
    Image decode(const std::vector<float> &normalizedPackedLatents, int height, int width, Progress = {});
    // ImageNet pixels -> sampled, FP16-rounded, normalized CHW reference latent.
    minimax_h3::VideoTensor encode(const Image &, std::mt19937 &, Progress = {}, Trace = {});
    void setMemoryPolicy(const MemoryPolicy &);

  private:
    std::string model_;
    Runtime &rt_, &full_;
    Weights weights_, fullWeights_;
    std::vector<float> mean_, std_;
    bool lowMemory_ = false;
    std::shared_ptr<nomas::HostMemoryBudget> hostBudget_;
    std::vector<std::pair<Tensor, Tensor>> packedQkv_;
    std::pair<Tensor, Tensor> rope_;
    std::pair<int, int> ropeShape_{};
    Tensor zero_;
};

struct Options {
    std::string prompt;
    int width = 2048, height = 2048;
    uint64_t seed = 42;
    std::vector<Image> images;
    int minGenerationArea = 1024; // CLI preserves exact-size diagnostic behavior.
    void validate() const;
    std::pair<int, int> canvas() const; // Width, height.
};

class Pipeline {
  public:
    Pipeline(const std::string &model, int device = 0, DType vaeType = DType::BF16, double gpuMemoryLimit = 0,
             double hostMemoryLimit = 0);
    const MemoryPolicy &memoryPolicy() const { return memory_; }
    Image generate(const Options &, Progress = {}, Trace = {}, const std::vector<float> *text = nullptr,
                   const std::vector<float> *noise = nullptr, bool decode = true);

  private:
    Image generateImpl(const Options &, Progress, Trace, const std::vector<float> *,
                       const std::vector<float> *, bool);
    std::string model_;
    Runtime rt_, full_, vaeRt_;
    MemoryPolicy memory_;
    std::unique_ptr<minimax_h3::TextEncoder> textEncoder_;
    std::unique_ptr<Transformer> transformer_;
    std::unique_ptr<VAE> decoder_;
    std::unique_ptr<nomas::QwenTokenizer> tokenizer_;
};
} // namespace fastllm::minimax_h3_image
