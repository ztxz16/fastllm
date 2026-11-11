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

// Native MiniMax-H3-Image checkpoint: no Diffusers weight conversion required.
class Transformer {
  public:
    Transformer(Runtime &, Runtime &fp32, const std::string &model);
    void prepare(Progress = {});
    Tensor forward(const Tensor &packedLatents, const Tensor &text, int height, int width, int step,
                   Trace = {}, const Conditioning * = nullptr);
    const std::vector<float> &sigmas() const { return sigmas_; }
    void unload();

  private:
    Tensor attention(const Tensor &, const std::string &, const Tensor * = nullptr, const Tensor * = nullptr);
    Tensor feedForward(const Tensor &, const std::string &);
    Tensor refine(const Tensor &);
    Runtime &rt_, &full_;
    Weights weights_, fullWeights_, heads_;
    std::vector<float> times_, sigmas_;
    std::vector<Tensor> modulation_;
    Tensor finalModulation_;
};

class VAE {
  public:
    VAE(Runtime &compute, Runtime &fp32, const std::string &model);
    // Unnormalized latent pixels in HWC order; returns patch rows [H*W,3*16*16].
    Tensor decodeTile(const Tensor &, int height, int width, Trace = {});
    Image decode(const std::vector<float> &normalizedPackedLatents, int height, int width, Progress = {});
    // ImageNet pixels -> sampled, FP16-rounded, normalized CHW reference latent.
    minimax_h3::VideoTensor encode(const Image &, std::mt19937 &, Progress = {}, Trace = {});

  private:
    std::string model_;
    Runtime &rt_, &full_;
    Weights weights_, fullWeights_;
    std::vector<float> mean_, std_;
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
    Pipeline(const std::string &model, int device = 0, DType vaeType = DType::BF16);
    Image generate(const Options &, Progress = {}, Trace = {}, const std::vector<float> *text = nullptr,
                   const std::vector<float> *noise = nullptr, bool decode = true);

  private:
    std::string model_;
    Runtime rt_, full_, vaeRt_;
};
} // namespace fastllm::minimax_h3_image
