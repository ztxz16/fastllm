#pragma once

// Standalone continuous-tensor inference: CUDA/cuBLAS only, no Python runtime.
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "models/hy_nomas/runtime.h"

namespace fastllm {
namespace qwen_image {

using Tensor = nomas::Tensor;
using Weights = nomas::Weights;
using QwenTokenizer = nomas::QwenTokenizer;

class Runtime : public nomas::Runtime {
  public:
    explicit Runtime(int device);
    // Qwen's zero-centered RMSNorm and VAE channel norm.
    Tensor specialNorm(const Tensor &, const Tensor &weight, int mode, float eps = 1e-6f);
    Tensor scale(const Tensor &, const Tensor &mod, int firstTarget, int offset);
    Tensor gatedAdd(const Tensor &, const Tensor &, const Tensor &mod, int firstTarget, int offset);
    Tensor textRope(const Tensor &, const std::vector<int> &positions, float theta, int dim = 128);
    Tensor visionRope(const Tensor &, const std::vector<int> &positions, int dim = 72);
    Tensor geluExact(const Tensor &);
    Tensor conv2d(const Tensor &, const Tensor &weight, const Tensor *bias, int height, int width, int kernel,
                  int stride = 1, int pad = 0, int padEnd = 0);
    Tensor resize2x(const Tensor &, int height, int width);
    Tensor duplicateUpsample(const Tensor &, int height, int width, int channels, int temporalFactor);
    Tensor averageDownsample(const Tensor &, int height, int width, int channels, int spatialFactor,
                             int temporalFactor);
};

struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgba;
};
struct Condition {
    Tensor features, latents;
    std::vector<int> imageSlots;
    std::vector<std::pair<int, int>> shapes;
};
struct KVCache {
    std::vector<Tensor> keys, values;
};
using Progress = std::function<void(const std::string &, int, int)>;
// Optional FP32 snapshots for differential tests; the server leaves this unset.
using Trace = std::function<void(const std::string &, const std::vector<float> &)>;

class TextEncoder {
  public:
    TextEncoder(Runtime &, const std::string &directory);
    Tensor encode(const std::vector<int> &tokens, const std::vector<Image> &images = {});
    void unload() { weights_.clear(); }

  private:
    std::vector<Tensor> vision(const Image &image);
    Runtime &rt_;
    Weights weights_;
};

class DiT {
  public:
    DiT(Runtime &, const std::string &directory);
    Tensor forward(const Tensor &latents, const Condition &, float timestep, int height, int width,
                   KVCache *cache = nullptr);
    void unload() { weights_.clear(); }

  private:
    Runtime &rt_;
    Weights weights_;
};

class VAE {
  public:
    VAE(Runtime &, const std::string &directory);
    Tensor decode(const Tensor &latents, int height, int width);
    Tensor encode(const Tensor &pixels, int height, int width);
    void unload() { weights_.clear(); }

  private:
    Tensor conv(const Tensor &, const std::string &, int h, int w, int stride = 1, int padEnd = 0);
    Tensor residual(const Tensor &, const std::string &, int h, int w);
    Tensor middle(const Tensor &, const std::string &, int h, int w);
    Runtime &rt_;
    Weights weights_;
};

struct GenerateOptions {
    std::string prompt;
    int width = 1024, height = 1024, steps = 40;
    uint64_t seed = 0;
    std::vector<Image> images;
    int referenceResolution = 1024;
    void validate() const;
};
class Pipeline {
  public:
    Pipeline(const std::string &directory, int device);
    Image generate(const GenerateOptions &, Progress progress = {}, Trace trace = {});

  private:
    Runtime rt_;
    QwenTokenizer tokenizer_;
    TextEncoder text_;
    DiT dit_;
    VAE vae_;
    std::vector<float> means_, stds_;
    double baseShift_ = .5, maxShift_ = .9, baseLength_ = 256, maxLength_ = 8192, terminal_ = .02;
};

std::string componentWeights(const std::string &directory);
std::vector<uint8_t> encodePNG(const Image &);
Image decodePNG(const std::string &);
Image resizeImage(const Image &, int width, int height);

} // namespace qwen_image
} // namespace fastllm
