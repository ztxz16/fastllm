#pragma once

#include "models/hy_nomas/runtime.h"
#include <functional>

namespace fastllm {
namespace minimax_h3 {
using Tensor = nomas::Tensor;
using DType = nomas::DType;
using Weights = nomas::Weights;
using Trace = std::function<void(const std::string &, const Tensor &)>;

class Runtime : public nomas::Runtime {
  public:
    explicit Runtime(int device, bool fp32 = false) : nomas::Runtime(device, fp32) {}
    Runtime(int device, DType dtype) : nomas::Runtime(device, dtype) {}
    Tensor rms(const Tensor &, const Tensor &weight, float eps, int width = 0);
    Tensor indices(const std::vector<int> &);
    Tensor gather(const Tensor &, const Tensor &indices);
    void scatter(Tensor &out, const Tensor &input, const Tensor &indices);
    Tensor modulateRows(const Tensor &, const Tensor &table, const Tensor &indices, int shift, int scale);
    Tensor residualRows(const Tensor &, const Tensor &, const Tensor &table, const Tensor &indices, int gate);
    Tensor rotary(const Tensor &, const Tensor &cos, const Tensor &sin, int headDim);
    std::pair<Tensor, Tensor> ropeTable(const std::vector<float> &positions, int frequencies, float theta);
    Tensor timestepEmbedding(const std::vector<float> &times, int dimensions);
    Tensor selfAttention(const Tensor &q, const Tensor &k, const Tensor &v, int heads);
    Tensor conv1d(const Tensor &, const Tensor &weight, const Tensor *bias, int kernel, int stride = 1,
                  int padding = 0, int dilation = 1, bool transposed = false);
    Tensor weightNorm(const Tensor &weight, const Tensor &gain);
    Tensor transposeConvWeight(const Tensor &, int kernel);
    Tensor audioFilter(const Tensor &, const Tensor &filter, bool upsample);
    Tensor snake(const Tensor &, const Tensor &alpha, const Tensor &beta);
    Tensor affine(const Tensor &, float scale, float bias = 0, bool clamp = false);
    Tensor step(const Tensor &sample, const Tensor &velocity, float timestep, float sigma, float nextSigma);
};

struct TransformerConfig {
    int hidden = 5376, heads = 56, headDim = 128, layers = 50, refiners = 2, ffn = 14336;
    int videoDim = 96, audioDim = 32, textDim = 5120, freqDim = 256, timeDim = 2688, ropeFreq = 16;
    float normEps = 1e-5f, qkEps = 1e-5f, finalEps = 1e-5f, ropeTheta = 10000.f;
    explicit TransformerConfig(const std::string &directory);
};

struct PackedLayout {
    std::vector<int> video, audio, text, timeIndices, tags;
    std::vector<float> positions;
    void validate(int timeCount) const;
};

class Transformer {
  public:
    Transformer(Runtime &, Runtime &fp32, const std::string &directory);
    const TransformerConfig &config() const { return config_; }
    Tensor timeEmbedding(const std::vector<float> &times);
    // input is SiLU(temb), rounded to the main runtime's dtype after activation.
    Tensor projectModulation(int layer, const Tensor &input);
    Tensor block(int layer, const Tensor &, const Tensor &mod, const Tensor &indices, const Tensor &cos,
                 const Tensor &sin, Trace trace = {});
    // Precompute all timestep/modality tables, freeing the large AdaLN weights layer by layer.
    void prepare(const std::vector<std::vector<float>> &plans, std::function<void(int, int)> progress = {});
    std::pair<Tensor, Tensor> forward(const Tensor &video, const Tensor &audio, const Tensor &text,
                                      const PackedLayout &, Trace trace = {});
    void unload();

  private:
    Tensor attention(const Tensor &, const std::string &, const Tensor *cos = nullptr,
                     const Tensor *sin = nullptr, Trace trace = {});
    Tensor feedForward(const Tensor &, const std::string &);
    Tensor refine(const Tensor &);
    Runtime &rt_, &fp32_;
    Weights weights_, fullWeights_;
    TransformerConfig config_;
    std::vector<Tensor> modulation_;
    Tensor outputModulation_;
};
std::string componentWeights(const std::string &directory);
class AudioVAE {
  public:
    AudioVAE(Runtime &, const std::string &directory);
    // One mono channel at a time: [latent frames,32] -> [samples,1].
    Tensor decode(const Tensor &, Trace trace = {});
    void unload() {
        weights_.clear();
        fused_.clear();
    }

  private:
    Tensor conv(const Tensor &, const std::string &, int kernel, int stride = 1, int dilation = 1,
                bool transposed = false);
    Tensor activation(const Tensor &, const std::string &);
    Runtime &rt_;
    Weights weights_;
    std::unordered_map<std::string, Tensor> fused_;
};
struct VideoTensor {
    int frames = 0, height = 0, width = 0, channels = 0;
    std::vector<float> values;
    VideoTensor() = default;
    VideoTensor(int t, int h, int w, int c)
        : frames(t), height(h), width(w), channels(c), values(size_t(t) * h * w * c) {}
};
class TextEncoder {
  public:
    TextEncoder(Runtime &, const std::string &directory);
    Tensor encode(const std::vector<int> &, Trace trace = {});
    void unload() { weights_.clear(); }

  private:
    Runtime &rt_;
    Weights weights_;
};
class VideoVAE {
  public:
    VideoVAE(Runtime &, const std::string &directory);
    Tensor decodeTile(const Tensor &, int frames, int height, int width, Trace trace = {});
    VideoTensor decode(const VideoTensor &, std::function<void(int, int)> progress = {});
    void unload() { weights_.clear(); }

  private:
    VideoTensor decodeClip(const VideoTensor &, std::function<void()> checkpoint = {});
    Runtime &rt_;
    Weights weights_;
};
using Progress = std::function<void(const std::string &, int, int)>;
struct GenerateOptions {
    std::string prompt;
    int width = 768, height = 448, frames = 124, steps = 50;
    uint64_t seed = 0;
    void validate() const;
};
struct Generation {
    int width = 0, height = 0, frames = 0, fps = 24, sampleRate = 32000;
    std::vector<uint8_t> rgb;
    std::vector<float> audio; // Interleaved stereo.
};
PackedLayout textToVideoLayout(int textTokens, int latentFrames, int latentHeight, int latentWidth,
                               int audioFrames);
std::vector<float> sigmaSchedule(int steps, float shift);
class Pipeline {
  public:
    Pipeline(const std::string &directory, int device, bool fp32 = false);
    Generation generate(const GenerateOptions &, Progress progress = {}, Trace trace = {},
                        const std::vector<float> *initialVideo = nullptr,
                        const std::vector<float> *initialAudio = nullptr);

  private:
    Runtime rt_, full_;
    nomas::QwenTokenizer tokenizer_;
    TextEncoder text_;
    Transformer transformer_;
    VideoVAE video_;
    AudioVAE audio_;
    std::vector<float> videoMean_, videoStd_, audioMean_, audioStd_;
};
// FFmpeg is only a media encoder; model inference remains entirely C++/CUDA.
void writeMP4(const Generation &, const std::string &path, const std::string &ffmpeg = "ffmpeg");
} // namespace minimax_h3
} // namespace fastllm
