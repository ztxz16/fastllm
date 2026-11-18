#pragma once
#include "models/duo/layout.h"
#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <atomic>
#include <map>
#include <mutex>

namespace fastllm::duo {
using Tensor = minimax_h3::Tensor;
using Runtime = minimax_h3::Runtime;
using DType = minimax_h3::DType;
struct Config {
    int hidden = 5376, heads = 56, headDim = 128, layers = 50, refiners = 2, ffn = 14336;
    int text = 5120, freq = 256, time = 2688, rope = 16;
    float eps = 1e-5f, qkEps = 1e-5f, finalEps = 1e-5f, theta = 10000;
    explicit Config(const json11::Json &);
};
struct ModelSource {
    std::string weights, adapter, stage, format;
    json11::Json entry;
    int steps;
    float shift, loraScale = 1;
    ModelSource(const std::string &model, const std::string &importedAdapter = "");
};
struct Cache {
    std::vector<Group> groups;
    std::vector<std::pair<Tensor, Tensor>> layers;
    size_t bytes() const;
    void evict(Runtime &, int firstChunk);
};
Tensor splitQkv(Runtime &, const Tensor &, int heads, int headDim, int part);
Tensor swapHalves(Runtime &, const Tensor &);
Tensor flowUpdate(Runtime &, const Tensor &, const Tensor &, float amount);
Tensor flowNoise(Runtime &, const Tensor &, const Tensor &, float time);
Tensor uploadFp32(Runtime &, const std::vector<float> &, int rows, int cols);
class Model {
  public:
    Model(int device, const ModelSource &, bool lowMemory = false);
    ~Model();
    const Config &config() const;
    Runtime &runtime();
    void prepare();
    std::pair<Tensor, Tensor> forward(const Layout &, const Tensor &video, const Tensor &audio,
                                      const Tensor &refinerText, float time, const Cache &past,
                                      Cache *committed = nullptr, std::function<void()> checkpoint = {});

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct ChunkResult {
    std::vector<float> video, audio;
    int chunk = 0;
    int64_t firstFrame = 0, firstSample = 0, endSample = 0;
    size_t cacheBytes = 0;
    double seconds = 0;
};
// One worker owns CUDA execution. Cancellation is atomic and may arrive from
// another thread; a cancelled/failed session requires reset before reuse.
class Session {
  public:
    Session(int device, const ModelSource &, int height, int width, int historyChunks = 2,
            bool lowMemory = false);
    void reset(const std::vector<float> &reference, const std::vector<float> &shared, int sharedRows,
               bool reuseSystemPrefix = false);
    // A separate device may consume the immutable denoised media while clean KV is built.
    // The caller must wait for successful return before publishing the chunk.
    ChunkResult generate(int chunk, const std::vector<float> &intent, int rows, uint64_t seed,
                         const std::vector<std::vector<float>> &noise = {},
                         const std::function<void(const ChunkResult &)> &mediaReady = {});
    void cancel() {
        std::lock_guard<std::mutex> lock(commitMutex_);
        cancelled_.store(true);
    }

  private:
    void checkpoint() const;
    ModelSource source_;
    Model model_;
    int height_, width_, history_, sharedRows_ = 0;
    Tensor reference_, shared_;
    bool keepPrefix_ = false;
    Cache cache_, prefixCache_;
    std::vector<float> prefixReference_, prefixShared_;
    Timeline timeline_;
    std::atomic<bool> cancelled_{false};
    std::mutex mutex_;
    std::mutex commitMutex_;
};
json11::Json readJson(const std::string &path);
} // namespace fastllm::duo
