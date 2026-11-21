#pragma once
#include "input.h"
#include "solarx.h"
#include "models/duo/model.h"
#include <atomic>
#include <mutex>
namespace fastllm::duo::live {
using Emit = std::function<void(json11::Json::object)>;
struct GpuLayout {
    int intent, speech, codec;
    std::vector<int> tilePeers;
    static GpuLayout fromProfile(const std::string &profile);
};

class Pipeline {
  public:
    Pipeline(std::string root, std::filesystem::path assets, int w, int h, GpuLayout layout, bool streamAvc = false,
             std::filesystem::path intentCache = {});
    void turn(const std::filesystem::path &input, const std::filesystem::path &dir, const Emit &emit);

    // Start a fresh conversation on resident weights. Receive one causal 17-frame block at a time.
    using NextBlock = std::function<bool(AVBlock &)>;
    void stream(const NextBlock &next, const std::filesystem::path &dir, const Emit &emit);

  private:
    std::string root;
    std::filesystem::path assets, intentCache;
    int w, h, global = 0;
    GpuLayout devices;
    bool streamAvc;
    Tokenizer tokens;
    InputWorker inputs;
    std::string prefixDescription;
    std::vector<float> prefixFeatures;
    int prefixLength = 0, prefixPosition = 0;
    std::unique_ptr<Intent> intent;
    std::unique_ptr<Session> real;
    std::unique_ptr<SolarX> solarx;
    std::unique_ptr<minimax_h3::Runtime> audioRuntime;
    std::unique_ptr<minimax_h3::AudioVAE> audio;
    std::atomic<bool> cancelled{false};
    void check() const;
    std::vector<float> prefix();
    using Plan = std::pair<std::vector<float>, std::vector<int>>;
    Plan plan(const Visual &visual, const std::vector<int> &body, bool hold, int index,
              const Intent::HiddenReady &waitReady = {});
    void run(const std::filesystem::path &dir, const Emit &emit, bool fresh,
             const std::function<bool(int, int, Plan &, const Intent::HiddenReady &)> &nextPlan);
};
} // namespace fastllm::duo::live
