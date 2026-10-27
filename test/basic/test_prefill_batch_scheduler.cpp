#include "models/basellm.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace fastllm;
using Clock = std::chrono::steady_clock;

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

// Run the real request loop with deterministic, cheap forwards. This catches
// admission/deadline/cancellation bugs independently of GPU kernel timings.
class SchedulerModel : public basellm {
    std::mutex eventsMutex;
    bool supported;
    int cachedPrefix;
public:
    struct Event { std::vector<int> lengths; Clock::time_point time; bool prefill; };
    std::vector<Event> events;
    SchedulerModel(bool supported = true, int cachedPrefix = 0)
        : supported(supported), cachedPrefix(cachedPrefix) {
        block_cnt = 1;
        maxBatch = 8;
        tokensLimit = max_positions = 1 << 20;
        dataType = kvCacheDataType = FLOAT32;
        chunkedPrefillSize = 8192;
        eos_token_id = -100;
        canDoBatchForward = true;
    }
    ~SchedulerModel() override { ShutdownRuntime(); }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    bool NeedAttentionMask(int, int) override { return false; }
    bool SupportsPrefillBatching() const override { return supported; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    bool DeferHistoryCacheRestore() const override { return cachedPrefix > 0; }
    bool TryRestoreHistoryCache(std::vector<int> &tokens, int &length) override {
        if (cachedPrefix <= 0 || tokens.size() <= (size_t)cachedPrefix) return false;
        length = cachedPrefix;
        tokens.erase(tokens.begin(), tokens.begin() + length);
        return true;
    }
    void OnResponseContextCreated(ResponseContext *ctx) override {
        if (!ctx->cacheLen) return;
        ctx->preTokens = ctx->cacheLen;
        ctx->intParams["add_special_tokens"] = 0;
        ctx->intParams["promptLen"] = ctx->inputTokens;
        ctx->intParams["index"] = -1;
        for (Data *cache : {&ctx->pastKeyValues[0].first, &ctx->pastKeyValues[0].second}) {
            cache->Resize({1, ctx->cacheLen, 1});
            cache->expansionDims = cache->dims;
        }
    }
    std::vector<Event> Events() {
        std::lock_guard<std::mutex> lock(eventsMutex);
        return events;
    }
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &,
            const std::vector<Data *> &positions, const std::vector<int> &lengths,
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &,
            const LastTokensManager &, std::vector<std::vector<float> *> *) override {
        bool prefill = false;
        int offset = 0;
        for (int row = 0; row < batch; ++row) {
            const int old = kv[row].first->dims.empty() ? 0 : kv[row].first->dims[1];
            prefill |= lengths[row] > 1;
            Check(positions[row] && positions[row]->Count(0) == lengths[row], "position length mismatch");
            for (int j = 0; j < lengths[row]; ++j)
                Check(((float *)positions[row]->cpuData)[j] == old + j, "positions crossed request boundaries");
            for (Data *cache : {kv[row].first, kv[row].second}) {
                cache->Resize({1, old + lengths[row], 1});
                cache->expansionDims = cache->dims;
            }
            offset += lengths[row];
        }
        Check(offset == ids.Count(0), "packed token count mismatch");
        {
            std::lock_guard<std::mutex> lock(eventsMutex);
            events.push_back({lengths, Clock::now(), prefill});
        }
        if (!prefill) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return std::vector<int>(batch, 7);
    }
    int Forward(const Data &ids, const Data &, const Data &positions,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
            const LastTokensManager &last, std::vector<float> *) override {
        std::vector<std::pair<Data *, Data *>> refs{{&kv[0].first, &kv[0].second}};
        return ForwardBatch(1, ids, {}, {const_cast<Data *>(&positions)}, {(int)ids.Count(0)}, refs,
                            {config}, last, nullptr)[0];
    }
};

static int Launch(SchedulerModel &model, int input, int output = 1) {
    GenerationConfig config;
    config.output_token_limit = output;
    config.add_special_tokens = false;
    config.top_k = 1;
    return model.LaunchResponseTokens(std::vector<int>(input, 1), config);
}

template<class Predicate> static void Wait(Predicate ready) {
    auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        Check(Clock::now() < deadline, "scheduler made no progress");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void Drain(SchedulerModel &model, const std::vector<int> &handles, int count = 1) {
    for (int handle : handles) {
        for (int i = 0; i <= count; ++i) {
            Wait([&] { return model.CanFetchResponse(handle); });
            Check(model.FetchResponseTokens(handle) == (i < count ? 7 : -1), "request output changed");
        }
    }
}

int main() {
    try {
#ifdef USE_CUDA
        // CUDA context creation precedes scheduler admission in production.
        // Exclude that one-time initialization from the deadline assertions.
        FastllmCudaGetFreeSizes();
        FastllmCudaTrimBigBuffer();
#endif
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(16384, 500), "config rejected");
            std::vector<int> handles;
            for (int i = 0; i < 3; ++i) handles.push_back(Launch(model, 4096));
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            Check(model.Events().empty(), "partial batch ran before target/deadline");
            auto ready = Clock::now();
            handles.push_back(Launch(model, 4096));
            Drain(model, handles);
            auto events = model.Events();
            Check(events.size() == 1 && events[0].lengths == std::vector<int>(4, 4096), "four prompts were not packed");
            Check(events[0].time - ready < std::chrono::milliseconds(300), "full batch waited for timeout");
            Check(!model.SetPrefillBatching(8192, 0), "live policy mutation was accepted");
        }
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(16384, 60), "config rejected");
            auto begin = Clock::now();
            Drain(model, {Launch(model, 4096)});
            auto elapsed = model.Events()[0].time - begin;
            Check(elapsed >= std::chrono::milliseconds(45) && elapsed < std::chrono::seconds(2), "single request deadline violated");
        }
        {
            SchedulerModel model;
            model.maxBatch = 2;
            Check(model.SetPrefillBatching(16384, 500), "config rejected");
            auto begin = Clock::now();
            Drain(model, {Launch(model, 1024), Launch(model, 1024)});
            auto events = model.Events();
            Check(events.size() == 1 && events[0].lengths.size() == 2, "available slots were not filled");
            Check(events[0].time - begin < std::chrono::milliseconds(300), "capacity-full batch waited unnecessarily");
        }
        {
            SchedulerModel model(true, 3072);
            Check(model.SetPrefillBatching(8192, 80), "config rejected");
            auto begin = Clock::now();
            Drain(model, {Launch(model, 4096), Launch(model, 4096)});
            auto events = model.Events();
            Check(events.size() == 1 && events[0].lengths == std::vector<int>(2, 1024), "prefix hits corrupted batching/admission");
            Check(events[0].time - begin >= std::chrono::milliseconds(60), "cached tokens incorrectly filled target");
        }
        {
            SchedulerModel model(true, 3072);
            Check(model.SetPrefillBatching(16384, 100), "config rejected");
            int first = Launch(model, 4096);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            int second = Launch(model, 5120);
            Drain(model, {first, second});
            auto events = model.Events();
            Check(events.size() == 1 && events[0].lengths == std::vector<int>({1024, 2048}),
                  "prefix restoration changed admission order");
        }
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(8192, 80), "config rejected");
            int running = Launch(model, 8192, 128);
            Wait([&] { return model.Events().size() >= 3; });
            int pending = Launch(model, 512);
            auto before = model.Events().size();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            auto during = model.Events();
            Check(during.size() > before + 2, "decode stalled during prefill collection");
            for (size_t i = before; i < during.size(); ++i)
                Check(!during[i].prefill, "pending prefill ignored its deadline");
            Drain(model, {pending});
            Drain(model, {running}, 128);
        }
        {
            SchedulerModel model;
            model.chunkedPrefillSize = 2048;
            Check(model.SetPrefillBatching(16384, 500), "config rejected");
            Drain(model, {Launch(model, 4096), Launch(model, 4096)});
            auto events = model.Events();
            Check(events.size() == 4, "long prompt did not retain chunking");
            for (auto &event : events) Check(event.lengths == std::vector<int>{2048}, "long prompts combined across chunk limit");
        }
        {
            SchedulerModel model(true, 3072);
            model.chunkedPrefillSize = 2048;
            Check(model.SetPrefillBatching(16384, 500), "config rejected");
            Drain(model, {Launch(model, 8192)});
            auto events = model.Events();
            Check(events.size() == 3 && events[0].lengths == std::vector<int>{2048} &&
                  events[1].lengths == std::vector<int>{2048} && events[2].lengths == std::vector<int>{1024},
                  "cached prefix bypassed long-prefill chunking");
        }
        {
            auto model = std::unique_ptr<SchedulerModel>(new SchedulerModel());
            Check(model->SetPrefillBatching(16384, 5000), "config rejected");
            int handle = Launch(*model, 1024);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            model->AbortResponse(handle);
            Wait([&] { int a, b, c; return !model->GetResponseStatistics(handle, a, b, c); });
            Check(model->Events().empty(), "cancelled prompt ran");
            Launch(*model, 1024);
            auto begin = Clock::now();
            model.reset();
            Check(Clock::now() - begin < std::chrono::seconds(1), "shutdown waited for batching timeout");
        }
        {
            SchedulerModel model(false);
            Check(!model.SetPrefillBatching(8192, 10), "unsupported model accepted policy");
            Check(model.SetPrefillBatching(0, 0), "default config rejected");
            Check(!model.SetPrefillBatching(0, 10) && !model.SetPrefillBatching(-1, 0), "invalid config accepted");
            Drain(model, {Launch(model, 4096), Launch(model, 4096)});
            Check(model.Events().size() == 2, "unsupported model's schedule changed");
        }
        std::cout << "ALL_PASS prefill batching, cache hits, deadlines, active decode, cancellation, chunking, model isolation\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
