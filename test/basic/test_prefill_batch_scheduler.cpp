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
    std::atomic<int> cachedPrefix;
    mutable PagedCacheManager previewPool;
public:
    struct Event { std::vector<int> lengths; Clock::time_point time; bool prefill; };
    std::vector<Event> events;
    std::atomic<int> restores{0};
    mutable std::atomic<int> previews{0};
    std::atomic<bool> holdDecode{false}, decodeBlocked{false};
    void SetPrefix(int n) { cachedPrefix = n; }
    std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagers(int, bool) const override {
        return {{-1, &previewPool}};
    }
    int PreviewPagedPrefixCacheExtra(ResponseContext *, int length, PrefillPrefixPreview &) const override {
        ++previews;
        return std::min(length, cachedPrefix.load());
    }
    // Preview uses the real trie, while admission uses the deterministic
    // history restore below, without allocating model KV tensors.
    int QueryPagedPrefixCacheExtra(ResponseContext *, int) const override { return 0; }
    SchedulerModel(bool supported = true, int cachedPrefix = 0)
        : supported(supported), cachedPrefix(cachedPrefix) {
        block_cnt = 1;
        maxBatch = 8;
        tokensLimit = max_positions = 1 << 20;
        dataType = kvCacheDataType = FLOAT32;
        chunkedPrefillSize = 8192;
        eos_token_id = -100;
        canDoBatchForward = true;
        previewPool.pageLen = 128;
        previewPool.SetMaxPages(std::max(1, cachedPrefix / previewPool.pageLen));
        if (cachedPrefix > 0) {
            std::vector<int> pages;
            for (int i = 0; i < cachedPrefix / previewPool.pageLen; ++i)
                pages.push_back(previewPool.GetUnusedPageIndex(true));
            previewPool.Record(std::vector<int>(cachedPrefix, 1), pages);
            previewPool.ReleasePageIndices(pages);
        }
    }
    ~SchedulerModel() override {
        holdDecode = false;
        ShutdownRuntime();
        if (previewPool.trieRoot) previewPool.EvictTrieSubtree(previewPool.trieRoot);
        previewPool.trieRoot = nullptr;
    }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    bool NeedAttentionMask(int, int) override { return false; }
    bool SupportsPrefillBatching() const override { return supported; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    bool DeferHistoryCacheRestore() const override { return cachedPrefix > 0; }
    bool TryRestoreHistoryCache(std::vector<int> &tokens, int &length) override {
        ++restores;
        const int n = cachedPrefix;
        if (n <= 0 || tokens.size() <= (size_t)n) return false;
        length = n;
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
        if (!prefill) {
            decodeBlocked = true;
            while (holdDecode) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            decodeBlocked = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
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

static int Seed(SchedulerModel &model, int input = 8192) {
    int handle = Launch(model, input, 10000);
    Wait([&] { return model.Events().size() >= 3; });
    return handle;
}
static void Hold(SchedulerModel &model) {
    model.holdDecode = true;
    Wait([&] { return model.decodeBlocked.load(); });
}
static std::vector<SchedulerModel::Event> Prefills(SchedulerModel &model, size_t from = 0) {
    auto events = model.Events();
    std::vector<SchedulerModel::Event> result;
    for (size_t i = from; i < events.size(); ++i) if (events[i].prefill) result.push_back(events[i]);
    return result;
}

int main() {
    try {
#ifdef USE_CUDA
        FastllmCudaGetFreeSizes();
        FastllmCudaTrimBigBuffer();
#endif
        // Pure release policy: targets are OR conditions and deadlines belong
        // to requests; capacity and idle decode release without waiting.
        {
            PrefillBatchPolicy policy{16384, 20000, 16, 24, 8};
            auto t = Clock::now();
            auto decide = [&](int tokens, int requests, int active, bool full, int ms) {
                return std::string(policy.Decide(tokens, requests, active, full,
                                                 t, t + std::chrono::milliseconds(ms)));
            };
            Check(decide(16078, 2, 64, true, 1000) == "capacity", "full batch kept waiting");
            Check(decide(1, 1, 0, false, 0) == "idle", "idle waits");
            Check(decide(1, 1, 7, false, 0) == "low_decode", "low decode waits");
            Check(decide(1, 1, 8, false, 0) == "wait", "decode threshold off by one");
            Check(decide(16384, 1, 64, false, 0) == "tokens", "token target ignored");
            Check(decide(16, 16, 64, false, 0) == "requests", "request target ignored");
            Check(decide(1, 1, 64, false, 20000) == "timeout", "deadline ignored");
        }
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(16384, 5000, 16, 24, 8), "config rejected");
            auto begin = Clock::now(); Drain(model, {Launch(model, 1024)});
            Check(model.Events()[0].time - begin < std::chrono::milliseconds(300), "idle waited for timeout");
            Seed(model); begin = Clock::now(); Drain(model, {Launch(model, 1024)});
            Check(Clock::now() - begin < std::chrono::milliseconds(300), "low decode waited for timeout");
            Check(!model.SetPrefillBatching(8192, 0), "live policy mutation accepted");
        }
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(16384, 500), "config rejected"); Seed(model);
            Hold(model); const size_t start = model.Events().size();
            std::vector<int> handles;
            for (int i = 0; i < 3; ++i) handles.push_back(Launch(model, 4096));
            model.holdDecode = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            Check(Prefills(model, start).empty(), "partial batch ran early");
            auto ready = Clock::now(); handles.push_back(Launch(model, 4096)); Drain(model, handles);
            auto events = Prefills(model, start);
            Check(events.size() == 1 && events[0].lengths == std::vector<int>(4, 4096), "token target batch not packed");
            Check(events[0].time - ready < std::chrono::milliseconds(300), "full batch waited");
        }
        {
            SchedulerModel model;
            model.chunkedPrefillSize = 32768;
            model.kvMemConfig.enabled = true;
            model.kvMemConfig.prefillTokens = 8192;
            Check(model.SetPrefillBatching(16384, 20000), "config rejected");
            Check(model.GetChunkedPrefillSize() == 8192 && model.GetBatchedPrefillTokenLimit() == 32768,
                  "aggregate budget was confused with the KVMem per-request chunk");
        }
        {
            SchedulerModel model;
            model.chunkedPrefillSize = 32768;
            Check(model.SetPrefillBatching(16384, 500), "config rejected"); Seed(model);
            Hold(model); const size_t start = model.Events().size();
            auto a = Launch(model, 10000), b = Launch(model, 9000); model.holdDecode = false;
            Drain(model, {a, b}); auto events = Prefills(model, start);
            Check(events.size() == 1 && events[0].lengths == std::vector<int>({10000, 9000}),
                  "soft token target incorrectly became a hard packing limit");
        }
        // The next request cannot fit, even though the target was not reached.
        {
            SchedulerModel model;
            Check(model.SetPrefillBatching(10000, 200), "config rejected"); Seed(model);
            Hold(model); const size_t start = model.Events().size(); auto begin = Clock::now();
            auto a = Launch(model, 6000), b = Launch(model, 5000); model.holdDecode = false;
            Drain(model, {a, b}); auto events = Prefills(model, start);
            Check(events.size() == 2 && events[0].lengths == std::vector<int>{6000}, "capacity packing changed");
            Check(events[0].time - begin < std::chrono::milliseconds(100), "capacity-full batch waited");
        }
        // Request target controls release; the independent cap controls actual admission.
        for (bool target : {false, true}) {
            SchedulerModel model;
            Check(model.SetPrefillBatching(16384, 120, target ? 2 : 0, target ? 3 : 2, 1), "config rejected");
            Seed(model); Hold(model); const size_t start = model.Events().size();
            std::vector<int> handles;
            for (int i = 0; i < 3; ++i) handles.push_back(Launch(model, 512));
            auto begin = Clock::now(); model.holdDecode = false; Drain(model, handles);
            auto events = Prefills(model, start);
            Check(events[0].lengths.size() == (target ? 3 : 2), "request target/cap conflated");
            Check(events[0].time - begin < std::chrono::milliseconds(100), "request release waited");
        }
        // Repeated previews must not restore or pin queued prefixes; decode continues.
        {
            SchedulerModel model(true, 3072);
            Check(model.SetPrefillBatching(16384, 100), "config rejected"); Seed(model, 11264);
            Hold(model); const size_t start = model.Events().size(); const int restores = model.restores;
            auto begin = Clock::now(); int a = Launch(model, 4096); model.holdDecode = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
            Check(model.restores == restores && model.previews > 1, "waiting restored queued cache");
            Check(model.Events().size() > start + 3 && Prefills(model, start).empty(), "waiting stalled decode");
            int b = Launch(model, 5120); Drain(model, {a, b}); auto events = Prefills(model, start);
            Check(events.size() == 1 && events[0].lengths == std::vector<int>({1024, 2048}), "cached token budget or order wrong");
            Check(events[0].time - begin >= std::chrono::milliseconds(80) &&
                  events[0].time - begin < std::chrono::milliseconds(130), "arrival reset oldest deadline");
            Check(model.restores == restores + 2, "request restored more than once at admission");
        }
        // Cache eligibility may change during the wait. Actual input/positions
        // must follow a fresh restore, never the old preview's match length.
        {
            SchedulerModel model(true, 3072);
            Check(model.SetPrefillBatching(16384, 100), "config rejected"); Seed(model, 11264);
            size_t start = model.Events().size(); int h = Launch(model, 4096);
            Wait([&] { return model.previews > 2; }); model.SetPrefix(0);
            Drain(model, {h}); auto events = Prefills(model, start);
            Check(events.size() == 1 && events[0].lengths == std::vector<int>{4096}, "stale preview skipped input tokens");
        }
        {
            SchedulerModel model(true, 3072); model.chunkedPrefillSize = 2048;
            Check(model.SetPrefillBatching(16384, 5000), "config rejected");
            Drain(model, {Launch(model, 8192)}); auto events = model.Events();
            Check(events.size() == 3 && events[0].lengths == std::vector<int>{2048} &&
                  events[1].lengths == std::vector<int>{2048} && events[2].lengths == std::vector<int>{1024},
                  "cached long prefill lost chunking or positions");
        }
        {
            auto model = std::unique_ptr<SchedulerModel>(new SchedulerModel());
            Check(model->SetPrefillBatching(16384, 5000), "config rejected"); Seed(*model);
            size_t start = model->Events().size(); int h = Launch(*model, 1024);
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); model->AbortResponse(h);
            Wait([&] { int a, b, c; return !model->GetResponseStatistics(h, a, b, c); });
            Check(Prefills(*model, start).empty(), "cancelled waiting prompt ran");
            Launch(*model, 1024); auto begin = Clock::now(); model.reset();
            Check(Clock::now() - begin < std::chrono::seconds(1), "shutdown waited for batch deadline");
        }
        {
            SchedulerModel model(false);
            Check(!model.SetPrefillBatching(8192, 10), "unsupported model accepted policy");
            Check(model.SetPrefillBatching(0, 0), "default config rejected");
            Check(!model.SetPrefillBatching(0, 10) && !model.SetPrefillBatching(-1, 0) &&
                  !model.SetPrefillBatching(0, 0, 1) && !model.SetPrefillBatching(8192, 10, 4, 2),
                  "invalid config accepted");
            Drain(model, {Launch(model, 4096), Launch(model, 4096)});
            Check(model.Events().size() == 2, "unsupported model schedule changed");
        }
        std::cout << "ALL_PASS preview/admission, capacity, requests, idle, cache invalidation, deadlines, decode, cancellation, chunking\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
