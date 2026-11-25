#include "models/basellm.h"
#include "models/qwen4_exp.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace fastllm;

namespace {
void Check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

enum class CacheView { Contiguous, Paged, TensorParallel };

// Use the real scheduler and page allocator, with constant non-EOS output.
// The TP view mirrors Qwen4's root: workers own pages, expansionDims counts
// their occupied space, and the root owns no extra page references.
class SchedulerModel : public basellm {
    CacheView view;
    PagedCacheManager pools[2];
    std::map<const Data *, std::vector<int>> workerPages;
    std::mutex cacheMutex, gateMutex;
    std::condition_variable gate;
    bool holdForward = false;
    bool pauseConcurrentDecode = false, insideConcurrentDecode = false;
    int activeCaches = 0;
public:
    int peakActiveCaches = 0;
    bool sawMultiRowBatch = false;
    void EnableSharedPool(int perRequest) {
        kvMemConfig.enabled = true;
        kvMemConfig.maxTokens = max_positions = perRequest;
    }
    bool SupportsKvMemBatch() const override { return true; }
    bool SupportsKvMemPrefixCache() const override { return true; }

    SchedulerModel(CacheView view, int capacity, int pageSize = 128, int batch = 1,
                   const basellm *capabilities = nullptr)
        : view(view) {
        block_cnt = 1;
        kvCacheId = 0;
        maxBatch = batch;
        tokensLimit = capacity;
        max_positions = 4096; // The configured budget must also bound generation.
        dataType = kvCacheDataType = FLOAT32;
        eos_token_id = -100;
        canDoBatchForward = capabilities ? capabilities->canDoBatchForward : false;
        canDoConcurrentForward = capabilities ? capabilities->canDoConcurrentForward : true;
        use_new_engine = false;
        for (auto &pool : pools) {
            pool.type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
            pool.pageLen = pageSize;
            pool.SetMaxPages((capacity + pageSize - 1) / pageSize);
        }
    }
    ~SchedulerModel() override {
        Resume();
        ShutdownRuntime();
    }
    std::string MakeInput(const std::string &, int, const std::string &input) override { return input; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &output) override {
        return output;
    }
    bool UseGenericHistoryCache() const override { return false; }
    bool NeedAttentionMask(int, int) override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    PagedCacheManager *GetPagedKVCacheManager(int layer, bool key) const override {
        return view == CacheView::Contiguous || layer != 0
            ? nullptr : const_cast<PagedCacheManager *>(&pools[key ? 0 : 1]);
    }
    void Pause() {
        std::lock_guard<std::mutex> lock(gateMutex);
        holdForward = true;
    }
    void PauseNextConcurrentDecode() {
        std::lock_guard<std::mutex> lock(gateMutex);
        pauseConcurrentDecode = true;
    }
    void WaitForConcurrentDecode() {
        std::unique_lock<std::mutex> lock(gateMutex);
        Check(gate.wait_for(lock, std::chrono::seconds(10), [&] { return insideConcurrentDecode; }),
              "fixture did not enter concurrent decode");
    }
    void Resume() {
        std::lock_guard<std::mutex> lock(gateMutex);
        holdForward = false;
        gate.notify_all();
    }
    int Forward(const Data &ids, const Data &, const Data &,
                std::vector<std::pair<Data, Data>> &cache,
                const GenerationConfig &, const LastTokensManager &, std::vector<float> *) override {
        return ForwardRow(ids.Count(0), cache[0].first, cache[0].second);
    }
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &,
            const std::vector<Data *> &, const std::vector<int> &lengths,
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &,
            const LastTokensManager &, std::vector<std::vector<float> *> *) override {
        Check(lengths.size() == batch && kv.size() == batch, "expected one KV pair per request");
        sawMultiRowBatch |= batch > 1;
        int total = 0;
        std::vector<int> result;
        for (int row = 0; row < batch; ++row) {
            total += lengths[row];
            result.push_back(ForwardRow(lengths[row], *kv[row].first, *kv[row].second));
        }
        Check(total == ids.Count(0), "packed batch token count mismatch");
        return result;
    }
private:
    int ForwardRow(int tokens, Data &key, Data &value) {
        bool concurrentDecode;
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            concurrentDecode = activeCaches >= 2 && !key.dims.empty();
        }
        {
            std::unique_lock<std::mutex> lock(gateMutex);
            if (pauseConcurrentDecode && concurrentDecode) {
                pauseConcurrentDecode = false;
                insideConcurrentDecode = holdForward = true;
                gate.notify_all();
            }
            Check(gate.wait_for(lock, std::chrono::seconds(10), [&] { return !holdForward; }),
                  "scheduler fixture was not resumed");
        }
        std::lock_guard<std::mutex> lock(cacheMutex);
        const bool fresh = key.dims.empty();
        const int length = (fresh ? 0 : key.dims[1]) + tokens;
        Check(length <= tokensLimit, "scheduler exceeded the logical token limit");
        if (fresh) peakActiveCaches = std::max(peakActiveCaches, ++activeCaches);
        for (int component = 0; component < 2; ++component) {
            Data &data = component == 0 ? key : value;
            data.Resize({1, length, 1});
            if (view == CacheView::Contiguous) {
                data.expansionDims = {1, tokensLimit, 1};
                continue;
            }
            auto &pool = pools[component];
            auto &pages = view == CacheView::Paged ? data.pageIndex : workerPages[&data];
            const int needed = (length + pool.pageLen - 1) / pool.pageLen;
            {
                std::lock_guard<std::mutex> lock(pool.pageIndexLocker);
                Check(pool.FreePageCount() >= needed - (int)pages.size(),
                      "scheduler oversubscribed the physical page pool");
            }
            while ((int)pages.size() < needed) pages.push_back(pool.GetUnusedPageIndex(true));
            if (view == CacheView::Paged) {
                data.isPagedKVCache = true;
                data.pagedKVCacheData = &pool;
                data.pageLen = pool.pageLen;
                data.lastPageLen = (length - 1) % pool.pageLen + 1;
            } else {
                data.expansionDims = {1, needed * pool.pageLen, 1};
                data.cudaDataBorrowed = true;
            }
        }
        return 7;
    }
public:
    void OnResponseContextRemoved(ResponseContext *context) override {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (!context->pastKeyValues[0].first.dims.empty()) --activeCaches;
        if (view != CacheView::TensorParallel) return;
        for (int component = 0; component < 2; ++component) {
            Data *data = component == 0 ? &context->pastKeyValues[0].first : &context->pastKeyValues[0].second;
            auto found = workerPages.find(data);
            if (found != workerPages.end()) {
                pools[component].ReleasePageIndices(found->second);
                workerPages.erase(found);
            }
        }
    }
    void CheckReleased() {
        std::lock_guard<std::mutex> lock(cacheMutex);
        Check(activeCaches == 0 && workerPages.empty(), "request cache was not released");
        for (auto &pool : pools) Check(pool.FreePageCount() == pool.maxPages, "request leaked pages");
    }
};

// Exercise the production scheduler with actual page ownership. Forward is cheap and
// its next token depends on the complete input prefix, including earlier outputs.
class ReplayModel : public SchedulerModel {
    int stopAt;
    std::map<const Data *, uint32_t> state;
    int Next(const Data &key, const float *tokens, int count) {
        uint32_t &hash = state[&key];
        if (key.dims[1] == count) hash = 0;
        for (int i = 0; i < count; ++i) hash = hash * 31 + (int)tokens[i];
        return key.dims[1] >= stopAt ? eos_token_id : 1 + hash % 251;
    }
public:
    int retractions = 0;
    bool cancelOnReplay = false;
    std::atomic<int> victimIndex{-1};
    ReplayModel(int stopAt, int reserve) : SchedulerModel(CacheView::Paged, 256, 8, 8), stopAt(stopAt) {
        max_positions = 128;
        canDoBatchForward = true;
        chunkedPrefillSize = 16;
        Check(SetPrefillBatching(32, 0) && SetPrefillScheduling(false, 0, true, 0, 0) &&
              SetKVAdmissionReserve(reserve), "KV replay configuration failed");
    }
    bool SupportsKVReplay() const override { return true; }
    bool SupportsSharedPagedKVPool() const override { return true; }
    bool SupportsPrefillBatching() const override { return true; }
    bool SupportsChunkedPrefillScheduling() const override { return true; }
    void ReleaseKVForReplay(ResponseContext *c) override {
        if (++retractions == 1 && cancelOnReplay) {
            Pause();
            victimIndex.store(c->allTokens.front() - 1);
        }
        OnResponseContextRemoved(c);
        basellm::ReleaseKVForReplay(c);
    }
    int Forward(const Data &ids, const Data &mask, const Data &positions,
                std::vector<std::pair<Data, Data>> &cache, const GenerationConfig &cfg,
                const LastTokensManager &hist, std::vector<float> *logits) override {
        SchedulerModel::Forward(ids, mask, positions, cache, cfg, hist, logits);
        return Next(cache[0].first, (const float *)ids.cpuData, ids.Count(0));
    }
    std::vector<int> ForwardBatch(int n, const Data &ids, const std::vector<Data *> &masks,
            const std::vector<Data *> &positions, const std::vector<int> &lengths,
            std::vector<std::pair<Data *,Data *>> &kv, const std::vector<GenerationConfig> &cfg,
            const LastTokensManager &hist, std::vector<std::vector<float> *> *logits) override {
        auto out = SchedulerModel::ForwardBatch(n, ids, masks, positions, lengths, kv, cfg, hist, logits);
        int offset = 0;
        for (int i = 0; i < n; ++i) {
            out[i] = Next(*kv[i].first, (const float *)ids.cpuData + offset, lengths[i]);
            offset += lengths[i];
        }
        return out;
    }
};

void ReplayPressure(int stopAt, int reserve, bool cancel = false) {
    constexpr int count = 8, prompt = 12;
    ReplayModel model(stopAt, reserve);
    model.cancelOnReplay = cancel;
    model.Pause();
    GenerationConfig cfg;
    cfg.output_token_limit = 110;
    cfg.add_special_tokens = false;
    std::vector<int> handles;
    for (int i = 0; i < count; ++i)
        handles.push_back(model.LaunchResponseTokens(std::vector<int>(prompt, i+1), cfg));
    model.Resume();
    std::vector<uint32_t> hashes(count);
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < prompt; ++j) hashes[i] = hashes[i] * 31 + i + 1;
    std::vector<int> emitted(count);
    std::vector<bool> done(count);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (std::find(done.begin(), done.end(), false) != done.end()) {
        Check(std::chrono::steady_clock::now() < deadline, "replayed request stalled");
        const int victim = model.victimIndex.load();
        if (victim >= 0 && !done[victim]) {
            int hit, miss, generated;
            Check(model.GetResponseStatistics(handles[victim], hit, miss, generated) &&
                  hit == 0 && miss == prompt && generated >= emitted[victim] && generated > 0,
                  "retraction reset public usage");
            model.AbortResponse(handles[victim]);
            done[victim] = true;
            model.Resume();
        }
        for (int i = 0; i < count; ++i) {
            if (done[i] || !model.CanFetchResponse(handles[i])) continue;
            const int token = model.FetchResponseTokens(handles[i]);
            if (token < 0) {
                Check(token == -1 && emitted[i] == std::min(stopAt-prompt, cfg.output_token_limit),
                      "replay lost output or changed stop condition");
                done[i] = true;
            } else {
                Check(token == (int)(1 + hashes[i] % 251), "replay changed streamed token");
                hashes[i] = hashes[i] * 31 + token;
                ++emitted[i];
            }
        }
        std::this_thread::yield();
    }
    model.CheckReleased();
    Check((model.retractions > 0) == (reserve > 0 && stopAt > 32), "expected pressure behavior");
    if (reserve && stopAt <= 32) Check(model.peakActiveCaches > 2, "reduced reservation did not admit more requests");
}

int Launch(SchedulerModel &model, int prompt, int output) {
    GenerationConfig config;
    config.output_token_limit = output;
    config.add_special_tokens = false;
    return model.LaunchResponseTokens(std::vector<int>(prompt, 1), config);
}

std::vector<int> Drain(SchedulerModel &model, const std::vector<int> &handles, int endToken = -1) {
    std::vector<int> counts(handles.size(), 0);
    std::vector<bool> done(handles.size(), false);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::find(done.begin(), done.end(), false) != done.end()) {
        Check(std::chrono::steady_clock::now() < deadline, "scheduler made no progress");
        for (size_t i = 0; i < handles.size(); ++i) {
            if (done[i] || !model.CanFetchResponse(handles[i])) continue;
            const int token = model.FetchResponseTokens(handles[i]);
            if (token < 0) {
                Check(token == endToken, "unexpected request termination");
                done[i] = true;
            } else {
                Check(token == 7, "unexpected output token");
                ++counts[i];
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    model.CheckReleased();
    return counts;
}

void CompletePagesOnly() {
    SchedulerModel model(CacheView::Paged, 256);
    ResponseContext ctx;
    ctx.Init(1, FLOAT32, FLOAT32);
    ctx.allTokens.assign(128, 7); // Includes a sampled token whose KV is absent.
    for (int part = 0; part < 2; ++part) {
        auto *pool = model.GetPagedKVCacheManager(0, part == 0);
        auto &cache = part ? ctx.pastKeyValues[0].second : ctx.pastKeyValues[0].first;
        cache.isPagedKVCache = true; cache.pagedKVCacheData = pool;
        cache.pageLen = 128; cache.lastPageLen = 127;
        cache.pageIndex.push_back(pool->GetUnusedPageIndex(true));
    }
    ctx.TryRecordPagedCache(&model);
    std::vector<int> pages;
    auto *pool = model.GetPagedKVCacheManager(0, true);
    pool->Query(ctx.allTokens, pages);
    Check(pages.empty(), "recorder published an unwritten tail token");
    ctx.pastKeyValues[0].first.lastPageLen = ctx.pastKeyValues[0].second.lastPageLen = 128;
    ctx.allTokens.push_back(8);
    ctx.TryRecordPagedCache(&model);
    pool->Query(ctx.allTokens, pages);
    Check(pages.size() == 1, "recorder lost a complete page");
}

void OverlappingPageViews() {
    PagedCacheManager pools[2];
    for (auto &pool : pools) { pool.pageLen = 128; pool.SetMaxPages(64); }
    std::vector<int> first{9, 2, 4, 7, 12}, second{9, 1}, held{9};
    pools[0].Pick(first); pools[1].Pick(second); pools[0].Pick(held);
    {
        ResponseContext context;
        context.pastKeyValues.resize(1);
        auto configure = [&](Data &view, int pool, std::vector<int> pages) {
            view.isPagedKVCache = true; view.pagedKVCacheData = &pools[pool];
            view.pageLen = view.lastPageLen = 128; view.pageIndex = std::move(pages);
        };
        auto &root = context.pastKeyValues[0].first;
        configure(root, 0, {9, 2, 9, 4}); root.multiDeviceData = true;
        for (int rank = 0; rank < 3; ++rank) root.multiDeviceDatas[rank] = new Data();
        configure(*root.multiDeviceDatas[0], 0, {4, 7, 2});
        configure(*root.multiDeviceDatas[1], 1, {9, 1, 9});
        configure(*root.multiDeviceDatas[2], 0, {7, 12});
    }
    Check(pools[0].pageRefCount[9] == 1, "root/local aliases released another request's page");
    for (int page : {2, 4, 7, 12}) Check(pools[0].pageRefCount[page] == 0, "overlap leaked a page");
    Check(pools[1].pageRefCount[9] == 0 && pools[1].pageRefCount[1] == 0, "different pools were deduplicated together");
    Check(std::vector<int>(pools[0].freePages.end()-4,pools[0].freePages.end()) == std::vector<int>({2,4,7,12}),
          "page release order changed");
    pools[0].ReleasePageIndices(held);
    Check(pools[0].FreePageCount() == 64 && pools[1].FreePageCount() == 64, "page pool partition changed");
}

void Single(CacheView view, int capacity, int pageSize, int prompt, int output, int expected) {
    SchedulerModel model(view, capacity, pageSize);
    const int actual = Drain(model, {Launch(model, prompt, output)})[0];
    std::cout << "capacity=" << capacity << " page=" << pageSize << " view=" << (int)view
              << " prompt=" << prompt << " requested=" << output << " output=" << actual << '\n';
    Check(actual == expected, "premature termination: expected " + std::to_string(expected) +
          " tokens, got " + std::to_string(actual));
}

void Concurrent(CacheView view, bool pressure, const basellm *capabilities = nullptr) {
    SchedulerModel model(view, 1024, 128, 2, capabilities);
    model.Pause();
    int first = Launch(model, 256, pressure ? 1500 : 320);
    int second = Launch(model, 256, pressure ? 1500 : 192);
    model.Resume();
    const auto counts = Drain(model, {first, second});
    std::cout << "CONCURRENT_RESULT view=" << (int)view << " pressure=" << pressure
              << " output=" << counts[0] << ',' << counts[1] << '\n';
    Check(model.peakActiveCaches == 2, "fixture did not exercise concurrent requests");
    if (capabilities && capabilities->canDoBatchForward)
        Check(model.sawMultiRowBatch, "fixture did not run a multi-request forward");
    if (!pressure) Check(counts == std::vector<int>({320, 192}), "concurrent request ended early");
    else Check(counts[0] > 0 && counts[1] > 0 && counts[0] <= 768 && counts[1] <= 768,
               "pool pressure violated the context limits");
    // Reuse the same pool after the concurrent requests have been removed.
    Check(Drain(model, {Launch(model, 256, 768)})[0] == 768, "pool reuse lost capacity");
    std::cout << "CONCURRENT_PASS view=" << (int)view << " pressure=" << pressure << '\n';
}

// A slow multi-request Forward must not hold the request dictionary mutex.
// Exercise cancellation and admission while the scheduler still owns KV.
void ConcurrentConsumers() {
    SchedulerModel model(CacheView::Paged, 4096, 128, 4);
    model.Pause();
    model.PauseNextConcurrentDecode();
    const int first = Launch(model, 128, 16), second = Launch(model, 128, 16);
    model.Resume();
    model.WaitForConcurrentDecode();
    auto consumer = std::async(std::launch::async, [&] {
        int cached, missed, output;
        Check(model.GetResponseStatistics(first, cached, missed, output), "lost active request");
        Check(model.CanFetchResponse(first), "published output is unavailable");
        model.AbortResponse(second);
        return Launch(model, 128, 9);
    });
    const bool responsive = consumer.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    model.Resume(); // Always release Forward, including when testing the old bug.
    const int third = consumer.get();
    Check(responsive, "concurrent Forward blocked consumers, cancellation or admission");
    Check(Drain(model, {first, third}) == std::vector<int>({16, 9}),
          "concurrent cancellation corrupted another request");
    Check(model.FetchResponseTokens(second) == -1, "aborted request remained active");
    std::cout << "CONCURRENT_CONSUMERS_PASS\n";
}

void PagedBatchLimit() {
    SchedulerModel model(CacheView::Paged, 8192, 128, 2);
    model.Pause();
    model.PauseNextConcurrentDecode();
    std::vector<int> handles;
    for (int i = 0; i < 4; ++i) handles.push_back(Launch(model, 128, 16));
    model.Resume();
    model.WaitForConcurrentDecode();
    const int firstCohort = model.peakActiveCaches;
    model.Resume();
    Check(firstCohort == 2, "paged requests without expansionDims bypassed maxBatch");
    Check(Drain(model, handles) == std::vector<int>(4, 16), "queued paged requests did not finish");
    Check(model.peakActiveCaches == 2, "an undrained request lost its KV slot");
}

void SharedPoolPressure() {
    SchedulerModel model(CacheView::Paged, 1024, 128, 4);
    model.EnableSharedPool(768);
    model.Pause();
    std::vector<int> handles;
    // Each request fits, but their future decode pages cannot all fit together.
    for (int i = 0; i < 6; ++i) handles.push_back(Launch(model, 256, 256));
    model.Resume();
    Check(Drain(model, handles) == std::vector<int>(6, 256),
          "shared pool pressure truncated output or stalled admission");
    Check(model.peakActiveCaches == 2, "shared pool failed to reserve remaining outputs");
    Check(Drain(model, {Launch(model, 700, 100)})[0] == 68,
          "per-request context limit was not enforced");
    Check(Drain(model, {Launch(model, 769, 1)}, -2)[0] == 0,
          "shared pool admitted an overlong request");
    Check(Drain(model, {Launch(model, 768, 1)}, -2)[0] == 0,
          "shared pool did not leave room for the first output");
    Check(Drain(model, {Launch(model, 256, -1)})[0] == 512,
          "unbounded output did not reserve through the context limit");
    std::cout << "SHARED_POOL_PASS\n";
}
}

int main() {
    try {
#ifdef USE_CUDA
        // CUDA builds query all devices when starting the legacy scheduler.
        // Initialize their contexts before the progress deadline starts.
        FastllmCudaGetFreeSizes();
#endif
        SetDeviceMap({{"cpu", 1}});
        SetMaxTokens(0);
        CompletePagesOnly();
        OverlappingPageViews();
        for (int stopAt : {20, 100, 128}) for (int reserve : {0, 8}) ReplayPressure(stopAt, reserve);
        ReplayPressure(100, 8, true);
        PagedBatchLimit();
        SharedPoolPressure();
        ConcurrentConsumers();
        for (auto view : {CacheView::Contiguous, CacheView::Paged, CacheView::TensorParallel}) {
            Single(view, 500, 128, 256, 244, 244);
        }
        for (auto view : {CacheView::Contiguous, CacheView::Paged, CacheView::TensorParallel}) {
            Single(view, 512, 128, 256, 512, 256);
            Single(view, 500, 128, 256, -1, 244);
            Single(view, 500, 128, 256, 20, 20);
            Single(view, 500, 128, 499, 10, 1);
        }
        for (auto view : {CacheView::Paged, CacheView::TensorParallel}) {
            for (int pageSize : {64, 256}) Single(view, 500, pageSize, 256, 244, 244);
            Single(view, 129, 128, 64, 256, 65);
            SchedulerModel model(view, 500);
            Check(Drain(model, {Launch(model, 501, 10)}, -2)[0] == 0, "overlong prompt was admitted");
            Concurrent(view, false);
            Concurrent(view, true);
        }
        // Model capability must permit simultaneous admission, not just a
        // queue that serves complete requests one at a time.
        Qwen4ExpModel qwen;
        Concurrent(CacheView::Paged, false, &qwen);
        Concurrent(CacheView::TensorParallel, false, &qwen);
        std::cout << "ALL_PASS\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
