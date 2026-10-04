#include "models/basellm.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
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
    int activeCaches = 0;
public:
    int peakActiveCaches = 0;

    SchedulerModel(CacheView view, int capacity, int pageSize = 128, int batch = 1)
        : view(view) {
        block_cnt = 1;
        kvCacheId = 0;
        maxBatch = batch;
        tokensLimit = capacity;
        max_positions = 4096; // The configured budget must also bound generation.
        dataType = kvCacheDataType = FLOAT32;
        eos_token_id = -100;
        canDoBatchForward = false;
        canDoConcurrentForward = true;
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
    void Resume() {
        std::lock_guard<std::mutex> lock(gateMutex);
        holdForward = false;
        gate.notify_all();
    }
    int Forward(const Data &ids, const Data &, const Data &,
                std::vector<std::pair<Data, Data>> &cache,
                const GenerationConfig &, const LastTokensManager &, std::vector<float> *) override {
        {
            std::unique_lock<std::mutex> lock(gateMutex);
            Check(gate.wait_for(lock, std::chrono::seconds(10), [&] { return !holdForward; }),
                  "scheduler fixture was not resumed");
        }
        std::lock_guard<std::mutex> lock(cacheMutex);
        const bool fresh = cache[0].first.dims.empty();
        const int length = (fresh ? 0 : cache[0].first.dims[1]) + ids.Count(0);
        Check(length <= tokensLimit, "scheduler exceeded the logical token limit");
        if (fresh) peakActiveCaches = std::max(peakActiveCaches, ++activeCaches);
        for (int component = 0; component < 2; ++component) {
            Data &data = component == 0 ? cache[0].first : cache[0].second;
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
            data.expansionDims = {1, (view == CacheView::Paged ? pool.maxPages : needed) * pool.pageLen, 1};
            if (view == CacheView::Paged) {
                data.isPagedKVCache = true;
                data.pagedKVCacheData = &pool;
                data.pageLen = pool.pageLen;
                data.lastPageLen = (length - 1) % pool.pageLen + 1;
            } else {
                data.cudaDataBorrowed = true;
            }
        }
        return 7;
    }
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

void Single(CacheView view, int capacity, int pageSize, int prompt, int output, int expected) {
    SchedulerModel model(view, capacity, pageSize);
    const int actual = Drain(model, {Launch(model, prompt, output)})[0];
    std::cout << "capacity=" << capacity << " page=" << pageSize << " view=" << (int)view
              << " prompt=" << prompt << " requested=" << output << " output=" << actual << '\n';
    Check(actual == expected, "premature termination: expected " + std::to_string(expected) +
          " tokens, got " + std::to_string(actual));
}

void Concurrent(CacheView view, bool pressure) {
    SchedulerModel model(view, 1024, 128, 2);
    model.Pause();
    int first = Launch(model, 256, pressure ? 1500 : 320);
    int second = Launch(model, 256, pressure ? 1500 : 192);
    model.Resume();
    const auto counts = Drain(model, {first, second});
    std::cout << "CONCURRENT_RESULT view=" << (int)view << " pressure=" << pressure
              << " output=" << counts[0] << ',' << counts[1] << '\n';
    Check(model.peakActiveCaches == 2, "fixture did not exercise concurrent requests");
    if (!pressure) Check(counts == std::vector<int>({320, 192}), "concurrent request ended early");
    else Check(counts[0] > 0 && counts[1] > 0 && counts[0] <= 768 && counts[1] <= 768,
               "pool pressure violated the context limits");
    // Reuse the same pool after the concurrent requests have been removed.
    Check(Drain(model, {Launch(model, 256, 768)})[0] == 768, "pool reuse lost capacity");
    std::cout << "CONCURRENT_PASS view=" << (int)view << " pressure=" << pressure << '\n';
}
}

int main() {
    try {
        SetDeviceMap({{"cpu", 1}});
        SetMaxTokens(0);
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
        std::cout << "ALL_PASS\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
