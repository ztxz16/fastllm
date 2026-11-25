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
    bool grouped = false;
    std::atomic<bool> blockAdmission{false};
    std::atomic<int> deferredAdmissions{0};
    bool TryAdmitResponseContext(ResponseContext *) override {
        if (!blockAdmission) return true;
        ++deferredAdmissions;
        return false;
    }
    int GetPrefillBatchGroups() const override { return grouped ? 4 : 1; }
    bool SupportsChunkedPrefillScheduling() const override { return supported; }
    std::atomic<bool> holdPrefill{false}, prefillBlocked{false};
    int GetPrefillBatchGroup(const ResponseContext *ctx) const override {
        return grouped && ctx->inputTokens > 0 ? ctx->allTokens[ctx->inputTokens - 1] : 0;
    }
    std::atomic<int> restores{0};
    std::atomic<bool> expireOnRestore{false};
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
        holdDecode = holdPrefill = false;
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
        if (expireOnRestore.exchange(false)) cachedPrefix = 0;
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
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &configs,
            const LastTokensManager &, std::vector<std::vector<float> *> *) override {
        bool prefill = false;
        int offset = 0;
        for (int row = 0; row < batch; ++row) {
            const int old = kv[row].first->dims.empty() ? 0 : kv[row].first->dims[1];
            prefill |= lengths[row] > 1 || configs[row].prefill_chunk_remaining >= 0;
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
        if (prefill) {
            prefillBlocked = true;
            while (holdPrefill) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            prefillBlocked = false;
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
        {
            PrefillDPAllocator allocator;
            for (int i = 0; i < 32; ++i) {
                Check(allocator.Assign(4, 3) == 3, "resident prefix lost affinity");
                Check(allocator.Assign(4) == i % 4, "affinity consumed a new-request turn");
            }
            std::vector<int> assigned(256);
            std::vector<std::thread> workers;
            for (int t = 0; t < 8; ++t) workers.emplace_back([&, t] {
                for (int i = t; i < 256; i += 8) assigned[i] = allocator.Assign(4);
            });
            for (auto &worker : workers) worker.join();
            for (int group = 0; group < 4; ++group)
                Check(std::count(assigned.begin(), assigned.end(), group) == 64,
                      "concurrent new requests lost RR balance");
            Check(allocator.Assign(1) == 0 && allocator.Assign(1, 0) == 0, "single-group placement");
            bool rejected = false;
            try { allocator.Assign(4, 4); } catch (const std::invalid_argument &) { rejected = true; }
            Check(rejected, "invalid cache owner was silently migrated");
        }
#ifdef USE_CUDA
        FastllmCudaGetFreeSizes();
        FastllmCudaTrimBigBuffer();
#endif
        // A model's physical slots may be full even below maxBatch. Rejected
        // prompts must remain untouched, and decode/output consumption must
        // continue until the model permits admission.
        for (bool batching : {false, true}) {
            SchedulerModel model(batching);
            if (batching) Check(model.SetPrefillBatching(8192, 0), "slot test batching");
            int active = Seed(model, 128);
            Hold(model);
            model.blockAdmission = true;
            const size_t before = model.Events().size();
            int pending = Launch(model, 128, 2);
            model.holdDecode = false;
            Wait([&] { return model.deferredAdmissions > 0 && model.Events().size() > before + 2; });
            Check(Prefills(model, before).empty(), "unadmitted prompt reached forward");
            model.blockAdmission = false;
            model.WakeResponseOutput();
            Drain(model, {pending}, 2);
            model.AbortResponse(active);
        }
        // Incompatible ready prompts keep ordinary single-request admission.
        for (const auto &lengths : {std::vector<int>{128, 128}, {128, 256}, {16, 16}}) {
            struct EqualShapeModel : SchedulerModel {
                int GetPrefillBatchKey(const ResponseContext &ctx, int) const override {
                    return ctx.currentTokens.size() < 64 ? -1 : int(ctx.currentTokens.size());
                }
            } model;
            Check(model.SetPrefillBatching(4096, 0), "shape policy rejected");
            Seed(model); Hold(model);
            const auto start = model.Events().size();
            auto a = Launch(model, lengths[0]), b = Launch(model, lengths[1]);
            model.holdDecode = false;
            Drain(model, {a, b});
            auto events = Prefills(model, start);
            if (lengths.front() == 128 && lengths.back() == 128)
                Check(events.size() == 1 && events[0].lengths == lengths, "compatible prompts split");
            else
                Check(events.size() == 2 && events[0].lengths.size() == 1 && events[1].lengths.size() == 1,
                      "incompatible prompts pooled");
        }
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
        // A grouped arrival must fill all available DP owners before taking
        // another request from one owner; cached prefixes are not work.
        for (int cached : {0, 3072}) for (int wait : {0, 50}) {
            SchedulerModel model(true, cached); model.grouped = true;
            Check(model.SetPrefillBatching(4096, wait, 0, 4), "group config rejected");
            Seed(model, cached + 8192); Hold(model);
            const size_t start = model.Events().size();
            std::vector<int> handles;
            for (int owner : {0, 0, 0, 1, 1, 1, 3, 3, 3}) {
                GenerationConfig cfg; cfg.output_token_limit = 1;
                cfg.add_special_tokens = false; cfg.top_k = 1;
                std::vector<int> tokens(cached + (owner == 0 ? 2048 : 1024), 1);
                tokens.back() = owner;
                handles.push_back(model.LaunchResponseTokens(tokens, cfg));
            }
            model.holdDecode = false; Drain(model, handles);
            auto events = Prefills(model, start);
            Check(events.size() == 3, "DP group batch count changed");
            for (const auto &e : events)
                Check(e.lengths == std::vector<int>({2048, 1024, 1024}),
                      "DP admission did not balance uncached tokens / preserve FIFO");
        }
        // A full DP owner must not repeatedly restore and release its waiting
        // requests while other owners fill the batch. Exercise zero-wait and
        // low-decode admission as well as the normal waiting path.
        for (int wait : {0, 50}) for (int minDecode : {0, 8}) {
            SchedulerModel model(true, 3072);
            model.grouped = true;
            model.maxBatch = 32;
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(512, wait, 0, 24, minDecode), "DP quota config");
            Check(model.SetPrefillScheduling(true, 0, true, 0, 0), "DP quota scheduling");
            Seed(model, 3200); Hold(model);
            const int restored = model.restores;
            const int previewed = model.previews;
            const size_t start = model.Events().size();
            std::vector<int> handles;
            for (int owner : {0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3}) {
                GenerationConfig cfg; cfg.output_token_limit = 1;
                cfg.add_special_tokens = false; cfg.top_k = 1;
                std::vector<int> tokens(3200, 1); tokens.back() = owner;
                handles.push_back(model.LaunchResponseTokens(tokens, cfg));
            }
            model.holdDecode = false;
            Drain(model, handles);
            Check(model.restores == restored + (int)handles.size(), "DP quota restored an unselected request");
            if (wait == 0)
                Check(model.previews == previewed + (int)handles.size(), "full DP owner kept previewing queued prefixes");
            auto events = Prefills(model, start);
            Check(events.size() == 4, "DP quota changed number of batches");
            for (const auto &e : events)
                Check(e.lengths == std::vector<int>({128, 128, 128}), "DP quota changed packed tokens");
        }
        // A cache disappearing after planning must use the full input, then
        // recheck the real batch budget instead of trusting preview lengths.
        {
            SchedulerModel model(true, 3072);
            model.chunkedPrefillSize = 4096;
            Check(model.SetPrefillBatching(4096, 0), "stale plan config");
            Seed(model, 7168); Hold(model);
            const size_t start = model.Events().size();
            int a = Launch(model, 4096), b = Launch(model, 4096);
            model.expireOnRestore = true;
            model.holdDecode = false;
            Drain(model, {a, b});
            auto events = Prefills(model, start);
            Check(events.size() == 2 && events[0].lengths == std::vector<int>{4096} &&
                  events[1].lengths == std::vector<int>{4096}, "stale admission plan changed input or budget");
        }
        // A mandatory decode slice must not query pending prefixes. Keep a
        // prefill in flight long enough to owe decode, then enqueue a new hit.
        {
            SchedulerModel model(true, 3072);
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(128, 50, 0, 4, 1), "decode-only batch config");
            Check(model.SetPrefillScheduling(false, 0, true, 0, 20), "decode-only scheduling");
            Seed(model, 3200); Hold(model);
            model.holdPrefill = true;
            int a = Launch(model, 3200);
            model.holdDecode = false;
            Wait([&] { return model.prefillBlocked.load(); });
            int b = Launch(model, 3200);
            const int previews = model.previews;
            const int restores = model.restores;
            model.holdDecode = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            model.holdPrefill = false;
            Wait([&] { return model.decodeBlocked.load(); });
            Check(model.previews == previews && model.restores == restores,
                  "decode-only slice scanned or restored a pending prefix");
            model.holdDecode = false;
            Drain(model, {a, b});
        }
        // A partial prefill must yield its group to older waiting work at a
        // chunk boundary, including short cached continuations. Rejoining at
        // the tail also lets cold requests progress under continued arrivals.
        for (bool grouped : {false, true}) {
            SchedulerModel model(true, 3072);
            model.grouped = grouped;
            model.maxBatch = 32;
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(grouped ? 512 : 128, 0, 0, 24), "rotation config");
            Check(model.SetPrefillScheduling(grouped, 0, true, 0, 0), "rotation scheduling");
            Seed(model, 3200); Hold(model);
            const size_t from = model.Events().size();
            int cold = Launch(model, 3072 + 1025);
            int warm = Launch(model, 3072 + 32);
            model.holdDecode = false;
            Drain(model, {warm, cold});
            auto events = Prefills(model, from);
            Check(events.size() >= 2 && events[0].lengths == std::vector<int>{128} &&
                  events[1].lengths == std::vector<int>{32},
                  "long prefill monopolized the group before a short cached request");
        }
        // Long prompts remain admitted across scheduler iterations. One-token
        // tails emit exactly one result and retain absolute KV positions.
        for (int batch : {1, 2, 8}) {
            SchedulerModel model;
            model.maxBatch = batch;
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(1024, 0), "chunk batching config rejected");
            Check(model.SetPrefillScheduling(false, 0, true, 0, 1), "chunk scheduling rejected");
            std::vector<int> handles;
            for (int i = 0; i < batch; ++i) handles.push_back(Launch(model, 513 + 128 * i, 3));
            Drain(model, handles, 3);
            for (const auto &event : Prefills(model))
                for (int n : event.lengths) Check(n <= 128, "long prompt did not yield at chunk boundary");
        }
        {
            SchedulerModel model;
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(1024, 0), "fairness batch config");
            Check(model.SetPrefillScheduling(false, 0, true, 0, 1), "fairness config");
            Seed(model, 128); Hold(model);
            const size_t from = model.Events().size();
            model.holdPrefill = true;
            int h = Launch(model, 1025, 1);
            model.holdDecode = false;
            Wait([&] { return model.prefillBlocked.load(); });
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            model.holdPrefill = false;
            Drain(model, {h});
            auto events = model.Events();
            bool first = false, decode = false, interleaved = false;
            for (size_t i = from; i < events.size(); ++i) {
                if (events[i].prefill) { if (first && decode) { interleaved = true; break; } first = true; }
                else if (first) decode = true;
            }
            Check(interleaved, "long prefill starved decode");
        }
        {
            SchedulerModel model;
            model.chunkedPrefillSize = 128;
            Check(model.SetPrefillBatching(1024, 0), "cancel batch config");
            Check(model.SetPrefillScheduling(false, 0, true, 0, 1), "cancel chunk config");
            model.holdPrefill = true;
            int h = Launch(model, 1025);
            Wait([&] { return model.prefillBlocked.load(); });
            model.AbortResponse(h);
            model.holdPrefill = false;
            Drain(model, {Launch(model, 129)});
            Check(model.Events().size() == 3, "cancelled partial prompt continued executing");
        }
        {
            PrefillSchedulingOptions options{true, true, 256, 1000, 1000};
            auto p = options.Effective({32768, 30000, 16, 24, 64}, 128);
            Check(p.targetTokens == 16384 && p.targetRequests == 8 && p.maxWaitMs == 15000,
                  "adaptive release target incorrect");
            std::vector<PrefillGroupLoad> groups(8);
            for (auto &g : groups) g.decoding = 16;
            groups[0].tokens = 32768; groups[0].requests = 16;
            auto t = Clock::now();
            Check(std::string(DecidePrefillGroups(p, groups, false, t, t)) == "wait", "one DP released empty peers");
            for (auto &g : groups) { g.tokens = 2048; g.requests = 1; }
            Check(std::string(DecidePrefillGroups(p, groups, false, t, t)) == "dp_targets", "ready DPs did not release");
            groups.back().tokens = 0; groups.back().requests = 0;
            Check(std::string(DecidePrefillGroups(p, groups, false, t, t + std::chrono::seconds(15))) == "timeout", "empty DP caused starvation");
        }
        std::cout << "ALL_PASS preview/admission, capacity, requests, idle, cache invalidation, deadlines, decode, cancellation, chunking\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
