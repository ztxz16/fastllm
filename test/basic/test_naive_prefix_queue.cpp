#include "models/naive_n05_flash.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

// Real prefix residency, slot eviction and request lifecycle, without model
// weights or GPU payloads. Only immutable checkpoint metadata is exercised.
namespace fastllm {
struct NaiveN05PrefixCacheTestAccess : public NaiveN05FlashModel {
    PagedCacheManager pages;
public:
    NaiveN05PrefixCacheTestAccess(int requests, bool draft = false) {
        block_cnt = 1; maxBatch = requests; draftEnabled = draft;
        saveHistoryChat = true;
        prefixPool = std::make_shared<PrefixPool>();
        prefixPool->slotBytes = 1;
        pages.pageLen = 128; pages.SetMaxPages(8);
        prefixPool->slots.Init(1, 2);
        Install(1); Install(2);
    }
    ~NaiveN05PrefixCacheTestAccess() override {
        ShutdownRuntime();
        activePrefixes.clear(); prefixHistory.clear(); prefixPool.reset();
        if (pages.trieRoot) pages.EvictTrieSubtree(pages.trieRoot);
        pages.trieRoot = nullptr;
    }
    void Install(int token) {
        auto snapshot = std::make_shared<PrefixSnapshot>();
        snapshot->pool = prefixPool; snapshot->slot = prefixPool->slots.Take(0);
        snapshot->length = 128; snapshot->tokens.assign(128, token);
        snapshot->tokenHash = PagedCacheManager::HashTokenPage(snapshot->tokens.data(), 128);
        snapshot->draftReady = true;
        const int page = pages.GetUnusedPageIndex(true);
        pages.Record(snapshot->tokens, {page});
        snapshot->fullPool = &pages; snapshot->fullPage = page;
        snapshot->fullGeneration = pages.pageTimestamp[page];
        pages.ReleasePageIndex(page);
        prefixHistory.push_back(std::move(snapshot));
    }
    void Create(ResponseContext &ctx, int token) {
        ctx.Init(1, FLOAT32, FLOAT32);
        ctx.allTokens.assign(160, token); ctx.currentTokens = ctx.allTokens;
        ctx.inputTokens = 160;
        OnResponseContextCreated(&ctx);
    }
    int Hit(ResponseContext &ctx) {
        PrefillPrefixPreview preview; preview.Reset(ctx.currentTokens);
        return PreviewPagedPrefixCacheExtra(&ctx, 159, preview);
    }
    bool TryPressure(ResponseContext &ctx) {
        auto snapshots = BeginPrefixSnapshots(ctx.pastKeyValues[0].first, 0, 128);
        const bool captured = !snapshots.empty();
        // Keep any newly allocated slot in use until this request is removed.
        activePrefixes.at(&ctx.pastKeyValues[0].first).prefill = std::move(snapshots);
        return captured;
    }
    void Pressure(ResponseContext &ctx) {
        Check(TryPressure(ctx), "checkpoint pressure could not reclaim an unprotected slot");
    }
    void PublishEquivalent(ResponseContext &ctx) {
        auto snapshot = activePrefixes.at(&ctx.pastKeyValues[0].first).prefill.front();
        auto previous = prefixHistory.front();
        snapshot->fullPool = previous->fullPool;
        snapshot->fullPage = previous->fullPage;
        snapshot->fullGeneration = previous->fullGeneration;
        snapshot->draftReady = true;
        TryRecordResponseContext(&ctx);
    }
    void InvalidateFullKV() {
        pages.EvictTrieSubtree(pages.trieRoot); pages.trieRoot = nullptr;
    }
    int FreePages() const { return pages.FreePageCount(); }
};

}
using PrefixModel = NaiveN05PrefixCacheTestAccess;

int main() {
    try {
        for (bool draft : {false, true}) {
            PrefixModel model(2, draft);
            ResponseContext waiting, pressure;
            model.Create(waiting, 1); model.Create(pressure, 3);
            Check(model.Hit(waiting) == 128, "initial SWA checkpoint unavailable");
            int free = model.FreePages();
            model.Pressure(pressure);
            Check(model.Hit(waiting) == 128, "queued SWA checkpoint lost under slot pressure");
            Check(model.FreePages() == free && waiting.pastKeyValues[0].first.pageIndex.empty(),
                  "waiting checkpoint retention pinned or restored full KV");
            model.OnResponseContextRemoved(&pressure);
            model.OnResponseContextRemoved(&waiting);
        }
        // Republishing an equivalent checkpoint must not orphan a queued
        // guard: the replacement can itself be evicted before admission.
        {
            PrefixModel model(2);
            ResponseContext waiting, pressure;
            model.Create(waiting, 1); model.Create(pressure, 1);
            model.Pressure(pressure); model.PublishEquivalent(pressure);
            model.OnResponseContextRemoved(&pressure);
            model.Create(pressure, 3); model.Pressure(pressure);
            Check(model.Hit(waiting) == 128, "deduplication orphaned a queued checkpoint guard");
            model.OnResponseContextRemoved(&pressure); model.OnResponseContextRemoved(&waiting);
        }
        // Retaining an SWA window must not revive a recycled full-KV prefix.
        {
            PrefixModel model(2);
            ResponseContext waiting;
            model.Create(waiting, 1);
            model.InvalidateFullKV();
            Check(model.Hit(waiting) == 0, "checkpoint guard revived invalid full KV");
            model.Pressure(waiting); // Cold admission drops the stale guard.
            model.OnResponseContextRemoved(&waiting);
        }
        // A full protected pool degrades to no checkpoint capture, without
        // invalidating waiting hits or requiring another allocation.
        {
            PrefixModel model(2);
            ResponseContext first, second, pressure;
            model.Create(first, 1); model.Create(second, 2); model.Create(pressure, 3);
            Check(!model.TryPressure(pressure), "full protected pool did not fall back safely");
            Check(model.Hit(first) == 128 && model.Hit(second) == 128, "fallback discarded a protected hit");
            model.OnResponseContextRemoved(&pressure);
            model.OnResponseContextRemoved(&first); model.OnResponseContextRemoved(&second);
        }
        // Protection is bounded by maxBatch, and cancellation releases it.
        {
            PrefixModel model(1);
            ResponseContext waiting, overflow, pressure;
            model.Create(waiting, 1); model.Create(overflow, 1); model.Create(pressure, 3);
            model.OnResponseContextRemoved(&waiting);
            model.Pressure(pressure);
            Check(model.Hit(overflow) == 0, "overflow waiter retained an unbounded checkpoint guard");
            model.OnResponseContextRemoved(&pressure); model.OnResponseContextRemoved(&overflow);
        }
        std::cout << "ALL_PASS queued SWA retention, bounded slots, cancellation, no KV pinning, draft metadata\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
