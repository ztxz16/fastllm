#include "models/naive_n05_flash.h"
#include <iostream>
#include <stdexcept>

namespace fastllm {
static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Exercise production checkpoint ownership and page-trie operations without
// allocating the KV payload or requiring a large model.
struct NaiveN05PrefixCacheTestAccess : NaiveN05FlashModel {
    explicit NaiveN05PrefixCacheTestAccess(int slots) {
        block_cnt = 2;
        slidingLayers = {0, 1};
        tpDevices = {0};
        prefixPool = std::make_shared<PrefixPool>();
        for (int i = 0; i < slots; ++i) prefixPool->freeSlots.push_back(i);
        tpPagedCaches.resize(1);
        tpPagedCaches[0].resize(4);
        auto pool = std::make_unique<PagedCacheManager>();
        pool->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        pool->pageLen = 128;
        pool->SetMaxPages(390625);
        tpPagedCaches[0][0] = std::move(pool);
        SetSaveHistoryChat(true);
    }
    std::unique_ptr<ResponseContext> Request(int id) {
        auto c = std::make_unique<ResponseContext>();
        c->pastKeyValues.resize(2);
        c->allTokens.assign(8448, id + 100);
        c->currentTokens = c->allTokens;
        auto &key = c->pastKeyValues[0].first;
        auto *manager = GetPagedKVCacheManager(0, true);
        key.pageLen = key.lastPageLen = 128;
        for (int i = 0; i < 8448 / 128; ++i)
            key.pageIndex.push_back(manager->GetUnusedPageIndex(true));
        activePrefixes.try_emplace(&key);
        return c;
    }
    int Checkpoint(ResponseContext &c, int past, int length) {
        auto &key = c.pastKeyValues[0].first;
        auto checkpoints = BeginPrefixSnapshots(key, past, length);
        FinishPrefixSnapshots(key, checkpoints, length > 1);
        return (int)checkpoints.size();
    }
    void Publish(ResponseContext &c) {
        TryRecordResponseContext(&c);
        GetPagedKVCacheManager(0, true)->Record(c.allTokens, c.pastKeyValues[0].first.pageIndex);
    }
    void Drain(ResponseContext &c) {
        activePrefixes.erase(&c.pastKeyValues[0].first);
        GetPagedKVCacheManager(0, true)->ReleasePageIndices(c.pastKeyValues[0].first.pageIndex);
    }
    int FullHit(ResponseContext &c) {
        std::vector<int> pages;
        GetPagedKVCacheManager(0, true)->Query(c.currentTokens, pages);
        return (int)pages.size() * 128;
    }
    int WindowHit(ResponseContext &c) {
        return QueryPagedPrefixCacheExtra(&c, 8448);
    }
    ~NaiveN05PrefixCacheTestAccess() {
        activePrefixes.clear();
        prefixHistory.clear();
    }

    static void BudgetPreviewOwnership() {
        NaiveN05PrefixCacheTestAccess model(4);
        auto c = model.Request(0);
        model.Checkpoint(*c, 0, 8192); model.Checkpoint(*c, 8192, 200);
        model.Checkpoint(*c, 8447, 1); model.Publish(*c); model.Drain(*c);
        auto *pool = model.GetPagedKVCacheManager(0, true);
        auto refs = pool->pageRefCount;
        PrefillPrefixPreview preview; preview.Reset(c->currentTokens);
        Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8192, preview) == 8192,
                "short preview did not honor its limit");
        for (int i = 0; i < 20; ++i)
            Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8448, preview) == model.WindowHit(*c),
                    "SWA preview differs from actual query");
        Require(pool->pageRefCount == refs, "SWA preview pinned full KV");
        // Publication changes the candidate set even if the request itself is
        // unchanged. Keep another owner of the replaced checkpoint to ensure
        // invalidation does not depend on weak-pointer expiry alone.
        auto held = model.prefixHistory.back();
        auto replacement = model.Request(0);
        model.Checkpoint(*replacement, 0, 8192);
        model.Checkpoint(*replacement, 8192, 200);
        model.Checkpoint(*replacement, 8447, 1);
        model.Publish(*replacement); model.Drain(*replacement);
        Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8448, preview) == model.WindowHit(*c),
                "publication reused a retired candidate");
        held.reset();
        // Weak metadata must not keep the four checkpoint slots alive.
        model.prefixHistory.clear();
        Require(model.prefixPool->freeSlots.size() == 4, "preview pinned an SWA checkpoint");
        Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8448, preview) == 0,
                "preview resurrected removed SWA checkpoint");
        auto other = model.Request(1); model.Checkpoint(*other, 0, 8192);
        model.Publish(*other); model.Drain(*other);
        c->currentTokens = other->currentTokens; preview.Reset(c->currentTokens);
        Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8448, preview) == model.WindowHit(*c),
                "query mutation reused stale token equality");
        pool->SetMaxPages(390625);
        Require(model.PreviewPagedPrefixCacheExtra(c.get(), 8448, preview) == 0,
                "preview trusted token equality after full KV eviction");
    }

    static void Capacity() {
        PrefixPool pool;
        pool.slotBytes = 3194880;
        for (int batch : {1, 2, 3, 4, 8, 16, 32, 64, 128, 220, 224, 400, 512}) {
            Require(pool.SlotCount(batch, 0) == batch * 4, "missing minimum checkpoint reservation");
            Require(pool.SlotCount(batch, -1) == batch * 4, "negative budget lost reservation");
            Require(pool.SlotCount(batch, int64_t(pool.slotBytes) * batch * 6 * 32) == batch * 6,
                    "completed-history headroom lost");
            Require(pool.SlotCount(batch, int64_t(pool.slotBytes) * batch * 10 * 32) == batch * 8,
                    "checkpoint pool grew without bound");
        }
        pool.slotBytes = 0;
        Require(pool.SlotCount(400, 1LL << 40) == 0, "allocated SWA checkpoints without SWA layers");
    }

    static void PinnedPressure() {
        NaiveN05PrefixCacheTestAccess model(452);
        std::vector<std::unique_ptr<ResponseContext>> requests;
        for (int i = 0; i < 400; ++i) {
            requests.push_back(model.Request(i));
            model.Checkpoint(*requests.back(), 0, 8192);
            model.Checkpoint(*requests.back(), 8192, 200);
        }
        for (auto &c : requests) model.Publish(*c);
        Require(model.prefixHistory.size() == 452, "pressure setup did not fill pool");
        Require(model.Checkpoint(*requests.back(), 8447, 1) == 0, "overcommitted pinned pool");
        Require(model.prefixHistory.size() == 452, "allocation failure erased pinned history");
        for (auto &c : requests) model.Drain(*c);
        int hits = 0;
        for (auto &c : requests) {
            Require(model.FullHit(*c) == 8448, "full KV prefix unexpectedly evicted");
            hits += model.WindowHit(*c) > 0;
        }
        Require(hits == 151, "published SWA checkpoints lost under pressure");
    }

    static void FourPerRequest() {
        NaiveN05PrefixCacheTestAccess model(1600);
        std::vector<std::unique_ptr<ResponseContext>> requests;
        for (int i = 0; i < 400; ++i) {
            requests.push_back(model.Request(i));
            auto &c = *requests.back();
            Require(model.Checkpoint(c, 0, 8192) == 2, "missing prefill boundary");
            Require(model.Checkpoint(c, 8192, 200) == 1, "missing third prefill checkpoint");
            Require(model.Checkpoint(c, 8447, 1) == 1, "missing decode checkpoint");
            model.Publish(c);
        }
        Require(model.prefixHistory.size() == 1600 && model.prefixPool->freeSlots.empty(),
                "four-checkpoint setup did not fill pool");
        auto &last = *requests.back();
        auto &active = model.activePrefixes.at(&last.pastKeyValues[0].first);
        auto pinned = active.decode;
        Require(model.Checkpoint(last, 8575, 1) == 0, "overwrote shared checkpoint");
        Require(active.decode == pinned && model.prefixHistory.size() == 1600,
                "failed replacement discarded old checkpoint");
        pinned.reset();
        Require(model.Checkpoint(last, 8575, 1) == 1, "did not recycle own unshared checkpoint");
        Require(active.prefill.size() == 3 && active.decode->length == 8576,
                "changed per-request checkpoint retention");
        for (auto &c : requests) model.Drain(*c);
        for (auto &c : requests)
            Require(model.FullHit(*c) == 8448 && model.WindowHit(*c) >= 8320,
                    "400 independent prefixes did not remain reusable");
    }

    static void EvictCompletedOnly() {
        NaiveN05PrefixCacheTestAccess model(4);
        auto old = model.Request(0);
        model.Checkpoint(*old, 0, 8192);
        model.Checkpoint(*old, 8192, 200);
        model.Checkpoint(*old, 8447, 1);
        model.Publish(*old);
        auto pinned = model.activePrefixes.at(&old->pastKeyValues[0].first).decode;
        model.Drain(*old);
        auto next = model.Request(1);
        Require(model.Checkpoint(*next, 0, 8192) == 2, "failed to evict completed LRU entries");
        Require(model.Checkpoint(*next, 8192, 200) == 1, "failed to reclaim third completed entry");
        Require(model.WindowHit(*old) == 8448, "evicted a checkpoint held by a fork");
        Require(model.Checkpoint(*next, 8447, 1) == 0, "recycled a pinned completed checkpoint");
        Require(model.WindowHit(*old) == 8448, "failed allocation erased forked history");
        pinned.reset();
        Require(model.Checkpoint(*next, 8447, 1) == 1, "released checkpoint did not become evictable");
        model.Publish(*next);
        model.Drain(*next);
        Require(model.WindowHit(*next) == 8448, "replacement history missing");
    }

    static void SharedReplacement() {
        for (bool prefill : {false, true}) {
            NaiveN05PrefixCacheTestAccess model(5);
            auto c = model.Request(0);
            model.Checkpoint(*c, 0, 8192);
            model.Checkpoint(*c, 8192, 200);
            model.Checkpoint(*c, 8447, 1);
            model.Publish(*c);
            auto &active = model.activePrefixes.at(&c->pastKeyValues[0].first);
            auto pinned = prefill ? active.prefill.front() : active.decode;
            Require(model.Checkpoint(*c, prefill ? 8392 : 8575, prefill ? 200 : 1) == 1,
                    "could not replace checkpoint using a spare slot");
            Require(model.QueryPagedPrefixCacheExtra(c.get(), pinned->length) == pinned->length,
                    "successful replacement erased history still held by a fork");
            Require(active.prefill.size() == 3 && active.decode, "changed checkpoint retention");
            model.Drain(*c);
        }
    }

    static void RollingConversations() {
        for (int count : {2, 8, 400}) {
            NaiveN05PrefixCacheTestAccess model(count * 4);
            std::vector<std::unique_ptr<ResponseContext>> conversations;
            for (int i = 0; i < count; ++i) {
                conversations.push_back(model.Request(i));
                auto &c = *conversations.back();
                model.Checkpoint(c, 0, 8192);
                model.Checkpoint(c, 8192, 200);
                model.Checkpoint(c, 8447, 1);
                model.Publish(c);
                model.Drain(c);
            }
            auto *manager = model.GetPagedKVCacheManager(0, true);
            auto &busy = *conversations.back();
            for (int round = 0; round < 8; ++round) {
                const int past = busy.allTokens.size();
                std::shared_ptr<PrefixSnapshot> restored;
                for (const auto &entry : model.prefixHistory)
                    if (entry->length == past && entry->tokens == busy.allTokens)
                        restored = entry;
                Require(restored != nullptr, "busy conversation lost its own checkpoint");
                auto &key = busy.pastKeyValues[0].first;
                model.activePrefixes[&key].prefill = {restored};
                restored.reset();
                manager->Pick(key.pageIndex);
                for (int i = 0; i < 5; ++i)
                    key.pageIndex.push_back(manager->GetUnusedPageIndex(true));
                busy.allTokens.resize(past + 640, count + 99);
                busy.currentTokens = busy.allTokens;
                model.Checkpoint(busy, past, 512);
                model.Checkpoint(busy, past + 639, 1);
                model.Publish(busy);
                model.Drain(busy);
                for (int i = 0; i < count - 1; ++i) {
                    Require(model.FullHit(*conversations[i]) == 8448, "idle full KV was evicted");
                    Require(model.WindowHit(*conversations[i]) == 8448,
                            "busy conversation evicted an idle conversation's latest checkpoint");
                }
                Require(model.prefixHistory.size() <= (size_t)count * 4,
                        "rolling history exceeded the unchanged slot budget");
            }
            std::cout << "rolling conversations=" << count << " rounds=8 idle_hits="
                      << count - 1 << " slots=" << count * 4 << "\n";
        }
    }
};
} // namespace fastllm

int main() {
    try {
        fastllm::NaiveN05PrefixCacheTestAccess::BudgetPreviewOwnership();
        fastllm::NaiveN05PrefixCacheTestAccess::Capacity();
        fastllm::NaiveN05PrefixCacheTestAccess::PinnedPressure();
        fastllm::NaiveN05PrefixCacheTestAccess::FourPerRequest();
        fastllm::NaiveN05PrefixCacheTestAccess::EvictCompletedOnly();
        fastllm::NaiveN05PrefixCacheTestAccess::SharedReplacement();
        fastllm::NaiveN05PrefixCacheTestAccess::RollingConversations();
        std::cout << "PREFIX CACHE PASS: capacity, pinned pressure, 400 independent prefixes, completed LRU\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
