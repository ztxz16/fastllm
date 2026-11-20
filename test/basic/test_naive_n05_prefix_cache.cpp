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
        prefixPool->slots.Init(1, slots);
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

    static void WindowOwnership() {
        NaiveN05PrefixCacheTestAccess model(2);
        model.attentionDP = 2;
        model.tpDevices = {5, 3, 7, 1};
        Require(model.WindowRequestsPerGroup(5) == 4, "incorrect uneven DP window capacity");
        model.activeWindowSlots = std::make_shared<WindowSlots>();
        model.activeWindowSlots->Init(2, 2, 1);
        ResponseContext a, b, queued;
        for (auto *ctx : {&a, &b, &queued}) ctx->pastKeyValues.resize(2);
        a.pastKeyValues[0].first.dataDeviceIds = queued.pastKeyValues[0].first.dataDeviceIds = {5, 3};
        b.pastKeyValues[0].first.dataDeviceIds = {7, 1};
        Require(model.TryAdmitResponseContext(&a) && model.TryAdmitResponseContext(&a) &&
                !model.CanAdmitResponseContext(&queued) && !model.TryAdmitResponseContext(&queued) &&
                model.TryAdmitResponseContext(&b), "owner-local admission overcommitted or blocked a peer");
        a.pastKeyValues[0].first.kvStorageOwner.reset();
        Require(model.TryAdmitResponseContext(&queued), "owner slot leaked across retirement");
        Require(model.RunningRequestsPerGroup(5) == 3 && model.RunningRequestsPerGroup(1) == 1,
                "uneven batches must permit queued work on any owner");
        model.attentionDP = 8;
        const int active = model.RunningRequestsPerGroup(512), physical = model.WindowRequestsPerGroup(512);
        WindowSlots slots;
        slots.Init(8, physical, physical - active);
        for (int owner = 0; owner < 8; ++owner) {
            for (int i = 0; i < 64; ++i) Require(slots.Take(owner) >= 0, "lost active DP capacity");
            Require(!slots.Available(owner) && slots.Take(owner) < 0, "65th request exceeded DP budget");
        }
        const int padding = slots.Take(0, false);
        Require(padding >= 0, "graph padding lost its physical slot");
        slots.Return(padding);
        Require(!slots.Available(0), "padding row became eligible for request admission");
        slots.Return(0);
        Require(slots.Take(0) == 0 && !slots.Available(1), "retirement must unblock only its owner");
        WindowSlots history;
        history.Init(2, 2);
        const int a0 = history.Take(0), a1 = history.Take(0), b0 = history.Take(1);
        Require(a0 != b0 && history.Row(a0) == history.Row(b0) && history.Row(a0) != history.Row(a1) &&
                history.Take(0) == -1, "logical checkpoint aliases a physical row in its group");
        history.Return(a0);
        Require(history.Take(0) == a0 && history.Take(1) >= 0, "checkpoint returned to wrong owner");
    }

    static void FinishBeforeDrain() {
        NaiveN05PrefixCacheTestAccess model(4);
        auto c = model.Request(0);
        model.Checkpoint(*c, 0, 8192); model.Publish(*c);
        const int hit = model.WindowHit(*c);
        Require(hit > 0, "finish fixture has no resident prefix");
        auto &root = c->pastKeyValues.front().first;
        auto *pool = model.GetPagedKVCacheManager(0, true);
        const int sharedPage = root.pageIndex.front();
        std::vector<int> borrowed{sharedPage};
        pool->Pick(borrowed); // Another request still borrows this page.
        auto *leaf = new Data();
        leaf->isPagedKVCache = true; leaf->pagedKVCacheData = pool;
        leaf->pageIndex = root.pageIndex;
        leaf->Resize({1, 8448, 1});
        root.multiDeviceData = true; root.multiDeviceDatas[0] = leaf;
        root.dataDeviceIds = {0}; root.isPagedKVCache = true;
        root.pagedKVCacheData = pool; // Non-owning TP root mirrors the leaf.
        root.Resize({1, 8448, 1});
        model.tpRequestIds[&root] = 77;
        c->inputTokens = 8448; c->cacheLen = 128; c->curTokens = 2;
        c->resultTokenQueue.push(11); c->resultTokenQueue.push(12);
        c->resultLogits.push(new std::vector<float>{.25f, .75f});
        const int free = pool->FreePageCount();
        model.OnResponseContextFinished(c.get());
        Require(pool->FreePageCount() == free && root.multiDeviceData,
                "unfinished request released its KV");
        c->isEnding = true;
        model.OnResponseContextFinished(c.get());
        Require(pool->FreePageCount() == pool->maxPages - 1 && pool->pageRefCount[sharedPage] == 1,
                "finished request pinned pages or released a shared prefix twice");
        Require(!root.multiDeviceData && root.multiDeviceDatas.empty() && root.pageIndex.empty() && root.dims.empty(),
                "finished request retained stale cache descriptors");
        Require(!model.tpRequestIds.count(&root) && !model.activePrefixes.count(&root),
                "finished request retained private state");
        Require(model.WindowHit(*c) == hit, "finish discarded published prefix history");
        Require(c->inputTokens == 8448 && c->cacheLen == 128 && c->curTokens == 2 &&
                c->resultTokenQueue.size() == 2 && c->resultTokenQueue.front() == 11 &&
                c->resultLogits.size() == 1 && c->resultLogits.front()->at(1) == .75f,
                "finish changed pending output or statistics");
        model.OnResponseContextFinished(c.get());
        model.OnResponseContextRemoved(c.get());
        c.reset();
        Require(pool->pageRefCount[sharedPage] == 1, "client drain released a shared page twice");
        pool->ReleasePageIndex(sharedPage);
        Require(pool->FreePageCount() == pool->maxPages, "finish/drain leaked pages");
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
        Require(model.prefixPool->slots.free[0].size() == 4, "preview pinned an SWA checkpoint");
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

    static void PublicationHashCollision() {
        NaiveN05PrefixCacheTestAccess model(8);
        auto first = model.Request(0), second = model.Request(1);
        model.Checkpoint(*first, 0, 8192); model.Publish(*first);
        model.Checkpoint(*second, 0, 8192); model.Publish(*second);
        const size_t count = model.prefixHistory.size();
        auto distinct = model.prefixHistory.front();
        auto published = model.prefixHistory.back();
        const uint64_t original = distinct->tokenHash;
        // Fingerprints are only a filter: a collision must not retire a
        // different conversation or alter the most-recent publication order.
        distinct->tokenHash = published->tokenHash;
        model.Publish(*second);
        Require(model.prefixHistory.size() == count, "hash collision retired a different prefix");
        Require(model.prefixHistory.back() == published, "publication LRU changed");
        distinct->tokenHash = original;
        model.Drain(*first); model.Drain(*second);
    }

    static void PreviewHashCollision() {
        NaiveN05PrefixCacheTestAccess model(8);
        auto first = model.Request(0), second = model.Request(1);
        model.Checkpoint(*first, 0, 8192); model.Publish(*first);
        model.Checkpoint(*second, 0, 8192); model.Publish(*second);
        // Keep only unrelated history, but forge all fingerprints to match
        // the first request. Residency remains valid; full token checks must
        // still reject both first-time and memoized collisions.
        for (auto it = model.prefixHistory.begin(); it != model.prefixHistory.end();) {
            if ((*it)->tokens.front() == first->currentTokens.front()) it = model.prefixHistory.erase(it);
            else {
                (*it)->tokenHash = PagedCacheManager::HashTokenPage(first->currentTokens.data(), (*it)->length);
                ++it;
            }
        }
        ++model.prefixPool->historyVersion;
        PrefillPrefixPreview preview; preview.Reset(first->currentTokens);
        for (int pass = 0; pass < 3; ++pass) {
            Require(model.PreviewPagedPrefixCacheExtra(first.get(), 8448, preview) == 0,
                    "preview accepted a fingerprint collision");
            ++model.prefixPool->historyVersion;
        }
        model.Drain(*first); model.Drain(*second);
    }

    static void PrefixFingerprints() {
        std::vector<int> input(200017);
        for (size_t i = 0; i < input.size(); ++i) input[i] = int((i * 731 + 17) % 151000);
        PrefillPrefixPreview preview;
        for (int pass = 0; pass < 2; ++pass) {
            preview.Reset(input);
            for (int length : {0, 1, 127, 128, 129, 200017, 81920, 98304, 65537, 130})
                Require(preview.TokenHash(length) == PagedCacheManager::HashTokenPage(input.data(), length),
                        "incremental fingerprint differs from full token hash");
            input[32]++;
        }
    }

    static void Capacity() {
        PrefixPool pool;
        pool.draftSlotBytes = 1023 * 4096 * 2;
        for (int ranks : {1, 2, 3, 4, 6, 8}) for (int slots : {1, 4, 17, 1536}) {
            pool.draftStorage.resize(ranks);
            int total = 0;
            for (int rank = 0; rank < ranks; ++rank) {
                const int count = pool.DraftShardSlots(slots, rank, ranks);
                total += count;
                Require(count <= (slots + ranks - 1) / ranks, "draft shard exceeds balanced budget");
            }
            Require(total == slots, "draft sharding changed checkpoint capacity");
            for (int slot = 0; slot < slots; ++slot) {
                const int rank = slot % ranks;
                Require(&pool.DraftShard(slot) == &pool.draftStorage[rank] &&
                        pool.DraftOffset(slot) + pool.draftSlotBytes <=
                            size_t(pool.DraftShardSlots(slots, rank, ranks)) * pool.draftSlotBytes,
                        "draft checkpoint falls outside its shard");
            }
        }
        pool.draftSlotBytes = 0;
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
        Require(model.prefixHistory.size() == 1600 && model.prefixPool->slots.free[0].empty(),
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

    static void PDGroupPreview() {
        NaiveN05PrefixCacheTestAccess model(3), fullOnly(3);
        model.tpDevices = {7};
        auto warm = model.Request(1);
        warm->pastKeyValues[0].first.dataDeviceIds = {7};
        model.Checkpoint(*warm, 0, 8192);
        model.Publish(*warm);
        model.Drain(*warm);

        // The other group has identical full-KV pages, but no SWA checkpoint.
        auto cold = fullOnly.Request(1);
        fullOnly.Publish(*cold);
        fullOnly.Drain(*cold);
        model.tpPagedCaches.insert(model.tpPagedCaches.begin(), std::move(fullOnly.tpPagedCaches.front()));
        model.tpDevices = {3, 7};
        model.attentionDP = 2;
        Require(model.PreviewPDResidentPrefix(warm->currentTokens, 0) == 0, "borrowed another group's SWA");
        Require(model.PreviewPDResidentPrefix(warm->currentTokens, 1) == 8192, "missed the owning group's prefix");
        Require(model.PreviewPDResidentPrefix(warm->currentTokens) == 8192, "unknown session missed a warm group");
    }

    static void PressureTelemetry() {
        for (bool fullPressure : {false, true}) {
            NaiveN05PrefixCacheTestAccess model(3);
            auto *pool = model.GetPagedKVCacheManager(0, true);
            if (fullPressure) pool->SetMaxPages(66);
            auto first = model.Request(1);
            model.Checkpoint(*first, 0, 8192); model.Checkpoint(*first, 8192, 200);
            model.Publish(*first); model.Drain(*first);
            const auto refs = pool->pageRefCount;
            Require(model.PreviewPDResidentPrefix(first->currentTokens) == 8320,
                    "PD resident preview did not include window boundary");
            Require(pool->pageRefCount == refs, "PD routing preview acquired KV references");
            auto second = model.Request(2);
            model.Checkpoint(*second, 0, 8192); model.Checkpoint(*second, 8192, 200);
            model.Publish(*second); model.Drain(*second);
            if (fullPressure) {
                Require(pool->CacheStats().pressureEvictedPages == 66 && model.prefixPool->orphanWindows > 0,
                        "full KV pressure was not distinguished from window eviction");
                Require(model.PreviewPDResidentPrefix(first->currentTokens) == 0, "routing revived evicted KV");
            } else {
                Require(pool->CacheStats().pressureEvictedPages == 0 && model.prefixPool->evictedWindows > 0,
                        "window-only pressure changed full KV eviction metrics");
            }
        }
    }

    static void AlignedExactRepeat() {
        for (int length : {12288, 16384, 65536}) {
            NaiveN05PrefixCacheTestAccess model(3);
            auto c = model.Request(1);
            auto *pool = model.GetPagedKVCacheManager(0, true);
            c->allTokens.resize(length, 101); c->currentTokens = c->allTokens;
            auto &pages = c->pastKeyValues[0].first.pageIndex;
            while ((int)pages.size() < length / 128) pages.push_back(pool->GetUnusedPageIndex(true));
            for (int past = 0; past < length; past += 4096) model.Checkpoint(*c, past, 4096);
            model.Publish(*c); model.Drain(*c);
            Require(model.PreviewPDResidentPrefix(c->currentTokens) == length - 128,
                    "aligned repeat lost the boundary needed for logits");
        }
    }

    static void PartialTailEviction() {
        NaiveN05PrefixCacheTestAccess model(3);
        auto *pool = model.GetPagedKVCacheManager(0, true);
        pool->SetMaxPages(66);
        auto c = model.Request(1);
        model.Checkpoint(*c, 0, 4096);
        model.Checkpoint(*c, 4096, 4096);
        model.Checkpoint(*c, 8192, 200);
        model.Publish(*c); model.Drain(*c);
        Require(model.FullHit(*c) == 8448 && model.WindowHit(*c) == 8320, "initial checkpoint changed");
        std::vector<int> allocated;
        for (int i = 0; i < 4; ++i) allocated.push_back(pool->GetUnusedPageIndex(true));
        Require(model.FullHit(*c) == 7936, "tail eviction removed too much full KV");
        Require(model.WindowHit(*c) >= 3968 && model.PreviewPDResidentPrefix(c->currentTokens) >= 3968,
                "small tail eviction invalidated every recovery checkpoint");
        Require(model.prefixHistory.size() <= 3, "checkpoint retention increased GPU slot budget");
        pool->ReleasePageIndices(allocated);
    }

    static void RetiredWindowReclaimsUnusablePages() {
        NaiveN05PrefixCacheTestAccess model(8);
        auto *pool = model.GetPagedKVCacheManager(0, true);
        pool->SetMaxPages(132);
        auto old = model.Request(1);
        model.Checkpoint(*old, 0, 8192); model.Publish(*old); model.Drain(*old);
        auto orphan = model.Request(2);
        model.Checkpoint(*orphan, 0, 8192); model.Publish(*orphan); model.Drain(*orphan);
        model.prefixHistory.erase(std::remove_if(model.prefixHistory.begin(), model.prefixHistory.end(),
            [&](const auto &entry) { return entry->tokens.front() == orphan->allTokens.front(); }), model.prefixHistory.end());
        std::vector<int> allocated;
        for (int i = 0; i < 16; ++i) allocated.push_back(pool->GetUnusedPageIndex(true));
        Require(model.PreviewPDResidentPrefix(old->currentTokens) == 8192,
                "newer unusable full KV displaced an older recoverable conversation");
        Require(model.PreviewPDResidentPrefix(orphan->currentTokens) == 0,
                "restored a prefix whose companion state was retired");
        pool->ReleasePageIndices(allocated);
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
        fastllm::NaiveN05PrefixCacheTestAccess::WindowOwnership();
        fastllm::NaiveN05PrefixCacheTestAccess::FinishBeforeDrain();
        fastllm::NaiveN05PrefixCacheTestAccess::BudgetPreviewOwnership();
        fastllm::NaiveN05PrefixCacheTestAccess::PublicationHashCollision();
        fastllm::NaiveN05PrefixCacheTestAccess::PreviewHashCollision();
        fastllm::NaiveN05PrefixCacheTestAccess::PrefixFingerprints();
        fastllm::NaiveN05PrefixCacheTestAccess::Capacity();
        fastllm::NaiveN05PrefixCacheTestAccess::PinnedPressure();
        fastllm::NaiveN05PrefixCacheTestAccess::FourPerRequest();
        fastllm::NaiveN05PrefixCacheTestAccess::EvictCompletedOnly();
        fastllm::NaiveN05PrefixCacheTestAccess::SharedReplacement();
        fastllm::NaiveN05PrefixCacheTestAccess::RollingConversations();
        fastllm::NaiveN05PrefixCacheTestAccess::PDGroupPreview();
        fastllm::NaiveN05PrefixCacheTestAccess::PressureTelemetry();
        fastllm::NaiveN05PrefixCacheTestAccess::AlignedExactRepeat();
        fastllm::NaiveN05PrefixCacheTestAccess::PartialTailEviction();
        fastllm::NaiveN05PrefixCacheTestAccess::RetiredWindowReclaimsUnusablePages();
        std::cout << "PREFIX CACHE PASS: capacity, pinned pressure, 400 independent prefixes, completed LRU\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
