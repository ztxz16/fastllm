// LEDGERBOUND unit test, header-only ledger. Modes (by environment):
//   FASTLLM_PREFIX_LEDGER_BOUND=0                       -> unbounded (every durable row and root ref kept)
//   FASTLLM_PREFIX_LEDGER_MAX_PAGE_REFS=100             -> bounded at 100 page refs
//   nothing set (V5 default, since 2026-11-08)          -> bound on, budget 0: only the newest idle row kept
//   FASTLLM_PREFIX_V5=0 (V4) or FASTLLM_PREFIX_CACHE_L2_HOST_KV=1 -> bound on, budget 1048576 (old protection)
// Build: g++ -std=c++17 -Iinclude test/basic/t0_ledger_bound.cpp -lpthread
#include "utils/prefix_cache_page_ledger.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) \
    do { if (!(condition)) throw std::runtime_error("ledger bound check failed: " #condition); } while (0)

using namespace fastllm;

namespace {
PrefixCachePageIdentity Page(uint64_t ordinal) {
    PrefixCachePageIdentity id;
    id.model = "qwen3.8-27b-fp8-with-a-long-identity-string";
    id.family = "qwen3.5";
    id.provenance = "input-prefix-provenance-digest-0123456789abcdef0123456789abcdef";
    id.dtype = "fp8_e4m3";
    id.layout = "paged-kv-v4";
    id.media = "text";
    id.parentHash = ordinal == 0 ? "" : "parent-" + std::to_string(ordinal - 1);
    id.pageHash = "page-hash-0123456789abcdef0123456789abcdef-" + std::to_string(ordinal);
    id.layer = 3;
    id.side = "k";
    id.rank = 0;
    id.pageOrdinal = ordinal;
    return id;
}

PrefixCachePageIdentity Root(int k) {
    auto id = Page(100000 + k);
    id.media = "checkpoint-root";
    return id;
}

// One long request: checkpoint k covers pages [0, k * step), like chunked prefill checkpoints.
void RegisterChain(PrefixCachePageLedger &ledger, int count, int step, std::vector<std::string> &keys) {
    for (int k = 1; k <= count; ++k) {
        std::vector<PrefixCachePageIdentity> pages;
        for (int p = 0; p < k * step; ++p) pages.push_back(Page(p));
        const std::string key = "ckpt-" + std::to_string(k);
        const uint64_t generation = ledger.RegisterCheckpoint(key, Root(k), pages);
        CHECK(generation != 0);
        CHECK(ledger.FinishCheckpoint(key, Root(k), generation, true));
        keys.push_back(key);
    }
}

void TestUnbounded() {
    PrefixCachePageLedger ledger;
    std::vector<std::string> keys;
    RegisterChain(ledger, 12, 10, keys);
    const auto stats = ledger.Stats();
    CHECK(stats.checkpoints == 12);
    CHECK(stats.forgotten == 0);
    size_t expectedRoots = 0;
    for (int k = 1; k <= 12; ++k) expectedRoots += size_t(k * 10);  // every page lists every root containing it
    CHECK(stats.rootRefs == expectedRoots);
    for (const auto &key : keys) {
        PrefixCachePageLedger::CheckpointSnapshot snap;
        CHECK(ledger.Checkpoint(key, snap));
        CHECK(snap.state == PrefixCachePageLedger::CheckpointState::DURABLE);
    }
    CHECK(ledger.CanRestoreCheckpoint(keys.front(), Root(1).CanonicalKey()));
    std::printf("unbounded: checkpoints=%zu root_refs=%zu forgotten=%zu\n",
                stats.checkpoints, stats.rootRefs, stats.forgotten);
}

void TestBounded() {
    CHECK(PrefixCachePageLedger::LedgerBoundEnabled());
    CHECK(PrefixCachePageLedger::LedgerMaxDurablePageRefs() == 100);
    PrefixCachePageLedger ledger;
    std::vector<std::string> keys;
    RegisterChain(ledger, 12, 10, keys);  // rows of 10, 20, ..., 120 pages
    const auto stats = ledger.Stats();
    // Only the newest idle row survives once a single row exceeds the budget; it is never forgotten.
    CHECK(stats.checkpoints == 1);
    CHECK(stats.forgotten == 11);
    CHECK(stats.durablePageRefs == 120);
    CHECK(stats.rootRefs == 120);  // the remaining row's own references, nothing stale
    PrefixCachePageLedger::CheckpointSnapshot snap;
    CHECK(ledger.Checkpoint(keys.back(), snap));
    CHECK(snap.state == PrefixCachePageLedger::CheckpointState::DURABLE);
    CHECK(ledger.CanRestoreCheckpoint(keys.back(), Root(12).CanonicalKey()));
    // A forgotten row is untracked exactly as after a restart: not found; erasable as any untracked root.
    CHECK(!ledger.Checkpoint(keys.front(), snap));
    CHECK(ledger.CanEraseCheckpoint(keys.front(), Root(1).CanonicalKey()) ==
          PrefixCachePageLedger::UnknownCheckpointErasable());
    // Re-registering a forgotten key starts a fresh row (as after a restart).
    std::vector<PrefixCachePageIdentity> pages{Page(0)};
    CHECK(ledger.RegisterCheckpoint(keys.front(), Root(1), pages) != 0);

    // Rows below the budget are all kept.
    PrefixCachePageLedger small;
    std::vector<std::string> smallKeys;
    RegisterChain(small, 4, 10, smallKeys);  // 10 + 20 + 30 + 40 = 100 page refs, equal to the budget
    CHECK(small.Stats().checkpoints == 4);
    CHECK(small.Stats().forgotten == 0);
    CHECK(small.Stats().durablePageRefs == 100);

    // A row with a page resident in L2 is never forgotten (L2 eviction needs its durable root).
    PrefixCachePageLedger l2;
    int manager = 0;
    const auto first = Page(0);
    CHECK(l2.ObserveL1(first, &manager, 7, 3));
    std::vector<std::string> l2Keys;
    RegisterChain(l2, 1, 60, l2Keys);  // row 1: pages 0..59, contains the L1 page
    auto reclaim = l2.BeginReclaim(first, 1);
    CHECK(reclaim.Valid());
    CHECK(l2.CommitL2(reclaim, 0x101));
    std::vector<PrefixCachePageIdentity> later;
    for (int p = 1000; p < 1080; ++p) later.push_back(Page(p));
    const uint64_t g = l2.RegisterCheckpoint("other", Root(50), later);
    CHECK(g != 0 && l2.FinishCheckpoint("other", Root(50), g, true));  // 60 + 80 > 100
    CHECK(l2.Checkpoint(l2Keys.front(), snap));                       // kept: page 0 is in L2
    std::printf("bounded: checkpoints=%zu root_refs=%zu durable_page_refs=%zu forgotten=%zu\n",
                stats.checkpoints, stats.rootRefs, stats.durablePageRefs, stats.forgotten);
}
}  // namespace

// V5 default: budget 0 -> only the newest idle durable row stays; inactive page records go with the rows.
void TestV5Default() {
    CHECK(PrefixCachePageLedger::LedgerBoundEnabled());
    CHECK(PrefixCachePageLedger::LedgerMaxDurablePageRefs() == 0);
    CHECK(PrefixCachePageLedger::UnknownCheckpointErasable());
    PrefixCachePageLedger ledger;
    std::vector<std::string> keys;
    RegisterChain(ledger, 12, 10, keys);
    const auto stats = ledger.Stats();
    CHECK(stats.checkpoints == 1);
    CHECK(stats.forgotten == 11);
    CHECK(stats.rootRefs == 120);
    CHECK(stats.pages == 120);   // only the newest row's pages are still referenced
    CHECK(ledger.CanRestoreCheckpoint(keys.back(), Root(12).CanonicalKey()));
    PrefixCachePageLedger::CheckpointSnapshot snap;
    CHECK(!ledger.Checkpoint(keys.front(), snap));
    CHECK(ledger.CanEraseCheckpoint(keys.front(), Root(1).CanonicalKey()));   // untracked -> store may reclaim
    // A row still being written is never forgotten.
    std::vector<PrefixCachePageIdentity> pages{Page(5000), Page(5001)};
    const uint64_t g = ledger.RegisterCheckpoint("pending", Root(77), pages);
    CHECK(g != 0);
    RegisterChain(ledger, 2, 10, keys);
    CHECK(ledger.Checkpoint("pending", snap));
    // Rows with disjoint pages: forgotten rows' page records must not stay behind as inactive records.
    CHECK(PrefixCachePageLedger::LedgerDefaultMaxRecords() == 262144);   // no sweep on the restore path
    PrefixCachePageLedger disjoint;
    for (int k = 0; k < 6; ++k) {
        std::vector<PrefixCachePageIdentity> own;
        for (int p = 0; p < 10; ++p) own.push_back(Page(20000 + k * 100 + p));
        const std::string key = "disjoint-" + std::to_string(k);
        const uint64_t gen = disjoint.RegisterCheckpoint(key, Root(200 + k), own);
        CHECK(gen != 0 && disjoint.FinishCheckpoint(key, Root(200 + k), gen, true));
    }
    CHECK(disjoint.Stats().checkpoints == 1);
    CHECK(disjoint.Stats().pages == 10);   // only the newest row's pages
    CHECK(disjoint.CanRestoreCheckpoint("disjoint-5", Root(205).CanonicalKey()));
    // a page shared by an older DURABLE row and a WRITE_PENDING row must keep its record when
    // the DURABLE row is forgotten, so the pending row still becomes restorable when it finishes.
    PrefixCachePageLedger shared;
    std::vector<PrefixCachePageIdentity> sp{Page(30000), Page(30001)};
    const uint64_t g0 = shared.RegisterCheckpoint("old", Root(300), sp);
    CHECK(g0 != 0 && shared.FinishCheckpoint("old", Root(300), g0, true));
    const uint64_t gp = shared.RegisterCheckpoint("pending", Root(301), sp);   // same pages, still writing
    CHECK(gp != 0);
    CHECK(shared.SetCheckpointActivity("old", Root(300).CanonicalKey(), 0, false));
    std::vector<PrefixCachePageIdentity> other{Page(30100)};
    const uint64_t g1 = shared.RegisterCheckpoint("newer", Root(302), other);
    CHECK(g1 != 0 && shared.FinishCheckpoint("newer", Root(302), g1, true));   // forgets "old"
    CHECK(!shared.Checkpoint("old", snap));
    // keep "newer" busy so finishing "pending" does not forget "pending" itself as an older idle row
    CHECK(shared.SetCheckpointActivity("newer", Root(302).CanonicalKey(), 1, false));
    CHECK(shared.FinishCheckpoint("pending", Root(301), gp, true));
    CHECK(shared.CanRestoreCheckpoint("pending", Root(301).CanonicalKey()));
    std::printf("v5-default: checkpoints=%zu pages=%zu root_refs=%zu forgotten=%zu\n",
                stats.checkpoints, stats.pages, stats.rootRefs, stats.forgotten);
}

int main() {
    try {
        const char *budget = std::getenv("FASTLLM_PREFIX_LEDGER_MAX_PAGE_REFS");
        if (!PrefixCachePageLedger::LedgerBoundEnabled()) TestUnbounded();
        else if (budget != nullptr) TestBounded();
        else if (PrefixCachePageLedger::V5LeanLedger()) TestV5Default();
        else { CHECK(PrefixCachePageLedger::LedgerMaxDurablePageRefs() == (size_t(1) << 20)); TestUnbounded(); }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
