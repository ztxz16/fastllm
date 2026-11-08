// T0 offline check for S1 gate 3 / acceptance gate G7 (CPU only, no GPU, no model, never the
// product SSD directory): after an L1 reclaim is abandoned (AbortReclaim), the page's last
// durable checkpoint must not be erasable, neither by the ledger nor by v4 FIFO admission.
// Part A uses the real header-only PrefixCachePageLedger. Part B wires the real ledger into the
// real DiskPrefixCache/DiskPrefixV4Store with the same predicate/erase lambdas as
// qwen3_5_persistent.inc ~1127-1138 (keep in sync). This does NOT prove that the CUDA spill
// path reaches AbortReclaim; that needs the runtime attempt in S1-GATE3-PLAN.md G5.
#include "utils/disk_prefix_cache.h"
#include "utils/prefix_cache_page_ledger.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace fastllm;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n";   \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static PrefixCachePageIdentity Page(const char *stream, const char *hash) {
    PrefixCachePageIdentity id;
    id.model = "qwen3.5";
    id.family = "qwen3.5";
    id.provenance = "input-prefix";
    id.dtype = "bf16";
    id.layout = "paged-kv-v4";
    id.media = stream;
    id.parentHash = "";
    id.pageHash = hash;
    id.layer = 4;
    id.side = "kv";
    id.rank = 0;
    id.pageOrdinal = 1;
    return id;
}

static bool HasRoot(const PrefixCachePageSnapshot &s, const std::string &key) {
    return std::find(s.durableCheckpointRoots.begin(), s.durableCheckpointRoots.end(), key) !=
           s.durableCheckpointRoots.end();
}

// Page resident in L1 with one durable root `key`, as after a successful background save.
static void SeedDurable(PrefixCachePageLedger &ledger, const PrefixCachePageIdentity &page,
                        const void *manager, const std::string &key, const std::string &identity) {
    CHECK(ledger.ObserveL1(page, manager, 7, 3));
    CHECK(ledger.RegisterCheckpoint(key, identity, {page}) != 0);
    PrefixCachePageLedger::CheckpointSnapshot root;
    CHECK(ledger.Checkpoint(key, root));
    CHECK(ledger.FinishCheckpoint(key, identity, root.generation, true));
}

// Part A: ledger transition and veto (S1-GATE3-PLAN.md G3 steps 1-7).
static void LedgerAbortKeepsSoleRoot() {
    PrefixCachePageLedger ledger;
    int manager = 0;
    const auto page = Page("g7", "g7-page");
    const std::string key = "root-g7", identity = "identity-g7";
    SeedDurable(ledger, page, &manager, key, identity);

    // While the page is still in L1 the SSD copy is not the only copy.
    CHECK(ledger.CanEraseCheckpoint(key, identity));

    auto lease = ledger.BeginReclaim(page, 5);
    CHECK(lease.Valid());
    CHECK(!ledger.CanEraseCheckpoint(key, identity));  // MIGRATING: veto

    CHECK(ledger.AbortReclaim(lease, "t0_admission_failed"));
    PrefixCachePageSnapshot s;
    CHECK(ledger.Snapshot(page, s));
    std::cout << "after_abort: tier=" << int(s.runtimeTier) << " state=" << int(s.runtimeState)
              << " durable=" << int(s.durableState) << " l1_page=" << s.l1Page
              << " l1_generation=" << s.l1Generation << " roots=" << s.durableCheckpointRoots.size()
              << " quarantined=" << s.failureQuarantined << "\n";
    CHECK(s.runtimeTier == PrefixCacheRuntimeTier::NONE);
    CHECK(s.runtimeState == PrefixCacheRuntimeState::RESIDENT);
    CHECK(s.durableState == PrefixCacheDurableState::DURABLE);
    CHECK(s.l1Manager == nullptr && s.l1Page == -1 && s.l1Generation == 0);
    CHECK(s.durableCheckpointRoots.size() == 1 && HasRoot(s, key));
    CHECK(!s.failureQuarantined && s.activeLeaseSerial == 0);

    // The sole remaining copy: no erase, no FIFO pick, survives trimming, still restorable.
    CHECK(!ledger.CanEraseCheckpoint(key, identity));
    CHECK(ledger.OldestEvictableCheckpoint().empty());
    ledger.TrimInactive(1);
    CHECK(ledger.Snapshot(page, s) && HasRoot(s, key));
    CHECK(ledger.CanRestoreCheckpoint(key, identity));

    // A consumed or stale lease is a no-op.
    CHECK(!ledger.AbortReclaim(lease, "again"));
    CHECK(ledger.Snapshot(page, s) && s.runtimeTier == PrefixCacheRuntimeTier::NONE);

    // Re-published in L1 later: SSD copy is no longer the only copy, erase allowed again.
    CHECK(ledger.ObserveL1(page, &manager, 9, 4));
    CHECK(ledger.CanEraseCheckpoint(key, identity));
}

// Part A: two roots after abort -> the older may go, the last one may not.
static void LedgerAbortTwoRoots() {
    PrefixCachePageLedger ledger;
    int manager = 0;
    const auto page = Page("g7-two", "g7-two-page");
    SeedDurable(ledger, page, &manager, "root-old", "identity");
    CHECK(ledger.RegisterCheckpoint("root-new", "identity", {page}) != 0);
    PrefixCachePageLedger::CheckpointSnapshot root;
    CHECK(ledger.Checkpoint("root-new", root));
    CHECK(ledger.FinishCheckpoint("root-new", "identity", root.generation, true));
    auto lease = ledger.BeginReclaim(page, 1);
    CHECK(lease.Valid() && ledger.AbortReclaim(lease, "t0"));
    CHECK(ledger.CanEraseCheckpoint("root-old", "identity"));
    CHECK(ledger.EraseCheckpoint("root-old", "identity"));
    CHECK(!ledger.CanEraseCheckpoint("root-new", "identity"));
    PrefixCachePageSnapshot s;
    CHECK(ledger.Snapshot(page, s) && s.durableCheckpointRoots.size() == 1 && HasRoot(s, "root-new"));
}

// Contrast: a failed copy (FailMigration) quarantines instead; the root is also kept.
static void LedgerFailMigrationContrast() {
    PrefixCachePageLedger ledger;
    int manager = 0;
    const auto page = Page("g7-fail", "g7-fail-page");
    SeedDurable(ledger, page, &manager, "root-fail", "identity");
    auto lease = ledger.BeginReclaim(page, 1);
    CHECK(lease.Valid() && ledger.FailMigration(lease, "d2h failed"));
    CHECK(!ledger.CanEraseCheckpoint("root-fail", "identity"));
    PrefixCachePageSnapshot s;
    CHECK(ledger.Snapshot(page, s) && s.failureQuarantined && HasRoot(s, "root-fail"));
}

static DiskPrefixV4Store::Object MakeObject(const std::string &bytes) {
    DiskPrefixV4Store::Object object;
    object.bytes = bytes;
    object.kind = 1;
    object.hash = DiskPrefixCache::Digest(bytes);
    return object;
}

// Physical-log size of two compacted roots, same method as test_disk_prefix_store_v4.cpp
// MeasureTwoCompactedPhysical.
static uint64_t MeasureTwo(const fs::path &root, const DiskPrefixV4Store::Object &a,
                           const DiskPrefixV4Store::Object &b) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return 0;
    const std::string identity = DiskPrefixCache::Digest("g7-sizing-identity");
    CHECK(store->Commit("sizing-a", identity, {a}, nullptr, error));
    CHECK(store->Commit("sizing-b", identity, {b}, nullptr, error));
    CHECK(store->Compact(error));
    return store->PhysicalBytes();
}

// Part B: real v4 FIFO admission consults the real ledger (S1-GATE3-PLAN.md G3 steps 8-9).
static void StoreFifoSkipsAbortedSoleRoot(const fs::path &base) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("g7-store-identity");
    const auto targetBytes = MakeObject(std::string(4096, 'T'));
    const auto controlBytes = MakeObject(std::string(4096, 'C'));
    const auto incomingBytes = MakeObject(std::string(4096, 'N'));
    const uint64_t quota = MeasureTwo(base / "sizing", targetBytes, controlBytes) + 8192;
    const fs::path root = base / "store";
    setenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT", "1", 1);
    auto store = DiskPrefixV4Store::OpenShared(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) return;
    std::vector<std::pair<std::string, std::string>> erased;
    auto observer = store->ObserveCheckpointErases(
        [&](const std::string &key, const std::string &id) { erased.emplace_back(key, id); });

    auto ledger = std::make_shared<PrefixCachePageLedger>();
    DiskPrefixCache cache(root.string(), identity, quota, identity, 1);
    std::weak_ptr<PrefixCachePageLedger> weakLedger = ledger;
    cache.SetCheckpointReclaimPredicate(
        [weakLedger](const std::string &key, const std::string &checkpointIdentity) {
            auto l = weakLedger.lock();
            return l && l->CanEraseCheckpoint(key, checkpointIdentity);
        });
    cache.SetCheckpointEraseCallback(
        [weakLedger](const std::string &key, const std::string &checkpointIdentity) {
            auto l = weakLedger.lock();
            if (l) l->EraseCheckpoint(key, checkpointIdentity);
        });

    // "target" is older (FIFO would pick it first); "control" is newer.
    CHECK(store->Commit("target", identity, {targetBytes}, nullptr, error));
    CHECK(store->Commit("control", identity, {controlBytes}, nullptr, error));
    CHECK(store->LiveCheckpoints() == 2);
    int manager = 0;
    const auto targetPage = Page("g7-store", "target-page");
    const auto controlPage = Page("g7-store", "control-page");
    SeedDurable(*ledger, targetPage, &manager, "target", identity);
    SeedDurable(*ledger, controlPage, &manager, "control", identity);
    // Reclaim of the target page is abandoned: its SSD root is now the only copy.
    auto lease = ledger->BeginReclaim(targetPage, 11);
    CHECK(lease.Valid() && ledger->AbortReclaim(lease, "t0_admission_failed"));
    // Control page also loses L1 the same way: both roots are sole copies.
    auto controlLease = ledger->BeginReclaim(controlPage, 12);
    CHECK(controlLease.Valid() && ledger->AbortReclaim(controlLease, "t0_admission_failed"));

    // B1: every candidate vetoed -> admission fails closed, nothing erased.
    const bool b1 = store->Commit("incoming", identity, {incomingBytes}, nullptr, error);
    std::cout << "b1: committed=" << b1 << " error=" << error << " erased=" << erased.size()
              << " live=" << store->LiveCheckpoints() << "\n";
    CHECK(!b1 && error == "capacity_unavailable");
    CHECK(erased.empty());
    CHECK(store->LiveCheckpoints() == 2);

    // B2: control is back in L1 (erasable); target still sole copy. FIFO must skip the older
    // target and retire control instead.
    CHECK(ledger->ObserveL1(controlPage, &manager, 8, 5));
    erased.clear();
    const bool b2 = store->Commit("incoming", identity, {incomingBytes}, nullptr, error);
    std::cout << "b2: committed=" << b2 << " error=" << error << " erased=" << erased.size();
    for (const auto &e : erased) std::cout << " " << e.first;
    std::cout << "\n";
    CHECK(erased.size() == 1);
    CHECK(!erased.empty() && erased.front().first == "control");
    for (const auto &e : erased) CHECK(e.first != "target");

    // Target is still loadable from the store and still recorded in the ledger.
    std::string loadedIdentity;
    std::vector<std::string> bytes;
    CHECK(store->Load("target", loadedIdentity, bytes, error));
    CHECK(bytes.size() == 1 && bytes[0] == targetBytes.bytes);
    PrefixCachePageSnapshot s;
    CHECK(ledger->Snapshot(targetPage, s) && HasRoot(s, "target"));
    CHECK(ledger->Snapshot(controlPage, s) && !HasRoot(s, "control"));  // erase callback ran

    observer.reset();
    cache.SetCheckpointReclaimPredicate(nullptr);
    cache.SetCheckpointEraseCallback(nullptr);
    unsetenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT");
}

int main(int argc, char **argv) {
    // The store fixture root must be given explicitly and must not be under /tmp or the product SSD.
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <empty fixture dir>\n";
        return 2;
    }
    const fs::path base = fs::absolute(argv[1]);
    const std::string b = base.string();
    if (b.rfind("/tmp", 0) == 0 || b.find("/LLM/0925/SSD") != std::string::npos || fs::exists(base)) {
        std::cerr << "REFUSED fixture dir: " << b << " (must be new, not /tmp, not SSD)\n";
        return 2;
    }
    fs::create_directories(base);

    LedgerAbortKeepsSoleRoot();
    LedgerAbortTwoRoots();
    LedgerFailMigrationContrast();
    StoreFifoSkipsAbortedSoleRoot(base);

    if (failures) {
        std::cout << "FAIL: t0 abort-reclaim keeps backup (" << failures << ")\n";
        return 1;
    }
    std::cout << "PASS: t0 abort-reclaim keeps backup (ledger veto + v4 FIFO skip)\n";
    return 0;
}
