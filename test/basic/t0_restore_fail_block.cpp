// T0 offline check for fix E item 4 (CPU only, header-only ledger): a failed SSD restore marks the
// checkpoint FAILED in memory so later admission skips it, without touching page ownership; after a
// restart (no ledger row) the failure is still recorded; a fresh write of the same key re-enables it.
#include "utils/prefix_cache_page_ledger.h"

#include <iostream>
#include <string>

using namespace fastllm;

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n";   \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static PrefixCachePageIdentity Page(const char *hash) {
    PrefixCachePageIdentity id;
    id.model = "qwen3.5";
    id.family = "qwen3.5";
    id.provenance = "input-prefix";
    id.dtype = "bf16";
    id.layout = "paged-kv-v4";
    id.media = "t0";
    id.parentHash = "";
    id.pageHash = hash;
    id.layer = 4;
    id.side = "kv";
    id.rank = 0;
    id.pageOrdinal = 1;
    return id;
}

static void Durable(PrefixCachePageLedger &ledger, const PrefixCachePageIdentity &page,
                    const std::string &key, const std::string &identity) {
    int manager = 0;
    CHECK(ledger.ObserveL1(page, &manager, 3, 1));
    CHECK(ledger.RegisterCheckpoint(key, identity, {page}) != 0);
    PrefixCachePageLedger::CheckpointSnapshot root;
    CHECK(ledger.Checkpoint(key, root));
    CHECK(ledger.FinishCheckpoint(key, identity, root.generation, true));
}

int main() {
    // Tracked root (same process): durable -> restore fails -> blocked, bytes/pages untouched.
    {
        PrefixCachePageLedger ledger;
        const auto page = Page("p1");
        Durable(ledger, page, "k1", "id1");
        CHECK(ledger.CanRestoreCheckpoint("k1", "id1"));
        CHECK(!ledger.FailRestoreCheckpoint("k1", "other-identity", "x"));   // identity must match
        CHECK(ledger.FailRestoreCheckpoint("k1", "id1", "ssd_import_identity_mismatch"));
        CHECK(!ledger.CanRestoreCheckpoint("k1", "id1"));
        CHECK(!ledger.CanEraseCheckpoint("k1", "id1"));   // stated side effect: FIFO keeps it
        PrefixCachePageSnapshot s;
        CHECK(ledger.Snapshot(page, s) && s.runtimeTier == PrefixCacheRuntimeTier::L1 &&
              !s.failureQuarantined && s.durableCheckpointRoots.size() == 1);
        PrefixCachePageLedger::CheckpointSnapshot root;
        CHECK(ledger.Checkpoint("k1", root) && root.failureReason == "ssd_import_identity_mismatch");
        CHECK(ledger.FailRestoreCheckpoint("k1", "id1", "again"));   // idempotent on FAILED
        // A fresh write of the same key re-enables it (RegisterCheckpoint accepts a FAILED row).
        CHECK(ledger.RegisterCheckpoint("k1", "id1", {page}) != 0);
        CHECK(ledger.Checkpoint("k1", root));
        CHECK(ledger.FinishCheckpoint("k1", "id1", root.generation, true));
        CHECK(ledger.CanRestoreCheckpoint("k1", "id1"));
    }
    // Untracked root (after restart): first failure creates a FAILED row; admission then rejects.
    {
        PrefixCachePageLedger ledger;
        PrefixCachePageLedger::CheckpointSnapshot root;
        CHECK(!ledger.Checkpoint("k2", root));
        CHECK(!ledger.FailRestoreCheckpoint("k2", "id2", "x"));      // tracked-only API refuses
        CHECK(!ledger.RegisterFailedRestoreCheckpoint("", "id2", "x"));
        CHECK(ledger.RegisterFailedRestoreCheckpoint("k2", "id2", "gpu_restore_failed"));
        CHECK(ledger.Checkpoint("k2", root) && root.state == PrefixCachePageLedger::CheckpointState::FAILED &&
              !root.writerActive && root.pages.empty());
        CHECK(!ledger.CanRestoreCheckpoint("k2", "id2"));
        CHECK(!ledger.CanEraseCheckpoint("k2", "id2"));
        CHECK(ledger.OldestEvictableCheckpoint().empty());
        CHECK(!ledger.RegisterFailedRestoreCheckpoint("k2", "other-identity", "x"));
    }
    // A write still in progress is never blocked by a read failure.
    {
        PrefixCachePageLedger ledger;
        const auto page = Page("p3");
        int manager = 0;
        CHECK(ledger.ObserveL1(page, &manager, 4, 1));
        CHECK(ledger.RegisterCheckpoint("k3", "id3", {page}) != 0);   // WRITE_PENDING, writer active
        CHECK(!ledger.FailRestoreCheckpoint("k3", "id3", "x"));
        CHECK(!ledger.RegisterFailedRestoreCheckpoint("k3", "id3", "x"));
    }
    if (failures) {
        std::cout << "FAIL: t0 restore-fail block (" << failures << ")\n";
        return 1;
    }
    std::cout << "PASS: t0 restore-fail block\n";
    return 0;
}
