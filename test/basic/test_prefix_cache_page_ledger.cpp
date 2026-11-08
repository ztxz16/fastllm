#include "utils/prefix_cache_page_ledger.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <thread>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition) \
    do { if (!(condition)) throw std::runtime_error("page ledger assertion failed: " #condition); } while (0)

using namespace fastllm;

namespace {
PrefixCachePageIdentity Page(const char *stream, uint64_t ordinal, const char *parent,
                             const char *hash) {
    PrefixCachePageIdentity id;
    id.model = "qwen3.5";
    id.family = "qwen3.5";
    id.provenance = "input-prefix";
    id.dtype = "bf16";
    id.layout = "paged-kv-v4";
    id.media = stream;
    id.parentHash = parent;
    id.pageHash = hash;
    id.layer = 4;
    id.side = "kv";
    id.rank = 0;
    id.pageOrdinal = ordinal;
    return id;
}

void TestReclaimAndPromotion() {
    PrefixCachePageLedger ledger;
    const auto page = Page("text", 1, "", "page-1");
    int manager = 0;
    assert(ledger.ObserveL1(page, &manager, 7, 3));

    auto held = ledger.AcquireRuntime(page);
    assert(held.Valid());
    assert(!ledger.BeginReclaim(page, 1).Valid());
    assert(ledger.ReleaseRuntime(held));

    auto reclaim = ledger.BeginReclaim(page, 1);
    assert(reclaim.Valid());
    auto stale = reclaim;
    stale.serial++;
    assert(!ledger.CommitL2(stale, 0x101));
    assert(ledger.CommitL2(reclaim, 0x101));

    PrefixCachePageSnapshot snapshot;
    assert(ledger.Snapshot(page, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::L2);
    assert(snapshot.l1Page == -1);
    assert(snapshot.l2HostAllocation == 0x101);

    auto promotion = ledger.BeginPromotion(page);
    assert(promotion.Valid());
    assert(!ledger.BeginPromotion(page).Valid());
    assert(ledger.CommitL1(promotion, &manager, 9, 4));
    assert(ledger.Snapshot(page, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::L1);
    assert(snapshot.l2HostAllocation == 0);
}

void TestFailedMigrationIsQuarantined() {
    PrefixCachePageLedger ledger;
    const auto page = Page("failure", 1, "", "failed-page");
    int manager = 0;
    assert(ledger.ObserveL1(page, &manager, 1, 1));
    auto reclaim = ledger.BeginReclaim(page, 17);
    assert(reclaim.Valid());
    assert(ledger.FailMigration(reclaim, "D2H event failed"));
    assert(!ledger.BeginReclaim(page, 18).Valid());
    PrefixCachePageSnapshot snapshot;
    assert(ledger.Snapshot(page, snapshot));
    assert(snapshot.failureQuarantined);
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::MIGRATING);
    assert(snapshot.l1Page == 1); // failed migration retains the source owner
}

void TestDurableRootsAndFifo() {
    PrefixCachePageLedger ledger;
    int manager = 0;
    const auto page = Page("shared", 1, "", "shared-page");
    assert(ledger.ObserveL1(page, &manager, 3, 1));
    auto reclaim = ledger.BeginReclaim(page, 1);
    assert(reclaim.Valid() && ledger.CommitL2(reclaim, 11));

    assert(ledger.RegisterCheckpoint("root-old", "identity", {page}, 10) != 0);
    PrefixCachePageLedger::CheckpointSnapshot root;
    assert(ledger.Checkpoint("root-old", root));
    assert(ledger.FinishCheckpoint("root-old", "identity", root.generation, true));
    assert(ledger.RegisterCheckpoint("root-new", "identity", {page}, 20) != 0);
    assert(ledger.Checkpoint("root-new", root));
    assert(ledger.FinishCheckpoint("root-new", "identity", root.generation, true));

    assert(ledger.CanEvictL2(page));
    assert(ledger.OldestEvictableCheckpoint() == "root-old");
    assert(ledger.EraseCheckpoint("root-old", "identity"));
    assert(ledger.CanEvictL2(page)); // root-new still makes the page durable
    assert(ledger.EraseCheckpoint("root-new", "identity"));
    assert(!ledger.CanEvictL2(page));

    // A failed write must be observable and must not masquerade as durable.
    const auto failed = Page("failed-root", 1, "", "failed-root-page");
    assert(ledger.RegisterCheckpoint("root-failed", "identity", {failed}, 30) != 0);
    assert(ledger.Checkpoint("root-failed", root));
    assert(ledger.FinishCheckpoint("root-failed", "identity", root.generation, false, "disk error"));
    PrefixCachePageSnapshot failedSnapshot;
    assert(ledger.Snapshot(failed, failedSnapshot));
    assert(failedSnapshot.durableState == PrefixCacheDurableState::FAILED);
}

void TestLongestCandidateDirectRestore() {
    const auto first = Page("candidate", 1, "", "candidate-1");
    const auto second = Page("candidate", 2, "candidate-1", "candidate-2");
    PrefixCacheCandidate l2;
    l2.tier = PrefixCacheCandidateTier::L2;
    l2.tokens = 131072;
    l2.complete = l2.contiguous = l2.identityValid = true;
    l2.pages = {first};
    PrefixCacheCandidate l3 = l2;
    l3.tier = PrefixCacheCandidateTier::L3;
    l3.tokens = 157696;
    l3.pages = {first, second};
    PrefixCacheSelection selection;
    assert(SelectLongestPrefix({l2, l3}, selection));
    assert(selection.valid && selection.tokens == 157696);
    assert(selection.directToL1 && !selection.promoteL2);
    assert(selection.tier == PrefixCacheCandidateTier::L3);
}

void TestConcurrentReadLease() {
    PrefixCachePageLedger ledger;
    const auto page = Page("concurrent", 1, "", "concurrent-page");
    int manager = 0;
    assert(ledger.ObserveL1(page, &manager, 1, 1));
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&ledger, &page]() {
            for (int n = 0; n < 100; ++n) {
                auto token = ledger.AcquireRuntime(page);
                assert(token.Valid());
                assert(ledger.ReleaseRuntime(token));
            }
        });
    }
    for (auto &worker : workers) worker.join();
    assert(ledger.BeginReclaim(page, 1).Valid());
}

void TestAtomicMultiPoolAdoption() {
    PrefixCachePageLedger ledger;
    int managerA = 0, managerB = 0;
    auto first = Page("batch", 0, "", "batch-0");
    auto second = Page("batch", 1, "batch-0", "batch-1");
    second.layer = 5;
    second.side = "value";
    second.rank = 1;
    assert(ledger.ObserveL1(first, &managerA, 1, 1));
    assert(ledger.ObserveL1(second, &managerB, 2, 1));
    auto reclaimA = ledger.BeginReclaim(first, 10);
    auto reclaimB = ledger.BeginReclaim(second, 11);
    assert(reclaimA.Valid() && reclaimB.Valid());
    assert(ledger.CommitL2(reclaimA, 101));
    assert(ledger.CommitL2(reclaimB, 102));
    auto promoteA = ledger.BeginPromotion(first);
    auto promoteB = ledger.BeginPromotion(second);
    assert(promoteA.Valid() && promoteB.Valid());

    auto staleB = promoteB;
    ++staleB.serial;
    std::vector<PrefixCachePageLedger::L1Adoption> failed = {
        {first, promoteA, &managerA, 21, 2, {}, {}},
        {second, staleB, &managerB, 22, 2, {}, {}}};
    assert(!ledger.CommitL1Batch(failed));
    PrefixCachePageSnapshot snapshot;
    assert(ledger.Snapshot(first, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::MIGRATING);
    assert(ledger.Snapshot(second, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::MIGRATING);

    std::vector<PrefixCachePageLedger::L1Adoption> committed = {
        {first, promoteA, &managerA, 21, 2, {}, {}},
        {second, promoteB, &managerB, 22, 2, {}, {}}};
    assert(ledger.CommitL1Batch(committed));
    assert(ledger.Snapshot(first, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::L1 && snapshot.l1Page == 21);
    assert(ledger.Snapshot(second, snapshot));
    assert(snapshot.runtimeTier == PrefixCacheRuntimeTier::L1 && snapshot.l1Page == 22);

    // Two logical pages must never claim the same physical manager/page pair.
    std::vector<PrefixCachePageLedger::L1Adoption> duplicateOwner = {
        {first, {}, &managerA, 21, 3, {}, {}},
        {second, {}, &managerA, 21, 3, {}, {}}};
    assert(!ledger.CommitL1Batch(duplicateOwner));
}

struct LargeBatch {
    int manager = 0;
    std::vector<PrefixCachePageIdentity> ids;
    std::vector<PrefixCachePageLedger::L1Adoption> adoptions;
    explicit LargeBatch(size_t n=8192) {
        ids.reserve(n); adoptions.reserve(n);
        for(size_t i=0;i<n;++i) {
            const auto hash="large-"+std::to_string(i);
            const auto parent=i?"large-"+std::to_string(i-1):"";
            ids.push_back(Page("large-preparation",i,parent.c_str(),hash.c_str()));
            adoptions.push_back({ids.back(),{},&manager,(int64_t)i,2,{},{}});
        }
    }
};
void TestLargeBatchInputValidation() {
    for(int control=0;control<10;++control) {
        PrefixCachePageLedger ledger;
        LargeBatch batch;
        auto expected=batch.ids;
        if(control==0) std::reverse(expected.begin(),expected.end());
        if(control==1) expected.back()=expected.front();
        if(control==2 || control==9) batch.adoptions.back().identity=batch.ids.front();
        if(control==3) batch.adoptions.back().page=0;
        if(control==4) expected.back().pageHash="not-an-adoption";
        if(control==5) batch.adoptions.back().identity.model.clear();
        if(control==6) batch.adoptions.back().manager=nullptr;
        if(control==7) batch.adoptions.back().page=-1;
        const bool expectedSuccess=control==0 || control==8;
        const bool ok=ledger.CommitL1Batch(batch.adoptions,control>=8?nullptr:&expected);
        assert(ok==expectedSuccess);
        for(size_t i=0;i<batch.ids.size();++i) {
            PrefixCachePageSnapshot snap;
            assert(ledger.Snapshot(batch.ids[i],snap)==expectedSuccess);
            if(expectedSuccess) {
                assert(snap.identity==batch.ids[i] && snap.l1Manager==&batch.manager);
                assert(snap.runtimeTier==PrefixCacheRuntimeTier::L1 && snap.l1Page==(int64_t)i);
            }
        }
    }
}
void TestLargeStaleLeaseIsAtomic() {
    PrefixCachePageLedger ledger;
    LargeBatch batch(4096);
    for(size_t i=0;i<batch.ids.size();++i) {
        assert(ledger.ObserveL1(batch.ids[i],&batch.manager,(int64_t)i,1));
        const auto reclaim=ledger.BeginReclaim(batch.ids[i],i+1);
        assert(reclaim.Valid() && ledger.CommitL2(reclaim,i+1));
        batch.adoptions[i].lease=ledger.BeginPromotion(batch.ids[i]);
        assert(batch.adoptions[i].lease.Valid());
    }
    auto invalid=batch.adoptions;
    ++invalid.back().lease.serial;
    assert(!ledger.CommitL1Batch(invalid,&batch.ids));
    for(size_t i=0;i<batch.ids.size();++i) {
        PrefixCachePageSnapshot snap;
        assert(ledger.Snapshot(batch.ids[i],snap));
        assert(snap.runtimeTier==PrefixCacheRuntimeTier::MIGRATING);
        assert(snap.activeLeaseSerial==batch.adoptions[i].lease.serial);
        assert(snap.l2HostAllocation==i+1);
    }
    assert(ledger.CommitL1Batch(batch.adoptions,&batch.ids));
    for(size_t i=0;i<batch.ids.size();++i) {
        PrefixCachePageSnapshot snap;
        assert(ledger.Snapshot(batch.ids[i],snap));
        assert(snap.runtimeTier==PrefixCacheRuntimeTier::L1 && snap.activeLeaseSerial==0);
    }
}
void TestLargePublicationConcurrentQuery() {
    PrefixCachePageLedger ledger;
    LargeBatch batch;
    std::atomic<bool> ready{false},done{false},partial{false};
    std::thread reader([&] {
        ready.store(true);
        do {
            auto view=ledger.Query(batch.ids.front(),2);
            if(!view.empty() && view.size()!=2) partial.store(true);
        } while(!done.load());
    });
    while(!ready.load()) std::this_thread::yield();
    const bool ok=ledger.CommitL1Batch(batch.adoptions,&batch.ids);
    done.store(true);
    reader.join();
    assert(ok && !partial.load());
    assert(ledger.Query(batch.ids.front(),2).size()==2);
}


std::vector<PrefixCachePageIdentity> TrimPages(const char *stream, size_t count) {
    std::vector<PrefixCachePageIdentity> pages;
    pages.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto hash = std::string(stream) + "-" + std::to_string(i);
        pages.push_back(Page(stream, i, "", hash.c_str()));
    }
    return pages;
}

size_t CountRetained(PrefixCachePageLedger &ledger,
                     const std::vector<PrefixCachePageIdentity> &pages) {
    size_t count = 0;
    for (const auto &page : pages) if (ledger.Find(page)) ++count;
    return count;
}

void TestBoundedTrimProtectsOwnersAndRecovers() {
    PrefixCachePageLedger ledger(2); // Soft inactive target; live owners exceed it.
    int manager = 0;
    const auto protectedPages = TrimPages("trim-protected", 6);
    const auto &hot = protectedPages[0];
    assert(ledger.ObserveL1(hot, &manager, 0, 1));
    const auto durableGeneration =
        ledger.RegisterCheckpoint("trim-durable", "identity", {protectedPages[1]});
    assert(durableGeneration != 0);
    assert(ledger.FinishCheckpoint("trim-durable", "identity", durableGeneration, true));
    const auto pendingGeneration =
        ledger.RegisterCheckpoint("trim-pending", "identity", {protectedPages[2]});
    assert(pendingGeneration != 0);
    assert(ledger.ObserveL1(protectedPages[3], &manager, 3, 1));
    const auto lease = ledger.BeginReclaim(protectedPages[3], 1);
    assert(lease.Valid()); // Must retain the live migration lease.
    assert(ledger.ObserveL1(protectedPages[4], &manager, 4, 1));
    const auto failing = ledger.BeginReclaim(protectedPages[4], 2);
    assert(failing.Valid() && ledger.FailMigration(failing, "CPU expected failure"));
    assert(ledger.ObserveL1(protectedPages[5], &manager, 5, 1));
    const auto runtime = ledger.AcquireRuntime(protectedPages[5]);
    assert(runtime.Valid());

    for (int i = 0; i < 40; ++i) assert(ledger.ObserveL1(hot, &manager, 0, 1));
    assert(CountRetained(ledger, protectedPages) == protectedPages.size());
    assert(ledger.TrimInactive(1) == 0); // None of these owners is disposable.

    const auto failed = TrimPages("trim-failed", 320);
    const auto generation = ledger.RegisterCheckpoint("trim-failed-root", "identity", failed);
    assert(generation != 0);
    assert(ledger.FinishCheckpoint("trim-failed-root", "identity", generation, false));
    size_t remaining = failed.size();
    bool removedWhileStillOverLimit = false;
    for (int i = 0; i < 32 && remaining; ++i) {
        assert(ledger.ObserveL1(hot, &manager, 0, 1));
        const size_t next = CountRetained(ledger, failed);
        // A call which found disposable records must remain bounded even though
        // six protected records already exceed the soft target of two.
        assert(next <= remaining && remaining - next <= 64);
        if (next < remaining) removedWhileStillOverLimit = true;
        remaining = next;
    }
    assert(removedWhileStillOverLimit && remaining == 0);
    assert(CountRetained(ledger, protectedPages) == protectedPages.size());

    // Previously visited records become eligible later; no negative-result
    // cache or one-shot cursor may hide this change forever.
    assert(ledger.EraseCheckpoint("trim-durable", "identity"));
    for (int i = 0; i < 8; ++i) assert(ledger.ObserveL1(hot, &manager, 0, 1));
    assert(!ledger.Find(protectedPages[1]));
    assert(ledger.FinishCheckpoint("trim-pending", "identity", pendingGeneration, false));
    for (int i = 0; i < 8; ++i) assert(ledger.ObserveL1(hot, &manager, 0, 1));
    assert(!ledger.Find(protectedPages[2]));
    assert(ledger.Find(protectedPages[3])->activeLeaseSerial == lease.serial);
    assert(ledger.Find(protectedPages[4])->failureQuarantined);
    assert(ledger.Find(protectedPages[5])->runtimeRefCount == 1);
    assert(ledger.ReleaseRuntime(runtime));
}

void TestTrimCursorSurvivesGrowthAndFullErase() {
    PrefixCachePageLedger ledger(1);
    int manager = 0;
    const auto live = TrimPages("trim-growth", 2049);
    // Insertions across a long-lived over-limit sweep force normal unordered
    // map growth/rehash. Existing owners must never become stale iterators.
    for (size_t i = 0; i < live.size(); ++i)
        assert(ledger.ObserveL1(live[i], &manager, (int64_t)i, 1));
    assert(CountRetained(ledger, live) == live.size());
    // A batch may replace the map's shared_ptr while a reader keeps the old
    // immutable record. Sweep progress must follow current canonical ownership,
    // never a saved PageRecord pointer.
    const auto heldBeforeBatch = ledger.Find(live[0]);
    assert(heldBeforeBatch && heldBeforeBatch->l1Generation == 1);
    const std::vector<PrefixCachePageLedger::L1Adoption> swapped = {
        {live[0], {}, &manager, 0, 2, {}, {}}};
    assert(ledger.CommitL1Batch(swapped));
    assert(heldBeforeBatch->l1Generation == 1 && heldBeforeBatch->l1Page == 0);
    assert(ledger.Find(live[0])->l1Generation == 2 &&
           ledger.Find(live[0])->l1Page == 0);
    const auto failed = TrimPages("trim-growth-failed", 1);
    const auto generation = ledger.RegisterCheckpoint("trim-growth-root", "identity", failed);
    assert(generation != 0 &&
           ledger.FinishCheckpoint("trim-growth-root", "identity", generation, false));
    for (int i = 0; i < 128 && ledger.Find(failed[0]); ++i)
        assert(ledger.ObserveL1(live[0], &manager, 0, 1));
    assert(!ledger.Find(failed[0]));
    assert(CountRetained(ledger, live) == live.size());

    // Full trim erases any key which could have been a hot sweep bookmark.
    const auto batch = TrimPages("trim-full-erase", 1024);
    const auto batchGeneration = ledger.RegisterCheckpoint("trim-full-root", "identity", batch);
    assert(batchGeneration != 0 &&
           ledger.FinishCheckpoint("trim-full-root", "identity", batchGeneration, false));
    assert(ledger.ObserveL1(live[0], &manager, 0, 1));
    const auto beforeFull = CountRetained(ledger, batch);
    assert(ledger.TrimInactive(1) == beforeFull);
    assert(CountRetained(ledger, batch) == 0);
    assert(CountRetained(ledger, live) == live.size());
    for (int i = 0; i < 8; ++i) assert(ledger.ObserveL1(live[0], &manager, 0, 1));

    const auto after = TrimPages("trim-after-full", 1);
    const auto afterGeneration = ledger.RegisterCheckpoint("trim-after-root", "identity", after);
    assert(afterGeneration != 0 &&
           ledger.FinishCheckpoint("trim-after-root", "identity", afterGeneration, false));
    for (int i = 0; i < 128 && ledger.Find(after[0]); ++i)
        assert(ledger.ObserveL1(live[0], &manager, 0, 1));
    assert(!ledger.Find(after[0]));
}

} // namespace

int main() {
    TestReclaimAndPromotion();
    TestFailedMigrationIsQuarantined();
    TestDurableRootsAndFifo();
    TestLongestCandidateDirectRestore();
    TestConcurrentReadLease();
    TestAtomicMultiPoolAdoption();
    TestLargeBatchInputValidation();
    TestLargeStaleLeaseIsAtomic();
    TestLargePublicationConcurrentQuery();
    TestBoundedTrimProtectsOwnersAndRecovers();
    TestTrimCursorSurvivesGrowthAndFullErase();
    std::printf("PASS ledger semantics: original controls, large-batch atomicity, concurrent query, bounded trim ownership and recovery\n");
    return 0;
}
