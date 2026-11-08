#include "fastllm.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void FillTrie(PagedCacheManager &manager, int pages,
                     const std::vector<std::string> &extra) {
    std::vector<int> ids;
    for (int i = 0; i < pages; ++i) {
        const int page = manager.GetUnusedPageIndex(true);
        Require(page >= 0, "initial page allocation failed");
        ids.push_back(page);
    }
    std::vector<int> tokens(pages * manager.pageLen);
    for (int i = 0; i < (int)tokens.size(); ++i) tokens[i] = i + 1;
    manager.Record(tokens, ids, &extra);
    manager.ReleasePageIndices(ids);
}

int main() {
    // A batched preflight must dispatch more than one migration while the
    // producer tickets remain outstanding.  No pending victim may be reused.
    PagedCacheManager manager;
    manager.type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
    manager.pageLen = 2;
    manager.SetMaxPages(5);
    std::vector<std::shared_ptr<PagedCacheManager::RecycleTicket>> pending;
    manager.onRecyclePage = [&](int page, uint64_t hash,
        const std::shared_ptr<PagedCacheManager::RecycleTicket> &ticket) {
        Require(page == ticket->pageIndex && hash == ticket->pageHash,
                "recycle callback metadata mismatch");
        pending.push_back(ticket); // simulated producer promise: not completed yet
    };
    manager.EnableDeferredRecycle();
    const auto producer = std::make_shared<int>(17);
    manager.recordProducerEvent = producer;
    FillTrie(manager, 4, {"rank0", "rank0", "rank0", "rank0"});
    // A later scheduling event must never replace a recorded victim's event.
    manager.recordProducerEvent = std::make_shared<int>(99);
    Require(manager.ReadyFreePageCount() == 0,
            "deferred ready count included trie pages or reserve");

    // The replacement reserve is not an ordinary free page.  When every
    // other page is pinned, allocation must fail without consuming the
    // reserve; the next real victim still needs that page as its destination.
    PagedCacheManager protectedReserve;
    protectedReserve.type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
    protectedReserve.pageLen = 2;
    protectedReserve.SetMaxPages(2);
    protectedReserve.EnableDeferredRecycle();
    auto held0 = protectedReserve.GetUnusedPageIndex(true);
    Require(held0 >= 0, "protected-reserve initial allocation failed");
    Require(protectedReserve.ReadyFreePageCount() == 0,
            "reserve-only pool reported a ready ordinary page");
    Require(!protectedReserve.PrepareFreePages(1),
            "reserve-only pool claimed a page without an eligible victim");
    bool protectedFailure = false;
    try {
        (void)protectedReserve.GetUnusedPageIndex(true);
    } catch (const std::runtime_error &) {
        protectedFailure = true;
    }
    Require(protectedFailure && protectedReserve.recycleReservePage >= 0,
            "deferred replacement reserve was consumed as an ordinary page");
    protectedReserve.ReleasePageIndex(held0);

    Require(!manager.PrepareFreePages(3),
            "preflight claimed pages before migration ownership completed");
    Require(pending.size() == 3, "batch preflight did not issue three tickets");
    const int firstPending = pending[0]->pageIndex;
    const int secondPending = pending[1]->pageIndex;
    Require(firstPending != secondPending, "two recycle tickets share a victim page");
    Require(pending[0]->generation > 0 && !pending[0]->extraKey.empty(),
            "ticket did not retain immutable node identity");
    Require(pending[0]->producerEvent == producer && pending[1]->producerEvent == producer,
            "recycle ticket used a later scheduling event");
    const int replacement = manager.GetUnusedPageIndex(true);
    Require(replacement >= 0 && replacement != firstPending && replacement != secondPending,
            "allocator reused an outstanding migration victim");
    Require(pending.size() == 4,
            "allocator did not issue a ticket for the final trie victim");
    bool capacityThrown = false;
    std::string capacityReason;
    try {
        (void)manager.GetUnusedPageIndex(true);
    } catch (const std::runtime_error &error) {
        capacityThrown = true;
        capacityReason = error.what();
    }
    Require(capacityThrown && capacityReason.find("capacity_unavailable") != std::string::npos,
            "deferred shortage did not report capacity_unavailable");
    std::cout << "[capacity] reason=" << capacityReason << "\n";
    for (const auto &ticket : pending) manager.CompleteRecyclePage(ticket, true);
    Require(pending[0]->terminal && pending[0]->success &&
            pending[1]->terminal && pending[1]->success,
            "successful recycle was not terminal");
    Require(manager.recycleReservePage >= 0 && manager.ReadyFreePageCount() == 3,
            "successful migrations did not restore reserve and ordinary pages");
    manager.CompleteRecyclePage(pending[0], true); // idempotent duplicate
    Require(manager.GetUnusedPageIndex(true) >= 0,
            "ready migrated page was not allocatable");

    // Tensor-parallel ranks own independent reserves.  A ticket on one rank
    // must never consume or satisfy the other rank's replacement page.
    PagedCacheManager rankA, rankB;
    std::vector<std::shared_ptr<PagedCacheManager::RecycleTicket>> rankTicketsA, rankTicketsB;
    auto prepareRank = [&](PagedCacheManager &rank, int tag,
                           std::vector<std::shared_ptr<PagedCacheManager::RecycleTicket>> &tickets) {
        rank.type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        rank.pageLen = 2;
        rank.SetMaxPages(3);
        rank.onRecyclePage = [&tickets](int, uint64_t,
            const std::shared_ptr<PagedCacheManager::RecycleTicket> &value) {
            tickets.push_back(value);
        };
        rank.EnableDeferredRecycle();
        FillTrie(rank, 2, {"rank" + std::to_string(tag), "rank" + std::to_string(tag)});
        Require(rank.ReadyFreePageCount() == 0,
                "rank preflight started with a reserve counted as ordinary free");
        Require(!rank.PrepareFreePages(1) && tickets.size() == 1,
                "rank preflight did not retain its first outstanding ticket");
        (void)rank.GetUnusedPageIndex(true); // dispatch the second victim and consume this rank's reserve
        Require(tickets.size() == 2,
                "rank allocator did not issue its second recycle ticket");
    };
    prepareRank(rankA, 0, rankTicketsA);
    prepareRank(rankB, 1, rankTicketsB);
    Require(rankA.recycleReservePage < 0 && rankB.recycleReservePage < 0 &&
                rankTicketsA[0]->pageIndex != rankTicketsA[1]->pageIndex &&
                rankTicketsB[0]->pageIndex != rankTicketsB[1]->pageIndex,
            "rank replacement allocation did not retain independent victims");
    for (const auto &ticket : rankTicketsA) rankA.CompleteRecyclePage(ticket, true);
    for (const auto &ticket : rankTicketsB) rankB.CompleteRecyclePage(ticket, true);
    Require(rankA.recycleReservePage >= 0 && rankB.recycleReservePage >= 0 &&
                rankA.ReadyFreePageCount() == 1 && rankB.ReadyFreePageCount() == 1,
            "rank recycle completion crossed reserve or free-page ownership");

    // A failed D2H must quarantine and retain the victim lease.  A duplicate
    // completion cannot turn the failed page into a reusable page.
    PagedCacheManager failed;
    failed.type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
    failed.pageLen = 2;
    failed.SetMaxPages(3);
    std::vector<std::shared_ptr<PagedCacheManager::RecycleTicket>> failedTickets;
    failed.onRecyclePage = [&](int page, uint64_t hash,
        const std::shared_ptr<PagedCacheManager::RecycleTicket> &ticket) {
        Require(page == ticket->pageIndex && hash == ticket->pageHash,
                "failed callback metadata mismatch");
        failedTickets.push_back(ticket);
    };
    failed.EnableDeferredRecycle();
    FillTrie(failed, 2, {"rank1", "rank1"});
    Require(!failed.PrepareFreePages(1) && failedTickets.size() == 1,
            "failed preflight did not issue its first ticket");
    (void)failed.GetUnusedPageIndex(true); // dispatch the second ticket and consume the reserve
    Require(failedTickets.size() == 2,
            "failed allocator did not issue its second recycle ticket");
    const int quarantined = failedTickets[0]->pageIndex;
    failed.CompleteRecyclePage(failedTickets[0], false, "producer_not_ready");
    failed.CompleteRecyclePage(failedTickets[1], true);
    Require(failedTickets[0]->terminal && !failedTickets[0]->success &&
            failedTickets[0]->failureReason == "producer_not_ready",
            "failed recycle terminal state was not retained");
    Require(failed.quarantinedRecyclePages.count(quarantined) != 0 &&
            failed.pageRefCount[quarantined] == 1,
            "failed victim was not quarantined with its lease");
    Require(failed.recycleReservePage >= 0 && failed.ReadyFreePageCount() == 0,
            "failed recycle did not preserve reserve or ordinary capacity");
    failed.CompleteRecyclePage(failedTickets[0], true);
    Require(failed.quarantinedRecyclePages.count(quarantined) != 0 &&
            failed.pageRefCount[quarantined] == 1,
            "duplicate completion resurrected a failed victim");

    std::cout << "Paged deferred recycle protocol: PASS\n";
    return 0;
}
