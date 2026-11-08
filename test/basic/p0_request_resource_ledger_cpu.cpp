#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <type_traits>
#ifndef LEDGER_TEST_HEADER
#define LEDGER_TEST_HEADER "utils/request_resource_ledger.h"
#endif
#include LEDGER_TEST_HEADER

static std::atomic<bool> failNextAllocation{false};
void *operator new(std::size_t size) {
    if (failNextAllocation.exchange(false)) throw std::bad_alloc();
    if (void *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

using fastllm::RequestResourceHold;
using fastllm::RequestResourceLedger;
static int failures = 0;
static void expect(bool ok, const char *name) {
    std::printf("{\"case\":\"%s\",\"pass\":%s}\n", name, ok ? "true" : "false");
    failures += !ok;
}
static bool balances(const fastllm::RequestResourceSnapshot &s, size_t owners,
                     size_t readers, size_t pending) {
    std::printf("{\"kind\":\"snapshot\",\"actual\":{\"transient_owners\":%zu,\"temporary_readers\":%zu,\"temporary_leases\":%zu,\"pending_operations\":%zu,\"consistency_error\":%s,\"resident_observed\":%s,\"resident_owners\":%zu,\"resident_generations\":%zu,\"live_generations\":[",
                s.transientOwners, s.temporaryReaders, s.temporaryLeases, s.pendingOperations,
                s.consistencyError ? "true" : "false", s.residentObserved ? "true" : "false",
                s.residentOwners, s.residentGenerations);
    bool first = true;
    for (auto generation : s.liveGenerations) {
        std::printf("%s%llu", first ? "" : ",", (unsigned long long)generation); first = false;
    }
    std::printf("]},\"expected\":{\"transient_owners\":%zu,\"temporary_readers\":%zu,\"temporary_leases\":%zu,\"pending_operations\":%zu}}\n",
                owners, readers, readers, pending);
    return s.transientOwners == owners && s.temporaryReaders == readers &&
           s.temporaryLeases == readers && s.pendingOperations == pending;
}
int main() {
    static_assert(!std::is_copy_constructible<RequestResourceHold>::value, "holder cannot copy");
    static_assert(std::is_nothrow_move_constructible<RequestResourceHold>::value, "move cannot throw");
    {
        failNextAllocation = true;
        auto off = RequestResourceHold::Reader({}, 42);
        auto offPending = RequestResourceHold::Pending({}, 42);
        bool noAllocation = failNextAllocation.exchange(false);
        expect(!off && !offPending && off.TokenId() == 0 && noAllocation, "default_off_inert_no_allocation");
    }
    {
        auto l = std::make_shared<RequestResourceLedger>();
        auto r1 = RequestResourceHold::Reader(l, 7);
        auto r2 = RequestResourceHold::Reader(l, 7);
        auto p = RequestResourceHold::Pending(l, 7);
        std::printf("{\"kind\":\"holder_tokens\",\"generation\":7,\"readers\":[%llu,%llu],\"pending\":%llu}\n",
                    (unsigned long long)r1.TokenId(), (unsigned long long)r2.TokenId(),
                    (unsigned long long)p.TokenId());
        auto s = l->Snapshot();
        expect(r1.TokenId() != r2.TokenId() && r1.TokenId() != p.TokenId(), "same_generation_unique_tokens");
        expect(balances(s, 4, 2, 1) && s.liveGenerations == std::set<uint64_t>{7}, "same_generation_intermediate_all_held");
        r1.Reset(); r1.Reset();
        s = l->Snapshot();
        expect(balances(s, 3, 1, 1) && !s.consistencyError && s.liveGenerations.count(7), "reset_idempotent_other_holders_still_live");
        l->ReleaseContextOwner();
        s = l->Snapshot();
        expect(balances(s, 2, 1, 1) && s.liveGenerations.count(7), "context_release_does_not_clear_real_holders");
        auto moved = std::move(r2);
        expect(!r2 && moved, "move_transfers_single_holder");
        p.Reset();
        s = l->Snapshot();
        expect(balances(s, 1, 1, 0) && s.liveGenerations.count(7), "pending_release_retains_reader");
        moved.Reset();
        s = l->Snapshot();
        expect(balances(s, 0, 0, 0) && s.liveGenerations.empty() && !s.consistencyError, "terminal_after_last_real_holder");
        expect(!s.residentObserved, "absent_resident_sample_is_unknown");
    }
    {
        auto l = std::make_shared<RequestResourceLedger>();
        auto r = l->AcquireReaderToken(9);
        auto p = l->BeginPendingToken(9);
        l->ReleaseToken(r);
        l->ReleaseToken(r);
        auto s = l->Snapshot();
        expect(balances(s, 2, 0, 1) && s.consistencyError && s.liveGenerations.count(9), "duplicate_release_does_not_consume_other_role");
        l->ReleaseToken(0);
        s = l->Snapshot();
        expect(balances(s, 2, 0, 1) && s.consistencyError, "unknown_token_preserves_balances");
        l->ReleaseToken(p); l->ReleaseContextOwner(); l->ReleaseContextOwner();
        s = l->Snapshot();
        expect(balances(s, 0, 0, 0) && s.consistencyError, "context_double_release_sticky_no_underflow");
    }
    {
        auto l = std::make_shared<RequestResourceLedger>();
        try {
            auto r = RequestResourceHold::Reader(l, 11);
            auto p = RequestResourceHold::Pending(l, 12);
            auto s = l->Snapshot();
            expect(balances(s, 3, 1, 1) && s.liveGenerations.size() == 2, "exception_path_live_before_throw");
            throw std::runtime_error("injected holder scope failure");
        } catch (const std::runtime_error &) {}
        auto s = l->Snapshot();
        expect(balances(s, 1, 0, 0) && s.liveGenerations.empty() && !s.consistencyError, "exception_unwind_releases_only_owned_tokens");
        l->ReleaseContextOwner();
    }
    {
        auto l = std::make_shared<RequestResourceLedger>();
        auto r = RequestResourceHold::Reader(l, 15);
        bool caught = false;
        failNextAllocation = true;
        try { auto p = RequestResourceHold::Pending(l, 16); }
        catch (const std::bad_alloc &) { caught = true; }
        failNextAllocation = false;
        auto s = l->Snapshot();
        expect(caught && balances(s, 2, 1, 0) && s.liveGenerations == std::set<uint64_t>{15} && s.consistencyError,
               "allocation_failure_preserves_preexisting_holder");
        r.Reset(); l->ReleaseContextOwner();
    }
    {
        auto l = std::make_shared<RequestResourceLedger>();
        auto a = RequestResourceHold::Reader(l, 21);
        auto b = RequestResourceHold::Pending(l, 22);
        a = std::move(b);
        auto s = l->Snapshot();
        expect(balances(s, 2, 0, 1) && s.liveGenerations == std::set<uint64_t>{22} && !b,
               "move_assignment_releases_previous_token");
        a = std::move(a);
        expect(balances(l->Snapshot(), 2, 0, 1), "self_move_preserves_holder");
        l->ObserveResident(3, 2);
        s = l->Snapshot();
        expect(s.residentObserved && s.residentOwners == 3 && s.residentGenerations == 2,
               "explicit_resident_sample_kept_separate");
        a.Reset(); l->ReleaseContextOwner();
    }
    std::printf("{\"result\":\"%s\",\"failures\":%d}\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
