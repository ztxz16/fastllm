#include "devices/multicuda/ncclsubmitrendezvous.h"

#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <thread>

using fastllm::NcclSubmitRendezvous;
using namespace std::chrono_literals;

static void Require(bool ok, const char *message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

static void RunRanks(int ranks, const std::function<void(int)> &fn) {
    std::vector<std::thread> workers;
    for (int rank = 0; rank < ranks; ++rank) {
        workers.emplace_back(fn, rank);
    }
    for (auto &worker : workers) worker.join();
}

static void CheckOrdering(int ranks) {
    constexpr int rounds = 128;
    NcclSubmitRendezvous group(ranks, 5s);
    std::vector<std::atomic<int>> prepared(rounds), submitted(rounds);
    for (int i = 0; i < rounds; ++i) {
        prepared[i] = 0;
        submitted[i] = 0;
    }
    RunRanks(ranks, [&](int rank) {
        for (int i = 0; i < rounds; ++i) {
            // Rotate the slow rank on each side of the collective. No rank
            // may enter NCCL before all previous host work has finished, or
            // enter the next GEMM before all NCCL host calls have returned.
            if (rank == i % ranks) std::this_thread::sleep_for(50us);
            ++prepared[i];
            Require(group.Wait(rank, NcclSubmitRendezvous::Before,
                               100 + i, i % 3), "before boundary failed");
            Require(prepared[i] == ranks, "entered submission before peers were ready");
            if (rank == (i + 1) % ranks) std::this_thread::sleep_for(50us);
            ++submitted[i];
            Require(group.Wait(rank, NcclSubmitRendezvous::After,
                               100 + i, i % 3), "after boundary failed");
            Require(submitted[i] == ranks, "next GEMM overtook peer submission");
        }
    });
    Require(group.Error().empty(), "successful group became broken");
}

static void CheckFailures() {
    // A missing rank releases all CPU waiters and leaves this group failed.
    NcclSubmitRendezvous missing(3, 40ms);
    auto start = std::chrono::steady_clock::now();
    RunRanks(2, [&](int rank) {
        Require(!missing.Wait(rank, NcclSubmitRendezvous::Before, 4, 0),
                "missing rank was accepted");
    });
    Require(std::chrono::steady_clock::now() - start < 2s, "timeout did not wake peers");
    Require(!missing.Wait(2, NcclSubmitRendezvous::Before, 4, 0), "failed group was reused");

    for (int mode = 0; mode < 4; ++mode) {
        NcclSubmitRendezvous mismatch(3, 5s);
        RunRanks(3, [&](int rank) {
            const int actualRank = mode == 0 ? 0 : rank;
            const int count = mode == 1 ? rank + 1 : 4;
            const int type = mode == 2 ? rank : 0;
            const auto phase = mode == 3 && rank == 1
                ? NcclSubmitRendezvous::After : NcclSubmitRendezvous::Before;
            Require(!mismatch.Wait(actualRank, phase, count, type),
                    "duplicate rank / metadata / phase mismatch was accepted");
        });
        Require(!mismatch.Error().empty(), "failure reason missing");
    }

    NcclSubmitRendezvous failedSubmit(3, 5s);
    std::atomic<int> ready{0};
    RunRanks(3, [&](int rank) {
        // Rank 1 waits for every Before call to return before injecting the
        // error, just as an actual NCCL submission failure occurs afterwards.
        Require(failedSubmit.Wait(rank, NcclSubmitRendezvous::Before, 4, 0),
                "setup boundary failed");
        ++ready;
        while (ready != 3) std::this_thread::yield();
        if (rank == 1) {
            failedSubmit.Abort("injected NCCL submission failure");
        } else {
            Require(!failedSubmit.Wait(rank, NcclSubmitRendezvous::After, 4, 0),
                    "submission failure did not wake peers");
        }
    });
    Require(failedSubmit.Error() == "injected NCCL submission failure", "lost failure reason");

    NcclSubmitRendezvous invalid(3, 5s);
    Require(!invalid.Wait(3, NcclSubmitRendezvous::Before, 4, 0), "invalid rank accepted");
    // A new communicator generation owns fresh state, independent of all
    // failed groups above. Two active domains must not share a barrier.
    std::thread a([] { CheckOrdering(3); });
    std::thread b([] { CheckOrdering(5); });
    a.join(); b.join();
}

int main() {
    for (int ranks : {3, 5, 7}) CheckOrdering(ranks);
    CheckFailures();
    std::cout << "PASS: odd-rank host ordering, independent groups, failure wakeup\n";
}
