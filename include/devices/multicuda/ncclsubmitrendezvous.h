#ifndef FASTLLM_NCCL_SUBMIT_RENDEZVOUS_H
#define FASTLLM_NCCL_SUBMIT_RENDEZVOUS_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace fastllm {

// One instance per NCCL communicator group, with one host caller per rank.
// These are host submission boundaries, not GPU completion barriers. Capture
// must bypass BOTH boundaries; replay does not execute this host code at all.
class NcclSubmitRendezvous {
public:
    enum Phase { Before = 0, After = 1 };

    explicit NcclSubmitRendezvous(int ranks,
            std::chrono::milliseconds timeout = std::chrono::minutes(5))
        : nextPhase(ranks, 0), timeout(timeout) {}

    bool Wait(int rank, Phase phase, int count, int dataType,
              std::chrono::microseconds spinBudget = std::chrono::microseconds::zero()) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!error.empty()) {
            return false;
        }
        if (rank < 0 || rank >= (int)nextPhase.size() ||
            nextPhase[rank] != epoch || epoch % 2 != (unsigned)phase) {
            return FailLocked("inconsistent AllReduce rank or submission phase");
        }
        if (arrived == 0 && phase == Before) {
            expectedCount = count;
            expectedDataType = dataType;
        }
        if (count != expectedCount || dataType != expectedDataType) {
            return FailLocked("inconsistent AllReduce count or data type");
        }
        const uint64_t current = epoch;
        ++nextPhase[rank];
        if (++arrived == (int)nextPhase.size()) {
            arrived = 0;
            ++epoch;
            completedEpoch.store(epoch, std::memory_order_release);
            cv.notify_all();
            return true;
        }
        // Spin without holding the rank metadata lock. Spare-core callers can
        // cover short CPU MoE work without a scheduler wakeup; other callers
        // retain the fixed short spin. Both fall back to the timed CV wait.
        using Clock = std::chrono::steady_clock;
        const bool timedSpin = spinBudget.count() > 0;
        const auto start = timedSpin ? Clock::now() : Clock::time_point{};
        const auto deadline = start + timeout;
        const auto spinDeadline = std::min(start + spinBudget, deadline);
        lock.unlock();
        for (unsigned spin = 0;; ++spin) {
            if (completedEpoch.load(std::memory_order_acquire) != current ||
                aborted.load(std::memory_order_acquire)) {
                return !aborted.load(std::memory_order_acquire);
            }
            if (timedSpin) {
                if ((spin & 63) == 0 && Clock::now() >= spinDeadline) break;
            } else if (spin == 2048) {
                break;
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield");
#endif
        }
        lock.lock();
        // A missing/failed rank must not leave its peers in a permanent CPU
        // wait. Allow long prefills; this timeout does not bound CUDA calls.
        const auto finished = [&] { return epoch != current || !error.empty(); };
        if (!(timedSpin ? cv.wait_until(lock, deadline, finished)
                       : cv.wait_for(lock, timeout, finished))) {
            return FailLocked("AllReduce host submission rendezvous timed out");
        }
        return error.empty();
    }

    void Abort(const char *reason) {
        std::lock_guard<std::mutex> lock(mutex);
        FailLocked(reason);
    }

    std::string Error() {
        std::lock_guard<std::mutex> lock(mutex);
        return error;
    }

private:
    bool FailLocked(const char *reason) {
        if (error.empty()) {
            error = reason;
        }
        aborted.store(true, std::memory_order_release);
        cv.notify_all();
        return false;
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint64_t> nextPhase;
    const std::chrono::milliseconds timeout;
    std::atomic<uint64_t> completedEpoch{0};
    std::atomic<bool> aborted{false};
    uint64_t epoch = 0;
    int arrived = 0;
    int expectedCount = 0;
    int expectedDataType = 0;
    std::string error;
};

} // namespace fastllm
#endif
