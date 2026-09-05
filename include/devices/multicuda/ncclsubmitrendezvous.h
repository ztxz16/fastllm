#ifndef FASTLLM_NCCL_SUBMIT_RENDEZVOUS_H
#define FASTLLM_NCCL_SUBMIT_RENDEZVOUS_H

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

    bool Wait(int rank, Phase phase, int count, int dataType) {
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
            cv.notify_all();
            return true;
        }
        // A missing/failed rank must not leave its peers in a permanent CPU
        // wait. Allow long prefills; this timeout does not bound CUDA calls.
        if (!cv.wait_for(lock, timeout, [&] {
                return epoch != current || !error.empty();
            })) {
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
        cv.notify_all();
        return false;
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint64_t> nextPhase;
    const std::chrono::milliseconds timeout;
    uint64_t epoch = 0;
    int arrived = 0;
    int expectedCount = 0;
    int expectedDataType = 0;
    std::string error;
};

} // namespace fastllm
#endif
