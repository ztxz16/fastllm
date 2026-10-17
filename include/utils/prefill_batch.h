#pragma once

#include <chrono>

namespace fastllm {
// Admission policy only: cache ownership and request selection stay in the
// scheduler. Use the oldest pending request's arrival time, never a sliding
// deadline, so a stream of newcomers cannot postpone prefill indefinitely.
struct PrefillBatchPolicy {
    using Clock = std::chrono::steady_clock;
    int targetTokens = 0;
    int maxWaitMs = 0;

    Clock::time_point Deadline(Clock::time_point oldest) const {
        return oldest + std::chrono::milliseconds(maxWaitMs);
    }
    bool ShouldWait(long long tokens, int requests, int availableSlots,
                    Clock::time_point oldest, Clock::time_point now) const {
        return targetTokens > 0 && maxWaitMs > 0 && requests > 0 &&
               requests < availableSlots && tokens < targetTokens && now < Deadline(oldest);
    }
};
}
