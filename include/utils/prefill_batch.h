#pragma once

#include <chrono>

namespace fastllm {
// Per-replica admission policy. Resource accounting and cache ownership stay
// in the scheduler; deadlines belong to requests, never to a sliding batch.
struct PrefillBatchPolicy {
    using Clock = std::chrono::steady_clock;
    int targetTokens = 0, maxWaitMs = 0;
    int targetRequests = 0, maxRequests = 0, minDecodeRequests = 0;

    Clock::time_point Deadline(Clock::time_point oldest) const {
        return oldest + std::chrono::milliseconds(maxWaitMs);
    }
    const char *Decide(long long tokens, int requests, int activeDecode,
                       bool capacityLimited,
                       Clock::time_point oldest, Clock::time_point now) const {
        if (targetTokens <= 0 || maxWaitMs <= 0) return "disabled";
        if (activeDecode == 0) return "idle";
        if (activeDecode < minDecodeRequests) return "low_decode";
        if (requests == 0) return "no_candidates";
        if (now >= Deadline(oldest)) return "timeout";
        if (tokens >= targetTokens) return "tokens";
        if (targetRequests > 0 && requests >= targetRequests) return "requests";
        if (capacityLimited) return "capacity";
        return "wait";
    }
};
}
