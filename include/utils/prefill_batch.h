#pragma once

#include <chrono>
#include <algorithm>
#include <map>
#include <vector>

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
// All targets retain their per-replica units. DP mode divides them across
// owners; scaling changes release thresholds, never allocation budgets.
struct PrefillSchedulingOptions {
    bool balanceDP = false, chunked = false;
    int adaptiveRequests = 0, chunkWaitMs = 0, fairnessMs = 0;
    PrefillBatchPolicy Effective(PrefillBatchPolicy policy, int active) const {
        if (adaptiveRequests <= 0) return policy;
        const long long reference = adaptiveRequests;
        const long long load = std::max((reference + 7) / 8,
                                       std::min<long long>(active, reference));
        policy.targetTokens = (policy.targetTokens * load + reference - 1) / reference;
        policy.targetRequests = (policy.targetRequests * load + reference - 1) / reference;
        policy.maxWaitMs = std::max(std::min(5000, policy.maxWaitMs),
                                   (int)(policy.maxWaitMs * load / reference));
        return policy;
    }
};

struct PrefillGroupLoad {
    long long tokens = 0;
    int requests = 0, decoding = 0;
    bool capacityLimited = false;
};

inline const char *DecidePrefillGroups(const PrefillBatchPolicy &policy,
        const std::vector<PrefillGroupLoad> &groups, bool batchFull,
        PrefillBatchPolicy::Clock::time_point oldest,
        PrefillBatchPolicy::Clock::time_point now) {
    if (policy.maxWaitMs <= 0) return "disabled";
    int requests = 0, decoding = 0;
    for (const auto &g : groups) { requests += g.requests; decoding += g.decoding; }
    if (!decoding) return "idle";
    if (!requests) return "no_candidates";
    if (now >= policy.Deadline(oldest)) return "timeout";
    const int n = std::max(1, (int)groups.size());
    const long long tokens = (policy.targetTokens + (long long)n - 1) / n;
    const int target = (policy.targetRequests + (long long)n - 1) / n;
    const int minimum = (policy.minDecodeRequests + (long long)n - 1) / n;
    bool ready = true;
    for (const auto &g : groups) {
        if (g.decoding < minimum) return "low_decode";
        ready &= g.capacityLimited || (g.requests > 0 && g.tokens >= tokens && g.requests >= target);
    }
    if (ready) return "dp_targets";
    // No additional request can be admitted until this wave runs.
    if (batchFull) return "capacity";
    return "wait";
}

// Measured forward time, not a count of iterations. The last indivisible
// forward may exceed the slice. Idle work cannot repay busy decode's budget.
struct PrefillDecodeBudget {
    double prefillMs = 0, decodeDueMs = 0;
    void Update(bool prefill, double elapsedMs, bool busy, int sliceMs) {
        if (!busy || sliceMs <= 0) { prefillMs = decodeDueMs = 0; return; }
        if (prefill) {
            prefillMs += elapsedMs;
            if (prefillMs >= sliceMs) { decodeDueMs = sliceMs; prefillMs = 0; }
        } else {
            prefillMs = 0;
            decodeDueMs = std::max(0.0, decodeDueMs - elapsedMs);
        }
    }
};

// FIFO within each attention DP group; prefer the least loaded group in a
// prefill batch. Ties use original admission order, so each new batch starts
// with the oldest request. Only admitted, uncached tokens count as work.
// The default single group is exactly FIFO. No missing group delays admission.
class PrefillBatchQueue {
    struct Group {
        std::vector<int> requests;
        size_t next = 0;
        long long tokens = 0;
    };
    std::map<int, Group> groups;
    int selected = 0;
public:
    void Push(int request, int group) { groups[group].requests.push_back(request); }
    int Pop() {
        int request = -1;
        for (auto &entry : groups) {
            const auto &group = entry.second;
            if (group.next == group.requests.size()) continue;
            const int next = group.requests[group.next];
            if (request < 0 || group.tokens < groups.at(selected).tokens ||
                (group.tokens == groups.at(selected).tokens && next < request)) {
                request = next;
                selected = entry.first;
            }
        }
        if (request >= 0) ++groups.at(selected).next;
        return request;
    }
    void AddTokens(int count) { groups.at(selected).tokens += count; }
};

}
