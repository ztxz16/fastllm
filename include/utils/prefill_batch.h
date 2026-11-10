#pragma once

#include <chrono>
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
