#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fastllm { namespace qwen35_packed {
// All non-final slices end on the same 64-token grid. The final tail is
// emitted once; leave 65 tokens rather than moving a grid boundary by one.
// available is the unconsumed suffix after any prefix restore, so no persistent
// activation carry or cache-format change is needed.
inline int CanonicalPrefillTokenCount(int available, int limit) {
    if (available <= 0 || limit <= 0) return 0;
    if (available <= limit) return available;
    int count = (limit / 64) * 64;
    if (available - count == 1) count -= 64;
    return std::max(0, count);
}

inline std::vector<std::vector<int>> PlanPrefillChunks(
        const std::vector<int> &lengths, int cap) {
    if (lengths.empty() || cap < 2)
        throw std::invalid_argument("packed prefill requires positive rows and cap >= 2");
    int rounds = 1;
    for (int length : lengths) {
        if (length < 2)
            throw std::invalid_argument("packed prefill requires row length >= 2");
        rounds = std::max(rounds, 1 + (length - 1) / cap);
    }
    for (int length : lengths) {
        if (length / rounds < 2)
            throw std::invalid_argument("packed prefill ragged row cannot remain a prefill");
    }
    std::vector<std::vector<int>> plan(rounds, std::vector<int>(lengths.size()));
    std::vector<int> remaining = lengths;
    for (int round = 0; round < rounds; ++round) {
        const int remainingRounds = rounds - round;
        for (size_t row = 0; row < lengths.size(); ++row) {
            const int count = 1 + (remaining[row] - 1) / remainingRounds;
            if (count < 2 || count > cap)
                throw std::invalid_argument("packed prefill chunk outside admitted range");
            plan[round][row] = count;
            remaining[row] -= count;
        }
    }
    return plan;
}
}} // namespace fastllm::qwen35_packed
