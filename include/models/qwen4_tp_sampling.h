#pragma once

#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fastllm {
namespace qwen4_tp {

// TopK(..., 1) scans vocabulary rows with 256 lanes. Keeping shard starts
// aligned preserves that scan's lane assignment, including equal logits.
inline std::pair<int, int> VocabRange(int vocabulary, int ranks, int rank) {
    const int blocks = vocabulary / 256;
    const int first = (int)((long long)blocks * rank / ranks) * 256;
    const int end = rank + 1 == ranks ? vocabulary
        : (int)((long long)blocks * (rank + 1) / ranks) * 256;
    return {first, end};
}

inline int Top1LaneOrder(int token) {
    unsigned lane = token & 255;
    lane = ((lane & 0x55) << 1) | ((lane >> 1) & 0x55);
    lane = ((lane & 0x33) << 2) | ((lane >> 2) & 0x33);
    return ((lane & 0x0f) << 4) | (lane >> 4);
}

// FastllmLayerNormKernelTop1 keeps the left side of a tied reduction.
// Its halving tree visits lanes in bit-reversed order, then each lane keeps
// its first matching vocabulary row. This is not minimum-token-ID argmax.
inline bool Top1Before(int id, float score, int bestId, float bestScore) {
    if (score != bestScore) return score > bestScore;
    const int order = Top1LaneOrder(id), bestOrder = Top1LaneOrder(bestId);
    return order < bestOrder || (order == bestOrder && id < bestId);
}

// Each shard supplies its maximum logit and that token's local softmax
// probability. Recover the global denominator without gathering the vocabulary.
inline float MergeTopProbability(const std::vector<std::pair<float, float>> &shards) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto &shard : shards) {
        if (std::isnan(shard.first) || shard.first == std::numeric_limits<float>::infinity()) return 0;
        maximum = std::max(maximum, (double)shard.first);
    }
    if (!std::isfinite(maximum)) return 0;
    double denominator = 0;
    for (const auto &shard : shards) {
        if (shard.first == -std::numeric_limits<float>::infinity()) continue;
        if (!(shard.second > 0 && shard.second <= 1)) return 0;
        denominator += std::exp((double)shard.first - maximum) / shard.second;
    }
    return std::isfinite(denominator) && denominator >= 1 ? (float)(1 / denominator) : 0;
}

} // namespace qwen4_tp
} // namespace fastllm
