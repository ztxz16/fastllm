#pragma once

#include <utility>

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

} // namespace qwen4_tp
} // namespace fastllm
