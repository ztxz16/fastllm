#pragma once
#include <cstdint>

namespace fastllm { namespace qwen35_legacy {
// The production legacy forward calls this at the real target/checkpoint boundary.
// False leaves cursor unchanged: a duplicate, stale generation or oversize step
// must never be published as another completed input slice.
inline bool CommitOneChunk(uint64_t ownerGeneration, uint64_t currentGeneration,
                           int total, int expectedBegin, int selected,
                           int completedEnd, int &cursor) {
    if (!ownerGeneration || ownerGeneration != currentGeneration || total <= 0 ||
        expectedBegin < 0 || cursor != expectedBegin || selected <= 0 ||
        completedEnd <= expectedBegin || completedEnd > total ||
        completedEnd - expectedBegin > selected) return false;
    cursor = completedEnd;
    return true;
}
inline int StepDelta(int begin, int end) { return end > begin ? end - begin : 0; }
} }
