#pragma once
#include <cstdint>

namespace fastllm {
// Observation only. All timestamps use the same steady clock. No model state,
// scheduling decisions, GPU synchronization or device memory is touched here.
struct NativeGenerationMetrics {
    uint64_t createdNs = 0, firstCommitNs = 0;
    uint64_t generationStartNs = 0, generationEndNs = 0, lastReportNs = 0;
    uint64_t generatedTokens = 0, prefillGeneratedTokens = 0, decodeTokens = 0;
    uint64_t prefillTokens = 0, prefillElapsedNs = 0, visionElapsedNs = 0;
    uint64_t excludedVisionNs = 0;
    uint64_t dflashSteps = 0, dflashAcceptedTotal = 0;
    bool prefillClockValid = true, generationClockValid = true;
    bool prefillReported = false, finishedReported = false, firstTokenReported = false;

    void DFlashStep(uint64_t acceptedTokens) {
        ++dflashSteps;
        dflashAcceptedTotal += acceptedTokens;
    }
    double DFlashAcceptMean() const {
        return dflashSteps > 0 ? double(dflashAcceptedTotal) / double(dflashSteps) : 0.0;
    }

    void Commit(bool prompt, uint64_t firstTimestamp = 0) {
        ++generatedTokens;
        if (prompt) ++prefillGeneratedTokens;
        else ++decodeTokens;
        if (firstCommitNs == 0 && firstTimestamp > 0) firstCommitNs = firstTimestamp;
    }
    void Prefill(uint64_t tokens, uint64_t begin, uint64_t end, uint64_t vision) {
        prefillTokens += tokens;
        if (end <= begin || vision > end - begin) prefillClockValid = false;
        else { prefillElapsedNs += end - begin - vision; excludedVisionNs += vision; }
        if (generationStartNs != 0) generationClockValid = false;
    }
    void Ready(uint64_t end) {
        if (generationStartNs == 0 && end > 0) {
            generationStartNs = generationEndNs = lastReportNs = end;
        }
    }
    void DecodeEnd(uint64_t end) {
        if (generationStartNs == 0 || end < generationEndNs) generationClockValid = false;
        generationEndNs = end;
    }
    bool HasDecodeRate() const {
        return generationClockValid && decodeTokens > 0 && generationStartNs > 0 &&
               generationEndNs > generationStartNs;
    }
    double DecodeTokensPerSecond() const {
        return HasDecodeRate() ? double(decodeTokens) * 1e9 /
            double(generationEndNs - generationStartNs) : -1.0;
    }
};
} // namespace fastllm
