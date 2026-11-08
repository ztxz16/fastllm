#ifndef FASTLLM_QWEN35_MIXED_ROWS_H
#define FASTLLM_QWEN35_MIXED_ROWS_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace fastllm {
namespace qwen35_mixed_rows {

// Pure CPU admission/layout oracle for one mixed target forward. Decode rows are
// emitted first; exactly one text prefill row may consume the remaining budget.
// It owns no model, KV, GDN, MTP, DFlash, page, or scheduler state.
struct Plan {
    std::vector<int> seqLens;
    std::vector<unsigned char> isPrefill;
    std::vector<std::size_t> tokenOffsets;
    int decodeTokens = 0;
    int prefillTokens = 0;
};

inline bool Fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

inline bool BuildPlan(const std::vector<int> &decodeSeqLens,
                      int prefillSeqLen, int tokenBudget, int maxPrefillTokens,
                      Plan &plan, std::string *error = nullptr) {
    Plan candidate;
    if (decodeSeqLens.empty()) return Fail(error, "mixed plan needs a decode row");
    if (prefillSeqLen <= 1) return Fail(error, "mixed plan needs a prefill row");
    if (tokenBudget <= 0 || maxPrefillTokens <= 1)
        return Fail(error, "mixed plan has invalid budget");
    if (prefillSeqLen > maxPrefillTokens)
        return Fail(error, "prefill row exceeds configured cap");

    for (int len : decodeSeqLens) {
        // Mixed rows deliberately disable MTP/DFlash for this round; decode rows
        // are therefore bounded to the target-only 1..9-token admission range.
        if (len < 1 || len > 9) return Fail(error, "decode row is outside target-only range");
        if (candidate.decodeTokens > std::numeric_limits<int>::max() - len)
            return Fail(error, "decode token count overflow");
        candidate.decodeTokens += len;
    }
    if (candidate.decodeTokens > tokenBudget ||
        prefillSeqLen > tokenBudget - candidate.decodeTokens)
        return Fail(error, "mixed plan exceeds token budget");

    candidate.prefillTokens = prefillSeqLen;
    candidate.seqLens = decodeSeqLens;
    candidate.seqLens.push_back(prefillSeqLen);
    candidate.isPrefill.assign(decodeSeqLens.size(), 0);
    candidate.isPrefill.push_back(1);
    candidate.tokenOffsets.assign(candidate.seqLens.size() + 1, 0);
    for (std::size_t i = 0; i < candidate.seqLens.size(); ++i) {
        candidate.tokenOffsets[i + 1] = candidate.tokenOffsets[i] +
            static_cast<std::size_t>(candidate.seqLens[i]);
    }
    plan = std::move(candidate);
    return true;
}

inline bool ValidateCommittedRows(const Plan &plan,
                                  const std::vector<int> &returnedRows,
                                  std::string *error = nullptr) {
    if (plan.seqLens.empty() || plan.seqLens.size() != plan.isPrefill.size() ||
        plan.tokenOffsets.size() != plan.seqLens.size() + 1)
        return Fail(error, "mixed plan shape mismatch");
    if (returnedRows.size() != plan.seqLens.size())
        return Fail(error, "native output row count mismatch");
    if (plan.isPrefill.back() == 0 || plan.prefillTokens <= 1)
        return Fail(error, "prefill row is not last");
    for (std::size_t i = 0; i < plan.seqLens.size(); ++i) {
        if (plan.seqLens[i] <= 0 || plan.tokenOffsets[i + 1] !=
            plan.tokenOffsets[i] + static_cast<std::size_t>(plan.seqLens[i]))
            return Fail(error, "mixed plan offsets are not prefix sums");
    }
    return true;
}

} // namespace qwen35_mixed_rows
} // namespace fastllm

#endif // FASTLLM_QWEN35_MIXED_ROWS_H
