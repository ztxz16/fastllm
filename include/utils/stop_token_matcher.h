#pragma once

#include <algorithm>
#include <vector>

inline int PushStopToken(const std::vector<std::vector<int>> &stopSequences,
                         std::vector<int> &pendingTokens,
                         int token,
                         std::vector<int> &readyTokens) {
    readyTokens.clear();
    pendingTokens.push_back(token);

    size_t matchedLength = 0;
    for (const auto &sequence : stopSequences) {
        if (sequence.empty() || sequence.size() > pendingTokens.size()) {
            continue;
        }
        auto suffix = pendingTokens.end() - sequence.size();
        if (std::equal(sequence.begin(), sequence.end(), suffix) &&
            sequence.size() > matchedLength) {
            matchedLength = sequence.size();
        }
    }
    if (matchedLength > 0) {
        readyTokens.insert(readyTokens.end(), pendingTokens.begin(),
                           pendingTokens.end() - matchedLength);
        pendingTokens.clear();
        return static_cast<int>(matchedLength);
    }

    size_t keepLength = 0;
    for (const auto &sequence : stopSequences) {
        size_t candidateLength = std::min(sequence.size(), pendingTokens.size());
        while (candidateLength > keepLength) {
            auto suffix = pendingTokens.end() - candidateLength;
            if (std::equal(suffix, pendingTokens.end(), sequence.begin())) {
                keepLength = candidateLength;
                break;
            }
            candidateLength--;
        }
    }

    readyTokens.insert(readyTokens.end(), pendingTokens.begin(),
                       pendingTokens.end() - keepLength);
    if (keepLength == 0) {
        pendingTokens.clear();
    } else {
        pendingTokens.erase(pendingTokens.begin(),
                            pendingTokens.end() - keepLength);
    }
    return 0;
}

inline void FlushPendingStopTokens(std::vector<int> &pendingTokens,
                                   std::vector<int> &readyTokens) {
    readyTokens = pendingTokens;
    pendingTokens.clear();
}

template <typename EmitToken>
inline void FlushPendingStopTokensTo(
        std::vector<int> &pendingTokens,
        EmitToken emitToken) {
    for (int token : pendingTokens) {
        emitToken(token);
    }
    pendingTokens.clear();
}
