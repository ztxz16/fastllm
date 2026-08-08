#pragma once

#include <set>
#include <string>
#include <vector>

#include "json11.hpp"

struct OpenAIStopEncoding {
    int exactTokenId = -1;
    std::vector<int> tokenSequence;
};

template <typename ExactTokenLookup, typename FallbackEncode>
inline OpenAIStopEncoding EncodeOpenAIStop(
        const std::string &stop,
        ExactTokenLookup exactTokenLookup,
        FallbackEncode fallbackEncode) {
    OpenAIStopEncoding encoding;
    exactTokenLookup(stop, encoding.exactTokenId);
    encoding.tokenSequence = fallbackEncode(stop);
    return encoding;
}

template <typename EncodeStop>
inline bool ParseOpenAIStop(
        const json11::Json &stop,
        EncodeStop encodeStop,
        std::multiset<int> &stopTokenIds,
        std::vector<std::vector<int>> &stopTokenSequences,
        std::vector<std::string> &parsedStopStrings,
        std::string &error) {
    error.clear();
    if (stop.is_null()) {
        return true;
    }

    std::vector<std::string> stopStrings;
    if (stop.is_string()) {
        stopStrings.push_back(stop.string_value());
    } else if (stop.is_array()) {
        for (const auto &item : stop.array_items()) {
            if (!item.is_string()) {
                error = "stop must be a string or an array of strings";
                return false;
            }
            stopStrings.push_back(item.string_value());
        }
    } else {
        error = "stop must be a string or an array of strings";
        return false;
    }

    std::set<int> parsedTokenIds;
    std::set<std::vector<int>> parsedTokenSequences;
    std::set<std::string> uniqueStopStrings;
    for (const auto &stopString : stopStrings) {
        if (stopString.empty()) {
            error = "stop strings must not be empty";
            return false;
        }
        uniqueStopStrings.insert(stopString);
        const OpenAIStopEncoding encoding = encodeStop(stopString);
        if (encoding.exactTokenId >= 0) {
            parsedTokenIds.insert(encoding.exactTokenId);
        }
        if (encoding.tokenSequence.size() == 1) {
            parsedTokenIds.insert(encoding.tokenSequence.front());
        } else if (encoding.tokenSequence.size() > 1) {
            parsedTokenSequences.insert(encoding.tokenSequence);
        }
        if (encoding.exactTokenId < 0 && encoding.tokenSequence.empty()) {
            error = "each stop string must encode to at least one token";
            return false;
        }
    }

    stopTokenIds.insert(parsedTokenIds.begin(), parsedTokenIds.end());
    stopTokenSequences.insert(stopTokenSequences.end(),
                              parsedTokenSequences.begin(),
                              parsedTokenSequences.end());
    parsedStopStrings.insert(parsedStopStrings.end(),
                             uniqueStopStrings.begin(),
                             uniqueStopStrings.end());
    return true;
}
