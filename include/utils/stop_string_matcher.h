#pragma once

#include <algorithm>
#include <string>
#include <vector>

inline bool PushStopText(const std::vector<std::string> &stopStrings,
                         std::string &pendingText,
                         const std::string &text,
                         std::string &readyText) {
    readyText.clear();
    pendingText += text;

    size_t matchPos = std::string::npos;
    for (const auto &stop : stopStrings) {
        if (stop.empty()) {
            continue;
        }
        size_t pos = pendingText.find(stop);
        if (pos != std::string::npos &&
            (matchPos == std::string::npos || pos < matchPos)) {
            matchPos = pos;
        }
    }
    if (matchPos != std::string::npos) {
        readyText = pendingText.substr(0, matchPos);
        pendingText.clear();
        return true;
    }

    size_t keepLength = 0;
    for (const auto &stop : stopStrings) {
        size_t candidateLength = std::min(stop.size(), pendingText.size());
        while (candidateLength > keepLength) {
            if (pendingText.compare(pendingText.size() - candidateLength,
                                    candidateLength, stop, 0,
                                    candidateLength) == 0) {
                keepLength = candidateLength;
                break;
            }
            candidateLength--;
        }
    }

    readyText = pendingText.substr(0, pendingText.size() - keepLength);
    pendingText.erase(0, pendingText.size() - keepLength);
    return false;
}

inline void FlushPendingStopText(std::string &pendingText,
                                 std::string &readyText) {
    readyText = pendingText;
    pendingText.clear();
}
