#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>

#include "json11.hpp"

constexpr int kDefaultOutputTokenLimit = 16384;

inline bool ParsePositiveInt(const std::string &text, int &value,
                             std::string &error) {
    if (text.empty()) {
        error = "value must be a positive integer";
        return false;
    }
    errno = 0;
    char *end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        error = "value must be a positive integer in range [1, " +
                std::to_string(INT_MAX) + "]";
        return false;
    }
    value = static_cast<int>(parsed);
    error.clear();
    return true;
}

inline bool ResolveOutputTokenLimit(const json11::Json &value,
                                    int defaultLimit,
                                    int &selectedLimit,
                                    std::string &error) {
    if (defaultLimit <= 0) {
        error = "server default output token limit must be positive";
        return false;
    }
    if (value.is_null()) {
        selectedLimit = defaultLimit;
        error.clear();
        return true;
    }
    if (!value.is_number()) {
        error = "max_tokens must be a positive integer";
        return false;
    }
    const double number = value.number_value();
    if (!std::isfinite(number) || number < 1.0 || number > INT_MAX ||
        std::floor(number) != number) {
        error = "max_tokens must be a positive integer in range [1, " +
                std::to_string(INT_MAX) + "]";
        return false;
    }
    selectedLimit = static_cast<int>(number);
    error.clear();
    return true;
}
