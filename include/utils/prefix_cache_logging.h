#pragma once

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <atomic>
#include "utils/request_status.h"
#include "utils/request_diagnostics.h"

namespace fastllm {

// Runtime observation only. Never include this setting in a cache identity or
// use it to change checkpoint admission, restoration, durability or counters.
enum class PrefixCacheLogLevel { Off, Info, Debug };

enum class FastllmObservabilityMode { Off, User, Debug };

inline FastllmObservabilityMode GetFastllmObservabilityMode() {
    static const FastllmObservabilityMode mode = [] {
        const char *text = std::getenv("FASTLLM_OBSERVABILITY_MODE");
        if (text == nullptr) return FastllmObservabilityMode::Off;
        std::string value(text);
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) value.clear();
        else value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
        for (char &c : value) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (value == "user") return FastllmObservabilityMode::User;
        if (value == "debug") return FastllmObservabilityMode::Debug;
        if (!value.empty() && value != "off") {
            std::fprintf(stderr, "[fastllm-observability] invalid FASTLLM_OBSERVABILITY_MODE; "
                                 "expected off, user or debug; using off.\n");
        }
        return FastllmObservabilityMode::Off;
    }();
    return mode;
}

inline bool FastllmObservabilityUserEnabled() {
    return GetFastllmObservabilityMode() == FastllmObservabilityMode::User;
}

inline bool FastllmObservabilityDebugEnabled() {
    return GetFastllmObservabilityMode() == FastllmObservabilityMode::Debug;
}

inline uint64_t FastllmObservationTimestampNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::string FastllmObservationWallTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &utc);
    char result[40] = {};
    std::snprintf(result, sizeof(result), "%s.%03lldZ", buffer,
                  static_cast<long long>(millis.count()));
    return result;
}

inline std::map<uint64_t, std::string> &FastllmPublicRequestIds() {
    static std::map<uint64_t, std::string> ids;
    return ids;
}

inline std::mutex &FastllmPublicRequestIdsMutex() {
    static std::mutex mutex;
    return mutex;
}

inline void RegisterFastllmPublicRequestId(uint64_t nativeHandle, const char *publicId) {
    if (publicId == nullptr || publicId[0] == '\0') return;
    std::lock_guard<std::mutex> guard(FastllmPublicRequestIdsMutex());
    FastllmPublicRequestIds()[nativeHandle] = publicId;
    if (GetFastllmObservabilityMode() != FastllmObservabilityMode::Off)
        RegisterRequestStatus(nativeHandle, publicId);
    RegisterRequestDiagnostics(nativeHandle, publicId);
}

inline void BeginFastllmNativeRequest(uint64_t nativeHandle) {
    std::lock_guard<std::mutex> guard(FastllmPublicRequestIdsMutex());
    FastllmPublicRequestIds().erase(nativeHandle);
    CreateRequestStatus(nativeHandle);
    CreateRequestDiagnostics(nativeHandle);
}
inline void UnregisterFastllmPublicRequestId(uint64_t nativeHandle, const std::string &publicId = {}) {
    std::lock_guard<std::mutex> guard(FastllmPublicRequestIdsMutex());
    const auto it = FastllmPublicRequestIds().find(nativeHandle);
    if (it == FastllmPublicRequestIds().end() || (!publicId.empty() && it->second != publicId)) return;
    const bool canRemove = DetachRequestStatus(nativeHandle, publicId);
    DetachRequestDiagnostics(nativeHandle, publicId);
    if (!canRemove) return;
    FastllmPublicRequestIds().erase(it);
    RemoveRequestStatus(nativeHandle, publicId);
}
inline void EndFastllmNativeRequest(uint64_t handle) {
    CloseRequestDiagnostics(handle);
    if (CloseRequestStatus(handle)) UnregisterFastllmPublicRequestId(handle);
}

inline std::string FastllmPublicRequestId(uint64_t nativeHandle) {
    std::lock_guard<std::mutex> guard(FastllmPublicRequestIdsMutex());
    const auto found = FastllmPublicRequestIds().find(nativeHandle);
    return found == FastllmPublicRequestIds().end() ? std::string() : found->second;
}

inline PrefixCacheLogLevel GetPrefixCacheLogLevel() {
    // Python exports the resolved CLI setting before constructing the model.
    // Native callers get the same default; changing the level requires restart.
    static const PrefixCacheLogLevel level = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_LOG_LEVEL");
        if (text == nullptr) return PrefixCacheLogLevel::Info;
        std::string value(text);
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) value.clear();
        else value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
        for (char &c : value) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (value == "off") return PrefixCacheLogLevel::Off;
        if (value == "info") return PrefixCacheLogLevel::Info;
        if (value == "debug") return PrefixCacheLogLevel::Debug;
        std::fprintf(stderr, "[PrefixCache] invalid FASTLLM_PREFIX_CACHE_LOG_LEVEL; "
                             "expected off, info or debug; using info.\n");
        return PrefixCacheLogLevel::Info;
    }();
    return level;
}

inline bool PrefixCacheInfoEnabled() {
    return GetPrefixCacheLogLevel() != PrefixCacheLogLevel::Off;
}

inline bool PrefixCacheDebugEnabled() {
    return GetPrefixCacheLogLevel() == PrefixCacheLogLevel::Debug;
}

inline void PrefixCacheInfo(const char *format, ...) {
    if (!PrefixCacheInfoEnabled()) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(stdout, format, args);
    va_end(args);
    std::fflush(stdout);
}

inline void PrefixCacheDebug(const char *format, ...) {
    if (!PrefixCacheDebugEnabled()) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(stdout, format, args);
    va_end(args);
    std::fflush(stdout);
}

// Structured event lines stay plain text on stdout.  The optional terminal
// colorizer consumes the category marker on a separate channel, so server.log
// never receives ANSI escape sequences.  Keeping the whole line under one
// mutex also prevents concurrent scheduler/read-worker writes from interleaving
// their request identifiers and measurements.
inline void FastllmEvent(const char *category, uint64_t requestId,
                         const char *event, const char *format = nullptr, ...) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) return;
    if (event != nullptr) {
        const std::string name(event);
        if (name == "vision_encode_begin") SetRequestStatusPhase(requestId, "vision", "vision_encode_begin");
        else if (name == "vision_encode_end") SetRequestStatusPhase(requestId, "preparing", "vision_encode_end");
        else if (name == "cache_restore_begin") SetRequestStatusPhase(requestId, "cache_restore", "cache_restore_begin");
    }
    // User is a terminal projection of the complete debug/raw stream.
    const std::string publicId = FastllmPublicRequestId(requestId);
    RequestStatusFields identity;
    ReadRequestStatus(requestId, publicId, identity);
    std::lock_guard<std::mutex> guard(FastllmObservationOutputMutex());
    const uint64_t seq = FastllmNextObservationSequence();
    const uint64_t timestamp = FastllmObservationTimestampNs();
    const std::string wallTimestamp = FastllmObservationWallTimestamp();
    std::fprintf(stdout, "[fastllm-event] seq=%llu timestamp_monotonic_ns=%llu "
                         "timestamp_utc=%s "
                         "source=native category=%s request=%llu native_handle=%llu",
                 (unsigned long long)seq, (unsigned long long)timestamp,
                 wallTimestamp.c_str(),
                 category != nullptr ? category : "startup",
                 (unsigned long long)requestId,
                 (unsigned long long)requestId);
    if (!publicId.empty()) std::fprintf(stdout, " public_request=%s", publicId.c_str());
    if (identity.count("native_generation"))
        std::fprintf(stdout, " native_generation=%s", identity["native_generation"].c_str());
    std::fprintf(stdout, " event=%s",
                 event != nullptr ? event : "log");
    if (format != nullptr && format[0] != '\0') {
        std::fputc(' ', stdout);
        va_list args;
        va_start(args, format);
        std::vfprintf(stdout, format, args);
        va_end(args);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// Warnings and errors remain unconditional stderr messages, including at off.
} // namespace fastllm
