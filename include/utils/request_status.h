#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <limits>
#if defined(_WIN32)
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace fastllm {
using RequestStatusFields = std::map<std::string, std::string>;
inline uint64_t RequestStatusNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline std::string StatusDecimal(double value, int digits = 3) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", digits, value);
    return text;
}
// Native and Python share sequence allocation; timestamps retain each writer's
// capture time. A sequence is an identity, not a cross-thread completion order.
inline std::atomic<uint64_t> &FastllmObservationSequence() {
    static std::atomic<uint64_t> sequence{0}; return sequence;
}
inline uint64_t FastllmNextObservationSequence() {
    return FastllmObservationSequence().fetch_add(1, std::memory_order_relaxed) + 1;
}
inline std::mutex &FastllmObservationOutputMutex() { static std::mutex mutex; return mutex; }
inline long FastllmObservationPipeAtomicLimit(int fd) noexcept {
#if defined(_WIN32)
    (void)fd;
    return 4096;
#else
    struct stat descriptor {};
    if (::fstat(fd, &descriptor) == 0 && S_ISREG(descriptor.st_mode))
        return std::numeric_limits<long>::max();
    const long value = ::fpathconf(fd, _PC_PIPE_BUF);
    return value > 0 ? value : 512; // POSIX minimum, never assume Linux 4096.
#endif
}

inline void FastllmObservationWriteDiagnostic(const char *reason, int fd,
                                              size_t bytes, size_t written,
                                              int error) noexcept {
    char message[256];
    const int count = std::snprintf(message, sizeof(message),
        "[fastllm-observability] write_diagnostic reason=%s fd=%d bytes=%zu "
        "written=%zu errno=%d\n", reason, fd, bytes, written, error);
    if (count <= 0) return;
    const size_t length = std::min<size_t>(static_cast<size_t>(count),
                                         sizeof(message) - 1);
#if defined(_WIN32)
    (void)_write(2, message, static_cast<unsigned int>(length));
#else
    (void)::write(STDERR_FILENO, message, length);
#endif
}

// Structured events are fully assembled before the stream is touched. A
// blocking pipe write <= its actual PIPE_BUF is indivisible between writers.
// A larger line is preserved but explicitly diagnosed: POSIX does not promise
// it is indivisible. Regular-file capture does not use the pipe size limit.
inline bool FastllmWriteObservationLine(FILE *stream,
                                        const std::string &line) noexcept {
    if (stream == nullptr || line.empty()) return true;
#if defined(_WIN32)
    const int fd = _fileno(stream);
#else
    const int fd = ::fileno(stream);
#endif
    if (fd < 0) {
        FastllmObservationWriteDiagnostic("invalid_fd", fd, line.size(), 0, errno);
        return false;
    }
    // Keep prior ordinary stdio output ahead of this event. The event itself is
    // submitted by one low-level write rather than several buffered calls.
    if (std::fflush(stream) != 0) {
        FastllmObservationWriteDiagnostic("flush_failed", fd, line.size(), 0, errno);
        return false;
    }
    // stdout's descriptor is fixed for one serving process. Cache the query;
    // fpathconf on every hot-path event would add an unnecessary syscall.
    static const long pipeLimit = FastllmObservationPipeAtomicLimit(fd);
    if (line.size() > static_cast<size_t>(std::max<long>(1, pipeLimit))) {
        static std::atomic<bool> reportedLongLine{false};
        if (!reportedLongLine.exchange(true, std::memory_order_relaxed))
            FastllmObservationWriteDiagnostic("line_exceeds_pipe_atomic_limit",
                                               fd, line.size(), 0, 0);
    }
    size_t offset = 0;
    while (offset < line.size()) {
        const size_t remaining = line.size() - offset;
#if defined(_WIN32)
        const int written = _write(fd, line.data() + offset,
                                   static_cast<unsigned int>(
                                       std::min<size_t>(remaining, 0x7ffff000u)));
#else
        const ssize_t written = ::write(fd, line.data() + offset, remaining);
#endif
        if (written > 0) {
            offset += static_cast<size_t>(written);
            if (offset != line.size())
                FastllmObservationWriteDiagnostic("partial_write", fd, line.size(),
                                                   offset, 0);
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        FastllmObservationWriteDiagnostic("write_failed", fd, line.size(), offset,
                                          written < 0 ? errno : 0);
        return false;
    }
    return true;
}

inline bool FastllmAppendObservationVFormat(std::string &out,
                                            const char *format,
                                            va_list args) noexcept {
    if (format == nullptr || format[0] == '\0') return true;
    va_list sizing;
    va_copy(sizing, args);
    const int needed = std::vsnprintf(nullptr, 0, format, sizing);
    va_end(sizing);
    if (needed < 0) return false;
    try {
        const size_t oldSize = out.size();
        out.resize(oldSize + static_cast<size_t>(needed) + 1);
        va_list writing;
        va_copy(writing, args);
        const int written = std::vsnprintf(&out[oldSize],
                                           static_cast<size_t>(needed) + 1,
                                           format, writing);
        va_end(writing);
        if (written < 0) return false;
        out.resize(oldSize + static_cast<size_t>(written));
        return written == needed;
    } catch (...) {
        return false;
    }
}
inline uint64_t RequestStatusWallNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
inline std::string RequestStatusWallTimestamp(uint64_t ns) {
    const std::time_t seconds = ns / 1000000000ULL;
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char buffer[32] = {}, text[48] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &utc);
    std::snprintf(text, sizeof(text), "%s.%09lluZ", buffer,
                  (unsigned long long)(ns % 1000000000ULL));
    return text;
}
struct RequestStatusSnapshot {
    std::string publicId;
    RequestStatusFields fields;
    uint64_t generation = 0, revision = 0;
    uint64_t sampleNs = 0, decodeWindowNs = 0, decodeWindowTokens = 0;
};
struct RequestStatusIdentity {
    std::string publicId;
    uint64_t generation = 0;
};
inline std::map<uint64_t, RequestStatusSnapshot> &RequestStatusSnapshots() {
    static std::map<uint64_t, RequestStatusSnapshot> snapshots; return snapshots;
}
inline std::mutex &RequestStatusMutex() { static std::mutex mutex; return mutex; }
inline RequestStatusIdentity ReadRequestStatusIdentity(uint64_t handle) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end()) return {};
    return {it->second.publicId, it->second.generation};
}
// Nonallocating generation capture for teardown, including partial begin failure.
inline uint64_t ReadRequestStatusGeneration(uint64_t handle) noexcept {
    try {
        std::lock_guard<std::mutex> lock(RequestStatusMutex());
        const auto found = RequestStatusSnapshots().find(handle);
        return found == RequestStatusSnapshots().end() ? 0 : found->second.generation;
    } catch (...) { return 0; }
}
inline uint64_t NextRequestStatusGeneration() {
    static std::atomic<uint64_t> generation{0}; return ++generation;
}
inline void ChangeRequestStatusPhaseLocked(uint64_t handle, RequestStatusSnapshot &snapshot,
                                           const char *phase, const char *boundary) {
    if (snapshot.fields["phase"] == phase) return;
    const auto prior = snapshot.fields["phase"];
    const uint64_t now = RequestStatusNowNs(), wall = RequestStatusWallNs();
    snapshot.fields["phase"] = phase;
    snapshot.fields["phase_revision"] = std::to_string(++snapshot.revision);
    snapshot.fields["phase_observed_ns"] = std::to_string(now);
    snapshot.fields["phase_observed_unix_ns"] = std::to_string(wall);
    snapshot.fields["phase_boundary"] = boundary;
    snapshot.fields["phase_semantics"] = "last_confirmed_native_boundary";
    const std::string stamp = RequestStatusWallTimestamp(wall);
    std::lock_guard<std::mutex> output(FastllmObservationOutputMutex());
    std::string line = "[fastllm-event] seq=" +
        std::to_string(FastllmNextObservationSequence()) +
        " timestamp_monotonic_ns=" + std::to_string(now) +
        " timestamp_utc=" + stamp +
        " source=native category=lifecycle request=" + std::to_string(handle) +
        " native_handle=" + std::to_string(handle) +
        " native_generation=" + std::to_string(snapshot.generation) +
        " event=request_phase phase=" + (phase == nullptr ? "" : phase) +
        " previous_phase=" + (prior.empty() ? "none" : prior) +
        " phase_revision=" + std::to_string(snapshot.revision) +
        " phase_boundary=" + (boundary == nullptr ? "" : boundary);
    if (!snapshot.publicId.empty()) line += " public_request=" + snapshot.publicId;
    line.push_back('\n');
    (void)FastllmWriteObservationLine(stdout, line);
}
inline void CreateRequestStatus(uint64_t handle) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto &snapshot = RequestStatusSnapshots()[handle];
    snapshot = RequestStatusSnapshot{};
    snapshot.generation = NextRequestStatusGeneration();
    snapshot.fields["native_context_observed"] = "1";
    ChangeRequestStatusPhaseLocked(handle, snapshot, "queued", "native_context_created");
}
inline void RegisterRequestStatus(uint64_t handle, const char *publicId) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto &snapshot = RequestStatusSnapshots()[handle];
    // Binding the public ID must preserve observations made before Python
    // regained control from launch. Re-registration of the same owner is idempotent.
    if (snapshot.generation == 0 || (!snapshot.publicId.empty() && snapshot.publicId != publicId)) {
        snapshot = RequestStatusSnapshot{};
        snapshot.generation = NextRequestStatusGeneration();
        ChangeRequestStatusPhaseLocked(handle, snapshot, "preparing", "public_id_bound_without_native_create");
    }
    snapshot.publicId = publicId;
    const auto now = RequestStatusNowNs();
    const auto stamp = RequestStatusWallTimestamp(RequestStatusWallNs());
    std::lock_guard<std::mutex> output(FastllmObservationOutputMutex());
    std::string line = "[fastllm-event] seq=" +
        std::to_string(FastllmNextObservationSequence()) +
        " timestamp_monotonic_ns=" + std::to_string(now) +
        " timestamp_utc=" + stamp +
        " source=native category=lifecycle request=" + std::to_string(handle) +
        " native_handle=" + std::to_string(handle) +
        " native_generation=" + std::to_string(snapshot.generation) +
        " public_request=" + (publicId == nullptr ? "" : publicId) +
        " event=native_observer_bound phase=" + snapshot.fields["phase"] + "\n";
    (void)FastllmWriteObservationLine(stdout, line);
}
inline bool DetachRequestStatus(uint64_t handle, const std::string &publicId) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end()) return true;
    if (!publicId.empty() && it->second.publicId != publicId) return false;
    auto &fields = it->second.fields;
    fields["observer_detached"] = "1";
    return fields["phase"] == "native_closed" || !fields.count("native_context_observed");
}
inline void RemoveRequestStatus(uint64_t handle, const std::string &publicId = {},
                                uint64_t generation = 0) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it != RequestStatusSnapshots().end() && (!generation || it->second.generation == generation) &&
        (publicId.empty() || it->second.publicId == publicId))
        RequestStatusSnapshots().erase(it);
}
inline void SetRequestStatusPhase(uint64_t handle, const char *phase,
                                  const char *boundary = "native_boundary") {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it != RequestStatusSnapshots().end())
        ChangeRequestStatusPhaseLocked(handle, it->second, phase, boundary);
}
inline bool CloseRequestStatus(uint64_t handle, uint64_t generation = 0) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end() ||
        (generation && it->second.generation != generation)) return false;
    ChangeRequestStatusPhaseLocked(handle, it->second, "native_closed", "native_context_cleanup_completed");
    it->second.fields["native_lifecycle"] = "closed";
    const bool detached = it->second.fields["observer_detached"] == "1";
    // A very short request can close before Python binds its public ID.
    // Keep its value snapshot until bind/release or the next handle generation.
    return detached;
}
inline void BeginRequestPrefillStatus(uint64_t handle, uint64_t total) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end()) return;
    ChangeRequestStatusPhaseLocked(handle, it->second, "prefill", "native_prefill_selected");
    it->second.fields["prefill_total_tokens"] = std::to_string(total);
}
// Only existing forward/chunk completion boundaries call this; no GPU sync.
inline void RecordRequestPrefillStatus(uint64_t handle, uint64_t completed,
                                       uint64_t beginNs, uint64_t endNs) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end()) return;
    auto &snapshot = it->second;
    ChangeRequestStatusPhaseLocked(handle, snapshot, "prefill", "native_prefill_forward_returned");
    snapshot.fields["prefill_completed_tokens"] = std::to_string(completed);
    snapshot.fields["prefill_measurement"] = "native_forward_returned_chunk_excluding_vision";
    if (endNs > beginNs && completed > 0) {
        const auto prior = snapshot.fields.find("prefill_completed_before");
        const uint64_t before = prior == snapshot.fields.end() ? 0 : std::stoull(prior->second);
        if (completed > before) {
            snapshot.fields["native_interval_tokens"] = std::to_string(completed - before);
            snapshot.fields["native_interval_begin_ns"] = std::to_string(beginNs);
            snapshot.fields["native_interval_end_ns"] = std::to_string(endNs);
            snapshot.fields["native_interval_tokens_s"] =
                StatusDecimal(double(completed - before) * 1e9 / double(endNs - beginNs));
            snapshot.fields["native_interval_ms"] = StatusDecimal(double(endNs - beginNs) / 1e6);
            snapshot.sampleNs = endNs;
        }
    }
    snapshot.fields["prefill_completed_before"] = std::to_string(completed);
}
inline void RecordRequestDecodeStatus(uint64_t handle, uint64_t generated,
                                      uint64_t decoded, uint64_t generationBeginNs,
                                      uint64_t endNs, bool final) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end()) return;
    auto &snapshot = it->second;
    const auto priorPhase = snapshot.fields["phase"];
    if (priorPhase != "decode" && priorPhase != "finishing") {
        snapshot.fields.erase("native_interval_tokens_s");
        snapshot.fields.erase("native_interval_ms");
        snapshot.fields.erase("native_interval_tokens");
        snapshot.fields.erase("native_interval_begin_ns");
        snapshot.fields.erase("native_interval_end_ns");
        snapshot.fields["prefill_measurement"] = "none";
        snapshot.sampleNs = 0;
    }
    ChangeRequestStatusPhaseLocked(handle, snapshot, final ? "finishing" : "decode",
                                   final ? "native_decode_ending" : "native_decode_ready_or_returned");
    snapshot.fields["native_output_tokens"] = std::to_string(generated);
    if (snapshot.decodeWindowNs == 0) snapshot.decodeWindowNs = generationBeginNs;
    if (endNs > snapshot.decodeWindowNs && decoded >= snapshot.decodeWindowTokens &&
        (final || endNs - snapshot.decodeWindowNs >= 1000000000ULL)) {
        snapshot.fields["native_interval_tokens"] = std::to_string(decoded - snapshot.decodeWindowTokens);
        snapshot.fields["native_interval_begin_ns"] = std::to_string(snapshot.decodeWindowNs);
        snapshot.fields["native_interval_end_ns"] = std::to_string(endNs);
        snapshot.fields["native_interval_tokens_s"] = StatusDecimal(
            double(decoded - snapshot.decodeWindowTokens) * 1e9 /
            double(endNs - snapshot.decodeWindowNs));
        snapshot.fields["native_interval_ms"] =
            StatusDecimal(double(endNs - snapshot.decodeWindowNs) / 1e6);
        snapshot.sampleNs = endNs;
        snapshot.decodeWindowNs = endNs;
        snapshot.decodeWindowTokens = decoded;
        snapshot.fields["prefill_measurement"] = "none";
    }
}
inline bool ReadRequestStatus(uint64_t handle, const std::string &publicId,
                               RequestStatusFields &fields) {
    std::lock_guard<std::mutex> lock(RequestStatusMutex());
    auto it = RequestStatusSnapshots().find(handle);
    if (it == RequestStatusSnapshots().end() || it->second.publicId != publicId) return false;
    fields = it->second.fields;
    fields.erase("prefill_completed_before");
    const auto now = RequestStatusNowNs();
    fields["native_generation"] = std::to_string(it->second.generation);
    fields["native_status_sample_ns"] = std::to_string(now);
    fields["native_status_sample_unix_ns"] = std::to_string(RequestStatusWallNs());
    fields["native_status_state"] = "observed";
    fields["native_lifecycle"] = fields.count("native_lifecycle") ? fields["native_lifecycle"] : "active";
    if (it->second.sampleNs && now >= it->second.sampleNs) {
        fields["native_sample_age_ms"] = StatusDecimal(double(now - it->second.sampleNs) / 1e6);
        fields["native_rate_state"] = now - it->second.sampleNs <= 2500000000ULL ? "fresh" : "stale";
    } else fields["native_rate_state"] = it->second.sampleNs ? "clock_inconsistent" : "pending";
    return true;
}
struct L1CapacityPageSample { uint64_t total = 0, ready = 0, empty = 0, pageLen = 0, bytes = 0; };
// Required layer/K/V/TP managers represent the same logical token positions.
inline RequestStatusFields L1CapacityFields(const std::vector<L1CapacityPageSample> &samples) {
    if (samples.empty()) return {{"l1_capacity_state", "unavailable"}};
    uint64_t total = UINT64_MAX, ready = UINT64_MAX, empty = UINT64_MAX, bytes = 0;
    for (const auto &sample : samples) {
        if (!sample.pageLen || sample.ready > sample.total || sample.empty > sample.ready)
            return {{"l1_capacity_state", "inconsistent"}};
        total = std::min(total, sample.total * sample.pageLen);
        ready = std::min(ready, sample.ready * sample.pageLen);
        empty = std::min(empty, sample.empty * sample.pageLen);
        bytes += sample.bytes;
    }
    ready = std::min(ready, total);
    empty = std::min(empty, ready);
    return {{"l1_capacity_state", "ok"}, {"l1_total_tokens", std::to_string(total)},
            {"l1_available_tokens", std::to_string(empty)},
            {"l1_allocatable_tokens", std::to_string(ready)},
            {"l1_used_tokens", std::to_string(total - empty)},
            {"l1_accounting", "limiting_required_kv_pool"}, {"l1_physical_bytes", std::to_string(bytes)}};
}
inline void ByteCapacityFields(RequestStatusFields &fields, const std::string &tier,
                                uint64_t used, uint64_t limit) {
    const uint64_t available = used < limit ? limit - used : 0;
    fields[tier + "_used_bytes"] = std::to_string(used);
    fields[tier + "_limit_bytes"] = std::to_string(limit);
    fields[tier + "_available_bytes"] = std::to_string(available);
    fields[tier + "_used_gib"] = StatusDecimal(double(used) / double(1ULL << 30));
    fields[tier + "_limit_gib"] = StatusDecimal(double(limit) / double(1ULL << 30));
    fields[tier + "_available_gib"] = StatusDecimal(double(available) / double(1ULL << 30));
}
} // namespace fastllm
