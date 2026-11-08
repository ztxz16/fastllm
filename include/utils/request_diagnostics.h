#pragma once
// Independent bounded request evidence. This registry never acquires status,
// model, allocator, cache, event-output or CUDA locks.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#if defined(__linux__)
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace fastllm {
using RequestDiagnosticFields = std::map<std::string, std::string>;
inline bool RequestDiagnosticsEnabled() {
    static const bool enabled = [] {
        const char *raw = std::getenv("FASTLLM_OBSERVABILITY_MODE");
        if (!raw) return false;
        std::string text(raw);
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return false;
        text = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
        for (char &c : text) c = char(std::tolower(static_cast<unsigned char>(c)));
        return text == "debug";
    }();
    return enabled;
}
inline uint64_t RequestDiagnosticNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint64_t RequestDiagnosticWallNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
inline uint64_t RequestDiagnosticThreadId() {
#if defined(__linux__)
    return uint64_t(syscall(SYS_gettid));
#else
    return 0; // Unsupported platform, never invent a thread ID.
#endif
}
inline std::string RequestDiagnosticHex(uint64_t value) {
    std::ostringstream out; out << "0x" << std::hex << value; return out.str();
}
inline std::string RequestDiagnosticResource(const char *name, const void *address) {
    if (!RequestDiagnosticsEnabled()) return {};
    return std::string(name) + "@" + RequestDiagnosticHex(reinterpret_cast<uintptr_t>(address));
}
inline std::string RequestDiagnosticQuote(const std::string &text) {
    static const char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        // No literal whitespace: the surrounding ABI is key=value per line.
        if (c <= 32 || c >= 127) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else result += char(c);
    }
    return result + "\"";
}
struct RequestDiagnosticStack {
    struct Frame { uint64_t address = 0, moduleBase = 0; std::string module; };
    uint64_t capturedNs = 0, capturedWallNs = 0, elapsedNs = 0, operationId = 0;
    std::vector<Frame> frames;
    std::string state = "not_captured";
};
struct RequestDiagnosticRow {
    uint64_t id = 0, threadId = 0, startNs = 0, startWallNs = 0, endNs = 0, endWallNs = 0;
    std::string operation, resource, state = "running";
    std::shared_ptr<RequestDiagnosticStack> stack;
};
struct RequestDiagnosticSnapshot {
    uint64_t generation = 0, revision = 0, nextOperation = 0, droppedActive = 0, retired = 0;
    uint64_t modelKey = 0, progressRevision = 0, progressEndNs = 0;
    std::string publicId;
    bool closed = false, detached = false, nativeCreated = false;
    std::map<uint64_t, RequestDiagnosticRow> active;
    std::vector<RequestDiagnosticRow> recent;
    std::map<std::string, std::shared_ptr<RequestDiagnosticStack>> stacks;
};
inline std::mutex &RequestDiagnosticsMutex() { static std::mutex mutex; return mutex; }
inline std::map<uint64_t, std::shared_ptr<RequestDiagnosticSnapshot>> &RequestDiagnosticsRegistry() {
    static std::map<uint64_t, std::shared_ptr<RequestDiagnosticSnapshot>> registry; return registry;
}
inline uint64_t NextRequestDiagnosticGeneration() {
    static std::atomic<uint64_t> generation{0}; return ++generation;
}
inline std::shared_ptr<RequestDiagnosticSnapshot> NewRequestDiagnosticSnapshot() {
    auto value = std::make_shared<RequestDiagnosticSnapshot>();
    value->generation = NextRequestDiagnosticGeneration(); value->recent.reserve(9); return value;
}
inline void CreateRequestDiagnostics(uint64_t handle) {
    if (!RequestDiagnosticsEnabled()) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    RequestDiagnosticsRegistry()[handle] = NewRequestDiagnosticSnapshot();
    RequestDiagnosticsRegistry()[handle]->nativeCreated = true;
}
inline void RegisterRequestDiagnostics(uint64_t handle, const char *publicId) {
    if (!RequestDiagnosticsEnabled() || !publicId || !*publicId) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    auto &snapshot = RequestDiagnosticsRegistry()[handle];
    if (!snapshot || (!snapshot->publicId.empty() && snapshot->publicId != publicId))
        snapshot = NewRequestDiagnosticSnapshot();
    snapshot->publicId = publicId; // Preserve close-before-bind and pre-bind work.
}
inline void RetireRequestDiagnosticsIfDetached(uint64_t handle,
        const std::shared_ptr<RequestDiagnosticSnapshot> &snapshot) {
    if ((snapshot->nativeCreated && !snapshot->closed) || !snapshot->detached || !snapshot->active.empty()) return;
    auto found = RequestDiagnosticsRegistry().find(handle);
    if (found != RequestDiagnosticsRegistry().end() && found->second == snapshot)
        RequestDiagnosticsRegistry().erase(found);
}
inline void DetachRequestDiagnostics(uint64_t handle, const std::string &publicId = {}) {
    if (!RequestDiagnosticsEnabled()) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    const auto found = RequestDiagnosticsRegistry().find(handle);
    if (found == RequestDiagnosticsRegistry().end() ||
        (!publicId.empty() && found->second->publicId != publicId)) return;
    auto snapshot = found->second;
    snapshot->detached = true; ++snapshot->revision;
    RetireRequestDiagnosticsIfDetached(handle, snapshot);
}
inline void CloseRequestDiagnostics(uint64_t handle) {
    if (!RequestDiagnosticsEnabled()) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    auto found = RequestDiagnosticsRegistry().find(handle);
    if (found == RequestDiagnosticsRegistry().end()) return;
    auto snapshot = found->second;
    snapshot->closed = true; ++snapshot->revision;
    RetireRequestDiagnosticsIfDetached(handle, snapshot);
}
struct RequestDiagnosticToken { uint64_t handle = 0, generation = 0; };
inline RequestDiagnosticToken CaptureRequestDiagnosticToken(uint64_t handle) {
    if (!RequestDiagnosticsEnabled()) return {};
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    const auto found = RequestDiagnosticsRegistry().find(handle);
    return found == RequestDiagnosticsRegistry().end() ? RequestDiagnosticToken{} :
        RequestDiagnosticToken{handle, found->second->generation};
}
inline void BindRequestDiagnosticModel(uint64_t handle, const void *model) {
    if (!RequestDiagnosticsEnabled()) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    const auto found = RequestDiagnosticsRegistry().find(handle);
    if (found != RequestDiagnosticsRegistry().end())
        found->second->modelKey = reinterpret_cast<uintptr_t>(model);
}
inline std::vector<uint64_t> RequestDiagnosticHandlesForModel(const void *model) {
    std::vector<uint64_t> handles;
    if (!RequestDiagnosticsEnabled()) return handles;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    for (const auto &entry : RequestDiagnosticsRegistry())
        if (!entry.second->closed && entry.second->modelKey == reinterpret_cast<uintptr_t>(model))
            handles.push_back(entry.first);
    return handles;
}
inline void RecordRequestDiagnosticProgress(uint64_t handle, uint64_t endNs) {
    if (!RequestDiagnosticsEnabled()) return;
    std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
    const auto found = RequestDiagnosticsRegistry().find(handle);
    if (found != RequestDiagnosticsRegistry().end() && endNs > found->second->progressEndNs) {
        ++found->second->progressRevision; found->second->progressEndNs = endNs;
    }
}
inline std::vector<uint64_t> &RequestDiagnosticCurrentHandles() {
    static thread_local std::vector<uint64_t> handles; return handles;
}
class RequestDiagnosticOperation {
    uint64_t handle = 0, id = 0;
    int exceptions = 0;
    std::shared_ptr<RequestDiagnosticSnapshot> snapshot;
public:
    RequestDiagnosticOperation(uint64_t handle, const char *operation, const char *resource,
                               uint64_t expectedGeneration = 0)
            : handle(handle) {
        if (!RequestDiagnosticsEnabled()) return;
        try {
        exceptions = std::uncaught_exceptions();
        std::shared_ptr<RequestDiagnosticStack> capture;
        {
            std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
            const auto found = RequestDiagnosticsRegistry().find(handle);
            if (found == RequestDiagnosticsRegistry().end() || found->second->closed ||
                (expectedGeneration && found->second->generation != expectedGeneration)) return;
            snapshot = found->second;
            if (snapshot->active.size() >= 32) { ++snapshot->droppedActive; snapshot.reset(); return; }
            RequestDiagnosticRow row;
            row.id = id = ++snapshot->nextOperation; row.threadId = RequestDiagnosticThreadId();
            row.startNs = RequestDiagnosticNowNs(); row.startWallNs = RequestDiagnosticWallNs();
            row.operation = std::string(operation ? operation : "unknown").substr(0, 128);
            row.resource = std::string(resource ? resource : "unknown").substr(0, 192);
            const auto key = row.operation + ":" + row.resource + ":" + std::to_string(row.threadId);
            auto saved = snapshot->stacks.find(key);
            if (saved != snapshot->stacks.end()) row.stack = saved->second;
            else if (snapshot->stacks.size() < 16) {
                row.stack = capture = std::make_shared<RequestDiagnosticStack>();
                capture->operationId = id; capture->state = "capture_pending";
                snapshot->stacks[key] = capture;
            }
            snapshot->active.emplace(id, row); ++snapshot->revision;
        }
        // First entry per operation/resource/thread only. No output, GPU call,
        // symbol-name demangling, signal handler, ptrace or service pause.
        if (capture) {
            RequestDiagnosticStack value;
            value.operationId = id; value.capturedNs = RequestDiagnosticNowNs();
            value.capturedWallNs = RequestDiagnosticWallNs();
#if defined(__linux__)
            void *addresses[8] = {};
            const int count = ::backtrace(addresses, 8);
            for (int i = 0; i < count; ++i) {
                RequestDiagnosticStack::Frame frame;
                frame.address = reinterpret_cast<uintptr_t>(addresses[i]);
                Dl_info info{};
                if (::dladdr(addresses[i], &info) && info.dli_fbase) {
                    frame.moduleBase = reinterpret_cast<uintptr_t>(info.dli_fbase);
                    if (info.dli_fname) frame.module = std::string(info.dli_fname).substr(0, 256);
                }
                value.frames.push_back(std::move(frame));
            }
            value.state = count > 0 ? "captured" : "empty";
#else
            value.state = "unsupported_platform";
#endif
            value.elapsedNs = RequestDiagnosticNowNs() - value.capturedNs;
            std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
            *capture = std::move(value);
        }
        } catch (...) {
            // Diagnostic allocation/unwind failure must not abort inference.
            if (snapshot && id) {
                std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
                snapshot->active.erase(id); ++snapshot->droppedActive;
            }
            snapshot.reset(); id = 0;
        }
    }
    RequestDiagnosticOperation(RequestDiagnosticToken token, const char *operation, const char *resource)
        : RequestDiagnosticOperation(token.handle, operation, resource,
              token.generation ? token.generation : UINT64_MAX) {}
    RequestDiagnosticOperation(const RequestDiagnosticOperation &) = delete;
    RequestDiagnosticOperation &operator=(const RequestDiagnosticOperation &) = delete;
    void Finish() {
        if (!snapshot || !id) return;
        const uint64_t ended = RequestDiagnosticNowNs(), wall = RequestDiagnosticWallNs();
        std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
        auto row = snapshot->active.find(id);
        if (row != snapshot->active.end()) {
            row->second.endNs = ended; row->second.endWallNs = wall;
            row->second.state = std::uncaught_exceptions() > exceptions ? "unwound" : "returned";
            snapshot->recent.push_back(std::move(row->second)); ++snapshot->retired;
            if (snapshot->recent.size() > 8) snapshot->recent.erase(snapshot->recent.begin());
            snapshot->active.erase(row); ++snapshot->revision;
        }
        RetireRequestDiagnosticsIfDetached(handle, snapshot);
        id = 0; snapshot.reset();
    }
    ~RequestDiagnosticOperation() { try { Finish(); } catch (...) {} }
};
class RequestDiagnosticBatchOperation {
    std::vector<std::unique_ptr<RequestDiagnosticOperation>> operations;
    std::vector<uint64_t> prior;
    bool enabled = false;
public:
    template<class Handles> RequestDiagnosticBatchOperation(
            const Handles &handles, const char *operation, const char *resource) {
        if (!RequestDiagnosticsEnabled()) return;
        try {
            prior = RequestDiagnosticCurrentHandles(); enabled = true;
            RequestDiagnosticCurrentHandles().clear();
            for (auto handle : handles) {
                RequestDiagnosticCurrentHandles().push_back(uint64_t(handle));
                operations.emplace_back(new RequestDiagnosticOperation(uint64_t(handle), operation, resource));
            }
        } catch (...) {
            operations.clear();
            if (enabled) RequestDiagnosticCurrentHandles().swap(prior);
            enabled = false;
        }
    }
    void Finish() {
        if (!enabled) return;
        operations.clear(); RequestDiagnosticCurrentHandles().swap(prior); enabled = false;
    }
    ~RequestDiagnosticBatchOperation() { Finish(); }
};
class RequestDiagnosticCurrentOperation : public RequestDiagnosticBatchOperation {
public:
    RequestDiagnosticCurrentOperation(const char *operation, const char *resource)
        : RequestDiagnosticBatchOperation(std::vector<uint64_t>(RequestDiagnosticCurrentHandles()), operation, resource) {}
};
inline std::string RequestDiagnosticRowJson(const RequestDiagnosticRow &row, uint64_t now) {
    std::ostringstream out;
    out << "{\"operation_id\":" << row.id << ",\"operation\":" << RequestDiagnosticQuote(row.operation)
        << ",\"resource\":" << RequestDiagnosticQuote(row.resource) << ",\"thread_id\":" << row.threadId
        << ",\"state\":" << RequestDiagnosticQuote(row.state)
        << ",\"start_monotonic_ns\":" << row.startNs << ",\"start_unix_ns\":" << row.startWallNs
        << ",\"end_monotonic_ns\":" << row.endNs << ",\"end_unix_ns\":" << row.endWallNs
        << ",\"elapsed_ns\":" << ((row.endNs ? row.endNs : now) >= row.startNs ?
                                  (row.endNs ? row.endNs : now) - row.startNs : 0);
    if (row.stack) {
        const auto &stack = *row.stack;
        out << ",\"stack_scope\":\"first_operation_entry_same_label_not_current\""
            << ",\"entry_stack_state\":" << RequestDiagnosticQuote(stack.state)
            << ",\"entry_stack_operation_id\":" << stack.operationId
            << ",\"entry_stack_capture_monotonic_ns\":" << stack.capturedNs
            << ",\"entry_stack_capture_unix_ns\":" << stack.capturedWallNs
            << ",\"entry_stack_capture_elapsed_ns\":" << stack.elapsedNs
            << ",\"entry_stack_frames\":[";
        bool first = true;
        for (const auto &frame : stack.frames) {
            if (!first) { out << ','; }
            first = false;
            out << "{\"address\":" << RequestDiagnosticQuote(RequestDiagnosticHex(frame.address))
                << ",\"module\":" << RequestDiagnosticQuote(frame.module)
                << ",\"module_base\":" << RequestDiagnosticQuote(RequestDiagnosticHex(frame.moduleBase))
                << ",\"relative_address\":" << RequestDiagnosticQuote(RequestDiagnosticHex(
                    frame.moduleBase && frame.address >= frame.moduleBase ? frame.address - frame.moduleBase : 0))
                << '}';
        }
        out << ']';
    } else out << ",\"entry_stack_state\":\"label_budget_exhausted\",\"stack_scope\":\"unavailable\"";
    out << '}'; return out.str();
}
inline bool ReadRequestDiagnostics(uint64_t handle, const std::string &publicId,
                                    RequestDiagnosticFields &fields) {
    fields.clear();
    fields["native_diag_sample_ns"] = std::to_string(RequestDiagnosticNowNs());
    fields["native_diag_sample_unix_ns"] = std::to_string(RequestDiagnosticWallNs());
    if (!RequestDiagnosticsEnabled()) { fields["native_diag_state"] = "disabled"; return true; }
    std::unique_lock<std::mutex> lock(RequestDiagnosticsMutex(), std::try_to_lock);
    if (!lock.owns_lock()) { fields["native_diag_state"] = "busy"; return true; }
    const auto found = RequestDiagnosticsRegistry().find(handle);
    if (found == RequestDiagnosticsRegistry().end()) { fields["native_diag_state"] = "missing"; return true; }
    if (found->second->publicId != publicId) { fields["native_diag_state"] = "owner_mismatch"; return true; }
    // Deep-copy bounded rows/stacks, release the sole diagnostic mutex before
    // JSON serialization. Reader cannot wait on any inference/cache lock.
    RequestDiagnosticSnapshot snapshot = *found->second;
    for (auto &row : snapshot.active) if (row.second.stack)
        row.second.stack = std::make_shared<RequestDiagnosticStack>(*row.second.stack);
    for (auto &row : snapshot.recent) if (row.stack)
        row.stack = std::make_shared<RequestDiagnosticStack>(*row.stack);
    lock.unlock();
    fields["native_diag_state"] = snapshot.closed ? "closed" : "ok";
    fields["native_diag_generation"] = std::to_string(snapshot.generation);
    fields["native_diag_revision"] = std::to_string(snapshot.revision);
    fields["native_diag_progress_revision"] = std::to_string(snapshot.progressRevision);
    fields["native_diag_progress_end_ns"] = std::to_string(snapshot.progressEndNs);
    fields["native_diag_observer_detached"] = snapshot.detached ? "1" : "0";
    fields["native_diag_active_count"] = std::to_string(snapshot.active.size());
    fields["native_diag_completed_count"] = std::to_string(snapshot.retired);
    fields["native_diag_dropped_active_count"] = std::to_string(snapshot.droppedActive);
    std::string rows = "["; size_t count = 0; bool truncated = false;
    const uint64_t now = RequestDiagnosticNowNs();
    auto append = [&](const RequestDiagnosticRow &row) {
        if (count >= 16) { truncated = true; return; }
        auto encoded = RequestDiagnosticRowJson(row, now);
        if (rows.size() + encoded.size() > 56000) { truncated = true; return; }
        if (count) { rows += ','; }
        rows += encoded; ++count;
    };
    for (const auto &row : snapshot.active) append(row.second);
    for (auto row = snapshot.recent.rbegin(); row != snapshot.recent.rend(); ++row) append(*row);
    rows += ']'; fields["native_diag_operations_json"] = std::move(rows);
    fields["native_diag_operations_truncated"] = truncated ? "1" : "0";
    fields["native_diag_stack_semantics"] = "bounded_first_entry_backtrace_not_current_stack";
    return true;
}
} // namespace fastllm
