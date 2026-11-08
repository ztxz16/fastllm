// CPU-only probe: g++ -std=c++17 -pthread -Iinclude this_file.cpp -o <build>/probe
// Run in separate processes with FASTLLM_OBSERVABILITY_MODE=off/user/debug.
#include "utils/prefix_cache_logging.h"
#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

bool ContainsLine(const std::string &text, const std::string &first,
                  const std::string &second, const std::string &third = "") {
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        const auto line = text.substr(start, end - start);
        if (line.find(first) != std::string::npos && line.find(second) != std::string::npos &&
            line.find(third) != std::string::npos) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

void RequireSnapshotCutoff(const std::string &text, uint64_t handle, uint64_t cutoff) {
    size_t start = 0;
    const std::string handleField = "native_handle=" + std::to_string(handle) + " ";
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        const auto line = text.substr(start, end - start);
        if (line.find("event=phase_status ") != std::string::npos &&
            line.find(handleField) != std::string::npos) {
            const auto timestamp = line.find("observed_monotonic_ns=");
            Require(timestamp != std::string::npos && std::stoull(line.substr(timestamp + 22)) <= cutoff,
                    "observer scheduled a new snapshot after finish");
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
}
}

int main() {
    int channel[2];
    Require(pipe(channel) == 0, "stdout capture pipe failed");
    const int saved = dup(STDOUT_FILENO);
    Require(saved >= 0 && dup2(channel[1], STDOUT_FILENO) >= 0, "stdout capture failed");
    close(channel[1]);
    std::string output;
    std::thread reader([&] {
        char buffer[4096];
        for (;;) {
            const auto n = read(channel[0], buffer, sizeof(buffer));
            if (n > 0) output.append(buffer, static_cast<size_t>(n));
            else if (n < 0 && errno == EINTR) continue;
            else break;
        }
        close(channel[0]);
    });

    using namespace fastllm;
    StartFastllmRequestPhase(199, 128);
    SetFastllmRequestPhase(199, "prefill", "unregistered_warmup", 64, 128);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    FinishFastllmRequestPhase(199);
    FastllmEvent("probe", 0, "unregistered_finished");
    StartFastllmRequestPhase(101, 4096);
    SetFastllmRequestPhase(101, "prefill", "cpu_probe", 128, 4096);
    RegisterFastllmPublicRequestId(101, "public-A");
    StartFastllmRequestPhase(102, 8192);
    RegisterFastllmPublicRequestId(102, "public-B");
    SetFastllmRequestPhase(102, "cache_lookup", "catalog", -1, 8192);
    // Both requests remain in an unchanged operation across the real 1s tick.
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    SetFastllmRequestPhase(101, "prefill", "cpu_probe", 128, 4096);
    std::this_thread::sleep_for(std::chrono::milliseconds(550));
    UnregisterFastllmPublicRequestId(102);
    const auto canceled102 = FastllmObservationTimestampNs();
    FastllmEvent("probe", 102, "cancel_done");
    FinishFastllmRequestPhase(101);
    FastllmEvent("probe", 101, "finish_done");

    // Recycled handle advances before public-id registration: attaching the
    // new ID must neither reset that phase nor reuse public-A.
    StartFastllmRequestPhase(101);
    SetFastllmRequestPhase(101, "vision_preprocess", "images", -1, 4096);
    RegisterFastllmPublicRequestId(101, "public-C");
    UnregisterFastllmPublicRequestId(101, "public-A");
    Require(FastllmPublicRequestId(101) == "public-C", "stale unregister erased recycled handle identity");
    StartFastllmRequestPhase(103, 2048);
    RegisterFastllmPublicRequestId(103, "public-D");
    std::thread updates([] {
        for (int i = 0; i < 64; ++i) {
            SetFastllmRequestPhase(103, "prefill", "chunks", i * 16, 2048);
        }
    });
    updates.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));
    UnregisterFastllmPublicRequestId(101);
    FinishFastllmRequestPhase(103);
    UnregisterFastllmPublicRequestId(103);
    const auto finishedAll = FastllmObservationTimestampNs();
    FastllmEvent("probe", 0, "all_finished");
    // No new snapshot is scheduled after finish; a write already in flight may
    // finish with its original copied public ID after finish returns.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    double blockedLifecycleMs = 0;
    if (GetFastllmObservabilityMode() != FastllmObservabilityMode::Off) {
        FastllmEvent("probe", 0, "backpressure_begin");
        // Saturate stdout before the next heartbeat. A delayed drainer recovers
        // even a broken implementation, so a regression fails instead of hangs.
        int blocked[2];
        Require(pipe(blocked) == 0, "blocked stdout pipe failed");
        const int originalFlags = fcntl(blocked[1], F_GETFL);
        Require(originalFlags >= 0 && fcntl(blocked[1], F_SETFL, originalFlags | O_NONBLOCK) == 0,
                "nonblocking pipe setup failed");
        std::string filler(4096, 'x');
        for (;;) {
            const auto n = write(blocked[1], filler.data(), filler.size());
            if (n >= 0 || (n < 0 && errno == EINTR)) continue;
            Require(errno == EAGAIN || errno == EWOULDBLOCK, "pipe saturation failed");
            break;
        }
        Require(fcntl(blocked[1], F_SETFL, originalFlags) == 0, "blocking pipe restore failed");
        const int previous = dup(STDOUT_FILENO);
        Require(previous >= 0 && dup2(blocked[1], STDOUT_FILENO) >= 0, "blocked stdout redirect failed");
        close(blocked[1]);
        StartFastllmRequestPhase(104, 256);
        RegisterFastllmPublicRequestId(104, "public-blocked");
        SetFastllmRequestPhase(104, "ssd_read", "backpressure_probe");
        std::this_thread::sleep_for(std::chrono::milliseconds(1150));
        const bool outputAvailable = FastllmObservationEventMutex().try_lock();
        if (outputAvailable) FastllmObservationEventMutex().unlock();
        std::thread drainer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            char buffer[4096];
            for (;;) {
                const auto n = read(blocked[0], buffer, sizeof(buffer));
                if (n > 0 || (n < 0 && errno == EINTR)) continue;
                break;
            }
            close(blocked[0]);
        });
        const auto before = std::chrono::steady_clock::now();
        SetFastllmRequestPhase(104, "ssd_import", "backpressure_probe");
        FinishFastllmRequestPhase(104);
        UnregisterFastllmPublicRequestId(104);
        blockedLifecycleMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - before).count();
        Require(dup2(previous, STDOUT_FILENO) >= 0, "stdout backpressure restore failed");
        close(previous);
        drainer.join();
        Require(!outputAvailable, "heartbeat did not encounter saturated stdout");
        Require(blockedLifecycleMs < 200.0, "model lifecycle waited for blocked stdout");
    }

    std::fflush(stdout);
    Require(dup2(saved, STDOUT_FILENO) >= 0, "stdout restore failed");
    close(saved);
    reader.join();
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) {
        Require(output.empty(), "off mode emitted output");
    } else {
        const auto unregistered = output.find("event=unregistered_finished");
        Require(unregistered != std::string::npos && output.substr(0, unregistered).find("event=phase_status") == std::string::npos,
                "unregistered warmup emitted heartbeat");
        Require(ContainsLine(output, "event=phase_status", "public_request=public-A",
                             "phase=prefill detail=cpu_probe"), "missing blocked prefill heartbeat");
        Require(ContainsLine(output, "public_request=public-A", "completed_tokens=128 total_tokens=4096"),
                "prefill counters changed without an explicit update");
        const auto firstA = output.find("public_request=public-A");
        const auto firstIdle = output.find("idle_ms=", firstA);
        Require(firstIdle != std::string::npos && std::stod(output.substr(firstIdle + 8)) >= 900.0,
                "unchanged labels incorrectly reset progress idle");
        Require(ContainsLine(output, "event=phase_status", "public_request=public-B",
                             "phase=cache_lookup detail=catalog"), "missing concurrent lookup heartbeat");
        Require(ContainsLine(output, "event=phase_status", "public_request=public-C",
                             "phase=vision_preprocess detail=images"), "registration reset advanced phase");
        Require(ContainsLine(output, "public_request=public-C", "completed_tokens=unknown total_tokens=4096"),
                "phase transition retained stale counters");
        Require(!ContainsLine(output, "public_request=public-A", "phase=vision_preprocess"),
                "recycled handle emitted stale public id");
        Require(ContainsLine(output, "public_request=public-D", "completed_tokens=1008 total_tokens=2048"),
                "concurrent updates lost true progress");
        RequireSnapshotCutoff(output, 102, canceled102);
        RequireSnapshotCutoff(output, 101, finishedAll);
        RequireSnapshotCutoff(output, 103, finishedAll);
    }
    std::cout << output;
    std::cerr << "request_phase_observer_probe: PASS blocked_lifecycle_ms=" << blockedLifecycleMs << "\n";
    return 0;
}
