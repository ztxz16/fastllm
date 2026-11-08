#pragma once

#include "prefix_cache_logging.h"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <thread>
#include <vector>

namespace fastllm {

// This observer owns copied metadata only, never a model/context pointer. A
// stalled native operation cannot prevent its clock from reporting that phase.
// No thread starts in off mode, or before a real public request is registered.
class FastllmRequestPhaseObserver {
    using Clock = std::chrono::steady_clock;
    struct Phase {
        char name[64] = "queued";
        char detail[128] = "scheduler_wait";
        int completedTokens = -1;
        int totalTokens = -1;
        bool registered = false;
        uint64_t generation = 0;
        std::string publicId;
        Clock::time_point started;
        Clock::time_point updated;
        Clock::time_point nextEmission;
    };
    struct Emission {
        uint64_t handle;
        Phase phase;
        Clock::time_point now;
    };

    // No lifecycle/state lock is held while writing stdout. All model-side
    // calls remain independent of output backpressure. Finish/unregister stop
    // future snapshots; an already in-flight snapshot may finish afterward,
    // retaining its copied original identity and observation timestamp.
    std::mutex stateMutex;
    std::condition_variable changed;
    std::map<uint64_t, Phase> phases;
    std::thread worker;
    bool stopping = false;
    uint64_t nextGeneration = 0;

    bool HasRegisteredPhase() const {
        for (const auto &item : phases) if (item.second.registered) return true;
        return false;
    }

    static void CopyField(char *target, size_t size, const char *source) {
        if (source == nullptr || source[0] == '\0') source = "unspecified";
        size_t i = 0;
        for (; i + 1 < size && source[i] != '\0'; ++i) {
            const unsigned char c = static_cast<unsigned char>(source[i]);
            target[i] = std::isspace(c) ? '_' : static_cast<char>(c);
        }
        target[i] = '\0';
    }

    void Run() {
        std::unique_lock<std::mutex> state(stateMutex);
        while (!stopping) {
            if (!HasRegisteredPhase()) {
                changed.wait(state, [this] { return stopping || HasRegisteredPhase(); });
                continue;
            }
            auto deadline = Clock::time_point::max();
            for (const auto &item : phases) {
                if (item.second.registered) deadline = std::min(deadline, item.second.nextEmission);
            }
            if (Clock::now() < deadline) {
                changed.wait_until(state, deadline);
                continue;
            }
            std::vector<Emission> due;
            const auto now = Clock::now();
            for (auto &item : phases) {
                if (!item.second.registered || item.second.nextEmission > now) continue;
                due.push_back({item.first, item.second, now});
                // Do not fabricate catch-up events after a delayed write.
                item.second.nextEmission = now + std::chrono::seconds(1);
            }
            state.unlock();
            for (const auto &item : due) {
                // Drop snapshots canceled/recycled before emission starts.
                // Finish never waits for a write that has already started.
                bool active;
                {
                    std::lock_guard<std::mutex> guard(stateMutex);
                    const auto current = phases.find(item.handle);
                    active = !stopping && current != phases.end() &&
                        current->second.generation == item.phase.generation &&
                        current->second.publicId == item.phase.publicId;
                }
                if (!active) continue;
                const double elapsed = std::chrono::duration<double, std::milli>(
                    item.now - item.phase.started).count();
                const double idle = std::chrono::duration<double, std::milli>(
                    item.now - item.phase.updated).count();
                char completed[32] = "unknown", total[32] = "unknown";
                if (item.phase.completedTokens >= 0) std::snprintf(completed, sizeof(completed), "%d", item.phase.completedTokens);
                if (item.phase.totalTokens >= 0) std::snprintf(total, sizeof(total), "%d", item.phase.totalTokens);
                const auto observedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    item.now.time_since_epoch()).count();
                FastllmEventWithPublicId("progress", item.handle, item.phase.publicId, "phase_status",
                    "phase=%s detail=%s phase_elapsed_ms=%.3f idle_ms=%.3f "
                    "completed_tokens=%s total_tokens=%s phase_generation=%llu observed_monotonic_ns=%llu",
                    item.phase.name, item.phase.detail, elapsed, idle,
                    completed, total, static_cast<unsigned long long>(item.phase.generation),
                    static_cast<unsigned long long>(observedNs));
            }
            state.lock();
        }
    }

public:
    FastllmRequestPhaseObserver() = default;
    FastllmRequestPhaseObserver(const FastllmRequestPhaseObserver &) = delete;
    FastllmRequestPhaseObserver &operator=(const FastllmRequestPhaseObserver &) = delete;

    ~FastllmRequestPhaseObserver() {
        {
            std::lock_guard<std::mutex> state(stateMutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }

    void Start(uint64_t handle, int totalTokens) {
        std::lock_guard<std::mutex> state(stateMutex);
        // Native allocation precedes public-id registration. Drop any identity
        // from a recycled handle before it can appear on the new phase stream.
        {
            std::lock_guard<std::mutex> ids(FastllmPublicRequestIdsMutex());
            FastllmPublicRequestIds().erase(handle);
        }
        Phase phase;
        phase.generation = ++nextGeneration;
        phase.totalTokens = totalTokens;
        phase.started = phase.updated = Clock::now();
        phase.nextEmission = phase.started + std::chrono::seconds(1);
        phases[handle] = phase;
        changed.notify_all();
    }

    void Set(uint64_t handle, const char *name, const char *detail,
             int completedTokens, int totalTokens) {
        std::lock_guard<std::mutex> state(stateMutex);
        const auto found = phases.find(handle);
        if (found == phases.end()) return;
        auto &phase = found->second;
        char nextName[sizeof(phase.name)];
        char nextDetail[sizeof(phase.detail)];
        CopyField(nextName, sizeof(nextName), name);
        CopyField(nextDetail, sizeof(nextDetail), detail);
        const bool newPhase = std::strcmp(nextName, phase.name) != 0;
        const bool stepChanged = newPhase || std::strcmp(nextDetail, phase.detail) != 0 ||
            (completedTokens >= 0 && completedTokens != phase.completedTokens);
        const auto now = Clock::now();
        if (newPhase) {
            phase.started = now;
            phase.completedTokens = phase.totalTokens = -1;
        }
        // Repeating an unchanged label is not evidence that work progressed.
        if (stepChanged) phase.updated = now;
        std::strcpy(phase.name, nextName);
        std::strcpy(phase.detail, nextDetail);
        if (completedTokens >= 0) phase.completedTokens = completedTokens;
        if (totalTokens >= 0) phase.totalTokens = totalTokens;
    }

    void Finish(uint64_t handle) {
        {
            std::lock_guard<std::mutex> state(stateMutex);
            phases.erase(handle);
        }
        changed.notify_all();
    }

    void Register(uint64_t handle, const char *publicId) {
        std::lock_guard<std::mutex> state(stateMutex);
        {
            std::lock_guard<std::mutex> ids(FastllmPublicRequestIdsMutex());
            FastllmPublicRequestIds()[handle] = publicId;
        }
        const auto found = phases.find(handle);
        if (found != phases.end()) {
            found->second.publicId = publicId;
            if (!found->second.registered) {
                found->second.registered = true;
                found->second.nextEmission = Clock::now() + std::chrono::seconds(1);
            }
            if (!worker.joinable()) worker = std::thread([this] { Run(); });
        }
        changed.notify_all();
        // Do not reset the phase: native work may already have advanced.
    }

    void Unregister(uint64_t handle, const char *expectedPublicId) {
        std::lock_guard<std::mutex> state(stateMutex);
        {
            std::lock_guard<std::mutex> ids(FastllmPublicRequestIdsMutex());
            if (expectedPublicId != nullptr) {
                const auto current = FastllmPublicRequestIds().find(handle);
                if (current == FastllmPublicRequestIds().end() || current->second != expectedPublicId) return;
            }
            FastllmPublicRequestIds().erase(handle);
        }
        phases.erase(handle);
        changed.notify_all();
    }
};

inline FastllmRequestPhaseObserver &FastllmRequestPhases() {
    // Construct dependencies first so they outlive the joined observer thread.
    (void)FastllmPublicRequestIdsMutex();
    (void)FastllmPublicRequestIds();
    (void)FastllmObservationSequence();
    (void)FastllmObservationEventMutex();
    static FastllmRequestPhaseObserver observer;
    return observer;
}

inline void StartFastllmRequestPhase(uint64_t nativeHandle, int totalTokens = -1) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) return;
    FastllmRequestPhases().Start(nativeHandle, totalTokens);
}

inline void SetFastllmRequestPhase(uint64_t nativeHandle, const char *phase,
                                  const char *detail, int completedTokens = -1,
                                  int totalTokens = -1) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) return;
    FastllmRequestPhases().Set(nativeHandle, phase, detail, completedTokens, totalTokens);
}

inline void FinishFastllmRequestPhase(uint64_t nativeHandle) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) return;
    FastllmRequestPhases().Finish(nativeHandle);
}

inline void RegisterFastllmObservedPublicRequestId(uint64_t nativeHandle, const char *publicId) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) {
        std::lock_guard<std::mutex> ids(FastllmPublicRequestIdsMutex());
        FastllmPublicRequestIds()[nativeHandle] = publicId;
        return;
    }
    FastllmRequestPhases().Register(nativeHandle, publicId);
}

inline void UnregisterFastllmObservedPublicRequestId(uint64_t nativeHandle, const char *expectedPublicId) {
    if (GetFastllmObservabilityMode() == FastllmObservabilityMode::Off) {
        std::lock_guard<std::mutex> ids(FastllmPublicRequestIdsMutex());
        if (expectedPublicId != nullptr) {
            const auto current = FastllmPublicRequestIds().find(nativeHandle);
            if (current == FastllmPublicRequestIds().end() || current->second != expectedPublicId) return;
        }
        FastllmPublicRequestIds().erase(nativeHandle);
        return;
    }
    FastllmRequestPhases().Unregister(nativeHandle, expectedPublicId);
}

} // namespace fastllm
