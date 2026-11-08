#pragma once
// Opt-in, rate-limited native checkpoint tracer.
// Disabled unless FASTLLM_TRACE_CHECKPOINTS=1, so default behaviour is unchanged.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sys/syscall.h>

namespace fastllm {

inline bool CheckpointTraceEnabled() {
    static const bool enabled = []() {
        const char *v = getenv("FASTLLM_TRACE_CHECKPOINTS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

inline long long CheckpointNowNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

inline void CheckpointTrace(const char *label, long long a = 0, long long b = 0) {
    if (!CheckpointTraceEnabled()) {
        return;
    }
    static thread_local long long lastNs = 0;
    long long now = CheckpointNowNs();
    // A/B profiling needs the complete per-chunk sequence; the default stays
    // rate limited so production logging is unchanged.
    const bool unlimited = getenv("FASTLLM_PROFILE_RECORD") != nullptr;
    if (!unlimited && now - lastNs < 100000000LL) {  // at most ~10 lines/s per thread
        return;
    }
    lastNs = now;
    fprintf(stderr, "[ckpt] tid=%ld label=%s a=%lld b=%lld mono_ns=%lld\n",
            (long)syscall(SYS_gettid), label, a, b, now);
    fflush(stderr);
}

}  // namespace fastllm
