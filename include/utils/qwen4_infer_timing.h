#pragma once

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"

namespace fastllm::qwen4_infer_timing {
inline bool Enabled() {
    static bool enabled = [] {
        const char *value = std::getenv("FASTLLM_QWEN4_INFER_TIMING");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
inline double Now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Sample {
    void *begin = nullptr;
    void *end = nullptr;
    std::string name;
    double host_begin = 0;
    double host_ms = 0;
};
inline thread_local std::vector<Sample> samples;
inline thread_local size_t cursor = 0;
inline thread_local bool draft = false;

class Scope {
    size_t index = 0;
    bool active = false;
public:
    explicit Scope(const char *name) : active(Enabled()) {
        if (!active || FastllmCudaGraphIsCapturingFast()) {
            active = false; return;
        }
        index = cursor++;
        if (samples.size() <= index) {
            Sample sample;
            sample.begin = FastllmCudaEventCreateTiming();
            sample.end = FastllmCudaEventCreateTiming();
            samples.push_back(std::move(sample));
        }
        Sample &sample = samples[index];
        sample.name = std::string(draft ? "draft." : "target.") + name;
        sample.host_begin = Now();
        FastllmCudaEventRecordCurrentThread(sample.begin);
    }
    ~Scope() {
        if (!active) return;
        Sample &sample = samples[index];
        FastllmCudaEventRecordCurrentThread(sample.end);
        sample.host_ms = (Now() - sample.host_begin) * 1000;
    }
};
class DraftGuard {
    bool previous;
public:
    DraftGuard() : previous(draft) { draft = true; }
    ~DraftGuard() { draft = previous; }
};
// Caller has already synchronized the ordinary stage completion stream.
// Event spans include device work and CPU submission gaps, not pure SM time.
inline void Emit(int stage, int rank, int rows) {
    if (!Enabled()) return;
    struct Total { int calls = 0; double host = 0; double stream = 0; };
    std::map<std::string, Total> totals;
    for (size_t i = 0; i < cursor; ++i) {
        const Sample &sample = samples[i];
        auto &total = totals[sample.name];
        ++total.calls;
        total.host += sample.host_ms;
        total.stream += FastllmCudaEventElapsedTime(sample.begin, sample.end);
    }
    for (const auto &item : totals) {
        std::printf("[INFER TIMING] stage=%d rank=%d rows=%d op=%s calls=%d "
                    "host_ms=%.4f stream_ms=%.4f\n", stage, rank, rows,
                    item.first.c_str(), item.second.calls,
                    item.second.host, item.second.stream);
    }
    cursor = 0;
    std::fflush(stdout);
}
}
#else
namespace fastllm::qwen4_infer_timing {
struct Scope { explicit Scope(const char *) {} };
struct DraftGuard {};
inline void Emit(int, int, int) {}
}
#endif
