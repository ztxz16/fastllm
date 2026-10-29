#pragma once

// Opt-in serving diagnostics. No model data or generated text is recorded.
// Callers must gate construction with Enabled()/CurrentStep() so the disabled
// path does not allocate records, read clocks, or submit CUDA timing events.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <locale>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fastllm { namespace inference_trace {

using Clock = std::chrono::steady_clock;
inline double Milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
inline long long UnixNanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
inline int ProcessId() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}
inline double Rate(long long tokens, double milliseconds) {
    return milliseconds > 0 ? tokens * 1000.0 / milliseconds : 0;
}
template <typename T> inline void Array(std::ostream &out, const std::vector<T> &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

class Sink {
    FILE *file = nullptr;
    std::mutex mutex;
    Clock::time_point lastFlush;
    std::atomic<bool> healthy{false};
public:
    std::atomic<unsigned long long> stepIds{0}, forwardIds{0};
    Sink() {
        const char *path = std::getenv("FASTLLM_INFERENCE_TRACE");
        if (!path || !*path) return;
        file = std::fopen(path, "a");
        if (!file) {
            std::fprintf(stderr, "[Fastllm] Cannot open inference trace: %s\n", path);
            return;
        }
        std::setvbuf(file, nullptr, _IOFBF, 1 << 20);
        lastFlush = Clock::now();
        healthy.store(true);
        std::fprintf(stderr, "[Fastllm] Inference trace enabled: %s\n", path);
    }
    ~Sink() { if (file) std::fclose(file); }
    bool Enabled() const { return healthy.load(std::memory_order_relaxed); }
    void Emit(const std::string &line) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!Enabled()) return;
        bool ok = std::fwrite(line.data(), 1, line.size(), file) == line.size();
        auto now = Clock::now();
        if (now - lastFlush >= std::chrono::seconds(1)) {
            ok = std::fflush(file) == 0 && ok;
            lastFlush = now;
        }
        if (!ok) {
            healthy.store(false, std::memory_order_relaxed);
            std::fprintf(stderr, "[Fastllm] Inference trace write failed; logging disabled.\n");
        }
    }
};
inline Sink &Output() { static Sink sink; return sink; }
inline bool Enabled() { return Output().Enabled(); }

class Step;
class Forward;
inline Step *&CurrentStep() { static thread_local Step *value = nullptr; return value; }
inline Forward *&CurrentForward() { static thread_local Forward *value = nullptr; return value; }
inline double &PreviousStepEmitMs() { static thread_local double value = 0; return value; }

class Stages {
protected:
    Clock::time_point start = Clock::now(), mark = start;
    std::vector<std::pair<const char *, double>> stages;
    void WriteStages(std::ostream &out) const {
        out << "\"timings_ms\":{";
        for (size_t i = 0; i < stages.size(); ++i) {
            if (i) out << ',';
            out << '"' << stages[i].first << "\":" << stages[i].second;
        }
        out << '}';
    }
public:
    void Mark(const char *name) {
        auto now = Clock::now();
        stages.emplace_back(name, Milliseconds(mark, now));
        mark = now;
    }
};

class Step : public Stages {
    Step *previous;
public:
    unsigned long long id;
    const char *phase = "unknown";
    const char *prefillDecision = nullptr;
    int prefillCandidates = 0, activeDecode = 0;
    long long prefillCandidateTokens = 0;
    std::vector<int> handles, lengths, contextBefore, cachedPrefix;
    long long timestampNs;
    double forwardMs = 0, forwardEmitMs = 0;
    int forwards = 0;
    bool submitted = false, failed = false;
    explicit Step() : previous(CurrentStep()), id(++Output().stepIds), timestampNs(UnixNanoseconds()) {
        CurrentStep() = this;
    }
    ~Step() {
        CurrentStep() = previous;
        if (!submitted) return;
        try {
            auto end = Clock::now();
            double elapsed = Milliseconds(start, end);
            long long tokens = std::accumulate(lengths.begin(), lengths.end(), 0LL);
            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << std::fixed << std::setprecision(6)
                << "{\"schema_version\":1,\"event\":\"scheduler_step\",\"pid\":" << ProcessId()
                << ",\"step_id\":" << id << ",\"timestamp_ns\":" << timestampNs
                << ",\"phase\":\"" << phase << "\",\"batch\":" << lengths.size()
                << ",\"completed\":" << (failed ? "false" : "true")
                << ",\"input_tokens\":" << tokens << ",\"handles\":";
            Array(out, handles);
            out << ",\"query_lengths\":"; Array(out, lengths);
            out << ",\"context_before\":"; Array(out, contextBefore);
            out << ",\"cached_prefix_tokens\":"; Array(out, cachedPrefix);
            out << ",\"total_ms\":" << elapsed << ",\"forward_api_ms\":" << forwardMs
                << ",\"outside_forward_ms\":" << std::max(0.0, elapsed - forwardMs)
                << ",\"forward_count\":" << forwards
                << ",\"tail_ms\":" << Milliseconds(mark, end)
                << ",\"step_tokens_per_second\":" << Rate(tokens, elapsed)
                << ",\"forward_trace_emit_ms\":" << forwardEmitMs
                << ",\"previous_step_trace_emit_ms\":" << PreviousStepEmitMs() << ',';
            if (prefillDecision)
                out << "\"prefill_coalescing\":{\"reason\":\"" << prefillDecision
                    << "\",\"candidates\":" << prefillCandidates
                    << ",\"new_tokens\":" << prefillCandidateTokens
                    << ",\"active_decode\":" << activeDecode << "},";
            WriteStages(out);
            out << "}\n";
            Output().Emit(out.str());
            PreviousStepEmitMs() = Milliseconds(end, Clock::now());
        } catch (...) {
            std::fprintf(stderr, "[Fastllm] Failed to serialize inference step trace.\n");
        }
    }
};

struct RankTiming {
    int device = -1;
    double modelGpuMs = 0, gatherGpuMs = 0, wallMs = 0, observerWaitMs = 0;
};

class Forward : public Stages {
    Step &step;
    Forward *previous;
    std::vector<int> lengths, contextBefore;
    long long timestampNs;
    int exceptionCount = std::uncaught_exceptions();
public:
    unsigned long long id;
    const char *mode = "eager";
    std::vector<RankTiming> ranks;
    Forward(Step &parent, std::vector<int> queryLengths, std::vector<int> contexts)
        : step(parent), previous(CurrentForward()), lengths(std::move(queryLengths)),
          contextBefore(std::move(contexts)), timestampNs(UnixNanoseconds()), id(++Output().forwardIds) {
        CurrentForward() = this;
        step.submitted = true;
    }
    ~Forward() {
        CurrentForward() = previous;
        auto end = Clock::now();
        double elapsed = Milliseconds(start, end);
        step.forwardMs += elapsed;
        ++step.forwards;
        try {
            long long tokens = std::accumulate(lengths.begin(), lengths.end(), 0LL);
            std::vector<int> contexts = contextBefore;
            for (size_t i = 0; i < contexts.size(); ++i)
                if (contexts[i] >= 0) contexts[i] += lengths[i];
            long long contextSum = std::accumulate(contexts.begin(), contexts.end(), 0LL);
            int contextMin = contexts.empty() ? 0 : *std::min_element(contexts.begin(), contexts.end());
            int contextMax = contexts.empty() ? 0 : *std::max_element(contexts.begin(), contexts.end());
            double gpuMax = 0;
            for (const auto &rank : ranks) gpuMax = std::max(gpuMax, rank.modelGpuMs);
            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << std::fixed << std::setprecision(6)
                << "{\"schema_version\":1,\"event\":\"forward\",\"pid\":" << ProcessId()
                << ",\"step_id\":" << step.id << ",\"forward_id\":" << id
                << ",\"timestamp_ns\":" << timestampNs << ",\"phase\":\"" << step.phase
                << "\",\"mode\":\"" << mode << "\",\"batch\":" << lengths.size()
                << ",\"completed\":" << (std::uncaught_exceptions() == exceptionCount ? "true" : "false")
                << ",\"input_tokens\":" << tokens << ",\"handles\":";
            Array(out, step.handles);
            out << ",\"query_lengths\":"; Array(out, lengths);
            out << ",\"context_before\":"; Array(out, contextBefore);
            out << ",\"context_after\":"; Array(out, contexts);
            out << ",\"context_min\":" << contextMin << ",\"context_max\":" << contextMax
                << ",\"context_sum\":" << contextSum
                << ",\"forward_ms\":" << elapsed << ",\"tokens_per_second\":" << Rate(tokens, elapsed)
                << ",\"gpu_model_max_ms\":" << gpuMax
                << ",\"gpu_model_tokens_per_second\":" << Rate(tokens, gpuMax)
                << ",\"tail_ms\":" << Milliseconds(mark, end) << ',';
            WriteStages(out);
            out << ",\"ranks\":[";
            for (size_t i = 0; i < ranks.size(); ++i) {
                if (i) out << ',';
                const auto &r = ranks[i];
                out << "{\"device\":" << r.device << ",\"model_gpu_ms\":" << r.modelGpuMs
                    << ",\"logits_gather_gpu_ms\":" << r.gatherGpuMs << ",\"wall_ms\":" << r.wallMs
                    << ",\"observer_wait_ms\":" << r.observerWaitMs << '}';
            }
            out << "]}\n";
            Output().Emit(out.str());
        } catch (...) {
            std::fprintf(stderr, "[Fastllm] Failed to serialize inference forward trace.\n");
        }
        step.forwardEmitMs += Milliseconds(end, Clock::now());
    }
};

}} // namespace fastllm::inference_trace
