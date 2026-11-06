#pragma once

// Opt-in serving diagnostics. No model data or generated text is recorded.
// Callers must gate construction with Enabled()/CurrentStep() so the disabled
// path does not allocate records, read clocks, or submit CUDA timing events.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <limits>
#include <ctime>
#include <iomanip>
#include <locale>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
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
    // Trace handles and token lengths are decimal ints. Keep the stream's
    // existing formatting for other types, including characters and booleans.
    if constexpr (std::is_same<T, int>::value) {
        char buffer[4096];
        char *end = buffer;
        *end++ = '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (buffer + sizeof(buffer) - end < 32) {
                out.write(buffer, end - buffer);
                end = buffer;
            }
            if (i) *end++ = ',';
            end = std::to_chars(end, buffer + sizeof(buffer), values[i]).ptr;
        }
        *end++ = ']';
        out.write(buffer, end - buffer);
        return;
    }
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

// Sampled host diagnostics use only the scheduler thread. Nested spans are
// inclusive wall times (not additive); region CPU time helps identify host
// descheduling / blocking. No CUDA calls or synchronization are introduced.
inline unsigned DetailInterval() {
    static const unsigned interval = [] {
        const char *value = std::getenv("FASTLLM_INFERENCE_TRACE_DETAIL_EVERY");
        if (!value || !*value) return 0u;
        unsigned parsed = 0;
        auto result = std::from_chars(value, value + std::strlen(value), parsed);
        return result.ec == std::errc() && *result.ptr == '\0' ? parsed : 0u;
    }();
    return interval;
}
inline double ThreadCpuMilliseconds() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
    return -1;
}
struct HostDetail {
    struct Metric {
        const char *region, *name;
        double wallMs = 0, cpuMs = 0;
        unsigned long long calls = 0, count = 0;
        bool cpuAvailable = false;
    };
    // Fixed storage keeps diagnostic probes from allocating per request/page.
    Metric metrics[128];
    size_t used = 0;
    unsigned long long dropped = 0;
    const char *region = "other";
    Metric *Get(const char *name) {
        for (size_t i = 0; i < used; ++i)
            if (!std::strcmp(metrics[i].region, region) && !std::strcmp(metrics[i].name, name))
                return &metrics[i];
        if (used == 128) { ++dropped; return nullptr; }
        auto *metric = &metrics[used++];
        metric->region = region; metric->name = name;
        return metric;
    }
    void Count(const char *name, unsigned long long value = 1) {
        if (auto *metric = Get(name)) metric->count += value;
    }
    void Write(std::ostream &out) const {
        out << ",\"host_detail\":{\"sample_every\":" << DetailInterval()
            << ",\"dropped_metrics\":" << dropped << ",\"metrics\":[";
        for (size_t i = 0; i < used; ++i) {
            if (i) out << ',';
            const auto &m = metrics[i];
            out << "{\"region\":\"" << m.region << "\",\"name\":\"" << m.name
                << "\",\"wall_ms\":" << m.wallMs << ",\"calls\":" << m.calls
                << ",\"count\":" << m.count;
            if (m.cpuAvailable) out << ",\"thread_cpu_ms\":" << m.cpuMs;
            out << '}';
        }
        out << "]}";
    }
};
inline HostDetail *&CurrentDetail() { static thread_local HostDetail *value = nullptr; return value; }
class DetailSpan {
    HostDetail *detail;
    HostDetail::Metric *metric = nullptr;
    Clock::time_point start;
public:
    explicit DetailSpan(const char *name, HostDetail *record = CurrentDetail()) : detail(record) {
        if (detail) { metric = detail->Get(name); start = Clock::now(); }
    }
    DetailSpan(const DetailSpan &) = delete;
    DetailSpan &operator=(const DetailSpan &) = delete;
    void Stop() {
        if (metric) { metric->wallMs += Milliseconds(start, Clock::now()); ++metric->calls; metric = nullptr; }
    }
    void Next(const char *name) {
        if (!detail) return;
        auto now = Clock::now();
        if (metric) { metric->wallMs += Milliseconds(start, now); ++metric->calls; }
        metric = detail->Get(name); start = now;
    }
    ~DetailSpan() { Stop(); }
};
class DetailRegion {
    HostDetail *detail = CurrentDetail();
    const char *previous = nullptr;
    HostDetail::Metric *metric = nullptr;
    Clock::time_point start;
    double cpuStart = -1;
public:
    explicit DetailRegion(const char *name) {
        if (!detail) return;
        previous = detail->region; detail->region = name;
        metric = detail->Get("total");
        cpuStart = ThreadCpuMilliseconds(); start = Clock::now();
    }
    DetailRegion(const DetailRegion &) = delete;
    DetailRegion &operator=(const DetailRegion &) = delete;
    void Stop() {
        if (!detail) return;
        auto end = Clock::now();
        double cpuEnd = ThreadCpuMilliseconds();
        if (metric) {
            metric->wallMs += Milliseconds(start, end); ++metric->calls;
            if (cpuStart >= 0 && cpuEnd >= cpuStart) {
                metric->cpuAvailable = true; metric->cpuMs += cpuEnd - cpuStart;
            }
        }
        detail->region = previous; detail = nullptr;
    }
    ~DetailRegion() { Stop(); }
};

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
    HostDetail *previousDetail;
    std::unique_ptr<HostDetail> detail;
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
    explicit Step() : previous(CurrentStep()), previousDetail(CurrentDetail()),
        id(++Output().stepIds), timestampNs(UnixNanoseconds()) {
        if (DetailInterval() && id % DetailInterval() == 0) detail.reset(new HostDetail());
        CurrentDetail() = detail.get();
        CurrentStep() = this;
    }
    ~Step() {
        CurrentStep() = previous;
        CurrentDetail() = previousDetail;
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
            if (detail) detail->Write(out);
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
