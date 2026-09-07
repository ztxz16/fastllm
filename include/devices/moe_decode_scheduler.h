#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace fastllm {

// Timing and admission policy only: independent of weight encoding, memory
// placement and kernels. A backend reports measured CPU/compute/refill costs
// and executes the returned GPU count (resident experts first).
class MoeDecodeScheduler {
public:
    static constexpr int maxExperts = 16;
    struct Estimate {
        double us = 0;
        bool initialized = false;
        void Observe(double sample) {
            // Discard cold-start inflation promptly, but smooth ordinary
            // variation and limit the impact of isolated scheduling stalls.
            if (!initialized || sample < us * 0.5) us = sample;
            else us = us * 0.9 + std::min(sample, us * 3) * 0.1;
            initialized = true;
        }
    };
    std::array<Estimate, maxExperts + 1> cpu{}, compute{};
    Estimate refill, dispatch, ensure;
    uint64_t calls = 0;

    int SelectGpuCount(int topk, int hits, int layers = 1) const {
        // Populate each layer once, then calibrate splits across layer calls:
        // timings are shared, so repeating every split at every layer would
        // needlessly penalize the first tokens. Sparse prime-interval probes
        // avoid refreshing only a subset of layers (e.g. 1024 vs 48).
        layers = std::max(1, layers);
        if (calls < uint64_t(layers)) return topk;
        if (calls < uint64_t(layers + 2 * (topk + 1)))
            return topk - (calls - layers) % (topk + 1);
        if (calls % 1021 == 0) return topk - (calls / 1021) % (topk + 1);
        int selected = 0;
        double best = cpu[topk].us;
        for (int g = std::max(1, hits); g <= topk; ++g) {
            const double gpuUs = ensure.us + compute[g].us + (g - hits) * refill.us;
            const double expected = dispatch.us + std::max(cpu[topk - g].us, gpuUs);
            if (expected < best) { best = expected; selected = g; }
        }
        return selected;
    }
};

// Observe every routed expert, including CPU work, in a payload-free LRU with
// the real cache capacity. This estimates reuse without copying cold weights
// into a small cache just to find out that they will be evicted immediately.
class MoeDecodePolicy {
public:
    enum class Mode { Hybrid, FillGpu, MeasureGpu, Gpu };

    MoeDecodePolicy(int records, int capacity)
        : links(records), capacity(std::min(records, capacity)) {}

    bool UseGpu() const { return mode != Mode::Hybrid; }
    Mode GetMode() const { return mode; }
    double HybridUs() const { return hybrid.us; }
    double GpuUs() const { return gpu.us; }
    double MissRate() const { return missRate; }
    uint64_t hybridSteps = 0, gpuSteps = 0;

    void ObserveRoutes(int base, const int *experts, int count) {
        for (int i = 0; i < count; ++i) {
            const int key = base + experts[i];
            if (experts[i] < 0 || key < 0 || key >= int(links.size()) || capacity <= 0) {
                continue;
            }
            ++routes;
            auto &entry = links[key];
            if (entry.present) {
                Unlink(key);
            } else {
                ++misses;
                if (used == capacity) {
                    const int victim = tail;
                    Unlink(victim);
                    links[victim].present = false;
                } else ++used;
                entry.present = true;
            }
            entry.previous = -1;
            entry.next = head;
            if (head >= 0) links[head].previous = key;
            else tail = key;
            head = key;
        }
    }

    // The caller measures a complete, already synchronized decode step. No
    // extra CUDA synchronization is needed for this model-level comparison.
    void ObserveStep(double us, double refillUs, double computeUs, int layers, int topk) {
        if (us <= 0) return;
        if (UseGpu()) ++gpuSteps;
        else ++hybridSteps;
        ++steps;
        if (mode == Mode::Hybrid) {
            hybrid.Observe(us);
            if (cooldown > 0) --cooldown;
            if (steps < 32) return;
            missRate = routes ? double(misses) / routes : 1.0;
            // Be conservative: give refill at most a quarter of the current
            // step budget. Actual whole-step timing, not this estimate, makes
            // the final decision. No datatype or GiB threshold is involved.
            const bool ready = routes >= uint64_t(layers * topk) * 16 &&
                refillUs > 0 && computeUs > 0;
            if (!cooldown && ready &&
                missRate * layers * topk * refillUs < hybrid.us * 0.25 &&
                layers * computeUs < hybrid.us * 0.75) {
                mode = Mode::FillGpu;
                gpu = {};
            }
            steps = 0;
            routes = misses = 0;
        } else if (mode == Mode::FillGpu) {
            // A cold large cache can take longer than one short window to
            // populate. Wait for a promising rate, with a bounded warmup,
            // then discard those fill/graph-creation samples before judging.
            gpu.Observe(us);
            if (steps >= 128 || (steps >= 32 && gpu.us < hybrid.us * 0.90)) {
                mode = Mode::MeasureGpu;
                steps = 0;
                gpu = {};
            }
        } else if (mode == Mode::MeasureGpu) {
            gpu.Observe(us);
            if (steps >= 16) {
                if (gpu.us < hybrid.us * 0.95) {
                    mode = Mode::Gpu;
                    steps = 0;
                } else {
                    ReturnToHybrid(512);
                }
            }
        } else {
            gpu.Observe(us);
            if (steps >= 32) {
                // Reevaluate after routing/context changes make the selected
                // path slower. Hysteresis avoids oscillating on normal noise.
                if (gpu.us > hybrid.us * 1.15) ReturnToHybrid(64);
                else steps = 0;
            }
        }
    }

private:
    struct Link { int previous = -1, next = -1; bool present = false; };
    std::vector<Link> links;
    int capacity, used = 0, head = -1, tail = -1;
    Mode mode = Mode::Hybrid;
    MoeDecodeScheduler::Estimate hybrid, gpu;
    int steps = 0, cooldown = 0;
    uint64_t routes = 0, misses = 0;
    double missRate = 1.0;

    void Unlink(int key) {
        const auto &entry = links[key];
        if (entry.previous >= 0) links[entry.previous].next = entry.next;
        else head = entry.next;
        if (entry.next >= 0) links[entry.next].previous = entry.previous;
        else tail = entry.previous;
    }
    void ReturnToHybrid(int delay) {
        mode = Mode::Hybrid;
        cooldown = delay;
        steps = used = 0;
        head = tail = -1;
        routes = misses = 0;
        std::fill(links.begin(), links.end(), Link{});
    }
};
} // namespace fastllm
