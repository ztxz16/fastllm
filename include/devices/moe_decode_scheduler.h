#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
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
        // Residency removes refill cost; it does not require executing every
        // resident expert on CUDA. A smaller resident subset can balance CPU
        // and GPU work better than either all hits or the CPU-only choice.
        for (int g = 1; g <= topk; ++g) {
            const double gpuUs = ensure.us + compute[g].us + std::max(0, g - hits) * refill.us;
            const double expected = dispatch.us + std::max(cpu[topk - g].us, gpuUs);
            if (expected < best) { best = expected; selected = g; }
        }
        return selected;
    }
};

// Resident experts stay on the GPU. Only misses are split between NUMA and
// temporary GPU storage; this policy never admits or evicts a cache entry.
// Keep one instance per layer: quantization, dimensions and CPU cost can vary.
class MoeDecodeOverlapScheduler {
public:
    using Estimate = MoeDecodeScheduler::Estimate;
    Estimate cpuExpert, residentExpert, copiedExpert, stagedExpert, dispatch;
    uint64_t calls = 0;

    // routeCounts describes reuse of each unique missed expert in a verifier.
    // Copies are charged once per expert; CPU/GPU arithmetic once per route.
    int SelectMisses(int misses, int hits, const int *routeCounts = nullptr) const {
        if (misses <= 0) return 0;
        // Measure useful work during warmup instead of running extra experts.
        // A single miss needs two calls to observe both CPU and transfer costs.
        if (!cpuExpert.initialized) return misses / 2;
        if (!copiedExpert.initialized || !stagedExpert.initialized)
            return std::max(1, misses / 2);
        const double resident = hits * residentExpert.us;
        int remaining = 0;
        for (int i = 0; i < misses; ++i) remaining += routeCounts ? routeCounts[i] : 1;
        double best = std::max(remaining * cpuExpert.us, resident);
        int selected = 0;
        double gpu = resident;
        for (int n = 1; n <= misses; ++n) {
            // Each expert can run as soon as its own DMA and the preceding
            // GPU work finish, overlapping transfers of subsequent experts.
            const int routes = routeCounts ? routeCounts[n - 1] : 1;
            remaining -= routes;
            gpu = std::max(gpu, n * copiedExpert.us) + routes * stagedExpert.us;
            const double cost = dispatch.us + std::max(remaining * cpuExpert.us, gpu);
            if (cost < best * .97) { best = cost; selected = n; }
        }
        // Refresh an unused path occasionally, so a change in CPU/PCIe speed
        // can recover from an all-CPU or all-GPU choice without permanent bias.
        if (calls % 127 == 126) {
            if (!selected) return 1;
            if (selected == misses) return selected - 1;
        }
        return selected;
    }

    struct RankPlan {
        const MoeDecodeOverlapScheduler *timing = nullptr;
        int hits = 0, misses = 0;
        const int *routeCounts = nullptr;
        int selected = 0;
    };

    // All ranks share one CPU worker pool. Account for its entire remaining
    // subset while each GPU has its own resident work and measured DMA cost.
    // A null timing leaves this rank's misses on CPU (e.g. no staging buffer).
    static void SelectParallelMisses(std::vector<RankPlan> &plans, const Estimate &cpu) {
        int remaining = 0;
        bool calibrating = !cpu.initialized;
        for (auto &p : plans) {
            p.selected = 0;
            for (int i = 0; i < p.misses; ++i)
                remaining += p.routeCounts ? p.routeCounts[i] : 1;
            calibrating |= p.timing && p.misses &&
                (!p.timing->copiedExpert.initialized || !p.timing->stagedExpert.initialized);
        }
        if (!remaining) return;
        if (calibrating) {
            for (auto &p : plans) if (p.timing && p.misses)
                p.selected = cpu.initialized ? std::max(1, p.misses / 2) : p.misses / 2;
            return;
        }
        std::vector<int> current(plans.size(), 0);
        std::vector<double> gpu(plans.size(), 0);
        double gpuBound = 0;
        for (size_t r = 0; r < plans.size(); ++r) if (plans[r].timing) {
            gpu[r] = plans[r].hits * plans[r].timing->residentExpert.us;
            gpuBound = std::max(gpuBound, gpu[r]);
        }
        double best = std::max(remaining * cpu.us, gpuBound);
        // Sweep increasing GPU completion-time limits. At each limit, taking
        // every affordable prefix minimizes the common CPU remainder. This
        // avoids enumerating every combination of per-rank expert counts.
        while (true) {
            int next = -1;
            double finish = 0, compute = 0;
            for (size_t r = 0; r < plans.size(); ++r) {
                const auto &p = plans[r];
                if (!p.timing || current[r] == p.misses) continue;
                const int routes = p.routeCounts ? p.routeCounts[current[r]] : 1;
                const double cost = std::max(gpu[r], (current[r] + 1) * p.timing->copiedExpert.us) +
                    routes * p.timing->stagedExpert.us;
                const double end = p.timing->dispatch.us + cost;
                if (next < 0 || end < finish) { next = r; finish = end; compute = cost; }
            }
            if (next < 0) break;
            const auto &p = plans[next];
            remaining -= p.routeCounts ? p.routeCounts[current[next]] : 1;
            ++current[next];
            gpu[next] = compute;
            gpuBound = std::max(gpuBound, finish);
            const double cost = std::max(remaining * cpu.us, gpuBound);
            if (cost < best * .97) {
                best = cost;
                for (size_t r = 0; r < plans.size(); ++r) plans[r].selected = current[r];
            }
        }
        for (auto &p : plans) if (p.timing && p.misses && p.timing->calls % 127 == 126) {
            if (!p.selected) p.selected = 1;
            else if (p.selected == p.misses) --p.selected;
        }
    }
};

// Observe every routed expert, including CPU work, in a payload-free LRU with
// the real capacity and record-size partitions. This estimates reuse without copying cold weights
// into a small cache just to find out that they will be evicted immediately.
class MoeDecodePolicy {
public:
    enum class Mode { Hybrid, FillGpu, MeasureGpu, Gpu };

    MoeDecodePolicy(int records, int capacity)
        : links(records), partitions(1) { partitions[0].capacity = std::max(0, std::min(records, capacity)); }

    MoeDecodePolicy(const std::vector<int> &partitionForKey, const std::vector<int> &capacities)
        : links(partitionForKey.size()), keyPartitions(partitionForKey), partitions(capacities.size()) {
        for (int id : keyPartitions)
            if (id < 0 || id >= int(partitions.size())) throw std::invalid_argument("invalid cache partition");
        for (size_t i = 0; i < capacities.size(); ++i)
            partitions[i].capacity = std::max(0, capacities[i]);
    }

    bool UseGpu() const { return mode != Mode::Hybrid; }
    Mode GetMode() const { return mode; }
    double HybridUs() const { return hybrid.us; }
    double GpuUs() const { return gpu.us; }
    double MissRate() const { return missRate; }
    uint64_t hybridSteps = 0, gpuSteps = 0;

    void ObserveRoutes(int base, const int *experts, int count) {
        for (int i = 0; i < count; ++i) {
            const int key = base + experts[i];
            if (experts[i] < 0 || key < 0 || key >= int(links.size())) {
                continue;
            }
            auto &part = partitions[keyPartitions.empty() ? 0 : keyPartitions[key]];
            ++routes;
            if (part.capacity <= 0) { ++misses; continue; }
            auto &entry = links[key];
            if (entry.present) {
                Unlink(key, part);
            } else {
                ++misses;
                if (part.used == part.capacity) {
                    const int victim = part.tail;
                    Unlink(victim, part);
                    links[victim].present = false;
                } else ++part.used;
                entry.present = true;
            }
            entry.previous = -1;
            entry.next = part.head;
            if (part.head >= 0) links[part.head].previous = key;
            else part.tail = key;
            part.head = key;
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
    struct Partition { int capacity = 0, used = 0, head = -1, tail = -1; };
    std::vector<int> keyPartitions;
    std::vector<Partition> partitions;
    Mode mode = Mode::Hybrid;
    MoeDecodeScheduler::Estimate hybrid, gpu;
    int steps = 0, cooldown = 0;
    uint64_t routes = 0, misses = 0;
    double missRate = 1.0;

    void Unlink(int key, Partition &part) {
        const auto &entry = links[key];
        if (entry.previous >= 0) links[entry.previous].next = entry.next;
        else part.head = entry.next;
        if (entry.next >= 0) links[entry.next].previous = entry.previous;
        else part.tail = entry.previous;
    }
    void ReturnToHybrid(int delay) {
        mode = Mode::Hybrid;
        cooldown = delay;
        steps = 0;
        for (auto &part : partitions) { part.used = 0; part.head = part.tail = -1; }
        routes = misses = 0;
        std::fill(links.begin(), links.end(), Link{});
    }
};
} // namespace fastllm
