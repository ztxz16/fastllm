#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void TestFrequencyAdmission() {
    using fastllm::MoeFrequencyPolicy;
    MoeFrequencyPolicy policy({0,0,0,0,1,1}, {0,1}, 2);
    int owners[] = {0,4}; policy.SetResidents(owners);
    int resident = 0, cold = 1;
    for (int step = 0; step < 8; ++step) {
        policy.BeginStep(); policy.Observe(0, &resident, 1);
    }
    policy.Observe(0, &cold, 1);
    Require(policy.Select(0,0,&cold,1).key < 0, "single miss admitted");
    // Hysteresis prevents replacing a resident at equal/slightly higher heat.
    for (int i = 0; i < 10; ++i) policy.Observe(0,&cold,1);
    Require(policy.Select(0,0,&cold,1).key < 0, "replacement hysteresis ignored");
    policy.Observe(0,&cold,1);
    auto admission = policy.Select(0,0,&cold,1);
    Require(admission.key == 1 && admission.slot == 0, "hot miss did not select compatible victim");
    policy.Admit(0,admission);
    Require(policy.Select(0,0,&resident,1).key < 0, "layer admitted twice");
    int duplicates[] = {1,1,-1,999};
    policy.Observe(0,duplicates,4);
    Require(policy.Score(1) == 13, "duplicate or invalid route inflated frequency");
    int other = 5;
    for (int i = 0; i < 30; ++i) policy.Observe(0,&other,1);
    admission = policy.Select(1,0,&other,1);
    Require(admission.slot == 1, "admission crossed a record-size partition");
    policy.Admit(1,admission);
    // A newly admitted expert cannot be churned out by the next hot route.
    policy.BeginStep();
    for (int i = 0; i < 100; ++i) policy.Observe(0,&resident,1);
    Require(policy.Select(0,0,&resident,1).key < 0, "minimum residence ignored");
    const float heat = policy.Score(1);
    for (int i = 0; i < 128; ++i) policy.BeginStep();
    Require(std::abs(policy.Score(1) - heat * .5f) < 1e-4f, "frequency half-life incorrect");
    admission = policy.Select(0,0,&resident,1);
    Require(admission.key == 0, "policy did not adapt to a changed hot set");
    // A prefill may change all slots; decode frequencies survive reconciliation.
    const float retainedHeat = policy.Score(1);
    int changed[] = {2,5}; policy.SetResidents(changed);
    Require(policy.Score(1) == retainedHeat, "prefill reconciliation lost decode history");
    bool rejected = false;
    try { int invalid[] = {5,2}; policy.SetResidents(invalid); }
    catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "invalid partition snapshot accepted");

    for (const int layers : {1, 3, 8, 10, 48}) {
        MoeFrequencyPolicy fair(std::vector<int>(2*layers,0), std::vector<int>(layers,0), layers);
        std::vector<int> ownersFair(layers);
        for (int i = 0; i < layers; ++i) ownersFair[i] = i;
        fair.SetResidents(ownersFair.data());
        for (int i = 0; i < MoeFrequencyPolicy::minimumResidence; ++i) fair.BeginStep();
        for (int layer = 0; layer < layers; ++layer) {
            fair.Observe(layers,&layer,1); fair.Observe(layers,&layer,1);
        }
        const int budget = std::min(layers, MoeFrequencyPolicy::maxReplacementsPerStep);
        std::vector<bool> visited(layers,false);
        for (int step = 0; step < (layers + budget - 1) / budget; ++step) {
            fair.BeginStep(); int admitted = 0;
            for (int layer = 0; layer < layers; ++layer) {
                auto a = fair.Select(layer,layers,&layer,1);
                if (a.key < 0) continue;
                fair.Admit(layer,a); visited[layer] = true; ++admitted;
            }
            Require(admitted > 0 && admitted <= budget, "per-token admission budget incorrect");
        }
        Require(std::all_of(visited.begin(),visited.end(),[](bool v) { return v; }), "later layers starved");
    }
    MoeFrequencyPolicy coldStart(std::vector<int>(10,0), std::vector<int>(8,0), 1);
    int empty[] = {-1,-1,-1,-1,-1,-1,-1,-1}; coldStart.SetResidents(empty);
    int routes[] = {0,1,2,3,4,5,6,7,8,9};
    for (int i = 0; i < 2; ++i) {
        coldStart.BeginStep(); coldStart.Observe(0,routes,10);
        for (int j = 0; j < 4; ++j) {
            auto a = coldStart.Select(0,0,routes,10);
            Require(a.key == i * 4 + j, "cold cache did not fill distinct empty slots"); coldStart.Admit(0,a);
        }
        Require(coldStart.Select(0,0,routes,10).key < 0, "cold fill budget or residence ignored");
    }
    MoeFrequencyPolicy zero({0,0}, {}, 1);
    zero.SetResidents(nullptr); zero.BeginStep();
    zero.Observe(0,&resident,1); zero.Observe(0,&resident,1);
    Require(zero.Select(0,0,&resident,1).key < 0, "zero capacity admitted an expert");
}

static void TestDecodeOverlap() {
    fastllm::MoeDecodeOverlapScheduler p;
    Require(p.SelectMisses(0, 10) == 0, "all-hit layer requested PCIe work");
    Require(p.SelectMisses(4, 6) == 2, "warmup did not sample CPU and PCIe together");
    Require(p.SelectMisses(1, 9) == 0, "single-miss warmup did not measure CPU first");
    p.cpuExpert.Observe(100);
    Require(p.SelectMisses(1, 9) == 1, "single-miss warmup did not measure PCIe next");
    p.residentExpert.Observe(5);
    p.copiedExpert.Observe(75);
    p.stagedExpert.Observe(10);
    p.dispatch.Observe(8);
    Require(p.SelectMisses(3, 7) == 2, "three-way overlap was treated as serialized work");
    for (int i = 0; i < 100; ++i) p.copiedExpert.Observe(500);
    Require(p.SelectMisses(3, 7) == 0, "PCIe slowdown did not shift work back to NUMA");
    p.calls = 126;
    Require(p.SelectMisses(3, 7) == 1, "CPU-only choice never reprobes PCIe");
    p.calls = 127;
    p.copiedExpert.Observe(10);
    Require(p.SelectMisses(3, 7) == 3, "faster PCIe did not recover GPU offload");
    p.calls = 253;
    Require(p.SelectMisses(3, 7) == 2, "GPU-only choice never reprobes NUMA");
    p.calls = 254;
    p.cpuExpert.Observe(10);
    Require(p.SelectMisses(3, 7) == 0, "faster NUMA did not reduce GPU offload");
    p.cpuExpert.Observe(100);
    p.residentExpert.Observe(1000);
    // Use sustained samples, rather than assuming one noisy observation wins.
    for (int i = 0; i < 100; ++i) {
        p.cpuExpert.Observe(100);
        p.residentExpert.Observe(1000);
    }
    Require(p.SelectMisses(3, 7) == 0, "busy resident GPU attracted additional experts");

    fastllm::MoeDecodeOverlapScheduler pipeline;
    pipeline.cpuExpert.Observe(100);
    pipeline.copiedExpert.Observe(60);
    pipeline.stagedExpert.Observe(40);
    Require(pipeline.SelectMisses(6, 0) == 4,
            "expert DMA and compute were treated as whole-batch serialization");
    const int reused[] = {4, 3, 1};
    Require(pipeline.SelectMisses(3, 0, reused) == 2,
            "verify split did not charge one copy and multiple computations per expert");
    fastllm::MoeDecodeOverlapScheduler transferBound;
    transferBound.cpuExpert.Observe(100);
    transferBound.copiedExpert.Observe(250);
    transferBound.stagedExpert.Observe(10);
    const int fourRows[] = {4, 4};
    Require(transferBound.SelectMisses(2, 0) == 0 &&
            transferBound.SelectMisses(2, 0, fourRows) == 1,
            "verifier reuse did not amortize expert transfer cost");
    fastllm::MoeDecodeOverlapScheduler computeBound;
    computeBound.cpuExpert.Observe(50);
    computeBound.copiedExpert.Observe(20);
    computeBound.stagedExpert.Observe(80);
    Require(computeBound.SelectMisses(6, 0) == 2,
            "compute-bound pipeline did not wait for the preceding expert");
}

static void TestParallelOverlap() {
    using Scheduler = fastllm::MoeDecodeOverlapScheduler;
    Scheduler a, b;
    Scheduler::Estimate cpu;
    std::vector<Scheduler::RankPlan> plans{{&a, 0, 3}, {&b, 0, 3}};
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected == 1 && plans[1].selected == 1,
            "parallel calibration did not retain a CPU subset");
    cpu.Observe(100);
    for (auto *p : {&a, &b}) {
        p->cpuExpert.Observe(100);
        p->copiedExpert.Observe(180);
        p->stagedExpert.Observe(10);
    }
    Require(a.SelectMisses(3, 0) == 1 && b.SelectMisses(3, 0) == 1,
            "independent-rank reference changed");
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected + plans[1].selected == 3,
            "parallel split counted the shared CPU worker pool twice");
    for (int i = 0; i < 100; ++i) b.copiedExpert.Observe(2000);
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected > 0 && plans[1].selected == 0,
            "parallel split ignored the slower PCIe link");
    b.calls = 126;
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[1].selected == 1, "parallel split did not reprobe an unused link");
    b.calls = 127;
    b.copiedExpert.Observe(1); b.stagedExpert.Observe(1);
    plans[0].timing = nullptr;
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected == 0 && plans[1].selected == 3,
            "CPU-only rank was omitted from the common CPU budget");
    const int reused[] = {4, 4};
    a.copiedExpert.Observe(250); a.stagedExpert.Observe(10);
    plans = {{&a, 0, 2, reused}, {&b, 0, 0}};
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected == 1, "parallel verifier charged repeated DMA for a shared expert");
    // The planner is independent of TP degree, including an empty rank.
    plans = {{&b, 0, 3}, {&b, 0, 3}, {&b, 0, 3}, {&b, 0, 0}};
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(plans[0].selected == 3 && plans[1].selected == 3 &&
            plans[2].selected == 3 && plans[3].selected == 0,
            "parallel planner assumes two ranks or assigns an empty rank");
    cpu.Observe(.01);
    Scheduler::SelectParallelMisses(plans, cpu);
    Require(std::all_of(plans.begin(), plans.end(), [](const auto &p) { return p.selected == 0; }),
            "parallel planner did not adapt to faster NUMA");
}

int main() {
    TestFrequencyAdmission();
    TestDecodeOverlap();
    TestParallelOverlap();
    using fastllm::MoeDecodePolicy;
    // Four alternating experts fit in a global cache, but cannot borrow
    // unused slots from a different record-size partition.
    MoeDecodePolicy global(8, 6), split({0,0,0,0,1,1,1,1}, {2,4});
    auto sample = [](MoeDecodePolicy &p, int step) {
        int ids[] = {(step % 2) * 2, (step % 2) * 2 + 1};
        p.ObserveRoutes(0, ids, 2);
        int resident[] = {0,1}; p.ObserveRoutes(4, resident, 2);
        p.ObserveStep(1000, 100, 10, 2, 2);
    };
    for (int i = 0; i < 32; ++i) { sample(global,i); sample(split,i); }
    Require(global.UseGpu() && global.MissRate() < .05, "global reference should fit");
    Require(split.MissRate() > .50 && split.MissRate() < .52, "partition capacity was borrowed");

    // Nonadjacent layers of equal record size share a single partition.
    MoeDecodePolicy shared({0,0,1,1,0,0}, {1,2});
    for (int i = 0; i < 32; ++i) {
        int id = 0;
        shared.ObserveRoutes(0,&id,1); shared.ObserveRoutes(4,&id,1);
        shared.ObserveStep(1000,1000,10,2,1);
    }
    Require(shared.MissRate() == 1 && !shared.UseGpu(), "shared partition did not evict across layers");

    // A failed GPU trial clears occupancy, but preserves partition ownership.
    for (int i = 0; i < 180; ++i) split.ObserveStep(2000,100,10,2,2);
    Require(!split.UseGpu(), "slower GPU trial did not return to hybrid");
    for (int i = 0; i < 64; ++i) sample(split,i);
    Require(std::abs(split.MissRate() - .5) < .001, "reset lost partition configuration");
    MoeDecodePolicy zero({0,0}, {0});
    for (int i = 0; i < 32; ++i) {
        int ids[] = {0,-1,9}; zero.ObserveRoutes(0,ids,3);
        zero.ObserveStep(1000,1000,10,1,1);
    }
    Require(zero.MissRate() == 1 && !zero.UseGpu(), "zero capacity incorrectly counted hits");
    std::cout << "ALL_PASS\n";
}
