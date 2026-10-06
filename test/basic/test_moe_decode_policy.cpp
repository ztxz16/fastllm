#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void TestFrequencyAdmission() {
    using fastllm::MoeCacheConfig;
    using fastllm::MoeFrequencyPolicy;
    MoeCacheConfig config;
    config.halfLife = 0;
    config.maxReplacements = 1;
    MoeFrequencyPolicy policy({0,0,0,0,1,1}, {0,1}, 2, config);
    int owners[] = {0,4}; policy.SetResidents(owners);
    int early = 1, late = 5;
    policy.BeginStep();
    for (int i = 0; i < 3; ++i) policy.Observe(0, &early, 1);
    for (int i = 0; i < 5; ++i) policy.Observe(0, &late, 1);
    const float heat = policy.Score(early);
    int duplicates[] = {1,1,-1,999};
    policy.Observe(0, duplicates, 4);
    Require(policy.Score(early) == heat + 1, "duplicate or invalid route inflated frequency");
    Require(policy.Slot(early) < 0 && policy.Slot(late) < 0, "admitted before the token ended");
    auto plan = policy.EndStep();
    Require(plan.size() == 1 && plan[0].key == late && plan[0].slot == 1,
            "global budget did not choose the higher-benefit later layer");
    Require(policy.EndStep().empty(), "repeated End spent the budget twice");
    policy.BeginStep();
    plan = policy.EndStep();
    Require(plan.size() == 1 && plan[0].key == early && plan[0].slot == 0,
            "previously observed hot miss was forgotten or crossed a size partition");
    // Scores survive ordinary cache reconciliation, but invalid ownership does not.
    const float retainedHeat = policy.Score(early);
    int changed[] = {2,5}; policy.SetResidents(changed);
    Require(policy.Score(early) == retainedHeat, "reconciliation lost heat");
    bool rejected = false;
    try { int invalid[] = {5,2}; policy.SetResidents(invalid); }
    catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "invalid partition snapshot accepted");

    // Both record bytes and expert count are hard per-update limits.
    for (bool byBytes : {false, true}) for (bool uniformSlots : {false, true}) {
        config.maxReplacements = 2; config.maxBytes = 100; config.rankByBytes = byBytes;
        MoeFrequencyPolicy bytes(uniformSlots ? std::vector<int>{0,0,0,0} : std::vector<int>{0,0,1,1},
            uniformSlots ? std::vector<int>{0,0} : std::vector<int>{0,1}, 2, config, {80,80,40,40});
        int old[] = {0,2}; bytes.SetResidents(old);
        bytes.BeginStep();
        int a=1,b=3;
        for(int i=0;i<4;++i) bytes.Observe(0,&a,1);
        for(int i=0;i<3;++i) bytes.Observe(0,&b,1);
        plan=bytes.EndStep();
        Require(plan.size()==1 && plan[0].key==(byBytes ? b : a),
                "byte ceiling or benefit-per-byte ranking failed");
    }
    config = MoeCacheConfig(); config.halfLife=0; config.maxReplacements=3;
    {
        auto weightedConfig=config;weightedConfig.rankByBytes=true;weightedConfig.replacementMargin=0;
        MoeFrequencyPolicy weighted({0,0,0},{0},1,weightedConfig,{80,80,40});
        int old=0;weighted.SetResidents(&old);weighted.BeginStep();int big=1,small=2;
        for(int i=0;i<4;++i)weighted.Observe(0,&big,1);
        for(int i=0;i<3;++i)weighted.Observe(0,&small,1);
        plan=weighted.EndStep();
        Require(plan.size()==1 && plan[0].key==small,"batch reused a reserved slot for a differently sized candidate");
    }
    MoeFrequencyPolicy empty(std::vector<int>(10,0), std::vector<int>(8,0), 1,config);
    int routes[]={0,1,2,3,4,5,6,7,8,9};
    empty.BeginStep(); empty.Observe(0,routes,10); plan=empty.EndStep();
    Require(plan.size()==3 && plan[0].key==0 && plan[2].key==2,
            "empty-slot fills exceeded the global budget or unstable tie ordering");
    MoeFrequencyPolicy zero({0,0}, {}, 1, config);
    zero.BeginStep(); zero.Observe(0,&early,1);
    Require(zero.EndStep().empty(), "zero capacity admitted an expert");

    config.minimumResidence=3;
    MoeFrequencyPolicy residence({0,0}, {0}, 1,config);
    residence.BeginStep(); residence.Observe(0,&early,1); plan=residence.EndStep();
    Require(plan.size()==1,"empty slot incorrectly subject to residence protection");
    int replacement=0;
    for(int step=0;step<3;++step) {
        residence.BeginStep();
        for(int i=0;i<10;++i) residence.Observe(0,&replacement,1);
        plan=residence.EndStep();
        Require(plan.size()==(step==2 ? 1u : 0u),"minimum residence not enforced");
    }

    config = MoeCacheConfig();config.halfLife=4;config.updateInterval=2;config.maxReplacements=0;
    MoeFrequencyPolicy decay({0,0},{0},1,config);
    decay.BeginStep();decay.Observe(0,&early,1);
    Require(decay.EndStep().empty() && decay.Score(early)==1,"decayed before the update interval");
    for(int step=1;step<4;++step){decay.BeginStep();Require(decay.EndStep().empty(),"disabled policy admitted");}
    Require(std::abs(decay.Score(early)-.5f)<1e-6f,"decay half-life did not account for update interval");
    for (int bad=0;bad<7;++bad) {
        auto invalid=MoeCacheConfig();
        switch(bad){case 0:invalid.halfLife=NAN;break;case 1:invalid.updateInterval=0;break;
        case 2:invalid.maxReplacements=-1;break;case 3:invalid.replacementFactor=.5;break;
        case 4:invalid.prefillPrior=2;break;case 5:invalid.minHeat=-1;break;case 6:invalid.replacementMargin=INFINITY;break;}
        rejected=false;
        try{invalid.Validate();}catch(const std::invalid_argument &){rejected=true;}
        Require(rejected,"invalid policy config accepted");
    }
}

static void TestPrefillAdmission() {
    using fastllm::MoeFrequencyPolicy;
    MoeFrequencyPolicy policy(std::vector<int>(24, 0), std::vector<int>(12, 0), 2);
    std::vector<int> counts(12,32);counts[0]=64;
    policy.ObservePrefill(0,counts,128);
    Require(std::abs(policy.Score(0)-32/std::log(2.0f))<1e-4f,"prefill scale changed");
    for(int e=0;e<12;++e){auto a=policy.SelectPrefill(e,0,counts);
        Require(a.key==e && a.slot==e,"prefill fill stopped at decode budget");policy.Admit(0,a);}
    const float firstLayerHeat=policy.Score(0);
    policy.ObservePrefill(12,std::vector<int>(12,128),128);
    Require(policy.Score(0)==firstLayerHeat,"other layer lost heat before observation");
    Require(policy.SelectPrefill(12,0,counts).key<0,"active borrowed cache slot evicted");
    auto a=policy.SelectPrefill(12,12,counts);
    Require(a.key==12 && a.slot!=0,"prefill did not select a cold inactive victim");
    policy.Admit(1,a);
    Require(policy.SelectPrefill(13,12,counts).slot!=a.slot,"prefill reused an active reservation");
    policy.BeginStep();
    Require(policy.Score(0)==0 && policy.Score(12)==0 && policy.Slot(12)==a.slot,
            "default decode prior did not clear heat while retaining payloads");
    int e=1;policy.Observe(0,&e,1);policy.EndStep();
    const float heat=policy.Score(1);
    policy.BeginStep();Require(policy.Score(1)==heat,"decode reset heat on every step");policy.EndStep();

    fastllm::MoeCacheConfig config;config.halfLife=32;config.prefillPrior=.5f;
    MoeFrequencyPolicy prior({0,0},{0},1,config);
    prior.ObservePrefill(0,{128,0},128);const float before=prior.Score(0);
    prior.BeginStep();Require(std::abs(prior.Score(0)-before*.125f)<1e-4f,"prefill prior weight/scale ignored");
    prior.EndStep();const float after=prior.Score(0);
    prior.ObservePrefill(1,{128},128);
    Require(std::abs(prior.Score(0)-after*4)<1e-4f,"next prefill did not restore heat scale once");
    prior.ObservePrefill(1,{128},128);
    Require(std::abs(prior.Score(0)-after*4)<1e-4f,"prefill rescaled unrelated layers repeatedly");
    MoeFrequencyPolicy partition({0,1},{1},1);
    partition.ObservePrefill(0,{128,128},128);
    Require(partition.SelectPrefill(0,0,{1,1}).key<0,"prefill crossed a record-size partition");
}

static void TestFrequencyHotSetChange() {
    using fastllm::MoeFrequencyPolicy;
    for(int layers : {1,3,8,24,48,100}) {
        MoeFrequencyPolicy policy(std::vector<int>(2*layers,0),std::vector<int>(layers,0),layers);
        std::vector<int> owners(layers);
        for(int layer=0;layer<layers;++layer) owners[layer]=2*layer;
        policy.SetResidents(owners.data());
        for(int layer=0;layer<layers;++layer) policy.ObservePrefill(2*layer,{16384,0},16384);
        const int hot=1;
        int lastHits=0;
        for(int step=0;step<16;++step) {
            policy.BeginStep();int hits=0;
            for(int layer=0;layer<layers;++layer){hits+=policy.Slot(2*layer+hot)>=0;policy.Observe(2*layer,&hot,1);}
            const auto plan=policy.EndStep();
            Require(plan.size()<=96,"hot-set transition exceeded global budget");lastHits=hits;
        }
        Require(lastHits==layers,"long prefill prevented adaptation or late layers starved");
        policy.BeginStep();const int cold=0;
        for(int layer=0;layer<layers;++layer) policy.Observe(2*layer,&cold,1);
        Require(policy.EndStep().empty(),"transient miss displaced established hot set");
    }
}

static void TestFrequencyIndexUpdates() {
    using fastllm::MoeCacheConfig;
    using fastllm::MoeFrequencyPolicy;
    // Decay can collapse different float scores to the same zero. The coldest
    // slot must then follow the original slot-ID tie break, not heap history.
    MoeCacheConfig config;
    config.halfLife = 1.0f / 128;
    config.replacementMargin = 0;
    config.maxReplacements = 1;
    MoeFrequencyPolicy ties({0,0,0,0}, {0,0}, 1, config);
    int owners[] = {0,1}; ties.SetResidents(owners);
    int hot = 0, miss = 3;
    ties.BeginStep(); ties.Observe(0, &hot, 1); ties.EndStep();
    Require(ties.Score(0) == 0, "tiny half-life did not collapse heat");
    ties.BeginStep(); ties.Observe(0, &miss, 1);
    auto plan = ties.EndStep();
    Require(plan.size() == 1 && plan[0].slot == 0,
            "decay ties retained the previous cold-slot order");
    // All candidate scores have also collapsed; equal new observations must
    // recover the expert-ID ordering regardless of previous heap positions.
    ties.BeginStep(); int equal[] = {2,0}; ties.Observe(0, equal, 2);
    plan = ties.EndStep();
    Require(plan.size() == 1 && plan[0].key == 0 && plan[0].slot == 0,
            "candidate score update did not restore deterministic ties");

    config.halfLife = 0;
    MoeFrequencyPolicy prefill(std::vector<int>(9,0), std::vector<int>(6,0), 1, config);
    int initial[] = {0,1,2,3,4,5}; prefill.SetResidents(initial);
    // Protect several cold ancestors. Search must still find the coldest
    // unprotected descendant, and later score changes must move it both ways.
    prefill.ObservePrefill(0, {1,2,3,4,5,6,128,0,0}, 128);
    auto a = prefill.SelectPrefill(6, 0, {1,1,1,0,0,0,1,0,0});
    Require(a.key == 6 && a.slot == 3, "protected prefill heap ancestors hid a cold descendant");
    prefill.Admit(0, a);
    prefill.ObservePrefill(0, std::vector<int>(9,0), 4096);
    prefill.ObservePrefill(7, {128}, 128);
    a = prefill.SelectPrefill(7, 7, {1});
    Require(a.key == 7 && a.slot == 0, "prefill score decrease left a stale resident order");
    prefill.Admit(0, a);
    // A residency refresh must repopulate candidates that were previously
    // resident, without losing their scores or depending on heap positions.
    prefill.SetResidents(initial);
    prefill.ObservePrefill(8, {64}, 128);
    a = prefill.SelectPrefill(7, 7, {1,1});
    Require(a.key == 7 && a.slot == 0, "residency refresh lost a former resident candidate");
}

static void TestFrequencyAvailability() {
    fastllm::MoeCacheConfig config;
    config.halfLife = 0; config.replacementMargin = 0;
    fastllm::MoeFrequencyPolicy p({0,0,0}, {0}, 1, config);
    int hot=0, warm=1;
    p.SetCandidateEligible(hot, false);
    p.BeginStep();
    for (int i=0;i<8;++i) p.Observe(0,&hot,1);
    p.Observe(0,&warm,1);
    auto plan=p.EndStep();
    Require(plan.size()==1 && plan[0].key==warm,"unavailable hot expert blocked available admission");
    Require(p.Score(hot)==8,"unavailable expert lost heat");
    p.SetCandidateEligible(hot,true);p.BeginStep();plan=p.EndStep();
    Require(plan.size()==1 && plan[0].key==hot,"returning payload did not recover its heat");
    p.SetCandidateEligible(hot,false);
    Require(p.Slot(hot)==0,"candidate availability evicted a live resident");
    Require(p.Evict(hot)==0 && p.Evict(hot)==-1 && p.Score(hot)==8,"migration lost heat or evicted twice");
    p.BeginStep();plan=p.EndStep();
    Require(plan.size()==1 && plan[0].key==warm,"excluded former resident reentered the candidate heap");
    int owner=2;p.SetResidents(&owner);p.BeginStep();plan=p.EndStep();
    Require(plan.size()==1 && plan[0].key==warm,"reconciliation forgot candidate eligibility");
    p.ObservePrefill(0,{128,1,1},128);
    Require(p.SelectPrefill(hot,0,{}).key<0,"prefill ignored candidate eligibility");
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

    fastllm::MoeDecodeOverlapScheduler decode;
    Require(decode.SelectDecodeMisses(8, 0) == 0,
            "decode did not measure the no-upload baseline first");
    decode.ObserveDecodeCpu(8, 600);
    Require(decode.SelectDecodeMisses(8, 0) == 4,
            "decode did not calibrate GPU after the CPU baseline");
    decode.ObserveDecodeCpu(6, 600); // fixed CPU overhead; linear extrapolation is wrong
    decode.copiedExpert.Observe(250);
    decode.stagedExpert.Observe(20);
    decode.dispatch.Observe(50);
    Require(decode.DecodeCpuUs(7) == 600 && decode.SelectDecodeMisses(8, 0) == 0,
            "decode ignored measured CPU costs and chose a slower split");
    decode.decodeCpu[6] = {};
    decode.ObserveDecodeCpu(6, 450);
    decode.copiedExpert = {};
    decode.copiedExpert.Observe(200);
    Require(decode.SelectDecodeMisses(8, 0) == 2,
            "decode serialized host dispatch with already running CPU workers");
    decode.calls = 126;
    Require(decode.SelectDecodeMisses(8, 0) == 0,
            "decode failed to refresh the no-upload baseline");
    decode.calls = 127;
    decode.copiedExpert.Observe(2000);
    for (int i = 0; i < 100; ++i) decode.copiedExpert.Observe(2000);
    Require(decode.SelectDecodeMisses(8, 0) == 0,
            "decode did not adapt to slower PCIe");
    decode.calls = 253;
    Require(decode.SelectDecodeMisses(8, 0) == 1,
            "decode stopped probing an unused GPU path");
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

static void TestSharedOverlap() {
    using Scheduler = fastllm::MoeDecodeOverlapScheduler;
    Scheduler a, b;
    Scheduler::Estimate cpu;
    cpu.Observe(100);
    for (auto *p : {&a, &b}) {
        p->residentExpert.Observe(10); p->copiedExpert.Observe(180); p->stagedExpert.Observe(10);
    }
    std::vector<Scheduler::SharedRankPlan> plans{{&a, 0, 6, 0}, {&b, 0, 6, 20}};
    auto owners = Scheduler::AssignSharedMisses(plans, cpu, {1,1,1,1,1,1}, 1);
    Require(std::find(owners.begin(), owners.end(), 0) != owners.end() &&
            std::find(owners.begin(), owners.end(), 1) != owners.end() &&
            std::find(owners.begin(), owners.end(), -1) != owners.end(), "shared planner did not use three compute paths");
    plans[1].handoffUs = 2000;
    owners = Scheduler::AssignSharedMisses(plans, cpu, {1,1,1,1,1,1}, 1);
    Require(std::find(owners.begin(), owners.end(), 1) == owners.end(), "shared planner ignored host-staged result cost");
    plans[0].capacity = 0;
    owners = Scheduler::AssignSharedMisses(plans, cpu, {1,1,1}, 1);
    Require(std::all_of(owners.begin(), owners.end(), [](int r) { return r == -1; }), "unprofitable GPU received work");
    plans = {{&a, 0, 1, 0}, {&b, 0, 1, 0}, {nullptr, 0, 0, 0}};
    owners = Scheduler::AssignSharedMisses(plans, cpu, {4,4,4,4}, 1);
    Require(std::count(owners.begin(), owners.end(), 0) <= 1 && std::count(owners.begin(), owners.end(), 1) <= 1 &&
            std::find(owners.begin(), owners.end(), 2) == owners.end(), "shared planner exceeded scratch capacity");
    cpu = {}; cpu.Observe(.01);
    owners = Scheduler::AssignSharedMisses(plans, cpu, {4,4,4,4}, 1);
    Require(std::all_of(owners.begin(), owners.end(), [](int r) { return r == -1; }), "shared planner ignored faster NUMA");
    Require(Scheduler::AssignSharedMisses(plans, cpu, {}, 1).empty(), "resident-only batch generated transfers");
    Require(Scheduler::AssignSharedMisses({}, cpu, {1,2}, 126) == std::vector<int>({-1,-1}),
            "missing GPUs did not leave every expert on CPU");
    Scheduler coldA, coldB;
    cpu = {};
    plans = {{&coldA, 0, 1, 0}, {&coldB, 0, 1, 0}};
    bool probed[3]{};
    for (int call = 0; call < 8; ++call) {
        owners = Scheduler::AssignSharedMisses(plans, cpu, {1}, call);
        probed[owners[0] + 1] = true;
    }
    Require(probed[0] && probed[1] && probed[2], "single-expert calibration starved a GPU or CPU");
}

int main() {
    TestFrequencyAdmission();
    TestPrefillAdmission();
    TestFrequencyHotSetChange();
    TestFrequencyIndexUpdates();
    TestFrequencyAvailability();
    TestDecodeOverlap();
    TestParallelOverlap();
    TestSharedOverlap();
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
