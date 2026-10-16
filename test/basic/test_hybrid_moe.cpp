#include "utils/hybrid_moe.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
std::vector<FastllmCudaMoeCacheLayer> prepared;
bool registered = false, streaming = false, prepareResult = true, beginResult = true;
int begun = 0, ended = 0;
std::vector<int> endOrder;
}

// Exercise the model/runtime boundary without allocating GPU or host weights.
// Numerical and format validation remain covered by the real cache tests.
bool FastllmCudaPrepareMoeCache(const FastllmCudaMoeCacheLayer *layers, int count,
        const std::function<void()> &registerWeights, bool allowStreaming) {
    prepared.assign(layers, layers + count);
    registered = bool(registerWeights);
    streaming = allowStreaming;
    if (registerWeights) registerWeights();
    return prepareResult;
}
void *FastllmCudaBeginMoeDecode(Data **, int, int, const std::vector<int> *) {
    ++begun;
    return beginResult ? new int(begun) : nullptr;
}
void FastllmCudaEndMoeDecode(void *state) {
    ++ended;
    endOrder.push_back(*static_cast<int *>(state));
    delete static_cast<int *>(state);
}
bool FastllmCudaPrefetchMoeGGUFHost(Data **, int) { return false; }
bool FastllmCudaMergeMOEHybrid(const Data &, const Data &, const Data &, Data &,
        Data **, int, int, const std::function<void()> &) { return false; }
bool FastllmCudaMergeMOEHybridOnDevices(const Data &, const Data &, const Data &, Data &,
        Data **, int, int, const std::vector<int> &, const std::function<void()> &) { return false; }

int main() {
    std::vector<Data *> target(6), draft(4), resident(4);
    int registrations = 0;
    auto registerWeights = [&] { ++registrations; };
    HybridMoeCachePlan empty;
    empty.Add(resident, "cuda:0");
    empty.Add(resident, "disk");
    Check(!empty.Prepare(registerWeights) && registrations == 0, "resident-only plan registered host weights");
    HybridMoeCachePlan mixed;
    mixed.Add(target, "numa:0");
    mixed.Add(resident, "cuda:1");
    mixed.Add(draft, "cpu");
    Check(mixed.Prepare(registerWeights), "mixed host plan failed");
    Check(!registered && registrations == 0, "CPU weights borrowed from a NUMA-only registration");
    Check(prepared.size() == 2 && prepared[0].weights == target.data() &&
          prepared[1].weights == draft.data() && prepared[0].weightsBatch == 6,
          "table identity or target/draft order changed");
    Check(!prepared[0].deepSeekV41 && !prepared[0].glm5, "standard math changed");
    for (auto math : {HybridMoeMath::Glm5, HybridMoeMath::DeepSeekV41}) {
        HybridMoeCachePlan plan(math, 7.0f);
        plan.Add(target, "numa");
        Check(plan.Prepare(registerWeights, true) && registered && streaming, "streaming NUMA registration lost");
        Check(prepared[0].swigluLimit == 7.0f &&
            prepared[0].glm5 == (math == HybridMoeMath::Glm5) &&
            prepared[0].deepSeekV41 == (math == HybridMoeMath::DeepSeekV41), "numerical profile changed");
    }
    prepareResult = false;
    Check(!mixed.Prepare(registerWeights), "cache rejection did not preserve fallback");
    Check(registrations == 2, "registration callback count changed");
    try {
        HybridMoeDecodeStep step;
        step.Begin(target.data(), target.size(), 2);
        HybridMoeDecodeStep moved(std::move(step));
        Check(!step && moved, "move left two policy owners");
        throw 1; // Simulate cancellation after ownership transfer.
    } catch (int) {}
    Check(ended == 1, "exception did not finish exactly one policy");
    {
        HybridMoeDecodeStep step;
        step.Begin(target.data(), target.size(), 2);
        step.Begin(draft.data(), draft.size(), 1);
        step.Finish();
        step.Finish();
    }
    Check(begun == 3 && ended == 3, "restarted or explicit-finish policy leaked/double-ended");
    {
        HybridMoeDecodeGroup group;
        beginResult = false;
        Check(!group.Begin(1, target.data(), target.size(), 2) && !group.Contains(1), "failed begin retained a scope");
        beginResult = true;
        Check(group.Begin(1, target.data(), target.size(), 2), "failed begin could not retry");
        Check(group.Begin(1, draft.data(), draft.size(), 1) && begun == 5, "device scope restarted within a step");
        Check(group.Begin(0, target.data(), target.size(), 2), "second device failed");
    }
    Check(ended == 5 && endOrder[3] == 6 && endOrder[4] == 5, "device policies ended in unstable order");
    try {
        HybridMoeDecodeGroup group;
        Check(group.Begin(0, target.data(), target.size(), 2), "PP first stage failed");
        group.Finish(2); // No eligible expert cache on this device.
        Check(ended == 5, "absent stage finished a different device");
        group.Finish(0);
        group.Finish(0);
        Check(ended == 6 && endOrder.back() == 7 && group.Contains(0),
              "PP stage was not finished exactly once before the next stage");
        Check(!group.Begin(0, target.data(), target.size(), 2) && begun == 7,
              "completed stage started a second frequency step");
        Check(group.Begin(1, draft.data(), draft.size(), 1), "PP next stage failed");
        Check(ended == 6, "starting the next stage re-ended the previous stage");
        throw 1; // Cancellation must still drain the remaining device scope.
    } catch (int) {}
    Check(begun == 8 && ended == 7 && endOrder.back() == 8,
          "PP cancellation lost or double-ended a device policy");
    std::cout << "ALL_PASS: hybrid cache preparation and policy lifetimes\n";
}
