#include "../../../models/hy4_flash_hybrid_graph.h"
#include "../../../models/hy4_flash_ops.h"
#include "hy4-flash-cuda.cuh"
#include "utils/hybrid_moe.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
namespace fastllm::hy4 {
struct HybridStageGraph {
    int device = -1;
    std::shared_ptr<DiskMoeV41Graph> disk;
    HybridMoeGraph graph;
    bool Launch() {
        return disk ? disk->Launch([this] { return graph.Launch(); }) : graph.Launch();
    }
};

struct HybridExecutable {
    std::vector<std::unique_ptr<Data>> outputs;
    std::vector<std::unique_ptr<HybridStageGraph>> stages;
    size_t children = 0, experts = 0;
    void Launch() {
        // A NUMA mailbox and its CUDA stream memory operations belong to
        // one device. Replay complete PP stages in dependency order.
        for (const auto &stage : stages) {
            Select(stage->device);
            if (!stage->Launch())
                throw std::runtime_error("HY4-Flash hybrid stage graph launch failed");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
};
bool ReplayHybrid(HybridGraphs &graphs, const std::vector<uintptr_t> &key, int kind, int rows) {
    auto found = graphs.replays.find(key);
    if (found == graphs.replays.end())
        return false;
    found->second->Launch();
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 hybrid replay] verify=%d rows=%d launch=%zu\n", kind, rows, ++graphs.launches);
    return true;
}
HybridGraphCapture::HybridGraphCapture(HybridGraphs *graphs, bool diskWeights)
    : CaptureSegments(graphs != nullptr), graphs(graphs),
      compiled(graphs ? std::make_shared<HybridExecutable>() : nullptr),
      disk(graphs && diskWeights ? std::make_shared<DiskMoeV41Graph>() : nullptr) {}

HybridGraphCapture::~HybridGraphCapture() {
    Abort();
    for (void *handle : nativeHandles)
        FastllmCudaReleaseNativeMoeFullGraph(handle);
}

void HybridGraphCapture::CloseStage() {
    if (!graphs)
        return;
    childGraphs.push_back(Close());
    auto stage = std::make_unique<HybridStageGraph>();
    stage->device = FastllmCudaGetDevice();
    stage->disk = std::move(disk);
    compiled->children += childGraphs.size();
    compiled->experts += stage->disk ? (childGraphs.size() - 1) / 2 : nativeHandles.size();
    if (!stage->graph.Compose(childGraphs, std::move(nativeHandles)))
        throw std::runtime_error("HY4-Flash hybrid stage graph composition failed");
    compiled->stages.push_back(std::move(stage));
    childGraphs.clear();
    nativeHandles.clear();
}

const Data &HybridGraphCapture::AddExperts(const Data &input, const Data &index, const Data &score,
                                           std::vector<Data *> &weights, int layer,
                                           const std::function<void()> &runShared) {
    childGraphs.push_back(Close());
    auto output = std::make_unique<Data>();
    void *child = nullptr;
    if (disk) {
        child = disk->Add(input, index, score, *output, weights, 10.f);
        AssertInFastLLM(child, "HY4-Flash failed to prepare a disk FP8 expert graph.");
    } else {
        Start(input.dataDeviceIds.front());
        runShared();
        void *shared = Close();
        auto *handle = FastllmCudaCreateNativeMoeFullGraph(input, index, score, *output, weights.data(),
                                                           weights.size(), layer, shared, &child);
        AssertInFastLLM(handle && child, "HY4-Flash failed to prepare a native expert graph.");
        nativeHandles.push_back(handle);
    }
    childGraphs.push_back(child);
    compiled->outputs.push_back(std::move(output));
    Start(input.dataDeviceIds.front());
    return *compiled->outputs.back();
}
void HybridGraphCapture::Finish(const GraphKey &key, int kind, int rows) {
    if (!graphs)
        return;
    CloseStage();
    compiled->Launch();
    if (graphs->replays.size() >= 16)
        graphs->replays.clear();
    graphs->replays[key] = compiled;
    if (GetFastllmEnv().printProfile)
        fprintf(stderr,
                "[HY4 hybrid full graph] verify=%d rows=%d children=%zu expert_layers=%zu launch=%zu stages=%zu\n", kind,
                rows, compiled->children, compiled->experts, ++graphs->launches, compiled->stages.size());
}
} // namespace fastllm::hy4
