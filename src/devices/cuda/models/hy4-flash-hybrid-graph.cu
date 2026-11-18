#include "../../../models/hy4_flash_hybrid_graph.h"
#include "../../../models/hy4_flash_ops.h"
#include "hy4-flash-cuda.cuh"
#include "utils/hybrid_moe.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
namespace fastllm::hy4 {
struct HybridExecutable {
    std::vector<std::shared_ptr<Data>> outputs;
    std::shared_ptr<DiskMoeV41Graph> disk;
    HybridMoeGraph graph;
    bool Launch() {
        return disk ? disk->Launch([this] { return graph.Launch(); }) : graph.Launch();
    }
};
bool ReplayHybrid(HybridGraphs &graphs, const std::vector<uintptr_t> &key, int kind, int rows) {
    auto found = graphs.replays.find(key);
    if (found == graphs.replays.end())
        return false;
    if (!found->second->Launch())
        throw std::runtime_error("HY4 native MoE graph replay failed");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 hybrid replay] verify=%d rows=%d launch=%zu\n", kind, rows, ++graphs.launches);
    return true;
}
HybridGraphCapture::HybridGraphCapture(HybridGraphs *graphs, bool diskWeights)
    : CaptureSegments(graphs != nullptr), graphs(graphs),
      disk(graphs && diskWeights ? std::make_shared<DiskMoeV41Graph>() : nullptr) {}

HybridGraphCapture::~HybridGraphCapture() {
    Abort();
    for (void *handle : nativeHandles)
        FastllmCudaReleaseNativeMoeFullGraph(handle);
}

const Data &HybridGraphCapture::AddExperts(const Data &input, const Data &index, const Data &score,
                                           std::vector<Data *> &weights, int layer) {
    childGraphs.push_back(Close());
    auto output = std::make_shared<Data>();
    void *child = nullptr;
    if (disk) {
        child = disk->Add(input, index, score, *output, weights, 10.f);
        AssertInFastLLM(child, "HY4-Flash failed to prepare a disk FP8 expert graph.");
    } else {
        auto *handle = FastllmCudaCreateNativeMoeFullGraph(input, index, score, *output, weights.data(),
                                                           weights.size(), layer, nullptr, &child);
        AssertInFastLLM(handle && child, "HY4-Flash failed to prepare a native FP8 expert graph.");
        nativeHandles.push_back(handle);
    }
    childGraphs.push_back(child);
    outputs.push_back(std::move(output));
    Start(input.dataDeviceIds.front());
    return *outputs.back();
}
void HybridGraphCapture::Finish(const GraphKey &key, int kind, int rows) {
    if (!graphs)
        return;
    childGraphs.push_back(Close());
    auto compiled = std::make_shared<HybridExecutable>();
    compiled->disk = std::move(disk);
    const size_t count = compiled->disk ? (childGraphs.size() - 1) / 2 : nativeHandles.size();
    if (!compiled->graph.Compose(childGraphs, std::move(nativeHandles)) || !compiled->Launch())
        throw std::runtime_error("HY4-Flash hybrid graph composition or launch failed");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    compiled->outputs = std::move(outputs);
    if (graphs->replays.size() >= 16)
        graphs->replays.clear();
    graphs->replays[key] = std::move(compiled);
    if (GetFastllmEnv().printProfile)
        fprintf(stderr,
                "[HY4 hybrid full graph] verify=%d rows=%d children=%zu expert_layers=%zu launch=%zu\n", kind,
                rows, childGraphs.size(), count, ++graphs->launches);
}
} // namespace fastllm::hy4
