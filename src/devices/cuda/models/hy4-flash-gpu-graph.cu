#include "../../../models/hy4_flash_gpu_graph.h"
#include "../../../models/hy4_flash_ops.h"
#include "hy4-flash-cuda.cuh"

namespace fastllm::hy4 {
bool ReplayGpu(GpuGraphs &graphs, const std::vector<uintptr_t> &key, int kind, int rows) {
    auto found = graphs.replays.find(key);
    if (found == graphs.replays.end())
        return false;
    Check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(found->second.get()), cudaStreamPerThread));
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 GPU replay] verify=%d rows=%d launch=%zu\n", kind, rows, ++graphs.launches);
    return true;
}
void GpuGraphCapture::Finish(const GraphKey &key, int kind, int rows) {
    if (!graphs)
        return;
    Close();
    cudaGraph_t parent = nullptr;
    Check(cudaGraphCreate(&parent, 0));
    struct Owner {
        cudaGraph_t graph;
        ~Owner() {
            cudaGraphDestroy(graph);
        }
    } owner{parent};
    cudaGraphNode_t prior = nullptr;
    for (void *c : children) {
        cudaGraphNode_t next = nullptr;
        auto child = static_cast<cudaGraph_t>(c);
        Check(cudaGraphAddChildGraphNode(&next, parent, prior ? &prior : nullptr, prior ? 1 : 0, child));
        prior = next;
    }
    cudaGraphExec_t executable = nullptr;
    Check(cudaGraphInstantiate(&executable, parent, nullptr, nullptr, 0));
    auto ownerExec = std::shared_ptr<void>(
        executable, [](void *p) { cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(p)); });
    if (graphs->replays.size() >= 16)
        graphs->replays.clear();
    graphs->replays[key] = std::move(ownerExec);
    Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    // Consumers can use the resulting CUDA tensors immediately on this stream.
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 full graph] verify=%d rows=%d stages=%zu capture=1 launch=%zu\n", kind, rows,
                children.size(), ++graphs->launches);
}

bool EnableGraphPeers(const std::vector<int> &devices) {
    for (size_t i = 1; i < devices.size(); ++i) {
        Select(devices[i]);
        int supported = 0;
        Check(cudaDeviceCanAccessPeer(&supported, devices[i], devices[i - 1]));
        if (!supported)
            return false;
        auto err = cudaDeviceEnablePeerAccess(devices[i - 1], 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
        else
            Check(err);
    }
    return true;
}
} // namespace fastllm::hy4
