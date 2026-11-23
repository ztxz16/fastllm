#pragma once
#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm-cuda-driver-entry.h"
#include <algorithm>

namespace fastllm { namespace cuda {
inline bool StripMoeTiming(cudaGraph_t graph, const std::vector<cudaEvent_t> &events) {
    // Batch-memop nodes have no CUDA Runtime node-type counterpart.
    // Query the Driver type while keeping the runtime graph handles.
    decltype(&cuGraphNodeGetType) nodeType = nullptr;
    if (!DriverEntry("cuGraphNodeGetType", 10000, nodeType)) return false;
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return false;
    for (auto node : nodes) {
        CUgraphNodeType type;
        if (nodeType(reinterpret_cast<CUgraphNode>(node), &type) != CUDA_SUCCESS) return false;
        if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
            cudaGraph_t child = nullptr;
            if (cudaGraphChildGraphNodeGetGraph(node, &child) != cudaSuccess ||
                !StripMoeTiming(child, events)) return false;
        } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
            cudaEvent_t event = nullptr;
            if (cudaGraphEventRecordNodeGetEvent(node, &event) != cudaSuccess) return false;
            if (std::find(events.begin(), events.end(), event) == events.end()) continue;
            size_t beforeCount = 0, afterCount = 0;
            if (FastllmCudaGraphNodeGetDependencies(node, nullptr, &beforeCount) != cudaSuccess ||
                FastllmCudaGraphNodeGetDependentNodes(node, nullptr, &afterCount) != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> before(beforeCount), after(afterCount);
            if (FastllmCudaGraphNodeGetDependencies(node, before.data(), &beforeCount) != cudaSuccess ||
                FastllmCudaGraphNodeGetDependentNodes(node, after.data(), &afterCount) != cudaSuccess) return false;
            for (auto next : after) {
                size_t n = 0;
                if (FastllmCudaGraphNodeGetDependencies(next, nullptr, &n) != cudaSuccess) return false;
                std::vector<cudaGraphNode_t> existing(n);
                if (FastllmCudaGraphNodeGetDependencies(next, existing.data(), &n) != cudaSuccess) return false;
                for (auto prev : before)
                    if (std::find(existing.begin(), existing.end(), prev) == existing.end() &&
                        FastllmCudaGraphAddDependencies(graph, &prev, &next, 1) != cudaSuccess) return false;
            }
            if (cudaGraphDestroyNode(node) != cudaSuccess) return false;
        }
    }
    return true;
}

// Instantiate the steady-state replay from the same capture. Only timing nodes
// are bypassed. External publication events and fork/join dependencies survive.
inline cudaGraphExec_t UntimedMoeGraph(cudaGraph_t source, const std::vector<cudaEvent_t> &events) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cudaGraphClone(&graph, source) == cudaSuccess && StripMoeTiming(graph, events) &&
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess;
    if (graph) cudaGraphDestroy(graph);
    if (!ok) {
        if (exec) cudaGraphExecDestroy(exec);
        cudaGetLastError();
        return nullptr;
    }
    return exec;
}
}} // namespace fastllm::cuda
