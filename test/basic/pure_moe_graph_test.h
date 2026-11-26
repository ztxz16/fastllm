#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "devices/cuda/fastllm-cuda.cuh"

// Persistent buffers intentionally outlive several different input/route sets.
// Callers compare every replay against their independent format oracle.
struct PureMoeGraphTest {
    fastllm::Data input, index, score, output;
    int device = -1;
    void *handle = nullptr, *child = nullptr, *graph = nullptr, *exec = nullptr;
    static void Check(bool ok, const char *stage = "graph device nodes") { if (!ok) throw std::runtime_error(std::string("pure MoE full graph regression failed: ") + stage); }
    static void DeviceNodes(cudaGraph_t graph) {
        size_t count = 0; Check(cudaGraphGetNodes(graph, nullptr, &count) == cudaSuccess);
        std::vector<cudaGraphNode_t> nodes(count);
        Check(cudaGraphGetNodes(graph, nodes.data(), &count) == cudaSuccess);
        for (auto node : nodes) {
            cudaGraphNodeType type; Check(cudaGraphNodeGetType(node, &type) == cudaSuccess);
            Check(type != cudaGraphNodeTypeHost);
            if (type == cudaGraphNodeTypeGraph) {
                cudaGraph_t child; Check(cudaGraphChildGraphNodeGetGraph(node, &child) == cudaSuccess);
                DeviceNodes(child);
            }
        }
    }
    fastllm::Data &Run(const fastllm::Data &x, const fastllm::Data &i, const fastllm::Data &s,
            fastllm::Data **weights, int count, int layer) {
        device = FastllmCudaGetDevice();
        input.CopyFrom(x); index.CopyFrom(i); score.CopyFrom(s);
        if (!handle) {
            Check(FastllmCudaNativeMoeGraphOwner(weights, count, x.dims[0]), "owner");
            handle = FastllmCudaCreateNativeMoeFullGraph(input, index, score, output, weights, count, layer, nullptr, &child);
            Check(handle && child, "capture"); DeviceNodes(reinterpret_cast<cudaGraph_t>(child));
            void *initial = nullptr;
            Check(FastllmCudaGraphCompose(const_cast<const void *const *>(&child), 1, &graph, &initial), "compose");
            FastllmCudaGraphExecDestroy(initial);
            Check(FastllmCudaCreateNativeMoeReplayGraph(graph, &handle, 1, &exec), "instantiate replay");
        }
        Check(FastllmCudaLaunchNativeMoeFullGraph(exec, &handle, 1, false), "launch");
        return output;
    }
    ~PureMoeGraphTest() {
        if (device < 0) return;
        const int previous = FastllmCudaGetDevice(); FastllmCudaSetDevice(device);
        if (exec) FastllmCudaGraphExecDestroy(exec);
        if (graph) FastllmCudaGraphDestroy(graph);
        if (handle) FastllmCudaReleaseNativeMoeFullGraph(handle);
        FastllmCudaSetDevice(previous);
    }
};
