#pragma once
#include "../fastllm-cuda-graph-compat.cuh"
// The caller supplies only mHC segments whose pre/post/comb outputs are first
// consumed at join[i], and the hidden input remains unchanged until that join.
// Each side branch owns its mix scratch; graph-pool temporaries may be reused.
namespace fastllm_v41_hc_graph {
// Keep the combined kernel ABI so its captured argument storage can be reused.
// The weight/mix arguments are unused by the normalization-only replacement.
template <bool Quantize>
__global__ void NormOnly(const __nv_bfloat16 *x, const float *, float *, const float *previousPre,
                         const float *normWeight, __nv_bfloat16 *out, __nv_bfloat16 *quantized, float eps) {
    V41HcPreNormBody<1024, 3, false, Quantize>(x, previousPre, normWeight, out, 4, 5120, eps, 0, nullptr,
                                               nullptr, nullptr, 0, 0.f, nullptr, nullptr, nullptr,
                                               quantized);
}
inline bool Compose(const void *const *children, int count, const int *start, const int *join, int branches,
                    float *saved, void **outGraph, void **outExec) {
    if (!children || count <= 0 || !start || !join || branches <= 0 || !saved || !outGraph || !outExec)
        return false;
    *outGraph = nullptr;
    *outExec = nullptr;
    std::vector<int> branchAt(count, -1);
    for (int b = 0; b < branches; ++b) {
        if (start[b] < 0 || start[b] >= count || join[b] <= start[b] || join[b] >= count ||
            branchAt[start[b]] >= 0)
            return false;
        branchAt[start[b]] = b;
    }
    cudaGraph_t graph = nullptr, cloned = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<cudaGraphNode_t> finishNodes(branches, nullptr);
    cudaGraphNode_t previous = nullptr;
    auto fail = [&] {
        if (cloned)
            cudaGraphDestroy(cloned);
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
        return false;
    };
    if (cudaGraphCreate(&graph, 0) != cudaSuccess)
        return false;
    for (int i = 0; i < count; ++i) {
        auto child = reinterpret_cast<cudaGraph_t>(const_cast<void *>(children[i]));
        const int b = branchAt[i];
        cudaGraphNode_t dotNode = nullptr;
        if (b >= 0) {
            if (cudaGraphClone(&cloned, child) != cudaSuccess) {
                return fail();
            }
            size_t n = 0;
            if (cudaGraphGetNodes(cloned, nullptr, &n) != cudaSuccess) {
                return fail();
            }
            std::vector<cudaGraphNode_t> nodes(n);
            if (cudaGraphGetNodes(cloned, nodes.data(), &n) != cudaSuccess) {
                return fail();
            }
            cudaGraphNode_t target = nullptr, combined = nullptr;
            cudaKernelNodeParams params{}, combinedParams{};
            bool quantize = false;
            for (auto node : nodes) {
                cudaGraphNodeType type;
                if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
                    return fail();
                }
                if (type != cudaGraphNodeTypeKernel)
                    continue;
                cudaKernelNodeParams candidate{};
                if (cudaGraphKernelNodeGetParams(node, &candidate) != cudaSuccess) {
                    return fail();
                }
                if (candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<false>) ||
                    candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<true>)) {
                    if (combined) {
                        return fail();
                    }
                    combined = node;
                    combinedParams = candidate;
                    quantize = candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<true>);
                }
                if (candidate.func == reinterpret_cast<void *>(V41HcDecodeFinish)) {
                    if (target) {
                        return fail();
                    }
                    target = node;
                    params = candidate;
                }
            }
            if (!target || !combined || !combinedParams.kernelParams || combinedParams.extra ||
                combinedParams.gridDim.x != 26 || combinedParams.gridDim.y != 1 ||
                combinedParams.gridDim.z != 1 || combinedParams.blockDim.x != 1024 ||
                combinedParams.blockDim.y != 1 || combinedParams.blockDim.z != 1 || !params.kernelParams ||
                params.extra || params.gridDim.x != 1 || params.gridDim.y != 1 || params.gridDim.z != 1 ||
                params.blockDim.x != 32 || params.blockDim.y != 1 || params.blockDim.z != 1 ||
                params.sharedMemBytes != 0) {
                return fail();
            }
            const float *source = *static_cast<const float **>(params.kernelParams[0]);
            if (source != *static_cast<float **>(combinedParams.kernelParams[2])) {
                return fail();
            }
            size_t dependencies = 0;
            if (fastllm_cuda_graph_compat::GetDependencies(target, nullptr, &dependencies) != cudaSuccess ||
                dependencies != 1) {
                return fail();
            }
            cudaGraphNode_t dependency;
            if (fastllm_cuda_graph_compat::GetDependencies(target, &dependency, &dependencies) != cudaSuccess ||
                dependency != combined) {
                return fail();
            }
            float *destination = saved + b * 32;
            const auto *x = *static_cast<const __nv_bfloat16 **>(combinedParams.kernelParams[0]);
            const float *fn = *static_cast<const float **>(combinedParams.kernelParams[1]);
            int flatDim = 20480;
            void *dotArgs[] = {&x, &fn, &flatDim, &destination};
            cudaKernelNodeParams dot{};
            dot.func = reinterpret_cast<void *>(V41HcDecodeDotsSm90<16>);
            dot.gridDim = dim3(25);
            dot.blockDim = dim3(256);
            dot.kernelParams = dotArgs;
            if (cudaGraphAddKernelNode(&dotNode, graph, nullptr, 0, &dot) != cudaSuccess) {
                return fail();
            }
            void *args[10];
            for (int a = 0; a < 10; ++a)
                args[a] = params.kernelParams[a];
            args[0] = &destination;
            params.kernelParams = args;
            // AddKernelNode copies argument values before SetParams invalidates
            // the original node-owned argument storage.
            if (cudaGraphAddKernelNode(&finishNodes[b], graph, &dotNode, 1, &params) != cudaSuccess) {
                return fail();
            }
            combinedParams.func = quantize ? reinterpret_cast<void *>(NormOnly<true>)
                                           : reinterpret_cast<void *>(NormOnly<false>);
            combinedParams.gridDim = dim3(1);
            if (cudaGraphKernelNodeSetParams(combined, &combinedParams) != cudaSuccess) {
                return fail();
            }
            size_t successors = 0;
            if (fastllm_cuda_graph_compat::GetDependentNodes(target, nullptr, &successors) != cudaSuccess) {
                return fail();
            }
            std::vector<cudaGraphNode_t> dependent(successors);
            if (fastllm_cuda_graph_compat::GetDependentNodes(target, dependent.data(), &successors) != cudaSuccess) {
                return fail();
            }
            // All work following Finish still waits for normalization. The
            // caller joins the detached mix before any use of its outputs.
            if (cudaGraphDestroyNode(target) != cudaSuccess) {
                return fail();
            }
            for (auto node : dependent) {
                if (fastllm_cuda_graph_compat::AddDependencies(cloned, &combined, &node, 1) != cudaSuccess) {
                    return fail();
                }
            }
            child = cloned;
        }
        std::vector<cudaGraphNode_t> deps;
        if (previous)
            deps.push_back(previous);
        for (int j = 0; j < branches; ++j)
            if (join[j] == i)
                deps.push_back(finishNodes[j]);
        cudaGraphNode_t next = nullptr;
        if (cudaGraphAddChildGraphNode(&next, graph, deps.data(), deps.size(), child) != cudaSuccess) {
            return fail();
        }
        if (cloned) {
            cudaGraphDestroy(cloned);
            cloned = nullptr;
        }
        if (b >= 0 && fastllm_cuda_graph_compat::AddDependencies(graph, &next, &dotNode, 1) != cudaSuccess) {
            return fail();
        }
        previous = next;
    }
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
        return fail();
    }
    *outGraph = graph;
    *outExec = exec;
    return true;
}
} // namespace fastllm_v41_hc_graph
