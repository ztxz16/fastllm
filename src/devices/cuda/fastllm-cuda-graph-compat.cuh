#pragma once
#include <cuda_runtime.h>

// CUDA 13 makes edge-data parameters part of the unversioned graph APIs.
// These adapters preserve the default full dependencies used by our graphs.
namespace fastllm_cuda_graph_compat {
#if CUDART_VERSION >= 11030
inline cudaError_t CaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *status,
        unsigned long long *id, cudaGraph_t *graph, const cudaGraphNode_t **deps, size_t *count) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, id, graph, deps, nullptr, count);
#else
    return cudaStreamGetCaptureInfo(stream, status, id, graph, deps, count);
#endif
}

inline cudaError_t UpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *deps,
        size_t count, unsigned int flags) {
#if CUDART_VERSION >= 13000
    return cudaStreamUpdateCaptureDependencies(stream, deps, nullptr, count, flags);
#else
    return cudaStreamUpdateCaptureDependencies(stream, deps, count, flags);
#endif
}
#endif

inline cudaError_t AddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
        const cudaGraphNode_t *to, size_t count) {
#if CUDART_VERSION >= 13000
    return cudaGraphAddDependencies(graph, from, to, nullptr, count);
#else
    return cudaGraphAddDependencies(graph, from, to, count);
#endif
}

inline cudaError_t GetDependencies(cudaGraphNode_t node, cudaGraphNode_t *deps, size_t *count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, deps, nullptr, count);
#else
    return cudaGraphNodeGetDependencies(node, deps, count);
#endif
}

inline cudaError_t GetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *deps, size_t *count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependentNodes(node, deps, nullptr, count);
#else
    return cudaGraphNodeGetDependentNodes(node, deps, count);
#endif
}
}  // namespace fastllm_cuda_graph_compat
