#pragma once
#include <cuda_runtime.h>

// CUDA 13 adds an edge metadata argument to the ordinary graph APIs.
inline cudaError_t FastllmCudaGraphNodeGetDependencies(cudaGraphNode_t node,
        cudaGraphNode_t *dependencies, size_t *count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, dependencies, nullptr, count);
#else
    return cudaGraphNodeGetDependencies(node, dependencies, count);
#endif
}

inline cudaError_t FastllmCudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
        cudaGraphNode_t *dependents, size_t *count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependentNodes(node, dependents, nullptr, count);
#else
    return cudaGraphNodeGetDependentNodes(node, dependents, count);
#endif
}

inline cudaError_t FastllmCudaGraphAddDependencies(cudaGraph_t graph,
        const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t count) {
#if CUDART_VERSION >= 13000
    return cudaGraphAddDependencies(graph, from, to, nullptr, count);
#else
    return cudaGraphAddDependencies(graph, from, to, count);
#endif
}
