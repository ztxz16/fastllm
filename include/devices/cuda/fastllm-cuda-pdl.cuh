#pragma once

#include "fastllm-cuda.cuh"

namespace fastllm::cuda_pdl {

enum class Producer { VerifyExperts, DraftNorm, VerifyExpertReduce, SwaQkv, VerifyRouter, SwaPrepare, SwaOutput };

#if !defined(USE_ROCM) && CUDART_VERSION >= 12030
struct CaptureTail {
    unsigned long long id = 0;
    cudaGraphNode_t node = nullptr;
    Producer producer{};
};
inline thread_local CaptureTail marked;

inline CaptureTail CurrentTail(Producer producer) {
    cudaStreamCaptureStatus status{};
    unsigned long long id = 0;
    size_t count = 0;
    const cudaGraphNode_t *nodes = nullptr;
    const cudaGraphEdgeData *edges = nullptr;
#if CUDART_VERSION >= 13000
    auto rc = cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, &id, nullptr, &nodes, &edges, &count);
#else
    auto rc = cudaStreamGetCaptureInfo_v3(cudaStreamPerThread, &status, &id, nullptr, &nodes, &edges, &count);
#endif
    if (rc == cudaSuccess && status == cudaStreamCaptureStatusActive && count == 1 &&
        edges[0].type == cudaGraphDependencyTypeDefault)
        return {id, nodes[0], producer};
    return {};
}

inline void Mark(Producer producer) {
    marked = FastllmCudaGraphIsCapturingFast() ? CurrentTail(producer) : CaptureTail{};
}

// An intervening kernel, a stream join or a different capture keeps its original dependency.
inline bool Prepare(Producer producer) {
    if (!marked.node || marked.producer != producer) return false;
    auto previous = marked;
    marked = {};
    const auto tail = CurrentTail(producer);
    if (tail.id != previous.id || tail.node != previous.node) return false;
    cudaGraphEdgeData edge{};
    edge.type = cudaGraphDependencyTypeProgrammatic;
    edge.from_port = cudaGraphKernelNodePortLaunchCompletion;
#if CUDART_VERSION >= 13000
    return cudaStreamUpdateCaptureDependencies(cudaStreamPerThread, &previous.node, &edge, 1,
                                               cudaStreamSetCaptureDependencies) == cudaSuccess;
#else
    return cudaStreamUpdateCaptureDependencies_v2(cudaStreamPerThread, &previous.node, &edge, 1,
                                                  cudaStreamSetCaptureDependencies) == cudaSuccess;
#endif
}
#else
inline void Mark(Producer) {}
inline bool Prepare(Producer) { return false; }
#endif

__device__ __forceinline__ void Wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ uint4 ReadWeight(const void *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint4 value;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                 "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) : "l"(ptr) : "memory");
    return value;
#else
    return *static_cast<const uint4 *>(ptr);
#endif
}

}  // namespace fastllm::cuda_pdl
