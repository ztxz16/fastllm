#include "devices/cuda/fastllm-mxfp4-ep-graph.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#if CUDA_VERSION >= 12090 && defined(__linux__)
#include "fastllm-mxfp4-ep-native.inc"
#else
struct FastllmMxfp4EpNativeState {};
std::shared_ptr<FastllmMxfp4EpNativeState> FastllmCudaCreateMxfp4EpNative(const std::vector<int> &,
    const std::vector<std::vector<std::vector<uint64_t>>> &) { return {}; }
void FastllmCudaBindMxfp4EpNativeWeights(FastllmMxfp4EpNativeState &, int) {}
std::vector<FastllmMxfp4EpGraph> FastllmCudaPrepareMxfp4EpNative(FastllmMxfp4EpNativeState &, int, int) { return {}; }
#endif

bool FastllmCudaMxfp4EpGraphSupported(const std::vector<int> &devices) {
    #if CUDA_VERSION >= 12090 && defined(__linux__)
    if (devices.size() != 4 && devices.size() != 8) return false;
    int planSms = 0;
    for (int device : devices) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || prop.major != 12 || prop.minor != 0)
            return false;
        if (planSms && prop.multiProcessorCount != planSms) return false;
        planSms = prop.multiProcessorCount;
        for (int peer : devices) if (peer != device) {
            int enabled = 0;
            if (cudaDeviceCanAccessPeer(&enabled, device, peer) != cudaSuccess || !enabled) return false;
        }
    }
    return std::ifstream(NativeEpPackage(devices) + "/manifest.json").good();
    #else
    return false;
    #endif
}

bool FastllmCudaMxfp4EpGraphCapturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(cudaStreamPerThread, &status) == cudaSuccess &&
           status != cudaStreamCaptureStatusNone;
}

__global__ void PrepareMxfp4EpRoutes(const int32_t *input, const float *weights,
        int64_t *output, float *outputWeights, int rows, int capacity, int topk,
        const int *liveKeys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity * topk) return;
    int row = i / topk;
    bool active = row < rows && (!liveKeys || liveKeys[row] > 0);
    output[i] = active ? input[i] : -1;
    outputWeights[i] = active ? weights[i] : 0.0f;
}

bool FastllmCudaReplayMxfp4EpGraph(const FastllmMxfp4EpGraph &graph,
        const void *input, const int32_t *indices, const float *scores, void *output,
        int rows, int hidden, int topk, const int *liveKeys) {
    if (!graph.graph || !graph.graphExec || rows < 0 || rows > graph.capacity) return false;
    auto stream = cudaStreamPerThread;
    if (rows && cudaMemcpyAsync(graph.input, input, (size_t)rows * hidden * 2,
                               cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return false;
    PrepareMxfp4EpRoutes<<<(graph.capacity * topk + 255) / 256, 256, 0, stream>>>(
        indices, scores, (int64_t *)graph.indices, (float *)graph.scores,
        rows, graph.capacity, topk, liveKeys);
    if (cudaGetLastError() != cudaSuccess) return false;
    CUstream caller = (CUstream)stream;
    CUstreamCaptureStatus status;
    cuuint64_t captureId;
    CUgraph parent;
    const CUgraphNode *dependencies;
    size_t dependencyCount;
#if CUDA_VERSION >= 13000
    CUresult result = cuStreamGetCaptureInfo(caller, &status, &captureId,
                                             &parent, &dependencies, nullptr, &dependencyCount);
#else
    CUresult result = cuStreamGetCaptureInfo(caller, &status, &captureId,
                                             &parent, &dependencies, &dependencyCount);
#endif
    if (result == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        // Same dependency protocol as FlashInfer's NativeGreenContextGraph:
        // join the preceding copies and make following copies depend on MoE.
        CUgraphNode child;
        result = cuGraphAddChildGraphNode(&child, parent, dependencies, dependencyCount, (CUgraph)graph.graph);
        if (result == CUDA_SUCCESS) {
#if CUDA_VERSION >= 13000
            result = cuStreamUpdateCaptureDependencies(caller, &child, nullptr, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES);
#else
            result = cuStreamUpdateCaptureDependencies(caller, &child, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES);
#endif
        }
    } else if (result == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_NONE) {
        result = cuGraphLaunch((CUgraphExec)graph.graphExec, caller);
    } else if (result == CUDA_SUCCESS) {
        result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    if (result != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(result, &message);
        fprintf(stderr, "SM120 MXFP4 EP graph launch failed: %s\n", message ? message : "unknown CUDA error");
        return false;
    }
    return !rows || cudaMemcpyAsync(output, graph.output, (size_t)rows * hidden * 2,
                                   cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
}
