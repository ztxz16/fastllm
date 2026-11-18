#pragma once
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>

#ifndef USE_ROCM
bool FastllmCudaMoeFP8CacheComputeLimited(
    const fastllm::Data &input, fastllm::Data &gateOutput, fastllm::Data &output,
    const FastllmCudaMoeFP8CacheView &view, const int32_t *slots, const float *scores,
    int topk, float *perExpert, const FastllmCudaMoeStageEvents &events, float swigluLimit);

bool FastllmCudaMoeFP8VerifyProjections(const float *input, float *gate, float *partial,
    const FastllmCudaMoeFP8CacheView &view, const int32_t *slots, int rows, int topk,
    int hidden, int inter, float swigluLimit);

inline bool FastllmMoeWaitDown(const FastllmCudaMoeStageEvents &events) {
    if (events.gateDone && cudaEventRecord(static_cast<cudaEvent_t>(events.gateDone),
            cudaStreamPerThread) != cudaSuccess) return false;
    if (events.downReady && cudaStreamWaitEvent(cudaStreamPerThread,
            static_cast<cudaEvent_t>(events.downReady), 0) != cudaSuccess) return false;
    return !events.downStart || cudaEventRecord(static_cast<cudaEvent_t>(events.downStart),
        cudaStreamPerThread) == cudaSuccess;
}
#endif
