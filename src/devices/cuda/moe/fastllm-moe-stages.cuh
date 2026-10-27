#pragma once
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>

#ifndef USE_ROCM
inline bool FastllmMoeWaitDown(const FastllmCudaMoeStageEvents &events) {
    if (events.gateDone && cudaEventRecord(static_cast<cudaEvent_t>(events.gateDone),
            cudaStreamPerThread) != cudaSuccess) return false;
    if (events.downReady && cudaStreamWaitEvent(cudaStreamPerThread,
            static_cast<cudaEvent_t>(events.downReady), 0) != cudaSuccess) return false;
    return !events.downStart || cudaEventRecord(static_cast<cudaEvent_t>(events.downStart),
        cudaStreamPerThread) == cudaSuccess;
}
#endif
