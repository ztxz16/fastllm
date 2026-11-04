#pragma once

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

namespace fastllm {
// Release after the last consumer on this thread's stream. A CPU MoE worker
// may allocate on a different stream: the pool event prevents early reuse.
// Captured, borrowed and persistent tensors must be excluded by the caller.
inline void ReleasePrefillTemporary(Data &data) {
    if (data.isFake) return;
    if (data.cudaData != nullptr && !data.cudaDataBorrowed) {
        const int original = FastllmCudaGetDevice();
        const int device = data.dataDeviceIds.empty() ? original : data.dataDeviceIds[0];
        FastllmCudaSetDevice(device);
        if (!data.directMemory && FastllmCudaFreeAfterCurrentThreadStream(data.cudaData)) {
            data.cudaData = nullptr;
        } else {
            ForceDeviceSync();
        }
        FastllmCudaSetDevice(original);
    }
    data.FreeSpace();
    data.expansionDims.clear();
    data.dims.clear();
    data.strides.clear();
}
} // namespace fastllm
