#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <limits>

namespace fastllm::cuda::low_latency {
struct ConstBuffer { const void *data = nullptr; size_t bytes = 0; };
struct Buffer { void *data = nullptr; size_t bytes = 0; };

// A caller may cap per-CTA resources to reserve occupancy. This never exceeds
// the physical device limit and does not change the logical batch or workspace.
struct PrepareOptions {
    size_t sharedMemoryLimit = std::numeric_limits<size_t>::max();
};

// Only pre-launch capability misses allow another implementation to run.
// Allocation, driver and asynchronous execution errors must reach the caller.
inline bool BackendUnavailable(cudaError_t error) {
    return error == cudaErrorNotSupported || error == cudaErrorNoKernelImageForDevice ||
           error == cudaErrorInvalidDeviceFunction;
}
} // namespace fastllm::cuda::low_latency
