#pragma once

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudaworkspace.h"

namespace fastllm {
// Explicit tensor binding only: this scope does NOT redirect unrelated
// allocations made by attention, weight loaders, KV caches or graph warmup.
inline std::shared_ptr<CudaWorkspace> &CurrentPrefillWorkspace() {
    static thread_local std::shared_ptr<CudaWorkspace> workspace;
    return workspace;
}
class PrefillWorkspaceScope {
    std::shared_ptr<CudaWorkspace> previous;
public:
    explicit PrefillWorkspaceScope(std::shared_ptr<CudaWorkspace> workspace)
        : previous(std::move(CurrentPrefillWorkspace())) {
        CurrentPrefillWorkspace() = std::move(workspace);
    }
    ~PrefillWorkspaceScope() { CurrentPrefillWorkspace() = std::move(previous); }
    PrefillWorkspaceScope(const PrefillWorkspaceScope &) = delete;
    PrefillWorkspaceScope &operator=(const PrefillWorkspaceScope &) = delete;
};
inline void BindPrefillTemporaries(std::initializer_list<Data *> tensors) {
    for (auto *tensor : tensors) tensor->cudaWorkspace = CurrentPrefillWorkspace();
}

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
    data.cudaWorkspace.reset();
}
} // namespace fastllm
