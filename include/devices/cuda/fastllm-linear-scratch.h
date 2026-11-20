#pragma once

namespace fastllm {
class Data;

// A caller may lend a dead, device-local tensor to Linear on its rank's stream.
// The owner retains the allocation; captured graphs use their own pinned lease.
class CudaLinearScratchScope {
public:
    explicit CudaLinearScratchScope(const Data *scratch) : previous(current) { current = scratch; }
    ~CudaLinearScratchScope() { current = previous; }
    CudaLinearScratchScope(const CudaLinearScratchScope &) = delete;
    CudaLinearScratchScope &operator=(const CudaLinearScratchScope &) = delete;
    static const Data *Get() { return current; }

private:
    inline static thread_local const Data *current = nullptr;
    const Data *previous;
};
}
