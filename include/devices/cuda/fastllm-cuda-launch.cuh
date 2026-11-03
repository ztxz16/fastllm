#pragma once
#include <cuda_runtime.h>

namespace fastllm { namespace cuda_launch {
// PDL hides launch/prologue latency without changing the dependency or
// arithmetic order. Every thread waits before reading predecessor outputs;
// all CTAs trigger before any early return. Ordinary stream launches and
// pre-SM90 code retain the original ordering.
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
__device__ __forceinline__ void DependencyStart() {
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <auto Kernel, class... Args>
bool Launch(dim3 grid, dim3 block, size_t shared, cudaStream_t stream, Args... args) {
#if CUDART_VERSION >= 12000
    // The cache belongs to each kernel specialization and host thread. Checking
    // PTX as well as the binary excludes SM80 PTX JITed on an SM90 device: its
    // device-side wait/trigger was compiled out.
    static thread_local int previousDevice = -1;
    static thread_local bool capable = false;
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    if (device != previousDevice) {
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, Kernel) != cudaSuccess)
            return false;
        capable = attr.binaryVersion >= 90 && attr.ptxVersion >= 90;
        previousDevice = device;
    }
    if (capable) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = shared;
        cfg.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        if (cudaLaunchKernelEx(&cfg, Kernel, args...) != cudaSuccess)
            return false;
        return cudaGetLastError() == cudaSuccess;
    }
#endif
    Kernel<<<grid, block, shared, stream>>>(args...);
    return cudaGetLastError() == cudaSuccess;
}

}} // namespace fastllm::cuda_launch
