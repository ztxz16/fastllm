#pragma once
#include <cuda_runtime.h>

namespace fastllm { namespace cuda {
// Ask for the ABI used by the caller. A compatible driver may be older
// than the toolkit and reject a lookup at the toolkit's latest version.
template<class Fn> bool DriverEntry(const char *name, unsigned abi, Fn &function) {
    void *entry = nullptr;
#if CUDART_VERSION >= 12050
    auto status = cudaGetDriverEntryPointByVersion(name, &entry, abi, cudaEnableDefault);
#else
    auto status = cudaGetDriverEntryPoint(name, &entry, cudaEnableDefault);
#endif
    function = reinterpret_cast<Fn>(entry);
    return status == cudaSuccess && function;
}
}} // namespace fastllm::cuda
