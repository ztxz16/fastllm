#pragma once
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace fastllm_native_prefill {
// An unset switch selects the validated automatic policy. Any explicit value
// other than 1/true disables it, including 0. This never bypasses CanRun.
inline bool ResolvePrefillSwitch(const char *value, bool automatic) {
    return value ? (!std::strcmp(value, "1") || !std::strcmp(value, "true")) : automatic;
}
inline bool AutomaticLinearPrefill(int bits, int major, int minor, int m, int n, int k) {
    if (major != 12 || minor != 0 || m > 4096) return false;
    // Qwen3.8-27B single-GPU merged projections, using N=output, K=input.
    if (bits == 8) {
        return m >= 32 && ((k == 5120 && (n == 16384 || n == 14336 || n == 34816)) ||
                          (n == 5120 && (k == 6144 || k == 17408)));
    }
    return bits == 4 && m >= 9 &&
           ((n == 34816 && k == 5120) || (n == 5120 && k == 17408));
}
inline bool AutomaticGdnPrepare(int major, int minor, int batch, int heads, int chunks,
                                int chunkSize, int headDim) {
    return major == 12 && minor == 0 && batch == 1 && heads == 48 &&
           chunks >= 1 && chunks <= 64 && chunkSize == 64 && headDim == 128;
}
inline bool LinearPrefillEnabled(int bits, int m, int n, int k) {
    const char *value = std::getenv(bits == 8 ? "FASTLLM_CUDA_NATIVE_FP8_PREFILL" :
                                             "FASTLLM_CUDA_NATIVE_NVFP4_PREFILL");
    if (value) return ResolvePrefillSwitch(value, false);
    // Reject decode and unvalidated shapes without querying the CUDA device.
    if (!AutomaticLinearPrefill(bits, 12, 0, m, n, k)) return false;
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
        return false;
    return AutomaticLinearPrefill(bits, major, minor, m, n, k);
}
} // namespace fastllm_native_prefill
