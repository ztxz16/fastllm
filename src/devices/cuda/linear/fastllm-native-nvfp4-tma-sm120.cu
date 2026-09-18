// SM120 native NVFP4 dispatch. Kernel implementation is self-contained.
#include "fastllm-native-nvfp4-tma.cuh"
#include "fastllm-native-nvfp4-fresh-sm120.cuh"

namespace fastllm_native_prefill {
bool Nvfp4TmaCanRun(int m, int n, int k, int mode) {
    const char *flag = std::getenv("FASTLLM_CUDA_NATIVE_NVFP4_TMA");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    return fresh_nvfp4::CanRun(m, n, k, mode);
}

bool TryNvfp4Tma(const uint8_t *x, const uint8_t *xs, const uint8_t *weight, const uint8_t *weightScales,
                 const float *rowScales, const float *weightGlobal, half *output, int m, int n, int k,
                 int mode) {
    // All rejection paths precede any output write. The caller can regenerate
    // cuBLASLt activation scales and take its existing generic fallback.
    if (!Nvfp4TmaCanRun(m, n, k, mode) || !x || !xs || !weight || !weightScales || !rowScales ||
        !weightGlobal || !output || uintptr_t(x) % 128 || uintptr_t(xs) % 16 || uintptr_t(weight) % 128 ||
        uintptr_t(weightScales) % 16 || uintptr_t(output) % 16 || uintptr_t(rowScales) % 4 ||
        uintptr_t(weightGlobal) % 4)
        return false;
    return fresh_nvfp4::Run(x, xs, weight, weightScales, rowScales, weightGlobal, output, m, n, k, mode);
}
} // namespace fastllm_native_prefill
