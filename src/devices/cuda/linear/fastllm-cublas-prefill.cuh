#pragma once

#include "fastllm-cuda.cuh"
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>

// FP8 and NVFP4 share one per-device handle and scratch budget. Inline linkage
// intentionally shares the registry across the two CUDA translation units.
namespace fastllm_cuda_prefill {

struct State {
    std::mutex mutex;
    half *scratch = nullptr;
    void *workspace = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t completed = nullptr;
    size_t capacity = 0;
    bool initialized = false;
    bool fp8Logged = false;
    bool nvfp4Logged = false;
    float *zero = nullptr;
    cublasPointerMode_t pointerMode = CUBLAS_POINTER_MODE_HOST;
};

inline void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Quantized prefill CUDA error: %s\n", cudaGetErrorString(status));
        throw("quantized prefill cuda error");
    }
}

inline void CheckCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Quantized prefill cuBLAS error: %d\n", int(status));
        throw("quantized prefill cublas error");
    }
}

inline State *GetState(int arch) {
    if (arch != 75) return nullptr;
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    // The global lock protects only the map. Never allocate under a cross-rank
    // lock: CUDA allocation may wait for outstanding NCCL work on this device.
    static std::mutex mapMutex;
    static std::map<int, std::unique_ptr<State>> states;
    State *state;
    {
        std::lock_guard<std::mutex> guard(mapMutex);
        auto &entry = states[device];
        if (!entry) entry = std::make_unique<State>();
        state = entry.get();
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->initialized) return state;
    cudaStreamCaptureStatus capture;
    CheckCuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    if (!FastllmCudaGetNcclForceSync() || capture != cudaStreamCaptureStatusNone) {
        return nullptr; // Never grow scratch while serving or capturing a graph.
    }
    constexpr size_t workspaceBytes = 8 * 1024 * 1024;
    // Match the existing GGUF dequant workspace convention: per-device model
    // operations run in order on the worker's per-thread default stream.
    // This is not a lease for concurrent independent streams/models.
    // Only the float arena holds disposable per-call intermediates; the int
    // arena contains cached attention plans and must not be overwritten.
    size_t available = 0;
    state->scratch = static_cast<half *>(FastllmCudaGetFlashInferFloatWorkspace(&available));
    state->capacity = available;
    if (state->scratch == nullptr || state->capacity == 0) return nullptr;
    // The process-lifetime float arena belongs to FlashInfer. Never free it or
    // enlarge it here. Keep cuBLAS scratch separate because both ranges are
    // live during GEMM. Materialize these before KV page budget calibration.
    CheckCuda(cudaMalloc(&state->workspace, workspaceBytes + sizeof(float)));
    state->zero = reinterpret_cast<float *>(static_cast<char *>(state->workspace) + workspaceBytes);
    CheckCuda(cudaMemsetAsync(state->zero, 0, sizeof(float), cudaStreamPerThread));
    CheckCublas(cublasCreate(&state->handle));
    CheckCublas(cublasSetStream(state->handle, cudaStreamPerThread));
    CheckCublas(cublasSetMathMode(state->handle, cublasMath_t(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    CheckCublas(cublasSetWorkspace(state->handle, state->workspace, workspaceBytes));
    CheckCuda(cudaEventCreateWithFlags(&state->completed, cudaEventDisableTiming));
    // Every returned state has a recorded event, including zero initialization.
    // Callers can always wait before reusing the handle and scratch.
    CheckCuda(cudaEventRecord(state->completed, cudaStreamPerThread));
    state->initialized = true;
    std::fprintf(stderr,
        "[Quantized prefill cuBLAS] GPU %d: borrowed FlashInfer float %zu MiB scratch at %p "
        "+ 8 MiB cuBLAS workspace during force-sync warmup.\n",
        device, state->capacity / 1024 / 1024, static_cast<void *>(state->scratch));
    return state;
}

inline void SetPointerMode(State *state, cublasPointerMode_t mode) {
    // Call only while holding state->mutex. FP16 GEMM uses host half scalars;
    // NVFP4 FP32 GEMM uses the device tensor scale and zero scalar.
    if (state->pointerMode != mode) {
        CheckCublas(cublasSetPointerMode(state->handle, mode));
        state->pointerMode = mode;
    }
}

} // namespace fastllm_cuda_prefill
