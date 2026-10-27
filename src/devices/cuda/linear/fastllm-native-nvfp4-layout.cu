#include "devices/cuda/fastllm-cuda-native-prefill.h"
#include "fastllm-cuda.cuh"
#include "fastllm-native-lowbit-prefill.cuh"
#include <cmath>
#include <mutex>
#include <stdexcept>
#if CUDART_VERSION >= 12080
#include "fastllm-native-nvfp4-layout.cuh"
namespace fastllm {
namespace {
struct Temporary {
    uint8_t *pointer;
    explicit Temporary(size_t bytes) : pointer(static_cast<uint8_t *>(FastllmCudaMalloc(bytes))) {}
    ~Temporary() {
        if (pointer)
            FastllmCudaForceFree(pointer);
    }
};
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) : previous(FastllmCudaGetDevice()) { FastllmCudaSetDevice(device); }
    ~DeviceGuard() { FastllmCudaSetDevice(previous); }
};
void Check(cudaError_t e, const char *op) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
bool Dense(const Data &d, int device) {
    if (d.dataDevice != DataDevice::CUDA || !d.cudaData || d.multiDeviceData || d.dims.empty() ||
        d.strides.size() != d.dims.size() ||
        (!d.dataDeviceIds.empty() && (d.dataDeviceIds.size() != 1 || d.dataDeviceIds[0] != device)))
        return false;
    uint64_t stride = 1;
    for (int i = int(d.dims.size()) - 1; i >= 0; i--) {
        if (d.dims[i] <= 0 || d.strides[i] != stride)
            return false;
        stride *= d.dims[i];
    }
    return true;
}
float *Global(Data &w) {
    return reinterpret_cast<float *>(static_cast<uint8_t *>(w.cudaData) +
                                     size_t(w.dims[0]) * w.dims[1] * 9 / 16);
}
bool Prepare(Data &w) {
    if (w.cudaNativeNvfp4Layout)
        return true;
    if (!fastllm_native_prefill::Enabled("FASTLLM_CUDA_NVFP4_NATIVE_LAYOUT") ||
        !FastllmCudaGetNcclForceSync() || w.IsRepacked || w.dataType != DataType::NVFP4_BLOCK_16 ||
        w.blockM != 16 || w.blockK != 1 || !w.isModelWeight || w.isFake || w.cudaDataBorrowed ||
        w.dims.size() != 2)
        return false;
    int N = w.dims[0], K = w.dims[1], device = 0, major = 0;
    if (!((N == 34816 && K == 5120) || (N == 5120 && K == 17408)) || cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major != 12 || !Dense(w, device))
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, fastllm_native_layout::Gemv<0>) != cudaSuccess ||
        attr.binaryVersion < 120) {
        cudaGetLastError();
        return false;
    }
    static std::mutex prepareMutex;
    std::lock_guard<std::mutex> lock(prepareMutex);
    if (w.cudaNativeNvfp4Layout)
        return true;
    float global = INFINITY;
    for (float v : w.scales)
        if (std::isfinite(v) && v > 0)
            global = std::min(global, v);
    if (!std::isfinite(global) || !std::isfinite(global * 128.f))
        return false;
    size_t codes = size_t(N) * K / 2, bytes = size_t(N) * K * 9 / 16;
    if (bytes + 8 > w.GetBytes())
        return false;
    Temporary temporary(bytes + 8);
    uint8_t *tmp = temporary.pointer;
    if (!tmp)
        return false;
    int *bad = reinterpret_cast<int *>(tmp + bytes + 4), invalid = 0;
    Check(cudaMemsetAsync(bad, 0, 4, cudaStreamPerThread), "native layout scale check init");
    fastllm_native_layout::Pack<<<(size_t(N) * K / 16 + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(w.cudaData), tmp, tmp + codes, N, K, global, bad);
    Check(cudaGetLastError(), "native layout pack");
    Check(cudaMemcpyAsync(&invalid, bad, 4, cudaMemcpyDeviceToHost, cudaStreamPerThread),
          "native layout scale check");
    Check(cudaStreamSynchronize(cudaStreamPerThread), "native layout pack sync");
    if (invalid)
        return false;
    // Only now overwrite the original allocation; no per-layer second copy survives.
    float processed = global * 128.f;
    Check(cudaMemcpyAsync(w.cudaData, tmp, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread),
          "native layout install");
    Check(cudaMemcpyAsync(Global(w), &processed, 4, cudaMemcpyHostToDevice, cudaStreamPerThread),
          "native layout global scale");
    Check(cudaStreamSynchronize(cudaStreamPerThread), "native layout install sync");
    w.IsRepacked = true;
    w.cudaNativeNvfp4Layout = true;
    return true;
}
void Gemv(const half *x, Data &w, const half *bias, half *out, int M, int mode) {
    int N = w.dims[0], K = w.dims[1], width = mode == 1 ? N / 2 : N;
    const auto *codes = static_cast<const uint8_t *>(w.cudaData), *scales = codes + size_t(N) * K / 2;
    if (M == 1 && (K == 5120 || K == 17408)) {
        dim3 tunedGrid((width + 127) / 128 * 16, M);
#define LAUNCH(KSIZE)                                                                                        \
    if (mode == 1)                                                                                           \
        fastllm_native_layout::Gemv<1, 8, KSIZE>                                                             \
            <<<tunedGrid, 256, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);   \
    else if (mode == 2)                                                                                      \
        fastllm_native_layout::Gemv<2, 8, KSIZE>                                                             \
            <<<tunedGrid, 256, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);   \
    else                                                                                                     \
        fastllm_native_layout::Gemv<0, 8, KSIZE>                                                             \
            <<<tunedGrid, 256, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);
        if (K == 5120) {
            LAUNCH(5120)
        } else {
            LAUNCH(17408)
        }
#undef LAUNCH
        Check(cudaGetLastError(), "native layout tuned GEMV");
        return;
    }
    dim3 grid((width + 127) / 128 * 32, M);
    if (mode == 1)
        fastllm_native_layout::Gemv<1>
            <<<grid, 128, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);
    else if (mode == 2)
        fastllm_native_layout::Gemv<2>
            <<<grid, 128, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);
    else
        fastllm_native_layout::Gemv<0>
            <<<grid, 128, 0, cudaStreamPerThread>>>(x, codes, scales, Global(w), bias, out, M, N, K);
    Check(cudaGetLastError(), "native layout GEMV");
}
bool Prefill(const half *x, Data &w, const half *bias, half *out, int M, int mode) {
    int N = w.dims[0], K = w.dims[1];
    auto *codes = static_cast<const uint8_t *>(w.cudaData);
    return fastllm_native_prefill::Fp4(x, reinterpret_cast<const uint32_t *>(codes),
                                       codes + size_t(N) * K / 2, Global(w), bias, out, M, N, N, K, mode,
                                       true);
}
static __global__ void Bias(half *y, const half *bias, int M, int N) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(M) * N)
        y[i] = __hadd(y[i], bias[i % N]);
}
void Fallback(const half *x, Data &w, const half *bias, half *out, int M) {
    cudaStreamCaptureStatus capture;
    Check(cudaStreamIsCapturing(cudaStreamPerThread, &capture), "native fallback capture check");
    if (M <= 8 || capture != cudaStreamCaptureStatusNone) {
        Gemv(x, w, bias, out, M, 0);
        return;
    }
    int N = w.dims[0], K = w.dims[1];
    size_t available = 0;
    bool own = false;
    half *tmp = static_cast<half *>(
        FastllmBorrowDequantScratch(std::min<size_t>(size_t(N) * K * 2, 32ull << 20), &available, &own));
    if (!tmp || available < size_t(K) * 2) {
        if (tmp)
            FastllmReleaseDequantScratch(tmp, own);
        Gemv(x, w, bias, out, M, 0);
        return;
    }
    int chunk = std::min<size_t>(N, available / (K * 2));
    const auto *codes = static_cast<const uint8_t *>(w.cudaData), *scales = codes + size_t(N) * K / 2;
    float one = 1, zero = 0;
    auto handle = getFastllmCublasHandle();
    for (int offset = 0; offset < N; offset += chunk) {
        int count = std::min(chunk, N - offset);
        fastllm_native_layout::Dequant<<<(size_t(count) * K / 2 + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            codes, scales, Global(w), tmp, offset, count, K);
        Check(cudaGetLastError(), "native layout fallback dequant");
        auto e = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, count, M, K, &one, tmp, CUDA_R_16F, K, x,
                              CUDA_R_16F, K, &zero, out + offset, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
        if (e != CUBLAS_STATUS_SUCCESS) {
            FastllmReleaseDequantScratch(tmp, own);
            throw std::runtime_error("native layout fallback GEMM failed");
        }
    }
    if (bias) {
        Bias<<<(size_t(M) * N + 255) / 256, 256, 0, cudaStreamPerThread>>>(out, bias, M, N);
        Check(cudaGetLastError(), "native layout fallback bias");
    }
    FastllmReleaseDequantScratch(tmp, own);
}
} // namespace
bool FastllmCudaTryNativeNvfp4Linear(const Data &input, Data &weight, const Data &bias, Data &output, int M,
                                     int K, int N) {
    if (!weight.cudaNativeNvfp4Layout && !Prepare(weight))
        return false;
    if (!(weight.dims == std::vector<int>({N, K}) && input.dataType == DataType::FLOAT16 &&
          output.dataType == DataType::FLOAT16))
        throw std::runtime_error("Native NVFP4 Linear shape/type mismatch");
    const half *b = bias.dims.empty() ? nullptr : static_cast<const half *>(weight.extraCudaHalfData[0]);
    const half *x = static_cast<const half *>(FastllmCudaPrepareInput(input));
    half *y = static_cast<half *>(FastllmCudaPrepareOutput(output));
    if (!Prefill(x, weight, b, y, M, 0))
        Fallback(x, weight, b, y, M);
    FastllmCudaFinishInput(input, const_cast<half *>(x));
    FastllmCudaFinishOutput(output, y);
    return true;
}
bool FastllmCudaNativeNvfp4LayoutFusedCanRun(const Data &input, const Data &weight, const Data &bias,
                                             const Data &output, bool gate) {
    int device = 0;
    if (!weight.cudaNativeNvfp4Layout || !bias.dims.empty() || weight.dims.size() != 2 ||
        cudaGetDevice(&device) != cudaSuccess || input.dataType != DataType::FLOAT16 ||
        output.dataType != DataType::FLOAT16 || !Dense(input, device) || !Dense(weight, device) ||
        !Dense(output, device))
        return false;
    int N = weight.dims[0], K = weight.dims[1], width = gate ? N / 2 : N;
    if (N <= 0 || K <= 0 || input.dims.back() != K || output.dims.back() != width || input.Count(0) % K)
        return false;
    size_t rows = input.Count(0) / K;
    if (rows < 1 || rows > 4096 || output.Count(0) != rows * width || uintptr_t(input.cudaData) % 4 ||
        uintptr_t(output.cudaData) % 4)
        return false;
    auto overlap = [](const Data &a, const Data &b) {
        uintptr_t x = (uintptr_t)a.cudaData, y = (uintptr_t)b.cudaData;
        return x < y + b.GetBytes() && y < x + a.GetBytes();
    };
    return !overlap(input, output) && !overlap(weight, output) && !overlap(input, weight);
}
bool FastllmCudaNativeNvfp4LayoutFused(Data &input, Data &weight, Data &output, bool gate) {
    int M = input.Count(0) / weight.dims[1], mode = gate ? 1 : 2;
    if (M <= 8) {
        Gemv(static_cast<const half *>(input.cudaData), weight, nullptr, static_cast<half *>(output.cudaData),
             M, mode);
        return true;
    }
    return Prefill(static_cast<const half *>(input.cudaData), weight, nullptr,
                   static_cast<half *>(output.cudaData), M, mode);
}
void FastllmCudaRestoreNativeNvfp4(Data &weight) {
    if (!weight.cudaNativeNvfp4Layout)
        return;
    if (!(weight.cudaData && weight.dataDevice == DataDevice::CUDA && weight.dims.size() == 2))
        throw std::runtime_error("Native NVFP4 restore requires CUDA matrix");
    int device = weight.dataDeviceIds.empty() ? FastllmCudaGetDevice() : weight.dataDeviceIds[0];
    DeviceGuard deviceGuard(device);
    int N = weight.dims[0], K = weight.dims[1];
    size_t bytes = size_t(N) * K * 12 / 16;
    Temporary temporary(bytes);
    uint8_t *tmp = temporary.pointer;
    if (!tmp)
        throw std::runtime_error("native layout restore allocation failed");
    auto *codes = static_cast<const uint8_t *>(weight.cudaData);
    fastllm_native_layout::Restore<<<(size_t(N) * K / 16 + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        codes, codes + size_t(N) * K / 2, Global(weight), tmp, N, K);
    Check(cudaGetLastError(), "native layout restore");
    Check(cudaMemcpyAsync(weight.cudaData, tmp, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread),
          "native layout restore copy");
    Check(cudaStreamSynchronize(cudaStreamPerThread), "native layout restore sync");
    weight.cudaNativeNvfp4Layout = false;
    weight.IsRepacked = false;
}
} // namespace fastllm
#else
namespace fastllm {
bool FastllmCudaTryNativeNvfp4Linear(const Data &, Data &, const Data &, Data &, int, int, int) {
    return false;
}
bool FastllmCudaNativeNvfp4LayoutFusedCanRun(const Data &, const Data &, const Data &, const Data &, bool) {
    return false;
}
bool FastllmCudaNativeNvfp4LayoutFused(Data &, Data &, Data &, bool) { return false; }
void FastllmCudaRestoreNativeNvfp4(Data &) {}
} // namespace fastllm
#endif
