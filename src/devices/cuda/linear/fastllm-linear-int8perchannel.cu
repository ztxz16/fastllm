//
// Signed symmetric INT8 per-channel weights (compressed-tensors W8A8 / W8A16).
//
// Storage (DataType::INT8_PERCHANNEL_S8 and INT8_PERCHANNEL_S8_W8A16):
//   * payload: plain [out, in] row-major int8, two's complement.
//   * Data::scales: one FP32 scale per output channel, value = scale * q.
//
// Decode and small batches stay on FP16/BF16 activations and use a fused
// signed int8 GEMV.  Prefill either quantizes activations per token and runs
// a cuBLASLt/cuBLAS integer tensor-core GEMM (W8A8) or dequantizes weight
// chunks before a floating-point GEMM (W8A16), mirroring the FP8 per-channel
// strategy used elsewhere in FastLLM.
//
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils.h"

#include <algorithm>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <map>
#include <memory>
#include <string>

namespace {

constexpr int INT8S8_GEMV_WARPS = 8;
// Maximum number of tokens a single GEMV launch accumulates in registers.
constexpr int INT8S8_GEMV_MAX_ROWS = 8;
// cuBLAS integer GEMM kernels are inefficient for token counts between 9 and
// 31; pad those batches so the efficient N>=32 tiling is selected.
constexpr int INT8S8_GEMM_MIN_TOKENS = 32;

template <class T> struct Int8S8Traits;

template <> struct Int8S8Traits<half> {
    static __device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
    static __device__ __forceinline__ half FromFloat(float v) { return __float2half_rn(v); }
    static constexpr cudaDataType_t CudaType = CUDA_R_16F;
};

template <> struct Int8S8Traits<__nv_bfloat16> {
    static __device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 FromFloat(float v) {
        return __float2bfloat16_rn(v);
    }
    static constexpr cudaDataType_t CudaType = CUDA_R_16BF;
};

inline size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// One warp per output row.  ``rows`` (<= INT8S8_GEMV_MAX_ROWS) activation rows
// share the single weight row read so decode and MTP verify do not multiply the
// weight traffic by the batch size.
template <class T, int WARPS>
__global__ void Int8S8PerChannelGemvKernel(
        const T *__restrict__ input, const int8_t *__restrict__ weight,
        T *__restrict__ output, const float *__restrict__ bias,
        const float *__restrict__ scales, int rows, int m, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS + warp;
    if (row >= k) {
        return;
    }
    const int8_t *weightRow = weight + (size_t)row * (size_t)m;
    float acc[INT8S8_GEMV_MAX_ROWS];
#pragma unroll
    for (int i = 0; i < INT8S8_GEMV_MAX_ROWS; i++) {
        acc[i] = 0.0f;
    }

    const int vectorEnd = m & ~15;
    for (int base = lane * 16; base < vectorEnd; base += 32 * 16) {
        const uint4 packedWeight = *reinterpret_cast<const uint4 *>(weightRow + base);
        const int8_t *weightValues = reinterpret_cast<const int8_t *>(&packedWeight);
#pragma unroll
        for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
            if (r >= rows) {
                break;
            }
            const T *inputRow = input + (size_t)r * (size_t)m;
            const uint4 lo = *reinterpret_cast<const uint4 *>(inputRow + base);
            const uint4 hi = *reinterpret_cast<const uint4 *>(inputRow + base + 8);
            const T *loValues = reinterpret_cast<const T *>(&lo);
            const T *hiValues = reinterpret_cast<const T *>(&hi);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[r] = fmaf((float)weightValues[j],
                              Int8S8Traits<T>::ToFloat(loValues[j]), acc[r]);
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[r] = fmaf((float)weightValues[8 + j],
                              Int8S8Traits<T>::ToFloat(hiValues[j]), acc[r]);
            }
        }
    }
    for (int j = vectorEnd + lane; j < m; j += 32) {
        const float weightValue = (float)weightRow[j];
#pragma unroll
        for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
            if (r >= rows) {
                break;
            }
            acc[r] = fmaf(weightValue,
                          Int8S8Traits<T>::ToFloat(input[(size_t)r * m + j]),
                          acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
        if (r >= rows) {
            break;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
        const float scale = __ldg(scales + row);
        const float biasValue = bias == nullptr ? 0.0f : __ldg(bias + row);
#pragma unroll
        for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
            if (r >= rows) {
                break;
            }
            output[(size_t)r * k + row] =
                Int8S8Traits<T>::FromFloat(acc[r] * scale + biasValue);
        }
    }
}

// Scalar fallback for rows whose input length is not a multiple of sixteen.
template <class T, int WARPS>
__global__ void Int8S8PerChannelGemvScalarKernel(
        const T *__restrict__ input, const int8_t *__restrict__ weight,
        T *__restrict__ output, const float *__restrict__ bias,
        const float *__restrict__ scales, int rows, int m, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS + warp;
    if (row >= k) {
        return;
    }
    const int8_t *weightRow = weight + (size_t)row * (size_t)m;
    float acc[INT8S8_GEMV_MAX_ROWS];
#pragma unroll
    for (int i = 0; i < INT8S8_GEMV_MAX_ROWS; i++) {
        acc[i] = 0.0f;
    }
    for (int j = lane; j < m; j += 32) {
        const float weightValue = (float)weightRow[j];
#pragma unroll
        for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
            if (r >= rows) {
                break;
            }
            acc[r] = fmaf(weightValue,
                          Int8S8Traits<T>::ToFloat(input[(size_t)r * m + j]),
                          acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
        if (r >= rows) {
            break;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
        const float scale = __ldg(scales + row);
        const float biasValue = bias == nullptr ? 0.0f : __ldg(bias + row);
#pragma unroll
        for (int r = 0; r < INT8S8_GEMV_MAX_ROWS; r++) {
            if (r >= rows) {
                break;
            }
            output[(size_t)r * k + row] =
                Int8S8Traits<T>::FromFloat(acc[r] * scale + biasValue);
        }
    }
}

// Per-token symmetric int8 quantization: scale = max|x| / 127, q = round(x /
// scale) clamped to [-127, 127].  Matches the dynamic activation quantization
// that compressed-tensors W8A8 checkpoints are calibrated with.
//
// One warp per token: the per-token maximum is reduced with shuffles instead of
// a block-wide shared-memory tree, and a whole matrix is quantized by a single
// launch.  Launching one block per token was ~70% of prefill wall time because
// each tiny launch costs more than the row itself.
template <class T, int WARPS_PER_BLOCK>
__global__ void Int8S8PerChannelQuantizeKernel(
        const T *__restrict__ input, int8_t *__restrict__ quantized,
        float *__restrict__ scales, int rows, int m) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) {
        return;
    }
    const T *source = input + (size_t)row * (size_t)m;
    int8_t *destination = quantized + (size_t)row * (size_t)m;
    const bool vectorized = (m % 8) == 0 &&
                            ((reinterpret_cast<uintptr_t>(source) & 15) == 0);
    float localMax = 0.0f;
    int i = lane * 8;
    if (vectorized) {
        const int vectorEnd = m & ~7;
        for (; i < vectorEnd; i += 32 * 8) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(source + i);
            const T *values = reinterpret_cast<const T *>(&packed);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                localMax = fmaxf(localMax,
                                 fabsf(Int8S8Traits<T>::ToFloat(values[j])));
            }
        }
        if (i >= m) {
            i = m;
        }
    }
    for (i = vectorized ? (m & ~7) + lane : lane; i < m; i += 32) {
        localMax = fmaxf(localMax, fabsf(Int8S8Traits<T>::ToFloat(source[i])));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffffu, localMax, offset));
    }
    localMax = __shfl_sync(0xffffffffu, localMax, 0);
    const float maxAbs = localMax;
    const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
    const float invScale = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
    if (vectorized) {
        const int vectorEnd = m & ~7;
        for (int j = lane * 8; j < vectorEnd; j += 32 * 8) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(source + j);
            const T *values = reinterpret_cast<const T *>(&packed);
            int8_t quantizedValues[8];
#pragma unroll
            for (int t = 0; t < 8; t++) {
                float value =
                    rintf(Int8S8Traits<T>::ToFloat(values[t]) * invScale);
                value = fminf(127.0f, fmaxf(-127.0f, value));
                quantizedValues[t] = (int8_t)value;
            }
            memcpy(destination + j, quantizedValues, sizeof(quantizedValues));
        }
        for (int j = vectorEnd + lane; j < m; j += 32) {
            float value = rintf(Int8S8Traits<T>::ToFloat(source[j]) * invScale);
            value = fminf(127.0f, fmaxf(-127.0f, value));
            destination[j] = (int8_t)value;
        }
    } else {
        for (int j = lane; j < m; j += 32) {
            float value = rintf(Int8S8Traits<T>::ToFloat(source[j]) * invScale);
            value = fminf(127.0f, fmaxf(-127.0f, value));
            destination[j] = (int8_t)value;
        }
    }
    if (lane == 0) {
        scales[row] = scale;
    }
}

// Epilogue for the integer GEMM: out = acc * rowScale * channelScale + bias.
// ``acc`` is row-major [rows, k] because the GEMM is issued transposed.
template <class T, int THREADS>
__global__ void Int8S8PerChannelEpilogueKernel(
        const int32_t *__restrict__ accum, T *__restrict__ output,
        const float *__restrict__ rowScales,
        const float *__restrict__ channelScales,
        const float *__restrict__ bias, int rows, int k) {
    const int column = blockIdx.x * THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= k || row >= rows) {
        return;
    }
    const size_t index = (size_t)row * (size_t)k + column;
    float value = (float)accum[index] * rowScales[row] *
                  __ldg(channelScales + column);
    if (bias != nullptr) {
        value += __ldg(bias + column);
    }
    output[index] = Int8S8Traits<T>::FromFloat(value);
}

template <class T>
__global__ void Int8S8PerChannelAddBiasKernel(
        T *__restrict__ output, const float *__restrict__ bias,
        int rows, int k) {
    const int64_t total = (int64_t)rows * (int64_t)k;
    for (int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (int64_t)gridDim.x * blockDim.x) {
        const int column = (int)(index % k);
        output[index] = Int8S8Traits<T>::FromFloat(
            Int8S8Traits<T>::ToFloat(output[index]) + __ldg(bias + column));
    }
}

// int8 -> FP16/BF16 dequantization used by the W8A16 prefill path.
template <class T>
__global__ void Int8S8PerChannelDequantKernel(
        const int8_t *__restrict__ weight, const float *__restrict__ scales,
        T *__restrict__ output, int columns) {
    const int row = blockIdx.x;
    const int8_t *source = weight + (size_t)row * (size_t)columns;
    T *destination = output + (size_t)row * (size_t)columns;
    const float scale = __ldg(scales + row);
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        destination[i] = Int8S8Traits<T>::FromFloat((float)source[i] * scale);
    }
}

float *EnsureInt8S8ScalesOnDevice(fastllm::Data &weight, int rows) {
    if (weight.extraCudaData.size() < 2) {
        weight.extraCudaData.resize(2, nullptr);
    }
    if (weight.extraCudaData[0] == nullptr) {
        // Guard the upload itself: a short or empty scale vector used to reach
        // cudaMemcpy with a null source and surface as an unrelated
        // cudaErrorInvalidValue deep inside the forward.
        fastllm::AssertInFastLLM(
            weight.scales.size() >= (size_t)rows,
            "INT8 per-channel weight \"" + weight.name +
                "\" has fewer scales than output channels (" +
                std::to_string(weight.scales.size()) + " < " +
                std::to_string(rows) + ").");
        float *deviceScales = (float *)FastllmCudaMalloc(rows * sizeof(float));
        if (deviceScales == nullptr) {
            return nullptr;
        }
        FastllmCudaCopyFromHostToDevice(deviceScales, weight.scales.data(),
                                        rows * sizeof(float));
        weight.extraCudaData[0] = deviceScales;
    }
    return (float *)weight.extraCudaData[0];
}

const float *GetInt8S8BiasOnDevice(fastllm::Data &weight,
                                   const fastllm::Data &bias, int rows) {
    if (bias.dims.size() == 0) {
        return nullptr;
    }
    if (bias.dataDevice == fastllm::DataDevice::CUDA && bias.cudaData != nullptr) {
        return (const float *)bias.cudaData;
    }
    if (weight.extraCudaData.size() < 2) {
        weight.extraCudaData.resize(2, nullptr);
    }
    if (weight.extraCudaData[1] == nullptr) {
        float *deviceBias = (float *)FastllmCudaMalloc(rows * sizeof(float));
        if (deviceBias == nullptr) {
            return nullptr;
        }
        FastllmCudaCopyFromHostToDevice(deviceBias, bias.cpuData,
                                        rows * sizeof(float));
        weight.extraCudaData[1] = deviceBias;
    }
    return (const float *)weight.extraCudaData[1];
}

// W8A16 prefill: dequantize weight chunks into the shared scratch and run a
// floating-point GEMM, mirroring the FP8 per-channel implementation.
template <class T>
bool LaunchInt8S8DequantGemm(const T *cudaInput, const int8_t *deviceWeight,
                             T *cudaOutput, const float *deviceBias,
                             const float *deviceScales, int n, int m, int k) {
#ifdef CUDA_NO_TENSOR_CORE
    // The floating-point fallback for pre-Turing parts writes FP32 output and
    // would need an extra conversion pass; keep those parts unsupported so the
    // caller reports a precise error instead of computing with a mismatch.
    return false;
#else
    size_t availableBytes = 0;
    bool own = false;
    T *dequantScratch = (T *)FastllmBorrowDequantScratch(
        (size_t)k * m * sizeof(T), &availableBytes, &own);
    if (dequantScratch == nullptr) {
        return false;
    }
    const size_t bytesPerRow = (size_t)m * sizeof(T);
    const int maxRows = (int)std::min<size_t>(
        (size_t)k, std::max<size_t>(1, availableBytes / bytesPerRow));
    if (maxRows <= 0) {
        FastllmReleaseDequantScratch(dequantScratch, own);
        return false;
    }
    auto handle = getFastllmCublasHandle();
    const cudaDataType_t inputType = Int8S8Traits<T>::CudaType;
    const cudaDataType_t outputType = Int8S8Traits<T>::CudaType;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    bool ok = true;
    // FP32 accumulation with a 16-bit output keeps the dequantized W8A16
    // projections accurate over long reduction dimensions; alpha/beta must be
    // FP32 host scalars for this compute type.
    float alpha = 1.0f, beta = 0.0f;
    for (int offset = 0; offset < k && ok; offset += maxRows) {
        const int chunk = std::min(maxRows, k - offset);
        Int8S8PerChannelDequantKernel<T><<<chunk, 256>>>(
            deviceWeight + (size_t)offset * m, deviceScales + offset,
            dequantScratch, m);
        status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, chunk, n, m, &alpha,
            dequantScratch, inputType, m, cudaInput, inputType, m, &beta,
            cudaOutput + (size_t)offset, outputType, k, CUDA_R_32F,
            static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        ok = status == CUBLAS_STATUS_SUCCESS;
    }
    FastllmReleaseDequantScratch(dequantScratch, own);
    if (!ok) {
        return false;
    }
    if (deviceBias != nullptr) {
        const int64_t total = (int64_t)n * k;
        const int threads = 256;
        const int blocks = (int)std::min<int64_t>(
            4096, std::max<int64_t>(1, (total + threads - 1) / threads));
        Int8S8PerChannelAddBiasKernel<T><<<blocks, threads>>>(
            cudaOutput, deviceBias, n, k);
    }
    return true;
#endif
}

// W8A8 prefill: per-token int8 activation quantization plus an integer
// tensor-core GEMM.  The GEMM is issued as C[k, n] = W[k, m] * A[n, m]^T so the
// int32 result is already row-major [n, k] and the epilogue is elementwise.
// cuBLASLt state for the integer prefill GEMM.  The transposed formulation
// C[k, n] = W[k, m] * A[n, m]^T keeps the int32 result row-major [n, k] so the
// epilogue stays elementwise.  Algorithms are selected once per shape and
// cached; every device gets its own handle and workspace.
struct Int8S8LtState {
    std::mutex mutex;
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    size_t workspaceBytes = 0;
    std::map<std::string, std::pair<cublasLtMatmulAlgo_t, size_t>> algorithms;
};

Int8S8LtState *GetInt8S8LtState() {
    static std::mutex mapMutex;
    static std::map<int, std::unique_ptr<Int8S8LtState>> states;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(mapMutex);
    auto &entry = states[device];
    if (entry == nullptr) {
        entry = std::make_unique<Int8S8LtState>();
        constexpr size_t workspaceBytes = 32ull << 20;
        if (cublasLtCreate(&entry->handle) != CUBLAS_STATUS_SUCCESS) {
            entry->handle = nullptr;
            return nullptr;
        }
        entry->workspace = FastllmCudaMalloc(workspaceBytes);
        entry->workspaceBytes = entry->workspace != nullptr ? workspaceBytes : 0;
        cublasLtMatmulPreference_t preference = nullptr;
        if (cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
    }
    return entry->handle != nullptr ? entry.get() : nullptr;
}

// Returns true when the integer GEMM was issued through cuBLASLt.
bool Int8S8MatmulInt32Lt(const int8_t *weight, const int8_t *activation,
                         int32_t *result, int rows, int m, int k) {
    Int8S8LtState *state = GetInt8S8LtState();
    if (state == nullptr) {
        return false;
    }
    const std::string key = std::to_string(rows) + "x" + std::to_string(m) +
                            "x" + std::to_string(k);
    std::lock_guard<std::mutex> guard(state->mutex);
    auto cached = state->algorithms.find(key);
    if (cached == state->algorithms.end()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            // Never pay for algorithm discovery inside a capture.
            return false;
        }
        cublasLtMatmulDesc_t operation = nullptr;
        cublasLtMatrixLayout_t weightLayout = nullptr;
        cublasLtMatrixLayout_t activationLayout = nullptr;
        cublasLtMatrixLayout_t resultLayout = nullptr;
        cublasLtMatmulPreference_t preference = nullptr;
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasOperation_t noTranspose = CUBLAS_OP_N;
        cublasLtMatmulHeuristicResult_t heuristic[8];
        int found = 0;
        bool discovered = false;
        if (cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32I,
                                     CUDA_R_32I) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_8I, m, k, m) ==
                CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&activationLayout, CUDA_R_8I, m, rows,
                                       m) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&resultLayout, CUDA_R_32I, k, rows, k) ==
                CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulDescSetAttribute(operation,
                                           CUBLASLT_MATMUL_DESC_TRANSA,
                                           &transpose, sizeof(transpose));
            cublasLtMatmulDescSetAttribute(operation,
                                           CUBLASLT_MATMUL_DESC_TRANSB,
                                           &noTranspose, sizeof(noTranspose));
            cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &state->workspaceBytes, sizeof(state->workspaceBytes));
            if (cublasLtMatmulAlgoGetHeuristic(
                    state->handle, operation, weightLayout, activationLayout,
                    resultLayout, resultLayout, preference, 8, heuristic,
                    &found) == CUBLAS_STATUS_SUCCESS &&
                found > 0 &&
                heuristic[0].workspaceSize <= state->workspaceBytes) {
                state->algorithms[key] = {heuristic[0].algo,
                                          heuristic[0].workspaceSize};
                discovered = true;
            }
        }
        if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
        if (weightLayout != nullptr) cublasLtMatrixLayoutDestroy(weightLayout);
        if (activationLayout != nullptr) cublasLtMatrixLayoutDestroy(activationLayout);
        if (resultLayout != nullptr) cublasLtMatrixLayoutDestroy(resultLayout);
        if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
        if (!discovered) {
            state->algorithms[key] = {cublasLtMatmulAlgo_t(), 0};
        }
        cached = state->algorithms.find(key);
    }
    if (cached->second.second > state->workspaceBytes) {
        return false;
    }
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weightLayout = nullptr;
    cublasLtMatrixLayout_t activationLayout = nullptr;
    cublasLtMatrixLayout_t resultLayout = nullptr;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasOperation_t noTranspose = CUBLAS_OP_N;
    bool ok = false;
    if (cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32I,
                                 CUDA_R_32I) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_8I, m, k, m) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&activationLayout, CUDA_R_8I, m, rows, m) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&resultLayout, CUDA_R_32I, k, rows, k) ==
            CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose, sizeof(transpose));
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &noTranspose, sizeof(noTranspose));
        int32_t alpha = 1, beta = 0;
        cublasLtMatmulAlgo_t algorithm = cached->second.first;
        ok = cublasLtMatmul(state->handle, operation, &alpha, weight,
                            weightLayout, activation, activationLayout, &beta,
                            result, resultLayout, result, resultLayout,
                            &algorithm, state->workspace,
                            state->workspaceBytes, cudaStreamPerThread) ==
             CUBLAS_STATUS_SUCCESS;
    }
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (weightLayout != nullptr) cublasLtMatrixLayoutDestroy(weightLayout);
    if (activationLayout != nullptr) cublasLtMatrixLayoutDestroy(activationLayout);
    if (resultLayout != nullptr) cublasLtMatrixLayoutDestroy(resultLayout);
    return ok;
}

bool Int8S8MatmulInt32(const int8_t *weight, const int8_t *activation,
                       int32_t *result, int rows, int m, int k) {
    if (Int8S8MatmulInt32Lt(weight, activation, result, rows, m, k)) {
        return true;
    }
    auto handle = getFastllmCublasHandle();
    int32_t alpha = 1, beta = 0;
    return cublasGemmEx(
               handle, CUBLAS_OP_T, CUBLAS_OP_N, k, rows, m, &alpha, weight,
               CUDA_R_8I, m, activation, CUDA_R_8I, m, &beta, result,
               CUDA_R_32I, k, CUBLAS_COMPUTE_32I,
               static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT)) ==
           CUBLAS_STATUS_SUCCESS;
}

template <class T>
bool LaunchInt8S8ActivationGemm(const T *cudaInput, const int8_t *deviceWeight,
                                T *cudaOutput, const float *deviceBias,
                                const float *deviceScales, int n, int m, int k) {
    if ((m % 16) != 0) {
        return false;
    }
    const int paddedTokens = (n > INT8S8_GEMV_MAX_ROWS && n < INT8S8_GEMM_MIN_TOKENS)
                                 ? INT8S8_GEMM_MIN_TOKENS
                                 : n;
    const size_t quantBytes = AlignUp((size_t)paddedTokens * m, 256);
    const size_t scaleBytes = AlignUp((size_t)paddedTokens * sizeof(float), 256);
    // Ask for a useful accumulator window instead of a single row: a tiny
    // borrow forces one cuBLAS call plus one epilogue launch per token, which
    // dominated prefill wall time.  512 rows keeps the scratch bounded while
    // leaving a handful of chunks for the largest projections.
    const size_t targetAccumRows =
        std::min<size_t>((size_t)paddedTokens, 512);
    const size_t targetAccumBytes = targetAccumRows * (size_t)k * sizeof(int32_t);
    size_t availableBytes = 0;
    bool own = false;
    uint8_t *scratch = (uint8_t *)FastllmBorrowDequantScratch(
        quantBytes + scaleBytes + targetAccumBytes, &availableBytes, &own);
    if (scratch == nullptr ||
        availableBytes < quantBytes + scaleBytes + (size_t)k * sizeof(int32_t)) {
        FastllmReleaseDequantScratch(scratch, own);
        return false;
    }
    int8_t *quantized = (int8_t *)scratch;
    float *activationScales = (float *)(scratch + quantBytes);
    int32_t *accumulator = (int32_t *)(scratch + quantBytes + scaleBytes);
    const size_t accumulatorCapacity = availableBytes - quantBytes - scaleBytes;
    const int chunkRows = (int)std::min<size_t>(
        (size_t)paddedTokens,
        accumulatorCapacity / ((size_t)k * sizeof(int32_t)));
    if (chunkRows <= 0) {
        FastllmReleaseDequantScratch(scratch, own);
        return false;
    }

    const int quantizeWarps = 8;
    const int quantizeRows = quantizeWarps;
    Int8S8PerChannelQuantizeKernel<T, quantizeWarps>
        <<<(n + quantizeRows - 1) / quantizeRows, quantizeWarps * 32>>>(
            cudaInput, quantized, activationScales, n, m);
    if (paddedTokens > n) {
        FastllmCudaMemset0(quantized + (size_t)n * m,
                           (size_t)(paddedTokens - n) * m);
    }

    bool ok = true;
    for (int start = 0; start < paddedTokens && ok; start += chunkRows) {
        const int rows = std::min(chunkRows, paddedTokens - start);
        ok = Int8S8MatmulInt32(deviceWeight, quantized + (size_t)start * m,
                               accumulator, rows, m, k);
        if (!ok) {
            break;
        }
        const int realRows = std::max(0, std::min(rows, n - start));
        if (realRows > 0) {
            constexpr int epilogueThreads = 256;
            const dim3 epilogueGrid((k + epilogueThreads - 1) / epilogueThreads,
                                    realRows);
            Int8S8PerChannelEpilogueKernel<T, epilogueThreads>
                <<<epilogueGrid, epilogueThreads>>>(
                    accumulator, cudaOutput + (size_t)start * k,
                    activationScales + start, deviceScales, deviceBias,
                    realRows, k);
        }
    }
    FastllmReleaseDequantScratch(scratch, own);
    return ok;
}

// W8A16 checkpoints keep FP16/BF16 activations for the projections the
// quantizer excluded from activation quantization (for example down_proj and
// out_proj).  Dequantizing those weights for an FP16 GEMM costs roughly twice
// the INT8 tensor-core time, so prefill may opt in to the INT8 activation path
// with FASTLLM_INT8S8_W8A16_PREFILL_ACTIVATION=1.  Decode always keeps the
// original FP16/BF16 activations.
inline bool Int8S8W8A16PrefillActivationEnabled() {
    static const bool enabled = []() {
        const char *env =
            std::getenv("FASTLLM_INT8S8_W8A16_PREFILL_ACTIVATION");
        return env != nullptr && env[0] != '0' && env[0] != 'f' &&
               env[0] != 'F';
    }();
    return enabled;
}

// Fused column-parallel linear + TP=2 allreduce(+residual) with a chunked
// pipeline: the direct P2P reduction of chunk i runs on a side stream while
// chunk i+1's GEMM/epilogue executes on the main stream.  Numerically identical
// to computing the whole linear and reducing once.
//
// ``partial`` receives the rank-local partial sums (also the reduce source),
template <class T, bool ACTIVATION_INT8>
bool Int8S8PerChannelLinear(const fastllm::Data &input, fastllm::Data &weight,
                            const fastllm::Data &bias, fastllm::Data &output,
                            int n, int m, int k) {
    if (n <= 0 || m <= 0 || k <= 0 || weight.cudaData == nullptr) {
        return false;
    }
    T *cudaInput = (T *)FastllmCudaPrepareInput(input);
    T *cudaOutput = (T *)FastllmCudaPrepareOutput(output);
    if (cudaInput == nullptr || cudaOutput == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }
    float *deviceScales = EnsureInt8S8ScalesOnDevice(weight, k);
    const float *deviceBias = GetInt8S8BiasOnDevice(weight, bias, k);
    const int8_t *deviceWeight = (const int8_t *)weight.cudaData;
    bool ok = false;
    if (deviceScales != nullptr && n <= INT8S8_GEMV_MAX_ROWS) {
        // Decode and MTP verify: fused signed GEMV on FP16/BF16 activations.
        if ((m % 16) == 0) {
            const int blocks = (k + INT8S8_GEMV_WARPS - 1) / INT8S8_GEMV_WARPS;
            Int8S8PerChannelGemvKernel<T, INT8S8_GEMV_WARPS>
                <<<blocks, INT8S8_GEMV_WARPS * 32>>>(
                    cudaInput, deviceWeight, cudaOutput, deviceBias,
                    deviceScales, n, m, k);
        } else {
            const int blocks = (k + INT8S8_GEMV_WARPS - 1) / INT8S8_GEMV_WARPS;
            Int8S8PerChannelGemvScalarKernel<T, INT8S8_GEMV_WARPS>
                <<<blocks, INT8S8_GEMV_WARPS * 32>>>(
                    cudaInput, deviceWeight, cudaOutput, deviceBias,
                    deviceScales, n, m, k);
        }
        ok = true;
    } else if (deviceScales != nullptr &&
               (ACTIVATION_INT8 ||
                (n >= INT8S8_GEMM_MIN_TOKENS &&
                 Int8S8W8A16PrefillActivationEnabled()))) {
        ok = LaunchInt8S8ActivationGemm<T>(
            cudaInput, deviceWeight, cudaOutput, deviceBias, deviceScales, n,
            m, k);
    }
    if (!ok && deviceScales != nullptr && n > INT8S8_GEMV_MAX_ROWS) {
        ok = LaunchInt8S8DequantGemm<T>(
            cudaInput, deviceWeight, cudaOutput, deviceBias, deviceScales, n,
            m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return ok;
}

} // namespace

bool FastllmCudaHalfMatMulFloatInt8PerChannelS8(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return Int8S8PerChannelLinear<half, true>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaBFloat16MatMulInt8PerChannelS8(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return Int8S8PerChannelLinear<__nv_bfloat16, true>(
        input, weight, bias, output, n, m, k);
}

bool FastllmCudaHalfMatMulFloatInt8PerChannelS8W8A16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return Int8S8PerChannelLinear<half, false>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaBFloat16MatMulInt8PerChannelS8W8A16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return Int8S8PerChannelLinear<__nv_bfloat16, false>(
        input, weight, bias, output, n, m, k);
}
