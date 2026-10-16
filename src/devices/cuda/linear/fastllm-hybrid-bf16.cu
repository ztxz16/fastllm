#include "../fastllm-hybrid-prefill.cuh"
#include "devices/cuda/fastllm-cuda-hybrid-prefill.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace {
thread_local bool hybridPrefill = false;

// A conservative forward-error bound admits FP32 only when both BF16
// rounding boundaries are out of reach, including the final FP32 rounding.
// Cancellation, extreme exponents and boundary cases keep the FP64 path.
__global__ void HybridHcPostBounded(const __nv_bfloat16 *x, const __nv_bfloat16 *r, const float *p,
                                    const float *c, __nv_bfloat16 *out, int rows, int dim) {
    size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= size_t(rows) * 4 * dim)
        return;
    int d = at % dim, target = (at / dim) % 4, row = at / (4 * dim);
    float xv = __bfloat162float(x[size_t(row) * dim + d]), pv = p[row * 4 + target];
    float rv[4], cv[4];
    float v = __fmul_rn(pv, xv), m = fabsf(v);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        rv[i] = __bfloat162float(r[(size_t(row) * 4 + i) * dim + d]);
        cv[i] = c[row * 16 + i * 4 + target];
        v = __fmaf_rn(cv[i], rv[i], v);
        m = __fadd_rn(m, fabsf(__fmul_rn(cv[i], rv[i])));
    }
    auto rounded = __float2bfloat16_rn(v);
    unsigned bits = __bfloat16_as_ushort(rounded) & 0x7fff;
    float av = fabsf(v), bound = __fmaf_rn(m, 16.f * FLT_EPSILON, 16.f * FLT_MIN);
    bool safe = isfinite(m) && isfinite(v) && bits > 0 && bits < 0x7f7f;
    if (safe) {
        float center = __uint_as_float(bits << 16), prev = __uint_as_float((bits - 1) << 16),
              next = __uint_as_float((bits + 1) << 16);
        float low = center + (prev - center) * .5f, high = center + (next - center) * .5f;
        safe = av > low + bound && av < high - bound;
    }
    if (!safe) {
        double exact = double(pv) * xv;
#pragma unroll
        for (int i = 0; i < 4; i++)
            exact += double(cv[i]) * rv[i];
        rounded = __float2bfloat16_rn(float(exact));
    }
    out[at] = rounded;
}

union alignas(16) Bf16Vector {
    uint4 raw;
    __nv_bfloat16 values[8];
};
union alignas(16) HalfVector {
    uint4 raw;
    half values[8];
};

// A power-of-two scale preserves BF16 operands exactly when they fit FP16.
// Check every element, including subnormals, instead of assuming a safe range.
__global__ void PackHybridBf16(const __nv_bfloat16 *input, half *output, int *exponents, int *unsafe,
                               int columns) {
    const int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[128];
    __shared__ int exponent;
    float maximum = 0;
    bool valid = true;
    for (int col = tid * 8; col < columns; col += 1024) {
        Bf16Vector value;
        value.raw = *reinterpret_cast<const uint4 *>(input + size_t(row) * columns + col);
        for (int j = 0; j < 8; ++j) {
            const float x = __bfloat162float(value.values[j]);
            maximum = fmaxf(maximum, fabsf(x));
            valid = valid && isfinite(x);
        }
    }
    maxima[tid] = maximum;
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (tid < step)
            maxima[tid] = fmaxf(maxima[tid], maxima[tid + step]);
        __syncthreads();
    }
    if (tid == 0) {
        exponent = max(-32, min(32, 14 - (maxima[0] > 0 ? ilogbf(maxima[0]) : 0)));
        exponents[row] = exponent;
    }
    __syncthreads();
    const float scale = ldexpf(1.f, exponent);
    for (int col = tid * 8; col < columns; col += 1024) {
        Bf16Vector value;
        HalfVector packed;
        value.raw = *reinterpret_cast<const uint4 *>(input + size_t(row) * columns + col);
        for (int j = 0; j < 8; ++j) {
            const float x = __bfloat162float(value.values[j]) * scale;
            packed.values[j] = __float2half_rn(x);
            valid = valid && isfinite(x) && __half2float(packed.values[j]) == x;
        }
        *reinterpret_cast<uint4 *>(output + size_t(row) * columns + col) = packed.raw;
    }
    if (!valid)
        atomicExch(unsafe, 1);
}

__global__ void FinishHybridBf16(const float *input, __nv_bfloat16 *output, const int *inputExponents,
                                 const int *weightExponents, int rows, int width, int stride) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < size_t(rows) * width) {
        output[(i / width) * stride + i % width] =
            __float2bfloat16_rn(ldexpf(input[i], -inputExponents[i / width] - weightExponents[i % width]));
    }
}
} // namespace

bool FastllmCudaSetHybridPrefill(bool enabled) {
    const bool previous = hybridPrefill;
    hybridPrefill = enabled;
    return previous;
}

bool FastllmCudaHybridBf16Gemm(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output,
                               int rows, int columns, int width, int outputStride, void *rawHandle) {
    const auto handle = static_cast<cublasHandle_t>(rawHandle);
    if (!hybridPrefill || rows < 1024 || columns < 128 || width < 128 || columns % 8 || width % 8 ||
        outputStride < width || !input || !weight || !output ||
        (reinterpret_cast<uintptr_t>(input) % alignof(uint4)) ||
        (reinterpret_cast<uintptr_t>(weight) % alignof(uint4)) || FastllmCudaRuntimeArch() != 75)
        return false;
    cublasPointerMode_t pointerMode;
    if (cublasGetPointerMode(handle, &pointerMode) != CUBLAS_STATUS_SUCCESS ||
        pointerMode != CUBLAS_POINTER_MODE_HOST)
        return false;
    cudaStream_t stream;
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS)
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return false;

    constexpr size_t budget = 128ull * 1024 * 1024;
    const size_t inputBytes = size_t(rows) * columns * 2 + size_t(rows) * 4 + 256;
    if (inputBytes >= budget)
        return false;
    int chunk = std::min(width, int((budget - inputBytes) / (size_t(columns) * 2 + size_t(rows) * 4 + 4)));
    chunk = std::min(4096, chunk / 8 * 8);
    if (chunk < 128)
        return false;
    const size_t bytes =
        size_t(rows + chunk) * columns * 2 + size_t(rows) * chunk * 4 + size_t(rows + chunk + 1) * 4;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, bytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
        return false;
    struct Scratch {
        void *ptr;
        cudaStream_t stream;
        ~Scratch() {
            if (!FastllmCudaFreeAfterStream(ptr, stream))
                FastllmCudaFree(ptr);
        }
    } scratch{allocation, stream};
    auto *x = static_cast<half *>(allocation);
    auto *w = x + size_t(rows) * columns;
    auto *y = reinterpret_cast<float *>(w + size_t(chunk) * columns);
    auto *xe = reinterpret_cast<int *>(y + size_t(rows) * chunk);
    auto *we = xe + rows, *unsafe = we + chunk;
    FastllmCudaHybridCheck(cudaMemsetAsync(unsafe, 0, sizeof(int), stream));
    PackHybridBf16<<<rows, 128, 0, stream>>>(input, x, xe, unsafe, columns);
    for (int first = 0; first < width; first += chunk) {
        const int count = std::min(chunk, width - first);
        PackHybridBf16<<<count, 128, 0, stream>>>(weight + size_t(first) * columns, w, we, unsafe, columns);
        int invalid = 0;
        FastllmCudaHybridCheck(
            cudaMemcpyAsync(&invalid, unsafe, sizeof(int), cudaMemcpyDeviceToHost, stream));
        FastllmCudaHybridCheck(cudaStreamSynchronize(stream));
        if (invalid)
            return false;
        const float one = 1, zero = 0;
        // FP32 accumulation/output; restore scale before the existing BF16
        // boundary. Tensor-core reduction order can differ from scalar GEMM.
        const auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, count, rows, columns, &one, w,
                                         CUDA_R_16F, columns, x, CUDA_R_16F, columns, &zero, y, CUDA_R_32F,
                                         count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS)
            return false;
        FinishHybridBf16<<<(size_t(rows) * count + 255) / 256, 256, 0, stream>>>(y, output + first, xe, we,
                                                                                 rows, count, outputStride);
        FastllmCudaHybridCheck(cudaGetLastError());
    }
    return true;
}

bool FastllmCudaHybridHcPost(const fastllm::Data &input, const fastllm::Data &residual, const float *post,
                             const float *comb, fastllm::Data &output, int tokens, int hcMult, int dim) {
    if (!hybridPrefill || tokens < 1024 || hcMult != 4 || dim <= 0 || input.dataType != fastllm::BFLOAT16 ||
        residual.dataType != fastllm::BFLOAT16 || output.dataType != fastllm::BFLOAT16 ||
        FastllmCudaRuntimeArch() != 75)
        return false;
    HybridHcPostBounded<<<(size_t(tokens) * 4 * dim + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData),
        static_cast<const __nv_bfloat16 *>(residual.cudaData), post, comb,
        static_cast<__nv_bfloat16 *>(output.cudaData), tokens, dim);
    FastllmCudaHybridCheck(cudaGetLastError());
    return true;
}
