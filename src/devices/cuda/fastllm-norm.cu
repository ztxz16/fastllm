#include "devices/cuda/fastllm-cuda-norm.h"
#include "devices/cuda/fastllm-cuda-norm.cuh"

namespace {
using namespace fastllm::cuda_norm;
using fastllm::cuda_launch::Launch;
cudaStream_t Stream(void *stream) { return stream ? (cudaStream_t)stream : cudaStreamPerThread; }

template<typename T>
bool FastllmCudaRMSNormImpl(const void *input, const float *weight, const void *gate, void *output, int rows,
                           int width, float eps, bool zeroCentered, void *stream) {
    if (!input || !weight || !output || rows < 1 || width < 1 || eps <= 0.f)
        return false;
    if (width == 2048 && rows <= 256 &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output) |
          reinterpret_cast<uintptr_t>(gate)) & 3) == 0) {
        return Launch<HiddenRMSNorm<T, false>>(rows, 128, 0, Stream(stream),
            (const T *)input, (T *)nullptr, weight,
            (const T *)gate, (T *)output, eps, zeroCentered);
    }
    if (rows >= 1024 && width == 128)
        return Launch<WarpRMSNorm<T, 128>>((rows + 3) / 4, 128, 0, Stream(stream),
            (const T *)input, weight, (const T *)gate, (T *)output, rows, eps, zeroCentered);
    if (rows >= 1024 && width == 256)
        return Launch<WarpRMSNorm<T, 256>>((rows + 3) / 4, 128, 0, Stream(stream),
            (const T *)input, weight, (const T *)gate, (T *)output, rows, eps, zeroCentered);
    return Launch<RMSNorm<T, true>>(rows, 256, 0, Stream(stream),
        (const T *)input, weight, (T *)output, width, eps, (const T *)gate, zeroCentered);
}
template<typename T>
bool FastllmCudaAddRMSNormImpl(void *residual, const void *input, const float *weight, void *output, int rows,
                              int width, float eps, bool zeroCentered, void *stream) {
    if (!residual || !input || !weight || !output || output == residual || output == input || rows < 1 ||
        width < 1 || eps <= 0.f)
        return false;
    if (width == 2048 && rows <= 256 &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output) |
          reinterpret_cast<uintptr_t>(residual)) & 3) == 0) {
        return Launch<HiddenRMSNorm<T, true>>(rows, 128, 0, Stream(stream),
            (const T *)input, (T *)residual, weight,
            (const T *)nullptr, (T *)output, eps, zeroCentered);
    }
    if (width == 2048)
        return Launch<AddRMSNorm<T, 2048>>(rows, 256, 0, Stream(stream),
            (T *)residual, (const T *)input, weight, (T *)output, width, eps, zeroCentered);
    return Launch<AddRMSNorm<T, 0>>(rows, 256, 0, Stream(stream),
        (T *)residual, (const T *)input, weight, (T *)output, width, eps, zeroCentered);
}
} // namespace

bool FastllmCudaRMSNorm(const void *input, const float *weight, const void *gate, void *output, int rows,
                           int width, float eps, bool zeroCentered, void *stream, bool fp16) {
    return fp16 ? FastllmCudaRMSNormImpl<__half>(input, weight, gate, output, rows, width, eps, zeroCentered, stream)
                : FastllmCudaRMSNormImpl<__nv_bfloat16>(input, weight, gate, output, rows, width, eps, zeroCentered, stream);
}

bool FastllmCudaAddRMSNorm(void *residual, const void *input, const float *weight, void *output, int rows,
                              int width, float eps, bool zeroCentered, void *stream, bool fp16) {
    return fp16 ? FastllmCudaAddRMSNormImpl<__half>(residual, input, weight, output, rows, width, eps, zeroCentered, stream)
                : FastllmCudaAddRMSNormImpl<__nv_bfloat16>(residual, input, weight, output, rows, width, eps, zeroCentered, stream);
}
