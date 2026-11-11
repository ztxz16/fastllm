#include "models/minimax_h3_image/ops.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace fastllm::minimax_h3_image {
namespace {
template <class T, int N> struct alignas(sizeof(T) * N) Pack { T values[N]; };
template <class T, int N>
__global__ void gateKernel(const T *input, T *output, size_t count, int width) {
    auto x = reinterpret_cast<const Pack<T, N> *>(input);
    auto y = reinterpret_cast<Pack<T, N> *>(output);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        size_t p = (i / width) * (size_t(2) * width) + i % width;
        auto gates = x[p], values = x[p + width];
        Pack<T, N> result;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float g = float(gates.values[j]);
            result.values[j] = T(float(T(g / (1.f + expf(-g)))) * float(values.values[j]));
        }
        y[i] = result;
    }
}
} // namespace
nomas::Tensor gatedSilu(const nomas::Tensor &input) {
    if (!input.data || input.rows < 1 || input.cols < 2 || input.cols % 2)
        throw std::invalid_argument("invalid image gated SiLU input");
    nomas::Tensor output(input.rows, input.cols / 2, input.dtype);
    auto launch = [&](auto scalar) {
        using T = decltype(scalar);
        const auto *x = reinterpret_cast<const T *>(input.data);
        auto *y = reinterpret_cast<T *>(output.data);
        if (sizeof(T) == 2 && output.cols % 8 == 0 && uintptr_t(x) % 16 == 0) {
            auto count = output.size() / 8;
            gateKernel<T, 8><<<std::min<size_t>((count + 255) / 256, 4096), 256, 0,
                               cudaStreamPerThread>>>(x, y, count, output.cols / 8);
        } else {
            gateKernel<T, 1><<<std::min<size_t>((output.size() + 255) / 256, 4096), 256, 0,
                               cudaStreamPerThread>>>(x, y, output.size(), output.cols);
        }
    };
    if (input.dtype == nomas::DType::BF16)
        launch(__nv_bfloat16{});
    else if (input.dtype == nomas::DType::FP16)
        launch(__half{});
    else
        launch(float{});
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return output;
}
} // namespace fastllm::minimax_h3_image
