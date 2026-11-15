#include "models/qwen_image/runtime.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace fastllm {
namespace qwen_image {
namespace {
using B = __nv_bfloat16;
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
int grid(size_t n) { return std::min<size_t>((n + 255) / 256, 4096); }
template<class T> T *ptr(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
__global__ void packWeight(const uint16_t *x, uint16_t *out, size_t n, int channels, int spatial) {
    int inner = channels * spatial;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int k = i % inner;
        out[i] = x[(i / inner) * inner + (k % channels) * spatial + k / channels];
    }
}
__global__ void im2col(const B *x, B *out, int h, int w, int c, int kernel, int stride, int pad,
                       int ow, int start, size_t n) {
    int inner = c * kernel * kernel;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int pixel = start + i / inner, k = i % inner, channel = k % c, spatial = k / c;
        int y = pixel / ow * stride - pad + spatial / kernel, z = pixel % ow * stride - pad + spatial % kernel;
        out[i] = y >= 0 && y < h && z >= 0 && z < w ? x[(size_t(y) * w + z) * c + channel] : B(0.f);
    }
}
// Broadcast bias as the epilogue source (zero row stride). Accumulation and
// bias addition stay FP32, with a single BF16 rounding directly into output.
bool implicitConv(const Tensor &x, const Tensor &w, const Tensor *bias, Tensor &out, int h, int width,
                  int kernel, int stride, int pad, int oh, int ow) {
    using E = cutlass::bfloat16_t;
    using L = cutlass::layout::TensorNHWC;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        E, L, E, L, E, L, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<E, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided>::Kernel;
    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
    cutlass::Tensor4DCoord input(1, h, width, x.cols), filter(w.rows, kernel, kernel, x.cols),
                          output(1, oh, ow, w.rows);
    cutlass::conv::Conv2dProblemSize problem(input, filter, {pad, pad, pad, pad},
                                            {stride, stride}, {1, 1}, output);
    typename Conv::Arguments args(problem,
        {ptr<E>(x), L::packed(input)}, {ptr<E>(w), L::packed(filter)},
        {bias ? ptr<E>(*bias) : ptr<E>(out), bias ? L(0, 0, 0) : L::packed(output)},
        {ptr<E>(out), L::packed(output)}, {1.f, bias ? 1.f : 0.f});
    if (Conv::can_implement(args) != cutlass::Status::kSuccess)
        return false;
    Conv conv;
    auto status = conv.initialize(args, nullptr, cudaStreamPerThread);
    if (status == cutlass::Status::kSuccess)
        status = conv(cudaStreamPerThread);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("Qwen VAE implicit convolution failed");
    return true;
}
} // namespace
Tensor Runtime::packConvWeight(const Tensor &w, int channels) {
    if (w.dtype != nomas::DType::BF16 || channels < 1 || w.cols % channels)
        throw std::invalid_argument("convolution weight packing shape");
    Tensor packed(w.rows, w.cols, w.dtype);
    packWeight<<<grid(w.size()), 256, 0, cudaStreamPerThread>>>(w.data, packed.data, w.size(), channels, w.cols / channels);
    check(cudaGetLastError());
    return packed;
}
Tensor Runtime::packedConv2d(const Tensor &x, const Tensor &w, const Tensor *bias, int h, int width,
                             int kernel, int stride, int pad, int padEnd) {
    if (x.dtype != nomas::DType::BF16 || w.dtype != x.dtype || x.rows != h * width ||
        kernel < 1 || stride < 1 || w.cols != x.cols * kernel * kernel ||
        (bias && (bias->dtype != x.dtype || bias->size() != size_t(w.rows))))
        throw std::invalid_argument("packed convolution shape");
    int oh = (h + 2 * pad + padEnd - kernel) / stride + 1,
        ow = (width + 2 * pad + padEnd - kernel) / stride + 1;
    if (oh < 1 || ow < 1)
        throw std::invalid_argument("empty packed convolution output");
    Tensor out(oh * ow, w.rows, x.dtype);
    if (x.cols % 8 == 0 && w.rows % 8 == 0 &&
        implicitConv(x, w, bias, out, h, width, kernel, stride, pad, oh, ow))
        return out;
    int tileRows = std::min<size_t>(32768, (size_t(128) << 20) / (w.cols * 2));
    tileRows = std::min(out.rows, std::max(256, tileRows / 256 * 256));
    Tensor workspace(tileRows, w.cols, x.dtype);
    for (int start = 0; start < out.rows; start += tileRows) {
        int count = std::min(tileRows, out.rows - start);
        auto col = workspace.rowSlice(0, count), tile = out.rowSlice(start, count);
        im2col<<<grid(col.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(col), h, width, x.cols,
            kernel, stride, pad, ow, start, col.size());
        check(cudaGetLastError());
        linearInto(col, w, tile, bias);
    }
    return out;
}
} // namespace qwen_image
} // namespace fastllm
