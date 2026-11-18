#include "models/duo/model.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

namespace fastllm::duo {
namespace {
template <class T>
__global__ void split(const T *in, T *out, size_t n, int width, int heads, int dim, int part) {
    for (size_t i = blockIdx.x * size_t(256) + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % width;
        out[i] = in[(i / width) * width * 3 + (c / dim) * dim * 3 + part * dim + c % dim];
    }
}
template <class T> __global__ void swap(const T *in, T *out, size_t n, int width) {
    for (size_t i = blockIdx.x * size_t(256) + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
        out[i] = in[i / width * width + (i % width + width / 2) % width];
}
__global__ void update(const float *x, const float *y, float *out, size_t n, float a, bool noise) {
    for (size_t i = blockIdx.x * size_t(256) + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
        out[i] = noise ? __fadd_rn(__fmul_rn(x[i], a), __fmul_rn(y[i], 1.f - a))
                       : __fadd_rn(x[i], __fmul_rn(y[i], a));
}
int grid(size_t n) {
    return int(std::min<size_t>((n + 255) / 256, 4096));
}
void check() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace
Tensor uploadFp32(Runtime &rt, const std::vector<float> &values, int rows, int cols) {
    if (rows < 1 || cols < 1 || values.size() != size_t(rows) * cols)
        throw std::invalid_argument("invalid FP32 upload shape");
    rt.activate();
    Tensor out(rows, cols, true);
    auto e =
        cudaMemcpyAsync(out.data, values.data(), out.bytes(), cudaMemcpyHostToDevice, cudaStreamPerThread);
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
    e = cudaStreamSynchronize(cudaStreamPerThread);
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
    return out;
}
Tensor splitQkv(Runtime &rt, const Tensor &x, int heads, int dim, int part) {
    if (heads < 1 || dim < 1 || x.cols != 3 * heads * dim || part < 0 || part > 2 || x.dtype == DType::FP16)
        throw std::invalid_argument("invalid Duo QKV layout");
    rt.activate();
    Tensor out(x.rows, heads * dim, x.dtype);
    if (x.isFloat32())
        split<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>((float *)x.data, (float *)out.data,
                                                                 out.size(), out.cols, heads, dim, part);
    else
        split<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(x.data, out.data, out.size(), out.cols,
                                                                 heads, dim, part);
    check();
    return out;
}
Tensor swapHalves(Runtime &rt, const Tensor &x) {
    if (x.cols % 2)
        throw std::invalid_argument("invalid Duo fused FFN");
    rt.activate();
    Tensor out(x.rows, x.cols, x.dtype);
    if (x.isFloat32())
        swap<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>((float *)x.data, (float *)out.data, x.size(),
                                                              x.cols);
    else
        swap<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(x.data, out.data, x.size(), x.cols);
    check();
    return out;
}
Tensor flowUpdate(Runtime &rt, const Tensor &x, const Tensor &y, float amount) {
    if (!x.isFloat32() || !y.isFloat32() || x.rows != y.rows || x.cols != y.cols)
        throw std::invalid_argument("Duo flow requires matching FP32 tensors");
    rt.activate();
    Tensor out(x.rows, x.cols, true);
    update<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>((float *)x.data, (float *)y.data,
                                                            (float *)out.data, x.size(), amount, false);
    check();
    return out;
}
Tensor flowNoise(Runtime &rt, const Tensor &x, const Tensor &y, float time) {
    if (!x.isFloat32() || !y.isFloat32() || x.rows != y.rows || x.cols != y.cols)
        throw std::invalid_argument("Duo noise requires matching FP32 tensors");
    rt.activate();
    Tensor out(x.rows, x.cols, true);
    update<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>((float *)x.data, (float *)y.data,
                                                            (float *)out.data, x.size(), time, true);
    check();
    return out;
}
} // namespace fastllm::duo
