#include "models/qwen_image/runtime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace qwen_image {
namespace {
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
float *ptr(const Tensor &x) {
    if (!x.isFloat32())
        throw std::invalid_argument("Qwen-Image kernels require FP32 tensors");
    return reinterpret_cast<float *>(x.data);
}
int grid(size_t n) { return std::min<size_t>((n + 255) / 256, 4096); }
__global__ void specialNormKernel(const float *x, const float *w, float *y, int c, int mode, float eps) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    float sum = 0;
    for (int j = threadIdx.x; j < c; j += 256) {
        float v = x[size_t(row) * c + j];
        sum += v * v;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (threadIdx.x < d)
            scratch[threadIdx.x] += scratch[threadIdx.x + d];
        __syncthreads();
    }
    float inv = mode == 0 ? rsqrtf(scratch[0] / c + eps) : 1.f / fmaxf(sqrtf(scratch[0]), 1e-12f);
    for (int j = threadIdx.x; j < c; j += 256) {
        float v = x[size_t(row) * c + j] * inv;
        if (mode == 0)
            v *= 1.f + w[j];
        else
            v = v * sqrtf(float(c)) * w[j];
        y[size_t(row) * c + j] = v;
    }
}
__global__ void modulationKernel(const float *x, const float *y, const float *m, float *out, size_t n, int c,
                                 int mc, int rows, int first, int offset) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int r = rows == 1 ? 0 : (int(i / c) < first ? 1 : 0);
        float v = m[size_t(r) * mc + offset * c + i % c];
        out[i] = y ? x[i] + y[i] * tanhf(v) : x[i] * (1.f + v);
    }
}
__global__ void im2col(const float *x, float *out, int h, int w, int c, int kernel, int stride, int pad,
                       int ow, int start, size_t n) {
    int kc = c * kernel * kernel;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int pixel = start + i / kc, k = i % kc, channel = k / (kernel * kernel);
        int ih = pixel / ow * stride - pad + k / kernel % kernel, iw = pixel % ow * stride - pad + k % kernel;
        out[i] = (ih >= 0 && ih < h && iw >= 0 && iw < w) ? x[(size_t(ih) * w + iw) * c + channel] : 0.f;
    }
}
template<class T> __global__ void upsample(const T *x, T *out, size_t n, int h, int w, int ic, int oc, int ft,
                         bool shortcut) {
    int repeats = oc * ft * 4 / ic;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % oc, p = i / oc, oh = p / (2 * w), ow = p % (2 * w);
        int src = shortcut ? (((c * ft + ft - 1) * 2 + oh % 2) * 2 + ow % 2) / repeats : c;
        out[i] = x[(size_t(oh / 2) * w + ow / 2) * ic + src];
    }
}
__global__ void avgDown(const float *x, float *out, size_t n, int w, int ic, int oc, int fs, int ft) {
    int group = ic * ft * fs * fs / oc;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % oc, p = i / oc, oh = p / (w / fs), ow = p % (w / fs);
        float sum = 0;
        for (int k = 0; k < group; ++k) {
            int z = c * group + k, sw = z % fs, sh = z / fs % fs, t = z / (fs * fs) % ft,
                ch = z / (fs * fs * ft);
            if (t == ft - 1)
                sum += x[(size_t(oh * fs + sh) * w + ow * fs + sw) * ic + ch];
        }
        out[i] = sum / group;
    }
}
} // namespace
Runtime::Runtime(int device) : nomas::Runtime(device, true) {}
Tensor Runtime::specialNorm(const Tensor &x, const Tensor &w, int mode, float eps) {
    if (w.size() != size_t(x.cols) || w.dtype != x.dtype)
        throw std::invalid_argument("special norm shape");
    Tensor out(x.rows, x.cols, true);
    specialNormKernel<<<x.rows, 256, 0, cudaStreamPerThread>>>(ptr(x), ptr(w), ptr(out), x.cols, mode, eps);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::scale(const Tensor &x, const Tensor &m, int first, int offset) {
    if (m.cols < (offset + 1) * x.cols || (m.rows != 1 && m.rows != 2) || m.dtype != x.dtype)
        throw std::invalid_argument("modulation shape");
    Tensor out(x.rows, x.cols, true);
    modulationKernel<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr(x), nullptr, ptr(m), ptr(out), x.size(), x.cols, m.cols, m.rows, first, offset);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::gatedAdd(const Tensor &x, const Tensor &y, const Tensor &m, int first, int offset) {
    if (x.rows != y.rows || x.cols != y.cols || x.dtype != y.dtype || m.dtype != x.dtype ||
        m.cols < (offset + 1) * x.cols || (m.rows != 1 && m.rows != 2))
        throw std::invalid_argument("gated residual shape");
    Tensor out(x.rows, x.cols, true);
    modulationKernel<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr(x), ptr(y), ptr(m), ptr(out), x.size(), x.cols, m.cols, m.rows, first, offset);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::conv2d(const Tensor &x, const Tensor &w, const Tensor *bias, int h, int width, int kernel,
                       int stride, int pad, int padEnd) {
    if (x.rows != h * width || w.cols != x.cols * kernel * kernel)
        throw std::invalid_argument("convolution shape");
    int oh = (h + 2 * pad + padEnd - kernel) / stride + 1,
        ow = (width + 2 * pad + padEnd - kernel) / stride + 1;
    if (oh < 1 || ow < 1)
        throw std::invalid_argument("empty convolution output");
    Tensor out(oh * ow, w.rows, true);
    for (int start = 0; start < out.rows; start += 4096) {
        int count = std::min(4096, out.rows - start);
        Tensor col(count, w.cols, true);
        im2col<<<grid(col.size()), 256, 0, cudaStreamPerThread>>>(ptr(x), ptr(col), h, width, x.cols, kernel,
                                                                  stride, pad, ow, start, col.size());
        check(cudaGetLastError());
        auto tile = linear(col, w, bias);
        check(cudaMemcpyAsync(out.rowSlice(start, count).data, tile.data, tile.bytes(),
                              cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    return out;
}
Tensor Runtime::resize2x(const Tensor &x, int h, int w) { return duplicateUpsample(x, h, w, x.cols, 0); }
Tensor Runtime::duplicateUpsample(const Tensor &x, int h, int w, int channels, int ft) {
    if (x.rows != h * w || channels < 1 || (ft && (channels * ft * 4) % x.cols))
        throw std::invalid_argument("upsample shape");
    Tensor out(h * w * 4, channels, x.dtype);
    if(x.isFloat32()) upsample<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
        ptr(x), ptr(out), out.size(), h, w, x.cols, channels, std::max(ft, 1), ft != 0);
    else upsample<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
        x.data, out.data, out.size(), h, w, x.cols, channels, std::max(ft, 1), ft != 0);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::averageDownsample(const Tensor &x, int h, int w, int channels, int fs, int ft) {
    if (x.rows != h * w || fs < 1 || ft < 1 || h % fs || w % fs || channels < 1 ||
        (x.cols * ft * fs * fs) % channels)
        throw std::invalid_argument("downsample shape");
    Tensor out(h * w / (fs * fs), channels, true);
    avgDown<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(ptr(x), ptr(out), out.size(), w, x.cols,
                                                               channels, fs, ft);
    check(cudaGetLastError());
    return out;
}
} // namespace qwen_image
} // namespace fastllm
