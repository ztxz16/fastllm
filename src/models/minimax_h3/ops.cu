#include "models/minimax_h3/runtime.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
namespace {
using B = __nv_bfloat16;
using H = __half;
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
template <class T> T *ptr(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
int grid(size_t n) { return std::min<size_t>((n + 255) / 256, 4096); }
template <class T> __device__ float value(T x) { return float(x); }
template <class T> __device__ float rounded(float x) { return value(T(x)); }
// H3 nn.RMSNorm rounds after multiplying the weight. The shared Qwen RMSNorm
// rounds before it, so reusing that BF16 kernel would change the model's arithmetic.
template <bool Modulate>
__global__ void rmsFused(const B *x, const B *w, B *out, int width, float eps, const B *mod,
                         const int *indices, int modCols, int shift, int scale) {
    __shared__ float sums[256];
    float sum = 0;
    size_t row = size_t(blockIdx.x) * width;
    for (int c = threadIdx.x; c < width; c += 256) {
        float v = value(x[row + c]);
        sum += v * v;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step)
            sums[threadIdx.x] += sums[threadIdx.x + step];
        __syncthreads();
    }
    float inv = rsqrtf(sums[0] / width + eps);
    for (int c = threadIdx.x; c < width; c += 256) {
        float v = rounded<B>(value(x[row + c]) * inv * value(w[c]));
        if constexpr (Modulate) {
            size_t p = size_t(indices[blockIdx.x]) * modCols + c;
            v = rounded<B>(v * rounded<B>(1.f + value(mod[p + scale * width]))) +
                value(mod[p + shift * width]);
        }
        out[row + c] = B(v);
    }
}
template <class T>
__global__ void selectRows(const T *x, T *out, const int *indices, size_t n, int c, bool scatter) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        size_t other = size_t(indices[i / c]) * c + i % c;
        if (scatter)
            out[other] = x[i];
        else
            out[i] = x[other];
    }
}
__global__ void siluMultiplyBf16(const B *x, B *out, size_t n, int width) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        size_t p = (i / width) * (2 * width) + i % width;
        float gate = value(x[p + width]);
        // Preserve activation's BF16 result before the elementwise multiply.
        out[i] = B(value(x[p]) * rounded<B>(gate / (1.f + expf(-gate))));
    }
}
template <class T>
__global__ void modulation(const T *x, const T *y, const T *m, const int *indices, T *out, size_t n, int c,
                           int mc, int shift, int scale) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        size_t p = size_t(indices[i / c]) * mc + i % c;
        float v =
            y ? value(x[i]) + rounded<T>(value(y[i]) * value(m[p + shift * c]))
              : rounded<T>(value(x[i]) * rounded<T>(1.f + value(m[p + scale * c]))) + value(m[p + shift * c]);
        out[i] = T(v);
    }
}
__global__ void residualBf16Pairs(const __nv_bfloat162 *x, const __nv_bfloat162 *y,
                                  const __nv_bfloat162 *m, const int *indices, __nv_bfloat162 *out,
                                  int width, int modCols, int gate) {
    size_t row = size_t(blockIdx.x) * width;
    size_t mod = size_t(indices[blockIdx.x]) * modCols + gate * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        // Keep BF16 multiplication and addition separate, matching eager rounding.
        out[row + c] = __hadd2(x[row + c], __hmul2(y[row + c], m[mod + c]));
    }
}
template <class T>
__global__ void rotate(const T *x, const float *cs, const float *sn, T *out, size_t n, int c, int d, int rd) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int j = i % d;
        if (j >= rd) {
            out[i] = x[i];
            continue;
        }
        size_t p = (i / c) * rd + j;
        float other = j < rd / 2 ? -value(x[i + rd / 2]) : value(x[i - rd / 2]);
        // Eager PyTorch rounds cos/sin, both products, then the sum in the activation dtype.
        out[i] = T(rounded<T>(value(x[i]) * rounded<T>(cs[p])) + rounded<T>(other * rounded<T>(sn[p])));
    }
}
__global__ void positionsToRope(const float *pos, const float *inv, float *cs, float *sn, size_t n, int f) {
    int rd = 6 * f;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int j = i % rd % (3 * f);
        float angle = pos[(i / rd) * 3 + j / f] * inv[j % f];
        cs[i] = cosf(angle);
        sn[i] = sinf(angle);
    }
}
// One warp per 128-wide head. Match rmsFused's reduction tree and all eager
// BF16 rounding boundaries, including the norm result before applying RoPE.
__global__ void qkNormRope128(const B *q, const B *k, const B *qw, const B *kw, const float *cs,
                            const float *sn, B *qo, B *ko, size_t heads, int headsPerToken,
                            int rotaryDim, float eps) {
    int lane = threadIdx.x % 32;
    for (size_t h = size_t(blockIdx.x) * 8 + threadIdx.x / 32; h < 2 * heads;
         h += size_t(gridDim.x) * 8) {
        bool isQ = h < heads;
        size_t head = isQ ? h : h - heads, offset = head * 128;
        const B *x = (isQ ? q : k) + offset, *w = isQ ? qw : kw;
        B *out = (isQ ? qo : ko) + offset;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = value(x[lane + j * 32]);
        float sum = (v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + v[3] * v[3]);
#pragma unroll
        for (int step = 16; step; step /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, step);
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / 128.f + eps);
        size_t pos = head / headsPerToken * rotaryDim;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int d = lane + j * 32;
            float normalized = rounded<B>(v[j] * inv * value(w[d]));
            if (d < rotaryDim) {
                int other = d < rotaryDim / 2 ? d + rotaryDim / 2 : d - rotaryDim / 2;
                float rotated = rounded<B>(value(x[other]) * inv * value(w[other]));
                if (d < rotaryDim / 2)
                    rotated = -rotated;
                normalized = rounded<B>(normalized * rounded<B>(cs[pos + d])) +
                             rounded<B>(rotated * rounded<B>(sn[pos + d]));
            }
            out[d] = B(normalized);
        }
    }
}
__global__ void timeFeatures(const float *times, float *out, size_t n, int d) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int j = i % d;
        float exponent = -9.210340371976184f * float(j % (d / 2));
        float angle = times[i / d] * expf(exponent / float(d / 2));
        out[i] = j < d / 2 ? cosf(angle) : sinf(angle);
    }
}
} // namespace
Tensor Runtime::rms(const Tensor &x, const Tensor &w, float eps, int width) {
    if (!width)
        width = x.cols;
    require(x.dtype != DType::FP16 && width > 0 && x.cols % width == 0 && w.size() == size_t(width) &&
                w.dtype == x.dtype,
            "invalid H3 RMSNorm shape/dtype");
    if (x.isFloat32())
        return norm(x, true, &w, nullptr, eps, width);
    Tensor out(x.rows, x.cols, x.dtype);
    rmsFused<false><<<x.size() / width, 256, 0, cudaStreamPerThread>>>(
        ptr<B>(x), ptr<B>(w), ptr<B>(out), width, eps, nullptr, nullptr, 0, 0, 0);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::rmsModulateRows(const Tensor &x, const Tensor &w, float eps, const Tensor &m,
                               const Tensor &idx, int shift, int scale) {
    require(x.dtype != DType::FP16 && x.dtype == w.dtype && x.dtype == m.dtype &&
                w.size() == size_t(x.cols) && idx.isFloat32() && idx.cols == 1 && idx.rows == x.rows &&
                shift >= 0 && scale >= 0 && m.cols >= (std::max(shift, scale) + 1) * x.cols,
            "invalid H3 RMSNorm/modulation shape/dtype");
    if (x.isFloat32())
        return modulateRows(rms(x, w, eps), m, idx, shift, scale);
    Tensor out(x.rows, x.cols, x.dtype);
    rmsFused<true><<<x.rows, 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(w), ptr<B>(out), x.cols,
                                                        eps, ptr<B>(m), ptr<int>(idx), m.cols, shift, scale);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::siluMultiply(const Tensor &x) {
    require(x.cols % 2 == 0, "invalid H3 gated activation width");
    int width = x.cols / 2;
    if (x.dtype != DType::BF16)
        return multiply(columns(x, 0, width), activation(columns(x, width, width)));
    Tensor out(x.rows, width, x.dtype);
    siluMultiplyBf16<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(out),
                                                                   out.size(), width);
    check(cudaGetLastError());
    return out;
}
std::pair<Tensor, Tensor> Runtime::qkNormRotary(const Tensor &q, const Tensor &k, const Tensor &qw,
                                             const Tensor &kw, float eps, const Tensor &cs,
                                             const Tensor &sn, int headDim) {
    require(headDim > 0 && q.rows == k.rows && q.cols == k.cols && q.cols % headDim == 0 &&
                q.dtype == k.dtype && qw.dtype == q.dtype && kw.dtype == q.dtype &&
                qw.size() == size_t(headDim) && kw.size() == size_t(headDim) &&
                cs.isFloat32() && sn.isFloat32() && cs.rows == q.rows && sn.rows == q.rows &&
                cs.cols == sn.cols && cs.cols > 0 && cs.cols <= headDim && cs.cols % 2 == 0,
            "invalid H3 QK norm/RoPE shape/dtype");
    if (q.dtype != DType::BF16 || headDim != 128)
        return {rotary(rms(q, qw, eps, headDim), cs, sn, headDim),
                rotary(rms(k, kw, eps, headDim), cs, sn, headDim)};
    Tensor qo(q.rows, q.cols, q.dtype), ko(k.rows, k.cols, k.dtype);
    size_t heads = q.size() / 128;
    qkNormRope128<<<std::min<size_t>((2 * heads + 7) / 8, 4096), 256, 0, cudaStreamPerThread>>>(
        ptr<B>(q), ptr<B>(k), ptr<B>(qw), ptr<B>(kw), ptr<float>(cs), ptr<float>(sn), ptr<B>(qo),
        ptr<B>(ko), heads, q.cols / 128, cs.cols, eps);
    check(cudaGetLastError());
    return {qo, ko};
}
Tensor Runtime::indices(const std::vector<int> &values) {
    require(!values.empty(), "empty H3 indices");
    Tensor out(values.size(), 1, true);
    check(cudaMemcpyAsync(out.data, values.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice,
                          cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    return out;
}
Tensor Runtime::gather(const Tensor &x, const Tensor &idx) {
    require(idx.isFloat32() && idx.cols == 1, "invalid H3 row indices");
    Tensor out(idx.rows, x.cols, x.dtype);
    if (x.isFloat32())
        selectRows<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(out), ptr<int>(idx), out.size(), x.cols, false);
    else
        selectRows<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<uint16_t>(x), ptr<uint16_t>(out), ptr<int>(idx), out.size(), x.cols, false);
    check(cudaGetLastError());
    return out;
}
void Runtime::scatter(Tensor &out, const Tensor &x, const Tensor &idx) {
    require(idx.isFloat32() && idx.cols == 1 && idx.rows == x.rows && out.cols == x.cols &&
                out.dtype == x.dtype,
            "invalid H3 scatter shape");
    if (x.isFloat32())
        selectRows<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(x), ptr<float>(out),
                                                                    ptr<int>(idx), x.size(), x.cols, true);
    else
        selectRows<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<uint16_t>(x), ptr<uint16_t>(out),
                                                                    ptr<int>(idx), x.size(), x.cols, true);
    check(cudaGetLastError());
}
Tensor Runtime::modulateRows(const Tensor &x, const Tensor &m, const Tensor &idx, int shift, int scale) {
    require(x.dtype != DType::FP16 && x.dtype == m.dtype && idx.isFloat32() && idx.cols == 1 &&
                idx.rows == x.rows && shift >= 0 && scale >= 0 &&
                m.cols >= (std::max(shift, scale) + 1) * x.cols,
            "invalid H3 modulation shape/dtype");
    Tensor out(x.rows, x.cols, x.dtype);
    if (x.isFloat32())
        modulation<float><<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), nullptr, ptr<float>(m), ptr<int>(idx), ptr<float>(out), x.size(), x.cols, m.cols,
            shift, scale);
    else
        modulation<B><<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), nullptr, ptr<B>(m),
                                                                       ptr<int>(idx), ptr<B>(out), x.size(),
                                                                       x.cols, m.cols, shift, scale);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::residualRows(const Tensor &x, const Tensor &y, const Tensor &m, const Tensor &idx, int gate) {
    require(x.dtype != DType::FP16 && x.dtype == y.dtype && x.dtype == m.dtype && x.rows == y.rows &&
                x.cols == y.cols && idx.isFloat32() && idx.cols == 1 && idx.rows == x.rows && gate >= 0 &&
                m.cols >= (gate + 1) * x.cols,
            "invalid H3 residual shape/dtype");
    Tensor out(x.rows, x.cols, x.dtype);
    if (x.isFloat32())
        modulation<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(y), ptr<float>(m), ptr<int>(idx), ptr<float>(out), x.size(), x.cols,
            m.cols, gate, 0);
    else if (x.cols >= 512 && x.cols % 2 == 0 && m.cols % 2 == 0 &&
             ((reinterpret_cast<uintptr_t>(x.data) | reinterpret_cast<uintptr_t>(y.data) |
               reinterpret_cast<uintptr_t>(m.data)) % alignof(__nv_bfloat162) == 0))
        residualBf16Pairs<<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<__nv_bfloat162>(x), ptr<__nv_bfloat162>(y), ptr<__nv_bfloat162>(m), ptr<int>(idx),
            ptr<__nv_bfloat162>(out), x.cols / 2, m.cols / 2, gate);
    else
        modulation<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<B>(x), ptr<B>(y), ptr<B>(m), ptr<int>(idx), ptr<B>(out), x.size(), x.cols, m.cols, gate, 0);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::rotary(const Tensor &x, const Tensor &cs, const Tensor &sn, int dim) {
    require(dim > 0 && x.cols % dim == 0 && cs.isFloat32() && sn.isFloat32() && cs.rows == x.rows &&
                sn.rows == x.rows && cs.cols == sn.cols && cs.cols <= dim && cs.cols % 2 == 0,
            "invalid H3 rotary shape");
    Tensor out(x.rows, x.cols, x.dtype);
    if (x.isFloat32())
        rotate<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(cs), ptr<float>(sn), ptr<float>(out), x.size(), x.cols, dim, cs.cols);
    else if (x.dtype == DType::FP16)
        rotate<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<H>(x), ptr<float>(cs), ptr<float>(sn),
                                                                ptr<H>(out), x.size(), x.cols, dim, cs.cols);
    else
        rotate<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<float>(cs), ptr<float>(sn),
                                                                ptr<B>(out), x.size(), x.cols, dim, cs.cols);
    check(cudaGetLastError());
    return out;
}
std::pair<Tensor, Tensor> Runtime::ropeTable(const std::vector<float> &pos, int f, float theta) {
    require(!pos.empty() && pos.size() % 3 == 0 && f > 0 && theta > 0, "invalid H3 positions");
    Tensor positions(pos.size() / 3, 3, true), cs(pos.size() / 3, 6 * f, true), sn(cs.rows, cs.cols, true);
    check(cudaMemcpyAsync(positions.data, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice,
                          cudaStreamPerThread));
    std::vector<float> frequencies(f);
    for (int j = 0; j < f; ++j)
        frequencies[j] = 1.f / std::pow(theta, float(j) / f);
    Tensor inv(1, f, true);
    check(cudaMemcpyAsync(inv.data, frequencies.data(), f * sizeof(float), cudaMemcpyHostToDevice,
                          cudaStreamPerThread));
    positionsToRope<<<grid(cs.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(positions), ptr<float>(inv), ptr<float>(cs), ptr<float>(sn), cs.size(), f);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(cudaStreamPerThread));
    return {cs, sn};
}
Tensor Runtime::timestepEmbedding(const std::vector<float> &times, int dimensions) {
    require(!times.empty() && dimensions > 0 && dimensions % 2 == 0 && isFloat32(),
            "invalid H3 timestep embedding");
    auto input = upload(times.data(), times.size(), 1);
    Tensor out(times.size(), dimensions, true);
    timeFeatures<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(input), ptr<float>(out),
                                                                    out.size(), dimensions);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::selfAttention(const Tensor &q, const Tensor &k, const Tensor &v, int heads) {
    require(heads > 0 && q.cols % heads == 0, "invalid H3 attention heads");
    if (!q.isFloat32())
        return attention(q, k, v, heads);
    // Match PyTorch's FP32 math SDPA: scale each input before the dot product.
    float scale = float(std::sqrt(1.0 / std::sqrt(double(q.cols / heads))));
    return attentionScaled(affine(q, scale), affine(k, scale), v, heads, 1.f);
}
namespace {
__global__ void audioColumns(const float *x, float *out, int length, int channels, int kernel, int stride,
                             int pad, int dilation, bool transposed, int first, size_t n) {
    int width = channels * kernel;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int t = first + i / width, c = i % width / kernel, k = i % kernel;
        int source = transposed ? t + pad - k : t * stride - pad + k * dilation;
        bool valid = !transposed || (source % stride == 0);
        if (transposed)
            source /= stride;
        out[i] = valid && source >= 0 && source < length ? x[size_t(source) * channels + c] : 0.f;
    }
}
__global__ void normalizeWeight(const float *x, const float *g, float *out, int width) {
    __shared__ float sums[256];
    size_t row = size_t(blockIdx.x) * width;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += 256)
        sum += x[row + c] * x[row + c];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            sums[threadIdx.x] += sums[threadIdx.x + s];
        __syncthreads();
    }
    float scale = g[blockIdx.x] / sqrtf(sums[0]);
    for (int c = threadIdx.x; c < width; c += 256)
        out[row + c] = x[row + c] * scale;
}
__global__ void transposeWeight(const float *x, float *out, int inputs, int outputs, int kernel, size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int k = i % kernel, ic = i / kernel % inputs, oc = i / kernel / inputs;
        out[i] = x[(size_t(ic) * outputs + oc) * kernel + k];
    }
}
__global__ void filterAudio(const float *x, const float *filter, float *out, int length, int channels,
                            int kernel, bool up, size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int t = i / channels, c = i % channels;
        float sum = 0;
        int pad = kernel / 2 - 1, crop = pad * 2 + (kernel - 2) / 2;
        for (int k = 0; k < kernel; ++k) {
            int s = up ? t + crop - k : t * 2 - (kernel / 2 - 1) + k;
            if (up) {
                if (s % 2)
                    continue;
                s = s / 2 - pad;
            }
            s = max(0, min(length - 1, s));
            sum += x[size_t(s) * channels + c] * filter[k];
        }
        out[i] = up ? sum * 2 : sum;
    }
}
__global__ void snakeAudio(const float *x, const float *alpha, const float *beta, float *out, int c,
                           size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        float a = expf(alpha[i % c]), b = expf(beta[i % c]);
        float s = sinf(a * x[i]);
        out[i] = x[i] + (1.f / (b + 1e-9f)) * (s * s);
    }
}
__global__ void affineValues(const float *x, float *out, float scale, float bias, bool clamp, size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        float v = x[i] * scale + bias;
        out[i] = clamp ? fmaxf(-1.f, fminf(1.f, v)) : v;
    }
}
} // namespace
Tensor Runtime::conv1d(const Tensor &x, const Tensor &w, const Tensor *bias, int kernel, int stride, int pad,
                       int dilation, bool transposed) {
    require(x.isFloat32() && w.isFloat32() && kernel > 0 && stride > 0 && dilation > 0 &&
                w.cols == x.cols * kernel,
            "invalid H3 audio convolution");
    int length = transposed ? (x.rows - 1) * stride - 2 * pad + kernel
                            : (x.rows + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    require(length > 0, "empty H3 convolution output");
    Tensor out(length, w.rows, true);
    for (int first = 0; first < length; first += 2048) {
        int count = std::min(2048, length - first);
        Tensor col(count, w.cols, true);
        audioColumns<<<grid(col.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(col), x.rows, x.cols, kernel, stride, pad, dilation, transposed, first,
            col.size());
        check(cudaGetLastError());
        auto y = linear(col, w, bias);
        check(cudaMemcpyAsync(out.rowSlice(first, count).data, y.data, y.bytes(), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
    }
    return out;
}
Tensor Runtime::weightNorm(const Tensor &w, const Tensor &g) {
    require(w.isFloat32() && g.isFloat32() && g.size() == size_t(w.rows), "invalid H3 weight norm");
    Tensor out(w.rows, w.cols, true);
    normalizeWeight<<<w.rows, 256, 0, cudaStreamPerThread>>>(ptr<float>(w), ptr<float>(g), ptr<float>(out),
                                                             w.cols);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::transposeConvWeight(const Tensor &w, int kernel) {
    require(w.isFloat32() && kernel > 0 && w.cols % kernel == 0, "invalid H3 transposed convolution weight");
    Tensor out(w.cols / kernel, w.rows * kernel, true);
    transposeWeight<<<grid(w.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(w), ptr<float>(out), w.rows,
                                                                     out.rows, kernel, w.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::audioFilter(const Tensor &x, const Tensor &filter, bool up) {
    require(x.isFloat32() && filter.isFloat32() && filter.size() == 12, "invalid H3 alias-free filter");
    Tensor out(up ? x.rows * 2 : (x.rows + 1) / 2, x.cols, true);
    filterAudio<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(filter), ptr<float>(out), x.rows, x.cols, filter.size(), up, out.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::snake(const Tensor &x, const Tensor &alpha, const Tensor &beta) {
    require(x.isFloat32() && alpha.isFloat32() && beta.isFloat32() && alpha.size() == size_t(x.cols) &&
                beta.size() == alpha.size(),
            "invalid H3 Snake shape");
    Tensor out(x.rows, x.cols, true);
    snakeAudio<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(alpha), ptr<float>(beta), ptr<float>(out), x.cols, x.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::affine(const Tensor &x, float scale, float bias, bool clamp) {
    require(x.isFloat32(), "H3 affine requires FP32");
    Tensor out(x.rows, x.cols, true);
    affineValues<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(x), ptr<float>(out), scale, bias,
                                                                  clamp, x.size());
    check(cudaGetLastError());
    return out;
}
namespace {
__global__ void schedulerStep(const float *x, const float *v, float *out, float sigmaFromTime, float ratio,
                              size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        float denoised = x[i] + sigmaFromTime * v[i];
        out[i] = ratio * x[i] + (1.f - ratio) * denoised;
    }
}
} // namespace
Tensor Runtime::step(const Tensor &x, const Tensor &v, float time, float sigma, float next) {
    require(x.isFloat32() && v.isFloat32() && x.rows == v.rows && x.cols == v.cols && sigma > 0,
            "invalid H3 scheduler input");
    Tensor out(x.rows, x.cols, true);
    schedulerStep<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(v), ptr<float>(out), 1.f - time, next / sigma, x.size());
    check(cudaGetLastError());
    return out;
}
} // namespace minimax_h3
} // namespace fastllm
