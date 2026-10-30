#include "models/minimax_h3/runtime.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
namespace {
using B = __nv_bfloat16;
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
__global__ void rmsFused(const B *x, const B *w, B *out, int width, float eps) {
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
    for (int c = threadIdx.x; c < width; c += 256)
        out[row + c] = B(value(x[row + c]) * inv * value(w[c]));
}
template <class T, class U> __global__ void convert(const T *x, U *y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
        y[i] = U(value(x[i]));
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
        // Eager PyTorch rounds cos/sin, both products, then the sum at BF16.
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
__global__ void timeFeatures(const float *times, float *out, size_t n, int d) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int j = i % d;
        float exponent = -9.210340371976184f * float(j % (d / 2));
        float angle = times[i / d] * expf(exponent / float(d / 2));
        out[i] = j < d / 2 ? cosf(angle) : sinf(angle);
    }
}
} // namespace
Tensor Runtime::cast(const Tensor &x, bool fp32) {
    if (x.fp32 == fp32)
        return x;
    Tensor out(x.rows, x.cols, fp32);
    if (fp32)
        convert<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<float>(out), x.size());
    else
        convert<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(x), ptr<B>(out), x.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::rms(const Tensor &x, const Tensor &w, float eps, int width) {
    if (!width)
        width = x.cols;
    require(width > 0 && x.cols % width == 0 && w.size() == size_t(width) && w.fp32 == x.fp32,
            "invalid H3 RMSNorm shape");
    if (x.fp32)
        return norm(x, true, &w, nullptr, eps, width);
    Tensor out(x.rows, x.cols, false);
    rmsFused<<<x.size() / width, 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(w), ptr<B>(out), width,
                                                                eps);
    check(cudaGetLastError());
    return out;
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
    require(idx.fp32 && idx.cols == 1, "invalid H3 row indices");
    Tensor out(idx.rows, x.cols, x.fp32);
    if (x.fp32)
        selectRows<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(out), ptr<int>(idx), out.size(), x.cols, false);
    else
        selectRows<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(out), ptr<int>(idx),
                                                                      out.size(), x.cols, false);
    check(cudaGetLastError());
    return out;
}
void Runtime::scatter(Tensor &out, const Tensor &x, const Tensor &idx) {
    require(idx.fp32 && idx.cols == 1 && idx.rows == x.rows && out.cols == x.cols && out.fp32 == x.fp32,
            "invalid H3 scatter shape");
    if (x.fp32)
        selectRows<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(x), ptr<float>(out),
                                                                    ptr<int>(idx), x.size(), x.cols, true);
    else
        selectRows<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(ptr<B>(x), ptr<B>(out), ptr<int>(idx),
                                                                    x.size(), x.cols, true);
    check(cudaGetLastError());
}
Tensor Runtime::modulateRows(const Tensor &x, const Tensor &m, const Tensor &idx, int shift, int scale) {
    require(x.fp32 == m.fp32 && idx.fp32 && idx.cols == 1 && idx.rows == x.rows && shift >= 0 && scale >= 0 &&
                m.cols >= (std::max(shift, scale) + 1) * x.cols,
            "invalid H3 modulation shape");
    Tensor out(x.rows, x.cols, x.fp32);
    if (x.fp32)
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
    require(x.fp32 == y.fp32 && x.fp32 == m.fp32 && x.rows == y.rows && x.cols == y.cols && idx.fp32 &&
                idx.cols == 1 && idx.rows == x.rows && gate >= 0 && m.cols >= (gate + 1) * x.cols,
            "invalid H3 residual shape");
    Tensor out(x.rows, x.cols, x.fp32);
    if (x.fp32)
        modulation<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(y), ptr<float>(m), ptr<int>(idx), ptr<float>(out), x.size(), x.cols,
            m.cols, gate, 0);
    else
        modulation<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<B>(x), ptr<B>(y), ptr<B>(m), ptr<int>(idx), ptr<B>(out), x.size(), x.cols, m.cols, gate, 0);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::rotary(const Tensor &x, const Tensor &cs, const Tensor &sn, int dim) {
    require(dim > 0 && x.cols % dim == 0 && cs.fp32 && sn.fp32 && cs.rows == x.rows && sn.rows == x.rows &&
                cs.cols == sn.cols && cs.cols <= dim && cs.cols % 2 == 0,
            "invalid H3 rotary shape");
    Tensor out(x.rows, x.cols, x.fp32);
    if (x.fp32)
        rotate<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(cs), ptr<float>(sn), ptr<float>(out), x.size(), x.cols, dim, cs.cols);
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
    if (!q.fp32)
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
    require(x.fp32 && w.fp32 && kernel > 0 && stride > 0 && dilation > 0 && w.cols == x.cols * kernel,
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
    require(w.fp32 && g.fp32 && g.size() == size_t(w.rows), "invalid H3 weight norm");
    Tensor out(w.rows, w.cols, true);
    normalizeWeight<<<w.rows, 256, 0, cudaStreamPerThread>>>(ptr<float>(w), ptr<float>(g), ptr<float>(out),
                                                             w.cols);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::transposeConvWeight(const Tensor &w, int kernel) {
    require(w.fp32 && kernel > 0 && w.cols % kernel == 0, "invalid H3 transposed convolution weight");
    Tensor out(w.cols / kernel, w.rows * kernel, true);
    transposeWeight<<<grid(w.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(w), ptr<float>(out), w.rows,
                                                                     out.rows, kernel, w.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::audioFilter(const Tensor &x, const Tensor &filter, bool up) {
    require(x.fp32 && filter.fp32 && filter.size() == 12, "invalid H3 alias-free filter");
    Tensor out(up ? x.rows * 2 : (x.rows + 1) / 2, x.cols, true);
    filterAudio<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(filter), ptr<float>(out), x.rows, x.cols, filter.size(), up, out.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::snake(const Tensor &x, const Tensor &alpha, const Tensor &beta) {
    require(x.fp32 && alpha.fp32 && beta.fp32 && alpha.size() == size_t(x.cols) &&
                beta.size() == alpha.size(),
            "invalid H3 Snake shape");
    Tensor out(x.rows, x.cols, true);
    snakeAudio<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(alpha), ptr<float>(beta), ptr<float>(out), x.cols, x.size());
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::affine(const Tensor &x, float scale, float bias, bool clamp) {
    require(x.fp32, "H3 affine requires FP32");
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
    require(x.fp32 && v.fp32 && x.rows == v.rows && x.cols == v.cols && sigma > 0,
            "invalid H3 scheduler input");
    Tensor out(x.rows, x.cols, true);
    schedulerStep<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(v), ptr<float>(out), 1.f - time, next / sigma, x.size());
    check(cudaGetLastError());
    return out;
}
} // namespace minimax_h3
} // namespace fastllm
