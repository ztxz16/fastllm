#include "models/qwen_image/runtime.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace fastllm {
namespace qwen_image {
namespace {
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
using B = __nv_bfloat16;
template<class T> T *ptr(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
template<class F> void dispatch(const Tensor &x, F f) {
    if (x.isFloat32())
        f(static_cast<float *>(nullptr));
    else if (x.dtype == nomas::DType::BF16)
        f(static_cast<B *>(nullptr));
    else
        throw std::invalid_argument("Qwen-Image kernels require BF16 or FP32 tensors");
}
int grid(size_t n) { return std::min<size_t>((n + 255) / 256, 4096); }
// Same descending reduction tree as the materialized native LayerNorm.
__device__ float sumBlock(float v, float *scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int d = 128; d >= 64; d /= 2) {
        if (threadIdx.x < d)
            scratch[threadIdx.x] += scratch[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float sum = scratch[threadIdx.x] + scratch[threadIdx.x + 32];
        for (int d = 16; d; d /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (threadIdx.x == 0)
            scratch[0] = sum;
    }
    __syncthreads();
    return scratch[0];
}
template<class T>
__global__ void normScaleKernel(const T *x, const T *mod, T *out, int mc, int rows, int first, int offset) {
    __shared__ float scratch[256];
    constexpr int c = 4096;
    int row = blockIdx.x, mr = rows == 1 || row >= first ? 0 : 1;
    float values[16], sum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = float(x[size_t(row) * c + threadIdx.x + i * 256]);
        sum += values[i];
    }
    float mean = sumBlock(sum, scratch) / c;
    __syncthreads();
    sum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] -= mean;
        sum += values[i] * values[i];
    }
    float inv = rsqrtf(sumBlock(sum, scratch) / c + 1e-6f);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int col = threadIdx.x + i * 256;
        float scale = float(T(1.f + float(mod[size_t(mr) * mc + offset * c + col])));
        out[size_t(row) * c + col] = T(float(T(values[i] * inv)) * scale);
    }
}
template<class T>
__global__ void qkNormRopeKernel(const T *q, const T *k, const T *qw, const T *kw,
                                const float *cs, const float *sn, T *oq, T *ok, int rows, int localHeads) {
    int blocks = rows * localHeads / 4;
    bool key = blockIdx.x >= blocks;
    int head = (int(blockIdx.x) % blocks) * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const T *x = key ? k : q, *w = key ? kw : qw;
    T *out = key ? ok : oq;
    size_t base = size_t(head) * 128;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = float(x[base + lane + 32 * i]);
    float sum = (v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + v[3] * v[3]);
#pragma unroll
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    float inv = rsqrtf(__shfl_sync(0xffffffffu, sum, 0) / 128 + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = lane + 32 * i, pos = (head / localHeads) * 128 + col;
        float value = float(T(float(T(v[i] * inv)) * float(w[col])));
        float pair = __shfl_xor_sync(0xffffffffu, value, 1);
        out[size_t(head) * 128 + col] = T(value * cs[pos] + (lane % 2 ? pair : -pair) * sn[pos]);
    }
}
template<class T>
__global__ void eulerKernel(T *latent, const T *noise, size_t n, float dt) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
        latent[i] = T(float(latent[i]) + float(T(dt * float(noise[i]))));
}
template<class T>
__global__ void specialNormKernel(const T *x, const T *w, T *y, int c, int mode, float eps) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    float sum = 0;
    for (int j = threadIdx.x; j < c; j += 256) {
        float v = float(x[size_t(row) * c + j]);
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
        float v = float(x[size_t(row) * c + j]) * inv;
        if (mode == 0)
            v *= 1.f + float(w[j]);
        else
            // VAE normalizes in FP32, then rounds normalize and scale separately.
            v = float(T(float(T(v)) * sqrtf(float(c)))) * float(w[j]);
        y[size_t(row) * c + j] = T(v);
    }
}
template<class T, int Pack> struct alignas(sizeof(T) * Pack) Values { T v[Pack]; };
// One warp per pixel, with the same descending sum tree as specialNormKernel.
// Keep normalize, channel scaling, gamma and SiLU rounding boundaries intact.
template<int Channels, bool Silu>
__global__ void vaeNormKernel(const B *x, const B *w, B *out, int rows) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows)
        return;
    constexpr int count = (Channels + 31) / 32;
    float values[count], sums[8]{};
#pragma unroll
    for (int i = 0; i < count; ++i) {
        int col = lane + 32 * i;
        values[i] = col < Channels ? float(x[size_t(row) * Channels + col]) : 0.f;
        // The original 256 threads accumulate channels separated by 256.
        sums[i % 8] += values[i] * values[i];
    }
#pragma unroll
    for (int d = 4; d; d /= 2)
#pragma unroll
        for (int i = 0; i < d; ++i)
            sums[i] += sums[i + d];
    float sum = sums[0];
#pragma unroll
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    float inv = 1.f / fmaxf(sqrtf(__shfl_sync(0xffffffffu, sum, 0)), 1e-12f);
#pragma unroll
    for (int i = 0; i < count; ++i) {
        int col = lane + 32 * i;
        if (col < Channels) {
            float v = float(B(values[i] * inv));
            v = float(B(v * sqrtf(float(Channels))));
            v = float(B(v * float(w[col])));
            out[size_t(row) * Channels + col] = B(Silu ? v / (1.f + expf(-v)) : v);
        }
    }
}
template<int Channels, bool Silu>
Tensor vaeNorm(const Tensor &x, const Tensor &w) {
    Tensor out(x.rows, x.cols, x.dtype);
    vaeNormKernel<Channels, Silu><<<(x.rows + 3) / 4, 128, 0, cudaStreamPerThread>>>(
        ptr<B>(x), ptr<B>(w), ptr<B>(out), x.rows);
    check(cudaGetLastError());
    return out;
}
template<bool Silu>
Tensor vaeNorm(const Tensor &x, const Tensor &w) {
    if (x.dtype != nomas::DType::BF16)
        return {};
    switch (x.cols) {
    case 96:
        return vaeNorm<96, Silu>(x, w);
    case 144:
        return vaeNorm<144, Silu>(x, w);
    case 192:
        return vaeNorm<192, Silu>(x, w);
    case 288:
        return vaeNorm<288, Silu>(x, w);
    case 384:
        return vaeNorm<384, Silu>(x, w);
    case 576:
        return vaeNorm<576, Silu>(x, w);
    case 768:
        return vaeNorm<768, Silu>(x, w);
    case 1152:
        return vaeNorm<1152, Silu>(x, w);
    default:
        return {};
    }
}
// Each half-warp processes one head using 128-bit loads/stores. Adjacent
// RoPE pairs stay in registers; retain both BF16 normalization roundings.
__global__ void packedQKVRopeKernel(const B *x, const B *qw, const B *kw, const float *cs, const float *sn,
                                     B *oq, B *ok, B *ov, int rows, int localHeads) {
    constexpr int pack = 8, lanes = 16, heads = 8;
    int blocks = (rows * localHeads + heads - 1) / heads, kind = blockIdx.x / blocks;
    int head = (blockIdx.x % blocks) * heads + threadIdx.x / lanes, lane = threadIdx.x % lanes;
    if (head >= rows * localHeads) return;
    int col = lane * pack;
    size_t base = (size_t(head / localHeads) * 3 + kind) * localHeads * 128 + (head % localHeads) * 128 + col;
    auto v = *reinterpret_cast<const Values<B, pack> *>(x + base);
    B *dst = kind == 0 ? oq : kind == 1 ? ok : ov;
    if (kind == 2) {
        *reinterpret_cast<Values<B, pack> *>(dst + size_t(head) * 128 + col) = v;
        return;
    }
    const B *weight = kind == 0 ? qw : kw;
    auto w = *reinterpret_cast<const Values<B, pack> *>(weight + col);
    float sum = 0;
#pragma unroll
    for (int j = 0; j < pack; ++j)
        sum += float(v.v[j]) * float(v.v[j]);
#pragma unroll
    for (int d = lanes / 2; d; d /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, d, lanes);
    float inv = rsqrtf(__shfl_sync(0xffffffffu, sum, 0, lanes) / 128 + 1e-6f);
    float values[pack];
#pragma unroll
    for (int j = 0; j < pack; ++j)
        values[j] = float(B(float(B(float(v.v[j]) * inv)) * float(w.v[j])));
    Values<B, pack> out;
#pragma unroll
    for (int j = 0; j < pack; ++j) {
        int pos = (head / localHeads) * 128 + col + j;
        out.v[j] = B(values[j] * cs[pos] + (j % 2 ? values[j ^ 1] : -values[j ^ 1]) * sn[pos]);
    }
    *reinterpret_cast<Values<B, pack> *>(dst + size_t(head) * 128 + col) = out;
}

template<class T, int Pack>
__global__ void siluProductKernel(const T *gate, const T *up, T *out, size_t n, int packedCols = 0) {
    for (size_t p = size_t(blockIdx.x) * 256 + threadIdx.x; p < n / Pack; p += size_t(gridDim.x) * 256) {
        size_t src = packedCols ? (p / (packedCols / Pack)) * (packedCols / Pack) + p : p;
        auto g = reinterpret_cast<const Values<T, Pack> *>(gate)[src];
        auto u = reinterpret_cast<const Values<T, Pack> *>(up)[src];
        Values<T, Pack> result;
#pragma unroll
        for (int j = 0; j < Pack; ++j) {
            float v = float(g.v[j]);
            result.v[j] = T(float(T(v / (1.f + expf(-v)))) * float(u.v[j]));
        }
        reinterpret_cast<Values<T, Pack> *>(out)[p] = result;
    }
}
template<class T, int Pack>
__global__ void modulationKernel(const T *x, const T *y, const T *m, T *out, size_t n, int c,
                                 int mc, int rows, int first, int offset) {
    for (size_t p = size_t(blockIdx.x) * 256 + threadIdx.x; p < n / Pack; p += size_t(gridDim.x) * 256) {
        size_t i = p * Pack;
        int r = rows == 1 ? 0 : (int(i / c) < first ? 1 : 0);
        auto xv = reinterpret_cast<const Values<T, Pack> *>(x)[p];
        auto mv = *reinterpret_cast<const Values<T, Pack> *>(m + size_t(r) * mc + offset * c + i % c);
        Values<T, Pack> yv{}, result;
        if (y)
            yv = reinterpret_cast<const Values<T, Pack> *>(y)[p];
#pragma unroll
        for (int j = 0; j < Pack; ++j) {
            float v = float(mv.v[j]);
            // Keep the eager tanh/product/sum or (1 + scale)/product rounding.
            result.v[j] = T(y ? float(xv.v[j]) + float(T(float(yv.v[j]) * float(T(tanhf(v)))))
                             : float(xv.v[j]) * float(T(1.f + v)));
        }
        reinterpret_cast<Values<T, Pack> *>(out)[p] = result;
    }
}
template<class T>
__global__ void im2col(const T *x, T *out, int h, int w, int c, int kernel, int stride, int pad,
                       int ow, int start, size_t n) {
    int kc = c * kernel * kernel;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int pixel = start + i / kc, k = i % kc, channel = k / (kernel * kernel);
        int ih = pixel / ow * stride - pad + k / kernel % kernel, iw = pixel % ow * stride - pad + k % kernel;
        out[i] = (ih >= 0 && ih < h && iw >= 0 && iw < w) ? x[(size_t(ih) * w + iw) * c + channel] : T(0.f);
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
template<class T>
__global__ void avgDown(const T *x, T *out, size_t n, int w, int ic, int oc, int fs, int ft) {
    int group = ic * ft * fs * fs / oc;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % oc, p = i / oc, oh = p / (w / fs), ow = p % (w / fs);
        float sum = 0;
        for (int k = 0; k < group; ++k) {
            int z = c * group + k, sw = z % fs, sh = z / fs % fs, t = z / (fs * fs) % ft,
                ch = z / (fs * fs * ft);
            if (t == ft - 1)
                sum += float(x[(size_t(oh * fs + sh) * w + ow * fs + sw) * ic + ch]);
        }
        out[i] = T(sum / group);
    }
}
} // namespace
Runtime::Runtime(int device, bool fp32)
    : nomas::Runtime(device, fp32 ? nomas::DType::FP32 : nomas::DType::BF16, {false, true, true}) {
    setLinearAutotuning(!fp32);
}
Tensor Runtime::siluMultiply(const Tensor &gate, const Tensor &up) {
    if (gate.cols % 8)
        return nomas::Runtime::siluMultiply(gate, up);
    if (gate.rows != up.rows || gate.cols != up.cols || gate.dtype != up.dtype)
        throw std::invalid_argument("SiLU product shape");
    Tensor out(gate.rows, gate.cols, gate.dtype);
    dispatch(gate, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        siluProductKernel<T, 8><<<grid(gate.size() / 8), 256, 0, cudaStreamPerThread>>>(
            ptr<T>(gate), ptr<T>(up), ptr<T>(out), gate.size());
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::siluMultiply(const Tensor &packed, bool gateFirst) {
    if (packed.cols % 16)
        return nomas::Runtime::siluMultiply(packed, gateFirst);
    int cols = packed.cols / 2;
    Tensor out(packed.rows, cols, packed.dtype);
    dispatch(packed, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T *g = ptr<T>(packed) + (gateFirst ? 0 : cols), *u = ptr<T>(packed) + (gateFirst ? cols : 0);
        siluProductKernel<T, 8><<<grid(out.size() / 8), 256, 0, cudaStreamPerThread>>>(
            g, u, ptr<T>(out), out.size(), cols);
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::specialNorm(const Tensor &x, const Tensor &w, int mode, float eps) {
    if (w.size() != size_t(x.cols) || w.dtype != x.dtype || (mode != 0 && mode != 1))
        throw std::invalid_argument("special norm shape");
    if (mode == 1) {
        auto out = vaeNorm<false>(x, w);
        if (out.data)
            return out;
    }
    Tensor out(x.rows, x.cols, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        specialNormKernel<<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<T>(x), ptr<T>(w), ptr<T>(out), x.cols, mode, eps);
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::vaeNormSilu(const Tensor &x, const Tensor &w) {
    if (w.size() != size_t(x.cols) || w.dtype != x.dtype)
        throw std::invalid_argument("VAE norm/SiLU shape");
    auto out = vaeNorm<true>(x, w);
    if (out.data)
        return out;
    return activation(specialNorm(x, w, 1));
}
Tensor Runtime::scale(const Tensor &x, const Tensor &m, int first, int offset) {
    if (m.cols < (offset + 1) * x.cols || (m.rows != 1 && m.rows != 2) || m.dtype != x.dtype)
        throw std::invalid_argument("modulation shape");
    Tensor out(x.rows, x.cols, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        if (x.cols % 8 == 0 && m.cols % 8 == 0)
            modulationKernel<T, 8><<<grid(x.size() / 8), 256, 0, cudaStreamPerThread>>>(
                ptr<T>(x), nullptr, ptr<T>(m), ptr<T>(out), x.size(), x.cols, m.cols, m.rows, first, offset);
        else
            modulationKernel<T, 1><<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
                ptr<T>(x), nullptr, ptr<T>(m), ptr<T>(out), x.size(), x.cols, m.cols, m.rows, first, offset);
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::normScale(const Tensor &x, const Tensor &m, int first, int offset) {
    if (x.cols != 4096)
        return scale(norm(x, false), m, first, offset);
    if (offset < 0 || m.cols < (offset + 1) * x.cols || (m.rows != 1 && m.rows != 2) || m.dtype != x.dtype)
        throw std::invalid_argument("norm modulation shape");
    Tensor out(x.rows, x.cols, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        normScaleKernel<<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<T>(x), ptr<T>(m), ptr<T>(out), m.cols, m.rows, first, offset);
    });
    check(cudaGetLastError());
    return out;
}
void Runtime::qkNormRope(const Tensor &q, const Tensor &k, const Tensor &qw, const Tensor &kw,
                         const nomas::RopeTable &table, Tensor &oq, Tensor &ok) {
    if (q.cols < 512 || q.cols > 4096 || q.cols % 512 || k.cols != q.cols || k.rows != q.rows || k.dtype != q.dtype ||
        qw.size() != 128 || kw.size() != 128 || qw.dtype != q.dtype || kw.dtype != q.dtype ||
        oq.rows != q.rows || ok.rows != q.rows || oq.cols != q.cols || ok.cols != q.cols ||
        oq.dtype != q.dtype || ok.dtype != q.dtype || table.cos.rows != q.rows || table.sin.rows != q.rows ||
        table.cos.cols != 128 || table.sin.cols != 128 || !table.cos.isFloat32() || !table.sin.isFloat32())
        throw std::invalid_argument("QK norm/RoPE shape");
    dispatch(q, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        qkNormRopeKernel<<<q.rows * (q.cols / 128) / 2, 128, 0, cudaStreamPerThread>>>(
            ptr<T>(q), ptr<T>(k), ptr<T>(qw), ptr<T>(kw), ptr<float>(table.cos), ptr<float>(table.sin),
            ptr<T>(oq), ptr<T>(ok), q.rows, q.cols / 128);
    });
    check(cudaGetLastError());
}
void Runtime::qkvNormRope(const Tensor &qkv, const Tensor &qw, const Tensor &kw,
                          const nomas::RopeTable &table, Tensor &q, Tensor &k, Tensor &v) {
    if (qkv.dtype != nomas::DType::BF16 || qkv.cols != 3 * q.cols ||
        q.cols < 512 || q.cols > 4096 || q.cols % 512 || k.cols != q.cols || v.cols != q.cols ||
        q.rows != qkv.rows || k.rows != q.rows || v.rows != q.rows ||
        qkv.dtype != q.dtype || k.dtype != q.dtype || v.dtype != q.dtype ||
        qw.size() != 128 || kw.size() != 128 || qw.dtype != q.dtype || kw.dtype != q.dtype ||
        table.cos.rows != q.rows || table.sin.rows != q.rows || table.cos.cols != 128 || table.sin.cols != 128 ||
        !table.cos.isFloat32() || !table.sin.isFloat32())
        throw std::invalid_argument("packed QKV norm/RoPE shape");
    packedQKVRopeKernel<<<((q.rows * (q.cols / 128) + 7) / 8) * 3, 128, 0, cudaStreamPerThread>>>(
        ptr<B>(qkv), ptr<B>(qw), ptr<B>(kw), ptr<float>(table.cos), ptr<float>(table.sin),
        ptr<B>(q), ptr<B>(k), ptr<B>(v), q.rows, q.cols / 128);
    check(cudaGetLastError());
}
void Runtime::eulerStep(Tensor &latent, const Tensor &noise, float dt) {
    if (latent.rows != noise.rows || latent.cols != noise.cols || latent.dtype != noise.dtype)
        throw std::invalid_argument("Euler step shape");
    dispatch(latent, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        eulerKernel<<<grid(latent.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<T>(latent), ptr<T>(noise), latent.size(), dt);
    });
    check(cudaGetLastError());
}
Tensor Runtime::gatedAdd(const Tensor &x, const Tensor &y, const Tensor &m, int first, int offset) {
    if (x.rows != y.rows || x.cols != y.cols || x.dtype != y.dtype || m.dtype != x.dtype ||
        m.cols < (offset + 1) * x.cols || (m.rows != 1 && m.rows != 2))
        throw std::invalid_argument("gated residual shape");
    Tensor out(x.rows, x.cols, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        if (x.cols % 8 == 0 && m.cols % 8 == 0)
            modulationKernel<T, 8><<<grid(x.size() / 8), 256, 0, cudaStreamPerThread>>>(
                ptr<T>(x), ptr<T>(y), ptr<T>(m), ptr<T>(out), x.size(), x.cols, m.cols, m.rows, first, offset);
        else
            modulationKernel<T, 1><<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
                ptr<T>(x), ptr<T>(y), ptr<T>(m), ptr<T>(out), x.size(), x.cols, m.cols, m.rows, first, offset);
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::conv2d(const Tensor &x, const Tensor &w, const Tensor *bias, int h, int width, int kernel,
                       int stride, int pad, int padEnd) {
    if (x.rows != h * width || w.cols != x.cols * kernel * kernel || x.dtype != w.dtype)
        throw std::invalid_argument("convolution shape");
    int oh = (h + 2 * pad + padEnd - kernel) / stride + 1,
        ow = (width + 2 * pad + padEnd - kernel) / stride + 1;
    if (oh < 1 || ow < 1)
        throw std::invalid_argument("empty convolution output");
    if (kernel == 1 && stride == 1 && pad == 0 && padEnd == 0)
        return linear(x, w, bias);
    Tensor out(oh * ow, w.rows, x.dtype);
    // Bound im2col scratch to 128 MiB, reuse it, and write GEMM directly into out.
    int tileRows = std::min<size_t>(32768, (size_t(128) << 20) / (w.cols * (x.isFloat32() ? 4 : 2)));
    tileRows = std::min(out.rows, std::max(256, tileRows / 256 * 256));
    Tensor workspace(tileRows, w.cols, x.dtype);
    for (int start = 0; start < out.rows; start += tileRows) {
        int count = std::min(tileRows, out.rows - start);
        auto col = workspace.rowSlice(0, count);
        dispatch(x, [&](auto *tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            im2col<<<grid(col.size()), 256, 0, cudaStreamPerThread>>>(
                ptr<T>(x), ptr<T>(col), h, width, x.cols, kernel, stride, pad, ow, start, col.size());
        });
        check(cudaGetLastError());
        auto tile = out.rowSlice(start, count);
        linearInto(col, w, tile, bias);
    }
    return out;
}
Tensor Runtime::resize2x(const Tensor &x, int h, int w) { return duplicateUpsample(x, h, w, x.cols, 0); }
Tensor Runtime::duplicateUpsample(const Tensor &x, int h, int w, int channels, int ft) {
    if (x.rows != h * w || channels < 1 || (ft && (channels * ft * 4) % x.cols))
        throw std::invalid_argument("upsample shape");
    Tensor out(h * w * 4, channels, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        upsample<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<T>(x), ptr<T>(out), out.size(), h, w, x.cols, channels, std::max(ft, 1), ft != 0);
    });
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::averageDownsample(const Tensor &x, int h, int w, int channels, int fs, int ft) {
    if (x.rows != h * w || fs < 1 || ft < 1 || h % fs || w % fs || channels < 1 ||
        (x.cols * ft * fs * fs) % channels)
        throw std::invalid_argument("downsample shape");
    Tensor out(h * w / (fs * fs), channels, x.dtype);
    dispatch(x, [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        avgDown<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<T>(x), ptr<T>(out), out.size(), w, x.cols, channels, fs, ft);
    });
    check(cudaGetLastError());
    return out;
}
} // namespace qwen_image
} // namespace fastllm
