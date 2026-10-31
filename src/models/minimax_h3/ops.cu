#include "models/minimax_h3/runtime.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <stdexcept>
#include <type_traits>

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
template <bool Modulate, class Input, class T>
__global__ void rmsFused(const Input *x, const Input *w, T *out, int width, float eps, const T *mod,
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
        float v = rounded<T>(value(x[row + c]) * inv * value(w[c]));
        if constexpr (Modulate) {
            size_t p = size_t(indices[blockIdx.x]) * modCols + c;
            v = rounded<T>(v * rounded<T>(1.f + value(mod[p + scale * width]))) +
                value(mod[p + shift * width]);
        }
        out[row + c] = T(v);
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
template <class T>
__device__ float blendVideo(float a, float b, int position, int extent) {
    float wb = rounded<T>(float(position) / extent), wa = rounded<T>(1.f - wb);
    return rounded<T>(rounded<T>(a * wa) + rounded<T>(b * wb));
}
template <class T>
__device__ float patchPixel(const T *patches, int t, int y, int x, int c, int height, int width) {
    size_t row = (size_t(t / 4) * height + y / 16) * width + x / 16;
    int column = ((c * 4 + t % 4) * 16 + y % 16) * 16 + x % 16;
    return value(patches[row * 3072 + column]);
}
template <class T>
__global__ void stitchVideoKernel(const T *tile, const T *above, const T *left, float *out,
                                  size_t n, int tileH, int tileW, int height, int width,
                                  int y0, int x0, int h, int w, int overlapY, int overlapX) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % 3, x = i / 3 % w, y = i / (size_t(3) * w) % h;
        int t = i / (size_t(3) * w * h);
        float v = patchPixel(tile, t, y, x, c, tileH, tileW);
        // Both neighbours are raw tiles; vertical blending precedes horizontal blending.
        if (y < overlapY)
            v = blendVideo<T>(patchPixel(above, t, tileH * 16 - overlapY + y, x, c, tileH, tileW),
                              v, y, overlapY);
        if (x < overlapX)
            v = blendVideo<T>(patchPixel(left, t, y, tileW * 16 - overlapX + x, c, tileH, tileW),
                              v, x, overlapX);
        out[((size_t(t) * height + y0 + y) * width + x0 + x) * 3 + c] = v;
    }
}
template <class T>
__global__ void copyVideoFramesKernel(float *out, const float *clip, const float *overlap,
                                      size_t n, int framePixels, int overlapFrames) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int t = i / (size_t(framePixels) * 3);
        out[i] = t < overlapFrames ? blendVideo<T>(overlap[i], clip[i], t, overlapFrames) : clip[i];
    }
}
template <int Mode> __device__ float pixelMultiply(float a, float b) {
    if constexpr (Mode == FE_DOWNWARD)
        return __fmul_rd(a, b);
    else if constexpr (Mode == FE_UPWARD)
        return __fmul_ru(a, b);
    else if constexpr (Mode == FE_TOWARDZERO)
        return __fmul_rz(a, b);
    else
        return __fmul_rn(a, b);
}
template <int Mode> __device__ float pixelAdd(float a, float b) {
    if constexpr (Mode == FE_DOWNWARD)
        return __fadd_rd(a, b);
    else if constexpr (Mode == FE_UPWARD)
        return __fadd_ru(a, b);
    else if constexpr (Mode == FE_TOWARDZERO)
        return __fadd_rz(a, b);
    else
        return __fadd_rn(a, b);
}
template <int Mode>
__global__ void videoRGBKernel(const float *pixels, uint8_t *rgb, size_t n) {
    const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        float v = pixelAdd<Mode>(pixelMultiply<Mode>(pixels[i], scale[i % 3]), mean[i % 3]);
        v = pixelMultiply<Mode>(fminf(fmaxf(v, 0.f), 1.f), 255.f);
        if constexpr (Mode == FE_DOWNWARD)
            rgb[i] = __float2int_rd(v);
        else if constexpr (Mode == FE_UPWARD)
            rgb[i] = __float2int_ru(v);
        else if constexpr (Mode == FE_TOWARDZERO)
            rgb[i] = __float2int_rz(v);
        else
            rgb[i] = __float2int_rn(v);
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
// VAE Q/K normalization accumulates in FP32 and rounds only its output.
// A warp owns a 64-wide head, preserving normKernel's 32,16,...,1 reduction tree.
template <class T, bool packed = false>
__global__ void videoQkNormRope64(const T *q, const T *k, const float *cs, const float *sn,
                                 T *qo, T *ko, size_t heads, int headsPerToken, int rotaryDim,
                                 T *vo = nullptr, int sequenceRows = 0) {
    int lane = threadIdx.x % 32;
    for (size_t h = size_t(blockIdx.x) * 8 + threadIdx.x / 32; h < (packed ? 3 : 2) * heads;
         h += size_t(gridDim.x) * 8) {
        size_t head = h % heads;
        size_t token = head / headsPerToken;
        const T *x;
        if constexpr (packed)
            x = q + (token * 3 * headsPerToken + (h / heads) * headsPerToken + head % headsPerToken) * 64;
        else
            x = (h < heads ? q : k) + head * 64;
        T *out = (h < heads ? qo : h < 2 * heads ? ko : vo) + head * 64;
        if constexpr (packed) {
            if (h >= 2 * heads) {
                out[lane] = x[lane];
                out[lane + 32] = x[lane + 32];
                continue;
            }
        }
        float a = value(x[lane]), b = value(x[lane + 32]);
        float sum = a * a + b * b;
        for (int step = 16; step; step /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, step);
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / 64.f + 1e-5f);
        size_t pos = (packed ? token % sequenceRows : token) * rotaryDim;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            int d = lane + j * 32;
            float normalized = rounded<T>(value(x[d]) * inv);
            if (d < rotaryDim) {
                int other = d < rotaryDim / 2 ? d + rotaryDim / 2 : d - rotaryDim / 2;
                float rotated = rounded<T>(value(x[other]) * inv);
                if (d < rotaryDim / 2)
                    rotated = -rotated;
                normalized = rounded<T>(normalized * rounded<T>(cs[pos + d])) +
                             rounded<T>(rotated * rounded<T>(sn[pos + d]));
            }
            out[d] = T(normalized);
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
Tensor Runtime::videoRms(const Tensor &x, const Tensor &w) {
    require(x.isFloat32() && w.isFloat32() && w.size() == size_t(x.cols), "invalid VAE RMSNorm");
    Tensor out(x.rows, x.cols, dtype());
    if (out.dtype == DType::FP16)
        rmsFused<false, float, H><<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(w), ptr<H>(out), x.cols, 1e-5f, nullptr, nullptr, 0, 0, 0);
    else if (out.dtype == DType::BF16)
        rmsFused<false, float, B><<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(w), ptr<B>(out), x.cols, 1e-5f, nullptr, nullptr, 0, 0, 0);
    else
        rmsFused<false, float, float><<<x.rows, 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(w), ptr<float>(out), x.cols, 1e-5f, nullptr, nullptr, 0, 0, 0);
    check(cudaGetLastError());
    return out;
}
std::pair<Tensor, Tensor> Runtime::videoQkNormRotary(const Tensor &q, const Tensor &k,
                                                   const Tensor &cs, const Tensor &sn) {
    require(q.rows == k.rows && q.cols == k.cols && q.cols % 64 == 0 && q.dtype == k.dtype &&
                cs.isFloat32() && sn.isFloat32() && cs.rows == q.rows && sn.rows == q.rows &&
                cs.cols == sn.cols && cs.cols > 0 && cs.cols <= 64 && cs.cols % 2 == 0,
            "invalid VAE Q/K norm and rotary inputs");
    Tensor qo(q.rows, q.cols, q.dtype), ko(k.rows, k.cols, k.dtype);
    size_t heads = q.size() / 64;
    int blocks = std::min<size_t>((2 * heads + 7) / 8, 4096);
    if (q.dtype == DType::FP16)
        videoQkNormRope64<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ptr<H>(q), ptr<H>(k), ptr<float>(cs), ptr<float>(sn), ptr<H>(qo), ptr<H>(ko),
            heads, q.cols / 64, cs.cols);
    else if (q.dtype == DType::BF16)
        videoQkNormRope64<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ptr<B>(q), ptr<B>(k), ptr<float>(cs), ptr<float>(sn), ptr<B>(qo), ptr<B>(ko),
            heads, q.cols / 64, cs.cols);
    else
        videoQkNormRope64<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ptr<float>(q), ptr<float>(k), ptr<float>(cs), ptr<float>(sn), ptr<float>(qo), ptr<float>(ko),
            heads, q.cols / 64, cs.cols);
    check(cudaGetLastError());
    return {qo, ko};
}
std::array<Tensor, 3> Runtime::videoQkvNormRotary(const Tensor &qkv, const Tensor &cs, const Tensor &sn) {
    require(qkv.rows > 0 && qkv.cols > 0 && qkv.cols % 192 == 0 && cs.isFloat32() && sn.isFloat32() &&
                cs.rows > 0 && qkv.rows % cs.rows == 0 && cs.rows == sn.rows && cs.cols == sn.cols &&
                cs.cols > 0 && cs.cols <= 64 && cs.cols % 2 == 0,
            "invalid packed VAE QKV inputs");
    int width = qkv.cols / 3;
    std::array<Tensor, 3> out{Tensor(qkv.rows, width, qkv.dtype), Tensor(qkv.rows, width, qkv.dtype),
                               Tensor(qkv.rows, width, qkv.dtype)};
    size_t heads = out[0].size() / 64;
    int blocks = std::min<size_t>((3 * heads + 7) / 8, 4096);
    auto launch = [&](auto *type) {
        using T = std::remove_pointer_t<decltype(type)>;
        videoQkNormRope64<T, true><<<blocks, 256, 0, cudaStreamPerThread>>>(
            ptr<T>(qkv), nullptr, ptr<float>(cs), ptr<float>(sn), ptr<T>(out[0]), ptr<T>(out[1]),
            heads, width / 64, cs.cols, ptr<T>(out[2]), cs.rows);
    };
    if (qkv.dtype == DType::FP16)
        launch(static_cast<H *>(nullptr));
    else if (qkv.dtype == DType::BF16)
        launch(static_cast<B *>(nullptr));
    else
        launch(static_cast<float *>(nullptr));
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::rms(const Tensor &x, const Tensor &w, float eps, int width) {
    if (!width)
        width = x.cols;
    require(x.dtype != DType::FP16 && width > 0 && x.cols % width == 0 && w.size() == size_t(width) &&
                w.dtype == x.dtype,
            "invalid H3 RMSNorm shape/dtype");
    if (x.isFloat32())
        return norm(x, true, &w, nullptr, eps, width);
    Tensor out(x.rows, x.cols, x.dtype);
    rmsFused<false, B, B><<<x.size() / width, 256, 0, cudaStreamPerThread>>>(
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
Tensor Runtime::stitchVideo(const std::vector<Tensor> &patches, int frames, int tileHeight, int tileWidth,
                            int height, int width, const std::vector<int> &ys, const std::vector<int> &xs) {
    require(frames > 0 && tileHeight > 0 && tileWidth > 0 && height >= tileHeight && width >= tileWidth &&
                int64_t(frames) * height * width <= INT32_MAX / 1024 &&
                !ys.empty() && !xs.empty() && patches.size() == ys.size() * xs.size(), "invalid video tiles");
    auto valid = [](const std::vector<int> &starts, int size, int tile) {
        if (starts.front() != 0 || starts.back() != size - tile)
            return false;
        for (size_t i = 1; i < starts.size(); ++i)
            if (starts[i] <= starts[i - 1] || starts[i] > starts[i - 1] + tile)
                return false;
        return true;
    };
    require(valid(ys, height, tileHeight) && valid(xs, width, tileWidth), "invalid video tile positions");
    for (const auto &p : patches)
        require(p.rows == int64_t(frames) * tileHeight * tileWidth && p.cols == 3072 && p.dtype == dtype(),
                "invalid video tile shape/dtype");
    Tensor out(frames * height * width * 1024, 3, DType::FP32);
    for (size_t y = 0; y < ys.size(); ++y)
        for (size_t x = 0; x < xs.size(); ++x) {
            const auto &tile = patches[y * xs.size() + x];
            const auto &above = y ? patches[(y - 1) * xs.size() + x] : tile;
            const auto &left = x ? patches[y * xs.size() + x - 1] : tile;
            int h = (y + 1 < ys.size() ? ys[y + 1] - ys[y] : tileHeight) * 16;
            int w = (x + 1 < xs.size() ? xs[x + 1] - xs[x] : tileWidth) * 16;
            int oy = y ? (ys[y - 1] + tileHeight - ys[y]) * 16 : 0;
            int ox = x ? (xs[x - 1] + tileWidth - xs[x]) * 16 : 0;
            size_t n = size_t(frames) * 4 * h * w * 3;
            auto launch = [&](auto *p) {
                using T = std::remove_pointer_t<decltype(p)>;
                stitchVideoKernel<<<grid(n), 256, 0, cudaStreamPerThread>>>(
                    p, ptr<T>(above), ptr<T>(left), ptr<float>(out), n, tileHeight, tileWidth,
                    height * 16, width * 16, ys[y] * 16, xs[x] * 16, h, w, oy, ox);
            };
            if (dtype() == DType::FP16)
                launch(ptr<H>(tile));
            else if (dtype() == DType::BF16)
                launch(ptr<B>(tile));
            else
                launch(ptr<float>(tile));
        }
    check(cudaGetLastError());
    return out;
}
void Runtime::copyVideoFrames(Tensor &out, const Tensor &clip, const Tensor *overlap, int framePixels,
                              int start) {
    require(framePixels > 0 && start >= 0 && out.isFloat32() && clip.isFloat32() &&
                out.cols == 3 && clip.cols == 3 && out.rows % framePixels == 0 &&
                clip.rows % framePixels == 0 && int64_t(start) * framePixels + out.rows <= clip.rows &&
                (!overlap || (overlap->isFloat32() && overlap->cols == 3 &&
                              overlap->rows % framePixels == 0)), "invalid video frame copy");
    const float *source = ptr<float>(clip) + size_t(start) * framePixels * 3;
    int overlapFrames = overlap ? overlap->rows / framePixels : 0;
    auto launch = [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        copyVideoFramesKernel<T><<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(out), source, overlap ? ptr<float>(*overlap) : nullptr, out.size(), framePixels,
            overlapFrames);
    };
    if (dtype() == DType::FP16)
        launch(static_cast<H *>(nullptr));
    else if (dtype() == DType::BF16)
        launch(static_cast<B *>(nullptr));
    else
        launch(static_cast<float *>(nullptr));
    check(cudaGetLastError());
}
std::vector<uint8_t> Runtime::downloadVideoRGB(const Tensor &pixels) {
    require(pixels.isFloat32() && pixels.cols == 3 && pixels.rows > 0, "invalid video RGB input");
    // Byte scratch uses the existing allocator; Tensor's public dtypes remain floating point.
    Tensor scratch(int((pixels.size() + 3) / 4), 1, DType::FP32);
    auto launch = [&](auto mode) {
        videoRGBKernel<decltype(mode)::value><<<grid(pixels.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(pixels), ptr<uint8_t>(scratch), pixels.size());
    };
    switch (std::fegetround()) {
    case FE_DOWNWARD: launch(std::integral_constant<int, FE_DOWNWARD>{}); break;
    case FE_UPWARD: launch(std::integral_constant<int, FE_UPWARD>{}); break;
    case FE_TOWARDZERO: launch(std::integral_constant<int, FE_TOWARDZERO>{}); break;
    default: launch(std::integral_constant<int, FE_TONEAREST>{}); break;
    }
    check(cudaGetLastError());
    std::vector<uint8_t> rgb(pixels.size());
    check(cudaMemcpyAsync(rgb.data(), scratch.data, rgb.size(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    return rgb;
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
__global__ void snakeAudio(const float *x, const float *alpha, const float *beta, float *out, int c, size_t n,
                           bool logarithmic) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        float a = logarithmic ? expf(alpha[i % c]) : alpha[i % c];
        float b = logarithmic ? expf(beta[i % c]) : beta[i % c];
        float s = sinf(a * x[i]);
        out[i] = x[i] + (1.f / (b + 1e-9f)) * (s * s);
    }
}
__global__ void poolHeads(const float *x, float *out, size_t n, int heads, int dim, int target) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        int c = i % target, first = c * dim / target, end = ((c + 1) * dim + target - 1) / target;
        float sum = 0;
        for (int d = first; d < end; ++d) {
            float value = 0;
            for (int head = 0; head < heads; ++head)
                value += x[(i / target * heads + head) * dim + d];
            sum += value / heads;
        }
        out[i] = sum / (end - first);
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
    for (int first = 0; first < length;) {
        // Narrow late-stage convolutions are launch-bound. Keep the small tail separate.
        int count = std::min(w.rows <= 64 ? 16384 : 2048, length - first);
        if (count > 2048)
            count -= count % 2048;
        Tensor col(count, w.cols, true);
        audioColumns<<<grid(col.size()), 256, 0, cudaStreamPerThread>>>(
            ptr<float>(x), ptr<float>(col), x.rows, x.cols, kernel, stride, pad, dilation, transposed, first,
            col.size());
        check(cudaGetLastError());
        auto target = out.rowSlice(first, count);
        linearInto(col, w, target, bias);
        first += count;
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
Tensor Runtime::poolAudioHeads(const Tensor &x, int heads, int dimensions) {
    require(x.isFloat32() && heads > 0 && x.cols % heads == 0 && dimensions > 0,
            "invalid H3 audio pooling shape");
    Tensor out(x.rows, dimensions, true);
    poolHeads<<<grid(out.size()), 256, 0, cudaStreamPerThread>>>(ptr<float>(x), ptr<float>(out), out.size(),
                                                                 heads, x.cols / heads, dimensions);
    check(cudaGetLastError());
    return out;
}
Tensor Runtime::snake(const Tensor &x, const Tensor &alpha, const Tensor &beta, bool logarithmic) {
    require(x.isFloat32() && alpha.isFloat32() && beta.isFloat32() && alpha.size() == size_t(x.cols) &&
                beta.size() == alpha.size(),
            "invalid H3 Snake shape");
    Tensor out(x.rows, x.cols, true);
    snakeAudio<<<grid(x.size()), 256, 0, cudaStreamPerThread>>>(
        ptr<float>(x), ptr<float>(alpha), ptr<float>(beta), ptr<float>(out), x.cols, x.size(), logarithmic);
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
