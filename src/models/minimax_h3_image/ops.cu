#include "models/minimax_h3_image/ops.h"
#include "models/hy_nomas/parallel.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <type_traits>

namespace fastllm::nomas {
cudaError_t fusedAttention(const void *, const void *, const void *, void *, int, int, int, int, int, bool,
                           bool, int, int, int, float, bool, cudaStream_t, int, int, int);
} // namespace fastllm::nomas

namespace fastllm::minimax_h3_image {
namespace {
template <class T, int N> struct alignas(sizeof(T) * N) Pack {
    T values[N];
};
template <class T, int N> __global__ void gateKernel(const T *input, T *output, size_t count, int width) {
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
            // BF16 inference permits relative-L2 alignment. The approximate
            // exponential avoids libdevice range/polynomial work while keeping
            // the SiLU-before-multiply rounding boundary.
            float exponential;
            if constexpr (std::is_same_v<T, __nv_bfloat16>)
                exponential = __expf(-g);
            else
                exponential = expf(-g);
            result.values[j] = T(float(T(g / (1.f + exponential))) * float(values.values[j]));
        }
        y[i] = result;
    }
}
template <class T> __device__ float imagePixel(const T *patches, int y, int x, int c, int width) {
    return float(patches[(size_t(y / 16) * (width / 16) + x / 16) * 768 + c * 256 + y % 16 * 16 + x % 16]);
}
template <bool Rotary>
__global__ void qkvNormKernel(__nv_bfloat16 *qkv, const __nv_bfloat16 *qw, const __nv_bfloat16 *kw,
                              const float *cs, const float *sn, int tokens, int heads, int rotaryDim) {
    using B = __nv_bfloat16;
    const int lane = threadIdx.x % 32, part = blockIdx.y;
    for (size_t index = size_t(blockIdx.x) * 8 + threadIdx.x / 32; index < size_t(tokens) * heads;
         index += size_t(gridDim.x) * 8) {
        const int token = index / heads, head = index % heads;
        B *x = qkv + (size_t(token) * 3 * heads + part * heads + head) * 128;
        const B *weight = part ? kw : qw;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = float(x[lane + j * 32]);
        float sum = (v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + v[3] * v[3]);
#pragma unroll
        for (int step = 16; step; step /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, step);
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / 128.f + 1e-5f);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = float(B(v[j] * inv * float(weight[lane + j * 32])));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int d = lane + j * 32;
            float result = v[j];
            if constexpr (Rotary) {
                int other = d < rotaryDim / 2 ? d + rotaryDim / 2 : d - rotaryDim / 2;
                // Every lane participates, including the non-rotary tail. Read
                // registers because another lane may already have written x.
                float rotated[4];
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    rotated[k] = __shfl_sync(0xffffffff, v[k], other % 32);
                if (d < rotaryDim) {
                    float value = rotated[other / 32];
                    if (d < rotaryDim / 2)
                        value = -value;
                    size_t p = size_t(token) * rotaryDim + d;
                    result = float(B(result * float(B(cs[p])))) + float(B(value * float(B(sn[p]))));
                }
            }
            x[d] = B(result);
        }
    }
}
template <bool Scatter>
__global__ void qkvVector(Pack<__nv_bfloat16, 8> *qkv, const Pack<__nv_bfloat16, 8> *qw,
                          const Pack<__nv_bfloat16, 8> *kw, const float *cs, const float *sn, int tokens,
                          int heads, Pack<__nv_bfloat16, 8> *kv, int groups, int kvRows, int kvOffset) {
    using B = __nv_bfloat16;
    using B4 = Pack<B, 8>;
    int lane = threadIdx.x % 16, part = blockIdx.y;
    for (size_t index = size_t(blockIdx.x) * 16 + threadIdx.x / 16; index < size_t(tokens) * heads;
         index += size_t(gridDim.x) * 16) {
        int token = index / heads, head = index % heads;
        B4 *x = qkv + (size_t(token) * 3 * heads + part * heads + head) * 16;
        auto pack = x[lane];
        if constexpr (Scatter) {
            if (part == 2) {
                int width = heads / groups, group = head / width;
                kv[((size_t(group) * kvRows + kvOffset + token) * 2 * width + width + head % width) * 16 +
                   lane] = pack;
                continue;
            }
        }
        auto weight = (part ? kw : qw)[lane];
        float v[8];
        const unsigned mask = __activemask();
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = float(pack.values[j]);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            sum += v[j] * v[j];
#pragma unroll
        for (int step = 8; step; step /= 2)
            sum += __shfl_down_sync(mask, sum, step, 16);
        float inv = rsqrtf(__shfl_sync(mask, sum, 0, 16) / 128.f + 1e-5f);
        unsigned normalized[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            union Bits {
                __nv_bfloat162 pair;
                unsigned u;
            } bits;
            bits.pair = __floats2bfloat162_rn(v[j * 2] * inv * float(weight.values[j * 2]),
                                              v[j * 2 + 1] * inv * float(weight.values[j * 2 + 1]));
            normalized[j] = bits.u;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            union Bits {
                __nv_bfloat162 pair;
                unsigned u;
            } value, other;
            value.u = normalized[j];
            other.u = __shfl_sync(mask, normalized[j], lane < 6 ? lane + 6 : lane - 6, 16);
            if (lane < 12) {
                if (lane < 6)
                    other.pair = __hneg2(other.pair);
                size_t p = size_t(token) * 96 + lane * 8 + j * 2;
                auto c = __floats2bfloat162_rn(cs[p], cs[p + 1]),
                     si = __floats2bfloat162_rn(sn[p], sn[p + 1]);
                value.pair = __hadd2(__hmul2(value.pair, c), __hmul2(other.pair, si));
            }
            pack.values[j * 2] = value.pair.x;
            pack.values[j * 2 + 1] = value.pair.y;
        }
        if constexpr (Scatter) {
            if (part == 1) {
                int width = heads / groups, group = head / width;
                kv[((size_t(group) * kvRows + kvOffset + token) * 2 * width + head % width) * 16 + lane] =
                    pack;
            } else
                x[lane] = pack;
        } else
            x[lane] = pack;
    }
}

// Image VAE: 64-wide heads, 32 rotary dimensions, FP32 RMS accumulation.
// Preserve the BF16 normalization and rotary multiply/add rounding boundaries.
// Eight lanes own one head; packed loads replace two scalar warp passes.
__global__ void vaeQkvVector(Pack<__nv_bfloat16, 8> *qkv, const float *cs, const float *sn, int tokens,
                             int sequence) {
    const int lane = threadIdx.x % 8, part = blockIdx.y;
    for (size_t index = size_t(blockIdx.x) * 32 + threadIdx.x / 8; index < size_t(tokens) * 16;
         index += size_t(gridDim.x) * 32) {
        const int token = index / 16, head = index % 16;
        auto *x = qkv + (size_t(token) * 48 + part * 16 + head) * 8;
        auto pack = x[lane];
        float v[8], sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = float(pack.values[j]);
            sum += v[j] * v[j];
        }
        const unsigned mask = __activemask();
#pragma unroll
        for (int d = 4; d; d /= 2)
            sum += __shfl_down_sync(mask, sum, d, 8);
        const float inv = rsqrtf(__shfl_sync(mask, sum, 0, 8) / 64.f + 1e-5f);
        unsigned normalized[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            union Bits {
                __nv_bfloat162 pair;
                unsigned u;
            } value;
            value.pair = __floats2bfloat162_rn(v[j * 2] * inv, v[j * 2 + 1] * inv);
            normalized[j] = value.u;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            union Bits {
                __nv_bfloat162 pair;
                unsigned u;
            } value, other;
            value.u = normalized[j];
            other.u = __shfl_sync(mask, normalized[j], lane < 2 ? lane + 2 : lane - 2, 8);
            if (lane < 4) {
                if (lane < 2)
                    other.pair = __hneg2(other.pair);
                const size_t pos = size_t(token % sequence) * 32 + lane * 8 + j * 2;
                auto cosine = __floats2bfloat162_rn(cs[pos], cs[pos + 1]);
                auto sine = __floats2bfloat162_rn(sn[pos], sn[pos + 1]);
                value.pair = __hadd2(__hmul2(value.pair, cosine), __hmul2(other.pair, sine));
            }
            pack.values[j * 2] = value.pair.x;
            pack.values[j * 2 + 1] = value.pair.y;
        }
        x[lane] = pack;
    }
}
template <bool Residual>
__global__ void normModulateKernel(const Pack<__nv_bfloat16, 8> *input, const Pack<__nv_bfloat16, 8> *other,
                                   const Pack<__nv_bfloat16, 8> *weight, const Pack<__nv_bfloat16, 8> *table,
                                   const int *indices, Pack<__nv_bfloat16, 8> *residual,
                                   Pack<__nv_bfloat16, 8> *output, int tableWidth, int gate, int shift,
                                   int scale) {
    using B = __nv_bfloat16;
    constexpr int width = 5376, packs = width / 8, count = (packs + 255) / 256;
    __shared__ float sums[8], inverse;
    const size_t row = size_t(blockIdx.x) * packs, mod = size_t(indices[blockIdx.x]) * tableWidth;
    float values[count][8];
    float sum = 0;
#pragma unroll
    for (int i = 0; i < count; ++i) {
        const int c = threadIdx.x + i * 256;
        if (c < packs) {
            auto x = input[row + c];
            if constexpr (Residual) {
                auto y = other[row + c], g = table[mod + gate * packs + c];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    x.values[j] = B(float(x.values[j]) + float(B(float(y.values[j]) * float(g.values[j]))));
                residual[row + c] = x;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                values[i][j] = float(x.values[j]);
                sum += values[i][j] * values[i][j];
            }
        }
    }
#pragma unroll
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    if (threadIdx.x % 32 == 0)
        sums[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
#pragma unroll
        for (int d = 4; d; d /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, d);
        if (threadIdx.x == 0)
            inverse = rsqrtf(sum / width + 1e-5f);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < count; ++i) {
        const int c = threadIdx.x + i * 256;
        if (c < packs) {
            auto w = weight[c], sh = table[mod + shift * packs + c], sc = table[mod + scale * packs + c];
            Pack<B, 8> out;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = float(B(values[i][j] * inverse * float(w.values[j])));
                out.values[j] = B(float(B(v * float(B(1.f + float(sc.values[j]))))) + float(sh.values[j]));
            }
            output[row + c] = out;
        }
    }
}
template <bool Residual>
std::pair<nomas::Tensor, nomas::Tensor>
normModulate(const nomas::Tensor &x, const nomas::Tensor *other, const nomas::Tensor &w,
             const nomas::Tensor &m, const nomas::Tensor &idx, int gate, int shift, int scale) {
    if (!x.data || x.dtype != nomas::DType::BF16 || x.rows < 1 || x.cols != 5376 || !w.data ||
        w.dtype != x.dtype || w.size() != size_t(x.cols) || !m.data || m.dtype != x.dtype ||
        m.cols % x.cols || shift < 0 || scale < 0 || m.cols / x.cols <= std::max({gate, shift, scale}) ||
        !idx.data || !idx.isFloat32() || idx.rows != x.rows || idx.cols != 1 ||
        (Residual && (!other || !other->data || other->rows != x.rows || other->cols != x.cols ||
                      other->dtype != x.dtype || gate < 0)))
        throw std::invalid_argument("invalid image residual/RMS modulation input");
    if ((uintptr_t(x.data) | uintptr_t(w.data) | uintptr_t(m.data) | (other ? uintptr_t(other->data) : 0)) %
        16)
        throw std::invalid_argument("image residual/RMS modulation requires aligned inputs");
    nomas::Tensor residual = Residual ? nomas::Tensor(x.rows, x.cols, x.dtype) : nomas::Tensor{};
    nomas::Tensor out(x.rows, x.cols, x.dtype);
    using P = Pack<__nv_bfloat16, 8>;
    normModulateKernel<Residual><<<x.rows, 256, 0, cudaStreamPerThread>>>(
        reinterpret_cast<const P *>(x.data), other ? reinterpret_cast<const P *>(other->data) : nullptr,
        reinterpret_cast<const P *>(w.data), reinterpret_cast<const P *>(m.data),
        reinterpret_cast<const int *>(idx.data), reinterpret_cast<P *>(residual.data),
        reinterpret_cast<P *>(out.data), m.cols / 8, gate, shift, scale);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return {std::move(residual), std::move(out)};
}
template <class T> __device__ float imageBlend(float a, float b, int i, int extent) {
    const float alpha = float(T(float(T(float(i))) / extent));
    return float(T(float(T(a * float(T(1.f - alpha)))) + float(T(b * alpha))));
}
template <class T>
__global__ void stitchKernel(const T *tile, const T *above, const T *left, ImageTile spec, uint8_t *rgba,
                             int imageWidth, int *invalid) {
    size_t count = size_t(spec.copyHeight) * spec.copyWidth;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        int y = i / spec.copyWidth, x = i % spec.copyWidth;
        const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
        size_t dst = (size_t(spec.y + y) * imageWidth + spec.x + x) * 4;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float v = imagePixel(tile, y, x, c, spec.width);
            if (y < spec.overlapY)
                v = imageBlend<T>(imagePixel(above, spec.height - spec.overlapY + y, x, c, spec.width), v, y,
                                  spec.overlapY);
            if (x < spec.overlapX)
                v = imageBlend<T>(imagePixel(left, y, spec.width - spec.overlapX + x, c, spec.width), v, x,
                                  spec.overlapX);
            if (!isfinite(v))
                atomicExch(invalid, 1);
            rgba[dst + c] = uint8_t(fminf(fmaxf(v * scale[c] + mean[c], 0.f), 1.f) * 255.f);
        }
        rgba[dst + 3] = 255;
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
            gateKernel<T, 8><<<std::min<size_t>((count + 255) / 256, 4096), 256, 0, cudaStreamPerThread>>>(
                x, y, count, output.cols / 8);
        } else {
            gateKernel<T, 1>
                <<<std::min<size_t>((output.size() + 255) / 256, 4096), 256, 0, cudaStreamPerThread>>>(
                    x, y, output.size(), output.cols);
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
void vaeQkvNormRotaryInPlace(nomas::Tensor &qkv, const nomas::Tensor &cs, const nomas::Tensor &sn) {
    if (!qkv.data || qkv.rows < 1 || qkv.cols != 3072 || qkv.dtype != nomas::DType::BF16 || !cs.data ||
        !sn.data || !cs.isFloat32() || !sn.isFloat32() || cs.rows < 1 || cs.cols != 32 ||
        sn.rows != cs.rows || sn.cols != 32 || qkv.rows % cs.rows || uintptr_t(qkv.data) % 16)
        throw std::invalid_argument("invalid packed image VAE QKV input");
    vaeQkvVector<<<dim3(std::min<size_t>((size_t(qkv.rows) * 16 + 31) / 32, 4096), 2), 256, 0,
                   cudaStreamPerThread>>>(reinterpret_cast<Pack<__nv_bfloat16, 8> *>(qkv.data),
                                          reinterpret_cast<const float *>(cs.data),
                                          reinterpret_cast<const float *>(sn.data), qkv.rows, cs.rows);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void qkvNormRotaryInPlace(nomas::Tensor &x, const nomas::Tensor &qw, const nomas::Tensor &kw,
                          const nomas::Tensor *cs, const nomas::Tensor *sn) {
    if (!x.data || x.rows < 1 || x.cols < 384 || x.cols % 384 || x.dtype != nomas::DType::BF16 || !qw.data ||
        !kw.data || qw.size() != 128 || kw.size() != 128 || qw.dtype != x.dtype || kw.dtype != x.dtype ||
        bool(cs) != bool(sn) ||
        (cs &&
         (!cs->data || !sn->data || !cs->isFloat32() || !sn->isFloat32() || cs->rows != x.rows ||
          sn->rows != x.rows || cs->cols != sn->cols || cs->cols < 2 || cs->cols > 128 || cs->cols % 2)))
        throw std::invalid_argument("invalid packed image QK norm/RoPE input");
    const int heads = x.cols / 384;
    dim3 blocks(std::min<size_t>((size_t(x.rows) * heads + 7) / 8, 4096), 2);
    auto launch = [&](auto rotary) {
        qkvNormKernel<decltype(rotary)::value><<<blocks, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<__nv_bfloat16 *>(x.data), reinterpret_cast<const __nv_bfloat16 *>(qw.data),
            reinterpret_cast<const __nv_bfloat16 *>(kw.data),
            cs ? reinterpret_cast<const float *>(cs->data) : nullptr,
            sn ? reinterpret_cast<const float *>(sn->data) : nullptr, x.rows, heads, cs ? cs->cols : 0);
    };
    if (cs && cs->cols == 96 && !(uintptr_t(x.data) % 16) && !(uintptr_t(qw.data) % 16) &&
        !(uintptr_t(kw.data) % 16)) {
        using P = Pack<__nv_bfloat16, 8>;
        qkvVector<false><<<dim3(std::min<size_t>((size_t(x.rows) * heads + 15) / 16, 4096), 2), 256, 0,
                           cudaStreamPerThread>>>(
            reinterpret_cast<P *>(x.data), reinterpret_cast<const P *>(qw.data),
            reinterpret_cast<const P *>(kw.data), reinterpret_cast<const float *>(cs->data),
            reinterpret_cast<const float *>(sn->data), x.rows, heads, nullptr, 1, 0, 0);
    } else if (cs)
        launch(std::true_type{});
    else
        launch(std::false_type{});
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
nomas::Tensor rmsModulate(const nomas::Tensor &x, const nomas::Tensor &w, const nomas::Tensor &m,
                          const nomas::Tensor &idx, int shift, int scale) {
    return normModulate<false>(x, nullptr, w, m, idx, -1, shift, scale).second;
}
std::pair<nomas::Tensor, nomas::Tensor>
residualRmsModulate(const nomas::Tensor &x, const nomas::Tensor &other, const nomas::Tensor &w,
                    const nomas::Tensor &m, const nomas::Tensor &idx, int gate, int shift, int scale) {
    return normModulate<true>(x, &other, w, m, idx, gate, shift, scale);
}
nomas::Tensor prepareSequenceQkv(nomas::Tensor &x, const nomas::Tensor &qw, const nomas::Tensor &kw,
                                 const nomas::Tensor &cs, const nomas::Tensor &sn, int groups,
                                 int gatheredRows, int rowOffset) {
    if (!x.data || x.rows < 1 || x.cols < 384 || x.cols % 384 || x.dtype != nomas::DType::BF16 ||
        groups < 1 || (x.cols / 384) % groups || !qw.data || !kw.data || qw.size() != 128 ||
        kw.size() != 128 || qw.dtype != x.dtype || kw.dtype != x.dtype || !cs.data || !sn.data ||
        !cs.isFloat32() || !sn.isFloat32() || cs.rows != x.rows || sn.rows != x.rows || cs.cols != 96 ||
        sn.cols != 96 || (uintptr_t(x.data) | uintptr_t(qw.data) | uintptr_t(kw.data)) % 16)
        throw std::invalid_argument("invalid image sequence QKV preparation input");
    if (!gatheredRows)
        gatheredRows = x.rows;
    if (rowOffset < 0 || gatheredRows < x.rows || rowOffset > gatheredRows - x.rows ||
        gatheredRows > INT_MAX / groups)
        throw std::invalid_argument("invalid image sequence QKV destination rows");
    const int heads = x.cols / 384;
    nomas::Tensor kv(groups * gatheredRows, heads / groups * 256, x.dtype);
    using P = Pack<__nv_bfloat16, 8>;
    qkvVector<true><<<dim3(std::min<size_t>((size_t(x.rows) * heads + 15) / 16, 4096), 3), 256, 0,
                      cudaStreamPerThread>>>(
        reinterpret_cast<P *>(x.data), reinterpret_cast<const P *>(qw.data),
        reinterpret_cast<const P *>(kw.data), reinterpret_cast<const float *>(cs.data),
        reinterpret_cast<const float *>(sn.data), x.rows, heads, reinterpret_cast<P *>(kv.data), groups,
        gatheredRows, rowOffset);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return kv;
}
nomas::Tensor sequenceAttention(const nomas::Tensor &qkv, const nomas::Tensor &kv, int heads, int firstHead,
                                int headCount) {
    if (!headCount)
        headCount = heads;
    if (heads < 1 || firstHead < 0 || headCount < 1 || firstHead + headCount > heads || !qkv.data ||
        !kv.data || qkv.rows < 1 || kv.rows < 1 || qkv.dtype != nomas::DType::BF16 || kv.dtype != qkv.dtype ||
        qkv.cols != heads * 384 || kv.cols != headCount * 256)
        throw std::invalid_argument("invalid image sequence attention input");
    nomas::Tensor out(qkv.rows, headCount * 128, qkv.dtype);
    auto status =
        nomas::fusedAttention(qkv.data + firstHead * 128, kv.data, kv.data + headCount * 128, out.data,
                              qkv.rows, kv.rows, headCount, headCount, 128, false, false, 0, 0, 0,
                              0.08838834764831845f, false, cudaStreamPerThread, qkv.cols, kv.cols, 1);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return out;
}
namespace {
__global__ void copyHeadGroup(const uint4 *input, uint4 *output, size_t count, int width, int stride,
                              int first) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        output[i / width * stride + first + i % width] = input[i];
}
} // namespace
void copyAttentionHeads(const nomas::Tensor &input, nomas::Tensor &output, int firstHead) {
    if (!input.data || !output.data || input.dtype != nomas::DType::BF16 || output.dtype != input.dtype ||
        input.rows != output.rows || input.cols % 128 || output.cols % 128 || firstHead < 0 ||
        firstHead * 128 + input.cols > output.cols || (uintptr_t(input.data) | uintptr_t(output.data)) % 16)
        throw std::invalid_argument("invalid image attention head copy");
    copyHeadGroup<<<std::min<size_t>((input.size() / 8 + 255) / 256, 4096), 256, 0, cudaStreamPerThread>>>(
        reinterpret_cast<const uint4 *>(input.data), reinterpret_cast<uint4 *>(output.data), input.size() / 8,
        input.cols / 8, output.cols / 8, firstHead * 16);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct SequenceAttentionPipeline::Impl {
    int device = -1, major = 0, minor = 0;
    cudaStream_t streams[2]{};
    std::vector<cudaEvent_t> ready, done;
    nomas::Tensor scratch;
    static void check(cudaError_t s) {
        if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
    }
    Impl() {
        check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    }
    bool concurrent(int rows, int ranks) const {
        return major == 12 && minor == 0 && ranks == 2 && rows >= 512 && rows <= 8192;
    }
    void prepare(int groups) {
        for (auto &stream : streams)
            if (!stream) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        while (ready.size() < size_t(groups)) {
            cudaEvent_t event;
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            ready.push_back(event);
        }
        while (done.size() < size_t(groups)) {
            cudaEvent_t event;
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            done.push_back(event);
        }
    }
    void drain() noexcept {
        for (auto stream : streams) if (stream) cudaStreamSynchronize(stream);
    }
    ~Impl() {
        int previous = device; cudaGetDevice(&previous); cudaSetDevice(device);
        drain();
        for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
        for (auto event : ready) cudaEventDestroy(event);
        for (auto event : done) cudaEventDestroy(event);
        cudaSetDevice(previous);
    }
};
SequenceAttentionPipeline::SequenceAttentionPipeline() : impl_(new Impl) {}
SequenceAttentionPipeline::~SequenceAttentionPipeline() = default;
void SequenceAttentionPipeline::release() {
    impl_->scratch = {};
}
int SequenceAttentionPipeline::groups(int rows, int ranks) const {
    // Eight heads per group balance two concurrent kernels with peer copies.
    if (impl_->concurrent(rows, ranks)) return 7;
    return rows >= 4096 ? 4 : rows >= 512 ? 7 : 1;
}
nomas::Tensor SequenceAttentionPipeline::forward(const nomas::Tensor &qkv, const nomas::Tensor &kv,
                                                 int totalRows, int groups,
                                                 const nomas::ParallelContext &parallel) {
    if (!parallel.group || !qkv.data || !kv.data || qkv.dtype != nomas::DType::BF16 ||
        kv.dtype != qkv.dtype || qkv.rows < 1 || qkv.cols % 384 || groups < 1 ||
        (qkv.cols / 384) % groups || kv.rows % groups || totalRows < 1 ||
        totalRows > kv.rows / groups || kv.cols != qkv.cols / 384 / groups * 256)
        throw std::invalid_argument("invalid sequence attention pipeline input");
    auto &i = *impl_;
    int device; Impl::check(cudaGetDevice(&device));
    if (device != i.device) throw std::invalid_argument("sequence attention pipeline device changed");
    const int heads = qkv.cols / 384, width = heads / groups;
    nomas::Tensor output(qkv.rows, heads * 128, qkv.dtype);
    // Restrict the measured concurrency policy to dual SM120 inference. Small
    // inputs, larger topologies and other architectures keep the serial path.
    const bool concurrent = i.concurrent(kv.rows / groups / parallel.size(), parallel.size()) && groups > 1;
    if (!concurrent) {
        release();
        parallel.group->gatherChunks(kv, groups, parallel.rank, [&](int g, const nomas::Tensor &all) {
            auto part = sequenceAttention(qkv, all.rowSlice(0, totalRows), heads, g * width, width);
            copyAttentionHeads(part, output, g * width);
        }, true);
        return output;
    }
    i.prepare(groups);
    // The previous call joined both side streams to the caller stream. Its
    // next ready event therefore orders all scratch reuse without a host wait.
    if (i.scratch.rows != qkv.rows * groups || i.scratch.cols != width * 128)
        i.scratch = nomas::Tensor(qkv.rows * groups, width * 128, qkv.dtype);
    std::vector<nomas::Tensor> parts, gathered;
    gathered.reserve(groups);
    for (int g = 0; g < groups; ++g) parts.push_back(i.scratch.rowSlice(g * qkv.rows, qkv.rows));
    try {
        parallel.group->gatherChunks(kv, groups, parallel.rank, [&](int g, const nomas::Tensor &all) {
            // The NCCL fallback returns a separate temporary per group. Retain
            // it until the caller stream has joined every attention consumer.
            gathered.push_back(all);
            const auto stream = i.streams[g % 2];
            // gatherChunks has inserted this group's transfer dependency into
            // the caller stream. Forward it without waiting for earlier groups.
            Impl::check(cudaEventRecord(i.ready[g], cudaStreamPerThread));
            Impl::check(cudaStreamWaitEvent(stream, i.ready[g], 0));
            auto &part = parts[g];
            Impl::check(nomas::fusedAttention(qkv.data + g * width * 128, all.data,
                all.data + width * 128, part.data, qkv.rows, totalRows, width, width, 128,
                false, false, 0, 0, 0, 0.08838834764831845f, false, stream, qkv.cols, all.cols, 1));
            copyHeadGroup<<<std::min<size_t>((part.size() / 8 + 255) / 256, 4096), 256, 0, stream>>>(
                reinterpret_cast<const uint4 *>(part.data), reinterpret_cast<uint4 *>(output.data),
                part.size() / 8, part.cols / 8, output.cols / 8, g * width * 16);
            Impl::check(cudaGetLastError());
            Impl::check(cudaEventRecord(i.done[g], stream));
        }, true);
        // Deallocation and the next projection stay on the caller stream.
        for (int g = 0; g < groups; ++g)
            Impl::check(cudaStreamWaitEvent(cudaStreamPerThread, i.done[g], 0));
    } catch (...) {
        // Locals must survive consumer/collective failure until side-stream
        // reads and writes finish. ParallelGroup also drains peer transfers.
        i.drain();
        cudaStreamSynchronize(cudaStreamPerThread);
        throw;
    }
    return output;
}
void stitchImageTile(const nomas::Tensor &tile, const nomas::Tensor *above, const nomas::Tensor *left,
                     const ImageTile &s, nomas::Tensor &rgba, nomas::Tensor &invalid) {
    if (s.height < 16 || s.width < 16 || s.height % 16 || s.width % 16 ||
        tile.rows != s.height / 16 * (s.width / 16) || tile.cols != 768 || !tile.data || s.y < 0 || s.x < 0 ||
        s.overlapY < 0 || s.overlapX < 0 || s.overlapY > s.height || s.overlapX > s.width ||
        s.copyHeight < 1 || s.copyWidth < 1 || s.copyHeight > s.height || s.copyWidth > s.width ||
        s.y + s.copyHeight > rgba.rows || s.x + s.copyWidth > rgba.cols || !rgba.isFloat32() || !rgba.data ||
        !invalid.isFloat32() || invalid.size() != 1 || !invalid.data)
        throw std::invalid_argument("invalid decoded image tile");
    auto checkNeighbour = [&](const nomas::Tensor *other, int overlap) {
        if (overlap && (!other || !other->data || other->rows != tile.rows || other->cols != tile.cols ||
                        other->dtype != tile.dtype))
            throw std::invalid_argument("invalid decoded image neighbour");
    };
    checkNeighbour(above, s.overlapY);
    checkNeighbour(left, s.overlapX);
    auto launch = [&](auto scalar) {
        using T = decltype(scalar);
        stitchKernel<T><<<std::min<size_t>((size_t(s.copyHeight) * s.copyWidth + 255) / 256, 4096), 256, 0,
                          cudaStreamPerThread>>>(reinterpret_cast<const T *>(tile.data),
                                                 above ? reinterpret_cast<const T *>(above->data) : nullptr,
                                                 left ? reinterpret_cast<const T *>(left->data) : nullptr, s,
                                                 reinterpret_cast<uint8_t *>(rgba.data), rgba.cols,
                                                 reinterpret_cast<int *>(invalid.data));
    };
    if (tile.dtype == nomas::DType::BF16)
        launch(__nv_bfloat16{});
    else if (tile.dtype == nomas::DType::FP16)
        launch(__half{});
    else
        launch(float{});
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace fastllm::minimax_h3_image
