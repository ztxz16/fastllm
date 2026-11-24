#pragma once
#include "fastllm-cuda-ordered-reduce.cuh"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <stdexcept>

// Uniform, dynamically indexed pointer lists belong in kernel parameter
// memory. Taking their address otherwise creates a private copy per thread.
#if defined(__CUDACC__) && !defined(__HIPCC__) && CUDART_VERSION >= 12000 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
#define FASTLLM_ROW_PARAMETER const __grid_constant__
#else
#define FASTLLM_ROW_PARAMETER const
#endif

namespace fastllm {
namespace cuda_attn_res {
// Fusion reference: vLLM's Inkling ops/attn_res.py (Apache-2.0).
// https://github.com/vllm-project/vllm/blob/fc1443152c0eb655f5c2762f56b540bebe93b196/vllm/models/inkling2/nvidia/ops/attn_res.py
// Sources are independent [tokens, width] matrices with a common storage type.
// Small lists travel as graph kernel arguments; larger lists live on the device.
template <class T> struct SourceList {
    static constexpr int Inline = 16;
    const T *items[Inline]{};
    const T *const *table = nullptr;

    __device__ const T *get(int index) const { return table ? table[index] : items[index]; }
};
using Sources = SourceList<float>;

// Queries are unscaled. A null output norm returns the FP32 mixture directly.
struct Query {
    const float *values = nullptr;
    const float *norm = nullptr;
};
struct Queries {
    static constexpr int Inline = 64; // 1 KiB of graph-stable query and norm pointers
    Query items[Inline]{};
};

template <class T> static __global__ void StoreSources(const T **dst, SourceList<T> src, int n) {
    if (threadIdx.x < n)
        dst[threadIdx.x] = src.items[threadIdx.x];
}

template<class T> struct alignas(4 * sizeof(T)) Channels4 { T x, y, z, w; };
template <class T> __device__ inline float4 LoadChannels4(const T *src, size_t i) {
    const auto v = reinterpret_cast<const Channels4<T> *>(src)[i];
    return make_float4(float(v.x), float(v.y), float(v.z), float(v.w));
}

// Split a row's warps across residual sources. Each source's two reductions
// run in parallel. Only the first warp computes softmax and its value scales.
template <bool Vectorized, class Round, class Out, class In>
__global__ void AttnResRows(FASTLLM_ROW_PARAMETER SourceList<In> sources, int count, FASTLLM_ROW_PARAMETER Queries queries, int nq, int rows, int width, float eps,
                            bool normalizeValues, Out *output, float *staging) {
    constexpr int tile = Sources::Inline;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int q = blockIdx.x % nq, row = blockIdx.x / nq;
    const size_t offset = size_t(row) * width;
    const Query query = queries.items[q];
    Out *out = output + (size_t(q) * rows + row) * width;
    __shared__ float2 partial[tile];
    __shared__ float coeff[tile], prior;
    __shared__ float normScratch[8];
    extern __shared__ float mixed[];
    float *buffer = nullptr;
    if constexpr (std::is_same_v<Out, float>) buffer = out;
    else if (staging) buffer = staging + (size_t(q) * rows + row) * width;
    if (query.norm && size_t(width) * sizeof(float) <= 16384) buffer = mixed;
    float prevMax = -CUDART_INF_F, prevSum = 0.f;
    float squareSum = 0.f;
    for (int first = 0; first < count; first += tile) {
        const int n = min(count - first, tile);
        for (int s = warp; s < n; s += 8) {
            const In *x = sources.get(first + s) + offset;
            float sq = 0.f, dot = 0.f;
            if (Vectorized &&
                ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(query.values)) & 15) == 0) {
                for (int j = lane; j < width / 4; j += 32) {
                    float4 v = LoadChannels4(x, j);
                    float4 k = reinterpret_cast<const float4 *>(query.values)[j];
                    sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                    dot += v.x * k.x + v.y * k.y + v.z * k.z + v.w * k.w;
                }
            } else {
                for (int j = lane; j < width; j += 32) {
                    const float v = float(x[j]);
                    sq += v * v;
                    dot += v * query.values[j];
                }
            }
            sq = cuda::WarpSum(sq);
            dot = cuda::WarpSum(dot);
            if (!lane)
                partial[s] = make_float2(sq, dot);
        }
        __syncthreads();
        if (!warp) {
            float inv = 0.f, logit = -CUDART_INF_F;
            if (lane < n) {
                const float2 sums = partial[lane];
                inv = rsqrtf(sums.x / width + eps);
                logit = sums.y * inv * rsqrtf(float(width));
            }
            float maximum = fmaxf(prevMax, logit);
            for (int delta = 16; delta; delta /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
            const float weight = expf(logit - maximum);
            const float old = first ? prevSum * expf(prevMax - maximum) : 0.f;
            const float sum = old + cuda::WarpSum(weight);
            if (lane < n)
                coeff[lane] = weight / sum * (normalizeValues ? inv : 1.f);
            if (!lane)
                prior = old / sum;
            prevSum = sum;
            prevMax = maximum;
        }
        __syncthreads();
        for (int j = tid; j < width; j += blockDim.x) {
            float acc = first ? prior * buffer[j] : 0.f;
            for (int s = 0; s < n; ++s)
                acc += coeff[s] * float(sources.get(first + s)[offset + j]);
            if (query.norm || first + n < count) buffer[j] = acc;
            else out[j] = Out(Round(acc));
            if (query.norm && first + n == count)
                squareSum += acc * acc;
        }
        if (first + n < count)
            __syncthreads();
    }
    if (query.norm) {
        float scale = rsqrtf(cuda::WarpFirstBlockSum256(squareSum, normScratch) / width + eps);
        for (int j = tid; j < width; j += blockDim.x)
            out[j] = Out(Round(buffer[j] * scale * query.norm[j]));
    }
}

// Use the vectorized FP32 RMSNorm summation tree for the same block size. Its scale is used
// by low-bit projections downstream, where a rounding boundary can change
// both the activation encoding and the experts selected later.
__device__ inline float Square4(float4 v) { return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
__device__ inline float RowRmsScale(float local, int width, float eps, float *warpSums) {
    const int tid = threadIdx.x;
    const float sum = cuda::WarpSum(local);
    if ((tid & 31) == 0)
        warpSums[tid / 32] = sum;
    __syncthreads();
    // Every warp sees the same totals; no final shared broadcast is needed.
    const int lane = tid & 31;
    return rsqrtf(cuda::WarpSum(lane < blockDim.x / 32 ? warpSums[lane] : 0.f) / width + eps);
}

// One CTA handles the entire row. Independent 128-lane reductions preserve
// the original per-source dot and square-sum trees, in parallel. The mixed
// channels remain in registers through the following RMSNorm.
template <int Slots, class Round, class Out, int FixedWidth = 0, bool Preload = false, class In = float>
__global__ void AttnResRegisterRows(FASTLLM_ROW_PARAMETER SourceList<In> sources, FASTLLM_ROW_PARAMETER Queries queries, int rows, int width,
                                    float eps, bool normalizeValues, Out *output, bool queryFirst) {
    constexpr int threads = 512, packs = 2;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int qi = queryFirst ? blockIdx.x : blockIdx.y;
    const int row = queryFirst ? blockIdx.y : blockIdx.x;
    const int cols = (FixedWidth ? FixedWidth : width) / 4;
    const Query query = queries.items[qi];
    float4 acc[packs]{};
    // A second scoring round leaves a gap in which the mix inputs can be
    // loaded. With a single round it is faster to load them directly below.
    float4 values[Preload ? Slots : 1][packs];
    if constexpr (Preload) {
#pragma unroll
        for (int s = 0; s < Slots; ++s) {
            const In *x = sources.items[s] + size_t(row) * width;
#pragma unroll
            for (int i = 0; i < packs; ++i)
                if (tid + i * threads < cols) values[s][i] = LoadChannels4(x, tid + i * threads);
        }
    }
    __shared__ float2 partial[Slots][128], scores[Slots];
    __shared__ float normScratch[threads / 32];
    for (int s = tid / 128; s < Slots; s += threads / 128) {
        const In *src = sources.items[s] + size_t(row) * width;
        float sq = 0.f, dot = 0.f;
        if constexpr (FixedWidth) {
            // Independent loads can be issued ahead of the ordered FMA chains.
#pragma unroll
            for (int item = 0; item < FixedWidth / 128; ++item) {
                const int j = item * 128 + tid % 128;
                float v = src[j];
                sq += v * v;
                dot += v * query.values[j];
            }
        } else {
#pragma unroll 4
            for (int j = tid % 128; j < width; j += 128) {
                float v = src[j];
                sq += v * v;
                dot += v * query.values[j];
            }
        }
        partial[s][tid % 128] = make_float2(sq, dot);
    }
    __syncthreads();
    if (warp < Slots) {
        const float2 *p = partial[warp] + lane;
        float sq = cuda::WarpSum((p[0].x + p[64].x) + (p[32].x + p[96].x));
        float dot = cuda::WarpSum((p[0].y + p[64].y) + (p[32].y + p[96].y));
        if (!lane) {
            const float inv = rsqrtf(sq / width + eps);
            scores[warp] = make_float2(dot * inv * rsqrtf(float(width)), inv);
        }
    }
    __syncthreads();
    // Every warp needs the same small coefficient vector. Broadcast it
    // within each warp so the mix can start without a third block barrier.
    float logit = lane < Slots ? scores[lane].x : -CUDART_INF_F;
    float mx = logit;
#pragma unroll
    for (int delta = 1; delta < Slots; delta *= 2)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, delta));
    float value = expf(logit - mx), den = 0.f;
#pragma unroll
    for (int s = 0; s < Slots; ++s)
        den += __shfl_sync(0xffffffffu, value, s);
    float weight = value / den;
    if (normalizeValues && lane < Slots) weight *= scores[lane].y;
#pragma unroll
    for (int s = 0; s < Slots; ++s) {
        const In *x = sources.items[s] + size_t(row) * width;
        const float w = __shfl_sync(0xffffffffu, weight, s);
#pragma unroll
        for (int i = 0; i < packs; ++i) {
            if (tid + i * threads >= cols)
                break;
            float4 v = Preload ? values[s][i] : LoadChannels4(x, tid + i * threads);
            acc[i].x += w * v.x;
            acc[i].y += w * v.y;
            acc[i].z += w * v.z;
            acc[i].w += w * v.w;
        }
    }
    float scale = 1.f;
    if (query.norm) {
        float sq = 0.f;
#pragma unroll
        for (int i = 0; i < packs; ++i)
            sq += Square4(acc[i]);
        scale = RowRmsScale(sq, width, eps, normScratch);
    }
    auto out = reinterpret_cast<Channels4<Out> *>(output) + (size_t(qi) * rows + row) * cols;
#pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int j = tid + i * threads;
        if (j >= cols)
            break;
        float4 v = acc[i];
        if (query.norm) {
            float4 w = reinterpret_cast<const float4 *>(query.norm)[j];
            v = make_float4(v.x * scale * w.x, v.y * scale * w.y, v.z * scale * w.z, v.w * scale * w.w);
        }
        out[j] = {Out(Round(v.x)), Out(Round(v.y)), Out(Round(v.z)), Out(Round(v.w))};
    }
}

// Output layout is [query, token, channel]. Vector loads require aligned row views. Query batches and source tiles only
// bound the launch arguments and shared memory, not the permitted shape.
// A narrow output needs FP32 staging only for multi-tile mixtures or a
// normalized row wider than shared memory. FP32 output doubles as staging.
template <class Out>
inline bool NeedsStaging(int count, const Query *queries, int nq, int width) {
    if constexpr (std::is_same_v<Out, float>) return false;
    return count > Sources::Inline || (size_t(width) * sizeof(float) > 16384 &&
        std::any_of(queries, queries + nq, [](const Query &q) { return q.norm != nullptr; }));
}

template <class Round = float, class Out = float, class In>
inline void Launch(SourceList<In> sources, int count, const Query *queries, int nq, int rows, int width, float eps,
                   bool normalizeValues, Out *out, cudaStream_t stream, float *staging = nullptr) {
    if constexpr (!std::is_same_v<Out, float>) {
        if (!staging && NeedsStaging<Out>(count, queries, nq, width))
            throw std::invalid_argument("AttnRes narrow output requires FP32 staging for this shape");
    }
    for (int first = 0; first < nq; first += Queries::Inline) {
        const int n = std::min(nq - first, Queries::Inline);
        Queries group{};
        bool norm = false;
        for (int i = 0; i < n; ++i) {
            group.items[i] = queries[first + i];
            norm |= group.items[i].norm != nullptr;
        }
        size_t bytes = size_t(width) * sizeof(float);
        size_t scratch = norm && bytes <= 16384 ? bytes : 0;
        auto dst = out + size_t(first) * rows * width;
        uintptr_t alignment = reinterpret_cast<uintptr_t>(dst);
        bool fits = count > 0 && count <= 8 && !sources.table && width % 4 == 0 && width <= 4096;
        if (fits) {
            for (int s = 0; s < count; ++s)
                alignment |= reinterpret_cast<uintptr_t>(sources.items[s]);
            for (int q = 0; q < n; ++q)
                alignment |= reinterpret_cast<uintptr_t>(group.items[q].values) |
                             reinterpret_cast<uintptr_t>(group.items[q].norm);
        }
        if (fits && !(alignment & 15)) {
            const auto kernelFor = [&](auto widthTag) {
                constexpr int w = decltype(widthTag)::value;
                const auto select = [&](auto sourceTag) {
                    constexpr int nsrc = decltype(sourceTag)::value;
                    // Single-token latency benefits from preloading across
                    // scoring rounds. Multi-token tiles keep their occupancy.
                    if constexpr (w && nsrc > 4) {
                        if (rows == 1) return AttnResRegisterRows<nsrc, Round, Out, w, true, In>;
                    }
                    return AttnResRegisterRows<nsrc, Round, Out, w, false, In>;
                };
                switch (count) {
                    case 1: return select(std::integral_constant<int, 1>{});
                    case 2: return select(std::integral_constant<int, 2>{});
                    case 3: return select(std::integral_constant<int, 3>{});
                    case 4: return select(std::integral_constant<int, 4>{});
                    case 5: return select(std::integral_constant<int, 5>{});
                    case 6: return select(std::integral_constant<int, 6>{});
                    case 7: return select(std::integral_constant<int, 7>{});
                    default: return select(std::integral_constant<int, 8>{});
                }
            };
            const auto kernel = width == 2304 ? kernelFor(std::integral_constant<int, 2304>{})
                                              : kernelFor(std::integral_constant<int, 0>{});
            // Adjacent query blocks reuse the same source row. CUDA grid.y permits at most 65535 rows.
            const bool queryFirst = n > 1 && rows > 1 && rows <= 65535;
            const dim3 grid = queryFirst ? dim3(n, rows) : dim3(rows, n);
            kernel<<<grid, 512, 0, stream>>>(sources, group, rows, width, eps, normalizeValues, dst, queryFirst);
        } else {
            const auto kernel = width % 4 ? AttnResRows<false, Round, Out, In> : AttnResRows<true, Round, Out, In>;
            kernel<<<rows * n, 256, scratch, stream>>>(sources, count, group, n, rows, width, eps, normalizeValues,
                dst, staging ? staging + size_t(first) * rows * width : nullptr);
        }
    }
}
} // namespace cuda_attn_res
} // namespace fastllm
