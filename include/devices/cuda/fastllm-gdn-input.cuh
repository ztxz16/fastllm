#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Row-scaled W8A16 GDN projection. Weight tiles are reused across independent
// request rows; each request owns a distinct convolution-state slot.
// BatchTile is exact: the launcher must never provide a partial batch tile.
namespace fastllm {
namespace gdn {
template <class T> __device__ __forceinline__ float Load(T x) { return float(x); }
template <class T> __device__ __forceinline__ T Round(float x) { return T(x); }

template <class T, int BatchTile, int Warps = 8, int Rows = 2, int Values = 8, int Chains = 4, bool Dynamic = false>
__global__ __launch_bounds__(Warps * 32, 2) void InputConvKernel(
    const T *__restrict__ input, const uint8_t *__restrict__ weight, const float *__restrict__ scales,
    const float *__restrict__ projectionBias, const float *__restrict__ convWeight,
    const float *__restrict__ convBias, T *__restrict__ cache, const int *__restrict__ slots,
    T *__restrict__ convOutput, T *__restrict__ z, int /*batch*/, int inputWidth = 5120, int channels = 10240, int zWidth = 6144) {
    static_assert(Chains > 0 && (Chains & (Chains - 1)) == 0);
    static_assert(Values == 8 || Values == 16);
    static_assert(BatchTile >= 1 && BatchTile <= 8);
    const int K = Dynamic ? inputWidth : 5120;
    const int C = Dynamic ? channels : 10240, Z = Dynamic ? zWidth : 6144;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * (Warps * Rows) + warp * Rows;
    int firstBatch = blockIdx.y * BatchTile;
    float accum[BatchTile][Rows][Chains] = {};
#pragma unroll 2
    for (int phase = 0; phase < K / (32 * Values); ++phase) {
        int col = phase * (32 * Values) + lane * Values;
        uint32_t codes[Rows][Values / 4];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if constexpr (Values == 8) {
                uint2 x = {};
                if (!Dynamic || row + r < C + Z)
                    x = *reinterpret_cast<const uint2 *>(weight + (size_t)(row + r) * K + col);
                codes[r][0] = x.x;
                codes[r][1] = x.y;
            } else {
                uint4 x = {};
                if (!Dynamic || row + r < C + Z)
                    x = *reinterpret_cast<const uint4 *>(weight + (size_t)(row + r) * K + col);
                codes[r][0] = x.x;
                codes[r][1] = x.y;
                codes[r][2] = x.z;
                codes[r][3] = x.w;
            }
        }
#pragma unroll
        for (int p = 0; p < Values / 2; ++p) {
            float2 decoded[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                // CUDA supplies exact software conversion below SM89 and a
                // native FP8-to-half conversion on SM89+. No tensor cores.
                __nv_fp8x2_e4m3 x;
                x.__x = uint16_t(codes[r][p / 2] >> ((p & 1) * 16));
                decoded[r] = static_cast<float2>(x);
            }
#pragma unroll
            for (int b = 0; b < BatchTile; ++b) {
                // A naturally aligned pair is shared by every output row.
                uint32_t bits = __ldg(
                    reinterpret_cast<const uint32_t *>(input + (size_t)(firstBatch + b) * K + col + p * 2));
                float2 a;
                if constexpr (__is_same(T, half))
                    a = __half22float2(*reinterpret_cast<const half2 *>(&bits));
                else
                    a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits));
                float a0 = a.x, a1 = a.y;
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    accum[b][r][(p * 2) & (Chains - 1)] =
                        fmaf(decoded[r].x, a0, accum[b][r][(p * 2) & (Chains - 1)]);
                    accum[b][r][(p * 2 + 1) & (Chains - 1)] =
                        fmaf(decoded[r].y, a1, accum[b][r][(p * 2 + 1) & (Chains - 1)]);
                }
            }
        }
    }
    // Transpose warp-owned totals into contiguous epilogue work. In FastLLM
    // the four history columns are adjacent.
    // A warp-coalesced epilogue avoids serial sparse in-place cache transactions.
    __shared__ float totals[BatchTile][Warps * Rows];
#pragma unroll
    for (int b = 0; b < BatchTile; ++b) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            float sum = 0;
#pragma unroll
            for (int c = 0; c < Chains; ++c)
                sum += accum[b][r][c];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0)
                totals[b][warp * Rows + r] = sum;
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < BatchTile * Warps * Rows; index += Warps * 32) {
        int b = index / (Warps * Rows), local = index % (Warps * Rows);
        int n = blockIdx.x * (Warps * Rows) + local, request = firstBatch + b;
        if (Dynamic && n >= C + Z) continue;
        float sum = totals[b][local];
        T projected = Round<T>(sum * scales[n] + (projectionBias ? projectionBias[n] : 0.f));
        if (n < C) {
            int slot = slots ? slots[request] : request;
            T *history = cache + ((size_t)slot * C + n) * 4;
            uint2 old = *reinterpret_cast<const uint2 *>(history);
            T *hv = reinterpret_cast<T *>(&old);
            T x0 = hv[1], x1 = hv[2], x2 = hv[3];
            float4 coeff = *reinterpret_cast<const float4 *>(convWeight + n * 4);
            float value = convBias ? convBias[n] : 0.f;
            value = fmaf(Load(x0), coeff.x, value);
            value = fmaf(Load(x1), coeff.y, value);
            value = fmaf(Load(x2), coeff.z, value);
            value = fmaf(Load(projected), coeff.w, value);
            // Preserve the projection and convolution storage boundaries.
            float rounded = Load(Round<T>(value));
            float e = __expf(-fabsf(rounded));
            float silu = (rounded >= 0.f ? rounded : rounded * e) * __fdividef(1.f, 1.f + e);
            convOutput[(size_t)request * C + n] = Round<T>(silu);
            uint2 updated;
            T *next = reinterpret_cast<T *>(&updated);
            next[0] = x0;
            next[1] = x1;
            next[2] = x2;
            next[3] = projected;
            *reinterpret_cast<uint2 *>(history) = updated;
        } else {
            z[(size_t)request * Z + n - C] = projected;
        }
    }
}
} // namespace gdn
} // namespace fastllm
