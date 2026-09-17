#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
namespace fastllm {
namespace fp8row {
// Row-scaled FP8 GEMV, with FP32 accumulation and one scale application
// per output. K is a multiple of 512; arbitrary N is masked at the tail.
// Ordinary SIMT operations, with software FP8 conversion on older SMs.
template <class T, int Warps = 8, int Rows = 1, int Chains = 2, int Values = 16>
__global__ __launch_bounds__(Warps * 32,
                             2) void Kernel(const T *__restrict__ input, const uint8_t *__restrict__ weight,
                                            const float *__restrict__ scales, const T *__restrict__ bias,
                                            T *__restrict__ output, int K, int N) {
    static_assert(__is_same(T, half) || __is_same(T, __nv_bfloat16));
    static_assert(Values == 8 || Values == 16);
    static_assert(Chains > 0 && (Chains & (Chains - 1)) == 0);
    static_assert(Warps > 0 && Warps <= 32 && Rows > 0);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * (Warps * Rows) + warp * Rows;
    float accum[Rows][Chains] = {};
#pragma unroll 2
    for (int phase = 0; phase < K / (32 * Values); ++phase) {
        int col = phase * (32 * Values) + lane * Values;
        uint32_t codes[Rows][Values / 4];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if constexpr (Values == 16) {
                uint4 x = {};
                if (row + r < N)
                    x = *reinterpret_cast<const uint4 *>(weight + (size_t)(row + r) * K + col);
                codes[r][0] = x.x;
                codes[r][1] = x.y;
                codes[r][2] = x.z;
                codes[r][3] = x.w;
            } else {
                uint2 x = {};
                if (row + r < N)
                    x = *reinterpret_cast<const uint2 *>(weight + (size_t)(row + r) * K + col);
                codes[r][0] = x.x;
                codes[r][1] = x.y;
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
            {
                // A naturally aligned pair is shared by every output row.
                uint32_t bits = __ldg(reinterpret_cast<const uint32_t *>(input + col + p * 2));
                float2 a;
                if constexpr (__is_same(T, half))
                    a = __half22float2(*reinterpret_cast<const half2 *>(&bits));
                else
                    a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits));
                float a0 = a.x, a1 = a.y;
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    accum[r][(p * 2) & (Chains - 1)] =
                        fmaf(decoded[r].x, a0, accum[r][(p * 2) & (Chains - 1)]);
                    accum[r][(p * 2 + 1) & (Chains - 1)] =
                        fmaf(decoded[r].y, a1, accum[r][(p * 2 + 1) & (Chains - 1)]);
                }
            }
        }
    }

    {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            float sum = 0;
#pragma unroll
            for (int c = 0; c < Chains; ++c)
                sum += accum[r][c];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0 && row + r < N) {
                int n = row + r;
                T projected = T(sum * scales[n] + (bias ? float(bias[n]) : 0.f));
                output[n] = projected;
            }
        }
    }
}
} // namespace fp8row
} // namespace fastllm
