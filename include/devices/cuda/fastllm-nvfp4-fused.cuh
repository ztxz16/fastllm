#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <stdint.h>
namespace fastllm {
namespace nvfused {
// Inverse of the existing Marlin 64-column scale permutation. No new weight
// allocation or repack is needed, so the original Linear fallback stays usable.
__device__ __forceinline__ int ScaleIndex(int n) {
    int t = ((n & 7) << 3) | ((n & 63) >> 3), lo = t & 3;
    return (n & ~63) + (t & ~3) + ((lo & 1) << 1) + (lo >> 1);
}
// Recover two E2M1 values and their normalized FP16 group scale. Marlin stores
// scales multiplied by 128; its tensor scale supplies the compensating factor.
// This produces the same representable half weights as the Marlin dequantizer.
// Older architectures/toolkits use the bit-exact Marlin half decoding below.
__device__ __forceinline__ float2 Decode(uint32_t q, half2 scale) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __nv_fp4x2_e2m1 code;
    code.__x = uint8_t((q & 15) | ((q >> 12) & 0xf0));
    uint32_t sb = *reinterpret_cast<uint32_t *>(&scale);
    uint32_t norm = sb ? sb - 0x38003800u : 0;
    return __half22float2(__hmul2(static_cast<half2>(code), *reinterpret_cast<half2 *>(&norm)));
#else
    // Marlin represents E2M1 values divided by 16384 as half subnormals.
    // Multiplication by the stored half scale recovers normalized half weights.
    uint32_t bits = ((q & 0x00080008u) << 12) | ((q & 0x00070007u) << 9);
    return __half22float2(__hmul2(*reinterpret_cast<half2 *>(&bits), scale));
#endif
}
template <class T> __device__ __forceinline__ float2 LoadPair(const T *p) {
    uint32_t b = __ldg(reinterpret_cast<const uint32_t *>(p));
    if constexpr (__is_same(T, half))
        return __half22float2(*reinterpret_cast<half2 *>(&b));
    else
        return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&b));
}
template <class T, int StaticN, int StaticK, bool Gate, int W, int Chains, int Unroll, int NW = 1, bool AddResidual = true>
__global__ __launch_bounds__(W * 32,
                             2) void GemvEpilogue(const T *__restrict__ x, const uint32_t *__restrict__ q,
                                                  const uint8_t *__restrict__ s,
                                                  const float *__restrict__ global, T *__restrict__ y,
                                                  int runtimeN = 0, int runtimeK = 0) {
    static_assert(W % NW == 0 && 8 % NW == 0 && StaticK % 16 == 0 && StaticN % 128 == 0);
    const int N = StaticN ? StaticN : runtimeN, K = StaticK ? StaticK : runtimeK;
    constexpr int Rows = Gate ? 4 : 2;
    const int Groups = K / 16;
    // Adjacent lanes read adjacent words of the Marlin tile. Four lanes select
    // N quadrants, the next two bits select K pairs, and the last bit selects
    // a second K group. NW warps cover adjacent N pairs; remaining warps split K.
    // One CTA owns every contribution to its output rows (including gate/up),
    // avoiding global partial sums, atomics, and a separate epilogue launch.
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int n = (blockIdx.x / (8 / NW)) * 64 + (blockIdx.x % (8 / NW)) * NW + (w % NW) + (lane & 3) * 16;
    int word = (n & 7) * 16 + ((lane >> 2) & 3) * 4 + (lane & 3);
    float acc[Rows][Chains] = {};
#pragma unroll Unroll
    for (int phase = 0; phase < (Groups + (W * 2 / NW) - 1) / (W * 2 / NW); ++phase) {
        int g = phase * (W * 2 / NW) + (w / NW) * 2 + (lane >> 4), k = g * 16 + ((lane >> 2) & 3) * 2;
        if (g < Groups) {
            float2 a = LoadPair(x + k), b = LoadPair(x + k + 8);
#pragma unroll
            for (int p = 0; p < (Gate ? 2 : 1); ++p) {
                int off = p * N / 2;
                uint32_t v = __ldcg(q + (size_t(g) * (N / 64) + (n + off) / 64) * 128 + word);
                int si = ScaleIndex(n) + off;
                uint32_t sb = __ldg(reinterpret_cast<const uint32_t *>(s + size_t(g) * N + (si & ~3))) >>
                              ((si & 3) * 8);
                half2 s0 = __half2half2(__ushort_as_half(uint16_t(sb & 255) << 7));
                half2 s1 = __half2half2(__ushort_as_half(uint16_t((sb >> 16) & 255) << 7));
                float2 a0 = Decode(v, s0), a1 = Decode(v >> 4, s0), b0 = Decode(v >> 8, s1),
                       b1 = Decode(v >> 12, s1);
                acc[p * 2][0] = fmaf(a0.x, a.x, acc[p * 2][0]);
                acc[p * 2][1 % Chains] = fmaf(a0.y, a.y, acc[p * 2][1 % Chains]);
                acc[p * 2][2 % Chains] = fmaf(a1.x, b.x, acc[p * 2][2 % Chains]);
                acc[p * 2][3 % Chains] = fmaf(a1.y, b.y, acc[p * 2][3 % Chains]);
                acc[p * 2 + 1][0] = fmaf(b0.x, a.x, acc[p * 2 + 1][0]);
                acc[p * 2 + 1][1 % Chains] = fmaf(b0.y, a.y, acc[p * 2 + 1][1 % Chains]);
                acc[p * 2 + 1][2 % Chains] = fmaf(b1.x, b.x, acc[p * 2 + 1][2 % Chains]);
                acc[p * 2 + 1][3 % Chains] = fmaf(b1.y, b.y, acc[p * 2 + 1][3 % Chains]);
            }
        }
    }
    // Reduce K within each N quadrant, then combine the K-split warps.
    __shared__ float sums[W][Rows][4];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float v = 0;
#pragma unroll
        for (int c = 0; c < Chains; ++c)
            v += acc[r][c];
#pragma unroll
        for (int d = 16; d >= 4; d /= 2)
            v += __shfl_down_sync(0xffffffff, v, d);
        if (lane < 4)
            sums[w][r][lane] = v;
    }
    __syncthreads();
    if (threadIdx.x < NW * 8) {
        int t = threadIdx.x, part = t % NW, quart = (t / NW) & 3, r = t / (NW * 4);
        float v = 0, u = 0;
#pragma unroll
        for (int j = 0; j < W / NW; ++j) {
            v += sums[j * NW + part][r][quart];
            if constexpr (Gate)
                u += sums[j * NW + part][r + 2][quart];
        }
        int dst = (blockIdx.x / (8 / NW)) * 64 + (blockIdx.x % (8 / NW)) * NW + part + quart * 16 + r * 8;
        // Preserve the original Linear output rounding before the fused epilogue.
        float scale = *global;
        T val = T(v * scale);
        if constexpr (Gate) {
            float g = float(val), up = float(T(u * scale)), e = __expf(-fabsf(g));
            y[dst] = T((g >= 0 ? g : g * e) * __fdividef(1, 1 + e) * up);
        } else if constexpr (AddResidual)
            y[dst] = T(float(val) + float(y[dst]));
        else
            y[dst] = val;
    }
}
} // namespace nvfused
} // namespace fastllm
