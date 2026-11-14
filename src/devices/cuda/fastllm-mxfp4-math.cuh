#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Shared BF16 x E2M1/UE8M0 arithmetic. Keep the pair accumulation and
// 16-lane reduction identical in linear projections and expert kernels.
namespace fastllm { namespace cuda { namespace mxfp4 {
__device__ inline float FP4(unsigned value, unsigned exponent) {
    // Normal E2M1 magnitudes 2..7 map to consecutive FP32 bit
    // patterns spaced by 1 << 22. Handle zero and the .5 subnormal
    // explicitly, preserving signed zero and the UE8M0 edge scales.
    const unsigned magnitude = value & 7;
    const unsigned bits = magnitude >= 2 ? (magnitude + 252) << 22 : magnitude * 0x3f000000u;
    const float decoded = __uint_as_float(bits | ((value & 8) << 28));
    const float scale = exponent == 0     ? __int_as_float(0x00400000)
                        : exponent == 255 ? __int_as_float(0x7fc00000)
                                          : __int_as_float(exponent << 23);
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(decoded, scale)));
}
__device__ inline float BFloat(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}
// Keep the scale edge cases and signed zero exact without divergent branches.
__device__ __forceinline__ unsigned SelectBits(unsigned a, unsigned b, bool condition) {
    unsigned result;
    asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.u32 %0,%1,%2,p; }"
        : "=r"(result) : "r"(a), "r"(b), "r"((unsigned)condition));
    return result;
}
__device__ __forceinline__ float2 FP4Pair(unsigned packed, unsigned exponent) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300
    // As in FlashInfer sparse_mla_sm120, unpack and scale a pair in BF16.
    // E2M1 values are exact; power-of-two scaling preserves the old FP32
    // product followed by BF16 rounding, including UE8M0 edge encodings.
    unsigned halves, scaleBits;
    asm("{ .reg .b8 lo, hi; mov.b16 {lo,hi}, %1; cvt.rn.f16x2.e2m1x2 %0, lo; }"
        : "=r"(halves) : "h"((unsigned short)packed));
    asm("cvt.rn.bf16x2.ue8m0x2 %0,%1;" : "=r"(scaleBits)
        : "h"((unsigned short)(exponent * 0x101u)));
    const __nv_bfloat162 values = __float22bfloat162_rn(
        __half22float2(*reinterpret_cast<const __half2 *>(&halves)));
    const __nv_bfloat162 scale = *reinterpret_cast<const __nv_bfloat162 *>(&scaleBits);
    return __bfloat1622float2(__hmul2(values, scale));
#else
    unsigned bits = SelectBits(exponent << 23, 0x00400000u, exponent != 0);
    bits = SelectBits(bits, 0x7fc00000u, exponent != 255);
    const float scale = __uint_as_float(bits);
    const unsigned lut = __byte_perm(0x3e3c3800u, 0x46444240u, packed & 0x77u);
    unsigned halves = __byte_perm(lut, 0u, 0x1404u);
    halves |= ((packed & 8u) << 12) | ((packed & 128u) << 24);
    const float2 values = __half22float2(*reinterpret_cast<const __half2 *>(&halves));
    return make_float2(BFloat(__fmul_rn(values.x,scale)), BFloat(__fmul_rn(values.y,scale)));
#endif
}
template<bool Planar = false>
__device__ inline float Dot(const __nv_bfloat16 *input, const uint8_t *weight, int columns,
                            const uint8_t *scales = nullptr) {
    const int lane = threadIdx.x & 15;
    float sum = 0;
    // Independent loads/decodes overlap; additions retain the original order.
    #pragma unroll 8
    for (int block = 0; block < columns / 32; ++block) {
        const uint8_t *w = weight + block * (Planar ? 16 : 17);
        const float2 value = FP4Pair(w[lane], Planar ? scales[block] : w[16]);
        // Each adjacent BF16 input pair is aligned, including every block.
        const unsigned x = reinterpret_cast<const unsigned *>(input)[block * 16 + lane];
        const float x0 = __uint_as_float(x << 16), x1 = __uint_as_float(x & 0xffff0000u);
        const float pair = __fadd_rn(__fmul_rn(x0, value.x), __fmul_rn(x1, value.y));
        sum = __fadd_rn(sum, pair);
    }
    for (int delta = 8; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta, 16));
    return sum;
}
} } }
