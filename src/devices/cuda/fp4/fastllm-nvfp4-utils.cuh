/*
 * Adapted from vLLM
 * csrc/libtorch_stable/{cuda_vec_utils.cuh,quantization/fp4/nvfp4_utils.cuh}
 * under the Apache-2.0 license.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace fastllm_nvfp4 {

// Match vLLM's NVFP4 activation quantization policy: CUDA 12.9 added the
// 256-bit PTX load used by PACK16; CUDA 12.8 uses PACK8 and 128-bit loads.
#if defined(NVFP4_ENABLE_ELTS16) && defined(CUDART_VERSION) && \
    CUDART_VERSION >= 12090
constexpr bool kPack16 = true;
constexpr int kElementsPerThread = 16;
#else
constexpr bool kPack16 = false;
constexpr int kElementsPerThread = 8;
#endif

constexpr int kScaleVectorSize = 16;
constexpr int kThreadsPerScale = kScaleVectorSize / kElementsPerThread;

struct alignas(32) U32x8 { uint32_t d[8]; };
struct U32x2 { uint32_t lo, hi; };

using Fp4Packed = std::conditional_t<kPack16, U32x2, uint32_t>;

template <typename T> struct PackedType;
template <> struct PackedType<half> { using Type = half2; };
template <> struct PackedType<__nv_bfloat16> { using Type = __nv_bfloat162; };

template <typename T>
struct alignas(kPack16 ? 32 : 16) PackedVec {
    typename PackedType<T>::Type elts[kElementsPerThread / 2];
};

__device__ __forceinline__ void Load256OrZero(U32x8 &value, const void *ptr, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.u32 p, %8, 0;\n"
        " mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n"
        " mov.u32 %4, 0; mov.u32 %5, 0; mov.u32 %6, 0; mov.u32 %7, 0;\n"
        " @p ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
        "}\n"
        : "=r"(value.d[0]), "=r"(value.d[1]), "=r"(value.d[2]), "=r"(value.d[3]),
          "=r"(value.d[4]), "=r"(value.d[5]), "=r"(value.d[6]), "=r"(value.d[7])
        : "r"((int)pred), "l"(ptr));
#else
    assert(false && "NVFP4 256-bit load requires SM100+ and CUDA 12.9+");
#endif
}

__device__ __forceinline__ void Load128OrZero(uint4 &value, const void *ptr, bool pred) {
    uint32_t r0, r1, r2, r3;
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.u32 p, %4, 0;\n"
        " mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n"
        " @p ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
        "}\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"((int)pred), "l"(ptr));
    value = uint4{r0, r1, r2, r3};
}

template <typename T>
__device__ __forceinline__ void LoadOrZero(PackedVec<T> &value, const void *ptr, bool pred) {
    static_assert(sizeof(PackedVec<T>) == sizeof(T) * kElementsPerThread);
    if constexpr (kPack16) {
        Load256OrZero(reinterpret_cast<U32x8 &>(value), ptr, pred);
    } else {
        Load128OrZero(reinterpret_cast<uint4 &>(value), ptr, pred);
    }
}

template <typename T>
__device__ __forceinline__ float2 ToFloat2(T value) {
    if constexpr (std::is_same_v<T, half2>) {
        return __half22float2(value);
    } else {
        return __bfloat1622float2(value);
    }
}

template <typename T>
__device__ __forceinline__ T FromFloat2(float2 value) {
    if constexpr (std::is_same_v<T, half2>) {
        return __float22half2_rn(value);
    } else {
        return __float22bfloat162_rn(value);
    }
}

// Address one scale shared by 16 adjacent K elements.
__device__ __forceinline__ uint8_t *SwizzledScaleAddress(
        int row, int scaleGroup, int kTiles, uint8_t *scales) {
    const int mTile = row >> 7;
    const int outerM = row & 31;
    const int innerM = (row >> 5) & 3;
    const int kTile = scaleGroup >> 2;
    const int innerK = scaleGroup & 3;
    const int64_t offset = ((int64_t)mTile * kTiles + kTile) * 512
                         + outerM * 16 + innerM * 4 + innerK;
    return scales + offset;
}

// PACK8 uses two adjacent threads per 16-element scale. Only the first writes.
__device__ __forceinline__ uint8_t *QuantScaleAddress(
        int row, int pack, int kTiles, uint8_t *scales) {
    if (threadIdx.x % kThreadsPerScale != 0) return nullptr;
    return SwizzledScaleAddress(row, pack / kThreadsPerScale, kTiles, scales);
}

__device__ __forceinline__ uint32_t PackE2M1(float2 (&values)[4]) {
    uint32_t output;
    asm volatile(
        "{\n"
        ".reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0,b1,b2,b3};\n"
        "}\n"
        : "=r"(output)
        : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
          "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y));
    return output;
}

__device__ __forceinline__ U32x2 PackE2M1(float2 (&values)[8]) {
    U32x2 output;
    asm volatile(
        "{\n"
        ".reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n"
        ".reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
        "mov.b32 %0, {b0,b1,b2,b3}; mov.b32 %1, {b4,b5,b6,b7};\n"
        "}\n"
        : "=r"(output.lo), "=r"(output.hi)
        : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
          "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
          "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
          "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
    return output;
}

__device__ __forceinline__ float ReciprocalApproximateFtz(float value) {
    float output;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(output) : "f"(value));
    return output;
}

template <typename T>
__device__ __forceinline__ Fp4Packed Quantize(
        PackedVec<T> &input, float globalScale, uint8_t *scaleOut) {
    auto localMax = __habs2(input.elts[0]);
#pragma unroll
    for (int i = 1; i < kElementsPerThread / 2; ++i) {
        localMax = __hmax2(localMax, __habs2(input.elts[i]));
    }
    if constexpr (kThreadsPerScale == 2) {
        localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
    }
    const float maxValue = float(__hmax(localMax.x, localMax.y));

    float scaleValue = globalScale * (maxValue * ReciprocalApproximateFtz(6.0f));
    __nv_fp8_e4m3 encoded(scaleValue);
    const uint8_t encodedByte = reinterpret_cast<const uint8_t &>(encoded);
    if (scaleOut != nullptr) *scaleOut = encodedByte;
    scaleValue = (float)encoded;
    const float outputScale = scaleValue == 0.0f
        ? 0.0f
        : ReciprocalApproximateFtz(scaleValue * ReciprocalApproximateFtz(globalScale));

    float2 values[kElementsPerThread / 2];
#pragma unroll
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
        values[i] = ToFloat2(input.elts[i]);
        values[i].x *= outputScale;
        values[i].y *= outputScale;
    }
    return PackE2M1(values);
}

__device__ __forceinline__ void StoreFp4(uint8_t *address, U32x2 value) {
    const uint64_t packed = (uint64_t(value.hi) << 32) | uint64_t(value.lo);
    *reinterpret_cast<uint64_t *>(address) = packed;
}

__device__ __forceinline__ void StoreFp4(uint8_t *address, uint32_t value) {
    *reinterpret_cast<uint32_t *>(address) = value;
}

template <typename T>
__device__ __forceinline__ PackedVec<T> SiluMul(
        const PackedVec<T> &gate, const PackedVec<T> &up) {
    PackedVec<T> output;
#pragma unroll
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
        float2 g = ToFloat2(gate.elts[i]);
        float2 u = ToFloat2(up.elts[i]);
        g.x = __fdividef(g.x, 1.0f + __expf(-g.x)) * u.x;
        g.y = __fdividef(g.y, 1.0f + __expf(-g.y)) * u.y;
        output.elts[i] = FromFloat2<typename PackedType<T>::Type>(g);
    }
    return output;
}

inline size_t SwizzledScaleBytes(int rows, int columns) {
    const int roundedRows = (rows + 127) / 128 * 128;
    const int roundedScaleColumns = ((columns / 16) + 3) / 4 * 4;
    return (size_t)roundedRows * roundedScaleColumns;
}

} // namespace fastllm_nvfp4
