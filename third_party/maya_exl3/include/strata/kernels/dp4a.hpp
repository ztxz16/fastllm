// include/strata/kernels/dp4a.hpp - __dp4a for Pascal's sm_60 (Tesla P100, GP100), which has no such instruction.
//
// The decode's GEMVs (glm_fast.cu, native_mmvq.cu, s2_expert_grouped.cu, iq_kernels.cu's ggml_cuda_dp4a) and the EXL3
// kernels (exl3.cu, exl3_legacy.cu) call __dp4a, the byte-wise dot product of compute capability 6.1 and newer.  CUDA
// declares no __dp4a for sm_60, so a build for a P100 stopped in those six files ("identifier __dp4a is undefined").
// Below __CUDA_ARCH__ 610 this header defines __dp4a by name as an overload set that mirrors CUDA's two integer forms -
// signed bytes into int (four PTX vmad.s32.s32.s32 with byte selectors), unsigned bytes into unsigned (four
// vmad.u32.u32.u32) - so every call site resolves as it does against the real intrinsic and compiles unchanged.  The
// sum is the same int32 (four exact byte products, the same wrap-around), bit for bit: tools/sm60_dp4a_check.cu checks
// both forms against the byte-wise definition.  On a P100 four vmad are also the fastest int8 dot product we found
// (~2.1x the byte loop in isolation).
//
// Include it after cuda_runtime.h.  sm_61 and newer (a P40 keeps its hardware dp4a) and HIP take nothing from it:
// there __CUDA_ARCH__ is 610 or more, or undefined, and __dp4a is the toolkit's (hip_compat's on HIP).
#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int strata_dp4a_sm60(const int a, const int b, const int c) {
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ unsigned strata_dp4a_sm60(const unsigned a, const unsigned b, const unsigned c) {
    unsigned r = c;
    asm("vmad.u32.u32.u32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.u32.u32.u32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.u32.u32.u32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.u32.u32.u32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
}
#define __dp4a(a, b, c) strata_dp4a_sm60((a), (b), (c))
#endif
