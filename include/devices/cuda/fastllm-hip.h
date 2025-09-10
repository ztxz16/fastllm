#pragma once

// only take effect when compiling with HIP
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

#if defined(USE_ROCM) && !defined(HIP_NO_TENSOR_CORE) // support tensor core
#include <rocwmma/rocwmma.hpp>
#endif

#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

namespace fastllm_hip {

    inline const hipblasHalf* ToHipblasHalfConst(const half* x) {
        return reinterpret_cast<const hipblasHalf*>(x);
        }
    
    inline hipblasHalf* ToHipblasHalf(half* x) {
        return reinterpret_cast<hipblasHalf*>(x);
        }

    inline hipblasStatus_t hipblasGemmEx(hipblasHandle_t      handle,
        hipblasOperation_t   transA,
        hipblasOperation_t   transB,
        int                  m,
        int                  n,
        int                  k,
        const void* alpha,
        const void* A,
        hipDataType          aType,
        int                  lda,
        const void* B,
        hipDataType          bType,
        int                  ldb,
        const void* beta,
        void* C,
        hipDataType          cType,
        int                  ldc,
        hipDataType computeType_,
        hipblasGemmAlgo_t    algo) {
        hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;
        switch (computeType_) {
            case HIP_R_16F:
                computeType = HIPBLAS_COMPUTE_16F;
                break;
            case HIP_R_32F:
                computeType = HIPBLAS_COMPUTE_32F;
                break;
            default:
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            
            return ::hipblasGemmEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, B, bType, ldb, beta, C, cType, ldc, computeType, algo);
        }

    inline hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t handle,
        hipblasOperation_t transA,
        hipblasOperation_t transB,
        int m,
        int n,
        int k,
        const half* alpha,
        const half* AP,
        int lda,
        long long strideA,
        const half* BP,
        int ldb,
        long long strideB,
        const half* beta,
        half* CP,
        int ldc,
        long long strideC,
        int batchCount) {
        return ::hipblasHgemmStridedBatched
        (handle, transA, transB, m, n, k, ToHipblasHalfConst(alpha), ToHipblasHalfConst(AP), lda, strideA, ToHipblasHalfConst(BP), ldb, strideB, ToHipblasHalfConst(beta), ToHipblasHalf(CP), ldc, strideC, batchCount);
        }

    inline hipblasStatus_t hipblasHgemm(hipblasHandle_t handle, 
        hipblasOperation_t transA, 
        hipblasOperation_t transB, 
        int m, 
        int n, 
        int k, 
        const half *alpha, 
        const half *AP, 
        int lda, 
        const half *BP, 
        int ldb, 
        const half *beta, 
        half *CP, 
        int ldc){
        return
        ::hipblasHgemm(handle, transA, transB, m, n, k, ToHipblasHalfConst(alpha), ToHipblasHalfConst(AP), lda, ToHipblasHalfConst(BP), ldb, ToHipblasHalfConst(beta), ToHipblasHalf(CP), ldc);
        }
} // namespace fastllm_hip

using fastllm_hip::hipblasGemmEx;
using fastllm_hip::hipblasHgemmStridedBatched;
using fastllm_hip::hipblasHgemm;
#endif