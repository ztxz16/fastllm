#pragma once

// only take effect when compiling with HIP
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

#if defined(USE_ROCM) && !defined(HIP_NO_TENSOR_CORE) // support tensor core
#include <rocwmma/rocwmma.hpp>
#endif


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
    // CUDA code uses 32-lane logical warps; HIP requires a 64-bit mask.
    __device__ __forceinline__ unsigned long long WarpMask(unsigned int mask) {
        const unsigned int lane = (threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y * threadIdx.z)) % warpSize;
        return static_cast<unsigned long long>(mask) << (lane & ~31u);
    }

    __device__ __forceinline__ unsigned int BallotSync(unsigned int mask, int predicate) {
        const unsigned int lane = (threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y * threadIdx.z)) % warpSize;
        return static_cast<unsigned int>(__ballot_sync(WarpMask(mask), predicate) >> (lane & ~31u));
    }

    template <typename Kernel>
    inline hipError_t FuncSetAttribute(Kernel kernel, hipFuncAttribute attr, int value) {
        return ::hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), attr, value);
    }

    template <typename T>
    __device__ __forceinline__ T LoadReadOnly(const T *address) {
        return *address;
    }

    __device__ __forceinline__ void SyncWarp(unsigned int mask = 0xffffffffu) {
        __syncwarp(WarpMask(mask));
    }

    template <typename T>
    __device__ __forceinline__ T ShflXorSync(unsigned int mask, T value,
                                            int laneMask, int width = 32) {
        return __shfl_xor_sync(WarpMask(mask), value, laneMask, width);
    }

    template <typename T>
    __device__ __forceinline__ T ShflDownSync(unsigned int mask, T value,
                                             unsigned int delta, int width = 32) {
        return __shfl_down_sync(WarpMask(mask), value, delta, width);
    }

    template <typename T>
    __device__ __forceinline__ T ShflSync(unsigned int mask, T value,
                                         int lane, int width = 32) {
        return __shfl_sync(WarpMask(mask), value, lane, width);
    }

    template <typename T>
    __device__ __forceinline__ T ShflUpSync(unsigned int mask, T value,
                                           unsigned int delta, int width = 32) {
        return __shfl_up_sync(WarpMask(mask), value, delta, width);
    }

    inline hipblasStatus_t hipblasHgemmBatched(
        hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
        int m, int n, int k, const half *alpha, const half *const A[], int lda,
        const half *const B[], int ldb, const half *beta, half *const C[], int ldc,
        int batchCount) {
        return ::hipblasHgemmBatched(handle, transA, transB, m, n, k,
            reinterpret_cast<const hipblasHalf *>(alpha),
            reinterpret_cast<const hipblasHalf *const *>(A), lda,
            reinterpret_cast<const hipblasHalf *const *>(B), ldb,
            reinterpret_cast<const hipblasHalf *>(beta),
            reinterpret_cast<hipblasHalf *const *>(C), ldc, batchCount);
    }

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
            case HIP_R_16F: {
                // HIP's explicit 16F compute mode accumulates in FP16 and
                // loses accuracy for long dot products (e.g. INT4 dequant
                // GEMM with K=4096). Preserve FP16 storage, use FP32 sums.
                // CUDA callers use host alpha/beta; retain the original
                // behavior for a handle explicitly using device scalars.
                hipblasPointerMode_t pointerMode;
                hipblasStatus_t status = ::hipblasGetPointerMode(handle, &pointerMode);
                if (status != HIPBLAS_STATUS_SUCCESS) return status;
                if (pointerMode == HIPBLAS_POINTER_MODE_HOST) {
                    const float alpha32 = __half2float(*static_cast<const half*>(alpha));
                    const float beta32 = __half2float(*static_cast<const half*>(beta));
                    return ::hipblasGemmEx(handle, transA, transB, m, n, k,
                        &alpha32, A, aType, lda, B, bType, ldb,
                        &beta32, C, cType, ldc, HIPBLAS_COMPUTE_32F, algo);
                }
                computeType = HIPBLAS_COMPUTE_16F;
                break;
            }
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

using fastllm_hip::hipblasHgemmBatched;
using fastllm_hip::hipblasGemmEx;
using fastllm_hip::hipblasHgemmStridedBatched;
using fastllm_hip::hipblasHgemm;
#endif