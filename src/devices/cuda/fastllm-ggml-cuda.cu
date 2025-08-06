//
// Created by huangyuyang on 8/6/25.
//
// This is the code compatible with GGUF in Fastllm
// Most code copy from 
// https://github.com/ggml-org/llama.cpp
// https://github.com/ikawrakow/ik_llama.cpp

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);

extern void *FastllmCudaPrepareInput(const fastllm::Data &input);
extern void *FastllmCudaPrepareOutput(fastllm::Data &output);
extern void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
extern void FastllmCudaFinishOutput(fastllm::Data &output, void *data);
extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

template <typename T>
static __global__ void quantize_q8_1(const T * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix0 < kx ? (float)x[ix1*kx + ix0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <typename T>
void quantize_row_q8_1_cuda(
    const T * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    assert(kx0_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1*channels, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx0_padded);
}

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(RDNA2)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(__gfx1010__) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= MIN_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
#define WARP_SIZE 32

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {        
        case GGML_TYPE_Q4_K   : return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q6_K   : return vec_dot_q6_K_q8_1;
        default               : return nullptr;
    }
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K    : return VDR_Q4_K_Q8_1_MMVQ;
        default                : return 1;
    }
}

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};

template <ggml_type type, int ncols_y, int nwarps, typename OType>
static __device__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, OType * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    //int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
    //    ncols_y < 4 ? 1 : 2 : 1;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = ncols_y < 4 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j*nrows_dst + row0 + threadIdx.x] = (OType)tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps, typename OType>
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, OType * __restrict__ dst, const char * __restrict__ ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0) {
    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        // We clear the buffer via cudaMemset instead
//#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
//        constexpr int rows_per_cuda_block = 1;
//#else
//        constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
//#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)
//        const int row0 = rows_per_cuda_block*blockIdx.x;
//        if (threadIdx.y == 0) {
//            dst = (float *)cdst;
//            for (int j = 0; j < ncols_y; ++j) {
//                if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
//                    dst[j*nrows_dst + row0 + threadIdx.x] = 0;
//                }
//            }
//        }
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    mul_mat_vec_q<type, ncols_y, nwarps, OType>(cx, cy, (OType *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <ggml_type type, int nwarps, typename OType>
static void mul_mat_vec_q_cuda_T(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {

    assert(ncols_x % ggml_blck_size(type) == 0);
    assert(ncols_y <= MMVQ_MAX_BATCH_SIZE);    

    int64_t rows_per_cuda_block = 1;
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps, OType><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        default:
            printf("fatal error\n");
            exit(0);
            break;
    }
}

template <ggml_type type, typename OType>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {
    int nwarps = 1;
    switch (nwarps) {
        case 1:
            mul_mat_vec_q_cuda_T<type, 1, OType>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case 2:
            mul_mat_vec_q_cuda_T<type, 2, OType>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            mul_mat_vec_q_cuda_T<type, 4, OType>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
    }
}

template <typename OType>
static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_K, OType>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

template <typename OType>
static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_K, OType>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

struct ggml_backend_cuda_context {

};

template <typename OType>
static void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0,
        const char * src0_dd_i, const char * src1_ddq_i, OType * dst_dd_i, const char * ids_data,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    /*
    int id = ggml_cuda_get_device();
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;
    */
    const int64_t nrows_dst = true ? ne0 : row_diff;

    switch (type) {
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda<OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda<OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            printf("Error: unsupport cuda linear type %s\n", ggml_type_name(type));
            exit(0);
            break;
    }
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
/*
    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    assert(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, 1, 0, 0, 0, 0,
        src0_dd_i, src1_ddq_i, dst_dd_i, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
*/
}

template <typename T>
void LaunchFastllmGemvGGUFKernel(float *input, uint8_t *weight, float *output, float *bias, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        // FastllmGemvGGUFKernel <256, 1, T> <<< k, 256 >>> (input + i * m, weight, output + i * k, bias, m, k);
    }
}

bool FastllmCudaMatMulFloatGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        auto state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaBiasData = (float*)weight.extraCudaData[0];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    // block_q8_1 * q8Input = (block_q8_1*)FastllmCudaMalloc(n * m / QK8_1 * sizeof(block_q8_1));
    block_q8_1 * q8Input = (block_q8_1*)FastllmCudaMalloc(n * m * sizeof(half));
    quantize_row_q8_1_cuda (
        cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, nullptr
    );

    ggml_backend_cuda_context ctx;
    if (n > 1) {
        for (int i = 0; i < n; i++) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1, 
                    0, 0, 0, 0,
                    (char*)weight.cudaData, 
                    (char*)(q8Input + i * (m / QK8_1)), 
                    cudaOutput + i * k, 
                    nullptr, 
                    0, k, 1, m, nullptr 
            );        
        }
    } else {
        ggml_cuda_op_mul_mat_vec_q_impl (
                ctx, (ggml_type)weight.ggmlType, m, k, 1, 
                0, 0, 0, 0,
                (char*)weight.cudaData, (char*)q8Input, cudaOutput, nullptr, 
                0, k, n, m, nullptr 
        );
    }
    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
    }

    FastllmCudaFree(q8Input);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || 
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }

    // float *cudaBiasData = (float*)weight.extraCudaData[0];
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    block_q8_1 * q8Input = (block_q8_1*)FastllmCudaMalloc(n * m * sizeof(half));
    quantize_row_q8_1_cuda (
        cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, nullptr
    );

    ggml_backend_cuda_context ctx;
    if (n > 1) {
        for (int i = 0; i < n; i++) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1, 
                    0, 0, 0, 0,
                    (char*)weight.cudaData, 
                    (char*)(q8Input + i * (m / QK8_1)), 
                    cudaOutput + i * k, 
                    nullptr, 
                    0, k, 1, m, nullptr 
            );        
        }
    } else {
        ggml_cuda_op_mul_mat_vec_q_impl (
                ctx, (ggml_type)weight.ggmlType, m, k, 1, 
                0, 0, 0, 0,
                (char*)weight.cudaData, (char*)q8Input, cudaOutput, nullptr, 
                0, k, n, m, nullptr 
        );
    }
    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
    }

    FastllmCudaFree(q8Input);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);

    return true;   
}