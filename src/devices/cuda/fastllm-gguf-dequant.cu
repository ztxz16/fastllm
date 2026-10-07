// GGUF matrix dequantization and typed dispatch. Kernel arithmetic is shared
// with direct GEMV through fastllm-gguf-gemv.cuh.
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_bf16.h>
#include <cassert>
#include <set>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
#include "fastllm-gguf-dequant.cuh"
#include "fastllm-gguf-gemv.cuh"

// Device function
__device__ inline void get_scale_min_k4_device(int j, const uint8_t * __restrict__ q,
                                               uint8_t * __restrict__ d,
                                               uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// 直接处理float数组
__global__ void dequantize_q4_K_cuda_simple(const block_q4_K * __restrict__ x,
                                            half * __restrict__ y,
                                            int64_t k) {
    const int nb = k / QK_K;

    // 每个block处理一个q4_K块
    const int block_id = blockIdx.x;
    if (block_id >= nb) return;

    const block_q4_K * xi = &x[block_id];
    half * yi = y + block_id * QK_K;

    const float d   = __half2float(xi->data.d);
    const float min = __half2float(xi->data.dmin);

    // 每个线程处理一个或多个元素
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int idx = tid; idx < QK_K; idx += stride) {
        // 确定这个元素属于哪个32元素的子块
        const int sub_block = idx / 32;
        // const int elem_in_block = idx % 32;

        // 获取scale和min
        uint8_t sc, m;
        get_scale_min_k4_device(sub_block, xi->scales, &sc, &m);

        const float d_scaled = d * sc;
        const float min_scaled = min * m;

        // 确定quantized值的位置
        // 每64个元素共享32个字节的qs
        const int group_of_64 = idx / 64;
        const int elem_in_64 = idx % 64;
        const int qs_idx = group_of_64 * 32 + elem_in_64 % 32;

        if (qs_idx < QK_K/2) {  // 边界检查
            uint8_t q_val = xi->qs[qs_idx];

            float result;
            if (elem_in_64 < 32) {
                // 前32个元素使用低4位
                result = d_scaled * (q_val & 0xF) - min_scaled;
            } else {
                // 后32个元素使用高4位
                result = d_scaled * (q_val >> 4) - min_scaled;
            }

            yi[idx] = __float2half(result);
        }
    }
}

// 封装函数
void dequantize_row_q4_K_cuda(const block_q4_K* d_x, half* d_y, int64_t k,
                              cudaStream_t stream = 0) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    // 使用简单版本或共享内存版本
    const int threads_per_block = 128;
    const int blocks = nb;  // 每个block处理一个q4_K块

    // 选择一个kernel
    dequantize_q4_K_cuda_simple<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, k);
    // dequantize_q4_K_cuda_shared<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, k);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Device function for q6_K - not needed since scales are directly accessible
// But keeping for consistency if needed for other operations
// CUDA kernel for dequantizing q6_K blocks
__global__ void dequantize_q6_K_cuda_simple(const block_q6_K * __restrict__ x,
                                            half * __restrict__ y,
                                            int64_t k) {
    const int nb = k / QK_K;

    // Each block processes one q6_K block
    const int block_id = blockIdx.x;
    if (block_id >= nb) return;

    const block_q6_K * xi = &x[block_id];
    half * yi = y + block_id * QK_K;

    const float d = __half2float(xi->d);

    // Each thread processes one or more elements
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Process QK_K elements (256 elements)
    for (int idx = tid; idx < QK_K; idx += stride) {
        // QK_K is processed in 2 groups of 128
        const int group_128 = idx / 128;  // Which 128-element group (0 or 1)
        const int idx_in_128 = idx % 128; // Position within the 128-element group

        // Within each 128-element group:
        // - 64 ql values (each storing 2 4-bit values)
        // - 32 qh values (each storing 4 2-bit values)
        // - 8 scale values (each used for 16 elements)

        // Calculate offsets for this group
        const int ql_offset = group_128 * 64;
        const int qh_offset = group_128 * 32;
        const int sc_offset = group_128 * 8;

        // Determine position within the 128-element group
        int l, pos_in_32;
        if (idx_in_128 < 32) {
            l = idx_in_128;
            pos_in_32 = 0;
        } else if (idx_in_128 < 64) {
            l = idx_in_128 - 32;
            pos_in_32 = 1;
        } else if (idx_in_128 < 96) {
            l = idx_in_128 - 64;
            pos_in_32 = 2;
        } else {
            l = idx_in_128 - 96;
            pos_in_32 = 3;
        }

        // Get the scale index (each scale covers 16 elements)
        const int is = l / 16;

        // Get ql and qh values
        uint8_t ql_val, qh_val;
        int8_t q_result;

        if (pos_in_32 == 0) {
            // First 32 elements: use lower 4 bits of ql[l] and bits 0-1 of qh[l]
            ql_val = xi->ql[ql_offset + l];
            qh_val = xi->qh[qh_offset + l];
            q_result = (int8_t)((ql_val & 0xF) | (((qh_val >> 0) & 3) << 4)) - 32;
            yi[idx] = __float2half(d * xi->scales[sc_offset + is + 0] * q_result);
        } else if (pos_in_32 == 1) {
            // Second 32 elements: use lower 4 bits of ql[l+32] and bits 2-3 of qh[l]
            ql_val = xi->ql[ql_offset + l + 32];
            qh_val = xi->qh[qh_offset + l];
            q_result = (int8_t)((ql_val & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32;
            yi[idx] = __float2half(d * xi->scales[sc_offset + is + 2] * q_result);
        } else if (pos_in_32 == 2) {
            // Third 32 elements: use upper 4 bits of ql[l] and bits 4-5 of qh[l]
            ql_val = xi->ql[ql_offset + l];
            qh_val = xi->qh[qh_offset + l];
            q_result = (int8_t)((ql_val >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
            yi[idx] = __float2half(d * xi->scales[sc_offset + is + 4] * q_result);
        } else { // pos_in_32 == 3
            // Fourth 32 elements: use upper 4 bits of ql[l+32] and bits 6-7 of qh[l]
            ql_val = xi->ql[ql_offset + l + 32];
            qh_val = xi->qh[qh_offset + l];
            q_result = (int8_t)((ql_val >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;
            yi[idx] = __float2half(d * xi->scales[sc_offset + is + 6] * q_result);
        }
    }
}

// Wrapper function for q6_K dequantization
void dequantize_row_q6_K_cuda(const block_q6_K* d_x, half* d_y, int64_t k,
                              cudaStream_t stream = 0) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    // Use 128 threads per block for good occupancy
    const int threads_per_block = 128;
    const int blocks = nb;  // Each block processes one q6_K block

    // Launch the kernel
    dequantize_q6_K_cuda_simple<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, k);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// 简单版本的kernel - 直接处理float数组
__global__ void dequantize_q8_0_cuda_simple(const block_q8_0 * __restrict__ x,
                                            half * __restrict__ y,
                                            int64_t k) {
    const int nb = k / QK8_0;

    // 每个block处理一个q8_0块
    const int block_id = blockIdx.x;
    if (block_id >= nb) return;

    const block_q8_0 * xi = &x[block_id];
    half * yi = y + block_id * QK8_0;

    // 获取scale因子
    const float d = __half2float(xi->d);

    // 每个线程处理一个或多个元素
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int idx = tid; idx < QK8_0; idx += stride) {
        // q8_0的dequantize很简单：y = q * d
        yi[idx] = __float2half(xi->qs[idx] * d);
    }
}

// 封装函数
void dequantize_row_q8_0_cuda(const block_q8_0* d_x, half* d_y, int64_t k,
                              cudaStream_t stream = 0) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    // 选择合适的kernel配置
    // 方案1: 简单版本 - 每个CUDA block处理一个q8_0 block
    const int threads_per_block = 32;  // QK8_0 = 32，使用32个线程正好
    const int blocks = nb;
    dequantize_q8_0_cuda_simple<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, k);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define CUDA_Q8_0_NE_ALIGN 2048

typedef half dfloat; // dequantize float
typedef half2 dfloat2;

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v = __hmul2(v, {d, d});
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx,
                                              dst_t * __restrict__ yy,
                                              int64_t blockCount) {
    dequantize_block_q4_0_impl<dst_t>(vx, yy, blockCount, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx,
                                              dst_t * __restrict__ yy,
                                              int64_t blockCount) {
    dequantize_block_q4_1_impl<dst_t>(vx, yy, blockCount, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx,
                                                 dst_t * __restrict__ yy) {
    dequantize_block_iq2_xxs_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx,
                                                dst_t * __restrict__ yy) {
    dequantize_block_iq2_xs_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx,
                                               dst_t * __restrict__ yy) {
    dequantize_block_iq2_s_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx,
                                                 dst_t * __restrict__ yy) {
    dequantize_block_iq3_xxs_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx,
                                               dst_t * __restrict__ yy) {
    dequantize_block_iq1_s_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx,
                                               dst_t * __restrict__ yy) {
    dequantize_block_iq1_m_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q2_0(const void *__restrict__ vx,
                                            dst_t *__restrict__ output,
                                            int64_t blocks) {
    dequantize_block_q2_0_impl<dst_t>(vx, output, blocks, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static void dequantize_row_q2_0_cuda(const void *vx, dst_t *y,
                                    int64_t nrows, int64_t n_per_row,
                                    cudaStream_t stream) {
    const int64_t blocks = nrows * n_per_row / QK2_0;
    if (blocks) dequantize_block_q2_0<<<(blocks + 3) / 4, 32, 0, stream>>>(vx, y, blocks);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void *vx, dst_t *y,
                                     int64_t nrows, int64_t n_per_row,
                                     cudaStream_t stream) {
    const int64_t elements = nrows * n_per_row;
    const int64_t blockCount = elements / QK4_0;
    dequantize_block_q4_0<<<(blockCount + 7) / 8, 32, 0, stream>>>(
        vx, y, blockCount);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void *vx, dst_t *y,
                                     int64_t nrows, int64_t n_per_row,
                                     cudaStream_t stream) {
    const int64_t elements = nrows * n_per_row;
    const int64_t blockCount = elements / QK4_1;
    dequantize_block_q4_1<<<(blockCount + 7) / 8, 32, 0, stream>>>(
        vx, y, blockCount);
}

#define FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(name)                              \
    template<typename dst_t>                                                 \
    static void dequantize_row_##name##_cuda(                                \
            const void *vx, dst_t *y, int64_t nrows, int64_t n_per_row,      \
            cudaStream_t stream) {                                            \
        const int64_t blocks = nrows * n_per_row / QK_K;                     \
        dequantize_block_##name<<<blocks, 32, 0, stream>>>(vx, y);            \
    }

FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq2_xxs)
FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq2_xs)
FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq2_s)
FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq3_xxs)
FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq1_s)
FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER(iq1_m)

#undef FASTLLM_DEFINE_IQ_DEQUANT_WRAPPER

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx,
                                                dst_t * __restrict__ y,
                                                const int64_t k) {
    dequantize_block_iq4_nl_impl<dst_t>(vx, y, k, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void *vx, dst_t *y,
                                       const int64_t nrows,
                                       const int64_t n_per_row,
                                       cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int blocks = (int)((k + QK_K - 1) / QK_K);
    dequantize_block_iq4_nl<<<blocks, 32, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx,
                                                dst_t * __restrict__ y,
                                                const int64_t k) {
    dequantize_block_iq4_xs_impl<dst_t>(vx, y, k, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void *vx, dst_t *y,
                                       const int64_t nrows,
                                       const int64_t n_per_row,
                                       cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int blocks = (int)((k + QK_K - 1) / QK_K);
    dequantize_block_iq4_xs<<<blocks, 32, 0, stream>>>(vx, y, k);
}

static __global__ void dequantize_block_q8_0_bf16(const void * __restrict__ vx,
                                                  __nv_bfloat16 * __restrict__ y,
                                                  const int64_t k) {
    const int64_t i = (int64_t)2 * (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const block_q8_0 * x = (const block_q8_0 *) vx;
    const int64_t ib = i / QK8_0;
    const int64_t iqs = i % QK8_0;
    const float d = __half2float(x[ib].d);

    y[i + 0] = DequantizeCast<__nv_bfloat16>::cast(d * x[ib].qs[iqs + 0]);
    if (i + 1 < k) {
        y[i + 1] = DequantizeCast<__nv_bfloat16>::cast(d * x[ib].qs[iqs + 1]);
    }
}

static void dequantize_block_q8_0_bf16_cuda(const void * __restrict__ vx,
                                            __nv_bfloat16 * __restrict__ y,
                                            const int64_t nrows,
                                            const int64_t n_per_row,
                                            cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int num_blocks = (k + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block_q8_0_bf16<<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static __global__ void dequantize_block_q8_0_f32(const void * __restrict__ vx,
                                                 float * __restrict__ y,
                                                 const int64_t k) {
    const int64_t i = (int64_t)2 * (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const block_q8_0 * x = (const block_q8_0 *) vx;
    const int64_t ib = i / QK8_0;
    const int64_t iqs = i % QK8_0;
    const float d = __half2float(x[ib].d);

    y[i + 0] = d * x[ib].qs[iqs + 0];
    if (i + 1 < k) {
        y[i + 1] = d * x[ib].qs[iqs + 1];
    }
}

static void dequantize_block_q8_0_f32_cuda(const void * __restrict__ vx,
                                           float * __restrict__ y,
                                           const int64_t nrows,
                                           const int64_t n_per_row,
                                           cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int num_blocks = (k + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block_q8_0_f32<<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}


template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_q2_K_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x, blockDim.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx,
                                              dst_t * __restrict__ yy) {
    dequantize_block_q3_K_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_q4_K_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_q5_K_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_q6_K_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx,
                                               dst_t * __restrict__ yy) {
    dequantize_block_iq3_s_impl<dst_t>(vx, yy, blockIdx.x, threadIdx.x);
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 128, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void *vx, dst_t *y,
                                      const int64_t nrows,
                                      const int64_t n_per_row,
                                      cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int blocks = k / QK_K;
    dequantize_block_q3_K<<<blocks, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

// Dequantize block_q4_k_r4: each block packs 4 rows of QK_K(=256) elements.
// 128 threads per CUDA block (32 threads per row x 4 rows), each thread decodes 8 elements.
template<typename dst_t>
static __global__ void dequantize_block_q4_K_r4(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                 const int64_t n_per_row) {
    const block_q4_k_r4 * x = (const block_q4_k_r4 *) vx;
    const int64_t i = blockIdx.x; // block index

    // tid = 0..127; k = tid/32 (row 0..3), ltid = tid%32 (position within row)
    const int64_t tid  = threadIdx.x;
    const int64_t k    = tid / 32;  // row 0..3
    const int64_t ltid = tid % 32;  // 0..31

    const int nblock = n_per_row / QK_K;
    const int64_t group = i / nblock;
    const int64_t ibl   = i % nblock;
    dst_t * y = yy + group * 4 * n_per_row + k * n_per_row + ibl * QK_K;

    const float dall = __half2float(x[i].d[k]);
    const float dmin = __half2float(x[i].d[k + 4]);

    // ltid = 0..31 maps to: ib = ltid/4 (sub-block 0..7), j = ltid%4 (0..3)
    const int ib = ltid / 4;
    const int j  = ltid % 4;

    // Decode scale and min for this sub-block
    const uint8_t sl = x[i].scales_l[4 * ib + k];
    const uint8_t Ld_low = sl & 0xf;
    const uint8_t Lm_low = sl >> 4;

    const int sh_idx = (4 * ib + k) % 16;
    const int sh_shift = 4 * ((4 * ib + k) / 16);
    const uint8_t sh = (x[i].scales_h[sh_idx] >> sh_shift) & 0xf;
    const uint8_t Ld_high = sh & 0x3;
    const uint8_t Lm_high = (sh >> 2) & 0x3;

    const uint8_t Ld = Ld_low | (Ld_high << 4);
    const uint8_t Lm = Lm_low | (Lm_high << 4);

    const float d_sc = dall * Ld;
    const float m_sc = dmin * Lm;

    // Each thread decodes 8 elements: L[32*ib + j + offset] for 8 offsets
    const uint8_t * qs_base = x[i].qs + 64 * ib + 4 * k;

    const uint8_t q0 = qs_base[j + 0];   // L[j+0]  | (L[j+8] << 4)
    const uint8_t q1 = qs_base[j + 32];  // L[j+4]  | (L[j+12] << 4)
    const uint8_t q2 = qs_base[j + 16];  // L[j+16] | (L[j+24] << 4)
    const uint8_t q3 = qs_base[j + 48];  // L[j+20] | (L[j+28] << 4)

    y[32 * ib + j + 0]  = DequantizeCast<dst_t>::cast(d_sc * (q0 & 0xf) - m_sc);
    y[32 * ib + j + 8]  = DequantizeCast<dst_t>::cast(d_sc * (q0 >> 4)   - m_sc);
    y[32 * ib + j + 4]  = DequantizeCast<dst_t>::cast(d_sc * (q1 & 0xf) - m_sc);
    y[32 * ib + j + 12] = DequantizeCast<dst_t>::cast(d_sc * (q1 >> 4)   - m_sc);
    y[32 * ib + j + 16] = DequantizeCast<dst_t>::cast(d_sc * (q2 & 0xf) - m_sc);
    y[32 * ib + j + 24] = DequantizeCast<dst_t>::cast(d_sc * (q2 >> 4)   - m_sc);
    y[32 * ib + j + 20] = DequantizeCast<dst_t>::cast(d_sc * (q3 & 0xf) - m_sc);
    y[32 * ib + j + 28] = DequantizeCast<dst_t>::cast(d_sc * (q3 >> 4)   - m_sc);
}

template<typename dst_t>
static void dequantize_row_q4_K_r4_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    // nrows must be a multiple of 4
    const int64_t nblocks = (nrows / 4) * (n_per_row / QK_K);
    dequantize_block_q4_K_r4<<<nblocks, 128, 0, stream>>>(vx, y, n_per_row);
}

// Dequantize q2_K_r4 (4-row interleaved) format back to float/half/bf16.
// Each block_q2_k_r4 encodes 4 rows x QK_K elements.
template<typename dst_t>
static __global__ void dequantize_block_q2_K_r4(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                 const int64_t n_per_row) {
    const block_q2_k_r4 * x = (const block_q2_k_r4 *) vx;
    const int64_t i = blockIdx.x;

    const int64_t tid  = threadIdx.x;
    const int64_t k    = tid / 32;     // row 0..3
    const int64_t ltid = tid % 32;     // 0..31

    const int nblock = n_per_row / QK_K;
    const int64_t group = i / nblock;
    const int64_t ibl   = i % nblock;
    dst_t * y = yy + group * 4 * n_per_row + k * n_per_row + ibl * QK_K;

    const float d = __half2float(x[i].d[k]);
    const float dmin = __half2float(x[i].d[k + 4]);

    // ltid = 0..31 -> ib = 0..7, j = 0..3. Each ib covers 32 values:
    // the first 16 and last 16 values use separate scale/min nibbles.
    const int ib = ltid / 4;
    const int j  = ltid % 4;

    const uint8_t sc0 = x[i].scales[8 * ib + k + 0];
    const uint8_t sc1 = x[i].scales[8 * ib + k + 4];
    const float d0 = d * (sc0 & 0xF);
    const float m0 = dmin * (sc0 >> 4);
    const float d1 = d * (sc1 & 0xF);
    const float m1 = dmin * (sc1 >> 4);

    const uint8_t * qs_base = x[i].qs + 32 * ib + 4 * k;
    const uint8_t q0 = qs_base[j + 0];
    const uint8_t q1 = qs_base[j + 16];

    y[32 * ib + j +  0] = DequantizeCast<dst_t>::cast(d0 * ((q0 >> 0) & 0x3) - m0);
    y[32 * ib + j +  4] = DequantizeCast<dst_t>::cast(d0 * ((q0 >> 2) & 0x3) - m0);
    y[32 * ib + j +  8] = DequantizeCast<dst_t>::cast(d0 * ((q0 >> 4) & 0x3) - m0);
    y[32 * ib + j + 12] = DequantizeCast<dst_t>::cast(d0 * ((q0 >> 6) & 0x3) - m0);
    y[32 * ib + j + 16] = DequantizeCast<dst_t>::cast(d1 * ((q1 >> 0) & 0x3) - m1);
    y[32 * ib + j + 20] = DequantizeCast<dst_t>::cast(d1 * ((q1 >> 2) & 0x3) - m1);
    y[32 * ib + j + 24] = DequantizeCast<dst_t>::cast(d1 * ((q1 >> 4) & 0x3) - m1);
    y[32 * ib + j + 28] = DequantizeCast<dst_t>::cast(d1 * ((q1 >> 6) & 0x3) - m1);
}

template<typename dst_t>
static void dequantize_row_q2_K_r4_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t nblocks = (nrows / 4) * (n_per_row / QK_K);
    dequantize_block_q2_K_r4<<<nblocks, 128, 0, stream>>>(vx, y, n_per_row);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

// Dequantize q5_K_r4 (4-row interleaved) format back to float/half
// Each block_q5_k_r4 encodes 4 rows × QK_K elements.
// Launch with 128 threads per block: tid/32 selects row (0..3), tid%32 selects position.
// Each thread decodes 8 elements (one 32-element sub-block).
// Q5_K formula: x = dall * Ld * (low4 | (high1 << 4)) - dmin * Lm
template<typename dst_t>
static __global__ void dequantize_block_q5_K_r4(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                 const int64_t n_per_row) {
    const block_q5_k_r4 * x = (const block_q5_k_r4 *) vx;
    const int64_t i = blockIdx.x; // r4 block index

    const int64_t tid  = threadIdx.x;  // 0..127
    const int64_t k    = tid / 32;     // row 0..3
    const int64_t ltid = tid % 32;     // 0..31

    const int nblock = n_per_row / QK_K;
    const int64_t group = i / nblock;
    const int64_t ibl   = i % nblock;

    // Output pointer for this thread's row
    dst_t * y = yy + group * 4 * n_per_row + k * n_per_row + ibl * QK_K;

    const float dall = __half2float(x[i].d[k]);
    const float dmin = __half2float(x[i].d[k + 4]);

    // ltid = 0..31 -> ib = ltid/4 (sub-block 0..7), j = ltid%4 (0..3)
    const int ib = ltid / 4;
    const int j  = ltid % 4;

    // Decode scale (Ld) and min (Lm) for this sub-block
    // repack stored: scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4)
    const uint8_t sl = x[i].scales_l[4 * ib + k];
    const uint8_t Ld_low = sl & 0xf;
    const uint8_t Lm_low = sl >> 4;

    // scales_h packing: h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2), stored at scales_h[(4*ib+k)%16] shifted by 4*((4*ib+k)/16)
    const int sh_idx = (4 * ib + k) % 16;
    const int sh_shift = 4 * ((4 * ib + k) / 16);
    const uint8_t sh = (x[i].scales_h[sh_idx] >> sh_shift) & 0xf;
    const uint8_t Ld_high = sh & 0x3;
    const uint8_t Lm_high = (sh >> 2) & 0x3;

    const uint8_t Ld = Ld_low | (Ld_high << 4);
    const uint8_t Lm = Lm_low | (Lm_high << 4);

    const float d_sc = dall * Ld;
    const float m_sc = dmin * Lm;

    // Read packed low 4-bit quants (same layout as q4_K_r4)
    // qs layout from repack: qs[64*ib + 4*k + i + offset]
    const uint8_t * qs_base = x[i].qs + 64 * ib + 4 * k;
    const uint8_t q0 = qs_base[j +  0];  // (L[j+0] & 0xf) | ((L[j+8] & 0xf) << 4)
    const uint8_t q1 = qs_base[j + 16];  // (L[j+16] & 0xf) | ((L[j+24] & 0xf) << 4)
    const uint8_t q2 = qs_base[j + 32];  // (L[j+4] & 0xf) | ((L[j+12] & 0xf) << 4)
    const uint8_t q3 = qs_base[j + 48];  // (L[j+20] & 0xf) | ((L[j+28] & 0xf) << 4)

    // Read packed high 1-bit quants (the 5th bit)
    // qh layout from repack: qh[16*ib + 4*k + i] packs 8 elements' bit4:
    //   bit0 = L[j+0]>>4, bit1 = L[j+8]>>4, bit2 = L[j+4]>>4, bit3 = L[j+12]>>4,
    //   bit4 = L[j+16]>>4, bit5 = L[j+24]>>4, bit6 = L[j+20]>>4, bit7 = L[j+28]>>4
    const uint8_t qh = x[i].qh[16 * ib + 4 * k + j];

    // Reconstruct 5-bit values and dequantize: x = d_sc * q5val - m_sc
    y[32 * ib + j +  0] = DequantizeCast<dst_t>::cast(d_sc * ((q0 & 0xf) | (((qh >> 0) & 1) << 4)) - m_sc);
    y[32 * ib + j +  8] = DequantizeCast<dst_t>::cast(d_sc * ((q0 >>  4) | (((qh >> 1) & 1) << 4)) - m_sc);
    y[32 * ib + j +  4] = DequantizeCast<dst_t>::cast(d_sc * ((q2 & 0xf) | (((qh >> 2) & 1) << 4)) - m_sc);
    y[32 * ib + j + 12] = DequantizeCast<dst_t>::cast(d_sc * ((q2 >>  4) | (((qh >> 3) & 1) << 4)) - m_sc);
    y[32 * ib + j + 16] = DequantizeCast<dst_t>::cast(d_sc * ((q1 & 0xf) | (((qh >> 4) & 1) << 4)) - m_sc);
    y[32 * ib + j + 24] = DequantizeCast<dst_t>::cast(d_sc * ((q1 >>  4) | (((qh >> 5) & 1) << 4)) - m_sc);
    y[32 * ib + j + 20] = DequantizeCast<dst_t>::cast(d_sc * ((q3 & 0xf) | (((qh >> 6) & 1) << 4)) - m_sc);
    y[32 * ib + j + 28] = DequantizeCast<dst_t>::cast(d_sc * ((q3 >>  4) | (((qh >> 7) & 1) << 4)) - m_sc);
}

template<typename dst_t>
static void dequantize_row_q5_K_r4_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    // nrows must be a multiple of 4
    const int64_t nblocks = (nrows / 4) * (n_per_row / QK_K);
    dequantize_block_q5_K_r4<<<nblocks, 128, 0, stream>>>(vx, y, n_per_row);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void *vx, dst_t *y,
                                       const int64_t nrows,
                                       const int64_t n_per_row,
                                       cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int blocks = k / QK_K;
    dequantize_block_iq3_s<<<blocks, 32, 0, stream>>>(vx, y);
}

// Dequantize q6_K_r4 (4-row interleaved) format back to float/half
// Each block_q6_k_r4 encodes 4 rows × QK_K elements.
// Launch with 128 threads per block: tid/32 selects row (0..3), tid%32 selects position.
// Each thread decodes 8 elements (one 32-element sub-block, 4 positions × 2 scale groups).
template<typename dst_t>
static __global__ void dequantize_block_q6_K_r4(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                 const int64_t n_per_row) {
    const block_q6_k_r4 * x = (const block_q6_k_r4 *) vx;
    const int64_t i = blockIdx.x; // r4 block index

    const int64_t tid  = threadIdx.x;  // 0..127
    const int64_t k    = tid / 32;     // row 0..3
    const int64_t ltid = tid % 32;     // 0..31

    const int nblock = n_per_row / QK_K;
    const int64_t group = i / nblock;
    const int64_t ibl   = i % nblock;

    // Output pointer for this thread's row
    dst_t * y = yy + group * 4 * n_per_row + k * n_per_row + ibl * QK_K;

    const float d = __half2float(x[i].d[k]);

    // ltid = 0..31 -> ib = ltid/4 (sub-block 0..7), j = ltid%4 (0..3)
    const int ib = ltid / 4;
    const int j  = ltid % 4;

    // Scales: scales[8*ib + k + 0] for first 16 elements, scales[8*ib + k + 4] for last 16
    const int8_t sc0 = x[i].scales[8 * ib + k + 0];  // original scales[2*ib+0]
    const int8_t sc1 = x[i].scales[8 * ib + k + 4];  // original scales[2*ib+1]

    // Read packed low 4-bit quants
    // ql layout from repack: ql[64*ib + 4*k + j + offset]
    const uint8_t * ql_base = x[i].ql + 64 * ib + 4 * k;
    const uint8_t ql0 = ql_base[j +  0];  // (L[j+0] & 0xf) | ((L[j+8] & 0xf) << 4)
    const uint8_t ql1 = ql_base[j + 16];  // (L[j+16] & 0xf) | ((L[j+24] & 0xf) << 4)
    const uint8_t ql2 = ql_base[j + 32];  // (L[j+4] & 0xf) | ((L[j+12] & 0xf) << 4)
    const uint8_t ql3 = ql_base[j + 48];  // (L[j+20] & 0xf) | ((L[j+28] & 0xf) << 4)

    // Read packed high 2-bit quants
    // qh layout from repack: qh[32*ib + 4*k + j + offset]
    const uint8_t * qh_base = x[i].qh + 32 * ib + 4 * k;
    const uint8_t qh0 = qh_base[j +  0]; // (L[j+0]>>4) | ((L[j+8]>>4)<<2) | ((L[j+4]>>4)<<4) | ((L[j+12]>>4)<<6)
    const uint8_t qh1 = qh_base[j + 16]; // (L[j+16]>>4) | ((L[j+24]>>4)<<2) | ((L[j+20]>>4)<<4) | ((L[j+28]>>4)<<6)

    // Reconstruct 6-bit values L[offset] = (ql_low_4bits) | (qh_2bits << 4), then dequant = d * scale * (L - 32)
    // From ql0: L[j+0] low4 = ql0 & 0xf,  L[j+8] low4 = ql0 >> 4
    // From qh0: L[j+0] high2 = qh0 & 3,   L[j+8] high2 = (qh0>>2)&3, L[j+4] high2 = (qh0>>4)&3, L[j+12] high2 = (qh0>>6)&3
    // From ql2: L[j+4] low4 = ql2 & 0xf,  L[j+12] low4 = ql2 >> 4
    // From ql1: L[j+16] low4 = ql1 & 0xf, L[j+24] low4 = ql1 >> 4
    // From qh1: L[j+16] high2 = qh1 & 3,  L[j+24] high2 = (qh1>>2)&3, L[j+20] high2 = (qh1>>4)&3, L[j+28] high2 = (qh1>>6)&3
    // From ql3: L[j+20] low4 = ql3 & 0xf, L[j+28] low4 = ql3 >> 4

    // First 16 elements of sub-block (scale = sc0): positions j+0, j+8, j+4, j+12
    y[32 * ib + j +  0] = DequantizeCast<dst_t>::cast(d * sc0 * ((int8_t)(( ql0       & 0xf) | (((qh0 >> 0) & 3) << 4)) - 32));
    y[32 * ib + j +  8] = DequantizeCast<dst_t>::cast(d * sc0 * ((int8_t)(( ql0       >>  4) | (((qh0 >> 2) & 3) << 4)) - 32));
    y[32 * ib + j +  4] = DequantizeCast<dst_t>::cast(d * sc0 * ((int8_t)(( ql2       & 0xf) | (((qh0 >> 4) & 3) << 4)) - 32));
    y[32 * ib + j + 12] = DequantizeCast<dst_t>::cast(d * sc0 * ((int8_t)(( ql2       >>  4) | (((qh0 >> 6) & 3) << 4)) - 32));

    // Last 16 elements of sub-block (scale = sc1): positions j+16, j+24, j+20, j+28
    y[32 * ib + j + 16] = DequantizeCast<dst_t>::cast(d * sc1 * ((int8_t)(( ql1       & 0xf) | (((qh1 >> 0) & 3) << 4)) - 32));
    y[32 * ib + j + 24] = DequantizeCast<dst_t>::cast(d * sc1 * ((int8_t)(( ql1       >>  4) | (((qh1 >> 2) & 3) << 4)) - 32));
    y[32 * ib + j + 20] = DequantizeCast<dst_t>::cast(d * sc1 * ((int8_t)(( ql3       & 0xf) | (((qh1 >> 4) & 3) << 4)) - 32));
    y[32 * ib + j + 28] = DequantizeCast<dst_t>::cast(d * sc1 * ((int8_t)(( ql3       >>  4) | (((qh1 >> 6) & 3) << 4)) - 32));
}

template<typename dst_t>
static void dequantize_row_q6_K_r4_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    // nrows must be a multiple of 4
    const int64_t nblocks = (nrows / 4) * (n_per_row / QK_K);
    dequantize_block_q6_K_r4<<<nblocks, 128, 0, stream>>>(vx, y, n_per_row);
}

// CPU MoE kernels interleave four IQ rows and scramble the seven sign bits.
// Decode that storage directly so NUMA GPU prefill can reuse the packed weights.
static __device__ __forceinline__ uint8_t iq_r4_signs(uint8_t packed) {
    const uint8_t original = (packed ^ (packed << 1)) & 127;
    return ksigns_iq2xs[original];
}

template<typename dst_t, bool iq3>
static __global__ void dequantize_block_iq_r4(const void * __restrict__ vx,
                                              dst_t * __restrict__ output,
                                              const int64_t n_per_row) {
    const int64_t block = blockIdx.x;
    const int row = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ib = lane % 8;
    const int il = lane / 8;
    const int64_t blocksPerRow = n_per_row / QK_K;
    dst_t *y = output + (block / blocksPerRow * 4 + row) * n_per_row +
        block % blocksPerRow * QK_K + 32 * ib + 8 * il;
    if constexpr (iq3) {
        const block_iq3_xxs_r4 &x = static_cast<const block_iq3_xxs_r4 *>(vx)[block];
        const uint8_t *sas = x.sas + 16 * ib + 4 * row;
        const int scale = (sas[0] & 1) | ((sas[1] & 1) << 1) |
                          ((sas[2] & 1) << 2) | ((sas[3] & 1) << 3);
        const uint8_t signs = iq_r4_signs(sas[il] >> 1);
        const uint8_t *qs = x.qs + 32 * ib + 8 * row + 2 * il;
        const uint32_t grid0 = iq3xxs_grid[qs[0]], grid1 = iq3xxs_grid[qs[1]];
        const float d = __half2float(x.d[row]) * (0.5f + scale) * 0.5f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q0 = (grid0 >> (8 * j)) & 255;
            const int q1 = (grid1 >> (8 * j)) & 255;
            y[j] = DequantizeCast<dst_t>::cast(d * q0 * ((signs & (1 << j)) ? -1.0f : 1.0f));
            y[j + 4] = DequantizeCast<dst_t>::cast(d * q1 * ((signs & (1 << (j + 4))) ? -1.0f : 1.0f));
        }
    } else {
        const block_iq2_xs_r4 &x = static_cast<const block_iq2_xs_r4 *>(vx)[block];
        const uint16_t q2 = x.qs[16 * ib + 4 * row + il];
        const uint64_t grid = iq2xs_grid[q2 & 511];
        const uint8_t signs = iq_r4_signs(q2 >> 9);
        const int scale = (x.scales[4 * ib + row] >> (4 * (il / 2))) & 15;
        const float d = __half2float(x.d[row]) * (0.5f + scale) * 0.25f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q = (grid >> (8 * j)) & 255;
            y[j] = DequantizeCast<dst_t>::cast(d * q * ((signs & (1 << j)) ? -1.0f : 1.0f));
        }
    }
}

template<typename dst_t, bool iq3>
static void dequantize_row_iq_r4_cuda(const void *vx, dst_t *y, const int64_t nrows,
                                     const int64_t n_per_row, cudaStream_t stream) {
    const int64_t blocks = nrows * n_per_row / (4 * QK_K);
    dequantize_block_iq_r4<dst_t, iq3><<<blocks, 128, 0, stream>>>(vx, y, n_per_row);
}

template<typename dst_t, bool iq2s>
static __global__ void dequantize_block_iq2_other_r4(const void *__restrict__ vx,
        dst_t *__restrict__ output, int64_t columns) {
    const int64_t block = blockIdx.x;
    const int row = threadIdx.x / 32;
    const int lane = threadIdx.x % 32, ib = lane % 8, il = lane / 8;
    const int64_t blocksPerRow = columns / QK_K;
    dst_t *y = output + (block / blocksPerRow * 4 + row) * columns +
        block % blocksPerRow * QK_K + 32 * ib + 8 * il;
    uint64_t grid;
    uint8_t signs;
    float d;
    if constexpr (iq2s) {
        const auto &x = static_cast<const block_iq2_s_r4 *>(vx)[block];
        const int index = 16 * ib + 4 * row + il;
        const int code = x.qs[index] | ((unsigned(x.qh[4 * ib + row]) << (8 - 2 * il)) & 0x300u);
        grid = iq2s_grid[code];
        signs = x.signs[index];
        const int scale = (x.scales[4 * ib + row] >> (4 * (il / 2))) & 15;
        d = __half2float(x.d[row]) * (0.5f + scale) * 0.25f;
    } else {
        const auto &x = static_cast<const block_iq2_xxs_r4 *>(vx)[block];
        const uint8_t *sas = x.sas + 16 * ib + 4 * row;
        const int scale = (sas[0] & 1) | ((sas[1] & 1) << 1) |
                          ((sas[2] & 1) << 2) | ((sas[3] & 1) << 3);
        grid = iq2xxs_grid[x.qs[16 * ib + 4 * row + il]];
        signs = iq_r4_signs(sas[il] >> 1);
        d = __half2float(x.d[row]) * (0.5f + scale) * 0.25f;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (grid >> (8 * j)) & 255;
        y[j] = DequantizeCast<dst_t>::cast(d * q * ((signs & (1 << j)) ? -1.0f : 1.0f));
    }
}

template<typename dst_t, bool iq2s>
static void dequantize_row_iq2_other_r4_cuda(const void *vx, dst_t *y,
        int64_t rows, int64_t columns, cudaStream_t stream) {
    const int64_t blocks = rows * columns / (4 * QK_K);
    if (blocks) dequantize_block_iq2_other_r4<dst_t, iq2s><<<blocks, 128, 0, stream>>>(vx, y, columns);
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
            return dequantize_row_q2_0_cuda<float>;
        case GGML_TYPE_IQ2_XXS_R4:
            return dequantize_row_iq2_other_r4_cuda<float, false>;
        case GGML_TYPE_IQ2_S_R4:
            return dequantize_row_iq2_other_r4_cuda<float, true>;
        case GGML_TYPE_IQ2_XS_R4:
            return dequantize_row_iq_r4_cuda<float, false>;
        case GGML_TYPE_IQ3_XXS_R4:
            return dequantize_row_iq_r4_cuda<float, true>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q8_0:
            return dequantize_block_q8_0_f32_cuda;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q4_K_R4:
            return dequantize_row_q4_K_r4_cuda;
        case GGML_TYPE_Q2_K_R4:
            return dequantize_row_q2_K_r4_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q5_K_R4:
            return dequantize_row_q5_K_r4_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_Q6_K_R4:
            return dequantize_row_q6_K_r4_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        default: {
            static std::set<ggml_type> warned_types;
            if (warned_types.find(type) == warned_types.end()) {
                warned_types.insert(type);
                printf("Warning: Type %s doesn't have FP32 dequant func. Prefill may be very slow...\n", ggml_type_name(type));
            }
            return nullptr;
        }
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
            return dequantize_row_q2_0_cuda<half>;
        case GGML_TYPE_IQ2_XXS_R4:
            return dequantize_row_iq2_other_r4_cuda<half, false>;
        case GGML_TYPE_IQ2_S_R4:
            return dequantize_row_iq2_other_r4_cuda<half, true>;
        case GGML_TYPE_IQ2_XS_R4:
            return dequantize_row_iq_r4_cuda<half, false>;
        case GGML_TYPE_IQ3_XXS_R4:
            return dequantize_row_iq_r4_cuda<half, true>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        // case GGML_TYPE_Q5_0:
        //     return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        // case GGML_TYPE_Q5_1:
        //     return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        // case GGML_TYPE_Q6_0:
        //     return dequantize_row_q6_0_cuda;
        case GGML_TYPE_Q8_0:
            // if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= CC_PASCAL) {
               // return dequantize_block_q8_0_f16_cuda;
            //}
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q4_K_R4:
            return dequantize_row_q4_K_r4_cuda;
        case GGML_TYPE_Q2_K_R4:
            return dequantize_row_q2_K_r4_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q5_K_R4:
            return dequantize_row_q5_K_r4_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_Q6_K_R4:
            return dequantize_row_q6_K_r4_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        // case GGML_TYPE_IQ1_KT:
        //    return dequantize_row_iq1_kt_cuda;
        // case GGML_TYPE_IQ2_KT:
        //    return dequantize_row_iq2_kt_cuda;
        // case GGML_TYPE_IQ3_KT:
        //    return dequantize_row_iq3_kt_cuda;
        // case GGML_TYPE_IQ4_KT:
        //    return dequantize_row_iq4_kt_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        // case GGML_TYPE_IQ1_S_R4:
        //    return dequantize_row_iq1_s_r4_cuda;
        // case GGML_TYPE_IQ1_M_R4:
        //    return dequantize_row_iq1_m_r4_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        // case GGML_TYPE_IQ1_BN:
        //    return dequantize_row_iq1_bn_cuda;
        // case GGML_TYPE_IQ2_BN:
        //    return dequantize_row_iq2_bn_cuda;
        // case GGML_TYPE_IQ4_NL:
        //    return dequantize_row_iq4_nl_cuda;
        // case GGML_TYPE_MXFP4:
        //    return dequantize_row_mxfp4_cuda;
        // case GGML_TYPE_IQ4_XS:
        //    return dequantize_row_iq4_xs_cuda;
        // case GGML_TYPE_IQ4_KS:
        //    return dequantize_row_iq4_ks_cuda;
        // case GGML_TYPE_IQ2_K:
        //    return dequantize_row_iq2_k_cuda;
        // case GGML_TYPE_IQ3_K:
        //    return dequantize_row_iq3_k_cuda;
        // case GGML_TYPE_IQ2_KL:
        //    return dequantize_row_iq2_kl_cuda;
        // case GGML_TYPE_IQ3_S:
        //    return dequantize_row_iq3_s_cuda;
        // case GGML_TYPE_F32:
        //    return convert_unary_cuda<float>;
        // case GGML_TYPE_BF16:
        //    return convert_from_bf16_cuda;
        // case GGML_TYPE_IQ2_K_R4:
        //    return dequantize_row_iq2_k_r4_cuda;
        // case GGML_TYPE_IQ3_K_R4:
        //    return dequantize_row_iq3_k_r4_cuda;
        // case GGML_TYPE_IQ4_K_R4:
        //    return dequantize_row_iq4_k_r4_cuda;
        // case GGML_TYPE_IQ4_KS_R4:
        //    return dequantize_row_iq4_ks_r4_cuda;
        // case GGML_TYPE_IQ5_K_R4:
        //    return dequantize_row_iq5_k_r4_cuda;
        default: {
            static std::set<ggml_type> warned_types;
            if (warned_types.find(type) == warned_types.end()) {
                warned_types.insert(type);
                printf("Warning: Type %s doesn't has dequant func. Prefill may be very slow...\n", ggml_type_name(type));
            }
            return nullptr;
        }
    }
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
            return dequantize_row_q2_0_cuda<__nv_bfloat16>;
        case GGML_TYPE_IQ2_XXS_R4:
            return dequantize_row_iq2_other_r4_cuda<__nv_bfloat16, false>;
        case GGML_TYPE_IQ2_S_R4:
            return dequantize_row_iq2_other_r4_cuda<__nv_bfloat16, true>;
        case GGML_TYPE_IQ2_XS_R4:
            return dequantize_row_iq_r4_cuda<__nv_bfloat16, false>;
        case GGML_TYPE_IQ3_XXS_R4:
            return dequantize_row_iq_r4_cuda<__nv_bfloat16, true>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q8_0:
            return dequantize_block_q8_0_bf16_cuda;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q4_K_R4:
            return dequantize_row_q4_K_r4_cuda;
        case GGML_TYPE_Q2_K_R4:
            return dequantize_row_q2_K_r4_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q5_K_R4:
            return dequantize_row_q5_K_r4_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_Q6_K_R4:
            return dequantize_row_q6_K_r4_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        default: {
            static std::set<ggml_type> warned_types;
            if (warned_types.find(type) == warned_types.end()) {
                warned_types.insert(type);
                printf("Warning: Type %s doesn't have BF16 dequant func. Prefill may be very slow...\n", ggml_type_name(type));
            }
            return nullptr;
        }
    }
}
