#pragma once
#include "fastllm-gguf-mmvq-common.cuh"
#include "fastllm-gguf-mmvq-dispatch.cuh"

template <ggml_type type, int ncols_y, int nwarps, typename OType, int StoreMode = 0>
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
    // Q5_K/Q6_K reuse decoded weights across input rows. One output row per CTA
    // avoids the register pressure of the two-output-row verifier tile.
    constexpr int rows_per_cuda_block = ((type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K) && ncols_y == 4) || ncols_y < 4 ? 1 : 2;
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

        if constexpr (type == GGML_TYPE_Q5_K && ncols_y >= 2 && ncols_y <= 4) {
            const FastllmQ5MmvqFragment fragment(
                ((const block_q5_K *)vx)[row0 * blocks_per_row_x + kbx], kqs);
#pragma unroll
            for (int j = 0; j < ncols_y; ++j)
                tmp[j][0] += fragment.Dot(&y[j * blocks_per_col_y + kby], kqs);
        } else if constexpr (type == GGML_TYPE_Q6_K && ncols_y >= 2 && ncols_y <= 4) {
            const FastllmQ6MmvqFragment fragment(
                ((const block_q6_K *)vx)[row0 * blocks_per_row_x + kbx], kqs);
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                tmp[j][0] += fragment.Dot(&y[j * blocks_per_col_y + kby], kqs);
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    // The final two-row tile may contain only one weight row.
                    // Mask the read as well as the output store below.
                    if (rows_per_cuda_block == 1 || row0 + i < nrows_x) {
                        tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
                    }
                }
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
            FastllmGgufStore<StoreMode>(dst + j*nrows_dst + row0 + threadIdx.x, tmp[j][threadIdx.x]);
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps, typename OType, int StoreMode = 0>
#if !defined(USE_ROCM)
__launch_bounds__(nwarps * WARP_SIZE, ((type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K) && ncols_y >= 2 && ncols_y <= 4 ? 4 : 1))
#endif
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
    mul_mat_vec_q<type, ncols_y, nwarps, OType, StoreMode>(cx, cy, (OType *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <ggml_type type, int nwarps, typename OType, int StoreMode = 0>
static void mul_mat_vec_q_cuda_T(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {

    assert(ncols_x % ggml_blck_size(type) == 0);
    assert(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    const int64_t rows_per_cuda_block = ((type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K) && ncols_y == 4) || ncols_y < 4 ? 1 : 2;
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps, OType, StoreMode><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        default:
            printf("fatal error: ncols_y = %d exceeds MMVQ_MAX_BATCH_SIZE\n", ncols_y);
            exit(0);
            break;
    }
}

#if !defined(USE_ROCM)
#include "fastllm-gguf-iq3-gemv.cuh"
#include "fastllm-gguf-q3-k-gemv.cuh"
#include "fastllm-gguf-small-mmvq.cuh"
#endif

// Pack independent decode outputs into one CTA while retaining the legacy
// four-warp dot-product and reduction order in four virtual warp groups.
// This reduces CTA scheduling overhead for broad Q5_K/Q6_K projections.
template <ggml_type type, typename Output, int Inputs = 1>
static __global__ void FastllmGgufMmvqWarpRows(
        const void * __restrict__ weights, const block_q8_1 * __restrict__ input,
        Output * __restrict__ output, int columns, int rows, int inputStride, int outputStride) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int blocksPerIter = vdr * 4 * WARP_SIZE / qi;
    constexpr vec_dot_q_cuda_t dot = get_vec_dot_q_cuda(type);
    const int blocksPerRow = columns / qk;
    const int row = blockIdx.x * 4 + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= rows) return;
    float partial[Inputs][4] = {};
#pragma unroll
    for (int warp = 0; warp < 4; ++warp) {
        const int tid = warp * WARP_SIZE + lane;
        const int qs = vdr * (tid % (qi / vdr));
        for (int block = tid / (qi / vdr); block < blocksPerRow; block += blocksPerIter) {
            if constexpr (Inputs == 1) {
                partial[0][warp] += dot(weights, input + block * (qk / QK8_1),
                    row * blocksPerRow + block, qs);
            } else {
                const FastllmQ6MmvqFragment fragment(
                    static_cast<const block_q6_K *>(weights)[row * blocksPerRow + block], qs);
#pragma unroll
                for (int j = 0; j < Inputs; ++j)
                    partial[j][warp] += fragment.Dot(
                        input + j * (inputStride / QK8_1) + block * (qk / QK8_1), qs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < Inputs; ++j) {
        const float sum = warp_reduce_sum(((partial[j][0] + partial[j][1]) + partial[j][2]) + partial[j][3]);
        if (lane == 0) FastllmGgufStore<0>(output + j * outputStride + row, sum);
    }
}

template <ggml_type type, typename OType, int StoreMode>
void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0,
    cudaStream_t stream) {
#if !defined(USE_ROCM)
    if constexpr (StoreMode == 0 && (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
        if ((ncols_y == 1 || (type == GGML_TYPE_Q6_K && ncols_y >= 2 && ncols_y <= 4)) &&
            ne2 == 1 && !ids_data && nrows_dst >= nrows_x &&
            ncols_x == 2560 && nrows_x >= (type == GGML_TYPE_Q5_K ? 2048 : 32768)) {
            int major = 0, minor = 0;
            const int device = FastllmCudaGetDevice();
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
                major == 7 && minor == 5) {
                if (ncols_y == 1) {
                    FastllmGgufMmvqWarpRows<type, OType, 1><<<(nrows_x + 3) / 4, 128, 0, stream>>>(
                        vx, static_cast<const block_q8_1 *>(vy), dst, ncols_x, nrows_x, nrows_y, nrows_dst);
                } else if constexpr (type == GGML_TYPE_Q6_K) {
                    switch (ncols_y) {
                        case 2: FastllmGgufMmvqWarpRows<type, OType, 2><<<(nrows_x + 3) / 4, 128, 0, stream>>>(
                            vx, static_cast<const block_q8_1 *>(vy), dst, ncols_x, nrows_x, nrows_y, nrows_dst); break;
                        case 3: FastllmGgufMmvqWarpRows<type, OType, 3><<<(nrows_x + 3) / 4, 128, 0, stream>>>(
                            vx, static_cast<const block_q8_1 *>(vy), dst, ncols_x, nrows_x, nrows_y, nrows_dst); break;
                        case 4: FastllmGgufMmvqWarpRows<type, OType, 4><<<(nrows_x + 3) / 4, 128, 0, stream>>>(
                            vx, static_cast<const block_q8_1 *>(vy), dst, ncols_x, nrows_x, nrows_y, nrows_dst); break;
                    }
                }
                return;
            }
        }
    }
    if constexpr (type == GGML_TYPE_Q3_K) {
        if (ncols_y == 1 && ne2 == 1 && ids_data == nullptr && nrows_dst >= nrows_x &&
            fastllm_gguf_q3_k::Supports(vy, ncols_x, nrows_x)) {
            fastllm_gguf_q3_k::Launch(vx, static_cast<const block_q8_1 *>(vy), dst,
                                      ncols_x, nrows_x, stream);
            return;
        }
    }
    if constexpr (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
        if (ncols_y == 1 && ne2 == 1 && ids_data == nullptr &&
            fastllm_gguf_iq3::Supports(vy, ncols_x, nrows_x)) {
            fastllm_gguf_iq3::Launch<type, false, OType, StoreMode>(
                vx, nullptr, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, stream);
            return;
        }
    }
    if constexpr (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS ||
                  type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q2_K) {
        if (ncols_y >= 2 && ncols_y <= 8 && ne2 == 1 && ids_data == nullptr &&
            fastllm_gguf_small_mmvq::Supports(vy, ncols_x, nrows_x, nrows_y, nrows_dst)) {
            fastllm_gguf_small_mmvq::LaunchBatch<type, OType, StoreMode>(
                vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,
                ncols_y, nrows_y, nrows_dst, stream);
            return;
        }
    }
#endif
    // Up to four input rows benefit from four-way K reduction. B5-B8 has
    // twice the live accumulator state after packing two output rows per
    // block, so a single warp preserves occupancy. Batched expert slices
    // also use one warp to avoid multiplying register pressure by ne2.
    // MTP verification retains decode's reduction order even at B5-B8.
    const bool exactBatch = ncols_y < fastllm::FastllmCudaGetLinearExactBatchThreshold();
    if (ne2 < 2 && (ncols_y <= 4 || exactBatch)) {
        mul_mat_vec_q_cuda_T<type, 4, OType, StoreMode>(
            vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y,
            nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
    } else {
        mul_mat_vec_q_cuda_T<type, 1, OType, StoreMode>(
            vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y,
            nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
    }
}


#define FASTLLM_INSTANTIATE_MMVQ(TYPE, OUTPUT, STORE) \
    template void mul_mat_vec_q_cuda<TYPE, OUTPUT, STORE>( \
        const void *, const void *, OUTPUT *, const char *, \
        int, int, int, int, int, int, uint64_t, uint64_t, uint64_t, int64_t, cudaStream_t);

#define FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(TYPE) \
    FASTLLM_INSTANTIATE_MMVQ(TYPE, float, 0) \
    FASTLLM_INSTANTIATE_MMVQ(TYPE, half, 0) \
    FASTLLM_INSTANTIATE_MMVQ(TYPE, __nv_bfloat16, 0)

#define FASTLLM_INSTANTIATE_MMVQ_STORES(TYPE) \
    FASTLLM_INSTANTIATE_MMVQ(TYPE, half, 1) \
    FASTLLM_INSTANTIATE_MMVQ(TYPE, half, 2)
