//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "fastllm-gguf-mmq-common.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"
#include "iqk_cuda_common.h"

#include <climits>
#include <cstdint>

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256
#define MMQ_NWARPS 8

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int & kbx0, const int & i_max, const int & stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_S_R4:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            fprintf(stderr, "Unhandled type %s (%d)\n", ggml_type_name(type_x), type_x);
            GGML_ABORT("fatal error");
            break;
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

static constexpr int get_mmq_x_max_host(const int cc) {
    // On NVIDIA Volta the device-side MMQ kernels are only instantiated up to
    // MMQ_DP4A_MAX_BATCH_SIZE when GGML_CUDA_FORCE_MMQ is defined (see
    // get_mmq_x_max_device() below). If the host dispatch picks a larger tile,
    // the kernel hits NO_DEVICE_CODE at runtime. Without GGML_CUDA_FORCE_MMQ
    // the host already capped at MMQ_DP4A_MAX_BATCH_SIZE so cuBLAS handles
    // larger batches. Either way the bound on Volta is the same, so collapse
    // the two paths and document the constraint.
    return int8_mma_available(cc) ? 128 :
        cc >= CC_VOLTA && cc < CC_OFFSET_AMD ? MMQ_DP4A_MAX_BATCH_SIZE : 64;
}

static constexpr __device__ int get_mmq_x_max_device() {
#ifdef INT8_MMA_AVAILABLE
    return 128;
#else // INT8_MMA_AVAILABLE

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    return 128;
#else // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= CC_VOLTA
#ifdef GGML_CUDA_FORCE_MMQ
    return MMQ_DP4A_MAX_BATCH_SIZE;
#else // GGML_CUDA_FORCE_MMQ
    return 128;
#endif // GGML_CUDA_FORCE_MMQ
#else // __CUDA_ARCH__ >= CC_VOLTA

    return 64;
#endif // __CUDA_ARCH__ >= CC_VOLTA

#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#endif // INT8_MMA_AVAILABLE
}

static constexpr int get_mmq_y_host(const int cc) {
    return cc >= CC_OFFSET_AMD ? (cc == CC_RDNA1 ? 64 : 128) : (cc >= CC_VOLTA ? 128 : 64);
}

static constexpr __device__ int get_mmq_y_device() {
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA1)
    return 64;
#else
    return 128;
#endif // defined RDNA1
#else
#if __CUDA_ARCH__ >= CC_VOLTA
    return 128;
#else
    return 64;
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_0   + mmq_y/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_1   + mmq_y/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*4/QI8_0 + mmq_y/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_1 + mmq_y/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE         + mmq_y,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y,                                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_K,                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_K   + mmq_y/QI5_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI6_K   + mmq_y/QI6_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    switch (type) {
        case GGML_TYPE_Q4_0    : return MMQ_DP4A_TXS_Q4_0;
        case GGML_TYPE_Q4_1    : return MMQ_DP4A_TXS_Q4_1;
        case GGML_TYPE_Q5_0    : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q5_1    : return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q6_0    : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q8_0    : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q2_K    : return MMQ_DP4A_TXS_Q2_K;
        case GGML_TYPE_Q3_K    : return MMQ_DP4A_TXS_Q3_K;
        case GGML_TYPE_Q4_K    : return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K    : return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_Q6_K    : return MMQ_DP4A_TXS_Q6_K;
        case GGML_TYPE_IQ2_XXS : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_XS  : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_S   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_XXS : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_S   : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S   : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S_R4: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_XS  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_NL  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_MXFP4   : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_KL  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_KS  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_KSS : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_KS  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_KS_R4  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ5_KS  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ5_KS_R4  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_KS  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_K   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_K_R4: return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_K   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_K_R4: return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ4_K   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ4_K_R4: return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ5_K   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ5_K_R4: return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ6_K   : return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ1_KT  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_KT  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_KT  : return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_KT  : return MMQ_DP4A_TXS_Q8_0;
        default                : return tile_x_sizes{0, 0, 0};
    }
}

#define MMQ_MMA_TILE_X_K_Q8_0 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q8_1 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q2_K (2*WARP_SIZE + WARP_SIZE                         + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2*WARP_SIZE + WARP_SIZE/2                       + 4)
#define MMQ_MMA_TILE_X_K_Q6_K (2*WARP_SIZE + WARP_SIZE/QI6_K     + WARP_SIZE/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");

static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0    : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_1    : return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_0    : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q5_1    : return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q6_0    : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q8_0    : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q2_K    : return MMQ_MMA_TILE_X_K_Q2_K;
        case GGML_TYPE_Q3_K    : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_Q4_K    : return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_K    : return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q6_K    : return MMQ_MMA_TILE_X_K_Q6_K;
        case GGML_TYPE_IQ2_XXS : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_XS  : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ2_S   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ3_XXS : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ3_S   : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ1_S   : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ1_S_R4: return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_XS  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_NL  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_MXFP4   : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_KL  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ3_KS  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_KSS : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_KS  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_KS_R4  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ5_KS  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ5_KS_R4  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_KS  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_K   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ2_K_R4: return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ3_K   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ3_K_R4: return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ4_K   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ4_K_R4: return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ5_K   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ5_K_R4: return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ6_K   : return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ1_KT  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_KT  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ3_KT  : return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_KT  : return MMQ_MMA_TILE_X_K_Q8_0;
        default                : return 0;
    }
}

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)

static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    return int8_mma_available(cc) && mmq_x >= 48 ? 16 : 8;
}

#ifdef INT8_MMA_AVAILABLE
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}
#else
static constexpr __device__ int mmq_get_granularity_device(const int /* mmq_x */) {
    return 8;
}
#endif // INT8_MMA_AVAILABLE

// ------------------------------------------------------------

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *)(x + i*stride) + kbx0 + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_s_r4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / 4;
    const int kqsx = threadIdx.x % 4;

    int32_t grid32[2];

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }
        const int i4 = i/4;
        const int ir = i%4;

        const block_iq1_s_r4 * bxi = (const block_iq1_s_r4 *)(x + 4*i4*stride + 4*sizeof(half)) + kbx0 + kbx;

        grid32[0] = iq1s_grid_gpu[bxi->qs[4*kqsx+ir] | (((bxi->qh[ir] >> 3*kqsx) & 7) << 8)];
        grid32[1] = ((grid32[0] >> 4) & 0x0f0f0f0f) << 3;
        grid32[0] = (grid32[0] & 0x0f0f0f0f) << 3;
        const int shift = bxi->qh[ir] & 0x8000 ? 0x09090909 : 0x07070707;

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kbx + 2*kqsx + 0] = __vsubss4(grid32[0], shift);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kbx + 2*kqsx + 1] = __vsubss4(grid32[1], shift);
#else
        // TODO
        //x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / 4;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }
        const int i4 = i/4;
        const int ir = i%4;

        const half * dptr = (const half *)(x + 4*i4*stride);
        const block_iq1_s_r4 * bxi = (const block_iq1_s_r4 *)(dptr + 4) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = 0.125f * __half2float(dptr[ir]) * (((bxi->qh[ir] >> 11) & 14) + 1);
#else
        // TODO
        //x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_0)];
                }

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(WARP_SIZE + 1) + k0/QR4_0], u,
                     x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *)(x + i*stride) + kbx0 + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + 0]     = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_1)];
                }

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(WARP_SIZE + 1) + k0/QR4_1], u,
                     x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI5_0;
    const int kqsx = threadIdx.x % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *)(x + i*stride) + kbx0 + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI5_1;
    const int kqsx = threadIdx.x % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *)(x + i*stride) + kbx0 + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI6_0;
    const int kqsx = threadIdx.x % QI6_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_0 * bxi = (const block_q6_0 *)(x + i*stride) + kbx0 + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, kqsx%2) >> 4*(kqsx/2);

        int qs0 = ((ql >> 0) & 0x0F0F0F0F) | ((qh << 4) & 0x30303030);
        int qs1 = ((ql >> 4) & 0x0F0F0F0F) | ((qh << 2) & 0x30303030);
        qs0 = __vsubss4(qs0, 0x20202020); // subtract 32
        qs1 = __vsubss4(qs1, 0x20202020); // subtract 32

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI6_0) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI6_0) + kqsx + QI6_0] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI6_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI6_0) + kqsx + QI6_0] = qs1;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_0) {
        int i = i0 + threadIdx.y * QI6_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_0 * bxi = (const block_q6_0 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI6_0) + i/QI6_0 + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *)(x + i*stride) + kbx0 + kbx;

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 0         + threadIdx.x] = get_int_b2(bxi[0].qs,               kqsx);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE/QI8_0].qs, kqsx);
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 0         + threadIdx.x] = get_int_b2(bxi[0].qs,               kqsx);
        x_qs[i*(2*WARP_SIZE + 1)     + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE/QI8_0].qs, kqsx);
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = 2*WARP_SIZE / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0/2) {
        int i = i0 + threadIdx.y * (QI8_0/2) + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0             + kbxd] = bxi->d;
#else
        x_df[i*(2*WARP_SIZE/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % WARP_SIZE],
                     x_df[i*(2*WARP_SIZE/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (WARP_SIZE/QI8_1)]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    mma_A A[ntx];
    float dA[ntx][mma_C::ne/2];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

    #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
        const int k0 = k00 + k01;
        mma_B  B;
        float dB[mma_C::ne/2];
        B.load(y_qs + k01, MMQ_TILE_Y_K);
        #pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = mma_C::get_j(l);
            if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            A[n].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
            #pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);
                dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
            mma_C C;
            C.mma_K8(A[n], B);
            #pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(n)*mma_C::ne + l] += C.x[l]*dA[n][l/2]*dB[l%2];
            }
        }
        #pragma unroll
        for (int j0 = ntx*mma_C::J; j0 < mmq_x; j0 += ntx*mma_C::J) {
            B.load(y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            #pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);
                if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma_K8(A[n], B);
                #pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += C.x[l]*dA[n][l/2]*dB[l%2];
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    mma_A    A[ntx][WARP_SIZE/QI8_1];
    float2 dmA[ntx][mma_C::ne/2][WARP_SIZE/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            A[n][k01/QI8_1].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B    B;
            float2 dsB[mma_C::ne/2];

            B.load(y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma_K8(A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*WARP_SIZE + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_A_I16K8 mma_A_K8;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/8].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            mma_B B[2];
            float dB[mma_C::ne/2];

            B[0].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma_K4(A[n][k01/4 + 0], B[0]);
                C[1].mma_K4(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI2_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI2_K) {
        int i = i0 + threadIdx.y*(WARP_SIZE/QI2_K) + threadIdx.x/QI2_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *)(x + i*stride) + kbx0;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k] = x_qs_k;
#endif // INT8_MMA_AVAILABLE
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#ifdef INT8_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i*(WARP_SIZE + 1)       + kqsx] = x_dm_ik;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[mmq_x/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                if (k01 < WARP_SIZE/2) {
                    constexpr int ns = 2;
                    sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                        &x_dm[i*(WARP_SIZE + 1) + k0/4], k01 < WARP_SIZE/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                        &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
                } else {
                    constexpr int ns = 1;
                    sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                        &x_dm[i*(WARP_SIZE + 1) + k0/4], k01 < WARP_SIZE/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                        &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_A_I16K8 mma_A_K8;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne/2][8];
    float  mA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/QI8_1].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float2 dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B B[2];

            B[0].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

            mma_C Cm[2];
            if (k01 >= WARP_SIZE * 3/4) {
                mma_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                Cm[0].mma_K4(A1, B[0]);
                Cm[1].mma_K4(A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C Cd[2];

                Cd[0].mma_K4(A[n][k01/4 + 0], B[0]);
                Cd[1].mma_K4(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= WARP_SIZE * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp*(k01 < WARP_SIZE/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE * 3/4; k01 += QI8_1) {
            float2 sB[mma_C::ne/2];

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI3_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI3_K) {
        int i = i0 + threadIdx.y * (WARP_SIZE/QI3_K) + threadIdx.x / QI3_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k] = x_qs_k;
#endif // INT8_MMA_AVAILABLE
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*8) {
        int i = i0 + threadIdx.y*8 + threadIdx.x/(WARP_SIZE/8);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        const int ksc = threadIdx.x % (WARP_SIZE/8);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#ifdef INT8_MMA_AVAILABLE
        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + sizeof(int)*(threadIdx.x % (WARP_SIZE/8)) + l] = d*sc8[l];
        }
#else
        x_sc[i*(WARP_SIZE/8) + i/8 + threadIdx.x % (WARP_SIZE/8)] = sc;
#endif // INT8_MMA_AVAILABLE
    }

#ifndef INT8_MMA_AVAILABLE
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*WARP_SIZE) {
        int i = (i0 + threadIdx.y*WARP_SIZE + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        x_df[i] = bxi->d;
    }
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(WARP_SIZE/8) + i/8)) + k0/4;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;
        const int qs0 = get_int_b4(bxi->qs, threadIdx.x);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(threadIdx.x/8) + threadIdx.x % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(threadIdx.x/8) + threadIdx.x % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // INT8_MMA_AVAILABLE
    }

#ifdef INT8_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*16) {
        int i = (i0 + threadIdx.y*16 + threadIdx.x/(WARP_SIZE/16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*QI4_K) {
        int i = (i0 + threadIdx.y*QI4_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;

        x_dm[i] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0 + (threadIdx.x % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(WARP_SIZE/8) + i/8 + ksc] = scales8;
    }
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(WARP_SIZE + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;
        const int ky = QR5_K*threadIdx.x;

        const int ql = get_int_b4(bxi->qs, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, threadIdx.x % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (threadIdx.x / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (threadIdx.x / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + QI5_K/4;

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = ql0 | qh0;
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = ql1 | qh1;
#endif // INT8_MMA_AVAILABLE
    }

#ifdef INT8_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*16) {
        int i = (i0 + threadIdx.y*16 + threadIdx.x/(WARP_SIZE/16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*QI5_K) {
        int i = (i0 + threadIdx.y*QI5_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

        x_dm[i] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*8) {
        int i = (i0 + threadIdx.y*8 + threadIdx.x/(WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(WARP_SIZE/8) + i/8 + ksc] = scales8;
    }
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_df + WARP_SIZE/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0;

        const int ql = get_int_b2(bxi->ql, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (threadIdx.x / (QI6_K/2)) + threadIdx.x % (QI6_K/4));
        const int qh0 = ((qh >> ((threadIdx.x & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((threadIdx.x & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*threadIdx.x - threadIdx.x % (QI6_K/2) + 0;
        const int kq1 = 2*threadIdx.x - threadIdx.x % (QI6_K/2) + QI6_K/2;

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K + kbxd] = bxi->d;
#endif // INT8_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0 + (threadIdx.x % (WARP_SIZE/8)) / 4;

#ifdef INT8_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x % (WARP_SIZE/8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K/8));
#else
        x_sc[i*(WARP_SIZE/8) + i/8   + threadIdx.x % (WARP_SIZE/8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K/8));
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * x_sc = (const int   *) x_df + WARP_SIZE/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    int   scA[ntx][mma_C::ne/2][8];
    float  dA[ntx][mma_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            A[n][k01/4 + 0].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + 0),        MMQ_MMA_TILE_X_K_Q6_K);
            A[n][k01/4 + 1].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + mma_A::K), MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

                const int      sc_packed = x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q6_K];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float tmp[ntx][mma_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            mma_B B[2];
            float dB[mma_C::ne/2];

            B[0].load(y_qs + j0*MMQ_TILE_Y_K + 0        + k01, MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + mma_B::K + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma_K4(A[n][k01/4 + 0], B[0]);
                C[1].mma_K4(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_nl(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_NL;
    const int kqsx = threadIdx.x % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *)(x + i*stride) + kbx0 + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_NL;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
        int i = i0 + threadIdx.y * QI4_NL + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *)(x + i*stride) + kbx0 + kbxd;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kbxd] = __half2float(bxi->d);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_mxfp4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP4, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_NL;
    const int kqsx = threadIdx.x % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *)(x + i*stride) + kbx0 + kbx;

        const int aux_q4 = get_int_b1(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_NL;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

    union { float f; uint32_t u; } helper;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
        int i = i0 + threadIdx.y * QI4_NL + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *)(x + i*stride) + kbx0 + kbxd;
        helper.u = bxi->e ? uint32_t(bxi->e) << 23u : 0x00400000;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = 0.5f * helper.f;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kbxd] = 0.5f * helper.f;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XXS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_XXS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_XXS) + threadIdx.x/(QI2_XXS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *)(x + i*stride) + kbx0;

        const uint32_t q2 = (uint32_t)get_int_b2(bxi->qs, 2*kqsx+0);
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const unsigned grid_index = (q2 >> (8u*(unsigned)l)) & 0xffu;
            const int * grid_pos = (const int *) (iq2xxs_grid + grid_index);
            const int signs_packed = ksigns_iq2xs[(aux32 >> (7*l)) & 0x7F];

            const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
            const int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);

            const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
            const int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid1;
#endif // INT8_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls*d + d/2)/4;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = (ls*d + d/2)/4;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_XS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_XS) + threadIdx.x/(QI2_XS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *)(x + i*stride) + kbx0;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint32_t * grid_pos = (const uint32_t *)(iq2xs_grid + (q2[l] & 0x000001FF));
            const uint32_t * signs    = (const uint32_t *)(ksigns64   + (q2[l] >> 9));

            const int grid_l = __vsub4(grid_pos[0] ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos[1] ^ signs[1], signs[1]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // INT8_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_S/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_S/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_S) + threadIdx.x/(QI2_S/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *)(x + i*stride) + kbx0;

        const uint32_t qs_packed =
            (uint32_t)get_int_b2(bxi->qs, kqsx);

        const int qh = bxi->qh[kqsx];

        const uint32_t signs_packed_32 =
            (uint32_t)get_int_b2(bxi->qs, QK_K/32 + kqsx);

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const unsigned byte_shift = 8u * (unsigned)l;
            const unsigned q = (qs_packed >> byte_shift) & 0xffu;
            const unsigned signs =
                (signs_packed_32 >> byte_shift) & 0xffu;
            const unsigned grid_index =
                q | ((unsigned)(qh << (8-2*l)) & 0x300u);
            const uint64_t grid_packed = iq2s_grid[grid_index];
            const int grid_pos0 = (int)(uint32_t)grid_packed;
            const int grid_pos1 = (int)(uint32_t)(grid_packed >> 32);

            const int signs0 = __vcmpne4(((signs & 0x03) << 7) | ((signs & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs & 0x30) << 3) | ((signs & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos0 ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos1 ^ signs1, signs1);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // INT8_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_XXS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI3_XXS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI3_XXS) + threadIdx.x/(QI3_XXS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *)(x + i*stride) + kbx0;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);

            const int * signs = (const int *)(ksigns64 + ((aux32 >> (7*l)) & 0x7F));

            const int grid_l = __vsub4(grid_pos.x ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos.y ^ signs[1], signs[1]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // INT8_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls*d + d/2)/2;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = (ls*d + d/2)/2;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_S/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI3_S/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI3_S) + threadIdx.x/(QI3_S/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *)(x + i*stride) + kbx0;

        const uint64_t qs_packed =
            (uint64_t)(uint32_t)get_int_b2(bxi->qs, 2*kqsx+0) |
            ((uint64_t)(uint32_t)get_int_b2(bxi->qs, 2*kqsx+1) << 32);

        const int qh = bxi->qh[kqsx];

        const uint32_t signs_packed = (uint32_t)get_int_b2(bxi->signs, kqsx);

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const unsigned q0 = (qs_packed >> (8u*(unsigned)(2*l+0))) & 0xffu;
            const unsigned q1 = (qs_packed >> (8u*(unsigned)(2*l+1))) & 0xffu;
            const unsigned signs = (signs_packed >> (8u*(unsigned)l)) & 0xffu;
            const int2 grid_pos = make_int2(
                iq3s_grid[q0 | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[q1 | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs & 0x03) << 7) | ((signs & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs & 0x30) << 3) | ((signs & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+1)] = grid_h;
#endif // INT8_MMA_AVAILABLE
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = ls*d;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = ls*d;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI1_S;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI1_S) {
        int i = i0 + threadIdx.y*(WARP_SIZE/QI1_S) + threadIdx.x/QI1_S;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *)(x + i*stride) + kbx0;

        const uint32_t qs_packed = (uint32_t)get_int_b2(bxi->qs, kqsx);

        const int qh = bxi->qh[kqsx];

    #pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const unsigned q = (qs_packed >> (8u*(unsigned)l)) & 0xffu;
            const int grid = iq1s_grid_gpu[q | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+1)] = grid1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+1)] = grid1;
#endif // INT8_MMA_AVAILABLE
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

#ifdef INT8_MMA_AVAILABLE
        x_ds[i*MMQ_MMA_TILE_X_K_Q8_1 + kqsx] = make_half2(d1q, d1q*delta);
#else
        x_ds[i*(WARP_SIZE/4) + i/4   + kqsx] = make_half2(d1q, d1q*delta);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kbx  = 0;           // threadIdx.x / QI4_XS
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_XS

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *)(x + i*stride) + kbx0 + kbx;

        const int q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *)(x + i*stride) + kbx0;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * (ls - 32);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * (ls - 32);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x%16;

#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

        uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
        uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
        uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
    }

#else // __CUDA_ARCH__

    const int * all_values = (const int *)iq2k_table;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#endif // INT8_MMA_AVAILABLE
    }
#endif // __CUDA_ARCH__

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = dptr[0];
        const block_iq2_ks * bxi = (const block_iq2_ks *)(dptr + 1) + kbx0;
        const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2*(threadIdx.x % 4))) & 0x10);
        const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2*(threadIdx.x % 4))) & 0x10);

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_k * bxi = (const block_iq2_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

#ifdef __CUDA_ARCH__

        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {
            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 2) & 0x44444444);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 0) & 0x44444444);
            int2 v1 = get_int_from_table_8(val1, iq2nl_values);
            int2 v2 = get_int_from_table_8(val2, iq2nl_values);
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
        }

#else

        auto all_values = (const int *)iq2k_table;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#endif // INT8_MMA_AVAILABLE

            extra >>= 8;
        }
#endif // __CUDA_ARCH__

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_k * bxi = (const block_iq3_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;

        uint16_t extra = bxi->extra >> (kqsx/4);
        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        int qh = get_int_b2(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            //extra << 3, extra << 1, extra >> 1, extra >> 3
            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 3) & 0x88888888)
                          | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 1) & 0x88888888)
                          | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            qh >>= 4;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
        }

        uint16_t sh = bxi->scales_h >> 2*kqsx;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = __half2float(dptr[0]);
        const block_iq3_ks * bxi = (const block_iq3_ks *)(dptr + 1) + kbx0;

        //uint16_t extra = bxi->extra >> 8;
        int qh = get_int_b2(bxi->qh, kqsx);

        uint32_t extra32 = uint32_t(bxi->extra >> 8) * 0x01010101;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qh << 2) & 0x04040404) | ((extra32 << 3) & 0x08080808)
                                                     | ((qh << 4) & 0x40404040) | ((extra32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qh << 1) & 0x04040404) | ((extra32 << 2) & 0x08080808)
                                                     | ((qh << 3) & 0x40404040) | ((extra32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            extra32 >>= 4;
            qh      >>= 4;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
        }

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_ks * bxi = (const block_iq4_ks *)(dptr + 1) + kbx0;
        const int ls = (bxi->scales[kqsx] & 254) - 127;

        auto values = iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 4*kqsx+j);
            const int2 v = get_int_from_table_16(q4, values);
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ls;
#endif // INT8_MMA_AVAILABLE
    }

}


template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_ks_r4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_KS_R4, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x/4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }
        int i4 = i/4;
        int ir = i%4;

        const float * dptr = (const float *)(x + 4*i4*stride);
        const block_iq4_ks_r4 * bxi = (const block_iq4_ks_r4 *)(dptr + 4) + kbx0;

        const int ls = (bxi->scales[4*kqsx + ir] & 254) - 127;
        auto values = iq4k_values + ((bxi->scales[4*kqsx+ir] & 1) << 4);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 16*kqsx+4*j+ir);
            const int2 v = get_int_from_table_16(q4, values);
            const int k0 = 8*kqsx + 4*(j%2) + j/2;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 2] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 2] = v.y;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[ir] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[ir] * ls;
#endif // INT8_MMA_AVAILABLE

    }

}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_kt * bxi = (const block_iq4_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto shb = bxi->qs;
        const auto ql  = (const uint8_t *)(shb + 8);
        const auto qh  = ql + 64;
        const uint32_t sh = shb[ib32] >> (8 + 6*j);
        uint32_t offset = 4096 + ((shb[ib32] & 1) << 15);
        uint32_t val1 = offset + ql[8*ib32+2*j+0] + ((qh[8*(ib32%4)+2*j+0] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 7) << 12);
        uint32_t val2 = offset + ql[8*ib32+2*j+1] + ((qh[8*(ib32%4)+2*j+1] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 56) << 9);
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val1 *= ka;
            val2 *= ka;
            v.x |= (ggml_cuda_dp4a(val1 & km, 0x01010101, -126) & 0xff) << 8*k;
            v.y |= (ggml_cuda_dp4a(val2 & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kt * bxi = (const block_iq4_kt *)(dptr + 1) + kbx0;
        const int ls = (bxi->qs[threadIdx.x % 8] & 0xff) >> 1;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = dptr[0] * (ls - 64);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = dptr[0] * (ls - 64);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_kt * bxi = (const block_iq2_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.05f;
        const block_iq2_kt * bxi = (const block_iq2_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = iq4k_values[(bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf];

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_kt * bxi = (const block_iq3_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        const auto qh = (const uint32_t *)bxi->qh;
        uint32_t mask = 0x01010101 << ib32;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        auto signs = __vcmpne4(qh[2*j+0] & mask, 0);
        v.x = __vsub4(v.x ^ signs, signs);
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        signs = __vcmpne4(qh[2*j+1] & mask, 0);
        v.y = __vsub4(v.y ^ signs, signs);
#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.01f;
        const block_iq3_kt * bxi = (const block_iq3_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = (bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq5_ks_r4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ5_KS_R4, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x/4;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }
        int i4 = i/4;
        int ir = i%4;

        const float * dptr = (const float *)(x + 4*i4*stride);
        const block_iq5_ks_r4 * bxi = (const block_iq5_ks_r4 *)(dptr + 4) + kbx0;

        const int ls = (bxi->scales[4*kqsx + ir] & 254) - 127;
        auto values = iq5nl_values + ((bxi->scales[4*kqsx+ir] & 1) << 5);

        int qh = *((const int *)bxi->qh + 4*kqsx + ir);
        const int * ql = (const int *)bxi->qs + 16*kqsx + ir;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            aux32[0] = ((ql[4*j] >> 0) & 0x0f0f0f0f) | ((qh << 4) & 0x10101010);
            aux32[1] = ((ql[4*j] >> 4) & 0x0f0f0f0f) | ((qh << 3) & 0x10101010);
            qh >>= 2;
            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);
            const int k0 = 8*kqsx + 4*(j%2) + j/2;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 2] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 2] = *(const int *)&val1;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[ir] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[ir] * ls;
#endif // INT8_MMA_AVAILABLE

    }

}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_k * bxi = (const block_iq4_k *)(x + i*stride) + kbx0;
        const uint16_t extra = bxi->extra >> 2*kqsx;

        auto values_l = iq4k_table + ((extra & 1) << 8);
        auto values_h = iq4k_table + ((extra & 2) << 7);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int q4 = get_int_b4(bxi->qs, (qstep/2)*kqsx + l);

            aux32[0] = (q4 >> 0) & 0x0f0f0f0f;
            aux32[1] = (q4 >> 4) & 0x0f0f0f0f;

            int val0 = int_from_table_x(aux8+0, values_l);
            int val1 = int_from_table_x(aux8+4, values_h);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + l + 0] = val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + l + 4] = val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + l + 0] = val0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + l + 4] = val1;
#endif // INT8_MMA_AVAILABLE
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ls1;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq5_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq5_k * bxi = (const block_iq5_k *)(x + i*stride) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);
        uint16_t extra = bxi->extra >> (kqsx/4);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) * 0x20202020); // this is very slightly faster
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) * 0x08080808); // then the version below
            //aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) ? 0x20202020 : 0);
            //aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) ? 0x20202020 : 0);
            qh    >>= 2;
            extra >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // INT8_MMA_AVAILABLE
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ls1;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq5_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ5_KS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0];
        const block_iq5_ks * bxi = (const block_iq5_ks *)(dptr + 1) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((bxi->scales[2*l+0] & 1) * 0x20202020);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((bxi->scales[2*l+1] & 1) * 0x20202020);
            qh    >>= 2;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // INT8_MMA_AVAILABLE
        }

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0               + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq6_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq6nl_values;
    int qh[2];

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq6_k * bxi = (const block_iq6_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

        qh[0] = get_int_b4(bxi->qh, kqsx+0);
        qh[1] = get_int_b4(bxi->qh, kqsx+8);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh[l/2] & 0x03030303) << 4) | ((extra & 1) * 0x40404040);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh[l/2] & 0x0c0c0c0c) << 2) | ((extra & 4) * 0x10101010);
            qh[l/2] >>= 4;
            extra   >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // INT8_MMA_AVAILABLE
        }


#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#endif // INT8_MMA_AVAILABLE
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
    const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[j*stride + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
    const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {

    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_C::I);
#ifdef INT8_MMA_AVAILABLE
    static_assert(nwarps*mma_C::I == mmq_y, "nwarps*mma_C::I != mmq_y");
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * mma_C::J + mma_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*mma_C::I + mma_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j*stride + i] = sum[(j0/mma_C::J + n)*mma_C::ne + l];
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_0> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XXS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xxs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_S> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_XXS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_xxs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_S> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ1_S> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq1_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ1_S_R4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq1_s_r4<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_NL> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_nl<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_MXFP4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_mxfp4<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_XS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_xs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_k<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_k<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_k<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ5_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq5_k<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ6_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq6_k<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_ks<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_ks<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_ks<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_KS_R4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_ks_r4<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_kt<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_kt<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_kt<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ5_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq5_ks<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ5_KS_R4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq5_ks_r4<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <ggml_type type, int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ void mul_mat_q_process_tile(
    const char * __restrict__ x, const char * __restrict__ yc, float * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int & ne00, const int & ne01, const int & stride01, const int & ne10, const int & ne11, const int & stride11, const int & ne0,
    const int & it, const int & jt, const int & kb0_start, const int & kb0_stop) {

    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::load_tiles;

    extern __shared__ char data_mul_mat_q[];
    int * tile_y = (int *) data_mul_mat_q;
    int * tile_x = tile_y + GGML_PAD(mmq_x*(WARP_SIZE + WARP_SIZE/QI8_1), nwarps*WARP_SIZE);

#ifdef INT8_MMA_AVAILABLE
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<mmq_x, mmq_y, nwarps, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, nwarps, need_check>;
#endif // INT8_MMA_AVAILABLE

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    const int tile_x_max_i = ne01 - it*mmq_y - 1;
    const int tile_y_max_j = ne11 - jt*mmq_x - 1;

    const int * y = (const int *) yc + jt*(mmq_x*sizeof(block_q8_1_mmq)/sizeof(int));

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x + int64_t(stride01)*it*mmq_y, tile_x, kb0, tile_x_max_i, stride01);

        {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 0*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 1*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, WARP_SIZE);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, dst + jt*mmq_x*ne0 + it*mmq_y, ne0, tile_x_max_i, tile_y_max_j);
    }
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int mmq_x, int nwarps, bool need_check>
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#else
#if __CUDA_ARCH__ >= CC_VOLTA
    __launch_bounds__(WARP_SIZE*nwarps, 1)
#else
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
static __global__ void mul_mat_q(
    const char * __restrict__ x, const char * __restrict__ yc, float * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int ne00, const int ne01, const int stride01, const int ne10, const int ne11, const int stride11, const int ne0) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y = get_mmq_y_device();

    // On AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
#if (defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < CC_VOLTA
    {
        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
                blockIdx.x, blockIdx.y, 0, ne00/qk);
        return;
    }
#endif // (defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < CC_VOLTA

    const     int64_t blocks_per_ne00 = ne00 / qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;

    const int ntx = (ne11 + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (ne01 + mmq_y - 1) / mmq_y; // Number of tiles y

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *blocks_per_ne00*ntx*nty / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*blocks_per_ne00*ntx*nty / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        const int jt =  kbc /    (blocks_per_ne00*nty);                    // j index of current tile.
        const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00; // i index of current tile.

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
             it, jt, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int jt =  kbc /    (blocks_per_ne00*nty);
    const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
        (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
            it, jt, kb0_start, kb0_stop);
}


template <ggml_type type, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q_stream_k_fixup(
    float * __restrict__ dst, const float * __restrict__ tmp_last_tile, const int ne00, const int ne01, const int ne11, const int ne0, const int block_num_mmq) {

    constexpr int     mmq_y           = get_mmq_y_device();
    constexpr int     qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;
    const     int64_t blocks_per_ne00 = ne00 / qk;

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    const int ntx = (ne11 + mmq_x - 1) / mmq_x;
    const int nty = (ne01 + mmq_y - 1) / mmq_y;

    bool any_fixup = false;

    const int bidx_start = ((blockIdx.y*nty + blockIdx.x)     * block_num_mmq)                           / (gridDim.y*gridDim.x);
    const int bidx_stop  = ((blockIdx.y*nty + blockIdx.x + 1) * block_num_mmq + gridDim.y*gridDim.x - 1) / (gridDim.y*gridDim.x);

    int64_t kbc_0;
    int64_t kbc_stop_0 = (int64_t) bidx_start*blocks_per_ne00*ntx*nty / block_num_mmq;

    for (int bidx = bidx_start; bidx < bidx_stop; ++bidx) {
        kbc_0 = kbc_stop_0;
        kbc_stop_0 = (int64_t) (bidx + 1)*blocks_per_ne00*ntx*nty / block_num_mmq;

        const int64_t kbc      = kbc_0      - (kbc_0      % blocks_per_ne00) % blocks_per_iter;
        const int64_t kbc_stop = kbc_stop_0 - (kbc_stop_0 % blocks_per_ne00) % blocks_per_iter;

        // Skip fixup tile if the MMQ CUDA block never wrote anything to it:
        if (kbc == kbc_stop || kbc_stop % blocks_per_ne00 == 0) {
            continue;
        }

        const int jt =  kbc_stop /    (blocks_per_ne00*nty);
        const int it = (kbc_stop - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;

        // Skip fixup tile if it's unrelated to the output tile assigned to this CUDA block:
        if (it != blockIdx.x || jt != blockIdx.y) {
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
            }
        }
    }

    if (!any_fixup) {
        return;
    }

    dst += blockIdx.y*mmq_x*ne0 + blockIdx.x*mmq_y;

    const int i_max = ne01 - blockIdx.x*mmq_y - 1;
    const int j_max = ne11 - blockIdx.y*mmq_x - 1;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[j*ne0 + i] += sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

struct mmq_args {
    const char * x; const char * y; float * dst;
    int64_t ne00; int64_t ne01; int64_t stride01;
    int64_t ne10; int64_t ne11; int64_t stride11;
    int64_t ne0;
};

template<ggml_type type>
static int mmq_get_shmem(const int mmq_x, const int mmq_y, const int cc) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const int shmem_x = int8_mma_available(cc) ? mmq_y*mmq_tile_x_k*sizeof(int) : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const int shmem_y = mmq_x*sizeof(block_q8_1_mmq);
    return shmem_x + GGML_PAD(shmem_y, MMQ_NWARPS*WARP_SIZE*sizeof(int));
}

template <ggml_type type, int mmq_x>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int mmq_y = get_mmq_y_host(cc);

    const dim3 block_dims(WARP_SIZE, MMQ_NWARPS, 1);

    const int shmem = mmq_get_shmem<type>(mmq_x, mmq_y, cc);

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
    static bool shmem_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shmem_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, MMQ_NWARPS, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, MMQ_NWARPS, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        shmem_limit_raised[id] = true;
    }
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))

    const int nty = (args.ne01 + mmq_y - 1) / mmq_y;
    const int ntx = (args.ne11 + mmq_x - 1) / mmq_x;
    const dim3 block_nums_xy_tiling(nty, ntx, 1);

    const bool use_stream_k = cc >= CC_VOLTA && cc < CC_OFFSET_AMD;
    if (!use_stream_k) {
        if (args.ne01 % mmq_y == 0) {
            constexpr bool need_check = false;
            mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        } else {
            constexpr bool need_check = true;
            mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        }
        return;
    }

    const dim3 block_nums_mmq(nsm, 1, 1);

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool, block_nums_mmq.x * mmq_x*mmq_y);

    if (args.ne01 % mmq_y == 0) {
        constexpr bool need_check = false;

        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);

        mul_mat_q_stream_k_fixup<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);

    } else {
        constexpr bool need_check = true;

        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);

        mul_mat_q_stream_k_fixup<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);

    }
}

template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id    = ggml_cuda_get_device();
    const int nsm   = ggml_cuda_info().devices[id].nsm;
    const int cc    = ggml_cuda_info().devices[id].cc;
    const int smpbo = ggml_cuda_info().devices[id].smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;
    const bool use_stream_k = cc >= CC_VOLTA && cc < CC_OFFSET_AMD;

    int mmq_x_best  = 0;
    int nparts_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && nparts_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);

        if (mmq_x % granularity != 0 || mmq_get_shmem<type>(mmq_x, mmq_y, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ne11 + mmq_x - 1) / mmq_x;
        const int nwaves_xy_tiling = ntiles_x*block_num_y;
        const int nparts = use_stream_k ? ntiles_x : nwaves_xy_tiling;

        if (nparts < nparts_best) {
            mmq_x_best  = mmq_x;
            nparts_best = nparts;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<type,   8>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "mmq_x_best=%d\n", mmq_x_best);
            GGML_ABORT("fatal error");
            break;
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_KL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_KS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_KSS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_KS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_KS_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ5_KS_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_KS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_K_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_K_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_K_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ5_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ5_K_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ5_KS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ6_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S_R4);
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_KT);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_KT);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_KT);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_KT);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_mul_mat_q(ggml_backend_cuda_context & ctx, enum ggml_type type, const mmq_args & args);

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11);
