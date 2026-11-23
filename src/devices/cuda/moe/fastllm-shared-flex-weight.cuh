#pragma once
#include <cuda.h>
#include <cute/arch/mma_sm100_umma.hpp>

// TMA unpacks sixteen E2M1 nibbles into sixteen bytes of shared memory.
// UMMA consumes the expanded tile, but the TMA barrier counts global bytes.
namespace fastllm_shared_flex {
__device__ __forceinline__ auto weight_instruction(bool mxfp4) {
    using FP8 = cutlass::float_e4m3_t;
    using FP4 = cutlass::detail::float_e2m1_unpacksmem_t;
    using SF = cutlass::float_ue8m0_t;
    constexpr auto major = cute::UMMA::Major::K;
    return mxfp4 ? cute::UMMA::make_instr_desc_block_scaled<FP4, FP8, float, SF, 128, 8, major, major>()
                 : cute::UMMA::make_instr_desc_block_scaled<FP8, FP8, float, SF, 128, 8, major, major>();
}
__device__ __forceinline__ int weight_tile_bytes(bool mxfp4) {
    return mxfp4 ? 128 * 64 : 128 * 128;
}

inline CUresult weight_map(CUtensorMap *map, const void *weights, int k, int n,
                           int experts, bool mxfp4) {
    const cuuint64_t rowBytes = mxfp4 ? k / 2 : k;
    cuuint64_t dims[] = {cuuint64_t(k), cuuint64_t(n), cuuint64_t(experts)};
    cuuint64_t strides[] = {rowBytes, n * rowBytes};
    cuuint32_t box[] = {128, 128, 1}, steps[] = {1, 1, 1};
    return cuTensorMapEncodeTiled(map, mxfp4 ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B :
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void *>(weights), dims, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
}
