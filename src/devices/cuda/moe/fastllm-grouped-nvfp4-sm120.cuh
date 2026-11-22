// Expert-sorted, batch-invariant SM120 projections over FastLLM's native NVFP4 layout.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm::cuda::routed_fp4 {
struct Block16;
struct Weights;
struct GroupedWeightMaps { CUtensorMap codes, scales; };
bool makeGroupedWeightMaps(GroupedWeightMaps &maps, const Weights &weights, int experts, int n, int k);
struct GroupedScratch { size_t metadata, gemm; };
struct GroupedBuffers {
    uint2 *codes;
    uint8_t *scales;
    float *accumulators;
    void *metadata, *gemm;
};

// Expert rows are compact in the code matrix. Block scales use CUTLASS
// 128-row atoms, with a separate aligned start for each expert.
struct GroupedInputRow {
    uint2 *codes;
    uint8_t *scales;
    __device__ GroupedInputRow(const GroupedBuffers &buffers, int start, int expert, int rank, int groups) {
        codes = buffers.codes + size_t(start + rank) * groups;
        const size_t atom = (size_t(start) + 127 * size_t(expert)) / 128 + rank / 128;
        scales = buffers.scales + atom * 128 * groups + (rank % 32 * 4 + rank % 128 / 32) * 4;
    }
    __device__ uint8_t &scale(int group) const { return scales[size_t(group / 4) * 512 + group % 4]; }
};

__device__ inline float scaledGroupedValue(float dot, float weightScale, float rowScale) {
    return float(__float2bfloat16_rn(dot * weightScale * rowScale));
}

inline bool useGroupedVerify(int rows, int topK, int experts, int hidden, int intermediate) {
    // A complete 128-channel stripe is read per CTA. At low route density
    // the small-row kernel costs less than sorting and scattering the batch.
    return int64_t(rows) * topK >= 2LL * experts && hidden % 128 == 0 && intermediate % 64 == 0;
}
inline size_t groupedScaleRows(int routes, int experts) {
    return (size_t(routes) + size_t(127) * experts + 127) / 128 * 128;
}
GroupedScratch groupedScratch(int experts, int smCount);
cudaError_t planGroupedRows(const int *ids, int *offsets, int *ranks, int experts, int routes, cudaStream_t stream);
cudaError_t packGroupedInput(const Block16 *input, const int *ids, const int *offsets, const int *ranks,
                             const GroupedBuffers &buffers, int routes, int k, int topK, cudaStream_t stream);
// Writes FP32 dots in expert order. The consumer applies the two outer scales
// and rounds each projection to BF16 before activation or route summation.
cudaError_t projectGroupedRows(const Weights &weights, const GroupedWeightMaps &maps, int *offsets,
                               const GroupedBuffers &buffers, int n, int k, int experts, int smCount, cudaStream_t stream);
cudaError_t sumGroupedRows(const float *sorted, float *output, const int *ids, const int *offsets, const int *ranks,
                           const float *weightScales, const float *rowScales, int rows, int n, int topK, cudaStream_t stream);
}
