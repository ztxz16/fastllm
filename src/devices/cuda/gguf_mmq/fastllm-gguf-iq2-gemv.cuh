#pragma once
#include "../fastllm-gguf-iq2-dot.cuh"

namespace iq2_decode {

template <ggml_type Type, int Rows, bool Fused, typename Output, int StoreMode = 0>
__global__ void IQ2SharedGemvKernel(
        const void *__restrict__ weights, const void *__restrict__ upWeights,
        const block_q8_1 *__restrict__ input, Output *__restrict__ output,
        int columns, int outputRows) {
    constexpr int gridSize = Type == GGML_TYPE_IQ2_S ? 1024 :
                             Type == GGML_TYPE_IQ2_XS ? 512 : 256;
    __shared__ uint64_t grid[gridSize];
    extern __shared__ int4 inputCache[];
    const int tid = threadIdx.x;
    const uint64_t *sourceGrid = Type == GGML_TYPE_IQ2_S ? iq2s_grid :
                                 Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
    for (int i = tid; i < gridSize; i += Rows * WARP_SIZE) {
        grid[i] = sourceGrid[i];
    }
    // K is divisible by 256 and Q8_1 input is 16-byte aligned, so these
    // vector copies are complete and naturally aligned.
    for (int i = tid; i < columns / QK8_1 * int(sizeof(block_q8_1)) / 16;
         i += Rows * WARP_SIZE) {
        inputCache[i] = reinterpret_cast<const int4 *>(input)[i];
    }
    __syncthreads();
    const auto *x = reinterpret_cast<const block_q8_1 *>(inputCache);
    const int row = blockIdx.x * Rows + tid / WARP_SIZE;
    if (row >= outputRows) return; // warp-uniform, after the barrier
    const int lane = tid % WARP_SIZE;
    const int blocks = columns / QK_K;
    const int quantIndex = 2 * (lane % 8);
    // Preserve the established four-warp floating-point accumulation order.
    // Four register accumulators represent the old warps; their K-block
    // strides and final left-to-right combination remain identical.
    float partial[4] = {0.0f}, upPartial[4] = {0.0f};
    for (int block = lane / 8; block < blocks; block += 16) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int weightColumnBlock = block + 4 * part;
            if (weightColumnBlock >= blocks) continue;
            const int weightBlock = row * blocks + weightColumnBlock;
            if constexpr (Type == GGML_TYPE_IQ2_S) {
                partial[part] += DotS(weights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
                if constexpr (Fused) upPartial[part] += DotS(upWeights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
            } else if constexpr (Type == GGML_TYPE_IQ2_XS) {
                partial[part] += DotXS(weights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
                if constexpr (Fused) upPartial[part] += DotXS(upWeights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
            } else {
                partial[part] += DotXXS(weights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
                if constexpr (Fused) upPartial[part] += DotXXS(upWeights, x + weightColumnBlock * 8, weightBlock, quantIndex, grid);
            }
        }
    }
    float sum = partial[0], upSum = upPartial[0];
#pragma unroll
    for (int part = 1; part < 4; ++part) {
        sum += partial[part];
        if constexpr (Fused) upSum += upPartial[part];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (Fused) upSum = warp_reduce_sum(upSum);
    if (lane == 0) {
        if constexpr (Fused) {
            const half gate = __float2half_rn(sum), up = __float2half_rn(upSum);
            const half activated = __hdiv(gate, __hadd(__float2half(1.0f), hexp(-gate)));
            output[row] = __hmul(activated, up);
        } else {
            FastllmGgufStore<StoreMode>(output + row, sum);
        }
    }
}

static bool Supports(const void *input, int columns, int rows) {
    // Portable CUDA operations, with no SM whitelist. The K cap bounds
    // shared Q8_1 input plus the largest IQ2 codebook to 28,928 bytes/block.
    return rows >= 128 && columns >= 256 && columns <= 18432 && columns % 256 == 0 &&
           (reinterpret_cast<uintptr_t>(input) & 15) == 0;
}

template <ggml_type Type, int Rows, bool Fused, typename Output, int StoreMode = 0>
static void LaunchRows(const void *weights, const void *upWeights, const block_q8_1 *input,
                       Output *output, int columns, int rows, cudaStream_t stream) {
    const size_t sharedBytes = columns / QK8_1 * sizeof(block_q8_1);
    IQ2SharedGemvKernel<Type, Rows, Fused, Output, StoreMode><<<
        (rows + Rows - 1) / Rows, Rows * WARP_SIZE, sharedBytes, stream>>>(
            weights, upWeights, input, output, columns, rows);
}

template <ggml_type Type, bool Fused, typename Output, int StoreMode = 0>
static void Launch(const void *weights, const void *upWeights, const block_q8_1 *input,
                   Output *output, int columns, int rows, cudaStream_t stream) {
    // Sixteen rows amortize input/codebook staging. Fused XS keeps eight
    // rows to balance its larger register footprint against resident blocks.
    constexpr int rowsPerBlock = Fused && Type == GGML_TYPE_IQ2_XS ? 8 : 16;
    LaunchRows<Type, rowsPerBlock, Fused, Output, StoreMode>(weights, upWeights, input, output,
                                        columns, rows, stream);
}

} // namespace iq2_decode
