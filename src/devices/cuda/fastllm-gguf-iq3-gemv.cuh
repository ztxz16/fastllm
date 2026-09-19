#pragma once

// Single-token IQ3 MMVQ for CUDA. Each warp computes one output row.
// Cache both the small codebook and Q8_1 activations per CUDA block: the
// legacy path saturates the L1/global-load instruction queue before DRAM.
// Included after the GGUF vec-dot helpers in fastllm-ggml-cuda.cu.
namespace fastllm_gguf_iq3 {

// IQ3 codebook bytes are strictly positive. Therefore packed signed-byte
// negation can use XOR plus one per negative byte without cross-byte carries.
// IQ3_XXS reconstructs the eighth sign by parity, as in llama.cpp, instead
// of loading the ksigns64 table for every eight weights.
static __device__ __forceinline__ float DotXXS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint32_t *grid) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);

        const uint32_t s7 = (aux32 >> (7*l0/2)) & 0x7F;
        const uint32_t s8 = (s7 | ((__popc(s7) & 1) << 7)) * 0x01010101u;
        const uint32_t signs[2] = {__vcmpne4(s8 & 0x08040201, 0), __vcmpne4(s8 & 0x80402010, 0)};

        const int grid_l = ((grid_pos.x ^ signs[0]) + (signs[0] & 0x01010101u));
        const int grid_h = ((grid_pos.y ^ signs[1]) + (signs[1] & 0x01010101u));

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float DotS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint32_t *grid) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = ((grid_pos.x ^ signs0) + (signs0 & 0x01010101u));
        const int grid_h = ((grid_pos.y ^ signs1) + (signs1 & 0x01010101u));

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


template <ggml_type Type, int Rows, bool Fused, typename Output>
__global__ void SharedGemvKernel(
        const void *__restrict__ weights, const void *__restrict__ upWeights,
        const block_q8_1 *__restrict__ input, Output *__restrict__ output,
        int columns, int outputRows) {
    constexpr int gridSize = Type == GGML_TYPE_IQ3_S ? 512 : 256;
    __shared__ uint32_t grid[gridSize];
    extern __shared__ int4 inputCache[];
    const int tid = threadIdx.x;
    const uint32_t *sourceGrid = Type == GGML_TYPE_IQ3_S ? iq3s_grid : iq3xxs_grid;
    for (int i = tid; i < gridSize; i += Rows * WARP_SIZE) {
        grid[i] = sourceGrid[i];
    }
    // K is a multiple of 256 and input is 16-byte aligned (checked by the
    // launcher), so all copied vectors are complete and naturally aligned.
    for (int i = tid; i < columns / QK8_1 * int(sizeof(block_q8_1)) / 16;
         i += Rows * WARP_SIZE) {
        inputCache[i] = reinterpret_cast<const int4 *>(input)[i];
    }
    __syncthreads();
    const auto *x = reinterpret_cast<const block_q8_1 *>(inputCache);
    const int row = blockIdx.x * Rows + tid / WARP_SIZE;
    if (row >= outputRows) return; // uniform within each warp, after the barrier
    const int lane = tid % WARP_SIZE;
    const int blocks = columns / QK_K;
    const int quantIndex = 2 * (lane % 8);
    float sum = 0.0f, upSum = 0.0f;
    for (int block = lane / 8; block < blocks; block += 4) {
        const int weightBlock = row * blocks + block;
        if constexpr (Type == GGML_TYPE_IQ3_S) {
            sum += DotS(weights, x + block * 8, weightBlock, quantIndex, grid);
            if constexpr (Fused) upSum += DotS(upWeights, x + block * 8, weightBlock, quantIndex, grid);
        } else {
            sum += DotXXS(weights, x + block * 8, weightBlock, quantIndex, grid);
            if constexpr (Fused) upSum += DotXXS(upWeights, x + block * 8, weightBlock, quantIndex, grid);
        }
    }
    sum = warp_reduce_sum(sum);
    if constexpr (Fused) upSum = warp_reduce_sum(upSum);
    if (lane == 0) {
        if constexpr (Fused) {
            output[row] = FastllmGgufHalfSiluMulValue((half)sum, (half)upSum);
        } else {
            output[row] = (Output)sum;
        }
    }
}

static bool Supports(const void *input, int columns, int rows) {
    // These kernels use portable CUDA operations, with no SM-specific dispatch.
    // The K cap bounds Q8_1 input plus codebook storage to 22,784 bytes/block.
    // Retain the established kernels for tiny projections and non-aligned views.
    return rows >= 128 && columns >= 256 && columns <= 18432 && columns % 256 == 0 &&
           (reinterpret_cast<uintptr_t>(input) & 15) == 0;
}

template <ggml_type Type, bool Fused, typename Output>
static void Launch(const void *weights, const void *upWeights, const block_q8_1 *input,
                   Output *output, int columns, int rows, cudaStream_t stream) {
    constexpr int rowsPerBlock = Fused ? 16 : 8;
    const size_t sharedBytes = columns / QK8_1 * sizeof(block_q8_1);
    SharedGemvKernel<Type, rowsPerBlock, Fused, Output><<<
        (rows + rowsPerBlock - 1) / rowsPerBlock, rowsPerBlock * WARP_SIZE, sharedBytes, stream>>>(
            weights, upWeights, input, output, columns, rows);
}

} // namespace fastllm_gguf_iq3
