#pragma once

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

// IQ2 dot products adapted from vecdotq.cuh (ggml authors / Iwan Kawrakow,
// MIT). Included inside fastllm_gguf_mmq, after its vec-dot helpers.
// Cache the codebook and Q8_1 input once per block to reduce global-load
// instruction pressure. Each warp computes one output row.
namespace iq2_decode {

// IQ2 codebook bytes are in [8, 43]. Packed signed-byte negation can use
// XOR plus one per negative byte without cross-byte carries. XS/XXS sign
// codes reconstruct the eighth bit by parity, eliminating sign-table loads.
// Keep the original integer scaling and truncating divisions unchanged.
static __device__ __forceinline__ float DotXXS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const uint32_t q2 = (uint32_t)get_int_b2(bq2->qs, iqs);
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const unsigned grid_index = (q2 >> (8u*(unsigned)(k0/2))) & 0xffu;
        const int * grid_pos = (const int *) (grid + grid_index);
        const unsigned s7 = (aux32 >> (7*k0/2)) & 0x7F;
        const int signs_packed = s7 | ((__popc(s7) & 1) << 7);

        const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
        const int grid0 = ((grid_pos[0] ^ signs0) + ((signs0) & 0x01010101u));
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
        const int grid1 = ((grid_pos[1] ^ signs1) + ((signs1) & 0x01010101u));
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


static __device__ __forceinline__ float DotXS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t * grid_pos = (const uint32_t *)(grid + (q2[l0/2] & 0x000001FF));
        const unsigned s7 = q2[l0/2] >> 9;
        const unsigned s8 = (s7 | ((__popc(s7) & 1) << 7)) * 0x01010101u;
        const uint32_t signs[2] = {__vcmpne4(s8 & 0x08040201, 0), __vcmpne4(s8 & 0x80402010, 0)};

        const int grid_l = ((grid_pos[0] ^ signs[0]) + ((signs[0]) & 0x01010101u));
        const int grid_h = ((grid_pos[1] ^ signs[1]) + ((signs[1]) & 0x01010101u));

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


static __device__ __forceinline__ float DotS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const uint32_t qs_packed = (uint32_t)get_int_b2(bq2->qs, iqs/2);

    const int qh = bq2->qh[iqs/2];

    const uint32_t signs_packed_32 =
        (uint32_t)get_int_b2(bq2->qs, QK_K/32 + iqs/2);

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const unsigned byte_shift = 8u * (unsigned)(l0 / 2);
        const unsigned q = (qs_packed >> byte_shift) & 0xffu;
        const unsigned signs = (signs_packed_32 >> byte_shift) & 0xffu;
        const unsigned grid_index =
            q | ((unsigned)(qh << (8-l0)) & 0x300u);
        const uint64_t grid_packed = grid[grid_index];
        const int grid_pos0 = (int)(uint32_t)grid_packed;
        const int grid_pos1 = (int)(uint32_t)(grid_packed >> 32);

        const int signs0 = __vcmpne4(((signs & 0x03) << 7) | ((signs & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs & 0x30) << 3) | ((signs & 0xC0) << 17), 0x00000000);

        const int grid_l = ((grid_pos0 ^ signs0) + ((signs0) & 0x01010101u));
        const int grid_h = ((grid_pos1 ^ signs1) + ((signs1) & 0x01010101u));

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

template <ggml_type Type, int Rows, bool Fused, typename Output>
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
            output[row] = static_cast<Output>(sum);
        }
    }
}

static bool Supports(const void *input, int columns, int rows) {
    // Portable CUDA operations, with no SM whitelist. The K cap bounds
    // shared Q8_1 input plus the largest IQ2 codebook to 28,928 bytes/block.
    return rows >= 128 && columns >= 256 && columns <= 18432 && columns % 256 == 0 &&
           (reinterpret_cast<uintptr_t>(input) & 15) == 0;
}

template <ggml_type Type, int Rows, bool Fused, typename Output>
static void LaunchRows(const void *weights, const void *upWeights, const block_q8_1 *input,
                       Output *output, int columns, int rows, cudaStream_t stream) {
    const size_t sharedBytes = columns / QK8_1 * sizeof(block_q8_1);
    IQ2SharedGemvKernel<Type, Rows, Fused, Output><<<
        (rows + Rows - 1) / Rows, Rows * WARP_SIZE, sharedBytes, stream>>>(
            weights, upWeights, input, output, columns, rows);
}

template <ggml_type Type, bool Fused, typename Output>
static void Launch(const void *weights, const void *upWeights, const block_q8_1 *input,
                   Output *output, int columns, int rows, cudaStream_t stream) {
    // Sixteen rows amortize input/codebook staging. Fused XS keeps eight
    // rows to balance its larger register footprint against resident blocks.
    constexpr int rowsPerBlock = Fused && Type == GGML_TYPE_IQ2_XS ? 8 : 16;
    LaunchRows<Type, rowsPerBlock, Fused>(weights, upWeights, input, output,
                                        columns, rows, stream);
}

} // namespace iq2_decode
