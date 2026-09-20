#pragma once

// Quantized dot arithmetic adapted from ggml / Iwan Kawrakow (MIT).
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// SPDX-License-Identifier: MIT

// Shared small-batch MMVQ for Q2_K, IQ1_M, IQ2, IQ3, IQ4_XS and Q4_K. Included after
// the GGUF vec-dot primitives; single-token kernels remain separate.
namespace fastllm_gguf_small_mmvq {

// Packed kvalues_iq4nl. Immediate operands avoid repeated table loads;
// the regression checks all sixteen entries against the GGUF definition.
static constexpr uint32_t iq4Values[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
static __device__ __forceinline__ int2 LookupIQ4(uint32_t q) {
    const uint32_t mask = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t lo = __byte_perm(__byte_perm(iq4Values[0], iq4Values[1], q),
                                    __byte_perm(iq4Values[2], iq4Values[3], q), mask);
    const uint32_t hi = __byte_perm(__byte_perm(iq4Values[0], iq4Values[1], q >> 16),
                                    __byte_perm(iq4Values[2], iq4Values[3], q >> 16), mask >> 16);
    return make_int2(__byte_perm(lo, hi, 0x6420), __byte_perm(lo, hi, 0x7531));
}

// Decode one lane's 32 weights once, then reuse the packed int8 values for
// every input token. Keep the format-specific integer scaling in the dot
// product: in particular IQ3_XXS rounds before applying the floating scale.
template <ggml_type Type>
static __device__ __forceinline__ void DecodeBatchWeights(const void *weights, int block, int iqs,
                                                          const void *codebook, int (&values)[8], float &d,
                                                          int &scale) {
    const auto *grid = static_cast<const uint32_t *>(codebook);
    if constexpr (Type == GGML_TYPE_IQ1_M) {
        const auto *w = static_cast<const block_iq1_m *>(weights) + block;
        const auto *iq1Grid = static_cast<const uint64_t *>(codebook);
        const int group = iqs / 2;
        const uint32_t indices = get_int_b4(w->qs, group);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int high = w->qh[2 * group + j / 2] >> (4 * (j % 2));
            const uint64_t q = iq1Grid[((indices >> (8 * j)) & 255) | ((high & 7) << 8)];
            uint32_t lo = uint32_t(q), hi = uint32_t(q >> 32);
            // Canonical {-1,0,1} codebook plus +/-1/8 offset. Multiplying
            // by eight packs exact signed integers into bytes for DP4A.
#pragma unroll
            for (int bit = 0; bit < 3; ++bit) {
                lo = __vadd4(lo, lo);
                hi = __vadd4(hi, hi);
            }
            const uint32_t delta = (high & 8) ? 0xffffffffu : 0x01010101u;
            values[2 * j] = __vadd4(lo, delta);
            values[2 * j + 1] = __vadd4(hi, delta);
        }
        const auto *sc = reinterpret_cast<const uint16_t *>(w->scales);
        iq1m_scale_t base;
        base.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
        d = __half2float(base.f16);
        const int factors = sc[group / 2] >> (6 * (group % 2));
        scale = (2 * (factors & 7) + 1) | ((2 * ((factors >> 3) & 7) + 1) << 4);
    } else if constexpr (Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_XXS) {
        const auto *iq2Grid = static_cast<const uint64_t *>(codebook);
        uint32_t indices, signsPacked;
        int2 indicesXS;
        int high = 0;
        if constexpr (Type == GGML_TYPE_IQ2_S) {
            const auto *w = static_cast<const block_iq2_s *>(weights) + block;
            indices = get_int_b2(w->qs, iqs / 2);
            signsPacked = get_int_b2(w->qs, QK_K / 32 + iqs / 2);
            high = w->qh[iqs / 2];
            scale = w->scales[iqs / 2];
            d = __half2float(w->d);
        } else if constexpr (Type == GGML_TYPE_IQ2_XS) {
            const auto *w = static_cast<const block_iq2_xs *>(weights) + block;
            indicesXS = make_int2(get_int_b2(w->qs, iqs), get_int_b2(w->qs, iqs + 1));
            scale = w->scales[iqs / 2];
            d = __half2float(w->d);
        } else {
            const auto *w = static_cast<const block_iq2_xxs *>(weights) + block;
            indices = get_int_b2(w->qs, iqs);
            signsPacked = get_int_b2(w->qs, iqs + 1);
            scale = signsPacked >> 28;
            d = __half2float(w->d);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t index, signs;
            if constexpr (Type == GGML_TYPE_IQ2_S) {
                index = ((indices >> (8 * j)) & 255) | ((high << (8 - 2 * j)) & 0x300);
                signs = (signsPacked >> (8 * j)) & 255;
            } else {
                uint32_t s7;
                if constexpr (Type == GGML_TYPE_IQ2_XS) {
                    const uint16_t q = reinterpret_cast<const uint16_t *>(&indicesXS)[j];
                    index = q & 511;
                    s7 = q >> 9;
                } else {
                    index = (indices >> (8 * j)) & 255;
                    s7 = (signsPacked >> (7 * j)) & 127;
                }
                signs = s7 | ((__popc(s7) & 1) << 7);
            }
            const uint64_t q = iq2Grid[index];
            const uint32_t repeated = signs * 0x01010101u;
            const uint32_t s0 = __vcmpne4(repeated & 0x08040201, 0);
            const uint32_t s1 = __vcmpne4(repeated & 0x80402010, 0);
            values[2 * j] = (uint32_t(q) ^ s0) + (s0 & 0x01010101u);
            values[2 * j + 1] = (uint32_t(q >> 32) ^ s1) + (s1 & 0x01010101u);
        }
    } else if constexpr (Type == GGML_TYPE_IQ3_S) {
        const auto *w = static_cast<const block_iq3_s *>(weights) + block;
        const int2 qsPacked = make_int2(get_int_b2(w->qs, iqs), get_int_b2(w->qs, iqs + 1));
        const auto *qs = reinterpret_cast<const uint8_t *>(&qsPacked);
        const int qh = w->qh[iqs / 2];
        const int signsPacked = get_int_b2(w->signs, iqs / 2);
        const auto *signs = reinterpret_cast<const uint8_t *>(&signsPacked);
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const uint32_t lo = grid[qs[j] | ((qh << (8 - j)) & 0x100)];
            const uint32_t hi = grid[qs[j + 1] | ((qh << (7 - j)) & 0x100)];
            const uint32_t s0 = __vcmpne4(((signs[j / 2] & 0x03) << 7) | ((signs[j / 2] & 0x0c) << 21), 0);
            const uint32_t s1 = __vcmpne4(((signs[j / 2] & 0x30) << 3) | ((signs[j / 2] & 0xc0) << 17), 0);
            values[j] = (lo ^ s0) + (s0 & 0x01010101u);
            values[j + 1] = (hi ^ s1) + (s1 & 0x01010101u);
        }
        d = __half2float(w->d);
        scale = 1 + 2 * ((w->scales[iqs / 4] >> ((iqs << 1) & 4)) & 15);
    } else if constexpr (Type == GGML_TYPE_IQ3_XXS) {
        const auto *w = static_cast<const block_iq3_xxs *>(weights) + block;
        const int2 qsPacked = make_int2(get_int_b2(w->qs, iqs), get_int_b2(w->qs, iqs + 1));
        const auto *qs = reinterpret_cast<const uint8_t *>(&qsPacked);
        const uint32_t aux = get_int_b2(w->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const uint32_t s7 = (aux >> (7 * j / 2)) & 127;
            const uint32_t s8 = (s7 | ((__popc(s7) & 1) << 7)) * 0x01010101u;
            const uint32_t s0 = __vcmpne4(s8 & 0x08040201, 0);
            const uint32_t s1 = __vcmpne4(s8 & 0x80402010, 0);
            values[j] = (grid[qs[j]] ^ s0) + (s0 & 0x01010101u);
            values[j + 1] = (grid[qs[j + 1]] ^ s1) + (s1 & 0x01010101u);
        }
        d = __half2float(w->d);
        scale = aux >> 28;
    } else {
        static_assert(Type == GGML_TYPE_IQ4_XS, "unsupported small MMVQ format");
        const auto *w = static_cast<const block_iq4_xs *>(weights) + block;
        const int offset = 2 * iqs;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 q = LookupIQ4(get_int_b4(w->qs, offset + j));
            values[j] = q.x;
            values[j + 4] = q.y;
        }
        const int low = (w->scales_l[offset / 8] >> (offset & 4)) & 15;
        const int high = (w->scales_h >> (offset / 2)) & 3;
        scale = (low | (high << 4)) - 32;
        d = __half2float(w->d);
    }
}

// All batch kernels cache a fixed-width input tile. Strides here are in
// Q8_1 blocks; the public launch accepts strides in logical elements.
template <int Tokens, int Warps, int Tile>
static __device__ __forceinline__ void LoadInputTile(block_q8_1 (&cache)[Tokens][Tile / QK8_1],
                                                     const block_q8_1 *input, int inputStride, int first,
                                                     int count) {
    static_assert(Tile % QK_K == 0, "input tiles must contain whole quantization blocks");
    constexpr int tileVectors = Tile / QK8_1 * sizeof(block_q8_1) / sizeof(int4);
    const int validVectors = count / QK8_1 * sizeof(block_q8_1) / sizeof(int4);
    for (int i = threadIdx.x; i < Tokens * tileVectors; i += Warps * WARP_SIZE) {
        const int token = i / tileVectors, offset = i % tileVectors;
        if (offset < validVectors) {
            reinterpret_cast<int4 *>(cache[token])[offset] =
                reinterpret_cast<const int4 *>(input + token * inputStride + first / QK8_1)[offset];
        }
    }
}

template <ggml_type Type>
static constexpr int CodebookWords = Type == GGML_TYPE_IQ1_M     ? 4096
                                     : Type == GGML_TYPE_IQ2_S   ? 2048
                                     : Type == GGML_TYPE_IQ2_XS  ? 1024
                                     : Type == GGML_TYPE_IQ2_XXS ? 512
                                     : Type == GGML_TYPE_IQ3_S   ? 512
                                     : Type == GGML_TYPE_IQ3_XXS ? 256
                                                                 : 1;

template <ggml_type Type, int Warps> static __device__ __forceinline__ void LoadCodebook(uint32_t *grid) {
    if constexpr (Type != GGML_TYPE_IQ4_XS) {
        const uint32_t *source;
        if constexpr (Type == GGML_TYPE_IQ1_M) {
            source = reinterpret_cast<const uint32_t *>(iq1s_grid);
        } else if constexpr (Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS ||
                             Type == GGML_TYPE_IQ2_XXS) {
            source = reinterpret_cast<const uint32_t *>(Type == GGML_TYPE_IQ2_S    ? iq2s_grid
                                                        : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid
                                                                                   : iq2xxs_grid);
        } else {
            static_assert(Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS, "unsupported codebook");
            source = Type == GGML_TYPE_IQ3_S ? iq3s_grid : iq3xxs_grid;
        }
        for (int i = threadIdx.x; i < CodebookWords<Type>; i += Warps * WARP_SIZE)
            grid[i] = source[i];
    }
}

// One lane owns an entire Q8_1 group. Cache the exact integer input sum
// once per tile, so Q4_K's minimum correction needs no repeated DP4A sum.
template <int Tokens, int Rows, int Tile, typename Output>
__global__ void Q4KBatchSharedGemvKernel(const void *__restrict__ weights,
                                         const block_q8_1 *__restrict__ input, Output *__restrict__ output,
                                         int columns, int outputRows, int inputStride, int outputStride) {
    constexpr int tileBlocks = Tile / QK8_1;
    __shared__ __align__(16) block_q8_1 cache[Tokens][tileBlocks];
    __shared__ int inputSums[Tokens][tileBlocks];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE;
    const int row = blockIdx.x * Rows + tid / WARP_SIZE, group = lane % 8;
    float sums[Tokens] = {};
    for (int first = 0; first < columns; first += Tile) {
        const int count = min(Tile, columns - first);
        const int validBlocks = count / QK8_1;
        LoadInputTile<Tokens, Rows, Tile>(cache, input, inputStride, first, count);
        __syncthreads();
        for (int i = tid; i < Tokens * validBlocks; i += Rows * WARP_SIZE) {
            const int token = i / validBlocks, b = i % validBlocks;
            int sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum = ggml_cuda_dp4a(0x01010101, get_int_b4(cache[token][b].qs, j), sum);
            inputSums[token][b] = sum;
        }
        __syncthreads();
        if (row < outputRows) {
            for (int b = lane / 8; b < count / QK_K; b += 4) {
                const auto *w =
                    static_cast<const block_q4_K *>(weights) + row * (columns / QK_K) + first / QK_K + b;
                int values[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    values[j] = (get_int_b4(w->qs, (group / 2) * 8 + j) >> (4 * (group % 2))) & 0x0f0f0f0f;
                const int sc = group < 4 ? (w->scales[group] & 63)
                                         : ((w->scales[group + 4] & 15) | ((w->scales[group - 4] >> 6) << 4));
                const int mn = group < 4 ? (w->scales[group + 4] & 63)
                                         : ((w->scales[group + 4] >> 4) | ((w->scales[group] >> 6) << 4));
                const float2 dm = __half22float2(w->dm);
#pragma unroll
                for (int token = 0; token < Tokens; ++token) {
                    const auto &x = cache[token][b * 8 + group];
                    int dot = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        dot = ggml_cuda_dp4a(values[j], get_int_b4(x.qs, j), dot);
                    const float d = __low2float(x.ds);
                    sums[token] +=
                        dm.x * (d * (dot * sc)) - dm.y * (d * (inputSums[token][b * 8 + group] * mn));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(sums[token]);
        if (lane == 0 && row < outputRows) output[token * outputStride + row] = (Output)sum;
    }
}

// Q2_K has two independent scale/min pairs per 32-value Q8_1 group.
// Cache the exact sums of the two quantized 16-value halves; ds.y is the
// rounded sum before input quantization and cannot be used here.
template <int Tokens, int Rows, int Tile, typename Output>
__global__ void Q2KBatchSharedGemvKernel(const void *__restrict__ weights,
                                         const block_q8_1 *__restrict__ input, Output *__restrict__ output,
                                         int columns, int outputRows, int inputStride, int outputStride) {
    constexpr int tileBlocks = Tile / QK8_1;
    __shared__ __align__(16) block_q8_1 cache[Tokens][tileBlocks];
    __shared__ int2 inputSums[Tokens][tileBlocks];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE;
    const int row = blockIdx.x * Rows + tid / WARP_SIZE;
    const int group = lane % 8;
    float sums[Tokens] = {};
    for (int first = 0; first < columns; first += Tile) {
        const int count = min(Tile, columns - first);
        const int validBlocks = count / QK8_1;
        LoadInputTile<Tokens, Rows, Tile>(cache, input, inputStride, first, count);
        __syncthreads();
        for (int i = tid; i < Tokens * validBlocks; i += Rows * WARP_SIZE) {
            const int token = i / validBlocks, b = i % validBlocks;
            int lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo = ggml_cuda_dp4a(0x01010101, get_int_b4(cache[token][b].qs, j), lo);
                hi = ggml_cuda_dp4a(0x01010101, get_int_b4(cache[token][b].qs, j + 4), hi);
            }
            inputSums[token][b] = make_int2(lo, hi);
        }
        __syncthreads();
        if (row < outputRows) {
            for (int b = lane / 8; b < count / QK_K; b += 4) {
                const auto *w =
                    static_cast<const block_q2_K *>(weights) + row * (columns / QK_K) + first / QK_K + b;
                int packed[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    packed[j] = (get_int_b4(w->qs, (group / 4) * 8 + j) >> (2 * (group % 4))) & 0x03030303;
                }
                const int sc0 = w->scales[2 * group], sc1 = w->scales[2 * group + 1];
                const float2 dm = __half22float2(w->dm);
#pragma unroll
                for (int token = 0; token < Tokens; ++token) {
                    const auto &x = cache[token][b * 8 + group];
                    int lo = 0, hi = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        lo = ggml_cuda_dp4a(packed[j], get_int_b4(x.qs, j), lo);
                        hi = ggml_cuda_dp4a(packed[j + 4], get_int_b4(x.qs, j + 4), hi);
                    }
                    const int2 sy = inputSums[token][b * 8 + group];
                    const float d = __low2float(x.ds);
                    sums[token] += dm.x * (d * (lo * (sc0 & 15) + hi * (sc1 & 15))) -
                                   dm.y * (d * (sy.x * (sc0 >> 4) + sy.y * (sc1 >> 4)));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(sums[token]);
        if (lane == 0 && row < outputRows) output[token * outputStride + row] = (Output)sum;
    }
}

template <ggml_type Type, int Tokens, int Rows, int Tile, typename Output>
__global__ void BatchSharedGemvKernel(const void *__restrict__ weights, const block_q8_1 *__restrict__ input,
                                      Output *__restrict__ output, int columns, int outputRows,
                                      int inputStride, int outputStride) {
    constexpr int tileBlocks = Tile / QK8_1;
    __shared__ __align__(8) uint32_t grid[CodebookWords<Type>];
    __shared__ __align__(16) block_q8_1 inputCache[Tokens][tileBlocks];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE;
    const int row = blockIdx.x * Rows + tid / WARP_SIZE;
    LoadCodebook<Type, Rows>(grid);
    float sums[Tokens] = {};
    for (int first = 0; first < columns; first += Tile) {
        const int count = min(Tile, columns - first);
        LoadInputTile<Tokens, Rows, Tile>(inputCache, input, inputStride, first, count);
        __syncthreads();
        if (row < outputRows) {
            for (int b = lane / 8; b < count / QK_K; b += 4) {
                int packed[8], scale;
                float d;
                DecodeBatchWeights<Type>(weights, row * (columns / QK_K) + first / QK_K + b, 2 * (lane % 8),
                                         grid, packed, d, scale);
#pragma unroll
                for (int token = 0; token < Tokens; ++token) {
                    const block_q8_1 &x = inputCache[token][b * 8 + lane % 8];
                    int dot = 0;
                    if constexpr (Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_IQ2_S ||
                                  Type == GGML_TYPE_IQ2_XS) {
                        int lo = 0, hi = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            lo = ggml_cuda_dp4a(packed[j], get_int_b4(x.qs, j), lo);
                            hi = ggml_cuda_dp4a(packed[j + 4], get_int_b4(x.qs, j + 4), hi);
                        }
                        if constexpr (Type == GGML_TYPE_IQ1_M)
                            dot = lo * (scale & 15) + hi * (scale >> 4);
                        else
                            dot = (lo * (scale & 15) + hi * (scale >> 4) + (lo + hi) / 2) / 4;
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            dot = ggml_cuda_dp4a(packed[j], get_int_b4(x.qs, j), dot);
                        if constexpr (Type == GGML_TYPE_IQ3_XXS)
                            dot = (scale * dot + dot / 2) / 2;
                        else if constexpr (Type == GGML_TYPE_IQ2_XXS)
                            dot = (scale * dot + dot / 2) / 4;
                        else
                            dot *= scale;
                    }
                    if constexpr (Type == GGML_TYPE_IQ1_M)
                        sums[token] += (d * __low2float(x.ds)) * (dot * 0.125f);
                    else
                        sums[token] += (d * __low2float(x.ds)) * dot;
                }
            }
        }
        // Incomplete output tiles must participate in both barriers.
        __syncthreads();
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(sums[token]);
        if (lane == 0 && row < outputRows) output[token * outputStride + row] = (Output)sum;
    }
}

// Each warp computes two output rows, sharing every Q8 input load across
// both dot products. Preserve the original per-row K accumulation order.
template <ggml_type Type, int Tokens, int Warps, int Tile, typename Output>
__global__ __launch_bounds__(Warps * WARP_SIZE, Warps <= 8 ? 4 : 2) void IQ3PairSharedGemvKernel(
    const void *__restrict__ weights, const block_q8_1 *__restrict__ input, Output *__restrict__ output,
    int columns, int outputRows, int inputStride, int outputStride) {
    static_assert(Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS);
    constexpr int tileBlocks = Tile / QK8_1;
    __shared__ uint32_t grid[CodebookWords<Type>];
    __shared__ __align__(16) block_q8_1 cache[Tokens][tileBlocks];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE;
    const int row0 = blockIdx.x * (2 * Warps) + tid / WARP_SIZE;
    LoadCodebook<Type, Warps>(grid);
    float sums0[Tokens] = {}, sums1[Tokens] = {};
    for (int first = 0; first < columns; first += Tile) {
        const int count = min(Tile, columns - first);
        LoadInputTile<Tokens, Warps, Tile>(cache, input, inputStride, first, count);
        __syncthreads();
        for (int b = lane / 8; b < count / QK_K; b += 4) {
            int w0[8] = {}, w1[8] = {}, sc0 = 0, sc1 = 0;
            float d0 = 0, d1 = 0;
            if (row0 < outputRows)
                DecodeBatchWeights<Type>(weights, row0 * (columns / QK_K) + first / QK_K + b, 2 * (lane % 8),
                                         grid, w0, d0, sc0);
            if (row0 + Warps < outputRows)
                DecodeBatchWeights<Type>(weights, (row0 + Warps) * (columns / QK_K) + first / QK_K + b,
                                         2 * (lane % 8), grid, w1, d1, sc1);
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                const auto &x = cache[t][b * 8 + lane % 8];
                int dot0 = 0, dot1 = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int v = get_int_b4(x.qs, j);
                    dot0 = ggml_cuda_dp4a(w0[j], v, dot0);
                    dot1 = ggml_cuda_dp4a(w1[j], v, dot1);
                }
                if constexpr (Type == GGML_TYPE_IQ3_XXS) {
                    dot0 = (sc0 * dot0 + dot0 / 2) / 2;
                    dot1 = (sc1 * dot1 + dot1 / 2) / 2;
                } else {
                    dot0 *= sc0;
                    dot1 *= sc1;
                }
                const float dx = __low2float(x.ds);
                sums0[t] += (d0 * dx) * dot0;
                sums1[t] += (d1 * dx) * dot1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        const float sum0 = warp_reduce_sum(sums0[t]), sum1 = warp_reduce_sum(sums1[t]);
        if (lane == 0) {
            if (row0 < outputRows) output[t * outputStride + row0] = (Output)sum0;
            if (row0 + Warps < outputRows) output[t * outputStride + row0 + Warps] = (Output)sum1;
        }
    }
}

static bool Supports(const void *input, int columns, int rows, int inputStride, int outputStride) {
    // Shared memory depends on the fixed tile, not the full matrix width.
    // The row threshold is a performance choice; other shapes use legacy MMVQ.
    return rows >= 128 && columns >= QK_K && columns % QK_K == 0 && inputStride >= columns &&
           inputStride % QK_K == 0 && outputStride >= rows && (reinterpret_cast<uintptr_t>(input) & 15) == 0;
}

template <ggml_type Type, int Tokens, int Rows, typename Output>
static void LaunchRows(const void *weights, const block_q8_1 *input, Output *output, int columns, int rows,
                       int inputStride, int outputStride, cudaStream_t stream) {
    constexpr int tile = 1024;
    const int blocks = (rows + Rows - 1) / Rows;
    if constexpr (Type == GGML_TYPE_Q2_K) {
        Q2KBatchSharedGemvKernel<Tokens, Rows, tile, Output><<<blocks, Rows * WARP_SIZE, 0, stream>>>(
            weights, input, output, columns, rows, inputStride / QK8_1, outputStride);
    } else if constexpr (Type == GGML_TYPE_Q4_K) {
        Q4KBatchSharedGemvKernel<Tokens, Rows, tile, Output><<<blocks, Rows * WARP_SIZE, 0, stream>>>(
            weights, input, output, columns, rows, inputStride / QK8_1, outputStride);
    } else {
        BatchSharedGemvKernel<Type, Tokens, Rows, tile, Output><<<blocks, Rows * WARP_SIZE, 0, stream>>>(
            weights, input, output, columns, rows, inputStride / QK8_1, outputStride);
    }
}

template <ggml_type Type, int Tokens, typename Output>
static void LaunchBatchTokens(const void *weights, const block_q8_1 *input, Output *output, int columns,
                              int rows, int inputStride, int outputStride, cudaStream_t stream) {
    // Larger batches amortize input copies across more output rows; smaller
    // matrices retain more CTAs. These are scheduling choices, not SM guards.
    if constexpr (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS) {
        if (rows >= 4096) {
            constexpr int warps = 8, outputRows = 2 * warps;
            IQ3PairSharedGemvKernel<Type, Tokens, warps, 1024, Output>
                <<<(rows + outputRows - 1) / outputRows, warps * WARP_SIZE, 0, stream>>>(
                    weights, input, output, columns, rows, inputStride / QK8_1, outputStride);
            return;
        }
    } else {
        constexpr int largeBatch = Type == GGML_TYPE_Q4_K ? 4 : 5;
        if (Tokens >= largeBatch && rows >= 4096) {
            LaunchRows<Type, Tokens, 16>(weights, input, output, columns, rows, inputStride, outputStride,
                                         stream);
            return;
        }
    }
    LaunchRows<Type, Tokens, 8>(weights, input, output, columns, rows, inputStride, outputStride, stream);
}

template <ggml_type Type, typename Output>
static void LaunchBatch(const void *weights, const block_q8_1 *input, Output *output, int columns, int rows,
                        int tokens, int inputStride, int outputStride, cudaStream_t stream) {
#define FASTLLM_SMALL_MMVQ_CASE(N)                                                                           \
    case N:                                                                                                  \
        LaunchBatchTokens<Type, N>(weights, input, output, columns, rows, inputStride, outputStride,         \
                                   stream);                                                                  \
        break
    switch (tokens) {
        FASTLLM_SMALL_MMVQ_CASE(2);
        FASTLLM_SMALL_MMVQ_CASE(3);
        FASTLLM_SMALL_MMVQ_CASE(4);
        FASTLLM_SMALL_MMVQ_CASE(5);
        FASTLLM_SMALL_MMVQ_CASE(6);
        FASTLLM_SMALL_MMVQ_CASE(7);
        FASTLLM_SMALL_MMVQ_CASE(8);
    }
#undef FASTLLM_SMALL_MMVQ_CASE
}

} // namespace fastllm_gguf_small_mmvq
