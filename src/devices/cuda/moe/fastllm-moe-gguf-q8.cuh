#pragma once
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
// IQ2 DP4A unpacking is adapted from gguf_mmq/fastllm-gguf-iq2-gemv.cuh.
// Apply the fractional block scale in float, without truncating integer dots.
// This matches dequantized GGUF weights against the Q8_1 activation oracle.
namespace gguf_cache_q8 {
static __device__ __forceinline__ int get_int_b2(const void *p, int i) {
    const auto *v = static_cast<const uint16_t *>(p);
    return int(uint32_t(v[2*i]) | (uint32_t(v[2*i+1]) << 16));
}
static __device__ __forceinline__ int get_int_b4(const void *p, int i) {
    return static_cast<const int *>(p)[i];
}
static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const auto *x = reinterpret_cast<const int8_t *>(&a);
    const auto *y = reinterpret_cast<const int8_t *>(&b);
    return c + x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3];
#endif
}
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
    const float scaled = (ls + 0.5f) * sumi * 0.25f;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * scaled;
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
    const float scaled = ((ls0 + 0.5f) * sumi0 + (ls1 + 0.5f) * sumi1) * 0.25f;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * scaled;
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
    const float scaled = ((ls0 + 0.5f) * sumi0 + (ls1 + 0.5f) * sumi1) * 0.25f;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * scaled;
}


// IQ1_M uses the canonical signed-byte codebook, staged once per CTA.
static __device__ __forceinline__ float DotIQ1M(const void *weight,
        const block_q8_1 *x, int block, int part, const uint64_t *grid) {
    const auto &q = static_cast<const block_iq1_m *>(weight)[block];
    const uint32_t indices = uint32_t(get_int_b4(q.qs, part));
    int sums[2] = {0, 0};
    float offsets[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned high = q.qh[2*part + i/2] >> (4*(i%2));
        const uint64_t values = grid[((indices >> (8*i)) & 255) | ((high & 7) << 8)];
        const int x0 = get_int_b4(x[part].qs, 2*i), x1 = get_int_b4(x[part].qs, 2*i+1);
        sums[i/2] = ggml_cuda_dp4a(int(values), x0, sums[i/2]);
        sums[i/2] = ggml_cuda_dp4a(int(values >> 32), x1, sums[i/2]);
        const int sumX = ggml_cuda_dp4a(0x01010101, x1,
                        ggml_cuda_dp4a(0x01010101, x0, 0));
        offsets[i/2] += (high & 8 ? -IQ1M_DELTA : IQ1M_DELTA) * sumX;
    }
    const auto *sc = reinterpret_cast<const uint16_t *>(q.scales);
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int bits = sc[part/2] >> (6*(part%2));
    const int s0 = 2*(bits & 7)+1, s1 = 2*((bits >> 3) & 7)+1;
    return __half2float(scale.f16) * __low2float(x[part].ds) *
        ((sums[0] + offsets[0])*s0 + (sums[1] + offsets[1])*s1);
}

// Expand eight consecutive Q2_0 codes into two DP4A byte vectors. Both
// expert GEMV and grouped MMQ use this packing without changing the weights.
static __device__ __forceinline__ int2 UnpackQ2(uint16_t codes) {
    const int even = __byte_perm(0x020100ff, 0x020100ff, codes);
    const int odd = __byte_perm(0x020100ff, 0x020100ff, codes >> 2);
    return make_int2(__byte_perm(even, odd, 0x5140),
                     __byte_perm(even, odd, 0x7362));
}

// Q2_0's 64-value block uses {-1, 0, 1, 2}; each lane dots 32 values.
static __device__ __forceinline__ float DotQ2(const void *weight,
        const block_q8_1 *x, int block, int part) {
    const auto &q = static_cast<const block_q2_0 *>(weight)[block];
    const auto *bits = reinterpret_cast<const uint16_t *>(q.qs) + part*4;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int2 values = UnpackQ2(bits[j]);
        sum = ggml_cuda_dp4a(values.x, get_int_b4(x[part].qs, 2*j), sum);
        sum = ggml_cuda_dp4a(values.y, get_int_b4(x[part].qs, 2*j+1), sum);
    }
    return __half2float(q.d) * __low2float(x[part].ds) * sum;
}

static __device__ __forceinline__ float DotIQ3XXS(
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
    const float scaled = (ls + 0.5f) * 0.5f * sumi;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * scaled;
}

static __device__ __forceinline__ float DotIQ3S(
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

// Expand the low/high nibbles through the shared 16-entry IQ4 codebook.
static __device__ __forceinline__ int2 UnpackIQ4(int q4) {
    uint32_t v1, v2, v3, v4, mask;
    const uint32_t * values32 = (const uint32_t *)kvalues_iq4nl;

    mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
    v1 = __byte_perm(values32[0], values32[1], q4);
    v2 = __byte_perm(values32[2], values32[3], q4);
    // Select between the low and high results based on the MSB of each index nibble.
    v3 = __byte_perm(v1, v2, mask);
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    v4 = __byte_perm(v1, v2, mask >> 16);

    return make_int2(__byte_perm(v3, v4, 0x6420), __byte_perm(v3, v4, 0x7531));
}

// One lane consumes a complete 32-value IQ4 sub-block. IQ4_XS stores eight
// such sub-blocks plus signed six-bit scales in each 256-value weight block.
template<ggml_type Type>
static __device__ __forceinline__ float DotIQ4(const void *weight,
        const block_q8_1 *x, int block, int part) {
    const uint8_t *qs;
    float scale;
    if constexpr (Type == GGML_TYPE_IQ4_NL) {
        const auto &q = static_cast<const block_iq4_nl *>(weight)[block];
        qs = q.qs;
        scale = __half2float(q.d);
    } else {
        const auto &q = static_cast<const block_iq4_xs *>(weight)[block];
        qs = q.qs + 16*part;
        const int low = (q.scales_l[part/2] >> (4*(part%2))) & 15;
        const int high = (q.scales_h >> (2*part)) & 3;
        scale = __half2float(q.d) * ((low | (high << 4)) - 32);
    }
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int2 values = UnpackIQ4(get_int_b2(qs, j));
        sum = ggml_cuda_dp4a(values.x, get_int_b4(x[part].qs, j), sum);
        sum = ggml_cuda_dp4a(values.y, get_int_b4(x[part].qs, j+4), sum);
    }
    return scale * __low2float(x[part].ds) * sum;
}

template<ggml_type Type> struct Format {
    static constexpr int block = Type == GGML_TYPE_IQ4_NL ? 32 : Type == GGML_TYPE_Q2_0 ? 64 : 256;
    static constexpr int parts = block/32;
    static constexpr int step = Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_Q2_0 ||
        Type == GGML_TYPE_IQ4_NL || Type == GGML_TYPE_IQ4_XS ? 1 : 2;
    static constexpr int gridSize = Type == GGML_TYPE_IQ1_M ? 2048 :
        Type == GGML_TYPE_IQ2_S ? 1024 : Type == GGML_TYPE_IQ2_XS ? 512 :
        Type == GGML_TYPE_IQ2_XXS ? 256 :
        Type == GGML_TYPE_IQ3_S ? 256 : Type == GGML_TYPE_IQ3_XXS ? 128 : 0;
};

template<ggml_type Type, int ThreadsPerRow = 32>
__device__ __forceinline__ float RowDot(const void *weight, const block_q8_1 *x,
                                      int columns, const uint64_t *grid) {
    static_assert(ThreadsPerRow == 32 || (ThreadsPerRow == 8 && Type == GGML_TYPE_Q2_0));
    using F = Format<Type>;
    float sum = 0;
    for (int k = threadIdx.x % ThreadsPerRow; k < columns/32; k += ThreadsPerRow) {
        const int b = k/F::parts, part = F::step*(k%F::parts);
        const auto *xb = x + b*F::parts;
        if constexpr (Type == GGML_TYPE_IQ2_S) sum += DotS(weight, xb, b, part, grid);
        else if constexpr (Type == GGML_TYPE_IQ2_XS) sum += DotXS(weight, xb, b, part, grid);
        else if constexpr (Type == GGML_TYPE_IQ2_XXS) sum += DotXXS(weight, xb, b, part, grid);
        else if constexpr (Type == GGML_TYPE_IQ1_M) sum += DotIQ1M(weight, xb, b, part, grid);
        else if constexpr (Type == GGML_TYPE_IQ3_XXS)
            sum += DotIQ3XXS(weight, xb, b, part, reinterpret_cast<const uint32_t *>(grid));
        else if constexpr (Type == GGML_TYPE_IQ3_S)
            sum += DotIQ3S(weight, xb, b, part, reinterpret_cast<const uint32_t *>(grid));
        else if constexpr (Type == GGML_TYPE_IQ4_NL || Type == GGML_TYPE_IQ4_XS)
            sum += DotIQ4<Type>(weight, xb, b, part);
        else if constexpr (ThreadsPerRow == 8) {
            // With at most sixteen Q8 blocks this combines k and k+8,
            // exactly the first nonzero stage of the full-warp reduction.
            // Keep the dot rounded before adding: an FMA here would change
            // the original projection's floating-point arithmetic.
            sum = __fadd_rn(sum, DotQ2(weight, xb, b, part));
        } else sum += DotQ2(weight, xb, b, part);
    }
    constexpr unsigned lanes = 0xffffffffu >> (32 - ThreadsPerRow);
    const unsigned mask = lanes << ((threadIdx.x % 32) / ThreadsPerRow * ThreadsPerRow);
#pragma unroll
    for (int offset = ThreadsPerRow / 2; offset; offset >>= 1)
        sum += __shfl_xor_sync(mask, sum, offset, ThreadsPerRow);
    return sum;
}

template<ggml_type Type>
__device__ __forceinline__ void StageGrid(uint64_t *grid) {
    if constexpr (Type == GGML_TYPE_IQ3_XXS || Type == GGML_TYPE_IQ3_S) {
        const uint32_t *source = Type == GGML_TYPE_IQ3_S ? iq3s_grid : iq3xxs_grid;
        auto *destination = reinterpret_cast<uint32_t *>(grid);
        for (int i = threadIdx.x; i < 2*Format<Type>::gridSize; i += blockDim.x)
            destination[i] = source[i];
    } else if constexpr (Format<Type>::gridSize) {
        const uint64_t *source = Type == GGML_TYPE_IQ1_M ? iq1s_grid :
            Type == GGML_TYPE_IQ2_S ? iq2s_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
        for (int i = threadIdx.x; i < Format<Type>::gridSize; i += blockDim.x) grid[i] = source[i];
    }
}
} // namespace gguf_cache_q8
