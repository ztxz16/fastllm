#pragma once

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

// IQ2 dot products adapted from vecdotq.cuh (ggml authors / Iwan Kawrakow,
// MIT). Requires the GGML CUDA type and vec-dot helper declarations.
namespace iq2_decode {

// Sign expansion by integer multiplication, as used by Strata ds4_iq.cu.
// Copyright (c) 2026 Niko1221 and the Strata contributors (MIT).
// Each nonzero codebook byte is <= 43, so the per-byte +1 cannot carry.
static __device__ __forceinline__ int PackedSign4(uint32_t grid, uint32_t signs) {
    const uint32_t ones = (((signs & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return int((grid ^ (ones * 255u)) + ones);
}

// IQ2 codebook bytes are in [8, 43]. Packed signed-byte negation can use
// XOR plus one per negative byte without cross-byte carries. XS/XXS sign
// codes reconstruct the eighth bit by parity, eliminating sign-table loads.
// Keep the original integer scaling and truncating divisions unchanged.
template<bool PackedSigns = false>
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
        const int grid0 = PackedSigns ? PackedSign4(grid_pos[0], signs_packed) : ((grid_pos[0] ^ signs0) + ((signs0) & 0x01010101u));
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
        const int grid1 = PackedSigns ? PackedSign4(grid_pos[1], signs_packed >> 4) : ((grid_pos[1] ^ signs1) + ((signs1) & 0x01010101u));
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 28;
    // Same signed truncation: (ls*s + trunc(s/2))/4 == (2*ls+1)*s/8.
    // The bounded 32-value int8 dot cannot overflow this integer product.
    if constexpr (PackedSigns) sumi = ((2*ls+1)*sumi)/8;
    else sumi = (ls*sumi + sumi/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


template<bool PackedSigns = false>
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

        const int grid_l = PackedSigns ? PackedSign4(grid_pos[0], s8) : ((grid_pos[0] ^ signs[0]) + ((signs[0]) & 0x01010101u));
        const int grid_h = PackedSigns ? PackedSign4(grid_pos[1], s8 >> 4) : ((grid_pos[1] ^ signs[1]) + ((signs[1]) & 0x01010101u));

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


template<bool PackedSigns = false>
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

        const int grid_l = PackedSigns ? PackedSign4(grid_pos0, signs) : ((grid_pos0 ^ signs0) + ((signs0) & 0x01010101u));
        const int grid_h = PackedSigns ? PackedSign4(grid_pos1, signs >> 4) : ((grid_pos1 ^ signs1) + ((signs1) & 0x01010101u));

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

} // namespace iq2_decode
