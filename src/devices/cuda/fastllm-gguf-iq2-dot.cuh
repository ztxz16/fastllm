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
template<bool PackedSigns = false, bool Packed = false>
static __device__ __forceinline__ float DotXXS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid, int row = 0) {

    uint32_t q2, aux32;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_xxs_r4 *>(vbq)[kbx];
        q2 = get_int_b4(q.qs, 4*(iqs/2) + row);
        aux32 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = q.sas[16*(iqs/2) + 4*row + j], signs = v >> 1;
            aux32 |= ((signs ^ (signs << 1)) & 127) << (7*j);
            aux32 |= (v & 1) << (28+j);
        }
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_xxs *>(vbq)[kbx];
        q2 = get_int_b2(q.qs, iqs);
        aux32 = get_int_b2(q.qs, iqs + 1);
        scale = __half2float(q.d);
    }

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
    const float d = scale * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


template<bool PackedSigns = false, bool Packed = false>
static __device__ __forceinline__ float DotXS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid, int row = 0) {

    uint16_t q2[4];
    int ls0, ls1;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_xs_r4 *>(vbq)[kbx];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = q.qs[16*(iqs/2) + 4*row + j], signs = v >> 9;
            q2[j] = (v & 511) | (((signs ^ (signs << 1)) & 127) << 9);
        }
        ls0 = q.scales[4*(iqs/2) + row] & 15;
        ls1 = q.scales[4*(iqs/2) + row] >> 4;
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_xs *>(vbq)[kbx];
#pragma unroll
        for (int j = 0; j < 4; ++j) q2[j] = q.qs[2*iqs + j];
        ls0 = q.scales[iqs/2] & 15;
        ls1 = q.scales[iqs/2] >> 4;
        scale = __half2float(q.d);
    }

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
    const float d = scale * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


template<bool PackedSigns = false, bool Packed = false>
static __device__ __forceinline__ float DotS(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const uint64_t *grid, int row = 0) {

    uint32_t qs_packed, signs_packed_32;
    int qh, ls0, ls1;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_s_r4 *>(vbq)[kbx];
        qs_packed = get_int_b4(q.qs, 4*(iqs/2) + row);
        signs_packed_32 = get_int_b4(q.signs, 4*(iqs/2) + row);
        qh = q.qh[4*(iqs/2) + row];
        ls0 = q.scales[4*(iqs/2) + row] & 15;
        ls1 = q.scales[4*(iqs/2) + row] >> 4;
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_s *>(vbq)[kbx];
        qs_packed = get_int_b2(q.qs, iqs/2);
        signs_packed_32 = get_int_b2(q.qs, QK_K/32 + iqs/2);
        qh = q.qh[iqs/2];
        ls0 = q.scales[iqs/2] & 15;
        ls1 = q.scales[iqs/2] >> 4;
        scale = __half2float(q.d);
    }

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

    const float d = scale * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}


// Decode each 32-weight subgroup once, then reuse its signed integer bytes
// for several activation rows. Scaling keeps the scalar dot's truncation.
struct Decoded32 {
    int values[8], lo, hi;
    float scale;
    template<bool XXS>
    __device__ __forceinline__ float Dot(const block_q8_1 &x) const {
        int a = 0, b = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) a = ggml_cuda_dp4a(values[i], get_int_b4(x.qs, i), a);
#pragma unroll
        for (int i = 4; i < 8; ++i) b = ggml_cuda_dp4a(values[i], get_int_b4(x.qs, i), b);
        const int sum = XXS ? ((2*lo+1)*(a+b))/8 : (a*lo+b*hi+(a+b)/2)/4;
        return (scale * __low2float(x.ds)) * sum;
    }
};
template<bool Packed>
static __device__ __forceinline__ Decoded32 DecodeXXS(
        const void *vbq, int kbx, int iqs, const uint64_t *grid, int row = 0) {
    uint32_t q2, aux32;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_xxs_r4 *>(vbq)[kbx];
        q2 = get_int_b4(q.qs, 4*(iqs/2) + row);
        aux32 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = q.sas[16*(iqs/2) + 4*row + j], signs = v >> 1;
            aux32 |= ((signs ^ (signs << 1)) & 127) << (7*j);
            aux32 |= (v & 1) << (28+j);
        }
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_xxs *>(vbq)[kbx];
        q2 = get_int_b2(q.qs, iqs);
        aux32 = get_int_b2(q.qs, iqs + 1);
        scale = __half2float(q.d);
    }

    Decoded32 decoded;
    decoded.scale = scale;
    decoded.lo = aux32 >> 28; decoded.hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint64_t value = grid[(q2 >> (8*j)) & 255u];
        const unsigned s7 = (aux32 >> (7*j)) & 127u;
        const unsigned signs = s7 | ((__popc(s7) & 1) << 7);
        decoded.values[2*j] = PackedSign4(uint32_t(value), signs);
        decoded.values[2*j+1] = PackedSign4(uint32_t(value >> 32), signs >> 4);
    }
    return decoded;
}
template<bool Packed>
static __device__ __forceinline__ Decoded32 DecodeXS(
        const void *vbq, int kbx, int iqs, const uint64_t *grid, int row = 0) {
    uint16_t q2[4];
    int ls0, ls1;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_xs_r4 *>(vbq)[kbx];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = q.qs[16*(iqs/2) + 4*row + j], signs = v >> 9;
            q2[j] = (v & 511) | (((signs ^ (signs << 1)) & 127) << 9);
        }
        ls0 = q.scales[4*(iqs/2) + row] & 15;
        ls1 = q.scales[4*(iqs/2) + row] >> 4;
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_xs *>(vbq)[kbx];
#pragma unroll
        for (int j = 0; j < 4; ++j) q2[j] = q.qs[2*iqs + j];
        ls0 = q.scales[iqs/2] & 15;
        ls1 = q.scales[iqs/2] >> 4;
        scale = __half2float(q.d);
    }

    Decoded32 decoded;
    decoded.scale = scale;
    decoded.lo = ls0; decoded.hi = ls1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint64_t value = grid[q2[j] & 511u];
        const unsigned s7 = q2[j] >> 9;
        const unsigned signs = s7 | ((__popc(s7) & 1) << 7);
        decoded.values[2*j] = PackedSign4(uint32_t(value), signs);
        decoded.values[2*j+1] = PackedSign4(uint32_t(value >> 32), signs >> 4);
    }
    return decoded;
}
template<bool Packed>
static __device__ __forceinline__ Decoded32 DecodeS(
        const void *vbq, int kbx, int iqs, const uint64_t *grid, int row = 0) {
    uint32_t qs_packed, signs_packed_32;
    int qh, ls0, ls1;
    float scale;
    if constexpr (Packed) {
        const auto &q = static_cast<const block_iq2_s_r4 *>(vbq)[kbx];
        qs_packed = get_int_b4(q.qs, 4*(iqs/2) + row);
        signs_packed_32 = get_int_b4(q.signs, 4*(iqs/2) + row);
        qh = q.qh[4*(iqs/2) + row];
        ls0 = q.scales[4*(iqs/2) + row] & 15;
        ls1 = q.scales[4*(iqs/2) + row] >> 4;
        scale = __half2float(q.d[row]);
    } else {
        const auto &q = static_cast<const block_iq2_s *>(vbq)[kbx];
        qs_packed = get_int_b2(q.qs, iqs/2);
        signs_packed_32 = get_int_b2(q.qs, QK_K/32 + iqs/2);
        qh = q.qh[iqs/2];
        ls0 = q.scales[iqs/2] & 15;
        ls1 = q.scales[iqs/2] >> 4;
        scale = __half2float(q.d);
    }

    Decoded32 decoded;
    decoded.scale = scale;
    decoded.lo = ls0; decoded.hi = ls1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned index = ((qs_packed >> (8*j)) & 255u) | ((unsigned(qh) << (8-2*j)) & 768u);
        const uint64_t value = grid[index];
        const unsigned signs = (signs_packed_32 >> (8*j)) & 255u;
        decoded.values[2*j] = PackedSign4(uint32_t(value), signs);
        decoded.values[2*j+1] = PackedSign4(uint32_t(value >> 32), signs >> 4);
    }
    return decoded;
}

} // namespace iq2_decode
