#include "gguf.h"
#include <assert.h>
#include <array>
#include <vector>

#ifdef __aarch64__
// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// #define HAVE_FANCY_SIMD

inline uint8_t scrambled_sign(uint8_t s) {
    static const uint8_t k_table[128] = {
        0x00, 0x7f, 0x7e, 0x01, 0x7c, 0x03, 0x02, 0x7d, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7b, 0x7a, 0x05,
        0x70, 0x0f, 0x0e, 0x71, 0x0c, 0x73, 0x72, 0x0d, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0b, 0x0a, 0x75,
        0x60, 0x1f, 0x1e, 0x61, 0x1c, 0x63, 0x62, 0x1d, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1b, 0x1a, 0x65,
        0x10, 0x6f, 0x6e, 0x11, 0x6c, 0x13, 0x12, 0x6d, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6b, 0x6a, 0x15,
        0x40, 0x3f, 0x3e, 0x41, 0x3c, 0x43, 0x42, 0x3d, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3b, 0x3a, 0x45,
        0x30, 0x4f, 0x4e, 0x31, 0x4c, 0x33, 0x32, 0x4d, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4b, 0x4a, 0x35,
        0x20, 0x5f, 0x5e, 0x21, 0x5c, 0x23, 0x22, 0x5d, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5b, 0x5a, 0x25,
        0x50, 0x2f, 0x2e, 0x51, 0x2c, 0x53, 0x52, 0x2d, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2b, 0x2a, 0x55,
    };
    return k_table[s];
}

static void repack_iq2_xxs(int nrows, int n_per_row, const block_iq2_xxs * x, block_iq2_xxs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xxs * x4[4];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    std::memcpy(aux32, x4[k][ibl].qs + 4*ib, 2*sizeof(uint32_t));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = aux8[i];
                    }
                    uint8_t scale = aux32[1] >> 28;
                    uint8_t s1 = (scrambled_sign((aux32[1] >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32[1] >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32[1] >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32[1] >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32[1] = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32[1];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_iq2_xs(int nrows, int n_per_row, const block_iq2_xs * x, block_iq2_xs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xs * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        uint16_t v = x4[k][ibl].qs[4*ib+i];
                        uint8_t s = v >> 9;
                        y[ibl].qs[16*ib+4*k+i] = (v & 511) | (scrambled_sign(s) << 9);
                    }
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_iq2_s(int nrows, int n_per_row, const block_iq2_s * x, block_iq2_s_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_s * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto signs = x4[k][ibl].qs + QK_K/8;
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = x4[k][ibl].qs[4*ib+i];
                        y[ibl].signs[16*ib+4*k+i] = signs[4*ib+i];
                    }
                    y[ibl].qh[4*ib+k] = x4[k][ibl].qh[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_iq3_xxs(int nrows, int n_per_row, const block_iq3_xxs * x, block_iq3_xxs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_xxs * x4[4];
    uint32_t aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto xsas = x4[k][ibl].qs + QK_K/4;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 8; ++i) {
                        y[ibl].qs[32*ib+8*k+i] = x4[k][ibl].qs[8*ib+i];
                    }
                    std::memcpy(&aux32, xsas + 4*ib, 4);
                    uint8_t scale = aux32 >> 28;
                    uint8_t s1 = (scrambled_sign((aux32 >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32 >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32 >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32 >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32 = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32;
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

inline void convert_q2_k(const block_q2_K& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    for (int n = 0; n < QK_K; n += 128) {
        for (int j = 0; j < 32; ++j) {
            L[n + j +  0] = (qs[j] >> 0) & 0x3;
            L[n + j + 32] = (qs[j] >> 2) & 0x3;
            L[n + j + 64] = (qs[j] >> 4) & 0x3;
            L[n + j + 96] = (qs[j] >> 6) & 0x3;
        }
        qs += 32;
    }
}

static void repack_q2_k(int nrows, int n_per_row, const block_q2_K * x, block_q2_k_r4 * y, [[maybe_unused]] bool online) {
    // printf("into repack_q2_k %d %d\n", nrows, n_per_row);
    // while (1);
    assert(nrows % 4 == 0);
    assert(n_per_row % QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q2_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                for (int ib = 0; ib < QK_K/16; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
                convert_q2_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

inline void convert_q3_k(const block_q3_K& x, uint8_t * L, uint8_t * Ld) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, x.scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(Ld, aux, 16);

    const uint8_t * q = x.qs;
    const uint8_t * hm = x.hmask;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                *L++ = ((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0);
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static void repack_q3_k(int nrows, int n_per_row, const block_q3_K * x, block_q3_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q3_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/16];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q3_k(x4[k][ibl], L, Ld);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    int is = 8*ib+k;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+0] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+0] >>  4) << 2*(is/16);
                    is += 4;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+1] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+1] >>  4) << 2*(is/16);
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
inline void convert_q4_k(const block_q4_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = x.qs[32*ib64+j] & 0xf;
            L[64*ib64+j+32] = x.qs[32*ib64+j] >>  4;
        }
    }
}

static void repack_q4_k(int nrows, int n_per_row, const block_q4_K * x, block_q4_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q4_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q4_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = L[32*ib+i+ 0] | (L[32*ib+i+ 8] << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = L[32*ib+i+16] | (L[32*ib+i+24] << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = L[32*ib+i+ 4] | (L[32*ib+i+12] << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = L[32*ib+i+20] | (L[32*ib+i+28] << 4);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

inline void convert_q6_k(const block_q6_K& x, uint8_t * L) {
    const uint8_t * ql = x.ql;
    const uint8_t * qh = x.qh;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            L[n + l +  0] = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            L[n + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            L[n + l + 64] = (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4);
            L[n + l + 96] = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
        ql += 64;
        qh += 32;
    }
}

inline void convert_q5_k(const block_q5_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = (x.qs[32*ib64+j] & 0xf) | (((x.qh[j] >> (2*ib64+0)) & 1) << 4);
            L[64*ib64+j+32] = (x.qs[32*ib64+j] >>  4) | (((x.qh[j] >> (2*ib64+1)) & 1) << 4);
        }
    }
}

static void repack_q5_k(int nrows, int n_per_row, const block_q5_K * x, block_q5_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q5_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q5_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+ 4] >> 4) << 2) | ((L[32*ib+i+12] >> 4) << 3) |
                                                    ((L[32*ib+i+16] >> 4) << 4) | ((L[32*ib+i+24] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}


static void repack_q6_k(int nrows, int n_per_row, const block_q6_K * x, block_q6_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q6_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q6_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[8*ib+k+0] = x4[k][ibl].scales[2*ib+0];
                    y[ibl].scales[8*ib+k+4] = x4[k][ibl].scales[2*ib+1];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].ql[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[32*ib+4*k+i+ 0] = (L[32*ib+i+ 0] >> 4) | ((L[32*ib+i+ 8] >> 4) << 2) | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 6);
                        y[ibl].qh[32*ib+4*k+i+16] = (L[32*ib+i+16] >> 4) | ((L[32*ib+i+24] >> 4) << 2) | ((L[32*ib+i+20] >> 4) << 4) | ((L[32*ib+i+28] >> 4) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

const Repack * get_repack_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Repack> k_map = {
        // { GGML_TYPE_IQ2_K,  { GGML_TYPE_IQ2_K_R4,  4,  (Repack::repack_func)repack_iq2_k}   },
        // { GGML_TYPE_IQ3_K,  { GGML_TYPE_IQ3_K_R4,  4,  (Repack::repack_func)repack_iq3_k}   },
        // { GGML_TYPE_IQ4_K,  { GGML_TYPE_IQ4_K_R4,  4,  (Repack::repack_func)repack_iq4_k}   },
        // { GGML_TYPE_IQ5_K,  { GGML_TYPE_IQ5_K_R4,  4,  (Repack::repack_func)repack_iq5_k}   },
        // { GGML_TYPE_IQ4_XS, { GGML_TYPE_IQ4_XS_R8, 8,  (Repack::repack_func)repack_iq4_xs}  },
        // { GGML_TYPE_IQ4_KS, { GGML_TYPE_IQ4_KS_R4, 4,  (Repack::repack_func)repack_iq4_ks}  },
        // { GGML_TYPE_IQ5_KS, { GGML_TYPE_IQ5_KS_R4, 4,  (Repack::repack_func)repack_iq5_ks}  },
        // { GGML_TYPE_IQ4_NL, { GGML_TYPE_IQ4_NL_R4, 4,  (Repack::repack_func)repack_iq4_nl}  },
        // { GGML_TYPE_IQ2_BN, { GGML_TYPE_IQ2_BN_R4, 4,  (Repack::repack_func)repack_iq2_bn}  },
        { GGML_TYPE_IQ2_XXS,{ GGML_TYPE_IQ2_XXS_R4,4,  (Repack::repack_func)repack_iq2_xxs} },
        { GGML_TYPE_IQ2_XS, { GGML_TYPE_IQ2_XS_R4, 4,  (Repack::repack_func)repack_iq2_xs}  },
        { GGML_TYPE_IQ2_S,  { GGML_TYPE_IQ2_S_R4,  4,  (Repack::repack_func)repack_iq2_s}   },
        { GGML_TYPE_IQ3_XXS,{ GGML_TYPE_IQ3_XXS_R4,4,  (Repack::repack_func)repack_iq3_xxs} },
        // { GGML_TYPE_IQ3_S,  { GGML_TYPE_IQ3_S_R4,  4,  (Repack::repack_func)repack_iq3_s}   },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
        { GGML_TYPE_Q6_K,   { GGML_TYPE_Q6_K_R4,   4,  (Repack::repack_func)repack_q6_k}    },
        // { GGML_TYPE_Q4_0,   { GGML_TYPE_Q4_0_R8,   8,  (Repack::repack_func)repack_q4_0}    },
        // { GGML_TYPE_Q5_0,   { GGML_TYPE_Q5_0_R4,   4,  (Repack::repack_func)repack_q5_0}    },
        // { GGML_TYPE_Q6_0,   { GGML_TYPE_Q6_0_R4,   4,  (Repack::repack_func)repack_q6_0}    },
        // { GGML_TYPE_Q8_0,   { GGML_TYPE_Q8_0_R8,   8,  (Repack::repack_func)repack_q8_0}    },
        // { GGML_TYPE_Q8_K,   { GGML_TYPE_Q8_K_R8,   8,  (Repack::repack_func)repack_q8_k}    },
        // { GGML_TYPE_Q8_KV,  { GGML_TYPE_Q8_KV_R8,  8,  (Repack::repack_func)repack_q8_KV}   },
#ifdef __AVX512BF16__
        // { GGML_TYPE_BF16,   { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_bf16_t>}},
        // { GGML_TYPE_F16,    { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_half>}  },
#endif
    };
    auto it = k_map.find(type);
    return it != k_map.end() ? &it->second : nullptr;
}

template <int nrc_y>
static void mul_mat_iq2_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_iq2_xs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_iq2_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_iq3_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
void mul_mat_q2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_q3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_q4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_q5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

template <int nrc_y>
static void mul_mat_q6_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
}

static void mul_mat_empty(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    return;
}

#define RETURN_MATMUL_FUNCTION(FUNC, X) \
    if ((X) == 1) return FUNC <1>; \
    if ((X) == 2) return FUNC <2>; \
    if ((X) == 3) return FUNC <3>; \
    if ((X) == 4) return FUNC <4>; \
    if ((X) == 5) return FUNC <5>; \
    if ((X) == 6) return FUNC <6>; \
    if ((X) == 7) return FUNC <7>; \
    if ((X) == 8) return FUNC <8>; \
    return nullptr;

mul_mat_t GetMulMatFunction(ggml_type type, int nrc_y) {
    if (type == GGML_TYPE_IQ2_XXS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_xxs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ2_XS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_xs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ2_S_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_s_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ3_XXS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq3_xxs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q2_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q2_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q3_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q3_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q4_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q4_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q5_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q5_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q6_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q6_k_r4_q8_k, nrc_y)
    } else {
        return nullptr;
    }
}
#elif defined(__AVX2__) && ((defined(__FMA__) && defined(__F16C__)) || defined(_MSC_VER))
// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// #define HAVE_FANCY_SIMD

// Keep IQ2_S in its ordinary GGUF layout on AVX2 and AVX512 builds. Four 10-bit codebook
// indices fit in one scalar register; expanding their high bits through a
// shared 2 KiB table avoids four separate shift/mask/OR sequences. This is
// an index table, not an expanded or repacked copy of the model weights.
static inline __m256i iq2_s_native_values(const uint8_t *qs, uint8_t qh) {
    static constexpr auto high = [] {
        std::array<uint64_t, 256> table{};
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 4; ++j)
                table[i] |= uint64_t((i >> (2 * j)) & 3) << (16 * j + 8);
        return table;
    }();
    uint32_t low;
    std::memcpy(&low, qs, sizeof(low));
    const uint64_t indices = uint64_t(_mm_cvtsi128_si64(
        _mm_cvtepu8_epi16(_mm_cvtsi32_si128(low)))) | high[qh];
    // Each 16-bit lane is already bounded by 1023. Word extraction avoids
    // repeating a 10-bit mask at every codebook load.
    return _mm256_setr_epi64x(
        iq2s_grid[uint16_t(indices)], iq2s_grid[uint16_t(indices >> 16)],
        iq2s_grid[uint16_t(indices >> 32)], iq2s_grid[uint16_t(indices >> 48)]);
}

// IQ2_S and IQ3_S both store one sign bit per value, in groups of 32.
static inline __m256i iq_s_native_signs(const uint8_t *signs) {
    uint32_t bits;
    std::memcpy(&bits, signs, sizeof(bits));
    const auto mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    const auto expanded = _mm256_shuffle_epi8(_mm256_set1_epi32(bits),
        _mm256_setr_epi64x(0, 0x0101010101010101LL,
                          0x0202020202020202LL, 0x0303030303030303LL));
    return _mm256_cmpeq_epi8(_mm256_and_si256(expanded, mask), mask);
}

template <int nrc_y, bool signed_extreme>
static void mul_mat_iq2_s_native_impl(int n, const void *vx, size_t bx,
                                    const DataInfo &info, int nrc_x) {
    static constexpr auto scale_shuffle = [] {
        std::array<uint16_t, 128> table{};
        for (int g = 0; g < 8; ++g)
            for (int j = 0; j < 16; ++j) table[16 * g + j] = 0x0100 + g * 0x0202;
        return table;
    }();
    const block_q8_K *y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy)
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    const int blocks = n / QK_K;
    for (int ix = 0; ix < nrc_x; ++ix) {
        const auto *x = reinterpret_cast<const block_iq2_s *>(
            static_cast<const char *>(vx) + size_t(ix) * bx);
        __m256 acc[nrc_y] = {};
        for (int b = 0; b < blocks; ++b) {
            uint64_t packed_scales;
            std::memcpy(&packed_scales, x[b].scales, sizeof(packed_scales));
            const auto scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(
                _mm_set_epi64x(packed_scales >> 4, packed_scales),
                _mm_set1_epi8(15)), 1), _mm_set1_epi8(1));
            const auto scales16 = _mm256_cvtepi8_epi16(scales8);
            // Two independent sums help decode. With more input rows there
            // are enough independent chains already; one sum per input
            // avoids spilling 2*nrc_y live integer vectors on AVX2.
            __m256i total0[nrc_y] = {}, total1[nrc_y <= 2 ? nrc_y : 1] = {};
            for (int g = 0; g < 8; g += 2) {
                auto q0 = iq2_s_native_values(x[b].qs + 4 * g, x[b].qh[g]);
                auto q1 = iq2_s_native_values(x[b].qs + 4 * g + 4, x[b].qh[g + 1]);
                const auto s0 = iq_s_native_signs(x[b].qs + QK_K / 8 + 4 * g);
                const auto s1 = iq_s_native_signs(x[b].qs + QK_K / 8 + 4 * g + 4);
                const auto scale0 = _mm256_shuffle_epi8(scales16,
                    _mm256_loadu_si256((const __m256i *)(scale_shuffle.data() + 16 * g)));
                const auto scale1 = _mm256_shuffle_epi8(scales16,
                    _mm256_loadu_si256((const __m256i *)(scale_shuffle.data() + 16 * (g + 1))));
                if constexpr (signed_extreme) {
                    q0 = _mm256_sign_epi8(q0, _mm256_or_si256(s0, _mm256_set1_epi8(1)));
                    q1 = _mm256_sign_epi8(q1, _mm256_or_si256(s1, _mm256_set1_epi8(1)));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const auto a0 = _mm256_loadu_si256((const __m256i *)(y[iy][b].qs + 32 * g));
                    const auto a1 = _mm256_loadu_si256((const __m256i *)(y[iy][b].qs + 32 * g + 32));
                    __m256i pair0, pair1;
                    if constexpr (signed_extreme) {
                        // A signed-byte negation cannot represent +128. IQ2
                        // codebook values fit in int8, so apply Q8's sign to
                        // those values and treat abs(-128) as unsigned 128.
                        pair0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(q0, a0));
                        pair1 = _mm256_maddubs_epi16(_mm256_abs_epi8(a1), _mm256_sign_epi8(q1, a1));
                    } else {
                        pair0 = _mm256_maddubs_epi16(q0, _mm256_sub_epi8(_mm256_xor_si256(s0, a0), s0));
                        pair1 = _mm256_maddubs_epi16(q1, _mm256_sub_epi8(_mm256_xor_si256(s1, a1), s1));
                    }
                    if constexpr (nrc_y <= 2) {
                        total0[iy] = _mm256_add_epi32(total0[iy], _mm256_madd_epi16(pair0, scale0));
                        total1[iy] = _mm256_add_epi32(total1[iy], _mm256_madd_epi16(pair1, scale1));
                    } else {
                        total0[iy] = _mm256_add_epi32(total0[iy], _mm256_add_epi32(
                            _mm256_madd_epi16(pair0, scale0), _mm256_madd_epi16(pair1, scale1)));
                    }
                }
            }
            const float dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)));
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto total = total0[iy];
                if constexpr (nrc_y <= 2) total = _mm256_add_epi32(total, total1[iy]);
                acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[iy][b].d),
                    _mm256_cvtepi32_ps(total), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            info.store(ix, iy, 0.125f * _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum))));
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_s_q8_k(int n, const void *vx, size_t bx,
                             const DataInfo &info, int nrc_x) {
    assert(n % QK_K == 0);
    if (nrc_x == 0) return;
    // The normal Q8_K quantizer produces [-127, 127]. Inspect each input
    // once per matrix call, rather than adding endpoint handling to every
    // output row's inner loop. External Q8_K buffers may also contain -128.
    auto extremes = _mm256_setzero_si256();
    for (int iy = 0; iy < nrc_y; ++iy) {
        const auto *y = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
        for (int b = 0; b < n / QK_K; ++b)
            for (int g = 0; g < 8; ++g)
                extremes = _mm256_or_si256(extremes, _mm256_cmpeq_epi8(
                    _mm256_loadu_si256((const __m256i *)(y[b].qs + 32 * g)), _mm256_set1_epi8(-128)));
    }
    if (_mm256_movemask_epi8(extremes))
        mul_mat_iq2_s_native_impl<nrc_y, true>(n, vx, bx, info, nrc_x);
    else if constexpr (nrc_y >= 5) {
        // Four input rows fit the AVX2 register file. Finish both input
        // groups on a small output tile while its ordinary weights are hot.
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const int rows = std::min(8, nrc_x - ix);
            const auto *x = static_cast<const char *>(vx) + size_t(ix) * bx;
            auto tile = info;
            tile.s += ix;
            mul_mat_iq2_s_native_impl<4, false>(n, x, bx, tile, rows);
            tile.cur_y += 4;
            mul_mat_iq2_s_native_impl<nrc_y - 4, false>(n, x, bx, tile, rows);
        }
    } else {
        mul_mat_iq2_s_native_impl<nrc_y, false>(n, vx, bx, info, nrc_x);
    }
}

// IQ3_S stays in its original GGUF layout. Expand the eight 9-bit
// codebook indices into two scalar registers. The 128-byte high-bit table
// describes the format; no model weights are expanded or repacked.
static inline __m256i iq3_s_native_values(const uint8_t *qs, uint8_t qh) {
    static constexpr auto high = [] {
        std::array<uint64_t, 16> table{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j)
                table[i] |= uint64_t((i >> j) & 1) << (16 * j + 8);
        return table;
    }();
    const auto low = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)qs));
    const uint64_t lo = uint64_t(_mm_cvtsi128_si64(low)) | high[qh & 15];
    const uint64_t hi = uint64_t(_mm_extract_epi64(low, 1)) | high[qh >> 4];
    return _mm256_setr_epi32(
        iq3s_grid[uint16_t(lo)], iq3s_grid[uint16_t(lo >> 16)],
        iq3s_grid[uint16_t(lo >> 32)], iq3s_grid[uint16_t(lo >> 48)],
        iq3s_grid[uint16_t(hi)], iq3s_grid[uint16_t(hi >> 16)],
        iq3s_grid[uint16_t(hi >> 32)], iq3s_grid[uint16_t(hi >> 48)]);
}

template <int Inputs, bool Extreme>
static void mul_mat_iq3_s_native_impl(int n, const void *vx, size_t bx,
                                    const DataInfo &info, int outputs) {
    const block_q8_K *y[Inputs];
    for (int t = 0; t < Inputs; ++t) y[t] = (const block_q8_K *)info.src1_row(t);
    for (int row = 0; row < outputs; ++row) {
        const auto *x = (const block_iq3_s *)((const char *)vx + size_t(row) * bx);
        __m256 accum[Inputs] = {};
        for (int b = 0; b < n / QK_K; ++b) {
            __m256i sum0[Inputs] = {}, sum1[Inputs <= 2 ? Inputs : 1] = {};
            // Unroll the four fixed group pairs in a 256-value IQ3_S block.
            // Independent lookup chains hide some scalar table-load latency.
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
            for (int g = 0; g < 8; g += 2) {
                auto q0 = iq3_s_native_values(x[b].qs + 8 * g, x[b].qh[g]);
                auto q1 = iq3_s_native_values(x[b].qs + 8 * (g + 1), x[b].qh[g + 1]);
                const auto s0 = iq_s_native_signs(x[b].signs + 4 * g);
                const auto s1 = iq_s_native_signs(x[b].signs + 4 * (g + 1));
                const auto scale0 = _mm256_set1_epi16(2 * (x[b].scales[g / 2] & 15) + 1);
                const auto scale1 = _mm256_set1_epi16(2 * (x[b].scales[g / 2] >> 4) + 1);
                if constexpr (Extreme) {
                    q0 = _mm256_sub_epi8(_mm256_xor_si256(q0, s0), s0);
                    q1 = _mm256_sub_epi8(_mm256_xor_si256(q1, s1), s1);
                }
                for (int t = 0; t < Inputs; ++t) {
                    const auto a0 = _mm256_loadu_si256((const __m256i *)(y[t][b].qs + 32 * g));
                    const auto a1 = _mm256_loadu_si256((const __m256i *)(y[t][b].qs + 32 * (g + 1)));
                    __m256i p0, p1;
                    if constexpr (Extreme) {
                        // Negating Q8's -128 would overflow a signed byte.
                        // IQ3_S values fit in int8, so negate weights instead.
                        p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(q0, a0));
                        p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(a1), _mm256_sign_epi8(q1, a1));
                    } else {
                        p0 = _mm256_maddubs_epi16(q0, _mm256_sub_epi8(_mm256_xor_si256(a0, s0), s0));
                        p1 = _mm256_maddubs_epi16(q1, _mm256_sub_epi8(_mm256_xor_si256(a1, s1), s1));
                    }
                    const auto v0 = _mm256_madd_epi16(p0, scale0), v1 = _mm256_madd_epi16(p1, scale1);
                    if constexpr (Inputs <= 2) {
                        sum0[t] = _mm256_add_epi32(sum0[t], v0);
                        sum1[t] = _mm256_add_epi32(sum1[t], v1);
                    } else {
                        sum0[t] = _mm256_add_epi32(sum0[t], _mm256_add_epi32(v0, v1));
                    }
                }
            }
            const float dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)));
            for (int t = 0; t < Inputs; ++t) {
                auto sum = sum0[t];
                if constexpr (Inputs <= 2) sum = _mm256_add_epi32(sum, sum1[t]);
                accum[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][b].d), _mm256_cvtepi32_ps(sum), accum[t]);
            }
        }
        for (int t = 0; t < Inputs; ++t) {
            auto v = _mm_add_ps(_mm256_castps256_ps128(accum[t]), _mm256_extractf128_ps(accum[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            info.store(row, t, _mm_cvtss_f32(_mm_add_ss(v, _mm_movehdup_ps(v))));
        }
    }
}

template <int Inputs>
static void mul_mat_iq3_s_q8_k(int n, const void *vx, size_t bx,
                             const DataInfo &info, int outputs) {
    assert(n % QK_K == 0);
    if (!outputs) return;
    // The built-in Q8_K quantizer emits [-127, 127]. Scan once per call
    // to retain support for external Q8_K buffers containing -128.
    auto extreme = _mm256_setzero_si256();
    for (int t = 0; t < Inputs; ++t) {
        const auto *y = (const block_q8_K *)info.src1_row(t);
        for (int b = 0; b < n / QK_K; ++b) for (int g = 0; g < 8; ++g)
            extreme = _mm256_or_si256(extreme, _mm256_cmpeq_epi8(
                _mm256_loadu_si256((const __m256i *)(y[b].qs + 32 * g)), _mm256_set1_epi8(-128)));
    }
    if (_mm256_movemask_epi8(extreme)) {
        mul_mat_iq3_s_native_impl<Inputs, true>(n, vx, bx, info, outputs);
    } else if constexpr (Inputs >= 5) {
        // Limit live accumulators to four inputs on AVX2. Both input groups
        // reuse a small tile of ordinary weights while it remains cached.
        for (int r = 0; r < outputs; r += 8) {
            auto tile = info;
            tile.s += r;
            const auto *x = (const char *)vx + size_t(r) * bx;
            const int nr = std::min(8, outputs - r);
            mul_mat_iq3_s_native_impl<4, false>(n, x, bx, tile, nr);
            tile.cur_y += 4;
            mul_mat_iq3_s_native_impl<Inputs - 4, false>(n, x, bx, tile, nr);
        }
    } else {
        mul_mat_iq3_s_native_impl<Inputs, false>(n, vx, bx, info, outputs);
    }
}

// Decode two adjacent IQ4_XS groups with wide shuffles and share the
// decoded values across inputs. The weights retain their GGUF layout.
template <int Inputs>
static void mul_mat_iq4_xs_native_impl(int n, const void *vx, size_t bx,
                                     const DataInfo &info, int outputs) {
    static constexpr int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const block_q8_K *y[Inputs];
    for (int t = 0; t < Inputs; ++t) y[t] = (const block_q8_K *)info.src1_row(t);
    const auto table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)values));
    const auto mask = _mm256_set1_epi8(15);
    const int blocks = n / QK_K;
    for (int row = 0; row < outputs; ++row) {
        const auto *x = (const block_iq4_xs *)((const char *)vx + size_t(row) * bx);
        __m256 accum[Inputs] = {};
        for (int b = 0; b < blocks; ++b) {
            // A block spans 136 bytes. Keep all prefetch addresses within
            // this row, including the final short lookahead window.
            constexpr int ahead = 2;
            if (b + ahead < blocks) {
                const auto *next = (const char *)&x[b + ahead];
                _mm_prefetch(next, _MM_HINT_T0);
                _mm_prefetch(next + 64, _MM_HINT_T0);
                _mm_prefetch(next + 128, _MM_HINT_T0);
            }
            __m256i sum0[Inputs] = {}, sum1[Inputs] = {};
            for (int g = 0; g < 8; g += 2) {
                const auto bits = _mm256_loadu_si256((const __m256i *)(x[b].qs + 16 * g));
                const auto lo = _mm256_shuffle_epi8(table, _mm256_and_si256(bits, mask));
                const auto hi = _mm256_shuffle_epi8(table,
                    _mm256_and_si256(_mm256_srli_epi16(bits, 4), mask));
                const auto q0 = _mm256_permute2x128_si256(lo, hi, 0x20);
                const auto q1 = _mm256_permute2x128_si256(lo, hi, 0x31);
                const int sc0 = ((x[b].scales_l[g / 2] & 15) |
                                (((x[b].scales_h >> (2 * g)) & 3) << 4)) - 32;
                const int sc1 = ((x[b].scales_l[g / 2] >> 4) |
                                (((x[b].scales_h >> (2 * g + 2)) & 3) << 4)) - 32;
                const auto s0 = _mm256_set1_epi16(sc0), s1 = _mm256_set1_epi16(sc1);
                for (int t = 0; t < Inputs; ++t) {
                    const auto a0 = _mm256_loadu_si256((const __m256i *)(y[t][b].qs + 32 * g));
                    const auto a1 = _mm256_loadu_si256((const __m256i *)(y[t][b].qs + 32 * (g + 1)));
                    // IQ4_XS values never equal -128. Applying the Q8 sign
                    // to the weights therefore also supports Q8's -128.
                    const auto p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(q0, a0));
                    const auto p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(a1), _mm256_sign_epi8(q1, a1));
                    sum0[t] = _mm256_add_epi32(sum0[t], _mm256_madd_epi16(p0, s0));
                    sum1[t] = _mm256_add_epi32(sum1[t], _mm256_madd_epi16(p1, s1));
                }
            }
            const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)));
            for (int t = 0; t < Inputs; ++t) {
                accum[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][b].d),
                    _mm256_cvtepi32_ps(_mm256_add_epi32(sum0[t], sum1[t])), accum[t]);
            }
        }
        for (int t = 0; t < Inputs; ++t) {
            auto v = _mm_add_ps(_mm256_castps256_ps128(accum[t]), _mm256_extractf128_ps(accum[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            info.store(row, t, _mm_cvtss_f32(_mm_add_ss(v, _mm_movehdup_ps(v))));
        }
    }
}

template <int Inputs>
static void mul_mat_iq4_xs_q8_k(int n, const void *vx, size_t bx,
                              const DataInfo &info, int outputs) {
    assert(n % QK_K == 0);
    if (!outputs) return;
    if constexpr (Inputs >= 5) {
        // Bound register pressure while reusing each small weight tile.
        for (int r = 0; r < outputs; r += 8) {
            auto tile = info;
            tile.s += r;
            const auto *x = (const char *)vx + size_t(r) * bx;
            const int nr = std::min(8, outputs - r);
            mul_mat_iq4_xs_native_impl<4>(n, x, bx, tile, nr);
            tile.cur_y += 4;
            mul_mat_iq4_xs_native_impl<Inputs - 4>(n, x, bx, tile, nr);
        }
    } else {
        mul_mat_iq4_xs_native_impl<Inputs>(n, vx, bx, info, outputs);
    }
}

// Interleave four rows without expanding their packed quants to 256 bytes.
// Each 128-bit lane contains a four-byte group for one half of the block.
static inline void store_r4_groups(uint8_t * dst, const __m256i * rows, int half_stride) {
    const auto a = _mm256_unpacklo_epi32(rows[0], rows[1]);
    const auto b = _mm256_unpacklo_epi32(rows[2], rows[3]);
    const auto packed = _mm256_unpacklo_epi64(a, b);
    _mm_storeu_si128((__m128i *)dst, _mm256_castsi256_si128(packed));
    _mm_storeu_si128((__m128i *)(dst + half_stride), _mm256_extracti128_si256(packed, 1));
}

static inline void repack_q2_k_block(const block_q2_K * const * x, int ibl, block_q2_k_r4 & y) {
    __m128i scales[4];
    for (int k = 0; k < 4; ++k) {
        y.d[k] = x[k][ibl].d;
        y.d[k + 4] = x[k][ibl].dmin;
        scales[k] = _mm_loadu_si128((const __m128i *)x[k][ibl].scales);
    }
    const auto lo01 = _mm_unpacklo_epi8(scales[0], scales[1]);
    const auto lo23 = _mm_unpacklo_epi8(scales[2], scales[3]);
    const auto hi01 = _mm_unpackhi_epi8(scales[0], scales[1]);
    const auto hi23 = _mm_unpackhi_epi8(scales[2], scales[3]);
    _mm_storeu_si128((__m128i *)(y.scales +  0), _mm_unpacklo_epi16(lo01, lo23));
    _mm_storeu_si128((__m128i *)(y.scales + 16), _mm_unpackhi_epi16(lo01, lo23));
    _mm_storeu_si128((__m128i *)(y.scales + 32), _mm_unpacklo_epi16(hi01, hi23));
    _mm_storeu_si128((__m128i *)(y.scales + 48), _mm_unpackhi_epi16(hi01, hi23));

    const auto transpose = _mm256_setr_epi8(
        0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15,
        0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
    const auto gather = _mm256_setr_epi8(
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        0,4,8,12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const auto mask = _mm256_set1_epi8(3);
    const auto powers = _mm256_set1_epi32(0x40100401);
    const auto ones = _mm256_set1_epi16(1);
    for (int half = 0; half < 2; ++half) {
        __m256i source[4];
        for (int k = 0; k < 4; ++k)
            source[k] = _mm256_shuffle_epi8(_mm256_loadu_si256(
                (const __m256i *)(x[k][ibl].qs + 32 * half)), transpose);
        for (int plane = 0; plane < 4; ++plane) {
            __m256i rows[4];
            for (int k = 0; k < 4; ++k) {
                const auto values = _mm256_and_si256(_mm256_srl_epi16(
                    source[k], _mm_cvtsi32_si128(2 * plane)), mask);
                rows[k] = _mm256_shuffle_epi8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(values, powers), ones), gather);
            }
            store_r4_groups(y.qs + 32 * (4 * half + plane), rows, 16);
        }
    }
}

static inline void repack_q4_k_quants(const block_q4_K * const * x, int ibl, block_q4_k_r4 & y) {
    const auto pairs = _mm256_setr_epi8(
        0,8,1,9,2,10,3,11, 4,12,5,13,6,14,7,15,
        0,8,1,9,2,10,3,11, 4,12,5,13,6,14,7,15);
    const auto gather = _mm256_setr_epi8(
        0,2,4,6,8,10,12,14, -1,-1,-1,-1,-1,-1,-1,-1,
        0,2,4,6,8,10,12,14, -1,-1,-1,-1,-1,-1,-1,-1);
    const auto mask = _mm256_set1_epi8(15);
    const auto powers = _mm256_set1_epi16(0x1001);
    for (int block = 0; block < 4; ++block) {
        __m256i source[4];
        for (int k = 0; k < 4; ++k)
            source[k] = _mm256_shuffle_epi8(_mm256_loadu_si256(
                (const __m256i *)(x[k][ibl].qs + 32 * block)), pairs);
        for (int plane = 0; plane < 2; ++plane) {
            __m256i rows[4];
            for (int k = 0; k < 4; ++k) {
                const auto values = _mm256_and_si256(_mm256_srl_epi16(
                    source[k], _mm_cvtsi32_si128(4 * plane)), mask);
                rows[k] = _mm256_shuffle_epi8(_mm256_maddubs_epi16(values, powers), gather);
            }
            uint8_t * dst = y.qs + 64 * (2 * block + plane);
            store_r4_groups(dst, rows, 16);
            for (auto & row : rows) row = _mm256_srli_si256(row, 4);
            store_r4_groups(dst + 32, rows, 16);
        }
    }
}

inline uint8_t scrambled_sign(uint8_t s) {
    static const uint8_t k_table[128] = {
        0x00, 0x7f, 0x7e, 0x01, 0x7c, 0x03, 0x02, 0x7d, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7b, 0x7a, 0x05,
        0x70, 0x0f, 0x0e, 0x71, 0x0c, 0x73, 0x72, 0x0d, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0b, 0x0a, 0x75,
        0x60, 0x1f, 0x1e, 0x61, 0x1c, 0x63, 0x62, 0x1d, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1b, 0x1a, 0x65,
        0x10, 0x6f, 0x6e, 0x11, 0x6c, 0x13, 0x12, 0x6d, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6b, 0x6a, 0x15,
        0x40, 0x3f, 0x3e, 0x41, 0x3c, 0x43, 0x42, 0x3d, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3b, 0x3a, 0x45,
        0x30, 0x4f, 0x4e, 0x31, 0x4c, 0x33, 0x32, 0x4d, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4b, 0x4a, 0x35,
        0x20, 0x5f, 0x5e, 0x21, 0x5c, 0x23, 0x22, 0x5d, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5b, 0x5a, 0x25,
        0x50, 0x2f, 0x2e, 0x51, 0x2c, 0x53, 0x52, 0x2d, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2b, 0x2a, 0x55,
    };
    return k_table[s];
}

static void repack_iq2_xxs(int nrows, int n_per_row, const block_iq2_xxs * x, block_iq2_xxs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xxs * x4[4];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    std::memcpy(aux32, x4[k][ibl].qs + 4*ib, 2*sizeof(uint32_t));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = aux8[i];
                    }
                    uint8_t scale = aux32[1] >> 28;
                    uint8_t s1 = (scrambled_sign((aux32[1] >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32[1] >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32[1] >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32[1] >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32[1] = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32[1];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_iq2_xs(int nrows, int n_per_row, const block_iq2_xs * x, block_iq2_xs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xs * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        uint16_t v = x4[k][ibl].qs[4*ib+i];
                        uint8_t s = v >> 9;
                        y[ibl].qs[16*ib+4*k+i] = (v & 511) | (scrambled_sign(s) << 9);
                    }
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_iq3_xxs(int nrows, int n_per_row, const block_iq3_xxs * x, block_iq3_xxs_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_xxs * x4[4];
    uint32_t aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto xsas = x4[k][ibl].qs + QK_K/4;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 8; ++i) {
                        y[ibl].qs[32*ib+8*k+i] = x4[k][ibl].qs[8*ib+i];
                    }
                    std::memcpy(&aux32, xsas + 4*ib, 4);
                    uint8_t scale = aux32 >> 28;
                    uint8_t s1 = (scrambled_sign((aux32 >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32 >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32 >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32 >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32 = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32;
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

static void repack_q2_k(int nrows, int n_per_row, const block_q2_K * x, block_q2_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows % 4 == 0);
    assert(n_per_row % QK_K == 0);
    const int nblock = n_per_row / QK_K;
    for (int row = 0; row < nrows; row += 4) {
        const block_q2_K * x4[4];
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock * k;
        for (int ibl = 0; ibl < nblock; ++ibl) repack_q2_k_block(x4, ibl, y[ibl]);
        x += 4 * nblock;
        y += nblock;
    }
}

inline void convert_q3_k(const block_q3_K& x, uint8_t * L, uint8_t * Ld) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, x.scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(Ld, aux, 16);

    const uint8_t * q = x.qs;
    const uint8_t * hm = x.hmask;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                *L++ = ((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0);
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static void repack_q3_k(int nrows, int n_per_row, const block_q3_K * x, block_q3_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q3_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/16];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q3_k(x4[k][ibl], L, Ld);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    int is = 8*ib+k;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+0] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+0] >>  4) << 2*(is/16);
                    is += 4;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+1] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+1] >>  4) << 2*(is/16);
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
static void repack_q4_k(int nrows, int n_per_row, const block_q4_K * x, block_q4_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows % 4 == 0);
    assert(n_per_row % QK_K == 0);
    const int nblock = n_per_row / QK_K;
    for (int row = 0; row < nrows; row += 4) {
        const block_q4_K * x4[4];
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock * k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_h, 0, QK_K / 16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                y[ibl].d[k + 4] = x4[k][ibl].dmin;
                for (int ib = 0; ib < QK_K / 32; ++ib) {
                    uint8_t d, m;
                    get_scale_min_k4(ib, x4[k][ibl].scales, d, m);
                    y[ibl].scales_l[4 * ib + k] = (d & 15) | ((m & 15) << 4);
                    const uint8_t h = (d >> 4) | ((m >> 4) << 2);
                    y[ibl].scales_h[(4 * ib + k) % 16] |= h << (4 * ((4 * ib + k) / 16));
                }
            }
            repack_q4_k_quants(x4, ibl, y[ibl]);
        }
        x += 4 * nblock;
        y += nblock;
    }
}

inline void convert_q6_k(const block_q6_K& x, uint8_t * L) {
    const uint8_t * ql = x.ql;
    const uint8_t * qh = x.qh;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            L[n + l +  0] = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            L[n + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            L[n + l + 64] = (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4);
            L[n + l + 96] = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
        ql += 64;
        qh += 32;
    }
}

inline void convert_q5_k(const block_q5_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = (x.qs[32*ib64+j] & 0xf) | (((x.qh[j] >> (2*ib64+0)) & 1) << 4);
            L[64*ib64+j+32] = (x.qs[32*ib64+j] >>  4) | (((x.qh[j] >> (2*ib64+1)) & 1) << 4);
        }
    }
}

static void repack_q5_k(int nrows, int n_per_row, const block_q5_K * x, block_q5_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q5_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q5_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+ 4] >> 4) << 2) | ((L[32*ib+i+12] >> 4) << 3) |
                                                    ((L[32*ib+i+16] >> 4) << 4) | ((L[32*ib+i+24] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}


static void repack_q6_k(int nrows, int n_per_row, const block_q6_K * x, block_q6_k_r4 * y, [[maybe_unused]] bool online) {
    assert(nrows%4 == 0);
    assert(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q6_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q6_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[8*ib+k+0] = x4[k][ibl].scales[2*ib+0];
                    y[ibl].scales[8*ib+k+4] = x4[k][ibl].scales[2*ib+1];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].ql[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[32*ib+4*k+i+ 0] = (L[32*ib+i+ 0] >> 4) | ((L[32*ib+i+ 8] >> 4) << 2) | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 6);
                        y[ibl].qh[32*ib+4*k+i+16] = (L[32*ib+i+16] >> 4) | ((L[32*ib+i+24] >> 4) << 2) | ((L[32*ib+i+20] >> 4) << 4) | ((L[32*ib+i+28] >> 4) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

const Repack * get_repack_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Repack> k_map = {
        // { GGML_TYPE_IQ2_K,  { GGML_TYPE_IQ2_K_R4,  4,  (Repack::repack_func)repack_iq2_k}   },
        // { GGML_TYPE_IQ3_K,  { GGML_TYPE_IQ3_K_R4,  4,  (Repack::repack_func)repack_iq3_k}   },
        // { GGML_TYPE_IQ4_K,  { GGML_TYPE_IQ4_K_R4,  4,  (Repack::repack_func)repack_iq4_k}   },
        // { GGML_TYPE_IQ5_K,  { GGML_TYPE_IQ5_K_R4,  4,  (Repack::repack_func)repack_iq5_k}   },
        // { GGML_TYPE_IQ4_XS, { GGML_TYPE_IQ4_XS_R8, 8,  (Repack::repack_func)repack_iq4_xs}  },
        // { GGML_TYPE_IQ4_KS, { GGML_TYPE_IQ4_KS_R4, 4,  (Repack::repack_func)repack_iq4_ks}  },
        // { GGML_TYPE_IQ5_KS, { GGML_TYPE_IQ5_KS_R4, 4,  (Repack::repack_func)repack_iq5_ks}  },
        // { GGML_TYPE_IQ4_NL, { GGML_TYPE_IQ4_NL_R4, 4,  (Repack::repack_func)repack_iq4_nl}  },
        // { GGML_TYPE_IQ2_BN, { GGML_TYPE_IQ2_BN_R4, 4,  (Repack::repack_func)repack_iq2_bn}  },
        { GGML_TYPE_IQ2_XXS,{ GGML_TYPE_IQ2_XXS_R4,4,  (Repack::repack_func)repack_iq2_xxs} },
        { GGML_TYPE_IQ2_XS, { GGML_TYPE_IQ2_XS_R4, 4,  (Repack::repack_func)repack_iq2_xs}  },
        { GGML_TYPE_IQ3_XXS,{ GGML_TYPE_IQ3_XXS_R4,4,  (Repack::repack_func)repack_iq3_xxs} },
        // { GGML_TYPE_IQ3_S,  { GGML_TYPE_IQ3_S_R4,  4,  (Repack::repack_func)repack_iq3_s}   },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
        { GGML_TYPE_Q6_K,   { GGML_TYPE_Q6_K_R4,   4,  (Repack::repack_func)repack_q6_k}    },
        // { GGML_TYPE_Q4_0,   { GGML_TYPE_Q4_0_R8,   8,  (Repack::repack_func)repack_q4_0}    },
        // { GGML_TYPE_Q5_0,   { GGML_TYPE_Q5_0_R4,   4,  (Repack::repack_func)repack_q5_0}    },
        // { GGML_TYPE_Q6_0,   { GGML_TYPE_Q6_0_R4,   4,  (Repack::repack_func)repack_q6_0}    },
        // { GGML_TYPE_Q8_0,   { GGML_TYPE_Q8_0_R8,   8,  (Repack::repack_func)repack_q8_0}    },
        // { GGML_TYPE_Q8_K,   { GGML_TYPE_Q8_K_R8,   8,  (Repack::repack_func)repack_q8_k}    },
        // { GGML_TYPE_Q8_KV,  { GGML_TYPE_Q8_KV_R8,  8,  (Repack::repack_func)repack_q8_KV}   },
#ifdef __AVX512BF16__
        // { GGML_TYPE_BF16,   { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_bf16_t>}},
        // { GGML_TYPE_F16,    { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_half>}  },
#endif
    };
    auto it = k_map.find(type);
    return it != k_map.end() ? &it->second : nullptr;
}

template <int nrc, typename block_q8 = block_q8_K> struct Q8 {
    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

#ifdef HAVE_FANCY_SIMD
    inline __m512i load_quants64(int iy, int i, int j) const { return _mm512_loadu_si512((const __m512i*)y[iy][i].qs + j); }
#endif
    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].qs + j); }
    inline __m256i load_bsums(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].bsums); }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

template <int nrc_y>
static void mul_mat_iq2_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto qs = iq2[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi64x(iq2xxs_grid[qs[ 3]], iq2xxs_grid[qs[ 2]], iq2xxs_grid[qs[ 1]], iq2xxs_grid[qs[ 0]]);
                qx[1] = _mm256_set_epi64x(iq2xxs_grid[qs[ 7]], iq2xxs_grid[qs[ 6]], iq2xxs_grid[qs[ 5]], iq2xxs_grid[qs[ 4]]);
                qx[2] = _mm256_set_epi64x(iq2xxs_grid[qs[11]], iq2xxs_grid[qs[10]], iq2xxs_grid[qs[ 9]], iq2xxs_grid[qs[ 8]]);
                qx[3] = _mm256_set_epi64x(iq2xxs_grid[qs[15]], iq2xxs_grid[qs[14]], iq2xxs_grid[qs[13]], iq2xxs_grid[qs[12]]);
                qs += 16;
                auto sas = _mm_loadu_si128((const __m128i *)iq2[ibl].sas + ib);
                auto scales = _mm_and_si128(sas, _mm_set1_epi8(1));
#ifdef HAVE_FANCY_SIMD
                scales = _mm_dpbusd_epi32(_mm_set1_epi32(1), scales, _mm_set1_epi32(0x10080402));
#else
                scales = _mm_maddubs_epi16(scales, _mm_set1_epi32(0x10080402));
                scales = _mm_add_epi32(_mm_madd_epi16(_mm_set1_epi16(1), scales), _mm_set1_epi32(1));
#endif
                auto scales32 = MM256_SET_M128I(scales, scales);
                auto signs128 = _mm_and_si128(sas, _mm_set1_epi8(-2)); // 0xfe = -2 as signed. Needed to shutup compiler warning.
                signs128 = _mm_xor_si128(signs128, _mm_srli_epi16(signs128, 1));
#ifdef HAVE_FANCY_SIMD
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y));
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y));
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y));
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1)));
                    auto sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2)));
                    auto sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3)));
                    auto sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4)));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_xs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    auto s_shuffle = _mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200);
    __m256i qx[4];
    union { __m256i vec; uint16_t val[16]; } helper;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto val = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs + ib);
                helper.vec = _mm256_and_si256(val, _mm256_set1_epi16(511));
                qx[0] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 3]], iq2xs_grid[helper.val[ 2]], iq2xs_grid[helper.val[ 1]], iq2xs_grid[helper.val[ 0]]);
                qx[1] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 7]], iq2xs_grid[helper.val[ 6]], iq2xs_grid[helper.val[ 5]], iq2xs_grid[helper.val[ 4]]);
                qx[2] = _mm256_set_epi64x(iq2xs_grid[helper.val[11]], iq2xs_grid[helper.val[10]], iq2xs_grid[helper.val[ 9]], iq2xs_grid[helper.val[ 8]]);
                qx[3] = _mm256_set_epi64x(iq2xs_grid[helper.val[15]], iq2xs_grid[helper.val[14]], iq2xs_grid[helper.val[13]], iq2xs_grid[helper.val[12]]);
                auto signs16 = _mm256_srli_epi16(val, 9);
                signs16 = _mm256_xor_si256(signs16, _mm256_slli_epi16(signs16, 1));
                auto signs128 = _mm_or_si128(_mm256_castsi256_si128(signs16), _mm_slli_epi16(_mm256_extracti128_si256(signs16, 1), 8));
                signs128 = _mm_shuffle_epi8(signs128, s_shuffle);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y)); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y)); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y)); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y)); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    if constexpr (nrc_y == 1) {
                        isum[0] = _mm256_add_epi32(isum[0], _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))));
                        isum[1] = _mm256_add_epi32(isum[1], _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))));
                        isum[2] = _mm256_add_epi32(isum[2], _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))));
                        isum[3] = _mm256_add_epi32(isum[3], _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))));
                    } else {
                        auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))); // blocks 4x0, 4x1, row 0
                        auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))); // blocks 4x2, 4x3, row 1
                        auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))); // blocks 4x4, 4x5, row 2
                        auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))); // blocks 4x6, 4x7, row 3
                        auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                        auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                        auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                        isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                    }
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                if constexpr (nrc_y == 1) {
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    isum[0] = isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
                } else {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                    isum[iy] = _mm256_setzero_si256();
                }
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
// One activation row, two output rows per vector. The R4 weights stay packed;
// a gather decodes eight codebook entries and mask bits apply their signs.
static void mul_mat_iq2_s_r4_q8_k_decode(int n, const void *vx, size_t bx,
                                       const DataInfo &info, int nrc_x) {
    assert(nrc_x % 4 == 0);
    const auto *y = reinterpret_cast<const block_q8_K *>(info.src1_row(0));
    const __m256i shift = _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2);
    const __m512i shuffle = _mm512_set_epi64(
        0x0706070607060706LL, 0x0706070607060706LL,
        0x0504050405040504LL, 0x0504050405040504LL,
        0x0302030203020302LL, 0x0302030203020302LL,
        0x0100010001000100LL, 0x0100010001000100LL);
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const auto *x = reinterpret_cast<const block_iq2_s_r4 *>((const char *)vx + size_t(ix) * bx);
        __m256 acc = _mm256_setzero_ps();
        for (int block = 0; block < n / QK_K; ++block) {
            __m512i total[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
            for (int group = 0; group < QK_K / 32; ++group) {
                const auto qy = _mm512_broadcast_i64x4(
                    _mm256_loadu_si256((const __m256i *)(y[block].qs + 32 * group)));
                uint32_t bits;
                std::memcpy(&bits, x[block].scales + 4 * group, sizeof(bits));
                auto s = _mm_set1_epi32(bits);
                s = _mm_and_si128(_mm_unpacklo_epi8(s, _mm_srli_epi16(s, 4)), _mm_set1_epi8(15));
                s = _mm_or_si128(_mm_slli_epi16(s, 1), _mm_set1_epi8(1));
                const auto scales = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(s));
                for (int pair = 0; pair < 2; ++pair) {
                    const auto lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        (const __m128i *)(x[block].qs + 16 * group + 8 * pair)));
                    const auto *qh = x[block].qh + 4 * group + 2 * pair;
                    auto hi = _mm256_set_m128i(_mm_set1_epi32(qh[1]), _mm_set1_epi32(qh[0]));
                    hi = _mm256_and_si256(_mm256_sllv_epi32(hi, shift), _mm256_set1_epi32(0x300));
                    const auto qx = _mm512_i32gather_epi64(_mm256_or_si256(lo, hi), iq2s_grid, 8);
                    uint64_t signs;
                    std::memcpy(&signs, x[block].signs + 16 * group + 8 * pair, sizeof(signs));
                    const auto sy = _mm512_mask_sub_epi8(qy, signs, _mm512_setzero_si512(), qy);
                    const auto sc = _mm512_shuffle_epi8(scales, _mm512_add_epi8(shuffle, _mm512_set1_epi8(8 * pair)));
                    total[pair] = _mm512_add_epi32(total[pair],
                        _mm512_madd_epi16(sc, _mm512_maddubs_epi16(qx, sy)));
                }
            }
            // Restore the AVX2 kernel's four-row / two-half layout before
            // converting to float, preserving its accumulation order.
            const auto a = _mm512_castsi512_si256(total[0]);
            const auto b = _mm512_extracti64x4_epi64(total[0], 1);
            const auto c = _mm512_castsi512_si256(total[1]);
            const auto d = _mm512_extracti64x4_epi64(total[1], 1);
            auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b));
            auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(c, d), _mm256_unpackhi_epi32(c, d));
            auto sum = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
            auto dx = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x[block].d));
            auto scale = _mm256_mul_ps(_mm256_set_m128(dx, dx), _mm256_set1_ps(y[block].d));
            acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(sum), acc);
        }
        auto sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        info.store(ix, 0, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
    }
}
#endif

template <int nrc_y>
static void mul_mat_iq2_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if constexpr (nrc_y == 1) {
        mul_mat_iq2_s_r4_q8_k_decode(n, vx, bx, info, nrc_x);
        return;
    }
#endif
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    __m256i qx[4];
    auto grid = iq2s_grid;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            auto ql = iq2[ibl].qs;
            auto qh = iq2[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi64x(grid[ql[ 3] | ((qh[0] << 2) & 0x300)], grid[ql[ 2] | ((qh[0] << 4) & 0x300)], grid[ql[ 1] | ((qh[0] << 6) & 0x300)], grid[ql[ 0] | ((qh[0] << 8) & 0x300)]);
                qx[1] = _mm256_set_epi64x(grid[ql[ 7] | ((qh[1] << 2) & 0x300)], grid[ql[ 6] | ((qh[1] << 4) & 0x300)], grid[ql[ 5] | ((qh[1] << 6) & 0x300)], grid[ql[ 4] | ((qh[1] << 8) & 0x300)]);
                qx[2] = _mm256_set_epi64x(grid[ql[11] | ((qh[2] << 2) & 0x300)], grid[ql[10] | ((qh[2] << 4) & 0x300)], grid[ql[ 9] | ((qh[2] << 6) & 0x300)], grid[ql[ 8] | ((qh[2] << 8) & 0x300)]);
                qx[3] = _mm256_set_epi64x(grid[ql[15] | ((qh[3] << 2) & 0x300)], grid[ql[14] | ((qh[3] << 4) & 0x300)], grid[ql[13] | ((qh[3] << 6) & 0x300)], grid[ql[12] | ((qh[3] << 8) & 0x300)]);
                ql += 16; qh += 4;
                auto signs128 = _mm_loadu_si128((const __m128i*)iq2[ibl].signs + ib);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y)); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y)); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y)); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y)); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    if constexpr (nrc_y == 1) {
                        isum[0] = _mm256_add_epi32(isum[0], _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))));
                        isum[1] = _mm256_add_epi32(isum[1], _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))));
                        isum[2] = _mm256_add_epi32(isum[2], _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))));
                        isum[3] = _mm256_add_epi32(isum[3], _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))));
                    } else {
                        auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))); // blocks 4x0, 4x1, row 0
                        auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))); // blocks 4x2, 4x3, row 1
                        auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))); // blocks 4x4, 4x5, row 2
                        auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))); // blocks 4x6, 4x7, row 3
                        auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                        auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                        auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                        isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                    }
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                if constexpr (nrc_y == 1) {
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    isum[0] = isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
                } else {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                    isum[iy] = _mm256_setzero_si256();
                }
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq3 = (const block_iq3_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_mul_ps(_mm_set1_ps(0.25f), _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d))); // TODO: absorb the 0.25 factor into d when quantizing/repacking
            auto d4 = _mm256_set_m128(dl, dl);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+ 7]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 6]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 5]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 4]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+ 3]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 2]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 1]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 0]]);
                qx[1] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+15]], iq3xxs_grid[iq3[ibl].qs[32*ib+14]], iq3xxs_grid[iq3[ibl].qs[32*ib+13]], iq3xxs_grid[iq3[ibl].qs[32*ib+12]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+11]], iq3xxs_grid[iq3[ibl].qs[32*ib+10]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 9]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 8]]);
                qx[2] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+23]], iq3xxs_grid[iq3[ibl].qs[32*ib+22]], iq3xxs_grid[iq3[ibl].qs[32*ib+21]], iq3xxs_grid[iq3[ibl].qs[32*ib+20]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+19]], iq3xxs_grid[iq3[ibl].qs[32*ib+18]], iq3xxs_grid[iq3[ibl].qs[32*ib+17]], iq3xxs_grid[iq3[ibl].qs[32*ib+16]]);
                qx[3] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+31]], iq3xxs_grid[iq3[ibl].qs[32*ib+30]], iq3xxs_grid[iq3[ibl].qs[32*ib+29]], iq3xxs_grid[iq3[ibl].qs[32*ib+28]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+27]], iq3xxs_grid[iq3[ibl].qs[32*ib+26]], iq3xxs_grid[iq3[ibl].qs[32*ib+25]], iq3xxs_grid[iq3[ibl].qs[32*ib+24]]);
                auto sas = _mm_loadu_si128((const __m128i *)iq3[ibl].sas + ib);
                auto scales = _mm_and_si128(sas, _mm_set1_epi8(1));
#ifdef HAVE_FANCY_SIMD
                scales = _mm_dpbusd_epi32(_mm_set1_epi32(1), scales, _mm_set1_epi32(0x10080402));
#else
                scales = _mm_maddubs_epi16(scales, _mm_set1_epi32(0x10080402));
                scales = _mm_add_epi32(_mm_madd_epi16(_mm_set1_epi16(1), scales), _mm_set1_epi32(1));
                //auto t1 = _mm_or_si128(_mm_and_si128(scales, _mm_set1_epi32(0x00000001)), _mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x00000100)), 7));
                //auto t2 = _mm_or_si128(_mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x00010000)), 14), _mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x01000000)), 21));
                //scales = _mm_or_si128(_mm_slli_epi32(_mm_or_si128(t1, t2), 1), _mm_set1_epi32(1));
#endif
                auto scales32 = MM256_SET_M128I(scales, scales);
                auto signs128 = _mm_and_si128(sas, _mm_set1_epi8(-2)); // 0xfe = -2 as signed. Needed to shutup compiler warning.
                signs128 = _mm_xor_si128(signs128, _mm_srli_epi16(signs128, 1));
#ifdef HAVE_FANCY_SIMD
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y));
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y));
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y));
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    // Keep each output row's integer accumulator until the end
                    // of the block. Apply scales in the widening multiply-add,
                    // avoiding per-group row shuffles and int32 multiplies.
                    if constexpr (nrc_y == 1) {
                        const __m256i scs[4] = {
                            _mm256_shuffle_epi8(scales32, _mm256_set1_epi16(0x0100)),
                            _mm256_shuffle_epi8(scales32, _mm256_set1_epi16(0x0504)),
                            _mm256_shuffle_epi8(scales32, _mm256_set1_epi16(0x0908)),
                            _mm256_shuffle_epi8(scales32, _mm256_set1_epi16(0x0d0c))};
                        isum[0] = _mm256_add_epi32(isum[0], _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))));
                        isum[1] = _mm256_add_epi32(isum[1], _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))));
                        isum[2] = _mm256_add_epi32(isum[2], _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))));
                        isum[3] = _mm256_add_epi32(isum[3], _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))));
                    } else {
                        auto sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1)));
                        auto sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2)));
                        auto sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3)));
                        auto sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4)));
                        auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                        auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                        auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                    }
                }
#endif
            }
#ifndef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                isum[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
            }
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
void mul_mat_q2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
        
    Q8<nrc_y, block_q8_K> q8(info);
    auto mxf = _mm256_set1_epi8(0xf);
    auto m03 = _mm256_set1_epi8(0x03);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y] = {};
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    int8_t scales[64];

    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q2_k_r4 * iq2 = (const block_q2_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dm = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dm), _mm256_castps256_ps128(dm));
            auto m4 = _mm256_set_m128(_mm256_extractf128_ps(dm, 1), _mm256_extractf128_ps(dm, 1));
            m4 = _mm256_mul_ps(m4, _mm256_set1_ps(-1.f));
            auto all_scales1 = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales+0);
            auto all_scales2 = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales+1);
            auto scales1 = _mm256_and_si256(_mm256_srli_epi16(all_scales1, 4), mxf);
            auto scales2 = _mm256_and_si256(_mm256_srli_epi16(all_scales2, 4), mxf);
            {
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    auto d8 = _mm256_set1_ps(q8.scale(iy, ibl));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(m4, d8), _mm256_cvtepi32_ps(sumi), acc[iy]);
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    auto d8 = _mm256_set1_ps(q8.scale(iy, ibl));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(m4, d8), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    if constexpr (nrc_y == 1) {
                        d4 = _mm256_mul_ps(d4, d8);
                    }
#endif
                }
            }
            all_scales1 = _mm256_and_si256(all_scales1, mxf);
            all_scales2 = _mm256_and_si256(all_scales2, mxf);
            _mm256_storeu_si256((__m256i *)scales+0, all_scales1);
            _mm256_storeu_si256((__m256i *)scales+1, all_scales2);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 8*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs+ib);
                qx[0] = _mm256_and_si256(lb, m03);
                qx[1] = _mm256_and_si256(_mm256_srli_epi16(lb, 2), m03);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(lb, 4), m03);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(lb, 6), m03);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(iscales, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...3, so we can add add up all of them as int16_t without overflowing
                    auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_q3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m30 = _mm256_set1_epi8(0x30);
    auto m32 = _mm256_set1_epi8(32);
    auto m03 = _mm256_set1_epi8(0x03);
    auto m04 = _mm256_set1_epi8(0x04);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y];
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    int8_t scales[64];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q3_k_r4 * iq3 = (const block_q3_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
#ifndef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                d4 = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(0, ibl)));
            }
#endif
            auto slb = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
            auto shbits = _mm_loadu_si128((const __m128i *)iq3[ibl].scales_h);
            auto shb = MM256_SET_M128I(_mm_srli_epi16(shbits, 2), shbits);
            auto scales1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(slb, m4), _mm256_and_si256(_mm256_slli_epi16(shb, 4), m30)), m32);
            auto scales2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(slb, 4), m4), _mm256_and_si256(shb, m30)), m32);
            _mm256_storeu_si256((__m256i *)scales+0, scales1);
            _mm256_storeu_si256((__m256i *)scales+1, scales2);
            {
#ifndef HAVE_FANCY_SIMD
                auto min = _mm256_mul_ps(d4, _mm256_set1_ps(-4.f));
#endif
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
#ifdef HAVE_FANCY_SIMD
                s1 = _mm256_mullo_epi16(s1, _mm256_set1_epi16(-4));
                s2 = _mm256_mullo_epi16(s2, _mm256_set1_epi16(-4));
                s3 = _mm256_mullo_epi16(s3, _mm256_set1_epi16(-4));
                s4 = _mm256_mullo_epi16(s4, _mm256_set1_epi16(-4));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    isum[iy] = sumi;
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(min, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(min, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 8*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs+ib);
                auto hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh+ib);
                auto hb = MM256_SET_M128I(hbits, _mm_slli_epi16(hbits, 4));
                qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03),                       _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(iscales, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...8, so we can add add up all of them as int16_t without overflowing
                    auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif

                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
inline void process_min_r4_b32(int ibl, __m256 m4, __m256i mins, const Q8<nrc_y, block_q8_K>& q8, __m256 * acc) {
    auto mins_l = _mm256_castsi256_si128(mins);
    auto mins_h = _mm256_extracti128_si256(mins, 1);
    auto aux1   = _mm_unpacklo_epi32(mins_l, mins_h);
    auto aux2   = _mm_unpackhi_epi32(mins_l, mins_h);
    auto ic1 = _mm256_cvtepi8_epi32(aux1);
    auto ic2 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(aux1, 0xee));
    auto ic3 = _mm256_cvtepi8_epi32(aux2);
    auto ic4 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(aux2, 0xee));
    if constexpr (nrc_y == 1) {
        auto bs = _mm256_loadu_ps((const float *)q8.y[0][ibl].bsums);
        auto sumf = _mm256_mul_ps(_mm256_cvtepi32_ps(ic1), _mm256_shuffle_ps(bs, bs, 0x00));
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic2), _mm256_shuffle_ps(bs, bs, 0x55), sumf);
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic3), _mm256_shuffle_ps(bs, bs, 0xaa), sumf);
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic4), _mm256_shuffle_ps(bs, bs, 0xff), sumf);
        acc[0] = _mm256_fmadd_ps(m4, sumf, acc[0]);
    } else {
        auto c1 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic1));
        auto c2 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic2));
        auto c3 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic3));
        auto c4 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic4));
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto bs = _mm256_loadu_ps((const float *)q8.y[iy][ibl].bsums);
            acc[iy] = _mm256_fmadd_ps(c1, _mm256_shuffle_ps(bs, bs, 0x00), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c2, _mm256_shuffle_ps(bs, bs, 0x55), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c3, _mm256_shuffle_ps(bs, bs, 0xaa), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c4, _mm256_shuffle_ps(bs, bs, 0xff), acc[iy]);
        }
    }
}

template <int nrc_y>
static void mul_mat_q4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = _mm256_set1_epi8(0xf);
    auto m3 = _mm256_set1_epi8(0x30);
    int nbl = n / QK_K;
    union { __m256i vec; uint32_t val[8]; } hd;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q4_k_r4 * iq4 = (const block_q4_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));
            auto m4 = _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_set_m128(_mm256_extractf128_ps(dl, 1), _mm256_extractf128_ps(dl, 1)));
            auto lbits = _mm256_loadu_si256((const __m256i *)iq4[ibl].scales_l);
            auto hbits128 = _mm_loadu_si128((const __m128i *)iq4[ibl].scales_h);
            auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
            hd.vec = _mm256_or_si256(_mm256_and_si256(lbits, mf), _mm256_and_si256(hbits, m3));
            auto mins = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits, 4), mf), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m3));
            process_min_r4_b32(ibl, m4, mins, q8, acc);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales_d = _mm256_cvtepi8_epi32(_mm_set1_epi32(hd.val[ib]));
#else
                auto aux = _mm_set1_epi32(hd.val[ib]);
                aux = _mm_cvtepu8_epi16(_mm_unpacklo_epi8(aux, aux));
                auto scales_d = MM256_SET_M128I(aux, aux);
#endif
                auto bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+0);
                auto bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+1);
                qx[0] = _mm256_and_si256(bits1, mf);
                qx[1] = _mm256_and_si256(bits2, mf);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), mf);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), mf);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales_d, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(scales_d, _mm256_add_epi16(sumi1, sumi2)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_q5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = _mm256_set1_epi8(0xf);
    auto m10 = _mm256_set1_epi8(0x10);
    auto m30 = _mm256_set1_epi8(0x30);
    int nbl = n / QK_K;
    union { __m256i vec; uint32_t val[8]; } hd;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q5_k_r4 * iq5 = (const block_q5_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq5[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));
            auto m4 = _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_set_m128(_mm256_extractf128_ps(dl, 1), _mm256_extractf128_ps(dl, 1)));
            auto lbits = _mm256_loadu_si256((const __m256i *)iq5[ibl].scales_l);
            auto hbits128 = _mm_loadu_si128((const __m128i *)iq5[ibl].scales_h);
            auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
            hd.vec = _mm256_or_si256(_mm256_and_si256(lbits, mf), _mm256_and_si256(hbits, m30));
            auto mins = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits, 4), mf), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m30));
            process_min_r4_b32(ibl, m4, mins, q8, acc);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales_d = _mm256_cvtepi8_epi32(_mm_set1_epi32(hd.val[ib]));
#else
                auto aux = _mm_set1_epi32(hd.val[ib]);
                aux = _mm_cvtepu8_epi16(_mm_unpacklo_epi8(aux, aux));
                auto scales_d = MM256_SET_M128I(aux, aux);
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+1);
                auto hbits128 = _mm_loadu_si128((const __m128i*)iq5[ibl].qh + ib);
                auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
                qx[0] = _mm256_or_si256(_mm256_and_si256(lbits1, mf), _mm256_and_si256(m10, hbits));
                qx[1] = _mm256_or_si256(_mm256_and_si256(lbits2, mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 2)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 1)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 3)));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales_d, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // To avoid overflow, we can only add up to 4 q5 x q8 products.
                    auto sumi = _mm256_add_epi32(_mm256_madd_epi16(scales_d, sumi1), _mm256_madd_epi16(scales_d, sumi2));
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_q6_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m3 = _mm256_set1_epi8(0x30);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y];
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q6_k_r4 * iq6 = (const block_q6_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
#ifndef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                d4 = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(0, ibl)));
            }
#endif
            {
#ifndef HAVE_FANCY_SIMD
                auto min = _mm256_mul_ps(d4, _mm256_set1_ps(-32.f));
#endif
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+2)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+3)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
#ifdef HAVE_FANCY_SIMD
                s1 = _mm256_mullo_epi16(s1, _mm256_set1_epi16(-32));
                s2 = _mm256_mullo_epi16(s2, _mm256_set1_epi16(-32));
                s3 = _mm256_mullo_epi16(s3, _mm256_set1_epi16(-32));
                s4 = _mm256_mullo_epi16(s4, _mm256_set1_epi16(-32));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    isum[iy] = sumi;
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(min, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(min, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
            const uint32_t * scales = (const uint32_t *)iq6[ibl].scales;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 2*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq6[ibl].ql+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq6[ibl].ql+2*ib+1);
                auto hbits  = _mm256_loadu_si256((const __m256i *)iq6[ibl].qh+ib);
                qx[0] = _mm256_or_si256(_mm256_and_si256(lbits1, m4), _mm256_and_si256(m3, _mm256_slli_epi16(hbits, 4)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(lbits2, m4), _mm256_and_si256(m3, hbits));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), m4), _mm256_and_si256(m3, _mm256_slli_epi16(hbits, 2)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), m4), _mm256_and_si256(m3, _mm256_srli_epi16(hbits, 2)));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(iscales, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...63, so we can add at most 4 as int16_t to be sure of no int16_t overflow
                    auto sumi = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

// Keep the ordinary IQ4_NL layout: neighboring output rows share the Q8_0
// loads without allocating another weight representation for CPU decode.
template <int rows>
static void mul_mat_iq4_nl_q8_0_rows(int blocks, const char *vx, size_t bx,
                                    const block_q8_0 *y, const DataInfo &info,
                                    int ix, int iy) {
    const __m128i values = _mm_setr_epi8(
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113);
    const __m128i mask = _mm_set1_epi8(15);
#if !defined(__AVX512VNNI__) || !defined(__AVX512VL__)
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    const block_iq4_nl *x[rows];
    __m256 even[rows], odd[rows];
    for (int r = 0; r < rows; ++r) {
        x[r] = reinterpret_cast<const block_iq4_nl *>(vx + r * bx);
        even[r] = odd[r] = _mm256_setzero_ps();
    }
    auto accumulate = [&](int block, __m256 *acc) {
        const __m256i qy = _mm256_loadu_si256((const __m256i *)y[block].qs);
        const __m256i ay = _mm256_abs_epi8(qy);
        const float dy = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(y[block].d)));
        for (int r = 0; r < rows; ++r) {
            const __m128i bits = _mm_loadu_si128((const __m128i *)x[r][block].qs);
            // Split the two nibbles across the 128-bit lanes, then decode
            // all 32 values with one shuffle instead of two plus an insert.
            const auto packed = _mm256_broadcastsi128_si256(bits);
            const auto indices = _mm256_and_si256(_mm256_blend_epi32(
                packed, _mm256_srli_epi16(packed, 4), 0xf0), _mm256_broadcastsi128_si256(mask));
            const auto qx = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(values), indices);
            // IQ4_NL never contains -128. Applying the Q8 sign to IQ4
            // therefore also handles Q8=-128, without signed-byte overflow.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const __m256i sum = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ay, _mm256_sign_epi8(qx, qy));
#else
            const __m256i products = _mm256_maddubs_epi16(ay, _mm256_sign_epi8(qx, qy));
            const __m256i sum = _mm256_madd_epi16(products, ones);
#endif
            const float dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[r][block].d)));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dx * dy), _mm256_cvtepi32_ps(sum), acc[r]);
        }
    };
    int block = 0;
    for (; block + 1 < blocks; block += 2) {
        // Interleaved row streams can starve decode when experts no
        // longer fit in cache. Keep the lookahead within each weight row.
        constexpr int ahead = 8;
        if (block + ahead < blocks)
            for (int r = 0; r < rows; ++r)
                _mm_prefetch((const char *)&x[r][block + ahead], _MM_HINT_T0);
        accumulate(block, even);
        accumulate(block + 1, odd);
    }
    if (block < blocks) accumulate(block, even);
    for (int r = 0; r < rows; ++r) {
        const __m256 acc = _mm256_add_ps(even[r], odd[r]);
        __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        info.store(ix + r, iy, _mm_cvtss_f32(sum));
    }
}

template <int nrc_y>
static void mul_mat_iq4_nl_q8_0(int n, const void *vx, size_t bx,
                               const DataInfo &info, int nrc_x) {
    assert(n % QK4_NL == 0);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    // Two output rows reduce register pressure for short decode dots while
    // retaining the original even/odd block accumulation order.
    constexpr int rows = nrc_y == 1 ? 2 : 4;
#elif defined(__AVX512F__)
    constexpr int rows = 4;
#else
    // Fewer simultaneous streams improve cold-weight access on AVX2 and
    // leave room for the unpack temporaries and even/odd accumulators.
    constexpr int rows = 2;
#endif
    for (int iy = 0; iy < nrc_y; ++iy) {
        const auto *y = reinterpret_cast<const block_q8_0 *>(info.src1_row(iy));
        int ix = 0;
        for (; ix + rows - 1 < nrc_x; ix += rows) {
            mul_mat_iq4_nl_q8_0_rows<rows>(n / QK4_NL, (const char *)vx + ix * bx, bx, y, info, ix, iy);
        }
        if (ix + 1 < nrc_x) {
            mul_mat_iq4_nl_q8_0_rows<2>(n / QK4_NL, (const char *)vx + ix * bx, bx, y, info, ix, iy);
            ix += 2;
        }
        if (ix < nrc_x) {
            mul_mat_iq4_nl_q8_0_rows<1>(n / QK4_NL, (const char *)vx + ix * bx, bx, y, info, ix, iy);
        }
    }
}

// Transpose each Q8 block once so a Q2 half-block can be unpacked with a
// broadcast and per-dword shifts, without interleaving its bytes for every row.
struct Q2_0InputBlock {
    __m256i qs;
    __m256i pair_sums;
    float d;
};

template <int nrc_y>
static void mul_mat_q2_0_q8_0(int n, const void *vx, size_t bx,
                             const DataInfo &info, int nrc_x) {
    assert(n % QK2_0 == 0);
    if (nrc_x <= 0) return;
    // A single output row cannot amortize the input preparation.
    if (nrc_x == 1) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            float result;
            ggml_vec_dot_q2_0_q8_0(n, &result, 0, vx, 0, info.src1_row(iy), 0, 1);
            info.store(0, iy, result);
        }
        return;
    }
    const int blocks = n / QK2_0;
    Q2_0InputBlock local[64];
    std::vector<Q2_0InputBlock> large;
    if (2 * blocks > 64) large.resize(2 * blocks);
    auto *prepared = 2 * blocks <= 64 ? local : large.data();
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i shifts = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i transpose = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i join = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int iy = 0; iy < nrc_y; ++iy) {
        const auto *y = reinterpret_cast<const block_q8_0 *>(info.src1_row(iy));
        for (int b = 0; b < 2 * blocks; ++b) {
            const auto q = _mm256_loadu_si256((const __m256i *)y[b].qs);
            prepared[b].qs = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(q, transpose), join);
            prepared[b].pair_sums = _mm256_maddubs_epi16(ones8, prepared[b].qs);
            prepared[b].d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(y[b].d)));
        }
        for (int ix = 0; ix < nrc_x; ++ix) {
            const auto *x = reinterpret_cast<const block_q2_0 *>((const char *)vx + ix * bx);
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            for (int b = 0; b < blocks; ++b) {
                const auto &y0 = prepared[2 * b];
                const auto &y1 = prepared[2 * b + 1];
                const auto bits0 = _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)x[b].qs));
                const auto bits1 = _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(x[b].qs + 8)));
                const auto q0 = _mm256_and_si256(_mm256_srlv_epi32(bits0, shifts), mask);
                const auto q1 = _mm256_and_si256(_mm256_srlv_epi32(bits1, shifts), mask);
                // Subtract in int16 to support Q8=-128 as well as zero codes.
                const auto p0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q0, y0.qs), y0.pair_sums);
                const auto p1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q1, y1.qs), y1.pair_sums);
                const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)));
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d * y0.d),
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d * y1.d),
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(p1, ones16)), acc1);
            }
            const auto acc = _mm256_add_ps(acc0, acc1);
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            info.store(ix, iy, _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum))));
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline float q8_0_scale(const block_q8_0 &block) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)block.d)));
}

static inline void accumulate_q8_0_q8_0_vnni(const block_q8_0 &x,
                                               __m256i qy, float dy,
                                               __m256 &acc) {
    const __m256i qx = _mm256_loadu_si256((const __m256i *)x.qs);
    const __m256i ax = _mm256_abs_epi8(qx);
    const __m256i sy = _mm256_sign_epi8(qy, qx);
    const __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
    const float scale = q8_0_scale(x) * dy;
    acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(dot), acc);
}

static inline float horizontal_sum_q8_0(__m256 value) {
    __m128 sum = _mm256_extractf128_ps(value, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}
#endif

template <int nrc_y>
static void mul_mat_q8_0_q8_0_fast(int n, const void * vx, size_t bx,
                                    const DataInfo& info, int nrc_x) {
    const int blocks = n / QK8_0;
    assert(n % QK8_0 == 0);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    for (int ix = 0; ix < nrc_x; ++ix) {
        const auto * x = (const block_q8_0 *)((const char *)vx + ix * bx);
        for (int iy = 0; iy < nrc_y; ++iy) {
            const auto * y = (const block_q8_0 *)info.src1_row(iy);
            __m256 acc0 = _mm256_setzero_ps();
            for (int ib = 0; ib < blocks; ++ib) {
                const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
                accumulate_q8_0_q8_0_vnni(x[ib], qy, q8_0_scale(y[ib]), acc0);
            }
            info.store(ix, iy, horizontal_sum_q8_0(acc0));
        }
    }
#else
    for (int ix = 0; ix < nrc_x; ++ix) {
        const void * x = (const char *)vx + ix * bx;
        for (int iy = 0; iy < nrc_y; ++iy) {
            float result = 0.0f;
            ggml_vec_dot_q8_0_q8_0(n, &result, 0, x, 0,
                                    info.src1_row(iy), 0, 1);
            info.store(ix, iy, result);
        }
    }
#endif
}

static void mul_mat_empty(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    return;
}

#define RETURN_MATMUL_FUNCTION(FUNC, X) \
    if ((X) == 1) return FUNC <1>; \
    if ((X) == 2) return FUNC <2>; \
    if ((X) == 3) return FUNC <3>; \
    if ((X) == 4) return FUNC <4>; \
    if ((X) == 5) return FUNC <5>; \
    if ((X) == 6) return FUNC <6>; \
    if ((X) == 7) return FUNC <7>; \
    if ((X) == 8) return FUNC <8>; \
    return nullptr;

mul_mat_t GetMulMatFunction(ggml_type type, int nrc_y) {
    if (type == GGML_TYPE_IQ4_NL) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq4_nl_q8_0, nrc_y)
    } else if (type == GGML_TYPE_Q8_0) {
        RETURN_MATMUL_FUNCTION(mul_mat_q8_0_q8_0_fast, nrc_y)
    } else if (type == GGML_TYPE_Q2_0) {
        RETURN_MATMUL_FUNCTION(mul_mat_q2_0_q8_0, nrc_y)
    } else if (type == GGML_TYPE_IQ2_XXS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_xxs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ2_XS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_xs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ2_S) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_s_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ3_S) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq3_s_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ4_XS) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq4_xs_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ2_S_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq2_s_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_IQ3_XXS_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_iq3_xxs_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q2_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q2_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q3_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q3_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q4_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q4_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q5_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q5_k_r4_q8_k, nrc_y)
    } else if (type == GGML_TYPE_Q6_K_R4) {
        RETURN_MATMUL_FUNCTION(mul_mat_q6_k_r4_q8_k, nrc_y)
    } else {
        return nullptr;
    }
}

#else
// Keep the original GGUF layout when the repacked kernels are unavailable.
// The caller then uses the ordinary quantized dot-product implementation.
const Repack * get_repack_info(ggml_type type) {
    return nullptr;
}

mul_mat_t GetMulMatFunction(ggml_type type, int nrc_y) {
    return nullptr;
}
#endif // architecture-specific repacked kernels
