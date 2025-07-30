#include "gguf.h"
#include <assert.h>


// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

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
        // { GGML_TYPE_IQ2_XXS,{ GGML_TYPE_IQ2_XXS_R4,4,  (Repack::repack_func)repack_iq2_xxs} },
        // { GGML_TYPE_IQ2_XS, { GGML_TYPE_IQ2_XS_R4, 4,  (Repack::repack_func)repack_iq2_xs}  },
        // { GGML_TYPE_IQ2_S,  { GGML_TYPE_IQ2_S_R4,  4,  (Repack::repack_func)repack_iq2_s}   },
        // { GGML_TYPE_IQ3_XXS,{ GGML_TYPE_IQ3_XXS_R4,4,  (Repack::repack_func)repack_iq3_xxs} },
        // { GGML_TYPE_IQ3_S,  { GGML_TYPE_IQ3_S_R4,  4,  (Repack::repack_func)repack_iq3_s}   },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        // { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
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
template void mul_mat_q2_k_r4_q8_k<1>(int, const void*, size_t, const DataInfo&, int);

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
template void mul_mat_q3_k_r4_q8_k<1>(int, const void*, size_t, const DataInfo&, int);

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
template void mul_mat_q4_k_r4_q8_k<1>(int, const void*, size_t, const DataInfo&, int);

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
template void mul_mat_q5_k_r4_q8_k<1>(int, const void*, size_t, const DataInfo&, int);

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
template void mul_mat_q6_k_r4_q8_k<1>(int, const void*, size_t, const DataInfo&, int);

