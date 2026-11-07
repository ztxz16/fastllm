// Fastllm's independently implemented EXL3 CPU dot products.
//
// mul1 has an affine byte-sum codebook. Omitting its per-weight FP16 rounding
// permits a byte-sum dot followed by one scale and offset. The exact decoder
// remains in exl3.cpp for reconstruction, prefill and numerical reference.
//
// Quantized dot: symmetric 12-bit activations per 128 inputs. Each decoded
// byte pair sums to at most 510, so maddubs against ones cannot saturate.
// The complete int32 block is bounded by 128 * 1020 * 2047 = 267256320.
// AVX2 and AVX512BW perform the same integer arithmetic; narrower ISA paths
// retain the existing exact decoder. No CPU implementation is imported from
// Maya. Format/provenance references are in third_party/maya_exl3/NOTICE.txt.
//
// Cache shifted word vectors once per tile. A row then selects at most two
// adjacent vectors, avoiding gathers in the 16-row inner dot product.
template <int Bits, int Half> struct Exl3AffineGeometry {
    static constexpr int base(int c) {
        const int col = c + Half;
        const int step = ((col >> 3) << 2) | ((col & 7) << 5);
        return ((step + 1) * Bits / 2) + 128 * Bits - 16;
    }
    static constexpr int delta(int row) {
        const int step = (row & 1) | ((row >> 3) << 1) | (((row >> 1) & 3) << 3);
        return (step + 1) * Bits / 2 - Bits / 2;
    }
    static constexpr int cached() {
        int n = 0;
        for (int c = 0; c < Exl3Lanes; ++c)
            n = std::max(n, (base(c) % 32 + delta(15)) / 32 + 2);
        return n;
    }
    struct Geometry {
        alignas(64) int index[cached()][Exl3Lanes];
        alignas(64) uint32_t right[16][Exl3Lanes], left[16][Exl3Lanes], tail[16][Exl3Lanes], funnel[16][Exl3Lanes];
        int first[16];
        unsigned select[16];
        bool crossing[16];
    };
    static constexpr Geometry make() {
        Geometry g{};
        for (int j = 0; j < cached(); ++j)
            for (int c = 0; c < Exl3Lanes; ++c)
                g.index[j][c] = (base(c) / 32 + j) % (4 * Bits);
        for (int row = 0; row < 16; ++row) {
            g.first[row] = 100;
            for (int c = 0; c < Exl3Lanes; ++c)
                g.first[row] = std::min(g.first[row], (base(c) % 32 + delta(row)) / 32);
            for (int c = 0; c < Exl3Lanes; ++c) {
                const int bit = base(c) % 32 + delta(row), rem = bit % 32;
                if (bit / 32 > g.first[row])
                    g.select[row] |= 1u << c;
                g.right[row][c] = rem <= 16 ? 16 - rem : 0;
                g.left[row][c] = rem > 16 ? rem - 16 : 0;
                g.tail[row][c] = 48 - rem;
                g.funnel[row][c] = (48 - rem) % 32;
                g.crossing[row] |= rem > 16;
            }
        }
        return g;
    }
    static constexpr Geometry geometry = make();
};

template <int Bits, int Half> struct Exl3AffineTile {
    using G = Exl3AffineGeometry<Bits, Half>;
#if defined(__AVX512F__)
    __m512i cached[G::cached()];
    struct Raw {
        __m512i chunk[(4 * Bits + 15) / 16];
        explicit Raw(const uint32_t *p) {
            for (int i = 0; i < (4 * Bits + 15) / 16; ++i) {
                const int count = std::min(16, 4 * Bits - 16 * i);
                chunk[i] = _mm512_maskz_loadu_epi32((1u << count) - 1, p + i * 16);
            }
        }
    };
    template <int J> inline __m512i select(const Raw &raw) {
        const auto idx = _mm512_loadu_si512(G::geometry.index[J]);
        if constexpr (Bits <= 4)
            return _mm512_permutexvar_epi32(idx, raw.chunk[0]);
        auto v = _mm512_permutex2var_epi32(raw.chunk[0], idx, raw.chunk[1]);
        if constexpr (Bits > 8) {
            constexpr unsigned mask = [] {
                unsigned m = 0;
                for (int c = 0; c < 16; ++c)
                    if (G::geometry.index[J][c] >= 32)
                        m |= 1u << c;
                return m;
            }();
            if constexpr (mask) {
                const auto hi = Bits <= 12 ? _mm512_permutexvar_epi32(idx, raw.chunk[2])
                                           : _mm512_permutex2var_epi32(raw.chunk[2], idx, raw.chunk[3]);
                v = _mm512_mask_blend_epi32(mask, v, hi);
            }
        }
        return v;
    }
#elif defined(__AVX2__)
    __m256i cached[G::cached()];
    struct Raw {
        __m256i chunk[(4 * Bits + 7) / 8];
        __m256i transposed[4];
        explicit Raw(const uint32_t *p) {
            for (int i = 0; i < (4 * Bits + 7) / 8; ++i) {
                alignas(32) int mask[8];
                for (int c = 0; c < 8; ++c)
                    mask[c] = (i * 8 + c < 4 * Bits) ? -1 : 0;
                if ((i + 1) * 8 <= 4 * Bits)
                    chunk[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + i * 8));
                else
                    chunk[i] = _mm256_maskload_epi32(reinterpret_cast<const int *>(p + i * 8),
                                                     _mm256_load_si256(reinterpret_cast<const __m256i *>(mask)));
            }
            if constexpr (Bits == 8) {
                const auto a = _mm256_unpacklo_epi32(chunk[0], chunk[1]);
                const auto b = _mm256_unpackhi_epi32(chunk[0], chunk[1]);
                const auto c = _mm256_unpacklo_epi32(chunk[2], chunk[3]);
                const auto d = _mm256_unpackhi_epi32(chunk[2], chunk[3]);
                transposed[0] = _mm256_unpacklo_epi64(a, c);
                transposed[1] = _mm256_unpackhi_epi64(a, c);
                transposed[2] = _mm256_unpacklo_epi64(b, d);
                transposed[3] = _mm256_unpackhi_epi64(b, d);
            }
        }
    };
    template <int J, int Chunk = 0> inline __m256i source_order(const Raw &raw) {
        constexpr unsigned mask = [] {
            unsigned m = 0;
            for (int c = 0; c < 8; ++c)
                if (G::geometry.index[J][c] / 8 == Chunk)
                    m |= 1u << (G::geometry.index[J][c] % 8);
            return m;
        }();
        __m256i v;
        if constexpr (Chunk + 1 < (4 * Bits + 7) / 8)
            v = source_order<J, Chunk + 1>(raw);
        else
            return raw.chunk[Chunk];
        if constexpr (mask)
            return _mm256_blend_epi32(v, raw.chunk[Chunk], mask);
        return v;
    }
    template <int J, int Chunk = 0> inline __m256i select(const Raw &raw) {
        if constexpr (Bits == 8) {
            alignas(32) static constexpr auto order = [] {
                std::array<int, 8> lanes{};
                for (int c = 0; c < 8; ++c) {
                    int at = G::geometry.index[J][c] / 4;
                    lanes[c] = ((at & 1) << 2) | (at >> 1);
                }
                return lanes;
            }();
            return _mm256_permutevar8x32_epi32(raw.transposed[G::geometry.index[J][0] % 4],
                                               _mm256_loadu_si256(reinterpret_cast<const __m256i *>(order.data())));
        }
        constexpr bool permutation = [] {
            unsigned m = 0;
            for (int c = 0; c < 8; ++c)
                m |= 1u << (G::geometry.index[J][c] % 8);
            return m == 255;
        }();
        if constexpr (Chunk == 0 && permutation) {
            return _mm256_permutevar8x32_epi32(
                source_order<J>(raw), _mm256_loadu_si256(reinterpret_cast<const __m256i *>(G::geometry.index[J])));
        }
        constexpr unsigned mask = [] {
            unsigned m = 0;
            for (int c = 0; c < 8; ++c)
                if (G::geometry.index[J][c] / 8 == Chunk)
                    m |= 1u << c;
            return m;
        }();
        __m256i v;
        if constexpr (Chunk + 1 < (4 * Bits + 7) / 8)
            v = select<J, Chunk + 1>(raw);
        else
            v = _mm256_setzero_si256();
        if constexpr (mask) {
            auto x = _mm256_permutevar8x32_epi32(
                raw.chunk[Chunk], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(G::geometry.index[J])));
            if constexpr (mask == 255)
                return x;
            else
                return _mm256_blend_epi32(v, x, mask);
        }
        return v;
    }
#else
    const uint32_t *words;
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    template <int J = 0> inline void prepare(const Raw &raw) {
        cached[J] = select<J>(raw);
        if constexpr (J + 1 < G::cached())
            prepare<J + 1>(raw);
    }
#endif
    explicit Exl3AffineTile(const uint32_t *p) {
#if defined(__AVX512F__) || defined(__AVX2__)
        Raw raw(p);
        prepare(raw);
#else
        words = p;
#endif
    }
    template <int Row, bool Integer = false> inline auto values() const {
        constexpr int j = G::geometry.first[Row];
        constexpr unsigned mask = G::geometry.select[Row];
#if defined(__AVX512F__)
        auto a = cached[j];
        if constexpr (mask)
            a = _mm512_mask_blend_epi32(mask, a, cached[j + 1]);
#if defined(__AVX512VBMI2__)
        if constexpr (G::geometry.crossing[Row]) {
            auto b = cached[j + 1];
            if constexpr (mask)
                b = _mm512_mask_blend_epi32(mask, b, cached[j + 2]);
            constexpr unsigned endMask = [] {
                unsigned m = 0;
                for (int c = 0; c < 16; ++c)
                    if (G::geometry.left[Row][c])
                        m |= 1u << c;
                return m;
            }();
            const auto end = _mm512_mask_blend_epi32(endMask, a, b);
            a = _mm512_shrdv_epi32(end, a, _mm512_loadu_si512(G::geometry.funnel[Row]));
        } else
            a = _mm512_srlv_epi32(a, _mm512_loadu_si512(G::geometry.right[Row]));
#else
        constexpr bool right = [] {
            for (int c = 0; c < Exl3Lanes; ++c)
                if (G::geometry.right[Row][c])
                    return true;
            return false;
        }();
        constexpr bool left = [] {
            for (int c = 0; c < Exl3Lanes; ++c)
                if (G::geometry.left[Row][c])
                    return true;
            return false;
        }();
        if constexpr (right)
            a = _mm512_srlv_epi32(a, _mm512_loadu_si512(G::geometry.right[Row]));
        if constexpr (left)
            a = _mm512_sllv_epi32(a, _mm512_loadu_si512(G::geometry.left[Row]));
        if constexpr (G::geometry.crossing[Row]) {
            auto b = cached[j + 1];
            if constexpr (mask)
                b = _mm512_mask_blend_epi32(mask, b, cached[j + 2]);
            a = _mm512_or_si512(a, _mm512_srlv_epi32(b, _mm512_loadu_si512(G::geometry.tail[Row])));
        }
#endif
        auto mixed = _mm512_mullo_epi32(_mm512_and_si512(a, _mm512_set1_epi32(65535)), _mm512_set1_epi32(0x83dcd12du));
        if constexpr (Integer)
            return mixed;
        else {
#if defined(__AVX512VNNI__)
            auto sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mixed, _mm512_set1_epi32(0x01010101));
#elif defined(__AVX512BW__)
            auto sum = _mm512_madd_epi16(_mm512_maddubs_epi16(mixed, _mm512_set1_epi8(1)), _mm512_set1_epi16(1));
#else
            auto p = _mm512_add_epi32(_mm512_and_si512(mixed, _mm512_set1_epi32(0x00ff00ff)),
                                      _mm512_and_si512(_mm512_srli_epi32(mixed, 8), _mm512_set1_epi32(0x00ff00ff)));
            auto sum = _mm512_and_si512(_mm512_add_epi32(p, _mm512_srli_epi32(p, 16)), _mm512_set1_epi32(65535));
#endif
            return _mm512_cvtepi32_ps(sum);
        }
#elif defined(__AVX2__)
        auto a = cached[j];
        if constexpr (mask)
            a = _mm256_blend_epi32(a, cached[j + 1], mask);
        if constexpr (Bits % 2 == 0) {
            constexpr int rem = (G::base(0) + G::delta(Row)) % 32;
            if constexpr (rem <= 16)
                a = _mm256_srli_epi32(a, 16 - rem);
            else {
                auto b = cached[j + 1];
                a = _mm256_or_si256(_mm256_slli_epi32(a, rem - 16), _mm256_srli_epi32(b, 48 - rem));
            }
        } else {
            a = _mm256_sllv_epi32(
                _mm256_srlv_epi32(a, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(G::geometry.right[Row]))),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(G::geometry.left[Row])));
            if constexpr (G::geometry.crossing[Row]) {
                auto b = cached[j + 1];
                if constexpr (mask)
                    b = _mm256_blend_epi32(b, cached[j + 2], mask);
                a = _mm256_or_si256(
                    a,
                    _mm256_srlv_epi32(b, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(G::geometry.tail[Row]))));
            }
        }
        auto mixed = _mm256_mullo_epi32(_mm256_and_si256(a, _mm256_set1_epi32(65535)), _mm256_set1_epi32(0x83dcd12du));
        if constexpr (Integer)
            return mixed;
        else {
            auto sum = _mm256_madd_epi16(_mm256_maddubs_epi16(mixed, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
            return _mm256_cvtepi32_ps(sum);
        }
#else
        float v[Exl3Lanes];
        for (int c = 0; c < Exl3Lanes; ++c) {
            const int bit = (G::base(c) + G::delta(Row)) % (128 * Bits), at = bit / 32;
            const uint64_t pair = (uint64_t(words[at]) << 32) | words[(at + 1) % (4 * Bits)];
            const uint32_t m = uint32_t(uint16_t(pair >> (48 - bit % 32))) * 0x83dcd12du;
            v[c] = float((m & 255) + ((m >> 8) & 255) + ((m >> 16) & 255) + (m >> 24));
        }
        return Load(v);
#endif
    }
};

template <int Bits, int Half> inline void Exl3AffineAccTile(const uint32_t *words, const float *x, float *result) {
    auto acc = Load(result);
    Exl3AffineTile<Bits, Half> tile(words);
#define EXL3_AFFINE_ROW(R) acc = Madd(tile.template values<R>(), x[R], acc)
    EXL3_AFFINE_ROW(0);
    EXL3_AFFINE_ROW(1);
    EXL3_AFFINE_ROW(2);
    EXL3_AFFINE_ROW(3);
    EXL3_AFFINE_ROW(4);
    EXL3_AFFINE_ROW(5);
    EXL3_AFFINE_ROW(6);
    EXL3_AFFINE_ROW(7);
    EXL3_AFFINE_ROW(8);
    EXL3_AFFINE_ROW(9);
    EXL3_AFFINE_ROW(10);
    EXL3_AFFINE_ROW(11);
    EXL3_AFFINE_ROW(12);
    EXL3_AFFINE_ROW(13);
    EXL3_AFFINE_ROW(14);
    EXL3_AFFINE_ROW(15);
#undef EXL3_AFFINE_ROW
    Store(result, acc);
}
#if defined(__AVX2__) && !defined(__AVX512F__)
template <int Bits, bool Quant> inline void Exl3AccPair(const uint32_t *p, const void *x, void *out) {
    Exl3AffineTile<Bits, 0> lo(p);
    Exl3AffineTile<Bits, 8> hi(p);
    if constexpr (Quant) {
        int *dst = static_cast<int *>(out);
        const volatile int *in = static_cast<const volatile int *>(x);
        auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
        auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + 8));
#define EXL3_PAIR(R)                                                                                                   \
    {                                                                                                                  \
        const auto q = _mm256_set1_epi16(in[R]);                                                                       \
        a = _mm256_add_epi32(                                                                                          \
            a, _mm256_madd_epi16(_mm256_maddubs_epi16(lo.template values<R, true>(), _mm256_set1_epi8(1)), q));        \
        b = _mm256_add_epi32(                                                                                          \
            b, _mm256_madd_epi16(_mm256_maddubs_epi16(hi.template values<R, true>(), _mm256_set1_epi8(1)), q));        \
    }
        EXL3_PAIR(0);
        EXL3_PAIR(1);
        EXL3_PAIR(2);
        EXL3_PAIR(3);
        EXL3_PAIR(4);
        EXL3_PAIR(5);
        EXL3_PAIR(6);
        EXL3_PAIR(7);
        EXL3_PAIR(8);
        EXL3_PAIR(9);
        EXL3_PAIR(10);
        EXL3_PAIR(11);
        EXL3_PAIR(12);
        EXL3_PAIR(13);
        EXL3_PAIR(14);
        EXL3_PAIR(15);
#undef EXL3_PAIR
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), a);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8), b);
    } else {
        float *dst = static_cast<float *>(out);
        const float *in = static_cast<const float *>(x);
        auto a = Load(dst), b = Load(dst + 8);
#define EXL3_PAIR(R)                                                                                                   \
    a = Madd(lo.template values<R>(), in[R], a);                                                                       \
    b = Madd(hi.template values<R>(), in[R], b)
        EXL3_PAIR(0);
        EXL3_PAIR(1);
        EXL3_PAIR(2);
        EXL3_PAIR(3);
        EXL3_PAIR(4);
        EXL3_PAIR(5);
        EXL3_PAIR(6);
        EXL3_PAIR(7);
        EXL3_PAIR(8);
        EXL3_PAIR(9);
        EXL3_PAIR(10);
        EXL3_PAIR(11);
        EXL3_PAIR(12);
        EXL3_PAIR(13);
        EXL3_PAIR(14);
        EXL3_PAIR(15);
#undef EXL3_PAIR
        Store(dst, a);
        Store(dst + 8, b);
    }
}
#endif
template <int Bits>
void Exl3AffineDot(const uint8_t *packed, int outputs, int inputs, int first, const float *x, float *result,
                   int count) {
    double sum = 0;
    for (int i = 0; i < inputs; ++i)
        sum += x[i];
    const float correction = float(sum) * (1024.0f * half_to_float(0x1eee) + half_to_float(0xc931));
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    std::fill(result, result + count, 0.0f);
    for (int i = 0; i < inputs; i += 16)
        for (int c = 0; c < count; c += 16) {
            const auto *p = words + (size_t(i / 16) * (outputs / 16) + (first + c) / 16) * (4 * Bits);
#if defined(__AVX2__) && !defined(__AVX512F__)
            Exl3AccPair<Bits, false>(p, x + i, result + c);
#else
            Exl3AffineAccTile<Bits, 0>(p, x + i, result + c);
            if constexpr (Exl3Lanes <= 8)
                Exl3AffineAccTile<Bits, 8>(p, x + i, result + c + 8);
            if constexpr (Exl3Lanes <= 4) {
                Exl3AffineAccTile<Bits, 4>(p, x + i, result + c + 4);
                Exl3AffineAccTile<Bits, 12>(p, x + i, result + c + 12);
            }
            if constexpr (Exl3Lanes == 1) {
#define EXL3_AFFINE_COL(C) Exl3AffineAccTile<Bits, C>(p, x + i, result + c + C)
                EXL3_AFFINE_COL(1);
                EXL3_AFFINE_COL(2);
                EXL3_AFFINE_COL(3);
                EXL3_AFFINE_COL(5);
                EXL3_AFFINE_COL(6);
                EXL3_AFFINE_COL(7);
                EXL3_AFFINE_COL(9);
                EXL3_AFFINE_COL(10);
                EXL3_AFFINE_COL(11);
                EXL3_AFFINE_COL(13);
                EXL3_AFFINE_COL(14);
                EXL3_AFFINE_COL(15);
#undef EXL3_AFFINE_COL
            }
#endif
        }
    const float gain = half_to_float(0x1eee);
    for (int c = 0; c < count; ++c)
        result[c] = std::fma(result[c], gain, correction);
}

#if defined(__AVX512F__) || defined(__AVX2__)
template <int Bits, int Half> inline void Exl3QuantAccTile(const uint32_t *words, const int *x, int *result) {
    Exl3AffineTile<Bits, Half> tile(words);
#if defined(__AVX512F__)
    auto acc = _mm512_loadu_si512(result);
#if defined(__AVX512BW__)
#define EXL3_QUANT_ROW(R)                                                                                              \
    acc = _mm512_add_epi32(                                                                                            \
        acc, _mm512_madd_epi16(_mm512_maddubs_epi16(tile.template values<R, true>(), _mm512_set1_epi8(1)),             \
                               _mm512_set1_epi16(x[R])))
#else
#define EXL3_QUANT_ROW(R)                                                                                              \
    acc = _mm512_add_epi32(acc,                                                                                        \
                           _mm512_mullo_epi32(_mm512_cvtps_epi32(tile.template values<R>()), _mm512_set1_epi32(x[R])))
#endif
#else
    auto acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result));
#define EXL3_QUANT_ROW(R)                                                                                              \
    acc = _mm256_add_epi32(                                                                                            \
        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(tile.template values<R, true>(), _mm256_set1_epi8(1)),             \
                               _mm256_set1_epi16(reinterpret_cast<const volatile int *>(x)[R])))
#endif
    EXL3_QUANT_ROW(0);
    EXL3_QUANT_ROW(1);
    EXL3_QUANT_ROW(2);
    EXL3_QUANT_ROW(3);
    EXL3_QUANT_ROW(4);
    EXL3_QUANT_ROW(5);
    EXL3_QUANT_ROW(6);
    EXL3_QUANT_ROW(7);
    EXL3_QUANT_ROW(8);
    EXL3_QUANT_ROW(9);
    EXL3_QUANT_ROW(10);
    EXL3_QUANT_ROW(11);
    EXL3_QUANT_ROW(12);
    EXL3_QUANT_ROW(13);
    EXL3_QUANT_ROW(14);
    EXL3_QUANT_ROW(15);
#undef EXL3_QUANT_ROW
#if defined(__AVX512F__)
    _mm512_storeu_si512(result, acc);
#else
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result), acc);
#endif
}
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
// Quantize once per input block using the current rounding mode. The max and
// integer reductions are exact, so this preserves the scalar quantizer's bits.
// Vectorizing preparation matters when a worker owns only 16 output rows.
inline bool Exl3QuantizeBlock(const float *x, int *q, float &scale, int &sum) {
#if defined(__AVX512F__)
    auto maximum = _mm512_setzero_ps();
    for (int i = 0; i < 128; i += 16) {
        auto v = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(_mm512_loadu_ps(x + i)), _mm512_set1_epi32(0x7fffffff)));
        if (_mm512_cmp_ps_mask(v, _mm512_set1_ps(INFINITY), _CMP_NLT_UQ))
            return false;
        maximum = _mm512_max_ps(maximum, v);
    }
    const float peak = _mm512_reduce_max_ps(maximum);
#else
    auto maximum = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 8) {
        auto v = _mm256_and_ps(_mm256_loadu_ps(x + i), _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
        if (_mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_set1_ps(INFINITY), _CMP_NLT_UQ)))
            return false;
        maximum = _mm256_max_ps(maximum, v);
    }
    auto m = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    const float peak = _mm_cvtss_f32(m);
#endif
    scale = peak / 2047;
    if (peak == 0) {
        sum = 0;
        return true;
    }
    const float inv = 1.0f / scale;
    if (!std::isfinite(inv) || scale == 0)
        return false;
#if defined(__AVX512F__)
    auto total = _mm512_setzero_si512();
    for (int i = 0; i < 128; i += 16) {
        auto v = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(inv)));
        _mm512_storeu_si512(q + i, v);
        total = _mm512_add_epi32(total, v);
    }
    sum = _mm512_reduce_add_epi32(total);
#else
    auto total = _mm256_setzero_si256();
    for (int i = 0; i < 128; i += 8) {
        auto v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(inv)));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(q + i), v);
        total = _mm256_add_epi32(total, v);
    }
    auto t = _mm_add_epi32(_mm256_castsi256_si128(total), _mm256_extracti128_si256(total, 1));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtsi128_si32(t);
#endif
    return true;
}
#endif

template <int Bits>
void Exl3QuantDot(const uint8_t *packed, int outputs, int inputs, int first, const float *x, float *result, int count) {
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr int block = 128;
    alignas(64) int local[256];
    std::vector<int> large;
    if (count > 256)
        large.resize(count);
    int *acc = count <= 256 ? local : large.data();
    std::fill(result, result + count, 0.0f);
    const float gain = half_to_float(0x1eee), zero = 1024 * gain + half_to_float(0xc931);
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    for (int k0 = 0; k0 < inputs; k0 += block) {
        alignas(64) int q[block];
        int sum;
        float scale;
        if (!Exl3QuantizeBlock(x + k0, q, scale, sum))
            return Exl3AffineDot<Bits>(packed, outputs, inputs, first, x, result, count);
        if (scale == 0)
            continue;
        std::fill(acc, acc + count, 0);
        for (int i = 0; i < block; i += 16)
            for (int c = 0; c < count; c += 16) {
                const auto *p = words + (size_t((k0 + i) / 16) * (outputs / 16) + (first + c) / 16) * (4 * Bits);
#if defined(__AVX2__) && !defined(__AVX512F__)
                Exl3AccPair<Bits, true>(p, q + i, acc + c);
#else
                Exl3QuantAccTile<Bits, 0>(p, q + i, acc + c);
#endif
            }
        const float correction = float(sum) * scale * zero;
        for (int c = 0; c < count; ++c)
            result[c] += std::fma(float(acc[c]), scale * gain, correction);
    }
#else
    Exl3AffineDot<Bits>(packed, outputs, inputs, first, x, result, count);
#endif
}
