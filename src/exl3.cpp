#include "exl3.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <memory>
#include <vector>
#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace fastllm {
namespace {
const std::array<float, 1024> &SumCodebook() {
    static const auto table = [] {
        std::array<float, 1024> result{};
        const float gain = half_to_float(0x1eee), bias = half_to_float(0xc931);
        for (unsigned sum = 0; sum < result.size(); ++sum)
            result[sum] = half_to_float(float_to_half(std::fma(1024.0f + sum, gain, bias)));
        return result;
    }();
    return table;
}
// A shared exact FP16 codebook avoids repeated multiplication and conversion in
// the portable decoder. The table is immutable after C++ thread-safe creation.
const std::array<float, 65536> &Codebook() {
    static const auto table = [] {
        std::array<float, 65536> result{};
        const float gain = half_to_float(0x1eee), bias = half_to_float(0xc931);
        for (uint32_t state = 0; state < result.size(); ++state) {
            uint32_t mixed = state * 0x83dcd12du;
            unsigned sum = (mixed & 255) + ((mixed >> 8) & 255) +
                           ((mixed >> 16) & 255) + (mixed >> 24);
            result[state] = half_to_float(float_to_half(std::fma(1024.0f + sum, gain, bias)));
        }
        return result;
    }();
    return table;
}
bool ActivationTypeSupported(const Data &data) {
    return data.dataType==FLOAT32 || data.dataType==FLOAT16 || data.dataType==BFLOAT16;
}
// The public operators validate once, before entering per-element loops.
float ReadValue(const Data &data, size_t i) {
    if (data.dataType == FLOAT32) return reinterpret_cast<const float *>(data.cpuData)[i];
    uint16_t bits = reinterpret_cast<const uint16_t *>(data.cpuData)[i];
    if (data.dataType == FLOAT16) return half_to_float(bits);
    uint32_t full = uint32_t(bits) << 16; float value; std::memcpy(&value, &full, 4); return value;
}
void WriteValue(Data &data, size_t i, float value) {
    if (data.dataType == FLOAT32) { reinterpret_cast<float *>(data.cpuData)[i] = value; return; }
    uint16_t bits;
    if (data.dataType == FLOAT16) bits = float_to_half(value);
    else {
        uint32_t full; std::memcpy(&full, &value, 4);
        full += 0x7fff + ((full >> 16) & 1); bits = uint16_t(full >> 16);
    }
    reinterpret_cast<uint16_t *>(data.cpuData)[i] = bits;
}
}

bool Exl3BitRateSupported(int bits) {
    return (bits >= 2 && bits <= 16 && bits % 2 == 0) || bits == 3 || bits == 5 || bits == 7;
}
uint64_t Exl3PartBytes(int outputs, int inputs, int twiceBits) {
    AssertInFastLLM(outputs > 0 && inputs > 0 && outputs % 128 == 0 && inputs % 128 == 0 &&
                    Exl3BitRateSupported(twiceBits), "EXL3: invalid dimensions or bit rate.");
    return uint64_t(outputs) * inputs * twiceBits / 16 + uint64_t(inputs + outputs) * 2;
}
uint64_t Exl3StorageBytes(const Data &weight) {
    if (weight.dims.size() == 3) {
        AssertInFastLLM(weight.blockM == weight.dims[0], "EXL3: invalid grouped projection count.");
        return uint64_t(weight.blockM) * Exl3PartBytes(weight.dims[1], weight.dims[2], weight.blockK);
    }
    AssertInFastLLM(weight.dims.size() == 2 && weight.blockM > 0 &&
                    weight.dims[0] % weight.blockM == 0, "EXL3: invalid projection count.");
    return weight.blockM * Exl3PartBytes(weight.dims[0] / weight.blockM, weight.dims[1], weight.blockK);
}
void Exl3Hadamard128(float *values, size_t count) {
    AssertInFastLLM(count % 128 == 0, "EXL3: rotations require complete 128-value blocks.");
    constexpr float normalization = 0.08838834764831845f;
    for (size_t base = 0; base < count; base += 128) {
        float *block = values + base;
        for (unsigned stride = 1; stride < 128; stride *= 2) {
            for (unsigned first = 0; first < 128; first += 2 * stride)
                for (unsigned j = 0; j < stride; ++j) {
                    const float a = block[first + j], b = block[first + j + stride];
                    block[first + j] = a + b; block[first + j + stride] = a - b;
                }
        }
        for (unsigned j = 0; j < 128; ++j) block[j] *= normalization;
    }
}
namespace {
struct Window { unsigned first, second, shift; };
const std::array<Window, 256> &Windows(int twiceBits) {
    static const auto all = [] {
        std::array<std::array<Window, 256>, 17> result{};
        for (int bits = 2; bits <= 16; ++bits) for (int c = 0; c < 16; ++c) for (int r = 0; r < 16; ++r) {
            const int step = (r & 1) | ((r >> 3) << 1) | ((c >> 3) << 2) |
                             (((r >> 1) & 3) << 3) | ((c & 7) << 5);
            const int start = (((step + 1) * bits / 2) + 128 * bits - 16) % (128 * bits);
            result[bits][c * 16 + r] = {unsigned(start / 32), unsigned((start / 32 + 1) % (4 * bits)), unsigned(48 - start % 32)};
        }
        return result;
    }();
    return all[twiceBits];
}
void DecodeOutputRange(const uint8_t *packed, int outputs, int inputs, int twiceBits,
                       int first, int count, float *decoded) {
    const auto &table = Codebook();
    const auto &windows = Windows(twiceBits);
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    for (int inTile = 0; inTile < inputs / 16; ++inTile)
        for (int outTile = first / 16; outTile < (first + count) / 16; ++outTile) {
            const uint32_t *tile = words + (size_t(inTile) * (outputs / 16) + outTile) * (4 * twiceBits);
            for (int c = 0; c < 16; ++c) {
                float *row = decoded + size_t(outTile * 16 + c - first) * inputs + inTile * 16;
                for (int r = 0; r < 16; ++r) {
                    const auto &window = windows[c * 16 + r];
                    const uint64_t pair = (uint64_t(tile[window.first]) << 32) | tile[window.second];
                    row[r] = table[uint16_t(pair >> window.shift)];
                }
            }
        }
}
struct Exl3Task : MultiThreadBaseOp {
    std::function<void()> body;
    explicit Exl3Task(std::function<void()> body) : body(std::move(body)) {}
    void Run() override { body(); }
};

// Vector lanes are output columns. This lets a decoded vector serve several
// activation rows, without horizontal reductions or an expanded FP32 matrix.
#if defined(__AVX512F__)
constexpr int Exl3Lanes = 16;
using Exl3Vec = __m512;
inline Exl3Vec Load(const float *p) { return _mm512_loadu_ps(p); }
inline void Store(float *p, Exl3Vec v) { _mm512_storeu_ps(p, v); }
inline Exl3Vec Zero() { return _mm512_setzero_ps(); }
inline Exl3Vec Madd(Exl3Vec w, float x, Exl3Vec y) { return _mm512_fmadd_ps(w, _mm512_set1_ps(x), y); }
#elif defined(__AVX2__)
constexpr int Exl3Lanes = 8;
using Exl3Vec = __m256;
inline Exl3Vec Load(const float *p) { return _mm256_loadu_ps(p); }
inline void Store(float *p, Exl3Vec v) { _mm256_storeu_ps(p, v); }
inline Exl3Vec Zero() { return _mm256_setzero_ps(); }
inline Exl3Vec Madd(Exl3Vec w, float x, Exl3Vec y) {
#ifdef __FMA__
    return _mm256_fmadd_ps(w, _mm256_set1_ps(x), y);
#else
    return _mm256_add_ps(y, _mm256_mul_ps(w, _mm256_set1_ps(x)));
#endif
}
#elif defined(__aarch64__)
constexpr int Exl3Lanes = 4;
using Exl3Vec = float32x4_t;
inline Exl3Vec Load(const float *p) { return vld1q_f32(p); }
inline void Store(float *p, Exl3Vec v) { vst1q_f32(p, v); }
inline Exl3Vec Zero() { return vdupq_n_f32(0); }
inline Exl3Vec Madd(Exl3Vec w, float x, Exl3Vec y) { return vfmaq_n_f32(y, w, x); }
#elif defined(__SSE2__)
constexpr int Exl3Lanes = 4;
using Exl3Vec = __m128;
inline Exl3Vec Load(const float *p) { return _mm_loadu_ps(p); }
inline void Store(float *p, Exl3Vec v) { _mm_storeu_ps(p, v); }
inline Exl3Vec Zero() { return _mm_setzero_ps(); }
inline Exl3Vec Madd(Exl3Vec w, float x, Exl3Vec y) { return _mm_add_ps(y, _mm_mul_ps(w, _mm_set1_ps(x))); }
#else
constexpr int Exl3Lanes = 1;
using Exl3Vec = float;
inline Exl3Vec Load(const float *p) { return *p; }
inline void Store(float *p, Exl3Vec v) { *p = v; }
inline Exl3Vec Zero() { return 0; }
inline Exl3Vec Madd(Exl3Vec w, float x, Exl3Vec y) { return w * x + y; }
#endif

// Transposed extraction windows, shared by all tiles of one bit rate.
struct LaneWindows {
    alignas(64) uint32_t first[16], second[16], right[16], left[16], tail[16];
};
const std::array<LaneWindows, 16> &VectorWindows(int bits) {
    static const auto all = [] {
        std::array<std::array<LaneWindows, 16>, 17> result{};
        for (int bits = 2; bits <= 16; ++bits) {
            const auto &windows = Windows(bits);
            for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
                auto &v = result[bits][r]; const auto &w = windows[c * 16 + r];
                v.first[c] = w.first; v.second[c] = w.second;
                v.right[c] = w.shift >= 32 ? w.shift - 32 : 0;
                v.left[c] = w.shift < 32 ? 32 - w.shift : 0;
                v.tail[c] = w.shift; // SIMD shifts >= 32 produce zero.
            }
        }
        return result;
    }();
    return all[bits];
}
struct PackedTile {
    const uint32_t *words;
#if defined(__AVX512F__)
    __m512i low, high, upperLow, upperHigh;
    bool upper;
#endif
    PackedTile(const uint32_t *words, int bits) : words(words) {
#if defined(__AVX512F__)
        const int count = 4 * bits;
        low = _mm512_maskz_loadu_epi32(count >= 16 ? 0xffff : (1u << count) - 1, words);
        high = count > 16 ? _mm512_maskz_loadu_epi32(count >= 32 ? 0xffff : (1u << (count - 16)) - 1, words + 16) : _mm512_setzero_si512();
        upper = count > 32;
        if (upper) {
            upperLow = _mm512_maskz_loadu_epi32(count >= 48 ? 0xffff : (1u << (count - 32)) - 1, words + 32);
            upperHigh = count > 48 ? _mm512_maskz_loadu_epi32(count >= 64 ? 0xffff : (1u << (count - 48)) - 1, words + 48) : _mm512_setzero_si512();
        }
#endif
    }
#if defined(__AVX512F__)
    __m512i Get(__m512i index) const {
        auto value = _mm512_permutex2var_epi32(low, index, high);
        if (upper) value = _mm512_mask_blend_epi32(_mm512_cmpge_epi32_mask(index, _mm512_set1_epi32(32)), value,
                                                  _mm512_permutex2var_epi32(upperLow, index, upperHigh));
        return value;
    }
#endif
};
inline Exl3Vec DecodeColumns(const PackedTile &tile, const LaneWindows &v, int c, const float *table) {
#if defined(__AVX512F__)
    auto a = tile.Get(_mm512_loadu_si512(v.first + c));
    auto b = tile.Get(_mm512_loadu_si512(v.second + c));
    a = _mm512_sllv_epi32(_mm512_srlv_epi32(a, _mm512_loadu_si512(v.right + c)), _mm512_loadu_si512(v.left + c));
    b = _mm512_srlv_epi32(b, _mm512_loadu_si512(v.tail + c));
    auto state = _mm512_and_si512(_mm512_or_si512(a, b), _mm512_set1_epi32(65535));
    auto mixed = _mm512_mullo_epi32(state, _mm512_set1_epi32(0x83dcd12du));
#if defined(__AVX512VNNI__)
    auto sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mixed, _mm512_set1_epi32(0x01010101));
#else
    auto pairs = _mm512_add_epi32(_mm512_and_si512(mixed, _mm512_set1_epi32(0x00ff00ff)),
                                  _mm512_and_si512(_mm512_srli_epi32(mixed, 8), _mm512_set1_epi32(0x00ff00ff)));
    auto sum = _mm512_and_si512(_mm512_add_epi32(pairs, _mm512_srli_epi32(pairs, 16)), _mm512_set1_epi32(65535));
#endif
    // Regenerate the exact FP16 code instead of a dependent gather from L1.
    // Retain the format's half rounding (including ties), unlike an affine
    // approximation that moves the codebook gain outside the dot product.
    auto code = _mm512_fmadd_ps(_mm512_add_ps(_mm512_cvtepi32_ps(sum), _mm512_set1_ps(1024.0f)),
        _mm512_set1_ps(half_to_float(0x1eee)), _mm512_set1_ps(half_to_float(0xc931)));
    return _mm512_cvtph_ps(_mm512_cvtps_ph(code, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(__AVX2__)
    auto a = _mm256_i32gather_epi32(reinterpret_cast<const int *>(tile.words), _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.first + c)), 4);
    auto b = _mm256_i32gather_epi32(reinterpret_cast<const int *>(tile.words), _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.second + c)), 4);
    a = _mm256_sllv_epi32(_mm256_srlv_epi32(a, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.right + c))), _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.left + c)));
    b = _mm256_srlv_epi32(b, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.tail + c)));
    auto state = _mm256_and_si256(_mm256_or_si256(a, b), _mm256_set1_epi32(65535));
    auto mixed = _mm256_mullo_epi32(state, _mm256_set1_epi32(0x83dcd12du));
    auto pairs = _mm256_add_epi32(_mm256_and_si256(mixed, _mm256_set1_epi32(0x00ff00ff)),
                                  _mm256_and_si256(_mm256_srli_epi32(mixed, 8), _mm256_set1_epi32(0x00ff00ff)));
    auto sum = _mm256_and_si256(_mm256_add_epi32(pairs, _mm256_srli_epi32(pairs, 16)), _mm256_set1_epi32(65535));
#if defined(__F16C__) && defined(__FMA__)
    auto code = _mm256_fmadd_ps(_mm256_add_ps(_mm256_cvtepi32_ps(sum), _mm256_set1_ps(1024.0f)),
        _mm256_set1_ps(half_to_float(0x1eee)), _mm256_set1_ps(half_to_float(0xc931)));
    return _mm256_cvtph_ps(_mm256_cvtps_ph(code, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
    return _mm256_i32gather_ps(table, sum, 4);
#endif
#else
    float values[Exl3Lanes];
    for (int j = 0; j < Exl3Lanes; ++j) {
        const int lane = c + j;
        const uint64_t pair = (uint64_t(tile.words[v.first[lane]]) << 32) | tile.words[v.second[lane]];
        const uint32_t mixed = uint32_t(uint16_t(pair >> v.tail[lane])) * 0x83dcd12du;
        values[j] = table[(mixed & 255) + ((mixed >> 8) & 255) + ((mixed >> 16) & 255) + (mixed >> 24)];
    }
    return Load(values);
#endif
}

// Compile-time extraction geometry removes dynamic window tables and the
// high-word fallback for low bit rates; arithmetic and accumulation order
// remain identical to the portable decoder.
template<int Bits, int R>
inline Exl3Vec DecodeFixedRow(const PackedTile &tile, int c, const float *table) {
    constexpr LaneWindows window = [] {
        LaneWindows v{};
        for (int col = 0; col < 16; ++col) {
            const int step = (R & 1) | ((R >> 3) << 1) | ((col >> 3) << 2) |
                             (((R >> 1) & 3) << 3) | ((col & 7) << 5);
            const int start = (((step + 1) * Bits / 2) + 128 * Bits - 16) % (128 * Bits);
            const int shift = 48 - start % 32;
            v.first[col] = start / 32; v.second[col] = (start / 32 + 1) % (4 * Bits);
            v.right[col] = shift >= 32 ? shift - 32 : 0;
            v.left[col] = shift < 32 ? 32 - shift : 0;
            v.tail[col] = shift;
        }
        return v;
    }();
    return DecodeColumns(tile, window, c, table);
}
template<int Bits>
void DecodeDotSingle(const uint8_t *packed, int outputs, int inputs,
                     int first, const float *x, float *result, int count = 128) {
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    const float *table = SumCodebook().data();
    for (int c = 0; c < count; c += Exl3Lanes) {
        auto acc = Zero();
        for (int i = 0; i < inputs; i += 16) {
            PackedTile tile(words + (size_t(i / 16) * (outputs / 16) + (first + c) / 16) * (4 * Bits), Bits);
            acc = Madd(DecodeFixedRow<Bits, 0>(tile, c % 16, table), x[i + 0], acc);
            acc = Madd(DecodeFixedRow<Bits, 1>(tile, c % 16, table), x[i + 1], acc);
            acc = Madd(DecodeFixedRow<Bits, 2>(tile, c % 16, table), x[i + 2], acc);
            acc = Madd(DecodeFixedRow<Bits, 3>(tile, c % 16, table), x[i + 3], acc);
            acc = Madd(DecodeFixedRow<Bits, 4>(tile, c % 16, table), x[i + 4], acc);
            acc = Madd(DecodeFixedRow<Bits, 5>(tile, c % 16, table), x[i + 5], acc);
            acc = Madd(DecodeFixedRow<Bits, 6>(tile, c % 16, table), x[i + 6], acc);
            acc = Madd(DecodeFixedRow<Bits, 7>(tile, c % 16, table), x[i + 7], acc);
            acc = Madd(DecodeFixedRow<Bits, 8>(tile, c % 16, table), x[i + 8], acc);
            acc = Madd(DecodeFixedRow<Bits, 9>(tile, c % 16, table), x[i + 9], acc);
            acc = Madd(DecodeFixedRow<Bits, 10>(tile, c % 16, table), x[i + 10], acc);
            acc = Madd(DecodeFixedRow<Bits, 11>(tile, c % 16, table), x[i + 11], acc);
            acc = Madd(DecodeFixedRow<Bits, 12>(tile, c % 16, table), x[i + 12], acc);
            acc = Madd(DecodeFixedRow<Bits, 13>(tile, c % 16, table), x[i + 13], acc);
            acc = Madd(DecodeFixedRow<Bits, 14>(tile, c % 16, table), x[i + 14], acc);
            acc = Madd(DecodeFixedRow<Bits, 15>(tile, c % 16, table), x[i + 15], acc);
        }
        Store(result + c, acc);
    }
}
template<int Rows>
void DecodeDot(const uint8_t *packed, int outputs, int inputs, int bits,
               int first, const float *x, float *result, int count = 128) {
    const auto &windows = VectorWindows(bits);
    const float *table = SumCodebook().data();
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    for (int c = 0; c < count; c += Exl3Lanes) {
        Exl3Vec acc[Rows];
        for (int row = 0; row < Rows; ++row) acc[row] = Zero();
        for (int i = 0; i < inputs; i += 16) {
            PackedTile tile(words + (size_t(i / 16) * (outputs / 16) + (first + c) / 16) * (4 * bits), bits);
            for (int r = 0; r < 16; ++r) {
                auto w = DecodeColumns(tile, windows[r], c % 16, table);
                for (int row = 0; row < Rows; ++row) acc[row] = Madd(w, x[size_t(row) * inputs + i + r], acc[row]);
            }
        }
        for (int row = 0; row < Rows; ++row) Store(result + row * 128 + c, acc[row]);
    }
}
template<int Rows>
void MultiplyTile(const float *x, int inputs, int firstInput, const float *w, float *result) {
    for (int c = 0; c < 128; c += Exl3Lanes) {
        Exl3Vec acc[Rows];
        for (int row = 0; row < Rows; ++row) acc[row] = Load(result + row * 128 + c);
        for (int k = 0; k < 32; ++k) {
            auto weight = Load(w + k * 128 + c);
            for (int row = 0; row < Rows; ++row) acc[row] = Madd(weight, x[size_t(row) * inputs + firstInput + k], acc[row]);
        }
        for (int row = 0; row < Rows; ++row) Store(result + row * 128 + c, acc[row]);
    }
}
void DecodeMatmul(const uint8_t *packed, int outputs, int inputs, int bits,
                  int first, const float *x, int rows, float *result, int count = 128) {
    switch (rows) {
        case 1:
            switch (bits) {
                case 2: return DecodeDotSingle<2>(packed, outputs, inputs, first, x, result, count);
                case 3: return DecodeDotSingle<3>(packed, outputs, inputs, first, x, result, count);
                case 4: return DecodeDotSingle<4>(packed, outputs, inputs, first, x, result, count);
                case 5: return DecodeDotSingle<5>(packed, outputs, inputs, first, x, result, count);
                case 6: return DecodeDotSingle<6>(packed, outputs, inputs, first, x, result, count);
                case 7: return DecodeDotSingle<7>(packed, outputs, inputs, first, x, result, count);
                case 8: return DecodeDotSingle<8>(packed, outputs, inputs, first, x, result, count);
                case 10: return DecodeDotSingle<10>(packed, outputs, inputs, first, x, result, count);
                case 12: return DecodeDotSingle<12>(packed, outputs, inputs, first, x, result, count);
                case 14: return DecodeDotSingle<14>(packed, outputs, inputs, first, x, result, count);
                case 16: return DecodeDotSingle<16>(packed, outputs, inputs, first, x, result, count);
            }
            return DecodeDot<1>(packed, outputs, inputs, bits, first, x, result, count);
        case 2: return DecodeDot<2>(packed, outputs, inputs, bits, first, x, result, count);
        case 3: return DecodeDot<3>(packed, outputs, inputs, bits, first, x, result, count);
        case 4: return DecodeDot<4>(packed, outputs, inputs, bits, first, x, result, count);
    }
    const auto &windows = VectorWindows(bits);
    const float *table = SumCodebook().data();
    const auto *words = reinterpret_cast<const uint32_t *>(packed);
    alignas(64) float tileWeights[32 * 128];
    std::fill(result, result + size_t(rows) * 128, 0.0f);
    for (int i = 0; i < inputs; i += 32) {
        for (int k = 0; k < 32; k += 16) for (int c = 0; c < 128; c += 16) {
            PackedTile tile(words + (size_t((i + k) / 16) * (outputs / 16) + (first + c) / 16) * (4 * bits), bits);
            for (int r = 0; r < 16; ++r) for (int j = 0; j < 16; j += Exl3Lanes)
                Store(tileWeights + (k + r) * 128 + c + j, DecodeColumns(tile, windows[r], j, table));
        }
        int row = 0;
        for (; row + 7 < rows; row += 8) MultiplyTile<8>(x + size_t(row) * inputs, inputs, i, tileWeights, result + row * 128);
        for (; row < rows; ++row) MultiplyTile<1>(x + size_t(row) * inputs, inputs, i, tileWeights, result + row * 128);
    }
}
}
namespace {
struct Exl3ShardView {
    int outputs, inputs, nodes;
    size_t partBytes, packedBytes;
    explicit Exl3ShardView(const Data &w, int nodes)
        : outputs(w.dims[0] / w.blockM), inputs(w.dims[1]), nodes(nodes),
          partBytes(Exl3PartBytes(outputs, inputs, w.blockK) / nodes),
          packedBytes(uint64_t(outputs) * inputs * w.blockK / 16 / nodes) {}
};
template<class Apply> void Exl3Scales(const Data &w, int part, int first, bool input, Apply apply) {
    const int nodes = w.numasData.empty() ? 1 : w.numasData.size();
    const Exl3ShardView v(w, nodes);
    const int count = (input ? v.inputs : v.outputs) / nodes;
    for (int c = 0; c < 128;) {
        const int index = first + c, node = index / count, offset = index % count;
        const auto *base = w.numasData.empty() ? w.cpuData : w.numasData[node];
        const auto *scales = reinterpret_cast<const uint16_t *>(base + part * v.partBytes + v.packedBytes)
            + (input ? 0 : v.inputs / nodes) + offset;
        const int length = std::min(128 - c, count - offset);
        for (int j = 0; j < length; ++j) apply(c + j, half_to_float(scales[j]));
        c += length;
    }
}
}
void Exl3CopyRowShard(const Data &weight, uint8_t *destination, int node, int nodes) {
    const Exl3ShardView v(weight, nodes);
    AssertInFastLLM(weight.cpuData && nodes > 0 && node >= 0 && node < nodes &&
        v.outputs % (16 * nodes) == 0 && v.inputs % (16 * nodes) == 0,
        "EXL3: invalid row shard");
    const size_t line = size_t(v.outputs / nodes) * weight.blockK;
    for (int part = 0; part < weight.blockM; ++part) {
        const auto *source = weight.cpuData + part * v.partBytes * nodes;
        auto *dst = destination + part * v.partBytes;
        for (int tile = 0; tile < v.inputs / 16; ++tile)
            std::memcpy(dst + tile * line, source + (size_t(tile) * nodes + node) * line, line);
        const auto *scales = source + v.packedBytes * nodes;
        std::memcpy(dst + v.packedBytes, scales + node * (v.inputs / nodes) * 2, v.inputs / nodes * 2);
        std::memcpy(dst + v.packedBytes + v.inputs / nodes * 2,
            scales + v.inputs * 2 + node * (v.outputs / nodes) * 2, v.outputs / nodes * 2);
    }
}
void Exl3CopyCanonical(const Data &weight, uint8_t *destination) {
    if (weight.cpuData) { std::memcpy(destination, weight.cpuData, weight.GetBytes()); return; }
    const int nodes = weight.numasData.size();
    AssertInFastLLM(nodes > 0, "EXL3: no host storage");
    const Exl3ShardView v(weight, nodes);
    const size_t line = size_t(v.outputs / nodes) * weight.blockK;
    for (int part = 0; part < weight.blockM; ++part) for (int node = 0; node < nodes; ++node) {
        const auto *source = weight.numasData[node] + part * v.partBytes;
        auto *dst = destination + part * v.partBytes * nodes;
        for (int tile = 0; tile < v.inputs / 16; ++tile)
            std::memcpy(dst + (size_t(tile) * nodes + node) * line, source + tile * line, line);
        auto *scales = dst + v.packedBytes * nodes;
        std::memcpy(scales + node * (v.inputs / nodes) * 2, source + v.packedBytes, v.inputs / nodes * 2);
        std::memcpy(scales + v.inputs * 2 + node * (v.outputs / nodes) * 2,
            source + v.packedBytes + v.inputs / nodes * 2, v.outputs / nodes * 2);
    }
}
void Exl3PrepareInputBlock(const Data &weight, int part, const float *input,
                          float *rotated, int first, bool roundBfloat) {
    Exl3Scales(weight, part, first, true, [&](int c, float scale) {
        const float x = roundBfloat ? RoundFloat32ToBFloat16RNE(input[first + c]) : input[first + c];
        rotated[first + c] = x * scale;
    });
    Exl3Hadamard128(rotated + first, 128);
    for (int c = 0; c < 128; ++c) rotated[first + c] = half_to_float(float_to_half(rotated[first + c]));
}
void Exl3DotRows(const Data &weight, int part, int node, int nodes,
                const float *rotated, float *output, int first, int count) {
    const int storedNodes = weight.numasData.empty() ? 1 : nodes;
    const Exl3ShardView v(weight, storedNodes);
    const int columns = v.outputs / storedNodes;
    const auto *packed = weight.numasData.empty() ? weight.cpuData : weight.numasData[node];
    const int start = weight.numasData.empty() ? node * (v.outputs / nodes) + first : first;
    DecodeMatmul(packed + part * v.partBytes, columns, v.inputs, weight.blockK,
        start, rotated, 1, output, count);
}
void Exl3FinishOutputBlock(const Data &weight, int part, float *output, int first) {
    Exl3Hadamard128(output + first, 128);
    Exl3Scales(weight, part, first, false, [&](int c, float scale) {
        output[first + c] = RoundFloat32ToBFloat16RNE(output[first + c] * scale);
    });
}
void Exl3DecodeInner(const uint8_t *packed, int outputs, int inputs, int twiceBits, float *decoded) {
    Exl3PartBytes(outputs, inputs, twiceBits);
    DecodeOutputRange(packed, outputs, inputs, twiceBits, 0, outputs, decoded);
}
void Exl3Reconstruct(const uint8_t *packed, int outputs, int inputs, int twiceBits, float *decoded) {
    Exl3DecodeInner(packed, outputs, inputs, twiceBits, decoded);
    const size_t bytes = uint64_t(outputs) * inputs * twiceBits / 16;
    const uint16_t *inScale = reinterpret_cast<const uint16_t *>(packed + bytes);
    const uint16_t *outScale = inScale + inputs;
    for (int c = 0; c < outputs; ++c) Exl3Hadamard128(decoded + size_t(c) * inputs, inputs);
    for (int first = 0; first < outputs; first += 128) {
        for (int r = 0; r < inputs; ++r) {
            float column[128];
            for (int c = 0; c < 128; ++c) column[c] = decoded[size_t(first + c) * inputs + r];
            Exl3Hadamard128(column, 128);
            for (int c = 0; c < 128; ++c)
                decoded[size_t(first + c) * inputs + r] = column[c] * half_to_float(inScale[r]) * half_to_float(outScale[first + c]);
        }
    }
}
namespace {
struct Projection {
    const Data *input, *weight, *bias;
    Data *output;
    int inputs, outputs, rows, part;
    const uint8_t *packed;
    const uint16_t *inScale, *outScale;
    std::vector<float> rotated;
    std::shared_ptr<std::vector<uint8_t>> canonical;
};
void AddProjections(const Data &input, const Data &weight, const Data &bias, Data &output,
                    std::vector<Projection> &plans) {
    AssertInFastLLM((weight.cpuData || !weight.numasData.empty()) && input.cpuData && weight.dataType == EXL3 &&
                    input.dims.back() == weight.dims[1] && ActivationTypeSupported(input) &&
                    ActivationTypeSupported(output) && (bias.dims.empty() || ActivationTypeSupported(bias)),
                    "EXL3: invalid CPU linear input, output or bias type.");
    Exl3StorageBytes(weight);
    const int inputs = weight.dims[1], totalOutputs = weight.dims[0];
    const int outputs = totalOutputs / weight.blockM, rows = input.Count(0) / inputs;
    const size_t partBytes = Exl3PartBytes(outputs, inputs, weight.blockK);
    const size_t trellisBytes = uint64_t(outputs) * inputs * weight.blockK / 16;
    output.Allocate(false);
    std::shared_ptr<std::vector<uint8_t>> canonical;
    if (!weight.cpuData) {
        canonical = std::make_shared<std::vector<uint8_t>>(weight.GetBytes());
        Exl3CopyCanonical(weight, canonical->data());
    }
    for (int part = 0; part < weight.blockM; ++part) {
        const auto *packed = (weight.cpuData ? weight.cpuData : canonical->data()) + part * partBytes;
        const auto *scale = reinterpret_cast<const uint16_t *>(packed + trellisBytes);
        plans.push_back({&input, &weight, &bias, &output, inputs, outputs, rows, part,
                         packed, scale, scale + inputs, std::vector<float>(size_t(rows) * inputs), canonical});
    }
}
void RotateProjectionRow(Projection &p, int row) {
                float *x = p.rotated.data() + size_t(row) * p.inputs;
                for (int i = 0; i < p.inputs; ++i) x[i] = ReadValue(*p.input, size_t(row) * p.inputs + i) * half_to_float(p.inScale[i]);
                Exl3Hadamard128(x, p.inputs);
                for (int i = 0; i < p.inputs; ++i) x[i] = half_to_float(float_to_half(x[i]));
}
void ProjectOutputBlock(Projection &p, int block, std::vector<float> &result) {
                result.resize(size_t(p.rows) * 128);
                const int firstOutput = block * 128;
                DecodeMatmul(p.packed, p.outputs, p.inputs, p.weight->blockK, firstOutput, p.rotated.data(), p.rows, result.data());
                for (int row = 0; row < p.rows; ++row) {
                    float *y = result.data() + size_t(row) * 128;
                    Exl3Hadamard128(y, 128);
                    for (int c = 0; c < 128; ++c) {
                        const int destination = p.part * p.outputs + firstOutput + c;
                        float value = y[c] * half_to_float(p.outScale[firstOutput + c]);
                        if (!p.bias->dims.empty()) value += ReadValue(*p.bias, destination);
                        WriteValue(*p.output, size_t(row) * p.weight->dims[0] + destination, value);
                    }
                }
}

void RunProjections(std::vector<Projection> &plans, const std::function<void()> &submitGpu = {}) {
    auto *pool = GetAlivePool();
    const int first = pool->curActivateThreadInterval.first;
    const int available = std::max(1, pool->curActivateThreadInterval.second - first);
    int totalRows = 0, totalBlocks = 0;
    for (const auto &p : plans) { totalRows += p.rows; totalBlocks += p.outputs / 128; }
    auto parallel = [&](int count, const std::function<void(int, int)> &body, bool submit = false) {
        const int threads = std::min(available, count);
        std::vector<std::unique_ptr<Exl3Task>> tasks;
        for (int worker = 0; worker < threads; ++worker) {
            tasks.emplace_back(new Exl3Task([&, worker] { body(worker, threads); }));
        }
        if (threads == 1) { if (submit && submitGpu) submitGpu(); tasks[0]->Run(); }
        else {
            for (int i = 0; i < threads; ++i) pool->PushOp(first + i, tasks[i].get());
            if (submit && submitGpu) submitGpu();
            for (int i = 0; i < threads; ++i) pool->Wait(first + i);
        }
    };
    if (plans.empty()) { if (submitGpu) submitGpu(); return; }
    parallel(totalRows, [&](int worker, int threads) {
        int offset = 0;
        for (auto &p : plans) {
            for (int row = (worker - offset % threads + threads) % threads; row < p.rows; row += threads) {
                RotateProjectionRow(p, row);
            }
            offset += p.rows;
        }
    });
    // Share workers across projections and, for decode, across routed experts.
    // Every worker writes complete, disjoint 128-column rotation blocks.
    parallel(totalBlocks, [&](int worker, int threads) {
        int offset = 0;
        std::vector<float> result;
        for (auto &p : plans) {
            const int blocks = p.outputs / 128;
            for (int block = (worker - offset % threads + threads) % threads; block < blocks; block += threads) {
                ProjectOutputBlock(p, block, result);
            }
            offset += blocks;
        }
    }, true);
}
}
void Exl3GroupedCPU(const Data &input, const Data &weight, Data &output) {
    AssertInFastLLM(weight.dims.size() == 3 && ActivationTypeSupported(input) &&
        ActivationTypeSupported(output), "EXL3: invalid grouped projection");
    Exl3StorageBytes(weight);
    const int groups = weight.dims[0], n = weight.dims[1], k = weight.dims[2];
    const int rows = input.Count(0) / (groups * k);
    AssertInFastLLM(input.Count(0) == uint64_t(rows) * groups * k &&
        output.Count(0) == uint64_t(rows) * groups * n, "EXL3: grouped dimension mismatch");
    Data empty;
    std::vector<Projection> plans;
    std::vector<std::unique_ptr<Data>> inputs, weights, outputs;
    const size_t partBytes = Exl3PartBytes(n, k, weight.blockK);
    for (int g = 0; g < groups; ++g) {
        inputs.emplace_back(new Data(input.dataType, {rows, k}));
        outputs.emplace_back(new Data(output.dataType, {rows, n}));
        weights.emplace_back(new Data());
        auto &x = *inputs.back(), &w = *weights.back(), &y = *outputs.back();
        x.Allocate(false);
        for (int row = 0; row < rows; ++row)
            std::memcpy(x.cpuData + size_t(row) * k * x.unitSize,
                input.cpuData + (size_t(row) * groups + g) * k * x.unitSize, size_t(k) * x.unitSize);
        w.FakeFrom(weight, g * partBytes); w.blockK = weight.blockK; w.blockM = 1; w.Resize({n, k});
        AddProjections(x, w, empty, y, plans);
    }
    RunProjections(plans);
    output.Allocate(false);
    for (int g = 0; g < groups; ++g) for (int row = 0; row < rows; ++row)
        std::memcpy(output.cpuData + (size_t(row) * groups + g) * n * output.unitSize,
            outputs[g]->cpuData + size_t(row) * n * output.unitSize, size_t(n) * output.unitSize);
}

void Exl3LinearCPU(const Data &input, const Data &weight, const Data &bias, Data &output) {
    std::vector<Projection> plans;
    AddProjections(input, weight, bias, output, plans);
    RunProjections(plans);
}
void Exl3MoeCPU(const Data &input, Data &output, const Data &index, const Data &score,
                Data **weights, int weightsBatch, float sharedScale, float limit) {
    const int rows = input.dims[0], k = input.dims[1], topk = index.dims.back();
    AssertInFastLLM(index.dataType == INT32 && score.dataType == FLOAT32 &&
                    ActivationTypeSupported(input) && ActivationTypeSupported(output),
                    "EXL3 MoE requires INT32 routes, F32 scores and floating activations.");
    const auto *ids = reinterpret_cast<const int32_t *>(index.cpuData);
    const auto *scores = reinterpret_cast<const float *>(score.cpuData);
    std::vector<float> sum(size_t(rows) * k, 0.0f);
    struct Expert {
        int id, width;
        std::vector<int> selected;
        std::vector<float> scales;
        Data gathered, middle, activated, result;
        Expert(int id, int width, int k, DataType type, std::vector<int> selected, std::vector<float> scales)
            : id(id), width(width), selected(std::move(selected)), scales(std::move(scales)),
              gathered(type, {int(this->selected.size()), k}),
              middle(type, {int(this->selected.size()), 2 * width}),
              activated(type, {int(this->selected.size()), width}),
              result(type, {int(this->selected.size()), k}) {}
    };
    std::vector<std::unique_ptr<Expert>> pending;
    auto run = [&] {
        Data empty;
        std::vector<Projection> plans;
        for (auto &e : pending) AddProjections(e->gathered, *weights[2 * e->id], empty, e->middle, plans);
        RunProjections(plans);
        plans.clear();
        for (auto &e : pending) {
            const int count = e->selected.size(), width = e->width;
            e->activated.Allocate(false);
            for (int row = 0; row < count; ++row) for (int c = 0; c < width; ++c) {
                float gate = ReadValue(e->middle, size_t(row) * 2 * width + c);
                float up = ReadValue(e->middle, size_t(row) * 2 * width + width + c);
                if (limit > 0) { gate = std::min(gate, limit); up = std::max(-limit, std::min(up, limit)); }
                WriteValue(e->activated, size_t(row) * width + c, gate / (1.0f + std::exp(-gate)) * up);
            }
            AddProjections(e->activated, *weights[2 * e->id + 1], empty, e->result, plans);
        }
        RunProjections(plans);
        for (auto &e : pending)
            for (int row = 0; row < int(e->selected.size()); ++row) for (int c = 0; c < k; ++c)
                sum[size_t(e->selected[row]) * k + c] += e->scales[row] * ReadValue(e->result, size_t(row) * k + c);
        pending.clear();
    };
    for (int expert = 0; expert < weightsBatch / 2; ++expert) {
        if (!weights[2 * expert] || !weights[2 * expert + 1]) continue;
        std::vector<int> selected;
        std::vector<float> scales;
        for (int row = 0; row < rows; ++row) {
            if (expert == 0 && sharedScale != 0) { selected.push_back(row); scales.push_back(sharedScale); }
            if (expert != 0) for (int j = 0; j < topk; ++j)
                if (ids[row * topk + j] == expert - 1) {
                    selected.push_back(row); scales.push_back(scores[row * topk + j]);
                }
        }
        if (selected.empty()) continue;
        const int count = selected.size(), width = weights[2 * expert]->dims[0] / 2;
        pending.emplace_back(new Expert(expert, width, k, input.dataType, std::move(selected), std::move(scales)));
        auto &e = *pending.back();
        e.gathered.Allocate(false);
        for (int row = 0; row < count; ++row)
            std::memcpy(e.gathered.cpuData + size_t(row) * k * input.unitSize,
                        input.cpuData + size_t(e.selected[row]) * k * input.unitSize, size_t(k) * input.unitSize);
        // Bound prefill scratch independently of the number of routed experts.
        // Small batches share the worker pool across experts to fill idle lanes.
        if (rows > 4) run();
    }
    if (!pending.empty()) run();
    output.Allocate(false);
    for (size_t i = 0; i < sum.size(); ++i) WriteValue(output, i, sum[i]);
}

void Exl3MoeCPUExperts(const float *input, float *output, int rows, Data **weights,
    const int32_t *indices, const int32_t *gpuIndices, const float *scores,
    int topk, float limit, const std::function<void()> &submitGpu) {
    AssertInFastLLM(weights && weights[2] && rows > 0 && topk > 0,
        "EXL3 host experts: invalid geometry");
    const int k = weights[2]->dims[1], mid = weights[2]->dims[0] / 2;
    struct Route {
        int at; Data x, gate, activated, y;
        Route(int at, int k, int mid) : at(at), x(BFLOAT16, {1,k}),
            gate(BFLOAT16, {1,2*mid}), activated(BFLOAT16, {1,mid}), y(BFLOAT16, {1,k}) {}
    };
    Data empty;
    std::vector<std::unique_ptr<Route>> routes;
    std::vector<Projection> plans;
    for (int at = 0; at < rows * topk; ++at) {
        if (indices[at] < 0 || (gpuIndices && gpuIndices[at] >= 0)) continue;
        auto route = std::make_unique<Route>(at, k, mid);
        route->x.Allocate(false);
        for (int c = 0; c < k; ++c) WriteValue(route->x, c, input[size_t(at/topk)*k+c]);
        AddProjections(route->x, *weights[2+2*indices[at]], empty, route->gate, plans);
        routes.push_back(std::move(route));
    }
    RunProjections(plans, submitGpu);
    plans.clear();
    for (auto &r : routes) {
        r->activated.Allocate(false);
        for (int c = 0; c < mid; ++c) {
            float g = ReadValue(r->gate,c), u = ReadValue(r->gate,mid+c);
            if (limit > 0) { g = std::min(g,limit); u = std::max(-limit,std::min(u,limit)); }
            WriteValue(r->activated,c,g/(1.0f+std::exp(-g))*u);
        }
        AddProjections(r->activated,*weights[3+2*indices[r->at]],empty,r->y,plans);
    }
    RunProjections(plans);
    for (const auto &r : routes) for (int c = 0; c < k; ++c)
        output[size_t(r->at)*k+c] = ReadValue(r->y,c) * (scores ? scores[r->at] : 1.0f);
}

}
