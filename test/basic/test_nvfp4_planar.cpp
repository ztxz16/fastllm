#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastllm {
    CPUInstructInfo *GetCPUInstructInfo();
    bool FastllmGemmBFloat16NVFP4Block16_AVX2(
        const void *, long, const void *, long, void *, long, int, int, int, int, int);
    bool FastllmGemmBFloat16NVFP4Block16E4M3Packed_AVX2(
        const void *, long, const void *, long, void *, long, int, int, int, int, int);
}

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void Run(int columns, bool crossSwiglu, bool useAvx512) {
    using namespace fastllm;
    constexpr int rows = 64;
    const int blocks = (columns + 15) / 16;
    std::mt19937 rng(781);
    std::vector<uint8_t> packed(GetNVFP4WeightBytes(rows, columns));
    std::vector<uint8_t> scales(rows * blocks);
    for (auto &v : packed) v = rng();
    for (auto &v : scales) v = 0x28 + rng() % 24;
    const std::vector<float> globals{0.73f, 1.31f};
    const size_t bytes = GetDataBytes(NVFP4_BLOCK_16, rows, columns);
    Require(bytes == GetDataBytes(NVFP4_BLOCK_16_PLANAR, rows, columns),
            "planar storage increased allocation");
    std::vector<uint8_t> legacy(bytes), planar(bytes);
    PackCompactE4M3NVFP4Block16Rows(rows, columns, packed.data(), scales.data(), globals,
        1, 16, legacy.data(), 0, rows, crossSwiglu);
    PackCompactE4M3NVFP4Block16Rows(rows, columns, packed.data(), scales.data(), globals,
        1, 16, planar.data(), 0, rows, crossSwiglu, true);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            const uint8_t *old = legacy.data() + (size_t(row) * blocks + block) * 12;
            Require(std::memcmp(old, planar.data() + NVFP4PlanarWeightOffset(row, blocks, block), 8) == 0,
                    "planar packing changed weight bytes");
            Require(std::memcmp(old + 8, planar.data() + NVFP4PlanarScaleOffset(row, blocks, block), 4) == 0,
                    "planar packing changed scale bits");
        }
    }
    auto *info = GetCPUInstructInfo();
    const bool originalAvx512 = info->hasAVX512BF16;
    info->hasAVX512BF16 = originalAvx512 && useAvx512;
    for (int batch : {1, 4, 32, 65}) {
        if (batch > 31 && columns % 32 != 0) continue; // BF16 GEMM requires complete vectors.
        std::vector<float> input(batch * columns);
        for (auto &v : input) v = (int(rng() % 31) - 15) / 32.0f;
        std::vector<uint16_t> bf16(input.size());
        for (size_t i = 0; i < input.size(); ++i) bf16[i] = Float32ToBFloat16RNEBits(input[i]);
        for (bool useBf16 : {false, true}) {
            const void *activation = useBf16 ? static_cast<void *>(bf16.data()) : input.data();
            const DataType atype = useBf16 ? BFLOAT16 : FLOAT32;
            std::vector<float> expected(batch * rows, -1234), actual(expected);
            for (int part = 0; part < 3; ++part) {
                // Partitions straddle both a row tile and an output SIMD group.
                const int edges[]{0, 19, 37, rows};
                FastllmGemm(batch, columns, rows, activation, GetDataBytes(atype, 1, columns),
                    legacy.data(), GetDataBytes(NVFP4_BLOCK_16, 1, columns), expected.data(), rows * sizeof(float),
                    edges[part], edges[part + 1], atype, NVFP4_BLOCK_16, FLOAT32);
                FastllmGemm(batch, columns, rows, activation, GetDataBytes(atype, 1, columns),
                    planar.data(), GetDataBytes(NVFP4_BLOCK_16_PLANAR, 1, columns), actual.data(), rows * sizeof(float),
                    edges[part], edges[part + 1], atype, NVFP4_BLOCK_16_PLANAR, FLOAT32);
            }
            // AVX2's interleaved kernel and the generic planar kernel use
            // different FP32 reduction orders. Compare accuracy, not bits.
            double error2 = 0, norm2 = 0;
            for (size_t i = 0; i < actual.size(); i++) {
                const double error = double(actual[i]) - expected[i];
                error2 += error * error;
                norm2 += double(expected[i]) * expected[i];
            }
            Require(error2 <= 1e-10 * std::max(1.0, norm2),
                    "planar CPU GEMM changed numerical output");
        }
    }
    info->hasAVX512BF16 = originalAvx512;
    std::printf("PASS columns=%d cross_swiglu=%d avx512=%d: equal storage and weight/scale bits; CPU outputs agree\n",
                columns, crossSwiglu, originalAvx512 && useAvx512);
}

static void CheckCompact(int columns, bool crossSwiglu, bool useAvx512) {
    using namespace fastllm;
    constexpr int rows = 64;
    const int blocks = (columns + 15) / 16;
    const int compactStride = GetDataBytes(NVFP4_BLOCK_16_E4M3_PACKED, 1, columns);
    std::mt19937 rng(121 + columns);
    std::vector<uint8_t> weights(GetNVFP4WeightBytes(rows, columns));
    std::vector<uint8_t> scales(rows * blocks);
    for (auto &v : weights) v = rng();
    for (auto &v : scales) v = rng() % 127;
    const std::vector<float> globals{.037f, .071f};
    std::vector<uint8_t> expanded(GetDataBytes(NVFP4_BLOCK_16, rows, columns));
    std::vector<uint8_t> compact(GetDataBytes(NVFP4_BLOCK_16_E4M3_PACKED, rows, columns));
    PackCompactE4M3NVFP4Block16Rows(rows, columns, weights.data(), scales.data(), globals,
        1, 16, expanded.data(), 0, rows, crossSwiglu);
    // Pack independently by shard, including the gate/up scale boundary.
    for (int start : {0, 32}) {
        PackCompactE4M3NVFP4Block16Rows(rows, columns, weights.data(), scales.data(), globals,
            1, 16, compact.data() + start * compactStride, start, 32, crossSwiglu, false, true);
    }
    static constexpr FP8E4M3ToFP32Manager fp8ToFloat;
    for (int row = 0; row < rows; row++) {
        int sourceRow = crossSwiglu ? ((row & 1) ? rows / 2 + row / 2 : row / 2) : row;
        const uint8_t *src = compact.data() + row * compactStride;
        float global;
        std::memcpy(&global, src, sizeof(float));
        Require(global == globals[sourceRow / (rows / 2)], "compact global scale changed");
        for (int block = 0; block < blocks; block++) {
            const uint8_t *packed = src + sizeof(float) + block * 9;
            const uint8_t *old = expanded.data() + (row * blocks + block) * 12;
            Require(std::memcmp(packed, old, 8) == 0, "compact FP4 weight bytes changed");
            Require(packed[8] == scales[sourceRow * blocks + block], "compact FP8 scale byte changed");
            const float decoded = fp8ToFloat.dict[packed[8]] * global;
            Require(std::memcmp(&decoded, old + 8, sizeof(float)) == 0, "compact combined scale changed");
        }
    }
    Data sizeCheck(NVFP4_BLOCK_16_E4M3_PACKED, {rows, columns});
    Require(sizeCheck.GetBytes() == compact.size(), "compact tensor allocation size mismatch");
    auto *info = GetCPUInstructInfo();
    bool saved = info->hasAVX512BF16;
    info->hasAVX512BF16 = saved && useAvx512;
    for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 16, 31, 32}) {
        if (batch > 31 && columns % 32 != 0) continue;
        const int lda = columns + 5, ldc = rows + 7;
        std::vector<float> input(batch * lda);
        std::vector<uint16_t> bf16(batch * lda);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = (int(rng() % 2001) - 1000) * .001f;
            bf16[i] = Float32ToBFloat16RNEBits(input[i]);
        }
        for (bool useBf16 : {false, true}) {
            const void *a = useBf16 ? static_cast<void*>(bf16.data()) : input.data();
            const auto type = useBf16 ? BFLOAT16 : FLOAT32;
            std::vector<float> expected(batch * ldc, -1234567.0f), actual(expected);
            for (int layout = 0; layout < 2; layout++) {
                FastllmGemm(batch, columns, rows, a, lda * (useBf16 ? 2 : 4),
                    layout ? compact.data() : expanded.data(), layout ? compactStride : blocks * 12,
                    layout ? actual.data() : expected.data(), ldc * sizeof(float), 3, 61,
                    type, layout ? NVFP4_BLOCK_16_E4M3_PACKED : NVFP4_BLOCK_16, FLOAT32);
            }
            double error2 = 0, norm2 = 0;
            for (int i = 0; i < batch; i++) for (int j = 0; j < ldc; j++) {
                int index = i * ldc + j;
                if (j < 3 || j >= 61) {
                    Require(actual[index] == -1234567.0f, "compact GEMM overwrote output padding");
                } else {
                    Require(std::isfinite(actual[index]), "non-finite compact GEMM result");
                    double error = double(actual[index]) - expected[index];
                    error2 += error * error;
                    norm2 += double(expected[index]) * expected[index];
                }
            }
            // FP32 input's compact generic kernel retains full input precision,
            // whereas the existing AVX512 kernel rounds input to BF16.
            const double tolerance = !useBf16 && info->hasAVX512BF16 && batch < 32 ? .004 : 1e-5;
            Require(error2 <= tolerance * tolerance * std::max(1.0, norm2),
                    "compact GEMM disagrees with expanded scales");
        }
    }
    info->hasAVX512BF16 = saved;
}

// Compare the small-batch BF16 path against an independent FP64 dot product.
// In particular, adjacent block-16 scales need not match when sharing a
// 512-bit dot product. Include padded strides and partitions across row tiles.
static void CheckBFloat16Reference(int columns, int batch) {
    using namespace fastllm;
    constexpr int rows = 64, first = 3, last = 61;
    constexpr float sentinel = -1234567.0f;
    const int blocks = (columns + 15) / 16;
    const int lda = columns + 5, ldc = rows + 7;
    const int ldb = blocks * 12 + 13;
    std::mt19937 rng(20260920 + columns + batch);
    const float table[16] = {
        0, .5f, 1, 1.5f, 2, 3, 4, 6,
        0, -.5f, -1, -1.5f, -2, -3, -4, -6
    };
    std::vector<uint8_t> codes(rows * blocks * 16);
    std::vector<float> scales(rows * blocks);
    for (auto &v : codes) v = rng() % 16;
    for (auto &v : scales) v = (rng() % 17) * .037f;
    // Exercise both equal and unequal adjacent scales, including zero.
    for (int row = 0; row < rows; row++) {
        for (int block = 1; block < blocks; block += 4) {
            scales[row * blocks + block] = scales[row * blocks + block - 1];
        }
    }
    std::vector<uint8_t> legacy(rows * ldb, 0), planar(rows * blocks * 12, 0);
    for (int row = 0; row < rows; row++) {
        for (int block = 0; block < blocks; block++) {
            uint8_t *dst = legacy.data() + row * ldb + block * 12;
            const uint8_t *src = codes.data() + (row * blocks + block) * 16;
            for (int i = 0; i < 8; i++) dst[i] = src[i * 2] | (src[i * 2 + 1] << 4);
            std::memcpy(dst + 8, &scales[row * blocks + block], sizeof(float));
            std::memcpy(planar.data() + NVFP4PlanarWeightOffset(row, blocks, block), dst, 8);
            std::memcpy(planar.data() + NVFP4PlanarScaleOffset(row, blocks, block), dst + 8, 4);
        }
    }
    std::vector<uint16_t> input(batch * lda);
    std::vector<float> inputF32(batch * lda);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = Float32ToBFloat16RNEBits((int(rng() % 2001) - 1000) * .001f);
        uint32_t bits = uint32_t(input[i]) << 16;
        std::memcpy(&inputF32[i], &bits, sizeof(float));
    }
    std::vector<float> outputs[2];
    for (int layout = 0; layout < 2; layout++) {
        auto &out = outputs[layout];
        out.assign(batch * ldc, sentinel);
        const int edges[]{first, 19, 37, last};
        for (int part = 0; part < 3; part++) {
            FastllmGemm(batch, columns, rows, input.data(), lda * sizeof(uint16_t),
                layout ? planar.data() : legacy.data(), ldb,
                out.data(), ldc * sizeof(float), edges[part], edges[part + 1],
                BFLOAT16, layout ? NVFP4_BLOCK_16_PLANAR : NVFP4_BLOCK_16, FLOAT32);
        }
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < ldc; j++) {
                const float actual = out[i * ldc + j];
                if (j < first || j >= last) {
                    Require(actual == sentinel, "block16 GEMM overwrote output padding");
                    continue;
                }
                double expected = 0, magnitude = 0;
                for (int l = 0; l < columns; l++) {
                    const double term = double(inputF32[i * lda + l]) *
                        table[codes[j * blocks * 16 + l]] * scales[j * blocks + l / 16];
                    expected += term;
                    magnitude += std::fabs(term);
                }
                Require(std::isfinite(actual) &&
                    std::fabs(double(actual) - expected) <= 1e-6 + magnitude * 1e-6,
                    "block16 BF16 GEMM disagrees with FP64 reference");
            }
        }
    }
    Require(std::memcmp(outputs[0].data(), outputs[1].data(), outputs[0].size() * sizeof(float)) == 0,
            "block16 BF16 layouts changed output bits");
}

// Call AVX2 directly even on AVX512 hosts. Compare against the original
// FP32-scale kernel, including the reduction order, tails and padded strides.
static void CheckCompactAvx2(int columns, int batch, bool specialValues) {
    using namespace fastllm;
    if (!GetCPUInstructInfo()->hasAVX2) return;
    constexpr int rows = 64, first = 3, last = 61;
    constexpr float sentinel = -1234567.0f;
    const int blocks = (columns + 15) / 16;
    const int lda = columns + 5, ldc = rows + 7;
    const int compactStride = 4 + blocks * 9 + 13, expandedStride = blocks * 12 + 7;
    std::vector<uint8_t> compact(rows * compactStride, 0), expanded(rows * expandedStride, 0);
    std::mt19937 rng(20260920 + columns + batch);
    static constexpr FP8E4M3ToFP32Manager fp8ToFloat;
    const float globals[] = {.037f, -.071f, 0.f, -0.f, 0x1p-126f, 0x1p120f};
    for (int j = 0; j < rows; j++) {
        const float global = specialValues ? globals[j % 6] : .037f;
        std::memcpy(compact.data() + j * compactStride, &global, sizeof(float));
        for (int block = 0; block < blocks; block++) {
            uint8_t *dst = compact.data() + j * compactStride + 4 + block * 9;
            for (int i = 0; i < 8; i++) dst[i] = rng();
            dst[8] = specialValues ? (j * blocks + block) % 256 : rng() % 127;
            uint8_t *old = expanded.data() + j * expandedStride + block * 12;
            std::memcpy(old, dst, 8);
            const float scale = fp8ToFloat.dict[dst[8]] * global;
            std::memcpy(old + 8, &scale, sizeof(float));
        }
    }
    std::vector<uint16_t> input(batch * lda);
    for (auto &v : input) v = Float32ToBFloat16RNEBits((int(rng() % 2001) - 1000) * .001f);
    if (specialValues) {
        const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x7f7f, 0x7f80, 0x7fc0};
        for (int r = 0; r < batch; r++) input[r * lda + (r * 17) % columns] = edge[r % 9];
    }
    std::vector<float> actual(batch * ldc, sentinel), expected(actual);
    const int edges[] = {first, 4, 19, 37, last};
    for (int part = 0; part < 4; part++) {
        if (!FastllmGemmBFloat16NVFP4Block16E4M3Packed_AVX2(input.data(), lda * 2,
            compact.data(), compactStride, actual.data(), ldc * sizeof(float),
            batch, columns, rows, edges[part], edges[part + 1])) return;
        Require(FastllmGemmBFloat16NVFP4Block16_AVX2(input.data(), lda * 2,
            expanded.data(), expandedStride, expected.data(), ldc * sizeof(float),
            batch, columns, rows, edges[part], edges[part + 1]), "AVX2 reference unavailable");
    }
    for (size_t i = 0; i < actual.size(); i++) {
        if (i % ldc < first || i % ldc >= last) {
            Require(actual[i] == sentinel, "compact AVX2 overwrote output padding");
        }
        Require((std::isnan(actual[i]) && std::isnan(expected[i])) ||
            std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0,
            "compact AVX2 changed output bits or overwrote padding");
    }
}

int main() {
    try {
        for (int columns : {17, 33, 128, 640})
            for (bool cross : {false, true})
                for (bool avx512 : {false, true}) Run(columns, cross, avx512);
        for (int columns : {1, 16, 17, 31, 32, 33, 48, 64, 80, 112, 160, 352, 640, 2560})
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31})
                CheckBFloat16Reference(columns, batch);
        std::puts("PASS block16 BF16 FP64 reference, strides, partitions and layouts");
        for (int columns : {1, 16, 17, 31, 32, 33, 640, 2560})
            for (bool cross : {false, true})
                for (bool avx512 : {false, true}) CheckCompact(columns, cross, avx512);
        std::puts("PASS compact raw FP8 scales, row shards, CPU decode and prefill");
        for (int columns : {1, 16, 17, 31, 32, 33, 48, 64, 80, 112, 160, 352, 640, 2560})
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32})
                for (bool special : {false, true}) CheckCompactAvx2(columns, batch, special);
        std::puts("PASS compact AVX2 output bits, special values, row tiles, tails and padding");
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
