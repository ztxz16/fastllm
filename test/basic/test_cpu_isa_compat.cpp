// Build this test with UNIT_TEST=ON. On GCC, exercise old x86 targets with
// -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=sandybridge" (AVX only) or
// -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=core2" (no AVX/XSAVE).
#include "fastllm.h"
#include "gguf.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastllm {
    void Float32ToFloat16(float *, uint16_t *, int);
    void Float32ToBFloat16(float *, uint16_t *, int);
}
void quantize_row_q8_1(const float *, void *, int64_t);

static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void TestConversions() {
    // Cover scalar tails as well as several complete SIMD vectors.
    for (int count : {1, 7, 8, 9, 31, 65}) {
        std::vector<float> input(count);
        std::vector<uint16_t> fp16(count), bf16(count);
        for (int i = 0; i < count; ++i) input[i] = (i - count / 2) * 0.125f;
        fastllm::Float32ToFloat16(input.data(), fp16.data(), count);
        for (int i = 0; i < count; ++i)
            Require(fp16[i] == fastllm::float_to_half(input[i]), "FP16 conversion failed");
        for (int i = 0; i < count; ++i) {
            const uint32_t bits = 0x3f800000u + (uint32_t(i) << 15);
            std::memcpy(&input[i], &bits, sizeof(bits));
            if (i % 2) input[i] = -input[i];
        }
        fastllm::Float32ToBFloat16(input.data(), bf16.data(), count);
        for (int i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &input[i], sizeof(bits));
            const uint16_t expected = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
            Require(bf16[i] == expected, "BF16 rounding failed");
        }
    }
}

static void Dequantize(ggml_type type, const void *data, float *out, int count) {
    if (type == GGML_TYPE_Q8_1) {
        const auto *blocks = static_cast<const block_q8_1 *>(data);
        for (int i = 0; i < count; ++i)
            out[i] = fastllm::half_to_float(blocks[i / QK8_1].d) * blocks[i / QK8_1].qs[i % QK8_1];
    } else if (type == GGML_TYPE_Q8_K) {
        const auto *blocks = static_cast<const block_q8_K *>(data);
        for (int i = 0; i < count; ++i)
            out[i] = blocks[i / QK_K].d * blocks[i / QK_K].qs[i % QK_K];
    } else {
        auto convert = ggml_type_to_float(type);
        Require(convert != nullptr, "Missing dequantizer");
        convert(data, out, count);
    }
}

static void QuantizeWeights(ggml_type type, const std::vector<float> &x,
                            std::vector<uint8_t> &packed, std::mt19937 &random) {
    const int count = x.size();
    switch (type) {
        case GGML_TYPE_Q2_K: quantize_row_q2_K_ref(x.data(), reinterpret_cast<block_q2_K *>(packed.data()), count); return;
        case GGML_TYPE_Q3_K: quantize_row_q3_K_ref(x.data(), reinterpret_cast<block_q3_K *>(packed.data()), count); return;
        case GGML_TYPE_Q4_K: quantize_row_q4_K_ref(x.data(), reinterpret_cast<block_q4_K *>(packed.data()), count); return;
        case GGML_TYPE_Q5_K: quantize_row_q5_K_ref(x.data(), reinterpret_cast<block_q5_K *>(packed.data()), count); return;
        case GGML_TYPE_Q6_K: quantize_row_q6_K_ref(x.data(), reinterpret_cast<block_q6_K *>(packed.data()), count); return;
        case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(x.data(), reinterpret_cast<block_q8_0 *>(packed.data()), count); return;
        default: break;
    }
    // These formats have dequantizers but no reference quantizer in this repo.
    // Generate valid packed values, including arbitrary high bits and signs.
    for (auto &byte : packed) byte = random();
    for (int b = 0; b < count / 32; ++b) {
        const uint16_t scale = fastllm::float_to_half((b % 5 + 1) * 0.125f);
        if (type == GGML_TYPE_Q5_0)
            reinterpret_cast<block_q5_0 *>(packed.data())[b].d = scale;
        else if (type == GGML_TYPE_Q5_1) {
            auto &block = reinterpret_cast<block_q5_1 *>(packed.data())[b];
            block.d = scale;
            block.m = fastllm::float_to_half(-1.25f);
        } else {
            Require(type == GGML_TYPE_IQ4_NL, "Unexpected packed type");
            reinterpret_cast<block_iq4_nl *>(packed.data())[b].d = scale;
        }
    }
}

static void TestQuantizedDots() {
    std::mt19937 random(20260920);
    for (ggml_type type : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0,
                          GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,
                          GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                          GGML_TYPE_Q6_K}) {
        for (int count : {256, 768}) {
            const auto rhsType = ggml_type_vec_dot_type(type);
            auto dot = ggml_type_vec_dot(type);
            Require(dot != nullptr, "Missing quantized dot product");
            std::vector<float> x(count), y(count), dx(count), dy(count);
            std::vector<uint8_t> qx(ggml_row_size(type, count)), qy(ggml_row_size(rhsType, count));
            for (int iteration = 0; iteration < 12; ++iteration) {
                for (int i = 0; i < count; ++i) {
                    x[i] = (int(random() % 20001) - 10000) / 997.0f;
                    y[i] = (int(random() % 20001) - 10000) / 991.0f;
                }
                if (iteration == 0) std::fill(x.begin(), x.end(), 0.0f);
                QuantizeWeights(type, x, qx, random);
                if (rhsType == GGML_TYPE_Q8_1) {
                    // Exercise the AVX-only hsum_i32_4 quantization path.
                    quantize_row_q8_1(y.data(), qy.data(), count);
                    const auto *blocks = reinterpret_cast<const block_q8_1 *>(qy.data());
                    for (int b = 0; b < count / QK8_1; ++b) {
                        float maximum = 0.0f;
                        int sum = 0;
                        for (int i = 0; i < QK8_1; ++i) {
                            maximum = std::max(maximum, std::abs(y[b * QK8_1 + i]));
                            sum += blocks[b].qs[i];
                        }
                        Require(blocks[b].s == fastllm::float_to_half((maximum / 127.0f) * sum),
                                "Q8_1 quantized sum failed");
                    }
                } else if (rhsType == GGML_TYPE_Q8_K) {
                    quantize_row_q8_K_ref(y.data(), reinterpret_cast<block_q8_K *>(qy.data()), count);
                } else {
                    quantize_row_q8_0_ref(y.data(), reinterpret_cast<block_q8_0 *>(qy.data()), count);
                }
                Dequantize(type, qx.data(), dx.data(), count);
                Dequantize(rhsType, qy.data(), dy.data(), count);
                double expected = 0.0, absoluteSum = 0.0;
                for (int i = 0; i < count; ++i) {
                    const double product = double(dx[i]) * dy[i];
                    expected += product;
                    absoluteSum += std::abs(product);
                }
                float actual = 0.0f;
                dot(count, &actual, 0, qx.data(), 0, qy.data(), 0, 1);
                // Q8_1 stores its correction sum in FP16 separately from d.
                const double tolerance = (rhsType == GGML_TYPE_Q8_1 ? 5e-4 : 2e-6) * (1 + absoluteSum);
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                    std::fprintf(stderr, "%s n=%d: actual=%g expected=%g tolerance=%g\n",
                                 ggml_type_name(type), count, actual, expected, tolerance);
                    throw std::runtime_error("Quantized dot product failed");
                }
            }
        }
    }
}

int main() {
    try {
        fastllm::CPUInstructInfo cpuInfo; // Must compile without -mxsave.
        TestConversions();
        TestQuantizedDots();
#if !defined(__AVX2__) || (!defined(_MSC_VER) && (!defined(__FMA__) || !defined(__F16C__)))
        for (ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS})
            Require(get_repack_info(type) == nullptr, "Unsupported GGUF repack enabled");
        Require(GetMulMatFunction(GGML_TYPE_Q4_K_R4, 1) == nullptr,
                "Unsupported repacked kernel enabled");
#endif
        std::puts("PASS: CPU detection, FP16/BF16 conversions, 216 GGUF dot products and repack fallback");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
