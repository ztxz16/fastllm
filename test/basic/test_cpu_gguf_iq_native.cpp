#include "ggml-iq-native.h"
#include "gguf_iq2_s_r4_fixture.h"
#include "devices/cpu/computeutils.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Cover every codebook index in every packed position, including the high
// index bit and sign broadcasts across 128-bit lanes. Unit scales make the
// decoded reference an exact integer, including Q8's -128 endpoint.
static void RunIQ3SCodebook() {
    const auto decode = ggml_type_to_float(GGML_TYPE_IQ3_S);
    const auto dot = ggml_type_vec_dot(GGML_TYPE_IQ3_S);
    const int8_t endpoints[] = {-128, -127, -1, 0, 1, 126, 127};
    for (int code = 0; code < 512; ++code)
    for (int signs : {0x00, 0xff, 0x55, 0xaa})
    for (int pattern = 0; pattern < 3; ++pattern) {
        block_iq3_s x {};
        block_q8_K y {};
        x.d = 0x3c00;
        y.d = 1.0f;
        std::memset(x.scales, 0xff, sizeof(x.scales));
        for (int g = 0; g < 8; ++g) {
            for (int i = 0; i < 8; ++i) {
                const int index = (code + g * 73 + i * 19) % 512;
                x.qs[8 * g + i] = index & 255;
                x.qh[g] |= (index >> 8) << i;
            }
            for (int i = 0; i < 4; ++i) x.signs[4 * g + i] = signs ^ (g * 37 + i * 19);
        }
        for (int i = 0; i < QK_K; ++i)
            y.qs[i] = pattern == 0 ? -128 : pattern == 1 ? 127 : endpoints[i % 7];
        float decoded[QK_K], actual;
        decode(&x, decoded, QK_K);
        double expected = 0;
        for (int i = 0; i < QK_K; ++i) expected += double(decoded[i]) * y.qs[i];
        dot(QK_K, &actual, 0, &x, 0, &y, 0, 1);
        Check(actual == expected, "IQ3_S codebook/sign endpoint differs from decoded reference");
        Check(iq_native::dot(QK_K, &x, &y) == expected, "IQ3_S inline codebook/sign endpoint differs");
    }
}

template<class Block> static void Run(ggml_type type) {
    Check(ggml_type_vec_dot_type(type) == GGML_TYPE_Q8_K, "wrong activation type");
    const auto dot = ggml_type_vec_dot(type);
    Check(dot != nullptr, "missing quantized dot");
    const auto decode = ggml_type_to_float(type);
    std::mt19937 random(23907 + type);
    for (int columns : {256, 512, 1280, 2560, 5120}) for (int inputs : {1, 4, 8, 17}) {
        constexpr int outputs = 19;
        const int blocks = columns / QK_K;
        std::vector<Block> x(outputs * blocks);
        std::vector<block_q8_K> y(inputs * blocks);
        const uint16_t scales[] = {0, 0x2800, 0xb000, 0x2c80, 1, 0x8155};
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < sizeof(Block); ++j) ((uint8_t *)&x[i])[j] = random();
            x[i].d = scales[i % 6];
        }
        for (auto &b : y) {
            b.d = (int(random() % 15) - 7) / 128.0f;
            for (int j = 0; j < QK_K; ++j) b.qs[j] = j % 3 == 0 ? -128 : int(random() % 256) - 128;
        }
        std::vector<float> result(inputs * outputs, -123456), bias(outputs), decoded(columns);
        for (int i = 0; i < outputs; ++i) bias[i] = (i - 10) * 0.125f;
        const auto at = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + GGML_TYPE_Q8_K);
        const auto wt = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + type);
        Check(fastllm::LinearQ8K_GGUF_Kernel((uint8_t *)y.data(), (uint8_t *)x.data(), bias.data(),
            result.data(), inputs, columns, outputs, 2, outputs - 2, at, wt), "linear dispatch failed");
        for (int col = 0; col < outputs; ++col) {
            decode(x.data() + col * blocks, decoded.data(), columns);
            for (int row = 0; row < inputs; ++row) {
                double expected = 0, magnitude = 0;
                for (int c = 0; c < columns; ++c) {
                    const auto &b = y[row * blocks + c / QK_K];
                    const double term = double(decoded[c]) * b.d * b.qs[c % QK_K];
                    expected += term; magnitude += std::abs(term);
                }
                float actual = 0;
                dot(columns, &actual, 0, x.data() + col * blocks, 0, y.data() + row * blocks, 0, 1);
                const double tolerance = 1e-5 + magnitude * 3e-6;
                auto near = [&](float value, double reference) {
                    if (!std::isfinite(value) || std::abs(value - reference) > tolerance) {
                        std::fprintf(stderr, "type=%d columns=%d row=%d col=%d actual=%.9g expected=%.12g\n",
                                     type, columns, row, col, value, reference);
                        throw std::runtime_error("dot differs from decoded FP64 reference");
                    }
                };
                near(actual, expected);
                near(iq_native::dot(columns, x.data() + col * blocks, y.data() + row * blocks), expected);
                if (col >= 2 && col < outputs - 2) near(result[row * outputs + col], expected + bias[col]);
                else Check(result[row * outputs + col] == -123456, "linear wrote outside requested columns");
            }
        }
    }
}

static void RunR4(ggml_type type) {
    const auto *pack = get_repack_info(type);
    const Repack iq2Fixture{GGML_TYPE_IQ2_S_R4, 4, PackIQ2SR4Fixture};
    if (!pack && type == GGML_TYPE_IQ2_S && GetMulMatFunction(GGML_TYPE_IQ2_S_R4, 1))
        pack = &iq2Fixture;
    if (!pack) return; // This build has no architecture-specific R4 kernels.
    const auto decode = ggml_type_to_float(type);
    std::mt19937 random(62851 + type);
    constexpr int outputs = 24, first = 4, last = 20;
    const uint16_t scales[] = {0, 0x2800, 0xb000, 0x2c80, 1, 0x8155};
    for (int columns : {256, 512, 2560}) for (int inputs : {1, 2, 8, 9}) {
        const size_t stride = ggml_row_size(type, columns);
        const size_t block = ggml_type_size(type);
        std::vector<uint8_t> raw(outputs * stride), packed(raw.size());
        for (auto &v : raw) v = random();
        for (size_t off = 0; off < raw.size(); off += block)
            std::memcpy(raw.data() + off, &scales[(off / block) % 6], sizeof(uint16_t));
        pack->repack(outputs, columns, (const char *)raw.data(), (char *)packed.data(), false);
        std::vector<block_q8_K> activation(inputs * columns / QK_K);
        for (auto &b : activation) {
            b.d = (int(random() % 15) - 7) / 128.0f;
            // The Q8_K row quantizer used by these R4 kernels emits [-127,127].
            for (int i = 0; i < QK_K; ++i)
                b.qs[i] = i % 3 == 0 ? -127 : i % 3 == 1 ? 127 : int(random() % 255) - 127;
        }
        std::vector<float> result(inputs * outputs, -123456), bias(outputs), row(columns);
        for (int col = 0; col < outputs; ++col) bias[col] = (col - 10) * .125f;
        const auto at = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + GGML_TYPE_Q8_K);
        const auto wt = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + pack->new_type);
        for (bool withBias : {false, true}) {
            std::fill(result.begin(), result.end(), -123456);
            Check(fastllm::LinearQ8K_GGUF_Kernel((uint8_t *)activation.data(), packed.data(),
                withBias ? bias.data() : nullptr, result.data(), inputs, columns, outputs,
                first, last, at, wt), "R4 linear dispatch failed");
            for (int col = 0; col < outputs; ++col) {
                decode(raw.data() + col * stride, row.data(), columns);
                for (int input = 0; input < inputs; ++input) {
                    if (col < first || col >= last) {
                        Check(result[input * outputs + col] == -123456, "R4 wrote outside requested slice");
                        continue;
                    }
                    double expected = withBias ? bias[col] : 0, magnitude = 0;
                    for (int c = 0; c < columns; ++c) {
                        const auto &b = activation[(input * columns + c) / QK_K];
                        const double term = double(row[c]) * b.d * b.qs[c % QK_K];
                        expected += term; magnitude += std::abs(term);
                    }
                    const float actual = result[input * outputs + col];
                    Check(std::isfinite(actual) && std::abs(actual - expected) <= 1e-5 + magnitude * 3e-6,
                          "IQ R4 decode/batch differs from decoded FP64 reference");
                }
            }
        }
    }
}

static void RunQ2() {
    Check(ggml_type_vec_dot_type(GGML_TYPE_Q2_0) == GGML_TYPE_Q8_0, "wrong Q2_0 activation type");
    std::mt19937 random(82403);
    constexpr int outputs = 19;
    const uint16_t scales[] = {0, 0x2800, 0xb000, 0x3c01, 1, 0x8155, 0x7bff};
    for (int columns : {64, 128, 192, 640, 2560, 5120}) for (int inputs : {1, 4, 17}) {
        const int blocks = columns / QK2_0;
        std::vector<block_q2_0> x(outputs * blocks);
        std::vector<block_q8_0> y(inputs * columns / QK8_0);
        for (size_t b = 0; b < x.size(); ++b) {
            x[b].d = scales[b % 7];
            for (auto &q : x[b].qs) q = b % 5 == 4 ? random() : (b % 5) * 0x55;
        }
        for (size_t b = 0; b < y.size(); ++b) {
            y[b].d = scales[(b + 3) % 6];
            for (int j = 0; j < QK8_0; ++j)
                y[b].qs[j] = j % 3 == 0 ? -128 : j % 3 == 1 ? 127 : int(random() % 256) - 128;
        }
        std::vector<float> result(inputs * outputs, -123456), bias(outputs);
        for (int i = 0; i < outputs; ++i) bias[i] = (i - 10) * 0.125f;
        const auto at = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + GGML_TYPE_Q8_0);
        const auto wt = fastllm::DataType(int(fastllm::DATA_GGUF_FORMAT) + GGML_TYPE_Q2_0);
        Check(fastllm::LinearQ8K_GGUF_Kernel((uint8_t *)y.data(), (uint8_t *)x.data(), bias.data(),
            result.data(), inputs, columns, outputs, 2, outputs - 2, at, wt), "Q2_0 linear dispatch failed");
        for (int row = 0; row < inputs; ++row) for (int col = 0; col < outputs; ++col) {
            const auto *w = x.data() + col * blocks;
            const auto *a = y.data() + row * columns / QK8_0;
            double expected = 0, magnitude = 0;
            for (int c = 0; c < columns; ++c) {
                const auto &xb = w[c / QK2_0];
                const auto &yb = a[c / QK8_0];
                const int j = c % QK2_0;
                const int q = ((xb.qs[j / 4] >> (2 * (j % 4))) & 3) - 1;
                const double term = double(GGML_FP16_TO_FP32(xb.d)) * q *
                                    GGML_FP16_TO_FP32(yb.d) * yb.qs[c % QK8_0];
                expected += term; magnitude += std::abs(term);
            }
            float actual = 0;
            ggml_vec_dot_q2_0_q8_0(columns, &actual, 0, w, 0, a, 0, 1);
            auto near = [&](float value, double reference) {
                if (!std::isfinite(value) || std::abs(value - reference) > 1e-5 + magnitude * 3e-6) {
                    std::fprintf(stderr, "Q2_0 columns=%d row=%d col=%d actual=%.9g expected=%.12g\n",
                                 columns, row, col, value, reference);
                    throw std::runtime_error("Q2_0 differs from decoded FP64 reference");
                }
            };
            near(actual, expected);
            if (col >= 2 && col < outputs - 2) near(result[row * outputs + col], expected + bias[col]);
            else Check(result[row * outputs + col] == -123456, "Q2_0 linear wrote outside requested columns");
        }
    }
}

int main() {
    try {
        RunQ2();
        RunIQ3SCodebook();
        Run<block_iq2_s>(GGML_TYPE_IQ2_S);
        Run<block_iq3_xxs>(GGML_TYPE_IQ3_XXS);
        Run<block_iq3_s>(GGML_TYPE_IQ3_S);
        Run<block_iq4_xs>(GGML_TYPE_IQ4_XS);
        RunR4(GGML_TYPE_IQ2_S);
        RunR4(GGML_TYPE_IQ3_XXS);
        std::puts("PASS: IQ3_S/IQ4_XS CPU, IQ2_S/IQ3_XXS R4 and Q2_0 dot, codebooks, signed endpoints, scales and linear dispatch");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
