#include "ggml-iq-native.h"
#include "devices/cpu/computeutils.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
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
int main() {
    try {
        Run<block_iq3_s>(GGML_TYPE_IQ3_S);
        Run<block_iq4_xs>(GGML_TYPE_IQ4_XS);
        std::puts("PASS: IQ3_S/IQ4_XS CPU dot, signed endpoints, scales and linear dispatch");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
