#include "gguf.h"
#include "devices/cpu/computeutils.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr float sentinel = -123456.0f;
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    int columns, outputs;
    size_t weightStride, inputStride;
    std::vector<uint8_t> weights, activations;
    Fixture(int columns, int outputs, int inputs, bool padding, bool signedExtreme)
        : columns(columns), outputs(outputs),
          weightStride(ggml_row_size(GGML_TYPE_IQ4_NL, columns) + (padding ? 10 : 0)),
          inputStride(ggml_row_size(GGML_TYPE_Q8_0, columns) + (padding ? 6 : 0)),
          weights(weightStride * outputs), activations(inputStride * inputs) {
        std::mt19937 random(20260929 + columns + outputs + inputs);
        // Include zero, signed scales and half subnormals without relying on
        // a float-to-half conversion in the fixture itself.
        const uint16_t scales[] = {0x0000, 0x2800, 0xb000, 0x2c80, 0xa400, 0x0001, 0x8155};
        for (int row = 0; row < outputs; ++row) {
            auto *blocks = reinterpret_cast<block_iq4_nl *>(weights.data() + row * weightStride);
            for (int b = 0; b < columns / QK4_NL; ++b) {
                blocks[b].d = scales[(b + row) % 7];
                for (auto &q : blocks[b].qs) q = random();
            }
        }
        for (int row = 0; row < inputs; ++row) {
            auto *blocks = reinterpret_cast<block_q8_0 *>(activations.data() + row * inputStride);
            for (int b = 0; b < columns / QK8_0; ++b) {
                blocks[b].d = scales[(b + row + 1) % 7];
                for (int i = 0; i < QK8_0; ++i) {
                    int q = int(random() % 255) - 127;
                    if (i % 4 == 0) q = signedExtreme ? -128 : -127;
                    else if (i % 4 == 1) q = 127;
                    else if (i % 4 == 2) q = 0;
                    blocks[b].qs[i] = q;
                }
            }
        }
    }
    double Reference(int output, int input, double *magnitude = nullptr) const {
        std::vector<float> x(columns), y(columns);
        dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(
            weights.data() + output * weightStride), x.data(), columns);
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(
            activations.data() + input * inputStride), y.data(), columns);
        double sum = 0.0, absolute = 0.0;
        for (int i = 0; i < columns; ++i) {
            const double product = double(x[i]) * y[i];
            sum += product;
            absolute += std::abs(product);
        }
        if (magnitude) *magnitude = absolute;
        return sum;
    }
};

void Near(float actual, double expected, double magnitude = 0.0) {
    // A nearly cancelled dot can have a tiny result despite large FP32
    // partial sums. Bound its absolute error using the dot's magnitude.
    if (!std::isfinite(actual) || std::abs(actual - expected) >
        2e-4 + 3e-6 * std::max(std::abs(expected), magnitude)) {
        std::fprintf(stderr, "actual=%.9g expected=%.12g\n", actual, expected);
        throw std::runtime_error("IQ4_NL result differs from dequantized FP64 reference");
    }
}

void CheckReference(float actual, const Fixture &f, int output, int input, float bias = 0.0f) {
    double magnitude;
    const double expected = f.Reference(output, input, &magnitude) + bias;
    Near(actual, expected, magnitude);
}

void TestDirect() {
    for (int columns : {32, 64, 96, 256, 640, 768, 2560}) {
        for (int outputs : {1, 2, 3, 4, 5, 7, 13, 64}) {
            for (int inputs = 1; inputs <= 8; ++inputs) {
                Fixture f(columns, outputs + 3, inputs + 1, true, true);
                const int stride = outputs + 9;
                std::vector<float> result((inputs + 2) * stride, sentinel);
                DataInfo info{result.data() + 2, (const char *)f.activations.data(),
                              size_t(stride), f.inputStride, 1, 1, nullptr, 0};
                auto kernel = GetMulMatFunction(GGML_TYPE_IQ4_NL, inputs);
                Check(kernel != nullptr, "Missing IQ4_NL input-row specialization");
                kernel(columns, f.weights.data() + 3 * f.weightStride, f.weightStride, info, outputs);
                for (int row = 0; row < inputs + 2; ++row) {
                    for (int col = 0; col < stride; ++col) {
                        if (row >= 1 && row <= inputs && col >= 2 && col < outputs + 2)
                            CheckReference(result[row * stride + col], f, col + 1, row);
                        else Check(result[row * stride + col] == sentinel, "IQ4_NL output canary overwritten");
                    }
                }
            }
        }
    }
    Check(GetMulMatFunction(GGML_TYPE_IQ4_NL, 0) == nullptr &&
          GetMulMatFunction(GGML_TYPE_IQ4_NL, 9) == nullptr, "Unsupported input-row count accepted");
    Check(get_repack_info(GGML_TYPE_IQ4_NL) == nullptr, "IQ4_NL layout unexpectedly changed");
}

void TestMappedRows() {
    Fixture f(96, 7, 6, true, true);
    const mmid_row_mapping mapping[] = {{0, 0}, {3, 1}, {0, 1}, {4, 0}};
    const int stride = 11, plane = 5 * stride;
    std::vector<float> result(2 * plane, sentinel);
    std::vector<double> expected(result.begin(), result.end()), magnitude(result.size());
    DataInfo info{result.data() + 1, (const char *)f.activations.data(),
                  size_t(stride), f.inputStride, 1, 3, mapping, size_t(plane)};
    auto kernel = GetMulMatFunction(GGML_TYPE_IQ4_NL, 3);
    kernel(f.columns, f.weights.data(), f.weightStride, info, f.outputs);
    for (int i = 1; i <= 3; ++i) {
        const auto &m = mapping[i];
        for (int col = 0; col < f.outputs; ++col) {
            const int offset = m.i1 * stride + m.i2 * plane + 1 + col;
            expected[offset] = f.Reference(col, m.i1 % 3 + m.i2 * 3, &magnitude[offset]);
        }
    }
    for (size_t i = 0; i < result.size(); ++i) {
        if (expected[i] == sentinel) Check(result[i] == sentinel, "Mapped output canary overwritten");
        else Near(result[i], expected[i], magnitude[i]);
    }
}

void TestLinearDispatch() {
    const auto at = static_cast<fastllm::DataType>(int(fastllm::DataType::DATA_GGUF_FORMAT) + GGML_TYPE_Q8_0);
    const auto wt = static_cast<fastllm::DataType>(int(fastllm::DataType::DATA_GGUF_FORMAT) + GGML_TYPE_IQ4_NL);
    for (int inputs : {1, 2, 8, 9, 17}) {
        Fixture f(640, 79, inputs, false, false);
        std::vector<float> result(inputs * f.outputs, sentinel), bias(f.outputs);
        for (int i = 0; i < f.outputs; ++i) bias[i] = (i - 40) * 0.25f;
        Check(fastllm::LinearQ8K_GGUF_Kernel(f.activations.data(), f.weights.data(), bias.data(),
              result.data(), inputs, f.columns, f.outputs, 3, 76, at, wt), "Linear dispatch failed");
        for (int row = 0; row < inputs; ++row) {
            for (int col = 0; col < f.outputs; ++col) {
                if (col >= 3 && col < 76) {
                    CheckReference(result[row * f.outputs + col], f, col, row, bias[col]);
                    // The decode dimension is an even number of blocks.
                    // Preserve the original kernel's accumulation order for
                    // its valid Q8 range, including subnormal half scales.
                    float original;
                    ggml_vec_dot_iq4_nl_q8_0(f.columns, &original, 0,
                        f.weights.data() + col * f.weightStride, 0,
                        f.activations.data() + row * f.inputStride, 0, 1);
                    Check(result[row * f.outputs + col] == original + bias[col],
                          "Even-block dot no longer matches the original kernel exactly");
                }
                else Check(result[row * f.outputs + col] == sentinel, "Linear dispatch wrote outside its columns");
            }
        }
    }
}
} // namespace

int main() {
    if (GetMulMatFunction(GGML_TYPE_IQ4_NL, 1) == nullptr) {
        std::puts("IQ4_NL multi-row kernel requires AVX2, FMA and F16C");
        return 77;
    }
    try {
        TestDirect();
        TestMappedRows();
        TestLinearDispatch();
        std::puts("PASS: IQ4_NL multi-row CPU kernel, tails, strides, row mapping and linear dispatch");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
