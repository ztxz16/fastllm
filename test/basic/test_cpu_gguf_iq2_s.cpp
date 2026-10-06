#include "gguf.h"
#include "devices/cpu/computeutils.h"

#include <algorithm>
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
          weightStride(ggml_row_size(GGML_TYPE_IQ2_S, columns) + (padding ? 10 : 0)),
          inputStride(ggml_row_size(GGML_TYPE_Q8_K, columns) + (padding ? 12 : 0)),
          weights(weightStride * outputs), activations(inputStride * inputs) {
        std::mt19937 random(20260929 + columns + outputs + inputs);
        // Include zero, signed scales and half subnormals without relying on
        // a float-to-half conversion in the fixture itself.
        const uint16_t scales[] = {0x0000, 0x2800, 0xb000, 0x2c80, 0xa400, 0x0001, 0x8155};
        for (int row = 0; row < outputs; ++row) {
            auto *blocks = reinterpret_cast<block_iq2_s *>(weights.data() + row * weightStride);
            for (int b = 0; b < columns / QK_K; ++b) {
                blocks[b].d = scales[(b + row) % 7];
                for (auto &q : blocks[b].qs) q = random();
                for (auto &q : blocks[b].qh) q = random();
                for (auto &q : blocks[b].scales) q = random();
            }
        }
        for (int row = 0; row < inputs; ++row) {
            auto *blocks = reinterpret_cast<block_q8_K *>(activations.data() + row * inputStride);
            for (int b = 0; b < columns / QK_K; ++b) {
                blocks[b].d = (int(random() % 15) - 7) / 128.0f;
                for (int i = 0; i < QK_K; ++i) {
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
        ggml_type_to_float(GGML_TYPE_IQ2_S)(reinterpret_cast<const block_iq2_s *>(
            weights.data() + output * weightStride), x.data(), columns);
        const auto *a = reinterpret_cast<const block_q8_K *>(activations.data() + input * inputStride);
        for (int i = 0; i < columns; ++i) y[i] = a[i / QK_K].d * a[i / QK_K].qs[i % QK_K];
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
        throw std::runtime_error("IQ2_S result differs from dequantized FP64 reference");
    }
}

void CheckReference(float actual, const Fixture &f, int output, int input, float bias = 0.0f) {
    double magnitude;
    const double expected = f.Reference(output, input, &magnitude) + bias;
    Near(actual, expected, magnitude);
}

void TestDirect() {
    for (int columns : {256, 512, 768, 1280, 2560, 5120}) {
        for (int outputs : {0, 1, 2, 3, 4, 5, 7, 13, 64}) {
            for (int inputs = 1; inputs <= 8; ++inputs) for (bool extreme : {false, true}) {
                Fixture f(columns, outputs + 3, inputs + 1, true, extreme);
                const int stride = outputs + 9;
                std::vector<float> result((inputs + 2) * stride, sentinel);
                DataInfo info{result.data() + 2, (const char *)f.activations.data(),
                              size_t(stride), f.inputStride, 1, 1, nullptr, 0};
                auto kernel = GetMulMatFunction(GGML_TYPE_IQ2_S, inputs);
                Check(kernel != nullptr, "Missing IQ2_S input-row specialization");
                kernel(columns, f.weights.data() + 3 * f.weightStride, f.weightStride, info, outputs);
                for (int row = 0; row < inputs + 2; ++row) {
                    for (int col = 0; col < stride; ++col) {
                        if (row >= 1 && row <= inputs && col >= 2 && col < outputs + 2)
                            CheckReference(result[row * stride + col], f, col + 1, row);
                        else Check(result[row * stride + col] == sentinel, "IQ2_S output canary overwritten");
                    }
                }
            }
        }
    }
    Check(GetMulMatFunction(GGML_TYPE_IQ2_S, 0) == nullptr &&
          GetMulMatFunction(GGML_TYPE_IQ2_S, 9) == nullptr, "Unsupported input-row count accepted");
    Check(get_repack_info(GGML_TYPE_IQ2_S) == nullptr, "IQ2_S layout unexpectedly changed");
}

void TestMappedRows() {
    const mmid_row_mapping mapping[] = {
        {0, 0}, {3, 1}, {0, 1}, {4, 0}, {2, 2}, {1, 0}, {4, 2}, {0, 0}, {2, 1}};
    for (int inputs : {1, 3, 5, 8}) for (bool extreme : {false, true}) {
        Fixture f(512, 13, 9, true, extreme);
        const int stride = 17, plane = 5 * stride;
        std::vector<float> result(3 * plane, sentinel);
        std::vector<double> expected(result.begin(), result.end()), magnitude(result.size());
        DataInfo info{result.data() + 1, (const char *)f.activations.data(),
                      size_t(stride), f.inputStride, 1, 3, mapping, size_t(plane)};
        auto kernel = GetMulMatFunction(GGML_TYPE_IQ2_S, inputs);
        kernel(f.columns, f.weights.data(), f.weightStride, info, f.outputs);
        for (int i = 1; i <= inputs; ++i) {
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
}

void TestLinearDispatch() {
    const auto at = static_cast<fastllm::DataType>(int(fastllm::DataType::DATA_GGUF_FORMAT) + GGML_TYPE_Q8_K);
    const auto wt = static_cast<fastllm::DataType>(int(fastllm::DataType::DATA_GGUF_FORMAT) + GGML_TYPE_IQ2_S);
    for (int inputs : {1, 2, 8, 9, 17}) {
        Fixture f(2560, 79, inputs, false, false);
        std::vector<float> result(inputs * f.outputs, sentinel), bias(f.outputs);
        for (int i = 0; i < f.outputs; ++i) bias[i] = (i - 40) * 0.25f;
        Check(fastllm::LinearQ8K_GGUF_Kernel(f.activations.data(), f.weights.data(), bias.data(),
              result.data(), inputs, f.columns, f.outputs, 3, 76, at, wt), "Linear dispatch failed");
        for (int row = 0; row < inputs; ++row) {
            for (int col = 0; col < f.outputs; ++col) {
                if (col >= 3 && col < 76) {
                    CheckReference(result[row * f.outputs + col], f, col, row, bias[col]);
                }
                else Check(result[row * f.outputs + col] == sentinel, "Linear dispatch wrote outside its columns");
            }
        }
    }
}

void TestCodebooks() {
    const auto kernel = GetMulMatFunction(GGML_TYPE_IQ2_S, 1);
    const auto decode = ggml_type_to_float(GGML_TYPE_IQ2_S);
    for (int code = 0; code < 1024; ++code)
    for (int pattern = 0; pattern < 4; ++pattern) {
        block_iq2_s x{}; block_q8_K y{};
        x.d = 0x3c00; y.d = 1.0f;
        for (int g = 0; g < 8; ++g) {
            x.scales[g] = uint8_t(g * 37 + code);
            for (int j = 0; j < 4; ++j) {
                int index = (code + g * 71 + j * 19) % 1024;
                x.qs[4 * g + j] = index & 255;
                x.qh[g] |= (index >> 8) << (2 * j);
                x.qs[32 + 4 * g + j] = uint8_t(pattern * 85 + g * 37 + j * 19);
            }
        }
        const int8_t endpoints[] = {-128, -127, -1, 0, 1, 126, 127};
        for (int j = 0; j < QK_K; ++j)
            y.qs[j] = pattern == 0 ? -128 : pattern == 1 ? 127 :
                      pattern == 2 ? endpoints[j % 7] : int((j * 31 + code) % 255) - 127;
        float result = 0, decoded[QK_K];
        DataInfo info{&result, (const char *)&y, 1, sizeof(y), 0, 1, nullptr, 0};
        kernel(QK_K, &x, sizeof(x), info, 1);
        decode(&x, decoded, QK_K);
        double expected = 0, magnitude = 0;
        for (int j = 0; j < QK_K; ++j) {
            const double product = double(decoded[j]) * y.qs[j];
            expected += product; magnitude += std::abs(product);
        }
        Near(result, expected, magnitude);
    }
}
} // namespace

int main() {
    if (GetMulMatFunction(GGML_TYPE_IQ2_S, 1) == nullptr) {
#if defined(__AVX2__) && \
    ((defined(__FMA__) && defined(__F16C__)) || defined(_MSC_VER))
        std::fputs("FAIL: Missing IQ2_S kernel in a supported x86 build\n", stderr);
        return 1;
#else
        std::puts("IQ2_S native kernel requires AVX2, FMA and F16C");
        return 77;
#endif
    }
    try {
        TestDirect();
        TestMappedRows();
        TestLinearDispatch();
        TestCodebooks();
        std::puts("PASS: IQ2_S native-layout CPU kernel, tails, strides, row mapping and linear dispatch");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
