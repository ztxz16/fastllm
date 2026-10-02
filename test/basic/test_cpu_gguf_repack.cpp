#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

// Decode the canonical bit planes independently, then write the established
// R4 layout. This also covers scale bits and arbitrary half bit patterns.
void Reference(ggml_type type, int rows, int columns, const uint8_t *src, uint8_t *dst) {
    const int blocks = columns / QK_K;
    for (int row = 0; row < rows; row += 4) {
        for (int b = 0; b < blocks; ++b) {
            for (int k = 0; k < 4; ++k) {
                uint8_t values[QK_K];
                if (type == GGML_TYPE_Q2_K) {
                    const auto &x = ((const block_q2_K *)src)[(row + k) * blocks + b];
                    auto &y = ((block_q2_k_r4 *)dst)[row / 4 * blocks + b];
                    y.d[k] = x.d; y.d[k + 4] = x.dmin;
                    for (int i = 0; i < 16; ++i) y.scales[4 * i + k] = x.scales[i];
                    for (int i = 0; i < QK_K; ++i)
                        values[i] = (x.qs[i / 128 * 32 + i % 32] >> (2 * (i % 128 / 32))) & 3;
                    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) {
                        y.qs[32 * i + 4 * k + j] = values[32*i+j] | (values[32*i+j+4] << 2) |
                            (values[32*i+j+8] << 4) | (values[32*i+j+12] << 6);
                        y.qs[32 * i + 4 * k + j + 16] = values[32*i+j+16] | (values[32*i+j+20] << 2) |
                            (values[32*i+j+24] << 4) | (values[32*i+j+28] << 6);
                    }
                } else {
                    const auto &x = ((const block_q4_K *)src)[(row + k) * blocks + b];
                    auto &y = ((block_q4_k_r4 *)dst)[row / 4 * blocks + b];
                    y.d[k] = x.d; y.d[k + 4] = x.dmin;
                    for (int i = 0; i < QK_K; ++i)
                        values[i] = (x.qs[i / 64 * 32 + i % 32] >> (4 * (i % 64 / 32))) & 15;
                    for (int i = 0; i < 8; ++i) {
                        const uint8_t d = i < 4 ? x.scales[i] & 63 :
                            (x.scales[i+4] & 15) | ((x.scales[i-4] >> 6) << 4);
                        const uint8_t m = i < 4 ? x.scales[i+4] & 63 :
                            (x.scales[i+4] >> 4) | ((x.scales[i] >> 6) << 4);
                        y.scales_l[4*i+k] = (d & 15) | ((m & 15) << 4);
                        y.scales_h[(4*i+k)%16] |= ((d >> 4) | ((m >> 4) << 2)) << (4*((4*i+k)/16));
                        for (int j = 0; j < 4; ++j) {
                            y.qs[64*i+4*k+j] = values[32*i+j] | (values[32*i+j+8] << 4);
                            y.qs[64*i+4*k+j+16] = values[32*i+j+16] | (values[32*i+j+24] << 4);
                            y.qs[64*i+4*k+j+32] = values[32*i+j+4] | (values[32*i+j+12] << 4);
                            y.qs[64*i+4*k+j+48] = values[32*i+j+20] | (values[32*i+j+28] << 4);
                        }
                    }
                }
            }
        }
    }
}

void TestBytes(ggml_type type) {
    const auto *repack = get_repack_info(type);
    std::mt19937 random(20261001 + type);
    for (int rows : {4, 8, 12, 64}) for (int columns : {256, 512, 2304, 5120}) {
        const size_t bytes = rows * ggml_row_size(type, columns);
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<uint8_t> source(bytes + 68, 0xa5), result(bytes + 36, 0xa5), expected(bytes, 0);
            // Two-byte aligned, deliberately unaligned for SIMD loads/stores.
            for (size_t i = 0; i < bytes; ++i) source[i+34] = pattern == 0 ? 0 :
                pattern == 1 ? 255 : pattern == 2 ? uint8_t(i) : uint8_t(random());
            const auto unchanged = source;
            Reference(type, rows, columns, source.data()+34, expected.data());
            repack->repack(rows, columns, (const char *)source.data()+34,
                (char *)result.data()+18, pattern & 1);
            Check(std::memcmp(expected.data(), result.data()+18, bytes) == 0, "R4 packed bytes differ");
            Check(source == unchanged, "Repack modified source");
            for (size_t i = 0; i < 18; ++i)
                Check(result[i] == 0xa5 && result[bytes+18+i] == 0xa5, "Repack overwrote output canary");
        }
    }
}

void TestLinear(ggml_type type) {
    using namespace fastllm;
    const int outputs = 32, columns = 512;
    std::vector<float> original(outputs * columns), bias(outputs);
    for (size_t i = 0; i < original.size(); ++i) original[i] = std::sin(i * .037f) * .03f;
    for (int i = 0; i < outputs; ++i) bias[i] = std::cos(i * .13f);
    Data weight(DATA_GGUF_FORMAT, type, {outputs, columns});
    weight.CreateFromOriData(WeightType::LINEAR, FLOAT32, (uint8_t *)original.data(), nullptr, nullptr);
    Data reference(DATA_GGUF_FORMAT, type, {outputs, columns});
    reference.Allocate(false);
    std::memset(reference.cpuData, 0, reference.GetBytes());
    Reference(type, outputs, columns, weight.cpuData, reference.cpuData);
    const auto packedType = get_repack_info(type)->new_type;
    ((ggml_tensor *)reference.ggmlTensor)->type = packedType;
    reference.ggmlType = packedType; reference.IsRepacked = true;
    weight.Repack();
    Check(std::memcmp(weight.cpuData, reference.cpuData, weight.GetBytes()) == 0, "Data::Repack bytes differ");
    weight.Repack();
    Check(std::memcmp(weight.cpuData, reference.cpuData, weight.GetBytes()) == 0, "Repack is not idempotent");
    for (int rows : {1, 2, 3, 6, 8, 17, 64}) {
        std::vector<float> input(rows * columns), actual(rows * outputs), expected(rows * outputs);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * .019f);
        RunLinearFloat32GGUF(input.data(), weight.cpuData, actual.data(), bias.data(), &weight,
            rows, columns, outputs, GetAlivePool(), 0, 4);
        RunLinearFloat32GGUF(input.data(), reference.cpuData, expected.data(), bias.data(), &reference,
            rows, columns, outputs, GetAlivePool(), 0, 4);
        Check(std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0, "Linear output differs");
    }
}
}

int main() {
    if (!get_repack_info(GGML_TYPE_Q2_K) || !get_repack_info(GGML_TYPE_Q4_K)) return 77;
    fastllm::SetThreads(4);
    fastllm::EnableAMX(false);
    try {
        for (auto type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K}) { TestBytes(type); TestLinear(type); }
        std::puts("PASS: Q2_K/Q4_K R4 byte-exact repack and multi-row linear");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
