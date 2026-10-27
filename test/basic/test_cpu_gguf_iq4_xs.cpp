#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include "gguf.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fastllm {
    extern CPUInstructInfo cpuInstructInfo;
    void Float32ToBFloat16(float *, uint16_t *, int);
    bool FastllmGemmBFloat16IQ4XS_AVX512BF16(
        const void *, long, const void *, long, void *, long,
        int, int, int, int, int);
}

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Test(int columns, int rows, int start, int end, bool padded, bool zero) {
    using namespace fastllm;
    const size_t payload = ggml_row_size(GGML_TYPE_IQ4_XS, columns);
    const size_t stride = payload + (padded ? 16 : 0);
    std::vector<uint8_t> weight(rows * stride, 0xcd);
    std::vector<uint16_t> input(columns), decoded(rows * columns);
    std::vector<float> unpacked(columns), expected(rows + 4, -9876), actual(rows + 4, -9876);
    uint32_t state = 17389;
    auto random = [&]() { state = state * 1664525U + 1013904223U; return state; };
    for (auto &x : input) {
        const float value = zero ? 0 : int(random() % 2001) / 257.0f - 4;
        x = Float32ToBFloat16RNEBits(value);
    }
    for (int r = 0; r < rows; ++r) {
        auto *blocks = reinterpret_cast<block_iq4_xs *>(weight.data() + r * stride);
        for (int b = 0; b < columns / QK_K; ++b) {
            auto &q = blocks[b];
            q.d = float_to_half(std::ldexp(float(int(random() % 65) - 32), -12));
            q.scales_h = random();
            for (auto &x : q.scales_l) x = random() >> 24;
            for (auto &x : q.qs) x = random() >> 24;
        }
        dequantize_row_iq4_xs(blocks, unpacked.data(), columns);
        Float32ToBFloat16(unpacked.data(), decoded.data() + r * columns, columns);
    }
    // The original fallback is an independent dequantizer + BF16 conversion
    // followed by the existing dense dot. Require identical FP32 output bits.
    MultiThreadLinearBFloat16BFloat16Op(input.data(), decoded.data(), nullptr,
        expected.data(), 1, columns, rows, start, end).Run();
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::fill(actual.begin(), actual.end(), -9876);
        Check(FastllmGemmBFloat16IQ4XS_AVX512BF16(input.data(), columns * 2,
            weight.data(), stride, actual.data(), rows * 4,
            1, columns, rows, start, end), "eligible IQ4_XS GEMV rejected");
        Check(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "IQ4_XS GEMV differs from dequantized BF16 reference or overwrote a guard");
    }
    for (int r = start; r < end; ++r) {
        double sum = 0, magnitude = 0;
        for (int c = 0; c < columns; ++c) {
            const double product = double(BFloat16BitsToFloat32(input[c])) *
                                   BFloat16BitsToFloat32(decoded[r * columns + c]);
            sum += product;
            magnitude += std::fabs(product);
        }
        Check(std::isfinite(actual[r]) && std::fabs(actual[r] - sum) <= 1e-6 * magnitude + 1e-7,
            "IQ4_XS GEMV differs from scalar BF16 dot reference");
    }
    if (!padded) {
        std::fill(actual.begin(), actual.end(), -9876);
        MultiThreadGemmOp(reinterpret_cast<uint8_t *>(input.data()), BFLOAT16, weight.data(),
            DataType(DATA_GGUF_FORMAT + GGML_TYPE_IQ4_XS),
            reinterpret_cast<uint8_t *>(actual.data()), FLOAT32,
            1, columns, rows, start, end).Run();
        Check(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "GEMM dispatcher did not preserve IQ4_XS decode results");
    }
    Check(!FastllmGemmBFloat16IQ4XS_AVX512BF16(input.data(), columns * 2,
        weight.data(), stride, actual.data(), rows * 4, 2, columns, rows, start, end),
        "prefill should retain its existing fallback");
    Check(!FastllmGemmBFloat16IQ4XS_AVX512BF16(input.data(), columns * 2,
        weight.data(), stride, actual.data(), rows * 4, 1, 128, rows, start, end),
        "partial IQ4_XS block was accepted");
}

int main() {
    try {
        if (!fastllm::cpuInstructInfo.hasAVX512BF16) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: IQ4_XS GEMV requires AVX512 BF16");
            return 0;
        }
        for (int columns : {256, 512, 2048, 4096}) {
            for (int rows : {1, 3, 4, 7, 17, 129}) {
                Test(columns, rows, 0, rows, false, false);
                Test(columns, rows, 0, rows, true, true);
                if (rows > 3) Test(columns, rows, 1, rows - 1, true, false);
            }
        }
        std::puts("PASS: IQ4_XS BF16 direct GEMV exact fallback and scalar reference");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
