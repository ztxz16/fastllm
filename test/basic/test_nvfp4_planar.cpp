#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastllm { CPUInstructInfo *GetCPUInstructInfo(); }

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
            Require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0,
                    "planar CPU GEMM changed output bits");
        }
    }
    info->hasAVX512BF16 = originalAvx512;
    std::printf("PASS columns=%d cross_swiglu=%d avx512=%d: equal storage, weight/scale bits and CPU outputs\n",
                columns, crossSwiglu, originalAvx512 && useAvx512);
}

int main() {
    try {
        for (int columns : {17, 33, 128, 640})
            for (bool cross : {false, true})
                for (bool avx512 : {false, true}) Run(columns, cross, avx512);
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
