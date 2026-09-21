#include "fastllm.h"
#include "executor.h"
#include "devices/numas/numasdevice.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
namespace fastllm { CPUInstructInfo *GetCPUInstructInfo(); }

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

// Sparse rows give a scalar oracle without reusing GEMM, weight packing,
// task partitioning or expert grouping from the implementation under test.
static int SourceColumn(int expert, int part, int row, int columns) {
    return (row * 17 + expert * 13 + part * 7) % columns;
}

static float WeightValue(int expert, int part, int row, int inter) {
    const float global = part ? .125f : (row < inter ? .5f : .25f);
    const float scale = ((expert + row) & 1) ? .5f : 1.f;
    return ((expert + row + part) % 3 == 0 ? -1.f : 1.f) * scale * global;
}

static float Bf16(float value) {
    // Independent finite FP32 -> BF16 round-to-nearest-even.
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void Run(int hidden, int inter, int experts, int topk, bool shared, DataType dtype) {
    constexpr float sharedScale = .375f;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights(2 * (experts + 1), nullptr), biases(weights);
    for (int expert = shared ? 0 : 1; expert <= experts; expert++) {
        for (int part = 0; part < 2; part++) {
            const int rows = part ? hidden : 2 * inter;
            const int cols = part ? inter : hidden;
            auto weight = std::make_unique<Data>(NVFP4_BLOCK_16_E4M3,
                                                  std::vector<int>{rows, cols});
            weight->blockK = 1;
            weight->blockM = 16;
            weight->Allocate();
            const size_t packedBytes = GetNVFP4WeightBytes(rows, cols);
            std::memset(weight->cpuData, 0, packedBytes);
            for (int row = 0; row < rows; row++) {
                const int col = SourceColumn(expert, part, row, cols);
                const uint8_t code = (expert + row + part) % 3 == 0 ? 0xa : 0x2;
                weight->cpuData[(size_t)row * (cols / 2) + col / 2] = code << ((col & 1) * 4);
                std::memset(weight->cpuData + packedBytes + (size_t)row * (cols / 16),
                            ((expert + row) & 1) ? 0x30 : 0x38, cols / 16);
            }
            weight->scales = part ? std::vector<float>{.125f} : std::vector<float>{.5f, .25f};
            weights[expert * 2 + part] = weight.get();
            owned.push_back(std::move(weight));
        }
    }
    Data input(dtype), ids(INT32), scores(FLOAT32), output(dtype);
    Data w1, w2, w3, curInput, curOutput;
    // Grow and shrink the same layer's cached buffers across the decode /
    // prefill boundary; start with unregistered compact weights as well.
    for (int rows : {16, 2, 64, 3, 31, 32, 1, 8, 33, 96, 4, 7}) {
        input.Resize({rows, hidden}); input.Allocate();
        ids.Resize({rows, topk}); ids.Allocate();
        scores.Resize({rows, topk}); scores.Allocate();
        std::vector<float> x(rows * hidden);
        for (int i = 0; i < (int)x.size(); i++) {
            x[i] = ((i * 23 + rows * 7) % 127 - 63) / 32.f;
            if (dtype == FLOAT32) ((float *)input.cpuData)[i] = x[i];
            else {
                uint32_t bits;
                std::memcpy(&bits, &x[i], sizeof(bits));
                ((uint16_t *)input.cpuData)[i] = bits >> 16;
            }
        }
        for (int mode = 0; mode < 4; mode++) {
            auto *index = (int32_t *)ids.cpuData;
            auto *score = (float *)scores.cpuData;
            for (int row = 0; row < rows; row++) for (int k = 0; k < topk; k++) {
                // Dispersed, identical, skewed and duplicate-id routes.
                int e = mode == 1 ? k * 7 : (row * 11 + k * 7);
                if (mode == 2 && (row % 3 || k < 2)) e = k;
                if (mode == 3) e = row % 3 + k / 2;
                index[row * topk + k] = e % experts;
                score[row * topk + k] = ((row + k * 3) % 9 - 4) / 8.f;
            }
            MergeMOE(input, ids, scores, weights, biases, w1, w2, w3,
                     curInput, curOutput, sharedScale, output, 0, MoeGateSwiglu,
                     false, 0.f, false, nullptr, 128);
            double error2 = 0, magnitude2 = 0;
            for (int row = 0; row < rows; row++) for (int col = 0; col < hidden; col++) {
                double expected = 0, magnitude = 0;
                for (int k = 0; k < topk + int(shared); k++) {
                    const int e = k < topk ? index[row * topk + k] + 1 : 0;
                    const float scoreValue = k < topk ? score[row * topk + k] : sharedScale;
                    const int mid = SourceColumn(e, 1, col, inter);
                    const float gate = x[row * hidden + SourceColumn(e, 0, mid, hidden)] *
                                       WeightValue(e, 0, mid, inter);
                    const float up = x[row * hidden + SourceColumn(e, 0, mid + inter, hidden)] *
                                     WeightValue(e, 0, mid + inter, inter);
                    const float activated = Bf16((gate / (1.f + std::exp(-gate))) * up);
                    const double term = activated * WeightValue(e, 1, col, inter) * scoreValue;
                    expected += term;
                    magnitude += std::fabs(term);
                }
                float actual;
                if (dtype == FLOAT32) actual = ((float *)output.cpuData)[row * hidden + col];
                else {
                    uint32_t bits = uint32_t(((uint16_t *)output.cpuData)[row * hidden + col]) << 16;
                    std::memcpy(&actual, &bits, sizeof(actual));
                    expected = Bf16(float(expected));
                }
                const double error = actual - expected;
                // SIMD exp approximations can cross a BF16 rounding boundary.
                Require(std::isfinite(actual) && std::fabs(error) <= 1e-6 + .012 * magnitude,
                        "NUMA NVFP4 MoE differs from sparse scalar reference");
                error2 += error * error;
                magnitude2 += magnitude * magnitude;
            }
            Require(error2 <= 1e-8 + 1e-5 * magnitude2, "NUMA NVFP4 MoE reference error too large");
        }
    }
    for (const auto &weight : owned) {
        if (weight->numasData.empty()) continue;
        Require(weight->dataType == NVFP4_BLOCK_16_E4M3_PACKED,
                "NUMA NVFP4 expanded FP8 scales");
    }
    ClearNumasMoeRuntimeCache();
    std::printf("PASS hidden=%d inter=%d experts=%d topk=%d shared=%d dtype=%d: routes, batch transitions, compact scales\n",
                hidden, inter, experts, topk, shared, int(dtype));
}

static void CheckDuplicateDecode() {
    constexpr int hidden = 256, inter = 128, experts = 7, rows = 31, topk = 4;
    std::mt19937 rng(20260920);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights(2 * (experts + 1), nullptr), biases(weights);
    for (int e = 1; e <= experts; e++) for (int part = 0; part < 2; part++) {
        int h = part ? hidden : 2 * inter, w = part ? inter : hidden;
        auto weight = std::make_unique<Data>(NVFP4_BLOCK_16_E4M3, std::vector<int>{h, w});
        weight->blockK = 1; weight->blockM = 16; weight->Allocate();
        size_t packed = GetNVFP4WeightBytes(h, w);
        for (size_t i = 0; i < packed; i++) weight->cpuData[i] = rng();
        for (size_t i = packed; i < weight->GetBytes(); i++) weight->cpuData[i] = 0x28 + rng() % 24;
        weight->scales = part ? std::vector<float>{.031f} : std::vector<float>{.037f, .071f};
        weights[e * 2 + part] = weight.get();
        owned.push_back(std::move(weight));
    }
    std::vector<uint16_t> input(rows * hidden), expected(rows * hidden);
    std::vector<int32_t> ids(rows * topk);
    std::vector<float> scores(rows * topk);
    for (auto &v : input) v = Float32ToBFloat16RNEBits((int(rng() % 127) - 63) / 64.f);
    for (int row = 0; row < rows; row++) for (int k = 0; k < topk; k++) {
        ids[row * topk + k] = k < 2 ? 0 : (row + k) % experts;
        scores[row * topk + k] = (k + 1) / 16.f;
    }
    Data output(BFLOAT16), w1, w2, w3, curInput, curOutput;
    auto run = [&](int start, int count) {
        Data x(BFLOAT16, {count, hidden}, DataDevice::CPU, input.data() + start * hidden);
        Data index(INT32, {count, topk}, DataDevice::CPU, ids.data() + start * topk);
        Data score(FLOAT32, {count, topk}, DataDevice::CPU, scores.data() + start * topk);
        MergeMOE(x, index, score, weights, biases, w1, w2, w3, curInput, curOutput,
                 0.f, output, 0, MoeGateSwiglu, false, 0.f, false, nullptr, 128);
    };
    for (int row = 0; row < rows; row++) {
        run(row, 1);
        std::memcpy(expected.data() + row * hidden, output.cpuData, hidden * sizeof(uint16_t));
    }
    run(0, rows);
    Require(std::memcmp(expected.data(), output.cpuData, expected.size() * sizeof(uint16_t)) == 0,
            "duplicate decode routes changed single-row output bits");
    ClearNumasMoeRuntimeCache();
    std::puts("PASS duplicate decode groups preserve single-row output bits");
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--avx2") == 0) {
            auto *info = GetCPUInstructInfo();
            if (!info->hasAVX2) return 77;
            info->hasAVX512F = info->hasAVX512BF16 = info->hasAVX512VNNI = false;
            info->hasAMX = false;
        }
        ((Executor *)GetExecutor())->SetFirstDevice("numa");
        // Vary shard tails, expert counts and fanout independently of model
        // dimensions. Include widths without a 64-column exact partition.
        const int shapes[][4] = {
            {96, 64, 11, 2}, {256, 128, 29, 5}, {256, 160, 29, 5},
            {352, 224, 17, 7}, {576, 320, 19, 4}
        };
        for (const auto &shape : shapes) for (bool shared : {false, true})
            for (DataType dtype : {FLOAT32, BFLOAT16})
                Run(shape[0], shape[1], shape[2], shape[3], shared, dtype);
        CheckDuplicateDecode();
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &e) {
        ClearNumasMoeRuntimeCache();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
