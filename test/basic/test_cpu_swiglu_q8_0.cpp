#include "gguf.h"
#include "devices/cpu/computeutils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace fastllm;
DataType GgufType(ggml_type type) {
    return static_cast<DataType>(int(DATA_GGUF_FORMAT) + int(type));
}
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void Test(bool quantizedGate, int rows, int shards, bool partial, bool zeros,
          DataType destinationType) {
    constexpr int columns = 2560, gateColumns = 1280, intermediate = 640;
    std::mt19937 random(1024 + rows + shards);
    std::vector<float> input(rows * columns), weights(gateColumns * columns);
    for (float &v : input) v = zeros ? 0.0f : (int(random() % 2049) - 1024) / 1024.0f;
    DataType inputType = FLOAT32, weightType = FLOAT32;
    uint8_t *inputData = reinterpret_cast<uint8_t *>(input.data());
    uint8_t *weightData = reinterpret_cast<uint8_t *>(weights.data());
    std::vector<uint8_t> packedInput, packedWeights;
    if (quantizedGate) {
        inputType = GgufType(GGML_TYPE_Q8_K);
        weightType = GgufType(GGML_TYPE_IQ2_XS_R4);
        packedInput.resize(GetDataBytes(inputType, rows, columns));
        ConvertFromFloat32(packedInput.data(), inputType, input.data(), rows, columns);
        std::vector<block_iq2_xs> original(gateColumns * columns / QK_K);
        for (auto &block : original) {
            block.d = 0x1400; // finite, nonzero half scale
            for (auto &q : block.qs) q = random();
            for (auto &s : block.scales) s = random();
        }
        packedWeights.resize(original.size() * sizeof(block_iq2_xs));
        const auto *repack = get_repack_info(GGML_TYPE_IQ2_XS);
        Check(repack && repack->new_type == GGML_TYPE_IQ2_XS_R4, "Missing IQ2_XS repack");
        repack->repack(gateColumns, columns, reinterpret_cast<const char *>(original.data()),
                       reinterpret_cast<char *>(packedWeights.data()), false);
        inputData = packedInput.data();
        weightData = packedWeights.data();
    } else {
        for (float &v : weights) v = (int(random() % 2049) - 1024) / 65536.0f;
    }

    // Reference uses the existing unfused GEMM/SwiGLU and full-row converter.
    std::vector<float> referenceGate(rows * gateColumns), referenceSwiglu(rows * intermediate);
    MultiThreadGemmAndCrossSwigluOp reference(
        inputData, inputType, weightData, weightType,
        reinterpret_cast<uint8_t *>(referenceGate.data()), FLOAT32,
        referenceSwiglu.data(), rows, columns, gateColumns, 0, gateColumns, 0);
    reference.Run();
    const size_t rowBytes = GetDataBytes(destinationType, 1, intermediate);
    const size_t blockBytes = GetDataBytes(destinationType, 1, 32);
    std::vector<uint8_t> expected(rows * rowBytes);
    ConvertFromFloat32(expected.data(), destinationType, referenceSwiglu.data(), rows, intermediate);

    constexpr float sentinel = -123456.0f;
    std::vector<float> gate(rows * gateColumns, sentinel), swiglu(rows * intermediate, sentinel);
    std::vector<uint8_t> destination(64 + rows * rowBytes + 64, 0xa5);
    const size_t weightRowBytes = GetDataBytes(weightType, 1, columns);
    const int first = partial ? 64 : 0, last = gateColumns - (partial ? 64 : 0);
    const int shardColumns = gateColumns / shards;
    // Execute ranges in reverse order to catch dependencies on an earlier
    // worker. Every range owns complete 32-element activation blocks.
    for (int shard = shards - 1; shard >= 0; --shard) {
        const int base = shard * shardColumns;
        for (int end = shardColumns; end > 0; end -= 64) {
            const int st = end - 64;
            if (base + st < first || base + end > last) continue;
            MultiThreadGemmAndCrossSwigluOp fused(
                inputData, inputType, weightData + base * weightRowBytes, weightType,
                reinterpret_cast<uint8_t *>(gate.data() + base), FLOAT32,
                swiglu.data(), rows, columns, gateColumns, st, end, base,
                destination.data() + 64, destinationType);
            fused.Run();
        }
    }
    for (size_t i = 0; i < destination.size(); ++i) {
        bool written = i >= 64 && i < 64 + rows * rowBytes;
        size_t offset = written ? i - 64 : 0;
        int column = written ? int((offset % rowBytes) / blockBytes) * 32 : 0;
        written = written && column >= first / 2 && column < last / 2;
        Check(destination[i] == (written ? expected[offset] : uint8_t(0xa5)),
              "Fused group quantization differs from unfused reference or overwrites a canary");
    }
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < gateColumns; ++c)
            Check(gate[r * gateColumns + c] ==
                (c >= first && c < last ? referenceGate[r * gateColumns + c] : sentinel),
                "Gate/up result or shard boundary changed");
    Check(std::all_of(swiglu.begin(), swiglu.end(), [](float v) { return v == sentinel; }),
          "Fused converter wrote the unused full SwiGLU intermediate");
}
} // namespace

int main() {
    if (!get_repack_info(GGML_TYPE_IQ2_XS)) return 77;
    try {
        for (bool quantized : {false, true})
            for (int rows : {1, 3})
                for (int shards : {1, 2, 4})
                    for (bool partial : {false, true})
                        Test(quantized, rows, shards, partial, false, GgufType(GGML_TYPE_Q8_0));
        Test(true, 1, 2, false, true, GgufType(GGML_TYPE_Q8_0));
        Test(true, 3, 4, true, false, INF_INT8_GROUP32);
        std::puts("PASS: fused Q8_0 SwiGLU matches unfused bytes, shard boundaries and group32 conversion");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
