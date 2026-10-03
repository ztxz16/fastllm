#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "devices/numas/numasdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;
namespace fastllm { void RegisterNumas(Data *, std::string); }

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static float ReadBf16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float Bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// One nonzero per row supplies an independent scalar oracle. Gate/up are
// initially separate halves, as in the GLM weights loaded onto NUMA.
static void FillWeight(Data &weight, bool down, int inter) {
    weight.blockK = 1;
    weight.blockM = 16;
    weight.Allocate();
    std::memset(weight.cpuData, 0, weight.GetBytes());
    const int rows = weight.dims[0], cols = weight.dims[1];
    const size_t stride = GetDataBytes(weight.dataType, 1, cols);
    for (int row = 0; row < rows; row++) {
        const int col = (row * 17 + (down ? 7 : 0)) % cols;
        const float scale = down ? .125f : (row < inter ? .5f : .25f);
        auto *dst = weight.cpuData + row * stride;
        const bool compact = weight.dataType == NVFP4_BLOCK_16_E4M3_PACKED;
        const int blockBytes = compact ? 9 : 12;
        if (compact) {
            std::memcpy(dst, &scale, sizeof(scale));
            dst += sizeof(scale);
        }
        for (int b = 0; b < cols / 16; b++) {
            if (compact) dst[b * blockBytes + 8] = 0x38; // E4M3 scale = 1.
            else std::memcpy(dst + b * blockBytes + 8, &scale, sizeof(scale));
        }
        dst[col / 16 * blockBytes + (col % 16) / 2] = 2 << ((col & 1) * 4);
    }
}

static void Run(DataType type, int rows, bool clamped) {
    constexpr int hidden = 256, inter = 128;
    Data gate(type, {2 * inter, hidden}), down(type, {hidden, inter});
    FillWeight(gate, false, inter);
    FillWeight(down, true, inter);
    std::vector<Data *> weights{nullptr, nullptr, &gate, &down}, biases(4, nullptr);
    Data input(BFLOAT16, {rows, hidden}), ids(INT32, {rows, 1}), scores(FLOAT32, {rows, 1});
    input.Allocate(); ids.Allocate(); scores.Allocate();
    std::vector<float> x(rows * hidden);
    for (int i = 0; i < rows * hidden; i++) {
        x[i] = (i % 37 - 18) / 8.f;
        ((uint16_t *)input.cpuData)[i] = Float32ToBFloat16RNEBits(x[i]);
    }
    for (int row = 0; row < rows; row++) {
        ((int *)ids.cpuData)[row] = 0;
        ((float *)scores.cpuData)[row] = 1.f;
    }
    input.ToDevice(DataDevice::CUDA, {0}, true);
    ids.ToDevice(DataDevice::CUDA, {0}, true);
    scores.ToDevice(DataDevice::CUDA, {0}, true);
    Data output(BFLOAT16), w1, w2, w3, curInput, curOutput;
    auto run = [&]() {
        ((Executor *)GetExecutor())->SetFirstDevice("numa");
        MergeMOE(input, ids, scores, weights, biases, w1, w2, w3, curInput, curOutput,
                 0.f, output, 0, MoeGateSwiglu, false, clamped ? .25f : 0.f,
                 clamped, nullptr, 128);
        Require(output.dataDevice == DataDevice::CUDA, "Expected CUDA hybrid output");
        output.ToDevice(DataDevice::CPU);
        std::vector<uint16_t> result(rows * hidden);
        std::memcpy(result.data(), output.cpuData, result.size() * sizeof(uint16_t));
        return result;
    };
    // Crucially, no CPU decode or eager registration precedes the first call.
    const auto cold = run();
    RegisterNumas(&gate, "linearSwiglu");
    RegisterNumas(&down, "linearColumn");
    const auto warm = run();
    Require(cold == warm, "Cold GPU expert output changes after NUMA registration");
    for (int row = 0; row < rows; row++) for (int col = 0; col < hidden; col++) {
        const float actual = ReadBf16(cold[row * hidden + col]);
        Require(std::isfinite(actual), "Nonfinite hybrid expert output");
        if (!clamped) {
            const int mid = (col * 17 + 7) % inter;
            const float g = x[row * hidden + mid * 17 % hidden] * .5f;
            const float u = x[row * hidden + (mid + inter) * 17 % hidden] * .25f;
            const float expected = Bf16(Bf16(g / (1.f + std::exp(-g)) * u) * .125f);
            Require(std::abs(actual - expected) <= 1e-6f + .012f * std::abs(expected),
                    "Cold hybrid output differs from sparse scalar reference");
        }
    }
    ClearNumasMoeRuntimeCache();
    std::printf("PASS type=%d rows=%d clamped=%d: cold/warm identical%s\n",
                int(type), rows, clamped, clamped ? "" : ", scalar reference correct");
}

int main() {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    try {
        // The only expert must first execute on CUDA, irrespective of the
        // runtime speed tracker or the host's CPU/GPU balance.
        setenv("FT_GPU_PREFILL", "1", 1);
        setenv("FT_EXPERT_LIMIT", "0", 1);
        SetDeviceMap({{"cuda:0", 1}});
        // Raw compact E4M3 takes the CPU fallback before packing; the existing
        // CPU regression covers it. Both formats here enter CUDA immediately.
        for (DataType type : {NVFP4_BLOCK_16, NVFP4_BLOCK_16_E4M3_PACKED})
            for (int rows : {32, 64, 129})
                for (bool clamped : {false, true}) Run(type, rows, clamped);
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &e) {
        ClearNumasMoeRuntimeCache();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
