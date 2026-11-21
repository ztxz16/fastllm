#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-nvfp4-prefill.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

void LaunchFastllmGemmFp16FP8E4M3(half *input, uint8_t *weight, half *output,
                                 half *bias, float *scales, int n, int m,
                                 int k, int blockM, int blockK);

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void *data = nullptr;
    explicit Buffer(size_t bytes) { Check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer &operator=(const Buffer&) = delete;
};

void RunCase(int rows, int inputDim, int outputDim, bool withBias) {
    std::mt19937 random(123);
    std::vector<half> input(rows * inputDim), bias(outputDim);
    std::vector<uint8_t> weight((size_t)inputDim * outputDim);
    std::vector<float> scales((inputDim / 128) * ((outputDim + 127) / 128));
    for (auto &v : input) v = __float2half(float(int(random() % 201) - 100) / 64.0f);
    for (auto &v : bias) v = __float2half(float(int(random() % 101) - 50) / 32.0f);
    for (auto &v : weight) v = uint8_t((random() % 127) | ((random() & 1) << 7));
    std::uniform_real_distribution<float> scaleDistribution(1e-5f, 1e-2f);
    for (auto &v : scales) v = scaleDistribution(random);

    Buffer a(input.size() * sizeof(half)), b(weight.size()),
           s(scales.size() * sizeof(float)), biasBuffer(bias.size() * sizeof(half)),
           actual(rows * outputDim * sizeof(half)), reference(rows * outputDim * sizeof(half));
    Check(cudaMemcpy(a.data, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(b.data, weight.data(), weight.size(), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(s.data, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(biasBuffer.data, bias.data(), bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    auto *biasPtr = withBias ? static_cast<half*>(biasBuffer.data) : nullptr;
    LaunchFastllmGemmFp16FP8E4M3(static_cast<half*>(a.data), static_cast<uint8_t*>(b.data),
        static_cast<half*>(actual.data), biasPtr, static_cast<float*>(s.data),
        rows, inputDim, outputDim, 128, 128);
    // Independent one-row GEMVs retain the established dispatch and must
    // agree bit for bit with each row of speculative verification.
    for (int row = 0; row < rows; ++row) {
        LaunchFastllmGemmFp16FP8E4M3(static_cast<half*>(a.data) + row * inputDim,
            static_cast<uint8_t*>(b.data), static_cast<half*>(reference.data) + row * outputDim,
            biasPtr, static_cast<float*>(s.data), 1, inputDim, outputDim, 128, 128);
    }
    Check(cudaDeviceSynchronize());
    std::vector<uint16_t> observed(rows * outputDim), expected(rows * outputDim);
    Check(cudaMemcpy(observed.data(), actual.data, observed.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(expected.data(), reference.data, expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (observed != expected) throw std::runtime_error("FP8 verification differs from independent rows");
    std::printf("FP8 n=%d K=%d N=%d bias=%d bitwise PASS\n", rows, inputDim, outputDim, withBias);
}

// Compare each request with the same single-request verifier, including its
// K reduction order. The 24-row down projection has a distinct output tile.
void RunBf16VerifyCase(int device, int inner, int columns) {
    using namespace fastllm;
    std::mt19937 rng(559);
    Data weight(FP8_E4M3, {columns, inner});
    weight.blockM = inner;
    weight.blockK = 1;
    weight.scales.resize(columns);
    for (int i = 0; i < columns; ++i) weight.scales[i] = .0312f + .00013f * (i % 97);
    weight.ToDevice(CUDA, {device}, false);
    weight.Allocate(false);
    std::vector<uint8_t> bytes(size_t(columns) * inner);
    for (auto &v : bytes) {
        const auto bits = rng();
        v = (bits & 128u) | ((bits >> 8) % 120);
    }
    Check(cudaMemcpy(weight.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    std::vector<float> values(24 * inner);
    for (auto &v : values) v = (int(rng() % 65537) - 32768) / 8192.f;
    Data input(BFLOAT16, {24, inner}, values);
    input.ToDevice(CUDA, {device}, true);
    auto project = [&](int first, int rows, uint16_t *result) {
        Data x, y(BFLOAT16, {rows, columns});
        x.FakeFrom(input, size_t(first) * inner * sizeof(uint16_t));
        x.Resize({rows, inner});
        y.ToDevice(CUDA, {device}, false);
        y.Allocate(false);
        FastllmCudaBFloat16MatMulFP8E4M3(x, weight, Data(), y, rows, inner, columns);
        FastllmCudaSyncCurrentThreadStream();
        Check(cudaMemcpy(result, y.cudaData, size_t(rows) * columns * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    };
    FastllmNvfp4PrefillPlan plan;
    for (int steps : {2, 3}) {
        std::vector<uint16_t> expected(8 * steps * columns);
        for (int b = 0; b < 8; ++b) project(b * steps, steps, expected.data() + b * steps * columns);
        for (int batch = 2; batch <= 8; ++batch) {
            std::vector<uint16_t> actual(batch * steps * columns);
            FastllmCudaNvfp4Scope scope(plan, true, 0, nullptr, steps);
            project(0, batch * steps, actual.data());
            if (!std::equal(actual.begin(), actual.end(), expected.begin()))
                throw std::runtime_error("Batched BF16/FP8 verifier differs from single-request verification");
        }
    }
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        for (int device = 0; device < devices; ++device) {
            Check(cudaSetDevice(device));
            std::printf("device=%d\n", device);
            cudaDeviceProp prop{};
            Check(cudaGetDeviceProperties(&prop, device));
            if (prop.major == 12 && prop.minor == 0) {
                RunBf16VerifyCase(device, 4096, 24576);
                RunBf16VerifyCase(device, 12288, 4096);
            }
            for (auto shape : std::vector<std::pair<int, int>>{
                    {128, 1}, {128, 3}, {128, 127}, {128, 128}, {128, 129}, {384, 255},
                    {5120, 17408}, {8704, 5120}, {5120, 8192}, {3072, 5120}, {5120, 7168}}) {
                for (int rows = 1; rows <= 8; ++rows) {
                    for (bool bias : {false, true}) RunCase(rows, shape.first, shape.second, bias);
                }
            }
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
