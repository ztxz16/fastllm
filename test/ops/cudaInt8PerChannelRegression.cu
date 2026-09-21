//
// Regression test for the signed symmetric INT8 per-channel CUDA paths that
// back compressed-tensors W8A8 / W8A16 checkpoints.
//
// Covers:
//   * fused signed GEMV (decode / MTP verify, n <= 8)
//   * per-token int8 activation quantization + integer tensor-core GEMM
//     (W8A8 prefill, n >= 9, including the 9..31 padding range)
//   * chunked int8 -> FP16 dequantization + FP16 GEMM (W8A16 prefill)
//   * bias handling and exactness against an INT32/FP32 CPU reference
//
#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {

void Expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void Check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

float HalfToFloat(uint16_t bits) {
    half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

std::vector<uint16_t> ReadHalf(const void *ptr, size_t count) {
    std::vector<uint16_t> result(count);
    Check(cudaMemcpy(result.data(), ptr, count * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost));
    return result;
}

struct Case {
    int n;
    int k;
    int m;
    bool w8a8;
};

// Reference: symmetric per-token int8 activation quantization (W8A8 prefill),
// FP16 activations for the fused GEMV decode path (n <= 8) and for every
// W8A16 projection.  Weights are int8 with per-output-channel scales and the
// accumulation is FP32 (W8A16) or exact INT64 (W8A8 quantized activations).
std::vector<float> Reference(const std::vector<float> &input,
                             const std::vector<int8_t> &weight,
                             const std::vector<float> &scales,
                             const std::vector<float> &bias, const Case &c,
                             bool quantizeActivation) {
    std::vector<float> quantized(input);
    std::vector<float> inputScales(c.n, 1.0f);
    if (quantizeActivation) {
        for (int row = 0; row < c.n; row++) {
            float maxAbs = 0.0f;
            for (int i = 0; i < c.m; i++) {
                maxAbs = std::max(maxAbs, std::fabs(input[(size_t)row * c.m + i]));
            }
            const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
            inputScales[row] = scale;
            const float inverse = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
            for (int i = 0; i < c.m; i++) {
                float value = std::rint(input[(size_t)row * c.m + i] * inverse);
                value = std::min(127.0f, std::max(-127.0f, value));
                quantized[(size_t)row * c.m + i] = value;
            }
        }
    }
    std::vector<float> output((size_t)c.n * c.k, 0.0f);
    for (int row = 0; row < c.n; row++) {
        for (int out = 0; out < c.k; out++) {
            double sum = 0.0;
            for (int i = 0; i < c.m; i++) {
                const float activation = quantizeActivation
                    ? quantized[(size_t)row * c.m + i]
                    : input[(size_t)row * c.m + i];
                sum += (double)activation * (double)weight[(size_t)out * c.m + i];
            }
            float value = (float)sum * inputScales[row] * scales[out];
            if (!bias.empty()) {
                value += bias[out];
            }
            output[(size_t)row * c.k + out] = value;
        }
    }
    return output;
}

void Run(int gpu) {
    FastllmCudaSetDevice(gpu);
    Executor executor;
    executor.SetFirstDevice("cuda:" + std::to_string(gpu));

    std::vector<Case> cases = {
        // gate/up: W8A8 (n small -> GEMV, n large -> integer GEMM)
        {1, 17408, 5120, true},
        {4, 17408, 5120, true},
        {8, 17408, 5120, true},
        // 9..31 exercises the padded integer GEMM
        {9, 17408, 5120, true},
        {17, 10240, 5120, true},
        {33, 10240, 5120, true},
        {128, 6144, 5120, true},
        {512, 6144, 5120, true},
        // down_proj / out_proj: W8A16
        {1, 5120, 17408, false},
        {8, 5120, 17408, false},
        {16, 5120, 17408, false},
        {64, 5120, 6144, false},
        {256, 5120, 6144, false},
        // non-multiple-of-sixteen input width falls back to the scalar GEMV
        {2, 512, 520, true},
    };

    for (const Case &c : cases) {
        std::mt19937 rng((unsigned)(c.n * 131 + c.k * 17 + c.m));
        std::vector<int8_t> weight((size_t)c.k * c.m);
        for (auto &value : weight) {
            value = (int8_t)((int)(rng() % 255) - 127);
        }
        std::vector<float> scales(c.k);
        for (auto &value : scales) {
            value = std::ldexp(0.5f + (float)(rng() % 512) / 1024.0f,
                               (int)(rng() % 7) - 11);
        }
        std::vector<float> input((size_t)c.n * c.m);
        std::vector<half> inputHalf(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = ((float)((int)(rng() % 2001) - 1000)) / 250.0f;
            inputHalf[i] = __float2half_rn(input[i]);
            input[i] = __half2float(inputHalf[i]);
        }
        std::vector<float> bias(c.k);
        for (auto &value : bias) {
            value = ((float)((int)(rng() % 101) - 50)) / 50.0f;
        }

        for (bool withBias : {false, true}) {
            Data weightData(
                c.w8a8 ? DataType::INT8_PERCHANNEL_S8
                       : DataType::INT8_PERCHANNEL_S8_W8A16,
                {c.k, c.m});
            weightData.scales = scales;
            weightData.perChannelAxis = 0;
            weightData.Allocate();
            std::memcpy(weightData.cpuData, weight.data(), weight.size());
            weightData.ToDevice(DataDevice::CUDA, std::vector<int>{gpu});

            Data inputData(DataType::FLOAT16, {c.n, c.m});
            inputData.Allocate();
            std::memcpy(inputData.cpuData, inputHalf.data(),
                        inputHalf.size() * sizeof(half));
            inputData.ToDevice(DataDevice::CUDA, std::vector<int>{gpu});

            Data biasData(DataType::FLOAT32);
            if (withBias) {
                biasData.Resize({c.k});
                biasData.Allocate();
                std::memcpy(biasData.cpuData, bias.data(),
                            bias.size() * sizeof(float));
                biasData.ToDevice(DataDevice::CUDA, std::vector<int>{gpu});
            }

            Data outputData(DataType::FLOAT16, {c.n, c.k});
            outputData.Allocate();
            outputData.ToDevice(DataDevice::CUDA, std::vector<int>{gpu});

            const bool launched = c.w8a8
                ? FastllmCudaHalfMatMulFloatInt8PerChannelS8(
                      inputData, weightData, biasData, outputData, c.n, c.m, c.k)
                : FastllmCudaHalfMatMulFloatInt8PerChannelS8W8A16(
                      inputData, weightData, biasData, outputData, c.n, c.m, c.k);
            Expect(launched, "int8 per-channel linear returned false");
            Check(cudaDeviceSynchronize());

            std::vector<uint16_t> got =
                ReadHalf(outputData.cudaData, (size_t)c.n * c.k);
            // The fused GEMV keeps FP16 activations; only the prefill integer
            // GEMM path quantizes them per token.
            const bool quantizeActivation =
                c.w8a8 && c.n > 8;
            std::vector<float> reference = Reference(
                input, weight, scales, withBias ? bias : std::vector<float>(),
                c, quantizeActivation);

            double maxError = 0.0;
            double maxValue = 1e-6;
            for (size_t i = 0; i < got.size(); i++) {
                const float value = HalfToFloat(got[i]);
                maxError = std::max(maxError,
                                    (double)std::fabs(value - reference[i]));
                maxValue = std::max(maxValue, (double)std::fabs(reference[i]));
            }
            const double relative = maxError / maxValue;
            std::cout << "n=" << c.n << " k=" << c.k << " m=" << c.m
                      << (c.w8a8 ? " W8A8" : " W8A16")
                      << (withBias ? " +bias" : "")
                      << " maxRelErr=" << relative << std::endl;
            Expect(relative < 2e-2, "int8 per-channel result out of tolerance");
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    // The W8A16 reference below assumes FP16 activations; force the opt-in
    // prefill activation quantization off so the check is independent of the
    // caller's environment.
    unsetenv("FASTLLM_INT8S8_W8A16_PREFILL_ACTIVATION");
    try {
        Run(gpu);
    } catch (const std::exception &error) {
        std::cerr << "int8 per-channel regression failed: " << error.what()
                  << std::endl;
        return 1;
    }
    std::cout << "int8 per-channel regression passed" << std::endl;
    return 0;
}