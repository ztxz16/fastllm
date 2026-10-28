// BF16 x block-32 MXFP8: an independent FP64 oracle, eager/graph equality,
// source-weight preservation, and prefill/tile boundaries. Run with memcheck too.
#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fastllm;

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Cuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

static float Bf16(float value) {
    return RoundFloat32ToBFloat16RNE(value);
}

static float DecodeE4M3(unsigned char code) {
    const int exponent = (code & 127) >> 3, mantissa = code & 7;
    const float magnitude = exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7)
                                     : std::ldexp(float(mantissa), -9);
    return code & 128 ? -magnitude : magnitude;
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        cudaDeviceProp properties{};
        Cuda(cudaGetDeviceProperties(&properties, 0));
        if (properties.major < 8) return 77;
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        std::mt19937 random(59);
        int cases = 0;
        std::vector<std::pair<int, int>> shapes = {
            {64, 128}, {128, 64}, {192, 256}, {256, 192}, {512, 5120},
            {1280, 5120}, {1792, 5120}, {4096, 1280}, {5120, 1024},
            {4608, 5120}, {5120, 2304}, {64, 512}, {192, 512}, {576, 5120}};
        // Both sides of the split-K dispatch boundary, plus a non-128 N tile.
        const int thinColumns = (properties.multiProcessorCount / 4) * 128;
        if (thinColumns > 0) {
            shapes.emplace_back(thinColumns, 512);
            shapes.emplace_back(thinColumns + 128, 512);
        }
        for (const auto &shape : shapes) {
            const int n = shape.first, k = shape.second;
            Data weight(DataType::FP8_E4M3, {n, k});
            weight.blockM = weight.blockK = 32;
            weight.Allocate();
            std::vector<unsigned char> bytes((size_t)n * k);
            for (auto &value : bytes) value = (random() % 2 ? 128 : 0) | (random() % 120);
            std::memcpy(weight.cpuData, bytes.data(), bytes.size());
            weight.scales.resize((size_t)(n / 32) * (k / 32));
            for (auto &value : weight.scales) value = std::ldexp(1.f, int(random() % 17) - 17);
            weight.ToDevice(DataDevice::CUDA, std::vector<int>{0});

            for (int m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 33, 64, 65, 128, 129, 257}) {
                std::vector<float> inputValues((size_t)m * k);
                for (auto &value : inputValues)
                    value = Bf16((int(random() % 2001) - 1000) * .0003f);
                Data input(DataType::FLOAT32, {m, k}, inputValues);
                ToDataType(input, DataType::BFLOAT16);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                Data output(DataType::BFLOAT16, {m, n});
                output.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                output.Allocate();
                auto launch = [&] {
                    Require(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(
                        input.cudaData, weight, output.cudaData, m, k, n), "MXFP8 dispatch failed");
                };
                launch();
                Cuda(cudaDeviceSynchronize());
                std::vector<unsigned char> eager((size_t)m * n * 2), replay(eager.size());
                Cuda(cudaMemcpy(eager.data(), output.cudaData, eager.size(), cudaMemcpyDeviceToHost));
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                launch();
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&executable, graph, 0));
                // Repeat small-M graphs to check that every split-K arrival
                // counter resets and no partial sum leaks into the next replay.
                for (int repeat = 0; repeat < (m <= 8 ? 8 : 1); ++repeat) {
                    Cuda(cudaMemset(output.cudaData, 0x7f, eager.size()));
                    Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                    Cuda(cudaDeviceSynchronize());
                    Cuda(cudaMemcpy(replay.data(), output.cudaData, replay.size(), cudaMemcpyDeviceToHost));
                    Require(replay == eager, "graph output differs from eager");
                }
                Data host;
                host.CopyFrom(output);
                ToDataType(host, DataType::FLOAT32);
                host.ToDevice(DataDevice::CPU);
                const float *actual = (const float *)host.cpuData;
                double error = 0, norm = 0, maxRelative = 0;
                for (int row = 0; row < m; ++row) {
                    for (int column = 0; column < n; ++column) {
                        const float value = actual[(size_t)row * n + column];
                        Require(std::isfinite(value), "nonfinite or unwritten output");
                        // Exhaust small matrices; sample large production shapes.
                        if (n * k > 256 * 512 && (row % 7 || column % 31)) continue;
                        double sum = 0;
                        for (int i = 0; i < k; ++i) {
                            const float scaledWeight = Bf16(DecodeE4M3(bytes[(size_t)column * k + i]) *
                                weight.scales[(column / 32) * (k / 32) + i / 32]);
                            sum += double(inputValues[(size_t)row * k + i]) * scaledWeight;
                        }
                        const double expected = Bf16(sum), delta = value - expected;
                        error += delta * delta;
                        norm += expected * expected;
                        maxRelative = std::max(maxRelative, std::abs(delta) / (std::abs(expected) + .01));
                    }
                }
                const double relative = std::sqrt(error / std::max(norm, 1e-30));
                std::cout << "M=" << m << " N=" << n << " K=" << k << " relative=" << relative
                          << " max_relative=" << maxRelative << std::endl;
                Require(relative < .004 && maxRelative < .02, "FP64 oracle mismatch");
                Cuda(cudaGraphExecDestroy(executable));
                Cuda(cudaGraphDestroy(graph));
                ++cases;
            }
            std::vector<unsigned char> after(bytes.size());
            Cuda(cudaMemcpy(after.data(), weight.cudaData, after.size(), cudaMemcpyDeviceToHost));
            Require(after == bytes, "source weight changed");
        }
        std::cout << "PASS " << cases << " eager + graph shapes" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
