// Generic BF16 x block-FP8 projection: independent FP64 oracle and tiled
// batch boundaries. The ordinary public operator is exercised without backend
// overrides, including eager/graph equality,
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
            {17, 64}, {33, 192}, {63, 64}, {129, 128}, {64, 128}, {128, 64}, {192, 256}, {256, 192}, {512, 5120},
            {1280, 5120}, {1792, 5120}, {4096, 1280}, {5120, 1024},
            {4608, 5120}, {4609, 5120}, {5120, 8192}, {32768, 1280}, {4097, 1280}, {5120, 2304}, {64, 512}, {192, 512}, {576, 5120}};
        // Both sides of the split-K dispatch boundary, plus a non-128 N tile.
        const int thinColumns = (properties.multiProcessorCount / 4) * 128;
        if (thinColumns > 0) {
            shapes.emplace_back(thinColumns, 512);
            shapes.emplace_back(thinColumns + 128, 512);
        }
        // Small MMA dispatch needs two output CTAs per SM; exercise both
        // sides, including an incomplete sixteen-channel tile.
        const int smallBoundary = (2 * properties.multiProcessorCount - 1) * 16;
        shapes.emplace_back(smallBoundary, 5120);
        shapes.emplace_back(smallBoundary + 1, 5120);
        for (const auto &shape : shapes) for (const auto &blocks :
                std::vector<std::pair<int,int>>{{32,32},{16,64},{64,16},{48,64},{128,128}}) {
            if (blocks.first != 32 && shape != std::make_pair(129,128) &&
                shape != std::make_pair(1280,5120) && shape != std::make_pair(4609,5120)) continue;
            const int n = shape.first, k = shape.second;
            Data weight(DataType::FP8_E4M3, {n, k});
            weight.blockM = blocks.first; weight.blockK = blocks.second;
            const int scaleColumns = (k + weight.blockM - 1) / weight.blockM;
            weight.Allocate();
            std::vector<unsigned char> bytes((size_t)n * k);
            for (auto &value : bytes) value = (random() % 2 ? 128 : 0) | (random() % 127);
            std::memcpy(weight.cpuData, bytes.data(), bytes.size());
            weight.scales.resize(size_t((n + weight.blockK - 1) / weight.blockK) * scaleColumns);
            for (auto &value : weight.scales) value = std::ldexp(1.f, int(random() % 17) - 17);
            weight.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            std::vector<float> biasValues(n, 0.f);
            Data bias;
            if (n == 17 || n == 4609) {
                for (auto &v : biasValues) v = Bf16((int(random() % 201)-100)*.001f);
                bias.dataType=FLOAT32;bias.Resize({n});bias.Allocate();
                std::memcpy(bias.cpuData,biasValues.data(),n*sizeof(float));
                bias.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            }


            std::vector<int> batches;
            for (int rows = 1; rows <= 35; ++rows) batches.push_back(rows);
            batches.insert(batches.end(), {63, 64, 65, 129});
            for (int m : batches) {
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
                    Require(FastllmCudaBFloat16MatMulFP8E4M3(
                        input, weight, bias, output, m, k, n), "FP8 linear dispatch failed");
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
                for (int repeat = 0; repeat < 3; ++repeat) {
                    if (repeat) {
                        for (auto &value : inputValues)
                            value = repeat == 1 ? 0.f : Bf16((int(random() % 2001) - 1000) * .0003f);
                        Data changed(FLOAT32, {m,k}, inputValues);
                        ToDataType(changed, BFLOAT16);
                        Cuda(cudaMemcpy(input.cudaData, changed.cpuData, input.GetBytes(), cudaMemcpyHostToDevice));
                        launch();
                        Cuda(cudaMemcpy(eager.data(), output.cudaData, eager.size(), cudaMemcpyDeviceToHost));
                    }
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
                        if (n * k > 256 * 512 && ((row % 7 && row != m - 1) || column % 31)) continue;
                        double sum = 0;
                        for (int i = 0; i < k; ++i) {
                            const float scaledWeight = Bf16(DecodeE4M3(bytes[(size_t)column * k + i]) *
                                weight.scales[(column / weight.blockK) * scaleColumns + i / weight.blockM]);
                            sum += double(inputValues[(size_t)row * k + i]) * scaledWeight;
                        }
                        const double expected = Bf16(sum + biasValues[column]), delta = value - expected;
                        error += delta * delta;
                        norm += expected * expected;
                        maxRelative = std::max(maxRelative, std::abs(delta) / (std::abs(expected) + .01));
                    }
                }
                const double relative = std::sqrt(error / std::max(norm, 1e-30));
                std::cout << "M=" << m << " N=" << n << " K=" << k << " blockM=" << weight.blockM << " blockK=" << weight.blockK << " relative=" << relative
                          << " max_relative=" << maxRelative << std::endl;
                Require(relative < .004 && (m >= 32 || maxRelative < .02), "FP64 oracle mismatch");
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
