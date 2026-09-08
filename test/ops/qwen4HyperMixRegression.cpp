#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace fastllm;

static void CheckCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static void ToGpu(Data &data) {
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    data.Allocate(false);
}

static void CheckEqual(const Data &expected, const Data &actual, int pattern) {
    std::vector<float> a(expected.Count(0)), b(a.size());
    CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
    CheckCuda(cudaMemcpy(a.data(), expected.cudaData, a.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(b.data(), actual.cudaData, b.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            std::cerr << "pattern=" << pattern << " channel=" << i
                      << " expected=" << std::hexfloat << a[i]
                      << " actual=" << b[i] << '\n';
            throw std::runtime_error("HC up/mix differs from the two CUDA operators");
        }
    }
}

static int CheckDispatch(Executor *executor) {
    struct Shape { int rows, groups, channels, rank; DataType type, weightType; };
    const Shape shapes[] = {
        {1, 4, 2560, 320, FLOAT32, FLOAT16},
        {8, 4, 2560, 320, FLOAT32, FLOAT16},
        {1, 4, 2560, 320, FLOAT16, FLOAT16},
        {1, 4, 2560, 320, BFLOAT16, FLOAT16},
        {1, 4, 2560, 320, FLOAT32, FLOAT32},
        {1, 4, 2560, 316, FLOAT32, FLOAT16},
        {1, 4, 2048, 320, FLOAT32, FLOAT16},
        {1, 2, 5120, 320, FLOAT32, FLOAT16},
        {2, 4, 2560, 320, FLOAT32, FLOAT16},
        {4, 4, 2560, 320, FLOAT32, FLOAT16}
    };
    int checks = 0;
    for (const Shape &s : shapes) {
        Data input(s.type, {s.rows, 1, s.groups * s.channels});
        Data lowRank(s.type, {s.rows, 1, s.rank});
        Data weight(s.weightType, {s.groups * s.channels, s.rank}), output;
        bool accepted = executor->CanRunOnFirstDevice("Qwen4HyperMix",
            {{"input", &input}, {"mixLogits", &lowRank},
             {"weight", &weight}, {"output", &output}}, {}, {{"groups", s.groups}});
#ifdef CUDA_NO_TENSOR_CORE
        const bool expected = false;
#else
        const bool expected = checks < 2;
#endif
        if (accepted != expected) throw std::runtime_error("unexpected projected HC dispatch");
        ++checks;
    }
    return checks;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        FastllmCudaSetDevice(0);
        auto *executor = static_cast<Executor *>(GetExecutor());
        executor->SetFirstDevice("cuda:0");
        int checks = CheckDispatch(executor);
#ifndef CUDA_NO_TENSOR_CORE
        constexpr int channels = 2560, groups = 4, rank = 320, width = channels * groups;
        std::mt19937 generator(817);
        std::uniform_real_distribution<float> random(-1.0f, 1.0f);
        std::vector<float> residual(width), activated(rank), weights(width * rank);
        Data normalized(FLOAT32, {1, 1, width}), lowRank(FLOAT32, {1, 1, rank});
        Data reference(FLOAT32, {1, 1, channels}), logits(FLOAT32, {1, 1, width});
        Data output(FLOAT32, {1, 1, channels});
        for (Data *data : {&normalized, &lowRank, &reference, &logits, &output}) ToGpu(*data);

        // Random, extreme, cancellation, signed-zero, subnormal and reduction-boundary inputs.
        for (int pattern = 0; pattern < 11; ++pattern) {
            for (float &v : weights) v = random(generator) * 0.25f;
            if (pattern == 3 || pattern == 7) {
                for (float &v : weights) v = 1.0f;
            }
            if (pattern == 6) {
                for (float &v : weights) v = std::ldexp(std::round(v * 4), -24);
            }
            Data weight(FLOAT16, {width, rank}, weights), emptyBias;
            ToGpu(weight);
            auto baseline = [&] {
                if (!FastllmCudaMatMulFloat16(lowRank, weight, emptyBias, logits, 1, rank, width) ||
                    !FastllmCudaQwen4HyperMix(normalized, logits, reference, groups)) {
                    throw std::runtime_error("reference HC launch rejected");
                }
            };
            auto fused = [&] {
                Qwen4HyperMixProjected(normalized, lowRank, weight, groups, output);
            };
            auto upload = [&](int replay) {
                for (int i = 0; i < width; ++i) {
                    residual[i] = random(generator);
                    if (pattern == 4) residual[i] = i % 2 ? -0.0f : 0.0f;
                    if (pattern == 7) residual[i] = i / channels == 0 ? 1e10f
                        : i / channels == 2 ? -1e10f : 1.0f;
                }
                for (int i = 0; i < rank; ++i) {
                    activated[i] = random(generator);
                    if (pattern == 1) activated[i] *= 1e-20f;
                    if (pattern == 2) activated[i] *= 1e10f;
                    if (pattern == 3) activated[i] = i % 4 == 0 ? 1e12f
                        : i % 4 == 2 ? -1e12f : (float)(replay + 1);
                    if (pattern == 4) activated[i] = i % 2 ? -0.0f : 0.0f;
                    if (pattern == 5) activated[i] = (i - 160) * std::numeric_limits<float>::denorm_min();
                    if (pattern == 7) activated[i] = 0.0f;
                    if (pattern >= 8 && pattern <= 10) activated[i] =
                        i == (pattern == 8 ? 127 + replay : pattern == 9 ? 255 + replay : 319 - replay)
                        ? (float)(replay + 1) : 0.0f;
                }
                CheckCuda(cudaMemcpy(normalized.cudaData, residual.data(), width * sizeof(float), cudaMemcpyHostToDevice));
                CheckCuda(cudaMemcpy(lowRank.cudaData, activated.data(), rank * sizeof(float), cudaMemcpyHostToDevice));
            };
            upload(0);
            baseline();
            fused();
            CheckEqual(reference, output, pattern);
            ++checks;

            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CheckCuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            if (!FastllmCudaGraphIsCapturing()) throw std::runtime_error("external capture was not latched");
            fused();
            CheckCuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            size_t nodes = 0;
            CheckCuda(cudaGraphGetNodes(graph, nullptr, &nodes));
            if (nodes != 1) throw std::runtime_error("fused HC graph must contain one kernel");
            CheckCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            void *address = output.cudaData;
            for (int replay = 0; replay < 3; ++replay) {
                upload(replay);
                baseline();
                CheckCuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                CheckEqual(reference, output, pattern);
                if (output.cudaData != address) throw std::runtime_error("HC replay changed output storage");
                ++checks;
            }
            CheckCuda(cudaGraphExecDestroy(executable));
            CheckCuda(cudaGraphDestroy(graph));
        }
#endif
        std::cout << "PASS: " << checks << " HC dispatch/bitwise/eager/graph checks\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
