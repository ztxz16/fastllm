#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Alloc(Data &d, DataType t, std::vector<int> dims) {
    d.dataType = t;
    d.UpdateUnitSize();
    d.Resize(dims);
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {FastllmCudaGetDevice()};
    d.Allocate();
}
template <class T> static std::vector<T> Read(Data &d) {
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<T> v(d.Count(0));
    Cuda(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void Validate(Data &logits, Data &ids, Data &scores, const std::vector<__nv_bfloat16> &x,
                     const std::vector<float> &w, const std::vector<float> &bias, int m, int k, float temp,
                     float scale, int experts = 384, int topk = 6) {
    auto got = Read<float>(logits), prob = Read<float>(scores);
    auto indices = Read<int>(ids);
    for (int row = 0; row < m; ++row) {
        std::vector<double> ref(experts);
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        for (int e = 0; e < experts; ++e) {
            double v = 0;
            for (int j = 0; j < k; ++j)
                v += double(__bfloat162float(x[row * k + j])) * w[e * k + j];
            v /= temp;
            ref[e] = sqrt(v > 20 ? v : log1p(exp(v)));
            Check(std::isfinite(got[row * experts + e]), "nonfinite score");
            Check(std::abs(got[row * experts + e] - ref[e]) < 5e-4, "score absolute error");
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return ref[a] + bias[a] > ref[b] + bias[b]; });
        double sum = 0;
        for (int t = 0; t < topk; ++t)
            sum += ref[order[t]];
        for (int t = 0; t < topk; ++t) {
            Check(indices[row * topk + t] == order[t], "TopK mismatch");
            Check(std::abs(prob[row * topk + t] - ref[order[t]] / sum * scale) < 2e-6,
                  "normalized weight error");
        }
    }
}
int main() {
    try {
        SetThreads(8);
        FastllmCudaSetDevice(0);
        std::mt19937 rng(20261024);
        std::normal_distribution<float> dist(0, .1f);
        int cases = 0;
        for (int experts : {384, 128}) {
            const int topk = experts == 384 ? 6 : 3;
        for (int k : {128, 512, 4096, 5120, 6144})
            for (int m : {5, 6, 7, 8}) {
                std::vector<__nv_bfloat16> x(m * k);
                std::vector<float> w(experts * k), bias(experts);
                for (auto &v : x)
                    v = __float2bfloat16(dist(rng));
                for (auto &v : w)
                    v = __bfloat162float(__float2bfloat16(dist(rng)));
                for (auto &v : bias)
                    v = dist(rng) * .1f;
                Data input, weight, biasData, logits, ids, scores;
                Alloc(input, BFLOAT16, {m, k});
                Alloc(weight, FLOAT32, {experts, k});
                Alloc(biasData, FLOAT32, {experts});
                Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(weight.cudaData, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(biasData.cudaData, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
                for (float temp : {.37f, 1.f, 2.5f}) {
                    Check(
                        FastllmCudaV41RouterSelectTopK(input, weight, biasData, temp, 2.5f, logits, ids, scores, topk),
                        "eligible path rejected");
                    Validate(logits, ids, scores, x, w, bias, m, k, temp, 2.5f, experts, topk);
                    ++cases;
                }
                // Same graph consumes new inputs; poison every output before replay.
                Check(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
                void *graph = nullptr, *exec = nullptr;
                Check(FastllmCudaGraphBeginCapture(), "begin capture");
                Check(FastllmCudaV41RouterSelectTopK(input, weight, biasData, 1, 1, logits, ids, scores, topk),
                      "capture router");
                Check(FastllmCudaGraphEndCapture(&graph), "end capture");
                Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
                for (int rep = 0; rep < 3; ++rep) {
                    for (auto &v : x)
                        v = __float2bfloat16(dist(rng));
                    Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
                    Cuda(cudaMemset(logits.cudaData, 255, logits.GetBytes()));
                    Cuda(cudaMemset(ids.cudaData, 255, ids.GetBytes()));
                    Cuda(cudaMemset(scores.cudaData, 255, scores.GetBytes()));
                    Check(FastllmCudaGraphLaunch(exec), "graph replay");
                    Validate(logits, ids, scores, x, w, bias, m, k, 1, 1, experts, topk);
                }
                FastllmCudaGraphExecDestroy(exec);
                FastllmCudaGraphDestroy(graph);
                Check(weight.dataType == FLOAT32 && Read<float>(weight) == w, "source weight mutated");
                // The exact-BF16 cache must be invalidated when its source is replaced.
                Data replacement;
                Alloc(replacement, FLOAT32, {experts, k});
                for (auto &v : w)
                    v = __bfloat162float(__float2bfloat16(dist(rng)));
                Cuda(cudaMemcpy(replacement.cudaData, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
                weight.CopyFrom(replacement);
                Check(FastllmCudaV41RouterSelectTopK(input, weight, biasData, 1, 1, logits, ids, scores, topk),
                      "cache refresh");
                Validate(logits, ids, scores, x, w, bias, m, k, 1, 1, experts, topk);
                w[0] = .1234567f;
                Cuda(cudaMemcpy(replacement.cudaData, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
                weight.CopyFrom(replacement);
                Check(!FastllmCudaV41RouterSelectTopK(input, weight, biasData, 1, 1, logits, ids, scores, topk),
                      "inexact FP32 must fall back");
                std::cout << "PASS E=" << experts << " M=" << m << " K=" << k << std::endl;
            }
        }
        // Exact ties follow vLLM's smallest expert ID, with finite normalized weights.
        for (int experts : {384, 128}) {
            const int topk = experts == 384 ? 6 : 3;
            Data input, weight, bias, logits, ids, scores;
            Alloc(input, BFLOAT16, {6, 5120});
            Alloc(weight, FLOAT32, {experts, 5120});
            Alloc(bias, FLOAT32, {experts});
            Cuda(cudaMemset(input.cudaData, 0, input.GetBytes()));
            Cuda(cudaMemset(weight.cudaData, 0, weight.GetBytes()));
            Cuda(cudaMemset(bias.cudaData, 0, bias.GetBytes()));
            Check(FastllmCudaV41RouterSelectTopK(input, weight, bias, 1, 1, logits, ids, scores, topk),
                  "ties rejected");
            auto index = Read<int>(ids);
            auto prob = Read<float>(scores);
            for (int i = 0; i < 6 * topk; ++i) {
                Check(index[i] == i % topk, "tie order");
                Check(std::abs(prob[i] - 1.f / topk) < 1e-6, "tie probability");
            }
            Check(!FastllmCudaV41RouterSelectTopK(input, weight, bias, 1, 1, input, ids, scores, topk),
                  "alias accepted");
            Check(!FastllmCudaV41RouterSelectTopK(input, weight, bias, 1, 1, logits, ids, scores, topk + 1), "unsupported topk accepted");
            input.dataDeviceIds = {0, 1, 2, 3, 4, 5, 6, 7};
            Check(FastllmCudaV41RouterSelectTopK(input, weight, bias, 1, 1, logits, ids, scores, topk), "root tensor with TP metadata rejected");
            input.dataDeviceIds = {1};
            Check(!FastllmCudaV41RouterSelectTopK(input, weight, bias, 1, 1, logits, ids, scores, topk),
                  "wrong input device accepted");
            input.dataDeviceIds = {0};
        }
        std::cout << "PASS " << cases
                  << " precision cases, generic/SM90, temperature/scale, graph replay, ties, unchanged "
                     "weights, cache invalidation and FP32 fallback"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
