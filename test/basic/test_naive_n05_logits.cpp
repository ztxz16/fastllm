#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "fastllm.h"
#include "models/speculative_sampling.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
static void Cuda(cudaError_t err) { Check(err == cudaSuccess, cudaGetErrorString(err)); }
static int CheckSamplingBoundaries() {
    Check(!FastllmNaiveCanSelectLogits(1, 0, 65, false), "Oversized top-k accepted");
    Check(!FastllmNaiveCanSelectLogits(kNaiveLogitsMaxShard + 1, 0, 4, false), "Oversized shard accepted");
    Check(!FastllmNaiveCanSelectLogits(1, kNaiveLogitsMaxVocab, 1, true), "Inexact token ID accepted");
    Check(FastllmNaiveCanSelectLogits(kNaiveLogitsMaxShard + 1, 0, 1, true), "Greedy shard unnecessarily limited");
    int checks = 4;
    for (int vocab : {1, 33, 1025, kNaiveLogitsMaxShard}) {
        const int offset = kNaiveLogitsMaxVocab - vocab, count = 64;
        std::vector<float> values(vocab);
        std::vector<std::pair<float, int>> expected;
        for (int i = 0; i < vocab; ++i) {
            // Ties, signed zero, a partial tile and the highest exact token ID.
            values[i] = i % 4 == 0 ? -0.0f : i % 4 == 1 ? 0.0f : float(i % 13 - 6);
            expected.emplace_back(-values[i], offset + i);
        }
        std::sort(expected.begin(), expected.end());
        Data input(FLOAT32, {1, vocab}, values), partial, output;
        input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        FastllmCudaNaiveLogitsSelect(input, offset, count, false, 1.0f, partial, output);
        output.ToDevice(DataDevice::CPU);
        const float *pairs = (float *)output.cpuData;
        for (int k = 0; k < count; ++k) {
            if (k < vocab) {
                Check(pairs[k * 2] == expected[k].second && pairs[k * 2 + 1] == -expected[k].first,
                      "Boundary TopK candidate mismatch");
            } else Check(pairs[k * 2] == -1, "Missing candidate was not padded");
            ++checks;
        }
    }
    return checks;
}
static int CheckSampling() {
    int checks = 0;
    for (int vocab : {257, 2053, 19072, 152576})
    for (int rows : {1, 8})
    for (int shards : {1, 8})
    for (int count : {1, 4, 50, 64}) {
        float scale = count == 4 ? 1.0f / .7f : 1.0f;
        std::vector<float> values(rows * vocab);
        for (int i = 0; i < rows * vocab; ++i)
            values[i] = count == 50 ? float(i % 7) : std::sin(i * .173f);
        std::vector<std::vector<std::pair<float,int>>> choices(rows);
        for (int rank = 0; rank < shards; ++rank) {
            int begin = (int64_t)vocab * rank / shards, end = (int64_t)vocab * (rank + 1) / shards;
            std::vector<float> part;
            for (int row = 0; row < rows; ++row)
                part.insert(part.end(), values.begin() + row * vocab + begin, values.begin() + row * vocab + end);
            Data input(FLOAT32, {rows, end - begin}, part), scratch, top;
            input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            FastllmCudaNaiveLogitsSelect(input, begin, count, false, scale, scratch, top);
            std::vector<float> result(rows * count * 2), replay(result.size());
            FastllmCudaCopyFromDeviceToHost(result.data(), top.cudaData, top.GetBytes());
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            FastllmCudaNaiveLogitsSelect(input, begin, count, false, scale, scratch, top);
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiateWithFlags(&exec, graph, 0));
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            FastllmCudaCopyFromDeviceToHost(replay.data(), top.cudaData, top.GetBytes());
            Check(result == replay, "Sampling TopK graph mismatch");
            for (int row = 0; row < rows; ++row)
                for (int k = 0; k < count; ++k) {
                    const float *p = result.data() + (row * count + k) * 2;
                    if (p[0] >= 0) choices[row].emplace_back(-p[1], (int)p[0]);
                }
            std::fill(part.begin(), part.end(), 0);
            for (int row = 0; row < rows; ++row) part[(row + 1) * (end - begin) - 1] = 7;
            FastllmCudaCopyFromHostToDevice(input.cudaData, part.data(), input.GetBytes());
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            FastllmCudaCopyFromDeviceToHost(replay.data(), top.cudaData, top.GetBytes());
            for (int row = 0; row < rows; ++row) Check(replay[row * count * 2] == end - 1, "Sampling graph stale logits");
            Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph)); ++checks;
        }
        for (int row = 0; row < rows; ++row) {
            auto &items = choices[row]; std::sort(items.begin(), items.end());
            std::vector<std::pair<float,int>> expected;
            for (int i = 0; i < vocab; ++i) expected.emplace_back(-values[row * vocab + i] * scale, i);
            std::partial_sort(expected.begin(), expected.begin() + count, expected.end());
            Data pairs(FLOAT32, {1, count * 2}); pairs.Allocate();
            for (int k = 0; k < count; ++k) {
                Check(items[k] == expected[k], "Sampling sharded TopK differs from CPU");
                ((float *)pairs.cpuData)[k * 2] = items[k].second;
                ((float *)pairs.cpuData)[k * 2 + 1] = -items[k].first;
            }
            GenerationConfig cfg; cfg.top_k = count; cfg.temperature = 1 / scale; cfg.top_p = .87f;
            Data full(FLOAT32, {1, vocab}, std::vector<float>(values.begin() + row * vocab, values.begin() + (row + 1) * vocab));
            for (int seed = 0; seed < 4; ++seed) {
                srand(seed); int a = LLMSampling(full, 0, cfg, LastTokensUnit());
                srand(seed); int b = LLMSamplingOnly(pairs, 0, cfg);
                Check(a == b, "Candidate sampling RNG/token mismatch"); ++checks;
            }
            if (scale == 1) {
                auto a = SpeculativeDistribution((float *)full.cpuData, vocab, cfg, LastTokensUnit());
                auto b = SpeculativeTopKDistribution((float *)pairs.cpuData, count, vocab, cfg);
                Check(a == b, "Speculative probability mismatch");
                for (double u : {0.0, .1, .49, .99, .999999})
                    Check(SampleSpeculativeDistribution(a, u) == SampleSpeculativeDistribution(b, u), "Speculative draw mismatch");
                ++checks;
            }
        }
    }
    return checks;
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        int checks = CheckSamplingBoundaries() + CheckSampling();
        for (int vocab : {1, 257, 2053, 19072, 152576})
        for (int rows : {1, 8})
        for (int shards : {1, 3, 8}) {
            if (shards > vocab) continue;
            for (int pattern = 0; pattern < 7; ++pattern) {
                std::vector<float> values(rows * vocab);
                for (int row = 0; row < rows; ++row)
                    for (int i = 0; i < vocab; ++i) {
                        float v = std::sin((i + row * 71) * .173f);
                        if (pattern == 1) v = 1;
                        if (pattern == 2) v = std::numeric_limits<float>::quiet_NaN();
                        if (pattern == 3) v = -std::numeric_limits<float>::infinity();
                        if (pattern == 4) v = i % 17 ? 0 : std::numeric_limits<float>::infinity();
                        if (pattern == 5) v = (i == 1 || i == 128 || i == 256) ? 9 : -9;
                        if (pattern == 6) v = i == vocab - 1 ? 9 : -std::numeric_limits<float>::infinity();
                        values[row * vocab + i] = v;
                    }
                Data full(FLOAT32, {rows, vocab}, values), reference;
                full.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                TopK(full, reference, 1); reference.ToDevice(DataDevice::CPU);
                std::vector<int> bestIds(rows, 0);
                std::vector<float> best(rows, -std::numeric_limits<float>::infinity());
                for (int rank = 0; rank < shards; ++rank) {
                    int begin = (int64_t)vocab * rank / shards, end = (int64_t)vocab * (rank + 1) / shards;
                    std::vector<float> part;
                    for (int row = 0; row < rows; ++row)
                        part.insert(part.end(), values.begin() + row * vocab + begin, values.begin() + row * vocab + end);
                    Data input(FLOAT32, {rows, end - begin}, part), partial, top;
                    input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                    FastllmCudaNaiveLogitsSelect(input, begin, 1, true, 1.0f, partial, top);
                    std::vector<float> eager(rows * 2), replay(rows * 2);
                    FastllmCudaCopyFromDeviceToHost(eager.data(), top.cudaData, top.GetBytes());
                    cudaGraph_t graph; cudaGraphExec_t exec;
                    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    FastllmCudaNaiveLogitsSelect(input, begin, 1, true, 1.0f, partial, top);
                    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    Cuda(cudaGraphInstantiateWithFlags(&exec, graph, 0));
                    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                    FastllmCudaCopyFromDeviceToHost(replay.data(), top.cudaData, top.GetBytes());
                    Check(eager == replay, "Top1 graph differs from eager");
                    for (int row = 0; row < rows; ++row) {
                        int id = (int)eager[row * 2]; float score = eager[row * 2 + 1];
                        if (id >= 0 && FastllmNaiveTop1Better(score, id, best[row], bestIds[row])) {
                            best[row] = score; bestIds[row] = id;
                        }
                    }
                    // Replaying must read new logits, with stable buffer addresses.
                    std::fill(part.begin(), part.end(), -3);
                    for (int row = 0; row < rows; ++row) part[(row + 1) * (end - begin) - 1] = 7;
                    FastllmCudaCopyFromHostToDevice(input.cudaData, part.data(), input.GetBytes());
                    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                    FastllmCudaCopyFromDeviceToHost(replay.data(), top.cudaData, top.GetBytes());
                    for (int row = 0; row < rows; ++row)
                        Check(replay[row * 2] == end - 1 && replay[row * 2 + 1] == 7, "Top1 graph has stale logits");
                    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
                    checks += 2;
                }
                for (int row = 0; row < rows; ++row) {
                    if (best[row] != ((float *)reference.cpuData)[row * 2 + 1])
                        std::cerr << "vocab=" << vocab << " rows=" << rows << " shards=" << shards
                                  << " pattern=" << pattern << " row=" << row << " got=" << best[row]
                                  << " expected=" << ((float *)reference.cpuData)[row * 2 + 1] << '\n';
                    Check(bestIds[row] == (int)((float *)reference.cpuData)[row * 2], "Sharded Top1 token differs from full Top1");
                    Check(best[row] == ((float *)reference.cpuData)[row * 2 + 1], "Sharded Top1 score differs from full Top1");
                    ++checks;
                }
            }
        }
        std::cout << "NAIVE LOGITS SELECTION PASS " << checks << " checks\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
