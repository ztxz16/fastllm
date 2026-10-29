// Verify the fused EP row reduction and shared join against the original
// kernels and an independent BF16-rounding oracle, including graph dependencies.
#include "fastllm.h"
#include "utils.h"
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;
static int checks = 0;
static void Check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
    ++checks;
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static std::vector<float> Read(const Data &d) {
    Check(d.dataType == FLOAT32, "MoE sum was rounded before shared expert addition");
    std::vector<float> v(d.Count(0));
    Cuda(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}
struct Fixture {
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights;
    Fixture(int hidden, int inter, int device, int experts) : weights(2 * (experts + 1), nullptr) {
        DeepSeekV41Model loader;
        loader.block_cnt = 1;
        loader.moeDeviceMap = {{"cuda:" + std::to_string(device), 1}};
        std::mt19937 rng(628);
        for (int expert = 0; expert < experts; ++expert)
            for (int part = 0; part < 2; ++part) {
                const int n = part ? hidden : 2 * inter, k = part ? inter : hidden;
                const std::string name =
                    "layers.0.ffn.experts." + std::to_string(expert) + (part ? ".w2.weight" : ".gateup.weight");
                auto w = std::make_unique<Data>(NVFP4, std::vector<int>{n, k});
                w->blockK = 1;
                w->blockM = 32;
                w->Allocate(false);
                for (size_t i = 0; i < GetNVFP4WeightBytes(n, k); ++i)
                    w->cpuData[i] = rng() >> 24;
                auto *s = GetNVFP4ScaleData(*w);
                for (size_t i = 0; i < GetNVFP4ScaleBytes(n, k, 1, 32); ++i)
                    s[i] = 119 + (rng() % 4);
                w->name = name;
                w->isModelWeight = true;
                loader.AddSpecialWeight(name, "linear", 0);
                Check(loader.MoveSpecialWeightToCudaIfNeeded(name, *w), "loader did not place PP weight");
                Check(w->directMemory, "PP loader retained source in weight slab");
                weights[2 * (expert + 1) + part] = w.get();
                owned.push_back(std::move(w));
            }
    }
};

static uint16_t Bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float Float(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
static std::vector<uint16_t> Bits(const Data &d) {
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<uint16_t> v(d.Count(0));
    Cuda(cudaMemcpy(v.data(), d.cudaData, v.size() * 2, cudaMemcpyDeviceToHost));
    return v;
}
static void EpJoin(int hidden, int inter, bool full, int topk = 6) {
    FastllmCudaSetDevice(0);
    Fixture f(hidden, inter, 0, 48);
    const std::vector<int> sizes = full ? std::vector<int>{2, 6, 8} : std::vector<int>{2, 3, 4, 5, 6, 7, 8};
    const std::vector<int> ownersList = full ? std::vector<int>{8} : std::vector<int>{1, 2, 8};
    void *slow;
    Cuda(cudaMalloc(&slow, 64 * 1024 * 1024));
    for (int owners : ownersList)
        for (int rows : sizes)
            for (int pattern = 0; pattern < 4; pattern++) {
                const int owner = owners - 1;
                std::mt19937 rng(919 + rows * 17 + pattern);
                std::vector<float> values(rows * hidden), scores(rows * topk), sharedValues(rows * hidden);
                std::vector<int32_t> indices(rows * topk);
                for (float &v : values)
                    v = float(int(rng() % 65521) - 32760) / 16384.f;
                for (float &v : sharedValues)
                    v = float(int(rng() % 65521) - 32760) / 32768.f;
                for (int i = 0; i < rows * topk; i++) {
                    int expert = pattern == 3 ? i % topk : (i * 13 + i / topk * 17) % 48;
                    int rank = pattern == 0 ? int(rng() % owners) : pattern == 2 ? (owner + 1) % owners : owner;
                    indices[i] = (pattern == 2 && owners == 1) ? -1 : expert * owners + rank;
                    scores[i] = i % 7 == 0 ? 0.f : float(int(rng() % 31) - 3) / 32;
                }
                Data input(FLOAT32, {rows, hidden}, values), score(FLOAT32, {rows, topk}, scores);
                Data shared(FLOAT32, {rows, hidden}, sharedValues), sharedSource(FLOAT32, {rows, hidden}, sharedValues);
                Data ids(INT32, {rows, topk});
                ids.Allocate();
                std::memcpy(ids.cpuData, indices.data(), indices.size() * 4);
                ToDataType(input, BFLOAT16);
                for (Data *d : {&input, &score, &shared, &sharedSource, &ids})
                    d->ToDevice(CUDA, {0}, true);
                Data joined(BFLOAT16, {rows, hidden}), expected(BFLOAT16, {rows, hidden});
                joined.ToDevice(CUDA, {0});
                joined.Allocate();
                expected.ToDevice(CUDA, {0});
                expected.Allocate();
                Data scratch, routed;
                auto old = [&] {
                    Check(FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
                              input, scratch, routed, f.weights.data(), f.weights.size(), (int32_t *)ids.cudaData,
                              (float *)score.cudaData, topk, owner, owners, true, 10.f),
                          "old routed");
                    Check(FastllmCudaDeepSeekV41MoeJoinBFloat16((float *)routed.cudaData, (float *)shared.cudaData,
                                                                expected.cudaData, rows * hidden),
                          "old shared join");
                };
                auto fused = [&](int layer) {
                    Check(FastllmCudaV41MarlinExpertParallelShared(
                              input, scratch, routed, f.weights.data(), f.weights.size(), (int32_t *)ids.cudaData,
                              (float *)score.cudaData, topk, owner, owners, 10.f, nullptr, shared, joined, layer),
                          "fused routed join");
                };
                old();
                fused(-1);
                Check(Bits(joined) == Bits(expected), "eager bits differ");
                // Delay shared on a distinct captured branch to exercise the moved join
                // dependency, including cases where the shared producer finishes last.
                Check(FastllmCudaGraphPrepareCaptureDevice(), "capture preparation");
                FastllmCudaGraphSetParallelMarkersEnabled(true);
                Check(FastllmCudaGraphBeginCapture(), "capture begin");
                FastllmCudaGraphMarkParallelFork(0);
                Cuda(cudaMemsetAsync(slow, 0, 64 * 1024 * 1024, cudaStreamPerThread));
                Cuda(cudaMemcpyAsync(shared.cudaData, sharedSource.cudaData, shared.GetBytes(),
                                     cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                FastllmCudaGraphMarkParallelFirstDone(0);
                FastllmCudaGraphMarkParallelSecondBegin(0);
                fused(0);
                void *graph = nullptr, *exec = nullptr;
                Check(FastllmCudaGraphEndCapture(&graph), "capture end");
                Check(FastllmCudaGraphOptimizeParallelRegions(graph) == 1, "parallel join graph not rewritten");
                Check(FastllmCudaGraphInstantiate(graph, &exec), "graph instantiate");
                for (int replay = 0; replay < 3; replay++) {
                    std::rotate(indices.begin(), indices.begin() + topk % indices.size(), indices.end());
                    for (float &v : sharedValues)
                        v = -v;
                    Cuda(cudaMemcpyAsync(ids.cudaData, indices.data(), indices.size() * 4, cudaMemcpyHostToDevice,
                                         cudaStreamPerThread));
                    Cuda(cudaMemcpyAsync(sharedSource.cudaData, sharedValues.data(), sharedValues.size() * 4,
                                         cudaMemcpyHostToDevice, cudaStreamPerThread));
                    Check(FastllmCudaGraphLaunch(exec), "replay");
                    auto actual = Bits(joined);
                    old();
                    Check(actual == Bits(expected), "parallel graph join changed bits");
                    auto raw = Read(routed);
                    for (size_t i = 0; i < actual.size(); i++)
                        Check(actual[i] == Bf16(Float(Bf16(raw[i])) + Float(Bf16(sharedValues[i]))),
                              "host BF16 boundary oracle");
                }
                FastllmCudaGraphExecDestroy(exec);
                FastllmCudaGraphDestroy(graph);
                auto dtype = input.dataType;
                input.dataType = FLOAT16;
                Check(!FastllmCudaV41MarlinExpertParallelShared(
                          input, scratch, routed, f.weights.data(), f.weights.size(), (int32_t *)ids.cudaData,
                          (float *)score.cudaData, topk, owner, owners, 10.f, nullptr, shared, joined, -1),
                      "unsupported dtype accepted");
                input.dataType = dtype;
                std::cout << "PASS join H=" << hidden << " M=" << rows << " owners=" << owners << " pattern=" << pattern
                          << std::endl;
            }
    Cuda(cudaFree(slow));
}
int main() {
    try {
        SetThreads(4);
        for (int topk : {1, 3, 6, 16})
            EpJoin(512, 256, false, topk);
        EpJoin(5120, 2304, true);
        std::cout << "PASS all shared join cases checks=" << checks << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
