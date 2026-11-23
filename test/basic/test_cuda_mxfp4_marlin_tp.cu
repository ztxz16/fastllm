// Exercise in-place TP packing, two layers sharing a rank workspace, and graph replay with new routes.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-mxfp4-marlin-tp.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>

using namespace fastllm;

static void Check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
static void Cuda(cudaError_t rc) { Check(rc == cudaSuccess, cudaGetErrorString(rc)); }

int main() {
#ifdef CUDA_NO_TENSOR_CORE
    return 77;
#endif
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    FastllmCudaSetDevice(0);
    if (FastllmCudaRuntimeArch() < 80)
        return 77;
    constexpr int h = 512, inter = 128, experts = 16, k = 8;
    std::mt19937 rng(8913);
    std::vector<std::unique_ptr<Data>> storage;
    std::vector<std::vector<Data *>> layers(3), reference(3);
    for (int layer : {0, 2}) {
        layers[layer].resize(2 + 2 * experts);
        reference[layer].resize(2 + 2 * experts);
        for (int slot = 2; slot < 2 + 2 * experts; ++slot) {
            const int n = slot % 2 ? h : 2 * inter, width = slot % 2 ? inter : h;
            auto make = [&]() {
                auto w = std::make_unique<Data>(NVFP4, std::vector<int>{n, width});
                w->blockK = 1;
                w->blockM = 32;
                w->isModelWeight = true;
                w->directMemory = layer == 0;
                w->Allocate(false);
                return w;
            };
            auto w = make(), ref = make();
            for (size_t j = 0; j < GetNVFP4WeightBytes(n, width); ++j)
                w->cpuData[j] = rng() >> 24;
            auto scales = GetNVFP4ScaleData(*w);
            for (size_t j = 0; j < GetNVFP4ScaleBytes(n, width, 1, 32); ++j)
                scales[j] = 119 + rng() % 4;
            std::memcpy(ref->cpuData, w->cpuData, w->GetBytes());
            for (auto *v : {w.get(), ref.get()})
                v->ToDevice(CUDA, {0}, true);
            layers[layer][slot] = w.get();
            reference[layer][slot] = ref.get();
            storage.push_back(std::move(w));
            storage.push_back(std::move(ref));
        }
    }
    void *original = layers[0][2]->cudaData;
    auto invalid = layers;
    invalid[2][3] = nullptr;
    Check(!PrepareMxfp4MarlinRank(0, invalid, h, experts, k), "accepted missing down weight");
    Check(layers[0][2]->cudaData == original, "rejected layout consumed weights");
    auto rank = PrepareMxfp4MarlinRank(0, layers, h, experts, k);
    Check(bool(rank), "valid shards rejected");
    PackMxfp4MarlinRank(*rank, layers);
    for (const auto &ws : layers)
        for (size_t i = 2; i < ws.size(); ++i)
            Check(!ws[i]->cudaData, "original compact layout still exposed");

    for (int rows : {1, 8, 19}) {
        std::vector<float> x(rows * h), sc(rows * k);
        for (auto &v : x)
            v = float(int(rng() % 201) - 100) / 100.f;
        for (int i = 0; i < rows * k; ++i)
            sc[i] = (1 + i % k) / 36.f;
        Data input(FLOAT32, {rows, h}, x), ids(INT32, {rows, k}), scores(FLOAT32, {rows, k}, sc);
        ToDataType(input, BFLOAT16);
        ids.Allocate();
        std::memset(ids.cpuData, 0, ids.GetBytes());
        for (auto *v : {&input, &ids, &scores})
            v->ToDevice(CUDA, {0}, true);
        Data act, result[2], refAct, expected;
        auto forward = [&]() {
            for (int i = 0; i < 2; ++i)
                Check(RunMxfp4MarlinTP(*rank, 2 * i, input, ids, scores, act, result[i]), "Marlin TP failed");
        };
        forward();
        Cuda(cudaDeviceSynchronize());
        auto stream = cudaStreamPerThread;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        forward();
        Cuda(cudaStreamEndCapture(stream, &graph));
        Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<int> idsHost(rows * k);
            for (int j = 0; j < rows * k; ++j)
                idsHost[j] = (j % k + pass * 5 + (j / k) * (pass + 1)) % experts;
            Cuda(cudaMemcpy(ids.cudaData, idsHost.data(), idsHost.size() * sizeof(int), cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec, stream));
            for (int i = 0; i < 2; ++i) {
                auto &ws = reference[2 * i];
                Check(FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed(
                          input, refAct, expected, ws.data(), ws.size(), (int32_t *)ids.cudaData,
                          (float *)scores.cudaData, rows, k, h, inter),
                      "compact reference failed");
                Cuda(cudaStreamSynchronize(stream));
                std::vector<__nv_bfloat16> y(rows * h), ref(rows * h);
                Cuda(cudaMemcpy(y.data(), result[i].cudaData, y.size() * 2, cudaMemcpyDeviceToHost));
                Cuda(cudaMemcpy(ref.data(), expected.cudaData, ref.size() * 2, cudaMemcpyDeviceToHost));
                double error = 0, norm = 0;
                for (size_t j = 0; j < y.size(); ++j) {
                    const double v = __bfloat162float(ref[j]), delta = __bfloat162float(y[j]) - v;
                    error += delta * delta;
                    norm += v * v;
                }
                const double l2 = std::sqrt(error / norm);
                std::printf("rows=%d layer=%d pass=%d relative_l2=%.7f\n", rows, 2 * i, pass, l2);
                Check(norm > 0 && l2 < .01, "Marlin TP and compact reference differ");
            }
        }
        Cuda(cudaGraphExecDestroy(exec));
        Cuda(cudaGraphDestroy(graph));
    }
    return 0;
}
