#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
void Require(bool ok, const char *message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(2); }
}
void Check(cudaError_t error) {
    Require(error == cudaSuccess, cudaGetErrorString(error));
}
void Tensor(fastllm::Data &data, fastllm::DataType type, const std::vector<int> &dims) {
    data.dataType = type;
    data.UpdateUnitSize();
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Resize(dims);
    data.Allocate(false);
}
half Value(int i, int seed) {
    return __float2half_rn(((i * 17 + seed * 29) % 101 - 50) / 128.0f);
}
void Run(int capacity, int runtimeBlock) {
    constexpr int heads = 4, kvHeads = 2, rows = 8, width = 128;
    const size_t cacheElements = (size_t)kvHeads * capacity * width;
    fastllm::Data q, key, value, cacheK, cacheV, output, length;
    Tensor(q, fastllm::FLOAT16, {heads, rows, width});
    Tensor(key, fastllm::FLOAT16, {kvHeads, rows, width});
    Tensor(value, fastllm::FLOAT16, {kvHeads, rows, width});
    Tensor(cacheK, fastllm::FLOAT16, {kvHeads, capacity, width});
    Tensor(cacheV, fastllm::FLOAT16, {kvHeads, capacity, width});
    Tensor(output, fastllm::FLOAT16, {heads, rows, width});
    Tensor(length, fastllm::INT32, {1});
    std::vector<half> queries(q.Count(0)), keys(key.Count(0)), values(value.Count(0));
    std::vector<half> cacheKeys(cacheElements), cacheValues(cacheElements), actual(output.Count(0));
    int cached = 1;
    auto upload = [&](int seed) {
        for (int i = 0; i < (int)queries.size(); ++i) queries[i] = Value(i, seed);
        for (int i = 0; i < (int)keys.size(); ++i) { keys[i] = Value(i, seed + 3); values[i] = Value(i, seed + 7); }
        for (int h = 0; h < kvHeads; ++h) {
            for (int t = 0; t < capacity; ++t) {
                for (int c = 0; c < width; ++c) {
                    const size_t i = ((size_t)h * capacity + t) * width + c;
                    cacheKeys[i] = t < cached ? Value(i, seed + 11) : __float2half_rn(NAN);
                    cacheValues[i] = t < cached ? Value(i, seed + 13) : __float2half_rn(NAN);
                }
            }
        }
        Check(cudaMemcpy(q.cudaData, queries.data(), queries.size() * sizeof(half), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(key.cudaData, keys.data(), keys.size() * sizeof(half), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(value.cudaData, values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(cacheK.cudaData, cacheKeys.data(), cacheElements * sizeof(half), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(cacheV.cudaData, cacheValues.data(), cacheElements * sizeof(half), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(length.cudaData, &cached, sizeof(int), cudaMemcpyHostToDevice));
    };
    auto body = [&]() {
        Require(FastllmCudaDFlashAppendKVForGraph(key, cacheK, length), "append key failed");
        Require(FastllmCudaDFlashAppendKVForGraph(value, cacheV, length), "append value failed");
        Require(FastllmCudaDFlashAttention(q, cacheK, cacheV, output,
                    heads / kvHeads, 1.0f / std::sqrt((float)width), runtimeBlock, 256, &length),
                "graph attention failed");
    };
    upload(0);
    body();
    Check(cudaDeviceSynchronize());
    Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare failed");
    Require(FastllmCudaGraphMemoryPoolBegin(), "pool begin failed");
    Require(FastllmCudaGraphBeginCapture(), "capture begin failed");
    body();
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> reserved;
    Require(FastllmCudaGraphEndCapture(&graph), "capture end failed");
    Require(FastllmCudaGraphMemoryPoolEnd(reserved), "pool end failed");
    Require(FastllmCudaGraphInstantiate(graph, &exec), "instantiate failed");
    float worst = 0.0f;
    // Grow, reach capacity, then shrink/change the prefix as a compaction or
    // new request would. The same graph must consume the new GPU length.
    const std::vector<int> lengths = {0, 1, 7, 31, capacity - rows, 3, 64, 127};
    for (int iteration = 0; iteration < 24; ++iteration) {
        cached = std::min(capacity - rows, lengths[iteration % lengths.size()]);
        upload(iteration + 1);
        Require(FastllmCudaGraphLaunch(exec), "replay failed");
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<half> stored(cacheElements);
        for (int which = 0; which < 2; ++which) {
            const auto &prefix = which == 0 ? cacheKeys : cacheValues;
            const auto &draft = which == 0 ? keys : values;
            Check(cudaMemcpy(stored.data(), which == 0 ? cacheK.cudaData : cacheV.cudaData,
                             cacheElements * sizeof(half), cudaMemcpyDeviceToHost));
            for (int h = 0; h < kvHeads; ++h) for (int t = 0; t < capacity; ++t) for (int c = 0; c < width; ++c) {
                const size_t i = ((size_t)h * capacity + t) * width + c;
                float expected = t < cached ? __half2float(prefix[i]) :
                    (t < cached + rows ? __half2float(draft[((size_t)h * rows + t - cached) * width + c]) : 0.0f);
                Require(__half2float(stored[i]) == expected, "cache prefix/draft/padding mismatch");
            }
        }
        fastllm::Data referenceK, referenceV, reference;
        Tensor(referenceK, fastllm::FLOAT16, {kvHeads, cached + rows, width});
        Tensor(referenceV, fastllm::FLOAT16, {kvHeads, cached + rows, width});
        Tensor(reference, fastllm::FLOAT16, {heads, rows, width});
        for (int which = 0; which < 2; ++which) {
            std::vector<half> compact((size_t)kvHeads * (cached + rows) * width);
            const auto &prefix = which == 0 ? cacheKeys : cacheValues;
            const auto &draft = which == 0 ? keys : values;
            for (int h = 0; h < kvHeads; ++h) for (int t = 0; t < cached + rows; ++t) for (int c = 0; c < width; ++c) {
                compact[((size_t)h * (cached + rows) + t) * width + c] = t < cached ?
                    prefix[((size_t)h * capacity + t) * width + c] :
                    draft[((size_t)h * rows + t - cached) * width + c];
            }
            Check(cudaMemcpy(which == 0 ? referenceK.cudaData : referenceV.cudaData,
                compact.data(), compact.size() * sizeof(half), cudaMemcpyHostToDevice));
        }
        Require(FastllmCudaDFlashAttention(q, referenceK, referenceV, reference,
                    heads / kvHeads, 1.0f / std::sqrt((float)width), runtimeBlock, 256),
                "eager reference attention failed");
        std::vector<half> expected(actual.size());
        Check(cudaMemcpy(expected.data(), reference.cudaData, expected.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
            float difference = std::abs(__half2float(actual[i]) - __half2float(expected[i]));
            Require(std::isfinite(difference) && difference <= 0.002f, "graph/eager attention mismatch");
            worst = std::max(worst, difference);
        }
    }
    FastllmCudaGraphExecDestroy(exec);
    FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(reserved);
    std::cout << "capacity=" << capacity << " runtime_block=" << runtimeBlock
              << " replays=24 max_abs_error=" << worst << '\n';
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    Check(cudaSetDevice(0));
    for (int capacity : {72, 272, 536}) for (int runtime : {2, 4, 8}) Run(capacity, runtime);
    std::cout << "DFlash draft graph attention regression passed\n";
}
