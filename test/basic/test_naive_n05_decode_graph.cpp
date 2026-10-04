#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static int checks = 0;
static void Require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
static void Cuda(cudaError_t err) { Require(err == cudaSuccess, cudaGetErrorString(err)); }
static void Upload(Data &d, std::vector<int> dims, int seed) {
    d.Resize(dims);
    d.Allocate();
    unsigned x = seed;
    for (size_t i = 0; i < d.Count(0); ++i) {
        x = 1664525u * x + 1013904223u;
        float v = ((int)(x >> 16) - 32768) / 32768.f;
        if (d.dataType == BFLOAT16)
            ((uint16_t *)d.cpuData)[i] = Float32ToBFloat16RNEBits(v);
        else
            ((float *)d.cpuData)[i] = v;
    }
    d.ToDevice(DataDevice::CUDA, {0}, true);
}
template <typename T> static std::vector<T> Read(const Data &d) {
    std::vector<T> v(d.GetBytes() / sizeof(T));
    Cuda(cudaMemcpy(v.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return v;
}
struct Graph {
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> reserved;
    template <class F> void Capture(F fn) {
        fn();
        Cuda(cudaDeviceSynchronize());
        Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare graph");
        Require(FastllmCudaGraphMemoryPoolBegin(), "begin memory pool");
        Require(FastllmCudaGraphBeginCapture(), "begin capture");
        fn();
        Require(FastllmCudaGraphEndCapture(&graph), FastllmCudaGraphLastError());
        Require(FastllmCudaGraphMemoryPoolEnd(reserved), "end memory pool");
        Require(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
    }
    void Run() {
        Require(FastllmCudaGraphLaunch(exec), "launch graph");
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ~Graph() {
        FastllmCudaGraphExecDestroy(exec);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
    }
};
static void TestAttention(int heads, int dim, int vd, int window, int capacity,
                          const std::vector<int> &lengths, bool fp8, bool withSink = true) {
    Data q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), key(BFLOAT16), value(BFLOAT16), sink(FLOAT32), live(INT32);
    Upload(q, {1, 1, heads * dim}, 1);
    Upload(iq, {1, 1, 16 * 128}, 2);
    Upload(iw, {1, 1, 16}, 3);
    int storage = window ? window : capacity, stride = dim + (window ? 0 : 128);
    Upload(key, {1, storage, stride}, 4);
    Upload(value, {1, storage, vd}, 5);
    if (withSink)
        Upload(sink, {heads}, 6);
    live.Resize({1});
    live.Allocate();
    ((int *)live.cpuData)[0] = lengths[0];
    live.ToDevice(DataDevice::CUDA, {0}, true);
    FastllmNaiveDecodeScratch scratch;
    Data indices, actual, expected;
    bool sparse = !window && capacity > 2048;
    auto body = [&]() {
        if (sparse)
            FastllmCudaNaiveDecodeIndexer(iq, iw, key, live, capacity, fp8, scratch, indices);
        FastllmCudaNaiveDecodeAttention(q, key, value, indices, sink, live, capacity, heads, 1, dim, vd,
                                        window, scratch, actual);
    };
    fprintf(stderr, "Attention capture heads=%d dim=%d window=%d capacity=%d fp8=%d\n", heads, dim, window,
            capacity, fp8);
    Graph graph;
    graph.Capture(body);
    for (int length : lengths) {
        Cuda(cudaMemcpy(live.cudaData, &length, sizeof(int), cudaMemcpyHostToDevice));
        int n = window ? std::min(window, length) : length;
        key.Resize({1, n, stride});
        value.Resize({1, n, vd});
        Data referenceIndices;
        if (sparse)
            FastllmCudaNaiveIndexer(iq, iw, key, 16, 128, n - 1, 2048, fp8, referenceIndices);
        else if (!window && capacity <= 256) {
            // Explicit identity indices retain the independent generic kernel
            // while the captured path uses the tiled short-decode kernel.
            referenceIndices.dataType = INT32;
            referenceIndices.Resize({1, n});
            referenceIndices.Allocate();
            for (int i = 0; i < n; ++i)
                ((int *)referenceIndices.cpuData)[i] = i;
            referenceIndices.ToDevice(DataDevice::CUDA, {0}, true);
        }
        FastllmCudaNaiveAttention(q, key, value, referenceIndices, sink, heads, 1, dim, vd, n - 1, window,
                                  expected);
        auto ref = Read<uint16_t>(expected);
        graph.Run();
        if (sparse)
            Require(Read<int>(indices) == Read<int>(referenceIndices), "dynamic TopK differs");
        auto got = Read<uint16_t>(actual);
        if (got != ref) {
            int count = 0;
            for (size_t i = 0; i < got.size(); ++i)
                if (got[i] != ref[i])
                    ++count;
            fprintf(stderr, "attention heads=%d dim=%d window=%d cap=%d live=%d fp8=%d mismatches=%d\n",
                    heads, dim, window, capacity, length, fp8, count);
        }
        Require(got == ref, "dynamic attention differs bitwise");
        ++checks;
    }
}
static void TestCache(int window) {
    const int kc = 320, vc = 128, rows = window ? window + 1 : 258;
    Data key(BFLOAT16), value(BFLOAT16), nk(BFLOAT16), nv(BFLOAT16), live(INT32);
    Upload(key, {1, rows, kc}, 13);
    Upload(value, {1, rows, vc}, 14);
    Upload(nk, {1, 1, kc}, 15);
    Upload(nv, {1, 1, vc}, 16);
    auto originalK = Read<uint16_t>(key), originalV = Read<uint16_t>(value), newK = Read<uint16_t>(nk),
         newV = Read<uint16_t>(nv);
    live.Resize({1});
    live.Allocate();
    ((int *)live.cpuData)[0] = 2;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    auto body = [&]() {
        FastllmCudaNaiveAppendDecodeCache(key, value, nk, nv, live, window);
        if (window)
            FastllmCudaNaiveTrimDecodeCache(key, value, live, window);
    };
    fprintf(stderr, "Cache capture window=%d\n", window);
    Graph graph;
    graph.Capture(body);
    for (int length : {2, 7, 8, 9, 127, 128, 129, 256, 257}) {
        auto k = originalK, v = originalV;
        Cuda(cudaMemcpy(key.cudaData, k.data(), key.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(value.cudaData, v.data(), value.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, &length, sizeof(int), cudaMemcpyHostToDevice));
        int row = window ? std::min(length - 1, window - 1) : length - 1;
        std::copy(newK.begin(), newK.end(), k.begin() + row * kc);
        std::copy(newV.begin(), newV.end(), v.begin() + row * vc);
        if (window && length >= window) {
            std::memmove(k.data(), k.data() + kc, (window - 1) * kc * sizeof(uint16_t));
            std::memmove(v.data(), v.data() + vc, (window - 1) * vc * sizeof(uint16_t));
        }
        graph.Run();
        Require(Read<uint16_t>(key) == k && Read<uint16_t>(value) == v,
                "dynamic append/trim or canary differs");
        ++checks;
    }
}
static void TestResidualNorm(int channels) {
    Data hidden(BFLOAT16), branch(BFLOAT16), weight(FLOAT32), actual(BFLOAT16);
    Data reference(BFLOAT16), expected(BFLOAT16);
    Upload(hidden, {1, 1, channels}, 91);
    Upload(reference, {1, 1, channels}, 91);
    Upload(branch, {1, 1, channels}, 92);
    Upload(weight, {channels}, 93);
    Upload(expected, {1, 1, channels}, 94);
    auto original = Read<uint16_t>(hidden);
    auto body = [&]() { FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, weight, 1e-5f, actual); };
    Graph graph;
    graph.Capture(body);
    for (int step = 0; step < 3; ++step) {
        Cuda(cudaMemcpy(hidden.cudaData, original.data(), hidden.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(reference.cudaData, original.data(), reference.GetBytes(), cudaMemcpyHostToDevice));
        Require(FastllmCudaAddTo(reference, branch, 1), "reference add");
        Require(FastllmCudaKimiK3RMSNorm(reference, weight, expected, 1e-5f), "reference norm");
        graph.Run();
        Require(Read<uint16_t>(hidden) == Read<uint16_t>(reference), "fused residual differs");
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "fused norm differs");
        original = Read<uint16_t>(reference);
        ++checks;
    }
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        FastllmCudaSetDevice(0);
        if (!FastllmCudaNaiveDecodeGraphSupported())
            return 77;
        for (int ch : {1, 127, 256, 1024, 2048, 4096, 5120})
            TestResidualNorm(ch);
        for (int w : {0, 8, 128})
            TestCache(w);
        for (int heads : {2, 4, 8, 16}) {
            for (bool sink : {false, true})
                TestAttention(heads, 192, 128, 0, 256, {1, 2, 127, 128, 129, 255, 256, 3}, true, sink);
            TestAttention(heads, 192, 128, 0, 2047, {257, 511, 512, 1023, 2047, 258}, true);
            TestAttention(heads, 192, 128, 128, 256, {2, 127, 128, 129, 256, 3}, true);
            TestAttention(heads, 32, 16, 8, 256, {2, 7, 8, 9, 256, 3}, false);
            for (bool fp8 : {false, true}) {
                TestAttention(heads, 192, 128, 0, 4096, {2049, 2050, 4095, 4096, 2051}, fp8);
                TestAttention(heads, 192, 128, 0, 32769, {8191, 8192, 8193, 32768, 32769, 8194}, fp8);
            }
        }
        printf("DECODE GRAPH PASS checks=%d\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
