#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static int checks = 0;
static void Require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
static void Cuda(cudaError_t err) {
    Require(err == cudaSuccess, cudaGetErrorString(err));
}
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

static float Float(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float x;
    std::memcpy(&x, &word, sizeof(x));
    return x;
}
static void Test(int batch, int heads, int kvHeads, int dim, int vd, int capacity, bool sparse, bool withSink,
                 bool unalignedKeys = false, bool mixed = false) {
    const int stride = kvHeads * dim + 128 + (unalignedKeys ? 1 : 0);
    Data q(BFLOAT16), sink(FLOAT32), live(INT32), output;
    Upload(q, {1, batch, heads * dim}, 31);
    if (withSink)
        Upload(sink, {heads}, 51);
    std::vector<int> lengths(batch, capacity);
    live.Resize({batch});
    live.Allocate();
    std::memcpy(live.cpuData, lengths.data(), batch * sizeof(int));
    live.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<Data> keys(batch, Data(BFLOAT16)), values(batch, Data(BFLOAT16)), indices(batch, Data(INT32));
    std::vector<std::vector<uint16_t>> hostKeys(batch), hostValues(batch);
    std::vector<std::vector<int>> hostIndices(batch);
    std::vector<FastllmNaiveDSARow> rows(batch);
    for (int b = 0; b < batch; ++b) {
        Upload(keys[b], {1, capacity, stride}, 7 + b);
        Upload(values[b], {1, capacity, kvHeads * vd}, 17 + b);
        hostKeys[b] = Read<uint16_t>(keys[b]);
        hostValues[b] = Read<uint16_t>(values[b]);
        auto &r = rows[b];
        r.key = keys[b].cudaData;
        r.value = values[b].cudaData;
        r.capacity = capacity;
        r.keyStride = stride;
        r.liveKeys = (const int *)live.cudaData + b;
        if (sparse && (!mixed || b % 2)) {
            indices[b].Resize({1, 2048});
            indices[b].Allocate();
            hostIndices[b].resize(2048);
            for (int i = 0; i < 2048; ++i)
                hostIndices[b][i] = i % 17 == 0 ? -1 : (i * 37 + b * 19) % capacity;
            std::memcpy(indices[b].cpuData, hostIndices[b].data(), 2048 * sizeof(int));
            indices[b].ToDevice(DataDevice::CUDA, {0}, true);
            r.indices = (const int *)indices[b].cudaData;
            r.selectedCount = 2048;
        }
    }
    auto hostQ = Read<uint16_t>(q);
    std::vector<float> hostSink = withSink ? Read<float>(sink) : std::vector<float>();
    FastllmNaiveDecodeScratch scratch;
    auto body = [&]() {
        FastllmCudaNaiveBatchDSADecode(q, sink, rows, heads, kvHeads, dim, vd, scratch, output);
    };
    Graph graph;
    graph.Capture(body);
    void *scratchAddress = scratch.attentionScores.cudaData;
    size_t scratchSize = scratch.attentionScores.GetBytes();
    for (int replay = 0; replay < 3; ++replay) {
        for (int b = 0; b < batch; ++b)
            lengths[b] = replay == 0   ? std::max(0, capacity - b * 13)
                         : replay == 1 ? std::min(capacity, b * 3 + 1)
                                       : std::max(0, capacity / 2 - b * 7);
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        graph.Run();
        auto actual = Read<uint16_t>(output);
        int different = 0;
        double oldMaxError = 0;
        FastllmNaiveDecodeScratch oldScratch;
        for (int b = 0; b < batch; ++b) {
            Data rowQuery(BFLOAT16), rowLive(INT32), oldOutput;
            rowQuery.Resize({1, 1, heads * dim});
            rowQuery.FakeFrom(q, (size_t)b * heads * dim * sizeof(uint16_t));
            rowLive.Resize({1});
            rowLive.FakeFrom(live, b * sizeof(int));
            FastllmCudaNaiveDecodeAttention(rowQuery, keys[b], values[b], indices[b], sink, rowLive, capacity,
                                            heads, kvHeads, dim, vd, 0, oldScratch, oldOutput);
            auto old = Read<uint16_t>(oldOutput);
            for (int i = 0; i < heads * vd; ++i) {
                uint16_t got = actual[(size_t)b * heads * vd + i];
                different += got != old[i];
                double delta = std::abs((double)Float(got) - Float(old[i]));
                oldMaxError = std::max(oldMaxError, delta);
                Require(delta <= 0.009 * std::abs(Float(old[i])) + 2e-6,
                        "DSA differs from original rounding by more than one BF16 ULP");
            }
        }
        if (capacity > 256)
            Require(different == 0, "Batch DSA differs from original decode");
        std::printf("Original decode comparison different=%d max=%.9g\n", different, oldMaxError);
        double squared = 0, referenceSquared = 0, maxError = 0;
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < heads; ++h) {
                bool selected = !hostIndices[b].empty();
                int kh = h / (heads / kvHeads), count = selected ? 2048 : lengths[b];
                std::vector<double> scores(count, -INFINITY);
                double maximum = withSink ? hostSink[h] : -INFINITY;
                for (int i = 0; i < count; ++i) {
                    int index = selected ? hostIndices[b][i] : i;
                    if (index < 0 || index >= lengths[b])
                        continue;
                    double dot = 0;
                    for (int d = 0; d < dim; ++d)
                        dot += (double)Float(hostQ[((size_t)b * heads + h) * dim + d]) *
                               Float(hostKeys[b][(size_t)index * stride + kh * dim + d]);
                    scores[i] = dot / std::sqrt((double)dim);
                    maximum = std::max(maximum, scores[i]);
                }
                double sum = withSink ? std::exp(hostSink[h] - maximum) : 0;
                std::vector<double> expected(vd, 0);
                for (int i = 0; i < count; ++i) {
                    if (!std::isfinite(scores[i]))
                        continue;
                    double weight = std::exp(scores[i] - maximum);
                    sum += weight;
                    int index = selected ? hostIndices[b][i] : i;
                    for (int d = 0; d < vd; ++d)
                        expected[d] += weight * Float(hostValues[b][((size_t)index * kvHeads + kh) * vd + d]);
                }
                for (int d = 0; d < vd; ++d) {
                    double ref = sum > 0 ? expected[d] / sum : 0;
                    double got = Float(actual[((size_t)b * heads + h) * vd + d]);
                    Require(std::isfinite(got), "nonfinite batch attention");
                    double delta = got - ref;
                    squared += delta * delta;
                    referenceSquared += ref * ref;
                    maxError = std::max(maxError, std::abs(delta));
                }
            }
        double nrmse = std::sqrt(squared / std::max(referenceSquared, 1e-20));
        std::printf(
            "DSA B=%d H=%d HK=%d D=%d V=%d cap=%d sparse=%d unaligned_keys=%d mixed=%d replay=%d NRMSE=%.6g max=%.6g\n",
            batch, heads, kvHeads, dim, vd, capacity, sparse, unalignedKeys, mixed, replay, nrmse, maxError);
        Require(nrmse < 0.008 || maxError < 0.002, "DSA exceeds BF16 reference tolerance");
        Require(scratch.attentionScores.cudaData == scratchAddress &&
                    scratch.attentionScores.GetBytes() == scratchSize,
                "graph replay changed scratch allocation");
        ++checks;
    }
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        FastllmCudaSetDevice(0);
        Test(1, 8, 1, 192, 128, 896, false, true);
        Test(2, 8, 1, 192, 128, 896, false, true);
        for (int batch : {3, 4, 8}) {
            Test(batch, 8, 1, 192, 128, 896, false, true);
            Test(batch, 8, 1, 192, 128, 2047, false, false);
        }
        Test(1, 8, 1, 192, 128, 4097, true, true);
        Test(4, 8, 1, 192, 128, 4097, true, true);
        Test(2, 16, 2, 192, 128, 896, false, false);
        Test(2, 32, 1, 192, 128, 64, false, true);
        Test(4, 8, 1, 192, 128, 896, false, true, true);
        Test(2, 8, 1, 192, 128, 4097, true, true, true);
        Test(33, 4, 1, 128, 256, 31, false, false);
        Test(2, 8, 1, 128, 256, 4097, true, true, true);
        Test(2, 8, 1, 192, 128, 1536, true, true, false, true);
        std::printf("BATCH DSA PASS %d checks\n", checks);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
