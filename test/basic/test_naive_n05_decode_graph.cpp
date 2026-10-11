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
#include <cuda_runtime_api.h>
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
    Data q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), key(BFLOAT16), value(BFLOAT16), sink(FLOAT32),
        live(INT32);
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
        FastllmCudaNaiveDecodeAttention(q, key, value, indices, sink, live, capacity, heads, 1, dim,
                                        vd, window, scratch, actual);
    };
    fprintf(stderr, "Attention capture heads=%d dim=%d window=%d capacity=%d fp8=%d\n", heads, dim,
            window, capacity, fp8);
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
        FastllmCudaNaiveAttention(q, key, value, referenceIndices, sink, heads, 1, dim, vd, n - 1,
                                  window, expected);
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
            fprintf(stderr,
                    "attention heads=%d dim=%d window=%d cap=%d live=%d fp8=%d "
                    "mismatches=%d\n",
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
    auto originalK = Read<uint16_t>(key), originalV = Read<uint16_t>(value),
         newK = Read<uint16_t>(nk), newV = Read<uint16_t>(nv);
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
static void TestResidualNorm(int channels, int rows = 1) {
    Data hidden(BFLOAT16), branch(BFLOAT16), weight(FLOAT32), actual(BFLOAT16);
    Data reference(BFLOAT16), expected(BFLOAT16);
    Upload(hidden, {1, rows, channels}, 91);
    Upload(reference, {1, rows, channels}, 91);
    Upload(branch, {1, rows, channels}, 92);
    Upload(weight, {channels}, 93);
    Upload(expected, {1, rows, channels}, 94);
    auto original = Read<uint16_t>(hidden);
    auto body = [&]() { FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, weight, 1e-5f, actual); };
    Graph graph;
    graph.Capture(body);
    for (int step = 0; step < 3; ++step) {
        Cuda(cudaMemcpy(hidden.cudaData, original.data(), hidden.GetBytes(),
                        cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(reference.cudaData, original.data(), reference.GetBytes(),
                        cudaMemcpyHostToDevice));
        Require(FastllmCudaAddTo(reference, branch, 1), "reference add");
        Require(FastllmCudaKimiK3RMSNorm(reference, weight, expected, 1e-5f), "reference norm");
        graph.Run();
        Require(Read<uint16_t>(hidden) == Read<uint16_t>(reference), "fused residual differs");
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "fused norm differs");
        original = Read<uint16_t>(reference);
        ++checks;
    }
}
static void TestVerifyGraph(int window, int capacity, const std::vector<int> &prefixes, bool fp8, int rows = 8, int heads = 8) {
    const int dim = 192, vd = 128;
    const int kc = dim + (window ? 0 : 128), storage = (window ? window + rows : capacity) + 1;
    Data q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), key(BFLOAT16), value(BFLOAT16);
    Data nk(BFLOAT16), nv(BFLOAT16), sink(FLOAT32), live(INT32), indices, actual;
    Upload(q, {1, rows, heads * dim}, 11);
    Upload(iq, {1, rows, 2048}, 12);
    Upload(iw, {1, rows, 16}, 13);
    Upload(key, {1, storage, kc}, 14);
    Upload(value, {1, storage, vd}, 15);
    Upload(nk, {1, rows, kc}, 16);
    Upload(nv, {1, rows, vd}, 17);
    Upload(sink, {heads}, 18);
    live.Resize({rows});
    live.Allocate();
    for (int row = 0; row < rows; ++row)
        ((int *)live.cpuData)[row] = prefixes[0] + row + 1;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    auto originalK = Read<uint16_t>(key), originalV = Read<uint16_t>(value);
    auto newK = Read<uint16_t>(nk), newV = Read<uint16_t>(nv);
    FastllmNaiveDecodeScratch scratch;
    const bool sparse = !window && capacity > 2048;
    auto body = [&]() {
        FastllmCudaNaiveAppendVerifyCache(key, value, nk, nv, live, window);
        if (sparse)
            FastllmCudaNaiveGraphVerifyIndexer(iq, iw, key, live, capacity, fp8, scratch, indices);
        FastllmCudaNaiveGraphVerifyAttention(q, key, value, indices, sink, live, capacity, heads, 1,
                                             dim, vd, window, scratch, actual);
    };
    Graph graph;
    graph.Capture(body);
    for (int past : prefixes) {
        std::vector<int> lengths(rows);
        for (int row = 0; row < rows; ++row)
            lengths[row] = past + row + 1;
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(key.cudaData, originalK.data(), key.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(
            cudaMemcpy(value.cudaData, originalV.data(), value.GetBytes(), cudaMemcpyHostToDevice));
        graph.Run();
        auto expectedK = originalK, expectedV = originalV;
        const int localPast = window ? std::min(past, window - 1) : past;
        std::copy(newK.begin(), newK.end(), expectedK.begin() + localPast * kc);
        std::copy(newV.begin(), newV.end(), expectedV.begin() + localPast * vd);
        Require(Read<uint16_t>(key) == expectedK && Read<uint16_t>(value) == expectedV,
                "verification append or canary differs");
        auto result = Read<uint16_t>(actual);
        std::vector<int> selected;
        if (sparse)
            selected = Read<int>(indices);
        for (int row = 0; row < rows; ++row) {
            const int end = localPast + row + 1, begin = window ? std::max(0, end - window) : 0;
            Data query(BFLOAT16, {1, 1, heads * dim}), indexQ(BFLOAT16, {1, 1, 2048});
            Data indexW(BFLOAT16, {1, 1, 16}), k(BFLOAT16, {1, end - begin, kc});
            Data v(BFLOAT16, {1, end - begin, vd}), expected, chosen;
            query.FakeFrom(q, (size_t)row * heads * dim * 2);
            indexQ.FakeFrom(iq, (size_t)row * 2048 * 2);
            indexW.FakeFrom(iw, (size_t)row * 16 * 2);
            k.FakeFrom(key, (size_t)begin * kc * 2);
            v.FakeFrom(value, (size_t)begin * vd * 2);
            if (sparse) {
                FastllmCudaNaiveIndexer(indexQ, indexW, k, 16, 128, end - 1, 2048, fp8, chosen);
                auto ref = Read<int>(chosen);
                Require(std::equal(ref.begin(), ref.end(), selected.begin() + row * 2048),
                        "verification graph TopK differs");
            } else if (!window && capacity <= 256) {
                chosen.dataType = INT32;
                chosen.Resize({1, end});
                chosen.Allocate();
                for (int j = 0; j < end; ++j)
                    ((int *)chosen.cpuData)[j] = j;
                chosen.ToDevice(DataDevice::CUDA, {0}, true);
            }
            FastllmCudaNaiveAttention(query, k, v, chosen, sink, heads, 1, dim, vd, end - begin - 1,
                                      window, expected);
            auto ref = Read<uint16_t>(expected);
            Require(std::equal(ref.begin(), ref.end(), result.begin() + row * heads * vd),
                    "verification graph attention differs bitwise");
            ++checks;
        }
    }
}
// The CUB fallback sorts more than topK positions, but a verification row
// borrows only topK entries. Detect writes beyond the final row independently
// of allocator padding or whichever tensor happens to follow the output.
static void TestVerifyIndexerView(int rows, int past, bool fp8) {
    const int topK = 2048, keys = past + rows, canary = 4096;
    Data q(BFLOAT16), w(BFLOAT16), k(BFLOAT16), storage(INT32), indices(INT32);
    Upload(q, {1, rows, 2048}, 71);
    Upload(w, {1, rows, 16}, 72);
    Upload(k, {1, keys, 320}, 73);
    storage.Resize({rows * topK + canary});
    storage.Allocate();
    std::fill((int *)storage.cpuData, (int *)storage.cpuData + storage.Count(0), 0x12345678);
    storage.ToDevice(DataDevice::CUDA, {0}, true);
    indices.Resize({rows, topK});
    indices.FakeFrom(storage, 0);
    FastllmCudaNaiveVerifyIndexer(q, w, k, 16, 128, past, topK, fp8, indices);
    auto got = Read<int>(storage);
    Require(std::all_of(got.begin() + rows * topK, got.end(),
                       [](int x) { return x == 0x12345678; }), "borrowed TopK output overrun");
    for (int row = 0; row < rows; ++row) {
        Data qr(BFLOAT16, {1, 1, 2048}), wr(BFLOAT16, {1, 1, 16});
        Data kr(BFLOAT16, {1, past + row + 1, 320}), ref;
        qr.FakeFrom(q, row * 2048 * 2);
        wr.FakeFrom(w, row * 16 * 2);
        kr.FakeFrom(k, 0);
        FastllmCudaNaiveIndexer(qr, wr, kr, 16, 128, past + row, topK, fp8, ref);
        auto values = Read<int>(ref);
        Require(std::equal(values.begin(), values.end(), got.begin() + row * topK),
                "borrowed TopK indices differ");
        ++checks;
    }
}
static void TestDraftAttention(int dim, int rows, int window, bool shortAttention) {
    const int heads = 8, kvHeads = 2, storage = window + rows;
    Data q(BFLOAT16), key(BFLOAT16), value(BFLOAT16), nk(BFLOAT16), nv(BFLOAT16);
    Data live(INT32), scores, actual;
    Upload(q, {1, rows, heads * dim}, 101);
    Upload(key, {1, storage, kvHeads * dim}, 102);
    Upload(value, {1, storage, kvHeads * dim}, 103);
    Upload(nk, {1, rows, kvHeads * dim}, 104);
    Upload(nv, {1, rows, kvHeads * dim}, 105);
    live.Resize({1}); live.Allocate();
    ((int *)live.cpuData)[0] = shortAttention ? 4 : 301;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    auto originalK = Read<uint16_t>(key), originalV = Read<uint16_t>(value);
    auto newK = Read<uint16_t>(nk), newV = Read<uint16_t>(nv);
    auto body = [&]() {
        FastllmCudaNaiveAppendVerifyCache(key, value, nk, nv, live, window);
        FastllmCudaNaiveDraftAttention(q, key, value, live, heads, kvHeads, dim,
                                       window, shortAttention, scores, actual);
    };
    Graph graph; graph.Capture(body);
    std::vector<int> prefixes = shortAttention ? std::vector<int>{0, 3, 80, 256 - rows, 5}
        : std::vector<int>{257 - rows, 300, 1023, 1024, 32768, 131064, 300};
    for (int prefix : prefixes) {
        int length = prefix + 1, past = std::min(prefix, window - 1);
        Cuda(cudaMemcpy(live.cudaData, &length, sizeof(length), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(key.cudaData, originalK.data(), key.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(value.cudaData, originalV.data(), value.GetBytes(), cudaMemcpyHostToDevice));
        graph.Run();
        auto expectedK = originalK, expectedV = originalV;
        std::copy(newK.begin(), newK.end(), expectedK.begin() + past * kvHeads * dim);
        std::copy(newV.begin(), newV.end(), expectedV.begin() + past * kvHeads * dim);
        Require(Read<uint16_t>(key) == expectedK && Read<uint16_t>(value) == expectedV,
                "draft append or canary differs");
        Data k(BFLOAT16, {1, past + rows, kvHeads * dim}), v(BFLOAT16, k.dims), expected;
        k.FakeFrom(key, 0); v.FakeFrom(value, 0);
        FastllmCudaNaiveAttention(q, k, v, Data(), Data(), heads, kvHeads, dim, dim,
                                  past, window, expected, false);
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected),
                "dynamic draft attention differs from eager");
        ++checks;
    }
}

// Exercise the production append/attention kernels across multiple wraps, then
// leave/re-enter the ring as prefill, verification and history restore do.
static void TestSwaRing(int heads, int kvHeads, bool fused) {
    const int kc = kvHeads * 192, vc = kvHeads * 128;
    Data q(BFLOAT16), packed(BFLOAT16), nk(BFLOAT16), nv(BFLOAT16), pos(FLOAT32), live(INT32), sink(FLOAT32);
    Data oldK(BFLOAT16), oldV(BFLOAT16), ringK(BFLOAT16), ringV(BFLOAT16), expected, actual, none;
    Upload(q, {1, 1, heads * 192}, 71);
    Upload(packed, {1, 1, (heads + kvHeads) * 192 + vc}, 72);
    Upload(nk, {1, 1, kc}, 73); Upload(nv, {1, 1, vc}, 74);
    Upload(pos, {1, 1}, 75); Upload(live, {1}, 76); Upload(sink, {heads}, 77);
    Upload(oldK, {1, 256, kc}, 78); Upload(oldV, {1, 256, vc}, 79);
    oldK.Expansion({1, 256, kc}); oldV.Expansion({1, 256, vc});
    ringK.CopyFrom(oldK); ringV.CopyFrom(oldV);
    ringK.Expansion({1, 256, kc}); ringV.Expansion({1, 256, vc});
    const auto initialK = Read<uint16_t>(oldK), initialV = Read<uint16_t>(oldV);
    FastllmNaiveDecodeScratch oldScratch, ringScratch;
    int length = 128;
    Cuda(cudaMemcpy(live.cudaData, &length, sizeof(length), cudaMemcpyHostToDevice));
    auto invoke = [&](bool ring) {
        Data &k = ring ? ringK : oldK, &v = ring ? ringV : oldV;
        if (fused) Require(FastllmCudaNaiveRopeAppendCache(q, nk, nv, none, pos, k, v,
            live, heads, kvHeads, 192, 128, 64, 10000.f, .75f, 128, &packed, ring), "fused ring append");
        else FastllmCudaNaiveAppendDecodeCache(k, v, nk, nv, live, 128, ring);
        FastllmCudaNaiveDecodeAttention(q, k, v, none, sink, live, 256, heads, kvHeads,
            192, 128, 128, ring ? ringScratch : oldScratch, ring ? actual : expected, ring);
        if (!ring) FastllmCudaNaiveTrimDecodeCache(k, v, live, 128);
    };
    Graph graph;
    graph.Capture([&]() { invoke(false); invoke(true); });
    auto rows = [](const Data &d) {
        std::vector<uint16_t> result((size_t)d.dims[1] * d.dims[2]);
        Cuda(cudaMemcpy(result.data(), d.cudaData, result.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        return result;
    };
    for (int prefix : {0, 126, 127, 128, 707, 2047}) {
        for (Data *k : {&oldK, &ringK})
            Cuda(cudaMemcpy(k->cudaData, initialK.data(), initialK.size() * 2, cudaMemcpyHostToDevice));
        for (Data *v : {&oldV, &ringV})
            Cuda(cudaMemcpy(v->cudaData, initialV.data(), initialV.size() * 2, cudaMemcpyHostToDevice));
        auto resize = [&](int past) {
            for (Data *k : {&oldK, &ringK}) k->Resize({1, std::min(past, 127), kc});
            for (Data *v : {&oldV, &ringV}) v->Resize({1, std::min(past, 127), vc});
        };
        resize(prefix);
        FastllmCudaNaiveSwaCacheLayout(ringK, ringV, prefix, true);
        for (int step = 0; step < 260; ++step) {
            length = prefix + step + 1;
            float position = length - 1;
            Cuda(cudaMemcpy(live.cudaData, &length, sizeof(length), cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(pos.cudaData, &position, sizeof(position), cudaMemcpyHostToDevice));
            graph.Run();
            Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "SWA ring output mismatch");
            resize(length);
            // Conversion itself must preserve every retained row, including at
            // wrap boundaries. This also simulates leaving/re-entering decode.
            FastllmCudaNaiveSwaCacheLayout(ringK, ringV, length, false);
            Require(rows(ringK) == rows(oldK), "SWA ring key mismatch");
            Require(rows(ringV) == rows(oldV), "SWA ring value mismatch");
            FastllmCudaNaiveSwaCacheLayout(ringK, ringV, length, true);
            ++checks;
        }
    }
}

static void TestSwaBatch(int batch, int heads, int kvHeads) {
    Data query(BFLOAT16), sink(FLOAT32), live(INT32), actual, expected(BFLOAT16), none;
    Upload(query, {1, batch, heads * 192}, 91);
    Upload(sink, {heads}, 92);
    Upload(live, {batch}, 93);
    Upload(expected, {1, batch, heads * 128}, 94);
    std::vector<Data> keys(batch, Data(BFLOAT16)), values(batch, Data(BFLOAT16));
    std::vector<FastllmNaiveSWARow> rows;
    for (int b = 0; b < batch; ++b) {
        Upload(keys[b], {1, 128, kvHeads * 192}, 95 + b);
        Upload(values[b], {1, 128, kvHeads * 128}, 125 + b);
        rows.push_back({keys[b].cudaData, values[b].cudaData});
    }
    std::vector<int> lengths(batch, 128);
    Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
    Graph graph;
    graph.Capture([&]() {
        FastllmCudaNaiveBatchSWADecode(query, sink, rows, live, heads, kvHeads, actual);
    });
    for (int prefix : {1, 127, 707}) {
        for (int b = 0; b < batch; ++b) lengths[b] = prefix + b;
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        graph.Run();
        for (int b = 0; b < batch; ++b) {
            Data q, length, result;
            q.Resize({1, 1, heads * 192}); q.FakeFrom(query, (size_t)b * heads * 192 * 2);
            length.Resize({1}); length.FakeFrom(live, b * sizeof(int));
            result.Resize({1, 1, heads * 128}); result.FakeFrom(expected, (size_t)b * heads * 128 * 2);
            FastllmNaiveDecodeScratch scratch;
            FastllmCudaNaiveDecodeAttention(q, keys[b], values[b], none, sink, length,
                128, heads, kvHeads, 192, 128, 128, scratch, result, true);
        }
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "batched SWA differs from individual rows");
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
        for (int batch : {1, 2, 3, 4, 17})
            for (int heads : {4, 8, 16})
                TestSwaBatch(batch, heads, heads == 16 ? 2 : 1);
        for (int heads : {4, 8, 16})
            for (bool fused : {false, true}) TestSwaRing(heads, heads == 16 ? 2 : 1, fused);
        for (int ch : {1, 127, 256, 1024, 2048, 4096, 5120})
            TestResidualNorm(ch);
        for (int ch : {127, 256, 4096})
            TestResidualNorm(ch, 8);
        for (int w : {0, 8, 128})
            TestCache(w);
        // Dense decode uses replay-time length, including decreases and the
        // 64-key staging tail; keep the optimized path bitwise equal to eager.
        for (int heads : {1, 8, 16, 32})
            for (bool sink : {false, true})
                TestAttention(heads, 192, 128, 0, 896,
                              {1, 63, 64, 65, 707, 748, 895, 896, 257}, true, sink);
        TestAttention(8, 192, 128, 0, 2048, {257, 1024, 2047, 2048, 513}, true);
        for (int heads : {2, 4, 8, 16}) {
            for (bool sink : {false, true})
                TestAttention(heads, 192, 128, 0, 256, {1, 2, 127, 128, 129, 255, 256, 3}, true,
                              sink);
            TestAttention(heads, 192, 128, 0, 2047, {257, 511, 512, 1023, 2047, 258}, true);
            TestAttention(heads, 192, 128, 128, 256, {2, 127, 128, 129, 256, 3}, true);
            TestAttention(heads, 32, 16, 8, 256, {2, 7, 8, 9, 256, 3}, false);
            for (bool fp8 : {false, true}) {
                TestAttention(heads, 192, 128, 0, 4096, {2049, 2050, 4095, 4096, 2051}, fp8);
                TestAttention(heads, 192, 128, 0, 32769, {8191, 8192, 8193, 32768, 32769, 8194},
                              fp8);
            }
        }
        TestVerifyGraph(0, 256, {3, 120, 127, 248, 7}, true);
        TestVerifyGraph(0, 2047, {256, 511, 2039, 257}, true);
        for (int w : {8, 128})
            TestVerifyGraph(w, 256, {3, 120, 127, 128, 32768, 7}, true);
        for (bool fp8 : {false, true})
            TestVerifyGraph(0, 131072, {2048, 4095, 32768, 131064, 2049}, fp8);
        for (int rows : {1, 7, 8})
            for (int past : {2048, 4090, 8183})
                for (bool fp8 : {false, true}) TestVerifyIndexerView(rows, past, fp8);
        for (int rows = 1; rows <= 8; ++rows)
            for (bool fp8 : {false, true})
                TestVerifyGraph(0, 32776, {2048, 32768, 2049}, fp8, rows);
        for (int heads : {4, 12, 16})
            TestVerifyGraph(0, 4096, {2048, 4088, 2049}, true, 8, heads);
        for (int dim : {32, 128, 256}) for (int rows : {2, 7, 8})
            for (bool shortAttention : {false, true})
                TestDraftAttention(dim, rows, 1024, shortAttention);
        printf("DECODE GRAPH PASS checks=%d\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
