#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "devices/cuda/naive-n05-swa-prefill.cuh"
#include "fastllm.h"
#include "blocks/baseblock.h"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <limits>
#include <numeric>
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
        else if (d.dataType == FP8_E4M3)
            d.cpuData[i] = __nv_fp8_e4m3(v).__x;
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
// Compare the producer's bytes with independent host quantization of its BF16 output.
struct PackedOutput {
    Data values{INT32}, scales{INT32};
    FastllmMxfp8Activation view{};
    void Init(int rows, int width) {
        int pitch = (rows + 3) / 4 * 4;
        Upload(values, {rows * width / 4}, 601);
        Upload(scales, {pitch * width / 128}, 602);
        view = {static_cast<uint8_t *>(values.cudaData), static_cast<uint8_t *>(scales.cudaData),
                rows, width, pitch};
    }
    void Check(const Data &output, int first = 0, float minimum = 0.f) const {
        auto bf16 = Read<uint16_t>(output);
        auto bytes = Read<uint8_t>(values), exponents = Read<uint8_t>(scales);
        for (int row = 0; row < int(bf16.size() / view.width); ++row) for (int col = 0; col < view.width; col += 32) {
            float block[32], maximum = minimum;
            for (int i = 0; i < 32; ++i) {
                uint32_t bits = uint32_t(bf16[row * view.width + col + i]) << 16;
                std::memcpy(&block[i], &bits, sizeof(bits));
                maximum = std::max(maximum, std::abs(block[i]));
            }
            int exponent = maximum ? std::max(-126, int(std::ceil(std::log2(maximum / 448.f)))) : -126;
            Require(exponents[(col / 128 * view.pitch + first + row) * 4 + col / 32 % 4] == exponent + 127,
                    "incorrect projection activation scale");
            for (int i = 0; i < 32; ++i)
                Require(bytes[(first + row) * view.width + col + i] == __nv_fp8_e4m3(std::ldexp(block[i], -exponent)).__x,
                        "incorrect projection activation value");
        }
    }
};
static void TestPrefillProjection(bool swa) {
    const int kvHeads = swa ? 1 : 2, heads = kvHeads * (swa ? 8 : 16), width = heads * 128;
    // A short SWA request in the same batch must leave its slice unclaimed.
    const std::vector<int> lengths = swa ? std::vector<int>{1025, 8192, 31} : std::vector<int>{129, 131, 133};
    PackedOutput packed;
    packed.Init(std::accumulate(lengths.begin(), lengths.end(), 0), width);
    int first = 0, completed = 0;
    for (int rows : lengths) {
        // The last DSA request has a scattered union of 32K keys per 16 rows.
        // Its direct FP8 kernel must fill the same packed output as the FMHA tiles.
        const bool direct = !swa && rows == lengths.back();
        Data q(BFLOAT16), k(direct ? FP8_E4M3 : BFLOAT16), v(k.dataType), ids(INT32), sink(FLOAT32);
        Data scratch(FLOAT32), actual(BFLOAT16), expected(BFLOAT16);
        // The second DSA request has a sparse history, multiple KV-head
        // stripes and a partial last query chunk, all at a nonzero row offset.
        const int keys = direct ? 65536 : !swa && first ? 4096 : rows;
        Upload(q, {1, rows, heads * 192}, 501 + first);
        Upload(k, {1, keys, kvHeads * 192}, 502 + first);
        Upload(v, {1, keys, kvHeads * 128}, 503 + first);
        auto slice = FastllmMxfp8PrefillOutput::ForRows(packed.view, first, rows, completed);
        if (swa) {
            Upload(sink, {heads}, 504);
            FastllmCudaNaivePrefillAttention(q, k, v, ids, sink, heads, kvHeads, 192, 128, 0, 128, expected, true, nullptr, nullptr);
            FastllmCudaNaivePrefillAttention(q, k, v, ids, sink, heads, kvHeads, 192, 128, 0, 128, actual, true, nullptr, nullptr, slice);
        } else {
            const int topK = direct ? 2048 : 32;
            scratch.Resize({direct ? 16 * 1024 * 1024 : 1536 * 1024});
            scratch.ToDevice(CUDA, {0}, false); scratch.Allocate(false);
            ids.Resize({rows, topK}); ids.Allocate();
            for (int row = 0; row < rows; ++row) {
                for (int i = 0; i < topK; ++i)
                    reinterpret_cast<int *>(ids.cpuData)[row * topK + i] =
                        direct ? i * 32 + row % 16 : (row % 5 ? i : -1);
            }
            ids.ToDevice(CUDA, {0}, true);
            FastllmNaivePrefillAttentionItem item{&q, &k, &v, &ids, &expected, keys - rows};
            Require(FastllmCudaTryNaiveIndexerAttention(item, sink, heads, kvHeads, 192, 128, scratch), "DSA prefill reference");
            item.output = &actual;
            item.packed = slice;
            Require(FastllmCudaTryNaiveIndexerAttention(item, sink, heads, kvHeads, 192, 128, scratch), "DSA prefill pack");
        }
        const bool packedRequest = !swa || rows != 31;
        Require(completed == first + (packedRequest ? rows : 0), "incorrect prefill completion count");
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "prefill BF16 changed");
        if (packedRequest) packed.Check(expected, first);
        first += rows;
    }
    Require((completed == packed.view.rows) == !swa, "mixed prefill admitted an incomplete matrix");
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
// One long-capacity graph must also handle shorter requests, padding, and
// dense/sparse transitions. Compare every live row to ordinary eager attention.
static void TestMixedRegionAttention() {
    constexpr int rows = 4, heads = 64, kvHeads = 4, capacity = 8192, stride = kvHeads * 192 + 128;
    Data q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), live(INT32), sink(FLOAT32), actual;
    Upload(q, {1, rows, heads * 192}, 73);
    Upload(iq, {1, rows, 16 * 128}, 79);
    Upload(iw, {1, rows, 16}, 83);
    Upload(sink, {heads}, 89);
    live.Resize({rows}); live.Allocate();
    std::fill_n((int *)live.cpuData, rows, 3);
    live.ToDevice(CUDA, {0}, true);
    std::vector<Data> keys(rows), values(rows), indices(rows);
    std::vector<FastllmNaiveDecodeScratch> indexScratch(rows);
    FastllmNaiveDecodeScratch attentionScratch;
    std::vector<FastllmNaiveIndexerRow> indexRows;
    for (int i = 0; i < rows; ++i) {
        keys[i].dataType = values[i].dataType = BFLOAT16;
        keys[i].UpdateUnitSize(); values[i].UpdateUnitSize();
        Upload(keys[i], {1, capacity, stride}, 101 + i);
        Upload(values[i], {1, capacity, kvHeads * 128}, 131 + i);
        indexRows.push_back({&keys[i], &indexScratch[i], &indices[i], i, capacity});
    }
    auto body = [&] {
        Require(FastllmCudaNaiveBatchDecodeIndexer(iq, iw, live, indexRows, true), "mixed-region Indexer unsupported");
        std::vector<FastllmNaiveDSARow> descriptors(rows);
        for (int i = 0; i < rows; ++i) {
            auto &d = descriptors[i];
            d.key = keys[i].cudaData; d.value = values[i].cudaData;
            d.indices = (const int *)indices[i].cudaData;
            d.liveKeys = (const int *)live.cudaData + i;
            d.capacity = capacity; d.selectedCount = 2048; d.keyStride = stride;
        }
        FastllmCudaNaiveBatchDSADecode(q, sink, descriptors, heads, kvHeads, 192, 128, attentionScratch, actual);
    };
    Graph graph; graph.Capture(body);
    const void *outputPointer = actual.cudaData;
    double maximumL2 = 0;
    for (const auto &lengths : {std::vector<int>{1, 255, 2047, 8192}, {2, 256, 2048, 4096},
                                {0, 257, 2049, 4097}, {8192, 2048, 3, 0}, {255, 1, 2047, 8192}}) {
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
        graph.Run(); const auto got = Read<uint16_t>(actual);
        Require(actual.cudaData == outputPointer, "mixed-region graph output moved");
        for (int i = 0; i < rows; ++i) {
            if (!lengths[i]) {
                for (int j = 0; j < heads * 128; ++j)
                    Require(got[(size_t)i * heads * 128 + j] == 0, "inactive graph row was not zero");
                ++checks; continue;
            }
            Data query(BFLOAT16, {1, 1, heads * 192}), indexQuery(BFLOAT16, {1, 1, 16 * 128}),
                weights(BFLOAT16, {1, 1, 16}), selected, expected;
            query.FakeFrom(q, (size_t)i * heads * 192 * 2);
            indexQuery.FakeFrom(iq, (size_t)i * 16 * 128 * 2);
            weights.FakeFrom(iw, (size_t)i * 16 * 2);
            keys[i].Resize({1, lengths[i], stride}); values[i].Resize({1, lengths[i], kvHeads * 128});
            if (lengths[i] > 2048)
                FastllmCudaNaiveIndexer(indexQuery, weights, keys[i], 16, 128, lengths[i] - 1, 2048, true, selected);
            FastllmCudaNaiveAttention(query, keys[i], values[i], selected, sink, heads, kvHeads,
                192, 128, lengths[i] - 1, 0, expected);
            const auto ref = Read<uint16_t>(expected);
            double error = 0, norm = 0;
            for (size_t j = 0; j < ref.size(); ++j) {
                uint32_t aBits = uint32_t(got[(size_t)i * heads * 128 + j]) << 16, bBits = uint32_t(ref[j]) << 16;
                float a, b; std::memcpy(&a, &aBits, sizeof(a)); std::memcpy(&b, &bBits, sizeof(b));
                Require(std::isfinite(a) && std::abs(a - b) <= .004f + .02f * std::abs(b), "mixed-region attention element error");
                error += double(a - b) * (a - b); norm += double(b) * b;
            }
            const double l2 = std::sqrt(error / std::max(norm, 1e-30));
            Require(l2 < .01, "mixed-region attention L2 error");
            maximumL2 = std::max(maximumL2, l2); ++checks;
        }
    }
    std::printf("MIXED REGION GRAPH PASS max_l2=%.9g\n", maximumL2);
}

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
static void TestResidualNorm(int channels, int rows = 1, bool unalignedWeight = false, bool verifyPack = false) {
    Data hidden(BFLOAT16), branch(BFLOAT16), weight(FLOAT32), actual(BFLOAT16);
    Data reference(BFLOAT16), expected(BFLOAT16);
    Upload(hidden, {1, rows, channels}, 91);
    Upload(reference, {1, rows, channels}, 91);
    Upload(branch, {1, rows, channels}, 92);
    Data weightStorage(FLOAT32);
    if (unalignedWeight) {
        Upload(weightStorage, {channels + 1}, 93);
        weight.Resize({channels});
        weight.FakeFrom(weightStorage, sizeof(float));
    } else Upload(weight, {channels}, 93);
    const bool hopperBlock32 = verifyPack && FastllmCudaRuntimeArch() == 90 && rows > 64 && rows <= 128;
    if (hopperBlock32) {
        // Exercise a zero group and a nonzero group smaller than the standalone quantizer floor.
        float gains[64]{};
        std::fill(gains + 32, gains + 64, 1.e-6f);
        Cuda(cudaMemcpy(weight.cudaData, gains, sizeof(gains), cudaMemcpyHostToDevice));
    }
    Upload(expected, {1, rows, channels}, 94);
    auto original = Read<uint16_t>(hidden);
    PackedOutput packed;
    if (channels % 128 == 0 && (rows <= 16 || verifyPack)) packed.Init(rows, channels);
    auto body = [&]() {
        auto view = packed.view;
        FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, weight, 1e-5f, actual, &view);
        Require(view.values == (channels == 4096 ? packed.view.values : nullptr), "unexpected RMSNorm projection admission");
    };
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
        if (channels == 4096 && packed.view.values) packed.Check(actual, 0, hopperBlock32 ? 1.e-4f : 0.f);
        original = Read<uint16_t>(reference);
        ++checks;
    }
}
static void TestVerifyGraph(int window, int capacity, const std::vector<int> &prefixes, bool fp8, int rows = 8, int heads = 8, int scorePattern = 0) {
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
    if (scorePattern == 1) Cuda(cudaMemset(iw.cudaData, 0, iw.GetBytes()));
    if (scorePattern == 2) {
        // More than TopK equal scores exercise stable compaction across CTAs.
        auto repeated = Read<uint16_t>(key);
        for (int row = 4; row < storage; ++row)
            std::copy_n(repeated.data() + (row % 4) * kc, kc, repeated.data() + (size_t)row * kc);
        Cuda(cudaMemcpy(key.cudaData, repeated.data(), key.GetBytes(), cudaMemcpyHostToDevice));
    }
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
// Independent causal blocks, mixed lengths and descriptor chunking must agree
// with the original one-query Indexer, including output-view canaries.
static void TestBatchVerifyIndexer(int requests, bool fp8, int width = 0, bool uniform = false) {
    int total = 0;
    std::vector<int> offsets, widths, past;
    for (int i = 0; i < requests; ++i) {
        offsets.push_back(total);
        widths.push_back(width ? width : 1 + i % 8);
        past.push_back(i == 0 ? 32765 : i % 3 == 0 ? 127 : i % 3 == 1 ? 2045 : 4097);
        total += widths.back();
    }
    Data q(BFLOAT16), w(BFLOAT16), live;
    Upload(q, {1, total, 2048}, 871);
    Upload(w, {1, total, 16}, 872);
    if (uniform) FastllmCudaMemset0(w.cudaData, w.GetBytes());
    std::vector<Data> keys(requests, Data(BFLOAT16)), storage(requests, Data(INT32)), outputs(requests);
    std::vector<FastllmNaiveVerifyIndexerItem> items;
    for (int i = 0; i < requests; ++i) {
        Upload(keys[i], {1, past[i] + widths[i], 320}, 873 + i);
        storage[i].Resize({widths[i] * 2048 + 32}); storage[i].Allocate();
        std::fill((int *)storage[i].cpuData, (int *)storage[i].cpuData + storage[i].Count(0), 0x12345678);
        storage[i].ToDevice(DataDevice::CUDA, {0}, true);
        outputs[i].Resize({widths[i], 2048}); outputs[i].FakeFrom(storage[i], 0);
        items.push_back({&keys[i], &outputs[i], offsets[i], widths[i], past[i]});
    }
    FastllmNaiveDecodeScratch scratch;
    Require(FastllmCudaNaiveBatchVerifyIndexer(q, w, items, fp8, scratch, live), "batch verification Indexer unavailable");
    for (int i = 0; i < requests; ++i) {
        auto got = Read<int>(storage[i]);
        Require(std::all_of(got.begin() + widths[i] * 2048, got.end(),
            [](int v) { return v == 0x12345678; }), "batched Indexer output canary overwritten");
        for (int row = 0; row < widths[i]; ++row) {
            Data qr(BFLOAT16, {1, 1, 2048}), wr(BFLOAT16, {1, 1, 16});
            Data kr(BFLOAT16, {1, past[i] + row + 1, 320}), expected;
            qr.FakeFrom(q, (size_t)(offsets[i] + row) * 2048 * 2);
            wr.FakeFrom(w, (size_t)(offsets[i] + row) * 16 * 2);
            kr.FakeFrom(keys[i], 0);
            FastllmCudaNaiveIndexer(qr, wr, kr, 16, 128, past[i] + row, 2048, fp8, expected);
            auto ref = Read<int>(expected);
            Require(std::equal(ref.begin(), ref.end(), got.begin() + row * 2048),
                "batched verification Indexer differs from independent causal row");
            ++checks;
        }
    }
    std::printf("BATCH VERIFY INDEXER PASS requests=%d rows=%d fp8_query=%d\n", requests, total, fp8);
}

// Normalizing a chunk must match the same rows in a full batch, including
// in-place output. The large batch exercises the general reduction kernel.
static void TestRMSNormChunks() {
    Data input(BFLOAT16), weight(FLOAT32), reference(BFLOAT16);
    Upload(input, {257, 4096}, 711);
    Upload(weight, {4096}, 712);
    Upload(reference, input.dims, 0);
    Require(FastllmCudaKimiK3RMSNorm(input, weight, reference, 1e-5f), "batch norm");
    auto expected = Read<uint16_t>(reference);
    for (int rows : {1, 2, 7, 8, 31, 128}) {
        Data chunk(BFLOAT16), output(BFLOAT16);
        Upload(chunk, {rows, 4096}, 711);
        Upload(output, chunk.dims, 0);
        Require(FastllmCudaKimiK3RMSNorm(chunk, weight, output, 1e-5f), "chunk norm");
        auto actual = Read<uint16_t>(output);
        Require(std::equal(actual.begin(), actual.end(), expected.begin()), "chunk norm differs");
        Require(FastllmCudaKimiK3RMSNorm(chunk, weight, chunk, 1e-5f), "in-place chunk norm");
        Require(Read<uint16_t>(chunk) == actual, "in-place norm differs");
        ++checks;
    }
}

static void TestDraftAttention(int dim, int rows, int window, bool shortAttention,
                               int heads = 8, int kvHeads = 2) {
    const int storage = window + rows;
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
    PackedOutput packed;
    if (heads == 64 && batch == 1) packed.Init(batch, heads * 128);
    Graph graph;
    graph.Capture([&]() {
        auto view = packed.view;
        FastllmCudaNaiveBatchSWADecode(query, sink, rows, live, heads, kvHeads, actual, &view);
        if (packed.view.values) Require(view.values, "SWA producer rejected eligible view");
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
        if (packed.view.values) packed.Check(actual);
        ++checks;
    }
    if (!packed.view.values) return;
    for (Data *cache : {&keys[0], &values[0]}) {
        std::vector<uint8_t> bytes(cache->Count(0));
        for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = uint8_t((i * 37 % 73) | ((i % 11 < 4) ? 128 : 0));
        Cuda(cudaMemcpy(cache->cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
    rows.front().fp8 = true;
    Graph fp8Graph;
    fp8Graph.Capture([&] {
        auto view = packed.view;
        FastllmCudaNaiveBatchSWADecode(query, sink, rows, live, heads, kvHeads, actual, &view);
        Require(view.values, "FP8 ring did not prepare projection input");
    });
    for (int length : {0, 1, 127, 128, 707}) {
        Cuda(cudaMemcpy(live.cudaData, &length, sizeof(length), cudaMemcpyHostToDevice));
        FastllmCudaNaiveBatchSWADecode(query, sink, rows, live, heads, kvHeads, expected);
        fp8Graph.Run();
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "FP8 SWA BF16 output differs");
        packed.Check(actual);
        ++checks;
    }
}

// Replay one verification graph with shared prefixes, independently changing
// lengths, a trailing singleton, and empty rows. Compare against the same BF16 attention.
static void TestVerifySwaPacking() {
    constexpr int rows = 129, heads = 16, kvHeads = 2;
    Data q(BFLOAT16), sink(FLOAT32), live(INT32), key(INT32), value(INT32), reference, actual;
    Upload(q, {1, rows, heads * 192}, 621);
    Upload(sink, {heads}, 622);
    Upload(live, {rows}, 623);
    auto cache = [](Data &d, int width) {
        Upload(d, {132, kvHeads * width / 4}, width);
        std::vector<uint8_t> bytes(d.GetBytes());
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 97 % 95) + 7;
        Cuda(cudaMemcpy(d.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    };
    cache(key, 192);
    cache(value, 128);
    std::vector<FastllmNaiveSWARow> desc(rows);
    for (int i = 0; i < rows; ++i) {
        desc[i].key = key.cudaData; desc[i].value = value.cudaData;
        desc[i].fp8 = true; desc[i].verifyOffset = i % 4;
    }
    std::vector<int> lengths(rows);
    PackedOutput packed;
    packed.Init(rows, heads * 128);
    auto bind = [&](int mode) {
        for (int i = 0; i < rows; ++i)
            lengths[i] = mode == 0 ? 128 + i % 4 : mode == 1 && i / 4 % 3 ? 37 + i % 2 : 0;
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), live.GetBytes(), cudaMemcpyHostToDevice));
    };
    bind(0);
    Graph graph;
    graph.Capture([&] {
        auto view = packed.view;
        FastllmCudaNaiveBatchSWADecode(q, sink, desc, live, heads, kvHeads, actual, &view);
        Require(view.values, "FP8 verification SWA did not produce packed output");
    });
    for (int mode : {0, 1, 2, 0}) {
        bind(mode);
        graph.Run();
        FastllmCudaNaiveBatchSWADecode(q, sink, desc, live, heads, kvHeads, reference);
        Require(Read<uint16_t>(reference) == Read<uint16_t>(actual), "packed verification SWA changed BF16");
        packed.Check(actual);
        ++checks;
    }
}

static void TestPaddedAttention(int batch, int heads, int dim, int vd, bool sparse, bool fp8, bool online = false) {
    const int kvHeads = online ? heads / 16 : 1;
    const int capacity = online ? 2049 : sparse ? 2048 : 128;
    const int pageLen = online ? 128 : capacity;
    const int pageCount = (capacity + pageLen - 1) / pageLen;
    const int storedRows = pageCount * pageLen;
    Data query(BFLOAT16), key(BFLOAT16), value(BFLOAT16), sink(FLOAT32), live(INT32), table(INT32),
        pages(INT32), indices(INT32), maps(INT32), actual;
    Upload(query, {1, batch, heads * dim}, 301);
    Upload(key, {1, storedRows, kvHeads * dim}, 302); Upload(value, {1, storedRows, kvHeads * vd}, 303);
    Upload(sink, {heads}, 304); Upload(live, {batch}, 305);
    if (fp8) {
        // Distinct finite E4M3 values expose token and GQA head addressing errors.
        unsigned seed = 302;
        for (Data *cache : {&key, &value}) {
            std::vector<uint8_t> payload(cache->Count(0));
            for (auto &byte : payload) {
                seed = 1664525u * seed + 1013904223u;
                byte = uint8_t((seed >> 24) & 0x80) | uint8_t((seed >> 16) % 57);
            }
            Cuda(cudaMemcpy(cache->cudaData, payload.data(), payload.size(), cudaMemcpyHostToDevice));
        }
    }
    Upload(table, {int(batch * 4 * sizeof(void *) / sizeof(int))}, 306);
    Upload(pages, {pageCount}, 307);
    std::vector<int> physicalPages(pageCount);
    for (int i = 0; i < pageCount; ++i) physicalPages[i] = pageCount - 1 - i;
    Cuda(cudaMemcpy(pages.cudaData, physicalPages.data(), pages.GetBytes(), cudaMemcpyHostToDevice));
    Upload(indices, {2048}, 308);
    std::vector<int> selected(2048);
    for (int i = 0; i < 2048; ++i) selected[i] = online ? (i * 37 + 13) % capacity : i;
    Cuda(cudaMemcpy(indices.cudaData, selected.data(), indices.GetBytes(), cudaMemcpyHostToDevice));
    if (!fp8 && heads == 8 && dim == 192 && vd == 128 && !sparse && FastllmCudaNaiveDSAMapsSupported()) {
        Upload(maps, {FastllmNaiveDSAMapBytes / int(sizeof(int))}, 309);
        alignas(128) unsigned char encoded[FastllmNaiveDSAMapBytes];
        Require(FastllmCudaNaiveEncodeDSAMaps(encoded, key.cudaData, value.cudaData, capacity, dim), "encode padded maps");
        Cuda(cudaMemcpy(maps.cudaData, encoded, sizeof(encoded), cudaMemcpyHostToDevice));
    }
    std::vector<FastllmNaiveDSARow> rows(batch);
    std::vector<void *> bindings(batch * 4);
    std::vector<int> lengths(batch, capacity);
    for (int b = 0; b < batch; ++b) {
        auto &r = rows[b];
        r.key = key.cudaData; r.value = value.cudaData;
        r.indices = sparse ? (const int *)indices.cudaData : nullptr;
        r.selectedCount = sparse ? 2048 : 0; r.capacity = capacity; r.keyStride = kvHeads * dim;
        r.liveKeys = (const int *)live.cudaData + b;
        r.cache = (const void *const *)table.cudaData + b * 4;
        r.tensorMaps = maps.cudaData; r.pageLen = pageLen; r.poolTokens = storedRows; r.fp8 = fp8;
    }
    auto bind = [&](int active) {
        for (int b = 0; b < batch; ++b) {
            lengths[b] = b < active ? capacity - (online ? b % 4 : 0) : 0;
            bindings[b * 4] = b < active ? key.cudaData : nullptr;
            bindings[b * 4 + 1] = b < active ? value.cudaData : nullptr;
            bindings[b * 4 + 2] = bindings[b * 4 + 3] = b < active ? pages.cudaData : nullptr;
        }
        Cuda(cudaMemcpy(table.cudaData, bindings.data(), bindings.size() * sizeof(void *), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    };
    bind(batch);
    FastllmNaiveDecodeScratch scratch;
    PackedOutput packed;
    if (online) packed.Init(batch, heads * vd);
    auto runOnline = [&] {
        auto view = packed.view;
        const bool ok = FastllmCudaNaiveOnlineDSADecode(query, sink, rows, heads, kvHeads, dim, vd, scratch, actual, &view);
        Require(view.values == (ok ? packed.view.values : nullptr), "unexpected DSA projection admission");
        return ok;
    };
    if (online && !runOnline()) return; // Optional device/image.
    Graph graph;
    graph.Capture([&] {
        if (online) Require(runOnline(), "online DSA unavailable during capture");
        else FastllmCudaNaiveBatchDSADecode(query, sink, rows, heads, kvHeads, dim, vd, scratch, actual);
    });
    graph.Run(); auto reference = Read<uint16_t>(actual);
    if (online) {
        Data expected; FastllmNaiveDecodeScratch oldScratch;
        FastllmCudaNaiveBatchDSADecode(query, sink, rows, heads, kvHeads, dim, vd, oldScratch, expected);
        auto old = Read<uint16_t>(expected);
        double error = 0, norm = 0;
        for (size_t i = 0; i < old.size(); ++i) {
            uint32_t a = uint32_t(reference[i]) << 16, b = uint32_t(old[i]) << 16;
            float got, want;
            std::memcpy(&got, &a, sizeof(got)); std::memcpy(&want, &b, sizeof(want));
            error += double(got-want)*(got-want); norm += double(want)*want;
        }
        Require(std::sqrt(error/std::max(norm, 1e-20)) < .005, "online DSA BF16 reference");
    }
    for (int active : {batch - 1, 1, 0, batch / 2, batch}) {
        bind(active); graph.Run(); auto result = Read<uint16_t>(actual);
        if (packed.view.values) packed.Check(actual);
        for (size_t i = 0; i < result.size(); ++i)
            Require(result[i] == (i < (size_t)active * heads * vd ? reference[i] : 0),
                    "padding changed active attention or left inactive output nonzero");
        ++checks;
    }
    if (online && packed.view.values) {
        // A live row can also have an exactly zero K32 block.
        Cuda(cudaMemset(value.cudaData, 0, value.GetBytes()));
        graph.Run();
        packed.Check(actual);
        ++checks;
    }
}

static void TestRouterBatch(int batch, int pattern, int experts = 256) {
    Data input(BFLOAT16), weight(FLOAT32), bias(FLOAT32), actual, expected(FLOAT32);
    Data indices(INT32), scores(FLOAT32), referenceIndices(INT32), referenceScores(FLOAT32);
    Upload(input, {1, batch, 4096}, 181); Upload(weight, {experts, 4096}, 182);
    Upload(bias, {experts}, 183); Upload(expected, {1, batch, experts}, 184);
    Upload(indices, {batch, 8}, 185); Upload(scores, {batch, 8}, 186);
    Upload(referenceIndices, {batch, 8}, 187); Upload(referenceScores, {batch, 8}, 188);
    if (pattern == 1) {
        Cuda(cudaMemset(weight.cudaData, 0, weight.GetBytes()));
        Cuda(cudaMemset(bias.cudaData, 0, bias.GetBytes()));
    }
    if (pattern == 2) {
        auto values = Read<float>(weight);
        for (auto &value : values) value *= .01f;
        Cuda(cudaMemcpy(weight.cudaData, values.data(), weight.GetBytes(), cudaMemcpyHostToDevice));
    }
    int previous = FastllmCudaGetLinearExactBatchThreshold();
    FastllmCudaSetLinearExactBatchThreshold(batch + 1);
    Graph graph;
    graph.Capture([&]() {
        Require(FastllmCudaNaiveRouterSigmoid(input, weight, actual), "batch router unsupported");
        Require(FastllmCudaSelectExpert(actual, &bias, indices, scores, 8, true, 1.7f), "batch experts unsupported");
    });
    double maxRouterL2 = 0; float maxRouterAbsolute = 0;
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) Cuda(cudaMemset(input.cudaData, 0, input.GetBytes()));
        graph.Run();
        for (int b = 0; b < batch; ++b) {
            Data row, result;
            row.Resize({1, 1, 4096}); row.FakeFrom(input, (size_t)b * 4096 * 2);
            result.Resize({1, 1, experts}); result.FakeFrom(expected, (size_t)b * experts * sizeof(float));
            Require(FastllmCudaNaiveRouterSigmoid(row, weight, result), "reference router unsupported");
        }
        const bool tiled = batch > 16 && FastllmCudaRuntimeArch() == 103;
        auto close = [&](const Data &a, const Data &b, const char *what) {
            auto x=Read<float>(const_cast<Data&>(a)), y=Read<float>(const_cast<Data&>(b));
            if (!tiled) { Require(x==y,what); return; }
            double se=0,ss=0; float maximum=0;
            for(size_t i=0;i<x.size();++i) {
                Require(std::isfinite(x[i]) && std::abs(x[i]-y[i])<1e-5f,what);
                double e=double(x[i])-y[i]; se+=e*e; ss+=double(y[i])*y[i];
                maximum=std::max(maximum,float(std::abs(e)));
            }
            const double relative = std::sqrt(se/std::max(ss,1e-30));
            maxRouterL2=std::max(maxRouterL2,relative);
            maxRouterAbsolute=std::max(maxRouterAbsolute,maximum);
            Require(relative<1e-6,what);
        };
        close(actual,expected,"batch router probability error");
        Require(FastllmCudaSelectExpert(expected, &bias, referenceIndices, referenceScores, 8, true, 1.7f), "reference experts unsupported");
        Require(Read<int>(indices) == Read<int>(referenceIndices), "batch router expert order changed");
        close(scores,referenceScores,"batch router expert score error");
        ++checks;
    }
    if(batch>16 && FastllmCudaRuntimeArch()==103)
        printf("ROUTER rows=%d pattern=%d max_absolute=%.9g relative_l2=%.9g\n",
            batch,pattern,maxRouterAbsolute,maxRouterL2);
    FastllmCudaSetLinearExactBatchThreshold(previous);
}

static void TestVerifyRouter() {
    constexpr int rows = 129, experts = 256, hidden = 4096, top = 8;
    Data x(BFLOAT16), w(FLOAT32), bias(FLOAT32), scratch(FLOAT32), ids(INT32), weights(FLOAT32), ref(FLOAT32);
    Upload(x, {rows, hidden}, 201);
    Upload(w, {experts, hidden}, 202);
    Upload(bias, {experts}, 203);
    scratch.Resize({3 * experts * hidden / 2 + 3 * rows * experts});
    scratch.ToDevice(DataDevice::CUDA, {0}, false);
    scratch.Allocate(false);
    Data smallScratch(FLOAT32);
    Require(!FastllmCudaNaiveVerifyRouter(x, w, bias, smallScratch, ids, weights, top, true, 1.f),
            "verify router accepted missing scratch");
    cudaDeviceProp prop;
    Cuda(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || (prop.minor != 0 && prop.minor != 3) || CUDART_VERSION < 12080) return;
    Data small;
    small.Resize({127, hidden});
    small.FakeFrom(x, 0);
    Require(!FastllmCudaNaiveVerifyRouter(small, w, bias, scratch, ids, weights, top, true, 1.f),
            "verify router accepted small batch");
    Graph graph;
    graph.Capture([&]() {
        Require(FastllmCudaNaiveVerifyRouter(x, w, bias, scratch, ids, weights, top, true, 1.7f),
                "large verify router unsupported");
    });
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            Cuda(cudaMemset(x.cudaData, 0, x.GetBytes()));
            Cuda(cudaMemset(bias.cudaData, 0, bias.GetBytes()));
        }
        graph.Run();
        const auto actualIds = Read<int>(ids);
        const auto actualWeights = Read<float>(weights);
        const auto correction = Read<float>(bias);
        int threshold = FastllmCudaGetLinearExactBatchThreshold();
        FastllmCudaSetLinearExactBatchThreshold(rows + 1);
        Require(FastllmCudaNaiveRouterSigmoid(x, w, ref), "reference router failed");
        FastllmCudaSetLinearExactBatchThreshold(threshold);
        const auto probabilities = Read<float>(ref);
        for (int r = 0; r < rows; ++r) {
            const float *pr = probabilities.data() + r * experts;
            std::vector<int> order(experts);
            for (int e = 0; e < experts; ++e) order[e] = e;
            std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
                float left = pr[a] + correction[a], right = pr[b] + correction[b];
                return left == right ? a < b : left > right;
            });
            float sum = 0.f;
            for (int k = 0; k < top; ++k) sum += pr[order[k]];
            for (int k = 0; k < top; ++k) {
                Require(actualIds[r * top + k] == order[k], "verify router stable top-8 mismatch");
                Require(std::abs(actualWeights[r * top + k] - 1.7f * pr[order[k]] / sum) < 1.e-5f,
                        "verify router mixture weight mismatch");
            }
        }
        ++checks;
    }
}

static void TestIndexerKeyNorm(int rows) {
    Data input(BFLOAT16), weight(FLOAT32), bias(FLOAT32), reference, expected;
    Upload(input, {1, rows, 128}, 917);
    Upload(weight, {128}, 918); Upload(bias, {128}, 919);
    auto original = Read<uint16_t>(input);
    Graph graph;
    graph.Capture([&] {
        Require(FastllmCudaNaiveIndexerKeyNorm(input, weight, bias), "Indexer norm layout rejected");
    });
    for (int pattern = 0; pattern < 4; ++pattern) {
        auto values = original;
        for (size_t i = 0; i < values.size(); ++i) {
            if (pattern == 1) values[i] = Float32ToBFloat16RNEBits(2.f);
            if (pattern == 2) values[i] = Float32ToBFloat16RNEBits((i % 2 ? -1.f : 1.f) * (i % 7 ? .001f : 100.f));
            if (pattern == 3) values[i] = Float32ToBFloat16RNEBits(1.f + (i % 5) / 128.f);
        }
        Cuda(cudaMemcpy(input.cudaData, values.data(), values.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        ToDataType(input, reference, FLOAT32);
        LayerNorm(reference, weight, bias, -1, reference);
        ToDataType(reference, expected, BFLOAT16);
        graph.Run();
        Require(Read<uint16_t>(input) == Read<uint16_t>(expected), "Indexer BF16 norm differs bitwise");
        ++checks;
    }
    auto before = Read<uint16_t>(input);
    input.strides[1]++;
    Require(!FastllmCudaNaiveIndexerKeyNorm(input, weight, bias), "Indexer norm accepted invalid stride");
    input.strides[1]--;
    Require(Read<uint16_t>(input) == before, "unsupported Indexer norm changed input");
}

static void TestEightRowSelection() {
    constexpr int rows = 8, experts = 256, top = 8;
    Data logits(FLOAT32), bias(FLOAT32), ids(INT32), scores(FLOAT32), refIds(INT32), refScores(FLOAT32);
    Upload(logits, {rows, experts}, 215); Upload(bias, {experts}, 216);
    Upload(ids, {rows, top}, 217); Upload(scores, {rows, top}, 218);
    Upload(refIds, {rows, top}, 219); Upload(refScores, {rows, top}, 220);
    for (bool normalize : {false, true}) {
        Graph graph;
        graph.Capture([&] {
            Require(FastllmCudaSelectExpert(logits, &bias, ids, scores, top, normalize, 1.7f), "batched selection");
        });
        std::vector<float> values(rows * experts), offsets(experts);
        for (int pattern = 0; pattern < 5; ++pattern) {
            for (int e = 0; e < experts; ++e) offsets[e] = pattern == 0 ? (e % 5) * .001f : 0.f;
            for (int i = 0; i < rows * experts; ++i) {
                const int e = i % experts;
                values[i] = pattern == 0 ? std::sin(i * .185f) :
                    pattern == 1 ? .5f : pattern == 2 ? (i % 17) * .025f :
                    pattern == 3 ? (e < 7 ? float(8 - e) : .25f) :
                    (e == 0 ? std::numeric_limits<float>::quiet_NaN() : float(e) / experts);
            }
            Cuda(cudaMemcpy(logits.cudaData, values.data(), logits.GetBytes(), cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(bias.cudaData, offsets.data(), bias.GetBytes(), cudaMemcpyHostToDevice));
            graph.Run();
            for (int row = 0; row < rows; ++row) {
                Data x(FLOAT32, {1, experts}), idx(INT32, {1, top}), mix(FLOAT32, {1, top});
                x.FakeFrom(logits, row * experts * sizeof(float));
                idx.FakeFrom(refIds, row * top * sizeof(int));
                mix.FakeFrom(refScores, row * top * sizeof(float));
                Require(FastllmCudaSelectExpert(x, &bias, idx, mix, top, normalize, 1.7f), "reference selection");
            }
            Require(Read<int>(ids) == Read<int>(refIds), "batched selection changed an expert");
            Require(Read<float>(scores) == Read<float>(refScores), "batched selection changed a weight");
            ++checks;
        }
    }
}

static void TestRouters() {
    TestEightRowSelection();
    for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 23, 24, 31, 32, 33, 48, 64, 65, 128})
        for (int pattern = 0; pattern < 3; ++pattern) TestRouterBatch(batch, pattern);
}

int main(int argc, char **argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        FastllmCudaSetDevice(0);
        if (argc == 2 && std::strcmp(argv[1], "--router-batch") == 0) {
            TestRouters();
            std::puts("ROUTER BATCH PASS");
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--verify-qkv-pack") == 0) {
            if (FastllmCudaRuntimeArch() != 90) return 77;
            for (int rows : {65, 128}) TestResidualNorm(4096, rows, false, true);
            std::printf("Verification QKV pack PASS (%d graph checks)\n", checks);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--projection-pack") == 0) {
            TestResidualNorm(4096, 1);
            TestResidualNorm(4096, 8);
            TestResidualNorm(2048, 2); // BF16 fallback clears the requested side output.
            TestSwaBatch(1, 64, 8);
            if (FastllmCudaRuntimeArch() == 103) {
                TestVerifySwaPacking();
                TestPrefillProjection(true);
                if (FastllmCudaHasBlackwellIndexerAttention()) TestPrefillProjection(false);
                TestResidualNorm(4096, 128, false, true);
                TestResidualNorm(4096, 2049, false, true);
                TestPaddedAttention(257, 32, 192, 128, true, true, true);
            }
            for (int rows : {1, 4}) TestPaddedAttention(rows, 64, 192, 128, true, true, true);
            std::printf("Projection pack PASS (%d graph checks)\n", checks);
            return 0;
        }
        for (int requests : {2, 70}) for (bool fp8 : {false, true}) TestBatchVerifyIndexer(requests, fp8);
        for (int requests : {1, 3, 128}) for (int width : {3, 4, 6, 8})
            TestBatchVerifyIndexer(requests, true, width);
        TestBatchVerifyIndexer(128, true, 6, true);
        for (int rows : {1, 2, 3, 4, 8, 33}) TestIndexerKeyNorm(rows);
        if (!FastllmCudaNaiveDecodeGraphSupported())
            return 77;
        TestMixedRegionAttention();
        for (int batch : {1, 4, 8, 32, 64})
            for (int heads : {16, 64}) TestPaddedAttention(batch, heads, 192, 128, true, true, true);
        for (int batch : {8, 40})
        for (bool sparse : {false, true})
        for (bool fp8 : {false, true}) {
            TestPaddedAttention(batch, 8, 192, 128, sparse, fp8);
            TestPaddedAttention(batch, 2, 64, 256, sparse, fp8);
        }
        for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 31, 32, 33, 48, 64, 128, 129, 255, 256})
            for (int pattern = 0; pattern < 3; ++pattern) TestRouterBatch(batch, pattern);
        TestRouterBatch(33, 2, 255);
        TestVerifyRouter();
        TestRouters();
        for (int batch : {1, 2, 3, 4, 17})
            for (int heads : {4, 8, 16})
                TestSwaBatch(batch, heads, heads == 16 ? 2 : 1);
        for (int heads : {4, 8, 16})
            for (bool fused : {false, true}) TestSwaRing(heads, heads == 16 ? 2 : 1, fused);
        for (int ch : {1, 127, 256, 1024, 2048, 4096, 5120})
            TestResidualNorm(ch);
        for (int ch : {127, 256, 4096})
            TestResidualNorm(ch, 8);
        for (int rows : {127, 128, 513, 4096, 32768})
            TestResidualNorm(4096, rows);
        TestResidualNorm(4096, 129, true);
        for (int w : {0, 8, 128})
            TestCache(w);
        // Dense decode uses replay-time length, including decreases and the
        // 64-key staging tail; keep the optimized path bitwise equal to eager.
        for (int heads : {1, 8, 16, 32})
            for (bool sink : {false, true})
                TestAttention(heads, 192, 128, 0, 896,
                              {1, 63, 64, 65, 707, 748, 895, 896, 257}, true, sink);
        TestAttention(8, 192, 128, 0, 2048, {257, 1024, 2047, 2048, 513}, true);
        // DP attention retains all heads on one device. Exercise the wide
        // SWA output tile across live-length growth and rollback in one graph.
        for (bool sink : {false, true})
            TestAttention(64, 192, 128, 128, 256, {1, 63, 127, 128, 129, 256, 3}, false, sink);
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
        // Multiple queries must keep separate causal lengths even when the
        // shared cache contains all candidates, including after length rollback.
        for (int rows : {2, 7, 8})
            for (int capacity : {512, 1024})
                TestVerifyGraph(0, capacity, {1, 255, capacity - rows, 63}, true, rows);

        for (int w : {8, 128})
            TestVerifyGraph(w, 256, {3, 120, 127, 128, 32768, 7}, true);
        for (int rows : {1, 3, 8, 11})
            TestVerifyGraph(128, 256, {3, 120, 127, 128, 32768, 7}, true, rows, 64);
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
        for (bool fp8 : {false, true})
            for (int pattern : {1, 2})
                for (int capacity : {4096, 32776})
                    TestVerifyGraph(0, capacity, {2048, capacity - 8, 2050}, fp8, 8, 8, pattern);
        for (int dim : {32, 128, 256}) for (int rows : {2, 7, 8})
            for (bool shortAttention : {false, true})
                TestDraftAttention(dim, rows, 1024, shortAttention);
        TestDraftAttention(128, 7, 1024, false, 32, 4);
        TestDraftAttention(128, 31, 1024, false, 32, 4);
        TestRMSNormChunks();
        printf("DECODE GRAPH PASS checks=%d\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
