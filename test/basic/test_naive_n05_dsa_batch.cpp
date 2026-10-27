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
#include <memory>
#include <numeric>
#include <random>
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
// Independent K/V permutations ensure logical positions cannot accidentally
// be used as physical offsets. Payloads are populated without the write kernel.
struct PagedTensor {
    PagedCacheManager pool;
    Data table{INT32}, cache{BFLOAT16};
    explicit PagedTensor(const Data &dense, int seed, int pageLen = 128) {
        const int length = dense.dims[1], width = dense.dims[2];
        const int pages = (length + pageLen - 1) / pageLen;
        pool.dataType = BFLOAT16; pool.pageLen = pageLen;
        pool.Resize({pages, pageLen, 1, width});
        pool.ToDevice(DataDevice::CUDA, {0}, false); pool.Allocate(); pool.SetMaxPages(pages);
        std::vector<int> ids(pages); std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
        table.Resize({pages}); table.Allocate();
        std::memcpy(table.cpuData, ids.data(), pages * sizeof(int));
        table.ToDevice(DataDevice::CUDA, {0}, true);
        for (int row = 0; row < length; row += pageLen)
            Cuda(cudaMemcpy((uint8_t *)pool.cudaData + (size_t)ids[row / pageLen] * pageLen * width * 2,
                (const uint8_t *)dense.cudaData + (size_t)row * width * 2,
                (size_t)std::min(pageLen, length - row) * width * 2, cudaMemcpyDeviceToDevice));
        cache.FakeFrom(table, 0); cache.dataType = BFLOAT16; cache.Resize(dense.dims);
        cache.isPagedKVCache = true; cache.pagedKVCacheData = &pool;
        cache.pageLen = pageLen; cache.pageIndex = ids;
        cache.lastPageLen = (length - 1) % pageLen + 1;
        cache.expansionDims = {1, pages * pageLen, width};
    }
};
static void Test(int batch, int heads, int kvHeads, int dim, int vd, int capacity, bool sparse, bool withSink,
                 bool unalignedKeys = false, bool mixed = false, bool paged = false) {
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
    std::vector<std::unique_ptr<PagedTensor>> pagedKeys, pagedValues;
    for (int b = 0; b < batch; ++b) {
        Upload(keys[b], {1, capacity, stride}, 7 + b);
        Upload(values[b], {1, capacity, kvHeads * vd}, 17 + b);
        hostKeys[b] = Read<uint16_t>(keys[b]);
        hostValues[b] = Read<uint16_t>(values[b]);
        auto &r = rows[b];
        r.key = keys[b].cudaData;
        r.value = values[b].cudaData;
        if (paged) {
            pagedKeys.emplace_back(new PagedTensor(keys[b], b + 11));
            pagedValues.emplace_back(new PagedTensor(values[b], b + 79));
            r.key = pagedKeys.back()->pool.cudaData; r.value = pagedValues.back()->pool.cudaData;
            r.keyPages = (const int *)pagedKeys.back()->table.cudaData;
            r.valuePages = (const int *)pagedValues.back()->table.cudaData;
            r.pageLen = 128; r.poolTokens = pagedKeys.back()->pool.maxPages * 128;
        }
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
        if (paged) {
            auto denseRows = rows;
            for (int b = 0; b < batch; ++b) {
                denseRows[b].key = keys[b].cudaData; denseRows[b].value = values[b].cudaData;
                denseRows[b].keyPages = denseRows[b].valuePages = nullptr;
                denseRows[b].pageLen = denseRows[b].poolTokens = 0;
            }
            Data reference; FastllmNaiveDecodeScratch referenceScratch;
            FastllmCudaNaiveBatchDSADecode(q, sink, denseRows, heads, kvHeads, dim, vd, referenceScratch, reference);
            Require(actual == Read<uint16_t>(reference), "paging changed batch DSA arithmetic");
        }
        int different = 0;
        double oldMaxError = 0;
        FastllmNaiveDecodeScratch oldScratch;
        // Bulk attention has its own rounding path. Paged bulk is compared
        // bit-for-bit with the same dense batch kernel above and FP64 below.
        for (int b = 0; b < batch && (!paged || batch < 128); ++b) {
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
        if (capacity > 256 && (!paged || batch < 128))
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
// Replace and reorder KV behind one captured graph. The original addresses
// remain valid but contain different values, so stale kernel parameters cannot
// pass this test accidentally.
static void TestRebinding(int batch, int capacity, bool sparse, bool generic = false) {
    const int heads = 8, kvHeads = 1, dim = generic ? 128 : 192, vd = generic ? 256 : 128, stride = dim + 128;
    Data q(BFLOAT16), sink(FLOAT32), live(INT32), table(INT32), maps(INT32), actual, expected;
    const bool useMaps = !generic && FastllmCudaNaiveDSAMapsSupported();
    std::vector<unsigned char> hostMaps(batch * FastllmNaiveDSAMapBytes);
    maps.Resize({(int)hostMaps.size() / (int)sizeof(int)});
    maps.ToDevice(DataDevice::CUDA, {0}, false); maps.Allocate();
    Upload(q, {1, batch, heads * dim}, 73);
    Upload(sink, {heads}, 11);
    live.Resize({batch}); live.Allocate(); live.ToDevice(DataDevice::CUDA, {0}, false); live.Allocate();
    table.Resize({batch * 2 * (int)(sizeof(void *) / sizeof(int))});
    table.ToDevice(DataDevice::CUDA, {0}, false); table.Allocate();
    std::vector<Data> keys(batch * 2, Data(BFLOAT16)), values(batch * 2, Data(BFLOAT16));
    std::vector<Data> indices(batch, Data(INT32));
    std::vector<void *> pointers(batch * 2);
    std::vector<FastllmNaiveDSARow> rows(batch), direct(batch);
    std::vector<int> lengths(batch, capacity);
    for (int i = 0; i < batch * 2; ++i) {
        Upload(keys[i], {1, capacity, stride}, 131 + i);
        Upload(values[i], {1, capacity, vd}, 517 + i);
    }
    for (int i = 0; i < batch; ++i) {
        auto &r = rows[i];
        r.key = keys[i].cudaData; r.value = values[i].cudaData;
        r.liveKeys = (const int *)live.cudaData + i;
        r.capacity = capacity; r.keyStride = stride;
        r.cache = (const void *const *)table.cudaData + i * 2;
        if (useMaps) r.tensorMaps = (const unsigned char *)maps.cudaData + i * FastllmNaiveDSAMapBytes;
        pointers[i * 2] = keys[i].cudaData; pointers[i * 2 + 1] = values[i].cudaData;
        if (sparse) {
            indices[i].Resize({1, 2048}); indices[i].Allocate();
            for (int j = 0; j < 2048; ++j) ((int *)indices[i].cpuData)[j] = (j * 7 + i * 13) % capacity;
            indices[i].ToDevice(DataDevice::CUDA, {0}, true);
            r.indices = (const int *)indices[i].cudaData; r.selectedCount = 2048;
        }
    }
    auto updateMaps = [&] {
        if (!useMaps) return;
        for (int i = 0; i < batch; ++i)
            Require(FastllmCudaNaiveEncodeDSAMaps(hostMaps.data() + i * FastllmNaiveDSAMapBytes,
                pointers[i * 2], pointers[i * 2 + 1], capacity, stride), "encode tensor maps");
        Cuda(cudaMemcpy(maps.cudaData, hostMaps.data(), hostMaps.size(), cudaMemcpyHostToDevice));
    };
    updateMaps();
    Cuda(cudaMemcpy(table.cudaData, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
    FastllmNaiveDecodeScratch scratch, referenceScratch;
    Graph graph;
    graph.Capture([&] { FastllmCudaNaiveBatchDSADecode(q, sink, rows, heads, kvHeads, dim, vd, scratch, actual); });
    for (int pass = 0; pass < 3; ++pass) {
        for (int i = 0; i < batch; ++i) {
            int source = pass == 0 ? batch - i - 1 : batch + (i + pass) % batch;
            pointers[i * 2] = keys[source].cudaData; pointers[i * 2 + 1] = values[source].cudaData;
            lengths[i] = pass == 1 ? 3 : capacity - i % 5;
            direct[i] = rows[i]; direct[i].cache = nullptr; direct[i].tensorMaps = nullptr;
            direct[i].key = pointers[i * 2]; direct[i].value = pointers[i * 2 + 1];
        }
        Cuda(cudaMemcpy(table.cudaData, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        updateMaps();
        graph.Run();
        FastllmCudaNaiveBatchDSADecode(q, sink, direct, heads, kvHeads, dim, vd, referenceScratch, expected);
        auto a = Read<uint16_t>(actual), b = Read<uint16_t>(expected);
        double error = 0, norm = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double delta = std::abs((double)Float(a[i]) - Float(b[i]));
            Require(delta <= 0.009 * std::abs(Float(b[i])) + 2e-6, "indirect DSA exceeds one BF16 ULP");
            error += delta * delta; norm += (double)Float(b[i]) * Float(b[i]);
        }
        Require(std::sqrt(error / std::max(norm, 1e-30)) <= .003, "indirect DSA relative error");
        ++checks;
    }
    std::printf("KV REBIND PASS batch=%d capacity=%d sparse=%d generic=%d\n", batch, capacity, sparse, generic);
}

static void TestPagedPrefill(int length, int queries = 3, int heads = 8, int pageLen = 128) {
    const int width = 320, dim = 192, vd = 128;
    Data k(BFLOAT16), v(BFLOAT16), q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), sink;
    Upload(k, {1, length, width}, 191); Upload(v, {1, length, vd}, 707);
    Upload(q, {1, queries, heads * dim}, 777); Upload(iq, {1, queries, 16 * 128}, 90);
    Upload(iw, {1, queries, 16}, 119);
    PagedTensor pk(k, 173, pageLen), pv(v, 301, pageLen);
    Data denseIndex, pagedIndex, expected, actual;
    FastllmCudaNaiveIndexer(iq, iw, k, 16, 128, length - queries, 2048, true, denseIndex);
    FastllmCudaNaiveIndexer(iq, iw, pk.cache, 16, 128, length - queries, 2048, true, pagedIndex);
    Require(Read<int>(denseIndex) == Read<int>(pagedIndex), "paged Indexer changed TopK positions");
    FastllmCudaNaiveAttention(q, k, v, denseIndex, sink, heads, 1, dim, vd, length - queries, 0, expected);
    FastllmCudaNaiveAttention(q, pk.cache, pv.cache, pagedIndex, sink, heads, 1, dim, vd, length - queries, 0, actual);
    Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "paged prefill changed attention");
    // A write spanning two page boundaries leaves all other rows untouched.
    Data input(BFLOAT16); Upload(input, {1, 131, width}, 952);
    FastllmCudaNaivePagedWrite(pk.cache, input, 127);
    auto old = Read<uint16_t>(k), add = Read<uint16_t>(input);
    auto physical = Read<uint16_t>(pk.pool);
    for (int r = 0; r < length; ++r) for (int c = 0; c < width; ++c) {
        uint16_t ref = r >= 127 && r < 258 ? add[(r - 127) * width + c] : old[r * width + c];
        Require(physical[((size_t)pk.cache.pageIndex[r / pageLen] * pageLen + r % pageLen) * width + c] == ref,
            "paged append wrote wrong row or modified shared neighbors");
    }
    ++checks;
    std::printf("PAGED PREFILL/INDEXER/WRITE PASS length=%d\n", length);
}

static void TestPagedBatchIndexer(int pageLen, bool generic) {
    const int batch = 4, capacity = 4097, width = generic ? 321 : 320;
    Data q(BFLOAT16), weights(BFLOAT16), live(INT32);
    Upload(q, {1, batch, 16 * 128}, 901);
    Upload(weights, {1, batch, 16}, 909);
    live.Resize({batch}); live.Allocate();
    std::vector<int> lengths(batch, capacity);
    std::memcpy(live.cpuData, lengths.data(), batch * sizeof(int));
    live.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<Data> keys(batch, Data(BFLOAT16)), expected(batch), actual(batch);
    std::vector<std::unique_ptr<PagedTensor>> pages;
    std::vector<FastllmNaiveDecodeScratch> denseScratch(batch), pagedScratch(batch);
    std::vector<FastllmNaiveIndexerRow> dense, paged;
    for (int row = 0; row < batch; ++row) {
        Upload(keys[row], {1, capacity, width}, 101 + row);
        pages.emplace_back(new PagedTensor(keys[row], row + 37, pageLen));
        dense.push_back({&keys[row], &denseScratch[row], &expected[row], row, capacity});
        paged.push_back({&pages.back()->cache, &pagedScratch[row], &actual[row], row, capacity});
    }
    Graph graph;
    graph.Capture([&] {
        Require(FastllmCudaNaiveBatchDecodeIndexer(q, weights, live, paged, true), "paged batch indexer unavailable");
    });
    for (int base : {4097, 2049, 129}) {
        for (int row = 0; row < batch; ++row) lengths[row] = base - row * 17;
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        graph.Run();
        Require(FastllmCudaNaiveBatchDecodeIndexer(q, weights, live, dense, true), "dense batch indexer unavailable");
        for (int row = 0; row < batch; ++row) {
            Require(Read<int>(expected[row]) == Read<int>(actual[row]), "paged batch indexer TopK mismatch");
            auto a = Read<unsigned>(denseScratch[row].indexScores), b = Read<unsigned>(pagedScratch[row].indexScores);
            Require(std::equal(a.begin(), a.begin() + lengths[row], b.begin()), "paged batch indexer score mismatch");
        }
        ++checks;
    }
    std::printf("PAGED BATCH INDEXER PASS page_len=%d generic=%d\n", pageLen, generic);
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
        for (int batch : {1, 4, 32, 128, 256}) {
            TestRebinding(batch, 707, false);
            TestRebinding(batch, 4097, true);
        }
        TestRebinding(4, 707, false, true);
        TestRebinding(33, 4097, true, true);
        for (int capacity : {127, 128, 129, 2047, 2048, 4097}) {
            Test(4, 8, 1, 192, 128, capacity, capacity > 2048, true, false, false, true);
            Test(2, 8, 2, 128, 256, capacity, capacity > 2048, false, false, false, true);
        }
        for (int batch : {2, 8, 16, 31})
            Test(batch, 8, 1, 192, 128, 4097, true, true, false, false, true);
        for (int capacity : {257, 707, 2048, 4097})
            Test(128, 8, 1, 192, 128, capacity, capacity > 2048, true, false, false, true);
        for (int length : {511, 2048, 2049, 4097}) TestPagedPrefill(length);
        for (int heads : {8, 16}) TestPagedPrefill(4097, 256, heads);
        TestPagedPrefill(511, 3, 8, 96);
        for (int pageLen : {96, 128}) {
            TestPagedBatchIndexer(pageLen, false);
            TestPagedBatchIndexer(pageLen, true);
        }
        std::printf("BATCH DSA PASS %d checks\n", checks);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
