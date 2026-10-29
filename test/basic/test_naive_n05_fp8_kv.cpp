#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <numeric>
#include <random>
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

static float Float(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float x;
    std::memcpy(&x, &word, sizeof(x));
    return x;
}
struct FP8Tensor {
    Data dense{BFLOAT16}, encoded{FP8_E4M3}, table{INT32}, cache{FP8_E4M3};
    PagedCacheManager pool;
    std::vector<uint8_t> bits;
    std::vector<int> ids;
    FP8Tensor(int length, int width, int seed, bool paged) {
        Upload(dense, {1, length, width}, seed);
        auto bf = Read<uint16_t>(dense);
        bits.resize(bf.size());
        for (size_t i = 0; i < bf.size(); ++i) {
            __nv_fp8_e4m3 x(Float(bf[i]));
            bits[i] = x.__x;
            bf[i] = Float32ToBFloat16RNEBits((float)x);
        }
        Cuda(cudaMemcpy(dense.cudaData, bf.data(), bf.size() * 2, cudaMemcpyHostToDevice));
        encoded.Resize(dense.dims);
        encoded.ToDevice(DataDevice::CUDA, {0}, false);
        encoded.Allocate();
        Cuda(cudaMemcpy(encoded.cudaData, bits.data(), bits.size(), cudaMemcpyHostToDevice));
        if (!paged)
            return;
        int pages = (length + 127) / 128;
        ids.resize(pages);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
        pool.dataType = FP8_E4M3;
        pool.pageLen = 128;
        pool.Resize({pages, 128, 1, width});
        pool.ToDevice(DataDevice::CUDA, {0}, false);
        pool.Allocate();
        pool.SetMaxPages(pages);
        table.Resize({pages});
        table.Allocate();
        std::memcpy(table.cpuData, ids.data(), pages * 4);
        table.ToDevice(DataDevice::CUDA, {0}, true);
        for (int row = 0; row < length; row += 128)
            Cuda(cudaMemcpy((uint8_t *)pool.cudaData + (size_t)ids[row / 128] * 128 * width,
                            bits.data() + (size_t)row * width, (size_t)std::min(128, length - row) * width,
                            cudaMemcpyHostToDevice));
        cache.FakeFrom(table, 0);
        cache.dataType = FP8_E4M3;
        cache.Resize(dense.dims);
        cache.isPagedKVCache = true;
        cache.pagedKVCacheData = &pool;
        cache.pageLen = 128;
        cache.pageIndex = ids;
        cache.lastPageLen = (length - 1) % 128 + 1;
        cache.expansionDims = {1, pages * 128, width};
    }
    Data &data() { return ids.empty() ? encoded : cache; }
};
static void Near(const Data &a, const Data &b, const char *name) {
    auto x = Read<uint16_t>(a), y = Read<uint16_t>(b);
    Require(x.size() == y.size(), "shape mismatch");
    double e = 0, n = 0, m = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double delta = Float(x[i]) - Float(y[i]);
        Require(std::isfinite(delta), "nonfinite output");
        e += delta * delta;
        n += (double)Float(y[i]) * Float(y[i]);
        m = std::max(m, std::abs(delta));
    }
    double nr = std::sqrt(e / std::max(n, 1e-30));
    if (nr > .004 || m > .004) {
        std::printf("FAIL %s nr=%g max=%g\n", name, nr, m);
        throw std::runtime_error(name);
    }
    ++checks;
}
static void FP8DSA(int batch, int length, bool sparse, bool paged, int heads = 8, int dim = 192,
                   int vd = 128, int padding = 0) {
    const int width = dim + 128 + padding;
    Data q(BFLOAT16), sink(FLOAT32), live(INT32), a, b;
    Upload(q, {1, batch, heads * dim}, 31);
    Upload(sink, {heads}, 51);
    live.Resize({batch});
    live.Allocate();
    for (int i = 0; i < batch; ++i)
        ((int *)live.cpuData)[i] = length - i % 7;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<std::unique_ptr<FP8Tensor>> k, v;
    std::vector<Data> idx(batch, Data(INT32));
    std::vector<FastllmNaiveDSARow> fp(batch), bf(batch);
    for (int i = 0; i < batch; ++i) {
        k.emplace_back(new FP8Tensor(length, width, 7 + i, paged));
        v.emplace_back(new FP8Tensor(length, vd, 19 + i, paged));
        auto &r = fp[i];
        r.key = paged ? k.back()->pool.cudaData : k.back()->encoded.cudaData;
        r.value = paged ? v.back()->pool.cudaData : v.back()->encoded.cudaData;
        r.fp8 = true;
        r.liveKeys = (const int *)live.cudaData + i;
        r.capacity = length;
        r.keyStride = width;
        if (paged) {
            r.keyPages = (const int *)k.back()->table.cudaData;
            r.valuePages = (const int *)v.back()->table.cudaData;
            r.pageLen = 128;
            r.poolTokens = k.back()->pool.maxPages * 128;
        }
        if (sparse) {
            idx[i].Resize({1, 2048});
            idx[i].Allocate();
            for (int j = 0; j < 2048; ++j)
                ((int *)idx[i].cpuData)[j] = j % 17 ? (j * 37 + i) % length : -1;
            idx[i].ToDevice(DataDevice::CUDA, {0}, true);
            r.indices = (const int *)idx[i].cudaData;
            r.selectedCount = 2048;
        }
        bf[i] = r;
        bf[i].key = k.back()->dense.cudaData;
        bf[i].value = v.back()->dense.cudaData;
        bf[i].fp8 = false;
        bf[i].keyPages = bf[i].valuePages = nullptr;
        bf[i].pageLen = bf[i].poolTokens = 0;
    }
    FastllmNaiveDecodeScratch sa, sb;
    Graph graph;
    graph.Capture([&] { FastllmCudaNaiveBatchDSADecode(q, sink, fp, heads, 1, dim, vd, sa, a); });
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < batch; ++i)
            ((int *)live.cpuData)[i] = pass ? std::max(1, length / 2 - i % 7) : length - i % 7;
        Cuda(cudaMemcpy(live.cudaData, live.cpuData, batch * 4, cudaMemcpyHostToDevice));
        graph.Run();
        FastllmCudaNaiveBatchDSADecode(q, sink, bf, heads, 1, dim, vd, sb, b);
        Near(a, b, "DSA");
    }
    printf("FP8 DSA PASS batch=%d length=%d sparse=%d paged=%d stride=%d\n", batch, length, sparse, paged, width);
    fflush(stdout);
}
static void FP8Prefill(int length, bool paged) {
    FP8Tensor k(length, 320, 19, paged), v(length, 128, 27, paged);
    Data q(BFLOAT16), iq(BFLOAT16), iw(BFLOAT16), i8, ib, a, b, sink;
    Upload(q, {1, 3, 8 * 192}, 17);
    Upload(iq, {1, 3, 16 * 128}, 31);
    Upload(iw, {1, 3, 16}, 49);
    FastllmCudaNaiveIndexer(iq, iw, k.data(), 16, 128, length - 3, 2048, true, i8);
    FastllmCudaNaiveIndexer(iq, iw, k.dense, 16, 128, length - 3, 2048, true, ib);
    Require(Read<int>(i8) == Read<int>(ib), "FP8 prefill indexer");
    FastllmCudaNaiveAttention(q, k.data(), v.data(), i8, sink, 8, 1, 192, 128, length - 3, 0, a);
    FastllmCudaNaiveAttention(q, k.dense, v.dense, ib, sink, 8, 1, 192, 128, length - 3, 0, b);
    Near(a, b, "prefill");
    Data input(BFLOAT16);
    Upload(input, {1, 131, 320}, 97);
    auto add = Read<uint16_t>(input);
    FastllmCudaNaivePagedWrite(k.data(), input, 127);
    auto actual = Read<uint8_t>(paged ? k.pool : k.encoded);
    for (int row = 0; row < length; ++row)
        for (int col = 0; col < 320; ++col) {
            auto ref = k.bits[(size_t)row * 320 + col];
            if (row >= 127 && row < 258)
                ref = __nv_fp8_e4m3(Float(add[(row - 127) * 320 + col])).__x;
            size_t physical = paged ? (size_t)k.ids[row / 128] * 128 + row % 128 : row;
            Require(actual[physical * 320 + col] == ref, "FP8 append quantization/address mismatch");
        }
    printf("FP8 PREFILL/INDEXER/WRITE PASS length=%d paged=%d\n", length, paged);
    fflush(stdout);
}

static void PrefillIndexerPaths(int queries, int length, bool paged, bool remote) {
    FP8Tensor keys(length, 320, 101, paged);
    Data q(BFLOAT16), weights(BFLOAT16);
    Upload(q, {1, queries, 16 * 128}, 103);
    Upload(weights, {1, queries, 16}, 107);
    for (Data *key : {&keys.data(), &keys.dense}) {
        Data local, shared;
        FastllmCudaNaiveIndexer(q, weights, *key, 16, 128, length - queries, 2048, true, local);
        FastllmCudaNaiveIndexer(q, weights, *key, 16, 128, length - queries, 2048, true, shared, true);
        auto expected = Read<int>(local);
        Require(expected == Read<int>(shared), "shared prefill Indexer TopK differs from local");
        ++checks;
        if (remote) {
            Data peerQuery(BFLOAT16), peerWeights(BFLOAT16), peer;
            Upload(peerQuery, q.dims, 103);
            Upload(peerWeights, weights.dims, 107);
            peerQuery.ToDevice(DataDevice::CUDA, {1}, true);
            peerWeights.ToDevice(DataDevice::CUDA, {1}, true);
            FastllmCudaSetDevice(1);
            FastllmCudaNaiveIndexer(peerQuery, peerWeights, *key, 16, 128,
                                   length - queries, 2048, true, peer, true);
            Require(expected == Read<int>(peer), "peer prefill Indexer TopK differs from local");
            ++checks;
            FastllmCudaSetDevice(0);
        }
    }
    printf("PREFILL INDEXER PATHS PASS queries=%d length=%d paged=%d remote=%d\n",
           queries, length, paged, remote);
    fflush(stdout);
}

// The merged launch must preserve every selected position, including ties.
static void BalancedPrefillIndexer(bool paged, bool remote) {
    if (FastllmCudaRuntimeArch() != 90)
        return;
    constexpr int length = 1024, keys = 65537;
    FP8Tensor raw(keys, 320, 121, paged);
    Data input(BFLOAT16), packed, q(BFLOAT16), w(BFLOAT16);
    Upload(input, {1, keys, 128}, 127);
    FastllmCudaNaivePackIndexer(input, packed);
    Upload(q, {1, length, 2048}, 131);
    Upload(w, {1, length, 16}, 137);
    if (remote) {
        q.ToDevice(DataDevice::CUDA, {1}, true);
        w.ToDevice(DataDevice::CUDA, {1}, true);
        FastllmCudaSetDevice(1);
    }
    for (const Data *key : {&raw.data(), &packed})
        for (int ranks : {2, 4, 8}) {
            int rows = length / ranks, part = rows / 2;
            for (bool ties : {false, true}) {
                if (ties)
                    Cuda(cudaMemset(w.cudaData, 0, w.GetBytes()));
                else {
                    FastllmCudaSetDevice(0);
                    Data fresh(BFLOAT16);
                    Upload(fresh, {1, length, 16}, 137);
                    if (remote)
                        fresh.ToDevice(DataDevice::CUDA, {1}, true);
                    FastllmCudaSetDevice(remote ? 1 : 0);
                    Cuda(cudaMemcpy(w.cudaData, fresh.cudaData, w.GetBytes(), cudaMemcpyDeviceToDevice));
                }
                for (int rank = 0; rank < ranks; ++rank) {
                    Data expected(INT32), actual;
                    expected.Resize({rows, 2048});
                    expected.ToDevice(DataDevice::CUDA, q.dataDeviceIds, false);
                    expected.Allocate(false);
                    for (int range = 0; range < 2; ++range) {
                        int first = range ? length - (rank + 1) * part : rank * part;
                        Data a(BFLOAT16, {1, part, 2048}), b(BFLOAT16, {1, part, 16}), out(INT32, {part, 2048});
                        a.FakeFrom(q, (size_t)first * 2048 * 2);
                        b.FakeFrom(w, (size_t)first * 16 * 2);
                        out.FakeFrom(expected, (size_t)range * part * 2048 * 4);
                        FastllmCudaNaiveIndexer(a, b, *key, 16, 128, keys - length + first, 2048, true, out, true);
                    }
                    Require(FastllmCudaNaiveBalancedIndexer(q, w, *key, keys - length, rank, ranks, true, true, actual),
                            "balanced prefill unexpectedly declined");
                    Require(Read<int>(expected) == Read<int>(actual), "balanced prefill changed TopK/tie order");
                    auto saved = Read<int>(actual);
                    Require(
                        !FastllmCudaNaiveBalancedIndexer(q, w, *key, keys - length, rank, ranks, true, false, actual),
                        "balanced prefill changed non-peer arithmetic");
                    Require(saved == Read<int>(actual), "rejected balanced prefill touched output");
                    ++checks;
                }
            }
        }
    FastllmCudaSetDevice(0);
    printf("BALANCED PREFILL PASS paged=%d remote=%d\n", paged, remote);
    fflush(stdout);
}

// Ragged requests share launches, never cache addresses or causal positions.
static void BatchBalancedPrefillIndexer(bool paged, bool remote, bool eight = false) {
    if (FastllmCudaRuntimeArch() != 90)
        return;
    constexpr int ranks = 4, topK = 2048;
    std::vector<int> lengths = eight ? std::vector<int>(8, 512) : std::vector<int>{512, 1024, 1536, 4096};
    std::vector<int> contexts = eight ? std::vector<int>(8, 32769) : std::vector<int>{32769, 16385, 8193, 65537};
    struct Request {
        std::unique_ptr<FP8Tensor> raw;
        Data packed, q{BFLOAT16}, weights{BFLOAT16}, expected{INT32}, storage{INT32}, output;
        const Data *key = nullptr;
    };
    std::vector<std::unique_ptr<Request>> requests;
    for (int i = 0; i < (int)lengths.size(); ++i) {
        auto item = std::make_unique<Request>();
        item->raw = std::make_unique<FP8Tensor>(contexts[i], 320, 151 + i, paged);
        if (i % 3 == 2) {
            Data input(BFLOAT16);
            Upload(input, {1, contexts[i], 128}, 161 + i);
            FastllmCudaNaivePackIndexer(input, item->packed);
            item->key = &item->packed;
        } else
            item->key = i % 3 ? &item->raw->dense : &item->raw->data();
        Upload(item->q, {1, lengths[i], 2048}, 171 + i);
        Upload(item->weights, {1, lengths[i], 16}, 181 + i);
        if (remote) {
            item->q.ToDevice(DataDevice::CUDA, {1}, true);
            item->weights.ToDevice(DataDevice::CUDA, {1}, true);
        }
        FastllmCudaSetDevice(remote ? 1 : 0);
        const int rows = lengths[i] / ranks;
        item->expected.Resize({rows, topK});
        item->expected.ToDevice(DataDevice::CUDA, {remote ? 1 : 0}, false);
        item->expected.Allocate(false);
        item->storage.Resize({rows + 2, topK});
        item->storage.ToDevice(DataDevice::CUDA, {remote ? 1 : 0}, false);
        item->storage.Allocate(false);
        Cuda(cudaMemset(item->storage.cudaData, 0x5a, item->storage.GetBytes()));
        item->output.Resize({rows, topK});
        item->output.FakeFrom(item->storage, topK * sizeof(int));
        requests.push_back(std::move(item));
        FastllmCudaSetDevice(0);
    }
    FastllmCudaSetDevice(remote ? 1 : 0);
    for (bool ties : {false, true}) {
        if (ties)
            for (auto &r : requests)
                Cuda(cudaMemset(r->weights.cudaData, 0, r->weights.GetBytes()));
        for (int rank : {0, 1, 3}) {
            std::vector<FastllmNaivePrefillIndexerItem> items;
            for (int i = 0; i < (int)requests.size(); ++i) {
                auto &r = *requests[i];
                const int part = lengths[i] / (2 * ranks);
                for (int range = 0; range < 2; ++range) {
                    int first = range ? lengths[i] - (rank + 1) * part : rank * part;
                    Data q(BFLOAT16, {1, part, 2048}), w(BFLOAT16, {1, part, 16}), out(INT32, {part, topK});
                    q.FakeFrom(r.q, (size_t)first * 2048 * 2);
                    w.FakeFrom(r.weights, (size_t)first * 16 * 2);
                    out.FakeFrom(r.expected, (size_t)range * part * topK * 4);
                    FastllmCudaNaiveIndexer(q, w, *r.key, 16, 128, contexts[i] - lengths[i] + first, topK, true, out,
                                            true);
                }
                items.push_back({&r.q, &r.weights, r.key, &r.output, contexts[i] - lengths[i]});
            }
            Require(FastllmCudaNaiveBatchBalancedIndexer(items, rank, ranks, true, true), "batch balanced declined");
            std::vector<std::vector<int>> saved;
            for (auto &r : requests) {
                Require(Read<int>(r->expected) == Read<int>(r->output), "batch balanced TopK/tie mismatch");
                auto whole = Read<int>(r->storage);
                for (int i = 0; i < topK; ++i)
                    Require(whole[i] == 0x5a5a5a5a && whole[whole.size() - topK + i] == 0x5a5a5a5a,
                            "batch output guard overwritten");
                saved.push_back(whole);
                ++checks;
            }
            items.back().past = -1;
            Require(!FastllmCudaNaiveBatchBalancedIndexer(items, rank, ranks, true, true), "invalid batch accepted");
            items.back().past = contexts.back() - lengths.back();
            if (!eight) {
                // Metadata-only large contexts must decline before reading KV.
                std::vector<std::unique_ptr<Data>> largeKeys;
                auto large = items;
                for (auto &item : large) {
                    auto key = std::make_unique<Data>(item.keys->dataType, item.keys->dims);
                    key->FakeFrom(*item.keys, 0);
                    key->dims[1] = 200000;
                    item.keys = key.get();
                    item.past = 200000 - item.query->dims[1];
                    largeKeys.push_back(std::move(key));
                }
                Require(!FastllmCudaNaiveBatchBalancedIndexer(large, rank, ranks, true, true), "score budget exceeded");
            }
            items.resize(9, items[0]);
            Require(!FastllmCudaNaiveBatchBalancedIndexer(items, rank, ranks, true, true), "oversized batch accepted");
            for (int i = 0; i < (int)requests.size(); ++i)
                Require(saved[i] == Read<int>(requests[i]->storage), "declined batch changed output");
        }
    }
    FastllmCudaSetDevice(0);
    printf("BATCH BALANCED PREFILL PASS paged=%d remote=%d requests=%zu\n", paged, remote, requests.size());
    fflush(stdout);
}
static void FusedPrefillAttention(bool paged) {
    for (int rows : {32, 65, 256, 2048})
        for (int kvHeads : {1, 2}) {
            const int heads = kvHeads * 16, length = 8193;
            FP8Tensor key(length, kvHeads * 192 + 128, 91, paged), value(length, kvHeads * 128, 97, paged);
            Data q(BFLOAT16), sink(FLOAT32), idx(INT32), reference, actual;
            Upload(q, {1, rows, heads * 192}, 53);
            Upload(sink, {heads}, 71);
            idx.Resize({rows, 2048});
            idx.Allocate();
            for (int r = 0; r < rows; ++r)
                for (int i = 0; i < 2048; ++i) {
                    int k = (i * 137 + r * 13) % length;
                    if (r == 0 || i % 97 == 0)
                        k = -1;
                    else if (i % 193 == 0)
                        k = length + 1;
                    ((int *)idx.cpuData)[r * 2048 + i] = k;
                }
            idx.ToDevice(DataDevice::CUDA, {0}, true);
            for (bool causal : {false, true}) {
                // Dense BF16 contains the exact dequantized FP8 values and uses
                // the materialized-score fallback instead of the FP8 fused path.
                FastllmCudaNaiveAttention(q, key.dense, value.dense, idx, sink, heads, kvHeads, 192, 128,
                                          length - rows, 0, reference, causal);
                FastllmCudaNaiveAttention(q, key.data(), value.data(), idx, sink, heads, kvHeads, 192, 128,
                                          length - rows, 0, actual, causal);
                Require(Read<uint16_t>(actual) == Read<uint16_t>(reference), "fused FP8 prefill changed exact output");
                ++checks;
                if (causal && rows % 8 == 0) {
                    Data balanced(INT32);
                    balanced.Resize({rows, 2048});
                    balanced.Allocate();
                    int tile = rows / 8;
                    auto original = Read<int>(idx);
                    for (int row = 0; row < rows; ++row) {
                        int rank = std::min(row, rows - 1 - row) / tile;
                        int target = (2 * rank + (row >= rows / 2)) * tile + row % tile;
                        std::memcpy((int *)balanced.cpuData + target * 2048, original.data() + row * 2048, 2048 * 4);
                    }
                    balanced.ToDevice(DataDevice::CUDA, {0}, true);
                    bool fused = FastllmCudaNaiveBalancedAttention(q, key.data(), value.data(), balanced, sink, heads,
                                                                   kvHeads, length - rows, 4, actual);
                    Require(fused == (rows >= 2048 && FastllmCudaRuntimeArch() == 90), "balanced fused dispatch");
                    Require(Read<uint16_t>(actual) == Read<uint16_t>(reference),
                            "balanced fused FP8 prefill changed exact output");
                    ++checks;
                }
            }
        }
    printf("FUSED FP8 PREFILL PASS paged=%d\n", paged);
    fflush(stdout);
}

static void PrefillSwaStaging(bool paged) {
    for (int rows : {33, 256, 257, 1024})
        for (int past : {0, 127, 7935}) {
            const int heads = 16, kvHeads = 2, length = past + rows;
            FP8Tensor key(length, kvHeads * 192, 91, paged), value(length, kvHeads * 128, 97, paged);
            Data q(BFLOAT16), sink(FLOAT32), idx, reference, actual;
            Upload(q, {1, rows, heads * 192}, 53);
            Upload(sink, {heads}, 71);
            auto kb = Read<uint8_t>(paged ? key.pool : key.encoded),
                 vb = Read<uint8_t>(paged ? value.pool : value.encoded);
            // Chunks below the staging threshold read FP8 directly. Every
            // query retains its absolute causal position and the same cache.
            std::vector<uint16_t> expected;
            for (int first = 0; first < rows;) {
                const int count = rows - first < 256 ? rows - first : 128;
                Data chunk;
                chunk.Resize({1, count, heads * 192});
                chunk.FakeFrom(q, (size_t)first * heads * 192 * q.unitSize);
                FastllmCudaNaiveAttention(chunk, key.data(), value.data(), idx, sink, heads, kvHeads, 192, 128,
                                          past + first, 128, reference);
                auto values = Read<uint16_t>(reference);
                expected.insert(expected.end(), values.begin(), values.end());
                first += count;
            }
            FastllmCudaNaiveAttention(q, key.data(), value.data(), idx, sink, heads, kvHeads, 192, 128, past, 128,
                                      actual);
            Require(expected == Read<uint16_t>(actual), "SWA staging changed exact output");
            Require(kb == Read<uint8_t>(paged ? key.pool : key.encoded), "SWA staging modified stored key");
            Require(vb == Read<uint8_t>(paged ? value.pool : value.encoded), "SWA staging modified stored value");
            ++checks;
        }
    printf("SWA PREFILL STAGING PASS paged=%d\n", paged);
    fflush(stdout);
}

static void FP8SWA(int batch) {
    Data q(BFLOAT16), sink(FLOAT32), live(INT32), a, b;
    Upload(q, {1, batch, 8 * 192}, 31);
    Upload(sink, {8}, 51);
    live.Resize({batch});
    live.Allocate();
    for (int i = 0; i < batch; ++i)
        ((int *)live.cpuData)[i] = 8192 + i * 37;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<std::unique_ptr<FP8Tensor>> k, v;
    std::vector<FastllmNaiveSWARow> fp, bf;
    for (int i = 0; i < batch; ++i) {
        k.emplace_back(new FP8Tensor(128, 192, 7 + i, false));
        v.emplace_back(new FP8Tensor(128, 128, 19 + i, false));
        fp.push_back({k.back()->encoded.cudaData, v.back()->encoded.cudaData, nullptr, true});
        bf.push_back({k.back()->dense.cudaData, v.back()->dense.cudaData, nullptr, false});
    }
    Graph graph;
    graph.Capture([&] { FastllmCudaNaiveBatchSWADecode(q, sink, fp, live, 8, 1, a); });
    graph.Run();
    FastllmCudaNaiveBatchSWADecode(q, sink, bf, live, 8, 1, b);
    Near(a, b, "SWA");
    printf("FP8 SWA PASS batch=%d\n", batch);
    fflush(stdout);
}
static double indexerRelativeError = 0, indexerMaxError = 0;
static void CheckIndexerScores(const std::vector<unsigned> &actual,
                               const std::vector<unsigned> &expected, int count) {
    auto value = [](unsigned bits) {
        bits = bits & 0x80000000u ? bits ^ 0x80000000u : ~bits;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return (double)result;
    };
    double error = 0, norm = 0, maximum = 0, scale = 0;
    for (int i = 0; i < count; ++i) {
        double a = value(actual[i]), b = value(expected[i]);
        Require(std::isfinite(a) && std::isfinite(b), "non-finite Indexer score");
        error += (a - b) * (a - b); norm += b * b;
        maximum = std::max(maximum, std::abs(a - b));
        scale = std::max(scale, std::abs(b));
    }
    // E4M3 values are unchanged, but FP8 k32 and BF16 k16 MMA can round
    // FP32 accumulators differently. Bound both aggregate error and outliers;
    // retain exact TopK comparisons separately below.
    double relative = std::sqrt(error / std::max(norm, 1e-30));
    indexerRelativeError = std::max(indexerRelativeError, relative);
    indexerMaxError = std::max(indexerMaxError, maximum);
    Require(relative <= 1e-7 && maximum <= 1e-6 + 2e-7 * scale, "FP8 indexer score error");
}

static void FP8BatchIndexer(int batch, bool paged, int width = 320, int length = 4097) {
    Data q(BFLOAT16), w(BFLOAT16), live(INT32);
    Upload(q, {1, batch, 16 * 128}, 191);
    Upload(w, {1, batch, 16}, 101);
    live.Resize({batch});
    live.Allocate();
    for (int i = 0; i < batch; ++i)
        ((int *)live.cpuData)[i] = length - i % 7;
    live.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<std::unique_ptr<FP8Tensor>> k;
    std::vector<Data> a(batch), b(batch);
    std::vector<FastllmNaiveDecodeScratch> sa(batch), sb(batch);
    std::vector<FastllmNaiveIndexerRow> fp, bf;
    for (int i = 0; i < batch; ++i) {
        k.emplace_back(new FP8Tensor(length, width, 37 + i, paged));
        fp.push_back({&k.back()->data(), &sa[i], &a[i], i, length});
        bf.push_back({&k.back()->dense, &sb[i], &b[i], i, length});
    }
    Graph graph;
    graph.Capture([&] {
        Require(FastllmCudaNaiveBatchDecodeIndexer(q, w, live, fp, true), "FP8 batch indexer fallback");
    });
    graph.Run();
    Require(FastllmCudaNaiveBatchDecodeIndexer(q, w, live, bf, true), "BF16 batch indexer fallback");
    for (int i = 0; i < batch; ++i) {
        Require(Read<int>(a[i]) == Read<int>(b[i]), "FP8 batch indexer TopK mismatch");
        auto x = Read<unsigned>(sa[i].indexScores), y = Read<unsigned>(sb[i].indexScores);
        CheckIndexerScores(x, y, length - i % 7);
    }
    printf("FP8 BATCH INDEXER PASS batch=%d paged=%d stride=%d\n", batch, paged, width);
    fflush(stdout);
    if (length <= 16384) return;
    // Reuse one captured graph across ragged live lengths, including a padded
    // row. Compare against the separate single-row scoring implementation.
    std::vector<int> lengths(batch);
    for (int i = 0; i < batch; ++i)
        lengths[i] = i % 4 == 0 ? length : i % 4 == 1 ? 8191 : i % 4 == 2 ? 257 : 0;
    Cuda(cudaMemcpy(live.cudaData, lengths.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
    graph.Run();
    std::vector<std::vector<int>> expected(batch);
    for (int i = 0; i < batch; ++i) {
        expected[i] = Read<int>(a[i]);
        if (!lengths[i]) continue;
        Data qi, wi, reference;
        qi.dataType = wi.dataType = BFLOAT16;
        qi.Resize({1, 1, 2048}); qi.FakeFrom(q, (size_t)i * 2048 * sizeof(uint16_t));
        wi.Resize({1, 1, 16}); wi.FakeFrom(w, (size_t)i * 16 * sizeof(uint16_t));
        Data liveRow(INT32);
        liveRow.Resize({1}); liveRow.FakeFrom(live, (size_t)i * sizeof(int));
        FastllmNaiveDecodeScratch scratch;
        FastllmCudaNaiveDecodeIndexer(qi, wi, k[i]->data(), liveRow,
                                     length, true, scratch, reference);
        Require(expected[i] == Read<int>(reference), "ragged Indexer changed scalar TopK");
    }
    // Serial execution of rank shards isolates partition/merge semantics from
    // transport. Model tests cover the actual cross-device exchange.
    for (int ranks : {2, 8}) {
        Data exchange(INT32);
        exchange.Resize({ranks * batch * 2048 * 2});
        exchange.ToDevice(DataDevice::CUDA, {0}, false); exchange.Allocate();
        Graph parallel;
        parallel.Capture([&] {
            for (int rank = 0; rank < ranks; ++rank)
                Require(FastllmCudaNaiveBatchDecodeIndexer(q, w, live, fp, true,
                    rank, ranks, &exchange), "context Indexer unavailable");
            FastllmCudaNaiveMergeDecodeIndexer(live, fp, exchange, ranks);
        });
        parallel.Run();
        for (int i = 0; i < batch; ++i)
            Require(expected[i] == Read<int>(a[i]), "context Indexer changed TopK");
    }
    printf("RAGGED INDEXER PASS capacity=%d paged=%d stride=%d\n", length, paged, width);
}

static void FP8Rope(bool paged, int window) {
    const int capacity = window ? 128 : 512;
    FP8Tensor key(capacity, window ? 192 : 320, 51, paged), value(capacity, 128, 73, paged);
    for (int past : {0, 126, 127}) {
        Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), ik(BFLOAT16), pos(FLOAT32), live(INT32), qr(BFLOAT16);
        Upload(q, {1, 1, 8 * 192}, 11);
        Upload(qr, q.dims, 11);
        Upload(k, {1, 1, 192}, 17);
        Upload(v, {1, 1, 128}, 31);
        if (!window)
            Upload(ik, {1, 1, 128}, 41);
        Upload(pos, {1, 1}, 77);
        float position = 8192 + past;
        Cuda(cudaMemcpy(pos.cudaData, &position, 4, cudaMemcpyHostToDevice));
        live.Resize({1});
        live.Allocate();
        ((int *)live.cpuData)[0] = past + 1;
        live.ToDevice(DataDevice::CUDA, {0}, true);
        auto kb = Read<uint8_t>(paged ? key.pool : key.encoded),
             vb = Read<uint8_t>(paged ? value.pool : value.encoded);
        Require(FastllmCudaNaiveRopeAppendCache(q, k, v, ik, pos, key.data(), value.data(), live, 8, 1, 192,
                                                128, 64, 1e7, .707, window, nullptr, window != 0),
                "FP8 rope path rejected");
        FastllmCudaNaiveRopeQKScaleV(qr, k, v, pos, 8, 1, 192, 128, 64, 1e7, .707);
        if (!window)
            FastllmCudaNaiveRope(ik, pos, 1, 128, 64, 1e7);
        Require(Read<uint16_t>(q) == Read<uint16_t>(qr), "FP8 rope changed Q");
        auto hk = Read<uint16_t>(k), hv = Read<uint16_t>(v),
             hi = window ? std::vector<uint16_t>{} : Read<uint16_t>(ik);
        int row = window ? past & 127 : past;
        int kp = paged ? key.ids[row / 128] * 128 + row % 128 : row,
            vp = paged ? value.ids[row / 128] * 128 + row % 128 : row;
        int width = window ? 192 : 320;
        for (int c = 0; c < width; ++c)
            kb[(size_t)kp * width + c] = __nv_fp8_e4m3(Float(c < 192 ? hk[c] : hi[c - 192])).__x;
        for (int c = 0; c < 128; ++c)
            vb[(size_t)vp * 128 + c] = __nv_fp8_e4m3(Float(hv[c])).__x;
        Require(kb == Read<uint8_t>(paged ? key.pool : key.encoded), "FP8 rope K rounding/address");
        Require(vb == Read<uint8_t>(paged ? value.pool : value.encoded), "FP8 rope V rounding/address");
        if (!window) {
            Data restored(BFLOAT16);
            restored.Resize({1, 131, width});
            restored.ToDevice(DataDevice::CUDA, {0}, false);
            restored.Allocate();
            FastllmCudaNaiveReadCache({{&key.data(), restored.cudaData, 127, 131}});
            auto actual = Read<uint16_t>(restored);
            for (int r = 0; r < 131; ++r)
                for (int c = 0; c < width; ++c) {
                    int physical = paged ? key.ids[(r + 127) / 128] * 128 + (r + 127) % 128 : r + 127;
                    __nv_fp8_e4m3 x;
                    x.__x = kb[(size_t)physical * width + c];
                    Require(actual[r * width + c] == Float32ToBFloat16RNEBits((float)x),
                            "FP8 history read across pages");
                }
        }
    }
    printf("FP8 ROPE/HISTORY PASS paged=%d window=%d\n", paged, window);
    fflush(stdout);
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        FastllmCudaSetDevice(0);
        int peerAccess = 0;
        if (devices > 1) {
            Cuda(cudaDeviceCanAccessPeer(&peerAccess, 1, 0));
            if (peerAccess) {
                FastllmCudaSetDevice(1);
                cudaError_t error = cudaDeviceEnablePeerAccess(0, 0);
                if (error == cudaErrorPeerAccessAlreadyEnabled)
                    cudaGetLastError();
                else
                    Cuda(error);
                FastllmCudaSetDevice(0);
            }
        }
        for (bool paged : {false, true}) {
            PrefillSwaStaging(paged);
            FusedPrefillAttention(paged);
            BatchBalancedPrefillIndexer(paged, false);
            if (peerAccess) BatchBalancedPrefillIndexer(paged, true);
            BalancedPrefillIndexer(paged,false);
            if(peerAccess)BalancedPrefillIndexer(paged,true);
        }
        BatchBalancedPrefillIndexer(true, peerAccess != 0, true);
        for (bool paged : {false, true}) {
            PrefillIndexerPaths(32, 32769, paged, peerAccess != 0);
            PrefillIndexerPaths(65, 16385, paged, peerAccess != 0);
            PrefillIndexerPaths(512, 32769, paged, peerAccess != 0);
            // Exceeds the score workspace target, including a ragged final
            // query tile. Local, shared and peer paths must keep identical TopK.
            PrefillIndexerPaths(2049, 32769, paged, peerAccess != 0);
        }
        for (bool paged : {false, true}) {
            for (int b : {1, 2, 4, 8, 32, 128, 256}) {
                FP8DSA(b, 707, false, paged);
                FP8DSA(b, 4097, true, paged);
            }
            FP8Prefill(4097, paged);
            FP8DSA(4, 4097, true, paged, 4, 128, 256);
            // An eight-element-aligned FP8 row cannot use 16-byte cp.async.
            FP8DSA(32, 4097, true, paged, 8, 192, 128, 8);
            FP8BatchIndexer(32, paged, 328);
        }
        for (int b : {1, 2, 4, 8, 32, 128, 256})
            FP8SWA(b);
        for (int b : {4, 32, 128, 256})
            for (bool paged : {false, true})
                FP8BatchIndexer(b, paged);
        for (int length : {16385, 131073, 200000})
            for (int width : {320, 328})
                FP8BatchIndexer(4, true, width, length);
        FP8Rope(false, 0);
        FP8Rope(true, 0);
        FP8Rope(false, 128);
        printf("ALL FP8 CHECKS PASSED %d indexer_relative=%.9g indexer_max_abs=%.9g\n",
               checks, indexerRelativeError, indexerMaxError);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
