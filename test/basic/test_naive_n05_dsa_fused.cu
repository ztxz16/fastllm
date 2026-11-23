#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include "naivert/attn_dsa/args.cuh"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <exception>
using namespace fastllm;
static void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Require(bool x, const char *s) {
    if (!x)
        throw std::runtime_error(s);
}
static unsigned rng = 17;
static bool fp8Cache = true;
static float Random() {
    rng = 1664525u * rng + 1013904223u;
    return (int(rng >> 16) - 32768) / 32768.f;
}
static void Make(Data &d, std::vector<int> shape) {
    d.Resize(shape);
    d.Allocate();
    if (d.dataType == BFLOAT16)
        for (size_t i = 0; i < d.Count(0); ++i)
            ((__nv_bfloat16 *)d.cpuData)[i] = __float2bfloat16(Random());
    d.ToDevice(CUDA, {0}, true);
}
template <class T> static std::vector<T> Read(Data &d) {
    std::vector<T> x(d.GetBytes() / sizeof(T));
    Check(cudaMemcpy(x.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return x;
}
static void Test(int past, int n, bool ties, bool sinkOn) {
    int cap = (past + n + 127) / 128 * 128, stride = fp8Cache ? 264 : 320;
    Data q(BFLOAT16), iq(BFLOAT16), w(BFLOAT16), key(BFLOAT16), value(BFLOAT16), indices(INT32), live(INT32),
        sink(FLOAT32), reference, actual;
    Make(q, {1, n, 1536});
    Make(iq, {1, n, 2048});
    Make(w, {1, n, 16});
    Make(value, {1, cap, 128});
    key.Resize({1, cap, stride});
    key.Allocate();
    for (int r = 0; r < cap; ++r) {
        auto *p = (__nv_bfloat16 *)key.cpuData + r * stride;
        for (int d = 0; d < 192; ++d)
            p[d] = __float2bfloat16(Random());
        if (fp8Cache) {
            auto *ix = (__nv_fp8_e4m3 *)(p + 192);
            for (int d = 0; d < 128; ++d)
                ix[d] = __nv_fp8_e4m3(ties ? 0.f : Random() * 4.f);
            float scale = .01f;
            std::memcpy((char *)ix + 128, &scale, 4);
            std::memset((char *)ix + 132, 0, 12);
        } else
            for (int d = 0; d < 128; ++d)
                p[192 + d] = __float2bfloat16(ties ? 0.f : Random());
    }
    key.expansionDims = key.dims;
    key.kvCacheFp8Indexer = fp8Cache;
    key.ToDevice(CUDA, {0}, true);
    Make(indices, {n, 2048});
    live.Resize({n});
    live.Allocate();
    for (int i = 0; i < n; ++i)
        ((int *)live.cpuData)[i] = past + i + 1;
    live.ToDevice(CUDA, {0}, true);
    if (sinkOn) {
        sink.Resize({8});
        sink.Allocate();
        for (int i = 0; i < 8; ++i)
            ((float *)sink.cpuData)[i] = i * .1f;
        sink.ToDevice(CUDA, {0}, true);
    }
    std::vector<FastllmNaiveVerifyIndexerItem> items = {{&key, &indices, 0, n, past, cap, nullptr}};
    std::vector<FastllmNaiveDSARow> rows(n);
    for (int i = 0; i < n; ++i) {
        auto &r = rows[i];
        r.key = key.cudaData;
        r.value = value.cudaData;
        r.length = past + i + 1;
        r.capacity = cap;
        r.keyStride = stride;
        r.liveKeys = (int *)live.cudaData + i;
        r.indices = (int *)indices.cudaData + size_t(i) * 2048;
        r.selectedCount = 2048;
    }
    FastllmNaiveDecodeScratch oldScratch, newScratch;
    auto old = [&] {
        Require(FastllmCudaNaiveBatchVerifyIndexer(iq, w, items, true, oldScratch, live),
                "old indexer rejected");
        FastllmCudaNaiveBatchDSADecode(q, sink, rows, 8, 1, 192, 128, oldScratch, reference);
    };
    auto fused = [&] {
        Require(FastllmCudaNaiveFusedDSAVerify(q, iq, w, sink, items, rows, true, 8, 1, 192, 128, newScratch,
                                               actual),
                "fused rejected");
    };
    old();
    Check(cudaDeviceSynchronize());
    auto expected = Read<int>(indices);
    auto ref = Read<__nv_bfloat16>(reference);
    fused();
    Check(cudaDeviceSynchronize());
    auto got = Read<int>(indices);
    auto out = Read<__nv_bfloat16>(actual);
    int different = 0, oi = 0;
    double se = 0, ss = 0;
    float maxerr = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != expected[i])
            ++different;
    for (size_t i = 0; i < out.size(); ++i) {
        float a = float(out[i]), b = float(ref[i]);
        se += (a - b) * (a - b);
        ss += b * b;
        maxerr = std::max(maxerr, std::fabs(a - b));
        oi += std::memcmp(&out[i], &ref[i], 2) != 0;
    }
    printf("past=%d rows=%d ties=%d sink=%d index_diff=%d output_diff=%d nrmse=%.8g max=%.8g\n", past, n,
           ties, sinkOn, different, oi, sqrt(se / std::max(ss, 1e-30)), maxerr);
    fflush(stdout);
    // Score comparison diagnoses rounding/order differences independently of selection.
    auto os = Read<unsigned>(oldScratch.indexScores);
    std::vector<float> ns(8 * 65536);
    if (n <= 8 && past > 2048) {
        Check(cudaMemcpy(ns.data(),
                         (char *)newScratch.dsaFused.cudaData +
                             fastllm_dsa_fused::AttnDsaScratchLayout::kIndexLogitsOffset,
                         ns.size() * 4, cudaMemcpyDeviceToHost));
        int sd = 0;
        float err = 0;
        for (int r = 0; r < n; ++r)
            for (int i = 0; i < past + r + 1; ++i) {
                unsigned bits = os[r * cap + i];
                bits = bits & 0x80000000u ? bits ^ 0x80000000u : ~bits;
                float old;
                std::memcpy(&old, &bits, 4);
                float nw = ns[r * 65536 + i];
                if (old != nw)
                    ++sd;
                err = std::max(err, std::fabs(old - nw));
            }
        printf("scores_diff=%d max=%.8g\n", sd, err);
        fflush(stdout);
    }
    // Benchmark graph replay, excluding CPU dispatch and allocation.
    auto bench = [&](auto fn) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        fn();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &g));
        Check(cudaGraphInstantiate(&e, g, 0));
        for (int i = 0; i < 10; ++i)
            Check(cudaGraphLaunch(e, cudaStreamPerThread));
        cudaEvent_t a, b;
        Check(cudaEventCreate(&a));
        Check(cudaEventCreate(&b));
        Check(cudaEventRecord(a, cudaStreamPerThread));
        for (int i = 0; i < 100; ++i)
            Check(cudaGraphLaunch(e, cudaStreamPerThread));
        Check(cudaEventRecord(b, cudaStreamPerThread));
        Check(cudaEventSynchronize(b));
        float ms;
        Check(cudaEventElapsedTime(&ms, a, b));
        Check(cudaGraphExecDestroy(e));
        Check(cudaGraphDestroy(g));
        Check(cudaEventDestroy(a));
        Check(cudaEventDestroy(b));
        return ms * 10;
    };
    printf("timing_us old=%.3f fused=%.3f\n", bench(old), bench(fused));
    fflush(stdout);
    Require(different == 0, "TopK mismatch");
    Require(oi == 0, "Attention mismatch");
}
struct PagedTensor {
    PagedCacheManager pool;
    Data table{INT32}, cache{BFLOAT16};
    explicit PagedTensor(const Data &dense, int seed, int pageLen = 128) {
        const int length = dense.dims[1], width = dense.dims[2];
        const int pages = (length + pageLen - 1) / pageLen;
        pool.dataType = BFLOAT16;
        pool.pageLen = pageLen;
        pool.Resize({pages, pageLen, 1, width});
        pool.ToDevice(DataDevice::CUDA, {0}, false);
        pool.Allocate();
        pool.SetMaxPages(pages);
        std::vector<int> ids(pages);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
        table.Resize({pages});
        table.Allocate();
        std::memcpy(table.cpuData, ids.data(), pages * sizeof(int));
        table.ToDevice(DataDevice::CUDA, {0}, true);
        for (int row = 0; row < length; row += pageLen)
            Check(cudaMemcpy((uint8_t *)pool.cudaData + (size_t)ids[row / pageLen] * pageLen * width * 2,
                             (const uint8_t *)dense.cudaData + (size_t)row * width * 2,
                             (size_t)std::min(pageLen, length - row) * width * 2, cudaMemcpyDeviceToDevice));
        cache.FakeFrom(table, 0);
        cache.dataType = BFLOAT16;
        cache.Resize(dense.dims);
        cache.isPagedKVCache = true;
        cache.pagedKVCacheData = &pool;
        cache.pageLen = pageLen;
        cache.pageIndex = ids;
        cache.kvCacheFp8Indexer = dense.kvCacheFp8Indexer;
        cache.lastPageLen = (length - 1) % pageLen + 1;
        cache.expansionDims = {1, pages * pageLen, width};
    }
};

struct Request {
    Data key{BFLOAT16}, value{BFLOAT16}, indices{INT32};
    std::unique_ptr<PagedTensor> pagedKey, pagedValue;
    int capacity, past, rows, stride;
    Request(int past, int rows, bool paged, int seed)
        : capacity((past + rows + 127) / 128 * 128), past(past), rows(rows), stride(fp8Cache ? 264 : 320) {
        rng = seed;
        Make(value, {1, capacity, 128});
        Make(indices, {rows, 2048});
        key.Resize({1, capacity, stride});
        key.Allocate();
        for (int r = 0; r < capacity; ++r) {
            auto *p = (__nv_bfloat16 *)key.cpuData + r * stride;
            for (int d = 0; d < 192; ++d)
                p[d] = __float2bfloat16(Random());
            if (fp8Cache) {
                auto *index = (__nv_fp8_e4m3 *)(p + 192);
                for (int d = 0; d < 128; ++d)
                    index[d] = __nv_fp8_e4m3(Random() * 4.f);
                float scale = .01f;
                std::memcpy((char *)index + 128, &scale, 4);
                std::memset((char *)index + 132, 0, 12);
            } else
                for (int d = 0; d < 128; ++d)
                    p[192 + d] = __float2bfloat16(Random());
        }
        key.expansionDims = key.dims;
        key.kvCacheFp8Indexer = fp8Cache;
        key.ToDevice(CUDA, {0}, true);
        if (paged) {
            pagedKey = std::make_unique<PagedTensor>(key, seed + 1);
            pagedValue = std::make_unique<PagedTensor>(value, seed + 2);
        }
    }
    Data &K() { return pagedKey ? pagedKey->cache : key; }
    Data &V() { return pagedValue ? pagedValue->cache : value; }
    const void *KData() { return pagedKey ? pagedKey->pool.cudaData : key.cudaData; }
    const void *VData() { return pagedValue ? pagedValue->pool.cudaData : value.cudaData; }
};
struct Captured {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    template <class F> void Capture(F fn) {
        fn();
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        fn();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, 0));
    }
    void Run() {
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ~Captured() {
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
static void Mixed(bool paged, bool batchedFallback = false) {
    const std::vector<int> sizes = batchedFallback ? std::vector<int>{8,8,8} : std::vector<int>{3,8,11};
    const std::vector<int> past = {16381, 8190, 24771};
    const int count = std::accumulate(sizes.begin(), sizes.end(), 0);
    Data q(BFLOAT16), iq(BFLOAT16), weights(BFLOAT16), live(INT32), bindings(INT32), sink(FLOAT32), expected,
        actual;
    Make(q, {1, count, 1536});
    Make(iq, {1, count, 2048});
    Make(weights, {1, count, 16});
    Make(live, {count});
    Make(bindings, {int(sizes.size()) * 8});
    sink.Resize({8});
    sink.Allocate();
    for (int i = 0; i < 8; ++i)
        ((float *)sink.cpuData)[i] = i * .1f;
    sink.ToDevice(CUDA, {0}, true);
    std::vector<std::unique_ptr<Request>> requests, alternate;
    std::vector<FastllmNaiveVerifyIndexerItem> items;
    std::vector<FastllmNaiveDSARow> descriptors;
    std::vector<int> lengths;
    for (size_t seq = 0, offset = 0; seq < sizes.size(); ++seq) {
        requests.emplace_back(std::make_unique<Request>(past[seq], sizes[seq], paged, 41 + seq));
        alternate.emplace_back(std::make_unique<Request>(past[seq], sizes[seq], paged, 97 + seq));
        auto &r = *requests.back();
        auto *cache = (const void *const *)bindings.cudaData + seq * 4;
        items.push_back({&r.K(), &r.indices, int(offset), sizes[seq], past[seq], r.capacity, cache});
        for (int row = 0; row < sizes[seq]; ++row) {
            FastllmNaiveDSARow d{};
            d.key = r.KData();
            d.value = r.VData();
            d.cache = cache;
            d.length = past[seq] + row + 1;
            d.capacity = r.capacity;
            d.keyStride = r.stride;
            d.indices = (int *)r.indices.cudaData + size_t(row) * 2048;
            d.selectedCount = 2048;
            d.liveKeys = (int *)live.cudaData + offset + row;
            if (paged) {
                d.pageLen = 128;
                d.poolTokens = r.pagedKey->pool.maxPages * 128;
                d.keyPages = (int *)r.K().cudaData;
                d.valuePages = (int *)r.V().cudaData;
            }
            descriptors.push_back(d);
            lengths.push_back(d.length);
        }
        offset += sizes[seq];
    }
    auto bind = [&](bool other) {
        std::vector<const void *> table;
        for (size_t i = 0; i < requests.size(); ++i) {
            auto &r = *(other ? alternate[i] : requests[i]);
            table.push_back(r.KData());
            table.push_back(r.VData());
            table.push_back(paged ? r.K().cudaData : nullptr);
            table.push_back(paged ? r.V().cudaData : nullptr);
        }
        Check(cudaMemcpy(bindings.cudaData, table.data(), table.size() * sizeof(void *),
                         cudaMemcpyHostToDevice));
    };
    bind(false);
    Check(cudaMemcpy(live.cudaData, lengths.data(), count * 4, cudaMemcpyHostToDevice));
    FastllmNaiveDecodeScratch oldScratch, scratch;
    auto old = [&] {
        Require(FastllmCudaNaiveBatchVerifyIndexer(iq, weights, items, true, oldScratch, live),
                "mixed old rejected");
        FastllmCudaNaiveBatchDSADecode(q, sink, descriptors, 8, 1, 192, 128, oldScratch, expected);
    };
    auto fused = [&] {
        const bool fused = FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, items, descriptors,
            true, 8, 1, 192, 128, scratch, actual);
        Require(fused != batchedFallback, "mixed DSA dispatch");
        if (!fused) {
            Require(FastllmCudaNaiveBatchVerifyIndexer(iq, weights, items, true, scratch, live),
                "mixed batched indexer rejected");
            FastllmCudaNaiveBatchDSADecode(q, sink, descriptors, 8, 1, 192, 128, scratch, actual);
        }
    };
    Captured a, b;
    a.Capture(old);
    b.Capture(fused);
    for (int replay = 0; replay < 5; ++replay) {
        bind(replay & 1);
        auto current = lengths;
        if (replay == 1)
            for (int i = 0; i < count; ++i)
                current[i] = i < 3 ? 0 : 121 + i;
        if (replay == 2)
            std::fill(current.begin(), current.end(), 0);
        if (replay == 3)
            for (int i = 0; i < count; ++i)
                current[i] = 2040 + i;
        if (replay == 4)
            for (int i = 0; i < count; ++i)
                if (i % 3)
                    current[i] = i % 3 == 1 ? 121 : 0;
        Check(cudaMemcpy(live.cudaData, current.data(), count * 4, cudaMemcpyHostToDevice));
        a.Run();
        auto out = Read<unsigned short>(expected);
        std::vector<std::vector<int>> selected;
        for (auto &r : requests)
            selected.push_back(Read<int>(r->indices));
        b.Run();
        Require(Read<unsigned short>(actual) == out, "mixed attention differs");
        for (size_t i = 0; i < requests.size(); ++i)
            Require(Read<int>(requests[i]->indices) == selected[i], "mixed TopK differs");
    }

    // Independent CUDA graphs/streams share read-only inputs but no scratch/output/indices.
    bind(false);
    Check(cudaMemcpy(live.cudaData, lengths.data(), count * 4, cudaMemcpyHostToDevice));
    a.Run();
    auto parallelExpected = Read<unsigned short>(expected);
    std::vector<std::vector<int>> parallelIndices;
    for (auto &r : requests)
        parallelIndices.push_back(Read<int>(r->indices));
    std::vector<std::unique_ptr<Data>> secondIndices;
    auto secondItems = items;
    auto secondRows = descriptors;
    for (size_t seq = 0, offset = 0; seq < requests.size(); ++seq) {
        secondIndices.emplace_back(std::make_unique<Data>(INT32));
        Make(*secondIndices.back(), {sizes[seq], 2048});
        secondItems[seq].indices = secondIndices.back().get();
        for (int row = 0; row < sizes[seq]; ++row)
            secondRows[offset + row].indices = (int *)secondIndices.back()->cudaData + row * 2048;
        offset += sizes[seq];
    }
    FastllmNaiveDecodeScratch secondScratch;
    Data secondOutput;
    Captured independent;
    independent.Capture([&] {
        const bool fused = FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, secondItems,
            secondRows, true, 8, 1, 192, 128, secondScratch, secondOutput);
        Require(fused != batchedFallback, "independent DSA dispatch");
        if (!fused) {
            Require(FastllmCudaNaiveBatchVerifyIndexer(iq, weights, secondItems, true, secondScratch, live),
                "independent batched indexer rejected");
            FastllmCudaNaiveBatchDSADecode(q, sink, secondRows, 8, 1, 192, 128, secondScratch, secondOutput);
        }
    });
    std::exception_ptr errors[2];
    auto worker = [&](int id, Captured &graph) {
        try {
            Check(cudaSetDevice(0));
            for (int iteration = 0; iteration < 6; ++iteration)
                graph.Run();
        } catch (...) {
            errors[id] = std::current_exception();
        }
    };
    std::thread first(worker, 0, std::ref(b)), second(worker, 1, std::ref(independent));
    first.join();
    second.join();
    for (auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    Require(Read<unsigned short>(actual) == parallelExpected &&
                Read<unsigned short>(secondOutput) == parallelExpected,
            "concurrent streams changed attention");
    for (size_t seq = 0; seq < requests.size(); ++seq)
        Require(Read<int>(requests[seq]->indices) == parallelIndices[seq] &&
                    Read<int>(*secondIndices[seq]) == parallelIndices[seq],
                "concurrent streams changed TopK");
    // Rejected geometry must leave outputs/scratch untouched.
    auto before = Read<unsigned short>(actual);
    void *arena = scratch.dsaFused.cudaData;
    Require(!FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, items, descriptors, true, 16, 1, 192, 128,
                                            scratch, actual),
            "unsupported heads admitted");
    Require(!FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, items, descriptors, false, 8, 1, 192, 128,
                                            scratch, actual),
            "unquantized indexer admitted");
    descriptors.back().fp4 = true;
    Require(!FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, items, descriptors, true, 8, 1, 192, 128,
                                            scratch, actual),
            "FP4 admitted");
    descriptors.back().fp4 = false;
    int savedCapacity = descriptors.back().capacity;
    descriptors.back().capacity = 65537;
    Require(!FastllmCudaNaiveFusedDSAVerify(q, iq, weights, sink, items, descriptors, true, 8, 1, 192, 128,
                                            scratch, actual),
            "oversized context admitted");
    descriptors.back().capacity = savedCapacity;
    Data shifted;
    shifted.Resize(q.dims);
    shifted.FakeFrom(q, 2);
    Require(!FastllmCudaNaiveFusedDSAVerify(shifted, iq, weights, sink, items, descriptors, true, 8, 1, 192,
                                            128, scratch, actual),
            "unaligned query admitted");
    Require(arena == scratch.dsaFused.cudaData && before == Read<unsigned short>(actual),
            "fallback mutated state");
    printf("MIXED paged=%d requests=3 total_rows=%d batched_fallback=%d replays=5 independent_streams=2 exact PASS\n", paged, count, batchedFallback);
    fflush(stdout);
}

int main(int argc, char **argv) {
    try {
        Check(cudaSetDevice(0));
        if (argc > 1 && std::strcmp(argv[1], "bf16") == 0) {
            fp8Cache = false;
            --argc;
            ++argv;
        }
        printf("INDEX_CACHE %s\n", fp8Cache ? "FP8" : "BF16");
        if (FastllmCudaRuntimeArch() != 103)
            return 77;
        if (argc > 1 && std::strcmp(argv[1], "mixed") == 0) {
            Mixed(false);
            Mixed(true);
            Mixed(false, true);
            Mixed(true, true);
        } else if (argc > 1)
            Test(atoi(argv[1]), argc > 2 ? atoi(argv[2]) : 8, false, true);
        else {
            for (int p : {121, 2040, 2048, 8192, 16384, 32760})
                for (int n : {1, 3, 8, 11})
                    Test(p, n, false, true);
            for (int p : {2048, 16384, 32760})
                Test(p, 8, true, false);
            Test(65528, 8, false, true);
            Test(65512, 24, false, true);
            Mixed(false);
            Mixed(true);
            Mixed(false, true);
            Mixed(true, true);
        }
        puts("FUSED DSA PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
