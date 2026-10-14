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
static void FP8BatchIndexer(int batch, bool paged, int width = 320) {
    const int length = 4097;
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
        Require(std::equal(x.begin(), x.begin() + length - i % 7, y.begin()), "FP8 indexer scores mismatch");
    }
    printf("FP8 BATCH INDEXER PASS batch=%d paged=%d stride=%d\n", batch, paged, width);
    fflush(stdout);
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
        FP8Rope(false, 0);
        FP8Rope(true, 0);
        FP8Rope(false, 128);
        printf("ALL FP8 CHECKS PASSED %d\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
