#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "models/naive_n05_flash.h"
#include "devices/cuda/fastllm-cuda-kvmem.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace fastllm {
// Exercise the real shared-prefix tail-copy path, including the FP8 Indexer
// sidecar. The page table is preallocated so no model workspace is needed.
struct NaiveN05FP4KVTestAccess : NaiveN05FlashModel {
    void Copy(Data &cache, PagedCacheManager &pool, PagedCacheManager &index) {
        max_positions = 384;
        tpIndexerSidecars.emplace(&pool, &index);
        if (!PreparePagedCache(cache, pool, 64))
            throw std::runtime_error("shared prefix did not copy on write");
    }
};
}
namespace {
void Require(bool v, const char *s) {
    if (!v)
        throw std::runtime_error(s);
}
void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
void LengthMetadata() {
    // Graph bindings retain the length address and physical row stride while
    // logical lengths grow. The shared helper must preserve the FP4 layout.
    for (DataType type : {BFLOAT16, FP8_E4M3, FP4_E2M1}) {
        Data cache(type, {1, 17, 192});
        cache.expansionDims = {1, 128, 192};
        cache.strides = {128 * 192, 192, 1};
        const auto strides = cache.strides;
        int *length = &cache.dims[1];
        Require(FastllmNaiveCanResizeKVLength(cache, 127, 192), "KV length fast path rejected supported layout");
        FastllmNaiveResizeKVLength(cache, 127, 192);
        Require(length == &cache.dims[1] && *length == 127 && cache.strides == strides,
                "KV length update changed graph metadata storage");
        Require(cache.GetBytes() == GetDataBytes(type, 1, 128 * 192), "KV length update changed physical storage bytes");
        Require(!FastllmNaiveCanResizeKVLength(cache, 129, 192), "KV length fast path exceeded capacity");
        Require(!FastllmNaiveCanResizeKVLength(cache, 127, 128), "KV length fast path ignored width change");
        cache.cudaNativeNvfp4Layout = true;
        Require(!FastllmNaiveCanResizeKVLength(cache, 127, 192), "native FP4 layout entered row-packed fast path");
    }
}
void Allocate(Data &d, const std::vector<int> &shape) {
    d.UpdateUnitSize();
    d.Resize(shape);
    d.ToDevice(DataDevice::CUDA, {0}, false);
    d.Allocate(true);
}
template <class T> std::vector<T> Read(const Data &d) {
    std::vector<T> out(d.GetBytes() / sizeof(T));
    Cuda(cudaMemcpy(out.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return out;
}
float Float(uint16_t bits) {
    unsigned u = unsigned(bits) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}
std::vector<uint16_t> Input(int count, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> uniform(-2, 2);
    std::vector<uint16_t> x(count);
    for (int i = 0; i < count; ++i)
        x[i] = Float32ToBFloat16RNEBits(i < 16 ? 0.f : uniform(gen));
    return x;
}
void Upload(Data &d, const std::vector<uint16_t> &x) {
    Require(d.GetBytes() == x.size() * 2, "upload shape");
    Cuda(cudaMemcpy(d.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
}
// Independent nearest-value reference; ties choose the even E2M1 code.
constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
uint8_t Encode(float x) {
    float best = INFINITY;
    int code = 0;
    for (int i = 0; i < 8; ++i) {
        float error = std::abs(std::abs(x) - levels[i]);
        if (error < best || (error == best && !(i & 1))) {
            best = error;
            code = i;
        }
    }
    return uint8_t(code | (std::signbit(x) ? 8 : 0));
}
std::vector<uint8_t> Quantize(const uint16_t *input, int width) {
    std::vector<uint8_t> out(width / 16 * 9);
    for (int block = 0; block < width / 16; ++block) {
        float maximum = 0;
        for (int i = 0; i < 16; ++i)
            maximum = std::max(maximum, std::abs(Float(input[block * 16 + i])));
        __nv_fp8_e4m3 scale(maximum / 6.f);
        out[width / 2 + block] = scale.__x;
        float inv = float(scale) > 0 ? 1.f / float(scale) : 0.f;
        for (int i = 0; i < 8; ++i)
            out[block * 8 + i] = Encode(Float(input[block * 16 + i * 2]) * inv) |
                                 (Encode(Float(input[block * 16 + i * 2 + 1]) * inv) << 4);
    }
    return out;
}
struct Cache {
    int length, width, rowBytes, pageLen = 128;
    Data table{INT32}, view{FP4_E2M1}, dense{BFLOAT16};
    PagedCacheManager pool;
    std::vector<int> ids;
    std::vector<uint8_t> expected;
    Cache(int length, int width, int seed) : length(length), width(width), rowBytes(width / 16 * 9) {
        const int pages = (length + pageLen - 1) / pageLen;
        ids.resize(pages);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
        pool.dataType = FP4_E2M1;
        pool.pageLen = pageLen;
        Allocate(pool, {pages, pageLen, 1, width});
        pool.SetMaxPages(pages);
        Allocate(table, {pages});
        Cuda(cudaMemcpy(table.cudaData, ids.data(), pages * 4, cudaMemcpyHostToDevice));
        view.FakeFrom(table, 0);
        view.dataType = FP4_E2M1;
        view.UpdateUnitSize();
        view.Resize({1, length, width});
        view.isPagedKVCache = true;
        view.pagedKVCacheData = &pool;
        view.pageLen = pageLen;
        view.pageIndex = ids;
        view.lastPageLen = (length - 1) % pageLen + 1;
        view.expansionDims = {1, pages * pageLen, width};
        expected.resize(pool.GetBytes());
        Allocate(dense, {1, length, width});
        Require(pool.GetBytes() == size_t(pages) * pageLen * rowBytes, "FP4 byte accounting");
        auto input = Input(length * width, seed);
        Data data(BFLOAT16);
        Allocate(data, {1, length, width});
        Upload(data, input);
        FastllmCudaNaivePagedWrite(view, data, 0);
        ReferenceWrite(input, 0);
        Verify();
        RefreshDense();
    }
    void ReferenceWrite(const std::vector<uint16_t> &input, int start) {
        for (int row = 0; row < int(input.size()) / width; ++row) {
            auto bytes = Quantize(input.data() + row * width, width);
            const int logical = start + row, physical = ids[logical / pageLen] * pageLen + logical % pageLen;
            std::copy(bytes.begin(), bytes.end(), expected.begin() + size_t(physical) * rowBytes);
        }
    }
    void Verify() { Require(Read<uint8_t>(pool) == expected, "FP4 packed bytes/scales/page address"); }
    std::vector<uint16_t> Decode() const {
        std::vector<uint16_t> out(size_t(length) * width);
        for (int r = 0; r < length; ++r) {
            const auto *p = expected.data() + size_t(ids[r / pageLen] * pageLen + r % pageLen) * rowBytes;
            for (int c = 0; c < width; ++c) {
                unsigned code = (p[c / 2] >> ((c & 1) * 4)) & 15;
                __nv_fp8_e4m3 scale;
                scale.__x = p[width / 2 + c / 16];
                float v = levels[code & 7] * (code & 8 ? -1.f : 1.f) * float(scale);
                out[size_t(r) * width + c] = Float32ToBFloat16RNEBits(v);
            }
        }
        return out;
    }
    void RefreshDense() { Upload(dense, Decode()); }
};
struct Graph {
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> reserved;
    template <class F> void Capture(F fn) {
        fn();
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare graph");
        Require(FastllmCudaGraphMemoryPoolBegin(), "graph pool");
        Require(FastllmCudaGraphBeginCapture(), "capture begin");
        fn();
        Require(FastllmCudaGraphEndCapture(&graph), FastllmCudaGraphLastError());
        Require(FastllmCudaGraphMemoryPoolEnd(reserved), "graph pool end");
        Require(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
    }
    void Run() {
        Require(FastllmCudaGraphLaunch(exec), "replay");
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ~Graph() {
        FastllmCudaGraphExecDestroy(exec);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
    }
};
void Near(const Data &a, const Data &b) {
    auto x = Read<uint16_t>(a), y = Read<uint16_t>(b);
    Require(x.size() == y.size(), "attention shape");
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double delta = Float(x[i]) - Float(y[i]);
        Require(std::isfinite(delta), "nonfinite attention");
        error += delta * delta;
        norm += double(Float(y[i])) * Float(y[i]);
        maximum = std::max(maximum, std::abs(delta));
    }
    Require(std::sqrt(error / std::max(norm, 1e-30)) <= .004 && maximum <= .004,
            "attention differs from dequantized BF16 reference");
}
void PrefixCopy() {
    for (int width : {128, 192}) {
        PagedCacheManager pool;
        pool.dataType = FP4_E2M1;
        pool.pageLen = 128;
        Allocate(pool, {3, 128, 1, width});
        pool.SetMaxPages(3);
        PagedCacheManager index;
        index.dataType = FP8_E4M3;
        Data cache(INT32);
        Allocate(index, {3, 128, 1, 128});
        Allocate(cache, {3});
        std::vector<uint8_t> original(pool.GetBytes()), sidecar(index.GetBytes());
        for (size_t i = 0; i < original.size(); ++i)
            original[i] = uint8_t(i * 37 + i / 257);
        for (size_t i = 0; i < sidecar.size(); ++i)
            sidecar[i] = uint8_t(i * 17 + i / 193);
        Cuda(cudaMemcpy(pool.cudaData, original.data(), original.size(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(index.cudaData, sidecar.data(), sidecar.size(), cudaMemcpyHostToDevice));
        const int old = pool.GetUnusedPageIndex(true);
        pool.pageRefCount[old] = 2;
        cache.dataType = FP4_E2M1;
        cache.UpdateUnitSize();
        cache.Resize({1, 63, width});
        cache.isPagedKVCache = true;
        cache.pagedKVCacheData = &pool;
        cache.pageLen = 128;
        cache.pageIndex = {old};
        NaiveN05FP4KVTestAccess model;
        model.Copy(cache, pool, index);
        const int copied = cache.pageIndex[0];
        Require(copied != old, "shared prefix reused the source page");
        const size_t bytes = 128 * width / 16 * 9, indexBytes = 128 * 128;
        std::copy_n(original.begin() + old * bytes, bytes, original.begin() + copied * bytes);
        std::copy_n(sidecar.begin() + old * indexBytes, indexBytes, sidecar.begin() + copied * indexBytes);
        Require(Read<uint8_t>(pool) == original, "prefix copy damaged FP4 values/scales or adjacent pages");
        Require(Read<uint8_t>(index) == sidecar, "prefix copy damaged FP8 Indexer sidecar");
        int devicePage = -1;
        Cuda(cudaMemcpy(&devicePage, cache.cudaData, 4, cudaMemcpyDeviceToHost));
        Require(devicePage == copied && pool.pageRefCount[old] == 1 && pool.pageRefCount[copied] == 1,
                "prefix copy metadata/refcounts");
        // Data owns the page table; the test owns physical-page references.
        pool.ReleasePageIndex(old);
        pool.ReleasePageIndex(copied);
        cache.pageIndex.clear();
    }
}
void ReadWrite() {
    for (int width : {128, 192, 384}) {
        Cache c(513, width, width);
        Data input(BFLOAT16);
        Allocate(input, {1, 131, width});
        auto x = Input(131 * width, 41);
        Upload(input, x);
        FastllmCudaNaivePagedWrite(c.view, input, 127);
        c.ReferenceWrite(x, 127);
        c.Verify();
        Data restored(BFLOAT16);
        Allocate(restored, {1, 513, width});
        FastllmCudaNaiveReadCache({{&c.view, restored.cudaData, 0, 513}});
        Require(Read<uint16_t>(restored) == c.Decode(), "FP4 paged prefill/history decode");
    }
}
void DSA(int batch, int length, bool sparse, int heads, bool withSink = false, bool uniform = false) {
    Cache k(length, 192, 71), v(length, 128, 79), k2(length, 192, 83), v2(length, 128, 89);
    Data q(BFLOAT16), live(INT32), indices(INT32), table(INT32), a, b, sink;
    Allocate(q, {1, batch, heads * 192});
    Upload(q, uniform ? std::vector<uint16_t>(batch * heads * 192, 0) : Input(batch * heads * 192, 97));
    if (withSink) {
        Allocate(sink, {heads});
        std::vector<float> values(heads);
        for (int h = 0; h < heads; ++h) values[h] = (h - heads / 2) * 0.35f;
        Cuda(cudaMemcpy(sink.cudaData, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    Allocate(live, {batch});
    Allocate(indices, {batch, 2048});
    Allocate(table, {batch * 4 * int(sizeof(void *) / 4)});
    std::vector<int> selected(batch * 2048);
    for (size_t i = 0; i < selected.size(); ++i)
        selected[i] = i % 17 == 0 ? -1 : i % 29 == 0 ? length + 1 : int(i * 37 % length);
    Cuda(cudaMemcpy(indices.cudaData, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));
    std::vector<FastllmNaiveDSARow> packed(batch), reference(batch);
    auto bind = [&](Cache &key, Cache &value, bool padded) {
        std::vector<const void *> pointers;
        std::vector<int> lengths(batch, length);
        if (padded)
            lengths.back() = 0;
        for (int i = 0; i < batch; ++i) {
            pointers.insert(pointers.end(), {key.pool.cudaData, value.pool.cudaData, key.table.cudaData,
                                             value.table.cudaData});
            auto &r = reference[i];
            r = packed[i];
            r.cache = nullptr;
            r.key = key.dense.cudaData;
            r.value = value.dense.cudaData;
            r.keyPages = r.valuePages = nullptr;
            r.pageLen = 0;
            r.fp4 = false;
        }
        Cuda(cudaMemcpy(table.cudaData, pointers.data(), pointers.size() * sizeof(void *),
                        cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), lengths.size() * 4, cudaMemcpyHostToDevice));
    };
    for (int i = 0; i < batch; ++i) {
        auto &r = packed[i];
        r.key = k.pool.cudaData;
        r.value = v.pool.cudaData;
        r.keyStride = 192;
        r.length = r.capacity = length;
        r.fp4 = true;
        r.pageLen = 128;
        r.poolTokens = k.ids.size() * 128;
        r.cache = (const void *const *)table.cudaData + i * 4;
        r.liveKeys = (const int *)live.cudaData + i;
        if (sparse) {
            r.indices = (const int *)indices.cudaData + i * 2048;
            r.selectedCount = 2048;
        }
    }
    bind(k, v, false);
    FastllmNaiveDecodeScratch sa, sb;
    Graph graph;
    graph.Capture([&] { FastllmCudaNaiveBatchDSADecode(q, sink, packed, heads, 1, 192, 128, sa, a); });
    for (int pass = 0; pass < 2; ++pass) {
        bind(pass ? k2 : k, pass ? v2 : v, pass != 0);
        graph.Run();
        FastllmCudaNaiveBatchDSADecode(q, sink, reference, heads, 1, 192, 128, sb, b);
        Near(a, b);
        if (FastllmCudaRuntimeArch() == 90 && sparse && heads == 16 && batch >= 32) {
            // Tensor Core accumulation may differ from the scalar fallback.
            // Keep the existing bounded-error check against both references,
            // including graph rebind, descriptor boundaries and inactive rows.
            std::vector<int> checked = {0, 31, 32, 255, 256, batch / 2, batch - 1};
            std::sort(checked.begin(), checked.end());
            checked.erase(std::unique(checked.begin(), checked.end()), checked.end());
            for (int row : checked) {
                if (row >= batch) continue;
                Data singleQuery(BFLOAT16, {1, 1, heads * 192}), singleOutput;
                singleQuery.FakeFrom(q, size_t(row) * heads * 192 * 2);
                FastllmNaiveDecodeScratch singleScratch;
                FastllmCudaNaiveBatchDSADecode(singleQuery, sink, {packed[row]},
                    heads, 1, 192, 128, singleScratch, singleOutput);
                Data actualRow(BFLOAT16, {1, 1, heads * 128});
                actualRow.FakeFrom(a, size_t(row) * heads * 128 * 2);
                Near(actualRow, singleOutput);
            }
        }
    }
}
void Prefill() {
    constexpr int length = 8192, queries = 2048, heads = 16;
    Cache k(length, 192, 701), v(length, 128, 709);
    Data q(BFLOAT16), indices(INT32), a, b, sink;
    Allocate(q, {1, queries, heads * 192});
    Upload(q, Input(queries * heads * 192, 719));
    Allocate(indices, {queries, 2048});
    std::vector<int> selected(queries * 2048);
    for (size_t i = 0; i < selected.size(); ++i)
        selected[i] = i % 17 == 0 ? -1 : i % 29 == 0 ? length + 1 : int(i * 37 % length);
    Cuda(cudaMemcpy(indices.cudaData, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));
    FastllmCudaNaiveAttention(q, k.view, v.view, indices, sink, heads, 1, 192, 128,
        length - queries, 0, a, true);
    // Subdivide only queries so the dequantized reference retains the
    // established fallback, independent of the large-prefill dispatch.
    b.dataType = BFLOAT16;
    Allocate(b, {1, queries, heads * 128});
    for (int first = 0; first < queries; first += 512) {
        Data qr(BFLOAT16, {1, 512, heads * 192}), ir(INT32, {512, 2048});
        Data result(BFLOAT16, {1, 512, heads * 128});
        qr.FakeFrom(q, size_t(first) * heads * 192 * 2);
        ir.FakeFrom(indices, size_t(first) * 2048 * 4);
        result.FakeFrom(b, size_t(first) * heads * 128 * 2);
        FastllmCudaNaiveAttention(qr, k.dense, v.dense, ir, sink, heads, 1, 192, 128,
            length - queries + first, 0, result, true);
    }
    Near(a, b);
    Require(Read<uint16_t>(a) == Read<uint16_t>(b),
            "FP4 prefill changed dequantized fallback bits");
}
void RopeGraph() {
    Cache key(512, 192, 101), value(512, 128, 103), otherKey(512, 192, 107), otherValue(512, 128, 109);
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), ik(BFLOAT16), pos(FLOAT32), live(INT32), bindings(INT32);
    Allocate(q, {1, 1, 8 * 192});
    Allocate(k, {1, 1, 192});
    Allocate(v, {1, 1, 128});
    Allocate(ik, {1, 1, 128});
    auto hq = Input(8 * 192, 111), hk = Input(192, 113), hv = Input(128, 127), hi = Input(128, 131);
    Upload(q, hq);
    Upload(k, hk);
    Upload(v, hv);
    Upload(ik, hi);
    Allocate(pos, {1});
    Allocate(live, {1});
    Allocate(bindings, {8 * int(sizeof(void *) / 4)});
    PagedCacheManager index;
    index.dataType = FP8_E4M3;
    index.pageLen = 128;
    index.SetMaxPages(4);
    Allocate(index, {4, 128, 1, 128});
    Data indexView;
    indexView.FakeFrom(key.table, 0);
    indexView.dataType = FP8_E4M3;
    indexView.UpdateUnitSize();
    indexView.Resize({1, 512, 128});
    indexView.isPagedKVCache = true;
    indexView.pagedKVCacheData = &index;
    indexView.pageLen = 128;
    indexView.expansionDims = {1, 512, 128};
    std::vector<FastllmNaiveRopeCacheRow> rows{
        {&key.view, &value.view, &ik, (const void *const *)bindings.cudaData, &indexView}};
    auto bind = [&](Cache &a, Cache &b, int length) {
        const void *p[8] = {a.pool.cudaData, b.pool.cudaData, a.table.cudaData, b.table.cudaData,
                            index.cudaData};
        Cuda(cudaMemcpy(bindings.cudaData, p, sizeof(p), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, &length, 4, cudaMemcpyHostToDevice));
    };
    bind(key, value, 0);
    Graph graph;
    graph.Capture([&] {
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, pos, rows, live, 8, 1, 192, 128, 64, 1e7f,
                                                     .707f, 0),
                "FP4 fused batch RoPE rejected");
    });
    for (int pass = 0; pass < 3; ++pass) {
        Cache &a = pass % 2 ? otherKey : key, &b = pass % 2 ? otherValue : value;
        int row = pass == 0 ? 0 : pass == 1 ? 127 : 256;
        float position = 8192.f + row;
        Cuda(cudaMemcpy(pos.cudaData, &position, 4, cudaMemcpyHostToDevice));
        Upload(q, hq);
        bind(a, b, row + 1);
        graph.Run();
        Data qr(BFLOAT16), kr(BFLOAT16), vr(BFLOAT16), ir(BFLOAT16);
        Allocate(qr, q.dims);
        Allocate(kr, k.dims);
        Allocate(vr, v.dims);
        Allocate(ir, ik.dims);
        Upload(qr, hq);
        Upload(kr, hk);
        Upload(vr, hv);
        Upload(ir, hi);
        FastllmCudaNaiveRopeQKScaleV(qr, kr, vr, pos, 8, 1, 192, 128, 64, 1e7f, .707f);
        FastllmCudaNaiveRope(ir, pos, 1, 128, 64, 1e7f);
        Require(Read<uint16_t>(q) == Read<uint16_t>(qr), "FP4 RoPE changed Q");
        a.ReferenceWrite(Read<uint16_t>(kr), row);
        b.ReferenceWrite(Read<uint16_t>(vr), row);
        a.Verify();
        b.Verify();
        auto indexBits = Read<uint8_t>(index);
        auto rotated = Read<uint16_t>(ir);
        int physical = a.ids[row / 128] * 128 + row % 128;
        for (int c = 0; c < 128; ++c)
            Require(indexBits[physical * 128 + c] == __nv_fp8_e4m3(Float(rotated[c])).__x,
                    "Indexer must remain FP8 and follow dynamic K page table");
    }
}
// Cover the later >=80-request RoPE dispatch and context-sharded Indexer.
// Each row owns distinct pages, so accidental BF16 stores or cross-row writes
// cannot hide behind identical inputs or aliased cache pointers.
void RopeBatch(int batch, int contextRank, bool packed = false, bool scaled = false) {
    constexpr int heads = 16, indexPage = 32;
    const int indexWidth = scaled ? 144 : 128; // 128 FP8 values, FP32 scale and padding.
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), ik(BFLOAT16), pos(FLOAT32), live(INT32), bindings(INT32);
    Allocate(q, {1, batch, heads * 192});
    Allocate(k, {1, batch, 192}); Allocate(v, {1, batch, 128});
    Allocate(ik, {1, batch, 128}); Allocate(pos, {batch}); Allocate(live, {batch});
    Allocate(bindings, {batch * 5 * int(sizeof(void *) / 4)});
    auto hq = Input(batch * heads * 192, 191), hk = Input(batch * 192, 193);
    auto hv = Input(batch * 128, 197), hi = Input(batch * 128, 199);
    Upload(q, hq); Upload(k, hk); Upload(v, hv); Upload(ik, hi);
    Data qkv(BFLOAT16);
    if (packed) {
        const int width = (heads + 1) * 192 + 128;
        Allocate(qkv, {1, batch, width});
        std::vector<uint16_t> input(batch * width);
        for (int b = 0; b < batch; ++b) {
            auto start = input.begin() + b * width;
            std::copy_n(hq.begin() + b * heads * 192, heads * 192, start);
            std::copy_n(hk.begin() + b * 192, 192, start + heads * 192);
            std::copy_n(hv.begin() + b * 128, 128, start + (heads + 1) * 192);
        }
        Upload(qkv, input);
    }
    std::vector<std::unique_ptr<Cache>> keys, values;
    std::vector<std::unique_ptr<PagedCacheManager>> indexes;
    std::vector<Data> indexViews(batch), indexInputs(batch);
    std::vector<FastllmNaiveRopeCacheRow> rows;
    std::vector<const void *> pointers;
    std::vector<int> lengths(batch);
    std::vector<float> positions(batch);
    for (int b = 0; b < batch; ++b) {
        keys.emplace_back(new Cache(256, 192, 211 + b));
        values.emplace_back(new Cache(256, 128, 311 + b));
        indexes.emplace_back(new PagedCacheManager());
        auto &index = *indexes.back();
        index.dataType = FP8_E4M3; index.pageLen = indexPage; index.SetMaxPages(2);
        Allocate(index, {2, indexPage, 1, indexWidth});
        auto &view = indexViews[b];
        view.FakeFrom(keys[b]->table, 0); view.dataType = FP8_E4M3; view.UpdateUnitSize();
        view.Resize({1, 64, indexWidth}); view.expansionDims = {1, 64, indexWidth};
        view.kvCacheFp8Indexer = scaled;
        view.isPagedKVCache = true; view.pagedKVCacheData = &index; view.pageLen = indexPage;
        indexInputs[b].FakeFrom(ik, size_t(b) * 128 * 2);
        indexInputs[b].Resize({1, 1, 128});
        rows.push_back({&keys[b]->view, &values[b]->view, &indexInputs[b],
            (const void *const *)bindings.cudaData + b * 5, &view, contextRank});
        pointers.insert(pointers.end(), {keys[b]->pool.cudaData, values[b]->pool.cudaData,
            keys[b]->table.cudaData, values[b]->table.cudaData, index.cudaData});
        lengths[b] = 128 + b % 128; positions[b] = 8192.f + lengths[b] - 1;
    }
    Cuda(cudaMemcpy(bindings.cudaData, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(live.cudaData, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(pos.cudaData, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice));
    Graph graph;
    graph.Capture([&] {
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, pos, rows, live,
            heads, 1, 192, 128, 64, 1e7f, .707f, 0, packed ? &qkv : nullptr), "FP4 batch/context RoPE rejected");
    });
    Data qr(BFLOAT16), kr(BFLOAT16), vr(BFLOAT16), ir(BFLOAT16);
    Allocate(qr, q.dims); Allocate(kr, k.dims); Allocate(vr, v.dims); Allocate(ir, ik.dims);
    std::vector<std::vector<uint8_t>> expectedIndex(batch);
    for (int b = 0; b < batch; ++b) expectedIndex[b].resize(indexes[b]->GetBytes());
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            // Rebind the captured graph, cross page boundaries and remove rows.
            for (int b = 0; b < batch; ++b) {
                const int target = (b + 1) % batch;
                lengths[b] = b % 7 ? 1 + (b * 37) % 255 : 0;
                positions[b] = 70000.f + lengths[b];
                const void *binding[5] = {keys[target]->pool.cudaData, values[target]->pool.cudaData,
                    keys[target]->table.cudaData, values[target]->table.cudaData, indexes[target]->cudaData};
                for (int j = 0; j < 5; ++j) pointers[b * 5 + j] = lengths[b] ? binding[j] : nullptr;
            }
            Cuda(cudaMemcpy(bindings.cudaData, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(live.cudaData, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(pos.cudaData, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        Upload(q, hq); graph.Run();
        Upload(qr, hq); Upload(kr, hk); Upload(vr, hv); Upload(ir, hi);
        FastllmCudaNaiveRopeQKScaleV(qr, kr, vr, pos, heads, 1, 192, 128, 64, 1e7f, .707f);
        FastllmCudaNaiveRope(ir, pos, 1, 128, 64, 1e7f);
        auto expectedQ = Read<uint16_t>(qr);
        for (int b = 0; b < batch; ++b) if (!lengths[b])
            std::fill_n(expectedQ.begin() + b * heads * 192, heads * 192, 0);
        Require(Read<uint16_t>(q) == expectedQ, "FP4 batch RoPE changed Q");
        auto rotatedK = Read<uint16_t>(kr), scaledV = Read<uint16_t>(vr), rotatedIndex = Read<uint16_t>(ir);
        for (int b = 0; b < batch; ++b) {
            if (!lengths[b]) continue;
            const int target = pass ? (b + 1) % batch : b, row = lengths[b] - 1;
            keys[target]->ReferenceWrite({rotatedK.begin() + b * 192, rotatedK.begin() + (b + 1) * 192}, row);
            values[target]->ReferenceWrite({scaledV.begin() + b * 128, scaledV.begin() + (b + 1) * 128}, row);
            if (row % 128 / indexPage == contextRank) {
                const int physical = keys[target]->ids[row / 128] * indexPage + row % indexPage;
                auto *expected = expectedIndex[target].data() + physical * indexWidth;
                float scale = 1.f;
                if (scaled) {
                    float maximum = 0.f;
                    for (int c = 0; c < 128; ++c) maximum = std::max(maximum, std::abs(Float(rotatedIndex[b * 128 + c])));
                    scale = std::max(maximum, 1e-4f) / 448.f;
                    std::memcpy(expected + 128, &scale, sizeof(scale));
                }
                for (int c = 0; c < 128; ++c) {
                    float x = Float(rotatedIndex[b * 128 + c]) / scale;
                    expected[c] = __nv_fp8_e4m3(scaled ? std::max(-448.f, std::min(448.f, x)) : x).__x;
                }
            }
        }
        for (int b = 0; b < batch; ++b) {
            keys[b]->Verify(); values[b]->Verify();
            Require(Read<uint8_t>(*indexes[b]) == expectedIndex[b], "FP4 changed sharded Indexer or adjacent rows");
        }
    }
    std::printf("PASS FP4 RoPE batch=%d rank=%d packed=%d scaled=%d with graph rebind\n", batch, contextRank, packed, scaled);
}
void SparseRoundTrip() {
    Cache key(512, 192, 137), value(512, 128, 139);
    KvMemSparseConfig c;
    c.maxRequests = c.maxBatch = 1;
    c.maxTokens = c.totalTokens = 512;
    c.residentTokens = c.maxSelected = 16;
    c.pageLen = 128;
    c.keyRowBytes = key.rowBytes;
    c.valueRowBytes = value.rowBytes;
    c.hostBudgetBytes = 512 * (key.rowBytes + value.rowBytes);
    KvMemSparseCache cache(c);
    std::memcpy(cache.HostKey(0), key.expected.data(), key.expected.size());
    std::memcpy(cache.HostValue(0), value.expected.data(), value.expected.size());
    Cuda(cudaMemcpy(cache.KeyPages(0), key.ids.data(), key.ids.size() * 4, cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(cache.ValuePages(0), value.ids.data(), value.ids.size() * 4, cudaMemcpyHostToDevice));
    Data requests(INT32), lengths(INT32), selection(INT32), count(INT32);
    Allocate(requests, {1});
    Allocate(lengths, {1});
    Allocate(selection, {16});
    Allocate(count, {1});
    int length = 512, n = 16;
    Cuda(cudaMemcpy(lengths.cudaData, &length, 4, cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(count.cudaData, &n, 4, cudaMemcpyHostToDevice));
    Data q(BFLOAT16), output, referenceOutput, sink;
    Allocate(q, {1, 1, 8 * 192});
    Upload(q, Input(8 * 192, 149));
    FastllmNaiveDecodeScratch scratch, referenceScratch;
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<int> selected(16);
        for (int i = 0; i < 16; ++i)
            selected[i] = (pass == 1 ? 256 : 120) + i;
        Cuda(cudaMemcpy(selection.cudaData, selected.data(), 64, cudaMemcpyHostToDevice));
        cache.Resolve((const int *)requests.cudaData, (const int *)lengths.cudaData,
                      (const int *)selection.cudaData, (const int *)count.cudaData, 1, cudaStreamPerThread);
        const void *ptr[2];
        int slots[16];
        unsigned errors = 1;
        Cuda(cudaMemcpy(ptr, cache.CachePointers(), sizeof(ptr), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(slots, cache.Indices(), sizeof(slots), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(&errors, cache.DeviceErrors(), 4, cudaMemcpyDeviceToHost));
        Require(!errors, "sparse FP4 cache error");
        FastllmNaiveDSARow hot;
        hot.key = key.pool.cudaData; hot.value = value.pool.cudaData;
        hot.keyStride = 192; hot.length = hot.capacity = 16; hot.fp4 = true;
        hot.cache = cache.CachePointers(); hot.indices = cache.Indices();
        hot.liveKeys = cache.Lengths(); hot.selectedCount = 16;
        auto reference = hot;
        reference.key = key.dense.cudaData; reference.value = value.dense.cudaData;
        reference.length = reference.capacity = 512; reference.fp4 = false;
        reference.cache = nullptr; reference.liveKeys = nullptr;
        reference.indices = (const int *)selection.cudaData;
        FastllmCudaNaiveBatchDSADecode(q, sink, {hot}, 8, 1, 192, 128, scratch, output);
        FastllmCudaNaiveBatchDSADecode(q, sink, {reference}, 8, 1, 192, 128, referenceScratch, referenceOutput);
        Near(output, referenceOutput);
        for (int i = 0; i < 16; ++i)
            for (int part = 0; part < 2; ++part) {
                auto &source = part ? value : key;
                std::vector<uint8_t> actual(source.rowBytes);
                Cuda(cudaMemcpy(actual.data(), (const uint8_t *)ptr[part] + slots[i] * source.rowBytes,
                                actual.size(), cudaMemcpyDeviceToHost));
                int row = source.ids[selected[i] / 128] * 128 + selected[i] % 128;
                Require(std::equal(actual.begin(), actual.end(),
                                   source.expected.begin() + size_t(row) * source.rowBytes),
                        "eviction/refill changed FP4 payload or scales");
            }
    }
}
} // namespace
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        FastllmCudaSetDevice(0);
        LengthMetadata();
        ReadWrite();
        Prefill();
        PrefixCopy();
        RopeGraph();
        for (int rank = 0; rank < 4; ++rank) RopeBatch(128, rank);
        for (int rank = 0; rank < 4; ++rank) RopeBatch(320, rank, true, true);
        for (int batch : {79, 80, 193, 448, 449}) RopeBatch(batch, 0, true, true);
        SparseRoundTrip();
        for (int heads : {8, 16}) for (int batch : {1, 4, 7, 32, 33, 64, 256, 257, 360, 380}) {
            DSA(batch, 257, false, heads);
            DSA(batch, 4097, true, heads);
        }
        DSA(449, 4097, true, 16); // Cross the compact descriptor boundary.
        DSA(64, 8193, true, 16, true);
        DSA(64, 8193, true, 16, true, true);
        Cuda(cudaDeviceSynchronize());
        std::puts("ALL FP4 DSA CHECKS PASSED");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FP4 DSA FAIL: %s\n", e.what());
        return 1;
    }
}
