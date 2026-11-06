// Hopper QKV/RoPE fusion regression: exact dyadic projection, shared graph
// workspace, mutable page tables, ring wrap, and rejection without side effects.
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace fastllm;
namespace {
void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Init(Data &x, const std::vector<int> &dims) {
    x.Resize(dims);
    x.ToDevice(DataDevice::CUDA, {0}, false);
    x.Allocate(false);
}
void Fill(Data &x, unsigned seed) {
    std::vector<__nv_bfloat16> values(x.Count(0));
    // Every product and FP32 partial sum is exactly representable. This tests
    // the fused epilogue bitwise without hiding errors behind a GEMM tolerance.
    for (auto &v : values) {
        seed = seed * 1664525u + 1013904223u;
        v = __float2bfloat16_rn((int(seed >> 28) - 8) / 16.f);
    }
    Check(cudaMemcpy(x.cudaData, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
}
std::vector<uint8_t> Read(const Data &x) {
    std::vector<uint8_t> result(x.GetBytes());
    Check(cudaMemcpy(result.data(), x.cudaData, result.size(), cudaMemcpyDeviceToHost));
    return result;
}
struct Case {
    int batch, capacity, pageLen, width;
    bool ring;
    Data x{BFLOAT16}, q{BFLOAT16}, k{BFLOAT16}, v{BFLOAT16}, packed{BFLOAT16}, index{BFLOAT16};
    Data positions{FLOAT32}, lengths{INT32}, pages{INT32}, table{INT32};
    Data key{BFLOAT16}, value{BFLOAT16}, otherKey{BFLOAT16}, otherValue{BFLOAT16};
    PagedCacheManager keyPool, valuePool;
    std::vector<std::unique_ptr<Data>> keys, values, indices;
    std::vector<FastllmNaiveRopeCacheRow> rows;
    cudaGraphExec_t graph = nullptr;
    Case(int batch, bool ring, int pageLen = 64, DataType cacheType = BFLOAT16)
        : batch(batch), capacity(ring ? 128 : pageLen * 4), pageLen(ring ? 0 : pageLen),
          width(ring ? 192 : 320), ring(ring) {
        Init(x, {1, batch, 4096}); Init(q, {1, batch, 1536});
        Init(k, {1, batch, 192}); Init(v, {1, batch, 128});
        Init(packed, {1, batch, 1856}); Init(index, {1, batch, 128});
        Init(positions, {1, batch}); Init(lengths, {batch});
        Init(pages, {batch, 8}); Init(table, {batch, 8});
        for (Data *cache : {&key, &value, &otherKey, &otherValue}) {
            cache->dataType = cacheType;
            cache->UpdateUnitSize();
        }
        Init(key, {batch * capacity, width}); Init(value, {batch * capacity, 128});
        Init(otherKey, key.dims); Init(otherValue, value.dims);
        keyPool.Resize(key.dims); keyPool.FakeFrom(key, 0);
        valuePool.Resize(value.dims); valuePool.FakeFrom(value, 0);
        for (int b = 0; b < batch; ++b) {
            keys.emplace_back(new Data(cacheType, {1, capacity, width}, DataDevice::CUDA,
                ring ? (char *)key.cudaData + (size_t)b * capacity * width * key.unitSize : (void *)((int *)pages.cudaData + b * 8)));
            values.emplace_back(new Data(cacheType, {1, capacity, 128}, DataDevice::CUDA,
                ring ? (char *)value.cudaData + (size_t)b * capacity * 128 * value.unitSize : (void *)((int *)pages.cudaData + b * 8 + 4)));
            indices.emplace_back(new Data(BFLOAT16, {1, 1, 128}, DataDevice::CUDA,
                (char *)index.cudaData + b * 128 * 2));
            if (!ring) {
                keys.back()->isPagedKVCache = values.back()->isPagedKVCache = true;
                keys.back()->pageLen = values.back()->pageLen = pageLen;
                keys.back()->pagedKVCacheData = &keyPool;
                values.back()->pagedKVCacheData = &valuePool;
            }
            rows.push_back({keys.back().get(), values.back().get(), ring ? nullptr : indices.back().get(),
                           (const void *const *)table.cudaData + b * 4});
        }
        Fill(x, batch + 17); Fill(index, batch + 37); Update(0);
    }
    ~Case() { if (graph) cudaGraphExecDestroy(graph); }
    void Update(int iteration) {
        std::vector<float> pos(batch);
        std::vector<int> live(batch), ids(batch * 8);
        std::vector<void *> bindings(batch * 4);
        const int boundaries[] = {0, 1, 16, 17, 63, 64, 127, 128, 129, capacity, capacity + 1};
        Data &dstKey = iteration % 2 ? otherKey : key, &dstValue = iteration % 2 ? otherValue : value;
        for (int b = 0; b < batch; ++b) {
            pos[b] = (iteration > 1 ? 1048500 : 8192) + b;
            live[b] = boundaries[(b + iteration) % 11];
            for (int j = 0; j < 4; ++j) {
                ids[b * 8 + j] = ((b + iteration) % batch) * 4 + (3 - j);
                ids[b * 8 + 4 + j] = ((batch - 1 - b + iteration) % batch) * 4 + j;
            }
            bindings[b * 4] = (char *)dstKey.cudaData + (ring ? (size_t)b * capacity * width * key.unitSize : 0);
            bindings[b * 4 + 1] = (char *)dstValue.cudaData + (ring ? (size_t)b * capacity * 128 * value.unitSize : 0);
            bindings[b * 4 + 2] = (int *)pages.cudaData + b * 8;
            bindings[b * 4 + 3] = (int *)pages.cudaData + b * 8 + 4;
        }
        Check(cudaMemcpy(positions.cudaData, pos.data(), batch * sizeof(float), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(lengths.cudaData, live.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(pages.cudaData, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(table.cudaData, bindings.data(), bindings.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }
    void Reset() {
        for (Data *data : {&q, &key, &value, &otherKey, &otherValue})
            Check(cudaMemset(data->cudaData, 0x3d, data->GetBytes()));
    }
    std::vector<std::vector<uint8_t>> Output() {
        return {Read(q), Read(key), Read(value), Read(otherKey), Read(otherValue)};
    }
    bool Fused(Data &weight, Data &scratch, const Data &bias = Data()) {
        return FastllmCudaNaiveBatchQkvRopeAppendCache(x, weight, bias, q, k, v, positions, rows,
            lengths, 8, 1, 192, 128, 64, ring ? 10000.f : 10000000.f, .707f, ring ? 128 : 0, ring, scratch);
    }
    void Reference(Data &weight) {
        Data bias;
        Require(FastllmCudaBFloat16MatMulBFloat16(x, weight, bias, packed, batch, 4096, 1856), "Linear failed");
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, rows, lengths,
            8, 1, 192, 128, 64, ring ? 10000.f : 10000000.f, .707f, ring ? 128 : 0, &packed, ring), "RoPE failed");
    }
};
void VerifyBatch(int requests, int mode, DataType type) {
    constexpr int heads = 8, dim = 192, vd = 128, width = 1856;
    const int rows = 6, total = requests * rows, capacity = mode == 2 ? 136 : 512;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), packed(BFLOAT16), index(BFLOAT16);
    Data positions(FLOAT32), live(INT32);
    Init(q, {1, total, heads * dim}); Init(k, {1, total, dim}); Init(v, {1, total, vd});
    Init(packed, {1, total, width}); Init(index, {1, total, 128});
    Init(positions, {1, total}); Init(live, {total});
    Fill(packed, 713); Fill(index, 19);
    std::vector<float> pos(total); std::vector<int> lengths(total);
    std::vector<std::unique_ptr<Data>> keys, values, indices;
    std::vector<FastllmNaiveRopeCacheRow> descriptors;
    auto view = [](Data &x, const Data &src, int offset, int count) {
        auto dims = src.dims; dims[1] = count; x.Resize(dims);
        x.FakeFrom(src, (size_t)offset * src.dims.back() * src.unitSize);
    };
    for (int b = 0; b < requests; ++b) {
        const int past = mode == 2 ? 4090 + b : 111 + b % 9;
        keys.emplace_back(new Data(type)); values.emplace_back(new Data(type));
        Init(*keys.back(), {1, capacity, mode ? dim : dim + 128});
        Init(*values.back(), {1, capacity, vd});
        keys.back()->expansionDims = keys.back()->dims; values.back()->expansionDims = values.back()->dims;
        keys.back()->Resize({1, mode == 2 ? 127 : past, mode ? dim : dim + 128});
        values.back()->Resize({1, mode == 2 ? 127 : past, vd});
        for (int r = 0; r < rows; ++r) {
            int n = b * rows + r; pos[n] = past + r; lengths[n] = past + r + 1;
            indices.emplace_back(new Data()); view(*indices.back(), index, n, 1);
            FastllmNaiveRopeCacheRow item{keys.back().get(), values.back().get(), mode ? nullptr : indices.back().get()};
            if (mode == 1) item.cacheOffset = past + r;
            if (mode == 2) item.verifyOffset = r;
            descriptors.push_back(item);
        }
    }
    Check(cudaMemcpy(positions.cudaData, pos.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(live.cudaData, lengths.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    auto reset = [&]() {
        for (auto *set : {&keys, &values}) for (auto &x : *set)
            Check(cudaMemset(x->cudaData, 0x3d, (size_t)capacity * x->dims.back() * x->unitSize));
        Check(cudaMemset(q.cudaData, 0x3d, q.GetBytes()));
    };
    auto result = [&]() {
        std::vector<std::vector<uint8_t>> out{Read(q)};
        for (auto *set : {&keys, &values}) for (auto &x : *set) {
            out.emplace_back((size_t)capacity * x->dims.back() * x->unitSize);
            Check(cudaMemcpy(out.back().data(), x->cudaData, out.back().size(), cudaMemcpyDeviceToHost));
        }
        return out;
    };
    reset();
    Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, descriptors, live,
        heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &packed), "verification batch rejected");
    auto actual = result(); reset();
    for (int b = 0; b < requests; ++b) {
        Data rq, rk, rv, rp, ri, raw, lengthsView;
        view(rq, q, b * rows, rows); view(rk, k, b * rows, rows); view(rv, v, b * rows, rows);
        view(raw, packed, b * rows, rows); if (!mode) view(ri, index, b * rows, rows);
        rp.Resize({1, rows}); rp.FakeFrom(positions, b * rows * sizeof(float));
        if (mode == 2) { lengthsView.Resize({1}); lengthsView.FakeFrom(live, b * rows * sizeof(int)); }
        Require(FastllmCudaNaiveRopeAppendCache(rq, rk, rv, ri, rp, *keys[b], *values[b], lengthsView,
            heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &raw, false, mode == 2), "verification reference rejected");
    }
    Require(actual == result(), "batched verification RoPE/cache differs from independent requests");
    reset(); auto before = result();
    descriptors.back().verifyOffset = 8;
    Require(!FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, descriptors, live,
        heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &packed), "invalid final descriptor accepted");
    Require(before == result(), "failed preflight changed earlier request");
}

}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Check(cudaSetDevice(0));
        for (auto type : {BFLOAT16, FP8_E4M3})
            for (int mode : {0, 1, 2}) for (int batch : {2, 8, 40}) VerifyBatch(batch, mode, type);
        if (FastllmCudaRuntimeArch() != 90) { puts("SKIP: requires SM90 fusion build"); return 77; }
        Data weight(BFLOAT16), scratch(BFLOAT16);
        Init(weight, {1856, 4096}); Fill(weight, 191);
        std::vector<std::unique_ptr<Case>> cases;
        const int batches[] = {256, 129, 128, 65, 64, 33, 32, 17, 16, 8, 7, 6, 5, 4, 3, 2, 1};
        void *address = nullptr;
        for (DataType type : {BFLOAT16, FP8_E4M3})
        for (int batch : batches) for (bool ring : {false, true}) {
            cases.emplace_back(new Case(batch, ring, 64, type));
            auto &c = *cases.back();
            Require(c.Fused(weight, scratch), "supported fusion rejected");
            if (!address) address = scratch.cudaData;
            Require(scratch.cudaData == address, "smaller batch relocated graph scratch");
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(c.Fused(weight, scratch), "capture rejected");
            cudaGraph_t graph;
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&c.graph, graph, 0));
            Check(cudaGraphDestroy(graph));
        }
        int checks = 0;
        for (auto &fixture : cases) {
            auto &c = *fixture;
            for (int replay = 0; replay < 4; ++replay) {
                c.Update(replay); c.Reset();
                Check(cudaGraphLaunch(c.graph, cudaStreamPerThread));
                auto actual = c.Output();
                c.Reset(); c.Reference(weight);
                Require(actual == c.Output(), "projection, rotary, page binding or untouched cache differs");
                Data bias(BFLOAT16, {1856});
                Require(!c.Fused(weight, scratch, bias), "bias incorrectly accepted");
                ++c.x.strides[1];
                Require(!c.Fused(weight, scratch), "bad stride incorrectly accepted");
                --c.x.strides[1];
                Require(actual == c.Output(), "rejected fusion modified outputs");
                ++checks;
            }
        }
        printf("PASS: %d exact graph replays, shared workspace and fallback checks\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
