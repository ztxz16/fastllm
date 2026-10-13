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
std::vector<uint16_t> Read(const Data &x) {
    std::vector<uint16_t> result(x.Count(0));
    Check(cudaMemcpy(result.data(), x.cudaData, result.size() * 2, cudaMemcpyDeviceToHost));
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
    Case(int batch, bool ring, int pageLen = 64)
        : batch(batch), capacity(ring ? 128 : pageLen * 4), pageLen(ring ? 0 : pageLen),
          width(ring ? 192 : 320), ring(ring) {
        Init(x, {1, batch, 4096}); Init(q, {1, batch, 1536});
        Init(k, {1, batch, 192}); Init(v, {1, batch, 128});
        Init(packed, {1, batch, 1856}); Init(index, {1, batch, 128});
        Init(positions, {1, batch}); Init(lengths, {batch});
        Init(pages, {batch, 8}); Init(table, {batch, 8});
        Init(key, {batch * capacity, width}); Init(value, {batch * capacity, 128});
        Init(otherKey, key.dims); Init(otherValue, value.dims);
        keyPool.Resize(key.dims); keyPool.FakeFrom(key, 0);
        valuePool.Resize(value.dims); valuePool.FakeFrom(value, 0);
        for (int b = 0; b < batch; ++b) {
            keys.emplace_back(new Data(BFLOAT16, {1, capacity, width}, DataDevice::CUDA,
                ring ? (char *)key.cudaData + (size_t)b * capacity * width * 2 : (void *)((int *)pages.cudaData + b * 8)));
            values.emplace_back(new Data(BFLOAT16, {1, capacity, 128}, DataDevice::CUDA,
                ring ? (char *)value.cudaData + (size_t)b * capacity * 128 * 2 : (void *)((int *)pages.cudaData + b * 8 + 4)));
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
            bindings[b * 4] = (char *)dstKey.cudaData + (ring ? (size_t)b * capacity * width * 2 : 0);
            bindings[b * 4 + 1] = (char *)dstValue.cudaData + (ring ? (size_t)b * capacity * 128 * 2 : 0);
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
            Check(cudaMemset(data->cudaData, 0x3d, data->Count(0) * 2));
    }
    std::vector<std::vector<uint16_t>> Output() {
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
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Check(cudaSetDevice(0));
        if (FastllmCudaRuntimeArch() != 90) { puts("SKIP: requires SM90 fusion build"); return 77; }
        Data weight(BFLOAT16), scratch(BFLOAT16);
        Init(weight, {1856, 4096}); Fill(weight, 191);
        std::vector<std::unique_ptr<Case>> cases;
        const int batches[] = {256, 129, 128, 65, 64, 33, 32, 17, 16, 8, 7, 6, 5, 4, 3, 2, 1};
        void *address = nullptr;
        for (int batch : batches) for (bool ring : {false, true}) {
            cases.emplace_back(new Case(batch, ring));
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
