#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "fastllm.h"
#include "blocks/baseblock.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static int checks = 0;
static void Require(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
    ++checks;
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void Init(Data &x, std::vector<int> shape, int device) {
    x.Resize(shape);
    x.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    x.Allocate(false);
}
static void Fill(Data &x, unsigned seed) {
    std::vector<uint16_t> bits(x.Count(0));
    for (auto &v : bits) {
        seed = seed * 1664525u + 1013904223u;
        v = seed >> 16;
        if ((v & 0x7f80) == 0x7f80) v &= 0x807f;
    }
    Cuda(cudaMemcpy(x.cudaData, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
}
static std::vector<uint16_t> Read(const Data &x) {
    std::vector<uint16_t> v(x.Count(0));
    Cuda(cudaMemcpy(v.data(), x.cudaData, v.size() * 2, cudaMemcpyDeviceToHost));
    return v;
}
static void CopyDevice(const Data &a, Data &b) {
    Cuda(cudaMemcpy(b.cudaData, a.cudaData, a.Count(0) * 2, cudaMemcpyDeviceToDevice));
}
static void Run(int device, int rows, int heads, int kvHeads, int dim, int vd,
                int rd, int indexDim, int window, float scale, bool merged = false) {
    Cuda(cudaSetDevice(device));
    const int capacity = window ? window - 1 + rows : 2050 + rows;
    const int kc = kvHeads * dim + indexDim, vc = kvHeads * vd;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), ik(BFLOAT16), qr(BFLOAT16), kr(BFLOAT16),
        vr(BFLOAT16), ir(BFLOAT16), key(BFLOAT16), val(BFLOAT16), keyRef(BFLOAT16),
        valRef(BFLOAT16), positions(FLOAT32), live(INT32), packed;
    Init(q, {1, rows, heads * dim}, device); Init(qr, q.dims, device);
    Init(k, {1, rows, kvHeads * dim}, device); Init(kr, k.dims, device);
    Init(v, {1, rows, vc}, device); Init(vr, v.dims, device);
    if (indexDim) { Init(ik, {1, rows, indexDim}, device); Init(ir, ik.dims, device); }
    Init(key, {1, capacity, kc}, device); Init(keyRef, key.dims, device);
    Init(val, {1, capacity, vc}, device); Init(valRef, val.dims, device);
    Init(positions, {1, rows}, device); Init(live, {rows}, device);
    Data qkv(BFLOAT16);
    Init(qkv, {1, rows, (heads + kvHeads) * dim + kvHeads * vd}, device);
    float theta = window ? 10000.f : 10000000.f;
    auto fused = [&] {
        Require(FastllmCudaNaiveRopeAppendCache(q, k, v, ik, positions, key, val, live,
            heads, kvHeads, dim, vd, rd, theta, scale, window, merged ? &qkv : nullptr), "fusion rejected valid layout");
    };
    int firstLength = 1;
    Cuda(cudaMemcpy(live.cudaData, &firstLength, sizeof(int), cudaMemcpyHostToDevice));
    Cuda(cudaMemset(positions.cudaData, 0, rows * sizeof(float)));
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    fused();
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<int> lengths = window ? std::vector<int>{1, window - 1, window, window + 3}
                                      : std::vector<int>{1, 127, 2048, 2051};
    for (int replay = 0; replay < 4; ++replay) {
        Fill(q, 11 + replay); Fill(k, 19 + replay); Fill(v, 29 + replay);
        Fill(key, 41 + replay); Fill(val, 53 + replay);
        CopyDevice(q, qr); CopyDevice(k, kr); CopyDevice(v, vr); CopyDevice(key, keyRef); CopyDevice(val, valRef);
        if (indexDim) { Fill(ik, 61 + replay); CopyDevice(ik, ir); }
        Data tmp; Cat(q, k, 2, tmp); Cat(tmp, v, 2, qkv);
        auto originalQkv = Read(qkv);
        auto originalK = Read(k), originalV = Read(v);
        std::vector<float> pos(rows);
        std::vector<int> len(rows);
        for (int i = 0; i < rows; ++i) {
            pos[i] = (replay < 2 ? 127 : 1048500) + i;
            len[i] = lengths[replay] + i;
        }
        Cuda(cudaMemcpy(positions.cudaData, pos.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, len.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
        FastllmCudaNaiveRopeQKScaleV(qr, kr, vr, positions, heads, kvHeads, dim, vd, rd, theta, scale);
        if (indexDim) {
            FastllmCudaNaiveRope(ir, positions, 1, indexDim, rd, theta);
            Cat(kr, ir, 2, packed);
        } else packed.CopyFrom(kr);
        if (rows == 1) FastllmCudaNaiveAppendDecodeCache(keyRef, valRef, packed, vr, live, window);
        else FastllmCudaNaiveAppendVerifyCache(keyRef, valRef, packed, vr, live, window);
        if (!replay) fused();
        else Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Require(Read(q) == Read(qr), "Q differs bitwise");
        Require(Read(key) == Read(keyRef), "K/index cache or untouched rows differ bitwise");
        Require(Read(val) == Read(valRef), "V cache or untouched rows differ bitwise");
        Require(Read(qkv) == originalQkv, "packed input changed");
        Require(Read(k) == originalK && Read(v) == originalV, "fusion changed source K/V");
    }
    auto oldQ = Read(q), oldKey = Read(key), oldValue = Read(val);
    auto stride = k.strides[1]; ++k.strides[1];
    Require(!FastllmCudaNaiveRopeAppendCache(q, k, v, ik, positions, key, val, live,
        heads, kvHeads, dim, vd, rd, theta, scale, window), "bad stride accepted");
    k.strides[1] = stride;
    Require(Read(q) == oldQ && Read(key) == oldKey && Read(val) == oldValue,
        "rejected layout changed outputs");
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
}
static void RunStatic(int device, int rows, int heads, int kvHeads, int dim, int vd,
                      int rd, int indexDim, int window, bool merged = false) {
    Cuda(cudaSetDevice(device));
    const int capacity = (window ? window - 1 : 2050) + rows;
    const int kc = kvHeads * dim + indexDim, vc = kvHeads * vd;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), ik(BFLOAT16), qr(BFLOAT16), kr(BFLOAT16),
        vr(BFLOAT16), ir(BFLOAT16), key(BFLOAT16), val(BFLOAT16), keyRef(BFLOAT16),
        valRef(BFLOAT16), positions(FLOAT32), live(INT32), empty, packed;
    Data qkv(BFLOAT16);
    Init(qkv, {1, rows, (heads + kvHeads) * dim + kvHeads * vd}, device);
    Init(q, {1, rows, heads * dim}, device); Init(qr, q.dims, device);
    Init(k, {1, rows, kvHeads * dim}, device); Init(kr, k.dims, device);
    Init(v, {1, rows, vc}, device); Init(vr, v.dims, device);
    if (indexDim) { Init(ik, {1, rows, indexDim}, device); Init(ir, ik.dims, device); }
    Init(key, {1, capacity, kc}, device); Init(keyRef, key.dims, device);
    Init(val, {1, capacity, vc}, device); Init(valRef, val.dims, device);
    key.expansionDims = key.dims; val.expansionDims = val.dims;
    Init(positions, {1, rows}, device); Init(live, {rows}, device);
    const float theta = window ? 10000.f : 10000000.f, scale = .707f;
    for (int past : {0, 1, 126, window ? window - 1 : 2050}) {
        Fill(q, 11 + past); Fill(k, 19 + past); Fill(v, 29 + past);
        Fill(key, 41 + past); Fill(val, 53 + past);
        CopyDevice(q, qr); CopyDevice(k, kr); CopyDevice(v, vr);
        CopyDevice(key, keyRef); CopyDevice(val, valRef);
        if (indexDim) { Fill(ik, 61 + past); CopyDevice(ik, ir); }
        const auto oldK = Read(k), oldV = Read(v);
        std::vector<float> pos(rows); std::vector<int> len(rows);
        for (int i = 0; i < rows; ++i) { pos[i] = 1048500 + i; len[i] = past + i + 1; }
        Cuda(cudaMemcpy(positions.cudaData, pos.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, len.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
        Data tmp; Cat(q, k, 2, tmp); Cat(tmp, v, 2, qkv);
        FastllmCudaNaiveRopeQKScaleV(qr, kr, vr, positions, heads, kvHeads, dim, vd, rd, theta, scale);
        if (indexDim) { FastllmCudaNaiveRope(ir, positions, 1, indexDim, rd, theta); Cat(kr, ir, 2, packed); }
        else packed.CopyFrom(kr);
        FastllmCudaNaiveAppendVerifyCache(keyRef, valRef, packed, vr, live, window);
        key.Resize({1, past, kc}); val.Resize({1, past, vc});
        Require(FastllmCudaNaiveRopeAppendCache(q, k, v, ik, positions, key, val, empty,
            heads, kvHeads, dim, vd, rd, theta, scale, window, merged ? &qkv : nullptr), "static length rejected");
        Require(key.dims[1] == past && val.dims[1] == past, "kernel changed host metadata");
        key.Resize({1, capacity, kc}); val.Resize({1, capacity, vc});
        Require(Read(q) == Read(qr), "static Q differs bitwise");
        Require(Read(key) == Read(keyRef), "static K cache differs bitwise");
        Require(Read(val) == Read(valRef), "static V cache differs bitwise");
        Require(Read(k) == oldK && Read(v) == oldV, "static fusion changed input K/V");
    }
    const auto oldQ = Read(q), oldKey = Read(key), oldValue = Read(val);
    Require(!FastllmCudaNaiveRopeAppendCache(q, k, v, ik, positions, key, val, empty,
        heads, kvHeads, dim, vd, rd, theta, scale, window), "static capacity overflow accepted");
    key.Resize({1, 1, kc}); val.Resize({1, 0, vc});
    Require(!FastllmCudaNaiveRopeAppendCache(q, k, v, ik, positions, key, val, empty,
        heads, kvHeads, dim, vd, rd, theta, scale, window), "static mismatched cache lengths accepted");
    key.Resize({1, capacity, kc}); val.Resize({1, capacity, vc});
    Require(Read(q) == oldQ && Read(key) == oldKey && Read(val) == oldValue,
        "rejected static append changed outputs");
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        for (int device = 0; device < std::min(devices, 2); ++device)
            for (int rows : {1, 2, 7, 8}) for (int window : {0, 8, 128})
                for (float scale : {1.f, .707f, -.3f}) {
                    Run(device, rows, 8, 1, 192, 128, 64, window ? 0 : 128, window, scale);
                    Run(device, rows, 6, 2, 35, 19, 16, window ? 0 : 31, window, scale);
                }
        Run(0, 8, 64, 4, 192, 128, 64, 128, 0, .707f);
        Run(0, 8, 64, 8, 192, 128, 64, 0, 128, .707f);
        for (int device = 0; device < std::min(devices, 2); ++device)
            for (int rows : {1, 2, 7, 8}) for (int window : {0, 128}) {
                RunStatic(device, rows, 64, window ? 8 : 4, 192, 128, 64, window ? 0 : 128, window);
                RunStatic(device, rows, 6, 2, 35, 19, 16, window ? 0 : 31, window);
            }
        for (int rows : {1, 2, 7, 8}) for (int ranks : {1, 2, 4, 8})
            for (int window : {0, 128}) {
                int kv = std::max(1, (window ? 8 : 4) / ranks);
                Run(0, rows, 64 / ranks, kv, 192, 128, 64, window ? 0 : 128, window, .707f, true);
                RunStatic(0, rows, 64 / ranks, kv, 192, 128, 64, window ? 0 : 128, window, true);
            }
        std::printf("Naive RoPE/cache: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Naive RoPE/cache: %s\n", e.what()); return 1;
    }
}
