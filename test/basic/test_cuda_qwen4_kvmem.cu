#include "devices/cuda/fastllm-cuda.cuh"
#include "models/qwen4_kvmem.h"
#include "models/qwen4_paged_cache.h"
#include "models/qwen4_exp.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void C(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
static void ModelPolicy() {
    setenv("FASTLLM_EXECUTION_MODE", "auto", 1);
    const auto oldDevices = GetDeviceMap(), oldMoe = GetMoeDeviceMap();
    for (int placement = 0; placement < 3; ++placement) {
        SetDeviceMap({{placement == 2 ? "cpu" : "cuda:0", 1}});
        SetMoeDeviceMap({{placement == 0 ? "cuda:0" : "cpu", 1}});
        Qwen4ExpModel model;
        model.weight.dicts = {{"num_hidden_layers", "4"}, {"num_experts", "2"},
                             {"num_experts_per_tok", "1"}};
        model.InitParams();
        model.dataType = model.kvCacheDataType = FLOAT16;
        model.maxBatch = 1;
        KvMemConfig cfg;
        cfg.enabled = true; cfg.maxTokens = 16384; cfg.residentTokens = 4096;
        cfg.hostBytes = 1ULL << 30;
        basellm &base = model;
        bool accepted = false;
        try { base.ConfigureKvMem(cfg); accepted = true; }
        catch (const std::invalid_argument &) {}
        Check(accepted == (placement == 1), "KVMem must only configure Hybrid Qwen4");
        Check(model.SupportsKvMem() == (placement == 1), "KVMem capability disagrees with execution mode");
        if (accepted) {
            Check(model.kvMemConfig.enabled && !model.saveHistoryChat && model.tokensLimit == 16384,
                  "native QSA configuration did not reach Hybrid adapter");
            model.kvMemLocked = true;
            cfg.enabled = false;
            bool rejected = false;
            try { base.ConfigureKvMem(cfg); } catch (const std::invalid_argument &) { rejected = true; }
            Check(rejected, "KVMem allowed reconfiguration after warmup");
        }
    }
    SetDeviceMap(oldDevices); SetMoeDeviceMap(oldMoe);
    unsetenv("FASTLLM_EXECUTION_MODE");
    puts("PASS Qwen4 Hybrid-only KVMem configuration and lifecycle guard");
}
static void GPU(Data &d, int device) {
    d.dataDevice = CUDA; d.dataDeviceIds = {device}; d.Allocate(false);
}
static void Equal(Data &a, Data &b) {
    std::vector<unsigned char> x(a.Count(0) * 2), y(x.size());
    C(cudaMemcpy(x.data(), a.cudaData, x.size(), cudaMemcpyDeviceToHost));
    C(cudaMemcpy(y.data(), b.cudaData, y.size(), cudaMemcpyDeviceToHost));
    Check(x == y, "KVMem differs from full GPU reference");
}
static void Run(int device, int heads, int dim, int rows, int resident, int group = 4, int length = 16384) {
    FastllmCudaSetDevice(device);
    printf("START gpu=%d heads=%d dim=%d rows=%d length=%d group=%d\n",
           device, heads, dim, rows, length, group);
    constexpr int width = 2051;
    KvMemConfig cfg;
    cfg.enabled = true; cfg.maxTokens = length; cfg.residentTokens = resident;
    auto owner = std::make_shared<Qwen4KvMem>(cfg, FLOAT16, heads, dim, device);
    Check(owner->gpuBytes < size_t(2) * length * heads * dim * 2, "unbounded device KV allocation");
    Data k(FLOAT16, {heads, length, dim}), v(FLOAT16, k.dims),
         hk(FLOAT16, {heads, 0, dim}), hv(FLOAT16, hk.dims);
    GPU(k, device); GPU(v, device);
    k.expansionDims = k.dims; v.expansionDims = v.dims;
    owner->Attach(hk, hv, owner);
    std::mt19937 rng(73);
    std::vector<half> input(k.Count(0));
    for (Data *tensor : {&k, &v}) {
        for (auto &x : input) x = __float2half_rn(float(int(rng() % 2001) - 1000) / 1000);
        C(cudaMemcpy(tensor->cudaData, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    }
    Check(FastllmCudaQwen4KVAppend(k, v, 0, hk, hv), "host append rejected");
    Qwen4ResizeKVCache(hk, length); Qwen4ResizeKVCache(hv, length);
    Data q(FLOAT16, {heads * group, rows, dim}), ids(INT32, {rows, width});
    GPU(q, device); GPU(ids, device);
    input.resize(q.Count(0));
    for (auto &x : input) x = __float2half_rn(float(int(rng() % 1001) - 500) / 1000);
    C(cudaMemcpy(q.cudaData, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    Data a(FLOAT16, q.dims), b(FLOAT16, q.dims);
    GPU(a, device); GPU(b, device);
    const float scale = 1.0f / std::sqrt(float(dim));
    std::vector<int> selected(rows * width);
    auto select = [&](int active, int seed) {
        std::mt19937 random(seed);
        for (int row = 0; row < rows; ++row) {
            for (int t = 0; t < width; t += 4) {
                int block = random() % (active / 4);
                for (int j = 0; j < 4 && t + j < width; ++j)
                    selected[row * width + t + j] = block * 4 + j;
            }
            selected[row * width] = -1;
            selected[row * width + 1] = active;
        }
        C(cudaMemcpy(ids.cudaData, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));
    };
    auto run = [&](Data &key, Data &value, Data &out, const int *meta = nullptr) {
        Check(rows == 1
            ? FastllmCudaQwen4SparseDecode(q, key, value, ids, out, group, scale, meta)
            : FastllmCudaQwen4SparsePrefill(q, key, value, ids, out, group, scale), "sparse dispatch");
    };
    for (int round = 0; round < 6; ++round) {
        select(length, round + 5);
        run(k, v, a); run(hk, hv, b); Equal(a, b);
    }
    const auto &stats = owner->Stats();
    Check(stats.tokens == length && stats.residentPages == resident / 4, "incorrect cache length stats");
    if (rows == 1) Check(stats.restoredPages > 0 && stats.evictedPages == stats.restoredPages,
                         "refill stats not updated");
    if (rows == 1) {
        Data ik(FLOAT16, {heads, 1, dim}), iv(FLOAT16, ik.dims), meta(INT32, {1});
        GPU(ik, device); GPU(iv, device); GPU(meta, device);
        C(cudaMemset(ik.cudaData, 0, ik.Count(0) * 2));
        C(cudaMemset(iv.cudaData, 0x3c, iv.Count(0) * 2));
        int previous = length - 1;
        C(cudaMemcpy(meta.cudaData, &previous, 4, cudaMemcpyHostToDevice));
        run(hk, hv, b, (int *)meta.cudaData);
        C(cudaDeviceSynchronize());
        Check(FastllmCudaGraphPrepareCaptureDevice(), "graph prepare");
        Check(FastllmCudaGraphBeginCapture(), "graph begin");
        Check(FastllmCudaQwen4KVAppendGraph(ik, iv, (int *)meta.cudaData, hk, hv), "graph append");
        run(hk, hv, b, (int *)meta.cudaData);
        void *graph = nullptr, *exec = nullptr;
        Check(FastllmCudaGraphEndCapture(&graph), "graph end");
        Check(FastllmCudaGraphInstantiate(graph, &exec), "graph instantiate");
        for (int active : {length, 8001, 8002, 4097, 8193, 16383}) {
            previous = active - 1;
            select(active, active);
            selected[2] = previous; // Verify overwrites of a cached partial page.
            C(cudaMemcpy(ids.cudaData, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));
            C(cudaMemcpy(meta.cudaData, &previous, 4, cudaMemcpyHostToDevice));
            Check(FastllmCudaGraphLaunch(exec), "graph replay");
            Check(FastllmCudaQwen4KVAppend(ik, iv, previous, k, v), "reference overwrite");
            k.Resize({heads, active, dim}); v.Resize(k.dims);
            run(k, v, a); Equal(a, b);
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    }
    owner->Reset();
    Check(owner->Stats().tokens == 0 && owner->Stats().restoredPages == 0,
          "reset retained cache stats");
    k.Resize({heads, length, dim}); v.Resize(k.dims);
    Check(FastllmCudaQwen4KVAppend(k, v, 0, hk, hv), "reused backing append");
    Qwen4ResizeKVCache(hk, length); Qwen4ResizeKVCache(hv, length);
    select(length, 91);
    run(k, v, a); run(hk, hv, b); Equal(a, b);
    printf("PASS GPU=%d heads=%d dim=%d rows=%d resident=%d GPU_KV=%.3f MiB host_KV=%.3f MiB\n",
           device, heads, dim, rows, resident, owner->gpuBytes / 1048576., owner->Stats().hostBytes / 1048576.);
}
int main() {
#if defined(USE_ROCM) || defined(CUDA_NO_TENSOR_CORE)
    puts("FASTLLM_TEST_SKIP_NO_DEVICE: QSA KVMem requires CUDA tensor cores");
    return 0;
#else
    try {
        int devices = 0;
        cudaDeviceProp properties{};
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 ||
            cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
            properties.major * 10 + properties.minor < 75) {
            puts("FASTLLM_TEST_SKIP_NO_DEVICE: QSA KVMem requires SM75+");
            return 0;
        }
        setvbuf(stdout, nullptr, _IONBF, 0);
        SetThreads(2);
        ModelPolicy();
        for (int device = 0; device < std::min(devices, 2); ++device)
            for (int heads : {1, 2})
                for (int dim : {128, 256})
                    for (int rows : {1, 32, 128})
                        Run(device, heads, dim, rows, rows == 1 ? 4096 : 1024);
        Run(0, 2, 256, 128, 1024, 24);
        Run(0, 2, 256, 1, 4);
        Run(0, 2, 256, 1, 4096, 4, 262144);
        Run(0, 2, 256, 32, 4096, 4, 262144);
        puts("PASS exact QSA host/device storage, overflow, eviction, graph append and rollback");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
#endif
}
