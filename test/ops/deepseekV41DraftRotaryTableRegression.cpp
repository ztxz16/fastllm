#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static uint16_t BF(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static void Alloc(Data &d, DataType type, std::vector<int> dims, int dev) {
    d.dataType = type;
    d.UpdateUnitSize();
    d.Resize(dims);
    d.ToDevice(CUDA, {dev}, false);
    d.Allocate(false);
}
static std::vector<uint16_t> Read(const Data &d) {
    std::vector<uint16_t> x(d.Count(0));
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Cuda(cudaMemcpy(x.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return x;
}
static void Upload(Data &d, const std::vector<uint16_t> &x) {
    Cuda(cudaMemcpyAsync(d.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice, cudaStreamPerThread));
}
static void Run(int dev, int m, int heads, int windowRows, int ropeDim) {
    Data q, kv, window, table, oq, ok, ot, position;
    Alloc(q, BFLOAT16, {1, m, heads, 512}, dev);
    Alloc(oq, BFLOAT16, q.dims, dev);
    Alloc(kv, BFLOAT16, {1, m, 512}, dev);
    Alloc(ok, BFLOAT16, kv.dims, dev);
    Alloc(window, BFLOAT16, {1, windowRows, 512}, dev);
    Alloc(ot, BFLOAT16, {1, windowRows + m, 512}, dev);
    Alloc(position, INT32, {1}, dev);
    std::mt19937 rng(127 + m + heads + windowRows);
    std::normal_distribution<float> n;
    std::vector<uint16_t> qv(q.Count(0)), kvv(kv.Count(0)), wv(window.Count(0));
    for (auto &x : qv)
        x = BF(n(rng));
    for (auto &x : kvv)
        x = BF(n(rng));
    for (auto &x : wv)
        x = BF(n(rng));
    Upload(window, wv);
    int pos = 511;
    auto inputs = [&] {
        Upload(q, qv);
        Upload(oq, qv);
        Upload(kv, kvv);
        Upload(ok, kvv);
        Cuda(cudaMemcpyAsync(position.cudaData, &pos, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
    };
    auto call = [&] {
        Check(FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f, 32, 1,
                                                     (int *)position.cudaData, table),
              "fused rejected");
    };
    auto old = [&] {
        Check(FastllmCudaDeepSeekV41RotaryQuantPosition(oq, ropeDim, 10000.f, 17, 1, false, 0, 1.f, 32, 1, 0,
                                                        512, 32, (int *)position.cudaData),
              "q reference");
        Check(FastllmCudaDeepSeekV41RotaryQuantPosition(ok, ropeDim, 10000.f, 17, 1, false, 0, 1.f, 32, 1, 1,
                                                        512, 32, (int *)position.cudaData),
              "kv reference");
        Cuda(cudaMemcpyAsync(ot.cudaData, window.cudaData, window.GetBytes(), cudaMemcpyDeviceToDevice,
                             cudaStreamPerThread));
        Cuda(cudaMemcpyAsync((char *)ot.cudaData + window.GetBytes(), ok.cudaData, ok.GetBytes(),
                             cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    };
    inputs();
    call();
    old();
    Check(Read(q) == Read(oq) && Read(kv) == Read(ok) && Read(table) == Read(ot), "eager changed bits");
    void *graph = nullptr, *exec = nullptr;
    Check(FastllmCudaGraphBeginCapture(), "begin capture");
    call();
    Check(FastllmCudaGraphEndCapture(&graph), "end capture");
    Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
    for (int replay = 0; replay < 3; ++replay) {
        pos = replay == 0 ? 0 : replay == 1 ? 4095 : 65535;
        std::rotate(qv.begin(), qv.begin() + 1, qv.end());
        std::rotate(kvv.begin(), kvv.begin() + 3, kvv.end());
        inputs();
        Check(FastllmCudaGraphLaunch(exec), "replay");
        old();
        Check(Read(q) == Read(oq), "Q changed bits");
        Check(Read(kv) == Read(ok), "KV changed bits");
        Check(Read(table) == Read(ot), "gather table changed bits");
        Check(Read(window) == wv, "history window overwritten");
    }
    FastllmCudaGraphExecDestroy(exec);
    FastllmCudaGraphDestroy(graph);
    // Rejection must precede mutations, including a partially overlapping view.
    const auto qb = Read(q), kb = Read(kv), tb = Read(table);
    Check(!FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f, 32, 1, nullptr,
                                                  q),
          "aliased output accepted");
    auto saved = kv.cudaData;
    kv.cudaData = (char *)q.cudaData + 2;
    Check(!FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f, 32, 1, nullptr,
                                                  table),
          "overlapping inputs accepted");
    kv.cudaData = saved;
    // Reusing a small logical output can expose a larger retained allocation.
    // Aliasing must be checked against the target shape before Resize grows it.
    auto tableDims = table.dims;
    table.Resize({1});
    saved = kv.cudaData;
    kv.cudaData = (char *)table.cudaData + 512 * 2;
    const bool accepted = FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f,
                                                                 32, 1, nullptr, table);
    kv.cudaData = saved;
    table.Resize(tableDims);
    Check(!accepted, "expanded output overlap accepted");
    auto ids = q.dataDeviceIds;
    q.dataDeviceIds = {dev + 1};
    Check(!FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f, 32, 1, nullptr,
                                                  table),
          "wrong device accepted");
    q.dataDeviceIds = ids;
    q.dataType = FLOAT16;
    Check(!FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f, 32, 1, nullptr,
                                                  table),
          "FP16 accepted");
    q.dataType = BFLOAT16;
    Check(Read(q) == qb && Read(kv) == kb && Read(table) == tb, "rejection modified data");
    // Fusion always quantizes KV; explicit non-quantized/reference modes
    // must reject it without mutating any input or output.
    for (const char *flag : {"FASTLLM_DSV41_DISABLE_FAKE_QUANT", "FASTLLM_DSV41_REFERENCE_MATH",
                             "FASTLLM_DSV41_LEGACY_ROTARY"}) {
        const char *previous = std::getenv(flag);
        const bool present = previous != nullptr;
        const std::string savedValue = present ? previous : "";
        setenv(flag, "1", 1);
        const bool fused = FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 17, 0, 1.f,
                                                                  32, 1, nullptr, table);
        if (present) setenv(flag, savedValue.c_str(), 1);
        else unsetenv(flag);
        Check(!fused, "explicit rotary/quantization mode ignored");
        Check(Read(q) == qb && Read(kv) == kb && Read(table) == tb, "mode rejection modified data");
    }
    // Host position fallback, with nontrivial YaRN table parameters.
    inputs();
    Check(FastllmCudaDeepSeekV41DraftRotaryTable(q, kv, window, ropeDim, 10000.f, 8191, 4096, 8.f, 32, 1,
                                                 nullptr, table),
          "host position");
    Check(FastllmCudaDeepSeekV41RotaryQuantPosition(oq, ropeDim, 10000.f, 8191, 1, false, 4096, 8.f, 32, 1, 0,
                                                    512, 32, nullptr),
          "host q");
    Check(FastllmCudaDeepSeekV41RotaryQuantPosition(ok, ropeDim, 10000.f, 8191, 1, false, 4096, 8.f, 32, 1, 1,
                                                    512, 32, nullptr),
          "host kv");
    Check(Read(q) == Read(oq) && Read(kv) == Read(ok), "host position changed bits");
}
int main(int argc, char **) {
    try {
        int devices = 0, cases = 0;
        Cuda(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        for (int dev = 0; dev < (argc > 1 ? 1 : devices); ++dev) {
            FastllmCudaSetDevice(dev);
            if (dev == 0 && argc == 1) {
                for (int m = 1; m <= 8; ++m)
                    for (int heads : {8, 64})
                        for (int rows : {1, 17, 512, 1024})
                            for (int rope : {32, 128}) {
                                Run(dev, m, heads, rows, rope);
                                ++cases;
                            }
            } else
                for (int m : {1, 5, 8}) {
                    Run(dev, m, 64, 512, 128);
                    ++cases;
                }
        }
        std::cout << "PASS " << cases
                  << " cases; eager and x3 changing-input graph replay bitwise; position, window, "
                     "alias, device and dtype checks"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
