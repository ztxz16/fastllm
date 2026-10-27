#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool b, const char *s) {
    if (!b)
        throw std::runtime_error(s);
}
static void Sync() {
    Check(cudaDeviceSynchronize() == cudaSuccess, "cuda synchronize");
}
static Data Weight(int n, int k) {
    Data w(DataType::FP8_E4M3, {n, k});
    w.blockM = w.blockK = 32;
    w.Allocate(false);
    for (size_t i = 0; i < w.Count(0); i++)
        w.cpuData[i] = uint8_t((i * 31 + i / k * 7) % 105) | uint8_t(i % 2 ? 128 : 0);
    w.scales.resize(size_t(n / 32) * (k / 32));
    for (size_t i = 0; i < w.scales.size(); i++)
        w.scales[i] = std::ldexp(1.f, int(i * 13 % 19) - 16);
    w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    return w;
}
static std::vector<uint16_t> Bytes(Data &x) {
    Sync();
    std::vector<uint16_t> v(x.Count(0));
    Check(cudaMemcpy(v.data(), x.cudaData, v.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess,
          "read");
    return v;
}
static bool Cached(Data &w) {
    return w.extraCudaData.size() > 7 && w.extraCudaData[7];
}
static bool Packed(Data &w) {
    return w.extraCudaData.size() > 6 && w.extraCudaData[4];
}
int main() {
    int count = 0, major = 0;
    size_t available = 0, totalMemory = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1)
        return 77;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
        major < 8)
        return 77;
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaMemGetInfo(&available, &totalMemory) != cudaSuccess || available < (size_t(10) << 30))
        return 77;
    try {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        setenv("FASTLLM_CUDA_FP8_MARLIN", "1", 1);
        for (auto shape : std::vector<std::pair<int, int>>{{512, 5120},
                                                           {1280, 5120},
                                                           {4096, 1280},
                                                           {4608, 5120},
                                                           {5120, 2304},
                                                           {25600, 6144}}) {
            int n = shape.first, k = shape.second, m = 128;
            Data w = Weight(n, k);
            std::vector<float> host(size_t(m) * k);
            for (size_t i = 0; i < host.size(); i++)
                host[i] = std::sin(float(i * 17 % 1031)) * .3f;
            Data x(DataType::BFLOAT16, {m, k}, host), out(DataType::BFLOAT16, {m, n});
            x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            out.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            out.Allocate(false);
            auto run = [&]() {
                Check(FastllmCudaBFloat16MatMulFP8E4M3(x, w, *GetEmptyData(), out, m, k, n),
                      "linear");
            };
            run();
            auto expected = Bytes(out);
            Check(!Cached(w), "unregistered weight cache");
            Check(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(x.cudaData, w, out.cudaData, 8, k, n),
                  "packed warmup");
            Check(Packed(w), "packed allocation");
            run();
            Check(Bytes(out) == expected, "cache build bits");
            bool eligible = size_t(n) * k * 2 <= (size_t(64) << 20);
            Check(Cached(w) == eligible, "cache size guard");
            void *address = Cached(w) ? w.extraCudaData[7] : nullptr;
            run();
            Check(Bytes(out) == expected, "cache replay bits");
            void *graph = nullptr, *exec = nullptr;
            Check(FastllmCudaGraphBeginCapture(), "capture begin");
            for (int i = 0; i < 4; i++)
                run();
            Check(FastllmCudaGraphEndCapture(&graph), "capture end");
            Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
            for (int i = 0; i < 3; i++)
                Check(FastllmCudaGraphLaunch(exec), "graph replay");
            Check(Bytes(out) == expected, "graph bits");
            if (address)
                Check(address == w.extraCudaData[7], "cache moved");
            FastllmCudaGraphExecDestroy(exec);
            FastllmCudaGraphDestroy(graph);
            w.ToDevice(DataDevice::CPU);
            Check(!Packed(w) && !Cached(w), "move did not release");
            w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            run();
            Check(Bytes(out) == expected, "move roundtrip bits");
            Check(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(x.cudaData, w, out.cudaData, 8, k, n),
                  "repack");
            run();
            Check(Bytes(out) == expected, "rebuild bits");
            w.FreeSpace();
            Check(!Packed(w) && !Cached(w), "free did not release");
            std::cout << "N=" << n << " K=" << k << " cache=" << eligible
                      << " bits=equal graph/move/free=pass" << std::endl;
        }
        // Warm allocator/library state, then repeatedly destroy a registered owner.
        size_t first = 0, last = 0, total = 0;
        for (int it = 0; it < 24; it++) {
            {
                Data w = Weight(1024, 1024);
                Data x(DataType::BFLOAT16, {8, 1024}, std::vector<float>(8 * 1024, .25f)),
                    out(DataType::BFLOAT16, {8, 1024});
                x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                out.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                out.Allocate(false);
                Check(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(x.cudaData, w, out.cudaData, 8,
                                                                1024, 1024),
                      "lifetime pack");
                Sync();
            }
            Sync();
            if (it == 3)
                Check(cudaMemGetInfo(&first, &total) == cudaSuccess, "memory first");
        }
        Check(cudaMemGetInfo(&last, &total) == cudaSuccess, "memory last");
        Check(first <= last + (size_t(4) << 20), "auxiliary ownership leak");
        std::cout << "PASS BF16 FP8 auxiliary cache; destructor growth_bytes="
                  << static_cast<long long>(first) - static_cast<long long>(last) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
