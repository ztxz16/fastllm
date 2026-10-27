// A small decode graph must not write into another pool borrower's memory
// after an eager prefill increases the Marlin reduction tile from 8 to 64.
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void Check(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    const bool fp8 = argc > 1 && std::strcmp(argv[1], "fp8") == 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    Check(cudaSetDevice(0), "device");
    cudaDeviceProp prop{};
    Check(cudaGetDeviceProperties(&prop, 0), "properties");
    if (prop.major * 10 + prop.minor < 75 ||
        !FastllmCudaMarlinNVFP4Supported(128, 4096)) return 77;
    constexpr int n = 128, k = 4096, maxM = 64;
    const size_t smallBytes = size_t(prop.multiProcessorCount) * 8 * 256 * sizeof(float);
    half *a = nullptr, *c = nullptr;
    uint32_t *b = nullptr;
    void *scales = nullptr;
    float *globalScale = nullptr;
    int *locks = nullptr;
    Check(cudaMalloc(&a, size_t(maxM) * k * sizeof(half)), "input allocation");
    Check(cudaMalloc(&c, size_t(maxM) * n * sizeof(half)), "output allocation");
    const size_t weightBytes = size_t(n) * k / (fp8 ? 1 : 2);
    const size_t scaleBytes = fp8 ? size_t(n) * (k / 128) * sizeof(half)
                                  : size_t(n) * (k / 16);
    Check(cudaMalloc(&b, weightBytes), "weight allocation");
    Check(cudaMalloc(&scales, scaleBytes), "scale allocation");
    Check(cudaMalloc(&globalScale, sizeof(float)), "global scale allocation");
    Check(cudaMalloc(&locks, size_t(prop.multiProcessorCount) * 4 * sizeof(int)), "locks allocation");
    Check(cudaMemset(a, 0, size_t(maxM) * k * sizeof(half)), "input init");
    Check(cudaMemset(c, 0, size_t(maxM) * n * sizeof(half)), "output init");
    Check(cudaMemset(b, 0, weightBytes), "weight init");
    Check(cudaMemset(scales, 0x38, scaleBytes), "scale init");
    Check(cudaMemset(locks, 0, size_t(prop.multiProcessorCount) * 4 * sizeof(int)), "locks init");
    const float one = 1.0f;
    Check(cudaMemcpy(globalScale, &one, sizeof(one), cudaMemcpyHostToDevice), "global scale init");
    auto linear = [&](int m) {
        bool ok = fp8 ? FastllmCudaMarlinHalfFP8Gemm(a, b, scales, c, m, n, k, 128, locks)
                      : FastllmCudaMarlinHalfNVFP4Gemm(a, b, scales, globalScale, c, m, n, k, locks, nullptr);
        if (!ok) {
            std::fprintf(stderr, "Marlin %s M=%d rejected\n", fp8 ? "FP8" : "NVFP4", m);
            std::exit(1);
        }
    };
    linear(1);
    Check(cudaDeviceSynchronize(), "warmup");
    if (!FastllmCudaGraphMemoryPoolBegin() || !FastllmCudaGraphBeginCapture()) return 1;
    linear(1);
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> pins;
    if (!FastllmCudaGraphEndCapture(&graph) || !FastllmCudaGraphMemoryPoolEnd(pins) ||
        !FastllmCudaGraphInstantiate(graph, &exec) || !FastllmCudaGraphLaunch(exec)) return 1;
    Check(cudaDeviceSynchronize(), "initial graph replay");
    linear(maxM);
    Check(cudaDeviceSynchronize(), "larger eager prefill");
    void *canary = FastllmCudaMalloc(smallBytes);
    if (canary == nullptr) return 1;
    Check(cudaMemset(canary, 0x5a, smallBytes), "canary init");
    Check(cudaMemset(c, 0x5a, size_t(n) * sizeof(half)), "output poison");
    if (!FastllmCudaGraphLaunch(exec)) return 1;
    Check(cudaDeviceSynchronize(), "graph replay after prefill");
    std::vector<unsigned char> host(smallBytes);
    Check(cudaMemcpy(host.data(), canary, smallBytes, cudaMemcpyDeviceToHost), "canary read");
    const size_t changed = std::count_if(host.begin(), host.end(), [](unsigned char value) { return value != 0x5a; });
    std::vector<half> output(n);
    Check(cudaMemcpy(output.data(), c, size_t(n) * sizeof(half), cudaMemcpyDeviceToHost), "output read");
    const bool zeroOutput = std::all_of(output.begin(), output.end(), [](half value) { return __half2float(value) == 0.0f; });
    FastllmCudaGraphExecDestroy(exec);
    FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(pins);
    FastllmCudaForceFree(canary);
    for (void *pointer : {static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c),
                         scales, static_cast<void *>(globalScale), static_cast<void *>(locks)}) {
        Check(cudaFree(pointer), "cleanup");
    }
    std::printf("%s: %s graph after M=64 prefill, canary changed bytes=%zu, zero output=%d\n",
                changed == 0 && zeroOutput ? "PASS" : "FAIL", fp8 ? "FP8" : "NVFP4", changed, int(zeroOutput));
    return changed == 0 && zeroOutput ? 0 : 1;
}
