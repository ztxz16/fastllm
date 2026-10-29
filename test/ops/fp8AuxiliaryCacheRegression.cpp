// Cached slices must be invalidated even when the source has never run GEMM.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

using namespace fastllm;

static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void Replace(Data &owner, unsigned char value, float scale) {
    Data replacement(FP8_E4M3, {256, 128});
    replacement.blockM = replacement.blockK = 32;
    replacement.scales.assign(8 * 4, scale);
    replacement.ToDevice(DataDevice::CUDA, {0}, false);
    replacement.Allocate(false);
    Check(cudaMemset(replacement.cudaData, value, replacement.GetBytes()) == cudaSuccess, "fill weight");
    owner.CopyFrom(replacement);
    // CopyFrom copies tensor storage; quantized weight metadata is caller-owned.
    owner.blockM = owner.blockK = 32;
    owner.scales = replacement.scales;
}

int main() {
    try {
        SetThreads(4);
        FastllmCudaSetDevice(0);
        cudaDeviceProp properties{};
        Check(cudaGetDeviceProperties(&properties, 0) == cudaSuccess, "device properties");
        if (properties.major < 8)
            return 77;
        Data owner;
        for (int generation = 0; generation < 3; ++generation) {
            if (generation == 2)
                owner.FreeSpace();
            const unsigned char value = 32 + generation;
            const float scale = float(1 << generation);
            Replace(owner, value, scale);
            Data *views[] = {
                FastllmCudaFP8RowShard(owner, 32, 32),
                FastllmCudaFP8PairedRowShard(owner, 32, 32),
                FastllmCudaFP8ColumnShard(owner, 32, 32),
            };
            for (Data *view : views) {
                Check(view && view->cudaData, "missing shard");
                unsigned char actual = 0;
                Check(cudaMemcpy(&actual, view->cudaData, 1, cudaMemcpyDeviceToHost) == cudaSuccess,
                      "read shard");
                Check(actual == value && view->scales.front() == scale,
                      "stale shard after source replacement/release");
            }
            Check(FastllmCudaFP8RowShard(owner, 32, 32) == views[0], "row cache miss");
            Check(FastllmCudaFP8PairedRowShard(owner, 32, 32) == views[1], "paired-row cache miss");
            Check(FastllmCudaFP8ColumnShard(owner, 32, 32) == views[2], "column cache miss");
        }
        std::puts("PASS FP8 row/paired-row/column caches: reuse, CopyFrom and FreeSpace invalidation");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
