#include "fastllm.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static int checks = 0;
static void Require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class T> static std::vector<T> Read(const Data &data) {
    size_t n = 1;
    for (int dim : data.dims)
        n *= dim;
    std::vector<T> values(n);
    Require(cudaMemcpy(values.data(), data.cudaData, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
            "read failed");
    return values;
}
static unsigned OrderedReference(float x) {
    unsigned b;
    std::memcpy(&b, &x, 4);
    if ((b & 0x7fffffffu) == 0)
        b = 0;
    return b & 0x80000000u ? ~b : b ^ 0x80000000u;
}
static void AdversarialSamples(int keys, int past) {
    const int rows = 5, top = 2048;
    for (int mode = 0; mode < 6; ++mode) {
        std::vector<float> values((size_t)rows * keys);
        std::vector<int> expected((size_t)rows * top);
        for (int row = 0; row < rows; ++row) {
            int count = past + row + 1;
            auto *v = values.data() + (size_t)row * keys;
            for (int i = 0; i < keys; ++i)
                v[i] = mode == 2 ? 7.f : float((i * 37) % 100003) / 100003.f;
            if (mode < 2)
                for (int i = 0; i < 1024; ++i)
                    v[(int)((int64_t)i * count / 1024)] = mode == 0 ? 10000.f : -10000.f;
            if (mode == 3)
                for (int i = 0; i < 3100; ++i)
                    v[i] = 2.f;
            if (mode >= 4) {
                unsigned pattern[] = {0x7fc00002u, 0x7fc00001u, 0x7f800000u, 0x3f800000u, 0u,         0x80000000u,
                                      0x80000001u, 0xbf800000u, 0xff800000u, 0xffc00001u, 0xffffffffu};
                for (int i = 0; i < count; ++i) {
                    unsigned bits = pattern[mode == 5 && i >= 1536 ? 10 : (i * 5 + row) % 11];
                    std::memcpy(v + i, &bits, 4);
                }
            }
            for (int i = count; i < keys; ++i) {
                unsigned bits = 0x7fffffffu;
                std::memcpy(v + i, &bits, 4);
            }
            std::vector<int> order(count);
            std::iota(order.begin(), order.end(), 0);
            const int keep = std::min(top, count);
            std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) {
                unsigned aa = OrderedReference(v[a]), bb = OrderedReference(v[b]);
                return aa > bb || (aa == bb && a < b);
            });
            auto *result = expected.data() + (size_t)row * top;
            std::copy_n(order.data(), keep, result);
            std::fill(result + keep, result + top, -1);
        }
        Data scores(FLOAT32), out(INT32);
        scores.Resize({rows, keys});
        scores.Allocate();
        std::memcpy(scores.cpuData, values.data(), values.size() * sizeof(float));
        scores.ToDevice(DataDevice::CUDA, {0}, true);
        out.Resize({rows, top});
        out.ToDevice(DataDevice::CUDA, {0}, false);
        out.Allocate(false);
        std::vector<size_t> allocations;
        {
            FastllmCudaAllocationRecorder record(&allocations);
            FastllmCudaNaiveTopK(scores, past, top, out);
        }
        Require(allocations.empty(), "batched Top2048 requested global scratch");
        Require(Read<int>(out) == expected, "TopK differs from exact CPU order");
        if (keys >= 16384) {
            Data scratch(INT32, {rows * (4096 * 2 + 2)});
            scratch.ToDevice(DataDevice::CUDA, {0}, false);
            scratch.Allocate(false);
            FastllmCudaNaiveTopK(scores, past, top, out, scratch.cudaData, scratch.GetBytes());
            Require(Read<int>(out) == expected, "sampled TopK differs from exact CPU order");
            ++checks;
        }
        ++checks;
        printf("adversarial mode=%d PASS\n", mode);
        fflush(stdout);
    }
}
// Increasing scores give an exact reference for every row, including causal
// padding and the first row of each tile. Preallocate the public result so
// the recorder measures only the operator's scratch requests.
static void CheckLargeBatch(int rows, int keys, int past, int top, int scratchRows = 0) {
    Data scores(FLOAT32, {rows, keys});
    scores.Allocate();
    for (int row = 0; row < rows; ++row) {
        auto *values = reinterpret_cast<float *>(scores.cpuData) + size_t(row) * keys;
        std::iota(values, values + keys, 0.f);
    }
    scores.ToDevice(DataDevice::CUDA, {0}, true);
    Data out(INT32, {rows, top});
    out.ToDevice(DataDevice::CUDA, {0}, false);
    out.Allocate(false);
    const int scratchWords = scratchRows * (4096 * 2 + 2);
    Data scratch(INT32, {scratchWords + 4});
    scratch.ToDevice(DataDevice::CUDA, {0}, false);
    scratch.Allocate(false);
    Require(cudaMemset(scratch.cudaData, 0xa5, scratch.GetBytes()) == cudaSuccess, "scratch init failed");
    std::vector<size_t> allocations;
    {
        FastllmCudaAllocationRecorder record(&allocations);
        FastllmCudaNaiveTopK(scores, past, top, out, static_cast<int *>(scratch.cudaData) + 2,
                             size_t(scratchWords) * sizeof(int));
    }
    const auto guard = Read<unsigned>(scratch);
    Require(guard[0] == 0xa5a5a5a5u && guard[1] == guard[0] &&
            guard[scratchWords + 2] == guard[0] && guard[scratchWords + 3] == guard[0],
            "TopK wrote outside the supplied workspace");
    if (top == 2048) {
        Require(allocations.empty(), "batched Top2048 requested global scratch");
    } else {
        // These full and compact cases would need larger buffers without row tiling.
        for (size_t bytes : allocations)
            Require(bytes <= (size_t(16) << 20), "generic TopK scratch grew with batch rows");
    }
    auto actual = Read<int>(out);
    for (int row = 0; row < rows; ++row)
        for (int i = 0; i < top; ++i)
            Require(actual[size_t(row) * top + i] == std::max(-1, past + row - i),
                    "wrong TopK index at a causal or tile boundary");
    ++checks;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        for (int keys : {4096, 4097, 7335, 8191, 8192, 16384, 82001, 131072, 131073, 200000})
            AdversarialSamples(keys, keys - 5);
        // Cold-prefix padding, the K boundary, and many identical candidates.
        for (int past : {0, 2046, 4094})
            AdversarialSamples(7335, past);
        CheckLargeBatch(7808, 8586, 0, 2048);
        CheckLargeBatch(778, 8586, 7808, 2048);
        CheckLargeBatch(385, 16385, 15999, 2048, 128);
        CheckLargeBatch(129, 32768, 32000, 2048, 127);
        CheckLargeBatch(1297, 2301, 3, 1024);
        CheckLargeBatch(1231, 8586, 2000, 1000);
        CheckLargeBatch(33, 79, 0, 100);
        Require(cudaDeviceSynchronize() == cudaSuccess, "final sync failed");
        std::printf("PASS: exact TopK, %d adversarial cases\n", checks);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
