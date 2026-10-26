#include "fastllm.h"
#include "devices/cuda/naive-n05-cuda.cuh"
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
static void AdversarialSamples(int keys) {
    const int rows = 5, past = keys - rows, top = 2048;
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
            std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
                unsigned aa = OrderedReference(v[a]), bb = OrderedReference(v[b]);
                return aa > bb || (aa == bb && a < b);
            });
            std::copy_n(order.data(), top, expected.data() + (size_t)row * top);
        }
        Data scores(FLOAT32), out;
        scores.Resize({rows, keys});
        scores.Allocate();
        std::memcpy(scores.cpuData, values.data(), values.size() * sizeof(float));
        scores.ToDevice(DataDevice::CUDA, {0}, true);
        FastllmCudaNaiveTopK(scores, past, top, out);
        Require(Read<int>(out) == expected, "sampled-threshold fallback differs from exact CPU order");
        ++checks;
        printf("adversarial mode=%d PASS\n", mode);
        fflush(stdout);
    }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        for (int keys : {8192, 16384, 82001, 131072, 131073, 200000})
            AdversarialSamples(keys);
        Require(cudaDeviceSynchronize() == cudaSuccess, "final sync failed");
        std::printf("PASS: exact sampled TopK, %d adversarial cases\n", checks);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
