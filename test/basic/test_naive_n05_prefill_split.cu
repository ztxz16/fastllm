// Compare split attention with the established fused kernel, bit for bit.
#include "../../src/devices/cuda/models/naive-n05-prefill-split.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

using BF = __nv_bfloat16;
static void Check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
template <class T> struct Buffer {
    T *data = nullptr;
    explicit Buffer(size_t count) { Check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
};
static void Fill(BF *dst, size_t count, unsigned seed) {
    std::vector<BF> values(count);
    for (auto &x : values) {
        seed = seed * 1664525u + 1013904223u;
        x = __float2bfloat16(((int)(seed >> 16) - 32768) / 16384.f);
    }
    Check(cudaMemcpy(dst, values.data(), count * sizeof(BF), cudaMemcpyHostToDevice));
}
static void Case(int rows, int past, int heads, bool causal, bool withSink, int indexTile) {
    using namespace naive_prefill_mma;
    const int kvHeads = heads / 16, keys = past + rows, count = 2048;
    const size_t outputs = (size_t)rows * heads * 128;
    Buffer<BF> q((size_t)rows * heads * 192), k((size_t)keys * kvHeads * 192), v((size_t)keys * kvHeads * 128),
        reference(outputs), actual(outputs), probabilities((size_t)rows * heads * 2056);
    Buffer<int> ids((size_t)rows * count);
    Buffer<float> bias(heads);
    Fill(q.data, (size_t)rows * heads * 192, 53);
    Fill(k.data, (size_t)keys * kvHeads * 192, 71);
    Fill(v.data, (size_t)keys * kvHeads * 128, 113);
    std::vector<int> selected((size_t)rows * count);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < count; ++col) {
            int key = (col * 137 + row * 13) % keys;
            if (row == 0 || col % 97 == 0)
                key = -1;
            else if (col % 193 == 1)
                key = keys + 3;
            selected[(size_t)row * count + col] = key;
        }
    std::vector<float> sink(heads);
    for (int h = 0; h < heads; ++h)
        sink[h] = (h % 3 - 1) * 80.f;
    Check(cudaMemcpy(ids.data, selected.data(), selected.size() * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(bias.data, sink.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
    const float *s = withSink ? bias.data : nullptr;
    FastllmNaiveKV key{}, value{};
    key.base = k.data;
    value.base = v.data;
    auto fused = Generic<2048, 0, 2>;
    Check(cudaFuncSetAttribute(fused, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage<2048, 2>)));
    const dim3 grid(rows, heads / 16);
    fused<<<grid, 256, sizeof(Storage<2048, 2>)>>>(q.data, key, value, ids.data, s, reference.data, heads, kvHeads,
                                                   kvHeads * 192, keys, count, past, causal, indexTile);
    SplitQK<2048, 0, 2><<<grid, 256, sizeof(QKStorage<2>)>>>(q.data, key, ids.data, probabilities.data, heads, kvHeads,
                                                             kvHeads * 192, keys, count, past, causal, indexTile);
    SplitNormalize<2048><<<dim3(rows, heads / 8), 256>>>(probabilities.data, s, heads, count);
    SplitPV<2048, 0, 2><<<grid, 256, sizeof(PVStorage<2>)>>>(probabilities.data, value, ids.data, actual.data, heads,
                                                             kvHeads, keys, count, past, causal, indexTile);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    std::vector<unsigned short> a(outputs), b(outputs);
    Check(cudaMemcpy(a.data(), reference.data, outputs * sizeof(BF), cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(b.data(), actual.data, outputs * sizeof(BF), cudaMemcpyDeviceToHost));
    if (a != b)
        throw std::runtime_error("Split attention differs from fused BF16 output");
    std::printf("rows=%d past=%d heads=%d causal=%d sink=%d index_tile=%d exact=%zu\n", rows, past, heads, causal,
                withSink, indexTile, outputs);
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 8)
        return 77;
    try {
        Case(2048, 8192, 16, true, false, 0);
        Case(2053, 8192, 32, false, true, 0);
        Case(2048, 77952, 16, true, true, 256);
        Case(65, 198975, 16, true, false, 0);
        std::puts("PASS: split prefill attention matches fused output exactly");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
