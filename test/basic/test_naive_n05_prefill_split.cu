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
    explicit Buffer(size_t count) {
        if (count) Check(cudaMalloc(&data, count * sizeof(T)));
    }
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
static void Case(int rows, int past, int heads, bool causal, bool withSink, int indexTile, bool paged = false) {
    using namespace naive_prefill_mma;
    const int kvHeads = heads / 16, keys = past + rows, count = 2048;
    constexpr int pageSize = 32;
    const int pageCount = paged ? (keys + pageSize - 1) / pageSize : 0;
    const int storageRows = paged ? pageCount * pageSize : keys;
    const size_t offset = paged ? 8 : 0;
    const size_t keyElements = (size_t)storageRows * kvHeads * 192 + offset;
    const size_t valueElements = (size_t)storageRows * kvHeads * 128 + offset;
    const size_t outputs = (size_t)rows * heads * 128;
    Buffer<BF> q((size_t)rows * heads * 192), k(keyElements), v(valueElements),
        reference(outputs), actual(outputs), probabilities((size_t)std::min(rows, 257) * heads * 2056);
    Buffer<int> ids((size_t)rows * count);
    Buffer<float> bias(heads);
    Fill(q.data, (size_t)rows * heads * 192, 53);
    Fill(k.data, keyElements, 71);
    Fill(v.data, valueElements, 113);
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
    Buffer<int> pages(pageCount);
    if (paged) {
        std::vector<int> table(pageCount);
        for (int i = 0; i < pageCount; ++i) table[i] = pageCount - i - 1;
        Check(cudaMemcpy(pages.data, table.data(), pageCount * sizeof(int), cudaMemcpyHostToDevice));
    }
    SplitBF16KV key(k.data, pages.data, pageSize), value(v.data, pages.data, pageSize);
    key.offset = value.offset = offset;
    auto fused = Generic<2048, 0, 2>;
    Check(cudaFuncSetAttribute(fused, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage<2048, 2>)));
    const dim3 grid(rows, heads / 16);
    fused<<<grid, 256, sizeof(Storage<2048, 2>)>>>(q.data, key, value, ids.data, s, reference.data, heads, kvHeads,
                                                   kvHeads * 192, keys, count, past, causal, indexTile);
    for(int first=0;first<rows;first+=257) {
        int tile=std::min(257,rows-first);
        SplitQK<2048,0,2><<<dim3(tile,heads/16),QKStorage<2>::Threads,sizeof(QKStorage<2>)>>>(q.data,key,ids.data,
            probabilities.data,heads,kvHeads,kvHeads*192,keys,count,past,causal,indexTile,first,rows);
        SplitNormalize<2048><<<dim3(heads / 4,tile),256>>>(probabilities.data,s,heads);
        SplitPV<2048,0,2><<<dim3(tile,heads/16),PVStorage<2>::Threads,sizeof(PVStorage<2>)>>>(probabilities.data,value,ids.data,
            actual.data,heads,kvHeads,keys,count,past,causal,indexTile,first,rows);
    }
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
        Case(65, 0, 16, true, false, 0);
        Case(257, 8192, 32, true, true, 0, true);
        Case(2048, 0, 64, true, false, 0);
        Case(2053, 0, 16, false, true, 0);
        Case(4096, 0, 64, true, true, 256);
        Case(2048, 8192, 16, true, false, 0);
        Case(2053, 8192, 32, false, true, 0);
        Case(2048, 77952, 16, true, true, 256);
        Case(65, 198975, 16, true, false, 0);
        Case(4096, 65536, 64, true, false, 0);
        std::puts("PASS: split prefill attention matches fused output exactly");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
