#include "../../src/devices/cuda/models/naive-n05-swa-query.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using BF = __nv_bfloat16;
static void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
    T *data = nullptr;
    explicit Buffer(size_t n) { Check(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
};
static void Fill(void *dst, size_t n, bool fp8, unsigned seed) {
    std::vector<unsigned char> bytes(n * (fp8 ? 1 : 2));
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float x = ((int)(seed >> 16) - 32768) / 16384.f;
        if (fp8)
            bytes[i] = __nv_fp8_e4m3(x).__x;
        else {
            BF b = __float2bfloat16(x);
            std::memcpy(bytes.data() + i * 2, &b, 2);
        }
    }
    Check(cudaMemcpy(dst, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
static void Case(int rows, int past, int heads, bool fp8, int pageLen, bool sink, bool offset) {
    using namespace naive_prefill_mma;
    constexpr int Q = 4;
    int kvHeads = heads / 8, keys = past + rows, np = pageLen ? (keys + pageLen - 1) / pageLen : 0;
    int physical = pageLen ? np * pageLen : keys, stride = kvHeads * 192 + 16;
    size_t n = (size_t)rows * heads * 128;
    Buffer<BF> q((size_t)rows * heads * 192), a(n), b(n);
    Buffer<unsigned char> k(((size_t)physical * stride + 16) * (fp8 ? 1 : 2));
    Buffer<unsigned char> v(((size_t)physical * kvHeads * 128 + 16) * (fp8 ? 1 : 2));
    Buffer<float> bias(heads);
    Buffer<int> table(std::max(np, 1));
    Fill(q.data, (size_t)rows * heads * 192, false, 19);
    Fill(k.data, (size_t)physical * stride + 16, fp8, 71);
    Fill(v.data, (size_t)physical * kvHeads * 128 + 16, fp8, 113);
    std::vector<float> sb(heads);
    for (int i = 0; i < heads; ++i)
        sb[i] = (i % 3 - 1) * 80.f;
    Check(cudaMemcpy(bias.data, sb.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
    if (pageLen) {
        std::vector<int> pages(np);
        for (int i = 0; i < np; ++i)
            pages[i] = np - 1 - i;
        Check(cudaMemcpy(table.data, pages.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    }
    FastllmNaiveKV key(k.data, pageLen ? table.data : nullptr, pageLen, fp8);
    FastllmNaiveKV value(v.data, pageLen ? table.data : nullptr, pageLen, fp8);
    if (offset) {
        key.offset = 8;
        value.offset = 8;
    }
    auto original = Generic<128, 128>;
    Check(cudaFuncSetAttribute(original, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage<128>)));
    Check(
        cudaFuncSetAttribute(SwaQueryTile<Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SwaQueryStorage<Q>)));
    original<<<dim3(rows, heads / 8), 256, sizeof(Storage<128>)>>>(q.data, key, value, nullptr,
                                                                   sink ? bias.data : nullptr, a.data, heads, kvHeads,
                                                                   stride, keys, 128, past, true, 0);
    SwaQueryTile<Q><<<dim3((rows + Q - 1) / Q, heads / 8), 256, sizeof(SwaQueryStorage<Q>)>>>(
        q.data, key, value, sink ? bias.data : nullptr, b.data, rows, heads, kvHeads, stride, keys, past);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    std::vector<unsigned short> ref(n), actual(n);
    Check(cudaMemcpy(ref.data(), a.data, n * 2, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(), b.data, n * 2, cudaMemcpyDeviceToHost));
    if (ref != actual)
        throw std::runtime_error("SWA query reuse differs from original BF16 output");
    printf("rows=%d past=%d heads=%d fp8=%d page=%d sink=%d offset=%d exact=%zu\n", rows, past, heads, fp8, pageLen,
           sink, offset, n);
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        cudaDeviceProp prop{};
        Check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8)
            return 77;
        for (bool fp8 : {false, true}) {
            Case(1, 127, 8, fp8, 0, false, false);
            Case(5, 128, 16, fp8, 64, true, true);
            Case(65, 77952, 32, fp8, 30, false, true);
            Case(129, 198975, 16, fp8, 128, true, false);
            Case(1025, 77952, 16, fp8, 64, false, true);
            Case(4147, 82944, 16, fp8, 0, true, false);
        }
        puts("PASS: exact SWA query reuse");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
