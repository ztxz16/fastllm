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
template<bool FP8> static void RingCase(int requests, int width, int heads) {
    using namespace naive_prefill_mma;
    const int kvHeads = heads / 8, count = requests * width, capacity = 128 + width;
    const size_t kStride = size_t(capacity) * kvHeads * 192 * (FP8 ? 1 : 2);
    const size_t vStride = size_t(capacity) * kvHeads * 128 * (FP8 ? 1 : 2);
    const size_t elements = size_t(count) * heads * 128;
    Buffer<BF> q(size_t(count) * heads * 192), expected(elements), actual(elements);
    Buffer<unsigned char> key(requests * kStride), value(requests * vStride);
    Buffer<int> live(count);
    Buffer<const void *> bindings(requests * 2);
    Buffer<float> sink(heads);
    Fill(q.data, size_t(count) * heads * 192, false, 83);
    Fill(key.data, requests * kStride / (FP8 ? 1 : 2), FP8, 127);
    Fill(value.data, requests * vStride / (FP8 ? 1 : 2), FP8, 139);
    std::vector<float> sinks(heads);
    for (int h = 0; h < heads; ++h) sinks[h] = (h % 3 - 1) * 80.f;
    Check(cudaMemcpy(sink.data, sinks.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<FastllmNaiveDSARow> rows(count);
    std::vector<SwaVerifyGroup> groups;
    for (int r = 0; r < requests; ++r) {
        for (int j = 0; j < width; ++j) {
            auto &row = rows[r * width + j];
            row.key = key.data + r * kStride; row.value = value.data + r * vStride;
            row.cache = bindings.data + r * 2;
            row.liveKeys = live.data + r * width + j;
            row.keyStride = kvHeads * 192; row.capacity = 128;
            row.verifyOffset = j; row.fp8 = FP8;
        }
        for (int j = 0; j < width; j += 4)
            groups.push_back({rows[r * width + j], r * width + j, std::min(4, width - j)});
    }
    constexpr size_t shared = sizeof(SwaQueryStorage<4>) > sizeof(naive_attention_bulk::Storage<128, FP8>)
        ? sizeof(SwaQueryStorage<4>) : sizeof(naive_attention_bulk::Storage<128, FP8>);
    auto optimized = SwaVerifyQueries<4, FP8>;
    auto original = naive_attention_bulk::Decode<128, true, FP8>;
    Check(cudaFuncSetAttribute(optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    Check(cudaFuncSetAttribute(original, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              sizeof(naive_attention_bulk::Storage<128, FP8>)));
    auto launch = [&] {
        for (int first = 0; first < (int)groups.size(); first += SwaVerifyLimit) {
            const int n = std::min(SwaVerifyLimit, (int)groups.size() - first);
            SwaVerifyBatch batch{};
            std::copy_n(groups.begin() + first, n, batch.groups);
            optimized<<<dim3(n, heads / 8), 256, shared, cudaStreamPerThread>>>(
                q.data, batch, sink.data, actual.data, heads, kvHeads);
        }
    };
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
    launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, 0));
    int pass = 0;
    for (int prefix : {0, 123, 127, 128, 707, 200000, 200001}) {
        std::vector<int> lengths(count);
        std::vector<const void *> ptrs(requests * 2);
        for (int r = 0; r < requests; ++r) {
            const int source = (r + pass) % requests;
            ptrs[r * 2] = key.data + source * kStride;
            ptrs[r * 2 + 1] = value.data + source * vStride;
            for (int j = 0; j < width; ++j) {
                int i = r * width + j;
                lengths[i] = prefix + r % 5 + j + 1;
                if (pass == 4 && i % 7 == 0) lengths[i] = 0;
                if (pass == 5) lengths[i] = i % 11 ? lengths[i] + (j % 3 ? 0 : 7) : 0;
                if (pass == 6 && j >= (width + 1) / 2) lengths[i] = 0;
            }
        }
        if (pass == 6) {
            // Padded candidates have not been written. Poison their V slots
            // so zero-probability reads cannot silently introduce NaNs.
            for (int r = 0; r < requests; ++r) {
                const size_t offset = (128 + (width + 1) / 2) * kvHeads * 128 * (FP8 ? 1 : 2);
                Check(cudaMemset(value.data + r * vStride + offset, 0xff, vStride - offset));
            }
        }
        Check(cudaMemcpy(live.data, lengths.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(bindings.data, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice));
        for (int first = 0; first < count; first += naive_attention_bulk::BatchLimit) {
            const int n = std::min(naive_attention_bulk::BatchLimit, count - first);
            naive_attention_bulk::Batch batch{};
            std::copy_n(rows.begin() + first, n, batch.rows);
            original<<<dim3(n, heads / 8), 256, sizeof(naive_attention_bulk::Storage<128, FP8>), cudaStreamPerThread>>>(
                q.data + size_t(first) * heads * 192, batch, sink.data,
                expected.data + size_t(first) * heads * 128, heads, kvHeads);
        }
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<unsigned short> a(elements), b(elements);
        Check(cudaMemcpy(a.data(), expected.data, elements * 2, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(b.data(), actual.data, elements * 2, cudaMemcpyDeviceToHost));
        if (a != b) throw std::runtime_error("batched ring query reuse differs from independent query bits");
        ++pass;
    }
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    printf("RING QUERY PASS requests=%d width=%d heads=%d fp8=%d graph_replays=%d\n", requests, width, heads, FP8, pass);
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
        for (int width : {1, 2, 3, 4, 5, 6, 8, 11}) {
            RingCase<false>(17, width, 8);
            RingCase<true>(17, width, 16);
        }
        RingCase<false>(128, 11, 16);
        RingCase<true>(128, 11, 16);
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
