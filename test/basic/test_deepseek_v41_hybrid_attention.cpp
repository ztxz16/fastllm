#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace fastllm;

#ifdef USE_CUDA
static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void Fill(Data &data, int seed) {
    data.Allocate(false);
    for (size_t i = 0; i < data.Count(0); ++i) {
        float value = std::sin(float(i * 17 + seed) * .013f) * .4f;
        if (data.dataType == FLOAT32) ((float *)data.cpuData)[i] = value;
        else {
            value = RoundFloat32ToBFloat16RNE(value);
            uint32_t bits; memcpy(&bits, &value, sizeof(bits));
            ((uint16_t *)data.cpuData)[i] = bits >> 16;
        }
    }
    data.ToDevice(CUDA);
}
int main() {
    try {
        bool tested = false;
        auto *executor = static_cast<Executor *>(GetExecutor());
        for (int device = 0; device < FastllmCudaGetDeviceCount(); ++device) {
            FastllmCudaSetDevice(device);
            if (FastllmCudaRuntimeArch() != 75) continue;
            tested = true;
            executor->SetFirstDevice("cuda:" + std::to_string(device));
            for (int heads : {32, 64}) for (int start : {0, 31, 127, 128, 2254})
            for (int width : {0, 33, 256, 1024}) {
                constexpr int window = 128, capacity = 512;
                Data q(BFLOAT16, {1, 1, heads, 512}), chunk(BFLOAT16, {1, 1, 512});
                Data ring(BFLOAT16, {1, window, 512}), kv(BFLOAT16, {1, capacity, 512});
                Data sink(FLOAT32, {heads}), indices(INT32, {1, 1, width}), expected, actual;
                Fill(q, 1); Fill(chunk, 2); Fill(ring, 3); Fill(kv, 4); Fill(sink, 5);
                indices.Allocate(false);
                for (int c = 0; c < width; ++c)
                    ((int *)indices.cpuData)[c] = c % 17 == 0 ? -1 : c % 29 == 0 ? capacity : (c * 7) % capacity;
                if (width) indices.ToDevice(CUDA);
                const float scale = 1.f / std::sqrt(512.f);
                Require(FastllmCudaDeepSeekV41SparseAttention(q, chunk, start ? &ring : nullptr,
                    width ? &kv : nullptr, width ? &indices : nullptr, sink, window, start, scale, expected),
                    "reference attention dispatch");
                expected.ToDevice(CPU);
                DataDict data{{"q", &q}, {"chunkKV", &chunk}, {"attnSink", &sink}, {"output", &actual}};
                if (start) data["ringKV"] = &ring;
                if (width) { data["compressedKV"] = &kv; data["cmpIdx"] = &indices; }
                for (bool hybrid : {false, true}) {
                    executor->Run("DeepSeekV41SparseAttention", data, {{"softmaxScale", scale}},
                        {{"windowSize", window}, {"startPos", start}, {"hybrid", hybrid}});
                    actual.ToDevice(CPU);
                    Require(expected.GetBytes() == actual.GetBytes() &&
                        !memcmp(expected.cpuData, actual.cpuData, expected.GetBytes()), "attention numerical mismatch");
                }
                q.Resize({1, 2, heads, 256});
                Require(!FastllmCudaDeepSeekV41HybridSparseAttention(q, chunk, &ring, &kv, &indices,
                    sink, window, start, 1.f, actual), "unsupported shape did not fall back");
            }
        }
        if (!tested) return 77;
        puts("hybrid attention: dispatch, bitwise parity and fallback PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

#else
int main() { return 77; }
#endif
