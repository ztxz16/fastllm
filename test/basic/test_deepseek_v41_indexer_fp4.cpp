#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "../../src/devices/cuda/models/deepseekv41-indexer.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void Cuda(cudaError_t status) {
    Check(status == cudaSuccess, cudaGetErrorString(status));
}

static float Value(unsigned code) {
    const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return code & 8 ? -values[code & 7] : values[code & 7];
}

static void Upload(Data &data, const void *src, size_t bytes) {
    data.ToDevice(CUDA, {0}, false);
    data.Allocate(false);
    Cuda(cudaMemcpy(data.cudaData, src, bytes, cudaMemcpyHostToDevice));
}

template<class T> static std::vector<T> Download(Data &data) {
    std::vector<T> out(data.GetBytes() / sizeof(T));
    Cuda(cudaMemcpy(out.data(), data.cudaData, data.GetBytes(), cudaMemcpyDeviceToHost));
    return out;
}

int main(int argc, char **argv) {
    try {
        if (!FastllmCudaGetDeviceCount()) return 77;
        FastllmCudaSetDevice(0);
        std::mt19937 random(68101);
        const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        int cases = 0;
        for (int tokens : bench ? std::vector<int>{32, 64, 128} : std::vector<int>{1, 3, 9, 17, 65}) {
            const int count = bench ? 492309 : tokens == 65 ? 32769 : 1027;
            const int batch = tokens == 17 ? 2 : 1, ratio = 2, position = count * ratio - tokens;
            Data q(BFLOAT16, {batch, tokens, 32, 128}), k(INT8, {batch, count, 68});
            Data weights(FLOAT32, {batch, tokens, 32});
            std::vector<uint16_t> query(size_t(batch) * tokens * 32 * 128);
            std::vector<float> original(query.size()), weight(size_t(batch) * tokens * 32);
            std::vector<uint8_t> key(size_t(batch) * count * 68);
            for (size_t r = 0; r < query.size() / 32; ++r) {
                const int exponent = int(random() % 9) - 6;
                for (int i = 0; i < 32; ++i) {
                    const float value = std::ldexp(Value(random() % 16), exponent);
                    original[r * 32 + i] = value;
                    unsigned bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    query[r * 32 + i] = bits >> 16;
                }
            }
            for (size_t r = 0; r < key.size() / 68; ++r) {
                for (int i = 0; i < 64; ++i) key[r * 68 + i] = uint8_t(random());
                for (int i = 0; i < 4; ++i) key[r * 68 + 64 + i] = 121 + random() % 9;
            }
            for (float &w : weight) w = float(int(random() % 201) - 100) / 177.f;
            Upload(q, query.data(), query.size() * sizeof(uint16_t));
            Upload(k, key.data(), key.size());
            Upload(weights, weight.data(), weight.size() * sizeof(float));
            Data baseline, actual, selectedBase, selectedActual;
            Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, k, ratio, position, baseline), "generic score");
            Check(FastllmCudaDeepSeekV41IndexerScoreFp4Query(q, weights, k, ratio, position, actual), "FP4 score");
            const auto ref = Download<float>(baseline), got = Download<float>(actual);
            double maxError = 0;
            for (size_t i = 0; i < got.size(); i += std::max<size_t>(1, got.size() / 4096)) {
                const int row = i / count, b = row / tokens, qr = row % tokens, kr = i % count;
                if (kr >= std::min(count, (position + qr + 1) / ratio)) {
                    // Generic builds may defer masking to TopK. Invalid score rows
                    // are not part of the score contract; final indices must match.
                    continue;
                }
                double expected = 0, norm = 0;
                const uint8_t *packed = key.data() + (size_t(b) * count + kr) * 68;
                for (int h = 0; h < 32; ++h) {
                    double dot = 0;
                    for (int c = 0; c < 128; ++c) {
                        dot += double(original[(size_t(row) * 32 + h) * 128 + c]) *
                            std::ldexp(Value((packed[c / 2] >> (4 * (c % 2))) & 15),
                                       int(packed[64 + c / 32]) - 127);
                    }
                    const double value = std::max(0., dot) * weight[size_t(row) * 32 + h];
                    expected += value;
                    norm += std::fabs(value);
                }
                const double error = std::fabs(got[i] - expected) / std::max(1., norm);
                maxError = std::max(maxError, error);
                Check(std::isfinite(got[i]) && error < 3e-6, "FP4 score vs CPU oracle");
                Check(std::fabs(ref[i] - expected) / std::max(1., norm) < 3e-6, "generic score vs CPU oracle");
            }
            Check(FastllmCudaDeepSeekV41IndexerTopK(baseline, nullptr, 512, ratio, position, 8, selectedBase),
                  "reference topk");
            Check(FastllmCudaDeepSeekV41IndexerTopK(actual, nullptr, 512, ratio, position, 8, selectedActual),
                  "native topk");
            Check(Download<int>(selectedBase) == Download<int>(selectedActual), "selected indices changed");
            setenv("FASTLLM_DSV41_LEGACY_INDEXER", "1", 1);
            Check(FastllmCudaDeepSeekV41IndexerScoreFp4Query(q, weights, k, ratio, position, actual), "explicit fallback");
            Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, k, ratio, position, baseline), "reference fallback");
            Check(Download<float>(actual) == Download<float>(baseline), "fallback differs");
            unsetenv("FASTLLM_DSV41_LEGACY_INDEXER");
            std::printf("PASS batch=%d tokens=%d count=%d error=%g exact-topk fallback\n",
                        batch, tokens, count, maxError);
            std::fflush(stdout);
            ++cases;
            if (bench) {
                cudaEvent_t begin, end;
                Cuda(cudaEventCreate(&begin));
                Cuda(cudaEventCreate(&end));
                for (bool native : {false, true}) {
                    const auto score = native ? FastllmCudaDeepSeekV41IndexerScoreFp4Query
                                              : FastllmCudaDeepSeekV41IndexerScore;
                    float total = 0;
                    for (int run = 0; run < 12; ++run) {
                        Cuda(cudaEventRecord(begin, cudaStreamPerThread));
                        Check(score(q, weights, k, ratio, position, actual), "benchmark score");
                        Cuda(cudaEventRecord(end, cudaStreamPerThread));
                        Cuda(cudaEventSynchronize(end));
                        float ms;
                        Cuda(cudaEventElapsedTime(&ms, begin, end));
                        if (run >= 2) total += ms;
                    }
                    std::printf("BENCH tokens=%d count=%d native=%d ms=%f\n", tokens, count, native, total / 10);
                    std::fflush(stdout);
                }
                Cuda(cudaEventDestroy(begin));
                Cuda(cudaEventDestroy(end));
            }
        }
        std::printf("PASS FP4 indexer cases=%d\n", cases);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
