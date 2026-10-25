#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
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

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static unsigned FloatKey(float value) {
    unsigned bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

static void Upload(Data &data, const void *source, size_t bytes) {
    data.ToDevice(CUDA, {0}, false);
    data.Allocate(false);
    Check(cudaMemcpy(data.cudaData, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess,
          "input upload failed");
}

static std::vector<int> Reference(const std::vector<float> &scores,
                                 const std::vector<uint8_t> *masks,
                                 int sequence, int count, int width,
                                 int ratio, int start, int blockSize) {
    const int blocks = (count + blockSize - 1) / blockSize;
    std::vector<int> expected;
    for (int row = 0; row < sequence; ++row) {
        std::vector<int> selected;
        const int visible = std::min(count, (start + row + 1) / ratio);
        for (int index = 0; index < visible; ++index) {
            if (!masks || (*masks)[size_t(row) * blocks + index / blockSize])
                selected.push_back(index);
        }
        std::sort(selected.begin(), selected.end(), [&](int a, int b) {
            const unsigned ka = FloatKey(scores[size_t(row) * count + a]);
            const unsigned kb = FloatKey(scores[size_t(row) * count + b]);
            return ka == kb ? a < b : ka > kb;
        });
        if (selected.size() > size_t(width)) selected.resize(width);
        std::sort(selected.begin(), selected.end());
        selected.resize(width, -1);
        expected.insert(expected.end(), selected.begin(), selected.end());
    }
    return expected;
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        FastllmCudaSetDevice(0);
        unsetenv("FASTLLM_DSV41_LEGACY_TOPK");
        std::mt19937 random(1627);
        int cases = 0;
        for (int count : {1, 31, 127, 512, 513, 2048, 8192, 32769, 65537}) {
            for (int sequence : {1, 3}) {
                for (int pattern : {0, 1, 2, 3}) {
                    for (int maskMode : {0, 1, 2, 3}) {
                        for (int blockSize : {8, 7}) {
                            const int ratio = 2;
                            const int start = pattern == 3 ? 3 : std::max(0, count * ratio - 4);
                            const int width = std::min(512, count);
                            const int blocks = (count + blockSize - 1) / blockSize;
                            std::vector<float> scores(size_t(sequence) * count);
                            for (size_t i = 0; i < scores.size(); ++i) {
                                switch (pattern) {
                                    case 0: scores[i] = float(int(random() % 2001) - 1000) / 137.f; break;
                                    case 1: scores[i] = 1.f; break;
                                    case 2: scores[i] = float((i * 17) % 53) / 8.f; break;
                                    default: scores[i] = std::ldexp(float(i % 19), -18); break;
                                }
                            }
                            std::vector<uint8_t> masks(size_t(sequence) * blocks);
                            for (size_t i = 0; i < masks.size(); ++i)
                                masks[i] = maskMode == 1 ? 1 : maskMode == 2 ? (i % 7 == 0) : 0;

                            Data score(FLOAT32, {1, sequence, count});
                            Data mask(INT8, {1, sequence, blocks}), output;
                            Upload(score, scores.data(), scores.size() * sizeof(float));
                            if (maskMode) Upload(mask, masks.data(), masks.size());
                            Check(FastllmCudaDeepSeekV41IndexerTopK(score, maskMode ? &mask : nullptr,
                                  512, ratio, start, blockSize, output), "Top-K rejected input");
                            const auto expected = Reference(scores, maskMode ? &masks : nullptr,
                                sequence, count, width, ratio, start, blockSize);
                            std::vector<int> actual(expected.size());
                            Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost) == cudaSuccess, "output download failed");
                            Check(actual == expected, "Top-K differs from independent CPU reference");
                            ++cases;
                        }
                    }
                }
            }
        }
        std::printf("TOPK_PASS cases=%d CPU exact, ties/masks/causal/tails/nonpower-block\n", cases);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
