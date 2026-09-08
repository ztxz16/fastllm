#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "models/qwen4_tp_sampling.h"

#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace fastllm;

static std::pair<int, float> Top1(const std::vector<float> &values,
                                  int begin, int end) {
    Data input(FLOAT32, {1, end - begin},
               std::vector<float>(values.begin() + begin, values.begin() + end));
    Data output(FLOAT32, {1, 2});
    input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    output.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    output.Allocate(false);
    if (!FastllmCudaTopK(input, output, 1)) throw std::runtime_error("TopK failed");
    output.ToDevice(DataDevice::CPU);
    const float *result = reinterpret_cast<const float *>(output.cpuData);
    return {(int)(result[0] + 1e-3f) + begin, result[1]};
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::mt19937 random(731);
    int checks = 0;
    for (int vocabulary : {1024, 2053, 248320}) {
        for (int ranks : {2, 3, 4, 8}) {
            if (vocabulary / 256 < ranks) continue;
            for (int pattern = 0; pattern < 9; ++pattern) {
                std::vector<float> values(vocabulary, -100.0f);
                for (int i = 0; i < vocabulary; ++i) {
                    if (pattern == 0) values[i] = (float)(random() % 10000) / 1000;
                    if (pattern == 1) values[i] = (float)(random() % 3);
                    if (pattern == 2) values[i] = 0;
                    if (pattern == 3) values[i] = -std::numeric_limits<float>::infinity();
                    if (pattern == 4) values[i] = std::numeric_limits<float>::quiet_NaN();
                }
                if (pattern == 5 || pattern == 6) {
                    // The later token wins because lane 128 precedes lane 1.
                    values[1] = values[vocabulary - vocabulary % 256 - 128] =
                        pattern == 5 ? 17 : std::numeric_limits<float>::infinity();
                }
                if (pattern == 7) {
                    // Same-lane ties must retain the earlier global row.
                    values[256] = values[768] = 17;
                }
                if (pattern == 8) {
                    // Maxima on both sides of every shard boundary and tail.
                    for (int r = 1; r < ranks; ++r) {
                        int start = qwen4_tp::VocabRange(vocabulary, ranks, r).first;
                        values[start - 1] = values[start] = 17;
                    }
                    values.back() = 17;
                }
                auto expected = Top1(values, 0, vocabulary);
                std::pair<int, float> actual;
                int previousEnd = 0;
                for (int r = 0; r < ranks; ++r) {
                    auto range = qwen4_tp::VocabRange(vocabulary, ranks, r);
                    if (range.first != previousEnd || range.first % 256 ||
                        range.second <= range.first) throw std::runtime_error("invalid shard range");
                    previousEnd = range.second;
                    auto candidate = Top1(values, range.first, range.second);
                    if (r == 0 || qwen4_tp::Top1Before(candidate.first, candidate.second,
                                                       actual.first, actual.second)) actual = candidate;
                }
                if (previousEnd != vocabulary || actual != expected) {
                    std::cerr << "vocab=" << vocabulary << " ranks=" << ranks
                              << " pattern=" << pattern << " expected=" << expected.first
                              << " actual=" << actual.first << '\n';
                    throw std::runtime_error("sharded top1 differs from full CUDA TopK");
                }
                ++checks;
            }
        }
    }
    std::cout << "PASS: " << checks << " CUDA top1 comparisons, including ties/NaNs/infinities/tails\n";
}
