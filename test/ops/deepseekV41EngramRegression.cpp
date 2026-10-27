#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-engram.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace fastllm;
static void Check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
        return 77;
    std::vector<std::vector<int>> groups{{0}};
    if (devices >= 3)
        groups.push_back({2, 0, 1});
    if (devices >= 8)
        groups.push_back({0, 1, 2, 3, 4, 5, 6, 7});
    // GPU tables deliberately leave 16 GiB free for the model and KV cache.
    for (const auto &group : groups)
        for (int device : group) {
            size_t available = 0, total = 0;
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaMemGetInfo(&available, &total) != cudaSuccess || available < (size_t(17) << 30))
                return 77;
        }
    try {
        const int rows = 1009, dim = 512, block = 32;
        std::vector<uint8_t> weights(size_t(rows) * dim), scales(size_t(rows) * (dim / block));
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = uint8_t(i * 137 + i / dim);
        for (size_t i = 0; i < scales.size(); ++i)
            scales[i] = uint8_t(i * 13 % 256);
        const FP8E4M3ToFP32Manager fp8;
        for (const auto &group : groups) {
            bool peerAccess = true;
            for (int device : group)
                for (int peer : group)
                    if (device != peer) {
                        int accessible = 0;
                        peerAccess &=
                            cudaDeviceCanAccessPeer(&accessible, device, peer) == cudaSuccess &&
                            accessible;
                    }
            if (!peerAccess)
                continue;
            void *table =
                FastllmCudaV41CreateEngram(group, weights.data(), scales.data(), rows, dim, block);
            Check(table != nullptr, "create table");
            for (int count : {1, 17, 511, 4096, 49152}) {
                std::vector<int64_t> ids(count);
                std::vector<uint16_t> expected(size_t(count) * dim);
                for (int i = 0; i < count; ++i) {
                    ids[i] = (i * 997 + count - 1) % rows;
                    for (int j = 0; j < dim; ++j) {
                        const uint8_t exponent = scales[ids[i] * (dim / block) + j / block];
                        const float factor = exponent == 255
                                                 ? std::numeric_limits<float>::quiet_NaN()
                                                 : std::ldexp(1.f, int(exponent) - 127);
                        const float value = fp8.dict[weights[ids[i] * dim + j]] * factor;
                        expected[size_t(i) * dim + j] = Float32ToBFloat16RNEBits(value);
                    }
                }
                for (int device : group) {
                    FastllmCudaSetDevice(device);
                    Data out(DataType::BFLOAT16, {1, count, dim});
                    out.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                    out.Allocate(false);
                    for (int repeat = 0; repeat < 3; ++repeat)
                        Check(FastllmCudaV41GatherEngram(table, ids, out), "gather");
                    Check(cudaDeviceSynchronize() == cudaSuccess, "synchronize");
                    std::vector<uint16_t> actual(expected.size());
                    Check(cudaMemcpy(actual.data(), out.cudaData, out.GetBytes(),
                                     cudaMemcpyDeviceToHost) == cudaSuccess,
                          "read");
                    for (size_t i = 0; i < actual.size(); ++i) {
                        const bool nan = (expected[i] & 0x7f80) == 0x7f80 && (expected[i] & 0x7f);
                        if (!(actual[i] == expected[i] ||
                              (nan && (actual[i] & 0x7f80) == 0x7f80 && (actual[i] & 0x7f)))) {
                            std::cerr << "device=" << device << " count=" << count
                                      << " element=" << i
                                      << " weight=" << int(weights[ids[i / dim] * dim + i % dim])
                                      << " scale="
                                      << int(scales[ids[i / dim] * (dim / block) + i % dim / block])
                                      << " expected=" << expected[i] << " actual=" << actual[i]
                                      << std::endl;
                            throw std::runtime_error("gather bits");
                        }
                    }
                    ids[0] = -1;
                    Check(!FastllmCudaV41GatherEngram(table, ids, out), "invalid row accepted");
                    ids[0] = (count - 1) % rows;
                }
                std::cout << "ranks=" << group.size() << " rows=" << count << " all_rank_bits=pass"
                          << std::endl;
            }
            FastllmCudaV41FreeEngram(table);
        }
        Check(!FastllmCudaV41CreateEngram({0, 0}, weights.data(), scales.data(), rows, dim, block),
              "duplicate rank accepted");
        std::cout << "PASS Engram sharded gather" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
