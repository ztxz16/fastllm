#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void Check(cudaError_t error) { Require(error == cudaSuccess, cudaGetErrorString(error)); }
static std::vector<unsigned char> Read(Data &data, int device) {
    Check(cudaSetDevice(device));
    Check(cudaDeviceSynchronize());
    std::vector<unsigned char> bytes(data.Count(0) * data.unitSize);
    Check(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}
int main() {
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess || !available)
        return 77;
    try {
        int checked = 0;
        for (int count : {1, 2, 8}) {
            if (count > available)
                continue;
            std::vector<int> devices(count);
            std::iota(devices.begin(), devices.end(), 0);
            for (DataType dtype : {FLOAT32, FLOAT16, BFLOAT16, INT32})
                for (int width : {3, 257, 512}) {
                    Data source(dtype, {1, 10, width}), destination;
                    PrepareMultiCudaReplicatedData(source, devices, false);
                    std::vector<std::vector<unsigned char>> oracle(count);
                    for (int rank = 0; rank < count; ++rank) {
                        Check(cudaSetDevice(rank));
                        auto &local = *source.multiDeviceDatas.at(rank);
                        local.Allocate();
                        oracle[rank].resize(local.GetBytes());
                        for (size_t i = 0; i < oracle[rank].size(); ++i)
                            oracle[rank][i] = (i * 31 + i / 13 + rank * 17) % 251;
                        Check(cudaMemcpy(local.cudaData, oracle[rank].data(), oracle[rank].size(),
                                         cudaMemcpyHostToDevice));
                    }
                    Require(MultiCudaCopyReplicatedRows(destination, source, 1, 8, devices),
                            "initial copy rejected");
                    std::vector<void *> addresses;
                    for (int device : devices)
                        addresses.push_back(destination.multiDeviceDatas.at(device)->cudaData);
                    const bool previous = MultiCudaSetPersistentAsyncDispatch(true);
                    for (int rows = 1; rows <= 8; ++rows)
                        for (int start = 0; start + rows <= 10; ++start) {
                            Require(MultiCudaCopyReplicatedRows(destination, source, start, rows, devices),
                                    "valid copy rejected");
                            Require(destination.IsTensorParallelReplicated() &&
                                        destination.dims == std::vector<int>({1, rows, width}),
                                    "replica metadata lost");
                            for (int rank = 0; rank < count; ++rank) {
                                auto &local = *destination.multiDeviceDatas.at(rank);
                                Require(local.cudaData == addresses[rank],
                                        "destination address changed within capacity");
                                const auto actual = Read(local, rank);
                                const size_t offset = size_t(start) * width * source.unitSize;
                                Require(
                                    std::equal(actual.begin(), actual.end(), oracle[rank].begin() + offset),
                                    "rank-local copy mismatch");
                                Require(Read(*source.multiDeviceDatas.at(rank), rank) == oracle[rank],
                                        "source changed");
                            }
                            ++checked;
                        }
                    MultiCudaSetPersistentAsyncDispatch(previous);
                    const auto before = Read(*destination.multiDeviceDatas.at(0), 0);
                    Require(!MultiCudaCopyReplicatedRows(source, source, 0, 1, devices),
                            "in-place copy accepted");
                    for (auto bad : std::vector<std::pair<int, int>>{{-1, 1}, {10, 1}, {0, 0}, {8, 3}})
                        Require(
                            !MultiCudaCopyReplicatedRows(destination, source, bad.first, bad.second, devices),
                            "invalid range accepted");
                    auto duplicate = devices;
                    duplicate.push_back(devices.back());
                    Require(!MultiCudaCopyReplicatedRows(destination, source, 0, 1, duplicate),
                            "duplicate rank accepted");
                    Require(Read(*destination.multiDeviceDatas.at(0), 0) == before,
                            "rejection changed output");
                    if (count > 1) {
                        const std::vector<int> subset{devices.back()};
                        Require(MultiCudaCopyReplicatedRows(destination, source, 0, 1, subset),
                                "device subset rejected");
                        Require(destination.multiDeviceDatas.size() == 1 &&
                                    destination.dataDeviceIds == subset,
                                "device change retained stale replicas");
                        const auto actual = Read(*destination.multiDeviceDatas.at(subset[0]), subset[0]);
                        Require(std::equal(actual.begin(), actual.end(), oracle.back().begin()),
                                "device subset bytes changed");
                    }
                }
        }
        std::cout << "PASS " << checked
                  << " replicated row slices; rank-local bytes, stable allocations and fallback checks"
                  << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
