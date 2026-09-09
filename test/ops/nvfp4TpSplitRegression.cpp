#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace fastllm;

// Check the packed nibbles, planar E4M3 scales, and unequal gate/up global
// scales byte-for-byte after non-contiguous row and column TP splits.
int main() {
    std::vector<int> devices = {0, 1, 2, 3};
    if (FastllmCudaGetDeviceCount() < 4) {
        std::cout << "SKIP: this regression requires four visible CUDA GPUs\n";
        return 77;
    }
    for (int testCase = 0; testCase < 4; ++testCase) {
        const int axis = testCase % 2;
        const bool withEmptyShards = testCase >= 2;
        const int activeRanks = withEmptyShards ? 2 : 4;
        constexpr int rows = 256, cols = 128;
        const int rowSpan = rows / 2 / activeRanks;
        const int colSpan = cols / 2 / activeRanks;
        Data source(NVFP4_BLOCK_16_E4M3, {rows, cols});
        source.blockK = 1;
        source.blockM = 16;
        const std::vector<float> globalScales = {0.125f, 0.75f};
        source.scales = globalScales;
        source.Allocate();
        std::vector<uint8_t> original(source.GetBytes());
        for (size_t i = 0; i < original.size(); ++i) original[i] = (i * 37 + i / 11) % 256;
        std::memcpy(source.cpuData, original.data(), original.size());
        DivisionScheme scheme;
        for (int rank = 0; rank < 4; ++rank) {
            if (withEmptyShards && rank % 2 != 0) {
                scheme[rank] = {{0, 0}};
                continue;
            }
            const int part = withEmptyShards ? rank / 2 : rank;
            if (axis == 0) {
                scheme[rank] = {{part * rowSpan, (part + 1) * rowSpan},
                    {rows / 2 + part * rowSpan, rows / 2 + (part + 1) * rowSpan}};
            } else {
                scheme[rank] = {{part * colSpan, (part + 1) * colSpan},
                    {cols / 2 + part * colSpan, cols / 2 + (part + 1) * colSpan}};
            }
        }
        Data bias;
        if (!SplitMultiCudaWeight(source, bias, devices, scheme, axis, true, true)) {
            throw std::runtime_error("compact NVFP4 split failed");
        }
        for (int rank = 0; rank < 4; ++rank) {
            Data &shard = *source.multiDeviceDatas.at(rank);
            if (shard.scales != globalScales) {
                throw std::runtime_error("global scales lost");
            }
            const bool empty = withEmptyShards && rank % 2 != 0;
            int nr = shard.dims[0], nc = shard.dims[1];
            if (nr != (axis == 0 ? (empty ? 0 : rows / activeRanks) : rows) ||
                nc != (axis == 1 ? (empty ? 0 : cols / activeRanks) : cols)) {
                throw std::runtime_error("incorrect shard dimensions");
            }
            if (empty) {
                if (shard.Count(0) != 0) throw std::runtime_error("nonempty tensor for an empty TP range");
                continue;
            }
            FastllmCudaSetDevice(rank);
            shard.ToDevice(DataDevice::CPU);
            const int part = withEmptyShards ? rank / 2 : rank;
            for (int row = 0; row < nr; ++row) {
                for (int col = 0; col < nc; col += 16) {
                    int sr = axis == 0 ? (row < rowSpan ? part * rowSpan + row
                        : rows / 2 + part * rowSpan + row - rowSpan) : row;
                    int sc = axis == 1 ? (col < colSpan ? part * colSpan + col
                        : cols / 2 + part * colSpan + col - colSpan) : col;
                    if (std::memcmp(shard.cpuData + (row * nc + col) / 2,
                                    original.data() + (sr * cols + sc) / 2, 8) ||
                        shard.cpuData[nr * nc / 2 + (row * nc + col) / 16] !=
                        original[rows * cols / 2 + (sr * cols + sc) / 16]) {
                        throw std::runtime_error("packed values or planar scales differ");
                    }
                }
            }
        }
    }
    std::cout << "PASS: four-GPU compact NVFP4 row/column shards and scales are lossless, including empty ranks\n";
}
