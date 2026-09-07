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
    for (int axis : {0, 1}) {
        constexpr int rows = 256, cols = 128;
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
            if (axis == 0) {
                scheme[rank] = {{rank * 32, rank * 32 + 32}, {128 + rank * 32, 160 + rank * 32}};
            } else {
                scheme[rank] = {{rank * 16, rank * 16 + 16}, {64 + rank * 16, 80 + rank * 16}};
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
            FastllmCudaSetDevice(rank);
            shard.ToDevice(DataDevice::CPU);
            int nr = shard.dims[0], nc = shard.dims[1];
            if (nr != (axis == 0 ? rows / 4 : rows) ||
                nc != (axis == 1 ? cols / 4 : cols)) {
                throw std::runtime_error("incorrect shard dimensions");
            }
            for (int row = 0; row < nr; ++row) {
                for (int col = 0; col < nc; col += 16) {
                    int sr = axis == 0 ? (row < 32 ? rank * 32 + row : 128 + rank * 32 + row - 32) : row;
                    int sc = axis == 1 ? (col < 16 ? rank * 16 + col : 64 + rank * 16 + col - 16) : col;
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
    std::cout << "PASS: four-GPU compact NVFP4 row/column shards and scales are lossless\n";
}
