// Exact index-set, tie-order and graph-replay regression for the CUDA indexer.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static unsigned FloatKey(float value) {
    unsigned bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
static void Allocate(Data &data, DataType type, const std::vector<int> &dims, int device) {
    data.dataType = type;
    data.UpdateUnitSize();
    data.dataDevice = CUDA;
    data.dataDeviceIds = {device};
    data.Resize(dims);
    data.Allocate(false);
}

static void Run(int device, int rows, int length, int blockSize, int pattern, bool masked) {
    const int blocks = (length + blockSize - 1) / blockSize;
    const int width = std::min(512, length), ratio = 2, start = length;
    Data scores, candidates, output;
    Allocate(scores, FLOAT32, {1, rows, length}, device);
    Allocate(candidates, INT8, {1, rows, blocks}, device);
    std::vector<float> host(size_t(rows) * length);
    std::vector<unsigned char> mask(size_t(rows) * blocks);
    std::mt19937 random(173 + device + rows + length);
    for (auto &value : host) {
        value = pattern == 0   ? float(int(random() % 100000) - 50000) * .001f
                : pattern == 1 ? float(int(random() % 13) - 6)
                               : 1.f;
    }
    for (auto &value : mask)
        value = random() % 5 == 0;
    Cuda(cudaMemcpyAsync(scores.cudaData, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
                         cudaStreamPerThread));
    Cuda(cudaMemcpyAsync(candidates.cudaData, mask.data(), mask.size(), cudaMemcpyHostToDevice,
                         cudaStreamPerThread));
    auto launch = [&] {
        Check(FastllmCudaDeepSeekV41IndexerTopK(scores, masked ? &candidates : nullptr, width, ratio, start,
                                                blockSize, output),
              "indexer top-k rejected");
    };
    launch();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    launch();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        std::rotate(host.begin(), host.begin() + 1, host.end());
        if (replay == 2)
            std::fill(mask.begin(), mask.end(), 0);
        Cuda(cudaMemcpyAsync(scores.cudaData, host.data(), host.size() * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
        Cuda(cudaMemcpyAsync(candidates.cudaData, mask.data(), mask.size(), cudaMemcpyHostToDevice,
                             cudaStreamPerThread));
        Cuda(cudaMemsetAsync(output.cudaData, 0x55, rows * width * sizeof(int), cudaStreamPerThread));
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<int> actual(rows * width), expected(rows * width, -1);
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            std::vector<int> indices;
            const int visible = std::min(length, (start + row + 1) / ratio);
            for (int index = 0; index < visible; ++index)
                if (!masked || mask[row * blocks + index / blockSize])
                    indices.push_back(index);
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                unsigned ka = FloatKey(host[row * length + a]);
                unsigned kb = FloatKey(host[row * length + b]);
                return ka == kb ? a < b : ka > kb;
            });
            if (indices.size() > size_t(width))
                indices.resize(width);
            std::sort(indices.begin(), indices.end());
            std::copy(indices.begin(), indices.end(), expected.begin() + row * width);
        }
        Check(actual == expected, "CPU reference or changed-input graph replay mismatch");
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        int cases = 0;
        for (int device = 0; device < devices; ++device) {
            Cuda(cudaSetDevice(device));
            for (int rows : {1, 2, 3, 4, 5, 6, 7, 8})
                for (int length : {1, 255, 512, 513, 1130, 2310, 8192, 8193, 16384, 16385, 32769})
                    for (bool masked : {false, true})
                        for (int pattern : {0, 1, 2}) {
                            Run(device, rows, length, pattern == 1 ? 3 : 8, pattern, masked);
                            ++cases;
                        }
        }
        std::cout << "PASS " << cases
                  << " cases x 3 changing-input graph replays; CPU index/tie/padding oracle" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
