// Exact index-set, tie-order and graph-replay regression for the CUDA indexer.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static bool longContext = false;

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
    const int width = std::min(512, length), ratio = 2;
    const int start = longContext ? 2 * length - rows : length;
    Data scores, candidates, output;
    Allocate(scores, FLOAT32, {1, rows, length}, device);
    Allocate(candidates, INT8, {1, rows, blocks}, device);
    std::vector<float> host(size_t(rows) * length);
    std::vector<unsigned char> mask(size_t(rows) * blocks);
    std::mt19937 random(173 + device + rows + length);
    for (auto &value : host) {
        if (pattern == 0) value = float(int(random() % 100000) - 50000) * .001f;
        else if (pattern == 1) value = float(int(random() % 13) - 6);
        else if (pattern == 2) value = 1.f;
        else if (pattern == 3) {
            // Many distinct FP32 keys in a single FP16 coarse bucket.
            unsigned bits = 0x3f800000u + random() % 10000;
            std::memcpy(&value, &bits, sizeof(value));
        } else if (pattern == 4) {
            value = float(int(random() % 200001) - 100000) * 100.f;
        } else {
            unsigned bits = random();
            if (pattern == 5) bits &= 0x800007ffu; // Signed zero and subnormals.
            else if ((bits & 0x7f800000u) == 0x7f800000u) bits &= ~0x00800000u;
            std::memcpy(&value, &bits, sizeof(value));
        }
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
        std::vector<int> expected(rows * width, -1);
        for (int row = 0; row < rows; ++row) {
            std::vector<int> indices;
            const int visible = std::min(length, (start + row + 1) / ratio);
            for (int index = 0; index < visible; ++index)
                if (!masked || mask[row * blocks + index / blockSize])
                    indices.push_back(index);
            std::partial_sort(indices.begin(), indices.begin() + std::min(indices.size(), size_t(width)),
                              indices.end(), [&](int a, int b) {
                unsigned ka = FloatKey(host[row * length + a]);
                unsigned kb = FloatKey(host[row * length + b]);
                return ka == kb ? a < b : ka > kb;
            });
            if (indices.size() > size_t(width))
                indices.resize(width);
            std::sort(indices.begin(), indices.end());
            std::copy(indices.begin(), indices.end(), expected.begin() + row * width);
        }
        for (bool graphReplay : {false, true}) {
            Cuda(cudaMemsetAsync(output.cudaData, 0x55, rows * width * sizeof(int), cudaStreamPerThread));
            if (graphReplay) Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            else launch();
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<int> actual(rows * width);
            Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
            Check(actual == expected, "CPU reference or changed-input graph replay mismatch");
        }
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}

static void RunCandidates(int rows, int length, int blockSize, int pattern) {
    const int blocks = (length + blockSize - 1) / blockSize;
    const int keep = pattern == 1 ? 16 : 2048, ratio = 2;
    Data scores, output;
    Allocate(scores, FLOAT32, {1, rows, length}, 0);
    std::vector<float> host(size_t(rows) * length);
    std::mt19937 rng(73 + rows + length + pattern);
    for (int replay = 0; replay < 3; ++replay) {
        for (size_t i = 0; i < host.size(); ++i) {
            float value = pattern == 0 ? float(int(rng() % 100003) - 50001) * .001f :
                          pattern == 1 ? float(int((i + replay) % 13) - 6) : 1.f;
            if ((i + replay) % 31 == 0) value = -INFINITY;
            if ((i + replay) % 127 == 0) value = std::numeric_limits<float>::quiet_NaN();
            host[i] = value;
        }
        const int start = std::max(0, 2 * length - rows - replay * 7);
        std::vector<uint8_t> expected(size_t(rows) * blocks);
        for (int row = 0; row < rows; ++row) {
            const int visible = std::min(length, (start + row + 1) / ratio);
            std::vector<unsigned> keys(blocks);
            for (int block = 0; block < blocks; ++block) {
                float maximum = -INFINITY;
                for (int j = block * blockSize; j < std::min(visible, (block + 1) * blockSize); ++j)
                    maximum = std::fmax(maximum, host[size_t(row) * length + j]);
                if (visible && block == (visible - 1) / blockSize) maximum = INFINITY;
                keys[block] = FloatKey(maximum);
            }
            auto sorted = keys;
            const int kth = std::min(keep, blocks) - 1;
            std::nth_element(sorted.begin(), sorted.begin() + kth, sorted.end(), std::greater<unsigned>());
            for (int block = 0; block < blocks; ++block)
                expected[size_t(row) * blocks + block] = keys[block] >= sorted[kth] && keys[block] > FloatKey(-INFINITY);
        }
        Cuda(cudaMemcpyAsync(scores.cudaData, host.data(), host.size() * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
        Check(FastllmCudaDeepSeekV41CandidateBlocks(scores, blockSize, keep, ratio, start, output),
              "candidate selection rejected");
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<uint8_t> actual(expected.size());
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size(), cudaMemcpyDeviceToHost));
        Check(actual == expected, "candidate CPU threshold/tie/newest-block oracle mismatch");
    }
}

int main(int argc, char **argv) {
    longContext = argc > 1 && std::string(argv[1]) == "--long-context";
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        int cases = 0;
        if (argc > 1 && std::string(argv[1]) == "--candidates") {
            Cuda(cudaSetDevice(0));
            for (int rows : {1, 2, 6, 8})
                for (int length : {1, 513, 16385, 131073, 262145, 1048576})
                    for (int block : {3, 8, 64})
                        for (int pattern : {0, 1, 2}) { RunCandidates(rows, length, block, pattern); ++cases; }
            std::cout << "PASS " << cases << " candidate masks x 3 changing inputs; CPU threshold/tie/causal oracle" << std::endl;
            return 0;
        }
        for (int device = 0; device < (longContext ? 1 : devices); ++device) {
            Cuda(cudaSetDevice(device));
            for (int rows : {1, 2, 3, 4, 5, 6, 7, 8})
                for (int length : (longContext
                        ? std::vector<int>{32769, 131073, 262145, 1048576}
                        : std::vector<int>{1, 255, 512, 513, 1130, 2310, 8192, 8193, 16384, 16385, 32769}))
                    for (bool masked : {false, true})
                        for (int pattern : {0, 1, 2}) {
                            Run(device, rows, length, pattern == 1 ? 3 : 8, pattern, masked);
                            ++cases;
                        }
        }
        if (longContext) {
            for (int rows : {1, 6, 8})
                for (int length : {65537, 262145})
                    for (bool masked : {false, true})
                        for (int pattern : {3, 4, 5, 6}) {
                            Run(0, rows, length, 8, pattern, masked);
                            ++cases;
                        }
        }
        Cuda(cudaSetDevice(0));
        Run(0, 6, 32769, 16384, 1, true);
        ++cases;
        std::cout << "PASS " << cases
                  << " cases x 3 changing-input graph replays; CPU index/tie/padding oracle" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
