#include "model.h"
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
using namespace fastllm;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::compress_ratios;
    using DeepSeekV41Model::DsparkCommitPrefix;
    using DeepSeekV41Model::engram_layer_ids;
    using DeepSeekV41Model::isKvSource;
    using DeepSeekV41Model::v41DsparkTpDevices;
    using DeepSeekV41Model::window_size;
};
static void req(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
static void cu(cudaError_t e) {
    req(e == cudaSuccess, cudaGetErrorString(e));
}
static const std::vector<int> dev{0, 1, 2, 3, 4, 5, 6, 7};
static void fill(Data &x, DataType type, int rows, int seed) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize({1, rows, 512});
    PrepareMultiCudaReplicatedData(x, dev, false);
    std::vector<unsigned char> v(x.GetBytes());
    for (int d : dev) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = (i * 17 + i / 11 + seed + 31 * d) % 251;
        cu(cudaSetDevice(d));
        auto &l = *x.multiDeviceDatas.at(d);
        l.Allocate();
        cu(cudaMemcpy(l.cudaData, v.data(), v.size(), cudaMemcpyHostToDevice));
    }
}
static std::vector<unsigned char> read(Data &x, int d) {
    cu(cudaSetDevice(d));
    auto &l = *x.multiDeviceDatas.at(d);
    std::vector<unsigned char> v(l.GetBytes());
    cu(cudaMemcpy(v.data(), l.cudaData, v.size(), cudaMemcpyDeviceToHost));
    return v;
}
struct Fixture {
    DeepSeekV41RequestState state;
    DeepSeekV41SpecScratch scratch;
    int pos, forward;
    Fixture(Access &m, int p, int n) : pos(p), forward(n) {
        state.layers.resize(40);
        scratch.windowKV.resize(40);
        scratch.rawKV.resize(40);
        scratch.rawScore.resize(40);
        scratch.prevRawTail.resize(40);
        scratch.prevBlocks.resize(40);
        for (int l = 0; l < 40; ++l) {
            auto &c = state.layers[l];
            c.totalLen = p + n;
            fill(c.windowKV, BFLOAT16, 128, l);
            fill(scratch.windowKV[l], BFLOAT16, n, l + 71);
            if (!m.isKvSource[l])
                continue;
            int ratio = m.compress_ratios[l];
            c.compressedBlocks = (p + n) / ratio;
            c.rawTail = (p + n) % ratio;
            scratch.prevRawTail[l] = p % ratio;
            scratch.prevBlocks[l] = p / ratio;
            fill(scratch.rawKV[l], FLOAT32, p % ratio + n, l + 91);
            if (ratio > 1)
                fill(scratch.rawScore[l], FLOAT32, p % ratio + n, l + 99);
            fill(c.compressedKV, BFLOAT16, (p + n) / ratio, l + 3);
            fill(c.indexK, BFLOAT16, (p + n) / ratio, l + 5);
            if (ratio > 1) {
                fill(c.rawTailKV, FLOAT32, 2, l + 7);
                fill(c.rawTailScore, FLOAT32, 2, l + 11);
            }
        }
    }
};
struct CommitTestStorage {
    std::vector<void *> requests, descriptors;
    ~CommitTestStorage() {
        for (size_t i = 0; i < requests.size(); ++i) {
            cudaSetDevice(dev[i]);
            cudaFree(requests[i]);
            cudaFree(descriptors[i]);
        }
    }
};
static void attachFused(Fixture &f, Access &m) {
    auto storage = std::make_shared<CommitTestStorage>();
    storage->requests.resize(dev.size());
    storage->descriptors.resize(dev.size());
    for (size_t rank = 0; rank < dev.size(); ++rank) {
        int device = dev[rank];
        cu(cudaSetDevice(device));
        cu(cudaMalloc(&storage->requests[rank], 40 * sizeof(FastllmV41FullDecodeRequest)));
        cu(cudaMalloc(&storage->descriptors[rank], 40 * sizeof(FastllmV41VerifyCommitLayer)));
        auto ptr = [device](Data &x) -> void * {
            auto it = x.multiDeviceDatas.find(device);
            return it == x.multiDeviceDatas.end() ? nullptr : it->second->cudaData;
        };
        std::vector<FastllmV41FullDecodeRequest> requests;
        std::vector<FastllmV41VerifyCommitLayer> layers;
        for (int layer = 0; layer < 40; ++layer) {
            auto &c = f.state.layers[layer];
            requests.push_back({ptr(c.windowKV), ptr(c.compressedKV), ptr(c.indexK), ptr(c.rawTailKV),
                                ptr(c.rawTailScore), f.pos});
            layers.push_back({static_cast<FastllmV41FullDecodeRequest *>(storage->requests[rank]) + layer,
                              ptr(f.scratch.windowKV[layer]), ptr(f.scratch.rawKV[layer]),
                              ptr(f.scratch.rawScore[layer]), 128,
                              m.isKvSource[layer] ? m.compress_ratios[layer] : 0});
        }
        cu(cudaMemcpy(storage->requests[rank], requests.data(), requests.size() * sizeof(requests[0]),
                      cudaMemcpyHostToDevice));
        cu(cudaMemcpy(storage->descriptors[rank], layers.data(), layers.size() * sizeof(layers[0]),
                      cudaMemcpyHostToDevice));
    }
    int forward = f.forward;
    f.scratch.commitKV = [storage, forward](int accepted) {
        req(MultiCudaRunDeviceCallbacks(
                dev,
                [&](int rank, int) {
                    req(FastllmCudaV41CommitVerifiedPrefix(
                            static_cast<const FastllmV41VerifyCommitLayer *>(storage->descriptors[rank]), 40,
                            accepted, forward),
                        "fused launch");
                }),
            "fused dispatch");
    };
}
static void compare(Fixture &a, Fixture &b, Access &m) {
    req(a.state.totalLen == b.state.totalLen, "state length");
    for (int l = 0; l < 40; ++l) {
        auto &x = a.state.layers[l], &y = b.state.layers[l];
        req(x.totalLen == y.totalLen, "layer length");
        for (int d : dev)
            req(read(x.windowKV, d) == read(y.windowKV, d), "window data");
        if (!m.isKvSource[l])
            continue;
        req(x.rawTail == y.rawTail && x.compressedBlocks == y.compressedBlocks, "compressor lengths");
        for (auto pair :
             {std::make_pair(&x.compressedKV, &y.compressedKV), std::make_pair(&x.indexK, &y.indexK)}) {
            req(pair.first->dims == pair.second->dims, "compressed shape");
            for (int d : dev)
                req(read(*pair.first, d) == read(*pair.second, d), "compressed data");
        }
        if (x.rawTail)
            for (auto pair : {std::make_pair(&x.rawTailKV, &y.rawTailKV),
                              std::make_pair(&x.rawTailScore, &y.rawTailScore)}) {
                req(pair.first->dims == pair.second->dims, "tail shape");
                for (int d : dev)
                    req(read(*pair.first, d) == read(*pair.second, d), "tail data");
            }
    }
}
static void smallWindowCommit() {
    cu(cudaSetDevice(0));
    std::vector<uint16_t> input(8 * 512), initial(128 * 512, 0xabcd), output(initial.size());
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = uint16_t(i + 1);
    void *src = nullptr, *dst = nullptr, *requests = nullptr, *layers = nullptr;
    cu(cudaMalloc(&src, input.size() * 2));
    cu(cudaMalloc(&dst, initial.size() * 2));
    cu(cudaMalloc(&requests, sizeof(FastllmV41FullDecodeRequest)));
    cu(cudaMalloc(&layers, sizeof(FastllmV41VerifyCommitLayer)));
    cu(cudaMemcpy(src, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    for (int window : {1, 2, 7, 128})
        for (int accepted : {1, 2, 8}) {
            cu(cudaMemcpy(dst, initial.data(), initial.size() * 2, cudaMemcpyHostToDevice));
            FastllmV41FullDecodeRequest request;
            request.windowKV = dst;
            request.position = 127;
            FastllmV41VerifyCommitLayer layer{
                static_cast<FastllmV41FullDecodeRequest *>(requests), src, nullptr, nullptr, window, 0};
            cu(cudaMemcpy(requests, &request, sizeof(request), cudaMemcpyHostToDevice));
            cu(cudaMemcpy(layers, &layer, sizeof(layer), cudaMemcpyHostToDevice));
            req(FastllmCudaV41CommitVerifiedPrefix(static_cast<FastllmV41VerifyCommitLayer *>(layers), 1,
                                                   accepted, accepted),
                "small window launch");
            cu(cudaStreamSynchronize(cudaStreamPerThread));
            cu(cudaMemcpy(output.data(), dst, initial.size() * 2, cudaMemcpyDeviceToHost));
            auto expected = initial;
            for (int row = 0; row < accepted; ++row)
                std::copy_n(input.data() + row * 512, 512, expected.data() + ((127 + row) % window) * 512);
            req(output == expected, "small window bytes mismatch");
        }
    cu(cudaFree(src));
    cu(cudaFree(dst));
    cu(cudaFree(requests));
    cu(cudaFree(layers));
    std::cout << "PASS 12 single-token/small-window commits, untouched capacity checked" << std::endl;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8)
        return 77;
    try {
        SetThreads(16);
        const std::string name = "multicuda:0,1,2,3,4,5,6,7";
        SetDeviceMap({{name, 1}});
        FastllmMultiCudaSetDevice(dev);
        FastllmCudaSetNcclForceSync(false);
        Access m;
        m.block_cnt = 40;
        m.deviceMap = {{name, 1}};
        m.window_size = 128;
        m.isKvSource.assign(40, 0);
        m.compress_ratios.assign(40, 0);
        m.engram_layer_ids.clear();
        m.v41DsparkTpDevices = dev;
        for (int l : {2, 8, 14, 20}) {
            m.isKvSource[l] = 1;
            m.compress_ratios[l] = l == 20 ? 1 : 2;
        }
        // Other window dtypes have an independent CPU oracle in deepseekV41DsparkCommitRegression.
        int cases = 0;
        for (int pos : {1, 2, 127, 128, 2367, 2368})
            for (int forward = 1; forward <= 8; ++forward) {
                for (int accept = 1; accept <= forward; ++accept) {
                    Fixture a(m, pos, forward), b(m, pos, forward);
                    m.DsparkCommitPrefix(a.state, a.scratch, pos, accept, forward);
                    attachFused(b, m);
                    m.DsparkCommitPrefix(b.state, b.scratch, pos, accept, forward);
                    compare(a, b, m);
                    ++cases;
                }
            }
        std::cout << "Correctness PASS " << cases
                  << " cases, all 8 ranks; full/partial acceptance, wrap, BF16 windows and FP32 tails"
                  << std::endl;
        smallWindowCommit();
        req(!FastllmCudaV41CommitVerifiedPrefix(nullptr, 40, 1, 2), "null descriptors accepted");
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
