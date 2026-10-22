#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
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
void Req(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
void Cu(cudaError_t e) { Req(e == cudaSuccess, cudaGetErrorString(e)); }
std::vector<unsigned char> Fill(Data &x, DataType type, int rows, const std::vector<int> &devices, int seed) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize({1, rows, 512});
    PrepareMultiCudaReplicatedData(x, devices, false);
    std::vector<unsigned char> value(x.GetBytes());
    for (size_t i = 0; i < value.size(); ++i)
        value[i] = (i * 17 + i / 11 + seed) % 251;
    for (int d : devices) {
        Cu(cudaSetDevice(d));
        auto &l = *x.multiDeviceDatas.at(d);
        l.Allocate();
        Cu(cudaMemcpy(l.cudaData, value.data(), value.size(), cudaMemcpyHostToDevice));
    }
    return value;
}
std::vector<unsigned char> Read(Data &x, int d) {
    Cu(cudaSetDevice(d));
    Cu(cudaDeviceSynchronize());
    std::vector<unsigned char> v(x.Count(0) * x.unitSize);
    Cu(cudaMemcpy(v.data(), x.cudaData, v.size(), cudaMemcpyDeviceToHost));
    return v;
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8)
        return 77;
    try {
        SetThreads(8);
        std::vector<int> dev(8);
        std::iota(dev.begin(), dev.end(), 0);
        const std::string name = "multicuda:0,1,2,3,4,5,6,7";
        SetDeviceMap({{name, 1}});
        Access m;
        m.block_cnt = 1;
        m.deviceMap = {{name, 1}};
        m.window_size = 128;
        m.compress_ratios = {2};
        m.isKvSource = {1};
        m.engram_layer_ids.clear();
        m.v41DsparkTpDevices = dev;
        int cases = 0;
        for (int pos : {1, 2, 127, 128, 8191, 8192})
            for (int forward = 2; forward <= 8; ++forward)
                for (int accept = 1; accept <= forward; ++accept) {
                    DeepSeekV41RequestState state;
                    state.layers.resize(1);
                    auto &cache = state.layers[0];
                    DeepSeekV41SpecScratch scratch;
                    scratch.windowKV.resize(1);
                    scratch.rawKV.resize(1);
                    scratch.rawScore.resize(1);
                    scratch.prevRawTail = {pos % 2};
                    scratch.prevBlocks = {pos / 2};
                    auto ring = Fill(cache.windowKV, BFLOAT16, 128, dev, 0);
                    auto input = Fill(scratch.windowKV[0], BFLOAT16, forward, dev, 3);
                    int total = pos % 2 + forward;
                    auto raw = Fill(scratch.rawKV[0], FLOAT32, total, dev, 7);
                    auto score = Fill(scratch.rawScore[0], FLOAT32, total, dev, 11);
                    Fill(cache.rawTailKV, FLOAT32, 2, dev, 0);
                    Fill(cache.rawTailScore, FLOAT32, 2, dev, 0);
                    Fill(cache.compressedKV, BFLOAT16, (pos + forward) / 2, dev, 0);
                    Fill(cache.indexK, BFLOAT16, (pos + forward) / 2, dev, 0);
                    std::vector<void *> addr, ringAddr, inputAddr;
                    for (int d : dev) {
                        addr.push_back(cache.rawTailKV.multiDeviceDatas.at(d)->cudaData);
                        ringAddr.push_back(cache.windowKV.multiDeviceDatas.at(d)->cudaData);
                        inputAddr.push_back(scratch.windowKV[0].multiDeviceDatas.at(d)->cudaData);
                    }
                    m.DsparkCommitPrefix(state, scratch, pos, accept, forward);
                    Req(state.totalLen == pos + accept && cache.compressedBlocks == (pos + accept) / 2 &&
                            cache.rawTail == (pos + accept) % 2,
                        "commit lengths mismatch");
                    for (int i = 0; i < accept; ++i)
                        std::copy_n(input.data() + i * 1024, 1024, ring.data() + ((pos + i) % 128) * 1024);
                    for (int d : dev) {
                        Req(cache.windowKV.multiDeviceDatas.at(d)->cudaData == ringAddr[d] &&
                            scratch.windowKV[0].multiDeviceDatas.at(d)->cudaData == inputAddr[d],
                            "window commit moved storage");
                        Req(Read(*scratch.windowKV[0].multiDeviceDatas.at(d), d) == input,
                            "window commit modified scratch");
                        for (Data *buffer : {&cache.compressedKV, &cache.indexK})
                            Req(buffer->dims[1] == (pos + accept) / 2 &&
                                buffer->multiDeviceDatas.at(d)->dims == buffer->dims,
                                "compressed replica lengths mismatch");
                        Req(Read(*cache.windowKV.multiDeviceDatas.at(d), d) == ring,
                            "committed ring mismatch");
                        if (cache.rawTail) {
                            Req(cache.rawTailKV.IsTensorParallelReplicated() &&
                                    cache.rawTailScore.IsTensorParallelReplicated(),
                                "commit collapsed replicas");
                            Req(cache.rawTailKV.multiDeviceDatas.at(d)->cudaData == addr[d],
                                "commit moved cache");
                            auto a = Read(*cache.rawTailKV.multiDeviceDatas.at(d), d),
                                 b = Read(*cache.rawTailScore.multiDeviceDatas.at(d), d);
                            int begin = (pos % 2 + accept - 1) * 2048;
                            Req(a.size() == 2048 && std::equal(a.begin(), a.end(), raw.begin() + begin),
                                "raw tail mismatch");
                            Req(b.size() == 2048 && std::equal(b.begin(), b.end(), score.begin() + begin),
                                "score tail mismatch");
                        }
                    }
                    ++cases;
                }
        std::cout << "PASS " << cases
                  << " TP8 DSpark commits, lengths 2..8, every accepted prefix, ring wrap, FP32 tails and "
                     "stable allocations"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
