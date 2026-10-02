#include "fastllm.h"
#include "models/naive_n05_flash.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace fastllm;
static void Require(bool v, const char *s) { if (!v) throw std::runtime_error(s); }
static int checks = 0;
static bool quick = false;
static void Upload(Data &d, const std::vector<int> &shape, int seed) {
    d.Resize(shape); d.Allocate();
    unsigned x = seed;
    for (uint64_t i = 0; i < d.Count(0); ++i) {
        x = 1664525u * x + 1013904223u;
        float v = ((int)(x >> 16) - 32768) / 32768.0f;
        if (d.dataType == BFLOAT16) ((uint16_t *)d.cpuData)[i] = Float32ToBFloat16RNEBits(v);
        else ((float *)d.cpuData)[i] = v;
    }
    d.ToDevice(DataDevice::CUDA, {0}, true);
}
template<class T> static std::vector<T> Read(const Data &d) {
    size_t count=1;for(int dim:d.dims)count*=dim;
    std::vector<T> v(count);
    Require(cudaMemcpy(v.data(), d.cudaData, count*sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "read failed");
    return v;
}
static void TestTopK() {
    for (int n : {1, 127, 2047, 2048, 2049, 7587, 8192, 8193, 32769, 131073})
    for (int mode = 0; mode < 3; ++mode) for (int fraction : {1, 2}) {
        if (quick && n != 127 && n != 7587) continue;
        Data scores(FLOAT32), result;
        scores.Resize({1, n}); scores.Allocate();
        auto *p = (float *)scores.cpuData;
        unsigned state = n;
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            p[i] = mode == 0 ? (i & 1 ? -0.0f : 0.0f) : mode == 1
                ? float(int(state % 31) - 15) : float(int(state >> 8) - (1 << 23)) / 128;
        }
        if (mode == 2 && n > 5) {
            p[1] = p[2] = std::numeric_limits<float>::infinity();
            p[3] = -std::numeric_limits<float>::infinity();
        }
        int valid = std::max(1, n / fraction), top = n < 2048 ? n + 7 : 2048;
        std::vector<int> order(valid); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        scores.ToDevice(DataDevice::CUDA, {0}, true);
        FastllmCudaNaiveTopK(scores, valid - 1, top, result);
        auto actual = Read<int>(result);
        for (int i = 0; i < top; ++i) {
            int expected=i < valid ? order[i] : -1;
            if(actual[i]!=expected) std::fprintf(stderr,"TopK n=%d mode=%d valid=%d i=%d got=%d expected=%d\n",n,mode,valid,i,actual[i],expected);
            Require(actual[i] == expected, "GPU TopK differs from stable CPU order");
        }
        ++checks;
    }
}
static std::vector<int> BatchedReference(const std::vector<float> &scores, int rows,
                                        int keys, int past, int top) {
    std::vector<int> result((size_t)rows * top, -1), order(keys);
    for (int row = 0; row < rows; ++row) {
        int valid = past + row + 1, keep = std::min(valid, top);
        std::iota(order.begin(), order.begin() + valid, 0);
        const float *p = scores.data() + (size_t)row * keys;
        std::partial_sort(order.begin(), order.begin() + keep, order.begin() + valid,
            [p](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        std::copy_n(order.data(), keep, result.data() + (size_t)row * top);
    }
    return result;
}
static std::vector<float> BatchedScores(int rows, int keys, int past, int mode, unsigned seed) {
    std::vector<float> values((size_t)rows * keys);
    for (int row = 0; row < rows; ++row) for (int col = 0; col < keys; ++col) {
        seed = seed * 1664525u + 1013904223u;
        float v = mode == 0 ? (col & 1 ? -0.0f : 0.0f) : mode == 1
            ? float(int(seed % 19) - 9) : float(int(seed >> 8) - (1 << 23)) / 8192.0f;
        if (mode == 1 && col < 3) v = col == 2 ? -std::numeric_limits<float>::infinity()
                                                            : std::numeric_limits<float>::infinity();
        // Future positions must never win, even with +infinity or NaN there.
        if (col > past + row) v = col & 1 ? std::numeric_limits<float>::infinity()
                                         : std::numeric_limits<float>::quiet_NaN();
        values[(size_t)row * keys + col] = v;
    }
    return values;
}
static void TestBatchedTopK() {
    Data result;
    struct Shape { int rows, keys, past, top; };
    const Shape shapes[] = {{2,2,0,7}, {3,127,0,129}, {33,2049,17,2048},
        {31,8192,8000,1}, {32,8192,8160,17}, {129,8193,8064,2048},
        {512,32768,32256,2048}, {512,65536,65024,2048}};
    for (const auto &s : shapes) for (int mode = 0; mode < 3; ++mode) {
        if (quick && s.rows > 33) continue;
        auto values = BatchedScores(s.rows, s.keys, s.past, mode, 713 + mode);
        auto expected = BatchedReference(values, s.rows, s.keys, s.past, s.top);
        Data scores(FLOAT32); scores.Resize({s.rows,s.keys}); scores.Allocate();
        std::memcpy(scores.cpuData, values.data(), values.size()*sizeof(float));
        scores.ToDevice(DataDevice::CUDA,{0},true);
        FastllmCudaNaiveTopK(scores,s.past,s.top,result);
        Require(result.dataDevice==DataDevice::CUDA && result.dims==std::vector<int>({s.rows,s.top}),
                "batched TopK changed output shape/device");
        auto actual=Read<int>(result);
        if (actual!=expected) {
            for(size_t i=0;i<actual.size();++i) if(actual[i]!=expected[i]) {
                std::fprintf(stderr,"BatchedTopK rows=%d keys=%d mode=%d i=%zu got=%d expected=%d\n",
                    s.rows,s.keys,mode,i,actual[i],expected[i]);break;
            }
        }
        Require(actual==expected,"Batched TopK differs from independent CPU comparator");
        ++checks;
    }
    // Change score contents between graph replays to catch stale host selection.
    for (const auto &s : {Shape{3,2049,1000,2048}, Shape{33,8192,8159,2048}}) {
        Data scores(FLOAT32), output; auto values=BatchedScores(s.rows,s.keys,s.past,2,19);
        scores.Resize({s.rows,s.keys});scores.Allocate();
        std::memcpy(scores.cpuData,values.data(),values.size()*sizeof(float));scores.ToDevice(DataDevice::CUDA,{0},true);
        FastllmCudaNaiveTopK(scores,s.past,s.top,output);
        Require(cudaDeviceSynchronize()==cudaSuccess,"TopK warmup failed");
        cudaGraph_t graph;cudaGraphExec_t exec;
        Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");
        FastllmCudaNaiveTopK(scores,s.past,s.top,output);
        Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
        Require(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess,"graph instantiate");
        for(int seed=41;seed<44;++seed) {
            values=BatchedScores(s.rows,s.keys,s.past,seed%3,seed);
            auto expected=BatchedReference(values,s.rows,s.keys,s.past,s.top);
            Require(cudaMemcpyAsync(scores.cudaData,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread)==cudaSuccess,"graph upload");
            Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"graph launch");
            Require(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"graph sync");
            Require(Read<int>(output)==expected,"graph TopK differs after score update");++checks;
        }
        cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
    }
}

struct CacheOps : NaiveN05FlashModel { using NaiveN05FlashModel::AppendCache; };
static void TestCache() {
    for (auto dims : {std::pair<int,int>{1536,1024}, {8,16}, {200,56}}) {
        Data key(BFLOAT16), value(BFLOAT16);
        std::vector<uint16_t> refK, refV;
        for (int step = 0; step < (quick ? 4 : 80); ++step) {
            int n = step == 0 ? 512 : step % 23 == 0 ? 256 : step % 17 == 0 ? 7 : 1;
            Data k(BFLOAT16), v(BFLOAT16);
            Upload(k, {1,n,dims.first}, 13 + step); Upload(v, {1,n,dims.second}, 137 + step);
            auto kr = Read<uint16_t>(k), vr = Read<uint16_t>(v);
            refK.insert(refK.end(), kr.begin(), kr.end()); refV.insert(refV.end(), vr.begin(), vr.end());
            if (refK.size() > 127u * dims.first) refK.erase(refK.begin(), refK.end() - 127 * dims.first);
            if (refV.size() > 127u * dims.second) refV.erase(refV.begin(), refV.end() - 127 * dims.second);
            void *oldK = key.cudaData, *oldV = value.cudaData;
            CacheOps::AppendCache(key,k); CacheOps::AppendCache(value,v);
            if (step && n == 1) Require(key.cudaData == oldK && value.cudaData == oldV, "single-token append reallocated retained capacity");
            auto kd = key.expansionDims, vd = value.expansionDims;
            auto kp = key.cudaData, vp = value.cudaData;
            FastllmCudaNaiveTrimCache(key,value,127);
            Require(key.cudaData == kp && value.cudaData == vp && key.expansionDims == kd && value.expansionDims == vd,
                    "trim changed pointer or capacity");
            Require(key.isKVCache && value.isKVCache && Read<uint16_t>(key) == refK && Read<uint16_t>(value) == refV,
                    "sliding suffix content mismatch");
            ++checks;
        }
    }
}
int main(int argc,char **argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        Require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--quick") == 0),
                "usage: naive_n05_decode_test [--quick]");
        quick = argc == 2;
        SetThreads(4);
        TestTopK(); TestBatchedTopK(); TestCache();
        Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA final synchronization failed");
        std::printf("Naive decode regression passed: %d cases\n",checks);
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
    return 0;
}
