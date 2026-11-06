#include "fastllm.h"
#include "utils/hybrid_moe.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cuda(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
static void Allocate(Data &data, DataType type, const std::vector<int> &dims, int device) {
    data.dataType = type; data.Resize(dims); data.ToDevice(CUDA, {device}, false); data.Allocate(false);
}
static void Run(int source, int rows) {
    const int device = 1-source, hidden = 2560, topk = 10;
    FastllmCudaSetDevice(source);
    Data input, index, score, output;
    Allocate(input, FLOAT32, {rows,hidden}, source);
    Allocate(index, INT32, {rows,topk}, source);
    Allocate(score, FLOAT32, {rows,topk}, source);
    const size_t inputBytes = input.GetBytes(), routeBytes = index.GetBytes();
    std::vector<unsigned char> expected(inputBytes + 2*routeBytes), actual(expected.size());
    HybridMoeRemoteWorkspace workspace;
    auto compute = [&](HybridMoeRemoteWorkspace &remote) {
        // A changing payload tests all three transfers. Temporary storage
        // exercises the eager allocation recorder and captured scratch pins.
        Data temporary;
        Allocate(temporary, FLOAT32, {rows,hidden}, device);
        Allocate(remote.output, FLOAT32, {rows,hidden + 2*topk}, device);
        Require(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
            temporary.cudaData, remote.input.cudaData, inputBytes), "remote scratch copy");
        auto *destination = static_cast<unsigned char *>(remote.output.cudaData);
        Require(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(destination, temporary.cudaData, inputBytes), "remote input copy");
        Require(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(destination + inputBytes, remote.index.cudaData, routeBytes), "remote index copy");
        Require(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(destination + inputBytes + routeBytes, remote.score.cudaData, routeBytes), "remote score copy");
    };
    RunRemoteHybridMoe(workspace,input,index,score,output,device,compute);
    FastllmCudaSyncCurrentThreadStream();
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> pins;
    Require(FastllmCudaGraphPrepareCaptureDevice() && FastllmCudaGraphMemoryPoolBegin() &&
        FastllmCudaGraphBeginCapture(), "remote capture start");
    RunRemoteHybridMoe(workspace,input,index,score,output,device,compute);
    Require(FastllmCudaGraphEndCapture(&graph) && graph &&
        FastllmCudaGraphMemoryPoolEnd(pins) && FastllmCudaGraphInstantiate(graph,&exec), "remote capture finish");
    const void *inputStorage=input.cudaData, *outputStorage=output.cudaData;
    for (int pass=0;pass<64;++pass) {
        for (size_t i=0;i<expected.size();++i) expected[i]=(i*19+(i>>8)*37+pass*13)&255;
        Cuda(cudaMemcpy(input.cudaData,expected.data(),inputBytes,cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(index.cudaData,expected.data()+inputBytes,routeBytes,cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(score.cudaData,expected.data()+inputBytes+routeBytes,routeBytes,cudaMemcpyHostToDevice));
        Require(FastllmCudaGraphLaunch(exec), "remote graph launch");
        FastllmCudaSyncCurrentThreadStream();
        Cuda(cudaMemcpy(actual.data(),output.cudaData,actual.size(),cudaMemcpyDeviceToHost));
        Require(actual==expected,"remote graph changed input/index/score bytes");
        Require(inputStorage==input.cudaData && outputStorage==output.cudaData &&
            input.dataDeviceIds==std::vector<int>{source} && output.dataDeviceIds==std::vector<int>{source},
            "remote graph migrated captured storage");
    }
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(pins);
    std::printf("PASS remote hybrid source=%d owner=%d rows=%d: 64 byte-exact replays\n",source,device,rows);
}
int main() {
    try {
        if (FastllmCudaGetDeviceCount()<2) return 77;
        SetCudaGraph(true);
        for (int source : {0,1,0}) for (int rows : {1,4,8}) Run(source,rows);
        std::puts("ALL_PASS remote hybrid transport"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
