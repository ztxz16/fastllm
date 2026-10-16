#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "Requirement failed at %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#include <cstdio>
#include <vector>
#include <algorithm>
#define CHECK(x) do { auto err=(x); if(err!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(err)); return 1; } } while(0)
int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) { puts("SKIP: CUDA unavailable"); return 77; }
    CHECK(cudaSetDevice(0));
    std::vector<size_t> outer,inner;
    {
        FastllmCudaAllocationRecorder record(&outer);
        auto *a=FastllmCudaMalloc(4096);
        { FastllmCudaAllocationRecorder nested(&inner); auto *b=FastllmCudaMalloc(8192); FastllmCudaFree(b); }
        auto *c=FastllmCudaMalloc(16384);FastllmCudaFree(c);FastllmCudaFree(a);
    }
    REQUIRE((outer==std::vector<size_t>{4096,16384}));
    REQUIRE((inner==std::vector<size_t>{8192}));
    constexpr size_t bytes=32768;
    void *output=nullptr;CHECK(cudaMalloc(&output,bytes));
    std::vector<size_t> requests;
    {
        FastllmCudaAllocationRecorder record(&requests);
        void *scratch=FastllmCudaMalloc(bytes);
        CHECK(cudaMemsetAsync(scratch,0x11,bytes,cudaStreamPerThread));
        CHECK(cudaMemcpyAsync(output,scratch,bytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        FastllmCudaFree(scratch);
    }
    CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    REQUIRE((requests==std::vector<size_t>{bytes}));
    void *occupied=FastllmCudaMalloc(bytes); // Consume the block from eager warmup.
    REQUIRE(FastllmCudaReserveGraphWorkspace(requests));
    REQUIRE(FastllmCudaGraphPrepareCaptureDevice());
    REQUIRE(FastllmCudaGraphMemoryPoolBegin());REQUIRE(FastllmCudaGraphBeginCapture());
    REQUIRE(!FastllmCudaReserveGraphWorkspace(requests));
    void *scratch=FastllmCudaMalloc(bytes);
    CHECK(cudaMemsetAsync(scratch,0x2a,bytes,cudaStreamPerThread));
    CHECK(cudaMemcpyAsync(output,scratch,bytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    FastllmCudaFree(scratch);
    void *graph=nullptr,*exec=nullptr;std::vector<void *> pins;
    REQUIRE(FastllmCudaGraphEndCapture(&graph));REQUIRE(graph);
    REQUIRE(FastllmCudaGraphMemoryPoolEnd(pins));REQUIRE(!pins.empty());
    REQUIRE(FastllmCudaGraphInstantiate(graph,&exec));
    std::vector<unsigned char> host(bytes);
    for(int n=0;n<8;++n) {
        REQUIRE(FastllmCudaGraphLaunch(exec));CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        CHECK(cudaMemcpy(host.data(),output,bytes,cudaMemcpyDeviceToHost));
        REQUIRE(std::all_of(host.begin(),host.end(),[](unsigned char x){return x==0x2a;}));
    }
    FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pins);
    FastllmCudaFree(occupied);CHECK(cudaFree(output));
    REQUIRE(!FastllmCudaGetThreadError());REQUIRE(!FastllmCudaGetGraphError());
    size_t freeBytes=0,totalBytes=0; CHECK(cudaMemGetInfo(&freeBytes,&totalBytes));
    REQUIRE(!FastllmCudaReserveGraphWorkspace({totalBytes+(size_t(1)<<20)}));
    REQUIRE(!FastllmCudaGetThreadError());REQUIRE(!FastllmCudaGetGraphError());
    puts("PASS: graph workspace: nested recording, occupied warmup pool, 8 graph replays, capture guard, capacity fallback");
}
