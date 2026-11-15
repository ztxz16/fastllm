#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include "devices/cuda/fastllm-v41-full-decode.h"
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
    // Hundreds of sequential allocations reuse a bounded physical working set.
    // Reserving their sum used to exhaust memory before capture even though
    // replay needs only the simultaneously-live pair below.
    std::vector<size_t> reused;
    constexpr size_t blockBytes = 2 * 1024 * 1024;
    {
        FastllmCudaAllocationRecorder record(&reused);
        void *live = FastllmCudaMalloc(blockBytes);
        for (int n = 0; n < 128; ++n) {
            void *temporary = FastllmCudaMalloc(blockBytes);
            REQUIRE(temporary && temporary != live);
            FastllmCudaFree(temporary);
        }
        { FastllmCudaAllocationRecorder pause(nullptr);
          void *temporary=FastllmCudaMalloc(4*blockBytes);FastllmCudaFree(temporary); }
        FastllmCudaFree(live);
    }
    REQUIRE(reused.size()==2);
    REQUIRE(reused[0]==blockBytes && reused[1]==blockBytes);
    REQUIRE(FastllmCudaReserveGraphWorkspace(reused));
    // Distinct recordings of the same vector must not retain stale pointer maps.
    reused.clear();
    { FastllmCudaAllocationRecorder record(&reused);
      void *a=FastllmCudaMalloc(blockBytes);FastllmCudaFree(a); }
    REQUIRE(reused.size()==1 && reused[0]==blockBytes);
    // A reservation between two serial graph segments must reuse the current
    // transaction's idle protected scratch instead of allocating it per layer.
    {
        constexpr size_t scratchBytes=17*1024*1024;
        REQUIRE(FastllmCudaReserveGraphWorkspace({scratchBytes}));
        REQUIRE(FastllmCudaGraphMemoryPoolBegin());
        REQUIRE(FastllmCudaGraphBeginCapture());
        void *scratch=FastllmCudaMalloc(scratchBytes);REQUIRE(scratch);
        REQUIRE(cudaMemsetAsync(scratch,0,scratchBytes,cudaStreamPerThread)==cudaSuccess);
        FastllmCudaFree(scratch);void *graph=nullptr;
        REQUIRE(FastllmCudaGraphEndCapture(&graph));
        size_t freeBefore=0,freeAfter=0,total=0;
        REQUIRE(cudaMemGetInfo(&freeBefore,&total)==cudaSuccess);
        REQUIRE(FastllmCudaReserveGraphWorkspace({scratchBytes}));
        REQUIRE(cudaMemGetInfo(&freeAfter,&total)==cudaSuccess);
        REQUIRE(freeBefore==freeAfter);
        REQUIRE(FastllmCudaGraphBeginCapture());
        void *again=FastllmCudaMalloc(scratchBytes);REQUIRE(again==scratch);
        REQUIRE(cudaMemsetAsync(again,1,scratchBytes,cudaStreamPerThread)==cudaSuccess);
        FastllmCudaFree(again);void *second=nullptr;
        REQUIRE(FastllmCudaGraphEndCapture(&second));
        std::vector<void*> pinned;REQUIRE(FastllmCudaGraphMemoryPoolEnd(pinned));
        FastllmCudaGraphDestroy(second);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pinned);
        REQUIRE(FastllmV41VerificationGraphKey(129,false)!=FastllmV41VerificationGraphKey(1,true));
        REQUIRE(FastllmV41VerificationGraphKey(2147483647,true)<0);
    }
    // Graph warmup filters eager-only implementations while preserving real
    // allocation/benchmark behavior. An inner ordinary scope cannot override it.
    REQUIRE(!FastllmCudaGraphPreparationActive());
    {
        FastllmCudaGraphPreparation outer;
        REQUIRE(FastllmCudaGraphPreparationActive());
        { FastllmCudaGraphPreparation inner(false); REQUIRE(FastllmCudaGraphPreparationActive()); }
        REQUIRE(FastllmCudaGraphPreparationActive());
        REQUIRE(!FastllmCudaGraphIsCapturingFast());
        void *scratch=FastllmCudaMalloc(2048);REQUIRE(scratch);FastllmCudaFree(scratch);
    }
    REQUIRE(!FastllmCudaGraphPreparationActive());
    constexpr size_t bytes=32768;
    void *output=nullptr,*input=nullptr;CHECK(cudaMalloc(&output,bytes));CHECK(cudaMalloc(&input,bytes));
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
    CHECK(cudaMemcpyAsync(scratch,input,bytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    CHECK(cudaMemcpyAsync(output,scratch,bytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    FastllmCudaFree(scratch);
    void *graph=nullptr,*exec=nullptr;std::vector<void *> pins;
    REQUIRE(FastllmCudaGraphEndCapture(&graph));REQUIRE(graph);
    REQUIRE(FastllmCudaGraphMemoryPoolEnd(pins));REQUIRE(!pins.empty());
    REQUIRE(FastllmCudaGraphInstantiate(graph,&exec));
    std::vector<unsigned char> host(bytes);
    for(int n=0;n<8;++n) {
        CHECK(cudaMemsetAsync(input,0x2a+n,bytes,cudaStreamPerThread));
        REQUIRE(FastllmCudaGraphLaunch(exec));CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        CHECK(cudaMemcpy(host.data(),output,bytes,cudaMemcpyDeviceToHost));
        REQUIRE(std::all_of(host.begin(),host.end(),[n](unsigned char x){return x==0x2a+n;}));
    }
    FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pins);
    FastllmCudaFree(occupied);CHECK(cudaFree(output));CHECK(cudaFree(input));
    REQUIRE(!FastllmCudaGetThreadError());REQUIRE(!FastllmCudaGetGraphError());
    size_t freeBytes=0,totalBytes=0; CHECK(cudaMemGetInfo(&freeBytes,&totalBytes));
    REQUIRE(!FastllmCudaReserveGraphWorkspace({totalBytes+(size_t(1)<<20)}));
    REQUIRE(!FastllmCudaGetThreadError());REQUIRE(!FastllmCudaGetGraphError());
#ifndef USE_ROCM
    // Exhaust independent Lt leases. Eager may still use its own workspace,
    // so an empty serial lease must reject capture instead of freezing GEMV.
    std::vector<std::shared_ptr<void>> leases(32);
    bool exhausted = false;
    for (auto &lease : leases) {
        FastllmCudaLinearGraphScope scope(lease);
        if (!lease) { exhausted = true; break; }
    }
    REQUIRE(exhausted);
    std::shared_ptr<void> missing;
    FastllmCudaClearThreadError();
    REQUIRE(FastllmCudaGraphBeginCapture());
    { FastllmCudaLinearGraphScope scope(missing); }
    REQUIRE(FastllmCudaGetThreadError());
    graph = nullptr;
    REQUIRE(FastllmCudaGraphEndCapture(&graph));
    FastllmCudaGraphDestroy(graph);
    FastllmCudaClearThreadError();
    leases.clear();
    { FastllmCudaLinearGraphScope scope(missing); REQUIRE(missing); }
    REQUIRE(FastllmCudaGraphBeginCapture());
    { FastllmCudaLinearGraphScope scope(missing); }
    REQUIRE(!FastllmCudaGetThreadError());
    graph = nullptr;
    REQUIRE(FastllmCudaGraphEndCapture(&graph));
    FastllmCudaGraphDestroy(graph);
#endif
    puts("PASS: graph workspace: nested recording, occupied warmup pool, 8 graph replays, capture guard, capacity fallback");
}
