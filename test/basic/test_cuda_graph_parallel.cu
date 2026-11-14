#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
__global__ void Branch(const int *input, int *scratch, int bias) {
    const int i = threadIdx.x;
    scratch[i] = input[i] * 3 + bias;
}
__global__ void Join(const int *a, const int *b, int *output) {
    const int i = threadIdx.x;
    output[i] = a[i] + b[i];
}

static void Nested(bool constrained) {
    constexpr size_t bytes = 8 * 1024 * 1024;
    Require(FastllmCudaGraphPrepareCaptureDevice(), "nested prepare");
    Require(FastllmCudaReserveGraphWorkspace(constrained ? std::vector<size_t>{bytes} :
        std::vector<size_t>{bytes,bytes,bytes}), "nested reserve");
    int *input,*a,*b,*c,*sum,*output;
    for(auto p:{&input,&a,&b,&c,&sum,&output})Check(cudaMalloc(p,256*sizeof(int)));
    std::vector<void*> owned;
    Require(FastllmCudaGraphMemoryPoolBegin(), "nested pool");
    Require(FastllmCudaGraphBeginCapture(), "nested capture");
    const int outer=FastllmCudaGraphBeginIndependentRegion();
    auto branch=[&](int *destination,int bias) {
        auto *temporary=static_cast<int*>(FastllmCudaMalloc(bytes));
        Require(temporary,"nested scratch");
        Branch<<<1,256,0,cudaStreamPerThread>>>(input,temporary,bias);
        Check(cudaMemcpyAsync(destination,temporary,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        FastllmCudaFree(temporary);
    };
    branch(a,11);
    FastllmCudaGraphMarkParallelFirstDone(outer);
    FastllmCudaGraphMarkParallelSecondBegin(outer);
    const int inner=FastllmCudaGraphBeginIndependentRegion();
    branch(b,31);
    FastllmCudaGraphMarkParallelFirstDone(inner);
    FastllmCudaGraphMarkParallelSecondBegin(inner);
    branch(c,47);
    FastllmCudaGraphMarkParallelJoin(inner);
    // Consecutive join markers exercise transitive dependency preservation.
    FastllmCudaGraphMarkParallelJoin(outer);
    Join<<<1,256,0,cudaStreamPerThread>>>(a,b,sum);
    Join<<<1,256,0,cudaStreamPerThread>>>(sum,c,output);
    void *graph=nullptr,*exec=nullptr;
    Require(FastllmCudaGraphEndCapture(&graph),"nested capture end");
    Require(FastllmCudaGraphMemoryPoolEnd(owned),"nested pool end");
    const int optimized=FastllmCudaGraphOptimizeParallelRegions(graph);
    Require(optimized>=0 && (constrained || optimized==2),"nested optimize");
    Require(FastllmCudaGraphInstantiate(graph,&exec),"nested instantiate");
    std::vector<int> in(256),out(256);
    for(int replay=0;replay<32;++replay){
        for(int i=0;i<256;++i)in[i]=i-replay;
        Check(cudaMemcpy(input,in.data(),256*sizeof(int),cudaMemcpyHostToDevice));
        Require(FastllmCudaGraphLaunch(exec),"nested replay");Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(out.data(),output,256*sizeof(int),cudaMemcpyDeviceToHost));
        for(int i=0;i<256;++i)Require(out[i]==9*in[i]+89,"nested dependency lost");
    }
    FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(owned);
    for(auto p:{input,a,b,c,sum,output})Check(cudaFree(p));
}

extern "C" int main(int, char **) {
    try {
        Require(FastllmCudaGraphQwen35MoeSelfTest(), "legacy parallel/serial/resource guards");
        std::vector<size_t> recorded;
        {
            FastllmCudaAllocationRecorder recording(&recorded);
            for (int pass=0;pass<8;++pass) {
                const int region=FastllmCudaGraphBeginIndependentRegion();
                Require(region<0,"warmup region was not recorded");
                void *a=FastllmCudaMalloc(4096); FastllmCudaFree(a);
                FastllmCudaGraphMarkParallelFirstDone(region);
                FastllmCudaGraphMarkParallelSecondBegin(region);
                void *b=FastllmCudaMalloc(4096); FastllmCudaFree(b);
                FastllmCudaGraphMarkParallelJoin(region);
            }
        }
        Require(recorded.size()==2,"warmup must reserve two concurrent lifetimes, not sixteen sequential allocations");
        Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
        Require(FastllmCudaReserveGraphWorkspace({4096,4096}), "reserve scratch");
        int *input, *first, *second, *output;
        for (auto p : {&input,&first,&second,&output}) Check(cudaMalloc(p,256*sizeof(int)));
        std::vector<void*> reserved;
        Require(FastllmCudaGraphMemoryPoolBegin(), "begin pool");
        Require(FastllmCudaGraphBeginCapture(), "begin capture");
        // Two regions in one capture verify unique ids and scratch reuse only
        // after a join. Input values change on each replay.
        std::vector<int> ids;
        for (int pass=0;pass<2;++pass) {
            const int id=FastllmCudaGraphBeginIndependentRegion();
            Require(id < 0 && (ids.empty() || id!=ids.front()), "capture region id"); ids.push_back(id);
            int *a=static_cast<int*>(FastllmCudaMalloc(4096));
            Branch<<<1,256,0,cudaStreamPerThread>>>(pass ? output : input,a,11);
            Check(cudaMemcpyAsync(first,a,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            FastllmCudaFree(a);
            FastllmCudaGraphMarkParallelFirstDone(id);
            FastllmCudaGraphMarkParallelSecondBegin(id);
            int *b=static_cast<int*>(FastllmCudaMalloc(4096));
            Require(a && b && a!=b,"concurrent branches shared temporary storage");
            Branch<<<1,256,0,cudaStreamPerThread>>>(pass ? output : input,b,31);
            Check(cudaMemcpyAsync(second,b,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            FastllmCudaFree(b);
            FastllmCudaGraphMarkParallelJoin(id);
            Join<<<1,256,0,cudaStreamPerThread>>>(first,second,output);
        }
        void *graph=nullptr,*exec=nullptr;
        Require(FastllmCudaGraphEndCapture(&graph), "end capture");
        Require(FastllmCudaGraphMemoryPoolEnd(reserved), "end pool");
        Require(FastllmCudaGraphOptimizeParallelRegions(graph)==2,"independent regions stayed serial");
        Require(FastllmCudaGraphInstantiate(graph,&exec), "instantiate");
        std::vector<int> in(256),out(256);
        for (int replay=0;replay<32;++replay) {
            for (int i=0;i<256;++i) in[i]=i-replay;
            Check(cudaMemcpy(input,in.data(),256*sizeof(int),cudaMemcpyHostToDevice));
            Require(FastllmCudaGraphLaunch(exec), "replay");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaMemcpy(out.data(),output,256*sizeof(int),cudaMemcpyDeviceToHost));
            for(int i=0;i<256;++i) Require(out[i]==36*in[i]+294,"parallel graph output mismatch");
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
        // Deliberately reserve only one large block. A missing parallel
        // scratch allocation must serialize the region, not abort capture.
        reserved.clear(); graph=nullptr; exec=nullptr;
        constexpr size_t large=16*1024*1024;
        Require(FastllmCudaReserveGraphWorkspace({large}),"reserve serial fallback scratch");
        Require(FastllmCudaGraphMemoryPoolBegin(),"begin fallback pool");
        Require(FastllmCudaGraphBeginCapture(),"begin fallback capture");
        const int region=FastllmCudaGraphBeginIndependentRegion();
        int *a=static_cast<int*>(FastllmCudaMalloc(large));
        Branch<<<1,256,0,cudaStreamPerThread>>>(input,a,11);
        Check(cudaMemcpyAsync(first,a,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        FastllmCudaFree(a);
        FastllmCudaGraphMarkParallelFirstDone(region);
        FastllmCudaGraphMarkParallelSecondBegin(region);
        int *b=static_cast<int*>(FastllmCudaMalloc(large));
        Require(a && b && a==b,"expected memory-pressure fallback");
        Branch<<<1,256,0,cudaStreamPerThread>>>(input,b,31);
        Check(cudaMemcpyAsync(second,b,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        FastllmCudaFree(b);
        FastllmCudaGraphMarkParallelJoin(region);
        Join<<<1,256,0,cudaStreamPerThread>>>(first,second,output);
        Require(FastllmCudaGraphEndCapture(&graph),"end fallback capture");
        Require(FastllmCudaGraphMemoryPoolEnd(reserved),"end fallback pool");
        Require(FastllmCudaGraphOptimizeParallelRegions(graph)==0,"shared scratch must stay serial");
        Require(FastllmCudaGraphInstantiate(graph,&exec),"instantiate fallback");
        Require(FastllmCudaGraphLaunch(exec),"replay fallback");
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(out.data(),output,256*sizeof(int),cudaMemcpyDeviceToHost));
        for (int i=0;i<256;++i) Require(out[i]==6*in[i]+42,"serial fallback output mismatch");
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
        for (auto p : {input,first,second,output}) Check(cudaFree(p));
        Nested(true); Nested(false);
        std::puts("PASS independent scratch, repeated/nested regions, dynamic replay and serial resource guards");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
