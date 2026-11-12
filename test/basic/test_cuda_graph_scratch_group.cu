#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
__global__ void Transform(const int *input, int *scratch, int bias) {
    int i=threadIdx.x; scratch[i]=3*input[i]+bias;
}
struct Graph {
    void *graph=nullptr,*exec=nullptr,*scratch=nullptr;
    std::vector<void*> pins;
    void Reset() {
        if (exec) FastllmCudaGraphExecDestroy(exec);
        if (graph) FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(pins);
        exec=graph=nullptr; pins.clear();
    }
    ~Graph() { Reset(); }
};
static void Run(size_t bytes) {
    int group=0,otherGroup=0;
    int *input,*output;
    Check(cudaMalloc(&input,256*sizeof(int))); Check(cudaMalloc(&output,256*sizeof(int)));
    Graph a,b,other,persistent;
    auto capture=[&](Graph &g,const void *owner,int bias,bool retain=false) {
        FastllmCudaGraphScratchGroup scope(owner);
        Require(FastllmCudaGraphPrepareCaptureDevice(),"prepare");
        Check(cudaDeviceSynchronize());
        Require(FastllmCudaReserveGraphWorkspace({bytes}),"reserve");
        Require(FastllmCudaGraphMemoryPoolBegin(),"pool begin");
        Require(FastllmCudaGraphBeginCapture(),"capture begin");
        auto *scratch=static_cast<int*>(FastllmCudaMalloc(bytes));g.scratch=scratch;
        Require(scratch,"scratch allocation");
        Transform<<<1,256,0,cudaStreamPerThread>>>(input,scratch,bias);
        Check(cudaMemcpyAsync(output,scratch,256*sizeof(int),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        if (!retain) FastllmCudaFree(scratch);
        Require(FastllmCudaGraphEndCapture(&g.graph),"capture end");
        Require(FastllmCudaGraphMemoryPoolEnd(g.pins),"pool end");
        Require(FastllmCudaGraphInstantiate(g.graph,&g.exec),"instantiate");
    };
    capture(a,&group,11); capture(b,&group,23);
    Require(a.scratch==b.scratch,"serial graphs should share scratch");
    capture(other,&otherGroup,41);
    Require(other.scratch!=a.scratch,"different groups must be isolated");
    // A live allocation pinned by a graph remains private even after Data releases it.
    capture(persistent,nullptr,61,true);
    FastllmCudaFree(persistent.scratch);
    {
        FastllmCudaGraphScratchGroup scope(&group);
        void *eager=FastllmCudaMalloc(bytes);
        Require(eager!=a.scratch && eager!=other.scratch && eager!=persistent.scratch,
                "eager allocation borrowed graph scratch");
        FastllmCudaFree(eager);
    }
    std::vector<int> in(256),out(256);
    for(int round=0;round<32;++round) {
        for(int i=0;i<256;++i)in[i]=i-round;
        Check(cudaMemcpy(input,in.data(),256*sizeof(int),cudaMemcpyHostToDevice));
        for(auto item:{std::pair<Graph*,int>{&a,11},{&b,23},{&other,41},{&persistent,61}}) {
            Require(FastllmCudaGraphLaunch(item.first->exec),"replay");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaMemcpy(out.data(),output,256*sizeof(int),cudaMemcpyDeviceToHost));
            for(int i=0;i<256;++i)Require(out[i]==3*in[i]+item.second,"shared graph output mismatch");
        }
    }
    a.Reset();
    void *eager=FastllmCudaMalloc(bytes);
    Require(eager!=b.scratch,"first graph destruction released another graph's pins");
    FastllmCudaFree(eager);
    Require(FastllmCudaGraphLaunch(b.exec),"surviving graph replay");
    Check(cudaDeviceSynchronize());
    b.Reset();other.Reset();persistent.Reset();
    Check(cudaFree(input));Check(cudaFree(output));
}
extern "C" int main(int,char**) {
    try {
        Run(4096);Run(12*1024*1024);
        std::puts("PASS shared scratch replay, group/eager isolation, persistent storage and pin lifetime");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
