#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" cudaError_t FastllmCudaCheckedMalloc(void **, size_t, const char *, int);

static void Check(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    try {
        constexpr size_t MiB=1024*1024;
        FastllmCudaSetDevice(0);
        // Pin one logically freed pool block in a live graph. OOM recovery
        // must retain its allocation even while reclaiming unrelated scratch.
        void *pinned=FastllmCudaMalloc(2*MiB);
        Check(pinned!=nullptr,"pin allocation");FastllmCudaFree(pinned);
        Check(FastllmCudaGraphPrepareCaptureDevice()&&FastllmCudaGraphMemoryPoolBegin()&&FastllmCudaGraphBeginCapture(),"capture begin");
        pinned=FastllmCudaMalloc(2*MiB);
        Check(cudaMemsetAsync(pinned,0x35,2*MiB,cudaStreamPerThread)==cudaSuccess,"record");
        FastllmCudaFree(pinned);
        void *graph=nullptr,*exec=nullptr;std::vector<void *> pins;
        Check(FastllmCudaGraphEndCapture(&graph)&&FastllmCudaGraphMemoryPoolEnd(pins)&&FastllmCudaGraphInstantiate(graph,&exec),"capture end");
        void *idle=FastllmCudaMalloc(512*MiB);
        Check(idle!=nullptr,"idle allocation");FastllmCudaFree(idle);
        Check(cudaDeviceSynchronize()==cudaSuccess,"idle completion");
        size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total)==cudaSuccess&&free>1024*MiB,"test memory capacity");
        void *filler=nullptr;Check(cudaMalloc(&filler,free-64*MiB)==cudaSuccess,"pressure allocation");
        void *direct=nullptr;
        // A managed capture transaction may belong to another host thread.
        // Even an idle local stream must leave that transaction's pool alone.
        Check(FastllmCudaGraphMemoryPoolBegin(),"begin capture transaction");
        Check(FastllmCudaCheckedMalloc(&direct,256*MiB,__FILE__,__LINE__)==cudaErrorMemoryAllocation&&direct==nullptr,
              "direct recovery ran during a managed capture transaction");
        cudaGetLastError();FastllmCudaGraphMemoryPoolAbort();
        Check(FastllmCudaCheckedMalloc(&direct,256*MiB,__FILE__,__LINE__)==cudaSuccess&&direct,"direct allocation did not recover idle pool");
        Check(cudaGetLastError()==cudaSuccess,"OOM remained in CUDA last-error slot");
        Check(FastllmCudaGraphLaunch(exec)&&cudaDeviceSynchronize()==cudaSuccess,"pinned graph replay after OOM");
        unsigned char value=0;
        Check(cudaMemcpy(&value,pinned,1,cudaMemcpyDeviceToHost)==cudaSuccess&&value==0x35,"pinned allocation lost");
        cudaFree(direct);cudaFree(filler);
        FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pins);
        FastllmCudaReleaseIdleCachedBuffers(0);
        std::cout<<"DIRECT ALLOCATION RETRY PASS: reclaimed idle memory and retained graph pins\n";
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
