// Regress per-thread default streams sharing the same CUDA stream handle.
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
namespace mx = strata::kernels::exl3;
static void Check(bool v, const char *message) { if (!v) throw std::runtime_error(message); }
static void Cuda(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    constexpr int k=5120,n=128,rows=8;
    Cuda(cudaSetDevice(0));
    std::vector<uint32_t> packed(k*n*6/32); std::mt19937 rng(791);
    for(auto &v:packed)v=rng();
    std::vector<uint16_t> scale(k+n,0x2800); // FP16 1/32
    uint32_t *tr;uint16_t *sc;
    Cuda(cudaMalloc(&tr,packed.size()*4));Cuda(cudaMemcpy(tr,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));
    Cuda(cudaMalloc(&sc,scale.size()*2));Cuda(cudaMemcpy(sc,scale.data(),scale.size()*2,cudaMemcpyHostToDevice));
    mx::Mat m; m.k=k;m.n=n;m.k2=6;m.trellis=tr;m.suh=sc;m.svh=sc+k;
    struct Slot { float *x,*y; std::vector<float> reference; };
    std::array<Slot,2> slots;
    for(int t=0;t<2;++t) {
        auto &slot=slots[t];std::vector<float> x(k*rows);
        for(auto &v:x)v=(int(rng()%1024)-512)/512.f;
        Cuda(cudaMalloc(&slot.x,x.size()*4));Cuda(cudaMemcpy(slot.x,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        Cuda(cudaMalloc(&slot.y,n*rows*4));
        mx::MvJob j;j.m=&m;j.xf=slot.x;j.y=slot.y;j.ncol=rows;j.xf_cs=k;j.y_cs=n;
        Check(mx::mv(&j,1,cudaStreamPerThread),"reference mv");Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        slot.reference.resize(n*rows);Cuda(cudaMemcpy(slot.reference.data(),slot.y,n*rows*4,cudaMemcpyDeviceToHost));
    }
    std::atomic<int> ready{0};std::array<std::exception_ptr,2> errors;
    auto run=[&](int t) {
        try {
            Cuda(cudaSetDevice(0));auto &slot=slots[t];
            Check(mx::prepare(cudaStreamPerThread),"thread prepare");
            mx::MvJob j;j.m=&m;j.xf=slot.x;j.y=slot.y;j.ncol=rows;j.xf_cs=k;j.y_cs=n;
            Check(mx::mv(&j,1,cudaStreamPerThread),"thread warmup");Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            ready.fetch_add(1);while(ready.load()<2)std::this_thread::yield();
            cudaGraph_t graph;cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            Check(mx::mv(&j,1,cudaStreamPerThread),"thread capture");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            for(int repeat=0;repeat<64;++repeat)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<float> output(n*rows);Cuda(cudaMemcpy(output.data(),slot.y,n*rows*4,cudaMemcpyDeviceToHost));
            Check(output==slot.reference,"per-thread workspace interference");
            Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
        }catch(...){errors[t]=std::current_exception();ready.store(2);}
    };
    std::thread a(run,0),b(run,1);a.join();b.join();
    for(auto &err:errors)if(err)std::rethrow_exception(err);
    for(auto &slot:slots){Cuda(cudaFree(slot.x));Cuda(cudaFree(slot.y));}
    Cuda(cudaFree(tr));Cuda(cudaFree(sc));
    Check(mx::launch_errors()==0,"Maya launch error");
    std::cout<<"EXL3 MAYA STREAMS PASS: concurrent per-thread streams and graph replay"<<std::endl;
}
