// Exercise the public single-device/PP dispatch, including loader ownership,
// grouped prefill, changing GPU routes, shrinking batches and Graph replay.
#include "fastllm.h"
#include "utils.h"
#include "models/deepseekv41.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static int checks = 0;
static void Check(bool ok, const char *msg) {
    if (!ok) throw std::runtime_error(msg);
    ++checks;
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static std::vector<float> Read(const Data &d) {
    Check(d.dataType == FLOAT32, "MoE sum was rounded before shared expert addition");
    std::vector<float> v(d.Count(0));
    Cuda(cudaMemcpy(v.data(), d.cudaData, v.size()*sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}
static void Compare(const std::vector<float> &a, const std::vector<float> &b, int rows) {
    double error=0, norm=0, maximum=0, delta=0;
    Check(a.size()==b.size(), "output shape mismatch");
    for (size_t i=0; i<a.size(); ++i) {
        Check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite MoE output");
        error += (a[i]-b[i])*(a[i]-b[i]);
        norm += b[i]*b[i];
        maximum=std::max(maximum, double(std::abs(b[i])));
        delta=std::max(delta, double(std::abs(a[i]-b[i])));
    }
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    std::printf("rows=%d relative_l2=%.8g max_relative_to_peak=%.8g\n",
                rows, relative, delta/std::max(maximum,1e-15));
    // Different grouped Tensor Core accumulation can cross an FP8 rounding
    // threshold. Independent precision-boundary fixtures also run separately.
    Check(relative <= .001 && delta <= .005*maximum+1e-6, "Marlin/reference mismatch");
}

struct Fixture {
    static constexpr int experts=8, topk=6;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights, reference;
    Fixture(int hidden, int inter, int device, bool useSlab=false) :
        weights(2*(experts+1),nullptr), reference(weights) {
        DeepSeekV41Model loader;
        loader.block_cnt=1;
        loader.moeDeviceMap={{"cuda:"+std::to_string(device),1}};
        std::mt19937 rng(628);
        for (int expert=0; expert<experts; ++expert) for (int part=0; part<2; ++part) {
            const int n=part ? hidden : 2*inter, k=part ? inter : hidden;
            const std::string name="layers.0.ffn.experts."+std::to_string(expert)+
                                  (part ? ".w2.weight" : ".gateup.weight");
            auto w=std::make_unique<Data>(NVFP4,std::vector<int>{n,k});
            w->blockK=1; w->blockM=32; w->Allocate(false);
            for (size_t i=0; i<GetNVFP4WeightBytes(n,k); ++i) w->cpuData[i]=rng()>>24;
            auto *s=GetNVFP4ScaleData(*w);
            for (size_t i=0; i<GetNVFP4ScaleBytes(n,k,1,32); ++i) s[i]=119+(rng()%4);
            auto ref=std::make_unique<Data>(NVFP4,std::vector<int>{n,k});
            ref->blockK=1; ref->blockM=32; ref->Allocate(false);
            std::memcpy(ref->cpuData,w->cpuData,w->GetBytes());
            ref->isModelWeight=true; ref->directMemory=true;
            ref->ToDevice(CUDA,{device},true);
            w->name=name; w->isModelWeight=true;
            if (useSlab) {
                w->ToDevice(CUDA,{device},true);
            } else {
                loader.AddSpecialWeight(name,"linear",0);
                Check(loader.MoveSpecialWeightToCudaIfNeeded(name,*w),"loader did not place PP weight");
                Check(w->directMemory,"PP loader retained source in weight slab");
            }
            weights[2*(expert+1)+part]=w.get();
            reference[2*(expert+1)+part]=ref.get();
            owned.push_back(std::move(w)); owned.push_back(std::move(ref));
        }
    }
};

static void Case(int hidden, int inter, int device, const std::vector<int> &sizes, bool useSlab=false) {
    FastllmCudaSetDevice(device);
    Fixture f(hidden,inter,device,useSlab);
    Data output, ref, a, b, c;
    std::mt19937 rng(89);
    std::normal_distribution<float> normal(0,1);
    for (int rows:sizes) {
        std::vector<float> x(size_t(rows)*hidden), scales(rows*Fixture::topk);
        for (size_t i=0; i<x.size(); ++i) x[i]=normal(rng)*std::ldexp(1.f,int(i/32%5)-2);
        Data input(FLOAT32,{rows,hidden},x);
        ToDataType(input,BFLOAT16);
        Data ids(INT32,{rows,Fixture::topk}); ids.Allocate();
        for (int r=0;r<rows;++r) for (int j=0;j<Fixture::topk;++j) {
            // Expert ordering differs from rank order; rows share experts.
            ((int32_t *)ids.cpuData)[r*Fixture::topk+j]=(7-j+r)%Fixture::experts;
            scales[r*Fixture::topk+j]=j==0 ? 0.f : j==1 ? -.17f : .31f*j;
        }
        Data scores(FLOAT32,{rows,Fixture::topk},scales);
        for (auto *d:{&input,&ids,&scores}) d->ToDevice(CUDA,{device},true);
        DoCudaMergeMOEV41Fallback(input,ref,ids,scores,f.reference.data(),
                                  f.reference.size(),0.f,10.f);
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
        DoCudaMergeMOE(input,output,ids,scores,a,b,c,f.weights.data(),nullptr,
                      0.f,MoeGateSwiglu,f.weights.size(),10.f,true);
        Check(FastllmCudaMergeMOEUsedGraphUnsafeFallback()==useSlab,
              "incorrect optimized/fallback selection");
        Check((f.weights[2]->cudaData==nullptr)==!useSlab,"repack ownership mismatch");
        Compare(Read(output),Read(ref),rows);
        if (rows==1 && !useSlab) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            DoCudaMergeMOE(input,output,ids,scores,a,b,c,f.weights.data(),nullptr,
                          0.f,MoeGateSwiglu,f.weights.size(),10.f,true);
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            std::vector<int32_t> changed(Fixture::topk);
            for(int j=0;j<Fixture::topk;++j) changed[j]=(j+3)%Fixture::experts;
            Cuda(cudaMemcpy(ids.cudaData,changed.data(),changed.size()*4,cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            const auto replay=Read(output);
            DoCudaMergeMOE(input,output,ids,scores,a,b,c,f.weights.data(),nullptr,
                          0.f,MoeGateSwiglu,f.weights.size(),10.f,true);
            Check(replay==Read(output),"graph replay ignored changed GPU routes");
            Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        }
    }
}

int main(int argc,char **argv) {
    try {
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
        cudaDeviceProp properties;
        Cuda(cudaGetDeviceProperties(&properties,0));
        if(properties.major<8) return 77; // In-tree Marlin requires SM80+.
        SetThreads(4);
        Case(256,256,0,{1,2,8,16,33,128,256,257,511,512,513,2049,1});
        Case(256,256,0,{1,8},true); // Unsupported allocator keeps the old path.
        if(devices>1) Case(256,256,1,{1,33,512,1}); // PP device changes/cache reuse.
        if(argc>1 && std::string(argv[1])=="--full")
            Case(5120,2304,0,{1,8,512,1});
        std::printf("V4.1 MXFP4 PP regression PASS: %d checks\n",checks);
    } catch(const std::exception &e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
