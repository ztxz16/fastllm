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
    const int experts, topk;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights, reference;
    Fixture(int hidden, int inter, int device, bool useSlab=false,
            int experts=8, int topk=6) : experts(experts), topk(topk),
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

static void Case(int hidden, int inter, int device, const std::vector<int> &sizes, bool useSlab=false,
                 int experts=8, int topk=6, bool roundOutput=false) {
    FastllmCudaSetDevice(device);
    Fixture f(hidden,inter,device,useSlab,experts,topk);
    Data output, ref, a, b, c;
    std::mt19937 rng(89);
    std::normal_distribution<float> normal(0,1);
    for (int rows:sizes) {
        std::vector<float> x(size_t(rows)*hidden), scales(rows*f.topk);
        for (size_t i=0; i<x.size(); ++i) x[i]=normal(rng)*std::ldexp(1.f,int(i/32%5)-2);
        Data input(FLOAT32,{rows,hidden},x);
        ToDataType(input,BFLOAT16);
        Data ids(INT32,{rows,f.topk}); ids.Allocate();
        for (int r=0;r<rows;++r) for (int j=0;j<f.topk;++j) {
            // Expert ordering differs from rank order; rows share experts.
            ((int32_t *)ids.cpuData)[r*f.topk+j]=(f.experts-1-j+r)%f.experts;
            scales[r*f.topk+j]=j==0 ? 0.f : j==1 ? -.17f : .31f*j;
        }
        Data scores(FLOAT32,{rows,f.topk},scales);
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
        if (rows<=8 && !useSlab) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            DoCudaMergeMOE(input,output,ids,scores,a,b,c,f.weights.data(),nullptr,
                          0.f,MoeGateSwiglu,f.weights.size(),10.f,true);
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            std::vector<int32_t> changed(rows*f.topk);
            for(int r=0;r<rows;++r) for(int j=0;j<f.topk;++j)
                changed[r*f.topk+j]=(j+3+r)%f.experts;
            Cuda(cudaMemcpy(ids.cudaData,changed.data(),changed.size()*4,cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            const auto replay=Read(output);
            DoCudaMergeMOE(input,output,ids,scores,a,b,c,f.weights.data(),nullptr,
                          0.f,MoeGateSwiglu,f.weights.size(),10.f,true);
            Check(replay==Read(output),"graph replay ignored changed GPU routes");
            Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        }
        if (roundOutput) ToDataType(output, BFLOAT16);
    }
}

// EP compaction must retain original route ids, including empty ranks and
// experts filling all eight tile rows. Check both initial execution and
// replay after the GPU routing/input tensors change against the native W4A8
// reference, not only against another Marlin launch.
static void ExpertParallelCase() {
    FastllmCudaSetDevice(0);
    Fixture f(5120,2304,0,false,12,6);
    Data scratch,output;
    for (int owners : {1,2,8}) for (int rows : {1,2,3,4,5,6,7,8})
    for (int pattern : {0,1,2,3}) {
        const int owner=owners-1;
        std::mt19937 rng(919+owners*131+rows*17+pattern);
        std::vector<float> values(rows*5120),scores(rows*6);
        std::vector<int32_t> indices(rows*6);
        for (float &v:values) v=float(int(rng()%65521)-32760)/16384.f;
        for (int i=0;i<rows*6;++i) {
            int expert=pattern==3 ? i%6 : (i*13+i/6*17)%12;
            int rank=pattern==0 ? int(rng()%owners) :
                     pattern==2 ? (owner+1)%owners : owner;
            indices[i]=expert*owners+rank;
            scores[i]=i%7==0 ? 0.f : float(int(rng()%31)-3)/32;
        }
        Data input(FLOAT32,{rows,5120},values),score(FLOAT32,{rows,6},scores);
        Data ids(INT32,{rows,6});ids.Allocate();
        std::memcpy(ids.cpuData,indices.data(),indices.size()*sizeof(int32_t));
        ToDataType(input,BFLOAT16);
        for (Data *d:{&input,&score,&ids}) d->ToDevice(CUDA,{0},true);
        auto run=[&] {
            Check(FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
                input,scratch,output,f.weights.data(),f.weights.size(),
                (int32_t*)ids.cudaData,(float*)score.cudaData,6,owner,owners,true,10.f),
                "EP compact Marlin launch failed");
        };
        run();Cuda(cudaDeviceSynchronize());
        cudaGraph_t graph;cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        run();Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for (int replay=0;replay<2;++replay) {
            if (replay) {
                std::rotate(indices.begin(),indices.begin()+6%indices.size(),indices.end());
                for (float &v:values) v=-v;
                Data changed(FLOAT32,{rows,5120},values);ToDataType(changed,BFLOAT16);
                changed.ToDevice(CPU);
                Cuda(cudaMemcpy(input.cudaData,changed.cpuData,changed.GetBytes(),cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(ids.cudaData,indices.data(),indices.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
            }
            Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            auto actual=Read(output);
            run();Check(actual==Read(output),"EP graph/eager result mismatch");
            std::vector<float> localScores=scores;
            Data localIds(INT32,{rows,6});localIds.Allocate();
            for (int i=0;i<rows*6;++i) {
                ((int32_t*)localIds.cpuData)[i]=indices[i]/owners;
                if (indices[i]%owners!=owner) localScores[i]=0.f;
            }
            Data ls(FLOAT32,{rows,6},localScores),reference;
            localIds.ToDevice(CUDA,{0},true);ls.ToDevice(CUDA,{0},true);
            DoCudaMergeMOEV41Fallback(input,reference,localIds,ls,f.reference.data(),
                                     f.reference.size(),0.f,10.f);
            std::printf("EP owners=%d rows=%d pattern=%d replay=%d ",owners,rows,pattern,replay);
            Compare(actual,Read(reference),rows);
        }
        Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
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
        if (argc > 1 && std::string(argv[1]) == "--ep") {
            ExpertParallelCase();
            std::printf("V4.1 MXFP4 EP regression PASS: %d checks\n",checks);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--verify") {
            Case(5120,2304,0,{1,2,3,4,5,6,7,8});
            std::printf("V4.1 MXFP4 verification regression PASS: %d checks\n",checks);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--draft") {
            Case(5120,2304,0,{5,5,1,2,3,4,5,6,5,1},false,128,3,true);
            std::printf("V4.1 MXFP4 draft regression PASS: %d checks\n",checks);
            return 0;
        }
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
