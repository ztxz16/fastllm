#include "fastllm.h"
#include "utils.h"
#include "utils/kernel_registry.h"
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_bf16.h>
using namespace fastllm;
static void Check(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
static void Cuda(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void Allocate(Data &d) { d.ToDevice(CUDA, {0}, false); d.Allocate(false); }
static void Fill(Data &d, int seed, float scale=1.f) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> dist(-1.f,1.f); std::vector<float> x(d.Count(0)); for(auto &v:x)v=scale*dist(rng);
    if(d.dataType==FLOAT32) Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));
    else if(d.dataType==BFLOAT16) {std::vector<__nv_bfloat16> h(x.size());for(size_t i=0;i<x.size();++i)h[i]=__float2bfloat16(x[i]);Cuda(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
    else {std::vector<half> h(x.size());for(size_t i=0;i<x.size();++i)h[i]=__float2half(x[i]);Cuda(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
}
static double L2(Data &a,Data &b) {
    std::vector<float>x(a.Count(0)),y(x.size());
    auto read=[](Data &d,std::vector<float>&v) {
        if(d.dataType==FLOAT32)Cuda(cudaMemcpy(v.data(),d.cudaData,d.GetBytes(),cudaMemcpyDeviceToHost));
        else {std::vector<uint16_t> h(v.size());Cuda(cudaMemcpy(h.data(),d.cudaData,d.GetBytes(),cudaMemcpyDeviceToHost));
            for(size_t i=0;i<v.size();++i) {
                if(d.dataType==FLOAT16){half z;std::memcpy(&z,&h[i],2);v[i]=__half2float(z);}
                else {__nv_bfloat16 z;std::memcpy(&z,&h[i],2);v[i]=__bfloat162float(z);}
            }
        }
    };read(a,x);read(b,y);
    double e=0,s=0; for(size_t i=0;i<x.size();++i){Check(std::isfinite(x[i])&&std::isfinite(y[i]),"nonfinite");e+=double(x[i]-y[i])*(x[i]-y[i]);s+=double(y[i])*y[i];}return sqrt(e/std::max(s,1e-20));
}
static void Case(int rows,int groups,int width,int rank,bool combine,int tile,bool timing,int columns=8,DataType type=FLOAT16,bool injectGate=true,bool planar=false,DataType mixedType=FLOAT32,bool selected=false) {
    if(selected) {
        std::string error;
        Check(kernels::Configure(R"({"mode":"quick","families":["qwen4.hyper_read"],"force":{"qwen4.hyper_read":"cuda.hc.planar.cols8"},"budget_ms":30000})",error),error.c_str());
    }
    const int channels=groups*width;
    Data input(FLOAT32,{1,rows,channels}),block(type,{1,rows,width}),old(type,{1,rows,groups}),norm(FLOAT32,{channels}),down(FLOAT16,{rank,channels}),inject(FLOAT16,{groups,channels});
    Data up(FLOAT16,{channels,rank}),mixed(mixedType,{1,rows,width}),refMixed(FLOAT32,{1,rows,width});
    Data residual(FLOAT32,input.dims),normalized(FLOAT32,input.dims),low(FLOAT32,{1,rows,rank}),gate(FLOAT32,{1,rows,groups});
    Data refResidual(FLOAT32,input.dims),refNorm(FLOAT32,input.dims),refLow(FLOAT32,low.dims),refGate(FLOAT32,gate.dims);
    for(Data *d:{&input,&block,&old,&norm,&down,&inject,&residual,&normalized,&low,&gate,&refResidual,&refNorm,&refLow,&refGate,&up,&mixed,&refMixed})Allocate(*d);
    Fill(up,23,.02f);Fill(norm,5);Fill(down,7,.02f);Fill(inject,9,.02f);
    std::function<void()> selectedReference;
    auto run=[&] {
        if(selected)Check(FastllmCudaQwen4SelectHyperRead(input,combine?&block:nullptr,
            combine?&old:nullptr,norm,down,up,injectGate?&inject:nullptr,
            combine?&residual:nullptr,mixed,injectGate?&gate:nullptr,groups,1e-6f,false,
            selectedReference),"selector rejected");
        else Check(FastllmCudaQwen4HyperRead(input,combine?&block:nullptr,combine?&old:nullptr,
            norm,down,up,injectGate?&inject:nullptr,combine?&residual:nullptr,mixed,
            injectGate?&gate:nullptr,groups,1e-6f,tile,columns,planar),"fused rejected");
    };
    auto reference=[&]{
        if(combine)Check(FastllmCudaQwen4HyperCombineRMSNorm(input,block,old,norm,refResidual,refNorm,1e-6f,groups,nullptr),"ref norm");
        else Check(FastllmCudaQwen4GroupedRMSNorm(input,norm,refNorm,1e-6f,groups),"ref norm");
        if(injectGate)Check(FastllmCudaQwen4HyperProject(refNorm,down,inject,refLow,refGate,groups),"ref project");
        else {Linear(refNorm,down,Data(),refLow);Mul(refLow,1.f/groups,refLow);Silu(refLow,refLow);}
        if(!FastllmCudaQwen4HyperMixProjected(refNorm,refLow,up,refMixed,groups)){Data logits;Linear(refLow,up,Data(),logits);Qwen4HyperMix(refNorm,logits,groups,refMixed);}
    };
    selectedReference=[&] {
        reference();ToDataType(refMixed,mixed,mixedType);
        if(combine)Cuda(cudaMemcpyAsync(residual.cudaData,refResidual.cudaData,residual.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        if(injectGate)Cuda(cudaMemcpyAsync(gate.cudaData,refGate.cudaData,gate.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    };
    Fill(input,0);Fill(block,1);Fill(old,2);reference();
    std::vector<size_t> requests;{FastllmCudaAllocationRecorder recorder(&requests);reference();run();}Cuda(cudaDeviceSynchronize());
    Check(FastllmCudaGraphPrepareCaptureDevice(),"prepare");Check(FastllmCudaReserveGraphWorkspace(requests),"reserve");
    Check(FastllmCudaGraphMemoryPoolBegin(),"pool");
    cudaGraph_t graph;cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run();Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::vector<void*> owned;Check(FastllmCudaGraphMemoryPoolEnd(owned),"pool end");
    double maxL2=0;
    for(int seed=1;seed<=3;++seed){Fill(input,seed*11);Fill(block,seed*13);Fill(old,seed*17);reference();Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaDeviceSynchronize());
        maxL2=std::max(maxL2,L2(mixed,refMixed));if(injectGate)maxL2=std::max(maxL2,L2(gate,refGate));
        if(combine)Check(L2(residual,refResidual)==0,"residual changed");
    }
    printf("check rows=%d tile=%d output_type=%d l2=%.9g\n",rows,tile,int(mixedType),maxL2); Check(maxL2<.0032,"l2");
    Check(!FastllmCudaQwen4HyperRead(input,nullptr,nullptr,norm,down,up,&inject,nullptr,input,&gate,groups,1e-6f,tile,columns,planar),"alias accepted");
    float fused=0,ref=0;
    if(timing){cudaEvent_t a,b;Cuda(cudaEventCreate(&a));Cuda(cudaEventCreate(&b));
        Cuda(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<300;++i)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaEventRecord(b,cudaStreamPerThread));Cuda(cudaEventSynchronize(b));Cuda(cudaEventElapsedTime(&fused,a,b));
        Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));FastllmCudaGraphMemoryPoolRelease(owned);owned.clear();Check(FastllmCudaGraphMemoryPoolBegin(),"ref pool");
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));reference();Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));Check(FastllmCudaGraphMemoryPoolEnd(owned),"ref pool end");
        Cuda(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<300;++i)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaEventRecord(b,cudaStreamPerThread));Cuda(cudaEventSynchronize(b));Cuda(cudaEventElapsedTime(&ref,a,b));Cuda(cudaEventDestroy(a));Cuda(cudaEventDestroy(b));}
    printf("columns=%d ",columns); printf("HC rows=%d groups=%d width=%d rank=%d combine=%d tile=%d l2=%.7g fused_us=%.4f ref_us=%.4f\n",rows,groups,width,rank,combine,tile,maxL2,fused*1000/300,ref*1000/300);fflush(stdout);
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));FastllmCudaGraphMemoryPoolRelease(owned);
}
int main(int,char**){try {FastllmCudaSetDevice(0);
    for(bool planar:{false,true}) {
    for(int rows:{1,2,3,4,7,8,9,17,33,65})for(int cols:{4,8})
        Case(rows,4,2560,320,true,4,false,cols,FLOAT16,true,planar);
    for(DataType type:{FLOAT32,FLOAT16,BFLOAT16})for(bool apply:{false,true})for(bool gate:{false,true})
        Case(5,4,2560,320,apply,4,false,8,type,gate,planar);
    }
    for(DataType out:{FLOAT16,BFLOAT16})for(int rows:{1,2,4,17})for(bool apply:{false,true})
        Case(rows,4,2560,320,apply,4,false,8,FLOAT16,true,true,out);
    for(DataType out:{FLOAT32,FLOAT16,BFLOAT16})for(int rows:{1,4,17})
        Case(rows,4,2560,320,true,4,false,8,FLOAT16,true,true,out,true);
    puts("PASS HC random inputs, types, arbitrary rows, tails, final read, changed-input graphs and exact residual");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
