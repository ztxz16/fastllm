#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include "utils/kernel_registry.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <random>
using namespace fastllm;
static void Check(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void Cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Policy(bool parallel){std::string s=std::string("{\"mode\":\"quick\",\"samples\":3,\"repeats\":2,\"budget_ms\":30000,\"force\":{\"qwen4.ple_gate\":\"cuda.ple_gate.")+(parallel?"block256":"serial")+"\"}}";Check(fastllm_kernel_configure(s.c_str())==1,fastllm_kernel_error());}
static void Upload(Data &d,const std::vector<float>&x){
    if(d.dataType==FLOAT32)Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));
    else {std::vector<uint16_t> h(x.size());for(size_t i=0;i<x.size();++i)h[i]=d.dataType==FLOAT16?__half_as_ushort(__float2half(x[i])):__bfloat16_as_ushort(__float2bfloat16(x[i]));Cuda(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
}
static void Case(DataType type,int rows,int groups,int width){
    Data key(type,{1,rows,groups*width}),query(type,key.dims),value(type,{1,rows,width}),output(FLOAT32,key.dims),reference(FLOAT32,key.dims);
    for(auto d:{&key,&query,&value,&output,&reference}){d->ToDevice(CUDA,{0},false);d->Allocate(false);}
    auto prepare=[&](int seed){std::mt19937 rng(seed+17);std::uniform_real_distribution<float>dist(-1,1);std::vector<float>k(key.Count(0)),q(k.size()),v(value.Count(0));for(size_t i=0;i<k.size();++i){k[i]=dist(rng);q[i]=seed==1 ? 0.f : seed==2 ? k[i]*(i%2?1.f:-1.f) : dist(rng);}for(auto &x:v)x=dist(rng);Upload(key,k);Upload(query,q);Upload(value,v);};
    auto run=[&](Data &out){Check(FastllmCudaQwen4PLEGate(key,query,value,out,groups),"gate rejected");};
    prepare(0);Policy(false);run(reference);Policy(true);run(output);Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph;cudaGraphExec_t exec;Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run(output);Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    double worst=0;
    Policy(false);
    for(int seed=1;seed<=3;++seed){prepare(seed);run(reference);Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaDeviceSynchronize());std::vector<float>a(output.Count(0)),b(a.size());Cuda(cudaMemcpy(a.data(),output.cudaData,output.GetBytes(),cudaMemcpyDeviceToHost));Cuda(cudaMemcpy(b.data(),reference.cudaData,reference.GetBytes(),cudaMemcpyDeviceToHost));double e=0,n=0;for(size_t i=0;i<a.size();++i){Check(std::isfinite(a[i]),"nonfinite");e+=double(a[i]-b[i])*(a[i]-b[i]);n+=double(b[i])*b[i];}worst=std::max(worst,sqrt(e/std::max(n,1e-30)));}
    Check(worst<.003,"PLE L2");printf("PASS PLE dtype=%d rows=%d groups=%d width=%d max_l2=%.8g changed-input graph\n",int(type),rows,groups,width,worst);
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
}
int main(int,char**){try{FastllmCudaSetDevice(0);for(auto type:{FLOAT32,FLOAT16,BFLOAT16}){for(int rows:{1,2,4,7,17,65})Case(type,rows,3,37);Case(type,1,4,2560);Case(type,4,4,2560);Case(type,9,2,257);}return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
