#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <memory>
#include <stdexcept>
#include <vector>
#include "utils/kernel_registry.h"
using namespace fastllm;
static void Check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
static void Cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Alloc(Data &d){d.ToDevice(CUDA,{0},false);d.Allocate(false);}
static void Fill(Data &d,int seed,float scale=1.f){std::mt19937 rng(seed);std::uniform_real_distribution<float> f(-scale,scale);std::vector<float>x(d.Count(0));for(auto&v:x)v=f(rng);
 if(d.dataType==FLOAT32)Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));
 else if(d.dataType==FLOAT16){std::vector<half>h(x.size());for(size_t i=0;i<h.size();++i)h[i]=__float2half(x[i]);Cuda(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
 else{std::vector<unsigned short>h(x.size());for(size_t i=0;i<h.size();++i){uint32_t u;memcpy(&u,&x[i],4);h[i]=u>>16;}Cuda(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}}
static double L2(Data&a,Data&b){std::vector<float>x(a.Count(0)),y(x.size());Cuda(cudaMemcpy(x.data(),a.cudaData,a.GetBytes(),cudaMemcpyDeviceToHost));Cuda(cudaMemcpy(y.data(),b.cudaData,b.GetBytes(),cudaMemcpyDeviceToHost));double s=0,e=0;for(size_t i=0;i<x.size();++i){Check(std::isfinite(x[i])&&std::isfinite(y[i]),"nonfinite");s+=double(y[i])*y[i];e+=double(x[i]-y[i])*(x[i]-y[i]);}return sqrt(e/std::max(s,1e-30));}

static std::vector<char> Bytes(Data&d){std::vector<char>x(d.GetBytes());Cuda(cudaMemcpy(x.data(),d.cudaData,x.size(),cudaMemcpyDeviceToHost));return x;}
static void Case(int batch,int channels,int taps,DataType type,bool silu,bool initialize,const char*candidate,bool unaligned=false){
 std::string error;json11::Json::object policy{{"mode","quick"},{"families",json11::Json::array{"conv.causal_depthwise.decode"}},{"budget_ms",30000}};
 if(candidate[0])policy["force"]=json11::Json::object{{"conv.causal_depthwise.decode",candidate}};
 Check(kernels::Configure(json11::Json(policy).dump(),error),error.c_str());
 Data x(type,{batch,channels,1}),w(FLOAT32,{channels,taps}),h(FLOAT32,{batch,channels,taps}),o(FLOAT32,{batch,1,channels}),rh(FLOAT32,h.dims),ro(FLOAT32,o.dims);
 Data historyStorage(FLOAT32,{batch*channels*taps+1}),weightStorage(FLOAT32,{channels*taps+1});
 if(unaligned){Alloc(historyStorage);Alloc(weightStorage);h.FakeFrom(historyStorage,4);w.FakeFrom(weightStorage,4);}
 for(Data*d:{&x,&w,&h,&o,&rh,&ro}){if(!unaligned || (d!=&h && d!=&w))Alloc(*d);Fill(*d,17,3);}Cuda(cudaMemcpy(rh.cudaData,h.cudaData,h.GetBytes(),cudaMemcpyDeviceToDevice));
 auto run=[&](bool ref,bool init){auto op=[&]{Check(FastllmCudaCausalDepthwiseConv1DDecode(x,w,ref?rh:h,ref?ro:o,taps,silu,init),"decode rejected");};if(ref){kernels::Bypass bypass;op();}else op();};
 auto before=Bytes(h);Data bad(FLOAT32,{batch,1,channels+1});Alloc(bad);
 Check(!FastllmCudaCausalDepthwiseConv1DDecode(x,w,h,bad,taps,silu,true),"bad shape accepted");Check(Bytes(h)==before,"bad shape mutated state");
 run(true,initialize);run(false,initialize);double maxL2=L2(o,ro);Check(Bytes(h)==Bytes(rh),"first tuned call advanced history more than once");
 cudaGraph_t graph;cudaGraphExec_t exec;Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run(false,false);Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));size_t nodes=0;Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));Check(nodes==1,"decode should capture one kernel");Cuda(cudaGraphInstantiate(&exec,graph,0,0,0));
 for(int step=0;step<7;++step){Fill(x,step*37+41,step==6?80.f:5.f);run(true,false);Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaDeviceSynchronize());Check(Bytes(h)==Bytes(rh),"history differs");maxL2=std::max(maxL2,L2(o,ro));}
 Check(maxL2<1e-5,"output L2");Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
 printf("PASS batch=%d channels=%d taps=%d type=%d silu=%d init=%d candidate=%s output_l2=%.9g history=exact\n",batch,channels,taps,int(type),int(silu),int(initialize),candidate,maxL2);fflush(stdout);
}
int main(int,char**){try{FastllmCudaSetDevice(0);
 for(auto type:{FLOAT32,FLOAT16,BFLOAT16})for(const char*candidate:{"cuda.conv.decode.vec4.block64","cuda.conv.decode.vec4.block128","cuda.conv.decode.vec4.block256"}){
  Case(1,10240,4,type,true,false,candidate);Case(2,10240,4,type,true,true,candidate);Case(4,10240,4,type,true,false,candidate);
  Case(7,129,4,type,false,true,candidate);Case(33,17,4,type,true,false,candidate);
 }
 for(int taps:{1,2,3,5,7})Case(3,131,taps,FLOAT16,true,true,"");
 Case(5,10240,4,FLOAT16,true,false,"");
 Case(3,129,4,FLOAT16,true,false,"",true);
 return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
