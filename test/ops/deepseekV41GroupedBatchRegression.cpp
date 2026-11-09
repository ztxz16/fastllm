// Grouped projection layout, tails and graph lifetime against independent rows.
#include "fastllm.h"
#include "gguf.h"
#include "executor.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static void C(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static std::vector<float> Read(Data &x){Data y;y.CopyFrom(x);y.ToDevice(CPU);ToDataTypeForceCPU(y,FLOAT32);auto p=(float*)y.cpuData;return {p,p+y.Count(0)};}
static double Compare(Data &x,Data &y,double limit){auto a=Read(x),b=Read(y);Check(a.size()==b.size(),"shape");double e=0,n=0;for(size_t i=0;i<a.size();++i){Check(std::isfinite(b[i]),"nonfinite");e+=double(a[i]-b[i])*(a[i]-b[i]);n+=double(a[i])*a[i];}double l=sqrt(e/std::max(n,1e-30));Check(l<=limit,"row-reference L2");return l;}
struct Graph{void *g=nullptr,*e=nullptr;std::vector<void*> memory;
 Graph(const std::function<void()> &f){C(cudaDeviceSynchronize());Check(FastllmCudaGraphMemoryPoolBegin()&&FastllmCudaGraphBeginCapture(),"capture begin");f();Check(FastllmCudaGraphEndCapture(&g),"capture end");Check(FastllmCudaGraphMemoryPoolEnd(memory),"capture pool");Check(FastllmCudaGraphInstantiate(g,&e),"instantiate");}
 void run(){Check(FastllmCudaGraphLaunch(e),"graph launch");}
 ~Graph(){C(cudaDeviceSynchronize());FastllmCudaGraphExecDestroy(e);FastllmCudaGraphDestroy(g);FastllmCudaGraphMemoryPoolRelease(memory);}};
int main(int argc,char **argv){try{
 bool bench=argc>1;ApplyDeviceMap({{"cuda:0",1}},0,1);FastllmCudaSetDevice(0);FastllmCudaSetLinearExactBatchThreshold(1);SetThreads(4);
 int cases=0;double worst=0;
 for(int exactThreshold: {1,32}){
  if(bench&&exactThreshold!=1)continue;
  FastllmCudaSetLinearExactBatchThreshold(exactThreshold);
 for(auto dtype: {FLOAT32,FLOAT16,BFLOAT16})for(int shape=0;shape<2;++shape){
  if(bench&&(dtype!=BFLOAT16||shape!=1))continue;
  int groups=shape?8:3,n=shape?1024:129,k=shape?4096:256;
  Data w(DATA_GGUF_FORMAT,GGML_TYPE_Q5_K,{groups*n,k});w.disableGGUFRepack=true;
  std::mt19937 rng(831+shape);std::vector<float> weights(size_t(groups)*n*k);for(auto &v:weights)v=(int(rng()%2001)-1000)*.0001f;
  w.CreateFromOriData(WeightType::LINEAR,FLOAT32,(uint8_t*)weights.data(),nullptr,nullptr);w.ToDevice(CUDA,{0},true);
  for(int tokens: {5,6,8,9,10,11,12,13,14,15,16,17,24,31,32,33,65}){
   if(bench&&tokens==5)continue;
   std::vector<float> values(size_t(tokens)*groups*k);for(auto &v:values)v=(int(rng()%2001)-1000)*.0003f;
   Data input(dtype,{1,tokens,groups,k},values),y,ref(dtype,{1,tokens,groups*n});input.ToDevice(CUDA,{0},true);ref.ToDevice(CUDA,{0},false);ref.Allocate(false);
   auto run=[&]{DeepSeekV4WoA(input,w,groups,n,y);};
   auto reference=[&]{for(int t=0;t<tokens;++t)for(int g=0;g<groups;++g){Data a,b,c;a.FakeFrom(input,size_t(t*groups+g)*k*input.unitSize);a.Resize({1,k});b.FakeFrom(w,size_t(g)*n*ggml_row_size(GGML_TYPE_Q5_K,k));b.ggmlType=GGML_TYPE_Q5_K;b.dims={n,k};c.FakeFrom(ref,size_t(t*groups+g)*n*input.unitSize);c.Resize({1,n});Linear(a,b,Data(),c);}};
   run();reference();worst=std::max(worst,Compare(ref,y,(tokens>16||(bench&&tokens>8))?.025:(dtype==FLOAT32?.0001:.004)));
   // The existing FLOAT32 MMQ fallback creates temporary bias storage during capture.
   // Check its eager result; graph coverage here targets the grouped kernel and half types.
   if(dtype==FLOAT32&&tokens>16){printf("EAGER legacy FLOAT32 rows=%d shape=%d\n",tokens,shape);++cases;continue;}
   Graph graph(run);
   if(bench){for(int rep=0;rep<3;++rep){for(int i=0;i<10;++i)graph.run();cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));C(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<100;++i)graph.run();C(cudaEventRecord(b,cudaStreamPerThread));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));C(cudaEventDestroy(a));C(cudaEventDestroy(b));printf("BENCH grouped rows=%d rep=%d us=%.6f\n",tokens,rep,ms*10);fflush(stdout);}}
   else for(int rep=0;rep<2;++rep){for(auto &v:values)v=rep?0.f:(int(rng()%2001)-1000)*.0003f;Data host(dtype,input.dims,values);C(cudaMemcpy(input.cudaData,host.cpuData,input.GetBytes(),cudaMemcpyHostToDevice));reference();C(cudaMemset(y.cudaData,255,y.GetBytes()));graph.run();worst=std::max(worst,Compare(ref,y,(tokens>16||(bench&&tokens>8))?.025:(dtype==FLOAT32?.0001:.004)));++cases;}
   printf("CASE exact=%d dtype=%d rows=%d shape=%d worst=%.9g\n",exactThreshold,int(dtype),tokens,shape,worst);fflush(stdout);
  }
 }
 }
 printf("PASS grouped batch cases=%d worstL2=%.9g\n",cases,worst);return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
