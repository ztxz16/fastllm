#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include "fastllm-cuda-graph-workspace.h"
#include "devices/cuda/fastllm-draft-head.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
using namespace fastllm;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static std::mt19937 rng(823);
static void alloc(Data &d,std::vector<int> dims,const std::vector<__nv_bfloat16>&v) {
 d.dataType=BFLOAT16;d.UpdateUnitSize();d.Resize(dims);d.dataDevice=CUDA;d.dataDeviceIds={0};d.Allocate(false);
 CHECK(cudaMemcpy(d.cudaData,v.data(),d.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
}
static std::vector<double> quantize(const std::vector<__nv_bfloat16> &v){
 std::vector<double> y(v.size());
 for(size_t b=0;b<v.size();b+=32){
  float maxv=1e-4f;for(int j=0;j<32;j++)maxv=std::max(maxv,std::abs(float(v[b+j])));
  float scale=std::exp2(std::ceil(std::log2(maxv/448.f)));
  for(int j=0;j<32;j++){__nv_fp8_e4m3 q(float(v[b+j])/scale);y[b+j]=double(float(q))*scale;}
 }
 return y;
}
static void test(int m,int n,int k){
 std::vector<__nv_bfloat16>x(m*k),w(n*k);
 for(auto &v:x)v=__float2bfloat16_rn(float(int(rng()%10001)-5000)/1751.f);
 for(auto &v:w)v=__float2bfloat16_rn(float(int(rng()%10007)-5003)/70013.f);
 Data input,weight,out;alloc(input,{m,k},x);alloc(weight,{n,k},w);
 unsetenv("FASTLLM_DSPARK_FP8_HEAD");CHECK(FastllmCudaTryFp8DraftHead(input,weight,out));
 CHECK(out.dataType==FLOAT32);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 std::vector<float>actual(m*n);CHECK(cudaMemcpy(actual.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
 auto qx=quantize(x),qw=quantize(w);double err=0,den=0,qerr=0,originalDen=0;
 for(int i=0;i<m;i++)for(int j=0;j<n;j++){
  double ref=0,original=0;
  for(int c=0;c<k;c++){ref+=qx[i*k+c]*qw[j*k+c];original+=double(float(x[i*k+c]))*float(w[j*k+c]);}
  float a=actual[i*n+j];CHECK(std::isfinite(a));err+=(a-ref)*(a-ref);den+=ref*ref;qerr+=(a-original)*(a-original);originalDen+=original*original;
 }
 const double rel=std::sqrt(err/den),loss=std::sqrt(qerr/originalDen);CHECK(rel<2e-5);CHECK(loss<0.06);
 // Shared target weights must be byte-identical after the draft copy is built.
 std::vector<__nv_bfloat16>after(w.size());CHECK(cudaMemcpy(after.data(),weight.cudaData,weight.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
 CHECK(std::memcmp(w.data(),after.data(),weight.GetBytes())==0);
 // Changing activations reuses weights but must not reuse quantized activations.
 for(auto &v:x)v=__float2bfloat16_rn(float(v)*0.5f);
 CHECK(cudaMemcpy(input.cudaData,x.data(),input.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
 CHECK(FastllmCudaTryFp8DraftHead(input,weight,out));CHECK(cudaDeviceSynchronize()==cudaSuccess);
 std::vector<float>half(actual.size());CHECK(cudaMemcpy(half.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
 for(size_t i=0;i<half.size();i++)CHECK(std::abs(half[i]-.5f*actual[i])<1e-5f);
 setenv("FASTLLM_DSPARK_FP8_HEAD","0",1);CHECK(!FastllmCudaTryFp8DraftHead(input,weight,out));unsetenv("FASTLLM_DSPARK_FP8_HEAD");
 CHECK(!FastllmCudaTryFp8DraftHead(input,weight,input));
 // Record/retain transient operands exactly as the model graph does.
 std::vector<size_t> allocations;
 {FastllmCudaAllocationRecorder recorder(&allocations);CHECK(FastllmCudaTryFp8DraftHead(input,weight,out));}
 CHECK(cudaDeviceSynchronize()==cudaSuccess);CHECK(FastllmCudaReserveGraphWorkspace(allocations));
 std::vector<void*> retained;void *graph=nullptr,*exec=nullptr;
 CHECK(FastllmCudaGraphPrepareCaptureDevice());CHECK(FastllmCudaGraphMemoryPoolBegin());
 CHECK(FastllmCudaGraphBeginCapture());CHECK(FastllmCudaTryFp8DraftHead(input,weight,out));
 CHECK(FastllmCudaGraphEndCapture(&graph));CHECK(FastllmCudaGraphMemoryPoolEnd(retained));
 CHECK(FastllmCudaGraphInstantiate(graph,&exec));
 for(int replay=0;replay<4;++replay){
  // Power-of-two scaling gives an exact independent replay oracle.
  for(auto &v:x)v=__float2bfloat16_rn(float(v)*0.5f);
  CHECK(cudaMemcpy(input.cudaData,x.data(),input.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
  CHECK(FastllmCudaGraphLaunch(exec));CHECK(cudaDeviceSynchronize()==cudaSuccess);
  CHECK(cudaMemcpy(half.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
  for(size_t i=0;i<half.size();++i)CHECK(std::abs(half[i]-std::ldexp(actual[i],-replay-2))<1e-5f);
 }
 FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);
 for(void *ptr:retained)FastllmCudaFree(ptr);
 std::printf("HEAD m=%d n=%d k=%d math_l2=%.8g quantization_l2=%.8g\n",m,n,k,rel,loss);
}
int main(){
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;FastllmCudaSetDevice(0);
 if(FastllmCudaRuntimeArch()!=120&&FastllmCudaRuntimeArch()!=121)return 77;
 for(int m:{1,2,3,5,7,8,9,16,17,24,32,48,65})for(auto shape:std::vector<std::vector<int>>{{32,32},{160,96},{256,1280},{512,5120}})test(m,shape[0],shape[1]);
 for(int m:{127,128,129,257})for(auto shape:std::vector<std::vector<int>>{{32,32},{160,96}})test(m,shape[0],shape[1]);
 puts("DRAFT_HEAD_PASS cases=60 graph_replays=240");
}
