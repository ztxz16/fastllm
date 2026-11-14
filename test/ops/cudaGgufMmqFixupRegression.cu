// Independent producer-partition oracle for typed stream-K output reduction.
#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-mmq-output.cuh"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <vector>
#include <random>
#include <cstring>
#include <cstdio>
#include <stdexcept>
namespace fastllm_gguf_mmq {
#include "mmq.cuh"
}
using namespace fastllm_gguf_mmq;
static void C(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Need(bool b){if(!b)throw std::runtime_error("fixup mismatch or graph failure");}
template<class T>T Convert(float x);template<>float Convert(float x){return x;}template<>half Convert(float x){return __float2half_rn(x);}template<>__nv_bfloat16 Convert(float x){return __float2bfloat16_rn(x);}
template<int Q,int X,class T>int Case(int m,int n,int k,int producers){
 const int y=get_mmq_y_host(ggml_cuda_info().devices[0].cc),nx=(m+X-1)/X,ny=(n+y-1)/y;
 const int stride=n+3;std::vector<float> source(size_t(m)*stride),last(size_t(producers)*X*y);
 std::vector<T> expected(size_t(m)*stride+8,Convert<T>(-17.f)),actual(expected.size());
 float *ds,*dl;T *out;C(cudaMalloc(&ds,source.size()*4));C(cudaMalloc(&dl,last.size()*4));C(cudaMalloc(&out,expected.size()*sizeof(T)));
 mmq_args args{};args.dst=ds;args.ne00=k;args.ne01=n;args.ne11=m;args.ne0=stride;
 auto run=[&]{launch_mmq_fixup_to_output<Q,X,true>(args,dl,out,dim3(ny,nx),producers,cudaStreamPerThread);};
 std::mt19937 random(123+Q+X+m+n+k);
 void *graph=nullptr,*exec=nullptr;Need(FastllmCudaGraphPrepareCaptureDevice());Need(FastllmCudaGraphBeginCapture());run();Need(FastllmCudaGraphEndCapture(&graph));Need(FastllmCudaGraphInstantiate(graph,&exec));
 for(int rep=0;rep<3;++rep){
  for(float &v:source)v=(int(random()%10001)-5000)*.001f;
  for(float &v:last)v=(int(random()%10001)-5000)*.0001f;
  std::fill(expected.begin(),expected.end(),Convert<T>(-17.f));
  for(int row=0;row<m;++row)for(int col=0;col<n;++col){
   volatile float sum=0.f;bool any=false;const int64_t tile=int64_t(row/X)*ny+col/y,blocks=k/Q,total=blocks*nx*ny;
   // Enumerate all producers independently of the GPU's tile-to-producer interval.
   for(int b=0;b<producers;++b){
    int64_t a=int64_t(b)*total/producers,z=int64_t(b+1)*total/producers;
    a-=(a%blocks)%(MMQ_ITER_K/Q);z-=(z%blocks)%(MMQ_ITER_K/Q);
    if(a==z||z%blocks==0||z/blocks!=tile)continue;
    any=true;sum=sum+last[size_t(b)*X*y+(row%X)*y+col%y];
   }
   float value=source[size_t(row)*stride+col];if(any)value+=sum;
   expected[size_t(row)*stride+col]=Convert<T>(value);
  }
  std::fill(actual.begin(),actual.end(),Convert<T>(-17.f));C(cudaMemcpy(out,actual.data(),actual.size()*sizeof(T),cudaMemcpyHostToDevice));C(cudaMemcpy(ds,source.data(),source.size()*4,cudaMemcpyHostToDevice));C(cudaMemcpy(dl,last.data(),last.size()*4,cudaMemcpyHostToDevice));
  if(rep)Need(FastllmCudaGraphLaunch(exec));else run();
  C(cudaStreamSynchronize(cudaStreamPerThread));C(cudaMemcpy(actual.data(),out,actual.size()*sizeof(T),cudaMemcpyDeviceToHost));Need(!memcmp(actual.data(),expected.data(),actual.size()*sizeof(T)));
 }
 FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);C(cudaFree(ds));C(cudaFree(dl));C(cudaFree(out));return 3;
}
template<int Q,int X>int Cases(){int count=0;for(auto shape:{std::vector<int>{6,7,256},{12,512,5120},{17,129,1280},{33,1024,4096},{129,1,256}})for(int p:{1,7,110}){
 count+=Case<Q,X,float>(shape[0],shape[1],shape[2],p);count+=Case<Q,X,half>(shape[0],shape[1],shape[2],p);count+=Case<Q,X,__nv_bfloat16>(shape[0],shape[1],shape[2],p);}return count;}
int main(){try{FastllmCudaSetDevice(0);int n=0;
#define CHECK(Q) n+=Cases<Q,8>();n+=Cases<Q,32>();n+=Cases<Q,64>();n+=Cases<Q,128>()
 CHECK(32);CHECK(64);CHECK(256);
#undef CHECK
 printf("PASS fixup bitwise CPU oracle/guard/changed-input graph cases=%d\n",n);return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
