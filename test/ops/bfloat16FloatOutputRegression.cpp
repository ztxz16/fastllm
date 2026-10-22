#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void Cu(cudaError_t e){Check(e==cudaSuccess,cudaGetErrorString(e));}
static float Value(uint64_t i,uint64_t seed){uint32_t x=uint32_t(i+seed)*1664525u+1013904223u;x^=x>>15;x*=2246822519u;
 return RoundFloat32ToBFloat16RNE((int(x%65521)-32760)*(1.0f/32760.0f));}
static void Upload(Data &d){d.ToDevice(DataDevice::CUDA,std::vector<int>{0});}
static std::vector<float> Read(Data &d){std::vector<float>x(d.Count(0));Cu(cudaMemcpy(x.data(),d.cudaData,x.size()*4,cudaMemcpyDeviceToHost));return x;}
int main(){
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1)return 77;
 cudaDeviceProp props;cudaGetDeviceProperties(&props,0);if(props.major<8)return 77;
 try{
  SetDeviceMap({{"cuda:0",1}});FastllmCudaSetDevice(0);
  constexpr int H=5120,V=16160;
  std::vector<float>w(H*V);for(size_t i=0;i<w.size();++i)w[i]=Value(i,29)*.0625f;
  Data weight(FLOAT32,{V,H},w),compact;Upload(weight);
  Check(FastllmCudaTryCompactBFloat16Weight(weight,compact),"exact checkpoint compaction rejected");
  int checks=0;double worst=0,oldWorst=0;
  for(int rows:{1,5,8}){
   std::vector<float>x(rows*H);for(size_t i=0;i<x.size();++i)x[i]=Value(i,7);
   Data input(BFLOAT16,{1,rows,H},x),output;Upload(input);
   auto run=[&](){Check(FastllmCudaBFloat16LinearFloat32(input,compact,output),"BF16 FP32-output projection rejected");};
   run();Cu(cudaDeviceSynchronize());Check(FastllmCudaGraphPrepareCaptureDevice(),"prepare");
   Check(FastllmCudaGraphMemoryPoolBegin(),"pool begin");Check(FastllmCudaGraphBeginCapture(),"capture begin");run();
   void *graph=nullptr,*exec=nullptr;std::vector<void*>pinned;
   Check(FastllmCudaGraphEndCapture(&graph),"capture end");Check(FastllmCudaGraphMemoryPoolEnd(pinned),"pool end");
   Check(FastllmCudaGraphInstantiate(graph,&exec),"instantiate");
   for(int iteration=0;iteration<10;++iteration){
    for(size_t i=0;i<x.size();++i)x[i]=Value(i,iteration*379+7);
    Data changed(BFLOAT16,{1,rows,H},x);Upload(changed);input.CopyFrom(changed);
    if(iteration&1)Check(FastllmCudaGraphLaunch(exec),"replay");else run();
    Cu(cudaDeviceSynchronize());Check(output.dataType==FLOAT32,"logits were rounded to BF16");auto actual=Read(output);
    Data floatInput(FLOAT32,{1,rows,H},x),reference;Upload(floatInput);Linear(floatInput,weight,Data(),reference);auto old=Read(reference);
    for(int row=0;row<rows;++row){
     auto first=actual.begin()+row*V;int top=std::max_element(first,first+V)-first;
     auto oldFirst=old.begin()+row*V;int oldTop=std::max_element(oldFirst,oldFirst+V)-oldFirst;
     float delta=0;for(int v=0;v<V;++v){Check(std::isfinite(first[v]),"nonfinite result");delta=std::max(delta,std::abs(first[v]-oldFirst[v]));}
     float runner=-INFINITY;for(int v=0;v<V;++v)if(v!=oldTop)runner=std::max(runner,oldFirst[v]);
     if(oldFirst[oldTop]-runner>2*delta)Check(top==oldTop,"argmax changed beyond measured error margin");
     for(int sample=0;sample<66;++sample){int v=sample<64?(sample*251+iteration*73)%V:sample==64?top:oldTop;
      double exact=0,absolute=0;for(int h=0;h<H;++h){double product=double(x[row*H+h])*w[v*H+h];exact+=product;absolute+=std::abs(product);}
      double err=std::abs(first[v]-exact),prev=std::abs(oldFirst[v]-exact);worst=std::max(worst,err);oldWorst=std::max(oldWorst,prev);
      Check(err<=std::max(2e-6,absolute*1e-6),"FP64 dot-product error bound exceeded");++checks;
     }
    }
   }
   Cu(cudaDeviceSynchronize());FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pinned);
  }
  for(float value:{0.1f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
   Data invalid(FLOAT32,{1,16},std::vector<float>(16,value)),unused;Upload(invalid);
   Check(!FastllmCudaTryCompactBFloat16Weight(invalid,unused),"lossy/nonfinite compaction accepted");
  }
  std::cout<<"PASS "<<checks<<" BF16 FP32-output oracle checks; max_abs_error="<<worst<<" original_fp32="<<oldWorst<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
