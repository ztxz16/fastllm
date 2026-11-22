#include "devices/cuda/low_latency/router.cuh"
#include "devices/cuda/low_latency/dense.cuh"
#include "devices/cuda/low_latency/dense.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <functional>
using B=__nv_bfloat16;
namespace ll=fastllm::cuda::low_latency;
void Check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
void C(cudaError_t e){Check(e==cudaSuccess,cudaGetErrorString(e));}
float Round(float x){return float(B(x));}
template<class T> struct Buf {
 T*p;std::vector<T> h;Buf(size_t n):h(n){C(cudaMalloc(&p,n*sizeof(T)));}~Buf(){cudaFree(p);}
 void Up(){C(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
 void Down(){C(cudaMemcpy(h.data(),p,h.size()*sizeof(T),cudaMemcpyDeviceToHost));}
 void Fill(unsigned seed,float scale){for(auto&v:h){seed=1664525*seed+1013904223;v=T((int(seed>>16)-32768)/32768.f*scale);}Up();}
};
void Replay(cudaStream_t s,const std::function<void()>&f){
 f();C(cudaStreamSynchronize(s));cudaGraph_t g;cudaGraphExec_t e;
 C(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));f();C(cudaStreamEndCapture(s,&g));
 C(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));for(int i=0;i<4;++i)C(cudaGraphLaunch(e,s));
 C(cudaStreamSynchronize(s));C(cudaGraphExecDestroy(e));C(cudaGraphDestroy(g));
}
template<int K> void Router(int rows,int experts,cudaStream_t stream){
 Buf<B>x(size_t(rows)*K);Buf<float>w(size_t(experts)*K),out(size_t(rows)*experts+16);
 x.Fill(11,.5f);w.Fill(17,.03f);for(auto&v:out.h)v=-1024;out.Up();
 auto run=[&]{if(rows==1)ll::RouterSigmoidSingle<K><<<experts,256,0,stream>>>(x.p,w.p,out.p+8);
 else ll::RouterSigmoidRows<4,K><<<dim3(experts,(rows+3)/4),256,0,stream>>>(x.p,w.p,out.p+8,rows,experts);C(cudaGetLastError());};
 Replay(stream,run);out.Down();
 for(int row=0;row<rows;++row){double err=0,mag=0;for(int e=0;e<experts;++e){double dot=0;
  for(int k=0;k<K;++k)dot+=double(float(x.h[row*K+k]))*w.h[e*K+k];
  float expected=1/(1+std::exp(-dot)),actual=out.h[8+row*experts+e];
  err+=std::pow(double(expected)-actual,2);mag+=double(expected)*expected;}
  Check(std::sqrt(err/mag)<1e-6,"router row L2");}
 if constexpr(K==4096){if(experts%4==0){auto ref=out.h;
  Replay(stream,[&]{ll::RouterSigmoidTiled<4,8,K><<<dim3(experts/4,(rows+7)/8),256,0,stream>>>(x.p,w.p,out.p+8,rows,experts);C(cudaGetLastError());});out.Down();
  for(int r=0;r<rows;++r){double err=0,mag=0;for(int e=0;e<experts;++e){float a=out.h[8+r*experts+e],b=ref[8+r*experts+e];err+=std::pow(double(a)-b,2);mag+=double(b)*b;}
   Check(std::sqrt(err/mag)<1e-6,"tiled router row L2");}
 }}
 for(int i=0;i<8;++i)Check(out.h[i]==-1024&&out.h[out.h.size()-i-1]==-1024,"router row tail overwrite");
}
template<class Kernel>bool Prepare(Kernel kernel,size_t bytes){
 cudaFuncAttributes a{};if(cudaFuncGetAttributes(&a,kernel)!=cudaSuccess){cudaGetLastError();return false;}
 int max=0;C(cudaDeviceGetAttribute(&max,cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
 if(bytes+a.sharedSizeBytes>size_t(max))return false;
 if(bytes+a.sharedSizeBytes>48*1024)C(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 return true;
}
template<int H,int I,bool Norm>void Dense(int rows,cudaStream_t stream,size_t limit=SIZE_MAX){
 constexpr int upBytes=ll::DenseGateUpSharedBytes<H>,downBytes=ll::DenseDownSharedBytes<I>;
 if constexpr(Norm) if(!Prepare(ll::DenseGateUp<H,I,Norm>,upBytes)||!Prepare(ll::DenseDown<I,H,false>,downBytes)||!Prepare(ll::DenseDown<I,H,true>,downBytes))return;
 Buf<B>x(size_t(rows)*H),gate(size_t(I)*H),up(size_t(I)*H),down(size_t(H)*I),mid(size_t(rows)*I+16),out(size_t(rows)*H+16),res(size_t(rows)*H);
 Buf<float>gamma(H);x.Fill(21,.5f);gate.Fill(22,.03f);up.Fill(23,.03f);down.Fill(24,.03f);res.Fill(25,.1f);
 gamma.Fill(26,.1f);for(auto&v:gamma.h)v+=1;gamma.Up();
 for(auto&v:mid.h)v=B(-1024);mid.Up();for(auto&v:out.h)v=B(-1024);out.Up();
 ll::DensePlan plan;
 ll::DenseBindings bindings{{x.p,x.h.size()*2},{gate.p,gate.h.size()*2},
  {up.p,up.h.size()*2},{down.p,down.h.size()*2},
  {mid.p+8,(mid.h.size()-8)*2},{out.p+8,(out.h.size()-8)*2}};
 if constexpr(Norm) {
  Replay(stream,[&]{ll::DenseGateUp<H,I,Norm><<<dim3(I/16,(rows+7)/8),256,upBytes,stream>>>(x.p,gate.p,up.p,gamma.p,1e-5f,mid.p+8,rows);C(cudaGetLastError());
   ll::DenseDown<I,H,false><<<dim3(H/64,(rows+7)/8),256,downBytes,stream>>>(mid.p+8,down.p,nullptr,out.p+8,rows);C(cudaGetLastError());});
 } else {
  C(ll::PrepareDense({rows,H,I},plan,stream,{limit}));
  if(limit<=99*1024 && H==8192)Check(plan.RowTile()<8,"Dense resource tile did not shrink");
  Replay(stream,[&]{C(ll::RunDense(plan,bindings,stream));});
  auto bad=bindings;bad.output.bytes=size_t(rows)*H*2-1;
  Check(ll::RunDense(plan,bad,stream)==cudaErrorInvalidValue,"short Dense output accepted");
  bad=bindings;bad.intermediate={x.p,x.h.size()*2};
  Check(ll::RunDense(plan,bad,stream)==cudaErrorInvalidValue,"Dense alias accepted");
  Check(ll::PrepareDense({rows,H,I},plan,stream,{1})==cudaErrorNotSupported,"impossible Dense budget accepted");
  Check(plan.Ready(),"failed Dense preparation erased plan");
  cudaGraph_t graph;cudaGraphExec_t exec;
  C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  ll::DensePlan rejected;
  Check(ll::PrepareDense({rows,H,I},rejected,stream)==cudaErrorStreamCaptureUnsupported,"Dense prepare inside capture");
  C(ll::RunDense(plan,bindings,stream));C(cudaStreamEndCapture(stream,&graph));
  size_t nodes=0;C(cudaGraphGetNodes(graph,nullptr,&nodes));Check(nodes==2,"Dense logical rows split into separate launches");
  C(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));C(cudaGraphLaunch(exec,stream));
  C(cudaStreamSynchronize(stream));C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(graph));
 }
 mid.Down();out.Down();double maxGate=0,maxDown=0;
 for(int row=0;row<rows;++row){std::vector<float> input(H),middle(I);double sq=0;for(int k=0;k<H;++k){input[k]=float(x.h[row*H+k]);sq+=double(input[k])*input[k];}
  if constexpr(Norm){float inv=1/std::sqrt(sq/H+1e-5f);for(int k=0;k<H;++k)input[k]=Round(Round(input[k]*inv)*gamma.h[k]);}
  double err=0,mag=0;for(int i=0;i<I;++i){double a=0,b=0;for(int k=0;k<H;++k){a+=double(input[k])*float(gate.h[i*H+k]);b+=double(input[k])*float(up.h[i*H+k]);}
   float g=Round(a),u=Round(b);middle[i]=Round(Round(g/(1+std::exp(-g)))*u);float actual=float(mid.h[8+row*I+i]);err+=std::pow(double(actual)-middle[i],2);mag+=double(middle[i])*middle[i];}
  double l2=std::sqrt(err/(mag+1e-30));maxGate=std::max(maxGate,l2);Check(l2<.006,"dense gate/up row L2");
  err=mag=0;for(int h=0;h<H;++h){double sum=0;for(int i=0;i<I;++i)sum+=double(float(mid.h[8+row*I+i]))*float(down.h[h*I+i]);
   float expected=Round(sum),actual=float(out.h[8+row*H+h]);err+=std::pow(double(actual)-expected,2);mag+=double(expected)*expected;}
  l2=std::sqrt(err/(mag+1e-30));maxDown=std::max(maxDown,l2);Check(l2<.006,"dense down row L2");
 }
 if(Prepare(ll::DenseDown<I,H,true>,downBytes)) {
  const auto before=out.h;
  Replay(stream,[&]{ll::DenseDown<I,H,true><<<dim3(H/64,(rows+7)/8),256,downBytes,stream>>>(mid.p+8,down.p,res.p,out.p+8,rows);C(cudaGetLastError());});out.Down();
  for(int i=0;i<rows*H;++i)Check(float(out.h[8+i])==Round(float(before[8+i])+float(res.h[i])),"residual BF16 boundary");
 }
 for(int i=0;i<8;++i)Check(float(mid.h[i])==-1024&&float(mid.h[mid.h.size()-i-1])==-1024&&float(out.h[i])==-1024&&float(out.h[out.h.size()-i-1])==-1024,"dense row tail overwrite");
 printf("DENSE rows=%d H=%d I=%d norm=%d gate_l2=%.7g down_l2=%.7g\n",rows,H,I,Norm,maxGate,maxDown);
}
int main(){try{int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess||!n)return 77;cudaDeviceProp p{};C(cudaGetDeviceProperties(&p,0));if(p.major<7)return 77;
 cudaStream_t s;C(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 for(int rows:{1,2,3,4,8,9,17,64}){Router<1024>(rows,17,s);Router<4096>(rows,256,s);}
#ifdef CUDA_NO_TENSOR_CORE
 const bool mma=false;
#else
 const bool mma=p.major>=8;
#endif
 if(mma){
 for(int rows:{1,2,4,8,9,17}){Dense<1024,256,true>(rows,s);Dense<1024,256,false>(rows,s);}
 for(int rows:{1,2,8,9}){Dense<4096,2048,true>(rows,s);Dense<4096,2048,false>(rows,s);}
 for(size_t limit:{size_t(48*1024),size_t(99*1024)})Dense<8192,2048,false>(9,s,limit);
 }else{
  ll::DensePlan rejected;
  Check(ll::PrepareDense({2,4096,2048},rejected,s)==cudaErrorNotSupported,"pre-SM80 Dense did not fall back");
  puts("SKIP unavailable BF16 MMA; portable router remains active");
 }
 C(cudaStreamDestroy(s));puts("PASS portable router and dense fusions");return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
