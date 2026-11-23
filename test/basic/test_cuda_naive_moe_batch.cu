// Multi-row MoE: existing eight-row prepared path is the reference.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "fastllm-cuda-router-topk.cuh"
#include "../../src/devices/cuda/moe/fastllm-shared-flex-tp.cu"
#include <cuda_fp8.h>
#include "fastllm-naive-flex-moe.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
using BF=__nv_bfloat16;
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool x,const char*s) { if(!x)throw std::runtime_error(s); }
__global__ void SelectReference(const float *router,const float *bias,int *ids,float *prob,int needNorm,float routeScale){
 __shared__ fastllm::cuda::RouterTopKScratch<8> scratch;
 fastllm::cuda::SelectRouterTopK<256,8>(router,bias,ids,prob,bias!=nullptr,needNorm,routeScale,scratch);
}
constexpr int H=4096,MaxRows=129;

__global__ void test_quantize(const BF*x,int rows,FastllmMoePreparedInput p){
 int t=blockIdx.x*256+threadIdx.x;
 if(blockIdx.x==0){p.counters[threadIdx.x]=0;if(threadIdx.x==0)++*p.epoch;}
 if(t>=rows*4096)return;float v=float(x[t]),mx=fabsf(v);
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
 unsigned e=max((__float_as_uint(mx)+0x7fffffu-0x600000u)>>23,113u)-8;
 p.values[t]=__nv_fp8_e4m3(v*__uint_as_float((254-e)<<23)).__x;
 if(t%32==0)p.scales[t/32]=e;
}
__global__ void test_coarse(FastllmMoePreparedInput p){
 int row=blockIdx.x/32,k=blockIdx.x%32,t=threadIdx.x;
 __nv_fp8_e4m3 v;v.__x=row<p.rows?p.values[row*4096+k*128+t]:0;
 unsigned old=row<p.rows?p.scales[row*128+k*4+t/32]:105;
 __shared__ unsigned exponents[4];
 if(t%32==0)exponents[t/32]=old;__syncthreads();
 unsigned e=__reduce_max_sync(0xffffffff,t%32<4?exponents[t%32]:0);
 p.block128[row*4112+k*128+t]=__nv_fp8_e4m3(float(v)*__uint_as_float(unsigned(max(0,127+int(old)-int(e)))<<23)).__x;
 if(t==0)reinterpret_cast<float*>(p.block128+32896)[k*8+row]=__uint_as_float(e<<23);
 if(k==0&&t<16)p.block128[row*4112+4096+t]=0;
}
struct StateHandle {
 void* state=nullptr;
 StateHandle() {
  int ids[]={0,1,2,3,4,5,6,7};
  require(!fastshared_tp_create(ids,1,&state),"create");
 }
 ~StateHandle(){fastshared_tp_destroy(state);}
 StateHandle(const StateHandle&)=delete;
 StateHandle& operator=(const StateHandle&)=delete;
};
struct Rank { void*w[4];BF*x,*y[2];int*ids[2];float*p[2],*router,*bias; };
__global__ void weight_fill(unsigned char* p,size_t n,unsigned seed) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x){
  unsigned a=unsigned(i)*1664525u+seed;a^=a>>11;a*=2246822519u;
  p[i]=static_cast<unsigned char>((a&0x80u)| (0x18u+(a%24u)));
 }
}
__global__ void input_fill(BF*x,int n,unsigned seed,float scale) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned v=(i+seed)*1664525u+1013904223u;
 x[i]=__float2bfloat16_rn((int(v>>16)-32768)/32768.f*scale);}
}
__global__ void scales_fill(unsigned char*p,size_t n,unsigned seed){
 for(size_t i=blockIdx.x*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256){unsigned v=unsigned(i)*1664525u+seed;v^=v>>13;p[i]=118+v%12;}
}
template<class F> void parallel(F f) {
 std::array<std::thread,8> workers;std::atomic<int> ready{0};std::exception_ptr error;std::atomic_flag lock=ATOMIC_FLAG_INIT;
 for(int r=0;r<8;++r)workers[r]=std::thread([&,r]{try{check(cudaSetDevice(r));++ready;while(ready.load()<8)std::this_thread::yield();f(r);}catch(...){while(lock.test_and_set()){}error=std::current_exception();lock.clear();}});
 for(auto&t:workers)t.join();if(error)std::rethrow_exception(error);
}
__global__ void Delay(unsigned long long cycles) {unsigned long long start=clock64();while(clock64()-start<cycles){}}
__global__ void RouteFill(float *router,int step) {
 int row=blockIdx.x,e=threadIdx.x,d=(e-(row%16+step*11)+512)%256;
 router[row*256+e]=d<8?.9f-d*.02f:.0001f+e*.000001f;
}
// Reuse one state across mailbox layouts; rotate a delayed rank on every step.
void CheckMixedShapes(std::array<Rank,8>& data,void** states) {
 require(static_cast<fastllm_shared_flex_tp::State*>(states[1])->batch32Supported,"32-row admission");
 static_cast<fastllm_shared_flex_tp::State*>(states[0])->batch32Supported=false;
 require(static_cast<fastllm_shared_flex_tp::State*>(states[1])->batch64Supported,"64-row admission");
 const std::vector<int> shapes={64,32,8,48,16,24,7,64,1,31,8,33,65,63,128,129,32};
 const size_t stride=size_t(MaxRows+1)*H,recordCount=stride*shapes.size();
 std::array<std::array<BF*,2>,8> records;
 parallel([&](int r){for(int j=0;j<2;++j)check(cudaMalloc(&records[r][j],recordCount*2));});
 auto launchStep=[&](int j,int r,int step){
   auto&a=data[r];const int n=shapes[step];
   if(r==step%8)Delay<<<1,32,0,cudaStreamPerThread>>>(200000);
   input_fill<<<(n*H+255)/256,256,0,cudaStreamPerThread>>>(a.x,n*H,17031+step*797,step%5==0?0.f:1.f);
   RouteFill<<<n,256,0,cudaStreamPerThread>>>(a.router,step);
   check(cudaMemsetAsync(a.y[j],0x55,stride*2,cudaStreamPerThread));
   if(n==1) {
     SelectReference<<<1,32,0,cudaStreamPerThread>>>(a.router,nullptr,a.ids[j],a.p[j],1,2.5f);
     require(!fastshared_tp_run(states[j],r,0,1,a.x,a.ids[j],a.p[j],a.y[j]),"single step");
   } else require(!fastshared_tp_run_router(states[j],r,0,n,a.x,a.router,nullptr,a.ids[j],a.p[j],a.y[j],1,2.5f),"mixed step");
   check(cudaMemcpyAsync(records[r][j]+stride*step,a.y[j],stride*2,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
 };
 std::array<std::array<cudaGraphExec_t,8>,2> graph;
 for(int j=0;j<2;++j) {
   parallel([&](int r){for(int step=0;step<int(shapes.size());++step)launchStep(j,r,step);check(cudaStreamSynchronize(cudaStreamPerThread));});
   parallel([&](int r){cudaGraph_t g;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
     for(int repeat=0;repeat<3;++repeat)for(int step=0;step<int(shapes.size());++step)launchStep(j,r,step);
     check(cudaStreamEndCapture(cudaStreamPerThread,&g));check(cudaGraphInstantiate(&graph[j][r],g,0));check(cudaGraphDestroy(g));});
 }
 double worst=0;size_t different=0;
 for(int replay=0;replay<8;++replay) {
   for(int j=0;j<2;++j)parallel([&](int r){check(cudaGraphLaunch(graph[j][r],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));});
   for(int r=0;r<8;++r){check(cudaSetDevice(r));std::vector<unsigned short>x(recordCount),y(recordCount);
     check(cudaMemcpy(x.data(),records[r][0],recordCount*2,cudaMemcpyDeviceToHost));check(cudaMemcpy(y.data(),records[r][1],recordCount*2,cudaMemcpyDeviceToHost));
     for(int step=0;step<int(shapes.size());++step) {
       for(int row=0;row<shapes[step];++row) {double se=0,ss=0;
         for(int c=0;c<H;++c){size_t i=step*stride+row*H+c;float u=__bfloat162float(__ushort_as_bfloat16(x[i])),v=__bfloat162float(__ushort_as_bfloat16(y[i]));
           require(std::isfinite(v),"mixed finite");se+=(u-v)*(u-v);ss+=u*u;different+=x[i]!=y[i];}
         double l2=sqrt(se/std::max(ss,1e-30));worst=std::max(worst,l2);require(l2<.003,"mixed L2");
       }
       for(size_t i=step*stride+size_t(shapes[step])*H;i<(step+1)*stride;++i)require(y[i]==0x5555,"mixed output guard");
     }
   }
   printf("MIXED_REPLAY %d max_row_l2=%.9g differences=%zu\n",replay,worst,different);fflush(stdout);
 }
 for(int j=0;j<2;++j)parallel([&](int r){check(cudaGraphExecDestroy(graph[j][r]));check(cudaFree(records[r][j]));});
 puts("MIXED_SHAPE_PASS 8 ranks,8/16/32/64 banks,ragged tails,delayed rank,changing inputs,graph replay");
}
int main(int argc,char**argv) { try {
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || count<8){puts("SKIP: eight SM103 GPUs required");return 77;}
 for(int r=0;r<8;++r){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,r)!=cudaSuccess || p.major!=10 || p.minor!=3){puts("SKIP: SM103 required");return 77;}}
 const bool stress=argc>1&&!strcmp(argv[1],"stress");
 const bool preparedMode=argc>1&&!strcmp(argv[1],"prepared");
 const bool borrowed=argc>1&&!strcmp(argv[1],"borrowed"),mixed=argc>1&&!strcmp(argv[1],"mixed");
 int needNorm=borrowed?0:1;float routeScale=borrowed?.75f:2.5f;
 StateHandle base,candidate;void* states[]={base.state,candidate.state};
 std::array<Rank,8> data;
 const size_t sizes[]={size_t(256)*512*4096,size_t(256)*4096*256,size_t(256)*512*4096/32,size_t(256)*4096*256/32};
 parallel([&](int r){auto&a=data[r];for(int j=0;j<4;++j){check(cudaMalloc(&a.w[j],sizes[j]));if(j<2)weight_fill<<<2048,256,0,cudaStreamPerThread>>>((unsigned char*)a.w[j],sizes[j],97+r*73+j*137);else scales_fill<<<2048,256,0,cudaStreamPerThread>>>((unsigned char*)a.w[j],sizes[j],73+r*137+j*97);}
  check(cudaMalloc(&a.x,(MaxRows+1)*H*2));check(cudaMalloc(&a.router,MaxRows*256*4));check(cudaMalloc(&a.bias,256*4));
  for(int j=0;j<2;++j){check(cudaMalloc(&a.ids[j],MaxRows*8*4));check(cudaMalloc(&a.p[j],MaxRows*8*4));check(cudaMalloc(&a.y[j],(MaxRows+1)*H*2));std::vector<uint64_t>ptrs(512);std::vector<float>gs(256*128),ds(256*64);
   for(int e=0;e<256;++e){ptrs[2*e]=uint64_t(a.w[0])+size_t(e)*512*4096;ptrs[2*e+1]=uint64_t(a.w[1])+size_t(e)*4096*256;}
   for(size_t i=0;i<gs.size();++i)gs[i]=std::ldexp(1.f,int((i*79+13+r)%7)-3);
   for(size_t i=0;i<ds.size();++i)ds[i]=std::ldexp(1.f,int((i*61+7+r)%7)-3);
   if(borrowed)require(!fastshared_tp_bind(states[j],r,0,a.w[0],a.w[1],a.w[2],a.w[3]),"borrowed bind");
   else require(!fastshared_tp_pack(states[j],r,0,ptrs.data(),gs.data(),ds.data()),"owned pack");}check(cudaStreamSynchronize(cudaStreamPerThread));});
 auto *candidateState=static_cast<fastllm_shared_flex_tp::State*>(candidate.state);
 auto *savedMap=candidateState->layers[0][7].gate128map;
 if(mixed)candidateState->layers[0][7].gate128map=nullptr;
 if(stress)CheckMixedShapes(data,states);
 else for(int n:{1,2,3,4,5,6,7,8,9,15,16,17,24,31,32,33,48,64,65,128,129,8,16}) {
  std::array<std::array<cudaGraphExec_t,8>,2> one{};
  auto fill=[&](int kind,unsigned seed,float scale){
   std::vector<float> router(n*256),bias(256,0);
   for(int row=0;row<n;++row)for(int e=0;e<256;++e){
    int shift=kind==0?0:kind==1?row*8:row*17;
    router[row*256+e]=float(((e+shift)%256)+1)/257.f;
    if(kind==3)router[row*256+e]=.5f;
    if(kind==5 || kind==8){int shift=kind==5?1:4;int d=(e-row*shift+256)%256;router[row*256+e]=d<8?.9f-d*.02f:.0001f+e*.000001f;}
    if(kind==10)router[row*256+e]=(e%13==0)?NAN:(e%17==0)?INFINITY:float(e%11)/12;
    if(kind==11)router[row*256+e]=0;
    if(kind==12)router[row*256+e]=NAN;
    bias[e]=(kind==2 || kind==10)?sinf(e*1.78f)*.01f:0;

   }
   if(kind==13) {
    // Mix membership counts across the N16/N32/N64 boundaries in one tile.
    std::fill(router.begin(),router.end(),.0001f);
    for(int row=0;row<n;++row) {
     const int ids[8]={0,row<17?1:64+row%32,row<33?2:128+row%32,
       row<16?3:160+row%32,row<32?4:192+row%32,8+row%8,16+row%16,32+row%32};
     for(int slot=0;slot<8;++slot)router[row*256+ids[slot]]=.9f-slot*.02f;
    }
   }
   parallel([&](int r){auto&a=data[r];input_fill<<<(n*H+255)/256,256,0,cudaStreamPerThread>>>(a.x,n*H,seed,scale);check(cudaMemcpyAsync(a.router,router.data(),n*256*4,cudaMemcpyHostToDevice,cudaStreamPerThread));check(cudaMemcpyAsync(a.bias,bias.data(),256*4,cudaMemcpyHostToDevice,cudaStreamPerThread));for(int j=0;j<2;++j){check(cudaMemsetAsync(a.y[j],0x55,(MaxRows+1)*H*2,cudaStreamPerThread));check(cudaMemsetAsync(a.ids[j],0x55,MaxRows*8*4,cudaStreamPerThread));check(cudaMemsetAsync(a.p[j],0x55,MaxRows*8*4,cudaStreamPerThread));}check(cudaStreamSynchronize(cudaStreamPerThread));});
  };
  auto run=[&](int j,int r){auto&a=data[r];
   if(n>8 && n<=16){
    FastllmMoePreparedInput prep{};
    bool available=fastshared_tp_prepared_input(states[j],r,n,&prep);
    require(available==(!borrowed && !(mixed && j==1)), "wide prepared map admission mismatch");
    if(preparedMode && j==1){
     require(available && prep.Valid(n),"wide prepared descriptor");
     auto*state=static_cast<fastllm_shared_flex_tp::State*>(states[j]);
     auto rank=wide_rank(state,state->layers[0][r]);rank.rows=n;
     fastllm_shared_flex_tp::prepare_block128<16><<<512,128,0,cudaStreamPerThread>>>(rank,a.x);
     return fastshared_tp_run_prepared(states[j],r,0,n,a.router,a.bias,a.ids[j],a.p[j],a.y[j],needNorm,routeScale);
    }
   }
   if((j==1 || preparedMode) && n>1)return fastshared_tp_run_router(states[j],r,0,n,a.x,a.router,a.bias,a.ids[j],a.p[j],a.y[j],needNorm,routeScale);
   for(int offset=0;offset<n;offset+=8){
    const int rows=std::min(8,n-offset);
    if(rows==1){
     SelectReference<<<1,32,0,cudaStreamPerThread>>>(a.router+offset*256,a.bias,a.ids[j]+offset*8,a.p[j]+offset*8,needNorm,routeScale);
     int rc=fastshared_tp_run(states[j],r,0,1,a.x+offset*H,a.ids[j]+offset*8,a.p[j]+offset*8,a.y[j]+offset*H);if(rc)return rc;
    }else{
     FastllmMoePreparedInput prep{};
     require(fastshared_tp_prepared_input(states[j],r,rows,&prep)&&prep.Valid(rows),"prepared chunk descriptor");
     test_quantize<<<rows*16,256,0,cudaStreamPerThread>>>(a.x+offset*H,rows,prep);
     if(prep.block128)test_coarse<<<256,128,0,cudaStreamPerThread>>>(prep);
     int rc=fastshared_tp_run_prepared(states[j],r,0,rows,a.router+offset*256,a.bias,a.ids[j]+offset*8,a.p[j]+offset*8,a.y[j]+offset*H,needNorm,routeScale);if(rc)return rc;
    }
   }
   return 0;
  };
  fill(0,123,1);
  for(int j=0;j<2;++j){parallel([&](int r){require(!run(j,r),"warmup");check(cudaStreamSynchronize(cudaStreamPerThread));});
   parallel([&](int r){cudaGraph_t g;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));require(!run(j,r),"capture run");check(cudaStreamEndCapture(cudaStreamPerThread,&g));check(cudaGraphInstantiate(&one[j][r],g,0));check(cudaGraphDestroy(g));});
  }
  auto launch=[&](int j){parallel([&](int r){check(cudaGraphLaunch(one[j][r],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));});};
  double maxL2=0;size_t different=0;
  auto verify=[&]{std::array<std::vector<unsigned short>,8> refs,got;
   parallel([&](int r){refs[r].resize((MaxRows+1)*H);got[r].resize((MaxRows+1)*H);check(cudaMemcpy(refs[r].data(),data[r].y[0],refs[r].size()*2,cudaMemcpyDeviceToHost));check(cudaMemcpy(got[r].data(),data[r].y[1],got[r].size()*2,cudaMemcpyDeviceToHost));});
   for(int r=0;r<8;++r){double worst=0;
   for(int row=0;row<n;++row){double diff=0,mag=0;
    for(int j=0;j<H;++j){int i=row*H+j;double a=__bfloat162float(__ushort_as_bfloat16(refs[r][i])),b=__bfloat162float(__ushort_as_bfloat16(got[r][i]));diff+=(a-b)*(a-b);mag+=a*a;}
    worst=std::max(worst,std::sqrt(diff/std::max(mag,1e-30)));}
   if(worst>.003){printf("L2FAIL rows=%d rank=%d worst=%g\n",n,r,worst);fflush(stdout);}
   maxL2=std::max(maxL2,worst);require(worst<.003,"row relative L2 > 0.3%");
   if(preparedMode)require(refs[r]==got[r],"prepared MoE output/guards not bitwise exact");
   for(int i=0;i<n*H;++i)different+=refs[r][i]!=got[r][i];
   std::vector<int>ia(MaxRows*8),ib(MaxRows*8);std::vector<float>pa(MaxRows*8),pb(MaxRows*8);
   check(cudaSetDevice(r));
   check(cudaMemcpy(ia.data(),data[r].ids[0],MaxRows*8*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(ib.data(),data[r].ids[1],MaxRows*8*4,cudaMemcpyDeviceToHost));
   check(cudaMemcpy(pa.data(),data[r].p[0],MaxRows*8*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(pb.data(),data[r].p[1],MaxRows*8*4,cudaMemcpyDeviceToHost));
   require(ia==ib && !memcmp(pa.data(),pb.data(),MaxRows*8*4),"IDs/probabilities or guards differ");for(int i=0;i<n*H;++i){require(got[r][i]==got[0][i],"rank mismatch");require(std::isfinite(__bfloat162float(__ushort_as_bfloat16(got[r][i]))),"nonfinite output");}for(int i=n*H;i<(MaxRows+1)*H;++i)require(got[r][i]==0x5555,"output guard changed");}
  };
  for(int kind:{0,1,2,3,5,8,10,11,12,13}) {
   for(float scale:{0.f,1.f}){fill(kind,919+n*3+kind,scale);launch(0);launch(1);verify();}

  }

  printf("CORRECT rows=%d max_row_l2=%.9g differences=%zu IDs_prob_guard_exact=1\n",n,maxL2,different);fflush(stdout);
  for(int j=0;j<2;++j)parallel([&](int r){check(cudaGraphExecDestroy(one[j][r]));});
 }
 if(mixed)candidateState->layers[0][7].gate128map=savedMap;
 parallel([&](int r){auto&a=data[r];for(auto*p:a.w)check(cudaFree(p));check(cudaFree(a.x));for(int j=0;j<2;++j){check(cudaFree(a.ids[j]));check(cudaFree(a.p[j]));}check(cudaFree(a.router));check(cudaFree(a.bias));for(auto*y:a.y)check(cudaFree(y));});
 puts("PASS: routed MoE, 8 ranks, 1..129 rows and chunk boundaries, shared/disjoint/random routes, dynamic graph inputs and guards");return 0;
 } catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
