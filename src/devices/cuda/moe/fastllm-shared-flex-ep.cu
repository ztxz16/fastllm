// FastLLM native SM103 FlexMoE. See docs/naive-moe-backends.md for ownership,
// admission, source provenance, and the preserved numerical limitations.
// SM103 EP8 experiment. Borrows unmodified MegaMoE packed weights.
#include <cuda.h>
#include <deep_gemm/mma/sm100.cuh>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <array>
namespace fastllm_shared_flex_ep {
#ifndef FM_RELAXED_MAILBOX
#define FM_RELAXED_MAILBOX 0
#endif
constexpr int H=4096,I=2048,E=256,R=8,K=8,MAX_M=8;
constexpr int G_SPLIT=8,D_SPLIT=8,USE_BATCH=1,STAGES=4,GRID=444;
struct Rank {
 const __nv_bfloat16*raw_x;int rows,device,rank;int *uniq,*members,*rowcounts,*nuniq,*gcount,*jobs,*count;
 CUtensorMap *gmap,*dmap;float *gpartial,*dpartial;
 const uint8_t *gate,*down,*gs,*ds;uint8_t *xq,*xs,*mid,*ms;
 uint64_t *epoch,*box,*peers[R];
};
struct State { Rank r[R]; std::vector<std::array<Rank,R>> layers; std::vector<void*> owned[R]; };
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void* alloc(size_t n){void*p=nullptr;check(cudaMalloc(&p,n));return p;}
__device__ __forceinline__ unsigned exponent(float a){return max((__float_as_uint(a)+0x7fffffu-0x600000u)>>23,113u)-8;}
__device__ __forceinline__ float sf(unsigned e){return __uint_as_float(e<<23);}
__device__ __forceinline__ size_t sf_offset(int expert,int row,int kb,int n,int k){
  int t=(row&~127)+(row%32)*4+(row%128)/32;
  return (size_t)expert*n*(k/32)+((size_t)(kb/4)*n+t)*4+kb%4;
}
__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,
   unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ unsigned ld4(const uint8_t*p){return *reinterpret_cast<const unsigned*>(p);}
__global__ void prepare(Rank a,const __nv_bfloat16*x,const int*ids,int rows){
 int t=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
 if(blockIdx.x==0){
  __shared__ int totals[2];int j=threadIdx.x;
  int mine=j<rows*K&&ids[j]/32==a.rank,prefix=mine;
  if(j==0)++*a.epoch;
  if(j<64){
   #pragma unroll
   for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,prefix,d);if(lane>=d)prefix+=v;}
   if(lane==31)totals[j/32]=prefix;
  }
  __syncthreads();
  if(j==0)*a.count=totals[0]+totals[1];
  if(mine)a.jobs[prefix-1+(j>=32?totals[0]:0)]=j;
 }
 if(t<rows*K*32)a.gcount[t]=0;
 if(blockIdx.x==0){
  __shared__ unsigned long long masks[32];int j=threadIdx.x;
  if(j<32)masks[j]=0;__syncthreads();
  if(j<rows*K&&ids[j]/32==a.rank)atomicOr(masks+ids[j]%32,1ull<<j);
  __syncthreads();
  if(j<32){unsigned long long mask=masks[j];int first=mask!=0,prefix=first;
   #pragma unroll
   for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,prefix,d);if(lane>=d)prefix+=v;}
   if(j==31)*a.nuniq=prefix;
   if(first){int item=prefix-1,n=0;a.uniq[item]=j;
    while(mask){a.members[item*8+n++]=__ffsll(mask)-1;mask&=mask-1;}
    a.rowcounts[item]=n;}
  }
 }
 if(t>=rows*H)return;
 float v=__bfloat162float(x[t]),mx=fabsf(v);
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 unsigned e=exponent(mx);a.xq[t]=__nv_fp8_e4m3(v*sf(254-e)).__x;
 if(lane==0)a.xs[t/32]=e;
}
__device__ __forceinline__ void store_release(uint64_t*p,uint64_t v){
#if FM_RELAXED_MAILBOX
  // Tag and payload are ONE aligned 64-bit transaction, as in NaiveRT's
  // tagged-word mailbox. There is no separate payload to order with release.
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;"::"l"(p),"l"(v):"memory");
#else
  asm volatile("st.global.release.sys.u64 [%0], %1;"::"l"(p),"l"(v):"memory");
#endif
}
__device__ __forceinline__ uint64_t load_acquire(const uint64_t*p){
  uint64_t v;
#if FM_RELAXED_MAILBOX
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];":"=l"(v):"l"(p):"memory");
#else
  asm volatile("ld.global.acquire.sys.u64 %0, [%1];":"=l"(v):"l"(p):"memory");
#endif
  return v;
}
}
#include "fastllm-shared-flex-ep-sm103.cuh"
extern "C" int fastshared_ep_supported(const int*,int);
extern "C" void fastshared_ep_destroy(void*state);
extern "C" int fastshared_ep_create(const int*devices,int layers,void**result){
 using namespace fastllm_shared_flex_ep;if(!devices||!result||layers<=0)return 1;*result=nullptr;
  if(!fastshared_ep_supported(devices,8))return 1;
  State*s=nullptr;
 try{
  for(int r=0;r<R;++r){cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,devices[r]));
   if(prop.major!=10||prop.minor!=3)throw std::runtime_error("SM103 required");
   for(int q=0;q<R;++q)if(q!=r){int ok;check(cudaDeviceCanAccessPeer(&ok,devices[r],devices[q]));if(!ok)throw std::runtime_error("Peer access required");}}
  s=new State{};s->layers.resize(layers);
  for(int r=0;r<R;++r){check(cudaSetDevice(devices[r]));auto&a=s->r[r];a.device=devices[r];a.rank=r;
   for(int q=0;q<R;++q)if(q!=r){auto e=cudaDeviceEnablePeerAccess(devices[q],0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e);}
   constexpr int gb=STAGES,db=STAGES;
   check(cudaFuncSetAttribute(single_execute,cudaFuncAttributeMaxDynamicSharedMemorySize,gb*18432));
   check(cudaFuncSetAttribute(execute<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,gb*18432));
   check(cudaFuncSetAttribute(execute<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,db*18432));
   a.uniq=(int*)alloc(32*4);a.members=(int*)alloc(32*8*4);a.rowcounts=(int*)alloc(32*4);a.nuniq=(int*)alloc(4);
   a.gcount=(int*)alloc(MAX_M*K*32*4);check(cudaMemset(a.gcount,0,MAX_M*K*32*4));a.jobs=(int*)alloc(MAX_M*K*4);a.count=(int*)alloc(4);
   a.gmap=(CUtensorMap*)alloc(128);a.dmap=(CUtensorMap*)alloc(128);
   a.gpartial=(float*)alloc(MAX_M*K*G_SPLIT*4096*4);a.dpartial=(float*)alloc(MAX_M*K*D_SPLIT*4096*4);
   a.xq=(uint8_t*)alloc(MAX_M*H);a.xs=(uint8_t*)alloc(MAX_M*H/32);
   a.mid=(uint8_t*)alloc(MAX_M*K*I);a.ms=(uint8_t*)alloc(MAX_M*K*I/32);
   a.epoch=(uint64_t*)alloc(8);check(cudaMemset(a.epoch,0,8));
   a.box=(uint64_t*)alloc(2*MAX_M*R*(H/2)*8);check(cudaMemset(a.box,0,2*MAX_M*R*(H/2)*8));
  }
  for(int r=0;r<R;++r)for(int q=0;q<R;++q)s->r[r].peers[q]=s->r[q].box;
  *result=s;return 0;
 }catch(const std::exception&e){fastshared_ep_destroy(s);fprintf(stderr,"EP create: %s\n",e.what());return 1;}
}
extern "C" int fastshared_ep_bind(void*state,int rank,int layer,const void*gate,const void*down,const void*gs,const void*ds){
 try {

 if(!state||rank<0||rank>=8||!gate||!down||!gs||!ds)return 1;
 auto*s=(fastllm_shared_flex_ep::State*)state;if(layer<0||layer>=(int)s->layers.size())return 1; auto&a=s->layers[layer][rank];if(a.gate)return 1;a=s->r[rank];a.gmap=nullptr;a.dmap=nullptr;a.gate=(const uint8_t*)gate;cudaSetDevice(a.device);a.gmap=(CUtensorMap*)fastllm_shared_flex_ep::alloc(128);a.dmap=(CUtensorMap*)fastllm_shared_flex_ep::alloc(128);a.gate=(const uint8_t*)gate;a.down=(const uint8_t*)down;a.gs=(const uint8_t*)gs;a.ds=(const uint8_t*)ds;
 if(cudaSetDevice(a.device)!=cudaSuccess)return 1;
 for(int which=0;which<2;++which){CUtensorMap map;int k=which?2048:4096;
  cuuint64_t dims[]={(uint64_t)k,4096,32},strides[]={(uint64_t)k,(uint64_t)4096*k};cuuint32_t box[]={128,128,1},steps[]={1,1,1};
  if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,(void*)(which?down:gate),dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)return 1;
  if(cudaMemcpy(which?a.dmap:a.gmap,&map,128,cudaMemcpyHostToDevice)!=cudaSuccess)return 1;
 }return 0;

 } catch (const std::exception &e) { fprintf(stderr,"Shared Flex bind: %s\n",e.what());return 1; }
}
extern "C" int fastshared_ep_run(void*state,int rank,int layer,int rows,const void*x,const void*ids,const void*prob,void*y){
 using namespace fastllm_shared_flex_ep;
 if(!state||rank<0||rank>=8||rows<0)return 1;if(!rows)return 0;if(!x||!ids||!prob||!y)return 1;
 auto*s=(State*)state;if(layer<0||layer>=(int)s->layers.size())return 1;auto a=s->layers[layer][rank];if(!a.gate||!a.down)return 1;auto stream=cudaStreamPerThread;
 constexpr int gb=STAGES,db=STAGES;
 for(size_t off=0;off<(size_t)rows;off+=8){int n=rows-off<8?rows-off:8;a.rows=n;
  auto xx=(const __nv_bfloat16*)x+off*H;auto ii=(const int*)ids+off*K;auto pp=(const float*)prob+off*K;auto yy=(__nv_bfloat16*)y+off*H;
  if(n==1){a.raw_x=xx;single_execute<<<GRID,128,gb*18432,stream>>>(a,ii,pp);}
  else{prepare<<<n*16,256,0,stream>>>(a,xx,ii,n);
       execute<true><<<GRID,128,gb*18432,stream>>>(a,ii,pp);}
  execute<false><<<GRID,128,db*18432,stream>>>(a,ii,pp);
  finish<<<dim3(16,n),128,0,stream>>>(a,yy,ii,n);
 }return cudaGetLastError();
}
extern "C" size_t fastshared_ep_workspace_bytes(){using namespace fastllm_shared_flex_ep;return 32*4+32*8*4+32*4+4+MAX_M*K*32*4+MAX_M*K*4+4+256+MAX_M*K*(G_SPLIT+D_SPLIT)*4096*4+MAX_M*H+MAX_M*H/32+MAX_M*K*I+MAX_M*K*I/32+8+2*MAX_M*R*(H/2)*8;}
extern "C" void fastshared_ep_destroy(void*state){auto*s=(fastllm_shared_flex_ep::State*)state;if(!s)return;
 for(auto &ls:s->layers)for(auto &a:ls)if(a.gate){cudaSetDevice(a.device);cudaDeviceSynchronize();cudaFree(a.gmap);cudaFree(a.dmap);}
 for(int r=0;r<8;++r){cudaSetDevice(s->r[r].device);for(void*p:s->owned[r])cudaFree(p);}
 for(auto&a:s->r){cudaSetDevice(a.device);cudaDeviceSynchronize();
  for(void*p:{(void*)a.uniq,(void*)a.members,(void*)a.rowcounts,(void*)a.nuniq,(void*)a.gcount,(void*)a.jobs,(void*)a.count,(void*)a.gmap,(void*)a.dmap,(void*)a.gpartial,(void*)a.dpartial,(void*)a.xq,(void*)a.xs,(void*)a.mid,(void*)a.ms,(void*)a.epoch,(void*)a.box})cudaFree(p);
 }delete s;
}

extern "C" int fastshared_ep_supported(const int*devices,int ranks) {
 using namespace fastllm_shared_flex_ep;
 if(!devices||ranks!=8)return 0;
 int previous=0; if(cudaGetDevice(&previous)!=cudaSuccess)return 0;
 bool ok=true;
 for(int r=0;r<ranks&&ok;++r){
  cudaDeviceProp prop{};ok=cudaGetDeviceProperties(&prop,devices[r])==cudaSuccess &&
   prop.major==10 && prop.minor==3 && prop.multiProcessorCount==148 && prop.sharedMemPerBlockOptin>=73728;
  for(int q=0;q<ranks&&ok;++q)if(q!=r){int access=0;ok=devices[q]!=devices[r]&&cudaDeviceCanAccessPeer(&access,devices[r],devices[q])==cudaSuccess&&access;}
  if(!ok||cudaSetDevice(devices[r])!=cudaSuccess){ok=false;break;}
  cudaFuncAttributes attributes{};
  ok=ok&&cudaFuncGetAttributes(&attributes,single_execute)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,execute<true>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,execute<false>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,finish)==cudaSuccess;
 }
 cudaGetLastError();cudaSetDevice(previous);return ok?1:0;
}
