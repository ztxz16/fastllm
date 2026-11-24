// FastLLM native SM103 FlexMoE. See docs/naive-moe-backends.md for ownership,
// admission, source provenance, and the preserved numerical limitations.
// EP8 borrows the unmodified packed expert weights from MegaMoE.
#include <cuda.h>
#include <deep_gemm/mma/sm100.cuh>
#include "fastllm-shared-flex-weight.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "fastllm-cuda-launch.cuh"
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
constexpr int G_SPLIT=8,D_SPLIT=8,STAGES=4,EP_SMS=148,GRID=3*EP_SMS;
struct Rank {
 bool mxfp4 = false;
 const __nv_bfloat16*raw_x;int device,rank;int *uniq,*members,*rowcounts,*nuniq,*gcount,*jobs,*count;
 CUtensorMap *gmap,*dmap;float *gpartial,*dpartial;
 const uint8_t *gate,*down,*gs,*ds;uint8_t *xq,*xs,*mid,*ms;
 uint64_t *epoch,*box,*peers[R];
};
// One 128-wide K page supplies the eight activation rows and the replicated
// scale bytes expected by the block-scaled UMMA instruction.
struct VerifyOperand {
 static_assert(MAX_M == 8 && K == 8);
 __host__ __device__ static bool supports(bool mxfp4,int rows) {
  return mxfp4 && rows == MAX_M;
 }
 static constexpr int Width = 128, Values = MAX_M * Width, Scales = 512;
 static constexpr int GateSplit = 2, DownSplit = 2, Stages = 2;
 static constexpr int ReadyOffset=(E/R)*(2*I/128);
 static constexpr int ReadyEnd=ReadyOffset+(E/R);
 static constexpr int CounterCount=MAX_M*K*(2*I/128);
 static_assert(ReadyEnd <= CounterCount);
 static constexpr int GateBytes = H / Width * (Values + Scales);
 static constexpr int DownBytes = I / Width * (Values + Scales);
 static constexpr size_t ScratchFloats = MAX_M * K * GateSplit * (2 * I);
 static constexpr size_t Bytes = GateBytes + (E / R) * DownBytes;
 static_assert(ScratchFloats * sizeof(float) + Bytes <= MAX_M * K * G_SPLIT * (2 * I) * sizeof(float));

 __device__ static uint8_t *base(Rank a) {
  // The verification Gate writes only GateSplit partials per route.
  return reinterpret_cast<uint8_t *>(a.gpartial + ScratchFloats);
 }
 __device__ static uint8_t *down(Rank a,int expert) {
  return base(a) + GateBytes + expert * DownBytes;
 }
 template<int NK> __device__ static void write(uint8_t *operand,int row,int k,uint8_t value,unsigned scale) {
  int page = k / Width, lane = k % 32;
  operand[page * Values + row * Width + ((k % Width) ^ (row * 16))] = value;
  if (lane < 16) {
   uint8_t *sf = operand + NK / Width * Values + page * Scales;
   sf[4 * ((row + 8 * (lane % 4)) * 4 + lane / 4) + k / 32 % 4] = scale;
  }
 }
};
struct State {
 bool pairedGateSupported = true;
 Rank r[R];
 std::vector<std::array<Rank,R>> layers;
};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void* alloc(size_t n){void*p=nullptr;check(cudaMalloc(&p,n));return p;}
__device__ __forceinline__ unsigned exponent(float a){return max((__float_as_uint(a)+0x7fffffu-0x600000u)>>23,113u)-8;}
__device__ __forceinline__ float sf(unsigned e){return __uint_as_float(e<<23);}
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
   if(first){int item=prefix-1;a.uniq[item]=j;
    if (VerifyOperand::supports(a.mxfp4,rows)) {
     // Fixed N8 columns correspond to original token rows in both projections.
     for (int row = 0; row < MAX_M; ++row) {
      unsigned slots = unsigned(mask >> (row * K)) & ((1u << K) - 1);
      a.members[item * MAX_M + row] = slots ? row * K + __ffs(slots) - 1 : -1;
     }
    } else {
     int n=0;
     while(mask){a.members[item*MAX_M+n++]=__ffsll(mask)-1;mask&=mask-1;}
     a.rowcounts[item]=n;
    }
   }
  }
 }
 if(t>=rows*H)return;
 float v=__bfloat162float(x[t]),mx=fabsf(v);
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 unsigned e=exponent(mx);
 uint8_t q=__nv_fp8_e4m3(v*sf(254-e)).__x;
 if (VerifyOperand::supports(a.mxfp4,rows)) {
  VerifyOperand::write<H>(VerifyOperand::base(a),t/H,t%H,q,e);
 } else {
  a.xq[t]=q;
  if(lane==0)a.xs[t/32]=e;
 }
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
extern "C" void fastshared_ep_destroy(void*);
extern "C" int fastshared_ep_create(const int*devices,int layers,void**result){
 using namespace fastllm_shared_flex_ep;if(!devices||!result||layers<=0)return 1;*result=nullptr;
  if(!fastshared_ep_supported(devices,8))return 1;
  State*s=nullptr;
 try{
  s=new State{};s->layers.resize(layers);
  for(int r=0;r<R;++r){check(cudaSetDevice(devices[r]));auto&a=s->r[r];a.device=devices[r];a.rank=r;
   for(int q=0;q<R;++q)if(q!=r){auto e=cudaDeviceEnablePeerAccess(devices[q],0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e);}
   cudaFuncAttributes attrs{};
   int clusterLaunch=0;
   if(cudaDeviceGetAttribute(&clusterLaunch,cudaDevAttrClusterLaunch,devices[r])!=cudaSuccess ||
      !clusterLaunch || cudaFuncGetAttributes(&attrs,paired_gate_rows8)!=cudaSuccess){
    s->pairedGateSupported=false; cudaGetLastError();
   }
   constexpr int gb=STAGES,db=STAGES;
   check(cudaFuncSetAttribute(single_execute,cudaFuncAttributeMaxDynamicSharedMemorySize,gb*18432));
   check(cudaFuncSetAttribute(execute<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,gb*18432));
   check(cudaFuncSetAttribute(execute<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,db*18432));
   a.uniq=(int*)alloc(32*4);a.members=(int*)alloc(32*8*4);a.rowcounts=(int*)alloc(32*4);a.nuniq=(int*)alloc(4);
   a.gcount=(int*)alloc(VerifyOperand::CounterCount*sizeof(int));check(cudaMemset(a.gcount,0,VerifyOperand::CounterCount*sizeof(int)));a.jobs=(int*)alloc(MAX_M*K*4);a.count=(int*)alloc(4);
   a.gpartial=(float*)alloc(MAX_M*K*G_SPLIT*4096*4);a.dpartial=(float*)alloc(MAX_M*K*D_SPLIT*4096*4);
   // Inactive columns of a Down tile may still be fetched by TMA.
   check(cudaMemset(reinterpret_cast<uint8_t *>(a.gpartial + VerifyOperand::ScratchFloats),0,VerifyOperand::Bytes));
   a.xq=(uint8_t*)alloc(MAX_M*H);a.xs=(uint8_t*)alloc(MAX_M*H/32);
   a.mid=(uint8_t*)alloc(MAX_M*K*I);a.ms=(uint8_t*)alloc(MAX_M*K*I/32);
   a.epoch=(uint64_t*)alloc(8);check(cudaMemset(a.epoch,0,8));
   a.box=(uint64_t*)alloc(2*MAX_M*R*(H/2)*8);check(cudaMemset(a.box,0,2*MAX_M*R*(H/2)*8));
  }
  for(int r=0;r<R;++r)for(int q=0;q<R;++q)s->r[r].peers[q]=s->r[q].box;
  *result=s;return 0;
 }catch(const std::exception&e){fastshared_ep_destroy(s);fprintf(stderr,"EP create: %s\n",e.what());return 1;}
}
extern "C" int fastshared_ep_bind_format(void *state, int rank, int layer, const void *gate,
        const void *down, const void *gs, const void *ds, bool mxfp4) {
    using namespace fastllm_shared_flex_ep;
    auto *s = static_cast<State *>(state);
    if (!s || rank < 0 || rank >= R || layer < 0 || layer >= (int)s->layers.size() ||
        !gate || !down || !gs || !ds) return 1;
    auto &a = s->layers[layer][rank];
    if (a.gate) return 1;
    a = s->r[rank];
    a.gate = static_cast<const uint8_t *>(gate);
    a.down = static_cast<const uint8_t *>(down);
    a.gs = static_cast<const uint8_t *>(gs);
    a.ds = static_cast<const uint8_t *>(ds);
    a.mxfp4 = mxfp4;
    try {
        check(cudaSetDevice(a.device));
        for (int downProjection = 0; downProjection < 2; ++downProjection) {
            CUtensorMap map;
            if (fastllm_shared_flex::weight_map(&map, downProjection ? down : gate,
                    downProjection ? I : H, downProjection ? H : 2 * I, E/R, mxfp4) != CUDA_SUCCESS)
                return 1;
            auto &dst = downProjection ? a.dmap : a.gmap;
            dst = static_cast<CUtensorMap *>(alloc(sizeof(map)));
            check(cudaMemcpy(dst, &map, sizeof(map), cudaMemcpyHostToDevice));
        }
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "Shared Flex EP bind: %s\n", e.what()); return 1; }
}
extern "C" int fastshared_ep_bind(void *state, int rank, int layer, const void *gate,
        const void *down, const void *gs, const void *ds) {
    return fastshared_ep_bind_format(state, rank, layer, gate, down, gs, ds, false);
}
extern "C" int fastshared_ep_run(void*state,int rank,int layer,int rows,const void*x,const void*ids,const void*prob,void*y){
 using namespace fastllm_shared_flex_ep;
 auto *s=static_cast<State*>(state);
 if(!s||rank<0||rank>=R||rows<0)return 1;
 if(!rows)return 0;
 if(layer<0||layer>=int(s->layers.size())||!x||!ids||!prob||!y)return 1;
 const auto &a=s->layers[layer][rank];
 if(!a.gate||!a.down)return 1;
 auto stream=cudaStreamPerThread;
 constexpr int gb=STAGES,db=STAGES;
 for(size_t off=0;off<(size_t)rows;off+=8){int n=rows-off<8?rows-off:8;
  auto xx=(const __nv_bfloat16*)x+off*H;auto ii=(const int*)ids+off*K;auto pp=(const float*)prob+off*K;auto yy=(__nv_bfloat16*)y+off*H;
  if(n>1)prepare<<<n*16,256,0,stream>>>(a,xx,ii,n);
  if (VerifyOperand::supports(a.mxfp4,n)) {
   constexpr int blocks = 8 * EP_SMS, smem = VerifyOperand::Stages * 18432;
   if(s->pairedGateSupported){
    paired_gate_rows8<<<blocks,128,smem,stream>>>(a,pp);
    auto rc=fastllm::cuda_launch::LaunchProgrammatic<grouped_mxfp4_rows8<false,true>>(
        dim3(blocks),dim3(128),smem,stream,a,pp);
    if(rc!=cudaSuccess)return rc;
   }else{
    grouped_mxfp4_rows8<true><<<blocks,128,smem,stream>>>(a,pp);
    grouped_mxfp4_rows8<false><<<blocks,128,smem,stream>>>(a,pp);
   }
   auto rc=fastllm::cuda_launch::LaunchProgrammatic<finish_rows8>(dim3(16,n),dim3(128),0,stream,a,yy,ii);
   if (rc!=cudaSuccess) return rc;
   continue;
  }
  if (n == 1) {
   auto single=a;
   single.raw_x=xx;
   single_execute<<<GRID,128,gb*18432,stream>>>(single,ii,pp);
  } else {
   execute<true><<<GRID,128,gb*18432,stream>>>(a,ii,pp);
  }
  execute<false><<<GRID,128,db*18432,stream>>>(a,ii,pp);
  finish<<<dim3(16,n),128,0,stream>>>(a,yy,ii,n);
 }return cudaGetLastError();
}
extern "C" size_t fastshared_ep_workspace_bytes(){using namespace fastllm_shared_flex_ep;return 32*4+32*8*4+32*4+4+VerifyOperand::CounterCount*sizeof(int)+MAX_M*K*4+4+MAX_M*K*(G_SPLIT+D_SPLIT)*4096*4+MAX_M*H+MAX_M*H/32+MAX_M*K*I+MAX_M*K*I/32+8+2*MAX_M*R*(H/2)*8;}
extern "C" void fastshared_ep_destroy(void*state){auto*s=(fastllm_shared_flex_ep::State*)state;if(!s)return;
 for(auto &ls:s->layers)for(auto &a:ls)if(a.gate){cudaSetDevice(a.device);cudaDeviceSynchronize();cudaFree(a.gmap);cudaFree(a.dmap);}
 for(auto&a:s->r){cudaSetDevice(a.device);cudaDeviceSynchronize();
  for(void*p:{(void*)a.uniq,(void*)a.members,(void*)a.rowcounts,(void*)a.nuniq,(void*)a.gcount,(void*)a.jobs,(void*)a.count,(void*)a.gpartial,(void*)a.dpartial,(void*)a.xq,(void*)a.xs,(void*)a.mid,(void*)a.ms,(void*)a.epoch,(void*)a.box})cudaFree(p);
 }delete s;
}

extern "C" int fastshared_ep_supported(const int*devices,int ranks) {
 using namespace fastllm_shared_flex_ep;
 if(!devices||ranks!=8)return 0;
 int previous=0; if(cudaGetDevice(&previous)!=cudaSuccess)return 0;
 bool ok=true;
 for(int r=0;r<ranks&&ok;++r){
  cudaDeviceProp prop{};ok=cudaGetDeviceProperties(&prop,devices[r])==cudaSuccess &&
   prop.major==10 && prop.minor==3 && prop.multiProcessorCount==EP_SMS && prop.sharedMemPerBlockOptin>=73728;
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
