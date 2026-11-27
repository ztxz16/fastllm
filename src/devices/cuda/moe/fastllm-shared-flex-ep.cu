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
#include "fastllm-norm-rows-router.cuh"
#include "fastllm-shared-flex-router-sm103.cuh"
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
  return mxfp4 && rows >= 2 && rows <= MAX_M;
 }
 static constexpr int Width = 128, Values = MAX_M * Width, Scales = 512;
 static constexpr int GateSplit = 2, DownSplit = 2, Stages = 2;
 static constexpr int ReadyOffset=(E/R)*(2*I/128);
 static constexpr int ReadyEnd=ReadyOffset+(E/R);
 static constexpr int CounterCount=MAX_M*K*(2*I/128), RouterArrival=CounterCount-1;
 static_assert(ReadyEnd <= RouterArrival); // Keep the router CTA election separate.
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
// Round up the active memberships of each local expert to complete N8 jobs.
struct WideOperand {
 static constexpr int Rows = 32, Groups = 2 * (E/R);
 static_assert((Rows*K + (E/R)*(MAX_M-1))/MAX_M <= Groups);
 __host__ __device__ static int tiles(int rows) { return (rows + MAX_M - 1) / MAX_M; }
 static_assert(VerifyOperand::ReadyOffset + Groups <= VerifyOperand::RouterArrival);
 static_assert((Groups + Groups * MAX_M) * sizeof(int) <= MAX_M * K * I);
 static_assert(Groups * (VerifyOperand::GateBytes + VerifyOperand::DownBytes)
               <= MAX_M*K*G_SPLIT*(2*I)*sizeof(float));
 static_assert(Rows*K*VerifyOperand::DownSplit*H <= MAX_M*K*D_SPLIT*H);
 __device__ static uint8_t *gate(Rank a, int job) {
  return reinterpret_cast<uint8_t *>(a.gpartial) + job * VerifyOperand::GateBytes;
 }
 __device__ static uint8_t *down(Rank a, int job) {
  return gate(a, Groups) + job * VerifyOperand::DownBytes;
 }
};
struct State {
 bool pairedGateSupported = true, routerSupported = true, preparedRouterSupported = true;
 bool supportsWide(bool mxfp4, int rows) const {
  return pairedGateSupported && mxfp4 && rows > MAX_M && rows <= WideOperand::Rows;
 }
 Rank r[R];
 std::vector<std::array<Rank,R>> layers;
};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void* alloc(size_t n){void*p=nullptr;check(cudaMalloc(&p,n));return p;}
__device__ __forceinline__ unsigned exponent(float a){return max((__float_as_uint(a)+0x7fffffu-0x600000u)>>23,113u)-8;}
__device__ __forceinline__ float sf(unsigned e){return __uint_as_float(e<<23);}
__device__ __forceinline__ unsigned ld4(const uint8_t*p){return *reinterpret_cast<const unsigned*>(p);}
// Both EP routing entry points use the same shortlist and tie fallback.
__device__ __forceinline__ void selectRoutes(const float *router,const float *bias,int *ids,float *scores,
        int needNorm,float scale,fastllm::cuda::RouterTopKScratch<K> &scratch) {
 if (!fastllm_shared_flex_tp::SelectShortlist(router,bias,ids,scores,needNorm,scale,scratch))
  fastllm::cuda::SelectRouterTopK<E,K>(router,bias,ids,scores,bias!=nullptr,needNorm,scale,scratch);
}
struct EpVerifyRoutes {
 __device__ static void store(const FastllmMoePreparedInput &p,int row,int col,uint2 q,unsigned scale) {
  int page=col/VerifyOperand::Width, lane=(col%VerifyOperand::Width)/8;
  *reinterpret_cast<uint2*>(p.values+page*VerifyOperand::Values+row*VerifyOperand::Width+((col%VerifyOperand::Width)^(row*16)))=q;
  unsigned scales=0;
  #pragma unroll
  for(int g=0;g<4;++g) scales|=__shfl_sync(0xffffffffu,scale,4*g,16)<<(8*g);
  reinterpret_cast<unsigned*>(p.scales+page*VerifyOperand::Scales)[(row+8*(lane%4))*4+lane/4]=scales;
 }
 __device__ static void complete(const FastllmMoePreparedInput &p,const float *router) {
  // Only the last router CTA reads the full logits. All earlier CTAs return;
  // no resident block waits for an unscheduled producer.
  __shared__ unsigned last;
  __shared__ fastllm::cuda::RouterTopKScratch<K> scratch[MAX_M];
  __shared__ int selected[MAX_M*K];
  __shared__ unsigned owners[MAX_M];
  int t=threadIdx.x,row=t/32,lane=t%32;
  __syncthreads();
  if(t==0){
   unsigned previous,limit=gridDim.x*gridDim.y-1;
   asm volatile("atom.acq_rel.gpu.global.inc.u32 %0,[%1],%2;":"=r"(previous):"l"(p.counters+VerifyOperand::RouterArrival),"r"(limit):"memory");
   last=previous==limit;
  }
  __syncthreads();if(!last)return;
  int *ids=selected+row*K;
  int route=-1;
  if (row < p.rows) {
   selectRoutes(router+row*E,p.bias,ids,p.scores+row*K,p.needNorm,p.routeScale,scratch[row]);
   __syncwarp();
   if(lane<K)p.indices[row*K+lane]=ids[lane];
   #pragma unroll
   for(int slot=0;slot<K;++slot)if(ids[slot]==p.epRank*32+lane)route=row*K+slot;
  }
  unsigned present=__ballot_sync(0xffffffffu,route>=0);
  if(lane==0)owners[row]=present;
  for(int i=t;i<VerifyOperand::ReadyEnd;i+=256)p.counters[i]=0;
  __syncthreads();
  unsigned all=0;
  #pragma unroll
  for(int r=0;r<MAX_M;++r)all|=owners[r];
  if(all&(1u<<lane)){
   int item=__popc(all&((1u<<lane)-1));
   p.members[item*MAX_M+row]=route;
   if(row==0)p.experts[item]=lane;
  }
  if(t==0)*p.unique=__popc(all);
 }
};
struct RouteSelection {
 const float *router, *bias;
 int *ids;
 float *scores;
 int needNorm;
 float scale;
};
// Each warp selects every eighth row. One lane per local expert builds N8 jobs
// across the whole verification batch, preserving token and slot order.
template<bool Select>
__global__ void wideRoutes(Rank a, const int *ids, RouteSelection selection, int rows) {
 __shared__ fastllm::cuda::RouterTopKScratch<K> scratch[MAX_M];
 __shared__ int chosen[MAX_M*K], member[E/R][WideOperand::Rows];
 int t=threadIdx.x, warp=t/32, lane=t%32;
 if constexpr (Select) cudaGridDependencySynchronize();
 for (int row=warp;row<rows;row+=MAX_M) {
  const int *routes=ids+row*K;
  if constexpr (Select) {
   selectRoutes(selection.router+row*E,selection.bias,chosen+warp*K,selection.scores+row*K,
                selection.needNorm,selection.scale,scratch[warp]);
   __syncwarp();
   if(lane<K) selection.ids[row*K+lane]=chosen[warp*K+lane];
   routes=chosen+warp*K;
  }
  int route=-1;
  for(int slot=0;slot<K;++slot) if(routes[slot]==a.rank*(E/R)+lane) route=row*K+slot;
  member[lane][row]=route;
  __syncwarp();
 }
 if(t<WideOperand::Groups) a.gcount[VerifyOperand::ReadyOffset+t]=0;
 if(t==0) *a.epoch+=WideOperand::tiles(rows);
 __syncthreads();
 if(warp!=0) return;
 unsigned mask=0;
 for(int i=0;i<rows;++i) if(member[lane][i]>=0) mask|=1u<<i;
 int jobs=(__popc(mask)+MAX_M-1)/MAX_M, end=jobs;
 for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,end,d);if(lane>=d)end+=v;}
 if(lane==31) *a.nuniq=end;
 for(int job=end-jobs;job<end;++job) {
  a.uniq[job]=lane;
  for(int col=0;col<MAX_M;++col) {
   a.members[job*MAX_M+col]=mask?member[lane][__ffs(mask)-1]:-1;
   mask&=mask-1;
  }
 }
}
__global__ void packWideGate(Rank a,const __nv_bfloat16 *x) {
 const int job=blockIdx.x;
 if(job>=*a.nuniq)return;
 const int row=threadIdx.x/32, route=a.members[job*MAX_M+row];
 if(route<0)return;
 const int lane=threadIdx.x%32;
 for(int k=blockIdx.y*VerifyOperand::Width+lane;k<(blockIdx.y+1)*VerifyOperand::Width;k+=32) {
  float v=__bfloat162float(x[(route/K)*H+k]), m=fabsf(v);
  for(int d=16;d;d/=2)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,d));
  const unsigned e=exponent(m);
  VerifyOperand::write<H>(WideOperand::gate(a,job),row,k,__nv_fp8_e4m3(v*sf(254-e)).__x,e);
 }
}
template<bool Select = false>
__global__ void prepare(Rank a,const __nv_bfloat16*x,const int*ids,int rows, RouteSelection selection = {}){
 // Only CTA zero builds the EP schedule. Its warps select independent rows
 // into shared memory; other CTAs can quantize the activation concurrently.
 if constexpr (Select) {
  __shared__ fastllm::cuda::RouterTopKScratch<K> scratch[MAX_M];
  __shared__ int selected[MAX_M*K];
  __shared__ float scores[MAX_M*K];
  // Logits, bias and the normalized input may all be produced on this stream.
  cudaGridDependencySynchronize();
  if (blockIdx.x == 0) {
   int row = threadIdx.x / 32;
   if (row < rows) {
    auto *idsOut = selected + row*K;
    auto *scoresOut = scores + row*K;
    const float *router = selection.router + row*E;
    selectRoutes(router,selection.bias,idsOut,scoresOut,selection.needNorm,selection.scale,scratch[row]);
   }
   __syncthreads();
   if (threadIdx.x < rows*K) {
    selection.ids[threadIdx.x] = selected[threadIdx.x];
    selection.scores[threadIdx.x] = scores[threadIdx.x];
   }
   ids = selected;
  }
 }
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
 const int counters = VerifyOperand::supports(a.mxfp4,rows) ? VerifyOperand::ReadyEnd : rows*K*32;
 if(t<counters)a.gcount[t]=0;
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
   if (cudaFuncGetAttributes(&attrs, prepare<true>) != cudaSuccess ||
       attrs.maxThreadsPerBlock < 256 || attrs.ptxVersion < 90 || attrs.binaryVersion < 90) {
    s->routerSupported = false;
    cudaGetLastError();
   }
   int clusterLaunch=0;
   if(cudaDeviceGetAttribute(&clusterLaunch,cudaDevAttrClusterLaunch,devices[r])!=cudaSuccess ||
      !clusterLaunch || cudaFuncGetAttributes(&attrs,paired_gate_rows8)!=cudaSuccess){
    s->pairedGateSupported=false; cudaGetLastError();
   }
   using namespace fastllm_norm_rows;
   for(const void *kernel : {reinterpret_cast<const void*>(NormRowsRouterDirect<8,2,true,8,false,false,EpVerifyRoutes>),
                             reinterpret_cast<const void*>(NormRowsRouterDirect<8,2,true,8,false,true,EpVerifyRoutes>)})
    if(cudaFuncGetAttributes(&attrs,kernel)!=cudaSuccess || attrs.maxThreadsPerBlock<256 ||
       attrs.ptxVersion<90 || attrs.binaryVersion<90){
     s->preparedRouterSupported=false;cudaGetLastError();
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
static int runEp(void*state,int rank,int layer,int rows,const void*x,const void*ids,const void*prob,void*y,
                 fastllm_shared_flex_ep::RouteSelection selection,bool prepared=false) {
 using namespace fastllm_shared_flex_ep;
 auto *s=static_cast<State*>(state);
 if(!s||rank<0||rank>=R||rows<0)return 1;
 if(!rows)return 0;
 if(layer<0||layer>=int(s->layers.size())||(!prepared&&!x)||!ids||!prob||!y)return 1;
 const auto &a=s->layers[layer][rank];
 if(!a.gate||!a.down)return 1;
 if(selection.router && rows>MAX_M && !s->supportsWide(a.mxfp4,rows))return 1;
 if(prepared && !(s->routerSupported && s->preparedRouterSupported && VerifyOperand::supports(a.mxfp4,rows)))return 1;
 auto stream=cudaStreamPerThread;
 if(s->supportsWide(a.mxfp4,rows) && !prepared) {
  // Non-packed kernels use mid/ms for activations; the wide kernel stores
  // both operands in the otherwise unused Gate partial buffer.
  auto wide=a;
  wide.uniq=reinterpret_cast<int*>(a.mid);
  wide.members=wide.uniq+WideOperand::Groups;
  if(selection.router) {
   auto rc=fastllm::cuda_launch::LaunchProgrammatic<wideRoutes<true>>(dim3(1),dim3(256),0,stream,wide,
       static_cast<const int*>(ids),selection,rows);
   if(rc!=cudaSuccess)return rc;
  } else wideRoutes<false><<<1,256,0,stream>>>(wide,static_cast<const int*>(ids),selection,rows);
  packWideGate<<<dim3(WideOperand::Groups,H/VerifyOperand::Width),256,0,stream>>>(wide,static_cast<const __nv_bfloat16*>(x));
  constexpr int blocks=8*EP_SMS, smem=VerifyOperand::Stages*18432;
  wideGate<<<blocks,128,smem,stream>>>(wide,static_cast<const float*>(prob));
  auto rc=fastllm::cuda_launch::LaunchProgrammatic<wideDown>(dim3(blocks),dim3(128),smem,stream,wide,static_cast<const float*>(prob));
  if(rc!=cudaSuccess)return rc;
  return fastllm::cuda_launch::LaunchProgrammatic<wideFinish>(dim3(16,MAX_M),dim3(128),0,stream,wide,
       static_cast<__nv_bfloat16*>(y),static_cast<const int*>(ids),rows);
 }
 constexpr int gb=STAGES,db=STAGES;
 for(size_t off=0;off<(size_t)rows;off+=8){int n=rows-off<8?rows-off:8;
  auto xx=x?(const __nv_bfloat16*)x+off*H:nullptr;auto ii=(const int*)ids+off*K;auto pp=(const float*)prob+off*K;auto yy=(__nv_bfloat16*)y+off*H;
  if (n > 1 && !prepared) {
   if (selection.router) {
    RouteSelection tile{selection.router + off*E, selection.bias, selection.ids + off*K,
                        selection.scores + off*K, selection.needNorm, selection.scale};
    auto rc = fastllm::cuda_launch::LaunchProgrammatic<prepare<true>>(
        dim3(n*16), dim3(256), 0, stream, a, xx, ii, n, tile);
    if (rc != cudaSuccess) return rc;
   } else {
    prepare<false><<<n*16,256,0,stream>>>(a,xx,ii,n);
   }
  }
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
extern "C" int fastshared_ep_supports_rows(void *state, bool mxfp4, int rows) {
    using namespace fastllm_shared_flex_ep;
    const auto *s = static_cast<const State *>(state);
    return s && rows > 0 && (rows <= MAX_M || s->supportsWide(mxfp4, rows));
}
extern "C" int fastshared_ep_uses_verify_reduce(void *state, int rank, int layer, int rows) {
    using namespace fastllm_shared_flex_ep;
    const auto *s = static_cast<const State *>(state);
    if (!s || rank < 0 || rank >= R || layer < 0 || size_t(layer) >= s->layers.size()) return 0;
    const auto &weights = s->layers[layer][rank];
    return weights.gate && weights.mxfp4 && (VerifyOperand::supports(weights.mxfp4,rows) || s->supportsWide(weights.mxfp4,rows));
}
extern "C" int fastshared_ep_prepared_input(void *state,int rank,int layer,int rows,FastllmMoePreparedInput *p) {
 using namespace fastllm_shared_flex_ep;
 auto *s=static_cast<State*>(state);
 if(!p||!s||!s->routerSupported||!s->preparedRouterSupported||rank<0||rank>=R||layer<0||layer>=int(s->layers.size()))return 0;
 const auto &a=s->layers[layer][rank];
 if(!VerifyOperand::supports(a.mxfp4,rows))return 0;
 *p={}; p->rows=rows;p->epRank=rank;p->overlapVerifyGate=true;
 p->values=reinterpret_cast<uint8_t*>(a.gpartial+VerifyOperand::ScratchFloats);
 p->scales=p->values+MAX_M*H;p->counters=a.gcount;p->epoch=a.epoch;
 p->unique=a.nuniq;p->experts=a.uniq;p->members=a.members;
 return 1;
}
extern "C" int fastshared_ep_prepared_norm(const void *x,const float *weight,float *router,const float *gamma,
        void *normed,float eps,FastllmMoePreparedInput p,bool afterSwa) {
 using namespace fastllm_shared_flex_ep;
 using namespace fastllm_norm_rows;
 if(!p.Valid(p.rows)||p.epRank<0||!x||!weight||!router||!gamma||!normed||!std::isfinite(eps)||eps<=0||
    ((uintptr_t(x)|uintptr_t(weight)|uintptr_t(gamma)|uintptr_t(normed)|uintptr_t(router))&15u))
  return cudaErrorInvalidValue;
 if(afterSwa)return fastllm::cuda_launch::LaunchProgrammatic<NormRowsRouterDirect<8,2,true,8,false,true,EpVerifyRoutes>>(
    dim3(32,4),dim3(256),0,cudaStreamPerThread,static_cast<const BF16*>(x),weight,router,gamma,
    static_cast<BF16*>(normed),eps,p.rows,p,(const BF16*)nullptr,(BF16*)nullptr);
 NormRowsRouterDirect<8,2,true,8,false,false,EpVerifyRoutes><<<dim3(32,4),256,0,cudaStreamPerThread>>>(
    static_cast<const BF16*>(x),weight,router,gamma,static_cast<BF16*>(normed),eps,p.rows,p);
 return cudaGetLastError();
}
extern "C" int fastshared_ep_run_prepared_rows(void *state,int rank,int layer,int rows,const int *ids,const float *prob,void *y) {
 return runEp(state,rank,layer,rows,nullptr,ids,prob,y,{},true);
}
extern "C" int fastshared_ep_run_prepared(void *state,int rank,int layer,const int *ids,const float *prob,void *y) {
 return fastshared_ep_run_prepared_rows(state,rank,layer,8,ids,prob,y);
}
extern "C" int fastshared_ep_run(void*state,int rank,int layer,int rows,const void*x,const void*ids,const void*prob,void*y) {
 return runEp(state,rank,layer,rows,x,ids,prob,y,{});
}
extern "C" int fastshared_ep_router_supported(void *state) {
 return state && static_cast<fastllm_shared_flex_ep::State *>(state)->routerSupported;
}
extern "C" int fastshared_ep_run_router(void *state, int rank, int layer, int rows, const void *x,
        const float *router, const float *bias, int *ids, float *scores, void *out, int needNorm, float scale) {
 if (!fastshared_ep_router_supported(state) || rows < 2 ||
     !router || reinterpret_cast<uintptr_t>(router) % 16) return 1;
 return runEp(state, rank, layer, rows, x, ids, scores, out, {router, bias, ids, scores, needNorm, scale});
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
