// FastLLM native SM103 FlexMoE. See docs/naive-moe-backends.md for ownership,
// admission, source provenance, and the preserved numerical limitations.
// SM103 TP8/EP1 MoE. H4096, I256/rank, E256, top8.
// Owns packed weights or borrows MegaMoE-format FP8/UE8M0 buffers.
// Workspace is allocated at creation; run() adds no allocation or conversion.
// Routed owned layers may use 32-row tiles; unsupported paths retain 16/8-row tiles.
// Requires identical valid top-8 IDs on all ranks (unique within each row),
// identical invocation order and exclusive per-state ordered streams.
// Run on the rank's current device, using cudaStreamPerThread.
// Single row: fused quantization+gate; fused warp down+BF16 packed collective.
// Multiple rows: GPU expert-membership dispatch, with nonwaiting producer
// fallback if the fused down grid exceeds measured residency capacity.
#include <cuda.h>
#include "fastllm-norm-rows-router.cuh"
#include "fastllm-cuda-launch.cuh"
#include "fastllm-naive-flex-moe.h"
#include "fastllm-shared-flex-router-sm103.cuh"
#include <deep_gemm/mma/sm100.cuh>
#include "fastllm-shared-flex-weight.cuh"
#include "fastllm-mxfp4-weight-pack.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <condition_variable>
namespace fastllm_shared_flex_tp {
#ifndef FM_RELAXED_MAILBOX
#define FM_RELAXED_MAILBOX 0
#endif
constexpr int H=4096,I=256,E=256,R=8,K=8,MAX_M=8;
struct Rank {
 bool mxfp4 = false;
  const __nv_bfloat16* raw_x;int rows;CUtensorMap*warpmap;CUtensorMap*xmap=nullptr;float*wlocal;int warp_capacity;
  CUtensorMap *gate128map=nullptr; int gate128ctas=0;
  bool pdl = false, multi_pdl = false;
  int pdl_warp_capacity = 0;
  int *uniq,*members,*rowcounts,*nuniq;int* gcount; CUtensorMap *gmap,*dmap; float *gpartial,*dpartial;
  int device,rank; const uint8_t *gate=nullptr,*down=nullptr,*gs=nullptr,*ds=nullptr;
  uint8_t *xq,*xs,*mid,*ms; int *jobs,*count; uint64_t *epoch,*box;
  uint64_t* peers[R];
};
struct State {
    // All ranks agree before any dense writer/collective is launched. CUDA
    // Graph replay bypasses this host-only admission. Two result slots prevent
    // a fast rank's next call from overwriting a slow rank's previous result.
    std::mutex denseMutex;std::condition_variable denseCondition;
    uint64_t denseGeneration=0;int denseArrived=0,denseRows=0;unsigned denseRanks=0;
    bool denseEligible=false,denseResults[2]={false,false};
    bool batch64Supported = true, batch32Supported = true, wideSupported = true, denseSupported = true, ownerDownSupported = true;
    bool packedOwnerDownSupported = true, routePipelineSupported = true;
    bool routerSupported = true, preparedSupported = true;
    // Rank workers can bind MXFP4 layers concurrently and disable the FP8-only layout.
    std::atomic<bool> gate128Supported{true};
    Rank r[R]; std::vector<std::array<Rank,R>> layers; std::vector<void*> owned[R]; };
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void* alloc(size_t n){void*p=nullptr;check(cudaMalloc(&p,n));return p;}
__device__ __forceinline__ unsigned exponent(float a){return max((__float_as_uint(a)+0x7fffffu-0x600000u)>>23,113u)-8;}
__device__ __forceinline__ float sf(unsigned e){return __uint_as_float(e<<23);}
__device__ __forceinline__ size_t sf_offset(int expert,int row,int kb,int n,int k){
  return size_t(expert)*n*(k/32)+fastllm_mxfp4::scaleOffset(row,kb,n);
}
__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,
   unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ unsigned ld4(const uint8_t*p){return *reinterpret_cast<const unsigned*>(p);}
template<bool Select, bool Shortlist = false, bool AdvanceEpoch = true>
__device__ __forceinline__ void PrepareRoutes(Rank a, const int *ids, int rows,
    const float *router, const float *bias, int *selected, float *probabilities, int needNorm, float routeScale) {
  const int lane = threadIdx.x % 32;
  const int *chosen=ids;
  if constexpr (Select) {
    __shared__ fastllm::cuda::RouterTopKScratch<8> scratch[8];
    __shared__ int localIds[64];
    __shared__ float localProb[64];
    int row=threadIdx.x/32;
    if (row < rows) {
      bool done = false;
      if constexpr (Shortlist)
        done = SelectShortlist(router+row*E, bias, localIds+row*K, localProb+row*K, needNorm, routeScale, scratch[row]);
      if (!done) fastllm::cuda::SelectRouterTopK<256,8>(router+row*E,bias,
          localIds+row*K,localProb+row*K,bias!=nullptr,needNorm,routeScale,scratch[row]);
    }
    __syncthreads();
    if(threadIdx.x<rows*8){
      selected[threadIdx.x]=localIds[threadIdx.x];
      probabilities[threadIdx.x]=localProb[threadIdx.x];
    }
    chosen=localIds;
  }
  if(threadIdx.x < rows*K) a.jobs[threadIdx.x] = threadIdx.x;
  if(threadIdx.x==0) { if constexpr (AdvanceEpoch) ++*a.epoch; *a.count=rows*K; }
  {
    __shared__ unsigned long long masks[256];__shared__ int totals[2];
    int j=threadIdx.x;masks[j]=0;__syncthreads();
    int e=j<rows*K?chosen[j]:0;
    if(j<rows*K)atomicOr(masks+e,1ull<<j);
    __syncthreads();
    unsigned long long mask=j<rows*K?masks[e]:0;
    int first=j<rows*K&&__ffsll(mask)-1==j,prefix=first;
    if(j<64){
      #pragma unroll
      for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,prefix,d);if(lane>=d)prefix+=v;}
      if(lane==31)totals[j/32]=prefix;
    }
    __syncthreads();
    if(j==0)*a.nuniq=totals[0]+totals[1];
    if(first){int item=prefix-1+(j>=32?totals[0]:0),count=0;
      a.uniq[item]=e;
      while(mask){a.members[item*8+count++]=__ffsll(mask)-1;mask&=mask-1;}
      a.rowcounts[item]=count;
    }
  }
}

// All router tiles read the original residual. Publish its rounded update only
// after they have finished, using gate scratch whose lifetime starts later.
__global__ void FinishResidualRoutes(Rank a, __nv_bfloat16 *hidden, const __nv_bfloat16 *sum,
        const float *router, const float *bias, int *ids, float *scores, int needNorm, float scale) {
    if (blockIdx.x == 0) {
        PrepareRoutes<true, true>(a, ids, a.rows, router, bias, ids, scores, needNorm, scale);
    } else {
        const int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
        reinterpret_cast<uint4 *>(hidden)[i] = reinterpret_cast<const uint4 *>(sum)[i];
    }
}

// The attention norm/router producer already wrote K32 activations and
// advanced the mailbox generation. Build the shared expert schedule once.
template<bool Pipeline>
__global__ void SelectPreparedRoutes(Rank a, const float *router, const float *bias,
        int *ids, float *prob, int needNorm, float routeScale) {
    if constexpr (Pipeline) cudaTriggerProgrammaticLaunchCompletion();
    PrepareRoutes<true,true,false>(a, ids, Pipeline ? 8 : a.rows, router, bias, ids, prob, needNorm, routeScale);
}

template<bool Select = false>
__global__ void prepare(Rank a,const __nv_bfloat16*x,const int*ids,int rows,
    const float *router=nullptr,const float *bias=nullptr,int *selected=nullptr,
    float *probabilities=nullptr,int needNorm=1,float routeScale=1.f){
  int t=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
  if(blockIdx.x==0) PrepareRoutes<Select>(a,ids,rows,router,bias,selected,probabilities,needNorm,routeScale);
  if(t<rows*K*4)a.gcount[t]=0;
  if(t>=rows*H)return;
  float v=__bfloat162float(x[t]),mx=fabsf(v);
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
  unsigned e=exponent(mx);
  a.xq[t]=__nv_fp8_e4m3(v*sf(254-e)).__x;
  if(lane==0)a.xs[t/32]=e;
}
// 32 intermediate outputs/CTA, 4 row groups x 2 K partitions.
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
// Fused peer waits require the entire grid to fit the measured occupancy.
// Larger grids use a separate nonwaiting producer and receive kernel.
}
#include "fastllm-shared-flex-tp-sm103.cuh"
#include "fastllm-shared-flex-grouped-sm103.cuh"
#include "fastllm-shared-flex-gate-sm103.cuh"
#include "fastllm-shared-flex-block128-sm103.cuh"
#include "fastllm-shared-flex-down-sm103.cuh"
#include "fastllm-shared-flex-wide-sm103.cuh"
#include "fastllm-shared-flex-dense-sm103.cuh"
extern "C" int fastshared_tp_supported(const int*,int);
extern "C" void fastshared_tp_destroy(void*state);
extern "C" int fastshared_tp_create(const int*devices,int layers,void**result){
  using namespace fastllm_shared_flex_tp;
  if(!devices||!result||layers<=0)return 1;
  *result=nullptr;
  if(!fastshared_tp_supported(devices,8))return 1;
  State*s=nullptr;
  try{
    // Validate every rank before allocating any device storage.
    for(int r=0;r<R;++r){
      cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,devices[r]));
      if(prop.major!=10||prop.minor!=3)throw std::runtime_error("This experimental cubin requires SM103");
      for(int q=0;q<r;++q)if(devices[q]==devices[r])throw std::runtime_error("Eight distinct devices required");
      for(int q=0;q<R;++q)if(q!=r){int access=0;check(cudaDeviceCanAccessPeer(&access,devices[r],devices[q]));
        if(!access)throw std::runtime_error("All-to-all CUDA peer access required");}
    }
    s=new State{};s->layers.resize(layers);
    for(int r=0;r<R;++r){
      check(cudaSetDevice(devices[r]));auto&a=s->r[r];a.device=devices[r];a.rank=r;
      for(int p=0;p<R;++p)if(p!=r){auto e=cudaDeviceEnablePeerAccess(devices[p],0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e);}
      check(cudaFuncSetAttribute(warp_down<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,73728));check(cudaFuncSetAttribute(warp_down<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,73728));
      int blocks;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,warp_down<true>,256,73728));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,devices[r]));a.warp_capacity=blocks*prop.multiProcessorCount;
      // PDL is optional. Check both images and the exact secondary's complete
      // peer-wait grid before selecting it; unsupported pairs use ordinary launches.
      cudaFuncAttributes producer{}, consumer{};
      a.pdl = cudaFuncGetAttributes(&producer, single_matmul<true, true>) == cudaSuccess &&
          producer.binaryVersion >= 90 && producer.ptxVersion >= 90 &&
          cudaFuncGetAttributes(&consumer, warp_down<true, true>) == cudaSuccess &&
          consumer.binaryVersion >= 90 && consumer.ptxVersion >= 90 &&
          cudaFuncSetAttribute(warp_down<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 73728) == cudaSuccess &&
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, warp_down<true, true>, 256, 73728) == cudaSuccess &&
          blocks * prop.multiProcessorCount >= 128;
      if (a.pdl) a.pdl_warp_capacity = blocks * prop.multiProcessorCount;
      else cudaGetLastError();
      a.multi_pdl = a.pdl &&
          cudaFuncGetAttributes(&producer, multi_gate<true>) == cudaSuccess &&
          producer.binaryVersion >= 90 && producer.ptxVersion >= 90 &&
          cudaFuncGetAttributes(&consumer, multi_down<true>) == cudaSuccess &&
          consumer.binaryVersion >= 90 && consumer.ptxVersion >= 90 &&
          cudaFuncSetAttribute(multi_gate<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 73728) == cudaSuccess &&
          cudaFuncSetAttribute(multi_down<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 36864) == cudaSuccess;
      if (!a.multi_pdl) cudaGetLastError();
      cudaFuncAttributes fused{};int fusedBlocks=0;
      if (cudaFuncGetAttributes(&fused, routed_gate) != cudaSuccess || fused.binaryVersion != 103 ||
          fused.maxThreadsPerBlock < 256 ||
          cudaFuncSetAttribute(routed_gate, cudaFuncAttributeMaxDynamicSharedMemorySize, RoutedGateBytes) != cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fusedBlocks, routed_gate, 256, RoutedGateBytes) != cudaSuccess || fusedBlocks < 1) {
          s->preparedSupported = false;cudaGetLastError();
      }
      cudaFuncAttributes block128{}; int block128Blocks=0;
      if (cudaFuncGetAttributes(&block128,block128_gate)!=cudaSuccess || block128.binaryVersion!=103 ||
          block128.maxThreadsPerBlock<384 ||
          cudaFuncSetAttribute(block128_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128GateBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block128Blocks,block128_gate,384,Block128GateBytes)!=cudaSuccess || block128Blocks<1 ||
          cudaFuncGetAttributes(&block128,coarsen_gate_weights)!=cudaSuccess || block128.maxThreadsPerBlock<256) {
          s->gate128Supported=false; cudaGetLastError();
      }
      cudaFuncAttributes wide{};int wideBlocks=0;
      if(cudaFuncGetAttributes(&wide,block128_wide_gate)!=cudaSuccess || wide.binaryVersion!=103 ||
          cudaFuncSetAttribute(block128_wide_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128WideGateBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wideBlocks,block128_wide_gate,384,Block128WideGateBytes)!=cudaSuccess || wideBlocks<1 ||
          cudaFuncSetAttribute(compact_down,cudaFuncAttributeMaxDynamicSharedMemorySize,WideDownBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wideBlocks,compact_down,128,WideDownBytes)!=cudaSuccess || wideBlocks<1 ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wideBlocks,compact_finish,64,0)!=cudaSuccess ||
          wideBlocks*prop.multiProcessorCount<16*32){s->wideSupported=false;cudaGetLastError();}
      cudaFuncAttributes batch{};int batchBlocks=0;
      if(cudaFuncGetAttributes(&batch,block128_batch_gate)!=cudaSuccess || batch.binaryVersion!=103 ||
          batch.maxThreadsPerBlock<384 ||
          cudaFuncSetAttribute(block128_batch_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128BatchGateBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,block128_batch_gate,384,Block128BatchGateBytes)!=cudaSuccess || batchBlocks<1 ||
          cudaFuncGetAttributes(&batch,batch32_down)!=cudaSuccess || batch.binaryVersion!=103 || batch.maxThreadsPerBlock<128 ||
          cudaFuncSetAttribute(batch32_down,cudaFuncAttributeMaxDynamicSharedMemorySize,Batch32DownBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,batch32_down,128,Batch32DownBytes)!=cudaSuccess || batchBlocks<1 ||
          cudaFuncGetAttributes(&batch,batch32_exchange_finish)!=cudaSuccess || batch.binaryVersion!=103 || batch.maxThreadsPerBlock<128 ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,batch32_exchange_finish,128,0)!=cudaSuccess ||
          batchBlocks*prop.multiProcessorCount<32*16) {
          s->batch32Supported=false;cudaGetLastError();
      }
      if(cudaFuncGetAttributes(&batch,block128_batch64_gate)!=cudaSuccess || batch.binaryVersion!=103 ||
          cudaFuncSetAttribute(block128_batch64_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128Batch64GateBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,block128_batch64_gate,384,Block128Batch64GateBytes)!=cudaSuccess || batchBlocks<1 ||
          cudaFuncGetAttributes(&batch,batch64_down)!=cudaSuccess || batch.binaryVersion!=103 ||
          cudaFuncSetAttribute(batch64_down,cudaFuncAttributeMaxDynamicSharedMemorySize,Batch64DownBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,batch64_down,128,Batch64DownBytes)!=cudaSuccess || batchBlocks<1 ||
          cudaFuncGetAttributes(&batch,batch64_exchange_finish)!=cudaSuccess || batch.binaryVersion!=103 ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks,batch64_exchange_finish,128,0)!=cudaSuccess ||
          batchBlocks*prop.multiProcessorCount<64*8) {s->batch64Supported=false;cudaGetLastError();}
      a.gate128ctas=prop.multiProcessorCount;
      // The producer never waits for peers. The receive grid must fit on every
      // device; one failed optional probe retains the old path for the group.
      if (!SupportsOwnerDown<false>(prop.multiProcessorCount)) {
          s->ownerDownSupported = false;
          cudaGetLastError();
      }
      if (!SupportsOwnerDown<true>(prop.multiProcessorCount)) {
          s->packedOwnerDownSupported = false;
          cudaGetLastError();
      }
      cudaFuncAttributes route{}, gate{};
      if (cudaFuncGetAttributes(&route, SelectPreparedRoutes<true>) != cudaSuccess ||
          route.binaryVersion < 90 || route.ptxVersion < 90 ||
          cudaFuncGetAttributes(&gate, prepared_gate) != cudaSuccess ||
          gate.binaryVersion < 90 || gate.ptxVersion < 90 ||
          cudaFuncSetAttribute(prepared_gate, cudaFuncAttributeMaxDynamicSharedMemorySize, GroupedGateBytes) != cudaSuccess) {
          s->routePipelineSupported = false;
          cudaGetLastError();
      }
      cudaFuncAttributes denseUp{},denseDown{}; int denseBlocks=0;
      if(!prop.cooperativeLaunch ||
          cudaFuncGetAttributes(&denseUp,dense_up_gate)!=cudaSuccess || denseUp.binaryVersion!=103 ||
          cudaFuncSetAttribute(dense_up_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,DenseUpBytes)!=cudaSuccess ||
          cudaFuncGetAttributes(&denseDown,dense_down_reduce)!=cudaSuccess || denseDown.binaryVersion!=103 ||
          cudaFuncSetAttribute(dense_down_reduce,cudaFuncAttributeMaxDynamicSharedMemorySize,DenseDownBytes)!=cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&denseBlocks,dense_down_reduce,256,DenseDownBytes)!=cudaSuccess ||
          denseBlocks*prop.multiProcessorCount<H/64) { s->denseSupported=false;cudaGetLastError(); }
      cudaFuncAttributes selection{};
      if (cudaFuncGetAttributes(&selection, prepare<true>) != cudaSuccess || selection.maxThreadsPerBlock < 256) {
        s->routerSupported = false; cudaGetLastError();
      }
      a.warpmap=(CUtensorMap*)alloc(128);a.wlocal=(float*)alloc(MAX_M*H*4);check(cudaFuncSetAttribute(multi_gate<>,cudaFuncAttributeMaxDynamicSharedMemorySize,73728));check(cudaFuncSetAttribute(multi_down<>,cudaFuncAttributeMaxDynamicSharedMemorySize,36864));
      a.uniq=(int*)alloc(64*4);a.members=(int*)alloc(64*8*4);a.rowcounts=(int*)alloc(64*4);a.nuniq=(int*)alloc(4);a.gcount=(int*)alloc(MAX_M*K*4*4);check(cudaMemset(a.gcount,0,MAX_M*K*4*4));a.gmap=(CUtensorMap*)alloc(128);a.dmap=(CUtensorMap*)alloc(128);
      a.gpartial=(float*)alloc(MAX_M*K*8*8*512*4);a.dpartial=(float*)alloc(MAX_M*K*1*4096*4);
      // Legacy gate partials use at most 32 experts * 8 K partitions *
      // 8 members * 512 floats = 4 MiB. Keep the upper half of the existing
      // 8 MiB arena as two permanent 16-row mailbox banks, across shape changes.
      static_assert(MAX_M*K*8*8*512*4==2*WideMailboxOffset);
      check(cudaMemset(reinterpret_cast<uint8_t*>(a.gpartial)+WideMailboxOffset,0,WideMailboxOffset));
      a.xq=(uint8_t*)alloc(MAX_M*H);a.xs=(uint8_t*)alloc(MAX_M*H/32);
      // One descriptor per rank/workspace, not one activation/weight copy.
      cudaFuncAttributes grouped{};int groupedBlocks=0;
      bool groupedSupported=cudaFuncGetAttributes(&grouped,grouped_tma_gate)==cudaSuccess &&
          grouped.binaryVersion==103 && grouped.maxThreadsPerBlock>=128 &&
          cudaFuncSetAttribute(grouped_tma_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,GroupedGateBytes)==cudaSuccess &&
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&groupedBlocks,grouped_tma_gate,128,GroupedGateBytes)==cudaSuccess && groupedBlocks>0;
      if(!groupedSupported)cudaGetLastError();
      if(groupedSupported){
        CUtensorMap map;cuuint64_t dims[]={H,MAX_M},strides[]={H};
        cuuint32_t box[]={128,8},steps[]={1,1};
        if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,a.xq,dims,strides,box,steps,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS){
          a.xmap=(CUtensorMap*)alloc(128);check(cudaMemcpy(a.xmap,&map,128,cudaMemcpyHostToDevice));
          check(cudaMemset(a.xq,0,MAX_M*H));
        }
      }
      a.mid=(uint8_t*)alloc(MAX_M*K*I);a.ms=(uint8_t*)alloc(MAX_M*K*I/32);
      a.jobs=(int*)alloc(MAX_M*K*4);a.count=(int*)alloc(4);a.epoch=(uint64_t*)alloc(8);
      check(cudaMemset(a.epoch,0,8));a.box=(uint64_t*)alloc(2*MAX_M*R*(H/2)*8);
      check(cudaMemset(a.box,0,2*MAX_M*R*(H/2)*8));
    }
    for(int r=0;r<R;++r)for(int p=0;p<R;++p)s->r[r].peers[p]=s->r[p].box;
    *result=s;return 0;
  }catch(const std::exception&e){fastshared_tp_destroy(s);fprintf(stderr,"fastshared_tp_create: %s\n",e.what());return 1;}
}
static int bind_tp(void *state, int rank, int layer, const void *gate, const void *down,
                   const void *gs, const void *ds, bool mxfp4) {
 try {
  using namespace fastllm_shared_flex_tp;
  auto *s=static_cast<State *>(state);
  if (!s || rank<0 || rank>=R || layer<0 || layer>=(int)s->layers.size() ||
      !gate || !down || !gs || !ds) return 1;
  auto &a=s->layers[layer][rank];
  if (a.gate) return 1;
  a=s->r[rank]; a.gmap=a.dmap=a.warpmap=nullptr;
  a.mxfp4=mxfp4;
  a.gate=static_cast<const uint8_t *>(gate); a.down=static_cast<const uint8_t *>(down);
  a.gs=static_cast<const uint8_t *>(gs); a.ds=static_cast<const uint8_t *>(ds);
  check(cudaSetDevice(a.device));
  for (int which=0;which<2;++which) {
    CUtensorMap map;
    if (fastllm_shared_flex::weight_map(&map, which ? down : gate, which ? I : H,
            which ? H : 2*I, E, mxfp4)!=CUDA_SUCCESS) return 1;
    auto &dst=which ? a.dmap : a.gmap;
    dst=static_cast<CUtensorMap *>(alloc(sizeof(map)));
    check(cudaMemcpy(dst,&map,sizeof(map),cudaMemcpyHostToDevice));
  }
  if (mxfp4) return 0;
  a.warpmap=static_cast<CUtensorMap *>(alloc(sizeof(CUtensorMap)));
  CUtensorMap map;cuuint64_t dims[]={256,4096,256},strides[]={256,4096*256};cuuint32_t box[]={128,32,1},steps[]={1,1,1};
  if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,(void*)down,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)return 1;
  if(cudaMemcpy(a.warpmap,&map,128,cudaMemcpyHostToDevice)!=cudaSuccess)return 1;
  return 0;

 } catch (const std::exception &e) { fprintf(stderr,"Shared Flex bind: %s\n",e.what());return 1; }
}
extern "C" int fastshared_tp_bind(void *state,int rank,int layer,const void *gate,
        const void *down,const void *gs,const void *ds) {
    return bind_tp(state,rank,layer,gate,down,gs,ds,false);
}

static void launch_prepared_down(const fastllm_shared_flex_tp::Rank &a,
    const int *ids,const float *prob,void *output,cudaStream_t stream,const fastllm_shared_flex_tp::State &state);
namespace fastllm_shared_flex_tp {
// Reuse the existing gate arena; preserve the FP8/32 -> block-128 rounding
// of the router producer, including padded rows. For 32 rows, select routes
// here in parallel; the smaller paths keep selection in gate.
template<int Capacity=8> __global__ void prepare_block128(Rank a,const __nv_bfloat16 *input,
    const float *router=nullptr,const float *bias=nullptr,int *ids=nullptr,float *prob=nullptr,
    int needNorm=1,float routeScale=1.f) {
    const int row=blockIdx.x/32,k=blockIdx.x%32,t=threadIdx.x;
    if(blockIdx.x==0){for(int i=t;i<256;i+=128)a.gcount[i]=0;if(t==0)++*a.epoch;}
    if constexpr(Capacity>=32) {
      // One existing quantization CTA per row selects routes concurrently.
      __shared__ fastllm::cuda::RouterTopKScratch<8> selection;
      if(k==0 && row<a.rows && t<32)
        block128_gate_detail::SelectGateRoutes(router+row*256,bias,ids+row*8,prob+row*8,
            needNorm,routeScale,selection);
    }
    const float v=row<a.rows?float(input[row*H+k*128+t]):0.f;
    float mx=fabsf(v);
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    const unsigned e=exponent(mx);
    const __nv_fp8_e4m3 q(v*sf(254-e));
    __shared__ unsigned exponents[4];
    if(t%32==0)exponents[t/32]=e;
    __syncthreads();
    const unsigned coarse=__reduce_max_sync(0xffffffff,t%32<4?exponents[t%32]:0);
    auto *out=reinterpret_cast<uint8_t*>(a.gpartial);
    out[row*4112+k*128+t]=__nv_fp8_e4m3(float(q)*sf(unsigned(max(0,127+int(e)-int(coarse))))).__x;
    if(t==0)reinterpret_cast<float*>(out+Capacity*4112)[k*Capacity+row]=sf(coarse);
    if(k==0&&t<16)out[row*4112+4096+t]=0;
}
}
// Wide tiles borrow the same arena for ordinary and router-prepared entry.
// The permanent upper mailbox banks remain untouched across shape changes.
static fastllm_shared_flex_tp::Rank wide_rank(fastllm_shared_flex_tp::State *s,
        fastllm_shared_flex_tp::Rank a) {
  using namespace fastllm_shared_flex_tp;
  auto *arena=reinterpret_cast<uint8_t*>(a.gpartial);
  a.mid=arena+WideScratch::Mid;a.ms=arena+WideScratch::Scales;
  a.uniq=reinterpret_cast<int*>(arena+WideScratch::Experts);
  a.rowcounts=reinterpret_cast<int*>(arena+WideScratch::Counts);
  a.members=reinterpret_cast<int*>(arena+WideScratch::Members);
  a.gpartial=reinterpret_cast<float*>(arena+WideScratch::Prepared);
  a.dpartial=reinterpret_cast<float*>(arena+WideScratch::DownPartials);
  a.box=reinterpret_cast<uint64_t*>(arena+WideMailboxOffset);
  for(int peer=0;peer<8;++peer)a.peers[peer]=reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(s->r[peer].gpartial)+WideMailboxOffset);
  return a;
}
static void launch_wide(const fastllm_shared_flex_tp::Block128GateArgs &args,
        __nv_bfloat16 *output,cudaStream_t stream) {
  using namespace fastllm_shared_flex_tp;
  const auto &a=args.rank;
  block128_wide_gate<<<a.gate128ctas,384,Block128WideGateBytes,stream>>>(args);
  compact_down<<<444,128,WideDownBytes,stream>>>(a);
  compact_finish<<<dim3(32,a.rows),64,0,stream>>>(a,output);
}
template<class Scratch>
static fastllm_shared_flex_tp::Rank batch_rank(fastllm_shared_flex_tp::Rank a) {
  auto *scratch=reinterpret_cast<uint8_t*>(a.dpartial);
  auto *partials=a.gpartial;
  a.mid=scratch+Scratch::Mid;a.ms=scratch+Scratch::Scales;
  a.uniq=reinterpret_cast<int*>(scratch+Scratch::Experts);
  a.rowcounts=reinterpret_cast<int*>(scratch+Scratch::Counts);
  a.members=reinterpret_cast<int*>(scratch+Scratch::Members);
  a.gpartial=reinterpret_cast<float*>(scratch+Scratch::Prepared);
  a.dpartial=partials;
  return a;
}
static int run_tp(void*state,int rank,int layer,int rows,const void*x,const void*ids,const void*p,void*y,
    const float *router=nullptr,const float *bias=nullptr,int needNorm=1,float routeScale=1.f){
  if(!state||rank<0||rank>=8||rows<0)return 1;
  if(rows==0)return 0;
  if(!x||!ids||!p||!y)return 1;
  auto*s=(fastllm_shared_flex_tp::State*)state;if(layer<0||layer>=(int)s->layers.size())return 1;auto base=s->layers[layer][rank];
  if(!base.gate||!base.down||!base.gs||!base.ds||(router&&!s->routerSupported))return 1;
  auto stream=cudaStreamPerThread;
  // Tile shape changes the collective layout. Every rank must make the same
  // choice, including optional per-layer map/bind failures during loading.
  bool wide=router && s->wideSupported && s->gate128Supported;
  for(int peer=0;peer<8 && wide;++peer)wide=s->layers[layer][peer].gate128map!=nullptr;
  const int tileRows=wide?(s->batch32Supported?(s->batch64Supported?64:32):16):8;
  for(size_t offset=0;offset<(size_t)rows;offset+=tileRows){
    int n=rows-offset<tileRows?rows-offset:tileRows;auto a=base;a.rows=n;
    auto xx=(const __nv_bfloat16*)x+(size_t)offset*4096;
    auto ii=(const int*)ids+(size_t)offset*8;auto pp=(const float*)p+(size_t)offset*8;
    auto yy=(__nv_bfloat16*)y+(size_t)offset*4096;a.raw_x=xx;
    if(router && n>32 && tileRows==64) {
      using namespace fastllm_shared_flex_tp;
      a=batch_rank<Batch64Scratch>(a);
      prepare_block128<64><<<2048,128,0,stream>>>(a,xx,router+offset*256,bias,
          const_cast<int*>(ii),const_cast<float*>(pp),needNorm,routeScale);
      Block128GateArgs args{a,router+offset*256,bias,const_cast<int*>(ii),const_cast<float*>(pp),
          needNorm,routeScale,a.gate128map,a.gs,reinterpret_cast<uint8_t*>(a.gpartial)};
      block128_batch64_gate<<<a.gate128ctas,384,Block128Batch64GateBytes,stream>>>(args);
      for(int half=0;half<2;++half) {
        batch64_down<<<444,128,Batch64DownBytes,stream>>>(a,half*(H/2));
        batch64_exchange_finish<<<dim3(8,n),128,0,stream>>>(a,yy,half*(H/2));
      }
      continue;
    }
    if(router && n>16 && tileRows>=32) {
      using namespace fastllm_shared_flex_tp;
      a=batch_rank<BatchScratch>(a);
      prepare_block128<32><<<1024,128,0,stream>>>(a,xx,router+offset*256,bias,
          const_cast<int*>(ii),const_cast<float*>(pp),needNorm,routeScale);
      Block128GateArgs args{a,router+offset*256,bias,const_cast<int*>(ii),const_cast<float*>(pp),
          needNorm,routeScale,a.gate128map,a.gs,reinterpret_cast<uint8_t*>(a.gpartial)};
      block128_batch_gate<<<a.gate128ctas,384,Block128BatchGateBytes,stream>>>(args);
      batch32_down<<<444,128,Batch32DownBytes,stream>>>(a);
      batch32_exchange_finish<<<dim3(16,n),128,0,stream>>>(a,yy);
      continue;
    }
    if(router && n>8 && wide){
      using namespace fastllm_shared_flex_tp;
      a=wide_rank(s,a);
      prepare_block128<16><<<16*32,128,0,stream>>>(a,xx);
      Block128GateArgs args{a,router+offset*256,bias,const_cast<int*>(ii),const_cast<float*>(pp),
          needNorm,routeScale,a.gate128map,a.gs,reinterpret_cast<uint8_t*>(a.gpartial)};
      launch_wide(args,yy,stream);
      continue;
    }
    if(router && n>=7 && s->gate128Supported && a.gate128map){
      fastllm_shared_flex_tp::prepare_block128<><<<256,128,0,stream>>>(a,xx);
      fastllm_shared_flex_tp::Block128GateArgs args{a,router+offset*256,bias,
          const_cast<int*>(ii),const_cast<float*>(pp),needNorm,routeScale,a.gate128map,a.gs,
          reinterpret_cast<uint8_t*>(a.gpartial)};
      fastllm_shared_flex_tp::block128_gate<<<a.gate128ctas,384,fastllm_shared_flex_tp::Block128GateBytes,stream>>>(args);
      launch_prepared_down(a,ii,pp,yy,stream,*s);
      continue;
    }
    if (n > 1 || router || a.mxfp4) {
      if (router) fastllm_shared_flex_tp::prepare<true><<<n*16,256,0,stream>>>(a,xx,ii,n,
          router+offset*256,bias,const_cast<int*>(ii),const_cast<float*>(pp),needNorm,routeScale);
      else fastllm_shared_flex_tp::prepare<><<<n*16,256,0,stream>>>(a,xx,ii,n);
    }
    if (a.mxfp4) {
      // The grouped gate shares compact E2M1 pages among routed tokens.
      // Smaller tiles keep the general block-scaled gate.
      using namespace fastllm_shared_flex_tp;
      if (n>=4 && a.xmap) grouped_tma_gate<<<444,128,GroupedGateBytes,stream>>>(a,ii,pp);
      else multi_gate<><<<444,128,73728,stream>>>(a,ii,pp);
      launch_prepared_down(a,ii,pp,yy,stream,*s);
      continue;
    }
    if (n == 1 && a.pdl) {
      // Selection has already completed. Down can prefetch IDs/weights while
      // gate/up computes; it waits before touching activations or counters.
      fastllm_shared_flex_tp::single_matmul<true, true><<<296,128,0,stream>>>(a,ii,pp);
      if (!fastllm::cuda_launch::Launch<fastllm_shared_flex_tp::warp_down<true, true>>(
              dim3(128), dim3(256), 73728, stream, a, ii, yy, n))
        return 1; // A launch failure is an error, never replay a partial collective.
      continue;
    }
    // Seven/eight-row disjoint routing is faster with ordinary launches;
    // retain that path without reading routing metadata back to the host.
    if (n > 1 && n <= 6 && a.multi_pdl && (n >= 4 || n * 128 <= a.pdl_warp_capacity)) {
      fastllm_shared_flex_tp::multi_gate<true><<<444,128,73728,stream>>>(a,ii,pp);
      if (n <= 3) {
        if (!fastllm::cuda_launch::Launch<fastllm_shared_flex_tp::warp_down<true, true>>(
                dim3(128,n), dim3(256), 73728, stream, a, ii, yy, n)) return 1;
      } else {
        if (!fastllm::cuda_launch::Launch<fastllm_shared_flex_tp::multi_down<true>>(
                dim3(444), dim3(128), 36864, stream, a, ii, pp)) return 1;
        fastllm_shared_flex_tp::down_finish<<<dim3(32,n),64,0,stream>>>(a,yy,n);
      }
      continue;
    }
    if(n==1){fastllm_shared_flex_tp::single_matmul<true><<<296,128,0,stream>>>(a,ii,pp);}
    else{
      if(n>=7 && a.xmap)
        fastllm_shared_flex_tp::grouped_tma_gate<<<444,128,fastllm_shared_flex_tp::GroupedGateBytes,stream>>>(a,ii,pp);
      else fastllm_shared_flex_tp::multi_gate<><<<444,128,73728,stream>>>(a,ii,pp);
    }
    if(n<=3){
      if(n*128<=a.warp_capacity)fastllm_shared_flex_tp::warp_down<true><<<dim3(128,n),256,73728,stream>>>(a,ii,yy,n);
      else{
        fastllm_shared_flex_tp::warp_down<false><<<dim3(128,n),256,73728,stream>>>(a,ii,yy,n);
        fastllm_shared_flex_tp::warp_receive<<<dim3(32,n),64,0,stream>>>(a,yy,n);
      }
    }else{
      fastllm_shared_flex_tp::multi_down<><<<444,128,36864,stream>>>(a,ii,pp);
      fastllm_shared_flex_tp::down_finish<<<dim3(32,n),64,0,stream>>>(a,yy,n);
    }
  }
  return cudaGetLastError();
}
extern "C" int fastshared_tp_run(void*state,int rank,int layer,int rows,
    const void*x,const void*ids,const void*p,void*y){
  return run_tp(state,rank,layer,rows,x,ids,p,y);
}
extern "C" int fastshared_tp_router_supported(void*state){
  return state&&static_cast<fastllm_shared_flex_tp::State*>(state)->routerSupported;
}
extern "C" int fastshared_tp_run_router(void*state,int rank,int layer,int rows,const void*x,
    const float*router,const float*bias,int*ids,float*prob,void*y,int needNorm,float routeScale){
  if(!router || rows < 2)return 1;
  return run_tp(state,rank,layer,rows,x,ids,prob,y,router,bias,needNorm,routeScale);
}
extern "C" int fastshared_tp_prepared_input(void *state,int rank,int rows,FastllmMoePreparedInput *input){
  using namespace fastllm_shared_flex_tp;
  if(!input)return 0;*input={};
  if(!state || rank<0 || rank>=8 || rows<2 || rows>16)return 0;
  auto *s=static_cast<State*>(state);
  if(!s->preparedSupported || !s->routerSupported)return 0;
  auto &a=s->r[rank];
  if(rows>8){
    if(!s->wideSupported || !s->gate128Supported)return 0;
    // This descriptor is shared by all MoE layers. Require owned/coarsened
    // maps on the whole group, excluding entirely unbound dense layers.
    bool bound=false;
    for(const auto &layer:s->layers){
      bool used=false;for(const auto &peer:layer)used=used || peer.gate;
      if(!used)continue;
      bound=true;for(const auto &peer:layer)if(!peer.gate128map)return 0;
    }
    if(!bound)return 0;
    *input={nullptr,nullptr,a.gcount,a.epoch,rows};
    input->block128=reinterpret_cast<uint8_t*>(a.gpartial)+WideScratch::Prepared;
    input->preferFusedSelection=true;return 1;
  }
  *input={a.xq,a.xs,a.gcount,a.epoch,rows};
  if(s->gate128Supported && rows>=4)input->block128=reinterpret_cast<uint8_t*>(a.gpartial);
  input->preferFusedSelection=a.xmap!=nullptr;
  input->jobs=a.jobs;input->count=a.count;input->unique=a.nuniq;
  input->experts=a.uniq;input->members=a.members;input->rowCounts=a.rowcounts;return 1;
}
// Prepared producer variants share the optional down/communication dispatch.
static void launch_prepared_down(const fastllm_shared_flex_tp::Rank &a,
    const int *ids,const float *prob,void *output,cudaStream_t stream,const fastllm_shared_flex_tp::State &state){
  using namespace fastllm_shared_flex_tp;
  const int rows=a.rows;
  if (state.packedOwnerDownSupported && a.mxfp4 && rows == 8) {
    owner_down<true><<<444,160,OwnerDownBytes,stream>>>(a,ids,prob);
    owner_finish<true><<<dim3(32,rows),64,0,stream>>>(a,(__nv_bfloat16*)output);
  } else if(state.ownerDownSupported && rows>=4 && rows<=8){
    owner_down<false><<<444,160,OwnerDownBytes,stream>>>(a,ids,prob);
    owner_finish<false><<<dim3(32,rows),64,0,stream>>>(a,(__nv_bfloat16*)output);
  }else if(!a.mxfp4 && rows<=3 && rows*128<=a.warp_capacity)warp_down<true><<<dim3(128,rows),256,73728,stream>>>(a,ids,(__nv_bfloat16*)output,rows);
  else if(!a.mxfp4 && rows<=3){
    warp_down<false><<<dim3(128,rows),256,73728,stream>>>(a,ids,(__nv_bfloat16*)output,rows);
    warp_receive<<<dim3(32,rows),64,0,stream>>>(a,(__nv_bfloat16*)output,rows);
  }else{
    multi_down<><<<444,128,36864,stream>>>(a,ids,prob);
    down_finish<<<dim3(32,rows),64,0,stream>>>(a,(__nv_bfloat16*)output,rows);
  }
}
extern "C" int fastshared_tp_residual_router_supported(void *state, int rank, int layer, int rows) {
    using namespace fastllm_shared_flex_tp;
    auto *s = static_cast<State *>(state);
    return s && rank >= 0 && rank < R && layer >= 0 && size_t(layer) < s->layers.size() &&
        rows >= 4 && rows <= MAX_M && s->preparedSupported &&
        s->routerSupported && s->layers[layer][rank].mxfp4 && s->layers[layer][rank].xmap;
}

extern "C" int fastshared_tp_residual_router(void *state, int rank, int layer, int rows, void *hidden,
        const void *branch, const float *gamma, float eps, void *normed, const float *weight,
        float *router, const float *bias, int *ids, float *scores, int needNorm, float routeScale) {
    using namespace fastllm_shared_flex_tp;
    if (!fastshared_tp_residual_router_supported(state, rank, layer, rows) || !hidden ||
        !branch || !gamma || !normed || !weight || !router || !ids || !scores) return cudaErrorInvalidValue;
    auto a = static_cast<State *>(state)->layers[layer][rank];
    a.rows = rows;
    auto *sum = reinterpret_cast<__nv_bfloat16 *>(a.gpartial);
    FastllmMoePreparedInput prepared{};
    prepared.values = a.xq;
    prepared.scales = a.xs;
    prepared.counters = a.gcount;
    auto stream = cudaStreamPerThread;
    fastllm_norm_rows::NormRowsRouterDirect<8, 2, true, 8, true><<<dim3(32,(rows+1)/2),256,0,stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden), weight, router, gamma, static_cast<__nv_bfloat16 *>(normed),
        eps, rows, prepared, static_cast<const __nv_bfloat16 *>(branch), sum);
    FinishResidualRoutes<<<1 + rows * 2,256,0,stream>>>(a, static_cast<__nv_bfloat16 *>(hidden),
        sum, router, bias, ids, scores, needNorm, routeScale);
    return cudaGetLastError();
}
extern "C" int fastshared_tp_run_prepared(void *state,int rank,int layer,int rows,
    const float *router,const float *bias,int *ids,float *prob,void *output,int needNorm,float routeScale){
  using namespace fastllm_shared_flex_tp;
  FastllmMoePreparedInput input{};
  if(!fastshared_tp_prepared_input(state,rank,rows,&input) || !router || !ids || !prob || !output)return 1;
  auto *s=static_cast<State*>(state);if(layer<0 || layer>=(int)s->layers.size())return 1;
  auto a=s->layers[layer][rank];if(!a.gate || !a.down || !a.gs || !a.ds)return 1;a.rows=rows;
  auto stream=cudaStreamPerThread;
  if(rows>8){
    a=wide_rank(s,a);
    Block128GateArgs args{a,router,bias,ids,prob,needNorm,routeScale,a.gate128map,a.gs,input.block128};
    launch_wide(args,(__nv_bfloat16*)output,stream);
    return cudaGetLastError();
  }
  if (s->routePipelineSupported && a.mxfp4 && a.xmap && rows == 8) {
    SelectPreparedRoutes<true><<<1,256,0,stream>>>(a,router,bias,ids,prob,needNorm,routeScale);
    cudaError_t err = fastllm::cuda_launch::LaunchProgrammatic<prepared_gate>(
        dim3(444), dim3(128), GroupedGateBytes, stream, a, prob);
    if (err != cudaSuccess) return err;
  } else if (a.mxfp4 && a.xmap && rows>=4) {
    SelectPreparedRoutes<false><<<1,256,0,stream>>>(a,router,bias,ids,prob,needNorm,routeScale);
    grouped_tma_gate<<<444,128,GroupedGateBytes,stream>>>(a,ids,prob);
  } else if(input.block128 && a.gate128map) {
    Block128GateArgs args{a,router,bias,ids,prob,needNorm,routeScale,a.gate128map,a.gs,input.block128};
    block128_gate<<<a.gate128ctas,384,Block128GateBytes,stream>>>(args);
  } else routed_gate<<<296,256,RoutedGateBytes,stream>>>(a,router,bias,ids,prob,needNorm,routeScale);
  launch_prepared_down(a, ids, prob, output, stream, *s);
  return cudaGetLastError();
}
extern "C" int fastshared_tp_run_preselected(void *state,int rank,int layer,int rows,
    const int *ids,const float *prob,void *output){
  using namespace fastllm_shared_flex_tp;
  FastllmMoePreparedInput input{};
  if(rows>8 || !fastshared_tp_prepared_input(state,rank,rows,&input) || !ids || !prob || !output)return 1;
  auto *s=static_cast<State*>(state);if(layer<0 || layer>=(int)s->layers.size())return 1;
  auto a=s->layers[layer][rank];if(!a.gate || !a.down || !a.gs || !a.ds)return 1;a.rows=rows;
  auto stream=cudaStreamPerThread;
  if(a.xmap && (rows>=7 || (a.mxfp4 && rows>=4)))grouped_tma_gate<<<444,128,GroupedGateBytes,stream>>>(a,ids,prob);
  else multi_gate<><<<444,128,73728,stream>>>(a,ids,prob);
  launch_prepared_down(a, ids, prob, output, stream, *s);
  return cudaGetLastError();
}
extern "C" size_t fastshared_tp_workspace_bytes(){
return 128+8*8*4*4+256+8*8*8*8*512*4+8*8*4096*4+8*4096+8*4096/32+8*8*256+8*8*256/32+8*8*4+4+8+2*8*8*(4096/2)*8+64*4+64*8*4+64*4+4+128+8*4096*4;
}
extern "C" void fastshared_tp_destroy(void*state){
  auto*s=(fastllm_shared_flex_tp::State*)state;if(!s)return;
  for(auto &ls:s->layers)for(auto &a:ls)if(a.gate){cudaSetDevice(a.device);cudaDeviceSynchronize();cudaFree(a.gmap);cudaFree(a.dmap);cudaFree(a.warpmap);cudaFree(a.gate128map);}
 for(int r=0;r<8;++r){cudaSetDevice(s->r[r].device);for(void*p:s->owned[r])cudaFree(p);}
 for(auto&a:s->r){cudaSetDevice(a.device);cudaDeviceSynchronize();
    for(void*p:{(void*)a.xmap,(void*)a.warpmap,(void*)a.wlocal,(void*)a.uniq,(void*)a.members,(void*)a.rowcounts,(void*)a.nuniq,(void*)a.gcount,(void*)a.gmap,(void*)a.dmap,(void*)a.gpartial,(void*)a.dpartial,(void*)a.xq,(void*)a.xs,(void*)a.mid,(void*)a.ms,(void*)a.jobs,(void*)a.count,(void*)a.epoch,(void*)a.box})cudaFree(p);
  }delete s;
}


// Load-time conversion from the original TP shards. Same per-32 UE8M0
// rounding, gate/up interleave and scale transpose as Blackwell MegaMoE.
namespace fastllm_shared_flex_tp {
__global__ void pack_native(const uint8_t*src,const float*scales,uint8_t*dst,uint8_t*out,int n,int k,bool gate){
 int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(group>=n*(k/32))return;
 int lane=threadIdx.x%32,row=group/(k/32),col=group%(k/32);
 int source=fastllm_mxfp4::sourceRow(row,n,gate);
 __nv_fp8_e4m3 q;q.__x=src[(size_t)source*k+col*32+lane];
 float value=float(q)*scales[(source/128)*(k/128)+col/4],mx=fabsf(value);
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
 unsigned e=exponent(mx);dst[(size_t)row*k+col*32+lane]=__nv_fp8_e4m3(value*sf(254-e)).__x;
 if(lane==0)out[sf_offset(0,row,col,n,k)]=e;
}
}
extern "C" int fastshared_tp_pack_mxfp4(void *state, int rank, int layer, const uint64_t *ptrs) {
    using namespace fastllm_shared_flex_tp;
    auto *s = static_cast<State *>(state);
    if (!s || !ptrs || rank < 0 || rank >= R || layer < 0 || layer >= (int)s->layers.size() ||
        s->layers[layer][rank].gate) return 1;
    try {
        check(cudaSetDevice(s->r[rank].device));
        uint8_t *values[2], *scales[2];
        for (int part = 0; part < 2; ++part) {
            const int n = part ? H : 2*I, k = part ? I : H;
            const size_t elements = size_t(n)*k;
            values[part] = static_cast<uint8_t *>(alloc(E*elements/2));
            s->owned[rank].push_back(values[part]);
            scales[part] = static_cast<uint8_t *>(alloc(E*elements/32));
            s->owned[rank].push_back(scales[part]);
            for (int e = 0; e < E; ++e) {
                if (!ptrs[e*2+part]) return 1;
                fastllm_mxfp4::packMxfp4<<<(elements/32+127)/128,128,0,cudaStreamPerThread>>>(
                    reinterpret_cast<const uint8_t *>(ptrs[e*2+part]),values[part]+e*elements/2,
                    scales[part]+e*elements/32,n,k,part==0);
            }
        }
        check(cudaGetLastError());
        check(cudaStreamSynchronize(cudaStreamPerThread));
        // K32 experts consume the native activation and scale buffers from the producer.
        // The coarse K128 image and its weight layout are specific to FP8.
        s->gate128Supported = false;
        return bind_tp(state,rank,layer,values[0],values[1],scales[0],scales[1],true);
    } catch (const std::exception &e) { fprintf(stderr,"Shared Flex MXFP4 pack: %s\n",e.what()); return 1; }
}

extern "C" int fastshared_tp_pack(void*state,int rank,int layer,const uint64_t*ptrs,const float*gs,const float*ds){
 using namespace fastllm_shared_flex_tp;auto*s=(State*)state;
 if(!s||rank<0||rank>=8||layer<0||layer>=(int)s->layers.size()||
    s->layers[layer][rank].gate||!ptrs||!gs||!ds)return 1;
 float*tmp=nullptr;
 try{check(cudaSetDevice(s->r[rank].device));
  size_t gb=(size_t)256*512*4096,db=(size_t)256*4096*256;
  auto get=[&](size_t n){void*p=alloc(n);s->owned[rank].push_back(p);return (uint8_t*)p;};
  auto*g=get(gb),*d=get(db),*gscale=get(gb/32),*dscale=get(db/32);
  tmp=(float*)alloc(256*192*sizeof(float));
  check(cudaMemcpy(tmp,gs,256*128*sizeof(float),cudaMemcpyHostToDevice));
  check(cudaMemcpy(tmp+256*128,ds,256*64*sizeof(float),cudaMemcpyHostToDevice));
  for(int e=0;e<256;++e){
   pack_native<<<512*4096/128,128,0,cudaStreamPerThread>>>((uint8_t*)ptrs[2*e],tmp+e*128,g+(size_t)e*512*4096,gscale+(size_t)e*512*4096/32,512,4096,true);
   pack_native<<<4096*256/128,128,0,cudaStreamPerThread>>>((uint8_t*)ptrs[2*e+1],tmp+256*128+e*64,d+(size_t)e*4096*256,dscale+(size_t)e*4096*256/32,4096,256,false);
  }
  check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaFree(tmp));tmp=nullptr;
  if(fastshared_tp_bind(state,rank,layer,g,d,gscale,dscale))return 1;
  // Only owned buffers may be coarsened. Borrowed bind() weights keep their
  // original numerics and fallback. Encode/allocate before changing weights.
  if(s->gate128Supported){
    CUtensorMap map; CUtensorMap *deviceMap=nullptr;
    cuuint64_t dims[]={128,512,32,256},strides[]={4096,128,512*4096};
    cuuint32_t box[]={128,32,8,1},steps[]={1,1,1,1};
    if(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,4,g,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS &&
        cudaMalloc(&deviceMap,sizeof(map))==cudaSuccess &&
        cudaMemcpy(deviceMap,&map,sizeof(map),cudaMemcpyHostToDevice)==cudaSuccess){
      s->layers[layer][rank].gate128map=deviceMap;
      coarsen_gate_weights<<<256*16*32*2,256,0,cudaStreamPerThread>>>(g,gscale);
      check(cudaGetLastError());check(cudaStreamSynchronize(cudaStreamPerThread));
    }else{cudaFree(deviceMap);cudaGetLastError();}
  }
  return 0;
 }catch(const std::exception&e){if(tmp)cudaFree(tmp);fprintf(stderr,"Shared Flex TP pack: %s\n",e.what());return 1;}
}

extern "C" int fastshared_tp_supported(const int*devices,int ranks) {
 using namespace fastllm_shared_flex_tp;
 if(!devices||ranks!=8)return 0;
 int previous=0; if(cudaGetDevice(&previous)!=cudaSuccess)return 0;
 bool ok=true;
 for(int r=0;r<ranks&&ok;++r){
  cudaDeviceProp prop{};ok=cudaGetDeviceProperties(&prop,devices[r])==cudaSuccess &&
   prop.major==10 && prop.minor==3 && prop.multiProcessorCount==148 && prop.sharedMemPerBlockOptin>=73728;
  for(int q=0;q<ranks&&ok;++q)if(q!=r){int access=0;ok=devices[q]!=devices[r]&&cudaDeviceCanAccessPeer(&access,devices[r],devices[q])==cudaSuccess&&access;}
  if(!ok||cudaSetDevice(devices[r])!=cudaSuccess){ok=false;break;}
  cudaFuncAttributes attributes{};
  ok=ok&&cudaFuncGetAttributes(&attributes,single_matmul<true>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,multi_gate<>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,multi_down<>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,warp_down<true>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,warp_down<false>)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,warp_receive)==cudaSuccess;
  ok=ok&&cudaFuncGetAttributes(&attributes,down_finish)==cudaSuccess;
 }
 cudaGetLastError();cudaSetDevice(previous);return ok?1:0;
}

extern "C" int fastshared_tp_dense_supported(void *state) {
    return state && static_cast<fastllm_shared_flex_tp::State*>(state)->denseSupported;
}
extern "C" int fastshared_tp_dense_admit(void *state,int rank,int rows,int eligible) {
    using namespace fastllm_shared_flex_tp;
    if(!state || rank<0 || rank>=R)return 0;
    auto &s=*static_cast<State*>(state);std::unique_lock<std::mutex> lock(s.denseMutex);
    const uint64_t generation=s.denseGeneration;
    if(!s.denseArrived){s.denseRows=rows;s.denseEligible=s.denseSupported && rows>=1 && rows<=64;s.denseRanks=0;}
    s.denseEligible=s.denseEligible && eligible && rows==s.denseRows && !(s.denseRanks&(1u<<rank));
    s.denseRanks|=1u<<rank;
    if(++s.denseArrived==R) {
        s.denseResults[generation&1]=s.denseEligible;s.denseArrived=0;++s.denseGeneration;
        s.denseCondition.notify_all();
    } else s.denseCondition.wait(lock,[&]{return s.denseGeneration!=generation;});
    return s.denseResults[generation&1];
}
extern "C" int fastshared_tp_run_dense(void *state,int rank,int rows,const void *input,
    const void *gate,const void *up,const void *down,const float *gamma,float eps,void *silu,void *output) {
    using namespace fastllm_shared_flex_tp;
    if(!fastshared_tp_dense_supported(state) || rank<0 || rank>=R || rows<1 || rows>64 || !std::isfinite(eps) || eps<=0)return 1;
    const void *pointers[]={input,gate,up,down,gamma,silu,output};
    for(const void *p:pointers)if(!p || reinterpret_cast<uintptr_t>(p)%16)return 1;
    FastllmMoePreparedInput::Range ranges[]={{input,size_t(rows)*H*2},{gate,size_t(DenseI)*H*2},
        {up,size_t(DenseI)*H*2},{down,size_t(H)*DenseI*2},{silu,size_t(rows)*DenseI*2},{output,size_t(rows)*H*2},{gamma,H*sizeof(float)}};
    for(int i=0;i<7;++i) {
        if(uintptr_t(ranges[i].data)>UINTPTR_MAX-ranges[i].bytes)return 1;
        for(int j=0;j<i;++j)if(!FastllmMoePreparedInput::Disjoint(ranges[i],ranges[j]))return 1;
    }
    const auto base=static_cast<State*>(state)->r[rank];
    auto stream=cudaStreamPerThread;
    for(int first=0;first<rows;first+=MAX_M) {
        auto a=base;a.rows=rows-first<MAX_M?rows-first:MAX_M;
        auto *x=static_cast<const __nv_bfloat16*>(input)+size_t(first)*H;
        auto *middle=static_cast<__nv_bfloat16*>(silu)+size_t(first)*DenseI;
        auto *result=static_cast<__nv_bfloat16*>(output)+size_t(first)*H;
        auto *g=static_cast<const __nv_bfloat16*>(gate),*u=static_cast<const __nv_bfloat16*>(up),*d=static_cast<const __nv_bfloat16*>(down);
        dense_up_gate<<<DenseI/16,256,DenseUpBytes,stream>>>(a,x,g,u,gamma,eps,middle);
        auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
        // No grid barrier is used. The residency probe and the existing
        // exclusive ordered-stream contract admit the peer waits, so this
        // projection can use an ordinary kernel node.
        dense_down_reduce<<<H/64,256,DenseDownBytes,stream>>>(a,middle,d,result);
        status=cudaGetLastError();if(status!=cudaSuccess)return status;
    }
    return cudaGetLastError();
}
