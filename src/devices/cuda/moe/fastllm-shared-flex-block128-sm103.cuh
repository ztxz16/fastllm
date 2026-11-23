#pragma once
// Block-128 prepared gate for SM103. Page pipeline and warp-MMA fragments are
// adapted from NaiveRT 79d5704a (Copyright 2026 NaiveAI, Apache-2.0).
// See third_party/naivert_flex/{LICENSE,NOTICE,UPSTREAM.json}.
// Original row-major weights are retained. Two adjacent groups16 share the
// MXFP8/32 output epilogue; routing keeps FastLLM's exact selector semantics.
#include <cute/arch/mma_sm80.hpp>
#include <cuda/ptx>
namespace fastllm_shared_flex_tp {
constexpr int Block128GateBytes=184448;
struct Block128GateArgs {
    Rank rank;
    const float *scores, *bias;
    int *expert_indices;
    float *expert_probs;
    int needNorm;
    float route_scale;
    const CUtensorMap *weightMap;
    const uint8_t *weightScales, *prepared;
};
namespace block128_gate_detail {
// The fixed 256-expert gate avoids generic bounds/owner branches and keeps
// the ordered probability sum in a register. Hardware reductions preserve
// every FP32 key bit. Equal keys use the existing legacy-list selector.
template<bool UseShortlist=false>
__device__ __forceinline__ void SelectGateRoutes(const float *input,const float *bias,
    int *ids,float *prob,int needNorm,float routeScale,fastllm::cuda::RouterTopKScratch<8> &scratch) {
    if constexpr (UseShortlist) {
        if (SelectShortlist(input,bias,ids,prob,needNorm,routeScale,scratch)) return;
    }
    constexpr unsigned Mask=0xffffffffu;
    const int lane=threadIdx.x%32,first=lane*8;
    float key[8],values[8];fastllm::cuda::LoadRouter8(input+first,values);
#pragma unroll
    for(int j=0;j<8;++j){const float v=values[j]+(bias?bias[first+j]:0.f);key[j]=isfinite(v)?v:-FLT_MAX;}
    float previous=0.f,sum=needNorm?0.f:1.f;bool ties=false;
// Keep the eight selections rolled to limit the gate instruction footprint.
#pragma unroll 1
    for(int rank=0;rank<8;++rank){
        float best=key[0];
#pragma unroll
        for(int j=1;j<8;++j)best=fmaxf(best,key[j]);
        unsigned bits=__float_as_uint(best==0.f?0.f:best);
        unsigned ordered=(bits&0x80000000u)?~bits:(bits^0x80000000u);
        ordered=__reduce_max_sync(Mask,ordered);
        best=__uint_as_float((ordered&0x80000000u)?(ordered^0x80000000u):~ordered);
        int index=256;
#pragma unroll
        for(int j=0;j<8;++j)if(key[j]==best)index=min(index,first+j);
        index=__reduce_min_sync(Mask,index);
        if(lane==0){ids[rank]=index;
            const float p=input[index],finite=isfinite(p)?p:0.f;
            scratch.probabilities[rank]=finite;if(needNorm)sum+=finite;}
        ties|=rank>0 && best==previous;previous=best;
#pragma unroll
        for(int j=0;j<8;++j)if(first+j==index)key[j]=-FLT_MAX;
    }
#pragma unroll
    for(int j=0;j<8;++j)ties|=key[j]==previous;
    if(__any_sync(Mask,ties)){
        fastllm::cuda::SelectRouterTopK<256,8>(input,bias,ids,prob,bias!=nullptr,needNorm,routeScale,scratch);
        return;
    }
    __syncwarp();
    if(lane==0 && needNorm && (!isfinite(sum)||fabsf(sum)<1.e-20f))sum=1.f;
    sum=__shfl_sync(Mask,sum,0);
    if(lane<8)prob[lane]=scratch.probabilities[lane]/sum*routeScale;
}

template<int Capacity=8, bool Primary=false> struct Body {
  static_assert(!Primary || Capacity==8);
  static_assert(Capacity==8 || Capacity==16 || Capacity==32 || Capacity==64);
  static constexpr bool Compact=Capacity>8;
  static constexpr int kRows8BarrierOffset=0x0000,kRows8ReleaseBarrierOffset=0x0040;
  static constexpr int kRows8ActivationBarrierOffset=0x0080,kRows8RouteBarrierOffset=0x00c0;
  static constexpr int kRows8RouteReleaseBarrierOffset=0x00e0,kRows8SelectedOffset=0x0100;
  static constexpr int kRows8DistinctOffset=Capacity==64?0x0900:Capacity>=32?0x0500:Compact?0x0300:0x0200;
  static constexpr int kRows8MaskOffset=Capacity==64?0x1100:Capacity>=32?0x0900:Compact?0x0500:0x0300;
  static constexpr int kRows8CountOffset=Capacity==64?0x1140:Capacity>=32?0x0940:Compact?0x0540:0x0340;
  static constexpr int kRows8BiasOffset=Capacity==64?0x1200:Capacity>=32?0x0a00:Compact?0x0600:0x0400;
  static constexpr int kRows8SigmoidOffset=Capacity==64?0x2000:Capacity>=32?0x1000:Compact?0x0a00:0x0800;
  static constexpr int kRows8MemberCountOffset=kRows8BiasOffset+Capacity*8*sizeof(float);
  static_assert(!Compact || kRows8MemberCountOffset+256<=kRows8SigmoidOffset);
  // Task count plus two sets of eight warp costs share this metadata region.
  static_assert(kRows8CountOffset+17*sizeof(int)<=kRows8BiasOffset);
  using Job = typename std::conditional<Capacity==64,uint16_t,uint8_t>::type;
  static constexpr int MemberTile=Capacity==64?32:16;
  static_assert(Capacity*8<=(size_t(1)<<(sizeof(Job)*8)), "Compact jobs fit in Job");
  static constexpr int kRows8RoutingSlotOffset=Capacity==64?0x4000:Capacity>=32?0x9000:Compact?0x4a00:0x2800;
  static constexpr int kRows8PaneBaseOffset=Capacity==64?0xe000:Capacity>=32?0xc000:Compact?0x6400:0x4000,kRows8PaneStride=33792;
  static constexpr int kRows8ActivationOffset=Capacity==64?0x2f000:Capacity>=32?0x2d000:Compact?0x27400:0x25000,kRows8ActivationStride=4112;
  static constexpr int kRows8Groups=16,kRows8Selected=8;
  static constexpr int kRows8Experts=256,kRows8Pages=4;
  static constexpr int kRows8PageK=1024,kRows8PackedRows=32,ScaleScratch=Capacity==64?0xc000:Capacity>=32?0xb000:Compact?0x5a00:0x3800;

  using Trans64=cutlass::arch::ClusterTransactionBarrier;
  __device__ static __forceinline__ unsigned decode_fp8_pair(unsigned value) {
    const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(value), __NV_E4M3);
    unsigned bits;
    memcpy(&bits, &pair, sizeof(bits));
    return bits;
  }

  __device__ static __forceinline__ void mma_with_decoded_activation(
      float& p0, float& p1, float& p2, float& p3, const uint4& a,
      unsigned b0lo, unsigned b1lo, unsigned b0hi, unsigned b1hi) {
    using Mma = cute::SM80_16x8x16_F32F16F16F32_TN;
    Mma::fma(p0,p1,p2,p3,decode_fp8_pair(a.x),decode_fp8_pair(a.y),
        decode_fp8_pair(a.z),decode_fp8_pair(a.w),b0lo,b1lo,p0,p1,p2,p3);
    Mma::fma(p0,p1,p2,p3,decode_fp8_pair(a.x>>16),decode_fp8_pair(a.y>>16),
        decode_fp8_pair(a.z>>16),decode_fp8_pair(a.w>>16),b0hi,b1hi,p0,p1,p2,p3);
  }

  __device__ static __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 4,256;" ::: "memory");
  }

  __device__ static __forceinline__ void slot_sync(int item_slot) {
    asm volatile("bar.sync %0,128;" :: "r"(item_slot+9) : "memory");
  }

  __device__ static __forceinline__ std::uint64_t* barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8BarrierOffset);
  }

  __device__ static __forceinline__ std::uint64_t* release_barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8ReleaseBarrierOffset);
  }

  __device__ static __forceinline__ std::uint64_t* activation_barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8ActivationBarrierOffset);
  }

  __device__ static __forceinline__ std::uint64_t* route_barrier(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8RouteBarrierOffset);
  }

  __device__ static __forceinline__ std::uint64_t* route_release_barrier(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8RouteReleaseBarrierOffset);
  }

  __device__ static __forceinline__ int* selected(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8SelectedOffset);
  }

  __device__ static __forceinline__ int* distinct(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8DistinctOffset);
  }

  __device__ static __forceinline__ unsigned* expert_mask(unsigned char* shared) {
    return reinterpret_cast<unsigned*>(shared + kRows8MaskOffset);
  }

  __device__ static __forceinline__ int* distinct_count(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8CountOffset);
  }

  __device__ static __forceinline__ float* bias_scratch(unsigned char* shared) {
    return reinterpret_cast<float*>(shared + kRows8BiasOffset);
  }

  __device__ static __forceinline__ float* sigmoid(unsigned char* shared, int token) {
    return reinterpret_cast<float*>(shared + kRows8SigmoidOffset) + token * kRows8Experts;
  }

  __device__ static __forceinline__ unsigned char* pane(unsigned char* shared, int index) {
    return shared + kRows8PaneBaseOffset + index * kRows8PaneStride;
  }

  __device__ static __forceinline__ __nv_fp8_e4m3* activation(unsigned char* shared, int token) {
    return reinterpret_cast<__nv_fp8_e4m3*>(shared + kRows8ActivationOffset +
                                            token * kRows8ActivationStride);
  }

  __device__ static __forceinline__ int item_of(int local, int item_slot, int items,unsigned char*shared) {
    if constexpr(Capacity==64){
      const int workers=distinct_count(shared)[1];
      return int(blockIdx.x)>=workers?items:2*int(blockIdx.x)+2*local*workers+item_slot;
    }
    const int pairs=items/2;
    const int waves=(pairs+int(gridDim.x)-1)/int(gridDim.x);
    const int workers=waves?(pairs+waves-1)/waves:1;
    return int(blockIdx.x)>=workers?items:2*int(blockIdx.x)+2*local*workers+item_slot;
  }

  __device__ static void compute_routes_warp(const Block128GateArgs& args,
                                             unsigned char* shared, int token) {
    if (token >= args.rank.rows) return;
    using Scratch = fastllm::cuda::RouterTopKScratch<8>;
    auto &scratch = reinterpret_cast<Scratch*>(shared + kRows8PaneBaseOffset)[token];
    static_assert(sizeof(Scratch) * Capacity < 4 * kRows8PaneStride);
    SelectGateRoutes<Primary>(sigmoid(shared,token),args.bias?bias_scratch(shared):nullptr,
        selected(shared)+token*8,scratch.probabilities,args.needNorm,args.route_scale,scratch);
    const int lane = threadIdx.x % 32;
    if (blockIdx.x == 0 && lane < 8) {
      args.expert_indices[token*8+lane] = selected(shared)[token*8+lane];
      args.expert_probs[token*8+lane] = scratch.probabilities[lane];
    }
  }

  __device__ static void build_item_list(unsigned char* shared, int rows) {
    const int tid = int(threadIdx.x);
    // One expert's membership mask per consumer, before the first join.
    for(int w=0;w<Capacity/8;++w)
      reinterpret_cast<unsigned long long*>(shared + kRows8RoutingSlotOffset)[tid*(Capacity/8)*sizeof(Job)+w] = 0ull;
    if constexpr(Capacity<32) if (tid < rows*8) {
      using Scratch = fastllm::cuda::RouterTopKScratch<8>;
      reinterpret_cast<float*>(shared+kRows8BiasOffset)[tid] =
          reinterpret_cast<Scratch*>(shared+kRows8PaneBaseOffset)[tid/8].probabilities[tid%8];
    }
    unsigned* mask = expert_mask(shared);
    if (tid < 8) {
      mask[tid] = 0u;
    }
    consumer_sync();
    for (int job=tid;job<rows*kRows8Selected;job+=256) {
      const int expert = selected(shared)[job];
      atomicOr(mask + (expert >> 5), 1u << (expert & 31));
      // Repeated experts in non-finite routing set multiple bits safely.
      atomicOr(reinterpret_cast<unsigned*>(shared + kRows8RoutingSlotOffset)
                   + expert * (Capacity/4)*sizeof(Job) + (job >> 5), 1u << (job & 31));
      if constexpr(Capacity<=32) break; // These tiles contain at most 256 jobs.
    }
    consumer_sync();
    {
      const int expert = tid;   // 256 consumer threads, one per expert
      const int word = expert >> 5;
      const unsigned bits = mask[word];
      int rank = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        rank += w < word ? __popc(mask[w]) : 0;
      }
      if ((bits >> (expert & 31)) & 1u) {
        rank += __popc(bits & ((1u << (expert & 31)) - 1u));
        distinct(shared)[rank] = expert;
      }
      if (tid == 0) {
        int total = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total += __popc(mask[w]);
        }
        *distinct_count(shared) = total;
      }
    }
    if constexpr (Compact) {
      // Load the complete routing bitset before reusing it for compact job IDs.
      auto *jobs = reinterpret_cast<Job*>(shared + kRows8RoutingSlotOffset) + tid * Capacity;
      unsigned bits[Capacity / 4];
#pragma unroll
      for (int w = 0; w < Capacity / 4; ++w)
        bits[w] = reinterpret_cast<const unsigned *>(jobs)[w];
      int count = 0;
#pragma unroll
      for (int w = 0; w < Capacity / 4; ++w) {
        unsigned mask = bits[w];
        while (mask) {
          if (count < Capacity) jobs[count] = w * 32 + __ffs(mask) - 1;
          ++count;
          mask &= mask - 1;
        }
      }
      for (int i = min(count, Capacity); i < Capacity; ++i) jobs[i] = Job(-1);
      // The unused half of the old bias tile needs no extra shared storage.
      shared[kRows8MemberCountOffset + tid] = count;
    }
    consumer_sync();
  }

  __device__ static void split_expert_work(unsigned char*shared) {
    const int t=threadIdx.x,lane=t%32,warp=t/32;
    const int count=member_count(shared,t),tasks=(count+MemberTile-1)/MemberTile;
    int inclusive=tasks;
#pragma unroll
    for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,inclusive,d);if(lane>=d)inclusive+=v;}
    int *warpCounts=distinct_count(shared)+1;
    if(lane==31)warpCounts[warp]=inclusive;
    consumer_sync();
    int prefix=inclusive-tasks;
#pragma unroll
    for(int w=0;w<8;++w)if(w<warp)prefix+=warpCounts[w];
    #pragma unroll
    for(int group=0;group<(Capacity+MemberTile-1)/MemberTile;++group)
      if(group<tasks)distinct(shared)[prefix+group]=t+group*256;
    if(t==255)*distinct_count(shared)=prefix+tasks;
    consumer_sync();
  }

  // Model both schedules using active N8 groups, not just the number of
  // output tiles. All CTAs make the same choice from their local route copy.
  __device__ static void choose_workers(unsigned char*shared) {
    const int t=threadIdx.x,pairs=*distinct_count(shared)*8;
    const int waves=(pairs+int(gridDim.x)-1)/int(gridDim.x);
    const int balanced=waves?(pairs+waves-1)/waves:1;
    int maximum=__reduce_max_sync(0xffffffff,member_count(shared,t));
    if(t%32==0)distinct_count(shared)[1+t/32]=maximum;
    consumer_sync();
    maximum=0;
#pragma unroll
    for(int w=0;w<8;++w)maximum=max(maximum,distinct_count(shared)[1+w]);
    consumer_sync();
    // With at most two N8 groups, weight traffic dominates; keep the even
    // wave schedule. Wider experts benefit from balancing active MMA work.
    if(maximum<=16) {
      if(t==0)distinct_count(shared)[1]=balanced;
      consumer_sync();return;
    }
    int cost[2]={};
#pragma unroll
    for(int mode=0;mode<2;++mode) {
      const int workers=mode?int(gridDim.x):balanced;
      if(t<workers)for(int pair=t;pair<pairs;pair+=workers) {
        const int work=distinct(shared)[pair/8],expert=work&255;
        const int count=min(MemberTile,member_count(shared,expert)-(work>>8)*MemberTile);
        // Two fixed units approximate page/epilogue work; each active
        // N8 group adds one MMA unit. Only a lower worst-CTA cost opts in.
        cost[mode]+=2+(count+7)/8;
      }
      cost[mode]=__reduce_max_sync(0xffffffff,cost[mode]);
      if(t%32==0)distinct_count(shared)[1+mode*8+t/32]=cost[mode];
    }
    consumer_sync();
    if(t<32) {
      int oldCost=t<8?distinct_count(shared)[1+t]:0;
      int fullCost=t<8?distinct_count(shared)[9+t]:0;
      oldCost=__reduce_max_sync(0xffffffff,oldCost);
      fullCost=__reduce_max_sync(0xffffffff,fullCost);
      if(t==0)distinct_count(shared)[1]=fullCost<oldCost?int(gridDim.x):balanced;
    }
    consumer_sync();
  }

  __device__ static __forceinline__ void load_weight_page(unsigned char *destination,
      const CUtensorMap *weightMap, std::uint64_t *barrier, int group, int page, int expert) {
    unsigned dst=unsigned(__cvta_generic_to_shared(destination));
    unsigned bar=unsigned(__cvta_generic_to_shared(barrier));
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;":"=l"(policy));
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3,%4}],[%5],%6;"
      ::"r"(dst),"l"(weightMap),"r"(group*32),"r"(page*8),"r"(expert),"r"(bar),"l"(policy):"memory");
  }

  __device__ static void helper_pages(unsigned char*shared,const CUtensorMap*weightMap,const uint8_t*weightScales) {
    const int t=threadIdx.x,lane=t%32;
    if(t<256)return;
    const int helper=(t-256)/32,item_slot=helper/2,page_parity=helper%2;
    auto*barrier=barriers(shared)+helper;
    while(!Trans64::try_wait(route_release_barrier(shared),0u)){}
    const int items=*distinct_count(shared)*kRows8Groups;
    for(int local=0;;++local){
      const int item=item_of(local,item_slot,items,shared);if(item>=items)break;
      const int work=distinct(shared)[item/16],expert=Capacity>=32?(work&255):work,group=item%16;
      for(int page=page_parity;page<4;page+=2){
        const int sequence=local*2+page/2;
        while(!Trans64::try_wait(release_barriers(shared)+helper,sequence&1)){}
        if constexpr (Primary) {
          if (lane == 0) {
            Trans64::expect_transaction(barrier,32768);
            load_weight_page(pane(shared,helper),weightMap,barrier,group,page,expert);
          }
        }
        if(lane<16){
          const int branch=lane/8,sg=lane%8;
          const unsigned e=weightScales[fastllm_shared_flex_tp::sf_offset(expert,
              group*32+branch*8,(page*8+sg)*4,512,4096)];
          reinterpret_cast<float*>(pane(shared,helper)+32768)[lane]=fastllm_shared_flex_tp::sf(e);
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncwarp();
        if(lane==0){
          if constexpr (Primary) {
            // Publish the page once both the weight transfer and the scale stores finish.
            Trans64::arrive(barrier);
          } else {
            Trans64::arrive_and_expect_tx(barrier,32768);
            load_weight_page(pane(shared,helper),weightMap,barrier,group,page,expert);
          }
        }
      }
    }
  }

  __device__ static __forceinline__ void wait_weight_page(unsigned char* shared, int item_slot,
                                                          int local, int page) {
    const int pane_slot = item_slot * 2 + (page & 1);
    const int sequence = local * (kRows8Pages / 2) + (page >> 1);
    while (!Trans64::try_wait(barriers(shared) + pane_slot, std::uint32_t(sequence & 1))) {
    }
  }

  __device__ static __forceinline__ void release_weight_page(unsigned char* shared,
                                                             int item_slot, int page) {
    const int pane_slot = item_slot * 2 + (page & 1);
    Trans64::arrive(release_barriers(shared) + pane_slot);
  }

  // Each expert computes only its selected rows. An N8 tile covers up to
  // eight memberships even when the input tile contains sixteen rows.
  __device__ static int member_job(unsigned char *shared,int expert,int index) {
    if constexpr(!Compact)return -1;
    const unsigned job = reinterpret_cast<const Job*>(shared+kRows8RoutingSlotOffset)[expert*Capacity+index];
    if constexpr(Capacity>=32)return index >= member_count(shared,expert) ? -1 : int(job);
    else return job == 255 ? -1 : int(job);
  }
  __device__ static int member_count(unsigned char *shared,int expert) {
    return shared[kRows8MemberCountOffset + expert];
  }
  __device__ static void accumulate_page_prepared(unsigned char* shared, int item_slot,
                                                  int page, float (&accum)[8],int sourceRow=0,int scaleRow0=0,int scaleRow1=0, const uint8_t *prepared=nullptr) {
    const int tid = int(threadIdx.x), lane = tid & 31;
    const int k_part = (tid >> 5) & 1;
    const int parity = ((tid >> 5) >> 1) & 1;
    const int lane_group = lane >> 2, lane_quad = lane & 3;
    const auto* weight = reinterpret_cast<const __nv_fp8_e4m3*>(
        pane(shared, item_slot * 2 + (page & 1)));
    const auto* scales = reinterpret_cast<const float*>(
        weight + kRows8PackedRows * kRows8PageK);
    const auto* act = (Capacity>=32
        ? reinterpret_cast<const __nv_fp8_e4m3*>(prepared+sourceRow*4112)
        : activation(shared,Compact?sourceRow:lane_group)) + page*kRows8PageK+lane_quad*4;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int sg = 2 * g + parity;
      float p[2][4] = {};
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
        const int block = sg * 4 + k_part + sp * 2;
        const unsigned b0 = *reinterpret_cast<const unsigned*>(act + block * 32);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(act + block * 32 + 16);
        const unsigned b0lo = decode_fp8_pair(b0), b1lo = decode_fp8_pair(b1);
        const unsigned b0hi = decode_fp8_pair(b0 >> 16), b1hi = decode_fp8_pair(b1 >> 16);
#pragma unroll
        for (int branch = 0; branch < 2; ++branch) {
          // Each eight-lane address group supplies one 8x8 b16 tile.
          // Four tiles match the original four packed FP8 fragment words.
          const int tile=lane/8,row=branch*8+lane%8+(tile%2)*16;
          const int k=block*32+(tile/2)*16;
          const int offset=(k/128)*4096+row*128+k%128;
          const unsigned address=unsigned(__cvta_generic_to_shared(weight+(offset^((row%8)*16))));
          uint4 a;
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
              :"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(address));
          mma_with_decoded_activation(p[branch][0], p[branch][1], p[branch][2], p[branch][3],
                                      a, b0lo, b1lo, b0hi, b1hi);
        }
      }
      const auto* input_scales = reinterpret_cast<const float*>(
          shared + ScaleScratch) + (page * 8 + sg) * Capacity;
      const float as0 = input_scales[Compact?scaleRow0:lane_quad * 2];
      const float as1 = input_scales[Compact?scaleRow1:lane_quad * 2 + 1];
#pragma unroll
      for (int branch = 0; branch < 2; ++branch) {
        const float ws = scales[branch * 8 + sg];
        const float scale0 = ws * as0, scale1 = ws * as1;
        const int stream = branch * 4;
        accum[stream + 0] += p[branch][0] * scale0;
        accum[stream + 1] += p[branch][1] * scale1;
        accum[stream + 2] += p[branch][2] * scale0;
        accum[stream + 3] += p[branch][3] * scale1;
      }
    }
  }

  __device__ static void finish_item_prepared(unsigned char* shared, int item_slot, int expert,
                                             int group, const float (&accum)[8], uint8_t*outputValues, uint8_t*outputScale,int firstMember=0) {
    const int tid = int(threadIdx.x), lane = tid & 31, k_part = (tid >> 5) & 1;
    const int parity = ((tid >> 5) >> 1) & 1;
    const int lane_group = lane >> 2, lane_quad = lane & 3;
    // Every consumer acquired route_release before the item loop. Routing has
    // finished reading the score area; reuse it as [item][branch][K part][parity][128].
    // The two existing slot joins publish these partials and protect their reuse.
    float* values = reinterpret_cast<float*>(shared + kRows8SigmoidOffset) + item_slot * 8 * 128;
#pragma unroll
    for (int branch = 0; branch < 2; ++branch) {
      float* mine = values + ((branch * 2 + k_part) * 2 + parity) * 128;
      const int base = branch * 4;
      mine[lane_group * 8 + lane_quad * 2] = accum[base];
      mine[lane_group * 8 + lane_quad * 2 + 1] = accum[base + 1];
      mine[(lane_group + 8) * 8 + lane_quad * 2] = accum[base + 2];
      mine[(lane_group + 8) * 8 + lane_quad * 2 + 1] = accum[base + 3];
    }
    slot_sync(item_slot);
    const int local = tid - item_slot * 128;
    const int output = local >> 3, token = local & 7, index = output * 8 + token;
    // Preserve parity addition before K-part addition, including FP32 rounding.
    const float g0 = values[index] + values[128 + index];
    const float g1 = values[256 + index] + values[384 + index];
    const float u0 = values[512 + index] + values[640 + index];
    const float u1 = values[768 + index] + values[896 + index];
    const float gate = __bfloat162float(__float2bfloat16_rn(g0+g1));
    const float up = __bfloat162float(__float2bfloat16_rn(u0+u1));
    const float activated = gate / (1.0f + __expf(-gate));
    int job;
    if constexpr(Compact)job=member_job(shared,expert,firstMember+token);
    else {const unsigned bits=shared[kRows8RoutingSlotOffset+expert*8+token];
      const int slot=31-__clz(bits);job=slot>=0?token*8+slot:-1;}
    const float probability=job>=0?reinterpret_cast<float*>(shared+kRows8BiasOffset)[job]:0.f;
    const float result = __fmul_rn(__fmul_rn(activated, up), probability);
    // Paired items always belong to the same expert and adjacent groups16.
    // Both 128-thread slots finish reading partials before the area is reused.
    consumer_sync();
    float* merged = reinterpret_cast<float*>(shared+kRows8SigmoidOffset);
    merged[token*32+item_slot*16+output] = result;
    consumer_sync();
    const int row=tid/32, col=tid%32;
    int selectedJob;
    if constexpr(Compact)selectedJob=member_job(shared,expert,firstMember+row);
    else {const unsigned chosen=shared[kRows8RoutingSlotOffset+expert*8+row];
      const int slot=31-__clz(chosen);selectedJob=slot>=0?row*8+slot:-1;}
    float v=merged[tid],mx=fabsf(v);
    for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    const unsigned e=fastllm_shared_flex_tp::exponent(mx);
    if(selectedJob>=0){
      const int i=selectedJob*256+(group/2)*32+col;
      outputValues[i]=__nv_fp8_e4m3(v*fastllm_shared_flex_tp::sf(254-e)).__x;
      if(col==0)outputScale[i/32]=e;
    }
    consumer_sync();
  }

  __device__ static void stage_inputs(const Block128GateArgs &args,unsigned char*shared) {
    const int t=threadIdx.x;
    if(t>=256){
      if(t==256){
        auto*route=route_barrier(shared);
        if constexpr(Capacity>=32) {
          Trans64::arrive_and_expect_tx(route,args.rank.rows*64);
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
              selected(shared),args.expert_indices,args.rank.rows*32,route);
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
              bias_scratch(shared),args.expert_probs,args.rank.rows*32,route);
        } else {
        Trans64::arrive_and_expect_tx(route,args.rank.rows*1024+(args.bias?1024:0));
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
            sigmoid(shared,0),args.scores,args.rank.rows*1024,route);
        if(args.bias)cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
            bias_scratch(shared),args.bias,1024,route);
        }
        auto*ready=activation_barriers(shared);
        Trans64::arrive_and_expect_tx(ready,Capacity*(Capacity>=32?128:4112+128));
        if constexpr(Capacity<32)
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
            activation(shared,0),args.prepared,Capacity*4112,ready);
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{},cuda::ptx::space_global_t{},
            shared+ScaleScratch,args.prepared+Capacity*4112,Capacity*128,ready);
      }
      return;
    }
    while(!Trans64::try_wait(route_barrier(shared),0u)){}
  }
  __device__ static void publish_routes(const Block128GateArgs &args,unsigned char*shared) {
    if(blockIdx.x!=0)return;
    const int t=threadIdx.x,rows=args.rank.rows,n=*distinct_count(shared);
    if constexpr(Capacity>=32)if(t<32) {
      unsigned maximum=0;
#pragma unroll
      for(int e=t;e<256;e+=32)maximum=max(maximum,unsigned(member_count(shared,e)));
      maximum=__reduce_max_sync(0xffffffff,maximum);
      if(t==0)*args.rank.count=maximum;
    }
    // Only the narrow down/fallback kernels consume the flat job list.
    // Compact down uses the expert membership lists below directly.
    if constexpr(!Compact){
      if(t<rows*8)args.rank.jobs[t]=t;
      if(t==0)*args.rank.count=rows*8;
    }
    if(t==0)*args.rank.nuniq=n;
    if(t<n){
      int expert=distinct(shared)[t],count=0;
      args.rank.uniq[t]=expert;
      if constexpr(Compact) {
        count=member_count(shared,expert);
        for(int j=0;j<count;++j)args.rank.members[t*Capacity+j]=member_job(shared,expert,j);
      } else {
        for(int j=0;j<rows*8;++j)if(selected(shared)[j]==expert)
          args.rank.members[t*Capacity+count++]=j;
      }
      args.rank.rowcounts[t]=count;
    }
  }
  __device__ static void run(const Block128GateArgs &args,unsigned char*shared) {
    const int t=threadIdx.x;
    if(t<4){
      Trans64::init(barriers(shared)+t,1u);
      Trans64::init(release_barriers(shared)+t,128u);
      (void)cuda::ptx::mbarrier_arrive(release_barriers(shared)+t,128u);
      if(t<2)Trans64::init(activation_barriers(shared)+t,1u);
    }
    if(t==0){Trans64::init(route_barrier(shared),1u);Trans64::init(route_release_barrier(shared),256u);}
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    stage_inputs(args,shared);
    if(t>=256){helper_pages(shared,args.weightMap,args.weightScales);return;}
    if constexpr(Capacity<32)
      for(int token=t/32;token<Capacity;token+=8)compute_routes_warp(args,shared,token);
    consumer_sync();
    build_item_list(shared,args.rank.rows);
    publish_routes(args,shared);
    if constexpr (Primary) {
      if (blockIdx.x == 0) {
        // Join every route writer before publishing to Down. The prepared
        // router clears gcount before this launch; the FP8 Gate uses no splits.
        consumer_sync();
        if (t == 0) {
          asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
                       "l"(args.rank.gcount), "r"(1u) : "memory");
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }
    }
    if constexpr(Capacity>=32) {
      consumer_sync();split_expert_work(shared);
      if constexpr(Capacity==64)choose_workers(shared);
    }
    while(!Trans64::try_wait(activation_barriers(shared),0u)){}
    Trans64::arrive(route_release_barrier(shared));
    while(!Trans64::try_wait(route_release_barrier(shared),0u)){}
    const int items=*distinct_count(shared)*16,item_slot=t/128;
    for(int local=0;;++local){
      const int item=item_of(local,item_slot,items,shared);if(item>=items)break;
      const int work=distinct(shared)[item/16],expert=Capacity>=32?(work&255):work,group=item%16;
      constexpr int RowGroups=Capacity>=32?MemberTile/8:Capacity/8;
      const int firstMember=Capacity>=32?(distinct(shared)[item/16]>>8)*MemberTile:0;
      float accum[RowGroups][8]{};
      const int groups=Compact?(Capacity>=32?
          (min(MemberTile,member_count(shared,expert)-firstMember)+7)/8:(member_count(shared,expert)+7)/8):1;
      int source[RowGroups],scale0[RowGroups],scale1[RowGroups];
      if constexpr(Compact)
#pragma unroll
      for(int q=0;q<RowGroups;++q)if(q<groups){
        source[q]=max(0,member_job(shared,expert,firstMember+q*8+(t%32)/4)/8);
        scale0[q]=max(0,member_job(shared,expert,firstMember+q*8+(t%4)*2)/8);
        scale1[q]=max(0,member_job(shared,expert,firstMember+q*8+(t%4)*2+1)/8);
      }
      for(int page=0;page<4;++page){
        wait_weight_page(shared,item_slot,local,page);
        #pragma unroll
        for(int q=0;q<RowGroups;++q)if(q<groups){
          if constexpr(Compact)accumulate_page_prepared(shared,item_slot,page,accum[q],source[q],scale0[q],scale1[q],args.prepared);
          else accumulate_page_prepared(shared,item_slot,page,accum[q]);
        }
        release_weight_page(shared,item_slot,page);
      }
      #pragma unroll
      for(int q=0;q<RowGroups;++q)if(q<groups)finish_item_prepared(shared,item_slot,expert,group,accum[q],args.rank.mid,args.rank.ms,firstMember+q*8);
    }
  }
};
} // namespace block128_gate_detail
__global__ __launch_bounds__(384,1)
void block128_gate(const __grid_constant__ Block128GateArgs args) {
    extern __shared__ __align__(1024) unsigned char block128Arena[];
    block128_gate_detail::Body<>::run(args,block128Arena);
}
__global__ __launch_bounds__(384,1)
void block128_gate_primary(const __grid_constant__ Block128GateArgs args) {
    if (blockIdx.x) cudaTriggerProgrammaticLaunchCompletion();
    extern __shared__ __align__(1024) unsigned char block128Arena[];
    block128_gate_detail::Body<8,true>::run(args, block128Arena);
}
constexpr int Block128WideGateBytes=0x27400+16*4112;
__global__ __launch_bounds__(384,1)
void block128_wide_gate(const __grid_constant__ Block128GateArgs args) {
    extern __shared__ __align__(1024) unsigned char block128Arena[];
    block128_gate_detail::Body<16>::run(args,block128Arena);
}
// Larger row tiles keep their small prepared activations in L2. Weight pages
// still use the same shared pipeline, avoiding a shared-memory size cliff.
constexpr int Block128BatchGateBytes=0x2d000;
__global__ __launch_bounds__(384,1)
void block128_batch_gate(const __grid_constant__ Block128GateArgs args) {
    extern __shared__ __align__(1024) unsigned char block128Arena[];
    block128_gate_detail::Body<32>::run(args,block128Arena);
}
constexpr int Block128Batch64GateBytes=0x2f000;
__global__ __launch_bounds__(384,1)
void block128_batch64_gate(const __grid_constant__ Block128GateArgs args) {
    extern __shared__ __align__(1024) unsigned char arena[];
    block128_gate_detail::Body<64>::run(args,arena);
}
// Group sixteen output channels and K128. Requantization only changes scales
// by powers of two. All old scale readers retire before an in-place update.
__global__ void coarsen_gate_weights(uint8_t *weights,uint8_t *scales) {
    const int t=threadIdx.x,branch=blockIdx.x%2,q=blockIdx.x/2;
    const int kb=q%32,group=q/32%16,expert=q/(32*16);
    unsigned e=0;
    if(t<64){const int row=group*16+t/4,inter=row/8*16+row%8+branch*8;
        e=scales[sf_offset(expert,inter,kb*4+t%4,512,4096)];}
    e=__reduce_max_sync(0xffffffff,e);
    __shared__ unsigned exponents[8];
    if(t%32==0)exponents[t/32]=e;
    __syncthreads();
    e=__reduce_max_sync(0xffffffff,t%32<8?exponents[t%32]:0);
    for(int i=t;i<16*128;i+=256){
        const int row=group*16+i/128,k=kb*128+i%128,inter=row/8*16+row%8+branch*8;
        const size_t offset=(size_t(expert)*512+inter)*4096+k;
        __nv_fp8_e4m3 value;value.__x=weights[offset];
        const unsigned old=scales[sf_offset(expert,inter,k/32,512,4096)];
        weights[offset]=__nv_fp8_e4m3(float(value)*sf(max(0,127+int(old)-int(e)))).__x;
    }
    __syncthreads();
    if(t<64){const int row=group*16+t/4,inter=row/8*16+row%8+branch*8;
        scales[sf_offset(expert,inter,kb*4+t%4,512,4096)]=e;}
}
} // namespace fastllm_shared_flex_tp
