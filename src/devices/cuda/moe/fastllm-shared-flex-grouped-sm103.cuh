#pragma once
// Full-K expert grouping for 7/8-row SM103 verification. TMA reads the existing
// row-major FP8 activation image directly; membership is needed only when
// scattering the output. Weights and numerical materialization are unchanged.
// Helper/consumer page scheduling and evict-first weight policy are adapted
// from NaiveRT (Copyright 2026 NaiveAI, Apache-2.0), pinned at 79d5704a.
// See third_party/naivert_flex/{LICENSE,NOTICE,UPSTREAM.json}. This adaptation
// uses four TMA/UMMA panes and preserves FastLLM's packed MXFP8 arithmetic.
namespace fastllm_shared_flex_tp {
template<bool Subgroup=false> __device__ __forceinline__
void GroupedTmaGate(Rank a,const float*prob,const CUtensorMap *xmap){
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=H,NN=2*I,PAGES=NK/128,BUFFERS=4;
  extern __shared__ __align__(1024) uint8_t arena[];
  auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
  auto xb=reinterpret_cast<uint8_t(*)[1024]>(arena+BUFFERS*16384);
  auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024));
  auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024+512));
  __shared__ __align__(8) uint64_t loaded[BUFFERS],released[BUFFERS];
  __shared__ unsigned tm;
  int tid=threadIdx.x;
  if(blockIdx.x>=*a.nuniq*(NN/128))return;
  if(tid==0){
    for(int b=0;b<BUFFERS;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      unsigned r=(unsigned)__cvta_generic_to_shared(&released[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],1;"::"r"(l),"r"(r));}
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(64,&tm);al.release_allocation_lock();}
  MatmulSync<Subgroup>();
  auto idesc=cute::UMMA::make_instr_desc_block_scaled<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  for(int task=blockIdx.x;task<*a.nuniq*(NN/128);task+=gridDim.x){
    int tile=task%(NN/128),job=task/(NN/128);
    int expert=a.uniq[job],member_count=a.rowcounts[job];
    const uint8_t*xs=a.xs;
    const uint8_t*scales=a.gs;
    // NaiveRT's helper/consumer page protocol, adapted to block-scaled UMMA:
    // warp 1 fills a ring; warp 0 consumes it and releases panes via UMMA commit.
    // Both loops keep K pages in exactly the same arithmetic order.
    if(tid>=32 && tid<64){
      int lane=tid%32;
      for(int kg=0;kg<PAGES;++kg){
        int b=kg%BUFFERS;
        if(kg>=BUFFERS)while(!parity_wait(&released[b],((kg/BUFFERS)-1)&1)){}
        int br=lane%8;
        unsigned scale=br<a.rows?ld4(xs+br*(H/32)+kg*4):0x7f7f7f7fu;
        #pragma unroll
        for(int q=0;q<4;++q)reinterpret_cast<unsigned*>(sfb[b])[lane*4+q]=scale;
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncwarp();
        if(lane==0){
          unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(16384+512+1024):"memory");
          // Expert weights stream through once per task; do not displace
          // reusable activations and routing metadata from L2.
          uint64_t policy;
          asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;":"=l"(policy));
          unsigned weightDst=(unsigned)__cvta_generic_to_shared(wa[b]);
          asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3,%4}],[%5],%6;"
            ::"r"(weightDst),"l"(a.gmap),"r"(kg*128),"r"(tile*128),"r"(expert),"r"(mb),"l"(policy):"memory");
          bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
          unsigned d=unsigned(__cvta_generic_to_shared(xb[b]));
          asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,0}],[%3];"
              ::"r"(d),"l"(xmap),"r"(kg*128),"r"(mb):"memory");
        }
      }
    }else if(tid<32){
      for(int kg=0;kg<PAGES;++kg){
        int b=kg%BUFFERS;
        while(!parity_wait(&loaded[b],(kg/BUFFERS)&1)){}
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        auto ad=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4*)wa[b],0,0);
        auto bd=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,8,128,128>((E4*)xb[b],0,0);
        auto sd1=dg::mma::sm100::make_sf_desc(sfa[b]),sd2=dg::mma::sm100::make_sf_desc(sfb[b]);
        // Only one K page is issued at a time. Four dedicated scale columns
        // per pane prevent the producer from overwriting a pending instruction.
        unsigned sa=tm+8+b*8,sb=tm+12+b*8;
        if(tid==0){cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd1,sa);cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd2,sb);}
        #pragma unroll
        for(int kk=0;kk<4;++kk){
          auto aa=ad,bb=bd;aa.start_address_+=kk*2;bb.start_address_+=kk*2;
          auto desc=dg::mma::sm100::make_runtime_instr_desc_with_sf_id(idesc,kk,kk);
          MMA::fma(aa,bb,tm,kg!=0||kk!=0,desc,sa,sb);
        }
        cutlass::arch::umma_arrive(&released[b]);
      }
      constexpr int last=PAGES-1;
      while(!parity_wait(&released[last%BUFFERS],(last/BUFFERS)&1)){}
    }
    MatmulSync<Subgroup>();
    asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    unsigned values[8];
    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tm+(unsigned(tid/32*32)<<16),
        values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]);
    cutlass::arch::fence_view_async_tmem_load();
    // All K pages have retired; this pane can now hold the eight output rows.
    float *scratch=reinterpret_cast<float*>(wa[0]);
    #pragma unroll
    for(int col=0;col<8;++col)if(col<a.rows)
      scratch[col*128+tid]=__uint_as_float(values[col]);
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    MatmulSync<Subgroup>();
    for(int col=tid/64;col<member_count;col+=2){
      int outputLane=tid%64,source_job=a.members[job*8+col];
      int i=tile*64+outputLane,g=(outputLane/8)*16+outputLane%8;
      int sourceCol=source_job/8;
      float gv=__bfloat162float(__float2bfloat16_rn(scratch[sourceCol*128+g]));
      float uv=__bfloat162float(__float2bfloat16_rn(scratch[sourceCol*128+g+8]));
      float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[source_job]),mx=fabsf(v);
      for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
      unsigned e=exponent(mx);a.mid[source_job*I+i]=__nv_fp8_e4m3(v*sf(254-e)).__x;
      if(tid%32==0)a.ms[source_job*(I/32)+i/32]=e;
    }
    MatmulSync<Subgroup>();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,64);}
}

__global__ __launch_bounds__(128,2)
void grouped_tma_gate(Rank a,const int *ids,const float *prob){
  // Preserve the existing split reductions for low expert counts and small
  // row counts. The grouped full-K path has the same UMMA accumulation order
  // as the original 7/8-row plain path, but shares expert weights across rows.
  if(a.rows>=7 && a.xmap && *a.nuniq>a.rows*4)GroupedTmaGate<>(a,prob,a.xmap);
  else if(*a.nuniq<=a.rows*4){
    if(*a.nuniq>=20 && *a.nuniq*16<=gridDim.x)batched_matmul<true,false,2>(a,ids,prob);
    else batched_matmul<true>(a,ids,prob);
  }else if(a.rows>=7)plain_matmul<true,1>(a,ids,prob);
  else if(a.rows>=3)plain_matmul<true,2>(a,ids,prob);
  else plain_matmul<true,4>(a,ids,prob);
}

} // namespace fastllm_shared_flex_tp
