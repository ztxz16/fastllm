// Split-K, 128x8 block-scaled UMMA specialization, consuming MegaMoE layout.
// Four K partitions are workspace only, not duplicate weights.
namespace fastllm_shared_flex_tp {
namespace dg=fastllm_blackwell_mega;
__device__ __forceinline__ bool parity_wait(uint64_t*p,unsigned phase){
  unsigned ok;unsigned sm=(unsigned)__cvta_generic_to_shared(p);
  asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.b32 %0,1,0,p;}"
    :"=r"(ok):"r"(sm),"r"(phase):"memory");return ok;
}
__device__ __forceinline__ void tma3(void*dst,const CUtensorMap*map,int k,int n,int e,uint64_t*bar){
  unsigned d=(unsigned)__cvta_generic_to_shared(dst),b=(unsigned)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
    "[%0],[%1,{%2,%3,%4}],[%5];"::"r"(d),"l"(map),"r"(k),"r"(n),"r"(e),"r"(b):"memory");
}
__device__ __forceinline__ void bulk(void*dst,const void*src,int bytes,uint64_t*bar){
  unsigned d=(unsigned)__cvta_generic_to_shared(dst),b=(unsigned)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
    ::"r"(d),"l"(src),"r"(bytes),"r"(b):"memory");
}
template<bool G,bool PDL=false> __global__ __launch_bounds__(128,2)
void single_matmul(Rank a,const int*ids,const float*prob){
  if constexpr(PDL) cudaTriggerProgrammaticLaunchCompletion();
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?8:1,PAGES=NK/128/SPLIT;
  __shared__ __align__(128) uint8_t wa[2][16384],xb[2][1024],sfa[2][512],sfb[2][512];
  __shared__ __align__(8) uint64_t loaded[2],done;
  __shared__ unsigned tm; __shared__ int last;
  int tid=threadIdx.x;
  if((a.rows*K)==0)return;
  if(G&&tid==0&&blockIdx.x==0&&a.rows==1)++*a.epoch;
  if(tid==0){
    for(int b=0;b<2;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(l));}
    unsigned d=(unsigned)__cvta_generic_to_shared(&done);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(d):"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(32,&tm);al.release_allocation_lock();}
  __syncthreads();
  auto idesc=cute::UMMA::make_instr_desc_block_scaled<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  unsigned phase=0,load_phase[2]={0,0};
  for(int task=blockIdx.x;task<(a.rows*K)*(NN/128)*SPLIT;task+=gridDim.x){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=task/((NN/128)*SPLIT);
    int expert=ids[job],row=job/K;
    const uint8_t*x=G?a.xq+row*H:a.mid+job*I;
    const uint8_t*xs=G?a.xs+row*(H/32):a.ms+job*(I/32);
    const uint8_t*scales=G?a.gs:a.ds;
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(16384+512):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
      }
      if(G&&a.rows==1){
        float v=__bfloat162float(a.raw_x[kg*128+tid]),mx=fabsf(v);
        #pragma unroll
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
        unsigned e=exponent(mx),q=__nv_fp8_e4m3(v*sf(254-e)).__x;
        unsigned pack=q|(__shfl_down_sync(0xffffffff,q,1)<<8)|(__shfl_down_sync(0xffffffff,q,2)<<16)|(__shfl_down_sync(0xffffffff,q,3)<<24);
        if(tid%4==0){
          #pragma unroll
          for(int br=0;br<8;++br)*reinterpret_cast<unsigned*>(xb[b]+br*128+(tid^((br&7)*16)))=pack;
        }
        #pragma unroll
        for(int br=tid%32;br<128;br+=32)sfb[b][((br%32)*4+br/32)*4+tid/32]=e;
      }else{
      #pragma unroll
      for(int j=tid;j<256;j+=128){int br=j/32,k=j%32*4;
        *reinterpret_cast<unsigned*>(xb[b]+br*128+(k^((br&7)*16)))=ld4(x+kg*128+k);}
      reinterpret_cast<unsigned*>(sfb[b])[tid]=ld4(xs+kg*4);
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    };
    load_tile(split*PAGES,0);__syncthreads();
    for(int page=0;page<PAGES;++page){
      int b=page&1;
      if(tid<32){
        while(!parity_wait(&loaded[b],load_phase[b])){}
        asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
        auto ad=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4*)wa[b],0,0);
        auto bd=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,8,128,128>((E4*)xb[b],0,0);
        auto sd1=dg::mma::sm100::make_sf_desc(sfa[b]),sd2=dg::mma::sm100::make_sf_desc(sfb[b]);
        if(tid==0){cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd1,tm+8);cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd2,tm+12);}
        #pragma unroll
        for(int kk=0;kk<4;++kk){
          auto aa=ad,bb=bd;aa.start_address_+=kk*2;bb.start_address_+=kk*2;
          auto desc=dg::mma::sm100::make_runtime_instr_desc_with_sf_id(idesc,kk,kk);
          MMA::fma(aa,bb,tm,page!=0||kk!=0,desc,tm+8,tm+12);
        }
        cutlass::arch::umma_arrive(&done);
      }
      // Load the next K page while the current UMMA reads the other buffer.
      if(page+1<PAGES)load_tile(split*PAGES+page+1,b^1);
      while(!parity_wait(&done,phase)){}
      asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
      __syncthreads();phase^=1;load_phase[b]^=1;
    }
    unsigned value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(tm+(unsigned(tid/32*32)<<16),value);
    cutlass::arch::fence_view_async_tmem_load();
    float*out=G?a.gpartial:a.dpartial;
    __stcg(out+((size_t)job*SPLIT+split)*NN+tile*128+tid,__uint_as_float(value));
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    if constexpr(G){
      // Publish every lane's partial before the last-partition election.
      __threadfence();__syncthreads();
      if(tid==0)last=atomicAdd(a.gcount+job*(2*I/128)+tile,1)==SPLIT-1;
      __syncthreads();
      if(last && tid<64){
        int i=tile*64+tid,g=tile*128+(tid/8)*16+tid%8;
        float gv=0,uv=0;
        #pragma unroll
        for(int sp=0;sp<SPLIT;++sp){const float*p=out+((size_t)job*SPLIT+sp)*NN;
          gv=__fadd_rn(gv,__ldcg(p+g));uv=__fadd_rn(uv,__ldcg(p+g+8));}
        gv=__bfloat162float(__float2bfloat16_rn(gv));uv=__bfloat162float(__float2bfloat16_rn(uv));
        float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[job]),mx=fabsf(v);
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
        unsigned e=exponent(mx);a.mid[job*I+i]=__nv_fp8_e4m3(v*sf(254-e)).__x;
        if(tid%32==0)a.ms[job*(I/32)+i/32]=e;
      }
    }
    __syncthreads();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,32);}
}
// The routed entry uses eight warps for selection and four for the existing
// GEMM. Its other four warps have exited; the named barrier has 128 members.
// Ordinary entries retain their original whole-block synchronization.
template<bool Subgroup> __device__ __forceinline__ void MatmulSync() {
  if constexpr(Subgroup) asm volatile("bar.sync 8, 128;":::"memory");
  else __syncthreads();
}
template<bool G,int GS,bool PDL=false,bool Subgroup=false> __device__ __forceinline__
void plain_matmul(Rank a,const int*ids,const float*prob){
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?GS:1,PAGES=NK/128/SPLIT,BUFFERS=PAGES<4?PAGES:4;
  extern __shared__ __align__(1024) uint8_t arena[];
  auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
  auto xb=reinterpret_cast<uint8_t(*)[1024]>(arena+BUFFERS*16384);
  auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024));
  auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024+512));
  __shared__ __align__(8) uint64_t loaded[BUFFERS],done;
  __shared__ unsigned tm; __shared__ int last;
  int tid=threadIdx.x;
  if(blockIdx.x>=*a.count*(NN/128)*SPLIT)return;
  if(tid==0){
    for(int b=0;b<BUFFERS;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(l));}
    unsigned d=(unsigned)__cvta_generic_to_shared(&done);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(d):"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(BUFFERS<=3?32:BUFFERS<=7?64:128,&tm);al.release_allocation_lock();}
  MatmulSync<Subgroup>();
  auto idesc=cute::UMMA::make_instr_desc_block_scaled<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  unsigned phase=0,load_phase[BUFFERS]={};
  for(int task=blockIdx.x;task<*a.count*(NN/128)*SPLIT;task+=gridDim.x){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=a.jobs[task/((NN/128)*SPLIT)];
    int expert=ids[job],row=job/K;
    const uint8_t*x=G?a.xq+row*H:a.mid+job*I;
    const uint8_t*xs=G?a.xs+row*(H/32):a.ms+job*(I/32);
    const uint8_t*scales=G?a.gs:a.ds;
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(16384+512):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
      }
      // Membership and weights were ready before gate/up. Only its activation
      // bytes/scales depend on the producer; do not touch them before this wait.
      if constexpr(PDL) if(task==blockIdx.x && kg==0) cudaGridDependencySynchronize();
      #pragma unroll
      for(int j=tid;j<256;j+=128){int br=j/32,k=j%32*4;
        *reinterpret_cast<unsigned*>(xb[b]+br*128+(k^((br&7)*16)))=ld4(x+kg*128+k);}
      reinterpret_cast<unsigned*>(sfb[b])[tid]=ld4(xs+kg*4);
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    };
    for(int group=0;group<PAGES;group+=BUFFERS){
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_tile(split*PAGES+group+b,b);
      MatmulSync<Subgroup>();
      if(tid<32){
        for(int b=0;b<BUFFERS&&group+b<PAGES;++b){
          while(!parity_wait(&loaded[b],load_phase[b])){}
          asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
          auto ad=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4*)wa[b],0,0);
          auto bd=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,8,128,128>((E4*)xb[b],0,0);
          auto sd1=dg::mma::sm100::make_sf_desc(sfa[b]),sd2=dg::mma::sm100::make_sf_desc(sfb[b]);
          unsigned sa=tm+8+b*8,sb=tm+12+b*8;
          if(tid==0){cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd1,sa);cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd2,sb);}
          #pragma unroll
          for(int kk=0;kk<4;++kk){
            auto aa=ad,bb=bd;aa.start_address_+=kk*2;bb.start_address_+=kk*2;
            auto desc=dg::mma::sm100::make_runtime_instr_desc_with_sf_id(idesc,kk,kk);
            MMA::fma(aa,bb,tm,group+b!=0||kk!=0,desc,sa,sb);
          }
        }
        cutlass::arch::umma_arrive(&done);
      }
      while(!parity_wait(&done,phase)){}
      asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
      MatmulSync<Subgroup>();phase^=1;
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_phase[b]^=1;
    }
    unsigned value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(tm+(unsigned(tid/32*32)<<16),value);
    cutlass::arch::fence_view_async_tmem_load();
    if constexpr(G&&SPLIT==1){
      float*scratch=reinterpret_cast<float*>(wa[0]);scratch[tid]=__uint_as_float(value);
      asm volatile("tcgen05.fence::before_thread_sync;":::"memory");MatmulSync<Subgroup>();
      if(tid<64){int i=tile*64+tid,g=(tid/8)*16+tid%8;
        float gv=__bfloat162float(__float2bfloat16_rn(scratch[g]));
        float uv=__bfloat162float(__float2bfloat16_rn(scratch[g+8]));
        float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[job]),mx=fabsf(v);
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
        unsigned e=exponent(mx);a.mid[job*I+i]=__nv_fp8_e4m3(v*sf(254-e)).__x;
        if(tid%32==0)a.ms[job*(I/32)+i/32]=e;
      }
    }else{
    float*out=G?a.gpartial:a.dpartial;
    __stcg(out+((size_t)job*SPLIT+split)*NN+tile*128+tid,__uint_as_float(value));
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    if constexpr(G){
      // Publish every lane's partial before the last-partition election.
      __threadfence();MatmulSync<Subgroup>();
      if(tid==0)last=atomicAdd(a.gcount+job*(2*I/128)+tile,1)==SPLIT-1;
      MatmulSync<Subgroup>();
      if(last && tid<64){
        int i=tile*64+tid,g=tile*128+(tid/8)*16+tid%8;
        float gv=0,uv=0;
        #pragma unroll
        for(int sp=0;sp<SPLIT;++sp){const float*p=out+((size_t)job*SPLIT+sp)*NN;
          gv=__fadd_rn(gv,__ldcg(p+g));uv=__fadd_rn(uv,__ldcg(p+g+8));}
        gv=__bfloat162float(__float2bfloat16_rn(gv));uv=__bfloat162float(__float2bfloat16_rn(uv));
        float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[job]),mx=fabsf(v);
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
        unsigned e=exponent(mx);a.mid[job*I+i]=__nv_fp8_e4m3(v*sf(254-e)).__x;
        if(tid%32==0)a.ms[job*(I/32)+i/32]=e;
      }
    }
    }
    MatmulSync<Subgroup>();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,BUFFERS<=3?32:BUFFERS<=7?64:128);}
}
template<bool G,bool PDL=false,int TaskGroup=1,bool Subgroup=false> __device__ __forceinline__
void batched_matmul(Rank a,const int*ids,const float*prob){
  static_assert(TaskGroup==1 || (G && TaskGroup==2));
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?8:1,PAGES=NK/128/SPLIT,BUFFERS=PAGES<4?PAGES:4;
  extern __shared__ __align__(1024) uint8_t arena[];
  auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
  auto xb=reinterpret_cast<uint8_t(*)[1024]>(arena+BUFFERS*16384);
  auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024));
  auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024+512));
  __shared__ __align__(8) uint64_t loaded[BUFFERS],done;
  __shared__ unsigned tm; __shared__ int last;
  int tid=threadIdx.x;
  if(blockIdx.x*TaskGroup>=*a.nuniq*(NN/128)*SPLIT)return;
  if(tid==0){
    for(int b=0;b<BUFFERS;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(l));}
    unsigned d=(unsigned)__cvta_generic_to_shared(&done);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(d):"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(BUFFERS<=3?32:BUFFERS<=7?64:128,&tm);al.release_allocation_lock();}
  MatmulSync<Subgroup>();
  auto idesc=cute::UMMA::make_instr_desc_block_scaled<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  unsigned phase=0,load_phase[BUFFERS]={};
  for(int task=blockIdx.x*TaskGroup;task<*a.nuniq*(NN/128)*SPLIT;
      task+=task%TaskGroup==TaskGroup-1?(gridDim.x-1)*TaskGroup+1:1){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=task/((NN/128)*SPLIT);
    int expert=a.uniq[job],member_count=a.rowcounts[job];
    const uint8_t*x=G?a.xq:a.mid;
    const uint8_t*xs=G?a.xs:a.ms;
    const uint8_t*scales=G?a.gs:a.ds;
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(16384+512):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
      }
      // Membership and weights were ready before gate/up. Only its activation
      // bytes/scales depend on the producer; do not touch them before this wait.
      if constexpr(PDL) if(task==blockIdx.x && kg==0) cudaGridDependencySynchronize();
      #pragma unroll
      for(int j=tid;j<256;j+=128){int br=j/32,k=j%32*4;
        int src=br<member_count?a.members[job*8+br]:0;
        unsigned v=br<member_count?ld4(x+(G?(src/K)*H:src*I)+kg*128+k):0;
        *reinterpret_cast<unsigned*>(xb[b]+br*128+(k^((br&7)*16)))=v;}
      int br=tid%8,src=br<member_count?a.members[job*8+br]:0;
      reinterpret_cast<unsigned*>(sfb[b])[(tid%32)*4+tid/32]=br<member_count?ld4(xs+(G?(src/K)*(H/32):src*(I/32))+kg*4):0x7f7f7f7fu;
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    };
    for(int group=0;group<PAGES;group+=BUFFERS){
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_tile(split*PAGES+group+b,b);
      MatmulSync<Subgroup>();
      if(tid<32){
        for(int b=0;b<BUFFERS&&group+b<PAGES;++b){
          while(!parity_wait(&loaded[b],load_phase[b])){}
          asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
          auto ad=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4*)wa[b],0,0);
          auto bd=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,8,128,128>((E4*)xb[b],0,0);
          auto sd1=dg::mma::sm100::make_sf_desc(sfa[b]),sd2=dg::mma::sm100::make_sf_desc(sfb[b]);
          unsigned sa=tm+8+b*8,sb=tm+12+b*8;
          if(tid==0){cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd1,sa);cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sd2,sb);}
          #pragma unroll
          for(int kk=0;kk<4;++kk){
            auto aa=ad,bb=bd;aa.start_address_+=kk*2;bb.start_address_+=kk*2;
            auto desc=dg::mma::sm100::make_runtime_instr_desc_with_sf_id(idesc,kk,kk);
            MMA::fma(aa,bb,tm,group+b!=0||kk!=0,desc,sa,sb);
          }
        }
        cutlass::arch::umma_arrive(&done);
      }
      while(!parity_wait(&done,phase)){}
      asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
      MatmulSync<Subgroup>();phase^=1;
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_phase[b]^=1;
    }
    float*out=G?a.gpartial:a.dpartial;
    unsigned values[8];cute::SM100_TMEM_LOAD_32dp32b8x::copy(tm+(unsigned(tid/32*32)<<16),values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]);
    cutlass::arch::fence_view_async_tmem_load();
    #pragma unroll
    for(int col=0;col<8;++col)if(col<member_count){unsigned value=values[col];
      size_t offset=G?(((size_t)job*SPLIT+split)*8+col)*NN:(size_t)a.members[job*8+col]*NN;
      __stcg(out+offset+tile*128+tid,__uint_as_float(value));
    }
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    if constexpr(G)if(task%TaskGroup==TaskGroup-1){
      // Store each original split separately and notify once per task group.
      // The final reducer preserves the eight-partition accumulation order.
      // Publish every lane's partial before the last-partition election.
      __threadfence();MatmulSync<Subgroup>();
      if(tid==0)last=atomicAdd(a.gcount+job*(2*I/128)+tile,TaskGroup)==SPLIT-TaskGroup;
      MatmulSync<Subgroup>();
      // Two independent 64-thread groups reduce different member rows.
      // Each row keeps the original split accumulation and quantization order.
      if(last)for(int col=tid/64;col<member_count;col+=2){
        int outputLane=tid%64;
        int source_job=a.members[job*8+col];
        int i=tile*64+outputLane,g=tile*128+(outputLane/8)*16+outputLane%8;
        float gv=0,uv=0;
        #pragma unroll
        for(int sp=0;sp<SPLIT;++sp){const float*p=out+(((size_t)job*SPLIT+sp)*8+col)*NN;
          gv=__fadd_rn(gv,__ldcg(p+g));uv=__fadd_rn(uv,__ldcg(p+g+8));}
        gv=__bfloat162float(__float2bfloat16_rn(gv));uv=__bfloat162float(__float2bfloat16_rn(uv));
        float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[source_job]),mx=fabsf(v);
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
        unsigned e=exponent(mx);a.mid[source_job*I+i]=__nv_fp8_e4m3(v*sf(254-e)).__x;
        if(tid%32==0)a.ms[source_job*(I/32)+i/32]=e;
      }
    }
    MatmulSync<Subgroup>();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,BUFFERS<=3?32:BUFFERS<=7?64:128);}
}

template<bool PDL=false> __global__ __launch_bounds__(128,2)
void multi_gate(Rank a,const int*ids,const float*prob){
  if constexpr(PDL) cudaTriggerProgrammaticLaunchCompletion();
  if(*a.nuniq<=a.rows*4){
    // Keep enough independent work, and avoid a second task group in a CTA.
    // Four output tiles x four split pairs give 16 grouped tasks per expert.
    if(*a.nuniq>=20 && *a.nuniq*16<=gridDim.x)batched_matmul<true,false,2>(a,ids,prob);
    else batched_matmul<true>(a,ids,prob);
  }
  else if(a.rows>=7)plain_matmul<true,1>(a,ids,prob);
  else if(a.rows>=3)plain_matmul<true,2>(a,ids,prob);
  else plain_matmul<true,4>(a,ids,prob);
}
template<bool PDL=false> __global__ __launch_bounds__(128,2)
void multi_down(Rank a,const int*ids,const float*prob){
  if(*a.nuniq<=a.rows*4)batched_matmul<false,PDL>(a,ids,prob);
  else plain_matmul<false,4,PDL>(a,ids,prob);
}
// Two BF16 values plus one generation tag in each aligned 64-bit mailbox word.
__device__ __forceinline__ unsigned pack_bf16(float x,float y){
 __nv_bfloat162 value=__floats2bfloat162_rn(x,y);
 return *reinterpret_cast<unsigned*>(&value);
}
template<int MailboxRows=MAX_M> __device__ __forceinline__ float2 collect_bf16(Rank a,int row,int pair,unsigned generation,unsigned payload){
 constexpr int PH=H/2;
 size_t bank=(size_t)(generation&1)*MailboxRows*R*PH;
 uint64_t word=(uint64_t(generation)<<32)|payload,words[R];
 #pragma unroll
 for(int peer=0;peer<R;++peer)
  words[peer]=peer==a.rank?word:load_acquire(a.box+bank+(size_t)(row*R+peer)*PH+pair);
 bool pending;
 do{pending=false;
  #pragma unroll
  for(int peer=0;peer<R;++peer)if(unsigned(words[peer]>>32)!=generation){
   words[peer]=load_acquire(a.box+bank+(size_t)(row*R+peer)*PH+pair);pending=true;}
 }while(pending);
 float2 total=make_float2(0.f,0.f);
 #pragma unroll
 for(int peer=0;peer<R;++peer){unsigned p=unsigned(words[peer]);
  float2 v=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&p));
  total.x=__fadd_rn(total.x,v.x);total.y=__fadd_rn(total.y,v.y);}
 return total;
}
template<int MailboxRows=MAX_M> __device__ __forceinline__ void publish_bf16(Rank a,int row,int pair,unsigned generation,unsigned payload){
 constexpr int PH=H/2;
 size_t bank=(size_t)(generation&1)*MailboxRows*R*PH;
 uint64_t word=(uint64_t(generation)<<32)|payload;
 #pragma unroll
 for(int peer=0;peer<R;++peer)if(peer!=a.rank)
  store_release(a.peers[peer]+bank+(size_t)(row*R+a.rank)*PH+pair,word);
}
__global__ void down_finish(Rank a,__nv_bfloat16*y,int rows){
 if(blockIdx.y==0&&blockIdx.x<4)a.gcount[blockIdx.x*64+threadIdx.x]=0;
 int pair=blockIdx.x*64+threadIdx.x,row=blockIdx.y;unsigned generation=(unsigned)*a.epoch;
 float2 sum=make_float2(0.f,0.f);
 #pragma unroll
 for(int slot=0;slot<K;++slot){int job=row*K+slot;
  float2 v=reinterpret_cast<const float2*>(a.dpartial+(size_t)job*H)[pair];
  sum.x=__fadd_rn(sum.x,v.x);sum.y=__fadd_rn(sum.y,v.y);}
 unsigned payload=pack_bf16(sum.x,sum.y);publish_bf16(a,row,pair,generation,payload);
 float2 total=collect_bf16(a,row,pair,generation,payload);
 reinterpret_cast<__nv_bfloat162*>(y+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);
}
template<bool FUSED,bool PDL=false> __global__ __launch_bounds__(256,3)
void warp_down(Rank a,const int*ids,__nv_bfloat16*y,int rows){
  extern __shared__ __align__(1024) uint8_t arena[];
  auto w=reinterpret_cast<uint8_t(*)[2][4096]>(arena);
  auto sc=reinterpret_cast<uint8_t(*)[2][512]>(arena+65536);
  __shared__ __align__(8) uint64_t loaded;
  __shared__ float partial[128];
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,group=warp%2,part=warp/2;
  int tile=blockIdx.x,row=blockIdx.y;
  if constexpr (!PDL) {
    if (FUSED && row == 0 && tile == 0) a.gcount[tid] = 0;
  }
  if(tid==0){unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster; mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(73728):"memory");}
  __syncthreads();
  if(lane==0){int slot=warp,expert=ids[row*8+slot];
    #pragma unroll
    for(int kg=0;kg<2;++kg){
      tma3(w[slot][kg],a.warpmap,kg*128,tile*32,expert,&loaded);
      bulk(sc[slot][kg],a.ds+(size_t)expert*H*8+((size_t)kg*H+(tile*32&~127))*4,512,&loaded);
    }
  }
  while(!parity_wait(&loaded,0)){}
  __syncthreads();
  if constexpr(PDL){
    // IDs and weights predate gate/up. Only activation/scale and counters
    // depend on that kernel; wait before reading or resetting those buffers.
    cudaGridDependencySynchronize();
    if(FUSED&&row==0&&tile==0)a.gcount[tid]=0;
  }
  int lr=lane/4,q=lane%4,wr=group*16+lr;
  float d0=0,d2=0;
  #pragma unroll
  for(int slot=0;slot<8;++slot){
    #pragma unroll
    for(int kb=part;kb<8;kb+=4){
      int kg=kb/4,k=(kb%4)*32+q*4;
      const uint8_t*ww=w[slot][kg],*x=a.mid+(row*8+slot)*I+kb*32;
      float p0=0,p1=0,p2=0,p3=0;
      mma(p0,p1,p2,p3,
          ld4(ww+wr*128+(k^((wr&7)*16))),ld4(ww+(wr+8)*128+(k^(((wr+8)&7)*16))),
          ld4(ww+wr*128+((k+16)^((wr&7)*16))),ld4(ww+(wr+8)*128+((k+16)^(((wr+8)&7)*16))),ld4(x+q*4),ld4(x+q*4+16));
      float xs=sf(a.ms[(row*8+slot)*8+kb]);
      int g0=((wr%32)*4+(tile%4))*4+kb%4;
      int g2=(((wr+8)%32)*4+(tile%4))*4+kb%4;
      d0=__fadd_rn(d0,__fmul_rn(p0,__fmul_rn(xs,sf(sc[slot][kg][g0]))));
      d2=__fadd_rn(d2,__fmul_rn(p2,__fmul_rn(xs,sf(sc[slot][kg][g2]))));
    }
  }
  if(q==0){partial[part*32+group*16+lr]=d0;partial[part*32+group*16+lr+8]=d2;}
  __syncthreads();
  if(tid<16){
    float2 p0=reinterpret_cast<float2*>(partial)[tid],p1=reinterpret_cast<float2*>(partial+32)[tid],
           p2=reinterpret_cast<float2*>(partial+64)[tid],p3=reinterpret_cast<float2*>(partial+96)[tid];
    float x=__fadd_rn(__fadd_rn(p0.x,p1.x),__fadd_rn(p2.x,p3.x));
    float z=__fadd_rn(__fadd_rn(p0.y,p1.y),__fadd_rn(p2.y,p3.y));
    int pair=tile*16+tid;unsigned generation=(unsigned)*a.epoch;
    if constexpr(!FUSED)reinterpret_cast<float2*>(a.wlocal+row*H)[pair]=make_float2(x,z);
    unsigned payload=pack_bf16(x,z);publish_bf16(a,row,pair,generation,payload);
    if constexpr(FUSED){float2 total=collect_bf16(a,row,pair,generation,payload);
      reinterpret_cast<__nv_bfloat162*>(y+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);}
  }

}
__global__ void warp_receive(Rank a,__nv_bfloat16*y,int rows){
 int row=blockIdx.y,pair=blockIdx.x*64+threadIdx.x;
 if(row==0&&blockIdx.x<4)a.gcount[blockIdx.x*64+threadIdx.x]=0;
 float2 local=reinterpret_cast<const float2*>(a.wlocal+row*H)[pair];
 float2 total=collect_bf16(a,row,pair,(unsigned)*a.epoch,pack_bf16(local.x,local.y));
 reinterpret_cast<__nv_bfloat162*>(y+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);
}
}
