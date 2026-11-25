// Split-K, 128x8 block-scaled UMMA specialization, consuming MegaMoE layout.
// Four K partitions are workspace only, not duplicate weights.
namespace fastllm_shared_flex_ep {
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
template<bool G,int GS,int DS> __device__ __forceinline__
void plain_matmul(Rank a,const int*ids,const float*prob){
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?GS:DS,PAGES=NK/128/SPLIT,BUFFERS=PAGES<4?PAGES:4;
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
  __syncthreads();
  auto idesc=fastllm_shared_flex::weight_instruction(a.mxfp4);
  unsigned phase=0,load_phase[BUFFERS]={};
  for(int task=blockIdx.x;task<*a.count*(NN/128)*SPLIT;task+=gridDim.x){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=a.jobs[task/((NN/128)*SPLIT)];
    int expert=ids[job]%32,row=job/K;
    const uint8_t*x=G?a.xq+row*H:a.mid+job*I;
    const uint8_t*xs=G?a.xs+row*(H/32):a.ms+job*(I/32);
    const uint8_t*scales=G?a.gs:a.ds;
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(fastllm_shared_flex::weight_tile_bytes(a.mxfp4)+512):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
      }
      #pragma unroll
      for(int j=tid;j<256;j+=128){int br=j/32,k=j%32*4;
        *reinterpret_cast<unsigned*>(xb[b]+br*128+(k^((br&7)*16)))=ld4(x+kg*128+k);}
      reinterpret_cast<unsigned*>(sfb[b])[tid]=ld4(xs+kg*4);
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    };
    for(int group=0;group<PAGES;group+=BUFFERS){
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_tile(split*PAGES+group+b,b);
      __syncthreads();
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
      __syncthreads();phase^=1;
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_phase[b]^=1;
    }
    unsigned value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(tm+(unsigned(tid/32*32)<<16),value);
    cutlass::arch::fence_view_async_tmem_load();
    if constexpr(G&&SPLIT==1){
      float*scratch=reinterpret_cast<float*>(wa[0]);scratch[tid]=__uint_as_float(value);
      asm volatile("tcgen05.fence::before_thread_sync;":::"memory");__syncthreads();
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
    }
    __syncthreads();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,BUFFERS<=3?32:BUFFERS<=7?64:128);}
}
// Each cluster owns both K partitions of one expert/output tile. Exchange
// the FP32 partials in distributed shared memory before rounding and K32
// quantization. An even grid keeps both CTAs on the same tile on every pass.
__device__ __forceinline__ void writePairedGate(Rank a,const float *prob,int expert,int job,int tile,
                                               float *partial,const unsigned (&values)[8]) {
 auto cluster=cooperative_groups::this_cluster();
 int tid=threadIdx.x, half=cluster.block_rank();
 #pragma unroll
 for(int col=0;col<MAX_M;++col)partial[col*128+tid]=__uint_as_float(values[col]);
 cluster.sync();
 const float *peer=cluster.map_shared_rank(partial,half^1);
 const int lane=tid%64,g=(lane/8)*16+lane%8;
 #pragma unroll
 for(int offset=0;offset<4;offset+=2){
  int row=half*4+offset+tid/64,route=a.members[job*MAX_M+row];
  if(route<0)continue;
  int index=row*128+g;
  float gate=__bfloat162float(__float2bfloat16_rn(__fadd_rn(partial[index],peer[index])));
  float up=__bfloat162float(__float2bfloat16_rn(__fadd_rn(partial[index+8],peer[index+8])));
  float v=__fmul_rn(__fmul_rn(gate/(1.f+__expf(-gate)),up),prob[route]),mx=fabsf(v);
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
  unsigned e=exponent(mx);
  VerifyOperand::write<I>(VerifyOperand::down(a,expert),row,tile*64+lane,__nv_fp8_e4m3(v*sf(254-e)).__x,e);
 }
 // Both CTAs' global stores must be visible to the Down TMA reader before
 // publishing this tile. Even a CTA with no selected rows joins the barrier.
 asm volatile("fence.proxy.async.global;":::"memory");
 cluster.sync();
 if(tid==0 && half==0){
  [[maybe_unused]] unsigned old;
  asm volatile("atom.acq_rel.gpu.global.or.b32 %0,[%1],%2;":"=r"(old):
               "l"(a.gcount+VerifyOperand::ReadyOffset+job),"r"(1u<<tile):"memory");
 }
}
template<bool G,int GS,int DS,bool Packed=false,bool PairedGate=false,bool StreamDown=false> __device__ __forceinline__
void batched_matmul(Rank a,const float*prob){
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?GS:DS,PAGES=NK/128/SPLIT,
                STAGE_LIMIT=Packed?VerifyOperand::Stages:STAGES,BUFFERS=PAGES<STAGE_LIMIT?PAGES:STAGE_LIMIT;
  extern __shared__ __align__(1024) uint8_t arena[];
  auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
  auto xb=reinterpret_cast<uint8_t(*)[1024]>(arena+BUFFERS*16384);
  auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024));
  auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024+512));
  __shared__ __align__(8) uint64_t loaded[BUFFERS],done;
  __shared__ unsigned tm; __shared__ int last;
  int tid=threadIdx.x;
  if(blockIdx.x>=*a.nuniq*(NN/128)*SPLIT)return;
  if(tid==0){
    for(int b=0;b<BUFFERS;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(l));}
    unsigned d=(unsigned)__cvta_generic_to_shared(&done);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(d):"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(BUFFERS<=3?32:BUFFERS<=7?64:128,&tm);al.release_allocation_lock();}
  __syncthreads();
  auto idesc=fastllm_shared_flex::weight_instruction(Packed || a.mxfp4);
  unsigned phase=0,load_phase[BUFFERS]={};
  for(int task=blockIdx.x;task<*a.nuniq*(NN/128)*SPLIT;task+=gridDim.x){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=task/((NN/128)*SPLIT);
    int expert=a.uniq[job],member_count=Packed?MAX_M:a.rowcounts[job];
    const uint8_t*x=G?a.xq:a.mid;
    const uint8_t*xs=G?a.xs:a.ms;
    const uint8_t*scales=G?a.gs:a.ds;
    const int activationBytes = Packed ? VerifyOperand::Values + VerifyOperand::Scales : 0;
    const int tileBytes = fastllm_shared_flex::weight_tile_bytes(Packed || a.mxfp4) + 512 + activationBytes;
    const uint8_t *operand = G ? VerifyOperand::base(a) : VerifyOperand::down(a,expert);
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(tileBytes):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
        if constexpr (Packed) {
          if constexpr(StreamDown){
            static_assert(!G && Packed && DS==2 && I/64==32);
            if(kg==split*PAGES){
              // Weights are immutable and already in flight. Each Down split
              // needs 16 consecutive 64-channel Gate tiles for this expert.
              unsigned want=0xffffu<<(16*split),ready=0;
              for(int spin=0;spin<128;++spin){
                asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(ready):
                             "l"(a.gcount+VerifyOperand::ReadyOffset+job):"memory");
                if((ready&want)==want)break;
              }
              if((ready&want)!=want)cudaGridDependencySynchronize();
            }
          }
          bulk(xb[b],operand + kg*VerifyOperand::Values,VerifyOperand::Values,&loaded[b]);
          bulk(sfb[b],operand + NK/128*VerifyOperand::Values + kg*VerifyOperand::Scales,VerifyOperand::Scales,&loaded[b]);
        }
      }
      if constexpr (!Packed) {
        #pragma unroll
        for(int j=tid;j<256;j+=128){int br=j/32,k=j%32*4;
          int src=br<member_count?a.members[job*8+br]:0;
          unsigned v=br<member_count?ld4(x+(G?(src/K)*H:src*I)+kg*128+k):0;
          *reinterpret_cast<unsigned*>(xb[b]+br*128+(k^((br&7)*16)))=v;}
        int br=tid%8,src=br<member_count?a.members[job*8+br]:0;
        reinterpret_cast<unsigned*>(sfb[b])[(tid%32)*4+tid/32]=br<member_count?ld4(xs+(G?(src/K)*(H/32):src*(I/32))+kg*4):0x7f7f7f7fu;
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
      }
    };
    for(int group=0;group<PAGES;group+=BUFFERS){
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_tile(split*PAGES+group+b,b);
      __syncthreads();
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
      __syncthreads();phase^=1;
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_phase[b]^=1;
    }
    float*out=G?a.gpartial:a.dpartial;
    unsigned values[8];cute::SM100_TMEM_LOAD_32dp32b8x::copy(tm+(unsigned(tid/32*32)<<16),values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]);
    cutlass::arch::fence_view_async_tmem_load();
    if constexpr(PairedGate){
      static_assert(G && Packed && SPLIT==2);
      asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
      writePairedGate(a,prob,expert,job,tile,reinterpret_cast<float*>(wa[0]),values);
    }else{
    #pragma unroll
    for(int col=0;col<8;++col)if(Packed ? a.members[job*MAX_M+col]>=0 : col<member_count){unsigned value=values[col];
      size_t offset=((size_t)a.members[job*8+col]*SPLIT+split)*NN;
      __stcg(out+offset+tile*128+tid,__uint_as_float(value));
    }
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    if constexpr(G){
      // Publish every lane's partial before the last-partition election.
      __threadfence();__syncthreads();
      if(tid==0)last=atomicAdd(a.gcount+job*(2*I/128)+tile,1)==SPLIT-1;
      __syncthreads();
      if(last && tid<64)for(int col=0;col<(Packed?MAX_M:member_count);++col){
        int source_job=a.members[job*MAX_M+col];
        if constexpr (Packed) { if (source_job < 0) continue; }
        int i=tile*64+tid,g=tile*128+(tid/8)*16+tid%8;
        float gv=0,uv=0;
        #pragma unroll
        for(int sp=0;sp<SPLIT;++sp){const float*p=out+((size_t)source_job*SPLIT+sp)*NN;
          gv=__fadd_rn(gv,__ldcg(p+g));uv=__fadd_rn(uv,__ldcg(p+g+8));}
        gv=__bfloat162float(__float2bfloat16_rn(gv));uv=__bfloat162float(__float2bfloat16_rn(uv));
        float v=__fmul_rn(__fmul_rn(gv/(1.f+__expf(-gv)),uv),prob[source_job]),mx=fabsf(v);
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
        unsigned e=exponent(mx);
        uint8_t q=__nv_fp8_e4m3(v*sf(254-e)).__x;
        if constexpr (Packed) {
          VerifyOperand::write<I>(VerifyOperand::down(a,expert),col,i,q,e);
        } else {
          a.mid[source_job*I+i]=q;
          if(tid%32==0)a.ms[source_job*(I/32)+i/32]=e;
        }
      }
    }
    }
    __syncthreads();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,BUFFERS<=3?32:BUFFERS<=7?64:128);}
}
__device__ __forceinline__ int split_for(Rank a){
 bool batch=*a.nuniq*2<=*a.count;
 int work=batch?*a.nuniq:*a.count;
 return work<=1?8:4;
}
template<bool G> __global__ __launch_bounds__(128,2)
void execute(Rank a,const int*ids,const float*prob){
 int split=split_for(a);
 if(*a.nuniq*2<=*a.count){
  if(split==8)batched_matmul<G,8,8>(a,prob);
  else batched_matmul<G,4,4>(a,prob);
 }else{
  if(split==8)plain_matmul<G,8,8>(a,ids,prob);
  else plain_matmul<G,4,4>(a,ids,prob);
 }
}
// Process every distinct local expert once per output tile, even when only
// two of the eight rows selected it.
template<bool Gate,bool StreamDown=false> __global__ __launch_bounds__(128,1)
void grouped_mxfp4_rows8(Rank a,const float *prob) {
 if constexpr (!Gate) cudaTriggerProgrammaticLaunchCompletion();
 batched_matmul<Gate,VerifyOperand::GateSplit,VerifyOperand::DownSplit,true,false,StreamDown>(a,prob);
}
__global__ __cluster_dims__(2,1,1) __launch_bounds__(128,1)
void paired_gate_rows8(Rank a,const float *prob){
 cudaTriggerProgrammaticLaunchCompletion();
 batched_matmul<true,VerifyOperand::GateSplit,VerifyOperand::DownSplit,true,true>(a,prob);
}
template<int GS> __device__ __forceinline__
void single_gate(Rank a,const int*ids,const float*prob){
  constexpr bool G=true;constexpr int DS=D_SPLIT;
  using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
  using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>;
  constexpr int NK=G?H:I,NN=G?2*I:H,SPLIT=G?GS:DS,PAGES=NK/128/SPLIT,BUFFERS=PAGES<4?PAGES:4;
  extern __shared__ __align__(1024) uint8_t arena[];
  auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
  auto xb=reinterpret_cast<uint8_t(*)[1024]>(arena+BUFFERS*16384);
  auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024));
  auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+BUFFERS*(16384+1024+512));
  __shared__ __align__(8) uint64_t loaded[BUFFERS],done;
  __shared__ unsigned tm; __shared__ int last;
  int tid=threadIdx.x;
  int localjobs[8],localn=0;
  for(int slot=0;slot<8;++slot)if(ids[slot]/32==a.rank)localjobs[localn++]=slot;
  if(blockIdx.x==0&&tid==0){++*a.epoch;*a.count=localn;*a.nuniq=localn;
    for(int j=0;j<localn;++j)a.jobs[j]=localjobs[j];}
  if(blockIdx.x>=localn*(NN/128)*SPLIT)return;
  if(tid==0){
    for(int b=0;b<BUFFERS;++b){unsigned l=(unsigned)__cvta_generic_to_shared(&loaded[b]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(l));}
    unsigned d=(unsigned)__cvta_generic_to_shared(&done);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(d):"memory");
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.allocate(BUFFERS<=3?32:BUFFERS<=7?64:128,&tm);al.release_allocation_lock();}
  __syncthreads();
  auto idesc=fastllm_shared_flex::weight_instruction(a.mxfp4);
  unsigned phase=0,load_phase[BUFFERS]={};
  for(int task=blockIdx.x;task<localn*(NN/128)*SPLIT;task+=gridDim.x){
    int split=task%SPLIT,tile=task/SPLIT%(NN/128),job=localjobs[task/((NN/128)*SPLIT)];
    int expert=ids[job]%32;
    const uint8_t*scales=G?a.gs:a.ds;
    auto load_tile=[&](int kg,int b){
      if(tid==0){
        unsigned mb=(unsigned)__cvta_generic_to_shared(&loaded[b]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(fastllm_shared_flex::weight_tile_bytes(a.mxfp4)+512):"memory");
        tma3(wa[b],G?a.gmap:a.dmap,kg*128,tile*128,expert,&loaded[b]);
        bulk(sfa[b],scales+(size_t)expert*NN*(NK/32)+((size_t)kg*NN+tile*128)*4,512,&loaded[b]);
      }
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
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    };
    for(int group=0;group<PAGES;group+=BUFFERS){
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_tile(split*PAGES+group+b,b);
      __syncthreads();
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
      __syncthreads();phase^=1;
      for(int b=0;b<BUFFERS&&group+b<PAGES;++b)load_phase[b]^=1;
    }
    unsigned value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(tm+(unsigned(tid/32*32)<<16),value);
    cutlass::arch::fence_view_async_tmem_load();
    if constexpr(G&&SPLIT==1){
      float*scratch=reinterpret_cast<float*>(wa[0]);scratch[tid]=__uint_as_float(value);
      asm volatile("tcgen05.fence::before_thread_sync;":::"memory");__syncthreads();
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
    }
    __syncthreads();
  }
  if(tid<32){cute::TMEM::Allocator1Sm al;al.free(tm,BUFFERS<=3?32:BUFFERS<=7?64:128);}
}
__global__ __launch_bounds__(128,2) void single_execute(Rank a,const int*ids,const float*prob){
 int n=0;for(int slot=0;slot<8;++slot)n+=ids[slot]/32==a.rank;
 if(n<=1)single_gate<8>(a,ids,prob);else single_gate<4>(a,ids,prob);
}
// Two BF16 values plus one generation tag in each aligned 64-bit mailbox word.
__device__ __forceinline__ unsigned pack_bf16(float x,float y){
 __nv_bfloat162 value=__floats2bfloat162_rn(x,y);
 return *reinterpret_cast<unsigned*>(&value);
}
__device__ __forceinline__ float2 collect_bf16(Rank a,int row,int pair,unsigned generation,unsigned payload){
 constexpr int PH=H/2;
 size_t bank=(size_t)(generation&1)*MAX_M*R*PH;
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
__device__ __forceinline__ void publish_bf16(Rank a,int row,int pair,unsigned generation,unsigned payload){
 constexpr int PH=H/2;
 size_t bank=(size_t)(generation&1)*MAX_M*R*PH;
 uint64_t word=(uint64_t(generation)<<32)|payload;
 #pragma unroll
 for(int peer=0;peer<R;++peer)if(peer!=a.rank)
  store_release(a.peers[peer]+bank+(size_t)(row*R+a.rank)*PH+pair,word);
}

template<int DS> __device__ __forceinline__ float2 roundedExpert(Rank a,int job,int pair) {
  float2 expert=make_float2(0.f,0.f);
  #pragma unroll
  for(int sp=0;sp<DS;++sp){float2 v=reinterpret_cast<const float2*>(a.dpartial+((size_t)job*DS+sp)*H)[pair];
   expert.x=__fadd_rn(expert.x,v.x);expert.y=__fadd_rn(expert.y,v.y);}
  // Preserve the EP/Mega contract: round EACH full expert result to BF16.
  __nv_bfloat162 ev=__floats2bfloat162_rn(expert.x,expert.y);return __bfloat1622float2(ev);
}
template<int DS> __device__ __forceinline__ void finish_impl(Rank a,__nv_bfloat16*y,const int*ids,int rows){
 if(blockIdx.y==0)a.gcount[blockIdx.x*128+threadIdx.x]=0;
 int pair=blockIdx.x*128+threadIdx.x,row=blockIdx.y;unsigned generation=(unsigned)*a.epoch;
 float2 sum=make_float2(0.f,0.f);
 #pragma unroll
 for(int slot=0;slot<K;++slot){int job=row*K+slot;if(ids[job]/32!=a.rank)continue;
  float2 expert=roundedExpert<DS>(a,job,pair);
  sum.x=__fadd_rn(sum.x,expert.x);sum.y=__fadd_rn(sum.y,expert.y);
 }
 unsigned payload=pack_bf16(sum.x,sum.y);publish_bf16(a,row,pair,generation,payload);
 // The next SWA can stage immutable scales while the peers finish publishing.
 if (rows == MAX_M) cudaTriggerProgrammaticLaunchCompletion();
 float2 total=collect_bf16(a,row,pair,generation,payload);
 reinterpret_cast<__nv_bfloat162*>(y+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);
}
// Rows with no local routes can send zero as soon as Down is launched.
// The route IDs and epoch were published before the producer (Down).
__global__ void finish_rows8(Rank a,__nv_bfloat16 *y,const int *ids) {
 int pair=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y;
 unsigned generation=unsigned(*a.epoch), mask=0;
 #pragma unroll
 for (int slot=0;slot<K;++slot)
  if (ids[row*K+slot]/(E/R)==a.rank) mask|=1u<<slot;
 bool empty=mask==0;
 if (!empty) cudaGridDependencySynchronize();
 float2 sum=make_float2(0.f,0.f);
 while (mask) {
  int slot=__ffs(mask)-1; mask&=mask-1;
  float2 expert=roundedExpert<VerifyOperand::DownSplit>(a,row*K+slot,pair);
  sum.x=__fadd_rn(sum.x,expert.x);sum.y=__fadd_rn(sum.y,expert.y);
 }
 unsigned payload=pack_bf16(sum.x,sum.y);
 publish_bf16(a,row,pair,generation,payload);
 cudaTriggerProgrammaticLaunchCompletion();
 if (empty) cudaGridDependencySynchronize();
 // Single-row Gate reuses these counters without a separate prepare.
 if (row==0 && pair<K*(2*I/128)) a.gcount[pair]=0;
 sum=collect_bf16(a,row,pair,generation,payload);
 reinterpret_cast<__nv_bfloat162*>(y+row*H)[pair]=__floats2bfloat162_rn(sum.x,sum.y);
}
__global__ void finish(Rank a,__nv_bfloat16*y,const int*ids,int rows){
 if(split_for(a)==8)finish_impl<8>(a,y,ids,rows);else finish_impl<4>(a,y,ids,rows);
}
}
