#pragma once
// Compact expert membership for bounded row tiles. The original FP8 weights
// are loaded once per expert/output tile and serve all matching rows. FP32
// partials are reduced in expert-slot order at the producer/collective boundary.
// No allocation or host routing readback.
namespace fastllm_shared_flex_tp {
constexpr size_t WideMailboxOffset=4*1024*1024;
constexpr int WideRows=16,WideDownBytes=38912;
// All offsets lie in the existing 8 MiB gate arena. Its upper 4 MiB stay
// reserved for both mailbox generations even when narrow and wide tiles mix.
struct WideScratch {
    static constexpr size_t Mid=0,Scales=32768,Experts=0x9000,Counts=0x9800;
    static constexpr size_t Members=0xa000,Prepared=0x10000;
    static constexpr size_t DownPartials=0x100000;
};
static_assert(WideScratch::Mid+WideRows*K*I<=WideScratch::Scales);
static_assert(WideScratch::Scales+WideRows*K*(I/32)<=WideScratch::Experts);
static_assert(WideScratch::Experts+WideRows*K*sizeof(int)<=WideScratch::Counts);
static_assert(WideScratch::Counts+WideRows*K*sizeof(int)<=WideScratch::Members);
static_assert(WideScratch::Members+WideRows*K*WideRows*sizeof(int)<=WideScratch::Prepared);
static_assert(WideScratch::Prepared+WideRows*(4112+128)<=WideScratch::DownPartials);
static_assert(WideScratch::DownPartials+WideRows*K*H*sizeof(float)<=WideMailboxOffset);
template<int Columns,int MemberStride,int OutputWidth> __device__ __forceinline__
void compact_down_tile(Rank a,int outputOffset,int item,int tile,int expert,int count,
                      unsigned tm,uint64_t *loaded,uint64_t *done,unsigned phase) {
    using E4=cutlass::float_e4m3_t;using UE=cutlass::float_ue8m0_t;
    using MMA=cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,Columns,
        cute::UMMA::Major::K,cute::UMMA::Major::K>;
    extern __shared__ __align__(1024) uint8_t arena[];
    auto wa=reinterpret_cast<uint8_t(*)[16384]>(arena);
    auto xb=reinterpret_cast<uint8_t(*)[Columns*128]>(arena+32768);
    auto sfa=reinterpret_cast<uint8_t(*)[512]>(arena+32768+Columns*256);
    auto sfb=reinterpret_cast<uint8_t(*)[512]>(arena+33792+Columns*256);
    const int tid=threadIdx.x;
    const auto desc=cute::UMMA::make_instr_desc_block_scaled<E4,E4,float,UE,128,Columns,
        cute::UMMA::Major::K,cute::UMMA::Major::K>();
    for(int k=0;k<2;++k){
        if(tid==0){
            unsigned b=unsigned(__cvta_generic_to_shared(loaded+k));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16896;"::"r"(b):"memory");
            tma3(wa[k],a.dmap,k*128,tile*128+outputOffset,expert,loaded+k);
            bulk(sfa[k],a.ds+size_t(expert)*H*8+(size_t(k)*H+tile*128+outputOffset)*4,512,loaded+k);
        }
        for(int j=tid;j<Columns*32;j+=128){
            const int col=j/32,kk=(j%32)*4,job=col<count?a.members[item*MemberStride+col]:0;
            *reinterpret_cast<unsigned*>(xb[k]+col*128+(kk^((col%8)*16)))=
                col<count?ld4(a.mid+job*I+k*128+kk):0;
        }
        const int col=tid%Columns,job=col<count?a.members[item*MemberStride+col]:0;
        reinterpret_cast<unsigned*>(sfb[k])[(tid%32)*4+tid/32]=
            col<count?ld4(a.ms+job*8+k*4):0x7f7f7f7fu;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
    if(tid<32){
        for(int k=0;k<2;++k){
            while(!parity_wait(loaded+k,phase)){}
            asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
            auto ad=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4*)wa[k],0,0);
            auto bd=dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,Columns,128,128>((E4*)xb[k],0,0);
            auto as=dg::mma::sm100::make_sf_desc(sfa[k]),bs=dg::mma::sm100::make_sf_desc(sfb[k]);
            unsigned ta=tm+Columns+k*8,tb=tm+Columns+4+k*8;
            if(tid==0){cute::SM100_UTCCP_4x32dp128bit_1cta::copy(as,ta);cute::SM100_UTCCP_4x32dp128bit_1cta::copy(bs,tb);}
#pragma unroll
            for(int kk=0;kk<4;++kk){auto aa=ad,bb=bd;aa.start_address_+=kk*2;bb.start_address_+=kk*2;
                auto runtime=dg::mma::sm100::make_runtime_instr_desc_with_sf_id(desc,kk,kk);
                MMA::fma(aa,bb,tm,k!=0||kk!=0,runtime,ta,tb);}
        }
        cutlass::arch::umma_arrive(done);
    }
    while(!parity_wait(done,phase)){}
    asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    constexpr int ReadColumns=Columns==16?16:32;
#pragma unroll
    for(int columnBase=0;columnBase<Columns;columnBase+=ReadColumns) {
      unsigned values[ReadColumns];
      if constexpr(ReadColumns==16)
        cute::SM100_TMEM_LOAD_32dp32b16x::copy(tm+columnBase+(unsigned(tid/32*32)<<16),
            values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14],values[15]);
      else
        cute::SM100_TMEM_LOAD_32dp32b32x::copy(tm+columnBase+(unsigned(tid/32*32)<<16),
            values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14],values[15],values[16],values[17],values[18],values[19],values[20],values[21],values[22],values[23],values[24],values[25],values[26],values[27],values[28],values[29],values[30],values[31]);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for(int col=0;col<ReadColumns;++col)if(columnBase+col<count) {
        const int job=a.members[item*MemberStride+columnBase+col];
        a.dpartial[size_t(job)*OutputWidth+tile*128+tid]=__uint_as_float(values[col]);
      }
    }
}
template<int Columns,int MemberStride,int OutputWidth=H,bool Adaptive=false> __device__ __forceinline__
void compact_down_impl(Rank a,int outputOffset=0) {
    __shared__ __align__(8) uint64_t loaded[2],done;
    __shared__ unsigned tm;
    constexpr int Tiles=OutputWidth/128,TmemColumns=Columns>32?128:64;
    const int tid=threadIdx.x,tasks=*a.nuniq*Tiles;
    if(blockIdx.x>=tasks)return;
    if(tid==0){
        for(int k=0;k<2;++k){unsigned b=unsigned(__cvta_generic_to_shared(loaded+k));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(b):"memory");}
        unsigned b=unsigned(__cvta_generic_to_shared(&done));
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"::"r"(b):"memory");
    }
    if(tid<32){cute::TMEM::Allocator1Sm allocator;allocator.allocate(TmemColumns,&tm);allocator.release_allocation_lock();}
    __syncthreads();
    unsigned phase=0;
    for(int task=blockIdx.x;task<tasks;task+=gridDim.x){
        const int item=task/Tiles,tile=task%Tiles,expert=a.uniq[item],count=a.rowcounts[item];
        // The branch is uniform within each CTA. Reuse the same barriers,
        // weight panes and TMEM allocation across differently sized experts.
        if constexpr(Adaptive) {
            if(count<=16)compact_down_tile<16,MemberStride,OutputWidth>(a,outputOffset,item,tile,expert,count,tm,loaded,&done,phase);
            else compact_down_tile<Columns,MemberStride,OutputWidth>(a,outputOffset,item,tile,expert,count,tm,loaded,&done,phase);
        } else compact_down_tile<Columns,MemberStride,OutputWidth>(a,outputOffset,item,tile,expert,count,tm,loaded,&done,phase);
        asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
        __syncthreads();phase^=1;
    }
    if(tid<32){cute::TMEM::Allocator1Sm allocator;allocator.free(tm,TmemColumns);}
}
__global__ __launch_bounds__(128,2)
void compact_down(Rank a) { compact_down_impl<16,16>(a); }
__global__ void compact_finish(Rank a,__nv_bfloat16 *output) {
    const int pair=blockIdx.x*64+threadIdx.x,row=blockIdx.y;
    float2 local={0.f,0.f};
#pragma unroll
    for(int slot=0;slot<8;++slot){
        float2 v=reinterpret_cast<const float2*>(a.dpartial+(size_t(row)*8+slot)*H)[pair];
        local.x=__fadd_rn(local.x,v.x);local.y=__fadd_rn(local.y,v.y);
    }
    unsigned generation=unsigned(*a.epoch),payload=pack_bf16(local.x,local.y);
    publish_bf16<16>(a,row,pair,generation,payload);
    float2 total=collect_bf16<16>(a,row,pair,generation,payload);
    reinterpret_cast<__nv_bfloat162*>(output+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);
}
// 32-row metadata and activation inputs reuse the original 1 MiB down arena.
// 32-row FP32 down partials occupy the lower 4 MiB of the gate arena, leaving
// the existing wide mailbox in its upper half untouched.
struct BatchScratch {
    static constexpr size_t Mid=0,Scales=0x10000,Experts=0x11000,Counts=0x11400;
    static constexpr size_t Members=0x11800,Prepared=0x1a000;
};
static_assert(BatchScratch::Mid+32*K*I<=BatchScratch::Scales);
static_assert(BatchScratch::Scales+32*K*(I/32)<=BatchScratch::Experts);
static_assert(BatchScratch::Experts+32*K*sizeof(int)<=BatchScratch::Counts);
static_assert(BatchScratch::Counts+32*K*sizeof(int)<=BatchScratch::Members);
static_assert(BatchScratch::Members+32*K*32*sizeof(int)<=BatchScratch::Prepared);
static_assert(BatchScratch::Prepared+32*(4112+128)<=MAX_M*K*H*sizeof(float));
static_assert(32*K*H*sizeof(float)<=WideMailboxOffset);
static_assert(4*32*(H/2)*sizeof(uint64_t)==2*MAX_M*R*(H/2)*sizeof(uint64_t));
constexpr int Batch32DownBytes=43008;
__global__ __launch_bounds__(128,2)
void batch32_down(Rank a) {
    // Keep the uniform N16 path; mixed N16/N32 experts share one launch.
    if(*a.count<=16)compact_down_impl<16,32>(a);
    else compact_down_impl<32,32,H,true>(a);
}
// One resident kernel performs rank-ordered reduce-scatter and allgather.
// Each input partial is sent to one output owner; only owners broadcast the
// rounded final result. Generation banks match the legacy 8-row mailbox.
template<int Width=H> __device__ __forceinline__
void batch_exchange_finish(Rank a,__nv_bfloat16 *output,int outputOffset=0) {
    constexpr int Part=Width/2/8,Words=32*(H/2);
    const int pair=blockIdx.x*128+threadIdx.x,row=blockIdx.y,owner=pair/Part;
    const unsigned gen=unsigned(*a.epoch);
    // Match the original 8-row collective's 1 MiB bank stride, so alternating
    // narrow/wide invocations never overwrite the other live generation.
    const size_t bank=size_t(gen&1)*(2*Words);
    float2 local={0.f,0.f};
#pragma unroll
    for(int slot=0;slot<8;++slot) {
      float2 v=reinterpret_cast<const float2*>(a.dpartial+(size_t(row)*8+slot)*Width)[pair];
      local.x=__fadd_rn(local.x,v.x);local.y=__fadd_rn(local.y,v.y);
    }
    const unsigned payload=pack_bf16(local.x,local.y);
    const uint64_t word=(uint64_t(gen)<<32)|payload;
    const int offset=pair%Part;
    store_release(a.peers[owner]+bank+size_t(row*8+a.rank)*Part+offset,word);
    const size_t result=Words+bank+size_t(row)*(Width/2)+pair;
    unsigned reduced;
    if(owner==a.rank) {
      uint64_t words[8];
#pragma unroll
      for(int peer=0;peer<8;++peer)
        words[peer]=peer==a.rank?word:load_acquire(a.box+bank+size_t(row*8+peer)*Part+offset);
      bool pending;
      do{pending=false;
#pragma unroll
        for(int peer=0;peer<8;++peer)if(unsigned(words[peer]>>32)!=gen) {
          words[peer]=load_acquire(a.box+bank+size_t(row*8+peer)*Part+offset);pending=true;
        }
      }while(pending);
      float2 total={0.f,0.f};
#pragma unroll
      for(int peer=0;peer<8;++peer) {
        unsigned p=unsigned(words[peer]);
        float2 v=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&p));
        total.x=__fadd_rn(total.x,v.x);total.y=__fadd_rn(total.y,v.y);
      }
      reduced=pack_bf16(total.x,total.y);
      const uint64_t finalWord=(uint64_t(gen)<<32)|reduced;
#pragma unroll
      for(int peer=0;peer<8;++peer)if(peer!=a.rank)store_release(a.peers[peer]+result,finalWord);
    } else {
      uint64_t value;
      do{value=load_acquire(a.box+result);}while(unsigned(value>>32)!=gen);
      reduced=unsigned(value);
    }
    reinterpret_cast<unsigned*>(output+row*H+outputOffset)[pair]=reduced;
}
__global__ void batch32_exchange_finish(Rank a,__nv_bfloat16 *output) {
    batch_exchange_finish<>(a,output);
}
// Reuse the 1 MiB down arena for 64-row inputs and metadata. Each 2048-wide
// FP32 output slice fits the existing 4 MiB partial arena; mailbox stride is
// unchanged across 8/16/32/64-row invocations.
struct Batch64Scratch {
    static constexpr size_t Mid=0,Scales=0x20000,Experts=0x21000,Counts=0x21400;
    static constexpr size_t Members=0x21800,Prepared=0x32000;
};
static_assert(Batch64Scratch::Mid+64*K*I<=Batch64Scratch::Scales);
static_assert(Batch64Scratch::Scales+64*K*(I/32)<=Batch64Scratch::Experts);
static_assert(Batch64Scratch::Experts+256*sizeof(int)<=Batch64Scratch::Counts);
static_assert(Batch64Scratch::Counts+256*sizeof(int)<=Batch64Scratch::Members);
static_assert(Batch64Scratch::Members+256*64*sizeof(int)<=Batch64Scratch::Prepared);
static_assert(Batch64Scratch::Prepared+64*(4112+128)<=MAX_M*K*H*sizeof(float));
static_assert(64*K*(H/2)*sizeof(float)<=WideMailboxOffset);
constexpr int Batch64DownBytes=51200;
__global__ __launch_bounds__(128,2)
void batch64_down(Rank a,int outputOffset) {
    if(outputOffset && blockIdx.x==0 && threadIdx.x==0)++*a.epoch;
    if(*a.count<=16)compact_down_impl<16,64,H/2>(a,outputOffset);
    else if(*a.count<=32)compact_down_impl<32,64,H/2,true>(a,outputOffset);
    // N64 retains its uniform path: adaptive dispatch loses on wide hot experts.
    else compact_down_impl<64,64,H/2>(a,outputOffset);
}
__global__ void batch64_exchange_finish(Rank a,__nv_bfloat16 *output,int outputOffset) {
    batch_exchange_finish<H/2>(a,output,outputOffset);
}

}
