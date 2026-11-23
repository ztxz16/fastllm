#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/body/dspark_context/fc_body.cuh"
#include "naivert/body/dspark_context/kv_body.cuh"
#include "naivert/pdl/grid_dependency.cuh"

namespace fastllm_draft_ops {
constexpr int kD12Splits = 8;
constexpr int kD12CtasPerSplit = kCtas / kD12Splits;
static_assert(kCtas % kD12Splits == 0);
static_assert((kTotalCols / 16) % kD12CtasPerSplit == 0);
constexpr int kD12Stride = kHidden + 32;
constexpr int kD12Scratch = kWindow * kD12Stride * 2 +
    (2 * kWindow * kColsPerCta + kWindow + kShellWarps * 128) * sizeof(float);
constexpr int kD12WeightOffset = (kD12Scratch + 127) & ~127;
constexpr int kD12WeightStride = kHidden / kD12Splits + 32;
constexpr int kD12WeightRows = kTotalCols / kD12CtasPerSplit;
constexpr int kD12WeightBytes = kD12WeightRows * kD12WeightStride * 2;
struct D12FinishCoefficients {
  float gamma[kKvRowsR], cosine[kHeadDim], sine[kHeadDim];
  std::int64_t destination;
};
constexpr int kD12FinishOffset = kWindow * kD12WeightStride * 2 + kShellWarps * 128 * sizeof(float);
static_assert(kD12FinishOffset + sizeof(D12FinishCoefficients) <= kD12WeightOffset);
static_assert(kThreads - kWindow * 32 == kHeadDim && kHeadDim == kKvRowsR);
struct FcCtxKvBody {
  struct InputMetadata { int rows,position,keep,window; float theta; };
  template<bool PreparedInput = false, bool PrecomputedFc = false, bool Announced = false>
  __device__ static void run(unsigned char* smem_raw,const FcCtxKvArgs& args,
      InputMetadata input = {8,0,0,1024,10000.f}, float fc_value = 0.f) {
    static_assert(kCtas == kCtasD2);
    // D12 has no programmatic predecessor (the tail launches it). Trigger its dependent at once
    // so the first D3's CTAs launch into the spare SMs and run their prologue during this
    // kernel; D3 waits for D12's full completion before touching its output. No-op without
    // a programmatic edge out of this kernel.
    naivert::pdl::trigger_dependents<true>();
    // Each producer publishes BF16 reduced values plus its SSQ tile. The
    // original full-grid barrier inside D1 makes ALL 4096 columns ready.
    constexpr int split_k = kHidden / kD12Splits;
    constexpr int query_stride = kD12WeightStride;
    constexpr int tiles_per_cta = (kTotalCols / 16) / kD12CtasPerSplit;
    constexpr int partial_elements = kLayers * 2 * kWindow * kKvRowsR;
    const int split = int(blockIdx.x) / kD12CtasPerSplit;
    const int bucket = int(blockIdx.x) % kD12CtasPerSplit;
    const int k_begin = split * split_k;
    const int tid = int(threadIdx.x), row = tid / 32, lane = tid % 32;
    auto* weights_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + kD12WeightOffset);
    auto prefetch = [&] {
      // Stage KV weights after FC MMA; transport may progress during TP8/RMSNorm.
      // Each copy reads one aligned 16-byte K segment of an original weight row.
      int pending = 0;
      for (int i = tid; i < kD12WeightRows * (split_k / 8); i += kThreads) {
        const int local_row = i / (split_k / 8), k = (i % (split_k / 8)) * 8;
        const int column = bucket * kD12WeightRows + local_row;
        const int layer = column / kColsPerLayer, kv = column % kColsPerLayer;
        const auto* source = (kv >= kKvRowsR ? args.kv.v_w[layer] : args.kv.k_w[layer]) +
            std::size_t(kv % kKvRowsR) * kHidden + k_begin + k;
        auto* destination = weights_s + local_row * kD12WeightStride + k;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(destination, source);
        if (++pending == 8) { cutlass::arch::cp_async_fence(); pending = 0; }
      }
      cutlass::arch::cp_async_fence();
    };
    FcCtxBody::run<true,PreparedInput,decltype(prefetch),PrecomputedFc,Announced>(smem_raw, args.fc, prefetch,input.rows,fc_value);
    auto* ctx_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* finish = reinterpret_cast<D12FinishCoefficients*>(smem_raw + kD12FinishOffset);
    // These four warps do not normalize query rows. Immutable finish inputs
    // are staged while the other eight warps perform the complete SSQ fold.
    if (row >= kWindow && int(blockIdx.x) < kFinishCtas) {
      const int component = tid - kWindow * 32;
      const int finish_layer = int(blockIdx.x) / kWindow, finish_query = int(blockIdx.x) % kWindow;
      const std::size_t rp = std::size_t(finish_query) * kHeadDim + component;
      finish->gamma[component] = args.kv.k_norm[finish_layer][component];
      if constexpr (PreparedInput) {
        const float angle=float(input.position+finish_query)*powf(input.theta,-2.f*(component%64)/128.f);
        finish->cosine[component]=float(__float2bfloat16_rn(cosf(angle)));
        finish->sine[component]=float(__float2bfloat16_rn(sinf(angle)));
        if(!component)finish->destination=finish_query<input.rows ?
            input.keep+finish_query : input.window+finish_query-1;
      } else {
        finish->cosine[component] = args.kv.rope_cos[rp];
        finish->sine[component] = args.kv.rope_sin[rp];
        if (component == 0) finish->destination = args.kv.kv_dst[finish_query];
      }
    }
    if (row < kWindow) {
      float sum = 0.0f;
      if constexpr (PreparedInput) {
        // Each lane owns four of the 128 partial sums. A warp reduction avoids
        // duplicating the entire dependent FP32 fold in every lane.
        const float4 piece = reinterpret_cast<const float4*>(args.fc.ssq + row * kCtas)[lane];
        sum = (piece.x + piece.y) + (piece.z + piece.w);
#pragma unroll
        for (int delta=16;delta;delta/=2) sum += __shfl_xor_sync(0xffffffffu,sum,delta);
      } else {
        // Retain the eager path's original accumulation order.
        float4 pieces[32];
#pragma unroll
        for (int owner = 0; owner < 32; ++owner)
          pieces[owner] = reinterpret_cast<const float4*>(args.fc.ssq + row * kCtas)[owner];
#pragma unroll
        for (int owner = 0; owner < 32; ++owner) {
          sum = __fadd_rn(sum, pieces[owner].x); sum = __fadd_rn(sum, pieces[owner].y);
          sum = __fadd_rn(sum, pieces[owner].z); sum = __fadd_rn(sum, pieces[owner].w);
        }
      }
      const float inverse_rms = rsqrtf(sum / float(kHidden) + args.fc.rms_epsilon);
      const bool valid = PreparedInput ? row<input.rows : __bfloat162float(args.fc.ctx_valid[row]) != 0.0f;
#pragma unroll
      for (int piece = 0; piece < split_k / 256; ++piece) {
        const int n = (piece * 32 + lane) * 8;
        const int column = k_begin + n;
        const uint4 in = *reinterpret_cast<const uint4*>(args.fc.ctx_sum + row * kHidden + column);
        const float4 lo = *reinterpret_cast<const float4*>(args.fc.hidden_norm + column);
        const float4 hi = *reinterpret_cast<const float4*>(args.fc.hidden_norm + column + 4);
        const float gamma[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const unsigned words[4] = {in.x, in.y, in.z, in.w};
        uint4 out; auto* packed = reinterpret_cast<unsigned*>(&out);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
          const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + word));
          const float a = float(__float2bfloat16_rn(value.x * inverse_rms)) * gamma[2 * word];
          const float b = float(__float2bfloat16_rn(value.y * inverse_rms)) * gamma[2 * word + 1];
          const auto normalized = __floats2bfloat162_rn(valid ? a : 0.0f, valid ? b : 0.0f);
          packed[word] = *reinterpret_cast<const unsigned*>(&normalized);
        }
        *reinterpret_cast<uint4*>(ctx_s + row * query_stride + n) = out;
        if constexpr (kExportDiagnostics) {
          if (bucket == 0) *reinterpret_cast<uint4*>(args.fc.ctx + row * kHidden + column) = out;
        }
      }
    }
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    auto* scratch = reinterpret_cast<float*>(smem_raw + kWindow * query_stride * 2);
    // One warp owns one of five 16-column tiles for this 512-K partition.
    // The eight partitions are exactly the original eight MMA K splits.
    naivert::attn_rows::project_strided_rows_mma<kWindow, split_k, 1, tiles_per_cta,
        kShellWarps, query_stride, true, true>(bucket * tiles_per_cta * 16, tiles_per_cta,
        [&](int column) {
          return weights_s + (column - bucket * kD12WeightRows) * kD12WeightStride;
        }, ctx_s, scratch,
        [&](int column, int query, float value) {
          const int layer = column / kColsPerLayer, kv = column % kColsPerLayer;
          const int offset = ((layer * 2 + kv / kKvRowsR) * kWindow + query) * kKvRowsR + kv % kKvRowsR;
          args.kv_partials[split * partial_elements + offset] = value;
        }, 0);
    grid_barrier(args.fc.barrier);
    if (int(blockIdx.x) >= kFinishCtas) return;
    finish_kv(args,finish,scratch,args.kv.cache_table,args.kv.table_requests,args.kv.table_request);
  }
  __device__ static void finish_kv(const FcCtxKvArgs& args,const D12FinishCoefficients* finish,
      float* scratch,const void*const*cache_table,int table_requests,int table_request,int finish_cta=int(blockIdx.x)) {
    const int tid=threadIdx.x,lane=tid%32;
    constexpr int partial_elements=kLayers*2*kWindow*kKvRowsR;
    const int layer = finish_cta / kWindow;
    const int query = finish_cta % kWindow;
    const bool is_k = tid < kKvRowsR;
    const bool is_v = tid >= kKvRowsR && tid < 2 * kKvRowsR;
    __nv_bfloat16 raw = __float2bfloat16_rn(0.0f);
    if (is_k || is_v) {
      const int offset = ((layer * 2 + int(is_v)) * kWindow + query) * kKvRowsR + tid % kKvRowsR;
      float value = 0.0f;
#pragma unroll
      for (int s = 0; s < kD12Splits; ++s) value = __fadd_rn(value, args.kv_partials[s * partial_elements + offset]);
      raw = __float2bfloat16_rn(value);
      if constexpr (kExportDiagnostics) args.kv.kv_raw[offset] = raw;
    }
    const std::size_t out = std::size_t(finish->destination) * kKvRowsR;
    if (is_v) static_cast<__nv_bfloat16*>(const_cast<void*>(cache_table[(layer * table_requests + table_request) * 2 + 1]))[out + tid - kKvRowsR] = raw;
    const float x = is_k ? __bfloat162float(raw) : 0.0f;
    float square = x * x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) square += __shfl_xor_sync(0xffffffffu, square, off);
    if (lane == 0) scratch[tid / 32] = square;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int warp = 0; warp < kKvRowsR / 32; ++warp) total += scratch[warp];
    const float scale = rsqrtf(total / float(kHeadDim) + args.kv.rms_epsilon);
    auto* head = reinterpret_cast<__nv_bfloat16*>(scratch + kShellWarps);
    if (is_k) head[tid] = __float2bfloat16_rn(float(__float2bfloat16_rn(x * scale)) * finish->gamma[tid]);
    __syncthreads();
    if (is_k) {
      const float value = __bfloat162float(head[tid]);
      const float rotated = tid < kHeadDim / 2 ? -__bfloat162float(head[tid + kHeadDim / 2]) : __bfloat162float(head[tid - kHeadDim / 2]);
      static_cast<__nv_bfloat16*>(const_cast<void*>(cache_table[(layer * table_requests + table_request) * 2]))[out + tid] = __float2bfloat16_rn(__fadd_rn(float(__float2bfloat16_rn(__fmul_rn(value, finish->cosine[tid]))), float(__float2bfloat16_rn(__fmul_rn(rotated, finish->sine[tid])))));
    }
  }
  template<int Requests,class Descriptor>
  __device__ static void run_requests(unsigned char* memory,const FcCtxKvArgs* args,
      const Descriptor* descriptors,int window,float theta) {
    static_assert(Requests==2 || Requests==4);
    constexpr int partial_elements=kLayers*2*kWindow*kKvRowsR;
    constexpr int split_k=kHidden/kD12Splits;
    constexpr int query_stride=kD12WeightStride;
    constexpr int tiles_per_cta=(kTotalCols/16)/kD12CtasPerSplit;
    constexpr int query_bytes=Requests*8*query_stride*2;
    constexpr int partial_bytes=Requests*kShellWarps*128*sizeof(float);
    static_assert(query_bytes+partial_bytes<=kD12WeightOffset);
    const int tid=threadIdx.x,cta=blockIdx.x;
    const int split=cta/kD12CtasPerSplit,bucket=cta%kD12CtasPerSplit,k_begin=split*split_k;
    naivert::pdl::trigger_dependents<true>();
    auto *projected=reinterpret_cast<float*>(memory+partial_bytes);
    const __nv_bfloat16* feature_ptrs[Requests];int feature_rows[Requests];
#pragma unroll
    for(int request=0;request<Requests;++request){const auto &d=descriptors[request];
      feature_ptrs[request]=static_cast<const __nv_bfloat16*>(d.features);
      feature_rows[request]=d.accepted?*d.accepted+1:d.rows;
    }
    // Queries are small and immutable; the bounded N8 groups share each FC
    // weight load. No 32-row shared image or persistent query buffer is needed.
    naivert::attn_rows::project_strided_rows_mma<Requests*8,4096,4,2,kShellWarps>(0,4096/16,
      [&](int column){return args[0].fc.fc+std::size_t(column)*4096;},
      args[0].fc.target_hidden,reinterpret_cast<float*>(memory),
      [&](int column,int row,float value){projected[row*32+column-cta*32]=value;},-1,
      [&](int row,int k){return row%8<feature_rows[row/8] ? __ldg(reinterpret_cast<const uint4*>(feature_ptrs[row/8]+(row%8)*4096+k)) : make_uint4(0,0,0,0);});
    float values[Requests];
#pragma unroll
    for(int request=0;request<Requests;++request)values[request]=tid<256?projected[request*256+tid]:0.f;
    __syncthreads();
    auto* weights_s=reinterpret_cast<__nv_bfloat16*>(memory+kD12WeightOffset);
    auto prefetch=[&]{
      int pending=0;
      for(int i=tid;i<kD12WeightRows*(split_k/8);i+=kThreads){
        const int local_row=i/(split_k/8),k=i%(split_k/8)*8;
        const int column=bucket*kD12WeightRows+local_row,layer=column/kColsPerLayer,kv=column%kColsPerLayer;
        const auto*source=(kv>=kKvRowsR?args[0].kv.v_w[layer]:args[0].kv.k_w[layer])+std::size_t(kv%kKvRowsR)*4096+k_begin+k;
        cutlass::arch::cp_async<16,cutlass::arch::CacheOperation::Global>(weights_s+local_row*query_stride+k,source);
        if(++pending==8){cutlass::arch::cp_async_fence();pending=0;}
      }
      cutlass::arch::cp_async_fence();
    };
    // Publish every independent request before waiting on peers. Reuse the
    // same shared receive tile, and join the grid only after all SSQ tiles exist.
    constexpr int pairs=kColsPerCta/2,items=kWindow*pairs;
    static_assert((kRanks-1)*items*sizeof(naivert::PayloadPair)<=8192);
    static_assert(8192+Requests*kWindow*kColsPerCta*sizeof(float)<=kD12WeightOffset);
    auto* exact=reinterpret_cast<float*>(memory+8192);
    auto* gathered=reinterpret_cast<naivert::PayloadPair*>(memory);
    unsigned generations[Requests],iterations[Requests];
#pragma unroll
    for(int request=0;request<Requests;++request){
      const auto&a=args[request].fc;
      iterations[request]=unsigned(*a.replay_generation);
      generations[request]=naivert::mailbox_generation(iterations[request],a.generation_salt);
      if(tid<256)exact[request*256+tid]=values[request];
    }
    __syncthreads();prefetch();
#pragma unroll
    for(int request=0;request<Requests;++request){
      const auto&a=args[request].fc;const unsigned gen=generations[request];
      const naivert::MailboxWatchdog watchdog{a.watchdog_limit_ns,a.watchdog_flag};
      if(!tid)rendezvous_wait(a.mailbox_peers[a.rank],a.rank,iterations[request],watchdog);
      __syncthreads();
      for(int job=tid;job<(kRanks-1)*items;job+=kThreads){
        const int peer=job/items,item=job%items,row=item/pairs,pair=item%pairs;
        const uint2 payload=*reinterpret_cast<const uint2*>(exact+request*256+item*2);
        const auto mailbox=naivert::TileMailbox<FcCtxLayout>::make(a.mailbox_peers,a.rank,gen,std::size_t(row)*naivert::kMailboxSlabStride);
        auto*target=mailbox.remote_word(naivert::mailbox_peer_rank(peer,a.rank),cta,pair*2);
        naivert::store_tagged_pair(target,naivert::tagged_word(gen,payload.x),naivert::tagged_word(gen,payload.y));
      }
    }
#pragma unroll
    for(int request=0;request<Requests;++request){
      const auto&a=args[request].fc;const unsigned gen=generations[request];
      const naivert::MailboxWatchdog watchdog{a.watchdog_limit_ns,a.watchdog_flag};
      for(int job=tid;job<(kRanks-1)*items;job+=kThreads){
        const int slot=job/items,item=job%items,row=item/pairs,pair=item%pairs;
        const auto mailbox=naivert::TileMailbox<FcCtxLayout>::make(a.mailbox_peers,a.rank,gen,std::size_t(row)*naivert::kMailboxSlabStride);
        gathered[job]=mailbox.wait_pair_bounded<true>(slot,cta,pair*2,watchdog,a.generation_salt);
      }
      __syncthreads();
      if(tid<items){
        const int row=tid/pairs,pair=tid%pairs;
        const uint2 own=*reinterpret_cast<const uint2*>(exact+request*256+tid*2);
        float sum[2]={0.f,0.f};
#pragma unroll
        for(int peer=0;peer<kRanks;++peer){
          const auto value=peer==a.rank?naivert::PayloadPair{own.x,own.y}:gathered[naivert::mailbox_source_slot(peer,a.rank)*items+tid];
          sum[0]=__fadd_rn(sum[0],__uint_as_float(value.lo));sum[1]=__fadd_rn(sum[1],__uint_as_float(value.hi));
        }
        float ssq=0.f;
#pragma unroll
        for(int i=0;i<2;++i){const auto v=__float2bfloat16_rn(row<feature_rows[request]?sum[i]:0.f);
          a.ctx_sum[row*kHidden+cta*kColsPerCta+pair*2+i]=v;const float x=float(v);ssq+=x*x;}
#pragma unroll
        for(int off=8;off;off>>=1)ssq+=__shfl_xor_sync(0xffffffffu,ssq,off);
        if(!pair)a.ssq[row*kCtas+cta]=ssq;
      }
      __syncthreads();
    }
    asm volatile("fence.proxy.async;" ::: "memory");
    grid_barrier(args[0].fc.barrier);
    if(!cta&&!tid)for(int request=0;request<Requests;++request)
      atomicAdd(reinterpret_cast<unsigned*>(args[request].fc.replay_generation),1u);
    auto* queries=reinterpret_cast<__nv_bfloat16*>(memory);
    const int row=tid/32,lane=tid%32;
#pragma unroll
    for(int request=0;request<Requests;++request)if(row<8){
      const auto &a=args[request].fc;const auto &d=descriptors[request];
      const int rows=d.accepted?*d.accepted+1:d.rows;
      const float4 piece=__ldcg(reinterpret_cast<const float4*>(a.ssq+row*kCtas)+lane);
      float sum=(piece.x+piece.y)+piece.z+piece.w;
      for(int delta=16;delta;delta>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
      const float inverse=rsqrtf(sum/4096.f+a.rms_epsilon);
      for(int part=0;part<split_k/256;++part){
        const int n=(part*32+lane)*8,column=k_begin+n;
        const uint4 in=__ldcg(reinterpret_cast<const uint4*>(a.ctx_sum+row*4096+column));
        const float4 lo=*reinterpret_cast<const float4*>(a.hidden_norm+column),hi=*reinterpret_cast<const float4*>(a.hidden_norm+column+4);
        const float gamma[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        const unsigned words[4]={in.x,in.y,in.z,in.w};uint4 out;auto*packed=reinterpret_cast<unsigned*>(&out);
        for(int i=0;i<4;++i){const float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words+i));
          const float x=float(__float2bfloat16_rn(v.x*inverse))*gamma[2*i],y=float(__float2bfloat16_rn(v.y*inverse))*gamma[2*i+1];
          auto normalized=__floats2bfloat162_rn(row<rows?x:0.f,row<rows?y:0.f);packed[i]=*reinterpret_cast<unsigned*>(&normalized);}
        *reinterpret_cast<uint4*>(queries+(request*8+row)*query_stride+n)=out;
      }
    }
    cutlass::arch::cp_async_wait<0>();__syncthreads();
    naivert::attn_rows::project_strided_rows_mma<Requests*8,split_k,1,tiles_per_cta,kShellWarps,query_stride,true,true>(
      bucket*tiles_per_cta*16,tiles_per_cta,[&](int column){return weights_s+(column-bucket*kD12WeightRows)*query_stride;},
      queries,reinterpret_cast<float*>(memory+query_bytes),[&](int column,int query,float value){
        const int layer=column/kColsPerLayer,kv=column%kColsPerLayer;
        const int offset=((layer*2+kv/kKvRowsR)*8+query%8)*kKvRowsR+kv%kKvRowsR;
        args[query/8].kv_partials[split*partial_elements+offset]=value;
      },0);
    grid_barrier(args[0].fc.barrier);
    auto* finish=reinterpret_cast<D12FinishCoefficients*>(memory+4096);
    for(int task=cta;task<Requests*kFinishCtas;task+=kCtas){
      const int request=task/kFinishCtas,finish_cta=task%kFinishCtas;
      const int layer=finish_cta/8,query=finish_cta%8;
      const auto &d=descriptors[request];const int rows=d.accepted?*d.accepted+1:d.rows;
      if(tid<128){
        const float angle=float(d.position+query)*powf(theta,-2.f*(tid%64)/128.f);
        finish->gamma[tid]=args[request].kv.k_norm[layer][tid];
        finish->cosine[tid]=float(__float2bfloat16_rn(cosf(angle)));finish->sine[tid]=float(__float2bfloat16_rn(sinf(angle)));
        if(!tid)finish->destination=query<rows?min(int(d.oldLength),window-1-rows)+query:window+query-1;
      }
      __syncthreads();
      finish_kv(args[request],finish,reinterpret_cast<float*>(memory),d.cacheTable,d.tableRequests,d.tableRequest,finish_cta);
      __syncthreads();
    }
  }

};
}  // namespace fastllm_draft_ops
