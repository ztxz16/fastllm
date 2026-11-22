#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/body/dspark_head_markov/markov_body.cuh"
namespace fastllm_draft_ops {
template<bool FromSlot>
__device__ __forceinline__ int markov_pick(const MarkovArgs& a,
 const naivert::TileMailbox<MarkovLayout>& mailbox,const unsigned long long* own,
 unsigned gen,float cv,int ci,int step) {
 const int lane=threadIdx.x%32;float v=-INFINITY;int i=INT_MAX;
 const naivert::MailboxWatchdog watchdog{a.watchdog_limit_ns,a.watchdog_flag};
 if(lane<8) {
  if(lane==a.rank) {
   if constexpr(FromSlot){auto p=naivert::wait_tagged_pair_bounded<true>(own,gen,watchdog,kSaltD6Cand+unsigned(step));v=__uint_as_float(naivert::tagged_payload(p.lo));i=int(naivert::tagged_payload(p.hi));}
   else {v=cv;i=ci;}
  } else {auto p=mailbox.wait_pair_bounded<true>(naivert::mailbox_source_slot(lane,a.rank),0,0,watchdog,kSaltD6Cand+unsigned(step));v=__uint_as_float(p.lo);i=int(p.hi);}
  if(kExportDiagnostics && blockIdx.x==0){a.cand_val[lane]=v;a.cand_idx[lane]=i;}
 }
#pragma unroll
 for(int o=4;o>0;o>>=1){float ov=__shfl_xor_sync(0xffffffffu,v,o);int oi=__shfl_xor_sync(0xffffffffu,i,o);markov_take(v,i,ov,oi);}
 return v > -INFINITY ? min(max(i,0),8*kMkVocab-1):8*kMkVocab-1;
}
template<int Threads, int Groups=1> __device__ __forceinline__ void markov_sync() {
  if constexpr(Groups==1) asm volatile("bar.sync 7, %0;" :: "n"(Threads) : "memory");
  else {
    const unsigned barrier=4+threadIdx.x/Threads;
    asm volatile("bar.sync %0, %1;" :: "r"(barrier), "n"(Threads) : "memory");
  }
}
template<int Threads, bool Distributed=false, int Stride=146, int Groups=1>
__device__ __forceinline__ void markov_cooperative_steps(const MarkovArgs& a,
    const float* logits,unsigned long long* partials,unsigned long long* winner,int steps) {
  if(threadIdx.x>=Threads*Groups)return;
  static_assert(Groups>=1 && Groups<=4 && Threads%32==0);
  const int group=threadIdx.x/Threads;
  constexpr int Warps=Threads/32,Ctas=148,Columns=144;
  static_assert(Warps<=8);
  __shared__ float all_values[Groups][2][Warps];
  __shared__ int all_indices[Groups][2][Warps], all_previous[Groups];
  auto &values=all_values[group];auto &indices=all_indices[group];auto &previous=all_previous[group];
  const int tid=threadIdx.x%Threads,warp=tid/32,lane=tid%32,cta=blockIdx.x;
  const unsigned epoch=unsigned(*a.replay_generation);
  const naivert::MailboxWatchdog watchdog{a.watchdog_limit_ns,a.watchdog_flag};
  int prev=min(max(int(a.block_ids[0]),0),8*kMkVocab-1);
  for(int j=0;j<steps;++j) {
    const unsigned gen=naivert::mailbox_generation(epoch,kSaltD6Cand+unsigned(j));
    if constexpr(kExportDiagnostics) {
      if(cta==0 && a.apply_markov)for(int k=tid;k<256;k+=Threads) {
        float v=__bfloat162float(a.markov_w1[std::size_t(prev)*256+k]);
        auto lat=__float2bfloat16_rn(__fadd_rn(0.f,v));
        a.lat_r[k]=__float2bfloat16_rn(prev/kMkVocab==a.rank?v:0.f);
        a.latent[k]=lat;a.lat_trace[j*256+k]=lat;
      }
    }
    float best=-INFINITY;int index=INT_MAX;
#pragma unroll
    for(int col=tid;col<Columns;col+=Threads) {
      int n=(cta+(col/16)*Ctas)*16+col%16;
      if(n<kMkVocab) {
        float value=logits[j*Stride+col];
        if(a.apply_markov) {
          auto bias=a.bias_table[std::size_t(prev)*kMkVocab+n];value=float(__float2bfloat16_rn(value+__bfloat162float(bias)));
          if constexpr(kExportDiagnostics){a.step_bias[n]=bias;a.bias_trace[j*kMkVocab+n]=bias;}
        }
        if constexpr(kExportDiagnostics){a.step_logits[n]=value;}
        // Verification needs the distribution used to draw this proposal,
        // before adding Gumbel noise, also in the production D56 build.
        if(kExportDiagnostics || a.sampling_state) a.logit_trace[j*kMkVocab+n]=value;
        const float score = a.sampling_state
            ? naivert_dspark::tempered(value, a.sampling_params ? a.sampling_params[4] : naivert_dspark::kInvTemperature)
                + sampling_gumbel(a.sampling_state, j, a.rank * kMkVocab + n)
            : value;
        markov_take(best,index,score,n);
      }
    }
#pragma unroll
    for(int o=16;o>0;o>>=1){float v=__shfl_xor_sync(0xffffffffu,best,o);int i=__shfl_xor_sync(0xffffffffu,index,o);markov_take(best,index,v,i);}
    if(lane==0){values[0][warp]=best;indices[0][warp]=index;}
    markov_sync<Threads,Groups>();
    if(warp==0) {
      float v=lane<Warps?values[0][lane]:-INFINITY;int i=lane<Warps?indices[0][lane]:INT_MAX;
#pragma unroll
      for(int o=4;o>0;o>>=1){float ov=__shfl_xor_sync(0xffffffffu,v,o);int oi=__shfl_xor_sync(0xffffffffu,i,o);markov_take(v,i,ov,oi);}
      if(lane==0)naivert::store_tagged_pair(partials+2*cta,naivert::tagged_word(gen,__float_as_uint(v)),naivert::tagged_word(gen,unsigned(i)));
    }
    if(cta==0) {
      float v=-INFINITY;int i=INT_MAX;
#pragma unroll
      for(int c=tid;c<Ctas;c+=Threads){auto p=naivert::wait_tagged_pair_bounded<true>(partials+2*c,gen,watchdog,kSaltD6Cand+unsigned(j));markov_take(v,i,__uint_as_float(naivert::tagged_payload(p.lo)),int(naivert::tagged_payload(p.hi)));}
#pragma unroll
      for(int o=16;o>0;o>>=1){float ov=__shfl_xor_sync(0xffffffffu,v,o);int oi=__shfl_xor_sync(0xffffffffu,i,o);markov_take(v,i,ov,oi);}
      // A separate pane prevents early global folds from overwriting the local
      // partials still being read by warp 0 before its publication above.
      if(lane==0){values[1][warp]=v;indices[1][warp]=i;}
      markov_sync<Threads,Groups>();
      if(warp==0) {
        float cv=lane<Warps?values[1][lane]:-INFINITY;int ci=lane<Warps?indices[1][lane]:INT_MAX;
#pragma unroll
        for(int o=4;o>0;o>>=1){float ov=__shfl_xor_sync(0xffffffffu,cv,o);int oi=__shfl_xor_sync(0xffffffffu,ci,o);markov_take(cv,ci,ov,oi);}
        const int gi=ci+a.rank*kMkVocab;
        const auto mailbox=naivert::TileMailbox<MarkovLayout>::make(a.mailbox_peers,a.rank,gen,kMkSlabCand*naivert::kMailboxSlabStride);
        if(lane==0) {
          if constexpr(Distributed)naivert::store_tagged_pair(winner,naivert::tagged_word(gen,__float_as_uint(cv)),naivert::tagged_word(gen,unsigned(gi)));
          mailbox.publish_pair(0,0,__float_as_uint(cv),unsigned(gi));
        }
        if constexpr(!Distributed) {
          const int pi=markov_pick<false>(a,mailbox,winner,gen,cv,gi,j);
          if(lane==0){a.draft_tokens[j]=pi;a.block_ids[j+1]=pi;a.prev_token[0]=pi;naivert::store_tagged_pair(winner,naivert::tagged_word(gen,unsigned(pi)),naivert::tagged_word(gen,unsigned(pi)));}
        }
      }
    }
    if constexpr(Distributed) {
      if(warp==0) {
        const auto mailbox=naivert::TileMailbox<MarkovLayout>::make(a.mailbox_peers,a.rank,gen,kMkSlabCand*naivert::kMailboxSlabStride);
        const int pi=markov_pick<true>(a,mailbox,winner,gen,0.f,0,j);
        if(lane==0){previous=pi;if(cta==0){a.draft_tokens[j]=pi;a.block_ids[j+1]=pi;a.prev_token[0]=pi;}}
      }
    } else if(tid==0){auto p=naivert::wait_tagged_pair_bounded<true>(winner,gen,watchdog,kSaltD6Cand+unsigned(j));previous=min(max(int(naivert::tagged_payload(p.lo)),0),8*kMkVocab-1);}
    markov_sync<Threads,Groups>();
    prev=previous;
  }
}
}
