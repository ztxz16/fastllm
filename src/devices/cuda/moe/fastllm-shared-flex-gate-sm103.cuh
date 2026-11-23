#pragma once
// Fused selection, expert membership, gate/up and SiLU on the existing packed
// FP8/UE8M0 weights. The original selector and MXFP8 GEMM retain their exact
// ordering and BF16/FP8 materialization boundaries. The grouped TMA path
// shares expert weights across 7/8 rows; other routes use the original GEMM.
// Each CTA computes its own
// routes; there is no inter-CTA wait. CTA 0 publishes metadata for ordinary down.
namespace fastllm_shared_flex_tp {
struct GateRouting {
    int ids[64], uniq[64], members[512], counts[64], unique, jobs[64], count;
    float probability[64];
};
struct GateSelection {
    fastllm::cuda::RouterTopKScratch<8> top[8];
    unsigned long long masks[256];
    int sums[2];
};
constexpr int RoutedGateBytes=81920;
static_assert(sizeof(GateSelection)<=73728);
static_assert(73728+sizeof(GateRouting)<=RoutedGateBytes);
__global__ __launch_bounds__(256,2) void routed_gate(Rank a,const float *router,
    const float *bias,int *ids,float *prob,int needNorm,float routeScale) {
    extern __shared__ __align__(1024) unsigned char arena[];
    auto &s=*reinterpret_cast<GateSelection*>(arena);
    auto &r=*reinterpret_cast<GateRouting*>(arena+73728);
    int t=threadIdx.x,warp=t/32,lane=t%32;
    if(warp<a.rows)fastllm::cuda::SelectRouterTopK<256,8>(router+warp*256,bias,
        r.ids+warp*8,r.probability+warp*8,bias!=nullptr,needNorm,routeScale,s.top[warp]);
    s.masks[t]=0;__syncthreads();
    int expert=t<a.rows*8?r.ids[t]:0;
    if(t<a.rows*8)atomicOr(s.masks+expert,1ull<<t);
    __syncthreads();
    unsigned long long mask=t<a.rows*8?s.masks[expert]:0;
    int first=t<a.rows*8&&__ffsll(mask)-1==t,prefix=first;
    if(t<64){
        for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffffu,prefix,d);if(lane>=d)prefix+=v;}
        if(lane==31)s.sums[t/32]=prefix;
    }
    __syncthreads();
    if(t==0)r.unique=s.sums[0]+s.sums[1];
    if(first){int item=prefix-1+(t>=32?s.sums[0]:0),count=0;r.uniq[item]=expert;
        while(mask){r.members[item*8+count++]=__ffsll(mask)-1;mask&=mask-1;}r.counts[item]=count;}
    if(t<64)r.jobs[t]=t; if(t==0)r.count=a.rows*8;
    __syncthreads();
    if(blockIdx.x==0){
        if(t<a.rows*8){ids[t]=r.ids[t];prob[t]=r.probability[t];a.jobs[t]=t;}
        if(t==0){*a.count=a.rows*8;*a.nuniq=r.unique;}
        if(t<r.unique){a.uniq[t]=r.uniq[t];a.rowcounts[t]=r.counts[t];}
        for(int i=t;i<r.unique*8;i+=256)if(i%8<r.counts[i/8])a.members[i]=r.members[i];
    }
    __syncthreads();
    if(t>=128)return;
    a.count=&r.count;a.nuniq=&r.unique;a.uniq=r.uniq;a.members=r.members;
    a.rowcounts=r.counts;a.jobs=r.jobs;
    if(a.rows>=7 && r.unique>a.rows*4 && a.xmap)
      GroupedTmaGate<true>(a,r.probability,a.xmap);
    else if(r.unique<=a.rows*4){
      if(r.unique>=20)batched_matmul<true,false,2,true>(a,r.ids,r.probability);
      else batched_matmul<true,false,1,true>(a,r.ids,r.probability);
    }else if(a.rows>=7)plain_matmul<true,1,false,true>(a,r.ids,r.probability);
    else if(a.rows>=3)plain_matmul<true,2,false,true>(a,r.ids,r.probability);
    else plain_matmul<true,4,false,true>(a,r.ids,r.probability);
}
} // namespace fastllm_shared_flex_tp
