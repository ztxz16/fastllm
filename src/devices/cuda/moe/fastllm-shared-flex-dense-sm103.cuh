#pragma once
#include "devices/cuda/low_latency/dense.cuh"
// TP8 adapter: common BF16 arithmetic, backend-owned epoch and publication.
// See third_party/naivert_projection and third_party/naivert_dsa for provenance.
namespace fastllm_shared_flex_tp {
constexpr int DenseI=2048;
constexpr int DenseUpBytes=fastllm::cuda::low_latency::DenseGateUpSharedBytes<H>;
constexpr int DenseDownBytes=fastllm::cuda::low_latency::DenseDownSharedBytes<DenseI>+8*64*2;
__global__ __launch_bounds__(256,1) void dense_up_gate(const __grid_constant__ Rank a,
    const __nv_bfloat16 *x,const __nv_bfloat16 *gate,const __nv_bfloat16 *up,
    const float *gamma,float eps,__nv_bfloat16 *out) {
    if(blockIdx.x==0 && threadIdx.x==0)++*a.epoch;
    fastllm::cuda::low_latency::DenseGateUpBody<H,DenseI,true>(x,gate,up,gamma,eps,out,a.rows);
}
__global__ __launch_bounds__(256,1) void dense_down_reduce(const __grid_constant__ Rank a,const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,__nv_bfloat16 *out) {
    extern __shared__ __align__(128) unsigned char shared[];
    auto *input=reinterpret_cast<__nv_bfloat16*>(shared);
    auto *partial=reinterpret_cast<float*>(shared+8*DenseI*2);
    auto *local=reinterpret_cast<__nv_bfloat16*>(partial+8*128);
    fastllm::cuda::low_latency::DenseLoad<DenseI>(x,input,a.rows);
    auto pointer=[&](int row){return weight+size_t(row)*DenseI;};
    auto emit=[&](int row,int query,float value){local[query*64+row%64]=__float2bfloat16_rn(value);};
    fastllm::cuda::low_latency::projection::project_rows_mma<8,DenseI,2,4,8,128>(0,H/16,pointer,input,partial,emit);
    const int row=threadIdx.x/32,pair=blockIdx.x*32+threadIdx.x%32;
    if(row<a.rows) {
        const unsigned payload=reinterpret_cast<const unsigned*>(local+row*64)[threadIdx.x%32];
        const unsigned generation=unsigned(*a.epoch);
        publish_bf16(a,row,pair,generation,payload);
        const float2 total=collect_bf16(a,row,pair,generation,payload);
        reinterpret_cast<__nv_bfloat162*>(out+row*H)[pair]=__floats2bfloat162_rn(total.x,total.y);
    }
}
}
