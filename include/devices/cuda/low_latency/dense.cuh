#pragma once
#include "bf16_projection.cuh"
// BF16 rounded gate/up -> rounded SiLU -> rounded product. Optional RMSNorm
// retains two BF16 rounding boundaries. Original row-major weights; no packing,
// global scratch, model state or collective state. The caller checks image and
// shared-memory availability before launch, otherwise uses its existing path.
namespace fastllm::cuda::low_latency {
using BF16 = __nv_bfloat16;
template<int H, int Rows=8> constexpr int DenseGateUpSharedBytes = Rows*H*2+8*128*4+32*Rows*4;
template<int K, int Rows=8> constexpr int DenseDownSharedBytes = Rows*K*2+8*128*4;
template<int Width, int Rows=8> __device__ __forceinline__ void DenseLoad(const __nv_bfloat16 *x,
    __nv_bfloat16 *shared, int rows) {
    for (int item=threadIdx.x;item<Rows*(Width/8);item+=256) {
        const int row=item/(Width/8),col=item%(Width/8)*8;
        auto *dst=reinterpret_cast<uint4*>(shared+row*Width+col);
        *dst=row<rows?*reinterpret_cast<const uint4*>(x+row*Width+col):make_uint4(0,0,0,0);
    }
    __syncthreads();
}
// Vector loads retain each row in registers while its warp reduces the RMS.
// The two BF16 normalization boundaries remain the native FastLLM contract.
template<int H, int Rows=8> __device__ void DenseNormalize(const __nv_bfloat16 *x,const float *gamma,
    __nv_bfloat16 *shared,int rows,float eps) {
    static_assert(H % 256 == 0);
    const int row=threadIdx.x/32,lane=threadIdx.x%32;
    if (row < Rows) {
    uint4 staged[H/256];float sum=0.f;
    #pragma unroll
    for(int piece=0;piece<H/256;++piece) {
        staged[piece]=row<rows?__ldg(reinterpret_cast<const uint4*>(x+row*H)+(lane+piece*32)):make_uint4(0,0,0,0);
        const unsigned words[]={staged[piece].x,staged[piece].y,staged[piece].z,staged[piece].w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            const float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words+j));
            sum=fmaf(v.x,v.x,sum);sum=fmaf(v.y,v.y,sum);
        }
    }
    #pragma unroll
    for(int step=16;step;step>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,step);
    const float inv=rsqrtf(sum/H+eps);
    #pragma unroll
    for(int piece=0;piece<H/256;++piece) {
        const int col=(lane+piece*32)*8;
        const unsigned words[]={staged[piece].x,staged[piece].y,staged[piece].z,staged[piece].w};
        const float4 lo=*reinterpret_cast<const float4*>(gamma+col),hi=*reinterpret_cast<const float4*>(gamma+col+4);
        const float g[]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};uint4 out;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words+j));
            const __nv_bfloat162 norm=__floats2bfloat162_rn(float(__float2bfloat16_rn(v.x*inv))*g[2*j],float(__float2bfloat16_rn(v.y*inv))*g[2*j+1]);
            reinterpret_cast<unsigned*>(&out)[j]=*reinterpret_cast<const unsigned*>(&norm);
        }
        *reinterpret_cast<uint4*>(shared+row*H+col)=out;
    }
    }
    __syncthreads();
}
template<int H, int DenseI, bool Normalize=true, int Rows=8>
__device__ __forceinline__ void DenseGateUpBody(const BF16 *x,const BF16 *gate,
    const BF16 *up,const float *gamma,float eps,BF16 *out,int rows) {
    extern __shared__ __align__(128) unsigned char shared[];
    auto *input=reinterpret_cast<__nv_bfloat16*>(shared);
    auto *partial=reinterpret_cast<float*>(shared+Rows*H*2);
    auto *projected=partial+8*128;
    if constexpr (Normalize) DenseNormalize<H,Rows>(x,gamma,input,rows,eps);
    else DenseLoad<H,Rows>(x,input,rows);
    auto weight=[&](int row) {
        const int index=(row/32)*16+row%16;
        return (row%32<16?gate:up)+size_t(index)*H;
    };
    auto emit=[&](int row,int query,float value) {
        projected[(row%32)*Rows+query]=__bfloat162float(__float2bfloat16_rn(value));
    };
    projection::project_rows_mma<Rows,H,4,2,8,(H%512==0?128:32)>(0,DenseI*2/16,weight,input,partial,emit);
    for(int item=threadIdx.x;item<16*Rows;item+=256) {
        const int feature=item/Rows,row=item%Rows;
        if(row<rows) {
            const float g=projected[feature*Rows+row],u=projected[(feature+16)*Rows+row];
            const float silu=__bfloat162float(__float2bfloat16_rn(g/(1.f+__expf(-g))));
            out[row*DenseI+blockIdx.x*16+feature]=__float2bfloat16_rn(silu*u);
        }
    }
}

template<int Hidden,int Intermediate,bool Normalize=true,int Rows=8>
__global__ __launch_bounds__(256,1) void DenseGateUp(const BF16 *x,const BF16 *gate,
    const BF16 *up,const float *gamma,float eps,BF16 *out,int rows) {
    static_assert(Hidden%256==0 && Intermediate%16==0);
    const int first=blockIdx.y*Rows, valid=min(Rows,rows-first);
    DenseGateUpBody<Hidden,Intermediate,Normalize,Rows>(x+size_t(first)*Hidden,gate,up,gamma,eps,
        out+size_t(first)*Intermediate,valid);
}
// Down projection with optional local residual. TP callers reduce local BF16
// projections before adding residual; they must not pass a residual here.
template<int Intermediate,int Hidden,bool Residual=false,int Rows=8>
__global__ __launch_bounds__(256,1) void DenseDown(const BF16 *x,const BF16 *weight,
    const BF16 *residual,BF16 *out,int rows) {
    static_assert(Intermediate%256==0 && Hidden%64==0);
    extern __shared__ __align__(128) unsigned char shared[];
    const int first=blockIdx.y*Rows,valid=min(Rows,rows-first);
    auto *input=reinterpret_cast<BF16*>(shared);
    auto *partial=reinterpret_cast<float*>(shared+Rows*Intermediate*2);
    DenseLoad<Intermediate,Rows>(x+size_t(first)*Intermediate,input,valid);
    auto pointer=[&](int row){return weight+size_t(row)*Intermediate;};
    auto emit=[&](int col,int row,float value){if(row<valid){
        BF16 v=__float2bfloat16_rn(value);
        if constexpr (Residual) v=__float2bfloat16_rn(float(v)+float(residual[size_t(first+row)*Hidden+col]));
        out[size_t(first+row)*Hidden+col]=v;
    }};
    projection::project_rows_mma<Rows,Intermediate,2,4,8,128>(0,Hidden/16,pointer,input,partial,emit);
}
} // namespace fastllm::cuda::low_latency
