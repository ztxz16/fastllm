#pragma once
#include "fastllm-moe-deepseekv41-cache.cuh"

namespace fastllm { namespace cuda { namespace dsv41_cache {
namespace batch_mma {
constexpr int Tokens = kExpertMmaRouteTile;
__device__ inline uint32_t Pair(const uint8_t *matrix, int row, int block, int pair,
        int inner, int outputs, bool planar) {
    const int blocks = inner / 32;
    const uint8_t *w = matrix + size_t(row) * blocks * (planar ? 16 : 17) + block * (planar ? 16 : 17);
    const unsigned scale = planar ? matrix[size_t(outputs) * inner / 2 + size_t(row) * blocks + block] : w[16];
    const float2 values = FP4Pair(w[pair], scale);
    const __nv_bfloat162 packed = __floats2bfloat162_rn(values.x, values.y);
    return *reinterpret_cast<const uint32_t *>(&packed);
}
__device__ inline void Mma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1, float *d) {
#if __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
}

// Four output fragments cover 32 gate and 32 up channels. Four independent
// K slices keep a whole FP8 activation block inside one CTA for its epilogue.
__device__ __forceinline__ void GateMmaBatchBody(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit, int topk,
        const int32_t *routeMap, int selected, int count, bool planar, const float *siluTable = nullptr) {
#if __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    using namespace batch_mma;
    const uint8_t *record = records + size_t(slots[selected]) * stride;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = lane / 4, part = lane % 4;
    const int split = warp / 4, fragment = warp % 4;
    const int col = blockIdx.x * 32 + (fragment % 2) * 16 + group;
    const int weightRow = col + (fragment / 2) * inter;
    float accum[2][4] = {};
    for (int block = split; block < hidden / 32; block += 4) {
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int pair = step * 8 + part;
            const uint32_t a0 = Pair(record,weightRow,block,pair,hidden,2*inter,planar);
            const uint32_t a1 = Pair(record,weightRow+8,block,pair,hidden,2*inter,planar);
            const uint32_t a2 = Pair(record,weightRow,block,pair+4,hidden,2*inter,planar);
            const uint32_t a3 = Pair(record,weightRow+8,block,pair+4,hidden,2*inter,planar);
#pragma unroll
            for (int tile = 0; tile < 2; ++tile) if (tile * 8 < count) {
                uint32_t b0 = 0, b1 = 0;
                if (tile * 8 + group < count) {
                    const int at = selected + tile * 8 + group;
                    const int route = routeMap ? routeMap[at] : at;
                    const auto *x = input + size_t(route / topk) * hidden + block * 32 + step * 16 + part * 2;
                    b0 = *reinterpret_cast<const uint32_t *>(x);
                    b1 = *reinterpret_cast<const uint32_t *>(x + 8);
                }
                Mma(a0,a1,a2,a3,b0,b1,accum[tile]);
            }
        }
    }
    __shared__ float partial[4][2][16][32];
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int c = (fragment % 2) * 16 + group;
        partial[split][fragment/2][tile*8+part*2][c] = accum[tile][0];
        partial[split][fragment/2][tile*8+part*2+1][c] = accum[tile][1];
        partial[split][fragment/2][tile*8+part*2][c+8] = accum[tile][2];
        partial[split][fragment/2][tile*8+part*2+1][c+8] = accum[tile][3];
    }
    __syncthreads();
    if (warp < count) {
        float gate = 0, up = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) { gate += partial[k][0][warp][lane]; up += partial[k][1][warp][lane]; }
        gate = BFloat(gate); up = BFloat(up);
        if (limit > 0) { gate=fminf(gate,limit);up=fmaxf(-limit,fminf(up,limit)); }
        const int route = routeMap ? routeMap[selected+warp] : selected+warp;
        const float value = BFloat(__fmul_rn(__fmul_rn(FastllmCudaBFloat16Silu(gate,siluTable),up),scores[route]));
        float amax = fmaxf(1e-4f,fabsf(value));
        for(int d=16;d;d>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
        const unsigned bits=__float_as_uint(amax/448.f);
        const int exponent=int((bits>>23)&255)-127+((bits&0x7fffff)!=0);
        const float scale=exp2f(float(exponent));
        const float q=fmaxf(-448.f,fminf(448.f,value/scale));
        activation[size_t(route)*inter+blockIdx.x*32+lane]=__float2bfloat16_rn(float(__nv_fp8_e4m3(q))*scale);
    }
#endif
}

__device__ __forceinline__ void DownMmaBatchBody(const __nv_bfloat16 *activation, const int32_t *slots,
        const uint8_t *records, float *output, int hidden, int inter, size_t stride,
        size_t downOffset, const int32_t *routeMap, int selected, int count, bool planar) {
#if __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    using namespace batch_mma;
    const auto *matrix=records+size_t(slots[selected])*stride+downOffset;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,part=lane%4;
    const int col=blockIdx.x*16+group;
    float accum[2][4]={};
    for(int block=warp;block<inter/32;block+=8){
#pragma unroll
        for(int step=0;step<2;++step){
            const int pair=step*8+part;
            const uint32_t a0=Pair(matrix,col,block,pair,inter,hidden,planar);
            const uint32_t a1=Pair(matrix,col+8,block,pair,inter,hidden,planar);
            const uint32_t a2=Pair(matrix,col,block,pair+4,inter,hidden,planar);
            const uint32_t a3=Pair(matrix,col+8,block,pair+4,inter,hidden,planar);
#pragma unroll
            for(int tile=0;tile<2;++tile)if(tile*8<count){
                uint32_t b0=0,b1=0;
                if(tile*8+group<count){
                    const int at=selected+tile*8+group,route=routeMap?routeMap[at]:at;
                    const auto *x=activation+size_t(route)*inter+block*32+step*16+part*2;
                    b0=*reinterpret_cast<const uint32_t*>(x);b1=*reinterpret_cast<const uint32_t*>(x+8);
                }
                Mma(a0,a1,a2,a3,b0,b1,accum[tile]);
            }
        }
    }
    __shared__ float partial[8][16][16];
#pragma unroll
    for(int tile=0;tile<2;++tile){
        partial[warp][tile*8+part*2][group]=accum[tile][0];
        partial[warp][tile*8+part*2+1][group]=accum[tile][1];
        partial[warp][tile*8+part*2][group+8]=accum[tile][2];
        partial[warp][tile*8+part*2+1][group+8]=accum[tile][3];
    }
    __syncthreads();
    const int token=threadIdx.x/16,channel=threadIdx.x%16;
    if(token<count){
        float value=0;
#pragma unroll
        for(int k=0;k<8;++k)value+=partial[k][token][channel];
        const int route=routeMap?routeMap[selected+token]:selected+token;
        output[size_t(route)*hidden+blockIdx.x*16+channel]=BFloat(value);
    }
#endif
}
// A CTA visits the eligible expert groups in one bounded route tile. Empty
// work scales with tiles instead of launching a full output grid per route.
__global__ void GateMmaBatch(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit, int topk,
        const int32_t *routeMap, int routes, bool planar, const float *siluTable = nullptr) {
    constexpr int capacity = batch_mma::Tokens;
    const int end = min(routes, int(blockIdx.y + 1) * capacity);
    for (int selected = blockIdx.y * capacity; selected < end;) {
        const int count = ExpertRouteTile<capacity>(slots, selected, routes);
        if (count > 1) {
            GateMmaBatchBody(input,slots,records,scores,activation,hidden,inter,
                stride,limit,topk,routeMap,selected,count,planar,siluTable);
            __syncthreads();
        }
        selected += max(count,1);
    }
}
__global__ void DownMmaBatch(const __nv_bfloat16 *activation, const int32_t *slots,
        const uint8_t *records, float *output, int hidden, int inter, size_t stride,
        size_t downOffset, const int32_t *routeMap, int routes, bool planar) {
    constexpr int capacity = batch_mma::Tokens;
    const int end = min(routes, int(blockIdx.y + 1) * capacity);
    for (int selected = blockIdx.y * capacity; selected < end;) {
        const int count = ExpertRouteTile<capacity>(slots, selected, routes);
        if (count > 1) {
            DownMmaBatchBody(activation,slots,records,output,hidden,inter,
                stride,downOffset,routeMap,selected,count,planar);
            __syncthreads();
        }
        selected += max(count,1);
    }
}
} } }
