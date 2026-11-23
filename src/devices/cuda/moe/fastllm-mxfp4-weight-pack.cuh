#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm_mxfp4 {
__device__ __forceinline__ int sourceRow(int row,int n,bool gate) {
    return gate ? row/16*8+row%8+(row%16>=8 ? n/2 : 0) : row;
}
__device__ __forceinline__ size_t scaleOffset(int row,int col,int n) {
    int transposed=(row&~127)+(row%32)*4+(row%128)/32;
    return ((size_t)(col/4)*n+transposed)*4+col%4;
}
// Native E2M1 nibbles and K32 UE8M0 exponents arrive in one allocation.
// Only permute rows and scales; TMA unpacks the weights into shared memory.
static __global__ void packMxfp4(const uint8_t *src,uint8_t *dst,uint8_t *sf,int n,int k,bool gate) {
    int group=blockIdx.x*blockDim.x+threadIdx.x;
    if (group>=n*(k/32)) return;
    int row=group/(k/32), col=group%(k/32);
    int source=sourceRow(row,n,gate)*(k/32)+col;
    reinterpret_cast<uint4 *>(dst)[group]=reinterpret_cast<const uint4 *>(src)[source];
    sf[scaleOffset(row,col,n)]=src[(size_t)n*k/2+source];
}
}
