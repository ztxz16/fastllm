// Resident V4.1 GGUF experts: full-batch routes and grouped compressed projections.
#include "../fastllm-gguf-mmvq-common.cuh"
#include "../fastllm-gguf-q3-k-gemv.cuh"
#include "../fastllm-gguf-iq1-dot.cuh"
#include "fastllm-moe-gguf-q8.cuh"
#include "fastllm-cuda-v41-gguf.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>
#include "../fastllm-gguf-iq1-grid.cuh"
#include "../fastllm-gguf-iq2-dot.cuh"
#include <type_traits>
#include <climits>
#include "fastllm-moe-route-groups.cuh"

namespace {
__device__ __forceinline__ float V41Fp8Boundary(float x) {
    const float maximum = fmaxf(warp_reduce_max(fabsf(x)), 1e-4f);
    const unsigned bits = __float_as_uint(maximum * (1.0f/448.0f));
    const int exponent = int((bits>>23)&255)-127+int((bits&0x7fffff)!=0);
    const float scale = exp2f(float(exponent));
    return __bfloat162float(__float2bfloat16_rn(
        float(__nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x/scale))))*scale));
}
__device__ __forceinline__ void V41StoreQ8(float x, block_q8_1 &q, int lane) {
    const float maximum = warp_reduce_max(fabsf(x));
    const float sum = warp_reduce_sum(x), scale = maximum/127.f;
    q.qs[lane] = maximum == 0 ? 0 : int8_t(roundf(x/scale));
    if (lane == 0) q.ds = __floats2half2_rn(scale, sum);
}
// The same FP8 block-32 -> BF16 -> Q8_1 boundaries as the ordinary CUDA
// Linear path. Input conversion is shared by every selected expert.
template<bool Fp8Boundary>
__global__ void V41GGUFQuantize(const __nv_bfloat16 *input, block_q8_1 *output, int columns) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= columns) return;
    float x = __bfloat162float(input[size_t(blockIdx.y)*columns+c]);
    if constexpr (Fp8Boundary) x = V41Fp8Boundary(x);
    auto &q = output[size_t(blockIdx.y)*(columns/32)+c/32];
    V41StoreQ8(x, q, c%32);
}

// Keep projection dispatch inline: sharing the fused gate/up helper changes
// NVCC register allocation for generic projection kernels despite equal math.
template<ggml_type Type, int VirtualWarps = 4, int LaneWidth = 32, bool Packed = false>
__device__ __forceinline__ float V41GGUFDot(const void *weight, const block_q8_1 *x, int columns, const uint32_t *iqGrid, int packedRow = 0) {
    if constexpr (Type == GGML_TYPE_Q2_0) {
        return gguf_cache_q8::RowDot<Type>(weight, x, columns, nullptr);
    } else {
        constexpr int qi = [] {
            if constexpr (Type == GGML_TYPE_IQ1_S) return QI1_S;
            else if constexpr (Type == GGML_TYPE_IQ1_M) return QI1_M;
            else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) return 16;
            else return ggml_cuda_type_traits<Type>::qi;
        }();
        constexpr int vdr = (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) ? 2 : get_vdr_mmvq(Type);
        constexpr int step = vdr*LaneWidth*VirtualWarps/qi;
        const int lane = threadIdx.x%LaneWidth;
        float partial[VirtualWarps];
#pragma unroll
        for (int warp = 0; warp < VirtualWarps; ++warp) {
            const int tid = lane+LaneWidth*warp;
            float value = 0;
            constexpr int qk = Type == GGML_TYPE_Q8_0 ? 32 : 256;
            for (int b = tid/(qi/vdr); b < columns/qk; b += step) {
                if constexpr (Type == GGML_TYPE_IQ1_S)
                    value += vec_dot_iq1_s_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_IQ1_M)
                    value += vec_dot_iq1_m_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_IQ2_XXS)
                    value += iq2_decode::DotXXS<(VirtualWarps != 4), Packed>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid), packedRow);
                else if constexpr (Type == GGML_TYPE_IQ2_XS)
                    value += iq2_decode::DotXS<(VirtualWarps != 4), Packed>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid), packedRow);
                else if constexpr (Type == GGML_TYPE_IQ2_S)
                    value += iq2_decode::DotS<(VirtualWarps != 4), Packed>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid), packedRow);
                else if constexpr (Type == GGML_TYPE_Q3_K)
                    value += fastllm_gguf_q3_k::Dot(weight, x+b*8, b, vdr*(tid%(qi/vdr)));
                else value += get_vec_dot_q_cuda(Type)(weight, x+b*(qk/32), b, vdr*(tid%(qi/vdr)));
            }
            partial[warp] = value;
        }
        // Retain the four-warp reference order. A half warp calculates both
        // corresponding logical lanes before the remaining shuffle reductions.
        float sum;
        if constexpr (VirtualWarps == 8 && LaneWidth == 16) {
            float a = partial[0], b = partial[1];
#pragma unroll
            for (int warp=1;warp<4;++warp) {a+=partial[2*warp];b+=partial[2*warp+1];}
            sum = a+b;
        } else if constexpr (VirtualWarps == 16 && LaneWidth == 8) {
            float a=partial[0], b=partial[1], c=partial[2], d=partial[3];
#pragma unroll
            for (int warp=1; warp<4; ++warp) {
                a+=partial[4*warp]; b+=partial[4*warp+1];
                c+=partial[4*warp+2]; d+=partial[4*warp+3];
            }
            // Reconstruct the reference xor-16, then xor-8 reductions locally.
            sum=(a+c)+(b+d);
        } else {
            sum=partial[0];
#pragma unroll
            for (int warp=1;warp<VirtualWarps;++warp)sum+=partial[warp];
        }
#pragma unroll
        for (int mask = LaneWidth/2; mask; mask >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, mask, LaneWidth);
        return sum;
    }
}

template<ggml_type Type, bool Packed = false>
__device__ __forceinline__ float V41BlockDot(const void *weight,const block_q8_1 *x,int b,int q,const uint32_t*grid,int packedRow = 0) {
    if constexpr(Type==GGML_TYPE_IQ1_S)return vec_dot_iq1_s_q8_1_grid(weight,x+b*8,b,q,grid);
    else if constexpr(Type==GGML_TYPE_IQ1_M)return vec_dot_iq1_m_q8_1_grid(weight,x+b*8,b,q,grid);
    else if constexpr(Type==GGML_TYPE_IQ2_XXS)return iq2_decode::DotXXS<true,Packed>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid),packedRow);
    else if constexpr(Type==GGML_TYPE_IQ2_XS)return iq2_decode::DotXS<true,Packed>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid),packedRow);
    else if constexpr(Type==GGML_TYPE_IQ2_S)return iq2_decode::DotS<true,Packed>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid),packedRow);
    else if constexpr(Type==GGML_TYPE_Q3_K)return fastllm_gguf_q3_k::Dot(weight,x+b*8,b,q);
    else return get_vec_dot_q_cuda(Type)(weight,x+b*8,b,q);
}
template<ggml_type Type,int LaneWidth,bool Packed = false>
__device__ __forceinline__ float2 V41PairDot(const void *gate,const void *up,const block_q8_1*x,int columns,const uint32_t*grid,int gateRow = 0,int upRow = 0) {
    if constexpr(Type==GGML_TYPE_Q8_0)return make_float2(V41GGUFDot<Type,4,LaneWidth>(gate,x,columns,grid),V41GGUFDot<Type,4,LaneWidth>(up,x,columns,grid));
    else if constexpr(Type==GGML_TYPE_Q2_0)return make_float2(V41GGUFDot<Type,1,LaneWidth>(gate,x,columns,grid),V41GGUFDot<Type,1,LaneWidth>(up,x,columns,grid));
    else {
        constexpr int qi=[] {
            if constexpr(Type==GGML_TYPE_IQ1_S)return QI1_S;
            else if constexpr(Type==GGML_TYPE_IQ1_M)return QI1_M;
            else if constexpr(Type==GGML_TYPE_IQ2_XXS||Type==GGML_TYPE_IQ2_XS||Type==GGML_TYPE_IQ2_S)return 16;
            else return ggml_cuda_type_traits<Type>::qi;
        }();
        constexpr int vdr=(Type==GGML_TYPE_IQ2_XXS||Type==GGML_TYPE_IQ2_XS||Type==GGML_TYPE_IQ2_S)?2:get_vdr_mmvq(Type);
        constexpr int step=vdr*LaneWidth/qi;
        const int lane=threadIdx.x%LaneWidth;
        float pg[4],pu[4];
#pragma unroll
        for(int w=0;w<4;++w) {
            const int tid=lane+LaneWidth*w,q=vdr*(tid%(qi/vdr));
            float g=0,u=0;
            for(int b=tid/(qi/vdr);b<columns/256;b+=step*4) {
                g+=V41BlockDot<Type,Packed>(gate,x,b,q,grid,gateRow);
                u+=V41BlockDot<Type,Packed>(up,x,b,q,grid,upRow);
            }
            pg[w]=g;pu[w]=u;
        }
        float g=pg[0],u=pu[0];
#pragma unroll
        for(int w=1;w<4;++w){g+=pg[w];u+=pu[w];}
#pragma unroll
        for(int mask=LaneWidth/2;mask;mask>>=1){g+=__shfl_xor_sync(0xffffffff,g,mask,LaneWidth);u+=__shfl_xor_sync(0xffffffff,u,mask,LaneWidth);}
        return make_float2(g,u);
    }
}

__device__ const uint8_t *V41Weight(const uint8_t *const *weights,
        const int32_t *indices, int route, int experts, bool gate) {
    const int expert = indices[route];
    return expert >= 0 && expert < experts ? weights[2*expert+(gate ? 0 : 1)] : nullptr;
}
__device__ const uint8_t *V41Weight(const FastllmCudaMoeGGUFCacheView &view,
        const int32_t *, int route, int, bool gate) {
    const int slot = view.routeSlots[route];
    if (slot < 0) return nullptr;
    if (view.recordPointers)
        return view.recordPointers[slot] + (gate ? 0 : view.downOffset);
    const size_t offset = view.slotOffsets ? view.slotOffsets[slot] : size_t(slot)*view.recordStride;
    return view.records + offset + (gate ? 0 : view.downOffset);
}

__device__ int V41Route(const uint8_t *const *, int route) { return route; }
__device__ int V41Route(const FastllmCudaMoeGGUFCacheView &view, int route) {
    return view.routeMap ? view.routeMap[route] : route;
}

__device__ int V41WeightRow(const uint8_t *const *, int row, int, bool) { return row; }
__device__ int V41WeightRow(const FastllmCudaMoeGGUFCacheView &view, int row, int rows, bool gate) {
    // NUMA shards concatenate into [gate0, up0, gate1, up1, ...]. Preserve
    // the canonical output order without copying either projection matrix.
    return gate && view.numaGateType >= 0 ? 2*(row % (rows/2)) + row/(rows/2) : row;
}

// Compact only on the GPU: no device-to-host route-count synchronization.
// The launch grid is bounded by the public route count; inactive CTAs exit
// before loading a codebook or writing a zero expert matrix.
struct V41EpWeights {
    const uint8_t *const *weights;
    const int32_t *routes; // count, packed -> original, original -> packed
};
__device__ const uint8_t *V41Weight(V41EpWeights view, const int32_t *indices,
        int route, int experts, bool gate) {
    return V41Weight(view.weights, indices, view.routes[1+route], experts, gate);
}
__device__ int V41Route(V41EpWeights view, int route) { return view.routes[1+route]; }
__device__ int V41WeightRow(V41EpWeights, int row, int, bool) { return row; }
template<class Weights> __device__ bool V41ActiveRoute(Weights, int) { return true; }
__device__ bool V41ActiveRoute(V41EpWeights view, int route) { return route < view.routes[0]; }

__global__ void V41EpRoutes(const uint8_t *const *weights, const int32_t *indices,
        int32_t *routes, int count, int experts) {
    if (threadIdx.x) return;
    int active = 0;
    for (int r = 0; r < count; ++r) {
        const int e = indices[r];
        const bool local = e >= 0 && e < experts && weights[2*e] && weights[2*e+1];
        routes[1+count+r] = local ? active : -1;
        if (local) routes[1+active++] = r;
    }
    routes[0] = active;
}

__global__ void V41EpSwigluQ8(const __nv_bfloat16 *gateup, block_q8_1 *output,
        const int32_t *routes, const float *scores, int inter, float limit) {
    const int route = blockIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    if (route >= routes[0] || c >= inter) return;
    const auto *input = gateup + size_t(route)*2*inter;
    float gate = __bfloat162float(input[c]), up = __bfloat162float(input[c+inter]);
    if (limit > 0) {
        gate = fminf(gate, limit);
        up = fmaxf(-limit, fminf(up, limit));
    }
    float x = (gate/(1.f+expf(-gate)))*up;
    x *= scores[routes[1+route]];
    x = __bfloat162float(__float2bfloat16_rn(x));
    x = V41Fp8Boundary(x);
    V41StoreQ8(x, output[size_t(route)*(inter/32)+c/32], c%32);
}

template<class Weights> __device__ bool V41WeightPacked(Weights, bool) { return false; }
__device__ bool V41WeightPacked(const FastllmCudaMoeGGUFCacheView &view, bool gate) {
    const int storage = gate ? view.numaGateType : view.numaDownType;
    return storage >= 0 && storage != (gate ? view.gateType : view.downType);
}

template<ggml_type Type, bool Gate, class Weights, int RowsPerBlock = 8, int StaticColumns = 0, int VirtualWarps = 4, int LaneWidth = 32>
__global__ void V41GGUFProjection(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int columns, int rows, size_t rowBytes) {
    if constexpr (StaticColumns) {
        columns = StaticColumns;
    }
    const int route = blockIdx.y;
    if (!V41ActiveRoute(weights, route)) return;
    const uint8_t *weight = V41Weight(weights, indices, route, experts, Gate);
    const int firstRow = blockIdx.x*RowsPerBlock + threadIdx.x/LaneWidth;
    // The decision is uniform within the CTA. Missing TP experts must still
    // clear every output row, but need neither the codebook nor activation.
    if (!weight) {
        for (int row = firstRow; row < rows && row < (blockIdx.x+1)*RowsPerBlock; row += 256/LaneWidth)
            if (threadIdx.x%LaneWidth == 0) output[size_t(route)*rows+row] = __float2bfloat16_rn(0.f);
        return;
    }
    const auto *x = input + size_t(Gate ? V41Route(weights, route)/topk : route)*(columns/32);
    extern __shared__ uint32_t activation[];
    constexpr int gridWords = (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_IQ2_S) ? 2048 :
        Type == GGML_TYPE_IQ2_XS ? 1024 : Type == GGML_TYPE_IQ2_XXS ? 512 : 2;
    __shared__ uint64_t gridStorage[gridWords/2];
    auto *iqGrid = reinterpret_cast<uint32_t *>(gridStorage);
    if constexpr (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x)
            iqGrid[i] = fastllm_iq1_grid_packed[i];
    } else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
        const uint64_t *source = Type == GGML_TYPE_IQ2_S ? iq2s_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
        for (int i = threadIdx.x; i < gridWords/2; i += blockDim.x) gridStorage[i] = source[i];
    }
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    for (int row = firstRow; row < rows && row < (blockIdx.x+1)*RowsPerBlock; row += 256/LaneWidth) {
        const int sourceRow = V41WeightRow(weights, row, rows, Gate);
        float value;
        if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
            if (V41WeightPacked(weights, Gate))
                value = V41GGUFDot<Type, VirtualWarps, LaneWidth, true>(weight+size_t(sourceRow/4)*4*rowBytes,
                    reinterpret_cast<const block_q8_1 *>(activation), columns, iqGrid, sourceRow%4);
            else
                value = V41GGUFDot<Type, VirtualWarps, LaneWidth>(weight+size_t(sourceRow)*rowBytes,
                    reinterpret_cast<const block_q8_1 *>(activation), columns, iqGrid);
        } else {
            value = V41GGUFDot<Type, VirtualWarps, LaneWidth>(weight+size_t(sourceRow)*rowBytes,
                reinterpret_cast<const block_q8_1 *>(activation), columns, iqGrid);
        }
        if (threadIdx.x%LaneWidth == 0) output[size_t(route)*rows+row] = __float2bfloat16_rn(value);
    }
}

// Expert groups span the complete route list. Tile is only the register
// tile: every group is processed to its end, independent of DSpark or
// request count. A weight subgroup is decoded once for all Tile activations.
__global__ void V41CacheGroupKeys(FastllmCudaMoeGGUFCacheView view,
        uint64_t *keys, int *entries, int routes) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < routes) {
        keys[r] = reinterpret_cast<uint64_t>(V41Weight(view, nullptr, r, 0, true));
        entries[r] = r;
    }
}
template<ggml_type Type, bool Packed>
__device__ __forceinline__ iq2_decode::Decoded32 V41IQ2Decode(
        const void *weight, int b, int q, const uint64_t *grid, int row) {
    if constexpr (Type == GGML_TYPE_IQ2_XXS) return iq2_decode::DecodeXXS<Packed>(weight,b,q,grid,row);
    else if constexpr (Type == GGML_TYPE_IQ2_XS) return iq2_decode::DecodeXS<Packed>(weight,b,q,grid,row);
    else return iq2_decode::DecodeS<Packed>(weight,b,q,grid,row);
}
template<ggml_type Type, bool Packed, int LaneWidth, int Tile>
__device__ __forceinline__ void V41IQ2BatchDot(const uint8_t *weight,
        const block_q8_1 *const (&input)[Tile], int columns, int sourceRow, size_t rowBytes,
        const uint64_t *grid, float (&sum)[Tile]) {
    const int lane = threadIdx.x % LaneWidth;
    weight += Packed ? size_t(sourceRow/4)*4*rowBytes : size_t(sourceRow)*rowBytes;
    constexpr int Parts = 32/LaneWidth;
    float partial[Parts][Tile] = {};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int p = 0; p < Parts; ++p) {
            const int tid = lane + p*LaneWidth + w*32;
            float value[Tile] = {};
            for (int b = tid/8; b < columns/256; b += 16) {
                const auto decoded = V41IQ2Decode<Type,Packed>(weight,b,2*(tid%8),grid,sourceRow%4);
#pragma unroll
                for (int t = 0; t < Tile; ++t)
                    value[t] += decoded.template Dot<Type == GGML_TYPE_IQ2_XXS>(input[t][b*8+tid%8]);
            }
#pragma unroll
            for (int t = 0; t < Tile; ++t) partial[p][t] += value[t];
        }
    }
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if constexpr (Parts == 4) sum[t] = (partial[0][t]+partial[2][t])+(partial[1][t]+partial[3][t]);
        else sum[t] = partial[0][t];
    }
#pragma unroll
    for (int mask = LaneWidth/2; mask; mask >>= 1)
#pragma unroll
        for (int t = 0; t < Tile; ++t) sum[t] += __shfl_xor_sync(0xffffffff, sum[t], mask, LaneWidth);
}

// XXS gate uses a smaller register tile to keep decoded weights and both
// projections in registers; the whole expert group is still processed.
template<ggml_type Type, bool Gate, int StaticColumns, int Tile = (Gate && Type == GGML_TYPE_IQ2_XXS) ? 2 : 4, int OutputTile = Gate ? 8 : 32, int LaneWidth = Gate ? 32 : 8>
__global__ __launch_bounds__(Gate ? 128 : 256)
void V41IQ2Grouped(const block_q8_1 *input, __nv_bfloat16 *output,
        FastllmCudaMoeGGUFCacheView view, const float *scores, int topk,
        int columns, int rows, size_t rowBytes, float limit,
        const uint64_t *records, const int *starts, const int *entries, const int *groupCount) {
    if constexpr (StaticColumns) columns = StaticColumns;
    const int rowTiles = (rows+OutputTile-1)/OutputTile;
    const int group = blockIdx.x / rowTiles;
    if (group >= *groupCount) return;
    const int firstRow = (blockIdx.x % rowTiles)*OutputTile;
    const int begin=starts[group],end=starts[group+1],blocks=columns/32;
    const auto *weight = reinterpret_cast<const uint8_t *>(records[group]) + (Gate ? 0 : view.downOffset);
    constexpr int GridSize = Type == GGML_TYPE_IQ2_XXS ? 256 : Type == GGML_TYPE_IQ2_XS ? 512 : 1024;
    __shared__ uint64_t grid[GridSize];
    const uint64_t *source = Type == GGML_TYPE_IQ2_XXS ? iq2xxs_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2s_grid;
    for (int i = threadIdx.x; i < GridSize; i += blockDim.x) grid[i] = source[i];
    __syncthreads();
    const bool packed = V41WeightPacked(view, Gate);
    for (int first = begin; first < end; first += Tile) {
        const int active=min(Tile,end-first);
        const block_q8_1 *x[Tile];
#pragma unroll
        for (int t=0;t<Tile;++t) {
            const int route = entries[first+(t<active?t:0)];
            const int token = Gate ? V41Route(view,route)/topk : route;
            x[t] = input + size_t(token)*blocks;
        }
        for (int r0 = threadIdx.x/LaneWidth; r0 < OutputTile; r0 += blockDim.x/LaneWidth) {
            const int row = firstRow+r0;
            if (row >= rows) continue;
            const int r = V41WeightRow(view,row,Gate ? rows*2 : rows,Gate);
            float g[Tile], u[Tile];
            const int ru = Gate ? V41WeightRow(view,row+rows,rows*2,true) : 0;
            if (active == 1) {
                // A singleton (or the last partial tile) has no weight reuse.
                // Reuse the scalar dot arithmetic without padded row work.
                if constexpr (Gate) {
                    float2 pair;
                    if (packed) pair=V41PairDot<Type,LaneWidth,true>(
                        weight+size_t(r/4)*4*rowBytes,weight+size_t(ru/4)*4*rowBytes,
                        x[0],columns,reinterpret_cast<const uint32_t *>(grid),r%4,ru%4);
                    else pair=V41PairDot<Type,LaneWidth>(weight+size_t(r)*rowBytes,weight+size_t(ru)*rowBytes,
                        x[0],columns,reinterpret_cast<const uint32_t *>(grid));
                    g[0]=pair.x;u[0]=pair.y;
                } else {
                    if (packed) g[0]=V41GGUFDot<Type,128/LaneWidth,LaneWidth,true>(
                        weight+size_t(r/4)*4*rowBytes,x[0],columns,reinterpret_cast<const uint32_t *>(grid),r%4);
                    else g[0]=V41GGUFDot<Type,128/LaneWidth,LaneWidth>(
                        weight+size_t(r)*rowBytes,x[0],columns,reinterpret_cast<const uint32_t *>(grid));
                }
            } else {
                if (packed) V41IQ2BatchDot<Type,true,LaneWidth>(weight,x,columns,r,rowBytes,grid,g);
                else V41IQ2BatchDot<Type,false,LaneWidth>(weight,x,columns,r,rowBytes,grid,g);
                if constexpr (Gate) {
                    if (packed) V41IQ2BatchDot<Type,true,LaneWidth>(weight,x,columns,ru,rowBytes,grid,u);
                    else V41IQ2BatchDot<Type,false,LaneWidth>(weight,x,columns,ru,rowBytes,grid,u);
                }
            }
            if (threadIdx.x%LaneWidth == 0) {
#pragma unroll
                for (int t = 0; t < Tile; ++t) if (t < active) {
                    const int route = entries[first+(t<active?t:0)];
                    float value = g[t];
                    if constexpr (Gate) {
                        value = __bfloat162float(__float2bfloat16_rn(value));
                        float up = __bfloat162float(__float2bfloat16_rn(u[t]));
                        if (limit > 0) { value = fminf(value,limit); up = fmaxf(-limit,fminf(up,limit)); }
                        value = (value/(1.f+expf(-value)))*up*scores[route];
                    }
                    output[size_t(route)*rows+row] = __float2bfloat16_rn(value);
                }
            }
        }
    }
}
bool V41IQ2Type(int type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
}
template<bool Gate>
void V41IQ2BatchProject(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores, int topk,
        int columns, int rows, float limit, const fastllm::moe_routes::Plan &plan) {
    constexpr int tile = Gate ? 8 : 32, threads = Gate ? 128 : 256;
    const unsigned blocks = unsigned((rows+tile-1)/tile)*unsigned(plan.count);
    switch (type) {
#define IQ2_LAUNCH(Type, Static) \
        V41IQ2Grouped<GGML_TYPE_##Type,Gate,Static><<<blocks,threads,0,cudaStreamPerThread>>>( \
            input,output,view,scores,topk,columns,rows,ggml_row_size(GGML_TYPE_##Type,columns),limit, \
            plan.records,plan.starts,plan.entries,plan.groups)
#define IQ2_BATCH(Type) case GGML_TYPE_##Type: \
        if (columns == (Gate ? 5120 : 2304)) { IQ2_LAUNCH(Type, (Gate ? 5120 : 2304)); } \
        else { IQ2_LAUNCH(Type, 0); } break;
        IQ2_BATCH(IQ2_XXS) IQ2_BATCH(IQ2_XS) IQ2_BATCH(IQ2_S)
#undef IQ2_BATCH
#undef IQ2_LAUNCH
    }
}

// Fuse both projections and weighted SwiGLU; FP8/Q8 conversion stays separate.
template<ggml_type Type, class Weights, int Tile = 8, int LaneWidth = 32, int Threads = 256, int StaticColumns = 0>
__device__ __forceinline__ void V41GGUFGateUpActBody(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    if constexpr (StaticColumns) columns = StaticColumns;
    const int route = blockIdx.y;
    const uint8_t *weight = V41Weight(weights, indices, route, experts, true);
    // Inactive cache routes still produce zero activation, but do not need
    // to stage the IQ codebook or input shared by the active projections.
    if (!weight) {
        const float zero = 0.f * (scores ? scores[route] : 1.f);
        for (int r = threadIdx.x; r < Tile; r += Threads) {
            const int row = blockIdx.x * Tile + r;
            if (row < inter) output[size_t(route) * inter + row] = __float2bfloat16_rn(zero);
        }
        return;
    }
    const auto *x = input + size_t(V41Route(weights, route)/topk)*(columns/32);
    extern __shared__ uint32_t activation[];
    constexpr int gridWords = (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_IQ2_S) ? 2048 :
        Type == GGML_TYPE_IQ2_XS ? 1024 : Type == GGML_TYPE_IQ2_XXS ? 512 : 2;
    __shared__ uint64_t gridStorage[gridWords/2];
    auto *iqGrid = reinterpret_cast<uint32_t *>(gridStorage);
    if constexpr (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += Threads) iqGrid[i] = fastllm_iq1_grid_packed[i];
    } else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
        const uint64_t *source = Type == GGML_TYPE_IQ2_S ? iq2s_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
        for (int i = threadIdx.x; i < gridWords/2; i += Threads) gridStorage[i] = source[i];
    }
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += Threads)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    for (int r = threadIdx.x/LaneWidth; r < Tile; r += Threads/LaneWidth) {
        const int row = blockIdx.x*Tile+r;
        float g=0, u=0;
        if (row < inter) {
            const int gateRow = V41WeightRow(weights,row,2*inter,true);
            const int upRow = V41WeightRow(weights,row+inter,2*inter,true);
            float2 pair;
            if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
                if (V41WeightPacked(weights,true))
                    pair=V41PairDot<Type,LaneWidth,true>(
                        weight+size_t(gateRow/4)*4*rowBytes,weight+size_t(upRow/4)*4*rowBytes,
                        reinterpret_cast<const block_q8_1 *>(activation),columns,iqGrid,gateRow%4,upRow%4);
                else
                    pair=V41PairDot<Type,LaneWidth>(weight+size_t(gateRow)*rowBytes,weight+size_t(upRow)*rowBytes,
                        reinterpret_cast<const block_q8_1 *>(activation),columns,iqGrid);
            } else {
                pair=V41PairDot<Type,LaneWidth>(weight+size_t(gateRow)*rowBytes,weight+size_t(upRow)*rowBytes,
                    reinterpret_cast<const block_q8_1 *>(activation),columns,iqGrid);
            }
            g=pair.x;u=pair.y;
        }
        if (threadIdx.x%LaneWidth==0) {
            g=__bfloat162float(__float2bfloat16_rn(g));
            u=__bfloat162float(__float2bfloat16_rn(u));
            if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
            float h=(g/(1.0f+expf(-g)))*u;
            if(scores)h*=scores[route];
            if(row<inter) output[size_t(route)*inter+row]=__float2bfloat16_rn(h);
        }
    }
}

// Keep unbounded kernels for the formats whose existing register allocation
// already performs well. IQ2 uses explicit residency targets measured on SM80.
template<ggml_type Type, class Weights, int Tile, int LaneWidth, int Threads, int StaticColumns>
__global__ void V41GGUFGateUpAct(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    V41GGUFGateUpActBody<Type,Weights,Tile,LaneWidth,Threads,StaticColumns>(
        input,output,weights,indices,scores,experts,topk,columns,inter,rowBytes,limit);
}

template<ggml_type Type, class Weights, int Tile, int Threads, int StaticColumns, int MinBlocks>
__global__ __launch_bounds__(Threads,MinBlocks)
void V41GGUFGateUpActBound(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    V41GGUFGateUpActBody<Type,Weights,Tile,32,Threads,StaticColumns>(
        input,output,weights,indices,scores,experts,topk,columns,inter,rowBytes,limit);
}

__global__ void V41GGUFReduce(const __nv_bfloat16 *partial, float *output,
        const int32_t *indices, int hidden, int topk, const int32_t *inverse = nullptr) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    const int base = blockIdx.y*topk;
    int order[32];
    for (int k = 0; k < topk; ++k) {
        int pos = k;
        while (pos && indices[base+order[pos-1]] > indices[base+k]) {
            order[pos] = order[pos-1]; --pos;
        }
        order[pos] = k;
    }
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = inverse ? inverse[base+order[k]] : base+order[k];
        const float value = route >= 0 ? __bfloat162float(partial[size_t(route)*hidden+c]) : 0.f;
        sum = __fadd_rn(sum, value);
    }
    // The caller adds the shared expert in FP32 before the BF16 HC boundary.
    output[size_t(blockIdx.y)*hidden+c] = sum;
}

// Keep the reference path available for numeric/performance regression checks.
// Small matrices retain the existing tile choices.
bool V41UseTunedGGUF(int hidden, int inter) {
    return hidden >= 1024 && inter >= 1024;
}

template<class Weights>
void ProjectGateActivation(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int routes, int hidden, int inter, float limit) {
    const size_t shared = size_t(hidden/32)*sizeof(block_q8_1);
    switch(type) {
#define GATE_ACT_ARGS(Type) input,output,weights,indices,scores,experts,topk,hidden,inter,ggml_row_size(GGML_TYPE_##Type,hidden),limit
#define LAUNCH_GATE_ACT(Type, Static) \
        if constexpr (GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ2_XS || GGML_TYPE_##Type == GGML_TYPE_IQ2_S) { \
            constexpr int blocks = threads == 128 ? 8 : 4; \
            V41GGUFGateUpActBound<GGML_TYPE_##Type,Weights,tile,threads,Static,blocks><<<dim3((inter+tile-1)/tile,routes),threads,shared,cudaStreamPerThread>>>(GATE_ACT_ARGS(Type)); \
        } else { \
            V41GGUFGateUpAct<GGML_TYPE_##Type,Weights,tile,32,threads,Static><<<dim3((inter+tile-1)/tile,routes),threads,shared,cudaStreamPerThread>>>(GATE_ACT_ARGS(Type)); \
        }
#define GATE_ACT(Type) case GGML_TYPE_##Type: { \
        constexpr int threads = GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ2_XS || GGML_TYPE_##Type == GGML_TYPE_Q3_K ? 128 : 256; \
        constexpr int tile = GGML_TYPE_##Type == GGML_TYPE_IQ2_S || GGML_TYPE_##Type == GGML_TYPE_Q4_K ? 16 : 8; \
        if (hidden == 5120) { LAUNCH_GATE_ACT(Type, 5120); } else { LAUNCH_GATE_ACT(Type, 0); } break; }
        GATE_ACT(Q8_0) GATE_ACT(Q2_0) GATE_ACT(Q2_K) GATE_ACT(Q3_K) GATE_ACT(Q4_K) GATE_ACT(IQ1_S) GATE_ACT(IQ1_M)
        GATE_ACT(IQ2_XXS) GATE_ACT(IQ2_XS) GATE_ACT(IQ2_S) GATE_ACT(IQ3_XXS)
#undef GATE_ACT
#undef LAUNCH_GATE_ACT
#undef GATE_ACT_ARGS
    }
}

template<bool Gate, class Weights>
void Project(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int routes, int columns, int rows, bool tuned = false) {
    const size_t shared = size_t(columns/32)*sizeof(block_q8_1);
    if constexpr (!Gate) {
        if (tuned) {
            // A quarter warp keeps the 128 logical reference lanes, while a
            // fixed column count removes the short-loop bounds and tail work.
            if (columns == 2304) {
                switch (type) {
#define DOWN_2304(Type) case GGML_TYPE_##Type: \
                    V41GGUFProjection<GGML_TYPE_##Type,false,Weights,32,2304,16,8><<<dim3((rows+31)/32,routes),256,shared,cudaStreamPerThread>>>( \
                        input,output,weights,indices,experts,topk,columns,rows,ggml_row_size(GGML_TYPE_##Type,columns)); return;
                    DOWN_2304(IQ2_XXS) DOWN_2304(IQ2_XS) DOWN_2304(IQ2_S) DOWN_2304(IQ3_XXS)
#undef DOWN_2304
                }
            }
            switch(type) {
#define DOWN(Type) case GGML_TYPE_##Type: { \
                constexpr bool full = GGML_TYPE_##Type == GGML_TYPE_Q8_0 || GGML_TYPE_##Type == GGML_TYPE_Q2_0 || GGML_TYPE_##Type == GGML_TYPE_IQ3_XXS; \
                constexpr int lanes = full ? 32 : 16; \
                constexpr int tile = GGML_TYPE_##Type == GGML_TYPE_Q3_K || GGML_TYPE_##Type == GGML_TYPE_Q4_K ? 16 : 32; \
                V41GGUFProjection<GGML_TYPE_##Type, false, Weights, tile, 0, full ? 4 : 8, lanes><<<dim3((rows+tile-1)/tile,routes),256,shared,cudaStreamPerThread>>>( \
                    input,output,weights,indices,experts,topk,columns,rows,ggml_row_size(GGML_TYPE_##Type,columns)); return; }
                DOWN(Q8_0) DOWN(Q2_0) DOWN(Q2_K) DOWN(Q3_K) DOWN(Q4_K) DOWN(IQ1_S) DOWN(IQ1_M)
                DOWN(IQ2_XXS) DOWN(IQ2_XS) DOWN(IQ2_S) DOWN(IQ3_XXS)
#undef DOWN
            }
        }
    }
    if constexpr (Gate) {
        // Amortize the staged IQ1_S codebook over four rows per warp.
        if (type == GGML_TYPE_IQ1_S && rows >= 1024) {
            constexpr int tile = 32;
            V41GGUFProjection<GGML_TYPE_IQ1_S, Gate, Weights, tile>
                <<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>(
                    input, output, weights, indices, experts, topk, columns, rows,
                    ggml_row_size(GGML_TYPE_IQ1_S, columns));
            return;
        }
    }
    if constexpr (!Gate) {
        // The V4.1 down projection reuses each Q8 activation across 16 rows.
        // Keep the four virtual-warp sums in their original order.
        if (type == GGML_TYPE_Q3_K && columns == 2304 && rows >= 1024) {
            constexpr int tile = 16;
            V41GGUFProjection<GGML_TYPE_Q3_K, false, Weights, tile, 2304>
                <<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>(
                    input, output, weights, indices, experts, topk, columns, rows,
                    ggml_row_size(GGML_TYPE_Q3_K, columns));
            return;
        }
    }
    // The wider IQ tiles are tuned for resident weights, including EP.
    // Hybrid cache routes keep smaller tiles for low expert occupancy.
    constexpr bool resident = std::is_same_v<Weights, const uint8_t *const *> ||
        std::is_same_v<Weights, V41EpWeights>;
    switch (type) {
#define PROJECT(Type) case GGML_TYPE_##Type: { \
        constexpr int tile = resident && (GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ1_M) ? 32 : \
            (GGML_TYPE_##Type == GGML_TYPE_IQ1_S || GGML_TYPE_##Type == GGML_TYPE_IQ1_M) ? 16 : 8; \
        V41GGUFProjection<GGML_TYPE_##Type, Gate, Weights, tile><<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>( \
            input, output, weights, indices, experts, topk, columns, rows, \
            ggml_row_size(GGML_TYPE_##Type, columns)); break; }
        PROJECT(Q8_0) PROJECT(Q2_0) PROJECT(Q2_K) PROJECT(Q3_K) PROJECT(Q4_K) PROJECT(IQ1_S) PROJECT(IQ1_M)
        PROJECT(IQ2_XXS) PROJECT(IQ2_XS) PROJECT(IQ2_S) PROJECT(IQ3_XXS)
#undef PROJECT
    }
}
}

bool FastllmCudaRunV41GGUF(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
        fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
        const int32_t *indices, const float *scores, int gateType, int downType,
        int hidden, int inter, int experts, int topk, float swigluLimit, bool compactEpRoutes) {
    const int rows = input.dims[0], routes = rows*topk;
    auto *qInput = static_cast<block_q8_1 *>(workspace.cudaData);
    auto *qGate = qInput + size_t(rows)*(hidden/32);
    auto *partial = reinterpret_cast<__nv_bfloat16 *>(qGate+size_t(routes)*(inter/32));
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    const size_t baseBytes = (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*hidden*sizeof(__nv_bfloat16);
    const size_t offset = (baseBytes+255)&~size_t(255);
    void *groupedScratch = static_cast<char *>(workspace.cudaData)+offset;
    V41GGUFQuantize<true><<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, hidden);
    if (compactEpRoutes) {
        auto *routeMap = static_cast<int32_t *>(groupedScratch);
        V41EpRoutes<<<1, 32, 0, cudaStreamPerThread>>>(weights, indices, routeMap, routes, experts);
        const V41EpWeights view{weights, routeMap};
        Project<true>(gateType, qInput, static_cast<__nv_bfloat16 *>(gate.cudaData),
            view, indices, experts, topk, routes, hidden, 2*inter);
        // Fuse the two activation kernels but retain every BF16/FP8 rounding
        // boundary, including the score multiplication before FP8 quantization.
        V41EpSwigluQ8<<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(gate.cudaData), qGate, routeMap, scores, inter, swigluLimit);
        Project<false>(downType, qGate, partial, view, indices, experts, topk, routes, inter, hidden);
        V41GGUFReduce<<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
            partial, static_cast<float *>(output.cudaData), indices, hidden, topk, routeMap+1+routes);
        return cudaGetLastError() == cudaSuccess;
    }
    const bool grouped = workspace.GetBytes() > offset;
    const bool tuned = !grouped && V41UseTunedGGUF(hidden, inter);
    if (tuned) {
        ProjectGateActivation(gateType,qInput,static_cast<__nv_bfloat16 *>(gate.cudaData),
            weights,indices,scores,experts,topk,routes,hidden,inter,swigluLimit);
        V41GGUFQuantize<true><<<dim3((inter+255)/256,routes),256,0,cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(gate.cudaData),qGate,inter);
    } else {
        if (grouped) {
            if (!FastllmCudaV41GGUFGroupedProjection(true, gateType, qInput, gate.cudaData,
                weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
        } else Project<true>(gateType, qInput, static_cast<__nv_bfloat16 *>(gate.cudaData),
            weights, indices, experts, topk, routes, hidden, 2*inter);
        if (!FastllmCudaDeepSeekV41SharedSwigluQuantized(gate, swigluLimit, activation, scores)) return false;
        V41GGUFQuantize<false><<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, inter);
    }
    if (grouped) {
        if (!FastllmCudaV41GGUFGroupedProjection(false, downType, qGate, partial,
            weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
    } else Project<false>(downType, qGate, partial, weights, indices, experts, topk, routes, inter, hidden, tuned);
    V41GGUFReduce<<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        partial, static_cast<float *>(output.cudaData), indices, hidden, topk);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
size_t V41CacheWorkspaceBytes(int rows, int routes, int hidden, int inter) {
    const size_t bytes = (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*(size_t(inter)+hidden)*sizeof(__nv_bfloat16) + size_t(routes)*sizeof(float);
    return (bytes + 255) & ~size_t(255);
}
__global__ void V41CacheQuantize(const __nv_bfloat16 *input, block_q8_1 *output,
        int columns, size_t elements, const int32_t *slots = nullptr) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int row = i / columns;
    float x = slots && slots[row] < 0 ? 0.f : __bfloat162float(input[i]);
    x = V41Fp8Boundary(x);
    V41StoreQ8(x, output[i/32], i%32);
}
__global__ void V41CachePartials(const __nv_bfloat16 *partial, float *output,
        int hidden, size_t count, const int32_t *routeMap, const int32_t *validSlots, bool preserveMissing) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) {
        const int route = i / hidden;
        const bool missing = validSlots[route] < 0;
        if (missing && preserveMissing) return;
        const size_t destination = size_t(routeMap ? routeMap[route] : route)*hidden + i%hidden;
        output[destination] = missing ? 0.f : __bfloat162float(partial[i]);
    }
}
__global__ void V41CacheScores(const float *scores, float *selected,
        const int32_t *routeMap, int routes) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < routes) selected[r] = scores[routeMap[r]];
}
}

bool FastllmCudaV41GGUFSupported(int gateType, int downType) {
    const auto supported = [](int type) {
        return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q2_K ||
            type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
            type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
            type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
            type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS;
    };
    return supported(gateType) && supported(downType);
}
bool FastllmCudaV41GGUFNumaSupported(int gateType, int downType, int numaGateType, int numaDownType) {
    const auto matches = [](int canonical, int storage) {
        return canonical == storage ||
            (canonical == GGML_TYPE_IQ2_XXS && storage == GGML_TYPE_IQ2_XXS_R4) ||
            (canonical == GGML_TYPE_IQ2_XS && storage == GGML_TYPE_IQ2_XS_R4) ||
            (canonical == GGML_TYPE_IQ2_S && storage == GGML_TYPE_IQ2_S_R4);
    };
    return FastllmCudaV41GGUFSupported(gateType, downType) &&
        matches(gateType, numaGateType) && matches(downType, numaDownType);
}
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter, int rows, int topk) {
    if (hidden <= 0 || inter <= 0 || hidden%256 || inter%256 || rows < 1 || topk < 1 ||
        topk > 16 || rows > INT_MAX/topk) return 0;
    const size_t groupBytes = fastllm::moe_routes::Plan::WorkspaceBytes(rows*topk);
    return groupBytes ? V41CacheWorkspaceBytes(rows, rows*topk, hidden, inter) + groupBytes : 0;
}
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter) {
    return FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden, inter, 8, 16);
}
bool FastllmCudaV41GGUFPrepareQ8Input(const fastllm::Data &input, void *destination, size_t bytes) {
    if (!destination || input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::CUDA ||
        !input.cudaData || input.dims.size() != 2 || input.dims[0] <= 0 || input.dims[1] <= 0 ||
        input.dims[1] % 32 || bytes < size_t(input.dims[0]) * (input.dims[1] / 32) * sizeof(block_q8_1))
        return false;
    const size_t elements = size_t(input.dims[0]) * input.dims[1];
    V41CacheQuantize<<<(elements + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), static_cast<block_q8_1 *>(destination),
        input.dims[1], elements);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaMoeV41CanonicalGGUFCacheCompute(
        const fastllm::Data &input,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores,
        int topk, float swigluLimit, float *perExpert) {
    if (input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::CUDA ||
        !input.cudaData || input.dims.size() != 2 || input.dims[0] < 1 ||
        input.dims[1] != view.hidden || view.hidden <= 0 || view.inter <= 0 ||
        view.hidden%256 || view.inter%256 || topk < 1 || topk > 16 || input.dims[0] > INT_MAX/topk ||
        !FastllmCudaV41GGUFSupported(view.gateType, view.downType) ||
        (!view.records && !view.recordPointers) || !view.routeSlots || !view.workspace || !scores || !perExpert ||
        view.activeRouteCount || (view.routeMap && (view.routeCount < 1 || view.routeCount > input.dims[0]*topk)) ||
        !((view.numaGateType < 0 && view.numaDownType < 0) ||
          FastllmCudaV41GGUFNumaSupported(view.gateType, view.downType, view.numaGateType, view.numaDownType)))
        return false;
    const int rows = input.dims[0], routes = view.routeMap ? view.routeCount : rows*topk;
    const int hidden = view.hidden, inter = view.inter;
    const size_t bytes = V41CacheWorkspaceBytes(rows, routes, hidden, inter);
    const size_t scratch = fastllm::moe_routes::Plan::TemporaryBytes(routes);
    fastllm::moe_routes::Plan grouped(static_cast<uint8_t *>(view.workspace)+bytes, routes, scratch);
    if (!scratch || view.workspaceBytes < bytes+grouped.bytes) return false;
    auto *localInput = static_cast<block_q8_1 *>(view.workspace);
    const auto *qInput = view.preparedQ8Input ? static_cast<const block_q8_1 *>(view.preparedQ8Input) : localInput;
    auto *qGate = localInput + size_t(rows)*(hidden/32);
    auto *gateData = reinterpret_cast<__nv_bfloat16 *>(qGate + size_t(routes)*(inter/32));
    auto *partial = gateData + size_t(routes)*inter;
    if (!view.preparedQ8Input && !view.q8InputPrepared &&
        !FastllmCudaV41GGUFPrepareQ8Input(input, localInput, view.workspaceBytes)) return false;
    if (view.routeMap) {
        auto *selectedScores = reinterpret_cast<float *>(partial + size_t(routes)*hidden);
        V41CacheScores<<<(routes+255)/256, 256, 0, cudaStreamPerThread>>>(
            scores, selectedScores, view.routeMap, routes);
        scores = selectedScores;
    }
    // Group every compatible IQ2 projection across the complete route list.
    const bool batch = V41IQ2Type(view.gateType) && V41IQ2Type(view.downType);
    if (batch) {
        V41CacheGroupKeys<<<(routes+255)/256,256,0,cudaStreamPerThread>>>(view,grouped.keys,grouped.routes,routes);
        if (!grouped.Build(cudaStreamPerThread)) return false;
        V41IQ2BatchProject<true>(view.gateType,qInput,gateData,view,scores,topk,hidden,inter,swigluLimit,grouped);
        V41CacheQuantize<<<(size_t(routes)*inter+255)/256,256,0,cudaStreamPerThread>>>(
            gateData,qGate,inter,size_t(routes)*inter,view.routeSlots);
        V41IQ2BatchProject<false>(view.downType,qGate,partial,view,scores,topk,inter,hidden,swigluLimit,grouped);
    } else {
        // Bound only CUDA's physical grid.y. Route maps remain global and
        // output slices retain the original compact route ordering.
        const int step = 65535/topk*topk;
        for (int first = 0; first < routes; first += step) {
            const int count = std::min(step,routes-first);
            auto slice = view; slice.routeSlots += first;
            if (slice.routeMap) slice.routeMap += first;
            const auto *x = qInput + (view.routeMap ? 0 : size_t(first/topk)*(hidden/32));
            auto *g = gateData + size_t(first)*inter;
            auto *q = qGate + size_t(first)*(inter/32);
            ProjectGateActivation(view.gateType,x,g,slice,nullptr,scores+first,0,topk,count,hidden,inter,swigluLimit);
            V41CacheQuantize<<<(size_t(count)*inter+255)/256,256,0,cudaStreamPerThread>>>(
                g,q,inter,size_t(count)*inter,slice.routeSlots);
            Project<false>(view.downType,q,partial+size_t(first)*hidden,slice,nullptr,0,topk,count,inter,hidden,
                V41UseTunedGGUF(hidden,inter));
        }
    }
    V41CachePartials<<<(size_t(routes)*hidden+255)/256,256,0,cudaStreamPerThread>>>(
        partial,perExpert,hidden,size_t(routes)*hidden,view.routeMap,view.routeSlots,view.preserveMissing);
    return cudaGetLastError() == cudaSuccess;
}
