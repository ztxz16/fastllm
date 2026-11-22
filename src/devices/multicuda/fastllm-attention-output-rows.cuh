#pragma once
// BF16 row blocks share O weights. Ordered GEMV remains the default; an
// explicitly allowed MMA path uses FP32 accumulation across up to eight rows.
// Keep the existing local-projection, rank-ordered reduction and residual
// rounding boundaries. The original row-major weights are never repacked.
#include "fastllm-swa-output.cuh"
#include "fastllm-naive-flex-moe.h"
#include "fastllm-norm-rows-router.cuh"
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include "fastllm-cuda-router-topk.cuh"
#include "../cuda/models/naive-n05-swa-mma.cuh"

namespace fastllm_swa_output {
constexpr int RowThreads = 512;

// Both fragments use the same K permutation. Share each original row-major
// weight load across eight query rows without a packed weight allocation.
__device__ __forceinline__ void ProjectMmaRows(const BF16 *input, const BF16 *weight,
        BF16 (&local)[MaxRows][32], int first, int rows) {
    constexpr int KWarps = RowThreads / 64;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    __shared__ __align__(128) unsigned char inputTile[8 * Inner * sizeof(BF16)];
    __shared__ float partial[2][KWarps][16][8];
    for (int i = t; i < 8 * (Inner / 8); i += RowThreads) {
        const int row = i / (Inner / 8), col = i % (Inner / 8) * 8;
        const int offset = fastllm_swa_mma::InputOffset(row, col);
        *reinterpret_cast<uint4 *>(inputTile + offset) = row < rows
            ? *reinterpret_cast<const uint4 *>(input + row * Inner + col) : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int group = warp / KWarps, part = warp % KWarps, g = lane / 4, quad = lane % 4;
    float c[4] = {};
#pragma unroll
    for (int col = part * 32; col < Inner; col += KWarps * 32) {
        const int k = col + quad * 8;
        const uint4 a = *reinterpret_cast<const uint4 *>(weight + size_t(first + group * 16 + g) * Inner + k);
        const uint4 b = *reinterpret_cast<const uint4 *>(weight + size_t(first + group * 16 + g + 8) * Inner + k);
        const int offset = fastllm_swa_mma::InputOffset(g, k);
        const uint4 x = *reinterpret_cast<const uint4 *>(inputTile + offset);
        fastllm_swa_mma::Step(c, a, b, x);
    }
    partial[group][part][g][quad * 2] = c[0];
    partial[group][part][g][quad * 2 + 1] = c[1];
    partial[group][part][g + 8][quad * 2] = c[2];
    partial[group][part][g + 8][quad * 2 + 1] = c[3];
    __syncthreads();
    for (int i = t; i < rows * 32; i += RowThreads) {
        const int row = i / 32, col = i % 32;
        float sum = 0;
#pragma unroll
        for (int p = 0; p < KWarps; ++p) sum += partial[col / 16][p][col % 16][row];
        local[row][col] = __float2bfloat16_rn(sum);
    }
}

template<int RowTile, bool Project = true, bool Mma = false>
static __global__ __launch_bounds__(RowThreads, 1) void OutputRowsResidual(
        const BF16 *input, const BF16 *weight, BF16 *projected, BF16 *hidden,
        Peers peers, uint64_t *epochs, int rank, int rows) {
    constexpr int Tile = 32;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int first = blockIdx.x * Tile;
    __shared__ BF16 local[MaxRows][Tile];
    __shared__ uint32_t received[MaxRows][Ranks][Tile / 2];
    __shared__ uint32_t step[MaxRows];
    if (t < rows) {
        // Advance only rows used by this invocation. A dormant row must not
        // reuse its old tag when another row's generation wraps.
        auto *generation = reinterpret_cast<uint32_t *>(epochs + t * MaxBlocks + blockIdx.x);
        step[t] = *generation;
        *generation = uint64_t(step[t]) + 1 == Period ? 0 : step[t] + 1;
    }
    if constexpr (Mma) {
        ProjectMmaRows(input, weight, local, first, rows);
    } else if constexpr (Project) {
        for (int column = warp; column < Tile; column += RowThreads / 32) {
            float value[RowTile][8] = {}, correction[RowTile][8] = {};
#pragma unroll
            for (int part = 0; part < 8; ++part) {
                const int col = (lane + part * 32) * 8;
                if (col < Inner) {
                    fastllm::cuda::Bf16OrderedDotPacket w;
                    w.bits = *reinterpret_cast<const uint4 *>(weight + size_t(first + column) * Inner + col);
#pragma unroll
                    for (int row = 0; row < RowTile; ++row) if (row < rows) {
                        fastllm::cuda::Bf16OrderedDotPacket x;
                        x.bits = *reinterpret_cast<const uint4 *>(input + row * Inner + col);
                        float sum = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum += float(x.values[i]) * float(w.values[i]);
                        value[row][part] += sum;
                    }
                }
            }
            // Retain the compensated 256-lane GEMV tree exactly, while sharing
            // each weight load across rows. MMA changes this rounding boundary.
#pragma unroll
            for (int stride = 4; stride; stride >>= 1)
#pragma unroll
                for (int part = 0; part < stride; ++part)
#pragma unroll
                    for (int row = 0; row < RowTile; ++row) {
                        float other = value[row][part + stride] - correction[row][part];
                        float sum = value[row][part] + other;
                        correction[row][part] = (sum - value[row][part]) - other;
                        value[row][part] = sum;
                    }
#pragma unroll
            for (int row = 0; row < RowTile; ++row) {
                float sum = value[row][0], diff = correction[row][0];
#pragma unroll
                for (int stride = 16; stride; stride >>= 1) {
                    float peer = __shfl_down_sync(0xffffffff, sum, stride);
                    if (lane < stride) {
                        float other = peer - diff, next = sum + other;
                        diff = (next - sum) - other; sum = next;
                    }
                }
                if (lane == 0 && row < rows) local[row][column] = __float2bfloat16_rn(sum);
            }
        }
    } else if (t < rows * Tile) {
        const int row = t / Tile, column = t % Tile;
        local[row][column] = projected[row * Width + first + column];
    }
    __syncthreads();
    // FrameBytes reserves all eight rows even for a one-row caller. This
    // fixed pitch and per-row generations isolate old/new row counts.
    for (int i = t; i < rows * Ranks * (Tile / 2); i += RowThreads) {
        const int row = i / (Ranks * (Tile / 2)), peer = i / (Tile / 2) % Ranks;
        const int column = i % (Tile / 2), word = first / 2 + column;
        const size_t frame = (step[row] % Frames) * FrameBytes;
        const uint32_t tag = step[row] + 1;
        const uint32_t bits = reinterpret_cast<const uint32_t *>(local[row])[column];
        Store(peers.data[peer] + frame + row * Ranks * SlotBytes + rank * SlotBytes + word * 8,
            (uint64_t(bits) << 32) | tag);
    }
    __syncthreads();
    for (int i = t; i < rows * Ranks * (Tile / 2); i += RowThreads) {
        const int row = i / (Ranks * (Tile / 2)), peer = i / (Tile / 2) % Ranks;
        const int column = i % (Tile / 2), word = first / 2 + column;
        const size_t frame = (step[row] % Frames) * FrameBytes;
        const uint32_t tag = step[row] + 1;
        uint64_t value;
        do {
            value = Load(peers.data[rank] + frame + row * Ranks * SlotBytes + peer * SlotBytes + word * 8);
        } while (uint32_t(value) != tag);
        received[row][peer][column] = uint32_t(value >> 32);
    }
    __syncthreads();
    if (t < rows * (Tile / 2)) {
        const int row = t / (Tile / 2), column = t % (Tile / 2);
        float a = 0, b = 0;
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            const uint32_t value = received[row][peer][column];
            const float x = float(__ushort_as_bfloat16(value & 65535));
            const float y = float(__ushort_as_bfloat16(value >> 16));
            if (peer == 0) { a = x; b = y; } else { a += x; b += y; }
        }
        const int index = row * Width + first + column * 2;
        const BF16 x = __float2bfloat16_rn(a), y = __float2bfloat16_rn(b);
        projected[index] = x; projected[index + 1] = y;
        hidden[index] = __float2bfloat16_rn(float(hidden[index]) + float(x));
        hidden[index + 1] = __float2bfloat16_rn(float(hidden[index + 1]) + float(y));
    }
}

inline const void *OutputRowsKernel(int rows, bool mma = false) {
    if (mma) return (const void *)OutputRowsResidual<8, true, true>;
    return rows <= 2 ? (const void *)OutputRowsResidual<2>
        : rows <= 4 ? (const void *)OutputRowsResidual<4>
        : rows <= 7 ? (const void *)OutputRowsResidual<8>
        : (const void *)OutputRowsResidual<8, false>;
}

static __global__ void NormRows(const BF16 *input, const float *weight,
        BF16 *output, float eps) {
    const int t = threadIdx.x, row = blockIdx.x;
    __shared__ float sums[8];
    float value[16], affine[16], sum = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int column = t + j * 256;
        value[j] = float(input[row * Width + column]);
        affine[j] = weight[column];
        sum += value[j] * value[j];
    }
    const float inv = rsqrtf(fastllm::cuda::WarpFirstBlockSum256(sum, sums) / Width + eps);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        output[row * Width + t + j * 256] =
            __float2bfloat16_rn(float(__float2bfloat16_rn(value[j] * inv)) * affine[j]);
}
// Match both the standalone norm and router's column order. Two rows share
// weights without changing their per-row sum trees or BF16 materializations.
template<int RowTile = 2>
static __global__ void NormRowsRouter(const BF16 *input, const float *weight,
        float *output, const float *normWeight, BF16 *normed, float eps, int rows) {
    const int t = threadIdx.x, first = blockIdx.y * RowTile, expert = blockIdx.x;
    __shared__ float sums[RowTile][8];
    float values[RowTile][16], affine[16], weights[16], partial[RowTile]{};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = t + i * 256;
        weights[i] = weight[size_t(expert) * Width + col];
        affine[i] = normWeight[col];
#pragma unroll
        for (int row = 0; row < RowTile; ++row)
            values[row][i] = first + row < rows ? float(input[(first + row) * Width + col]) : 0;
    }
#pragma unroll
    for (int row = 0; row < RowTile; ++row) {
        float square = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) square += values[row][i] * values[row][i];
        const float inv = rsqrtf(fastllm::cuda::WarpFirstBlockSum256(square, sums[row]) / Width + eps);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[row][i] = float(__float2bfloat16_rn(float(__float2bfloat16_rn(values[row][i] * inv)) * affine[i]));
            if (expert == 0 && first + row < rows)
                normed[(first + row) * Width + t + i * 256] = __float2bfloat16_rn(values[row][i]);
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int row = 0; row < RowTile; ++row) partial[row] += values[row][i] * weights[i];
    __syncthreads(); // Every warp has consumed the norm sums before reuse.
#pragma unroll
    for (int row = 0; row < RowTile; ++row) {
#pragma unroll
        for (int stride = 1; stride < 32; stride *= 2)
            partial[row] += __shfl_down_sync(0xffffffff, partial[row], stride);
        if (t % 32 == 0) sums[row][t / 32] = partial[row];
    }
    __syncthreads();
    const int row = t / 32, lane = t % 32;
    if (row < RowTile) {
        float value = lane < 8 ? sums[row][lane] : 0;
#pragma unroll
        for (int stride = 1; stride < 8; stride *= 2)
            value += __shfl_down_sync(0xffffffff, value, stride);
        if (lane == 0 && first + row < rows)
            output[(first + row) * 256 + expert] = 1.0 / (1.0 + expf(-(value + 0.f)));
    }
}
using fastllm_norm_rows::StoreGateBlock128;
using fastllm_norm_rows::ClearGateBlock128Tail;

template<int Experts = 4, bool Prepare = false>
static __device__ __forceinline__ void NormRowsRouterExpertsBody(const BF16 *input,const float *weight,float *output,
        const float *gamma,BF16 *normed,float eps,int rows, FastllmMoePreparedInput prepared = {}) {
    constexpr int Rows = 2;
    const int t=threadIdx.x,first=blockIdx.y*Rows,expert=blockIdx.x*Experts;
    if constexpr(Prepare) if(blockIdx.x==0 && blockIdx.y==0){
        prepared.counters[t]=0;
        if(t==0)++*prepared.epoch;
        ClearGateBlock128Tail<Rows>(prepared.block128,rows);
    }
    float normalized[Rows][16];
    __shared__ float sums[Experts*Rows][8];
    float affine[16];
    #pragma unroll
    for(int i=0;i<16;++i)affine[i]=gamma[t+i*256];
    #pragma unroll
    for(int row=0;row<Rows;++row){
        float x[16],square=0;
        #pragma unroll
        for(int i=0;i<16;++i){
            x[i]=first+row<rows?float(input[(first+row)*Width+t+i*256]):0;
            square+=x[i]*x[i];
        }
        float inv=rsqrtf(fastllm::cuda::WarpFirstBlockSum256(square,sums[row])/Width+eps);
        #pragma unroll
        for(int i=0;i<16;++i){
            BF16 v=__float2bfloat16_rn(float(__float2bfloat16_rn(x[i]*inv))*affine[i]);
            normalized[row][i]=float(v);
            if(expert==0 && first+row<rows)normed[(first+row)*Width+t+i*256]=v;
        }
    }
    if constexpr(Prepare) if(blockIdx.x<16){
        const int i=blockIdx.x;
        #pragma unroll
        for(int row=0;row<Rows;++row){
            float v=normalized[row][i],mx=fabsf(v);
            for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
            unsigned e=max((__float_as_uint(mx)+0x7fffffu-0x600000u)>>23,113u)-8;
            const int col=i*256+t;
            size_t offset=size_t(first+row)*Width+col;
            __nv_fp8_e4m3 q(v*__uint_as_float((254-e)<<23));
            if(first+row<rows){
                prepared.values[offset]=q.__x;
                if(t%32==0)prepared.scales[offset/32]=e;
            }
            if(prepared.block128){
                if(t%32==0)sums[row][t/32]=float(e);
                __syncthreads();
                unsigned coarse=0;
                for(int j=0;j<4;++j)coarse=max(coarse,unsigned(sums[row][t/128*4+j]));
                const float ratio=__uint_as_float(unsigned(max(0,127+int(e)-int(coarse)))<<23);
                prepared.block128[(first+row)*4112+col]=__nv_fp8_e4m3(float(q)*ratio).__x;
                if(t%128==0)reinterpret_cast<float*>(prepared.block128+32896)[col/128*8+first+row]=__uint_as_float(coarse<<23);
                if(col<16)prepared.block128[(first+row)*4112+4096+col]=0;
            }
        }
    }
    __syncthreads();
    float partial[Experts][Rows]{};
    #pragma unroll
    for(int i=0;i<16;++i){
        float w[Experts];
        #pragma unroll
        for(int e=0;e<Experts;++e)w[e]=weight[size_t(expert+e)*Width+t+i*256];
        #pragma unroll
        for(int row=0;row<Rows;++row){
            float x=normalized[row][i];
            #pragma unroll
            for(int e=0;e<Experts;++e)partial[e][row]+=x*w[e];
        }
    }
    #pragma unroll
    for(int e=0;e<Experts;++e)
    #pragma unroll
    for(int row=0;row<Rows;++row){
        #pragma unroll
        for(int stride=1;stride<32;stride*=2)partial[e][row]+=__shfl_down_sync(0xffffffff,partial[e][row],stride);
        if(t%32==0)sums[e*Rows+row][t/32]=partial[e][row];
    }
    __syncthreads();
    for(int task=t/8;task<((Experts*Rows+3)/4)*4;task+=32){
        int row=task%Rows,e=task/Rows,lane=t%8;
        float value=task<Experts*Rows?sums[task][lane]:0;
        #pragma unroll
        for(int stride=1;stride<8;stride*=2)value+=__shfl_down_sync(0xffffffff,value,stride,8);
        if(task<Experts*Rows&&lane==0&&first+row<rows)output[(first+row)*256+expert+e]=1.0/(1.0+expf(-(value+0.f)));
    }
}

template<int Experts = 4, bool Prepare = false>
static __global__ void NormRowsRouterExperts(const BF16 *input,const float *weight,float *output,
        const float *gamma,BF16 *normed,float eps,int rows,FastllmMoePreparedInput prepared = {}) {
    NormRowsRouterExpertsBody<Experts,Prepare>(input,weight,output,gamma,normed,eps,rows,prepared);
}
// All CTA outputs are published by grid.sync before one CTA selects routes.
// The launch must be admitted for complete grid residency before capture.
static __global__ void NormRowsRouterRoutes(const BF16 *input,const float *weight,float *output,
        const float *gamma,BF16 *normed,float eps,int rows,FastllmMoePreparedInput prepared) {
    NormRowsRouterExpertsBody<4,true>(input,weight,output,gamma,normed,eps,rows,prepared);
    cooperative_groups::this_grid().sync();
    if(blockIdx.x || blockIdx.y)return;
    const int t=threadIdx.x,lane=t%32,row=t/32;
    __shared__ __align__(16) float probabilities[2048],bias[256];
    __shared__ fastllm::cuda::RouterTopKScratch<8> scratch[8];
    __shared__ int ids[64],totals[2];
    __shared__ float scores[64];
    for(int i=t;i<rows*256;i+=256)probabilities[i]=output[i];
    bias[t]=prepared.bias?prepared.bias[t]:0.f;
    __syncthreads();
    if(row<rows)fastllm::cuda::SelectRouterTopK<256,8>(probabilities+row*256,bias,
        ids+row*8,scores+row*8,prepared.bias!=nullptr,prepared.needNorm,prepared.routeScale,scratch[row]);
    __syncthreads();
    if(t<rows*8){prepared.indices[t]=ids[t];prepared.scores[t]=scores[t];prepared.jobs[t]=t;}
    if(t==0)*prepared.count=rows*8;
    // Selection has finished reading the cached probabilities on every warp.
    auto *masks=reinterpret_cast<unsigned long long*>(probabilities);
    masks[t]=0;__syncthreads();
    int expert=t<rows*8?ids[t]:0;
    if(t<rows*8)atomicOr(masks+expert,1ull<<t);
    __syncthreads();
    unsigned long long mask=t<rows*8?masks[expert]:0;
    int first=t<rows*8&&__ffsll(mask)-1==t,prefix=first;
    if(t<64){
        for(int d=1;d<32;d*=2){int v=__shfl_up_sync(0xffffffff,prefix,d);if(lane>=d)prefix+=v;}
        if(lane==31)totals[t/32]=prefix;
    }
    __syncthreads();
    if(t==0)*prepared.unique=totals[0]+totals[1];
    if(first){int item=prefix-1+(t>=32?totals[0]:0),count=0;
        prepared.experts[item]=expert;
        while(mask){prepared.members[item*8+count++]=__ffsll(mask)-1;mask&=mask-1;}
        prepared.rowCounts[item]=count;
    }
}

// Vectorize the BF16 normalization/FP8 preparation, then transpose through
// CTA-local BF16 storage into the original ordered FP32 router dot. The
// staging adds no rounding: the normalization already materializes BF16.
// Reuse each router weight across two rows and preserve all reference sums.
template<int Experts>
static __global__ void NormRowsRouterPreparedVector(
        const BF16 *input, const float *weight, float *output,
        const float *gamma, BF16 *normed, float eps, int rows,
        FastllmMoePreparedInput prepared) {
    constexpr int Rows = 2, Chunks = 2, Warps = 8;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int expert = blockIdx.x * Experts, first = blockIdx.y * Rows;
    __shared__ float normSums[Rows][Warps], dotSums[Experts * Rows][Warps];
    __shared__ __align__(16) BF16 tile[Rows][Width];
    float4 affine[Chunks * 2];
#pragma unroll
    for (int i = 0; i < Chunks * 2; ++i) {
        const int col = t * 8 + (i / 2) * 2048 + (i % 2) * 4;
        affine[i] = *reinterpret_cast<const float4 *>(gamma + col);
    }
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        prepared.counters[t] = 0;
        if (t == 0) ++*prepared.epoch;
        ClearGateBlock128Tail<Rows>(prepared.block128,rows);
    }
    uint4 raw[Rows][Chunks];
    float inv[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        float square = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = first + row < rows ? float(input[(first + row) * Width + t + i * 256]) : 0.f;
            square += v * v;
        }
        inv[row] = rsqrtf(fastllm::cuda::WarpFirstBlockSum256(square, normSums[row]) / Width + eps);
#pragma unroll
        for (int c = 0; c < Chunks; ++c)
            raw[row][c] = first + row < rows
                ? *reinterpret_cast<const uint4 *>(input + (first + row) * Width + t * 8 + c * 2048)
                : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float v[8], mx = 0.f;
            uint4 rounded;
            auto *dest = reinterpret_cast<__nv_bfloat162 *>(&rounded);
            const auto *src = reinterpret_cast<const __nv_bfloat162 *>(&raw[row][c]);
            const float *gm = reinterpret_cast<const float *>(&affine[c * 2]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 x = __bfloat1622float2(src[j]);
                const float2 n = __bfloat1622float2(__floats2bfloat162_rn(x.x * inv[row], x.y * inv[row]));
                dest[j] = __floats2bfloat162_rn(n.x * gm[j * 2], n.y * gm[j * 2 + 1]);
                const float2 z = __bfloat1622float2(dest[j]);
                v[j * 2] = z.x; v[j * 2 + 1] = z.y;
                mx = fmaxf(mx, fmaxf(fabsf(z.x), fabsf(z.y)));
            }
            const int col = t * 8 + c * 2048;
            *reinterpret_cast<uint4 *>(tile[row] + col) = rounded;
            if (expert == 0 && first + row < rows)
                *reinterpret_cast<uint4 *>(normed + (first + row) * Width + col) = rounded;
            if (blockIdx.x == warp + c * Warps) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
                const unsigned e = max((__float_as_uint(mx) + 0x7fffffu - 0x600000u) >> 23, 113u) - 8;
                const float scale = __uint_as_float((254 - e) << 23);
                uint2 q{};
                auto *halves = reinterpret_cast<unsigned short *>(&q);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    halves[j] = __nv_cvt_float2_to_fp8x2(make_float2(v[j * 2] * scale, v[j * 2 + 1] * scale),
                        __NV_SATFINITE, __NV_E4M3);
                const size_t offset = size_t(first + row) * Width + col;
                if (first + row < rows) {
                    *reinterpret_cast<uint2 *>(prepared.values + offset) = q;
                    if (t % 4 == 0) prepared.scales[offset / 32] = e;
                }
                if (prepared.block128) StoreGateBlock128(prepared.block128, first + row, col, q, e);
            }
        }
    }
    __syncthreads();
    float partial[Experts][Rows]{};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float w[Experts];
#pragma unroll
        for (int e = 0; e < Experts; ++e)
            w[e] = weight[size_t(expert + e) * Width + t + i * 256];
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const float x = float(tile[row][t + i * 256]);
#pragma unroll
            for (int e = 0; e < Experts; ++e) partial[e][row] += x * w[e];
        }
    }
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
#pragma unroll
            for (int d = 1; d < 32; d *= 2)
                partial[e][row] += __shfl_down_sync(0xffffffff, partial[e][row], d);
            if (lane == 0) dotSums[e * Rows + row][warp] = partial[e][row];
        }
    }
    __syncthreads();
    for (int task = t / 8; task < Experts * Rows; task += 32) {
        float value = dotSums[task][t % 8];
#pragma unroll
        for (int d = 1; d < 8; d *= 2) value += __shfl_down_sync(0xffffffff, value, d, 8);
        if (t % 8 == 0 && first + task % Rows < rows)
            output[(first + task % Rows) * 256 + expert + task / Rows] = 1.0 / (1.0 + expf(-(value + 0.f)));
    }
}

using fastllm_norm_rows::NormRowsRouterDirect;

} // namespace fastllm_swa_output
