#pragma once
#include "devices/cuda/naive-n05-kv.cuh"

// TP shards with eight Q heads cannot fill the existing 16-head MMA tile.
// Tile the selected keys instead, sharing each translated/vector-loaded row
// across the heads while preserving the original FP32 reduction order.
namespace naive_prefill {
using BF16 = __nv_bfloat16;
constexpr int Heads = 8, ScoreHeads = 4, QkDim = 192, ValueDim = 128;

__device__ __forceinline__ int Key(const int *indices, int query, int slot,
                                  int count, int keys, int past, bool causal) {
    if (slot >= count) return -1;
    int key = indices ? indices[(size_t)query * count + slot] : slot;
    return key >= 0 && key < keys && (!causal || key <= past + query) ? key : -1;
}

__global__ void Scores(const BF16 *q, FastllmNaiveKV k, const int *indices,
        float *scores, int heads, int kvHeads, int stride, int keys,
        int count, int past, bool causal) {
    constexpr int Tile = 64;
    __shared__ __align__(16) BF16 ks[Tile][QkDim];
    __shared__ int valid[Tile];
    const int query = blockIdx.x, first = blockIdx.z * Tile;
    const int h0 = blockIdx.y * ScoreHeads, kvHead = h0 / (heads / kvHeads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int slot = warp; slot < Tile; slot += 4) {
        int key = Key(indices, query, first + slot, count, keys, past, causal);
        if (lane == 0) valid[slot] = key >= 0;
        if (lane < QkDim / 8)
            *reinterpret_cast<uint4*>(&ks[slot][lane * 8]) = key >= 0
                ? (k.Row(key, stride) + kvHead * QkDim + lane * 8).Load8()
                : make_uint4(0,0,0,0);
    }
    float queryValues[ScoreHeads][QkDim / 32];
    #pragma unroll
    for (int h = 0; h < ScoreHeads; ++h)
        #pragma unroll
        for (int j = 0; j < QkDim / 32; ++j)
            queryValues[h][j] = (float)q[((size_t)query * heads + h0 + h) * QkDim + lane + j * 32];
    __syncthreads();
    for (int slot = warp; slot < Tile; slot += 4) {
        float dot[ScoreHeads] = {};
        #pragma unroll
        for (int j = 0; j < QkDim / 32; ++j) {
            float value = (float)ks[slot][lane + j * 32];
            #pragma unroll
            for (int h = 0; h < ScoreHeads; ++h) dot[h] += queryValues[h][j] * value;
        }
        #pragma unroll
        for (int h = 0; h < ScoreHeads; ++h) {
            #pragma unroll
            for (int step = 16; step; step >>= 1)
                dot[h] += __shfl_down_sync(0xffffffff, dot[h], step);
            if (lane == 0 && first + slot < count) {
                float value = __bfloat162float(__float2bfloat16(dot[h]));
                value = __bfloat162float(__float2bfloat16(value * rsqrtf(float(QkDim))));
                scores[((size_t)query * heads + h0 + h) * count + first + slot] = valid[slot] ? value : -INFINITY;
            }
        }
    }
}

__global__ void Values(const float *prob, FastllmNaiveKV v, const int *indices,
        BF16 *out, int heads, int kvHeads, int keys, int count, int past, bool causal) {
    constexpr int Tile = 128;
    __shared__ __align__(16) BF16 vs[Tile][ValueDim];
    __shared__ int valid[Tile];
    const int query = blockIdx.x, h0 = blockIdx.y * Heads;
    const int kvHead = h0 / (heads / kvHeads), h = h0 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, d = lane * 4;
    float sum[4] = {};
    for (int first = 0; first < count; first += Tile) {
        for (int slot = threadIdx.x >> 5; slot < Tile; slot += 8) {
            int key = Key(indices, query, first + slot, count, keys, past, causal);
            if (lane == 0) valid[slot] = key >= 0;
            if (lane < ValueDim / 8)
                *reinterpret_cast<uint4*>(&vs[slot][lane * 8]) = key >= 0
                    ? (v.Row(key, kvHeads * ValueDim) + kvHead * ValueDim + lane * 8).Load8()
                    : make_uint4(0,0,0,0);
        }
        __syncthreads();
        #pragma unroll 1
        for (int slot = 0; slot < min(Tile, count - first); ++slot) if (valid[slot]) {
            float weight = prob[((size_t)query * heads + h) * count + first + slot];
            uint2 packed = *reinterpret_cast<const uint2*>(&vs[slot][d]);
            const BF16 *values = reinterpret_cast<const BF16*>(&packed);
            #pragma unroll
            for (int j = 0; j < 4; ++j) sum[j] += weight * (float)values[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        out[((size_t)query * heads + h) * ValueDim + d + j] = __float2bfloat16(sum[j]);
}
}
