#pragma once
namespace naive_indexer_batch {

// Native FP8 keys only: copy the key scale in the same async group as K.
// The caller retains the original path for BF16-backed Indexer tails.
__device__ __forceinline__ void LoadStagedKey(const Row &r,int first,int n,int t,
                                             unsigned char *destination) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_mma;
    constexpr int Pitch=kIndexerDim+16;
    const bool onePage=!r.key.pages || r.key.pageLen%kKeys==0;
    const auto tile=onePage?r.key.IndexRow(first,r.stride):FastllmNaiveKVRow{};
    for(int i=t;i<kKeys*8;i+=128) {
        const int key=i/8,d=i%8*16;
        const auto base=onePage?FastllmNaiveKVRow((const char*)tile.data+(size_t)key*r.stride,true)
            :r.key.IndexRow(min(first+key,n-1),r.stride);
        LoadScoreVector<false>(destination+key*Pitch+d,(const unsigned char*)base.data+d,first+key<n);
    }
    if(t<kKeys && r.key.scaledIndexer) {
        const auto base=onePage?FastllmNaiveKVRow((const char*)tile.data+(size_t)t*r.stride,true)
            :r.key.IndexRow(min(first+t,n-1),r.stride);
        LoadScoreVector<false>(destination+t*Pitch+128,(const unsigned char*)base.data+128,first+t<n);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
// Reuse the released K tile with a 17-float pitch. Every key retains
// its original ordered sixteen-head FP32 FMA sequence.
__device__ __forceinline__ int SharedHeadOffset(int row, int head) { return row * 17 + head; }
__global__ __launch_bounds__(128, 10) void HopperScoresSharedHead(const BF16 *weights,Batch batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && CUDART_VERSION >= 12040
    using namespace naive_dsa_mma;
    constexpr int Threads=128,Padding=16,Step=32;
    Row r=batch.rows[blockIdx.y];r.key.Bind(r.cache);
    const int begin=r.Begin(),n=begin+r.Count(),tileStride=int(gridDim.x)*kKeys;
    int first=begin+int(blockIdx.x)*kKeys;
    if(first>=n)return;
    __shared__ __align__(16) unsigned char qs[kHeads][kIndexerDim+Padding];
    __shared__ __align__(16) unsigned char buffers[2][kKeys][kIndexerDim+Padding];
    __shared__ float scales[kHeads],weightsF[kHeads];
    const int t=threadIdx.x,lane=t&31,warp=t>>5;
    for(int i=t;i<kHeads*kIndexerDim;i+=Threads)
        qs[i/kIndexerDim][i%kIndexerDim]=__nv_cvt_float_to_fp8((float)r.query[i],__NV_SATFINITE,__NV_E4M3);
    if(t<kHeads){scales[t]=r.scale[t];weightsF[t]=(float)weights[r.sequence*kHeads+t];}
    LoadStagedKey(r,first,n,t,&buffers[0][0][0]);
    int stage=0;
    for(;first<n;first+=tileStride,stage^=1) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // The alternate buffer was released by the prior iteration's final
        // block barrier. Issue its next load before computing this tile.
        if(first+tileStride<n)LoadStagedKey(r,first+tileStride,n,t,&buffers[stage^1][0][0]);
        auto &ks=buffers[stage];
        float dot[2][4] = {};
#pragma unroll
        for (int d = 0; d < kIndexerDim; d += Step) {
            uint32_t a[4];
            Load4(a, &ks[warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][d + (lane >> 4) * (Step / 2)]);
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                uint32_t b[2];
                Load2(b, &qs[group * 8 + (lane & 7)][d + ((lane >> 3) & 1) * (Step / 2)]);
                MmaFp8(dot[group], a, b);
            }
        }
        // Each lane scales only its own head values, before exchanging them.
#pragma unroll
        for (int group = 0; group < 2; ++group)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                dot[group][i] = fmaxf(dot[group][i] * scales[group * 8 + (lane & 3) * 2 + i % 2], 0.f);
        // Preserve a scaled-cache key scale before optionally reusing K.
        const float keyScale = t < kKeys && r.key.scaledIndexer
            ? *(const float *)&ks[t][128] : 1.f;
        __syncthreads();
        float *values = reinterpret_cast<float *>(&ks[0][0]);
#pragma unroll
        for (int group = 0; group < 2; ++group)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = warp * 16 + (lane >> 2) + (i / 2) * 8;
                const int head = group * 8 + (lane & 3) * 2 + i % 2;
                values[SharedHeadOffset(row, head)] = dot[group][i];
            }
        __syncthreads();
        // One thread owns one key. The sixteen fmaf operations retain the
        // original head order and every FP32 rounding; there is no tree sum.
        if (t < kKeys && first + t < n) {
            float sum = 0.f;
#pragma unroll
            for (int head = 0; head < kHeads; ++head)
                sum = fmaf(values[SharedHeadOffset(t, head)], weightsF[head], sum);
            r.scores[first + t - begin] = naive_topk::OrderedScoreBits(sum * keyScale);
        }

        // Release current K before the next iteration can overwrite it.
        __syncthreads();
    }
#endif
}

}
