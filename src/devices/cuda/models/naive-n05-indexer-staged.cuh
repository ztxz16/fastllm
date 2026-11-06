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
template<int Queries = 1>
__global__ __launch_bounds__(128, Queries == 4 ? 6 : Queries == 2 ? 8 : 10)
void HopperScoresSharedHead(const BF16 *weights,Batch batch) {
    static_assert(Queries == 1 || Queries == 2 || Queries == 4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && CUDART_VERSION >= 12040
    using namespace naive_dsa_mma;
    constexpr int Threads=128,Padding=16,Step=32;
    Row r=batch.rows[blockIdx.y*Queries];r.key.Bind(r.cache);
    const int begin=r.Begin(),tileStride=int(gridDim.x)*kKeys;
    int ends[Queries],n=begin;
#pragma unroll
    for(int q=0;q<Queries;++q) {
        ends[q]=begin+batch.rows[blockIdx.y*Queries+q].Count();
        n=max(n,ends[q]);
    }
    int first=begin+int(blockIdx.x)*kKeys;
    if(first>=n)return;
    __shared__ __align__(16) unsigned char qs[Queries][kHeads][kIndexerDim+Padding];
    __shared__ __align__(16) unsigned char buffers[2][kKeys][kIndexerDim+Padding];
    __shared__ float scales[Queries][kHeads],weightsF[Queries][kHeads];
    const int t=threadIdx.x,lane=t&31,warp=t>>5;
#pragma unroll
    for(int q=0;q<Queries;++q) {
        const auto &row=batch.rows[blockIdx.y*Queries+q];
        const bool live=ends[q]>begin;
        for(int i=t;i<kHeads*kIndexerDim;i+=Threads)
            qs[q][i/kIndexerDim][i%kIndexerDim]=live?
                __nv_cvt_float_to_fp8((float)row.query[i],__NV_SATFINITE,__NV_E4M3):0;
        if(t<kHeads){scales[q][t]=live?row.scale[t]:0.f;weightsF[q][t]=(float)weights[row.sequence*kHeads+t];}
    }
    LoadStagedKey(r,first,n,t,&buffers[0][0][0]);
    int stage=0;
    for(;first<n;first+=tileStride,stage^=1) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // The alternate buffer was released by the prior iteration's final
        // block barrier. Issue its next load before computing this tile.
        if(first+tileStride<n)LoadStagedKey(r,first+tileStride,n,t,&buffers[stage^1][0][0]);
        auto &ks=buffers[stage];
        float dots[Queries][2][4] = {};
#pragma unroll
        for (int d = 0; d < kIndexerDim; d += Step) {
            uint32_t a[4];
            Load4(a, &ks[warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][d + (lane >> 4) * (Step / 2)]);
#pragma unroll
            for (int q = 0; q < Queries; ++q) {
#pragma unroll
                for (int group = 0; group < 2; ++group) {
                    uint32_t b[2];
                    Load2(b, &qs[q][group * 8 + (lane & 7)][d + ((lane >> 3) & 1) * (Step / 2)]);
                    MmaFp8(dots[q][group], a, b);
                }
            }
        }
        // Preserve a scaled-cache key scale before optionally reusing K.
        const float keyScale = t < kKeys && r.key.scaledIndexer
            ? *(const float *)&ks[t][128] : 1.f;
        __syncthreads();
        float *values = reinterpret_cast<float *>(&ks[0][0]);
#pragma unroll
        for (int q = 0; q < Queries; ++q) {
            auto &dot = dots[q];
#pragma unroll
            for (int group = 0; group < 2; ++group)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    dot[group][i] = fmaxf(dot[group][i] * scales[q][group * 8 + (lane & 3) * 2 + i % 2], 0.f);
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
            if (t < kKeys && first + t < ends[q]) {
                float sum = 0.f;
#pragma unroll
                for (int head = 0; head < kHeads; ++head)
                    sum = fmaf(values[SharedHeadOffset(t, head)], weightsF[q][head], sum);
                batch.rows[blockIdx.y*Queries+q].scores[first + t - begin] = naive_topk::OrderedScoreBits(sum * keyScale);
            }
            // Release the reduction scratch before the next query or K tile.
            __syncthreads();
        }
    }
#endif
}

}
