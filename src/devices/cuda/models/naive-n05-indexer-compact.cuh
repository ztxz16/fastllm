#pragma once
namespace naive_indexer_compact {
using namespace naive_indexer_batch;
// Score-only fields; selection/merge metadata stays in the original Row.
// Shared page geometry and CP parameters are represented once per launch.
struct CompactRow {
    const BF16* base;
    const int* pages;
    const void* const* cache;
    size_t offset;
    const int* live;
    BF16* query;
    float* scale;
    unsigned* scores;
    int sequence;
};
constexpr int CompactLimit=384;
struct CompactBatch {
    CompactRow rows[CompactLimit];
    int stride,pageLen,pageShift,cpRank,cpRanks,contextPageLen;
    bool scaledIndexer;
};
static_assert(sizeof(CompactRow)==72);
static_assert(sizeof(CompactBatch)+sizeof(void*)+128<=32764);
__device__ __forceinline__ Row Expand(const CompactRow& x,const CompactBatch& b) {
    Row r{};
    r.key.base=x.base;r.key.pages=x.pages;r.key.offset=x.offset;
    r.key.pageLen=b.pageLen;r.key.pageShift=b.pageShift;
    r.key.fp8=true;r.key.scaledIndexer=b.scaledIndexer;
    r.cache=x.cache;r.live=x.live;r.query=x.query;r.scale=x.scale;r.scores=x.scores;
    r.sequence=x.sequence;r.stride=b.stride;
    r.cpRank=b.cpRank;r.cpRanks=b.cpRanks;r.contextPageLen=b.contextPageLen;
    return r;
}
__global__ void QuantizeCompact(const BF16* query,const __grid_constant__ CompactBatch batch) {
    const auto& row=batch.rows[blockIdx.y];
    if (*row.live<=0)return;
    __shared__ float maximum[128];
    const int d=threadIdx.x,head=blockIdx.x;
    float x=(float)query[(size_t)row.sequence*2048+head*128+d];
    maximum[d]=fabsf(x);
    __syncthreads();
    for(int step=64;step;step>>=1) {
        if(d<step)maximum[d]=fmaxf(maximum[d],maximum[d+step]);
        __syncthreads();
    }
    float scale=fmaxf(maximum[0],1e-4f)/448.0f;
    // Same quantization and scale. E4M3 values are exactly representable in
    // BF16; retain their bits instead of round-tripping through BF16 storage.
    __nv_fp8_e4m3 value(fmaxf(-448.0f,fminf(448.0f,x/scale)));
    reinterpret_cast<unsigned char*>(row.query)[head*128+d]=value.__x;
    if(d==0)row.scale[head]=scale;
}
__device__ __forceinline__ void LoadPairedPages(const Row& r,int first,int n,int t,unsigned char* destination) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_mma;
    if(!r.key.pages || r.key.pageLen!=32 || (first&31)) {
        LoadStagedKey(r,first,n,t,destination);
        return;
    }
    constexpr int Pitch=kIndexerDim+16;
#pragma unroll
    for(int page=0;page<2;++page) {
        const int firstKey=first+page*32;
        const auto* base=static_cast<const unsigned char*>(
            r.key.IndexRow(min(firstKey,n-1),r.stride).data);
#pragma unroll
        for(int half=0;half<2;++half) {
            const int key=t/8+half*16,column=(t%8)*16;
            LoadScoreVector<false>(destination+(page*32+key)*Pitch+column,
                base+size_t(key)*r.stride+column,firstKey+key<n);
        }
        if(t<32 && r.key.scaledIndexer)
            LoadScoreVector<false>(destination+(page*32+t)*Pitch+128,
                base+size_t(t)*r.stride+128,firstKey+t<n);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template<bool AsyncQuery, bool PairedPages = false>
__global__ __launch_bounds__(128, 10) void Scores(const BF16 *weights,const __grid_constant__ CompactBatch batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && CUDART_VERSION >= 12040
    using namespace naive_dsa_mma;
    constexpr int Threads=128,Padding=16,Step=32;
    Row r=Expand(batch.rows[blockIdx.y],batch);r.key.Bind(r.cache);
    const int begin=r.Begin(),n=begin+r.Count(),tileStride=int(gridDim.x)*kKeys;
    int first=begin+int(blockIdx.x)*kKeys;
    if(first>=n)return;
    __shared__ __align__(16) unsigned char qs[kHeads][kIndexerDim+Padding];
    __shared__ __align__(16) unsigned char buffers[2][kKeys][kIndexerDim+Padding];
    __shared__ float scales[kHeads],weightsF[kHeads];
    const int t=threadIdx.x,lane=t&31,warp=t>>5;
    const auto* packed=reinterpret_cast<const unsigned char*>(r.query);
    // One aligned 16-byte vector per thread; no repeated FP8 conversion.
    const int head=t/8,column=(t%8)*16;
    if constexpr(AsyncQuery) {
        LoadScoreVector<true>(&qs[head][column],packed+t*16,true);
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
        *reinterpret_cast<uint4*>(&qs[head][column])=
            *reinterpret_cast<const uint4*>(packed+t*16);
    }
    if(t<kHeads){scales[t]=r.scale[t];weightsF[t]=(float)weights[r.sequence*kHeads+t];}
    if constexpr(PairedPages) LoadPairedPages(r,first,n,t,&buffers[0][0][0]);
    else LoadStagedKey(r,first,n,t,&buffers[0][0][0]);
    int stage=0;
    for(;first<n;first+=tileStride,stage^=1) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // The alternate buffer was released by the prior iteration's final
        // block barrier. Issue its next load before computing this tile.
        if(first+tileStride<n) {
            if constexpr(PairedPages) LoadPairedPages(r,first+tileStride,n,t,&buffers[stage^1][0][0]);
            else LoadStagedKey(r,first+tileStride,n,t,&buffers[stage^1][0][0]);
        }
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
