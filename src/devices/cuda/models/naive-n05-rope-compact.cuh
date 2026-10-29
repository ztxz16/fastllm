#pragma once
// Decode cache bindings/lengths remain device pointers and can change on graph replay.
__device__ __forceinline__ RopeCacheRow RopeRequestRow(const RopeCacheBatch &batch, int row) {
    return batch.rows[row];
}
#if CUDART_VERSION >= 12010
struct CompactRopeRequest {
    BF16 *q;
    const BF16 *k,*v,*indexKey,*qkv;
    const float *positions;
    const int *length;
    const void *const *cache;
    int indexContextRank;
};
constexpr int kCompactRopeRequests=448;
struct CompactRopeRequests {
    int keyRows,valueRows,pageLen,indexPageLen;
    bool fp8,scaledIndexer,separateIndex;
    CompactRopeRequest rows[kCompactRopeRequests];
};
static_assert(sizeof(CompactRopeRequest)==72);
static_assert(sizeof(CompactRopeRequests)+128<=32764);
__device__ __forceinline__ RopeCacheRow RopeRequestRow(const CompactRopeRequests &b, int row){
    const auto &x=b.rows[row];
    RopeCacheRow r{};
    r.q=x.q;r.k=x.k;r.v=x.v;r.indexKey=x.indexKey;r.qkv=x.qkv;
    r.positions=x.positions;r.length=x.length;r.cache=x.cache;
    r.keyRows=b.keyRows;r.valueRows=b.valueRows;r.pageLen=b.pageLen;
    r.indexPageLen=b.indexPageLen;r.fp8=b.fp8;r.scaledIndexer=b.scaledIndexer;
    r.separateIndex=b.separateIndex;r.indexContextRank=x.indexContextRank;
    return r;
}
inline bool PackRope(const RopeCacheRow *rows,int count,CompactRopeRequests &batch){
    const auto &first=rows[0];
    batch.keyRows=first.keyRows;batch.valueRows=first.valueRows;batch.pageLen=first.pageLen;
    batch.indexPageLen=first.indexPageLen;batch.fp8=first.fp8;
    batch.scaledIndexer=first.scaledIndexer;batch.separateIndex=first.separateIndex;
    for(int i=0;i<count;++i){
        const auto &r=rows[i];
        if(!r.cache||!r.length||r.keyRows!=batch.keyRows||r.valueRows!=batch.valueRows||
           r.pageLen!=batch.pageLen||r.indexPageLen!=batch.indexPageLen||
           r.fp8!=batch.fp8||r.scaledIndexer!=batch.scaledIndexer||r.separateIndex!=batch.separateIndex)return false;
        batch.rows[i]={r.q,r.k,r.v,r.indexKey,r.qkv,r.positions,r.length,r.cache,r.indexContextRank};
    }
    return true;
}

#endif
