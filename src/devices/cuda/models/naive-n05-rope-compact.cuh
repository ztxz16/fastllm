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

// Staged verification has contiguous projections, but independently rebound KV.
// Preserve the original head-per-CTA calculation while reducing argument size.
struct VerifyRopeBinding {
    const void *const *cache;
    int offset;
};
constexpr int kVerifyRopeRows = 1024;
struct VerifyRopeBatch {
    RopeCacheRow geometry;
    VerifyRopeBinding rows[kVerifyRopeRows];
};
static_assert(sizeof(VerifyRopeBatch) + 128 <= 32764);
inline bool CanPackVerifyRope(const std::vector<RopeCacheRow> &rows,
        int heads, int kvHeads, int dim, int valueDim) {
    const auto &first = rows.front();
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &r = rows[i];
        if (!r.cache || !r.length || !r.ringVerify || r.pageLen || r.keyPages ||
            r.valuePages || r.indexKey || r.indexCache || r.separateIndex || r.scaledIndexer ||
            r.keyRows != first.keyRows || r.valueRows != first.valueRows || r.fp8 != first.fp8 ||
            r.q != first.q + i * heads * dim || r.k != first.k + i * kvHeads * dim ||
            r.v != first.v + i * kvHeads * valueDim || r.positions != first.positions + i ||
            r.length != first.length + i ||
            r.qkv != (first.qkv ? first.qkv + i * ((heads + kvHeads) * dim + kvHeads * valueDim) : nullptr))
            return false;
    }
    return true;
}
__device__ __forceinline__ RopeCacheRow VerifyRopeRow(const VerifyRopeBatch &batch,
        int i, int heads, int kvHeads, int dim, int valueDim) {
    auto r = batch.geometry;
    r.q += (size_t)i * heads * dim;
    r.k += (size_t)i * kvHeads * dim;
    r.v += (size_t)i * kvHeads * valueDim;
    if (r.qkv) r.qkv += (size_t)i * ((heads + kvHeads) * dim + kvHeads * valueDim);
    r.positions += i;
    r.length += i;
    r.cache = batch.rows[i].cache;
    r.verifyOffset = batch.rows[i].offset;
    return r;
}
#endif
