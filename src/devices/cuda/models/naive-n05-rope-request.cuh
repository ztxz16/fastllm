// One CTA handles one request. All heads use the same BF16-rounded angles;
// each multiply, subtraction and final BF16 conversion matches RopeHeadTo.
template<class Output>
__device__ __forceinline__ void RotateCachedPair(const BF16 *src, Output dst,
        int d, int rd, const float *cosine, const float *sine) {
    const float a = (float)src[d], b = (float)src[d + rd / 2];
    const float c = cosine[d], s = sine[d];
    dst[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
    dst[d + rd / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
}

template<bool Packed, class Batch = RopeCacheBatch, int FixedKvHeads = 0>
__global__ __launch_bounds__(128) void RequestRope(
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        Batch batch,
        int headsArg, int kvHeadsArg, int dimArg, int vdArg, int idArg, int rdArg,
        float theta, BF16 valueScale, int windowArg, bool ringArg) {
    const int heads = FixedKvHeads ? 16 : headsArg;
    const int kvHeads = FixedKvHeads ? FixedKvHeads : kvHeadsArg;
    const int dim = FixedKvHeads ? 192 : dimArg, vd = FixedKvHeads ? 128 : vdArg;
    const int id = FixedKvHeads ? (FixedKvHeads == 2 ? 0 : 128) : idArg;
    const int rd = FixedKvHeads ? 64 : rdArg;
    const int window = FixedKvHeads ? (FixedKvHeads == 2 ? 128 : 0) : windowArg;
    const bool ring = FixedKvHeads ? FixedKvHeads == 2 : ringArg;
    const auto r = RopeRequestRow(batch, blockIdx.x);
    const int t = threadIdx.x;
    const int past = r.length ? (window ? min(*r.length - 1, window - 1) : *r.length - 1) : r.cachePast;
    const int row = ring ? ((*r.length - 1) & (kSwaWindow - 1)) : past;
    if (past < 0) {
        for (int i = t; i < heads * dim; i += 128) r.q[i] = __float2bfloat16(0.f);
        return;
    }
    if (row >= r.keyRows || row >= r.valueRows) return;
    __shared__ float cosine[96], sine[96];
    __shared__ BF16 rotatedIndex[128];
    if (t < rd / 2) {
        const float angle = r.positions[0] * powf(theta, -2.0f * t / rd);
        cosine[t] = RoundBF16(cosf(angle));
        sine[t] = RoundBF16(sinf(angle));
    }
    __syncthreads();
    for (int i = t; i < heads * (rd / 2); i += 128) {
        const int h = i / (rd / 2), d = i % (rd / 2);
        BF16 *dst = r.q + h * dim;
        const BF16 *src = Packed ? r.qkv + h * dim : dst;
        RotateCachedPair(src, dst, d, rd, cosine, sine);
    }
    if constexpr (Packed) {
        for (int i = t; i < heads * (dim - rd); i += 128) {
            const int h = i / (dim - rd), d = rd + i % (dim - rd);
            r.q[h * dim + d] = r.qkv[h * dim + d];
        }
    }
    const int *kp = r.pageLen && r.cache ? (const int *)r.cache[2] : r.keyPages;
    const int *vp = r.pageLen && r.cache ? (const int *)r.cache[3] : r.valuePages;
    const int physicalK = kp ? kp[row / r.pageLen] * r.pageLen + row % r.pageLen : row;
    const int physicalV = vp ? vp[row / r.pageLen] * r.pageLen + row % r.pageLen : row;
    const int tail = r.scaledIndexer ? FastllmNaiveIndexerBytes / (r.fp8 ? 1 : 2) : id;
    FastllmNaiveKVWriteRow key{r.cache ? (void *)r.cache[0] : r.key, r.fp8};
    FastllmNaiveKVWriteRow value{r.cache ? (void *)r.cache[1] : r.value, r.fp8};
    const int keyStride = kvHeads * dim + (r.separateIndex ? 0 : tail);
    for (int i = t; i < kvHeads * (rd / 2); i += 128) {
        const int h = i / (rd / 2), d = i % (rd / 2);
        const BF16 *src = Packed ? r.qkv + (heads + h) * dim : r.k + h * dim;
        auto dst = key + (size_t)physicalK * keyStride + h * dim;
        RotateCachedPair(src, dst, d, rd, cosine, sine);
    }
    for (int i = t; i < kvHeads * (dim - rd); i += 128) {
        const int h = i / (dim - rd), d = rd + i % (dim - rd);
        const BF16 *src = Packed ? r.qkv + (heads + h) * dim : r.k + h * dim;
        (key + (size_t)physicalK * keyStride + h * dim)[d] = src[d];
    }
    for (int i = t; i < kvHeads * vd; i += 128) {
        const BF16 *src = Packed ? r.qkv + (heads + kvHeads) * dim : r.v;
        (value + (size_t)physicalV * kvHeads * vd)[i] =
            __float2bfloat16_rn((float)src[i] * (float)valueScale);
    }
    if (!id) return;
    auto dst = key + (size_t)physicalK * keyStride + kvHeads * dim;
    if (r.separateIndex) {
        int physical = physicalK;
        if (r.indexPageLen && r.indexPageLen != r.pageLen) {
            if (row % r.pageLen / r.indexPageLen != r.indexContextRank) return;
            physical = physical / r.pageLen * r.indexPageLen + row % r.indexPageLen;
        }
        dst = FastllmNaiveKVWriteRow{r.cache ? (void *)r.cache[r.pageLen ? 4 : 2] : r.indexCache, r.fp8} + (size_t)physical * tail;
    }
    if (r.scaledIndexer) {
        if (t < rd / 2) RotateCachedPair(r.indexKey, rotatedIndex, t, rd, cosine, sine);
        for (int i = rd + t; i < id; i += 128) rotatedIndex[i] = r.indexKey[i];
        __syncthreads();
        FastllmNaiveStoreIndexer((float)rotatedIndex[t], dst.data);
    } else {
        if (t < rd / 2) RotateCachedPair(r.indexKey, dst, t, rd, cosine, sine);
        for (int i = rd + t; i < id; i += 128) dst[i] = r.indexKey[i];
    }
}
