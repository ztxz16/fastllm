template<class Act> struct RelativeInputs {
    const Act *q, *r, *k, *v, *dk, *dv;
    const float *qs, *rs, *kw, *vw, *table;
    int heads, kvHeads, dim, rel, extent, taps;
    float qClip, rClip, eps;
};

__device__ float headScale(float v, float clip, int width, float eps, float *scratch) {
    const float sum = blockSum(v * v, scratch);
    return clip > 0 ? fminf(1.f, clip / (sqrtf(sum) + 1e-6f)) : rsqrtf(sum / width + eps);
}

template<class T, class Act>
__global__ void prepareRelative(const DecodeVisit *visit, RelativeInputs<Act> in, int tiles, float *q, float *bias) {
    __shared__ float scratch[threads], relative[threads];
    const int col = threadIdx.x, task = blockIdx.x;
    if (task < in.heads) {
        int i = task * in.dim + col;
        float x = col < in.dim ? float(in.q[i]) : 0.f;
        float w = col < in.dim ? in.qs[i] : 0.f;
        float scale = headScale(in.qClip > 0 ? x*w : x, in.qClip, in.dim, in.eps, scratch);
        if (col < in.dim) q[i] = x * scale * w;
        return;
    }
    int idx = task - in.heads;
    if (idx < in.heads * tiles) {
        const int head = idx / tiles, dist = idx % tiles * threads + col;
        if (idx % tiles * threads > visit->past) return;
        int i = head * in.rel + col;
        float x = col < in.rel ? float(in.r[i]) : 0.f;
        float w = col < in.rel ? in.rs[i] : 0.f;
        float scale = headScale(in.rClip > 0 ? x*w : x, in.rClip, in.rel, in.eps, scratch);
        relative[col] = x * scale * w;
        __syncthreads();
        if (dist < in.extent && dist <= visit->past) {
            float sum = 0;
            for (int j = 0; j < in.rel; ++j) sum += relative[j] * in.table[j * in.extent + dist];
            bias[head * in.extent + dist] = sum;
        }
        return;
    }
    idx -= in.heads * tiles;
    const int which = idx / in.kvHeads, channel = idx % in.kvHeads * in.dim + col;
    const int width = in.kvHeads * in.dim;
    float x = col < in.dim ? updateConvChannel(which ? in.v : in.k, visit, which, which ? in.vw : in.kw,
                                                which ? in.dv : in.dk, width, in.taps, in.kvHeads, channel) : 0.f;
    float scale = headScale(x, which ? 0.f : in.qClip, in.dim, in.eps, scratch);
    if (col < in.dim) {
        auto dst = static_cast<T *>(which ? visit->pages.value : visit->pages.key);
        dst[visit->pages.offset(visit->past, which, width) + channel] = T(x * scale);
    }
}
