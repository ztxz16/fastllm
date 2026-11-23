// Split the paged prefill scan across the device for a short chain of
// verification queries. The merge includes the sink exactly once per query.
struct VerifyRelative : PrefillRelative {
    template<class Params>
    __device__ VerifyRelative(const Params &p, unsigned batch, uint8_t *smem) : PrefillRelative(p, batch, smem) {
        // FlashInfer normally starts splitting at the first query's window edge.
        // Fix the split grid to the page origin for batch-independent reductions.
        this->window_left = this->kv_len;
    }
    REGISTER_LOGITS_TRANSFORM(p, score, batch, row, key, head, kvHead, {
        if (p.window_left >= 0 && int(p.past + row - key) > p.window_left) return -flashinfer::math::inf;
        return PrefillRelative::LogitsTransform(p, score, batch, row, key, head, kvHead);
    })
    REGISTER_OUTPUT_TRANSFORM(p, out, batch, row, head, m, d, scale, {
        return d > 0.f ? out / d : 0.f;
    })
};
using VerifyTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal, 32, 1, 2, 16, 8, 2, 1,
    flashinfer::PosEncodingMode::kNone, half, half, float, float, int, VerifyRelative, false>;

struct VerifyPrefillParams : PrefillParams<half, 128> {
    const DecodeVisit *visits;
    int chain, maxSplits, minTilesPerSplit;
};

// Partition by absolute position. Rows on opposite sides of a grain
// boundary run as separate segments so their FP32 reduction order matches
// single-token decode, even in a clipped, multi-request graph replay.
struct VerifyPartition {
    static constexpr int minTiles = 4;
    int origin, chunk, first, end;
    __device__ VerifyPartition(const DecodeVisit &visit, int row, int live, int limit, int windowLeft, int minimum) {
        // Keep the same origin in a packed chain and in single-row decode.
        // Only pages intersecting the window are addressed, even after eviction.
        int past = visit.past;
        origin = windowLeft < 0 ? 0 : max(0, past + row - windowLeft) / visit.pages.pageLen * visit.pages.pageLen;
        past -= origin;
        constexpr int tile = VerifyTraits::CTA_TILE_KV;
        const int tiles = (past + row) / tile + 1;
        const int grain = max(minimum, (tiles - 1) / limit + 1);
        chunk = grain * tile;
        first = grain == minimum ? 0 : int(max(0LL, 1LL * (grain - 1) * tile * limit - past));
        end = int(min(static_cast<long long>(live), 1LL * chunk * limit - past));
        if (windowLeft >= 0) {
            if (origin) first = max(first, windowLeft - past);
            end = int(min(static_cast<long long>(end), 1LL * windowLeft + visit.pages.pageLen - past));
        }
    }
};

// The first visible key may lie partway through its page.
int verifySplitCount(int limit, int window, int pageLen, int minTiles) {
    if (!window) return limit;
    const int grain = minTiles * VerifyTraits::CTA_TILE_KV;
    return int(std::min<int64_t>(limit, (int64_t(window) - 1 + pageLen + grain - 1) / grain));
}

__device__ __forceinline__ void verifySegment(VerifyPrefillParams p, const DecodeVisit &visit, int first,
                                              int rows, int tile, int split, const VerifyPartition &part,
                                              int kvHead, int kvHeads) {
    int chunk = part.chunk;
    p.paged_kv.origin = part.origin;
    p.past = visit.past - part.origin;
    p.rows = rows;
    const int length = p.past + rows;
    if (tile * VerifyTraits::CTA_TILE_Q >= rows * unsigned(p.group_size) || split > (length - 1) / chunk) return;
    p.relative += size_t(first) * p.num_qo_heads * p.distances;
    p.q += size_t(first) * p.q_stride_n;
    p.o += size_t(first) * p.maxSplits * p.num_qo_heads * 128;
    p.lse += size_t(first) * p.maxSplits * p.num_qo_heads;
    p.paged_kv.length = length;
    const int stride = kvHeads * 128;
    p.paged_kv.k_data = {visit.pages, nullptr, 0, visit.past, stride, false};
    p.paged_kv.v_data = {visit.pages, nullptr, 0, visit.past, stride, true};
    int zero = 0, indptr[] = {0, length};
    p.request_indices = p.q_indptr = p.o_indptr = &zero;
    p.qo_tile_indices = &tile;
    p.kv_tile_indices = &split;
    p.kv_chunk_size_ptr = &chunk;
    p.paged_kv.indptr = indptr;
    extern __shared__ uint8_t smem[];
    auto &storage = *reinterpret_cast<VerifyTraits::SharedStoragePaged *>(smem);
    flashinfer::BatchPrefillWithPagedKVCacheDevice<VerifyTraits>(p, storage, threadIdx, 0, kvHead, kvHeads);
}

__global__ __launch_bounds__(VerifyTraits::NUM_THREADS) void pagedVerifyAttention(VerifyPrefillParams p) {
    const int group = p.group_size;
    const int tiles = (p.chain * group + VerifyTraits::CTA_TILE_Q - 1) / VerifyTraits::CTA_TILE_Q;
    const int first = blockIdx.x / tiles * p.chain, tile = blockIdx.x % tiles;
    const DecodeVisit &visit = p.visits[first];
    const int live = p.chain == 1 ? 1 : visit.verifyRemaining;
    for (int row = 0; row < live;) {
        const VerifyPartition part(visit, row, live, p.maxSplits, p.window_left, p.minTilesPerSplit);
        verifySegment(p, p.visits[first + row], first + row, part.end - row, tile, blockIdx.y, part, blockIdx.z, gridDim.z);
        row = part.end;
        if (row < live) __syncthreads();
    }
}

// A single window query needs one query tile. Put adjacent KV heads in grid.x
// so the SMs can scan the same page across heads before advancing the split.
__global__ __launch_bounds__(VerifyTraits::NUM_THREADS) void pagedWindowDecode(VerifyPrefillParams p) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    const DecodeVisit &visit = *p.visits;
    const VerifyPartition part(visit, 0, 1, p.maxSplits, p.window_left, p.minTilesPerSplit);
    // FlashInfer waits for preparation before reading the packed query, bias or KV.
    verifySegment(p, visit, 0, 1, 0, blockIdx.y, part, blockIdx.x, gridDim.x);
}

template<class Act, bool WarpPerHead, bool WindowDecode = false>
__global__ void mergeVerifyAttention(VerifyPrefillParams p, const Act *sinks, const Act *gate, Act *out,
                                  int sinkStride, int gateStride) {
    constexpr int channels = WarpPerHead ? 4 : 1;
    const int lane = threadIdx.x % 32;
    const int row = WindowDecode ? 0 : blockIdx.x;
    const int head = WarpPerHead ? blockIdx.y * 4 + threadIdx.x / 32 : blockIdx.y;
    if (head >= p.num_qo_heads) return;
    const int channel = WarpPerHead ? lane * channels : threadIdx.x;
    const int local = WindowDecode ? 0 : row % p.chain, first = row - local;
    const size_t dst = (size_t(row) * p.num_qo_heads + head) * 128 + channel;
    const DecodeVisit &visit = p.visits[first];
    const int live = WindowDecode || p.chain == 1 ? 1 : visit.verifyRemaining;
    if (local >= live) {
#pragma unroll
        for (int j = 0; j < channels; ++j) out[dst + j] = Act(0.f);
        return;
    }
    const VerifyPartition part(visit, local, live, p.maxSplits, p.window_left, p.minTilesPerSplit);
    const int past = visit.past - part.origin;
    const int splits = (past + part.end - 1) / part.chunk + 1, active = (past + local) / part.chunk + 1;
    const size_t base = WindowDecode ? size_t(head) : (size_t(first + part.first) * p.maxSplits + size_t(local - part.first) * splits) * p.num_qo_heads + head;
    const float sink = float(sinks[size_t(row) * sinkStride + head]) * RelativeAttention::sm_scale_log2;
    const int firstSplit = p.window_left < 0 ? 0 : max(0, past + local - p.window_left) / part.chunk;
    float maximum = sink, laneWeight = 0.f;
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // The visit and sink precede this pipeline. Only partial results depend on the scan.
    if constexpr (WindowDecode) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if constexpr (WarpPerHead || WindowDecode) {
        const float lse = lane >= firstSplit && lane < active ? p.lse[base + lane * p.num_qo_heads] : -INFINITY;
        maximum = fmaxf(sink, lse);
#pragma unroll
        for (int step = 16; step; step /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
        laneWeight = exp2f(lse - maximum);
    } else {
        for (int s = firstSplit; s < active; ++s) maximum = fmaxf(maximum, p.lse[base + s * p.num_qo_heads]);
    }
    float denom = exp2f(sink - maximum), sum[channels]{};
    for (int s = firstSplit; s < active; ++s) {
        const size_t i = base + s * p.num_qo_heads;
        const float w = (WarpPerHead || WindowDecode) ? __shfl_sync(0xffffffff, laneWeight, s) : exp2f(p.lse[i] - maximum);
        denom += w;
        if constexpr (WarpPerHead) {
            const float4 v = reinterpret_cast<const float4 *>(p.o + i * 128)[lane];
            sum[0] += w * v.x;
            sum[1] += w * v.y;
            sum[2] += w * v.z;
            sum[3] += w * v.w;
        } else {
            sum[0] += w * p.o[i * 128 + channel];
        }
    }
#pragma unroll
    for (int j = 0; j < channels; ++j) {
        const float g = float(gate[size_t(row) * gateStride + head * 128 + channel + j]);
        out[dst + j] = Act(sum[j] / denom / (1.f + expf(-g)));
    }
}

template<class Act>
void launchVerifyAttention(const AttentionRow &a, const Matrix &scratch, const Act *sink, const Act *gate,
                        Act *out, int rows, int chain, int maxSplits, int smCount, int minTiles, bool pipelineWindow = false) {
    VerifyPrefillParams p{};
    p.q = reinterpret_cast<half *>(const_cast<float *>(a.q));
    p.o = scratch.p;
    p.lse = p.o + size_t(rows) * maxSplits * a.heads * 128;
    p.visits = a.visit;
    p.chain = chain;
    p.maxSplits = maxSplits;
    p.minTilesPerSplit = minTiles;
    p.relative = a.relative;
    p.distances = a.distances;
    p.floor = a.floor;
    p.alpha = a.alpha;
    p.num_qo_heads = a.heads;
    const int group = a.heads / a.kvHeads;
    p.group_size = group;
    p.q_stride_h = 256;
    p.q_stride_n = a.heads * p.q_stride_h;
    p.qkScale = 1.f / sqrtf(128.f);
    p.sm_scale = 1.f;
    p.window_left = a.window ? a.window - 1 : -1;
    p.partition_kv = true;
#if CUDART_VERSION >= 12080
    if (pipelineWindow) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.stream = cudaStreamPerThread;
        config.attrs = &attribute;
        config.numAttrs = 1;
        config.gridDim = dim3(a.kvHeads, maxSplits);
        config.blockDim = dim3(32, VerifyTraits::NUM_WARPS_Q);
        config.dynamicSmemBytes = sizeof(VerifyTraits::SharedStoragePaged);
        check(cudaLaunchKernelEx(&config, pagedWindowDecode, p));
        config.gridDim = dim3(1, a.heads);
        config.blockDim = 128;
        config.dynamicSmemBytes = 0;
        check(cudaLaunchKernelEx(&config, mergeVerifyAttention<Act, false, true>, p, sink, gate, out,
                                 a.sinkStride, a.gateStride));
        return;
    }
#endif
    const int tiles = (chain * group + VerifyTraits::CTA_TILE_Q - 1) / VerifyTraits::CTA_TILE_Q;
    pagedVerifyAttention<<<dim3(rows / chain * tiles, maxSplits, a.kvHeads), dim3(32, VerifyTraits::NUM_WARPS_Q),
                        sizeof(VerifyTraits::SharedStoragePaged), cudaStreamPerThread>>>(p);
    // Each warp broadcasts up to 32 split weights and reads four contiguous
    // channels per lane. Use one head per CTA when this would underfill the GPU.
    const int headTiles = (a.heads + 3) / 4;
    const bool warpPerHead = maxSplits <= 32 && int64_t(headTiles) * rows >= smCount;
    auto merge = warpPerHead ? mergeVerifyAttention<Act, true> : mergeVerifyAttention<Act, false>;
    merge<<<dim3(rows, warpPerHead ? headTiles : a.heads), 128, 0, cudaStreamPerThread>>>(
        p, sink, gate, out, a.sinkStride, a.gateStride);
}
