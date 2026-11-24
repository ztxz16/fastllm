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
struct WindowVerifyRelative : VerifyRelative, RelativeScoreTile<true, float, true> {
    template<class P> __device__ WindowVerifyRelative(const P &p, unsigned batch, uint8_t *smem)
        : VerifyRelative(p, batch, smem), RelativeScoreTile<true, float, true>(p, *p.qo_tile_indices) {}
};

template<class Relative>
using VerifyTile = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal, 32, 1, 2, 16, 8, 2, 1,
    flashinfer::PosEncodingMode::kNone, half, half, float, float, int, Relative, false>;
using VerifyTraits = VerifyTile<VerifyRelative>;
using WindowVerifyTraits = VerifyTile<WindowVerifyRelative>;
static_assert(sizeof(WindowVerifyTraits::SharedStoragePaged) <= 48 * 1024);

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

template<class Traits>
__device__ __forceinline__ void verifySegment(VerifyPrefillParams p, const DecodeVisit &visit, int first,
                                              int rows, int tile, int split, const VerifyPartition &part,
                                              int kvHead, int kvHeads) {
    int chunk = part.chunk;
    p.paged_kv.origin = part.origin;
    p.past = visit.past - part.origin;
    p.rows = rows;
    const int length = p.past + rows;
    if (tile * Traits::CTA_TILE_Q >= rows * unsigned(p.group_size) || split > (length - 1) / chunk) return;
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
    auto &storage = *reinterpret_cast<typename Traits::SharedStoragePaged *>(smem);
    flashinfer::BatchPrefillWithPagedKVCacheDevice<Traits>(p, storage, threadIdx, 0, kvHead, kvHeads);
}

#include "window-verify.cuh"

template<class Traits, bool RegisterQuery = false>
__global__ __launch_bounds__(Traits::NUM_THREADS) void pagedVerifyAttention(VerifyPrefillParams p) {
    cuda_launch::DependencyStart();
    const int group = p.group_size;
    const int tiles = (p.chain * group + Traits::CTA_TILE_Q - 1) / Traits::CTA_TILE_Q;
    const int first = blockIdx.x / tiles * p.chain, tile = blockIdx.x % tiles;
    const DecodeVisit &visit = p.visits[first];
    const int live = p.chain == 1 ? 1 : visit.verifyRemaining;
    int row = 0, split = blockIdx.y;
    bool firstSegmentOnly = false;
    if constexpr (std::is_same_v<Traits, WindowVerifyTraits>) {
        // On a lightly occupied grid, let separate CTAs scan the two sides of a
        // page boundary. Both write the same partials consumed by the usual merge.
        if (gridDim.y > p.maxSplits) {
            if (split < p.maxSplits) firstSegmentOnly = true;
            else {
                if (live <= 0) return;
                row = VerifyPartition(visit, 0, live, p.maxSplits, p.window_left, p.minTilesPerSplit).end;
                split -= p.maxSplits;
            }
        }
    }
    while (row < live) {
        const VerifyPartition part(visit, row, live, p.maxSplits, p.window_left, p.minTilesPerSplit);
        if constexpr (RegisterQuery)
            scanVerifyWindow(p, p.visits[first + row], first + row, part.end - row, tile, split, part, blockIdx.z, gridDim.z);
        else
            verifySegment<Traits>(p, p.visits[first + row], first + row, part.end - row, tile, split, part, blockIdx.z, gridDim.z);
        if (firstSegmentOnly) break;
        row = part.end;
        if (row < live) __syncthreads();
    }
}

// A single decode query needs one query tile. Put adjacent KV heads in grid.x
// so the SMs can scan the same page across heads before advancing the split.
__global__ __launch_bounds__(VerifyTraits::NUM_THREADS) void pagedSingletonDecode(VerifyPrefillParams p) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cuda_launch::DependencyStart();
#endif
    const DecodeVisit &visit = *p.visits;
    const VerifyPartition part(visit, 0, 1, p.maxSplits, p.window_left, p.minTilesPerSplit);
    // Preparation has published the packed query, bias and latest KV.
    verifySegment<VerifyTraits>(p, visit, 0, 1, 0, blockIdx.y, part, blockIdx.x, gridDim.x);
}

template<class Act, bool WarpPerHead, bool WindowDecode = false, bool PackQ8 = false, bool Pdl = false>
__global__ void mergeVerifyAttention(VerifyPrefillParams p, const Act *sinks, const Act *gate, std::conditional_t<PackQ8, InklingQ81, Act> *out,
                                  int sinkStride, int gateStride) {
    constexpr int channels = WarpPerHead ? 4 : 1;
    const int lane = threadIdx.x % 32;
    const int row = WindowDecode ? 0 : blockIdx.x;
    if constexpr (Pdl || PackQ8) cuda_launch::DependencyStart();
    const int head = WarpPerHead ? blockIdx.y * 4 + threadIdx.x / 32 : blockIdx.y;
    if (head >= p.num_qo_heads) return;
    const int channel = WarpPerHead ? lane * channels : threadIdx.x;
    const int local = WindowDecode ? 0 : row % p.chain, first = row - local;
    const size_t dst = (size_t(row) * p.num_qo_heads + head) * 128 + channel;
    const DecodeVisit &visit = p.visits[first];
    const int live = WindowDecode || p.chain == 1 ? 1 : visit.verifyRemaining;
    if constexpr (PackQ8) {
        if (local >= live) {
            if (channel < 32) Q8RowStore::store<Act>(out, size_t(row) * p.num_qo_heads * 32 + head * 32 + channel, make_float4(0,0,0,0));
            return;
        }
    } else {
        if (local >= live) {
#pragma unroll
            for (int j = 0; j < channels; ++j) out[dst + j] = Act(0.f);
            return;
        }
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
    if constexpr (WindowDecode && !Pdl && !PackQ8) cuda_launch::DependencyStart();
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
    if constexpr (PackQ8) {
        static_assert(!WarpPerHead && !WindowDecode);
        __shared__ __align__(16) Act rowValues[128];
        const float g = float(gate[size_t(row) * gateStride + head * 128 + channel]);
        rowValues[channel] = Act(sum[0] / denom / (1.f + expf(-g)));
        __syncthreads();
        if (channel < 32) {
            const auto v = reinterpret_cast<const cuda_attn_res::Channels4<Act> *>(rowValues)[channel];
            Q8RowStore::store<Act>(out, size_t(row) * p.num_qo_heads * 32 + head * 32 + channel, make_float4(float(v.x), float(v.y), float(v.z), float(v.w)));
        }
    } else {
#pragma unroll
        for (int j = 0; j < channels; ++j) {
            const float g = float(gate[size_t(row) * gateStride + head * 128 + channel + j]);
            out[dst + j] = Act(sum[j] / denom / (1.f + expf(-g)));
        }
    }
}

#if CUDART_VERSION >= 12080
// Each captured branch scans and merges disjoint request rows. Rejoin every
// branch before the output projection consumes the full batch.
template<class Launch>
void captureWindowRanges(int requests, int minRequests, Launch launch) {
    constexpr int maxBranches = 4;
    const int groups = std::min(maxBranches, requests / minRequests);
    cudaStream_t stream = cudaStreamPerThread;
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t *nodes = nullptr;
    size_t count = 0;
    if (groups > 1) {
        // Preserve an upstream capture with custom edge metadata as it stands.
        const auto result = FastllmCudaStreamGetCaptureDependencies(stream, &status, nullptr, &nodes, &count);
        if (result == cudaErrorLossyQuery) {
            cudaGetLastError();
            launch(0, requests);
            return;
        }
        check(result);
    }
    if (groups < 2 || status != cudaStreamCaptureStatusActive || count != 1) {
        launch(0, requests);
        return;
    }
    cudaGraphNode_t producer = nodes[0];
    std::array<cudaGraphNode_t, maxBranches> tails;
    for (int group = 0; group < groups; ++group) {
        if (group) check(FastllmCudaStreamUpdateCaptureDependencies(stream, &producer, 1, cudaStreamSetCaptureDependencies));
        const int first = int(int64_t(requests) * group / groups);
        const int end = int(int64_t(requests) * (group + 1) / groups);
        launch(first, end - first);
        check(FastllmCudaStreamGetCaptureDependencies(stream, &status, nullptr, &nodes, &count));
        if (count != 1) throw std::runtime_error("Inkling window merge has no unique graph completion");
        tails[group] = nodes[0];
    }
    check(FastllmCudaStreamUpdateCaptureDependencies(stream, tails.data(), groups, cudaStreamSetCaptureDependencies));
}
#endif

template<class Act>
void launchVerifyAttention(const AttentionRow &a, const Matrix &scratch, const Act *sink, const Act *gate,
                        Act *out, int rows, int chain, int maxSplits, int smCount, int minTiles, bool pipelineDecode = false, InklingQ81 *packed = nullptr, bool verifyPipeline = false) {
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
    if (verifyPipeline) {
        const int tiles = (chain * group + VerifyTraits::CTA_TILE_Q - 1) / VerifyTraits::CTA_TILE_Q;
        const int requests = rows / chain;
        const dim3 block(32, VerifyTraits::NUM_WARPS_Q);
        const int warpBlocks = (a.heads + 3) / 4;
        const bool warpMerge = !packed && maxSplits <= 32 && int64_t(warpBlocks) * rows >= smCount;
        const bool registerQuery = a.window && chain > 1 && int64_t(requests) * tiles * a.kvHeads >= smCount;
        auto launch = [&](int first, int count) {
            const size_t row = size_t(first) * chain;
            auto range = p;
            range.visits += row;
            range.q += row * p.q_stride_n;
            range.relative += row * a.heads * p.distances;
            range.o += row * maxSplits * a.heads * 128;
            range.lse += row * maxSplits * a.heads;
            dim3 grid(count * tiles, maxSplits, a.kvHeads);
            bool launched;
            if (registerQuery) {
                launched = cuda_launch::Launch<pagedVerifyAttention<WindowVerifyTraits, true>>(grid, block,
                    2 * verifyTileVectors * sizeof(uint4), cudaStreamPerThread, range);
            } else if (a.window) {
                if (chain > 1 && uint64_t(grid.x) * grid.y * grid.z <= smCount) grid.y *= 2;
                launched = cuda_launch::Launch<pagedVerifyAttention<WindowVerifyTraits>>(grid, block,
                    sizeof(WindowVerifyTraits::SharedStoragePaged), cudaStreamPerThread, range);
            } else {
                launched = cuda_launch::Launch<pagedVerifyAttention<VerifyTraits>>(grid, block,
                    sizeof(VerifyTraits::SharedStoragePaged), cudaStreamPerThread, range);
            }
            if (!launched) throw std::runtime_error("Inkling verify attention launch failed");
            auto merge = [&](auto *dest, auto perWarp) {
                constexpr bool pack = std::is_same_v<decltype(dest), InklingQ81 *>, warp = decltype(perWarp)::value;
                return cuda_launch::Launch<mergeVerifyAttention<Act, warp, false, pack, true>>(
                    dim3(count * chain, warp ? warpBlocks : a.heads), 128, 0, cudaStreamPerThread, range,
                    sink + row * a.sinkStride, gate + row * a.gateStride, dest, a.sinkStride, a.gateStride);
            };
            const bool ok = packed ? merge(packed, std::false_type{}) :
                (warpMerge ? merge(out + row * a.heads * 128, std::true_type{}) : merge(out, std::false_type{}));
            if (!ok) throw std::runtime_error("Inkling verify attention merge failed");
        };
        if (registerQuery && warpMerge) {
            // Keep a full wave available to both kernels in each independent range.
            const int blocks = std::min(tiles * a.kvHeads, chain * warpBlocks);
            captureWindowRanges(requests, std::max(1, (smCount + blocks - 1) / blocks), launch);
        } else {
            launch(0, requests);
        }
        return;
    }
    if (pipelineDecode || packed) {
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
        check(cudaLaunchKernelEx(&config, pagedSingletonDecode, p));
        config.gridDim = dim3(1, a.heads);
        config.blockDim = 128;
        config.dynamicSmemBytes = 0;
        if (packed)
            check(cudaLaunchKernelEx(&config, mergeVerifyAttention<Act, false, false, true>, p, sink, gate, packed,
                                     a.sinkStride, a.gateStride));
        else if (a.window) check(cudaLaunchKernelEx(&config, mergeVerifyAttention<Act, false, true>, p, sink, gate, out,
                                     a.sinkStride, a.gateStride));
        else check(cudaLaunchKernelEx(&config, mergeVerifyAttention<Act, false, false, false, true>, p, sink, gate,
                                     out, a.sinkStride, a.gateStride));
        return;
    }
#endif
    const int tiles = (chain * group + VerifyTraits::CTA_TILE_Q - 1) / VerifyTraits::CTA_TILE_Q;
    pagedVerifyAttention<VerifyTraits><<<dim3(rows / chain * tiles, maxSplits, a.kvHeads), dim3(32, VerifyTraits::NUM_WARPS_Q),
                        sizeof(VerifyTraits::SharedStoragePaged), cudaStreamPerThread>>>(p);
    // Each warp broadcasts up to 32 split weights and reads four contiguous
    // channels per lane. Use one head per CTA when this would underfill the GPU.
    const int headTiles = (a.heads + 3) / 4;
    const bool warpPerHead = maxSplits <= 32 && int64_t(headTiles) * rows >= smCount;
    auto merge = warpPerHead ? mergeVerifyAttention<Act, true> : mergeVerifyAttention<Act, false>;
    merge<<<dim3(rows, warpPerHead ? headTiles : a.heads), 128, 0, cudaStreamPerThread>>>(
        p, sink, gate, out, a.sinkStride, a.gateStride);
}
