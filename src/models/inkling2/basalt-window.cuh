// Adapted from Basalt v0.5.6, Copyright (c) 2026 Jesús Gulfo, MIT.
// https://github.com/jesdga95/basalt/blob/v0.5.6/src/models/inkling2/kernels/attn.cu
// License: third_party/basalt/LICENSE. This specialization uses FastLLM's
// FP16 page table and convolution history, with a PDL edge to the output projection.
namespace basalt_window {
constexpr int kWarps = 4, kTile = 32, kGroup = 3, kHeadDim = 128, kRel = 16;
constexpr int kClusterSplits = 8, kVStride = kHeadDim / 2 + 4;
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
struct Smem {
    float q[kGroup][kHeadDim];
    float rs[kGroup][kRel];
    uint32_t v[kWarps][kTile][kVStride];   // a tile's value rows as fp16 pairs
    float m[kWarps][kGroup], l[kWarps][kGroup];
    float acc[kWarps][kGroup][kHeadDim];
};

struct FusedSmem {
    Smem s;
    alignas(16) half kw[kHeadDim], vw[kHeadDim];
    float pm[kGroup], pl[kGroup], pacc[kGroup][kHeadDim];
};

template<class Out> __global__ void __launch_bounds__(128) attn_fused_kernel(const DecodeVisit *visits, WindowInput in,
                                                         int window, Out *out) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    namespace cg = cooperative_groups;
    extern __shared__ __align__(16) unsigned char shared[];
    FusedSmem &fs = *reinterpret_cast<FusedSmem *>(shared);
    Smem &sm = fs.s;
    constexpr int kLoads = kHeadDim * sizeof(half) / sizeof(uint4);
    const int g = blockIdx.y, s = blockIdx.z, w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const auto &v = *visits;
    const int p = v.past, lo = max(0, p - window + 1), splits = (p - lo + 128) / 128;
    const int s_lo = lo + 128*s, s_hi = min(s_lo+128,p+1), key = s_lo + w*32 + lane;
    const bool active = s < splits, valid = key < s_hi;
    const int stride = in.kvHeads * kHeadDim;
    uint4 kr[kLoads], vr[kLoads];
    if (valid && key < p) {
        const auto &pages = v.pages;
        const int page = pages.shift >= 0 ? key >> pages.shift : key / pages.pageLen;
        const int row = key - page * pages.pageLen;
        const int2 ids = pages.ids[page - pages.first];
        const auto *kp = reinterpret_cast<const uint4 *>(static_cast<const half *>(pages.key) +
            (size_t(ids.x) * pages.pageLen + row) * stride + g * kHeadDim);
        const auto *vp = reinterpret_cast<const uint4 *>(static_cast<const half *>(pages.value) +
            (size_t(ids.y) * pages.pageLen + row) * stride + g * kHeadDim);
#pragma unroll
        for (int i=0;i<kLoads;++i) { kr[i]=kp[i]; vr[i]=vp[i]; }
    } else {
#pragma unroll
        for (int i=0;i<kLoads;++i) kr[i]=vr[i]=make_uint4(0,0,0,0);
    }
    float tv[kRel];
#pragma unroll
    for (int i=0;i<kRel;++i) tv[i] = valid && p-key < in.extent ? in.table[i*in.extent+p-key] : 0.f;
    const int mergeHead = g*kGroup+s;
    cuda_launch::DependencyWait();
    const float sink = s<kGroup ? in.sources.items[3].read(0,mergeHead) : 0.f;
    const float gate = s<kGroup ? in.sources.items[2].read(0,mergeHead*kHeadDim+threadIdx.x) : 0.f;
    if (active) {
        if (w >= 2) {
            // Keep K/V convolution and query clipping on distinct warps, as in
            // Basalt's prep::q_heads. Both query rows start their loads together.
            const int first = w == 2 ? 0 : 2;
            float q[2][4], rel[3];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = (g * 3 + first + h) * 128 + lane + j * 32;
                    q[h][j] = first + h < 3 ? in.sources.items[0].read(0, c) * in.qs[c] : 0.f;
                }
            }
            if (w == 3) {
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    const int c = (g * 3 + h) * 16 + lane;
                    rel[h] = lane < 16 ? in.sources.items[1].read(0, c) * in.rs[c] : 0.f;
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (first + h >= 3) break;
                const float factor = prefillScale(q[h], in.qClip, 128, in.eps, false);
#pragma unroll
                for (int j = 0; j < 4; ++j) sm.q[first+h][lane+j*32] = q[h][j]*factor;
            }
            if (w == 3) {
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    const float v4[4] = {rel[h], 0.f, 0.f, 0.f};
                    const float f = prefillScale(v4, in.rClip, 16, in.eps, false);
                    if (lane < 16) sm.rs[h][lane] = rel[h]*f;
                }
            }
        } else if (s == splits - 1) {
            const bool value = w == 1;
            const auto src = in.sources.items[value ? 6 : 4];
            float a[4], x[4], h1[4], h2[4];
            float *hist = v.history[w];
            const float *wt = value ? in.vw : in.kw;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = g*kHeadDim+lane+32*j;
                const float4 t = reinterpret_cast<const float4 *>(wt)[c];
                x[j] = src.read(0, c);
                h1[j] = hist[stride + c];
                h2[j] = hist[2*stride + c];
                const float h0 = hist[c];
                float conv = fmaf(x[j], t.w, x[j]);
                conv = fmaf(h2[j], t.z, conv);
                conv = fmaf(h1[j], t.y, conv);
                a[j] = fmaf(h0, t.x, conv) * rsqrtf(1.f+in.taps);
            }
            const float factor = prefillScale(a, value ? 0.f : in.qClip, 128, in.eps, false);
            half *cache = static_cast<half *>(value ? v.pages.value : v.pages.key) + v.pages.offset(p,value,stride);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = g*kHeadDim+lane+32*j;
                hist[c] = h1[j]; hist[stride+c] = h2[j]; hist[2*stride+c] = x[j];
                (value ? fs.vw : fs.kw)[lane+32*j] = cache[c] = half(a[j]*factor);
            }
        }
    }
    __syncthreads();
    cuda_launch::TriggerDependents();
    if (active) {
        if (valid && key == p) {   // a row of this window, as prepared above
            const uint4* kp = reinterpret_cast<const uint4*>(fs.kw);
            const uint4* vp = reinterpret_cast<const uint4*>(fs.vw);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
        }
        float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            m[h] = -INFINITY;
            l[h] = 0.f;
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
        }
        if (s_lo + w * kTile < s_hi) {
            float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                float f[8];
                unpack8_f16(kr[i], f);
#pragma unroll
                for (int d = 0; d < 8; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][8 * i + d], f[d], sc[h]);
            }
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                float b = 0.f;
#pragma unroll
                for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], tv[i], b);
                sc[h] = valid ? sc[h] * 0.08838834764831845f + b : -INFINITY;
            }
            uint32_t* vrow = sm.v[w][lane];

#pragma unroll
            for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr[i];
            float pr[kGroup];
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float mt = warp_max(sc[h]);
                const float mn = fmaxf(m[h], mt);
                const float corr = m[h] == -INFINITY ? 0.f : __expf(m[h] - mn);
                pr[h] = sc[h] == -INFINITY ? 0.f : __expf(sc[h] - mn);
                l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
                for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
                m[h] = mn;
            }
            __syncwarp();
            for (int j = 0; j < kTile; ++j) {
                const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][j][lane * 2]);
                const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                    acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                    acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                    acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                    acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
                }
            }
            __syncwarp();
        }
        // the four warps' parts of this slice, merged in warp order
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            if (lane == 0) {
                sm.m[w][h] = m[h];
                sm.l[w][h] = l[h];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) sm.acc[w][h][lane * 4 + u] = acc[h][u];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x) {
            const int h = i / kHeadDim, d = i % kHeadDim;
            float mm = -INFINITY;
            for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[ww][h]);
            float ll = 0.f, aa = 0.f;
            for (int ww = 0; ww < kWarps; ++ww) {
                if (sm.m[ww][h] == -INFINITY) continue;
                const float f = __expf(sm.m[ww][h] - mm);
                ll = fmaf(sm.l[ww][h], f, ll);
                aa = fmaf(sm.acc[ww][h][d], f, aa);
            }
            fs.pacc[h][d] = aa;
            if (d == 0) {
                fs.pm[h] = mm;
                fs.pl[h] = ll;
            }
        }
    }
    // the slices' shared memory to the merging blocks: a release and an acquire of shared memory only (a cluster-wide
    // release of every state space is a GPU-scope fence that waits for the cache and ring stores; no block of the cluster
    // reads those)
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n\tbarrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.aligned;\n\tfence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
    if (s < kGroup) {   // Merge the slices for head g * 3 + s, then apply the gate.
        const int h = s, d = threadIdx.x;
        cg::cluster_group cluster = cg::this_cluster();
        // every slice's maximum, sum and value read at once, then merged in slice order
        float pm[kClusterSplits], pl[kClusterSplits], pv[kClusterSplits];
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const FusedSmem* part = cluster.map_shared_rank(&fs, q);
                pm[q] = part->pm[h];
                pl[q] = part->pl[h];
                pv[q] = part->pacc[h][d];
            }
        float mm = sink;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) mm = fmaxf(mm, pm[q]);
        float ll = __expf(sink - mm), aa = 0.f;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const float f = __expf(pm[q] - mm);
                ll = fmaf(pl[q], f, ll);
                aa = fmaf(pv[q], f, aa);
            }
        float o = __fdividef(aa, ll);
        const int c = (g * kGroup + h) * kHeadDim + d;
        o *= 1.f / (1.f + __expf(-gate));
        if constexpr (std::is_same_v<Out, __nv_bfloat16>) {
            out[c] = __float2bfloat16_rn(o);
        } else {
            const float amax = warp_max(fabsf(o));
            const float sum = warp_sum(o);
            const float dq = fminf(65504.f, amax/127.f);
            auto &block = out[c/32];
            block.qs[c%32] = amax==0.f ? 0 : int(fmaxf(-127.f,fminf(127.f,roundf(o/dq))));
            if (!(c%32)) block.ds = __floats2half2_rn(dq,fmaxf(-65504.f,fminf(65504.f,sum)));
        }
    }
    // no block leaves while the merging blocks still read its slice: an arrival without release semantics (a release
    // here is a GPU-scope fence that waits for the output stores to be acknowledged); the merging blocks' remote loads
    // are complete, their values are in the stores above
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
#endif
}

#if CUDART_VERSION >= 12080
inline cudaLaunchConfig_t clusterConfig(int heads, size_t smem, cudaLaunchAttribute &attr) {
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, kClusterSplits};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1, heads, kClusterSplits);
    cfg.blockDim = 128;
    cfg.dynamicSmemBytes = smem;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cfg.stream = cudaStreamPerThread;
    return cfg;
}
#endif

template<class Out> inline size_t configure(int device, int heads) {
#if CUDART_VERSION >= 12080
    cudaFuncAttributes attr{};
    int arch = 0, perSm = 0, reserved = 0, resident = 0;
    if (cudaDeviceGetAttribute(&arch, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && arch == 12 &&
        cudaFuncGetAttributes(&attr, attn_fused_kernel<Out>) == cudaSuccess && attr.binaryVersion >= 120 && attr.ptxVersion >= 120 &&
        cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device) == cudaSuccess) {
        // Basalt's singleton schedule spreads the eight CTAs over separate SMs.
        const size_t smem = std::max(sizeof(FusedSmem), size_t(std::max(0, perSm / 2 - reserved + 128)));
        cudaLaunchAttribute cluster{};
        auto cfg = clusterConfig(heads, smem, cluster);
        if (cudaFuncSetAttribute(attn_fused_kernel<Out>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)) == cudaSuccess &&
            cudaOccupancyMaxActiveClusters(&resident, attn_fused_kernel<Out>, &cfg) == cudaSuccess && resident > 0) return smem;
    }
    cudaGetLastError();
#endif
    return 0;
}

template<class Out> inline void launch(const DecodeVisit *visit, WindowInput in, int window, Out *out, size_t smem) {
#if CUDART_VERSION >= 12080
    cudaLaunchAttribute attrs[2]{};
    auto cfg = clusterConfig(in.kvHeads, smem, attrs[0]);
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.numAttrs = 2;
    check(cudaLaunchKernelEx(&cfg, attn_fused_kernel<Out>, visit, in, window, out));
#else
    throw std::logic_error("Unavailable Inkling cluster attention");
#endif
}
} // namespace basalt_window
