// Eight-CTA cluster selector adapted from NaiveRT attn_dsa/cluster_select.cuh.
// Exact/coarse bin transforms follow SGLang deepseek_v4/topk_impl.cuh.
namespace cluster_sel {
using Scratch = fastllm_dsa_fused::AttnDsaScratchLayout;
namespace topk {
__device__ __forceinline__ uint32_t extract_exact_bin(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ constexpr float padding_value() { return std::numeric_limits<float>::quiet_NaN(); }

__device__ constexpr float infinity_value() { return std::numeric_limits<float>::infinity(); }

template <uint32_t kBits> __device__ __forceinline__ uint32_t extract_coarse_bin(float x) {
    static_assert(0 < kBits && kBits < 15);
    uint32_t b = (uint32_t)__half_as_ushort(__float2half_rn(x)) << 16;
    uint32_t s = (uint32_t)((int32_t)b >> 31);
    return (b ^ (s | 0x80000000u)) >> (32 - kBits);
}

__device__ __forceinline__ uint16_t coarse_bin_to_bits_finite(uint32_t bin) {
    const uint16_t ob = static_cast<uint16_t>(bin);
    return (ob & 0x8000) ? static_cast<uint16_t>(ob ^ 0x8000) : static_cast<uint16_t>(~ob);
}

// Smallest fp32 `v` for which `extract_coarse_bin<kBits>(v) >= bin`, i.e. the
// lower fp32 boundary of coarse bin `bin`. The collect pass classifies with two
// comparisons against these instead of recomputing the fp16 bin per element, so
// this must agree with `extract_coarse_bin` on every value -- a score sitting
// exactly on a boundary included. Two pairs no fp32 threshold can separate are
// left: -0.0 at the zero bin, and +inf at a NaN-key bin.
template <uint32_t kBits> __device__ __forceinline__ float coarse_bin_lower_bound(uint32_t bin) {
    constexpr uint32_t kShift = 16 - kBits;
    constexpr uint32_t kInfBin = 0xFC00u >> kShift; // bin holding the +inf key
    const uint32_t key = bin << kShift;             // ordered16 key at the low edge
    // ordered16 -> fp16 value (inverse of the transform in extract_coarse_bin);
    constexpr auto to_finite_val = [](uint32_t okey) -> float {
        const uint16_t hb = coarse_bin_to_bits_finite(okey);
        return __half2float(__ushort_as_half(hb));
    };
    constexpr auto step_up = [](float v) -> float {
        const int32_t b = __float_as_int(v);
        return __int_as_float(b >= 0 ? b + 1 : b - 1);
    };
    // Fast path, hoisted above the per-key special cases so both keys are
    // range-checked at once: `key` and `key - 1` both land in the finite band
    // [0x0401, 0xFBFF] -- every boundary a finite-score threshold produces. fp16
    // rounds to nearest, so the boundary is the midpoint between the fp16 values
    // at `key` and `key - 1`.
    if (key - 0x0401u <= 0xFBFFu - 0x0401u) {
        const float mid = 0.5f * (to_finite_val(key) + to_finite_val(key - 1));
        // fp32 -> fp16 rounds to nearest EVEN, so on the ~half of bins whose fp16
        // value has an odd significand the midpoint still bins as `bin - 1`.
        return (coarse_bin_to_bits_finite(key) & 1u) ? step_up(mid) : mid;
    }
    // Slow path: an edge of `bin` touches the +/-inf keys or NaN key space. The
    // ordered-key line is: [0, 0x03FF) negative-NaN space, 0x03FF = -inf,
    // [0x0400, 0xFC00) finite, 0xFC00 = +inf, (0xFC00, 0xFFFF] positive-NaN
    // space. The +/-inf keys stand in as +/-65536, one ideal step past fp16 max,
    // so the midpoint lands on the +/-65520 fp32 -> fp16 overflow threshold.
    if (bin == 0)
        return -infinity_value(); // every value bins at >= 0
    if (bin > kInfBin)
        return infinity_value(); // NaN key space: nothing bins that high
    const auto to_val = [&](uint32_t okey) -> float {
        if (okey < 0x03FFu)
            return -infinity_value();
        if (okey == 0x03FFu)
            return -65536.0f;
        if (okey == 0xFC00u)
            return 65536.0f;
        return to_finite_val(okey);
    };
    // The +/-65536 stand-ins are not real fp16 neighbours, so the parity rule
    // does not apply here; test the property directly instead.
    const float mid = 0.5f * (to_val(key) + to_val(key - 1));
    return extract_coarse_bin<kBits>(mid) < bin ? step_up(mid) : mid;
}

} // namespace topk
constexpr unsigned kRanks = 8, kTopK = 2048, kThreads = 1024, kWarps = kThreads / 32;
constexpr unsigned kBins = 4096, kChunk = 32, kChunks = kBins / kChunk; // chunk c = bins [32c, 32c + 32)
constexpr unsigned kKeep = 4;                                           // blocks per warp held between the two passes
constexpr unsigned kRegion = 512;                                       // candidates per source rank
constexpr unsigned kWords = Scratch::kMaxSeqLen / (32u * kRanks), kBlocks = kWords / 4u;
constexpr unsigned kCutBins = 1024, kCutShift = 8; // the tie cut's radix bins: relative key bits 17..8
constexpr unsigned kWriteBlocks = 8; // up to 32 x this many blocks, warp w writes the w-th 32nd a bit per lane
static_assert(kChunks == 4u * kWarps);

__device__ __forceinline__ unsigned smem_u32(const void *p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ unsigned remote(const void *p, unsigned rank) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));
    return r;
}
__device__ __forceinline__ void mbar_init(unsigned long long *b, unsigned arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(arrivals));
}
// This CTA's arrival on its own barrier, adding `bytes` to the transaction count (they may have landed
// already: the count goes negative, and the phase waits for this arrival).
__device__ __forceinline__ void expect_tx(unsigned long long *b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long *b) { // phase 0 (one use per launch)
    unsigned done;
    do {
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\nselp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done)
                     : "r"(smem_u32(b))
                     : "memory");
    } while (!done);
}
__device__ __forceinline__ void st_async(unsigned addr, uint4 v, unsigned bar) {
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(bar)
        : "memory");
}
// The sum of v over the lanes in mask, by bit planes (one ballot each): exact for v < 2^kBits.
template <unsigned kBits> __device__ __forceinline__ unsigned lanes_sum(unsigned v, unsigned mask) {
    unsigned s = 0u;
#pragma unroll
    for (unsigned k = 0; k < kBits; ++k)
        s += unsigned(__popc(__ballot_sync(0xffffffffu, (v >> k) & 1u) & mask)) << k;
    return s;
}
// The sum of min(v, 4095) over the lanes in mask: exact for deciding whether a cut at <= 2048 keys
// lies above or within these lanes.
__device__ __forceinline__ unsigned lanes_sum_cut(unsigned v, unsigned mask) {
    return lanes_sum<12>(min(v, 4095u), mask);
}

struct Select {
    struct Shared {
        unsigned long long part_bar;             // every rank's chunk totals
        unsigned long long tbin_bar;             // every rank's bins of the threshold chunk
        unsigned long long cand_bar;             // the other ranks' candidates (the bytes known once the bin is)
        alignas(16) unsigned hist[kBins];        // this rank's histogram
        alignas(16) unsigned bitmap[kWords];     // this rank's winners (every block's words written)
        alignas(16) unsigned bcount[kBlocks];    // per block: its winners
        alignas(16) unsigned cut_hist[kCutBins]; // the tie cut's radix bins
        alignas(16) unsigned cut_coarse[kCutBins / 32u]; // and their sums by 32
        alignas(16) unsigned part[kRanks][kChunks];      // per source rank: its chunk totals
        alignas(16) unsigned tbins[kRanks][kChunk];      // per source rank: its bins of the threshold chunk
        alignas(16) unsigned bin[4];             // the threshold bin, the keys over it, its width flag, ordered(lowest)
        alignas(16) unsigned layout[4];          // candidates kept, lower ranks' keys over the bin, flag, exchange
        alignas(16) unsigned first[kRanks];      // per source rank: its candidates' first flat index
        alignas(16) unsigned kept[kRanks];       // per source rank: its candidates kept (at most kRegion)
        alignas(16) uint2 cand[kRanks][kRegion]; // per source rank: {ordered key, token}
        unsigned warp_sum[kWarps];
        unsigned won[kRanks];                         // per source rank: candidates that win
        unsigned eq;                                  // this rank's candidates
        unsigned match[3];                            // the tie cut: bin, keys over it, keys in it
        unsigned members;                             // the cut bin's keys in member
        alignas(16) uint2 member[kRanks * kRegion];   // the cut bin's {key - the bin's lowest, token}
        alignas(16) float4 stage[kKeep * kWarps][32]; // the kept blocks, per lane, for the classify pass
    };

    static __device__ __forceinline__ unsigned ordered(float v) { return topk::extract_exact_bin(v); }
    // extract_coarse_bin<12> of a and b from one packed conversion (both passes use it).
    static __device__ __forceinline__ uint2 coarse2(float a, float b) {
        const __half2 h = __floats2half2_rn(a, b);
        const unsigned bits = *reinterpret_cast<const unsigned *>(&h);
        const unsigned x = bits << 16, y = bits & 0xffff0000u;
        return make_uint2((x ^ (unsigned(int(x) >> 31) | 0x80000000u)) >> 20,
                          (y ^ (unsigned(int(y) >> 31) | 0x80000000u)) >> 20);
    }
    static __device__ __forceinline__ float value_of(unsigned key) { // inverse of ordered()
        return __uint_as_float((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
    }

    static __device__ void run(const float *scores, int *ids, unsigned *flags, const int *lengths, Shared &s) {
        const unsigned q = blockIdx.x, rank = blockIdx.y, tx = threadIdx.x, lane = tx & 31u, warp = tx >> 5;
        const unsigned visible = unsigned(lengths[q]);

        int *out = ids + q * kTopK;
        if (visible <= kTopK) {
            if (rank == 0) {
                if (tx == 0)
                    flags[q] = 0u;
#pragma unroll 1
                for (unsigned i = tx; i < kTopK; i += kThreads)
                    out[i] = visible ? int(i) : -1;
            }
            return;
        }
        const unsigned above_mask = ~((2u << lane) - 1u), below_mask = (1u << lane) - 1u;
        {
            // bar-0, waited before the first remote write: every rank's barriers (all it publishes; the rest is
            // sent with st.async), so it is arrived relaxed, first thing.
            if (tx == kThreads - 32u) {
                mbar_init(&s.part_bar, 1u);
                mbar_init(&s.tbin_bar, 1u);
                mbar_init(&s.cand_bar, 1u);
                expect_tx(&s.part_bar, unsigned(sizeof(s.part)));
                expect_tx(&s.tbin_bar, unsigned(sizeof(s.tbins)));
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
            __syncwarp();
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        }

        if (tx < kRanks)
            s.won[tx] = 0u;
        if (tx == 0)
            s.eq = s.members = 0u;
        reinterpret_cast<uint4 *>(s.hist)[tx] = make_uint4(0u, 0u, 0u, 0u);
        if (tx < kCutBins / 4u)
            reinterpret_cast<uint4 *>(s.cut_hist)[tx] = make_uint4(0u, 0u, 0u, 0u);
        if (tx < kCutBins / 128u)
            reinterpret_cast<uint4 *>(s.cut_coarse)[tx] = make_uint4(0u, 0u, 0u, 0u);
        // span is a multiple of 32 below visible / 7, so every rank's range is word aligned and nonempty.
        const unsigned span = (visible + 32u * kRanks - 1u) / (32u * kRanks) * 32u;
        const unsigned lo = rank * span, len = min(span, visible - lo), words = (len + 31u) / 32u,
                       blocks = (len + 127u) / 128u;
        const float *row = scores + std::size_t(q) * Scratch::kMaxSeqLen + lo;
        // Lane l's scores of block g: tokens 128 g + 32 j + l (j < 4); -inf past the range.
        const float kNegInf = __uint_as_float(0xff800000u);
        auto load = [&](unsigned g) {
            const unsigned t = 128u * g + lane;
            float4 x;
            if (128u * g + 128u <= len)
                return make_float4(row[t], row[t + 32u], row[t + 64u], row[t + 96u]);
            x.x = t < len ? row[t] : kNegInf;
            x.y = t + 32u < len ? row[t + 32u] : kNegInf;
            x.z = t + 64u < len ? row[t + 64u] : kNegInf;
            x.w = t + 96u < len ? row[t + 96u] : kNegInf;
            return x;
        };
        float4 kept[kKeep];
#pragma unroll
        for (unsigned i = 0; i < kKeep; ++i)
            if (warp + 32u * i < blocks)
                kept[i] = load(warp + 32u * i);
        __syncthreads();

        // Every block g of this rank, warp uniform: the kept blocks from registers in the histogram pass (which
        // stages them, each lane its own) and from the stage in the classify pass's rolled loop, the rest
        // streamed through a four-deep queue in both.
        const unsigned held = blocks > warp ? min((blocks - warp + 31u) / 32u, kKeep) : 0u;
        auto stream = [&](auto &&fn) {
            unsigned g = warp + 32u * kKeep;
            if (g >= blocks)
                return;
            float4 x0 = load(g), x1, x2, x3;
            if (g + 32u < blocks)
                x1 = load(g + 32u);
            if (g + 64u < blocks)
                x2 = load(g + 64u);
            if (g + 96u < blocks)
                x3 = load(g + 96u);
#pragma unroll 1
            for (; g < blocks; g += 32u) {
                fn(x0, g);
                x0 = x1;
                x1 = x2;
                x2 = x3;
                if (g + 128u < blocks)
                    x3 = load(g + 128u);
            }
        };
        auto tally = [&](const float4 &x, unsigned g) {
            const unsigned t = 128u * g + lane;
            const uint2 b01 = coarse2(x.x, x.y), b23 = coarse2(x.z, x.w);
            if (t < len)
                atomicAdd(&s.hist[b01.x], 1u);
            if (t + 32u < len)
                atomicAdd(&s.hist[b01.y], 1u);
            if (t + 64u < len)
                atomicAdd(&s.hist[b23.x], 1u);
            if (t + 96u < len)
                atomicAdd(&s.hist[b23.y], 1u);
        };
#pragma unroll
        for (unsigned i = 0; i < kKeep; ++i)
            if (i < held) {
                tally(kept[i], warp + 32u * i);
                s.stage[warp + 32u * i][lane] = kept[i];
            }
        stream(tally);
        __syncthreads();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); // bar-0

        // Warp w's chunks 4w..4w+3 (bins [128 w, 128 w + 128)) to every rank.
        {
            unsigned t[4];
#pragma unroll
            for (unsigned k = 0; k < 4u; ++k)
                t[k] = __reduce_add_sync(0xffffffffu, s.hist[128u * warp + 32u * k + lane]);
            {
                if (lane < kRanks)
                    st_async(remote(&s.part[rank][4u * warp], lane), make_uint4(t[0], t[1], t[2], t[3]),
                             remote(&s.part_bar, lane));
            }
        }

        // The threshold chunk and bin (every rank), by warp 31 while the others wait at barrier 1.
        if (warp == kWarps - 1u) {
            mbar_wait(&s.part_bar);
            // Lane l: chunks 4l..4l+3, over every rank and over the lower ranks.
            unsigned v[4] = {0u, 0u, 0u, 0u}, low[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (unsigned r = 0; r < kRanks; ++r) {
                const uint4 p = *reinterpret_cast<const uint4 *>(&s.part[r][4u * lane]);
                v[0] += p.x, v[1] += p.y, v[2] += p.z, v[3] += p.w;
                if (r < rank)
                    low[0] += p.x, low[1] += p.y, low[2] += p.z, low[3] += p.w;
            }
            unsigned above = lanes_sum_cut(v[0] + v[1] + v[2] + v[3], above_mask), chunk = ~0u, chunk_above = 0u;
#pragma unroll
            for (int j = 3; j >= 0; --j) {
                if (above < kTopK && above + v[j] >= kTopK) {
                    chunk = 4u * lane + unsigned(j);
                    chunk_above = above;
                }
                above += v[j];
            }
            const unsigned src = __ffs(__ballot_sync(0xffffffffu, chunk != ~0u)) - 1u;
            chunk = __shfl_sync(0xffffffffu, chunk, int(src));
            chunk_above = __shfl_sync(0xffffffffu, chunk_above, int(src));
            {
                // This rank's bins of it to every rank: lane l bins 4 (l & 7) .. + 3 to ranks l / 8 and l / 8 + 4.
                const unsigned mine = s.hist[kChunk * chunk + lane], j0 = 4u * (lane & 7u);
                const uint4 m = make_uint4(
                    __shfl_sync(0xffffffffu, mine, int(j0)), __shfl_sync(0xffffffffu, mine, int(j0 + 1u)),
                    __shfl_sync(0xffffffffu, mine, int(j0 + 2u)), __shfl_sync(0xffffffffu, mine, int(j0 + 3u)));
#pragma unroll
                for (unsigned h = 0; h < 2u; ++h) {
                    const unsigned dst = lane / 8u + 4u * h;
                    st_async(remote(&s.tbins[rank][j0], dst), m, remote(&s.tbin_bar, dst));
                }
            }
            unsigned low_over = 0u; // the lower ranks' keys in this lane's chunks over the threshold chunk
#pragma unroll
            for (unsigned j = 0; j < 4u; ++j)
                if (4u * lane + j > chunk)
                    low_over += low[j];
            low_over = __reduce_add_sync(0xffffffffu, low_over);
            mbar_wait(&s.tbin_bar);
            // Lane l: bin 32 chunk + l, over every rank (c) and the lower ranks, and per rank.
            unsigned c = 0u, c_low = 0u, cr[kRanks];
#pragma unroll
            for (unsigned r = 0; r < kRanks; ++r) {
                cr[r] = s.tbins[r][lane];
                c += cr[r];
                if (r < rank)
                    c_low += cr[r];
            }
            const unsigned b = kChunk * chunk + lane, over = chunk_above + lanes_sum_cut(c, above_mask);
            // A bin over 2^18 ordered keys is too wide for the radix round (zero / inf / NaN edges). Every
            // lane's bounds, overlapping the loads.
            const float bin_lo = topk::coarse_bin_lower_bound<12>(b), bin_hi = topk::coarse_bin_lower_bound<12>(b + 1u);
            const bool wide =
                !isfinite(bin_lo) || !isfinite(bin_hi) || ordered(bin_hi) - ordered(bin_lo) > (kCutBins << kCutShift);
            const unsigned at = __ffs(__ballot_sync(0xffffffffu, over < kTopK && over + c >= kTopK)) - 1u;
            if (lane == at)
                *reinterpret_cast<uint4 *>(s.bin) = make_uint4(b, over, wide ? 1u : 0u, ordered(bin_lo));
            asm volatile("bar.arrive 1, %0;" ::"n"(kThreads) : "memory");
            // The candidates' layout, for after the classify pass: lane r < 8 rank r's keys in the bin.
            const unsigned below = low_over + __reduce_add_sync(0xffffffffu, lane > at ? c_low : 0u);
            unsigned eq = 0u;
#pragma unroll
            for (unsigned r = 0; r < kRanks; ++r) {
                const unsigned e = __shfl_sync(0xffffffffu, cr[r], int(at));
                if (lane == r)
                    eq = e;
            }
            const unsigned kept_l = min(eq, kRegion), total = __reduce_add_sync(0xffffffffu, kept_l);
            const unsigned wide_at = (__ballot_sync(0xffffffffu, wide) >> at) & 1u;
            const unsigned flag = __any_sync(0xffffffffu, eq > kRegion) || wide_at != 0u ? 1u : 0u;
            const unsigned remain = kTopK - __shfl_sync(0xffffffffu, over, int(at));
            const unsigned exchange = flag == 0u && total != remain ? 1u : 0u;
            const unsigned first_l = lanes_sum<10>(kept_l, below_mask);
            const unsigned bytes =
                __reduce_add_sync(0xffffffffu, lane < kRanks && lane != rank ? 16u * ((kept_l + 1u) / 2u) : 0u);
            if (lane < kRanks) {
                s.first[lane] = first_l;
                s.kept[lane] = kept_l;
            }
            if (lane == 0u) {
                *reinterpret_cast<uint4 *>(s.layout) = make_uint4(total, below, flag, exchange);
                {
                    if (exchange)
                        expect_tx(&s.cand_bar, bytes);
                }
            }
            __syncwarp();
        } else {
            asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
        }

        // Classify (every rank): winners into the bitmap, candidates into this rank's region. A block whose
        // scores are all below the threshold bin's lowest value is skipped (NaN-safe: NaN is never below; the
        // bound's two unseparable keys, -0.0 and +inf, compare equal or above it), the rest get the exact bins.
        const unsigned threshold = s.bin[0];
        const float floor_value = value_of(s.bin[3]); // coarse_bin_lower_bound(threshold)
        auto classify = [&](const float4 &x, unsigned g) {
            const unsigned t = 128u * g + lane;
            unsigned w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
            if (__any_sync(0xffffffffu,
                           !(x.x < floor_value && x.y < floor_value && x.z < floor_value && x.w < floor_value))) {
                const uint2 b01 = coarse2(x.x, x.y), b23 = coarse2(x.z, x.w);
                const bool v0 = t < len, v1 = t + 32u < len, v2 = t + 64u < len, v3 = t + 96u < len;
                w0 = __ballot_sync(0xffffffffu, v0 && b01.x > threshold);
                w1 = __ballot_sync(0xffffffffu, v1 && b01.y > threshold);
                w2 = __ballot_sync(0xffffffffu, v2 && b23.x > threshold);
                w3 = __ballot_sync(0xffffffffu, v3 && b23.y > threshold);
                const unsigned eq = (v0 && b01.x == threshold ? 1u : 0u) | (v1 && b01.y == threshold ? 2u : 0u) |
                                    (v2 && b23.x == threshold ? 4u : 0u) | (v3 && b23.y == threshold ? 8u : 0u);
                if (eq != 0u) { // into this rank's own region; sent to the other ranks after the pass
                    unsigned slot = atomicAdd(&s.eq, unsigned(__popc(eq)));
                    for (unsigned m = eq; m != 0u && slot < kRegion; m &= m - 1u, ++slot) {
                        const unsigned j = unsigned(__ffs(m)) - 1u;
                        const float e = j == 0u ? x.x : j == 1u ? x.y : j == 2u ? x.z : x.w;
                        s.cand[rank][slot] = make_uint2(ordered(e), lo + t + 32u * j);
                    }
                }
            }
            if (lane == 0u) {
                reinterpret_cast<uint4 *>(s.bitmap)[g] = make_uint4(w0, w1, w2, w3);
                s.bcount[g] = __popc(w0) + __popc(w1) + __popc(w2) + __popc(w3);
            }
        };
        if (held != 0u) {
            float4 x = s.stage[warp][lane];
#pragma unroll 1
            for (unsigned i = 0; i < held; ++i) {
                const float4 next = i + 1u < held ? s.stage[warp + 32u * (i + 1u)][lane] : x;
                classify(x, warp + 32u * i);
                x = next;
            }
        }
        stream(classify);
        __syncthreads();
        const uint4 layout = *reinterpret_cast<const uint4 *>(s.layout);
        const unsigned total = layout.x, over = s.bin[1], remain = kTopK - over;
        if (layout.z != 0u) { // flagged (every rank): the fallback selects
            if (rank == 0 && tx == 0)
                flags[q] = 5u;
            return;
        }
        auto own = [&](unsigned token) { // a winning candidate of this rank: its bitmap bit
            const unsigned t = token - lo;
            atomicOr(&s.bitmap[t >> 5], 1u << (t & 31u));
            atomicAdd(&s.bcount[t >> 7], 1u);
        };
        if (layout.w == 0u) { // every candidate wins (every rank), none sent
            for (unsigned slot = tx; slot < s.kept[rank]; slot += kThreads)
                own(s.cand[rank][slot].y);
            if (tx < kRanks)
                s.won[tx] = s.kept[tx];
        } else {
            unsigned p[kRanks];
            {
                // This rank's candidates to the other ranks, two per message (thread i: pair i / 8 to rank i % 8).
                const unsigned pairs = (s.kept[rank] + 1u) / 2u;
                for (unsigned i = tx; i < kRanks * pairs; i += kThreads) {
                    const unsigned r = i % kRanks, slot = 2u * (i / kRanks);
                    if (r != rank)
                        st_async(remote(&s.cand[rank][slot], r), *reinterpret_cast<const uint4 *>(&s.cand[rank][slot]),
                                 remote(&s.cand_bar, r));
                }
                // Cut the candidates (every rank, all of them). Flat candidate i < total: region r (p[r] <= i <
                // p[r + 1]), slot i - p[r].
                const uint4 f0 = *reinterpret_cast<const uint4 *>(&s.first[0]),
                            f1 = *reinterpret_cast<const uint4 *>(&s.first[4]);
                p[0] = f0.x;
                p[1] = f0.y;
                p[2] = f0.z;
                p[3] = f0.w;
                p[4] = f1.x;
                p[5] = f1.y;
                p[6] = f1.z;
                p[7] = f1.w;
            }
            auto locate = [&](unsigned i, unsigned &r, unsigned &slot) {
                r = 0u;
                unsigned base = 0u;
#pragma unroll
                for (unsigned k = 1; k < kRanks; ++k)
                    if (i >= p[k]) {
                        r = k;
                        base = p[k];
                    }
                slot = i - base;
            };
            mbar_wait(&s.cand_bar);
            // One radix round of key - the bin's lowest key (under 2^18 of them: bits 17..8), warp 0's cut,
            // then the cut bin's keys ranked by (key, lowest token).
            const unsigned base = s.bin[3];
            auto bin_of = [&](unsigned key) { return min((key - base) >> kCutShift, kCutBins - 1u); };
#pragma unroll 1
            for (unsigned i = tx; i < total; i += kThreads) {
                unsigned r, slot;
                locate(i, r, slot);
                const unsigned b = bin_of(s.cand[r][slot].x);
                atomicAdd(&s.cut_hist[b], 1u);
                atomicAdd(&s.cut_coarse[b / 32u], 1u);
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned sum = s.cut_coarse[lane], above = lanes_sum_cut(sum, above_mask);
                const unsigned chunk = __ffs(__ballot_sync(0xffffffffu, above < remain && above + sum >= remain)) - 1u;
                const unsigned chunk_above = __shfl_sync(0xffffffffu, above, int(chunk));
                const unsigned c = s.cut_hist[32u * chunk + lane], over_c = chunk_above + lanes_sum_cut(c, above_mask);
                if (over_c < remain && over_c + c >= remain) {
                    s.match[0] = 32u * chunk + lane;
                    s.match[1] = over_c;
                    s.match[2] = c;
                }
            }
            __syncthreads();
            const unsigned cut = s.match[0], left = remain - s.match[1], in_cut = s.match[2];
#pragma unroll 1
            for (unsigned i0 = 32u * warp; i0 < total; i0 += kThreads) { // warp uniform
                const unsigned i = i0 + lane;
                const bool live = i < total;
                unsigned r = 0u, slot = 0u;
                if (live)
                    locate(i, r, slot);
                const uint2 c = live ? s.cand[r][slot] : make_uint2(base, 0u);
                const unsigned b = bin_of(c.x);
                const bool member = live && b == cut;
                const bool wins = live && (b > cut || (member && in_cut <= left));
                if (wins)
                    atomicAdd(&s.won[r], 1u);
                if (wins && r == rank)
                    own(c.y);
                if (member && in_cut > left)
                    s.member[atomicAdd(&s.members, 1u)] = make_uint2(c.x - base, c.y);
            }
            if (in_cut > left) {
                __syncthreads();
#pragma unroll 1
                for (unsigned i = tx; i < in_cut; i += kThreads) {
                    const uint2 me = s.member[i];
                    unsigned beat = 0u;
#pragma unroll 4
                    for (unsigned j = 0; j < in_cut; ++j) {
                        const uint2 o = s.member[j];
                        beat += (o.x > me.x || (o.x == me.x && o.y < me.y)) ? 1u : 0u;
                    }
                    if (beat < left) {
                        const unsigned owner = me.y / span;
                        atomicAdd(&s.won[owner], 1u);
                        if (owner == rank)
                            own(me.y);
                    }
                }
            }
        }
        __syncthreads();

        // Ascending write after the lower ranks' winners.
        const unsigned won = lane < kRanks ? s.won[lane] : 0u;
        const unsigned all = over + __reduce_add_sync(0xffffffffu, won);
        unsigned pos = layout.y + __reduce_add_sync(0xffffffffu, lane < rank ? won : 0u);
        if (rank == 0 && tx == 0)
            flags[q] = all != kTopK ? 4u : 0u;
        if (all != kTopK)
            return;
        if (blocks <= 32u * kWriteBlocks) {
            // Only the warps holding blocks, no barrier: warp w writes blocks [kb w, kb w + kb) a bit per lane
            // (coalesced), after the earlier warps' (lane l counts warp l's).
            const unsigned kb = (blocks + 31u) / 32u;
            if (kb * warp < blocks) {
                unsigned before = 0u;
#pragma unroll
                for (unsigned i = 0; i < kWriteBlocks; ++i)
                    if (const unsigned g = kb * lane + i; i < kb && g < blocks)
                        before += s.bcount[g];
                pos += __reduce_add_sync(0xffffffffu, lane < warp ? before : 0u);
                const unsigned end = min(kb * warp + kb, blocks);
                uint4 b = reinterpret_cast<const uint4 *>(s.bitmap)[kb * warp];
#pragma unroll 1
                for (unsigned g = kb * warp; g < end; ++g) {
                    const uint4 next = g + 1u < end ? reinterpret_cast<const uint4 *>(s.bitmap)[g + 1u] : b;
                    const unsigned t = lo + 128u * g + lane;
                    if ((b.x >> lane) & 1u)
                        out[pos + __popc(b.x & below_mask)] = int(t);
                    pos += __popc(b.x);
                    if ((b.y >> lane) & 1u)
                        out[pos + __popc(b.y & below_mask)] = int(t + 32u);
                    pos += __popc(b.y);
                    if ((b.z >> lane) & 1u)
                        out[pos + __popc(b.z & below_mask)] = int(t + 64u);
                    pos += __popc(b.z);
                    if ((b.w >> lane) & 1u)
                        out[pos + __popc(b.w & below_mask)] = int(t + 96u);
                    pos += __popc(b.w);
                    b = next;
                }
            }
        } else {
            const unsigned per = (words + kThreads - 1u) / kThreads, base_word = tx * per;
            unsigned w[4], count = 0u;
#pragma unroll
            for (unsigned j = 0; j < 4u; ++j) {
                w[j] = j < per && base_word + j < words ? s.bitmap[base_word + j] : 0u;
                count += __popc(w[j]);
            }
            const unsigned before = lanes_sum<8>(count, below_mask); // count <= 128
            if (lane == 31u)
                s.warp_sum[warp] = before + count;
            __syncthreads();
            pos += __reduce_add_sync(0xffffffffu, lane < warp ? s.warp_sum[lane] : 0u) + before;
#pragma unroll
            for (unsigned j = 0; j < 4u; ++j)
                for (unsigned b = w[j]; b != 0u; b &= b - 1u)
                    out[pos++] = int(lo + 32u * (base_word + j) + unsigned(__ffs(b) - 1));
        }
    }
};

// Recover this query on its own CTA. The cluster has finished its DSM traffic.
struct ExactRetry {
    unsigned bins[256], warpGreater[32], warpEqual[32];
    unsigned prefix, need, previousGreater, previousEqual;
};
__device__ void RetryQuery(const float *scores, int *ids, int n, ExactRetry &s) {
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    if (t == 0) {
        s.prefix = 0;
        s.need = kTopK;
        s.previousGreater = 0;
        s.previousEqual = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256)
            s.bins[t] = 0;
        __syncthreads();
        const unsigned prefix = s.prefix;
        const unsigned mask = shift == 24 ? 0 : (0xffffffffu << (shift + 8));
        for (int i = t; i < n; i += blockDim.x) {
            unsigned key = Select::ordered(scores[i]);
            if ((key & mask) == prefix)
                atomicAdd(s.bins + ((key >> shift) & 255), 1u);
        }
        __syncthreads();
        if (t == 0) {
            unsigned remain = s.need;
            for (int b = 255; b >= 0; --b) {
                if (remain <= s.bins[b]) {
                    s.need = remain;
                    s.prefix |= unsigned(b) << shift;
                    break;
                }
                remain -= s.bins[b];
            }
        }
        __syncthreads();
    }
    const unsigned cutoff = s.prefix, keep = s.need;
    for (int base = 0; base < n; base += blockDim.x) {
        int token = base + t;
        unsigned key = token < n ? Select::ordered(scores[token]) : 0;
        bool greater = token < n && key > cutoff, equal = token < n && key == cutoff;
        unsigned m1 = __ballot_sync(0xffffffff, greater), m2 = __ballot_sync(0xffffffff, equal);
        if (!lane) {
            s.warpGreater[warp] = __popc(m1);
            s.warpEqual[warp] = __popc(m2);
        }
        __syncthreads();
        if (!t) {
            unsigned g = s.previousGreater, e = s.previousEqual;
            for (int w = 0; w < 32; ++w) {
                unsigned ng = g + s.warpGreater[w], ne = e + s.warpEqual[w];
                s.warpGreater[w] = g;
                s.warpEqual[w] = e;
                g = ng;
                e = ne;
            }
            s.previousGreater = g;
            s.previousEqual = e;
        }
        __syncthreads();
        const unsigned prior = (1u << lane) - 1, eg = s.warpGreater[warp] + __popc(m1 & prior),
                       ee = s.warpEqual[warp] + __popc(m2 & prior);
        if (greater || (equal && ee < keep))
            ids[eg + min(ee, keep)] = token;
        __syncthreads();
    }
}

__global__ __launch_bounds__(1024, 1) void select(const float *scores, int *ids, unsigned *flags, const int *lengths) {
    extern __shared__ __align__(16) unsigned char buf[];
    Select::run(scores, ids, flags, lengths, *reinterpret_cast<Select::Shared *>(buf));
    cooperative_groups::this_cluster().sync();
    if (blockIdx.y == 0 && flags[blockIdx.x]) {
        const int q = blockIdx.x;
        RetryQuery(scores + size_t(q) * Scratch::kMaxSeqLen, ids + q * kTopK, lengths[q],
                   *reinterpret_cast<ExactRetry *>(buf));
    }
}

inline cudaLaunchConfig_t Config(cudaLaunchAttribute &attr, cudaStream_t stream) {
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, kRanks, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(Scratch::kQueryRows, kRanks);
    cfg.blockDim = kThreads;
    cfg.dynamicSmemBytes = sizeof(Select::Shared);
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cfg.stream = stream;
    return cfg;
}
inline bool Supported(int device) {
    cudaLaunchAttribute attr{};
    auto cfg = Config(attr, cudaStreamPerThread);
    int clusters = 0, active = 0;
    return cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, device) == cudaSuccess && clusters &&
           cudaFuncSetAttribute(select, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.dynamicSmemBytes) ==
               cudaSuccess &&
           cudaOccupancyMaxActiveClusters(&active, select, &cfg) == cudaSuccess && active > 0;
}
inline cudaError_t Launch(const float *scores, int *ids, unsigned *retry, const int *lengths, cudaStream_t stream) {
    cudaLaunchAttribute attr{};
    auto cfg = Config(attr, stream);
    return cudaLaunchKernelEx(&cfg, select, scores, ids, retry, lengths);
}
} // namespace cluster_sel
