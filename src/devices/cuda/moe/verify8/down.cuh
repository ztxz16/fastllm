#pragma once
// General down: each 160-thread group computes one (token, 64-row tile) output
// from that token's own eight selected experts (per-token weight streams).
namespace down_per_token {
using namespace moe_down;
struct PreparedTp {
    unsigned long long *send[kPeersPerWarp];
    unsigned generation;
};
// Resolve send addresses before the down loop so the TP tail only stores.
__device__ __forceinline__ PreparedTp prepare_tp(const Args &args, int token, int tile, int tid) {
    const int peer_group = tid / 32, lane = tid & 31, rank = args.rank;
    PreparedTp tp{};
    tp.generation = generation(args);
#pragma unroll
    for (int i = 0; i < kPeersPerWarp; ++i)
        tp.send[i] = mailbox(args, tp.generation, peer_group * kPeersPerWarp + i, rank, token, tile, lane);
    asm volatile("" ::"l"(tp.send[0]), "l"(tp.send[1]), "r"(tp.generation) : "memory");
    return tp;
}
__device__ __forceinline__ void finish_tp(unsigned char *s, int tid, const PreparedTp &tp) {
    tid &= 31;
    const auto *weighted = reinterpret_cast<const float *>(s + kWeightedOffset);
    const float2 even = *reinterpret_cast<const float2 *>(weighted + tid * 2);
    const float2 odd = *reinterpret_cast<const float2 *>(weighted + 64 + tid * 2);
    const __nv_bfloat162 pair = __floats2bfloat162_rn(__fadd_rn(even.x, odd.x), __fadd_rn(even.y, odd.y));
    const unsigned bits = *reinterpret_cast<const unsigned *>(&pair);
    const auto tagged = naivert::tagged_word(tp.generation, bits);
#pragma unroll
    for (int i = 0; i < kPeersPerWarp; ++i)
        naivert::store_tagged(tp.send[i], tagged);
}

__device__ __forceinline__ void stage_weight_half(const Args &args, unsigned char *s, int token, int tile,
                                                  const int *route_ids, int pair, int kg, int physical_slot = -1) {
    const int slot = physical_slot < 0 ? kg : physical_slot;
    if ((threadIdx.x & 31) == 0) {
        B::arrive_and_expect_tx(full(s) + slot, 2 * kHalfTxBytes);
#pragma unroll
        for (int ring = 0; ring < 2; ++ring) {
            const int expert = route_ids[token * 8 + pair * 2 + ring];
            tma_fp4_half(&args.downMap, s + kPerTokenWeightOffset + slot * 16384 + ring * 8192, expert, tile, kg,
                         full(s) + slot);
        }
    }
}
__device__ void produce_and_issue(const Args &args, unsigned char *s, unsigned tmem, int token, int tile,
                                  const int *route_ids) {
    auto *done = reinterpret_cast<std::uint64_t *>(s + kDoneOffset);
    auto *first_done = reinterpret_cast<std::uint64_t *>(s + kPerTokenFirstHalfDoneOffset);
    auto *activation_ready = common_ready(s);
    auto *tmem_free = reinterpret_cast<std::uint64_t *>(s + kTmemFreeOffset);
    using E = cutlass::float_e4m3_t;
    using Atom = cute::UMMA::Layout_K_SW128_Atom<E>;
#if CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 1
    using Mma = cute::SM100_MMA_F8F6F4_SS;
#else
    using Mma = cute::SM100_MMA_F8F6F4_SS<E, E, float, 128, 8, cute::UMMA::Major::K, cute::UMMA::Major::K>;
#endif
    const auto al = cute::tile_to_shape(Atom{}, cute::Shape<cute::Int<128>, cute::Int<128>>{});
    const auto bl = cute::tile_to_shape(Atom{}, cute::Shape<cute::Int<8>, cute::Int<128>>{});
    auto id = cute::UMMA::make_instr_desc<E, E, float, 128, 8, cute::UMMA::Major::K, cute::UMMA::Major::K>();
    id.a_format_ = kE2m1Format;
    const auto desc = cute::UMMA::make_runtime_instr_desc<>(id);

    stage_weight_half(args, s, token, tile, route_ids, 0, 0);
    stage_weight_half(args, s, token, tile, route_ids, 0, 1);
#pragma unroll 1
    for (int pair = 0; pair < 4; ++pair) {
        B::wait(activation_ready + pair, 0);
        B::wait(tmem_free + (pair & 1), unsigned(pair / 2) & 1);
        auto *b_image = common_image(s, pair);
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
            B::wait(full(s) + kg, unsigned(pair) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            const auto a = cute::make_tensor(
                cute::make_smem_ptr(reinterpret_cast<E *>(s + kPerTokenWeightOffset + kg * 16384)), al);
            const auto b = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<E *>(b_image + kg * 1024)), bl);
            const auto da = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(a);
            const auto db = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(b);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                auto dak = da, dbk = db;
                dak.start_address_ = static_cast<unsigned short>(da.start_address_ + kk * 2);
                dbk.start_address_ = static_cast<unsigned short>(db.start_address_ + kk * 2);
                // FP4: each K32 into its own accumulator (its rows' scales differ).
                Mma::fma(static_cast<unsigned long long>(dak), static_cast<unsigned long long>(dbk),
                         tmem + (pair & 1) * kAccPairStride + kg * kAccHalfStride + kk * kAccK32Stride, 0u, desc);
            }
            if (kg == 0 && pair < 3) {
                cutlass::arch::umma_arrive(first_done);
                B::wait(first_done, unsigned(pair) & 1);
                stage_weight_half(args, s, token, tile, route_ids, pair + 1, 0);
            }
        }
        cutlass::arch::umma_arrive(done + (pair & 1));
        if (pair < 3) {
            B::wait(done + (pair & 1), unsigned(pair / 2) & 1);
            stage_weight_half(args, s, token, tile, route_ids, pair + 1, 1);
        }
    }
}

__device__ __forceinline__ void prepare_pair(const Args &args, unsigned char *s, int token, int pair, unsigned tag,
                                             int warp, int lane) {
    const int slot = pair * 2 + warp / 2;
    const int kg = warp & 1;
    auto *b_image = common_image(s, pair);
    auto *scales = common_scales(s);

    const auto *input = args.tagged_up + (token * 8 + slot) * 256 + kg * 128 + lane * 4;
    unsigned scale;
    const unsigned packed = quantize_tagged_k32(input, tag, scale);
    *reinterpret_cast<unsigned *>(b_image + kg * 1024 + sw128(slot, lane * 4)) = packed;
    if (lane == 0)
        scales[slot * 2 + kg] = scale;
    // MMA reads all eight columns, including unused ones. Each pair owns a
    // separate complete image; zero the other six columns with disjoint writers.
#pragma unroll
    for (int other = warp / 2; other < 8; other += 2)
        if (other != slot)
            *reinterpret_cast<unsigned *>(b_image + kg * 1024 + sw128(other, lane * 4)) = 0u;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    consumer_sync();
    B::arrive(common_ready(s) + pair);
}

// Initialization is ordered by the entry's existing full-CTA join in the fused path.
__device__ __forceinline__ void initialize(unsigned char *s, int tid) {
    auto *done = reinterpret_cast<std::uint64_t *>(s + kDoneOffset);
    if (tid < kDownRingSlots)
        B::init(full(s) + tid, 1);
    auto *tmem_free = reinterpret_cast<std::uint64_t *>(s + kTmemFreeOffset);
    if (tid < 2) {
        B::init(done + tid, 1);
        B::init(tmem_free + tid, 128);
        naivert::barrier_arrive_count(tmem_free + tid, 128);
    }
    if (tid == 0)
        B::init(reinterpret_cast<std::uint64_t *>(s + kPerTokenFirstHalfDoneOffset), 1);
    // Shared readiness was initialized before the entry CTA join.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ void run(const Args &args, unsigned char *s, unsigned tmem, int logical_tile, const int *route_ids,
                    unsigned tag) {
    if (logical_tile >= 512)
        return;
    const int tid = group_tid(), lane = tid & 31, warp = tid / 32;
    const int token = logical_tile / 64, tile = logical_tile % 64;
    auto *done = reinterpret_cast<std::uint64_t *>(s + kDoneOffset);
    auto *tmem_free = reinterpret_cast<std::uint64_t *>(s + kTmemFreeOffset);
    if (tid >= 128) {
        produce_and_issue(args, s, tmem, token, tile, route_ids);
        return;
    }
    auto *scales = common_scales(s);
    // Overlap read-only row-scale fetches with activation preparation.
    const int scale_partition = (int(threadIdx.x) / 32) & 3;
    const int scale_row = tile * 64 + (scale_partition & 1) * 32 + lane;
    Fp4RowScales scale0 = fp4_row_scales(args, route_ids[token * 8 + scale_partition / 2], scale_row);
    fp4_pin(scale0);
    const Fp4RowScales scale1 = fp4_row_scales(args, route_ids[token * 8 + 2 + scale_partition / 2], scale_row);
    const Fp4RowScales scale2 = fp4_row_scales(args, route_ids[token * 8 + 4 + scale_partition / 2], scale_row);
    const Fp4RowScales scale3 = fp4_row_scales(args, route_ids[token * 8 + 6 + scale_partition / 2], scale_row);
    fp4_pin(scale1);
    fp4_pin(scale2);
    fp4_pin(scale3);
    prepare_pair(args, s, token, 0, tag, warp, lane);
    prepare_pair(args, s, token, 1, tag, warp, lane);
    const auto tp = prepare_tp(args, token, tile, tid);
    const int partition = (int(threadIdx.x) / 32) & 3;
    float weighted_accumulator = 0.f;
#pragma unroll 1
    for (int base = 0; base < 8; base += 2) {
        if (base > 0 && base + 2 < 8)
            prepare_pair(args, s, token, base / 2 + 1, tag, warp, lane);
        const int pass = partition / 2, slot = base + pass;

        const int pair = base / 2;
        float value = 0.f;
        // This thread's TMEM lane is row (partition & 1) * 32 + lane of the slot's expert tile.
        const auto row_scales = base == 0 ? scale0 : base == 2 ? scale1 : base == 4 ? scale2 : scale3;
        const unsigned as0 = scales[slot * 2], as1 = scales[slot * 2 + 1];
        fp4_pin(row_scales);
        asm volatile("" ::"r"(as0), "r"(as1) : "memory");
        B::wait(done + (pair & 1), unsigned(pair / 2) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // One wait per K128 half (one wait for all eight accumulators measured +2.7 us/round).
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
            unsigned bits[4];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
                cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem + (pair & 1) * kAccPairStride + kg * kAccHalfStride +
                                                          kk * kAccK32Stride + slot + (unsigned(partition * 32) << 16),
                                                      bits[kk]);
            cutlass::arch::fence_view_async_tmem_load();
            float half = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
                half =
                    __fmaf_rn(__uint_as_float(bits[kk]),
                              fp4_scale(row_scales, kg * 4 + kk) * fastllm_rows8_k32::scale(kg ? as1 : as0, kk), half);
            value = __fadd_rn(value, half);
        }

        reinterpret_cast<float *>(s + kPartialOffset)[slot * 64 + (partition & 1) * 32 + lane] = value;
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        B::arrive(tmem_free + (pair & 1));
    }
    consumer_sync();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int original_slot = 2 * i + partition / 2;
        weighted_accumulator = __fadd_rn(
            weighted_accumulator,
            reinterpret_cast<const float *>(s + kPartialOffset)[original_slot * 64 + (partition & 1) * 32 + lane]);
    }
    reinterpret_cast<float *>(s + kWeightedOffset)[partition * 32 + lane] = weighted_accumulator;
    consumer_sync();
    finish_tp(s, tid, tp);
}
} // namespace down_per_token
