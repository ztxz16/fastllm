#pragma once
#include "k32_activation.cuh"
namespace moe_down {
using B = naivert::Barrier;
constexpr int kPerTokenTmemGroupColumns = 128;
constexpr int kAccPairStride = 64, kAccHalfStride = 32, kAccK32Stride = 8;
constexpr int kHalfTxBytes = 4096;
constexpr unsigned kE2m1Format = 5; // UMMA MXF8F6F4Format E2M1 (cute/arch/mma_sm100_desc.hpp)
// Transaction bytes of one 64-row x 128-K half: TMA counts the packed MXFP4 bytes it reads, not the padded
// image it writes (CUTLASS sm100 blockscaled ABTmaTransactionBytes).
__device__ __forceinline__ void tma_fp4_half(const void *map, unsigned char *dst, int expert, int tile, int kg,
                                             std::uint64_t *bar) {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(policy));
    const int k = kg * 128;
    const int row = expert * 4096 + tile * 64;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                 "l"(map), "r"(k), "r"(row), "r"(unsigned(__cvta_generic_to_shared(bar))), "l"(policy)
                 : "memory");
}
using Fp4RowScales = uint2;
__device__ __forceinline__ Fp4RowScales fp4_row_scales(const Args &args, int expert, int row) {
    const auto *base = args.downScales + size_t(expert) * 4096 * 8;
    if (args.gatePages)
        return reinterpret_cast<const uint2 *>(base)[row];
    return make_uint2(*reinterpret_cast<const unsigned *>(base + fastllm_mxfp4::scaleOffset(row, 0, 4096)),
                      *reinterpret_cast<const unsigned *>(base + fastllm_mxfp4::scaleOffset(row, 4, 4096)));
}
// FP32 2^(ue8m0 - 127) of byte `k32` of the row's eight scale bytes.
__device__ __forceinline__ float fp4_scale(uint2 bytes, int k32) {
    const unsigned word = k32 < 4 ? bytes.x : bytes.y;
    return __uint_as_float(((word >> (8 * (k32 & 3))) & 0xffu) << 23);
}
// Keep the scale loads ahead of the MMA wait.
__device__ __forceinline__ void fp4_pin(uint2 s) { asm volatile("" ::"r"(s.x), "r"(s.y) : "memory"); }
// Each CTA runs two 160-thread Down groups over disjoint TMEM columns.
constexpr int kGroupStride = 44032, kPerTokenWeightOffset = 5120, kDownRingSlots = 2, kDownGroups = 2;
constexpr int kRouteScratch = kDownGroups * kGroupStride + 8192;
static_assert(kPerTokenWeightOffset + kDownRingSlots * 16384 <= kGroupStride);
static_assert(kGroupStride % 1024 == 0 && kPerTokenWeightOffset % 1024 == 0);
constexpr int kDoneOffset = 0x80, kTmemFreeOffset = 0xc0, kPerTokenFirstHalfDoneOffset = 0x200;
constexpr int kPartialOffset = 1024, kWeightedOffset = 3200;
static_assert(kPerTokenFirstHalfDoneOffset + sizeof(std::uint64_t) <= kPartialOffset);
static_assert(kPartialOffset + 8 * 64 * sizeof(float) <= kWeightedOffset);
static_assert(kWeightedOffset + 4 * 32 * sizeof(float) <= kPerTokenWeightOffset);

constexpr int kCommonB = kDownGroups * kGroupStride;
constexpr int kCommonScales = kRouteScratch + 1024;
constexpr int kCommonReady = kRouteScratch + 1152;
constexpr int kSecondB = kRouteScratch + 2048;
static_assert(kCommonB + 4 * 2048 == kRouteScratch);
static_assert(kRouteScratch + sizeof(sel::PreparedRouteScratch) <= kCommonScales);
static_assert(kCommonScales + 2 * 16 * sizeof(float) <= kCommonReady);
static_assert(kCommonReady + 2 * 4 * sizeof(std::uint64_t) <= kSecondB);
static_assert(kSecondB + 4 * 2048 <= kShared);
static_assert(kSecondB + 4 * 2048 <= kTmemSlotOffset);
// Compact CTAs have two consumer warpgroups followed by their two issuer warps.
// Group IDs remain the owner of the shared-image and named-barrier mapping.
__device__ __forceinline__ int group_id() {
    return threadIdx.x < 256 ? int(threadIdx.x) / 128 : (int(threadIdx.x) - 256) / 32;
}
__device__ __forceinline__ int group_tid() {
    return threadIdx.x < 256 ? int(threadIdx.x) % 128 : 128 + (int(threadIdx.x) & 31);
}
__device__ unsigned char *cta_base(unsigned char *local) { return local - group_id() * kGroupStride; }
__device__ unsigned char *common_image(unsigned char *local, int pair) {
    const int owner = group_id() & 1;
    return cta_base(local) + (owner ? kSecondB : kCommonB) + pair * 2048;
}
__device__ unsigned *common_scales(unsigned char *local) {
    return reinterpret_cast<unsigned *>(cta_base(local) + kCommonScales + (group_id() & 1) * 64);
}
__device__ std::uint64_t *common_ready(unsigned char *local) {
    return reinterpret_cast<std::uint64_t *>(cta_base(local) + kCommonReady + (group_id() & 1) * 32);
}

__device__ unsigned generation(const Args &args) { return unsigned(*args.epoch); }
__device__ std::uint64_t *full(unsigned char *s) { return reinterpret_cast<std::uint64_t *>(s); }
__device__ void consumer_sync() { naivert::named_barrier_sync(128, 8 + group_id()); }
__device__ int sw128(int row, int k) { return row * 128 + (((k / 16) ^ (row & 7)) * 16) + k % 16; }

// Each consumer warp sends this tile to two TP peers.
constexpr int kPeersPerWarp = 2;
__device__ __forceinline__ unsigned long long *mailbox(const Args &args, unsigned epoch, int destination, int source,
                                                       int row, int tile, int lane) {
    const size_t bank = (epoch & 1u) * 8 * 8 * 2048;
    return args.peers[destination] + bank + (row * 8 + source) * 2048 + tile * 32 + lane;
}

// e4m3 codes of this lane's four values of a warp-wide 128-value block (tagged words a, b) and the block's scale.
__device__ __forceinline__ unsigned quantize_k32(const naivert::TaggedPair &a, const naivert::TaggedPair &b,
                                                 unsigned &scale) {
    const float4 v =
        make_float4(__uint_as_float(naivert::tagged_payload(a.lo)), __uint_as_float(naivert::tagged_payload(a.hi)),
                    __uint_as_float(naivert::tagged_payload(b.lo)), __uint_as_float(naivert::tagged_payload(b.hi)));
    return fastllm_rows8_k32::quantize(v, scale);
}
__device__ __forceinline__ bool tagged_block_ready(const naivert::TaggedPair &a, const naivert::TaggedPair &b,
                                                   unsigned tag) {
    return naivert::tagged_tag(a.lo) == tag && naivert::tagged_tag(a.hi) == tag && naivert::tagged_tag(b.lo) == tag &&
           naivert::tagged_tag(b.hi) == tag;
}
__device__ __forceinline__ unsigned quantize_tagged_k32(const unsigned long long *input, unsigned tag,
                                                        unsigned &scale) {
    auto a = naivert::load_tagged_pair<false>(input);
    auto b = naivert::load_tagged_pair<false>(input + 2);
    while (!tagged_block_ready(a, b, tag)) {
        a = naivert::load_tagged_pair<false>(input);
        b = naivert::load_tagged_pair<false>(input + 2);
    }
    return quantize_k32(a, b, scale);
}
} // namespace moe_down
