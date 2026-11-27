#pragma once
#include "api.cuh"
#include "k32_activation.cuh"
#include "fastllm-shared-flex-router-sm103.cuh"
#include "naivert/common/prepared_mlp_input_layout.hpp"
#include "naivert/common/tp8_mailbox.cuh"
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cstdint>
#include <cstring>

#include <cuda/ptx>

#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/slot_ring/protocol.cuh"
#include "fastllm-mxfp4-weight-pack.cuh"
namespace moe_select_gate_up {
using Args = FastllmVerify8Args;
constexpr int kFp4CodeBytes = fastllm_verify8::GateCodeBytes, kFp4PageBytes = fastllm_verify8::GatePageBytes;
constexpr int kRows8Threads = 384, kRows8Rows = 8, kRows8ConsumerThreads = 256, kRows8HelperBaseThread = 256;
constexpr int kRows8Experts = 256, kRows8Selected = 8, kRows8Pages = 4, kRows8Groups = 16;
constexpr int kRows8Outputs = 16, kRows8PageK = 1024, kRows8Hidden = 4096;
constexpr int kRows8MaxDistinct = kRows8Rows * kRows8Selected;
static_assert(kRows8Experts == 256 && kRows8Selected == 8, "warp routing: 8 experts per lane");
static_assert(kRows8Pages == 4, "two pages per parity helper");
static_assert(kRows8Hidden % 16 == 0);

// Shared map: barriers and routing scratch below two weight panes, fp8 activations after.
constexpr int kRows8BarrierOffset = 0x0000;             // pane full [2]
constexpr int kRows8ReleaseBarrierOffset = 0x0040;      // pane free [2]
constexpr int kRows8ActivationBarrierOffset = 0x0360;   // disjoint from down barriers
constexpr int kRows8RouteBarrierOffset = 0x00c0;        // scores + bias staged
constexpr int kRows8RouteReleaseBarrierOffset = 0x00e0; // routes + item list ready
constexpr int kWeightIdsReadyOffset = 0x00f0;
static_assert(kWeightIdsReadyOffset >= kRows8RouteReleaseBarrierOffset + 8);
static_assert(kWeightIdsReadyOffset + 8 <= 0x0100);
constexpr int kRows8SelectedOffset = 0x0100; // int[8 tokens][8 slots]
constexpr int kRows8DistinctOffset = 0x0200; // int[64]
constexpr int kRows8MaskOffset = 0x0300;     // unsigned[8]
constexpr int kRows8CountOffset = 0x0340;    // int: distinct experts
constexpr int kRows8BiasOffset = 0x0400;     // float[256]
constexpr int kRows8SigmoidOffset = 0x0800;  // float[8][256]: RMS-stage sigmoid scores
constexpr int kRows8PartialOffset = 0x2800;
// Item partials occupy [0x800,0x2800). [0x2800,0x3000) holds immutable
// byte [expert][token] selected-slot bitsets (zero if absent).
constexpr int kRows8RoutingSlotOffset = kRows8PartialOffset;
static_assert(kRows8RoutingSlotOffset + kRows8Experts * kRows8Rows <=
              kRows8PartialOffset + 2 * 4 * 16 * 8 * sizeof(float));
// Item partials reuse completed routing scratch: eight partial planes per item.
static_assert(kRows8SigmoidOffset + 2 * 8 * 16 * 8 * sizeof(float) == kRows8PartialOffset);
constexpr int kRows8PaneBaseOffset = 0x4000;
constexpr int kRows8PaneStride = kFp4PageBytes;
constexpr int kRows8ActivationOffset = kRows8PaneBaseOffset + 2 * kRows8PaneStride;
constexpr int kRows8ActivationStride = kRows8Hidden + 16; // token rows 16 B apart mod 128: no bank conflicts on B
constexpr int kRows8ActivationBytes = kRows8Rows * kRows8ActivationStride;
constexpr int kRows8SharedBytes = kRows8ActivationOffset + kRows8ActivationBytes;

struct PreparedRouteScratch {
    int selected[kRows8Rows * kRows8Selected];
    int count;
};
static_assert(kRows8PartialOffset + 2 * 4 * 16 * 8 * 4 <= kRows8PaneBaseOffset);
static_assert(kRows8ActivationOffset == 51200);
static_assert(kRows8SharedBytes <= 232448, "opt-in shared memory");
static_assert(kRows8ActivationBarrierOffset >= kRows8CountOffset + sizeof(int));
template <bool Pages, bool Preselected> struct SelectGateUpBody {
    using Trans64 = naivert::Barrier;

    __device__ static __forceinline__ unsigned decode_fp8_pair(unsigned value) {
        const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(value), __NV_E4M3);
        unsigned bits;
        memcpy(&bits, &pair, sizeof(bits));
        return bits;
    }

    __device__ static __forceinline__ void fma_pair(float &a, float &b, float x, float y, float scale) {
        asm("{\n"
            ".reg .b64 accum, values, scales;\n"
            "mov.b64 accum, {%0, %1};\n"
            "mov.b64 values, {%2, %3};\n"
            "mov.b64 scales, {%4, %4};\n"
            "fma.rn.f32x2 accum, values, scales, accum;\n"
            "mov.b64 {%0, %1}, accum;\n"
            "}"
            : "+f"(a), "+f"(b)
            : "f"(x), "f"(y), "f"(scale));
    }

    // Sixteen e2m1 weights (8 bytes) -> eight f16x2 registers, the same two MMAs, and the K32 dot
    // scaled by its rows' ue8m0 scales (s_lo: row g, s_hi: row g + 8) as it is added to C.
    __device__ static __forceinline__ void mma_fp4_weight(float &p0, float &p1, float &p2, float &p3, const uint2 &a,
                                                          unsigned b0lo, unsigned b1lo, unsigned b0hi, unsigned b1hi,
                                                          float s_lo, float s_hi, float as0, float as1) {
        unsigned h[8];
        asm("{\n"
            ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
            "mov.b32 {b0, b1, b2, b3}, %8;\n"
            "mov.b32 {b4, b5, b6, b7}, %9;\n"
            "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "cvt.rn.f16x2.e2m1x2 %1, b4;\n"
            "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
            "cvt.rn.f16x2.e2m1x2 %3, b6;\n"
            "cvt.rn.f16x2.e2m1x2 %4, b1;\n"
            "cvt.rn.f16x2.e2m1x2 %5, b5;\n"
            "cvt.rn.f16x2.e2m1x2 %6, b3;\n"
            "cvt.rn.f16x2.e2m1x2 %7, b7;\n"
            "}"
            : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]), "=r"(h[4]), "=r"(h[5]), "=r"(h[6]), "=r"(h[7])
            : "r"(a.x), "r"(a.y));
        float q0, q1, q2, q3;
        using Mma = cute::SM80_16x8x16_F32F16F16F32_TN;
        Mma::fma(q0, q1, q2, q3, h[0], h[1], h[2], h[3], b0lo, b1lo, 0.f, 0.f, 0.f, 0.f);
        Mma::fma(q0, q1, q2, q3, h[4], h[5], h[6], h[7], b0hi, b1hi, q0, q1, q2, q3);
        // Two independent FP32 FMAs per instruction. Keep each K32 subtotal
        // and its original accumulation order; no scale folding into MMA inputs.
        q0 *= as0;
        q1 *= as1;
        q2 *= as0;
        q3 *= as1;
        fma_pair(p0, p1, q0, q1, s_lo);
        fma_pair(p2, p3, q2, q3, s_hi);
    }

    __device__ static __forceinline__ void consumer_sync() { naivert::consumer_group_named_sync(); }

    // The four consumer warps of one item slot.
    __device__ static __forceinline__ void slot_sync() { naivert::named_barrier_sync(128u, 9u); }

    __device__ static __forceinline__ std::uint64_t *barriers(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kRows8BarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t *release_barriers(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kRows8ReleaseBarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t *activation_barriers(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kRows8ActivationBarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t *route_barrier(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kRows8RouteBarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t *route_release_barrier(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kRows8RouteReleaseBarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t *weight_ids_ready(unsigned char *shared) {
        return reinterpret_cast<std::uint64_t *>(shared + kWeightIdsReadyOffset);
    }
    __device__ static __forceinline__ int *selected(unsigned char *shared) {
        return reinterpret_cast<int *>(shared + kRows8SelectedOffset);
    }
    __device__ static __forceinline__ int *distinct(unsigned char *shared) {
        return reinterpret_cast<int *>(shared + kRows8DistinctOffset);
    }
    __device__ static __forceinline__ unsigned *expert_mask(unsigned char *shared) {
        return reinterpret_cast<unsigned *>(shared + kRows8MaskOffset);
    }
    __device__ static __forceinline__ int *distinct_count(unsigned char *shared) {
        return reinterpret_cast<int *>(shared + kRows8CountOffset);
    }
    __device__ static __forceinline__ float *bias_scratch(unsigned char *shared) {
        return reinterpret_cast<float *>(shared + kRows8BiasOffset);
    }
    __device__ static __forceinline__ float *sigmoid(unsigned char *shared, int token) {
        return reinterpret_cast<float *>(shared + kRows8SigmoidOffset) + token * kRows8Experts;
    }
    __device__ static __forceinline__ unsigned char *pane(unsigned char *shared, int index) {
        return shared + kRows8PaneBaseOffset + index * kRows8PaneStride;
    }
    __device__ static __forceinline__ __nv_fp8_e4m3 *activation(unsigned char *shared, int token) {
        return reinterpret_cast<__nv_fp8_e4m3 *>(shared + kRows8ActivationOffset + token * kRows8ActivationStride);
    }
    __device__ static __forceinline__ int item_of(int local) { return int(blockIdx.x) + local * int(gridDim.x); }

    // Helper 0 stages the sigmoid scores, the bias and the prepared FP8 activation packet.
    __device__ static void stage_inputs(const Args &args, unsigned char *shared, const unsigned char *prepared) {
        const std::uint32_t kScoresBytes = std::uint32_t(args.rows * kRows8Experts * sizeof(float));
        constexpr std::uint32_t kBiasBytes = std::uint32_t(kRows8Experts * 4);
        const int tid = int(threadIdx.x);
        if (tid >= kRows8HelperBaseThread) {
            const int helper = (tid - kRows8HelperBaseThread) >> 5;
            // Helpers can arrive here independently after consumer-only work.
            // A fixed owner remains unique even when the warp is temporarily divergent.
            if ((tid & 31) != 0) {
                return;
            }
            if (helper == 0) {
                if constexpr (!Preselected) {
                    auto *routing_barrier = route_barrier(shared);
                    Trans64::arrive_and_expect_tx(routing_barrier, kScoresBytes + kBiasBytes);
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                             sigmoid(shared, 0), args.router, kScoresBytes, routing_barrier);
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                             bias_scratch(shared), args.bias, kBiasBytes, routing_barrier);
                }
                auto *ready = activation_barriers(shared);
                // The existing router writes dense K32 E4M3 rows and one UE8M0 byte per group.
                const int rows = args.rows;
                Trans64::arrive_and_expect_tx(ready, rows * (4096 + 128));
#pragma unroll
                for (int row = 0; row < rows; ++row)
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                             activation(shared, row), prepared + row * 4096, 4096u, ready);
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                         shared + naivert::prepared_mlp::kScaleScratch, args.xs, unsigned(rows * 128),
                                         ready);
            }
            return;
        }
        if constexpr (!Preselected)
            Trans64::wait(route_barrier(shared), 0u);
        consumer_sync();
    }
    __device__ static void build_item_list(unsigned char *shared) {
        const int tid = int(threadIdx.x);
        // One expert's eight slots per consumer, before the existing first join.
        reinterpret_cast<unsigned long long *>(shared + kRows8RoutingSlotOffset)[tid] = 0ull;
        unsigned *mask = expert_mask(shared);
        if (tid < 8) {
            mask[tid] = 0u;
        }
        consumer_sync();
        if (tid < kRows8MaxDistinct) {
            const int expert = selected(shared)[tid];
            atomicOr(mask + (expert >> 5), 1u << (expert & 31));
            // Record each routed slot in its token's byte of the expert mask.
            atomicOr(reinterpret_cast<unsigned *>(shared + kRows8RoutingSlotOffset) + expert * 2 + (tid >> 5),
                     1u << (tid & 31));
        }
        consumer_sync();
        {
            // Reuse eight mask popcounts across each consumer warp.
            const int expert = tid, word = expert >> 5, lane = expert & 31;
            const unsigned lane_bits = lane < 8 ? mask[lane] : 0u;
            const unsigned count = __popc(lane_bits);
            const unsigned total = __reduce_add_sync(0xffffffffu, count);
            const unsigned before = __reduce_add_sync(0xffffffffu, lane < word ? count : 0u);
            const unsigned bits = __shfl_sync(0xffffffffu, lane_bits, word);
            if ((bits >> lane) & 1u) {
                const unsigned rank = before + __popc(bits & ((1u << lane) - 1u));
                distinct(shared)[rank] = expert;
            }
            if (tid == 0)
                *distinct_count(shared) = int(total);
        }
        consumer_sync();
    }
    // Fast pages place the MMA fragments and K32 scales in one bulk transaction.
    // The native layout gathers the same operands with TMA.
    __device__ static void helper_pages(const Args &args, unsigned char *shared) {
        const int helper = (threadIdx.x - kRows8HelperBaseThread) / 32, lane = threadIdx.x & 31;
        if (helper >= 2)
            return;
        Trans64::wait(weight_ids_ready(shared), 0u);
        const int items = *distinct_count(shared) * 16;
        for (int local = 0; item_of(local) < items; ++local) {
            const int item = item_of(local), expert = distinct(shared)[item / 16], row = (item % 16) * 32 + lane;
            if constexpr (Pages) {
                if (lane == 0) {
                    const auto *base = args.gatePages + (size_t(expert) * 16 + item % 16) * 4 * kFp4PageBytes;
                    for (int page = helper; page < 4; page += 2) {
                        auto *ready = barriers(shared) + helper;
                        Trans64::wait(release_barriers(shared) + helper, unsigned((local * 2 + page / 2) & 1));
                        Trans64::arrive_and_expect_tx(ready, unsigned(kFp4PageBytes));
                        unsigned long long policy;
                        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
                        asm volatile(
                            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], "
                            "%2, [%3], %4;" ::"r"(unsigned(__cvta_generic_to_shared(pane(shared, helper)))),
                            "l"(base + page * kFp4PageBytes), "r"(kFp4PageBytes),
                            "r"(unsigned(__cvta_generic_to_shared(ready))), "l"(policy)
                            : "memory");
                    }
                }
            } else {
                const auto *scales = args.gateScales + size_t(expert) * 512 * 128;
#pragma unroll
                for (int page = helper; page < 4; page += 2) {
                    auto *ready = barriers(shared) + helper;
                    Trans64::wait(release_barriers(shared) + helper, unsigned((local * 2 + page / 2) & 1));
                    auto *dst = pane(shared, helper);
#pragma unroll
                    for (int k = 0; k < 32; k += 4)
                        *reinterpret_cast<unsigned *>(dst + 16384 + lane * 32 + k) =
                            *reinterpret_cast<const unsigned *>(scales +
                                                                fastllm_mxfp4::scaleOffset(row, page * 32 + k, 512));
                    __syncwarp();
                    if (lane == 0) {
                        Trans64::arrive_and_expect_tx(ready, 16384u);
#pragma unroll
                        for (int segment = 0; segment < 4; ++segment) {
                            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(
                                             unsigned(__cvta_generic_to_shared(dst + segment * 4096))),
                                         "l"(&args.gateMap), "r"(page * 512 + segment * 128),
                                         "r"(expert * 512 + (item % 16) * 32),
                                         "r"(unsigned(__cvta_generic_to_shared(ready)))
                                         : "memory");
                        }
                    }
                }
            }
        }
    }
    __device__ static __forceinline__ void wait_weight_page(unsigned char *shared, int local, int page) {
        Trans64::wait(barriers(shared) + (page & 1), unsigned((local * 2 + page / 2) & 1));
    }

    __device__ static __forceinline__ void release_weight_page(unsigned char *shared, int page) {
        Trans64::arrive(release_barriers(shared) + (page & 1));
    }
    // scale-group parity, and reuses its decoded activation across gate and up.
    __device__ static void accumulate_page(unsigned char *shared, int pane_index, int page, float (&accum)[8]) {
        const int tid = int(threadIdx.x), lane = tid & 31;
        const int k_part = (tid >> 5) & 1;
        const int parity = ((tid >> 5) >> 1) & 1;
        const int lane_group = lane >> 2, lane_quad = lane & 3;
        const auto *weight = reinterpret_cast<const __nv_fp8_e4m3 *>(pane(shared, pane_index));

        const auto *act = activation(shared, lane_group) + page * kRows8PageK + lane_quad * 4;
        const auto *codes = reinterpret_cast<const unsigned char *>(weight);
        const unsigned char *mx_scales = codes + kFp4CodeBytes;
        uint2 fp4_a[4][2][2];
#pragma unroll
        for (int g = 0; g < 4; ++g)
#pragma unroll
            for (int sp = 0; sp < 2; ++sp)
#pragma unroll
                for (int branch = 0; branch < 2; ++branch) {
                    if constexpr (Pages) {
                        const int block = (2 * g + parity) * 4 + k_part + sp * 2;
                        fp4_a[g][sp][branch] = reinterpret_cast<const uint2 *>(
                            codes)[branch * 1024 + block * 32 + lane_group * 4 + lane_quad];
                    } else {
                        const int kb = ((2 * g + parity) * 4 + k_part + sp * 2) * 16 + lane_quad * 2;
                        const int row = branch * 8 + lane_group;
                        const auto *a = codes + (kb / 128) * 4096 + row * 128 + ((kb % 128) ^ ((row & 7) * 16));
                        const auto *b = a + 16 * 128;
                        const unsigned x = *reinterpret_cast<const unsigned short *>(a) |
                                           (unsigned(*reinterpret_cast<const unsigned short *>(a + 8)) << 16);
                        const unsigned y = *reinterpret_cast<const unsigned short *>(b) |
                                           (unsigned(*reinterpret_cast<const unsigned short *>(b + 8)) << 16);
                        fp4_a[g][sp][branch] = make_uint2(x, y);
                    }
                }
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int sg = 2 * g + parity;
            const auto *scales32 = reinterpret_cast<const unsigned *>(shared + naivert::prepared_mlp::kScaleScratch);
            const unsigned t0 = scales32[(lane_quad * 2) * 32 + page * 8 + sg],
                           t1 = scales32[(lane_quad * 2 + 1) * 32 + page * 8 + sg];
            float p[2][4] = {};
#pragma unroll
            for (int sp = 0; sp < 2; ++sp) {
                const int block = sg * 4 + k_part + sp * 2;
                const unsigned b0 = *reinterpret_cast<const unsigned *>(act + block * 32);
                const unsigned b1 = *reinterpret_cast<const unsigned *>(act + block * 32 + 16);
                const unsigned b0lo = decode_fp8_pair(b0), b1lo = decode_fp8_pair(b1);
                const unsigned b0hi = decode_fp8_pair(b0 >> 16), b1hi = decode_fp8_pair(b1 >> 16);
#pragma unroll
                for (int branch = 0; branch < 2; ++branch) {
                    const unsigned char *row_scales = mx_scales + (Pages ? (branch * 32 + block) * 16 + lane_group
                                                                         : (branch * 8 + lane_group) * 32 + block);
                    mma_fp4_weight(p[branch][0], p[branch][1], p[branch][2], p[branch][3], fp4_a[g][sp][branch], b0lo,
                                   b1lo, b0hi, b1hi, __uint_as_float(unsigned(row_scales[0]) << 23),
                                   __uint_as_float(unsigned(row_scales[Pages ? 8 : 16 * 32]) << 23),
                                   fastllm_rows8_k32::scale(t0, block & 3), fastllm_rows8_k32::scale(t1, block & 3));
                }
            }
#pragma unroll
            for (int branch = 0; branch < 2; ++branch) {
                const int stream = branch * 4;
                accum[stream + 0] += p[branch][0];
                accum[stream + 1] += p[branch][1];
                accum[stream + 2] += p[branch][2];
                accum[stream + 3] += p[branch][3];
            }
        }
    }
    __device__ static void finish_item(const Args &args, unsigned char *shared, int expert, int group,
                                       const float (&accum)[8], unsigned long long *tagged_y, unsigned tag) {
        const int tid = int(threadIdx.x), lane = tid & 31, k_part = (tid >> 5) & 1;
        const int parity = ((tid >> 5) >> 1) & 1;
        const int lane_group = lane >> 2, lane_quad = lane & 3;
        // Every consumer acquired route_release before the item loop. Routing has
        // finished reading [0x800, 0x2800); reuse it as [item][branch][K part][parity][128].
        // The two existing slot joins publish these partials and protect their reuse.
        float *values = reinterpret_cast<float *>(shared + kRows8SigmoidOffset);
#pragma unroll
        for (int branch = 0; branch < 2; ++branch) {
            float *mine = values + ((branch * 2 + k_part) * 2 + parity) * 128;
            const int base = branch * 4;
            mine[lane_group * 8 + lane_quad * 2] = accum[base];
            mine[lane_group * 8 + lane_quad * 2 + 1] = accum[base + 1];
            mine[(lane_group + 8) * 8 + lane_quad * 2] = accum[base + 2];
            mine[(lane_group + 8) * 8 + lane_quad * 2 + 1] = accum[base + 3];
        }
        slot_sync();
        const int local = tid;
        const int output = local >> 3, token = local & 7, index = output * 8 + token;
        // Preserve parity addition before K-part addition, including FP32 rounding.
        const float g0 = values[index] + values[128 + index];
        const float g1 = values[256 + index] + values[384 + index];
        const float u0 = values[512 + index] + values[640 + index];
        const float u1 = values[768 + index] + values[896 + index];
        publish_output(args, shared, expert, group, output, token, g0 + g1, u0 + u1, tagged_y, tag);
        slot_sync();
    }

    // silu(gate) x up as tagged y[token][slot][group * 16 + output] if the token selected the expert.
    __device__ static __forceinline__ void publish_output(const Args &args, unsigned char *shared, int expert,
                                                          int group, int output, int token, float gate, float up,
                                                          unsigned long long *tagged_y, unsigned tag) {
        const float gv = __bfloat162float(__float2bfloat16_rn(gate));
        const float uv = __bfloat162float(__float2bfloat16_rn(up));
        // The item-list joins and route_release published this immutable lookup.
        // Highest set bit preserves the original last-occurrence search; zero
        // yields -1 and keeps the existing absent-token guard.
        const unsigned bits = shared[kRows8RoutingSlotOffset + expert * 8 + token];
        const int slot = 31 - __clz(bits);
        if (slot >= 0) {
            const int index = token * kRows8Selected * 256 + slot * 256 + group * kRows8Outputs + output;
            const float value =
                __fmul_rn(__fmul_rn(gv / (1.f + __expf(-gv)), uv), bias_scratch(shared)[token * kRows8Selected + slot]);
            naivert::store_tagged(tagged_y + index, naivert::tagged_word(tag, __float_as_uint(value)));
        }
    }
    // Waits for the RMS stage via PDL. Every thread leaves through the full-CTA
    // register handoff barrier; route_scratch keeps this CTA's routes for down.
    template <int DownColumns, int DownSlot, int DownWarp>
    __device__ static void run(const Args &args, unsigned char *shared, const unsigned char *prepared,
                               unsigned long long *tagged_y, unsigned &tag, const uint64_t *epoch,
                               unsigned char *route_scratch) {
        static_assert(naivert::prepared_mlp::kScaleScratch >= kRows8PartialOffset + 2 * 4 * 16 * 8 * 4);
        static_assert(naivert::prepared_mlp::kScaleScratch + naivert::prepared_mlp::kScaleBytes <=
                      kRows8PaneBaseOffset);
        static_assert(naivert::prepared_mlp::kScaleOffset == kRows8ActivationBytes);
        const int tid = int(threadIdx.x);
        constexpr unsigned kReleaseCount = 128;
        if (tid < 2) {
            Trans64::init(barriers(shared) + tid, 1u);
            Trans64::init(release_barriers(shared) + tid, kReleaseCount);
            naivert::barrier_arrive_count(release_barriers(shared) + tid, kReleaseCount);
        }
        if (tid == 0)
            Trans64::init(activation_barriers(shared), 1u);
        if (tid == 0) {
            Trans64::init(route_barrier(shared), 1u);
            Trans64::init(route_release_barrier(shared), kRows8ConsumerThreads);
            Trans64::init(weight_ids_ready(shared), 1u);
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
        __syncthreads();

        naivert::pdl::wait_for_primary<true>();
        // The router advances the epoch. Read it only after acquiring its generation.
        tag = unsigned(*epoch);
        // Helpers issue the input copies while consumers expand their register budget.
        if (tid < kRows8ConsumerThreads)
            cutlass::arch::warpgroup_reg_alloc<96>();
        if (args.rows < 8 && tid < kRows8ConsumerThreads) {
            for (int row = args.rows; row < 8; ++row) {
                reinterpret_cast<uint4 *>(activation(shared, row))[tid] = {};
                if (tid < 32)
                    reinterpret_cast<unsigned *>(shared + naivert::prepared_mlp::kScaleScratch)[row * 32 + tid] = 0;
            }
        }
        stage_inputs(args, shared, prepared);
        if (tid >= kRows8HelperBaseThread) {
            static_assert(DownColumns > 0 && DownWarp == 10);
            static_assert(DownSlot >= kRows8SharedBytes && DownSlot % alignof(unsigned) == 0);
            if (tid >= DownWarp * 32 && tid < (DownWarp + 1) * 32) {
                cute::TMEM::Allocator1Sm allocator;
                allocator.allocate(DownColumns, reinterpret_cast<unsigned *>(shared + DownSlot));
                allocator.release_allocation_lock();
            }
            helper_pages(args, shared);
            // Different warp roles arrive from distinct instruction locations.
            // Use the non-aligned PTX form, which permits this control flow.
            asm volatile("barrier.sync 0, %0;" ::"n"(kRows8Threads) : "memory");
            cutlass::arch::warpgroup_reg_alloc<64>();
            return;
        }
        using Scratch = fastllm::cuda::RouterTopKScratch<8>;
        auto *sort = reinterpret_cast<Scratch *>(shared + kRows8PaneBaseOffset);
        auto *prob = reinterpret_cast<float *>(sort + 7);
        constexpr int lastScratch = kRows8ActivationOffset + kRows8ActivationBytes;
        static_assert(7 * sizeof(Scratch) + 64 * sizeof(float) <= 2 * kRows8PaneStride);
        static_assert(lastScratch + sizeof(Scratch) <= 2 * 44032 + 8192);
        const int row = tid >> 5;
        auto &ws = row < 7 ? sort[row] : *reinterpret_cast<Scratch *>(shared + lastScratch);
        const float *scores = sigmoid(shared, min(row, args.rows - 1));
        int *ids = selected(shared) + 8 * row;
        float *out = prob + 8 * row;
        if constexpr (Preselected) {
            int lane = tid & 31;
            if (lane < 8) {
                const int source = min(row, args.rows - 1) * 8 + lane;
                ids[lane] = args.ids[source];
                out[lane] = args.prob[source];
            }
        } else {
            if (!fastllm_shared_flex_tp::SelectShortlist(scores, bias_scratch(shared), ids, out, 1, 1.f, ws))
                fastllm::cuda::SelectRouterTopK<256, 8>(scores, bias_scratch(shared), ids, out, 1, 1, 1.f, ws);
        }
        consumer_sync();
        if (tid < 64)
            bias_scratch(shared)[tid] = prob[tid];
        consumer_sync();
        // build_item_list joins after independent mask/slot initialization.
        // That join also publishes every token's selected IDs before their first read.
        build_item_list(shared);
        // The existing consumer join has completed all distinct-ID/count writes.
        // Only weight loaders use this earlier signal; down metadata and MMA keep
        // the original saved-route/activation dependency.
        if (tid == 0)
            Trans64::arrive(weight_ids_ready(shared));
        {
            // Preserve this CTA's routing across shared-memory reuse. The down
            // stage consumes tagged activations without a grid-wide join.
            auto *routes = reinterpret_cast<PreparedRouteScratch *>(route_scratch);
            if (tid < kRows8Rows * kRows8Selected) {
                const int token = tid / kRows8Selected, slot = tid % kRows8Selected;
                const int *chosen = selected(shared) + token * kRows8Selected;
                routes->selected[tid] = chosen[slot];
                const float p = bias_scratch(shared)[tid];
                if (!Preselected && token < args.rows && int(blockIdx.x) == 0) {
                    args.ids[token * kRows8Selected + slot] = chosen[slot];
                    args.prob[token * kRows8Selected + slot] = p;
                }
            }
            const int count = *distinct_count(shared);
            if (tid == 0)
                routes->count = count;
            consumer_sync();
        }
        Trans64::wait(activation_barriers(shared), 0u);
        Trans64::arrive(route_release_barrier(shared));
        Trans64::wait(route_release_barrier(shared), 0u);

        const int items = *distinct_count(shared) * kRows8Groups;
        // Routing uses both consumer warpgroups. One lends its registers to gate/up
        // after publishing its metadata, then returns as a down consumer.
        if (tid >= 128) {
            cutlass::arch::warpgroup_reg_dealloc<32>();
            asm volatile("barrier.sync 0, %0;" ::"n"(kRows8Threads) : "memory");
            cutlass::arch::warpgroup_reg_alloc<88>();
            return;
        }
        cutlass::arch::warpgroup_reg_alloc<176>();
        for (int local = 0; item_of(local) < items; ++local) {
            const int item = item_of(local), expert = distinct(shared)[item >> 4], group = item & 15;
            float accum[8]{};
#pragma unroll 1
            for (int page = 0; page < kRows8Pages; ++page) {
                wait_weight_page(shared, local, page);
                accumulate_page(shared, (page & 1), page, accum);
                release_weight_page(shared, page);
            }
            finish_item(args, shared, expert, group, accum, tagged_y, tag);
        }
        cutlass::arch::warpgroup_reg_dealloc<88>();
        // Different warp roles arrive from distinct instruction locations.
        // Use the non-aligned PTX form, which permits this control flow.
        asm volatile("barrier.sync 0, %0;" ::"n"(kRows8Threads) : "memory");
    }
};

} // namespace moe_select_gate_up
