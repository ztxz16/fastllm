#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

// Copyright (c) 2025 DeepSeek. MIT license; see ../../../LICENSE.
// Adapted for dense 1x32 W8A8 projections. Weight scales stay in their original
// FP32 [N,K/32] layout: TMA stages the four scales alongside each K128 tile.
// Small M swaps operands and optionally splits K; large M uses TMA writeback.
namespace deep_gemm {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages,
          bool Swapped, bool Split, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(128 + (BLOCK_M <= 64 ? 128 : 256), 1) void
sm90_fp8_gemm_block32_impl(void* output, uint32_t logical_rows,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    constexpr uint32_t BLOCK_K = 128, SF_GRAN = 32;
    constexpr uint32_t kNumMathThreads = BLOCK_M <= 64 ? 128 : 256;
    constexpr uint32_t kSwizzleAMode = 128, kSwizzleBMode = 128;
    constexpr uint32_t kScaleGroupsPerStage = BLOCK_K / SF_GRAN;
    constexpr uint32_t kAccumBanks = 2;
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "Unaligned N tile");
    DG_STATIC_ASSERT(not Split or Swapped, "Split-K uses the direct FP32 epilogue");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
    // Shared memory
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * kScaleGroupsPerStage * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    constexpr uint32_t SMEM_SFB_SIZE = kNumStages * BLOCK_N * kScaleGroupsPerStage * sizeof(float);

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = shape_k / BLOCK_K / (Split ? 2 : 1);
    const uint32_t k_block_start = Split ? blockIdx.y * num_total_k_blocks : 0;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + SMEM_SFB_SIZE);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, 1, true, kNumSMs>(shape_m, shape_n, shape_k);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = (k_block_idx + k_block_start) * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, m_block_idx * BLOCK_M);
                    if constexpr (Swapped) {
                        tma::copy<kScaleGroupsPerStage, BLOCK_M, 0>(&tensor_map_sfa, &full_barrier,
                            smem_sfa[stage_idx], (k_block_idx + k_block_start) * kScaleGroupsPerStage,
                            m_block_idx * BLOCK_M, 1);
                    } else {
                        tma::copy<BLOCK_M, kScaleGroupsPerStage, 0>(&tensor_map_sfa, &full_barrier,
                            smem_sfa[stage_idx], m_block_idx * BLOCK_M,
                            (k_block_idx + k_block_start) * kScaleGroupsPerStage, 1);
                    }

                    // Issue TMA B
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx, n_block_idx * BLOCK_N);
                    if constexpr (Swapped) {
                        tma::copy<BLOCK_N, kScaleGroupsPerStage, 0>(&tensor_map_sfb, &full_barrier,
                            smem_sfb + stage_idx * BLOCK_N * kScaleGroupsPerStage,
                            n_block_idx * BLOCK_N, (k_block_idx + k_block_start) * kScaleGroupsPerStage, 1);
                    } else {
                        tma::copy<kScaleGroupsPerStage, BLOCK_N, 0>(&tensor_map_sfb, &full_barrier,
                            smem_sfb + stage_idx * BLOCK_N * kScaleGroupsPerStage,
                            (k_block_idx + k_block_start) * kScaleGroupsPerStage, n_block_idx * BLOCK_N, 1);
                    }
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE
                        + BLOCK_N * kScaleGroupsPerStage * sizeof(float));
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum * kAccumBanks], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if (lane_idx == 0) empty_barriers[stage_idx]->arrive();
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                #pragma unroll 8
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                    const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                    full_barriers[stage_idx]->wait(phase);
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                        const auto m_offset = local_idx * WAVE_BLOCK_M;
                        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                        float sa0[kScaleGroupsPerStage], sa1[kScaleGroupsPerStage];
                        // Capture all scales before arrive: non-storing warps can preload
                        // the next tile immediately after the final WGMMA retires.
                        #pragma unroll
                        for (uint32_t sub_k = 0; sub_k < kScaleGroupsPerStage; ++ sub_k) {
                            const auto sf_a = smem_sfa[stage_idx];
                            const uint32_t a0 = Swapped ? (m_offset + r_0) * 4 + sub_k : sub_k * BLOCK_M + m_offset + r_0;
                            const uint32_t a1 = Swapped ? (m_offset + r_1) * 4 + sub_k : sub_k * BLOCK_M + m_offset + r_1;
                            sa0[sub_k] = do_wgmma_store ? ptx::ld_shared(sf_a + a0) : 0;
                            sa1[sub_k] = do_wgmma_store ? ptx::ld_shared(sf_a + a1) : 0;
                        }
                        auto promote = [&](const uint32_t sub_k, float* partial) {
                            if (not do_wgmma_store) return;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // Weight scales retain every original FP32 bit.
                                const uint32_t bcol = i * 8 + (lane_idx % 4) * 2;
                                const uint32_t b0idx = Swapped ? sub_k * BLOCK_N + bcol : bcol * 4 + sub_k;
                                const uint32_t b1idx = Swapped ? sub_k * BLOCK_N + bcol + 1 : (bcol + 1) * 4 + sub_k;
                                const float b0 = ptx::ld_shared(smem_sfb + stage_idx * 4 * BLOCK_N + b0idx);
                                const float b1 = ptx::ld_shared(smem_sfb + stage_idx * 4 * BLOCK_N + b1idx);
                                shifted_accum[i * 4 + 0] += (sa0[sub_k]*b0) * partial[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (sa0[sub_k]*b1) * partial[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (sa1[sub_k]*b0) * partial[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (sa1[sub_k]*b1) * partial[i * 4 + 3];
                            }
                        };
                        #pragma unroll
                        for (uint32_t sub_k = 0; sub_k < kScaleGroupsPerStage; ++ sub_k) {
                            auto partial = accum + (sub_k % kAccumBanks) * WGMMA::kNumAccum;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(partial[i]);
                            ptx::warpgroup_arrive();
                            a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + sub_k * SF_GRAN) / 16;
                            b_desc.reg32_[0] = b_desc_base_lo + sub_k * SF_GRAN / 16;
                            WGMMA::wgmma(a_desc, b_desc, partial, 0);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(partial[i]);
                            if (sub_k > 0) {
                                ptx::warpgroup_wait<1>();
                                promote(sub_k - 1, accum + ((sub_k - 1) % kAccumBanks) * WGMMA::kNumAccum);
                            }
                        }
                        ptx::warpgroup_wait<0>();
                        promote(kScaleGroupsPerStage - 1, accum + (kAccumBanks - 1) * WGMMA::kNumAccum);
                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1) {
                            __syncwarp();
                            empty_barrier_arrive();
                        }
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            if constexpr (Swapped) {
                if (do_wgmma_store) {
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                        const uint32_t row0 = m_block_idx * BLOCK_M + local_idx * WAVE_BLOCK_M + r_0;
                        const uint32_t row1 = m_block_idx * BLOCK_M + local_idx * WAVE_BLOCK_M + r_1;
                        auto values = final_accum + WGMMA::kNumAccum * local_idx;
                        auto store = [&](uint32_t row, uint32_t col, float value) {
                            if (row >= shape_m || col >= logical_rows) return;
                            if constexpr (Split)
                                static_cast<float*>(output)[(size_t)blockIdx.y * shape_m * logical_rows + (size_t)col * shape_m + row] = value;
                            else
                                static_cast<__nv_bfloat16*>(output)[(size_t)col * shape_m + row] = __float2bfloat16_rn(value);
                        };
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                            const uint32_t col = n_block_idx * BLOCK_N + i * 8 + (lane_idx % 4) * 2;
                            store(row0, col, values[4*i]);
                            store(row0, col+1, values[4*i+1]);
                            store(row1, col, values[4*i+2]);
                            store(row1, col+1, values[4*i+3]);
                        }
                    }
                }
            } else {
                // Ordinary output tiles use 128-byte-swizzled TMA stores.
                constexpr uint32_t kSwizzleDMode = 128;
                constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
                constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode / kNumElemBytes;
                constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
                DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
                DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                                "Unaligned TMA store or too many TMA store instructions");
                DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

                // Skip WGMMA store for the unfilled parts
                if (not do_wgmma_store)
                    continue;

                // Wait last TMA store to be finished
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                // Write back to shared memory using STSM and issue TMA stores
                DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        auto row = in_atom_offset / 8 + lane_idx;
                        auto col = in_atom_offset;
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +           // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset

                        // NOTES: only 16 lanes' addresses are used
                        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                // Use TMA store to write back to global memory
                DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                    auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                    auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                    auto n_idx = n_block_idx * BLOCK_N + in_block_n_offset;
                    auto m_idx = m_block_idx * BLOCK_M;
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
