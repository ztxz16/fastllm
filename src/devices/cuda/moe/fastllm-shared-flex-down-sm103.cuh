#pragma once
// Each CTA owns 128 output features of one row and sums all eight experts in
// the original order. The two-pane helper reads packed FP8 or MXFP4 weights.
// The producer publishes BF16 mailboxes without waiting for any peer; the
// following receive kernel retains the established collective boundary.
namespace fastllm_shared_flex_tp {
constexpr int OwnerDownBytes = 70656;

template<bool KeepExperts, bool PDL = false>
__device__ __forceinline__ void OwnerDownBody(Rank a, const int *ids, int tile, int row) {
    using E4 = cutlass::float_e4m3_t;
    using UE = cutlass::float_ue8m0_t;
    using MMA = cute::SM100_MMA_MXF8F6F4_SS<E4,E4,float,UE,128,8,
        cute::UMMA::Major::K,cute::UMMA::Major::K>;
    extern __shared__ __align__(1024) uint8_t arena[];
    auto weights = reinterpret_cast<uint8_t (*)[2][16384]>(arena);
    auto weightScales = reinterpret_cast<uint8_t (*)[2][512]>(arena + 65536);
    auto input = reinterpret_cast<uint8_t (*)[1024]>(arena + 67584);
    auto inputScales = reinterpret_cast<uint8_t (*)[512]>(arena + 69632);
    __shared__ __align__(8) uint64_t ready[2], released[2], done;
    __shared__ unsigned tmem;
    // The packed eight-row path leaves one N8 accumulator per expert in TMEM.
    // Weight and scale panes can be refilled as soon as their MMA completes.
    constexpr int TmemColumns = KeepExperts ? 128 : 32;
    const int tid = threadIdx.x;
    if (tid == 0) {
        for (int b = 0; b < 2; ++b) {
            unsigned r = unsigned(__cvta_generic_to_shared(&ready[b]));
            unsigned f = unsigned(__cvta_generic_to_shared(&released[b]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],1;"
                         :: "r"(r), "r"(f) : "memory");
        }
        unsigned d = unsigned(__cvta_generic_to_shared(&done));
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1; fence.mbarrier_init.release.cluster;"
                     :: "r"(d) : "memory");
    }
    if (tid < 32) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.allocate(TmemColumns, &tmem);
        allocator.release_allocation_lock();
    }
    __syncthreads();
    if (tid >= 128) {
        const int lane = tid - 128;
        for (int expertSlot = 0; expertSlot < K; ++expertSlot) {
            const int b = expertSlot % 2;
            if (expertSlot >= 2)
                while (!parity_wait(&released[b], (expertSlot / 2 - 1) & 1)) {}
            if (lane == 0) {
                unsigned mb = unsigned(__cvta_generic_to_shared(&ready[b]));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                             :: "r"(mb), "r"(2*(fastllm_shared_flex::weight_tile_bytes(a.mxfp4)+512)) : "memory");
            }
            __syncwarp();
            if (lane < 2) {
                const int expert = ids[row * K + expertSlot];
                tma3(weights[b][lane], a.dmap, lane * 128, tile * 128, expert, &ready[b]);
                bulk(weightScales[b][lane], a.ds + size_t(expert) * H * (I / 32) +
                     (size_t(lane) * H + tile * 128) * 4, 512, &ready[b]);
            }
        }
        return;
    }
    // All expert activations share an immutable N8 image. Each expert's
    // epilogue reads its own column, without broadcasting or requantizing.
    // Gate signalled after selection. Only its activation bytes and scales
    // depend on its completion, so the weight-loading warp can run ahead.
    if constexpr (PDL) cudaGridDependencySynchronize();
    for (int kg = 0; kg < 2; ++kg) {
        for (int j = tid; j < 256; j += 128) {
            const int slot = j / 32, k = j % 32 * 4;
            *reinterpret_cast<unsigned *>(input[kg] + slot * 128 + (k ^ (slot * 16))) =
                ld4(a.mid + (row * K + slot) * I + kg * 128 + k);
        }
        reinterpret_cast<unsigned *>(inputScales[kg])[(tid % 32) * 4 + tid / 32] =
            ld4(a.ms + (row * K + tid % 8) * (I / 32) + kg * 4);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    MatmulSync<true>();
    auto descriptor = fastllm_shared_flex::weight_instruction(a.mxfp4);
    float sum = 0.f;
    for (int expertSlot = 0; expertSlot < K; ++expertSlot) {
        const int b = expertSlot % 2;
        if (tid < 32) {
            while (!parity_wait(&ready[b], (expertSlot / 2) & 1)) {}
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int kg = 0; kg < 2; ++kg) {
                auto wa = dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,128,128,128>((E4 *)weights[b][kg], 0, 0);
                auto xb = dg::mma::sm100::make_umma_desc<cute::UMMA::Major::K,8,128,128>((E4 *)input[kg], 0, 0);
                auto sa = dg::mma::sm100::make_sf_desc(weightScales[b][kg]);
                auto sb = dg::mma::sm100::make_sf_desc(inputScales[kg]);
                const unsigned ta = tmem + (KeepExperts ? 64 + b * 16 : 8) + kg * 8, tb = ta + 4;
                if (tid == 0) {
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sa, ta);
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sb, tb);
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    auto aa = wa, bb = xb;
                    aa.start_address_ += k * 2; bb.start_address_ += k * 2;
                    auto runtime = dg::mma::sm100::make_runtime_instr_desc_with_sf_id(descriptor, k, k);
                    MMA::fma(aa, bb, tmem + (KeepExperts ? expertSlot * 8 : 0), kg != 0 || k != 0, runtime, ta, tb);
                }
            }
            if constexpr (KeepExperts) {
                cutlass::arch::umma_arrive(&released[b]);
                if (expertSlot == K - 1) cutlass::arch::umma_arrive(&done);
            } else cutlass::arch::umma_arrive(&done);
        }
        if constexpr (!KeepExperts) {
            while (!parity_wait(&done, expertSlot & 1)) {}
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            unsigned value;
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem + expertSlot + (unsigned(tid / 32 * 32) << 16), value);
            cutlass::arch::fence_view_async_tmem_load();
            sum = __fadd_rn(sum, __uint_as_float(value));
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            MatmulSync<true>();
            if (tid == 0) {
                unsigned mb = unsigned(__cvta_generic_to_shared(&released[b]));
                asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(mb) : "memory");
            }
        }
    }
    if constexpr (KeepExperts) {
        while (!parity_wait(&done, 0)) {}
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const unsigned rowOffset = unsigned(tid / 32 * 32) << 16;
        unsigned values[K];
#pragma unroll
        for (int slot = 0; slot < K; ++slot)
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem + slot * 8 + slot + rowOffset, values[slot]);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int slot = 0; slot < K; ++slot) sum = __fadd_rn(sum, __uint_as_float(values[slot]));
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        MatmulSync<true>();
    } else {
        a.wlocal[row * H + tile * 128 + tid] = sum;
    }
    const float other = __shfl_down_sync(0xffffffff, sum, 1);
    if (tid % 2 == 0) {
        const unsigned local = pack_bf16(sum, other);
        const int pair = tile * 64 + tid / 2;
        if constexpr (KeepExperts) reinterpret_cast<unsigned *>(a.wlocal)[row * (H / 2) + pair] = local;
        publish_bf16(a, row, pair, unsigned(*a.epoch), local);
    }
    if (tid < 32) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.free(tmem, TmemColumns);
    }
}

template<bool KeepExperts, bool PDL = false> __global__ __launch_bounds__(160,2)
void owner_down(Rank a, const int *ids, const float *prob) {
    if (*a.nuniq <= a.rows * 4) {
        if (threadIdx.x < 128) batched_matmul<false, PDL, 1, true>(a, ids, prob);
        return;
    }
    const int task = blockIdx.x;
    if (task >= a.rows * (H / 128)) return;
    // Neighboring eight-row CTAs request the same output tile. Rows selecting
    // the same expert can reuse those cache lines without a global partial.
    const int tile = KeepExperts ? task / a.rows : task % (H / 128);
    const int row = KeepExperts ? task % a.rows : task / (H / 128);
    OwnerDownBody<KeepExperts, PDL>(a, ids, tile, row);
}

template<bool PackedLocal> __global__
void owner_finish(Rank a, __nv_bfloat16 *output) {
    if (blockIdx.y == 0 && blockIdx.x < 4) a.gcount[blockIdx.x * 64 + threadIdx.x] = 0;
    const int pair = blockIdx.x * 64 + threadIdx.x, row = blockIdx.y;
    const unsigned generation = unsigned(*a.epoch);
    unsigned payload;
    if (*a.nuniq <= a.rows * 4) {
        float2 sum = make_float2(0.f, 0.f);
#pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            float2 value = reinterpret_cast<const float2 *>(a.dpartial + size_t(row * K + slot) * H)[pair];
            sum.x = __fadd_rn(sum.x, value.x); sum.y = __fadd_rn(sum.y, value.y);
        }
        payload = pack_bf16(sum.x, sum.y);
        publish_bf16(a, row, pair, generation, payload);
    } else if constexpr (PackedLocal) {
        payload = reinterpret_cast<const unsigned *>(a.wlocal)[row * (H / 2) + pair];
    } else {
        float2 sum = reinterpret_cast<const float2 *>(a.wlocal + row * H)[pair];
        payload = pack_bf16(sum.x, sum.y);
    }
    float2 total = collect_bf16(a, row, pair, generation, payload);
    reinterpret_cast<__nv_bfloat162 *>(output + row * H)[pair] = __floats2bfloat162_rn(total.x, total.y);
}

template<bool Packed>
static bool SupportsOwnerDown(int smCount) {
    cudaFuncAttributes producer{}, receiver{};
    int producerBlocks = 0, receiverBlocks = 0;
    return cudaFuncGetAttributes(&producer, owner_down<Packed>) == cudaSuccess &&
        producer.binaryVersion == 103 && producer.maxThreadsPerBlock >= 160 &&
        cudaFuncSetAttribute(owner_down<Packed>, cudaFuncAttributeMaxDynamicSharedMemorySize, OwnerDownBytes) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&producerBlocks, owner_down<Packed>, 160, OwnerDownBytes) == cudaSuccess &&
        producerBlocks > 0 && cudaFuncGetAttributes(&receiver, owner_finish<Packed>) == cudaSuccess &&
        receiver.binaryVersion == 103 && receiver.maxThreadsPerBlock >= 64 &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&receiverBlocks, owner_finish<Packed>, 64, 0) == cudaSuccess &&
        receiverBlocks * smCount >= MAX_M * (H / 128);
}
} // namespace fastllm_shared_flex_tp
