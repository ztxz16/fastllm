#pragma once
namespace sm120_verify {
constexpr int OutputBlocks = 128;
struct __align__(16) VerifyProjectionStorage {
    float part[16][128];
    uint8_t panel[32768];
};
constexpr size_t VerifyExchangeBytes = 2 * 8 * 4096 * sizeof(BF);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
// Four interleaved K splits and three accumulator chains reproduce the eight-row Lt plan.
template <bool Ring>
__device__ __forceinline__ void ProjectVerifyQkv(const Args &a, int tile, uint8_t *stage,
                                                 float (&part)[16][128]) {
    constexpr int Width = 4096, splits = 4;
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, lane = t % 4, row = g;
    const int lo = fastllm_swa_mma::QkvColumn(tile, g), hi = fastllm_swa_mma::QkvColumn(tile, g + 8);
    const int split = warp / 3, chain = warp % 3;
    float acc[4]{};
    const auto weights = static_cast<const uint8_t *>(a.qkvWeight);
    const auto scales = a.qkvScales;
    auto activation = a.prepared.values ? a.prepared : a.quant;
    activation.width = Width;
    constexpr int Chunk = 1024;
    // Ping-pong 16 KiB panels. XOR the 32-byte groups to avoid shared bank conflicts.
    auto copy = [&](int start, int stageId) {
        for (int i = t; i < 16 * Chunk / 16; i += Threads) {
            int r = i / (Chunk / 16), k = i % (Chunk / 16) * 16;
            int column = fastllm_swa_mma::QkvColumn(tile, r);
            auto dst = static_cast<unsigned>(
                __cvta_generic_to_shared(stage + stageId * 16384 + r * Chunk + (k ^ (r % 8 * 32))));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst),
                         "l"(weights + column * Width + start + k)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    copy(0, 0);
    for (int start = 0; start < Width; start += Chunk) {
        int stageId = start / Chunk % 2;
        if (start + Chunk < Width) {
            copy(start + Chunk, 1 - stageId);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        if constexpr (Ring) {
            if (!start) fastllm::cuda_pdl::Wait();
        }
        __syncthreads();
        if (split < splits) {
#pragma unroll
            for (int kt = split; kt < Chunk / 128; kt += splits) {
#pragma unroll
                for (int ix = 0; ix < 2; ++ix) {
                    int sub = chain + ix * 3;
                    if (sub < 4) {
                        int offset = (kt * 4 + sub) * 32, k = start + offset, group = k / 32;
                        NativeStep(acc, stage + stageId * 16384 + g * Chunk + (offset ^ (g % 8 * 32)),
                                   stage + stageId * 16384 + (g + 8) * Chunk + (offset ^ (g % 8 * 32)),
                                   activation.values + row * Width + k,
                                   Ring ? a.swaScales[fastllm_mxfp8::ScaleIndex(activation, lane % 2 ? hi : lo, k)] :
                                          __float_as_uint(scales[(lane % 2 ? hi : lo) * (Width / 32) + group]) >> 23,
                                   activation.scales[fastllm_mxfp8::ScaleIndex(activation, row, k)]);
                    }
                }
            }
        }
        __syncthreads();
    }
    FinishVerifyQkv<Ring>(a, tile, acc, part);
}
__device__ __forceinline__ void StageVerifyOutputWeight(const Args &a, uint8_t *sm) {
    const int t = threadIdx.x;
    const uint4 *weight = static_cast<const uint4 *>(a.outWeight) + blockIdx.x * 2048;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = i * 512 + t, r = v / 64, k = v % 64 * 16;
        uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(sm + r * 1024 + (k ^ (r % 8 * 32))));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst), "l"(weight + v) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void ProjectVerifyOutput(const Args &a, uint8_t *sm, float (&partials)[16][128]) {
    int t = threadIdx.x, lane = t % 32, g = lane / 4, q = lane % 4;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int w = t / 32, group = w / 3, chain = w % 3;
    if (group < 2) {
        int base = group * 16, lo = base + g, hi = lo + 8;
        auto act = a.quant;
        act.width = 1024;
        float acc[4]{};
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                int sub = chain + j * 3;
                if (sub < 4) {
                    int kg = kt * 4 + sub, k = kg * 32;
                    NativeStep(
                        acc, sm + lo * 1024 + (k ^ (lo % 8 * 32)), sm + hi * 1024 + (k ^ (hi % 8 * 32)),
                        act.values + g * 1024 + k,
                        __float_as_uint(a.outScales[(blockIdx.x * 32 + (q % 2 ? hi : lo)) * 32 + kg]) >> 23,
                        act.scales[fastllm_mxfp8::ScaleIndex(act, g, k)]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
            partials[w][(g + i / 2 * 8) * 8 + q * 2 + i % 2] = acc[i];
    }
    __syncthreads();
    if (t < 256) {
        int col = t % 32, row = t / 32, i = col % 16 * 8 + row, dt = col / 16 * 3;
        int tileId = blockIdx.x * 2 + col / 16, rot = (3 - (tileId % 4) % 3) % 3;
        float sum =
            (partials[dt + rot][i] + partials[dt + (rot + 1) % 3][i]) + partials[dt + (rot + 2) % 3][i];
        BF rounded = __float2bfloat16_rn(sum);
        a.projection[row * 4096 + blockIdx.x * 32 + col] = rounded;
        const int ch = blockIdx.x * 32 + col;
        static_cast<BF *>(a.exchange.scratch[ch / 512])[(a.rank * 8 + row) * 512 + ch % 512] = rounded;
    }
}
__device__ __forceinline__ void FinishVerifyOutput(const Args &a, uint8_t *sm, float (&part)[16][128],
                                                   cooperative_groups::grid_group grid) {
    const int cta = blockIdx.x;
    if (cta < OutputBlocks)
        ProjectVerifyOutput(a, sm, part);
    __threadfence_system();
    grid.sync();
    if (cta < a.rows) {
        int t = threadIdx.x;
        auto *own = static_cast<CustomArNormPacketType *>(a.exchange.scratch[a.rank]);
        CustomArBarrierStart<8>(a.signals, a.signals.signals[a.rank], a.rank);
        // Each owner reduces its contiguous 512 channels in reference rank order.
        auto *tile = reinterpret_cast<CustomArNormPacketType *>(part);
        if (t < 64) {
            CustomArRankData shards;
#pragma unroll
            for (int rank = 0; rank < 8; ++rank)
                shards.ptrs[rank] = own + rank * 512;
            tile[t] = CustomArReducePacket<8>(shards, cta * 64 + t);
        }
        __syncthreads();
        // Replicate each owned shard into the second (all-gather) half of every mailbox.
        auto *dst = static_cast<CustomArNormPacketType *>(a.exchange.scratch[t / 64]);
        dst[4096 + cta * 512 + a.rank * 64 + t % 64] = tile[t % 64];
        __threadfence_system();
        CustomArBarrierPublish<8>(a.signals, a.signals.signals[a.rank], a.rank);
        // The next start handshake also protects these reads before mailbox reuse.
        auto reduced = own[4096 + cta * 512 + t];
        if (!a.gamma) {
            auto *slot = reinterpret_cast<CustomArNormPacketType *>(a.hidden + cta * 4096) + t;
            auto h = *slot;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair)
                reinterpret_cast<__nv_bfloat162 *>(&h)[pair] = __hadd2(
                    reinterpret_cast<const __nv_bfloat162 *>(&h)[pair],
                    reinterpret_cast<const __nv_bfloat162 *>(&reduced)[pair]);
            *slot = h;
            return;
        }
        CustomArNormPacket(reduced, a.hidden + cta * 4096, a.gamma, a.normed + cta * 4096, a.eps, part[0],
                           threadIdx.x);
    }
}
#endif
} // namespace sm120_verify
