#pragma once
namespace sm120_verify {
constexpr int SwaGrid = OutputBlocks + 8 * Heads;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
// SM90's history staging schedule with SM120 FP8 cache rows. Each attention
// CTA starts reading the already committed part while QKV CTAs produce the candidates.
struct RingTile {
    uint8_t key[128][192], value[128][128];
    float scores[128];
};
template <bool Committed>
__device__ __forceinline__ void StageRingKV(const Args &a, int task, RingTile &mem) {
    auto r = FastllmNaiveResolveDSARow(a.batch.rows[task / 8]);
    int live = max(0, min(r.capacity, *r.liveKeys)), count = min(128, live),
        prefix = max(0, count - r.verifyOffset - 1);
    int first = Committed ? 0 : prefix, end = Committed ? prefix : count;
    auto *key = static_cast<const uint8_t *>(r.key);
    auto *value = static_cast<const uint8_t *>(r.value);
    for (int i = threadIdx.x; i < (end - first) * 20; i += Threads) {
        int slot = first + i / 20, c = i % 20, physical = Token<true>(r, slot, live);
        const uint8_t *src = c < 12 ? key + physical * 192 + c * 16 : value + physical * 128 + (c - 12) * 16;
        uint8_t *dst = c < 12 ? mem.key[slot] + c * 16 : mem.value[slot] + (c - 12) * 16;
        unsigned addr = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(addr), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ float RingWarpMax(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1)
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    return x;
}
__device__ __forceinline__ void RingWindowAttention(const Args &a, int task, RingTile &m) {
    int t = threadIdx.x, warp = t / 32, lane = t % 32, h = task % 8;
    auto r = FastllmNaiveResolveDSARow(a.batch.rows[task / 8]);
    int count = max(0, min(min(128, r.capacity), *r.liveKeys));
    StageRingKV<false>(a, task, m);
    float q[6];
#pragma unroll
    for (int j = 0; j < 6; j++)
        q[j] = float(a.query[task * 192 + lane + j * 32]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int slot = warp + iter * 16;
        float dot = 0;
        if (slot < count) {
#pragma unroll
            for (int j = 0; j < 6; j++)
                dot = fmaf(q[j], float(reinterpret_cast<const __nv_fp8_e4m3 *>(m.key[slot])[lane + j * 32]),
                           dot);
            dot = CustomArNormWarpSum(dot);
        }
        if (lane == 0)
            m.scores[slot] =
                slot < count ? CustomArNormRound(CustomArNormRound(dot) * rsqrtf(192.f)) : -INFINITY;
    }
    __syncthreads();
    // A warp spans the window in four stripes. Preserve the two pairwise
    // sums and the sink contribution while keeping all reductions warp-local.
    if (warp == 0) {
        const float bias = a.sink ? a.sink[h] : -INFINITY;
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = m.scores[lane + i * 32];
        float mx = RingWarpMax(fmaxf(bias, fmaxf(fmaxf(p[0], p[2]), fmaxf(p[1], p[3]))));
        mx = __shfl_sync(0xffffffff, mx, 0);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            p[i] = lane + i * 32 < count ? expf(p[i] - mx) : 0.f;
        const float first = lane == 0 && a.sink ? p[0] + expf(bias - mx) : p[0];
        float total = CustomArNormWarpSum((first + p[2]) + (p[1] + p[3]));
        total = __shfl_sync(0xffffffff, total, 0);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            m.scores[lane + i * 32] = total > 0 ? CustomArNormRound(p[i] / total) : 0.f;
    }
    __syncthreads();
    if (t < 128) {
        float sum = 0.f;
        if (count == 128) {
#pragma unroll 8
            for (int k = 0; k < 128; k++)
                sum = fmaf(m.scores[k], float(reinterpret_cast<const __nv_fp8_e4m3 *>(m.value[k])[t]), sum);
        } else
            for (int k = 0; k < count; k++)
                sum = fmaf(m.scores[k], float(reinterpret_cast<const __nv_fp8_e4m3 *>(m.value[k])[t]), sum);
        StoreOutput(a, task * 128 + t, sum);
    }
}
#endif
__global__ __launch_bounds__(Threads, 2) void SwaQkvGemm(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
        Args a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
    cudaTriggerProgrammaticLaunchCompletion();
    __shared__ VerifyProjectionStorage s;
    ProjectVerifyQkv<true>(a, blockIdx.x, s.panel, s.part);
#else
    asm("trap;");
#endif
}
__global__ __launch_bounds__(Threads, 2) void SwaPackedLayer(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
        Args a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
    auto grid = cooperative_groups::this_grid();
    __shared__ union {
        RingTile history;
        VerifyProjectionStorage projection;
    } store;
    auto *sm = store.projection.panel;
    auto &part = store.projection.part;
    int cta = blockIdx.x, task = cta - OutputBlocks;
    if (cta < OutputBlocks)
        StageVerifyOutputWeight(a, sm);
    else
        StageRingKV<true>(a, task, store.history);
    fastllm::cuda_pdl::Wait();
    if (cta < 116) {
        const int t = threadIdx.x;
        if (t < 128)
            part[0][t] = float(a.projection[(t % 8) * 1856 + fastllm_swa_mma::QkvColumn(cta, t / 8)]);
        __syncthreads();
        CommitQkv<true>(a, cta, part[0]);
    }
    grid.sync();
    if (task >= 0)
        RingWindowAttention(a, task, store.history);
    grid.sync();
    // As in the B300 packed tail, start the router prefetch before output projection and TP.
    cudaTriggerProgrammaticLaunchCompletion();
    FinishVerifyOutput(a, sm, part, grid);
#else
    asm("trap;");
#endif
}
} // namespace sm120_verify
