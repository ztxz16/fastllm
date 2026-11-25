#pragma once
#include "fastllm-draft-layer-sm120.cuh"

namespace sm120_draft {
constexpr int MlpWidth = 512, MlpGrid = 128;
struct MlpWorkspace {
    Exchange exchange;
    B intermediate[Rows][MlpWidth];
};
static_assert(offsetof(MlpWorkspace, exchange) == 0);
struct MlpArgs {
    const B *input;
    const uint2 *gateUp;
    const uint4 *down;
    B *hidden, *normed;
    const float *gamma;
    MlpWorkspace *w;
    CustomArPointers peers;
    CustomArRankSignals signals;
    float eps;
    int rank;
};
// Eight columns per tile: adjacent columns are the Gate and Up weights for
// one channel. Each warp reads contiguous native MMA B fragments.
__global__ void PackMlpGateUp(const B *weight, uint2 *packed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, lane = i % 32;
    const int tile = i / (256 * 32), step = (i / 32) % 256;
    const int group = lane / 4, channel = tile * 4 + group / 2 + (group % 2) * MlpWidth;
    const B *p = weight + channel * 4096 + step * 16 + lane % 4 * 2;
    packed[i] = make_uint2(Pair(p), Pair(p + 8));
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
__device__ __forceinline__ B SwiGLU(float gate, float up) {
    const float g = float(__float2bfloat16_rn(gate));
    const float u = float(__float2bfloat16_rn(up));
    return __float2bfloat16_rn(float(__float2bfloat16_rn(g / (1.f + expf(-g)))) * u);
}

__device__ __forceinline__ void GateUpPairs(const MlpArgs &a, float (&part)[8][128]) {
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const B *x = a.input + min(lane / 4, Rows - 1) * 4096 + lane % 4 * 2;
    const uint2 *w = a.gateUp + blockIdx.x * 8192 + lane;
    float acc[4]{};
#pragma unroll 8
    for (int j = 0; j < 256; j += 8) {
        const int kt = j + warp;
        uint2 weights = w[kt * 32];
        unsigned lo = Pair(x + kt * 16), hi = Pair(x + kt * 16 + 8);
        Mma(acc, make_uint4(lo, lo, hi, hi), weights.x, weights.y);
    }
    auto *p = reinterpret_cast<float2 *>(part);
    p[t] = make_float2(acc[0], acc[1]);
    __syncthreads();
    if (warp == 0 && lane / 4 < Rows) {
        float g = 0.f, u = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const auto v = p[i * 32 + lane];
            g += v.x;
            u += v.y;
        }
        a.w->intermediate[lane / 4][blockIdx.x * 4 + lane % 4] = SwiGLU(g, u);
    }
}
#endif

__global__ __launch_bounds__(Threads, 3) void FusedDraftMlpSm120(MlpArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    auto grid = cooperative_groups::this_grid();
    __shared__ float part[8][128];
    int tile = blockIdx.x, t = threadIdx.x;
    if (t == 0) {
        PrefetchWeight(a.gateUp + tile * 8192, 65536);
        PrefetchWeight(a.down + tile * 2048, 32768);
    }
    fastllm::cuda_pdl::Wait();
    GateUpPairs(a, part);
    grid.sync();
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int outputTile = tile * 2 + n;
        uint4 weight[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            weight[i] = a.down[(outputTile * 4 + i) * Threads + t];
        ProjectAndSend(&a.w->intermediate[0][0], a.peers, a.rank, weight, part, outputTile);
        __syncthreads();
    }
    FinishLayer(a, a.w->exchange, part, grid);
#else
    asm("trap;");
#endif
}
} // namespace sm120_draft
