#pragma once
namespace sm120_draft {
using B = __nv_bfloat16;
constexpr int Rows = 7, Threads = 256;
struct Exchange {
    // O tiles push 512-channel slices to their owners. The two rendezvous
    // separate peer writes to incoming from the all-gather into returned.
    CustomArNormPacketType incoming[Rows][8][64], returned[Rows][512];
};
__device__ __forceinline__ unsigned Pair(const B *x) { return *reinterpret_cast<const unsigned *>(x); }
// Store each immutable 16x16 BF16 panel in native m16n8k16 A-fragment order.
__global__ void PackProjection(const B *weight, uint4 *packed, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = i % 32;
    int kt = (i / 32) % (width / 16), tile = i / (2 * width);
    const B *p = weight + (tile * 16 + lane / 4) * width + kt * 16 + lane % 4 * 2;
    packed[i] = make_uint4(Pair(p), Pair(p + 8 * width), Pair(p + 8), Pair(p + 8 * width + 8));
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
// Only immutable weight panels may be touched before the PDL acquire.
__device__ __forceinline__ void PrefetchWeight(const void *src, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void Mma(float (&acc)[4], uint4 a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ReduceWarps(float (&acc)[4], float (&part)[8][128]) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        part[warp][(lane / 4 + i / 2 * 8) * 8 + lane % 4 * 2 + i % 2] = acc[i];
    __syncthreads();
}
__device__ __forceinline__ float ColumnSum(float (&part)[8][128], int i) {
    float x = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        x += part[w][i];
    return x;
}
__device__ __forceinline__ void ProjectAndSend(const B *input, CustomArPointers peers, int rank,
                                               uint4 (&w)[4], float (&part)[8][128], int tile) {
    int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const B *x = input + min(lane / 4, Rows - 1) * 512 + lane % 4 * 2;
    float acc[4]{};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int k = (j * 8 + warp) * 16;
        Mma(acc, w[j], Pair(x + k), Pair(x + k + 8));
    }
    ReduceWarps(acc, part);
    // Transpose the native MMA accumulator into complete 16-byte peer stores.
    B value = t < 128 ? __float2bfloat16_rn(ColumnSum(part, t)) : B(0.f);
    __syncthreads();
    B *rows = reinterpret_cast<B *>(part);
    if (t < 128)
        rows[t % 8 * 16 + t / 8] = value;
    __syncthreads();
    if (t < Rows * 2) {
        const int packet = tile * 2 + t % 2;
        auto *owner = const_cast<Exchange *>(static_cast<const Exchange *>(peers.Load().ptrs[packet / 64]));
        CustomArStoreGlobal16(reinterpret_cast<const CustomArNormPacketType *>(rows)[t],
                              &owner->incoming[t / 2][rank][0], packet % 64);
    }
}
template <class LayerArgs>
__device__ __forceinline__ void FinishLayer(const LayerArgs &a, Exchange &local, float (&part)[8][128],
                                            cooperative_groups::grid_group grid) {
    int t = threadIdx.x, b = blockIdx.x;
    __threadfence_system();
    grid.sync();
    cudaTriggerProgrammaticLaunchCompletion();
    if (b < Rows) {
        auto *self = a.signals.signals[a.rank];
        CustomArBarrierStart<8>(a.signals, self, a.rank);
        auto *slice = reinterpret_cast<CustomArNormPacketType *>(part);
        if (t < 64) {
            CustomArRankData sources;
#pragma unroll
            for (int peer = 0; peer < 8; ++peer)
                sources.ptrs[peer] = local.incoming[b][peer];
            slice[t] = CustomArReducePacket<8>(sources, t);
        }
        __syncthreads();
        const auto peers = a.peers.Load();
        const auto reduced = slice[t % 64];
#pragma unroll
        for (int dest = t / 64; dest < 8; dest += 4) {
            auto *peer = const_cast<Exchange *>(static_cast<const Exchange *>(peers.ptrs[dest]));
            CustomArStoreGlobal16(reduced, peer->returned[b], a.rank * 64 + t % 64);
        }
        CustomArBarrierPublish<8>(a.signals, self, a.rank);
        for (int i = t; i < 512; i += Threads)
            reinterpret_cast<CustomArNormPacketType *>(a.normed)[b * 512 + i] = local.returned[b][i];
        __syncthreads();
        fastllm::cuda_norm::AddRMSNormBody<B, 4096, true, false>(a.hidden, a.normed, a.gamma, a.normed, 4096,
                                                                 a.eps, false,
                                                                 fastllm::cuda_norm::NoNormSideOutput{});
    }
}
#endif
} // namespace sm120_draft
