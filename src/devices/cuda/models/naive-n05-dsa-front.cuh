#pragma once

// Original row-major BF16 weights and caller-owned scratch. No persistent
// packed weights. Readiness belongs to the calling workspace. Cooperative
// admission keeps every producer resident; acquire/release tags publish only
// the projection tiles and KV rows required by each consumer.
namespace naive_dsa_front {
constexpr int Blocks = 128, Threads = 256, Qkv = 1856, Hidden = 4096;
struct Args {
    const BF16 *input, *weight, *indexWeight;
    const float *normWeight, *normBias, *sink;
    BF16 *raw, *index, *query, *output;
    RopeCacheRow cache[8];
    FastllmNaiveDSABatch attention;
    float *partial;
    unsigned *completion, *ready;
    int rows, rotary;
    bool sharedRows;
    float theta, valueScale;
};
__device__ __forceinline__ unsigned LoadReady(const unsigned *p) {
    unsigned value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(value) : "l"(p) : "memory");
#endif
    return value;
}
__device__ __forceinline__ void Publish(unsigned *p, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.gpu.global.u32 [%0],%1;" :: "l"(p), "r"(value) : "memory");
#endif
}
__device__ __forceinline__ void Wait(const unsigned *p, unsigned value) {
    while (LoadReady(p) != value) {}
}
template<int Rows> __device__ __forceinline__ int InputOffset(int row, int k) {
    return (k / 64) * (Rows * 64) + row * 64 + (((k % 64) / 8) ^ (row & 7)) * 8;
}
__device__ __forceinline__ int WeightOffset(int row, int k) {
    return (k / 64) * 2048 + row * 128 + (((k % 64) / 8) ^ (row & 7)) * 16;
}
__device__ __forceinline__ void Mma(float (&c)[4], uint4 a, uint4 b, uint4 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(x.x),"r"(x.y));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a.z),"r"(b.z),"r"(a.w),"r"(b.w),"r"(x.z),"r"(x.w));
#endif
}
__device__ __forceinline__ void NormalizeIndex(BF16 *x, const float *weight, const float *bias) {
    __shared__ float sums[64], squares[64], mean, deviation;
    const int t = threadIdx.x;
    if (t < 64) {
        float total = 0, square = 0;
        for (int i = t; i < 128; i += 64) { float v = float(x[i]); total += v; square += v * v; }
        sums[t] = total; squares[t] = square;
    }
    __syncthreads();
    for (int step = 32; step; step >>= 1) {
        if (t < step) { sums[t] += sums[t + step]; squares[t] += squares[t + step]; }
        __syncthreads();
    }
    if (!t) {
        mean = sums[0] / 128;
        deviation = squares[0] + mean * mean * 128 - 2 * mean * 128 * mean;
        deviation = sqrt(deviation / 128 + 1e-10);
    }
    __syncthreads();
    if (t < 128) x[t] = __float2bfloat16_rn((float(x[t]) - mean) / deviation * weight[t] + bias[t]);
    __syncthreads();
}
template<int Rows, bool Ordered = false>
__global__ __launch_bounds__(Threads, 1) void Front(const __grid_constant__ Args a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, g = lane / 4, quad = lane % 4;
    __shared__ unsigned generation;
    if (!t) generation = LoadReady(a.ready + blockIdx.x) + 1;
    extern __shared__ __align__(16) unsigned char bytes[];
    if (blockIdx.x < (Qkv + 128) / 16) {
        const bool index = blockIdx.x >= Qkv / 16;
        const int tile = index ? blockIdx.x - Qkv / 16 : blockIdx.x;
        const BF16 *weight = (index ? a.indexWeight : a.weight) + size_t(tile) * 16 * Hidden;
        if constexpr (Ordered) {
            // Verification keeps the compensated GEMV reduction tree. Reusing
            // the decode MMA tree can amplify rounding through later MoE routes.
            for (int job = warp; job < 16 * a.rows; job += Threads / 32) {
                const int row = job / 16, channel = job % 16;
                BF16 value = fastllm::cuda::Bf16OrderedWarpDot<Hidden>(
                    a.input + row * Hidden, weight + size_t(channel) * Hidden);
                if (!lane) (index ? a.index : a.raw)[row * (index ? 128 : Qkv) + tile * 16 + channel] = value;
            }
        } else {
            auto *inputs = reinterpret_cast<BF16 *>(bytes + 131072);
            auto *partial = reinterpret_cast<float *>(inputs + Rows * Hidden);
            for (int i = t; i < 16 * Hidden / 8; i += Threads) {
                const int r = i / 512, k = i % 512 * 8;
                unsigned dst = unsigned(__cvta_generic_to_shared(bytes + WeightOffset(r, k)));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst), "l"(weight + size_t(r) * Hidden + k) : "memory");
            }
            for (int i = t; i < Rows * Hidden / 8; i += Threads) {
                uint4 x = {};
                if (i < a.rows * Hidden / 8) x = reinterpret_cast<const uint4 *>(a.input)[i];
                int r = i / 512, k = i % 512 * 8;
                *reinterpret_cast<uint4 *>(inputs + InputOffset<Rows>(r, k)) = x;
            }
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            float c[4] = {};
        #pragma unroll 4
            for (int col = warp * 32; col < Hidden; col += Threads) {
                int k = col + quad * 8;
                uint4 w0 = *reinterpret_cast<const uint4 *>(bytes + WeightOffset(g, k));
                uint4 w1 = *reinterpret_cast<const uint4 *>(bytes + WeightOffset(g + 8, k));
                uint4 x = {};
                if (g < Rows) x = *reinterpret_cast<const uint4 *>(inputs + InputOffset<Rows>(g, k));
                Mma(c, w0, w1, x);
            }
            // Each lane owns two request columns and two weight rows of the MMA tile.
            for (int j = 0; j < 2; ++j) if (quad * 2 + j < Rows) {
                partial[(warp * 16 + g) * Rows + quad * 2 + j] = c[j];
                partial[(warp * 16 + g + 8) * Rows + quad * 2 + j] = c[j + 2];
            }
            __syncthreads();
            for (int i = t; i < 16 * a.rows; i += Threads) {
                const int r = i / a.rows, b = i % a.rows;
                float sum = 0;
        #pragma unroll
                for (int w = 0; w < 8; ++w) sum += partial[(w * 16 + r) * Rows + b];
                (index ? a.index : a.raw)[b * (index ? 128 : Qkv) + tile * 16 + r] = __float2bfloat16_rn(sum);
            }
        }
    }
    __syncthreads();
    if (!t) Publish(a.ready + blockIdx.x, generation);
    // Every request's Indexer norm and cache write stays in the same CTA.
    // No request can observe another request's temporary values or page table.
    for (int i = blockIdx.x; i < a.rows * 11; i += Blocks) {
        int row = i / 11, head = i % 11;
        const int firstTile = head < 9 ? head * 12 : head == 9 ? 108 : 116;
        const int tiles = head < 9 ? 12 : 8;
        if (t < tiles) Wait(a.ready + firstTile + t, generation);
        __syncthreads();
        if (head == 10) NormalizeIndex(a.index + row * 128, a.normWeight, a.normBias);
        AppendRopeCacheRow<true>(a.cache[row], 0, head, 8, 1, 192, 128, 128,
            a.rotary, a.theta, __float2bfloat16_rn(a.valueScale), 0, false);
        __syncthreads();
        if (!t && head < 10) Publish(a.ready + Blocks + row * 10 + head, generation);
    }
    // Keep inactive rows on the same generation too. A later batch expansion
    // must not mistake an old tag for the current one after uint32 wraparound.
    if (!t && blockIdx.x >= a.rows * 10 && blockIdx.x < 8 * 10)
        Publish(a.ready + Blocks + blockIdx.x, generation);
    for (int i = blockIdx.x; i < a.rows * naive_dsa_split::Splits; i += Blocks) {
        if (t < 10) Wait(a.ready + Blocks + (i / naive_dsa_split::Splits) * 10 + t, generation);
        // Verification rows can read K/V produced by other candidates in the
        // same request. All producers are resident; publish every shared row
        // before attention reads the causal prefix of that cache.
        if (a.sharedRows && t < a.rows * 2)
            Wait(a.ready + Blocks + (t / 2) * 10 + 8 + t % 2, generation);
        __syncthreads();
        naive_dsa_split::Attend(a.query, a.attention.rows[i / naive_dsa_split::Splits], a.sink, a.output, a.partial,
            a.completion, 8, 1, i / naive_dsa_split::Splits, 0, i % naive_dsa_split::Splits);
        __syncthreads();
    }
#endif
}
template<int Rows> constexpr size_t SharedBytes() {
    return 131072 + Rows * Hidden * sizeof(BF16) + 8 * 16 * Rows * sizeof(float);
}
} // namespace naive_dsa_front
