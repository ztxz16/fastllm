#pragma once
// SM103 TP8 single-row SWA. Existing BF16 weights, ring cache and mailbox are
// borrowed. QKV, PV and O use BF16 tensor cores without repacking weights.
// QKV producers publish per-tile readiness; each attention head waits only for
// its own Q tiles and the shared K/V tiles. Keep the remaining cooperative barrier
// before O so all heads have finished consuming the temporary QKV projection.
// The output is the residual only: its norm is fused into the following router.
#include <cooperative_groups.h>
#include "fastllm-swa-output.cuh"
namespace fastllm_swa_layer {
using namespace fastllm_swa_output;
using fastllm::cuda::WarpSum;

constexpr int LayerThreads = 512, LayerBlocks = 128, StagingBytes = 131072;
constexpr int QkvTiles = 116, RouterThreads = 256, Experts = 256;
static_assert(QkvTiles + 1 <= MaxBlocks, "QKV readiness must fit the existing epoch allocation");
__device__ __forceinline__ float RoundBF16(float x) {
    return float(__float2bfloat16_rn(x));
}

// Only the first 256 threads contribute input squares. The other half of the
// CTA participates in the barriers, then joins QKV and output projection.
__device__ __forceinline__ float SumWide(float value, float *scratch) {
    value = WarpSum(value);
    if (threadIdx.x < 256 && threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
    __syncthreads();
    value = WarpSum(threadIdx.x < 8 ? scratch[threadIdx.x] : 0.f);
    if (threadIdx.x == 0) scratch[0] = value;
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ void SwaMma(float (&c)[4], unsigned a0, unsigned a1,
        unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
}
__device__ __forceinline__ int SwaMmaOffset(int row, int k, int rows) {
    return (k / 64) * (rows * 128) + row * 128 + (((k % 64) / 8) ^ (row & 7)) * 16;
}
__device__ __forceinline__ void Output(
        const BF16 *input, const BF16 *weight, BF16 *projected,
        BF16 *hidden,
        Peers peers, uint64_t *epochs, int rank, const BF16 *branch) {
    constexpr int Tile = 32;
    __shared__ BF16 partial[Tile];
    __shared__ uint64_t step;

    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int first = blockIdx.x * Tile;
    if (t == 0) {
        // Each epoch fits in 32 bits. Reserve the unused high word of
        // epochs[0] for the local completion counter; initialize via the
        // existing zeroed epoch allocation and never update it here.
        auto *generation = reinterpret_cast<uint32_t *>(epochs + blockIdx.x);
        step = *generation;
        *generation = step + 1 == Period ? 0 : step + 1;
    }
    // Split K across eight warps for each set of 16 output rows. Read the
    // original row-major BF16 matrix directly; no packed copy or extra stage.
    __shared__ float partialO[16][16];
    const int rowGroup = warp / 8, part = warp % 8;
    const int group = lane / 4, quad = lane % 4;
    float accum[4] = {};
#pragma unroll
    for (int col = part * 32; col < Inner; col += 256) {
        const int k = col + quad * 8;
        uint4 a0 = *reinterpret_cast<const uint4 *>(
            weight + size_t(first + rowGroup * 16 + group) * Inner + k);
        uint4 a1 = *reinterpret_cast<const uint4 *>(
            weight + size_t(first + rowGroup * 16 + group + 8) * Inner + k);
        uint4 x = *reinterpret_cast<const uint4 *>(input + k);
        SwaMma(accum, a0.x, a1.x, a0.y, a1.y, x.x, x.y);
        SwaMma(accum, a0.z, a1.z, a0.w, a1.w, x.z, x.w);
    }
    if (quad == 0) {
        partialO[warp][group] = accum[0];
        partialO[warp][group + 8] = accum[2];
    }
    __syncthreads();
    if (t < Tile) {
        float sum = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) sum += partialO[t / 16 * 8 + w][t % 16];
        partial[t] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    const size_t base = (step % Frames) * FrameBytes;
    __shared__ uint32_t received[Ranks][Tile / 2];
    // Each lane owns one (peer, BF16 pair). Publish every peer before
    // polling, then acquire the independent sources concurrently.
    if (t < Ranks * (Tile / 2)) {
        const int peer = t / (Tile / 2), index = t % (Tile / 2);
        const int wordIndex = first / 2 + index;
        const uint32_t bits = reinterpret_cast<const uint32_t *>(partial)[index];
        Store(peers.data[peer] + base + rank * SlotBytes + wordIndex * 8,
            (uint64_t(bits) << 32) | tag);
    }
    __syncthreads();
    if (t < Ranks * (Tile / 2)) {
        const int peer = t / (Tile / 2), index = t % (Tile / 2);
        const int wordIndex = first / 2 + index;
        uint64_t word;
        do { word = Load(peers.data[rank] + base + peer * SlotBytes + wordIndex * 8); }
        while (uint32_t(word) != tag);
        received[peer][index] = uint32_t(word >> 32);
    }
    __syncthreads();
    if (t < Tile / 2) {
        const int wordIndex = first / 2 + t;
        float a = 0, b = 0;
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            const uint32_t data = received[peer][t];
            const float x = float(__ushort_as_bfloat16(data & 65535u));
            const float y = float(__ushort_as_bfloat16(data >> 16));
            // Fetch order changes, but accumulation remains rank 0..7 with
            // the original BF16 projection and reduction boundaries.
            if (peer == 0) { a = x; b = y; } else { a += x; b += y; }
        }
        const BF16 x = __float2bfloat16_rn(a), y = __float2bfloat16_rn(b);
        projected[wordIndex * 2] = x; projected[wordIndex * 2 + 1] = y;
        hidden[wordIndex * 2] = __float2bfloat16_rn((branch ? RoundBF16(float(hidden[wordIndex * 2]) + float(branch[wordIndex * 2])) : float(hidden[wordIndex * 2])) + float(x));
        hidden[wordIndex * 2 + 1] = __float2bfloat16_rn((branch ? RoundBF16(float(hidden[wordIndex * 2 + 1]) + float(branch[wordIndex * 2 + 1])) : float(hidden[wordIndex * 2 + 1])) + float(y));
    }
}

__device__ __forceinline__ void AttendStaged(
        BF16 *q, BF16 *k, BF16 *v, const float *sink,
        BF16 *out, const int *live, const BF16 *raw, const float *position, float theta, float valueScale) {
    constexpr int Threads = LayerThreads;
    const int length = *live, keys = min(128, length), first = max(0, length - 128);
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, h = blockIdx.x;
    if (keys <= 0) {
        if (t < 128) out[h * 128 + t] = __float2bfloat16_rn(0.f);
        return;
    }
    __shared__ __align__(16) BF16 localQ[192], freshK[192], freshV[128];
    if (t < 192) {
        if (t < 32) {
            const float angle = *position * powf(theta, -2.f * t / 64.f);
            const float cs = RoundBF16(cosf(angle)), sn = RoundBF16(sinf(angle));
            float qa = float(raw[h * 192 + t]), qb = float(raw[h * 192 + t + 32]);
            float ka = float(raw[1536 + t]), kb = float(raw[1536 + t + 32]);
            localQ[t] = __float2bfloat16_rn(RoundBF16(qa * cs) - RoundBF16(qb * sn));
            localQ[t + 32] = __float2bfloat16_rn(RoundBF16(qb * cs) + RoundBF16(qa * sn));
            freshK[t] = __float2bfloat16_rn(RoundBF16(ka * cs) - RoundBF16(kb * sn));
            freshK[t + 32] = __float2bfloat16_rn(RoundBF16(kb * cs) + RoundBF16(ka * sn));
        } else if (t >= 64) {
            localQ[t] = raw[h * 192 + t];
            freshK[t] = raw[1536 + t];
        }
        if (t < 128) freshV[t] = __float2bfloat16_rn(float(raw[1728 + t]) * RoundBF16(valueScale));
    }
    __syncthreads();
    if (t < 192) q[h * 192 + t] = localQ[t];
    if (h == 0) {
        if (t < 192) k[((length - 1) & 127) * 192 + t] = freshK[t];
        if (t < 128) v[((length - 1) & 127) * 128 + t] = freshV[t];
    }
    __shared__ float score[128], scratch[256], maximum, denominator;

    // Reuse the QKV staging memory after this CTA has finished its QKV
    // tile. Swizzle the V columns so transposed ldmatrix loads avoid conflicts.
    extern __shared__ __align__(128) unsigned char staging[];
    BF16 *stagedV = reinterpret_cast<BF16 *>(staging);
#pragma unroll
    for (int i = t; i < 128 * 16; i += Threads) {
        const int slot = i / 16, dim = i % 16 * 8;
        BF16 *dst = stagedV + slot * 128 + ((dim / 8) ^ (slot & 7)) * 8;
        if (slot < keys - 1) {
            const BF16 *src = v + ((first + slot) & 127) * 128 + dim;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            unsigned address = unsigned(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;"
                :: "r"(address), "l"(src) : "memory");
#else
            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
#endif
        } else if (slot == keys - 1) {
            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(freshV + dim);
        } else {
            *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    if (warp < 8) {
        const int g = lane / 4, quad = lane % 4;
        float c[4] = {};
#pragma unroll
        for (int col = 0; col < 192; col += 32) {
            const int ki = col + quad * 8;
            const int slot0 = warp * 16 + g, slot1 = warp * 16 + g + 8;
            uint4 a{}, b{};
            if (slot0 < keys) a = *reinterpret_cast<const uint4 *>(
                slot0 == keys - 1 ? freshK + ki : k + ((first + slot0) & 127) * 192 + ki);
            if (slot1 < keys) b = *reinterpret_cast<const uint4 *>(
                slot1 == keys - 1 ? freshK + ki : k + ((first + slot1) & 127) * 192 + ki);
            uint4 x = *reinterpret_cast<const uint4 *>(localQ + ki);
            SwaMma(c, a.x, b.x, a.y, b.y, x.x, x.y);
            SwaMma(c, a.z, b.z, a.w, b.w, x.z, x.w);
        }
        if (quad == 0) {
            if (warp * 16 + g < keys) score[warp * 16 + g] = RoundBF16(RoundBF16(c[0]) * rsqrtf(192.f));
            if (warp * 16 + g + 8 < keys) score[warp * 16 + g + 8] = RoundBF16(RoundBF16(c[2]) * rsqrtf(192.f));
        }
    }
    __syncthreads();
    const float bias = sink ? sink[h] : -INFINITY;
    if (t < 256) scratch[t] = t < keys ? fmaxf(bias, score[t]) : bias;
    __syncthreads();
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]), b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]), d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float mx = fmaxf(fmaxf(a, b), fmaxf(c, d));
        for (int offset = 16; offset; offset >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
        if (t == 0) maximum = mx;
    }
    __syncthreads();
    float p = t < keys ? expf(score[t] - maximum) : 0.f, sum = p;
    if (t == 0 && sink) sum += expf(bias - maximum);
    if (t < 256) scratch[t] = sum;
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192],
              c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (t == 0) denominator = total;
    }
    __syncthreads();
    if (t < keys) score[t] = denominator > 0 ? RoundBF16(p / denominator) : 0.f;
    __syncthreads();

    __shared__ __align__(16) BF16 probability[128];
    if (t < 128) probability[t] = __float2bfloat16_rn(t < keys ? score[t] : 0.f);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
    if (warp < 8) {
        const int group = lane / 4, quad = lane % 4;
        float accum[4] = {};
#pragma unroll
        for (int col = 0; col < 128; col += 16) {
            const int row = col + lane % 8 + lane / 16 * 8;
            const int dim = warp * 16 + lane / 8 % 2 * 8;
            unsigned address = unsigned(__cvta_generic_to_shared(
                stagedV + row * 128 + ((dim / 8) ^ (row & 7)) * 8));
            unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(address));
#endif
            unsigned b0 = *reinterpret_cast<const unsigned *>(probability + col + quad * 2);
            unsigned b1 = *reinterpret_cast<const unsigned *>(probability + col + quad * 2 + 8);
            SwaMma(accum, a0, a1, a2, a3, b0, b1);
        }
        if (quad == 0) {
            out[h * 128 + warp * 16 + group] = __float2bfloat16_rn(accum[0]);
            out[h * 128 + warp * 16 + group + 8] = __float2bfloat16_rn(accum[2]);
        }
    }
}

// The host launcher also uses these arguments for the following norm/router.
struct Args {
    const BF16 *branch, *qkvWeight, *oWeight;
    BF16 *hidden, *normed, *q, *k, *v, *attn, *projected;
    const float *inputNorm, *postNorm, *sink, *position;
    const int *live;
    const void *const *cache;
    float eps, theta, valueScale;
    Peers peers;
    uint64_t *epochs;
    int rank;
    const float *routerWeight;
    float *routerOut;
};

// Low words retain the established mailbox generations. High word zero is
// reserved for OutputNorm's completion counter; high words 1..116 are unused
// there and hold these local readiness flags. NormRouter clears them after the
// entire Layer kernel finishes on the same stream, before the next invocation.
// Thus interleaved old output kernels and epoch wrap do not leave stale tags.
__device__ __forceinline__ void PublishQkv(uint64_t *epochs, int tile) {
    auto *flag = reinterpret_cast<unsigned *>(epochs + tile + 1) + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.gpu.global.u32 [%0],1;" :: "l"(flag) : "memory");
#endif
}
__device__ __forceinline__ void WaitQkv(uint64_t *epochs, int tile) {
    auto *flag = reinterpret_cast<unsigned *>(epochs + tile + 1) + 1;
    unsigned ready = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(ready) : "l"(flag) : "memory");
    } while (!ready);
#endif
}
static __global__ __launch_bounds__(LayerThreads, 1) void Layer(Args a) {
    auto grid = cooperative_groups::this_grid();
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, g = lane / 4, quad = lane % 4;
    extern __shared__ __align__(128) unsigned char staging[];
    __shared__ __align__(16) BF16 normalized[4096];
    __shared__ float sums[8], partial[LayerThreads / 32][16];
    if (blockIdx.x < QkvTiles) {
        for (int item = t; item < 16 * 512; item += LayerThreads) {
            int row = item / 512, k = item % 512 * 8;
            unsigned dst = unsigned(__cvta_generic_to_shared(staging + SwaMmaOffset(row, k, 16)));
            const BF16 *src = a.qkvWeight + (size_t(blockIdx.x) * 16 + row) * 4096 + k;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst), "l"(src) : "memory");
#endif
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    float values[16], affine[16], sum = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (t >= 256) { values[j] = 0; affine[j] = 0; continue; }
        int d = t + j * 256;
        values[j] = RoundBF16(float(a.hidden[d]) + float(a.branch[d]));
        affine[j] = a.inputNorm[d];
        sum += values[j] * values[j];
    }
    float inv = rsqrtf(SumWide(sum, sums) / 4096 + a.eps);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        if (t < 256) normalized[t + j * 256] = __float2bfloat16_rn(RoundBF16(values[j] * inv) * affine[j]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
    if (blockIdx.x < QkvTiles) {
        float c[4] = {};
#pragma unroll 4
        for (int col = warp * 32; col < 4096; col += LayerThreads) {
            int k = col + quad * 8;
            uint4 a0 = *reinterpret_cast<const uint4 *>(staging + SwaMmaOffset(g, k, 16));
            uint4 a1 = *reinterpret_cast<const uint4 *>(staging + SwaMmaOffset(g + 8, k, 16));
            uint4 x = *reinterpret_cast<const uint4 *>(normalized + k);
            SwaMma(c, a0.x, a1.x, a0.y, a1.y, x.x, x.y);
            SwaMma(c, a0.z, a1.z, a0.w, a1.w, x.z, x.w);
        }
        if (quad == 0) { partial[warp][g] = c[0]; partial[warp][g + 8] = c[2]; }
        __syncthreads();
        if (t < 16) {
            float value = 0;
#pragma unroll
            for (int w = 0; w < LayerThreads / 32; ++w) value += partial[w][t];
            a.projected[blockIdx.x * 16 + t] = __float2bfloat16_rn(value);
        }
        // Publish only after every thread has completed its QKV stores.
        __syncthreads();
        if (t == 0) PublishQkv(a.epochs, blockIdx.x);
    }
    // Only the eight attention CTAs poll. Cooperative admission keeps all
    // producers resident, even when a head starts before other Q tiles finish.
    if (blockIdx.x < 8) {
        if (t < 12) WaitQkv(a.epochs, blockIdx.x * 12 + t);
        if (t >= 12 && t < 32) WaitQkv(a.epochs, 96 + t - 12);
        __syncthreads();
    }
    if (blockIdx.x == 0 && *a.live <= 0)
        for (int d = t; d < 1536; d += LayerThreads) a.q[d] = __float2bfloat16_rn(0.f);
    if (blockIdx.x < 8) {
        // Resolve request-owned rings from the stable pointer table at replay.
        BF16 *k = a.cache ? (BF16 *)a.cache[0] : a.k;
        BF16 *v = a.cache ? (BF16 *)a.cache[1] : a.v;
        AttendStaged(a.q, k, v, a.sink, a.attn, a.live, a.projected, a.position, a.theta, a.valueScale);
    }
    // QKV/attention readers finish before Output overwrites projected and
    // applies the deferred branch residual, preserving its BF16 boundary.
    grid.sync();
    Output(a.attn, a.oWeight, a.projected, a.hidden, a.peers, a.epochs, a.rank, a.branch);
}

// Reuse the same TP mailbox and BF16 boundaries for DSA. Norm belongs to
// the following router, avoiding an otherwise serialized full-width reduction.
static __global__ __launch_bounds__(LayerThreads, 1) void OutputResidual(
        const BF16 *input, const BF16 *weight, BF16 *projected, BF16 *hidden,
        Peers peers, uint64_t *epochs, int rank) {
    Output(input, weight, projected, hidden, peers, epochs, rank, nullptr);
}

static __global__ void NormRouter4096(const BF16 *input, const float *weight,
        float *output, const float *normWeight, BF16 *normed, float eps, uint64_t *epochs) {
    // Layer has finished before this kernel runs. Reuse its epoch high words
    // without allocating another buffer; keep the low-word generations intact.
    constexpr int Groups = 4096 / RouterThreads / 4, Warps = RouterThreads / 32;
    const int t = threadIdx.x;
    if (epochs && blockIdx.x == 0 && t < QkvTiles) reinterpret_cast<unsigned *>(epochs + t + 1)[1] = 0;
    __shared__ float sums[Warps];
    alignas(16) float values[Groups * 4], affine[Groups * 4], weights[Groups * 4];
    float square = 0;
#pragma unroll
    for (int j = 0; j < Groups; ++j) {
        const int offset = t * 4 + j * RouterThreads * 4;
        uint2 packed = *reinterpret_cast<const uint2 *>(input + offset);
        values[j * 4] = float(__ushort_as_bfloat16(packed.x & 65535));
        values[j * 4 + 1] = float(__ushort_as_bfloat16(packed.x >> 16));
        values[j * 4 + 2] = float(__ushort_as_bfloat16(packed.y & 65535));
        values[j * 4 + 3] = float(__ushort_as_bfloat16(packed.y >> 16));
        *reinterpret_cast<float4 *>(affine + j * 4) = *reinterpret_cast<const float4 *>(normWeight + offset);
        *reinterpret_cast<float4 *>(weights + j * 4) =
            *reinterpret_cast<const float4 *>(weight + size_t(blockIdx.x) * 4096 + offset);
    }
#pragma unroll
    for (int j = 0; j < Groups * 4; ++j) square += values[j] * values[j];
    square = WarpSum(square);
    if (t % 32 == 0) sums[t / 32] = square;
    __syncthreads();
    square = WarpSum(t < Warps ? sums[t] : 0.f);
    if (t == 0) sums[0] = square;
    __syncthreads();
    const float inv = rsqrtf(sums[0] / 4096 + eps);
#pragma unroll
    for (int j = 0; j < Groups * 4; ++j) {
        values[j] = RoundBF16(RoundBF16(values[j] * inv) * affine[j]);
        if (blockIdx.x == 0)
            normed[t * 4 + (j / 4) * RouterThreads * 4 + j % 4] = __float2bfloat16_rn(values[j]);
    }
    float partial = 0;
#pragma unroll
    for (int j = 0; j < Groups * 4; ++j) partial += values[j] * weights[j];
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2)
        partial += __shfl_down_sync(0xffffffff, partial, stride);
    if (t % 32 == 0) sums[t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        float x = t < Warps ? sums[t] : 0.f;
#pragma unroll
        for (int stride = 1; stride < Warps; stride *= 2)
            x += __shfl_down_sync(0xffffffff, x, stride);
        // Keep the standalone router's double-precision sigmoid division.
        if (t == 0) output[blockIdx.x] = 1.0 / (1.0 + expf(-(x + 0.f)));
    }
}

} // namespace fastllm_swa_layer
