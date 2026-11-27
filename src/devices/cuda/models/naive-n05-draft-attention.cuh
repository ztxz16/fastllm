#pragma once
#include "naive-n05-draft-mlp.cuh"

namespace fastllm_draft_fused {
// Keep the TP graph's existing parameter layout and PDL edges. A replica records
// the same operator with a local cooperative grid and a replay-time KV binding.
template <class Policy, class... Extra>
__global__ __launch_bounds__(Policy::Threads, Policy::MinBlocks) void FusedDraftAttention(const __grid_constant__
                                                                                          typename Policy::Args args,
                                                                                          Extra... extra) {
    extern __shared__ __align__(128) unsigned char memory[];
    Policy::Run(args, memory, extra...);
}

struct ReplicaAttention {
    using BF = __nv_bfloat16;
    static constexpr int Dim = 128, Heads = 32, KVHeads = 4, Width = Dim * Heads;
    static constexpr int KVWidth = KVHeads * Dim, Packed = Width + 2 * KVWidth;
    // The split record holds 128 FP32 PV values, maximum and denominator, aligned to 16 bytes.
    static constexpr int Chunk = 128, Stride = Dim + 4, Threads = DraftReplicaProjection::Threads;
    static constexpr int MinBlocks = 3;
    struct Args {
        const BF *input, *qkvWeight, *oWeight;
        const float *qNorm, *kNorm, *postNorm, *positions;
        const int *live;
        const void *const *binding;
        BF *raw, *attention, *residual, *output;
        float *partials, *normScratch;
        int rows, window;
        float eps, theta;
    };
    // A tile covers one query head, every draft row, and 128 keys.
    struct Tile {
        BF query[8][Dim], suffix[8][Dim];
        float scores[8][Chunk];
        BF probability[8][Chunk];
        BF value[8][16][16];
    };
    __device__ static float Round(float x) { return float(__float2bfloat16_rn(x)); }

    __device__ static void PrepareHead(const Args &a, int head, int row, BF *dst) {
        const int lane = threadIdx.x % 32, sourceRow = min(row, a.rows - 1);
        const BF *src = a.raw + sourceRow * Packed + head * Dim;
        const float *gamma = head < Heads ? a.qNorm : a.kNorm;
        float x[4], sum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j] = float(src[j * 32 + lane]);
            sum += x[j] * x[j];
        }
        for (int bit = 16; bit; bit /= 2)
            sum += __shfl_xor_sync(~0u, sum, bit);
        const float inv = rsqrtf(sum / Dim + a.eps);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            x[j] = Round(Round(x[j] * inv) * gamma[j * 32 + lane]);
        const BF *cs = reinterpret_cast<const BF *>(a.normScratch) + sourceRow * Dim;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = 32 * j + lane;
            const float c = float(cs[col]), s = float(cs[col + 64]);
            dst[col] = __float2bfloat16_rn(Round(x[j] * c) - Round(x[j + 2] * s));
            dst[col + 64] = __float2bfloat16_rn(Round(x[j + 2] * c) + Round(x[j] * s));
        }
    }
    __device__ static void Attend(const Args &a, int task, int base, BF *key, BF *value, Tile &s) {
        const int head = task % Heads, first = task / Heads * Chunk;
        const int count = min(Chunk, base + a.rows - first);
        const int t = threadIdx.x, warp = t / 32, lane = t % 32, group = lane / 4, pair = lane % 4;
        const int kv = head / (Heads / KVHeads) * Dim;
        PrepareHead(a, head, warp, s.query[warp]);
        PrepareHead(a, Heads + head / (Heads / KVHeads), warp, s.suffix[warp]);
        __syncthreads();
        // All tiles read the new suffix from QKV scratch. Only the first tile of each KV head
        // publishes it to the current request, so readers never race the cache writes.
        if (task < Heads && head % (Heads / KVHeads) == 0 && warp < a.rows) {
            const int col = (threadIdx.x % 32) * 4;
            *reinterpret_cast<uint2 *>(key + (base + warp) * KVWidth + kv + col) =
                *reinterpret_cast<const uint2 *>(&s.suffix[warp][col]);
            *reinterpret_cast<uint2 *>(value + (base + warp) * KVWidth + kv + col) =
                *reinterpret_cast<const uint2 *>(a.raw + warp * Packed + Width + KVWidth + kv + col);
        }
        auto keyRow = [&](int pos) -> const BF * {
            return pos < base ? key + pos * KVWidth + kv : s.suffix[pos - base];
        };
        auto valueRow = [&](int pos) -> const BF * {
            return pos < base ? value + pos * KVWidth + kv : a.raw + (pos - base) * Packed + Width + KVWidth + kv;
        };
        using fastllm::cuda::low_latency::projection::mma_bf16_16816;
        for (int firstToken = warp * 16; firstToken < Chunk; firstToken += 128) {
            const int token = firstToken + group;
            const BF *q = s.query[group];
            const BF *k0 = keyRow(first + min(token, count - 1));
            const BF *k1 = keyRow(first + min(token + 8, count - 1));
            float dot[4] = {};
#pragma unroll
            for (int k = 0; k < Dim; k += 16) {
                const int col = k + pair * 4;
                auto x = *reinterpret_cast<const uint2 *>(k0 + col);
                auto y = *reinterpret_cast<const uint2 *>(k1 + col);
                auto z = *reinterpret_cast<const uint2 *>(q + col);
                mma_bf16_16816(dot, x.x, y.x, x.y, y.y, z.x, z.y);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = pair * 2 + (j & 1), pos = token + (j / 2) * 8;
                const int left = max(0, base + row - a.window + 1);
                s.scores[row][pos] =
                    pos < count && first + pos >= left ? Round(Round(dot[j]) * rsqrtf(float(Dim))) : -CUDART_INF_F;
            }
        }
        __syncthreads();
        float maximum = -CUDART_INF_F;
#pragma unroll
        for (int k = lane; k < Chunk; k += 32)
            maximum = fmaxf(maximum, s.scores[warp][k]);
        for (int bit = 16; bit; bit /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(~0u, maximum, bit));
        float sum = 0;
#pragma unroll
        for (int k = lane; k < Chunk; k += 32) {
            const float e = expf(s.scores[warp][k] - maximum);
            s.probability[warp][k] = __float2bfloat16_rn(e);
            sum += e;
        }
        for (int bit = 16; bit; bit /= 2)
            sum += __shfl_xor_sync(~0u, sum, bit);
        float *record = a.partials + size_t(task) * a.rows * Stride;
        if (!lane && warp < a.rows) {
            record[warp * Stride + Dim] = maximum;
            record[warp * Stride + Dim + 1] = sum;
        }
        __syncthreads();
        float out[4] = {};
#pragma unroll
        for (int k = 0; k < Chunk; k += 16) {
            const int offset = k + lane / 2, col = (lane % 2) * 8;
            *reinterpret_cast<uint4 *>(&s.value[warp][lane / 2][col]) =
                offset < count ? *reinterpret_cast<const uint4 *>(valueRow(first + offset) + warp * 16 + col)
                               : make_uint4(0, 0, 0, 0);
            __syncwarp();
            unsigned v0, v1, v2, v3;
            const auto addr = static_cast<unsigned>(
                __cvta_generic_to_shared(&s.value[warp][lane % 8 + (lane / 16) * 8][(lane / 8 % 2) * 8]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                         : "r"(addr)
                         : "memory");
            const auto *p = reinterpret_cast<const unsigned *>(&s.probability[group][k + 2 * pair]);
            mma_bf16_16816(out, v0, v1, v2, v3, p[0], p[4]);
            __syncwarp();
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = pair * 2 + (j & 1);
            if (row < a.rows)
                record[row * Stride + warp * 16 + group + (j / 2) * 8] = out[j];
        }
        __syncthreads();
    }
    __device__ static void Merge(const Args &a, int task, int splits) {
        const int row = task / Heads, head = task % Heads, lane = threadIdx.x;
        const int stride = Heads * a.rows * Stride;
        const float *p = a.partials + (head * a.rows + row) * Stride;
        __shared__ float coeff[32];
        if (lane < 32) {
            float m = lane < splits ? p[lane * stride + Dim] : -CUDART_INF_F;
            float maxval = m;
            for (int j = 16; j; j /= 2)
                maxval = fmaxf(maxval, __shfl_xor_sync(~0u, maxval, j));
            float scale = expf(m - maxval), den = lane < splits ? scale * p[lane * stride + Dim + 1] : 0.f;
            for (int j = 16; j; j /= 2)
                den += __shfl_xor_sync(~0u, den, j);
            coeff[lane] = scale / den;
        }
        __syncthreads();
        if (lane < Dim) {
            float sum = 0.f;
            for (int i = 0; i < splits; ++i)
                sum = fmaf(coeff[i], p[i * stride + lane], sum);
            a.attention[row * Width + head * Dim + lane] = __float2bfloat16_rn(sum);
        }
        __syncthreads();
    }
    template<bool Batched> __device__ __forceinline__ static void RunOwner(Args a) {
        using Op = ReplicaAttention;
        using BF = Op::BF;
        const auto grid = cooperative_groups::this_grid();
        if constexpr (Batched) {
            const int owner = blockIdx.y, first = owner * a.rows;
            a.input += first * Width; a.raw += first * Packed;
            a.attention += first * Width; a.output += first * Width; a.residual += first * Width;
            a.positions += first; a.live += owner; a.binding += 2 * owner;
            a.normScratch += first * (Width / 8);
            const int maxSplits = (a.window - 1 + a.rows + Chunk - 1) / Chunk;
            a.partials += size_t(first) * maxSplits * Heads * Stride;
        }
        BF *key = (BF *)a.binding[0], *value = (BF *)a.binding[1];
        const bool live = key && value && *a.live > 0;
        if constexpr (!Batched) if (!live) return;
        // An idle owner still participates in every cooperative grid barrier.
        const int base = min(max(*a.live, 1) - 1, a.window - 1), splits = (base + a.rows + Op::Chunk - 1) / Op::Chunk;
        __shared__ __align__(16) unsigned char storage[sizeof(Op::Tile)];
        auto *partials = reinterpret_cast<float *>(storage);
        auto &sharedTile = *reinterpret_cast<Op::Tile *>(storage);
        for (int tile = blockIdx.x; live && tile < Op::Packed / 16; tile += gridDim.x) {
            float lo = 0, hi = 0;
            DraftReplicaProjection::DotTile<8 * Op::Width>(a.qkvWeight + size_t(tile) * 16 * Op::Width, a.input, a.rows,
                                                           tile, partials, lo, hi);
            if (threadIdx.x < 64 && threadIdx.x % 8 < a.rows) {
                BF *dst = a.raw + (threadIdx.x % 8) * Op::Packed + tile * 16 + threadIdx.x / 8;
                dst[0] = __float2bfloat16_rn(lo);
                dst[8] = __float2bfloat16_rn(hi);
            }
            __syncthreads();
        }
        // Reuse residual-norm scratch for RoPE until the O projection starts.
        if (live && blockIdx.x < a.rows && threadIdx.x < 64) {
            const int d = threadIdx.x, r = blockIdx.x;
            BF *cs = reinterpret_cast<BF *>(a.normScratch) + r * 128;
            const float angle = a.positions[r] * powf(a.theta, -float(d) / 64.f);
            cs[d] = __float2bfloat16_rn(cosf(angle));
            cs[d + 64] = __float2bfloat16_rn(sinf(angle));
        }
        grid.sync();
        for (int task = blockIdx.x; live && task < splits * Op::Heads; task += gridDim.x)
            Op::Attend(a, task, base, key, value, sharedTile);
        grid.sync();
        for (int task = blockIdx.x; live && task < a.rows * Op::Heads; task += gridDim.x)
            Op::Merge(a, task, splits);
        grid.sync();
        for (int tile = blockIdx.x; live && tile < Op::Width / 16; tile += gridDim.x) {
            float lo = 0, hi = 0;
            DraftReplicaProjection::DotTile<8 * Op::Width>(a.oWeight + size_t(tile) * 16 * Op::Width, a.attention,
                                                           a.rows, tile, partials, lo, hi);
            // All eight warps have contributed their partials before reusing shared memory.
            __syncthreads();
            const int t = threadIdx.x, row = t % 8;
            if (t < 64 && row < a.rows) {
                BF *dst = a.residual + row * Op::Width + tile * 16 + t / 8;
                BF x = __float2bfloat16_rn(float(dst[0]) + float(__float2bfloat16_rn(lo)));
                BF y = __float2bfloat16_rn(float(dst[8]) + float(__float2bfloat16_rn(hi)));
                dst[0] = x;
                dst[8] = y;
                partials[t] = float(x) * float(x);
                partials[t + 64] = float(y) * float(y);
            }
            __syncthreads();
            if (t < 16 && row < a.rows) {
                float sum = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    sum += partials[(t / 8) * 64 + 8 * j + row];
                a.normScratch[row * (Op::Width / 8) + tile * 2 + t / 8] = sum;
            }
            __syncthreads();
        }
        grid.sync();
        if (live) DraftReplicaProjection::Normalize(a.rows, a.residual, a.postNorm, a.eps, a.normScratch, a.output);
    }
    __device__ static void Run(const Args &a, unsigned char *) { RunOwner<false>(a); }
};
struct ReplicaAttentionBatch : ReplicaAttention {
    __device__ static void Run(const Args &a, unsigned char *) { RunOwner<true>(a); }
};
} // namespace fastllm_draft_fused
