#pragma once
// BF16 ring specialization with the original BF16 / softmax boundaries.
// The default PV keeps slot-ordered FP32 accumulation. Verification may select
// Tensor Core PV: the same BF16 probabilities/values, with a different FP32 sum
// order. Keep the scalar implementation for existing callers and older SMs.
template <int Threads = 1024, int Tile = 32, bool Dynamic = false, typename Binding = const void *const *,
          bool MmaPV = false, bool ReuseShared = false, bool TensorQK = false>
__device__ __forceinline__ void SwaAttentionBf16Body(const BF16 *q, const BF16 *k, const BF16 *v,
                                                     const float *sink, BF16 *out, const int *live, int heads,
                                                     int kvHeads, const Binding &cache, int row, int head,
                                                     int valueTile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool TensorPV = MmaPV;
#else
    constexpr bool TensorPV = false;
#endif
    // Serving graphs bind request-owned rings through the stable four-pointer
    // table. Resolve it at replay; padded rows never dereference a null entry.
    constexpr bool Descriptors = std::is_same<Binding, BatchSwaKeys>::value;
    q += (size_t)row * heads * 192;
    out += (size_t)row * heads * 128;
    int length;
    if constexpr (Descriptors)
        length = cache.Live(row);
    else
        length = live[row];
    const int keys = min(128, length), first = max(0, length - 128);
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, h = head;
    const int kh = h / (heads / kvHeads), dim0 = valueTile * Tile;
    if (keys <= 0) {
        if (t < Tile)
            out[h * 128 + dim0 + t] = __float2bfloat16_rn(0.f);
        return;
    }
    if constexpr (Descriptors) {
        const auto &r = cache.rows[row];
        k = static_cast<const BF16 *>(r.cache ? r.cache[0] : r.key);
        v = static_cast<const BF16 *>(r.cache ? r.cache[1] : r.value);
    } else if constexpr (Dynamic) {
        k = static_cast<const BF16 *>(cache[(size_t)row * 4]);
        v = static_cast<const BF16 *>(cache[(size_t)row * 4 + 1]);
    }
    // Candidate rows live after the committed ring. Each descriptor retains
    // its own causal end, including ragged requests and replay-time lengths.
    auto physical = [&](int slot) {
        const int logical = first + slot;
        if constexpr (Descriptors) {
            const int offset = cache.rows[row].verifyOffset;
            const int prefix = length - offset - 1;
            if (offset >= 0 && logical >= prefix)
                return 128 + logical - prefix;
        }
        return logical & 127;
    };
    __shared__ float score[128], scratch[256], maximum, denominator;
    using Value = BF16;
    // QKV and attention execute sequentially within each CTA. A fused caller
    // can reuse QKV's dynamic input tile instead of adding both shared buffers.
    __shared__ __align__(16) Value localValue[ReuseShared ? 1 : 128][Tile];
    extern __shared__ __align__(128) unsigned char stage[];
    Value(*value)[Tile] = ReuseShared ? reinterpret_cast<Value(*)[Tile]>(stage) : localValue;
    for (int i = t; i < (TensorPV ? 128 : keys) * (Tile / 8); i += Threads) {
        const int slot = i / (Tile / 8), col = i % (Tile / 8) * 8;
        const uint4 packed =
            !TensorPV || slot < keys
                ? *reinterpret_cast<const uint4 *>(v + physical(slot) * kvHeads * 128 + kh * 128 + dim0 + col)
                : make_uint4(0, 0, 0, 0);
        // XOR-swizzled 8-element vectors avoid ldmatrix bank conflicts.
        const int column = TensorPV ? col ^ ((slot & (Tile / 8 - 1)) * 8) : col;
        *reinterpret_cast<uint4 *>(&value[slot][column]) = packed;
    }
    if constexpr (TensorQK && TensorPV) {
        for (int task = warp; task < 8; task += Threads / 32) {
            const int group = lane / 4, quad = lane % 4;
            float c[4] = {};
#pragma unroll
            for (int col = 0; col < 192; col += 32) {
                int ki = col + quad * 8, s0 = task * 16 + group, s1 = s0 + 8;
                uint4 a{}, b{};
                if (s0 < keys)
                    a = *reinterpret_cast<const uint4 *>(k + physical(s0) * kvHeads * 192 + kh * 192 + ki);
                if (s1 < keys)
                    b = *reinterpret_cast<const uint4 *>(k + physical(s1) * kvHeads * 192 + kh * 192 + ki);
                uint4 x = *reinterpret_cast<const uint4 *>(q + h * 192 + ki);
                fastllm_swa_mma::Step(c, a, b, x);
            }
            if (quad == 0) {
                if (task * 16 + group < keys)
                    score[task * 16 + group] = RoundBF16(RoundBF16(c[0]) * rsqrtf(192.f));
                if (task * 16 + group + 8 < keys)
                    score[task * 16 + group + 8] = RoundBF16(RoundBF16(c[2]) * rsqrtf(192.f));
            }
        }
    } else {
        float query[6];
#pragma unroll
        for (int j = 0; j < 6; ++j)
            query[j] = float(q[h * 192 + lane + j * 32]);
        for (int slot = warp; slot < keys; slot += Threads / 32) {
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < 6; ++j)
                dot += query[j] * float(k[physical(slot) * kvHeads * 192 + kh * 192 + lane + j * 32]);
            dot = WarpSum(dot);
            if (lane == 0)
                score[slot] = RoundBF16(RoundBF16(dot) * rsqrtf(192.f));
        }
    }
    __syncthreads();
    const float bias = sink ? sink[h] : -INFINITY;
    if (t < 256)
        scratch[t] = t < keys ? fmaxf(bias, score[t]) : bias;
    __syncthreads();
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]), b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]), d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float mx = fmaxf(fmaxf(a, b), fmaxf(c, d));
        for (int offset = 16; offset; offset >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
        if (t == 0)
            maximum = mx;
    }
    __syncthreads();
    float p = t < keys ? expf(score[t] - maximum) : 0.f, sum = p;
    if (t == 0 && sink)
        sum += expf(bias - maximum);
    if (t < 256)
        scratch[t] = sum;
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192],
              c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (t == 0)
            denominator = total;
    }
    __syncthreads();
    if (t < keys)
        score[t] = denominator > 0 ? RoundBF16(p / denominator) : 0.f;
    __syncthreads();
    if constexpr (TensorPV) {
        __shared__ __align__(16) BF16 probability[128];
        if (t < 128)
            probability[t] = __float2bfloat16_rn(t < keys ? score[t] : 0.f);
        __syncthreads();
        // One warp computes 16 output columns. Replicate P across the MMA's
        // eight columns and retain column zero; no extra global workspace.
        if (warp < Tile / 16) {
            const int group = lane / 4, quad = lane % 4;
            float result[4] = {};
#pragma unroll
            for (int slot = 0; slot < 128; slot += 16) {
                const int row = slot + lane % 8 + lane / 16 * 8;
                const int column = warp * 16 + lane / 8 % 2 * 8;
                const unsigned address =
                    (unsigned)__cvta_generic_to_shared(&value[row][column ^ ((row & (Tile / 8 - 1)) * 8)]);
                unsigned a0, a1, a2, a3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "r"(address));
                const unsigned b0 = *reinterpret_cast<const unsigned *>(probability + slot + quad * 2);
                const unsigned b1 = *reinterpret_cast<const unsigned *>(probability + slot + quad * 2 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                             : "+f"(result[0]), "+f"(result[1]), "+f"(result[2]), "+f"(result[3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            if (quad == 0) {
                out[h * 128 + dim0 + warp * 16 + group] = __float2bfloat16_rn(result[0]);
                out[h * 128 + dim0 + warp * 16 + group + 8] = __float2bfloat16_rn(result[2]);
            }
        }
    } else {
        if (t < Tile) {
            float result = 0.f;
            for (int firstSlot = 0; firstSlot < keys; firstSlot += 8) {
                float p[8], val[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (firstSlot + j < keys) {
                        p[j] = score[firstSlot + j];
                        val[j] = float(value[firstSlot + j][t]);
                    }
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (firstSlot + j < keys)
                        result += p[j] * val[j];
            }
            out[h * 128 + dim0 + t] = __float2bfloat16_rn(result);
        }
    }
}

template <int Threads = 1024, int Tile = 32, bool Dynamic = false, typename Binding = const void *const *,
          bool MmaPV = false>
__global__ __launch_bounds__(Threads) void NaiveSwaAttentionBf16(const BF16 *q, const BF16 *k, const BF16 *v,
                                                                 const float *sink, BF16 *out,
                                                                 const int *live, int heads, int kvHeads,
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
                                                                 const __grid_constant__
#endif
                                                                     Binding cache = {}) {
    constexpr bool Descriptors = std::is_same<Binding, BatchSwaKeys>::value;
    SwaAttentionBf16Body<Threads, Tile, Dynamic, Binding, MmaPV>(
        q, k, v, sink, out, live, heads, kvHeads, cache, Dynamic || Descriptors ? blockIdx.y : 0, blockIdx.x,
        blockIdx.z);
}
