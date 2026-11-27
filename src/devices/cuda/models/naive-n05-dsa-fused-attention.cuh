#pragma once
#include "devices/cuda/naive-n05-kv.cuh"
// Split attention for native cache bindings. The default path retains FastLLM's
// established operand orientation and reduction order. Compact mode adapts
// NaiveRT's eight-head QK/PV fragments and distributed combine; see NOTICE.
// Both paths retain BF16 probabilities, attention sinks and caller-owned scratch.
#include <cuda/atomic>
#include <type_traits>
namespace fastllm_dsa_fused::native_attention {
using BF16 = __nv_bfloat16;
__device__ __forceinline__ void Load4(uint32_t (&r)[4], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void Load2(uint32_t (&r)[2], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void Load4Transpose(uint32_t (&r)[4], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void Load2Transpose(uint32_t (&r)[2], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}
__device__ __forceinline__ void Mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ inline void Copy(void *dst, const void *src, bool valid) {
    unsigned address = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
}
constexpr int Heads = 8, Dim = 192, Values = 128, Splits = 16, Threads = 384;
constexpr int Partial = Splits * Heads * Values;
constexpr int Scratch = Partial + Splits * Heads * 2;
template<int Tokens = 128, int GroupHeads = Heads> struct StorageFor {
    BF16 query[GroupHeads][Dim + 8];
    union {
        BF16 key[Tokens][Dim + 8];
        BF16 probability[GroupHeads][Tokens + 8];
    };
    BF16 value[Tokens][136];
    // Offset successive 16-head rows across banks in the warp softmax.
    float scores[Tokens][GroupHeads + (GroupHeads == 16 ? 1 : 0)];
    int valid[Tokens];
};

using Storage = StorageFor<>;

// Compact QK/PV fragments derive from NaiveRT's qk_eight_heads/pv_eight_heads.
// The legacy orientation remains available for callers requiring its exact order.
template <bool Fast> __device__ __forceinline__ float Exponential(float x) {
    if constexpr (Fast)
        return __expf(x);
    else
        return expf(x);
}
// Split phases let a cooperative caller schedule more tasks than resident
// CTAs. Its grid join must publish all partials before any Combine call.
enum class Phase { Fused, Produce, Combine };

struct NoOutputEpilogue {
    __device__ __forceinline__ void operator()(int, int, BF16) const {}
};
// Expand one stored KV block directly into the MMA operand tile. FP4 payload
// and its E4M3 scale belong to the same physical row.
template<int Bits> __device__ __forceinline__ void LoadStoredKV(
        BF16 *dst, const void *base, int physical, int stride, int column) {
    uint4 values[2]{};
    if (physical >= 0) {
        if constexpr (Bits == 4) {
            const auto *row = (const uint8_t *)base + size_t(physical) * (stride / 16 * 9);
            FastllmNaiveKVRow(row, false, row + stride / 2, column).LoadFP4Pairs<8>((unsigned *)values);
        } else {
            auto src = FastllmNaiveKVRow((const uint8_t *)base + size_t(physical) * stride + column, true);
            values[0] = src.Load8();
            values[1] = (src + 8).Load8();
        }
    }
    reinterpret_cast<uint4 *>(dst)[0] = values[0];
    reinterpret_cast<uint4 *>(dst)[1] = values[1];
}
template <bool Compact = false, Phase Mode = Phase::Fused, int MaxSplits = Splits, int CombineColumns = 64, int Tokens = 128,
          bool CachePhysical = false, bool FullSelection = false, class OutputEpilogue = NoOutputEpilogue, int GroupHeads = Heads, bool LocalPartial = false, int KVBits = 16>
__device__ __forceinline__ void Attend(const BF16 *query, FastllmNaiveDSARow descriptor, const float *sink,
                                       BF16 *output, float *scratch, unsigned *completion, int heads,
                                       int kvHeads, int row, int group, int split, StorageFor<Tokens, GroupHeads> &s,
                                       OutputEpilogue epilogue = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(GroupHeads == 8 || (GroupHeads == 16 && Compact && Mode != Phase::Fused));
    constexpr int Heads = GroupHeads, Partial = Splits * Heads * Values, Scratch = Partial + Splits * Heads * 2;
    static_assert(Mode == Phase::Fused || Compact, "split production/fold uses compact arithmetic");
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int h0 = group * Heads, kh = h0 / (heads / kvHeads);
    const int work = row * (heads / Heads) + group;
    const auto r = FastllmNaiveResolveDSARow(descriptor);
    const int length = max(0, min(r.capacity, r.liveKeys ? *r.liveKeys : r.length));
    const int count = length ? (FullSelection ? 2048 : (r.indices ? r.selectedCount : length)) : 0;
    if (!count) {
        if constexpr (!std::is_same_v<OutputEpilogue, NoOutputEpilogue>) {
            if (t < CombineColumns)
                epilogue(row, h0 * Values + split * CombineColumns + t, __float2bfloat16(0.f));
        }
        if (!split)
            for (int i = t; i < Heads * Values; i += Threads)
                output[(size_t)row * heads * Values + h0 * Values + i] = __float2bfloat16(0.f);
        return;
    }
    const int tiles = (count + 15) / 16;
    const int active = min(tiles, count <= 512 ? min(8, MaxSplits) : MaxSplits);
    static_assert(!LocalPartial || Mode == Phase::Produce);
    const int destSplit = LocalPartial ? 0 : split;
    auto *part = LocalPartial ? scratch : scratch + (size_t)work * Scratch;
    auto *stats = part + (LocalPartial ? Heads * Values : Partial);
    if constexpr (!Compact) {
        if (split >= active)
            return;
    }
    if (Mode != Phase::Combine && split < active) {
        const int begin = tiles * split / active * 16;
        const int end = tiles * (split + 1) / active * 16;
        const int tokens = end - begin;

        // A selected token supplies 24 K vectors and 16 V vectors. Resolve
        // each page table once and share its physical row across these loads.
        __shared__ int physicalValue[CachePhysical ? Tokens : 1];
        for (int i = t; i < tokens; i += Threads) {
            const int slot = begin + i;
            const int key = slot < count ? (r.indices ? r.indices[slot] : slot) : -1;
            if constexpr (CachePhysical) {
                const bool valid = key >= 0 && key < length;
                s.valid[i] = valid ? FastllmNaivePhysicalRow(r, key) : -1;
                physicalValue[i] = valid ? FastllmNaivePhysicalRow(r, key, true) : -1;
            } else {
                s.valid[i] = key >= 0 && key < length ? key : -1;
            }
        }
        __syncthreads();
        if constexpr (KVBits == 8) {
            static_assert(CachePhysical);
            for (int i = t; i < tokens * 12; i += Threads) {
                const int row = i / 12, d = i % 12 * 16, physical = s.valid[row];
                const auto *src = (const uint8_t *)r.key + size_t(max(0, physical)) * r.keyStride + kh * Dim + d;
                Copy(&s.key[row][d], src, physical >= 0);
            }
            for (int i = t; i < tokens * 8; i += Threads) {
                const int row = i / 8, d = i % 8 * 16, physical = physicalValue[row];
                const auto *src = (const uint8_t *)r.value + (size_t(max(0, physical)) * kvHeads + kh) * Values + d;
                Copy(&s.value[row][d], src, physical >= 0);
            }
        } else if constexpr (KVBits == 4) {
            static_assert(CachePhysical && (KVBits == 4 || KVBits == 8));
            for (int i = t; i < tokens * 12; i += Threads) {
                int row = i / 12, d = i % 12 * 16;
                LoadStoredKV<KVBits>(&s.key[row][d], r.key, s.valid[row], r.keyStride, kh * Dim + d);
            }
            for (int i = t; i < tokens * 8; i += Threads) {
                int row = i / 8, d = i % 8 * 16;
                LoadStoredKV<KVBits>(&s.value[row][d], r.value, physicalValue[row], kvHeads * Values, kh * Values + d);
            }
        } else {
            for (int i = t; i < tokens * 24; i += Threads) {
                const int k = s.valid[i / 24], d = i % 24 * 8;
                const auto *src = static_cast<const BF16 *>(r.key);
                if (k >= 0)
                    src += size_t(CachePhysical ? k : FastllmNaivePhysicalRow(r, k)) * r.keyStride + kh * Dim + d;
                Copy(&s.key[i / 24][d], src, k >= 0);
            }
            for (int i = t; i < tokens * 16; i += Threads) {
                const int k = CachePhysical ? physicalValue[i / 16] : s.valid[i / 16], d = i % 16 * 8;
                const auto *src = static_cast<const BF16 *>(r.value);
                if (k >= 0)
                    src += size_t(CachePhysical ? k : FastllmNaivePhysicalRow(r, k, true)) * kvHeads * Values + kh * Values + d;
                Copy(&s.value[i / 16][d], src, k >= 0);
            }
        }
        for (int i = t; i < Heads * 24; i += Threads)
            Copy(&s.query[i / 24][i % 24 * 8], query + ((size_t)row * heads + h0 + i / 24) * Dim + i % 24 * 8,
                 true);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        if constexpr (KVBits == 8) {
            auto expand = [](BF16 *dst) {
                const uint4 raw = *reinterpret_cast<uint4 *>(dst);
                reinterpret_cast<uint4 *>(dst)[0] = make_uint4(
                    FastllmNaiveKVRow::ConvertPair(raw.x), FastllmNaiveKVRow::ConvertPair(raw.x >> 16),
                    FastllmNaiveKVRow::ConvertPair(raw.y), FastllmNaiveKVRow::ConvertPair(raw.y >> 16));
                reinterpret_cast<uint4 *>(dst)[1] = make_uint4(
                    FastllmNaiveKVRow::ConvertPair(raw.z), FastllmNaiveKVRow::ConvertPair(raw.z >> 16),
                    FastllmNaiveKVRow::ConvertPair(raw.w), FastllmNaiveKVRow::ConvertPair(raw.w >> 16));
            };
            for (int i = t; i < tokens * 12; i += Threads) expand(&s.key[i / 12][i % 12 * 16]);
            for (int i = t; i < tokens * 8; i += Threads) expand(&s.value[i / 8][i % 8 * 16]);
        }
        __syncthreads();
        if constexpr (Compact) {
#pragma unroll
            for (int wave = 0; wave < (Tokens + 191) / 192; ++wave) {
              const int tile = warp + wave * (Threads / 32);
              if (tile < tokens / 16) {
                float acc[Heads / 8][4] = {};
#pragma unroll
                for (int k = 0; k < Dim; k += 16) {
                    uint32_t a[4], b[2];
                    const int matrix = lane >> 3;
                    Load4(a, &s.key[tile * 16 + (lane & 7) + ((matrix & 1) << 3)][k + ((matrix >> 1) << 3)]);
                    for (int half = 0; half < Heads / 8; ++half) {
                        Load2(b, &s.query[half * 8 + (lane & 7)][k + (((lane >> 3) & 1) << 3)]);
                        Mma(acc[half], a, b);
                    }
                }
                const int token = tile * 16 + (lane >> 2), head = (lane & 3) * 2;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int row = token + (j >= 2 ? 8 : 0);
                    for (int half = 0; half < Heads / 8; ++half)
                        s.scores[row][half * 8 + head + (j & 1)] =
                            s.valid[row] >= 0 ? acc[half][j] * rsqrtf(float(Dim)) : -INFINITY;
                }
            }
            }
        } else {
            if (warp < tokens / 16) {
                float acc[2][4] = {};
#pragma unroll
                for (int k = 0; k < Dim; k += 16) {
                    uint32_t a[4];
                    // The upper eight MMA rows duplicate the lower eight and are discarded.
                    Load4(a, &s.query[lane & 7][k + (lane >> 4) * 8]);
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        uint32_t b[2];
                        Load2(b, &s.key[warp * 16 + n * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                        Mma(acc[n], a, b);
                    }
                }
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int token = warp * 16 + n * 8 + (lane & 3) * 2;
                    // Keep FP32 QK accumulators through scaling. The only probability
                    // rounding is BF16 immediately before the PV Tensor Core operation.
                    for (int j = 0; j < 2; ++j)
                        s.scores[token + j][lane >> 2] =
                            s.valid[token + j] >= 0 ? acc[n][j] * rsqrtf(float(Dim)) : -INFINITY;
                }
            }
        }
        __syncthreads(); // key storage is now free for probabilities.
        if (warp < Heads / 2) {
            const int h = warp * 2 + (lane & 1), token = lane / 2;
            float maximum = -INFINITY;
            for (int k = token; k < tokens; k += 16)
                maximum = fmaxf(maximum, s.scores[k][h]);
            for (int d = 16; d >= 2; d /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
            float sum = 0;
            for (int k = token; k < tokens; k += 16) {
                BF16 p = __float2bfloat16(
                    maximum == -INFINITY ? 0.f : Exponential<Compact>(s.scores[k][h] - maximum));
                s.probability[h][k] = p;
                sum += __bfloat162float(p);
            }
            for (int d = 16; d >= 2; d /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, d);
            if (lane < 2) {
                stats[(destSplit * Heads + h) * 2] = maximum;
                stats[(destSplit * Heads + h) * 2 + 1] = sum;
            }
        }
        __syncthreads();
        if constexpr (Compact) {
            if (warp < 8) {
                float acc[Heads / 8][4] = {};
                for (int k = 0; k < tokens; k += 16) {
                    uint32_t a[4], b[2];
                    const int matrix = lane >> 3;
                    Load4Transpose(
                        a, &s.value[k + (lane & 7) + ((matrix >> 1) << 3)][warp * 16 + ((matrix & 1) << 3)]);
                    for (int half = 0; half < Heads / 8; ++half) {
                        Load2(b, &s.probability[half * 8 + (lane & 7)][k + (((lane >> 3) & 1) << 3)]);
                        Mma(acc[half], a, b);
                    }
                }
                const int dimension = warp * 16 + (lane >> 2), head = (lane & 3) * 2;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    for (int half = 0; half < Heads / 8; ++half)
                        part[(destSplit * Heads + half * 8 + head + (j & 1)) * Values + dimension + (j >= 2 ? 8 : 0)] = acc[half][j];
            }
        } else {
            if (warp < 8) {
                float acc[2][4] = {};
                for (int k = 0; k < tokens; k += 16) {
                    uint32_t a[4];
                    Load4(a, &s.probability[lane & 7][k + (lane >> 4) * 8]);
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        uint32_t b[2];
                        Load2Transpose(b, &s.value[k + (lane & 15)][warp * 16 + n * 8]);
                        Mma(acc[n], a, b);
                    }
                }
#pragma unroll
                for (int n = 0; n < 2; ++n)
                    for (int j = 0; j < 2; ++j)
                        part[(destSplit * Heads + (lane >> 2)) * Values + warp * 16 + n * 8 + (lane & 3) * 2 +
                             j] = acc[n][j];
            }
        }
        if constexpr (Compact) {
            __syncthreads();
            if constexpr (Mode == Phase::Fused) {
                if (!t) {
                    cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
                    counter.fetch_add(1, cuda::memory_order_acq_rel);
                }
            }
        } else {
            // Publish every lane's partial before the CTA's release operation. The
            // acquire/release chain makes all preceding splits visible to the last CTA.
            __syncthreads();
            __shared__ int last;
            if (!t) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
                last = counter.fetch_add(1, cuda::memory_order_acq_rel) == unsigned(active - 1);
            }
            __syncthreads();
            if (!last)
                return;
            if (warp < 8) {
                float maximum = lane < active ? stats[(lane * Heads + warp) * 2] : -INFINITY;
                for (int d = 16; d; d /= 2)
                    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
                maximum = __shfl_sync(0xffffffff, maximum, 0);
                if (sink)
                    maximum = fmaxf(maximum, sink[h0 + warp]);
                // Reuse score storage; each head has its own sixteen correction factors.
                float *correction = reinterpret_cast<float *>(s.scores) + warp * Splits;
                if (lane < active)
                    correction[lane] = maximum == -INFINITY
                                           ? 0.f
                                           : Exponential<Compact>(stats[(lane * Heads + warp) * 2] - maximum);
                __syncwarp();
                float denominator = 0;
                if (!lane) {
                    for (int k = 0; k < active; ++k)
                        denominator = fmaf(stats[(k * Heads + warp) * 2 + 1], correction[k], denominator);
                    if (sink)
                        denominator += Exponential<Compact>(sink[h0 + warp] - maximum);
                }
                denominator = __shfl_sync(0xffffffff, denominator, 0);
                for (int d = lane; d < Values; d += 32) {
                    float v = 0;
                    for (int k = 0; k < active; ++k)
                        v = fmaf(part[(k * Heads + warp) * Values + d], correction[k], v);
                    output[((size_t)row * heads + h0 + warp) * Values + d] =
                        __float2bfloat16(denominator > 0 ? v / denominator : 0.f);
                }
            }
            __syncthreads();
            if (!t) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
                counter.store(0, cuda::memory_order_release);
            }
        }
    } // active split
    if constexpr (Mode == Phase::Produce) {
        __syncthreads();
        return;
    }
    if constexpr (Compact) {
        // All sixteen CTAs now combine one (head, half-value) tile each, following
        // NaiveRT's distributed combine rather than concentrating the work in one CTA.
        if constexpr (Mode == Phase::Fused) {
            if (!t) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
                while (counter.load(cuda::memory_order_acquire) < unsigned(active))
                    __nanosleep(64);
            }
        }
        __syncthreads();
        static_assert(CombineColumns == 64 || CombineColumns == 128);
        constexpr int Parts = Values / CombineColumns;
        const int head = split / Parts, dimensionBegin = (split % Parts) * CombineColumns;
        float *received = reinterpret_cast<float *>(&s), *correction = received + Splits * CombineColumns;
        for (int i = t; i < active * (CombineColumns / 4); i += Threads) {
            int k = i / (CombineColumns / 4), dim = i % (CombineColumns / 4) * 4;
            Copy(received + k * CombineColumns + dim, part + (k * Heads + head) * Values + dimensionBegin + dim, true);
        }
        asm volatile("cp.async.commit_group;cp.async.wait_group 0;" ::: "memory");
        if (warp == 0) {
            float maximum = lane < active ? stats[(lane * Heads + head) * 2] : -INFINITY;
            for (int d = 16; d; d /= 2)
                maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
            maximum = __shfl_sync(0xffffffff, maximum, 0);
            if (sink)
                maximum = fmaxf(maximum, sink[h0 + head]);
            if (lane < active)
                correction[lane] = maximum == -INFINITY
                                       ? 0.f
                                       : Exponential<Compact>(stats[(lane * Heads + head) * 2] - maximum);
            __syncwarp();
            if (!lane) {
                float denominator = 0.f;
                for (int k = 0; k < active; ++k)
                    denominator = fmaf(stats[(k * Heads + head) * 2 + 1], correction[k], denominator);
                if (sink)
                    denominator += Exponential<Compact>(sink[h0 + head] - maximum);
                correction[Splits] = denominator;
            }
        }
        __syncthreads();
        if (t < CombineColumns) {
            float value = 0.f;
#pragma unroll
            for (int k = 0; k < active; ++k)
                value = fmaf(received[k * CombineColumns + t], correction[k], value);
            float denominator = correction[Splits];
            const auto rounded = __float2bfloat16(denominator > 0 ? value / denominator : 0.f);
            output[((size_t)row * heads + h0 + head) * Values + dimensionBegin + t] = rounded;
            epilogue(row, (h0 + head) * Values + dimensionBegin + t, rounded);
        }
    }
#endif
}
} // namespace fastllm_dsa_fused::native_attention
