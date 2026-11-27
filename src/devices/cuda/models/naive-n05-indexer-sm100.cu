// Blackwell Indexer: FP8 UMMA, four-stage paged TMA and two accumulator stages.
// Inspired by the UMMA/TMA and paired FP32 epilogue in NaiveSglang naive_ops_lib.
// Reads existing standalone FP8 or BF16-plus-FP8 KV rows without repacking.
// K and scales share each pipeline stage; rebound bases use cp.async fallback.
#include "naive-n05-indexer-sm100.cuh"
#include <cuda.h>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <map>
namespace naive_indexer_sm100 {
// A verify CTA shares one KV binding across four queries. Only the length,
// query, weights index and destinations vary within the group.
struct VerifyKey {
    FastllmNaiveKV key;
    const void *const *cache;
    int stride;
};
struct VerifyQuery {
    const int *live;
    __nv_bfloat16 *query;
    float *scale;
    unsigned *scores;
    int sequence;
    __device__ int Count() const { return *live; }
};
template<int Queries> struct ScoreRows {
    Row rows[BatchLimit];
    __device__ Row Key(int owner) const { return rows[owner]; }
    void Bind(const Row *source, int count) { std::copy_n(source, count, rows); }
};
template<> struct ScoreRows<VerifyQueries> {
    VerifyQuery rows[VerifyLimit];
    VerifyKey owners[VerifyLimit / VerifyQueries];
    __device__ VerifyKey Key(int owner) const { return owners[owner]; }
    void Bind(const Row *source, int count) {
        for (int i = 0; i < count; ++i) {
            const auto &r = source[i];
            rows[i] = {r.live, r.query, r.scale, r.scores, r.sequence};
            if (i % VerifyQueries == 0) owners[i / VerifyQueries] = {r.key, r.cache, r.stride};
        }
    }
};
template<int Queries> struct TmaBatch {
    ScoreRows<Queries> bindings;
    CUtensorMap map, scaleMap;
    const void *base;
    int rowBytes;
    bool scaled;
};
static_assert(sizeof(TmaBatch<VerifyQueries>) + 128 <= 32764);
__device__ unsigned address(const void *p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ uint64_t descriptor(void *p) {
    cute::UMMA::SmemDescriptor d{};
    d.version_ = 1;
    d.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
    d.start_address_ = address(p) >> 4;
    d.stride_byte_offset_ = 1024 >> 4;
    return d;
}
__device__ unsigned OrderedScoreBits(float value) {
    unsigned bits = value == 0.f ? 0u : __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ void LoadVector(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address(dst)), "l"(src),
                 "r"(valid ? 16 : 0)
                 : "memory");
}
template<class Key> __device__ void load_tma(const Key &r, int first, unsigned char *out, const CUtensorMap *map,
                         const CUtensorMap *scaleMap, void *scales, uint64_t *barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        int physical = r.key.PhysicalRow(first);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(address(barrier)),
                     "r"(r.key.scaledIndexer ? 18432 : 16384)
                     : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
                     "{%2, %3}], [%4];" ::"r"(address(out)),
                     "l"(map), "r"(0), "r"(physical), "r"(address(barrier))
                     : "memory");
        if (r.key.scaledIndexer)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
                "{%2, %3}], [%4];" ::"r"(address(scales)),
                "l"(scaleMap), "r"(0), "r"(physical), "r"(address(barrier))
                : "memory");
    }
#endif
}

__device__ void wait_bar(uint64_t *p, int phase) {
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(
            address(p)),
        "r"(phase)
        : "memory");
}
__device__ void arrive_bar(uint64_t *p) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(address(p)) : "memory");
}
struct VerifyStripe { int owner, offset, stride; };

template <int Stages = 4, int Outputs = 2, int Queries = 1>
__global__ __launch_bounds__(256, 2) void Pipeline(const __nv_bfloat16 *weights,
                                                   const __grid_constant__ TmaBatch<Queries> batch,
                                                   const VerifyStripe *schedule = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    VerifyStripe stripe{int(blockIdx.y), int(blockIdx.x), int(gridDim.x)};
    if constexpr (Queries == VerifyQueries) {
        stripe = schedule[blockIdx.x];
        if (stripe.owner < 0) return;
    }
    const auto *rows = batch.bindings.rows + stripe.owner * Queries;
    auto r = batch.bindings.Key(stripe.owner);
    r.key.Bind(r.cache);
    const int begin = stripe.offset * 128, step = stripe.stride * 128;
    int lengths[Queries], end = 0;
#pragma unroll
    for (int j = 0; j < Queries; ++j) {
        lengths[j] = rows[j].Count();
        end = max(end, lengths[j]);
    }
    if (begin >= end) return;
    const int tiles = (end - begin + step - 1) / step, t = threadIdx.x, warp = t / 32, lane = t % 32;
    // 128-byte swizzle repeats every 1 KiB. Align both staging areas so TMA
    // and the cp.async fallback use the same relative XOR permutation.
    __shared__ __align__(1024) unsigned char q[Queries * 16 * 128];
    extern __shared__ __align__(1024) unsigned char data[];
    auto k = reinterpret_cast<unsigned char(*)[128 * 128]>(data);
    // Stage the padded scale tail too: wide KV rows otherwise cause one
    // scattered global load per key in the epilogue.
    __shared__ __align__(128) unsigned char keyScales[Stages][128][16];
    __shared__ __align__(8) float w[Queries][16];
    __shared__ uint32_t tmem;
    __shared__ uint64_t fullK[Stages], emptyK[Stages], fullC[Outputs], emptyC[Outputs];
    bool tma = r.key.base == batch.base && r.key.scaledIndexer == batch.scaled &&
               r.stride * (r.key.fp8 ? 1 : 2) == batch.rowBytes && r.key.pages && r.key.pageLen % 128 == 0 &&
               r.key.offset == 0;
    if (warp == 0) {
        cute::TMEM::Allocator1Sm a;
        a.allocate(Outputs * Queries * 16, &tmem);
        a.release_allocation_lock();
    }
    if (t == 0) {
        for (int i = 0; i < Stages; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address(&fullK[i])),
                         "r"(tma ? 1 : 32));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&emptyK[i])));
            arrive_bar(&emptyK[i]);
        }
        for (int i = 0; i < Outputs; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&fullC[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&emptyC[i])));
            arrive_bar(&emptyC[i]);
        }
    }
    if constexpr (Queries == VerifyQueries) {
        for (int i = t; i < Queries * 16 * 8; i += 256) {
            const int row = i / 8, col = i % 8;
            const auto *packed = reinterpret_cast<const uint4 *>(rows[row / 16].query);
            reinterpret_cast<uint4 *>(q)[row * 8 + (col ^ (row & 7))] =
                lengths[row / 16] > 0 ? packed[(row % 16) * 8 + col] : make_uint4(0, 0, 0, 0);
        }
    } else {
        for (int i = t; i < 16 * 128; i += 256) {
            const int row = i / 128, col = i % 128;
            q[row * 128 + (col ^ ((row & 7) * 16))] =
                __nv_cvt_float_to_fp8(float(rows[0].query[i]), __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (t < Queries * 16) {
        const auto &query = rows[t / 16];
        w[t / 16][t % 16] = Queries == 1 || lengths[t / 16] > 0
            ? query.scale[t % 16] * float(weights[query.sequence * 16 + t % 16]) : 0.f;
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        for (int i = 0; i < tiles; ++i) {
            int stage = i % Stages, first = begin + i * step;
            if (lane == 0) wait_bar(&emptyK[stage], (i / Stages) & 1);
            __syncwarp();
            if (tma)
                load_tma(r, first, k[stage], &batch.map, &batch.scaleMap, keyScales[stage], &fullK[stage]);
            else {
                for (int item = lane; item < 128 * 8; item += 32) {
                    int row = item / 8, col = item % 8 * 16;
                    auto src = r.key.IndexRow(min(first + row, end - 1), r.stride);
                    LoadVector(k[stage] + row * 128 + (col ^ ((row & 7) * 16)), (const char *)src.data + col,
                               first + row < end);
                    if ((item / 32) % 8 == 7) asm volatile("cp.async.commit_group;" ::: "memory");
                }
                if (r.key.scaledIndexer)
                    for (int row = lane; row < 128; row += 32) {
                        auto src = r.key.IndexRow(min(first + row, end - 1), r.stride);
                        LoadVector(keyScales[stage][row], (const char *)src.data + 128, first + row < end);
                    }
                asm volatile("cp.async.commit_group;" ::: "memory");
                asm volatile("cp.async.wait_all;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                arrive_bar(&fullK[stage]);
            }
        }
    } else if (warp == 1) {
        const uint64_t id =
            cute::UMMA::make_runtime_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, Queries * 16,
                                                cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto qdesc = descriptor(q);
        for (int i = 0; i < tiles; ++i) {
            int stage = i % Stages, out = i % Outputs;
            if (lane == 0) {
                wait_bar(&fullK[stage], (i / Stages) & 1);
                wait_bar(&emptyC[out], (i / Outputs) & 1);
            }
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            auto kd = descriptor(k[stage]);
#pragma unroll
            for (int d = 0; d < 128; d += 32)
                cute::SM100_MMA_F8F6F4_SS::fma(kd + d / 16, qdesc + d / 16, tmem + out * Queries * 16, d != 0, id);
            if (lane == 0) {
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                        address(&fullC[out]))
                    : "memory");
            }
        }
    } else if (warp >= 4) {
        const int row = t - 128;
        for (int i = 0; i < tiles; ++i) {
            int out = i % Outputs, first = begin + i * step;

            wait_bar(&fullC[out], (i / Outputs) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            float ks = r.key.scaledIndexer ? *(float *)keyScales[i % Stages][row] : 1.f;
#pragma unroll
            for (int j = 0; j < Queries; ++j) {
                uint32_t v[16];
                cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
                    tmem + (out * Queries + j) * 16 + ((row / 32) << 21),
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (j == Queries - 1) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    // Release K only after all scale reads and TMEM loads have completed.
                    if (row == 0) {
                        arrive_bar(&emptyC[out]);
                        arrive_bar(&emptyK[i % Stages]);
                    }
                }
                const auto *headWeights = reinterpret_cast<const uint64_t *>(w[j]);
                uint64_t values[8];
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    uint64_t x = uint64_t(v[h * 2]) | (uint64_t(v[h * 2 + 1]) << 32),
                             abs = x & 0x7fffffff7fffffffull;
                    asm volatile("add.f32x2 %0, %1, %2;" : "=l"(values[h]) : "l"(x), "l"(abs));
                    asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(values[h]) : "l"(values[h]), "l"(headWeights[h]));
                }
#pragma unroll
                for (int d = 4; d; d /= 2)
#pragma unroll
                    for (int h = 0; h < d; ++h)
                        asm volatile("add.f32x2 %0, %1, %2;"
                                     : "=l"(values[h])
                                     : "l"(values[h]), "l"(values[h + d]));
                float sum =
                    (__uint_as_float(uint32_t(values[0])) + __uint_as_float(uint32_t(values[0] >> 32))) * 0.5f;
                if (first + row < lengths[j]) rows[j].scores[first + row] = OrderedScoreBits(sum * ks);
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        cute::TMEM::Allocator1Sm a;
        a.free(tmem, Outputs * Queries * 16);
    }
#endif
}

// Pack Q into E4M3 once per head. Its scale is the same as the BF16-expanded
// quantizer used by the fallback. Eight warps handle eight independent heads.
__device__ void PackHead(const __nv_bfloat16 *src, unsigned *packed, float *scales, int head) {
    const int lane = threadIdx.x % 32;
    union { uint2 vector; __nv_bfloat16 values[4]; } v;
    v.vector = reinterpret_cast<const uint2 *>(src)[size_t(head) * 32 + lane];
    float maximum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) maximum = fmaxf(maximum, fabsf(float(v.values[i])));
    #pragma unroll
    for (int delta = 16; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    const float scale = fmaxf(maximum, 1.e-4f) / 448.f;
    unsigned word = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float value = fmaxf(-448.f, fminf(448.f, float(v.values[i]) / scale));
        word |= unsigned(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)) << (i * 8);
    }
    packed[size_t(head) * 32 + lane] = word;
    if (!lane) scales[head] = scale;
}

__global__ void PackPrefillQueries(const __nv_bfloat16 *src, unsigned *packed, float *scales, int heads) {
    const int head = blockIdx.x * 8 + threadIdx.x / 32;
    if (head < heads) PackHead(src, packed, scales, head);
}

// The first CTA shares this launch with Q packing. It partitions key tiles
// using current device lengths, including graph replays with different requests.
__global__ void PrepareVerify(const __grid_constant__ TmaBatch<VerifyQueries> batch,
                              const __nv_bfloat16 *input, int owners, int blocks, int uniformStripes,
                              VerifyStripe *schedule) {
    if (blockIdx.y) {
        const auto &r = batch.bindings.rows[blockIdx.y - 1];
        if (*r.live > 0) PackHead(input + size_t(r.sequence) * 16 * 128, reinterpret_cast<unsigned *>(r.query),
                                   r.scale, blockIdx.x * 8 + threadIdx.x / 32);
        return;
    }
    if (blockIdx.x) return;
    const int t = threadIdx.x;
    int tiles = 0;
    if (t < owners) {
        for (int j = 0; j < VerifyQueries; ++j) tiles = max(tiles, batch.bindings.rows[t * VerifyQueries + j].Count());
        tiles = (tiles + 127) / 128;
    }
    __shared__ int firstLength;
    if (!t) firstLength = tiles;
    __syncthreads();
    if (!__syncthreads_or(t < owners && tiles != firstLength)) {
        for (int i = t; i < blocks; i += blockDim.x) {
            const int owner = i / uniformStripes;
            schedule[i] = {owner < owners ? owner : -1, i % uniformStripes, uniformStripes};
        }
        return;
    }
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage storage;
    int before, total;
    Scan(storage).ExclusiveSum(tiles, before, total);
    if (t < owners) {
        // One stripe per owner; distribute all remaining stripes by tile count.
        const int extra = blocks - owners;
        const int first = t + before * extra / total;
        const int end = t + 1 + (before + tiles) * extra / total;
        for (int i = first; i < end; ++i) schedule[i] = {t, i - first, end - first};
    }
}

__global__ void GatherPrefillIndexKeys(FastllmNaiveKV cache, int stride, int first, int count, uint4 *packed, float *scales) {
    const int row = blockIdx.x * 16 + threadIdx.x / 8, lane = threadIdx.x % 8;
    if (row >= count) return;
    const auto *src = static_cast<const uint4 *>(cache.IndexRow(first + row, stride).data);
    packed[size_t(row) * 8 + lane] = src[lane];
    if (cache.scaledIndexer && lane == 0) scales[row] = cache.IndexScale(first + row, stride);
}

// Eight groups of 16 query heads form the N dimension of one 128 x 128
// UMMA. Each CTA reuses that Q tile across up to eight consecutive K tiles.
constexpr int PrefillQueries = 8, PrefillKeys = 128, PrefillTiles = 8;
__global__ __launch_bounds__(128) void PrefillScores(const unsigned char *queries, const float *queryScales,
        const __nv_bfloat16 *weights, const unsigned char *keys, const float *keyScales, float *scores,
        int rows, int cols, int past, int keyOffset, int keyCount) {
    const int t = threadIdx.x, firstQ = PrefillQueries * blockIdx.y;
    const int firstK = PrefillKeys * PrefillTiles * blockIdx.x;
    const int end = min(keyCount, past + min(rows, firstQ + PrefillQueries) - keyOffset);
    const int liveTiles = min(PrefillTiles, (max(0, end - firstK) + PrefillKeys - 1) / PrefillKeys);
    // Preserve the dense score contract for both the ordered and bitmap selectors.
    for (int n = liveTiles; n < PrefillTiles; ++n) {
        const int col = firstK + n * PrefillKeys + t;
        if (col >= keyCount) break;
        #pragma unroll
        for (int i = 0; i < PrefillQueries; ++i)
            if (firstQ + i < rows) scores[size_t(firstQ + i) * cols + keyOffset + col] = -INFINITY;
    }
    if (!liveTiles) return;
    __shared__ __align__(1024) unsigned char q[128 * 128], k[128 * 128];
    __shared__ __align__(16) float factors[128];
    __shared__ uint64_t barrier;
    __shared__ uint32_t tmem;
    if (t < 32) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.allocate(128, &tmem);
        allocator.release_allocation_lock();
    }
    if (!t) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address(&barrier)));
    const size_t head = size_t(firstQ) * 16 + t;
    // Per-head activation scales are nonnegative, so they can be folded
    // into the weight before the paired FP32 dot-product reduction.
    factors[t] = firstQ + t / 16 < rows ? float(weights[head]) * queryScales[head] : 0.f;
    for (int j = t; j < 128 * 8; j += 128) {
        const int h = j / 8, offset = j % 8 * 16, row = firstQ + h / 16;
        LoadVector(q + h * 128 + (offset ^ ((h & 7) * 16)),
            queries + (size_t(min(rows - 1, row)) * 16 + h % 16) * 128 + offset, row < rows);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    const uint64_t instruction = cute::UMMA::make_runtime_instr_desc<cutlass::float_e4m3_t,
        cutlass::float_e4m3_t, float, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>();
    for (int tile = 0; tile < liveTiles; ++tile) {
        const int first = firstK + tile * PrefillKeys;
        for (int j = t; j < 128 * 8; j += 128) {
            const int h = j / 8, offset = j % 8 * 16;
            LoadVector(k + h * 128 + (offset ^ ((h & 7) * 16)),
                keys + size_t(min(first + h, keyCount - 1)) * 128 + offset, first + h < keyCount);
        }
        asm volatile("cp.async.commit_group; cp.async.wait_all; fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (t < 32) {
            const auto kd = descriptor(k), qd = descriptor(q);
            #pragma unroll
            for (int d = 0; d < 128; d += 32)
                cute::SM100_MMA_F8F6F4_SS::fma(kd + d / 16, qd + d / 16, tmem, d != 0, instruction);
            if (!t) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(address(&barrier)) : "memory");
        }
        wait_bar(&barrier, tile & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const int col = first + t;
        const float ks = keyScales && col < keyCount ? keyScales[col] : 1.f;
        #pragma unroll
        for (int i = 0; i < PrefillQueries; ++i) {
            uint32_t v[16];
            cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(tmem + i * 16 + ((t / 32) << 21),
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint64_t sum2 = 0;
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                uint32_t lo = __float_as_uint(fmaxf(__uint_as_float(v[2 * h]), 0.f));
                uint32_t hi = __float_as_uint(fmaxf(__uint_as_float(v[2 * h + 1]), 0.f));
                uint64_t values = (uint64_t(hi) << 32) | lo;
                uint64_t factors2 = reinterpret_cast<const uint64_t *>(factors + i * 16)[h];
                asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(sum2) : "l"(values), "l"(factors2), "l"(sum2));
            }
            float sum = __uint_as_float(uint32_t(sum2)) + __uint_as_float(uint32_t(sum2 >> 32));
            if (firstQ + i < rows && col < keyCount)
                scores[size_t(firstQ + i) * cols + keyOffset + col] =
                    keyOffset + col <= past + firstQ + i ? sum * ks : -INFINITY;
        }
        __syncthreads();
    }
    if (t < 32) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.free(tmem, 128);
    }
}

template<int Queries>
bool Score(const Row *rows, int count, int capacity, const __nv_bfloat16 *weights, cudaStream_t stream,
           const __nv_bfloat16 *input = nullptr, void *scratch = nullptr, size_t scratchBytes = 0) {
    constexpr int limit = Queries == 1 ? BatchLimit : VerifyLimit;
    if (!rows || !weights || count < 32 || count > limit || count % Queries || capacity < 4096) return false;
    for (int i = 0; i < count; ++i) {
        const auto &r = rows[i];
        const int rowBytes = r.stride * (r.key.fp8 ? 1 : 2);
        if ((!r.key.scaledIndexer && !r.key.fp8) || r.key.fp4 || r.key.offset || !r.key.pages ||
            r.key.pageLen <= 0 || r.key.pageLen % 128 || !r.key.base || !r.live || !r.query || !r.scale || !r.scores ||
            rowBytes < (r.key.scaledIndexer ? 144 : 128) || rowBytes % 16 || (uintptr_t)r.key.base % 16)
            return false;
    }
    constexpr int shared = 4 * 128 * 128;
    struct Support {
        bool ok;
        int sms;
        const void *base = nullptr;
        int rowBytes = 0;
        bool scaled = false;
        CUtensorMap map{}, scaleMap{};
    };
    static thread_local std::map<int, Support> devices;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    auto entry = devices.find(device);
    if (entry == devices.end()) {
        cudaDeviceProp prop{};
        cudaFuncAttributes attr{};
        bool ok = cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.major == 10 &&
                  (prop.minor == 0 || prop.minor == 3) &&
                  cudaFuncGetAttributes(&attr, Pipeline<4, 2, Queries>) == cudaSuccess && attr.binaryVersion >= 100 &&
                  cudaFuncSetAttribute(Pipeline<4, 2, Queries>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) ==
                      cudaSuccess;
        if (!ok) cudaGetLastError();
        entry = devices.emplace(device, Support{ok, prop.multiProcessorCount}).first;
    }
    if (!entry->second.ok) return false;
    // A descriptor is immutable in a graph. If its KV base is rebound at replay,
    // the loader uses cp.async with the live binding instead of the old TMA map.
    auto &cached = entry->second;
    const int rowBytes = rows[0].stride * (rows[0].key.fp8 ? 1 : 2);
    const bool scaled = rows[0].key.scaledIndexer;
    if (cached.base != rows[0].key.base || cached.rowBytes != rowBytes || cached.scaled != scaled) {
        CUtensorMap map{};
        cuuint64_t shape[] = {128, 2147483647}, strides[] = {(cuuint64_t)rowBytes};
        cuuint32_t box[] = {128, 128}, steps[] = {1, 1};
        void *base = (char *)rows[0].key.base + rowBytes - (scaled ? 144 : 128);
        if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, shape, strides, box, steps,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return false;
        CUtensorMap scaleMap{};
        shape[0] = 16;
        box[0] = 16;
        if (scaled && cuTensorMapEncodeTiled(&scaleMap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (char *)base + 128,
                                             shape, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                             CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return false;
        cached.base = rows[0].key.base;
        cached.rowBytes = rowBytes;
        cached.scaled = scaled;
        cached.map = map;
        cached.scaleMap = scaleMap;
    }
    TmaBatch<Queries> batch{};
    batch.bindings.Bind(rows, count);
    batch.map = cached.map;
    batch.scaleMap = cached.scaleMap;
    batch.base = rows[0].key.base;
    batch.rowBytes = rowBytes;
    batch.scaled = scaled;
    const int owners = count / Queries;
    const int maxStripes = std::min((capacity + 127) / 128, std::max(1, 2 * cached.sms / owners));
    int stripes = 1;
    while (2 * stripes <= maxStripes) stripes *= 2;
    if constexpr (Queries == VerifyQueries) {
        const int tasks = static_cast<int>(std::min(scratchBytes / sizeof(VerifyStripe), size_t(2 * cached.sms)));
        if (!input || (reinterpret_cast<uintptr_t>(input) & 7) || !scratch ||
            reinterpret_cast<uintptr_t>(scratch) % alignof(VerifyStripe) || tasks < owners * stripes) return false;
        for (int i = 0; i < count; ++i)
            if (reinterpret_cast<uintptr_t>(rows[i].query) & 15) return false;
        auto *schedule = static_cast<VerifyStripe *>(scratch);
        PrepareVerify<<<dim3(2, count + 1), 256, 0, stream>>>(batch, input, owners, tasks, stripes, schedule);
        if (cudaGetLastError() != cudaSuccess) return false;
        Pipeline<4, 2, Queries><<<tasks, 256, shared, stream>>>(weights, batch, schedule);
    } else {
        Pipeline<4, 2, Queries><<<dim3(stripes, owners), 256, shared, stream>>>(weights, batch);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace naive_indexer_sm100

bool FastllmCudaNaiveIndexerSm100(const naive_indexer_sm100::Row *rows, int count, int capacity,
                                  const __nv_bfloat16 *weights, cudaStream_t stream) {
    return naive_indexer_sm100::Score<1>(rows, count, capacity, weights, stream);
}

bool FastllmCudaNaiveVerifyIndexerSm100(const naive_indexer_sm100::VerifyBatch &batch, int count, int capacity,
        const __nv_bfloat16 *query, const __nv_bfloat16 *weights, void *scratch, size_t scratchBytes,
        cudaStream_t stream) {
    using namespace naive_indexer_sm100;
    return Score<VerifyQueries>(batch.rows, count, capacity, weights, stream, query, scratch, scratchBytes);
}

bool FastllmCudaNaivePrefillIndexerSm100(const __nv_bfloat16 *query, const __nv_bfloat16 *weights,
        FastllmNaiveKV cache, int stride, int rows, int keys, int past, void *scratch, size_t scratchBytes,
        float *queryScales, float *scores, cudaStream_t stream) {
    using namespace naive_indexer_sm100;
    if (!query || !weights || !scratch || !queryScales || !scores || stride <= 0 ||
        rows < 128 || keys < 1024 || rows > 65535 * PrefillQueries || past < 0 ||
        int64_t(past) + rows > keys || !cache.base || cache.fp4 || (!cache.scaledIndexer && !cache.fp8) ||
        cache.offset || (cache.pages && cache.pageLen <= 0) ||
        ((uintptr_t(query) | uintptr_t(cache.base) | uintptr_t(scratch)) & 15)) return false;
    const size_t rowBytes = size_t(stride) * (cache.fp8 ? 1 : 2);
    if (rowBytes < (cache.scaledIndexer ? FastllmNaiveIndexerBytes : 128) || rowBytes % 16) return false;
    const size_t qBytes = size_t(rows) * 16 * 128;
    const bool gather = cache.pages || rowBytes != 128 || cache.scaledIndexer;
    if (scratchBytes < qBytes) return false;
    int blockKeys = keys;
    if (gather) {
        const size_t packedRowBytes = 128 + (cache.scaledIndexer ? sizeof(float) : 0);
        blockKeys = std::min<size_t>(keys, (scratchBytes - qBytes) / packedRowBytes);
        if (blockKeys < keys) blockKeys -= blockKeys % PrefillKeys;
        if (!blockKeys) return false;
    }
    // An SM-specific fatbin may not contain the active Blackwell variant.
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, PrefillScores) != cudaSuccess || attr.binaryVersion < 100) {
        cudaGetLastError();
        return false;
    }
    auto *packedQ = static_cast<unsigned char *>(scratch);
    auto *packedK = packedQ + qBytes;
    auto *keyScales = cache.scaledIndexer ? reinterpret_cast<float *>(packedK + size_t(blockKeys) * 128) : nullptr;
    PackPrefillQueries<<<(rows * 16 + 7) / 8, 256, 0, stream>>>(query, reinterpret_cast<unsigned *>(packedQ), queryScales, rows * 16);
    // The quantized Q is shared by all K blocks. Dense scores retain their
    // full logical stride so causal masking and ordered selection are unchanged.
    for (int first = 0; first < keys;) {
        int count = std::min(blockKeys, keys - first);
        if (gather) GatherPrefillIndexKeys<<<1 + (count - 1) / 16, 128, 0, stream>>>(
            cache, stride, first, count, reinterpret_cast<uint4 *>(packedK), keyScales);
        dim3 grid(1 + (count - 1) / (PrefillKeys * PrefillTiles), (rows + PrefillQueries - 1) / PrefillQueries);
        PrefillScores<<<grid, 128, 0, stream>>>(packedQ, queryScales, weights,
            gather ? packedK : reinterpret_cast<const unsigned char *>(cache.base), keyScales, scores,
            rows, keys, past, first, count);
        first += count;
    }
    return cudaGetLastError() == cudaSuccess;
}
