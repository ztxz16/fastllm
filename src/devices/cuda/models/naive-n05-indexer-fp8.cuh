#pragma once
#include <algorithm>
#include <cuda.h>
#include <cuda_fp8.h>
namespace native_fp8 {
using BF = __nv_bfloat16;
struct WorkEntry {
    naive_indexer::FP8Entry source, prepared;
    int keyRows;
};
struct Batch {
    WorkEntry entries[8];
    unsigned mask;
    int count;
};
__global__ void PackQueries(Batch batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, entry = blockIdx.y;
    if (!(batch.mask & (1u << entry)))
        return;
    auto e = batch.entries[entry];
    if (i < e.source.rows * 2048)
        ((unsigned char *)e.prepared.q)[i] = __nv_fp8_e4m3((float)e.source.q[i]).__x;
}
// Copy FP8 payloads/scales without dequantization. Reading a peer cache once
// avoids replaying the smaller WGMMA tile's reads across NVLink for every Q tile.
__global__ void StageKeys(Batch batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, entry = blockIdx.y;
    if (!(batch.mask & (1u << entry)))
        return;
    auto e = batch.entries[entry];
    if (i >= e.keyRows * 32)
        return;
    int row = i / 32, d = i % 32 * 4;
    const auto *src = (const unsigned char *)e.source.key.IndexRow(row, e.source.stride).data + d;
    auto *dst = (unsigned char *)e.prepared.key.base + (size_t)row * 144 + d;
    if (((size_t)src & 3) == 0)
        *(unsigned *)dst = *(const unsigned *)src;
    else
        for (int j = 0; j < 4; ++j)
            dst[j] = src[j];
    if (d == 0)
        *(float *)(dst + 128) = e.source.key.IndexScale(row, e.source.stride);
}
constexpr int kQueries = 8, kThreads = 256;
constexpr int kFp8SharedBytes = 128 * 64 * sizeof(float) + 128 * 8;

__device__ __forceinline__ unsigned long long descriptor(const void *p) {
    return ((unsigned)__cvta_generic_to_shared(p) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void mmaFp8(float (&c)[32], unsigned long long a, unsigned long long b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                 "%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
                   "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]),
                   "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]),
                   "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]),
                   "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
                 : "l"(a), "l"(b));
#endif
}
__device__ __forceinline__ int storageOffset(int row, int col) { return row * 128 + (col ^ ((row & 7) * 16)); }
// Fold sixteen heads into the MMA M dimension. Q keeps its original E4M3
// values/scales, K keeps its storage precision, and weighted heads accumulate
// in ascending order. Reuse Q storage for dot products after both groups finish.
__global__ __launch_bounds__(256) void IndexerScores(Batch batch) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    int entry = blockIdx.z;
    if (!(batch.mask & (1u << entry)))
        return;
    const auto &e = batch.entries[entry].prepared;
    const BF *q = e.q, *weights = e.weights;
    FastllmNaiveKV k = e.key;
    const float *qscale = e.scales;
    float *scores = e.scores;
    int rows = e.rows, keys = e.keys, past = e.past;
    if (blockIdx.y * kQueries >= rows)
        return;
    constexpr int Queries = kQueries, Threads = kThreads, M = Queries * 16;
    extern __shared__ __align__(1024) unsigned char buffer[];
    using Storage = unsigned char;
    Storage *sq = (Storage *)buffer, *sk = sq + M * 128;
    constexpr int scratch = M * 64 * sizeof(float);
    float *sc = (float *)(buffer + scratch), *w = sc + M;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wg = tid / 128;
    int firstQ = blockIdx.y * Queries, firstK = blockIdx.x * 64;
    int lastRow = min(rows - 1, firstQ + Queries - 1);
    if (firstK > past + lastRow + (e.part && lastRow >= e.part ? e.gap : 0)) {
        for (int i = tid; i < Queries * 64; i += Threads) {
            int row = firstQ + i / 64, col = firstK + i % 64;
            if (row < rows && col < keys)
                scores[(size_t)row * keys + col] = -INFINITY;
        }
        return;
    }
    // Q has already been quantized to E4M3 and expanded exactly to BF16.
    // Repacking changes storage only; K bytes and scales remain unchanged.
    for (int i = tid; i < M * 8; i += Threads) {
        int row = i / 8, d = i % 8 * 16;
        const auto *src =
            (const unsigned char *)q + ((size_t)min(firstQ + row / 16, rows - 1) * 16 + row % 16) * 128 + d;
        unsigned dst = (unsigned)__cvta_generic_to_shared(sq + storageOffset(row, d));
        int valid = firstQ + row / 16 < rows ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid)
                     : "memory");
    }
    // LaunchFP8 always stages K as local, aligned 144-byte
    // rows. Do not resolve the generic paged/peer cache view again here.
    for (int i = tid; i < 64 * 8; i += Threads) {
        int row = i / 8, d = i % 8 * 16;
        const auto *src = (const unsigned char *)k.base +
                          (size_t)min(firstK + row, batch.entries[entry].keyRows - 1) * 144 + d;
        unsigned dst = (unsigned)__cvta_generic_to_shared(sk + storageOffset(row, d));
        int valid = firstK + row < batch.entries[entry].keyRows ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid)
                     : "memory");
    }
    if (tid < M) {
        int row = firstQ + tid / 16, h = tid % 16;
        sc[tid] = row < rows ? qscale[row * 16 + h] : 0.f;
        w[tid] = row < rows ? (float)weights[row * 16 + h] : 0.f;
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    float dot[32] = {};
    // The explicitly selected FP8 mode uses native FP8 accumulation. Submit
    // all four K tiles before waiting, keeping WGMMA asynchronous.
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int d = 0; d < 128; d += 32)
        mmaFp8(dot, descriptor(sq + storageOffset(wg * 64, d)), descriptor(sk + storageOffset(0, d)));
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
    __syncthreads(); // Every warp group must finish reading the swizzled Q storage.
    float *partial = (float *)sq;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = warp * 16 + lane / 4 + half * 8, col = n * 8 + (lane % 4) * 2;
            *(float2 *)(partial + m * 64 + (col ^ ((m & 7) * 8))) =
                make_float2(dot[n * 4 + half * 2], dot[n * 4 + half * 2 + 1]);
        }
    }
    __syncthreads();
    for (int i = tid; i < Queries * 64; i += Threads) {
        int row = i / 64, col = i % 64;
        float total = 0;
#pragma unroll
        for (int h = 0; h < 16; ++h) {
            int m = row * 16 + h;
            float scaled = partial[m * 64 + (col ^ ((m & 7) * 8))] * sc[m];
            total += fmaxf(scaled, 0.f) * w[m];
        }
        if (firstQ + row < rows && firstK + col < keys)
            scores[(size_t)(firstQ + row) * keys + firstK + col] =
                firstK + col <= past + firstQ + row + (e.part && firstQ + row >= e.part ? e.gap : 0)
                    ? total *
                          *(const float *)((const unsigned char *)k.base + (size_t)(firstK + col) * 144 + 128)
                    : -INFINITY;
    }
#endif
}

} // namespace native_fp8

#include "naive-n05-indexer-fp8-stream.cuh"

namespace naive_indexer {
unsigned LaunchFP8(const FP8Entry *entries, int count, fastllm::Data &scratch, FP8Entry *staged) {
    if (!entries || !staged || count < 1 || count > 8 || FastllmCudaRuntimeArch() != 90 ||
        FastllmCudaGraphIsCapturingFast())
        return 0;
    native_fp8::Batch batch{};
    batch.count = count;
    size_t bytes = 0;
    int maxRows = 0, maxKeys = 0;
    for (int i = 0; i < count; ++i) {
        const auto &e = entries[i];
        if (!(e.key.fp8 || e.key.scaledIndexer) || e.rows < 32 || e.past < 0 || e.rows > 32768 || e.gap < 0 ||
            (e.part && e.part * 2 != e.rows) || (int64_t)e.past + e.rows + e.gap > e.keys)
            continue;
        auto &work = batch.entries[i];
        work.source = work.prepared = e;
        work.keyRows = e.past + e.rows + e.gap;
        bytes += (size_t)e.rows * 2048 + (size_t)work.keyRows * 144;
        maxRows = std::max(maxRows, e.rows);
        maxKeys = std::max(maxKeys, e.keys);
        batch.mask |= 1u << i;
    }
    if (!batch.mask || bytes > (size_t(256) << 20))
        return 0;
    scratch.dataType = fastllm::FP8_E4M3;
    scratch.Resize({(int)bytes});
    scratch.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    scratch.Allocate();
    auto *cursor = (unsigned char *)scratch.cudaData;
    for (int i = 0; i < count; ++i)
        if (batch.mask & (1u << i)) {
            auto &work = batch.entries[i];
            auto &e = work.prepared;
            e.q = (const __nv_bfloat16 *)cursor;
            cursor += (size_t)e.rows * 2048;
            e.key = FastllmNaiveKV(cursor);
            e.key.fp8 = e.key.scaledIndexer = true;
            e.stride = 144;
            cursor += (size_t)work.keyRows * 144;
            staged[i] = e;
            staged[i].q = work.source.q; // BF16 candidate scoring uses the original rounded Q.
        }
    native_fp8::PackQueries<<<dim3((maxRows * 2048 + 255) / 256, count), 256>>>(batch);
    native_fp8::StageKeys<<<dim3((maxKeys * 32 + 255) / 256, count), 256>>>(batch);
    const unsigned result = batch.mask;
    batch.mask &= ~native_fp8::LaunchStream(batch);
    if (batch.mask)
        native_fp8::IndexerScores<<<dim3((maxKeys + 63) / 64, (maxRows + 7) / 8, count), 256,
                                    native_fp8::kFp8SharedBytes>>>(batch);
    // The caller owns error reporting; never fall back after a partial launch.
    return result;
}
} // namespace naive_indexer

namespace native_fp8 {
// One query has sixteen heads: place keys in M and heads in N so the
// candidate refinement computes 64x16 products, without repeating Q four times.
__device__ __forceinline__ void mmaRefine(float (&c)[8], unsigned long long k, unsigned long long q) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0; }"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
                   "+f"(c[7])
                 : "l"(k), "l"(q));
#endif
}
template <bool LocalStaged, bool Expanded = false>
__global__ __launch_bounds__(128) void RefineScores(naive_indexer::FP8Entry e, unsigned long long *candidates,
                                                    const int *ends, int count, const BF *expanded = nullptr) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    const int row = blockIdx.y, first = blockIdx.x * 64;
    const int valid = min(ends[e.rows + row] - row * count, count);
    if (first >= valid)
        return;
    extern __shared__ __align__(1024) unsigned char buffer[];
    BF *k = (BF *)buffer, *q = k + 64 * 128;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if constexpr (Expanded) {
        // One warp gathers 16 candidates. Broadcast each row address and
        // copy both operands directly into the WGMMA shared layout.
        const int pos = first + warp * 16 + lane;
        const int ownKey = pos < valid ? int(0xffffffffu - unsigned(candidates[(size_t)row * count + pos])) : -1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = warp * 256 + j * 32 + lane, r = i / 16, d = i % 16 * 8;
            const int key = __shfl_sync(0xffffffff, ownKey, r % 16);
            const unsigned dst = __cvta_generic_to_shared(k + naive_indexer::offset(r, d, 64));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                "r"(dst), "l"(expanded + (size_t)max(0, key) * 128 + d), "r"(key >= 0 ? 16 : 0) : "memory");
        }
        for (int i = tid; i < 16 * 16; i += 128) {
            const int h = i / 16, d = i % 16 * 8;
            const unsigned dst = __cvta_generic_to_shared(q + naive_indexer::offset(h, d, 16));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"(dst), "l"(e.q + (size_t)row * 2048 + h * 128 + d) : "memory");
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    } else {
        for (int i = tid; i < 64 * 16; i += 128) {
            int r = i / 16, d = i % 16 * 8, pos = first + r;
            int key = pos < valid ? (int)(0xffffffffu - (unsigned)candidates[(size_t)row * count + pos]) : -1;
            uint4 values = make_uint4(0, 0, 0, 0);
            if (key >= 0) {
                // LaunchFP8 stages local contiguous rows once. Avoid repeating the
                // paged/peer/FP4 address machinery for every candidate vector.
                if constexpr (LocalStaged)
                    values =
                        FastllmNaiveKVRow((const unsigned char *)e.key.base + (size_t)key * 144 + d, true).Load8();
                else
                    values = (e.key.IndexRow(key, e.stride) + d).Load8();
            }
            *(uint4 *)(k + naive_indexer::offset(r, d, 64)) = values;
        }
        for (int i = tid; i < 16 * 16; i += 128) {
            int h = i / 16, d = i % 16 * 8;
            *(uint4 *)(q + naive_indexer::offset(h, d, 16)) =
                *(const uint4 *)(e.q + (size_t)row * 2048 + h * 128 + d);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta; wgmma.fence.sync.aligned;" ::: "memory");
    float dot[8] = {};
#pragma unroll
    for (int d = 0; d < 128; d += 16)
        mmaRefine(dot, naive_indexer::descriptor(k + naive_indexer::offset(0, d, 64)),
                  naive_indexer::descriptor(q + naive_indexer::offset(0, d, 16)));
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
    __syncthreads();
    // Reuse Q after its async readers finish. Swizzle only the key dimension;
    // each output thread still accumulates all sixteen heads in original order.
    float *dots = (float *)q;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int key = warp * 16 + lane / 4 + half * 8, h = n * 8 + (lane % 4) * 2;
            int swizzled = key ^ (((h / 2) & 3) * 8);
            dots[h * 64 + swizzled] = dot[n * 4 + half * 2];
            dots[(h + 1) * 64 + swizzled] = dot[n * 4 + half * 2 + 1];
        }
    }
    __syncthreads();
    const int pos = first + tid;
    if (tid < 64 && pos < valid) {
        int key = (int)(0xffffffffu - (unsigned)candidates[(size_t)row * count + pos]);
        float sum = 0.f;
#pragma unroll
        for (int h = 0; h < 16; ++h)
            sum += fmaxf(dots[h * 64 + (tid ^ (((h / 2) & 3) * 8))] * e.scales[row * 16 + h], 0.f) *
                   (float)e.weights[row * 16 + h];
        if (key >= 0) {
            float keyScale;
            if constexpr (LocalStaged)
                keyScale = *(const float *)((const unsigned char *)e.key.base + (size_t)key * 144 + 128);
            else
                keyScale = e.key.IndexScale(key, e.stride);
            float score = sum * keyScale;
            candidates[(size_t)row * count + pos] =
                ((unsigned long long)__float_as_uint(score) << 32) | (0xffffffffu - (unsigned)key);
        }
    }
#endif
}
// Conversion is exact and is performed once after the first shortlist has
// consumed its score rows. The caller reuses those dead rows as storage.
__global__ void ExpandRefinementKeys(const unsigned char *source, BF *output, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 16)
        return;
    const int row = i / 16, d = i % 16 * 8;
    *(uint4 *)(output + (size_t)row * 128 + d) =
        FastllmNaiveKVRow(source + (size_t)row * 144 + d, true).Load8();
}
} // namespace native_fp8
namespace naive_indexer {
bool StageRefinementKeys(const FP8Entry &e, int consumedRows) {
    const int64_t rows = (int64_t)e.past + e.rows + e.gap;
    const uint64_t bytes = rows > 0 ? (uint64_t)rows * 128 * sizeof(__nv_bfloat16) : 0;
    // Limit the expanded read set, and never allocate another serving buffer.
    // Small query tiles, long contexts and nonlocal layouts use direct reads.
    if (!bytes || rows > e.keys || rows > (int64_t)e.rows * 192 || !e.scores || consumedRows < 1 ||
        consumedRows > e.rows || bytes > (size_t(32) << 20) ||
        bytes > (uint64_t)consumedRows * e.keys * sizeof(float) || !e.key.fp8 || e.key.fp4 ||
        !e.key.scaledIndexer || e.key.pageLen || e.key.offset || e.stride != 144)
        return false;
    native_fp8::ExpandRefinementKeys<<<(rows * 16 + 255) / 256, 256>>>((const unsigned char *)e.key.base,
                                                                       (__nv_bfloat16 *)e.scores, rows);
    return true;
}
void RefineFP8(const FP8Entry &e, unsigned long long *candidates, const int *ends, int count,
               const __nv_bfloat16 *expanded) {
    const dim3 grid((count + 63) / 64, e.rows);
    if (expanded)
        native_fp8::RefineScores<true, true><<<grid, 128, 20480>>>(e, candidates, ends, count, expanded);
    else if (e.key.fp8 && !e.key.fp4 && e.key.scaledIndexer && !e.key.pageLen && !e.key.offset &&
             e.stride == 144)
        native_fp8::RefineScores<true><<<grid, 128, 20480>>>(e, candidates, ends, count);
    else
        native_fp8::RefineScores<false><<<grid, 128, 20480>>>(e, candidates, ends, count);
}
} // namespace naive_indexer
