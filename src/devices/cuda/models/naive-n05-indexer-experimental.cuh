#pragma once
#include <cuda_fp8.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
namespace experimental_fp8 {
using BF = __nv_bfloat16;
struct WorkEntry { naive_indexer::ExperimentalEntry source, prepared; int keyRows; };
struct Batch { WorkEntry entries[8]; unsigned mask; int count; };
static std::atomic<unsigned long long> launches{0};
__global__ void PackQueries(Batch batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, entry = blockIdx.y;
    if (!(batch.mask & (1u << entry))) return;
    auto e = batch.entries[entry];
    if (i < e.source.rows * 2048)
        ((unsigned char *)e.prepared.q)[i] = __nv_fp8_e4m3((float)e.source.q[i]).__x;
}
// Copy FP8 payloads/scales without dequantization. Reading a peer cache once
// avoids replaying the smaller WGMMA tile's reads across NVLink for every Q tile.
__global__ void StageKeys(Batch batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, entry = blockIdx.y;
    if (!(batch.mask & (1u << entry))) return;
    auto e = batch.entries[entry];
    if (i >= e.keyRows * 32) return;
    int row = i / 32, d = i % 32 * 4;
    const auto *src = (const unsigned char *)e.source.key.IndexRow(row, e.source.stride).data + d;
    auto *dst = (unsigned char *)e.prepared.key.base + (size_t)row * 144 + d;
    if (((size_t)src & 3) == 0) *(unsigned *)dst = *(const unsigned *)src;
    else for (int j = 0; j < 4; ++j) dst[j] = src[j];
    if (d == 0) *(float *)(dst + 128) = e.source.key.IndexScale(row, e.source.stride);
}
constexpr int kQueries = 8, kThreads = 256;
constexpr int kSharedBytes = (128 + 64) * 256 + 128 * 8;
constexpr int kFp8SharedBytes = 128 * 64 * sizeof(float) + 128 * 8;

// K-major 128-byte swizzle. Each 64-column slab spans all matrix rows.
__device__ __forceinline__ int offset(int row, int col, int rows) {
    return col / 64 * rows * 64 + row * 64 + ((col % 64) ^ ((row % 8) * 8));
}
__device__ __forceinline__ unsigned long long descriptor(const void *p) {
    return ((unsigned)__cvta_generic_to_shared(p) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void mma(float (&c)[32], unsigned long long a, unsigned long long b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                 "%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
                   "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]),
                   "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]),
                   "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]),
                   "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
                 : "l"(a), "l"(b));
#endif
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
template<bool Fp8> __device__ __forceinline__ int storageOffset(int row, int col, int rows) {
    return Fp8 ? row * 128 + (col ^ ((row & 7) * 16)) : offset(row, col, rows);
}
// Fold sixteen heads into the MMA M dimension. Q keeps its original E4M3
// values/scales, K keeps its storage precision, and weighted heads accumulate
// in ascending order. Reuse Q storage for dot products after both groups finish.
template<bool Fp8>
__global__ __launch_bounds__(256) void IndexerScores(Batch batch) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    int entry = blockIdx.z;
    if (!(batch.mask & (1u << entry))) return;
    const auto &e = batch.entries[entry].prepared;
    const BF *q = e.q, *weights = e.weights;
    FastllmNaiveKV k = e.key;
    const float *qscale = e.scales;
    float *scores = e.scores;
    int rows = e.rows, keys = e.keys, past = e.past, stride = e.stride;
    if (blockIdx.y * kQueries >= rows) return;
    constexpr int Queries = kQueries, Threads = kThreads, M = Queries * 16;
    extern __shared__ __align__(1024) unsigned char buffer[];
    using Storage = typename std::conditional<Fp8, unsigned char, BF>::type;
    Storage *sq = (Storage *)buffer, *sk = sq + M * 128;
    constexpr int scratch = Fp8 ? M * 64 * sizeof(float) : (M + 64) * 128 * sizeof(BF);
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
    if constexpr (Fp8) {
        // Q has already been quantized to E4M3 and expanded exactly to BF16.
        // Repacking changes storage only; K bytes and scales remain unchanged.
        for(int i=tid;i<M*8;i+=Threads) {
            int row=i/8,d=i%8*16;
            const auto *src=(const unsigned char*)q+((size_t)min(firstQ+row/16,rows-1)*16+row%16)*128+d;
            unsigned dst=(unsigned)__cvta_generic_to_shared(sq+storageOffset<true>(row,d,M));
            int valid=firstQ+row/16<rows?16:0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(valid):"memory");
        }
        // LaunchExperimentalFP8 always stages K as local, aligned 144-byte
        // rows. Do not resolve the generic paged/peer cache view again here.
        for (int i = tid; i < 64 * 8; i += Threads) {
            int row = i / 8, d = i % 8 * 16;
            const auto *src = (const unsigned char *)k.base +
                (size_t)min(firstK + row, batch.entries[entry].keyRows - 1) * 144 + d;
            unsigned dst = (unsigned)__cvta_generic_to_shared(sk + storageOffset<true>(row,d,64));
            int valid = firstK + row < batch.entries[entry].keyRows ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                "r"(dst), "l"(src), "r"(valid) : "memory");
        }
    } else {
    for (int i = tid; i < M * 16; i += Threads) {
        int row = i / 16, d = i % 16 * 8;
        const BF *src = q + ((size_t)min(firstQ + row / 16, rows - 1) * 16 + row % 16) * 128 + d;
        unsigned dst = (unsigned)__cvta_generic_to_shared(sq + offset(row, d, M));
        int valid = firstQ + row / 16 < rows ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid)
                     : "memory");
    }
    for (int i = tid; i < 64 * 16; i += Threads) {
        int row = i / 16, d = i % 16 * 8;
        auto src = k.IndexRow(min(firstK + row, batch.entries[entry].keyRows - 1), stride) + d;
        if (stride % 8 == 0 && !src.fp8) {
            unsigned dst = (unsigned)__cvta_generic_to_shared(sk + offset(row, d, 64));
            int valid = firstK + row < batch.entries[entry].keyRows ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src.data), "r"(valid)
                         : "memory");
        } else if (stride % 8 == 0)
            *(uint4 *)(sk + offset(row, d, 64)) = firstK + row < keys ? src.Load8() : make_uint4(0, 0, 0, 0);
        else
            for (int j = 0; j < 8; ++j)
                sk[offset(row, d + j, 64)] = firstK + row < keys ? src[j] : __float2bfloat16(0.f);
    }
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
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int d = 0; d < 128; d += Fp8 ? 32 : 16) {
        auto qa = descriptor(sq + storageOffset<Fp8>(wg * 64, d, M));
        auto ka = descriptor(sk + storageOffset<Fp8>(0, d, 64));
        if constexpr (Fp8) mmaFp8(dot, qa, ka);
        else mma(dot, qa, ka);
    }
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
    {
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
                    firstK + col <= past + firstQ + row + (e.part && firstQ + row >= e.part ? e.gap : 0) ? total * (Fp8 ? *(const float *)((const unsigned char *)k.base + (size_t)(firstK + col) * 144 + 128) : k.IndexScale(firstK + col, stride)) : -INFINITY;
        }
    }
#endif
}


} // namespace experimental_fp8

namespace naive_indexer {
unsigned long long ExperimentalFP8Launches() { return experimental_fp8::launches.load(); }
unsigned LaunchExperimentalFP8(const ExperimentalEntry *entries, int count) {
    const char *option = std::getenv("FASTLLM_EXPERIMENTAL_FP8_INDEXER");
    if (!option || std::strcmp(option, "1") || count < 1 || count > 8 ||
        FastllmCudaRuntimeArch() != 90 || FastllmCudaGraphIsCapturingFast()) return 0;
    experimental_fp8::Batch batch{};
    batch.count = count;
    size_t bytes = 0;
    int maxRows = 0, maxKeys = 0;
    for (int i = 0; i < count; ++i) {
        const auto &e = entries[i];
        if (!(e.key.fp8 || e.key.scaledIndexer) || e.rows < 32 || e.past < 0 ||
            e.rows > 32768 || e.gap < 0 || (e.part && e.part * 2 != e.rows) ||
            (int64_t)e.past + e.rows + e.gap > e.keys) continue;
        auto &work = batch.entries[i];
        work.source = work.prepared = e;
        work.keyRows = e.past + e.rows + e.gap;
        bytes += (size_t)e.rows * 2048 + (size_t)work.keyRows * 144;
        maxRows = std::max(maxRows, e.rows);
        maxKeys = std::max(maxKeys, work.keyRows);
        batch.mask |= 1u << i;
    }
    if (!batch.mask || bytes > (size_t(256) << 20)) return 0;
    fastllm::Data scratch(fastllm::FP8_E4M3);
    scratch.Resize({(int)bytes});
    scratch.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    scratch.Allocate();
    auto *cursor = (unsigned char *)scratch.cudaData;
    for (int i = 0; i < count; ++i) if (batch.mask & (1u << i)) {
        auto &work = batch.entries[i];
        auto &e = work.prepared;
        e.q = (const __nv_bfloat16 *)cursor;
        cursor += (size_t)e.rows * 2048;
        e.key = FastllmNaiveKV(cursor);
        e.key.fp8 = e.key.scaledIndexer = true;
        e.stride = 144;
        cursor += (size_t)work.keyRows * 144;
    }
    experimental_fp8::PackQueries<<<dim3((maxRows * 2048 + 255) / 256, count), 256>>>(batch);
    experimental_fp8::StageKeys<<<dim3((maxKeys * 32 + 255) / 256, count), 256>>>(batch);
    experimental_fp8::IndexerScores<true><<<dim3((maxKeys + 63) / 64, (maxRows + 7) / 8, count),
        256, experimental_fp8::kFp8SharedBytes>>>(batch);
    experimental_fp8::launches.fetch_add(1);
    static thread_local std::unordered_set<int> announced;
    int device = FastllmCudaGetDevice();
    if (announced.insert(device).second) {
        fprintf(stderr, "[experimental_fp8_indexer] enabled device=%d entries=%d rows=%d keys=%d scratch=%zu\n",
                device, count, maxRows, maxKeys, bytes);
        fflush(stderr);
    }
    return batch.mask;
}
} // namespace naive_indexer
