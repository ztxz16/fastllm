#include "devices/cuda/fastllm-cuda.cuh"
#include "naive-n05-indexer.cuh"
#include <unordered_set>

namespace naive_indexer {
namespace {
using BF = __nv_bfloat16;
constexpr int kQueries = 8, kThreads = 256;
constexpr int kSharedBytes = (128 + 64) * 256 + 128 * 8;

// K-major 128-byte swizzle. Each 64-column slab spans all matrix rows.
__device__ __forceinline__ int offset(int row, int col, int rows) {
    return col / 64 * rows * 64 + row * 64 + ((col % 64) ^ ((row % 8) * 8));
}
__device__ __forceinline__ unsigned long long descriptor(const BF *p) {
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
// Fold sixteen heads into the MMA M dimension. Q keeps its original E4M3
// values/scales, K keeps its storage precision, and weighted heads accumulate
// in ascending order. Reuse Q storage for dot products after both groups finish.
__global__ __launch_bounds__(256) void IndexerScores(const BF *q, FastllmNaiveKV k, const float *qscale,
                                                     const BF *weights, float *scores, int rows, int keys,
                                                     int past, int stride) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    constexpr int Queries = kQueries, Threads = kThreads, M = Queries * 16;
    extern __shared__ __align__(1024) unsigned char buffer[];
    BF *sq = (BF *)buffer, *sk = sq + M * 128;
    float *sc = (float *)(sk + 64 * 128), *w = sc + M;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wg = tid / 128;
    int firstQ = blockIdx.y * Queries, firstK = blockIdx.x * 64;
    if (firstK > past + min(rows - 1, firstQ + Queries - 1)) {
        for (int i = tid; i < Queries * 64; i += Threads) {
            int row = firstQ + i / 64, col = firstK + i % 64;
            if (row < rows && col < keys)
                scores[(size_t)row * keys + col] = -INFINITY;
        }
        return;
    }
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
        auto src = k.IndexRow(min(firstK + row, keys - 1), stride) + d;
        if (stride % 8 == 0 && !src.fp8) {
            unsigned dst = (unsigned)__cvta_generic_to_shared(sk + offset(row, d, 64));
            int valid = firstK + row < keys ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src.data), "r"(valid)
                         : "memory");
        } else if (stride % 8 == 0)
            *(uint4 *)(sk + offset(row, d, 64)) = firstK + row < keys ? src.Load8() : make_uint4(0, 0, 0, 0);
        else
            for (int j = 0; j < 8; ++j)
                sk[offset(row, d + j, 64)] = firstK + row < keys ? src[j] : __float2bfloat16(0.f);
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
    for (int d = 0; d < 128; d += 16)
        mma(dot, descriptor(sq + offset(wg * 64, d, M)), descriptor(sk + offset(0, d, 64)));
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
                    firstK + col <= past + firstQ + row ? total * k.IndexScale(firstK + col, stride) : -INFINITY;
        }
    }
#endif
}
} // namespace

bool Prepare() {
    if (FastllmCudaRuntimeArch() != 90)
        return false;
    static thread_local std::unordered_set<int> initialized;
    int device = FastllmCudaGetDevice();
    if (initialized.count(device))
        return true;
    if (FastllmCudaGraphIsCapturingFast())
        return false;
    if (cudaFuncSetAttribute(IndexerScores, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes) !=
        cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    initialized.insert(device);
    return true;
}

void Launch(const BF *q, FastllmNaiveKV keys, const float *scales, const BF *weights, float *scores,
            int queries, int keyCount, int past, int stride) {
    IndexerScores<<<dim3((past + queries + 63) / 64, (queries + kQueries - 1) / kQueries), kThreads,
                    kSharedBytes>>>(q, keys, scales, weights, scores, queries, keyCount, past, stride);
}
} // namespace naive_indexer
