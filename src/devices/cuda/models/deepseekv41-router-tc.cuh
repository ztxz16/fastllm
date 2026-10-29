#pragma once
// BF16 router matmul with FP32 accumulation. CP.async tiles are reused by
// 16 experts and up to 8 tokens. SM90 reduces split-K in CTA-cluster DSMEM;
// other Tensor Core GPUs use a persistent FP32 scratch buffer.
// PDL waits precede dependent input reads; bias/weight loads may run ahead.
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float V41TcRouterTransform(float x) {
    return sqrtf(x > 20.f ? x : __logf(1.f + __expf(x)));
}

template <int KTile> __device__ __forceinline__ int V41TcRouterSwizzle(int row, int k) {
    return row * KTile + (k ^ ((row & 7) * 8));
}
template <int KTile, int Warps>
__device__ __forceinline__ void V41TcRouterCopy(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                __nv_bfloat16 *tile, int rows, int n, int k, int expert,
                                                int start) {
    for (int chunk = threadIdx.x; chunk < 24 * KTile / 8; chunk += Warps * 32) {
        if (chunk == 16 * KTile / 8 + threadIdx.x) {
#if __CUDA_ARCH__ >= 900
            cudaGridDependencySynchronize();
#endif
        }
        int row = chunk / (KTile / 8), kk = (chunk % (KTile / 8)) * 8;
        bool valid = start + kk < k && (row < 16 ? expert + row < n : row - 16 < rows);
        const __nv_bfloat16 *src =
            row < 16 ? w + (size_t)(expert + row) * k + start + kk : x + (size_t)(row - 16) * k + start + kk;
        unsigned dst = __cvta_generic_to_shared(tile + V41TcRouterSwizzle<KTile>(row, kk));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int Warps, int Splits, int KTile, int Stages, bool Cluster>
__global__ void V41TcRouterPipeline(const __nv_bfloat16 *x, const __nv_bfloat16 *w, float *y, int rows, int n,
                                    int k, float invTemp) {
#if __CUDA_ARCH__ >= 800
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, group = lane / 4, pair = lane % 4;
    const int expert = blockIdx.y * 16, split = blockIdx.x;
#if __CUDA_ARCH__ >= 900
    if constexpr (Cluster)
        cooperative_groups::this_cluster().barrier_arrive();
#endif
#if __CUDA_ARCH__ >= 900
    if (tid == 0)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    __shared__ __align__(16) __nv_bfloat16 tiles[Stages][24 * KTile];
    __shared__ float sums[Warps][128];
    __shared__ float parts[Cluster ? Splits : 1][128];
    int count = ((k + KTile - 1) / KTile + Splits - 1 - split) / Splits;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < Stages - 1; ++i)
        V41TcRouterCopy<KTile, Warps>(x, w, tiles[i], rows, n, k, expert, (split + i * Splits) * KTile);
    for (int it = 0; it < count; ++it) {
        V41TcRouterCopy<KTile, Warps>(x, w, tiles[(it + Stages - 1) % Stages], rows, n, k, expert,
                                      (split + (it + Stages - 1) * Splits) * KTile);
        asm volatile("cp.async.wait_group %0;" ::"n"(Stages - 1) : "memory");
        __syncthreads();
        const __nv_bfloat16 *tile = tiles[it % Stages];
#pragma unroll
        for (int q = warp * 16; q < KTile; q += Warps * 16) {
            unsigned a0, a1, a2, a3, b0, b1;
            unsigned ap =
                __cvta_generic_to_shared(tile + V41TcRouterSwizzle<KTile>(lane % 16, q + (lane / 16) * 8));
            unsigned bp = __cvta_generic_to_shared(
                tile + V41TcRouterSwizzle<KTile>(16 + lane % 8, q + ((lane / 8) % 2) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "r"(ap));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b0), "=r"(b1)
                         : "r"(bp));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    int idx = group * 8 + pair * 2;
    sums[warp][idx] = c0;
    sums[warp][idx + 1] = c1;
    sums[warp][idx + 64] = c2;
    sums[warp][idx + 65] = c3;
    __syncthreads();
    for (int i = tid; i < 128; i += Warps * 32) {
        float v = 0;
#pragma unroll
        for (int z = 0; z < Warps; ++z)
            v += sums[z][i];
        if constexpr (Cluster)
            parts[split][i] = v;
        else if (expert + i / 8 < n && i % 8 < rows)
            y[((size_t)split * rows + i % 8) * n + expert + i / 8] = v;
    }
    if constexpr (Cluster) {
#if __CUDA_ARCH__ >= 900
        auto cluster = cooperative_groups::this_cluster();
        cluster.barrier_wait();
        __syncthreads();
        for (int i = tid; i < 128; i += Warps * 32) {
            float v = parts[split][i];
            unsigned address = __cvta_generic_to_shared(&parts[split][i]);
#pragma unroll
            for (int peer = 0; peer < Splits; ++peer)
                if (peer != split) {
                    unsigned remote;
                    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                                 : "=r"(remote)
                                 : "r"(address), "r"(peer));
                    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(v) : "memory");
                }
        }
        cluster.sync();
        if (split == 0)
            for (int i = tid; i < 128; i += Warps * 32)
                if (expert + i / 8 < n && i % 8 < rows) {
                    float v = 0;
#pragma unroll
                    for (int z = 0; z < Splits; ++z)
                        v += parts[z][i];
                    v *= invTemp;
                    y[(size_t)(i % 8) * n + expert + i / 8] = V41TcRouterTransform(v);
                }
#endif
    }
#endif
}

template <int Splits> __global__ void V41TcRouterFinish(const float *p, float *y, int count, float invTemp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
#endif
    if (i < count) {
        float v = 0;
#pragma unroll
        for (int s = 0; s < Splits; ++s)
            v += p[(size_t)s * count + i];
        v *= invTemp;
        y[i] = V41TcRouterTransform(v);
    }
}

template <int Warps, int Splits, int KTile, int Stages, bool Cluster>
cudaError_t V41TcLaunchPipeline(const __nv_bfloat16 *x, const __nv_bfloat16 *w, float *p, float *y, int m,
                                int n, int k, float invTemp, cudaStream_t stream, bool pdl) {
#if CUDART_VERSION >= 12000
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(Splits, (n + 15) / 16);
    cfg.blockDim = Warps * 32;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[2] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim = {Splits, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = Cluster ? 2 : (pdl ? 1 : 0);
    if constexpr (Cluster)
        return cudaLaunchKernelEx(&cfg, V41TcRouterPipeline<Warps, Splits, KTile, Stages, true>, x, w, y, m,
                                  n, k, invTemp);
    else {
        cudaError_t e = cudaLaunchKernelEx(&cfg, V41TcRouterPipeline<Warps, Splits, KTile, Stages, false>, x,
                                           w, p, m, n, k, invTemp);
        if (e != cudaSuccess)
            return e;
        cfg.gridDim = (m * n + 255) / 256;
        cfg.blockDim = 256;
        return cudaLaunchKernelEx(&cfg, V41TcRouterFinish<Splits>, p, y, m * n, invTemp);
    }
#else
    if constexpr (Cluster)
        return cudaErrorNotSupported;
    else {
        V41TcRouterPipeline<Warps, Splits, KTile, Stages, false>
            <<<dim3(Splits, (n + 15) / 16), Warps * 32, 0, stream>>>(x, w, p, m, n, k, invTemp);
        V41TcRouterFinish<Splits><<<(m * n + 255) / 256, 256, 0, stream>>>(p, y, m * n, invTemp);
        return cudaGetLastError();
    }
#endif
}

// Each warp owns a row. Values and corrected selection scores stay in registers.
template <int N, int TopK>
__global__ void V41TcRouterTopK(const float *input, const float *bias, int *ids, float *weights,
                                float scale) {
#if __CUDA_ARCH__ >= 800
    static_assert(N % 32 == 0 && TopK > 0 && N >= TopK);
    int row = blockIdx.x, lane = threadIdx.x;
    float value[N / 32], score[N / 32];
#pragma unroll
    for (int j = 0; j < N / 32; ++j)
        score[j] = bias[j * 32 + lane];
#if __CUDA_ARCH__ >= 900
    if (lane == 0)
        cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
#endif
#pragma unroll
    for (int j = 0; j < N / 32; ++j) {
        int e = j * 32 + lane;
        float v = input[row * N + e];
        value[j] = v;
        score[j] += v;
        if (isnan(score[j]))
            score[j] = -1e30f;
    }
    float selected[TopK], total = 0;
    int chosen[TopK];
#pragma unroll
    for (int t = 0; t < TopK; ++t) {
        float best = -CUDART_INF_F;
#pragma unroll
        for (int j = 0; j < N / 32; ++j)
            best = fmaxf(best, score[j]);
        unsigned bits = __float_as_uint(best);
        bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
        unsigned maximum = __reduce_max_sync(0xffffffff, bits);
        maximum ^= (maximum & 0x80000000u) ? 0x80000000u : 0xffffffffu;
        best = __uint_as_float(maximum);
        unsigned localId = 0xffffffffu;
#pragma unroll
        for (int j = 0; j < N / 32; ++j)
            if (score[j] == best)
                localId = min(localId, (unsigned)(j * 32 + lane));
        int id = __reduce_min_sync(0xffffffff, localId);
        float v = 0;
#pragma unroll
        for (int j = 0; j < N / 32; ++j)
            if (id == j * 32 + lane) {
                v = value[j];
                score[j] = -CUDART_INF_F;
            }
        v = __shfl_sync(0xffffffff, v, id % 32);
        selected[t] = v;
        chosen[t] = id;
        total += v;
    }
#pragma unroll
    for (int t = 0; t < TopK; ++t)
        if (lane == t) {
            ids[row * TopK + t] = chosen[t];
            weights[row * TopK + t] = selected[t] * (scale / (total > 0.f ? total : 1.f));
        }
#endif
}
