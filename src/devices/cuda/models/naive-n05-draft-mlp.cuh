#pragma once

#include "low_latency/bf16_projection.cuh"
#include "fastllm-cuda-norm.cuh"
#include <cooperative_groups.h>

namespace fastllm_draft_fused {

// Both the sharded and replicated single-request graph record this entry.
template<class Policy>
__global__ __launch_bounds__(Policy::Threads, Policy::MinBlocks) void FusedDraftMlp(
        const __grid_constant__ typename Policy::Args args) {
    extern __shared__ __align__(128) unsigned char memory[];
    Policy::Run(args, memory);
}

// Row-major BF16 projection tiles and the rounded residual normalization shared by the two replica kernels.
struct DraftReplicaProjection {
    using BF = __nv_bfloat16;
    static constexpr int Width = 4096, Threads = 256, Warps = Threads / 32;
    struct VectorPair { uint4 first, second; };
    __device__ static VectorPair LoadWeight(const BF *p) {
        VectorPair w;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        asm("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(w.first.x), "=r"(w.first.y), "=r"(w.first.z), "=r"(w.first.w),
              "=r"(w.second.x), "=r"(w.second.y), "=r"(w.second.z), "=r"(w.second.w) : "l"(p));
#else
        const uint4 *pair = reinterpret_cast<const uint4 *>(p);
        w = {pair[0], pair[1]};
#endif
        return w;
    }
    template<int SecondOffset>
    __device__ __forceinline__ static void DotTile(const BF *weights, const BF *input, int rows, int tile, float *partials, float &lowSum, float &highSum) {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const int group = lane / 4, pair = lane % 4;
        weights += size_t(group) * Width;
        input += min(group, rows - 1) * Width; // Discard the spare MMA column.
        float accum[4] = {};
        // SecondOffset pairs a second group of eight weight rows in the same MMA.
        // Stagger K across tiles to avoid issuing every CTA to the same sectors.
#pragma unroll 8
        for (int step = 0; step < Width / Warps; step += 64) {
            const int k = (warp * (Width / Warps) + step + tile * 128) % Width + pair * 16;
            const auto low = LoadWeight(weights + k);
            VectorPair high = {};
            if constexpr (SecondOffset != 0) high = LoadWeight(weights + SecondOffset + k);
            const auto *x = reinterpret_cast<const uint4 *>(input + k);
            using fastllm::cuda::low_latency::projection::mma_bf16_16816;
            mma_bf16_16816(accum, low.first.x, high.first.x, low.first.y, high.first.y, x[0].x, x[0].y);
            mma_bf16_16816(accum, low.first.z, high.first.z, low.first.w, high.first.w, x[0].z, x[0].w);
            mma_bf16_16816(accum, low.second.x, high.second.x, low.second.y, high.second.y, x[1].x, x[1].y);
            mma_bf16_16816(accum, low.second.z, high.second.z, low.second.w, high.second.w, x[1].z, x[1].w);
        }
        float *out = partials + warp * 128 + group * 8 + pair * 2;
        *reinterpret_cast<float2 *>(out) = make_float2(accum[0], accum[1]);
        if constexpr (SecondOffset != 0)
            *reinterpret_cast<float2 *>(out + 64) = make_float2(accum[2], accum[3]);
        __syncthreads();
        if (threadIdx.x < 64 && threadIdx.x % 8 < rows) {
#pragma unroll
            for (int s = 0; s < Warps; ++s) {
                lowSum += partials[s * 128 + threadIdx.x];
                if constexpr (SecondOffset != 0) highSum += partials[s * 128 + 64 + threadIdx.x];
            }
        }
    }

    __device__ __forceinline__ static void Normalize(int rows, const BF *residual, const float *gamma, float eps,
            const float *normPartials, BF *output) {
        constexpr int TilesPerRow = Width / Threads;
        // The down phase already rounded the branch and residual. Reduce each
        // row's tile sums, then normalize disjoint columns of that row.
        __shared__ float scratch[8];
        for (int task = blockIdx.x; task < rows * TilesPerRow; task += gridDim.x) {
            const int row = task / TilesPerRow;
            const int col = (task % TilesPerRow) * Threads + threadIdx.x;
            const float *ss = normPartials + row * (Width / 8);
            const float partial = ss[threadIdx.x] + ss[threadIdx.x + Threads];
            const float inv = rsqrtf(fastllm::cuda::WarpFirstBlockSum256(partial, scratch) / Width + eps);
            const float normalized = float(__float2bfloat16_rn(float(residual[row * Width + col]) * inv));
            output[row * Width + col] = __float2bfloat16_rn(normalized * gamma[col]);
            __syncthreads();
        }
    }
};

struct ReplicaMlp : DraftReplicaProjection {
    // Four resident CTAs per SM give the 512 weight tiles enough parallel readers.
    static constexpr int MinBlocks = 4;
    static constexpr int Rows = 8;
    struct Args {
        const BF *normalized, *gateUp, *down;
        BF *middle;
        float *normPartials;
        BF *residual, *output;
        const float *gamma;
        float eps;
        int rows;
    };

    template<bool GateUp>
    __device__ __forceinline__ static void Project(const Args &a, int tile, float *partials) {
        float low = 0.f, high = 0.f;
        const BF *weights = (GateUp ? a.gateUp : a.down) + size_t(tile * 8) * Width;
        DotTile<GateUp ? Width * Width : 0>(weights, GateUp ? a.normalized : a.middle,
            a.rows, tile, partials, low, high);
        if (threadIdx.x < 64 && threadIdx.x % 8 < a.rows) {
            const int offset = (threadIdx.x % 8) * Width + tile * 8 + threadIdx.x / 8;
            if constexpr (GateUp) {
                const float gate = float(__float2bfloat16_rn(low));
                const float up = float(__float2bfloat16_rn(high));
                const float silu = float(__float2bfloat16_rn(gate / (1.f + expf(-gate))));
                a.middle[offset] = __float2bfloat16_rn(silu * up);
            } else {
                const BF sum = __float2bfloat16_rn(float(a.residual[offset]) + float(__float2bfloat16_rn(low)));
                a.residual[offset] = sum;
                partials[threadIdx.x] = float(sum) * float(sum);
            }
        }
        __syncthreads();
        if constexpr (!GateUp) {
            if (threadIdx.x < a.rows) {
                float ss = 0.f;
#pragma unroll
                for (int col = 0; col < 8; ++col) ss += partials[col * 8 + threadIdx.x];
                a.normPartials[threadIdx.x * (Width / 8) + tile] = ss;
            }
            __syncthreads();
        }
    }
    __device__ __forceinline__ static void Run(const Args &a, unsigned char *) {
        __shared__ __align__(16) float partials[Warps * 128];
        const auto grid = cooperative_groups::this_grid();
        for (int tile = blockIdx.x; tile < Width / 8; tile += gridDim.x) Project<true>(a, tile, partials);
        grid.sync();
        for (int tile = blockIdx.x; tile < Width / 8; tile += gridDim.x) Project<false>(a, tile, partials);
        grid.sync();
        Normalize(a.rows, a.residual, a.gamma, a.eps, a.normPartials, a.output);
    }
};

// Put both requests in the sixteen MMA rows. The eight columns are eight output channels.
struct ReplicaMlpPair : ReplicaMlp {
    template<bool GateUp>
    __device__ __forceinline__ static void ProjectPair(const Args &a, int tile, float *partials) {
        const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
        const int row = lane / 4, pair = lane % 4;
        const BF *weights = (GateUp ? a.gateUp : a.down) + size_t(tile * 8 + row) * Width;
        const BF *input = GateUp ? a.normalized : a.middle;
        float lo[4]{}, hi[4]{};
#pragma unroll 8
        for (int step = 0; step < Width / Warps; step += 64) {
            const int k = (warp * (Width / Warps) + step + tile * 128) % Width + pair * 16;
            const auto x = LoadWeight(input + row * Width + k);
            const auto y = LoadWeight(input + min(row + 8, a.rows - 1) * Width + k);
            const auto w = LoadWeight(weights + k);
            using fastllm::cuda::low_latency::projection::mma_bf16_16816;
            mma_bf16_16816(lo, x.first.x, y.first.x, x.first.y, y.first.y, w.first.x, w.first.y);
            mma_bf16_16816(lo, x.first.z, y.first.z, x.first.w, y.first.w, w.first.z, w.first.w);
            mma_bf16_16816(lo, x.second.x, y.second.x, x.second.y, y.second.y, w.second.x, w.second.y);
            mma_bf16_16816(lo, x.second.z, y.second.z, x.second.w, y.second.w, w.second.z, w.second.w);
            if constexpr (GateUp) {
                const auto u = LoadWeight(weights + Width * Width + k);
                mma_bf16_16816(hi, x.first.x, y.first.x, x.first.y, y.first.y, u.first.x, u.first.y);
                mma_bf16_16816(hi, x.first.z, y.first.z, x.first.w, y.first.w, u.first.z, u.first.w);
                mma_bf16_16816(hi, x.second.x, y.second.x, x.second.y, y.second.y, u.second.x, u.second.y);
                mma_bf16_16816(hi, x.second.z, y.second.z, x.second.w, y.second.w, u.second.z, u.second.w);
            }
        }
        float *out = partials + warp * 128 + lane * 2;
        *reinterpret_cast<float2 *>(out) = make_float2(lo[0], lo[1]);
        *reinterpret_cast<float2 *>(out + 64) = make_float2(lo[2], lo[3]);
        if constexpr (GateUp) {
            *reinterpret_cast<float2 *>(out + Warps * 128) = make_float2(hi[0], hi[1]);
            *reinterpret_cast<float2 *>(out + Warps * 128 + 64) = make_float2(hi[2], hi[3]);
        }
        __syncthreads();
        if (tid < a.rows * 8) {
            float low = 0.f, high = 0.f;
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                low += partials[w * 128 + tid];
                if constexpr (GateUp) high += partials[(Warps + w) * 128 + tid];
            }
            const int offset = (tid / 8) * Width + tile * 8 + tid % 8;
            if constexpr (GateUp) {
                const float gate = float(__float2bfloat16_rn(low));
                const float up = float(__float2bfloat16_rn(high));
                const float silu = float(__float2bfloat16_rn(gate / (1.f + expf(-gate))));
                a.middle[offset] = __float2bfloat16_rn(silu * up);
            } else {
                const BF sum = __float2bfloat16_rn(float(a.residual[offset]) + float(__float2bfloat16_rn(low)));
                a.residual[offset] = sum;
                // Each lane owns these cells in every reduction plane.
                partials[tid] = float(sum) * float(sum);
            }
        }
        __syncthreads();
        if constexpr (!GateUp) {
            if (tid < a.rows) {
                float ss = 0.f;
#pragma unroll
                for (int col = 0; col < 8; ++col) ss += partials[tid * 8 + col];
                a.normPartials[tid * (Width / 8) + tile] = ss;
            }
            __syncthreads();
        }
    }
    __device__ __forceinline__ static void Run(const Args &a, unsigned char *) {
        __shared__ __align__(16) float partials[2 * Warps * 128];
        const auto grid = cooperative_groups::this_grid();
        for (int tile = blockIdx.x; tile < Width / 8; tile += gridDim.x) ProjectPair<true>(a, tile, partials);
        grid.sync();
        for (int tile = blockIdx.x; tile < Width / 8; tile += gridDim.x) ProjectPair<false>(a, tile, partials);
        grid.sync();
        Normalize(a.rows, a.residual, a.gamma, a.eps, a.normPartials, a.output);
    }
};
} // namespace fastllm_draft_fused
