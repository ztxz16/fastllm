#pragma once
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include "low_latency/bf16_projection.cuh"

namespace fastllm_draft_owner {
using B = __nv_bfloat16;
constexpr int Threads = 128, Tile = 256;
using Reduce = cub::BlockReduce<float, Threads>;
using Scan = cub::BlockScan<double, Threads>;

struct Proposal {
    const B *base, *embedding, *weight;
    const float *temperature;
    const uint64_t *seed;
    float *ids, *q;
    float2 *stats;
    int vocab, steps;
};

// The already allocated sampler buffer holds tile statistics.
// Each step publishes the same normalized distribution from which it draws the next ID.
__global__ __launch_bounds__(Threads) void FusedDraftHeadProposal(Proposal a) {
    auto grid = cooperative_groups::this_grid();
    __shared__ union { Reduce::TempStorage reduce; Scan::TempStorage scan; } scratch;
    __shared__ float logits[Tile], tileMax, globalMax;
    __shared__ int selectedTile;
    __shared__ double selectedOffset, drawMass;
    const int tid = threadIdx.x, lane = tid % 32;
    const int begin = blockIdx.x * Tile, end = min(begin + Tile, a.vocab);
    const float scale = 1.f / *a.temperature;
    curandStatePhilox4_32_10_t rng;
    if (!tid) curand_init(*a.seed, 0, 0, &rng);
    for (int step = 0; step < a.steps; ++step) {
        int prev = int(a.ids[step]);
        if (prev < 0 || prev >= a.vocab) prev = 0;
        const B *latent = a.embedding + size_t(prev) * 256;
        float *q = a.q + size_t(step) * a.vocab;
        const B *base = a.base + size_t(step) * a.vocab;
        // The eight MMA query columns contain the same 256-element embedding.
        // Keep the reference's two BF16 rounding boundaries around the base addition.
        for (int group = begin + tid / 32 * 16; group < end; group += (Threads / 32) * 16) {
            const int n = group + lane / 4;
            const B *w = a.weight + size_t(n) * 256;
            float sum[4] = {};
            #pragma unroll
            for (int k0 = 0; k0 < 256; k0 += 32) {
                const int k = k0 + (lane % 4) * 8;
                const auto lo = *reinterpret_cast<const uint4 *>(w + k);
                const auto hi = *reinterpret_cast<const uint4 *>(w + 8 * 256 + k);
                const auto v = *reinterpret_cast<const uint4 *>(latent + k);
                using fastllm::cuda::low_latency::projection::mma_bf16_16816;
                mma_bf16_16816(sum, lo.x,hi.x,lo.y,hi.y,v.x,v.y);
                mma_bf16_16816(sum, lo.z,hi.z,lo.w,hi.w,v.z,v.w);
            }
            if (lane % 4 == 0) {
                logits[n - begin] = float(B(float(base[n]) + float(B(sum[0])))) * scale;
                logits[n + 8 - begin] = float(B(float(base[n + 8]) + float(B(sum[2])))) * scale;
            }
        }
        __syncthreads();
        const int col = begin + 2 * tid;
        float x = col < end ? logits[2 * tid] : -INFINITY;
        float y = col < end ? logits[2 * tid + 1] : -INFINITY;
        float maximum = Reduce(scratch.reduce).Reduce(fmaxf(x, y), cub::Max());
        if (!tid) tileMax = maximum;
        __syncthreads();
        x = tileMax == -INFINITY ? 0.f : expf(x - tileMax);
        y = tileMax == -INFINITY ? 0.f : expf(y - tileMax);
        float mass = Reduce(scratch.reduce).Sum(x + y);
        if (!tid) a.stats[blockIdx.x] = make_float2(tileMax, mass);
        grid.sync();

        // Each CTA can join the small array of tile statistics from L2. This
        // keeps its vocabulary tile in registers until both export and draw.
        maximum = -INFINITY;
        for (int i = tid; i < gridDim.x; i += Threads) maximum = fmaxf(maximum, a.stats[i].x);
        maximum = Reduce(scratch.reduce).Reduce(maximum, cub::Max());
        if (!tid) globalMax = maximum;
        __syncthreads();
        const int perLane = (gridDim.x + Threads - 1) / Threads;
        const int first = tid * perLane, last = min(first + perLane, int(gridDim.x));
        double localMass = 0.;
        for (int i = first; i < last; ++i) {
            float2 v = a.stats[i];
            localMass += double(v.y) * expf(v.x - globalMax);
        }
        double prefix, total;
        Scan(scratch.scan).ExclusiveSum(localMass, prefix, total);
        if (!tid) {
            // One Philox4 draw per step matches offsets 0, 4, 8, ...
            drawMass = fmin(double(curand_uniform4(&rng).x), 1. - 0x1p-53) * total;
            selectedTile = gridDim.x - 1;
            selectedOffset = 1.;
        }
        __syncthreads();
        if (prefix <= drawMass && drawMass < prefix + localMass) {
            for (int i = first; i < last; ++i) {
                float2 v = a.stats[i];
                double next = prefix + double(v.y) * expf(v.x - globalMax);
                if (next > drawMass) {
                    selectedTile = i;
                    selectedOffset = (drawMass - prefix) / total;
                    break;
                }
                prefix = next;
            }
        }
        __syncthreads();
        const float factor = float(expf(tileMax - globalMax) / total);
        x *= factor; y *= factor;
        if (col < end) *reinterpret_cast<float2 *>(q + col) = make_float2(x, y);
        if (int(blockIdx.x) == selectedTile) {
            Scan(scratch.scan).ExclusiveSum(double(x) + y, prefix);
            __shared__ int picked;
            if (!tid) picked = INT_MAX;
            __syncthreads();
            if (prefix <= selectedOffset && selectedOffset < prefix + double(x) + y)
                atomicMin(&picked, col + int(selectedOffset >= prefix + x));
            using Last = cub::BlockReduce<int, Threads>;
            __shared__ typename Last::TempStorage lastStorage;
            int lastPositive = Last(lastStorage).Reduce(y > 0.f ? col + 1 : x > 0.f ? col : -1, cub::Max());
            __syncthreads();
            if (!tid) a.ids[step + 1] = float(picked == INT_MAX ? max(0, lastPositive) : picked);
        }
        grid.sync();
    }
}
} // namespace fastllm_draft_owner
