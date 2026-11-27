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
template<int TilesPerBlock, int Rows = 1>
__device__ __forceinline__ void Propose(Proposal a) {
    const auto maximumOf = [](auto a, auto b) { return max(a, b); };
    auto grid = cooperative_groups::this_grid();
    __shared__ union { Reduce::TempStorage reduce; Scan::TempStorage scan; } scratch;
    __shared__ float logits[Rows][Tile], tileMax, globalMax;
    __shared__ int selectedTile;
    __shared__ double selectedOffset, drawMass;
    const int tid = threadIdx.x, lane = tid % 32;
    const int tiles = (a.vocab - 1) / Tile + 1;
    const int inputRow = (lane / 4) % Rows;
    curandStatePhilox4_32_10_t rng[Rows];
    if (!tid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) curand_init(*a.seed, row, 0, &rng[row]);
    }
    for (int step = 0; step < a.steps; ++step) {
        int prev = int(a.ids[step * Rows + inputRow]);
        if (prev < 0 || prev >= a.vocab) prev = 0;
        const B *latent = a.embedding + size_t(prev) * 256;
        float2 values[Rows][TilesPerBlock];
#pragma unroll
        for (int slot = 0; slot < TilesPerBlock; ++slot) {
            const int tile = blockIdx.x + slot * gridDim.x;
            if (tile >= tiles) break;
            const int begin = tile * Tile, end = min(begin + Tile, a.vocab);
            // The MMA query columns interleave request embeddings and share the vocabulary weight load.
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
#pragma unroll
                    for (int row = 0; row < Rows; ++row) {
                        const B *base = a.base + size_t(row * a.steps + step) * a.vocab;
                        const float scale = 1.f / a.temperature[row];
                        logits[row][n - begin] = float(B(float(base[n]) + float(B(sum[row])))) * scale;
                        logits[row][n + 8 - begin] = float(B(float(base[n + 8]) + float(B(sum[2 + row])))) * scale;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const int col = begin + 2 * tid;
                float x = col < end ? logits[row][2 * tid] : -INFINITY;
                float y = col < end ? logits[row][2 * tid + 1] : -INFINITY;
                float maximum = Reduce(scratch.reduce).Reduce(fmaxf(x, y), maximumOf);
                if (!tid) tileMax = maximum;
                __syncthreads();
                x = tileMax == -INFINITY ? 0.f : expf(x - tileMax);
                y = tileMax == -INFINITY ? 0.f : expf(y - tileMax);
                float mass = Reduce(scratch.reduce).Sum(x + y);
                if (!tid) a.stats[row * tiles + tile] = make_float2(tileMax, mass);
                values[row][slot] = make_float2(x, y);
                __syncthreads();
            }
        }
        grid.sync();

#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const float2 *stats = a.stats + row * tiles;
            float *q = a.q + size_t(row * a.steps + step) * a.vocab;
            // Each CTA can join the small array of tile statistics from L2. This
            // keeps its vocabulary tile in registers until both export and draw.
            float maximum = -INFINITY;
            for (int i = tid; i < tiles; i += Threads) maximum = fmaxf(maximum, stats[i].x);
            maximum = Reduce(scratch.reduce).Reduce(maximum, maximumOf);
            if (!tid) globalMax = maximum;
            __syncthreads();
            const int perLane = (tiles + Threads - 1) / Threads;
            const int first = tid * perLane, last = min(first + perLane, tiles);
            double localMass = 0.;
            for (int i = first; i < last; ++i) {
                float2 v = stats[i];
                localMass += double(v.y) * expf(v.x - globalMax);
            }
            double prefix, total;
            Scan(scratch.scan).ExclusiveSum(localMass, prefix, total);
            if (!tid) {
                // One Philox4 draw per step matches offsets 0, 4, 8, ...
                drawMass = fmin(double(curand_uniform4(&rng[row]).x), 1. - 0x1p-53) * total;
                selectedTile = tiles - 1;
                selectedOffset = 1.;
            }
            __syncthreads();
            if (prefix <= drawMass && drawMass < prefix + localMass) {
                for (int i = first; i < last; ++i) {
                    float2 v = stats[i];
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
            // Keep each logical tile's probability reduction and Philox order unchanged.
#pragma unroll
            for (int slot = 0; slot < TilesPerBlock; ++slot) {
                const int tile = blockIdx.x + slot * gridDim.x;
                if (tile >= tiles) break;
                const int begin = tile * Tile, end = min(begin + Tile, a.vocab), col = begin + 2 * tid;
                float x = values[row][slot].x, y = values[row][slot].y;
                const float factor = float(expf(stats[tile].x - globalMax) / total);
                x *= factor; y *= factor;
                if (col < end) *reinterpret_cast<float2 *>(q + col) = make_float2(x, y);
                if (tile == selectedTile) {
                    Scan(scratch.scan).ExclusiveSum(double(x) + y, prefix);
                    __shared__ int picked;
                    if (!tid) picked = INT_MAX;
                    __syncthreads();
                    if (prefix <= selectedOffset && selectedOffset < prefix + double(x) + y)
                        atomicMin(&picked, col + int(selectedOffset >= prefix + x));
                    using Last = cub::BlockReduce<int, Threads>;
                    __shared__ typename Last::TempStorage lastStorage;
                    int lastPositive = Last(lastStorage).Reduce(y > 0.f ? col + 1 : x > 0.f ? col : -1, maximumOf);
                    __syncthreads();
                    if (!tid) a.ids[(step + 1) * Rows + row] = float(picked == INT_MAX ? max(0, lastPositive) : picked);
                }
            }
            if constexpr (Rows > 1) __syncthreads();
        }
        grid.sync();
    }
}
// Keep the single-tile kernel's original register policy on larger devices.
__global__ __launch_bounds__(Threads) void FusedDraftHeadProposal(Proposal a) { Propose<1>(a); }

// Two vocabulary tiles per CTA fit a full proposal on smaller Hopper grids.
__global__ __launch_bounds__(Threads, 4) void FusedDraftHeadProposalPaired(Proposal a) {
    Propose<2>(a);
}
__global__ __launch_bounds__(Threads, 4) void FusedDraftHeadProposalBatch(Proposal a) {
    Propose<2, 2>(a);
}
// Each warp projects 16 vocabulary entries for all requests. One warp per
// request normalizes the tile, retaining its exponentials in registers.
// One CTA per request draws from the published q, and the cooperative grid
// reuses the selected IDs as the next step's embedding indices.
constexpr int WideTile = 512;

// FlashInfer's deterministic sampler scans 4 values per lane, then 32 lanes
// per warp and 32 warps per 4096-token slice. Keep that FP32 addition order.
__device__ __forceinline__ float ProposalWarpUpsweep(float value, int lane) {
    #pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        float other = __shfl_up_sync(0xffffffffu, value, d);
        if ((lane + 1) % (2 * d) == 0) value += other;
    }
    return value;
}

__device__ __forceinline__ float ProposalWarpExclusive(float value, int lane) {
    value = ProposalWarpUpsweep(value, lane);
    if (lane == 31) value = 0.f;
    #pragma unroll
    for (int d = 16; d; d /= 2) {
        float other = __shfl_xor_sync(0xffffffffu, value, d);
        const int position = (lane + 1) % (2 * d);
        if (position == 0) value = other + value;
        if (position == d) value = other;
    }
    return value;
}

// A single CTA per request makes the draw from the normalized q. Each thread
// first reduces one complete FlashInfer slice. Only the selected slice needs
// the lane-level CDF. Warp aggregates were published during normalization.
template<int BlockThreads = Threads>
__device__ __forceinline__ int SampleWideProposal(const float *q, int vocab, const float2 *warpStats,
        int warps, float u) {
    const int tid = threadIdx.x, lane = tid % 32;
    const int slices = (warps + 31) / 32;
    __shared__ float sliceMass[BlockThreads], warpOffsets[32], lower;
    __shared__ int chosen, picked, lastValid;
    if (!tid) { chosen = -1; picked = vocab; lastValid = -1; lower = 0; }
    __syncthreads();
    for (int base = 0; base < slices; base += BlockThreads) {
        const int slice = base + tid;
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int w = slice * 32 + i;
            if (w < warps) sum += warpStats[w].x;
        }
        sliceMass[tid] = sum;
        __syncthreads();
        if (!tid) {
            for (int i = 0; i < min(BlockThreads, slices - base); ++i) {
                const float next = lower + sliceMass[i];
                if (sliceMass[i] > 0.f) lastValid = base + i;
                if (next > u) { chosen = base + i; break; }
                lower = next;
            }
        }
        __syncthreads();
        if (chosen >= 0) break;
    }
    if (!tid && chosen < 0) { chosen = max(0, lastValid); lower = -INFINITY; }
    __syncthreads();
    const int slice = chosen;
    if (tid < 32) {
        const int w = slice * 32 + lane;
        warpOffsets[lane] = ProposalWarpExclusive(w < warps ? warpStats[w].y : 0.f, lane);
    }
    if (!tid) lastValid = -1;
    __syncthreads();
    for (int warp = tid / 32; warp < 32; warp += BlockThreads / 32) {
        const int first = (slice * 32 + warp) * 128 + lane * 4;
        float4 p = first < vocab ? *reinterpret_cast<const float4 *>(q + first) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float local[4]{p.x, p.y, p.z, p.w};
        float partial[4], sum = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { sum += local[j]; partial[j] = sum; }
        const float prefix = warpOffsets[warp] + ProposalWarpExclusive(sum, lane);
        int id = vocab, last = -1;
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (local[j] > 0.f) {
            last = first + j;
            if ((prefix + partial[j]) + lower > u) id = min(id, first + j);
        }
        for (int d = 16; d; d /= 2) {
            id = min(id, __shfl_down_sync(0xffffffffu, id, d));
            last = max(last, __shfl_down_sync(0xffffffffu, last, d));
        }
        if (!lane) {
            if (id < vocab) atomicMin(&picked, id);
            if (last >= 0) atomicMax(&lastValid, last);
        }
    }
    __syncthreads();
    return picked < vocab ? picked : max(0, lastValid);
}

template<int Rows>
__global__ __launch_bounds__(Threads, 4) void FusedDraftHeadProposalWide(Proposal a) {
    static_assert(Rows == 3 || Rows == 4);
    constexpr unsigned all = 0xffffffffu;
    constexpr int perLane = WideTile / 32;
    const auto grid = cooperative_groups::this_grid();
    const int tid = threadIdx.x, lane = tid % 32, row = tid / 32;
    const int first = blockIdx.x * WideTile, end = min(first + WideTile, a.vocab);
    const int tiles = gridDim.x;
    __shared__ float logits[Rows][WideTile];
    const int warps = (a.vocab - 1) / 128 + 1;
    // Both arrays fit in the existing rows-by-vocabulary sampler scratch.
    float2 *warpStats = a.stats + Rows * tiles;
    curandStatePhilox4_32_10_t rng;
    if (!tid && blockIdx.x < Rows) curand_init(*a.seed, blockIdx.x, 0, &rng);

    for (int step = 0; step < a.steps; ++step) {
        const int inputRow = (lane / 4) % Rows;
        int token = int(a.ids[size_t(step) * Rows + inputRow]);
        if (token < 0 || token >= a.vocab) token = 0;
        const B *embedding = a.embedding + size_t(token) * 256 + lane % 4 * 8;
        uint4 latent[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            latent[k] = *reinterpret_cast<const uint4 *>(embedding + k * 32);
        for (int group = first + row * 16; group < end; group += Threads / 32 * 16) {
            const int col = group + lane / 4;
            const B *w = a.weight + size_t(col) * 256 + lane % 4 * 8;
            float acc[4] = {};
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const auto lo = *reinterpret_cast<const uint4 *>(w + k * 32);
                const auto hi = *reinterpret_cast<const uint4 *>(w + 8 * 256 + k * 32);
                using fastllm::cuda::low_latency::projection::mma_bf16_16816;
                mma_bf16_16816(acc, lo.x, hi.x, lo.y, hi.y, latent[k].x, latent[k].y);
                mma_bf16_16816(acc, lo.z, hi.z, lo.w, hi.w, latent[k].z, latent[k].w);
            }
            // Each MMA lane owns two request columns and two vocabulary rows.
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                if (lane % 4 != r / 2) continue;
                const B *base = a.base + (size_t(r) * a.steps + step) * a.vocab;
                const float scale = 1.f / a.temperature[r];
                logits[r][col - first] = float(B(float(base[col]) + float(B(acc[r % 2])))) * scale;
                logits[r][col + 8 - first] = float(B(float(base[col + 8]) + float(B(acc[2 + r % 2])))) * scale;
            }
        }
        __syncthreads();
        float values[perLane], tileMax = -INFINITY;
        if (row < Rows) {
            #pragma unroll
            for (int i = 0; i < perLane; ++i) {
                values[i] = first + 32 * i + lane < end ? logits[row][32 * i + lane] : -INFINITY;
                tileMax = fmaxf(tileMax, values[i]);
            }
            #pragma unroll
            for (int d = 16; d; d /= 2) tileMax = fmaxf(tileMax, __shfl_xor_sync(all, tileMax, d));
            float sum = 0;
            #pragma unroll
            for (int i = 0; i < perLane; ++i) {
                values[i] = tileMax == -INFINITY ? 0.f : expf(values[i] - tileMax);
                sum += values[i];
            }
            #pragma unroll
            for (int d = 16; d; d /= 2) sum += __shfl_down_sync(all, sum, d);
            if (!lane) a.stats[row * tiles + blockIdx.x] = make_float2(tileMax, sum);
        }
        grid.sync();
        if (row < Rows) {
            const float2 *stats = a.stats + row * tiles;
            float maximum = -INFINITY;
            for (int i = lane; i < tiles; i += 32) maximum = fmaxf(maximum, stats[i].x);
            #pragma unroll
            for (int d = 16; d; d /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(all, maximum, d));

            double mass = 0.;
            for (int i = lane; i < tiles; i += 32) {
                if (stats[i].y > 0.f) mass += double(stats[i].y) * expf(stats[i].x - maximum);
            }
            #pragma unroll
            for (int d = 16; d; d /= 2) mass += __shfl_down_sync(all, mass, d);
            const double total = __shfl_sync(all, mass, 0);
            const float factor = total > 0. ? float(expf(tileMax - maximum) / total) : 0.f;
            float *q = a.q + (size_t(row) * a.steps + step) * a.vocab + first;
            #pragma unroll
            for (int i = 0; i < perLane; ++i) {
                const int col = 32 * i + lane;
                const float p = values[i] * factor;
                logits[row][col] = p;
                if (first + col < end) q[col] = p;
            }
            __syncwarp();
            using WarpReduce = cub::WarpReduce<float>;
            __shared__ typename WarpReduce::TempStorage reduce[Threads / 32];
            #pragma unroll
            for (int part = 0; part < WideTile / 128; ++part) {
                const int w = blockIdx.x * (WideTile / 128) + part;
                if (w >= warps) break;
                const float *p = &logits[row][part * 128 + lane * 4];
                float v[4]{p[0], p[1], p[2], p[3]};
                float sum = WarpReduce(reduce[row]).Sum(cub::ThreadReduce(v, cuda::std::plus<>{}));
                const float scan = ProposalWarpUpsweep(((v[0] + v[1]) + v[2]) + v[3], lane);
                const float warpSum = __shfl_sync(all, scan, 31);
                if (!lane) warpStats[row * warps + w] = make_float2(sum, warpSum);
            }
        }
        grid.sync();
        if (blockIdx.x < Rows) {
            __shared__ float uniform;
            if (!tid) uniform = curand_uniform4(&rng).x;
            __syncthreads();
            const int r = blockIdx.x;
            const float *q = a.q + (size_t(r) * a.steps + step) * a.vocab;
            const int id = SampleWideProposal(q, a.vocab, warpStats + r * warps, warps, uniform);
            if (!tid) a.ids[(size_t(step) + 1) * Rows + r] = float(id);
        }
        grid.sync();
    }
}
} // namespace fastllm_draft_owner
