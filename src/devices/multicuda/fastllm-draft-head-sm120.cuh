#pragma once
#include "fastllm-draft-layer-sm120.cuh"

namespace sm120_draft {
constexpr int HeadVocab = 152576, HeadShard = HeadVocab / 8, HeadGrid = 298;
constexpr int HeadTile = HeadVocab / HeadGrid, BiasSlots = 256;
constexpr int HeadPanels = HeadShard / (16 * HeadGrid), HeadItems = HeadTile / Threads;
static_assert(HeadPanels * HeadGrid * 16 == HeadShard && HeadTile == HeadItems * Threads);
struct ProposalScratch {
    int ids[Rows];
    unsigned ready[8];
    // The winner publishes its token and replay epoch in one release store.
    unsigned long long selected[Rows - 1];
    // A late CTA can still read the previous step while others publish the next.
    float2 stats[2][HeadGrid];
    alignas(16) B logits[Rows][HeadVocab];
    int tags[BiasSlots][HeadGrid];
    B bias[BiasSlots][HeadVocab];
};
struct HeadArgs120 {
    const uint4 *weight;
    const B *input, *w1;
    const uint4 *w2;
    ProposalScratch *scratch;
    unsigned *epoch;
    int rank;
    const FastllmDraftSamplingRequest *request;
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
// The 149 MiB per-rank LM head is packed once in native BF16 MMA fragment order.
// Each pair of warps shares one 16-column panel and reduces two disjoint K slices.
struct HeadProjectTile {
    float parts[8][32][4];
    B out[HeadPanels][8][16];
};
__device__ __forceinline__ void ProjectDraftHead(HeadArgs120 a, HeadProjectTile &tile) {
    int t = threadIdx.x, warp = t / 32, lane = t % 32, panel = warp % HeadPanels;
    const uint4 *w = a.weight + (blockIdx.x * HeadPanels + panel) * 8192 + lane;
    const B *x = a.input + min(lane / 4, Rows - 1) * 4096 + (lane % 4) * 2;
    float accum[4]{}, odd[4]{};
    constexpr int stride = 8 / HeadPanels;
#pragma unroll 4
    for (int k = warp / HeadPanels; k < 256; k += 2 * stride) {
        uint4 evenW = w[k * 32], oddW = w[(k + stride) * 32];
        Mma(accum, evenW, Pair(x + k * 16), Pair(x + k * 16 + 8));
        Mma(odd, oddW, Pair(x + (k + stride) * 16), Pair(x + (k + stride) * 16 + 8));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        accum[j] += odd[j];
    *reinterpret_cast<float4 *>(tile.parts[warp][lane]) = make_float4(accum[0], accum[1], accum[2], accum[3]);
    __syncthreads();
    if (warp < HeadPanels) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sum = accum[j];
#pragma unroll
            for (int w = HeadPanels; w < 8; w += HeadPanels)
                sum += tile.parts[warp + w][lane][j];
            tile.out[panel][lane % 4 * 2 + j % 2][lane / 4 + j / 2 * 8] = __float2bfloat16_rn(sum);
        }
    }
    __syncthreads();
    if (t < Rows * HeadPanels * 2) {
        const int row = t / (HeadPanels * 2), pkt = t % (HeadPanels * 2);
        int offset = a.rank * HeadShard + blockIdx.x * HeadPanels * 16;
        reinterpret_cast<uint4 *>(a.scratch->logits[row] + offset)[pkt] =
            reinterpret_cast<const uint4 *>(tile.out[pkt / 2][row])[pkt % 2];
    }
}
// Each warp evaluates sixteen vocabulary rows against one replicated Markov query.
__device__ __forceinline__ void FillBiasTile(const HeadArgs120 &a, int previous, B *tile) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const B *x = a.w1 + size_t(previous) * 256 + lane % 4 * 2;
    for (int panel = warp; panel < HeadTile / 16; panel += Threads / 32) {
        const uint4 *weights = a.w2 + (blockIdx.x * (HeadTile / 16) + panel) * 512 + lane;
        float sum[4]{};
#pragma unroll
        for (int k = 0; k < 16; ++k)
            Mma(sum, weights[k * 32], Pair(x + k * 16), Pair(x + k * 16 + 8));
        if (lane % 4 == 0) {
            tile[panel * 16 + lane / 4] = __float2bfloat16_rn(sum[0]);
            tile[panel * 16 + lane / 4 + 8] = __float2bfloat16_rn(sum[2]);
        }
    }
}
struct HeadMaximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct ProposalBlock {
    typename cub::BlockReduce<float, Threads>::TempStorage reduce;
    typename cub::BlockScan<double, Threads>::TempStorage scan;
    float max, globalMax;
    double limits[Threads * 2 + 1], start, end, total;
    float draws[Rows];
    int hit, previous;
    unsigned slot;
};
__device__ __forceinline__ void SampleDraftRows(HeadArgs120 a, ProposalBlock &s,
                                                cooperative_groups::grid_group grid) {
    const int t = threadIdx.x, b = blockIdx.x, first = b * HeadTile;
    auto &p = *a.scratch;
    const auto request = *a.request;
    const float invTemp = 1.f / request.temperature;
    if (t < Rows) {
        curandStatePhilox4_32_10_t rng;
        curand_init(request.seed, 0, 4 * t, &rng);
        s.draws[t] = curand_uniform(&rng);
    }
    __syncthreads();
    const unsigned epoch = *a.epoch;
    for (int step = 0; step < Rows; ++step) {
        if (!t) {
            int previous = request.anchor;
            if (step) {
                unsigned long long word;
                do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(word)
                                 : "l"(&p.selected[step - 1]) : "memory");
                } while (unsigned(word >> 32) != epoch);
                previous = int(unsigned(word));
            }
            s.previous = previous;
            const unsigned slot = (unsigned(previous) * 2654435761u) >> 24;
            s.slot = slot;
            s.hit = p.tags[slot][b] == previous + 1;
        }
        __syncthreads();
        const int previous = s.previous;
        // Each CTA owns its cache tile. L2 and the bounded 256-row cache are shared by all seven steps.
        B *bias = p.bias[s.slot] + first;
        if (!s.hit) {
            FillBiasTile(a, previous, bias);
            if (t == 0)
                p.tags[s.slot][b] = previous + 1;
            __syncthreads();
        }
        float vals[HeadItems], localMax = -INFINITY;
#pragma unroll
        for (int j = 0; j < HeadItems; ++j) {
            int k = t * HeadItems + j;
            vals[j] = float(__float2bfloat16_rn(float(p.logits[step][first + k]) + float(bias[k])));
            localMax = fmaxf(localMax, vals[j]);
        }
        using Reduce = cub::BlockReduce<float, Threads>;
        float mx = Reduce(s.reduce).Reduce(localMax, HeadMaximum{});
        if (!t)
            s.max = mx;
        __syncthreads();
        float own = 0;
#pragma unroll
        for (int j = 0; j < HeadItems; ++j) {
            vals[j] = expf((vals[j] - s.max) * invTemp);
            own += vals[j];
        }
        float sum = Reduce(s.reduce).Sum(own);
        if (!t)
            p.stats[step % 2][b] = make_float2(s.max, sum);
        grid.sync();
        // Combine tile masses against a common maximum before selecting a token in vocabulary order.
        constexpr int Groups = (HeadGrid + Threads - 1) / Threads;
        float2 other[Groups];
        float maxOfTiles = -INFINITY;
#pragma unroll
        for (int i = 0; i < Groups; ++i) {
            int group = t * Groups + i;
            other[i] = group < HeadGrid ? p.stats[step % 2][group] : make_float2(-INFINITY, 0.f);
            maxOfTiles = fmaxf(maxOfTiles, other[i].x);
        }
        float globalMax = Reduce(s.reduce).Reduce(maxOfTiles, HeadMaximum{});
        if (!t)
            s.globalMax = globalMax;
        __syncthreads();
        double masses[Groups], weight = 0;
#pragma unroll
        for (int i = 0; i < Groups; ++i) {
            masses[i] = double(other[i].y) * expf((other[i].x - s.globalMax) * invTemp);
            weight += masses[i];
        }
        double prefix;
        using Scan = cub::BlockScan<double, Threads>;
        Scan(s.scan).ExclusiveSum(weight, prefix);
        if (!t)
            s.limits[0] = 0;
#pragma unroll
        for (int i = 0; i < Groups; ++i) {
            prefix += masses[i];
            s.limits[t * Groups + i + 1] = prefix;
        }
        __syncthreads();
        if (!t) {
            s.start = s.limits[b];
            s.end = s.limits[b + 1];
            s.total = s.limits[HeadGrid];
        }
        __syncthreads();
        const float normalize = float(expf((s.max - s.globalMax) * invTemp) / s.total);
#pragma unroll
        for (int j = 0; j < HeadItems; ++j)
            request.probabilities[size_t(step) * HeadVocab + first + t * HeadItems + j] = vals[j] * normalize;
        // curand_uniform can return 1. Clamp to the final nonempty CDF interval.
        const double draw = fmin(double(s.draws[step]) * s.total, nextafter(s.total, 0.));
        if (s.start <= draw && draw < s.end) {
            double local;
            Scan(s.scan).InclusiveSum(double(own), local);
            s.limits[t + 1] = local;
            __syncthreads();
            double target = fmin((draw - s.start) / (s.end - s.start) * s.limits[Threads],
                                 nextafter(s.limits[Threads], 0.));
            if (s.limits[t] <= target && target < s.limits[t + 1]) {
                double cdf = 0;
                int chosen = 0;
#pragma unroll
                for (int j = 0; j < HeadItems; ++j) {
                    if (vals[j] > 0)
                        chosen = j;
                    cdf += vals[j];
                    if (target - s.limits[t] < cdf)
                        break;
                }
                int token = first + t * HeadItems + chosen;
                p.ids[step] = token;
                if (step + 1 < Rows) {
                    unsigned long long word = (static_cast<unsigned long long>(epoch) << 32) | unsigned(token);
                    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::
                                 "l"(&p.selected[step]), "l"(word) : "memory");
                }
            }
        }
        __syncthreads();
    }
}
#endif
template <bool Owner>
__global__ __launch_bounds__(Threads, 3) void FusedDraftHeadProposalSm120(HeadArgs120 a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    auto grid = cooperative_groups::this_grid();
    __shared__ union {
        HeadProjectTile tile;
        ProposalBlock sampling;
    } shared;
    if (!threadIdx.x)
        PrefetchWeight(a.weight + blockIdx.x * HeadPanels * 8192, 65536);
    fastllm::cuda_pdl::Wait();
    ProjectDraftHead(a, shared.tile);
    // Publish every peer store before rank zero samples from the joined vocabulary.
    if (!Owner && threadIdx.x < Rows * HeadPanels * 2)
        __threadfence_system();
    grid.sync();
    if (blockIdx.x == 0) {
        __shared__ unsigned epoch;
        if (threadIdx.x == 0) {
            epoch = ++*a.epoch;
            CustomArStoreRelease(a.scratch->ready + a.rank, epoch);
        }
        __syncthreads();
        if (Owner && threadIdx.x < 8)
            while (CustomArLoadAcquire(a.scratch->ready + threadIdx.x) != epoch) {
            }
    }
    if (Owner) {
        grid.sync();
        SampleDraftRows(a, shared.sampling, grid);
    }
#else
    asm("trap;");
#endif
}
} // namespace sm120_draft
