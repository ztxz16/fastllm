#pragma once
#include "fastllm-draft-layer-sm120.cuh"

// One TP8 request. KV bindings and absolute position remain live during replay.
namespace sm120_draft {
constexpr int Heads = 4, Dim = 128, Window = 1024, Pitch = 1040, Grid = 256, Splits = 4;
struct Workspace {
    Exchange exchange;
    float split[Splits][768][8];
    B query[Rows][Heads][Dim], scores[Heads][Rows][Pitch], attention[Rows][Heads * Dim];
};
static_assert(offsetof(Workspace, exchange) == 0);
struct Args {
    const B *input;
    const uint4 *qkv, *out;
    const float *qNorm, *kNorm, *gamma;
    B *hidden, *normed;
    B *const *cache;
    const int *live;
    Workspace *w;
    CustomArPointers peers;
    CustomArRankSignals signals;

    float eps, theta;
    int rank;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
__device__ __forceinline__ void ProjectQkv(const Args &a, float (&part)[8][128]) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, token = lane / 4;
    int tile = blockIdx.x / Splits, split = blockIdx.x % Splits;
    const B *x = a.input + min(token, Rows - 1) * 4096 + split * 1024 + lane % 4 * 2;
    const uint4 *weight = a.qkv + (tile * 256 + split * 64) * 32 + lane;
    float acc[4]{};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int kt = j * 8 + warp;
        Mma(acc, weight[kt * 32], Pair(x + kt * 16), Pair(x + kt * 16 + 8));
    }
    ReduceWarps(acc, part);
    if (threadIdx.x < 128)
        a.w->split[split][tile * 16 + threadIdx.x / 8][threadIdx.x % 8] = ColumnSum(part, threadIdx.x);
}
__device__ __forceinline__ B Raw(const Args &a, int col, int row) {
    float sum = 0;
#pragma unroll
    for (int s = 0; s < Splits; ++s)
        sum += a.w->split[s][col][row];
    return __float2bfloat16_rn(sum);
}
__device__ __forceinline__ void NormRope(const Args &a, int past, float *sums, B *normalized) {
    // Explicit products preserve the two BF16 roundings on each rotary pair.
    int row = blockIdx.x / 5, head = blockIdx.x % 5, t = threadIdx.x;
    float x = t < Dim ? float(Raw(a, head * Dim + t, row)) : 0.f;
    float ss = fastllm::cuda_norm::WarpFirstBlockSum256(x * x, sums);
    if (t < Dim) {
        const float *weight = head == Heads ? a.kNorm : a.qNorm;
        normalized[t] =
            __float2bfloat16_rn(float(__float2bfloat16_rn(x * rsqrtf(ss / Dim + a.eps))) * weight[t]);
        if (head == Heads)
            a.cache[1][(past + row) * Dim + t] = Raw(a, 640 + t, row);
    }
    __syncthreads();
    if (t < Dim / 2) {
        float angle = float(*a.live - 1 + row) * powf(a.theta, -2.f * t / Dim);
        B c = __float2bfloat16_rn(cosf(angle)), s = __float2bfloat16_rn(sinf(angle));
        B x0 = normalized[t], x1 = normalized[t + Dim / 2];
        B *dst = head == Heads ? a.cache[0] + (past + row) * Dim : a.w->query[row][head];
        dst[t] = __hsub(__hmul_rn(x0, c), __hmul_rn(x1, s));
        dst[t + Dim / 2] = __hadd(__hmul_rn(x1, c), __hmul_rn(x0, s));
    }
}
__device__ __forceinline__ void Scores(const Args &a, int past) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int tile = blockIdx.x * 8 + warp, head = tile % Heads, first = tile / Heads * 16;
    int count = past + Rows;
    if (first >= count)
        return;
    int g = lane / 4, k = lane % 4 * 2;
    const B *lo = a.cache[0] + min(first + g, count - 1) * Dim + k;
    const B *hi = a.cache[0] + min(first + g + 8, count - 1) * Dim + k;
    const B *q = a.w->query[min(g, Rows - 1)][head] + k;
    float acc[4]{};
#pragma unroll
    for (int j = 0; j < Dim; j += 16)
        Mma(acc, make_uint4(Pair(lo + j), Pair(hi + j), Pair(lo + j + 8), Pair(hi + j + 8)), Pair(q + j),
            Pair(q + j + 8));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = k + i % 2, key = first + g + i / 2 * 8;
        if (row < Rows && key < count)
            a.w->scores[head][row][key] =
                key >= max(0, past + row - Window + 1)
                    ? __float2bfloat16_rn(float(__float2bfloat16_rn(acc[i])) * rsqrtf(float(Dim)))
                    : __float2bfloat16_rn(-INFINITY);
    }
}
// A CTA owns a complete query/head softmax. Publish rounded probabilities
// once for all value tiles. Preserve the reference 256-lane denominator tree.
__device__ __forceinline__ void Softmax(const Args &a, int count, float *tmp) {
    int t = threadIdx.x, head = blockIdx.x / Rows, row = blockIdx.x % Rows;
    B *p = a.w->scores[head][row];
    float val[5], top = -INFINITY;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        int key = t + i * 256;
        val[i] = key < count ? float(p[key]) : -INFINITY;
        top = fmaxf(top, val[i]);
    }
    tmp[t] = top;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (t < d)
            tmp[t] = fmaxf(tmp[t], tmp[t + d]);
        __syncthreads();
    }
    top = tmp[0];
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        val[i] = expf(val[i] - top);
        sum += val[i];
    }
    __syncthreads();
    tmp[t] = sum;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (t < d)
            tmp[t] += tmp[t + d];
        __syncthreads();
    }
    sum = tmp[0];
#pragma unroll
    for (int i = 0; i < 5; ++i)
        if (t + i * 256 < ((count + 15) & ~15))
            p[t + i * 256] = __float2bfloat16_rn(val[i] / sum);
}
__device__ __forceinline__ void Values(const Args &a, int count, float (&part)[8][128]) {
    int t = threadIdx.x, warp = t / 32, lane = t % 32;
    int head = blockIdx.x / 8, col = blockIdx.x % 8 * 16;
    auto &p = a.w->scores[head];
    int g = lane / 4, k = lane % 4 * 2;
    float acc[4]{};
    for (int first = warp * 16; first < count; first += 128) {
        const B *v = a.cache[1];
        unsigned word[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int key = first + k + j / 2 * 8, ch = col + g + j % 2 * 8;
            B lo = v[min(key, count - 1) * Dim + ch], hi = v[min(key + 1, count - 1) * Dim + ch];
            word[j] = __bfloat16_as_ushort(lo) | (unsigned(__bfloat16_as_ushort(hi)) << 16);
        }
        const B *pr = p[min(g, Rows - 1)] + first + k;
        Mma(acc, make_uint4(word[0], word[1], word[2], word[3]), Pair(pr), Pair(pr + 8));
    }
    ReduceWarps(acc, part);
    if (t < 128 && t % 8 < Rows)
        a.w->attention[t % 8][head * Dim + col + t / 8] = __float2bfloat16_rn(ColumnSum(part, t));
}
#endif
__global__ __launch_bounds__(Threads, 3) void FusedDraftAttentionSm120(Args a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    auto grid = cooperative_groups::this_grid();
    int b = blockIdx.x, t = threadIdx.x;
    if (t == 0) {
        if (b < 48 * Splits) PrefetchWeight(a.qkv + b * 2048, 32768);
        PrefetchWeight(a.out + b * 1024, 16384);
    }
    fastllm::cuda_pdl::Wait();
    int past = min(*a.live - 1, Window - 1);
    __shared__ union {
        float part[8][128];
        float lanes[256];
    } tile;
    auto &part = tile.part;
    __shared__ B normalized[Dim];
    if (b < 48 * Splits)
        ProjectQkv(a, part);
    grid.sync();
    if (b < Rows * 5)
        NormRope(a, past, tile.lanes, normalized);
    // Fill O fragments while other CTAs normalize Q/K and append the suffix.
    uint4 weight[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        weight[i] = a.out[(b * 4 + i) * Threads + t];
    grid.sync();
    Scores(a, past);
    grid.sync();
    if (b < Heads * Rows)
        Softmax(a, past + Rows, tile.lanes);
    grid.sync();
    if (b < Heads * 8)
        Values(a, past + Rows, part);
    grid.sync();
    ProjectAndSend(&a.w->attention[0][0], a.peers, a.rank, weight, part, b);
    FinishLayer(a, a.w->exchange, part, grid);
#else
    asm("trap;");
#endif
}
} // namespace sm120_draft
