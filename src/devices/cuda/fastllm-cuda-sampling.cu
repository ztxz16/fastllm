#include "fastllm-cuda-sampling.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int Threads = 256, Items = 4, Tile = Threads * Items;
struct Rows {
    static constexpr int Capacity = 32;
    FastllmCudaSamplingRow row[Capacity];
};
static_assert(sizeof(Rows) + 128 < 4096, "sampling kernel argument limit");
struct Layout {
    float *scaled, *sorted;
    int *ids, *sortedIds, *offsets, *errors;
    double *sums;
    void *sortScratch;
    size_t bytes = 0, sortBytes = 0;
    Layout(void *base, int batch, int vocab, size_t temporary) {
        auto take = [&](size_t n) {
            size_t offset = bytes;
            bytes += (n + 255) / 256 * 256;
            return base ? (char *)base + offset : nullptr;
        };
        size_t count = size_t(batch) * vocab;
        int chunks = (vocab + Tile - 1) / Tile;
        scaled = (float *)take(count * sizeof(float));
        sorted = (float *)take(count * sizeof(float));
        ids = (int *)take(count * sizeof(int));
        sortedIds = (int *)take(count * sizeof(int));
        offsets = (int *)take((size_t(batch) + 1) * sizeof(int));
        errors = (int *)take(size_t(batch) * chunks * sizeof(int));
        sums = (double *)take(size_t(batch) * chunks * sizeof(double));
        sortScratch = take(temporary); sortBytes = temporary;
    }
};
bool SortBytes(int batch, int vocab, size_t &bytes) {
    if (batch <= 0 || vocab <= 0 || size_t(batch) * vocab > INT_MAX - Tile) return false;
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, bytes,
        (const float *)nullptr, (float *)nullptr, (const int *)nullptr, (int *)nullptr,
        batch * vocab, batch, (const int *)nullptr, (const int *)nullptr) == cudaSuccess;
}
__device__ int LowerBound(const int *ids, int count, int token) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (ids[mid] < token) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
__global__ void Prepare(Rows rows, int begin, int batch, int vocab, int chunks, Layout w) {
    int row = begin + blockIdx.y, tile = blockIdx.x;
    auto request = rows.row[blockIdx.y];
    int invalid = 0;
    for (int j = 0; j < Items; ++j) {
        int token = tile * Tile + threadIdx.x + j * Threads;
        if (token >= vocab) continue;
        bool allowed = true;
        if (request.allowedCount) {
            int i = LowerBound(request.allowedTokens, request.allowedCount, token);
            allowed = i < request.allowedCount && request.allowedTokens[i] == token;
        }
        float raw = allowed ? request.logits[token] : -INFINITY;
        if (allowed && request.repeatCount) {
            int i = LowerBound(request.repeatTokens, request.repeatCount, token);
            for (; i < request.repeatCount && request.repeatTokens[i] == token; ++i)
                raw = raw < 0.f ? raw * request.repeatPenalty : __fdiv_rn(raw, request.repeatPenalty);
        }
        float value = request.policy == FastllmCudaSamplingPolicy::LLMSampling
            ? raw * __fdiv_rn(1.f, request.temperature) : __fdiv_rn(raw, request.temperature);
        invalid |= isnan(raw) || raw == INFINITY || value == INFINITY;
        // Canonicalize signed zero; both CPU comparison and stable radix sort
        // must resolve equal scores by the original token order.
        value = value == 0.f ? 0.f : value;
        w.scaled[size_t(row) * vocab + token] = isnan(value) ? -INFINITY : value;
        w.ids[size_t(row) * vocab + token] = token;
    }
    __shared__ cub::BlockReduce<int, Threads>::TempStorage reduce;
    invalid = cub::BlockReduce<int, Threads>(reduce).Sum(invalid);
    if (!threadIdx.x) {
        w.errors[row * chunks + tile] = invalid;
        if (!tile) {
            w.offsets[row] = row * vocab;
            if (row + 1 == batch) w.offsets[batch] = batch * vocab;
        }
    }
}
__device__ double Mass(const float *scores, int token, int vocab, float cutoff) {
    if (token >= vocab || scores[token] < cutoff || scores[token] == -INFINITY) return 0.;
    // CPU reference evaluates exp in FP32 and accumulates its mass in FP64.
    return double(expf(scores[token] - scores[0]));
}
__global__ void PartialMass(Rows rows, int begin, int vocab, int chunks, Layout w) {
    int row = begin + blockIdx.y;
    const float *scores = w.sorted + size_t(row) * vocab;
    int k = rows.row[blockIdx.y].topK;
    float cutoff = k > 0 && k < vocab ? scores[k - 1] : -INFINITY;
    int limit = rows.row[blockIdx.y].policy == FastllmCudaSamplingPolicy::LLMSampling && k > 0 && k < vocab ? k : vocab;
    double sum = 0.;
    for (int j = 0; j < Items; ++j)
        sum += Mass(scores, blockIdx.x * Tile + threadIdx.x + j * Threads, limit, cutoff);
    __shared__ cub::BlockReduce<double, Threads>::TempStorage reduce;
    sum = cub::BlockReduce<double, Threads>(reduce).Sum(sum);
    if (!threadIdx.x) w.sums[row * chunks + blockIdx.x] = sum;
}
__global__ void Draw(Rows rows, int begin, int vocab, int chunks, Layout w,
        FastllmCudaSamplingResult *output) {
    using Scan = cub::BlockScan<double, Threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ double total, prefix, kept;
    __shared__ int tile, selected, invalid, lastKept, fallback;
    int row = begin + blockIdx.x;
    auto request = rows.row[blockIdx.x];
    const float *scores = w.sorted + size_t(row) * vocab;
    const double *sums = w.sums + row * chunks;
    float cutoff = request.topK > 0 && request.topK < vocab ? scores[request.topK - 1] : -INFINITY;
    int limit = request.policy == FastllmCudaSamplingPolicy::LLMSampling && request.topK > 0 && request.topK < vocab ? request.topK : vocab;
    if (!threadIdx.x) {
        total = 0.; invalid = !isfinite(scores[0]);
        for (int c = 0; c < chunks; ++c) { total += sums[c]; invalid |= w.errors[row * chunks + c]; }
        invalid |= !(total > 0.) || !isfinite(total);
        output[row] = {-1, invalid ? 1 : 0};
        kept = total;
        lastKept = limit - 1;
    }
    __syncthreads();
    if (invalid) return;
    // Locate only the top-p boundary tile, then scan it in stable token order.
    if (request.topP < 1.f) {
        if (!threadIdx.x) {
            prefix = 0.; tile = 0; selected = Tile;
            while (tile + 1 < chunks && (prefix + sums[tile]) / total <= request.topP)
                prefix += sums[tile++];
        }
        __syncthreads();
        double values[Items];
        for (int j = 0; j < Items; ++j)
            values[j] = Mass(scores, tile * Tile + threadIdx.x * Items + j, limit, cutoff);
        Scan(scan).InclusiveSum(values, values);
        for (int j = 0; j < Items; ++j)
            if ((prefix + values[j]) / total > request.topP)
                atomicMin(&selected, int(threadIdx.x) * Items + j);
        __syncthreads();
        for (int j = 0; j < Items; ++j)
            if (int(threadIdx.x) * Items + j == selected) {
                kept = prefix + values[j]; lastKept = tile * Tile + selected;
            }
        __syncthreads();
    }
    double draw = request.uniform * kept;
    if (!threadIdx.x) {
        prefix = 0.; tile = 0; selected = Tile; fallback = -1;
        while (tile + 1 < chunks && (tile + 1) * Tile <= lastKept && prefix + sums[tile] <= draw)
            prefix += sums[tile++];
    }
    __syncthreads();
    double values[Items];
    for (int j = 0; j < Items; ++j) {
        int i = int(threadIdx.x) * Items + j;
        values[j] = tile * Tile + i <= lastKept ? Mass(scores, tile * Tile + i, limit, cutoff) : 0.;
        if (values[j] > 0.) atomicMax(&fallback, i);
    }
    Scan(scan).InclusiveSum(values, values);
    for (int j = 0; j < Items; ++j)
        if (tile * Tile + threadIdx.x * Items + j <= lastKept && prefix + values[j] > draw)
            atomicMin(&selected, int(threadIdx.x) * Items + j);
    __syncthreads();
    if (!threadIdx.x) {
        // At u immediately below 1, different FP64 reduction orders may
        // round the draw to the last cumulative mass. Keep its last token.
        if (selected == Tile) selected = fallback;
        int token = tile * Tile + selected;
        output[row] = selected >= 0 && token < vocab && selected < Tile
            ? FastllmCudaSamplingResult{w.sortedIds[size_t(row) * vocab + token], 0}
            : FastllmCudaSamplingResult{-1, 1};
    }
}
}

size_t FastllmCudaSampleLogitsWorkspaceBytes(int batch, int vocab) {
    size_t temporary = 0;
    return SortBytes(batch, vocab, temporary) ? Layout(nullptr, batch, vocab, temporary).bytes : 0;
}
bool FastllmCudaSampleLogits(const FastllmCudaSamplingRow *rows, int batch,
        int vocab, FastllmCudaSamplingResult *output, void *workspace,
        size_t workspaceBytes, void *stream) {
    size_t temporary = 0;
    if (!rows || !output || !workspace || uintptr_t(workspace) % 256 || !SortBytes(batch, vocab, temporary)) return false;
    Layout w(workspace, batch, vocab, temporary);
    if (workspaceBytes < w.bytes) return false;
    for (int r = 0; r < batch; ++r)
        if (!rows[r].logits || !std::isfinite(rows[r].temperature) || rows[r].temperature <= 0.f ||
            !std::isfinite(rows[r].topP) || rows[r].topP <= 0.f || rows[r].topP > 1.f ||
            !std::isfinite(rows[r].uniform) || rows[r].uniform < 0. || rows[r].uniform >= 1. ||
            rows[r].allowedCount < 0 || (rows[r].allowedCount && !rows[r].allowedTokens) ||
            rows[r].repeatCount < 0 || (rows[r].repeatCount && !rows[r].repeatTokens) ||
            (rows[r].policy != FastllmCudaSamplingPolicy::TopKWithTies &&
             rows[r].policy != FastllmCudaSamplingPolicy::LLMSampling) ||
            !std::isfinite(rows[r].repeatPenalty) || rows[r].repeatPenalty <= 0.f) return false;
    cudaStream_t cudaStream = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    int chunks = (vocab + Tile - 1) / Tile;
    auto launch = [&](bool prepare) {
        for (int begin = 0; begin < batch; begin += Rows::Capacity) {
            Rows params{};
            int count = std::min(batch - begin, Rows::Capacity);
            for (int i = 0; i < count; ++i) params.row[i] = rows[begin + i];
            if (prepare) Prepare<<<dim3(chunks, count), Threads, 0, cudaStream>>>(params, begin, batch, vocab, chunks, w);
            else {
                PartialMass<<<dim3(chunks, count), Threads, 0, cudaStream>>>(params, begin, vocab, chunks, w);
                Draw<<<count, Threads, 0, cudaStream>>>(params, begin, vocab, chunks, w, output);
            }
        }
    };
    launch(true);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cub::DeviceSegmentedRadixSort::SortPairsDescending(w.sortScratch, w.sortBytes,
        w.scaled, w.sorted, w.ids, w.sortedIds, batch * vocab, batch,
        w.offsets, w.offsets + 1, 0, 32, cudaStream) != cudaSuccess) return false;
    launch(false);
    return cudaGetLastError() == cudaSuccess;
}
