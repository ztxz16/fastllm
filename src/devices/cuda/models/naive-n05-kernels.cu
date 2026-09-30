#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {
using BF16 = __nv_bfloat16;
__device__ float RoundBF16(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ float WarpSum(float x) {
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}
void Output(fastllm::Data &out, fastllm::DataType type, const std::vector<int> &dims) {
    out.dataType = type;
    out.Resize(dims);
    out.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    out.Allocate(false);
}
void CheckLaunch() {
    auto status = cudaGetLastError();
    fastllm::AssertInFastLLM(status == cudaSuccess,
        std::string("Naive-N0.5 CUDA: ") + cudaGetErrorString(status));
}

struct IndexerTopKOp : fastllm::MultiThreadBaseOp {
    const float *scores;
    int *indices;
    int queries, keys, topK, queryStart, first, step;

    void Run() override {
        std::vector<int> order(keys);
        // Each worker owns its scratch and writes disjoint query rows. Keep
        // the original comparator and partial_sort so ties and index order
        // remain identical to the serial path.
        for (int row = first; row < queries; row += step) {
            int end = queryStart + row + 1;
            std::iota(order.begin(), order.begin() + end, 0);
            const float *values = scores + (size_t)row * keys;
            auto before = [&](int a, int b) {
                return values[a] > values[b] || (values[a] == values[b] && a < b);
            };
            int valid = std::min(topK, end);
            std::partial_sort(order.begin(), order.begin() + valid, order.begin() + end, before);
            int *target = indices + (size_t)row * topK;
            std::copy_n(order.data(), valid, target);
            std::fill(target + valid, target + topK, -1);
        }
    }
};

void IndexerTopK(const float *scores, int *indices, int queries, int keys,
                int queryStart, int topK) {
    // Decode and small verification batches stay on the caller thread.
    // Longer prefill reuses the existing pool, respecting its active range.
    auto *pool = queries > 32 ? fastllm::GetAlivePool() : nullptr;
    int first = pool ? pool->curActivateThreadInterval.first : 0;
    int available = pool ? pool->curActivateThreadInterval.second - first : 1;
    int threads = std::min(std::max(1, available), (queries + 31) / 32);
    std::vector<IndexerTopKOp> tasks(threads);
    for (int t = 0; t < threads; ++t) {
        auto &task = tasks[t];
        task.scores = scores; task.indices = indices;
        task.queries = queries; task.keys = keys; task.topK = topK;
        task.queryStart = queryStart; task.first = t; task.step = threads;
    }
    if (threads == 1) {
        tasks[0].Run();
    } else {
        for (int t = 0; t < threads; ++t) pool->PushOp(first + t, &tasks[t]);
        for (int t = 0; t < threads; ++t) pool->Wait(first + t);
    }
}

__global__ void Rope(BF16 *data, const float *positions, int heads, int dim,
                     int rotaryDim, float theta) {
    int row = blockIdx.x, d = threadIdx.x;
    if (d >= rotaryDim / 2) return;
    float angle = positions[row / heads] * powf(theta, -2.0f * d / rotaryDim);
    float c = RoundBF16(cosf(angle)), s = RoundBF16(sinf(angle));
    BF16 *x = data + (size_t)row * dim;
    float a = (float)x[d], b = (float)x[d + rotaryDim / 2];
    // Match eager GPT-NeoX RoPE, including each BF16 multiplication.
    x[d] = __float2bfloat16(RoundBF16(a * c) - RoundBF16(b * s));
    x[d + rotaryDim / 2] = __float2bfloat16(RoundBF16(b * c) + RoundBF16(a * s));
}

__global__ void RoundIndexer(const BF16 *input, float *output, int stride,
                             int offset, bool fp8) {
    __shared__ float maximum[128];
    int d = threadIdx.x, row = blockIdx.x;
    float x = (float)input[(size_t)row * stride + offset + d];
    maximum[d] = fabsf(x);
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (d < step) maximum[d] = fmaxf(maximum[d], maximum[d + step]);
        __syncthreads();
    }
    if (fp8) {
        float scale = fmaxf(maximum[0], 1e-4f) / 448.0f;
        x = (float)__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / scale))) * scale;
    }
    output[(size_t)row * 128 + d] = x;
}

__global__ void IndexScores(const float *q, const float *k, const BF16 *weights,
                            float *scores, int heads, int keys, int queryStart) {
    int query = blockIdx.y;
    int key = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (key >= keys) return;
    float score = 0;
    for (int h = 0; h < heads; h++) {
        float dot = 0;
        for (int d = lane; d < 128; d += 32)
            dot += q[((size_t)query * heads + h) * 128 + d] * k[(size_t)key * 128 + d];
        dot = WarpSum(dot);
        score += fmaxf(dot, 0.0f) * (float)weights[query * heads + h];
    }
    if (lane == 0)
        scores[(size_t)query * keys + key] = key <= queryStart + query ? score : -INFINITY;
}

__device__ int KeyIndex(const int *indices, int query, int slot, int count,
                        int past, int window) {
    if (indices) return indices[(size_t)query * count + slot];
    return window ? max(0, past + query - window + 1) + slot : slot;
}

__global__ void AttentionScores(const BF16 *q, const BF16 *k, const int *indices,
                                float *scores, int heads, int kvHeads, int dim,
                                int keyStride, int keys, int count, int past, int window, bool causal) {
    int query = blockIdx.y, h = blockIdx.x;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int kvHead = h / (heads / kvHeads);
    for (int slot = warp; slot < count; slot += 8) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        float dot = 0;
        bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
        if (valid) {
            for (int d = lane; d < dim; d += 32)
                dot += (float)q[((size_t)query * heads + h) * dim + d] *
                       (float)k[(size_t)key * keyStride + kvHead * dim + d];
        }
        dot = WarpSum(dot);
        if (lane == 0)
            scores[((size_t)query * heads + h) * count + slot] =
                valid ? RoundBF16(RoundBF16(dot) * rsqrtf((float)dim)) : -INFINITY;
    }
}

__global__ void AttentionSoftmax(float *scores, const float *sink, int heads, int count) {
    __shared__ float scratch[256];
    int row = blockIdx.x, t = threadIdx.x;
    float *values = scores + (size_t)row * count;
    float maximum = sink ? sink[row % heads] : -INFINITY;
    for (int i = t; i < count; i += 256) maximum = fmaxf(maximum, values[i]);
    scratch[t] = maximum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] = fmaxf(scratch[t], scratch[t + step]);
        __syncthreads();
    }
    maximum = scratch[0];
    float sum = 0;
    for (int i = t; i < count; i += 256) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    if (t == 0 && sink) sum += expf(sink[row % heads] - maximum);
    // All warps must consume scratch[0] (the maximum) before warp zero
    // overwrites it with the denominator. Racecheck catches this on the
    // unfused long/sparse path even when ordinary runs happen to agree.
    __syncthreads();
    scratch[t] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] += scratch[t + step];
        __syncthreads();
    }
    sum = scratch[0];
    for (int i = t; i < count; i += 256)
        values[i] = sum > 0 ? RoundBF16(values[i] / sum) : 0;
}

// Short full attention and the 128-token sliding window fit in shared memory.
// Keep eager's BF16 score/probability rounding, sink and reduction order while
// removing the score tensor round-trip and two launches per layer.
__global__ void AttentionShort(const BF16 *q, const BF16 *k, const BF16 *v,
                               const int *indices, const float *sink, BF16 *out,
                               int heads, int kvHeads, int dim, int valueDim,
                               int keyStride, int keys, int count, int past, int window, bool causal) {
    __shared__ float scores[256], scratch[256];
    int query = blockIdx.y, h = blockIdx.x, t = threadIdx.x;
    int lane = t % 32, warp = t / 32, kvHead = h / (heads / kvHeads);
    for (int slot = warp; slot < count; slot += 8) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        float dot = 0;
        bool valid = key >= 0 && key < keys && (!causal || key <= past + query);
        if (valid) for (int d = lane; d < dim; d += 32)
            dot += (float)q[((size_t)query * heads + h) * dim + d] *
                   (float)k[(size_t)key * keyStride + kvHead * dim + d];
        dot = WarpSum(dot);
        if (lane == 0) scores[slot] = valid ? RoundBF16(RoundBF16(dot) * rsqrtf((float)dim)) : -INFINITY;
    }
    __syncthreads();
    float maximum = sink ? sink[h] : -INFINITY;
    if (t < count) maximum = fmaxf(maximum, scores[t]);
    scratch[t] = maximum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] = fmaxf(scratch[t], scratch[t + step]);
        __syncthreads();
    }
    maximum = scratch[0];
    float probability = t < count ? expf(scores[t] - maximum) : 0;
    float sum = probability;
    if (t == 0 && sink) sum += expf(sink[h] - maximum);
    __syncthreads();
    scratch[t] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] += scratch[t + step];
        __syncthreads();
    }
    if (t < count) scores[t] = scratch[0] > 0 ? RoundBF16(probability / scratch[0]) : 0;
    __syncthreads();
    for (int d = t; d < valueDim; d += 256) {
        float value = 0;
        for (int slot = 0; slot < count; ++slot) {
            int key = KeyIndex(indices, query, slot, count, past, window);
            if (key >= 0 && key < keys && (!causal || key <= past + query))
                value += scores[slot] * (float)v[((size_t)key * kvHeads + kvHead) * valueDim + d];
        }
        out[((size_t)query * heads + h) * valueDim + d] = __float2bfloat16(value);
    }
}

__global__ void AttentionValues(const float *prob, const BF16 *v, const int *indices,
                                BF16 *out, int heads, int kvHeads, int dim,
                                int keys, int count, int past, int window, bool causal) {
    int query = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    if (d >= dim) return;
    int kvHead = h / (heads / kvHeads);
    float sum = 0;
    const float *p = prob + ((size_t)query * heads + h) * count;
    for (int slot = 0; slot < count; slot++) {
        int key = KeyIndex(indices, query, slot, count, past, window);
        if (key >= 0 && key < keys && (!causal || key <= past + query))
            sum += p[slot] * (float)v[((size_t)key * kvHeads + kvHead) * dim + d];
    }
    out[((size_t)query * heads + h) * dim + d] = __float2bfloat16(sum);
}
}

void FastllmCudaNaiveRope(fastllm::Data &input, const fastllm::Data &positions,
                         int heads, int dim, int rotaryDim, float theta) {
    fastllm::AssertInFastLLM(input.dataType == fastllm::DataType::BFLOAT16 &&
                            rotaryDim > 0 && rotaryDim % 2 == 0 && rotaryDim <= dim,
                            "Invalid Naive-N0.5 RoPE input.");
    Rope<<<input.Count(0) / dim, 128>>>((BF16 *)input.cudaData,
        (const float *)positions.cudaData, heads, dim, rotaryDim, theta);
    CheckLaunch();
}

void FastllmCudaNaiveIndexer(const fastllm::Data &query, const fastllm::Data &weights,
                            const fastllm::Data &packedKeys, int heads, int dim,
                            int queryStart, int topK, bool fp8, fastllm::Data &indices) {
    using namespace fastllm;
    int queries = query.dims[1], keys = packedKeys.dims[1];
    int stride = packedKeys.dims[2];
    Data q, k, scores;
    Output(q, DataType::FLOAT32, {queries, heads, dim});
    Output(k, DataType::FLOAT32, {keys, dim});
    Output(scores, DataType::FLOAT32, {queries, keys});
    RoundIndexer<<<queries * heads, 128>>>((const BF16 *)query.cudaData,
        (float *)q.cudaData, dim, 0, fp8);
    RoundIndexer<<<keys, 128>>>((const BF16 *)packedKeys.cudaData,
        (float *)k.cudaData, stride, stride - dim, fp8);
    IndexScores<<<dim3((keys + 7) / 8, queries), 256>>>((const float *)q.cudaData,
        (const float *)k.cudaData, (const BF16 *)weights.cudaData,
        (float *)scores.cudaData, heads, keys, queryStart);
    CheckLaunch();
    // The stable tie rule in the released model selects the earliest key.
    // Keep selection bounded to one prefill chunk, and transfer only scores.
    scores.ToDevice(DataDevice::CPU);
    indices.ToDevice(DataDevice::CPU);
    indices.dataType = DataType::INT32;
    indices.Resize({queries, topK});
    indices.Allocate();
    IndexerTopK((const float *)scores.cpuData, (int *)indices.cpuData,
                queries, keys, queryStart, topK);
    indices.ToDevice(DataDevice::CUDA, query.dataDeviceIds);
}

void FastllmCudaNaiveAttention(const fastllm::Data &query, const fastllm::Data &key,
                              const fastllm::Data &value, const fastllm::Data &indices,
                              const fastllm::Data &sink, int heads, int kvHeads,
                              int dim, int valueDim, int pastLength, int window,
                              fastllm::Data &output, bool causal) {
    using namespace fastllm;
    int queries = query.dims[1], keys = key.dims[1];
    int count = indices.dims.empty() ? (window ? std::min(window + (causal ? 0 : queries - 1), keys) : keys) : indices.dims[1];
    const int *selected = indices.dims.empty() ? nullptr : (const int *)indices.cudaData;
    Output(output, DataType::BFLOAT16, {1, queries, heads * valueDim});
    if (count <= 256) {
        AttentionShort<<<dim3(heads, queries), 256>>>((const BF16 *)query.cudaData,
            (const BF16 *)key.cudaData, (const BF16 *)value.cudaData, selected,
            sink.dims.empty() ? nullptr : (const float *)sink.cudaData, (BF16 *)output.cudaData,
            heads, kvHeads, dim, valueDim, key.dims[2], keys, count, pastLength, window, causal);
        CheckLaunch();
        return;
    }
    Data scores;
    Output(scores, DataType::FLOAT32, {queries, heads, count});
    AttentionScores<<<dim3(heads, queries), 256>>>((const BF16 *)query.cudaData,
        (const BF16 *)key.cudaData, selected, (float *)scores.cudaData,
        heads, kvHeads, dim, key.dims[2], keys, count, pastLength, window, causal);
    AttentionSoftmax<<<queries * heads, 256>>>((float *)scores.cudaData,
        sink.dims.empty() ? nullptr : (const float *)sink.cudaData, heads, count);
    AttentionValues<<<dim3(heads, queries), 256>>>((const float *)scores.cudaData,
        (const BF16 *)value.cudaData, selected, (BF16 *)output.cudaData,
        heads, kvHeads, valueDim, keys, count, pastLength, window, causal);
    CheckLaunch();
}
