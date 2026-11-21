#include "fastllm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "devices/cuda/fastllm-cuda-conv.cuh"
#include "devices/cuda/fastllm-cuda.cuh"

void FastllmCudaBf16ExactRows(const void *, const void *, void *, int, int, int);

void FastllmCudaGdnBf16DecodeRows(const fastllm::Data &, const fastllm::Data &, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, const std::vector<fastllm::Data*> &, int, int, fastllm::Data &);

void FastllmCudaGdnBf16VerifyRows(const fastllm::Data &, const fastllm::Data &, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, const std::vector<fastllm::Data*> &,
    std::vector<std::pair<fastllm::Data,fastllm::Data>> &, int, int, int, bool, fastllm::Data &);

namespace fastllm {
namespace {
__device__ float LogSumExp(float a, float b) {
    float m = fmaxf(a, b);
    return m + log1pf(expf(-fabsf(a - b)));
}

// Follow the float4 summation and the two warp reductions used by FP32 RMSNorm.
__device__ __forceinline__ void RouterRmsScale(float squares, float *partial, float &scale) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int s = 16; s; s >>= 1) squares += __shfl_down_sync(0xffffffff, squares, s);
    if (lane == 0) partial[warp] = squares;
    __syncthreads();
    if (warp == 0) {
        float total = partial[lane];
        for (int s = 16; s; s >>= 1) total += __shfl_down_sync(0xffffffff, total, s);
        if (lane == 0) scale = rsqrtf(total / 8192.0f + 1e-6f);
    }
    __syncthreads();
}

__global__ void RouterNormalize(const __nv_bfloat16 *streams, float4 *normalized) {
    constexpr int width = 8192;
    const auto *row = reinterpret_cast<const uint2 *>(streams + size_t(blockIdx.x) * width);
    const int tid = threadIdx.x;
    float4 values[2];
    float squares = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint2 packed = row[tid + 1024 * i];
        const auto *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
        const float2 xy = __bfloat1622float2(pairs[0]), zw = __bfloat1622float2(pairs[1]);
        values[i] = make_float4(xy.x, xy.y, zw.x, zw.y);
        squares += xy.x * xy.x + xy.y * xy.y + zw.x * zw.x + zw.y * zw.y;
    }
    __shared__ float partial[32], scale;
    RouterRmsScale(squares, partial, scale);
    normalized += size_t(blockIdx.x) * (width / 4) + tid;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float4 v = values[i];
        normalized[1024 * i] = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }
}

// One CTA per logit keeps the 8192 normalized values on chip. Match the
// FP32 RMSNorm's float4 reduction and the FP32 GEMV's ascending 256-lane tree.
__global__ void RouterLogits(const __nv_bfloat16 *streams, const float *weight,
                             const float *bias, float *logits) {
    constexpr int width = 8192;
    streams += blockIdx.y * width;
    logits += blockIdx.y * 8;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ __align__(16) float values[width];
    __shared__ float partial[32], scale;
    float squares = 0.f;
    for (int i = tid * 4; i < width; i += 1024 * 4) {
        float4 v = make_float4(float(streams[i]), float(streams[i + 1]),
                               float(streams[i + 2]), float(streams[i + 3]));
        squares += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        reinterpret_cast<float4 *>(values)[i / 4] = v;
    }
    RouterRmsScale(squares, partial, scale);
    if (tid < 256) {
        float dot = 0.f;
        for (int i = tid; i < width; i += 256)
            dot = fmaf(values[i] * scale, weight[blockIdx.x * width + i], dot);
        for (int s = 1; s < 32; s *= 2) dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0) partial[warp] = dot;
    }
    __syncthreads();
    if (warp == 0) {
        float dot = lane < 8 ? partial[lane] : 0.f;
        for (int s = 1; s < 8; s *= 2) dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0) logits[blockIdx.x] = dot + bias[blockIdx.x];
    }
}

// One block owns one token. The routing matrix is [destination, source].
// Cast coefficients before mixing, matching the BF16 matmuls in the checkpoint.
__global__ void Route(const float *logits, const __nv_bfloat16 *streams, int dim,
                      int iterations, __nv_bfloat16 *branch, __nv_bfloat16 *mix) {
    int t = blockIdx.x;
    __shared__ float pre[2];
    if (threadIdx.x == 0) {
        const float *l = logits + t * 8;
        for (int i = 0; i < 2; ++i) {
            pre[i] = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-l[i]))));
            mix[t * 6 + i] = __float2bfloat16(2.f / (1.f + expf(-l[i + 2])));
        }
        float a = l[4], b = l[5], c = l[6], d = l[7];
        for (int i = 0; i < iterations; ++i) {
            float r0 = LogSumExp(a, b), r1 = LogSumExp(c, d);
            a -= r0; b -= r0; c -= r1; d -= r1;
            float c0 = LogSumExp(a, c), c1 = LogSumExp(b, d);
            a -= c0; c -= c0; b -= c1; d -= c1;
        }
        float coeff[4] = {a, b, c, d};
        for (int i = 0; i < 4; ++i) mix[t * 6 + 2 + i] = __float2bfloat16(expf(coeff[i]));
    }
    __syncthreads();
    streams += (size_t)t * 2 * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        branch[(size_t)t * dim + i] = __float2bfloat16(pre[0] * __bfloat162float(streams[i]) +
                                                      pre[1] * __bfloat162float(streams[dim + i]));
    }
}

__global__ void Merge(const __nv_bfloat16 *branch, const __nv_bfloat16 *mix,
                      const __nv_bfloat16 *streams, int dim, __nv_bfloat16 *output) {
    int tiles = (dim + blockDim.x - 1) / blockDim.x;
    int t = blockIdx.x / tiles;
    int d = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    size_t base = (size_t)t * 2 * dim + d;
    float x = __bfloat162float(streams[base]), y = __bfloat162float(streams[base + dim]);
    float update = __bfloat162float(branch[(size_t)t * dim + d]);
    for (int s = 0; s < 2; ++s) {
        const __nv_bfloat16 *row = mix + t * 6 + 2 + 2 * s;
        float mixed = __bfloat162float(__float2bfloat16(x * __bfloat162float(row[0]) + y * __bfloat162float(row[1])));
        float delta = __bfloat162float(__float2bfloat16(update * __bfloat162float(mix[t * 6 + s])));
        output[base + s * dim] = __float2bfloat16(mixed + delta);
    }
}

// Advance the four-tap history, preserving the convolution and SiLU BF16
// rounding points before the FP32 recurrent update.
struct ConvStateRows {
    static constexpr int capacity = 32;
    float4 *state[capacity];
    float4 *prefix[2][capacity];
};
struct NoConvStateRows {};
template <bool VERIFY = false, bool BATCH = false>
__global__ void ConvDecode(const __nv_bfloat16 *input, const float4 *weights,
                           float4 *state, __nv_bfloat16 *beta, float *out,
                           int channels, int heads, int rows = 1,
                           int inputPitch = 0, int gatePitch = 0,
                           typename std::conditional<BATCH, ConvStateRows, NoConvStateRows>::type states = {}) {
    if constexpr (BATCH) {
        state = states.state[blockIdx.y];
        const size_t first = size_t(blockIdx.y) * rows;
        input += first * (VERIFY ? inputPitch : channels);
        beta += first * (VERIFY ? gatePitch : heads);
        out += first * channels;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float4 prev = state[i], w = weights[i];
    for (int t = 0; t < (VERIFY ? rows : 1); ++t) {
        if (i < heads) {
            float b = __bfloat162float(beta[t * gatePitch + i]);
            beta[t * gatePitch + i] = __float2bfloat16_rn(1.f / (1.f + expf(-b)));
        }
        float4 next = make_float4(prev.y, prev.z, prev.w, __bfloat162float(input[t * inputPitch + i]));
        if constexpr (VERIFY) {
            (t + 1 < rows ? states.prefix[t][blockIdx.y] : state)[i] = next;
        } else {
            state[i] = next;
        }
        float value = cuda_conv::Accumulate<__nv_bfloat16, false>(0.f, next.x, w.x);
        value = cuda_conv::Accumulate<__nv_bfloat16, false>(value, next.y, w.y);
        value = cuda_conv::Accumulate<__nv_bfloat16, false>(value, next.z, w.z);
        value = cuda_conv::Accumulate<__nv_bfloat16, false>(value, next.w, w.w);
        float x = __bfloat162float(__float2bfloat16_rn(value));
        out[t * channels + i] = __bfloat162float(__float2bfloat16_rn(x / (1.f + expf(-x))));
        prev = next;
    }
}

// Match the BF16 rounding boundaries of the unfused activation and multiply.
__global__ void GateUp(const __nv_bfloat16 *pair, __nv_bfloat16 *out, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    pair += blockIdx.y * 2 * width;
    out += blockIdx.y * width;
    float x = __bfloat162float(pair[i]);
    float gate = __bfloat162float(__float2bfloat16_rn(x / (1.0f + expf(-x))));
    out[i] = __float2bfloat16_rn(gate * __bfloat162float(pair[width + i]));
}

// One 64-thread block per 128-wide GDN value head. Keep the pairwise
// sum of squares and two-warp reduction used by the BF16 RMSNorm operator.
__global__ void GdnOutput(const float *core, const __nv_bfloat16 *z,
                          const float *weight, float eps, __nv_bfloat16 *out, int heads = 0, int gatePitch = 0) {
    int lane = threadIdx.x, col = lane * 2, i = blockIdx.x * 128 + col;
    if (gatePitch) z += (blockIdx.x / heads) * (gatePitch - heads * 128);
    float a = __bfloat162float(__float2bfloat16_rn(core[i]));
    float b = __bfloat162float(__float2bfloat16_rn(core[i + 1]));
    float sum = a * a + b * b;
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float partial[2], scale;
    if ((lane & 31) == 0) partial[lane / 32] = sum;
    __syncthreads();
    if (lane == 0) scale = rsqrtf((partial[0] + partial[1]) / 128.f + eps);
    __syncthreads();
    for (int j = 0; j < 2; ++j) {
        float normalized = __bfloat162float(__float2bfloat16_rn((j ? b : a) * scale));
        float scaled = __bfloat162float(__float2bfloat16_rn(normalized * __bfloat162float(__float2bfloat16_rn(weight[col + j]))));
        float gate = __bfloat162float(z[i + j]);
        float activated = cuda_conv::SiluPromoted(gate);
        out[i + j] = __float2bfloat16_rn(scaled * activated);
    }
}

__global__ void ApplyRotary(__nv_bfloat16 *data, const __nv_bfloat16 *table,
                            int heads, int dim, int halfRotary, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / halfRotary, d = i % halfRotary;
    const auto *cs = table + (row / heads) * 2 * halfRotary;
    auto *x = data + (size_t)row * dim;
    float a = __bfloat162float(x[d]), b = __bfloat162float(x[d + halfRotary]);
    float c = __bfloat162float(cs[d]), s = __bfloat162float(cs[d + halfRotary]);
    // Transformers applies rotary as two BF16 products, then a BF16 add.
    x[d] = __float2bfloat16(float(__float2bfloat16(a * c)) - float(__float2bfloat16(b * s)));
    x[d + halfRotary] = __float2bfloat16(float(__float2bfloat16(b * c)) + float(__float2bfloat16(a * s)));
}

void AllocateLike(Data &out, const Data &input, const std::vector<int> &shape, DataType type = BFLOAT16) {
    out.ToDevice(input.dataDevice, input.dataDeviceIds, false);
    out.dataType = type;
    out.UpdateUnitSize();
    out.Resize(shape);
    out.Allocate(false);
}
void CheckLaunch() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
}

void SenseNovaCopyState(const Data &src, Data &dst) {
    if (dst.dims.empty()) AllocateLike(dst, src, src.dims, src.dataType);
    cudaError_t status = cudaMemcpyAsync(dst.cudaData, src.cudaData, src.GetBytes(), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void SenseNovaBf16GateProjection(const Data &input, Data &weight, Data &output) {
    weight.ToDevice(input.dataDevice, input.dataDeviceIds);
    AllocateLike(output, input, {1, input.dims[1], weight.dims[0]});
    FastllmCudaBf16ExactRows(input.cudaData, weight.cudaData, output.cudaData,
                            input.dims[1], input.dims.back(), weight.dims[0]);
    CheckLaunch();
}

// GDN, paged attention and the router each produce one contiguous tensor per
// prompt. Use the copy engine for whole tensors (up to several MiB per row).
void SenseNovaJoinPrefillRows(const std::vector<Data> &rows, int length, Data &out) {
    const Data &first = rows.front();
    const size_t bytes = first.GetBytes();
    const int width = first.Count(0) / length;
    AllocateLike(out, first, {1, int(rows.size()) * length, width}, first.dataType);
    auto *dst = static_cast<uint8_t *>(out.cudaData);
    for (const Data &row : rows) {
        if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(dst, row.cudaData, bytes))
            throw std::runtime_error("SenseNova prefill row copy failed.");
        dst += bytes;
    }
}

void SenseNovaRouterNormalize(const Data &streams, Data &out) {
    AllocateLike(out, streams, {1, streams.dims[1], 8192}, FLOAT32);
    RouterNormalize<<<streams.dims[1], 1024>>>((const __nv_bfloat16 *)streams.cudaData,
                                              (float4 *)out.cudaData);
    CheckLaunch();
}

void SenseNovaRouterLogits(const Data &streams, Data &weight, Data &bias, Data &out) {
    weight.ToDevice(streams.dataDevice, streams.dataDeviceIds);
    bias.ToDevice(streams.dataDevice, streams.dataDeviceIds);
    AllocateLike(out, streams, {1, streams.dims[1], 8}, FLOAT32);
    RouterLogits<<<dim3(8, streams.dims[1]), 1024>>>((const __nv_bfloat16 *)streams.cudaData,
        (const float *)weight.cudaData, (const float *)bias.cudaData, (float *)out.cudaData);
    CheckLaunch();
}

void SenseNovaConvDecode(const Data &input, const Data &weight, Data &state, Data &beta, Data &out) {
    int channels = input.dims.back();
    AllocateLike(out, input, input.dims, FLOAT32);
    ConvDecode<<<(channels + 255) / 256, 256>>>((const __nv_bfloat16 *)input.cudaData,
        (const float4 *)weight.cudaData, (float4 *)state.cudaData, (__nv_bfloat16 *)beta.cudaData,
        (float *)out.cudaData, channels, (int)beta.Count(0));
    CheckLaunch();
}

void SenseNovaBatchRecurrent(const Data &input, const Data &alpha, Data &beta,
        Data &convWeight, Data &aLog, Data &dtBias, const std::vector<Data*> &convCaches,
        const std::vector<Data*> &states, int keyHeads, int valueHeads, Data &core) {
    const int rows = states.size(), channels = input.dims.back();
    Data convolved;
    AllocateLike(convolved, input, input.dims, FLOAT32);
    AllocateLike(core, input, {1, rows, valueHeads * 128}, FLOAT32);
    for (Data *w : {&convWeight, &aLog, &dtBias}) w->ToDevice(input.dataDevice, input.dataDeviceIds);
    const std::vector<int> convShape{1, channels, 4}, stateShape{1, valueHeads, 128, 128};
    for (int first = 0; first < rows; first += ConvStateRows::capacity) {
        ConvStateRows table{};
        int count = std::min(rows - first, ConvStateRows::capacity);
        for (int j = 0; j < count; ++j) {
            const Data &conv = *convCaches[first + j], &state = *states[first + j];
            if (conv.dims != convShape || state.dims != stateShape || conv.dataType != FLOAT32 ||
                state.dataType != FLOAT32 || !conv.cudaData || !state.cudaData)
                throw std::runtime_error("Unexpected SenseNova batched decode cache layout.");
            table.state[j] = (float4*)conv.cudaData;
        }
        ConvDecode<false, true><<<dim3((channels + 255) / 256, count), 256>>>(
            (const __nv_bfloat16*)input.cudaData + size_t(first) * channels,
            (const float4*)convWeight.cudaData, nullptr,
            (__nv_bfloat16*)beta.cudaData + size_t(first) * valueHeads,
            (float*)convolved.cudaData + size_t(first) * channels, channels, valueHeads,
            1, 0, 0, table);
    }
    FastllmCudaGdnBf16DecodeRows(convolved, alpha, beta, aLog, dtBias, states, keyHeads, valueHeads, core);
    CheckLaunch();
}

void SenseNovaGateUp(const Data &packed, Data &out) {
    int width = packed.dims.back() / 2;
    AllocateLike(out, packed, {1, packed.dims[1], width});
    GateUp<<<dim3((width + 255) / 256, packed.dims[1]), 256>>>((const __nv_bfloat16 *)packed.cudaData,
                                        (__nv_bfloat16 *)out.cudaData, width);
    CheckLaunch();
}

void SenseNovaGdnOutput(const Data &core, const Data &z, Data &weight, float eps, Data &out) {
    weight.ToDevice(core.dataDevice, core.dataDeviceIds);
    if (weight.dataType != FLOAT32 || weight.Count(0) != 128) throw std::runtime_error("Unexpected GDN output scale.");
    AllocateLike(out, z, z.dims);
    GdnOutput<<<core.Count(0) / 128, 64>>>((const float *)core.cudaData,
        (const __nv_bfloat16 *)z.cudaData, (const float *)weight.cudaData, eps,
        (__nv_bfloat16 *)out.cudaData);
    CheckLaunch();
}

// Verify consecutive rows using the very same one-token convolution, GDN
// recurrence and rounding kernels as ordinary decode. Each gate writes into
// its final row and snapshots stay on the request worker's CUDA stream.
void SenseNovaVerifyRecurrent(const Data &qkv, const Data &z, const Data &alpha, Data &beta, Data &convWeight,
                              Data &aLog, Data &dtBias, Data &normWeight, const std::vector<Data *> &convCaches,
                              const std::vector<Data *> &states, int keyHeads, int valueHeads, float eps,
                              std::vector<std::pair<Data, Data>> &prefixes, Data &gated, bool mergedQkv,
                              bool mergedBA) {
    const int batch = states.size(), total = qkv.dims[1];
    if (batch == 0 || batch > ConvStateRows::capacity || convCaches.size() != states.size() || total % batch)
        throw std::runtime_error("Unexpected SenseNova verification batch shape.");
    const int rows = total / batch;
    const int channels = (2 * keyHeads + valueHeads) * 128, width = valueHeads * 128;
    const int inputPitch = qkv.dims.back(), gatePitch = beta.dims.back();
    if (rows < 2 || rows > 3)
        throw std::runtime_error("Unexpected SenseNova verification batch shape.");
    if (prefixes.size() < size_t(batch * (rows - 1)))
        prefixes.resize(batch * (rows - 1));
    AllocateLike(gated, z, {1, total, width});
    Data convolved, core;
    AllocateLike(convolved, qkv, {1, total, channels}, FLOAT32);
    AllocateLike(core, qkv, {1, total, width}, FLOAT32);
    ConvStateRows table{};
    for (int b = 0; b < batch; ++b) {
        Data &c = *convCaches[b], &s = *states[b];
        if (c.dims != std::vector<int>({1, channels, 4}) || s.dims != std::vector<int>({1, valueHeads, 128, 128}))
            throw std::runtime_error("Unexpected SenseNova MTP recurrent cache shape.");
        table.state[b] = (float4 *)c.cudaData;
        for (int t = 0; t + 1 < rows; ++t) {
            auto &prefix = prefixes[b * (rows - 1) + t];
            AllocateLike(prefix.first, c, c.dims, FLOAT32);
            AllocateLike(prefix.second, s, s.dims, FLOAT32);
            table.prefix[t][b] = (float4 *)prefix.first.cudaData;
        }
    }
    ConvDecode<true, true><<<dim3((channels + 255) / 256, batch), 256>>>(
        (const __nv_bfloat16 *)qkv.cudaData, (const float4 *)convWeight.cudaData, nullptr,
        (__nv_bfloat16 *)beta.cudaData, (float *)convolved.cudaData, channels, valueHeads, rows,
        inputPitch, gatePitch, table);
    CheckLaunch();
    if (batch > 1) {
        FastllmCudaGdnBf16VerifyRows(convolved, alpha, beta, aLog, dtBias, states, prefixes, rows, keyHeads,
                                     valueHeads, mergedBA, core);
    } else {
        // Use the original one-row recurrent kernel for every token. Recompiling
        // its FP32 update as a multi-row kernel can change a cancellation by one ULP.
        Data &state = *states.front();
        for (int t = 0; t < rows; ++t) {
            Data q, a, b, rowCore;
            auto view = [t](const Data &source, int columns, Data &row, int offset = 0) {
                row.FakeFrom(source, (size_t(t) * source.dims.back() + offset) * source.unitSize);
                row.Resize({1, 1, columns});
            };
            view(convolved, channels, q);
            view(alpha, valueHeads, a, mergedBA ? valueHeads : 0);
            view(beta, valueHeads, b);
            view(core, width, rowCore);
            if (!FastllmCudaGatedDeltaRuleSequence(q, a, b, aLog, dtBias, state, rowCore, keyHeads, valueHeads, 128,
                                                   128, 1e-6f / 128, nullptr, true))
                throw std::runtime_error("SenseNova MTP recurrence rejected its row.");
            if (t + 1 < rows)
                SenseNovaCopyState(state, prefixes[t].second);
        }
    }
    GdnOutput<<<total * valueHeads, 64>>>((const float *)core.cudaData,
                                          (const __nv_bfloat16 *)z.cudaData + (mergedQkv ? channels : 0),
                                          (const float *)normWeight.cudaData, eps, (__nv_bfloat16 *)gated.cudaData,
                                          valueHeads, mergedQkv ? inputPitch : 0);
    CheckLaunch();
}

void SenseNovaApplyRotary(Data &x, const Data &table) {
    int half = table.dims.back() / 2, heads = x.dims[2], dim = x.dims.back();
    int n = x.dims[1] * heads * half;
    ApplyRotary<<<(n + 255) / 256, 256>>>((__nv_bfloat16 *)x.cudaData,
        (const __nv_bfloat16 *)table.cudaData, heads, dim, half, n);
    CheckLaunch();
}
void SenseNovaRoute(const Data &logits, const Data &streams, int iterations, Data &branch, Data &mix) {
    int dim = streams.dims.back(), tokens = streams.Count(0) / (2 * dim);
    AllocateLike(branch, streams, {1, tokens, dim});
    AllocateLike(mix, streams, {tokens, 6});
    Route<<<tokens, 256>>>((const float *)logits.cudaData, (const __nv_bfloat16 *)streams.cudaData,
                           dim, iterations, (__nv_bfloat16 *)branch.cudaData, (__nv_bfloat16 *)mix.cudaData);
    CheckLaunch();
}
void SenseNovaMerge(const Data &branch, const Data &mix, const Data &streams, Data &output) {
    int dim = streams.dims.back(), tokens = streams.Count(0) / (2 * dim);
    AllocateLike(output, streams, streams.dims);
    Merge<<<((dim + 255) / 256) * tokens, 256>>>((const __nv_bfloat16 *)branch.cudaData,
        (const __nv_bfloat16 *)mix.cudaData, (const __nv_bfloat16 *)streams.cudaData,
        dim, (__nv_bfloat16 *)output.cudaData);
    CheckLaunch();
}
}
