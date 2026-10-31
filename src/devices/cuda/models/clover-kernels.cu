// Clover's preview equations are described by the Apache-2.0 reference:
// Diffuspace/Clover-35B-A3B-Preview, modeling_clover_moe.py (bd_gdn_*).
// Native CUDA implementation; no Transformers, Triton or SGLang runtime needed.
#include "devices/cuda/fastllm-clover.h"
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include "devices/cuda/fastllm-cuda-ordered-reduce.cuh"
#include "devices/cuda/fastllm-cuda-router.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>
#include <mma.h>
#include <type_traits>
#include <algorithm>

namespace {
template<typename T> __device__ T CloverCast(float x);
template<> __device__ __nv_bfloat16 CloverCast<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template<> __device__ __half CloverCast<__half>(float x) { return __float2half_rn(x); }
__device__ float CloverFloat(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ float CloverFloat(__half x) { return __half2float(x); }
template<typename T> using CloverPair = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
template<typename T> __device__ CloverPair<T> CloverCast2(float x, float y) {
    if constexpr (std::is_same_v<T, __half>) return __floats2half2_rn(x, y);
    else return __floats2bfloat162_rn(x, y);
}
__device__ float2 CloverFloat2(__half2 x) { return __half22float2(x); }
__device__ float2 CloverFloat2(__nv_bfloat162 x) { return __bfloat1622float2(x); }

using fastllm::cuda_launch::DependencyStart;
using fastllm::cuda_launch::Launch;

using fastllm::cuda::WarpSum;
using fastllm::cuda::WarpFirstBlockSum256;

template<typename T, bool Indirect>
__global__ void CloverPreviewConv(const T *input,
        const T *history, const T *weight,
        const T *rightWeight, const int *slots,
        const unsigned char *valid, T *output, int tokens, int channels, CloverCudaGdnCaches caches) {
    DependencyStart();
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int request = blockIdx.z, token = blockIdx.y;
    const size_t row = (size_t)request * tokens + token;
    const int slot = slots ? slots[request] : request;
    if (slot < 0 || (valid && !valid[row])) {
        output[row * channels + channel] = CloverCast<T>(0.f);
        return;
    }
    if constexpr (Indirect) history = (const T *)caches.conv[slot];
    const int cacheSlot = Indirect ? 0 : slot;
    float sum = 0.f;
    for (int tap = 0; tap < 4; ++tap) {
        const int source = token + tap - 3;
        float x = 0.f;
        if (source < 0)
            x = CloverFloat(history[((size_t)cacheSlot * channels + channel) * 3 + source + 3]);
        else if (!valid || valid[(size_t)request * tokens + source])
            x = CloverFloat(input[((size_t)request * tokens + source) * channels + channel]);
        sum += x * CloverFloat(weight[channel * 4 + tap]);
    }
    for (int distance = 1; distance < 4; ++distance) {
        const int source = token + distance;
        if (source < tokens && (!valid || valid[(size_t)request * tokens + source]))
            sum += CloverFloat(input[((size_t)request * tokens + source) * channels + channel]) *
                   CloverFloat(rightWeight[channel * 3 + distance - 1]);
    }
    output[row * channels + channel] = CloverCast<T>(sum / (1.f + expf(-sum)));
}

// A CTA evaluates several queries against the same head. Key normalization,
// value loads and state loads are shared; each query keeps the original FP32
// accumulation order and BF16 output boundary.
template<typename T, int FixedTokens, int Queries, bool Indirect>
__global__ void CloverPreviewGdn(const T *query,
        const T *key, const T *value, const float *g,
        const float *state, const float *logitScale, const float *historyBias,
        const float *historyScale, const int *slots, const unsigned char *valid,
        T *output, int runtimeTokens, int keyHeads, int valueHeads, int qkStride, int vStride, CloverCudaGdnCaches caches) {
    DependencyStart();
    constexpr int dim = 128, Capacity = FixedTokens ? FixedTokens : 32;
    const int tokens = FixedTokens ? FixedTokens : runtimeTokens;
    const int request = blockIdx.z, first = blockIdx.x * Queries, head = blockIdx.y;
    const int d = threadIdx.x, lane = d % 32, warp = d / 32;
    const int keyHead = head / (valueHeads / keyHeads);
    const int slot = slots ? slots[request] : request;
    if (slot < 0) {
#pragma unroll
        for (int qi = 0; qi < Queries; ++qi)
            output[(((size_t)request * tokens + first + qi) * valueHeads + head) * dim + d] = CloverCast<T>(0.f);
        return;
    }
    if constexpr (Indirect) state = caches.state[slot];
    const int cacheSlot = Indirect ? 0 : slot;
    __shared__ float q[Queries][dim], k[Capacity][dim], weights[Queries][Capacity];
    __shared__ float partial[Queries][4], historyWeight[Queries], histories[Queries][dim];
    float qv[Queries];
#pragma unroll
    for (int qi = 0; qi < Queries; ++qi) {
        const size_t row = (size_t)request * tokens + first + qi;
        qv[qi] = !valid || valid[row] ? CloverFloat(query[row * qkStride + keyHead * dim + d]) : 0.f;
        float sum = WarpSum(qv[qi] * qv[qi]);
        if (lane == 0) partial[qi][warp] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int qi = 0; qi < Queries; ++qi)
        q[qi][d] = qv[qi] * rsqrtf(partial[qi][0] + partial[qi][1] + partial[qi][2] + partial[qi][3] + 1.e-6f);
#pragma unroll
    for (int t = warp; t < tokens; t += 4) {
        float v[4], sum = 0.f;
        const size_t offset = ((size_t)request * tokens + t) * qkStride + keyHead * dim;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = CloverFloat(key[offset + lane + i * 32]);
            sum += v[i] * v[i];
        }
        const float inv = rsqrtf(WarpSum(sum) + 1.e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) k[t][lane + i * 32] = v[i] * inv;
    }
    __syncthreads();
    float qs[Queries][4];
#pragma unroll
    for (int qi = 0; qi < Queries; ++qi)
#pragma unroll
        for (int i = 0; i < 4; ++i) qs[qi][i] = q[qi][lane + i * 32];
#pragma unroll
    for (int t = warp; t < tokens; t += 4) {
        float dots[Queries] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float kv = k[t][lane + i * 32];
#pragma unroll
            for (int qi = 0; qi < Queries; ++qi) dots[qi] += qs[qi][i] * kv;
        }
#pragma unroll
        for (int qi = 0; qi < Queries; ++qi) {
            float dot = WarpSum(dots[qi]);
            if (lane == 0) weights[qi][t] = !valid || valid[(size_t)request * tokens + t]
                ? dot * logitScale[head] : -CUDART_INF_F;
        }
    }
    __syncthreads();
    if (d < Queries) {
        int count = 0;
#pragma unroll
        for (int t = 0; t < tokens; ++t) count += !valid || valid[(size_t)request * tokens + t];
        const float historyLogit = historyBias[head] + historyScale[head] * logf(fmaxf(count, 1) / 32.f);
        float maximum = historyLogit;
#pragma unroll
        for (int t = 0; t < tokens; ++t) maximum = fmaxf(maximum, weights[d][t]);
        const float historyExp = expf(historyLogit - maximum);
        float denominator = historyExp;
#pragma unroll
        for (int t = 0; t < tokens; ++t) {
            weights[d][t] = expf(weights[d][t] - maximum);
            denominator += weights[d][t];
        }
#pragma unroll
        for (int t = 0; t < tokens; ++t) weights[d][t] /= denominator;
        historyWeight[d] = historyExp / denominator;
    }
    __syncthreads();
#pragma unroll 4
    for (int valueIndex = warp; valueIndex < dim; valueIndex += 4) {
        const size_t stateOffset = (((size_t)cacheSlot * valueHeads + head) * dim + valueIndex) * dim;
        float sums[Queries] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float sv = state[stateOffset + lane + i * 32];
#pragma unroll
            for (int qi = 0; qi < Queries; ++qi) sums[qi] += qs[qi][i] * sv;
        }
#pragma unroll
        for (int qi = 0; qi < Queries; ++qi) {
            float sum = WarpSum(sums[qi]);
            if (lane == 0) histories[qi][valueIndex] = sum;
        }
    }
    __syncthreads();
    float locals[Queries] = {};
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
        const float v = CloverFloat(value[((size_t)request * tokens + t) * vStride + head * dim + d]);
#pragma unroll
        for (int qi = 0; qi < Queries; ++qi) locals[qi] += weights[qi][t] * v;
    }
#pragma unroll
    for (int qi = 0; qi < Queries; ++qi) {
        const size_t row = (size_t)request * tokens + first + qi;
        const float result = historyWeight[qi] * histories[qi][d] * (1.f / sqrtf(128.f)) *
            expf(g[row * valueHeads + head]) + locals[qi];
        output[(row * valueHeads + head) * dim + d] = CloverCast<T>(!valid || valid[row] ? result : 0.f);
    }
}

#include "clover-gdn-preview.cuh"

bool ValidCaches(const CloverCudaGdnCaches *caches, int batch, bool conv) {
    if (!caches) return true;
    if (batch < 1 || batch > CloverCudaGdnCaches::Capacity) return false;
    for (int b = 0; b < batch; ++b)
        if (conv ? !caches->conv[b] : !caches->state[b]) return false;
    return true;
}

bool ValidBlock(int batch, int tokens) {
    return batch > 0 && batch <= 65535 && tokens > 0 && tokens <= 32 && 32 % tokens == 0;
}

template<typename T, bool Indirect>
__global__ void CloverCausalConv(const T *input,
        const T *history, const T *weight,
        T *output, T *snapshots, int batch, int tokens, int channels, CloverCudaGdnCaches caches) {
    DependencyStart();
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int request = blockIdx.z, token = blockIdx.y;
    if constexpr (Indirect) {
        history = (const T *)caches.conv[request];
        snapshots = (T *)caches.convSnapshots[request];
    }
    const int cacheSlot = Indirect ? 0 : request, cacheBatch = Indirect ? 1 : batch;
    float sum = 0.f;
    for (int tap = 0; tap < 4; ++tap) {
        const int source = token + tap - 3;
        const T x = source < 0
            ? history[((size_t)cacheSlot * channels + channel) * 3 + source + 3]
            : input[((size_t)request * tokens + source) * channels + channel];
        // The official causal convolution rounds each product to input dtype
        // before its FP32 accumulation. Preview intentionally uses FP32 products.
        sum += CloverFloat(CloverCast<T>(CloverFloat(x) *
                                               CloverFloat(weight[channel * 4 + tap])));
    }
    output[((size_t)request * tokens + token) * channels + channel] =
        CloverCast<T>(sum / (1.f + expf(-sum)));
    if (snapshots) {
        for (int tap = 0; tap < 3; ++tap) {
            const int source = token + tap - 2;
            snapshots[(((size_t)token * cacheBatch + cacheSlot) * channels + channel) * 3 + tap] = source < 0
                ? history[((size_t)cacheSlot * channels + channel) * 3 + source + 3]
                : input[((size_t)request * tokens + source) * channels + channel];
        }
    }
}

// Reuse each input and weight across eight adjacent outputs. Each thread
// retains three BF16 samples; multiplication rounding and tap order match
// CloverCausalConv<T>. A separate commit keeps old history readable by all CTAs.
template<typename T>
__global__ void CloverTiledCausalConv(const T *input, const T *history,
        const T *weight, T *output, int tokens, int channels) {
    DependencyStart();
    constexpr int Tile = 8;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int batch = blockIdx.z, base = blockIdx.y * Tile;
    T x[4], w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = weight[c * 4 + j];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        int source = base + j - 3;
        x[j] = source < 0 ? history[((size_t)batch * channels + c) * 3 + source + 3]
                         : input[((size_t)batch * tokens + source) * channels + c];
    }
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if (base + t >= tokens) break;
        x[3] = input[((size_t)batch * tokens + base + t) * channels + c];
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sum += CloverFloat(CloverCast<T>(CloverFloat(x[j]) * CloverFloat(w[j])));
        output[((size_t)batch * tokens + base + t) * channels + c] =
            CloverCast<T>(sum / (1.f + expf(-sum)));
        x[0] = x[1]; x[1] = x[2]; x[2] = x[3];
    }
}

template<typename T, bool Indirect>
__global__ void CloverCommitConv(const T *input, T *history,
        int tokens, int channels, CloverCudaGdnCaches caches) {
    DependencyStart();
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int request = blockIdx.y;
    if constexpr (Indirect) history = (T *)caches.conv[request];
    const int cacheSlot = Indirect ? 0 : request;
    T tail[3];
    for (int tap = 0; tap < 3; ++tap) {
        const int source = tokens + tap - 3;
        tail[tap] = source < 0
            ? history[((size_t)cacheSlot * channels + channel) * 3 + source + 3]
            : input[((size_t)request * tokens + source) * channels + channel];
    }
    for (int tap = 0; tap < 3; ++tap)
        history[((size_t)cacheSlot * channels + channel) * 3 + tap] = tail[tap];
}

template<typename T>
__global__ void CloverPrepareCausalGdn(const T *query, const T *key,
        const float *g, float *qout, float *kout, float *decay,
        int tokens, int keyHeads, int valueHeads, int qkStride) {
    DependencyStart();
    int lane = threadIdx.x, head = blockIdx.x, token = blockIdx.y, request = blockIdx.z;
    size_t row = (size_t)request * tokens + token;
    size_t offset = (row * keyHeads + head) * 128;
    size_t inputOffset = row * qkStride + head * 128;
    float q[4], k[4], qs = 0.f, ks = 0.f;
    for (int j = 0; j < 4; ++j) {
        q[j] = CloverFloat(query[inputOffset + lane + j * 32]);
        k[j] = CloverFloat(key[inputOffset + lane + j * 32]);
        qs += q[j] * q[j]; ks += k[j] * k[j];
    }
    float qinv = rsqrtf(WarpSum(qs) + 1.e-6f) / sqrtf(128.f);
    float kinv = rsqrtf(WarpSum(ks) + 1.e-6f);
    for (int j = 0; j < 4; ++j) {
        qout[offset + lane + j * 32] = q[j] * qinv;
        kout[offset + lane + j * 32] = k[j] * kinv;
    }
    int ratio = valueHeads / keyHeads;
    for (int i = lane; i < ratio; i += 32)
        decay[row * valueHeads + head * ratio + i] = expf(g[row * valueHeads + head * ratio + i]);
}
template<typename T, bool Prepared, bool Indirect>
__global__ void CloverCausalGdn(const T *query, const T *key,
        const T *value, const float *g, const T *beta,
        float *state, T *output, float *snapshots,
        int batch, int tokens, int keyHeads, int valueHeads, const float *prepared, int qkStride, int vStride, CloverCudaGdnCaches caches) {
    DependencyStart();
    const int lane = threadIdx.x % 32, valueIndex = blockIdx.x * 8 + threadIdx.x / 32;
    const int head = blockIdx.y, request = blockIdx.z;
    const int keyHead = head / (valueHeads / keyHeads);
    if constexpr (Indirect) {
        state = caches.state[request];
        snapshots = caches.stateSnapshots[request];
    }
    const int cacheSlot = Indirect ? 0 : request, cacheBatch = Indirect ? 1 : batch;
    const size_t stateOffset = (((size_t)cacheSlot * valueHeads + head) * 128 + valueIndex) * 128;
    const size_t qkCount = (size_t)batch * tokens * keyHeads * 128;
    float s[4];
    for (int j = 0; j < 4; ++j) s[j] = state[stateOffset + lane + j * 32];
    for (int token = 0; token < tokens; ++token) {
        const size_t row = (size_t)request * tokens + token;
        const size_t qkOffset = (row * keyHeads + keyHead) * 128;
        const size_t inputOffset = row * qkStride + keyHead * 128;
        float q[4], k[4], qs = 0.f, ks = 0.f;
        for (int j = 0; j < 4; ++j) {
            if (Prepared) {
                q[j] = prepared[qkOffset + lane + j * 32];
                k[j] = prepared[qkCount + qkOffset + lane + j * 32];
            } else {
                q[j] = CloverFloat(query[inputOffset + lane + j * 32]);
                k[j] = CloverFloat(key[inputOffset + lane + j * 32]);
                qs += q[j] * q[j]; ks += k[j] * k[j];
            }
        }
        const float qinv = Prepared ? 1.f : rsqrtf(WarpSum(qs) + 1.e-6f) / sqrtf(128.f);
        const float kinv = Prepared ? 1.f : rsqrtf(WarpSum(ks) + 1.e-6f);
        const float decay = Prepared ? prepared[2 * qkCount + row * valueHeads + head] :
            expf(g[row * valueHeads + head]);
        float remembered = 0.f;
        for (int j = 0; j < 4; ++j) {
            if (!Prepared) { q[j] *= qinv; k[j] *= kinv; }
            s[j] *= decay;
            remembered += s[j] * k[j];
        }
        const float delta = (CloverFloat(value[row * vStride + head * 128 + valueIndex]) -
                             WarpSum(remembered)) * CloverFloat(beta[row * valueHeads + head]);
        float y = 0.f;
        for (int j = 0; j < 4; ++j) {
            s[j] += k[j] * delta;
            y += s[j] * q[j];
            if (snapshots)
                snapshots[(size_t)token * cacheBatch * valueHeads * 128 * 128 + stateOffset + lane + j * 32] = s[j];
        }
        y = WarpSum(y);
        if (lane == 0) output[(row * valueHeads + head) * 128 + valueIndex] = CloverCast<T>(y);
    }
    for (int j = 0; j < 4; ++j) state[stateOffset + lane + j * 32] = s[j];
}
// Long prefill reuses a 32-token Q/K/V tile across sixteen value rows.
// Eight lanes own each row, retaining the original four-term partial sums.
// The first two levels of the old 32-lane reduction move into registers;
// the remaining shuffle tree and the FP32 state recurrence are unchanged.
template<typename T>
__global__ void CloverPrefillGdn(const float *prepared, const T *value,
        const T *beta, float *state, T *output,
        int batch, int tokens, int keyHeads, int valueHeads, int vStride) {
    DependencyStart();
    constexpr int Group = 8, Threads = 128, Tile = 32, Values = Threads / Group, Parts = 32 / Group;
    const int lane = threadIdx.x % Group, localValue = threadIdx.x / Group;
    const int valueIndex = blockIdx.x * Values + localValue;
    const int head = blockIdx.y, request = blockIdx.z, keyHead = head / (valueHeads / keyHeads);
    const size_t stateOffset = (((size_t)request * valueHeads + head) * 128 + valueIndex) * 128;
    const size_t qkCount = (size_t)batch * tokens * keyHeads * 128;
    float s[Parts][4];
#pragma unroll
    for (int p = 0; p < Parts; ++p)
        for (int j = 0; j < 4; ++j) s[p][j] = state[stateOffset + lane + p * Group + j * 32];
    __shared__ float qtile[Tile * 128], ktile[Tile * 128], vtile[Tile * Values], decays[Tile], rates[Tile];
    for (int base = 0; base < tokens; base += Tile) {
        for (int i = threadIdx.x; i < Tile * 128; i += Threads) {
            int token = base + i / 128;
            if (token < tokens) {
                size_t pos = (((size_t)request * tokens + token) * keyHeads + keyHead) * 128 + i % 128;
                qtile[i] = prepared[pos]; ktile[i] = prepared[qkCount + pos];
            }
        }
        for (int i = threadIdx.x; i < Tile * Values; i += Threads) {
            int token = base + i / Values, d = blockIdx.x * Values + i % Values;
            if (token < tokens)
                vtile[i] = CloverFloat(value[((size_t)request * tokens + token) * vStride + head * 128 + d]);
        }
        if (threadIdx.x < Tile && base + threadIdx.x < tokens) {
            size_t pos = ((size_t)request * tokens + base + threadIdx.x) * valueHeads + head;
            decays[threadIdx.x] = prepared[2 * qkCount + pos];
            rates[threadIdx.x] = CloverFloat(beta[pos]);
        }
        __syncthreads();
        for (int t = 0; t < Tile && base + t < tokens; ++t) {
            float q[Parts][4], k[Parts][4], remembered[Parts] = {};
#pragma unroll
            for (int p = 0; p < Parts; ++p) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int pos = t * 128 + lane + p * Group + j * 32;
                    q[p][j] = qtile[pos]; k[p][j] = ktile[pos];
                    s[p][j] *= decays[t]; remembered[p] += s[p][j] * k[p][j];
                }
            }
#pragma unroll
            for (int step = Parts / 2; step; step /= 2)
                for (int p = 0; p < step; ++p) remembered[p] += remembered[p + step];
            float rem = remembered[0];
#pragma unroll
            for (int step = Group / 2; step; step /= 2)
                rem += __shfl_down_sync(0xffffffff, rem, step, Group);
            rem = __shfl_sync(0xffffffff, rem, 0, Group);
            float delta = (vtile[t * Values + localValue] - rem) * rates[t];
            float y[Parts] = {};
#pragma unroll
            for (int p = 0; p < Parts; ++p) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[p][j] += k[p][j] * delta;
                    y[p] += s[p][j] * q[p][j];
                }
            }
#pragma unroll
            for (int step = Parts / 2; step; step /= 2)
                for (int p = 0; p < step; ++p) y[p] += y[p + step];
            float result = y[0];
#pragma unroll
            for (int step = Group / 2; step; step /= 2)
                result += __shfl_down_sync(0xffffffff, result, step, Group);
            if (lane == 0)
                output[(((size_t)request * tokens + base + t) * valueHeads + head) * 128 + valueIndex] =
                    CloverCast<T>(result);
        }
        __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < Parts; ++p)
        for (int j = 0; j < 4; ++j) state[stateOffset + lane + p * Group + j * 32] = s[p][j];
}
cudaStream_t Stream(void *stream) {
    return stream ? static_cast<cudaStream_t>(stream) : cudaStreamPerThread;
}
} // namespace

template<typename T>
bool FastllmCudaCloverPreviewConvImpl(const void *input, const void *history, const void *weight,
                                  const void *rightWeight, const int *slots, const unsigned char *valid,
                                  void *output, int batch, int tokens, int channels, void *stream, const CloverCudaGdnCaches *caches) {
    if (!input || (!history && !caches) || !weight || !rightWeight || !output || !ValidBlock(batch, tokens) ||
        channels <= 0 || !ValidCaches(caches, batch, true))
        return false;
    auto launch = [&](auto indirect) {
        return Launch<CloverPreviewConv<T, decltype(indirect)::value>>(
            dim3((channels + 255) / 256, tokens, batch), 256, 0, Stream(stream),
            static_cast<const T *>(input), static_cast<const T *>(history),
            static_cast<const T *>(weight), static_cast<const T *>(rightWeight), slots,
            valid, static_cast<T *>(output), tokens, channels, caches ? *caches : CloverCudaGdnCaches{});
    };
    return caches ? launch(std::true_type{}) : launch(std::false_type{});
}

template<typename T>
bool FastllmCudaCloverPreviewGdnImpl(const void *query, const void *key, const void *value, const float *g,
                                 const float *state, const float *logitScale, const float *historyBias,
                                 const float *historyScale, const int *slots, const unsigned char *valid,
                                 void *output, int batch, int tokens, int keyHeads, int valueHeads,
                                 void *stream, int qkvRowStride, const CloverCudaGdnCaches *caches) {
    if (!query || !key || !value || !g || (!state && !caches) || !logitScale || !historyBias || !historyScale || !output ||
        !ValidBlock(batch, tokens) || !ValidCaches(caches, batch, false) || keyHeads <= 0 || valueHeads <= 0 || valueHeads > 65535 ||
        valueHeads % keyHeads || qkvRowStride < 0 || (qkvRowStride && qkvRowStride < valueHeads * 128))
        return false;
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    if ((tokens == 16 || tokens == 32) && CloverPreviewMmaAvailable()) {
        auto launch = [&](auto width, auto indirect) {
            return Launch<CloverPreviewGdnMma<decltype(width)::value, decltype(indirect)::value>>(
                dim3(tokens / 16, valueHeads, batch), 256, 0, Stream(stream),
                (const T *)query, (const T *)key, (const T *)value,
                g, state, logitScale, historyBias, historyScale, slots, valid, (T *)output,
                keyHeads, valueHeads, qkvRowStride ? qkvRowStride : keyHeads * 128,
                qkvRowStride ? qkvRowStride : valueHeads * 128, caches ? *caches : CloverCudaGdnCaches{});
        };
        if (tokens == 16) return caches ? launch(std::integral_constant<int, 16>{}, std::true_type{})
                                        : launch(std::integral_constant<int, 16>{}, std::false_type{});
        return caches ? launch(std::integral_constant<int, 32>{}, std::true_type{})
                      : launch(std::integral_constant<int, 32>{}, std::false_type{});
    }
    auto dispatch = [&](auto indirect) {
        auto launch = [&](auto width, auto count) {
            constexpr int queries = decltype(count)::value;
            return Launch<CloverPreviewGdn<T, decltype(width)::value, queries, decltype(indirect)::value>>(
                dim3(tokens / queries, valueHeads, batch), 128, 0, Stream(stream),
                (const T *)query, (const T *)key, (const T *)value,
                g, state, logitScale, historyBias, historyScale, slots, valid, (T *)output,
                tokens, keyHeads, valueHeads, qkvRowStride ? qkvRowStride : keyHeads * 128,
                qkvRowStride ? qkvRowStride : valueHeads * 128, caches ? *caches : CloverCudaGdnCaches{});
        };
        auto grouped = [&](auto width) {
            // Reusing a small, hot state helps microbenchmarks but reduces CTA
            // parallelism with cold model state. Keep one query per CTA for the
            // common single-request 16-token block; wider blocks retain reuse.
            if (batch == 1) {
                if constexpr (decltype(width)::value == 16)
                    return launch(width, std::integral_constant<int, 1>{});
                else return launch(width, std::integral_constant<int, 2>{});
            }
            return launch(width, std::integral_constant<int, 4>{});
        };
        if (tokens == 16) return grouped(std::integral_constant<int, 16>{});
        if (tokens == 32) return grouped(std::integral_constant<int, 32>{});
        return launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
    };
    return caches ? dispatch(std::true_type{}) : dispatch(std::false_type{});
}

template<typename T>
bool FastllmCudaCloverCausalConvImpl(const void *input, void *history, const void *weight, void *output,
                                 void *snapshots, int batch, int tokens, int channels, void *stream, const CloverCudaGdnCaches *caches) {
    if (!input || (!history && !caches) || !weight || !output || batch <= 0 || batch > 65535 || tokens <= 0 ||
        tokens > 65535 || channels <= 0 || !ValidCaches(caches, batch, true))
        return false;
    auto dispatch = [&](auto indirect) {
        if (!caches && !snapshots && tokens >= 128 && channels >= 2048) {
            if (!Launch<CloverTiledCausalConv<T>>(dim3((channels + 255) / 256, (tokens + 7) / 8, batch), 256,
                                                     0, Stream(stream), static_cast<const T *>(input),
                                                     static_cast<const T *>(history),
                                                     static_cast<const T *>(weight),
                                                     static_cast<T *>(output), tokens, channels))
                return false;
        } else {
            if (!Launch<CloverCausalConv<T, decltype(indirect)::value>>(
                    dim3((channels + 255) / 256, tokens, batch), 256, 0, Stream(stream),
                    static_cast<const T *>(input), static_cast<const T *>(history),
                    static_cast<const T *>(weight), static_cast<T *>(output),
                    static_cast<T *>(snapshots), batch, tokens, channels, caches ? *caches : CloverCudaGdnCaches{}))
                return false;
        }
        return Launch<CloverCommitConv<T, decltype(indirect)::value>>(dim3((channels + 255) / 256, batch), 256, 0, Stream(stream),
                                              static_cast<const T *>(input),
                                              static_cast<T *>(history), tokens, channels, caches ? *caches : CloverCudaGdnCaches{});
    };
    return caches ? dispatch(std::true_type{}) : dispatch(std::false_type{});
}

#include "clover-gdn-prefill.cuh"

size_t FastllmCudaCloverPrefillGdnWorkspaceBytes(int batch, int tokens,
        int keyHeads, int valueHeads) {
    return clover_prefill::Bytes(batch, tokens, keyHeads, valueHeads);
}

bool FastllmCudaCloverPrefillGdn(const void *query, const void *key, const void *value,
        const float *g, const void *beta, float *state, void *output,
        int batch, int tokens, int keyHeads, int valueHeads,
        void *workspace, size_t workspaceBytes, void *stream, int qkvRowStride) {
    return clover_prefill::Run(query, key, value, g, beta, state, output,
        batch, tokens, keyHeads, valueHeads, qkvRowStride, workspace, workspaceBytes, stream);
}

template<typename T>
bool FastllmCudaCloverCausalGdnImpl(const void *query, const void *key, const void *value, const float *g,
                                const void *beta, float *state, void *output, float *snapshots, int batch,
                                int tokens, int keyHeads, int valueHeads, void *stream, float *workspace,
                                int qkvRowStride, const CloverCudaGdnCaches *caches) {
    if (!query || !key || !value || !g || !beta || (!state && !caches) || !output || batch <= 0 || batch > 65535 || !ValidCaches(caches, batch, false) ||
        tokens <= 0 || keyHeads <= 0 || valueHeads <= 0 || valueHeads > 65535 || valueHeads % keyHeads ||
        qkvRowStride < 0 || (qkvRowStride && qkvRowStride < valueHeads * 128))
        return false;
    const int qkStride = qkvRowStride ? qkvRowStride : keyHeads * 128;
    const int vStride = qkvRowStride ? qkvRowStride : valueHeads * 128;
    auto dispatch = [&](auto indirect) {
        if (workspace && tokens >= 16 && tokens <= 65535) {
            size_t qk = (size_t)batch * tokens * keyHeads * 128;
            if (!Launch<CloverPrepareCausalGdn<T>>(dim3(keyHeads, tokens, batch), 32, 0, Stream(stream),
                                                      (const T *)query, (const T *)key, g,
                                                      workspace, workspace + qk, workspace + 2 * qk, tokens,
                                                      keyHeads, valueHeads, qkStride))
                return false;
            if (!caches && !snapshots && tokens >= 128 && valueHeads >= 16) {
                if (!Launch<CloverPrefillGdn<T>>(dim3(8, valueHeads, batch), 128, 0, Stream(stream), workspace,
                                                    (const T *)value, (const T *)beta,
                                                    state, (T *)output, batch, tokens, keyHeads,
                                                    valueHeads, vStride))
                    return false;
            } else {
                if (!Launch<CloverCausalGdn<T, true, decltype(indirect)::value>>(
                        dim3(16, valueHeads, batch), 256, 0, Stream(stream), (const T *)query,
                        (const T *)key, (const T *)value, g, (const T *)beta,
                        state, (T *)output, snapshots, batch, tokens, keyHeads, valueHeads, workspace,
                        qkStride, vStride, caches ? *caches : CloverCudaGdnCaches{}))
                    return false;
            }
        } else {
            if (!Launch<CloverCausalGdn<T, false, decltype(indirect)::value>>(
                    dim3(16, valueHeads, batch), 256, 0, Stream(stream),
                    static_cast<const T *>(query), static_cast<const T *>(key),
                    static_cast<const T *>(value), g, static_cast<const T *>(beta), state,
                    static_cast<T *>(output), snapshots, batch, tokens, keyHeads, valueHeads, nullptr,
                    qkStride, vStride, caches ? *caches : CloverCudaGdnCaches{}))
                return false;
        }
        return true;
    };
    return caches ? dispatch(std::true_type{}) : dispatch(std::false_type{});
}

namespace {
template<typename T>
__global__ void CloverRotary(T *input, int heads, int dim,
        int rotaryDim, int position, float theta) {
    DependencyStart();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * rotaryDim / 2) return;
    int h = i / (rotaryDim / 2), d = i % (rotaryDim / 2);
    float phase = (position + blockIdx.y) * (1.f / powf(theta, 2.f * d / rotaryDim));
    float c = CloverFloat(CloverCast<T>(cosf(phase)));
    float s = CloverFloat(CloverCast<T>(sinf(phase)));
    size_t p = ((size_t)blockIdx.y * heads + h) * dim + d;
    float x = CloverFloat(input[p]), y = CloverFloat(input[p + rotaryDim / 2]);
    // Eager BF16 rotary rounds each multiply before adding/subtracting.
    float xc = CloverFloat(CloverCast<T>(x * c));
    float ys = CloverFloat(CloverCast<T>(y * s));
    float yc = CloverFloat(CloverCast<T>(y * c));
    float xs = CloverFloat(CloverCast<T>(x * s));
    input[p] = CloverCast<T>(xc - ys);
    input[p + rotaryDim / 2] = CloverCast<T>(yc + xs);
}

// Decode Q/gate split, Q/K RMSNorm and partial RoPE share one CTA per head.
// Retain the 256-thread norm tree and every BF16 rounding boundary.
template<typename T>
__global__ void CloverPrepareAttention(const T *qgate, const T *key,
        const float *qWeight, const float *kWeight, T *query,
        T *normalizedKey, T *gate, int queryHeads, int keyHeads,
        int rotaryDim, int position, float theta, float eps, int qkvStride, T *value) {
    DependencyStart();
    constexpr int dim = 256;
    const int d = threadIdx.x, token = blockIdx.y;
    const bool isQuery = blockIdx.x < queryHeads;
    const int head = isQuery ? blockIdx.x : blockIdx.x - queryHeads;
    const size_t row = (size_t)token * (isQuery ? queryHeads : keyHeads) + head;
    const auto *source = isQuery
        ? qgate + (size_t)token * (qkvStride ? qkvStride : queryHeads * dim * 2) + head * dim * 2
        : key + (size_t)token * (qkvStride ? qkvStride : keyHeads * dim) + head * dim;
    if (!isQuery && value) value[row * dim + d] = source[keyHeads * dim + d];
    const auto *weight = isQuery ? qWeight : kWeight;
    auto *destination = (isQuery ? query : normalizedKey) + row * dim;
    __shared__ float scratch[8];
    __shared__ T normalized[dim];
    const float x = CloverFloat(source[d]);
    const float inv = rsqrtf(WarpFirstBlockSum256(x * x, scratch) / dim + eps);
    normalized[d] = CloverCast<T>(x * inv * (weight[d] + 1.f));
    if (isQuery) gate[row * dim + d] = source[dim + d];
    __syncthreads();
    if (d >= rotaryDim) destination[d] = normalized[d];
    if (d < rotaryDim / 2) {
        float phase = (position + token) * (1.f / powf(theta, 2.f * d / rotaryDim));
        float c = CloverFloat(CloverCast<T>(cosf(phase)));
        float s = CloverFloat(CloverCast<T>(sinf(phase)));
        float a = CloverFloat(normalized[d]), b = CloverFloat(normalized[d + rotaryDim / 2]);
        float ac = CloverFloat(CloverCast<T>(a * c));
        float bs = CloverFloat(CloverCast<T>(b * s));
        float bc = CloverFloat(CloverCast<T>(b * c));
        float as = CloverFloat(CloverCast<T>(a * s));
        destination[d] = CloverCast<T>(ac - bs);
        destination[d + rotaryDim / 2] = CloverCast<T>(bc + as);
    }
}

// Stable 256-way bitonic sorting, using shuffle within a warp and shared
// exchange only for the three cross-warp stages. No external kernel library.
__device__ void CloverSort256(float &value, int &id, float *sharedValue, int *sharedId, int count) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int width = 2; width <= 256; width *= 2) {
        if (width / 2 >= count) break;
#pragma unroll
        for (int stride = width / 2; stride > 0; stride /= 2) {
            float other; int otherId;
            if (stride < 32) {
                other = __shfl_xor_sync(0xffffffff, value, stride);
                otherId = __shfl_xor_sync(0xffffffff, id, stride);
            } else {
                sharedValue[tid] = value; sharedId[tid] = id;
                __syncthreads();
                other = sharedValue[tid ^ stride]; otherId = sharedId[tid ^ stride];
                __syncthreads();
            }
            bool otherFirst = other > value || (other == value && otherId < id);
            bool keepFirst = ((tid & stride) == 0) == ((tid & width) == 0);
            if (otherFirst == keepFirst) { value = other; id = otherId; }
        }
    }
}
__global__ void CloverCoreset(const int *ids, const float *scores,
        unsigned char *allowed, int tokens, int experts, int topk, float topP) {
    __shared__ float values[256], reduction[8], routeScores[512];
    __shared__ int sortedIds[256], routes[512];
    __shared__ double totals[8];
    int e = threadIdx.x, lane = e % 32, warp = e / 32;
    const size_t offset = (size_t)blockIdx.x * tokens * topk;
    for (int r = e; r < tokens * topk; r += 256) {
        routes[r] = ids[offset + r]; routeScores[r] = scores[offset + r];
    }
    __syncthreads();
    float value = 0.f;
    if (e < experts) {
        // Keep token/route accumulation order before sorting the expert masses.
        for (int r = 0; r < tokens * topk; ++r)
            if (routes[r] == e) value += routeScores[r];
    }
    float total = WarpFirstBlockSum256(value, reduction);
    value = e < experts ? value / fmaxf(total, 1.e-9f) : -CUDART_INF_F;
    int id = e;
    CloverSort256(value, id, values, sortedIds, experts);
    double mass = e < experts ? value : 0., cumulative = mass;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        double v = __shfl_up_sync(0xffffffff, cumulative, d);
        if (lane >= d) cumulative += v;
    }
    if (lane == 31) totals[warp] = cumulative;
    __syncthreads();
    for (int w = 0; w < warp; ++w) cumulative += totals[w];
    if (id < experts) {
        float before = (float)(cumulative - mass);
        allowed[(size_t)blockIdx.x * experts + id] = e == 0 || before < topP;
    }
}
template<typename T>
__device__ void CloverExpertGateRow(const T *input, const T *weight,
        const int *indices, T *middle, int hidden, int intermediate,
        int experts, int topk, int task, int d) {
    int lane = threadIdx.x % 32;
    if (d >= intermediate) return;
    int expert = indices[task];
    if (expert < 0 || expert >= experts) return;
    const T *x = input + (size_t)(task / topk) * hidden;
    const T *w = weight + ((size_t)expert * intermediate * 2 + d) * hidden;
    float gate = 0.f, up = 0.f;
    if (hidden % 8 == 0 && ((reinterpret_cast<uintptr_t>(x) |
            reinterpret_cast<uintptr_t>(w)) & 15) == 0) {
        float ga[4] = {}, ua[4] = {};
        for (int k = lane * 8; k < hidden; k += 256) {
            uint4 xv = *(const uint4 *)(x + k);
            uint4 gv = *(const uint4 *)(w + k);
            uint4 uv = *(const uint4 *)(w + (size_t)intermediate * hidden + k);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 a = CloverFloat2(((const CloverPair<T> *)&xv)[j]);
                float2 g = CloverFloat2(((const CloverPair<T> *)&gv)[j]);
                float2 u = CloverFloat2(((const CloverPair<T> *)&uv)[j]);
                ga[j] += a.x * g.x; ga[j] += a.y * g.y;
                ua[j] += a.x * u.x; ua[j] += a.y * u.y;
            }
        }
        gate = (ga[0] + ga[1]) + (ga[2] + ga[3]);
        up = (ua[0] + ua[1]) + (ua[2] + ua[3]);
    } else {
        for (int k = lane; k < hidden; k += 32) {
            float a = CloverFloat(x[k]);
            gate += a * CloverFloat(w[k]);
            up += a * CloverFloat(w[(size_t)intermediate * hidden + k]);
        }
    }
    gate = WarpSum(gate); up = WarpSum(up);
    if (lane == 0) {
        gate = CloverFloat(CloverCast<T>(gate));
        up = CloverFloat(CloverCast<T>(up));
        middle[(size_t)task * intermediate + d] = CloverCast<T>(gate / (1.f + expf(-gate)) * up);
    }
}
template<typename T>
__device__ void CloverExpertDownRow(const T *middle, const T *weight,
        const int *indices, const float *scores, T *parts,
        int hidden, int intermediate, int experts, int task, int d) {
    int lane = threadIdx.x % 32;
    if (d >= hidden) return;
    int expert = indices[task];
    if (expert < 0 || expert >= experts) return;
    const T *x = middle + (size_t)task * intermediate;
    const T *w = weight + ((size_t)expert * hidden + d) * intermediate;
    float sum = 0.f;
    if (intermediate % 8 == 0 && ((reinterpret_cast<uintptr_t>(x) |
            reinterpret_cast<uintptr_t>(w)) & 15) == 0) {
        float partial[4] = {};
        for (int k = lane * 8; k < intermediate; k += 256) {
            uint4 xv = *(const uint4 *)(x + k), wv = *(const uint4 *)(w + k);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 a = CloverFloat2(((const CloverPair<T> *)&xv)[j]);
                float2 b = CloverFloat2(((const CloverPair<T> *)&wv)[j]);
                partial[j] += a.x * b.x; partial[j] += a.y * b.y;
            }
        }
        sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    } else {
        for (int k = lane; k < intermediate; k += 32)
            sum += CloverFloat(x[k]) * CloverFloat(w[k]);
    }
    sum = WarpSum(sum);
    if (lane == 0) parts[(size_t)task * hidden + d] = CloverCast<T>(sum * scores[task]);
}
template<typename T>
__global__ void CloverExpertGate(const T *input, const T *weight,
        const int *indices, T *middle, int hidden, int intermediate,
        int experts, int topk) {
    CloverExpertGateRow<T>(input, weight, indices, middle, hidden, intermediate, experts,
        topk, blockIdx.y, blockIdx.x * 8 + threadIdx.x / 32);
}
template<typename T>
__global__ void CloverExpertDown(const T *middle, const T *weight,
        const int *indices, const float *scores, T *parts,
        int hidden, int intermediate, int experts) {
    CloverExpertDownRow<T>(middle, weight, indices, scores, parts, hidden, intermediate,
        experts, blockIdx.y, blockIdx.x * 8 + threadIdx.x / 32);
}
// Group routes in their original token order. An expert can occur at most
// once per token. Each expert therefore needs only `rows` task slots.
__global__ void CloverExpertRows(const int *indices, int *workspace, int rows, int topk) {
    const int expert = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ int offsets[4], base;
    if (tid == 0) base = 0;
    __syncthreads();
    for (int start = 0; start < rows * topk; start += 128) {
        int task = start + tid;
        bool match = task < rows * topk && indices[task] == expert;
        unsigned mask = __ballot_sync(0xffffffff, match);
        int rank = __popc(mask & ((1u << lane) - 1));
        if (lane == 0) offsets[warp] = __popc(mask);
        __syncthreads();
        int before = 0;
        for (int w = 0; w < warp; ++w) before += offsets[w];
        if (match) workspace[(size_t)expert * (rows + 1) + base + before + rank] = task;
        __syncthreads();
        if (tid == 0) base += offsets[0] + offsets[1] + offsets[2] + offsets[3];
        __syncthreads();
    }
    if (tid == 0) workspace[(size_t)expert * (rows + 1) + rows] = base;
}

__global__ void CloverClearExpertCounts(int *workspace, int rows, int experts) {
    int expert = threadIdx.x;
    if (expert < experts) workspace[(size_t)expert * (rows + 1) + rows] = 0;
}
__global__ void CloverScatterExpertRows(const int *indices, int *workspace, int rows, int topk) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= rows * topk) return;
    int expert = indices[task];
    int offset = atomicAdd(workspace + (size_t)expert * (rows + 1) + rows, 1);
    workspace[(size_t)expert * (rows + 1) + offset] = task;
}
void CloverBuildExpertRows(const int *indices, int *workspace, int rows, int topk,
        int experts, cudaStream_t stream) {
    if (rows >= 1024 && experts >= 32) {
        // Bucket independent token rows without rescanning every route for
        // every expert. Row order may vary; GEMMs scatter back by task id and
        // the final sum still follows ascending expert id.
        CloverClearExpertCounts<<<1, 256, 0, stream>>>(workspace, rows, experts);
        CloverScatterExpertRows<<<(rows * topk + 255) / 256, 256, 0, stream>>>(
            indices, workspace, rows, topk);
    } else CloverExpertRows<<<experts, 128, 0, stream>>>(indices, workspace, rows, topk);
}
template<int TileRows = 16>
__global__ void CloverExpertTiles(int *workspace, int rows, int experts) {
    int e = threadIdx.x, lane = e % 32, warp = e / 32;
    int rowsForExpert = e < experts ? workspace[(size_t)e * (rows + 1) + rows] : 0;
    int count = (rowsForExpert + TileRows - 1) / TileRows, inclusive = count;
    for (int d = 1; d < 32; d *= 2) {
        int v = __shfl_up_sync(0xffffffff, inclusive, d);
        if (lane >= d) inclusive += v;
    }
    __shared__ int totals[8];
    if (lane == 31) totals[warp] = inclusive;
    __syncthreads();
    int before = inclusive - count, total = 0;
    for (int w = 0; w < 8; ++w) {
        total += totals[w];
        if (w < warp) before += totals[w];
    }
    int *tiles = workspace + (size_t)experts * (rows + 1);
    if (e == 0) tiles[0] = total;
    for (int m = 0; m < count; ++m) tiles[1 + before + m] = e * (rows + 1) + m * TileRows;
}

// Small decode batches build stable expert rows and tile descriptors in one
// CTA. Warp peer masks give each route its rank; preceding warp/chunk counts
// preserve token order without atomically scattering the output list.
__global__ void CloverExpertRowsAndTiles(const int *indices, int *workspace,
        int rows, int topk, int experts) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ int counts[8][256], totals[256], warpTotals[8];
    totals[tid] = 0;
    for (int start = 0; start < rows * topk; start += 256) {
#pragma unroll
        for (int w = 0; w < 8; ++w) counts[w][tid] = 0;
        __syncthreads();
        const int task = start + tid, expert = task < rows * topk ? indices[task] : -1;
        unsigned peers;
#if __CUDA_ARCH__ >= 700
        peers = __match_any_sync(0xffffffff, expert);
#else
        peers = 0;
        for (int i = 0; i < 32; ++i)
            if (__shfl_sync(0xffffffff, expert, i) == expert) peers |= 1u << i;
#endif
        if (expert >= 0 && lane == __ffs(peers) - 1) counts[warp][expert] = __popc(peers);
        __syncthreads();
        if (expert >= 0) {
            int rank = totals[expert] + __popc(peers & ((1u << lane) - 1));
            for (int w = 0; w < warp; ++w) rank += counts[w][expert];
            workspace[(size_t)expert * (rows + 1) + rank] = task;
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < 8; ++w) totals[tid] += counts[w][tid];
        __syncthreads();
    }
    const int count = (totals[tid] + 15) / 16;
    if (tid < experts) workspace[(size_t)tid * (rows + 1) + rows] = totals[tid];
    int inclusive = count;
    for (int d = 1; d < 32; d *= 2) {
        int v = __shfl_up_sync(0xffffffff, inclusive, d);
        if (lane >= d) inclusive += v;
    }
    if (lane == 31) warpTotals[warp] = inclusive;
    __syncthreads();
    int before = inclusive - count, total = 0;
    for (int w = 0; w < 8; ++w) {
        total += warpTotals[w];
        if (w < warp) before += warpTotals[w];
    }
    int *tiles = workspace + (size_t)experts * (rows + 1);
    if (tid == 0) tiles[0] = total;
    for (int m = 0; m < count; ++m) tiles[1 + before + m] = tid * (rows + 1) + m * 16;
}

void CloverBuildExpertTiles(const int *indices, int *workspace, int rows, int topk,
                           int experts, cudaStream_t stream) {
    if (rows * topk <= 512)
        CloverExpertRowsAndTiles<<<1, 256, 0, stream>>>(indices, workspace, rows, topk, experts);
    else {
        CloverBuildExpertRows(indices, workspace, rows, topk, experts, stream);
        CloverExpertTiles<16><<<1, 256, 0, stream>>>(workspace, rows, experts);
    }
}

__device__ __forceinline__ void CloverCopyAsync16(void *destination, const void *source, bool valid) {
#if __CUDA_ARCH__ >= 800
    unsigned address = (unsigned)__cvta_generic_to_shared(destination);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"(address), "l"(source), "r"(valid ? 16 : 0));
#else
    *(uint4 *)destination = valid ? *(const uint4 *)source : make_uint4(0, 0, 0, 0);
#endif
}

// Up to 16 token rows share a weight tile, including experts with only one
// active row. Compact descriptors avoid empty expert launches and separate
// sparse GEMVs that reread weights for each token. Tensor Core products
// accumulate in FP32; gate/up and routed-down boundaries round to storage
// precision. WMMA requires SM70+ for FP16 and SM80+ for BF16.
template<typename T, bool Gate, int N, int K>
__global__ void CloverExpertGemm(const T *input, const T *weight,
        const float *scores, const int *workspace, T *output,
        int rows, int hidden, int intermediate, int experts, int topk) {
#if __CUDA_ARCH__ >= 700
    if constexpr (std::is_same_v<T, __half> || __CUDA_ARCH__ >= 800) {
    using namespace nvcuda;
    constexpr int M = 16, LD = K + 8;
    const int *tiles = workspace + (size_t)experts * (rows + 1);
    if (blockIdx.y >= tiles[0]) return;
    const int index = tiles[1 + blockIdx.y], expert = index / (rows + 1);
    const int start = index % (rows + 1), count = workspace[expert * (rows + 1) + rows];
    const int inputWidth = Gate ? hidden : intermediate;
    const int outputWidth = Gate ? intermediate : hidden;
    const int column = blockIdx.x * N, tid = threadIdx.x, warp = tid / 32;
    __shared__ __align__(32) T a[2 * M * LD], b[2 * N * LD], u[Gate ? 2 * N * LD : 1];
    __shared__ int tasks[M], inputRows[M];
    if (tid < M) {
        int task = start + tid < count ? workspace[index + tid] : -1;
        tasks[tid] = task;
        inputRows[tid] = Gate && task >= 0 ? task / topk : task;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc, up;
    wmma::fill_fragment(acc, 0.f);
    if (Gate) wmma::fill_fragment(up, 0.f);
    auto load = [&](int k, int stage) {
        auto *pa = a + stage * M * LD;
        auto *pb = b + stage * N * LD;
        auto *pu = Gate ? u + stage * N * LD : nullptr;
        if (inputWidth % 8 == 0 && ((reinterpret_cast<uintptr_t>(input) |
            reinterpret_cast<uintptr_t>(weight)) & 15) == 0) {
            for (int i = tid * 8; i < M * K; i += N * 2 * 8) {
                int row = inputRows[i / K], d = k + i % K;
                bool valid = row >= 0 && d < inputWidth;
                CloverCopyAsync16(pa + (i / K) * LD + i % K,
                    input + (valid ? (size_t)row * inputWidth + d : 0), valid);
            }
            for (int i = tid * 8; i < N * K; i += N * 2 * 8) {
                int n = column + i / K, d = k + i % K;
                size_t offset = ((size_t)expert * outputWidth * (Gate ? 2 : 1) + n) * inputWidth + d;
                bool valid = n < outputWidth && d < inputWidth;
                CloverCopyAsync16(pb + (i / K) * LD + i % K, weight + (valid ? offset : 0), valid);
                if (Gate) CloverCopyAsync16(pu + (i / K) * LD + i % K,
                    weight + (valid ? offset + (size_t)outputWidth * inputWidth : 0), valid);
            }
        } else {
            for (int i = tid; i < M * K; i += N * 2) {
                int row = inputRows[i / K], d = k + i % K;
                pa[(i / K) * LD + i % K] = row >= 0 && d < inputWidth
                    ? input[(size_t)row * inputWidth + d] : CloverCast<T>(0.f);
            }
            for (int i = tid; i < N * K; i += N * 2) {
                int n = column + i / K, d = k + i % K;
                size_t offset = ((size_t)expert * outputWidth * (Gate ? 2 : 1) + n) * inputWidth + d;
                pb[(i / K) * LD + i % K] = n < outputWidth && d < inputWidth ? weight[offset] : CloverCast<T>(0.f);
                if (Gate) pu[(i / K) * LD + i % K] = n < outputWidth && d < inputWidth
                    ? weight[offset + (size_t)outputWidth * inputWidth] : CloverCast<T>(0.f);
            }
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
#endif
    };
    load(0, 0);
    for (int k = 0; k < inputWidth; k += K) {
        int stage = (k / K) % 2;
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;");
#endif
        __syncthreads();
        if (k + K < inputWidth) load(k + K, 1 - stage);
        auto *pa = a + stage * M * LD;
        auto *pb = b + stage * N * LD;
        auto *pu = Gate ? u + stage * N * LD : nullptr;
        for (int j = 0; j < K; j += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, pa + j, LD);
            wmma::load_matrix_sync(fb, pb + warp * 16 * LD + j, LD);
            wmma::mma_sync(acc, fa, fb, acc);
            if (Gate) {
                wmma::load_matrix_sync(fb, pu + warp * 16 * LD + j, LD);
                wmma::mma_sync(up, fa, fb, up);
            }
        }
        __syncthreads();
    }
    // Staging is finished; reuse it for the two FP32 epilogue tiles.
    float *result = (float *)b, *upResult = Gate ? (float *)u : nullptr;
    wmma::store_matrix_sync(result + warp * 16, acc, N, wmma::mem_row_major);
    if (Gate) wmma::store_matrix_sync(upResult + warp * 16, up, N, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < M * N; i += N * 2) {
        int task = tasks[i / N], n = column + i % N;
        if (task < 0 || n >= outputWidth) continue;
        float v = result[i];
        if (Gate) {
            v = CloverFloat(CloverCast<T>(v));
            float uv = CloverFloat(CloverCast<T>(upResult[i]));
            v = v / (1.f + expf(-v)) * uv;
        } else v *= scores[task];
        output[(size_t)task * outputWidth + n] = CloverCast<T>(v);
    }
    }
#endif
}

template<typename T>
bool CloverTensorCores() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local int previous = -1, capability = 0;
    if (device != previous) {
        cudaFuncAttributes attributes;
        if (cudaFuncGetAttributes(&attributes, CloverExpertGemm<T, true, 32, 128>) != cudaSuccess) return false;
        capability = std::min(attributes.binaryVersion, attributes.ptxVersion);
        previous = device;
    }
    return capability >= (std::is_same_v<T, __half> ? 70 : 80);
}
template<typename T, bool Shared>
__global__ void CloverExpertSum(const T *parts, const int *indices, T *output,
                                int hidden, int topk, const T *shared,
                                const T *sharedGate, int sharedGateStride) {
    int d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    __shared__ int order[16];
    // Sort the routes once per CTA, preserving the reference accumulation order.
    if (threadIdx.x < topk) {
        int k = threadIdx.x, rank = 0;
        for (int j = 0; j < topk; ++j)
            rank += indices[row * topk + j] < indices[row * topk + k];
        order[rank] = k;
    }
    __syncthreads();
    if (d >= hidden) return;
    float sum = 0.f;
    for (int rank = 0; rank < topk; ++rank)
        sum += CloverFloat(parts[((size_t)row * topk + order[rank]) * hidden + d]);
    auto result = CloverCast<T>(sum);
    if constexpr (Shared) {
        // Preserve sigmoid, shared product, routed sum and final add rounding.
        auto probability = CloverCast<T>(1.f / (1.f + expf(-CloverFloat(sharedGate[(size_t)row * sharedGateStride]))));
        auto contribution = CloverCast<T>(CloverFloat(shared[(size_t)row * hidden + d]) *
                                             CloverFloat(probability));
        result = CloverCast<T>(CloverFloat(result) + CloverFloat(contribution));
    }
    output[(size_t)row * hidden + d] = result;
}

template<typename T, bool Shared>
__global__ void CloverExpertSumVector(const T *parts, const int *indices, T *output,
                                      int rows, int hidden, int topk, const T *shared,
                                      const T *sharedGate, int sharedGateStride) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    __shared__ int order[4][16];
    if (lane < topk) {
        int expert = indices[row * topk + lane], rank = 0;
        for (int j = 0; j < topk; ++j) rank += indices[row * topk + j] < expert;
        order[warp][rank] = lane;
    }
    __syncwarp();
    float probability = 0.f;
    if constexpr (Shared)
        probability =
            CloverFloat(CloverCast<T>(1.f / (1.f + expf(-CloverFloat(sharedGate[(size_t)row * sharedGateStride])))));
    for (int d = lane * 8; d < hidden; d += 32 * 8) {
        float sums[8] = {};
        for (int k = 0; k < topk; ++k) {
            uint4 value = *reinterpret_cast<const uint4 *>(
                parts + ((size_t)row * topk + order[warp][k]) * hidden + d);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sums[j] += CloverFloat(reinterpret_cast<T *>(&value)[j]);
        }
        uint4 value;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            auto result = CloverCast<T>(sums[j]);
            if constexpr (Shared) {
                auto contribution =
                    CloverCast<T>(CloverFloat(shared[(size_t)row * hidden + d + j]) * probability);
                result = CloverCast<T>(CloverFloat(result) + CloverFloat(contribution));
            }
            reinterpret_cast<T *>(&value)[j] = result;
        }
        *reinterpret_cast<uint4 *>(output + (size_t)row * hidden + d) = value;
    }
}
template<typename T, bool Shared>
void CloverSumExperts(const T *parts, const int *indices, T *output, int rows,
                      int hidden, int topk, cudaStream_t stream, const T *shared,
                      const T *sharedGate, int sharedGateStride) {
    // A warp per row reduces prefill scheduling overhead. Small decode blocks
    // need the original wider grid; unaligned views retain scalar accesses.
    if (rows >= 1024 && hidden >= 512 && hidden % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(parts) | reinterpret_cast<uintptr_t>(output)) & 15) == 0)
        CloverExpertSumVector<T, Shared><<<(rows + 3) / 4, 128, 0, stream>>>(parts, indices, output, rows,
                                                                          hidden, topk, shared, sharedGate, sharedGateStride);
    else
        CloverExpertSum<T, Shared><<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
            parts, indices, output, hidden, topk, shared, sharedGate, sharedGateStride);
}
}

#include "clover-moe-sm90.cuh"

template<typename T>
bool FastllmCudaCloverRotaryImpl(void *input, int tokens, int heads, int dim, int rotaryDim, int position,
                             float theta, void *stream) {
    if (!input || tokens < 1 || tokens > 65535 || heads < 1 || rotaryDim < 2 || rotaryDim > dim ||
        rotaryDim % 2 || position < 0 || theta <= 0.f)
        return false;
    return Launch<CloverRotary<T>>(dim3((heads * rotaryDim / 2 + 255) / 256, tokens), 256, 0,
                                      stream ? (cudaStream_t)stream : cudaStreamPerThread,
                                      (T *)input, heads, dim, rotaryDim, position, theta);
}
template<typename T>
bool FastllmCudaCloverPrepareAttentionImpl(const void *qgate, const void *key, const float *qWeight,
        const float *kWeight, void *query, void *normalizedKey, void *gate, int tokens,
        int queryHeads, int keyHeads, int rotaryDim, int position, float theta, float eps, void *stream,
        int qkvStride, void *value) {
    if (!qgate || !key || !qWeight || !kWeight || !query || !normalizedKey || !gate ||
        tokens < 1 || tokens > 65535 || queryHeads < 1 || keyHeads < 1 ||
        queryHeads > 65535 || keyHeads > 65535 || rotaryDim < 2 || rotaryDim > 256 ||
        rotaryDim % 2 || position < 0 || theta <= 0.f || eps <= 0.f ||
        qkvStride < 0 || (qkvStride && (qkvStride < (queryHeads + keyHeads) * 512 || !value)) ||
        (!qkvStride && value)) return false;
    return Launch<CloverPrepareAttention<T>>(dim3(queryHeads + keyHeads, tokens), 256, 0, Stream(stream),
        (const T *)qgate, (const T *)key, qWeight, kWeight,
        (T *)query, (T *)normalizedKey, (T *)gate,
        queryHeads, keyHeads, rotaryDim, position, theta, eps, qkvStride, (T *)value);
}
template<typename T>
bool FastllmCudaCloverRouteImpl(const void *logits, int *indices, float *scores,
        int *scratchIds, float *scratchScores, unsigned char *allowed,
        int batch, int tokens, int experts, int topk, float coresetTopP, void *stream, int logitsStride) {
    bool coreset = coresetTopP > 0.f;
    if (logitsStride == 0) logitsStride = experts;
    if (logitsStride < experts) return false;
    if (!logits || !indices || !scores || batch < 1 || tokens < 1 || experts < 1 ||
        experts > 256 || topk < 1 || topk > experts || topk > 16 ||
        (coreset && (!scratchIds || !scratchScores || !allowed ||
                     coresetTopP > 1.f || !ValidBlock(batch, tokens)))) return false;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    if (coreset) {
        if (!FastllmCudaFusedSoftmaxSelectExpert(logits, scratchIds, scratchScores,
                batch * tokens, experts, topk, false, logitsStride, nullptr, tokens,
                stream, std::is_same_v<T, __half>)) return false;
        CloverCoreset<<<batch, 256, 0, s>>>(scratchIds, scratchScores, allowed,
            tokens, experts, topk, coresetTopP);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    return FastllmCudaFusedSoftmaxSelectExpert(logits, indices, scores, batch * tokens,
        experts, topk, true, logitsStride, coreset ? allowed : nullptr, tokens,
        stream, std::is_same_v<T, __half>);
}
template<typename T>
bool FastllmCudaCloverExpertsImpl(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream, int *workspace, void *packed,
    const void *shared, const void *sharedGate, int sharedGateStride) {
    if ((shared == nullptr) != (sharedGate == nullptr) || sharedGateStride < 1)
        return false;
    if (!input || !gateUp || !down || !indices || !scores || !middle || !parts || !output ||
        rows < 1 || hidden < 1 || intermediate < 1 || experts < 1 || topk < 1 ||
        topk > experts || topk > 16 || rows > 65535 / topk) return false;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
#if CUDART_VERSION >= 12000
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    if (workspace && packed &&
        FastllmCudaCloverExpertsPackedBytes(rows, hidden, intermediate, experts, topk) &&
        CloverExpertsTma(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden,
                         intermediate, experts, topk, s, workspace, packed, shared, sharedGate, sharedGateStride))
        return cudaGetLastError() == cudaSuccess;
#endif
    bool grouped = workspace && rows >= 4 && hidden >= 512 && intermediate >= 128 &&
        experts <= 256 && CloverTensorCores<T>();
    if (grouped) {
        CloverBuildExpertTiles(indices, workspace, rows, topk, experts, s);
        int tiles = min(experts, rows * topk) + (rows * topk + 15) / 16;
        if (rows <= 32) {
            CloverExpertGemm<T, true, 32, 128><<<dim3((intermediate + 31) / 32, tiles), 64, 0, s>>>(
                (const T *)input, (const T *)gateUp, scores, workspace,
                (T *)middle, rows, hidden, intermediate, experts, topk);
            CloverExpertGemm<T, false, 32, 128><<<dim3((hidden + 31) / 32, tiles), 64, 0, s>>>(
                (const T *)middle, (const T *)down, scores, workspace,
                (T *)parts, rows, hidden, intermediate, experts, topk);
        } else {
            CloverExpertGemm<T, true, 64, 64><<<dim3((intermediate + 63) / 64, tiles), 128, 0, s>>>(
                (const T *)input, (const T *)gateUp, scores, workspace,
                (T *)middle, rows, hidden, intermediate, experts, topk);
            CloverExpertGemm<T, false, 64, 64><<<dim3((hidden + 63) / 64, tiles), 128, 0, s>>>(
                (const T *)middle, (const T *)down, scores, workspace,
                (T *)parts, rows, hidden, intermediate, experts, topk);
        }
    }
    if (!grouped) {
        CloverExpertGate<T><<<dim3((intermediate + 7) / 8, rows * topk), 256, 0, s>>>(
            (const T *)input, (const T *)gateUp, indices,
            (T *)middle, hidden, intermediate, experts, topk);
        CloverExpertDown<T><<<dim3((hidden + 7) / 8, rows * topk), 256, 0, s>>>(
            (const T *)middle, (const T *)down, indices, scores,
            (T *)parts, hidden, intermediate, experts);
    }
    if (shared)
        CloverSumExperts<T, true>((const T *)parts, indices, (T *)output, rows, hidden,
                               topk, s, (const T *)shared, (const T *)sharedGate, sharedGateStride);
    else
        CloverSumExperts<T, false>((const T *)parts, indices, (T *)output, rows, hidden,
                                topk, s, nullptr, nullptr, 1);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
__global__ void CloverPenalizeLogits(float *logits, const int *counts,
        const int *stops, int numStops, const float *rowBias, int vocab, CloverCudaPenalty p) {
    DependencyStart();
    int token = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (token >= vocab) return;
    float value = logits[(size_t)row * vocab + token];
    if (counts[token]) {
        value = __fsub_rn(value, __fmul_rn(float(counts[token]), p.frequency));
        value = __fsub_rn(value, p.presence);
        if (p.repetition != 1.f) value = value < 0.f ? value * p.repetition : value / p.repetition;
    }
    bool stop = false;
    for (int i = 0; i < numStops; ++i) stop |= token == stops[i];
    if (stop) value += rowBias[row * 2];
    if (token == p.thinkToken) value += rowBias[row * 2 + 1];
    if ((stop && p.firstPosition + row <= p.minTokens) || token == p.maskToken) value = -CUDART_INF_F;
    logits[(size_t)row * vocab + token] = value;
}
__device__ void CloverWarpArgmax(float &value, int &index) {
    for (int d = 16; d; d >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, d);
        int id = __shfl_down_sync(0xffffffff, index, d);
        if (other > value || (other == value && id < index)) { value = other; index = id; }
    }
}
__device__ double CloverBlockSum64(double value, double *scratch) {
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.;
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
    if (threadIdx.x == 0) scratch[0] = value;
    __syncthreads();
    return scratch[0];
}
__global__ void CloverSummarizeLogits(const float *logits, const int *candidates,
        CloverCudaTokenStats *output, int rows, int vocab, bool confidence, bool entropy) {
    DependencyStart();
    const float *scores = logits + (size_t)blockIdx.x * vocab;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    float best = -CUDART_INF_F;
    int index = 0x7fffffff;
    bool nan = false;
    for (int i = tid; i < vocab; i += 256) {
        float v = scores[i]; nan |= isnan(v);
        if (v > best || (v == best && i < index)) { best = v; index = i; }
    }
    int invalid = __syncthreads_or(nan);
    CloverWarpArgmax(best, index);
    __shared__ float maxima[8];
    __shared__ int indices[8];
    __shared__ double scratch[8];
    if (lane == 0) { maxima[warp] = best; indices[warp] = index; }
    __syncthreads();
    if (warp == 0) {
        best = tid < 8 ? maxima[tid] : -CUDART_INF_F;
        index = tid < 8 ? indices[tid] : 0x7fffffff;
        CloverWarpArgmax(best, index);
    }
    __syncthreads();
    if (tid == 0) { maxima[0] = best; indices[0] = index; }
    __syncthreads();
    best = maxima[0]; index = indices[0];
    if (invalid || !isfinite(best)) {
        if (tid == 0) output[blockIdx.x] = {-1, 1, 0.f, 0.f, 0.f};
        return;
    }
    double total = 1., h = 0.;
    if (confidence || entropy) {
        double sum = 0.;
        for (int i = tid; i < vocab; i += 256) sum += expf(scores[i] - best);
        total = CloverBlockSum64(sum, scratch);
        // A second reduction reuses scratch only after every thread reads total.
        __syncthreads();
        if (entropy) {
            double ent = 0.;
            for (int i = tid; i < vocab; i += 256) {
                float probability = float(double(expf(scores[i] - best)) / total);
                ent -= __fmul_rn(probability, logf(fmaxf(probability, 1.e-12f)));
            }
            h = CloverBlockSum64(ent, scratch);
        }
    }
    if (tid == 0) {
        int proposal = candidates && blockIdx.x + 1 < rows ? candidates[blockIdx.x + 1] : index;
        float probability = proposal >= 0 && proposal < vocab
            ? float(double(expf(scores[proposal] - best)) / total) : 0.f;
        output[blockIdx.x] = {index, 0, float(1. / total), probability, float(h)};
    }
}
}
bool FastllmCudaCloverPenalize(float *logits, const int *counts, const int *stops, int numStops,
                               const float *rowBias, int rows, int vocab, CloverCudaPenalty penalty,
                               void *stream) {
    if (!logits || !counts || !rowBias || rows < 1 || rows > 65535 || vocab < 1 || numStops < 0 ||
        (numStops && !stops))
        return false;
    return Launch<CloverPenalizeLogits>(dim3((vocab + 255) / 256, rows), 256, 0, Stream(stream), logits,
                                              counts, stops, numStops, rowBias, vocab, penalty);
}
namespace {
// Each row has a global summary followed by one partial per 8192 logits.
struct CloverStatsTile {
    float maximum;
    int index, error, padding;
    // The summary stores inverse total in the second double; partials store entropy.
    double sum, entropy;
};
__device__ void CloverStatsMaximum(float &best, int &index) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float maxima[8];
    __shared__ int indices[8];
    CloverWarpArgmax(best, index);
    if (lane == 0) { maxima[warp] = best; indices[warp] = index; }
    __syncthreads();
    best = threadIdx.x < 8 ? maxima[threadIdx.x] : -CUDART_INF_F;
    index = threadIdx.x < 8 ? indices[threadIdx.x] : 0x7fffffff;
    CloverWarpArgmax(best, index);
}
__global__ void CloverStatsMaxTiles(const float *logits, CloverStatsTile *workspace, int vocab, int tiles) {
    DependencyStart();
    int begin = blockIdx.x * 8192, end = min(vocab, begin + 8192);
    const float *scores = logits + (size_t)blockIdx.y * vocab;
    float best = -CUDART_INF_F; int index = 0x7fffffff; bool nan = false;
    for (int i = begin + threadIdx.x; i < end; i += 256) {
        float v = scores[i]; nan |= isnan(v);
        if (v > best || (v == best && i < index)) { best = v; index = i; }
    }
    int error = __syncthreads_or(nan);
    CloverStatsMaximum(best, index);
    if (threadIdx.x == 0) workspace[(size_t)blockIdx.y * (tiles + 1) + blockIdx.x + 1] =
        {best, index, error, 0, 0., 0.};
}
__device__ CloverCudaTokenStats CloverStatsResult(const float *scores, const int *candidates,
        const CloverStatsTile &summary, int row, int rows, int vocab) {
    if (summary.error) return {-1, 1, 0.f, 0.f, 0.f};
    int proposed = candidates && row + 1 < rows ? candidates[row + 1] : summary.index;
    float p = proposed >= 0 && proposed < vocab ? float(double(expf(scores[proposed] - summary.maximum)) / summary.sum) : 0.f;
    return {summary.index, 0, float(1. / summary.sum), p, 0.f};
}
__global__ void CloverStatsMergeMax(const float *logits, const int *candidates,
        CloverStatsTile *workspace, CloverCudaTokenStats *output, int rows, int vocab, int tiles, bool greedy) {
    DependencyStart();
    CloverStatsTile *work = workspace + (size_t)blockIdx.x * (tiles + 1);
    float best = -CUDART_INF_F; int index = 0x7fffffff; bool invalid = false;
    for (int i = threadIdx.x; i < tiles; i += 256) {
        auto tile = work[i + 1]; invalid |= tile.error;
        if (tile.maximum > best || (tile.maximum == best && tile.index < index)) { best = tile.maximum; index = tile.index; }
    }
    int error = __syncthreads_or(invalid);
    CloverStatsMaximum(best, index);
    if (threadIdx.x == 0) {
        work[0] = {best, index, error || !isfinite(best), 0, 1., 0.};
        if (greedy) output[blockIdx.x] = CloverStatsResult(logits + (size_t)blockIdx.x * vocab, candidates, work[0], blockIdx.x, rows, vocab);
    }
}
template<bool Entropy>
__global__ void CloverStatsSumTiles(const float *logits, CloverStatsTile *workspace, int vocab, int tiles) {
    DependencyStart();
    CloverStatsTile *work = workspace + (size_t)blockIdx.y * (tiles + 1);
    auto summary = work[0];
    if (summary.error) return;
    const float *scores = logits + (size_t)blockIdx.y * vocab;
    int begin = blockIdx.x * 8192, end = min(vocab, begin + 8192);
    double sum = 0.;
    for (int i = begin + threadIdx.x; i < end; i += 256) {
        float p = expf(scores[i] - summary.maximum);
        if (Entropy) {
            // One FP64 reciprocal per row avoids a double division per logit.
            // Probabilities still round to FP32 before the entropy product.
            p = float(double(p) * summary.entropy);
            sum -= __fmul_rn(p, logf(fmaxf(p, 1.e-12f)));
        } else sum += p;
    }
    __shared__ double scratch[8];
    sum = CloverBlockSum64(sum, scratch);
    if (threadIdx.x == 0) {
        if (Entropy) work[blockIdx.x + 1].entropy = sum;
        else work[blockIdx.x + 1].sum = sum;
    }
}
template<bool Entropy>
__global__ void CloverStatsMergeSum(const float *logits, const int *candidates,
        CloverStatsTile *workspace, CloverCudaTokenStats *output, int rows, int vocab, int tiles) {
    DependencyStart();
    CloverStatsTile *work = workspace + (size_t)blockIdx.x * (tiles + 1);
    if (work[0].error) {
        if (threadIdx.x == 0) output[blockIdx.x] = {-1, 1, 0.f, 0.f, 0.f};
        return;
    }
    double sum = 0.;
    for (int i = threadIdx.x; i < tiles; i += 256) sum += Entropy ? work[i + 1].entropy : work[i + 1].sum;
    __shared__ double scratch[8];
    sum = CloverBlockSum64(sum, scratch);
    if (threadIdx.x == 0) {
        if (Entropy) output[blockIdx.x].entropy = float(sum);
        else {
            work[0].sum = sum; work[0].entropy = 1. / sum;
            output[blockIdx.x] = CloverStatsResult(logits + (size_t)blockIdx.x * vocab, candidates, work[0], blockIdx.x, rows, vocab);
        }
    }
}
}
static bool CloverTokenStatsSplit(const float *logits, const int *candidates,
        CloverCudaTokenStats *output, int rows, int vocab, bool confidence, bool entropy, void *stream, void *scratch) {
    if (!logits || !output || !scratch || rows < 1 || vocab < 1) return false;
    auto s = Stream(stream); auto *work = (CloverStatsTile *)scratch;
    int tiles = (vocab - 1) / 8192 + 1;
    if (!Launch<CloverStatsMaxTiles>(dim3(tiles, rows), 256, 0, s, logits, work, vocab, tiles))
        return false;
    if (!Launch<CloverStatsMergeMax>(rows, 256, 0, s, logits, candidates, work, output, rows, vocab,
                                           tiles, !confidence && !entropy))
        return false;
    if (confidence || entropy) {
        if (!Launch<CloverStatsSumTiles<false>>(dim3(tiles, rows), 256, 0, s, logits, work, vocab,
                                                      tiles))
            return false;
        if (!Launch<CloverStatsMergeSum<false>>(rows, 256, 0, s, logits, candidates, work, output, rows,
                                                      vocab, tiles))
            return false;
    }
    if (entropy) {
        if (!Launch<CloverStatsSumTiles<true>>(dim3(tiles, rows), 256, 0, s, logits, work, vocab,
                                                     tiles))
            return false;
        if (!Launch<CloverStatsMergeSum<true>>(rows, 256, 0, s, logits, candidates, work, output, rows,
                                                     vocab, tiles))
            return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

std::size_t FastllmCudaCloverTokenStatsWorkspaceBytes(int rows, int vocab) {
    if (rows < 1 || rows > 65535 || vocab < 16384) return 0;
    return (size_t)rows * ((vocab - 1) / 8192 + 2) * sizeof(CloverStatsTile);
}

bool FastllmCudaCloverTokenStats(const float *logits, const int *candidates, CloverCudaTokenStats *output,
                                 int rows, int vocab, bool confidence, bool entropy, void *stream,
                                 void *workspace) {
    if (!logits || !output || rows < 1 || vocab < 1)
        return false;
    if (workspace && FastllmCudaCloverTokenStatsWorkspaceBytes(rows, vocab))
        return CloverTokenStatsSplit(logits, candidates, output, rows, vocab, confidence, entropy, stream,
                                     workspace);
    return Launch<CloverSummarizeLogits>(rows, 256, 0, Stream(stream), logits, candidates, output, rows,
                                               vocab, confidence, entropy);
}

namespace {
template<typename T, int Vector = 1>
__global__ void CloverUnpackGdn(const T *input, const float *aLog, const float *bias,
        T *qkv, T *z, float *g, T *beta,
        int channels, int heads) {
    DependencyStart();
    int d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    int values = heads * 128, width = channels + values + 2 * heads;
    size_t offset = (size_t)row * width;
    if constexpr (Vector == 8) {
        // Large prefills copy eight BF16 values per lane. This reduces the
        // number of tiny copy CTAs without changing the scalar gate math.
        if (d < channels / 8) reinterpret_cast<uint4 *>(qkv + (size_t)row * channels)[d] =
            reinterpret_cast<const uint4 *>(input + offset)[d];
        if (d < values / 8) reinterpret_cast<uint4 *>(z + (size_t)row * values)[d] =
            reinterpret_cast<const uint4 *>(input + offset + channels)[d];
    } else {
        if (d < channels) qkv[(size_t)row * channels + d] = input[offset + d];
        if (d < values) z[(size_t)row * values + d] = input[offset + channels + d];
    }
    if (d < heads) {
        float a = CloverFloat(input[offset + channels + values + d]) + bias[d];
        g[(size_t)row * heads + d] = -expf(aLog[d]) * (a > 20.f ? a : log1pf(expf(a)));
        if (beta) {
            float b = CloverFloat(input[offset + channels + values + heads + d]);
            beta[(size_t)row * heads + d] = CloverCast<T>(1.f / (1.f + expf(-b)));
        }
    }
}
}
template<typename T>
bool FastllmCudaCloverUnpackGdnImpl(const void *input, const float *aLog, const float *bias, void *qkv, void *z,
                                float *g, void *beta, int rows, int channels, int heads, void *stream) {
    if (!input || !aLog || !bias || !qkv || !z || !g || rows < 1 || rows > 65535 || channels < 1 || heads < 1)
        return false;
    int width = max(channels, heads * 128);
    // Each projected row must also remain aligned: its final two head vectors
    // contribute 2 * heads BF16 elements to the stride.
    if (rows >= 128 && channels % 8 == 0 && heads % 4 == 0 &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(qkv) |
          reinterpret_cast<uintptr_t>(z)) &
         15) == 0) {
        if (!Launch<CloverUnpackGdn<T, 8>>(dim3((width / 8 + 255) / 256, rows), 256, 0, Stream(stream),
                                              (const T *)input, aLog, bias, (T *)qkv,
                                              (T *)z, g, (T *)beta, channels, heads))
            return false;
    } else {
        if (!Launch<CloverUnpackGdn<T, 1>>(dim3((width + 255) / 256, rows), 256, 0, Stream(stream),
                                              (const T *)input, aLog, bias, (T *)qkv,
                                              (T *)z, g, (T *)beta, channels, heads))
            return false;
    }
    return true;
}

bool FastllmCudaCloverPreviewConv(const void *input, const void *history, const void *weight,
                                  const void *rightWeight, const int *slots, const unsigned char *valid,
                                  void *output, int batch, int tokens, int channels, void *stream, const CloverCudaGdnCaches *caches, bool fp16) {
    return fp16 ? FastllmCudaCloverPreviewConvImpl<__half>(input, history, weight, rightWeight, slots, valid, output, batch, tokens, channels, stream, caches)
                : FastllmCudaCloverPreviewConvImpl<__nv_bfloat16>(input, history, weight, rightWeight, slots, valid, output, batch, tokens, channels, stream, caches);
}

bool FastllmCudaCloverPreviewGdn(const void *query, const void *key, const void *value, const float *g,
                                 const float *state, const float *logitScale, const float *historyBias,
                                 const float *historyScale, const int *slots, const unsigned char *valid,
                                 void *output, int batch, int tokens, int keyHeads, int valueHeads,
                                 void *stream, int qkvRowStride, const CloverCudaGdnCaches *caches, bool fp16) {
    return fp16 ? FastllmCudaCloverPreviewGdnImpl<__half>(query, key, value, g, state, logitScale, historyBias, historyScale, slots, valid, output, batch, tokens, keyHeads, valueHeads, stream, qkvRowStride, caches)
                : FastllmCudaCloverPreviewGdnImpl<__nv_bfloat16>(query, key, value, g, state, logitScale, historyBias, historyScale, slots, valid, output, batch, tokens, keyHeads, valueHeads, stream, qkvRowStride, caches);
}

bool FastllmCudaCloverCausalConv(const void *input, void *history, const void *weight, void *output,
                                 void *snapshots, int batch, int tokens, int channels, void *stream, const CloverCudaGdnCaches *caches, bool fp16) {
    return fp16 ? FastllmCudaCloverCausalConvImpl<__half>(input, history, weight, output, snapshots, batch, tokens, channels, stream, caches)
                : FastllmCudaCloverCausalConvImpl<__nv_bfloat16>(input, history, weight, output, snapshots, batch, tokens, channels, stream, caches);
}

bool FastllmCudaCloverCausalGdn(const void *query, const void *key, const void *value, const float *g,
                                const void *beta, float *state, void *output, float *snapshots, int batch,
                                int tokens, int keyHeads, int valueHeads, void *stream, float *workspace,
                                int qkvRowStride, const CloverCudaGdnCaches *caches, bool fp16) {
    return fp16 ? FastllmCudaCloverCausalGdnImpl<__half>(query, key, value, g, beta, state, output, snapshots, batch, tokens, keyHeads, valueHeads, stream, workspace, qkvRowStride, caches)
                : FastllmCudaCloverCausalGdnImpl<__nv_bfloat16>(query, key, value, g, beta, state, output, snapshots, batch, tokens, keyHeads, valueHeads, stream, workspace, qkvRowStride, caches);
}

bool FastllmCudaCloverRotary(void *input, int tokens, int heads, int dim, int rotaryDim, int position,
                             float theta, void *stream, bool fp16) {
    return fp16 ? FastllmCudaCloverRotaryImpl<__half>(input, tokens, heads, dim, rotaryDim, position, theta, stream)
                : FastllmCudaCloverRotaryImpl<__nv_bfloat16>(input, tokens, heads, dim, rotaryDim, position, theta, stream);
}

bool FastllmCudaCloverPrepareAttention(const void *qgate, const void *key, const float *qWeight,
        const float *kWeight, void *query, void *normalizedKey, void *gate, int tokens,
        int queryHeads, int keyHeads, int rotaryDim, int position, float theta, float eps, void *stream,
        int qkvStride, void *value, bool fp16) {
    return fp16 ? FastllmCudaCloverPrepareAttentionImpl<__half>(qgate, key, qWeight, kWeight, query, normalizedKey, gate, tokens, queryHeads, keyHeads, rotaryDim, position, theta, eps, stream, qkvStride, value)
                : FastllmCudaCloverPrepareAttentionImpl<__nv_bfloat16>(qgate, key, qWeight, kWeight, query, normalizedKey, gate, tokens, queryHeads, keyHeads, rotaryDim, position, theta, eps, stream, qkvStride, value);
}

bool FastllmCudaCloverRoute(const void *logits, int *indices, float *scores,
        int *scratchIds, float *scratchScores, unsigned char *allowed,
        int batch, int tokens, int experts, int topk, float coresetTopP, void *stream, int logitsStride, bool fp16) {
    return fp16 ? FastllmCudaCloverRouteImpl<__half>(logits, indices, scores, scratchIds, scratchScores, allowed, batch, tokens, experts, topk, coresetTopP, stream, logitsStride)
                : FastllmCudaCloverRouteImpl<__nv_bfloat16>(logits, indices, scores, scratchIds, scratchScores, allowed, batch, tokens, experts, topk, coresetTopP, stream, logitsStride);
}

bool FastllmCudaCloverExperts(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream, int *workspace, void *packed,
    const void *shared, const void *sharedGate, int sharedGateStride, bool fp16) {
    return fp16 ? FastllmCudaCloverExpertsImpl<__half>(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden, intermediate, experts, topk, stream, workspace, packed, shared, sharedGate, sharedGateStride)
                : FastllmCudaCloverExpertsImpl<__nv_bfloat16>(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden, intermediate, experts, topk, stream, workspace, packed, shared, sharedGate, sharedGateStride);
}

bool FastllmCudaCloverUnpackGdn(const void *input, const float *aLog, const float *bias, void *qkv, void *z,
                                float *g, void *beta, int rows, int channels, int heads, void *stream, bool fp16) {
    return fp16 ? FastllmCudaCloverUnpackGdnImpl<__half>(input, aLog, bias, qkv, z, g, beta, rows, channels, heads, stream)
                : FastllmCudaCloverUnpackGdnImpl<__nv_bfloat16>(input, aLog, bias, qkv, z, g, beta, rows, channels, heads, stream);
}
