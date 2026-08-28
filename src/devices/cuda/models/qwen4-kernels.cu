#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <vector>

namespace {
    template <typename T>
    __device__ __forceinline__ float Qwen4CudaToFloat(T value) {
        return (float)value;
    }

    template <>
    __device__ __forceinline__ float Qwen4CudaToFloat<half>(half value) {
        return __half2float(value);
    }

    template <>
    __device__ __forceinline__ float Qwen4CudaToFloat<__nv_bfloat16>(
            __nv_bfloat16 value) {
        return __bfloat162float(value);
    }

    template <typename T>
    __device__ __forceinline__ T Qwen4CudaFromFloat(float value);

    template <>
    __device__ __forceinline__ float Qwen4CudaFromFloat<float>(float value) {
        return value;
    }

    template <>
    __device__ __forceinline__ half Qwen4CudaFromFloat<half>(float value) {
        return __float2half_rn(value);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16
    Qwen4CudaFromFloat<__nv_bfloat16>(float value) {
        return __float2bfloat16_rn(value);
    }

    template <typename T>
    __device__ __forceinline__ float Qwen4CudaRound(float value) {
        return Qwen4CudaToFloat(Qwen4CudaFromFloat<T>(value));
    }

    __device__ __forceinline__ float Qwen4CudaSigmoid(float value) {
        return 1.0f / (1.0f + expf(-value));
    }

    template <typename T>
    __device__ __forceinline__ float Qwen4CudaSigmoidRounded(T value) {
        return Qwen4CudaRound<T>(Qwen4CudaSigmoid(
            Qwen4CudaToFloat(value)));
    }

    template <>
    __device__ __forceinline__ float Qwen4CudaSigmoidRounded<float>(
            float value) {
        return 1.0 / (1.0 + expf(-value));
    }

    template <>
    __device__ __forceinline__ float Qwen4CudaSigmoidRounded<half>(
            half value) {
#ifdef CUDA_NO_TENSOR_CORE
        return __half2float(__float2half_rn(
            1.0f / (1.0f + expf(-__half2float(value)))));
#else
        return __half2float(__hdiv(
            __float2half_rn(1.0f),
            __hadd(__float2half_rn(1.0f), hexp(-value))));
#endif
    }

    template <typename T, int THREADS>
    __global__ void Qwen4GroupedRMSNormKernel(
            const T *input, const float *weight, T *output,
            int rows, int groups, int groupChannels, float eps) {
        const int item = blockIdx.x;
        if (item >= rows * groups) {
            return;
        }
        const int group = item % groups;
        const int row = item / groups;
        const int channels = groups * groupChannels;
        const uint64_t base = (uint64_t)row * channels +
                              (uint64_t)group * groupChannels;

        const T *groupInput = input + base;
        T *groupOutput = output + base;
        const float *groupWeight = weight +
                                   (uint64_t)group * groupChannels;

        // Match the vector mapping used by FastllmCudaRMSNorm.  Besides being
        // faster than scalar loads, keeping the same reduction tree preserves
        // the legacy path's floating-point result when the per-stream width is
        // vector aligned (as it is for Qwen4-Exp).
        float sum = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            if ((groupChannels & 3) == 0) {
                const int packedChannels = groupChannels / 4;
                const float4 *packedInput =
                    reinterpret_cast<const float4 *>(groupInput);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float4 value = packedInput[index];
                    sum += value.x * value.x + value.y * value.y +
                           value.z * value.z + value.w * value.w;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const float value = groupInput[channel];
                    sum += value * value;
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const half2 *packedInput =
                    reinterpret_cast<const half2 *>(groupInput);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float2 value = __half22float2(packedInput[index]);
                    sum += value.x * value.x + value.y * value.y;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const float value = __half2float(groupInput[channel]);
                    sum += value * value;
                }
            }
        } else {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const __nv_bfloat162 *packedInput =
                    reinterpret_cast<const __nv_bfloat162 *>(groupInput);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const __nv_bfloat162 value = packedInput[index];
                    const float low = __bfloat162float(value.x);
                    const float high = __bfloat162float(value.y);
                    sum += low * low + high * high;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const float value =
                        __bfloat162float(groupInput[channel]);
                    sum += value * value;
                }
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        __shared__ float warpSums[THREADS / 32];
        __shared__ float scale;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warpSums[warp] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            float total = lane < THREADS / 32 ? warpSums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            if (lane == 0) {
                scale = rsqrtf(total / groupChannels + eps);
            }
        }
        __syncthreads();

        if constexpr (std::is_same_v<T, float>) {
            if ((groupChannels & 3) == 0) {
                const int packedChannels = groupChannels / 4;
                const float4 *packedInput =
                    reinterpret_cast<const float4 *>(groupInput);
                float4 *packedOutput =
                    reinterpret_cast<float4 *>(groupOutput);
                const float4 *packedWeight =
                    reinterpret_cast<const float4 *>(groupWeight);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float4 value = packedInput[index];
                    const float4 currentWeight = packedWeight[index];
                    packedOutput[index] = {
                        value.x * scale * currentWeight.x,
                        value.y * scale * currentWeight.y,
                        value.z * scale * currentWeight.z,
                        value.w * scale * currentWeight.w};
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupOutput[channel] = groupInput[channel] * scale *
                                           groupWeight[channel];
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const half2 *packedInput =
                    reinterpret_cast<const half2 *>(groupInput);
                half2 *packedOutput = reinterpret_cast<half2 *>(groupOutput);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float2 value = __half22float2(packedInput[index]);
                    packedOutput[index] = __floats2half2_rn(
                        value.x * scale * groupWeight[index * 2],
                        value.y * scale * groupWeight[index * 2 + 1]);
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupOutput[channel] = __float2half_rn(
                        __half2float(groupInput[channel]) * scale *
                        groupWeight[channel]);
                }
            }
        } else {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const __nv_bfloat162 *packedInput =
                    reinterpret_cast<const __nv_bfloat162 *>(groupInput);
                __nv_bfloat162 *packedOutput =
                    reinterpret_cast<__nv_bfloat162 *>(groupOutput);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const __nv_bfloat162 value = packedInput[index];
                    __nv_bfloat162 result;
                    result.x = __float2bfloat16_rn(
                        __bfloat162float(value.x) * scale *
                        groupWeight[index * 2]);
                    result.y = __float2bfloat16_rn(
                        __bfloat162float(value.y) * scale *
                        groupWeight[index * 2 + 1]);
                    packedOutput[index] = result;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupOutput[channel] = __float2bfloat16_rn(
                        __bfloat162float(groupInput[channel]) * scale *
                        groupWeight[channel]);
                }
            }
        }
    }

    template <typename T, typename LogitT>
    __global__ void Qwen4HyperMixKernel(
            const T *normalized, const LogitT *mixLogits, T *output,
            uint64_t count, int groups, int outputChannels) {
        const uint64_t outputIndex = (uint64_t)blockIdx.x * blockDim.x +
                                     threadIdx.x;
        if (outputIndex >= count) {
            return;
        }
        const uint64_t row = outputIndex / outputChannels;
        const int channel = (int)(outputIndex % outputChannels);
        const uint64_t inputBase = row * groups * outputChannels;
        float sum = 0.0f;
        for (int group = 0; group < groups; group++) {
            const uint64_t inputIndex = inputBase +
                (uint64_t)group * outputChannels + channel;
            const float nativeGate = Qwen4CudaSigmoidRounded<LogitT>(
                mixLogits[inputIndex]);
            const float gate = Qwen4CudaRound<T>(nativeGate);
            const float product = Qwen4CudaRound<T>(
                Qwen4CudaToFloat(normalized[inputIndex]) * gate);
            sum = group == 0 ? product : Qwen4CudaRound<T>(sum + product);
        }
        output[outputIndex] = Qwen4CudaFromFloat<T>(
            Qwen4CudaRound<T>(sum / groups));
    }

    template <typename T, typename LogitT>
    void Qwen4LaunchHyperMix(const void *normalized, const void *mixLogits,
                             void *output, int blocks, int threads,
                             uint64_t count, int groups,
                             int outputChannels) {
        Qwen4HyperMixKernel<T, LogitT><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const T*)normalized, (const LogitT*)mixLogits, (T*)output,
            count, groups, outputChannels);
    }

    template <typename T>
    __global__ void Qwen4HyperInjectKernel(
            const T *input, T *output, uint64_t count, int groups) {
        const uint64_t index = (uint64_t)blockIdx.x * blockDim.x +
                               threadIdx.x;
        if (index >= count) {
            return;
        }
        const float scaled = Qwen4CudaRound<T>(
            Qwen4CudaToFloat(input[index]) / groups);
        const float gate = Qwen4CudaSigmoidRounded<T>(
            Qwen4CudaFromFloat<T>(scaled));
        output[index] = Qwen4CudaFromFloat<T>(
            Qwen4CudaRound<T>(gate * 2.0f));
    }

    template <typename T, int THREADS>
    __global__ void Qwen4PLEGateKernel(
            const T *key, const T *query, const T *value,
            float *output, int rows, int groups, int groupChannels) {
        const int item = blockIdx.x;
        if (item >= rows * groups) {
            return;
        }
        const int row = item / groups;
        const int group = item % groups;
        const int channels = groups * groupChannels;
        const uint64_t base = (uint64_t)row * channels +
                              (uint64_t)group * groupChannels;

        __shared__ float probability;
        if (threadIdx.x == 0) {
            // This score feeds MoE routing after PLE. Even a few ULPs from a
            // parallel reduction can change an expert tie and amplify across
            // later layers, so preserve the reference's left-to-right sum.
            float dot = 0.0f;
            for (int channel = 0; channel < groupChannels; channel++) {
                dot += Qwen4CudaToFloat(key[base + channel]) *
                       Qwen4CudaToFloat(query[base + channel]);
            }
            // Match RunPLE's host expression exactly.  rsqrtf differs by one
            // ULP on this model and that is enough to perturb a later MoE tie.
            const float inverseSqrt = 1.0f / sqrtf((float)groupChannels);
            float gate = dot * inverseSqrt;
            if (gate != 0.0f) {
                gate = copysignf(
                    sqrtf(fmaxf(fabsf(gate), 1e-6f)), gate);
            }
            probability = Qwen4CudaSigmoid(gate);
        }
        __syncthreads();

        const uint64_t valueBase = (uint64_t)row * groupChannels;
        for (int channel = threadIdx.x; channel < groupChannels;
             channel += THREADS) {
            output[base + channel] = probability *
                Qwen4CudaToFloat(value[valueBase + channel]);
        }
    }

    __global__ void Qwen4PLECausalConvKernel(
            const float *input, const float *gated, const float *weight,
            const float *history, float *output, uint64_t total,
            int sequence, int channels, int kernel, int dilation,
            int historyLength) {
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int token = (int)(item / channels);
            const int channel = (int)(item % channels);
            float sum = 0.0f;
            for (int tap = 0; tap < kernel; tap++) {
                const int relative =
                    token - historyLength + tap * dilation;
                const float sample = relative >= 0
                    ? input[(uint64_t)relative * channels + channel]
                    : history[(uint64_t)(historyLength + relative) *
                                  channels + channel];
                sum += sample * weight[(uint64_t)channel * kernel + tap];
            }
            output[item] = gated[item] + sum / (1.0f + expf(-sum));
        }
    }

    __global__ void Qwen4PLEUpdateHistoryKernel(
            const float *input, const float *history, float *newHistory,
            uint64_t total, int sequence, int channels,
            int historyLength) {
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int historyIndex = (int)(item / channels);
            const int channel = (int)(item % channels);
            const int sourceToken =
                sequence - historyLength + historyIndex;
            newHistory[item] = sourceToken >= 0
                ? input[(uint64_t)sourceToken * channels + channel]
                : history[(uint64_t)(historyLength + sourceToken) *
                              channels + channel];
        }
    }

    __device__ __forceinline__ float Qwen4CudaLoadActivation(
            const void *data, int type, uint64_t index) {
        if (type == (int)fastllm::DataType::FLOAT32) {
            return ((const float*)data)[index];
        }
        if (type == (int)fastllm::DataType::FLOAT16) {
            return __half2float(((const half*)data)[index]);
        }
        return __bfloat162float(((const __nv_bfloat16*)data)[index]);
    }

    template <typename T>
    __global__ void Qwen4HyperCombineKernel(
            const T *hyperInput, const void *blockOutput, int blockType,
            const void *injection, int injectionType,
            T *output, uint64_t count, int groups, int blockChannels) {
        const uint64_t index = (uint64_t)blockIdx.x * blockDim.x +
                               threadIdx.x;
        if (index >= count) {
            return;
        }
        const int channels = groups * blockChannels;
        const uint64_t row = index / channels;
        const int withinRow = (int)(index % channels);
        const int group = withinRow / blockChannels;
        const int channel = withinRow % blockChannels;
        const float blockValue = Qwen4CudaRound<T>(
            Qwen4CudaLoadActivation(blockOutput, blockType,
                                    row * blockChannels + channel));
        const float scale = Qwen4CudaRound<T>(Qwen4CudaLoadActivation(
            injection, injectionType, row * groups + group));
        const float injected = Qwen4CudaRound<T>(blockValue * scale);
        output[index] = Qwen4CudaFromFloat<T>(Qwen4CudaRound<T>(
            Qwen4CudaToFloat(hyperInput[index]) + injected));
    }

    __global__ void CausalDepthwiseConv1DDecodeKernel(
            const void *input, int inputType, const float *weight,
            float *state, float *output, int total, int channels,
            int kernel, bool silu) {
        const int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= total) {
            return;
        }
        const int channel = item % channels;
        float *stateRow = state + (uint64_t)item * kernel;
        for (int tap = 0; tap + 1 < kernel; tap++) {
            stateRow[tap] = stateRow[tap + 1];
        }
        stateRow[kernel - 1] =
            Qwen4CudaLoadActivation(input, inputType, item);
        const float *weightRow = weight + (uint64_t)channel * kernel;
        float value = 0.0f;
        #pragma unroll
        for (int tap = 0; tap < kernel; tap++) {
            value += stateRow[tap] * weightRow[tap];
        }
        if (silu) {
            value = value / (1.0 + expf(-value));
        }
        output[item] = value;
    }

    template <typename T>
    __global__ void CausalDepthwiseConv1DPrefillKernel(
            const T *input, const float *weight, const float *state,
            float *output, uint64_t total, int sequence, int channels,
            int kernel, bool silu) {
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int channel = (int)(item % channels);
            const uint64_t row = item / channels;
            const int token = (int)(row % sequence);
            const int batch = (int)(row / sequence);
            const float *weightRow = weight + (uint64_t)channel * kernel;
            const float *stateRow = state +
                ((uint64_t)batch * channels + channel) * kernel;
            float value = 0.0f;
            for (int tap = 0; tap < kernel; tap++) {
                const int sourceToken = token - kernel + 1 + tap;
                const float sample = sourceToken >= 0
                    ? Qwen4CudaToFloat(input[
                          ((uint64_t)batch * sequence + sourceToken) *
                              channels + channel])
                    : stateRow[kernel + sourceToken];
                value += sample * weightRow[tap];
            }
            if (silu) {
                // Keep bitwise parity with FastllmSiluKernel(float).
                value = value / (1.0 + expf(-value));
            }
            output[item] = value;
        }
    }

    template <typename T>
    __global__ void CausalDepthwiseConv1DPrefillStateKernel(
            const T *input, float *state,
            int total, int sequence, int channels, int kernel) {
        const int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= total) {
            return;
        }
        const int batch = item / channels;
        const int channel = item % channels;
        float *stateRow = state + (uint64_t)item * kernel;
        if (sequence >= kernel) {
            for (int slot = 0; slot < kernel; slot++) {
                stateRow[slot] = Qwen4CudaToFloat(input[
                    ((uint64_t)batch * sequence + sequence - kernel + slot) *
                        channels + channel]);
            }
        } else {
            for (int slot = 0; slot < kernel - sequence; slot++) {
                stateRow[slot] = stateRow[slot + sequence];
            }
            for (int token = 0; token < sequence; token++) {
                stateRow[kernel - sequence + token] =
                    Qwen4CudaToFloat(input[
                        ((uint64_t)batch * sequence + token) * channels +
                            channel]);
            }
        }
    }

    template <typename T>
    __global__ __launch_bounds__(128) void Qwen4GatedDeltaRuleDecodeKernel(
            const float *qkv, const T *alpha, const T *beta,
            const float *aLog, const float *dtBias, float *state,
            float *output, int keyHeads, int valueHeads,
            float recurrentEps, float inverseHead) {
        constexpr int KEY_DIM = 128;
        constexpr int VALUE_DIM = 128;
        const int item = blockIdx.x;
        const int batch = item / valueHeads;
        const int valueHead = item - batch * valueHeads;
        const int repeat = valueHeads / keyHeads;
        const int keyHead = valueHead / repeat;
        const int qkvChannels = (2 * keyHeads + valueHeads) * KEY_DIM;
        const uint64_t qkvBase = (uint64_t)batch * qkvChannels;
        const float *queryRaw = qkv + qkvBase + keyHead * KEY_DIM;
        const float *keyRaw = qkv + qkvBase +
                              (keyHeads + keyHead) * KEY_DIM;
        const float *value = qkv + qkvBase +
                             (2 * keyHeads + valueHead) * KEY_DIM;
        float *headState = state +
            (uint64_t)item * KEY_DIM * VALUE_DIM;

        __shared__ float queryNorm[KEY_DIM];
        __shared__ float keyNorm[KEY_DIM];
        __shared__ float normScales[2];
        __shared__ float gateValues[2];

        const int tid = threadIdx.x;
        if (tid < 32) {
            const float4 queryValue = ((const float4*)queryRaw)[tid];
            const float4 keyValue = ((const float4*)keyRaw)[tid];
            float querySquares = queryValue.x * queryValue.x +
                                 queryValue.y * queryValue.y +
                                 queryValue.z * queryValue.z +
                                 queryValue.w * queryValue.w;
            float keySquares = keyValue.x * keyValue.x +
                               keyValue.y * keyValue.y +
                               keyValue.z * keyValue.z +
                               keyValue.w * keyValue.w;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                querySquares += __shfl_down_sync(
                    0xffffffffu, querySquares, offset);
                keySquares += __shfl_down_sync(
                    0xffffffffu, keySquares, offset);
            }
            if (tid == 0) {
                normScales[0] = rsqrtf(
                    querySquares / KEY_DIM + recurrentEps);
                normScales[1] = rsqrtf(
                    keySquares / KEY_DIM + recurrentEps);
            }
        }
        if (tid == 0) {
            const float alphaValue = Qwen4CudaToFloat(alpha[item]);
            const float biasedAlpha = alphaValue + dtBias[valueHead];
            const float softplus = biasedAlpha > 20.0f
                ? biasedAlpha : log1p(expf(biasedAlpha));
            const float betaValue = Qwen4CudaToFloat(beta[item]);
            // The reference path casts both projections to float32 before
            // Sigmoid/MambaSoftplus.  Mirror its expressions exactly while
            // consuming the original activation values in-place.
            gateValues[0] = 1.0 / (1.0 + expf(-betaValue));
            gateValues[1] = expf(
                -expf((double)aLog[valueHead]) * softplus);
        }
        __syncthreads();

        // The reference executes RMSNorm and the extra query scaling as two
        // operations.  Preserve the intervening float32 write/read boundary;
        // otherwise nvcc may reassociate four multiplies and shift the final
        // fp16 core output by one ULP.
        queryNorm[tid] = queryRaw[tid] * normScales[0] * inverseHead;
        keyNorm[tid] = keyRaw[tid] * normScales[1] * inverseHead;
        __syncthreads();
        queryNorm[tid] *= inverseHead;
        __syncthreads();

        float memory = 0.0f;
#pragma unroll
        for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
            const uint64_t stateIndex =
                (uint64_t)keyChannel * VALUE_DIM + tid;
            const float scaled = headState[stateIndex] * gateValues[1];
            headState[stateIndex] = scaled;
            memory += scaled * keyNorm[keyChannel];
        }
        const float delta = (value[tid] - memory) * gateValues[0];
        float core = 0.0f;
#pragma unroll
        for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
            const uint64_t stateIndex =
                (uint64_t)keyChannel * VALUE_DIM + tid;
            const float updated = headState[stateIndex] +
                                  keyNorm[keyChannel] * delta;
            headState[stateIndex] = updated;
            core += updated * queryNorm[keyChannel];
        }
        output[(uint64_t)item * VALUE_DIM + tid] = core;
    }

    template <typename T>
    __global__ void Qwen4QSAScoreKernel(
            const T *query, const float *compressedKeys, float *scores,
            int rows, int blocks, int heads, int headDim,
            float inverseSqrt, int queryStart, int compressRatio) {
        const int lane = threadIdx.x & 31;
        const int warpInBlock = threadIdx.x >> 5;
        const int warpsPerBlock = blockDim.x >> 5;
        const uint64_t total = (uint64_t)rows * blocks;
        for (uint64_t item =
                 (uint64_t)blockIdx.x * warpsPerBlock + warpInBlock;
             item < total;
             item += (uint64_t)warpsPerBlock * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = queryStart >= 0
                ? (queryStart + row + 1) / compressRatio : blocks;
            if (block >= rowBlocks) {
                continue;
            }
            const float *blockKey = compressedKeys +
                                    (uint64_t)block * headDim;
            float dot = 0.0f;
            if (lane < heads) {
                const T *headQuery = query +
                    ((uint64_t)row * heads + lane) * headDim;
                // One lane owns one indexer head and retains the reference's
                // serial column accumulation order. The warp only parallelizes
                // independent heads, so score values and TopK ties stay stable.
                for (int column = 0; column < headDim; column++) {
                    dot += Qwen4CudaToFloat(headQuery[column]) *
                           blockKey[column];
                }
            }
            float score = 0.0f;
            for (int head = 0; head < heads; head++) {
                score += fmaxf(__shfl_sync(0xffffffffu, dot, head), 0.0f);
            }
            if (lane == 0) {
                scores[item] = score * inverseSqrt;
            }
        }
    }

    template <typename T>
    __global__ void Qwen4QSAScoreSerialKernel(
            const T *query, const float *compressedKeys, float *scores,
            int rows, int blocks, int heads, int headDim,
            float inverseSqrt, int queryStart, int compressRatio) {
        const uint64_t total = (uint64_t)rows * blocks;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = queryStart >= 0
                ? (queryStart + row + 1) / compressRatio : blocks;
            if (block >= rowBlocks) {
                continue;
            }
            const float *blockKey = compressedKeys +
                                    (uint64_t)block * headDim;
            float score = 0.0f;
            for (int head = 0; head < heads; head++) {
                const T *headQuery = query +
                    ((uint64_t)row * heads + head) * headDim;
                float dot = 0.0f;
                // Match the reference and warp kernel's per-head serial
                // accumulation order exactly. A thread owns the independent
                // score item, avoiding a mostly idle warp when heads is four.
                for (int column = 0; column < headDim; column++) {
                    dot += Qwen4CudaToFloat(headQuery[column]) *
                           blockKey[column];
                }
                score += fmaxf(dot, 0.0f);
            }
            scores[item] = score * inverseSqrt;
        }
    }

    template <typename T>
    __global__ void Qwen4QSAScoreFourHeadKernel(
            const T *query, const float *compressedKeys, float *scores,
            int rows, int blocks, int heads, int headDim,
            float inverseSqrt, int queryStart, int compressRatio) {
        constexpr int lanesPerItem = 4;
        const int lane = threadIdx.x & (lanesPerItem - 1);
        const int itemInBlock = threadIdx.x / lanesPerItem;
        const int itemsPerBlock = blockDim.x / lanesPerItem;
        const uint64_t total = (uint64_t)rows * blocks;
        for (uint64_t item =
                 (uint64_t)blockIdx.x * itemsPerBlock + itemInBlock;
             item < total;
             item += (uint64_t)itemsPerBlock * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = queryStart >= 0
                ? (queryStart + row + 1) / compressRatio : blocks;
            if (block >= rowBlocks) {
                continue;
            }
            const float *blockKey = compressedKeys +
                                    (uint64_t)block * headDim;
            float dot = 0.0f;
            if (lane < heads) {
                const T *headQuery = query +
                    ((uint64_t)row * heads + lane) * headDim;
                for (int column = 0; column < headDim; column++) {
                    dot += Qwen4CudaToFloat(headQuery[column]) *
                           blockKey[column];
                }
            }
            float score = 0.0f;
            for (int head = 0; head < heads; head++) {
                score += fmaxf(__shfl_sync(
                    0xffffffffu, dot, head, lanesPerItem), 0.0f);
            }
            if (lane == 0) {
                scores[item] = score * inverseSqrt;
            }
        }
    }

    __device__ __forceinline__ uint32_t Qwen4QSAOrderedFloatBits(
            float value) {
        uint32_t bits = __float_as_uint(value);
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    // Four radix histogram passes identify the exact K-th score without
    // sorting every compressed block. Equal scores are resolved by the lower
    // block id, and the emitted block ids are already in ascending order.
    __global__ void Qwen4QSARadixSelectKernel(
            const float *scores, int32_t *selectedBlocks,
            int rows, int blocks, int selectedK,
            int queryStart, int compressRatio) {
        const int rowIndex = blockIdx.x;
        if (rowIndex >= rows) {
            return;
        }
        const float *row = scores + (uint64_t)rowIndex * blocks;
        int32_t *output = selectedBlocks +
                          (uint64_t)rowIndex * selectedK;
        const int rowBlocks = queryStart >= 0
            ? (queryStart + rowIndex + 1) / compressRatio : blocks;
        if (rowBlocks <= selectedK) {
            for (int block = threadIdx.x; block < selectedK;
                 block += blockDim.x) {
                output[block] = block < rowBlocks ? block : -1;
            }
            return;
        }

        __shared__ uint32_t histogram[256];
        __shared__ uint32_t prefix;
        __shared__ uint32_t remaining;
        __shared__ uint32_t threadHigher[256];
        __shared__ uint32_t threadPivot[256];
        __shared__ uint32_t threadPivotChosen[256];
        __shared__ uint32_t threadOutputOffset[256];
        if (threadIdx.x == 0) {
            prefix = 0;
            remaining = selectedK;
        }
        __syncthreads();
        #pragma unroll
        for (int round = 0; round < 4; round++) {
            if (threadIdx.x < 256) {
                histogram[threadIdx.x] = 0;
            }
            __syncthreads();
            const int shift = 24 - round * 8;
            const uint32_t mask = round == 0
                ? 0u : (~0u << (32 - round * 8));
            const uint32_t currentPrefix = prefix;
            for (int block = threadIdx.x; block < rowBlocks;
                 block += blockDim.x) {
                const uint32_t ordered =
                    Qwen4QSAOrderedFloatBits(row[block]);
                if ((ordered & mask) == currentPrefix) {
                    atomicAdd(&histogram[(ordered >> shift) & 0xffu], 1u);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t higher = 0;
                for (int bucket = 255; bucket >= 0; bucket--) {
                    const uint32_t count = histogram[bucket];
                    if (higher + count >= remaining) {
                        prefix = currentPrefix |
                                 ((uint32_t)bucket << shift);
                        remaining -= higher;
                        break;
                    }
                    higher += count;
                }
            }
            __syncthreads();
        }
        // Emit the chosen set in physical cache order without a single CUDA
        // thread rescanning the entire context. Give every thread a contiguous
        // block-id range, prefix-sum its counts, then compact in parallel. The
        // first pivot-valued block ids still win ties exactly as before.
        const uint32_t pivot = prefix;
        const int chunk = (rowBlocks + blockDim.x - 1) / blockDim.x;
        const int begin = min(rowBlocks, (int)threadIdx.x * chunk);
        const int end = min(rowBlocks, begin + chunk);
        uint32_t higher = 0;
        uint32_t equal = 0;
        for (int block = begin; block < end; block++) {
            const uint32_t ordered = Qwen4QSAOrderedFloatBits(row[block]);
            higher += ordered > pivot;
            equal += ordered == pivot;
        }
        threadHigher[threadIdx.x] = higher;
        threadPivot[threadIdx.x] = equal;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t higherTotal = 0;
            for (int thread = 0; thread < blockDim.x; thread++) {
                higherTotal += threadHigher[thread];
            }
            uint32_t pivotRemaining = selectedK - higherTotal;
            uint32_t outputOffset = 0;
            for (int thread = 0; thread < blockDim.x; thread++) {
                const uint32_t chosen = min(
                    threadPivot[thread], pivotRemaining);
                threadPivotChosen[thread] = chosen;
                threadOutputOffset[thread] = outputOffset;
                outputOffset += threadHigher[thread] + chosen;
                pivotRemaining -= chosen;
            }
        }
        __syncthreads();
        uint32_t outputIndex = threadOutputOffset[threadIdx.x];
        uint32_t pivotSeen = 0;
        const uint32_t pivotChosen = threadPivotChosen[threadIdx.x];
        for (int block = begin; block < end; block++) {
            const uint32_t ordered = Qwen4QSAOrderedFloatBits(row[block]);
            if (ordered > pivot ||
                (ordered == pivot && pivotSeen++ < pivotChosen)) {
                output[outputIndex++] = block;
            }
        }
    }

    __global__ void Qwen4QSAExpandIndicesKernel(
            const int32_t *selectedBlocks, int32_t *indices,
            int rows, int selectedK, int compressRatio,
            int completeBlocks, int keyLength, int outputWidth,
            int queryStart) {
        const uint64_t total = (uint64_t)rows * outputWidth;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int row = item / outputWidth;
            const int column = item - (uint64_t)row * outputWidth;
            const int visibleLength = queryStart >= 0
                ? queryStart + row + 1 : keyLength;
            const int rowBlocks = queryStart >= 0
                ? visibleLength / compressRatio : completeBlocks;
            const int selectedTokens = min(selectedK, rowBlocks) *
                                       compressRatio;
            if (column < selectedTokens) {
                const int block = selectedBlocks[
                    (uint64_t)row * selectedK + column / compressRatio];
                indices[item] = block * compressRatio +
                                column % compressRatio;
            } else {
                const int tail = column - selectedTokens;
                const int remainder = visibleLength -
                                      rowBlocks * compressRatio;
                indices[item] = tail < remainder
                    ? rowBlocks * compressRatio + tail : -1;
            }
        }
    }

    template <typename T>
    __global__ void Qwen4QSAFillMaskKernel(T *mask, uint64_t count) {
        for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x +
                              threadIdx.x;
             index < count; index += (uint64_t)blockDim.x * gridDim.x) {
            mask[index] = Qwen4CudaFromFloat<T>(1.0f);
        }
    }

    template <typename T>
    __global__ void Qwen4QSAScatterMaskKernel(
            const int32_t *indices, T *mask, int rows,
            int width, int keyLength) {
        const int count = rows * width;
        for (int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < count; index += blockDim.x * gridDim.x) {
            const int row = index / width;
            const int token = indices[index];
            if (token >= 0 && token < keyLength) {
                mask[(uint64_t)row * keyLength + token] =
                    Qwen4CudaFromFloat<T>(0.0f);
            }
        }
    }

    template <typename T>
    __global__ void Qwen4GatherKVKernel(
            const T *key, const T *value, const int32_t *indices,
            T *compactKey, T *compactValue, int keyHeads,
            int keyLength, int headDim, int width) {
        const uint64_t total =
            (uint64_t)keyHeads * width * headDim;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int column = item % headDim;
            const int selected = (item / headDim) % width;
            const int head = item / ((uint64_t)width * headDim);
            const int sourceToken = indices[selected];
            if (sourceToken >= 0 && sourceToken < keyLength) {
                const uint64_t source =
                    ((uint64_t)head * keyLength + sourceToken) * headDim +
                    column;
                compactKey[item] = key[source];
                compactValue[item] = value[source];
            } else {
                compactKey[item] = Qwen4CudaFromFloat<T>(0.0f);
                compactValue[item] = Qwen4CudaFromFloat<T>(0.0f);
            }
        }
    }

    template <typename T>
    __global__ void Qwen4PackSparseQueryKernel(
            const T *query, T *packedQuery, int queryHeads,
            int sequence, int headDim, int rowStart, int rows) {
        const uint64_t total = (uint64_t)rows * queryHeads * headDim;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int column = item % headDim;
            const uint64_t rowHead = item / headDim;
            const int head = rowHead % queryHeads;
            const int row = rowHead / queryHeads;
            const uint64_t source =
                ((uint64_t)head * sequence + rowStart + row) * headDim +
                column;
            packedQuery[item] = query[source];
        }
    }

    template <typename T>
    __global__ void Qwen4GatherSparseBatchKVKernel(
            const T *key, const T *value, const int32_t *indices,
            T *compactKey, T *compactValue, int keyHeads,
            int keyLength, int headDim, int width,
            int rowStart, int rows) {
        const uint64_t total =
            (uint64_t)rows * keyHeads * width * headDim;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int column = item % headDim;
            const uint64_t vector = item / headDim;
            const int selected = vector % width;
            const int head = (vector / width) % keyHeads;
            const int row = vector / ((uint64_t)width * keyHeads);
            const int sourceToken = indices[
                (uint64_t)(rowStart + row) * width + selected];
            if (sourceToken >= 0 && sourceToken < keyLength) {
                const uint64_t source =
                    ((uint64_t)head * keyLength + sourceToken) * headDim +
                    column;
                compactKey[item] = key[source];
                compactValue[item] = value[source];
            } else {
                compactKey[item] = Qwen4CudaFromFloat<T>(0.0f);
                compactValue[item] = Qwen4CudaFromFloat<T>(0.0f);
            }
        }
    }

    template <typename T>
    __global__ void Qwen4SparsePaddingMaskKernel(
            const int32_t *indices, T *mask, int keyLength,
            int width, int rowStart, int rows) {
        const int total = rows * width;
        for (int item = blockIdx.x * blockDim.x + threadIdx.x;
             item < total; item += blockDim.x * gridDim.x) {
            const int row = item / width;
            const int selected = item - row * width;
            const int sourceToken = indices[
                (uint64_t)(rowStart + row) * width + selected];
            mask[item] = Qwen4CudaFromFloat<T>(
                sourceToken >= 0 && sourceToken < keyLength ? 0.0f : 1.0f);
        }
    }

    template <typename T>
    __global__ void Qwen4UnpackSparseOutputKernel(
            const T *packedOutput, T *output, int queryHeads,
            int sequence, int headDim, int rowStart, int rows) {
        const uint64_t total = (uint64_t)rows * queryHeads * headDim;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int column = item % headDim;
            const uint64_t rowHead = item / headDim;
            const int head = rowHead % queryHeads;
            const int row = rowHead / queryHeads;
            const uint64_t destination =
                ((uint64_t)head * sequence + rowStart + row) * headDim +
                column;
            output[destination] = packedOutput[item];
        }
    }

    bool Qwen4CudaActivationType(fastllm::DataType type) {
        return type == fastllm::DataType::FLOAT32 ||
               type == fastllm::DataType::FLOAT16 ||
               type == fastllm::DataType::BFLOAT16;
    }

}

bool FastllmCudaQwen4GroupedRMSNorm(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &output, float eps, int groups) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        output.cudaData == nullptr || input.dims.empty() || groups <= 0 ||
        input.dims.back() % groups != 0 ||
        !Qwen4CudaActivationType(input.dataType) ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        weight.Count(0) != (uint64_t)input.dims.back() ||
        output.dataType != input.dataType || output.dims != input.dims) {
        return false;
    }
    const int channels = input.dims.back();
    const int groupChannels = channels / groups;
    const int rows = (int)(input.Count(0) / channels);
    // Qwen4-Exp uses 2560 channels per hyper stream; FastLLM's regular
    // RMSNorm maps widths in [512, 4096) to 512 threads.  Use the same launch
    // geometry so the fused grouped form has identical reduction order.
    constexpr int threads = 512;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        Qwen4GroupedRMSNormKernel<float, threads><<<
            rows * groups, threads, 0, cudaStreamPerThread>>>(
            (const float*)input.cudaData, (const float*)weight.cudaData,
            (float*)output.cudaData, rows, groups, groupChannels, eps);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Qwen4GroupedRMSNormKernel<half, threads><<<
            rows * groups, threads, 0, cudaStreamPerThread>>>(
            (const half*)input.cudaData, (const float*)weight.cudaData,
            (half*)output.cudaData, rows, groups, groupChannels, eps);
    } else {
        Qwen4GroupedRMSNormKernel<__nv_bfloat16, threads><<<
            rows * groups, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)input.cudaData,
            (const float*)weight.cudaData,
            (__nv_bfloat16*)output.cudaData, rows, groups,
            groupChannels, eps);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4PLEGate(
        const fastllm::Data &key, const fastllm::Data &query,
        const fastllm::Data &value, fastllm::Data &output, int groups) {
    const int channels = key.dims.empty() ? 0 : key.dims.back();
    const int groupChannels = groups > 0 ? channels / groups : 0;
    const int rows = channels > 0
        ? (int)(key.Count(0) / channels) : 0;
    if (key.dataDevice != fastllm::DataDevice::CUDA ||
        query.dataDevice != fastllm::DataDevice::CUDA ||
        value.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        key.cudaData == nullptr || query.cudaData == nullptr ||
        value.cudaData == nullptr || output.cudaData == nullptr ||
        key.dims.empty() || key.dims != query.dims || groups <= 0 ||
        channels % groups != 0 || rows <= 0 ||
        key.dataType != query.dataType || key.dataType != value.dataType ||
        !Qwen4CudaActivationType(key.dataType) ||
        value.Count(0) != (uint64_t)rows * groupChannels ||
        output.dataType != fastllm::DataType::FLOAT32 ||
        output.dims != key.dims) {
        return false;
    }
    constexpr int threads = 512;
    const int blocks = rows * groups;
    if (key.dataType == fastllm::DataType::FLOAT32) {
        Qwen4PLEGateKernel<float, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)key.cudaData, (const float*)query.cudaData,
            (const float*)value.cudaData, (float*)output.cudaData,
            rows, groups, groupChannels);
    } else if (key.dataType == fastllm::DataType::FLOAT16) {
        Qwen4PLEGateKernel<half, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)query.cudaData,
            (const half*)value.cudaData, (float*)output.cudaData,
            rows, groups, groupChannels);
    } else {
        Qwen4PLEGateKernel<__nv_bfloat16, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)key.cudaData,
            (const __nv_bfloat16*)query.cudaData,
            (const __nv_bfloat16*)value.cudaData,
            (float*)output.cudaData, rows, groups, groupChannels);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4PLECausalConv(
        const fastllm::Data &normalized, const fastllm::Data &gated,
        const fastllm::Data &weight, const fastllm::Data &history,
        fastllm::Data &output, fastllm::Data &newHistory,
        int kernel, int dilation) {
    const int channels = normalized.dims.size() == 3
        ? normalized.dims[2] : 0;
    const int sequence = normalized.dims.size() == 3
        ? normalized.dims[1] : 0;
    const int historyLength = (kernel - 1) * dilation;
    if (normalized.dataDevice != fastllm::DataDevice::CUDA ||
        gated.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        history.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        newHistory.dataDevice != fastllm::DataDevice::CUDA ||
        normalized.cudaData == nullptr || gated.cudaData == nullptr ||
        weight.cudaData == nullptr || history.cudaData == nullptr ||
        output.cudaData == nullptr || newHistory.cudaData == nullptr ||
        normalized.dims.size() != 3 || normalized.dims[0] != 1 ||
        normalized.dims != gated.dims || sequence <= 0 || channels <= 0 ||
        kernel <= 1 || dilation <= 0 ||
        normalized.dataType != fastllm::DataType::FLOAT32 ||
        gated.dataType != fastllm::DataType::FLOAT32 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        history.dataType != fastllm::DataType::FLOAT32 ||
        output.dataType != fastllm::DataType::FLOAT32 ||
        newHistory.dataType != fastllm::DataType::FLOAT32 ||
        weight.Count(0) != (uint64_t)channels * kernel ||
        history.dims != std::vector<int>({historyLength, channels}) ||
        output.dims != normalized.dims ||
        newHistory.dims != history.dims) {
        return false;
    }

    constexpr int threads = 256;
    const uint64_t outputCount = (uint64_t)sequence * channels;
    const uint64_t historyCount = (uint64_t)historyLength * channels;
    const int outputBlocks = std::min<uint64_t>(
        65535, (outputCount + threads - 1) / threads);
    const int historyBlocks = std::min<uint64_t>(
        65535, (historyCount + threads - 1) / threads);
    Qwen4PLECausalConvKernel<<<
        outputBlocks, threads, 0, cudaStreamPerThread>>>(
        (const float*)normalized.cudaData,
        (const float*)gated.cudaData,
        (const float*)weight.cudaData,
        (const float*)history.cudaData,
        (float*)output.cudaData, outputCount, sequence, channels,
        kernel, dilation, historyLength);
    Qwen4PLEUpdateHistoryKernel<<<
        historyBlocks, threads, 0, cudaStreamPerThread>>>(
        (const float*)normalized.cudaData,
        (const float*)history.cudaData,
        (float*)newHistory.cudaData, historyCount, sequence, channels,
        historyLength);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4HyperMix(
        const fastllm::Data &normalized,
        const fastllm::Data &mixLogits,
        fastllm::Data &output, int groups) {
    if (normalized.dataDevice != fastllm::DataDevice::CUDA ||
        mixLogits.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        normalized.cudaData == nullptr || mixLogits.cudaData == nullptr ||
        output.cudaData == nullptr || normalized.dims.empty() || groups <= 0 ||
        normalized.dims != mixLogits.dims ||
        normalized.dims.back() % groups != 0 ||
        output.dataType != normalized.dataType ||
        !Qwen4CudaActivationType(normalized.dataType) ||
        !Qwen4CudaActivationType(mixLogits.dataType)) {
        return false;
    }
    const int outputChannels = normalized.dims.back() / groups;
    const uint64_t count = normalized.Count(0) / groups;
    constexpr int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    if (normalized.dataType == fastllm::DataType::FLOAT32) {
        if (mixLogits.dataType == fastllm::DataType::FLOAT32) {
            Qwen4LaunchHyperMix<float, float>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else if (mixLogits.dataType == fastllm::DataType::FLOAT16) {
            Qwen4LaunchHyperMix<float, half>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else {
            Qwen4LaunchHyperMix<float, __nv_bfloat16>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        }
    } else if (normalized.dataType == fastllm::DataType::FLOAT16) {
        if (mixLogits.dataType == fastllm::DataType::FLOAT32) {
            Qwen4LaunchHyperMix<half, float>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else if (mixLogits.dataType == fastllm::DataType::FLOAT16) {
            Qwen4LaunchHyperMix<half, half>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else {
            Qwen4LaunchHyperMix<half, __nv_bfloat16>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        }
    } else {
        if (mixLogits.dataType == fastllm::DataType::FLOAT32) {
            Qwen4LaunchHyperMix<__nv_bfloat16, float>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else if (mixLogits.dataType == fastllm::DataType::FLOAT16) {
            Qwen4LaunchHyperMix<__nv_bfloat16, half>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        } else {
            Qwen4LaunchHyperMix<__nv_bfloat16, __nv_bfloat16>(
                normalized.cudaData, mixLogits.cudaData, output.cudaData,
                blocks, threads, count, groups, outputChannels);
        }
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4HyperInject(
        const fastllm::Data &input, fastllm::Data &output, int groups) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        input.dims.empty() || groups <= 0 || input.dims.back() != groups ||
        input.dataType != output.dataType ||
        !Qwen4CudaActivationType(input.dataType)) {
        return false;
    }
    const uint64_t count = input.Count(0);
    constexpr int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        Qwen4HyperInjectKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)input.cudaData, (float*)output.cudaData,
            count, groups);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Qwen4HyperInjectKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)input.cudaData, (half*)output.cudaData,
            count, groups);
    } else {
        Qwen4HyperInjectKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)input.cudaData,
            (__nv_bfloat16*)output.cudaData, count, groups);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4HyperCombine(
        const fastllm::Data &hyperInput,
        const fastllm::Data &blockOutput,
        const fastllm::Data &injection,
        fastllm::Data &output, int groups) {
    if (hyperInput.dataDevice != fastllm::DataDevice::CUDA ||
        blockOutput.dataDevice != fastllm::DataDevice::CUDA ||
        injection.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        hyperInput.cudaData == nullptr || blockOutput.cudaData == nullptr ||
        injection.cudaData == nullptr || output.cudaData == nullptr ||
        hyperInput.dims.empty() || groups <= 0 ||
        hyperInput.dims.back() % groups != 0 ||
        hyperInput.dataType != output.dataType ||
        !Qwen4CudaActivationType(hyperInput.dataType) ||
        !Qwen4CudaActivationType(blockOutput.dataType) ||
        !Qwen4CudaActivationType(injection.dataType)) {
        return false;
    }
    const int blockChannels = hyperInput.dims.back() / groups;
    const int rows = (int)(hyperInput.Count(0) / hyperInput.dims.back());
    if (blockOutput.Count(0) != (uint64_t)rows * blockChannels ||
        injection.Count(0) != (uint64_t)rows * groups) {
        return false;
    }
    const uint64_t count = hyperInput.Count(0);
    constexpr int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    if (hyperInput.dataType == fastllm::DataType::FLOAT32) {
        Qwen4HyperCombineKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (float*)output.cudaData, count, groups, blockChannels);
    } else if (hyperInput.dataType == fastllm::DataType::FLOAT16) {
        Qwen4HyperCombineKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (half*)output.cudaData, count, groups, blockChannels);
    } else {
        Qwen4HyperCombineKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (__nv_bfloat16*)output.cudaData, count, groups,
            blockChannels);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4QSASelect(
        const fastllm::Data &query,
        const fastllm::Data &compressedKeys,
        fastllm::Data &indices, int keyLength,
        int heads, int headDim, int tokenBudget,
        int compressRatio, int queryStart) {
    const int completeBlocks = compressRatio > 0
        ? keyLength / compressRatio : 0;
    const int selectedK = compressRatio > 0
        ? tokenBudget / compressRatio : 0;
    const int rows = heads > 0 && headDim > 0
        ? (int)(query.Count(0) / ((uint64_t)heads * headDim)) : 0;
    const int outputWidth = compressRatio > 0
        ? (queryStart >= 0
            ? tokenBudget + compressRatio - 1
            : tokenBudget + keyLength % compressRatio)
        : 0;
    if (query.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKeys.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        query.cudaData == nullptr || compressedKeys.cudaData == nullptr ||
        indices.cudaData == nullptr || !Qwen4CudaActivationType(query.dataType) ||
        compressedKeys.dataType != fastllm::DataType::FLOAT32 ||
        indices.dataType != fastllm::DataType::INT32 ||
        compressedKeys.dims.size() != 2 ||
        compressedKeys.dims[0] < completeBlocks ||
        compressedKeys.dims[1] != headDim || rows <= 0 || heads <= 0 ||
        heads > 32 ||
        headDim <= 0 || tokenBudget <= 0 || compressRatio <= 0 ||
        tokenBudget % compressRatio != 0 || completeBlocks < selectedK ||
        (queryStart != -1 &&
         (queryStart < 0 || queryStart + rows > keyLength)) ||
        indices.dims != std::vector<int>({rows, outputWidth})) {
        return false;
    }

    float *scores = (float*)FastllmCudaMalloc(
        (size_t)rows * completeBlocks * sizeof(float));
    int32_t *selectedBlocks = (int32_t*)FastllmCudaMalloc(
        (size_t)rows * selectedK * sizeof(int32_t));
    if (scores == nullptr || selectedBlocks == nullptr) {
        FastllmCudaFree(scores);
        FastllmCudaFree(selectedBlocks);
        return false;
    }
    constexpr int threads = 256;
    constexpr int warpsPerBlock = threads / 32;
    const int scoreItemsPerBlock = heads == 4
        ? threads / 4 : (heads < 4 ? threads : warpsPerBlock);
    const int scoreBlocks = std::min<uint64_t>(
        1024,
        ((uint64_t)rows * completeBlocks + scoreItemsPerBlock - 1) /
            scoreItemsPerBlock);
    const float inverseSqrt = 1.0f / std::sqrt((float)headDim);
    if (query.dataType == fastllm::DataType::FLOAT32) {
        auto kernel = heads == 4
            ? Qwen4QSAScoreFourHeadKernel<float>
            : (heads < 4 ? Qwen4QSAScoreSerialKernel<float>
                         : Qwen4QSAScoreKernel<float>);
        kernel<<<scoreBlocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)query.cudaData,
            (const float*)compressedKeys.cudaData, scores,
            rows, completeBlocks, heads, headDim, inverseSqrt,
            queryStart, compressRatio);
    } else if (query.dataType == fastllm::DataType::FLOAT16) {
        auto kernel = heads == 4
            ? Qwen4QSAScoreFourHeadKernel<half>
            : (heads < 4 ? Qwen4QSAScoreSerialKernel<half>
                         : Qwen4QSAScoreKernel<half>);
        kernel<<<scoreBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)query.cudaData,
            (const float*)compressedKeys.cudaData, scores,
            rows, completeBlocks, heads, headDim, inverseSqrt,
            queryStart, compressRatio);
    } else {
        auto kernel = heads == 4
            ? Qwen4QSAScoreFourHeadKernel<__nv_bfloat16>
            : (heads < 4
                   ? Qwen4QSAScoreSerialKernel<__nv_bfloat16>
                   : Qwen4QSAScoreKernel<__nv_bfloat16>);
        kernel<<<scoreBlocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)query.cudaData,
            (const float*)compressedKeys.cudaData, scores,
            rows, completeBlocks, heads, headDim, inverseSqrt,
            queryStart, compressRatio);
    }
    Qwen4QSARadixSelectKernel<<<
        rows, threads, 0, cudaStreamPerThread>>>(
        scores, selectedBlocks, rows, completeBlocks, selectedK,
        queryStart, compressRatio);
    const int expandBlocks = std::min<uint64_t>(
        1024,
        ((uint64_t)rows * outputWidth + threads - 1) / threads);
    Qwen4QSAExpandIndicesKernel<<<
        expandBlocks, threads, 0, cudaStreamPerThread>>>(
        selectedBlocks, (int32_t*)indices.cudaData,
        rows, selectedK, compressRatio, completeBlocks,
        keyLength, outputWidth, queryStart);
    DeviceSync();
    const cudaError_t status = cudaGetLastError();
    FastllmCudaFree(scores);
    FastllmCudaFree(selectedBlocks);
    return status == cudaSuccess;
}

bool FastllmCudaQwen4QSABuildMask(
        const fastllm::Data &indices,
        const fastllm::Data &reference,
        fastllm::Data &mask, int keyLength) {
    const int rows = indices.dims.size() == 2 ? indices.dims[0] : 0;
    const int width = indices.dims.size() == 2 ? indices.dims[1] : 0;
    if (indices.dataDevice != fastllm::DataDevice::CUDA ||
        reference.dataDevice != fastllm::DataDevice::CUDA ||
        mask.dataDevice != fastllm::DataDevice::CUDA ||
        indices.cudaData == nullptr || reference.cudaData == nullptr ||
        mask.cudaData == nullptr ||
        indices.dataType != fastllm::DataType::INT32 ||
        !Qwen4CudaActivationType(reference.dataType) ||
        mask.dataType != reference.dataType ||
        indices.dims.size() != 2 || rows <= 0 || width <= 0 ||
        keyLength <= 0 ||
        mask.dims != std::vector<int>({1, rows, keyLength})) {
        return false;
    }
    constexpr int threads = 256;
    const uint64_t maskCount = (uint64_t)rows * keyLength;
    const int fillBlocks = std::min<uint64_t>(
        1024, (maskCount + threads - 1) / threads);
    const int scatterBlocks = std::min(
        1024, (rows * width + threads - 1) / threads);
    if (reference.dataType == fastllm::DataType::FLOAT32) {
        Qwen4QSAFillMaskKernel<<<
            fillBlocks, threads, 0, cudaStreamPerThread>>>(
            (float*)mask.cudaData, maskCount);
        Qwen4QSAScatterMaskKernel<<<
            scatterBlocks, threads, 0, cudaStreamPerThread>>>(
            (const int32_t*)indices.cudaData, (float*)mask.cudaData,
            rows, width, keyLength);
    } else if (reference.dataType == fastllm::DataType::FLOAT16) {
        Qwen4QSAFillMaskKernel<<<
            fillBlocks, threads, 0, cudaStreamPerThread>>>(
            (half*)mask.cudaData, maskCount);
        Qwen4QSAScatterMaskKernel<<<
            scatterBlocks, threads, 0, cudaStreamPerThread>>>(
            (const int32_t*)indices.cudaData, (half*)mask.cudaData,
            rows, width, keyLength);
    } else {
        Qwen4QSAFillMaskKernel<<<
            fillBlocks, threads, 0, cudaStreamPerThread>>>(
            (__nv_bfloat16*)mask.cudaData, maskCount);
        Qwen4QSAScatterMaskKernel<<<
            scatterBlocks, threads, 0, cudaStreamPerThread>>>(
            (const int32_t*)indices.cudaData,
            (__nv_bfloat16*)mask.cudaData,
            rows, width, keyLength);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4GatherKV(
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, fastllm::Data &compactKey,
        fastllm::Data &compactValue) {
    const int keyHeads = key.dims.size() == 3 ? key.dims[0] : 0;
    const int keyLength = key.dims.size() == 3 ? key.dims[1] : 0;
    const int headDim = key.dims.size() == 3 ? key.dims[2] : 0;
    const int width = indices.dims.size() == 2 ? indices.dims[1] : 0;
    const std::vector<int> compactDims = {keyHeads, width, headDim};
    if (key.dataDevice != fastllm::DataDevice::CUDA ||
        value.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        compactKey.dataDevice != fastllm::DataDevice::CUDA ||
        compactValue.dataDevice != fastllm::DataDevice::CUDA ||
        key.cudaData == nullptr || value.cudaData == nullptr ||
        indices.cudaData == nullptr || compactKey.cudaData == nullptr ||
        compactValue.cudaData == nullptr ||
        !Qwen4CudaActivationType(key.dataType) ||
        value.dataType != key.dataType ||
        compactKey.dataType != key.dataType ||
        compactValue.dataType != key.dataType ||
        indices.dataType != fastllm::DataType::INT32 ||
        key.dims.size() != 3 || value.dims != key.dims ||
        indices.dims.size() != 2 || indices.dims[0] != 1 ||
        keyHeads <= 0 || keyLength <= 0 || headDim <= 0 || width <= 0 ||
        compactKey.dims != compactDims || compactValue.dims != compactDims) {
        return false;
    }
    constexpr int threads = 256;
    const uint64_t count = (uint64_t)keyHeads * width * headDim;
    const int blocks = std::min<uint64_t>(
        1024, (count + threads - 1) / threads);
    if (key.dataType == fastllm::DataType::FLOAT32) {
        Qwen4GatherKVKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)key.cudaData, (const float*)value.cudaData,
            (const int32_t*)indices.cudaData, (float*)compactKey.cudaData,
            (float*)compactValue.cudaData, keyHeads, keyLength,
            headDim, width);
    } else if (key.dataType == fastllm::DataType::FLOAT16) {
        Qwen4GatherKVKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            (const int32_t*)indices.cudaData, (half*)compactKey.cudaData,
            (half*)compactValue.cudaData, keyHeads, keyLength,
            headDim, width);
    } else {
        Qwen4GatherKVKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)key.cudaData,
            (const __nv_bfloat16*)value.cudaData,
            (const int32_t*)indices.cudaData,
            (__nv_bfloat16*)compactKey.cudaData,
            (__nv_bfloat16*)compactValue.cudaData,
            keyHeads, keyLength, headDim, width);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4PrepareSparseBatch(
        const fastllm::Data &query, const fastllm::Data &key,
        const fastllm::Data &value, const fastllm::Data &indices,
        fastllm::Data &packedQuery, fastllm::Data &compactKey,
        fastllm::Data &compactValue, fastllm::Data &paddingMask,
        int rowStart, int rows) {
    const int queryHeads = query.dims.size() == 3 ? query.dims[0] : 0;
    const int sequence = query.dims.size() == 3 ? query.dims[1] : 0;
    const int headDim = query.dims.size() == 3 ? query.dims[2] : 0;
    const int keyHeads = key.dims.size() == 3 ? key.dims[0] : 0;
    const int keyLength = key.dims.size() == 3 ? key.dims[1] : 0;
    const int width = indices.dims.size() == 2 ? indices.dims[1] : 0;
    const std::vector<int> packedQueryDims = {
        rows * queryHeads, 1, headDim};
    const std::vector<int> compactDims = {
        rows * keyHeads, width, headDim};
    const std::vector<int> maskDims = {rows, 1, width};
    if (query.dataDevice != fastllm::DataDevice::CUDA ||
        key.dataDevice != fastllm::DataDevice::CUDA ||
        value.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        packedQuery.dataDevice != fastllm::DataDevice::CUDA ||
        compactKey.dataDevice != fastllm::DataDevice::CUDA ||
        compactValue.dataDevice != fastllm::DataDevice::CUDA ||
        paddingMask.dataDevice != fastllm::DataDevice::CUDA ||
        query.cudaData == nullptr || key.cudaData == nullptr ||
        value.cudaData == nullptr || indices.cudaData == nullptr ||
        packedQuery.cudaData == nullptr || compactKey.cudaData == nullptr ||
        compactValue.cudaData == nullptr || paddingMask.cudaData == nullptr ||
        !Qwen4CudaActivationType(query.dataType) ||
        key.dataType != query.dataType || value.dataType != query.dataType ||
        packedQuery.dataType != query.dataType ||
        compactKey.dataType != query.dataType ||
        compactValue.dataType != query.dataType ||
        paddingMask.dataType != query.dataType ||
        indices.dataType != fastllm::DataType::INT32 ||
        query.dims.size() != 3 || key.dims.size() != 3 ||
        value.dims != key.dims || indices.dims.size() != 2 ||
        indices.dims[0] != sequence || queryHeads <= 0 || keyHeads <= 0 ||
        keyLength <= 0 || headDim <= 0 || key.dims[2] != headDim ||
        width <= 0 || rowStart < 0 || rows <= 0 ||
        rowStart + rows > sequence ||
        packedQuery.dims != packedQueryDims ||
        compactKey.dims != compactDims || compactValue.dims != compactDims ||
        paddingMask.dims != maskDims) {
        return false;
    }

    constexpr int threads = 256;
    const uint64_t queryCount =
        (uint64_t)rows * queryHeads * headDim;
    const uint64_t kvCount =
        (uint64_t)rows * keyHeads * width * headDim;
    const int maskCount = rows * width;
    const int queryBlocks = std::min<uint64_t>(
        1024, (queryCount + threads - 1) / threads);
    const int kvBlocks = std::min<uint64_t>(
        1024, (kvCount + threads - 1) / threads);
    const int maskBlocks = std::min(
        1024, (maskCount + threads - 1) / threads);
    const int32_t *indexData = (const int32_t*)indices.cudaData;
    if (query.dataType == fastllm::DataType::FLOAT32) {
        Qwen4PackSparseQueryKernel<<<
            queryBlocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)query.cudaData, (float*)packedQuery.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
        Qwen4GatherSparseBatchKVKernel<<<
            kvBlocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)key.cudaData, (const float*)value.cudaData,
            indexData, (float*)compactKey.cudaData,
            (float*)compactValue.cudaData, keyHeads, keyLength, headDim,
            width, rowStart, rows);
        Qwen4SparsePaddingMaskKernel<<<
            maskBlocks, threads, 0, cudaStreamPerThread>>>(
            indexData, (float*)paddingMask.cudaData, keyLength,
            width, rowStart, rows);
    } else if (query.dataType == fastllm::DataType::FLOAT16) {
        Qwen4PackSparseQueryKernel<<<
            queryBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)query.cudaData, (half*)packedQuery.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
        Qwen4GatherSparseBatchKVKernel<<<
            kvBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            indexData, (half*)compactKey.cudaData,
            (half*)compactValue.cudaData, keyHeads, keyLength, headDim,
            width, rowStart, rows);
        Qwen4SparsePaddingMaskKernel<<<
            maskBlocks, threads, 0, cudaStreamPerThread>>>(
            indexData, (half*)paddingMask.cudaData, keyLength,
            width, rowStart, rows);
    } else {
        Qwen4PackSparseQueryKernel<<<
            queryBlocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)query.cudaData,
            (__nv_bfloat16*)packedQuery.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
        Qwen4GatherSparseBatchKVKernel<<<
            kvBlocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)key.cudaData,
            (const __nv_bfloat16*)value.cudaData, indexData,
            (__nv_bfloat16*)compactKey.cudaData,
            (__nv_bfloat16*)compactValue.cudaData,
            keyHeads, keyLength, headDim, width, rowStart, rows);
        Qwen4SparsePaddingMaskKernel<<<
            maskBlocks, threads, 0, cudaStreamPerThread>>>(
            indexData, (__nv_bfloat16*)paddingMask.cudaData,
            keyLength, width, rowStart, rows);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4UnpackSparseBatch(
        const fastllm::Data &packedOutput, fastllm::Data &output,
        int rowStart, int rows) {
    const int queryHeads = output.dims.size() == 3 ? output.dims[0] : 0;
    const int sequence = output.dims.size() == 3 ? output.dims[1] : 0;
    const int headDim = output.dims.size() == 3 ? output.dims[2] : 0;
    const std::vector<int> packedDims = {
        rows * queryHeads, 1, headDim};
    if (packedOutput.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        packedOutput.cudaData == nullptr || output.cudaData == nullptr ||
        !Qwen4CudaActivationType(output.dataType) ||
        packedOutput.dataType != output.dataType ||
        output.dims.size() != 3 || packedOutput.dims != packedDims ||
        queryHeads <= 0 || sequence <= 0 || headDim <= 0 ||
        rowStart < 0 || rows <= 0 || rowStart + rows > sequence) {
        return false;
    }
    constexpr int threads = 256;
    const uint64_t count = (uint64_t)rows * queryHeads * headDim;
    const int blocks = std::min<uint64_t>(
        1024, (count + threads - 1) / threads);
    if (output.dataType == fastllm::DataType::FLOAT32) {
        Qwen4UnpackSparseOutputKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)packedOutput.cudaData, (float*)output.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
    } else if (output.dataType == fastllm::DataType::FLOAT16) {
        Qwen4UnpackSparseOutputKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)packedOutput.cudaData, (half*)output.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
    } else {
        Qwen4UnpackSparseOutputKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)packedOutput.cudaData,
            (__nv_bfloat16*)output.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaCausalDepthwiseConv1DDecode(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &state, fastllm::Data &output,
        int kernel, bool silu, bool initializeState) {
    const int batch = input.dims.empty() ? 0 : input.dims[0];
    const int channels = input.dims.size() == 3 ? input.dims[1] : 0;
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        state.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        state.cudaData == nullptr || output.cudaData == nullptr ||
        !Qwen4CudaActivationType(input.dataType) ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        state.dataType != fastllm::DataType::FLOAT32 ||
        output.dataType != fastllm::DataType::FLOAT32 ||
        batch <= 0 || channels <= 0 || kernel <= 0 ||
        input.dims.size() != 3 || input.dims[2] != 1 ||
        weight.Count(0) != (uint64_t)channels * kernel ||
        state.dims != std::vector<int>({batch, channels, kernel}) ||
        output.dims != std::vector<int>({batch, 1, channels})) {
        return false;
    }
    if (initializeState) {
        FastllmCudaMemset0(state.cudaData, state.GetBytes());
    }
    const int total = batch * channels;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    CausalDepthwiseConv1DDecodeKernel<<<
        blocks, threads, 0, cudaStreamPerThread>>>(
        input.cudaData, (int)input.dataType, (const float*)weight.cudaData,
        (float*)state.cudaData, (float*)output.cudaData,
        total, channels, kernel, silu);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaCausalDepthwiseConv1DPrefill(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &state, fastllm::Data &output,
        int kernel, bool silu, bool initializeState) {
    const int batch = input.dims.empty() ? 0 : input.dims[0];
    const int sequence = input.dims.size() == 3 ? input.dims[1] : 0;
    const int channels = input.dims.size() == 3 ? input.dims[2] : 0;
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        state.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        state.cudaData == nullptr || output.cudaData == nullptr ||
        !Qwen4CudaActivationType(input.dataType) ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        state.dataType != fastllm::DataType::FLOAT32 ||
        output.dataType != fastllm::DataType::FLOAT32 ||
        batch <= 0 || sequence <= 0 || channels <= 0 || kernel <= 0 ||
        input.dims.size() != 3 ||
        weight.Count(0) != (uint64_t)channels * kernel ||
        state.dims != std::vector<int>({batch, channels, kernel}) ||
        output.dims != input.dims) {
        return false;
    }
    if (initializeState) {
        FastllmCudaMemset0(state.cudaData, state.GetBytes());
    }
    constexpr int threads = 256;
    const uint64_t outputCount = (uint64_t)batch * sequence * channels;
    const int outputBlocks = std::min<uint64_t>(
        65535, (outputCount + threads - 1) / threads);
    const int stateItems = batch * channels;
    const int stateBlocks = (stateItems + threads - 1) / threads;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        CausalDepthwiseConv1DPrefillKernel<<<
            outputBlocks, threads, 0, cudaStreamPerThread>>>(
                (const float*)input.cudaData,
                (const float*)weight.cudaData,
                (const float*)state.cudaData, (float*)output.cudaData,
                outputCount, sequence, channels, kernel, silu);
        CausalDepthwiseConv1DPrefillStateKernel<<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
                (const float*)input.cudaData, (float*)state.cudaData,
                stateItems, sequence, channels, kernel);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        CausalDepthwiseConv1DPrefillKernel<<<
            outputBlocks, threads, 0, cudaStreamPerThread>>>(
                (const half*)input.cudaData,
                (const float*)weight.cudaData,
                (const float*)state.cudaData, (float*)output.cudaData,
                outputCount, sequence, channels, kernel, silu);
        CausalDepthwiseConv1DPrefillStateKernel<<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
                (const half*)input.cudaData, (float*)state.cudaData,
                stateItems, sequence, channels, kernel);
    } else {
        CausalDepthwiseConv1DPrefillKernel<<<
            outputBlocks, threads, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16*)input.cudaData,
                (const float*)weight.cudaData,
                (const float*)state.cudaData, (float*)output.cudaData,
                outputCount, sequence, channels, kernel, silu);
        CausalDepthwiseConv1DPrefillStateKernel<<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16*)input.cudaData,
                (float*)state.cudaData,
                stateItems, sequence, channels, kernel);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4GatedDeltaRuleDecode(
        const fastllm::Data &qkv, const fastllm::Data &alpha,
        const fastllm::Data &beta,
        const fastllm::Data &aLog, const fastllm::Data &dtBias,
        fastllm::Data &state, fastllm::Data &output,
        int keyHeads, int valueHeads, int keyDim, int valueDim,
        float recurrentEps) {
    const int batch = qkv.dims.empty() ? 0 : qkv.dims[0];
    const int qkvChannels = 2 * keyHeads * keyDim +
                            valueHeads * valueDim;
    if (qkv.dataDevice != fastllm::DataDevice::CUDA ||
        alpha.dataDevice != fastllm::DataDevice::CUDA ||
        beta.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        state.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        qkv.cudaData == nullptr ||
        alpha.cudaData == nullptr || beta.cudaData == nullptr ||
        aLog.cudaData == nullptr || dtBias.cudaData == nullptr ||
        state.cudaData == nullptr ||
        output.cudaData == nullptr || qkv.dataType != fastllm::DataType::FLOAT32 ||
        state.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        !Qwen4CudaActivationType(alpha.dataType) ||
        beta.dataType != alpha.dataType ||
        output.dataType != fastllm::DataType::FLOAT32 || keyHeads <= 0 ||
        valueHeads <= 0 || valueHeads % keyHeads != 0 ||
        keyDim != 128 || valueDim != 128 || qkv.dims.size() != 3 ||
        qkv.dims[1] != 1 || qkv.dims[2] != qkvChannels ||
        alpha.Count(0) != (uint64_t)batch * valueHeads ||
        beta.Count(0) != (uint64_t)batch * valueHeads ||
        aLog.Count(0) != (uint64_t)valueHeads ||
        dtBias.Count(0) != (uint64_t)valueHeads ||
        state.dims != std::vector<int>({batch, valueHeads, keyDim, valueDim})) {
        return false;
    }

    const int blocks = batch * valueHeads;
    // RunLinearAttention constructs this value on the host and then uses it
    // as both the RMSNorm weight and the subsequent query scale.  Passing the
    // same host result avoids the one-ULP difference of device rsqrtf().
    const float inverseHead = 1.0f / std::sqrt((float)keyDim);
    if (alpha.dataType == fastllm::DataType::FLOAT32) {
        Qwen4GatedDeltaRuleDecodeKernel<<<
            blocks, 128, 0, cudaStreamPerThread>>>(
            (const float*)qkv.cudaData, (const float*)alpha.cudaData,
            (const float*)beta.cudaData,
            (const float*)aLog.cudaData, (const float*)dtBias.cudaData,
            (float*)state.cudaData, (float*)output.cudaData,
            keyHeads, valueHeads, recurrentEps, inverseHead);
    } else if (alpha.dataType == fastllm::DataType::FLOAT16) {
        Qwen4GatedDeltaRuleDecodeKernel<<<
            blocks, 128, 0, cudaStreamPerThread>>>(
            (const float*)qkv.cudaData, (const half*)alpha.cudaData,
            (const half*)beta.cudaData,
            (const float*)aLog.cudaData, (const float*)dtBias.cudaData,
            (float*)state.cudaData, (float*)output.cudaData,
            keyHeads, valueHeads, recurrentEps, inverseHead);
    } else {
        Qwen4GatedDeltaRuleDecodeKernel<<<
            blocks, 128, 0, cudaStreamPerThread>>>(
            (const float*)qkv.cudaData,
            (const __nv_bfloat16*)alpha.cudaData,
            (const __nv_bfloat16*)beta.cudaData,
            (const float*)aLog.cudaData, (const float*)dtBias.cudaData,
            (float*)state.cudaData, (float*)output.cudaData,
            keyHeads, valueHeads, recurrentEps, inverseHead);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}
