#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
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

    // Match the public ToDataType conversion kernels. In particular,
    // FastllmCudaFloat2HalfKernel uses round-toward-zero, which is observable
    // when the converted activation feeds the recurrent GDN.
    template <typename T>
    __device__ __forceinline__ T Qwen4CudaCastFromFloat(float value);

    template <>
    __device__ __forceinline__ float Qwen4CudaCastFromFloat<float>(
            float value) {
        return value;
    }

    template <>
    __device__ __forceinline__ half Qwen4CudaCastFromFloat<half>(float value) {
        return __float2half_rz(value);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16
    Qwen4CudaCastFromFloat<__nv_bfloat16>(float value) {
        return __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void Qwen4CudaStoreCast(
            void *data, int type, uint64_t index, float value) {
        if (type == (int)fastllm::DataType::FLOAT32) {
            ((float*)data)[index] = value;
        } else if (type == (int)fastllm::DataType::FLOAT16) {
            ((half*)data)[index] =
                Qwen4CudaCastFromFloat<half>(value);
        } else {
            ((__nv_bfloat16*)data)[index] =
                Qwen4CudaCastFromFloat<__nv_bfloat16>(value);
        }
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

    __global__ void Qwen4HyperMixFourGroupFloatKernel(
            const float *normalized, const float *mixLogits, float *output,
            uint64_t count, int outputChannels) {
        constexpr int groups = 4;
        const int group = threadIdx.x & (groups - 1);
        const int itemInBlock = threadIdx.x / groups;
        const int itemsPerBlock = blockDim.x / groups;
        const uint64_t outputIndex =
            (uint64_t)blockIdx.x * itemsPerBlock + itemInBlock;
        float inputValue = 0.0f;
        float gate = 0.0f;
        if (outputIndex < count) {
            const uint64_t row = outputIndex / outputChannels;
            const int channel = (int)(outputIndex % outputChannels);
            const uint64_t inputIndex =
                row * groups * outputChannels +
                (uint64_t)group * outputChannels + channel;
            inputValue = normalized[inputIndex];
            gate = Qwen4CudaSigmoidRounded<float>(mixLogits[inputIndex]);
        }

        // Evaluate the expensive sigmoid for all four groups in parallel.
        // The first lane performs the products and additions in the original
        // expression order, allowing the compiler to retain the reference
        // kernel's multiply-add contraction instead of introducing a new
        // float rounding boundary before the shuffle.
        const unsigned int mask = __activemask();
        const float input0 = __shfl_sync(mask, inputValue, 0, groups);
        const float input1 = __shfl_sync(mask, inputValue, 1, groups);
        const float input2 = __shfl_sync(mask, inputValue, 2, groups);
        const float input3 = __shfl_sync(mask, inputValue, 3, groups);
        const float gate0 = __shfl_sync(mask, gate, 0, groups);
        const float gate1 = __shfl_sync(mask, gate, 1, groups);
        const float gate2 = __shfl_sync(mask, gate, 2, groups);
        const float gate3 = __shfl_sync(mask, gate, 3, groups);
        if (group == 0 && outputIndex < count) {
            // The generic float kernel contracts groups 1..3 into the
            // running sum. Spell that sequence explicitly so nvcc cannot
            // swap which group's product receives the standalone round.
            float sum = __fmul_rn(input0, gate0);
            sum = __fmaf_rn(input1, gate1, sum);
            sum = __fmaf_rn(input2, gate2, sum);
            sum = __fmaf_rn(input3, gate3, sum);
            output[outputIndex] = sum / groups;
        }
    }

    template <typename T, typename LogitT>
    void Qwen4LaunchHyperMix(const void *normalized, const void *mixLogits,
                             void *output, int blocks, int threads,
                             uint64_t count, int groups,
                             int outputChannels) {
        if constexpr (std::is_same_v<T, float> &&
                      std::is_same_v<LogitT, float>) {
            if (groups == 4) {
                const int itemsPerBlock = threads / groups;
                const int fourGroupBlocks = (int)(
                    (count + itemsPerBlock - 1) / itemsPerBlock);
                Qwen4HyperMixFourGroupFloatKernel<<<
                    fourGroupBlocks, threads, 0, cudaStreamPerThread>>>(
                    (const float*)normalized, (const float*)mixLogits,
                    (float*)output, count, outputChannels);
                return;
            }
        }
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

    template <typename T>
    __device__ __forceinline__ T Qwen4HyperActivatedValue(
            T input, int groups) {
        const T scale = Qwen4CudaFromFloat<T>(1.0f / groups);
        const float scaled = Qwen4CudaRound<T>(
            Qwen4CudaToFloat(input) * Qwen4CudaToFloat(scale));
        if constexpr (std::is_same_v<T, float>) {
            return scaled / (1.0 + expf(-scaled));
        } else {
            return Qwen4CudaFromFloat<T>(
                scaled / (1.0f + expf(-scaled)));
        }
    }

    template <>
    __device__ __forceinline__ half Qwen4HyperActivatedValue<half>(
            half input, int groups) {
#ifdef CUDA_NO_TENSOR_CORE
        const half scale = __float2half_rn(1.0f / groups);
        const half rounded = __float2half_rn(
            __half2float(input) * __half2float(scale));
        const float value = __half2float(rounded);
        return __float2half_rn(value / (1.0 + expf(-value)));
#else
        const half scaled = __hmul(input, __float2half_rn(1.0f / groups));
        return __hdiv(
            scaled, __hadd(__float2half_rn(1.0f), hexp(-scaled)));
#endif
    }

    template <typename T>
    __global__ void Qwen4HyperPrepareKernel(
            const T *input, T *activated,
            uint64_t count, int groups) {
        const uint64_t index = (uint64_t)blockIdx.x * blockDim.x +
                               threadIdx.x;
        if (index >= count) {
            return;
        }
        activated[index] = Qwen4HyperActivatedValue(input[index], groups);
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
    __device__ __forceinline__ T Qwen4HyperCombinedValue(
            const T *hyperInput, const void *blockOutput, int blockType,
            const void *injection, int injectionType,
            uint64_t hyperIndex, uint64_t blockIndex,
            uint64_t injectionIndex) {
        const float blockValue = Qwen4CudaRound<T>(
            Qwen4CudaLoadActivation(blockOutput, blockType, blockIndex));
        const float injectionScale = Qwen4CudaRound<T>(
            Qwen4CudaLoadActivation(
                injection, injectionType, injectionIndex));
        const float injected = Qwen4CudaRound<T>(
            blockValue * injectionScale);
        return Qwen4CudaFromFloat<T>(Qwen4CudaRound<T>(
            Qwen4CudaToFloat(hyperInput[hyperIndex]) + injected));
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
        output[index] = Qwen4HyperCombinedValue(
            hyperInput, blockOutput, blockType, injection, injectionType,
            index, row * blockChannels + channel, row * groups + group);
    }

    template <typename T, int THREADS>
    __global__ void Qwen4HyperCombineRMSNormKernel(
            const T *hyperInput, const void *blockOutput, int blockType,
            const void *injection, int injectionType,
            const float *weight, T *residual, T *normalized,
            int rows, int groups, int groupChannels, float eps) {
        const int item = blockIdx.x;
        if (item >= rows * groups) {
            return;
        }
        const int group = item % groups;
        const int row = item / groups;
        const int channels = groups * groupChannels;
        const uint64_t hyperBase = (uint64_t)row * channels +
                                   (uint64_t)group * groupChannels;
        const uint64_t blockBase = (uint64_t)row * groupChannels;
        const uint64_t injectionIndex = (uint64_t)row * groups + group;
        T *groupResidual = residual + hyperBase;
        T *groupNormalized = normalized + hyperBase;
        const float *groupWeight = weight +
                                   (uint64_t)group * groupChannels;

        // Keep the same packed channel-to-thread mapping and reduction tree
        // as Qwen4GroupedRMSNormKernel.  The residual is rounded before it
        // contributes to the square sum, matching the unfused operation pair.
        float sum = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            if ((groupChannels & 3) == 0) {
                const int packedChannels = groupChannels / 4;
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const int channel = index * 4;
                    float4 value;
                    value.x = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    value.y = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel + 1,
                        blockBase + channel + 1, injectionIndex);
                    value.z = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel + 2,
                        blockBase + channel + 2, injectionIndex);
                    value.w = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel + 3,
                        blockBase + channel + 3, injectionIndex);
                    reinterpret_cast<float4*>(groupResidual)[index] = value;
                    sum += value.x * value.x + value.y * value.y +
                           value.z * value.z + value.w * value.w;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const float value = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    groupResidual[channel] = value;
                    sum += value * value;
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const int channel = index * 2;
                    const half low = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    const half high = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel + 1,
                        blockBase + channel + 1, injectionIndex);
                    reinterpret_cast<half2*>(groupResidual)[index] =
                        __halves2half2(low, high);
                    const float lowValue = __half2float(low);
                    const float highValue = __half2float(high);
                    sum += lowValue * lowValue + highValue * highValue;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const half value = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    groupResidual[channel] = value;
                    const float floatValue = __half2float(value);
                    sum += floatValue * floatValue;
                }
            }
        } else {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const int channel = index * 2;
                    __nv_bfloat162 value;
                    value.x = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    value.y = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel + 1,
                        blockBase + channel + 1, injectionIndex);
                    reinterpret_cast<__nv_bfloat162*>(
                        groupResidual)[index] = value;
                    const float lowValue = __bfloat162float(value.x);
                    const float highValue = __bfloat162float(value.y);
                    sum += lowValue * lowValue + highValue * highValue;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    const __nv_bfloat16 value = Qwen4HyperCombinedValue(
                        hyperInput, blockOutput, blockType, injection,
                        injectionType, hyperBase + channel,
                        blockBase + channel, injectionIndex);
                    groupResidual[channel] = value;
                    const float floatValue = __bfloat162float(value);
                    sum += floatValue * floatValue;
                }
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        __shared__ float warpSums[THREADS / 32];
        __shared__ float normScale;
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
                normScale = rsqrtf(total / groupChannels + eps);
            }
        }
        __syncthreads();

        if constexpr (std::is_same_v<T, float>) {
            if ((groupChannels & 3) == 0) {
                const int packedChannels = groupChannels / 4;
                const float4 *packedResidual =
                    reinterpret_cast<const float4*>(groupResidual);
                float4 *packedNormalized =
                    reinterpret_cast<float4*>(groupNormalized);
                const float4 *packedWeight =
                    reinterpret_cast<const float4*>(groupWeight);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float4 value = packedResidual[index];
                    const float4 currentWeight = packedWeight[index];
                    packedNormalized[index] = {
                        value.x * normScale * currentWeight.x,
                        value.y * normScale * currentWeight.y,
                        value.z * normScale * currentWeight.z,
                        value.w * normScale * currentWeight.w};
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupNormalized[channel] = groupResidual[channel] *
                        normScale * groupWeight[channel];
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const half2 *packedResidual =
                    reinterpret_cast<const half2*>(groupResidual);
                half2 *packedNormalized =
                    reinterpret_cast<half2*>(groupNormalized);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const float2 value =
                        __half22float2(packedResidual[index]);
                    packedNormalized[index] = __floats2half2_rn(
                        value.x * normScale * groupWeight[index * 2],
                        value.y * normScale * groupWeight[index * 2 + 1]);
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupNormalized[channel] = __float2half_rn(
                        __half2float(groupResidual[channel]) * normScale *
                        groupWeight[channel]);
                }
            }
        } else {
            if ((groupChannels & 1) == 0) {
                const int packedChannels = groupChannels / 2;
                const __nv_bfloat162 *packedResidual =
                    reinterpret_cast<const __nv_bfloat162*>(groupResidual);
                __nv_bfloat162 *packedNormalized =
                    reinterpret_cast<__nv_bfloat162*>(groupNormalized);
                for (int index = threadIdx.x; index < packedChannels;
                     index += THREADS) {
                    const __nv_bfloat162 value = packedResidual[index];
                    __nv_bfloat162 result;
                    result.x = __float2bfloat16_rn(
                        __bfloat162float(value.x) * normScale *
                        groupWeight[index * 2]);
                    result.y = __float2bfloat16_rn(
                        __bfloat162float(value.y) * normScale *
                        groupWeight[index * 2 + 1]);
                    packedNormalized[index] = result;
                }
            } else {
                for (int channel = threadIdx.x; channel < groupChannels;
                     channel += THREADS) {
                    groupNormalized[channel] = __float2bfloat16_rn(
                        __bfloat162float(groupResidual[channel]) *
                        normScale * groupWeight[channel]);
                }
            }
        }
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

    template <typename T, typename OutputT>
    __global__ void CausalDepthwiseConv1DPrefillKernel(
            const T *input, const float *weight, const float *state,
            OutputT *output, uint64_t total, int sequence, int channels,
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
            output[item] = Qwen4CudaCastFromFloat<OutputT>(value);
        }
    }

    template <typename T>
    __global__ void CausalDepthwiseConv1DPrefillStateKernel(
            const T *input, float *state, int total, int sequence,
            int channels, int kernel) {
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

    template <typename InputT, typename OutputT>
    void Qwen4LaunchCausalDepthwiseConv1DPrefill(
            const fastllm::Data &input, const fastllm::Data &weight,
            fastllm::Data &state, fastllm::Data &output,
            uint64_t outputCount, int outputBlocks,
            int stateItems, int stateBlocks,
            int sequence, int channels, int kernel, bool silu) {
        constexpr int threads = 256;
        CausalDepthwiseConv1DPrefillKernel<InputT, OutputT><<<
            outputBlocks, threads, 0, cudaStreamPerThread>>>(
                (const InputT*)input.cudaData,
                (const float*)weight.cudaData,
                (const float*)state.cudaData, (OutputT*)output.cudaData,
                outputCount, sequence, channels, kernel, silu);
        CausalDepthwiseConv1DPrefillStateKernel<<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
                (const InputT*)input.cudaData, (float*)state.cudaData,
                stateItems, sequence, channels, kernel);
    }

    template <typename InputT>
    void Qwen4DispatchCausalDepthwiseConv1DPrefillOutput(
            const fastllm::Data &input, const fastllm::Data &weight,
            fastllm::Data &state, fastllm::Data &output,
            uint64_t outputCount, int outputBlocks,
            int stateItems, int stateBlocks,
            int sequence, int channels, int kernel, bool silu) {
        if (output.dataType == fastllm::DataType::FLOAT32) {
            Qwen4LaunchCausalDepthwiseConv1DPrefill<InputT, float>(
                input, weight, state, output, outputCount, outputBlocks,
                stateItems, stateBlocks, sequence, channels, kernel, silu);
        } else if (output.dataType == fastllm::DataType::FLOAT16) {
            Qwen4LaunchCausalDepthwiseConv1DPrefill<InputT, half>(
                input, weight, state, output, outputCount, outputBlocks,
                stateItems, stateBlocks, sequence, channels, kernel, silu);
        } else {
            Qwen4LaunchCausalDepthwiseConv1DPrefill<
                InputT, __nv_bfloat16>(
                    input, weight, state, output, outputCount,
                    outputBlocks, stateItems, stateBlocks,
                    sequence, channels, kernel, silu);
        }
    }

    template <typename T, bool OUT_OF_PLACE_STATE>
    __global__ __launch_bounds__(128) void Qwen4GatedDeltaRuleDecodeKernel(
            const float *qkv, const T *alpha, const T *beta,
            const float *aLog, const float *dtBias, float *state,
            float *stateOutput, float *output,
            int keyHeads, int valueHeads,
            int sequence, float recurrentEps, float inverseHead) {
        constexpr int KEY_DIM = 128;
        constexpr int VALUE_DIM = 128;
        const int item = blockIdx.x;
        const int batchIndex = item / valueHeads;
        const int valueHead = item - batchIndex * valueHeads;
        const int repeat = valueHeads / keyHeads;
        const int keyHead = valueHead / repeat;
        const int qkvChannels = (2 * keyHeads + valueHeads) * KEY_DIM;
        float *headState = state +
            (uint64_t)item * KEY_DIM * VALUE_DIM;
        float *headNextState = headState;
        if constexpr (OUT_OF_PLACE_STATE) {
            headNextState = stateOutput +
                (uint64_t)item * KEY_DIM * VALUE_DIM;
        }

        __shared__ float queryNorm[KEY_DIM];
        __shared__ float keyNorm[KEY_DIM];
        __shared__ float normScales[2];
        __shared__ float gateValues[2];

        const int tid = threadIdx.x;
        for (int token = 0; token < sequence; token++) {
            const uint64_t row =
                (uint64_t)batchIndex * sequence + token;
            const uint64_t qkvBase = row * qkvChannels;
            const float *queryRaw =
                qkv + qkvBase + keyHead * KEY_DIM;
            const float *keyRaw = qkv + qkvBase +
                (keyHeads + keyHead) * KEY_DIM;
            const float *value = qkv + qkvBase +
                (2 * keyHeads + valueHead) * KEY_DIM;

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
                const uint64_t gateIndex = row * valueHeads + valueHead;
                const float alphaValue =
                    Qwen4CudaToFloat(alpha[gateIndex]);
                const float biasedAlpha =
                    alphaValue + dtBias[valueHead];
                const float softplus = biasedAlpha > 20.0f
                    ? biasedAlpha : log1p(expf(biasedAlpha));
                const float betaValue =
                    Qwen4CudaToFloat(beta[gateIndex]);
                // Preserve the established single-token expressions exactly.
                gateValues[0] = 1.0 / (1.0 + expf(-betaValue));
                gateValues[1] = expf(
                    -expf((double)aLog[valueHead]) * softplus);
            }
            __syncthreads();

            // Preserve the float32 write/read boundary between normalization
            // and the additional query scale used by the decode operation.
            queryNorm[tid] =
                queryRaw[tid] * normScales[0] * inverseHead;
            keyNorm[tid] = keyRaw[tid] * normScales[1] * inverseHead;
            __syncthreads();
            queryNorm[tid] *= inverseHead;
            __syncthreads();

            float memory = 0.0f;
            const float *headPreviousState = headState;
            if constexpr (OUT_OF_PLACE_STATE) {
                if (token > 0) {
                    headPreviousState = headNextState;
                }
            }
#pragma unroll
            for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
                const uint64_t stateIndex =
                    (uint64_t)keyChannel * VALUE_DIM + tid;
                const float previous = headPreviousState[stateIndex];
                const float scaled = previous * gateValues[1];
                memory += scaled * keyNorm[keyChannel];
            }
            const float delta =
                (value[tid] - memory) * gateValues[0];
            float core = 0.0f;
#pragma unroll
            for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
                const uint64_t stateIndex =
                    (uint64_t)keyChannel * VALUE_DIM + tid;
                // Recompute the rounded float32 decay product instead of
                // writing it to global memory in the first pass and loading
                // it here.  Every thread owns one value column, so the state
                // remains unchanged until this pass without introducing a
                // dependency.  This preserves the established arithmetic
                // while reducing recurrent-state traffic from four to three
                // float words per element.
                const float scaled =
                    headPreviousState[stateIndex] * gateValues[1];
                const float updated = scaled +
                                      keyNorm[keyChannel] * delta;
                headNextState[stateIndex] = updated;
                core += updated * queryNorm[keyChannel];
            }
            output[(row * valueHeads + valueHead) * VALUE_DIM + tid] =
                core;
            __syncthreads();
        }
    }

    // Exact SM120 short-sequence mapping.  A warp owns 32 adjacent value
    // channels, so each lane retains the generic kernel's strictly increasing
    // K reduction order.  Four independent blocks per value head expose the
    // same 128 lanes to four times as many SMs while caching each state tile
    // across all speculative tokens.
    template <typename T, bool OUT_OF_PLACE_STATE>
    __global__ __launch_bounds__(32)
    void Qwen4GatedDeltaRuleSequenceValueTileSm120Kernel(
            const float *qkv, const T *alpha, const T *beta,
            const float *aLog, const float *dtBias, float *state,
            float *stateOutput, float *output,
            int keyHeads, int valueHeads,
            int sequence, float recurrentEps, float inverseHead) {
        constexpr int KEY_DIM = 128;
        constexpr int VALUE_DIM = 128;
        constexpr int VALUE_TILE = 32;
        constexpr int TILES = VALUE_DIM / VALUE_TILE;

        const int item = blockIdx.x / TILES;
        const int tile = blockIdx.x - item * TILES;
        const int batchIndex = item / valueHeads;
        const int valueHead = item - batchIndex * valueHeads;
        const int repeat = valueHeads / keyHeads;
        const int keyHead = valueHead / repeat;
        const int qkvChannels = (2 * keyHeads + valueHeads) * KEY_DIM;
        const int lane = threadIdx.x;
        const int valueChannel = tile * VALUE_TILE + lane;
        float *headState = state +
            (uint64_t)item * KEY_DIM * VALUE_DIM;
        float *headNextState = headState;
        if constexpr (OUT_OF_PLACE_STATE) {
            headNextState = stateOutput +
                (uint64_t)item * KEY_DIM * VALUE_DIM;
        }

        __shared__ float stateTile[KEY_DIM * VALUE_TILE];
        __shared__ float queryNorm[KEY_DIM];
        __shared__ float keyNorm[KEY_DIM];
        __shared__ float normScales[2];
        __shared__ float gateValues[2];

#pragma unroll
        for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
            const uint64_t stateIndex =
                (uint64_t)keyChannel * VALUE_DIM + valueChannel;
            const float previous = headState[stateIndex];
            stateTile[keyChannel * VALUE_TILE + lane] = previous;
        }
        __syncwarp();

        for (int token = 0; token < sequence; token++) {
            const uint64_t row =
                (uint64_t)batchIndex * sequence + token;
            const uint64_t qkvBase = row * qkvChannels;
            const float *queryRaw =
                qkv + qkvBase + keyHead * KEY_DIM;
            const float *keyRaw = qkv + qkvBase +
                (keyHeads + keyHead) * KEY_DIM;
            const float *value = qkv + qkvBase +
                (2 * keyHeads + valueHead) * KEY_DIM;

            const float4 queryValue =
                reinterpret_cast<const float4 *>(queryRaw)[lane];
            const float4 keyValue =
                reinterpret_cast<const float4 *>(keyRaw)[lane];
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
            if (lane == 0) {
                normScales[0] = rsqrtf(
                    querySquares / KEY_DIM + recurrentEps);
                normScales[1] = rsqrtf(
                    keySquares / KEY_DIM + recurrentEps);
                const uint64_t gateIndex = row * valueHeads + valueHead;
                const float alphaValue =
                    Qwen4CudaToFloat(alpha[gateIndex]);
                const float biasedAlpha =
                    alphaValue + dtBias[valueHead];
                const float softplus = biasedAlpha > 20.0f
                    ? biasedAlpha : log1p(expf(biasedAlpha));
                const float betaValue =
                    Qwen4CudaToFloat(beta[gateIndex]);
                gateValues[0] = 1.0 / (1.0 + expf(-betaValue));
                gateValues[1] = expf(
                    -expf((double)aLog[valueHead]) * softplus);
            }
            __syncwarp();

#pragma unroll
            for (int part = 0; part < 4; part++) {
                const int channel = lane + part * 32;
                queryNorm[channel] =
                    queryRaw[channel] * normScales[0] * inverseHead;
                keyNorm[channel] =
                    keyRaw[channel] * normScales[1] * inverseHead;
            }
            __syncwarp();
#pragma unroll
            for (int part = 0; part < 4; part++) {
                const int channel = lane + part * 32;
                queryNorm[channel] *= inverseHead;
            }
            __syncwarp();

            float memory = 0.0f;
#pragma unroll
            for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
                const float scaled =
                    stateTile[keyChannel * VALUE_TILE + lane] *
                    gateValues[1];
                memory += scaled * keyNorm[keyChannel];
            }
            const float delta =
                (value[valueChannel] - memory) * gateValues[0];
            float core = 0.0f;
#pragma unroll
            for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
                const int stateIndex =
                    keyChannel * VALUE_TILE + lane;
                const float scaled =
                    stateTile[stateIndex] * gateValues[1];
                const float updated = scaled +
                    keyNorm[keyChannel] * delta;
                stateTile[stateIndex] = updated;
                core += updated * queryNorm[keyChannel];
            }
            output[(row * valueHeads + valueHead) * VALUE_DIM +
                   valueChannel] = core;
            __syncwarp();
        }

#pragma unroll
        for (int keyChannel = 0; keyChannel < KEY_DIM; keyChannel++) {
            headNextState[
                (uint64_t)keyChannel * VALUE_DIM + valueChannel] =
                stateTile[keyChannel * VALUE_TILE + lane];
        }
    }

    // Reproduce the verifier's materialized operation chain exactly:
    //   FP32 core -> FP16(rz) -> RMSNorm(FP16, 128) -> Sigmoid(FP16) -> Mul(FP16).
    // One warp evaluates the same two legacy warp reductions independently
    // and combines them in the same order as FastllmRMSNormHalf128ExactKernel.
    __global__ __launch_bounds__(32) void Qwen4GdnOutputGateExactKernel(
            const float *input, const float *weight, const half *gate,
            half *output, float eps) {
        constexpr int CHANNELS = 128;
        const int row = blockIdx.x;
        const int lane = threadIdx.x;
        input += (uint64_t)row * CHANNELS;
        gate += (uint64_t)row * CHANNELS;
        output += (uint64_t)row * CHANNELS;

        const float2 raw0 = *reinterpret_cast<const float2 *>(
            input + lane * 2);
        const float2 raw1 = *reinterpret_cast<const float2 *>(
            input + (lane + 32) * 2);
        const half2 rounded0 = __halves2half2(
            __float2half_rz(raw0.x), __float2half_rz(raw0.y));
        const half2 rounded1 = __halves2half2(
            __float2half_rz(raw1.x), __float2half_rz(raw1.y));
        const float2 value0 = __half22float2(rounded0);
        const float2 value1 = __half22float2(rounded1);
        float sum0 = value0.x * value0.x + value0.y * value0.y;
        float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
            sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        }
        float scale = 0.0f;
        if (lane == 0) {
            scale = rsqrtf((sum0 + sum1) / CHANNELS + eps);
        }
        scale = __shfl_sync(0xffffffffu, scale, 0);

        const half2 gate0 = reinterpret_cast<const half2 *>(gate)[lane];
        const half2 gate1 =
            reinterpret_cast<const half2 *>(gate)[lane + 32];
        half2 *output2 = reinterpret_cast<half2 *>(output);
        const float2 values[2] = {value0, value1};
        const half2 gates[2] = {gate0, gate1};
#pragma unroll
        for (int part = 0; part < 2; part++) {
            const int index = lane + part * 32;
            const half rms0 = __float2half_rn(
                values[part].x * scale * __ldg(weight + index * 2));
            const half rms1 = __float2half_rn(
                values[part].y * scale * __ldg(weight + index * 2 + 1));
            const half gateInput0 = __low2half(gates[part]);
            const half gateInput1 = __high2half(gates[part]);
#ifdef CUDA_NO_TENSOR_CORE
            const half sigmoid0 = __float2half_rn(
                1.0f / (1.0f + expf(-__half2float(gateInput0))));
            const half sigmoid1 = __float2half_rn(
                1.0f / (1.0f + expf(-__half2float(gateInput1))));
            const half result0 = __float2half_rn(
                __half2float(sigmoid0) * __half2float(rms0));
            const half result1 = __float2half_rn(
                __half2float(sigmoid1) * __half2float(rms1));
#else
            const half one = __float2half_rn(1.0f);
            const half sigmoid0 = __hdiv(
                one, __hadd(one, hexp(-gateInput0)));
            const half sigmoid1 = __hdiv(
                one, __hadd(one, hexp(-gateInput1)));
            const half result0 = __hmul(rms0, sigmoid0);
            const half result1 = __hmul(rms1, sigmoid1);
#endif
            output2[index] = __halves2half2(result0, result1);
        }
    }

    struct Qwen4LinearReplayPointers {
        const void *convInput;
        const float *convWeight;
        const float *convCheckpoint;
        float *convState;
        float *convolved;
        const void *alpha;
        const void *beta;
        const float *aLog;
        const float *dtBias;
        const float *recurrentCheckpoint;
        float *recurrentState;
        float *coreOutput;
    };

    static_assert(sizeof(Qwen4LinearReplayPointers) == 12 * sizeof(void*),
                  "Qwen4 replay pointer table must not contain padding");

    __global__ void Qwen4LinearReplayRestoreKernel(
            const Qwen4LinearReplayPointers *layers,
            uint64_t totalVectors, int vectorsPerLayer,
            int convVectors) {
        for (uint64_t item =
                 (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
             item < totalVectors;
             item += (uint64_t)blockDim.x * gridDim.x) {
            const int layer = item / vectorsPerLayer;
            const int local = item - (uint64_t)layer * vectorsPerLayer;
            const Qwen4LinearReplayPointers &pointers = layers[layer];
            if (local < convVectors) {
                ((float4*)pointers.convState)[local] =
                    ((const float4*)pointers.convCheckpoint)[local];
            } else {
                const int recurrentLocal = local - convVectors;
                ((float4*)pointers.recurrentState)[recurrentLocal] =
                    ((const float4*)pointers.recurrentCheckpoint)[
                        recurrentLocal];
            }
        }
    }

    template <typename T>
    __global__ void Qwen4LinearReplayConvOutputKernel(
            const Qwen4LinearReplayPointers *layers,
            uint64_t total, int sequence, int channels,
            int kernel, bool silu) {
        for (uint64_t item =
                 (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
             item < total;
             item += (uint64_t)blockDim.x * gridDim.x) {
            const uint64_t layerStride =
                (uint64_t)sequence * channels;
            const int layer = item / layerStride;
            const uint64_t local = item - (uint64_t)layer * layerStride;
            const int token = local / channels;
            const int channel = local - (uint64_t)token * channels;
            const Qwen4LinearReplayPointers &pointers = layers[layer];
            const T *input = (const T*)pointers.convInput;
            const float *weightRow = pointers.convWeight +
                (uint64_t)channel * kernel;
            const float *stateRow = pointers.convState +
                (uint64_t)channel * kernel;
            float value = 0.0f;
            for (int tap = 0; tap < kernel; tap++) {
                const int sourceToken = token - kernel + 1 + tap;
                const float sample = sourceToken >= 0
                    ? Qwen4CudaToFloat(input[
                          (uint64_t)sourceToken * channels + channel])
                    : stateRow[kernel + sourceToken];
                value += sample * weightRow[tap];
            }
            if (silu) {
                value = value / (1.0 + expf(-value));
            }
            pointers.convolved[local] = value;
        }
    }

    template <typename T>
    __global__ void Qwen4LinearReplayConvStateKernel(
            const Qwen4LinearReplayPointers *layers,
            int total, int sequence, int channels, int kernel) {
        const int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= total) {
            return;
        }
        const int layer = item / channels;
        const int channel = item - layer * channels;
        const Qwen4LinearReplayPointers &pointers = layers[layer];
        const T *input = (const T*)pointers.convInput;
        float *stateRow = pointers.convState +
            (uint64_t)channel * kernel;
        if (sequence >= kernel) {
            for (int slot = 0; slot < kernel; slot++) {
                stateRow[slot] = Qwen4CudaToFloat(input[
                    ((uint64_t)sequence - kernel + slot) * channels +
                        channel]);
            }
        } else {
            for (int slot = 0; slot < kernel - sequence; slot++) {
                stateRow[slot] = stateRow[slot + sequence];
            }
            for (int token = 0; token < sequence; token++) {
                stateRow[kernel - sequence + token] =
                    Qwen4CudaToFloat(input[
                        (uint64_t)token * channels + channel]);
            }
        }
    }

    template <typename T>
    __global__ __launch_bounds__(128)
    void Qwen4LinearReplayGatedDeltaStateKernel(
            const Qwen4LinearReplayPointers *layers,
            int valueHeads, int keyHeads, int sequence,
            float recurrentEps, float inverseHead) {
        constexpr int KEY_DIM = 128;
        constexpr int VALUE_DIM = 128;
        const int item = blockIdx.x;
        const int layer = item / valueHeads;
        const int valueHead = item - layer * valueHeads;
        const int repeat = valueHeads / keyHeads;
        const int keyHead = valueHead / repeat;
        const int qkvChannels =
            (2 * keyHeads + valueHeads) * KEY_DIM;
        const Qwen4LinearReplayPointers &pointers = layers[layer];
        const float *qkv = pointers.convolved;
        const T *alpha = (const T*)pointers.alpha;
        const T *beta = (const T*)pointers.beta;
        float *headState = pointers.recurrentState +
            (uint64_t)valueHead * KEY_DIM * VALUE_DIM;
        float *output = pointers.coreOutput;

        __shared__ float queryNorm[KEY_DIM];
        __shared__ float keyNorm[KEY_DIM];
        __shared__ float normScales[2];
        __shared__ float gateValues[2];
        const int tid = threadIdx.x;

        for (int token = 0; token < sequence; token++) {
            const uint64_t qkvBase = (uint64_t)token * qkvChannels;
            const float *queryRaw = qkv + qkvBase +
                keyHead * KEY_DIM;
            const float *keyRaw = qkv + qkvBase +
                (keyHeads + keyHead) * KEY_DIM;
            const float *value = qkv + qkvBase +
                (2 * keyHeads + valueHead) * KEY_DIM;
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
                const uint64_t gateIndex =
                    (uint64_t)token * valueHeads + valueHead;
                const float alphaValue =
                    Qwen4CudaToFloat(alpha[gateIndex]);
                const float biasedAlpha =
                    alphaValue + pointers.dtBias[valueHead];
                const float softplus = biasedAlpha > 20.0f
                    ? biasedAlpha : log1p(expf(biasedAlpha));
                const float betaValue =
                    Qwen4CudaToFloat(beta[gateIndex]);
                gateValues[0] = 1.0 / (1.0 + expf(-betaValue));
                gateValues[1] = expf(
                    -expf((double)pointers.aLog[valueHead]) * softplus);
            }
            __syncthreads();

            queryNorm[tid] =
                queryRaw[tid] * normScales[0] * inverseHead;
            keyNorm[tid] =
                keyRaw[tid] * normScales[1] * inverseHead;
            __syncthreads();
            queryNorm[tid] *= inverseHead;
            __syncthreads();

            float memory = 0.0f;
#pragma unroll
            for (int keyChannel = 0;
                 keyChannel < KEY_DIM; keyChannel++) {
                const uint64_t stateIndex =
                    (uint64_t)keyChannel * VALUE_DIM + tid;
                const float scaled =
                    headState[stateIndex] * gateValues[1];
                memory += scaled * keyNorm[keyChannel];
            }
            const float delta =
                (value[tid] - memory) * gateValues[0];
            float core = 0.0f;
#pragma unroll
            for (int keyChannel = 0;
                 keyChannel < KEY_DIM; keyChannel++) {
                const uint64_t stateIndex =
                    (uint64_t)keyChannel * VALUE_DIM + tid;
                const float scaled =
                    headState[stateIndex] * gateValues[1];
                const float updated = scaled +
                    keyNorm[keyChannel] * delta;
                headState[stateIndex] = updated;
                core += updated * queryNorm[keyChannel];
            }
            output[((uint64_t)token * valueHeads + valueHead) *
                       VALUE_DIM + tid] = core;
            __syncthreads();
        }
    }

    template <typename T>
    __global__ void Qwen4QSAScoreKernel(
            const T *query, const float *compressedKeys, float *scores,
            int rows, int blocks, int heads, int headDim,
            float inverseSqrt, int queryStart, int compressRatio,
            const int32_t *decodeMeta) {
        const int lane = threadIdx.x & 31;
        const int warpInBlock = threadIdx.x >> 5;
        const int warpsPerBlock = blockDim.x >> 5;
        const uint64_t total = (uint64_t)rows * blocks;
        const int dynamicQueryStart =
            queryStart == -2 && decodeMeta != nullptr
                ? decodeMeta[0] : queryStart;
        const int actualBlocks = decodeMeta == nullptr
            ? blocks : (decodeMeta[0] + 1) / compressRatio;
        for (uint64_t item =
                 (uint64_t)blockIdx.x * warpsPerBlock + warpInBlock;
             item < total;
             item += (uint64_t)warpsPerBlock * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = dynamicQueryStart >= 0
                ? (dynamicQueryStart + row + 1) / compressRatio
                : actualBlocks;
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
            float inverseSqrt, int queryStart, int compressRatio,
            const int32_t *decodeMeta) {
        const uint64_t total = (uint64_t)rows * blocks;
        const int dynamicQueryStart =
            queryStart == -2 && decodeMeta != nullptr
                ? decodeMeta[0] : queryStart;
        const int actualBlocks = decodeMeta == nullptr
            ? blocks : (decodeMeta[0] + 1) / compressRatio;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = dynamicQueryStart >= 0
                ? (dynamicQueryStart + row + 1) / compressRatio
                : actualBlocks;
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
            float inverseSqrt, int queryStart, int compressRatio,
            const int32_t *decodeMeta) {
        constexpr int lanesPerItem = 4;
        const int lane = threadIdx.x & (lanesPerItem - 1);
        const int itemInBlock = threadIdx.x / lanesPerItem;
        const int itemsPerBlock = blockDim.x / lanesPerItem;
        const uint64_t total = (uint64_t)rows * blocks;
        const int dynamicQueryStart =
            queryStart == -2 && decodeMeta != nullptr
                ? decodeMeta[0] : queryStart;
        const int actualBlocks = decodeMeta == nullptr
            ? blocks : (decodeMeta[0] + 1) / compressRatio;
        for (uint64_t item =
                 (uint64_t)blockIdx.x * itemsPerBlock + itemInBlock;
             item < total;
             item += (uint64_t)itemsPerBlock * gridDim.x) {
            const int row = item / blocks;
            const int block = item - (uint64_t)row * blocks;
            const int rowBlocks = dynamicQueryStart >= 0
                ? (dynamicQueryStart + row + 1) / compressRatio
                : actualBlocks;
            if (block >= rowBlocks) {
                continue;
            }
            const float *blockKey = compressedKeys +
                                    (uint64_t)block * headDim;
            float dot = 0.0f;
            if (lane < heads) {
                const T *headQuery = query +
                    ((uint64_t)row * heads + lane) * headDim;
                if (headDim == 128) {
#pragma unroll
                    for (int column = 0; column < 128; column++) {
                        dot += Qwen4CudaToFloat(headQuery[column]) *
                               blockKey[column];
                    }
                } else {
                    for (int column = 0; column < headDim; column++) {
                        dot += Qwen4CudaToFloat(headQuery[column]) *
                               blockKey[column];
                    }
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
            int queryStart, int compressRatio,
            const int32_t *decodeMeta) {
        const int rowIndex = blockIdx.x;
        if (rowIndex >= rows) {
            return;
        }
        const float *row = scores + (uint64_t)rowIndex * blocks;
        int32_t *output = selectedBlocks +
                          (uint64_t)rowIndex * selectedK;
        const int dynamicQueryStart =
            queryStart == -2 && decodeMeta != nullptr
                ? decodeMeta[0] : queryStart;
        const int rowBlocks = dynamicQueryStart >= 0
            ? (dynamicQueryStart + rowIndex + 1) / compressRatio
            : (decodeMeta == nullptr
                ? blocks : (decodeMeta[0] + 1) / compressRatio);
        if (rowBlocks <= selectedK) {
            for (int block = threadIdx.x; block < selectedK;
                 block += blockDim.x) {
                output[block] = block < rowBlocks ? block : -1;
            }
            return;
        }

        constexpr int sortItemsPerThread = 5;
        constexpr int sortCapacity = 256 * sortItemsPerThread;
        using BlockRadixSort = cub::BlockRadixSort<
            uint32_t, 256, sortItemsPerThread>;
        __shared__ typename BlockRadixSort::TempStorage sortStorage;
        __shared__ uint32_t histogram[256];
        __shared__ uint32_t prefix;
        __shared__ uint32_t remaining;
        __shared__ uint32_t threadHigher[256];
        __shared__ uint32_t threadPivot[256];
        __shared__ uint32_t threadPivotChosen[256];
        __shared__ uint32_t threadOutputOffset[256];
        if (rowBlocks <= sortCapacity) {
            uint32_t keys[sortItemsPerThread];
#pragma unroll
            for (int item = 0; item < sortItemsPerThread; item++) {
                const int block = threadIdx.x * sortItemsPerThread + item;
                keys[item] = block < rowBlocks
                    ? Qwen4QSAOrderedFloatBits(row[block]) : 0u;
            }
            BlockRadixSort(sortStorage).SortDescending(keys);
            const int pivotRank = selectedK - 1;
            if (threadIdx.x == pivotRank / sortItemsPerThread) {
                prefix = keys[pivotRank % sortItemsPerThread];
            }
            __syncthreads();
        } else {
            if (threadIdx.x == 0) {
                prefix = 0;
                remaining = selectedK;
            }
            __syncthreads();
#pragma unroll
            for (int round = 0; round < 4; round++) {
                histogram[threadIdx.x] = 0;
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
                        atomicAdd(
                            &histogram[(ordered >> shift) & 0xffu], 1u);
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
            int queryStart, const int32_t *decodeMeta) {
        const int dynamicQueryStart =
            queryStart == -2 && decodeMeta != nullptr
                ? decodeMeta[0] : queryStart;
        if (decodeMeta != nullptr) {
            keyLength = decodeMeta[0] + 1;
            completeBlocks = keyLength / compressRatio;
        }
        const uint64_t total = (uint64_t)rows * outputWidth;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int row = item / outputWidth;
            const int column = item - (uint64_t)row * outputWidth;
            const int visibleLength = dynamicQueryStart >= 0
                ? dynamicQueryStart + row + 1 : keyLength;
            const int rowBlocks = dynamicQueryStart >= 0
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
            int keyLength, int keyHeadStride, int valueHeadStride,
            int headDim, int width, const int32_t *decodeMeta) {
        if (decodeMeta != nullptr) {
            keyLength = decodeMeta[0] + 1;
        }
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
                    (uint64_t)sourceToken * headDim + column;
                compactKey[item] = key[
                    (uint64_t)head * keyHeadStride + source];
                compactValue[item] = value[
                    (uint64_t)head * valueHeadStride + source];
            } else {
                compactKey[item] = Qwen4CudaFromFloat<T>(0.0f);
                compactValue[item] = Qwen4CudaFromFloat<T>(0.0f);
            }
        }
    }

    __global__ void Qwen4QSAAppendGraphKernel(
            const float *rawKey, const float *position,
            const int32_t *decodeMeta, float *tailKeys,
            float *tailPositions, int headDim, int tokenOffset,
            int compressRatio) {
        const int slot = (decodeMeta[0] + tokenOffset) % compressRatio;
        for (int column = threadIdx.x; column < headDim;
             column += blockDim.x) {
            tailKeys[(uint64_t)slot * headDim + column] =
                rawKey[column];
        }
        if (threadIdx.x == 0) {
            tailPositions[slot] = position[0];
        }
    }

    __global__ void Qwen4QSACommitGraphKernel(
            const float *compressedKey, const int32_t *decodeMeta,
            float *compressedKeys, int headDim, int compressRatio,
            int tokenOffset, int compressedCapacity) {
        const int keyLength = decodeMeta[0] + tokenOffset + 1;
        if (keyLength % compressRatio != 0) {
            return;
        }
        const int block = keyLength / compressRatio - 1;
        if (block < 0 || block >= compressedCapacity) {
            return;
        }
        for (int column = threadIdx.x; column < headDim;
             column += blockDim.x) {
            compressedKeys[(uint64_t)block * headDim + column] =
                compressedKey[column];
        }
    }

    // Qwen3.8-Flash verifies four speculative rows at once.  The established
    // path appends each row to a four-row tail, materializes five Split
    // outputs, adds the tail rows in order, and then runs float32 RMSNorm and
    // RoPE before conditionally committing the completed block.  Keep that
    // exact arithmetic and reduction mapping in one graph-safe kernel.
    __global__ __launch_bounds__(64)
    void Qwen4QSAAppendCompress4ExactKernel(
            const float *rawKeys, const float *positions,
            const float *normWeight, const float *sin, const float *cos,
            const int32_t *decodeMeta, int previousLength,
            float *tailKeys,
            float *tailPositions, float *compressedKeys,
            int compressedCapacity, int sinCosStride, float eps) {
        constexpr int kSequence = 4;
        constexpr int kHeadDim = 128;
        constexpr int kRotaryPart = 64;
        constexpr int kThreads = 64;
        const int tid = threadIdx.x;
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int baseLength = decodeMeta == nullptr
            ? previousLength : decodeMeta[0];
        const int oldTail = baseLength & (kSequence - 1);
        const int commitToken = kSequence - oldTail - 1;

        __shared__ __align__(16) float averaged[kHeadDim];
        __shared__ __align__(16) float normalized[kHeadDim];
        __shared__ float warpSums[2];
        __shared__ float scale;
        __shared__ int ropeIndex;
        __shared__ int commitBlock;

        // Reconstruct the tail exactly as it appears immediately after the
        // row that closes this compression group.  Slots before oldTail are
        // the persistent old tail; the rest come from the new token rows.
        for (int column = tid; column < kHeadDim;
             column += kThreads) {
            float pooled = oldTail > 0
                ? tailKeys[column]
                : rawKeys[column];
#pragma unroll
            for (int slot = 1; slot < kSequence; slot++) {
                const float member = slot < oldTail
                    ? tailKeys[(uint64_t)slot * kHeadDim + column]
                    : rawKeys[(uint64_t)(slot - oldTail) * kHeadDim +
                              column];
                pooled += member * 1.0f;
            }
            averaged[column] = pooled * (1.0f / (float)kSequence);
        }
        if (tid == 0) {
            const float firstPosition = oldTail > 0
                ? tailPositions[0] : positions[0];
            ropeIndex = (int)firstPosition;
            commitBlock =
                (baseLength + commitToken + 1) / kSequence - 1;
        }
        __syncthreads();

        // Match FastllmRMSNormKernelInner1<64>(float, channels=128):
        // lanes 0..31 each reduce one float4, the second warp contributes 0,
        // and warp 0 performs the same two-warp final reduction.
        float sum2 = 0.0f;
        if (tid < kHeadDim / 4) {
            const float4 value =
                reinterpret_cast<const float4 *>(averaged)[tid];
            sum2 += value.x * value.x + value.y * value.y +
                    value.z * value.z + value.w * value.w;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum2;
        }
        __syncthreads();
        if (warp == 0) {
            float value = lane < 2 ? warpSums[lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(
                    0xffffffffu, value, offset);
            }
            if (lane == 0) {
                scale = rsqrtf(value / kHeadDim + eps);
            }
        }
        __syncthreads();

        if (tid < kHeadDim / 4) {
            const float4 value =
                reinterpret_cast<const float4 *>(averaged)[tid];
            const float4 weight =
                reinterpret_cast<const float4 *>(normWeight)[tid];
            float4 output;
            output.x = value.x * scale * weight.x;
            output.y = value.y * scale * weight.y;
            output.z = value.z * scale * weight.z;
            output.w = value.w * scale * weight.w;
            reinterpret_cast<float4 *>(normalized)[tid] = output;
        }
        __syncthreads();

        const int block = commitBlock;
        if (block >= 0 && block < compressedCapacity) {
            if (tid < kRotaryPart / 2) {
                const float currentSin =
                    sin[(uint64_t)ropeIndex * sinCosStride + tid];
                const float currentCos =
                    cos[(uint64_t)ropeIndex * sinCosStride + tid];
                const float low = normalized[tid];
                const float high = normalized[
                    tid + kRotaryPart / 2];
                compressedKeys[(uint64_t)block * kHeadDim + tid] =
                    low * currentCos - high * currentSin;
                compressedKeys[(uint64_t)block * kHeadDim + tid +
                               kRotaryPart / 2] =
                    low * currentSin + high * currentCos;
            }
            compressedKeys[(uint64_t)block * kHeadDim +
                           kRotaryPart + tid] =
                normalized[kRotaryPart + tid];
        }
        __syncthreads();

        // Four new rows overwrite every tail slot exactly once.  Delay this
        // final state update until the completed block has consumed the old
        // prefix slots above.
        for (int item = tid; item < kSequence * kHeadDim;
             item += kThreads) {
            const int token = item / kHeadDim;
            const int column = item - token * kHeadDim;
            const int slot = (oldTail + token) & (kSequence - 1);
            tailKeys[(uint64_t)slot * kHeadDim + column] =
                rawKeys[(uint64_t)token * kHeadDim + column];
        }
        if (tid < kSequence) {
            const int slot = (oldTail + tid) & (kSequence - 1);
            tailPositions[slot] = positions[tid];
        }
    }

    template <typename T>
    __global__ void Qwen4KVAppendKernel(
            const T *key, const T *value, const int32_t *decodeMeta,
            T *keyCache, T *valueCache, int heads, int headDim,
            int sequence, int previousLength,
            int keyInputHeadStride, int valueInputHeadStride,
            int keyCacheHeadStride, int valueCacheHeadStride,
            int capacity) {
        const int tokenBase = decodeMeta == nullptr
            ? previousLength : decodeMeta[0];
        if (tokenBase < 0 || tokenBase + sequence > capacity) {
            return;
        }
        const int count = heads * sequence * headDim;
        for (int item = blockIdx.x * blockDim.x + threadIdx.x;
             item < count; item += blockDim.x * gridDim.x) {
            const int column = item % headDim;
            const int token = (item / headDim) % sequence;
            const int head = item / (sequence * headDim);
            keyCache[(uint64_t)head * keyCacheHeadStride +
                     (uint64_t)(tokenBase + token) * headDim + column] =
                key[(uint64_t)head * keyInputHeadStride +
                    (uint64_t)token * headDim + column];
            valueCache[(uint64_t)head * valueCacheHeadStride +
                       (uint64_t)(tokenBase + token) * headDim + column] =
                value[(uint64_t)head * valueInputHeadStride +
                      (uint64_t)token * headDim + column];
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
            T *compactKey, T *compactValue, T *paddingMask, int keyHeads,
            int keyLength, int keyHeadStride, int valueHeadStride,
            int headDim, int width,
            int rowStart, int rows, const int32_t *decodeMeta,
            int appendedSequence) {
        if (decodeMeta != nullptr) {
            keyLength = decodeMeta[0] + appendedSequence;
        }
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
            const bool valid = sourceToken >= 0 && sourceToken < keyLength;
            if (valid) {
                const uint64_t source =
                    (uint64_t)sourceToken * headDim + column;
                compactKey[item] = key[
                    (uint64_t)head * keyHeadStride + source];
                compactValue[item] = value[
                    (uint64_t)head * valueHeadStride + source];
            } else {
                compactKey[item] = Qwen4CudaFromFloat<T>(0.0f);
                compactValue[item] = Qwen4CudaFromFloat<T>(0.0f);
            }
            if (column == 0 && head == 0) {
                paddingMask[(uint64_t)row * width + selected] =
                    Qwen4CudaFromFloat<T>(valid ? 0.0f : 1.0f);
            }
        }
    }

    __global__ void Qwen4GatherSparseBatchKVHalf8Kernel(
            const half *key, const half *value, const int32_t *indices,
            half *compactKey, half *compactValue, half *paddingMask,
            int keyHeads, int keyLength, int keyHeadStride,
            int valueHeadStride, int headDim, int width,
            int rowStart, int rows, const int32_t *decodeMeta,
            int appendedSequence) {
        if (decodeMeta != nullptr) {
            keyLength = decodeMeta[0] + appendedSequence;
        }
        constexpr int valuesPerVector = sizeof(uint4) / sizeof(half);
        const int vectorsPerToken = headDim / valuesPerVector;
        const uint64_t total =
            (uint64_t)rows * keyHeads * width * vectorsPerToken;
        for (uint64_t item = (uint64_t)blockIdx.x * blockDim.x +
                             threadIdx.x;
             item < total; item += (uint64_t)blockDim.x * gridDim.x) {
            const int vectorColumn = item % vectorsPerToken;
            const uint64_t tokenVector = item / vectorsPerToken;
            const int selected = tokenVector % width;
            const int head = (tokenVector / width) % keyHeads;
            const int row = tokenVector / ((uint64_t)width * keyHeads);
            const int sourceToken = indices[
                (uint64_t)(rowStart + row) * width + selected];
            const bool valid = sourceToken >= 0 && sourceToken < keyLength;
            uint4 keyVector = make_uint4(0, 0, 0, 0);
            uint4 valueVector = make_uint4(0, 0, 0, 0);
            if (valid) {
                const uint64_t source =
                    (uint64_t)sourceToken * headDim +
                    (uint64_t)vectorColumn * valuesPerVector;
                keyVector = *reinterpret_cast<const uint4 *>(
                    key + (uint64_t)head * keyHeadStride + source);
                valueVector = *reinterpret_cast<const uint4 *>(
                    value + (uint64_t)head * valueHeadStride + source);
            }
            reinterpret_cast<uint4 *>(compactKey)[item] = keyVector;
            reinterpret_cast<uint4 *>(compactValue)[item] = valueVector;
            if (vectorColumn == 0 && head == 0) {
                paddingMask[(uint64_t)row * width + selected] =
                    __float2half_rn(valid ? 0.0f : 1.0f);
            }
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

    template <typename T, bool OUT_OF_PLACE_STATE>
    void Qwen4LaunchGatedDeltaRule(
            const fastllm::Data &qkv, const fastllm::Data &alpha,
            const fastllm::Data &beta, const fastllm::Data &aLog,
            const fastllm::Data &dtBias, fastllm::Data &state,
            fastllm::Data *stateOutput, fastllm::Data &output,
            int blocks, int keyHeads, int valueHeads, int sequence,
            float recurrentEps, float inverseHead,
            bool useValueTileSm120) {
        float *nextState = OUT_OF_PLACE_STATE
            ? (float*)stateOutput->cudaData : nullptr;
        if (useValueTileSm120) {
            Qwen4GatedDeltaRuleSequenceValueTileSm120Kernel<
                T, OUT_OF_PLACE_STATE><<<
                    blocks * 4, 32, 0, cudaStreamPerThread>>>(
                (const float*)qkv.cudaData, (const T*)alpha.cudaData,
                (const T*)beta.cudaData,
                (const float*)aLog.cudaData,
                (const float*)dtBias.cudaData,
                (float*)state.cudaData, nextState,
                (float*)output.cudaData, keyHeads, valueHeads,
                sequence, recurrentEps, inverseHead);
        } else {
            Qwen4GatedDeltaRuleDecodeKernel<T, OUT_OF_PLACE_STATE><<<
                blocks, 128, 0, cudaStreamPerThread>>>(
                (const float*)qkv.cudaData, (const T*)alpha.cudaData,
                (const T*)beta.cudaData,
                (const float*)aLog.cudaData,
                (const float*)dtBias.cudaData,
                (float*)state.cudaData, nextState,
                (float*)output.cudaData, keyHeads, valueHeads,
                sequence, recurrentEps, inverseHead);
        }
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

bool FastllmCudaQwen4HyperPrepare(
        const fastllm::Data &input,
        fastllm::Data &activated, int groups) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        activated.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || activated.cudaData == nullptr ||
        input.dims.empty() || groups <= 0 ||
        input.dataType != activated.dataType ||
        !Qwen4CudaActivationType(input.dataType)) {
        return false;
    }
    const uint64_t count = input.Count(0);
    if (activated.Count(0) != count) {
        return false;
    }
    constexpr int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        Qwen4HyperPrepareKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)input.cudaData, (float*)activated.cudaData,
            count, groups);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Qwen4HyperPrepareKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)input.cudaData, (half*)activated.cudaData,
            count, groups);
    } else {
        Qwen4HyperPrepareKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)input.cudaData,
            (__nv_bfloat16*)activated.cudaData,
            count, groups);
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

bool FastllmCudaQwen4HyperCombineRMSNorm(
        const fastllm::Data &hyperInput,
        const fastllm::Data &blockOutput,
        const fastllm::Data &injection,
        const fastllm::Data &normWeight,
        fastllm::Data &residual,
        fastllm::Data &normalized,
        float eps, int groups) {
    if (hyperInput.dataDevice != fastllm::DataDevice::CUDA ||
        blockOutput.dataDevice != fastllm::DataDevice::CUDA ||
        injection.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        residual.dataDevice != fastllm::DataDevice::CUDA ||
        normalized.dataDevice != fastllm::DataDevice::CUDA ||
        hyperInput.cudaData == nullptr || blockOutput.cudaData == nullptr ||
        injection.cudaData == nullptr || normWeight.cudaData == nullptr ||
        residual.cudaData == nullptr || normalized.cudaData == nullptr ||
        hyperInput.dims.empty() || groups <= 0 ||
        hyperInput.dims.back() % groups != 0 ||
        hyperInput.dataType != residual.dataType ||
        hyperInput.dataType != normalized.dataType ||
        residual.dims != hyperInput.dims ||
        normalized.dims != hyperInput.dims ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        !Qwen4CudaActivationType(hyperInput.dataType) ||
        !Qwen4CudaActivationType(blockOutput.dataType) ||
        !Qwen4CudaActivationType(injection.dataType)) {
        return false;
    }
    const int groupChannels = hyperInput.dims.back() / groups;
    const int rows = (int)(hyperInput.Count(0) / hyperInput.dims.back());
    if (blockOutput.Count(0) != (uint64_t)rows * groupChannels ||
        injection.Count(0) != (uint64_t)rows * groups ||
        normWeight.Count(0) != (uint64_t)hyperInput.dims.back()) {
        return false;
    }

    constexpr int threads = 512;
    const int blocks = rows * groups;
    if (hyperInput.dataType == fastllm::DataType::FLOAT32) {
        Qwen4HyperCombineRMSNormKernel<float, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (const float*)normWeight.cudaData,
            (float*)residual.cudaData, (float*)normalized.cudaData,
            rows, groups, groupChannels, eps);
    } else if (hyperInput.dataType == fastllm::DataType::FLOAT16) {
        Qwen4HyperCombineRMSNormKernel<half, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (const float*)normWeight.cudaData,
            (half*)residual.cudaData, (half*)normalized.cudaData,
            rows, groups, groupChannels, eps);
    } else {
        Qwen4HyperCombineRMSNormKernel<__nv_bfloat16, threads><<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)hyperInput.cudaData,
            blockOutput.cudaData, (int)blockOutput.dataType,
            injection.cudaData, (int)injection.dataType,
            (const float*)normWeight.cudaData,
            (__nv_bfloat16*)residual.cudaData,
            (__nv_bfloat16*)normalized.cudaData,
            rows, groups, groupChannels, eps);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

static bool FastllmCudaQwen4QSASelectLaunch(
        const fastllm::Data &query,
        const fastllm::Data &compressedKeys,
        const int32_t *decodeMeta,
        float *scores, int32_t *selectedBlocks,
        fastllm::Data &indices,
        int rows, int scoreCapacity, int selectedK, int outputWidth,
        int heads, int headDim,
        int compressRatio, int queryStart, int keyLength) {
    constexpr int threads = 256;
    constexpr int warpsPerBlock = threads / 32;
    const int scoreItemsPerBlock = heads == 4
        ? threads / 4 : (heads < 4 ? threads : warpsPerBlock);
    const int scoreBlocks = std::min<uint64_t>(
        1024,
        ((uint64_t)rows * scoreCapacity + scoreItemsPerBlock - 1) /
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
            rows, scoreCapacity, heads, headDim, inverseSqrt,
            queryStart, compressRatio, decodeMeta);
    } else if (query.dataType == fastllm::DataType::FLOAT16) {
        auto kernel = heads == 4
            ? Qwen4QSAScoreFourHeadKernel<half>
            : (heads < 4 ? Qwen4QSAScoreSerialKernel<half>
                         : Qwen4QSAScoreKernel<half>);
        kernel<<<scoreBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)query.cudaData,
            (const float*)compressedKeys.cudaData, scores,
            rows, scoreCapacity, heads, headDim, inverseSqrt,
            queryStart, compressRatio, decodeMeta);
    } else {
        auto kernel = heads == 4
            ? Qwen4QSAScoreFourHeadKernel<__nv_bfloat16>
            : (heads < 4
                   ? Qwen4QSAScoreSerialKernel<__nv_bfloat16>
                   : Qwen4QSAScoreKernel<__nv_bfloat16>);
        kernel<<<scoreBlocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)query.cudaData,
            (const float*)compressedKeys.cudaData, scores,
            rows, scoreCapacity, heads, headDim, inverseSqrt,
            queryStart, compressRatio, decodeMeta);
    }
    Qwen4QSARadixSelectKernel<<<
        rows, threads, 0, cudaStreamPerThread>>>(
        scores, selectedBlocks, rows, scoreCapacity, selectedK,
        queryStart, compressRatio, decodeMeta);
    const int expandBlocks = std::min<uint64_t>(
        1024,
        ((uint64_t)rows * outputWidth + threads - 1) / threads);
    Qwen4QSAExpandIndicesKernel<<<
        expandBlocks, threads, 0, cudaStreamPerThread>>>(
        selectedBlocks, (int32_t*)indices.cudaData,
        rows, selectedK, compressRatio, scoreCapacity,
        keyLength, outputWidth, queryStart, decodeMeta);
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
    const bool success = FastllmCudaQwen4QSASelectLaunch(
        query, compressedKeys, nullptr, scores, selectedBlocks, indices,
        rows, completeBlocks, selectedK, outputWidth,
        heads, headDim, compressRatio, queryStart, keyLength);
    FastllmCudaFree(scores);
    FastllmCudaFree(selectedBlocks);
    return success;
}

bool FastllmCudaQwen4QSAAppendGraph(
        const fastllm::Data &rawKey,
        const fastllm::Data &position,
        const int32_t *decodeMeta, int tokenOffset,
        int compressRatio,
        fastllm::Data &tailKeys,
        fastllm::Data &tailPositions) {
    const int headDim = tailKeys.dims.size() == 2
        ? tailKeys.dims[1] : 0;
    const int tailCapacity = tailKeys.expansionDims.size() == 2
        ? tailKeys.expansionDims[0]
        : (tailKeys.dims.size() == 2 ? tailKeys.dims[0] : 0);
    const int positionCapacity = tailPositions.expansionDims.size() == 1
        ? tailPositions.expansionDims[0]
        : (tailPositions.dims.size() == 1 ? tailPositions.dims[0] : 0);
    if (decodeMeta == nullptr || tokenOffset < 0 || compressRatio <= 0 ||
        rawKey.dataDevice != fastllm::DataDevice::CUDA ||
        position.dataDevice != fastllm::DataDevice::CUDA ||
        tailKeys.dataDevice != fastllm::DataDevice::CUDA ||
        tailPositions.dataDevice != fastllm::DataDevice::CUDA ||
        rawKey.cudaData == nullptr || position.cudaData == nullptr ||
        tailKeys.cudaData == nullptr || tailPositions.cudaData == nullptr ||
        rawKey.dataType != fastllm::DataType::FLOAT32 ||
        position.dataType != fastllm::DataType::FLOAT32 ||
        tailKeys.dataType != fastllm::DataType::FLOAT32 ||
        tailPositions.dataType != fastllm::DataType::FLOAT32 ||
        headDim <= 0 || rawKey.Count(0) != (uint64_t)headDim ||
        position.Count(0) < 1 || tailCapacity < compressRatio ||
        positionCapacity < compressRatio || tailKeys.strides.size() != 2 ||
        tailKeys.strides[0] != (uint64_t)headDim) {
        return false;
    }
    constexpr int threads = 128;
    Qwen4QSAAppendGraphKernel<<<
        1, threads, 0, cudaStreamPerThread>>>(
        (const float*)rawKey.cudaData,
        (const float*)position.cudaData, decodeMeta,
        (float*)tailKeys.cudaData,
        (float*)tailPositions.cudaData, headDim, tokenOffset,
        compressRatio);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4QSACommitGraph(
        const fastllm::Data &compressedKey,
        const int32_t *decodeMeta, int tokenOffset,
        int compressRatio,
        fastllm::Data &compressedKeys) {
    const int headDim = compressedKeys.dims.size() == 2
        ? compressedKeys.dims[1] : 0;
    const int capacity = compressedKeys.expansionDims.size() == 2
        ? compressedKeys.expansionDims[0]
        : (compressedKeys.dims.size() == 2
            ? compressedKeys.dims[0] : 0);
    if (decodeMeta == nullptr || tokenOffset < 0 || compressRatio <= 0 ||
        compressedKey.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKeys.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKey.cudaData == nullptr ||
        compressedKeys.cudaData == nullptr ||
        compressedKey.dataType != fastllm::DataType::FLOAT32 ||
        compressedKeys.dataType != fastllm::DataType::FLOAT32 ||
        headDim <= 0 || capacity <= 0 ||
        compressedKey.Count(0) != (uint64_t)headDim ||
        compressedKeys.strides.size() != 2 ||
        compressedKeys.strides[0] != (uint64_t)headDim) {
        return false;
    }
    constexpr int threads = 128;
    Qwen4QSACommitGraphKernel<<<
        1, threads, 0, cudaStreamPerThread>>>(
        (const float*)compressedKey.cudaData, decodeMeta,
        (float*)compressedKeys.cudaData, headDim,
        compressRatio, tokenOffset, capacity);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

static bool FastllmCudaQwen4QSAAppendCompress4Launch(
        const fastllm::Data &rawKeys,
        const fastllm::Data &positions,
        const fastllm::Data &normWeight,
        const fastllm::Data &sinData,
        const fastllm::Data &cosData,
        const int32_t *decodeMeta, int previousLength,
        fastllm::Data &tailKeys,
        fastllm::Data &tailPositions,
        fastllm::Data &compressedKeys,
        float eps) {
    constexpr int sequence = 4;
    constexpr int headDim = 128;
    const int tailCapacity = tailKeys.expansionDims.size() == 2
        ? tailKeys.expansionDims[0]
        : (tailKeys.dims.size() == 2 ? tailKeys.dims[0] : 0);
    const int positionCapacity = tailPositions.expansionDims.size() == 1
        ? tailPositions.expansionDims[0]
        : (tailPositions.dims.size() == 1
            ? tailPositions.dims[0] : 0);
    const int compressedCapacity =
        compressedKeys.expansionDims.size() == 2
            ? compressedKeys.expansionDims[0]
            : (compressedKeys.dims.size() == 2
                ? compressedKeys.dims[0] : 0);
    if ((decodeMeta == nullptr && previousLength < 0) ||
        rawKeys.dataDevice != fastllm::DataDevice::CUDA ||
        positions.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        sinData.dataDevice != fastllm::DataDevice::CUDA ||
        cosData.dataDevice != fastllm::DataDevice::CUDA ||
        tailKeys.dataDevice != fastllm::DataDevice::CUDA ||
        tailPositions.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKeys.dataDevice != fastllm::DataDevice::CUDA ||
        rawKeys.cudaData == nullptr || positions.cudaData == nullptr ||
        normWeight.cudaData == nullptr || sinData.cudaData == nullptr ||
        cosData.cudaData == nullptr || tailKeys.cudaData == nullptr ||
        tailPositions.cudaData == nullptr ||
        compressedKeys.cudaData == nullptr ||
        rawKeys.dataType != fastllm::DataType::FLOAT32 ||
        positions.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        sinData.dataType != fastllm::DataType::FLOAT32 ||
        cosData.dataType != fastllm::DataType::FLOAT32 ||
        tailKeys.dataType != fastllm::DataType::FLOAT32 ||
        tailPositions.dataType != fastllm::DataType::FLOAT32 ||
        compressedKeys.dataType != fastllm::DataType::FLOAT32 ||
        rawKeys.dims != std::vector<int>({sequence, headDim}) ||
        positions.Count(0) < sequence ||
        normWeight.Count(0) != headDim ||
        sinData.dims.size() != 2 || cosData.dims != sinData.dims ||
        sinData.dims[1] < 32 || tailCapacity < sequence ||
        positionCapacity < sequence || compressedCapacity <= 0 ||
        (decodeMeta == nullptr &&
         previousLength / sequence + 1 > compressedCapacity) ||
        tailKeys.strides.size() != 2 ||
        tailKeys.strides[0] != headDim ||
        compressedKeys.strides.size() != 2 ||
        compressedKeys.strides[0] != headDim) {
        return false;
    }
    Qwen4QSAAppendCompress4ExactKernel<<<
        1, 64, 0, cudaStreamPerThread>>>(
        (const float *)rawKeys.cudaData,
        (const float *)positions.cudaData,
        (const float *)normWeight.cudaData,
        (const float *)sinData.cudaData,
        (const float *)cosData.cudaData,
        decodeMeta, previousLength,
        (float *)tailKeys.cudaData,
        (float *)tailPositions.cudaData,
        (float *)compressedKeys.cudaData,
        compressedCapacity, sinData.dims[1], eps);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4QSAAppendCompress4(
        const fastllm::Data &rawKeys,
        const fastllm::Data &positions,
        const fastllm::Data &normWeight,
        const fastllm::Data &sinData,
        const fastllm::Data &cosData,
        int previousLength,
        fastllm::Data &tailKeys,
        fastllm::Data &tailPositions,
        fastllm::Data &compressedKeys,
        float eps) {
    return FastllmCudaQwen4QSAAppendCompress4Launch(
        rawKeys, positions, normWeight, sinData, cosData,
        nullptr, previousLength, tailKeys, tailPositions,
        compressedKeys, eps);
}

bool FastllmCudaQwen4QSAAppendCompress4Graph(
        const fastllm::Data &rawKeys,
        const fastllm::Data &positions,
        const fastllm::Data &normWeight,
        const fastllm::Data &sinData,
        const fastllm::Data &cosData,
        const int32_t *decodeMeta,
        fastllm::Data &tailKeys,
        fastllm::Data &tailPositions,
        fastllm::Data &compressedKeys,
        float eps) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaQwen4QSAAppendCompress4Launch(
        rawKeys, positions, normWeight, sinData, cosData,
        decodeMeta, 0, tailKeys, tailPositions,
        compressedKeys, eps);
}

bool FastllmCudaQwen4QSASelectGraph(
        const fastllm::Data &query,
        const fastllm::Data &compressedKeys,
        const int32_t *decodeMeta, fastllm::Data &scores,
        fastllm::Data &selectedBlocks, fastllm::Data &indices,
        int heads, int headDim, int tokenBudget,
        int compressRatio) {
    const int selectedK = compressRatio > 0
        ? tokenBudget / compressRatio : 0;
    const int rows = heads > 0 && headDim > 0
        ? (int)(query.Count(0) / ((uint64_t)heads * headDim)) : 0;
    const int outputWidth = indices.dims.size() == 2
        ? indices.dims[1] : 0;
    const int capacity = compressedKeys.expansionDims.size() == 2
        ? compressedKeys.expansionDims[0]
        : (compressedKeys.dims.size() == 2
            ? compressedKeys.dims[0] : 0);
    if (decodeMeta == nullptr ||
        query.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKeys.dataDevice != fastllm::DataDevice::CUDA ||
        scores.dataDevice != fastllm::DataDevice::CUDA ||
        selectedBlocks.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        query.cudaData == nullptr || compressedKeys.cudaData == nullptr ||
        scores.cudaData == nullptr || selectedBlocks.cudaData == nullptr ||
        indices.cudaData == nullptr || !Qwen4CudaActivationType(query.dataType) ||
        compressedKeys.dataType != fastllm::DataType::FLOAT32 ||
        scores.dataType != fastllm::DataType::FLOAT32 ||
        selectedBlocks.dataType != fastllm::DataType::INT32 ||
        indices.dataType != fastllm::DataType::INT32 ||
        compressedKeys.dims.size() != 2 ||
        compressedKeys.dims[1] != headDim ||
        compressedKeys.strides.size() != 2 ||
        compressedKeys.strides[0] != (uint64_t)headDim ||
        rows <= 0 || heads <= 0 || heads > 32 || headDim <= 0 ||
        tokenBudget <= 0 || compressRatio <= 0 ||
        tokenBudget % compressRatio != 0 || capacity < selectedK ||
        outputWidth < tokenBudget ||
        outputWidth >= tokenBudget + compressRatio ||
        scores.dims != std::vector<int>({rows, capacity}) ||
        selectedBlocks.dims != std::vector<int>({rows, selectedK}) ||
        indices.dims != std::vector<int>({rows, outputWidth})) {
        return false;
    }

    const int queryStart = rows > 1 ? -2 : -1;
    return FastllmCudaQwen4QSASelectLaunch(
        query, compressedKeys, decodeMeta,
        reinterpret_cast<float *>(scores.cudaData),
        reinterpret_cast<int32_t *>(selectedBlocks.cudaData), indices,
        rows, capacity, selectedK, outputWidth,
        heads, headDim, compressRatio, queryStart,
        capacity * compressRatio);
}

static bool FastllmCudaQwen4KVAppendImpl(
        const fastllm::Data &key, const fastllm::Data &value,
        const int32_t *decodeMeta, int previousLength,
        fastllm::Data &keyCache, fastllm::Data &valueCache) {
    const int heads = key.dims.size() == 3 ? key.dims[0] : 0;
    const int sequence = key.dims.size() == 3 ? key.dims[1] : 0;
    const int headDim = key.dims.size() == 3 ? key.dims[2] : 0;
    const int keyCacheHeads = keyCache.dims.size() == 3
        ? keyCache.dims[0]
        : (keyCache.expansionDims.size() == 3
            ? keyCache.expansionDims[0] : 0);
    const int valueCacheHeads = valueCache.dims.size() == 3
        ? valueCache.dims[0]
        : (valueCache.expansionDims.size() == 3
            ? valueCache.expansionDims[0] : 0);
    const int keyCacheDim = keyCache.dims.size() == 3
        ? keyCache.dims[2]
        : (keyCache.expansionDims.size() == 3
            ? keyCache.expansionDims[2] : 0);
    const int valueCacheDim = valueCache.dims.size() == 3
        ? valueCache.dims[2]
        : (valueCache.expansionDims.size() == 3
            ? valueCache.expansionDims[2] : 0);
    const int keyCapacity = keyCache.expansionDims.size() == 3
        ? keyCache.expansionDims[1]
        : (keyCache.dims.size() == 3 ? keyCache.dims[1] : 0);
    const int valueCapacity = valueCache.expansionDims.size() == 3
        ? valueCache.expansionDims[1]
        : (valueCache.dims.size() == 3 ? valueCache.dims[1] : 0);
    if (key.dataDevice != fastllm::DataDevice::CUDA ||
        value.dataDevice != fastllm::DataDevice::CUDA ||
        keyCache.dataDevice != fastllm::DataDevice::CUDA ||
        valueCache.dataDevice != fastllm::DataDevice::CUDA ||
        key.cudaData == nullptr || value.cudaData == nullptr ||
        keyCache.cudaData == nullptr || valueCache.cudaData == nullptr ||
        !Qwen4CudaActivationType(key.dataType) ||
        value.dataType != key.dataType || keyCache.dataType != key.dataType ||
        valueCache.dataType != key.dataType || key.dims.size() != 3 ||
        value.dims != key.dims || heads <= 0 || sequence <= 0 ||
        headDim <= 0 || keyCacheHeads != heads ||
        valueCacheHeads != heads || keyCacheDim != headDim ||
        valueCacheDim != headDim || keyCapacity <= 0 ||
        valueCapacity <= 0 || key.strides.size() != 3 ||
        value.strides.size() != 3 || keyCache.strides.size() != 3 ||
        valueCache.strides.size() != 3 ||
        key.strides[1] != (uint64_t)headDim ||
        value.strides[1] != (uint64_t)headDim ||
        keyCache.strides[1] != (uint64_t)headDim ||
        valueCache.strides[1] != (uint64_t)headDim ||
        (decodeMeta == nullptr &&
         (previousLength < 0 || previousLength + sequence > keyCapacity ||
          previousLength + sequence > valueCapacity)) ||
        (decodeMeta != nullptr &&
         (sequence > keyCapacity || sequence > valueCapacity))) {
        return false;
    }
    constexpr int threads = 256;
    const int blocks = std::min(
        1024, (heads * sequence * headDim + threads - 1) / threads);
    if (key.dataType == fastllm::DataType::FLOAT32) {
        Qwen4KVAppendKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const float*)key.cudaData, (const float*)value.cudaData,
            decodeMeta, (float*)keyCache.cudaData,
            (float*)valueCache.cudaData, heads, headDim,
            sequence, previousLength,
            (int)key.strides[0], (int)value.strides[0],
            (int)keyCache.strides[0], (int)valueCache.strides[0],
            std::min(keyCapacity, valueCapacity));
    } else if (key.dataType == fastllm::DataType::FLOAT16) {
        Qwen4KVAppendKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            decodeMeta, (half*)keyCache.cudaData,
            (half*)valueCache.cudaData, heads, headDim,
            sequence, previousLength,
            (int)key.strides[0], (int)value.strides[0],
            (int)keyCache.strides[0], (int)valueCache.strides[0],
            std::min(keyCapacity, valueCapacity));
    } else {
        Qwen4KVAppendKernel<<<
            blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)key.cudaData,
            (const __nv_bfloat16*)value.cudaData, decodeMeta,
            (__nv_bfloat16*)keyCache.cudaData,
            (__nv_bfloat16*)valueCache.cudaData, heads, headDim,
            sequence, previousLength,
            (int)key.strides[0], (int)value.strides[0],
            (int)keyCache.strides[0], (int)valueCache.strides[0],
            std::min(keyCapacity, valueCapacity));
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4KVAppend(
        const fastllm::Data &key, const fastllm::Data &value,
        int previousLength,
        fastllm::Data &keyCache, fastllm::Data &valueCache) {
    return FastllmCudaQwen4KVAppendImpl(
        key, value, nullptr, previousLength, keyCache, valueCache);
}

bool FastllmCudaQwen4KVAppendGraph(
        const fastllm::Data &key, const fastllm::Data &value,
        const int32_t *decodeMeta,
        fastllm::Data &keyCache, fastllm::Data &valueCache) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaQwen4KVAppendImpl(
        key, value, decodeMeta, 0, keyCache, valueCache);
}

static bool FastllmCudaQwen4GatherKVImpl(
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const int32_t *decodeMeta,
        fastllm::Data &compactKey, fastllm::Data &compactValue) {
    const int keyHeads = key.dims.size() == 3 ? key.dims[0] : 0;
    const int keyLength = key.dims.size() == 3 ? key.dims[1] : 0;
    const int headDim = key.dims.size() == 3 ? key.dims[2] : 0;
    const int width = indices.dims.size() == 2 ? indices.dims[1] : 0;
    const std::vector<int> compactDims = {keyHeads, width, headDim};
    const bool graphMode = decodeMeta != nullptr;
    const int capacity = graphMode && key.expansionDims.size() == 3
        ? key.expansionDims[1] : keyLength;
    const int valueCapacity = value.expansionDims.size() == 3
        ? value.expansionDims[1]
        : (value.dims.size() == 3 ? value.dims[1] : 0);
    if (key.dataDevice != fastllm::DataDevice::CUDA ||
        value.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        compactKey.dataDevice != fastllm::DataDevice::CUDA ||
        compactValue.dataDevice != fastllm::DataDevice::CUDA ||
        key.cudaData == nullptr || value.cudaData == nullptr ||
        indices.cudaData == nullptr || compactKey.cudaData == nullptr ||
        compactValue.cudaData == nullptr ||
        !Qwen4CudaActivationType(key.dataType) ||
        value.dataType != key.dataType || compactKey.dataType != key.dataType ||
        compactValue.dataType != key.dataType ||
        indices.dataType != fastllm::DataType::INT32 ||
        key.dims.size() != 3 || value.dims != key.dims ||
        indices.dims.size() != 2 || indices.dims[0] != 1 ||
        keyHeads <= 0 || keyLength <= 0 || headDim <= 0 || width <= 0 ||
        capacity <= 0 ||
        compactKey.dims != compactDims || compactValue.dims != compactDims ||
        key.strides.size() != 3 || value.strides.size() != 3 ||
        indices.strides.size() != 2 ||
        key.strides[1] != (uint64_t)headDim ||
        value.strides[1] != (uint64_t)headDim ||
        indices.strides[0] != (uint64_t)width ||
        (graphMode &&
         (valueCapacity != capacity || compactKey.strides.size() != 3 ||
          compactValue.strides.size() != 3 ||
          compactKey.strides[0] != (uint64_t)width * headDim ||
          compactValue.strides[0] != (uint64_t)width * headDim))) {
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
            (float*)compactValue.cudaData, keyHeads, capacity,
            (int)key.strides[0], (int)value.strides[0],
            headDim, width, decodeMeta);
    } else if (key.dataType == fastllm::DataType::FLOAT16) {
        Qwen4GatherKVKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            (const int32_t*)indices.cudaData, (half*)compactKey.cudaData,
            (half*)compactValue.cudaData, keyHeads, capacity,
            (int)key.strides[0], (int)value.strides[0],
            headDim, width, decodeMeta);
    } else {
        Qwen4GatherKVKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)key.cudaData,
            (const __nv_bfloat16*)value.cudaData,
            (const int32_t*)indices.cudaData,
            (__nv_bfloat16*)compactKey.cudaData,
            (__nv_bfloat16*)compactValue.cudaData, keyHeads, capacity,
            (int)key.strides[0], (int)value.strides[0],
            headDim, width, decodeMeta);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4GatherKVGraph(
        const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indices, const int32_t *decodeMeta,
        fastllm::Data &compactKey, fastllm::Data &compactValue) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaQwen4GatherKVImpl(
        key, value, indices, decodeMeta, compactKey, compactValue);
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
    return FastllmCudaQwen4GatherKVImpl(
        key, value, indices, nullptr, compactKey, compactValue);
}

static bool FastllmCudaQwen4PrepareSparseBatchImpl(
        const fastllm::Data &query, const fastllm::Data &key,
        const fastllm::Data &value, const fastllm::Data &indices,
        fastllm::Data &packedQuery, fastllm::Data &compactKey,
        fastllm::Data &compactValue, fastllm::Data &paddingMask,
        int rowStart, int rows, const int32_t *decodeMeta,
        int appendedSequence) {
    const int queryHeads = query.dims.size() == 3 ? query.dims[0] : 0;
    const int sequence = query.dims.size() == 3 ? query.dims[1] : 0;
    const int headDim = query.dims.size() == 3 ? query.dims[2] : 0;
    const int keyHeads = key.dims.size() == 3 ? key.dims[0] : 0;
    const int keyLength = key.dims.size() == 3 ? key.dims[1] : 0;
    const bool graphMode = decodeMeta != nullptr;
    const int keyCapacity = graphMode && key.expansionDims.size() == 3
        ? key.expansionDims[1] : keyLength;
    const int valueCapacity = graphMode && value.expansionDims.size() == 3
        ? value.expansionDims[1]
        : (value.dims.size() == 3 ? value.dims[1] : 0);
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
        keyLength <= 0 || keyCapacity <= 0 || headDim <= 0 ||
        key.dims[2] != headDim ||
        width <= 0 || rowStart < 0 || rows <= 0 ||
        rowStart + rows > sequence ||
        (graphMode &&
         (appendedSequence <= 0 || appendedSequence != sequence ||
          valueCapacity != keyCapacity)) ||
        packedQuery.dims != packedQueryDims ||
        compactKey.dims != compactDims || compactValue.dims != compactDims ||
        paddingMask.dims != maskDims ||
        key.strides.size() != 3 || value.strides.size() != 3 ||
        key.strides[1] != (uint64_t)headDim ||
        value.strides[1] != (uint64_t)headDim) {
        return false;
    }

    constexpr int threads = 256;
    const uint64_t queryCount =
        (uint64_t)rows * queryHeads * headDim;
    const uint64_t kvCount =
        (uint64_t)rows * keyHeads * width * headDim;
    const int queryBlocks = std::min<uint64_t>(
        1024, (queryCount + threads - 1) / threads);
    const int kvBlocks = std::min<uint64_t>(
        1024, (kvCount + threads - 1) / threads);
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
            (float*)compactValue.cudaData, (float*)paddingMask.cudaData,
            keyHeads, keyCapacity,
            (int)key.strides[0], (int)value.strides[0], headDim,
            width, rowStart, rows, decodeMeta, appendedSequence);
    } else if (query.dataType == fastllm::DataType::FLOAT16 &&
               headDim % (int)(sizeof(uint4) / sizeof(half)) == 0 &&
               ((uintptr_t)key.cudaData % alignof(uint4)) == 0 &&
               ((uintptr_t)value.cudaData % alignof(uint4)) == 0 &&
               ((uintptr_t)compactKey.cudaData % alignof(uint4)) == 0 &&
               ((uintptr_t)compactValue.cudaData % alignof(uint4)) == 0) {
        Qwen4PackSparseQueryKernel<<<
            queryBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)query.cudaData, (half*)packedQuery.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
        constexpr int valuesPerVector = sizeof(uint4) / sizeof(half);
        const uint64_t vectorCount = kvCount / valuesPerVector;
        const int vectorBlocks = std::min<uint64_t>(
            1024, (vectorCount + threads - 1) / threads);
        Qwen4GatherSparseBatchKVHalf8Kernel<<<
            vectorBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            indexData, (half*)compactKey.cudaData,
            (half*)compactValue.cudaData, (half*)paddingMask.cudaData,
            keyHeads, keyCapacity,
            (int)key.strides[0], (int)value.strides[0], headDim,
            width, rowStart, rows, decodeMeta, appendedSequence);
    } else if (query.dataType == fastllm::DataType::FLOAT16) {
        Qwen4PackSparseQueryKernel<<<
            queryBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)query.cudaData, (half*)packedQuery.cudaData,
            queryHeads, sequence, headDim, rowStart, rows);
        Qwen4GatherSparseBatchKVKernel<<<
            kvBlocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)key.cudaData, (const half*)value.cudaData,
            indexData, (half*)compactKey.cudaData,
            (half*)compactValue.cudaData, (half*)paddingMask.cudaData,
            keyHeads, keyCapacity,
            (int)key.strides[0], (int)value.strides[0], headDim,
            width, rowStart, rows, decodeMeta, appendedSequence);
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
            (__nv_bfloat16*)paddingMask.cudaData,
            keyHeads, keyCapacity,
            (int)key.strides[0], (int)value.strides[0], headDim,
            width, rowStart, rows, decodeMeta, appendedSequence);
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
    return FastllmCudaQwen4PrepareSparseBatchImpl(
        query, key, value, indices, packedQuery, compactKey,
        compactValue, paddingMask, rowStart, rows, nullptr, 0);
}

bool FastllmCudaQwen4PrepareSparseBatchGraph(
        const fastllm::Data &query, const fastllm::Data &key,
        const fastllm::Data &value, const fastllm::Data &indices,
        const int32_t *decodeMeta, int appendedSequence,
        fastllm::Data &packedQuery, fastllm::Data &compactKey,
        fastllm::Data &compactValue, fastllm::Data &paddingMask,
        int rowStart, int rows) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaQwen4PrepareSparseBatchImpl(
        query, key, value, indices, packedQuery, compactKey,
        compactValue, paddingMask, rowStart, rows,
        decodeMeta, appendedSequence);
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
        !Qwen4CudaActivationType(output.dataType) ||
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
        Qwen4DispatchCausalDepthwiseConv1DPrefillOutput<float>(
            input, weight, state, output, outputCount, outputBlocks,
            stateItems, stateBlocks, sequence, channels, kernel, silu);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Qwen4DispatchCausalDepthwiseConv1DPrefillOutput<half>(
            input, weight, state, output, outputCount, outputBlocks,
            stateItems, stateBlocks, sequence, channels, kernel, silu);
    } else {
        Qwen4DispatchCausalDepthwiseConv1DPrefillOutput<__nv_bfloat16>(
            input, weight, state, output, outputCount, outputBlocks,
            stateItems, stateBlocks, sequence, channels, kernel, silu);
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
        float recurrentEps, fastllm::Data *stateOutput) {
    const int batch = qkv.dims.empty() ? 0 : qkv.dims[0];
    const int sequence = qkv.dims.size() == 3 ? qkv.dims[1] : 0;
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
        sequence <= 0 || qkv.dims[2] != qkvChannels ||
        alpha.Count(0) !=
            (uint64_t)batch * sequence * valueHeads ||
        beta.Count(0) !=
            (uint64_t)batch * sequence * valueHeads ||
        aLog.Count(0) != (uint64_t)valueHeads ||
        dtBias.Count(0) != (uint64_t)valueHeads ||
        state.dims != std::vector<int>({batch, valueHeads, keyDim, valueDim}) ||
        (stateOutput != nullptr &&
         (stateOutput->dataDevice != fastllm::DataDevice::CUDA ||
          stateOutput->cudaData == nullptr ||
          stateOutput->dataType != fastllm::DataType::FLOAT32 ||
          stateOutput->dims != state.dims))) {
        return false;
    }
    const int blocks = batch * valueHeads;
    // RunLinearAttention constructs this value on the host and then uses it
    // as both the RMSNorm weight and the subsequent query scale.  Passing the
    // same host result avoids the one-ULP difference of device rsqrtf().
    const float inverseHead = 1.0f / std::sqrt((float)keyDim);
    int device = -1;
    int major = 0;
    int minor = 0;
    const bool useValueTileSm120 = sequence > 1 && sequence <= 9 &&
        cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, device) ==
            cudaSuccess &&
        major == 12 && minor == 0;
    if (alpha.dataType == fastllm::DataType::FLOAT32) {
        if (stateOutput == nullptr) {
            Qwen4LaunchGatedDeltaRule<float, false>(
                qkv, alpha, beta, aLog, dtBias, state, nullptr, output,
                blocks, keyHeads, valueHeads, sequence, recurrentEps,
                inverseHead, useValueTileSm120);
        } else {
            Qwen4LaunchGatedDeltaRule<float, true>(
                qkv, alpha, beta, aLog, dtBias, state, stateOutput, output,
                blocks, keyHeads, valueHeads, sequence, recurrentEps,
                inverseHead, useValueTileSm120);
        }
    } else if (alpha.dataType == fastllm::DataType::FLOAT16) {
        if (stateOutput == nullptr) {
            Qwen4LaunchGatedDeltaRule<half, false>(
                qkv, alpha, beta, aLog, dtBias, state, nullptr, output,
                blocks, keyHeads, valueHeads, sequence, recurrentEps,
                inverseHead, useValueTileSm120);
        } else {
            Qwen4LaunchGatedDeltaRule<half, true>(
                qkv, alpha, beta, aLog, dtBias, state, stateOutput, output,
                blocks, keyHeads, valueHeads, sequence, recurrentEps,
                inverseHead, useValueTileSm120);
        }
    } else if (stateOutput == nullptr) {
        Qwen4LaunchGatedDeltaRule<__nv_bfloat16, false>(
            qkv, alpha, beta, aLog, dtBias, state, nullptr, output,
            blocks, keyHeads, valueHeads, sequence, recurrentEps,
            inverseHead, useValueTileSm120);
    } else {
        Qwen4LaunchGatedDeltaRule<__nv_bfloat16, true>(
            qkv, alpha, beta, aLog, dtBias, state, stateOutput, output,
            blocks, keyHeads, valueHeads, sequence, recurrentEps,
            inverseHead, useValueTileSm120);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4GdnOutputGateExact(
        const fastllm::Data &input,
        const fastllm::Data &normWeight,
        const fastllm::Data &gate,
        fastllm::Data &output, float eps) {
    constexpr int CHANNELS = 128;
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        gate.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        gate.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        input.cudaData == nullptr || normWeight.cudaData == nullptr ||
        gate.cudaData == nullptr || output.cudaData == nullptr ||
        input.multiDeviceData || normWeight.multiDeviceData ||
        gate.multiDeviceData || output.multiDeviceData ||
        input.dims.empty() || input.dims.back() != CHANNELS ||
        input.dims != gate.dims || input.dims != output.dims ||
        normWeight.Count(0) != CHANNELS || !(eps > 0.0f)) {
        return false;
    }
    const uint64_t count = input.Count(0);
    if (count == 0 || count % CHANNELS != 0 ||
        count / CHANNELS > (uint64_t)std::numeric_limits<int>::max()) {
        return false;
    }
    const int device = input.dataDeviceIds.empty()
        ? FastllmCudaGetDevice() : input.dataDeviceIds[0];
    const auto onDevice = [&](const fastllm::Data &data) {
        return data.dataDeviceIds.empty() ||
               data.dataDeviceIds[0] == device;
    };
    if (!onDevice(normWeight) || !onDevice(gate) || !onDevice(output)) {
        return false;
    }

    Qwen4GdnOutputGateExactKernel<<<
        (int)(count / CHANNELS), 32, 0, cudaStreamPerThread>>>(
        (const float*)input.cudaData,
        (const float*)normWeight.cudaData,
        (const half*)gate.cudaData,
        (half*)output.cudaData, eps);
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4ReplayLinearAttentionBatch(
        const std::vector<FastllmCudaQwen4LinearReplayItem> &items,
        fastllm::Data &pointerWorkspace,
        std::vector<uint8_t> &pointerCache,
        int sequence, int convKernel, bool silu,
        int keyHeads, int valueHeads, int keyDim, int valueDim,
        float recurrentEps) {
    if (items.empty() || sequence <= 0 || convKernel <= 0 ||
        keyHeads <= 0 || valueHeads <= 0 ||
        valueHeads % keyHeads != 0 ||
        keyDim != 128 || valueDim != 128) {
        return false;
    }
    const int qkvChannels =
        2 * keyHeads * keyDim + valueHeads * valueDim;
    const fastllm::Data *firstInput = items.front().convInput;
    if (firstInput == nullptr ||
        firstInput->dataDevice != fastllm::DataDevice::CUDA ||
        firstInput->cudaData == nullptr ||
        !Qwen4CudaActivationType(firstInput->dataType)) {
        return false;
    }
    const int device = firstInput->dataDeviceIds.empty()
        ? FastllmCudaGetDevice() : firstInput->dataDeviceIds[0];
    const fastllm::DataType inputType = firstInput->dataType;
    fastllm::DataType gateType = fastllm::DataType::FLOAT32;
    bool gateTypeReady = false;
    auto onDevice = [&](const fastllm::Data *data) {
        return data != nullptr &&
            data->dataDevice == fastllm::DataDevice::CUDA &&
            data->cudaData != nullptr &&
            (data->dataDeviceIds.empty() ||
             data->dataDeviceIds[0] == device);
    };

    const size_t pointerBytes =
        items.size() * sizeof(Qwen4LinearReplayPointers);
    bool pointerCacheMatches = pointerCache.size() == pointerBytes;
    auto makeReplayPointers = [](
            const FastllmCudaQwen4LinearReplayItem &item) {
        return Qwen4LinearReplayPointers {
            item.convInput->cudaData,
            (const float*)item.convWeight->cudaData,
            (const float*)item.convCheckpoint->cudaData,
            (float*)item.convState->cudaData,
            (float*)item.convolved->cudaData,
            item.alpha->cudaData,
            item.beta->cudaData,
            (const float*)item.aLog->cudaData,
            (const float*)item.dtBias->cudaData,
            (const float*)item.recurrentCheckpoint->cudaData,
            (float*)item.recurrentState->cudaData,
            (float*)item.coreOutput->cudaData
        };
    };
    auto hasExactDims3 = [](const fastllm::Data *data,
                            int dim0, int dim1, int dim2) {
        return data->dims.size() == 3 &&
            data->dims[0] == dim0 && data->dims[1] == dim1 &&
            data->dims[2] == dim2;
    };
    auto hasExactDims4 = [](const fastllm::Data *data,
                            int dim0, int dim1, int dim2, int dim3) {
        return data->dims.size() == 4 &&
            data->dims[0] == dim0 && data->dims[1] == dim1 &&
            data->dims[2] == dim2 && data->dims[3] == dim3;
    };
    size_t pointerOffset = 0;
    for (const auto &item : items) {
        if (!onDevice(item.convInput) || !onDevice(item.convWeight) ||
            !onDevice(item.convCheckpoint) ||
            !onDevice(item.convState) || !onDevice(item.convolved) ||
            !onDevice(item.alpha) || !onDevice(item.beta) ||
            !onDevice(item.aLog) || !onDevice(item.dtBias) ||
            !onDevice(item.recurrentCheckpoint) ||
            !onDevice(item.recurrentState) ||
            !onDevice(item.coreOutput) ||
            item.convInput->dataType != inputType ||
            item.convInput->dims.size() != 3 ||
            item.convInput->dims[0] != 1 ||
            item.convInput->dims[1] < sequence ||
            item.convInput->dims[2] != qkvChannels ||
            item.convWeight->dataType != fastllm::DataType::FLOAT32 ||
            item.convWeight->Count(0) !=
                (uint64_t)qkvChannels * convKernel ||
            item.convCheckpoint->dataType !=
                fastllm::DataType::FLOAT32 ||
            !hasExactDims3(
                item.convCheckpoint, 1, qkvChannels, convKernel) ||
            item.convState->dataType != fastllm::DataType::FLOAT32 ||
            !hasExactDims3(
                item.convState, 1, qkvChannels, convKernel) ||
            item.convolved->dataType != fastllm::DataType::FLOAT32 ||
            !hasExactDims3(
                item.convolved, 1, sequence, qkvChannels) ||
            !Qwen4CudaActivationType(item.alpha->dataType) ||
            item.beta->dataType != item.alpha->dataType ||
            item.alpha->Count(0) <
                (uint64_t)sequence * valueHeads ||
            item.beta->Count(0) <
                (uint64_t)sequence * valueHeads ||
            item.aLog->dataType != fastllm::DataType::FLOAT32 ||
            item.dtBias->dataType != fastllm::DataType::FLOAT32 ||
            item.aLog->Count(0) != (uint64_t)valueHeads ||
            item.dtBias->Count(0) != (uint64_t)valueHeads ||
            item.recurrentState->dataType !=
                fastllm::DataType::FLOAT32 ||
            item.recurrentCheckpoint->dataType !=
                fastllm::DataType::FLOAT32 ||
            !hasExactDims4(
                item.recurrentCheckpoint, 1, valueHeads,
                keyDim, valueDim) ||
            !hasExactDims4(
                item.recurrentState, 1, valueHeads,
                keyDim, valueDim) ||
            item.coreOutput->dataType !=
                fastllm::DataType::FLOAT32 ||
            !hasExactDims3(
                item.coreOutput, 1, sequence,
                valueHeads * valueDim)) {
            return false;
        }
        if (!gateTypeReady) {
            gateType = item.alpha->dataType;
            gateTypeReady = true;
        } else if (gateType != item.alpha->dataType) {
            return false;
        }
        if (pointerCacheMatches) {
            const Qwen4LinearReplayPointers current =
                makeReplayPointers(item);
            pointerCacheMatches = std::memcmp(
                pointerCache.data() + pointerOffset, &current,
                sizeof(current)) == 0;
        }
        pointerOffset += sizeof(Qwen4LinearReplayPointers);
    }

    FastllmCudaSetDevice(device);
    const bool workspaceCompatible =
        pointerWorkspace.dataType == fastllm::DataType::INT8 &&
        pointerWorkspace.dataDevice == fastllm::DataDevice::CUDA &&
        pointerWorkspace.cudaData != nullptr &&
        pointerWorkspace.Count(0) == pointerBytes &&
        !pointerWorkspace.dataDeviceIds.empty() &&
        pointerWorkspace.dataDeviceIds[0] == device;
    if (!workspaceCompatible) {
        pointerWorkspace.dataType = fastllm::DataType::INT8;
        pointerWorkspace.UpdateUnitSize();
        pointerWorkspace.Resize({(int)pointerBytes});
        pointerWorkspace.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>({device}));
        pointerWorkspace.Allocate(false);
    }
    if (pointerWorkspace.cudaData == nullptr) {
        return false;
    }
    const bool pointersChanged =
        !workspaceCompatible || !pointerCacheMatches;
    if (pointersChanged) {
        pointerCache.resize(pointerBytes);
        for (size_t i = 0; i < items.size(); i++) {
            const Qwen4LinearReplayPointers current =
                makeReplayPointers(items[i]);
            std::memcpy(
                pointerCache.data() +
                    i * sizeof(Qwen4LinearReplayPointers),
                &current, sizeof(current));
        }
        if (cudaMemcpy(
                pointerWorkspace.cudaData, pointerCache.data(), pointerBytes,
                cudaMemcpyHostToDevice) != cudaSuccess) {
            pointerCache.clear();
            return false;
        }
    }
    const Qwen4LinearReplayPointers *devicePointers =
        (const Qwen4LinearReplayPointers*)pointerWorkspace.cudaData;
    constexpr int threads = 256;
    const int convVectors = qkvChannels * convKernel / 4;
    const int recurrentVectors = valueHeads * keyDim * valueDim / 4;
    bool restoreRecurrent = false;
    for (const auto &item : items) {
        restoreRecurrent = restoreRecurrent ||
            item.recurrentCheckpoint->cudaData !=
                item.recurrentState->cudaData;
    }
    const int vectorsPerLayer = convVectors +
        (restoreRecurrent ? recurrentVectors : 0);
    const uint64_t restoreVectors =
        (uint64_t)items.size() * vectorsPerLayer;
    const int restoreBlocks = std::min<uint64_t>(
        65535, (restoreVectors + threads - 1) / threads);
    Qwen4LinearReplayRestoreKernel<<<
        restoreBlocks, threads, 0, cudaStreamPerThread>>>(
        devicePointers, restoreVectors, vectorsPerLayer, convVectors);
    const uint64_t convCount =
        (uint64_t)items.size() * sequence * qkvChannels;
    const int convBlocks = std::min<uint64_t>(
        65535, (convCount + threads - 1) / threads);
    const int stateCount = (int)items.size() * qkvChannels;
    const int stateBlocks = (stateCount + threads - 1) / threads;
    if (inputType == fastllm::DataType::FLOAT32) {
        Qwen4LinearReplayConvOutputKernel<float><<<
            convBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, convCount, sequence, qkvChannels,
            convKernel, silu);
        Qwen4LinearReplayConvStateKernel<float><<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, stateCount, sequence, qkvChannels,
            convKernel);
    } else if (inputType == fastllm::DataType::FLOAT16) {
        Qwen4LinearReplayConvOutputKernel<half><<<
            convBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, convCount, sequence, qkvChannels,
            convKernel, silu);
        Qwen4LinearReplayConvStateKernel<half><<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, stateCount, sequence, qkvChannels,
            convKernel);
    } else {
        Qwen4LinearReplayConvOutputKernel<__nv_bfloat16><<<
            convBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, convCount, sequence, qkvChannels,
            convKernel, silu);
        Qwen4LinearReplayConvStateKernel<__nv_bfloat16><<<
            stateBlocks, threads, 0, cudaStreamPerThread>>>(
            devicePointers, stateCount, sequence, qkvChannels,
            convKernel);
    }

    const int recurrentBlocks = (int)items.size() * valueHeads;
    const float inverseHead = 1.0f / std::sqrt((float)keyDim);
    if (gateType == fastllm::DataType::FLOAT32) {
        Qwen4LinearReplayGatedDeltaStateKernel<float><<<
            recurrentBlocks, 128, 0, cudaStreamPerThread>>>(
            devicePointers, valueHeads, keyHeads, sequence,
            recurrentEps, inverseHead);
    } else if (gateType == fastllm::DataType::FLOAT16) {
        Qwen4LinearReplayGatedDeltaStateKernel<half><<<
            recurrentBlocks, 128, 0, cudaStreamPerThread>>>(
            devicePointers, valueHeads, keyHeads, sequence,
            recurrentEps, inverseHead);
    } else {
        Qwen4LinearReplayGatedDeltaStateKernel<__nv_bfloat16><<<
            recurrentBlocks, 128, 0, cudaStreamPerThread>>>(
            devicePointers, valueHeads, keyHeads, sequence,
            recurrentEps, inverseHead);
    }
    // The model executor may dispatch the following layer on another host
    // thread (and therefore another per-thread CUDA stream).  Complete this
    // rejected-prefix transition before publishing the restored caches.
    ForceDeviceSync();
    return cudaGetLastError() == cudaSuccess;
}
