#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

    template <typename T>
    __global__ void Qwen4HyperCombineKernel(
            const T *hyperInput, const T *blockOutput,
            const T *injection, T *output, uint64_t count,
            int groups, int blockChannels) {
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
        const float injected = Qwen4CudaRound<T>(
            Qwen4CudaToFloat(blockOutput[row * blockChannels + channel]) *
            Qwen4CudaToFloat(injection[row * groups + group]));
        output[index] = Qwen4CudaFromFloat<T>(Qwen4CudaRound<T>(
            Qwen4CudaToFloat(hyperInput[index]) + injected));
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
        hyperInput.dataType != blockOutput.dataType ||
        hyperInput.dataType != injection.dataType ||
        hyperInput.dataType != output.dataType ||
        !Qwen4CudaActivationType(hyperInput.dataType)) {
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
            (const float*)blockOutput.cudaData,
            (const float*)injection.cudaData, (float*)output.cudaData,
            count, groups, blockChannels);
    } else if (hyperInput.dataType == fastllm::DataType::FLOAT16) {
        Qwen4HyperCombineKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const half*)hyperInput.cudaData,
            (const half*)blockOutput.cudaData,
            (const half*)injection.cudaData, (half*)output.cudaData,
            count, groups, blockChannels);
    } else {
        Qwen4HyperCombineKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)hyperInput.cudaData,
            (const __nv_bfloat16*)blockOutput.cudaData,
            (const __nv_bfloat16*)injection.cudaData,
            (__nv_bfloat16*)output.cudaData, count, groups,
            blockChannels);
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
