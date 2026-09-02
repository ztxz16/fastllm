//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

typedef union __align__(16) _union_bf16_4_fp16 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
} union_bf16_4_fp16;

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvBf16Fp16Kernel2MultiRow(__nv_bfloat16 *A, half *B, __nv_bfloat16 *C, __nv_bfloat16 *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_bf16_4_fp16 regA;
    union_half4 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const half *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint2 *>(A + i + x * m);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __bfloat162float(regA.out2[0].x) * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += __bfloat162float(regA.out2[0].y) * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += __bfloat162float(regA.out2[1].x) * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += __bfloat162float(regA.out2[1].y) * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __bfloat162float(A[i + x * m]) * __half2float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        diff[x] = 0.0f;
    }
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2bfloat16_rn(sdata[x][0] + __bfloat162float(bias[p]));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2bfloat16_rn(sdata[x][0]);
        }
    }
}

template <int THREAD_PER_BLOCK, int PART, bool WITH_SHARED_GATE,
          bool INDEPENDENT_ROW_REDUCTION = false>
__global__ void FastllmGemvFp16Fp16Kernel2MultiRow(
        half *A, half *B, half *C, half *bias, int m, int k, bool addTo,
        half *sharedGateB, half *sharedGateC, bool sigmoidSharedGate) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    union_half8 regA;
    union_half8 regB;

    // 1. 计算
    int st = blockIdx.x;
    const bool isSharedGate = WITH_SHARED_GATE && st == k;
    int p = isSharedGate ? 0 : st;
    float accumulators[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) accumulators[x] = 0.0f;
        
    const half *baseB = isSharedGate ? sharedGateB : B + p * m;

    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
            regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1]);
                if (i + 4 < m)
                    sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2]);
                if (i + 5 < m)
                    sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2]);
                if (i + 6 < m)
                    sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3]);
                if (i + 7 < m)
                    sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3]);
                accumulators[x] = __fadd_rn(accumulators[x], sum);
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            const float weightValue = (float)baseB[i];
#pragma unroll
            for (int x = 0; x < PART; x++) {
                accumulators[x] = __fadd_rn(
                    accumulators[x],
                    (float)A[i + x * m] * weightValue);
            }
        }
    }
#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata[x][tid] = accumulators[x];
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        diff[x] = 0.0f;
    }
    if constexpr (INDEPENDENT_ROW_REDUCTION) {
        // Preserve the legacy 256-thread compensated tree through stride 32,
        // then execute the identical final five stages with warp shuffles.
        // This removes five CTA barriers from every exact verifier GEMV
        // without changing the operation order of any output row.
        for (unsigned int s = THREAD_PER_BLOCK / 2; s >= 32; s >>= 1) {
            if (tid < s) {
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    float other = sdata[x][tid + s] - diff[x];
                    float sumTmp = sdata[x][tid] + other;
                    diff[x] = (sumTmp - sdata[x][tid]) - other;
                    sdata[x][tid] = sumTmp;
                }
            }
            __syncthreads();
        }
        if (tid < 32) {
            float values[PART];
#pragma unroll
            for (int x = 0; x < PART; x++) {
                values[x] = sdata[x][tid];
            }
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    float rhs = __shfl_down_sync(
                        0xffffffffu, values[x], s);
                    if (tid < (unsigned int)s) {
                        float other = rhs - diff[x];
                        float sumTmp = values[x] + other;
                        diff[x] = (sumTmp - values[x]) - other;
                        values[x] = sumTmp;
                    }
                }
            }
            if (tid == 0) {
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    sdata[x][0] = values[x];
                }
            }
        }
    } else {
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    float other = sdata[x][tid + s] - diff[x];
                    float sumTmp = sdata[x][tid] + other;
                    diff[x] = (sumTmp - sdata[x][tid]) - other;
                    sdata[x][tid] = sumTmp;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        if (isSharedGate) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                half outputValue = (half)sdata[x][0];
                if (sigmoidSharedGate) {
#ifdef CUDA_NO_TENSOR_CORE
                    outputValue = __float2half(
                        1.0f / (1.0f + expf(-__half2float(outputValue))));
#else
                    outputValue = __hdiv(
                        1.0, __hadd(__float2half(1.0), hexp(-outputValue)));
#endif
                }
                sharedGateC[x] = outputValue;
            }
        } else if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float val = sdata[x][0] + (float)(__ldg(bias + p));
                C[p + k * x] = addTo ?
                    (half)(val + (float)C[p + k * x]) : (half)val;
            }
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float val = sdata[x][0];
                C[p + k * x] = addTo ?
                    (half)(val + (float)C[p + k * x]) : (half)val;
            }
        }
    }
    if constexpr (!INDEPENDENT_ROW_REDUCTION) {
        __syncthreads();
    }
}

namespace {
    constexpr int FASTLLM_HALF_MULTI_LINEAR_MAX = 4;

    // This is the exact small-batch native GEMV above with blockIdx.x spread
    // across several independent weights. A CTA still owns one output row,
    // so its products, compensated tree and final FP16 rounding are unchanged.
    template <int PART>
    __global__ void FastllmHalfMultiLinearExactKernel(
            const half *input,
            const half *weight0, const half *weight1,
            const half *weight2, const half *weight3,
            half *output0, half *output1,
            half *output2, half *output3,
            int outputRows0, int outputRows1,
            int outputRows2, int outputRows3,
            int columns) {
        __shared__ float partial[PART][256];
        const int tid = threadIdx.x;
        const int globalRow = blockIdx.x;
        const int end0 = outputRows0;
        const int end1 = end0 + outputRows1;
        const int end2 = end1 + outputRows2;
        int row;
        int outputRows;
        const half *weightBase;
        half *output;
        if (globalRow < end0) {
            row = globalRow;
            outputRows = outputRows0;
            weightBase = weight0;
            output = output0;
        } else if (globalRow < end1) {
            row = globalRow - end0;
            outputRows = outputRows1;
            weightBase = weight1;
            output = output1;
        } else if (globalRow < end2) {
            row = globalRow - end1;
            outputRows = outputRows2;
            weightBase = weight2;
            output = output2;
        } else {
            row = globalRow - end2;
            outputRows = outputRows3;
            weightBase = weight3;
            output = output3;
        }
        const half *weight = weightBase + (size_t)row * columns;

        float accumulators[PART];
#pragma unroll
        for (int item = 0; item < PART; item++) {
            accumulators[item] = 0.0f;
        }

        union_half8 packedInput;
        union_half8 packedWeight;
        if ((columns & 7) == 0) {
            for (int column = tid * 8; column < columns;
                 column += blockDim.x * 8) {
                packedWeight.in = *reinterpret_cast<const uint4 *>(
                    weight + column);
#pragma unroll
                for (int item = 0; item < PART; item++) {
                    packedInput.in = *reinterpret_cast<const uint4 *>(
                        input + (size_t)item * columns + column);
                    float sum = 0.0f;
                    sum += __low2float(packedInput.out2[0]) *
                           __low2float(packedWeight.out2[0]);
                    sum += __high2float(packedInput.out2[0]) *
                           __high2float(packedWeight.out2[0]);
                    sum += __low2float(packedInput.out2[1]) *
                           __low2float(packedWeight.out2[1]);
                    sum += __high2float(packedInput.out2[1]) *
                           __high2float(packedWeight.out2[1]);
                    sum += __low2float(packedInput.out2[2]) *
                           __low2float(packedWeight.out2[2]);
                    sum += __high2float(packedInput.out2[2]) *
                           __high2float(packedWeight.out2[2]);
                    sum += __low2float(packedInput.out2[3]) *
                           __low2float(packedWeight.out2[3]);
                    sum += __high2float(packedInput.out2[3]) *
                           __high2float(packedWeight.out2[3]);
                    accumulators[item] =
                        __fadd_rn(accumulators[item], sum);
                }
            }
        } else {
            for (int column = tid; column < columns;
                 column += blockDim.x) {
                const float weightValue = (float)weight[column];
#pragma unroll
                for (int item = 0; item < PART; item++) {
                    accumulators[item] = __fadd_rn(
                        accumulators[item],
                        (float)input[(size_t)item * columns + column] *
                            weightValue);
                }
            }
        }

#pragma unroll
        for (int item = 0; item < PART; item++) {
            partial[item][tid] = accumulators[item];
        }
        __syncthreads();

        float corrections[PART];
#pragma unroll
        for (int item = 0; item < PART; item++) {
            corrections[item] = 0.0f;
        }
        for (int stride = 128; stride >= 32; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int item = 0; item < PART; item++) {
                    float other =
                        partial[item][tid + stride] - corrections[item];
                    float sum = partial[item][tid] + other;
                    corrections[item] =
                        (sum - partial[item][tid]) - other;
                    partial[item][tid] = sum;
                }
            }
            __syncthreads();
        }

        if (tid < 32) {
            float values[PART];
#pragma unroll
            for (int item = 0; item < PART; item++) {
                values[item] = partial[item][tid];
            }
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
#pragma unroll
                for (int item = 0; item < PART; item++) {
                    const float rhs = __shfl_down_sync(
                        0xffffffffu, values[item], stride);
                    if (tid < stride) {
                        const float other = rhs - corrections[item];
                        const float sum = values[item] + other;
                        corrections[item] =
                            (sum - values[item]) - other;
                        values[item] = sum;
                    }
                }
            }
            if (tid == 0) {
#pragma unroll
                for (int item = 0; item < PART; item++) {
                    output[row + (size_t)outputRows * item] =
                        (half)values[item];
                }
            }
        }
    }
}

bool FastllmCudaHalfMultiLinearExact(
        const fastllm::Data &input,
        const fastllm::Data *const *weights,
        fastllm::Data *const *outputs, int count) {
    if (count <= 0 || count > FASTLLM_HALF_MULTI_LINEAR_MAX ||
        weights == nullptr || outputs == nullptr ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        input.cudaData == nullptr || input.dims.empty() ||
        input.multiDeviceData) {
        return false;
    }
    const int columns = input.dims.back();
    const uint64_t inputCount = input.Count(0);
    const int rows = columns > 0 ? (int)(inputCount / columns) : 0;
    const int inputDevice = input.dataDeviceIds.empty()
        ? FastllmCudaGetDevice() : input.dataDeviceIds[0];
    if (columns <= 0 || inputCount != (uint64_t)rows * columns ||
        rows < 1 || rows > 8) {
        return false;
    }

    const half *cudaWeights[FASTLLM_HALF_MULTI_LINEAR_MAX]{};
    half *cudaOutputs[FASTLLM_HALF_MULTI_LINEAR_MAX]{};
    int outputRows[FASTLLM_HALF_MULTI_LINEAR_MAX]{};
    int totalOutputRows = 0;
    for (int i = 0; i < count; i++) {
        const fastllm::Data *weight = weights[i];
        fastllm::Data *output = outputs[i];
        const int weightDevice = weight != nullptr &&
                !weight->dataDeviceIds.empty()
            ? weight->dataDeviceIds[0] : inputDevice;
        const int outputDevice = output != nullptr &&
                !output->dataDeviceIds.empty()
            ? output->dataDeviceIds[0] : inputDevice;
        if (weight == nullptr || output == nullptr ||
            weight->dataDevice != fastllm::DataDevice::CUDA ||
            weight->dataType != fastllm::DataType::FLOAT16 ||
            weight->cudaData == nullptr || weight->dims.size() != 2 ||
            weight->dims[1] != columns || weight->dims[0] <= 0 ||
            weight->multiDeviceData ||
            output->dataDevice != fastllm::DataDevice::CUDA ||
            output->dataType != fastllm::DataType::FLOAT16 ||
            output->cudaData == nullptr || output->multiDeviceData ||
            weightDevice != inputDevice || outputDevice != inputDevice) {
            return false;
        }
        std::vector<int> expectedDims = input.dims;
        expectedDims.back() = weight->dims[0];
        if (output->dims != expectedDims) {
            return false;
        }
        cudaWeights[i] = reinterpret_cast<const half *>(
            weight->cudaData);
        cudaOutputs[i] = reinterpret_cast<half *>(output->cudaData);
        outputRows[i] = weight->dims[0];
        totalOutputRows += weight->dims[0];
    }
    if (totalOutputRows <= 0) {
        return false;
    }

#define FASTLLM_HALF_MULTI_LINEAR_LAUNCH(PART) \
    FastllmHalfMultiLinearExactKernel<PART><<< \
        totalOutputRows, 256, 0, cudaStreamPerThread>>>( \
            reinterpret_cast<const half *>(input.cudaData), \
            cudaWeights[0], cudaWeights[1], cudaWeights[2], cudaWeights[3], \
            cudaOutputs[0], cudaOutputs[1], cudaOutputs[2], cudaOutputs[3], \
            outputRows[0], outputRows[1], outputRows[2], outputRows[3], \
            columns)
    switch (rows) {
        case 1: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(1); break;
        case 2: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(2); break;
        case 3: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(3); break;
        case 4: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(4); break;
        case 5: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(5); break;
        case 6: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(6); break;
        case 7: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(7); break;
        case 8: FASTLLM_HALF_MULTI_LINEAR_LAUNCH(8); break;
        default: return false;
    }
#undef FASTLLM_HALF_MULTI_LINEAR_LAUNCH
    return cudaGetLastError() == cudaSuccess;
}

// Batch-1 projection specialization for the two input widths used by Qwen3.5
// decode.  The legacy GEMV assigns one 256-thread CTA to every output row even
// though a 2048-wide row gives each thread only eight values (and a 256-wide
// row leaves seven warps idle).  Here one warp owns a row and a CTA computes
// eight adjacent rows.  For INPUT_SIZE=2048 every lane explicitly reproduces
// eight legacy lanes before continuing the same binary reduction tree with
// warp shuffles, so the final FP16 result keeps the legacy accumulation order.
template <int INPUT_SIZE, int WARPS_PER_BLOCK = 8>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32)
void FastllmGemvFp16Fp16WarpRowsKernel(
        const half *A, const half *B, half *C, const half *bias,
        int k, bool addTo) {
    static_assert(INPUT_SIZE == 256 || INPUT_SIZE == 2048,
                  "warp-row GEMV only supports tuned decode input widths");
    constexpr int WARP_SIZE = 32;
    constexpr int VALUES_PER_LOAD = 8;
    constexpr int LEGACY_LANES_PER_WARP =
        INPUT_SIZE / (WARP_SIZE * VALUES_PER_LOAD);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= k) {
        return;
    }

    const half *baseB = B + (size_t)row * INPUT_SIZE;
    float values[LEGACY_LANES_PER_WARP];
    float diffs[LEGACY_LANES_PER_WARP];

#pragma unroll
    for (int virtualLane = 0; virtualLane < LEGACY_LANES_PER_WARP;
         virtualLane++) {
        int i = (lane + virtualLane * WARP_SIZE) * VALUES_PER_LOAD;
        union_half8 regA;
        union_half8 regB;
        regA.in = *reinterpret_cast<const uint4 *>(A + i);
        regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
        float sum = 0.0f;
        sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0]);
        sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0]);
        sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1]);
        sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1]);
        sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2]);
        sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2]);
        sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3]);
        sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3]);
        values[virtualLane] = sum;
        diffs[virtualLane] = 0.0f;
    }

    // Reproduce the legacy 256 -> 128 -> 64 -> 32 lane stages locally.
#pragma unroll
    for (int stride = LEGACY_LANES_PER_WARP / 2; stride > 0; stride >>= 1) {
#pragma unroll
        for (int virtualLane = 0; virtualLane < stride; virtualLane++) {
            float other =
                values[virtualLane + stride] - diffs[virtualLane];
            float sum = values[virtualLane] + other;
            diffs[virtualLane] =
                (sum - values[virtualLane]) - other;
            values[virtualLane] = sum;
        }
    }

    float value = values[0];
    float diff = diffs[0];
    unsigned int mask = __activemask();
#pragma unroll
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        float rhs = __shfl_down_sync(mask, value, stride);
        if (lane < stride) {
            float other = rhs - diff;
            float sum = value + other;
            diff = (sum - value) - other;
            value = sum;
        }
    }

    if (lane == 0) {
        if (bias != nullptr) {
            value += (float)__ldg(bias + row);
        }
        C[row] = addTo ? (half)(value + (float)C[row]) : (half)value;
    }
}

// Qwen3.5 single-row router specialization (1x2048 @ 256x2048). One 64-thread
// CTA computes one expert logit. Each thread reproduces four lanes of the
// legacy 256-thread kernel, including its compensated reduction order, then
// replaces only the final five block barriers with the equivalent warp tree.
// Small batched decode uses FastllmGemvFp16Fp16Kernel2MultiRow above so it can
// preserve the legacy multi-row accumulation order.
template <bool WITH_SHARED_GATE>
__global__ __launch_bounds__(64) void FastllmGemvFp16Fp16Router2048x256Kernel(
        const half *A, const half *B, half *C, const half *bias, bool addTo,
        const half *sharedGateB = nullptr, half *sharedGateC = nullptr,
        bool sigmoidSharedGate = false) {
    constexpr int THREADS = 64;
    constexpr int HIDDEN = 2048;
    int tid = threadIdx.x;
    const bool isSharedGate = WITH_SHARED_GATE && blockIdx.x == 256;
    int row = isSharedGate ? 0 : blockIdx.x;
    const half *baseB = isSharedGate
        ? sharedGateB
        : B + (size_t)row * HIDDEN;

    int offset0 = tid * 8;
    int offset1 = (tid + THREADS) * 8;
    int offset2 = (tid + THREADS * 2) * 8;
    int offset3 = (tid + THREADS * 3) * 8;
    union_half8 a0, a1, a2, a3, b0, b1, b2, b3;
    a0.in = *reinterpret_cast<const uint4 *>(A + offset0);
    b0.in = *reinterpret_cast<const uint4 *>(baseB + offset0);
    a1.in = *reinterpret_cast<const uint4 *>(A + offset1);
    b1.in = *reinterpret_cast<const uint4 *>(baseB + offset1);
    a2.in = *reinterpret_cast<const uint4 *>(A + offset2);
    b2.in = *reinterpret_cast<const uint4 *>(baseB + offset2);
    a3.in = *reinterpret_cast<const uint4 *>(A + offset3);
    b3.in = *reinterpret_cast<const uint4 *>(baseB + offset3);

    float part0 = 0.0f;
    float part1 = 0.0f;
    float part2 = 0.0f;
    float part3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        part0 += (float)a0.out[i] * (float)b0.out[i];
        part1 += (float)a1.out[i] * (float)b1.out[i];
        part2 += (float)a2.out[i] * (float)b2.out[i];
        part3 += (float)a3.out[i] * (float)b3.out[i];
    }

    // Emulate legacy strides 128 and 64 before reducing the 64 live values.
    float diff = 0.0f;
    float other = part2 - diff;
    float value0 = part0 + other;
    diff = (value0 - part0) - other;
    float diff1 = 0.0f;
    other = part3 - diff1;
    float value1 = part1 + other;
    other = value1 - diff;
    float value = value0 + other;
    diff = (value - value0) - other;

    __shared__ float sdata[THREADS];
    sdata[tid] = value;
    __syncthreads();

    if (tid < 32) {
        other = sdata[tid + 32] - diff;
        float sumTmp = value + other;
        diff = (sumTmp - value) - other;
        value = sumTmp;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float laneOther = __shfl_down_sync(0xffffffffu, value, offset);
            if (tid < offset) {
                other = laneOther - diff;
                sumTmp = value + other;
                diff = (sumTmp - value) - other;
                value = sumTmp;
            }
        }
        if (tid == 0) {
            if (!isSharedGate && bias != nullptr) {
                value += (float)__ldg(bias + row);
            }
            half *output = isSharedGate ? sharedGateC : C + row;
            half outputValue = (!isSharedGate && addTo)
                ? (half)(value + (float)*output)
                : (half)value;
            if (isSharedGate && sigmoidSharedGate) {
#ifdef CUDA_NO_TENSOR_CORE
                outputValue = __float2half(
                    1.0f / (1.0f + expf(-__half2float(outputValue))));
#else
                outputValue = __hdiv(
                    1.0, __hadd(__float2half(1.0), hexp(-outputValue)));
#endif
            }
            *output = outputValue;
        }
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow(half *A, half *B, half *C, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half8 regA;
    union_half8 regB;

    int p = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata[x][tid] = 0;
    }

    const half *baseB = B + p * m;
    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1]);
                if (i + 4 < m)
                    sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2]);
                if (i + 5 < m)
                    sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2]);
                if (i + 6 < m)
                    sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3]);
                if (i + 7 < m)
                    sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += (float)A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();

    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        diff[x] = 0.0f;
    }
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
#ifdef CUDA_NO_TENSOR_CORE
            C[p + k * x] = __float2half(__half2float(C[p + k * x]) + __half2float(__float2half_rn(sdata[x][0])));
#else
            C[p + k * x] = __hadd(C[p + k * x], __float2half_rn(sdata[x][0]));
#endif
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART, int FIXED_INPUT_SIZE = 0>
__global__ void FastllmGemvFp32Fp16Kernel2MultiRow(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    float4 regA;
    union_half4 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
        
    const int inputSize = FIXED_INPUT_SIZE > 0 ? FIXED_INPUT_SIZE : m;
    const half *baseB = B + p * inputSize;
    if (FIXED_INPUT_SIZE > 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < FIXED_INPUT_SIZE;
             i += THREAD_PER_BLOCK * 4) {
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = FETCH_FLOAT4(A[i + x * FIXED_INPUT_SIZE]);
                float sum = 0.0f;
                sum += regA.x * __low2float(regB.out2[0]);
                sum += regA.y * __high2float(regB.out2[0]);
                sum += regA.z * __low2float(regB.out2[1]);
                sum += regA.w * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = FETCH_FLOAT4(A[i + x * m]);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += regA.x * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += regA.y * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += regA.z * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += regA.w * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        diff[x] = 0.0f;
    }
    // Keep the legacy compensated tree through stride 32. The remaining
    // five stages involve only the first warp, so shuffles reproduce the
    // same pairwise additions without paying for five block-wide barriers.
    for (unsigned int s = THREAD_PER_BLOCK / 2; s >= 32; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        float values[PART];
#pragma unroll
        for (int x = 0; x < PART; x++) {
            values[x] = sdata[x][tid];
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float rhs = __shfl_down_sync(
                    0xffffffffu, values[x], s);
                if (tid < (unsigned int)s) {
                    float other = rhs - diff[x];
                    float sumTmp = values[x] + other;
                    diff[x] = (sumTmp - values[x]) - other;
                    values[x] = sumTmp;
                }
            }
        }
        if (tid == 0) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][0] = values[x];
            }
        }
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] + __ldg(bias + p);
        }
    }
}

// The two Qwen4 hyper-connection projections consume the same activation and
// are immediately followed by independent elementwise transforms.  Combining
// their output-row grids removes three launches without changing the GEMV
// lane mapping or compensated reduction tree used by the regular FP32xFP16
// path.  This kernel uses only portable CUDA primitives; unsupported batch
// sizes retain the regular two-Linear executor path below.
template <int THREAD_PER_BLOCK, int PART, int FIXED_INPUT_SIZE = 0>
__global__ void FastllmGemvFp32Fp16HyperProjectKernel(
        const float *A, const half *downWeight, const half *injectionWeight,
        float *activated, float *injection,
        const float *downBias, const float *injectionBias,
        int m, int downRows, int injectionRows, int groups) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    const unsigned int tid = threadIdx.x;
    const int combinedRow = blockIdx.x;
    const bool isInjection = combinedRow >= downRows;
    const int row = isInjection ? combinedRow - downRows : combinedRow;
    if (row >= (isInjection ? injectionRows : downRows)) {
        return;
    }

#pragma unroll
    for (int x = 0; x < PART; ++x) {
        sdata[x][tid] = 0.0f;
    }
    const int inputSize = FIXED_INPUT_SIZE > 0 ? FIXED_INPUT_SIZE : m;
    const half *baseWeight = isInjection
        ? injectionWeight + (size_t)row * inputSize
        : downWeight + (size_t)row * inputSize;
    if (FIXED_INPUT_SIZE > 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < FIXED_INPUT_SIZE;
             i += THREAD_PER_BLOCK * 4) {
            union_half4 packedWeight;
            packedWeight.in =
                *reinterpret_cast<const uint2 *>(baseWeight + i);
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                const float4 packedInput =
                    *reinterpret_cast<const float4 *>(
                        A + x * FIXED_INPUT_SIZE + i);
                float sum = 0.0f;
                sum += packedInput.x * __low2float(packedWeight.out2[0]);
                sum += packedInput.y * __high2float(packedWeight.out2[0]);
                sum += packedInput.z * __low2float(packedWeight.out2[1]);
                sum += packedInput.w * __high2float(packedWeight.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else if ((m & 3) == 0) {
        for (int i = tid * 4; i + 3 < m;
             i += THREAD_PER_BLOCK * 4) {
            union_half4 packedWeight;
            packedWeight.in =
                *reinterpret_cast<const uint2 *>(baseWeight + i);
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                const float4 packedInput =
                    *reinterpret_cast<const float4 *>(A + x * m + i);
                float sum = 0.0f;
                sum += packedInput.x * __low2float(packedWeight.out2[0]);
                sum += packedInput.y * __high2float(packedWeight.out2[0]);
                sum += packedInput.z * __low2float(packedWeight.out2[1]);
                sum += packedInput.w * __high2float(packedWeight.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                sdata[x][tid] += A[i + x * m] * (float)baseWeight[i];
            }
        }
    }
    __syncthreads();

    float difference[PART];
#pragma unroll
    for (int x = 0; x < PART; ++x) {
        difference[x] = 0.0f;
    }
    for (unsigned int stride = THREAD_PER_BLOCK / 2;
         stride >= 32; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                const float other =
                    sdata[x][tid + stride] - difference[x];
                const float sum = sdata[x][tid] + other;
                difference[x] = (sum - sdata[x][tid]) - other;
                sdata[x][tid] = sum;
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        float values[PART];
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            values[x] = sdata[x][tid];
        }
#pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                const float rhs = __shfl_down_sync(
                    0xffffffffu, values[x], stride);
                if (tid < (unsigned int)stride) {
                    const float other = rhs - difference[x];
                    const float sum = values[x] + other;
                    difference[x] = (sum - values[x]) - other;
                    values[x] = sum;
                }
            }
        }
        if (tid == 0) {
            const float bias = isInjection
                ? __ldg(injectionBias + row)
                : __ldg(downBias + row);
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                // __fadd_rn retains the materialized Linear output's rounding
                // boundary before applying either activation.
                const float projected = __fadd_rn(values[x], bias);
                if (isInjection) {
                    const float scaled = projected / groups;
                    const float gate = 1.0 / (1.0 + expf(-scaled));
                    injection[row + (size_t)injectionRows * x] =
                        gate * 2.0f;
                } else {
                    const float scale = 1.0f / groups;
                    const float scaled = __fmul_rn(projected, scale);
                    activated[row + (size_t)downRows * x] =
                        scaled / (1.0 + expf(-scaled));
                }
            }
        }
    }
}

// The Qwen hyper-connection up projection is a particularly skinny
// FP32xFP16 GEMV (M=320) repeated for four verifier rows. The generic kernel
// assigns one 256-thread CTA to every output even though only 80 legacy lanes
// load data. One warp below reproduces those 80 lanes locally, including the
// compensated 64->32 and final warp reduction order, and a CTA handles eight
// adjacent outputs. This is a shape specialization only; all other devices
// and shapes retain the generic path.
template <int WARPS_PER_BLOCK = 8>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32)
void FastllmGemvFp32Fp16M320MultiRow4WarpRowsKernel(
        const float *A, const half *B, float *C, const float *bias, int k) {
    constexpr int PART = 4;
    constexpr int INPUT_SIZE = 320;
    constexpr int VALUES_PER_LOAD = 4;
    constexpr int VIRTUAL_LANES = 3;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= k) {
        return;
    }

    const half *baseB = B + (size_t)row * INPUT_SIZE;
    float values[PART][VIRTUAL_LANES];
#pragma unroll
    for (int virtualLane = 0; virtualLane < VIRTUAL_LANES;
         ++virtualLane) {
        const int i =
            (lane + virtualLane * 32) * VALUES_PER_LOAD;
        if (i + VALUES_PER_LOAD <= INPUT_SIZE) {
            union_half4 regB;
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                float4 regA = *reinterpret_cast<const float4 *>(
                    A + x * INPUT_SIZE + i);
                float sum = 0.0f;
                sum += regA.x * __low2float(regB.out2[0]);
                sum += regA.y * __high2float(regB.out2[0]);
                sum += regA.z * __low2float(regB.out2[1]);
                sum += regA.w * __high2float(regB.out2[1]);
                values[x][virtualLane] = sum;
            }
        } else {
#pragma unroll
            for (int x = 0; x < PART; ++x) {
                values[x][virtualLane] = 0.0f;
            }
        }
    }

    float reduced[PART];
    float differences[PART];
#pragma unroll
    for (int x = 0; x < PART; ++x) {
        reduced[x] = values[x][0];
        differences[x] = 0.0f;
        // Legacy stride 128 is an all-zero addition for M=320. These are
        // exactly its stride-64 and stride-32 pairs for lanes 0..31.
        float other = values[x][2] - differences[x];
        float sum = reduced[x] + other;
        differences[x] = (sum - reduced[x]) - other;
        reduced[x] = sum;
        other = values[x][1] - differences[x];
        sum = reduced[x] + other;
        differences[x] = (sum - reduced[x]) - other;
        reduced[x] = sum;
    }

    const unsigned int mask = __activemask();
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            float rhs = __shfl_down_sync(mask, reduced[x], stride);
            if (lane < stride) {
                float other = rhs - differences[x];
                float sum = reduced[x] + other;
                differences[x] = (sum - reduced[x]) - other;
                reduced[x] = sum;
            }
        }
    }

    if (lane == 0) {
        const float biasValue =
            bias == nullptr ? 0.0f : __ldg(bias + row);
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            C[row + (size_t)k * x] = reduced[x] + biasValue;
        }
    }
}

// Companion specialization for M=640. Five virtual 32-lane groups represent
// the 160 active lanes of the legacy CTA; zero-filled groups reproduce the
// unused upper lanes in the 256-thread compensated reduction tree.
template <int WARPS_PER_BLOCK = 8>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32)
void FastllmGemvFp32Fp16M640MultiRow4WarpRowsKernel(
        const float *A, const half *B, float *C, const float *bias, int k) {
    constexpr int PART = 4;
    constexpr int INPUT_SIZE = 640;
    constexpr int VALUES_PER_LOAD = 4;
    constexpr int VIRTUAL_LANES = 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= k) {
        return;
    }

    const half *baseB = B + (size_t)row * INPUT_SIZE;
    float values[PART][VIRTUAL_LANES];
#pragma unroll
    for (int virtualLane = 0; virtualLane < VIRTUAL_LANES;
         ++virtualLane) {
        const int i =
            (lane + virtualLane * 32) * VALUES_PER_LOAD;
        union_half4 regB;
        regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            float4 regA = *reinterpret_cast<const float4 *>(
                A + x * INPUT_SIZE + i);
            float sum = 0.0f;
            sum += regA.x * __low2float(regB.out2[0]);
            sum += regA.y * __high2float(regB.out2[0]);
            sum += regA.z * __low2float(regB.out2[1]);
            sum += regA.w * __high2float(regB.out2[1]);
            values[x][virtualLane] = sum;
        }
    }

    float reduced[PART];
    float differences[PART];
#pragma unroll
    for (int x = 0; x < PART; ++x) {
        // Stride 128: legacy lane 0 adds lane 128. Lanes 32..127
        // add zeros, so their values remain v1/v2/v3.
        reduced[x] = values[x][0];
        differences[x] = 0.0f;
        float other = values[x][4] - differences[x];
        float sum = reduced[x] + other;
        differences[x] = (sum - reduced[x]) - other;
        reduced[x] = sum;

        // Stride 64 updates both halves that will meet at stride 32.
        other = values[x][2] - differences[x];
        sum = reduced[x] + other;
        differences[x] = (sum - reduced[x]) - other;
        reduced[x] = sum;

        float upperDifference = 0.0f;
        other = values[x][3] - upperDifference;
        float upper = values[x][1] + other;
        upperDifference = (upper - values[x][1]) - other;

        // Stride 32 consumes the upper value, not its compensation state.
        other = upper - differences[x];
        sum = reduced[x] + other;
        differences[x] = (sum - reduced[x]) - other;
        reduced[x] = sum;
    }

    const unsigned int mask = __activemask();
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            float rhs = __shfl_down_sync(mask, reduced[x], stride);
            if (lane < stride) {
                float other = rhs - differences[x];
                float sum = reduced[x] + other;
                differences[x] = (sum - reduced[x]) - other;
                reduced[x] = sum;
            }
        }
    }

    if (lane == 0) {
        const float biasValue =
            bias == nullptr ? 0.0f : __ldg(bias + row);
#pragma unroll
        for (int x = 0; x < PART; ++x) {
            C[row + (size_t)k * x] = reduced[x] + biasValue;
        }
    }
}

static void FastllmCudaFP16EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
}

static void FastllmCudaFP16EnsureBiasHalfOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }
}

void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k) {
    if (n == 1) {
        // With four input elements per thread, m <= 512 needs at most 128
        // active lanes. The old 256-thread launch reduced an all-zero upper
        // half before doing exactly the same 128-lane reduction. Keeping the
        // load mapping and reduction tree below 128 unchanged preserves the
        // float accumulation order while doubling resident blocks for small
        // decode GEMVs such as HyperConnection's 320 -> hidden projection.
        if (m <= 512 && m % 4 == 0) {
            FastllmGemvFp32Fp16Kernel2MultiRow<128, 1> <<< k, 128 >>>(input, weight, output, bias, m, k);
        } else {
            FastllmGemvFp32Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
        }
    } else if (n == 2) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 4 && m == 320) {
        FastllmGemvFp32Fp16M320MultiRow4WarpRowsKernel<8>
            <<<(k + 7) / 8, 256>>>(input, weight, output, bias, k);
    } else if (n == 4 && m == 640) {
        FastllmGemvFp32Fp16M640MultiRow4WarpRowsKernel<8>
            <<<(k + 7) / 8, 256>>>(input, weight, output, bias, k);
    } else if (n == 4 && m == 2560 && k <= 2560) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 4, 2560>
            <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp32Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, m, k);
        }
        return;

        printf("Error: LaunchFastllmGemmFp32Fp16: n > 7.\n");
        exit(0);
    }
}

bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP16EnsureBiasOnDevice(weight, bias, k);
    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n < 8) {
        LaunchFastllmGemmFp32Fp16(cudaInput, (half*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
        half *cudaFp16Input, *cudaFp16Output;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
        FastllmCudaFree(cudaFp16Input);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
        //cudaDeviceSynchronize();

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
#endif
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// Run two FP16 projections of the same activation after materializing a shared
// FP16 input when needed. This follows FastllmCudaMatMulFloat16's prefill
// arithmetic exactly; only the duplicate input conversion is removed.
static bool FastllmCudaMatMulFloat16PairSharedInput(
        const fastllm::Data &input,
        fastllm::Data &firstWeight, fastllm::Data &firstOutput,
        fastllm::Data &secondWeight, fastllm::Data &secondOutput,
        int n, int m, int firstK, int secondK) {
    void *cudaInput = FastllmCudaPrepareInput(input);
    float *cudaFirstOutput = (float*)FastllmCudaPrepareOutput(firstOutput);
    float *cudaSecondOutput = (float*)FastllmCudaPrepareOutput(secondOutput);
    if (cudaInput == nullptr || cudaFirstOutput == nullptr ||
        cudaSecondOutput == nullptr) {
        if (cudaInput != nullptr) {
            FastllmCudaFinishInput(input, cudaInput);
        }
        if (cudaFirstOutput != nullptr) {
            FastllmCudaFinishOutput(firstOutput, cudaFirstOutput);
        }
        if (cudaSecondOutput != nullptr) {
            FastllmCudaFinishOutput(secondOutput, cudaSecondOutput);
        }
        return false;
    }

    const bool convertInput = input.dataType == fastllm::DataType::FLOAT32;
    half *cudaFp16Input = convertInput
        ? (half*)FastllmCudaMalloc((size_t)n * m * sizeof(half))
        : (half*)cudaInput;
    if (cudaFp16Input == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(firstOutput, cudaFirstOutput);
        FastllmCudaFinishOutput(secondOutput, cudaSecondOutput);
        return false;
    }
    if (convertInput) {
        const int inputLength = n * m;
        const int threads = std::min(256, inputLength);
        FastllmCudaFloat2HalfKernel<<<
            (inputLength - 1) / threads + 1, threads>>>(
            (float*)cudaInput, cudaFp16Input, inputLength);
    }

    auto handle = getFastllmCublasHandle();
    auto project = [&](fastllm::Data &weight, float *output, int k) {
#ifdef CUDA_NO_TENSOR_CORE
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
            &alpha, (half*)weight.cudaData, CUDA_R_16F, m,
            cudaFp16Input, CUDA_R_16F, m, &beta,
            output, CUDA_R_32F, k, CUDA_R_32F,
            static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        return status == CUBLAS_STATUS_SUCCESS;
#else
        half *cudaFp16Output =
            (half*)FastllmCudaMalloc((size_t)n * k * sizeof(half));
        if (cudaFp16Output == nullptr) {
            return false;
        }
        const half alpha = __float2half_rn(1.0f);
        const half beta = __float2half_rn(0.0f);
        const cublasStatus_t status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
            &alpha, (half*)weight.cudaData, CUDA_R_16F, m,
            cudaFp16Input, CUDA_R_16F, m, &beta,
            cudaFp16Output, CUDA_R_16F, k, CUDA_R_16F,
            static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status == CUBLAS_STATUS_SUCCESS) {
            const int outputLength = n * k;
            const int outputThreads = std::min(256, outputLength);
            FastllmCudaHalf2FloatKernel<<<
                (outputLength - 1) / outputThreads + 1, outputThreads>>>(
                cudaFp16Output, output, outputLength);
        }
        FastllmCudaFree(cudaFp16Output);
        return status == CUBLAS_STATUS_SUCCESS;
#endif
    };

    const bool success =
        project(firstWeight, cudaFirstOutput, firstK) &&
        project(secondWeight, cudaSecondOutput, secondK);
    if (convertInput) {
        FastllmCudaFree(cudaFp16Input);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(firstOutput, cudaFirstOutput);
    FastllmCudaFinishOutput(secondOutput, cudaSecondOutput);
    return success && cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaQwen4HyperMixProjected(
        const fastllm::Data &normalized,
        const fastllm::Data &lowRank,
        fastllm::Data &upWeight,
        fastllm::Data &output, int groups) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    if (normalized.dataDevice != fastllm::DataDevice::CUDA ||
        lowRank.dataDevice != fastllm::DataDevice::CUDA ||
        upWeight.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        normalized.dataType != fastllm::DataType::FLOAT32 ||
        lowRank.dataType != fastllm::DataType::FLOAT32 ||
        upWeight.dataType != fastllm::DataType::FLOAT16 ||
        normalized.dims.empty() || lowRank.dims.empty() ||
        upWeight.dims.size() != 2 || groups <= 0 ||
        normalized.dims.back() % groups != 0 ||
        lowRank.dims.back() != upWeight.dims[1] ||
        normalized.dims.back() != upWeight.dims[0]) {
        return false;
    }
    const int m = lowRank.dims.back();
    const int n = (int)(lowRank.Count(0) / m);
    const int k = upWeight.dims[0];
    if (n < 8 || normalized.Count(0) != (uint64_t)n * k) {
        return false;
    }

    float *cudaInput = (float*)FastllmCudaPrepareInput(lowRank);
    half *cudaFp16Input =
        (half*)FastllmCudaMalloc((size_t)n * m * sizeof(half));
    fastllm::Data projected(fastllm::DataType::FLOAT16);
    projected.Resize(normalized.dims);
    projected.ToDevice(fastllm::DataDevice::CUDA,
                       normalized.dataDeviceIds);
    projected.Allocate(false);
    half *cudaProjected =
        (half*)FastllmCudaPrepareOutput(projected);
    if (cudaInput == nullptr || cudaProjected == nullptr ||
        cudaFp16Input == nullptr) {
        if (cudaInput != nullptr) {
            FastllmCudaFinishInput(lowRank, cudaInput);
        }
        if (cudaProjected != nullptr) {
            FastllmCudaFinishOutput(projected, cudaProjected);
        }
        if (cudaFp16Input != nullptr) {
            FastllmCudaFree(cudaFp16Input);
        }
        return false;
    }

    const int inputLength = n * m;
    const int threads = std::min(256, inputLength);
    FastllmCudaFloat2HalfKernel<<<
        (inputLength - 1) / threads + 1, threads>>>(
        cudaInput, cudaFp16Input, inputLength);
    const half alpha = __float2half_rn(1.0f);
    const half beta = __float2half_rn(0.0f);
    const cublasStatus_t status = cublasGemmEx(
        getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
        k, n, m, &alpha,
        (half*)upWeight.cudaData, CUDA_R_16F, m,
        cudaFp16Input, CUDA_R_16F, m, &beta,
        cudaProjected, CUDA_R_16F, k, CUDA_R_16F,
        static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    FastllmCudaFree(cudaFp16Input);
    FastllmCudaFinishInput(lowRank, cudaInput);
    FastllmCudaFinishOutput(projected, cudaProjected);
    if (status != CUBLAS_STATUS_SUCCESS ||
        cudaGetLastError() != cudaSuccess) {
        return false;
    }
    output.Allocate(false);
    return FastllmCudaQwen4HyperMixPromotedFloatLogits(
        normalized, projected, output, groups);
#endif
}

bool FastllmCudaQwen4HyperProject(
        const fastllm::Data &input, fastllm::Data &downWeight,
        fastllm::Data &injectionWeight, fastllm::Data &activated,
        fastllm::Data &injection, int groups) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        downWeight.dataDevice != fastllm::DataDevice::CUDA ||
        injectionWeight.dataDevice != fastllm::DataDevice::CUDA ||
        activated.dataDevice != fastllm::DataDevice::CUDA ||
        injection.dataDevice != fastllm::DataDevice::CUDA ||
        (input.dataType != fastllm::DataType::FLOAT32 &&
         input.dataType != fastllm::DataType::FLOAT16) ||
        downWeight.dataType != fastllm::DataType::FLOAT16 ||
        injectionWeight.dataType != fastllm::DataType::FLOAT16 ||
        activated.dataType != fastllm::DataType::FLOAT32 ||
        injection.dataType != fastllm::DataType::FLOAT32 ||
        input.dims.empty() || downWeight.dims.size() != 2 ||
        injectionWeight.dims.size() != 2 || groups <= 0 ||
        input.dims.back() != downWeight.dims[1] ||
        input.dims.back() != injectionWeight.dims[1] ||
        injectionWeight.dims[0] != groups || input.cudaData == nullptr ||
        downWeight.cudaData == nullptr || injectionWeight.cudaData == nullptr ||
        activated.cudaData == nullptr || injection.cudaData == nullptr) {
        return false;
    }

    const int m = input.dims.back();
    const int n = (int)(input.Count(0) / m);
    const int downRows = downWeight.dims[0];
    const int injectionRows = injectionWeight.dims[0];
    if (activated.Count(0) != (uint64_t)n * downRows ||
        injection.Count(0) != (uint64_t)n * injectionRows) {
        return false;
    }

    // Prefill keeps the tuned GEMM implementations. Small float32
    // decode/verifier batches use the fused row grid, which exactly mirrors
    // the native GEMV; prepared float16 storage is a prefill-only input.
    if (n >= 8) {
        if (!FastllmCudaMatMulFloat16PairSharedInput(
                input, downWeight, activated,
                injectionWeight, injection,
                n, m, downRows, injectionRows) ||
            !FastllmCudaQwen4HyperPrepare(
                activated, activated, groups) ||
            !FastllmCudaQwen4HyperInject(
                injection, injection, groups)) {
            return false;
        }
        return true;
    }
    if (n <= 0 || input.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }

    fastllm::Data emptyBias;
    FastllmCudaFP16EnsureBiasOnDevice(downWeight, emptyBias, downRows);
    FastllmCudaFP16EnsureBiasOnDevice(
        injectionWeight, emptyBias, injectionRows);
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaActivated = (float*)FastllmCudaPrepareOutput(activated);
    float *cudaInjection = (float*)FastllmCudaPrepareOutput(injection);
    float *downBias = (float*)downWeight.extraCudaData[0];
    float *injectionBias = (float*)injectionWeight.extraCudaData[0];

#define FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(PART_VALUE) \
    FastllmGemvFp32Fp16HyperProjectKernel<256, PART_VALUE> \
        <<<(downRows + injectionRows), 256>>>( \
            cudaInput, (const half*)downWeight.cudaData, \
            (const half*)injectionWeight.cudaData, \
            cudaActivated, cudaInjection, downBias, injectionBias, \
            m, downRows, injectionRows, groups)
#define FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH(PART_VALUE) \
    FastllmGemvFp32Fp16HyperProjectKernel<256, PART_VALUE, 10240> \
        <<<(downRows + injectionRows), 256>>>( \
            cudaInput, (const half*)downWeight.cudaData, \
            (const half*)injectionWeight.cudaData, \
            cudaActivated, cudaInjection, downBias, injectionBias, \
            m, downRows, injectionRows, groups)
    switch (n) {
        case 1:
            if (m == 10240) {
                FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH(1);
            } else {
                FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(1);
            }
            break;
        case 2:
            if (m == 10240) {
                FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH(2);
            } else {
                FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(2);
            }
            break;
        case 3:
            if (m == 10240) {
                FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH(3);
            } else {
                FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(3);
            }
            break;
        case 4:
            if (m == 10240) {
                FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH(4);
            } else {
                FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(4);
            }
            break;
        case 5: FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(5); break;
        case 6: FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(6); break;
        case 7: FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH(7); break;
        default: break;
    }
#undef FASTLLM_QWEN4_HYPER_PROJECT_FIXED_LAUNCH
#undef FASTLLM_QWEN4_HYPER_PROJECT_LAUNCH

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(activated, cudaActivated);
    FastllmCudaFinishOutput(injection, cudaInjection);
    return cudaGetLastError() == cudaSuccess;
}

void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias,
                               int n, int m, int k, bool addTo,
                               bool allowRouterSpecialization) {
    // DFlash verification must retain the compensated reduction state of each
    // q1 row independently.  A PART=n launch still lets those rows share each
    // weight read without changing their per-row accumulation order.
    const bool exactRows = n > 1 && n <= 8 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold();
#define FASTLLM_FP16_EXACT_LAUNCH(PARTVAL) \
    FastllmGemvFp16Fp16Kernel2MultiRow<256, PARTVAL, false, true> \
        <<<k, 256>>>(input, weight, output, bias, m, k, addTo, \
                     nullptr, nullptr, false)
    if (exactRows) {
        switch (n) {
            case 2: FASTLLM_FP16_EXACT_LAUNCH(2); return;
            case 3: FASTLLM_FP16_EXACT_LAUNCH(3); return;
            case 4: FASTLLM_FP16_EXACT_LAUNCH(4); return;
            case 5: FASTLLM_FP16_EXACT_LAUNCH(5); return;
            case 6: FASTLLM_FP16_EXACT_LAUNCH(6); return;
            case 7: FASTLLM_FP16_EXACT_LAUNCH(7); return;
            case 8: FASTLLM_FP16_EXACT_LAUNCH(8); return;
            default: break;
        }
    }
#undef FASTLLM_FP16_EXACT_LAUNCH
    if (allowRouterSpecialization && n == 1 && m == 2048 && k == 256 &&
        bias == nullptr && !addTo) {
        FastllmGemvFp16Fp16Router2048x256Kernel<false><<<k, 64>>>(
            input, weight, output, bias, addTo);
    } else if (allowRouterSpecialization && n == 1 &&
               m == 2048 && k == 2048) {
        FastllmGemvFp16Fp16WarpRowsKernel<2048, 4>
            <<< (k + 3) / 4, 128 >>>(
                input, weight, output, bias, k, addTo);
    } else if (allowRouterSpecialization && n == 1 &&
               m == 256 && k == 2048) {
        FastllmGemvFp16Fp16WarpRowsKernel<256, 4>
            <<< (k + 3) / 4, 128 >>>(
                input, weight, output, bias, k, addTo);
    } else if (n == 1) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 1, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 2) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 2, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 3) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 3, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 4) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 4, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 5) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 5, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 6) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 6, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 7) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 7, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else if (n == 8) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 8, false>
            <<<k, 256>>>(input, weight, output, bias, m, k, addTo,
                         nullptr, nullptr, false);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16: n > 8.\n");
        exit(0);
    }
}

void LaunchFastllmGemmFp16Fp16AddToNoBias(half *input, half *weight, half *output, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Fp16AddToNoBiasKernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, m, k);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16AddToNoBias: n > 7.\n");
        exit(0);
    }
}

FastllmCudaLinearFp16Path FastllmCudaResolveLinearFp16AutoPath(
        int n, int m, int k, bool addTo, bool hasBias) {
    if (n >= 1 && n <= 8 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        return FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE;
    }
    if (n >= 1 && n < 8) {
        return FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE;
    }
    // Qwen3.6's B8 GDN projection is the only N=8 shape where the native
    // multi-row kernel has been verified to beat cuBLAS in both eager and
    // CUDA Graph execution. Keep every other N=8 shape on the established
    // cuBLAS path instead of extrapolating from eager launch overhead.
    if (n == 8 && m == 5120 && k == 48 && !addTo && !hasBias) {
        return FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE;
    }
    return FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS;
}

namespace {

    static bool RunFastllmCudaLinearFp16Cublas(
            half *input, half *weight, half *output, half *bias,
            int n, int m, int k, bool addTo) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output =
            (float *)FastllmCudaMalloc((size_t)n * k * sizeof(float));
        if (cudaFp32Output == nullptr) {
            return false;
        }
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F;
        cudaDataType_t CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        status = cublasGemmEx(
            fastllmCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            k, n, m, &h_alpha, weight, AType, m, input, BType, m,
            &h_beta, cudaFp32Output, CType, k, ComputeType,
            static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        __half h_alpha = __float2half_rn(1.0f);
        __half h_beta = addTo ? __float2half_rn(1.0f)
                              : __float2half_rn(0.0f);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F;
        cudaDataType_t CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        status = cublasGemmEx(
            fastllmCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            k, n, m, &h_alpha, weight, AType, m, input, BType, m,
            &h_beta, output, CType, k, ComputeType,
            static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "FastLLM FP16 cuBLAS GEMM failed: status=%d, "
                    "shape=(n=%d,m=%d,k=%d), addTo=%d, graphCapture=%d.\n",
                    (int)status, n, m, k, addTo ? 1 : 0,
                    FastllmCudaGraphIsCapturingFast() ? 1 : 0);
            fflush(stderr);
#ifdef CUDA_NO_TENSOR_CORE
            FastllmCudaFree(cudaFp32Output);
#endif
            return false;
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        if (addTo) {
            half *cudaTempOutput =
                (half *)FastllmCudaMalloc((size_t)len * sizeof(half));
            if (cudaTempOutput == nullptr) {
                FastllmCudaFree(cudaFp32Output);
                return false;
            }
            FastllmCudaFloat2HalfKernel
                <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                    cudaFp32Output, cudaTempOutput, len);
            FastllmAddToKernel
                <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                    output, cudaTempOutput, __float2half_rn(1.0f), len);
            FastllmCudaFree(cudaTempOutput);
        } else {
            FastllmCudaFloat2HalfKernel
                <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(
                    cudaFp32Output, output, len);
        }
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias != nullptr) {
            FastllmCudaBiasKernel<<<n, 256>>>(output, bias, k);
        }
        return true;
    }

    static bool RunFastllmCudaLinearFp16Native(
            half *input, half *weight, half *output, half *bias,
            int n, int m, int k, bool addTo,
            bool allowRouterSpecialization) {
        LaunchFastllmGemmFp16Fp16(
            input, weight, output, bias, n, m, k, addTo,
            allowRouterSpecialization);
        cudaError_t state = cudaPeekAtLastError();
        if (state != cudaSuccess) {
            fprintf(stderr,
                    "FastLLM native FP16 GEMM launch failed: cuda=%d (%s), "
                    "shape=(n=%d,m=%d,k=%d), addTo=%d, graphCapture=%d.\n",
                    (int)state, cudaGetErrorString(state), n, m, k,
                    addTo ? 1 : 0,
                    FastllmCudaGraphIsCapturingFast() ? 1 : 0);
            fflush(stderr);
            cudaGetLastError();
            return false;
        }
        return true;
    }

    static void ThrowFastllmCudaLinearFp16CublasError() {
        printf("Error: cublas error.\n");
        throw("cublas error");
    }
}

bool FastllmCudaHalfMatMulFloat16WithPath(
                                 const fastllm::Data &input, fastllm::Data &weight,
                                 const fastllm::Data &bias, fastllm::Data &output,
                                 int n, int m, int k, bool addTo,
                                 bool allowRouterSpecialization,
                                 FastllmCudaLinearFp16Path path) {
    FastllmCudaFP16EnsureBiasOnDevice(weight, bias, k);
    FastllmCudaFP16EnsureBiasHalfOnDevice(weight, bias, k);

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];

    bool ok = true;
    if (path == FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE) {
        ok = n >= 1 && n <= 8 && RunFastllmCudaLinearFp16Native(
            cudaInput, (half *)weight.cudaData, cudaOutput, cudaBiasData,
            n, m, k, addTo, allowRouterSpecialization);
    } else if (path == FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS) {
        ok = RunFastllmCudaLinearFp16Cublas(
            cudaInput, (half *)weight.cudaData, cudaOutput, cudaBiasData,
            n, m, k, addTo);
    } else if (path == FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO) {
        FastllmCudaLinearFp16Path autoPath =
            FastllmCudaResolveLinearFp16AutoPath(
                n, m, k, addTo, cudaBiasData != nullptr);
        if (autoPath == FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE) {
            ok = RunFastllmCudaLinearFp16Native(
                cudaInput, (half *)weight.cudaData, cudaOutput, cudaBiasData,
                n, m, k, addTo, allowRouterSpecialization);
        }
        if (autoPath == FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS || !ok) {
            ok = RunFastllmCudaLinearFp16Cublas(
                cudaInput, (half *)weight.cudaData, cudaOutput, cudaBiasData,
                n, m, k, addTo);
        }
    } else {
        ok = false;
    }
    if (!ok && path == FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO) {
        ThrowFastllmCudaLinearFp16CublasError();
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return ok;
}

bool FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                                 const fastllm::Data &input, fastllm::Data &weight,
                                 const fastllm::Data &bias, fastllm::Data &output,
                                 int n, int m, int k, bool addTo,
                                 bool allowRouterSpecialization) {
    return FastllmCudaHalfMatMulFloat16WithPath(
        input, weight, bias, output, n, m, k, addTo,
        allowRouterSpecialization, FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO);
}

bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight,
                                 const fastllm::Data &bias, fastllm::Data &output,
                                 int n, int m, int k, bool addTo) {
    return FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
        input, weight, bias, output, n, m, k, addTo, true);
}

bool FastllmCudaQwen35RouterSharedGateFloat16(
        const fastllm::Data &input,
        fastllm::Data &routerWeight,
        fastllm::Data &sharedGateWeight,
        fastllm::Data &routerOutput,
        fastllm::Data &sharedGateOutput,
        bool sigmoidSharedGate) {
    constexpr int HIDDEN = 2048;
    constexpr int EXPERTS = 256;
    constexpr int MAX_GEMV_ROWS = 7;
    const int inputCount = input.Count(0);
    const int rows = inputCount > 0 ? inputCount / HIDDEN : 0;
    if (input.dataType != fastllm::DataType::FLOAT16 ||
        routerWeight.dataType != fastllm::DataType::FLOAT16 ||
        sharedGateWeight.dataType != fastllm::DataType::FLOAT16 ||
        routerOutput.dataType != fastllm::DataType::FLOAT16 ||
        sharedGateOutput.dataType != fastllm::DataType::FLOAT16 ||
        input.dims.empty() ||
        inputCount <= 0 ||
        inputCount % HIDDEN != 0 ||
        rows > MAX_GEMV_ROWS ||
        input.dims.back() != HIDDEN ||
        routerWeight.dims.size() != 2 ||
        routerWeight.dims[0] != EXPERTS ||
        routerWeight.dims[1] != HIDDEN ||
        sharedGateWeight.dims.size() != 2 ||
        sharedGateWeight.dims[0] != 1 ||
        sharedGateWeight.dims[1] != HIDDEN ||
        routerOutput.Count(0) != rows * EXPERTS ||
        sharedGateOutput.Count(0) != rows ||
        routerWeight.cudaData == nullptr ||
        sharedGateWeight.cudaData == nullptr) {
        return false;
    }

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaRouterOutput = (half*)FastllmCudaPrepareOutput(routerOutput);
    half *cudaSharedGateOutput =
        (half*)FastllmCudaPrepareOutput(sharedGateOutput);
    if (rows == 1) {
        FastllmGemvFp16Fp16Router2048x256Kernel<true>
            <<<EXPERTS + 1, 64>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, false, (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else if (rows == 2) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 2, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else if (rows == 3) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 3, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else if (rows == 4) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 4, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else if (rows == 5) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 5, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else if (rows == 6) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 6, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    } else {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 7, true>
            <<<EXPERTS + 1, 256>>>(
                cudaInput, (half*)routerWeight.cudaData, cudaRouterOutput,
                nullptr, HIDDEN, EXPERTS, false,
                (half*)sharedGateWeight.cudaData,
                cudaSharedGateOutput, sigmoidSharedGate);
    }
    checkCudaErrors(
        "Error: CUDA error in FastllmCudaQwen35RouterSharedGateFloat16.",
        cudaGetLastError());
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(routerOutput, cudaRouterOutput);
    FastllmCudaFinishOutput(sharedGateOutput, cudaSharedGateOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloat16AddToNoBias(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, int n, int m, int k) {
    if (n >= 8 || input.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT16 || output.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);
    LaunchFastllmGemmFp16Fp16AddToNoBias(cudaInput, (half*)weight.cudaData, cudaOutput, n, m, k);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// Fused Linear + Swiglu kernel for FP16
// Each block computes one output element p (0 <= p < k).
// It reads two rows of the weight matrix: row p (gate) and row p+k (up),
// computes dot(input[x], weight_gate[p]) and dot(input[x], weight_up[p+k]),
// adds bias, then applies swiglu: output = silu(gate) * up = gate / (1 + exp(-gate)) * up
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Fp16SwigluKernel(half *A, half *B, half *C, half *bias, int m, int k) {
    // A: input,  shape [PART, m]
    // B: weight, shape [2*k, m], row-major. Row p is the gate row, row p+k is the up row.
    // C: output, shape [PART, k]
    // bias: [2*k] or nullptr
    // m: input dim,  k: output dim (after swiglu)
    __shared__ float sdata_gate[PART][THREAD_PER_BLOCK];
    __shared__ float sdata_up[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half8 regA;
    union_half8 regB_gate;
    union_half8 regB_up;

    int p = blockIdx.x; // output index, 0 <= p < k

#pragma unroll
    for (int x = 0; x < PART; x++) {
        sdata_gate[x][tid] = 0;
        sdata_up[x][tid] = 0;
    }

    const half *baseB_gate = B + p * m;        // gate row
    const half *baseB_up   = B + (p + k) * m;  // up row

    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                regB_gate.in = *reinterpret_cast<const uint4 *>(baseB_gate + i);
                regB_up.in = *reinterpret_cast<const uint4 *>(baseB_up + i);
                float sum_gate = 0.0f;
                float sum_up = 0.0f;
                sum_gate += __low2float(regA.out2[0]) * __low2float(regB_gate.out2[0]);
                sum_up   += __low2float(regA.out2[0]) * __low2float(regB_up.out2[0]);
                sum_gate += __high2float(regA.out2[0]) * __high2float(regB_gate.out2[0]);
                sum_up   += __high2float(regA.out2[0]) * __high2float(regB_up.out2[0]);
                sum_gate += __low2float(regA.out2[1]) * __low2float(regB_gate.out2[1]);
                sum_up   += __low2float(regA.out2[1]) * __low2float(regB_up.out2[1]);
                sum_gate += __high2float(regA.out2[1]) * __high2float(regB_gate.out2[1]);
                sum_up   += __high2float(regA.out2[1]) * __high2float(regB_up.out2[1]);
                sum_gate += __low2float(regA.out2[2]) * __low2float(regB_gate.out2[2]);
                sum_up   += __low2float(regA.out2[2]) * __low2float(regB_up.out2[2]);
                sum_gate += __high2float(regA.out2[2]) * __high2float(regB_gate.out2[2]);
                sum_up   += __high2float(regA.out2[2]) * __high2float(regB_up.out2[2]);
                sum_gate += __low2float(regA.out2[3]) * __low2float(regB_gate.out2[3]);
                sum_up   += __low2float(regA.out2[3]) * __low2float(regB_up.out2[3]);
                sum_gate += __high2float(regA.out2[3]) * __high2float(regB_gate.out2[3]);
                sum_up   += __high2float(regA.out2[3]) * __high2float(regB_up.out2[3]);
                sdata_gate[x][tid] += sum_gate;
                sdata_up[x][tid] += sum_up;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float a_val = (float)A[i + x * m];
                sdata_gate[x][tid] += a_val * (float)baseB_gate[i];
                sdata_up[x][tid] += a_val * (float)baseB_up[i];
            }
        }
    }
    __syncthreads();

    // Reduction
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata_gate[x][tid] += sdata_gate[x][tid + s];
                sdata_up[x][tid] += sdata_up[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float gate_val = sdata_gate[x][0];
            float up_val = sdata_up[x][0];
            if (bias != nullptr) {
                gate_val += (float)(__ldg(bias + p));
                up_val += (float)(__ldg(bias + p + k));
            }
            // swiglu: silu(gate) * up = gate / (1 + exp(-gate)) * up
            float silu_gate = gate_val / (1.0f + expf(-gate_val));
            C[p + k * x] = (half)(silu_gate * up_val);
        }
    }
    __syncthreads();
}

void LaunchFastllmGemmFp16Fp16Swiglu(half *input, half *weight, half *output, half *bias, int n, int m, int k) {
    // k is the output dim (after swiglu), weight has 2*k rows
    if (n == 1) {
        FastllmGemvFp16Fp16SwigluKernel<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Fp16SwigluKernel<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Fp16SwigluKernel<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Fp16SwigluKernel<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Fp16SwigluKernel<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Fp16SwigluKernel<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Fp16SwigluKernel<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16Swiglu: n > 7.\n");
        exit(0);
    }
}

bool FastllmCudaHalfMatMulFloat16Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (n >= 8) {
        return false;
    }

    output.Allocate(false);

    int biasK = k * 2;
    FastllmCudaFP16EnsureBiasOnDevice(weight, bias, biasK);
    FastllmCudaFP16EnsureBiasHalfOnDevice(weight, bias, biasK);

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    LaunchFastllmGemmFp16Fp16Swiglu(cudaInput, (half*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// ============ BF16 input × FP16 weight -> BF16 output ============

static void FastllmCudaFP16EnsureBiasBf16OnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    // Store BF16 bias in extraCudaData[1] (extraCudaData[0] is float bias)
    if (weight.cudaData == nullptr || weight.extraCudaData.size() < 2) {
        __nv_bfloat16 *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(__nv_bfloat16));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2Bf16Kernel <<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(__nv_bfloat16));
        }
        checkCudaErrors("Error: CUDA error when moving bias (bf16 for BF16×FP16) to device!", state);
        if (weight.extraCudaData.size() < 2)
            weight.extraCudaData.push_back((void *)cudaBiasData);
        else
            weight.extraCudaData[1] = (void *)cudaBiasData;
    }
}

// BF16 -> FP16 逐元素转换
__global__ void FastllmCudaBf16ToHalfKernelFP16(const __nv_bfloat16 *src, half *dst, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        dst[idx] = __float2half_rn(__bfloat162float(src[idx]));
}

void LaunchFastllmGemmBf16Fp16(__nv_bfloat16 *input, half *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvBf16Fp16Kernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvBf16Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

// BF16 input × FP16 weight -> BF16 output
bool FastllmCudaBFloat16MatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP16EnsureBiasOnDevice(weight, bias, k);
    FastllmCudaFP16EnsureBiasBf16OnDevice(weight, bias, k);

    __nv_bfloat16 *cudaInput = (__nv_bfloat16 *)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16 *)FastllmCudaPrepareOutput(output);
    __nv_bfloat16 *cudaBiasData = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *)weight.extraCudaData[1];
    half *weightPtr = (half *)weight.cudaData;

    if (n < 8) {
        LaunchFastllmGemmBf16Fp16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else {
        // 大 batch：将 BF16 input 转为 FP16，使用 cublas FP16 gemm，再将输出转为 BF16
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input = (half *)FastllmCudaMalloc(n * m * sizeof(half));
        if (cudaFp16Input == nullptr) {
            FastllmCudaSetThreadError();
            return false;
        }

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaBf16ToHalfKernelFP16 <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        cublasStatus_t status;
#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * k * sizeof(float));
        if (cudaFp32Output == nullptr) {
            FastllmCudaFree(cudaFp16Input);
            FastllmCudaSetThreadError();
            return false;
        }
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weightPtr, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp32Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        half *cudaFp16Output = (half *)FastllmCudaMalloc(n * k * sizeof(half));
        if (cudaFp16Output == nullptr) {
            FastllmCudaFree(cudaFp16Input);
            FastllmCudaSetThreadError();
            return false;
        }
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weightPtr, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (BFloat16MatMulFloat16).\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        // FP32 output -> BF16 output
        len = n * k;
        threadPerBlock = std::min(256, len);
        FastllmCudaFloat2Bf16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp32Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, (__nv_bfloat16 *)weight.extraCudaData[1], k);
        }
        FastllmCudaFree(cudaFp32Output);
#else
        // FP16 output -> BF16 output
        len = n * k;
        threadPerBlock = std::min(256, len);
        FastllmCudaHalf2BF16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, (__nv_bfloat16 *)weight.extraCudaData[1], k);
        }
        FastllmCudaFree(cudaFp16Output);
#endif
        FastllmCudaFree(cudaFp16Input);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
