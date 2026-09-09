//
// BFloat16 Linear: 小规模时使用自定义 GEMV kernel，大规模时使用 cublas (仿照 fastllm-linear-fp16.cu)
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

typedef union __align__(16) _union_bf16_4 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
} union_bf16_4;

typedef union __align__(16) _union_bf16_8 {
    uint4 in;
    __nv_bfloat16 out[8];
    __nv_bfloat162 out2[4];
} union_bf16_8;

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvBf16Bf16Kernel2MultiRow(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, __nv_bfloat16 *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // Keep exact small batches on the very same PART=1 kernel as ordinary
    // decode.  blockIdx.y only selects an independent input/output row, so
    // each row retains the same instructions and reduction order as a
    // separate one-row launch on every CUDA architecture.
    if constexpr (PART == 1) {
        const size_t gridRow = (size_t)blockIdx.y;
        A += gridRow * m;
        C += gridRow * k;
    }
    union_bf16_8 regA;
    union_bf16_8 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;

    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __bfloat162float(regA.out2[0].x) * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += __bfloat162float(regA.out2[0].y) * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += __bfloat162float(regA.out2[1].x) * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += __bfloat162float(regA.out2[1].y) * __bfloat162float(regB.out2[1].y);
                if (i + 4 < m)
                    sum += __bfloat162float(regA.out2[2].x) * __bfloat162float(regB.out2[2].x);
                if (i + 5 < m)
                    sum += __bfloat162float(regA.out2[2].y) * __bfloat162float(regB.out2[2].y);
                if (i + 6 < m)
                    sum += __bfloat162float(regA.out2[3].x) * __bfloat162float(regB.out2[3].x);
                if (i + 7 < m)
                    sum += __bfloat162float(regA.out2[3].y) * __bfloat162float(regB.out2[3].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __bfloat162float(A[i + x * m]) * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
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
    __syncthreads();
}

// FP16 input × BF16 weight -> FP16 output (用于 FastllmCudaHalfMatMulBFloat16)
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Bf16Kernel2MultiRow(half *A, __nv_bfloat16 *B, half *C, half *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half4 regA;
    union_bf16_4 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint2 *>(A + i + x * m);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __bfloat162float(regB.out2[1].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __half2float(A[i + x * m]) * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
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
                C[p + k * x] = __float2half_rn(sdata[x][0] + __half2float(__ldg(bias + p)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2half_rn(sdata[x][0]);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__device__ __forceinline__ void FastllmGemvFp32Bf16Rows(float *A, const __nv_bfloat16 *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    float4 regA;
    union_bf16_4 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = *reinterpret_cast<const float4 *>(A + i + x * m);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += regA.x * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += regA.y * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += regA.z * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += regA.w * __bfloat162float(regB.out2[1].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += A[i + x * m] * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
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
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] + __ldg(bias + p);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Bf16Kernel2MultiRow(float *A, __nv_bfloat16 *B, float *C, float *bias, int m, int k) {
    // Exact speculative batches retain the ordinary one-row reduction tree.
    if constexpr (PART == 1) {
        A += (size_t)blockIdx.y * m;
        C += (size_t)blockIdx.y * k;
    }
    FastllmGemvFp32Bf16Rows<THREAD_PER_BLOCK, PART>(A, B, C, bias, m, k);
}

// By-value pointer chunks need no device registry or host route readback.
// They fit the 4 KiB kernel argument limit on older CUDA architectures and
// are owned by the graph node when captured. Only resident weights qualify.
struct FastllmBf16ExpertPointers {
    const __nv_bfloat16 *weights[256];
};

template <bool DOWN>
__global__ void FastllmMoeFp32Bf16IndexedKernel(
        float *input, float *output, const int32_t *indices,
        FastllmBf16ExpertPointers table, int first, int count,
        int topk, int m, int k) {
    const int task = blockIdx.y;
    const int expert = indices[task] - first;
    if (expert < 0 || expert >= count) return;
    const int row = DOWN ? task : task / topk;
    FastllmGemvFp32Bf16Rows<256, 1>(input + (size_t)row * m,
        table.weights[expert], output + (size_t)task * k, nullptr, m, k);
}

__device__ __forceinline__ float FastllmMoeFp32Bf16Dot4(
        const float *input, const __nv_bfloat16 *weight) {
    const float4 a = *reinterpret_cast<const float4 *>(input);
    union_bf16_4 b;
    b.in = *reinterpret_cast<const uint2 *>(weight);
    float sum = 0.0f;
    sum += a.x * __bfloat162float(b.out2[0].x);
    sum += a.y * __bfloat162float(b.out2[0].y);
    sum += a.z * __bfloat162float(b.out2[1].x);
    sum += a.w * __bfloat162float(b.out2[1].y);
    return sum;
}

template <int INTER>
__global__ void FastllmMoeFp32Bf16DownWarpKernel(
        const float *input, float *output, const int32_t *indices,
        FastllmBf16ExpertPointers table, int first, int count, int hidden) {
    const int task = blockIdx.y, expert = indices[task] - first;
    const int col = blockIdx.x * 8 + threadIdx.x / 32;
    if (expert < 0 || expert >= count || col >= hidden) return;
    const int lane = threadIdx.x % 32;
    input += (size_t)task * INTER;
    const __nv_bfloat16 *weight = table.weights[expert] + (size_t)col * INTER;
    float sum = FastllmMoeFp32Bf16Dot4(input + lane * 4, weight + lane * 4);
    float diff = 0.0f;
    // Only 32/64 lanes of the generic 256-thread reduction contain data.
    // Retain its compensated tree, including the upper 32 partial sums for
    // INTER=256, while assigning eight independent output columns per CTA.
    if constexpr (INTER == 256) {
        const float other = FastllmMoeFp32Bf16Dot4(
            input + (lane + 32) * 4, weight + (lane + 32) * 4);
        const float next = sum + other;
        diff = (next - sum) - other;
        sum = next;
    }
    for (int step = 16; step > 0; step >>= 1) {
        const float peer = __shfl_down_sync(0xffffffffu, sum, step);
        if (lane < step) {
            const float other = peer - diff;
            const float next = sum + other;
            diff = (next - sum) - other;
            sum = next;
        }
    }
    if (lane == 0) output[(size_t)task * hidden + col] = sum;
}

__global__ void FastllmMoeFp32Bf16ReduceKernel(
        const float *parts, const int32_t *indices, const float *scores,
        float *output, int hidden, int topk, int experts, bool singleRow) {
    __shared__ int order[16];
    const int row = blockIdx.y;
    indices += row * topk;
    scores += row * topk;
    if (threadIdx.x == 0) {
        for (int i = 0; i < topk; ++i) {
            int j = i;
            // The multi-row fallback visits experts in ascending id order;
            // single-row decode instead follows the router's top-k order.
            while (!singleRow && j > 0 && indices[order[j - 1]] > indices[i]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = i;
        }
    }
    __syncthreads();
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    float sum = 0.0f;
    bool first = true;
    for (int j = 0; j < topk; ++j) {
        const int route = order[j];
        if (indices[route] < 0 || indices[route] >= experts) continue;
        const float value = parts[((size_t)row * topk + route) * hidden + col];
        sum = singleRow && first ? value * scores[route]
            : fmaf(value, scores[route], sum);
        first = false;
    }
    output[(size_t)row * hidden + col] = sum;
}

bool FastllmCudaFloat32MergeMOEBFloat16Indexed(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data &gate,
        fastllm::Data &middle, fastllm::Data &parts, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch) {
    using namespace fastllm;
    if (input.dataType != FLOAT32 || input.dims.size() != 2 ||
        input.dims[0] < 1 || input.dims[0] > 4 || input.dims[1] <= 0 ||
        index.dataType != INT32 || score.dataType != FLOAT32 ||
        index.dims.size() != 2 || index.dims[0] != input.dims[0] ||
        index.dims[1] < 1 || index.dims[1] > 16 || score.dims != index.dims ||
        weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        weights[0] != nullptr || weights[1] != nullptr || weights[2] == nullptr ||
        weights[2]->dims.size() != 2 || weights[2]->dims[0] <= 0 ||
        (weights[2]->dims[0] & 1)) return false;
    const int device = FastllmCudaGetDevice();
    auto resident = [device](const Data &data) {
        return data.dataDevice == DataDevice::CUDA && data.cudaData != nullptr &&
            !data.multiDeviceData && !data.lockInCPU && !data.isDiskWeight &&
            (data.dataDeviceIds.empty() ||
             (data.dataDeviceIds.size() == 1 && data.dataDeviceIds[0] == device));
    };
    if (!resident(input) || !resident(index) || !resident(score)) return false;
    const int batch = input.dims[0], hidden = input.dims[1];
    const int topk = index.dims[1], inter = weights[2]->dims[0] / 2;
    const int experts = weightsBatch / 2 - 1;
    std::vector<FastllmBf16ExpertPointers> gateTables((experts + 255) / 256);
    std::vector<FastllmBf16ExpertPointers> downTables(gateTables.size());
    for (int e = 0; e < experts; ++e) {
        const Data *g = weights[2 * (e + 1)], *d = weights[2 * (e + 1) + 1];
        if (g == nullptr || d == nullptr || g->dataType != BFLOAT16 ||
            d->dataType != BFLOAT16 || g->dims.size() != 2 || d->dims.size() != 2 ||
            g->dims[0] != 2 * inter || g->dims[1] != hidden ||
            d->dims[0] != hidden || d->dims[1] != inter || !resident(*g) || !resident(*d)) return false;
        gateTables[e / 256].weights[e % 256] = (const __nv_bfloat16 *)g->cudaData;
        downTables[e / 256].weights[e % 256] = (const __nv_bfloat16 *)d->cudaData;
    }
    auto prepare = [&](Data &data, int rows, int cols) {
        data.dataType = FLOAT32;
        data.UpdateUnitSize();
        data.Resize({rows, cols});
        data.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
        data.Allocate(false);
    };
    prepare(gate, batch * topk, 2 * inter);
    prepare(middle, batch * topk, inter);
    prepare(parts, batch * topk, hidden);
    prepare(output, batch, hidden);
    for (int first = 0; first < experts; first += 256) {
        FastllmMoeFp32Bf16IndexedKernel<false><<<dim3(2 * inter, batch * topk), 256>>>(
            (float *)input.cudaData, (float *)gate.cudaData, (const int32_t *)index.cudaData,
            gateTables[first / 256], first, std::min(256, experts - first), topk, hidden, 2 * inter);
    }
    FastllmCudaSwiglu(gate, middle);
    for (int first = 0; first < experts; first += 256) {
        if (inter == 128) {
            FastllmMoeFp32Bf16DownWarpKernel<128><<<dim3((hidden + 7) / 8, batch * topk), 256>>>(
                (const float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), hidden);
        } else if (inter == 256) {
            FastllmMoeFp32Bf16DownWarpKernel<256><<<dim3((hidden + 7) / 8, batch * topk), 256>>>(
                (const float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), hidden);
        } else {
            FastllmMoeFp32Bf16IndexedKernel<true><<<dim3(hidden, batch * topk), 256>>>(
                (float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), topk, inter, hidden);
        }
    }
    FastllmMoeFp32Bf16ReduceKernel<<<dim3((hidden + 255) / 256, batch), 256>>>(
        (const float *)parts.cudaData, (const int32_t *)index.cudaData,
        (const float *)score.cudaData, (float *)output.cudaData,
        hidden, topk, experts, batch == 1);
    return true;
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp32KernelForBf16(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

static void FastllmCudaBF16EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }
}

static void FastllmCudaBF16EnsureBiasBf16OnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
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
        checkCudaErrors("Error: CUDA error when moving bias (bf16) to device!", state);
        if (weight.extraCudaData.size() < 2)
            weight.extraCudaData.push_back((void *)cudaBiasData);
        else
            weight.extraCudaData[1] = (void *)cudaBiasData;
    }
}

// Half (FP16) bias for FP16×BF16 matmul output
static void FastllmCudaBF16EnsureBiasHalfOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || (bias.dims.size() > 0 && weight.extraCudaHalfData.size() == 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias (half for FP16×BF16) to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }
}

void LaunchFastllmGemmFp32Bf16(float *input, __nv_bfloat16 *weight, float *output, float *bias, int n, int m, int k) {
    if (n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        if (n <= 65535) {
            FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>
                <<<dim3(k, n), 256>>>(
                    input, weight, output, bias, m, k);
        } else {
            for (int i = 0; i < n; ++i) {
                FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>
                    <<<k, 256>>>(input + (size_t)i * m, weight,
                                 output + (size_t)i * k, bias, m, k);
            }
        }
    } else if (n == 1) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 2> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 3> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 4> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 5> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 6> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 7> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp32Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

// BF16 -> FP16 逐元素转换，供 cublas FP16 gemm 使用
__global__ void FastllmCudaBF162HalfKernel(const __nv_bfloat16 *src, half *dst, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        dst[idx] = __float2half_rn(__bfloat162float(src[idx]));
}

// FP16 -> BF16 逐元素转换，供 FP16 input 转 BF16 后走 cublas BF16 gemm
__global__ void FastllmCudaHalf2Bf16Kernel(const half *src, __nv_bfloat16 *dst, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        dst[idx] = __float2bfloat16_rn(__half2float(src[idx]));
}

// cublas 输出 C 为列主序 (k×n)：C[i,j] 在 src[i+j*k]，且 C[i,j]=output[j,i]；写入行主序 dst：dst[row*k+col]=output[row,col]=src[col+row*k]
__global__ void Bf16ToHalfTransposeKernel(const __nv_bfloat16 *src, half *dst, int n, int k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n * k) return;
    int row = idx / k;
    int col = idx % k;
    dst[idx] = __float2half_rn(__bfloat162float(src[col + row * k]));
}

void LaunchFastllmGemmFp16Bf16(half *input, __nv_bfloat16 *weight, half *output, half *bias, int n, int m, int k) {
    if (n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        for (int i = 0; i < n; ++i) {
            FastllmGemvFp16Bf16Kernel2MultiRow<256, 1>
                <<<k, 256>>>(input + (size_t)i * m, weight,
                             output + (size_t)i * k, bias, m, k);
        }
    } else if (n == 1) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 2> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 3> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 4> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 5> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 6> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 7> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

void LaunchFastllmGemmBf16Bf16(__nv_bfloat16 *input, __nv_bfloat16 *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, int n, int m, int k) {
    if (n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        // CUDA guarantees at least 65,535 blocks in grid.y. Preserve the
        // previous launch-per-row fallback if an external caller requests a
        // larger exact batch instead of relying on a device-specific limit.
        if (n <= 65535) {
            FastllmGemvBf16Bf16Kernel2MultiRow<256, 1>
                <<<dim3(k, n), 256>>>(
                    input, weight, output, bias, m, k);
        } else {
            for (int i = 0; i < n; ++i) {
                FastllmGemvBf16Bf16Kernel2MultiRow<256, 1>
                    <<<k, 256>>>(input + (size_t)i * m, weight,
                                 output + (size_t)i * k, bias, m, k);
            }
        }
    } else if (n == 1) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 2> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 3> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 4> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 5> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 6> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 7> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvBf16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

bool FastllmCudaMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
    float *cudaBiasData = (float *)weight.extraCudaData[0];
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    if (n < 8) {
        LaunchFastllmGemmFp32Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        __nv_bfloat16 *cudaBf16Input = (__nv_bfloat16 *)FastllmCudaMalloc(n * m * sizeof(__nv_bfloat16));

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2Bf16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaBf16Input, len);

        cublasStatus_t status;
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weightPtr, AType,
                              m, cudaBf16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (MatMulBFloat16).\n");
            throw("cublas error");
            exit(0);
        }

        FastllmCudaFree(cudaBf16Input);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// FP16 input × BF16 weight -> FP16 output
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
    FastllmCudaBF16EnsureBiasHalfOnDevice(weight, bias, k);

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *)weight.extraCudaHalfData[0];
    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    bool exactRows = n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold();
    if (n < 8 || exactRows) {
        LaunchFastllmGemmFp16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else if (false) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc((size_t)n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        int threadPerBlock = std::min(256, m);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            int chunkLen = kc * m;
            FastllmCudaBF162HalfKernel <<<(chunkLen - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (__nv_bfloat16 *)weight.cudaData + (size_t)kOff * m,
                cudaFp16Weight, chunkLen);
#ifdef CUDA_NO_TENSOR_CORE
            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaFp32Output + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        __nv_bfloat16 *cudaBF16Input;
        __nv_bfloat16 *cudaBF16Output;
        cudaBF16Input = (__nv_bfloat16 *) FastllmCudaMalloc(n * m * sizeof(__nv_bfloat16));
        cudaBF16Output = (__nv_bfloat16 *) FastllmCudaMalloc(n * k * sizeof(__nv_bfloat16));
        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaHalf2Bf16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaBF16Input, len);

        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, weightPtr, AType,
                                m, cudaBF16Input, BType,
                                m, &h_beta,
                                cudaBF16Output, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        threadPerBlock = std::min(256, len);
        FastllmCudaBF162HalfKernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaBF16Output, cudaOutput, len);

        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaBF16Input);
        FastllmCudaFree(cudaBF16Output);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// BF16 input × BF16 weight -> BF16 output
bool FastllmCudaBFloat16MatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
    FastllmCudaBF16EnsureBiasBf16OnDevice(weight, bias, k);

    __nv_bfloat16 *cudaInput = (__nv_bfloat16 *)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16 *)FastllmCudaPrepareOutput(output);
    __nv_bfloat16 *cudaBiasData = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *)weight.extraCudaData[1];
    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    bool exactRows = n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold();
    if (n < 8 || exactRows) {
        LaunchFastllmGemmBf16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weightPtr, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (BFloat16MatMulBFloat16).\n");
            throw("cublas error");
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, (__nv_bfloat16 *)weight.extraCudaData[1], k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// BF16 input × FP32 weight -> BF16 output
bool FastllmCudaBFloat16MatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float *)weight.extraCudaData[0];
    float *cudaInput = (float *)FastllmCudaMalloc(input.Count(0) * sizeof(float));
    float *cudaOutput = (float *)FastllmCudaMalloc(output.Count(0) * sizeof(float));
    int inputLen = input.Count(0);
    FastllmCudaBF162FloatKernel <<< (inputLen - 1) / 256 + 1, 256 >>>((uint16_t *)input.cudaData, cudaInput, inputLen);

    if (n > 1) {
        float h_alpha = 1.0f, h_beta = 0.0f;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weight.cudaData, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFree(cudaInput);
            FastllmCudaFree(cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32KernelForBf16<256, 1> <<< k, 256 >>>(cudaInput, (float *)weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    int outputLen = output.Count(0);
    FastllmCudaFloat2Bf16Kernel <<< (outputLen - 1) / 256 + 1, 256 >>>(cudaOutput, (__nv_bfloat16 *)output.cudaData, outputLen);
    FastllmCudaFree(cudaInput);
    FastllmCudaFree(cudaOutput);
    DeviceSync();
    return true;
}
