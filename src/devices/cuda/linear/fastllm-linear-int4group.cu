//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "devices/cuda/fastllm-awq-sm70.cuh"

#include <cmath>
#include <unordered_map>

#if !defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

__device__ __forceinline__ float FastllmCudaDequantInt4GroupValue(
        float q, float scale, float minOrZero, bool useZeroPoint) {
    return useZeroPoint ? scale * (q - minOrZero) : scale * q + minOrZero;
}

__device__ __forceinline__ float FastllmCudaDequantInt4GroupHalfValue(
        float q, float scale, float minOrZero, bool useZeroPoint) {
    // W4A16 kernels must consume the same FP16-rounded weights as the
    // dequantize+cuBLAS and Marlin paths. Keeping scale * (q - zero) in FP32
    // introduces a small systematic delta that compounds across deep MoE
    // models and can eventually change routing and logits.
    return __half2float(__float2half_rn(
        FastllmCudaDequantInt4GroupValue(q, scale, minOrZero, useZeroPoint)));
}

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per,
                                                int group, int groupCnt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = idx / per * group + (idx % per / groupCnt);
    if (idx < len) {
        if (idx % 2 == 1) {
            b[idx] = __float2half(scales[gid] * (a[idx / 2] & 0xF) + mins[gid]);
        } else {
            b[idx] = __float2half(scales[gid] * (a[idx / 2] >> 4) + mins[gid]);
        }
    }
}

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, half *scales, half *mins, half *b,
                                                int k, int m, int group, int groupCnt,
                                                bool useZeroPoint) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;
    half2 scalesBuffer;
    half2 minBuffer;
    int threshold = ST128_FP16_COUNT;
    for (int i = tid * ST128_FP16_COUNT; i < m; i += blockDim.x * ST128_FP16_COUNT) {
        int index = st * m + i;
        int startIdx = st * group + i / groupCnt;
        int endIdx = st * group + (i + ST128_FP16_COUNT - 1) / groupCnt;
        scalesBuffer.x = scalesBuffer.y = __ldg(scales + startIdx);
        minBuffer.x = minBuffer.y = __ldg(mins + startIdx);
        if (endIdx > startIdx) {
            threshold = (i + ST128_FP16_COUNT - 1) % groupCnt;
            scalesBuffer.y = __ldg(scales + endIdx);
            minBuffer.y = __ldg(mins + endIdx);
        }
        // 读取
        union_char4 aBuffer;
        union_half8 bBuffer;
        aBuffer.in = *reinterpret_cast<const uint32_t *>(a + index / 2);
        // 处理
        for (int j = 0; j < ST128_FP16_COUNT / 2; j++) {
            if (i + j * 2 + 1 < m) {
                float scale = __half2float(j * 2 < threshold ? scalesBuffer.x : scalesBuffer.y);
                float min = __half2float(j * 2 < threshold ? minBuffer.x : minBuffer.y);
                bBuffer.out[j * 2] = __float2half(FastllmCudaDequantInt4GroupValue(
                    (float)(aBuffer.out[j] >> 4), scale, min, useZeroPoint));
                bBuffer.out[j * 2 + 1] = __float2half(FastllmCudaDequantInt4GroupValue(
                    (float)(aBuffer.out[j] & 0xF), scale, min, useZeroPoint));
            }
        }
        reinterpret_cast<uint4 *>(b)[index / ST128_FP16_COUNT] = bBuffer.in;
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4GroupKernel3(float *A, uint8_t *B, float *C,
                                             float *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt,
                                             bool useZeroPoint) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    #pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 2]);

        for (int p = st; p < end; p++) {
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(B + p * m / 2 + i);
            int g = p * group + (i * 2 / groupCnt);
            float curmin = __half2float(__ldg(mins + g)), curscale = __half2float(__ldg(scales + g));
            sdata[p - st][tid] += aBuffer.x * FastllmCudaDequantInt4GroupValue(
                                            (float)((bBuffer >> 4) & 15), curscale, curmin, useZeroPoint)
                         + aBuffer.y * FastllmCudaDequantInt4GroupValue(
                                            (float)(bBuffer & 15), curscale, curmin, useZeroPoint);
            sdata[p - st][tid] += aBuffer.z * FastllmCudaDequantInt4GroupValue(
                                            (float)(bBuffer >> 12), curscale, curmin, useZeroPoint)
                         + aBuffer.w * FastllmCudaDequantInt4GroupValue(
                                            (float)((bBuffer >> 8) & 15), curscale, curmin, useZeroPoint);
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4GroupKernel2(float *A, uint8_t *B, float *C,
                                             float *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt,
                                             bool useZeroPoint) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    #pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 8]);
        float4 bBuffer = FETCH_FLOAT4(A[i * 8 + 4]);

        for (int p = st; p < end; p++) {
            uint8_t now0 = B[p * m / 2 + i * 4];
            uint8_t now1 = B[p * m / 2 + i * 4 + 1];
            uint8_t now2 = B[p * m / 2 + i * 4 + 2];
            uint8_t now3 = B[p * m / 2 + i * 4 + 3];
            int g = p * group + (i * 8 / groupCnt);
            float curmin = (float)mins[g], curscale = (float)scales[g];
            sdata[p - st][tid] += aBuffer.x * FastllmCudaDequantInt4GroupValue(
                                            (float)(now0 >> 4), curscale, curmin, useZeroPoint)
                         + aBuffer.y * FastllmCudaDequantInt4GroupValue(
                                            (float)(now0 & 15), curscale, curmin, useZeroPoint);
            sdata[p - st][tid] += aBuffer.z * FastllmCudaDequantInt4GroupValue(
                                            (float)(now1 >> 4), curscale, curmin, useZeroPoint)
                         + aBuffer.w * FastllmCudaDequantInt4GroupValue(
                                            (float)(now1 & 15), curscale, curmin, useZeroPoint);
            sdata[p - st][tid] += bBuffer.x * FastllmCudaDequantInt4GroupValue(
                                            (float)(now2 >> 4), curscale, curmin, useZeroPoint)
                         + bBuffer.y * FastllmCudaDequantInt4GroupValue(
                                            (float)(now2 & 15), curscale, curmin, useZeroPoint);
            sdata[p - st][tid] += bBuffer.z * FastllmCudaDequantInt4GroupValue(
                                            (float)(now3 >> 4), curscale, curmin, useZeroPoint)
                         + bBuffer.w * FastllmCudaDequantInt4GroupValue(
                                            (float)(now3 & 15), curscale, curmin, useZeroPoint);
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfInt4GroupKernelMultiRow(half *A, uint8_t *B, half *C,
                                             half *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt,
                                             bool useZeroPoint) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int end = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    union_char4 bBuffer;
    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *reinterpret_cast<const uint32_t *>(B + st * m / 2 + i * 4);
        // uint8_t now0 = B[st * m / 2 + i * 4];
        // uint8_t now1 = B[st * m / 2 + i * 4 + 1];
        // uint8_t now2 = B[st * m / 2 + i * 4 + 2];
        // uint8_t now3 = B[st * m / 2 + i * 4 + 3];
        int g = st * group + (i * 8 / groupCnt);
        float curmin = (float)mins[g], curscale = (float)scales[g];
        for (int x = 0; x < PART; x++) {
            union_half8 aBuffer;
            aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + i * 8);
            sdata[x][tid] += (float)aBuffer.out[0] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[0] >> 4), curscale, curmin, useZeroPoint)
                         + (float)aBuffer.out[1] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[0] & 15), curscale, curmin, useZeroPoint);
            sdata[x][tid] += (float)aBuffer.out[2] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[1] >> 4), curscale, curmin, useZeroPoint)
                         + (float)aBuffer.out[3] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[1] & 15), curscale, curmin, useZeroPoint);
            sdata[x][tid] += (float)aBuffer.out[4] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[2] >> 4), curscale, curmin, useZeroPoint)
                         + (float)aBuffer.out[5] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[2] & 15), curscale, curmin, useZeroPoint);
            sdata[x][tid] += (float)aBuffer.out[6] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[3] >> 4), curscale, curmin, useZeroPoint)
                         + (float)aBuffer.out[7] * FastllmCudaDequantInt4GroupHalfValue(
                                            (float)(bBuffer.out[3] & 15), curscale, curmin, useZeroPoint);
        }
    }

    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[x][tid] += sdata[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + st)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
}

// 优化版本: 每个 warp 负责一行输出, 每个 lane 一次加载 8 个 INT4 权重字节 (uint2 = 16 个 nibble),
// 使用 warp shuffle 归约, 去除 shared memory 与 __syncthreads。float 输入版本。
// 参照 FP8 的 warp kernel。要求 m % 16 == 0 且 groupCnt % 16 == 0。
template <int WARPS_PER_BLOCK, int PART>
__global__ void FastllmGemvFloatInt4GroupKernelWarpMultiRow(
        const float * __restrict__ A, const uint8_t * __restrict__ B,
        float * __restrict__ C, const float * __restrict__ bias,
        const half * __restrict__ scales, const half * __restrict__ mins,
        int m, int k, int group, int groupCnt, bool useZeroPoint) {
    const int warpId = threadIdx.x >> 5;
    const int laneId = threadIdx.x & 31;
    const int st = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (st >= k) return;

    const uint8_t *baseB = B + (size_t)st * (m / 2);
    const half *rowScales = scales + (size_t)st * group;
    const half *rowMins = mins + (size_t)st * group;

    float acc[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) acc[x] = 0.0f;

    const int numUnits = m >> 4;  // 每单元 16 个元素 (8 字节权重)
    for (int u = laneId; u < numUnits; u += 32) {
        const int i = u << 4;
        const int g = i / groupCnt;
        const float curScale = __half2float(__ldg(rowScales + g));
        const float curMin = __half2float(__ldg(rowMins + g));

        union_char8 bw;
        bw.in = *reinterpret_cast<const uint2 *>(baseB + (size_t)u * 8);
        float wval[16];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const uint8_t byteVal = bw.out[b];
            wval[b * 2] = FastllmCudaDequantInt4GroupValue(
                (float)(byteVal >> 4), curScale, curMin, useZeroPoint);
            wval[b * 2 + 1] = FastllmCudaDequantInt4GroupValue(
                (float)(byteVal & 15), curScale, curMin, useZeroPoint);
        }

#pragma unroll
        for (int x = 0; x < PART; x++) {
            const float *Ax = A + (size_t)x * m + i;
            float4 a0 = *reinterpret_cast<const float4 *>(Ax);
            float4 a1 = *reinterpret_cast<const float4 *>(Ax + 4);
            float4 a2 = *reinterpret_cast<const float4 *>(Ax + 8);
            float4 a3 = *reinterpret_cast<const float4 *>(Ax + 12);
            acc[x] += a0.x * wval[0]  + a0.y * wval[1]  + a0.z * wval[2]  + a0.w * wval[3]
                    + a1.x * wval[4]  + a1.y * wval[5]  + a1.z * wval[6]  + a1.w * wval[7]
                    + a2.x * wval[8]  + a2.y * wval[9]  + a2.z * wval[10] + a2.w * wval[11]
                    + a3.x * wval[12] + a3.y * wval[13] + a3.z * wval[14] + a3.w * wval[15];
        }
    }

#pragma unroll
    for (int x = 0; x < PART; x++) {
        float v = acc[x];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        acc[x] = v;
    }

    if (laneId == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            C[st + (size_t)k * x] = acc[x] + bias[st];
        }
    }
}

void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias,
                                    half *scales, half *mins, int n, int m, int k,
                                    int group, int groupCnt, bool useZeroPoint) {
    // 满足 16 对齐时走 warp 优化版 GEMV (参照 FP8 warp kernel)。
    if ((m & 15) == 0 && groupCnt > 0 && (groupCnt & 15) == 0) {
        constexpr int W = 8;  // 每个 block 8 个 warp (256 线程)
        const int grid = (k + W - 1) / W;
#define FASTLLM_INT4G_WARP_LAUNCH_F32(PARTVAL, OFF) \
        FastllmGemvFloatInt4GroupKernelWarpMultiRow<W, PARTVAL> <<< grid, W * 32 >>>( \
            input + (OFF) * m, weight, output + (OFF) * k, bias, scales, mins, m, k, group, groupCnt, useZeroPoint)
        switch (n) {
            case 1:  FASTLLM_INT4G_WARP_LAUNCH_F32(1, 0);  return;
            case 2:  FASTLLM_INT4G_WARP_LAUNCH_F32(2, 0);  return;
            case 3:  FASTLLM_INT4G_WARP_LAUNCH_F32(3, 0);  return;
            case 4:  FASTLLM_INT4G_WARP_LAUNCH_F32(4, 0);  return;
            case 5:  FASTLLM_INT4G_WARP_LAUNCH_F32(5, 0);  return;
            case 6:  FASTLLM_INT4G_WARP_LAUNCH_F32(6, 0);  return;
            case 7:  FASTLLM_INT4G_WARP_LAUNCH_F32(7, 0);  return;
            case 8:  FASTLLM_INT4G_WARP_LAUNCH_F32(8, 0);  return;
            default: break;
        }
        int i = 0;
        for (; i + 7 < n; i += 8) FASTLLM_INT4G_WARP_LAUNCH_F32(8, i);
        for (; i + 3 < n; i += 4) FASTLLM_INT4G_WARP_LAUNCH_F32(4, i);
        for (; i + 1 < n; i += 2) FASTLLM_INT4G_WARP_LAUNCH_F32(2, i);
        for (; i < n; i++)        FASTLLM_INT4G_WARP_LAUNCH_F32(1, i);
#undef FASTLLM_INT4G_WARP_LAUNCH_F32
        return;
    }

    for (int i = 0; i < n; i++) {
#ifdef CUDA_NO_TENSOR_CORE
        FastllmGemvInt4GroupKernel3<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
#else
        FastllmGemvInt4GroupKernel2<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
#endif
    }
}

static constexpr int INT4GROUP_CUDA_SCALES_IDX = 0;
static constexpr int INT4GROUP_CUDA_MINS_IDX = 1;
static constexpr int INT4GROUP_CUDA_BIAS_IDX = 2;
static constexpr int INT4GROUP_MARLIN_WEIGHT_IDX = 3;
static constexpr int INT4GROUP_MARLIN_ZEROS_IDX = 4;
static constexpr int INT4GROUP_MARLIN_WORKSPACE_IDX = 5;
static constexpr int INT4GROUP_MOE_POINTER_TABLE_IDX = 6;
static constexpr int INT4GROUP_MOE_QUANT_META_IDX = 7;

static constexpr int INT4GROUP_HALF_SCALES_IDX = 0;
static constexpr int INT4GROUP_HALF_MINS_IDX = 1;
static constexpr int INT4GROUP_HALF_BIAS_IDX = 2;
static constexpr int INT4GROUP_MARLIN_SCALES_HALF_IDX = 3;

// ==================== SM70 (V100) AWQ via TurboMind s884 ====================
// Marlin 需要 sm_75+，在 V100 上不可用，INT4_GROUP 会退化为 dequant + cublas，
// 速度与 FP8 无异。这里为 SM70 提供一条真正的 W4A16 GEMM 路径：把 INT4_GROUP
// 权重重排为 TurboMind 所需的解包权重 / scale / zero，交由移植的 s884 内核计算。
// Handle 为主机侧指针，不能放入 extraCudaData（会被 FastllmCudaFree），单独缓存。
static std::unordered_map<const fastllm::Data*, void*> g_sm70AwqHandles;

// 重排成功后释放原始 INT4_GROUP 权重，定义在后面，这里前置声明。
static void FastllmCudaInt4GroupReleaseOriginalWeight(fastllm::Data &weight);

// weight: [k, m] 每字节两个 nibble（输出在外、输入在内），偶数输入在高位。
// 输出 out: [K=m, N=k] 行主序，out[in * k + outIdx] = 该 (输入 in, 输出 outIdx) 的 4bit 值。
__global__ void FastllmInt4GroupToAwqU16Kernel(const uint8_t *weight, uint16_t *out, int m, int k) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)m * k;
    if (idx >= total) {
        return;
    }
    int in = (int)(idx / k);
    int outIdx = (int)(idx - (size_t)in * k);
    uint8_t byte = weight[(size_t)outIdx * (m / 2) + in / 2];
    uint16_t q = (in & 1) ? (byte & 0xF) : (byte >> 4);
    out[idx] = q;
}

static bool FastllmCudaInt4GroupSm70AwqEnabled(int n, int m, int k, int groupCnt) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    static const bool disabled = (getenv("FASTLLM_DISABLE_SM70_AWQ") != nullptr);
    if (disabled) {
        return false;
    }
    if (!fastllm::awq_sm70::Supported()) {
        return false;
    }
    // K = m (输入维), N = k (输出维)，TurboMind 要求 K/N 为 8 的倍数，按 K 分组。
    return groupCnt > 0 && (groupCnt == 32 || groupCnt == 64 || groupCnt == 128) &&
           m % groupCnt == 0 && m % 8 == 0 && k % 8 == 0;
#endif
}

static bool FastllmCudaInt4GroupEnsureSm70AwqOnDevice(fastllm::Data &weight, int m, int k) {
    auto it = g_sm70AwqHandles.find(&weight);
    if (it != g_sm70AwqHandles.end()) {
        return it->second != nullptr;
    }
    int group = weight.group, groupCnt = weight.groupCnt;
    if (weight.cudaData == nullptr || group <= 0 || groupCnt <= 0 ||
        weight.scales.size() != (size_t)k * group ||
        weight.mins.size() != (size_t)k * group) {
        g_sm70AwqHandles[&weight] = nullptr;
        return false;
    }

    const int K = m, N = k, numGroups = group;

    uint16_t *dU16 = (uint16_t*)FastllmCudaMalloc((size_t)K * N * sizeof(uint16_t));
    if (dU16 == nullptr) {
        printf("FastllmAwqSm70 prepare error: FastllmCudaMalloc(dU16, %zu bytes) failed (likely OOM). m=%d k=%d\n",
               (size_t)K * N * sizeof(uint16_t), m, k);
        g_sm70AwqHandles[&weight] = nullptr;
        return false;
    }
    size_t total = (size_t)K * N;
    int threads = 256;
    FastllmInt4GroupToAwqU16Kernel <<< (total + threads - 1) / threads, threads >>>(
        (const uint8_t*)weight.cudaData, dU16, m, k);

    std::vector<half> hScales((size_t)numGroups * N);
    std::vector<half> hZeros((size_t)numGroups * N);
    for (int g = 0; g < numGroups; g++) {
        for (int nn = 0; nn < N; nn++) {
            float s = weight.scales[(size_t)nn * group + g];
            float mn = weight.mins[(size_t)nn * group + g];
            size_t meta = (size_t)nn * group + g;
            int z = weight.zeros.size() == (size_t)N * group
                ? weight.zeros[meta]
                : ((s == 0.0f) ? 0 : (int)std::lroundf(-mn / s));
            z = std::max(0, std::min(15, z));
            hScales[(size_t)g * N + nn] = __float2half(s);
            hZeros[(size_t)g * N + nn] = __float2half((float)z);
        }
    }
    half *dScales = (half*)FastllmCudaMalloc(hScales.size() * sizeof(half));
    half *dZeros = (half*)FastllmCudaMalloc(hZeros.size() * sizeof(half));
    if (dScales == nullptr || dZeros == nullptr) {
        printf("FastllmAwqSm70 prepare error: FastllmCudaMalloc(scales=%p zeros=%p) failed (likely OOM). "
               "numGroups=%d N=%d\n", (void*)dScales, (void*)dZeros, numGroups, N);
        FastllmCudaFree(dU16);
        if (dScales) FastllmCudaFree(dScales);
        if (dZeros) FastllmCudaFree(dZeros);
        g_sm70AwqHandles[&weight] = nullptr;
        return false;
    }
    FastllmCudaCopyFromHostToDevice(dScales, hScales.data(), hScales.size() * sizeof(half));
    FastllmCudaCopyFromHostToDevice(dZeros, hZeros.data(), hZeros.size() * sizeof(half));

    // dU16 已经从原始权重重排出量化值，scale/zero 也已拷到 device，原始 INT4_GROUP
    // 权重（weight.cudaData）此后不再需要：GEMM 走 handle->tmWeight/tmScales。
    // 这里在 Prepare 之前就释放，并用 FastllmCudaClearBigBuffer 把池中空闲显存真正
    // 归还给 OS——Prepare 内部用的是原生 cudaMalloc，只能向 OS 申请显存。否则
    // 原始权重虽被标记空闲仍滞留在 fastllm 显存池里，Prepare 仍会 OOM。
    FastllmCudaInt4GroupReleaseOriginalWeight(weight);
    FastllmCudaClearBigBuffer();

    void *handle = fastllm::awq_sm70::Prepare(dU16, dScales, dZeros, K, N, numGroups, groupCnt, 0);

    FastllmCudaFree(dU16);
    FastllmCudaFree(dScales);
    FastllmCudaFree(dZeros);

    g_sm70AwqHandles[&weight] = handle;
    return handle != nullptr;
}

static bool FastllmCudaInt4GroupHasMarlinOnDevice(const fastllm::Data &weight) {
    return (int)weight.extraCudaData.size() > INT4GROUP_MARLIN_WORKSPACE_IDX &&
           (int)weight.extraCudaHalfData.size() > INT4GROUP_MARLIN_SCALES_HALF_IDX &&
           weight.extraCudaData[INT4GROUP_MARLIN_WEIGHT_IDX] != nullptr &&
           weight.extraCudaData[INT4GROUP_MARLIN_ZEROS_IDX] != nullptr &&
           weight.extraCudaData[INT4GROUP_MARLIN_WORKSPACE_IDX] != nullptr &&
           weight.extraCudaHalfData[INT4GROUP_MARLIN_SCALES_HALF_IDX] != nullptr;
}

static void FastllmCudaInt4GroupFallbackUnavailable() {
    printf("Error: INT4_GROUP original CUDA weight was released after Marlin repack; fallback path is unavailable.\n");
    throw("int4group marlin-only fallback error");
    exit(0);
}

static void FastllmCudaInt4GroupReleaseExtraCudaData(fastllm::Data &weight, int index) {
    if ((int)weight.extraCudaData.size() <= index || weight.extraCudaData[index] == nullptr) {
        return;
    }

    void *ptr = weight.extraCudaData[index];
    for (int i = 0; i < (int)weight.extraCudaHalfData.size(); i++) {
        if (weight.extraCudaHalfData[i] == ptr) {
            weight.extraCudaHalfData[i] = nullptr;
        }
    }
    FastllmCudaFree(ptr);
    weight.extraCudaData[index] = nullptr;
}

static void FastllmCudaInt4GroupReleaseFallbackCaches(fastllm::Data &weight) {
    FastllmCudaInt4GroupReleaseExtraCudaData(weight, INT4GROUP_CUDA_SCALES_IDX);
    FastllmCudaInt4GroupReleaseExtraCudaData(weight, INT4GROUP_CUDA_MINS_IDX);
    FastllmCudaInt4GroupReleaseExtraCudaData(weight, INT4GROUP_CUDA_BIAS_IDX);
    FastllmCudaInt4GroupReleaseExtraCudaData(weight, INT4GROUP_MOE_POINTER_TABLE_IDX);
    FastllmCudaInt4GroupReleaseExtraCudaData(weight, INT4GROUP_MOE_QUANT_META_IDX);
}

static void FastllmCudaInt4GroupReleaseOriginalWeight(fastllm::Data &weight) {
    if (weight.cudaData != nullptr) {
        FastllmCudaFree(weight.cudaData);
        weight.cudaData = nullptr;
    }
    FastllmCudaInt4GroupReleaseFallbackCaches(weight);
}

static void FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr) {
        FastllmCudaInt4GroupFallbackUnavailable();
    }

    if ((int)weight.extraCudaData.size() <= INT4GROUP_CUDA_BIAS_IDX) {
        weight.extraCudaData.resize(INT4GROUP_CUDA_BIAS_IDX + 1, nullptr);
    }

    cudaError_t state = cudaSuccess;
    int group = weight.group;
    const size_t quantMetaCount = (size_t)k * group;
    const bool useZeroPoint = weight.zeros.size() == quantMetaCount;

    if (weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX] == nullptr) {
        half *cudaScales;
        state = cudaMalloc(&cudaScales, k * group * sizeof(half));
        half *scales = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            scales[i] = (half)weight.scales[i];
        }
        state = cudaMemcpy(cudaScales, scales, k * group * sizeof(half), cudaMemcpyHostToDevice);
        weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX] = (void*)cudaScales;
        delete[] scales;
    }

    if (weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX] == nullptr) {
        half *cudaMins;
        state = cudaMalloc(&cudaMins, k * group * sizeof(half));
        half *mins = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            mins[i] = useZeroPoint ? (half)weight.zeros[i] : (half)weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * group * sizeof(half), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX] = (void*)cudaMins;
    }

    if (weight.extraCudaData[INT4GROUP_CUDA_BIAS_IDX] == nullptr) {
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData[INT4GROUP_CUDA_BIAS_IDX] = (void*)cudaBiasData;
    }
}

bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, 
                                    int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    const bool useZeroPoint = weight.zeros.size() == (size_t)k * group;
    FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(weight, bias, k);

    half *cudaScales = (half*)weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX];
    half *cudaMins = (half*)weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX];
    float *cudaBiasData = (float*)weight.extraCudaData[INT4GROUP_CUDA_BIAS_IDX];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc((size_t)n * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc((size_t)n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc((size_t)n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif

        // 借 workspace 作为 INT4 -> FP16 weight 的反量化临时缓冲，按 K 维分块
        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);

            FastllmCudaInt4Group2HalfKernel <<< kc, 64 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m / 2,
                cudaScales + (size_t)kOff * group,
                cudaMins + (size_t)kOff * group,
                cudaFp16Weight, kc, m, group, groupCnt, useZeroPoint);

#ifdef CUDA_NO_TENSOR_CORE
            status = cublasGemmEx(fastllmCublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  kc, n, m,
                                  &h_alpha, cudaFp16Weight, AType,
                                  m, cudaFp16Input, BType,
                                  m, &h_beta,
                                  cudaOutput + kOff, CType,
                                  k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
            status = cublasGemmEx(fastllmCublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  kc, n, m,
                                  &h_alpha, cudaFp16Weight, AType,
                                  m, cudaFp16Input, BType,
                                  m, &h_beta,
                                  cudaFp16Output + kOff, CType,
                                  k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error. status = %d\n", status);
                throw("cublas error");
                exit(0);
            }
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
#endif
        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        LaunchFastllmGemmFp32Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData,
                                       cudaScales, cudaMins, n, m, k, group, groupCnt, useZeroPoint);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// 优化版本: 每个 warp 负责一行输出, 每个 lane 一次加载 8 个 INT4 权重字节 (uint2 = 16 个 nibble),
// 使用 warp shuffle 归约, 去除 shared memory 与 __syncthreads, 显著提升访存带宽利用率。
// 参照 FP8 的 FastllmGemvHalfFP8E4M3KernelWarpMultiRow。
// 要求 m % 16 == 0 且 groupCnt % 16 == 0 (AWQ groupCnt 32/64/128 均满足),
// 这样每 16 个元素的对齐单元必定落在同一量化组内, scale/min 取一次即可。
template <int WARPS_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfInt4GroupKernelWarpMultiRow(
        const half * __restrict__ A, const uint8_t * __restrict__ B,
        half * __restrict__ C, const half * __restrict__ bias,
        const half * __restrict__ scales, const half * __restrict__ mins,
        int m, int k, int group, int groupCnt, bool useZeroPoint) {
    const int warpId = threadIdx.x >> 5;
    const int laneId = threadIdx.x & 31;
    const int st = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (st >= k) return;

    const uint8_t *baseB = B + (size_t)st * (m / 2);
    const half *rowScales = scales + (size_t)st * group;
    const half *rowMins = mins + (size_t)st * group;

    float acc[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) acc[x] = 0.0f;

    const int numUnits = m >> 4;  // 每单元 16 个元素 (8 字节权重)
    for (int u = laneId; u < numUnits; u += 32) {
        const int i = u << 4;
        const int g = i / groupCnt;
        const float curScale = __half2float(__ldg(rowScales + g));
        const float curMin = __half2float(__ldg(rowMins + g));

        union_char8 bw;
        bw.in = *reinterpret_cast<const uint2 *>(baseB + (size_t)u * 8);
        // 解出 16 个权重值: 字节 b 的高 nibble = 元素 2b, 低 nibble = 元素 2b+1
        float wval[16];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const uint8_t byteVal = bw.out[b];
            wval[b * 2] = FastllmCudaDequantInt4GroupHalfValue(
                (float)(byteVal >> 4), curScale, curMin, useZeroPoint);
            wval[b * 2 + 1] = FastllmCudaDequantInt4GroupHalfValue(
                (float)(byteVal & 15), curScale, curMin, useZeroPoint);
        }

#pragma unroll
        for (int x = 0; x < PART; x++) {
            const half *Ax = A + (size_t)x * m + i;
            union_half8 a0, a1;
            a0.in = *reinterpret_cast<const uint4 *>(Ax);
            a1.in = *reinterpret_cast<const uint4 *>(Ax + 8);
            float gsum = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) gsum += (float)a0.out[j] * wval[j];
#pragma unroll
            for (int j = 0; j < 8; j++) gsum += (float)a1.out[j] * wval[8 + j];
            acc[x] += gsum;
        }
    }

#pragma unroll
    for (int x = 0; x < PART; x++) {
        float v = acc[x];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        acc[x] = v;
    }

    if (laneId == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float r = acc[x];
            if (bias != nullptr) r += (float)bias[st];
            C[st + (size_t)k * x] = (half)r;
        }
    }
}

void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias,
                                    half *scales, half *mins, int n, int m, int k,
                                    int group, int groupCnt, bool useZeroPoint) {
    // 满足 16 对齐时走 warp 优化版 GEMV (参照 FP8 warp kernel)。
    if ((m & 15) == 0 && groupCnt > 0 && (groupCnt & 15) == 0) {
        constexpr int W = 8;  // 每个 block 8 个 warp (256 线程)
        const int grid = (k + W - 1) / W;
#define FASTLLM_INT4G_WARP_LAUNCH(PARTVAL, OFF) \
        FastllmGemvHalfInt4GroupKernelWarpMultiRow<W, PARTVAL> <<< grid, W * 32 >>>( \
            input + (OFF) * m, weight, output + (OFF) * k, bias, scales, mins, m, k, group, groupCnt, useZeroPoint)
        switch (n) {
            case 1:  FASTLLM_INT4G_WARP_LAUNCH(1, 0);  return;
            case 2:  FASTLLM_INT4G_WARP_LAUNCH(2, 0);  return;
            case 3:  FASTLLM_INT4G_WARP_LAUNCH(3, 0);  return;
            case 4:  FASTLLM_INT4G_WARP_LAUNCH(4, 0);  return;
            case 5:  FASTLLM_INT4G_WARP_LAUNCH(5, 0);  return;
            case 6:  FASTLLM_INT4G_WARP_LAUNCH(6, 0);  return;
            case 7:  FASTLLM_INT4G_WARP_LAUNCH(7, 0);  return;
            case 8:  FASTLLM_INT4G_WARP_LAUNCH(8, 0);  return;
            default: break;
        }
        int i = 0;
        for (; i + 7 < n; i += 8) FASTLLM_INT4G_WARP_LAUNCH(8, i);
        for (; i + 3 < n; i += 4) FASTLLM_INT4G_WARP_LAUNCH(4, i);
        for (; i + 1 < n; i += 2) FASTLLM_INT4G_WARP_LAUNCH(2, i);
        for (; i < n; i++)        FASTLLM_INT4G_WARP_LAUNCH(1, i);
#undef FASTLLM_INT4G_WARP_LAUNCH
        return;
    }

    if (n == 1) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 2) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 3) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 4) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 5) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 6) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 7) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 8) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 9) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 10) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 11) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 12) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 13) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 14) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 15) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else if (n == 16) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 16> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt, useZeroPoint);
        }
        return;
    }
    
}

__global__ void FastllmCudaInt4GroupToMarlinQWeightKernel(const uint8_t *weight, uint32_t *qweight, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int packsPerRow = m / 8;
    int total = packsPerRow * k;
    if (idx >= total) {
        return;
    }

    int pack = idx / k;
    int out = idx - pack * k;
    int xBase = pack * 8;
    const uint8_t *row = weight + (size_t)out * m / 2;

    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t packed = row[(xBase + i) >> 1];
        uint32_t q = ((xBase + i) & 1) ? (packed & 15) : (packed >> 4);
        v |= q << (i * 4);
    }
    qweight[idx] = v;
}

static bool FastllmCudaInt4GroupMarlinEnabled(int n, int m, int k, int groupCnt) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int dev = 0;
    int major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
        major * 10 + minor < 75) {
        return false;
    }
    return n >= 1 && (groupCnt == 32 || groupCnt == 128) && m % groupCnt == 0 &&
           groupCnt % 16 == 0 && m % 64 == 0 && k % 64 == 0;
#endif
}

static void FastllmBuildMarlinPermutedScalesAndZeros(const fastllm::Data &weight,
                                                      std::vector<half> &scales,
                                                      std::vector<uint32_t> &zeros,
                                                      int m, int k) {
    int group = weight.group;
    scales.resize((size_t)group * k);
    std::vector<uint8_t> zeroValues((size_t)group * k);

    const int scalePerm[64] = {
        0, 8, 16, 24, 32, 40, 48, 56,
        1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,
        5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,
        7, 15, 23, 31, 39, 47, 55, 63
    };
    const int zpInterleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};

    std::vector<float> scaleGN((size_t)group * k);
    std::vector<uint8_t> zeroGN((size_t)group * k);
    for (int g = 0; g < group; g++) {
        for (int out = 0; out < k; out++) {
            size_t dst = (size_t)g * k + out;
            size_t src = (size_t)out * group + g;
            float s = weight.scales[src];
            float minv = weight.mins[src];
            int z = weight.zeros.size() == (size_t)k * group
                ? weight.zeros[src]
                : (s == 0.0f ? 0 : (int)std::lroundf(-minv / s));
            z = std::max(0, std::min(15, z));
            scaleGN[dst] = s;
            zeroGN[dst] = (uint8_t)z;
        }
    }

    for (size_t base = 0; base < scaleGN.size(); base += 64) {
#pragma unroll
        for (int i = 0; i < 64; i++) {
            scales[base + i] = (half)scaleGN[base + scalePerm[i]];
            zeroValues[base + i] = zeroGN[base + scalePerm[i]];
        }
    }

    for (size_t base = 0; base < zeroValues.size(); base += 8) {
        uint8_t tmp[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            tmp[i] = zeroValues[base + zpInterleave[i]];
        }
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            packed |= ((uint32_t)tmp[i]) << (i * 4);
        }
        zeros.push_back(packed);
    }
}

static bool FastllmCudaInt4GroupEnsureMarlinOnDevice(fastllm::Data &weight, int m, int k) {
    if (FastllmCudaInt4GroupHasMarlinOnDevice(weight)) {
        return true;
    }

    if (weight.group <= 0 || weight.groupCnt <= 0 ||
        weight.scales.size() != (size_t)k * weight.group ||
        weight.mins.size() != (size_t)k * weight.group ||
        weight.cudaData == nullptr) {
        return false;
    }

    size_t qweightCount = (size_t)(m / 8) * k;
    uint32_t *stdQWeight = (uint32_t*)FastllmCudaMalloc(qweightCount * sizeof(uint32_t));
    uint32_t *marlinQWeight = (uint32_t*)FastllmCudaMalloc(qweightCount * sizeof(uint32_t));

    int threads = 256;
    int blocks = (int)((qweightCount + threads - 1) / threads);
    FastllmCudaInt4GroupToMarlinQWeightKernel <<< blocks, threads >>>(
        (const uint8_t*)weight.cudaData, stdQWeight, m, k);

    bool repacked = FastllmCudaGptqMarlinRepack(stdQWeight, marlinQWeight, m, k);
    FastllmCudaFree(stdQWeight);
    if (!repacked) {
        FastllmCudaFree(marlinQWeight);
        return false;
    }
    FastllmCudaInt4GroupReleaseOriginalWeight(weight);
    // Repacking temporarily needs the source INT4 buffer and an unpacked GPTQ
    // buffer in addition to the final Marlin buffer. FastllmCudaFree normally
    // keeps both temporary allocations in the reusable pool; when many MoE
    // experts are prepared lazily that pool can retain several GB and starve
    // KV/prefill allocations. The final Marlin buffer is still marked busy, so
    // clearing idle pool entries here only releases the two obsolete buffers.
    FastllmCudaClearBigBuffer();

    std::vector<half> hostScales;
    std::vector<uint32_t> hostZeros;
    hostZeros.reserve((size_t)weight.group * k / 8);
    FastllmBuildMarlinPermutedScalesAndZeros(weight, hostScales, hostZeros, m, k);

    half *marlinScales = (half*)FastllmCudaMalloc(hostScales.size() * sizeof(half));
    uint32_t *marlinZeros = (uint32_t*)FastllmCudaMalloc(hostZeros.size() * sizeof(uint32_t));
    FastllmCudaCopyFromHostToDevice(marlinScales, hostScales.data(), hostScales.size() * sizeof(half));
    FastllmCudaCopyFromHostToDevice(marlinZeros, hostZeros.data(), hostZeros.size() * sizeof(uint32_t));

    int workspaceInts = std::max(1, (k / 64) * 16);
    int *workspace = (int*)FastllmCudaMalloc((size_t)workspaceInts * sizeof(int));
    FastllmCudaMemset0(workspace, (size_t)workspaceInts * sizeof(int));

    if ((int)weight.extraCudaData.size() <= INT4GROUP_MARLIN_WORKSPACE_IDX) {
        weight.extraCudaData.resize(INT4GROUP_MARLIN_WORKSPACE_IDX + 1, nullptr);
    }
    weight.extraCudaData[INT4GROUP_MARLIN_WEIGHT_IDX] = (void*)marlinQWeight;
    weight.extraCudaData[INT4GROUP_MARLIN_ZEROS_IDX] = (void*)marlinZeros;
    weight.extraCudaData[INT4GROUP_MARLIN_WORKSPACE_IDX] = (void*)workspace;

    if ((int)weight.extraCudaHalfData.size() <= INT4GROUP_MARLIN_SCALES_HALF_IDX) {
        weight.extraCudaHalfData.resize(INT4GROUP_MARLIN_SCALES_HALF_IDX + 1, nullptr);
    }
    weight.extraCudaHalfData[INT4GROUP_MARLIN_SCALES_HALF_IDX] = (void*)marlinScales;
    return true;
}

static half *FastllmCudaInt4GroupEnsureHalfBiasDataOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if ((int)weight.extraCudaHalfData.size() <= INT4GROUP_HALF_BIAS_IDX) {
        weight.extraCudaHalfData.resize(INT4GROUP_HALF_BIAS_IDX + 1, nullptr);
    }
    if (weight.extraCudaHalfData[INT4GROUP_HALF_BIAS_IDX] == nullptr) {
        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData[INT4GROUP_HALF_BIAS_IDX] = (void*)cudaBiasData;
    }
    return (half*)weight.extraCudaHalfData[INT4GROUP_HALF_BIAS_IDX];
}

static void FastllmCudaInt4GroupEnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(weight, bias, k);
    if ((int)weight.extraCudaHalfData.size() <= INT4GROUP_HALF_BIAS_IDX) {
        weight.extraCudaHalfData.resize(INT4GROUP_HALF_BIAS_IDX + 1, nullptr);
    }
    if (weight.extraCudaHalfData[INT4GROUP_HALF_SCALES_IDX] == nullptr) {
        weight.extraCudaHalfData[INT4GROUP_HALF_SCALES_IDX] =
            weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX];
    }
    if (weight.extraCudaHalfData[INT4GROUP_HALF_MINS_IDX] == nullptr) {
        weight.extraCudaHalfData[INT4GROUP_HALF_MINS_IDX] =
            weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX];
    }
    FastllmCudaInt4GroupEnsureHalfBiasDataOnDevice(weight, bias, k);
}

namespace {
    static constexpr int INT4GROUP_MOE_WARPS = 8;

    struct Int4GroupMoeBatch1Table {
        const uint8_t **gateupWeights = nullptr;
        const uint8_t **downWeights = nullptr;
        const half *gateupScales = nullptr;
        const half *gateupOffsets = nullptr;
        const half *downScales = nullptr;
        const half *downOffsets = nullptr;
        bool useZeroPoint = false;
        int expertCount = 0;
        int hidden = 0;
        int inter = 0;
        int gateupRows = 0;
        int gateupGroup = 0;
        int gateupGroupCnt = 0;
        int downGroup = 0;
        int downGroupCnt = 0;
    };

    static bool PrepareInt4GroupMoeBatch1Table(fastllm::Data **weights,
                                                int weightsBatch,
                                                Int4GroupMoeBatch1Table &table) {
        if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) != 0 ||
            weights[0] != nullptr || weights[1] != nullptr ||
            weights[2] == nullptr || weights[3] == nullptr) {
            return false;
        }

        const int expertCount = weightsBatch / 2 - 1;
        fastllm::Data &firstGateup = *weights[2];
        fastllm::Data &firstDown = *weights[3];
        if (firstGateup.dataType != fastllm::DataType::INT4_GROUP ||
            firstDown.dataType != fastllm::DataType::INT4_GROUP ||
            firstGateup.dims.size() != 2 || firstDown.dims.size() != 2 ||
            (firstGateup.dims[0] & 1) != 0) {
            return false;
        }

        const int gateupRows = firstGateup.dims[0];
        const int inter = gateupRows / 2;
        const int hidden = firstGateup.dims[1];
        if (expertCount <= 0 || inter <= 0 || hidden <= 0 ||
            firstDown.dims[0] != hidden || firstDown.dims[1] != inter ||
            hidden % 16 != 0 || inter % 16 != 0 ||
            firstGateup.group <= 0 || firstGateup.groupCnt <= 0 ||
            firstDown.group <= 0 || firstDown.groupCnt <= 0 ||
            firstGateup.groupCnt % 16 != 0 || firstDown.groupCnt % 16 != 0) {
            return false;
        }

        const int gateupGroup = firstGateup.group;
        const int gateupGroupCnt = firstGateup.groupCnt;
        const int downGroup = firstDown.group;
        const int downGroupCnt = firstDown.groupCnt;
        const size_t gateupPerExpert = (size_t)gateupRows * gateupGroup;
        const size_t downPerExpert = (size_t)hidden * downGroup;
        const bool useZeroPoint = firstGateup.zeros.size() == gateupPerExpert &&
                                  firstDown.zeros.size() == downPerExpert;

        for (int expert = 0; expert < expertCount; expert++) {
            fastllm::Data *gateup = weights[(expert + 1) * 2];
            fastllm::Data *down = weights[(expert + 1) * 2 + 1];
            if (gateup == nullptr || down == nullptr ||
                gateup->dataType != fastllm::DataType::INT4_GROUP ||
                down->dataType != fastllm::DataType::INT4_GROUP ||
                gateup->dims != firstGateup.dims || down->dims != firstDown.dims ||
                gateup->group != gateupGroup || gateup->groupCnt != gateupGroupCnt ||
                down->group != downGroup || down->groupCnt != downGroupCnt ||
                gateup->cudaData == nullptr || down->cudaData == nullptr ||
                gateup->scales.size() != gateupPerExpert || gateup->mins.size() != gateupPerExpert ||
                down->scales.size() != downPerExpert || down->mins.size() != downPerExpert ||
                (gateup->zeros.size() == gateupPerExpert) != useZeroPoint ||
                (down->zeros.size() == downPerExpert) != useZeroPoint) {
                return false;
            }
        }

        if ((int)firstGateup.extraCudaData.size() <= INT4GROUP_MOE_QUANT_META_IDX) {
            firstGateup.extraCudaData.resize(INT4GROUP_MOE_QUANT_META_IDX + 1, nullptr);
        }

        void *pointerBlock = firstGateup.extraCudaData[INT4GROUP_MOE_POINTER_TABLE_IDX];
        void *quantMetaBlock = firstGateup.extraCudaData[INT4GROUP_MOE_QUANT_META_IDX];
        if (pointerBlock == nullptr || quantMetaBlock == nullptr) {
            std::vector<uint8_t*> hostPointers((size_t)expertCount * 2);
            for (int expert = 0; expert < expertCount; expert++) {
                hostPointers[expert] = (uint8_t*)weights[(expert + 1) * 2]->cudaData;
                hostPointers[expertCount + expert] = (uint8_t*)weights[(expert + 1) * 2 + 1]->cudaData;
            }

            const size_t gateupCount = gateupPerExpert * expertCount;
            const size_t downCount = downPerExpert * expertCount;
            std::vector<half> hostMeta(gateupCount * 2 + downCount * 2);
            size_t gateupScaleOffset = 0;
            size_t gateupMinOffset = gateupCount;
            size_t downScaleOffset = gateupCount * 2;
            size_t downMinOffset = gateupCount * 2 + downCount;
            for (int expert = 0; expert < expertCount; expert++) {
                const fastllm::Data &gateup = *weights[(expert + 1) * 2];
                const fastllm::Data &down = *weights[(expert + 1) * 2 + 1];
                for (size_t i = 0; i < gateupPerExpert; i++) {
                    hostMeta[gateupScaleOffset + (size_t)expert * gateupPerExpert + i] = (half)gateup.scales[i];
                    hostMeta[gateupMinOffset + (size_t)expert * gateupPerExpert + i] =
                        useZeroPoint ? (half)gateup.zeros[i] : (half)gateup.mins[i];
                }
                for (size_t i = 0; i < downPerExpert; i++) {
                    hostMeta[downScaleOffset + (size_t)expert * downPerExpert + i] = (half)down.scales[i];
                    hostMeta[downMinOffset + (size_t)expert * downPerExpert + i] =
                        useZeroPoint ? (half)down.zeros[i] : (half)down.mins[i];
                }
            }

            pointerBlock = FastllmCudaMalloc(hostPointers.size() * sizeof(uint8_t*));
            quantMetaBlock = FastllmCudaMalloc(hostMeta.size() * sizeof(half));
            if (pointerBlock == nullptr || quantMetaBlock == nullptr) {
                if (pointerBlock != nullptr) FastllmCudaFree(pointerBlock);
                if (quantMetaBlock != nullptr) FastllmCudaFree(quantMetaBlock);
                return false;
            }
            FastllmCudaCopyFromHostToDevice(pointerBlock, hostPointers.data(),
                                            hostPointers.size() * sizeof(uint8_t*));
            FastllmCudaCopyFromHostToDevice(quantMetaBlock, hostMeta.data(),
                                            hostMeta.size() * sizeof(half));
            firstGateup.extraCudaData[INT4GROUP_MOE_POINTER_TABLE_IDX] = pointerBlock;
            firstGateup.extraCudaData[INT4GROUP_MOE_QUANT_META_IDX] = quantMetaBlock;
        }

        const size_t gateupCount = gateupPerExpert * expertCount;
        const size_t downCount = downPerExpert * expertCount;
        table.gateupWeights = (const uint8_t**)pointerBlock;
        table.downWeights = table.gateupWeights + expertCount;
        table.gateupScales = (const half*)quantMetaBlock;
        table.gateupOffsets = table.gateupScales + gateupCount;
        table.downScales = table.gateupOffsets + gateupCount;
        table.downOffsets = table.downScales + downCount;
        table.useZeroPoint = useZeroPoint;
        table.expertCount = expertCount;
        table.hidden = hidden;
        table.inter = inter;
        table.gateupRows = gateupRows;
        table.gateupGroup = gateupGroup;
        table.gateupGroupCnt = gateupGroupCnt;
        table.downGroup = downGroup;
        table.downGroupCnt = downGroupCnt;
        return true;
    }

    template <int WARPS_PER_BLOCK>
    __global__ void FastllmCudaInt4GroupMoeGateupSwigluBatch1Kernel(
            const half *input,
            const uint8_t *const *gateupWeights,
            const half *scales,
            const half *mins,
            const int32_t *indices,
            half *middle,
            int topk, int expertCount, int hidden, int inter,
            int gateupRows, int group, int groupCnt, bool useZeroPoint) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        const int totalTasks = topk * inter;
        if (task >= totalTasks) {
            return;
        }

        const int route = task / inter;
        const int out = task - route * inter;
        const int expert = __ldg(indices + route);
        float gateAcc = 0.0f;
        float upAcc = 0.0f;
        if (expert >= 0 && expert < expertCount) {
            const uint8_t *weight = gateupWeights[expert];
            if (weight != nullptr) {
                const int gateRow = out;
                const int upRow = inter + out;
                const uint8_t *gateWeight = weight + (size_t)gateRow * hidden / 2;
                const uint8_t *upWeight = weight + (size_t)upRow * hidden / 2;
                const half *gateScales = scales + ((size_t)expert * gateupRows + gateRow) * group;
                const half *gateMins = mins + ((size_t)expert * gateupRows + gateRow) * group;
                const half *upScales = scales + ((size_t)expert * gateupRows + upRow) * group;
                const half *upMins = mins + ((size_t)expert * gateupRows + upRow) * group;
                const int units = hidden >> 4;
                for (int unit = lane; unit < units; unit += 32) {
                    const int x = unit << 4;
                    const int gid = x / groupCnt;
                    const float gateScale = __half2float(__ldg(gateScales + gid));
                    const float gateMin = __half2float(__ldg(gateMins + gid));
                    const float upScale = __half2float(__ldg(upScales + gid));
                    const float upMin = __half2float(__ldg(upMins + gid));
                    union_char8 gatePacked, upPacked;
                    union_half8 input0, input1;
                    gatePacked.in = *reinterpret_cast<const uint2*>(gateWeight + (size_t)unit * 8);
                    upPacked.in = *reinterpret_cast<const uint2*>(upWeight + (size_t)unit * 8);
                    input0.in = *reinterpret_cast<const uint4*>(input + x);
                    input1.in = *reinterpret_cast<const uint4*>(input + x + 8);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int pair = (i & 3) * 2;
                        const float a0 = i < 4 ? __half2float(input0.out[pair])
                                               : __half2float(input1.out[pair]);
                        const float a1 = i < 4 ? __half2float(input0.out[pair + 1])
                                               : __half2float(input1.out[pair + 1]);
                        const uint8_t g0 = gatePacked.out[i];
                        const uint8_t u0 = upPacked.out[i];
                        gateAcc += a0 * FastllmCudaDequantInt4GroupHalfValue(
                            (float)(g0 >> 4), gateScale, gateMin, useZeroPoint);
                        gateAcc += a1 * FastllmCudaDequantInt4GroupHalfValue(
                            (float)(g0 & 15), gateScale, gateMin, useZeroPoint);
                        upAcc += a0 * FastllmCudaDequantInt4GroupHalfValue(
                            (float)(u0 >> 4), upScale, upMin, useZeroPoint);
                        upAcc += a1 * FastllmCudaDequantInt4GroupHalfValue(
                            (float)(u0 & 15), upScale, upMin, useZeroPoint);
                    }
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gateAcc += __shfl_down_sync(0xffffffff, gateAcc, offset);
            upAcc += __shfl_down_sync(0xffffffff, upAcc, offset);
        }
        if (lane == 0) {
            const float activated = gateAcc / (1.0f + expf(-gateAcc));
            middle[(size_t)route * inter + out] = __float2half_rn(activated * upAcc);
        }
    }

    template <int WARPS_PER_BLOCK>
    __global__ void FastllmCudaInt4GroupMoeDownReduceBatch1Kernel(
            const half *middle,
            const uint8_t *const *downWeights,
            const half *scales,
            const half *mins,
            const int32_t *indices,
            const float *scores,
            half *output,
            int topk, int expertCount, int hidden, int inter,
            int group, int groupCnt, bool useZeroPoint) {
        __shared__ float routeValues[WARPS_PER_BLOCK];
        const int route = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int out = blockIdx.x;
        float acc = 0.0f;
        if (route < topk) {
            const int expert = __ldg(indices + route);
            if (expert >= 0 && expert < expertCount) {
                const uint8_t *weight = downWeights[expert];
                if (weight != nullptr) {
                    const uint8_t *rowWeight = weight + (size_t)out * inter / 2;
                    const half *rowScales = scales + ((size_t)expert * hidden + out) * group;
                    const half *rowMins = mins + ((size_t)expert * hidden + out) * group;
                    const half *routeInput = middle + (size_t)route * inter;
                    const int units = inter >> 4;
                    for (int unit = lane; unit < units; unit += 32) {
                        const int x = unit << 4;
                        const int gid = x / groupCnt;
                        const float scale = __half2float(__ldg(rowScales + gid));
                        const float minValue = __half2float(__ldg(rowMins + gid));
                        union_char8 packed;
                        union_half8 input0, input1;
                        packed.in = *reinterpret_cast<const uint2*>(rowWeight + (size_t)unit * 8);
                        input0.in = *reinterpret_cast<const uint4*>(routeInput + x);
                        input1.in = *reinterpret_cast<const uint4*>(routeInput + x + 8);
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            const uint8_t q = packed.out[i];
                            const int pair = (i & 3) * 2;
                            const float a0 = i < 4 ? __half2float(input0.out[pair])
                                                   : __half2float(input1.out[pair]);
                            const float a1 = i < 4 ? __half2float(input0.out[pair + 1])
                                                   : __half2float(input1.out[pair + 1]);
                            acc += a0 * FastllmCudaDequantInt4GroupHalfValue(
                                (float)(q >> 4), scale, minValue, useZeroPoint);
                            acc += a1 * FastllmCudaDequantInt4GroupHalfValue(
                                (float)(q & 15), scale, minValue, useZeroPoint);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            routeValues[route] = route < topk ? acc * __ldg(scores + route) : 0.0f;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                sum += routeValues[i];
            }
            output[out] = __float2half_rn(sum);
        }
    }
}

bool FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(const fastllm::Data &input,
                                                   fastllm::Data &scratch,
                                                   fastllm::Data &output,
                                                   fastllm::Data **weights,
                                                   int weightsBatch,
                                                   const int32_t *indices,
                                                   const float *scores,
                                                   int topk) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        input.dims.size() != 2 || input.dims[0] != 1 ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        indices == nullptr || scores == nullptr ||
        topk <= 0 || topk > INT4GROUP_MOE_WARPS) {
        return false;
    }

    Int4GroupMoeBatch1Table table;
    if (!PrepareInt4GroupMoeBatch1Table(weights, weightsBatch, table) ||
        input.dims[1] != table.hidden) {
        return false;
    }

    scratch.dataType = fastllm::DataType::FLOAT16;
    scratch.dataDevice = fastllm::DataDevice::CUDA;
    scratch.dataDeviceIds = input.dataDeviceIds;
    scratch.Resize({topk, table.inter});
    scratch.Allocate(false);
    if (scratch.cudaData == nullptr) {
        return false;
    }

    const int gateupTasks = topk * table.inter;
    FastllmCudaInt4GroupMoeGateupSwigluBatch1Kernel<INT4GROUP_MOE_WARPS>
        <<< (gateupTasks + INT4GROUP_MOE_WARPS - 1) / INT4GROUP_MOE_WARPS,
             INT4GROUP_MOE_WARPS * 32, 0, cudaStreamPerThread >>>(
            (const half*)input.cudaData, table.gateupWeights,
            table.gateupScales, table.gateupOffsets, indices,
            (half*)scratch.cudaData, topk, table.expertCount,
            table.hidden, table.inter, table.gateupRows,
            table.gateupGroup, table.gateupGroupCnt, table.useZeroPoint);
    FastllmCudaInt4GroupMoeDownReduceBatch1Kernel<INT4GROUP_MOE_WARPS>
        <<< table.hidden, INT4GROUP_MOE_WARPS * 32, 0, cudaStreamPerThread >>>(
            (const half*)scratch.cudaData, table.downWeights,
            table.downScales, table.downOffsets, indices, scores,
            (half*)output.cudaData, topk, table.expertCount,
            table.hidden, table.inter, table.downGroup, table.downGroupCnt,
            table.useZeroPoint);
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA INT4_GROUP batch-1 fused MoE failed.", state);
        return false;
    }
    return true;
#endif
}

bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    const bool useZeroPoint = weight.zeros.size() == (size_t)k * group;
    // Individual routed-expert tensors are consumed by the batch-1 fused MoE
    // kernel during decode. Keep their source AWQ layout intact during prefill;
    // converting even one selected expert to Marlin would invalidate the
    // device-side expert pointer table needed by the fused path.
    bool routedMoeWeight = weight.name.find(".mlp.experts.") != std::string::npos ||
                           weight.name.find(".block_sparse_moe.experts.") != std::string::npos;
    bool useMarlin = weight.zeros.size() == (size_t)k * group &&
                     !routedMoeWeight &&
                     FastllmCudaInt4GroupMarlinEnabled(n, m, k, groupCnt) &&
                     FastllmCudaInt4GroupEnsureMarlinOnDevice(weight, m, k);
    bool useSm70Awq = !useMarlin && weight.zeros.size() == (size_t)k * group &&
                      FastllmCudaInt4GroupSm70AwqEnabled(n, m, k, groupCnt) &&
                      FastllmCudaInt4GroupEnsureSm70AwqOnDevice(weight, m, k);
    if (!useMarlin && !useSm70Awq) {
        if (weight.cudaData == nullptr) {
            FastllmCudaInt4GroupFallbackUnavailable();
        }
        FastllmCudaInt4GroupEnsureHalfBiasOnDevice(weight, bias, k);
    }

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (useSm70Awq) {
        bool ok = fastllm::awq_sm70::Gemm(g_sm70AwqHandles[&weight], cudaInput, cudaOutput, n, 0);
        if (!ok) {
            printf("Error: INT4_GROUP SM70 AWQ GEMM failed.\n");
            throw("int4group sm70 awq gemm error");
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasData = FastllmCudaInt4GroupEnsureHalfBiasDataOnDevice(weight, bias, k);
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }
    } else if (useMarlin) {
        uint32_t *marlinQWeight = (uint32_t*)weight.extraCudaData[INT4GROUP_MARLIN_WEIGHT_IDX];
        uint32_t *marlinZeros = (uint32_t*)weight.extraCudaData[INT4GROUP_MARLIN_ZEROS_IDX];
        int *marlinWorkspace = (int*)weight.extraCudaData[INT4GROUP_MARLIN_WORKSPACE_IDX];
        half *marlinScales = (half*)weight.extraCudaHalfData[INT4GROUP_MARLIN_SCALES_HALF_IDX];

        bool marlinOk = FastllmCudaMarlinHalfInt4Gemm(cudaInput, marlinQWeight, marlinScales, marlinZeros,
                                                      cudaOutput, n, k, m, groupCnt, marlinWorkspace);
        if (!marlinOk) {
            printf("Error: INT4_GROUP Marlin GEMM failed after original CUDA weight was released.\n");
            throw("int4group marlin gemm error");
            exit(0);
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasData = FastllmCudaInt4GroupEnsureHalfBiasDataOnDevice(weight, bias, k);
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }
    } else if (n > 16) {
        half *cudaScales = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_SCALES_IDX];
        half *cudaMins = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_MINS_IDX];
        auto fastllmCublasHandle = getFastllmCublasHandle();

        // 借用 FlashInfer 的 d_float_workspace 作为 INT4 -> FP16 的反量化临时缓冲；
        // 两次 attention 之间该 workspace 内容是无效的（attention 入口会重新 plan 覆盖）。
        // 如果一次 dequant 不下整张 weight (k*m*2B)，按 K 维分块多次执行。
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

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);

            FastllmCudaInt4Group2HalfKernel <<< kc, 256 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m / 2,
                cudaScales + (size_t)kOff * group,
                cudaMins + (size_t)kOff * group,
                cudaFp16Weight, kc, m, group, groupCnt, useZeroPoint);

#ifdef CUDA_NO_TENSOR_CORE
            // 子矩阵写入 cudaFp32Output 的 [kOff:kOff+kc, :] 行段，ldc 仍为 k
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
                printf("Error: cublas error. status = %d\n", status);
                throw("cublas error");
                exit(0);
            }
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_BIAS_IDX];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        half *cudaScales = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_SCALES_IDX];
        half *cudaMins = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_MINS_IDX];
        half *cudaBiasData = (half*)weight.extraCudaHalfData[INT4GROUP_HALF_BIAS_IDX];
        LaunchFastllmGemmFp16Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData,
                                       cudaScales, cudaMins, n, m, k, group, groupCnt, useZeroPoint);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// ==================== FLOAT16 x INT4_GROUP128 ====================
// INT4_GROUP128 数据布局: 每行按128个元素分组
// 每个group: [64B int4 data] [4B float min] [4B float scale]，共72字节
// int4数据经过CPU端重排:
//   AVX512VNNI: 每32字节(64元素), packed[k]低4位=元素k, 高4位=元素k+32, halfBlock=32
//   AVX2:       每16字节(32元素), packed[k]低4位=元素k, 高4位=元素k+16, halfBlock=16

static int GetInt4Group128HalfBlock() {
    static int halfBlock = -1;
    if (halfBlock < 0) {
#if !defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        bool hasAVX512VNNI = (ecx & (1 << 11)) != 0;
        if (hasAVX512VNNI) {
            halfBlock = 32;
        } else {
            __get_cpuid(1, &eax, &ebx, &ecx, &edx);
            bool hasAVX2 = false;
            if (ecx & (1 << 27)) {
                unsigned int eax7, ebx7, ecx7, edx7;
                __get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7);
                hasAVX2 = (ebx7 & (1 << 5)) != 0;
            }
            halfBlock = hasAVX2 ? 16 : 0;
        }
#elif defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, 7, 0);
        bool hasAVX512VNNI = (regs[2] & (1 << 11)) != 0;
        if (hasAVX512VNNI) {
            halfBlock = 32;
        } else {
            __cpuid(regs, 1);
            bool hasAVX2 = false;
            if (regs[2] & (1 << 27)) {
                __cpuidex(regs, 7, 0);
                hasAVX2 = (regs[1] & (1 << 5)) != 0;
            }
            halfBlock = hasAVX2 ? 16 : 0;
        }
#else
        halfBlock = 0;
#endif
    }
    return halfBlock;
}

// halfBlock: 重排的半块大小 (32=AVX512VNNI, 16=AVX2, 0=无重排)
__global__ void FastllmCudaInt4Group1282HalfKernel(uint8_t *a, half *b, int k, int m, int halfBlock) {
    const int groupCnt = 128;
    const int groupStride = groupCnt / 2 + sizeof(float) * 2;
    int groups = m / groupCnt;

    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.x;

    for (int g = 0; g < groups; g++) {
        uint8_t *groupData = a + row * groups * groupStride + g * groupStride;
        float minVal = *(float *)(groupData + groupCnt / 2);
        float scaleVal = *(float *)(groupData + groupCnt / 2 + sizeof(float));

        for (int i = tid; i < groupCnt / 2; i += blockDim.x) {
            uint8_t packed = groupData[i];
            if (halfBlock > 0) {
                int blockId = i / halfBlock;
                int offset = i % halfBlock;
                int outBase = row * m + g * groupCnt + blockId * halfBlock * 2;
                b[outBase + offset] = __float2half(scaleVal * (packed & 0xF) + minVal);
                b[outBase + offset + halfBlock] = __float2half(scaleVal * (packed >> 4) + minVal);
            } else {
                int outIdx = row * m + g * groupCnt + i * 2;
                b[outIdx] = __float2half(scaleVal * (packed >> 4) + minVal);
                b[outIdx + 1] = __float2half(scaleVal * (packed & 0xF) + minVal);
            }
        }
    }
}

// halfBlock: 重排的半块大小 (32=AVX512VNNI, 16=AVX2)
// 重排后: 每halfBlock字节中, packed[k]低4位=元素k, 高4位=元素k+halfBlock
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfInt4Group128KernelMultiRow(half *A, uint8_t *B, half *C,
                                             half *bias, int m, int k, int halfBlock) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    const int groupCnt = 128;
    const int groupStride = groupCnt / 2 + sizeof(float) * 2;
    int groups = m / groupCnt;
    int numBlocks = groupCnt / 2 / halfBlock;

    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    for (int g = 0; g < groups; g++) {
        uint8_t *groupData = B + st * groups * groupStride + g * groupStride;
        float minVal = *(float *)(groupData + groupCnt / 2);
        float scaleVal = *(float *)(groupData + groupCnt / 2 + sizeof(float));

        for (int blk = 0; blk < numBlocks; blk++) {
            for (int i = tid; i < halfBlock / 4; i += THREAD_PER_BLOCK) {
                int byteOff = blk * halfBlock + i * 4;
                union_char4 bBuffer;
                bBuffer.in = *reinterpret_cast<const uint32_t *>(groupData + byteOff);

                int elemBase = g * groupCnt + blk * halfBlock * 2;
                int loBase = elemBase + i * 4;
                int hiBase = elemBase + i * 4 + halfBlock;

                for (int x = 0; x < PART; x++) {
                    float aLo0 = (float)A[x * m + loBase + 0];
                    float aLo1 = (float)A[x * m + loBase + 1];
                    float aLo2 = (float)A[x * m + loBase + 2];
                    float aLo3 = (float)A[x * m + loBase + 3];
                    float aHi0 = (float)A[x * m + hiBase + 0];
                    float aHi1 = (float)A[x * m + hiBase + 1];
                    float aHi2 = (float)A[x * m + hiBase + 2];
                    float aHi3 = (float)A[x * m + hiBase + 3];

                    sdata[x][tid] += aLo0 * (minVal + scaleVal * (bBuffer.out[0] & 15))
                                   + aHi0 * (minVal + scaleVal * (bBuffer.out[0] >> 4));
                    sdata[x][tid] += aLo1 * (minVal + scaleVal * (bBuffer.out[1] & 15))
                                   + aHi1 * (minVal + scaleVal * (bBuffer.out[1] >> 4));
                    sdata[x][tid] += aLo2 * (minVal + scaleVal * (bBuffer.out[2] & 15))
                                   + aHi2 * (minVal + scaleVal * (bBuffer.out[2] >> 4));
                    sdata[x][tid] += aLo3 * (minVal + scaleVal * (bBuffer.out[3] & 15))
                                   + aHi3 * (minVal + scaleVal * (bBuffer.out[3] >> 4));
                }
            }
        }
    }

    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[x][tid] += sdata[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + st)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
}

// 无重排版本的GEMV kernel
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfInt4Group128KernelMultiRowNoRepack(half *A, uint8_t *B, half *C,
                                             half *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    const int groupCnt = 128;
    const int groupStride = groupCnt / 2 + sizeof(float) * 2;
    int groups = m / groupCnt;

    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    for (int g = 0; g < groups; g++) {
        uint8_t *groupData = B + st * groups * groupStride + g * groupStride;
        float minVal = *(float *)(groupData + groupCnt / 2);
        float scaleVal = *(float *)(groupData + groupCnt / 2 + sizeof(float));

        for (int i = tid; i < groupCnt / 8; i += THREAD_PER_BLOCK) {
            union_char4 bBuffer;
            bBuffer.in = *reinterpret_cast<const uint32_t *>(groupData + i * 4);

            int baseIdx = g * groupCnt + i * 8;
            for (int x = 0; x < PART; x++) {
                union_half8 aBuffer;
                aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + baseIdx);
                sdata[x][tid] += ((float)aBuffer.out[0] * (minVal + scaleVal * (bBuffer.out[0] >> 4))
                             + (float)aBuffer.out[1] * (minVal + scaleVal * (bBuffer.out[0] & 15)));
                sdata[x][tid] += ((float)aBuffer.out[2] * (minVal + scaleVal * (bBuffer.out[1] >> 4))
                             + (float)aBuffer.out[3] * (minVal + scaleVal * (bBuffer.out[1] & 15)));
                sdata[x][tid] += ((float)aBuffer.out[4] * (minVal + scaleVal * (bBuffer.out[2] >> 4))
                             + (float)aBuffer.out[5] * (minVal + scaleVal * (bBuffer.out[2] & 15)));
                sdata[x][tid] += ((float)aBuffer.out[6] * (minVal + scaleVal * (bBuffer.out[3] >> 4))
                             + (float)aBuffer.out[7] * (minVal + scaleVal * (bBuffer.out[3] & 15)));
            }
        }
    }

    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[x][tid] += sdata[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + st)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
}

#define LAUNCH_GEMV_INT4G128(TPB, N, ...) \
    FastllmGemvHalfInt4Group128KernelMultiRow<TPB, N> <<< k, TPB >>>(__VA_ARGS__)

#define LAUNCH_GEMV_INT4G128_NOREPACK(TPB, N, ...) \
    FastllmGemvHalfInt4Group128KernelMultiRowNoRepack<TPB, N> <<< k, TPB >>>(__VA_ARGS__)

void LaunchFastllmGemmFp16Int4Group128(half *input, uint8_t *weight, half *output, half *bias, int n, int m, int k, int halfBlock) {
    if (halfBlock > 0) {
        if (n == 1) { LAUNCH_GEMV_INT4G128(128, 1, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 2) { LAUNCH_GEMV_INT4G128(128, 2, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 3) { LAUNCH_GEMV_INT4G128(128, 3, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 4) { LAUNCH_GEMV_INT4G128(128, 4, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 5) { LAUNCH_GEMV_INT4G128(128, 5, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 6) { LAUNCH_GEMV_INT4G128(128, 6, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 7) { LAUNCH_GEMV_INT4G128(128, 7, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 8) { LAUNCH_GEMV_INT4G128(128, 8, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 9) { LAUNCH_GEMV_INT4G128(128, 9, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 10) { LAUNCH_GEMV_INT4G128(128, 10, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 11) { LAUNCH_GEMV_INT4G128(128, 11, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 12) { LAUNCH_GEMV_INT4G128(128, 12, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 13) { LAUNCH_GEMV_INT4G128(128, 13, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 14) { LAUNCH_GEMV_INT4G128(128, 14, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 15) { LAUNCH_GEMV_INT4G128(128, 15, input, weight, output, bias, m, k, halfBlock); }
        else if (n == 16) { LAUNCH_GEMV_INT4G128(128, 16, input, weight, output, bias, m, k, halfBlock); }
        else {
            for (int i = 0; i < n; i++) {
                LAUNCH_GEMV_INT4G128(128, 1, input + i * m, weight, output + i * k, bias, m, k, halfBlock);
            }
        }
    } else {
        if (n == 1) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 1, input, weight, output, bias, m, k); }
        else if (n == 2) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 2, input, weight, output, bias, m, k); }
        else if (n == 3) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 3, input, weight, output, bias, m, k); }
        else if (n == 4) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 4, input, weight, output, bias, m, k); }
        else if (n == 5) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 5, input, weight, output, bias, m, k); }
        else if (n == 6) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 6, input, weight, output, bias, m, k); }
        else if (n == 7) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 7, input, weight, output, bias, m, k); }
        else if (n == 8) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 8, input, weight, output, bias, m, k); }
        else if (n == 9) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 9, input, weight, output, bias, m, k); }
        else if (n == 10) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 10, input, weight, output, bias, m, k); }
        else if (n == 11) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 11, input, weight, output, bias, m, k); }
        else if (n == 12) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 12, input, weight, output, bias, m, k); }
        else if (n == 13) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 13, input, weight, output, bias, m, k); }
        else if (n == 14) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 14, input, weight, output, bias, m, k); }
        else if (n == 15) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 15, input, weight, output, bias, m, k); }
        else if (n == 16) { LAUNCH_GEMV_INT4G128_NOREPACK(64, 16, input, weight, output, bias, m, k); }
        else {
            for (int i = 0; i < n; i++) {
                LAUNCH_GEMV_INT4G128_NOREPACK(64, 1, input + i * m, weight, output + i * k, bias, m, k);
            }
        }
    }
}

static void FastllmCudaInt4Group128EnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.extraCudaHalfData.size() == 0) {
        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void*)cudaBiasData);
    }
}

bool FastllmCudaHalfMatMulFloatInt4Group128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaInt4Group128EnsureHalfBiasOnDevice(weight, bias, k);

    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half*)weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    int halfBlock = GetInt4Group128HalfBlock();

    if (n > 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        // INT4_GROUP128 的物理 layout：每行字节数 = (m/128) * 72
        const int kGroupCnt = 128;
        const int kGroupStride = kGroupCnt / 2 + (int)sizeof(float) * 2;
        const int kRowBytes = (m / kGroupCnt) * kGroupStride;

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc((size_t)n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);

            FastllmCudaInt4Group1282HalfKernel <<< kc, 256 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * kRowBytes,
                cudaFp16Weight, kc, m, halfBlock);

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
                printf("Error: cublas error. status = %d\n", status);
                throw("cublas error");
                exit(0);
            }
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        LaunchFastllmGemmFp16Int4Group128(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k, halfBlock);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
