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
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2bfloat16_rn(sdata[x][0] + __bfloat162float(__ldg(bias + p)));
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
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
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
__global__ void FastllmGemvFp32Bf16Kernel2MultiRow(float *A, __nv_bfloat16 *B, float *C, float *bias, int m, int k) {
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
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
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
    if (n == 1) {
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
    if (n == 1) {
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
    if (n == 1) {
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

    if (n < 8) {
        LaunchFastllmGemmFp16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else if (false) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = k * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaBF162HalfKernel <<<(len - 1) / (threadPerBlock) + 1, threadPerBlock>>>((__nv_bfloat16 *)weight.cudaData, cudaFp16Weight, len);
#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaFp32Output, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
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

    if (n < 8) {
        LaunchFastllmGemmBf16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
        __nv_bfloat16 h_alpha = __float2bfloat16_rn(1.0f), h_beta = __float2bfloat16_rn(0.0f);
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
