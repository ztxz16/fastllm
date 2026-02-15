//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

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

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Fp16Kernel2MultiRow(half *A, half *B, half *C, half *bias, int m, int k, bool addTo) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    union_half8 regA;
    union_half8 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
        
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
            for (int x = 0; x < PART; x++) {
                float val = sdata[x][0] + (float)(__ldg(bias + p));
                C[p + k * x] = addTo ? (half)(val + (float)C[p + k * x]) : (half)val;
            }
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float val = sdata[x][0];
                C[p + k * x] = addTo ? (half)(val + (float)C[p + k * x]) : (half)val;
            }
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
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
        
    const half *baseB = B + p * m;
    if (m % 4 == 0) {
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
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
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
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
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

void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k, bool addTo) {
    if (n == 1) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 2) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 3) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 4) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 5) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 6) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else if (n == 7) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k, addTo);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16: n > 7.\n");
        exit(0);
    }
}

bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo) {
    FastllmCudaFP16EnsureBiasOnDevice(weight, bias, k);
    FastllmCudaFP16EnsureBiasHalfOnDevice(weight, bias, k);

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];

    if (n < 8) {
        LaunchFastllmGemmFp16Fp16(cudaInput, (half*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k, addTo);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        status = cublasGemmEx(fastllmCublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            k, n, m,
                            &h_alpha, (half *) weight.cudaData, AType,
                            m, cudaInput, BType,
                            m, &h_beta,
                            cudaFp32Output, CType,
                            k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = addTo ? __float2half_rn(1.0) : __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        status = cublasGemmEx(fastllmCublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            k, n, m,
                            &h_alpha, (half *) weight.cudaData, AType,
                            m, cudaInput, BType,
                            m, &h_beta,
                            cudaOutput, CType,
                            k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw ("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        if (addTo) {
            half *cudaTempOutput = (half *) FastllmCudaMalloc(len * sizeof(half));
            FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaTempOutput, len);
            FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaOutput, cudaTempOutput, __float2half_rn(1.0), len);
            FastllmCudaFree(cudaTempOutput);
        } else {
            FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        }
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, (half *) weight.extraCudaHalfData[0], k);
        }
    }

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

    output.Allocate();

    int biasK = k * 2; // weight has 2*k rows, bias has 2*k elements
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

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaBf16ToHalfKernelFP16 <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        cublasStatus_t status;
#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *)FastllmCudaMalloc(n * k * sizeof(float));
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
