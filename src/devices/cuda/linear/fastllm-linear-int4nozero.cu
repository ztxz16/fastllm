//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

__global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float2 scalesBuffer;
    float2 minBuffer;
    int threshold = ST128_FP16_COUNT;
    for (int index = idx * ST128_FP16_COUNT; index < len; index += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
        int startIdx = index / per;
        int endIdx = (index + ST128_FP16_COUNT - 1) / per;
        scalesBuffer.x = scalesBuffer.y = __ldg(scales + startIdx);
        minBuffer.x = minBuffer.y = __ldg(mins + startIdx);
        if (endIdx > startIdx) {
            threshold = (idx + ST128_FP16_COUNT - 1) % per;
            scalesBuffer.y = __ldg(scales + endIdx);
            minBuffer.y = __ldg(mins + endIdx);
        }
        // 读取
        union_char4 aBuffer;
        union_half8 bBuffer;
        aBuffer.in = *reinterpret_cast<const uint32_t *>(a + index / 2);
        // 处理
        for (int i = 0; i < ST128_FP16_COUNT / 2; i++) {
            if (index + i * 2 + 1 < len) {
                float scale = i * 2 < threshold ? scalesBuffer.x : scalesBuffer.y;
                float min = i * 2 < threshold ? minBuffer.x : minBuffer.y;
                bBuffer.out[i * 2] = __float2half(scale * (aBuffer.out[i] >> 4) + min);
                bBuffer.out[i * 2 + 1] = __float2half(scale * (aBuffer.out[i] & 0xF) + min);
            }
            // if (a[index + i] != aBuffer.out[i] && index < 100)
                // printf("%d - %d : %d\n", index + i, a[index + i], aBuffer.out[i]);
        }
        reinterpret_cast<uint4 *>(b)[idx] = bBuffer.in;
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel2(half *A, uint8_t *B, half *C,
                                                half *bias, float *scales, float *mins,
                                                int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        float minv = mins[p] / scales[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += ((float)A[i * 2] * (minv + (now >> 4)) + (float)A[i * 2 + 1] * (minv + (now & 15)));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = (half)(sdata[0] * scales[p]);
            } else {
                C[p] = (half)(sdata[0] * scales[p] + (float)bias[p]);
            }
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1MultiRow(float *A, uint8_t *B, float *C,
                                                     float *bias, float *scales, float *mins,
                                                     int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const uint8_t *baseB = B + p * m / 2;
    float minv = __ldg(mins + p) / __ldg(scales + p);
    for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
        uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(baseB + i);
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float4 aBuffer = FETCH_FLOAT4(A[i * 2 + x * m]);
            sdata[x][tid] += aBuffer.x * (minv + ((bBuffer >> 4) & 15)) + aBuffer.y * (minv + (bBuffer & 15));
            sdata[x][tid] += aBuffer.z * (minv + (bBuffer >> 12)) + aBuffer.w * (minv + ((bBuffer >> 8) & 15));
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
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] * scales[p];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] * scales[p] + bias[p];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel1MultiRow(half *A, uint8_t *B, half *C,
                                                     half *bias, float *scales, float *mins,
                                                     int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    union_char4 bBuffer;
    float minv = __ldg(mins + p) / __ldg(scales + p);

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *reinterpret_cast<const uint32_t *>(B + st * m / 2 + i * 4);
        // uint8_t now0 = B[st * m / 2 + i * 4];
        // uint8_t now1 = B[st * m / 2 + i * 4 + 1];
        // uint8_t now2 = B[st * m / 2 + i * 4 + 2];
        // uint8_t now3 = B[st * m / 2 + i * 4 + 3];
        for (int x = 0; x < PART; x++) {
            union_half8 aBuffer;
            aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + i * 8);
            sdata[x][tid] += (__low2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] >> 4)) 
                         + __high2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] >> 4)) 
                         + __high2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] >> 4)) 
                         + __high2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] >> 4)) 
                         + __high2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] & 15)));
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
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] * scales[p]);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] * scales[p] + float(bias[p]));
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1(float *A, uint8_t *B, float *C,
                                             float *bias, float *scales, float *mins,
                                             int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        const uint8_t *baseB = B + p * m / 2;
        float minv = __ldg(mins + p) / __ldg(scales + p);
        for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
            float4 aBuffer = FETCH_FLOAT4(A[i * 2]);
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(baseB + i);
            sdata[tid] += aBuffer.x * (minv + ((bBuffer >> 4) & 15)) + aBuffer.y * (minv + (bBuffer & 15));
            sdata[tid] += aBuffer.z * (minv + (bBuffer >> 12)) + aBuffer.w * (minv + ((bBuffer >> 8) & 15));
        }
        __syncthreads();

        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }
        //if (tid <= 32)
            //warpReduce(sdata, tid);
        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = sdata[0] * scales[p];
            } else {
                C[p] = sdata[0] * scales[p] + bias[p];
            }
        }
        __syncthreads();
    }
}

static void FastllmCudaInt4NoZeroEnsureScalesMinsBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaScales;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        float *cudaMins;
        state = cudaMalloc(&cudaMins, k * sizeof(float));
        float *mins = new float[k];
        for (int i = 0; i < k; i++) {
            mins[i] = weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * sizeof(float), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData.push_back((void*)cudaMins);

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

static void FastllmCudaInt4NoZeroEnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        FastllmCudaInt4NoZeroEnsureScalesMinsBiasOnDevice(weight, bias, k);
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[1]);

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

void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt4NoZeroKernel1<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
        return;
    }
}

bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaInt4NoZeroEnsureScalesMinsBiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaData[0];
    float *cudaMins = (float*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                   cudaScales, cudaMins,
                                                                   cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
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
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int4NoZero(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4NoZero(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 1> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 2> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 3> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 4> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 5> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 6> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 7> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp16Int4NoZeroKernel2<64, 1> <<< k / 1, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
    }
}

bool FastllmCudaHalfMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaInt4NoZeroEnsureScalesMinsBiasOnDevice(weight, bias, k);
    FastllmCudaInt4NoZeroEnsureHalfBiasOnDevice(weight, bias, k);
    float *cudaScales = (float*)weight.extraCudaHalfData[0];
    float *cudaMins = (float*)weight.extraCudaHalfData[1];

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
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

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                    cudaScales,
                                                                    cudaMins,
                                                                    cudaFp16Weight, len, m);

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
            half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
        LaunchFastllmGemmFp16Int4NoZero(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
