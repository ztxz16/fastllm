//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

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

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, half *scales, half *mins, half *b, int k, int m, int group, int groupCnt) {
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
                bBuffer.out[j * 2] = __float2half(scale * (aBuffer.out[j] >> 4) + min);
                bBuffer.out[j * 2 + 1] = __float2half(scale * (aBuffer.out[j] & 0xF) + min);
            }
        }
        reinterpret_cast<uint4 *>(b)[index / ST128_FP16_COUNT] = bBuffer.in;
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4GroupKernel3(float *A, uint8_t *B, float *C,
                                             float *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt) {
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
            sdata[p - st][tid] += aBuffer.x * (curmin + curscale * (float)((bBuffer >> 4) & 15)) 
                         + aBuffer.y * (curmin + curscale * (float)(bBuffer & 15));
            sdata[p - st][tid] += aBuffer.z * (curmin + curscale * (float)(bBuffer >> 12)) 
                         + aBuffer.w * (curmin + curscale * (float)((bBuffer >> 8) & 15));
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
                                             int m, int k, int group, int groupCnt) {
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
            sdata[p - st][tid] += (aBuffer.x * (curmin + (float)curscale * (now0 >> 4)) 
                         + aBuffer.y * (curmin + (float)curscale * (now0 & 15)));
            sdata[p - st][tid] += (aBuffer.z * (curmin + (float)curscale * (now1 >> 4)) 
                         + aBuffer.w * (curmin + (float)curscale * (now1 & 15)));
            sdata[p - st][tid] += (bBuffer.x * (curmin + (float)curscale * (now2 >> 4)) 
                         + bBuffer.y * (curmin + (float)curscale * (now2 & 15)));
            sdata[p - st][tid] += (bBuffer.z * (curmin + (float)curscale * (now3 >> 4)) 
                         + bBuffer.w * (curmin + (float)curscale * (now3 & 15)));
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
                                             int m, int k, int group, int groupCnt) {
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
            sdata[x][tid] += ((float)aBuffer.out[0] * (curmin + curscale * (bBuffer.out[0] >> 4)) 
                         + (float)aBuffer.out[1] * (curmin + curscale * (bBuffer.out[0] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[2] * (curmin + curscale * (bBuffer.out[1] >> 4)) 
                         + (float)aBuffer.out[3] * (curmin + curscale * (bBuffer.out[1] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[4] * (curmin + curscale * (bBuffer.out[2] >> 4)) 
                         + (float)aBuffer.out[5] * (curmin + curscale * (bBuffer.out[2] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[6] * (curmin + curscale * (bBuffer.out[3] >> 4)) 
                         + (float)aBuffer.out[7] * (curmin + curscale * (bBuffer.out[3] & 15)));
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

void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    for (int i = 0; i < n; i++) {
#ifdef CUDA_NO_TENSOR_CORE
        FastllmGemvInt4GroupKernel3<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#else
        FastllmGemvInt4GroupKernel2<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#endif
    }
}

static void FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        int group = weight.group;

        half *cudaScales;
        state = cudaMalloc(&cudaScales, k * group * sizeof(half));
        half *scales = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            scales[i] = (half)weight.scales[i];
        }
        state = cudaMemcpy(cudaScales, scales, k * group * sizeof(half), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);
        delete[] scales;

        half *cudaMins;
        state = cudaMalloc(&cudaMins, k * group * sizeof(half));
        half *mins = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            mins[i] = (half)weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * group * sizeof(half), cudaMemcpyHostToDevice);
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

bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, 
                                    int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(weight, bias, k);

    half *cudaScales = (half*)weight.extraCudaData[0];
    half *cudaMins = (half*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    if (n >= 8) {
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
        FastllmCudaInt4Group2HalfKernel <<< k, 64 >>>((uint8_t*)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

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
            printf("Error: cublas error. status = %d\n", status);
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
        LaunchFastllmGemmFp32Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    if (n == 1) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 2) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 3) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 4) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 5) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 6) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 7) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 8) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 9) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 10) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 11) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 12) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 13) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 14) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 15) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 16) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 16> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
        }
        return;
    }
    
}

static void FastllmCudaInt4GroupEnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(weight, bias, k);
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
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

bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    FastllmCudaInt4GroupEnsureScalesMinsAndBiasOnDevice(weight, bias, k);
    FastllmCudaInt4GroupEnsureHalfBiasOnDevice(weight, bias, k);

    half *cudaScales = (half*)weight.extraCudaHalfData[0];
    half *cudaMins = (half*)weight.extraCudaHalfData[1];

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n > 16) {
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

        FastllmCudaInt4Group2HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

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
            printf("Error: cublas error. status = %d\n", status);
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
        LaunchFastllmGemmFp16Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
