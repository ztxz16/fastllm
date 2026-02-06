//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

__global__ void FastllmCudaInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per) {
    #ifdef CUDA_NO_TENSOR_CORE
        float scalesBuffer[2];
        uint8_t zerosBuffer[2];
        int threshold = ST128_FP16_COUNT;
        int index = (threadIdx.x + blockIdx.x * blockDim.x) * ST128_FP16_COUNT;
        for (int idx = index; idx < len; idx += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
            int startIdx = idx / per;
            int endIdx = (idx + ST128_FP16_COUNT - 1) / per;
            scalesBuffer[1] = scalesBuffer[0] = scales[startIdx];
            zerosBuffer[1] = zerosBuffer[0] = zeros[startIdx];
            if (endIdx > startIdx) {
                threshold = (idx + ST128_FP16_COUNT - 1) % per;
                scalesBuffer[1] = scales[endIdx];
                zerosBuffer[1] = zeros[endIdx];
            }
            // 读取
            union_char8 aBuffer[2];
            half bBuffer[ST128_FP16_COUNT];
            aBuffer[0].in = *reinterpret_cast<const uint2 *>(a + idx);
            // 处理
            for (int i=0; i<ST128_FP16_COUNT; i++) {
                if (idx + i < len) {
                    int scaleIdx = i < threshold ? 0 : 1;
                    bBuffer[i] = __float2half(scalesBuffer[scaleIdx] * ((float)aBuffer[0].out[i] - zerosBuffer[scaleIdx]));
                }
            }
            reinterpret_cast<uint4 *>(b)[idx / ST128_FP16_COUNT] = *reinterpret_cast<uint4 *>(bBuffer);
        }
    #else
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < len) {
            b[idx] = __float2half(scales[idx / per] * ((float)a[idx] - zeros[idx / per]));
        }
    #endif
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt8Kernel2(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 读入fdata
    /*for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        fdata[i] = A[i];
    }
    __syncthreads();*/

    float4 regA;
    union_char4 regB;
    
    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        const uint8_t *baseB = B + p * m;
#ifdef CUDA_NO_TENSOR_CORE
#pragma unroll
        for (int i = tid*4; i < m; i += THREAD_PER_BLOCK*4) {
            regA = FETCH_FLOAT4(A[i]);
            regB.in = *reinterpret_cast<const uint32_t *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += regA.x * (float)(regB.out[0] - zero);
            if (i + 1 < m)
                sum += regA.y * (float)(regB.out[1] - zero);
            if (i + 2 < m)
                sum += regA.z * (float)(regB.out[2] - zero);
            if (i + 3 < m)
                sum += regA.w * (float)(regB.out[3] - zero);
            sdata[tid] += sum;
        }
#else
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (B[p * m + i] - zero);
        }
#endif
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

        if (tid == 0) {
            C[p] = sdata[0] * __ldg(scales + p) + __ldg(bias + p);
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int8Kernel2(half *A, uint8_t *B, half *C,
                                       half *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half8 regA;
    union_char8 regB;
    
    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        const uint8_t *baseB = B + p * m;
#pragma unroll
        for (int i = tid * ST128_FP16_COUNT; i < m; i += THREAD_PER_BLOCK * ST128_FP16_COUNT) {
            regA.in = *reinterpret_cast<const uint4 *>(A + i);
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += __low2float(regA.out2[0]) * (float)(regB.out[0] - zero);
            if (i + 1 < m)
                sum += __high2float(regA.out2[0]) * (float)(regB.out[1] - zero);
            if (i + 2 < m)
                sum += __low2float(regA.out2[1]) * (float)(regB.out[2] - zero);
            if (i + 3 < m)
                sum += __high2float(regA.out2[1]) * (float)(regB.out[3] - zero);
            if (i + 4 < m)
                sum += __low2float(regA.out2[2]) * (float)(regB.out[4] - zero);
            if (i + 5 < m)
                sum += __high2float(regA.out2[2]) * (float)(regB.out[5] - zero);
            if (i + 6 < m)
                sum += __low2float(regA.out2[3]) * (float)(regB.out[6] - zero);
            if (i + 7 < m)
                sum += __high2float(regA.out2[3]) * (float)(regB.out[7] - zero);
            sdata[tid] += sum;
        }

        __syncthreads();
        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (bias != nullptr) {
                C[p] = (half)(sdata[0] * __ldg(scales + p) + (float)__ldg(bias + p));
            } else {
                C[p] = (half)(sdata[0] * __ldg(scales + p));
            }
        }

        __syncthreads();
    }
}

void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvInt8Kernel2<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

static void FastllmCudaInt8EnsureScalesZerosAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaScales;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.zeros[i];
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void*)cudaZeropoints);

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

bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaInt8EnsureScalesZerosAndBiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
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
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
#ifdef CUDA_NO_TENSOR_CORE
        int gridSize = (len - 1) / (threadPerBlock * ST128_FP16_COUNT) + 1;
        FastllmCudaInt82HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                   cudaScales,
                                                                   cudaZeropoints,
                                                                   cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        FastllmCudaInt82HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                                         cudaScales,
                                                                                         cudaZeropoints,
                                                                                         cudaFp16Weight, len, m);

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
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int8(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvFp16Int8Kernel2 <256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

static void FastllmCudaInt8EnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        FastllmCudaInt8EnsureScalesZerosAndBiasOnDevice(weight, bias, k);
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

bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaInt8EnsureScalesZerosAndBiasOnDevice(weight, bias, k);
    FastllmCudaInt8EnsureHalfBiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaHalfData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaHalfData[1];

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

        FastllmCudaInt82HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                                            cudaScales,
                                                                                            cudaZeropoints,
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
        half *cudaBiasData = bias.dims.size() > 0 ? (half*)weight.extraCudaHalfData[2] : nullptr;
        LaunchFastllmGemmFp16Int8(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
