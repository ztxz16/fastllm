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
    if (weight.cudaData == nullptr) {
        return;
    }
    if (weight.extraCudaData.size() < 3 || weight.extraCudaData[0] == nullptr ||
        weight.extraCudaData[1] == nullptr || weight.extraCudaData[2] == nullptr) {
        if (weight.extraCudaData.size() < 3) {
            weight.extraCudaData.resize(3, nullptr);
        }
        cudaError_t state = cudaSuccess;
        float *cudaScales;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData[0] = (void*)cudaScales;

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.zeros[i];
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData[1] = (void*)cudaZeropoints;

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData[2] = (void*)cudaBiasData;
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

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            int chunkLen = kc * m;

#ifdef CUDA_NO_TENSOR_CORE
            int gridSize = (chunkLen - 1) / (threadPerBlock * ST128_FP16_COUNT) + 1;
            FastllmCudaInt82HalfKernel <<< gridSize, threadPerBlock>>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + kOff,
                cudaZeropoints + kOff,
                cudaFp16Weight, chunkLen, m);

            status = cublasGemmEx(fastllmCublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  kc, n, m,
                                  &h_alpha, cudaFp16Weight, AType,
                                  m, cudaFp16Input, BType,
                                  m, &h_beta,
                                  cudaOutput + kOff, CType,
                                  k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
            FastllmCudaInt82HalfKernel <<< (chunkLen - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + kOff,
                cudaZeropoints + kOff,
                cudaFp16Weight, chunkLen, m);

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
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
#endif
        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
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
    if (weight.cudaData == nullptr) {
        return;
    }
    if (weight.extraCudaHalfData.size() < 3 || weight.extraCudaHalfData[0] == nullptr ||
        weight.extraCudaHalfData[1] == nullptr || weight.extraCudaHalfData[2] == nullptr) {
        FastllmCudaInt8EnsureScalesZerosAndBiasOnDevice(weight, bias, k);
        weight.extraCudaHalfData.resize(3, nullptr);
        weight.extraCudaHalfData[0] = (void*)weight.extraCudaData[0];
        weight.extraCudaHalfData[1] = (void*)weight.extraCudaData[1];

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
        weight.extraCudaHalfData[2] = (void*)cudaBiasData;
    }
}

namespace {
    static constexpr int INT8_MOE_POINTER_TABLE_IDX = 3;
    static constexpr int INT8_MOE_SCALE_TABLE_IDX = 4;
    static constexpr int INT8_MOE_ZERO_TABLE_IDX = 5;
    static constexpr int INT8_MOE_WARPS = 8;

    struct Int8MoeBatch1Table {
        const uint8_t **gateupWeights = nullptr;
        const uint8_t **downWeights = nullptr;
        const float *gateupScales = nullptr;
        const float *downScales = nullptr;
        const uint8_t *gateupZeros = nullptr;
        const uint8_t *downZeros = nullptr;
        int expertCount = 0;
        int hidden = 0;
        int inter = 0;
        int gateupRows = 0;
    };

    static bool PrepareInt8MoeBatch1Table(fastllm::Data **weights,
                                          int weightsBatch,
                                          Int8MoeBatch1Table &table) {
        if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) != 0 ||
            weights[0] != nullptr || weights[1] != nullptr ||
            weights[2] == nullptr || weights[3] == nullptr) {
            return false;
        }
        const int expertCount = weightsBatch / 2 - 1;
        fastllm::Data &firstGateup = *weights[2];
        fastllm::Data &firstDown = *weights[3];
        if (firstGateup.dataType != fastllm::DataType::INT8 ||
            firstDown.dataType != fastllm::DataType::INT8 ||
            firstGateup.dims.size() != 2 || firstDown.dims.size() != 2 ||
            (firstGateup.dims[0] & 1) != 0) {
            return false;
        }
        const int gateupRows = firstGateup.dims[0];
        const int inter = gateupRows / 2;
        const int hidden = firstGateup.dims[1];
        if (expertCount <= 0 || hidden <= 0 || inter <= 0 ||
            hidden % 8 != 0 || inter % 8 != 0 ||
            firstDown.dims[0] != hidden || firstDown.dims[1] != inter) {
            return false;
        }

        for (int expert = 0; expert < expertCount; expert++) {
            fastllm::Data *gateup = weights[(expert + 1) * 2];
            fastllm::Data *down = weights[(expert + 1) * 2 + 1];
            if (gateup == nullptr || down == nullptr ||
                gateup->dataType != fastllm::DataType::INT8 ||
                down->dataType != fastllm::DataType::INT8 ||
                gateup->dims != firstGateup.dims || down->dims != firstDown.dims ||
                gateup->cudaData == nullptr || down->cudaData == nullptr ||
                gateup->scales.size() != (size_t)gateupRows ||
                gateup->zeros.size() != (size_t)gateupRows ||
                down->scales.size() != (size_t)hidden ||
                down->zeros.size() != (size_t)hidden) {
                return false;
            }
        }

        if ((int)firstGateup.extraCudaData.size() <= INT8_MOE_ZERO_TABLE_IDX) {
            firstGateup.extraCudaData.resize(INT8_MOE_ZERO_TABLE_IDX + 1, nullptr);
        }
        void *pointerBlock = firstGateup.extraCudaData[INT8_MOE_POINTER_TABLE_IDX];
        void *scaleBlock = firstGateup.extraCudaData[INT8_MOE_SCALE_TABLE_IDX];
        void *zeroBlock = firstGateup.extraCudaData[INT8_MOE_ZERO_TABLE_IDX];
        if (pointerBlock == nullptr || scaleBlock == nullptr || zeroBlock == nullptr) {
            std::vector<uint8_t*> hostPointers((size_t)expertCount * 2);
            const size_t gateupCount = (size_t)expertCount * gateupRows;
            const size_t downCount = (size_t)expertCount * hidden;
            std::vector<float> hostScales(gateupCount + downCount);
            std::vector<uint8_t> hostZeros(gateupCount + downCount);
            for (int expert = 0; expert < expertCount; expert++) {
                const fastllm::Data &gateup = *weights[(expert + 1) * 2];
                const fastllm::Data &down = *weights[(expert + 1) * 2 + 1];
                hostPointers[expert] = (uint8_t*)gateup.cudaData;
                hostPointers[expertCount + expert] = (uint8_t*)down.cudaData;
                std::copy(gateup.scales.begin(), gateup.scales.end(),
                          hostScales.begin() + (size_t)expert * gateupRows);
                std::copy(gateup.zeros.begin(), gateup.zeros.end(),
                          hostZeros.begin() + (size_t)expert * gateupRows);
                std::copy(down.scales.begin(), down.scales.end(),
                          hostScales.begin() + gateupCount + (size_t)expert * hidden);
                std::copy(down.zeros.begin(), down.zeros.end(),
                          hostZeros.begin() + gateupCount + (size_t)expert * hidden);
            }

            pointerBlock = FastllmCudaMalloc(hostPointers.size() * sizeof(uint8_t*));
            scaleBlock = FastllmCudaMalloc(hostScales.size() * sizeof(float));
            zeroBlock = FastllmCudaMalloc(hostZeros.size() * sizeof(uint8_t));
            if (pointerBlock == nullptr || scaleBlock == nullptr || zeroBlock == nullptr) {
                if (pointerBlock != nullptr) FastllmCudaFree(pointerBlock);
                if (scaleBlock != nullptr) FastllmCudaFree(scaleBlock);
                if (zeroBlock != nullptr) FastllmCudaFree(zeroBlock);
                return false;
            }
            FastllmCudaCopyFromHostToDevice(pointerBlock, hostPointers.data(),
                                            hostPointers.size() * sizeof(uint8_t*));
            FastllmCudaCopyFromHostToDevice(scaleBlock, hostScales.data(),
                                            hostScales.size() * sizeof(float));
            FastllmCudaCopyFromHostToDevice(zeroBlock, hostZeros.data(),
                                            hostZeros.size() * sizeof(uint8_t));
            firstGateup.extraCudaData[INT8_MOE_POINTER_TABLE_IDX] = pointerBlock;
            firstGateup.extraCudaData[INT8_MOE_SCALE_TABLE_IDX] = scaleBlock;
            firstGateup.extraCudaData[INT8_MOE_ZERO_TABLE_IDX] = zeroBlock;
        }

        const size_t gateupCount = (size_t)expertCount * gateupRows;
        table.gateupWeights = (const uint8_t**)pointerBlock;
        table.downWeights = table.gateupWeights + expertCount;
        table.gateupScales = (const float*)scaleBlock;
        table.downScales = table.gateupScales + gateupCount;
        table.gateupZeros = (const uint8_t*)zeroBlock;
        table.downZeros = table.gateupZeros + gateupCount;
        table.expertCount = expertCount;
        table.hidden = hidden;
        table.inter = inter;
        table.gateupRows = gateupRows;
        return true;
    }

    template <int WARPS_PER_BLOCK>
    __global__ void FastllmCudaInt8MoeGateupSwigluBatch1Kernel(
            const half *input,
            const uint8_t *const *gateupWeights,
            const float *scales,
            const uint8_t *zeros,
            const int32_t *indices,
            half *middle,
            int topk, int expertCount, int hidden, int inter, int gateupRows) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int task = blockIdx.x * WARPS_PER_BLOCK + warp;
        if (task >= topk * inter) {
            return;
        }
        const int route = task / inter;
        const int out = task - route * inter;
        const int expert = __ldg(indices + route);
        float gateAcc = 0.0f, upAcc = 0.0f;
        float gateScale = 0.0f, upScale = 0.0f;
        if (expert >= 0 && expert < expertCount) {
            const uint8_t *weight = gateupWeights[expert];
            if (weight != nullptr) {
                const int gateRow = out;
                const int upRow = inter + out;
                const uint8_t gateZero = __ldg(zeros + (size_t)expert * gateupRows + gateRow);
                const uint8_t upZero = __ldg(zeros + (size_t)expert * gateupRows + upRow);
                gateScale = __ldg(scales + (size_t)expert * gateupRows + gateRow);
                upScale = __ldg(scales + (size_t)expert * gateupRows + upRow);
                const uint8_t *gateWeight = weight + (size_t)gateRow * hidden;
                const uint8_t *upWeight = weight + (size_t)upRow * hidden;
                const int units = hidden >> 3;
                for (int unit = lane; unit < units; unit += 32) {
                    const int x = unit << 3;
                    union_half8 inputValues;
                    union_char8 gateValues, upValues;
                    inputValues.in = *reinterpret_cast<const uint4*>(input + x);
                    gateValues.in = *reinterpret_cast<const uint2*>(gateWeight + x);
                    upValues.in = *reinterpret_cast<const uint2*>(upWeight + x);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float a = __half2float(inputValues.out[i]);
                        gateAcc += a * ((float)gateValues.out[i] - (float)gateZero);
                        upAcc += a * ((float)upValues.out[i] - (float)upZero);
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
            const float gate = gateAcc * gateScale;
            const float up = upAcc * upScale;
            middle[(size_t)route * inter + out] =
                __float2half_rn((gate / (1.0f + expf(-gate))) * up);
        }
    }

    template <int WARPS_PER_BLOCK>
    __global__ void FastllmCudaInt8MoeDownReduceBatch1Kernel(
            const half *middle,
            const uint8_t *const *downWeights,
            const float *scales,
            const uint8_t *zeros,
            const int32_t *indices,
            const float *scores,
            half *output,
            int topk, int expertCount, int hidden, int inter) {
        __shared__ float routeValues[WARPS_PER_BLOCK];
        const int route = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int out = blockIdx.x;
        float acc = 0.0f, scale = 0.0f;
        if (route < topk) {
            const int expert = __ldg(indices + route);
            if (expert >= 0 && expert < expertCount) {
                const uint8_t *weight = downWeights[expert];
                if (weight != nullptr) {
                    const uint8_t zero = __ldg(zeros + (size_t)expert * hidden + out);
                    scale = __ldg(scales + (size_t)expert * hidden + out);
                    const uint8_t *rowWeight = weight + (size_t)out * inter;
                    const half *routeInput = middle + (size_t)route * inter;
                    const int units = inter >> 3;
                    for (int unit = lane; unit < units; unit += 32) {
                        const int x = unit << 3;
                        union_half8 inputValues;
                        union_char8 weightValues;
                        inputValues.in = *reinterpret_cast<const uint4*>(routeInput + x);
                        weightValues.in = *reinterpret_cast<const uint2*>(rowWeight + x);
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            acc += __half2float(inputValues.out[i]) *
                                   ((float)weightValues.out[i] - (float)zero);
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
            routeValues[route] = route < topk ? acc * scale * __ldg(scores + route) : 0.0f;
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

bool FastllmCudaHalfMergeMOEInt8Batch1Indexed(const fastllm::Data &input,
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
        indices == nullptr || scores == nullptr || topk <= 0 || topk > INT8_MOE_WARPS) {
        return false;
    }
    Int8MoeBatch1Table table;
    if (!PrepareInt8MoeBatch1Table(weights, weightsBatch, table) ||
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
    FastllmCudaInt8MoeGateupSwigluBatch1Kernel<INT8_MOE_WARPS>
        <<< (gateupTasks + INT8_MOE_WARPS - 1) / INT8_MOE_WARPS,
             INT8_MOE_WARPS * 32, 0, cudaStreamPerThread >>>(
            (const half*)input.cudaData, table.gateupWeights,
            table.gateupScales, table.gateupZeros, indices,
            (half*)scratch.cudaData, topk, table.expertCount,
            table.hidden, table.inter, table.gateupRows);
    FastllmCudaInt8MoeDownReduceBatch1Kernel<INT8_MOE_WARPS>
        <<< table.hidden, INT8_MOE_WARPS * 32, 0, cudaStreamPerThread >>>(
            (const half*)scratch.cudaData, table.downWeights,
            table.downScales, table.downZeros, indices, scores,
            (half*)output.cudaData, topk, table.expertCount,
            table.hidden, table.inter);
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA INT8 batch-1 fused MoE failed.", state);
        return false;
    }
    return true;
#endif
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

        int threadPerBlock = std::min(256, n * m);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            int chunkLen = kc * m;

            FastllmCudaInt82HalfKernel <<< (chunkLen - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + kOff,
                cudaZeropoints + kOff,
                cudaFp16Weight, chunkLen, m);

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
            half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        half *cudaBiasData = bias.dims.size() > 0 ? (half*)weight.extraCudaHalfData[2] : nullptr;
        LaunchFastllmGemmFp16Int8(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
