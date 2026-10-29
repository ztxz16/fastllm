//
// Created by huangyuyang on 2/6/26.
//

// fp32的乘法计算，早期验证使用，几乎不会被用到

#include "fastllm-cuda.cuh"
#include "fastllm.h"

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp32Kernel2(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    if constexpr (PART == 1) {
        const size_t gridRow = (size_t)blockIdx.y;
        A += gridRow * m;
        C += gridRow * k;
    }

    // 1. 计算
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

// Group rows while keeping the original per-thread FMA order and ascending
// 256-lane reduction tree. Warp shuffles replace the intra-warp barriers;
// each block reads an expert/output row once for several input rows.
template<int Rows>
__global__ void FastllmGemvFp32GroupedRowsKernel(const float* __restrict__ input, const float* __restrict__ weight,
                        float* __restrict__ output, const float* __restrict__ bias,
                        int n, int m, int k) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int first = blockIdx.y * Rows, expert = blockIdx.x;
    float sums[Rows] = {};
    for (int i = tid; i < m; i += 256) {
        const float w = weight[size_t(expert) * m + i];
        #pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (first + r < n) sums[r] += input[size_t(first + r) * m + i] * w;
    }
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float other = __shfl_down_sync(0xffffffff, sums[r], offset);
            if ((lane & (2 * offset - 1)) == 0) sums[r] += other;
        }
    }
    __shared__ float partial[Rows][8];
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) partial[r][warp] = sums[r];
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            float sum = lane < 8 ? partial[r][lane] : 0.0f;
            #pragma unroll
            for (int offset = 1; offset < 8; offset *= 2) {
                const float other = __shfl_down_sync(0xffffffff, sum, offset);
                if ((lane & (2 * offset - 1)) == 0) sum += other;
            }
            if (lane == 0 && first + r < n)
                output[size_t(first + r) * k + expert] = sum + bias[expert];
        }
    }
}


// Tile both requests and output experts. Each output retains the original
// i=threadIdx.x, i+=256 FMA sequence and ascending256-lane reduction tree.
template<int Rows,int Columns>
__global__ void FastllmGemvFp32OutputTileUnpredicatedKernel(const float* __restrict__ input,const float* __restrict__ weight,
                            float* __restrict__ output,const float* __restrict__ bias,
                            int n,int m,int k) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int first=blockIdx.y*Rows,firstExpert=blockIdx.x*Columns;
    float sums[Rows][Columns]={};
    for(int i=tid;i<m;i+=256) {
        float x[Rows],w[Columns];
#pragma unroll
        for(int r=0;r<Rows;++r)x[r]=first+r<n?input[size_t(first+r)*m+i]:0.f;
#pragma unroll
        for(int c=0;c<Columns;++c)w[c]=firstExpert+c<k?weight[size_t(firstExpert+c)*m+i]:0.f;
#pragma unroll
        for(int r=0;r<Rows;++r)
#pragma unroll
            for(int c=0;c<Columns;++c)sums[r][c]+=x[r]*w[c];
    }
#pragma unroll
    for(int offset=1;offset<32;offset*=2)
#pragma unroll
        for(int r=0;r<Rows;++r)
#pragma unroll
            for(int c=0;c<Columns;++c) {
                const float other=__shfl_down_sync(0xffffffff,sums[r][c],offset);
                sums[r][c]+=other;
            }
    __shared__ float partial[Rows][Columns][8];
    if(lane==0) {
#pragma unroll
        for(int r=0;r<Rows;++r)
#pragma unroll
            for(int c=0;c<Columns;++c)partial[r][c][warp]=sums[r][c];
    }
    __syncthreads();
    if(warp==0) {
#pragma unroll
        for(int r=0;r<Rows;++r)
#pragma unroll
            for(int c=0;c<Columns;++c) {
                float sum=lane<8?partial[r][c][lane]:0.f;
#pragma unroll
                for(int offset=1;offset<8;offset*=2) {
                    const float other=__shfl_down_sync(0xffffffff,sum,offset);
                    sum+=other;
                }
                if(lane==0 && first+r<n && firstExpert+c<k)
                    output[size_t(first+r)*k+firstExpert+c]=sum+bias[firstExpert+c];
            }
    }
}


static void FastllmCudaFloat32EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
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

bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFloat32EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    const bool exactRows = n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold();
    // Small-output projections (such as MoE routers) benefit from the
    // existing GEMV reduction, with independent rows in grid.y.
    const bool smallBatch = n > 1 && n <= 8 && k <= 1024;
    if ((smallBatch || exactRows) && n <= 65535) {
        if (n >= 64 && m >= 4096 && k >= 128 && k <= 1024 &&
            FastllmCudaRuntimeArch() == 90) {
            // Reuse each input across four output columns, while preserving
            // the original per-output FP32 FMA and 256-lane reduction order.
            FastllmGemvFp32OutputTileUnpredicatedKernel<8, 4><<<dim3((k + 3) / 4, (n + 7) / 8), 256>>>(
                cudaInput, (float *)weight.cudaData, cudaOutput,
                cudaBiasData, n, m, k);
        } else if (n >= 16) {
            FastllmGemvFp32GroupedRowsKernel<4><<<dim3(k, (n + 3) / 4), 256>>>(
                cudaInput, (float *)weight.cudaData, cudaOutput,
                cudaBiasData, n, m, k);
        } else {
            FastllmGemvFp32GroupedRowsKernel<2><<<dim3(k, (n + 1) / 2), 256>>>(
                cudaInput, (float *)weight.cudaData, cudaOutput,
                cudaBiasData, n, m, k);
        }
    } else if (exactRows) {
        for (int row = 0; row < n; row++) {
            FastllmGemvFp32Fp32Kernel2<256, 1>
                <<<k, 256>>>(
                    cudaInput + (size_t)row * m,
                    (float *)weight.cudaData,
                    cudaOutput + (size_t)row * k,
                    cudaBiasData, m, k);
        }
    } else if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
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
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFloat32EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaMalloc(input.Count(0) * sizeof(float));
    float *cudaOutput = (float*)FastllmCudaMalloc(output.Count(0) * sizeof(float));
    int inputLen = input.Count(0);
    FastllmCudaHalf2FloatKernel <<< (inputLen - 1) / 256 + 1, 256 >>>((half*)input.cudaData, cudaInput, inputLen);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
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
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }
    
    int outputLen = output.Count(0);
    FastllmCudaFloat2HalfKernel <<< (outputLen - 1) / 256 + 1, 256>>>(cudaOutput, (half*)output.cudaData, outputLen);
    FastllmCudaFree(cudaInput);
    FastllmCudaFree(cudaOutput);
    DeviceSync();
    return true;
}
