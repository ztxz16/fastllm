// Included by deepseekv4-kernels.cu after the compatibility FP8 decoder.
// Prefill GEMM for arbitrary groups, ranks and FP8 scale-block dimensions.
// Scratch is per invocation, with no persistent expanded weight cache.
#pragma once

template <typename OutT>
__global__ void DeepSeekV4WoAExpandFp8Kernel(
        const uint8_t *weight, const float *scales, OutT *expanded,
        uint64_t count, int cols, int blockRows, int blockCols, int scaleCols) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        const int row = i / cols, col = i % cols;
        expanded[i] = Dsv4FromFloat<OutT>(DeepSeekV4Fp8E4M3ScaledHalfToFloat(
            weight[i], scales[(uint64_t)(row / blockRows) * scaleCols + col / blockCols]));
    }
}

template <typename InT>
__global__ void DeepSeekV4WoAExpandInputKernel(
        const InT *input, float *expanded, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        expanded[i] = Dsv4ToFloat(input[i]);
    }
}

__global__ void DeepSeekV4WoARoundOutputKernel(
        const float *input, __nv_bfloat16 *output, uint64_t count) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (uint64_t)gridDim.x * blockDim.x) {
        output[i] = __float2bfloat16_rn(input[i]);
    }
}

// A power-of-two scale leaves at most four significand bits in an FP8
// value. Rounding it to FP16 (including FP16 subnormals) cannot add bits, so
// the compatibility-decoded weight is exactly representable in BF16.
// Check the actual scales on every invocation: no pointer-keyed stale cache.
bool DeepSeekV4WoAHasBf16ExactScales(const fastllm::Data &weight) {
    for (float scale : weight.scales) {
        uint32_t bits;
        std::memcpy(&bits, &scale, sizeof(bits));
        if ((bits & 0x807fffffu) != 0 || (bits & 0x7f800000u) == 0x7f800000u) {
            return false;
        }
    }
    return true;
}

bool DeepSeekV4TryWoABf16Gemm(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &output, int tokens, int groups, int rank, int cols,
        int scaleCols) {
#if CUBLAS_VERSION >= 11000
    if (FastllmCudaRuntimeArch() < 80 ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        !DeepSeekV4WoAHasBf16ExactScales(weight)) {
        return false;
    }
    fastllm::Data expandedWeight;
    if (!DeepSeekV4PrepareCudaOutput(expandedWeight, fastllm::DataType::BFLOAT16,
                                     {groups * rank, cols})) {
        return false;
    }
    const uint64_t count = weight.Count(0);
    const int blocks = (int)std::min<uint64_t>(65535, (count + 255) / 256);
    DeepSeekV4WoAExpandFp8Kernel<<<blocks, 256>>>(
        (const uint8_t*)weight.cudaData, (const float*)weight.extraCudaData[0],
        (__nv_bfloat16*)expandedWeight.cudaData, count, cols,
        weight.blockK, weight.blockM, scaleCols);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    constexpr int tileRows = 4096;
    const int fullDim = groups * cols, outDim = groups * rank;
    const float alpha = 1.0f, beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    for (int start = 0; start < tokens; start += tileRows) {
        const int rows = std::min(tileRows, tokens - start);
        const cublasStatus_t status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, rank, rows, cols,
            &alpha, expandedWeight.cudaData, CUDA_R_16BF, cols, (long long)rank * cols,
            (const __nv_bfloat16*)input.cudaData + (uint64_t)start * fullDim,
            CUDA_R_16BF, fullDim, cols,
            &beta, (__nv_bfloat16*)output.cudaData + (uint64_t)start * outDim,
            CUDA_R_16BF, outDim, rank, groups, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
                printf("DeepSeekV4WoA BF16 GEMM fallback: status=%d\n", (int)status);
            }
            return false;
        }
    }
    if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
        printf("DeepSeekV4WoA BF16 GEMM: M=%d N=%d K=%d G=%d tile=%d\n",
               tokens, rank, cols, groups, tileRows);
    }
    return true;
#else
    return false;
#endif
}

bool DeepSeekV4LaunchWoAFp8Gemm(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &output, int tokens, int groups, int rank, int cols) {
#if CUBLAS_VERSION >= 11000
    // BF16/FP16 operands and the legacy FP16-rounded weights fit exactly in
    // TF32's significand. Expanding to FP32 avoids an additional BF16 weight
    // rounding or an FP16 input range restriction. Accumulation stays FP32,
    // with one final BF16 rounding; reduction order differs from the SIMT path.
    // Unrestricted FLOAT32 and pre-SM80 devices use pedantic FP32 GEMM.
    if (tokens < 16 || FastllmCudaRuntimeArch() < 75 ||
        (input.dataType != fastllm::DataType::BFLOAT16 &&
         input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::FLOAT32) ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        weight.blockK <= 0 || weight.blockM <= 0) {
        return false;
    }
    const int weightRows = groups * rank;
    const int scaleRows = (weightRows - 1) / weight.blockK + 1;
    const int scaleCols = (cols - 1) / weight.blockM + 1;
    if (weight.scales.size() != (size_t)scaleRows * scaleCols) {
        return false;
    }
    fastllm::Data &mutableWeight = const_cast<fastllm::Data&>(weight);
    fastllm::Data emptyBias;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(mutableWeight, emptyBias, weightRows);
    if (mutableWeight.extraCudaData.empty() || mutableWeight.extraCudaData[0] == nullptr) {
        return false;
    }

    if (DeepSeekV4TryWoABf16Gemm(input, weight, output, tokens, groups, rank, cols, scaleCols)) {
        return true;
    }

    // Cap token scratch at 160 MiB (1024 rows for the 8x4096 -> 8x1024
    // projection). Larger prompts are tiled, including an arbitrary last tile.
    const int fullDim = groups * cols;
    const int outDim = groups * rank;
    const uint64_t bytesPerToken = ((uint64_t)fullDim + outDim) * sizeof(float);
    const int tileRows = std::min(tokens, (int)std::max<uint64_t>(
        1, std::min<uint64_t>(1024, (160ULL << 20) / bytesPerToken)));
    fastllm::Data expandedWeight, expandedInput, floatOutput;
    if (!DeepSeekV4PrepareCudaOutput(expandedWeight, fastllm::DataType::FLOAT32,
                                     {weightRows, cols}) ||
        !DeepSeekV4PrepareCudaOutput(expandedInput, fastllm::DataType::FLOAT32,
                                     {tileRows, fullDim}) ||
        !DeepSeekV4PrepareCudaOutput(floatOutput, fastllm::DataType::FLOAT32,
                                     {tileRows, outDim})) {
        return false;
    }
    auto blocks = [](uint64_t count) { return (int)std::min<uint64_t>(65535, (count + 255) / 256); };
    DeepSeekV4WoAExpandFp8Kernel<<<blocks(weight.Count(0)), 256>>>(
        (const uint8_t*)weight.cudaData, (const float*)mutableWeight.extraCudaData[0],
        (float*)expandedWeight.cudaData, weight.Count(0), cols,
        weight.blockK, weight.blockM, scaleCols);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    const float alpha = 1.0f, beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    for (int start = 0; start < tokens; start += tileRows) {
        const int rows = std::min(tileRows, tokens - start);
        const uint64_t inputCount = (uint64_t)rows * fullDim;
        if (input.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4WoAExpandInputKernel<<<blocks(inputCount), 256>>>(
                (const __nv_bfloat16*)input.cudaData + (uint64_t)start * fullDim,
                (float*)expandedInput.cudaData, inputCount);
        } else if (input.dataType == fastllm::DataType::FLOAT16) {
            DeepSeekV4WoAExpandInputKernel<<<blocks(inputCount), 256>>>(
                (const half*)input.cudaData + (uint64_t)start * fullDim,
                (float*)expandedInput.cudaData, inputCount);
        } else {
            DeepSeekV4WoAExpandInputKernel<<<blocks(inputCount), 256>>>(
                (const float*)input.cudaData + (uint64_t)start * fullDim,
                (float*)expandedInput.cudaData, inputCount);
        }
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
        const cublasStatus_t status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, rank, rows, cols,
            &alpha, expandedWeight.cudaData, CUDA_R_32F, cols, (long long)rank * cols,
            expandedInput.cudaData, CUDA_R_32F, fullDim, cols,
            &beta, floatOutput.cudaData, CUDA_R_32F, outDim, rank,
            groups, FastllmCudaRuntimeArch() >= 80 && input.dataType != fastllm::DataType::FLOAT32 ?
                CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
                printf("DeepSeekV4WoA FP8 GEMM fallback: status=%d M=%d N=%d K=%d G=%d\n",
                       (int)status, rows, rank, cols, groups);
            }
            return false;
        }
        const uint64_t outputCount = (uint64_t)rows * outDim;
        DeepSeekV4WoARoundOutputKernel<<<blocks(outputCount), 256>>>(
            (const float*)floatOutput.cudaData,
            (__nv_bfloat16*)output.cudaData + (uint64_t)start * outDim, outputCount);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
    }
    if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
        printf("DeepSeekV4WoA FP8 expanded GEMM: M=%d N=%d K=%d G=%d block=%dx%d tile=%d\n",
               tokens, rank, cols, groups, weight.blockK, weight.blockM, tileRows);
    }
    return true;
#else
    return false;
#endif
}
