/*
 * Weight-only FP8 Marlin (W8A16) for FastLLM dense Linear on SM75+.
 *
 * Matches vLLM MarlinFP8ScaledMMLinearKernel:
 *   apply_fp8_marlin_linear -> ops.marlin_gemm(b_q_type=float8_e4m3fn)
 *
 * Launch uses dense Marlin (fastllm-marlin-fp8-dense.cu).
 */

#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

constexpr int FP8_MARLIN_WEIGHT_IDX = 2;
constexpr int FP8_MARLIN_WORKSPACE_IDX = 3;
constexpr int FP8_MARLIN_SCALES_HALF_IDX = 1;
constexpr int FP8_GROUP_SIZE = 128;
constexpr int FP8_MARLIN_CONVERT_MAX_M = 8;

extern "C" bool FastllmCudaMarlinHalfFP8Gemm(
        const void *a, const uint32_t *b_q_weight, const void *b_scales,
        void *c, int size_m, int size_n, int size_k, int group_size,
        int *workspace);

// Runtime gate only: SM75+ uses dense FP8 Marlin; SM70/60 still compile
// (device stubs under __CUDA_ARCH__ < 750) and fall back to GEMV here.
static bool Fp8MarlinDeviceSupported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return false;
    }
    return major * 10 + minor >= 75;
#endif
}

static bool HasFp8MarlinOnDevice(const fastllm::Data &weight) {
    return (int)weight.extraCudaData.size() > FP8_MARLIN_WORKSPACE_IDX &&
           weight.extraCudaData[FP8_MARLIN_WEIGHT_IDX] != nullptr &&
           weight.extraCudaData[FP8_MARLIN_WORKSPACE_IDX] != nullptr &&
           (int)weight.extraCudaHalfData.size() > FP8_MARLIN_SCALES_HALF_IDX &&
           weight.extraCudaHalfData[FP8_MARLIN_SCALES_HALF_IDX] != nullptr;
}

__global__ void FastllmFp8PackToGptqKernel(const uint8_t *__restrict__ weight,
                                           uint32_t *__restrict__ qweight,
                                           int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int packs = k / 4;
    int total = packs * n;
    if (idx >= total) return;
    int pack = idx / n;
    int out = idx - pack * n;
    const uint8_t *row = weight + (size_t)out * k + (size_t)pack * 4;
    qweight[(size_t)pack * n + out] = *reinterpret_cast<const uint32_t *>(row);
}

static void BuildFp8MarlinPermutedScales(const fastllm::Data &weight, int m, int k,
                                         std::vector<half> &outScales) {
    const int blockM = weight.blockM;
    const int blockK = weight.blockK;
    const int ms = (m + blockM - 1) / blockM;
    const int numGroups = m / FP8_GROUP_SIZE;
    const float expBias = exp2f(8.0f);

    std::vector<float> groupN((size_t)numGroups * k);
    for (int g = 0; g < numGroups; g++) {
        for (int out = 0; out < k; out++) {
            int nBlock = out / blockK;
            float s = weight.scales[(size_t)nBlock * ms + g];
            groupN[(size_t)g * k + out] = s * expBias;
        }
    }

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
    outScales.resize(groupN.size());
    for (size_t base = 0; base < groupN.size(); base += 64) {
        for (int i = 0; i < 64; i++) {
            outScales[base + i] = (half)groupN[base + scalePerm[i]];
        }
    }
}

static bool EnsureFp8MarlinOnDevice(fastllm::Data &weight, int m, int k) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (HasFp8MarlinOnDevice(weight)) return true;
    if (weight.cudaData == nullptr ||
        weight.blockM != FP8_GROUP_SIZE || weight.blockK != FP8_GROUP_SIZE ||
        m % FP8_GROUP_SIZE != 0 || k % 64 != 0 || m % 64 != 0 ||
        weight.scales.empty()) {
        return false;
    }

    FastllmCudaClearThreadError();
    size_t qweightCount = (size_t)(m / 4) * k;
    size_t qweightBytes = qweightCount * sizeof(uint32_t);
    uint32_t *stdQWeight = (uint32_t *)FastllmCudaMalloc(qweightBytes);
    uint32_t *marlinQWeight = (uint32_t *)FastllmCudaMalloc(qweightBytes);
    if (stdQWeight == nullptr || marlinQWeight == nullptr || FastllmCudaGetThreadError()) {
        if (stdQWeight) FastllmCudaFree(stdQWeight);
        if (marlinQWeight) FastllmCudaFree(marlinQWeight);
        FastllmCudaClearThreadError();
        return false;
    }

    int threads = 256;
    int blocks = (int)((qweightCount + threads - 1) / threads);
    FastllmFp8PackToGptqKernel<<<blocks, threads>>>(
        (const uint8_t *)weight.cudaData, stdQWeight, k, m);
    if (cudaPeekAtLastError() != cudaSuccess ||
        !FastllmCudaGptqMarlinRepackBits(stdQWeight, marlinQWeight, m, k, 8)) {
        FastllmCudaFree(stdQWeight);
        FastllmCudaFree(marlinQWeight);
        return false;
    }
    FastllmCudaFree(stdQWeight);

    std::vector<half> hostScales;
    BuildFp8MarlinPermutedScales(weight, m, k, hostScales);
    half *marlinScales = (half *)FastllmCudaMalloc(hostScales.size() * sizeof(half));
    if (marlinScales == nullptr || FastllmCudaGetThreadError()) {
        FastllmCudaFree(marlinQWeight);
        FastllmCudaClearThreadError();
        return false;
    }
    FastllmCudaCopyFromHostToDevice(marlinScales, hostScales.data(),
                                    hostScales.size() * sizeof(half));

    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int workspaceInts = std::max(1, sms * 4);
    int *workspace = (int *)FastllmCudaMalloc((size_t)workspaceInts * sizeof(int));
    if (workspace == nullptr || FastllmCudaGetThreadError()) {
        FastllmCudaFree(marlinQWeight);
        FastllmCudaFree(marlinScales);
        FastllmCudaClearThreadError();
        return false;
    }
    FastllmCudaMemset0(workspace, (size_t)workspaceInts * sizeof(int));

    if ((int)weight.extraCudaData.size() <= FP8_MARLIN_WORKSPACE_IDX) {
        weight.extraCudaData.resize(FP8_MARLIN_WORKSPACE_IDX + 1, nullptr);
    }
    weight.extraCudaData[FP8_MARLIN_WEIGHT_IDX] = (void *)marlinQWeight;
    weight.extraCudaData[FP8_MARLIN_WORKSPACE_IDX] = (void *)workspace;
    if ((int)weight.extraCudaHalfData.size() <= FP8_MARLIN_SCALES_HALF_IDX) {
        weight.extraCudaHalfData.resize(FP8_MARLIN_SCALES_HALF_IDX + 1, nullptr);
    }
    weight.extraCudaHalfData[FP8_MARLIN_SCALES_HALF_IDX] = (void *)marlinScales;

    if (weight.cudaData != nullptr) {
        cudaDeviceSynchronize();
        FastllmCudaForceFree(weight.cudaData);
        weight.cudaData = nullptr;
    }
    FastllmCudaClearThreadError();
    return true;
}

}  // namespace

extern "C" bool FastllmCudaTryMarlinHalfMatMulFloatFP8E4M3(
        const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k) {
    if (!Fp8MarlinDeviceSupported() || n < 1 ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != FP8_GROUP_SIZE || weight.blockK != FP8_GROUP_SIZE ||
        m % 64 != 0 || k % 64 != 0 || m % FP8_GROUP_SIZE != 0) {
        return false;
    }

    if (!HasFp8MarlinOnDevice(weight)) {
        // Convert during warmup (NCCL force-sync) on small M only.
        // Microbench / pre-NCCL also has force-sync default true.
        if (n > FP8_MARLIN_CONVERT_MAX_M || !FastllmCudaGetNcclForceSync()) {
            return false;
        }
        if (!EnsureFp8MarlinOnDevice(weight, m, k)) {
            return false;
        }
    }

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
    auto *marlinWeight = (const uint32_t *)weight.extraCudaData[FP8_MARLIN_WEIGHT_IDX];
    auto *workspace = (int *)weight.extraCudaData[FP8_MARLIN_WORKSPACE_IDX];
    auto *marlinScales = (const half *)weight.extraCudaHalfData[FP8_MARLIN_SCALES_HALF_IDX];

    bool ok = FastllmCudaMarlinHalfFP8Gemm(
        cudaInput, marlinWeight, marlinScales, cudaOutput,
        n, k, m, FP8_GROUP_SIZE, workspace);

    if (ok && bias.dims.size() > 0 &&
        !weight.extraCudaHalfData.empty() &&
        weight.extraCudaHalfData[0] != nullptr) {
        FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, (half *)weight.extraCudaHalfData[0], k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return ok;
}
