/*
 * Weight-only FP8 Marlin (W8A16) for FastLLM dense Linear on SM75+.
 *
 * Matches vLLM MarlinFP8ScaledMMLinearKernel:
 *   apply_fp8_marlin_linear -> ops.marlin_gemm(b_q_type=float8_e4m3fn)
 *
 * Launch uses dense Marlin (fastllm-marlin-fp8-dense.cu).
 */

#include "fastllm-cuda.cuh"
#include "fastllm-cublas-prefill.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr int FP8_MARLIN_WORKSPACE_IDX = 3;
constexpr int FP8_MARLIN_SCALES_HALF_IDX = 1;
constexpr int FP8_GROUP_SIZE = 128;
constexpr int FP8_MARLIN_CONVERT_MAX_M = 8;

static bool Fp8MarlinEnvFlagDefaultEnabled(const char *name, bool defaultValue) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

// Runtime gate only: SM75+ uses dense FP8 Marlin; SM70/60 still compile
// (device stubs under __CUDA_ARCH__ < 750) and fall back to GEMV here. SM120
// and SM121 use CUTLASS FP8 prefill kernels that require the original row-major
// weights, so do not destructively repack them during warmup by default.
static bool Fp8MarlinDeviceSupported(int &runtimeArch) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return false;
    }
    int arch = major * 10 + minor;
    runtimeArch = arch;
    bool defaultEnabled = true;
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120)
    defaultEnabled = defaultEnabled && arch != 120;
#endif
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
    defaultEnabled = defaultEnabled && arch != 121;
#endif
    return arch >= 75 &&
           Fp8MarlinEnvFlagDefaultEnabled("FASTLLM_CUDA_FP8_MARLIN", defaultEnabled);
#endif
}

static bool Fp8MarlinShouldPreserveRowMajorForTriton(int arch) {
#if defined(_WIN32) || defined(USE_ROCM)
    (void)arch;
    return false;
#else
    if (arch != 89) {
        return false;
    }
    const char *globalEnv = std::getenv("FASTLLM_CUDA_TRITON");
    if (globalEnv != nullptr && globalEnv[0] != '\0') {
        return Fp8MarlinEnvFlagDefaultEnabled(
                   "FASTLLM_CUDA_TRITON", false) &&
               Fp8MarlinEnvFlagDefaultEnabled(
                   "FASTLLM_CUDA_TRITON_LINEAR_FP8", true);
    }
    return Fp8MarlinEnvFlagDefaultEnabled(
        "FASTLLM_CUDA_TRITON_LINEAR_FP8", false);
#endif
}

constexpr int FP8_PREFILL_MIN_ROWS = 1024;
constexpr int FP8_PREFILL_SMALL_WORKSPACE_MIN_ROWS = 1536;

static bool Fp8PrefillMatrixSupported(int sizeN, int sizeK) {
    // Small matrices remain launch-bound and can still favour Marlin at large M.
    return sizeN >= 1024 && sizeK >= 1024;
}

static bool Fp8PrefillEnabled() {
    static const bool enabled = Fp8MarlinEnvFlagDefaultEnabled(
        "FASTLLM_CUDA_FP8_PREFILL_CUBLAS", true);
    return enabled;
}

using fastllm_cuda_prefill::State;
using fastllm_cuda_prefill::CheckCuda;
using fastllm_cuda_prefill::CheckCublas;

#include "fastllm-fp8-prefill-dequant.cuh"

static bool Fp8PrefillCublas(State *state,
        const half *input, const uint32_t *weight, const half *scales,
        half *output, int rows, int sizeN, int sizeK) {
    if (state == nullptr || rows < FP8_PREFILL_MIN_ROWS) return false;
    cudaStreamCaptureStatus capture;
    CheckCuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    const int chunkN = int(std::min(size_t(sizeN),
        state->capacity / (size_t(sizeK) * sizeof(half))) / 64 * 64);
    if (chunkN < 1024) return false;
    // Full-matrix and >=64 MiB sliced scans cross Marlin around M=960; 1024
    // retains a margin across SM75 shapes. 32 MiB slices have regressions at
    // M=1024, so use 1536 there. Smaller unmeasured slices stay on Marlin.
    if (chunkN < sizeN && state->capacity < 64ULL * 1024 * 1024 &&
        (state->capacity < 32ULL * 1024 * 1024 ||
         rows < FP8_PREFILL_SMALL_WORKSPACE_MIN_ROWS)) {
        return false;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    // Order quantized prefill submissions to the dedicated cuBLAS handle.
    // Shared attention scratch also relies on the existing worker stream order.
    CheckCuda(cudaStreamWaitEvent(cudaStreamPerThread, state->completed, 0));
    fastllm_cuda_prefill::SetPointerMode(state, CUBLAS_POINTER_MODE_HOST);
    const half alpha = __float2half_rn(1), beta = __float2half_rn(0);
    for (int offset = 0; offset < sizeN; offset += chunkN) {
        const int count = std::min(chunkN, sizeN - offset);
        Fp8PrefillDequantKernel<<<dim3(count / 64, (sizeK + 255) / 256), 256, 0, cudaStreamPerThread>>>(
            weight, scales, state->scratch, sizeN, sizeK, offset);
        CheckCuda(cudaPeekAtLastError());
        // FP16 accumulation is intentional for the measured SM75 speedup.
        // Dequantized weights match Marlin, but GEMM results are not bitwise equal.
        CheckCublas(cublasGemmEx(state->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            count, rows, sizeK, &alpha, state->scratch, CUDA_R_16F, sizeK,
            input, CUDA_R_16F, sizeK, &beta, output + offset, CUDA_R_16F, sizeN,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CheckCuda(cudaEventRecord(state->completed, cudaStreamPerThread));
    if (!state->fp8Logged) {
        int device = 0;
        CheckCuda(cudaGetDevice(&device));
        std::fprintf(stderr,
            "[FP8 prefill cuBLAS] GPU %d: active, rows=%d N=%d K=%d; capacity=%zu.\n",
            device, rows, sizeN, sizeK, state->capacity);
        state->fp8Logged = true;
    }
    return true;
}

static bool HasFp8MarlinOnDevice(const fastllm::Data &weight) {
    return (int)weight.extraCudaData.size() > FP8_MARLIN_WORKSPACE_IDX &&
           weight.cudaData != nullptr &&
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

__device__ __forceinline__ void FastllmFp8MarlinDequantX4(
        uint32_t q, half2 &h01, half2 &h23) {
    uint32_t q01 = ((q & 0x000000FFu) << 8) | ((q & 0x0000FF00u) << 16);
    uint32_t q23 = ((q & 0x00FF0000u) >> 8) | (q & 0xFF000000u);
    uint32_t o01 = (q01 & 0x80008000u) | ((q01 & 0x7F007F00u) >> 1);
    uint32_t o23 = (q23 & 0x80008000u) | ((q23 & 0x7F007F00u) >> 1);
    h01 = *reinterpret_cast<const half2 *>(&o01);
    h23 = *reinterpret_cast<const half2 *>(&o23);
}

// Batch-one GEMV over the in-place Marlin weight layout. A block owns the eight
// outputs encoded by one storage warp; its compute warps split the K-scale
// groups and consume packed words in storage order. This keeps weight reads
// fully coalesced and exposes enough warps for narrow output matrices without
// using Marlin GEMM or retaining a second copy of the original FP8 weights.
__global__ void FastllmFp8MarlinLayoutGemvKernel(
        const half *__restrict__ input,
        const uint32_t *__restrict__ weight,
        const float *__restrict__ scales,
        const half *__restrict__ bias,
        half *__restrict__ output,
        int sizeN, int sizeK) {
    constexpr int kWarpsPerBlock = 8;
    __shared__ float partial[kWarpsPerBlock][8];

    const int computeWarp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int storageWarp = blockIdx.x & 7;
    const int nTile = blockIdx.x >> 3;
    const int word = storageWarp * 32 + lane;
    const int rowPair = lane >> 3;
    const int slot = lane & 7;
    const int outLocal =
        (slot >> 1) * 16 + storageWarp + (slot & 1) * 8;
    const int out = nTile * 64 + outLocal;
    const int nTiles = sizeN >> 6;
    const int scaleCols = sizeK >> 7;

    float acc = 0.0f;
    for (int group = computeWarp; group < scaleCols;
         group += kWarpsPerBlock) {
        float groupAcc = 0.0f;
        const size_t tileStride = (size_t)nTiles * 256;
        size_t offset =
            ((size_t)(group * 8) * nTiles + nTile) * 256 + word;
#pragma unroll
        for (int tile = 0; tile < 8; tile++) {
            half2 w01, w23;
            FastllmFp8MarlinDequantX4(weight[offset], w01, w23);
            offset += tileStride;

            const int in = group * 128 + tile * 16 + rowPair * 2;
            const half2 a01 = __halves2half2(input[in], input[in + 8]);
            const half2 a23 = __halves2half2(input[in + 1], input[in + 9]);
            const half2 sum =
                __hadd2(__hmul2(a01, w01), __hmul2(a23, w23));
            groupAcc += __half2float(__low2half(sum)) +
                        __half2float(__high2half(sum));
        }
        acc += groupAcc *
               (scales[(size_t)(nTile >> 1) * scaleCols + group] * 256.0f);
    }

    // Four lanes, spaced by eight, cover the 16 K values for one output in
    // each tile. Reduce those lanes while keeping the eight outputs per warp.
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    if (lane < 8) {
        partial[computeWarp][lane] = acc;
    }
    __syncthreads();

    if (computeWarp == 0 && lane < 8) {
        float value = partial[0][lane];
#pragma unroll
        for (int warp = 1; warp < kWarpsPerBlock; warp++) {
            value += partial[warp][lane];
        }
        if (bias != nullptr) value += __half2float(bias[out]);
        output[out] = __float2half_rn(value);
    }
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
    std::vector<half> hostScales;
    BuildFp8MarlinPermutedScales(weight, m, k, hostScales);
    half *marlinScales = (half *)FastllmCudaMalloc(hostScales.size() * sizeof(half));
    if (marlinScales == nullptr || FastllmCudaGetThreadError()) {
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
        FastllmCudaFree(marlinScales);
        FastllmCudaClearThreadError();
        return false;
    }
    FastllmCudaMemset0(workspace, (size_t)workspaceInts * sizeof(int));

    // The Marlin FP8 layout has exactly the same byte size as the source FP8
    // matrix. Model weights normally come from a large CUDA weight slab, so
    // releasing one source pointer after allocating a second persistent copy
    // cannot return the slab while its neighbouring weights are still alive.
    // Repack back into the source allocation instead and keep a single weight
    // copy resident. The temporary GPTQ layout is only needed during warmup.
    size_t qweightCount = (size_t)(m / 4) * k;
    size_t qweightBytes = qweightCount * sizeof(uint32_t);
    uint32_t *stdQWeight = (uint32_t *)FastllmCudaMalloc(qweightBytes);
    if (stdQWeight == nullptr || FastllmCudaGetThreadError()) {
        if (stdQWeight) FastllmCudaFree(stdQWeight);
        FastllmCudaFree(workspace);
        FastllmCudaFree(marlinScales);
        FastllmCudaClearThreadError();
        return false;
    }

    int threads = 256;
    int blocks = (int)((qweightCount + threads - 1) / threads);
    FastllmFp8PackToGptqKernel<<<blocks, threads>>>(
        (const uint8_t *)weight.cudaData, stdQWeight, k, m);
    bool repacked = cudaPeekAtLastError() == cudaSuccess &&
                    FastllmCudaGptqMarlinRepackBits(
                        stdQWeight, (uint32_t *)weight.cudaData, m, k, 8);
    cudaError_t syncState = cudaStreamSynchronize(cudaStreamPerThread);
    // This buffer is conversion-only. Returning it to the reusable pool would
    // intentionally retain up to 300 MB of idle big buffers per device.
    FastllmCudaForceFree(stdQWeight);
    if (!repacked || syncState != cudaSuccess) {
        FastllmCudaFree(workspace);
        FastllmCudaFree(marlinScales);
        if (syncState != cudaSuccess) {
            printf("Error: FP8 Marlin in-place repack failed: %s.\n",
                   cudaGetErrorString(syncState));
            throw("fp8 marlin repack error");
        }
        return false;
    }

    if ((int)weight.extraCudaData.size() <= FP8_MARLIN_WORKSPACE_IDX) {
        weight.extraCudaData.resize(FP8_MARLIN_WORKSPACE_IDX + 1, nullptr);
    }
    weight.extraCudaData[FP8_MARLIN_WORKSPACE_IDX] = (void *)workspace;
    if ((int)weight.extraCudaHalfData.size() <= FP8_MARLIN_SCALES_HALF_IDX) {
        weight.extraCudaHalfData.resize(FP8_MARLIN_SCALES_HALF_IDX + 1, nullptr);
    }
    weight.extraCudaHalfData[FP8_MARLIN_SCALES_HALF_IDX] = (void *)marlinScales;

    FastllmCudaClearThreadError();
    return true;
}

}  // namespace

extern "C" bool FastllmCudaHasFp8MarlinLayout(
        const fastllm::Data &weight) {
    return HasFp8MarlinOnDevice(weight);
}

extern "C" bool FastllmCudaTryMarlinHalfMatMulFloatFP8E4M3(
        const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k) {
    int arch = 0;
    if (!Fp8MarlinDeviceSupported(arch) || n < 1 ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != FP8_GROUP_SIZE || weight.blockK != FP8_GROUP_SIZE ||
        m % 64 != 0 || k % 64 != 0 || m % FP8_GROUP_SIZE != 0) {
        return false;
    }

#if defined(FASTLLM_ENABLE_DEEPGEMM_FP8_SM90)
    // DeepGEMM prefill and native GEMV share row-major weights on SM90.
    // Preserve fresh eligible weights; already packed weights still use Marlin.
    if (arch == 90 && k % 128 == 0 &&
        weight.scales.size() == (size_t)(m / 128) * (k / 128) &&
        !HasFp8MarlinOnDevice(weight)) {
        return false;
    }
#endif

    // Triton SM89 FP8 consumes row-major weights, while Marlin repacks cudaData
    // in place during warmup. Preserve fresh eligible weights so Triton remains
    // available; already packed weights continue using Marlin.
    if (bias.dims.empty() && !weight.scales.empty() &&
        !HasFp8MarlinOnDevice(weight) &&
        Fp8MarlinShouldPreserveRowMajorForTriton(arch)) {
        return false;
    }

    // Warm the dedicated handle before the KV budget is calibrated, even
    // though the small warmup GEMM itself still goes through Marlin/GEMV.
    // Decode avoids the state lookup and CUDA event operations entirely.
    auto *prefillState = arch == 75 && Fp8PrefillMatrixSupported(k, m) &&
        (n >= FP8_PREFILL_MIN_ROWS || FastllmCudaGetNcclForceSync()) &&
        Fp8PrefillEnabled() ? fastllm_cuda_prefill::GetState(arch) : nullptr;

    // Keep batch-one decode on GEMV. Before conversion the caller can use the
    // original-layout GEMV directly; during synchronized warmup convert first
    // and use the coalesced GEMV above, avoiding a second CUDA module path.
    if (n == 1) {
        if (!HasFp8MarlinOnDevice(weight)) {
            if (!FastllmCudaGetNcclForceSync() ||
                !EnsureFp8MarlinOnDevice(weight, m, k)) {
                return false;
            }
        }

        half *cudaInput = (half *)FastllmCudaPrepareInput(input);
        half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
        auto *marlinWeight = (const uint32_t *)weight.cudaData;
        auto *cudaScales = (const float *)weight.extraCudaData[0];
        auto *cudaBias = bias.dims.empty()
            ? nullptr : (const half *)weight.extraCudaHalfData[0];
        FastllmFp8MarlinLayoutGemvKernel<<<k / 8, 256, 0, cudaStreamPerThread>>>(
            cudaInput, marlinWeight, cudaScales, cudaBias, cudaOutput, k, m);
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("Error: FP8 Marlin-layout batch-one GEMV launch failed.\n");
            throw("fp8 marlin-layout gemv error");
        }
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return true;
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
    auto *marlinWeight = (const uint32_t *)weight.cudaData;
    auto *workspace = (int *)weight.extraCudaData[FP8_MARLIN_WORKSPACE_IDX];
    auto *marlinScales = (const half *)weight.extraCudaHalfData[FP8_MARLIN_SCALES_HALF_IDX];

    bool ok = Fp8PrefillCublas(prefillState, cudaInput, marlinWeight,
                              marlinScales, cudaOutput, n, k, m);
    if (!ok) {
        ok = FastllmCudaMarlinHalfFP8Gemm(
            cudaInput, marlinWeight, marlinScales, cudaOutput,
            n, k, m, FP8_GROUP_SIZE, workspace);
    }

    if (!ok) {
        printf("Error: FP8 Marlin GEMM failed after the CUDA weight was repacked in place.\n");
        throw("fp8 marlin gemm error");
    }

    if (ok && bias.dims.size() > 0 &&
        !weight.extraCudaHalfData.empty() &&
        weight.extraCudaHalfData[0] != nullptr) {
        FastllmCudaBiasKernel<<<n, 256>>>(cudaOutput, (half *)weight.extraCudaHalfData[0], k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return ok;
}
