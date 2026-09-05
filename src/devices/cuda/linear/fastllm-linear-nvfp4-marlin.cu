/*
 * Weight-only NVFP4 Marlin (W4A16, group size 16) for FastLLM dense Linear.
 *
 * The source NVFP4_BLOCK_16 layout is row-major and interleaves every sixteen
 * packed FP4 values with one effective float scale.  During synchronized
 * warmup this file converts it in place to vLLM's Marlin weight and special
 * S0E5M3 scale layouts.  Unsupported devices/shapes keep the original layout
 * and fall back to FastLLM's native NVFP4 kernels.
 */

#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

constexpr int NVFP4_GROUP_SIZE = 16;
constexpr int NVFP4_MARLIN_CONVERT_MAX_M = 8;
constexpr int NVFP4_MARLIN_OUTPUT_ALIGNMENT = 64;

enum class Nvfp4MarlinMode {
    AUTO,
    DISABLED,
    ENABLED,
};

static bool Nvfp4MarlinEnvFlag(const char *name, bool defaultValue) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return defaultValue;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

static Nvfp4MarlinMode Nvfp4MarlinModeFromEnv() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_MARLIN");
    if (value == nullptr || value[0] == '\0' ||
        std::strcmp(value, "auto") == 0 ||
        std::strcmp(value, "AUTO") == 0) {
        return Nvfp4MarlinMode::AUTO;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
        return Nvfp4MarlinMode::DISABLED;
    }
    if (std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0) {
        return Nvfp4MarlinMode::ENABLED;
    }
    // Unknown values are treated as auto instead of unexpectedly forcing a
    // destructive in-place conversion.
    return Nvfp4MarlinMode::AUTO;
}

static bool Nvfp4MarlinArchitectureSupported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    static thread_local int cachedDevice = -1;
    static thread_local bool cachedSupported = false;
    if (cachedDevice == device) {
        return cachedSupported;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) {
        return false;
    }
    cachedDevice = device;
    cachedSupported = major * 10 + minor >= 75;
    return cachedSupported;
#endif
}

static const char *Nvfp4MarlinModeName(Nvfp4MarlinMode mode) {
    if (mode == Nvfp4MarlinMode::DISABLED) return "disabled";
    if (mode == Nvfp4MarlinMode::ENABLED) return "enabled";
    return "auto";
}

static bool Nvfp4MarlinDeviceSupported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    Nvfp4MarlinMode mode = Nvfp4MarlinModeFromEnv();
    if (mode == Nvfp4MarlinMode::DISABLED) return false;
    // This is the same architecture gate used by vLLM's FP4 Marlin path.
    // SM75 uses Turing MMA with a two-stage pipeline; SM80+ uses four stages.
    return Nvfp4MarlinArchitectureSupported();
#endif
}

static bool HasNvfp4MarlinOnDevice(const fastllm::Data &weight) {
    // IsRepacked is shared with the SM70 TurboMind representation. Include
    // the current architecture in the discriminator so the SM70 layout is
    // not interpreted as Marlin (or vice versa) during generic dispatch.
    return weight.cudaData != nullptr &&
           weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
           weight.blockM == NVFP4_GROUP_SIZE && weight.blockK == 1 &&
           weight.IsRepacked && Nvfp4MarlinArchitectureSupported();
}

static bool Nvfp4MarlinShapeSupported(int sizeN, int sizeK) {
    if (sizeN <= 0 || sizeK <= 0 || sizeN % 64 != 0 || sizeK % 64 != 0) {
        return false;
    }
    // Every Marlin thread-block shape is either K64xN128 or K128xN64
    // (K128xN128 is covered by both conditions).
    return (sizeK % 64 == 0 && sizeN % 128 == 0) ||
           (sizeK % 128 == 0 && sizeN % 64 == 0);
}

static bool GetNvfp4MarlinPackedOutputDim(int logicalN, int &packedN) {
    if (logicalN <= 0 ||
        logicalN > INT_MAX - (NVFP4_MARLIN_OUTPUT_ALIGNMENT - 1)) {
        return false;
    }
    packedN = ((logicalN + NVFP4_MARLIN_OUTPUT_ALIGNMENT - 1) /
               NVFP4_MARLIN_OUTPUT_ALIGNMENT) * NVFP4_MARLIN_OUTPUT_ALIGNMENT;
    return true;
}

// The source format occupies twelve bytes per group (eight packed weight
// bytes plus one float scale), while the Marlin layout needs only nine bytes
// (eight packed weight bytes plus one byte scale).  Keep Marlin's tiny lock
// workspace and tensor scale in that reclaimed tail.  Besides avoiding
// redundant allocations, this matters for hybrid-attention models where
// batch-sized recurrent state already leaves very little free VRAM.
static bool GetNvfp4MarlinTailPointers(
        fastllm::Data &weight, int sizeK, int sizeN, int sms, int sizeM,
        int *&workspace, float *&globalScale, float *&cTmp,
        half **paddedOutput = nullptr) {
    workspace = nullptr;
    globalScale = nullptr;
    cTmp = nullptr;
    if (paddedOutput != nullptr) *paddedOutput = nullptr;
    if (weight.cudaData == nullptr || sizeK <= 0 || sizeN <= 0 || sms <= 0) {
        return false;
    }

    const size_t qweightBytes = (size_t)sizeK * sizeN / 2;
    const size_t scaleBytes = (size_t)(sizeK / NVFP4_GROUP_SIZE) * sizeN;
    const size_t workspaceBytes = (size_t)sms * 4 * sizeof(int);
    const size_t metadataOffset = qweightBytes + scaleBytes;
    const size_t metadataBytes =
        metadataOffset + workspaceBytes + sizeof(float);
    const size_t cTmpOffset = (metadataBytes + 15) & ~(size_t)15;
    size_t cTmpBytes = 0;
    if (sizeM > 0) {
        const int maxMBlock = sizeM <= 8
            ? 8 : std::min(64, ((sizeM + 15) / 16) * 16);
        cTmpBytes = (size_t)sms * maxMBlock * 256 * sizeof(float);
    }
    const size_t requiredBytes = cTmpOffset + cTmpBytes;
    if (metadataBytes > weight.GetBytes() || metadataOffset % alignof(int) != 0 ||
        (metadataOffset + workspaceBytes) % alignof(float) != 0 ||
        cTmpOffset % 16 != 0) {
        return false;
    }

    auto *tail = static_cast<uint8_t *>(weight.cudaData) + metadataOffset;
    workspace = reinterpret_cast<int *>(tail);
    globalScale = reinterpret_cast<float *>(tail + workspaceBytes);
    // Small shards may not reclaim enough source-layout tail space for the
    // largest warmup batch's temporary reduction buffer.  The launcher already
    // owns a per-device cached fallback for a null cTmp, so use the in-place
    // tail only when the complete buffer fits.  Workspace and globalScale must
    // always remain in-place because they are persistent repack metadata.
    if (sizeM > 0 && requiredBytes <= weight.GetBytes()) {
        cTmp = reinterpret_cast<float *>(
            static_cast<uint8_t *>(weight.cudaData) + cTmpOffset);
    }
    // Decode/verify can usually keep the padded output after the reduction
    // scratch in the reclaimed source-layout tail. Larger prefills borrow a
    // stream-ordered pool buffer in the caller instead.
    const size_t outputOffset = cTmp != nullptr ? requiredBytes : cTmpOffset;
    if (paddedOutput != nullptr && sizeM > 0 &&
        outputOffset <= weight.GetBytes() &&
        (size_t)sizeM * sizeN * sizeof(half) <= weight.GetBytes() - outputOffset) {
        *paddedOutput = reinterpret_cast<half *>(
            static_cast<uint8_t *>(weight.cudaData) + outputOffset);
    }
    return true;
}

// Convert FastLLM's interleaved source into two temporary standard layouts:
//   qweight: [K / 8, N] uint32, ready for gptq_marlin_repack(num_bits=4)
//   scales:  vLLM marlin_permute_scales followed by NVFP4 S0E5M3 processing
// Both destinations are temporary so that writes cannot overwrite unread
// source rows while performing the in-place conversion.
__global__ void FastllmNvfp4BuildMarlinInputsKernel(
        const uint8_t *__restrict__ source,
        uint32_t *__restrict__ qweight,
        uint8_t *__restrict__ scales,
        int logicalN, int sizeN, int sizeK, float commonGlobalScale) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = sizeK / NVFP4_GROUP_SIZE;
    const int packs = sizeK / 8;
    const size_t qweightCount = (size_t)packs * sizeN;
    const size_t scaleCount = (size_t)groups * sizeN;
    const int sourceRowBytes = groups * (8 + (int)sizeof(float));

    if (id < qweightCount) {
        int pack = id / sizeN;
        int out = id - pack * sizeN;
        int group = pack >> 1;
        int word = pack & 1;
        uint32_t packed = 0;
        if (out < logicalN) {
            const uint8_t *src = source + (size_t)out * sourceRowBytes +
                                 group * 12 + word * 4;
            packed = *reinterpret_cast<const uint32_t *>(src);
        }
        qweight[id] = packed;
    }

    if (id < scaleCount) {
        // nvfp4_marlin_process_scales applies [0,2,1,3] within each group
        // of four after marlin_permute_scales' 64-element permutation.
        constexpr int processPerm[4] = {0, 2, 1, 3};
        size_t block = id & ~(size_t)63;
        int position = id & 63;
        int afterProcess = (position & ~3) + processPerm[position & 3];
        int scaleSource = (afterProcess >> 3) + 8 * (afterProcess & 7);
        size_t transposedFlat = block + scaleSource;
        int group = transposedFlat / sizeN;
        int out = transposedFlat - group * sizeN;

        float effectiveScale = 0.0f;
        if (out < logicalN) {
            const uint8_t *src = source + (size_t)out * sourceRowBytes +
                                 group * 12 + 8;
            effectiveScale = *reinterpret_cast<const float *>(src);
        }
        half normalized = __float2half_rn(effectiveScale / commonGlobalScale);
        half shifted = __hmul(normalized, __float2half_rn(128.0f));
        uint16_t bits = __half_as_ushort(shifted);
        scales[id] = __half2float(shifted) < 2.0f
                         ? 0
                         : static_cast<uint8_t>(bits >> 7);
    }
}

__global__ void FastllmNvfp4MarlinCropOutputKernel(
        const half *padded, half *output, const half *bias,
        int rows, int logicalN, int packedN) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= logicalN) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        half value = padded[(size_t)row * packedN + col];
        output[(size_t)row * logicalN + col] =
            bias == nullptr ? value : __hadd(value, bias[col]);
    }
}

static bool EnsureNvfp4MarlinOnDevice(fastllm::Data &weight,
                                       int sizeK, int logicalN, int sizeN) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (HasNvfp4MarlinOnDevice(weight)) return true;
    if (weight.cudaData == nullptr || weight.blockM != NVFP4_GROUP_SIZE ||
        weight.blockK != 1 || !Nvfp4MarlinShapeSupported(sizeN, sizeK) ||
        weight.scales.empty()) {
        return false;
    }

    float commonGlobalScale = INFINITY;
    for (float scale : weight.scales) {
        if (std::isfinite(scale) && scale > 0.0f) {
            commonGlobalScale = std::min(commonGlobalScale, scale);
        }
    }
    if (!std::isfinite(commonGlobalScale) || commonGlobalScale <= 0.0f) {
        return false;
    }

    int device = 0, sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess ||
        sms <= 0) {
        return false;
    }

    int workspaceInts = std::max(1, sms * 4);
    int *workspace = nullptr;
    float *globalScale = nullptr;
    float *cTmp = nullptr;
    if (!GetNvfp4MarlinTailPointers(
            weight, sizeK, sizeN, sms, 0, workspace, globalScale, cTmp)) {
        return false;
    }
    float processedGlobalScale = commonGlobalScale * 128.0f;

    FastllmCudaClearThreadError();
    const size_t qweightCount = (size_t)(sizeK / 8) * sizeN;
    const size_t qweightBytes = qweightCount * sizeof(uint32_t);
    const size_t scaleCount = (size_t)(sizeK / NVFP4_GROUP_SIZE) * sizeN;
    const size_t temporaryBytes = qweightBytes + scaleCount;
    uint8_t *temporary = static_cast<uint8_t *>(
        FastllmCudaMalloc(temporaryBytes));
    if (temporary == nullptr || FastllmCudaGetThreadError()) {
        if (temporary != nullptr) FastllmCudaForceFree(temporary);
        FastllmCudaClearThreadError();
        return false;
    }

    auto *standardQweight = reinterpret_cast<uint32_t *>(temporary);
    uint8_t *temporaryScales = temporary + qweightBytes;
    const size_t workItems = std::max(qweightCount, scaleCount);
    const int threads = 256;
    const int blocks = static_cast<int>((workItems + threads - 1) / threads);
    FastllmNvfp4BuildMarlinInputsKernel<<<blocks, threads, 0,
                                          cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), standardQweight,
        temporaryScales, logicalN, sizeN, sizeK, commonGlobalScale);

    bool repacked = cudaPeekAtLastError() == cudaSuccess &&
                    FastllmCudaGptqMarlinRepackBitsStream(
                        standardQweight,
                        static_cast<uint32_t *>(weight.cudaData),
                        sizeK, sizeN, 4,
                        reinterpret_cast<void *>(cudaStreamPerThread));
    if (repacked) {
        cudaError_t copyState = cudaMemcpyAsync(
            static_cast<uint8_t *>(weight.cudaData) + qweightBytes,
            temporaryScales, scaleCount, cudaMemcpyDeviceToDevice,
            cudaStreamPerThread);
        repacked = copyState == cudaSuccess;
    }
    cudaError_t syncState = cudaStreamSynchronize(cudaStreamPerThread);
    // Conversion scratch can be very large and should not stay in FastLLM's
    // reusable CUDA pool after every model weight has been prepared.
    FastllmCudaForceFree(temporary);

    if (!repacked || syncState != cudaSuccess) {
        if (syncState != cudaSuccess) {
            printf("Error: NVFP4 Marlin in-place repack failed: %s.\n",
                   cudaGetErrorString(syncState));
            throw("nvfp4 marlin repack error");
        }
        // A failed repack may already have overwritten the source allocation;
        // do not silently execute the original-layout kernel in that case.
        if (cudaPeekAtLastError() != cudaSuccess || !repacked) {
            printf("Error: NVFP4 Marlin conversion failed after in-place repack began.\n");
            throw("nvfp4 marlin conversion error");
        }
        return false;
    }

    // The tail still contained unread source blocks until conversion was
    // complete, so initialise the in-place metadata only after the stream has
    // synchronized.  IsRepacked is the durable layout marker; CUDA copies of
    // this Data preserve both the marker and the tail bytes together.
    FastllmCudaMemset0(workspace, (size_t)workspaceInts * sizeof(int));
    FastllmCudaCopyFromHostToDevice(globalScale, &processedGlobalScale,
                                    sizeof(float));
    weight.IsRepacked = true;

    if (Nvfp4MarlinEnvFlag(
            "FASTLLM_CUDA_NVFP4_MARLIN_TRACE", false)) {
        int device = 0, major = 0, minor = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device);
        Nvfp4MarlinMode mode = Nvfp4MarlinModeFromEnv();
        printf("FastLLM NVFP4 Marlin: repacked %s [N=%d, packedN=%d, K=%d] on SM%d "
               "(mode=%s).\n", weight.name.c_str(), logicalN, sizeN, sizeK,
               major * 10 + minor, Nvfp4MarlinModeName(mode));
    }
    FastllmCudaClearThreadError();
    return true;
}

}  // namespace

extern "C" bool FastllmCudaHasNVFP4MarlinLayout(
        const fastllm::Data &weight) {
    return HasNvfp4MarlinOnDevice(weight);
}

extern "C" bool FastllmCudaTryMarlinHalfMatMulFloatNVFP4Block16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    int packedN = 0;
    if (!GetNvfp4MarlinPackedOutputDim(k, packedN)) return false;
    if (HasNvfp4MarlinOnDevice(weight)) {
        // Once converted in place, the original-layout fallback is no longer
        // valid, so always continue through Marlin even if an environment flag
        // is changed later in the process.
    } else {
        if (!Nvfp4MarlinDeviceSupported() || n < 1 ||
            weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            weight.blockM != NVFP4_GROUP_SIZE || weight.blockK != 1 ||
            !Nvfp4MarlinShapeSupported(packedN, m) || weight.scales.empty()) {
            return false;
        }
        // In auto mode this probes the actual compiled specialization and its
        // dynamic shared-memory configuration before the source buffer is
        // destructively repacked. Forced mode keeps the same safety check.
        if (!FastllmCudaMarlinNVFP4Supported(packedN, m)) return false;
        // Destructive preparation is restricted to synchronized small-M
        // warmup, avoiding allocations or repacks during CUDA graph capture.
        if (n > NVFP4_MARLIN_CONVERT_MAX_M ||
            !FastllmCudaGetNcclForceSync() ||
            !EnsureNvfp4MarlinOnDevice(weight, m, k, packedN)) {
            return false;
        }
    }

    half *cudaInput = static_cast<half *>(FastllmCudaPrepareInput(input));
    half *cudaOutput = static_cast<half *>(FastllmCudaPrepareOutput(output));
    auto *marlinWeight = static_cast<const uint32_t *>(weight.cudaData);
    const size_t qweightBytes = (size_t)m * packedN / 2;
    const uint8_t *marlinScales =
        static_cast<const uint8_t *>(weight.cudaData) + qweightBytes;
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    int *workspace = nullptr;
    float *globalScale = nullptr;
    float *cTmp = nullptr;
    half *paddedOutput = nullptr;
    if (!GetNvfp4MarlinTailPointers(
            weight, m, packedN, sms, n, workspace, globalScale, cTmp,
            packedN != k ? &paddedOutput : nullptr)) {
        printf("Error: NVFP4 Marlin in-place metadata is unavailable.\n");
        throw("nvfp4 marlin metadata error");
    }

    bool ownPaddedOutput = false;
    if (packedN != k && paddedOutput == nullptr) {
        void *temporary = nullptr;
        if (FastllmCudaTryMalloc(&temporary, (size_t)n * packedN * sizeof(half)) !=
            FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            printf("Error: NVFP4 Marlin padded output allocation failed.\n");
            throw("nvfp4 marlin padded output allocation error");
        }
        paddedOutput = static_cast<half *>(temporary);
        ownPaddedOutput = true;
    }
    auto releasePaddedOutput = [&]() {
        if (ownPaddedOutput &&
            !FastllmCudaFreeAfterStream(paddedOutput, cudaStreamPerThread)) {
            FastllmCudaFree(paddedOutput);
        }
    };
    bool ok = FastllmCudaMarlinHalfNVFP4Gemm(
        cudaInput, marlinWeight, marlinScales, globalScale,
        packedN != k ? paddedOutput : cudaOutput,
        n, packedN, m, workspace, cTmp);
    if (!ok) {
        releasePaddedOutput();
        printf("Error: NVFP4 Marlin GEMM failed after the CUDA weight was repacked in place.\n");
        throw("nvfp4 marlin gemm error");
    }

    const half *cudaBias = bias.dims.size() > 0 && !weight.extraCudaHalfData.empty()
        ? static_cast<const half *>(weight.extraCudaHalfData[0]) : nullptr;
    if (packedN != k) {
        const int threads = 256;
        dim3 grid((k + threads - 1) / threads, std::min(n, 65535));
        FastllmNvfp4MarlinCropOutputKernel<<<grid, threads, 0, cudaStreamPerThread>>>(
            paddedOutput, cudaOutput, cudaBias, n, k, packedN);
        releasePaddedOutput();
    } else if (cudaBias != nullptr) {
        FastllmCudaBiasKernel<<<n, 256, 0, cudaStreamPerThread>>>(
            cudaOutput, static_cast<half *>(weight.extraCudaHalfData[0]), k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
