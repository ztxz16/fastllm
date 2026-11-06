/* Dense NVFP4 Marlin W4A16 integration, adapted from vLLM's
 * marlin_utils_fp4.py and quantization/marlin kernels (Apache-2.0). */
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
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

namespace {

struct Cache {
    const void *source = nullptr;
    uint32_t *weight = nullptr;
    uint8_t *scales = nullptr;
    float *globalScale = nullptr;
    int *workspace = nullptr;
    int inFeatures = 0;
    int outFeatures = 0;
};

std::mutex cacheMutex;
std::map<std::pair<const fastllm::Data *, int>, Cache> caches;

enum class BackendState : uint8_t {
    Uninitialized,
    Marlin,
    Legacy,
};

std::mutex stateMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool StrictW4A4Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_W4A4_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device, BackendState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    backendStates[{&weight, device}] = state;
}

static void FixLegacyBackend(const fastllm::Data &weight, int device,
                             const char *reason) {
    SetBackendState(weight, device, BackendState::Legacy);
    if (TraceEnabled()) {
        std::fprintf(stderr,
            "[fastllm][nvfp4] backend=legacy state=fixed device=%d reason=%s name=%s\n",
            device, reason, weight.name.c_str());
    }
}

static int RuntimeArch() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return 0;
    return major * 10 + minor;
}

__global__ void Block16ToGptqKernel(const uint8_t *source, uint32_t *qweight,
                                    int inFeatures, int outFeatures,
                                    int bytesPerRow) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int packs = inFeatures / 8;
    if (index >= packs * outFeatures) return;
    const int pack = index / outFeatures;
    const int out = index - pack * outFeatures;
    const int block = pack / 2;
    const int within = (pack & 1) * 4;
    const uint8_t *src = source + (size_t)out * bytesPerRow
                       + (size_t)block * (8 + sizeof(float)) + within;
    qweight[index] = *reinterpret_cast<const uint32_t *>(src);
}

__global__ void Block16ToMarlinScaleKernel(const uint8_t *source,
                                           uint8_t *scales,
                                           int inFeatures, int outFeatures,
                                           int bytesPerRow) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = inFeatures / 16;
    if (index >= groups * outFeatures) return;
    const int group = index / outFeatures;
    const int out = index - group * outFeatures;
    const uint8_t *block = source + (size_t)out * bytesPerRow
                         + (size_t)group * (8 + sizeof(float));
    const float value = *reinterpret_cast<const float *>(block + 8);
    // vLLM NVFP4 Marlin uses a special unsigned S0E5M3 byte, not ordinary
    // E4M3.  It is the high byte of FP16(scale * 2^7) after a one-bit shift.
    const half scaled = __float2half_rn(value * 128.0f);
    const uint16_t bits = reinterpret_cast<const uint16_t &>(scaled);
    const uint8_t encoded = (uint8_t)(bits >> 7);

    const int block64 = index & ~63;
    const int inner64 = index & 63;
    const int permuted64 = block64 + (inner64 & 7) * 8 + (inner64 >> 3);
    const int block4 = permuted64 & ~3;
    const int inner4 = permuted64 & 3;
    const int inverse4 = inner4 == 1 ? 2 : (inner4 == 2 ? 1 : inner4);
    scales[block4 + inverse4] = encoded;
}

static void Release(Cache &cache) {
    if (cache.weight != nullptr) FastllmCudaFree(cache.weight);
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    if (cache.globalScale != nullptr) FastllmCudaFree(cache.globalScale);
    if (cache.workspace != nullptr) FastllmCudaFree(cache.workspace);
    cache = Cache();
}

static Cache *GetCache(fastllm::Data &weight, int inFeatures, int outFeatures) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto key = std::make_pair((const fastllm::Data *)&weight, device);
    Cache &cache = caches[key];
    if (cache.inFeatures == inFeatures && cache.outFeatures == outFeatures &&
        cache.weight != nullptr &&
        cache.scales != nullptr && cache.globalScale != nullptr &&
        cache.workspace != nullptr) return &cache;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    Release(cache);
    if (weight.cudaData == nullptr &&
        !weight.RestoreCudaDataForRepackedWeight()) return nullptr;

    const size_t qweightCount = (size_t)(inFeatures / 8) * outFeatures;
    uint32_t *standard = static_cast<uint32_t *>(FastllmCudaMalloc(qweightCount * sizeof(uint32_t)));
    cache.weight = static_cast<uint32_t *>(FastllmCudaMalloc(qweightCount * sizeof(uint32_t)));
    cache.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
        (size_t)(inFeatures / 16) * outFeatures));
    cache.globalScale = static_cast<float *>(FastllmCudaMalloc(sizeof(float)));
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cache.workspace = static_cast<int *>(FastllmCudaMalloc(
        (size_t)std::max(1, sms * 4) * sizeof(int)));
    if (standard == nullptr || cache.weight == nullptr || cache.scales == nullptr ||
        cache.globalScale == nullptr || cache.workspace == nullptr) {
        if (standard != nullptr) FastllmCudaFree(standard);
        Release(cache); return nullptr;
    }

    const int bytesPerRow = (inFeatures / 16) * (8 + sizeof(float));
    const int threads = 256;
    Block16ToGptqKernel<<<(qweightCount + threads - 1) / threads, threads, 0,
                           cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), standard,
        inFeatures, outFeatures, bytesPerRow);
    const size_t scaleCount = (size_t)(inFeatures / 16) * outFeatures;
    Block16ToMarlinScaleKernel<<<(scaleCount + threads - 1) / threads, threads, 0,
                                 cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), cache.scales,
        inFeatures, outFeatures, bytesPerRow);
    // The FP4 and scale dequant fast paths intentionally omit exponent-bias
    // multiplies.  For FP16 output vLLM compensates them with 2^(14 - 7).
    const float globalScale = 128.0f;
    bool ok = cudaPeekAtLastError() == cudaSuccess &&
              cudaMemcpyAsync(cache.globalScale, &globalScale, sizeof(float),
                              cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess &&
              cudaMemsetAsync(cache.workspace, 0,
                              (size_t)std::max(1, sms * 4) * sizeof(int),
                              cudaStreamPerThread) == cudaSuccess &&
              FastllmCudaGptqMarlinRepackBitsStream(
                  standard, cache.weight, inFeatures, outFeatures, 4,
                  (void *)cudaStreamPerThread);
    FastllmCudaFree(standard);
    if (!ok) { Release(cache); return nullptr; }
    cache.source = weight.cudaData;
    cache.inFeatures = inFeatures;
    cache.outFeatures = outFeatures;
    return &cache;
}

} // namespace

bool FastllmCudaTryMarlinHalfMatMulNVFP4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Legacy) return false;
    const bool warmup = state == BackendState::Uninitialized;
    const int arch = RuntimeArch();
    const bool canRun = arch >= 75 &&
        input.dataType == fastllm::DataType::FLOAT16 &&
        output.dataType == fastllm::DataType::FLOAT16 &&
        weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
        weight.blockM == 16 && bias.dims.empty() && n > 0 &&
        m > 0 && k > 0 && m % 64 == 0 && k % 64 == 0 &&
        (weight.cudaData == nullptr ||
         (!weight.cudaDataBorrowed && !weight.multiDeviceData &&
          (weight.cpuData != nullptr || !weight.numasData.empty())));
    if (!canRun) {
        if (warmup) {
            FixLegacyBackend(weight, device, "Marlin unsupported");
            return false;
        }
        throw std::runtime_error(
            "NVFP4 Marlin backend received an unsupported shape or dtype after warmup");
    }
    Cache *cache = GetCache(weight, m, k);
    if (cache == nullptr) {
        if (warmup) FixLegacyBackend(weight, device, "Marlin cache unavailable");
        else throw std::runtime_error("NVFP4 Marlin cache disappeared after warmup");
        return false;
    }
    half *cudaInput = static_cast<half *>(FastllmCudaPrepareInput(input));
    half *cudaOutput = static_cast<half *>(FastllmCudaPrepareOutput(output));
    if (cudaInput == nullptr || cudaOutput == nullptr) {
        if (warmup) {
            Release(*cache);
            FixLegacyBackend(weight, device, "Marlin input/output unavailable");
            return false;
        }
        throw std::runtime_error("NVFP4 Marlin input/output preparation failed after warmup");
    }
    bool ok = FastllmCudaMarlinHalfNVFP4Gemm(
        cudaInput, cache->weight, cache->scales, cache->globalScale,
        cudaOutput, n, k, m, cache->workspace);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    if (ok && warmup) ok = cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    if (ok && warmup) {
        if (!weight.ReleaseCudaDataForRepackedWeight()) ok = false;
        else {
            cache->source = nullptr;
            SetBackendState(weight, device, BackendState::Marlin);
            if (TraceEnabled()) {
                std::fprintf(stderr,
                    "[fastllm][nvfp4] backend=marlin state=fixed device=%d name=%s\n",
                    device, weight.name.c_str());
            }
        }
    }
    if (!ok && warmup) {
        Release(*cache);
        FixLegacyBackend(weight, device, "Marlin warmup failed");
    } else if (!ok) {
        throw std::runtime_error(
            "NVFP4 Marlin backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

namespace {

using Nvfp4BackendState = FastllmCudaNvfp4BackendState;

struct Nvfp4BackendCandidate {
    Nvfp4BackendState backend;
    bool changesActivationSemantics;
    const char *name;
};

constexpr std::array<Nvfp4BackendCandidate, 3> nvfp4W4A4Candidates = {{
    {Nvfp4BackendState::CutlassW4A4, false, "cutlass-w4a4"},
    {Nvfp4BackendState::MarlinW4A16Fallback, true, "marlin-w4a16-fallback"},
    {Nvfp4BackendState::NativeW4A16Fallback, true, "native-w4a16-fallback"},
}};

static void TraceSelectedBackend(const fastllm::Data &weight, int device,
                                 const Nvfp4BackendCandidate &candidate) {
    if (!TraceEnabled()) return;
    std::fprintf(stderr,
        "[fastllm][nvfp4] backend=%s state=fixed device=%d "
        "activation_semantics=%s name=%s\n",
        candidate.name, device,
        candidate.changesActivationSemantics ? "w4a16" : "w4a4",
        weight.name.c_str());
}

} // namespace

/**
 * 为NVFP4 W4A4 Linear选择并执行固定后端。
 *
 * 首次调用按CUTLASS W4A4、Marlin W4A16兼容路径、原生W4A16兼容路径
 * 的顺序选择。CUTLASS保持W4A4激活语义；后两者不量化激活，因此只在
 * 非STRICT模式下作为兼容降级。选择结果按“权重对象 + GPU”写入统一状态，
 * 后续调用只执行已固定的后端，不再逐个探测。固定后端运行失败时由对应
 * 执行函数直接抛错，不允许在正式推理期间再次切换。
 *
 * 参数沿用FastLLM Linear语义：output[n,k] = input[n,m] * weight[k,m]^T。
 *
 * @param input  FP16或BF16输入，逻辑形状为[n,m]。
 * @param weight NVFP4_BLOCK_16权重，逻辑形状为[k,m]。
 * @param bias   可选FP32偏置，长度为k。
 * @param output 输出张量，逻辑形状为[n,k]。
 * @param n      输入行数，通常为token数。
 * @param m      输入特征数。
 * @param k      输出特征数。
 * @return true表示CUTLASS或Marlin已完成计算；false表示已固定使用原生
 *         W4A16兼容路径，调用方应进入普通NVFP4_BLOCK_16 Linear。
 */
bool FastllmCudaTryNvfp4W4A4Linear(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16) return false;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    Nvfp4BackendState state =
        FastllmCudaGetNvfp4W4A4BackendState(weight, device);

    if (state == Nvfp4BackendState::CutlassW4A4) {
        return TryCudaCutlassNvfp4W4A4(input, weight, bias, output, n, k, m);
    }
    if (state == Nvfp4BackendState::MarlinW4A16Fallback) {
        return FastllmCudaTryMarlinHalfMatMulNVFP4(
            input, weight, bias, output, n, m, k);
    }
    if (state == Nvfp4BackendState::NativeW4A16Fallback) return false;

    // Prepared和Uninitialized都属于首次选择；CUTLASS内部负责真实GEMM
    // 验证，并在失败时恢复原始权重、把CUTLASS候选标记为Rejected。
    if (state == Nvfp4BackendState::Prepared ||
        state == Nvfp4BackendState::Uninitialized) {
        if (TryCudaCutlassNvfp4W4A4(
                input, weight, bias, output, n, k, m)) {
            return true;
        }
        state = FastllmCudaGetNvfp4W4A4BackendState(weight, device);
    }

    if (state != Nvfp4BackendState::Rejected) {
        throw std::runtime_error(
            "NVFP4 W4A4 backend selection entered an invalid state");
    }
    if (StrictW4A4Enabled()) {
        throw std::runtime_error(
            "strict NVFP4 W4A4 requires the native CUTLASS backend");
    }

    // CUTLASS被拒绝后才允许尝试改变激活语义的兼容后端。Marlin函数
    // 自己完成首次同步验证；失败后保留原始CUDA权重供原生W4A16使用。
    const Nvfp4BackendCandidate &marlin = nvfp4W4A4Candidates[1];
    if (FastllmCudaTryMarlinHalfMatMulNVFP4(
            input, weight, bias, output, n, m, k)) {
        FastllmCudaSetNvfp4W4A4BackendState(weight, device, marlin.backend);
        TraceSelectedBackend(weight, device, marlin);
        return true;
    }

    const Nvfp4BackendCandidate &native = nvfp4W4A4Candidates[2];
    FastllmCudaSetNvfp4W4A4BackendState(weight, device, native.backend);
    TraceSelectedBackend(weight, device, native);
    return false;
}

void FastllmCudaReleaseNvfp4MarlinCache(const fastllm::Data *weight) {
    // Marlin cache只会以NVFP4_BLOCK_16权重为键。通用Data生命周期也会
    // 调用本接口，因此先按类型拒绝普通临时张量，避免每次Decode为
    // input、position、embedding和logits获取无关的全局互斥锁。
    if (weight == nullptr ||
        weight->dataType != fastllm::DataType::NVFP4_BLOCK_16) {
        return;
    }
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        bool hasCache = false;
        for (const auto &entry : caches) {
            if (entry.first.first == weight) { hasCache = true; break; }
        }
        if (hasCache) {
            int originalDevice = 0;
            cudaGetDevice(&originalDevice);
            for (auto it = caches.begin(); it != caches.end();) {
                if (it->first.first == weight) {
                    cudaSetDevice(it->first.second);
                    Release(it->second);
                    it = caches.erase(it);
                } else ++it;
            }
            cudaSetDevice(originalDevice);
        }
    }
    {
        std::lock_guard<std::mutex> guard(stateMutex);
        for (auto it = backendStates.begin(); it != backendStates.end();) {
            if (it->first.first == weight) it = backendStates.erase(it);
            else ++it;
        }
    }
}

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

constexpr int NVFP4_GROUP_SIZE = 16;
constexpr int NVFP4_MARLIN_CONVERT_MAX_M = 8;

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

static const char *Nvfp4MarlinModeName(Nvfp4MarlinMode mode) {
    if (mode == Nvfp4MarlinMode::DISABLED) return "disabled";
    if (mode == Nvfp4MarlinMode::ENABLED) return "enabled";
    return "auto";
}

static bool Nvfp4MarlinDeviceSupported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) {
        return false;
    }
    Nvfp4MarlinMode mode = Nvfp4MarlinModeFromEnv();
    if (mode == Nvfp4MarlinMode::DISABLED) return false;
    // This is the same architecture gate used by vLLM's FP4 Marlin path.
    // SM75 uses Turing MMA with a two-stage pipeline; SM80+ uses four stages.
    return major * 10 + minor >= 75;
#endif
}

static bool HasNvfp4MarlinOnDevice(const fastllm::Data &weight) {
    return weight.cudaData != nullptr &&
           weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
           weight.blockM == NVFP4_GROUP_SIZE && weight.blockK == 1 &&
           weight.IsRepacked;
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

// The source format occupies twelve bytes per group (eight packed weight
// bytes plus one float scale), while the Marlin layout needs only nine bytes
// (eight packed weight bytes plus one byte scale).  Keep Marlin's tiny lock
// workspace and tensor scale in that reclaimed tail.  Besides avoiding
// redundant allocations, this matters for hybrid-attention models where
// batch-sized recurrent state already leaves very little free VRAM.
static bool GetNvfp4MarlinTailPointers(
        fastllm::Data &weight, int sizeK, int sizeN, int sms, int sizeM,
        int *&workspace, float *&globalScale, float *&cTmp) {
    workspace = nullptr;
    globalScale = nullptr;
    cTmp = nullptr;
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
        int sizeN, int sizeK, float commonGlobalScale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = sizeK / NVFP4_GROUP_SIZE;
    const int packs = sizeK / 8;
    const int qweightCount = packs * sizeN;
    const int scaleCount = groups * sizeN;
    const int sourceRowBytes = groups * (8 + (int)sizeof(float));

    if (id < qweightCount) {
        int pack = id / sizeN;
        int out = id - pack * sizeN;
        int group = pack >> 1;
        int word = pack & 1;
        const uint8_t *src = source + (size_t)out * sourceRowBytes +
                             group * 12 + word * 4;
        qweight[id] = *reinterpret_cast<const uint32_t *>(src);
    }

    if (id < scaleCount) {
        // nvfp4_marlin_process_scales applies [0,2,1,3] within each group
        // of four after marlin_permute_scales' 64-element permutation.
        constexpr int processPerm[4] = {0, 2, 1, 3};
        int block = id & ~63;
        int position = id & 63;
        int afterProcess = (position & ~3) + processPerm[position & 3];
        int scaleSource = (afterProcess >> 3) + 8 * (afterProcess & 7);
        int transposedFlat = block + scaleSource;
        int group = transposedFlat / sizeN;
        int out = transposedFlat - group * sizeN;

        const uint8_t *src = source + (size_t)out * sourceRowBytes +
                             group * 12 + 8;
        float effectiveScale = *reinterpret_cast<const float *>(src);
        half normalized = __float2half_rn(effectiveScale / commonGlobalScale);
        half shifted = __hmul(normalized, __float2half_rn(128.0f));
        uint16_t bits = __half_as_ushort(shifted);
        scales[id] = __half2float(shifted) < 2.0f
                         ? 0
                         : static_cast<uint8_t>(bits >> 7);
    }
}

static bool EnsureNvfp4MarlinOnDevice(fastllm::Data &weight,
                                       int sizeK, int sizeN) {
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
        temporaryScales, sizeN, sizeK, commonGlobalScale);

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
        printf("FastLLM NVFP4 Marlin: repacked %s [N=%d, K=%d] on SM%d "
               "(mode=%s).\n", weight.name.c_str(), sizeN, sizeK,
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
    if (HasNvfp4MarlinOnDevice(weight)) {
        // Once converted in place, the original-layout fallback is no longer
        // valid, so always continue through Marlin even if an environment flag
        // is changed later in the process.
    } else {
        if (!Nvfp4MarlinDeviceSupported() || n < 1 ||
            weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            weight.blockM != NVFP4_GROUP_SIZE || weight.blockK != 1 ||
            !Nvfp4MarlinShapeSupported(k, m) || weight.scales.empty()) {
            return false;
        }
        // In auto mode this probes the actual compiled specialization and its
        // dynamic shared-memory configuration before the source buffer is
        // destructively repacked. Forced mode keeps the same safety check.
        if (!FastllmCudaMarlinNVFP4Supported(k, m)) return false;
        // Destructive preparation is restricted to synchronized small-M
        // warmup, avoiding allocations or repacks during CUDA graph capture.
        if (n > NVFP4_MARLIN_CONVERT_MAX_M ||
            !FastllmCudaGetNcclForceSync() ||
            !EnsureNvfp4MarlinOnDevice(weight, m, k)) {
            return false;
        }
    }

    half *cudaInput = static_cast<half *>(FastllmCudaPrepareInput(input));
    half *cudaOutput = static_cast<half *>(FastllmCudaPrepareOutput(output));
    auto *marlinWeight = static_cast<const uint32_t *>(weight.cudaData);
    const size_t qweightBytes = (size_t)m * k / 2;
    const uint8_t *marlinScales =
        static_cast<const uint8_t *>(weight.cudaData) + qweightBytes;
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    int *workspace = nullptr;
    float *globalScale = nullptr;
    float *cTmp = nullptr;
    if (!GetNvfp4MarlinTailPointers(
            weight, m, k, sms, n, workspace, globalScale, cTmp)) {
        printf("Error: NVFP4 Marlin in-place metadata is unavailable.\n");
        throw("nvfp4 marlin metadata error");
    }

    bool ok = FastllmCudaMarlinHalfNVFP4Gemm(
        cudaInput, marlinWeight, marlinScales, globalScale, cudaOutput,
        n, k, m, workspace, cTmp);
    if (!ok) {
        printf("Error: NVFP4 Marlin GEMM failed after the CUDA weight was repacked in place.\n");
        throw("nvfp4 marlin gemm error");
    }

    if (bias.dims.size() > 0 && !weight.extraCudaHalfData.empty() &&
        weight.extraCudaHalfData[0] != nullptr) {
        FastllmCudaBiasKernel<<<n, 256, 0, cudaStreamPerThread>>>(
            cudaOutput, static_cast<half *>(weight.extraCudaHalfData[0]), k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
