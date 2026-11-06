#include "fastllm-cuda.cuh"
#include "utils/math.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

struct WeightCache {
    uint8_t *weight = nullptr;
    uint8_t *scales = nullptr;
    float *alpha = nullptr;
    int sourceRows = 0;
    int sourceColumns = 0;
    int paddedRows = 0;
    int paddedColumns = 0;
};

/**
 * NVFP4 W4A4动态激活量化使用的GPU临时缓冲区。
 *
 * activation保存按两个E2M1元素打包为一个字节的[M, paddedK]激活；
 * scales保存与激活对应、已按CUTLASS要求完成swizzle的E4M3 block scale。
 * 两个Bytes字段记录当前分配容量，用于同一GPU上的后续计算复用和扩容，
 * 不表示某一次计算实际使用的元素数量。缓冲区按GPU持久复用，只扩不缩。
 */
struct ActivationScratch {
    uint8_t *activation = nullptr;
    uint8_t *scales = nullptr;
    size_t activationBytes = 0;
    size_t scaleBytes = 0;
};

struct OutputScratch {
    void *output = nullptr;
    size_t bytes = 0;
};

std::mutex cacheMutex;
// CUTLASS重排权重缓存。键为“逻辑权重Data对象地址 + CUDA设备号”，值中
// 保存该权重在对应GPU上的打包E2M1权重、swizzle E4M3 group scale和
// FP32 global scale。相同Data在不同GPU上必须分别重排，不能共享设备指针。
// 所有读取、创建和销毁操作均由cacheMutex保护。
std::map<std::pair<const fastllm::Data *, int>, WeightCache> weightCaches;
std::map<int, ActivationScratch> activationScratch;
std::map<int, OutputScratch> outputScratch;

using BackendState = FastllmCudaNvfp4BackendState;

enum class FusionState : uint8_t {
    Uninitialized,
    Enabled,
    Disabled,
};

std::mutex stateMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;
std::map<std::pair<const fastllm::Data *, int>, FusionState> fusionStates;

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_W4A4");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

// 严格模式供算子验证和性能测试使用：Dense W4A4只允许CUTLASS，
// 权重预重排、首次GEMM验证或固定后端任一阶段失败都直接报错。
static bool StrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_W4A4_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[noreturn]] static void ThrowStrict(const char *reason) {
    throw std::runtime_error(
        std::string("strict NVFP4 dense requires the CUTLASS backend: ") + reason);
}

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int RuntimeArch() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return 0;
    return major * 10 + minor;
}

static void Trace(const char *path, const char *reason, int n, int m, int k, int arch) {
    if (TraceEnabled()) {
        std::fprintf(stderr, "[fastllm][nvfp4] path=%s reason=%s n=%d m=%d k=%d sm=%d\n",
                     path, reason, n, m, k, arch);
    }
}

static bool TraceSynchronize(const char *stage, bool force = false) {
    if (!force && !TraceEnabled()) return true;
    const cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[fastllm][nvfp4] stage=%s status=%s\n",
                     stage, cudaGetErrorString(status));
        return false;
    }
    return true;
}

static bool SemanticsSupported(const fastllm::Data &input,
                               const fastllm::Data &weight,
                               const fastllm::Data &bias,
                               const fastllm::Data &output,
                               int n, int m, int k, bool checkBackendPolicy,
                               const char **reason) {
    // 首次后端选择只检查显式开关，不使用运行时M作为CUTLASS准入条件。
    if (checkBackendPolicy && !Enabled()) { *reason = "disabled"; return false; }
    if (input.dataType != fastllm::DataType::FLOAT16 &&
        input.dataType != fastllm::DataType::BFLOAT16) {
        *reason = "activation is not FP16/BF16"; return false;
    }
    if (output.dataType != input.dataType) { *reason = "output dtype differs from input"; return false; }
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 || weight.blockM != 16) {
        *reason = "weight is not NVFP4_BLOCK_16"; return false;
    }
    if (m <= 0 || k <= 0 || m % 16 != 0) {
        *reason = "K is not compatible with NVFP4 block-16"; return false;
    }
    const size_t expectedScales = (size_t)k * ((size_t)m / 16);
    if (weight.nvfp4GroupScales.size() != expectedScales ||
        !std::isfinite(weight.nvfp4GlobalScale) || weight.nvfp4GlobalScale <= 0.0f) {
        *reason = "original E4M3/global scales are unavailable"; return false;
    }
    if (!bias.dims.empty() &&
        (bias.dataType != fastllm::DataType::FLOAT32 || bias.Count(0) != (uint64_t)k)) {
        *reason = "bias must be empty or FP32 with N elements"; return false;
    }
    if (weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m) {
        *reason = "weight shape mismatch"; return false;
    }
    if (weight.cudaData != nullptr &&
        (weight.cudaDataBorrowed || weight.multiDeviceData ||
         (weight.cpuData == nullptr && weight.numasData.empty()))) {
        *reason = "original CUDA weight has no releasable host source"; return false;
    }
    return true;
}

static int CurrentDevice() {
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess ? device : -1;
}

/**
 * 获取指定NVFP4权重在指定GPU上的CUTLASS后端状态。
 *
 * 状态以“Data对象地址 + CUDA设备号”为键保存，因此同一权重移动到
 * 另一张GPU后会拥有独立的后端生命周期。读取过程由stateMutex保护；
 * 如果尚未登记状态，则返回Uninitialized，但不会向状态表插入新记录。
 *
 * @param weight 用于标识模型权重对象。
 * @param device CUDA设备号。
 * @return 当前后端状态；除CUTLASS生命周期外，也可能是统一选择器固定的
 *         MarlinW4A16Fallback或NativeW4A16Fallback。
 */
static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device, BackendState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    backendStates[{&weight, device}] = state;
}

static FusionState GetFusionState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = fusionStates.find({&weight, device});
    return it == fusionStates.end() ? FusionState::Uninitialized : it->second;
}

static void ReleaseWeightCacheForDevice(const fastllm::Data *weight, int device);

static void SetFusionState(const fastllm::Data &weight, int device, FusionState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    fusionStates[{&weight, device}] = state;
}

/**
 * 将当前GPU上的NVFP4后端固定为Legacy，并确保原始CUDA权重可用。
 *
 * 该函数只用于首次CUTLASS GEMM验证失败。预重排权重先被销毁，再从
 * CPU/NUMA/mmap保存的原始格式恢复到当前GPU，避免失败后长期保留两份
 * 设备权重。恢复成功后状态固定为Rejected，正式推理不再尝试CUTLASS。
 *
 * @param weight 首次CUTLASS验证失败的NVFP4权重。
 * @param device 权重当前所在的CUDA设备号。
 * @return true表示Legacy原始CUDA权重已经就绪。
 */
static bool RejectCutlassAndRestoreLegacy(fastllm::Data &weight, int device) {
    ReleaseWeightCacheForDevice(&weight, device);
    if (weight.cudaData == nullptr &&
        !weight.RestoreCudaDataForRepackedWeight(device)) {
        return false;
    }
    SetBackendState(weight, device, BackendState::Rejected);
    return true;
}

static void ReleaseCache(WeightCache &cache) {
    if (cache.weight != nullptr) FastllmCudaFree(cache.weight);
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    if (cache.alpha != nullptr) FastllmCudaFree(cache.alpha);
    cache = WeightCache();
}

/**
 * 获取或创建指定权重在当前GPU上的CUTLASS格式cache。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 * cache中保存重排后的NVFP4权重、swizzle后的E4M3 group scale和FP32
 * global scale。权重表示只由N/K决定，M不会参与cache匹配或权重重排；
 * 保留M参数是为了与上层CUTLASS执行接口保持一致。
 *
 * 创建cache时，如果原始CUDA权重已释放，会先从host/mmap源临时恢复。
 * releaseSourceAfterRepack为true时，重排成功后立即释放这份原始CUDA
 * 权重；为false时，由外层warmup生命周期在验证成功后决定是否释放。
 *
 * @param weight                   NVFP4_BLOCK_16原始权重及两级scale元数据。
 * @param m                        GEMM的M维；不影响权重cache，仅保持接口一致。
 * @param n                        GEMM的N维，即输出特征数和权重行数。
 * @param k                        GEMM的K维，即输入特征数和权重列数。
 * @param releaseSourceAfterRepack 是否在重排成功后立即释放原始CUDA权重。
 * @return 成功时返回当前GPU对应的WeightCache；失败时返回nullptr。
 */
static WeightCache *GetWeightCache(fastllm::Data &weight, int m, int n, int k,
                                   bool releaseSourceAfterRepack) {
    // M不改变B矩阵的内容或布局，因此不进入cache键和重排尺寸计算。
    (void)m;

    // 每张GPU保存独立cache，避免跨设备复用无效的CUDA指针。
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto key = std::make_pair((const fastllm::Data *)&weight, device);
    WeightCache &cache = weightCaches[key];
    const int paddedK = fastllm::round_to_next_multiple_of(k, 32);
    const int paddedN = fastllm::round_to_next_multiple_of(n, 32);

    // N/K及已分配内容完全匹配时直接复用，不重复上传和重排权重。
    if (cache.sourceRows == n && cache.sourceColumns == k &&
        cache.paddedRows == paddedN &&
        cache.paddedColumns == paddedK &&
        cache.weight != nullptr && cache.scales != nullptr && cache.alpha != nullptr) return &cache;

    // CUDA Graph捕获期间不能安全地分配、释放或重建持久权重cache。
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    ReleaseCache(cache);

    // 原始CUDA表示可能已被上一次成功warmup释放；需要重建cache时，
    // 从保留的host/mmap源恢复一份临时原始权重。
    if (weight.cudaData == nullptr &&
        !weight.RestoreCudaDataForRepackedWeight()) return nullptr;

    // 分配CUTLASS持久权重、scale和global scale。sourceScales只是把host
    // 中的原始E4M3 group scale送入重排kernel的临时设备缓冲区。
    cache.weight = static_cast<uint8_t *>(FastllmCudaMalloc((size_t)paddedN * paddedK / 2));
    cache.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
        FastllmCudaNvfp4SwizzledScaleBytes(paddedN, paddedK)));
    cache.alpha = static_cast<float *>(FastllmCudaMalloc(sizeof(float)));
    const size_t sourceScaleBytes = (size_t)n * (k / 16);
    uint8_t *sourceScales = static_cast<uint8_t *>(FastllmCudaMalloc(sourceScaleBytes));
    if (cache.weight == nullptr || cache.scales == nullptr || cache.alpha == nullptr) {
        if (sourceScales != nullptr) FastllmCudaFree(sourceScales);
        ReleaseCache(cache); return nullptr;
    }

    // 保留原始两级缩放：group scale只做布局重排，不重新量化；FP32
    // global scale单独复制到alpha，由CUTLASS GEMM直接消费。
    const bool queued = sourceScales != nullptr &&
        cudaMemcpyAsync(sourceScales, weight.nvfp4GroupScales.data(), sourceScaleBytes,
                        cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess &&
        FastllmCudaNvfp4Block16ToCutlassPadded(
            static_cast<const uint8_t *>(weight.cudaData), sourceScales, cache.weight,
            cache.scales, n, k, paddedN, paddedK, (void *)cudaStreamPerThread) &&
        cudaMemcpyAsync(cache.alpha, &weight.nvfp4GlobalScale,
                        sizeof(weight.nvfp4GlobalScale), cudaMemcpyHostToDevice,
                        cudaStreamPerThread) == cudaSuccess;

    // sourceScales和可能恢复的原始权重都有生命周期要求，因此等待上传与
    // 重排完成后再释放临时缓冲区，并把异步错误转换为cache创建失败。
    const cudaError_t syncStatus = cudaStreamSynchronize(cudaStreamPerThread);
    if (sourceScales != nullptr) FastllmCudaFree(sourceScales);
    if (!queued || syncStatus != cudaSuccess) {
        ReleaseCache(cache);
        return nullptr;
    }

    // 记录cache对应的原始N/K和补齐尺寸，供后续调用精确判断能否复用。
    cache.sourceRows = n;
    cache.sourceColumns = k;
    cache.paddedRows = paddedN;
    cache.paddedColumns = paddedK;

    // 预重排路径可在此直接释放原始CUDA权重；普通Linear路径会等warmup
    // GEMM验证成功后再由外层释放，保留一次性降级机会。
    const bool released = !releaseSourceAfterRepack ||
                          weight.ReleaseCudaDataForRepackedWeight();
    if (!released) {
        ReleaseCache(cache);
        return nullptr;
    }
    if (releaseSourceAfterRepack && released && TraceEnabled()) {
        std::fprintf(stderr,
                     "[fastllm][nvfp4] weight_source=released bytes=%zu name=%s\n",
                     (size_t)weight.expansionBytes, weight.name.c_str());
    }
    return &cache;
}

/**
 * 释放指定权重在指定GPU上的CUTLASS重排缓存。
 *
 * 函数只处理weightCaches中精确匹配“weight + device”的条目。释放前切换
 * 到缓存所属GPU，避免在错误设备上释放CUDA指针；完成后恢复调用前设备。
 * 本函数不会恢复原始CUDA权重，也不会修改BackendState或FusionState，
 * 后端状态如何变化由调用方根据warmup、换卡或Data生命周期决定。
 *
 * @param weight 用于定位缓存的逻辑权重对象；不取得其所有权。
 * @param device 缓存所属的CUDA设备号。
 */
static void ReleaseWeightCacheForDevice(const fastllm::Data *weight, int device) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto it = weightCaches.find({weight, device});
    if (it != weightCaches.end()) {
        int originalDevice = 0;
        cudaGetDevice(&originalDevice);
        cudaSetDevice(device);
        ReleaseCache(it->second);
        weightCaches.erase(it);
        cudaSetDevice(originalDevice);
    }
}

/**
 * 获取或扩容当前GPU上的NVFP4激活临时缓冲区。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 * scratch包含打包后的E2M1激活和对应的swizzle E4M3 block scale。
 * 激活表示只由M/K决定，N不影响缓冲区大小；保留N参数是为了与上层
 * CUTLASS执行接口保持一致。缓冲区按GPU复用，只扩容、不主动缩小。
 *
 * @param m GEMM的M维，即需要量化的激活行数。
 * @param n GEMM的N维；不影响激活scratch，仅保持接口一致。
 * @param k GEMM的K维；这里通常是补齐后的输入特征数。
 * @return 成功时返回当前GPU对应的ActivationScratch；失败时返回nullptr。
 */
static ActivationScratch *GetActivationScratch(int m, int n, int k) {
    // N不改变A矩阵及其scale的存储尺寸，因此不参与容量计算。
    (void)n;

    // 激活scratch包含CUDA指针，必须按当前GPU分别保存和复用。
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;

    // 每个E2M1元素占4bit；scale空间按CUTLASS要求的M×K swizzle布局计算。
    const size_t activationBytes = (size_t)m * k / 2;
    const size_t scaleBytes = FastllmCudaNvfp4SwizzledScaleBytes(m, k);
    std::lock_guard<std::mutex> guard(cacheMutex);
    ActivationScratch &scratch = activationScratch[device];

    // 已有缓冲区容量足够时直接复用，避免每层、每token重复分配显存。
    if (scratch.activationBytes >= activationBytes && scratch.scaleBytes >= scaleBytes &&
        scratch.activation != nullptr && scratch.scales != nullptr) return &scratch;

    // CUDA Graph捕获期间不能进行动态显存分配或替换已有指针。
    if (FastllmCudaGraphIsCapturing()) return nullptr;

    // 先完整申请新缓冲区；任一申请失败时保留旧scratch仍然有效。
    uint8_t *newActivation = static_cast<uint8_t *>(FastllmCudaMalloc(activationBytes));
    uint8_t *newScales = static_cast<uint8_t *>(FastllmCudaMalloc(scaleBytes));
    if (newActivation == nullptr || newScales == nullptr) {
        if (newActivation != nullptr) FastllmCudaFree(newActivation);
        if (newScales != nullptr) FastllmCudaFree(newScales);
        return nullptr;
    }

    // 新空间全部申请成功后再释放旧空间，并更新指针和容量记录。
    if (scratch.activation != nullptr) FastllmCudaFree(scratch.activation);
    if (scratch.scales != nullptr) FastllmCudaFree(scratch.scales);
    scratch.activation = newActivation;
    scratch.scales = newScales;
    scratch.activationBytes = activationBytes;
    scratch.scaleBytes = scaleBytes;
    return &scratch;
}

static OutputScratch *GetOutputScratch(size_t bytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    OutputScratch &scratch = outputScratch[device];
    if (scratch.output != nullptr && scratch.bytes >= bytes) return &scratch;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    void *newOutput = FastllmCudaMalloc(bytes);
    if (newOutput == nullptr) return nullptr;
    if (scratch.output != nullptr) FastllmCudaFree(scratch.output);
    scratch.output = newOutput;
    scratch.bytes = bytes;
    return &scratch;
}

template <typename T>
__global__ void FinalizeOutputKernel(const T *source, T *destination,
                                     const float *bias, int rows, int columns,
                                     int sourceStride) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = (size_t)rows * columns;
    if (index >= elements) return;
    const int row = index / columns;
    const int column = index - (size_t)row * columns;
    float value;
    if constexpr (std::is_same<T, half>::value) {
        value = __half2float(source[(size_t)row * sourceStride + column]);
        if (bias != nullptr) value += bias[column];
        destination[index] = __float2half_rn(value);
    } else {
        value = __bfloat162float(source[(size_t)row * sourceStride + column]);
        if (bias != nullptr) value += bias[column];
        destination[index] = __float2bfloat16_rn(value);
    }
}

static bool FinalizeOutput(const void *source, void *destination,
                           const float *bias, fastllm::DataType dataType,
                           int rows, int columns, int sourceStride) {
    const size_t elements = (size_t)rows * columns;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    if (dataType == fastllm::DataType::FLOAT16) {
        FinalizeOutputKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const half *>(source), static_cast<half *>(destination),
            bias, rows, columns, sourceStride);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FinalizeOutputKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(source),
            static_cast<__nv_bfloat16 *>(destination), bias,
            rows, columns, sourceStride);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

/**
 * 查询NVFP4权重在指定GPU上的固定后端状态。
 *
 * MoE grouped路径复用Dense权重的状态表，把首个routed expert作为整层
 * 生命周期锚点；这样权重上传阶段的Prepared状态和首次真实GEMM验证可以
 * 连续衔接，无需再维护一份容易失效的MoE指针状态表。
 *
 * @param weight NVFP4_BLOCK_16逻辑权重。
 * @param device CUDA设备号。
 * @return 当前设备上的Uninitialized、Prepared、Cutlass或Rejected状态。
 */
FastllmCudaNvfp4BackendState FastllmCudaGetNvfp4W4A4BackendState(
        const fastllm::Data &weight, int device) {
    return GetBackendState(weight, device);
}

/**
 * 更新NVFP4权重在指定GPU上的固定后端状态。
 *
 * 该接口仅同步生命周期结论，不创建、释放或恢复设备权重；调用方必须先
 * 完成对应的cache提交或Legacy原始权重恢复，再写入最终状态。
 *
 * @param weight NVFP4_BLOCK_16逻辑权重。
 * @param device CUDA设备号。
 * @param state  已完成资源转换后需要记录的最终状态。
 */
void FastllmCudaSetNvfp4W4A4BackendState(
        const fastllm::Data &weight, int device,
        FastllmCudaNvfp4BackendState state) {
    SetBackendState(weight, device, state);
}

/**
 * 仅释放NVFP4权重在指定GPU上的CUTLASS重排cache。
 *
 * 本函数用于MoE首次验证失败时的两阶段回退：调用方先清空整层所有
 * routed expert cache，再统一恢复原始CUDA权重，避免逐个恢复时出现整层
 * 原始权重和整层CUTLASS权重同时驻留的峰值。
 *
 * @param weight 需要释放重排表示的逻辑权重。
 * @param device cache所属CUDA设备号。
 */
void FastllmCudaReleaseNvfp4W4A4CacheForDevice(
        const fastllm::Data *weight, int device) {
    ReleaseWeightCacheForDevice(weight, device);
}

bool FastllmCudaPrepareNvfp4W4A4Weight(
        fastllm::Data &weight, int inFeatures, int outFeatures,
        const uint8_t **packedWeight, const uint8_t **scales,
        const float **alpha) {
    if (packedWeight == nullptr || scales == nullptr || alpha == nullptr ||
        weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16 || inFeatures <= 0 || outFeatures <= 0 ||
        inFeatures % 32 != 0 || outFeatures % 32 != 0 ||
        weight.nvfp4GroupScales.size() !=
            (size_t)outFeatures * (inFeatures / 16) ||
        !std::isfinite(weight.nvfp4GlobalScale) ||
        weight.nvfp4GlobalScale <= 0.0f) return false;
    // 这里只预重排权重，没有真实激活M维；M不影响权重cache，传1即可。
    WeightCache *cache = GetWeightCache(weight, 1, outFeatures, inFeatures, true);
    if (cache == nullptr || cache->paddedColumns != inFeatures ||
        cache->paddedRows != outFeatures) return false;
    *packedWeight = cache->weight;
    *scales = cache->scales;
    *alpha = cache->alpha;
    return true;
}

/**
 * 在NVFP4权重进入当前GPU时预先选择并构建dense CUTLASS W4A4后端。
 *
 * 本函数只依赖权重自身和当前GPU能力，不依赖运行时M。满足条件时按
 * N/K完成padding、权重打包和scale重排，随后立即释放刚上传的原始CUDA
 * 权重，把状态记为Prepared。首个真实Linear仍会执行一次同步GEMM验证：
 * 成功后固定为Cutlass；失败则从host/mmap恢复原始权重并固定为Legacy。
 *
 * 该入口由Data::ToDevice逐个权重调用，因此任一时刻只需同时容纳当前层
 * 的原始格式和最终CUTLASS格式，不再等整模型上传后才逐层重排。
 *
 * @param weight 已上传到当前GPU的NVFP4_BLOCK_16二维权重[N,K]。
 * @return true表示已存在或成功创建CUTLASS格式；false表示固定走Legacy。
 */
bool FastllmCudaTryPrepackNvfp4W4A4Weight(fastllm::Data &weight) {
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16 || weight.dims.size() != 2) {
        return false;
    }
    const int device = CurrentDevice();
    if (device < 0) {
        if (StrictEnabled()) ThrowStrict("no current CUDA device");
        return false;
    }

    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Prepared || state == BackendState::CutlassW4A4) {
        return true;
    }
    if (state == BackendState::MarlinW4A16Fallback ||
        state == BackendState::NativeW4A16Fallback) {
        return false;
    }
    if (state == BackendState::Rejected) {
        if (StrictEnabled()) ThrowStrict("backend was already rejected");
        return false;
    }

    const int arch = RuntimeArch();
    const int n = weight.dims[0];
    const int k = weight.dims[1];
    const bool weightSupported = Enabled() && arch >= 100 && arch < 130 &&
        n > 0 && k > 0 && k % 16 == 0 &&
        weight.nvfp4GroupScales.size() == (size_t)n * (k / 16) &&
        std::isfinite(weight.nvfp4GlobalScale) &&
        weight.nvfp4GlobalScale > 0.0f && weight.cudaData != nullptr &&
        !weight.cudaDataBorrowed && !weight.multiDeviceData &&
        (weight.cpuData != nullptr || !weight.numasData.empty());
    if (!weightSupported) {
        SetBackendState(weight, device, BackendState::Rejected);
        Trace("fallback", "weight-load backend selection rejected", 1, k, n, arch);
        if (StrictEnabled()) ThrowStrict("weight-load backend selection rejected");
        return false;
    }

    // 这里没有真实激活，M传1；权重重排和缓存匹配只由N/K决定。
    WeightCache *cache = GetWeightCache(weight, 1, n, k, true);
    if (cache == nullptr) {
        ReleaseWeightCacheForDevice(&weight, device);
        SetBackendState(weight, device, BackendState::Rejected);
        Trace("fallback", "weight-load prepack failed", 1, k, n, arch);
        if (StrictEnabled()) ThrowStrict("weight-load prepack failed");
        return false;
    }
    SetBackendState(weight, device, BackendState::Prepared);
    Trace("w4a4-cutlass", "weight prepacked; first GEMM validation pending",
          1, k, n, arch);
    return true;
}

/**
 * 执行一次NVFP4 W4A4 CUTLASS线性计算。
 *
 * 本函数只负责单次计算，不决定后端的最终生命周期。执行流程为：检查
 * GPU与张量语义、取得CUTLASS重排权重、准备临时缓冲区、将FP16/BF16
 * 激活动态量化为NVFP4、调用对应架构的CUTLASS GEMM，最后处理padding
 * 和bias。首次后端选择时可强制同步，以便在释放原始权重前发现异步错误。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 *
 * @param input                FP16或BF16输入；普通路径形状为[m, k]，融合
 *                             路径形状为[m, 2 * k]。
 * @param weight               NVFP4_BLOCK_16权重，逻辑形状为[n, k]。
 * @param bias                 可选FP32偏置，长度为n。
 * @param output               输出张量，逻辑形状为[m, n]。
 * @param m                    GEMM的M维，激活行数，通常为token数。
 * @param n                    GEMM的N维，输出特征数。
 * @param k                    GEMM的K维，输入特征数。
 * @param siluMulInput         true表示把SwiGLU计算与激活量化融合。
 * @param checkBackendPolicy   true表示首次选择时检查CUTLASS后端开关；后端
 *                             选择不受M影响，固定后的正式计算传false。
 * @param validateWarmup       true表示末尾强制同步，验证首次warmup的异步错误。
 * @return true表示本次CUTLASS计算及必要的后处理成功；false表示某个阶段
 *         不满足条件或执行失败，具体阶段通过Trace记录。
 */
static bool RunCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int m, int n, int k, bool siluMulInput,
        bool checkBackendPolicy, bool validateWarmup) {
    // 先检查运行架构、数据类型、权重格式、scale元数据和维度语义。
    const int arch = RuntimeArch();
    const char *reason = "unsupported";
    const bool supportedArch = arch >= 100 && arch < 130;
    if (!supportedArch ||
        !SemanticsSupported(input, weight, bias, output, m, k, n,
                            checkBackendPolicy, &reason)) {
        Trace("fallback", !supportedArch ? "runtime SM is not 100-129" : reason,
              m, k, n, arch);
        return false;
    }
    if (input.dims.empty() ||
        input.dims.back() != (siluMulInput ? k * 2 : k)) {
        Trace("fallback", "activation shape mismatch", m, k, n, arch);
        return false;
    }

    // 获取或创建当前GPU专属的CUTLASS格式权重。此处暂不释放原始权重；
    // 是否释放由外层后端生命周期函数在warmup成功后决定。
    WeightCache *cache = GetWeightCache(weight, m, n, k, false);
    if (cache == nullptr) {
        Trace("fallback", "weight cache unavailable", m, k, n, arch);
        return false;
    }
    if (!TraceSynchronize("weight repack")) {
        Trace("fallback", "weight repack", m, k, n, arch);
        return false;
    }

    // 激活量化需要补齐后的A和scale缓冲区；输出特征被补齐时，GEMM先
    // 写入临时输出，随后再切回真实的n列。
    ActivationScratch *activation = GetActivationScratch(
        m, n, cache->paddedColumns);
    const bool paddedOutput = cache->paddedRows != n;
    OutputScratch *padded = paddedOutput
        ? GetOutputScratch((size_t)m * cache->paddedRows * sizeof(uint16_t)) : nullptr;
    if (activation == nullptr || (paddedOutput && padded == nullptr)) {
        Trace("fallback", "activation/output scratch unavailable", m, k, n, arch);
        return false;
    }

    // 接入FastLLM张量的设备指针，并为可选bias取得输入指针。
    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    void *biasData = bias.dims.empty() ? nullptr : FastllmCudaPrepareInput(bias);
    void *gemmOutput = paddedOutput ? padded->output : outputData;
    const char *failureStage = "prepare input/output";
    bool ok = inputData != nullptr && outputData != nullptr &&
              (bias.dims.empty() || biasData != nullptr);
    if (ok) {
        // 普通路径只做动态NVFP4量化；融合路径先计算SwiGLU，再直接生成
        // CUTLASS所需的NVFP4激活和E4M3 block scale。
        failureStage = siluMulInput ? "SwiGLU activation quantization" :
                                      "activation quantization";
        ok = siluMulInput
            ? FastllmCudaSiluMulNvfp4QuantizePadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  m, k, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4QuantizeActivationPadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  m, k, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread);
        if (ok) ok = TraceSynchronize(failureStage);
    }
    if (ok) {
        // SM100/SM120使用各自实例化的CUTLASS内核。GEMM直接消费NVFP4
        // 激活、NVFP4权重、两侧block scale以及权重global scale。
        failureStage = arch < 120 ? "SM100 CUTLASS GEMM" : "SM120 CUTLASS GEMM";
        ok = arch < 120
            ? FastllmCudaNvfp4CutlassGemmSm100(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  m, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4CutlassGemmSm120(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  m, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread);
    }
    if (ok && (paddedOutput || biasData != nullptr)) {
        // 去掉输出N维padding，并在需要时同时加上FP32 bias。
        failureStage = "output finalize";
        ok = FinalizeOutput(gemmOutput, outputData, static_cast<const float *>(biasData),
                            output.dataType, m, n, cache->paddedRows);
    }
    if (ok && validateWarmup) {
        // CUDA调用通常异步返回。首次warmup必须同步，避免内核实际失败后
        // 外层仍将CUTLASS标记为固定后端并释放原始CUDA权重。
        failureStage = "warmup validation";
        ok = TraceSynchronize(failureStage, true);
    }

    // 与PrepareInput/PrepareOutput成对收尾，并记录本次最终路径或失败阶段。
    if (biasData != nullptr) FastllmCudaFinishInput(bias, biasData);
    if (inputData != nullptr) FastllmCudaFinishInput(input, inputData);
    if (outputData != nullptr) FastllmCudaFinishOutput(output, outputData);
    Trace(ok ? (siluMulInput ? "w4a4-swiglu-cutlass" : "w4a4-cutlass") : "fallback",
          ok ? "success" : failureStage, m, k, n, arch);
    return ok;
}

/**
 * 尝试使用CUTLASS执行NVFP4 W4A4线性计算。
 *
 * 该函数同时负责当前权重在当前GPU上的后端生命周期管理：第一次调用
 * 会执行带同步验证的warmup；成功后固定使用CUTLASS并释放原始CUDA
 * 权重，失败后固定拒绝CUTLASS，由统一选择器继续选择兼容后端。后端一旦
 * 固定为CUTLASS，正式推理期间发生的执行错误将直接抛出，不再动态切换。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 *
 * @param input  FP16或BF16激活，逻辑形状为[m, k]。
 * @param weight NVFP4_BLOCK_16权重，逻辑形状为[n, k]；可能在warmup
 *               成功后释放其原始CUDA表示。
 * @param bias   可选的FP32偏置，长度为n。
 * @param output 输出张量，逻辑形状为[m, n]，类型与input相同。
 * @param m      GEMM的M维，激活行数，通常是本次参与计算的token数。
 * @param n      GEMM的N维，输出特征数。
 * @param k      GEMM的K维，输入特征数。
 * @return true表示本次已由CUTLASS完成；false表示CUTLASS未接管，调用方
 *         应使用已经选定的其他后端。
 */
bool TryCudaCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int m, int n, int k) {
    // 本入口只处理每16个权重共享一组缩放因子的NVFP4权重。
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16) return false;
    const int device = CurrentDevice();
    if (device < 0) {
        if (StrictEnabled()) ThrowStrict("no current CUDA device");
        return false;
    }

    // 后端状态按“权重对象 + GPU”保存。同一份权重换到另一张GPU后，
    // 需要在目标GPU上重新完成一次后端选择和权重重排。
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::MarlinW4A16Fallback ||
        state == BackendState::NativeW4A16Fallback) {
        return false;
    }
    if (state == BackendState::Rejected) {
        // 首次warmup已经确认CUTLASS不可用，交给统一选择器决定兼容后端，
        // 避免每次Linear都重复创建cache并再次尝试失败。
        Trace("fallback", "CUTLASS backend rejected during warmup", m, k, n,
              RuntimeArch());
        if (StrictEnabled()) ThrowStrict("backend rejected during warmup");
        return false;
    }

    // Uninitialized兼容未经过ToDevice预重排的直接调用；Prepared表示权重
    // 已在上传阶段重排并释放原始CUDA表示。两种状态都只验证一次真实GEMM。
    const bool warmup = state == BackendState::Uninitialized ||
                        state == BackendState::Prepared;
    const bool checkBackendPolicy = state == BackendState::Uninitialized;
    bool ok = RunCutlassNvfp4W4A4(input, weight, bias, output, m, n, k, false,
                                  checkBackendPolicy, warmup);
    if (ok && warmup) {
        // Uninitialized直达路径直到warmup成功才释放原始CUDA权重；Prepared
        // 路径已在ToDevice预重排阶段释放，不再重复调用释放接口。
        if (state == BackendState::Uninitialized &&
            !weight.ReleaseCudaDataForRepackedWeight()) {
            ReleaseWeightCacheForDevice(&weight, device);
            SetBackendState(weight, device, BackendState::Rejected);
            if (StrictEnabled()) ThrowStrict("original CUDA weight release failed");
            return false;
        }
        SetBackendState(weight, device, BackendState::CutlassW4A4);
        if (TraceEnabled()) {
            std::fprintf(stderr,
                "[fastllm][nvfp4] backend=cutlass state=fixed device=%d name=%s\n",
                device, weight.name.c_str());
        }
    } else if (!ok && warmup) {
        // 只允许在首次warmup阶段降级。若上传阶段已经释放原始CUDA权重，
        // 此处会销毁CUTLASS格式并从host/mmap恢复Legacy格式。
        if (!RejectCutlassAndRestoreLegacy(weight, device)) {
            throw std::runtime_error(
                "NVFP4 CUTLASS warmup failed and legacy weight restore failed");
        }
        if (StrictEnabled()) ThrowStrict("first GEMM validation failed");
    } else if (!ok) {
        // 后端固定后再失败不能静默切换，否则同一层会在推理过程中
        // 改变权重表示和计算路径；此时直接报错暴露真实故障。
        throw std::runtime_error(
            "NVFP4 CUTLASS backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

bool FastllmCudaCutlassNvfp4W4A4FromSwiglu(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    const int device = CurrentDevice();
    if (device < 0) {
        if (StrictEnabled()) ThrowStrict("no current CUDA device");
        return false;
    }
    const BackendState backend = GetBackendState(weight, device);
    const FusionState fusion = GetFusionState(weight, device);
    if (backend == BackendState::MarlinW4A16Fallback ||
        backend == BackendState::NativeW4A16Fallback) {
        return false;
    }
    if (backend == BackendState::Rejected) {
        if (StrictEnabled()) ThrowStrict("backend rejected before fused SwiGLU");
        return false;
    }
    // 融合能力与Linear后端分开；融合关闭时仍允许普通SwiGLU后接严格CUTLASS Linear。
    if (fusion == FusionState::Disabled) return false;

    const bool backendWarmup = backend == BackendState::Uninitialized ||
                               backend == BackendState::Prepared;
    const bool fusionWarmup = fusion == FusionState::Uninitialized;
    // 融合入口仍使用FastLLM旧维度命名，在这里映射为标准GEMM M/N/K。
    bool ok = RunCutlassNvfp4W4A4(input, weight, bias, output, n, k, m, true,
                                  backend == BackendState::Uninitialized,
                                  backendWarmup || fusionWarmup);
    if (ok) {
        if (backendWarmup) {
            // 与普通Linear一致：只有未经过ToDevice预重排的直达路径需要
            // 在首次融合GEMM成功后释放原始CUDA权重。
            if (backend == BackendState::Uninitialized &&
                !weight.ReleaseCudaDataForRepackedWeight()) {
                ReleaseWeightCacheForDevice(&weight, device);
                SetBackendState(weight, device, BackendState::Rejected);
                SetFusionState(weight, device, FusionState::Disabled);
                if (StrictEnabled()) ThrowStrict("original CUDA weight release failed");
                return false;
            }
            SetBackendState(weight, device, BackendState::CutlassW4A4);
        }
        if (fusionWarmup) SetFusionState(weight, device, FusionState::Enabled);
    } else if (fusionWarmup) {
        // 融合能力单独降级。保留Linear后端的未初始化状态，随后由普通
        // SwiGLU + Linear路径独立完成CUTLASS warmup和后端选择。
        SetFusionState(weight, device, FusionState::Disabled);
    } else {
        throw std::runtime_error(
            "NVFP4 fused SwiGLU backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

void FastllmCudaReleaseNvfp4W4A4Cache(const fastllm::Data *weight) {
    // Data析构、CopyFrom和ToDevice会统一调用缓存释放接口。只有
    // NVFP4_BLOCK_16权重可能成为W4A4 cache/state的键，普通激活、
    // position和logits必须在获取全局互斥锁前直接返回。
    if (weight == nullptr ||
        weight->dataType != fastllm::DataType::NVFP4_BLOCK_16) {
        return;
    }
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        bool hasCache = false;
        for (const auto &entry : weightCaches) {
            if (entry.first.first == weight) { hasCache = true; break; }
        }
        if (hasCache) {
            int originalDevice = 0;
            cudaGetDevice(&originalDevice);
            for (auto it = weightCaches.begin(); it != weightCaches.end();) {
                if (it->first.first == weight) {
                    cudaSetDevice(it->first.second);
                    ReleaseCache(it->second);
                    it = weightCaches.erase(it);
                } else {
                    ++it;
                }
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
        for (auto it = fusionStates.begin(); it != fusionStates.end();) {
            if (it->first.first == weight) it = fusionStates.erase(it);
            else ++it;
        }
    }
}
