// Apache-2.0. Native FastLLM adaptation of vLLM scaled_mm SM90 INT8/FP8 and
// SM120 FP8 kernels.  The public wrappers intentionally reject every layout
// except INT8/FP8 W8A8 activation quantization + symmetric per-output-channel
// or tensorwise weight scale.
#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fastllm_w8a8_dense_detail {
using namespace cute;

using BackendState = FastllmCudaFp8W8A8BackendState;

/**
 * Dense W8A8执行阶段按GPU复用的临时显存。
 *
 * INT8/FP8权重已经是CUTLASS直接消费的[N,K]格式，不保留
 * 第二份重排权重。本结构缓存量化激活、per-token scale、可选
 * zero-point和CUTLASS workspace；融合epilogue直接写最终输出，各缓冲区
 * 只扩容、不缩小。
 */
struct ExecutionScratch {
    void *quantized = nullptr;
    float *tokenScales = nullptr;
    int32_t *tokenZeroPoints = nullptr;
    void *workspace = nullptr;
    size_t quantizedBytes = 0;
    size_t tokenScaleBytes = 0;
    size_t tokenZeroPointBytes = 0;
    size_t workspaceBytes = 0;
};

/**
 * 缓存Dense INT8/FP8权重供CUTLASS直接消费的稳定设备参数。
 *
 * 权重本身已经是CUTLASS所需的[N,K]物理布局，因此不复制权重。
 * 缓存记录原权重指针、设备端scale、INT8 AZP所需的权重行和，并
 * 保存由FastLLM FP32 bias一次性转换出的FP16/BF16 bias。缓存按
 * 权重、bias、GPU和输出类型隔离，保证CUDA Graph重放期间地址稳定。
 */
struct WeightCache {
    const void *weight = nullptr;
    const float *scales = nullptr;
    int32_t *azpAdj = nullptr;
    void *bias = nullptr;
    const void *biasSource = nullptr;
    int scaleCount = 0;
    int n = 0;
    int32_t azpMultiplier = 1;
};

/**
 * 按Dense INT8/FP8权重和GPU保存CUTLASS使用的设备端scale。
 *
 * scale源保留在Data::scales主机数组中；权重迁移GPU时销毁旧设备副本，
 * 再从主机源直接在目标GPU创建，禁止复用Data::extraCudaData中的旧卡地址。
 */
struct ScaleCache {
    const void *weight = nullptr;
    float *scales = nullptr;
    int count = 0;
};

using WeightCacheKey = std::tuple<const fastllm::Data *, const fastllm::Data *,
                                  int, fastllm::DataType>;

std::mutex scratchMutex;
std::map<int, ExecutionScratch> executionScratch;
std::mutex backendMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;
std::mutex weightCacheMutex;
std::map<WeightCacheKey, WeightCache> weightCaches;
std::map<std::pair<const fastllm::Data *, int>, ScaleCache> scaleCaches;

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool StrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool ForceGemmFailureForTest() {
    const char *value = std::getenv(
        "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[noreturn]] static void ThrowFixedBackend(int arch, const char *reason) {
    throw std::runtime_error(
        "fixed SM" + std::to_string(arch) +
        " FP8 W8A8 CUTLASS backend failed: " + reason);
}

static void Trace(const char *path, const char *reason,
                  int m, int n, int k) {
    if (!TraceEnabled()) return;
    std::fprintf(stderr,
                 "[fastllm][w8a8] path=%s reason=%s m=%d n=%d k=%d sm=%d\n",
                 path, reason, m, n, k, FastllmCudaRuntimeArch());
}

static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(backendMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device,
                            BackendState state) {
    std::lock_guard<std::mutex> guard(backendMutex);
    backendStates[{&weight, device}] = state;
}

static void ReleaseScratch(ExecutionScratch &scratch) {
    if (scratch.quantized != nullptr) FastllmCudaFree(scratch.quantized);
    if (scratch.tokenScales != nullptr) FastllmCudaFree(scratch.tokenScales);
    if (scratch.tokenZeroPoints != nullptr) FastllmCudaFree(scratch.tokenZeroPoints);
    if (scratch.workspace != nullptr) FastllmCudaFree(scratch.workspace);
    scratch = ExecutionScratch();
}

/**
 * 获取或扩容当前GPU上的W8A8执行临时区。
 *
 * 首次GEMM会完成分配，后续正式推理直接复用稳定地址；CUDA Graph捕获
 * 期间禁止扩容。申请新空间失败时保留原临时区，避免破坏其他调用。
 *
 * @param quantizedBytes   INT8/FP8量化激活所需字节数。
 * @param tokenScaleBytes  per-token FP32 scale所需字节数。
 * @param tokenZeroPointBytes per-token INT32 zero-point所需字节数；
 *                            对称量化传0。
 * @return 容量足够时返回当前GPU临时区，否则返回nullptr。
 */
static ExecutionScratch *GetExecutionScratch(size_t quantizedBytes,
                                             size_t tokenScaleBytes,
                                             size_t tokenZeroPointBytes = 0) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;

    std::lock_guard<std::mutex> guard(scratchMutex);
    ExecutionScratch &scratch = executionScratch[device];

    if (scratch.quantized != nullptr && scratch.tokenScales != nullptr &&
        (tokenZeroPointBytes == 0 || scratch.tokenZeroPoints != nullptr) &&
        scratch.quantizedBytes >= quantizedBytes &&
        scratch.tokenScaleBytes >= tokenScaleBytes &&
        scratch.tokenZeroPointBytes >= tokenZeroPointBytes) {
        return &scratch;
    }

    if (FastllmCudaGraphIsCapturing()) return nullptr;

    ExecutionScratch replacement;
    replacement.quantized = FastllmCudaMalloc(quantizedBytes);
    replacement.tokenScales = static_cast<float *>(
        FastllmCudaMalloc(tokenScaleBytes));
    if (tokenZeroPointBytes != 0) {
        replacement.tokenZeroPoints = static_cast<int32_t *>(
            FastllmCudaMalloc(tokenZeroPointBytes));
    }

    if (replacement.quantized == nullptr || replacement.tokenScales == nullptr ||
        (tokenZeroPointBytes != 0 && replacement.tokenZeroPoints == nullptr)) {
        ReleaseScratch(replacement);
        return nullptr;
    }

    replacement.quantizedBytes = quantizedBytes;
    replacement.tokenScaleBytes = tokenScaleBytes;
    replacement.tokenZeroPointBytes = tokenZeroPointBytes;

    ReleaseScratch(scratch);
    scratch = replacement;

    return &scratch;
}

static void *GetWorkspace(ExecutionScratch &scratch, size_t bytes) {
    if (bytes == 0) return nullptr;

    if (scratch.workspace != nullptr && scratch.workspaceBytes >= bytes) {
        return scratch.workspace;
    }

    if (FastllmCudaGraphIsCapturing()) return nullptr;

    void *replacement = FastllmCudaMalloc(bytes);
    if (replacement == nullptr) return nullptr;

    if (scratch.workspace != nullptr) FastllmCudaFree(scratch.workspace);

    scratch.workspace = replacement;
    scratch.workspaceBytes = bytes;
    return scratch.workspace;
}

template <typename T> __device__ float ToFloat(T value);
template <> __device__ float ToFloat(half value) { return __half2float(value); }
template <> __device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename Quant>
__device__ __forceinline__ Quant QuantizeValue(float value, float scale) {
    float scaled = value / scale;
    if constexpr (std::is_same_v<Quant, int8_t>) {
        int q = max(-127, min(127, __float2int_rn(scaled)));
        return static_cast<int8_t>(q);
    } else {
        scaled = fmaxf(-448.0f, fminf(scaled, 448.0f));
        return Quant::bitcast(static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3)));
    }
}

struct QuantizeMaxOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return fmaxf(lhs, rhs);
    }
};

struct QuantizeMinOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return fminf(lhs, rhs);
    }
};

template <typename Input, typename Quant, int MaxValue>
__global__ void QuantizePerToken(const Input *input, Quant *quant,
                                 float *scales, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    static_assert(sizeof(Input) == 2 && sizeof(Quant) == 1);

    // 与vLLM dynamic_per_token_scaled_fp8_quant保持一致：每次处理16个
    // 元素。FP16/BF16侧读取32字节，FP8侧写入16字节。
    constexpr int ValuesPerVector = 16;
    const int vectors = cols / ValuesPerVector;

    struct __align__(32) InputVector {
        Input values[ValuesPerVector];
    };

    struct __align__(16) QuantVector {
        Quant values[ValuesPerVector];
    };

    const InputVector *vectorInput = reinterpret_cast<const InputVector *>(
        input + (size_t)row * cols);
    QuantVector *vectorOutput = reinterpret_cast<QuantVector *>(
        quant + (size_t)row * cols);

    float local = 0.0f;
    for (int vector = threadIdx.x; vector < vectors;
         vector += blockDim.x) {
        const InputVector values = vectorInput[vector];
#pragma unroll
        for (int i = 0; i < ValuesPerVector; ++i) {
            local = fmaxf(local, fabsf(ToFloat(values.values[i])));
        }
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage reductionStorage;
    float maximum = BlockReduce(reductionStorage).Reduce(
        local, QuantizeMaxOp{});

    __shared__ float rowScale;
    if (threadIdx.x == 0) {
        const float minScale = std::is_same_v<Quant, int8_t>
            ? 1.1920928955078125e-7f
            : 1.0f / ((float)MaxValue * 512.0f);
        rowScale = fmaxf(maximum / (float)MaxValue, minScale);
        scales[row] = rowScale;
    }

    __syncthreads();

    for (int vector = threadIdx.x; vector < vectors;
         vector += blockDim.x) {
        const InputVector source = vectorInput[vector];
        QuantVector target;
#pragma unroll
        for (int i = 0; i < ValuesPerVector; ++i) {
            target.values[i] = QuantizeValue<Quant>(
                ToFloat(source.values[i]), rowScale);
        }

        vectorOutput[vector] = target;
    }
}

/**
 * 将FP16/BF16激活逐token动态非对称量化为有符号INT8。
 *
 * 每行独立归约真实min/max，与vLLM一致地计算
 * scale=(max-min)/255和zeroPoint=round(-128-min/scale)，再写出
 * q=clamp(round(x/scale)+zeroPoint,-128,127)。修正项由CUTLASS epilogue处理。
 *
 * @param input       FP16或BF16输入，逻辑形状为[rows, cols]。
 * @param quant       INT8输出，逻辑形状为[rows, cols]。
 * @param scales      每行一个FP32激活scale，长度为rows。
 * @param zeroPoints  每行一个INT32激活zero-point，长度为rows。
 * @param rows        token行数。
 * @param cols        每个token的特征数，必须为16的倍数。
 */
template <typename Input>
__global__ void QuantizePerTokenAsymmetric(
        const Input *input, int8_t *quant, float *scales,
        int32_t *zeroPoints, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;
    for (int index = threadIdx.x; index < cols; index += blockDim.x) {
        const float value = ToFloat(input[(size_t)row * cols + index]);
        localMin = fminf(localMin, value);
        localMax = fmaxf(localMax, value);
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage reductionStorage;
    const float minimum = BlockReduce(reductionStorage).Reduce(
        localMin, QuantizeMinOp{});
    __syncthreads();
    const float maximum = BlockReduce(reductionStorage).Reduce(
        localMax, QuantizeMaxOp{});

    __shared__ float rowScale;
    __shared__ int32_t rowZeroPoint;
    if (threadIdx.x == 0) {
        rowScale = (maximum - minimum) / 255.0f;
        rowZeroPoint = __float2int_rn(-128.0f - minimum / rowScale);
        scales[row] = rowScale;
        zeroPoints[row] = rowZeroPoint;
    }
    __syncthreads();

    for (int index = threadIdx.x; index < cols; index += blockDim.x) {
        const float value = ToFloat(input[(size_t)row * cols + index]);
        const int q = max(-128, min(127,
            __float2int_rn(value / rowScale) + rowZeroPoint));
        quant[(size_t)row * cols + index] = static_cast<int8_t>(q);
    }
}

/**
 * 使用checkpoint给定的tensorwise参数量化INT8 W8A8激活。
 *
 * @param input      FP16或BF16输入，逻辑形状为[rows, cols]。
 * @param quant      INT8输出，逻辑形状为[rows, cols]。
 * @param scale      正的tensorwise FP32激活scale。
 * @param zeroPoint  tensorwise INT8 zero-point，以INT32传递。
 * @param elements   输入元素总数。
 */
template <typename Input>
__global__ void QuantizePerTensorStatic(
        const Input *input, int8_t *quant, float scale,
        int32_t zeroPoint, size_t elements) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)blockDim.x * gridDim.x) {
        const int q = max(-128, min(127,
            __float2int_rn(ToFloat(input[index]) / scale) + zeroPoint));
        quant[index] = static_cast<int8_t>(q);
    }
}

/**
 * 预计算每个INT8输出通道的K维和，供AZP epilogue复用。
 *
 * @param weight 对称INT8权重，逻辑形状为[rows, cols]。
 * @param sums   INT32行和输出，长度为rows。
 * @param rows   输出通道数。
 * @param cols   K维长度。
 * @param multiplier 动态AZP传1；静态AZP传checkpoint zero-point，
 *                   直接缓存zeroPoint*sumB。
 */
__global__ void SumInt8WeightRows(const int8_t *weight, int32_t *sums,
                                  int rows, int cols, int32_t multiplier) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    int local = 0;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local += (int)weight[(size_t)row * cols + col];
    }

    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage reductionStorage;
    const int sum = BlockReduce(reductionStorage).Sum(local);
    if (threadIdx.x == 0) sums[row] = sum * multiplier;
}

/**
 * 将静态tensorwise激活量化参数展开到按token复用的稳定缓冲区。
 *
 * @param scales      FP32 per-token scale输出，长度为rows。
 * @param zeroPoints  可选INT32 per-token zero-point输出，对称路径为nullptr。
 * @param rows        token数。
 * @param scale       checkpoint中的tensorwise scale。
 * @param zeroPoint   checkpoint中的tensorwise zero-point。
 */
__global__ void FillStaticQuantParams(float *scales, int32_t *zeroPoints,
                                      int rows, float scale,
                                      int32_t zeroPoint) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += blockDim.x * gridDim.x) {
        scales[row] = scale;
        if (zeroPoints != nullptr) zeroPoints[row] = zeroPoint;
    }
}

template <typename Output>
__global__ void ConvertBias(const float *source, Output *target, int count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        target[index] = Output(source[index]);
    }
}

static void ReleaseWeightCache(WeightCache &cache) {
    if (cache.bias != nullptr) FastllmCudaFree(cache.bias);
    if (cache.azpAdj != nullptr) FastllmCudaFree(cache.azpAdj);
    cache = WeightCache();
}

static void ReleaseScaleCache(ScaleCache &cache) {
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    cache = ScaleCache();
}

static ScaleCache *GetScaleCache(fastllm::Data &weight, int device) {
    const std::pair<const fastllm::Data *, int> key{&weight, device};
    auto found = scaleCaches.find(key);

    if (found != scaleCaches.end()) {
        ScaleCache &cache = found->second;

        if (cache.weight == weight.cudaData &&
            cache.count == static_cast<int>(weight.scales.size())) {
            return &cache;
        }

        if (FastllmCudaGraphIsCapturing()) return nullptr;

        ReleaseScaleCache(cache);
        scaleCaches.erase(found);
    }

    if (FastllmCudaGraphIsCapturing() || weight.scales.empty()) return nullptr;

    ScaleCache cache;
    cache.weight = weight.cudaData;
    cache.count = static_cast<int>(weight.scales.size());

    cache.scales = static_cast<float *>(
        FastllmCudaMalloc((size_t)cache.count * sizeof(float)));
    if (cache.scales == nullptr) return nullptr;

    FastllmCudaCopyFromHostToDevice(
        cache.scales, weight.scales.data(),
        (size_t)cache.count * sizeof(float));

    auto inserted = scaleCaches.emplace(key, cache);
    return &inserted.first->second;
}

static void ReleaseWeightCacheForDevice(const fastllm::Data *weight,
                                        int device) {
    std::lock_guard<std::mutex> guard(weightCacheMutex);
    int originalDevice = 0;
    cudaGetDevice(&originalDevice);
    cudaSetDevice(device);
    for (auto it = weightCaches.begin(); it != weightCaches.end();) {
        if (std::get<0>(it->first) == weight &&
            std::get<2>(it->first) == device) {
            ReleaseWeightCache(it->second);
            it = weightCaches.erase(it);
        } else {
            ++it;
        }
    }
    auto scale = scaleCaches.find({weight, device});
    if (scale != scaleCaches.end()) {
        ReleaseScaleCache(scale->second);
        scaleCaches.erase(scale);
    }
    cudaSetDevice(originalDevice);
}

/**
 * 取得Dense INT8/FP8权重对应的CUTLASS参数缓存。
 *
 * 权重和scale沿用模型上传后的设备表示，不创建第二份权重。FastLLM的
 * Linear bias固定为FP32，而vLLM SM120 scaled-mm要求bias与输出同类型；
 * 因此首次调用时将bias转换为FP16/BF16并缓存。CUDA Graph捕获期间不允许
 * 新建缓存，避免捕获到临时地址或内存分配操作。
 *
 * @tparam Output CUTLASS最终输出以及融合bias的数据类型。
 * @param weight  标准FP8权重，逻辑形状为[N,K]。
 * @param bias    可选FP32 bias，逻辑形状为[N]。
 * @param device  当前CUDA设备编号。
 * @param n       输出特征数。
 * @param k       输入特征数。
 * @param needAzpAdj true表示需要缓存每个INT8权重行的K维和。
 * @param azpMultiplier 动态AZP传1；静态AZP传激活zero-point。
 * @return 成功时返回地址稳定的参数缓存；失败时返回nullptr。
 */
template <typename Output>
static WeightCache *GetWeightCache(fastllm::Data &weight,
                                   const fastllm::Data &bias,
                                   int device, int n, int k,
                                   bool needAzpAdj = false,
                                   int32_t azpMultiplier = 1) {
    const fastllm::DataType outputType =
        std::is_same_v<Output, cutlass::half_t>
            ? fastllm::DataType::FLOAT16
            : fastllm::DataType::BFLOAT16;
    const WeightCacheKey key{&weight, &bias, device, outputType};

    std::lock_guard<std::mutex> guard(weightCacheMutex);
    auto found = weightCaches.find(key);

    if (found != weightCaches.end()) {
        WeightCache &cache = found->second;

        if (cache.weight == weight.cudaData && cache.n == n &&
            cache.biasSource == bias.cudaData &&
            (!needAzpAdj || (cache.azpAdj != nullptr &&
                             cache.azpMultiplier == azpMultiplier))) {
            return &cache;
        }

        if (FastllmCudaGraphIsCapturing()) return nullptr;

        ReleaseWeightCache(cache);
        weightCaches.erase(found);
    }

    if (FastllmCudaGraphIsCapturing()) return nullptr;

    ScaleCache *scaleCache = GetScaleCache(weight, device);
    if (scaleCache == nullptr || scaleCache->scales == nullptr) return nullptr;

    WeightCache cache;
    cache.weight = weight.cudaData;
    cache.scales = scaleCache->scales;
    cache.scaleCount = scaleCache->count;
    cache.n = n;
    cache.biasSource = bias.cudaData;
    cache.azpMultiplier = azpMultiplier;

    if (needAzpAdj) {
        cache.azpAdj = static_cast<int32_t *>(
            FastllmCudaMalloc((size_t)n * sizeof(int32_t)));
        if (cache.azpAdj == nullptr) {
            ReleaseWeightCache(cache);
            return nullptr;
        }

        SumInt8WeightRows<<<n, 256>>>(
            static_cast<const int8_t *>(weight.cudaData), cache.azpAdj,
            n, k, azpMultiplier);
        if (cudaGetLastError() != cudaSuccess) {
            ReleaseWeightCache(cache);
            return nullptr;
        }
    }

    if (!bias.dims.empty()) {
        cache.bias = FastllmCudaMalloc((size_t)n * sizeof(Output));
        if (cache.bias == nullptr) {
            ReleaseWeightCache(cache);
            return nullptr;
        }

        const int threads = 256;
        const int blocks = std::min(4096, (n + threads - 1) / threads);

        ConvertBias<<<blocks, threads>>>(
            static_cast<const float *>(bias.cudaData),
            static_cast<Output *>(cache.bias), n);

        if (cudaGetLastError() != cudaSuccess) {
            ReleaseWeightCache(cache);
            return nullptr;
        }
    }

    auto inserted = weightCaches.emplace(key, cache);
    return &inserted.first->second;
}

template <typename Kernel>
struct EnableSm90Only : Kernel {
    template <typename... Args> CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
        Kernel::operator()(std::forward<Args>(args)...);
#elif defined(__CUDA_ARCH__)
        asm("trap;");
#endif
    }
};

template <typename Kernel>
struct EnableSm120Only : Kernel {
    template <typename... Args> CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1210)
        Kernel::operator()(std::forward<Args>(args)...);
#elif defined(__CUDA_ARCH__)
        asm("trap;");
#endif
    }
};

/**
 * 定义与vLLM scaled-mm等价的CUTLASS融合缩放epilogue。
 *
 * 累加器先乘权重scale[N]或标量，再乘激活scale[M]或标量；有bias版本
 * 随后叠加与输出同类型的bias[N]，并直接转换为FP16/BF16输出。动态stride
 * 使同一内核同时支持per-token/per-channel和tensorwise广播语义。
 */
template <typename ElementAcc, typename ElementD, typename TileShape,
          bool HasBias, bool SwapAB = false>
struct FusedScaledEpilogue {
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ScaleAStride = Stride<bool, _0, _0>;
    using ScaleBStride = Stride<_0, bool, _0>;
    using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, float, float, ScaleAStride, 4, false>;
    using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, float, float, ScaleBStride, 4, false>;
    using Bias = std::conditional_t<SwapAB,
        cutlass::epilogue::fusion::Sm90ColBroadcast<
            0, TileShape, ElementD, ElementD, Stride<_1, _0, _0>,
            128 / sizeof_bits_v<ElementD>, false>,
        cutlass::epilogue::fusion::Sm90RowBroadcast<
            0, TileShape, ElementD, ElementD, Stride<_0, _1, _0>,
            128 / sizeof_bits_v<ElementD>, false>>;
    using MultiplyB = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledB = cutlass::epilogue::fusion::Sm90EVT<
        MultiplyB, ScaleB, Accum>;
    using FinalCompute = std::conditional_t<HasBias,
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::homogeneous_multiply_add, ElementD, float,
            cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::multiplies, ElementD, float,
            cutlass::FloatRoundStyle::round_to_nearest>>;
    using EVTCompute = std::conditional_t<HasBias,
        cutlass::epilogue::fusion::Sm90EVT<
            FinalCompute, ScaleA, ScaledB, Bias>,
        cutlass::epilogue::fusion::Sm90EVT<
            FinalCompute, ScaleA, ScaledB>>;
    using Arguments = typename EVTCompute::Arguments;

    static Arguments Prepare(const float *scaleA, bool perToken,
                             const float *scaleB, bool perChannel,
                             const ElementD *bias) {
        const float *kernelScaleA = SwapAB ? scaleB : scaleA;
        const float *kernelScaleB = SwapAB ? scaleA : scaleB;
        const bool kernelPerRow = SwapAB ? perChannel : perToken;
        const bool kernelPerColumn = SwapAB ? perToken : perChannel;

        typename ScaleA::Arguments aArgs{
            kernelScaleA, 0.0f,
            ScaleAStride{kernelPerRow, _0{}, _0{}}};
        typename ScaleB::Arguments bArgs{
            kernelScaleB, 0.0f,
            ScaleBStride{_0{}, kernelPerColumn, _0{}}};
        typename ScaledB::Arguments scaledBArgs{bArgs, {}, {}};
        if constexpr (HasBias) {
            typename Bias::Arguments biasArgs{bias, ElementD(0), {}};
            return Arguments{aArgs, scaledBArgs, biasArgs, {}};
        } else {
            return Arguments{aArgs, scaledBArgs, {}};
        }
    }
};

/**
 * 融合执行SM90 INT8激活zero-point修正、反量化与bias。
 *
 * 动态模式计算accum-zpA[M]*sumB[N]；静态模式接收已经乘过标量zpA的
 * azpWithAdj[N]。修正后的INT32累加值依次乘权重scale和激活scale，最后
 * 加可选bias并直接写FP16/BF16，避免物化[M,N]修正矩阵。
 */
template <typename ElementAcc, typename ElementD, typename TileShape,
          bool PerToken>
struct FusedScaledAzpEpilogue {
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ScaleAStride = Stride<bool, _0, _0>;
    using ScaleBStride = Stride<_0, bool, _0>;
    using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, float, float, ScaleAStride, 4, false>;
    using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, float, float, ScaleBStride, 4, false>;
    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, ElementD, ElementD, Stride<_0, _1, _0>,
        128 / sizeof_bits_v<ElementD>, true>;
    using AzpAdj = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, int32_t, int32_t, Stride<_0, _1, _0>, 4, false>;
    using Azp = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, int32_t, int32_t, Stride<_1, _0, _0>, 4, false>;

    using MultiplyAzp = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, int32_t, int32_t,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using AzpProduct = cutlass::epilogue::fusion::Sm90EVT<
        MultiplyAzp, Azp, AzpAdj>;
    using Subtract = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::minus, float, int32_t,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using CorrectedDynamic = cutlass::epilogue::fusion::Sm90EVT<
        Subtract, Accum, AzpProduct>;
    using CorrectedStatic = cutlass::epilogue::fusion::Sm90EVT<
        Subtract, Accum, AzpAdj>;
    using Corrected = std::conditional_t<PerToken,
        CorrectedDynamic, CorrectedStatic>;

    using MultiplyB = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledB = cutlass::epilogue::fusion::Sm90EVT<
        MultiplyB, ScaleB, Corrected>;
    using FinalCompute = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::homogeneous_multiply_add, ElementD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<
        FinalCompute, ScaleA, ScaledB, Bias>;
    using Arguments = typename EVTCompute::Arguments;

    static Arguments Prepare(const float *scaleA, bool perTokenScale,
                             const float *scaleB, bool perChannel,
                             const int32_t *azpAdj, const int32_t *azp,
                             const ElementD *bias) {
        typename ScaleA::Arguments aArgs{
            scaleA, 0.0f, ScaleAStride{perTokenScale, _0{}, _0{}}};
        typename ScaleB::Arguments bArgs{
            scaleB, 0.0f, ScaleBStride{_0{}, perChannel, _0{}}};
        typename Bias::Arguments biasArgs{bias, ElementD(0), {}};
        typename AzpAdj::Arguments adjArgs{azpAdj, int32_t(0), {}};

        if constexpr (PerToken) {
            typename Azp::Arguments azpArgs{azp, int32_t(0), {}};
            typename AzpProduct::Arguments productArgs{azpArgs, adjArgs, {}};
            typename CorrectedDynamic::Arguments correctedArgs{
                {}, productArgs, {}};
            typename ScaledB::Arguments scaledArgs{bArgs, correctedArgs, {}};
            return Arguments{aArgs, scaledArgs, biasArgs, {}};
        } else {
            typename CorrectedStatic::Arguments correctedArgs{{}, adjArgs, {}};
            typename ScaledB::Arguments scaledArgs{bArgs, correctedArgs, {}};
            return Arguments{aArgs, scaledArgs, biasArgs, {}};
        }
    }
};

template <
    typename Element,
    typename Output,
    typename Arch,
    typename Tile,
    typename Cluster,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    bool HasBias,
    typename EpilogueTile =
        cutlass::epilogue::collective::EpilogueTileAuto>
struct DenseKernel {
    using ElementA = Element;
    using ElementB = Element;
    using ElementD = Output;
    using ElementAccumulator = std::conditional_t<
        std::is_same_v<Element, int8_t>, int32_t, float>;

    static constexpr bool HasBiasValue = HasBias;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int Alignment =
        128 / cutlass::sizeof_bits<Element>::value;
    static constexpr int AlignmentD =
        128 / cutlass::sizeof_bits<Output>::value;

    using EpilogueOperation = FusedScaledEpilogue<
        ElementAccumulator, Output, Tile, HasBias>;
    using EVTCompute = typename EpilogueOperation::EVTCompute;

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch,
        cutlass::arch::OpClassTensorOp,
        Tile,
        Cluster,
        EpilogueTile,
        ElementAccumulator,
        float,
        void,
        LayoutD,
        AlignmentD,
        Output,
        LayoutD,
        AlignmentD,
        EpilogueSchedule,
        EVTCompute>::CollectiveOp;

    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        Alignment,
        Element,
        LayoutB,
        Alignment,
        ElementAccumulator,
        Tile,
        Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using Base = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue>;
    using EnabledKernel = std::conditional_t<
        std::is_same_v<Arch, cutlass::arch::Sm90>,
        EnableSm90Only<Base>,
        EnableSm120Only<Base>>;

    struct GemmKernel : EnabledKernel {};
};

template <typename Output, typename Tile, typename Cluster,
          typename MainloopSchedule, bool PerToken>
struct Sm90Int8AzpDenseKernel {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementD = Output;
    using ElementAccumulator = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int Alignment = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<Output>::value;

    using EpilogueOperation = FusedScaledAzpEpilogue<
        ElementAccumulator, Output, Tile, PerToken>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, float, void, LayoutD, AlignmentD,
        Output, LayoutD, AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized,
        typename EpilogueOperation::EVTCompute>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        int8_t, LayoutA, Alignment, int8_t, LayoutB, Alignment,
        ElementAccumulator, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;
    using Base = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue>;
    struct GemmKernel : EnableSm90Only<Base> {};
};

#if defined(FASTLLM_CUTLASS_W8A8_SM90)
/**
 * 定义与vLLM SM90标准FP8 scaled-mm一致的CUTLASS GEMM。
 *
 * SM90使用FP8 FastAccum mainloop和Persistent tile scheduler。SwapAB为true
 * 时实际计算D^T=B*A^T，以便小M形状把原N维映射到kernel的M维；输入布局、
 * 输出布局、scale广播和bias广播均随之转置，但对外数学语义保持不变。
 */
template <
    typename Element,
    typename Output,
    typename Tile,
    typename Cluster,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    bool HasBias,
    bool SwapAB>
struct Sm90Fp8DenseKernel {
    using ElementA = Element;
    using ElementB = Element;
    using ElementD = Output;
    using ElementAccumulator = float;

    static constexpr bool HasBiasValue = HasBias;
    static constexpr bool SwapABValue = SwapAB;

    using OriginalLayoutA = cutlass::layout::RowMajor;
    using OriginalLayoutB = cutlass::layout::ColumnMajor;
    using OriginalLayoutD = cutlass::layout::RowMajor;
    using TransposedLayoutA = typename cutlass::layout::LayoutTranspose<
        OriginalLayoutA>::type;
    using TransposedLayoutB = typename cutlass::layout::LayoutTranspose<
        OriginalLayoutB>::type;
    using TransposedLayoutD = typename cutlass::layout::LayoutTranspose<
        OriginalLayoutD>::type;
    using LayoutA = std::conditional_t<
        SwapAB, TransposedLayoutB, OriginalLayoutA>;
    using LayoutB = std::conditional_t<
        SwapAB, TransposedLayoutA, OriginalLayoutB>;
    using LayoutD = std::conditional_t<
        SwapAB, TransposedLayoutD, OriginalLayoutD>;

    static constexpr int Alignment =
        128 / cutlass::sizeof_bits<Element>::value;
    static constexpr int AlignmentD =
        128 / cutlass::sizeof_bits<Output>::value;

    using EpilogueOperation = FusedScaledEpilogue<
        ElementAccumulator, Output, Tile, HasBias, SwapAB>;
    using EVTCompute = typename EpilogueOperation::EVTCompute;

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Tile,
        Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        float,
        void,
        LayoutD,
        AlignmentD,
        Output,
        LayoutD,
        AlignmentD,
        EpilogueSchedule,
        EVTCompute>::CollectiveOp;

    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        Alignment,
        Element,
        LayoutB,
        Alignment,
        ElementAccumulator,
        Tile,
        Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using Base = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue,
        cutlass::gemm::PersistentScheduler>;

    struct GemmKernel : EnableSm90Only<Base> {};
};
#endif

/**
 * 按指定CUTLASS定义提交一次带融合缩放的W8A8 GEMM。
 *
 * 本函数负责把已经量化的激活和权重组装为CUTLASS参数、检查当前定义能否
 * 实现该形状、取得可复用workspace并向指定CUDA Stream提交GEMM。缩放和
 * 可选bias由epilogue完成；函数不负责激活动态量化、后端生命周期选择，
 * 也不等待异步计算结束。调用成功仅表示kernel已成功提交，首次后端固定
 * 所需的异步错误检查由外层同步完成。
 *
 * 数学语义为d[M,N] = a[M,K] * b[N,K]^T + bias[N]。scaleA按
 * perToken选择[M]或标量广播，scaleB按perChannel选择[N]或标量广播。
 *
 * @tparam Definition         CUTLASS kernel定义，包含元素类型、布局、tile、
 *                             调度策略以及融合epilogue。
 * @param a                   已量化激活，行主序逻辑形状为[m, k]。
 * @param b                   已量化权重，逻辑形状为[n, k]，计算时按转置使用。
 * @param d                   输出，行主序逻辑形状为[m, n]。
 * @param scaleA              激活FP32 scale；perToken为true时长度为m，否则
 *                             长度为1。
 * @param perToken            true表示激活scale按token广播；false表示全张量
 *                             共用一个scale。
 * @param scaleB              权重FP32 scale；perChannel为true时长度为n，
 *                             否则长度为1。
 * @param perChannel          true表示权重scale按输出通道广播；false表示
 *                             全张量共用一个scale。
 * @param bias                可选偏置，类型与输出一致、长度为n；无bias的
 *                             Definition应传nullptr。
 * @param m                   GEMM的M维，通常为本次处理的token数。
 * @param n                   GEMM的N维，即输出特征数。
 * @param k                   GEMM的K维，即输入特征数。
 * @param stream              提交CUTLASS kernel的CUDA Stream。
 * @param scratch             当前GPU可复用临时区，用于提供CUTLASS workspace。
 * @return true表示参数可实现、workspace准备成功且kernel已成功提交；false
 *         表示形状不受支持、workspace不足或提交阶段出现CUDA/CUTLASS错误。
 */
template <typename Definition>
bool Run(typename Definition::ElementA const *a,
         typename Definition::ElementB const *b,
         typename Definition::ElementD *d,
         const float *scaleA, bool perToken,
         const float *scaleB, bool perChannel,
         const typename Definition::ElementD *bias,
         int m, int n, int k, cudaStream_t stream,
         ExecutionScratch &scratch) {
    using Kernel = typename Definition::GemmKernel;
    using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    StrideA strideA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB strideB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideD strideD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

    typename Kernel::MainloopArguments mainloop{a, strideA, b, strideB};

    auto callbackArgs = Definition::EpilogueOperation::Prepare(
        scaleA, perToken, scaleB, perChannel, bias);
    typename Kernel::EpilogueArguments epilogue{
        callbackArgs, d, strideD, d, strideD};

    cutlass::KernelHardwareInfo hw;
    hw.device_id = FastllmCudaGetDevice();
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);

    typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    make_shape(m, n, k, 1), mainloop, epilogue, hw};

    Adapter gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t workspaceBytes = Adapter::get_workspace_size(args);
    void *workspace = GetWorkspace(scratch, workspaceBytes);
    if (workspaceBytes && workspace == nullptr) return false;

    cutlass::Status status = gemm.run(args, workspace, stream);
    return status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess;
}

/**
 * 构造AZP CUTLASS参数并提交一次SM90 INT8 GEMM。
 *
 * @return true表示形状可实现、workspace可用且kernel成功提交。
 */
template <typename Definition>
bool RunAzp(const int8_t *a, const int8_t *b,
            typename Definition::ElementD *d,
            const float *scaleA, bool perTokenScale,
            const float *scaleB, bool perChannel,
            const int32_t *azpAdj, const int32_t *azp,
            const typename Definition::ElementD *bias,
            int m, int n, int k, cudaStream_t stream,
            ExecutionScratch &scratch) {
    using Kernel = typename Definition::GemmKernel;
    using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    StrideA strideA = cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(m, k, 1));
    StrideB strideB = cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(n, k, 1));
    StrideD strideD = cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(m, n, 1));
    typename Kernel::MainloopArguments mainloop{a, strideA, b, strideB};
    auto callbackArgs = Definition::EpilogueOperation::Prepare(
        scaleA, perTokenScale, scaleB, perChannel,
        azpAdj, azp, bias);
    typename Kernel::EpilogueArguments epilogue{
        callbackArgs, d, strideD, d, strideD};

    cutlass::KernelHardwareInfo hw;
    hw.device_id = FastllmCudaGetDevice();
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw.device_id);
    typename Kernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        make_shape(m, n, k, 1), mainloop, epilogue, hw};

    Adapter gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    const size_t workspaceBytes = Adapter::get_workspace_size(args);
    void *workspace = GetWorkspace(scratch, workspaceBytes);
    if (workspaceBytes != 0 && workspace == nullptr) return false;
    const cutlass::Status status = gemm.run(args, workspace, stream);
    return status == cutlass::Status::kSuccess &&
           cudaGetLastError() == cudaSuccess;
}

/**
 * 按M/N边界选择与vLLM SM90 INT8相同的tile和warp调度。
 *
 * @return true表示选中的AZP kernel已成功提交。
 */
template <typename Output, bool PerToken>
bool DispatchAzp(const int8_t *a, const int8_t *b, Output *d,
                 const float *scaleA, bool perTokenScale,
                 const float *scaleB, bool perChannel,
                 const int32_t *azpAdj, const int32_t *azp,
                 const Output *bias, int m, int n, int k,
                 cudaStream_t stream, ExecutionScratch &scratch) {
    if (m <= 32 && n < 8192) {
        using D = Sm90Int8AzpDenseKernel<Output,
            Shape<_64,_64,_256>, Shape<_1,_8,_1>,
            cutlass::gemm::KernelTmaWarpSpecialized, PerToken>;
        return RunAzp<D>(a, b, d, scaleA, perTokenScale, scaleB,
            perChannel, azpAdj, azp, bias, m, n, k, stream, scratch);
    }
    if (m <= 32) {
        using D = Sm90Int8AzpDenseKernel<Output,
            Shape<_64,_128,_256>, Shape<_1,_4,_1>,
            cutlass::gemm::KernelTmaWarpSpecialized, PerToken>;
        return RunAzp<D>(a, b, d, scaleA, perTokenScale, scaleB,
            perChannel, azpAdj, azp, bias, m, n, k, stream, scratch);
    }
    if (m <= 64) {
        using D = Sm90Int8AzpDenseKernel<Output,
            Shape<_64,_64,_256>, Shape<_1,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecialized, PerToken>;
        return RunAzp<D>(a, b, d, scaleA, perTokenScale, scaleB,
            perChannel, azpAdj, azp, bias, m, n, k, stream, scratch);
    }
    if (m <= 128) {
        using D = Sm90Int8AzpDenseKernel<Output,
            Shape<_64,_128,_128>, Shape<_2,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecializedPingpong, PerToken>;
        return RunAzp<D>(a, b, d, scaleA, perTokenScale, scaleB,
            perChannel, azpAdj, azp, bias, m, n, k, stream, scratch);
    }
    using D = Sm90Int8AzpDenseKernel<Output,
        Shape<_128,_128,_128>, Shape<_2,_1,_1>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong, PerToken>;
    return RunAzp<D>(a, b, d, scaleA, perTokenScale, scaleB,
        perChannel, azpAdj, azp, bias, m, n, k, stream, scratch);
}

#if defined(FASTLLM_CUTLASS_W8A8_SM90)
/**
 * 提交一次SM90标准FP8 W8A8 GEMM，并处理可选的A/B转置计算。
 *
 * SwapAB=false时直接计算D[M,N]=A[M,K]*B[N,K]^T；SwapAB=true时构造
 * 等价问题D^T[N,M]=B[N,K]*A[M,K]^T。转置只改变CUTLASS看到的布局和
 * 参数顺序，不搬运输入、权重或输出数据。
 *
 * @return true表示CUTLASS接受参数且kernel成功提交；false表示形状、
 *         workspace或提交阶段失败。
 */
template <typename Definition>
bool RunSm90Fp8(typename Definition::ElementA const *a,
                typename Definition::ElementB const *b,
                typename Definition::ElementD *d,
                const float *scaleA, bool perToken,
                const float *scaleB, bool perChannel,
                const typename Definition::ElementD *bias,
                int m, int n, int k, cudaStream_t stream,
                ExecutionScratch &scratch) {
    using Kernel = typename Definition::GemmKernel;
    using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    const int kernelM = Definition::SwapABValue ? n : m;
    const int kernelN = Definition::SwapABValue ? m : n;

    StrideA strideA = cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(kernelM, k, 1));
    StrideB strideB = cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(kernelN, k, 1));
    StrideD strideD = cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(kernelM, kernelN, 1));

    const auto *kernelA = Definition::SwapABValue ? b : a;
    const auto *kernelB = Definition::SwapABValue ? a : b;
    typename Kernel::MainloopArguments mainloop{
        kernelA, strideA, kernelB, strideB};

    auto callbackArgs = Definition::EpilogueOperation::Prepare(
        scaleA, perToken, scaleB, perChannel, bias);
    typename Kernel::EpilogueArguments epilogue{
        callbackArgs, d, strideD, d, strideD};

    cutlass::KernelHardwareInfo hw;
    hw.device_id = FastllmCudaGetDevice();
    hw.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw.device_id);

    typename Kernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        make_shape(kernelM, kernelN, k, 1), mainloop, epilogue, hw};

    Adapter gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    const size_t workspaceBytes = Adapter::get_workspace_size(args);
    void *workspace = GetWorkspace(scratch, workspaceBytes);
    if (workspaceBytes != 0 && workspace == nullptr) return false;

    const cutlass::Status status = gemm.run(args, workspace, stream);
    return status == cutlass::Status::kSuccess &&
           cudaGetLastError() == cudaSuccess;
}
#endif

template <typename Element, typename Output, bool HasBias>
bool Dispatch(int arch, Element const *a, Element const *b, Output *d,
              const float *scaleA, bool perToken,
              const float *scaleB, bool perChannel,
              const Output *bias,
              int m, int n, int k, cudaStream_t stream,
              ExecutionScratch &scratch) {
#if defined(FASTLLM_CUTLASS_W8A8_SM90)
    if constexpr (std::is_same_v<Element, cutlass::float_e4m3_t>) {
        if (arch != 90) return false;

        if (m <= 16) {
            if (n <= 1280) {
                using D = Sm90Fp8DenseKernel<
                    Element, Output, Shape<_64,_16,_256>, Shape<_1,_2,_1>,
                    cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
                    cutlass::epilogue::TmaWarpSpecialized,
                    HasBias, true>;
                return RunSm90Fp8<D>(
                    a, b, d, scaleA, perToken, scaleB, perChannel,
                    bias, m, n, k, stream, scratch);
            }

            using D = Sm90Fp8DenseKernel<
                Element, Output, Shape<_64,_16,_256>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
                cutlass::epilogue::TmaWarpSpecialized,
                HasBias, true>;
            return RunSm90Fp8<D>(
                a, b, d, scaleA, perToken, scaleB, perChannel,
                bias, m, n, k, stream, scratch);
        }

        if (m <= 64) {
            if (n <= 1280) {
                using D = Sm90Fp8DenseKernel<
                    Element, Output, Shape<_64,_16,_256>, Shape<_1,_4,_1>,
                    cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
                    cutlass::epilogue::TmaWarpSpecialized,
                    HasBias, true>;
                return RunSm90Fp8<D>(
                    a, b, d, scaleA, perToken, scaleB, perChannel,
                    bias, m, n, k, stream, scratch);
            }

            using D = Sm90Fp8DenseKernel<
                Element, Output, Shape<_64,_64,_256>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
                cutlass::epilogue::TmaWarpSpecialized,
                HasBias, true>;
            return RunSm90Fp8<D>(
                a, b, d, scaleA, perToken, scaleB, perChannel,
                bias, m, n, k, stream, scratch);
        }

        if (m <= 128) {
            using D = Sm90Fp8DenseKernel<
                Element, Output, Shape<_64,_128,_128>, Shape<_2,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                cutlass::epilogue::TmaWarpSpecialized,
                HasBias, false>;
            return RunSm90Fp8<D>(
                a, b, d, scaleA, perToken, scaleB, perChannel,
                bias, m, n, k, stream, scratch);
        }

        if (m >= 8192 && k >= 6144) {
            using D = Sm90Fp8DenseKernel<
                Element, Output, Shape<_256,_128,_128>, Shape<_2,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                cutlass::epilogue::TmaWarpSpecializedCooperative,
                HasBias, false>;
            return RunSm90Fp8<D>(
                a, b, d, scaleA, perToken, scaleB, perChannel,
                bias, m, n, k, stream, scratch);
        }

        using D = Sm90Fp8DenseKernel<
            Element, Output, Shape<_128,_128,_128>, Shape<_2,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
            cutlass::epilogue::TmaWarpSpecialized,
            HasBias, false>;
        return RunSm90Fp8<D>(
            a, b, d, scaleA, perToken, scaleB, perChannel,
            bias, m, n, k, stream, scratch);
    }

    if constexpr (std::is_same_v<Element, int8_t>) {
        if (arch != 90) return false;
        if (m <= 32 && n < 8192) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_8,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized, HasBias>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm90,
                Shape<_64,_128,_256>, Shape<_1,_4,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized, HasBias>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        if (m <= 64) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized, HasBias>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        if (m <= 128) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm90,
                Shape<_64,_128,_128>, Shape<_2,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::TmaWarpSpecialized, HasBias>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        using D = DenseKernel<Element, Output, cutlass::arch::Sm90,
            Shape<_128,_128,_128>, Shape<_2,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecializedPingpong,
            cutlass::epilogue::TmaWarpSpecialized, HasBias>;
        return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                      bias, m, n, k, stream, scratch);
    }
#endif
#if defined(FASTLLM_CUTLASS_W8A8_SM120)
    if constexpr (std::is_same_v<Element, cutlass::float_e4m3_t>) {
        if (arch != 120) return false;
        if (m <= 16) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm120,
                Shape<_16,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                HasBias,
                Shape<_16,_32>>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm120,
                Shape<_32,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                HasBias,
                Shape<_32,_32>>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        if (m <= 256) {
            using D = DenseKernel<Element, Output, cutlass::arch::Sm120,
                Shape<_64,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                HasBias>;
            return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                          bias, m, n, k, stream, scratch);
        }
        using D = DenseKernel<Element, Output, cutlass::arch::Sm120,
            Shape<_128,_128,_128>, Shape<_1,_1,_1>,
            cutlass::gemm::collective::KernelScheduleAuto,
            cutlass::epilogue::collective::EpilogueScheduleAuto,
            HasBias>;
        return Run<D>(a, b, d, scaleA, perToken, scaleB, perChannel,
                      bias, m, n, k, stream, scratch);
    }
#endif
    return false;
}

/**
 * 执行一次动态量化W8A8线性计算。
 *
 * 本函数只负责单次计算，不决定固定后端的生命周期，也不主动同步CUDA
 * Stream。执行流程为：取得当前GPU可复用的临时区，准备权重的per-channel
 * scale与可选bias，将输入逐token动态量化为Quant，再调用对应架构的
 * CUTLASS GEMM。scale乘法、bias和FP16/BF16转换全部在CUTLASS epilogue
 * 内完成，不产生完整的FP32中间矩阵，也不启动独立缩放kernel。kernel
 * 提交后的异步错误由外层首次后端选择同步检查。
 *
 * 数学语义为output[M,N] = input[M,K] * weight[N,K]^T + bias[N]。
 * input和output为FP16或BF16；weight为CUTLASS直接消费的行主序量化权重；
 * 激活scale采用per-token布局[M]，权重scale采用per-channel布局[N]或
 * tensorwise标量布局[1]。
 *
 * @tparam Input       输入和最终输出对应的CUDA标量类型。
 * @tparam Quant       激活及权重参与GEMM时使用的量化标量类型。
 * @tparam Output      CUTLASS融合epilogue直接写出的FP16/BF16标量类型。
 * @tparam MaxValue    动态量化计算scale时采用的Quant最大有限值。
 * @param input        CUDA输入张量，逻辑形状为[m, k]。
 * @param weight       CUDA量化权重，逻辑形状为[n, k]；CUTLASS cache保存
 *                     当前GPU的scale和类型化bias，不复制权重数据。
 * @param bias         可选偏置，逻辑形状为[n]；为空时不执行偏置加法。
 * @param output       CUDA输出张量，逻辑形状为[m, n]。
 * @param m            GEMM的M维，通常为本次处理的token数。
 * @param k            GEMM的K维，即输入特征数。
 * @param n            GEMM的N维，即输出特征数。
 * @param arch         当前GPU计算能力，例如SM90传90、SM120传120。
 * @return true表示量化、GEMM和缩放写回均已成功提交；false表示临时区、
 *         scale/bias准备、CUTLASS分派或CUDA kernel启动失败。
 */
template <typename Input, typename Quant, typename Output, int MaxValue>
bool Execute(const fastllm::Data &input, fastllm::Data &weight,
             const fastllm::Data &bias, fastllm::Data &output,
             int m, int k, int n, int arch) {
    ExecutionScratch *scratch = GetExecutionScratch(
        (size_t)m * k * sizeof(Quant), (size_t)m * sizeof(float));

    const int device = FastllmCudaGetDevice();
    WeightCache *cache = GetWeightCache<Output>(weight, bias, device, n, k);

    if (scratch == nullptr || cache == nullptr || cache->weight == nullptr ||
        cache->scales == nullptr) return false;

    Quant *quant = static_cast<Quant *>(scratch->quantized);
    float *tokenScales = scratch->tokenScales;
    const float *channelScales = cache->scales;
    const Output *biasData = static_cast<const Output *>(cache->bias);

    cudaStream_t stream = 0;
    QuantizePerToken<Input, Quant, MaxValue><<<m, 256, 0, stream>>>(
        (const Input*)input.cudaData, quant, tokenScales, m, k);

    if (cudaGetLastError() != cudaSuccess) return false;

    Output *outputData = static_cast<Output *>(output.cudaData);
    const Quant *weightData = static_cast<const Quant *>(cache->weight);
    const bool perChannel = cache->scaleCount != 1;

    return biasData == nullptr
        ? Dispatch<Quant, Output, false>(
              arch, quant, weightData, outputData,
              tokenScales, true, channelScales, perChannel, nullptr,
              m, n, k, stream, *scratch)
        : Dispatch<Quant, Output, true>(
              arch, quant, weightData, outputData,
              tokenScales, true, channelScales, perChannel, biasData,
              m, n, k, stream, *scratch);
}

/**
 * 执行SM90 INT8 W8A8，并按模型元数据选择对称/非对称和动态/静态激活量化。
 *
 * 权重始终为对称per-channel INT8。非对称激活额外生成zero-point，并在
 * CUTLASS epilogue中用预缓存的权重行和完成修正。静态tensorwise参数会
 * 展开到可复用的per-token临时区，以保持所有kernel参数地址可被CUDA Graph
 * 稳定捕获。
 *
 * @return true表示激活量化、AZP参数准备和CUTLASS GEMM均成功。
 */
template <typename Input, typename Output>
bool ExecuteInt8(const fastllm::Data &input, fastllm::Data &weight,
                 const fastllm::Data &bias, fastllm::Data &output,
                 int m, int k, int n) {
    const bool asymmetric = !weight.w8a8InputSymmetric;
    const bool staticAzp = asymmetric && !weight.w8a8InputDynamic;
    ExecutionScratch *scratch = GetExecutionScratch(
        (size_t)m * k, (size_t)m * sizeof(float),
        asymmetric && weight.w8a8InputDynamic
            ? (size_t)m * sizeof(int32_t) : 0);
    const int device = FastllmCudaGetDevice();
    WeightCache *cache = GetWeightCache<Output>(
        weight, bias, device, n, k, asymmetric,
        staticAzp ? weight.w8a8InputZeroPoint : 1);
    if (scratch == nullptr || cache == nullptr || cache->weight == nullptr ||
        cache->scales == nullptr || (asymmetric && cache->azpAdj == nullptr)) {
        return false;
    }

    auto *quant = static_cast<int8_t *>(scratch->quantized);
    float *tokenScales = scratch->tokenScales;
    int32_t *tokenZeroPoints = scratch->tokenZeroPoints;
    cudaStream_t stream = 0;

    if (weight.w8a8InputDynamic) {
        if (asymmetric) {
            QuantizePerTokenAsymmetric<Input><<<m, 256, 0, stream>>>(
                static_cast<const Input *>(input.cudaData), quant,
                tokenScales, tokenZeroPoints, m, k);
        } else {
            QuantizePerToken<Input, int8_t, 127><<<m, 256, 0, stream>>>(
                static_cast<const Input *>(input.cudaData), quant,
                tokenScales, m, k);
        }
    } else {
        if (!(weight.w8a8InputScale > 0.0f) ||
            !std::isfinite(weight.w8a8InputScale)) return false;
        const size_t elements = (size_t)m * k;
        const int blocks = std::min<size_t>(4096, (elements + 255) / 256);
        QuantizePerTensorStatic<Input><<<blocks, 256, 0, stream>>>(
            static_cast<const Input *>(input.cudaData), quant,
            weight.w8a8InputScale, weight.w8a8InputZeroPoint, elements);
        const int paramBlocks = std::min(4096, (m + 255) / 256);
        FillStaticQuantParams<<<paramBlocks, 256, 0, stream>>>(
            tokenScales, nullptr, m,
            weight.w8a8InputScale, weight.w8a8InputZeroPoint);
    }
    if (cudaGetLastError() != cudaSuccess) return false;

    Output *outputData = static_cast<Output *>(output.cudaData);
    const auto *weightData = static_cast<const int8_t *>(cache->weight);
    const bool perChannel = cache->scaleCount != 1;
    const Output *biasData = static_cast<const Output *>(cache->bias);
    if (asymmetric) {
        const bool success = staticAzp
            ? DispatchAzp<Output, false>(
                  quant, weightData, outputData, tokenScales, true,
                  cache->scales, perChannel, cache->azpAdj,
                  nullptr, biasData, m, n, k, stream, *scratch)
            : DispatchAzp<Output, true>(
                  quant, weightData, outputData, tokenScales, true,
                  cache->scales, perChannel, cache->azpAdj,
                  tokenZeroPoints, biasData, m, n, k, stream, *scratch);
        Trace(success ? "w8a8-int8-azp-cutlass" : "fallback",
              success ? "success" : "INT8 AZP GEMM failed", m, n, k);
        return success;
    }
    const bool success = biasData == nullptr
        ? Dispatch<int8_t, Output, false>(
              90, quant, weightData, outputData, tokenScales, true,
              cache->scales, perChannel, nullptr,
              m, n, k, stream, *scratch)
        : Dispatch<int8_t, Output, true>(
              90, quant, weightData, outputData, tokenScales, true,
              cache->scales, perChannel, biasData,
              m, n, k, stream, *scratch);
    Trace(success ? "w8a8-int8-cutlass" : "fallback",
          success ? "success" : "INT8 GEMM failed", m, n, k);
    return success;
}
} // namespace fastllm_w8a8_dense_detail
#endif

/**
 * 清除指定Dense INT8/FP8权重记录的W8A8设备缓存。
 *
 * Dense权重不需要额外的CUTLASS重排副本，但设备scale、类型化bias、
 * INT8权重行和与FP8后端选择状态均按Data地址和GPU缓存。张量销毁、
 * 覆盖或迁移设备前必须删除对应记录，防止复用旧卡地址。
 *
 * @param weight 即将销毁、覆盖或迁移的权重；nullptr表示无需处理。
 */
void FastllmCudaReleaseFp8W8A8BackendState(const fastllm::Data *weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
    using namespace fastllm_w8a8_dense_detail;
    if (weight == nullptr) return;
    {
        std::lock_guard<std::mutex> guard(backendMutex);
        for (auto it = backendStates.begin(); it != backendStates.end();) {
            if (it->first.first == weight) {
                it = backendStates.erase(it);
            } else {
                ++it;
            }
        }
    }
    {
        std::lock_guard<std::mutex> guard(weightCacheMutex);
        int originalDevice = 0;
        cudaGetDevice(&originalDevice);
        for (auto it = weightCaches.begin(); it != weightCaches.end();) {
            if (std::get<0>(it->first) == weight) {
                cudaSetDevice(std::get<2>(it->first));
                ReleaseWeightCache(it->second);
                it = weightCaches.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = scaleCaches.begin(); it != scaleCaches.end();) {
            if (it->first.first == weight) {
                cudaSetDevice(it->first.second);
                ReleaseScaleCache(it->second);
                it = scaleCaches.erase(it);
            } else {
                ++it;
            }
        }
        cudaSetDevice(originalDevice);
    }
#else
    (void)weight;
#endif
    // Dense INT8/FP8和Blockwise FP8共用Data生命周期入口。
    FastllmCudaReleaseFp8W8A8Block128BackendState(weight);
}

FastllmCudaFp8W8A8BackendState FastllmCudaGetFp8W8A8BackendState(
        const fastllm::Data &weight, int device) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
    return fastllm_w8a8_dense_detail::GetBackendState(weight, device);
#else
    (void)weight;
    (void)device;
    return FastllmCudaFp8W8A8BackendState::Rejected;
#endif
}

/**
 * 在标准FP8权重上传当前GPU后预先记录Dense W8A8候选后端。
 *
 * 标准FP8权重已经是CUTLASS直接消费的[N,K]布局，因此本阶段不复制或
 * 重排权重，也不释放原始CUDA表示；这里只检查与M无关的GPU、权重形状
 * 和per-channel/tensorwise scale条件，预建稳定的设备scale缓存并记录
 * Prepared。首次真实GEMM负责补齐类型化bias缓存、分配执行scratch、同步
 * 验证内核，随后固定为Cutlass或Rejected。
 *
 * @param weight 已上传CUDA的标准FP8_E4M3二维线性权重[N,K]。
 * @return true表示权重已处于Prepared/Cutlass；false表示不是目标格式或
 *         已固定交给Legacy。
 */
bool FastllmCudaTryPrepareFp8W8A8Weight(fastllm::Data &weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && \
    (defined(FASTLLM_CUTLASS_W8A8_SM90) || \
     defined(FASTLLM_CUTLASS_W8A8_SM120))
    using namespace fastllm_w8a8_dense_detail;
    if (weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.weightType != fastllm::WeightType::LINEAR ||
        weight.dims.size() != 2) return false;
    const int device = FastllmCudaGetDevice();
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Prepared || state == BackendState::Cutlass) return true;
    if (state == BackendState::Rejected) return false;
    const int arch = FastllmCudaRuntimeArch();
    bool archSupported = false;
#if defined(FASTLLM_CUTLASS_W8A8_SM90)
    archSupported = archSupported || arch == 90;
#endif
#if defined(FASTLLM_CUTLASS_W8A8_SM120)
    archSupported = archSupported || arch == 120;
#endif

    const int n = weight.dims[0], k = weight.dims[1];
    const bool supported = Enabled() && archSupported &&
        n > 0 && k > 0 && n % 8 == 0 && k % 16 == 0 &&
        weight.blockK == 1 && weight.blockM == k &&
        (weight.scales.size() == 1 || weight.scales.size() == (size_t)n) &&
        weight.cudaData != nullptr;
    if (!supported) {
        SetBackendState(weight, device, BackendState::Rejected);
        Trace("fallback", "weight-load backend selection rejected", 1, n, k);
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict SM" + std::to_string(arch) +
                " FP8 W8A8 weight-load selection failed");
        }
        return false;
    }
    ScaleCache *scaleCache = nullptr;
    {
        std::lock_guard<std::mutex> guard(weightCacheMutex);
        scaleCache = GetScaleCache(weight, device);
    }
    if (scaleCache == nullptr || scaleCache->scales == nullptr) {
        SetBackendState(weight, device, BackendState::Rejected);
        Trace("fallback", "weight scale cache creation failed", 1, n, k);
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict SM" + std::to_string(arch) +
                " FP8 W8A8 scale cache creation failed");
        }
        return false;
    }
    SetBackendState(weight, device, BackendState::Prepared);
    Trace("w8a8-cutlass", "weight prepared; first GEMM validation pending",
          1, n, k);
    return true;
#else
    (void)weight;
    return false;
#endif
}

/**
 * 在SM90上执行Dense INT8 W8A8 CUTLASS线性计算。
 *
 * 权重固定为对称per-channel INT8。激活量化语义由权重携带的
 * compressed-tensors元数据决定，支持动态per-token和静态tensorwise的
 * 对称/非对称INT8；非对称路径在CUTLASS epilogue中融合zero-point修正。
 *
 * @param input  CUDA上的FP16或BF16激活，逻辑形状为[m, k]。
 * @param weight CUDA上的对称INT8权重，逻辑形状为[n, k]，带n个FP32 scale。
 * @param bias   可选CUDA FP32偏置，逻辑形状为[n]。
 * @param output CUDA上的FP16或BF16输出，逻辑形状为[m, n]。
 * @param m      token数。
 * @param k      输入特征数，必须按16对齐。
 * @param n      输出特征数，必须按8对齐。
 * @return true表示量化和CUTLASS GEMM已成功提交；false表示语义不支持
 *         或资源/kernel提交失败。
 */
bool FastllmCudaCutlassLinearInt8W8A8Sm90(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int m, int k, int n) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && defined(FASTLLM_CUTLASS_W8A8_SM90)
    using namespace fastllm_w8a8_dense_detail;
    if (!Enabled() || FastllmCudaRuntimeArch() != 90 ||
        m <= 0 || k <= 0 || n <= 0 ||
        k % 16 || n % 8 || input.cudaData == nullptr || weight.cudaData == nullptr ||
        output.cudaData == nullptr || weight.dataType != fastllm::DataType::INT8_W8A8 ||
        weight.dims != std::vector<int>({n, k}) || weight.perChannelAxis != 0 ||
        !weight.mins.empty() || !weight.zeros.empty() || weight.scales.size() != (size_t)n ||
        (weight.w8a8InputSymmetric && weight.w8a8InputZeroPoint != 0) ||
        (!weight.w8a8InputDynamic && (!(weight.w8a8InputScale > 0.0f) ||
                                     !std::isfinite(weight.w8a8InputScale))) ||
        (input.dataType != fastllm::DataType::FLOAT16 && input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType ||
        (!bias.dims.empty() && (bias.dataType != fastllm::DataType::FLOAT32 ||
                               bias.cudaData == nullptr || bias.Count(0) != n))) return false;
    if (input.dataType == fastllm::DataType::FLOAT16)
        return ExecuteInt8<half, cutlass::half_t>(
            input, weight, bias, output, m, k, n);
    return ExecuteInt8<__nv_bfloat16, cutlass::bfloat16_t>(
        input, weight, bias, output, m, k, n);
#else
    (void)input; (void)weight; (void)bias; (void)output; (void)m; (void)k; (void)n;
    return false;
#endif
}

/**
 * 在SM90或SM120上执行标准FP8 W8A8 CUTLASS线性计算并维护固定后端状态。
 *
 * 本入口只接管FP8_E4M3二维权重，以及tensorwise标量或per-channel
 * FP32权重scale；blockwise权重不属于本路径。输入必须为FP16或BF16，
 * 执行时先按token动态量化为FP8，再由CUTLASS完成GEMM、scale广播、
 * 可选bias和输出类型转换。数学语义为：
 * output[M,N] = input[M,K] * weight[N,K]^T + bias[N]。
 *
 * 后端按“权重对象、GPU”维护生命周期。首次真实GEMM会同步验证异步
 * CUDA错误，成功后固定为Cutlass，失败则固定为Rejected；固定为Cutlass
 * 后若运行条件改变或kernel失败会抛出异常，禁止同一权重静默切换到
 * Legacy。普通调用不额外同步CUDA Stream。
 *
 * @param input   CUDA上的FP16或BF16激活，逻辑形状为[m, k]。
 * @param weight  CUDA上的FP8_E4M3线性权重，逻辑形状为[n, k]；scale数量
 *                为1表示tensorwise，为n表示per-channel，并要求
 *                blockK=1、blockM=k以排除blockwise布局。
 * @param bias    可选CUDA FP32偏置，逻辑形状为[n]；空张量表示无偏置。
 * @param output  CUDA输出，逻辑形状为[m, n]，数据类型必须与input一致。
 * @param m       GEMM的M维，通常为本次处理的token数。
 * @param k       GEMM的K维，即输入特征数，必须按16对齐。
 * @param n       GEMM的N维，即输出特征数，必须按8对齐。
 * @return true表示本次CUTLASS计算已成功提交，并在首次选择时完成同步
 *         验证；false表示该权重不属于本路径，或首次选择失败且允许交给
 *         Legacy。严格模式及固定Cutlass后的失败通过异常报告。
 */
bool FastllmCudaCutlassLinearFp8W8A8(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int m, int k, int n) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && \
    (defined(FASTLLM_CUTLASS_W8A8_SM90) || \
     defined(FASTLLM_CUTLASS_W8A8_SM120))
    using namespace fastllm_w8a8_dense_detail;
    // 本入口只接管标准FP8权重及per-channel/tensorwise scale；blockwise
    // 等其他布局继续交给原有后端。
    if (weight.dataType != fastllm::DataType::FP8_E4M3) return false;

    const int device = FastllmCudaGetDevice();
    const int arch = FastllmCudaRuntimeArch();
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Rejected) {
        Trace("fallback", "backend fixed to legacy", m, n, k);
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict SM" + std::to_string(arch) +
                " FP8 W8A8 requires the CUTLASS backend");
        }
        return false;
    }

    const bool selecting = state == BackendState::Uninitialized ||
                           state == BackendState::Prepared;
    bool archSupported = false;
#if defined(FASTLLM_CUTLASS_W8A8_SM90)
    archSupported = archSupported || arch == 90;
#endif
#if defined(FASTLLM_CUTLASS_W8A8_SM120)
    archSupported = archSupported || arch == 120;
#endif

    const char *failure = nullptr;
    if (selecting && !Enabled()) failure = "backend disabled";
    else if (!archSupported) failure = "runtime SM is not enabled in this build";
    else if (m <= 0 || k <= 0 || n <= 0) failure = "invalid GEMM dimensions";
    else if (k % 16 != 0) failure = "K is not aligned to 16";
    else if (n % 8 != 0) failure = "N is not aligned to 8";
    else if (input.cudaData == nullptr || weight.cudaData == nullptr ||
             output.cudaData == nullptr) failure = "CUDA tensor pointer is null";
    else if (weight.dims != std::vector<int>({n, k})) failure = "weight shape mismatch";
    else if (weight.blockK != 1 || weight.blockM != k ||
             (weight.scales.size() != 1 &&
              weight.scales.size() != (size_t)n)) {
        failure = "weight scale layout mismatch";
    }
    else if (input.dataType != fastllm::DataType::FLOAT16 &&
             input.dataType != fastllm::DataType::BFLOAT16) failure = "input is not FP16/BF16";
    else if (output.dataType != input.dataType) failure = "output dtype differs from input";
    else if (!bias.dims.empty() &&
             (bias.dataType != fastllm::DataType::FLOAT32 ||
              bias.cudaData == nullptr || bias.Count(0) != n)) {
        failure = "bias must be empty or FP32[N] on CUDA";
    }

    if (failure != nullptr) {
        Trace("fallback", failure, m, n, k);
        if (selecting) {
            ReleaseWeightCacheForDevice(&weight, device);
            SetBackendState(weight, device, BackendState::Rejected);
            if (StrictEnabled()) {
                throw std::runtime_error(
                    "strict SM" + std::to_string(arch) +
                    " FP8 W8A8 selection failed: " + failure);
            }
            return false;
        }
        ThrowFixedBackend(arch, failure);
    }

    const bool ok = input.dataType == fastllm::DataType::FLOAT16
        ? Execute<half, cutlass::float_e4m3_t, cutlass::half_t, 448>(
              input, weight, bias, output, m, k, n, arch)
        : Execute<__nv_bfloat16, cutlass::float_e4m3_t,
                  cutlass::bfloat16_t, 448>(
              input, weight, bias, output, m, k, n, arch);

    // CUDA调用异步返回。后端只在首次真实GEMM同步验证成功后固定，避免
    // 把异步内核错误误判为成功并在后续请求中静默混用Legacy。
    cudaError_t validation = selecting && ok
        ? cudaStreamSynchronize(0) : (ok ? cudaSuccess : cudaErrorUnknown);
    const bool forcedFailure = ok && validation == cudaSuccess &&
                               ForceGemmFailureForTest();
    if (forcedFailure && !selecting) {
        // 测试固定后端运行失败时先排空刚提交的合法kernel，避免故障注入
        // 留下异步工作影响后续case；生产环境不设置该测试开关。
        validation = cudaStreamSynchronize(0);
    }
    if (!ok || validation != cudaSuccess || forcedFailure) {
        const char *reason = !ok ? "CUTLASS launch failed" :
            (validation != cudaSuccess ? cudaGetErrorString(validation) :
             "injected CUTLASS GEMM failure");
        Trace("fallback", reason, m, n, k);
        if (selecting) {
            ReleaseWeightCacheForDevice(&weight, device);
            SetBackendState(weight, device, BackendState::Rejected);
            if (StrictEnabled()) {
                throw std::runtime_error(
                    "strict SM" + std::to_string(arch) +
                    " FP8 W8A8 warmup failed: " + reason);
            }
            return false;
        }
        ThrowFixedBackend(arch, reason);
    }

    if (selecting) SetBackendState(weight, device, BackendState::Cutlass);
    Trace("w8a8-cutlass", selecting ? "backend fixed" : "success", m, n, k);
    return true;
#else
    (void)input; (void)weight; (void)bias; (void)output; (void)m; (void)k; (void)n;
    return false;
#endif
}

/**
 * 保留旧SM120入口的兼容包装，不参与SM90生产分派。
 *
 * @return 运行时为SM120时转发到通用Dense FP8 W8A8入口，否则返回false。
 */
bool FastllmCudaCutlassLinearFp8W8A8Sm120(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int m, int k, int n) {
    if (FastllmCudaRuntimeArch() != 120) return false;

    return FastllmCudaCutlassLinearFp8W8A8(
        input, weight, bias, output, m, k, n);
}
