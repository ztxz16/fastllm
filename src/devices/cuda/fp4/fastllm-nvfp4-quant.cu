/* Apache-2.0: kernels adapted from vLLM nvfp4_quant_kernels.cu and
 * activation_nvfp4_quant_fusion_kernels.cu. */
#include "fastllm-cuda.cuh"
#define NVFP4_ENABLE_ELTS16 1
#include "fastllm-nvfp4-utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

// 与vLLM量化kernel一致：block最多使用512线程，并通过launch bounds限制
// 单block寄存器占用。SM101/110/120/121每个SM最多1536线程，
// SM100/103按2048线程计算；每个SM最多请求4个block。
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1010 || __CUDA_ARCH__ == 1100 || \
     __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
#define FASTLLM_NVFP4_MAX_THREADS_PER_SM 1536
#else
#define FASTLLM_NVFP4_MAX_THREADS_PER_SM 2048
#endif
#define FASTLLM_NVFP4_BLOCKS_PER_SM \
    (((FASTLLM_NVFP4_MAX_THREADS_PER_SM / 512) < 4) \
         ? (FASTLLM_NVFP4_MAX_THREADS_PER_SM / 512) : 4)

static int Nvfp4BlocksPerSm(int blockThreads) {
    int device = 0;
    int maxThreadsPerSm = FASTLLM_NVFP4_MAX_THREADS_PER_SM;
    if (cudaGetDevice(&device) == cudaSuccess) {
        cudaDeviceGetAttribute(&maxThreadsPerSm,
                               cudaDevAttrMaxThreadsPerMultiProcessor, device);
    }
    return std::min(4, std::max(1, maxThreadsPerSm / std::max(1, blockThreads)));
}

static int Nvfp4MultiprocessorCount() {
    int device = 0, count = 1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    }
    return std::max(1, count);
}

/**
 * 保存SwiGLU+NVFP4第三版热路径所需的单GPU启动属性。
 *
 * 属性由当前CPU线程首次访问某块GPU时从CUDA Runtime读取。后续调用直接
 * 复用缓存，避免为每次融合量化重复查询计算能力、SM数量和每SM线程上限。
 * 缓存只影响host侧启动参数，不保存张量或CUDA显存。
 */
struct Nvfp4CachedLaunchProperties {
    int device = -1;
    int arch = 0;
    int multiprocessorCount = 1;
    int maxThreadsPerSm = FASTLLM_NVFP4_MAX_THREADS_PER_SM;
};

/**
 * 取得当前GPU的SwiGLU+NVFP4启动属性，并在CPU线程内缓存查询结果。
 *
 * 同一推理线程通常固定使用一块GPU，因此命中缓存后只需确认当前device，
 * 不再重复执行cudaDeviceGetAttribute。线程切换GPU时重新读取属性，避免把
 * 一块GPU的grid配置错误用于另一块GPU；CUDA查询失败时不更新缓存。
 *
 * @param properties 返回当前GPU编号、计算能力、SM数量和每SM线程上限。
 * @return 属性读取成功且GPU计算能力位于SM100至SM129时返回true；否则
 *         返回false，调用方不得启动NVFP4融合kernel。
 */
static bool GetCachedNvfp4LaunchProperties(
        Nvfp4CachedLaunchProperties &properties) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    static thread_local Nvfp4CachedLaunchProperties cached;
    if (cached.device != device) {
        // GPU发生变化时一次性读取所有启动属性；只有全部成功才发布缓存。
        int major = 0, minor = 0, multiprocessorCount = 1;
        int maxThreadsPerSm = FASTLLM_NVFP4_MAX_THREADS_PER_SM;
        if (cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(
                &multiprocessorCount, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            cudaDeviceGetAttribute(
                &maxThreadsPerSm,
                cudaDevAttrMaxThreadsPerMultiProcessor, device) != cudaSuccess) {
            return false;
        }
        cached.device = device;
        cached.arch = major * 10 + minor;
        cached.multiprocessorCount = std::max(1, multiprocessorCount);
        cached.maxThreadsPerSm = std::max(1, maxThreadsPerSm);
    }
    properties = cached;
    return properties.arch >= 100 && properties.arch < 130;
}

template <typename T>
__global__ void __launch_bounds__(512, FASTLLM_NVFP4_BLOCKS_PER_SM)
FastllmNvfp4QuantKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int columns,
        int paddedColumns,
        int roundedRows, int kTiles, float globalScale) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = paddedColumns / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> values;
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        const bool valid = row < rows && element < columns;
        const T *source = valid ? input + (size_t)row * columns + element : input;
        fastllm_nvfp4::LoadOrZero(values, source, valid);
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            row, pack, kTiles, outputScales);
        const auto packed = fastllm_nvfp4::Quantize(values, globalScale, scale);
        if (row < rows) {
            fastllm_nvfp4::StoreFp4(
                output + (size_t)row * (paddedColumns / 2) + element / 2, packed);
        }
    }
}

/**
 * 融合执行SwiGLU并将结果动态量化为CUTLASS使用的NVFP4激活。
 *
 * 每个线程处理一组连续hidden元素：分别读取gate和up，计算
 * SiLU(gate) * up，再使用globalScale生成E2M1打包值和对应的E4M3
 * 局部scale。kernel只遍历真实rows；paddedHidden仅决定输出行跨度和
 * scale布局，越过真实hidden的列写入零值，以满足后续CUTLASS GEMM布局。
 *
 * @tparam T          输入元素类型，仅支持half或__nv_bfloat16。
 * @param input       输入激活，逻辑形状为[rows, 2 * hidden]；每行前半部分
 *                    为gate，后半部分为up。
 * @param output      NVFP4 E2M1打包输出，逻辑形状为[rows, paddedHidden]，
 *                    每两个FP4元素占一个字节。
 * @param outputScales E4M3局部scale输出，使用CUTLASS要求的swizzle布局。
 * @param rows        真实激活行数，通常为token数。
 * @param hidden      SwiGLU输出宽度，即每行gate或up的元素数。
 * @param paddedHidden 对齐后的输出宽度；不得小于hidden。
 * @param kTiles      paddedHidden对应的64元素tile数量，用于计算scale地址。
 * @param globalScale FP32全局scale；与局部scale共同还原量化值。
 */
template <typename T>
__global__ void __launch_bounds__(512, FASTLLM_NVFP4_BLOCKS_PER_SM)
FastllmSiluMulNvfp4QuantOptimizedKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int hidden,
        int paddedHidden, int kTiles, float globalScale) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    // scale缓冲区仍按128行分块分配，但融合量化只处理真实输入行。
    // CUTLASS GEMM的逻辑M就是rows，不会读取M维补齐区域。
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> gate, up;
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        const bool valid = row < rows && element < hidden;
        const T *rowInput = valid ? input + (size_t)row * hidden * 2 : input;
        fastllm_nvfp4::LoadOrZero(gate, rowInput + element, valid);
        fastllm_nvfp4::LoadOrZero(up, rowInput + hidden + element, valid);
        auto activated = fastllm_nvfp4::SiluMul(gate, up);
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            row, pack, kTiles, outputScales);
        const auto packed = fastllm_nvfp4::Quantize(activated, globalScale, scale);
        if (row < rows) {
            fastllm_nvfp4::StoreFp4(
                output + (size_t)row * (paddedHidden / 2) + element / 2, packed);
        }
    }
}

/**
 * 调度优化前的融合kernel，仅供算子横向性能对比。
 *
 * 该版本固定最多256线程，并把M补齐到128后执行；生产路径不调用它。
 */
template <typename T>
__global__ void FastllmSiluMulNvfp4QuantBaselineKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int hidden,
        int paddedHidden, int roundedRows, int kTiles, float globalScale) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> gate, up;
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        const bool valid = row < rows && element < hidden;
        const T *rowInput = valid ? input + (size_t)row * hidden * 2 : input;
        fastllm_nvfp4::LoadOrZero(gate, rowInput + element, valid);
        fastllm_nvfp4::LoadOrZero(up, rowInput + hidden + element, valid);
        auto activated = fastllm_nvfp4::SiluMul(gate, up);
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            row, pack, kTiles, outputScales);
        const auto packed = fastllm_nvfp4::Quantize(activated, globalScale, scale);
        if (row < rows) {
            fastllm_nvfp4::StoreFp4(
                output + (size_t)row * (paddedHidden / 2) + element / 2, packed);
        }
    }
}

// 将 FastLLM 的 FP4 权重和 checkpoint 原始 E4M3 group scale
// 重排为 CUTLASS 块缩放 GEMM 要求的两个布局。
// 这里只搬运并改变布局，不对 scale 做解码或二次量化。
__global__ void FastllmNvfp4Block16ToCutlassKernel(
        const uint8_t *__restrict__ source,
        const uint8_t *__restrict__ groupScales,
        uint8_t *__restrict__ packedWeight,
        uint8_t *__restrict__ weightScales, int rows, int columns,
        int paddedRows, int paddedColumns,
        int sourceBytesPerRow, int kTiles) {
    // K 维每相邻 16 个权重共享一个缩放因子。
    const int group = blockIdx.y * blockDim.x + threadIdx.x;
    const int groups = paddedColumns / 16;
    if (group >= groups) return;
    // CUTLASS 缩放因子布局要求行维度补齐到 128；遍历完整分块，
    // 同时将补齐权重矩阵之外的缩放因子初始化为零。
    for (int row = blockIdx.x; row < ((paddedRows + 127) / 128) * 128; row += gridDim.x) {
        uint64_t packed = 0;
        uint8_t scaleByte = 0;
        if (row < rows && group < columns / 16) {
            const uint8_t *block = source + (size_t)row * sourceBytesPerRow
                                 + (size_t)group * (8 + sizeof(float));
            // 旧 fallback 仍使用 12 字节 block；此处只读其中的 8 字节 FP4。
            // 相邻 block 间隔 12 字节，因此用两次 32-bit 读取避免非对齐 uint64_t 访问。
            const uint32_t packedLo = *reinterpret_cast<const uint32_t *>(block);
            const uint32_t packedHi = *reinterpret_cast<const uint32_t *>(block + 4);
            packed = uint64_t(packedLo) | (uint64_t(packedHi) << 32);
            // 原始 E4M3 字节直接进入 CUTLASS swizzled scale 布局。
            scaleByte = groupScales[(size_t)row * (columns / 16) + group];
        }
        // 补齐区域使用零 FP4 权重和零缩放因子。
        if (row < paddedRows) {
            reinterpret_cast<uint64_t *>(
                packedWeight + (size_t)row * (paddedColumns / 2))[group] = packed;
        }
        *fastllm_nvfp4::SwizzledScaleAddress(row, group, kTiles, weightScales) = scaleByte;
    }
}

static bool FastllmNvfp4RuntimeSupported() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return false;
    }
    const int arch = major * 10 + minor;
    return arch >= 100 && arch < 130;
}

template <typename T>
static bool LaunchQuant(const void *input, uint8_t *output, uint8_t *scales,
                        int rows, int columns, int paddedColumns, float globalScale,
                        cudaStream_t stream) {
    const int packs = paddedColumns / fastllm_nvfp4::kElementsPerThread;
    const int realPacks = std::max(1, columns / fastllm_nvfp4::kElementsPerThread);
    const int threads = std::min(512, realPacks);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (paddedColumns + 63) / 64;
    const int gridY = (packs + threads - 1) / threads;
    const int gridX = std::min(
        roundedRows,
        std::max(1, Nvfp4MultiprocessorCount() * Nvfp4BlocksPerSm(threads) / gridY));
    dim3 block(threads);
    dim3 grid(gridX, gridY);
    FastllmNvfp4QuantKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, columns,
        paddedColumns, roundedRows, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

/**
 * 使用缓存的GPU属性启动普通激活NVFP4动态量化。
 *
 * MoE会按活跃expert多次调用普通激活量化。该热路径直接复用当前GPU的
 * SM数量、每SM线程上限和计算能力，避免每个expert重复查询CUDA设备属性。
 * kernel保持异步执行；稳定热路径不插入设备或流同步。
 *
 * @tparam T             输入元素类型，仅支持half或__nv_bfloat16。
 * @param input          输入激活，逻辑形状为[rows, columns]。
 * @param output         打包E2M1输出，每两个FP4元素占一个字节。
 * @param scales         CUTLASS swizzle布局的E4M3局部scale输出。
 * @param rows           本次量化的真实行数。
 * @param columns        输入的真实列数，必须是16的倍数。
 * @param paddedColumns  输出补齐列数，必须是32的倍数且不小于columns。
 * @param globalScale    激活动态量化使用的正数FP32全局scale。
 * @param properties     当前GPU已缓存的启动属性。
 * @param stream         异步启动kernel所使用的CUDA流。
 * @return kernel启动参数有效时返回true；异步执行错误由后续同步边界取得。
 */
template <typename T>
static bool LaunchQuantCached(
        const void *input, uint8_t *output, uint8_t *scales,
        int rows, int columns, int paddedColumns, float globalScale,
        const Nvfp4CachedLaunchProperties &properties,
        cudaStream_t stream) {
    const int packs = paddedColumns / fastllm_nvfp4::kElementsPerThread;
    const int realPacks = std::max(
        1, columns / fastllm_nvfp4::kElementsPerThread);
    const int threads = std::min(512, realPacks);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (paddedColumns + 63) / 64;
    const int gridY = (packs + threads - 1) / threads;
    const int blocksPerSm = std::min(
        4, std::max(1, properties.maxThreadsPerSm / std::max(1, threads)));
    const int gridX = std::min(
        roundedRows,
        std::max(1, properties.multiprocessorCount * blocksPerSm / gridY));
    FastllmNvfp4QuantKernel<T><<<dim3(gridX, gridY), dim3(threads), 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, columns,
        paddedColumns, roundedRows, kTiles, globalScale);

    // 每块GPU只检查首次launch的即时配置错误；该检查不会同步GPU。
    static thread_local int validatedDevice = -1;
    if (validatedDevice != properties.device) {
        if (cudaGetLastError() != cudaSuccess) return false;
        validatedDevice = properties.device;
    }
    return true;
}

template <typename T>
static bool LaunchSiluMulQuantOptimized(
        const void *input, uint8_t *output, uint8_t *scales,
        int rows, int hidden, int paddedHidden, float globalScale,
        cudaStream_t stream) {
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    const int realPacks = std::max(1, hidden / fastllm_nvfp4::kElementsPerThread);
    const int threads = std::min(512, realPacks);
    const int kTiles = (paddedHidden + 63) / 64;
    const int gridY = (packs + threads - 1) / threads;
    const int gridX = std::min(
        rows,
        std::max(1, Nvfp4MultiprocessorCount() * Nvfp4BlocksPerSm(threads) / gridY));
    dim3 block(threads);
    dim3 grid(gridX, gridY);
    FastllmSiluMulNvfp4QuantOptimizedKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, hidden,
        paddedHidden, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

/**
 * 使用缓存的GPU属性启动第三版SwiGLU+NVFP4融合量化。
 *
 * device kernel与Optimized版本相同，差异只在host热路径：SM数量和线程上限
 * 由调用方缓存后传入，避免每次启动重复查询CUDA设备属性；每个CPU线程在
 * 某块GPU上的首次启动检查一次同步launch错误，后续异步错误由调用方在
 * 结果依赖或显式同步边界取得。该函数不分配显存，也不执行流同步。
 *
 * @tparam T           输入元素类型，仅支持half或__nv_bfloat16。
 * @param input        输入激活，逻辑形状为[rows, 2 * hidden]。
 * @param output       打包E2M1输出，容量至少为rows * paddedHidden / 2字节。
 * @param scales       CUTLASS swizzle布局的E4M3局部scale输出。
 * @param rows         真实激活行数，通常为token数。
 * @param hidden       SwiGLU输出宽度。
 * @param paddedHidden 对齐后的输出宽度，不得小于hidden。
 * @param globalScale  激活动态量化使用的正数FP32全局scale。
 * @param properties   当前GPU已缓存的启动属性。
 * @param stream       异步启动kernel所使用的CUDA流。
 * @return 启动参数有效且首次launch检查成功时返回true；后续异步错误由
 *         调用方通过CUDA同步接口取得。
 */
template <typename T>
static bool LaunchSiluMulQuantCached(
        const void *input, uint8_t *output, uint8_t *scales,
        int rows, int hidden, int paddedHidden, float globalScale,
        const Nvfp4CachedLaunchProperties &properties,
        cudaStream_t stream) {
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    const int realPacks = std::max(1, hidden / fastllm_nvfp4::kElementsPerThread);
    const int threads = std::min(512, realPacks);
    const int kTiles = (paddedHidden + 63) / 64;
    const int gridY = (packs + threads - 1) / threads;
    const int blocksPerSm = std::min(
        4, std::max(1, properties.maxThreadsPerSm / std::max(1, threads)));
    const int gridX = std::min(
        rows,
        std::max(1, properties.multiprocessorCount * blocksPerSm / gridY));
    dim3 block(threads);
    dim3 grid(gridX, gridY);
    FastllmSiluMulNvfp4QuantOptimizedKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, hidden,
        paddedHidden, kTiles, globalScale);

    // 首次在该GPU启动时检查配置错误；benchmark warmup完成后不再让
    // cudaGetLastError进入每次迭代的host计时路径。
    static thread_local int validatedDevice = -1;
    if (validatedDevice != properties.device) {
        if (cudaGetLastError() != cudaSuccess) return false;
        validatedDevice = properties.device;
    }
    return true;
}

/** 启动优化前的固定256线程、M补齐128版本，仅用于benchmark。 */
template <typename T>
static bool LaunchSiluMulQuantBaseline(
        const void *input, uint8_t *output, uint8_t *scales,
        int rows, int hidden, int paddedHidden, float globalScale,
        cudaStream_t stream) {
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    const int threads = std::min(256, packs);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (paddedHidden + 63) / 64;
    dim3 block(threads);
    dim3 grid(std::min(roundedRows, 128), (packs + threads - 1) / threads);
    FastllmSiluMulNvfp4QuantBaselineKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, hidden,
        paddedHidden, roundedRows, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

#undef FASTLLM_NVFP4_BLOCKS_PER_SM
#undef FASTLLM_NVFP4_MAX_THREADS_PER_SM

} // namespace

size_t FastllmCudaNvfp4SwizzledScaleBytes(int rows, int columns) {
    return rows > 0 && columns > 0 && columns % 16 == 0
        ? fastllm_nvfp4::SwizzledScaleBytes(rows, columns) : 0;
}

bool FastllmCudaNvfp4QuantizeActivation(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int columns, float globalScale, void *streamPtr) {
    return FastllmCudaNvfp4QuantizeActivationPadded(
        input, inputType, output, outputScales, rows, columns, columns,
        globalScale, streamPtr);
}

bool FastllmCudaNvfp4QuantizeActivationPadded(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int columns, int paddedColumns,
        float globalScale, void *streamPtr) {
    Nvfp4CachedLaunchProperties properties;
    if (!GetCachedNvfp4LaunchProperties(properties) ||
        input == nullptr || output == nullptr ||
        outputScales == nullptr || rows <= 0 || columns <= 0 || columns % 16 != 0 ||
        paddedColumns < columns || paddedColumns % 32 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchQuantCached<half>(
            input, output, outputScales, rows, columns, paddedColumns,
            globalScale, properties, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchQuantCached<__nv_bfloat16>(
            input, output, outputScales, rows, columns, paddedColumns,
            globalScale, properties, stream);
    }
    return false;
}

bool FastllmCudaSiluMulNvfp4Quantize(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, float globalScale, void *streamPtr) {
    return FastllmCudaSiluMulNvfp4QuantizePadded(
        input, inputType, output, outputScales, rows, hidden, hidden,
        globalScale, streamPtr);
}

/**
 * 显式选择一个SwiGLU+NVFP4融合量化版本并异步启动计算。
 *
 * 本接口只用于同一二进制内的算子正确性和性能横向比较，不改变正式推理
 * 使用的版本。Baseline保留旧启动策略，Optimized只处理真实M并按GPU占用
 * 率设置grid，Cached在Optimized基础上缓存GPU启动属性并移除稳定热路径的
 * 重复launch检查。三个版本生成相同的E2M1数据和E4M3 swizzle scale布局。
 *
 * 输入和输出的数学语义为：output = NVFP4Quantize(SiLU(gate) * up)，其中
 * input逻辑形状为[rows, 2 * hidden]，output逻辑形状为[rows, hidden]。
 * kernel异步错误由测试框架在结果拷贝或显式同步时取得。
 *
 * @param input       FP16或BF16输入，逻辑形状为[rows, 2 * hidden]。
 * @param inputType   输入类型，仅支持FLOAT16和BFLOAT16。
 * @param output      打包E2M1输出，每两个FP4元素占一个字节。
 * @param outputScales CUTLASS swizzle布局的E4M3局部scale输出。
 * @param rows        输入行数，通常为本次参与计算的token数。
 * @param hidden      SwiGLU输出宽度，必须是32的倍数。
 * @param globalScale 激活动态量化使用的正数FP32全局scale。
 * @param version     待测试的Baseline、Optimized或Cached版本。
 * @param streamPtr   CUDA流指针；为空时使用默认流。
 * @return 参数、GPU能力和对应版本的kernel启动成功时返回true；否则返回
 *         false，异步执行错误需由调用方同步后取得。
 */
bool FastllmCudaSiluMulNvfp4QuantizeVersion(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, float globalScale,
        FastllmCudaNvfp4SiluMulVersion version, void *streamPtr) {
    if (input == nullptr || output == nullptr || outputScales == nullptr ||
        rows <= 0 || hidden <= 0 || hidden % 32 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (version == FastllmCudaNvfp4SiluMulVersion::Cached) {
        Nvfp4CachedLaunchProperties properties;
        if (!GetCachedNvfp4LaunchProperties(properties)) return false;
        if (inputType == fastllm::DataType::FLOAT16) {
            return LaunchSiluMulQuantCached<half>(
                input, output, outputScales, rows, hidden, hidden,
                globalScale, properties, stream);
        }
        if (inputType == fastllm::DataType::BFLOAT16) {
            return LaunchSiluMulQuantCached<__nv_bfloat16>(
                input, output, outputScales, rows, hidden, hidden,
                globalScale, properties, stream);
        }
        return false;
    }
    // 保留旧版本原有的逐次能力检查，使多版本benchmark只比较指定改动。
    if (!FastllmNvfp4RuntimeSupported()) return false;
    if (inputType == fastllm::DataType::FLOAT16) {
        return version == FastllmCudaNvfp4SiluMulVersion::Baseline
            ? LaunchSiluMulQuantBaseline<half>(
                  input, output, outputScales, rows, hidden, hidden,
                  globalScale, stream)
            : LaunchSiluMulQuantOptimized<half>(
                  input, output, outputScales, rows, hidden, hidden,
                  globalScale, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return version == FastllmCudaNvfp4SiluMulVersion::Baseline
            ? LaunchSiluMulQuantBaseline<__nv_bfloat16>(
                  input, output, outputScales, rows, hidden, hidden,
                  globalScale, stream)
            : LaunchSiluMulQuantOptimized<__nv_bfloat16>(
                  input, output, outputScales, rows, hidden, hidden,
                  globalScale, stream);
    }
    return false;
}

/**
 * 融合执行SwiGLU并把结果动态量化为CUTLASS使用的NVFP4激活布局。
 *
 * 输入的每一行按[gate(hidden), up(hidden)]连续存放。kernel先计算
 * SiLU(gate) * up，再以每16个元素一组生成E2M1 FP4值和E4M3 group
 * scale。FP4结果按两个元素一个字节写入output；group scale直接写成
 * CUTLASS要求的swizzle布局。hidden到paddedHidden之间的补齐区域写零。
 *
 * 正式推理固定使用Cached第三版：当前CPU线程首次访问GPU时读取并缓存
 * 启动属性，后续热调用复用缓存。本函数不执行流同步；调用方需要在
 * warmup或依赖结果的边界检查异步CUDA错误。
 *
 * @param input        FP16或BF16输入，逻辑形状为[rows, 2 * hidden]。
 * @param inputType    输入类型，仅支持FLOAT16和BFLOAT16。
 * @param output       打包E2M1输出，容量至少为rows * paddedHidden / 2字节。
 * @param outputScales swizzle E4M3 scale输出，容量至少为
 *                     FastllmCudaNvfp4SwizzledScaleBytes(rows, paddedHidden)。
 * @param rows         输入行数，通常为本次参与计算的token数。
 * @param hidden       SwiGLU输出宽度，必须是16的倍数。
 * @param paddedHidden CUTLASS使用的补齐宽度，必须不小于hidden且为32的倍数。
 * @param globalScale  激活动态量化使用的正数全局缩放因子。
 * @param streamPtr    CUDA流指针；为空时使用默认流。
 * @return 参数、GPU架构和kernel启动均成功时返回true，否则返回false。
 */
bool FastllmCudaSiluMulNvfp4QuantizePadded(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, int paddedHidden,
        float globalScale, void *streamPtr) {
    if (input == nullptr || output == nullptr || outputScales == nullptr ||
        rows <= 0 || hidden <= 0 || hidden % 16 != 0 ||
        paddedHidden < hidden || paddedHidden % 32 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    Nvfp4CachedLaunchProperties properties;
    if (!GetCachedNvfp4LaunchProperties(properties)) return false;
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchSiluMulQuantCached<half>(
            input, output, outputScales, rows, hidden,
            paddedHidden, globalScale, properties, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchSiluMulQuantCached<__nv_bfloat16>(
            input, output, outputScales, rows, hidden,
            paddedHidden, globalScale, properties, stream);
    }
    return false;
}

bool FastllmCudaNvfp4Block16ToCutlass(
        const uint8_t *source, const uint8_t *groupScales,
        uint8_t *packedWeight, uint8_t *weightScales,
        int rows, int columns, void *streamPtr) {
    return FastllmCudaNvfp4Block16ToCutlassPadded(
        source, groupScales, packedWeight, weightScales,
        rows, columns, rows, columns,
        streamPtr);
}

bool FastllmCudaNvfp4Block16ToCutlassPadded(
        const uint8_t *source, const uint8_t *groupScales,
        uint8_t *packedWeight, uint8_t *weightScales,
        int rows, int columns, int paddedRows, int paddedColumns,
        void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || source == nullptr || packedWeight == nullptr ||
        groupScales == nullptr || weightScales == nullptr ||
        rows <= 0 || columns <= 0 || columns % 16 != 0) {
        return false;
    }
    if (paddedRows < rows || paddedColumns < columns || paddedRows % 32 != 0 ||
        paddedColumns % 32 != 0) return false;
    const int groups = paddedColumns / 16;
    const int threads = std::min(256, groups);
    const int sourceBytesPerRow = (columns / 16) * (8 + sizeof(float));
    const int kTiles = (paddedColumns + 63) / 64;
    dim3 grid(std::min((paddedRows + 127) / 128 * 128, 128),
              (groups + threads - 1) / threads);
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    FastllmNvfp4Block16ToCutlassKernel<<<grid, threads, 0, stream>>>(
        source, groupScales, packedWeight, weightScales, rows, columns,
        paddedRows, paddedColumns, sourceBytesPerRow, kTiles);
    return cudaGetLastError() == cudaSuccess;
}
