//
// SM70 (V100) quantized A16 GEMM bridge over ported TurboMind s884 kernels.
//
// The logic mirrors 1Cat-vLLM's csrc/quantization/awq/awq_sm70_gemm.cu, but is
// rewritten to use raw CUDA pointers instead of torch::Tensor so it can be
// driven directly from FastLLM's INT4_GROUP, FP8_E4M3, and NVFP4_BLOCK_16
// weights.
//

#include "devices/cuda/fastllm-awq-sm70.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "fastllm-cuda.cuh"

namespace fastllm {
namespace awq_sm70 {

namespace tm = turbomind::gemm;

namespace {

constexpr int kFp8GroupSize = 128;
constexpr int kPackedOutputAlignment = 32;
constexpr int kNvfp4GroupSize = 16;

struct Handle {
    void *tmWeight = nullptr;   // device, uint4 packed
    void *tmScales = nullptr;   // device, int32 packed
    int K = 0;                  // input dim
    int N = 0;                  // output dim
    int kLd = 0;
    int qLd = 0;
    int groupSize = 0;
};

struct WorkspaceHolder {
    void *barriers = nullptr;
    void *partials = nullptr;
    void *tensormaps = nullptr;
    int *flags = nullptr;
    tm::Workspace workspace{};
};

enum class DenseWeightKind : int {
    kAwq,
    kNvfp4,
    kFp8,
};

using DenseTuneKey = std::tuple<int, int, int, int, int>;

struct DeviceRuntime {
    tm::Gemm gemm;
    WorkspaceHolder workspace;
    std::mutex runMutex;
    std::set<DenseTuneKey> tunedShapes;
};

std::mutex g_prepareMutex;
std::mutex g_runtimeMutex;
std::unordered_map<int, std::unique_ptr<DeviceRuntime>> g_runtimes;

tm::DispatchPolicy SelectDispatch(DeviceRuntime &runtime,
                                  DenseWeightKind kind,
                                  int m, int n, int k, int group_size) {
    // Measure and reuse the best small-M specialization by default on SM70.
    // Larger calls use TurboMind's regular dispatch policy.
    constexpr int kTuneMaxRows = 16;
    if (m > kTuneMaxRows) {
        return tm::DispatchPolicy::kDefault;
    }
    // TurboMind's dispatch cache is operand-type-sensitive, so keep each
    // quantized format distinct even when its dimensions are identical.
    const auto key = std::make_tuple(
        static_cast<int>(kind), m, n, k, group_size);
    // The caller holds runtime.runMutex, so this per-device cache requires no
    // process-wide lock in the steady-state decode path.
    const bool needsMeasure = runtime.tunedShapes.insert(key).second;
    return needsMeasure ? tm::DispatchPolicy::kMeasure
                        : tm::DispatchPolicy::kReuse;
}

int CurrentDevice() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

DeviceRuntime &GetRuntime(int device) {
    // FastLLM normally has one worker thread per CUDA device. Cache stable
    // pointers after the first lookup so decode does not contend on a global
    // host mutex across tensor-parallel GPUs.
    thread_local int cachedDevice = -1;
    thread_local DeviceRuntime *cachedRuntime = nullptr;
    if (cachedDevice == device) {
        return *cachedRuntime;
    }
    std::lock_guard<std::mutex> lock(g_runtimeMutex);
    auto it = g_runtimes.find(device);
    if (it == g_runtimes.end()) {
        auto runtime = std::make_unique<DeviceRuntime>();
        auto &holder = runtime->workspace;
        cudaMalloc(&holder.barriers, tm::Gemm::kBarriersSize);
        cudaMalloc(&holder.partials, tm::Gemm::kPartialsSize);
        cudaMalloc(&holder.tensormaps, (size_t)8192 * 128);
        cudaMalloc((void **)&holder.flags, sizeof(int));
        cudaMemset(holder.barriers, 0, tm::Gemm::kBarriersSize);
        cudaMemset(holder.partials, 0, tm::Gemm::kPartialsSize);
        cudaMemset(holder.flags, 0, sizeof(int));
        holder.workspace.barriers = holder.barriers;
        holder.workspace.barriers_size = tm::Gemm::kBarriersSize;
        holder.workspace.partials = holder.partials;
        holder.workspace.partials_size = tm::Gemm::kPartialsSize;
        holder.workspace.tensormaps = holder.tensormaps;
        holder.workspace.tensormaps_size = (size_t)8192 * 128;
        holder.workspace.flags = holder.flags;
        cachedRuntime = runtime.get();
        g_runtimes.emplace(device, std::move(runtime));
    } else {
        cachedRuntime = it->second.get();
    }
    cachedDevice = device;
    return *cachedRuntime;
}

// dst[N, K] = src[K, N] (uint16, row-major transpose)
__global__ void TransposeU16Kernel(uint16_t *dst, const uint16_t *src, int K, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = K * N;
    if (idx >= total) {
        return;
    }
    int kk = idx / N;
    int nn = idx - kk * N;
    dst[(size_t)nn * K + kk] = src[idx];
}

// Decode FastLLM's per-output-row NVFP4_BLOCK_16 source directly into the
// uint16 input expected by TurboMind's layout converter. The common SM70
// converter is row-major and therefore consumes the transposed [N, K] view;
// retaining the other branch keeps this bridge correct if converter policy
// changes later.
__global__ void UnpackNvfp4Block16Kernel(const uint8_t *source,
                                         uint16_t *unpacked,
                                         half *scales,
                                         int K, int N,
                                         int sourceRowBytes,
                                         bool transposeWeight) {
    size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)K * N;
    if (flat >= total) {
        return;
    }
    const int out = (int)(flat / K);
    const int in = (int)(flat - (size_t)out * K);
    const int group = in >> 4;
    const int offset = in & 15;
    const uint8_t *block = source + (size_t)out * sourceRowBytes +
                           (size_t)group * 12;
    const uint8_t packed = block[offset >> 1];
    const uint16_t fp4 = (offset & 1) ? (packed >> 4) : (packed & 0xf);
    unpacked[transposeWeight ? flat : (size_t)in * N + out] = fp4;
    if (offset == 0) {
        scales[(size_t)group * N + out] =
            __float2half_rn(*reinterpret_cast<const float *>(block + 8));
    }
}

// Widen the row-major E4M3 bytes for TurboMind's converter and expand each
// FastLLM 2-D block scale across the output channels covered by that block.
__global__ void PrepareFp8SourcesKernel(const uint8_t *sourceWeight,
                                        const float *sourceScales,
                                        uint16_t *unpackedWeight,
                                        half *expandedScales,
                                        int K, int N,
                                        int inputBlockSize,
                                        int outputBlockSize,
                                        bool keepOutputMajor) {
    const size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)K * N;
    if (flat >= total) {
        return;
    }
    const int out = (int)(flat / K);
    const int in = (int)(flat - (size_t)out * K);
    unpackedWeight[keepOutputMajor ? flat : (size_t)in * N + out] =
        sourceWeight[flat];
    if (in % inputBlockSize == 0) {
        const int inputGroup = in / inputBlockSize;
        const int inputGroups = K / inputBlockSize;
        expandedScales[(size_t)inputGroup * N + out] = __float2half_rn(
            sourceScales[(size_t)(out / outputBlockSize) * inputGroups +
                         inputGroup]);
    }
}

// 把当前二进制在该 device 上实际执行的内核编译期 __CUDA_ARCH__ 写回主机。
// 与 TurboMind s884 GEMM 内核走相同编译路径：若用 -DCUDA_ARCH=60 之类未包含 sm_70
// 的目标编译，运行在 7.0 设备上时此处取到的也是 600，恰好对应 GEMM 内核被
// if constexpr (is_compatible(__CUDA_ARCH__)) 编译为空壳的情况。
__global__ void DetectCompiledArchKernel(int *out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__;
#else
    *out = 0;
#endif
}

// 返回当前 device 上实际执行内核的编译 arch（如 700）。失败返回 0。
int CompiledArchOnCurrentDevice() {
    int *d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        return 0;
    }
    int h = 0;
    DetectCompiledArchKernel<<<1, 1>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) {
        if (cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            h = 0;
        }
    } else {
        h = 0;
    }
    cudaFree(d);
    return h;
}

struct PackedLayouts {
    const tm::LayoutConverter *weightConverter = nullptr;
    const tm::LayoutConverter *scaleConverter = nullptr;
    tm::MatrixLayout weightSource{};
    tm::MatrixLayout packedWeight{};
    tm::MatrixLayout scaleSource{};
    tm::MatrixLayout packedScale{};
};

bool MakePackedLayouts(int K, int N, int groupSize,
                       turbomind::DataType packedWeightType,
                       PackedLayouts &layouts) {
    // Packing_v2<HMMA_884, OPERAND_B> packs output rows in groups of 32.
    if (K <= 0 || N <= 0 || groupSize <= 0 || K % groupSize != 0 ||
        N % kPackedOutputAlignment != 0) {
        return false;
    }
    auto weightConverters = tm::GetConverters(
        turbomind::kHalf, packedWeightType,
        turbomind::kHalf, true, 70);
    // Both dense FP8 and NVFP4 use an FP16 V operand. NVFP4's own scale
    // converter is for E8M0, so use the E4M3 converter for this shared layout.
    auto scaleConverters = tm::GetConverters(
        turbomind::kHalf, turbomind::kFloat8_e4m3,
        turbomind::kHalf, true, 70);
    layouts.weightConverter = weightConverters[0];
    layouts.scaleConverter = scaleConverters[1];
    if (layouts.weightConverter == nullptr || layouts.scaleConverter == nullptr) {
        return false;
    }

    const auto orderW = layouts.weightConverter->order;
    const bool weightIsA =
        tm::get_operand_tag(layouts.weightConverter->pack) == tm::OPERAND_A;
    layouts.weightSource = {
        turbomind::kHalf, orderW, N, K,
        orderW == tm::kRowMajor ? K : N,
    };
    if (!weightIsA) {
        std::swap(layouts.weightSource.rows, layouts.weightSource.cols);
        layouts.weightSource.order = ~layouts.weightSource.order;
    }
    layouts.packedWeight = layouts.weightSource;
    layouts.packedWeight.type = packedWeightType;
    layouts.packedWeight.pack = layouts.weightConverter->pack;
    if (weightIsA) {
        layouts.packedWeight = tm::transpose(layouts.packedWeight);
    }
    layouts.packedWeight.ld =
        layouts.weightConverter->GetConvertedLd(layouts.weightSource);

    const int groups = K / groupSize;
    const auto orderS = layouts.scaleConverter->order;
    const bool scaleIsU =
        tm::get_operand_tag(layouts.scaleConverter->pack) == tm::OPERAND_U;
    layouts.scaleSource = {
        turbomind::kUint16, orderS, N, groups, N,
    };
    if (!scaleIsU) {
        std::swap(layouts.scaleSource.rows, layouts.scaleSource.cols);
        layouts.scaleSource.order = ~layouts.scaleSource.order;
    }
    layouts.packedScale = layouts.scaleSource;
    layouts.packedScale.pack = layouts.scaleConverter->pack;
    if (scaleIsU) {
        layouts.packedScale = tm::transpose(layouts.packedScale);
    }
    layouts.packedScale.ld =
        layouts.scaleConverter->GetConvertedLd(layouts.scaleSource);
    return true;
}

bool MakeNvfp4Layouts(int K, int N, PackedLayouts &layouts) {
    return MakePackedLayouts(K, N, kNvfp4GroupSize,
                             turbomind::kFloat4_e2m1, layouts);
}

bool MakeFp8Layouts(int K, int N, PackedLayouts &layouts) {
    return MakePackedLayouts(K, N, kFp8GroupSize,
                             turbomind::kFloat8_e4m3, layouts);
}

}  // namespace

bool Supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return false;
    }
    static thread_local std::unordered_map<int, bool> supportedByDevice;
    auto cached = supportedByDevice.find(dev);
    if (cached != supportedByDevice.end()) {
        return cached->second;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        supportedByDevice[dev] = false;
        return false;
    }
    if (major != 7 || minor != 0) {
        supportedByDevice[dev] = false;
        return false;
    }
    // 设备算力为 7.0 还不够：必须确认当前二进制确实为 sm_70 编译了 tensor core 内核。
    // 若用未包含 sm_70 的目标（如 -DCUDA_ARCH=60）编译，TurboMind s884 内核体会被
    // if constexpr (is_compatible(__CUDA_ARCH__)) 整段编译为空，GEMM 不写输出 -> 乱码。
    // 这里运行时探测实际编译的 __CUDA_ARCH__，<700 则禁用本路径、退回原生实现。
    const int compiledArch = CompiledArchOnCurrentDevice();
    if (compiledArch < 700) {
        static std::once_flag warning;
        std::call_once(warning, [compiledArch]() {
            printf("[Fastllm] SM70 TurboMind kernels disabled: binary not compiled for sm_70 "
                   "(detected __CUDA_ARCH__=%d). Falling back to native paths.\n",
                   compiledArch);
        });
        supportedByDevice[dev] = false;
        return false;
    }
    bool grouped = true;
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    const bool supported = converters[0] != nullptr && converters[1] != nullptr;
    supportedByDevice[dev] = supported;
    return supported;
}

bool Fp8Supported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    if (!Supported()) {
        return false;
    }
    static const bool convertersAvailable = []() {
        PackedLayouts layouts;
        return MakeFp8Layouts(
            kFp8GroupSize, kPackedOutputAlignment, layouts);
    }();
    return convertersAvailable;
#endif
}

bool Nvfp4Supported() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    if (!Supported()) {
        return false;
    }
    static const bool convertersAvailable = []() {
        PackedLayouts layouts;
        return MakeNvfp4Layouts(
            kNvfp4GroupSize, kPackedOutputAlignment, layouts);
    }();
    return convertersAvailable;
#endif
}

void *Prepare(const uint16_t *d_qvals_u16, const half *d_scales, const half *d_zeros,
              int K, int N, int num_groups, int group_size, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_prepareMutex);

    // 入参判空：任意源指针为空都会让后续 TurboMind Convert 命中
    // "'S' Must be non NULL" 断言并 abort，这里先打印出来便于定位。
    if (d_qvals_u16 == nullptr || d_scales == nullptr || d_zeros == nullptr) {
        printf("FastllmAwqSm70 Prepare error: null input pointer "
               "(qvals=%p scales=%p zeros=%p) K=%d N=%d num_groups=%d group_size=%d\n",
               (const void *)d_qvals_u16, (const void *)d_scales, (const void *)d_zeros,
               K, N, num_groups, group_size);
        return nullptr;
    }

    const bool grouped = (group_size != K);
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    const auto *conv_w = converters[0];
    const auto *conv_s = converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        printf("FastllmAwqSm70 Prepare error: no compatible TurboMind converters.\n");
        return nullptr;
    }

    // ---- weight ----
    const auto order_w = conv_w->order;
    const bool is_A_w = tm::get_operand_tag(conv_w->pack) == tm::OPERAND_A;
    const bool is_B_w = !is_A_w;

    const uint16_t *srcU16 = d_qvals_u16;  // [K, N]
    uint16_t *transposed = nullptr;
    if (order_w == tm::kRowMajor) {
        cudaError_t e = cudaMalloc((void **)&transposed, (size_t)K * N * sizeof(uint16_t));
        if (e != cudaSuccess || transposed == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(transposed, %zu bytes) failed: %s\n",
                   (size_t)K * N * sizeof(uint16_t), cudaGetErrorString(e));
            return nullptr;
        }
        int total = K * N;
        int threads = 256;
        TransposeU16Kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(transposed, d_qvals_u16, K, N);
        srcU16 = transposed;  // [N, K]
    }

    tm::MatrixLayout w_desc{
        turbomind::kHalf,
        order_w,
        N,
        K,
        order_w == tm::kRowMajor ? K : N,
    };
    if (is_B_w) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }

    tm::MatrixLayout k_desc = w_desc;
    k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
    k_desc.pack = conv_w->pack;
    if (is_A_w) {
        k_desc = tm::transpose(k_desc);
    }

    void *tmWeight = nullptr;
    {
        cudaError_t e = cudaMalloc(&tmWeight, (size_t)K * N / 2);  // uint4 packed
        if (e != cudaSuccess || tmWeight == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(tmWeight, %zu bytes) failed: %s\n",
                   (size_t)K * N / 2, cudaGetErrorString(e));
            if (transposed) cudaFree(transposed);
            return nullptr;
        }
    }
    if (srcU16 == nullptr) {
        printf("FastllmAwqSm70 Prepare error: weight source pointer is null before Convert "
               "(order_w=%d K=%d N=%d).\n", (int)order_w, K, N);
        if (transposed) cudaFree(transposed);
        cudaFree(tmWeight);
        return nullptr;
    }
    if (conv_w->Convert(srcU16, w_desc, tmWeight, k_desc, stream) != 0) {
        printf("FastllmAwqSm70 Prepare error: weight conversion failed.\n");
        if (transposed) cudaFree(transposed);
        cudaFree(tmWeight);
        return nullptr;
    }
    if (transposed) {
        cudaFree(transposed);
    }

    // ---- scales + zeros ----
    half *fused = nullptr;
    {
        cudaError_t e = cudaMalloc((void **)&fused, (size_t)num_groups * N * 2 * sizeof(half));
        if (e != cudaSuccess || fused == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(fused, %zu bytes) failed: %s\n",
                   (size_t)num_groups * N * 2 * sizeof(half), cudaGetErrorString(e));
            cudaFree(tmWeight);
            return nullptr;
        }
    }
    turbomind::fuse_scales_and_zeros(fused, d_scales, const_cast<half *>(d_zeros),
                                     (size_t)num_groups * N, stream);

    const auto order_s = conv_s->order;
    const bool is_A_s = tm::get_operand_tag(conv_s->pack) == tm::OPERAND_U;
    const bool is_B_s = !is_A_s;

    tm::MatrixLayout s_desc{
        turbomind::kUint32,
        order_s,
        N,
        num_groups,
        N,
    };
    if (is_B_s) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }

    tm::MatrixLayout q_desc = s_desc;
    q_desc.pack = conv_s->pack;
    if (is_A_s) {
        q_desc = tm::transpose(q_desc);
    }

    void *tmScales = nullptr;
    {
        cudaError_t e = cudaMalloc(&tmScales, (size_t)num_groups * N * sizeof(int32_t));
        if (e != cudaSuccess || tmScales == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(tmScales, %zu bytes) failed: %s\n",
                   (size_t)num_groups * N * sizeof(int32_t), cudaGetErrorString(e));
            cudaFree(fused);
            cudaFree(tmWeight);
            return nullptr;
        }
    }
    if (fused == nullptr) {
        printf("FastllmAwqSm70 Prepare error: scale source pointer is null before Convert "
               "(num_groups=%d N=%d).\n", num_groups, N);
        cudaFree(fused);
        cudaFree(tmWeight);
        cudaFree(tmScales);
        return nullptr;
    }
    if (conv_s->Convert(fused, s_desc, tmScales, q_desc, stream) != 0) {
        printf("FastllmAwqSm70 Prepare error: scale conversion failed.\n");
        cudaFree(fused);
        cudaFree(tmWeight);
        cudaFree(tmScales);
        return nullptr;
    }
    cudaFree(fused);

    auto *handle = new Handle();
    handle->tmWeight = tmWeight;
    handle->tmScales = tmScales;
    handle->K = K;
    handle->N = N;
    handle->kLd = k_desc.ld;
    handle->qLd = q_desc.ld;
    handle->groupSize = group_size;
    return handle;
}

bool Gemm(void *handlePtr, const half *in, half *out, int tokens, cudaStream_t stream) {
    if (handlePtr == nullptr) {
        return false;
    }
    auto *handle = static_cast<Handle *>(handlePtr);
    const int m = tokens;          // tokens
    const int k = handle->K;       // input dim
    const int n = handle->N;       // output dim
    const int group_size = handle->groupSize;

    const int device = CurrentDevice();

    const bool grouped = (group_size != k);
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    const auto *conv_w = converters[0];
    const auto *conv_s = converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        return false;
    }

    tm::MatrixLayout desc_A{turbomind::kHalf, tm::kRowMajor, m, k, k};
    tm::MatrixLayout desc_U{};

    const auto order_w = conv_w->order;
    const bool is_A_w = tm::get_operand_tag(conv_w->pack) == tm::OPERAND_A;
    const bool is_B_w = !is_A_w;

    tm::MatrixLayout w_desc{turbomind::kHalf, order_w, n, k, order_w == tm::kRowMajor ? k : n};
    if (is_B_w) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }
    tm::MatrixLayout desc_B = w_desc;
    desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
    desc_B.pack = conv_w->pack;
    if (is_A_w) {
        desc_B = tm::transpose(desc_B);
    }
    desc_B.ld = handle->kLd;

    const auto order_s = conv_s->order;
    const bool is_A_s = tm::get_operand_tag(conv_s->pack) == tm::OPERAND_U;
    const bool is_B_s = !is_A_s;
    const int num_groups = k / group_size;

    tm::MatrixLayout s_desc{turbomind::kUint32, order_s, n, num_groups, n};
    if (is_B_s) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }
    tm::MatrixLayout desc_V = s_desc;
    desc_V.pack = conv_s->pack;
    if (is_A_s) {
        desc_V = tm::transpose(desc_V);
    }
    desc_V.ld = handle->qLd;

    tm::MatrixLayout desc_D{turbomind::kHalf, tm::kRowMajor, m, n, n};

    tm::Operation op{};
    op.epilogue = tm::Epilogue::kNone;
    op.quant_a = {tm::QuantType::kNone, 0};
    op.quant_b = {tm::QuantType::kK, group_size};
    op.batch_dim = 0;

    DeviceRuntime &runtime = GetRuntime(device);
    std::lock_guard<std::mutex> runLock(runtime.runMutex);
    op.dispatch = SelectDispatch(
        runtime, DenseWeightKind::kAwq, m, n, k, group_size);
    const int ec = runtime.gemm.Run(
        op, 1.f, in, desc_A, nullptr, desc_U, handle->tmWeight, desc_B,
        handle->tmScales, desc_V, 0.f, out, desc_D, out, desc_D,
        runtime.workspace.workspace, stream);
    if (ec != 0) {
        printf("FastllmAwqSm70 Gemm error: TurboMind GEMM failed (ec=%d).\n", ec);
        return false;
    }
    return true;
}

void Free(void *handlePtr) {
    if (handlePtr == nullptr) {
        return;
    }
    auto *handle = static_cast<Handle *>(handlePtr);
    if (handle->tmWeight) cudaFree(handle->tmWeight);
    if (handle->tmScales) cudaFree(handle->tmScales);
    delete handle;
}

bool PrepareFp8InPlace(uint8_t *weight, const float *blockScales,
                       half **packedScales, int K, int N,
                       int inputBlockSize, int outputBlockSize,
                       cudaStream_t stream) {
    if (packedScales == nullptr) {
        return false;
    }
    *packedScales = nullptr;
    if (weight == nullptr || blockScales == nullptr ||
        inputBlockSize != kFp8GroupSize ||
        outputBlockSize != kFp8GroupSize || !Fp8Supported()) {
        return false;
    }

    PackedLayouts layouts;
    if (!MakeFp8Layouts(K, N, layouts)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_prepareMutex);

    const size_t elementCount = (size_t)K * N;
    const size_t unpackedWeightBytes = elementCount * sizeof(uint16_t);
    const size_t scaleBytes =
        (size_t)(K / inputBlockSize) * N * sizeof(half);
    const size_t scratchBytes =
        unpackedWeightBytes + scaleBytes + elementCount;

    FastllmCudaClearThreadError();
    auto *scratch = static_cast<uint8_t *>(FastllmCudaMalloc(scratchBytes));
    if (scratch == nullptr || FastllmCudaGetThreadError()) {
        if (scratch != nullptr) {
            FastllmCudaForceFree(scratch);
        }
        FastllmCudaClearThreadError();
        return false;
    }

    half *convertedScales = nullptr;
    cudaError_t operationState = cudaMalloc(
        reinterpret_cast<void **>(&convertedScales), scaleBytes);
    if (operationState != cudaSuccess || convertedScales == nullptr) {
        FastllmCudaForceFree(scratch);
        cudaGetLastError();
        FastllmCudaClearThreadError();
        return false;
    }

    auto *unpackedWeight = reinterpret_cast<uint16_t *>(scratch);
    auto *expandedScales = reinterpret_cast<half *>(
        scratch + unpackedWeightBytes);
    void *convertedWeight =
        scratch + unpackedWeightBytes + scaleBytes;

    cudaGetLastError();
    const int threads = 256;
    const int blocks = static_cast<int>((elementCount + threads - 1) /
                                        threads);
    PrepareFp8SourcesKernel<<<blocks, threads, 0, stream>>>(
        weight, blockScales, unpackedWeight, expandedScales,
        K, N, inputBlockSize, outputBlockSize,
        layouts.weightConverter->order == tm::kRowMajor);

    operationState = cudaPeekAtLastError();
    bool converted = operationState == cudaSuccess;
    if (converted) {
        converted = layouts.weightConverter->Convert(
            unpackedWeight, layouts.weightSource, convertedWeight,
            layouts.packedWeight, stream) == 0;
    }
    if (converted) {
        converted = layouts.scaleConverter->Convert(
            expandedScales, layouts.scaleSource, convertedScales,
            layouts.packedScale, stream) == 0;
    }
    if (converted) {
        operationState = cudaPeekAtLastError();
        converted = operationState == cudaSuccess;
    }

    bool overwriteStarted = false;
    if (converted) {
        operationState = cudaMemcpyAsync(
            weight, convertedWeight, elementCount,
            cudaMemcpyDeviceToDevice, stream);
        overwriteStarted = operationState == cudaSuccess;
        converted = overwriteStarted;
    }

    const cudaError_t syncState = cudaStreamSynchronize(stream);
    FastllmCudaForceFree(scratch);
    if (!converted || syncState != cudaSuccess) {
        const cudaError_t errorState = syncState != cudaSuccess
            ? syncState : operationState;
        cudaFree(convertedScales);
        if (overwriteStarted) {
            printf("Fastllm FP8 SM70 conversion failed after in-place copy began%s%s.\n",
                   errorState == cudaSuccess ? "" : ": ",
                   errorState == cudaSuccess ? "" : cudaGetErrorString(errorState));
            throw("fp8 sm70 in-place conversion error");
        }
        if (errorState != cudaSuccess) {
            printf("Fastllm FP8 SM70 conversion failed: %s.\n",
                   cudaGetErrorString(errorState));
        }
        cudaGetLastError();
        FastllmCudaClearThreadError();
        return false;
    }

    *packedScales = convertedScales;
    FastllmCudaClearThreadError();
    return true;
}

bool GemmFp8(const uint8_t *packedWeight, const half *packedScales,
             const half *in, half *out, int tokens, int K, int N,
             int groupSize, cudaStream_t stream) {
    if (packedWeight == nullptr || packedScales == nullptr || in == nullptr ||
        out == nullptr || tokens <= 0 || groupSize != kFp8GroupSize) {
        return false;
    }
    PackedLayouts layouts;
    if (!MakeFp8Layouts(K, N, layouts)) {
        return false;
    }

    tm::MatrixLayout descA{
        turbomind::kHalf, tm::kRowMajor, tokens, K, K,
    };
    tm::MatrixLayout descU{};
    tm::MatrixLayout descD{
        turbomind::kHalf, tm::kRowMajor, tokens, N, N,
    };
    tm::Operation op{};
    op.epilogue = tm::Epilogue::kNone;
    op.quant_a = {tm::QuantType::kNone, 0};
    op.quant_b = {tm::QuantType::kK, groupSize};
    op.batch_dim = 0;

    DeviceRuntime &runtime = GetRuntime(CurrentDevice());
    std::lock_guard<std::mutex> runLock(runtime.runMutex);
    op.dispatch = SelectDispatch(
        runtime, DenseWeightKind::kFp8, tokens, N, K, groupSize);
    const int ec = runtime.gemm.Run(
        op, 1.f, in, descA, nullptr, descU,
        packedWeight, layouts.packedWeight,
        packedScales, layouts.packedScale,
        0.f, out, descD, out, descD,
        runtime.workspace.workspace, stream);
    if (ec != 0) {
        printf("Fastllm FP8 SM70 GEMM failed (ec=%d, M=%d, N=%d, K=%d).\n",
               ec, tokens, N, K);
        return false;
    }
    return true;
}

bool PrepareNvfp4InPlace(uint8_t *storage, size_t storageBytes,
                         int K, int N, cudaStream_t stream) {
    if (storage == nullptr || !Nvfp4Supported()) {
        return false;
    }
    PackedLayouts layouts;
    if (!MakeNvfp4Layouts(K, N, layouts)) {
        return false;
    }

    const size_t elementCount = (size_t)K * N;
    const size_t sourceRowBytes =
        (size_t)(K / kNvfp4GroupSize) * 12;
    const size_t sourceBytes = (size_t)N * sourceRowBytes;
    const size_t packedWeightBytes = elementCount / 2;
    const size_t scaleBytes =
        (size_t)(K / kNvfp4GroupSize) * N * sizeof(half);
    const size_t persistentBytes = packedWeightBytes + scaleBytes;
    if (storageBytes < sourceBytes || storageBytes < persistentBytes) {
        return false;
    }

    // The converter cannot safely write over the interleaved source layout.
    // Keep all temporary inputs and outputs in one allocation, then copy the
    // smaller persistent representation back only after both conversions have
    // been enqueued successfully.
    const size_t unpackedWeightBytes = elementCount * sizeof(uint16_t);
    const size_t scratchBytes = unpackedWeightBytes + scaleBytes +
                                packedWeightBytes + scaleBytes;
    FastllmCudaClearThreadError();
    auto *scratch = static_cast<uint8_t *>(FastllmCudaMalloc(scratchBytes));
    if (scratch == nullptr || FastllmCudaGetThreadError()) {
        if (scratch != nullptr) {
            FastllmCudaForceFree(scratch);
        }
        FastllmCudaClearThreadError();
        return false;
    }

    auto *unpackedWeight = reinterpret_cast<uint16_t *>(scratch);
    auto *unpackedScales = reinterpret_cast<half *>(
        scratch + unpackedWeightBytes);
    void *convertedWeight = scratch + unpackedWeightBytes + scaleBytes;
    void *convertedScales = scratch + unpackedWeightBytes + scaleBytes +
                            packedWeightBytes;

    cudaGetLastError();
    const int threads = 256;
    const int blocks = static_cast<int>((elementCount + threads - 1) /
                                        threads);
    UnpackNvfp4Block16Kernel<<<blocks, threads, 0, stream>>>(
        storage, unpackedWeight, unpackedScales, K, N,
        static_cast<int>(sourceRowBytes),
        layouts.weightConverter->order == tm::kRowMajor);

    cudaError_t operationState = cudaPeekAtLastError();
    bool converted = operationState == cudaSuccess;
    if (converted) {
        converted = layouts.weightConverter->Convert(
            unpackedWeight, layouts.weightSource, convertedWeight,
            layouts.packedWeight, stream) == 0;
    }
    if (converted) {
        converted = layouts.scaleConverter->Convert(
            unpackedScales, layouts.scaleSource, convertedScales,
            layouts.packedScale, stream) == 0;
    }
    if (converted) {
        operationState = cudaPeekAtLastError();
        converted = operationState == cudaSuccess;
    }

    bool overwriteStarted = false;
    if (converted) {
        cudaError_t state = cudaMemcpyAsync(
            storage, convertedWeight, packedWeightBytes,
            cudaMemcpyDeviceToDevice, stream);
        operationState = state;
        overwriteStarted = state == cudaSuccess;
        converted = overwriteStarted;
    }
    if (converted) {
        cudaError_t state = cudaMemcpyAsync(
            storage + packedWeightBytes, convertedScales, scaleBytes,
            cudaMemcpyDeviceToDevice, stream);
        operationState = state;
        converted = state == cudaSuccess;
    }

    const cudaError_t syncState = cudaStreamSynchronize(stream);
    FastllmCudaForceFree(scratch);
    if (!converted || syncState != cudaSuccess) {
        const cudaError_t errorState = syncState != cudaSuccess
            ? syncState : operationState;
        if (overwriteStarted) {
            printf("Fastllm NVFP4 SM70 conversion failed after in-place copy began%s%s.\n",
                   errorState == cudaSuccess ? "" : ": ",
                   errorState == cudaSuccess ? "" : cudaGetErrorString(errorState));
            throw("nvfp4 sm70 in-place conversion error");
        }
        if (errorState != cudaSuccess) {
            printf("Fastllm NVFP4 SM70 conversion failed: %s.\n",
                   cudaGetErrorString(errorState));
        }
        cudaGetLastError();
        FastllmCudaClearThreadError();
        return false;
    }
    FastllmCudaClearThreadError();
    return true;
}

bool GemmNvfp4(const uint8_t *storage, const half *in, half *out,
               int tokens, int K, int N, cudaStream_t stream) {
    if (storage == nullptr || in == nullptr || out == nullptr || tokens <= 0) {
        return false;
    }
    PackedLayouts layouts;
    if (!MakeNvfp4Layouts(K, N, layouts)) {
        return false;
    }

    tm::MatrixLayout descA{
        turbomind::kHalf, tm::kRowMajor, tokens, K, K,
    };
    tm::MatrixLayout descU{};
    tm::MatrixLayout descB = layouts.packedWeight;
    tm::MatrixLayout descV = layouts.packedScale;
    tm::MatrixLayout descD{
        turbomind::kHalf, tm::kRowMajor, tokens, N, N,
    };
    const size_t packedWeightBytes = (size_t)K * N / 2;

    tm::Operation op{};
    const int device = CurrentDevice();
    op.epilogue = tm::Epilogue::kNone;
    op.quant_a = {tm::QuantType::kNone, 0};
    op.quant_b = {tm::QuantType::kK, kNvfp4GroupSize};
    op.batch_dim = 0;

    DeviceRuntime &runtime = GetRuntime(device);
    std::lock_guard<std::mutex> runLock(runtime.runMutex);
    op.dispatch = SelectDispatch(
        runtime, DenseWeightKind::kNvfp4, tokens, N, K,
        kNvfp4GroupSize);
    const int ec = runtime.gemm.Run(
        op, 1.f, in, descA, nullptr, descU, storage, descB,
        storage + packedWeightBytes, descV, 0.f, out, descD, out, descD,
        runtime.workspace.workspace, stream);
    if (ec != 0) {
        printf("Fastllm NVFP4 SM70 GEMM failed (ec=%d, M=%d, N=%d, K=%d).\n",
               ec, tokens, N, K);
        return false;
    }
    return true;
}

}  // namespace awq_sm70
}  // namespace fastllm
