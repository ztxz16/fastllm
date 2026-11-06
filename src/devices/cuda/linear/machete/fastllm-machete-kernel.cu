#include "fastllm-machete-kernel.cuh"
#include "machete_prepack_kernel.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fastllm_machete {
namespace {

/** 检查CUDA调用，保留实际错误信息；失败不允许回退到其他后端。 */
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void *ptr = nullptr;
    int device = -1;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (!ptr) return;
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaFree(ptr);
        cudaSetDevice(previous);
    }
    void Allocate(size_t bytes) {
        Check(cudaGetDevice(&device));
        if (bytes) Check(cudaMalloc(&ptr, bytes));
    }
};

template<int TileM>
struct Schedule {
    using TileShapeNM = cute::Shape<cute::_128, cute::Int<TileM>>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileScheduler = cutlass::gemm::StreamKScheduler;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

template<class T, class Q, int TileM, bool WithZero>
using Kernel = machete::MacheteKernelTemplate<T, Q, T, float, T,
    std::conditional_t<WithZero, T, void>, void, void,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative, Schedule<TileM>>;

/** 将FastLLM高半字节在前的INT4转为CUTLASS逻辑元素顺序，不重新量化。 */
__global__ void SwapNibbles(const unsigned char *input, unsigned char *output, size_t bytes) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += size_t(blockDim.x) * gridDim.x) {
        unsigned char q = input[i];
        output[i] = (q >> 4) | (q << 4);
    }
}

/** 对Machete输出增加FP32 bias并转换回激活类型，使用同一stream。 */
template<class T>
__global__ void AddBias(T *output, const float *bias, size_t count, int n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        output[i] = T(float(output[i]) + bias[i % n]);
    }
}

template<class T, class Q, bool WithZero>
struct TypedWeight final : Weight {
    int n, k, groupSize, device;
    Buffer packed, scales, offsets;
    // 老workspace保留到权重释放，避免已有CUDA Graph引用失效。
    std::vector<std::unique_ptr<Buffer>> workspaces;
    size_t workspaceCapacity = 0;
    void *workspace = nullptr;

    TypedWeight(int n_, int k_, int groupSize_) : n(n_), k(k_), groupSize(groupSize_) {
        Check(cudaGetDevice(&device));
    }

    template<int TileM>
    void Execute(const void *input, void *output, int rows, cudaStream_t stream) {
        using Impl = Kernel<T, Q, TileM, WithZero>;
        auto args = Impl::create_arguments(rows, n, k, input, packed.ptr,
                                          output, scales.ptr, offsets.ptr, groupSize);
        if (!Impl::can_implement(args)) {
            throw std::runtime_error("Machete CUTLASS can_implement rejected Linear shape/layout");
        }
        size_t required = Impl::get_workspace_size(args);
        if (required > workspaceCapacity) {
            cudaStreamCaptureStatus capture;
            Check(cudaStreamIsCapturing(stream, &capture));
            if (capture != cudaStreamCaptureStatusNone) {
                throw std::runtime_error("Machete workspace growth during graph capture; warm up this shape first");
            }
            auto next = std::make_unique<Buffer>();
            next->Allocate(required);
            workspace = next->ptr;
            workspaceCapacity = required;
            residentBytes += required;
            workspaces.push_back(std::move(next));
        }
        Impl::run(args, workspace, stream);
    }

    void Run(const void *input, void *output, const float *bias,
             int rows, cudaStream_t stream) override {
        int current;
        Check(cudaGetDevice(&current));
        if (current != device) throw std::runtime_error("Machete weight used on a different CUDA device");
        if (rows <= 16) Execute<16>(input, output, rows, stream);
        else if (rows <= 64) Execute<64>(input, output, rows, stream);
        else Execute<128>(input, output, rows, stream);
        if (bias) {
            AddBias<T><<<std::min<size_t>(65535, (size_t(rows) * n + 255) / 256), 256, 0, stream>>>(
                static_cast<T*>(output), bias, size_t(rows) * n, n);
        }
        Check(cudaGetLastError());
    }
};

/**
 * 将CPU量化参数转为Machete布局，并在捕获外完成一次权重预打包。
 * @return 拥有全部资源的句柄；参数舍入误差保存在句柄中。
 */
template<class T, class Q, bool WithZero = true>
std::unique_ptr<Weight> CreateTyped(const void *source, int n, int k, int groupSize,
                                  const float *srcScales, const float *srcOffsets,
                                  cudaStream_t stream) {
    using Impl = Kernel<T, Q, 16, WithZero>;
    using Layout = typename Impl::PrepackedLayoutB;
    auto result = std::make_unique<TypedWeight<T, Q, WithZero>>(n, k, groupSize);
    result->encoding = WithZero ? "unsigned_scale_offset" : "biased_integer_scale";
    const size_t bytes = size_t(n) * k * cute::sizeof_bits_v<Q> / 8;
    const int groups = k / groupSize;
    std::vector<T> scales(size_t(groups) * n), offsets(scales.size());
    for (int channel = 0; channel < n; channel++) {
        for (int group = 0; group < groups; group++) {
            size_t src = size_t(channel) * groups + group;
            size_t dst = size_t(group) * n + channel;
            scales[dst] = T(srcScales[src]);
            offsets[dst] = WithZero ? T(srcOffsets[src]) : T(0.0f);
            if (!std::isfinite(float(scales[dst])) || !std::isfinite(float(offsets[dst]))) {
                throw std::runtime_error("Machete scale/offset is non-finite after A16 conversion");
            }
            result->maxScaleError = std::max(result->maxScaleError,
                                             std::abs(float(scales[dst]) - srcScales[src]));
            if constexpr (WithZero) {
                result->maxOffsetError = std::max(result->maxOffsetError,
                                                  std::abs(float(offsets[dst]) - srcOffsets[src]));
            }
        }
    }
    result->packed.Allocate(bytes);
    result->scales.Allocate(scales.size() * sizeof(T));
    result->offsets.Allocate(offsets.size() * sizeof(T));
    result->residentBytes = bytes + (scales.size() + offsets.size()) * sizeof(T);
    Check(cudaMemcpyAsync(result->scales.ptr, scales.data(), scales.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    Check(cudaMemcpyAsync(result->offsets.ptr, offsets.data(), offsets.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    Buffer unpackOrder;
    if constexpr (cute::sizeof_bits_v<Q> == 4) {
        unpackOrder.Allocate(bytes);
        SwapNibbles<<<std::min<size_t>(65535, (bytes + 255) / 256), 256, 0, stream>>>(
            static_cast<const unsigned char*>(source), static_cast<unsigned char*>(unpackOrder.ptr), bytes);
        source = unpackOrder.ptr;
    }
    auto layout = cute::make_layout(cute::make_shape(n, k, 1),
                                   cute::make_stride(int64_t(k), cute::_1{}, int64_t(0)));
    machete::prepack_B_template<Layout>(stream, static_cast<const Q*>(source), layout,
                                        static_cast<Q*>(result->packed.ptr));
    Check(cudaGetLastError());
    // 捕获外、一次性初始化：保证CPU临时参数和跨线程使用的权重已就绪。
    Check(cudaStreamSynchronize(stream));
    return result;
}
}

std::unique_ptr<Weight> Create(int bits, bool bf16, const void *source,
                             int n, int k, int groupSize,
                             const float *scales, const float *offsets,
                             cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    Check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::runtime_error("Machete prepack during graph capture; warm up the model first");
    }
    if (n <= 0 || k <= 0 || n % 128 || k % 64 || groupSize <= 0 ||
        k % groupSize || (groupSize != k && groupSize % 64) ||
        !source || !scales || !offsets || (bits != 4 && bits != 8)) {
        throw std::runtime_error("Invalid Machete weight shape, group size, pointers or bit width");
    }
    // 固定对称偏移使用上游GPTQ biased整数类型，避免unsigned乘scale后再相减的消减误差。
    bool symmetric = true;
    const float zero = bits == 4 ? 8.0f : 128.0f;
    for (size_t i = 0; i < size_t(n) * (k / groupSize); i++) {
        symmetric = symmetric && offsets[i] == -scales[i] * zero;
    }
    if (symmetric) {
        if (bf16) {
            if (bits == 4) return CreateTyped<cutlass::bfloat16_t, cutlass::vllm_uint4b8_t, false>(source, n, k, groupSize, scales, offsets, stream);
            return CreateTyped<cutlass::bfloat16_t, cutlass::vllm_uint8b128_t, false>(source, n, k, groupSize, scales, offsets, stream);
        }
        if (bits == 4) return CreateTyped<cutlass::half_t, cutlass::vllm_uint4b8_t, false>(source, n, k, groupSize, scales, offsets, stream);
        return CreateTyped<cutlass::half_t, cutlass::vllm_uint8b128_t, false>(source, n, k, groupSize, scales, offsets, stream);
    }
    if (bf16) {
        if (bits == 4) return CreateTyped<cutlass::bfloat16_t, cutlass::uint4b_t>(source, n, k, groupSize, scales, offsets, stream);
        return CreateTyped<cutlass::bfloat16_t, uint8_t>(source, n, k, groupSize, scales, offsets, stream);
    }
    if (bits == 4) return CreateTyped<cutlass::half_t, cutlass::uint4b_t>(source, n, k, groupSize, scales, offsets, stream);
    return CreateTyped<cutlass::half_t, uint8_t>(source, n, k, groupSize, scales, offsets, stream);
}
}
