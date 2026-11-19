#include "distributed/sm70_expert_gemm.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace fastllm {
namespace distributed {
namespace {

template <typename Output>
using FixedGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    Output, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<Output, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 8, 8, false>;

/** 检查 CUDA 提交结果；保留原错误，不尝试其他算法。 */
void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("SM70 expert GEMM CUDA: ") + cudaGetErrorString(status));
}

/** 将 FP16 bias 广播为输出类型，供同一 stream 后续 GEMM 的 FP32 epilogue 相加。 */
template <typename Output>
__global__ void FillBias(const half *bias, Output *output, int count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = Output(__half2float(bias[i % n]));
}

/** 在捕获前确认固定 kernel 可在当前设备执行；无需每次 GEMM 查询设备或设置属性。 */
template <typename Output>
void ValidateKernel(const cudaDeviceProp &properties) {
    using Kernel = typename FixedGemm<Output>::GemmKernel;
    // 固定配置必须能直接提交；禁止在 Graph 捕获中临时调整动态共享内存属性。
    static_assert(sizeof(typename Kernel::SharedStorage) < 48 * 1024,
                  "SM70 expert GEMM must not opt into shared memory during capture");
    cudaFuncAttributes attributes{};
    CheckCuda(cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>));
    if (attributes.maxThreadsPerBlock < Kernel::kThreadCount ||
        properties.sharedMemPerBlock < sizeof(typename Kernel::SharedStorage))
        throw std::runtime_error("SM70 expert GEMM kernel resource limit");
}

/** 构造固定配置并提交 bias/GEMM；检查布局和零工作区要求，不同步或分配。 */
template <typename Output>
void Launch(const half *input, const half *weight, const half *bias, void *output,
            int m, int n, int k, cudaStream_t stream) {
    using Gemm = FixedGemm<Output>;
    auto y = static_cast<Output *>(output);
    typename Gemm::Arguments args(
        {m, n, k}, {reinterpret_cast<const cutlass::half_t *>(input), k},
        {reinterpret_cast<const cutlass::half_t *>(weight), k}, {y, n}, {y, n},
        {1.0f, bias ? 1.0f : 0.0f}, 1);
    auto status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess || Gemm::get_workspace_size(args) != 0)
        throw std::runtime_error("SM70 expert GEMM unsupported layout: M=" + std::to_string(m) +
            " N=" + std::to_string(n) + " K=" + std::to_string(k) +
            " status=" + cutlass::cutlassGetStatusString(status));

    if (bias) {
        FillBias<<<(m * n + 255) / 256, 256, 0, stream>>>(bias, y, m * n, n);
        CheckCuda(cudaGetLastError());
    }
    Gemm operation;
    status = operation(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("SM70 expert GEMM launch: ") + cutlass::cutlassGetStatusString(status));
}
} // namespace

void ValidateSm70ExpertGemmDevice() {
    int device = 0;
    cudaDeviceProp properties{};
    CheckCuda(cudaGetDevice(&device));
    CheckCuda(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 7 || properties.minor != 0)
        throw std::runtime_error("batch-invariant MoE requires an SM70 GPU (V100)");
    ValidateKernel<cutlass::half_t>(properties);
    ValidateKernel<float>(properties);
}

void Sm70ExpertGemm(const half *input, const half *weight, const half *bias, void *output,
                    int m, int n, int k, bool fp32, cudaStream_t stream) {
    const auto aligned = [](const void *p, size_t alignment) {
        return p && reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };
    const int limit = std::numeric_limits<int>::max() - 255;
    if (m < 0 || n <= 0 || k <= 0 || k % 8 || !stream ||
        int64_t(m) * n > limit || int64_t(m) * k > limit || int64_t(n) * k > limit)
        throw std::invalid_argument("SM70 expert GEMM invalid shape/stream");
    if (!m) return;
    if (!aligned(input, 16) || !aligned(weight, 16) ||
        !aligned(output, fp32 ? sizeof(float) : sizeof(half)) ||
        (bias && !aligned(bias, sizeof(half))))
        throw std::invalid_argument("SM70 expert GEMM requires aligned contiguous tensors");

    if (fp32) Launch<float>(input, weight, bias, output, m, n, k, stream);
    else Launch<cutlass::half_t>(input, weight, bias, output, m, n, k, stream);
}

} // namespace distributed
} // namespace fastllm
