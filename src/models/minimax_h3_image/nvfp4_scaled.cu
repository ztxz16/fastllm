#include "quantized_kernels.h"
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>
namespace fastllm::minimax_h3_image::detail {
namespace {
using namespace cute;
struct Config {
    static constexpr int N = 128, K = 256;
    using Shape = cute::Shape<_128, Int<N>, Int<K>>;
    using Cluster = cute::Shape<_1, _1, _1>;
    using F4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Shape, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, cutlass::layout::RowMajor, 8,
        cutlass::bfloat16_t, cutlass::layout::RowMajor, 8, cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, float, void, float>>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, F4, cutlass::layout::RowMajor, 32,
        F4, cutlass::layout::ColumnMajor, 32, float, Shape, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Main, Epi,
                                                        cutlass::gemm::StaticPersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using Nvfp4Kernel = Config::Kernel;
__global__ __launch_bounds__(
    Nvfp4Kernel::MaxThreadsPerBlock,
    Nvfp4Kernel::MinBlocksPerMultiprocessor) void nvfp4_128x128x256(__grid_constant__ const
                                                                        Nvfp4Kernel::Params params) {
    extern __shared__ char smem[];
    Nvfp4Kernel kernel;
    kernel(params, smem);
}
} // namespace
void nvfp4Gemm(void *y, const void *x, const void *w, const void *xs, const void *ws, const float *alpha,
               int m, int n, int k, cudaStream_t stream) {
    using C = Config;
    using G = typename C::Gemm;
    using S = typename C::Main::Sm1xxBlkScaledConfig;
    typename G::Arguments a{};
    a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = cute::make_shape(m, n, k, 1);
    a.mainloop.ptr_A = static_cast<const cutlass::float_e2m1_t *>(x);
    a.mainloop.ptr_B = static_cast<const cutlass::float_e2m1_t *>(w);
    a.mainloop.ptr_SFA = static_cast<const cutlass::float_ue4m3_t *>(xs);
    a.mainloop.ptr_SFB = static_cast<const cutlass::float_ue4m3_t *>(ws);
    a.mainloop.dA = cutlass::make_cute_packed_stride(typename C::Kernel::StrideA{}, {m, k, 1});
    a.mainloop.dB = cutlass::make_cute_packed_stride(typename C::Kernel::StrideB{}, {n, k, 1});
    a.mainloop.layout_SFA = S::tile_atom_to_shape_SFA(a.problem_shape);
    a.mainloop.layout_SFB = S::tile_atom_to_shape_SFB(a.problem_shape);
    a.epilogue.ptr_D = static_cast<cutlass::bfloat16_t *>(y);
    a.epilogue.dD = cutlass::make_cute_packed_stride(typename C::Kernel::StrideD{}, {m, n, 1});
    a.epilogue.dC = a.epilogue.dD;
    a.epilogue.thread.alpha_ptr = alpha;
    a.scheduler.max_swizzle_size = 1;
    cudaDeviceProp prop{};
    auto deviceStatus = cudaGetDevice(&a.hw_info.device_id);
    if (deviceStatus == cudaSuccess)
        deviceStatus = cudaGetDeviceProperties(&prop, a.hw_info.device_id);
    if (deviceStatus != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(deviceStatus));
    a.hw_info.sm_count = prop.multiProcessorCount;
    a.hw_info.cluster_shape = dim3(1, 1, 1);
    a.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    G g;
    auto check = [](cutlass::Status s, const char *stage) {
        if (s != cutlass::Status::kSuccess)
            throw std::runtime_error(std::string(stage) + ": " + cutlass::cutlassGetStatusString(s) +
                                     " CUDA=" + cudaGetErrorString(cudaGetLastError()));
    };
    // Static tile scheduling and this epilogue require no global scratch.
    if (g.get_workspace_size(a) != 0)
        throw std::runtime_error("unexpected NVFP4 GEMM workspace requirement");
    check(g.can_implement(a), "can_implement");
    check(C::Kernel::initialize_workspace(a, nullptr, stream), "workspace");
    auto params = C::Kernel::to_underlying_arguments(a, nullptr);
    auto e = cudaFuncSetAttribute(nvfp4_128x128x256, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  C::Kernel::SharedStorageSize);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("NVFP4 kernel attribute: ") + cudaGetErrorString(e));
    nvfp4_128x128x256<<<C::Kernel::get_grid_shape(params), C::Kernel::get_block_shape(),
                        C::Kernel::SharedStorageSize, stream>>>(params);
    e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace fastllm::minimax_h3_image::detail
