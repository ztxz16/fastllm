/* Apache-2.0: adapted from vLLM nvfp4_blockwise_moe_kernel.cu. */
#include "fastllm-cuda.cuh"

#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM100)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace {
using namespace cute;

/**
 * 取得当前GPU的CUTLASS硬件信息，并缓存稳定设备属性。
 *
 * grouped MoE每次前向会执行两次GEMM。该函数只在调用线程首次使用某块
 * GPU时读取SM数量，避免每次GEMM重复调用cudaDeviceGetAttribute。
 *
 * @param hardwareInfo 返回当前GPU编号及SM数量。
 * @return CUDA设备及属性查询成功时返回true，否则返回false。
 */
static bool GetCachedHardwareInfo(cutlass::KernelHardwareInfo &hardwareInfo) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local int cachedDevice = -1;
    static thread_local int cachedSmCount = 0;
    if (cachedDevice != device) {
        int smCount = 0;
        if (cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess) {
            return false;
        }
        cachedDevice = device;
        cachedSmCount = smCount;
    }
    hardwareInfo.device_id = cachedDevice;
    hardwareInfo.sm_count = cachedSmCount;
    return true;
}

template <typename Out>
struct GroupedConfig {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementScale = cutlass::float_ue4m3_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using Accumulator = float;
    using Tile = Shape<_128, _128, _128>;
    using Cluster = Shape<_1, _1, _1>;
    static constexpr int AlignAB = 32;
    static constexpr int AlignD = 128 / cutlass::sizeof_bits<Out>::value;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        Shape<_128, _64>, Accumulator, Accumulator,
        Out, LayoutD *, AlignD, Out, LayoutD *, AlignD,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, LayoutA *, AlignAB, ElementB, LayoutB *, AlignAB,
        Accumulator, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void SetupMetadata(int32_t *shapes, StrideA *strideA,
                              StrideB *strideB, StrideD *strideD,
                              LayoutSFA *layoutA, LayoutSFB *layoutB,
                              const int *rows, int groups, int n, int k) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    shapes[group * 3] = rows[group];
    shapes[group * 3 + 1] = n;
    shapes[group * 3 + 2] = k;
    // Internal pointer-array strides are represented by one int64 value for
    // these packed row/column-major layouts, exactly as in vLLM's launcher.
    reinterpret_cast<int64_t *>(strideA)[group] = k;
    reinterpret_cast<int64_t *>(strideB)[group] = k;
    reinterpret_cast<int64_t *>(strideD)[group] = n;
    layoutA[group] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(rows[group], n, k, 1));
    layoutB[group] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(rows[group], n, k, 1));
}

template <typename T>
T *AllocArray(size_t count) {
    return static_cast<T *>(FastllmCudaMalloc(sizeof(T) * count));
}

template <typename Out>
bool RunGrouped(const uint8_t *const *hostA, const uint8_t *const *hostB,
                const uint8_t *const *hostScaleA,
                const uint8_t *const *hostScaleB,
                const float *const *hostAlpha, void *const *hostD,
                const int *hostRows, int groups, int n, int k,
                cudaStream_t stream) {
    using Config = GroupedConfig<Out>;
    using Gemm = typename Config::Gemm;
    using Kernel = typename Config::Kernel;
    using ShapeType = typename Config::ProblemShape::UnderlyingProblemShape;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
    using ScaleConfig = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto a = AllocArray<const cutlass::float_e2m1_t *>(groups);
    auto b = AllocArray<const cutlass::float_e2m1_t *>(groups);
    auto sa = AllocArray<const typename Config::ElementScale *>(groups);
    auto sb = AllocArray<const typename Config::ElementScale *>(groups);
    auto alpha = AllocArray<const float *>(groups);
    auto d = AllocArray<Out *>(groups);
    auto rows = AllocArray<int>(groups);
    auto shapes = AllocArray<int32_t>((size_t)groups * 3);
    auto strideA = AllocArray<StrideA>(groups);
    auto strideB = AllocArray<StrideB>(groups);
    auto strideD = AllocArray<StrideD>(groups);
    auto layoutA = AllocArray<LayoutSFA>(groups);
    auto layoutB = AllocArray<LayoutSFB>(groups);
    std::vector<const cutlass::float_e2m1_t *> castA(groups), castB(groups);
    std::vector<const typename Config::ElementScale *> castSA(groups), castSB(groups);
    std::vector<Out *> castD(groups);
    for (int i = 0; i < groups; ++i) {
        castA[i] = reinterpret_cast<const cutlass::float_e2m1_t *>(hostA[i]);
        castB[i] = reinterpret_cast<const cutlass::float_e2m1_t *>(hostB[i]);
        castSA[i] = reinterpret_cast<const typename Config::ElementScale *>(hostScaleA[i]);
        castSB[i] = reinterpret_cast<const typename Config::ElementScale *>(hostScaleB[i]);
        castD[i] = static_cast<Out *>(hostD[i]);
    }
    bool ok = a && b && sa && sb && alpha && d && rows && shapes &&
              strideA && strideB && strideD && layoutA && layoutB;
#define COPY_META(dst, src, bytes) \
    do { if (ok) ok = cudaMemcpyAsync((dst), (src), (bytes), cudaMemcpyHostToDevice, stream) == cudaSuccess; } while (0)
    COPY_META(a, castA.data(), sizeof(void *) * groups);
    COPY_META(b, castB.data(), sizeof(void *) * groups);
    COPY_META(sa, castSA.data(), sizeof(void *) * groups);
    COPY_META(sb, castSB.data(), sizeof(void *) * groups);
    COPY_META(alpha, hostAlpha, sizeof(void *) * groups);
    COPY_META(d, castD.data(), sizeof(void *) * groups);
    COPY_META(rows, hostRows, sizeof(int) * groups);
#undef COPY_META
    if (ok) {
        SetupMetadata<StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB, ScaleConfig>
            <<<1, std::min(256, groups), 0, stream>>>(
                shapes, strideA, strideB, strideD, layoutA, layoutB,
                rows, groups, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    void *workspace = nullptr;
    if (ok) {
        typename Kernel::MainloopArguments mainloop{
            a, strideA, b, strideB, sa, layoutA, sb, layoutB};
        typename Kernel::EpilogueArguments epilogue{
            {}, nullptr, strideD, d, strideD};
        epilogue.thread.alpha_ptr_array = const_cast<float **>(alpha);
        epilogue.thread.dAlpha = {_0{}, _0{}, 1};
        cutlass::KernelHardwareInfo hw;
        ok = GetCachedHardwareInfo(hw);
        if (ok) {
            typename Kernel::TileSchedulerArguments scheduler;
            using Raster = typename cutlass::gemm::kernel::detail::
                PersistentTileSchedulerSm100GroupParams<ShapeType>::RasterOrderOptions;
            scheduler.raster_order = Raster::AlongM;
            typename Gemm::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGrouped,
                {groups, reinterpret_cast<ShapeType *>(shapes), nullptr},
                mainloop, epilogue, hw, scheduler};
            Gemm gemm;
            size_t workspaceBytes = Gemm::get_workspace_size(args);
            workspace = workspaceBytes ? FastllmCudaMalloc(workspaceBytes) : nullptr;
            ok = (workspaceBytes == 0 || workspace != nullptr) &&
                 gemm.can_implement(args) == cutlass::Status::kSuccess &&
                 gemm.initialize(args, workspace, stream) == cutlass::Status::kSuccess &&
                 gemm.run(args, workspace, stream) == cutlass::Status::kSuccess &&
                 cudaGetLastError() == cudaSuccess;
        }
    }
    FastllmCudaFree(workspace);
    FastllmCudaFree(a); FastllmCudaFree(b); FastllmCudaFree(sa); FastllmCudaFree(sb);
    FastllmCudaFree(alpha); FastllmCudaFree(d); FastllmCudaFree(rows);
    FastllmCudaFree(shapes); FastllmCudaFree(strideA); FastllmCudaFree(strideB);
    FastllmCudaFree(strideD); FastllmCudaFree(layoutA); FastllmCudaFree(layoutB);
    return ok;
}
} // namespace
#endif

bool FastllmCudaNvfp4GroupedGemmSm100(
        const uint8_t *const *a, const uint8_t *const *b,
        const uint8_t *const *scaleA, const uint8_t *const *scaleB,
        const float *const *alpha, void *const *d, const int *rows,
        int groups, int n, int k, fastllm::DataType outputType,
        void *streamPtr) {
#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM100)
    if (!a || !b || !scaleA || !scaleB || !alpha || !d || !rows ||
        groups <= 0 || groups > 256 || n <= 0 || k <= 0 ||
        n % 32 != 0 || k % 32 != 0) return false;
    cudaStream_t stream = streamPtr ? static_cast<cudaStream_t>(streamPtr) : 0;
    if (outputType == fastllm::DataType::FLOAT16)
        return RunGrouped<cutlass::half_t>(a, b, scaleA, scaleB, alpha, d, rows, groups, n, k, stream);
    if (outputType == fastllm::DataType::BFLOAT16)
        return RunGrouped<cutlass::bfloat16_t>(a, b, scaleA, scaleB, alpha, d, rows, groups, n, k, stream);
#else
    (void)a; (void)b; (void)scaleA; (void)scaleB; (void)alpha; (void)d;
    (void)rows; (void)groups; (void)n; (void)k; (void)outputType; (void)streamPtr;
#endif
    return false;
}
