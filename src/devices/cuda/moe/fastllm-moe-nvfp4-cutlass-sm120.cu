/* Apache-2.0: 忠实移植自 vLLM nvfp4_blockwise_moe_kernel.cu 的 SM120 路径。 */
#include "fastllm-cuda.cuh"

#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include <algorithm>
#include <climits>
#include <map>
#include <vector>

// NVCC 12.8会为__global__模板生成host stub。这里必须使用具名namespace，
// 否则stub中的匿名namespace名称可能与CUTE头文件内部的匿名namespace冲突。
namespace fastllm_nvfp4_moe_sm120 {
using namespace cute;

struct GroupedConfigSm120 {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
    using ElementType = cutlass::float_e2m1_t;
    using ElementSFType = cutlass::float_ue4m3_t;
    using ElementA = cutlass::nv_float4_t<ElementType>;
    using ElementB = cutlass::nv_float4_t<ElementType>;
    // 与 vLLM 一致：SM120 grouped NVFP4 当前固定输出 BF16。
    using ElementC = cutlass::bfloat16_t;
    using ElementD = ElementC;
    using Accumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using Arch = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using Cluster = Shape<_1, _1, _1>;
    using Tile = Shape<_128, _128, _128>;
    using Fusion = cutlass::epilogue::fusion::LinearCombination<
        ElementD, Accumulator, ElementC, Accumulator>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, OperatorClass, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator, Accumulator, ElementC, LayoutC *, AlignmentC,
        ElementD, LayoutD *, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OperatorClass, ElementA, LayoutA *, AlignmentA,
        ElementB, LayoutB *, AlignmentB, Accumulator, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static_assert(
        cute::is_base_of_v<
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
            typename Kernel::CollectiveMainloop::DispatchPolicy::Schedule>,
        "SM120 NVFP4 grouped GEMM requires the cooperative pointer-array schedule");
};

template <typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void SetupMetadataSm120(int32_t *shapes, StrideA *strideA,
                                   StrideB *strideB, StrideD *strideD,
                                   LayoutSFA *layoutA, LayoutSFB *layoutB,
                                   const int *rows, int groups, int n, int k) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    shapes[group * 3] = rows[group];
    shapes[group * 3 + 1] = n;
    shapes[group * 3 + 2] = k;
    reinterpret_cast<int64_t *>(strideA)[group] = k;
    reinterpret_cast<int64_t *>(strideB)[group] = k;
    reinterpret_cast<int64_t *>(strideD)[group] = n;
    layoutA[group] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(rows[group], n, k, 1));
    layoutB[group] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(rows[group], n, k, 1));
}

template <typename ElementA, typename ElementD, typename ElementSF,
          typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void SetupDeviceRouteMetadataSm120(
        const uint8_t *aBase, const uint8_t *scaleABase, void *dBase,
        const int *expertOffsets, const int *blockscaleOffsets,
        const ElementA **a, ElementD **d, const ElementSF **sa, int *rows,
        int32_t *shapes, StrideA *strideA, StrideB *strideB,
        StrideD *strideD, LayoutSFA *layoutA, LayoutSFB *layoutB,
        int groups, int n, int k) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const int begin = expertOffsets[group];
    const int groupRows = expertOffsets[group + 1] - begin;
    const int roundedScaleColumns = ((k / 16) + 3) / 4 * 4;
    a[group] = reinterpret_cast<const ElementA *>(
        aBase + (size_t)begin * (k / 2));
    sa[group] = reinterpret_cast<const ElementSF *>(
        scaleABase + (size_t)blockscaleOffsets[group] * roundedScaleColumns);
    d[group] = reinterpret_cast<ElementD *>(dBase) + (size_t)begin * n;
    rows[group] = groupRows;
    shapes[group * 3] = groupRows;
    shapes[group * 3 + 1] = n;
    shapes[group * 3 + 2] = k;
    reinterpret_cast<int64_t *>(strideA)[group] = k;
    reinterpret_cast<int64_t *>(strideB)[group] = k;
    reinterpret_cast<int64_t *>(strideD)[group] = n;
    layoutA[group] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(groupRows, n, k, 1));
    layoutB[group] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(groupRows, n, k, 1));
}

/** SM120 grouped GEMM复用的device metadata和CUTLASS workspace。 */
struct GroupedScratchSm120 {
    uint8_t *metadata = nullptr;
    void *workspace = nullptr;
    size_t metadataCapacity = 0;
    size_t workspaceCapacity = 0;
    int device = -1;
    int smCount = 0;
    bool staticMetadataReady = false;
    int staticGroups = 0;
    int staticN = 0;
    int staticK = 0;
};

using GroupedScratchKeySm120 = std::pair<int, const void *>;
static thread_local std::map<GroupedScratchKeySm120, GroupedScratchSm120>
    groupedScratchSm120;

template <typename T>
static bool EnsureBufferSm120(T *&buffer, size_t &capacity, size_t bytes) {
    if (bytes == 0 || (buffer != nullptr && capacity >= bytes)) return true;
    T *replacement = static_cast<T *>(FastllmCudaMalloc(bytes));
    if (replacement == nullptr) return false;
    if (buffer != nullptr) FastllmCudaFree(buffer);
    buffer = replacement;
    capacity = bytes;
    return true;
}

static size_t AlignMetadataSm120(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

/** 获取当前GPU/线程的持久metadata；thread_local防止per-thread stream互相覆盖。 */
static GroupedScratchSm120 *GetGroupedScratchSm120(
        size_t metadataBytes, const void *staticKey = nullptr) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    GroupedScratchSm120 &scratch = groupedScratchSm120[{device, staticKey}];
    scratch.device = device;
    // SM数量属于稳定设备属性，只在该线程首次使用当前GPU时读取一次。
    if (scratch.smCount <= 0 &&
        cudaDeviceGetAttribute(&scratch.smCount,
                               cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess) {
        scratch.smCount = 0;
        return nullptr;
    }
    if (FastllmCudaGraphIsCapturing() &&
        (scratch.metadata == nullptr || scratch.metadataCapacity < metadataBytes)) {
        return nullptr;
    }
    return EnsureBufferSm120(
        scratch.metadata, scratch.metadataCapacity, metadataBytes) ? &scratch : nullptr;
}

bool RunGroupedSm120(const uint8_t *const *hostA, const uint8_t *const *hostB,
                     const uint8_t *const *hostScaleA,
                     const uint8_t *const *hostScaleB,
                     const float *const *hostAlpha, void *const *hostD,
                     const int *hostRows, int groups, int n, int k,
                     cudaStream_t stream) {
    using Config = GroupedConfigSm120;
    using Gemm = Config::Gemm;
    using Kernel = Config::Kernel;
    using ShapeType = Config::ProblemShape::UnderlyingProblemShape;
    using StrideA = Kernel::InternalStrideA;
    using StrideB = Kernel::InternalStrideB;
    using StrideD = Kernel::InternalStrideD;
    using LayoutSFA = Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = Kernel::CollectiveMainloop::InternalLayoutSFB;
    using ScaleConfig = Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    // 所有device metadata放入一块持久buffer，只在容量增长时重新分配。
    size_t metadataBytes = 0;
#define RESERVE_META_SM120(type, count) \
    do { metadataBytes = AlignMetadataSm120(metadataBytes, alignof(type)); \
         metadataBytes += sizeof(type) * (size_t)(count); } while (0)
    RESERVE_META_SM120(const Config::ElementType *, groups); // A
    RESERVE_META_SM120(const Config::ElementType *, groups); // B
    RESERVE_META_SM120(const Config::ElementSFType *, groups); // scale A
    RESERVE_META_SM120(const Config::ElementSFType *, groups); // scale B
    RESERVE_META_SM120(const float *, groups); // alpha
    RESERVE_META_SM120(Config::ElementD *, groups); // D
    RESERVE_META_SM120(int, groups); // rows
    RESERVE_META_SM120(int32_t, (size_t)groups * 3); // shapes
    RESERVE_META_SM120(StrideA, groups);
    RESERVE_META_SM120(StrideB, groups);
    RESERVE_META_SM120(StrideD, groups);
    RESERVE_META_SM120(LayoutSFA, groups);
    RESERVE_META_SM120(LayoutSFB, groups);
#undef RESERVE_META_SM120
    GroupedScratchSm120 *scratch = GetGroupedScratchSm120(metadataBytes);
    if (scratch == nullptr) return false;
    size_t metadataOffset = 0;
#define TAKE_META_SM120(name, type, count) \
    metadataOffset = AlignMetadataSm120(metadataOffset, alignof(type)); \
    auto name = reinterpret_cast<type *>(scratch->metadata + metadataOffset); \
    metadataOffset += sizeof(type) * (size_t)(count)
    TAKE_META_SM120(a, const Config::ElementType *, groups);
    TAKE_META_SM120(b, const Config::ElementType *, groups);
    TAKE_META_SM120(sa, const Config::ElementSFType *, groups);
    TAKE_META_SM120(sb, const Config::ElementSFType *, groups);
    TAKE_META_SM120(alpha, const float *, groups);
    TAKE_META_SM120(d, Config::ElementD *, groups);
    TAKE_META_SM120(rows, int, groups);
    TAKE_META_SM120(shapes, int32_t, (size_t)groups * 3);
    TAKE_META_SM120(strideA, StrideA, groups);
    TAKE_META_SM120(strideB, StrideB, groups);
    TAKE_META_SM120(strideD, StrideD, groups);
    TAKE_META_SM120(layoutA, LayoutSFA, groups);
    TAKE_META_SM120(layoutB, LayoutSFB, groups);
#undef TAKE_META_SM120
    std::vector<const Config::ElementType *> castA(groups), castB(groups);
    std::vector<const Config::ElementSFType *> castSA(groups), castSB(groups);
    std::vector<Config::ElementD *> castD(groups);
    for (int i = 0; i < groups; ++i) {
        castA[i] = reinterpret_cast<const Config::ElementType *>(hostA[i]);
        castB[i] = reinterpret_cast<const Config::ElementType *>(hostB[i]);
        castSA[i] = reinterpret_cast<const Config::ElementSFType *>(hostScaleA[i]);
        castSB[i] = reinterpret_cast<const Config::ElementSFType *>(hostScaleB[i]);
        castD[i] = static_cast<Config::ElementD *>(hostD[i]);
    }
    bool ok = true;
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
        SetupMetadataSm120<StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB, ScaleConfig>
            <<<1, std::min(256, groups), 0, stream>>>(
                shapes, strideA, strideB, strideD, layoutA, layoutB,
                rows, groups, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (ok) {
        typename Kernel::MainloopArguments mainloop{
            a, strideA, b, strideB, sa, layoutA, sb, layoutB};
        typename Kernel::EpilogueArguments epilogue{{}, nullptr, strideD, d, strideD};
        epilogue.thread.alpha_ptr_array = const_cast<float **>(alpha);
        epilogue.thread.dAlpha = {_0{}, _0{}, 1};
        epilogue.thread.beta = 0.0f;
        cutlass::KernelHardwareInfo hw;
        hw.device_id = scratch->device;
        hw.sm_count = scratch->smCount;
        typename Kernel::TileSchedulerArguments scheduler;
        using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
        scheduler.raster_order = Raster::AlongM;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {groups, reinterpret_cast<ShapeType *>(shapes), nullptr},
            mainloop, epilogue, hw, scheduler};
        Gemm gemm;
        size_t workspaceBytes = Gemm::get_workspace_size(args);
        ok = EnsureBufferSm120(
                 scratch->workspace, scratch->workspaceCapacity, workspaceBytes) &&
             gemm.can_implement(args) == cutlass::Status::kSuccess &&
             gemm.initialize(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             gemm.run(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             cudaGetLastError() == cudaSuccess;
    }
    return ok;
}

/**
 * 使用GPU端expert偏移执行SM120 NVFP4 grouped GEMM。
 *
 * routed expert的M值、A/D起始地址和激活scale地址全部由GPU metadata
 * kernel根据expertOffsets生成；权重指针属于层级静态数据，只在该
 * gate/up或down权重组首次使用时上传。这样正式decode不再依赖CPU路由
 * 结果，也不重复搬运稳定权重metadata。
 *
 * @param aBase              按expert排序后的FP4激活基址。
 * @param scaleABase         按expert分块存放的E4M3激活scale基址。
 * @param dBase              按expert排序的BF16输出基址。
 * @param hostB              各expert的CUTLASS FP4权重指针。
 * @param hostScaleB         各expert的E4M3权重scale指针。
 * @param hostAlpha          各expert的FP32全局scale指针。
 * @param expertOffsets      GPU端expert路由前缀和，长度groups+1。
 * @param blockscaleOffsets  GPU端scale行前缀和，长度groups+1，按128行补齐。
 * @param groups             routed expert数量。
 * @param n                  GEMM输出宽度。
 * @param k                  GEMM归约宽度。
 * @param stream             正式推理CUDA流。
 * @return metadata准备和CUTLASS启动均成功时返回true。
 */
bool RunGroupedDeviceRoutesSm120(
        const uint8_t *aBase, const uint8_t *scaleABase, void *dBase,
        const uint8_t *const *hostB, const uint8_t *const *hostScaleB,
        const float *const *hostAlpha, const int *expertOffsets,
        const int *blockscaleOffsets, int groups, int n, int k,
        cudaStream_t stream) {
    using Config = GroupedConfigSm120;
    using Gemm = Config::Gemm;
    using Kernel = Config::Kernel;
    using ShapeType = Config::ProblemShape::UnderlyingProblemShape;
    using StrideA = Kernel::InternalStrideA;
    using StrideB = Kernel::InternalStrideB;
    using StrideD = Kernel::InternalStrideD;
    using LayoutSFA = Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = Kernel::CollectiveMainloop::InternalLayoutSFB;
    using ScaleConfig = Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    size_t metadataBytes = 0;
#define RESERVE_DEVICE_META(type, count) \
    do { metadataBytes = AlignMetadataSm120(metadataBytes, alignof(type)); \
         metadataBytes += sizeof(type) * (size_t)(count); } while (0)
    RESERVE_DEVICE_META(const Config::ElementType *, groups);
    RESERVE_DEVICE_META(const Config::ElementType *, groups);
    RESERVE_DEVICE_META(const Config::ElementSFType *, groups);
    RESERVE_DEVICE_META(const Config::ElementSFType *, groups);
    RESERVE_DEVICE_META(const float *, groups);
    RESERVE_DEVICE_META(Config::ElementD *, groups);
    RESERVE_DEVICE_META(int, groups);
    RESERVE_DEVICE_META(int32_t, (size_t)groups * 3);
    RESERVE_DEVICE_META(StrideA, groups);
    RESERVE_DEVICE_META(StrideB, groups);
    RESERVE_DEVICE_META(StrideD, groups);
    RESERVE_DEVICE_META(LayoutSFA, groups);
    RESERVE_DEVICE_META(LayoutSFB, groups);
#undef RESERVE_DEVICE_META
    GroupedScratchSm120 *scratch = GetGroupedScratchSm120(
        metadataBytes, hostB[0]);
    if (scratch == nullptr) return false;
    if (FastllmCudaGraphIsCapturing() && !scratch->staticMetadataReady) return false;
    size_t offset = 0;
#define TAKE_DEVICE_META(name, type, count) \
    offset = AlignMetadataSm120(offset, alignof(type)); \
    auto name = reinterpret_cast<type *>(scratch->metadata + offset); \
    offset += sizeof(type) * (size_t)(count)
    TAKE_DEVICE_META(a, const Config::ElementType *, groups);
    TAKE_DEVICE_META(b, const Config::ElementType *, groups);
    TAKE_DEVICE_META(sa, const Config::ElementSFType *, groups);
    TAKE_DEVICE_META(sb, const Config::ElementSFType *, groups);
    TAKE_DEVICE_META(alpha, const float *, groups);
    TAKE_DEVICE_META(d, Config::ElementD *, groups);
    TAKE_DEVICE_META(rows, int, groups);
    TAKE_DEVICE_META(shapes, int32_t, (size_t)groups * 3);
    TAKE_DEVICE_META(strideA, StrideA, groups);
    TAKE_DEVICE_META(strideB, StrideB, groups);
    TAKE_DEVICE_META(strideD, StrideD, groups);
    TAKE_DEVICE_META(layoutA, LayoutSFA, groups);
    TAKE_DEVICE_META(layoutB, LayoutSFB, groups);
#undef TAKE_DEVICE_META

    bool ok = true;
    if (!scratch->staticMetadataReady || scratch->staticGroups != groups ||
        scratch->staticN != n || scratch->staticK != k) {
        std::vector<const Config::ElementType *> castB(groups);
        std::vector<const Config::ElementSFType *> castSB(groups);
        for (int i = 0; i < groups; ++i) {
            castB[i] = reinterpret_cast<const Config::ElementType *>(hostB[i]);
            castSB[i] = reinterpret_cast<const Config::ElementSFType *>(hostScaleB[i]);
        }
        ok = cudaMemcpyAsync(b, castB.data(), sizeof(void *) * groups,
                             cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemcpyAsync(sb, castSB.data(), sizeof(void *) * groups,
                             cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemcpyAsync(alpha, hostAlpha, sizeof(void *) * groups,
                             cudaMemcpyHostToDevice, stream) == cudaSuccess;
        if (ok) {
            scratch->staticMetadataReady = true;
            scratch->staticGroups = groups;
            scratch->staticN = n;
            scratch->staticK = k;
        }
    }
    if (ok) {
        const int threads = std::min(256, groups);
        SetupDeviceRouteMetadataSm120<
            Config::ElementType, Config::ElementD, Config::ElementSFType,
            StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB, ScaleConfig>
            <<<(groups + threads - 1) / threads, threads, 0, stream>>>(
                aBase, scaleABase, dBase, expertOffsets, blockscaleOffsets,
                a, d, sa, rows, shapes, strideA, strideB, strideD,
                layoutA, layoutB, groups, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        typename Kernel::MainloopArguments mainloop{
            a, strideA, b, strideB, sa, layoutA, sb, layoutB};
        typename Kernel::EpilogueArguments epilogue{{}, nullptr, strideD, d, strideD};
        epilogue.thread.alpha_ptr_array = const_cast<float **>(alpha);
        epilogue.thread.dAlpha = {_0{}, _0{}, 1};
        epilogue.thread.beta = 0.0f;
        cutlass::KernelHardwareInfo hw;
        hw.device_id = scratch->device;
        hw.sm_count = scratch->smCount;
        typename Kernel::TileSchedulerArguments scheduler;
        using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
        scheduler.raster_order = Raster::AlongM;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {groups, reinterpret_cast<ShapeType *>(shapes), nullptr},
            mainloop, epilogue, hw, scheduler};
        Gemm gemm;
        const size_t workspaceBytes = Gemm::get_workspace_size(args);
        if (FastllmCudaGraphIsCapturing() &&
            (scratch->workspaceCapacity < workspaceBytes ||
             (workspaceBytes > 0 && scratch->workspace == nullptr))) {
            return false;
        }
        ok = EnsureBufferSm120(
                 scratch->workspace, scratch->workspaceCapacity, workspaceBytes) &&
             gemm.can_implement(args) == cutlass::Status::kSuccess &&
             gemm.initialize(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             gemm.run(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             cudaGetLastError() == cudaSuccess;
    }
    return ok;
}
} // namespace fastllm_nvfp4_moe_sm120
#endif

bool FastllmCudaNvfp4GroupedGemmSm120(
        const uint8_t *const *a, const uint8_t *const *b,
        const uint8_t *const *scaleA, const uint8_t *const *scaleB,
        const float *const *alpha, void *const *d, const int *rows,
        int groups, int n, int k, fastllm::DataType outputType,
        void *streamPtr) {
#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
    if (!a || !b || !scaleA || !scaleB || !alpha || !d || !rows ||
        groups <= 0 || groups > 256 || n <= 0 || k <= 0 ||
        n % 32 != 0 || k % 32 != 0 ||
        outputType != fastllm::DataType::BFLOAT16) return false;
    cudaStream_t stream = streamPtr ? static_cast<cudaStream_t>(streamPtr) : 0;
    return fastllm_nvfp4_moe_sm120::RunGroupedSm120(
        a, b, scaleA, scaleB, alpha, d, rows, groups, n, k, stream);
#else
    (void)a; (void)b; (void)scaleA; (void)scaleB; (void)alpha; (void)d;
    (void)rows; (void)groups; (void)n; (void)k; (void)outputType; (void)streamPtr;
#endif
    return false;
}

/**
 * 使用GPU端路由元数据执行一次SM120 NVFP4 W4A4 grouped GEMM。
 *
 * 本函数是SM120设备侧MoE路由路径的公开入口，只负责检查参数并调用
 * CUTLASS grouped GEMM，不负责生成expert路由、量化激活或决定后端生命周期。
 * 激活和输出已按expert连续排列；函数根据GPU端expertOffsets为每个expert
 * 构造独立的[M_e,N,K]问题，并通过blockscaleOffsets定位各组激活scale。
 * 所有分组共享N和K，且N、K必须为32的倍数；当前输出仅支持BF16。
 *
 * @param aBase              按expert连续排列的NVFP4 E2M1激活基址，逻辑上为
 *                           各组A_e[M_e,K]依次拼接。
 * @param scaleABase         激活对应的E4M3 block scale基址。
 * @param dBase              按expert连续写入的BF16输出基址，逻辑上为各组
 *                           D_e[M_e,N]依次拼接。
 * @param b                  各expert的CUTLASS NVFP4权重指针数组，长度为groups，
 *                           每组权重逻辑形状为[N,K]。
 * @param scaleB             各expert的E4M3权重scale指针数组，长度为groups。
 * @param alpha              各expert的FP32全局scale指针数组，长度为groups。
 * @param expertOffsets      GPU端路由行前缀和，长度为groups+1；第e组行数为
 *                           expertOffsets[e+1]-expertOffsets[e]。
 * @param blockscaleOffsets  GPU端激活scale行前缀和，长度为groups+1；各组按
 *                           CUTLASS要求独立对齐。
 * @param groups             grouped GEMM的问题数量，即routed expert数量，范围
 *                           为1到256。
 * @param n                  每组GEMM的N维，即输出特征数，必须为32的倍数。
 * @param k                  每组GEMM的K维，即输入特征数，必须为32的倍数。
 * @param outputType         输出数据类型，当前只接受BFLOAT16。
 * @param streamPtr          可选CUDA stream；为空时使用默认stream。
 * @return 参数合法、SM120 CUTLASS后端已编译且metadata准备和GEMM启动均
 *         成功时返回true；否则返回false。
 */
bool FastllmCudaNvfp4GroupedGemmSm120DeviceRoutes(
        const uint8_t *aBase, const uint8_t *scaleABase, void *dBase,
        const uint8_t *const *b, const uint8_t *const *scaleB,
        const float *const *alpha, const int *expertOffsets,
        const int *blockscaleOffsets, int groups, int n, int k,
        fastllm::DataType outputType, void *streamPtr) {
#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
    if (!aBase || !scaleABase || !dBase || !b || !scaleB || !alpha ||
        !expertOffsets || !blockscaleOffsets || groups <= 0 || groups > 256 ||
        n <= 0 || k <= 0 || n % 32 != 0 || k % 32 != 0 ||
        outputType != fastllm::DataType::BFLOAT16) return false;
    cudaStream_t stream = streamPtr ? static_cast<cudaStream_t>(streamPtr) : 0;
    return fastllm_nvfp4_moe_sm120::RunGroupedDeviceRoutesSm120(
        aBase, scaleABase, dBase, b, scaleB, alpha,
        expertOffsets, blockscaleOffsets, groups, n, k, stream);
#else
    (void)aBase; (void)scaleABase; (void)dBase; (void)b; (void)scaleB;
    (void)alpha; (void)expertOffsets; (void)blockscaleOffsets; (void)groups;
    (void)n; (void)k; (void)outputType; (void)streamPtr;
#endif
    return false;
}
