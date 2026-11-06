/* Apache-2.0: adapted from vLLM nvfp4_scaled_mm_sm120_kernels.cu. */
#include "fastllm-cuda.cuh"

#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
using namespace cute;

bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void TraceFailure(const char *stage, cutlass::Status status) {
    if (TraceEnabled()) {
        std::fprintf(stderr, "[fastllm][nvfp4][sm120] stage=%s status=%s\n",
                     stage, cutlassGetStatusString(status));
    }
}

struct ConfigMedium {
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileScheduler = void;
    using TileShape = Shape<_128, _128, _128>;
};
struct ConfigLarge {
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using TileShape = Shape<_256, _128, _128>;
};

template <typename Config, typename Out>
struct KernelConfig {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using Accumulator = float;
    using Cluster = Shape<_1, _1, _1>;
    static constexpr int AlignAB = 32;
    static constexpr int AlignD = 128 / cutlass::sizeof_bits<Out>::value;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        typename Config::TileShape, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator, Accumulator, Out, LayoutD, AlignD,
        Out, LayoutD, AlignD, typename Config::EpilogueSchedule>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA, LayoutA, AlignAB, ElementB, LayoutB, AlignAB,
        Accumulator, typename Config::TileShape, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        typename Config::KernelSchedule>::CollectiveOp;
    using Universal = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue,
        typename Config::TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Universal>;
};

template <typename Config, typename Out>
bool Run(const uint8_t *a, const uint8_t *b, const uint8_t *scaleA,
         const uint8_t *scaleB, const float *alpha, Out *d,
         int m, int n, int k, cudaStream_t stream) {
    using KC = KernelConfig<Config, Out>;
    using Gemm = typename KC::Gemm;
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using BlockScale = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto strideA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto strideB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto strideD = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layoutA = BlockScale::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layoutB = BlockScale::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {reinterpret_cast<const ElementA *>(a), strideA,
         reinterpret_cast<const ElementB *>(b), strideB,
         reinterpret_cast<const cutlass::float_ue4m3_t *>(scaleA), layoutA,
         reinterpret_cast<const cutlass::float_ue4m3_t *>(scaleB), layoutB},
        {{}, d, strideD, d, strideD}};
    args.epilogue.thread.alpha_ptr = alpha;
    Gemm gemm;
    size_t bytes = Gemm::get_workspace_size(args);
    void *workspace = bytes == 0 ? nullptr : FastllmCudaMalloc(bytes);
    if (bytes != 0 && workspace == nullptr) {
        if (TraceEnabled()) {
            std::fprintf(stderr, "[fastllm][nvfp4][sm120] stage=workspace status=allocation_failed bytes=%zu\n",
                         bytes);
        }
        return false;
    }
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) TraceFailure("can_implement", status);
    if (status == cutlass::Status::kSuccess) {
        status = gemm.initialize(args, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            TraceFailure("initialize", status);
        }
    }
    if (status == cutlass::Status::kSuccess) {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) TraceFailure("run", status);
    }
    if (workspace != nullptr) FastllmCudaFree(workspace);
    const cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess && TraceEnabled()) {
        std::fprintf(stderr, "[fastllm][nvfp4][sm120] stage=launch status=%s\n",
                     cudaGetErrorString(cudaStatus));
    }
    return status == cutlass::Status::kSuccess && cudaStatus == cudaSuccess;
}

template <typename Out>
bool Dispatch(const uint8_t *a, const uint8_t *b, const uint8_t *scaleA,
              const uint8_t *scaleB, const float *alpha, Out *d,
              int m, int n, int k, cudaStream_t stream) {
    if (m <= 256) return Run<ConfigMedium>(a, b, scaleA, scaleB, alpha, d, m, n, k, stream);
    return Run<ConfigLarge>(a, b, scaleA, scaleB, alpha, d, m, n, k, stream);
}
} // namespace
#endif

bool FastllmCudaNvfp4CutlassGemmSm120(
        const uint8_t *a, const uint8_t *b, const uint8_t *scaleA,
        const uint8_t *scaleB, const float *alpha, void *d,
        fastllm::DataType outputType, int m, int n, int k, void *streamPtr) {
#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
    if (a == nullptr || b == nullptr || scaleA == nullptr || scaleB == nullptr ||
        alpha == nullptr || d == nullptr || m <= 0 || n % 32 != 0 || k % 32 != 0) return false;
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (outputType == fastllm::DataType::FLOAT16)
        return Dispatch(a, b, scaleA, scaleB, alpha, static_cast<cutlass::half_t *>(d), m, n, k, stream);
    if (outputType == fastllm::DataType::BFLOAT16)
        return Dispatch(a, b, scaleA, scaleB, alpha, static_cast<cutlass::bfloat16_t *>(d), m, n, k, stream);
#else
    (void)a; (void)b; (void)scaleA; (void)scaleB; (void)alpha; (void)d;
    (void)outputType; (void)m; (void)n; (void)k; (void)streamPtr;
#endif
    return false;
}
