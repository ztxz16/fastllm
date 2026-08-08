/*
 * Ada SM89 FP8 scaled GEMM for FastLLM.
 * Kernel configuration and epilogue structure are adapted from vLLM's
 * c2x CUTLASS scaled-mm implementation (Apache-2.0).
 */

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && defined(FASTLLM_CUTLASS_FP8_ENABLE_SM89)

// CUTLASS 2.x headers are order-sensitive. Keep this order aligned with the
// upstream scaled-mm implementation.
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

namespace fastllm_cuda_cutlass_fp8_sm89 {

using namespace cute;

// CUTLASS 2.x emits the Ada FP8 mma.sync instruction.  Keep the device guard
// exact so a fat binary can never launch this specialization on Hopper or a
// pre-Ada GPU by accident.
template <typename Kernel>
struct FastllmEnableSm89 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
        Kernel::invoke(std::forward<Args>(args)...);
#else
        printf("FastLLM CUTLASS scaled-mm kernel only supports sm89.\n");
        asm("trap;");
#endif
#endif
    }
};

template <typename ElementD, typename OutputTileThreadMap>
struct FastllmScaledEpilogue {
private:
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ScaleA = cutlass::epilogue::threadblock::VisitorColBroadcast<
        OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>, false>;
    using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>, false>;
    using ComputeB = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleBAccum =
        cutlass::epilogue::threadblock::Sm80EVT<ComputeB, ScaleB, Accum>;
    using ComputeA = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;

public:
    using EVTCompute =
        cutlass::epilogue::threadblock::Sm80EVT<ComputeA, ScaleA, ScaleBAccum>;
    using ArgumentType = typename EVTCompute::Arguments;

    static ArgumentType prepare_args(const float *aScales,
                                     const float *bScales) {
        typename ScaleA::Arguments aArgs{
            aScales, 0.0f, Stride<Int<1>, Int<0>, Int<0>>{}};
        typename ScaleB::Arguments bArgs{
            bScales, 0.0f, Stride<Int<0>, Int<1>, Int<0>>{}};
        typename ScaleBAccum::Arguments scaledAccumArgs{bArgs, {}, {}};
        return ArgumentType{aArgs, scaledAccumArgs, {}};
    }
};

template <typename ElementD_, typename TileShape, typename WarpShape,
          int MainLoopStages,
          typename MathOperator = cutlass::arch::OpMultiplyAdd>
struct FastllmSm89ScaledGemm {
    using ElementAB = cutlass::float_e4m3_t;
    using ElementD = ElementD_;
    using ElementAcc = float;
    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            TileShape, WarpShape, float, 4, 1>;
    using Epilogue = FastllmScaledEpilogue<ElementD, OutputTileThreadMap>;
    using EVTCompute = typename Epilogue::EVTCompute;
    using D = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementD,
        cutlass::FloatRoundStyle::round_to_nearest,
        Stride<int64_t, Int<1>, Int<0>>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

    static constexpr int AlignmentAB =
        128 / cutlass::sizeof_bits<ElementAB>::value;
    static constexpr int AlignmentCD = 4;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using KernelType = FastllmEnableSm89<
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            ElementAB, cutlass::layout::RowMajor,
            cutlass::ComplexTransform::kNone, AlignmentAB,
            ElementAB, cutlass::layout::ColumnMajor,
            cutlass::ComplexTransform::kNone, AlignmentAB,
            float, cutlass::layout::RowMajor, AlignmentCD,
            ElementAcc, float, cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm89, TileShape, WarpShape, InstructionShape,
            EVTD, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
            MainLoopStages, MathOperator, 1>::GemmKernel>;
    using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

struct FastllmSm89Scratch {
    cutlass::float_e4m3_t *input = nullptr;
    float *inputScales = nullptr;
    void *workspace = nullptr;
    size_t inputElements = 0;
    size_t scaleElements = 0;
    size_t workspaceBytes = 0;
    // Captured graphs retain addresses.  Old allocations are deliberately
    // retained when a later warmup grows a buffer.
    std::vector<cutlass::float_e4m3_t*> retiredInputs;
    std::vector<float*> retiredScales;
    std::vector<void*> retiredWorkspaces;
};

static std::mutex g_scratchMutex;
static std::map<int, FastllmSm89Scratch> g_scratchByDevice;

static bool FastllmSm89IsCapturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(stream, &status);
    return state == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

static bool FastllmSm89EnsureInputScratch(
    int rows, int cols, cudaStream_t stream,
    cutlass::float_e4m3_t *&input, float *&scales) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    size_t inputElements = (size_t)rows * cols;
    size_t scaleElements = (size_t)rows;
    std::lock_guard<std::mutex> guard(g_scratchMutex);
    FastllmSm89Scratch &scratch = g_scratchByDevice[device];
    if (scratch.inputElements < inputElements) {
        if (FastllmSm89IsCapturing(stream)) {
            return false;
        }
        cutlass::float_e4m3_t *next =
            static_cast<cutlass::float_e4m3_t *>(
                FastllmCudaMalloc(
                    inputElements * sizeof(cutlass::float_e4m3_t)));
        if (next == nullptr) {
            return false;
        }
        if (scratch.input != nullptr) {
            scratch.retiredInputs.push_back(scratch.input);
        }
        scratch.input = next;
        scratch.inputElements = inputElements;
    }
    if (scratch.scaleElements < scaleElements) {
        if (FastllmSm89IsCapturing(stream)) {
            return false;
        }
        float *next = static_cast<float *>(
            FastllmCudaMalloc(scaleElements * sizeof(float)));
        if (next == nullptr) {
            return false;
        }
        if (scratch.inputScales != nullptr) {
            scratch.retiredScales.push_back(scratch.inputScales);
        }
        scratch.inputScales = next;
        scratch.scaleElements = scaleElements;
    }
    input = scratch.input;
    scales = scratch.inputScales;
    return input != nullptr && scales != nullptr;
}

static bool FastllmSm89EnsureWorkspace(
    size_t bytes, cudaStream_t stream, void *&workspace) {
    workspace = nullptr;
    if (bytes == 0) {
        return true;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    std::lock_guard<std::mutex> guard(g_scratchMutex);
    FastllmSm89Scratch &scratch = g_scratchByDevice[device];
    if (scratch.workspaceBytes < bytes) {
        if (FastllmSm89IsCapturing(stream)) {
            return false;
        }
        void *next = FastllmCudaMalloc(bytes);
        if (next == nullptr) {
            return false;
        }
        if (scratch.workspace != nullptr) {
            scratch.retiredWorkspaces.push_back(scratch.workspace);
        }
        scratch.workspace = next;
        scratch.workspaceBytes = bytes;
    }
    workspace = scratch.workspace;
    return true;
}

template <typename T>
__device__ __forceinline__ float FastllmSm89ToFloat(T value);

template <>
__device__ __forceinline__ float FastllmSm89ToFloat(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float FastllmSm89ToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__global__ void __launch_bounds__(256) FastllmSm89QuantPerRowKernel(
    const T *__restrict__ input, uint8_t *__restrict__ quant,
    float *__restrict__ scales, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    constexpr int kWarps = 8;
    __shared__ float warpMax[kWarps];
    __shared__ float rowScale;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const T *rowInput = input + (size_t)row * cols;
    uint8_t *rowQuant = quant + (size_t)row * cols;

    float maxAbs = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        maxAbs = fmaxf(maxAbs, fabsf(FastllmSm89ToFloat(rowInput[col])));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxAbs = fmaxf(maxAbs,
                       __shfl_down_sync(0xffffffffu, maxAbs, offset));
    }
    if (lane == 0) {
        warpMax[warp] = maxAbs;
    }
    __syncthreads();
    if (warp == 0) {
        maxAbs = lane < kWarps ? warpMax[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxAbs = fmaxf(maxAbs,
                           __shfl_down_sync(0xffffffffu, maxAbs, offset));
        }
        if (lane == 0) {
            rowScale = maxAbs > 0.0f ? maxAbs * (1.0f / 448.0f) : 1.0f;
            scales[row] = rowScale;
        }
    }
    __syncthreads();
    float invScale = 1.0f / rowScale;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = FastllmSm89ToFloat(rowInput[col]) * invScale;
        rowQuant[col] = (uint8_t)__nv_cvt_float_to_fp8(
            value, __NV_SATFINITE, __NV_E4M3);
    }
}

template <typename T>
__global__ void FastllmSm89AddFloatBiasKernel(
    T *output, const float *bias, size_t elements, int columns) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += (size_t)blockDim.x * gridDim.x) {
        float value = (float)output[index] + bias[index % columns];
        output[index] = T(value);
    }
}

template <typename Gemm>
static bool FastllmRunSm89Gemm(
    const cutlass::float_e4m3_t *input,
    const cutlass::float_e4m3_t *weight,
    const float *inputScales, const float *weightScales,
    typename Gemm::ElementD *output,
    int rows, int outFeatures, int inFeatures, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problemSize{rows, outFeatures, inFeatures};
    int64_t lda = inFeatures;
    int64_t ldb = inFeatures;
    int64_t ldc = outFeatures;
    using StrideD = Stride<int64_t, Int<1>, Int<0>>;
    StrideD outputStride{ldc, Int<1>{}, Int<0>{}};
    typename Gemm::D::Arguments dArgs{output, outputStride};
    auto scaleArgs = Gemm::Epilogue::prepare_args(inputScales, weightScales);
    typename Gemm::EVTD::Arguments epilogueArgs{scaleArgs, dArgs};
    typename Gemm::Op::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
        problemSize,
        1,
        epilogueArgs,
        input,
        weight,
        nullptr,
        nullptr,
        0, 0, 0, 0,
        lda, ldb, ldc, ldc};

    typename Gemm::Op gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    size_t workspaceBytes = gemm.get_workspace_size(args);
    void *workspace = nullptr;
    if (!FastllmSm89EnsureWorkspace(workspaceBytes, stream, workspace)) {
        return false;
    }
    return gemm(args, workspace, stream) == cutlass::Status::kSuccess;
}

template <typename Primary, typename Fallback>
static bool FastllmRunSm89WithFallback(
    const cutlass::float_e4m3_t *input,
    const cutlass::float_e4m3_t *weight,
    const float *inputScales, const float *weightScales,
    typename Primary::ElementD *output,
    int rows, int outFeatures, int inFeatures, cudaStream_t stream) {
    int device = 0;
    int maxShared = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(
        &maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if ((int)sizeof(typename Primary::KernelType::SharedStorage) <= maxShared) {
        return FastllmRunSm89Gemm<Primary>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    return FastllmRunSm89Gemm<Fallback>(
        input, weight, inputScales, weightScales, output,
        rows, outFeatures, inFeatures, stream);
}

template <typename OutType, typename TileShape, typename WarpShape, int Stages,
          typename MathOperator = cutlass::arch::OpMultiplyAddFastAccum>
using FastllmSm89Gemm = FastllmSm89ScaledGemm<
    OutType, TileShape, WarpShape, Stages, MathOperator>;

template <typename OutType, typename Gemm>
static bool FastllmRunSelectedSm89(
    const cutlass::float_e4m3_t *input,
    const cutlass::float_e4m3_t *weight,
    const float *inputScales, const float *weightScales,
    OutType *output, int rows, int outFeatures, int inFeatures,
    cudaStream_t stream) {
    using Fallback = FastllmSm89Gemm<
        OutType, cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>, 5,
        cutlass::arch::OpMultiplyAdd>;
    return FastllmRunSm89WithFallback<Gemm, Fallback>(
        input, weight, inputScales, weightScales, output,
        rows, outFeatures, inFeatures, stream);
}

static uint32_t FastllmNextPowerOfTwo(uint32_t value) {
    if (value <= 1) {
        return 1;
    }
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

template <typename OutType>
static bool FastllmDispatchSm89(
    const cutlass::float_e4m3_t *input,
    const cutlass::float_e4m3_t *weight,
    const float *inputScales, const float *weightScales,
    OutType *output, int rows, int outFeatures, int inFeatures,
    cudaStream_t stream) {
    uint32_t np2 = FastllmNextPowerOfTwo((uint32_t)outFeatures);
    if (rows <= 16) {
        using Warp = cutlass::gemm::GemmShape<16, 64, 64>;
        if (np2 <= 8192) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<16, 64, 128>, Warp, 5>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        if (np2 <= 24576) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<16, 128, 64>, Warp, 5>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<32, 64, 128>, Warp, 5>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (rows <= 32) {
        if (np2 <= 8192) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<32, 64, 128>,
                cutlass::gemm::GemmShape<16, 64, 64>, 5>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        if (np2 <= 16384) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<32, 128, 128>,
                cutlass::gemm::GemmShape<32, 64, 64>, 4>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<32, 64, 128>,
            cutlass::gemm::GemmShape<16, 64, 64>, 5>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (rows <= 64) {
        if (np2 <= 8192) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<64, 64, 128>,
                cutlass::gemm::GemmShape<32, 64, 64>, 5,
                cutlass::arch::OpMultiplyAdd>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        if (np2 <= 16384) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<64, 128, 128>,
                cutlass::gemm::GemmShape<64, 64, 64>, 3>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<64, 64, 128>,
            cutlass::gemm::GemmShape<32, 64, 64>, 5,
            cutlass::arch::OpMultiplyAdd>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (rows <= 128) {
        if (np2 <= 8192) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<64, 128, 128>,
                cutlass::gemm::GemmShape<64, 64, 64>, 3>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        if (np2 <= 16384) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<128, 128, 64>,
                cutlass::gemm::GemmShape<64, 64, 64>, 5>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<128, 64, 128>,
            cutlass::gemm::GemmShape<64, 64, 64>, 3>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (rows <= 256) {
        if (np2 <= 4096) {
            using Gemm = FastllmSm89Gemm<
                OutType, cutlass::gemm::GemmShape<64, 128, 128>,
                cutlass::gemm::GemmShape<64, 64, 64>, 3>;
            return FastllmRunSelectedSm89<OutType, Gemm>(
                input, weight, inputScales, weightScales, output,
                rows, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>, 5>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (np2 <= 4096) {
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>, 5>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    if (np2 <= 8192) {
        using Gemm = FastllmSm89Gemm<
            OutType, cutlass::gemm::GemmShape<256, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>, 3>;
        return FastllmRunSelectedSm89<OutType, Gemm>(
            input, weight, inputScales, weightScales, output,
            rows, outFeatures, inFeatures, stream);
    }
    using Gemm = FastllmSm89Gemm<
        OutType, cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>, 5>;
    return FastllmRunSelectedSm89<OutType, Gemm>(
        input, weight, inputScales, weightScales, output,
        rows, outFeatures, inFeatures, stream);
}

} // namespace fastllm_cuda_cutlass_fp8_sm89

#endif

bool FastllmCudaCutlassLinearFP8E4M3PerChannel(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output,
    int n, int m, int k) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && defined(FASTLLM_CUTLASS_FP8_ENABLE_SM89)
    using namespace fastllm_cuda_cutlass_fp8_sm89;
    if (n <= 0 || m <= 0 || k <= 0 || (m % 16) != 0 || (k % 4) != 0 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        weight.cudaData == nullptr ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.dims.size() != 2 || weight.dims[0] != k ||
        weight.dims[1] != m || weight.blockK != 1 ||
        weight.blockM < m || weight.scales.size() != (size_t)k ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType ||
        (bias.dims.size() > 0 &&
         (bias.dataType != fastllm::DataType::FLOAT32 ||
          bias.cudaData == nullptr))) {
        return false;
    }
    int device = 0;
    cudaDeviceProp props;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&props, device) != cudaSuccess ||
        props.major * 10 + props.minor != 89) {
        return false;
    }

    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    if (weight.extraCudaData.empty() || weight.extraCudaData[0] == nullptr) {
        return false;
    }
    cudaStream_t stream = cudaStreamPerThread;
    cutlass::float_e4m3_t *quantInput = nullptr;
    float *inputScales = nullptr;
    if (!FastllmSm89EnsureInputScratch(
            n, m, stream, quantInput, inputScales)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSm89QuantPerRowKernel<<<n, 256, 0, stream>>>(
            (const half *)inputData, (uint8_t *)quantInput,
            inputScales, n, m);
    } else {
        FastllmSm89QuantPerRowKernel<<<n, 256, 0, stream>>>(
            (const __nv_bfloat16 *)inputData, (uint8_t *)quantInput,
            inputScales, n, m);
    }
    bool ok = cudaGetLastError() == cudaSuccess;
    const auto *cutlassWeight =
        reinterpret_cast<const cutlass::float_e4m3_t *>(weight.cudaData);
    const float *weightScales =
        reinterpret_cast<const float *>(weight.extraCudaData[0]);
    if (ok && input.dataType == fastllm::DataType::FLOAT16) {
        ok = FastllmDispatchSm89(
            quantInput, cutlassWeight, inputScales, weightScales,
            (cutlass::half_t *)outputData, n, k, m, stream);
    } else if (ok) {
        ok = FastllmDispatchSm89(
            quantInput, cutlassWeight, inputScales, weightScales,
            (cutlass::bfloat16_t *)outputData, n, k, m, stream);
    }
    if (ok && !bias.dims.empty()) {
        size_t elements = (size_t)n * k;
        int threads = 256;
        int blocks = (int)std::min<size_t>(
            4096, (elements + threads - 1) / threads);
        if (input.dataType == fastllm::DataType::FLOAT16) {
            FastllmSm89AddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (cutlass::half_t *)outputData,
                (const float *)bias.cudaData, elements, k);
        } else {
            FastllmSm89AddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (cutlass::bfloat16_t *)outputData,
                (const float *)bias.cudaData, elements, k);
        }
        ok = cudaGetLastError() == cudaSuccess;
    }
    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return ok;
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    return false;
#endif
}
