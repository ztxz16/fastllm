#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace {
using Bf16 = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Column = cutlass::layout::ColumnMajor;
using Bf16Output = cutlass::epilogue::thread::LinearCombination<Bf16, 8, float, float>;

struct SoftcapParams { float limit; };

template <typename Fragment> struct Bf16Softcap {
    using Arguments = SoftcapParams;

    CUTLASS_HOST_DEVICE Fragment operator()(const Fragment &acc, const Arguments &params) const {
        Fragment result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Fragment::kElements; ++i) {
            // The checkpoint rounds the projection to BF16 before softcapping.
            result[i] = params.limit * tanhf(float(Bf16(acc[i])) / params.limit);
        }
        return result;
    }
};
using CappedOutput = cutlass::epilogue::thread::LinearCombinationGeneric<
    Bf16Softcap, float, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;

template <int Channels, class OutputOp, class Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          int TileN = 64, int WarpM = Channels / 2, int WarpN = 32, int Stages = 3>
using ProjectionGemm = cutlass::gemm::device::Gemm<Bf16, Row, Bf16, Column, typename OutputOp::ElementOutput, Column,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<Channels, TileN, 64>, cutlass::gemm::GemmShape<WarpM, WarpN, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>, OutputOp, Swizzle, Stages>;

template <int Channels, class OutputOp>
bool project(const void *input, const void *weight, typename OutputOp::ElementOutput *output,
             int rows, int width, int columns, const typename OutputOp::Params &epilogue) {
    using Gemm = ProjectionGemm<Channels, OutputOp>;
    typename Gemm::Arguments args({columns, rows, width}, {static_cast<const Bf16 *>(weight), width},
        {static_cast<const Bf16 *>(input), width}, {output, columns}, {output, columns}, epilogue);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    auto status = gemm(args, nullptr, cudaStreamPerThread);
    if (status != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(status));
    return true;
}

template <class OutputOp>
bool projectVocab(const void *input, const void *weight, typename OutputOp::ElementOutput *output,
                  int rows, int width, int vocab, const typename OutputOp::Params &epilogue) {
    constexpr int elements = OutputOp::kCount;
    constexpr size_t alignment = elements * sizeof(typename OutputOp::ElementOutput);
    if (rows <= 0 || width <= 0 || width % 8 || vocab <= 0 || vocab % elements ||
        ((uintptr_t(input) | uintptr_t(weight)) & 15) || uintptr_t(output) % alignment)
        return false;
    if (rows > 32)
        return project<64, OutputOp>(input, weight, output, rows, width, vocab, epilogue);
    return project<32, OutputOp>(input, weight, output, rows, width, vocab, epilogue);
}

// Each CTA gets a tile-local view of one projection. Q, K and V can have
// different input matrices and keep their ordinary, independently strided outputs.
struct TileOrigin : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> {
    CUTLASS_DEVICE static cutlass::gemm::GemmCoord get_tile_offset(int) { return {0, 0, 0}; }
};

struct AttentionSegment {
    const Bf16 *input, *weight;
    Bf16 *output;
    int columns, firstTile;
};

template<int Groups> struct AttentionSegments { AttentionSegment group[Groups]; };

// A contracting projection with at most one MMA row tile benefits from more CTAs across channels.
using NarrowOutput = cutlass::epilogue::thread::LinearCombination<Bf16, 2, float, float>;
template<int Rows>
using AttentionGemm = std::conditional_t<Rows == 16, ProjectionGemm<16, NarrowOutput, TileOrigin, 32, 16, 8, 4>,
                                         ProjectionGemm<Rows, Bf16Output, TileOrigin>>;
constexpr int smallAttentionTileRows = 32;
template<int TileRows>
__host__ __device__ constexpr bool overlapAttentionProjection(int rows) {
    return rows <= smallAttentionTileRows || rows < TileRows;
}

template<int TileRows, int Groups>
__global__ void attentionTiles(AttentionSegments<Groups> segments, int rows, int width) {
    using Kernel = typename AttentionGemm<TileRows>::GemmKernel;
    int group = 0;
    if constexpr (Groups == 3)
        group = (int(blockIdx.y) >= segments.group[1].firstTile) + (int(blockIdx.y) >= segments.group[2].firstTile);
    const auto &s = segments.group[group];
    constexpr int tileWidth = Kernel::Mma::Shape::kN;
    const int row = blockIdx.x * TileRows, col = (int(blockIdx.y) - s.firstTile) * tileWidth;
    const int tileColumns = min(tileWidth, s.columns - col);
    const auto *weight = s.weight + size_t(col) * width;
    // These weight cache lines are independent of the preceding activation.
    const int prefetchColumn = threadIdx.x / 2;
    if (prefetchColumn < tileColumns && int(threadIdx.x % 2) * 32 < width) {
        const auto *line = weight + size_t(prefetchColumn) * width + (threadIdx.x % 2) * 32;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(line));
    }
    if (overlapAttentionProjection<TileRows>(rows)) fastllm::cuda_launch::DependencyStart();
    auto *out = s.output + size_t(row) * s.columns + col;
    typename Kernel::Params params({min(TileRows, rows - row), tileColumns, width}, {1, 1, 1},
        {const_cast<Bf16 *>(s.input + size_t(row) * width), width}, {const_cast<Bf16 *>(weight), width},
        {out, s.columns}, {out, s.columns}, {1.f, 0.f});
    extern __shared__ int storage[];
    Kernel{}(params, *reinterpret_cast<typename Kernel::SharedStorage *>(storage));
}

template<int TileRows, int Groups>
void launchAttentionTiles(const AttentionSegments<Groups> &segments, int rows, int width, int tiles) {
    using Kernel = typename AttentionGemm<TileRows>::GemmKernel;
    constexpr size_t shared = sizeof(typename Kernel::SharedStorage);
    if constexpr (shared >= 48 * 1024) {
        if (cudaFuncSetAttribute(attentionTiles<TileRows, Groups>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 int(shared)) != cudaSuccess)
            throw std::runtime_error("BF16 attention projection shared-memory configuration failed");
    }
    const dim3 grid(1 + (rows - 1) / TileRows, tiles);
    // Early consumers can overlap a small tile or a partially occupied large
    // tile. For larger, full tiles, let the projection retire first.
    if (!overlapAttentionProjection<TileRows>(rows)) {
        attentionTiles<TileRows, Groups><<<grid, Kernel::kThreadCount, shared, cudaStreamPerThread>>>(segments, rows, width);
        if (cudaGetLastError() == cudaSuccess) return;
    } else if (fastllm::cuda_launch::Launch<attentionTiles<TileRows, Groups>>(
            grid, Kernel::kThreadCount, shared, cudaStreamPerThread, segments, rows, width)) {
        return;
    }
    throw std::runtime_error("BF16 attention projection launch failed");
}

template<int Groups>
bool attentionProjections(const void *const *input, const void *weight, void *output,
                          int rows, int width, const int *columns) {
    if (rows <= 0 || width <= 0 || width % 8 || ((uintptr_t(weight) | uintptr_t(output)) & 15)) return false;
    AttentionSegments<Groups> segments;
    size_t offset = 0;
    const bool narrow = Groups == 1 && rows <= 16 && columns[0] < width && columns[0] <= 32 * 65535;
    const int tileWidth = narrow ? 32 : 64;
    int tiles = 0;
    for (int i = 0; i < Groups; ++i) {
        if (columns[i] <= 0 || columns[i] % 8 || (uintptr_t(input[i]) & 15)) return false;
        const int count = 1 + (columns[i] - 1) / tileWidth;
        if (count > 65535 - tiles) return false;
        segments.group[i] = {static_cast<const Bf16 *>(input[i]), static_cast<const Bf16 *>(weight) + offset * width,
                             static_cast<Bf16 *>(output) + offset * rows, columns[i], tiles};
        tiles += count;
        offset += columns[i];
    }
    if constexpr (Groups == 1) {
        if (narrow) {
            launchAttentionTiles<16>(segments, rows, width, tiles);
            return true;
        }
    }
    if (rows > smallAttentionTileRows && columns[0] / 2 >= width)
        launchAttentionTiles<64>(segments, rows, width, tiles);
    else
        launchAttentionTiles<smallAttentionTileRows>(segments, rows, width, tiles);
    return true;
}

bool attentionProjection(const void *input, const void *weight, void *output,
                         int rows, int width, int columns) {
    // One query-row tile uses the same reduction as single-row decode.
    constexpr int rowsPerTile = AttentionGemm<smallAttentionTileRows>::GemmKernel::Mma::Shape::kN;
    if (rows > rowsPerTile && columns < width &&
        FastllmCudaBf16ProjectionGemm(input, weight, output, rows, width, columns, 1)) return true;
    if (attentionProjections<1>(&input, weight, output, rows, width, &columns)) return true;
    if (rows <= 0 || width <= 0 || columns <= 0 || width % 8 || columns % 8 ||
        ((uintptr_t(input) | uintptr_t(weight) | uintptr_t(output)) & 15)) return false;
    if (rows > 32 && columns / 2 >= width)
        return project<64, Bf16Output>(input, weight, static_cast<Bf16 *>(output), rows, width, columns, {1.f, 0.f});
    return project<32, Bf16Output>(input, weight, static_cast<Bf16 *>(output), rows, width, columns, {1.f, 0.f});
}

bool cappedLogits(const void *input, const void *weight, float *logits,
                  int rows, int width, int vocab, float limit) {
    if (!(limit > 0)) return false;
    CappedOutput::Params epilogue;
    epilogue.limit = limit;
    return projectVocab<CappedOutput>(input, weight, logits, rows, width, vocab, epilogue);
}

bool draftLogits(const void *input, const void *weight, void *output,
                 int rows, int width, int vocab) {
    using DraftOutput = cutlass::epilogue::thread::LinearCombination<Bf16, 2, float, float>;
    return projectVocab<DraftOutput>(input, weight, static_cast<Bf16 *>(output), rows, width, vocab, {1.f, 0.f});
}
} // namespace

const FastllmBf16MmaProjections *FastllmCudaGetBf16MmaProjections() {
    static constexpr FastllmBf16MmaProjections ops{attentionProjection, draftLogits, cappedLogits, attentionProjections<3>};
    return FastllmCudaRuntimeArch() == 120 ? &ops : nullptr;
}
