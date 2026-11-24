// Expert-sorted SM120 grouped GEMM for Inkling NVFP4 and MXFP8 projections.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm { struct CudaMarlinMxfp8Weights; }
namespace fastllm::cuda::grouped_sm120 {
struct Scratch { size_t metadata, gemm; };

// Compact codes and tiled scales use different row origins for each expert.
__host__ __device__ inline size_t scaleRow(size_t start, int expert, int rank) {
    return (start + 127 * size_t(expert)) / 128 * 128 + rank;
}
inline size_t scaleRows(int routes, int experts) {
    return scaleRow(size_t(routes) + 127, experts, 0);
}
__host__ __device__ inline size_t scaleOffset(size_t row, int group, int groups) {
    return row / 128 * 128 * groups + (row % 32 * 4 + row % 128 / 32) * 4 + group / 4 * 512 + group % 4;
}
cudaError_t prefixRouteCounts(int *offsets, int experts, cudaStream_t stream);

// The size query and the launch bind the same CUTLASS descriptor arrays.
template<class Gemm> struct Metadata {
    using Kernel = typename Gemm::GemmKernel;
    using Mainloop = typename Kernel::CollectiveMainloop;
    using Shape = typename Kernel::ProblemShape::UnderlyingProblemShape;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutA = typename Mainloop::InternalLayoutSFA;
    using LayoutB = typename Mainloop::InternalLayoutSFB;
    using Element = typename Mainloop::ElementA;
    using Scale = typename Mainloop::ElementSF;
    using Output = typename Kernel::ElementD;

    Metadata(void *storage, int groups) : base(static_cast<char *>(storage)), groups(groups) {}
    static Scratch scratchSize(int groups, int smCount) {
        typename Gemm::Arguments args{};
        args.mode = decltype(args.mode)::kGrouped;
        args.problem_shape = {groups, nullptr, nullptr};
        args.hw_info.sm_count = smCount;
        return {Metadata(nullptr, groups).bytes, Gemm::get_workspace_size(args)};
    }
private:
    char *base;
    int groups;
    size_t bytes = 0;
    template<class T> T *take() {
        T *ptr = base ? reinterpret_cast<T *>(base + bytes) : nullptr;
        bytes += (sizeof(T) * groups + 15) / 16 * 16;
        return ptr;
    }
public:
    Shape *shape = take<Shape>();
    const Element **a = take<const Element *>(), **b = take<const Element *>();
    const Scale **sa = take<const Scale *>(), **sb = take<const Scale *>();
    Output **d = take<Output *>();
    StrideA *da = take<StrideA>();
    StrideB *db = take<StrideB>();
    StrideD *dd = take<StrideD>();
    LayoutA *la = take<LayoutA>();
    LayoutB *lb = take<LayoutB>();
};
} // namespace fastllm::cuda::grouped_sm120

namespace fastllm::cuda::routed_fp4 {
struct Block16;
struct Weights;
struct GroupedWeightMaps { CUtensorMap codes, scales; };
bool makeGroupedWeightMaps(GroupedWeightMaps &maps, const Weights &weights, int experts, int n, int k);
struct GroupedBuffers {
    uint2 *codes;
    uint8_t *scales;
    float *accumulators;
    void *metadata, *gemm;
};

// Expert rows are compact in the code matrix. Block scales use CUTLASS
// 128-row atoms, with a separate aligned start for each expert.
struct GroupedInputRow {
    uint2 *codes;
    uint8_t *scales;
    __device__ GroupedInputRow(const GroupedBuffers &buffers, int start, int expert, int rank, int groups) {
        codes = buffers.codes + size_t(start + rank) * groups;
        scales = buffers.scales + grouped_sm120::scaleOffset(grouped_sm120::scaleRow(start, expert, rank), 0, groups);
    }
    __device__ uint8_t &scale(int group) const { return scales[size_t(group / 4) * 512 + group % 4]; }
};

__device__ inline float scaledGroupedValue(float dot, float weightScale, float rowScale) {
    return float(__float2bfloat16_rn(dot * weightScale * rowScale));
}

inline bool useGroupedVerify(int rows, int topK, int experts, int hidden, int intermediate) {
    // A complete 128-channel stripe is read per CTA. At low route density
    // the small-row kernel costs less than sorting and scattering the batch.
    return int64_t(rows) * topK >= 2LL * experts && hidden % 128 == 0 && intermediate % 64 == 0;
}
grouped_sm120::Scratch groupedScratch(int experts, int smCount);
cudaError_t planGroupedRows(const int *ids, int *offsets, int *ranks, int experts, int routes, cudaStream_t stream);
cudaError_t packGroupedInput(const Block16 *input, const int *ids, const int *offsets, const int *ranks,
                             const GroupedBuffers &buffers, int routes, int k, int topK, cudaStream_t stream);
// Writes FP32 dots in expert order. The consumer applies the two outer scales
// and rounds each projection to BF16 before activation or route summation.
cudaError_t projectGroupedRows(const Weights &weights, const GroupedWeightMaps &maps, int *offsets,
                               const GroupedBuffers &buffers, int n, int k, int experts, int smCount, cudaStream_t stream);
cudaError_t sumGroupedRows(const float *sorted, float *output, const int *ids, const int *offsets, const int *ranks,
                           const float *weightScales, const float *rowScales, int rows, int n, int topK, cudaStream_t stream);
}

namespace fastllm::cuda::routed_mxfp8 {
struct Buffers {
    uint8_t *codes, *scales, *weight;
    __nv_bfloat16 *output;
    void *metadata, *workspace;
};
grouped_sm120::Scratch scratchSize(int experts, int smCount);
cudaError_t project(const CudaMarlinMxfp8Weights &weight, const Buffers &buffers, int *offsets,
                    int n, int k, int experts, int smCount, cudaStream_t stream);
}
