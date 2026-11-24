// Same expert-sorted, swap-AB SM120 grouped GEMM as FlashInfer/SGLang's
// inkling2_grouped_sm120. Keep FastLLM's four FP32 accumulation lanes.
#include "fastllm-grouped-sm120.cuh"
#include "fastllm-nvfp4-routed-sm120.cuh"
#include "flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh"
#include <limits>

namespace fastllm::cuda::grouped_sm120 {
namespace {
__global__ void prefixGroupedRoutes(int *offset, int experts) {
    int sum = 0;
    for (int e = 0; e < experts; ++e) {
        int n = offset[e];
        offset[e] = sum;
        sum += n;
    }
    offset[experts] = sum;
}
}

cudaError_t prefixRouteCounts(int *offsets, int experts, cudaStream_t stream) {
    prefixGroupedRoutes<<<1, 1, 0, stream>>>(offsets, experts);
    return cudaGetLastError();
}
} // namespace fastllm::cuda::grouped_sm120

namespace fastllm::cuda::routed_fp4 {
namespace {
using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;
using ShapeMNK = GroupProblemShape<Shape<int,int,int>>;
using Tile = Shape<_128,_32,_256>;
using Cluster = Shape<_1,_1,_1>;
using Fp4 = float_e2m1_t;
using Scale = float_ue4m3_t;
using Epilogue = epilogue::collective::CollectiveBuilder<arch::Sm120, arch::OpClassBlockScaledTensorOp,
    Tile, Cluster, epilogue::collective::EpilogueTileAuto, float, float, void, layout::ColumnMajor*, 0,
    float, layout::ColumnMajor*, 4, epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using StandardMainloop = gemm::collective::CollectiveBuilder<arch::Sm120, arch::OpClassBlockScaledTensorOp,
    nv_float4_t<Fp4>, layout::RowMajor*, 32, nv_float4_t<Fp4>, layout::ColumnMajor*, 32, float, Tile, Cluster,
    gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    gemm::collective::KernelScheduleAuto>::CollectiveOp;
struct PackedWeights {
    static constexpr int stripes = size<0>(Tile{}) / 16, tileBands = size<2>(Tile{}) / 64;
    template<class Params, class Coord, class Storage>
    CUTLASS_DEVICE void operator()(const Params &p, const Coord &coord, int e, int kTile, int stage, Storage &smem,
                                   uint64_t *barrier) const {
        const int stripe = e * p.stripesPerExpert + get<0>(coord) * stripes;
        auto *codes = reinterpret_cast<char *>(&smem.smem_A) + stage * stripes * tileBands * 512;
        auto *scales = reinterpret_cast<char *>(smem.smem_SFA.begin()) + stage * stripes * tileBands * 64;
        constexpr auto hint = uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
        SM90_TMA_LOAD_3D::copy(&p.maps.codes, barrier, hint, codes, 0, kTile * tileBands, stripe);
        SM90_TMA_LOAD_3D::copy(&p.maps.scales, barrier, hint, scales, 0, kTile * tileBands, stripe);
    }
    template<class Params, class Coord, class Mma, class Frag, class ScaleFrag, class KBlock, class Storage>
    CUTLASS_DEVICE void operator()(const Params &, const Coord &, const Mma &mma, Frag &frag,
                                   ScaleFrag &sf, KBlock kb, int, int stage, int tid, Storage &smem) const {
        int lane = tid % 32;
        auto rows = mma.partition_A(make_identity_tensor(select<0,2>(Tile{})));
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(frag); ++m) {
            int stripe = get<0>(rows(0,m,kb)) / 16;
            int tile = (stage * stripes + stripe) * tileBands + int(kb);
            // TMA zero-fills the last K tile, including its scale bytes.
            uint4 a = reinterpret_cast<const uint4 *>(&smem.smem_A)[tile * 32 + lane];
            unsigned sc = reinterpret_cast<const unsigned *>(smem.smem_SFA.begin())[tile * 16 + lane / 4 * 2 + (lane & 1)];
            auto reg = recast<unsigned>(frag(_,m,kb));
            reg(0) = a.x; reg(1) = a.y; reg(2) = a.z; reg(3) = a.w;
            auto scales = recast<unsigned>(sf(_,m,kb));
            scales(0) = sc;
        }
    }
};
struct Mainloop : StandardMainloop {
    template<class Base> struct WeightParams : Base {
        GroupedWeightMaps maps;
        int stripesPerExpert;
    };
    using Arguments = WeightParams<StandardMainloop::Arguments>;
    using Params = WeightParams<StandardMainloop::Params>;
    template<class Problem>
    static Params to_underlying_arguments(const Problem &problem, const Arguments &args, void *workspace) {
        return {StandardMainloop::to_underlying_arguments(problem, args, workspace), args.maps, args.stripesPerExpert};
    }
    // CUTLASS uses a mock L coordinate on the TMA producer. The tensor-map
    // update supplies the actual expert selected by the persistent scheduler.
    int currentGroup = 0;
    template<class Inputs, class Problem>
    CUTLASS_DEVICE auto tensors_perform_update(const Inputs &inputs, const Params &params, const Problem &problem, int group) {
        currentGroup = group;
        return StandardMainloop::tensors_perform_update(inputs, params, problem, group);
    }
    template<class... Args> CUTLASS_DEVICE void load(Args&&... args) {
        StandardMainloop::template load<PackedWeights>(static_cast<Args&&>(args)..., currentGroup);
    }
    template<class... Args> CUTLASS_DEVICE void mma(Args&&... args) {
        StandardMainloop::template mma<4, PackedWeights>(static_cast<Args&&>(args)...);
    }
};
using Kernel = kernel::GemmUniversal<ShapeMNK, Mainloop, Epilogue>;
using Gemm = device::GemmUniversalAdapter<Kernel>;
using Metadata = grouped_sm120::Metadata<Gemm>;

__global__ void countGroupedRoutes(const int *ids, int *counts, int *rank, int routes) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < routes) rank[r] = atomicAdd(counts + ids[r], 1);
}
__global__ void packGroupedRows(const Block16 *src, const int *ids, const int *offset, const int *rank,
                                GroupedBuffers buffers, int k, int topK) {
    cudaTriggerProgrammaticLaunchCompletion();
    const int r = blockIdx.x, expert = ids[r], groups = k / 16;
    GroupedInputRow dst(buffers, offset[expert], expert, rank[r], groups);
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const auto &x = src[size_t(r / topK) * groups + g];
        dst.codes[g] = make_uint2(x.code[0], x.code[1]);
        dst.scale(g) = x.scale;
    }
}

__global__ void sumGroupedProjections(const float *sorted, float *output, const int *ids,
                                      const int *offsets, const int *ranks, const float *weights,
                                      const float *rowScales, int width, int topK) {
    const int token = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= width) return;
    float sum = 0;
    // Sum in original top-k order, irrespective of the atomic rank within an expert.
    for (int j = 0; j < topK; ++j) {
        const int route = token * topK + j, expert = ids[route];
        const int row = offsets[expert] + ranks[route];
        sum += scaledGroupedValue(sorted[size_t(row) * width + col], weights[expert], rowScales[route]);
    }
    output[size_t(token) * width + col] = sum;
}

} // namespace

// One map spans all expert stripes. Keep K as an independent coordinate so
// TMA can zero-fill a partial final tile without crossing into the next stripe.
bool makeGroupedWeightMaps(GroupedWeightMaps &maps, const Weights &weights, int experts, int n, int k) {
    if (experts <= 0 || n <= 0 || k <= 0 || n % 128 || k % 64) return false;
    const cuuint64_t bands = k / 64, stripes = cuuint64_t(n / 16) * experts;
    if (stripes > std::numeric_limits<int>::max()) return false;
    auto encode = [&](CUtensorMap &map, const void *data, unsigned words) {
        const cuuint64_t dims[] = {words, bands, stripes};
        const cuuint64_t strides[] = {words * sizeof(uint32_t), words * sizeof(uint32_t) * bands};
        const cuuint32_t box[] = {words, PackedWeights::tileBands, PackedWeights::stripes}, steps[] = {1, 1, 1};
        return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, const_cast<void *>(data), dims,
                                      strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    };
    return encode(maps.codes, weights.codes, 128) && encode(maps.scales, weights.scales, 16);
}

grouped_sm120::Scratch groupedScratch(int groups, int smCount) {
    return Metadata::scratchSize(groups, smCount);
}

cudaError_t planGroupedRows(const int *ids, int *offsets, int *ranks, int experts, int routes, cudaStream_t stream) {
    auto status = cudaMemsetAsync(offsets, 0, sizeof(int) * experts, stream);
    if (status != cudaSuccess) return status;
    countGroupedRoutes<<<(routes + 255) / 256, 256, 0, stream>>>(ids, offsets, ranks, routes);
    return grouped_sm120::prefixRouteCounts(offsets, experts, stream);
}

cudaError_t packGroupedInput(const Block16 *input, const int *ids, const int *offsets, const int *ranks,
                             const GroupedBuffers &buffers, int routes, int k, int topK, cudaStream_t stream) {
    packGroupedRows<<<routes, 128, 0, stream>>>(input, ids, offsets, ranks, buffers, k, topK);
    return cudaGetLastError();
}

cudaError_t projectGroupedRows(const Weights &weights, const GroupedWeightMaps &maps, int *offsets,
                               const GroupedBuffers &buffers, int n, int k, int groups, int smCount, cudaStream_t stream) {
    Metadata meta(buffers.metadata, groups);
    auto prepare = flashinfer::group_gemm::compute_sm120_cutlass_nvfp4_group_gemm_args<
        true, 16, Mainloop::Sm1xxBlkScaledConfig, Fp4, Fp4, Scale, Scale, float,
        Metadata::Shape, Metadata::StrideA, Metadata::StrideB, Metadata::StrideD, Metadata::LayoutA, Metadata::LayoutB>;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((groups + 127) / 128);
    config.blockDim = dim3(128);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    auto status = cudaLaunchKernelEx(&config, prepare,
        reinterpret_cast<Fp4 *>(const_cast<uint4 *>(weights.codes)), reinterpret_cast<Fp4 *>(buffers.codes),
        reinterpret_cast<Scale *>(const_cast<uint32_t *>(weights.scales)), reinterpret_cast<Scale *>(buffers.scales),
        buffers.accumulators, offsets, n, k, groups, meta.shape, meta.a, meta.b, meta.sa, meta.sb, meta.d,
        meta.da, meta.db, meta.dd, meta.la, meta.lb);
    if (status != cudaSuccess) return status;
    cutlass::KernelHardwareInfo hardware{};
    hardware.sm_count = smCount;
    typename Gemm::Arguments args{GemmUniversalMode::kGrouped, {groups, meta.shape, nullptr},
        {{meta.a, meta.da, meta.b, meta.db, meta.sa, meta.la, meta.sb, meta.lb}, maps, n / 16}, {{}, nullptr, nullptr, meta.d, meta.dd}, hardware};
    args.epilogue.thread.alpha = 1;
    Gemm gemm;
    CUTLASS_CHECK(gemm.initialize(args, buffers.gemm, stream));
    // The grouped scheduler reads the freshly written shapes and descriptor
    // arrays without a grid-dependency wait. Keep this edge fully ordered.
    CUTLASS_CHECK(gemm.run(stream));
    return cudaGetLastError();
}

cudaError_t sumGroupedRows(const float *sorted, float *output, const int *ids, const int *offsets, const int *ranks,
                           const float *weightScales, const float *rowScales, int rows, int n, int topK, cudaStream_t stream) {
    sumGroupedProjections<<<dim3(rows, (n + 255) / 256), 256, 0, stream>>>(
        sorted, output, ids, offsets, ranks, weightScales, rowScales, n, topK);
    return cudaGetLastError();
}
} // namespace fastllm::cuda::routed_fp4
