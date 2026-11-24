// Large expert batches use native block-scaled E4M3 x E4M3 Tensor Cores.
// Decode and prefill share one persistent copy of the Marlin weights.
#include "fastllm-grouped-sm120.cuh"
#include "devices/cuda/fastllm-cuda-moe-activation.h"
#include "flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh"

namespace fastllm::cuda::routed_mxfp8 {
namespace {
using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;
using Problems = GroupProblemShape<Shape<int, int, int>>;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Fp8 = float_e4m3_t;
using Scale = float_ue8m0_t;
using Out = bfloat16_t;
using Epilogue = epilogue::collective::CollectiveBuilder<arch::Sm120, arch::OpClassBlockScaledTensorOp,
    Tile, Cluster, epilogue::collective::EpilogueTileAuto, float, float, void, layout::RowMajor *, 0,
    Out, layout::RowMajor *, 8, epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = gemm::collective::CollectiveBuilder<arch::Sm120, arch::OpClassBlockScaledTensorOp,
    mx_float8_t<Fp8>, layout::RowMajor *, 16, mx_float8_t<Fp8>, layout::ColumnMajor *, 16,
    float, Tile, Cluster, gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = kernel::GemmUniversal<Problems, Mainloop, Epilogue>;
using Gemm = device::GemmUniversalAdapter<Kernel>;
using Metadata = grouped_sm120::Metadata<Gemm>;

// Undo the 16x64 Marlin packing in shared memory. Both the tiled read
// and the row-major write transfer contiguous 128-byte warp segments.
__global__ void unpackMxfp8Weights(CudaMarlinMxfp8Weights src, uint32_t *dst, uint8_t *sf, int n, int k) {
    __shared__ uint32_t panels[8][256];
    int e = blockIdx.z, n0 = blockIdx.y * 64, k0 = blockIdx.x * 128;
    size_t expert = size_t(e) * n * k;
    int tid = threadIdx.x;
    #pragma unroll
    for (int panel = 0; panel < 8; ++panel)
        panels[panel][tid] = src.codes[expert / 4 + ((k0 / 16 + panel) * (n / 64) + blockIdx.y) * 256 + tid];
    __syncthreads();
    int word = tid % 32, part = word % 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row = tid / 32 + i * 8;
        int idx = (row % 8 * 4 + (part % 2) * 2) * 8 + row / 16 * 2 + row % 16 / 8;
        unsigned a = panels[word / 4][idx], b = panels[word / 4][idx + 8];
        dst[(expert + size_t(n0 + row) * k + k0) / 4 + word] = __byte_perm(a, b, part < 2 ? 0x6420 : 0x7531);
    }
    // Invert the Marlin scale transpose and byte permutation.
    int row = n0 + tid / 4, group = k0 / 32 + tid % 4;
    int flat = group * n + row, pos = flat % 64;
    int transposed = (pos % 8) * 8 + pos / 8;
    int perm = (transposed & ~3) + ((transposed & 1) << 1) + ((transposed & 2) >> 1);
    sf[expert / 32 + grouped_sm120::scaleOffset(row, group, k / 32)] =
        src.scales[expert / 32 + (flat & ~63) + perm];
}
} // namespace

grouped_sm120::Scratch scratchSize(int groups, int sms) {
    return Metadata::scratchSize(groups, sms);
}

cudaError_t project(const CudaMarlinMxfp8Weights &weights, const Buffers &buf, int *offsets, int n, int k,
                    int groups, int sms, cudaStream_t stream) {
    auto *sf = buf.weight + size_t(n) * k * groups;
    unpackMxfp8Weights<<<dim3(k / 128, n / 64, groups), 256, 0, stream>>>(
        weights, reinterpret_cast<uint32_t *>(buf.weight), sf, n, k);
    Metadata meta(buf.metadata, groups);
    flashinfer::group_gemm::compute_sm120_cutlass_nvfp4_group_gemm_args<
        false, 32, Mainloop::Sm1xxBlkScaledConfig>
        <<<(groups + 127) / 128, 128, 0, stream>>>(
            reinterpret_cast<Fp8 *>(buf.codes), reinterpret_cast<Fp8 *>(buf.weight),
            reinterpret_cast<Scale *>(buf.scales), reinterpret_cast<Scale *>(sf),
            reinterpret_cast<Out *>(buf.output), offsets, n, k, groups, meta.shape, meta.a, meta.b,
            meta.sa, meta.sb, meta.d, meta.da, meta.db, meta.dd, meta.la, meta.lb);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    KernelHardwareInfo hw{};
    hw.sm_count = sms;
    typename Gemm::Arguments args{GemmUniversalMode::kGrouped, {groups, meta.shape, nullptr},
        {meta.a, meta.da, meta.b, meta.db, meta.sa, meta.la, meta.sb, meta.lb},
        {{}, nullptr, nullptr, meta.d, meta.dd}, hw};
    args.epilogue.thread.alpha = 1;
    Gemm gemm;
    CUTLASS_CHECK(gemm.initialize(args, buf.workspace, stream));
    CUTLASS_CHECK(gemm.run(stream));
    return cudaGetLastError();
}
}
