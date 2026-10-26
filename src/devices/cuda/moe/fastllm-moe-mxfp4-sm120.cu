// V4.1 host-expert prefill: block-32 FP8 activations x MXFP4 weights.
// Keep the compact host/cache records intact; scratch belongs to the existing
// per-device MoE worker, so packing and GEMM need no new host synchronization.
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>
// Define the tensor transform overload before SM120's FP4 shift traits.
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

namespace fastllm_v41_mxfp4 {
using namespace cute;
using Shape = cute::Shape<_128, _128, _128>;
using Cluster = cute::Shape<_1, _1, _1>;
using Epilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Shape, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, cutlass::layout::RowMajor, 8, cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::layout::RowMajor, 16,
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, cutlass::layout::ColumnMajor, 32,
    float, Shape, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
    Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

__device__ size_t ScaleOffset(int row, int block, int kTiles) {
    return (size_t(row / 128) * kTiles + block / 4) * 512 +
        (row % 32) * 16 + (row % 128) / 32 * 4 + block % 4;
}

// Four lanes per block. Input has already crossed V4.1's BF16/FP8 rounding
// boundary; this encodes the same values in native E4M3 + UE8M0 form.
__global__ void PackActivation(const __nv_bfloat16 *input, uint8_t *quant,
                               uint8_t *scales, int rows, int cols) {
    const int lane = threadIdx.x & 3, task = blockIdx.x * 64 + threadIdx.x / 4;
    const int blocks = cols / 32;
    if (task >= rows * blocks) return;
    const int row = task / blocks, block = task % blocks;
    const size_t offset = size_t(row) * cols + block * 32 + lane * 8;
    const uint4 packed = *reinterpret_cast<const uint4 *>(input + offset);
    const auto *bf = reinterpret_cast<const __nv_bfloat16 *>(&packed);
    float v[8], maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = __bfloat162float(bf[i]); maximum = fmaxf(maximum, fabsf(v[i])); }
    const unsigned active = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 2, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 1, 4));
    const int exponent = max(0, min(254, int(ceilf(log2f(fmaxf(maximum, 1e-4f) / 448.f))) + 127));
    const float scale = ldexpf(1.f, exponent - 127);
    __nv_fp8x4_e4m3 lo(make_float4(v[0]/scale,v[1]/scale,v[2]/scale,v[3]/scale));
    __nv_fp8x4_e4m3 hi(make_float4(v[4]/scale,v[5]/scale,v[6]/scale,v[7]/scale));
    *reinterpret_cast<uint2 *>(quant + offset) = make_uint2(lo.__x, hi.__x);
    if (!lane) scales[ScaleOffset(row, block, cols / 128)] = exponent;
}

// Each NUMA record is 16 packed E2M1 bytes followed by one UE8M0 byte.
// Split these into CUTLASS operands without materializing BF16 weights.
__global__ void PackWeight(const uint8_t *input, uint8_t *quant, uint8_t *scales,
                           int rows, int cols) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks = cols / 32;
    if (task >= rows * blocks) return;
    const uint8_t *source = input + size_t(task) * 17;
    uint4 value;
    auto *bytes = reinterpret_cast<uint8_t *>(&value);
#pragma unroll
    for (int i = 0; i < 16; ++i) bytes[i] = source[i];
    reinterpret_cast<uint4 *>(quant)[task] = value;
    scales[ScaleOffset(task / blocks, task % blocks, cols / 128)] = source[16];
}

Gemm::Arguments Arguments(void *out, void *a, void *b, void *sa, void *sb, int m, int n, int k) {
    Gemm::Arguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<Gemm::ElementA *>(a);
    args.mainloop.ptr_B = static_cast<Gemm::ElementB *>(b);
    args.mainloop.ptr_SFA = static_cast<cutlass::float_ue8m0_t *>(sa);
    args.mainloop.ptr_SFB = static_cast<cutlass::float_ue8m0_t *>(sb);
    args.mainloop.dA = cutlass::make_cute_packed_stride(Kernel::StrideA{}, {m, k, 1});
    args.mainloop.dB = cutlass::make_cute_packed_stride(Kernel::StrideB{}, {n, k, 1});
    args.mainloop.layout_SFA = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
    args.epilogue.ptr_D = static_cast<cutlass::bfloat16_t *>(out);
    args.epilogue.dD = cutlass::make_cute_packed_stride(Kernel::StrideD{}, {m, n, 1});
    args.epilogue.dC = args.epilogue.dD;
    args.epilogue.thread.alpha = 1.f;
    args.epilogue.thread.beta = 0.f;
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    return args;
}
}

bool FastllmCudaV41Mxfp4Linear(const fastllm::Data &input, const fastllm::Data &weight,
                             fastllm::Data &output, fastllm::Data &scratch) {
    using namespace fastllm;
    using namespace fastllm_v41_mxfp4;
    const char *enabled = std::getenv("FASTLLM_CUTLASS_MXFP4");
    if ((enabled && enabled[0] == '0') || FastllmCudaGraphIsCapturingFast() ||
        FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() > 121 ||
        input.dataType != BFLOAT16 || input.dims.size() != 2 || !input.cudaData ||
        weight.dataType != NVFP4_BLOCK_32_E8M0 || weight.IsRepacked || !weight.cudaData ||
        weight.dims.size() != 2 || input.dims[1] != weight.dims[1] ||
        input.multiDeviceData || weight.multiDeviceData || output.multiDeviceData ||
        &input == &output || &weight == &output) return false;
    const int m = input.dims[0], n = weight.dims[0], k = weight.dims[1];
    if (m < 128 || n <= 0 || k <= 0 || n % 128 || k % 128) return false;
    const auto align = [](size_t bytes) { return (bytes + 255) & ~size_t(255); };
    const size_t ab = align(size_t(m) * k), bb = align(size_t(n) * k / 2);
    const size_t as = align(size_t((m + 127) / 128) * 128 * (k / 32));
    const size_t bs = align(size_t(n) * (k / 32));
    const size_t workspaceBytes = Gemm::get_workspace_size(Arguments(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
    const size_t total = ab + bb + as + bs + workspaceBytes;
    if (total > INT_MAX) return false;
    scratch.dataType = INT8;
    scratch.UpdateUnitSize();
    scratch.Resize({int(total)});
    scratch.ToDevice(CUDA, {FastllmCudaGetDevice()}, false);
    scratch.Allocate(false);
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize({m, n});
    output.ToDevice(CUDA, {FastllmCudaGetDevice()}, false);
    output.Allocate(false);
    auto *a = static_cast<uint8_t *>(scratch.cudaData), *b = a + ab, *sa = b + bb, *sb = sa + as;
    auto args = Arguments(output.cudaData, a, b, sa, sb, m, n, k);
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess) return false;
    const auto stream = cudaStreamPerThread;
    if (m % 128) cudaMemsetAsync(sa, 0, as, stream);
    PackActivation<<<(m * (k/32) + 63)/64,256,0,stream>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), a, sa, m, k);
    PackWeight<<<(n * (k/32) + 255)/256,256,0,stream>>>(
        static_cast<const uint8_t *>(weight.cudaData), b, sb, n, k);
    Gemm gemm;
    AssertInFastLLM(gemm.initialize(args, sb + bs, stream) == cutlass::Status::kSuccess &&
                   gemm.run(stream) == cutlass::Status::kSuccess,
                   "V4.1 SM120 W4A8 GEMM failed.");
    return true;
}
