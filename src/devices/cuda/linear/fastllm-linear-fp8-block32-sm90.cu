// Native Hopper W8A8 for row-wise FP8 weights with one FP32 scale per 32 values.
// Weight bytes/scales are never expanded, requantized or copied into a cache.
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "devices/cuda/fastllm-mxfp8-projection.h"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "fastllm-bf16-lt.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <deep_gemm/impls/sm90_fp8_gemm_block32.cuh>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace {
constexpr int kWorkers = 78;
constexpr int kMaxRows = 4096;

// Four lanes quantize one K32 group; padded rows are zeroed for swapped GEMM.
__global__ void QuantizeBlock32(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output,
                              float *scales, int rows, int inner, int paddedRows) {
    const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    const int lane = threadIdx.x % 4;
    const int row = group / (inner / 32), col = group % (inner / 32) * 32 + lane * 8;
    if (row >= paddedRows) return;
    alignas(16) __nv_bfloat16 values[8]{};
    if (row < rows)
        *reinterpret_cast<uint4 *>(values) = *reinterpret_cast<const uint4 *>(input + size_t(row) * inner + col);
    float amax = 1.0e-4f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(values[i])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
    if (lane == 0) scales[size_t(col / 32) * paddedRows + row] = scale;
    alignas(8) __nv_fp8_e4m3 packed[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) packed[i] = __nv_fp8_e4m3(__bfloat162float(values[i]) / scale);
    *reinterpret_cast<uint2 *>(output + size_t(row) * inner + col) = *reinterpret_cast<uint2 *>(packed);
}

__global__ void ReduceSplit(const float *partial, __nv_bfloat16 *output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(partial[i] + partial[count + i]);
}

bool MakeTma(cute::TmaDescriptor &desc, CUtensorMapDataType type, void *ptr,
             int inner, int outer, int stride, int tileInner, int tileOuter,
             CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    const cuuint64_t dims[] = {uint64_t(inner), uint64_t(outer)}, strides[] = {uint64_t(stride)};
    const cuuint32_t box[] = {uint32_t(tileInner), uint32_t(tileOuter)}, element[] = {1, 1};
    return cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&desc), type, 2, ptr,
        dims, strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template<int BM, int BN, int Stages, bool Swapped, bool Split, bool Packed = false, bool Pdl = false,
         bool BlockSplit = false>
struct Projection {
    static constexpr int TeamBytes = (BM * BN * 2 + 1023) / 1024 * 1024 +
        Stages * (BM * 128 + BN * 128 + BM * 16 + BN * 16) + Stages * 16;
    static constexpr int Shared = BlockSplit ? 2 * ((TeamBytes + 1023) / 1024 * 1024) + BM * BN * sizeof(float) : TeamBytes;
    static auto Kernel() {
        return deep_gemm::sm90_fp8_gemm_block32_impl<BM, BN, Stages, Swapped, Split, kWorkers, Packed, BlockSplit>;
    }
    static bool Configure() {
        return cudaFuncSetAttribute(Kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
    }
    static bool Run(void *a, void *b, float *sa, float *sb, __nv_bfloat16 *out, float *partial,
                    int rows, int inner, int channels, int paddedRows) {
        cute::TmaDescriptor ta, tb, td, tsa, tsb;
        int m = rows, n = channels;
        if constexpr (Swapped) { std::swap(a, b); std::swap(sa, sb); m = channels; n = paddedRows; }
        if (!MakeTma(ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, a, inner, m, inner, 128, BM, CU_TENSOR_MAP_SWIZZLE_128B) ||
            !MakeTma(tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, b, inner, n, inner, 128, BN, CU_TENSOR_MAP_SWIZZLE_128B))
            return false;
        if constexpr (Swapped) {
            if (!MakeTma(tsa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sa, inner / 32, channels, inner / 8, 4, BM) ||
                !MakeTma(tsb, Packed ? CU_TENSOR_MAP_DATA_TYPE_UINT32 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sb, paddedRows,
                         inner / (Packed ? 128 : 32), paddedRows * 4, BN, Packed ? 1 : 4)) return false;
            // The small-M epilogue stores directly; use a valid descriptor for its prefetch.
            td = ta;
        } else {
            if (!MakeTma(tsa, Packed ? CU_TENSOR_MAP_DATA_TYPE_UINT32 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                         sa, paddedRows, inner / (Packed ? 128 : 32), paddedRows * 4, BM, Packed ? 1 : 4) ||
                !MakeTma(tsb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sb, inner / 32, channels, inner / 8, 4, BN) ||
                !MakeTma(td, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, out, channels, rows, channels * 2, 64, BM, CU_TENSOR_MAP_SWIZZLE_128B)) return false;
        }
        constexpr bool separateSplit = Split && !Packed;
        void *result = separateSplit ? static_cast<void *>(partial) : static_cast<void *>(out);
        auto kernel = Kernel();
        constexpr int threads = (128 + (BM <= 64 ? 128 : BM == 192 ? 384 : 256)) * (BlockSplit ? 2 : 1);
        const dim3 grid(kWorkers, separateSplit ? 2 : 1);
        cudaError_t error;
        if constexpr (Pdl) {
            cudaLaunchAttribute attribute{};
            attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config{grid, dim3(threads), Shared, cudaStreamPerThread, &attribute, 1};
            error = cudaLaunchKernelEx(&config, kernel, result, rows, m, n, inner, ta, tb, td, tsa, tsb);
        } else {
            kernel<<<grid, threads, Shared, cudaStreamPerThread>>>(result, rows, m, n, inner, ta, tb, td, tsa, tsb);
            error = cudaGetLastError();
        }
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        if constexpr (separateSplit) {
            const size_t length = size_t(rows) * channels;
            ReduceSplit<<<(length + 255) / 256, 256, 0, cudaStreamPerThread>>>(partial, out, length);
        }
        return true;
    }
};

// Fixed persistent grid size is part of the scheduler, independent of physical
// SM count. These kernels are valid on all SM90 devices, including H100/H800.
using Small8 = Projection<256, 8, 6, true, false>;
using Small16 = Projection<256, 16, 5, true, false>;
using Split8 = Projection<128, 8, 6, true, true>;
using Split16 = Projection<128, 16, 6, true, true>;
using Swapped32 = Projection<128, 32, 6, true, false>;
using Swapped64 = Projection<128, 64, 4, true, false>;
using Split32 = Projection<128, 32, 4, true, true>;
using Split64 = Projection<128, 64, 4, true, true>;
using Medium32 = Projection<32, 64, 4, false, false>;
using Medium64 = Projection<64, 64, 4, false, false>;
using Large = Projection<128, 64, 6, false, false>;

auto SelectProjection(int rows, int inner, int channels, bool splitFits) {
    const bool canSplit = splitFits && inner >= 2048 && inner % 256 == 0;
    if (rows <= 16) {
        if (canSplit && channels <= 8192)
            return rows <= 8 ? Split8::Run : Split16::Run;
        return rows <= 8 ? Small8::Run : Small16::Run;
    }
    if (inner >= 2048 && channels >= 4096) {
        // Put the short token dimension on WGMMA N. Narrow, long-K
        // outputs also benefit from two CTAs working on the same tile.
        if (rows <= 64) {
            if (canSplit && channels == 4096 && inner >= (rows <= 32 ? 4096 : 8192))
                return rows <= 32 ? Split32::Run : Split64::Run;
            return Swapped32::Run;
        }
        // A second 128-row M tile would be at most half full.
        if (rows > 128 && rows <= 192 && channels >= 8192)
            return Swapped64::Run;
    }
    return rows <= 32 ? Medium32::Run : rows <= 64 ? Medium64::Run : Large::Run;
}

using Packed8 = Projection<256, 8, 6, true, false, true>;
using PackedSplit8 = Projection<64, 8, 6, true, true, true>;
using Packed16 = Projection<256, 16, 5, true, false, true>;
using PackedSplit16 = Projection<64, 16, 6, true, true, true>;

bool PrepareKernels() {
    static thread_local std::map<int, bool> ready;
    const int device = FastllmCudaGetDevice();
    auto found = ready.find(device);
    if (found != ready.end()) return found->second;
    if (FastllmCudaGraphIsCapturingFast()) return false;
    const bool ok = Small8::Configure() && Small16::Configure() && Split8::Configure() &&
        Split16::Configure() && Swapped32::Configure() && Swapped64::Configure() &&
        Split32::Configure() && Split64::Configure() && Medium32::Configure() &&
        Medium64::Configure() && Large::Configure();
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}

int PackedVerifyRows(const fastllm::Data &w, bool verify) {
    if (!verify || w.dataType != fastllm::FP8_E4M3 || w.dims.size() != 2 || w.blockM != 32 ||
        w.blockK != 1 || w.IsRepacked || !w.cudaData || w.dataDevice != fastllm::CUDA ||
        w.strides.size() != 2 || w.strides[0] != w.dims[1] || w.strides[1] != 1) return 0;
    const int out = w.dims[0], in = w.dims[1];
    if (in == 4096 && (out == 13568 || out == 14848)) return 128;
    return in == 8192 && out == 4096 ? 16 : 0;
}
void PreparePackedVerify(fastllm::Data &w) {
    if (!PackedVerifyRows(w, true)) return;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(w, *fastllm::GetEmptyData(), w.dims[0]);
}
bool PackedVerifyKernelsReady() {
    static thread_local std::map<int, bool> configured;
    const int device = FastllmCudaGetDevice();
    auto it = configured.find(device);
    if (it == configured.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        const bool ok = Packed8::Configure() && PackedSplit8::Configure() &&
                        Packed16::Configure() && PackedSplit16::Configure();
        if (!ok) cudaGetLastError();
        it = configured.emplace(device, ok).first;
    }
    return it->second;
}

template<class Kernel> bool SwaKernelReady() {
    static thread_local std::map<int, bool> ready;
    const int device = FastllmCudaGetDevice();
    auto it = ready.find(device);
    if (it == ready.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        it = ready.emplace(device, Kernel::Configure()).first;
        if (!it->second) cudaGetLastError();
    }
    return it->second;
}

// Three 64-channel warp groups cover one 192-channel SWA QKV tile.
using SwaQuery32 = Projection<192, 32, 5, true, false, true>;

// Wide tiles keep activations on the M axis and write BF16 with TMA.
using VerifyProjection128 = Projection<128, 64, 4, false, false, true>;
template<class Kernel>
bool RunVerifyProjection(FastllmMxfp8Activation x, const fastllm::Data &w, void *out) {
    if (!SwaKernelReady<Kernel>()) return false;
    return Kernel::Run(x.values, w.cudaData, reinterpret_cast<float *>(x.scales),
        static_cast<float *>(w.extraCudaData[0]), static_cast<__nv_bfloat16 *>(out), nullptr,
        x.rows, x.width, w.dims[0], x.pitch);
}

template<bool Pdl> bool ProjectPackedVerify(FastllmMxfp8Activation x, const fastllm::Data &w, void *out) {
    if (x.rows > PackedVerifyRows(w, true) || x.swizzledRowOffset >= 0 || x.rows < 2 ||
        !x.Matches(x.rows, w.dims[1]) || w.extraCudaData.empty() || !w.extraCudaData[0]) return false;
    if (x.rows > 64)
        return RunVerifyProjection<VerifyProjection128>(x, w, out);
    if (x.rows > 32) {
        // Three warp groups cover 192 output channels per tile with up to 64 verification rows.
        using VerifyQuery64 = Projection<192, 64, 4, true, false, true>;
        return RunVerifyProjection<VerifyQuery64>(x, w, out);
    }
    if (x.rows > 16) {
        return RunVerifyProjection<SwaQuery32>(x, w, out);
    }
    if (!PackedVerifyKernelsReady()) return false;
    const bool split = w.dims[0] == 4096;
    const auto run = x.rows > 8
        ? (split ? Projection<64, 16, 6, true, true, true, Pdl>::Run
                 : Projection<256, 16, 5, true, false, true, Pdl>::Run)
        : split ? Projection<64, 8, 6, true, true, true, Pdl>::Run
                           : Projection<256, 8, 6, true, false, true, Pdl>::Run;
    return run(x.values, w.cudaData, reinterpret_cast<float *>(x.scales), static_cast<float *>(w.extraCudaData[0]),
               static_cast<__nv_bfloat16 *>(out), nullptr, x.rows, x.width, w.dims[0], x.pitch);
}

// Two warp groups compute opposite K halves and reduce their FP32 fragments in shared memory.
using SwaOutput32 = Projection<64, 32, 4, true, true, true, false, true>;
int SwaOutputRows(const fastllm::Data &w, bool verify) {
    return PackedVerifyRows(w, verify) && w.dims[1] == 8192 ? 128 : 0;
}
bool ProjectVerifyOutput(FastllmMxfp8Activation x, const fastllm::Data &w, void *out) {
    if (!SwaOutputRows(w, true) || (x.rows <= 16 || x.rows > 128) || !x.Matches(x.rows, 8192) || x.swizzledRowOffset >= 0 ||
        w.extraCudaData.empty() || !w.extraCudaData[0]) return false;
    if (x.rows > 64)
        return RunVerifyProjection<VerifyProjection128>(x, w, out);
    using VerifyOutput64 = Projection<64, 64, 4, true, true, true, false, true>;
    return x.rows > 32 ? RunVerifyProjection<VerifyOutput64>(x, w, out) : RunVerifyProjection<SwaOutput32>(x, w, out);
}
} // namespace

const FastllmMxfp8ProjectionBackend *FastllmCudaSm90SwaOutputProjection() {
    static const FastllmMxfp8ProjectionBackend backend{PreparePackedVerify, SwaOutputRows, ProjectVerifyOutput};
    return &backend;
}

const FastllmMxfp8ProjectionBackend *FastllmCudaSm90PackedVerifyProjection() {
    static const auto ops = [] {
        FastllmMxfp8ProjectionBackend backend{PreparePackedVerify, PackedVerifyRows, ProjectPackedVerify<false>};
        backend.projectPdl = ProjectPackedVerify<true>;
        return backend;
    }();
    return &ops;
}

// Raw entry shared by the Data dispatcher and numerical/Graph tests. Scales
// use the same original row-major FP32 layout as the generic fallback.
bool FastllmCudaBlock32W8A8SM90(const void *input, const void *weight, const float *scales,
                             void *output, int rows, int inner, int channels) {
    if (FastllmCudaRuntimeArch() != 90 || rows <= 0 || inner < 128 || inner % 128 ||
        channels <= 0 || channels % 64 || !input || !weight || !scales || !output ||
        (reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weight) |
         reinterpret_cast<uintptr_t>(scales) | reinterpret_cast<uintptr_t>(output)) % 16 || !PrepareKernels()) return false;
    // Share the existing 8 MiB Linear lease. This also works inside model KV
    // arena scopes, where ordinary scratch allocation is forbidden in capture.
    // Large M is tiled to fit; no new cache or serving-time allocation is needed.
    const size_t capacity = fastllm_bf16_lt::workspaceBytes;
    int rowLimit = std::min(size_t(kMaxRows), capacity / (size_t(inner) + size_t(inner / 32) * sizeof(float)));
    const int rowAlignment = rowLimit >= 128 ? 128 : 8;
    rowLimit = rowLimit / rowAlignment * rowAlignment;
    if (!rowLimit) return false;
    const int chunk = std::min(rows, rowLimit), padded = (chunk + 7) / 8 * 8;
    const size_t inputBytes = (size_t(padded) * inner + 255) / 256 * 256;
    const size_t scaleBytes = (size_t(padded) * (inner / 32) * sizeof(float) + 255) / 256 * 256;
    if (inputBytes + scaleBytes > capacity) return false;
    const size_t partialCapacity = capacity - inputBytes - scaleBytes;
    const bool capturing = FastllmCudaGraphIsCapturingFast();
    auto state = fastllm_bf16_lt::GetState(FastllmCudaGetDevice(), capturing);
    if (!state) return false;
    void *scratch = capturing ? fastllm_bf16_lt::CaptureWorkspace(*state) : state->eager->data;
    if (!scratch) return false;
    auto quantized = static_cast<__nv_fp8_e4m3 *>(scratch);
    auto activationScales = reinterpret_cast<float *>(static_cast<char *>(scratch) + inputBytes);
    auto partial = reinterpret_cast<float *>(static_cast<char *>(scratch) + inputBytes + scaleBytes);
    for (int offset = 0; offset < rows; offset += chunk) {
        const int count = std::min(rows - offset, chunk), scaleRows = (count + 7) / 8 * 8;
        const auto src = static_cast<const __nv_bfloat16 *>(input) + size_t(offset) * inner;
        auto dst = static_cast<__nv_bfloat16 *>(output) + size_t(offset) * channels;
        QuantizeBlock32<<<(size_t(scaleRows) * inner / 8 + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            src, quantized, activationScales, count, inner, scaleRows);
        const bool splitFits = size_t(count) * channels * 2 * sizeof(float) <= partialCapacity;
        const auto run = SelectProjection(count, inner, channels, splitFits);
        const bool ok = run(quantized, const_cast<void *>(weight), activationScales,
            const_cast<float *>(scales), dst, partial, count, inner, channels, scaleRows);
        if (!ok) throw std::runtime_error("SM90 block32 FP8 TMA descriptor rejected");
    }
    return true;
}
