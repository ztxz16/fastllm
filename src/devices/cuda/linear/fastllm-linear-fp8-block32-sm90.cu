// Native Hopper W8A8 for row-wise FP8 weights with one FP32 scale per 32 values.
// Weight bytes/scales are never expanded, requantized or copied into a cache.
#include "fastllm-cuda.cuh"
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

template<int BM, int BN, int Stages, bool Swapped, bool Split>
struct Projection {
    static constexpr int Shared = (BM * BN * 2 + 1023) / 1024 * 1024 +
        Stages * (BM * 128 + BN * 128 + BM * 16 + BN * 16) + Stages * 16;
    static auto Kernel() {
        return deep_gemm::sm90_fp8_gemm_block32_impl<BM, BN, Stages, Swapped, Split, kWorkers>;
    }
    static bool Configure() {
        return cudaFuncSetAttribute(Kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
    }
    static bool Run(void *a, void *b, float *sa, float *sb, void *out,
                    int rows, int inner, int channels, int paddedRows) {
        cute::TmaDescriptor ta, tb, td, tsa, tsb;
        int m = rows, n = channels;
        if constexpr (Swapped) { std::swap(a, b); std::swap(sa, sb); m = channels; n = paddedRows; }
        if (!MakeTma(ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, a, inner, m, inner, 128, BM, CU_TENSOR_MAP_SWIZZLE_128B) ||
            !MakeTma(tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, b, inner, n, inner, 128, BN, CU_TENSOR_MAP_SWIZZLE_128B))
            return false;
        if constexpr (Swapped) {
            if (!MakeTma(tsa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sa, inner / 32, channels, inner / 8, 4, BM) ||
                !MakeTma(tsb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sb, paddedRows, inner / 32, paddedRows * 4, BN, 4)) return false;
            // The small-M epilogue stores directly; use a valid descriptor for its prefetch.
            td = ta;
        } else {
            if (!MakeTma(tsa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sa, paddedRows, inner / 32, paddedRows * 4, BM, 4) ||
                !MakeTma(tsb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sb, inner / 32, channels, inner / 8, 4, BN) ||
                !MakeTma(td, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, out, channels, rows, channels * 2, 64, BM, CU_TENSOR_MAP_SWIZZLE_128B)) return false;
        }
        auto kernel = Kernel();
        kernel<<<dim3(kWorkers, Split ? 2 : 1), 128 + (BM <= 64 ? 128 : 256), Shared, cudaStreamPerThread>>>(
            out, rows, m, n, inner, ta, tb, td, tsa, tsb);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return true;
    }
};

// Fixed persistent grid size is part of the scheduler, independent of physical
// SM count. These kernels are valid on all SM90 devices, including H100/H800.
using Small8 = Projection<256, 8, 6, true, false>;
using Small16 = Projection<256, 16, 5, true, false>;
using Split8 = Projection<128, 8, 6, true, true>;
using Split16 = Projection<128, 16, 6, true, true>;
using Medium32 = Projection<32, 64, 4, false, false>;
using Medium64 = Projection<64, 64, 4, false, false>;
using Large = Projection<128, 64, 6, false, false>;

bool PrepareKernels() {
    static thread_local std::map<int, bool> ready;
    const int device = FastllmCudaGetDevice();
    auto found = ready.find(device);
    if (found != ready.end()) return found->second;
    if (FastllmCudaGraphIsCapturingFast()) return false;
    const bool ok = Small8::Configure() && Small16::Configure() && Split8::Configure() &&
        Split16::Configure() && Medium32::Configure() && Medium64::Configure() && Large::Configure();
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}

} // namespace

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
    const bool split = chunk <= 16 && channels <= 8192 && inner >= 2048 && inner % 256 == 0;
    const size_t inputBytes = (size_t(padded) * inner + 255) / 256 * 256;
    const size_t scaleBytes = (size_t(padded) * (inner / 32) * sizeof(float) + 255) / 256 * 256;
    const size_t partialBytes = split ? size_t(chunk) * channels * 2 * sizeof(float) : 0;
    if (inputBytes + scaleBytes + partialBytes > capacity) return false;
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
        // A final tail of a large call uses the ordinary orientation; workspace
        // sizing never depends on an unallocated split-K tail buffer.
        void *result = split ? static_cast<void *>(partial) : dst;
        const auto run = split ? (count <= 8 ? Split8::Run : Split16::Run) :
            chunk <= 16 ? (count <= 8 ? Small8::Run : Small16::Run) :
            count <= 32 ? Medium32::Run : count <= 64 ? Medium64::Run : Large::Run;
        const bool ok = run(quantized, const_cast<void *>(weight), activationScales,
            const_cast<float *>(scales), result, count, inner, channels, scaleRows);
        if (!ok) throw std::runtime_error("SM90 block32 FP8 TMA descriptor rejected");
        if (split) {
            const size_t length = size_t(count) * channels;
            ReduceSplit<<<(length + 255) / 256, 256, 0, cudaStreamPerThread>>>(partial, dst, length);
        }
    }
    return true;
}
