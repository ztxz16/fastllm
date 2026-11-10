/*
 * SM120 MXFP8 W8A8 linear for DeepSeek V4.1.
 *
 * The V4.1 checkpoint stores E4M3 weights with one exact UE8M0 scale per
 * 32x32 block.  CUTLASS's SM120 MXFP8 kernels consume row-major FP8 A,
 * column-major FP8 B, and the FlashInfer 128x4 swizzled E8M0 layout.  This
 * adapter keeps the checkpoint bytes untouched, expands the compact weight
 * scales once, and quantizes each BF16 activation block online.
 */

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "devices/cuda/fastllm-draft-head.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120)

// FlashInfer uses this guard to expose the BF16 output specialization.
#ifndef ENABLE_BF16
#define ENABLE_BF16 1
#endif
#include "flashinfer/gemm/mxfp8_gemm_cutlass_template_sm120.h"

namespace flashinfer {
namespace gemm {

// The bundled CUTLASS requires an N tile of at least 128 for MXFP8 TMA
// scale loads. Instantiate the supported prefill tile.
INSTANTIATE_MXFP8_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 128, 128, 128, false)
INSTANTIATE_MXFP8_GEMM_KERNEL_LAUNCHER_SM120(float, 128, 128, 128, false)

}  // namespace gemm
}  // namespace flashinfer

namespace {

struct Mxfp8WeightCache {
    int device = -1;
    int rows = 0;
    int cols = 0;
    uint8_t *scales = nullptr;
    uint8_t *quant = nullptr;
    bool rejected = false;
};

struct Mxfp8WeightRegistry {
    std::mutex mutex;
    std::unordered_map<const fastllm::Data *, Mxfp8WeightCache> entries;
};

Mxfp8WeightRegistry &Mxfp8Registry() {
    static auto *registry = new Mxfp8WeightRegistry;
    return *registry;
}

// Lossy draft-head copies must never satisfy a lossless WoA cache lookup.
Mxfp8WeightRegistry &DraftHeadRegistry() {
    static auto *registry = new Mxfp8WeightRegistry;
    return *registry;
}

bool Mxfp8EnvEnabled() {
    const char *value = std::getenv("FASTLLM_CUTLASS_MXFP8");
    if (value == nullptr || value[0] == '\0') return true;
    return value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
           value[0] != 'f' && value[0] != 'F';
}

bool Mxfp8ScaleByte(float scale, uint8_t *encoded) {
    if (encoded == nullptr || !std::isfinite(scale) || scale <= 0.0f) return false;
    int exponent = 0;
    const float mantissa = std::frexp(scale, &exponent);
    // UE8M0 is an exact power of two.  Do not silently change a non-MXFP8
    // checkpoint into a different quantization scheme.
    if (mantissa != 0.5f || exponent < -126 || exponent > 128) return false;
    const int biased = exponent - 1 + 127;
    if (biased < 0 || biased > 254) return false;
    *encoded = static_cast<uint8_t>(biased);
    return true;
}

bool BuildMxfp8WeightScales(const fastllm::Data &weight, int rows, int cols,
                            std::vector<uint8_t> &swizzled) {
    if (rows <= 0 || cols <= 0 || rows % 32 != 0 || cols % 32 != 0 ||
        weight.blockK != 32 || weight.blockM != 32 ||
        weight.scales.size() != static_cast<size_t>(rows / 32) * (cols / 32)) {
        return false;
    }
    const int mTiles = (rows + 127) / 128;
    const int kTiles = (cols + 127) / 128;
    const size_t bytes = static_cast<size_t>(mTiles) * 128 * kTiles * 4;
    swizzled.assign(bytes, 0);
    std::vector<uint8_t> compact(weight.scales.size());
    for (size_t i = 0; i < compact.size(); ++i) {
        if (!Mxfp8ScaleByte(weight.scales[i], &compact[i])) return false;
    }

    const int scaleCols = cols / 32;
    for (int mt = 0; mt < mTiles; ++mt) {
        for (int kt = 0; kt < kTiles; ++kt) {
            for (int m4 = 0; m4 < 4; ++m4) {
                for (int m32 = 0; m32 < 32; ++m32) {
                    const int row = mt * 128 + m4 * 32 + m32;
                    for (int k4 = 0; k4 < 4; ++k4) {
                        const int scaleCol = kt * 4 + k4;
                        const size_t dst =
                            ((static_cast<size_t>(mt) * kTiles + kt) * 512 +
                             m32 * 16 + m4 * 4 + k4);
                        if (row < rows && scaleCol < scaleCols) {
                            swizzled[dst] = compact[(row / 32) * scaleCols + scaleCol];
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool EnsureMxfp8WeightScales(const fastllm::Data &weight, int rows, int cols,
                             const uint8_t *&deviceScales) {
    const int device = FastllmCudaGetDevice();
    auto &registry = Mxfp8Registry();
    // Serialize first-use publication so concurrent readers cannot replace and
    // free a scale buffer that another reader is about to submit to CUDA.
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto cached = registry.entries.find(&weight);
    if (cached != registry.entries.end() && cached->second.device == device &&
        cached->second.rows == rows && cached->second.cols == cols && cached->second.scales) {
        deviceScales = cached->second.scales;
        return true;
    }

    std::vector<uint8_t> hostScales;
    if (!BuildMxfp8WeightScales(weight, rows, cols, hostScales)) return false;
    void *storage = nullptr;
    if (FastllmCudaTryDirectMalloc(&storage, hostScales.size()) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
        return false;
    auto *allocated = static_cast<uint8_t *>(storage);
    FastllmCudaCopyFromHostToDevice(allocated, hostScales.data(), hostScales.size());
    if (FastllmCudaGetThreadError()) {
        FastllmCudaDirectFree(allocated);
        FastllmCudaClearThreadError();
        return false;
    }

    // A Data object is not moved while a linear call is in flight.  Replace a
    // stale entry only after the new copy has completed so a failed allocation
    // never destroys the old usable cache.
    Mxfp8WeightCache old;
    if (cached != registry.entries.end()) old = cached->second;
    registry.entries[&weight] = {device, rows, cols, allocated};
    deviceScales = allocated;

    if (old.scales != nullptr) {
        const int previous = FastllmCudaGetDevice();
        FastllmCudaSetDevice(old.device);
        FastllmCudaDirectFree(old.quant ? old.quant : old.scales);
        FastllmCudaSetDevice(previous);
    }
    return true;
}

// Four lanes cover a block-32 scale; each lane loads/stores eight values.
// This avoids millions of tiny CTAs for a 16K WoA input, and emits the
// CUTLASS scale swizzle directly. Interleaved groups become batched operands.
template <bool Interleaved = false>
__global__ void QuantizeToMxfp8Kernel(const __nv_bfloat16 *__restrict__ input,
        uint8_t *__restrict__ quant, uint8_t *__restrict__ scales,
        int rows, int cols, int groups, size_t scaleStride, int *inexact = nullptr) {
    const int lane = threadIdx.x & 3;
    const int task = blockIdx.x * 64 + threadIdx.x / 4;
    const int blocks = cols / 32, kTiles = (cols + 127) / 128;
    if (task >= rows * groups * blocks) return;
    const int block = task % blocks, row = (task / blocks) % rows;
    const int group = task / (blocks * rows);
    const size_t dst = ((size_t)group * rows + row) * cols + block * 32 + lane * 8;
    const size_t src = Interleaved ?
        ((size_t)row * groups + group) * cols + block * 32 + lane * 8 : dst;
    uint4 packed = *reinterpret_cast<const uint4 *>(input + src);
    const auto *values = reinterpret_cast<const __nv_bfloat16 *>(&packed);
    float value[8], maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value[i] = __bfloat162float(values[i]);
        maximum = fmaxf(maximum, fabsf(value[i]));
    }
    // Each four-lane subgroup is either entirely active or entirely out of range.
    const unsigned active = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 2, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, 1, 4));
    int biased = int(ceilf(log2f(fmaxf(maximum, 1e-4f) / 448.f))) + 127;
    biased = max(0, min(254, biased));
    const float scale = ldexpf(1.f, biased - 127);
    __nv_fp8x4_e4m3 lo(make_float4(value[0]/scale,value[1]/scale,value[2]/scale,value[3]/scale));
    __nv_fp8x4_e4m3 hi(make_float4(value[4]/scale,value[5]/scale,value[6]/scale,value[7]/scale));
    *reinterpret_cast<uint2 *>(quant + dst) = make_uint2(lo.__x,hi.__x);
    if (inexact) {
        uint2 encoded = make_uint2(lo.__x,hi.__x);
        const uint8_t *codes = reinterpret_cast<const uint8_t *>(&encoded);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            __nv_fp8_e4m3 q; q.__x = codes[i];
            if (float(q) * scale != value[i]) atomicExch(inexact, 1);
        }
    }
    if (lane == 0) {
        const size_t sf = group * scaleStride + ((size_t)(row / 128) * kTiles + block / 4) * 512 +
                          (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
        scales[sf] = uint8_t(biased);
    }
}

bool EnsureMxfp8Weight(const fastllm::Data &weight, int rows, int cols,
                       const uint8_t *&quant, const uint8_t *&scales, bool allowLossy = false) {
    const int device = FastllmCudaGetDevice();
    auto &registry = allowLossy ? DraftHeadRegistry() : Mxfp8Registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto cached = registry.entries.find(&weight);
    if (cached != registry.entries.end() && cached->second.device == device &&
        cached->second.rows == rows && cached->second.cols == cols) {
        if (cached->second.rejected) return false;
        if (cached->second.quant) {
            quant = cached->second.quant; scales = cached->second.scales;
            return true;
        }
    }
    // Graphs may reuse a fully initialized immutable weight, never allocate
    // or synchronize a new conversion during capture.
    if (FastllmCudaGraphIsCapturingFast()) return false;
    const size_t quantBytes = (size_t)rows * cols;
    const size_t scaleBytes = (size_t)((rows + 127) / 128) * ((cols + 127) / 128) * 512;
    const size_t total = quantBytes + scaleBytes + (allowLossy ? 0 : sizeof(int));
    size_t freeBytes = 0, deviceBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &deviceBytes) != cudaSuccess || freeBytes < total + (512u << 20))
        return false;
    void *storage = nullptr;
    if (FastllmCudaTryDirectMalloc(&storage, total) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    auto *allocation = static_cast<uint8_t *>(storage);
    auto *flag = allowLossy ? nullptr : reinterpret_cast<int *>(allocation + quantBytes + scaleBytes);
    cudaStream_t stream = cudaStreamPerThread;
    cudaMemsetAsync(allocation + quantBytes, 0, total - quantBytes, stream);
    QuantizeToMxfp8Kernel<<<(rows * (cols / 32) + 63) / 64, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(weight.cudaData), allocation, allocation + quantBytes,
        rows, cols, 1, scaleBytes, flag);
    int inexact = allowLossy ? 0 : 1;
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && !allowLossy)
        status = cudaMemcpyAsync(&inexact, flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) { FastllmCudaDirectFree(allocation); cudaGetLastError(); return false; }
    Mxfp8WeightCache old;
    if (cached != registry.entries.end()) old = cached->second;
    if (inexact) {
        FastllmCudaDirectFree(allocation);
        registry.entries[&weight] = {device, rows, cols, nullptr, nullptr, true};
    } else {
        quant = allocation; scales = allocation + quantBytes;
        registry.entries[&weight] = {device, rows, cols, allocation + quantBytes, allocation, false};
    }
    if (old.scales) {
        FastllmCudaSetDevice(old.device);
        FastllmCudaDirectFree(old.quant ? old.quant : old.scales);
        FastllmCudaSetDevice(device);
    }
    return !inexact;
}

using Mxfp8Config = flashinfer::gemm::CutlassGemmConfig;

Mxfp8Config MakeMxfp8Config() {
    return Mxfp8Config(flashinfer::gemm::CutlassTileConfigSM120::CtaShape128x128x128B,
                       flashinfer::gemm::MainloopScheduleType::AUTO,
                       flashinfer::gemm::EpilogueScheduleType::AUTO,
                       flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
                       /*swap_ab=*/false);
}

template <typename Output = __nv_bfloat16>
size_t Mxfp8DispatchWorkspace(const Mxfp8Config &config, int m, int n, int k, int batch = 1) {
    return flashinfer::gemm::genericMxfp8GemmKernelLauncherSm120<
            Output, cute::Int<128>, cute::Int<128>, cute::Int<128>, false>(
                nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, batch,
                config, nullptr, 0, cudaStreamPerThread, nullptr);
}

template <typename Output = __nv_bfloat16>
void Mxfp8DispatchGemm(const Mxfp8Config &config, void *output, const void *input,
                       const void *weight, const void *inputScales, const void *weightScales,
                       int m, int n, int k, char *workspace, size_t workspaceBytes,
                       cudaStream_t stream) {
    flashinfer::gemm::genericMxfp8GemmKernelLauncherSm120<
            Output, cute::Int<128>, cute::Int<128>, cute::Int<128>, false>(
                output, input, weight, inputScales, weightScales, m, n, k, 1,
                config, workspace, workspaceBytes, stream, nullptr);
}

}  // namespace

extern "C" void FastllmCudaReleaseMxfp8CutlassCache(const fastllm::Data *weight) {
    if (!weight) return;
    for (auto *registry : {&Mxfp8Registry(), &DraftHeadRegistry()}) {
        Mxfp8WeightCache owned;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            auto it = registry->entries.find(weight);
            if (it == registry->entries.end()) continue;
            owned = it->second;
            registry->entries.erase(it);
        }
        if (owned.scales) {
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(owned.device);
            FastllmCudaDirectFree(owned.quant ? owned.quant : owned.scales);
            FastllmCudaSetDevice(previous);
        }
    }
}

// Draft only: block-32 W8A8 with FP32 accumulation and FP32 logits.
// The shared target BF16 weight remains intact. Warm the immutable conversion
// before capture; transient operands use the graph-owned allocation pool.
extern "C" bool FastllmCudaTryFp8DraftHead(const fastllm::Data &input,
        const fastllm::Data &weight, fastllm::Data &output) {
    using namespace fastllm;
    const char *flag = std::getenv("FASTLLM_DSPARK_FP8_HEAD");
    if ((flag && flag[0] && std::strcmp(flag, "1") != 0) || !Mxfp8EnvEnabled() ||
        FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() > 121 ||
        &input == &output || &weight == &output || input.multiDeviceData || weight.multiDeviceData || output.multiDeviceData ||
        input.dataType != BFLOAT16 || weight.dataType != BFLOAT16 || weight.isFake ||
        input.dataDevice != CUDA || weight.dataDevice != CUDA || !input.cudaData || !weight.cudaData ||
        input.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} || weight.dataDeviceIds != input.dataDeviceIds ||
        input.dims.empty() || weight.dims.size() != 2 || input.dims.back() != weight.dims[1] ||
        weight.dims[0] <= 0 || weight.dims[1] <= 0 || weight.dims[0] % 32 || weight.dims[1] % 32 ||
        (reinterpret_cast<uintptr_t>(input.cudaData) & 15) || (reinterpret_cast<uintptr_t>(weight.cudaData) & 15)) return false;
    const int n = weight.dims[0], k = weight.dims[1];
    const uint64_t rows = input.Count(0) / k;
    if (rows < 1 || rows * uint64_t(k) > INT_MAX || rows * uint64_t(n) > INT_MAX ||
        weight.Count(0) > INT_MAX) return false;
    auto dense = [](const Data &d) {
        if (d.strides.size() != d.dims.size()) return false;
        uint64_t stride = 1;
        for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
            if (d.dims[i] <= 0 || d.strides[i] != stride) return false;
            stride *= d.dims[i];
        }
        return true;
    };
    if (!dense(input) || !dense(weight)) return false;
    auto overlap = [](const Data &a, const Data &b) {
        if (!a.cudaData || !b.cudaData) return false;
        const uintptr_t x = reinterpret_cast<uintptr_t>(a.cudaData), y = reinterpret_cast<uintptr_t>(b.cudaData);
        return x < y + b.GetBytes() && y < x + a.GetBytes();
    };
    if (overlap(output, input) || overlap(output, weight)) return false;
    const uint8_t *w = nullptr, *sw = nullptr;
    if (!EnsureMxfp8Weight(weight, n, k, w, sw, true)) return false;
    const auto config = MakeMxfp8Config();
    size_t workspaceBytes = 0;
    try { workspaceBytes = Mxfp8DispatchWorkspace<float>(config, int(rows), n, k); }
    catch (...) { FastllmCudaClearThreadError(); return false; }
    const size_t quantBytes = rows * k;
    const size_t scaleBytes = size_t((rows + 127) / 128) * size_t((k + 127) / 128) * 512;
    const size_t totalBytes = quantBytes + scaleBytes + workspaceBytes;
    if (totalBytes > INT_MAX) return false;
    // A shared temporary buffer can grow after capture and invalidate replay
    // addresses. Pooled Data storage is instead retained by the graph owner.
    Data scratchData(INT8, {int(totalBytes)});
    scratchData.ToDevice(CUDA, input.dataDeviceIds, false);
    scratchData.Allocate(false);
    auto *scratch = static_cast<uint8_t *>(scratchData.cudaData);
    if (!scratch) return false;
    auto dims = input.dims; dims.back() = n;
    output.dataType = FLOAT32; output.UpdateUnitSize(); output.Resize(dims);
    output.ToDevice(CUDA, input.dataDeviceIds); output.Allocate();
    cudaStream_t stream = cudaStreamPerThread;
    cudaMemsetAsync(scratch + quantBytes, 0, scaleBytes, stream);
    QuantizeToMxfp8Kernel<<<(int(rows) * (k / 32) + 63) / 64, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), scratch, scratch + quantBytes, int(rows), k, 1, scaleBytes);
    bool ok = output.cudaData && cudaGetLastError() == cudaSuccess;
    if (ok) try {
        Mxfp8DispatchGemm<float>(config, output.cudaData, scratch, w, scratch + quantBytes, sw,
                                int(rows), n, k, reinterpret_cast<char *>(scratch + quantBytes + scaleBytes), workspaceBytes, stream);
        ok = cudaGetLastError() == cudaSuccess;
    } catch (...) { FastllmCudaClearThreadError(); ok = false; }
    return ok;
}

extern "C" bool FastllmCudaTryMxfp8QuantizedLinear(const fastllm::Data &input,
        fastllm::Data &weight, fastllm::Data &output) {
    if (!Mxfp8EnvEnabled() || &input == &output || &weight == &output ||
        FastllmCudaGraphIsCapturingFast() ||
        FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() > 121 ||
        input.multiDeviceData || weight.multiDeviceData || output.multiDeviceData ||
        input.dataType != fastllm::DataType::BFLOAT16 || input.dims.empty() ||
        input.dataDevice != fastllm::DataDevice::CUDA || !input.cudaData ||
        (reinterpret_cast<uintptr_t>(input.cudaData) & 15) ||
        weight.dataDevice != fastllm::DataDevice::CUDA || !weight.cudaData ||
        weight.dataType != fastllm::DataType::FP8_E4M3 || weight.dims.size() != 2 ||
        weight.isFake || weight.blockM != 32 || weight.blockK != 32 ||
        input.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
        weight.dataDeviceIds != input.dataDeviceIds ||
        weight.dims[1] != input.dims.back() || input.dims.back() <= 0 ||
        input.dims.back() % 32 || weight.dims[0] <= 0 || weight.dims[0] % 32) return false;
    const int rows = input.Count(0) / input.dims.back();
    if (rows < 128) return false;
    const int m = rows, n = weight.dims[0], k = input.dims.back();

    const uint8_t *weightScales = nullptr;
    const uint8_t *weightData = static_cast<const uint8_t *>(weight.cudaData);
    if (!EnsureMxfp8WeightScales(weight, n, k, weightScales)) return false;

    auto dims = input.dims;
    dims.back() = weight.dims[0];
    output.dataType = fastllm::DataType::BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(dims);
    output.ToDevice(fastllm::DataDevice::CUDA, input.dataDeviceIds);
    output.Allocate();
    void *inputData = input.cudaData;
    void *outputData = output.cudaData;
    if (!outputData) return false;

    const int mTiles = (m + 127) / 128;
    const int kTiles = (k + 127) / 128;
    const size_t quantBytes = static_cast<size_t>(m) * k;
    const size_t scaleBytes = static_cast<size_t>(mTiles) * 128 * kTiles * 4;
    const auto config = MakeMxfp8Config();
    size_t workspaceBytes = 0;
    try {
        workspaceBytes = Mxfp8DispatchWorkspace(config, m, n, k);
    } catch (...) {
        FastllmCudaClearThreadError();
        return false;
    }
    size_t borrowedBytes = 0;
    bool ownWorkspace = false;
    const size_t totalBytes = quantBytes + scaleBytes + workspaceBytes;
    uint8_t *scratch = static_cast<uint8_t *>(
        FastllmBorrowCudaTempBuffer(totalBytes, &borrowedBytes, &ownWorkspace));
    if (scratch == nullptr || borrowedBytes < totalBytes) {
        FastllmReleaseCudaTempBuffer(scratch, ownWorkspace);
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;
    uint8_t *quantInput = scratch;
    uint8_t *inputScales = scratch + quantBytes;
    char *workspace = reinterpret_cast<char *>(scratch + quantBytes + scaleBytes);
    if (m % 128 || k % 128) cudaMemsetAsync(inputScales, 0, scaleBytes, stream);
    const int scaleCols = k / 32;
    const int tasks = m * scaleCols;
    QuantizeToMxfp8Kernel<<<(tasks + 63) / 64, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(inputData), quantInput, inputScales,
        m, k, 1, scaleBytes);

    bool ok = cudaGetLastError() == cudaSuccess;
    if (ok) {
        try {
            Mxfp8DispatchGemm(config, outputData, quantInput, weightData,
                              inputScales, weightScales, m, n, k, workspace,
                              workspaceBytes, stream);
            ok = cudaGetLastError() == cudaSuccess;
        } catch (const std::exception &error) {
            if (std::getenv("FASTLLM_CUDA_LINEAR_FP8_TRACE"))
                std::fprintf(stderr, "MXFP8 fallback: %s\n", error.what());
            ok = false;
            FastllmCudaClearThreadError();
        }
    }
    FastllmReleaseCudaTempBuffer(scratch, ownWorkspace);
    DeviceSync();
    return ok;
}

extern "C" bool FastllmCudaTryMxfp8WoA(const fastllm::Data &input,
        const fastllm::Data &weight, int groups, int rank, fastllm::Data &output) {
    const char *disable = std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM");
    if (!Mxfp8EnvEnabled() || &input == &output || &weight == &output ||
        FastllmCudaGraphIsCapturingFast() ||
        (disable && std::strcmp(disable, "0")) ||
        FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() > 121 ||
        input.dataType != fastllm::DataType::BFLOAT16 || input.dims.size() != 4 ||
        input.multiDeviceData || weight.multiDeviceData || output.multiDeviceData || weight.isFake ||
        input.dataDevice != fastllm::DataDevice::CUDA || !input.cudaData ||
        (reinterpret_cast<uintptr_t>(input.cudaData) & 15) ||
        weight.dataDevice != fastllm::DataDevice::CUDA || !weight.cudaData ||
        input.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
        weight.dataDeviceIds != input.dataDeviceIds ||
        (weight.dataType != fastllm::DataType::BFLOAT16 && weight.dataType != fastllm::DataType::FP8_E4M3) || weight.dims.size() != 2 ||
        groups <= 0 || rank <= 0 || rank % 128) return false;
    const int rows = input.dims[0] * input.dims[1], cols = weight.dims[1];
    if (rows < 128 || cols <= 0 || cols % 128 || weight.dims[0] != groups * rank ||
        input.Count(0) != (size_t)rows * groups * cols) return false;
    const uint8_t *weightScales = nullptr;
    const uint8_t *weightData = static_cast<const uint8_t *>(weight.cudaData);
    if (weight.dataType == fastllm::DataType::FP8_E4M3) {
        if (!EnsureMxfp8WeightScales(weight, groups * rank, cols, weightScales)) return false;
    } else if (!EnsureMxfp8Weight(weight, groups * rank, cols, weightData, weightScales)) return false;
    output.dataType = fastllm::DataType::BFLOAT16;
    output.UpdateUnitSize();
    output.Resize({input.dims[0], input.dims[1], groups * rank});
    output.ToDevice(fastllm::DataDevice::CUDA, input.dataDeviceIds);
    output.Allocate();
    const size_t quantBytes = (size_t)rows * groups * cols;
    const size_t scaleStride = (size_t)((rows + 127) / 128) * (cols / 128) * 512;
    const auto config = MakeMxfp8Config();
    const size_t workspaceBytes = Mxfp8DispatchWorkspace(config, rows, rank, cols, groups);
    const size_t total = quantBytes + groups * scaleStride + workspaceBytes;
    size_t borrowed = 0; bool owned = false;
    auto *scratch = static_cast<uint8_t *>(FastllmBorrowCudaTempBuffer(total, &borrowed, &owned));
    if (!scratch || borrowed < total) { FastllmReleaseCudaTempBuffer(scratch, owned); return false; }
    auto *scales = scratch + quantBytes;
    auto *workspace = reinterpret_cast<char *>(scales + groups * scaleStride);
    cudaStream_t stream = cudaStreamPerThread;
    if (rows % 128) cudaMemsetAsync(scales, 0, groups * scaleStride, stream);
    QuantizeToMxfp8Kernel<true><<<(rows * groups * (cols / 32) + 63) / 64, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), scratch, scales,
        rows, cols, groups, scaleStride);
    bool ok = cudaGetLastError() == cudaSuccess;
    using Gemm = flashinfer::gemm::DeviceGemmMxfp8GemmSm120___nv_bfloat16_128_128_128false::Gemm;
    if (ok) {
        Gemm gemm;
        auto args = flashinfer::gemm::prepareGemmArgsSm120___nv_bfloat16_128_128_128false<Gemm>(
            output.cudaData, scratch, weightData, scales, weightScales, rows, rank, cols, groups);
        // Batch all groups in one launch, including the short bounded-replay
        // tail. Keep the output interleaved, avoiding a separate transpose.
        args.epilogue.dC = cute::make_stride(int64_t(groups * rank), cute::_1{}, int64_t(rank));
        args.epilogue.dD = args.epilogue.dC;
        ok = gemm.can_implement(args) == cutlass::Status::kSuccess &&
             gemm.initialize(args, workspace, stream) == cutlass::Status::kSuccess &&
             gemm.run(args, workspace, stream, nullptr, true) == cutlass::Status::kSuccess;
    }
    FastllmReleaseCudaTempBuffer(scratch, owned);
    return ok && cudaGetLastError() == cudaSuccess;
}

#else

extern "C" void FastllmCudaReleaseMxfp8CutlassCache(const fastllm::Data *) {}
extern "C" bool FastllmCudaTryFp8DraftHead(const fastllm::Data &, const fastllm::Data &, fastllm::Data &) { return false; }

extern "C" bool FastllmCudaTryMxfp8QuantizedLinear(const fastllm::Data &,
        fastllm::Data &, fastllm::Data &) { return false; }
extern "C" bool FastllmCudaTryMxfp8WoA(const fastllm::Data &, const fastllm::Data &,
        int, int, fastllm::Data &) { return false; }

#endif
