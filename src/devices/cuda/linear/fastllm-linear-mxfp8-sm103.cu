// Native Blackwell MXFP8 Linear. Both GEMV and GEMM retain the original FP8
// values and FP32 scales. The block-scaled operands live only in borrowed scratch.
#include <array>
#include "fastllm-cuda.cuh"
#include "fastllm-fp8-auxiliary.h"
#include "fastllm-linear-scratch.h"
#include "fastllm-bf16-lt.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int kSMs = 148;
constexpr size_t kWorkspace = fastllm_bf16_lt::workspaceBytes;

// Copy four consecutive K32 exponents per word; transpose the row and K128
// indices with a bank-conflict-free shared tile. No persistent packed copy.
template <int TileRows, int TileGroups, int Threads>
__global__ void PrepareOperands(const float4 *src, uint32_t *dst, const __nv_bfloat16 *x, uint8_t *q,
                                uint8_t *s, int rows, int n, int k, int pitch, int tilesPerRow,
                                int weightTiles) {
    const unsigned thread = threadIdx.x;
    if (blockIdx.x < weightTiles) {
        __shared__ uint32_t tile[TileRows][TileGroups + 1];
        const int groups = k / 128, firstGroup = blockIdx.x % tilesPerRow * TileGroups,
                  firstRow = blockIdx.x / tilesPerRow * TileRows;
        for (int i = thread; i < TileRows * TileGroups; i += Threads) {
            const int r = i / TileGroups, g = i % TileGroups;
            if (firstRow + r < n && firstGroup + g < groups) {
                const float4 v = src[size_t(firstRow + r) * groups + firstGroup + g];
                tile[r][g] = (__float_as_uint(v.x) >> 23) | ((__float_as_uint(v.y) >> 23) << 8) |
                             ((__float_as_uint(v.z) >> 23) << 16) | ((__float_as_uint(v.w) >> 23) << 24);
            }
        }
        __syncthreads();
        for (int i = thread; i < TileGroups * TileRows; i += Threads) {
            const int g = i / TileRows, r = i % TileRows;
            if (firstRow + r < n && firstGroup + g < groups)
                dst[size_t(firstGroup + g) * n + firstRow + r] = tile[r][g];
        }
        return;
    }

    const int item = (blockIdx.x - weightTiles) * blockDim.x + thread;
    const int row = item / (k / 8), col = item % (k / 8) * 8;
    if (row >= rows) return;
    alignas(16) __nv_bfloat16 v[8];
    *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(x + size_t(row) * k + col);
    __nv_bfloat162 maxPair = __habs2(*reinterpret_cast<const __nv_bfloat162 *>(v));
#pragma unroll
    for (int i = 1; i < 4; ++i)
        maxPair = __hmax2(maxPair, __habs2(reinterpret_cast<const __nv_bfloat162 *>(v)[i]));
    const float2 pair = __bfloat1622float2(maxPair);
    float maximum = fmaxf(0.f, fmaxf(pair.x, pair.y));
    unsigned mask = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2, 4));
    uint32_t bits = __float_as_uint(maximum / 448.f);
    uint32_t exponent = min(254u, max(1u, (bits >> 23) + ((bits & 0x7fffffu) != 0)));
    float inv = __uint_as_float(exponent == 254 ? 0x00400000u : (254u - exponent) << 23);
    union {
        uint2 word;
        __nv_fp8x2_storage_t pair[4];
    } encoded;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(v)[i]);
        f.x *= inv;
        f.y *= inv;
        encoded.pair[i] = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2 *>(q + size_t(row) * k + col) = encoded.word;
    if (!(item & 3)) s[(size_t(col / 128) * pitch + row) * 4 + (col / 32) % 4] = exponent;
}

using Desc = cute::TmaDescriptor;
bool Descriptor(Desc &d, void *ptr, CUtensorMapDataType type, int inner, int outer, int bytes, int boxX,
                int boxY, bool swizzle) {
    const cuuint64_t dims[] = {uint64_t(inner), uint64_t(outer)}, stride[] = {uint64_t(inner) * bytes};
    const cuuint32_t box[] = {uint32_t(boxX), uint32_t(boxY)}, step[] = {1, 1};
    return cuTensorMapEncodeTiled(
               reinterpret_cast<CUtensorMap *>(&d), type, 2, ptr, dims, stride, box, step,
               CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int M, int Cluster, bool FixedQA = false, bool PanelB = false, bool VerifyOB = false, int OutputSlices = 1> struct Projection {
    // Large-M prefill uses the 128x256x128, five-stage Blackwell configuration
    // with B multicast. Small rows use the transposed MMA and A multicast.
    static constexpr bool Prefill = M == 128;
    static constexpr int N = Prefill ? 256 : 128;
    static constexpr int Grid = VerifyOB ? 5120 / N : FixedQA ? 1792 / N : kSMs;
    static constexpr int Stages = OutputSlices > 1 ? 11 : FixedQA ? 10 : Prefill ? 5 : M == 16 ? (Cluster == 2 ? 12 : 11) : 10;
    static constexpr int Shared = Prefill ? 205116 : 230204;
    static auto Kernel() {
        return &dg::sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K, 32, 32, 32, VerifyOB ? 6 : 0, VerifyOB ? 5120 : FixedQA ? 1792 : 0,
                                                 VerifyOB ? 8192 : FixedQA ? 5120 : 0, M, N, 128, 1, 128, 128, 128, Stages, 2, 128, 128, Cluster,
                                                 (!Prefill && Cluster == 2), Grid, !Prefill, true, dg::GemmType::Normal, false,
                                                 cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                 cutlass::bfloat16_t, dg::epilogue::operators::Identity, PanelB, VerifyOB, OutputSlices>;
    }
    static bool WeightDescriptor(Desc &map, void *weight, int n, int k) {
        if constexpr (OutputSlices > 1) {
            const cuuint64_t dims[] = {128, 128 / OutputSlices, OutputSlices, uint64_t(k / 128), uint64_t(n / 128)};
            const cuuint64_t strides[] = {128, 16384 / OutputSlices, 16384, uint64_t(128) * k};
            const cuuint32_t tile[] = {128, 128, 1, 1, 1}, step[] = {1, 1, 1, 1, 1};
            return cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&map), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                5, weight, dims, strides, tile, step, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
        }
        return Descriptor(map, weight, CU_TENSOR_MAP_DATA_TYPE_UINT8, PanelB ? 128 : k,
                          PanelB ? n * (k / 128) : n, 1, 128, 128, true);
    }
    static bool Configure() {
        return cudaFuncSetAttribute(Kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) ==
               cudaSuccess;
    }
    static bool Run(void *q, void *w, void *sa, void *sw, void *y, int rows, int n, int k, int pitch) {
        Desc a, b, as, bs, c;
        if (!Descriptor(a, q, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, 1, 128, Prefill ? M : M / Cluster, true) ||
            !WeightDescriptor(b, w, n, k) ||
            !Descriptor(as, sa, CU_TENSOR_MAP_DATA_TYPE_INT32, pitch, k / 128, 4, 128, 1, false) ||
            !Descriptor(bs, sw, CU_TENSOR_MAP_DATA_TYPE_INT32, n, k / 128, 4, N, 1, false) ||
            !Descriptor(c, y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows, 2, 64, Prefill ? M : 16, true))
            return false;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Cluster, 1, 1};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(Grid, 1, OutputSlices);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = Shared;
        cfg.stream = cudaStreamPerThread;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        dg::EpilogueOperatorArgs ep{};
        if constexpr (OutputSlices > 1) ep.directOutput = y;
        auto status = cudaLaunchKernelEx(&cfg, Kernel(), (int *)nullptr, (uint32_t)rows, (uint32_t)n,
                                         (uint32_t)k, ep, a, b, as, bs, c);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
};
// For these three decode shapes a single CTA per cluster avoids multicast
// setup. The block-scaled MMA order and BF16 epilogue remain identical.
bool V41SingleCtaProjection(int rows, int n, int k) {
    static const bool enabled = [] {
        const char *value = std::getenv("FASTLLM_DSV41_SINGLE_CTA_PROJECTIONS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && rows <= 16 && ((n == 1792 && k == 5120) ||
        (n == 32768 && k == 1280) || (n == 5120 && k == 8192));
}

bool UseDecodeProjections() {
    static const bool enabled = std::getenv("FASTLLM_DSV41_TUNED_PROJECTIONS") &&
        std::strcmp(std::getenv("FASTLLM_DSV41_TUNED_PROJECTIONS"), "1") == 0;
    return enabled;
}

bool FixedQAAvailable(int device, bool capturing) {
    static thread_local std::map<int, bool> ready;
    const auto it = ready.find(device);
    if (it != ready.end()) return it->second;
    if (capturing) return false;
    const bool ok = Projection<16, 1, true>::Configure();
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}

bool Available(int device, bool capturing) {
    static thread_local std::map<int, bool> ready;
    auto it = ready.find(device);
    if (it != ready.end()) return it->second;
    if (capturing) return false;
    int sms = 0;
    const bool ok = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                    sms == kSMs && Projection<16, 1>::Configure() && Projection<16, 2>::Configure() &&
                    Projection<32, 1>::Configure() && Projection<32, 2>::Configure();
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}
bool PrefillAvailable(int device) {
    static thread_local std::map<int, bool> ready;
    auto [it, inserted] = ready.try_emplace(device, false);
    if (inserted) {
        it->second = Projection<128, 2>::Configure();
        if (!it->second) cudaGetLastError();
    }
    return it->second;
}
} // namespace

static bool Mxfp8Projection(const void *x, fastllm::Data &w, void *y, int rows, int k, int n, bool block32, fastllm::Data *owned = nullptr) {
    // Streaming measurements favor GEMV on narrow single-row TP shards.
    if (!block32 && rows == 1 && (k < 4096 || n < 4096 || size_t(k) * n < (size_t(24) << 20))) return false;
    if (w.dataType != fastllm::FP8_E4M3 || w.IsRepacked || w.isFake || w.blockK != (block32 ? 32 : 1) || w.blockM != 32 ||
        w.dims.size() != 2 || w.dims[0] != n || w.dims[1] != k) return false;
    if (FastllmCudaRuntimeArch() != 103 || rows < 1 || k < 128 || k % 128 || n <= 0 || n % 64 || !x || !y ||
        !w.cudaData || w.extraCudaData.empty() || !w.extraCudaData[0] ||
        ((uintptr_t(x) | uintptr_t(y) | uintptr_t(w.cudaData) | uintptr_t(w.extraCudaData[0])) & 15))
        return false;
    const size_t scaleCount = size_t(n) * k / 32;
    const size_t scaleArea = block32 ? 0 : (scaleCount + 255) & ~size_t(255);
    if (w.scales.size() != scaleCount / w.blockK) return false;
    const int device = FastllmCudaGetDevice();
    // Look up the existing Linear lease without creating a handle, allocating
    // a workspace or refilling the spare pool. CaptureWorkspace pins its owner.
    const bool capturing = FastllmCudaGraphIsCapturing();
    if (!Available(device, capturing) || !FastllmCudaHasExactMxfp8Scales(w.extraCudaData[0], device)) return false;
    uint8_t *base = nullptr;
    size_t capacity = kWorkspace;
    if (owned) {
        // A V4.1 graph slot owns its workspace through all child graph
        // captures/replays. Another slot or independent batch gets its own.
        const int packedRows = (std::min(rows, 256) + 3) / 4 * 4;
        const size_t bytes = size_t(packedRows) * (k + k / 32);
        if (capturing && (!owned->cudaData || owned->expansionBytes < bytes)) return false;
        owned->dataType = fastllm::BFLOAT16;
        owned->UpdateUnitSize();
        owned->Resize({int(bytes / 2)});
        owned->ToDevice(fastllm::CUDA, {device}, false);
        owned->Allocate();
        base = static_cast<uint8_t *>(owned->cudaData);
        capacity = bytes;
    } else {
        auto *state = fastllm_bf16_lt::GetState(device, true);
        if (!state) return false;
        base = static_cast<uint8_t *>(capturing ? fastllm_bf16_lt::CaptureWorkspace(*state) :
                                     state->eager ? state->eager->data : nullptr);
    }
    const auto *scratch = fastllm::CudaLinearScratchScope::Get();
    if (!owned && !capturing && rows >= 1024 && scratch && scratch->cudaData && !scratch->isFake &&
        !scratch->cudaDataBorrowed && (reinterpret_cast<uintptr_t>(scratch->cudaData) & 15) == 0 &&
        scratch->dataDevice == fastllm::DataDevice::CUDA && scratch->dataDeviceIds.size() == 1 &&
        scratch->dataDeviceIds[0] == device && scratch->expansionBytes > capacity) {
        base = static_cast<uint8_t *>(scratch->cudaData);
        capacity = scratch->expansionBytes;
    }
    if (!base || scaleArea >= capacity) return false;
    const size_t maxRows = (capacity - scaleArea) / (size_t(k) + k / 32) / 4 * 4;
    if (maxRows < 4) return false;
    const int limit = static_cast<int>(std::min<size_t>(rows, maxRows));
    auto *packed = block32 ? const_cast<uint32_t *>(FastllmCudaBlock32PackedScales(w, device, capturing))
                           : reinterpret_cast<uint32_t *>(base);
    if (!packed) return false;
    auto *qx = base + scaleArea;
    auto *sx = qx + ((size_t(limit) + 3) / 4 * 4) * k;
    // Narrower tiles expose more parallel copies for wide projections.
    const bool wideK = k >= 8192;
    const int tile = wideK ? 16 : 32;
    const int threads = wideK ? 128 : 256;
    const int tilesPerRow = (k / 128 + tile - 1) / tile;
    auto prepare = wideK ? PrepareOperands<16, 16, 128> : PrepareOperands<32, 32, 256>;
    // A pair of CTAs must cover two complete N tiles. A trailing odd tile
    // needs a single-CTA kernel or the multicast barrier cannot complete.
    bool paired = ((n + 127) / 128) % 2 == 0;
    // A paired, non-transposed 128x256 tile shares B across two M tiles.
    // Keep the short-row kernels for tails and shapes without a full cluster.
    const bool prefill = !capturing && rows >= 1024 && n % 256 == 0 && limit >= 256 && PrefillAvailable(device);
    for (int start = 0; start < rows;) {
        int count = std::min(rows - start, limit);
        // Multicast needs an even number of M tiles. A partially filled
        // second tile is legal. For a short, narrow GEMM, splitting off
        // a lone tile costs more than using the existing single-launch path.
        const bool pairedM = ((count - 1) / 128 + 1) % 2 == 0;
        const bool fullWave = int64_t(count / 128) * (n / 256) >= kSMs;
        const bool largeTile = prefill && count >= 256 && (pairedM || fullWave);
        if (largeTile && !pairedM) count = count / 256 * 256;
        const int pitch = (count + 3) / 4 * 4;
        const int weightTiles = !block32 && start == 0 ? tilesPerRow * ((n + tile - 1) / tile) : 0;
        const int activationTiles = (size_t(count) * k / 8 + threads - 1) / threads;
        prepare<<<weightTiles + activationTiles, threads, 0, cudaStreamPerThread>>>(
            static_cast<const float4 *>(w.extraCudaData[0]), packed,
            static_cast<const __nv_bfloat16 *>(x) + size_t(start) * k, qx, sx, count, n, k, pitch, tilesPerRow,
            weightTiles);
        auto run = paired ? Projection<32, 2>::Run : Projection<32, 1>::Run;
        if (largeTile) run = Projection<128, 2>::Run;
        else if (count <= 16)
            run = paired && !(block32 && V41SingleCtaProjection(count, n, k)) ?
                Projection<16, 2>::Run : Projection<16, 1>::Run;
        if (owned && block32 && count <= 16 && n == 1792 && k == 5120 && UseDecodeProjections() && FixedQAAvailable(device, capturing))
            run = Projection<16, 1, true>::Run;
        if (!run(qx, w.cudaData, sx, packed, static_cast<__nv_bfloat16 *>(y) + size_t(start) * n, count, n, k,
                 pitch))
            return false;
        start += count;
    }
    return true;
}

void *FastllmCudaV41PrepareBlock32Kernel() {
    return reinterpret_cast<void *>(PrepareOperands<32, 32, 256>);
}

bool FastllmCudaMxfp8LinearSM103(const void *x, fastllm::Data &w, void *y, int rows, int k, int n) {
    return Mxfp8Projection(x, w, y, rows, k, n, false);
}

bool FastllmCudaTryBlock32Mxfp8Sm103(const fastllm::Data &x, fastllm::Data &w, fastllm::Data &y, fastllm::Data &workspace) {
    using namespace fastllm;
    if (FastllmCudaRuntimeArch() != 103 || &x == &y || &w == &y ||
        x.multiDeviceData || w.multiDeviceData || y.multiDeviceData || x.dataType != BFLOAT16 ||
        w.dataType != FP8_E4M3 || w.blockK != 32 || w.blockM != 32 || w.isFake || w.IsRepacked ||
        x.dataDevice != CUDA || w.dataDevice != CUDA || !x.cudaData || !w.cudaData ||
        x.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} || w.dataDeviceIds != x.dataDeviceIds ||
        w.dims.size() != 2 || x.dims.empty() || x.dims.back() != w.dims[1]) return false;
    const int n = w.dims[0], k = w.dims[1];
    if (k <= 0 || k % 128 || n <= 0 || n % 64 || x.Count(0) / k > INT_MAX) return false;
    const int rows = x.Count(0) / k;
    if (!rows || size_t(n) * k < (size_t(4) << 20)) return false;
    if (w.extraCudaData.empty() || !w.extraCudaData[0]) {
        if (FastllmCudaGraphIsCapturing()) return false;
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(w, *GetEmptyData(), n);
    }
    auto dims = x.dims; dims.back() = n;
    y.dataType = BFLOAT16; y.UpdateUnitSize(); y.Resize(dims);
    y.ToDevice(CUDA, x.dataDeviceIds); y.Allocate();
    return Mxfp8Projection(x.cudaData, w, y.cudaData, rows, k, n, true, &workspace);
}

#include "fastllm-v41-output-sm103.cuh"
