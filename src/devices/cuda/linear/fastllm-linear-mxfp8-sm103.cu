// Native Blackwell MXFP8 Linear. Both GEMV and GEMM retain the original FP8
// values and FP32 scales. The block-scaled operands live only in borrowed scratch.
#include "fastllm-cuda.cuh"
#include "fastllm-fp8-auxiliary.h"
#include "fastllm-linear-scratch.h"
#include "fastllm-bf16-lt.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int kSMs = 148;
constexpr size_t kWorkspace = fastllm_bf16_lt::workspaceBytes;

// Copy four consecutive K32 exponents per word; transpose the row and K128
// indices with a bank-conflict-free shared tile. No persistent packed copy.
__global__ void PrepareOperands(const float4 *src, uint32_t *dst, const __nv_bfloat16 *x, uint8_t *q,
                                uint8_t *s, int rows, int n, int k, int pitch, int tilesPerRow,
                                int weightTiles) {
    const unsigned thread = threadIdx.x;
    if (blockIdx.x < weightTiles) {
        __shared__ uint32_t tile[32][33];
        const int lane = thread % 32, warp = thread / 32, groups = k / 128;
        const int groupTile = blockIdx.x % tilesPerRow * 32;
        const int rowTile = blockIdx.x / tilesPerRow * 32;
        for (int r = warp; r < 32; r += 8) {
            if (groupTile + lane < groups && rowTile + r < n) {
                const float4 v = src[size_t(rowTile + r) * groups + groupTile + lane];
                tile[r][lane] = (__float_as_uint(v.x) >> 23) | ((__float_as_uint(v.y) >> 23) << 8) |
                                ((__float_as_uint(v.z) >> 23) << 16) | ((__float_as_uint(v.w) >> 23) << 24);
            }
        }
        __syncthreads();
        for (int r = warp; r < 32; r += 8)
            if (rowTile + lane < n && groupTile + r < groups)
                dst[size_t(groupTile + r) * n + rowTile + lane] = tile[lane][r];
        return;
    }

    const int item = (blockIdx.x - weightTiles) * blockDim.x + thread;
    const int row = item / (k / 8), col = item % (k / 8) * 8;
    if (row >= rows) return;
    alignas(16) __nv_bfloat16 v[8];
    *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(x + size_t(row) * k + col);
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        maximum = fmaxf(maximum, fabsf(float(v[i])));
    unsigned mask = __activemask();
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2, 4));
    uint32_t bits = __float_as_uint(maximum / 448.f);
    uint32_t exponent = min(254u, max(1u, (bits >> 23) + ((bits & 0x7fffffu) != 0)));
    float inv = ldexpf(1.f, 127 - int(exponent));
    alignas(8) __nv_fp8_e4m3 encoded[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        encoded[i] = __nv_fp8_e4m3(float(v[i]) * inv);
    *reinterpret_cast<uint2 *>(q + size_t(row) * k + col) = *reinterpret_cast<uint2 *>(encoded);
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

template <int M, int Cluster> struct Projection {
    // Large-M prefill uses the 128x256x128, five-stage Blackwell configuration
    // with B multicast. Small rows use the transposed MMA and A multicast.
    static constexpr bool Prefill = M == 128;
    static constexpr int N = Prefill ? 256 : 128;
    static constexpr int Stages = Prefill ? 5 : M == 16 ? (Cluster == 2 ? 12 : 11) : 10;
    static constexpr int Shared = Prefill ? 205116 : 230204;
    static auto Kernel() {
        return &dg::sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K, 32, 32, 32, 0, 0,
                                                 0, M, N, 128, 1, 128, 128, 128, Stages, 2, 128, 128, Cluster,
                                                 (!Prefill && Cluster == 2), kSMs, !Prefill, true, dg::GemmType::Normal, false,
                                                 cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                 cutlass::bfloat16_t, dg::epilogue::operators::Identity>;
    }
    static bool Configure() {
        return cudaFuncSetAttribute(Kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) ==
               cudaSuccess;
    }
    static bool Run(void *q, void *w, void *sa, void *sw, void *y, int rows, int n, int k, int pitch) {
        Desc a, b, as, bs, c;
        if (!Descriptor(a, q, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, 1, 128, Prefill ? M : M / Cluster, true) ||
            !Descriptor(b, w, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, n, 1, 128, 128, true) ||
            !Descriptor(as, sa, CU_TENSOR_MAP_DATA_TYPE_INT32, pitch, k / 128, 4, 128, 1, false) ||
            !Descriptor(bs, sw, CU_TENSOR_MAP_DATA_TYPE_INT32, n, k / 128, 4, N, 1, false) ||
            !Descriptor(c, y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows, 2, 64, Prefill ? M : 16, true))
            return false;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Cluster, 1, 1};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(kSMs);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = Shared;
        cfg.stream = cudaStreamPerThread;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        auto status = cudaLaunchKernelEx(&cfg, Kernel(), (int *)nullptr, (uint32_t)rows, (uint32_t)n,
                                         (uint32_t)k, dg::EpilogueOperatorArgs{}, a, b, as, bs, c);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
};
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

bool FastllmCudaMxfp8LinearSM103(const void *x, fastllm::Data &w, void *y, int rows, int k, int n) {
    // Streaming measurements favor GEMV on narrow single-row TP shards.
    if (rows == 1 && (k < 4096 || n < 4096 || size_t(k) * n < (size_t(24) << 20))) return false;
    if (w.dataType != fastllm::FP8_E4M3 || w.IsRepacked || w.isFake || w.blockK != 1 || w.blockM != 32 ||
        w.dims.size() != 2 || w.dims[0] != n || w.dims[1] != k) return false;
    if (FastllmCudaRuntimeArch() != 103 || rows < 1 || k < 128 || k % 128 || n <= 0 || n % 64 || !x || !y ||
        !w.cudaData || w.extraCudaData.empty() || !w.extraCudaData[0] ||
        ((uintptr_t(x) | uintptr_t(y) | uintptr_t(w.cudaData) | uintptr_t(w.extraCudaData[0])) & 15))
        return false;
    const size_t scaleCount = size_t(n) * k / 32;
    const size_t scaleArea = (scaleCount + 255) & ~size_t(255);
    if (w.scales.size() != scaleCount) return false;
    const int device = FastllmCudaGetDevice();
    // Look up the existing Linear lease without creating a handle, allocating
    // a workspace or refilling the spare pool. CaptureWorkspace pins its owner.
    auto *state = fastllm_bf16_lt::GetState(device, true);
    if (!state) return false;
    const bool capturing = FastllmCudaGraphIsCapturing();
    if (!Available(device, capturing) || !FastllmCudaHasExactMxfp8Scales(w.extraCudaData[0], device)) return false;
    auto *base = static_cast<uint8_t *>(capturing      ? fastllm_bf16_lt::CaptureWorkspace(*state)
                                        : state->eager ? state->eager->data
                                                       : nullptr);
    size_t capacity = kWorkspace;
    const auto *scratch = fastllm::CudaLinearScratchScope::Get();
    if (!capturing && rows >= 1024 && scratch && scratch->cudaData && !scratch->isFake &&
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
    auto *packed = reinterpret_cast<uint32_t *>(base);
    auto *qx = base + scaleArea;
    auto *sx = qx + ((size_t(limit) + 3) / 4 * 4) * k;
    const int tilesPerRow = (k / 128 + 31) / 32;
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
        const int weightTiles = start == 0 ? tilesPerRow * ((n + 31) / 32) : 0;
        const int activationTiles = (size_t(count) * k / 8 + 255) / 256;
        PrepareOperands<<<weightTiles + activationTiles, 256, 0, cudaStreamPerThread>>>(
            static_cast<const float4 *>(w.extraCudaData[0]), packed,
            static_cast<const __nv_bfloat16 *>(x) + size_t(start) * k, qx, sx, count, n, k, pitch, tilesPerRow,
            weightTiles);
        auto run = paired ? Projection<32, 2>::Run : Projection<32, 1>::Run;
        if (largeTile) run = Projection<128, 2>::Run;
        else if (count <= 16) run = paired ? Projection<16, 2>::Run : Projection<16, 1>::Run;
        if (!run(qx, w.cudaData, sx, packed, static_cast<__nv_bfloat16 *>(y) + size_t(start) * n, count, n, k,
                 pitch))
            return false;
        start += count;
    }
    return true;
}
