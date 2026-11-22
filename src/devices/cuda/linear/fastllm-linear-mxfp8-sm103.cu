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

template <int M, int Cluster, int N = 128> struct Projection {
    // Full M tiles share weights between CTAs; short rows use the transposed MMA.
    static constexpr bool LargeTile = M == 128;
    static constexpr int LoadM = LargeTile ? M : M / Cluster;
    static constexpr int LoadN = LargeTile ? N / Cluster : N;
    static constexpr int ScaleN = N < 128 ? 128 : N;
    static constexpr int Stages = LargeTile ? (N == 64 ? 9 : N == 128 ? 7 : 5)
                                            : (M == 16 ? (Cluster == 2 ? 12 : 11) : 10);
    using SharedStorage = dg::layout::SM100FP8FP4GemmSharedStorage<
        Stages, 2, 2, LoadM, LoadN, 128, LargeTile ? M : 16, LargeTile ? 64 : N, 128, ScaleN, 1,
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
    static auto Kernel() {
        return &dg::sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K, 32, 32, 32, 0, 0,
                                                 0, M, N, 128, 1, 128, 128, 128, Stages, 2, 128, 128, Cluster,
                                                 (!LargeTile && Cluster == 2), kSMs, !LargeTile, true, dg::GemmType::Normal, false,
                                                 cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                 cutlass::bfloat16_t, dg::epilogue::operators::Identity>;
    }
    static bool Configure() {
        return cudaFuncSetAttribute(Kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)) ==
               cudaSuccess;
    }
    static bool Run(void *q, void *w, void *sa, void *sw, void *y, int rows, int n, int k, int pitch) {
        Desc a, b, as, bs, c;
        if (!Descriptor(a, q, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, 1, 128, LoadM, true) ||
            !Descriptor(b, w, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, n, 1, 128, LoadN, true) ||
            !Descriptor(as, sa, CU_TENSOR_MAP_DATA_TYPE_INT32, pitch, k / 128, 4, 128, 1, false) ||
            !Descriptor(bs, sw, CU_TENSOR_MAP_DATA_TYPE_INT32, n, k / 128, 4, ScaleN, 1, false) ||
            !Descriptor(c, y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows, 2, 64, LargeTile ? M : 16, true))
            return false;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Cluster, 1, 1};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(kSMs);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = sizeof(SharedStorage);
        cfg.stream = cudaStreamPerThread;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        auto status = cudaLaunchKernelEx(&cfg, Kernel(), (int *)nullptr, (uint32_t)rows, (uint32_t)n,
                                         (uint32_t)k, dg::EpilogueOperatorArgs{}, a, b, as, bs, c);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
};
struct ProjectionSupport {
    bool smallRows = false;
    bool columns256 = false;
    bool columns128 = false;
    bool columns64 = false;
};

ProjectionSupport GetProjectionSupport(int device, bool capturing) {
    static thread_local std::map<int, ProjectionSupport> cache;
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    if (capturing) return {};
    int sms = 0;
    ProjectionSupport support;
    support.smallRows = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                        sms == kSMs && Projection<16, 1>::Configure() && Projection<16, 2>::Configure() &&
                        Projection<32, 1>::Configure() && Projection<32, 2>::Configure();
    // Warm every supported shape outside capture, including on a single-row warmup.
    if (support.smallRows) {
        support.columns256 = Projection<128, 2, 256>::Configure();
        support.columns128 = Projection<128, 2, 128>::Configure();
        support.columns64 = Projection<128, 2, 64>::Configure();
    }
    if (!support.columns256 || !support.columns128 || !support.columns64) cudaGetLastError();
    cache.emplace(device, support);
    return support;
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
    const bool capturing = FastllmCudaGraphIsCapturing();
    const auto support = GetProjectionSupport(device, capturing);
    if (!support.smallRows || !FastllmCudaHasExactMxfp8Scales(w.extraCudaData[0], device)) return false;
    // Eager prefill can supply the entire workspace on a worker that has never
    // run cuBLASLt. Graphs can only use storage pinned to the capture.
    auto *state = fastllm_bf16_lt::GetState(device, true);
    uint8_t *base = nullptr;
    if (state) {
        if (capturing) base = static_cast<uint8_t *>(fastllm_bf16_lt::CaptureWorkspace(*state));
        else if (state->eager) base = static_cast<uint8_t *>(state->eager->data);
    }
    size_t capacity = base ? kWorkspace : 0;
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
    // Smaller tiles distribute the scale transpose for wide input or output projections.
    const bool smallTile = k >= 8192 || (rows >= 128 && rows <= 512 && k >= 4096 && n >= 8192);
    const int tile = smallTile ? 16 : 32;
    const int threads = smallTile ? 128 : 256;
    const int tilesPerRow = (k / 128 + tile - 1) / tile;
    auto prepare = smallTile ? PrepareOperands<16, 16, 128> : PrepareOperands<32, 32, 256>;
    // A pair of CTAs must cover two complete N tiles. A trailing odd tile
    // needs a single-CTA kernel or the multicast barrier cannot complete.
    const bool paired = ((n + 127) / 128) % 2 == 0;
    for (int start = 0; start < rows;) {
        int count = std::min(rows - start, limit);
        // Multicast needs an even number of M tiles. A partially filled
        // second tile is legal. For a short, narrow GEMM, splitting off
        // a lone tile costs more than using the existing single-launch path.
        const int rowTiles = (count - 1) / 128 + 1;
        const bool pairedM = rowTiles % 2 == 0;
        const int64_t tiles = int64_t(count / 128) * (n / 256);
        // Preserve the wide-column prefill path and use it in graphs when
        // there are at least 64 complete 128x256 output tiles.
        const bool enoughWork = tiles >= 64 || (!capturing && rows >= 1024);
        auto run = paired ? Projection<32, 2>::Run : Projection<32, 1>::Run;
        // Prefer a single well-occupied wave. Two adjacent M tiles must be
        // present for weight multicast, including when the last tile is partial.
        const int64_t tiles64 = int64_t(rowTiles) * (n / 64);
        if (pairedM && tiles64 >= 96 && tiles64 <= kSMs && support.columns64) {
            run = Projection<128, 2, 64>::Run;
        } else if (pairedM && tiles64 > kSMs && tiles64 <= 2 * kSMs && n % 128 == 0 && support.columns128) {
            run = Projection<128, 2, 128>::Run;
        } else if (support.columns256 && n % 256 == 0 && count >= 256 && enoughWork && (pairedM || tiles >= kSMs)) {
            if (!pairedM) count = count / 256 * 256;
            run = Projection<128, 2, 256>::Run;
        } else if (count <= 16) {
            run = paired ? Projection<16, 2>::Run : Projection<16, 1>::Run;
        }
        const int pitch = (count + 3) / 4 * 4;
        const int weightTiles = start == 0 ? tilesPerRow * ((n + tile - 1) / tile) : 0;
        const int activationTiles = (size_t(count) * k / 8 + threads - 1) / threads;
        prepare<<<weightTiles + activationTiles, threads, 0, cudaStreamPerThread>>>(
            static_cast<const float4 *>(w.extraCudaData[0]), packed,
            static_cast<const __nv_bfloat16 *>(x) + size_t(start) * k, qx, sx, count, n, k, pitch, tilesPerRow,
            weightTiles);
        if (!run(qx, w.cudaData, sx, packed, static_cast<__nv_bfloat16 *>(y) + size_t(start) * n, count, n, k,
                 pitch))
            return false;
        start += count;
    }
    return true;
}
