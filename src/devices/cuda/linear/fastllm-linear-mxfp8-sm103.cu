// Native Blackwell MXFP8 Linear. Attention shards can own a compact scale layout;
// other weights use the original FP32 scales and the bounded Linear scratch.
#include "fastllm-cuda.cuh"
#include "fastllm-fp8-auxiliary.h"
#include "fastllm-linear-scratch.h"
#include "fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-bf16-lt.cuh"
#include <cuda.h>
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <map>
#include <cstring>
#include <stdexcept>

namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int kSMs = 148;
constexpr size_t kWorkspace = fastllm_bf16_lt::workspaceBytes;

// Copy four consecutive K32 exponents per word; transpose the row and K128
// indices with a bank-conflict-free shared tile. No persistent packed copy.
template <int TileRows, int TileGroups, int Threads>
__device__ __forceinline__ void PackWeightTile(const float4 *src, uint32_t *dst, int n,
                                               int k, int tilesPerRow, int tileIndex) {
    __shared__ uint32_t tile[TileRows][TileGroups + 1];
    const int groups = k / 128, firstGroup = tileIndex % tilesPerRow * TileGroups,
              firstRow = tileIndex / tilesPerRow * TileRows;
    for (int i = threadIdx.x; i < TileRows * TileGroups; i += Threads) {
        const int r = i / TileGroups, g = i % TileGroups;
        if (firstRow + r < n && firstGroup + g < groups) {
            const float4 v = src[size_t(firstRow + r) * groups + firstGroup + g];
            tile[r][g] = (__float_as_uint(v.x) >> 23) | ((__float_as_uint(v.y) >> 23) << 8) |
                         ((__float_as_uint(v.z) >> 23) << 16) | ((__float_as_uint(v.w) >> 23) << 24);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileGroups * TileRows; i += Threads) {
        const int g = i / TileRows, r = i % TileRows;
        if (firstRow + r < n && firstGroup + g < groups)
            dst[size_t(firstGroup + g) * n + firstRow + r] = tile[r][g];
    }
}

template <int TileRows, int TileGroups, int Threads>
__global__ void PrepareOperands(const float4 *src, uint32_t *dst, const __nv_bfloat16 *x, uint8_t *q,
                                uint8_t *s, int rows, int n, int k, int pitch, int tilesPerRow,
                                int weightTiles) {
    const unsigned thread = threadIdx.x;
    if (blockIdx.x < weightTiles) {
        PackWeightTile<TileRows, TileGroups, Threads>(src, dst, n, k, tilesPerRow, blockIdx.x);
        return;
    }

    const int item = (blockIdx.x - weightTiles) * blockDim.x + thread;
    const int row = item / (k / 8), col = item % (k / 8) * 8;
    if (row >= rows) return;
    alignas(16) __nv_bfloat16 v[8];
    *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(x + size_t(row) * k + col);
    uint32_t exponent;
    const uint2 encoded = fastllm_mxfp8::PackActivation(reinterpret_cast<const __nv_bfloat162 *>(v), exponent, __activemask());
    *reinterpret_cast<uint2 *>(q + size_t(row) * k + col) = encoded;
    if (!(item & 3)) s[(size_t(col / 128) * pitch + row) * 4 + (col / 32) % 4] = exponent;
}

// One CTA per verification row preserves AddRMSNorm's 256-lane reduction and
// both BF16 rounding points. The other CTAs transpose immutable weight scales.
__global__ void PrepareSwaQkv(__nv_bfloat16 *hidden, const __nv_bfloat16 *branch,
                              const float *gamma, float eps, const float4 *weightScales,
                              uint32_t *packed, uint8_t *qx, uint8_t *sx, int rows, int pitch,
                              const float *positions, float theta, __nv_bfloat162 *rotary) {
    // The GEMM can initialize its descriptors and tensor memory while these
    // CTAs prepare the operands. It waits before issuing its first TMA read.
    cudaTriggerProgrammaticLaunchCompletion();
    constexpr int width = 4096, channels = 1856, threads = 512;
    const int row = blockIdx.x, t = threadIdx.x;
    if (row == rows + channels / 16) {
        if (t < rows * 32) {
            const float angle = positions[t / 32] * powf(theta, -2.f * (t % 32) / 64.f);
            rotary[t] = __floats2bfloat162_rn(cosf(angle), sinf(angle));
        }
        return;
    }
    if (row >= rows) {
        PackWeightTile<16, 32, threads>(weightScales, packed, channels, width, 1, row - rows);
        return;
    }
    using Pair = __nv_bfloat162;
    union Vector {
        uint4 words;
        Pair pairs[4];
        __device__ Vector() {}
    };
    const int col = t * 8, index = row * width + col;
    Vector residual, addend;
    residual.words = *reinterpret_cast<const uint4 *>(hidden + index);
    addend.words = *reinterpret_cast<const uint4 *>(branch + index);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        residual.pairs[j] = __hadd2(residual.pairs[j], addend.pairs[j]);
    *reinterpret_cast<uint4 *>(hidden + index) = residual.words;

    __shared__ __align__(16) __nv_bfloat16 values[width];
    __shared__ float partial[8];
    *reinterpret_cast<uint4 *>(values + col) = residual.words;
    __syncthreads();
    float square = 0.f;
    if (t < 256) {
#pragma unroll
        for (int j = 0; j < width / 256; ++j) {
            const float v = float(values[t + j * 256]);
            square += v * v;
        }
        square = fastllm::cuda::WarpSum(square);
        if ((t & 31) == 0) partial[t / 32] = square;
    }
    __syncthreads();
    if (t < 32) {
        const float total = fastllm::cuda::WarpSum(t < 8 ? partial[t] : 0.f);
        if (t == 0) partial[0] = rsqrtf(total / width + eps);
    }
    __syncthreads();
    const float inv = partial[0];
    const float4 g0 = *reinterpret_cast<const float4 *>(gamma + col);
    const float4 g1 = *reinterpret_cast<const float4 *>(gamma + col + 4);
    const float gains[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(residual.pairs[j]);
        const auto scaled = __floats2bfloat162_rn(v.x * inv, v.y * inv);
        const float2 z = __bfloat1622float2(scaled);
        residual.pairs[j] = __floats2bfloat162_rn(z.x * gains[j * 2], z.y * gains[j * 2 + 1]);
    }
    uint32_t exponent;
    *reinterpret_cast<uint2 *>(qx + index) = fastllm_mxfp8::PackActivation(residual.pairs, exponent, __activemask());
    if ((t & 3) == 0) sx[(col / 128 * pitch + row) * 4 + col / 32 % 4] = exponent;
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

// Called by every CTA after the native GEMM has waited for operand preparation.
struct SwaIdentity : dg::epilogue::operators::Identity {
    CUTLASS_DEVICE explicit SwaIdentity(const dg::EpilogueOperatorArgs &args)
        : dg::epilogue::operators::Identity(args) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
};

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
    template<bool Swa = false> static auto Kernel() {
        return &dg::sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K, 32, 32, 32, 0, 0,
                                                 0, M, N, 128, 1, 128, 128, 128, Stages, 2, 128, 128, Cluster,
                                                 (!LargeTile && Cluster == 2), kSMs, !LargeTile, true, dg::GemmType::Normal, false,
                                                 cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                 cutlass::bfloat16_t, std::conditional_t<Swa, SwaIdentity, dg::epilogue::operators::Identity>>;
    }
    template<bool Swa = false> static bool Configure() {
        return cudaFuncSetAttribute(Kernel<Swa>(), cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)) ==
               cudaSuccess;
    }
    template<bool Overlap = false, class Prepare>
    static bool RunPrepared(void *q, void *w, void *sa, void *sw, void *y, int rows,
                            int n, int k, int pitch, Prepare prepare) {
        Desc a, b, as, bs, c;
        if (!Descriptor(a, q, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, 1, 128, LoadM, true) ||
            !Descriptor(b, w, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, n, 1, 128, LoadN, true) ||
            !Descriptor(as, sa, CU_TENSOR_MAP_DATA_TYPE_INT32, pitch, k / 128, 4, 128, 1, false) ||
            !Descriptor(bs, sw, CU_TENSOR_MAP_DATA_TYPE_INT32, n, k / 128, 4, ScaleN, 1, false) ||
            !Descriptor(c, y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows, 2, 64, LargeTile ? M : 16, true))
            return false;
        cudaLaunchAttribute attrs[2]{};
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {Cluster, 1, 1};
        if constexpr (Overlap) {
            attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[1].val.programmaticStreamSerializationAllowed = 1;
        }
        cudaLaunchConfig_t cfg{};
        // Small-row SWA has one M tile. Launch its live N tiles, leaving
        // enough SMs for the cooperative consumer's independent prologue.
        static_assert(!Overlap || (M == 16 && N == 128 && Cluster == 1));
        cfg.gridDim = dim3(Overlap ? (n + N - 1) / N : kSMs);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = sizeof(SharedStorage);
        cfg.stream = cudaStreamPerThread;
        cfg.attrs = attrs;
        cfg.numAttrs = Overlap ? 2 : 1;
        prepare();
        auto status = cudaLaunchKernelEx(&cfg, Kernel<Overlap>(), (int *)nullptr, (uint32_t)rows, (uint32_t)n,
                                         (uint32_t)k, dg::EpilogueOperatorArgs{}, a, b, as, bs, c);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
    static bool Run(void *q, void *w, void *sa, void *sw, void *y, int rows, int n, int k, int pitch) {
        return RunPrepared(q, w, sa, sw, y, rows, n, k, pitch, [] {});
    }

};
struct ProjectionSupport {
    bool smallRows = false;
    bool swaSignal = false;
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
        support.swaSignal = Projection<16, 1>::Configure<true>();
        support.columns256 = Projection<128, 2, 256>::Configure();
        support.columns128 = Projection<128, 2, 128>::Configure();
        support.columns64 = Projection<128, 2, 64>::Configure();
    }
    if (!support.columns256 || !support.columns128 || !support.columns64 || !support.swaSignal) cudaGetLastError();
    cache.emplace(device, support);
    return support;
}
} // namespace

const float *FastllmCudaVerifyMxfp8Scales(const fastllm::Data &w, int channels, int inner, int device) {
    if (w.dataType != fastllm::FP8_E4M3 || w.IsRepacked || w.isFake || w.blockK != 1 ||
        w.blockM != 32 || w.scales.size() != size_t(channels) * inner / 32 || w.extraCudaData.empty())
        return nullptr;
    const auto *s = static_cast<const float *>(w.extraCudaData[0]);
    if (!s || (uintptr_t(s) & 15) || !FastllmCudaHasExactMxfp8Scales(s,device)) return nullptr;
    auto *state = fastllm_bf16_lt::GetState(device,true);
    const bool capture = FastllmCudaGraphIsCapturing();
    if (!state || !state->eager || !state->eager->data ||
        (capture && !fastllm_bf16_lt::SerialGraphWorkspace()) || !GetProjectionSupport(device,capture).smallRows)
        return nullptr;
    return s;
}

bool FastllmCudaMxfp8SwaQkv(void *hidden, const void *branch, const float *gamma, float eps,
                            const fastllm::Data &weight, const float *scales, void *output, int rows,
                            const float *positions, float theta, void *rotary) {
    constexpr int k = 4096, n = 1856;
    constexpr size_t scaleBytes = size_t(n) * k / 32;
    static_assert(scaleBytes + 8 * (k + k / 32) <= kWorkspace);
    if (rows < 2 || rows > 8 || !scales || !positions || !rotary) return false;
    const int device = FastllmCudaGetDevice();
    const bool capture = FastllmCudaGraphIsCapturing();
    auto *state = fastllm_bf16_lt::GetState(device, true);
    if (!state || !GetProjectionSupport(device, capture).swaSignal) return false;
    auto *scratch = static_cast<uint8_t *>(capture ? fastllm_bf16_lt::CaptureWorkspace(*state)
                                         : state->eager ? state->eager->data : nullptr);
    if (!scratch) return false;
    const int pitch = (rows + 3) / 4 * 4;
    auto *qx = scratch + scaleBytes;
    auto *sx = qx + pitch * k;
    // 1856 output channels end on an unpaired N tile. Encode the same
    // descriptors as native Linear before the first residual write.
    return Projection<16, 1>::RunPrepared<true>(qx, weight.cudaData, sx, scratch, output, rows, n, k, pitch, [&] {
        PrepareSwaQkv<<<rows + n / 16 + 1, 512, 0, cudaStreamPerThread>>>(
            static_cast<__nv_bfloat16 *>(hidden), static_cast<const __nv_bfloat16 *>(branch),
            gamma, eps, reinterpret_cast<const float4 *>(scales), reinterpret_cast<uint32_t *>(scratch),
            qx, sx, rows, pitch, positions, theta, static_cast<__nv_bfloat162 *>(rotary));
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    });
}

// Both packed producers and standalone Linear use the same tile geometry.
static auto SelectProjection(int &count, int rows, int n, bool capturing, const ProjectionSupport &support) {
    const int rowTiles = (count - 1) / 128 + 1;
    const bool pairedM = rowTiles % 2 == 0;
    const int64_t tiles = int64_t(count / 128) * (n / 256);
    // Preserve the wide-column prefill path and use it in graphs when
    // there are at least 64 complete 128x256 output tiles.
    const bool enoughWork = tiles >= 64 || (!capturing && rows >= 1024);
    const bool paired = ((n + 127) / 128) % 2 == 0;
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
    return run;
}

bool FastllmCudaMxfp8LinearSM103(const void *x, fastllm::Data &w, void *y, int rows, int k, int n) {
    // Streaming measurements favor GEMV on narrow single-row TP shards.
    if (rows == 1 && (k < 4096 || n < 4096 || size_t(k) * n < (size_t(24) << 20))) return false;
    if (w.dataType != fastllm::FP8_E4M3 || w.IsRepacked || w.isFake || w.blockK != 1 || w.blockM != 32 ||
        w.dims.size() != 2 || w.dims[0] != n || w.dims[1] != k) return false;
    auto *compact = (uint32_t *)FastllmCudaCompactMxfp8(w);
    if (FastllmCudaRuntimeArch() != 103 || rows < 1 || k < 128 || k % 128 || n <= 0 || n % 64 || !x || !y ||
        !w.cudaData || (!compact && (w.extraCudaData.empty() || !w.extraCudaData[0])) ||
        ((uintptr_t(x) | uintptr_t(y) | uintptr_t(w.cudaData) | uintptr_t(w.extraCudaData[0])) & 15))
        return false;
    const size_t scaleCount = size_t(n) * k / 32;
    const size_t scaleArea = compact ? 0 : ((scaleCount + 255) & ~size_t(255));
    if (w.scales.size() != scaleCount) return false;
    const int device = FastllmCudaGetDevice();
    const bool capturing = FastllmCudaGraphIsCapturing();
    auto *state = fastllm_bf16_lt::GetState(device, true);
    // The first eager request on a worker may precede any BF16 Linear. Use
    // the bounded Linear pool instead of materializing another scale layout.
    if (compact && !capturing && (!state || !state->eager))
        state = fastllm_bf16_lt::GetState(device, false);
    if (!state) return false;
    const auto support = GetProjectionSupport(device, capturing);
    if (!support.smallRows || (!compact && !FastllmCudaHasExactMxfp8Scales(w.extraCudaData[0], device))) return false;
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
    auto *packed = compact ? compact : reinterpret_cast<uint32_t *>(base);
    auto *qx = base + scaleArea;
    auto *sx = qx + ((size_t(limit) + 3) / 4 * 4) * k;
    // Smaller tiles distribute the scale transpose for wide input or output projections.
    const bool smallTile = k >= 8192 || (rows >= 128 && rows <= 512 && k >= 4096 && n >= 8192);
    const int tile = smallTile ? 16 : 32;
    const int threads = smallTile ? 128 : 256;
    const int tilesPerRow = (k / 128 + tile - 1) / tile;
    auto prepare = smallTile ? PrepareOperands<16, 16, 128> : PrepareOperands<32, 32, 256>;
    for (int start = 0; start < rows;) {
        int count = std::min(rows - start, limit);
        auto run = SelectProjection(count, rows, n, capturing, support);
        const int pitch = (count + 3) / 4 * 4;
        const int weightTiles = !compact && start == 0 ? tilesPerRow * ((n + tile - 1) / tile) : 0;
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

const void *FastllmCudaCompactMxfp8(const fastllm::Data &w) {
    return w.dataType == fastllm::FP8_E4M3 && !w.isFake && !w.IsRepacked &&
           w.blockK == 1 && w.blockM == 32 && w.extraCudaData.size() > FastllmCompactMxfp8Slot
        ? w.extraCudaData[FastllmCompactMxfp8Slot] : nullptr;
}

namespace {
constexpr int kPreparedProjectionRows = 16;

void PrepareCompactScales(fastllm::Data &w) {
    if (w.dataType != fastllm::FP8_E4M3 || w.isFake || w.IsRepacked || !w.cudaData ||
        w.blockK != 1 || w.blockM != 32 || w.dims.size() != 2 || FastllmCudaCompactMxfp8(w) ||
        (!w.extraCudaData.empty() && w.extraCudaData[0])) return;
    const int n = w.dims[0], k = w.dims[1];
    if (k < 4096 || n < 4096 || k % 128 || n % 64 || size_t(n) * k < (size_t(24) << 20) ||
        w.scales.size() != size_t(n) * k / 32) return;
    std::vector<uint32_t> packed(w.scales.size() / 4);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < k / 128; ++group) {
            uint32_t word = 0;
            for (int i = 0; i < 4; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &w.scales[size_t(row) * (k / 32) + group * 4 + i], sizeof(bits));
                if ((bits & 0x807fffffu) || bits == 0 || bits >= 0x7f800000u) return;
                word |= (bits >> 23) << (8 * i);
            }
            packed[size_t(group) * n + row] = word;
        }
    }
    w.extraCudaData.resize(std::max(w.extraCudaData.size(), FastllmCompactMxfp8Slot + 1), nullptr);
    void *scales = nullptr;
    const size_t bytes = packed.size() * sizeof(uint32_t);
    auto status = cudaMalloc(&scales, bytes);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaMemcpy(scales, packed.data(), bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(scales);
        throw std::runtime_error(cudaGetErrorString(status));
    }
    w.extraCudaData[FastllmCompactMxfp8Slot] = scales;
}

int PreparedProjectionRows(const fastllm::Data &w, bool verifying) {
    if (!FastllmCudaCompactMxfp8(w)) return 0;
    return verifying ? int(kWorkspace / (w.dims[1] + w.dims[1] / 32)) : kPreparedProjectionRows;
}

bool ProjectPrepared(FastllmMxfp8Activation input, const fastllm::Data &w, void *output) {
    auto *scales = const_cast<void *>(FastllmCudaCompactMxfp8(w));
    if (!scales) return false;
    int count = input.rows;
    const bool capturing = FastllmCudaGraphIsCapturing();
    const auto support = GetProjectionSupport(FastllmCudaGetDevice(), capturing);
    if (!support.smallRows) return false;
    auto run = SelectProjection(count, input.rows, w.dims[0], capturing, support);
    if (count != input.rows) return false; // Linear handles a separate trailing tile.
    return run(input.values, w.cudaData, input.scales, scales, output,
               input.rows, w.dims[0], input.width, input.pitch);
}
} // namespace

const FastllmMxfp8ProjectionBackend *FastllmCudaBlackwellMxfp8Projection() {
    static const FastllmMxfp8ProjectionBackend backend{PrepareCompactScales, PreparedProjectionRows, ProjectPrepared};
    return GetProjectionSupport(FastllmCudaGetDevice(), FastllmCudaGraphIsCapturing()).smallRows
        ? &backend : nullptr;
}
