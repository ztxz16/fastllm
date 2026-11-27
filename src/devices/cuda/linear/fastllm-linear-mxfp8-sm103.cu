// Native Blackwell MXFP8 Linear. Attention shards can own a compact scale layout;
// other weights use the original FP32 scales and the bounded Linear scratch.
#include "fastllm-cuda.cuh"
#include "fastllm-fp8-auxiliary.h"
#include "fastllm-linear-scratch.h"
#include "fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-bf16-lt.cuh"
#include "devices/cuda/fastllm-swa-prepare.cuh"
#include <cuda.h>
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <map>
#include <cstring>
#include <stdexcept>

namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int kSMs = 148;
constexpr size_t kWorkspace = fastllm_bf16_lt::workspaceBytes;

using fastllm_swa_qkv::PackWeightTile;
using fastllm_swa_qkv::PrepareSwaQkv;

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

using Desc = cute::TmaDescriptor;
bool Descriptor(Desc &d, void *ptr, CUtensorMapDataType type, int inner, int outer, int bytes, int boxX,
                int boxY, bool swizzle, int pitch = 0) {
    const cuuint64_t dims[] = {uint64_t(inner), uint64_t(outer)}, stride[] = {uint64_t(pitch ? pitch : inner) * bytes};
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
            !Descriptor(as, sa, CU_TENSOR_MAP_DATA_TYPE_INT32, (rows + 3) / 4 * 4, k / 128, 4, 128, 1, false, pitch) ||
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
        if (!prepare()) return false;
        auto status = cudaLaunchKernelEx(&cfg, Kernel<Overlap>(), (int *)nullptr, (uint32_t)rows, (uint32_t)n,
                                         (uint32_t)k, dg::EpilogueOperatorArgs{}, a, b, as, bs, c);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
    static bool Run(void *q, void *w, void *sa, void *sw, void *y, int rows, int n, int k, int pitch) {
        return RunPrepared(q, w, sa, sw, y, rows, n, k, pitch, [] { return true; });
    }

};
template <int M, int Cluster, bool FixedQA = false, bool PanelB = false, bool VerifyOB = false, int OutputSlices = 1> struct V41Projection {
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
        cudaLaunchAttribute attributes[2]{};
        auto &attr = attributes[0];
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Cluster, 1, 1};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(Grid, 1, OutputSlices);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = Shared;
        cfg.stream = cudaStreamPerThread;
        cfg.attrs = attributes;
        cfg.numAttrs = 1;
        if constexpr (VerifyOB && OutputSlices == 1) {
            attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[1].val.programmaticStreamSerializationAllowed = 1;
            cfg.numAttrs = 2;
        }
        dg::EpilogueOperatorArgs ep{};
        if constexpr (VerifyOB) ep.directOutput = y;
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
    const bool ok = V41Projection<16, 1, true>::Configure();
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}

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
                            const float *positions, float theta, void *rotary, bool afterExpertReduce) {
    constexpr int k = 4096, n = 1856;
    constexpr size_t scaleBytes = size_t(n) * k / 32;
    static_assert(scaleBytes + 16 * (k + k / 32) <= kWorkspace);
    if (rows < 2 || rows > 16 || !scales || !positions || !rotary) return false;
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
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(rows + n / 16 + 1);
        cfg.blockDim = dim3(512);
        cfg.stream = cudaStreamPerThread;
        cudaLaunchAttribute attr{};
        if (afterExpertReduce) {
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
        }
        auto kernel = afterExpertReduce ? PrepareSwaQkv<true> : PrepareSwaQkv<false>;
        const auto status = cudaLaunchKernelEx(&cfg, kernel,
            static_cast<__nv_bfloat16 *>(hidden), static_cast<const __nv_bfloat16 *>(branch),
            gamma, eps, reinterpret_cast<const float4 *>(scales), reinterpret_cast<uint32_t *>(scratch),
            qx, sx, rows, pitch, positions, theta, static_cast<__nv_bfloat162 *>(rotary),
            static_cast<const __nv_bfloat16 *>(nullptr), static_cast<__nv_bfloat16 *>(nullptr));
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
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
    const auto support = GetProjectionSupport(device, capturing);
    if (!support.smallRows || (!compact && !FastllmCudaHasExactMxfp8Scales(w.extraCudaData[0], device))) return false;
    uint8_t *base = nullptr;
    if (auto *state = fastllm_bf16_lt::GetState(device, true)) {
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
    // A cold eager worker can use the caller's scratch without opening the
    // Linear pool. Compact weights need that pool only if no four rows fit.
    if (compact && !capturing && capacity < 4 * (size_t(k) + k / 32)) {
        auto *state = fastllm_bf16_lt::GetState(device, false);
        if (state && state->eager) {
            base = static_cast<uint8_t *>(state->eager->data);
            capacity = kWorkspace;
        }
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

bool CanPackPrefill(const fastllm::Data &w, int rows) {
    if (!FastllmCudaCompactMxfp8(w) || FastllmCudaGraphIsCapturingFast()) return false;
    return rows > 0 && GetProjectionSupport(FastllmCudaGetDevice(), false).smallRows;
}

bool ProjectPrepared(FastllmMxfp8Activation input, const fastllm::Data &w, void *output) {
    auto *scales = const_cast<void *>(FastllmCudaCompactMxfp8(w));
    if (!scales) return false;
    int count = input.rows;
    const bool capturing = FastllmCudaGraphIsCapturing();
    const auto support = GetProjectionSupport(FastllmCudaGetDevice(), capturing);
    if (!support.smallRows) return false;
    auto run = SelectProjection(count, input.rows, w.dims[0], capturing, support);
    const int n = w.dims[0], k = input.width, tail = input.rows - count;
    auto prefix = [&] {
        return run(input.values, w.cudaData, input.scales, scales, output, count, n, k, input.pitch);
    };
    if (!tail) return prefix();
    // SelectProjection splits only the N256 bulk cluster, on a complete M pair.
    // Encode the tail first so an unavailable descriptor cannot leave a partial output.
    auto *q = input.values + size_t(count) * k;
    auto *s = input.scales + size_t(count) * sizeof(uint32_t);
    auto *y = static_cast<__nv_bfloat16 *>(output) + size_t(count) * n;
    if (tail <= 16)
        return Projection<16, 2>::RunPrepared(q, w.cudaData, s, scales, y, tail, n, k, input.pitch, prefix);
    return Projection<32, 2>::RunPrepared(q, w.cudaData, s, scales, y, tail, n, k, input.pitch, prefix);
}
bool ProjectPreparedPdl(FastllmMxfp8Activation input, const fastllm::Data &weight, void *output) {
    auto *scales = const_cast<void *>(FastllmCudaCompactMxfp8(weight));
    if (!scales || input.rows < 1 || input.rows > 16 ||
        !GetProjectionSupport(FastllmCudaGetDevice(), FastllmCudaGraphIsCapturing()).swaSignal)
        return false;
    return Projection<16, 1>::RunPrepared<true>(input.values, weight.cudaData, input.scales, scales,
        output, input.rows, weight.dims[0], input.width, input.pitch, [] { return true; });
}

} // namespace

const FastllmMxfp8ProjectionBackend *FastllmCudaBlackwellMxfp8Projection() {
    static const auto backend = [] {
        FastllmMxfp8ProjectionBackend ops{PrepareCompactScales, PreparedProjectionRows, ProjectPrepared, CanPackPrefill};
        ops.projectPdl = ProjectPreparedPdl;
        return ops;
    }();
    return GetProjectionSupport(FastllmCudaGetDevice(), FastllmCudaGraphIsCapturing()).smallRows
        ? &backend : nullptr;
}

void *FastllmCudaV41PrepareBlock32Kernel() {
    return reinterpret_cast<void *>(PrepareOperands<32, 32, 256>);
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
    const int device = FastllmCudaGetDevice();
    const bool capturing = FastllmCudaGraphIsCapturing();
    const auto support = GetProjectionSupport(device, capturing);
    if (!support.smallRows || w.scales.size() != size_t(n / 32) * (k / 32) ||
        !y.cudaData || ((uintptr_t(x.cudaData) | uintptr_t(w.cudaData) | uintptr_t(y.cudaData) |
                        uintptr_t(w.extraCudaData[0])) & 15)) return false;
    const uint32_t *packed = FastllmCudaBlock32PackedScales(w, device, capturing);
    if (!packed) return false;

    // The graph slot owns this arena until its last replay completes.
    const int limit = std::min(rows, 256);
    const int packedRows = (limit + 3) / 4 * 4;
    const size_t bytes = size_t(packedRows) * (k + k / 32);
    if (capturing && (!workspace.cudaData || workspace.expansionBytes < bytes)) return false;
    workspace.dataType = BFLOAT16;
    workspace.UpdateUnitSize();
    workspace.Resize({int(bytes / 2)});
    workspace.ToDevice(CUDA, {device}, false);
    workspace.Allocate();
    auto *codes = static_cast<uint8_t *>(workspace.cudaData);
    auto *scales = codes + size_t(packedRows) * k;
    const int threads = k >= 8192 ? 128 : 256;
    const auto prepare = k >= 8192 ? PrepareOperands<16, 16, 128> : PrepareOperands<32, 32, 256>;
    for (int start = 0; start < rows;) {
        int count = std::min(rows - start, limit);
        auto run = SelectProjection(count, rows, n, capturing, support);
        if (count <= 16 && V41SingleCtaProjection(count, n, k)) run = Projection<16, 1>::Run;
        if (count <= 16 && n == 1792 && k == 5120 && UseDecodeProjections() && FixedQAAvailable(device, capturing))
            run = V41Projection<16, 1, true>::Run;
        const int pitch = (count + 3) / 4 * 4;
        const size_t tiles = (size_t(count) * k / 8 + threads - 1) / threads;
        prepare<<<tiles, threads, 0, cudaStreamPerThread>>>(nullptr, nullptr,
            static_cast<const __nv_bfloat16 *>(x.cudaData) + size_t(start) * k, codes, scales, count, n, k, pitch, 0, 0);
        if (!run(codes, w.cudaData, scales, const_cast<uint32_t *>(packed),
                 static_cast<__nv_bfloat16 *>(y.cudaData) + size_t(start) * n, count, n, k, pitch)) return false;
        start += count;
    }
    return true;
}

#include "fastllm-v41-output-sm103.cuh"
