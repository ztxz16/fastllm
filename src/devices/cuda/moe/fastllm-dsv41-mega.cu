// Host adapter for the pinned upstream SM100 MegaMoE kernel. No JIT runtime.
#include "devices/cuda/fastllm-dsv41-mega.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int H = 5120, I = 2304, E = 384, R = 8, K = 6;
constexpr int DraftExperts = 128, DraftTopK = 3;
constexpr int CAP = FastllmDsv41MegaMaxRows / R, BM = 16, BN = 128, BK = 128, SMS = 148;
// All ranks may route all source tokens to the same destination. Include
// tile padding for every local expert and the scheduler ring alignment.
constexpr int RING = 3840, SFRING = RING / BM * 128;
constexpr int STAGES = 11;
std::mutex readyMutex;
std::set<const void *> readyKeys, sharedKeys;
// Same fixed and per-stage sizes as upstream get_pipeline_config_for_mega_moe.
constexpr int SMEM = 2048 + 4 * H + 4096 + 8 * 8 * 4 +
    2 * sizeof(dg::sched::TaskInfo<true>) + (4 + 4 + 16 + 4) * 8 + 4 +
    STAGES * (8 * 128 + 128 * 128 + 128 * 4 * 2 + 16);
static_assert(SMEM <= 232448);
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(
        std::string("DSV41 MegaMoE: ") + cudaGetErrorString(status));
}
void *allocate(size_t bytes) { void *p = nullptr; check(cudaMalloc(&p, bytes)); return p; }
dg::layout::MegaMoEBuffer buffer(void *base, uint32_t shared = 0, int experts = E) {
    if (experts == DraftExperts) return {base, H, I, R, DraftExperts, CAP, DraftTopK, RING, SFRING, true, shared};
    return {base, H, I, R, E, CAP, K, RING, SFRING, true, shared};
}
using Desc = cute::TmaDescriptor;
void descriptor(Desc &out, CUtensorMapDataType type, void *ptr, int rank,
                const cuuint64_t *dims, const cuuint64_t *strides,
                const cuuint32_t *box, int swizzle) {
    cuuint32_t step[] = {1, 1, 1, 1};
    auto mode = swizzle == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE;
    auto status = cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&out), type,
        rank, ptr, dims, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
        mode, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) throw std::runtime_error(
        "DSV41 MegaMoE TMA descriptor: " + std::to_string(status));
}
void desc2(Desc &out, void *ptr, int inner, int outer, int bi, int bo,
           int bytes, int swizzle) {
    cuuint64_t dims[] = {(uint64_t)inner, (uint64_t)outer};
    cuuint64_t strides[] = {(uint64_t)inner * bytes};
    cuuint32_t box[] = {(uint32_t)bi, (uint32_t)bo};
    descriptor(out, bytes == 4 ? CU_TENSOR_MAP_DATA_TYPE_INT32 :
        CU_TENSOR_MAP_DATA_TYPE_UINT8, ptr, 2, dims, strides, box, swizzle);
}
void descWeight(Desc &out, void *ptr, int n, int k, int localExperts) {
    cuuint64_t dims[] = {(uint64_t)k, (uint64_t)n / 2, 2, (uint64_t)localExperts};
    cuuint64_t strides[] = {(uint64_t)k / 2, (uint64_t)n * k / 4, (uint64_t)n * k / 2};
    cuuint32_t box[] = {128, 128, 1, 1};
    descriptor(out, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, ptr, 4, dims, strides, box, 128);
}
// FastLLM compact MXFP4 stores row-major packed nibbles followed by UE8M0
// scale bytes. Upstream requires interleaved gate/up rows and UTCCP SF layout.
__global__ void pack(const uint8_t *src, uint8_t *dst, uint8_t *sf,
                     int n, int k, bool gate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * k / 2) {
        const int row = idx / (k / 2), col = idx % (k / 2);
        const int source = gate ? (row / 16) * 8 + row % 8 + (row % 16 >= 8 ? n / 2 : 0) : row;
        dst[idx] = src[source * (k / 2) + col];
    }
    if (idx < n * k / 32) {
        const int row = idx / (k / 32), col = idx % (k / 32);
        const int source = gate ? (row / 16) * 8 + row % 8 + (row % 16 >= 8 ? n / 2 : 0) : row;
        const int transposed = (row & ~127) + (row % 32) * 4 + (row % 128) / 32;
        sf[(col / 4 * n + transposed) * 4 + col % 4] = src[n * k / 2 + source * (k / 32) + col];
    }
}
// Shared experts retain the original MXFP8 values and block-32 scales.
__global__ void packShared(const uint8_t *src, const uint8_t *scales, uint8_t *dst,
                           uint8_t *packed, int n, int k, bool gate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*k) return;
    const int row = i/k;
    const int source = gate ? row/16*8 + row%8 + (row%16 >= 8 ? n/2 : 0) : row;
    dst[i] = src[source*k + i%k];
    if (i%32 == 0) {
        const int group = (i%k)/32;
        const int transposed = (row & ~127) + (row%32)*4 + (row%128)/32;
        packed[(group/4*n + transposed)*4 + group%4] = scales[(source/32)*(k/32) + group];
    }
}
constexpr int sharedSfRows = dg::layout::get_num_max_shared_sf_tokens(CAP);
struct Maps { Desc a1, s1, w1, ws1, d1, a2, s2, w2, ws2; };
// Shard replicated TP tokens by row modulo EP size. Remap FastLLM's modulo
// expert placement to the contiguous expert IDs expected by upstream.
template <int Experts = E, int TopK = K>
__device__ __forceinline__ void writeRoute(int slot, const int32_t *ids, const float *scores,
        int64_t *qi, float *qs, int row, int localRow) {
    const int expert = ids[row * TopK + slot];
    qi[localRow * TopK + slot] = expert < 0 ? -1 : (expert % R) * (Experts / R) + expert / R;
    qs[localRow * TopK + slot] = scores[row * TopK + slot];
}
__global__ void publishLocalRoutes(const int32_t *ids, const float *scores, int64_t *qi, float *qs) {
    const int row = blockIdx.x, lane = threadIdx.x;
    if (lane < K) writeRoute(lane, ids, scores, qi, qs, row, row);
}
template <bool PackOnly = false, int Experts = E, int TopK = K>
__global__ void prepare(const __nv_bfloat16 *x, const int32_t *ids, const float *scores,
    uint8_t *qx, uint8_t *sf, int64_t *qi, float *qs, int rank, int rows, bool ownerOutput, uint8_t *sharedSf = nullptr, bool sourceLocal = false) {
    const int localRow = blockIdx.x / (H / 32), group = blockIdx.x % (H / 32);
    const int row = sourceLocal ? localRow : ownerOutput ? rank * (rows / R) + min(rank, rows % R) + localRow : localRow * R + rank;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    float value = __bfloat162float(x[row * H + group * 32 + lane]);
    float amax = fabsf(value);
    for (int mask = 16; mask; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    const uint32_t exp = dg::math::get_ue8m0_sf_exp(amax);
    const float inverse = dg::math::get_ue8m0_sf_inv<float>(exp);
    qx[localRow * H + group * 32 + lane] = __nv_fp8_e4m3(value * inverse).__x;
    if (lane == 0) {
        sf[localRow * (H / 32) + group] = exp;
        if (sharedSf) {
            const int rowIndex = localRow / BM * 128 + (localRow % BM) * 4;
            sharedSf[(group/4 * sharedSfRows + rowIndex)*4 + group%4] = exp;
        }
    }
    if constexpr (!PackOnly) {
        if (group == 0) {
            if (lane < TopK) writeRoute<Experts, TopK>(lane, ids, scores, qi, qs, row, localRow);
        }
    }
}
__global__ void scatter(dg::layout::SymBuffer<R> sym, const int32_t *ids,
                       float *out, int rows, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * H) {
        const int row = i / H;
        const auto local = dg::layout::MegaMoEBuffer(reinterpret_cast<void *>(sym.base), H, I, R, E,
            CAP, K, RING, SFRING, true, 0);
        const auto combine = reinterpret_cast<const __nv_bfloat16 *>(
            reinterpret_cast<int64_t>(local.combine_token_buffer.base) + sym.offsets[row % R]);
        // Preserve Marlin's FP32 sum of this rank's expert contributions. Read
        // our completed L2 writes from the token owner's combine slots, avoiding
        // the upstream final BF16 sum and a changed shared-expert reduction order.
        float value = 0.f;
        for (int slot = 0; slot < K; ++slot) {
            const int expert = ids[row * K + slot];
            if (expert >= 0 && expert % R == rank)
                value += __bfloat162float(combine[(slot * CAP + row / R) * H + i % H]);
        }
        out[i] = value;
    }
}
// The local MegaMoE kernel has waited for all six peer contributions to
// each locally owned token. Sum them in the original expert-rank order and
// publish this contiguous slice with one all-gather instead of an all-reduce.
template <bool WithShared = false, bool WithPost = false, int TopK = K>
__global__ void combineOwnedRows(const __nv_bfloat16 *slots, const int32_t *ids,
                                 void *out, int count, int start, const __nv_bfloat16 *residual = nullptr,
                                 const float *post = nullptr, const float *comb = nullptr, const int *routeOrder = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * H) return;
    const int local = i / H, column = i % H, row = start + local;
    float partial[R] = {};
#pragma unroll
    for (int t = 0; t < TopK; ++t) {
        const int id = ids[row * TopK + t];
        if (id >= 0) partial[id % R] += __bfloat162float(slots[(t * CAP + local) * H + column]);
    }
    float total = partial[0];
#pragma unroll
    for (int r = 1; r < R; ++r) total += partial[r];
    if constexpr (WithShared) {
        const float shared = __bfloat162float(slots[(TopK * CAP + local) * H + column]);
        const auto value = __float2bfloat16_rn(total + shared);
        if constexpr (WithPost) {
            auto *hidden = static_cast<__nv_bfloat16 *>(out) + local * 4 * H + column;
            residual += local * 4 * H + column;
            float channels[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) channels[c] = __bfloat162float(residual[c*H]);
#pragma unroll
            for (int target = 0; target < 4; ++target) {
                float sum = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    sum = __fadd_rn(sum, __fmul_rn(channels[c], comb[local*16 + c*4 + target]));
                hidden[target*H] = __float2bfloat16_rn(__fadd_rn(sum,
                    __fmul_rn(post[local*4 + target], __bfloat162float(value))));
            }
        } else {
            static_cast<__nv_bfloat16 *>(out)[i] = value;
        }
    } else {
        static_cast<float *>(out)[row * H + column] = total;
    }
}
template <int Shared = 0, int ActiveSms = SMS, bool RegisterPull = false, bool SingleHistogram = false, int Experts = E, int TopK = K> auto kernel() {
    return &dg::sm100_fp8_fp4_mega_moe_impl<CAP, H, I, Experts, Shared, TopK, BM, BN, BK,
        8, 8, 128, 128, RING, SFRING, STAGES, H, 128, 128, 256, ActiveSms, R,
        10.0f, false, cutlass::detail::float_e2m1_unpacksmem_t, true, (Shared != 0), RegisterPull, SingleHistogram>;
}
} // namespace

struct FastllmDsv41MegaLayer {
    bool registerPull = false, singleHistogram = false;
    cudaStream_t packStream = nullptr;
    cudaEvent_t inputReady = nullptr, packetReady = nullptr;
    bool canPackEarly = false;
    int *routeOrder = nullptr;
    int device = -1, rank = -1, experts = E;
    const void *cacheKey = nullptr;
    void *gate = nullptr, *down = nullptr, *gateSf = nullptr, *downSf = nullptr;
    void *workspace = nullptr;
    uint8_t *domains = nullptr;
    std::array<int64_t, R> peers{};
    Maps routed, combinedRouted, shared;
    void *sharedGate = nullptr, *sharedDown = nullptr, *sharedGateSf = nullptr, *sharedDownSf = nullptr;
    ~FastllmDsv41MegaLayer() {
        { std::lock_guard<std::mutex> lock(readyMutex); readyKeys.erase(cacheKey); sharedKeys.erase(cacheKey); }
        int previous = 0; cudaGetDevice(&previous);
        if (device >= 0) cudaSetDevice(device);
        if (routeOrder) cudaFree(routeOrder);
        if (packetReady) cudaEventDestroy(packetReady);
        if (inputReady) cudaEventDestroy(inputReady);
        if (packStream) cudaStreamDestroy(packStream);
        for (void *p : {gate, down, gateSf, downSf, sharedGate, sharedDown, sharedGateSf, sharedDownSf, (void *)domains}) if (p) cudaFree(p);
        cudaSetDevice(previous);
    }
};
static bool FastllmDsv41MegaSupported(int device) {
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
        prop.major == 10 && prop.minor == 3 && prop.multiProcessorCount == SMS;
}
static void routedMaps(FastllmDsv41MegaLayer *layer, Maps &m, int sharedCount) {
    const int localExperts = layer->experts / R;
    auto b = buffer(layer->workspace, sharedCount, layer->experts);
    desc2(m.a1, b.l1_token_buffer.base, H, RING, BK, BM / 2, 1, 128);
    desc2(m.s1, b.l1_sf_buffer.base, SFRING, H / 128, 128, 1, 4, 0);
    descWeight(m.w1, layer->gate, I * 2, H, localExperts);
    desc2(m.ws1, layer->gateSf, I * 2, H / 128 * localExperts, 128, 1, 4, 0);
    desc2(m.d1, b.l2_token_buffer.base, I, RING, 64, 8, 1, 64);
    desc2(m.a2, b.l2_token_buffer.base, I, RING, BK, BM / 2, 1, 128);
    desc2(m.s2, b.l2_sf_buffer.base, SFRING, I / 128, 128, 1, 4, 0);
    descWeight(m.w2, layer->down, H, I, localExperts);
    desc2(m.ws2, layer->downSf, H, I / 128 * localExperts, 128, 1, 4, 0);
}

void FastllmDsv41MegaEnableShared(FastllmDsv41MegaLayer *layer, const void *gate,
        const uint8_t *gateScales, const void *down, const uint8_t *downScales) {
    if (!layer || !gate || !down || !gateScales || !downScales || layer->sharedGate)
        throw std::invalid_argument("Invalid DSV41 shared expert weights");
    auto b = buffer(layer->workspace, 1, layer->experts);
    auto &m = layer->shared;
    auto prepareWeight = [&](const void *src, const uint8_t *scales, int n, int k, bool interleave,
                             void *&weight, void *&packed, Desc &w, Desc &sf) {
        weight = allocate(size_t(n)*k);
        packed = allocate(size_t(n)*k/32);
        const size_t scaleBytes = size_t(n/32)*(k/32);
        std::unique_ptr<void, decltype(&cudaFree)> tmp(allocate(scaleBytes), cudaFree);
        check(cudaMemcpy(tmp.get(), scales, scaleBytes, cudaMemcpyHostToDevice));
        packShared<<<(n*k+255)/256, 256>>>(static_cast<const uint8_t *>(src),
            static_cast<const uint8_t *>(tmp.get()), static_cast<uint8_t *>(weight),
            static_cast<uint8_t *>(packed), n, k, interleave);
        check(cudaGetLastError()); check(cudaStreamSynchronize(cudaStreamPerThread));
        cuuint64_t dims[] = {(uint64_t)k, (uint64_t)n/2, 2, 1};
        cuuint64_t strides[] = {(uint64_t)k, (uint64_t)n*k/2, (uint64_t)n*k};
        cuuint32_t box[] = {128,128,1,1};
        descriptor(w, CU_TENSOR_MAP_DATA_TYPE_UINT8, weight, 4, dims, strides, box, 128);
        desc2(sf, packed, n, k/128, 128, 1, 4, 0);
    };
    prepareWeight(gate, gateScales, I*2, H, true, layer->sharedGate, layer->sharedGateSf, m.w1, m.ws1);
    prepareWeight(down, downScales, H, I, false, layer->sharedDown, layer->sharedDownSf, m.w2, m.ws2);
    desc2(m.a1, b.shared_l1_token_buffer.base, H, CAP, BK, BM/2, 1, 128);
    desc2(m.s1, b.shared_l1_sf_buffer.base, sharedSfRows, H/128, 128, 1, 4, 0);
    desc2(m.d1, b.shared_l2_token_buffer.base, I, CAP, 64, 8, 1, 64);
    desc2(m.a2, b.shared_l2_token_buffer.base, I, CAP, BK, BM/2, 1, 128);
    desc2(m.s2, b.shared_l2_sf_buffer.base, sharedSfRows, I/128, 128, 1, 4, 0);
    routedMaps(layer, layer->combinedRouted, 1);
    if (layer->experts == DraftExperts) {
        check(cudaFuncSetAttribute(kernel<1, SMS - 2, true, true, DraftExperts, DraftTopK>(),
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        return;
    }
    check(cudaFuncSetAttribute(kernel<1, SMS - 8>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    check(cudaFuncSetAttribute(kernel<1, SMS - 2>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    const char *pull = std::getenv("FASTLLM_DSV41_REGISTER_PULL");
    if (pull && !std::strcmp(pull, "1")) {
        layer->registerPull = cudaFuncSetAttribute(kernel<1, SMS - 2, true>(),
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        if (!layer->registerPull) cudaGetLastError();
    }
    const char *histogram = std::getenv("FASTLLM_DSV41_SINGLE_HISTOGRAM");
    if (histogram && !std::strcmp(histogram, "1")) {
        auto variant = layer->registerPull ? kernel<1, SMS - 2, true, true>() : kernel<1, SMS - 2, false, true>();
        layer->singleHistogram = cudaFuncSetAttribute(variant, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        if (!layer->singleHistogram) cudaGetLastError();
    }
    check(cudaFuncSetAttribute(kernel<1>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    if (layer->cacheKey) { std::lock_guard<std::mutex> lock(readyMutex); sharedKeys.insert(layer->cacheKey); }
}
bool FastllmDsv41MegaCanPrepare(const int *devices, int count) {
    if (count != R) return false;
    for (int i = 0; i < R; ++i) {
        if (!FastllmDsv41MegaSupported(devices[i])) return false;
        for (int j = 0; j < R; ++j) if (i != j) {
            int accessible = 0;
            if (cudaDeviceCanAccessPeer(&accessible, devices[i], devices[j]) != cudaSuccess ||
                !accessible) return false;
        }
    }
    return true;
}
size_t FastllmDsv41MegaWorkspaceBytes() { return buffer(nullptr, 1).get_num_bytes(); }
void FastllmDsv41MegaInitWorkspace(void *workspace, const int *devices, int rank) {
    int device = -1; check(cudaGetDevice(&device));
    if (!workspace || rank < 0 || rank >= R || devices[rank] != device || !FastllmDsv41MegaSupported(device))
        throw std::runtime_error("DSV41 MegaMoE requires eight 148-SM Blackwell GPUs");
    for (int r = 0; r < R; ++r) if (r != rank) {
        int accessible = 0; check(cudaDeviceCanAccessPeer(&accessible, device, devices[r]));
        if (!accessible) throw std::runtime_error("DSV41 MegaMoE requires all-to-all CUDA peer access");
        auto status = cudaDeviceEnablePeerAccess(devices[r], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(status);
    }
    check(cudaMemset(workspace, 0, FastllmDsv41MegaWorkspaceBytes()));
}
FastllmDsv41MegaLayer *FastllmDsv41MegaCreate(const void *const *gate,
    const void *const *down, void *const *workspaces, int rank, const void *cacheKey, int experts) {
    auto layer = std::make_unique<FastllmDsv41MegaLayer>();
    check(cudaGetDevice(&layer->device)); layer->rank = rank; layer->experts = experts;
    if (experts != E && experts != DraftExperts) throw std::invalid_argument("Unsupported DSV41 expert count");
    const int localExperts = experts / R;
    if (rank < 0 || rank >= R || !FastllmDsv41MegaSupported(layer->device))
        throw std::runtime_error("Invalid DSV41 MegaMoE rank/device");
    layer->workspace = workspaces[rank];
    const char *order = std::getenv("FASTLLM_DSV41_ORDER_WORD");
    if (order && !std::strcmp(order, "1") && cudaMalloc(&layer->routeOrder, CAP * sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        layer->routeOrder = nullptr;
    }
    const char *early = std::getenv("FASTLLM_DSV41_EARLY_MOE_PACK");
    if (early && !std::strcmp(early, "1")) {
        layer->canPackEarly = cudaStreamCreateWithFlags(&layer->packStream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaEventCreateWithFlags(&layer->inputReady, cudaEventDisableTiming) == cudaSuccess &&
            cudaEventCreateWithFlags(&layer->packetReady, cudaEventDisableTiming) == cudaSuccess;
        if (!layer->canPackEarly) cudaGetLastError();
    }
    for (int r = 0; r < R; ++r) layer->peers[r] = reinterpret_cast<int64_t>(workspaces[r]);
    layer->gate = allocate((size_t)localExperts * I * H);
    layer->down = allocate((size_t)localExperts * H * I / 2);
    layer->gateSf = allocate((size_t)localExperts * I * 2 * H / 32);
    layer->downSf = allocate((size_t)localExperts * H * I / 32);
    // Use upstream's even TPC mapping when localized allocation is unavailable.
    // Hardware SM IDs may be sparse, so cover the architecture's full ID range.
    layer->domains = static_cast<uint8_t *>(allocate(2048));
    std::array<uint8_t, 2048> domains{};
    for (size_t i = 0; i < domains.size(); ++i) domains[i] = (i / 2) % 2;
    check(cudaMemcpy(layer->domains, domains.data(), domains.size(), cudaMemcpyHostToDevice));
    for (int e = 0; e < localExperts; ++e) {
        if (!gate[e] || !down[e]) throw std::runtime_error("Missing DSV41 MegaMoE expert");
        pack<<<(I * H + 255) / 256, 256>>>(static_cast<const uint8_t *>(gate[e]),
            static_cast<uint8_t *>(layer->gate) + (size_t)e * I * H,
            static_cast<uint8_t *>(layer->gateSf) + (size_t)e * I * 2 * H / 32, I * 2, H, true);
        pack<<<(H * I / 2 + 255) / 256, 256>>>(static_cast<const uint8_t *>(down[e]),
            static_cast<uint8_t *>(layer->down) + (size_t)e * H * I / 2,
            static_cast<uint8_t *>(layer->downSf) + (size_t)e * H * I / 32, H, I, false);
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    if (experts == E) {
        routedMaps(layer.get(), layer->routed, 0);
        check(cudaFuncSetAttribute(kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    }
    layer->cacheKey = cacheKey;
    if (cacheKey) { std::lock_guard<std::mutex> lock(readyMutex); readyKeys.insert(cacheKey); }
    return layer.release();
}
bool FastllmDsv41MegaGraphReady(const void *cacheKey, int rows) {
    if (!cacheKey || rows < 1 || rows > CAP * R) return false;
    std::lock_guard<std::mutex> lock(readyMutex);
    return (rows >= 16 && readyKeys.count(cacheKey)) ||
        ((rows == 1 || rows % R == 0) && sharedKeys.count(cacheKey));
}
void FastllmDsv41MegaDestroy(FastllmDsv41MegaLayer *layer) { delete layer; }
FastllmDsv41InputPacket FastllmDsv41OwnerInputPacket(FastllmDsv41MegaLayer *layer, int rows) {
    const char *enable = std::getenv("FASTLLM_DSV41_POST_PACKET");
    if (!layer || !layer->sharedGate || rows < 1 || rows > CAP || !enable || std::strcmp(enable, "1")) return {};
    auto b = buffer(layer->workspace, 1);
    auto packet = FastllmDsv41InputPacket{b.input_token_buffer.get_base_ptr<uint8_t>(),
        b.input_sf_buffer.get_base_ptr<uint8_t>(), b.shared_l1_sf_buffer.get_base_ptr<uint8_t>(), sharedSfRows};
    const char *direct = std::getenv("FASTLLM_DSV41_DIRECT_ROUTES");
    if (direct && !std::strcmp(direct, "1")) {
        packet.routeIds = b.input_topk_idx_buffer.get_base_ptr<int64_t>();
        packet.routeScores = b.input_topk_weights_buffer.get_base_ptr<float>();
        packet.routeOrder = layer->routeOrder;
    }
    return packet;
}
// Small owners publish during mHC; for GEMM-sized owners this branch overlaps router projection.
void *FastllmDsv41PackLocalInput(FastllmDsv41MegaLayer *layer, const void *input, int rows) {
    if (!layer || !layer->canPackEarly || !layer->sharedGate || !input || rows < 16 || rows > CAP) return nullptr;
    auto b = buffer(layer->workspace, 1);
    check(cudaEventRecord(layer->inputReady, cudaStreamPerThread));
    check(cudaStreamWaitEvent(layer->packStream, layer->inputReady));
    prepare<true><<<rows * (H / 32), 32, 0, layer->packStream>>>(
        static_cast<const __nv_bfloat16 *>(input), nullptr, nullptr, b.input_token_buffer.get_base_ptr<uint8_t>(),
        b.input_sf_buffer.get_base_ptr<uint8_t>(), nullptr, nullptr, layer->rank, rows, true,
        b.shared_l1_sf_buffer.get_base_ptr<uint8_t>(), true);
    check(cudaEventRecord(layer->packetReady, layer->packStream));
    return layer->packetReady;
}
template <bool Shared, bool Local = false, bool Post = false, bool Draft = false> static void run(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, void *output, int rows, void *rawStream, bool ownerOutput,
    const void *residual = nullptr, const float *post = nullptr, const float *comb = nullptr, cudaEvent_t mixDone = nullptr, cudaEvent_t packetReady = nullptr, bool packed = false, bool routesReady = false) {
    if (!layer || rows < (Local ? 0 : 1) || rows > (Local ? CAP : CAP * R) ||
        (!Shared && ownerOutput && rows % R) || (rows && (!input || !ids || !scores || !output)))
        throw std::runtime_error("Invalid DSV41 MegaMoE decode input");
    auto stream = rawStream ? static_cast<cudaStream_t>(rawStream) : cudaStreamPerThread;
    constexpr int experts = Draft ? DraftExperts : E, topk = Draft ? DraftTopK : K;
    dg::layout::MegaMoEBuffer b = buffer(layer->workspace, int(Shared), experts);
    const int localRows = Local ? rows : (rows + R - 1 - layer->rank) / R;
    if (packetReady || packed) {
        if (!routesReady) publishLocalRoutes<<<localRows, 32, 0, stream>>>(ids, scores,
            b.input_topk_idx_buffer.get_base_ptr<int64_t>(), b.input_topk_weights_buffer.get_base_ptr<float>());
        if (packetReady) check(cudaStreamWaitEvent(stream, packetReady));
    } else if (localRows) prepare<false, experts, topk><<<localRows * (H / 32), 32, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(input), ids, scores,
        b.input_token_buffer.get_base_ptr<uint8_t>(), b.input_sf_buffer.get_base_ptr<uint8_t>(),
        b.input_topk_idx_buffer.get_base_ptr<int64_t>(), b.input_topk_weights_buffer.get_base_ptr<float>(), layer->rank, rows, ownerOutput, Shared ? b.shared_l1_sf_buffer.get_base_ptr<uint8_t>() : nullptr, Local);
    cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    int activeSms = SMS;
    decltype(kernel<int(Shared)>()) megaKernel;
    if constexpr (Draft) {
        activeSms = SMS - 2;
        megaKernel = kernel<1, SMS - 2, true, true, DraftExperts, DraftTopK>();
    } else {
        megaKernel = kernel<int(Shared)>();
    }
    if constexpr (Local && Post && !Draft) {
        static const int reserveClusters = [] {
            const char *value = std::getenv("FASTLLM_DSV41_MOE_RESERVE_CLUSTER");
            if (value && std::strcmp(value, "4") == 0) return 4;
            return value && std::strcmp(value, "1") == 0 ? 1 : 0;
        }();
        if (reserveClusters) {
            // The persistent kernel's software barriers count CTAs. Its template
            // and launch grid must agree on every rank, including empty owners.
            activeSms = SMS - 2 * reserveClusters;
            megaKernel = reserveClusters == 4 ? kernel<1, SMS - 8>() :
                (layer->registerPull ? kernel<1, SMS - 2, true>() : kernel<1, SMS - 2>());
            if (reserveClusters == 1 && layer->singleHistogram)
                megaKernel = layer->registerPull ? kernel<1, SMS - 2, true, true>() : kernel<1, SMS - 2, false, true>();
        }
    }
    cudaLaunchConfig_t config{}; config.gridDim = dim3(activeSms); config.blockDim = dim3(512);
    config.dynamicSmemBytes = SMEM; config.stream = stream; config.attrs = &attr; config.numAttrs = 1;
    const auto &r = Shared ? layer->combinedRouted : layer->routed;
    const auto &sh = Shared ? layer->shared : layer->routed;
    check(cudaLaunchKernelEx(&config, megaKernel, (void *)nullptr, (int *)nullptr,
        (uint32_t)localRows, dg::layout::SymBuffer<R>(layer->peers, layer->rank),
        r.a1, r.s1, r.w1, r.ws1, r.d1,
        r.a2, r.s2, r.w2, r.ws2,
        sh.a1, sh.s1, sh.w1, sh.ws1, sh.d1,
        sh.a2, sh.s2, sh.w2, sh.ws2, (const uint8_t *)layer->domains, dg::PackedTokenInput{}));
    if (ownerOutput) {
        if (!localRows) return;
        if constexpr (Local && Post) {
            if (mixDone) check(cudaStreamWaitEvent(stream, mixDone));
        }
        const int start = Local ? 0 : layer->rank * (rows/R) + std::min(layer->rank, rows%R);
        combineOwnedRows<Shared, Post, topk><<<(localRows * H + 255) / 256, 256, 0, stream>>>(
            b.combine_token_buffer.get_base_ptr<__nv_bfloat16>(), ids, output, localRows, start,
            static_cast<const __nv_bfloat16 *>(residual), post, comb, routesReady ? layer->routeOrder : nullptr);
    } else {
        scatter<<<(rows * H + 255) / 256, 256, 0, stream>>>(
            dg::layout::SymBuffer<R>(layer->peers, layer->rank), ids, static_cast<float *>(output), rows, layer->rank);
    }
    check(cudaGetLastError());
}

void FastllmDsv41MegaRun(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, float *output, int rows, void *stream, bool ownerOutput) {
    run<false>(layer, input, ids, scores, output, rows, stream, ownerOutput);
}
void FastllmDsv41MegaRunShared(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, void *local, int rows, void *stream) {
    if (!layer || !layer->sharedGate) throw std::invalid_argument("DSV41 shared expert is unprepared");
    run<true>(layer, input, ids, scores, local, rows, stream, true);
}

void FastllmDsv41MegaRunLocal(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, void *output, int rows, void *stream) {
    if (!layer || !layer->sharedGate) throw std::invalid_argument("DSV41 local expert is unprepared");
    run<true, true>(layer, input, ids, scores, output, rows, stream, true);
}

void FastllmDsv41MegaRunLocalPacket(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int rows, void *stream, void *mixDone, void *packetReady, bool packed, bool routesReady) {
    if (!layer || !layer->sharedGate || (rows && (!residual || !post || !comb)))
        throw std::invalid_argument("Invalid DSV41 local expert residual");
    run<true, true, true>(layer, input, ids, scores, hidden, rows, stream, true, residual, post, comb, static_cast<cudaEvent_t>(mixDone), static_cast<cudaEvent_t>(packetReady), packed, routesReady);
}
void FastllmDsv41MegaDraftPost(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int rows) {
    run<true, true, true, true>(layer, input, ids, scores, hidden, rows, nullptr, true, residual, post, comb);
}
void FastllmDsv41MegaRunLocalPostAfterMix(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int rows, void *stream, void *mixDone) {
    FastllmDsv41MegaRunLocalPacket(layer,input,ids,scores,residual,post,comb,hidden,rows,stream,mixDone,nullptr);
}
void FastllmDsv41MegaRunLocalPost(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int rows, void *stream) {
    FastllmDsv41MegaRunLocalPostAfterMix(layer, input, ids, scores, residual, post, comb, hidden, rows, stream, nullptr);
}

const void *FastllmDsv41OwnerPostKernel() {
    return reinterpret_cast<const void *>(combineOwnedRows<true, true>);
}

const void *FastllmDsv41PrepareKernel() {
    return reinterpret_cast<const void *>(prepare<false>);
}
