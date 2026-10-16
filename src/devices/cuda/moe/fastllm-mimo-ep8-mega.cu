// H20 EP8 MegaMoE: 32 complete experts per rank; caller owns persistent symmetric storage.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "devices/cuda/fastllm-mimo-ep8-mega-kernel.h"
#include <deep_gemm/impls/sm90_fp8_mega_moe_fastllm.cuh>
#include <deep_gemm/impls/sm90_mega_moe_pre_dispatch.cuh>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
// NVCC 13 emits an unqualified inf literal in host stubs for float NTTPs.
static constexpr float inf = __builtin_inff();
namespace {
using namespace fastllm_ep8_deep_gemm;
constexpr int H = 4096, I = 2048, E = 256, R = 8, LOCAL_E = E / R, K = 8, CAP = 64;
constexpr int BM = 64, BN = 128, BK = 128, STAGES = 8, SMS = 78;
constexpr int PREFILL_CAP = 1024, PREFILL_BN = 256, PREFILL_STAGES = 4;
int PrefillCapacity(int inputRows) {
    if (inputRows <= R * CAP) return 0;
    // Every rank must choose the same layout, including uneven shards. Use
    // global rows, not local allocation sizes, which can differ across ranks.
    const int rows = inputRows / R + (inputRows % R != 0);
    int capacity = CAP * 2;
    while (capacity < rows && capacity < PREFILL_CAP) capacity *= 2;
    return capacity;
}
// All local experts share a compact token pool. Across ranks there are at most
// R * CAP * K routes in total; each expert adds at most BM-1 alignment slots.
// The sum of aligned expert spans is a multiple of BM, so rounding this bound
// down is safe. Do not reserve R * CAP rows separately for every local expert.
constexpr int PoolTokens(int capacity) {
    return (R * capacity * K + LOCAL_E * (BM - 1)) / BM * BM;
}
static_assert(PoolTokens(CAP) == 6080, "Unexpected EP8 pool capacity");
thread_local std::string error;
void check(cudaError_t c) {
    if (c != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(c));
}
struct alignas(128) Metadata {
    cute::TmaDescriptor gate[LOCAL_E], down[LOCAL_E];
    const float *gateScales[LOCAL_E], *downScales[LOCAL_E];
};
struct Scratch {
    void *x, *xs, *idx, *scores, *l1, *l1s, *l2, *l2s;
    Metadata *metadata;
    cute::TmaDescriptor a1, sa1, d1, a2, sa2;
    size_t slice(void *base, int capacity = CAP) {
        const int pool = PoolTokens(capacity), sfpool = pool / BM * 128;
        layout::SM90Workspace workspace(base, R, E, capacity, K);
        auto add = [](void *p, int bytes, int count, bool align = true) {
            return layout::Buffer(layout::Data(bytes, align), 1, count, p);
        };
        auto bx = add(workspace.get_end_ptr(), H, capacity);
        auto bs = add(bx.get_end_ptr(), H / 32, capacity);
        auto bi = add(bs.get_end_ptr(), K * 8, capacity, false);
        auto bw = add(bi.get_end_ptr(), K * 4, capacity, false);
        auto bl1 = add(bw.get_end_ptr(), H, pool);
        auto bl1s = add(bl1.get_end_ptr(), H / 32, sfpool);
        auto bl1w = add(bl1s.get_end_ptr(), 4, pool, false);
        auto bl2 = add(bl1w.get_end_ptr(), I, pool);
        auto bl2s = add(bl2.get_end_ptr(), I / 16, sfpool);
        auto combine = layout::Buffer(layout::Data(H * 2), K, capacity, bl2s.get_end_ptr());
        x = bx.base;
        xs = bs.base;
        idx = bi.base;
        scores = bw.base;
        l1 = bl1.base;
        l1s = bl1s.base;
        l2 = bl2.base;
        l2s = bl2s.base;
        size_t offset =
            reinterpret_cast<uintptr_t>(combine.get_end_ptr()) - reinterpret_cast<uintptr_t>(base);
        auto reserve = [&](size_t bytes) {
            offset = (offset + 255) & ~size_t(255);
            void *p = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) + offset);
            offset += bytes;
            return p;
        };
        // Descriptors persist across layers and Graph replay; eliminate a
        // host-to-device transfer on every MoE call without copying weights.
        // Borrowed prefill scratch is reused by successive layers, so only
        // the current layer's descriptors are needed there.
        metadata = static_cast<Metadata *>(reserve((capacity == CAP ? 48 : 1) * sizeof(Metadata)));
        return (offset + 255) & ~size_t(255);
    }
};
void descriptor(cute::TmaDescriptor &desc, CUtensorMapDataType type, void *ptr, uint64_t inner,
                uint64_t outer, uint64_t stride, uint32_t bi, uint32_t bo, bool swizzle) {
    const cuuint64_t dims[] = {inner, outer}, strides[] = {stride};
    const cuuint32_t box[] = {bi, bo}, element[] = {1, 1};
    auto status = cuTensorMapEncodeTiled(
        reinterpret_cast<CUtensorMap *>(&desc), type, 2, ptr, dims, strides, box, element,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("MegaMoE TMA descriptor error " + std::to_string(status));
}
void activationDescriptors(Scratch &s, int capacity) {
    const int pool = PoolTokens(capacity), sfpool = pool / BM * 128;
    descriptor(s.a1, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l1, H, pool, H, BK, BM, true);
    descriptor(s.sa1, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, s.l1s, sfpool, H / 128, sfpool * 4, BM, 1, false);
    // Each prefill warpgroup stores 64 SwiGLU columns; decode's two
    // narrower warpgroups jointly store the same 64-column tile.
    descriptor(s.d1, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l2, I, pool, I, 64, BM, false);
    descriptor(s.a2, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l2, I, pool, I, BK, BM, true);
    descriptor(s.sa2, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, s.l2s, sfpool, I / 64, sfpool * 4, BM, 1, false);
}
template <int capacity = CAP>
auto kernel() {
    constexpr bool Prefill = capacity > CAP;
    constexpr int pool = PoolTokens(capacity), sfpool = pool / BM * 128;
    return &sm90_fp8_mega_moe_impl<capacity, H, I, E, K, LOCAL_E, BM,
                                   Prefill ? PREFILL_BN : BN, BK, pool, sfpool,
                                   Prefill ? PREFILL_STAGES : STAGES, 64, 64, 256, SMS, R,
                                   cute::numeric_limits<float>::infinity(), true, 0, false, false, true,
                                   false, !Prefill>;
}
constexpr int SMEM =
    1024 + 8192 + 20480 + STAGES * (BM * BK + BN * BK + 2 * BM * 4) + (2 + 2 * STAGES + 16) * 8;
constexpr int PREFILL_SMEM = 1024 + 8192 + BM * PREFILL_BN * 2 +
    PREFILL_STAGES * (BM * BK + PREFILL_BN * BK + 2 * BM * 4) + (2 + 2 * PREFILL_STAGES + 16) * 8;
struct Layer {
    int device, rank;
    std::array<int64_t, R> peers;
    Scratch scratch;
    Scratch prefill;
    void *prefillBase = nullptr;
    int prefillCapacity = 0;
    Metadata prefillMetadata;
};
} // namespace
extern "C" const char *fastllm_ep8_mega_moe_error() {
    return error.c_str();
}
extern "C" int fastllm_ep8_mega_moe_max_rows() {
    return CAP;
}
extern "C" size_t fastllm_ep8_mega_moe_workspace_bytes() {
    Scratch s{};
    return s.slice(nullptr);
}
extern "C" size_t fastllm_ep8_mega_moe_prefill_workspace_bytes(int inputRows) {
    Scratch s{};
    const int capacity = PrefillCapacity(inputRows);
    return capacity ? s.slice(nullptr, capacity) : 0;
}
extern "C" int fastllm_ep8_mega_moe_init_workspace(void *workspace, size_t bytes,
                                                   const int *devices, int rank) {
    try {
        int device;
        check(cudaGetDevice(&device));
        if (!workspace || bytes < fastllm_ep8_mega_moe_workspace_bytes() ||
            rank < 0 || rank >= R || devices[rank] != device)
            throw std::runtime_error("Invalid EP8 workspace initialization");
        for (int r = 0; r < R; ++r) {
            if (r == rank) continue;
            int accessible = 0;
            check(cudaDeviceCanAccessPeer(&accessible, device, devices[r]));
            if (!accessible) throw std::runtime_error("EP8 requires all-to-all peer access");
            auto status = cudaDeviceEnablePeerAccess(devices[r], 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else check(status);
        }
        check(cudaMemset(workspace, 0, bytes));
        check(cudaDeviceSynchronize());
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" int fastllm_ep8_mega_moe_supported() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 0;
    static thread_local std::map<int, bool> supported;
    auto it = supported.find(device);
    if (it != supported.end())
        return it->second;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return 0;
    bool ok = prop.major == 9 && prop.minor == 0 && prop.multiProcessorCount == SMS &&
              std::strstr(prop.name, "H20") != nullptr;
    supported[device] = ok;
    return ok;
}
extern "C" void *fastllm_ep8_mega_moe_create(const void *const *gate, const void *const *down,
                                         const void *const *gs, const void *const *ds, void *workspace,
                                         size_t workspaceBytes, const void *const *peers, int rank, int slot) {
    try {
        if (!fastllm_ep8_mega_moe_supported())
            throw std::runtime_error("MegaMoE requires H20 SM90 with 78 SMs");
        if (!workspace || reinterpret_cast<uintptr_t>(workspace) % 256 ||
            workspaceBytes < fastllm_ep8_mega_moe_workspace_bytes())
            throw std::runtime_error(
                "MegaMoE requires an existing aligned workspace; no allocation is allowed");
        auto layer = std::make_unique<Layer>();
        check(cudaGetDevice(&layer->device));
        if (rank < 0 || rank >= R || peers[rank] != workspace || slot < 0 || slot >= 48)
            throw std::runtime_error("Invalid EP8 rank or peer workspace");
        layer->rank = rank;
        for (int r = 0; r < R; ++r) layer->peers[r] = reinterpret_cast<int64_t>(peers[r]);
        auto &s = layer->scratch;
        s.slice(workspace);
        s.metadata += slot;
        // Decode metadata lives in persistent device storage; prefill keeps
        // an unpinned host copy to initialize the borrowed arena per layer.
        Metadata metadata{};
        for (int e = 0; e < LOCAL_E; ++e) {
            descriptor(metadata.gate[e], CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       const_cast<void *>(gate[e]), H, 2 * I, H, BK, 8, true);
            descriptor(metadata.down[e], CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       const_cast<void *>(down[e]), I, H, I, BK, BN, true);
            metadata.gateScales[e] = static_cast<const float *>(gs[e]);
            metadata.downScales[e] = static_cast<const float *>(ds[e]);
        }
        activationDescriptors(s, CAP);
        check(cudaMemcpy(s.metadata, &metadata, sizeof(Metadata), cudaMemcpyHostToDevice));
        check(cudaFuncSetAttribute(kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        layer->prefillMetadata = metadata;
        for (int e = 0; e < LOCAL_E; ++e)
            descriptor(layer->prefillMetadata.down[e], CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       const_cast<void *>(down[e]), I, H, I, BK, PREFILL_BN, true);
        check(cudaFuncSetAttribute(kernel<128>(), cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM));
        check(cudaFuncSetAttribute(kernel<256>(), cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM));
        check(cudaFuncSetAttribute(kernel<512>(), cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM));
        check(cudaFuncSetAttribute(kernel<PREFILL_CAP>(), cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM));
        return layer.release();
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
extern "C" int fastllm_ep8_mega_moe_prepare_prefill(void *handle, void *workspace, size_t bytes,
                                                   int inputRows) {
    try {
        const int capacity = PrefillCapacity(inputRows);
        Scratch required{};
        if (!capacity || !workspace || reinterpret_cast<uintptr_t>(workspace) % 256 ||
            bytes < required.slice(nullptr, capacity)) return 0;
        if (!handle) throw std::runtime_error("Missing EP8 prefill handle");
        auto &l = *static_cast<Layer *>(handle);
        int device;
        check(cudaGetDevice(&device));
        if (device != l.device) throw std::runtime_error("MegaMoE prefill device mismatch");
        if (l.prefillBase != workspace || l.prefillCapacity != capacity) {
            l.prefill.slice(workspace, capacity);
            activationDescriptors(l.prefill, capacity);
            l.prefillBase = workspace;
            l.prefillCapacity = capacity;
        }
        // The caller lends dead router activations. Reset only control data;
        // dispatch/combine overwrite all live token data. The caller records
        // an event after these writes and makes every peer stream wait for it.
        const size_t controls = layout::SM90Workspace(workspace, R, E, capacity, K).get_num_bytes();
        check(cudaMemsetAsync(workspace, 0, controls, cudaStreamPerThread));
        check(cudaMemcpyAsync(l.prefill.metadata, &l.prefillMetadata, sizeof(Metadata),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
        return capacity;
    } catch (const std::exception &e) { error = e.what(); return -1; }
}
template <int capacity>
static int Run(void *handle, const void *x, void *y, const int32_t *idx,
               const float *scores, int rows, const void *const *peers = nullptr, const int *liveKeys = nullptr) {
    try {
        constexpr bool Prefill = capacity > CAP;
        if (!handle || !x || !y || !idx || !scores || rows < 0 || rows > capacity)
            throw std::runtime_error("MegaMoE requires valid pointers and 0..capacity rows");
        auto &l = *static_cast<Layer *>(handle);
        auto &s = Prefill ? l.prefill : l.scratch;
        int device;
        check(cudaGetDevice(&device));
        if (device != l.device)
            throw std::runtime_error("MegaMoE device mismatch");
        // Persistent symmetric workspace is zeroed once before any rank runs.
        // MegaMoE cleans its control state collectively; never memset between ranks.
        // Zero-token ranks still dispatch a fully masked table and participate
        // in expert computation/barriers. Their combine writes no output rows.
        std::array<int64_t, R> addresses = l.peers;
        if constexpr (Prefill) {
            if (!peers || !l.prefillBase || l.prefillCapacity != capacity ||
                peers[l.rank] != l.prefillBase)
                throw std::runtime_error("Unprepared EP8 prefill arena");
            for (int r = 0; r < R; ++r) {
                if (!peers[r]) throw std::runtime_error("Missing EP8 prefill peer");
                addresses[r] = reinterpret_cast<int64_t>(peers[r]);
            }
        }
        int padBlocks = ((capacity - rows) * K + (H / 8) - 1) / (H / 8);
        sm90_mega_moe_pre_dispatch_kernel<128, false><<<rows + padBlocks, H / 8, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(x), idx, scores,
            static_cast<__nv_fp8_e4m3 *>(s.x), static_cast<float *>(s.xs), static_cast<int64_t *>(s.idx),
            static_cast<float *>(s.scores), rows, capacity, H, H / 128, K, 1.0f, liveKeys);
        auto sym = layout::SymBuffer<R>(addresses, l.rank);
        kernel<capacity>()<<<SMS, 384, Prefill ? PREFILL_SMEM : SMEM, cudaStreamPerThread>>>(y, nullptr, rows, sym, s.a1, s.sa1, s.metadata->gate,
            s.metadata->gateScales, s.d1, s.a2, s.sa2, s.metadata->down, s.metadata->downScales);
        check(cudaGetLastError());
        return 1;
    } catch (const std::exception &e) {
        error = e.what();
        return 0;
    }
}
extern "C" int fastllm_ep8_mega_moe_run(void *handle, const void *x, void *y, const int32_t *idx,
                                      const float *scores, int rows) {
    return Run<CAP>(handle, x, y, idx, scores, rows);
}
extern "C" int fastllm_ep8_mega_moe_run_masked(void *handle, const void *x, void *y, const int32_t *idx,
        const float *scores, int rows, const int *liveKeys) {
    return Run<CAP>(handle, x, y, idx, scores, rows, nullptr, liveKeys);
}
extern "C" int fastllm_ep8_mega_moe_run_prefill(void *handle, const void *const *peers,
        const void *x, void *y, const int32_t *idx, const float *scores, int rows) {
    if (handle) {
        switch (static_cast<Layer *>(handle)->prefillCapacity) {
            case 128: return Run<128>(handle, x, y, idx, scores, rows, peers);
            case 256: return Run<256>(handle, x, y, idx, scores, rows, peers);
            case 512: return Run<512>(handle, x, y, idx, scores, rows, peers);
            case PREFILL_CAP: return Run<PREFILL_CAP>(handle, x, y, idx, scores, rows, peers);
        }
    }
    error = "Unprepared EP8 prefill arena";
    return 0;
}
extern "C" void fastllm_ep8_mega_moe_destroy(void *handle) {
    delete static_cast<Layer *>(handle);
}
