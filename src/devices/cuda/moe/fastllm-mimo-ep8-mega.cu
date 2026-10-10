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
// All local experts share a compact token pool. Across ranks there are at most
// R * CAP * K routes in total; each expert adds at most BM-1 alignment slots.
// The sum of aligned expert spans is a multiple of BM, so rounding this bound
// down is safe. Do not reserve R * CAP rows separately for every local expert.
constexpr int POOL = (R * CAP * K + LOCAL_E * (BM - 1)) / BM * BM;
constexpr int SFPOOL = (POOL / BM) * 128;
static_assert(POOL == 6080 && SFPOOL == 12160, "Unexpected EP8 pool capacity");
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
    size_t slice(void *base) {
        layout::SM90Workspace workspace(base, R, E, CAP, K);
        auto add = [](void *p, int bytes, int count, bool align = true) {
            return layout::Buffer(layout::Data(bytes, align), 1, count, p);
        };
        auto bx = add(workspace.get_end_ptr(), H, CAP);
        auto bs = add(bx.get_end_ptr(), H / 32, CAP);
        auto bi = add(bs.get_end_ptr(), K * 8, CAP, false);
        auto bw = add(bi.get_end_ptr(), K * 4, CAP, false);
        auto bl1 = add(bw.get_end_ptr(), H, POOL);
        auto bl1s = add(bl1.get_end_ptr(), H / 32, SFPOOL);
        auto bl1w = add(bl1s.get_end_ptr(), 4, POOL, false);
        auto bl2 = add(bl1w.get_end_ptr(), I, POOL);
        auto bl2s = add(bl2.get_end_ptr(), I / 16, SFPOOL);
        auto combine = layout::Buffer(layout::Data(H * 2), K, CAP, bl2s.get_end_ptr());
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
        metadata = static_cast<Metadata *>(reserve(48 * sizeof(Metadata)));
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
auto kernel() {
    return &sm90_fp8_mega_moe_impl<CAP, H, I, E, K, LOCAL_E, BM, BN, BK, POOL, SFPOOL, STAGES, 64, 64, 256, SMS, R,
                                   cute::numeric_limits<float>::infinity(), true, 0, false, false, true,
                                   false, true>;
}
constexpr int SMEM =
    1024 + 8192 + 20480 + STAGES * (BM * BK + BN * BK + 2 * BM * 4) + (2 + 2 * STAGES + 16) * 8;
struct Layer {
    int device, rank;
    std::array<int64_t, R> peers;
    Scratch scratch;
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
        // cudaMemcpy completes the host read before returning. Keep metadata
        // only on the stack during initialization, not pinned for every layer.
        Metadata metadata{};
        for (int e = 0; e < LOCAL_E; ++e) {
            descriptor(metadata.gate[e], CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       const_cast<void *>(gate[e]), H, 2 * I, H, BK, 8, true);
            descriptor(metadata.down[e], CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       const_cast<void *>(down[e]), I, H, I, BK, BN, true);
            metadata.gateScales[e] = static_cast<const float *>(gs[e]);
            metadata.downScales[e] = static_cast<const float *>(ds[e]);
        }
        descriptor(s.a1, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l1, H, POOL, H, BK, BM, true);
        descriptor(s.sa1, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, s.l1s, SFPOOL, H / 128, SFPOOL * 4, BM, 1, false);
        descriptor(s.d1, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l2, I, POOL, I, BN / 2, BM, false);
        descriptor(s.a2, CU_TENSOR_MAP_DATA_TYPE_UINT8, s.l2, I, POOL, I, BK, BM, true);
        descriptor(s.sa2, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, s.l2s, SFPOOL, I / 64, SFPOOL * 4, BM, 1, false);
        check(cudaMemcpy(s.metadata, &metadata, sizeof(Metadata), cudaMemcpyHostToDevice));
        check(cudaFuncSetAttribute(kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        return layer.release();
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
extern "C" int fastllm_ep8_mega_moe_run(void *handle, const void *x, void *y, const int32_t *idx,
                                    const float *scores, int rows) {
    try {
        if (!handle || rows < 1 || rows > CAP)
            throw std::runtime_error("MegaMoE row count exceeds its compiled capacity");
        auto &l = *static_cast<Layer *>(handle);
        auto &s = l.scratch;
        int device;
        check(cudaGetDevice(&device));
        if (device != l.device)
            throw std::runtime_error("MegaMoE device mismatch");
        // Persistent symmetric workspace is zeroed once before any rank runs.
        // MegaMoE cleans its control state collectively; never memset between ranks.
        int padBlocks = ((CAP - rows) * K + (H / 8) - 1) / (H / 8);
        sm90_mega_moe_pre_dispatch_kernel<128, false><<<rows + padBlocks, H / 8, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(x), idx, scores,
            static_cast<__nv_fp8_e4m3 *>(s.x), static_cast<float *>(s.xs), static_cast<int64_t *>(s.idx),
            static_cast<float *>(s.scores), rows, CAP, H, H / 128, K, 1.0f);
        auto sym = layout::SymBuffer<R>(l.peers, l.rank);
        kernel()<<<SMS, 384, SMEM, cudaStreamPerThread>>>(y, nullptr, rows, sym, s.a1, s.sa1, s.metadata->gate,
            s.metadata->gateScales, s.d1, s.a2, s.sa2, s.metadata->down, s.metadata->downScales);
        check(cudaGetLastError());
        return 1;
    } catch (const std::exception &e) {
        error = e.what();
        return 0;
    }
}
extern "C" void fastllm_ep8_mega_moe_destroy(void *handle) {
    delete static_cast<Layer *>(handle);
}
