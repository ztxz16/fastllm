// Native adapter for FlashInfer v0.7.4 SM90 MXFP4 MegaMoE (Apache-2.0).
// See third_party/flashinfer_mxfp4. No Torch, TVM-FFI or Python runtime.
#include "devices/cuda/fastllm-mxfp4-ep.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#define FI_MEGAMOE_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <flashinfer/megamoe/sm90_mxfp4_mega_moe_fused.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// NVCC emits this spelling in host stubs for floating-point template arguments.
static constexpr float inf = __builtin_inff();
namespace {
namespace fm = flashinfer::megamoe;
constexpr int H = 4096, I = 2048, E = 256, K = 8, DECODE_CAP = 128, PREFILL_CAP = 2048, SMS = 78;
constexpr int SMEM = 232448;
thread_local std::string error;
void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Allocation {
    void *p = nullptr;
    ~Allocation() { if (p) cudaFree(p); }
    void allocate(size_t bytes) { check(cudaMalloc(&p, bytes)); }
};
void descriptor(cute::TmaDescriptor &d, void *p, int width, int height,
        int boxWidth, int boxHeight, bool fp32 = false, bool swizzle = false) {
    cuuint64_t dims[] = {(uint64_t)width, (uint64_t)height};
    cuuint64_t strides[] = {(uint64_t)width * (fp32 ? 4 : 1)};
    cuuint32_t box[] = {(uint32_t)boxWidth, (uint32_t)boxHeight}, step[] = {1, 1};
    CUresult s = cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&d),
        fp32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, p, dims, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (s != CUDA_SUCCESS) throw std::runtime_error("MXFP4 EP TMA descriptor: " + std::to_string(s));
}

// Convert canonical adjacent E2M1 nibbles directly to the upstream braided,
// chunk-major tiles. The power-of-two scale shift is exactly compensated by
// an expert-global multiplier, applied BEFORE SwiGLU in the upstream kernel.
__global__ void pack(const uint8_t *src, uint8_t *dst, int n, int k, int shift, bool gateup) {
    constexpr int tileBytes = 128 * 68;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = (size_t)n * k / 2 + (size_t)n * k / 32;
    if (i >= size) return;
    size_t tile = i / tileBytes;
    int byte = i % tileBytes, kb = tile % (k / 128), nr = tile / (k / 128) * 128;
    if (byte < 8192) {
        int row = nr + (byte / 16) % 128;
        int col = kb * 128 + (byte / 2048) * 32 + (byte % 16) * 2;
        if (gateup) row = row / 16 * 8 + row % 8 + (row % 16 >= 8 ? n / 2 : 0);
        const uint8_t *w = src + (size_t)row * k / 2;
        int group = col / 8 * 8, pair = col % 8 / 2;
        constexpr int signSource[8] = {4, 0, 5, 1, 6, 2, 7, 3};
        unsigned a = w[col / 2] & 0x77;
        int s0 = group + signSource[2 * pair], s1 = group + signSource[2 * pair + 1];
        a |= ((w[s0 / 2] >> (s0 % 2 * 4)) & 8);
        a |= ((w[s1 / 2] >> (s1 % 2 * 4)) & 8) << 4;
        dst[i] = a;
    } else {
        int row = nr + (byte - 8192) / 4, col = kb * 4 + (byte - 8192) % 4;
        if (gateup) row = row / 16 * 8 + row % 8 + (row % 16 >= 8 ? n / 2 : 0);
        int value = src[(size_t)n * k / 2 + (size_t)row * (k / 32) + col];
        dst[i] = value + shift;
    }
}

__global__ void stage(const __nv_bfloat16 *x, const int32_t *idx, const float *scores,
        __nv_fp8_e4m3 *out, float *sf, int64_t *routes, float *weights,
        int rows, const int *live) {
    int row = blockIdx.x, lane = threadIdx.x;
    bool valid = row < rows && (!live || live[row] > 0);
    if (lane < K) {
        routes[row * K + lane] = valid ? idx[row * K + lane] : -1;
        weights[row * K + lane] = valid ? scores[row * K + lane] : 0;
    }
    // Uniform across this CTA. Invalid routes never read token/scaling data.
    if (!valid) return;
    for (int block = 0; block < H / 128; ++block) {
        float v = __bfloat162float(x[row * H + block * 128 + lane]);
        float a = fabsf(v);
        for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
        __shared__ float warpMax[4];
        if (lane % 32 == 0) warpMax[lane / 32] = a;
        __syncthreads();
        a = fmaxf(fmaxf(warpMax[0], warpMax[1]), fmaxf(warpMax[2], warpMax[3]));
        float scale = exp2f(ceilf(log2f(fmaxf(a / 448.0f, 0x1p-126f))));
        if (lane == 0) sf[row * (H / 128) + block] = scale;
        out[row * H + block * 128 + lane] = __nv_fp8_e4m3(v / scale);
        __syncthreads();
    }
}

struct Base { int ranks, device; virtual ~Base() = default; };
template<int R> struct Mega {
    static constexpr int LE = E / R;
    static constexpr int DecodeStages = R == 4 ? 4 : 3;
    using Buffers = fm::layout::MegaMoEBuffers;
    template<int CAP> struct Plan {
        static constexpr int POOL = fm::layout::get_num_max_pool_tokens(R, CAP, K, LE);
        // Decode includes 8-row tiles; prefill only uses 24/64 (EP4)
        // or 64 (EP8). Each tile owns 128 scaling-factor rows.
        static constexpr int MIN_BM = CAP == DECODE_CAP ? 8 : (R == 4 ? 24 : 64);
        static constexpr int SFPOOL = POOL / MIN_BM * 128;
        static Buffers buffers(void *p) { return Buffers(p, R, E, CAP, K, H, I, POOL, SFPOOL); }
        static size_t outputOffset() {
            auto b = buffers(nullptr);
            return ((size_t)b.combine_token_buffer.get_end_ptr() + 255) & ~size_t(255);
        }
        static size_t bytes() { return outputOffset() + CAP * H * sizeof(uint16_t); }
        template<int BM, int STAGES, bool SWAP, bool RS> static auto kernel() {
            constexpr int EPW = BM == 8 ? 16 : LE;
            return &fm::sm90_mxfp4_mega_moe_fused_impl<SMS, R, H, I, E, K, CAP,
                EPW, BM, 256, POOL, SFPOOL, STAGES,
                cute::numeric_limits<float>::infinity(), true, SWAP, SWAP, true, true, RS>;
        }
    };
    using Decode = Plan<DECODE_CAP>;
    using Prefill = Plan<PREFILL_CAP>;
    // Independent control counters and data arenas let large eager calls and
    // captured decode calls alternate without allocation or counter resets.
    static size_t bytes() { return Decode::bytes() + Prefill::bytes(); }
    struct Descriptors { cute::TmaDescriptor a1, s1, out, a2, s2; };
    struct Layer : Base {
        int rank;
        void *workspace;
        std::array<int64_t, R> peers;
        Allocation gate, down, globals;
        std::array<Descriptors, 3> decodeDescriptors, prefillDescriptors;
        Layer() { ranks = R; check(cudaGetDevice(&device)); }
    };
    template<int CAP> static void prepareDescriptors(Descriptors &d, Buffers &b, int bm) {
        constexpr int POOL = Plan<CAP>::POOL, SFPOOL = Plan<CAP>::SFPOOL;
        descriptor(d.a1, b.l1_token_buffer.base, H, POOL, 128, bm, false, true);
        descriptor(d.s1, b.l1_sf_buffer.base, SFPOOL, H / 128, bm, 1, true);
        descriptor(d.out, b.l2_token_buffer.base, I, POOL, 128, bm);
        descriptor(d.a2, b.l2_token_buffer.base, I, POOL, 128, bm, false, true);
        descriptor(d.s2, b.l2_sf_buffer.base, SFPOOL, I / 128, bm, 1, true);
    }
    static Layer *create(const void *const *gate, const void *const *down,
            void *workspace, size_t size, const void *const *peers, int rank) {
        if (!workspace || size < bytes() || rank < 0 || rank >= R || peers[rank] != workspace)
            throw std::runtime_error("Invalid MXFP4 EP workspace");
        auto l = std::make_unique<Layer>();
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, l->device));
        if (prop.major != 9 || prop.minor != 0 || prop.multiProcessorCount != SMS)
            throw std::runtime_error("MXFP4 EP adapter requires H20 SM90/78 SMs");
        l->rank = rank; l->workspace = workspace;
        for (int r = 0; r < R; ++r) l->peers[r] = (int64_t)peers[r];
        const size_t gateBytes = (size_t)(2 * I) * H / 32 * 17;
        const size_t downBytes = (size_t)H * I / 32 * 17;
        l->gate.allocate(LE * gateBytes); l->down.allocate(LE * downBytes);
        l->globals.allocate(2 * LE * sizeof(float));
        std::vector<float> gs(2 * LE);
        for (int e = 0; e < LE; ++e) for (int leg = 0; leg < 2; ++leg) {
            const int n = leg ? H : 2 * I, k = leg ? I : H;
            const auto *src = (const uint8_t *)(leg ? down[e] : gate[e]);
            if (!src) throw std::runtime_error("Missing MXFP4 expert");
            std::vector<uint8_t> scales((size_t)n * k / 32);
            check(cudaMemcpy(scales.data(), src + (size_t)n * k / 2, scales.size(), cudaMemcpyDeviceToHost));
            auto bounds = std::minmax_element(scales.begin(), scales.end());
            int lo = *bounds.first, hi = *bounds.second;
            if (lo == 0 || hi == 255 || hi - lo > 11)
                throw std::runtime_error("MXFP4 expert " + std::to_string(rank * LE + e) +
                    (leg ? " down" : " gate/up") + " scales [" + std::to_string(lo) + "," +
                    std::to_string(hi) + "] exceed lossless FP8 decoder range; use --ep 1");
            int shift = 122 - lo;
            gs[leg * LE + e] = std::ldexp(1.0f, -shift);
            if (!std::isfinite(gs[leg * LE + e]) || gs[leg * LE + e] == 0)
                throw std::runtime_error("MXFP4 expert global scale is not representable");
            size_t packedBytes = leg ? downBytes : gateBytes;
            auto *dst = (uint8_t *)(leg ? l->down.p : l->gate.p) + e * packedBytes;
            pack<<<(packedBytes + 255) / 256, 256, 0, cudaStreamPerThread>>>(src, dst, n, k, shift, !leg);
            check(cudaGetLastError());
        }
        check(cudaMemcpyAsync(l->globals.p, gs.data(), gs.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        auto decode = Decode::buffers(workspace);
        prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[0], decode, 8);
        prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[1], decode, 24);
        prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[2], decode, 64);
        auto prefill = Prefill::buffers((char *)workspace + Decode::bytes());
        if constexpr (R == 4)
            prepareDescriptors<PREFILL_CAP>(l->prefillDescriptors[1], prefill, 24);
        prepareDescriptors<PREFILL_CAP>(l->prefillDescriptors[2], prefill, 64);
        check(cudaFuncSetAttribute(Decode::template kernel<8, 8, true, true>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        check(cudaFuncSetAttribute(Decode::template kernel<24, DecodeStages, true, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        check(cudaFuncSetAttribute(Decode::template kernel<64, 3, false, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if constexpr (R == 4)
            check(cudaFuncSetAttribute(Prefill::template kernel<24, 3, true, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        check(cudaFuncSetAttribute(Prefill::template kernel<64, 3, false, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        return l.release();
    }
    template<int CAP, int BM, int STAGES, bool SWAP, bool RS>
    static void launch(Layer &l, int rows, int di) {
        const size_t offset = CAP == DECODE_CAP ? 0 : Decode::bytes();
        const auto &d = (CAP == DECODE_CAP ? l.decodeDescriptors : l.prefillDescriptors)[di];
        auto peers = l.peers;
        for (auto &p : peers) p += offset;
        auto sym = fm::layout::SymBuffer<R>(peers, l.rank);
        void *y = (char *)l.workspace + offset + Plan<CAP>::outputOffset();
        Plan<CAP>::template kernel<BM, STAGES, SWAP, RS>()<<<SMS, fm::sm90_mxfp4_num_threads(SWAP, RS, BM), SMEM, cudaStreamPerThread>>>(
            y, nullptr, rows, sym, d.a1, d.s1, d.out, d.a2, d.s2,
            (const uint8_t *)l.gate.p, (const uint8_t *)l.down.p,
            (const float *)l.globals.p, (const float *)l.globals.p + LE, nullptr);
    }
    template<int CAP> static void runPlan(Layer &l, const void *x, void *y,
            const int32_t *idx, const float *scores, int rows, int commonRows, const int *live) {
        const size_t offset = CAP == DECODE_CAP ? 0 : Decode::bytes();
        auto b = Plan<CAP>::buffers((char *)l.workspace + offset);
        // Clear the entire selected arena's route table, including idle sources.
        stage<<<CAP, 128, 0, cudaStreamPerThread>>>((const __nv_bfloat16 *)x, idx, scores,
            (__nv_fp8_e4m3 *)b.input_token_buffer.base, (float *)b.input_sf_buffer.base,
            (int64_t *)b.input_topk_idx_buffer.base, (float *)b.input_topk_weights_buffer.base, rows, live);
        const int routedRows = commonRows * R * K;
        if constexpr (CAP == DECODE_CAP) {
            // Keep the register-source tile for tiny local batches. Wider
            // tiles use EP-wide density to avoid padding twice as many experts.
            if (commonRows <= 16) launch<CAP, 8, 8, true, true>(l, commonRows, 0);
            else if (routedRows <= 24 * E) launch<CAP, 24, DecodeStages, true, false>(l, commonRows, 1);
            else launch<CAP, 64, 3, false, false>(l, commonRows, 2);
        } else {
            if constexpr (R == 4) {
                if (routedRows <= 24 * E) launch<CAP, 24, 3, true, false>(l, commonRows, 1);
                else launch<CAP, 64, 3, false, false>(l, commonRows, 2);
            } else launch<CAP, 64, 3, false, false>(l, commonRows, 2);
        }
        check(cudaGetLastError());
        if (rows) check(cudaMemcpyAsync(y, (char *)l.workspace + offset + Plan<CAP>::outputOffset(),
            (size_t)rows * H * 2, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    static void run(Layer &l, const void *x, void *y, const int32_t *idx,
            const float *scores, int rows, int commonRows, const int *live) {
        if (rows < 0 || rows > commonRows || commonRows < 1 || commonRows > PREFILL_CAP || !x || !y || !idx || !scores)
            throw std::runtime_error("Invalid MXFP4 EP row count or pointer");
        int device; check(cudaGetDevice(&device));
        if (device != l.device) throw std::runtime_error("MXFP4 EP device mismatch");
        if (commonRows <= DECODE_CAP) runPlan<DECODE_CAP>(l, x, y, idx, scores, rows, commonRows, live);
        else runPlan<PREFILL_CAP>(l, x, y, idx, scores, rows, commonRows, live);
    }
};
}

extern "C" const char *fastllm_mxfp4_ep_error() { return error.c_str(); }
extern "C" int fastllm_mxfp4_ep_capacity() { return DECODE_CAP; }
extern "C" int fastllm_mxfp4_ep_prefill_capacity() { return PREFILL_CAP; }
extern "C" size_t fastllm_mxfp4_ep_workspace_bytes(int ranks) {
    return ranks == 4 ? Mega<4>::bytes() : ranks == 8 ? Mega<8>::bytes() : 0;
}
extern "C" int fastllm_mxfp4_ep_init(void *p, size_t bytes, const int *devices, int rank, int ranks) {
    try {
        size_t required = fastllm_mxfp4_ep_workspace_bytes(ranks);
        int device; check(cudaGetDevice(&device));
        if (!required || !p || bytes < required || rank < 0 || rank >= ranks || devices[rank] != device)
            throw std::runtime_error("Invalid MXFP4 EP initialization");
        for (int r = 0; r < ranks; ++r) if (r != rank) {
            int accessible; check(cudaDeviceCanAccessPeer(&accessible, device, devices[r]));
            if (!accessible) throw std::runtime_error("MXFP4 EP requires all-pair peer access");
            auto s = cudaDeviceEnablePeerAccess(devices[r], 0);
            if (s == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(s);
        }
        check(cudaMemset(p, 0, bytes)); check(cudaDeviceSynchronize()); return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" void *fastllm_mxfp4_ep_create(const void *const *gate, const void *const *down,
        void *workspace, size_t bytes, const void *const *peers, int rank, int ranks) {
    try {
        if (ranks == 4) return Mega<4>::create(gate, down, workspace, bytes, peers, rank);
        if (ranks == 8) return Mega<8>::create(gate, down, workspace, bytes, peers, rank);
        throw std::runtime_error("MXFP4 EP requires four or eight ranks");
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
extern "C" int fastllm_mxfp4_ep_run(void *handle, const void *x, void *y, const int32_t *idx,
        const float *scores, int rows, int commonRows, const int *live) {
    try {
        if (!handle) throw std::runtime_error("Missing MXFP4 EP handle");
        if (static_cast<Base *>(handle)->ranks == 4)
            Mega<4>::run(*static_cast<Mega<4>::Layer *>(handle), x, y, idx, scores, rows, commonRows, live);
        else Mega<8>::run(*static_cast<Mega<8>::Layer *>(handle), x, y, idx, scores, rows, commonRows, live);
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" void fastllm_mxfp4_ep_destroy(void *handle) {
    if (!handle) return;
    int device; cudaGetDevice(&device); auto *p = static_cast<Base *>(handle);
    cudaSetDevice(p->device); delete p; cudaSetDevice(device);
}
