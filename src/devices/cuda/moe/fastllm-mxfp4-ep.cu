// Native adapter for FlashInfer v0.7.4 SM90 MXFP4 MegaMoE (Apache-2.0).
// See third_party/flashinfer_mxfp4. No Torch, TVM-FFI or Python runtime.
#include "devices/cuda/fastllm-mxfp4-ep.h"
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include "devices/cuda/fastllm-cuda-pdl.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#define FI_MEGAMOE_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <flashinfer/megamoe/sm90_mxfp4_ep8_verify.cuh>
#include <deep_gemm/impls/sm90_mega_moe_pre_dispatch.cuh>
#include "fastllm-cuda-router-topk.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// NVCC emits this spelling in host stubs for floating-point template arguments.
static constexpr float inf = __builtin_inff();
namespace {
namespace fm = flashinfer::megamoe;
constexpr int H = 4096, DECODE_CAP = 128, MAIN_PREFILL_CAP = 2048;
constexpr int EP_CTAS = 78;  // EP4/EP8 cross-rank scheduler specialization.
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

// One 16-byte load supplies both register rows for two successive K32 slices.
__device__ __forceinline__ int localWeightWord(int row, int group, int slice) {
    const int lane = (row % 8) * 4 + group;
    const int halfTile = (row / 16) * 2 + slice / 2;
    return (halfTile * 32 + lane) * 4 + (slice % 2) * 2 + (row % 16) / 8;
}

// Place the two scale words used by one lane beside each other.
__device__ __forceinline__ int localScaleRow(int row) {
    return (row / 16) * 16 + (row % 8) * 2 + (row % 16) / 8;
}

// Convert canonical adjacent E2M1 nibbles directly to the upstream braided,
// chunk-major tiles. The power-of-two scale shift is exactly compensated by
// an expert-global multiplier, applied BEFORE SwiGLU in the upstream kernel.
template<bool LanePacked>
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
        // A local RS lane owns four K values from each half of one K32 group.
        // Keep both halves in one word so the lane can decode its own registers.
        const auto sourceColumn = [&](int element) {
            if constexpr (LanePacked)
                return group / 32 * 32 + group % 32 / 8 * 4 + element / 4 * 16 + element % 4;
            return group + element;
        };
        unsigned a = w[sourceColumn(2 * pair) / 2] & 0x77;
        int s0 = sourceColumn(signSource[2 * pair]), s1 = sourceColumn(signSource[2 * pair + 1]);
        a |= ((w[s0 / 2] >> (s0 % 2 * 4)) & 8);
        a |= ((w[s1 / 2] >> (s1 % 2 * 4)) & 8) << 4;
        if constexpr (LanePacked) {
            const int word = localWeightWord(byte / 16 % 128, byte % 16 / 4, byte / 2048);
            dst[tile * tileBytes + word * 4 + byte % 4] = a;
        } else {
            dst[i] = a;
        }
    } else {
        int row = nr + (byte - 8192) / 4, col = kb * 4 + (byte - 8192) % 4;
        if (gateup) row = row / 16 * 8 + row % 8 + (row % 16 >= 8 ? n / 2 : 0);
        int value = src[(size_t)n * k / 2 + (size_t)row * (k / 32) + col];
        if constexpr (LanePacked) {
            const uint32_t physical = (byte - 8192) / 4;
            const uint32_t scaleRow = localScaleRow(physical);
            dst[tile * tileBytes + 8192 + scaleRow * 4 + byte % 4] = value + shift;
        } else {
            dst[i] = value + shift;
        }
    }
}

// Recentring must represent every UE8M0 scale and the reciprocal global scale exactly.
bool expertScaleShift(const uint8_t *source, int n, int k, int &shift) {
    std::vector<uint8_t> scales(static_cast<size_t>(n) * k / 32);
    check(cudaMemcpy(scales.data(), source + static_cast<size_t>(n) * k / 2,
                     scales.size(), cudaMemcpyDeviceToHost));
    const auto bounds = std::minmax_element(scales.begin(), scales.end());
    const int lo = *bounds.first, hi = *bounds.second;
    shift = 122 - lo;
    const float global = std::ldexp(1.0f, -shift);
    if (lo && hi != 255 && hi - lo <= 11 && std::isfinite(global) && global != 0) return true;
    error = "scales [" + std::to_string(lo) + "," + std::to_string(hi) + "] exceed the lossless FP8 range";
    return false;
}

struct Base {
    int device;
    virtual ~Base() = default;
    virtual void forward(const void *, void *, const int32_t *, const float *, int, int, const int *) = 0;
};
template<int Rows> struct SelectLocalRoutes {
    static constexpr int tileRows = Rows;
    const float *bias;
    uint8_t *slots = nullptr;
    float *expertWeights = nullptr;
    struct Scratch {
        fastllm::cuda::RouterTopKScratch<8> top;
        int32_t ids[8];
        float weights[8];
    };
    __device__ void selectWarp(uint32_t row, const float *probabilities, float scale, Scratch &s) const {
        fastllm::cuda::SelectRouterTopK<256,8>(probabilities + row * 256, bias, s.ids, s.weights,
                                              true, true, scale, s.top);
        __syncwarp();
    }
    __device__ void publish(uint32_t row, uint32_t k, int64_t *outIds, const Scratch &s) const {
        const unsigned lane = threadIdx.x % 32;
        if (lane < 8) {
            outIds[row * k + lane] = s.ids[lane];
            slots[s.ids[lane] * Rows + row] = lane + 1;
            expertWeights[s.ids[lane] * Rows + row] = s.weights[lane];
        }
    }
    template<uint32_t Stride>
    __device__ void write(uint32_t row, uint32_t k, const int32_t *, const float *probabilities,
                          int64_t *outIds, float *, float scale) const {
        __shared__ Scratch scratch;
        if (threadIdx.x < 256) {
            slots[threadIdx.x * Rows + row] = 0;
            expertWeights[threadIdx.x * Rows + row] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            selectWarp(row, probabilities, scale, scratch);
            publish(row, k, outIds, scratch);
        }
    }
};

__device__ __forceinline__ int physicalEpExpert(int logical, const uint8_t *map) {
    return map && static_cast<unsigned>(logical) < 256 ? map[logical] : logical;
}

struct RemapSelectedRoutes {
    const uint8_t *map;
    template<uint32_t Stride>
    __device__ void write(uint32_t row, uint32_t k, const int32_t *ids, const float *weights,
                          int64_t *out, float *outWeights, float scale) const {
        if (threadIdx.x < k) {
            const unsigned slot = row * k + threadIdx.x;
            out[slot] = physicalEpExpert(ids[slot], map);
            outWeights[slot * Stride] = weights[slot] * scale;
        }
    }
};

// Single-source-row EP8 verification. Dispatch warps write the same FP8 row
// and route records as pre-dispatch, before the existing dispatch grid barrier.
struct PrepareEpSource {
    using Argument = PrepareEpSource;
    static constexpr unsigned slots = 8;
    const __nv_bfloat16 *input;
    const int32_t *ids;
    const float *scores;
    const int *live;
    const uint8_t *map = nullptr;

    __device__ bool active() const { return input && (!live || *live > 0); }
    __device__ int expert(uint32_t slot) const { return active() ? physicalEpExpert(ids[slot], map) : -1; }
    __device__ void prepareSource(const fm::layout::MegaMoEBuffers &b) const {
        const uint32_t t = blockIdx.x * 32 + threadIdx.x;
        if (t < 8) {
            static_cast<int64_t *>(b.input_topk_idx_buffer.base)[t] = expert(t);
            static_cast<float *>(b.input_topk_weights_buffer.base)[t] = active() ? scores[t] : 0.f;
        }
        if (t < H / 8 && active())
            fastllm_ep8_deep_gemm::quantize_input_vector<128, true, 1, 1>(input,
                static_cast<__nv_fp8_e4m3 *>(b.input_token_buffer.base),
                static_cast<float *>(b.input_sf_buffer.base), 0, t, H, H / 128);
    }
};

// A DP verification source owns one or more candidate rows. Padded sources still
// publish empty routes and participate in exactly the same EP barriers.
template<unsigned MaxRows> struct PrepareEpRows : PrepareEpSource {
    using Argument = PrepareEpRows;
    static constexpr unsigned slots = MaxRows * 8;
    unsigned rows;
    __device__ bool active(unsigned row) const { return row < rows && (!live || live[row] > 0); }
    __device__ int expert(unsigned slot) const { return active(slot / 8) ? physicalEpExpert(ids[slot], map) : -1; }
    template<unsigned Threads = 32>
    __device__ void prepareSource(const fm::layout::MegaMoEBuffers &b) const {
        const unsigned t = blockIdx.x * Threads + threadIdx.x;
        if (t < slots) {
            static_cast<int64_t *>(b.input_topk_idx_buffer.base)[t] = expert(t);
            static_cast<float *>(b.input_topk_weights_buffer.base)[t] = active(t / 8) ? scores[t] : 0.f;
        }
        for (unsigned i = t; i < rows * (H / 8); i += gridDim.x * Threads) {
            const unsigned row = i / (H / 8);
            if (active(row))
                fastllm_ep8_deep_gemm::quantize_input_vector<128, true, 1, 1>(input,
                    static_cast<__nv_fp8_e4m3 *>(b.input_token_buffer.base),
                    static_cast<float *>(b.input_sf_buffer.base), row, i % (H / 8), H, H / 128);
        }
    }
};

struct ExpertTileMaps {
    cute::TmaDescriptor a1, s1, out, a2, s2;
};
template<unsigned Rows> struct PrepareEpVerify : PrepareEpRows<Rows> {
    using Argument = PrepareEpVerify;
    ExpertTileMaps narrowMaps;
};

template<int Rows, int ScaleStride> struct PrepareVerificationInput {
    static constexpr int tileRows = Rows;
    using Argument = PrepareVerificationInput;
    const __nv_bfloat16 *input;
    const float *probabilities, *bias;
    __device__ void run(const fm::layout::MegaMoEBuffers &b, unsigned rows, uint8_t *scratch) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // TMA descriptor prefetch and CTA-private initialization precede this wait.
        if constexpr (Rows == 8) cudaGridDependencySynchronize();
#endif
        const SelectLocalRoutes<Rows> routes{bias, b.input_token_buffer.get_base_ptr<uint8_t>(),
                                             b.local_routed_weights(Rows)};
        auto &top = *reinterpret_cast<typename SelectLocalRoutes<Rows>::Scratch *>(scratch);
        // The compact plan repacks rows after the grid barrier. The smaller
        // plans write the WGMMA input and its strided scales in place.
        auto *tokens = (Rows == 32 ? b.l2_token_buffer : b.l1_token_buffer).get_base_ptr<__nv_fp8_e4m3>();
        auto *scales = (Rows == 32 ? b.input_sf_buffer : b.l1_sf_buffer).get_base_ptr<float>();
        constexpr unsigned stripes = Rows == 32 ? 2 : 4;
        constexpr unsigned vectors = H / 8 / stripes;
        static_assert(H % (8 * stripes) == 0 && vectors <= 256 && vectors % 32 == 0);
        for (unsigned stripe = blockIdx.x; stripe < stripes * rows; stripe += gridDim.x) {
            const unsigned row = stripe / stripes;
            const bool select = stripe % stripes == 0;
            if (threadIdx.x < 256) {
                if (select) {
                    routes.slots[threadIdx.x * Rows + row] = 0;
                    if constexpr (Rows != 32) routes.expertWeights[threadIdx.x * Rows + row] = 0.f;
                }
                if (threadIdx.x < vectors)
                    fastllm_ep8_deep_gemm::quantize_input_vector<128, true, ScaleStride, 1>(
                        input, tokens, scales, row, (stripe % stripes) * vectors + threadIdx.x, H, H / 128);
            } else if (select && threadIdx.x < 288) {
                routes.selectWarp(row, probabilities, 1.f, top);
            }
            if (select) {
                __syncthreads();
                if (threadIdx.x >= 256 && threadIdx.x < 288)
                    routes.publish(row, 8, b.input_topk_idx_buffer.get_base_ptr<int64_t>(), top);
            }
        }
        if constexpr (Rows != 32) asm volatile("fence.proxy.async.global;" ::: "memory");
    }
};

template<int R, int E = 256, int K = 8, int I = 2048> struct Mega {
    static constexpr int LE = E / R;
    static constexpr int PREFILL_CAP = E == 128 ? 256 : MAIN_PREFILL_CAP;
    static constexpr int DecodeBlock = R == 8 ? 32 : 24;
    static constexpr int DecodeStages = R == 4 ? 4 : 3;
    using Buffers = fm::layout::MegaMoEBuffers;
    template<int CAP> struct Plan {
        static constexpr int POOL = R == 1 ? kFastllmMxfp4LocalMaxRows * K * 8 : fm::layout::get_num_max_pool_tokens(R, CAP, K, LE);
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
        template<int BM, int STAGES, bool SWAP, bool RS, int LocalRows = (R == 1 ? 1 : 0), bool Grouped = false, typename Prologue = fm::Mxfp4ProfileArgument> static constexpr auto kernel() {
            constexpr int EPW = R == 1 ? LE : BM == 8 ? 16 : LE;
            if constexpr (std::is_same_v<Prologue, PrepareEpVerify<64>> || std::is_same_v<Prologue, PrepareEpVerify<128>>) {
                static_assert(R == 8 && E == 256 && K == 8 && BM == 64 && !SWAP && !RS);
                return &fm::sm90_mxfp4_ep8_verify<EP_CTAS, H, I, CAP, POOL, SFPOOL, Prologue>;
            } else {
                return &fm::sm90_mxfp4_mega_moe_fused_impl<EP_CTAS, R, H, I, E, K, CAP,
                    EPW, BM, 256, POOL, SFPOOL, STAGES,
                    cute::numeric_limits<float>::infinity(), true, SWAP, SWAP, true, R != 1 && !std::is_same_v<Prologue, PrepareEpSource>,
                    RS, false, LocalRows, Grouped, R == 1, Prologue>;
            }
        }
    };
    using Decode = Plan<DECODE_CAP>;
    template<int Rows> using VerifyInput = PrepareVerificationInput<Rows, Rows == 32 ? 32 : Decode::SFPOOL>;
    using Prefill = Plan<PREFILL_CAP>;
    // Independent control counters and data arenas let large eager calls and
    // captured decode calls alternate without allocation or counter resets.
    static size_t bytes() { return Decode::bytes() + (R == 1 ? 0 : Prefill::bytes()); }
    static bool supportsDevice(const cudaDeviceProp &prop) {
        if (prop.major != 9 || prop.minor != 0 || prop.sharedMemPerBlockOptin < SMEM) return false;
        return R == 1 ? prop.multiProcessorCount > 0 : prop.multiProcessorCount == EP_CTAS;
    }
    // One persistent CTA per visible SM. Check each local variant before a model
    // commits to packed-only expert weights.
    static bool configureLocalKernels() {
        const auto resident = [](auto kernel) {
            int blocks = 0;
            constexpr int threads = fm::sm90_mxfp4_num_threads(true, true, 8);
            if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, SMEM) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            return blocks >= 1;
        };
        return resident(Decode::template kernel<8,8,true,true>()) &&
               resident(Decode::template kernel<8,8,true,true,kFastllmMxfp4LocalMaxRows>()) &&
               resident(Decode::template kernel<8,8,true,true,8,true,VerifyInput<8>>()) &&
               resident(Decode::template kernel<16,8,true,true,16,true>()) &&
               resident(Decode::template kernel<8,8,true,true,32,true,VerifyInput<32>>());
    }
    using Descriptors = ExpertTileMaps;
    struct Layer : Base {
        int rank, gridCtas;
        void *workspace;
        std::array<int64_t, R> peers;
        Allocation gate, down, globals, expertMap;
        std::array<Descriptors, 3> decodeDescriptors, prefillDescriptors;
        Layer() { check(cudaGetDevice(&device)); }
        void forward(const void *input, void *output, const int32_t *ids, const float *scores,
                     int rows, int commonRows, const int *live) override {
            Mega::run(*this, input, output, ids, scores, rows, commonRows, live);
        }
    };
    template<int CAP> static void prepareDescriptors(Descriptors &d, Buffers &b, int tileRows) {
        constexpr int POOL = Plan<CAP>::POOL, SFPOOL = Plan<CAP>::SFPOOL;
        const bool compact = R == 1 && tileRows == 32;
        const int bm = compact ? 8 : tileRows;
        descriptor(d.a1, b.l1_token_buffer.base, H, compact ? E : POOL, 128, bm, false, true);
        descriptor(d.s1, b.l1_sf_buffer.base, SFPOOL, H / 128, bm, 1, true);
        void *intermediate = b.l2_token_buffer.base;
        int intermediateRows = POOL;
        if constexpr (R == 1) {
            if (bm == 16) {
                // Local verification stores the input once. Use spare rows in its
                // FP8 input pool for one intermediate tile per distinct expert.
                constexpr int rows = 16;
                constexpr int maxPoolRows = rows * K * rows;
                static_assert((rows + 8) * H + maxPoolRows * I <= POOL * H && maxPoolRows <= SFPOOL);
                static_assert(E * rows * sizeof(float) <= CAP * (H / 32));
                intermediate = b.l1_token_buffer.get_data_buffer(rows + 8).base;
                intermediateRows = maxPoolRows;
            } else if (compact) {
                constexpr int rows = 32;
                constexpr int pairs = rows * K;
                constexpr int maxPoolRows = pairs * 8;
                // The existing L1 arena holds up to 256 dense inputs followed
                // by at most 256 BM8 intermediate tiles, one per route pair.
                static_assert(E == pairs && rows * H <= POOL * I);
                static_assert(maxPoolRows * I <= (POOL - pairs) * H && maxPoolRows <= SFPOOL);
                static_assert(E * rows * (1 + sizeof(float)) <= CAP * H);
                intermediate = b.l1_token_buffer.get_data_buffer(pairs).base;
                intermediateRows = maxPoolRows;
            }
        }
        descriptor(d.out, intermediate, I, intermediateRows, 128, bm);
        descriptor(d.a2, intermediate, I, intermediateRows, 128, bm, false, true);
        descriptor(d.s2, b.l2_sf_buffer.base, SFPOOL, I / 128, bm, 1, true);
    }
    static Layer *create(const void *const *gate, const void *const *down,
            void *workspace, size_t size, const void *const *peers, int rank, const int *shifts = nullptr) {
        if (!workspace || size < bytes() || rank < 0 || rank >= R || peers[rank] != workspace)
            throw std::runtime_error("Invalid MXFP4 EP workspace");
        auto l = std::make_unique<Layer>();
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, l->device));
        if (!supportsDevice(prop))
            throw std::runtime_error("Unsupported device for this SM90 MXFP4 MegaMoE grid");
        if constexpr (R == 1) {
            if (!configureLocalKernels())
                throw std::runtime_error("SM90 TP-local MXFP4 requires a fully resident grid");
        }
        l->gridCtas = R == 1 ? prop.multiProcessorCount : EP_CTAS;
        l->rank = rank; l->workspace = workspace;
        for (int r = 0; r < R; ++r) l->peers[r] = (int64_t)peers[r];
        const size_t gateBytes = (size_t)(2 * I) * H / 32 * 17;
        const size_t downBytes = (size_t)H * I / 32 * 17;
        l->gate.allocate(LE * gateBytes);
        l->down.allocate(LE * downBytes);
        l->globals.allocate(2 * LE * sizeof(float));
        std::vector<float> gs(2 * LE);
        for (int e = 0; e < LE; ++e) for (int leg = 0; leg < 2; ++leg) {
            const int n = leg ? H : 2 * I, k = leg ? I : H;
            const auto *src = (const uint8_t *)(leg ? down[e] : gate[e]);
            if (!src) throw std::runtime_error("Missing MXFP4 expert");
            int shift = 0;
            if (shifts) shift = shifts[leg * LE + e];
            else if (!expertScaleShift(src, n, k, shift))
                throw std::runtime_error("MXFP4 expert " + std::to_string(rank * LE + e) +
                    (leg ? " down: " : " gate/up: ") + error);
            gs[leg * LE + e] = std::ldexp(1.0f, -shift);
            size_t packedBytes = leg ? downBytes : gateBytes;
            auto *dst = (uint8_t *)(leg ? l->down.p : l->gate.p) + e * packedBytes;
            pack<R == 1><<<(packedBytes + 255) / 256, 256, 0, cudaStreamPerThread>>>(src, dst, n, k, shift, !leg);
            check(cudaGetLastError());
        }
        check(cudaMemcpyAsync(l->globals.p, gs.data(), gs.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        auto decode = Decode::buffers(workspace);
        prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[0], decode, 8);
        if constexpr (R == 1) {
            prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[1], decode, 16);
            prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[2], decode, 32);
        } else {
            if constexpr (R == 8 && E == 256 && K == 8) {
                check(cudaFuncSetAttribute(Decode::template kernel<64, 3, false, false, 0, false, PrepareEpVerify<64>>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
                check(cudaFuncSetAttribute(Decode::template kernel<64, 3, false, false, 0, false, PrepareEpVerify<128>>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
                check(cudaFuncSetAttribute(Decode::template kernel<32,3,true,false,0,false,PrepareEpRows<16>>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
                check(cudaFuncSetAttribute(Decode::template kernel<32,3,true,false,0,false,PrepareEpRows<32>>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
                check(cudaFuncSetAttribute(Decode::template kernel<8,4,true,true,0,false,PrepareEpRows<8>>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
                check(cudaFuncSetAttribute(Decode::template kernel<8,4,true,true,0,false,PrepareEpSource>(),
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            }
            check(cudaFuncSetAttribute(Decode::template kernel<8, 8, true, true>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[1], decode, DecodeBlock);
            prepareDescriptors<DECODE_CAP>(l->decodeDescriptors[2], decode, 64);
            auto prefill = Prefill::buffers((char *)workspace + Decode::bytes());
            if constexpr (R == 4)
                prepareDescriptors<PREFILL_CAP>(l->prefillDescriptors[1], prefill, 24);
            prepareDescriptors<PREFILL_CAP>(l->prefillDescriptors[2], prefill, 64);
            check(cudaFuncSetAttribute(Decode::template kernel<DecodeBlock, DecodeStages, true, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            check(cudaFuncSetAttribute(Decode::template kernel<64, 3, false, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            if constexpr (R == 4)
                check(cudaFuncSetAttribute(Prefill::template kernel<24, 3, true, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            check(cudaFuncSetAttribute(Prefill::template kernel<64, 3, false, false>(), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        }
        return l.release();
    }
    template<int CAP, int BM, int STAGES, bool SWAP, bool RS, int LocalRows = (R == 1 ? 1 : 0), bool Grouped = false, class Prologue = fm::Mxfp4ProfileArgument>
    static void launch(Layer &l, void *y, int rows, int di, typename Prologue::Argument prepare = nullptr) {
        const size_t offset = CAP == DECODE_CAP ? 0 : Decode::bytes();
        const auto &d = (CAP == DECODE_CAP ? l.decodeDescriptors : l.prefillDescriptors)[di];
        auto peers = l.peers;
        for (auto &p : peers) p += offset;
        auto sym = fm::layout::SymBuffer<R>(peers, l.rank);
        constexpr auto kernel = Plan<CAP>::template kernel<BM, STAGES, SWAP, RS, LocalRows, Grouped, Prologue>();
        constexpr int threads = fm::sm90_mxfp4_num_threads(SWAP, RS, BM);
        if constexpr (R == 1 && std::is_same_v<Prologue, VerifyInput<8>>) {
            if (fastllm::cuda_pdl::Follows(fastllm::cuda_pdl::Producer::VerifyRouter)) {
                check(fastllm::cuda_launch::LaunchProgrammatic<kernel>(l.gridCtas, threads, SMEM, cudaStreamPerThread,
                    y, static_cast<int *>(nullptr), uint32_t(rows), sym, d.a1, d.s1, d.out, d.a2, d.s2,
                    (const uint8_t *)l.gate.p, (const uint8_t *)l.down.p,
                    (const float *)l.globals.p, (const float *)l.globals.p + LE, prepare));
                return;
            }
        }
        kernel<<<l.gridCtas, threads, SMEM, cudaStreamPerThread>>>(
            y, nullptr, rows, sym, d.a1, d.s1, d.out, d.a2, d.s2,
            (const uint8_t *)l.gate.p, (const uint8_t *)l.down.p,
            (const float *)l.globals.p, (const float *)l.globals.p + LE, prepare);
    }
    template<int CAP, bool prepareInKernel = false, class Routes = fastllm_ep8_deep_gemm::CopySelectedRoutes>
    static void runPlan(Layer &l, const void *x, void *y,
            const int32_t *idx, const float *scores, int rows, int commonRows, const int *live, Routes routes = {}) {
        if constexpr (CAP == DECODE_CAP && R == 8 && E == 256 && K == 8) {
            // Every rank must use the same collective, including empty ranks
            // and callers that omit a local mask for a fully active source.
            static_assert(DECODE_CAP == 128);
            const size_t off = Plan<CAP>::outputOffset();
            const bool stage = rows < commonRows || reinterpret_cast<uintptr_t>(y) % alignof(uint4) != 0;
            void *dst = stage ? static_cast<char *>(l.workspace) + off : y;
            PrepareEpSource source{rows ? static_cast<const __nv_bfloat16 *>(x) : nullptr, idx, scores, live, static_cast<const uint8_t *>(l.expertMap.p)};
            if (commonRows == 1)
                launch<CAP, 8, 4, true, true, 0, false, PrepareEpSource>(l, dst, 1, 0, source);
            else if (commonRows <= 8)
                launch<CAP, 8, 4, true, true, 0, false, PrepareEpRows<8>>(l, dst, commonRows, 0,
                    PrepareEpRows<8>{source, static_cast<unsigned>(rows)});
            else if (commonRows <= 16)
                launch<CAP, DecodeBlock, DecodeStages, true, false, 0, false, PrepareEpRows<16>>(l, dst, commonRows, 1,
                    PrepareEpRows<16>{source, static_cast<unsigned>(rows)});
            else if (commonRows <= 32)
                launch<CAP, DecodeBlock, DecodeStages, true, false, 0, false, PrepareEpRows<32>>(l, dst, commonRows, 1,
                    PrepareEpRows<32>{source, static_cast<unsigned>(rows)});
            else if (commonRows <= 64)
                launch<CAP, 64, 3, false, false, 0, false, PrepareEpVerify<64>>(l, dst, commonRows, 2,
                    PrepareEpVerify<64>{{source, static_cast<unsigned>(rows)}, l.decodeDescriptors[1]});
            else
                launch<CAP, 64, 3, false, false, 0, false, PrepareEpVerify<128>>(l, dst, commonRows, 2,
                    PrepareEpVerify<128>{{source, static_cast<unsigned>(rows)}, l.decodeDescriptors[1]});
            check(cudaGetLastError());
            if (stage && rows) check(cudaMemcpyAsync(y, dst, size_t(rows) * H * sizeof(uint16_t), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            return;
        }
        const size_t offset = CAP == DECODE_CAP ? 0 : Decode::bytes();
        auto b = Plan<CAP>::buffers((char *)l.workspace + offset);
        constexpr bool grouped = !std::is_same_v<Routes, fastllm_ep8_deep_gemm::CopySelectedRoutes> &&
                                 !std::is_same_v<Routes, RemapSelectedRoutes>;
        if constexpr (!prepareInKernel) {
            // Clear the entire selected arena's route table, including idle sources.
            constexpr int threads = H / 8;
            const int paddedRows = R == 1 ? rows : CAP;
            const int paddingBlocks = ((paddedRows - rows) * K + threads - 1) / threads;
            if constexpr (grouped) {
                // Local TP does not use the EP exchange arenas.
                routes.slots = static_cast<uint8_t *>(b.input_token_buffer.base);
                routes.expertWeights = static_cast<float *>(b.input_sf_buffer.base);
            }
            fastllm_ep8_deep_gemm::sm90_mega_moe_pre_dispatch_kernel<128, false, true,
                R == 1 ? Plan<CAP>::SFPOOL : 1, R == 1 && !grouped ? 8 : 1, Routes>
                <<<rows + paddingBlocks, threads, 0, cudaStreamPerThread>>>(
                    static_cast<const __nv_bfloat16 *>(x), idx, scores,
                    static_cast<__nv_fp8_e4m3 *>(R == 1 ? b.l1_token_buffer.base : b.input_token_buffer.base),
                    static_cast<float *>(R == 1 ? b.l1_sf_buffer.base : b.input_sf_buffer.base),
                    static_cast<int64_t *>(b.input_topk_idx_buffer.base),
                    static_cast<float *>(R == 1 ? b.l1_topk_weights_buffer.base : b.input_topk_weights_buffer.base),
                    rows, paddedRows, H, H / 128, K, 1.0f, live, routes);
        }
        // The fused combine writes commonRows via 16-byte-aligned TMA stores.
        // Stage short or unaligned output views to preserve their bounds.
        void *dst = y;
        if (rows < commonRows || reinterpret_cast<uintptr_t>(y) % alignof(uint4) != 0)
            dst = static_cast<char *>(l.workspace) + offset + Plan<CAP>::outputOffset();
        const int routedRows = commonRows * R * K;
        if constexpr (R == 1) {
            if constexpr (prepareInKernel) {
                constexpr int tile = Routes::tileRows;
                static_assert(tile == 8 || tile == 32);
                launch<CAP, 8, 8, true, true, tile, true, Routes>(l, dst, rows, tile == 32 ? 2 : 0, routes);
            } else if constexpr (grouped) {
                constexpr int bm = Routes::tileRows;
                launch<CAP,bm,8,true,true,bm,true>(l, dst, rows, bm == 8 ? 0 : 1);
            }
            else if (rows == 1) launch<CAP, 8, 8, true, true>(l, dst, rows, 0);
            else launch<CAP, 8, 8, true, true, kFastllmMxfp4LocalMaxRows>(l, dst, rows, 0);
        } else if constexpr (CAP == DECODE_CAP) {
            // The cutoff is 24 routed rows per expert for both EP4 and EP8.
            if (commonRows <= 16) launch<CAP, 8, 8, true, true>(l, dst, commonRows, 0);
            else if (routedRows <= 24 * E) launch<CAP, DecodeBlock, DecodeStages, true, false>(l, dst, commonRows, 1);
            else launch<CAP, 64, 3, false, false>(l, dst, commonRows, 2);
        } else if constexpr (R == 4) {
            if (routedRows <= 24 * E) launch<CAP, 24, 3, true, false>(l, dst, commonRows, 1);
            else launch<CAP, 64, 3, false, false>(l, dst, commonRows, 2);
        } else launch<CAP, 64, 3, false, false>(l, dst, commonRows, 2);
        check(cudaGetLastError());
        if (rows && dst != y) check(cudaMemcpyAsync(y, dst,
            (size_t)rows * H * 2, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    static void run(Layer &l, const void *x, void *y, const int32_t *idx,
            const float *scores, int rows, int commonRows, const int *live) {
        if (rows < 0 || rows > commonRows || commonRows < 1 || commonRows > PREFILL_CAP || !x || !y || !idx || !scores)
            throw std::runtime_error("Invalid MXFP4 EP row count or pointer");
        int device; check(cudaGetDevice(&device));
        if (device != l.device) throw std::runtime_error("MXFP4 EP device mismatch");
        if constexpr (R == 8 && E == 256 && K == 8) {
            if (l.expertMap.p) {
                RemapSelectedRoutes map{static_cast<const uint8_t *>(l.expertMap.p)};
                if (commonRows <= DECODE_CAP) runPlan<DECODE_CAP>(l, x, y, idx, scores, rows, commonRows, live, map);
                else runPlan<PREFILL_CAP>(l, x, y, idx, scores, rows, commonRows, live, map);
                return;
            }
        }
        if constexpr (R == 1) {
            if (commonRows != rows || rows > kFastllmMxfp4LocalMaxRows)
                throw std::runtime_error("Invalid local TP MegaMoE row count");
            runPlan<DECODE_CAP>(l, x, y, idx, scores, rows, commonRows, live);
        } else if (commonRows <= DECODE_CAP) runPlan<DECODE_CAP>(l, x, y, idx, scores, rows, commonRows, live);
        else runPlan<PREFILL_CAP>(l, x, y, idx, scores, rows, commonRows, live);
    }
};
}

// Each TP rank runs an independent one-rank MegaMoE over its own 256-wide
// intermediate shard. The model reduces the eight partial hidden vectors.
using LocalTPMega = Mega<1, 256, 8, 256>;
extern "C" size_t fastllm_mxfp4_tp8_workspace_bytes(int device) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device));
    if (!LocalTPMega::supportsDevice(prop)) return 0;
    int owner;
    check(cudaGetDevice(&owner));
    if (owner != device) check(cudaSetDevice(device));
    const bool resident = LocalTPMega::configureLocalKernels();
    if (owner != device) check(cudaSetDevice(owner));
    return resident ? LocalTPMega::bytes() : 0;
}
extern "C" int fastllm_mxfp4_tp8_validate(const void *const *gate, const void *const *down, int *shifts) {
    try {
        for (int e = 0; e < 256; ++e) {
            if (!expertScaleShift(static_cast<const uint8_t *>(gate[e]), 512, H, shifts[e]) ||
                !expertScaleShift(static_cast<const uint8_t *>(down[e]), H, 256, shifts[256 + e])) return 0;
        }
        return 1;
    } catch (const std::exception &e) { error = e.what(); return -1; }
}

extern "C" void *fastllm_mxfp4_tp8_create(const void *const *gate, const void *const *down, const int *shifts,
        void *workspace, size_t bytes) {
    try {
        const void *peers[] = {workspace};
        return LocalTPMega::create(gate, down, workspace, bytes, peers, 0, shifts);
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}


namespace {
// Inverse of pack(): restore eight E2M1 values per thread and one UE8M0
// scale per K32 group. Gate/up rows are interleaved only in the packed tile.
template<int N, int K, bool Gate>
__global__ void restoreExpertWeights(const uint8_t *packed, const float *globals, uint8_t *output) {
    constexpr size_t bytes = N * K / 32 * 17;
    const int expert = blockIdx.z, row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (col >= K || row >= N) return;
    const int physical = Gate ? (row % (N / 2) / 8) * 16 + row % 8 + (row >= N / 2 ? 8 : 0) : row;
    const uint8_t *tile = packed + expert * bytes + ((physical / 128) * (K / 128) + col / 128) * 8704;
    // Undo the local lane packing and restore the original E2M1 sign bits.
    const auto *words = reinterpret_cast<const uint32_t*>(tile);
    uint32_t restored = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = (col + i) % 32;
        const uint32_t word = words[localWeightWord(physical % 128, k % 16 / 4, col % 128 / 32)];
        const int element = k % 4 + k / 16 * 4;
        const int sign = element < 4 ? 2 * element + 1 : 2 * (element - 4);
        const uint32_t nibble = ((word >> (4 * element)) & 7u) | ((word >> (4 * sign)) & 8u);
        restored |= nibble << (4 * i);
    }
    auto *dst = output + expert * bytes;
    reinterpret_cast<uint32_t *>(dst)[row * (K / 8) + col / 8] = restored;
    if (col % 32 == 0)
        dst[N * K / 2 + row * (K / 32) + col / 32] =
            tile[8192 + localScaleRow(physical % 128) * 4 + col % 128 / 32] + ilogbf(globals[expert]);
}
}

extern "C" int fastllm_mxfp4_tp8_restore(void *layer, void *staging) {
    try {
        auto &l = *static_cast<LocalTPMega::Layer *>(layer);
        const dim3 threads(32, 8);
        auto *out = static_cast<uint8_t *>(staging);
        const auto *globals = static_cast<const float *>(l.globals.p);
        restoreExpertWeights<512, H, true><<<dim3(16, 64, 256), threads, 0, cudaStreamPerThread>>>(
            static_cast<const uint8_t *>(l.gate.p), globals, out);
        check(cudaGetLastError());
        restoreExpertWeights<H, 256, false><<<dim3(1, 512, 256), threads, 0, cudaStreamPerThread>>>(
            static_cast<const uint8_t *>(l.down.p), globals + 256, out + 256ull * (512 * H / 32 * 17));
        check(cudaGetLastError());
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" const char *fastllm_mxfp4_ep_error() { return error.c_str(); }
extern "C" int fastllm_mxfp4_ep_capacity() { return DECODE_CAP; }
extern "C" int fastllm_mxfp4_ep_prefill_capacity() { return MAIN_PREFILL_CAP; }
extern "C" size_t fastllm_mxfp4_ep_workspace_bytes(int ranks) {
    return ranks == 4 ? Mega<4>::bytes() : ranks == 8 ? Mega<8>::bytes() : 0;
}
extern "C" size_t fastllm_mxfp4_ep_draft_workspace(int ranks, int *prefillCapacity) {
    size_t bytes = ranks == 4 ? Mega<4, 128, 3>::bytes() : ranks == 8 ? Mega<8, 128, 3>::bytes() : 0;
    if (prefillCapacity) *prefillCapacity = bytes ? Mega<8, 128, 3>::PREFILL_CAP : 0;
    return bytes;
}
extern "C" int fastllm_mxfp4_ep_init(void *p, size_t bytes, const int *devices, int rank, int ranks) {
    try {
        // Initialization only clears the arena. Layer creation checks its exact geometry.
        const size_t required = fastllm_mxfp4_ep_draft_workspace(ranks, nullptr);
        int device; check(cudaGetDevice(&device));
        if (!required || !p || bytes < required || rank < 0 || rank >= ranks || !devices || devices[rank] != device)
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
extern "C" void *fastllm_mxfp4_ep_create_shape(const void *const *gate, const void *const *down,
        void *workspace, size_t bytes, const void *const *peers, int rank, int ranks, int experts, int topk) {
    try {
        if (experts == 128 && topk == 3 && ranks == 4) return Mega<4, 128, 3>::create(gate, down, workspace, bytes, peers, rank);
        if (experts == 128 && topk == 3 && ranks == 8) return Mega<8, 128, 3>::create(gate, down, workspace, bytes, peers, rank);
        if (experts != 256 || topk != 8) throw std::runtime_error("Unsupported SM90 MXFP4 expert geometry");
        if (ranks == 4) return Mega<4>::create(gate, down, workspace, bytes, peers, rank);
        if (ranks == 8) return Mega<8>::create(gate, down, workspace, bytes, peers, rank);
        throw std::runtime_error("MXFP4 EP requires four or eight ranks");
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
extern "C" void *fastllm_mxfp4_ep_create(const void *const *gate, const void *const *down,
        void *workspace, size_t bytes, const void *const *peers, int rank, int ranks) {
    return fastllm_mxfp4_ep_create_shape(gate, down, workspace, bytes, peers, rank, ranks, 256, 8);
}
extern "C" int fastllm_mxfp4_tp8_run_routed(void *handle, const void *x, void *y,
        const float *probabilities, const float *bias, int rows) {
    try {
        if (!handle || !x || !y || !probabilities || !bias || rows < 1 || rows > kFastllmMxfp4LocalRoutedMaxRows)
            throw std::runtime_error("Invalid TP-local routed MegaMoE inputs");
        auto &l = *static_cast<LocalTPMega::Layer *>(handle);
        int device; check(cudaGetDevice(&device));
        if (device != l.device) throw std::runtime_error("MXFP4 TP device mismatch");
        const auto *input = static_cast<const __nv_bfloat16 *>(x);
        if (rows <= kFastllmMxfp4LocalMaxRows) {
            LocalTPMega::runPlan<DECODE_CAP, true>(l, x, y, nullptr, probabilities, rows, rows, nullptr,
                LocalTPMega::VerifyInput<8>{input, probabilities, bias});
        } else if (rows <= 16) {
            LocalTPMega::runPlan<DECODE_CAP>(l, x, y, nullptr, probabilities, rows, rows, nullptr,
                SelectLocalRoutes<16>{bias});
        } else {
            LocalTPMega::runPlan<DECODE_CAP, true>(l, x, y, nullptr, probabilities, rows, rows, nullptr,
                LocalTPMega::VerifyInput<32>{input, probabilities, bias});
        }
        if (rows <= 16) fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::VerifyExperts);
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" int fastllm_mxfp4_ep_run(void *handle, const void *x, void *y, const int32_t *idx,
        const float *scores, int rows, int commonRows, const int *live) {
    try {
        if (!handle) throw std::runtime_error("Missing MXFP4 EP handle");
        static_cast<Base *>(handle)->forward(x, y, idx, scores, rows, commonRows, live);
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
extern "C" void fastllm_mxfp4_ep_destroy(void *handle) {
    if (!handle) return;
    int device; cudaGetDevice(&device); auto *p = static_cast<Base *>(handle);
    cudaSetDevice(p->device); delete p; cudaSetDevice(device);
}

extern "C" int fastllm_mxfp4_ep_set_map(void *handle, const uint8_t *table) {
    try {
        if (!handle || !table) throw std::runtime_error("Missing MXFP4 EP placement.");
        auto *layer = dynamic_cast<Mega<8>::Layer *>(static_cast<Base *>(handle));
        if (!layer) throw std::runtime_error("Expert placement requires a 256-expert EP8 layer.");
        int device;
        check(cudaGetDevice(&device));
        if (device != layer->device) throw std::runtime_error("MXFP4 EP device mismatch");
        if (!layer->expertMap.p) layer->expertMap.allocate(256);
        check(cudaMemcpy(layer->expertMap.p, table, 256, cudaMemcpyHostToDevice));
        return 1;
    } catch (const std::exception &e) { error = e.what(); return 0; }
}
