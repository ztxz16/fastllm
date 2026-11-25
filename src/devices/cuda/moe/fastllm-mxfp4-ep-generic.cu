// SM120 complete-expert EP. The grouped MXFP8 x MXFP4 MMA is built from the
// vendored DeepGEMM headers (MIT, third_party/deep_gemm). The P2P exchange,
// routing and workspace layout below are independent of a particular GPU SKU.
#include "devices/cuda/fastllm-mxfp4-ep-generic.h"
#include "devices/cuda/fastllm-cuda-router-topk.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using BF = __nv_bfloat16;
constexpr int Tile = 32, DecodeTile = 16;
constexpr int WideTile = 64;
constexpr int PrefillChunk = 1024;
constexpr int MinOwnerCombineRows = 384;
constexpr int CompactPrefillLimit = 512;
constexpr int CompactGateBlocks = 76, CompactDownBlocks = 34;
constexpr int GateBlocks = 72, EarlyDownBlocks = 38;

struct PrefillTile {
    unsigned completedColumns;
    enum Pass : unsigned { Unassigned, Early, Deferred };
    unsigned pass;
};

// The early FC2 may run concurrently with FC1. If a row tile takes too long,
// all its producer and math warps leave it for the ordered second pass.
// Each kernel can finish even when the driver serializes the two streams.
struct PrefillSelection {
    PrefillTile *tiles = nullptr;
    unsigned gateColumns = 0;
    bool deferred = false;

    __device__ bool accept(unsigned m) const {
        using Atomic = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;
        Atomic pass(tiles[m].pass);
        unsigned selected = pass.load(cuda::memory_order_acquire);
        if (deferred) return selected != PrefillTile::Early;
        if (selected == PrefillTile::Unassigned) {
            Atomic completed(tiles[m].completedColumns);
            unsigned proposed = PrefillTile::Deferred;
            for (int poll = 0; poll < 512; ++poll) {
                if (completed.load(cuda::memory_order_acquire) == gateColumns) {
                    proposed = PrefillTile::Early;
                    break;
                }
                __nanosleep(64);
            }
            if (pass.compare_exchange_strong(selected, proposed, cuda::memory_order_acq_rel,
                                             cuda::memory_order_acquire))
                selected = proposed;
        }
        return selected == PrefillTile::Early;
    }
};
template <bool Notify>
struct QuantizedSwiGLU;

// FC2 uses a wider N tile for long batches. FC1 stages only the activated half
// of its output tile. The shared-memory request must also include aligned SFA.
template <class T, int M, bool Pipelined = (M == WideTile)>
struct TileConfig {
    static constexpr bool wideDown = Pipelined && std::is_same_v<T, cutlass::bfloat16_t>;
    static constexpr int n = wideDown ? 256 : 128;
    static constexpr bool fourStages = M == DecodeTile || (Pipelined && M == Tile);
    static constexpr int stages = wideDown ? 2 : fourStages ? 4 : 3;
    static constexpr int epilogueRows = M == WideTile ? 32 : M;
    static constexpr int scaleA = M > 32 ? M * 4 : 128;
    static constexpr int shared = epilogueRows * n * sizeof(T) +
        stages * ((M + n) * 128 + scaleA + n * 4 + 16);
};
template <class T, int M = Tile, bool Pipelined = (M == WideTile)>
constexpr int Smem = TileConfig<T, M, Pipelined>::shared;
template <class T, int M = Tile, bool FuseGate = false, bool Pipelined = (M == WideTile)>
constexpr auto gemm =
    &deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0, 0, 0, 32, 32, 1, M, TileConfig<T, M, Pipelined>::n, 128, 128, 128, 128,
                                             TileConfig<T, M, Pipelined>::stages, M == DecodeTile ? 32 : 128, 128, 0,
                                             deep_gemm::GemmType::MGroupedContiguous, false, T,
                                             deep_gemm::epilogue::transform::EpilogueIdentity, false, true,
                                             false, true, false, TileConfig<T, M, Pipelined>::epilogueRows, 1, true,
                                             std::is_same_v<T, cutlass::bfloat16_t>,
                                             std::conditional_t<FuseGate, QuantizedSwiGLU<Pipelined>, void>,
                                             std::conditional_t<Pipelined, PrefillSelection, void>>;

void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string("SM120 generic EP: ") + cudaGetErrorString(s));
}
struct Restore {
    int device;
    Restore() { check(cudaGetDevice(&device)); }
    ~Restore() { cudaSetDevice(device); }
};
struct Storage {
    int device = -1;
    std::vector<void *> blocks;
    Storage() = default;
    Storage(const Storage &) = delete;
    ~Storage() {
        int prev;
        if (device < 0 || cudaGetDevice(&prev) != cudaSuccess)
            return;
        cudaSetDevice(device);
        for (void *p : blocks)
            cudaFree(p);
        cudaSetDevice(prev);
    }
    template <class T> T *allocate(size_t count) {
        void *p;
        blocks.reserve(blocks.size() + 1);
        check(cudaMalloc(&p, count * sizeof(T)));
        blocks.push_back(p);
        return static_cast<T *>(p);
    }
};
struct Shape {
    int h, i, ranks, localE, k, cap, stride, pool, inputPitch;
};
struct Mail {
    uint8_t *send, *receive;
    BF *returned;
    unsigned long long *flags;

    // Dense EP8 appends a word per expert rank after the eight arrivals and the epoch.
    __host__ __device__ unsigned long long *directShares() const { return flags + 9; }
};
struct Buffers {
    uint8_t *incoming, *a;
    uint32_t *sa;
    float *gateUp = nullptr;
    cutlass::bfloat16_t **returnRows;
    int *counts, *bases, *positions, *layout;
    int *compactLayout = nullptr;
    uint8_t *ownerSlots = nullptr;
    int2 *tasks;
};
struct DenseExchange {
    std::array<uint8_t *, 8> receive{};
    Mail *peers = nullptr;
};
struct Rank {
    Storage storage;
    cudaStream_t prefillStream = nullptr;
    cudaEvent_t prefillEvent = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaEvent_t copyStart = nullptr, copyDone = nullptr;
    DenseExchange prefetched;
    ~Rank() {
        if (!prefillStream && !prefillEvent) return;
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(storage.device) != cudaSuccess) return;
        if (copyStart) cudaEventDestroy(copyStart);
        if (copyDone) cudaEventDestroy(copyDone);
        if (copyStream) cudaStreamDestroy(copyStream);
        if (prefillEvent) cudaEventDestroy(prefillEvent);
        if (prefillStream) cudaStreamDestroy(prefillStream);
        cudaSetDevice(previous);
    }
    PrefillTile *tiles = nullptr;
    int sms;
    bool tunedEp8;
    bool compactPrefill = false;
    bool relayAcrossNuma = false;
    DenseExchange current;
    Mail *pairedReturnPeers = nullptr;
    float *rowScores = nullptr;
    uint8_t *downInput;
    uint32_t *downScales;
    Mail mail;
    Mail *peers;
    Buffers b;
    struct Projection {
        cute::TmaDescriptor a, b, sa, sb, d, decodeA, decodeSa, decodeD, wideA, wideSa, wideB, wideSb;
    };
    struct Layer {
        Storage storage;
        Projection gate, down;
    };
    std::map<int, Layer> layers;
};

// Publish the preceding stream work and await all peers. Only the owner writes
// its epoch; each sender has an independent mailbox on every receiver.
__device__ void joinPeers(Mail *peers, int rank, int ranks) {
    __shared__ unsigned long long next;
    if (threadIdx.x == 0)
        next = ++peers[rank].flags[ranks];
    __syncthreads();
    for (int r = threadIdx.x; r < ranks; r += blockDim.x) {
        asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(peers[r].flags + rank), "l"(next)
                     : "memory");
    }
    __syncthreads();
    // Fail a broken peer collective instead of leaving every serving rank spinning forever.
    constexpr unsigned long long peerTimeoutNs = 60'000'000'000ULL;
    unsigned long long started;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
    for (int r = threadIdx.x; r < ranks; r += blockDim.x) {
        unsigned long long seen;
        do {
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                         : "=l"(seen)
                         : "l"(peers[rank].flags + r)
                         : "memory");
            if (seen < next) {
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (now - started > peerTimeoutNs)
                    asm volatile("trap;");
            }
        } while (seen < next);
    }
}
__global__ void rendezvous(Mail *peers, int rank, int ranks) { joinPeers(peers, rank, ranks); }


// Eight adjacent lanes quantize one E8M0-scaled block of 32 FP8 values.
template <bool Fc1Output = false>
__device__ uint32_t quantizeFour(float4 values, uint8_t &scaleByte) {
    float amax = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)), fmaxf(fabsf(values.z), fabsf(values.w)));
    #pragma unroll
    for (int delta = 4; delta > 0; delta /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta, 8));
    // Input staging floors its scale. The fused FC1 epilogue uses the actual block maximum.
    const int bits = __float_as_int(Fc1Output ? amax * (1.f / 448.f) : fmaxf(amax, 1e-4f) / 448.f);
    scaleByte = (bits >> 23) + ((bits & 0x7fffff) != 0);
    const float invScale = __uint_as_float((254 - scaleByte) << 23);
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(values.x * invScale, values.y * invScale),
                                                 __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(values.z * invScale, values.w * invScale),
                                                 __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}
// Neighboring N fragments contain eight gate columns and their eight up columns.
template <bool Notify>
struct QuantizedSwiGLU {
    uint8_t *data;
    uint8_t *scales;
    float *scores;
    int pool;

    PrefillTile *tiles = nullptr;

    __device__ static float activate(float gate, float up, float score) {
        if constexpr (Notify) {
            // Match the packaged FC1 arithmetic before converting the result to MXFP8.
            float exponent, reciprocal;
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponent) : "f"(gate * -1.4426950408889634f));
            const float denominator = 1.f + exponent;
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denominator));
            return (up * gate) * reciprocal * score;
        }
        return (gate / (1.f + __expf(-gate))) * up * score;
    }

    template <int M, int N, int MT, int NT, int Threads>
    __device__ void store(float *tile, int outputColumns, int mBase, int nBase, int mTile, int nTile,
                          int lane, const float *accum) const {
        constexpr int chunksPerRow = N / 2 / 4;
        const int inter = outputColumns / 2;
        static_assert(N == 128 && NT % 2 == 0);
        cutlass::arch::NamedBarrier::sync(Threads, 0);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row = (mTile + mt) * 16 + lane / 4;
            const float score0 = scores[mBase + row], score1 = scores[mBase + row + 8];
            #pragma unroll
            for (int nt = 0; nt < NT; nt += 2) {
                const float *g = accum + (mt * NT + nt) * 4;
                const float *u = g + 4;
                const int col = (nTile + nt) / 2 * 8 + (lane % 4) * 2;
                *reinterpret_cast<float2 *>(tile + row * 64 + col) = make_float2(
                    activate(g[0], u[0], score0), activate(g[1], u[1], score0));
                *reinterpret_cast<float2 *>(tile + (row + 8) * 64 + col) = make_float2(
                    activate(g[2], u[2], score1), activate(g[3], u[3], score1));
            }
        }
        cutlass::arch::NamedBarrier::sync(Threads, 0);
        for (int idx = threadIdx.x; idx < M * chunksPerRow; idx += Threads) {
            int row = mBase + idx / chunksPerRow, col = nBase / 2 + (idx % chunksPerRow) * 4;
            uint8_t sf;
            auto quant = quantizeFour<Notify>(reinterpret_cast<const float4 *>(tile)[idx], sf);
            *reinterpret_cast<uint32_t *>(data + size_t(row) * inter + col) = quant;
            if (!(col % 32)) scales[(size_t(col / 128) * pool + row) * 4 + col % 128 / 32] = sf;
        }
        if constexpr (Notify) {
            // Publish every writer's quantized data and scales to subsequent TMA reads.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            cutlass::arch::NamedBarrier::sync(Threads, 0);
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_device> done(tiles[mBase / M].completedColumns);
                done.fetch_add(1, cuda::memory_order_acq_rel);
            }
        }
    }
};
__device__ void packInputValues(uint8_t *p, const BF *input, int row, int hidden,
                                bool active, int first, int step) {
    for (int chunk = first; chunk < hidden / 4; chunk += step) {
        const auto *chunks = active ? reinterpret_cast<const uint2 *>(input + size_t(row) * hidden) : nullptr;
        const uint2 raw = active ? chunks[chunk] : make_uint2(0, 0);
        const float4 values = make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                                          __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
        uint8_t scale;
        reinterpret_cast<uint32_t *>(p)[chunk] = quantizeFour(values, scale);
        if (chunk % 8 == 0) p[hidden + chunk / 8] = scale;
    }
}
__device__ void packInputRow(uint8_t *p, const BF *input, const int *ids, const float *scores,
                             int row, int hidden, int topk, bool active) {
    int *routes = reinterpret_cast<int *>(p + hidden + hidden / 32);
    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        routes[j] = active ? ids[row * topk + j] : -1;
        reinterpret_cast<float *>(routes + topk)[j] = active ? scores[row * topk + j] : 0.f;
    }
    packInputValues(p, input, row, hidden, active, threadIdx.x, blockDim.x);
}
__global__ void publish(Mail mail, Shape s, const BF *input, const int *ids, const float *scores, int rows,
                        const int *live) {
    const int row = blockIdx.x;
    packInputRow(mail.send + size_t(row) * s.stride, input, ids, scores, row, s.h, s.k,
                 row < rows && (!live || live[row] > 0));
}
// Publish each activation once per selected owner. Empty owners only need
// the routing metadata to discard their slots. PCIe writes avoid remote reads.
__global__ void dispatch(Mail *peers, Shape s, int rank, int rows) {
    const int row = blockIdx.x;
    // Start each sender at a different peer to spread simultaneous PCIe writes.
    int owner = rank + blockIdx.y;
    if (owner >= s.ranks) owner -= s.ranks;
    const auto *packet = peers[rank].send + size_t(row) * s.stride;
    const auto *ids = reinterpret_cast<const int *>(packet + s.h + s.h / 32);
    bool hit = false;
    for (int j = threadIdx.x; j < s.k; j += blockDim.x) {
        int e = ids[j];
        hit |= e >= owner * s.localE && e < (owner + 1) * s.localE;
    }
    int first = __syncthreads_or(hit) ? 0 : (s.h + s.h / 32) / 16;
    auto *dst = reinterpret_cast<uint4 *>(peers[owner].receive + size_t(rank * rows + row) * s.stride);
    const auto *src = reinterpret_cast<const uint4 *>(packet);
    for (int j = first + threadIdx.x; j < s.stride / 16; j += blockDim.x)
        dst[j] = src[j];
}
constexpr unsigned Ep8MetadataOffset = 4096 + 4096 / 32;
constexpr unsigned Ep8PacketBytes = Ep8MetadataOffset + 8 * (sizeof(int) + sizeof(float));

__device__ void forwardEp8Packet(const uint8_t *packet, Mail *peers, int destination, int owners,
                                 int source, int row, int capacity) {
    const auto *ids = reinterpret_cast<const int *>(packet + Ep8MetadataOffset);
    const int firstExpert = (destination & ~(owners - 1)) * 32;
    bool selected = false;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        selected |= static_cast<unsigned>(ids[i] - firstExpert) < owners * 32;
    const unsigned offset = selected ? 0 : Ep8MetadataOffset;
    auto *dst = peers[destination].receive + (size_t(source) * capacity + row) * Ep8PacketBytes + offset;
    const unsigned address = __cvta_generic_to_shared(packet + offset);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                 "l"(dst), "r"(address), "r"(Ep8PacketBytes - offset) : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
}

// Four peers share one quantized packet. A peer with no routes only needs its metadata.
constexpr int DispatchPeers = 4;
__global__ void dispatchDecodeRows(Mail *peers, int rank, const BF *input, const int *ids, const float *scores,
                                   int rows, int cap, const int *live) {
    constexpr int hidden = 4096, topk = 8, ranks = 8;
    __shared__ __align__(16) uint8_t packet[Ep8PacketBytes];
    const int row = blockIdx.x;
    packInputRow(packet, input, ids, scores, row, hidden, topk, row < rows && (!live || live[row] > 0));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < DispatchPeers) {
        const int owner = (rank + blockIdx.y * DispatchPeers + threadIdx.x) % ranks;
        forwardEp8Packet(packet, peers, owner, 1, rank, row, cap);
    }
}
struct DecodeSelection {
    const float *probabilities = nullptr, *bias = nullptr;
    int *ids = nullptr;
    bool normalize = false;
    float scale = 1.f;
    bool logits = false;
};

// Warp zero selects the exact TopK while the other warps quantize the input.
// Each peer group needs the whole packet; only group zero publishes IDs for combine.
__global__ void dispatchRoutedRow(Mail *peers, int rank, const BF *input, DecodeSelection selection,
                                  const int *live, int rows, int capacity) {
    __shared__ __align__(16) uint8_t packet[Ep8PacketBytes];
    __shared__ fastllm::cuda::RouterTopKScratch<8> scratch;
    __shared__ __align__(16) float probabilities[256];
    const int row = blockIdx.x;
    const bool active = row < rows && live[row] > 0;
    const int t = threadIdx.x;
    auto *ids = reinterpret_cast<int *>(packet + Ep8MetadataOffset);
    auto *scores = reinterpret_cast<float *>(ids + 8);
    if (t < 32) {
        if (active) {
            const float *source = selection.probabilities + size_t(row) * 256;
            if (selection.logits) {
                for (int expert = t; expert < 256; expert += 32)
                    probabilities[expert] = 1.0 / (1.0 + expf(-source[expert]));
                __syncwarp();
                source = probabilities;
            }
            fastllm::cuda::SelectRouterTopK<256, 8>(source, selection.bias,
                ids, scores, 1, selection.normalize, selection.scale, scratch);
        } else if (t < 8) {
            ids[t] = -1;
            scores[t] = 0.f;
        }
        __syncwarp();
        if (!blockIdx.y && t < 8) selection.ids[row * 8 + t] = ids[t];
    } else {
        packInputValues(packet, input, row, 4096, active, t - 32, blockDim.x - 32);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (t < DispatchPeers)
        forwardEp8Packet(packet, peers, (rank + blockIdx.y * DispatchPeers + t) % 8, 1, rank, row, capacity);
}
__global__ void dispatchEp8ToNumaPeers(Mail *peers, int rank, const BF *input, const int *ids,
                                       const float *scores, int rows, int cap, const int *live) {
    __shared__ __align__(16) uint8_t packet[Ep8PacketBytes];
    const int row = blockIdx.x;
    packInputRow(packet, input, ids, scores, row, 4096, 8, row < rows && (!live || live[row] > 0));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const int lane = threadIdx.x;
    if (lane < 4) {
        forwardEp8Packet(packet, peers, (rank & 4) + ((rank + lane) % 4), 1, rank, row, cap);
    } else if (lane < 6) {
        const int relay = ((rank ^ 4) & 4) + 2 * (lane - 4) + rank % 2;
        forwardEp8Packet(packet, peers, relay, 2, rank, row, cap);
    }
}

__global__ void forwardEp8WithinNumaNode(Mail *peers, int rank, int cap) {
    __shared__ __align__(16) uint8_t packet[Ep8PacketBytes];
    const int row = blockIdx.x;
    const int source = ((rank ^ 4) & 4) + 2 * blockIdx.y + rank % 2;
    const auto *src = peers[rank].receive + (size_t(source) * cap + row) * Ep8PacketBytes;
    for (int i = threadIdx.x; i < Ep8PacketBytes / 16; i += blockDim.x)
        reinterpret_cast<uint4 *>(packet)[i] = reinterpret_cast<const uint4 *>(src)[i];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) forwardEp8Packet(packet, peers, rank ^ 1, 1, source, row, cap);
}
__device__ int expert(Buffers b, Shape s, int route) {
    const uint8_t *p = b.incoming + size_t(route / s.k) * s.stride + s.h + s.h / 32;
    return reinterpret_cast<const int *>(p)[route % s.k];
}
__global__ void countRoutes(Buffers b, Shape s, int rank, int routes) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= routes)
        return;
    int e = expert(b, s, t) - rank * s.localE;
    b.positions[t] = (unsigned)e < (unsigned)s.localE ? atomicAdd(b.counts + e, 1) : -1;
}
template <int M, bool WithScores, bool BalanceReturns = false>
__global__ void layout(Buffers b, Shape s, float *rowScores, Mail *peers = nullptr, int rank = 0, int cap = 0) {
    int e = blockIdx.x;
    int alignment = M;
    if constexpr (BalanceReturns) {
        // This EP8 specialization launches one warp per local expert.
        static_assert(M == Tile);
        int count = b.counts[threadIdx.x];
        int total = count;
        for (int step = 16; step; step /= 2)
            total += __shfl_xor_sync(0xffffffffu, total, step);
        // A heavily loaded rank amortizes 64-row tiles and benefits from sending
        // one reduced vector per remote owner. Other ranks can still use 32 rows.
        bool overloaded = total > 2 * cap * s.k;
        alignment = overloaded ? WideTile : Tile;
        int *inactive = overloaded ? b.compactLayout : b.layout;
        if (!overloaded) b.layout = b.compactLayout;
        if (e == 0) {
            bool hot = __any_sync(0xffffffffu, count > cap / 2);
            if (threadIdx.x == 0) inactive[-1] = 0;
            if (threadIdx.x < 8) peers[threadIdx.x].directShares()[rank] = overloaded ? 0 : hot ? 2 : 4;
        }
    }
    int base = 0, compact = 0;
    for (int j = 0; j < e; ++j) {
        base += (b.counts[j] + alignment - 1) / alignment * alignment;
        compact += b.counts[j];
    }
    const int padded = (b.counts[e] + alignment - 1) / alignment * alignment;
    for (int j = b.counts[e] + threadIdx.x; j < padded; j += blockDim.x) {
        b.returnRows[base + j] = nullptr;
        if constexpr (WithScores) rowScores[base + j] = 0.f;
    }
    if (!threadIdx.x) {
        b.bases[e] = base;
        b.bases[s.localE + e] = compact;
        if (e + 1 == s.localE) {
            b.counts[s.localE] = compact + b.counts[e];
            b.layout[-1] = base + padded;
        }
    }
    for (int j = threadIdx.x * alignment; j < b.counts[e]; j += blockDim.x * alignment)
        b.layout[base + j] = e;
}
// Spread direct FC2 writes across destination rows and expert ranks. Both sides
// use the same bucket so the requester preserves the per-owner BF16 rounding.
__device__ bool directReturn(unsigned row, unsigned owner, unsigned share) {
    return ((row + 3 * owner) & 7) < share;
}
template <bool WithScores, bool OwnerReturn = false, bool BalanceReturns = false>
__global__ void makeTasks(Buffers b, Shape s, int rank, int routes, const Mail *peers, int rowsPerPeer,
                          float *rowScores, bool crossSocketOnly = false, const unsigned long long *shares = nullptr) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = t < routes ? b.positions[t] : -1;
    unsigned slots = 0;
    if constexpr (OwnerReturn) {
        // Each group of eight lanes handles one token's top-8 slots.
        slots = (__ballot_sync(0xffffffffu, offset >= 0) >> (threadIdx.x & 24)) & 0xff;
        // Dense dispatch interleaves the two sockets in consecutive source slots.
        if (crossSocketOnly && ((t / (rowsPerPeer * 8)) & 1) == rank / 4) slots = 0;
        if constexpr (BalanceReturns) {
            if (directReturn((t / 8) % rowsPerPeer, rank, shares[rank])) slots = 0;
        }
        if (t < routes && !(t % 8)) b.ownerSlots[t / 8] = slots;
    }
    if (offset < 0) return;
    const int e = expert(b, s, t) - rank * s.localE;
    const int row = b.bases[e] + offset;
    const int slotsPerPeer = rowsPerPeer * s.k;
    auto *mailbox = peers[t / slotsPerPeer].returned + size_t(t % slotsPerPeer) * s.h;
    if constexpr (OwnerReturn) {
        if (slots & (slots - 1)) {
            mailbox = reinterpret_cast<BF *>(b.a) + size_t(row) * s.h;
            b.positions[t] = row;
        }
    }
    if constexpr (WithScores) {
        const auto *packet = b.incoming + size_t(t / s.k) * s.stride;
        const auto *scores = reinterpret_cast<const float *>(packet + s.h + s.h / 32 + s.k * sizeof(int));
        rowScores[row] = scores[t % s.k];
    }
    b.returnRows[row] = reinterpret_cast<cutlass::bfloat16_t *>(mailbox);
    b.tasks[b.bases[s.localE + e] + offset] = make_int2(t, row);
}
// The complete routing table for four rows/rank fits in a single CTA (8 ranks x top-8).
template<int Slots> __device__ void planRoutes(Buffers b, Shape s, int rank, int cap, const Mail *peers, float *rowScores) {
    __shared__ int counts[32], bases[32], compact[32];
    int t = threadIdx.x;
    if (t < 32) counts[t] = 0;
    __syncthreads();
    int owners[Slots], offsets[Slots];
#pragma unroll
    for (int i = 0; i < Slots; ++i) {
        const int slot = t + i * blockDim.x;
        const int e = slot < cap * 64 ? expert(b, s, slot) - rank * 32 : -1;
        owners[i] = e;
        offsets[i] = unsigned(e) < 32 ? atomicAdd(counts + e, 1) : -1;
    }
    __syncthreads();
    if (t < 32) {
        const int count = counts[t];
        const int padded = (count + DecodeTile - 1) / DecodeTile * DecodeTile;
        int sum = count, total = padded;
        for (int d = 1; d < 32; d *= 2) {
            int c = __shfl_up_sync(0xffffffff, sum, d), p = __shfl_up_sync(0xffffffff, total, d);
            if (t >= d) { sum += c; total += p; }
        }
        bases[t] = total - padded;
        compact[t] = sum - count;
        if (t == 31) { b.layout[-1] = total; b.counts[32] = sum; }
    }
    __syncthreads();
    for (int group = t / 32; group < 32; group += blockDim.x / 32) {
        const int base = bases[group], count = counts[group], lane = t % 32;
        const int padded = (count + DecodeTile - 1) / DecodeTile * DecodeTile;
        for (int j = lane * DecodeTile; j < count; j += 32 * DecodeTile) b.layout[base + j] = group;
        if (count + lane < padded) {
            b.returnRows[base + count + lane] = nullptr;
            rowScores[base + count + lane] = 0.f;
        }
    }
#pragma unroll
    for (int i = 0; i < Slots; ++i) {
        const int offset = offsets[i], e = owners[i], slot = t + i * blockDim.x;
        if (offset < 0) continue;
        const int row = bases[e] + offset, slots = cap * 8;
        b.returnRows[row] = reinterpret_cast<cutlass::bfloat16_t *>(peers[slot / slots].returned + size_t(slot % slots) * s.h);
        const auto *packet = b.incoming + size_t(slot / 8) * s.stride;
        rowScores[row] = reinterpret_cast<const float *>(packet + s.h + s.h / 32 + 8 * sizeof(int))[slot % 8];
        b.tasks[compact[e] + offset] = make_int2(slot, row);
    }
}
__global__ void planTinyRoutes(Buffers b, Shape s, int rank, int cap, const Mail *peers, float *rowScores) {
    planRoutes<1>(b, s, rank, cap, peers, rowScores);
}

__device__ void gatherRows(Buffers b, Shape s) {
    int count = b.counts[s.localE];
    for (int job = blockIdx.x; job < count; job += gridDim.x) {
        int2 task = b.tasks[job];
        auto *p = b.incoming + size_t(task.x / s.k) * s.stride;
        for (int j = threadIdx.x; j < s.h / 16; j += blockDim.x)
            reinterpret_cast<uint4 *>(b.a + size_t(task.y) * s.inputPitch)[j] = reinterpret_cast<const uint4 *>(p)[j];
        for (int j = threadIdx.x; j < s.h / 128; j += blockDim.x)
            b.sa[size_t(j) * s.pool + task.y] = reinterpret_cast<const uint32_t *>(p + s.h)[j];
    }
}
__global__ void gather(Buffers b, Shape s) { gatherRows(b, s); }

__global__ void activate(Buffers b, Shape s) {
    int count = b.counts[s.localE];
    for (int job = blockIdx.x; job < count; job += gridDim.x) {
        int2 task = b.tasks[job];
        const auto *meta = b.incoming + size_t(task.x / s.k) * s.stride + s.h + s.h / 32 + s.k * sizeof(int);
        float score = reinterpret_cast<const float *>(meta)[task.x % s.k];
        const float *gate = b.gateUp + size_t(task.y) * 2 * s.i;
        auto *activation = reinterpret_cast<uint32_t *>(b.a + size_t(task.y) * s.i);
        for (int chunk = threadIdx.x; chunk < s.i / 4; chunk += blockDim.x) {
            const float4 g = reinterpret_cast<const float4 *>(gate)[chunk];
            const float4 u = reinterpret_cast<const float4 *>(gate + s.i)[chunk];
            const float4 values = make_float4((g.x / (1.f + __expf(-g.x))) * u.x * score,
                                              (g.y / (1.f + __expf(-g.y))) * u.y * score,
                                              (g.z / (1.f + __expf(-g.z))) * u.z * score,
                                              (g.w / (1.f + __expf(-g.w))) * u.w * score);
            uint8_t scale;
            activation[chunk] = quantizeFour(values, scale);
            if (chunk % 8 == 0)
                reinterpret_cast<uint8_t *>(b.sa)[(size_t(chunk / 32) * s.pool + task.y) * 4 + chunk % 32 / 8] = scale;
        }
    }
}
// Single-hit tokens already went straight to their destination from FC2. For repeated
// hits, sum the BF16 expert rows in FP32 and send one BF16 vector in the first slot.
__global__ void publishOwnerRows(Buffers b, const Mail *peers, int cap) {
    constexpr int hidden = 4096, width = sizeof(uint4) / sizeof(BF);
    __shared__ __align__(16) uint4 packet[hidden / width];
    const int token = blockIdx.x;
    const unsigned slots = b.ownerSlots[token];
    if (!(slots & (slots - 1))) return;
    const auto *positions = b.positions + token * 8;
    const auto *rows = reinterpret_cast<const uint4 *>(b.a);
    for (int col = threadIdx.x; col < hidden / width; col += blockDim.x) {
        float sums[width] = {};
        for (unsigned remaining = slots; remaining; remaining &= remaining - 1) {
            const int slot = __ffs(remaining) - 1;
            const uint4 value = rows[size_t(positions[slot]) * (hidden / width) + col];
            const auto *elements = reinterpret_cast<const BF *>(&value);
            #pragma unroll
            for (int i = 0; i < width; ++i) sums[i] += __bfloat162float(elements[i]);
        }
        uint4 packed;
        auto *elements = reinterpret_cast<BF *>(&packed);
        #pragma unroll
        for (int i = 0; i < width; ++i) elements[i] = __float2bfloat16(sums[i]);
        packet[col] = packed;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        const int first = __ffs(slots) - 1;
        auto *dst = peers[token / cap].returned + (size_t(token % cap) * 8 + first) * hidden;
        const unsigned addr = __cvta_generic_to_shared(packet);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                     "l"(dst), "r"(addr), "r"(unsigned(sizeof(packet))) : "memory");
        asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
    }
}
// Round each owner partial to BF16, then accumulate owners in rank order,
// as in the packaged prefill reduction. An owner can also send its partial directly.
enum class OwnerReturn { Direct, CrossSocket, Reduced };
template <bool BalanceReturns = false, OwnerReturn Mode = OwnerReturn::CrossSocket>
__global__ void combineDenseEp8Rows(Mail mail, const int *ids, BF *out, int rank) {
    constexpr int topk = 8, hidden = 4096, width = sizeof(uint4) / sizeof(BF);
    const int row = blockIdx.x;
    __shared__ uint8_t groups[topk];
    if (threadIdx.x < topk) groups[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x < topk) {
        const unsigned slot = threadIdx.x;
        const unsigned expertId = ids[row * topk + slot];
        const unsigned owner = expertId / 32;
        const unsigned mask = __match_any_sync(0xffu, owner);
        const unsigned first = __ffs(mask) - 1;
        bool direct = Mode == OwnerReturn::Direct || (Mode == OwnerReturn::CrossSocket && owner / 4 == unsigned(rank / 4));
        if constexpr (BalanceReturns) {
            if (owner < 8) direct |= directReturn(row, owner, mail.directShares()[owner]);
        }
        const unsigned contributions = direct ? mask : 1u << first;
        if (expertId < 256 && slot == first) groups[owner] = contributions;
    }
    __syncthreads();
    const auto *rows = reinterpret_cast<const uint4 *>(mail.returned) + size_t(row) * topk * (hidden / width);
    for (int col = threadIdx.x; col < hidden / width; col += blockDim.x) {
        float total[width] = {};
        for (int owner = 0; owner < topk; ++owner) {
            const unsigned slots = groups[owner];
            if (!slots) continue;
            const int first = __ffs(slots) - 1;
            const uint4 initial = rows[first * (hidden / width) + col];
            float partial[width];
            #pragma unroll
            for (int k = 0; k < width; ++k)
                partial[k] = __bfloat162float(reinterpret_cast<const BF *>(&initial)[k]);
            unsigned rest = slots & (slots - 1);
            if (rest) {
                do {
                    const int slot = __ffs(rest) - 1;
                    const uint4 value = rows[slot * (hidden / width) + col];
                    #pragma unroll
                    for (int k = 0; k < width; ++k)
                        partial[k] += __bfloat162float(reinterpret_cast<const BF *>(&value)[k]);
                    rest &= rest - 1;
                } while (rest);
                #pragma unroll
                for (int k = 0; k < width; ++k)
                    partial[k] = __bfloat162float(__float2bfloat16(partial[k]));
            }
            #pragma unroll
            for (int k = 0; k < width; ++k) total[k] += partial[k];
        }
        uint4 packet;
        #pragma unroll
        for (int k = 0; k < width; ++k) reinterpret_cast<BF *>(&packet)[k] = __float2bfloat16(total[k]);
        reinterpret_cast<uint4 *>(out)[size_t(row) * (hidden / width) + col] = packet;
    }
}
template <int FixedTopK = 0, bool OwnerPartial = false>
__device__ void combineRows(Mail local, Shape s, const int *ids, BF *out, const int *live, int row, int x, int groups) {
    const bool active = !live || live[row] > 0;
    const int topk = FixedTopK ? FixedTopK : s.k;
    constexpr int width = sizeof(uint4) / sizeof(BF);
    const int start = FixedTopK ? x * blockDim.x + threadIdx.x : threadIdx.x;
    const int step = FixedTopK ? groups * blockDim.x : blockDim.x;
    unsigned firstSlots = 0;
    if constexpr (OwnerPartial) {
        static_assert(FixedTopK == 8);
        unsigned seen = 0;
        for (int j = 0; active && j < FixedTopK; ++j) {
            unsigned expertId = ids[row * FixedTopK + j];
            if (expertId >= 256) continue;
            unsigned owner = 1u << (expertId / 32);
            if (!(seen & owner)) firstSlots |= 1u << j;
            seen |= owner;
        }
    }
    for (int chunk = start; chunk < s.h / width; chunk += step) {
        float sum[width] = {};
        if (active) {
            for (int j = 0; j < topk; ++j) {
                const int slot = row * topk + j;
                if constexpr (OwnerPartial) {
                    if (!(firstSlots & (1u << j))) continue;
                } else if (static_cast<unsigned>(ids[slot]) >= static_cast<unsigned>(s.ranks * s.localE)) continue;
                const auto *values = reinterpret_cast<const uint4 *>(local.returned + size_t(slot) * s.h);
                const uint4 packed = values[chunk];
                const uint32_t pairs[] = {packed.x, packed.y, packed.z, packed.w};
                #pragma unroll
                for (int k = 0; k < width / 2; ++k) {
                    sum[2 * k] += __uint_as_float(pairs[k] << 16);
                    sum[2 * k + 1] += __uint_as_float(pairs[k] & 0xffff0000u);
                }
            }
        }
        uint32_t pairs[width / 2];
        #pragma unroll
        for (int k = 0; k < width / 2; ++k) {
            const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16(sum[2 * k]));
            const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16(sum[2 * k + 1]));
            pairs[k] = lo | (hi << 16);
        }
        auto *dest = reinterpret_cast<uint4 *>(out + size_t(row) * s.h);
        dest[chunk] = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    }
}
template <int FixedTopK = 0, bool OwnerPartial = false>
__global__ void combine(Mail local, Shape s, const int *ids, BF *out, const int *live) {
    combineRows<FixedTopK, OwnerPartial>(local, s, ids, out, live, blockIdx.x, blockIdx.y, gridDim.y);
}

__global__ void packGatePairs(const uint8_t *src, uint8_t *dst, uint32_t *scales, int n, int k) {
    const int row = blockIdx.x;
    // Interleave eight gate rows and eight up rows so one warp owns both MMA fragments.
    const int srcRow = (row / 16) * 8 + row % 8 + ((row / 8) % 2) * (n / 2);
    if (dst) {
        auto *in = reinterpret_cast<const uint4 *>(src) + size_t(srcRow) * (k / 32);
        auto *out = reinterpret_cast<uint4 *>(dst) + size_t(row) * (k / 32);
        for (int j = threadIdx.x; j < k / 32; j += blockDim.x) out[j] = in[j];
    }
    auto *sf = reinterpret_cast<const uint32_t *>(src + size_t(n) * k / 2);
    for (int j = threadIdx.x; j < k / 128; j += blockDim.x)
        scales[size_t(j) * n + row] = sf[size_t(srcRow) * (k / 128) + j];
}
__global__ void packScales(const uint32_t *src, uint32_t *dst, int n, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n * cols)
        dst[j] = src[(j % n) * cols + j / n];
}
cute::TmaDescriptor tensor(void *p, CUtensorMapDataType type, size_t inner, size_t outer, size_t pitch,
                           unsigned bx, unsigned by, CUtensorMapSwizzle swizzle) {
    cute::TmaDescriptor out{};
    cuuint64_t dims[] = {inner, outer}, strides[] = {pitch};
    cuuint32_t box[] = {bx, by}, steps[] = {1, 1};
    auto rc = cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&out), type, 2, p, dims, strides, box,
                                     steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                     CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS)
        throw std::runtime_error("SM120 generic EP: invalid TMA layout (" + std::to_string(rc) + ")");
    return out;
}
// Build activation descriptors for this workspace and projection.
void bindActivation(Rank::Projection &p, const Rank &rank, const Shape &s, bool down, int k) {
    auto *a = down ? rank.downInput : rank.b.a;
    auto *scales = down ? rank.downScales : rank.b.sa;
    auto activation = [&](int m) {
        return tensor(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, s.pool, down ? k : s.inputPitch, 128, m, CU_TENSOR_MAP_SWIZZLE_128B);
    };
    auto scale = [&](int m) {
        return tensor(scales, CU_TENSOR_MAP_DATA_TYPE_INT32, s.pool, k / 128, s.pool * 4, m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    };
    p.a = activation(Tile);
    p.sa = scale(Tile);
    p.decodeA = activation(DecodeTile);
    p.decodeSa = scale(DecodeTile);
    if (s.cap > 128) {
        p.wideA = activation(WideTile);
        p.wideSa = scale(WideTile);
    }
}
Rank::Projection bindProjection(Storage &owner, const Rank &rank, Shape s, const std::vector<uint64_t> &ptrs,
                                int part, int n, int k, void *sharedWeight) {
    const auto &b = rank.b;
    size_t values = size_t(n) * k / 2, words = size_t(n) * k / 128;
    auto *w = sharedWeight ? static_cast<uint8_t *>(sharedWeight) : owner.allocate<uint8_t>(values * s.localE);
    auto *sf = owner.allocate<uint32_t>(words * s.localE);
    for (int e = 0; e < s.localE; ++e) {
        auto *src = reinterpret_cast<const uint8_t *>(ptrs[2 * e + part]);
        if (!part && rank.tunedEp8) {
            packGatePairs<<<n, 128, 0, cudaStreamPerThread>>>(src, sharedWeight ? nullptr : w + e * values, sf + e * words, n, k);
            continue;
        }
        if (!sharedWeight) check(cudaMemcpyAsync(w + e * values, src, values, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        packScales<<<(words + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const uint32_t *>(src + values), sf + e * words, n, k / 128);
    }
    auto sw = CU_TENSOR_MAP_SWIZZLE_128B;
    auto none = CU_TENSOR_MAP_SWIZZLE_NONE;
    auto output = [&](int m) {
        return (part || rank.tunedEp8) ? cute::TmaDescriptor{} : tensor(b.gateUp, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, n, s.pool, n * 4, 32, m, sw);
    };
    Rank::Projection p{};
    p.b = tensor(w, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, k, size_t(n) * s.localE, k / 2, 128, 128, sw);
    p.sb = tensor(sf, CU_TENSOR_MAP_DATA_TYPE_INT32, n, size_t(k / 128) * s.localE, n * 4, 128, 1, none);
    if (s.cap > 128 && part) {
        p.wideB = tensor(w, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, k, size_t(n) * s.localE, k / 2, 128, 256, sw);
        p.wideSb = tensor(sf, CU_TENSOR_MAP_DATA_TYPE_INT32, n, size_t(k / 128) * s.localE, n * 4, 256, 1, none);
    }
    p.d = output(Tile);
    p.decodeD = output(DecodeTile);
    bindActivation(p, rank, s, part != 0, k);
    return p;
}
template <class T, int M, bool Pipelined = (M == WideTile), class Output>
void multiply(const Rank &rank, const Rank::Projection &p, Output output, int n, int k, cudaStream_t stream, int mmaBlocks, PrefillSelection selection = {}) {
    auto &b = rank.b;
    // Both projections read the actual padded row count from the device layout.
    constexpr bool small = M == DecodeTile;
    const auto &a = M == WideTile ? p.wideA : small ? p.decodeA : p.a;
    const auto &sa = M == WideTile ? p.wideSa : small ? p.decodeSa : p.sa;
    constexpr bool wideDown = TileConfig<T, M, Pipelined>::wideDown;
    auto workspace = [&]() {
        if constexpr (Pipelined) return selection;
        else return static_cast<float *>(nullptr);
    }();
    int *layout = Pipelined && M == Tile ? b.compactLayout : b.layout;
    gemm<T, M, std::is_same_v<Output, QuantizedSwiGLU<Pipelined>>, Pipelined>
        <<<mmaBlocks, small ? 160 : 256, Smem<T, M, Pipelined>, stream>>>(
        output, nullptr, reinterpret_cast<__nv_fp8_e4m3 *>(b.a), nullptr, layout,
        nullptr, workspace, 0, n, k, n, 0, 0, a, wideDown ? p.wideB : p.b, sa, wideDown ? p.wideSb : p.sb,
        small ? p.decodeD : p.d);
}

struct VerifyMaps {
    cute::TmaDescriptor a, b, sa, sb;
    explicit VerifyMaps(const Rank::Projection &p) : a(p.decodeA), b(p.b), sa(p.decodeSa), sb(p.sb) {}
};
template<class T, class Output>
__device__ __forceinline__ void verifyProjection(const VerifyMaps &m, Buffers b, Output output, int n, int k) {
    constexpr bool gate = std::is_same_v<T, float>;
    deep_gemm::sm120_fp8_fp4_gemm_1d1d_body<0, 0, 0, 32, 32, 1, 16, 128, 128, 128, 128, 128,
        4, 32, 128, 0, deep_gemm::GemmType::MGroupedContiguous, false, T,
        deep_gemm::epilogue::transform::EpilogueIdentity, false, true, false, true, false, 16, 1,
        true, !gate, std::conditional_t<gate, Output, void>, void, true>(
        output, nullptr, nullptr, nullptr, b.layout, nullptr, nullptr, 0, n, k, n, 0, 0,
        m.a, m.b, m.sa, m.sb, m.a);
}

// One resident grid per rank. Peer rendezvous publish both quantized input and
// completed expert rows. All ranks must enter, including padded DP owners.
__global__ __launch_bounds__(160, 1) void VerifyMegaMoE(Mail *peers, int rank, Buffers b, Shape s,
        const BF *input, const int *ids, const float *scores, BF *output, const int *live, int rows,
        const __grid_constant__ VerifyMaps gate, const __grid_constant__ VerifyMaps down,
        QuantizedSwiGLU<false> activated) {
    constexpr int cap = 8;
    auto grid = cooperative_groups::this_grid();
    extern __shared__ __align__(1024) uint8_t storage[];
    if (blockIdx.x < cap * 2) {
        const int row = blockIdx.x / 2;
        packInputRow(storage, input, ids, scores, row, 4096, 8, row < rows && (!live || live[row] > 0));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (threadIdx.x < DispatchPeers)
            forwardEp8Packet(storage, peers, (rank + (blockIdx.x % 2) * DispatchPeers + threadIdx.x) % 8,
                             1, rank, row, cap);
    }
    grid.sync();
    if (!blockIdx.x) {
        joinPeers(peers, rank, 8);
        __syncthreads();
        planRoutes<4>(b, s, rank, cap, peers, activated.scores);
    }
    grid.sync();
    gatherRows(b, s);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    grid.sync();
    verifyProjection<float>(gate, b, activated, 4096, 4096);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    grid.sync();
    verifyProjection<cutlass::bfloat16_t>(down, b, b.returnRows, 4096, 2048);
    // The grid join publishes FC2 stores to the signaling CTA. Its system-scope
    // release is cumulative, exactly as for the dispatch rendezvous above.
    grid.sync();
    if (!blockIdx.x) joinPeers(peers, rank, 8);
    grid.sync();
    const int row = blockIdx.x / 4;
    if (row < rows) combineRows<8>(peers[rank], s, ids, output, live, row, blockIdx.x % 4, 4);
}
} // namespace

struct FastllmGenericEp {
    bool fusedVerify = true;
    Shape s;
    std::vector<std::unique_ptr<Rank>> ranks;
};
bool FastllmGenericEpSupported(const std::vector<int> &devices) {
    if (devices.size() < 2)
        return false;
    Restore restore;
    for (int d : devices) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess || p.major != 12 || p.minor != 0 ||
            p.sharedMemPerBlockOptin < Smem<float>)
            return false;
        check(cudaSetDevice(d));
        cudaFuncAttributes attr;
        if (cudaFuncGetAttributes(&attr, gemm<float>) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        for (int peer : devices)
            if (peer != d) {
                int access;
                if (cudaDeviceCanAccessPeer(&access, d, peer) != cudaSuccess || !access)
                    return false;
            }
    }
    return true;
}
namespace {
bool pairedEp8Numa(const std::vector<int> &devices) {
    if (devices.size() != 8) return false;
    int nodes[8];
    for (int r = 0; r < 8; ++r) {
        if (cudaDeviceGetAttribute(&nodes[r], cudaDevAttrHostNumaId, devices[r]) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (nodes[r] < 0 || nodes[r] != nodes[r & 4]) return false;
    }
    return nodes[0] != nodes[4];
}
bool supportsCompactPrefill(const std::vector<int> &devices) {
    return std::all_of(devices.begin(), devices.end(), [](int device) {
        int shared;
        check(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        return shared >= Smem<float, Tile, true>;
    });
}
bool prefillPipelineEnabled() {
    const char *option = std::getenv("FASTLLM_SM120_EP_PREFILL_PIPELINE");
    return !option || std::string(option) == "1";
}

std::unique_ptr<FastllmGenericEp> createWorkspace(const std::vector<int> &devices, int h, int i,
                                                  int experts, int topk, int capacity) {
    Restore restore;
    auto ep = std::make_unique<FastllmGenericEp>();
    Shape &s = ep->s;
    s = {h, i, (int)devices.size(), experts / (int)devices.size(), topk};
    s.cap = capacity; // Per-round workspace capacity; larger requests are chunked.
    const int alignment = capacity > 128 ? WideTile : Tile;
    const int64_t limit = std::numeric_limits<int>::max();
    if (int64_t(s.cap) * s.ranks * topk + (alignment - 1LL) * s.localE > limit ||
        int64_t(h) + h / 32 + int64_t(topk) * 8 + 15 > limit || i > limit / 4 || int64_t(h) * i / 64 > limit)
        throw std::runtime_error("SM120 generic EP tensor dimensions exceed kernel indexing");
    s.stride = (h + h / 32 + topk * 8 + 15) / 16 * 16;
    s.pool = (s.cap * s.ranks * topk + (alignment - 1) * s.localE) / alignment * alignment;
    std::vector<Mail> mail, prefetchMail;
    const bool relayAcrossNuma = capacity > 128 && pairedEp8Numa(devices);
    const bool compactPrefill = relayAcrossNuma && supportsCompactPrefill(devices);
    // Once a gate tile completes, FC2 can reuse its FP8 input rows for BF16 partials.
    s.inputPitch = relayAcrossNuma ? h * int(sizeof(BF)) : h;
    for (int d : devices) {
        check(cudaSetDevice(d));
        for (int peer : devices)
            if (d != peer) {
                auto err = cudaDeviceEnablePeerAccess(peer, 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled)
                    cudaGetLastError();
                else
                    check(err);
            }
        check(cudaFuncSetAttribute(gemm<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<float>));
        check(cudaFuncSetAttribute(gemm<cutlass::bfloat16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Smem<cutlass::bfloat16_t>));
        auto rank = std::make_unique<Rank>();
        rank->relayAcrossNuma = relayAcrossNuma;
        auto &a = rank->storage;
        a.device = d;
        check(cudaDeviceGetAttribute(&rank->sms, cudaDevAttrMultiProcessorCount, d));
        int maxShared;
        check(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, d));
        rank->tunedEp8 = s.ranks == 8 && h == 4096 && i == 2048 && experts == 256 && topk == 8 &&
                              rank->sms == 110 && maxShared >= Smem<float, DecodeTile>;
        // Every rank must enter the same peer rendezvous, including empty local slices.
        ep->fusedVerify = ep->fusedVerify && rank->tunedEp8;
        if (ep->fusedVerify) {
            int cooperative = 0, resident = 0;
            check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, d));
            ep->fusedVerify = cooperative && cudaFuncSetAttribute(VerifyMegaMoE,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<float, DecodeTile>) == cudaSuccess &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, VerifyMegaMoE, 160,
                    Smem<float, DecodeTile>) == cudaSuccess && resident > 0;
            if (!ep->fusedVerify) cudaGetLastError();
        }
        if (capacity > 128) {
            rank->compactPrefill = compactPrefill && rank->tunedEp8;
            if (rank->compactPrefill) {
                check(cudaFuncSetAttribute(gemm<float, Tile, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           Smem<float, Tile, true>));
                check(cudaFuncSetAttribute(gemm<cutlass::bfloat16_t, Tile, false, true>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<cutlass::bfloat16_t, Tile, true>));
            }
            rank->tiles = a.allocate<PrefillTile>(s.pool / (rank->compactPrefill ? Tile : WideTile));
            check(cudaStreamCreateWithFlags(&rank->prefillStream, cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&rank->prefillEvent, cudaEventDisableTiming));
            check(cudaFuncSetAttribute(gemm<float, WideTile, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<float, WideTile>));
            check(cudaFuncSetAttribute(gemm<cutlass::bfloat16_t, WideTile>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<cutlass::bfloat16_t, WideTile>));
        }
        if (rank->tunedEp8) {
            check(cudaFuncSetAttribute(gemm<float, Tile, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<float>));
            check(cudaFuncSetAttribute(gemm<float, DecodeTile, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       Smem<float, DecodeTile>));
            check(cudaFuncSetAttribute(gemm<cutlass::bfloat16_t, DecodeTile>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       Smem<cutlass::bfloat16_t, DecodeTile>));
        }
        auto &m = rank->mail;
        if (relayAcrossNuma) {
            check(cudaStreamCreateWithFlags(&rank->copyStream, cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&rank->copyStart, cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&rank->copyDone, cudaEventDisableTiming));
            Mail next{};
            next.receive = a.allocate<uint8_t>(size_t(s.cap) * s.ranks * s.stride);
            // The copy stream rendezvous can overlap the normal completion rendezvous.
            next.flags = a.allocate<unsigned long long>(s.ranks + 1);
            check(cudaMemset(next.flags, 0, (s.ranks + 1) * sizeof(*next.flags)));
            prefetchMail.push_back(next);
        }
        m.send = a.allocate<uint8_t>(size_t(s.cap) * s.stride);
        m.returned = a.allocate<BF>(size_t(s.cap) * s.k * h);
        m.flags = a.allocate<unsigned long long>(s.ranks + 1 + (rank->compactPrefill ? s.ranks : 0));
        check(cudaMemset(m.flags, 0, (s.ranks + 1) * sizeof(*m.flags)));

        rank->peers = a.allocate<Mail>(s.ranks);
        size_t tokens = s.cap * s.ranks;
        auto &b = rank->b;
        m.receive = b.incoming = a.allocate<uint8_t>(tokens * s.stride);
        mail.push_back(m);
        b.a = a.allocate<uint8_t>(size_t(s.pool) * std::max(s.inputPitch, i));
        b.sa = a.allocate<uint32_t>(size_t(s.pool) * std::max(h, i) / 128);
        if (rank->tunedEp8) {
            rank->rowScores = a.allocate<float>(s.pool);
            rank->downInput = a.allocate<uint8_t>(size_t(s.pool) * i);
            rank->downScales = a.allocate<uint32_t>(size_t(s.pool) * (i / 128));
        } else {
            b.gateUp = a.allocate<float>(size_t(s.pool) * 2 * i);
            rank->downInput = b.a;
            rank->downScales = b.sa;
        }
        b.returnRows = a.allocate<cutlass::bfloat16_t *>(s.pool);
        b.counts = a.allocate<int>(s.localE + 1);
        b.bases = a.allocate<int>(2 * s.localE);
        b.tasks = a.allocate<int2>(tokens * s.k);
        b.positions = a.allocate<int>(tokens * s.k);
        if (relayAcrossNuma) b.ownerSlots = a.allocate<uint8_t>(tokens);
        b.layout = a.allocate<int>(s.pool + 1) + 1;
        if (rank->compactPrefill) b.compactLayout = a.allocate<int>(s.pool + 1) + 1;
        ep->ranks.push_back(std::move(rank));
    }
    for (auto &r : ep->ranks) {
        check(cudaSetDevice(r->storage.device));
        if (relayAcrossNuma) {
            // Adjacent packets in the dense buffer originate at socket counterparts.
            std::array<Mail, 8> paired;
            for (int rank = 0; rank < 8; ++rank) {
                r->current.receive[rank] = mail[rank].receive;
                r->prefetched.receive[rank] = prefetchMail[rank].receive;
                paired[2 * (rank % 4) + rank / 4] = mail[rank];
            }
            r->current.peers = r->peers;
            r->prefetched.peers = r->storage.allocate<Mail>(prefetchMail.size());
            check(cudaMemcpy(r->prefetched.peers, prefetchMail.data(), prefetchMail.size() * sizeof(Mail),
                             cudaMemcpyHostToDevice));
            r->pairedReturnPeers = r->storage.allocate<Mail>(paired.size());
            check(cudaMemcpy(r->pairedReturnPeers, paired.data(), sizeof(paired), cudaMemcpyHostToDevice));
        }
        check(cudaMemcpy(r->peers, mail.data(), mail.size() * sizeof(Mail), cudaMemcpyHostToDevice));
    }
    return ep;
}
} // namespace

bool FastllmGenericEpTunedEp8(const std::vector<int> &devices) {
    if (devices.size() != 8 || !FastllmGenericEpSupported(devices)) return false;
    constexpr int requiredShared = std::max({Smem<float, DecodeTile>, Smem<float, WideTile>,
                                             Smem<cutlass::bfloat16_t, WideTile>});
    for (int device : devices) {
        int sms, shared;
        check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        check(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (sms != 110 || shared < requiredShared) return false;
    }
    return true;
}
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8Decode(const std::vector<int> &devices) {
    // Called only after validating the shipped EP8 weight layout and hardware.
    return createWorkspace(devices, 4096, 2048, 256, 8, 128);
}
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8(const std::vector<int> &devices, bool &densePrefill) {
    densePrefill = prefillPipelineEnabled() && pairedEp8Numa(devices) && supportsCompactPrefill(devices);
    if (!densePrefill) return FastllmCreateSharedEp8Decode(devices);
    return createWorkspace(devices, 4096, 2048, 256, 8, PrefillChunk);
}
std::shared_ptr<FastllmGenericEp> FastllmCreateGenericEp(const std::vector<int> &devices, int h, int i,
                                                         int experts, int topk) {
    if (!FastllmGenericEpSupported(devices) || h <= 0 || i <= 0 || h % 128 || i % 128 || experts <= 0 ||
        experts % devices.size() || topk <= 0 || topk > experts)
        throw std::runtime_error("Unsupported SM120 generic EP geometry");
    bool usePipeline = prefillPipelineEnabled() && h == 4096 && i == 2048 &&
                       experts == 256 && topk == 8 && FastllmGenericEpTunedEp8(devices);
    return createWorkspace(devices, h, i, experts, topk, usePipeline ? PrefillChunk : 128);
}
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &ptrs) {
    FastllmBindGenericEp(ep, rank, layer, ptrs, {});
}
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &ptrs, const std::array<void *, 2> &sharedWeights) {
    auto &r = *ep.ranks.at(rank);
    auto &s = ep.s;
    if (ptrs.size() != 2 * s.localE)
        throw std::runtime_error("Invalid generic EP weight count");
    Restore restore;
    check(cudaSetDevice(r.storage.device));
    auto inserted = r.layers.try_emplace(layer);
    if (!inserted.second)
        throw std::runtime_error("Generic EP layer weights already bound");
    auto &l = inserted.first->second;
    l.storage.device = r.storage.device;
    l.gate = bindProjection(l.storage, r, s, ptrs, 0, 2 * s.i, s.h, sharedWeights[0]);
    l.down = bindProjection(l.storage, r, s, ptrs, 1, s.h, s.i, sharedWeights[1]);
    check(cudaStreamSynchronize(cudaStreamPerThread));
}
namespace {
// Send each packed activation across the socket link once. Its socket partner
// and its local copy then form one contiguous DMA for each of the other three GPUs.
void dispatchDenseEp8(const DenseExchange &exchange, const Shape &s, int rank, const BF *input, const int *ids,
                      const float *scores, int cap, cudaStream_t stream) {
    const auto &receive = exchange.receive;
    const size_t bytes = size_t(cap) * s.stride;
    const size_t pairOffset = 2 * (rank % 4) * bytes;
    const size_t offset = pairOffset + (rank / 4) * bytes;
    Mail local{};
    local.send = receive[rank] + offset;
    publish<<<cap, 256, 0, stream>>>(local, s, input, ids, scores, cap, nullptr);
    check(cudaMemcpyAsync(receive[rank ^ 4] + offset, local.send, bytes, cudaMemcpyDeviceToDevice, stream));
    rendezvous<<<1, 32, 0, stream>>>(exchange.peers, rank, s.ranks);
    for (int step = 1; step < 4; ++step) {
        int peer = (rank & 4) + ((rank + step) % 4);
        check(cudaMemcpyAsync(receive[peer] + pairOffset, receive[rank] + pairOffset,
                              2 * bytes, cudaMemcpyDeviceToDevice, stream));
    }
}
void dispatchSparseEp8(Mail *peers, int rank, const BF *input, const int *ids, const float *scores,
                       int rows, int cap, const int *live, cudaStream_t stream) {
    dispatchEp8ToNumaPeers<<<cap, 256, 0, stream>>>(peers, rank, input, ids, scores, rows, cap, live);
    rendezvous<<<1, 32, 0, stream>>>(peers, rank, 8);
    forwardEp8WithinNumaNode<<<dim3(cap, 2), 128, 0, stream>>>(peers, rank, cap);
}
struct PrefetchChunk {
    int capacity = 0;
    int rows = 0;
    bool dense = false;
};
template <int M>
void prefillGemm(Rank &r, const Shape &s, const Rank::Layer &l, const Buffers &buffers, cudaStream_t stream,
                int gateBlocks, int downBlocks) {
    // Both tile sizes can be captured in one graph. Layout sets the inactive
    // GEMM's row count to zero, including when replay changes the routing.
    const QuantizedSwiGLU<true> output{r.downInput, reinterpret_cast<uint8_t *>(r.downScales), r.rowScores, s.pool, r.tiles};
    const unsigned columns = (2 * s.i + TileConfig<float, M, true>::n - 1) / TileConfig<float, M, true>::n;
    check(cudaMemsetAsync(r.tiles, 0, (s.pool / M) * sizeof(PrefillTile), stream));
    check(cudaEventRecord(r.prefillEvent, stream));
    check(cudaStreamWaitEvent(r.prefillStream, r.prefillEvent, 0));
    multiply<float, M, true>(r, l.gate, output, 2 * s.i, s.h, stream, gateBlocks);
    if constexpr (M == Tile)
        multiply<float, WideTile>(r, l.gate, output, 2 * s.i, s.h, stream, GateBlocks);
    multiply<cutlass::bfloat16_t, M, true>(r, l.down, buffers.returnRows, s.h, s.i, r.prefillStream,
                                            downBlocks, {r.tiles, columns, false});
    if constexpr (M == Tile)
        multiply<cutlass::bfloat16_t, WideTile>(r, l.down, buffers.returnRows, s.h, s.i, r.prefillStream,
                                                EarlyDownBlocks, {r.tiles, columns, false});
    check(cudaEventRecord(r.prefillEvent, r.prefillStream));
    check(cudaStreamWaitEvent(stream, r.prefillEvent, 0));
    multiply<cutlass::bfloat16_t, M, true>(r, l.down, buffers.returnRows, s.h, s.i, stream, r.sms,
                                            {r.tiles, columns, true});
    if constexpr (M == Tile)
        multiply<cutlass::bfloat16_t, WideTile>(r, l.down, buffers.returnRows, s.h, s.i, stream, r.sms,
                                                {r.tiles, columns, true});
}
void runChunk(Rank &r, const FastllmGenericEp &ep, const Rank::Layer &l, int rank, const BF *rowsIn, const int *index,
              const float *weights, BF *out, int n, int cap, const int *keys, cudaStream_t stream,
              bool dense, bool prefetched, PrefetchChunk next, DecodeSelection selection = {}) {
    const auto &s = ep.s;
    if (ep.fusedVerify && cap == 8) {
        cudaLaunchAttribute cooperative{};
        cooperative.id = cudaLaunchAttributeCooperative;
        cooperative.val.cooperative = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = r.sms;
        cfg.blockDim = 160;
        cfg.dynamicSmemBytes = Smem<float, DecodeTile>;
        cfg.stream = stream;
        cfg.attrs = &cooperative;
        cfg.numAttrs = 1;
        QuantizedSwiGLU<false> activated{r.downInput, reinterpret_cast<uint8_t *>(r.downScales), r.rowScores, s.pool};
        check(cudaLaunchKernelEx(&cfg, VerifyMegaMoE, r.peers, rank, r.b, s, rowsIn, index, weights, out,
                                 keys, n, VerifyMaps(l.gate), VerifyMaps(l.down), activated));
        return;
    }
    auto buffers = r.b;
    if (prefetched) buffers.incoming = r.prefetched.receive[rank];
    const bool shortBatch = r.tunedEp8 && cap <= 32;
    const bool paired = dense && r.pairedReturnPeers && r.tunedEp8 && cap > 128;
    // Sparse chunks have fewer repeated hits to amortize the owner reduction.
    const bool ownerPartial = buffers.ownerSlots && cap >= (paired ? MinOwnerCombineRows : 1024);
    // Mid-sized dense chunks benefit from avoiding a local publish pass. Full
    // chunks reuse the input transfer pipeline and aggregate on both sockets.
    const bool crossSocketOnly = paired && ownerPartial && cap < PrefillChunk;
    const bool compact = r.compactPrefill && crossSocketOnly && cap < CompactPrefillLimit;
    const Mail *returnPeers = paired ? r.pairedReturnPeers : r.peers;
    if (selection.probabilities) {
        dispatchRoutedRow<<<dim3(cap, 8 / DispatchPeers), 256, 0, stream>>>(r.peers, rank, rowsIn, selection, keys, n, cap);
    } else if (prefetched) {
        check(cudaStreamWaitEvent(stream, r.copyDone, 0));
    } else if (paired) {
        dispatchDenseEp8(r.current, s, rank, rowsIn, index, weights, cap, stream);
    } else if (r.tunedEp8 && cap > 128 && r.relayAcrossNuma) {
        dispatchSparseEp8(r.peers, rank, rowsIn, index, weights, n, cap, keys, stream);
    } else if (shortBatch || (r.tunedEp8 && cap > 128)) {
        dispatchDecodeRows<<<dim3(cap, s.ranks / DispatchPeers), 256, 0, stream>>>(
            r.peers, rank, rowsIn, index, weights, n, cap, keys);
    } else {
        publish<<<cap, 256, 0, stream>>>(r.mail, s, rowsIn, index, weights, n, keys);
        dispatch<<<dim3(cap, s.ranks), 128, 0, stream>>>(r.peers, s, rank, cap);
    }
    rendezvous<<<1, 32, 0, stream>>>(r.peers, rank, s.ranks);
    if (next.capacity) {
        // Separate mailboxes and barrier epochs let the next dispatch overlap this chunk.
        check(cudaEventRecord(r.copyStart, stream));
        check(cudaStreamWaitEvent(r.copyStream, r.copyStart, 0));
        const BF *input = next.rows ? rowsIn + size_t(cap) * s.h : nullptr;
        const int *ids = next.rows ? index + size_t(cap) * s.k : nullptr;
        const float *scores = next.rows ? weights + size_t(cap) * s.k : nullptr;
        if (next.dense) {
            dispatchDenseEp8(r.prefetched, s, rank, input, ids, scores, next.capacity, r.copyStream);
        } else {
            const int *live = next.rows && keys ? keys + cap : nullptr;
            dispatchSparseEp8(r.prefetched.peers, rank, input, ids, scores, next.rows, next.capacity, live, r.copyStream);
        }
        check(cudaEventRecord(r.copyDone, r.copyStream));
    }
    int routes = cap * s.ranks * s.k;
    if (shortBatch && cap <= 4) {
        planTinyRoutes<<<1, 256, 0, stream>>>(buffers, s, rank, cap, r.peers, r.rowScores);
    } else {
        check(cudaMemsetAsync(buffers.counts, 0, s.localE * sizeof(int), stream));
        countRoutes<<<(routes + 255) / 256, 256, 0, stream>>>(buffers, s, rank, routes);
        if (r.tunedEp8) {
            if (shortBatch) layout<DecodeTile, true><<<s.localE, 32, 0, stream>>>(buffers, s, r.rowScores);
            else if (compact) layout<Tile, true, true><<<s.localE, 32, 0, stream>>>(buffers, s, r.rowScores, r.peers, rank, cap);
            else if (cap > 128) layout<WideTile, true><<<s.localE, 32, 0, stream>>>(buffers, s, r.rowScores);
            else layout<Tile, true><<<s.localE, 32, 0, stream>>>(buffers, s, r.rowScores);
            if (compact)
                makeTasks<true, true, true><<<(routes + 255) / 256, 256, 0, stream>>>(
                    buffers, s, rank, routes, returnPeers, cap, r.rowScores, true, r.mail.directShares());
            else if (ownerPartial)
                makeTasks<true, true><<<(routes + 255) / 256, 256, 0, stream>>>(
                    buffers, s, rank, routes, returnPeers, cap, r.rowScores, crossSocketOnly);
            else
                makeTasks<true><<<(routes + 255) / 256, 256, 0, stream>>>(
                    buffers, s, rank, routes, returnPeers, cap, r.rowScores);
        } else {
            layout<Tile, false><<<s.localE, 32, 0, stream>>>(buffers, s, nullptr);
            makeTasks<false><<<(routes + 255) / 256, 256, 0, stream>>>(buffers, s, rank, routes, r.peers, cap, nullptr);
        }
    }
    int blocks = std::min(routes, 4 * r.sms);
    gather<<<blocks, 128, 0, stream>>>(buffers, s);
    if (r.tiles && cap > 128) {
        if (compact) prefillGemm<Tile>(r, s, l, buffers, stream, CompactGateBlocks, CompactDownBlocks);
        else prefillGemm<WideTile>(r, s, l, buffers, stream, GateBlocks, EarlyDownBlocks);
    } else {
        if (r.tunedEp8) {
            const QuantizedSwiGLU<false> output{r.downInput, reinterpret_cast<uint8_t *>(r.downScales), r.rowScores, s.pool};
            if (shortBatch) multiply<float, DecodeTile>(r, l.gate, output, 2 * s.i, s.h, stream, r.sms);
            else multiply<float, Tile>(r, l.gate, output, 2 * s.i, s.h, stream, r.sms);
        } else {
            multiply<float, Tile>(r, l.gate, buffers.gateUp, 2 * s.i, s.h, stream, r.sms);
            activate<<<blocks, 256, 0, stream>>>(buffers, s);
        }
        if (shortBatch) multiply<cutlass::bfloat16_t, DecodeTile>(r, l.down, buffers.returnRows, s.h, s.i, stream, r.sms);
        else multiply<cutlass::bfloat16_t, Tile>(r, l.down, buffers.returnRows, s.h, s.i, stream, r.sms);
    }
    if (ownerPartial)
        publishOwnerRows<<<s.ranks * cap, 256, 0, stream>>>(buffers, returnPeers, cap);
    rendezvous<<<1, 32, 0, stream>>>(r.peers, rank, s.ranks);
    if (compact && n)
        combineDenseEp8Rows<true><<<n, 256, 0, stream>>>(r.mail, index, out, rank);
    else if (crossSocketOnly && n)
        combineDenseEp8Rows<><<<n, 256, 0, stream>>>(r.mail, index, out, rank);
    else if (paired && n && ownerPartial)
        combineDenseEp8Rows<false, OwnerReturn::Reduced><<<n, 256, 0, stream>>>(r.mail, index, out, rank);
    else if (paired && n)
        combineDenseEp8Rows<false, OwnerReturn::Direct><<<n, 256, 0, stream>>>(r.mail, index, out, rank);
    else if (ownerPartial && n)
        combine<8, true><<<n, 256, 0, stream>>>(r.mail, s, index, out, keys);
    else if (n && shortBatch)
        combine<8><<<dim3(n, 4), 128, 0, stream>>>(r.mail, s, index,
            out, keys);
    else if (n)
        combine<><<<n, 256, 0, stream>>>(r.mail, s, index, out, keys);
    check(cudaGetLastError());
}
} // namespace

void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *out, int rows, int commonRows, const int *live) {
    FastllmRunGenericEp(ep, rank, layer, input, ids, scores, out, rows, commonRows, live, 0);
}

void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *out, int rows, int commonRows, const int *live, int densePrefixRows) {
    auto &r = *ep.ranks.at(rank);
    const auto &s = ep.s;
    // Validate the layer even for an empty batch, as in the original entry point.
    const auto &primaryLayer = r.layers.at(layer);
    if (rows < 0 || rows > commonRows || densePrefixRows < 0 || densePrefixRows > rows || (live && densePrefixRows))
        throw std::runtime_error("Invalid generic EP row count");
    const auto stream = cudaStreamPerThread;
    bool prefetched = false;
    for (int64_t start = 0; start < commonRows; start += s.cap) {
        int cap = std::min<int64_t>(commonRows - start, s.cap);
        int first = std::min<int64_t>(start, rows), n = std::min(rows - first, cap);
        PrefetchChunk next;
        const int nextCap = std::min<int64_t>(s.cap, commonRows - start - cap);
        const bool nextDense = start + cap + nextCap <= densePrefixRows;
        if (!prefetched && r.copyStream && nextCap > 128 && (nextDense || nextCap >= 512)) {
            next.capacity = nextCap;
            next.rows = std::min<int64_t>(nextCap, std::max<int64_t>(0, int64_t(rows) - start - cap));
            next.dense = nextDense;
        }
        runChunk(r, ep, primaryLayer, rank,
                 n ? static_cast<const BF *>(input) + size_t(first) * s.h : nullptr,
                 n ? ids + size_t(first) * s.k : nullptr, n ? scores + size_t(first) * s.k : nullptr,
                 n ? static_cast<BF *>(out) + size_t(first) * s.h : nullptr, n, cap,
                 n && live ? live + first : nullptr, stream, start + cap <= densePrefixRows, prefetched, next);
        prefetched = next.capacity != 0;
    }
}

bool FastllmTryGenericEpRoutedDecode(FastllmGenericEp &ep, int rank, int layer, const void *input,
    const float *probabilities, const float *bias, void *output, const int *live, bool normalize, float scale,
    int rows, int capacity, bool logits) {
    auto &r = *ep.ranks.at(rank);
    if (!r.tunedEp8 || capacity > ep.s.cap || (ep.fusedVerify && capacity == 8)) return false;
    // Direct dispatch bypasses the local send packet. Its storage remains
    // available to combine until this rank finishes the layer on the same stream.
    int *ids = reinterpret_cast<int *>(r.mail.send);
    runChunk(r, ep, r.layers.at(layer), rank, static_cast<const BF *>(input), ids, nullptr,
        static_cast<BF *>(output), rows, capacity, live, cudaStreamPerThread, false, false, {},
        {probabilities, bias, ids, normalize, scale, logits});
    return true;
}
