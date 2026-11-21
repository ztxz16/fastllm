// SM120 complete-expert EP. The grouped MXFP8 x MXFP4 MMA is built from the
// vendored DeepGEMM headers (MIT, third_party/deep_gemm). The P2P exchange,
// routing and workspace layout below are independent of a particular GPU SKU.
#include "devices/cuda/fastllm-mxfp4-ep-generic.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using BF = __nv_bfloat16;
constexpr int Tile = 32, Stages = 3;
// D staging, then A, B and two scale tiles per stage, then mbarriers.
template <class T>
constexpr int Smem = Tile * 128 * sizeof(T) + Stages * ((Tile + 128) * 128 + (Tile + 128) * 4 + 16);
template <class T>
constexpr auto gemm =
    &deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0, 0, 0, 32, 32, 1, Tile, 128, 128, 128, 128, 128, Stages, 128,
                                             128, 0, deep_gemm::GemmType::MGroupedContiguous, false, T,
                                             deep_gemm::epilogue::transform::EpilogueIdentity, false, true,
                                             false, true, false, Tile, 1, true, std::is_same_v<T, cutlass::bfloat16_t>>;

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
    int h, i, ranks, localE, k, cap, stride, pool;
};
struct Mail {
    uint8_t *send, *receive;
    BF *returned;
    unsigned long long *flags;
};
struct Buffers {
    uint8_t *incoming, *a;
    uint32_t *sa;
    float *gateUp;
    cutlass::bfloat16_t **returnRows;
    int *counts, *bases, *positions, *layout;
    int2 *tasks;
};
struct Rank {
    Storage storage;
    int sms;
    Mail mail;
    Mail *peers;
    Buffers b;
    struct Projection {
        cute::TmaDescriptor a, b, sa, sb, d;
    };
    struct Layer {
        Storage storage;
        Projection gate, down;
    };
    std::map<int, Layer> layers;
};

// Publish the preceding stream work and await all peers. Only the owner writes
// its epoch; each sender has an independent mailbox on every receiver.
__global__ void rendezvous(Mail *peers, int rank, int ranks) {
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

__device__ uint8_t quantize(float x, uint8_t &sf) {
    float a = fabsf(x);
    for (int d = 16; d; d /= 2)
        a = fmaxf(a, __shfl_xor_sync(~0u, a, d));
    const float scale = fmaxf(a, 1e-4f) / 448.f;
    int exp = __float_as_int(scale);
    sf = (exp >> 23) + ((exp & 0x7fffff) != 0);
    return __nv_cvt_float_to_fp8(x * __uint_as_float((254 - sf) << 23), __NV_SATFINITE, __NV_E4M3);
}
__global__ void publish(Mail mail, Shape s, const BF *input, const int *ids, const float *scores, int rows,
                        const int *live) {
    int row = blockIdx.x;
    uint8_t *p = mail.send + size_t(row) * s.stride;
    bool active = row < rows && (!live || live[row] > 0);
    int *routes = reinterpret_cast<int *>(p + s.h + s.h / 32);
    for (int j = threadIdx.x; j < s.k; j += blockDim.x) {
        routes[j] = active ? ids[row * s.k + j] : -1;
        reinterpret_cast<float *>(routes + s.k)[j] = active ? scores[row * s.k + j] : 0.f;
    }
    for (int c = threadIdx.x; c < s.h; c += blockDim.x) {
        uint8_t sf;
        p[c] = quantize(active ? __bfloat162float(input[size_t(row) * s.h + c]) : 0.f, sf);
        if (!(c % 32))
            p[s.h + c / 32] = sf;
    }
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
__global__ void layout(Buffers b, Shape s) {
    int e = blockIdx.x;
    int base = 0, compact = 0;
    for (int j = 0; j < e; ++j) {
        base += (b.counts[j] + Tile - 1) / Tile * Tile;
        compact += b.counts[j];
    }
    const int padded = (b.counts[e] + Tile - 1) / Tile * Tile;
    for (int j = b.counts[e] + threadIdx.x; j < padded; j += blockDim.x)
        b.returnRows[base + j] = nullptr;
    if (!threadIdx.x) {
        b.bases[e] = base;
        b.bases[s.localE + e] = compact;
        if (e + 1 == s.localE) {
            b.counts[s.localE] = compact + b.counts[e];
            b.layout[-1] = base + padded;
        }
    }
    for (int j = threadIdx.x * Tile; j < b.counts[e]; j += blockDim.x * Tile)
        b.layout[base + j] = e;
}
__global__ void makeTasks(Buffers b, Shape s, int rank, int routes, const Mail *peers, int rowsPerPeer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= routes || b.positions[t] < 0)
        return;
    int e = expert(b, s, t) - rank * s.localE, offset = b.positions[t];
    const int row = b.bases[e] + offset;
    const int slotsPerPeer = rowsPerPeer * s.k;
    auto *mailbox = peers[t / slotsPerPeer].returned + size_t(t % slotsPerPeer) * s.h;
    b.returnRows[row] = reinterpret_cast<cutlass::bfloat16_t *>(mailbox);
    b.tasks[b.bases[s.localE + e] + offset] = make_int2(t, row);
}
__global__ void gather(Buffers b, Shape s) {
    int count = b.counts[s.localE];
    for (int job = blockIdx.x; job < count; job += gridDim.x) {
        int2 task = b.tasks[job];
        auto *p = b.incoming + size_t(task.x / s.k) * s.stride;
        for (int j = threadIdx.x; j < s.h / 16; j += blockDim.x)
            reinterpret_cast<uint4 *>(b.a + size_t(task.y) * s.h)[j] = reinterpret_cast<const uint4 *>(p)[j];
        for (int j = threadIdx.x; j < s.h / 128; j += blockDim.x)
            b.sa[size_t(j) * s.pool + task.y] = reinterpret_cast<const uint32_t *>(p + s.h)[j];
    }
}
__global__ void activate(Buffers b, Shape s) {
    int count = b.counts[s.localE];
    for (int job = blockIdx.x; job < count; job += gridDim.x) {
        int2 task = b.tasks[job];
        const auto *meta = b.incoming + size_t(task.x / s.k) * s.stride + s.h + s.h / 32 + s.k * sizeof(int);
        float score = reinterpret_cast<const float *>(meta)[task.x % s.k];
        const float *gate = b.gateUp + size_t(task.y) * 2 * s.i;
        for (int col = threadIdx.x; col < s.i; col += blockDim.x) {
            float g = gate[col];
            float v = (g / (1.f + __expf(-g))) * gate[s.i + col];
            uint8_t sf;
            b.a[size_t(task.y) * s.i + col] = quantize(v * score, sf);
            if (!(col % 32))
                reinterpret_cast<uint8_t *>(
                    b.sa)[(size_t(col / 128) * s.pool + task.y) * 4 + col % 128 / 32] = sf;
        }
    }
}
__global__ void combine(Mail local, Shape s, const int *ids, BF *out, const int *live) {
    const int row = blockIdx.x;
    const bool active = !live || live[row] > 0;
    constexpr int width = sizeof(uint4) / sizeof(BF);
    for (int chunk = threadIdx.x; chunk < s.h / width; chunk += blockDim.x) {
        float sum[width] = {};
        if (active) {
            for (int j = 0; j < s.k; ++j) {
                const int slot = row * s.k + j;
                if (static_cast<unsigned>(ids[slot]) >= static_cast<unsigned>(s.ranks * s.localE)) continue;
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
Rank::Projection bindProjection(Storage &owner, Buffers b, Shape s, const std::vector<uint64_t> &ptrs,
                                int part, int n, int k) {
    size_t values = size_t(n) * k / 2, words = size_t(n) * k / 128;
    auto *w = owner.allocate<uint8_t>(values * s.localE);
    auto *sf = owner.allocate<uint32_t>(words * s.localE);
    for (int e = 0; e < s.localE; ++e) {
        auto *src = reinterpret_cast<const uint8_t *>(ptrs[2 * e + part]);
        check(cudaMemcpyAsync(w + e * values, src, values, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        packScales<<<(words + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const uint32_t *>(src + values), sf + e * words, n, k / 128);
    }
    auto sw = CU_TENSOR_MAP_SWIZZLE_128B;
    auto none = CU_TENSOR_MAP_SWIZZLE_NONE;
    return {tensor(b.a, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, s.pool, k, 128, Tile, sw),
            tensor(w, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, k, size_t(n) * s.localE, k / 2, 128, 128, sw),
            tensor(b.sa, CU_TENSOR_MAP_DATA_TYPE_INT32, s.pool, k / 128, s.pool * 4, Tile, 1, none),
            tensor(sf, CU_TENSOR_MAP_DATA_TYPE_INT32, n, size_t(k / 128) * s.localE, n * 4, 128, 1, none),
            part ? cute::TmaDescriptor{} : tensor(b.gateUp, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, n, s.pool, n * 4, 32, Tile, sw)};
}
template <class T, class Output>
void multiply(const Rank &rank, const Rank::Projection &p, Output output, int n, int k) {
    auto &b = rank.b;
    // Both projections read the actual padded row count from the device layout.
    gemm<T><<<rank.sms, 256, Smem<T>, cudaStreamPerThread>>>(
        output, nullptr, reinterpret_cast<__nv_fp8_e4m3 *>(b.a), nullptr, b.layout,
        nullptr, nullptr, 0, n, k, n, 0, 0, p.a, p.b, p.sa, p.sb, p.d);
}
} // namespace

struct FastllmGenericEp {
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
std::shared_ptr<FastllmGenericEp> FastllmCreateGenericEp(const std::vector<int> &devices, int h, int i,
                                                         int experts, int topk) {
    if (!FastllmGenericEpSupported(devices) || h <= 0 || i <= 0 || h % 128 || i % 128 || experts <= 0 ||
        experts % devices.size() || topk <= 0 || topk > experts)
        throw std::runtime_error("Unsupported SM120 generic EP geometry");
    Restore restore;
    auto ep = std::make_shared<FastllmGenericEp>();
    Shape &s = ep->s;
    s = {h, i, (int)devices.size(), experts / (int)devices.size(), topk};
    s.cap = 128; // Per-round workspace capacity; larger requests are chunked.
    const int64_t limit = std::numeric_limits<int>::max();
    if (int64_t(s.cap) * s.ranks * topk + (Tile - 1LL) * s.localE > limit ||
        int64_t(h) + h / 32 + int64_t(topk) * 8 + 15 > limit || i > limit / 4 || int64_t(h) * i / 64 > limit)
        throw std::runtime_error("SM120 generic EP tensor dimensions exceed kernel indexing");
    s.stride = (h + h / 32 + topk * 8 + 15) / 16 * 16;
    s.pool = (s.cap * s.ranks * topk + (Tile - 1) * s.localE) / Tile * Tile;
    std::vector<Mail> mail;
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
        auto &a = rank->storage;
        a.device = d;
        check(cudaDeviceGetAttribute(&rank->sms, cudaDevAttrMultiProcessorCount, d));
        auto &m = rank->mail;
        m.send = a.allocate<uint8_t>(size_t(s.cap) * s.stride);
        m.returned = a.allocate<BF>(size_t(s.cap) * s.k * h);
        m.flags = a.allocate<unsigned long long>(s.ranks + 1);
        check(cudaMemset(m.flags, 0, (s.ranks + 1) * sizeof(*m.flags)));

        rank->peers = a.allocate<Mail>(s.ranks);
        size_t tokens = s.cap * s.ranks;
        auto &b = rank->b;
        m.receive = b.incoming = a.allocate<uint8_t>(tokens * s.stride);
        mail.push_back(m);
        b.a = a.allocate<uint8_t>(size_t(s.pool) * std::max(h, i));
        b.sa = a.allocate<uint32_t>(size_t(s.pool) * std::max(h, i) / 128);
        b.gateUp = a.allocate<float>(size_t(s.pool) * 2 * i);
        b.returnRows = a.allocate<cutlass::bfloat16_t *>(s.pool);
        b.counts = a.allocate<int>(s.localE + 1);
        b.bases = a.allocate<int>(2 * s.localE);
        b.tasks = a.allocate<int2>(tokens * s.k);
        b.positions = a.allocate<int>(tokens * s.k);
        b.layout = a.allocate<int>(s.pool + 1) + 1;
        ep->ranks.push_back(std::move(rank));
    }
    for (auto &r : ep->ranks) {
        check(cudaSetDevice(r->storage.device));
        check(cudaMemcpy(r->peers, mail.data(), mail.size() * sizeof(Mail), cudaMemcpyHostToDevice));
    }
    return ep;
}
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &ptrs) {
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
    l.gate = bindProjection(l.storage, r.b, s, ptrs, 0, 2 * s.i, s.h);
    l.down = bindProjection(l.storage, r.b, s, ptrs, 1, s.h, s.i);
    check(cudaStreamSynchronize(cudaStreamPerThread));
}
void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *out, int rows, int commonRows, const int *live) {
    auto &r = *ep.ranks.at(rank);
    const auto &s = ep.s;
    const auto &l = r.layers.at(layer);
    if (rows < 0 || rows > commonRows)
        throw std::runtime_error("Invalid generic EP row count");
    auto stream = cudaStreamPerThread;
    for (int64_t start = 0; start < commonRows; start += s.cap) {
        int cap = std::min<int64_t>(commonRows - start, s.cap);
        int first = std::min<int64_t>(start, rows), n = std::min(rows - first, cap);
        auto *keys = live ? live + first : nullptr;
        const int *index = n ? ids + size_t(first) * s.k : nullptr;
        publish<<<cap, 256, 0, stream>>>(r.mail, s,
                                         n ? static_cast<const BF *>(input) + size_t(first) * s.h : nullptr,
                                         index, n ? scores + size_t(first) * s.k : nullptr, n, keys);
        dispatch<<<dim3(cap, s.ranks), 128, 0, stream>>>(r.peers, s, rank, cap);
        rendezvous<<<1, 32, 0, stream>>>(r.peers, rank, s.ranks);
        int routes = cap * s.ranks * s.k;
        check(cudaMemsetAsync(r.b.counts, 0, s.localE * sizeof(int), stream));
        countRoutes<<<(routes + 255) / 256, 256, 0, stream>>>(r.b, s, rank, routes);
        layout<<<s.localE, 32, 0, stream>>>(r.b, s);
        makeTasks<<<(routes + 255) / 256, 256, 0, stream>>>(r.b, s, rank, routes, r.peers, cap);
        int blocks = std::min(routes, 4 * r.sms);
        gather<<<blocks, 128, 0, stream>>>(r.b, s);
        multiply<float>(r, l.gate, r.b.gateUp, 2 * s.i, s.h);
        activate<<<blocks, 256, 0, stream>>>(r.b, s);
        multiply<cutlass::bfloat16_t>(r, l.down, r.b.returnRows, s.h, s.i);
        rendezvous<<<1, 32, 0, stream>>>(r.peers, rank, s.ranks);
        if (n)
            combine<<<n, 256, 0, stream>>>(r.mail, s, index, static_cast<BF *>(out) + size_t(first) * s.h,
                                           keys);
        check(cudaGetLastError());
    }
}
