#pragma once
// Hopper dense-KV specialization. TMA reads a swizzled tile directly;
// eight heads share it without gathering or materializing global scores.
// Eager launches carry tensor maps by value. Serving graphs can instead read
// fixed device descriptors refreshed when the request-to-KV binding changes.
// Sparse or unsupported layouts use Decode in the companion CUDA header.
#include "naive-n05-attention-bulk.cuh"
#if CUDART_VERSION >= 12010
#include <cuda.h>
namespace naive_attention_bulk {
__device__ inline int swizzle(int p) { return p ^ (((p >> 6) & 7) * 8); }
__device__ inline void tma_load(void *dst, const CUtensorMap *map, void *bar, int first,
                                int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned d = (unsigned)__cvta_generic_to_shared(dst),
             b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
        "[%1, {0, 0, %2}], [%3];" ::"r"(d),
        "l"(map), "r"(first), "r"(b)
        : "memory");
#endif
}
__device__ inline void tma_wait(void *bar, int phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile(
        "{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p "
        "bra wait_loop; }" ::"r"(b),
        "r"(phase)
        : "memory");
#endif
}
template <int Max> struct TmaStorage {
    union {
        BF keys[128 * 192];
        BF values[128 * 128];
    } tile;
    BF qs[8][200];
    BF prob[8][Max + 8];
    unsigned long long barrier;
};
// Direct and indirect descriptors both fit the 32-KiB parameter limit.
constexpr int TmaBatchLimit = 64;
struct TmaBatch {
    FastllmNaiveDSARow rows[TmaBatchLimit];
    CUtensorMap maps[TmaBatchLimit * 2];
};
static_assert(sizeof(TmaBatch) + 128 < 32764, "TMA arguments exceed CUDA parameter space");
static_assert(sizeof(CUtensorMap) * 2 == FastllmNaiveDSAMapBytes, "Unexpected tensor map size");
inline bool EncodeMaps(CUtensorMap *maps, const void *key, const void *value, int capacity, int stride) {
    for (int j = 0; j < 2; ++j) {
        uint64_t shape[3] = {64, (uint64_t)(j ? 2 : 3), (uint64_t)capacity};
        uint64_t strides[2] = {128, (uint64_t)(j ? 128 : stride) * 2};
        uint32_t box[3] = {64, (uint32_t)(j ? 2 : 3), 128}, steps[3] = {1, 1, 1};
        if (cuTensorMapEncodeTiled(maps + j, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
            (void *)(j ? value : key), shape, strides, box, steps,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return false;
    }
    return true;
}
inline bool PrepareTmaBatch(TmaBatch &batch, const FastllmNaiveDSARow *rows, int count) {
    for (int i = 0; i < count; ++i) {
        const auto &r = batch.rows[i] = rows[i];
        if (r.fp8 || (r.pageLen && r.pageLen % 128) || r.indices || r.capacity <= 0 || (r.cache && !r.tensorMaps)) return false;
        if (!r.tensorMaps && !EncodeMaps(batch.maps + i * 2, r.key, r.value, r.poolTokens ? r.poolTokens : r.capacity, r.keyStride)) return false;
    }
    return true;
}
template <int Max>
__global__ __launch_bounds__(256) void HopperDecode(const BF *q,
                                                    const __grid_constant__ TmaBatch batch,
                                                    const float *sink, BF *out, int heads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace naive_dsa_decode;
    extern __shared__ __align__(1024) char tbuf[];
    auto &s = *(TmaStorage<Max> *)tbuf;
    const int row = blockIdx.x, group = blockIdx.y, head = group * 8, t = threadIdx.x, l = t % 32,
              w = t / 32;
    auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    if (ZeroInactive(r, out + ((size_t)row * heads + head) * 128, 8 * 128)) return;
    const int n = Length(r);
    const auto *maps = r.tensorMaps ? (const CUtensorMap *)r.tensorMaps : batch.maps + row * 2;
    if (t == 0 && r.tensorMaps) {
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(maps) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(maps + 1) : "memory");
    }
    for (int i = t; i < 8 * 24; i += 256) {
        int h = i / 24, d = i % 24 * 8;
        *(uint4 *)&s.qs[h][d] = *(const uint4 *)(q + ((size_t)row * heads + head + h) * 192 + d);
    }
    if (t == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
                         (unsigned)__cvta_generic_to_shared(&s.barrier))
                     : "memory");
    // Publish initialization before other warps wait or TMA completes a transaction.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    int phase = 0;
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < n; first += 128) {
        if (t == 0)
            tma_load(s.tile.keys, maps, &s.barrier, FastllmNaivePhysicalRow(r, first), 128 * 192 * 2);
        tma_wait(&s.barrier, phase);
        phase ^= 1;
        __syncthreads();
        float acc[4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], b[2];
            ld4(a, &s.tile.keys[swizzle((w * 16 + ar) * 192 + d + ac)]);
            ld2(b, &s.qs[l & 7][d + ((l >> 3) & 1) * 8]);
            mma(acc, a, b);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            if (first + k < Max)
                s.prob[h][first + k] = __float2bfloat16(
                    first + k < n ? Round(Round(acc[i]) * rsqrtf(192.f)) : -INFINITY);
        }
        __syncthreads();
    }
    // One warp owns a head; reproduce the virtual 256-lane softmax tree.
    Normalize<Max, true>(&s.prob[0][0], sink, head, n, w, l);
    __syncthreads();
    float acc[4] = {};
    for (int first = 0; first < n; first += 128) {
        if (t == 0)
            tma_load(s.tile.values, maps + 1, &s.barrier, FastllmNaivePhysicalRow(r, first, true), 128 * 128 * 2);
        tma_wait(&s.barrier, phase);
        phase ^= 1;
        if (first + 128 > n)
            for (int i = t; i < 128 * 128; i += 256)
                if (first + i / 128 >= n)
                    s.tile.values[swizzle(i)] = __float2bfloat16(0.f);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            uint32_t a[4], b[2];
            ldt4(a, &s.tile.values[swizzle((k + (l & 7) + (l >> 4) * 8) * 128 + w * 16 +
                                           ((l >> 3) & 1) * 8)]);
            ld2(b, &s.prob[l & 7][first + k + ((l >> 3) & 1) * 8]);
            mma(acc, a, b);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
        out[((size_t)row * heads + head + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 +
            (i / 2) * 8] = __float2bfloat16(acc[i]);

#endif
}
} // namespace naive_attention_bulk
#endif
