#pragma once
#include "naive-n05-dsa-mma.cuh"
#include "naive-n05-dsa-decode.cuh"
#include <cuda/atomic>
#include "devices/cuda/fastllm-mxfp8-activation.cuh"

// Split attention for BF16 GQA. No cooperative launch or resident-CTA spin:
// the last split combines the partials, allowing arbitrary grid sizes.
// Only standard SM80 BF16 MMA/cp.async instructions are required.
namespace naive_dsa_split {
using BF16 = __nv_bfloat16;
constexpr int Heads = 8, Dim = 192, Values = 128, Splits = 16, Threads = 256;
constexpr int Partial = Splits * Heads * Values;
constexpr int Scratch = Partial + Splits * Heads * 2;
struct Storage {
    BF16 query[Heads][Dim + 8];
    union {
        BF16 key[128][Dim + 8];
        BF16 probability[Heads][136];
    };
    BF16 value[128][136];
    float scores[128][Heads];
    int valid[128];
};

template<bool Pack = false>
__device__ __forceinline__ void Attend(const BF16 *query,
        FastllmNaiveDSARow descriptor, const float *sink, BF16 *output,
        float *scratch, unsigned *completion, int heads, int kvHeads, int row, int group, int split, FastllmMxfp8Activation packed = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_mma;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int h0 = group * Heads, kh = h0 / (heads / kvHeads);
    const int work = row * (heads / Heads) + group;
    const auto r = FastllmNaiveResolveDSARow(descriptor);
    const int length = naive_dsa_decode::Length(r);
    const int count = length ? (r.indices ? r.selectedCount : length) : 0;
    if (!count) {
        if (!split) for (int i = t; i < Heads * Values; i += Threads) {
            const BF16 value[] = {__float2bfloat16(0.f)};
            output[(size_t)row * heads * Values + h0 * Values + i] = value[0];
            if constexpr (Pack) fastllm_mxfp8::StoreGroup(packed, row, h0 * Values + i, value);
        }
        return;
    }
    const int tiles = (count + 15) / 16;
    const int active = min(tiles, count <= 512 ? 8 : Splits);
    if (split >= active) return;
    const int begin = tiles * split / active * 16;
    const int end = tiles * (split + 1) / active * 16;
    const int tokens = end - begin;
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    auto *part = scratch + (size_t)work * Scratch;
    auto *stats = part + Partial;
    for (int i = t; i < tokens; i += Threads) {
        const int slot = begin + i;
        const int key = slot < count ? (r.indices ? r.indices[slot] : slot) : -1;
        s.valid[i] = key >= 0 && key < length ? key : -1;
    }
    __syncthreads();
    for (int i = t; i < tokens * 24; i += Threads) {
        const int k = s.valid[i / 24], d = i % 24 * 8;
        const auto *src = static_cast<const BF16 *>(r.key);
        if (k >= 0) src += (size_t)FastllmNaivePhysicalRow(r, k) * r.keyStride + kh * Dim + d;
        naive_dsa_decode::Copy<true>(&s.key[i / 24][d], src, k >= 0);
    }
    for (int i = t; i < tokens * 16; i += Threads) {
        const int k = s.valid[i / 16], d = i % 16 * 8;
        const auto *src = static_cast<const BF16 *>(r.value);
        if (k >= 0) src += (size_t)FastllmNaivePhysicalRow(r, k, true) * kvHeads * Values + kh * Values + d;
        naive_dsa_decode::Copy<true>(&s.value[i / 16][d], src, k >= 0);
    }
    for (int i = t; i < Heads * 24; i += Threads)
        naive_dsa_decode::Copy<true>(&s.query[i / 24][i % 24 * 8],
            query + ((size_t)row * heads + h0 + i / 24) * Dim + i % 24 * 8, true);
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (warp < tokens / 16) {
        float acc[2][4] = {};
        #pragma unroll
        for (int k = 0; k < Dim; k += 16) {
            uint32_t a[4];
            // The upper eight MMA rows duplicate the lower eight and are discarded.
            Load4(a, &s.query[lane & 7][k + (lane >> 4) * 8]);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                uint32_t b[2];
                Load2(b, &s.key[warp * 16 + n * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                Mma(acc[n], a, b);
            }
        }
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int token = warp * 16 + n * 8 + (lane & 3) * 2;
            // Keep FP32 QK accumulators through scaling. The only probability
            // rounding is BF16 immediately before the PV Tensor Core operation.
            for (int j = 0; j < 2; ++j)
                s.scores[token + j][lane >> 2] = s.valid[token + j] >= 0
                    ? acc[n][j] * rsqrtf(float(Dim)) : -INFINITY;
        }
    }
    __syncthreads(); // key storage is now free for probabilities.
    if (warp < 4) {
        const int h = warp * 2 + (lane & 1), token = lane / 2;
        float maximum = -INFINITY;
        for (int k = token; k < tokens; k += 16) maximum = fmaxf(maximum, s.scores[k][h]);
        for (int d = 16; d >= 2; d /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
        float sum = 0;
        for (int k = token; k < tokens; k += 16) {
            BF16 p = __float2bfloat16(maximum == -INFINITY ? 0.f : expf(s.scores[k][h] - maximum));
            s.probability[h][k] = p;
            sum += __bfloat162float(p);
        }
        for (int d = 16; d >= 2; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
        if (lane < 2) {
            stats[(split * Heads + h) * 2] = maximum;
            stats[(split * Heads + h) * 2 + 1] = sum;
        }
    }
    __syncthreads();
    float acc[2][4] = {};
    for (int k = 0; k < tokens; k += 16) {
        uint32_t a[4];
        Load4(a, &s.probability[lane & 7][k + (lane >> 4) * 8]);
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            uint32_t b[2];
            Load2Transpose(b, &s.value[k + (lane & 15)][warp * 16 + n * 8]);
            Mma(acc[n], a, b);
        }
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < 2; ++j)
            part[(split * Heads + (lane >> 2)) * Values + warp * 16 + n * 8 + (lane & 3) * 2 + j] = acc[n][j];
    // Publish every lane's partial before the CTA's release operation. The
    // acquire/release chain makes all preceding splits visible to the last CTA.
    __syncthreads();
    __shared__ int last;
    if (!t) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
        last = counter.fetch_add(1, cuda::memory_order_acq_rel) == unsigned(active - 1);
    }
    __syncthreads();
    if (!last) return;
    float maximum = lane < active ? stats[(lane * Heads + warp) * 2] : -INFINITY;
    for (int d = 16; d; d /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    if (sink) maximum = fmaxf(maximum, sink[h0 + warp]);
    // Reuse score storage; each head has its own sixteen correction factors.
    float *correction = reinterpret_cast<float *>(s.scores) + warp * Splits;
    if (lane < active) correction[lane] = maximum == -INFINITY ? 0.f
        : expf(stats[(lane * Heads + warp) * 2] - maximum);
    __syncwarp();
    float denominator = 0;
    if (!lane) {
        for (int k = 0; k < active; ++k)
            denominator = fmaf(stats[(k * Heads + warp) * 2 + 1], correction[k], denominator);
        if (sink) denominator += expf(sink[h0 + warp] - maximum);
    }
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    for (int d = lane; d < Values; d += 32) {
        float v = 0;
        for (int k = 0; k < active; ++k)
            v = fmaf(part[(k * Heads + warp) * Values + d], correction[k], v);
        const BF16 value[] = {__float2bfloat16(denominator > 0 ? v / denominator : 0.f)};
        output[((size_t)row * heads + h0 + warp) * Values + d] = value[0];
        if constexpr (Pack) fastllm_mxfp8::StoreGroup(packed, row, (h0 + warp) * Values + d, value);
    }
    __syncthreads();
    if (!t) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(completion[work]);
        counter.store(0, cuda::memory_order_release);
    }
#endif
}
template<bool Pack> __global__ __launch_bounds__(Threads, 1) void Attention(const BF16 *query,
        FastllmNaiveDSABatch batch, const float *sink, BF16 *output,
        float *scratch, unsigned *completion, int heads, int kvHeads, FastllmMxfp8Activation packed) {
    Attend<Pack>(query, batch.rows[blockIdx.y], sink, output, scratch, completion, heads, kvHeads,
        blockIdx.y, blockIdx.z, blockIdx.x, packed);
}
} // namespace naive_dsa_split
