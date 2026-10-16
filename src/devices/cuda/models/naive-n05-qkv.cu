#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-qkv.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <unordered_set>

namespace naive_qkv {
namespace {
using BF = __nv_bfloat16;
constexpr int kInput = 4096, kQuery = 1536, kQK = 1728, kOutput = 1856;
union __align__(16) union_bf16_8 {
    uint4 in;
    BF out[8];
};

// Pair both rotary halves inside each four-channel GEMV CTA and WGMMA tile.
__device__ int Channel(int c) {
    return c < kQK && c % 192 < 64 ? c / 192 * 192 + (c % 64) / 2 + (c % 2) * 32 : c;
}
__device__ float Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ BF Rotate(BF value, BF peer, int d, float position, float theta) {
    float angle = position * powf(theta, -2.0f * (d % 32) / 64.0f);
    float cs = Round(cosf(angle)), sn = Round(sinf(angle));
    float a = Round((float)value * cs), b = Round((float)peer * sn);
    return __float2bfloat16_rn(d < 32 ? a - b : a + b);
}
__device__ int CacheRow(Params p, int b) {
    int row = p.lengths[b] - 1;
    if (row < 0)
        return -1;
    if (p.ring)
        row &= 127;
    return row < p.keyRows && row < p.valueRows ? row : -1;
}
__device__ int PhysicalRow(Params p, int b, int row, bool value) {
    if (!p.pageLen)
        return row;
    const int *pages = (const int *)p.cache[b * 4 + (value ? 3 : 2)];
    return pages[row / p.pageLen] * p.pageLen + row % p.pageLen;
}
__device__ void Store(Params p, int b, int logical, BF value, BF peer) {
    int row = CacheRow(p, b);
    if (row < 0) {
        int c = Channel(logical);
        if (c < kQuery) p.q[b * kQuery + c] = __float2bfloat16(0.f);
        return;
    }
    int c = Channel(logical), d = c % 192;
    if (c < kQK && d < 64)
        value = Rotate(value, peer, d, p.positions[b], p.theta);
    if (c < kQuery)
        p.q[b * kQuery + c] = value;
    else if (c < kQK) {
        FastllmNaiveKVWriteRow key{(void *)p.cache[b * 4], p.fp8};
        const int indexWidth = p.indexKey ? (p.scaledIndexer ? FastllmNaiveIndexerBytes / (p.fp8 ? 1 : 2) : 128) : 0;
        key[(size_t)PhysicalRow(p, b, row, false) * (192 + indexWidth) + c - kQuery] = value;
    } else {
        FastllmNaiveKVWriteRow v{(void *)p.cache[b * 4 + 1], p.fp8};
        v[(size_t)PhysicalRow(p, b, row, true) * 128 + c - kQK] =
            __float2bfloat16_rn((float)value * (float)p.valueScale);
    }
}
__device__ void StoreIndex(Params p, int b) {
    int d = threadIdx.x, row = CacheRow(p, b);
    if (!p.indexKey || row < 0)
        return;
    const BF *src = p.indexKey + b * 128;
    BF value = d < 128 ? src[d] : __float2bfloat16(0.f);
    if (d < 64) value = Rotate(value, src[d ^ 32], d, p.positions[b], p.theta);
    FastllmNaiveKVWriteRow key{(void *)p.cache[b * 4], p.fp8};
    if (p.scaledIndexer) {
        auto dst = key + (size_t)PhysicalRow(p, b, row, false) *
            (192 + FastllmNaiveIndexerBytes / (p.fp8 ? 1 : 2)) + 192;
        FastllmNaiveStoreIndexer((float)value, dst.data);
    } else if (d < 128) key[(size_t)PhysicalRow(p, b, row, false) * 320 + 192 + d] = value;
}
__global__ void QkvGemv(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, Params p) {
    if (blockIdx.x == kOutput / 4) {
        StoreIndex(p, 0);
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int logical = blockIdx.x * 4 + warp;
    const int row = Channel(logical);
    __shared__ BF sums[4];
    float values[8] = {}, correction[8] = {};
#pragma unroll
    for (int part = 0; part < 8; ++part) {
#pragma unroll
        for (int column = (lane + part * 32) * 8; column < kInput; column += 256 * 8) {
            union_bf16_8 a, b;
            b.in = *reinterpret_cast<const uint4 *>(weight + (size_t)row * kInput + column);
            a.in = *reinterpret_cast<const uint4 *>(input + column);
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                sum += __bfloat162float(a.out[i]) * __bfloat162float(b.out[i]);
            values[part] += sum;
        }
    }
#pragma unroll
    for (int stride = 4; stride > 0; stride >>= 1) {
#pragma unroll
        for (int part = 0; part < stride; ++part) {
            float other = values[part + stride] - correction[part];
            float sum = values[part] + other;
            correction[part] = (sum - values[part]) - other;
            values[part] = sum;
        }
    }
    float value = values[0], diff = correction[0];
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        float peer = __shfl_down_sync(0xffffffffu, value, stride);
        if (lane < stride) {
            float other = peer - diff;
            float sum = value + other;
            diff = (sum - value) - other;
            value = sum;
        }
    }
    if (lane == 0)
        sums[warp] = __float2bfloat16_rn(value);
    __syncthreads();
    if (lane == 0)
        Store(p, 0, logical, sums[warp], sums[warp ^ 1]);
}

__device__ void Copy(void *dst, const void *src, bool valid) {
    unsigned address = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
}
__global__ void ReduceQkv(const float *partial, Params p, int rows, int splits) {
    int logical = blockIdx.x * 256 + threadIdx.x, batch = blockIdx.y;
    if (blockIdx.x == (kOutput + 255) / 256) {
        StoreIndex(p, batch);
        return;
    }
    if (logical >= kOutput)
        return;
    float value = 0, peer = 0;
    for (int s = 0; s < splits; ++s) {
        size_t offset = ((size_t)s * rows + batch) * kOutput;
        value += partial[offset + logical];
        peer += partial[offset + (logical ^ 1)];
    }
    // Projection rounds to BF16 before rotary products, just like Linear + RoPE.
    Store(p, batch, logical, __float2bfloat16_rn(value), __float2bfloat16_rn(peer));
}
__device__ __forceinline__ unsigned long long Desc(const void *ptr) {
    return ((__cvta_generic_to_shared(ptr) >> 4) & 0x3fff) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void Wgmma16(float (&c)[8], unsigned long long a, unsigned long long b) {
    asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0;}"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
                   "+f"(c[7])
                 : "l"(a), "l"(b));
}
__device__ __forceinline__ void Wgmma32(float (&c)[16], unsigned long long a, unsigned long long b) {
    asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1,0,0;}"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
                   "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]),
                   "+f"(c[14]), "+f"(c[15])
                 : "l"(a), "l"(b));
}
__device__ __forceinline__ int Offset(int rows, int r, int k) {
    return (k / 64) * rows * 64 + r * 64 + ((k % 64) ^ ((r % 8) * 8));
}
template <int BM>
__device__ __forceinline__ void LoadWG(BF *a, BF *x, const BF *w, const BF *in, int M, int kk, bool valid) {
#pragma unroll
    for (int i = 0; i < 64 * 128 / 8 / 128; ++i) {
        int t = threadIdx.x + i * 128, r = t / 16, c = t % 16 * 8;
        Copy(a + Offset(64, r, c), w + (size_t)Channel(blockIdx.x * 64 + r) * kInput + kk + c, valid);
    }
#pragma unroll
    for (int i = 0; i < BM * 128 / 8 / 128; ++i) {
        int t = threadIdx.x + i * 128, r = t / 16, c = t % 16 * 8, b = blockIdx.y * BM + r;
        Copy(x + Offset(BM, r, c), in + (size_t)(b < M ? b : 0) * kInput + kk + c, valid && b < M);
    }
    asm volatile("cp.async.commit_group;");
}
// Three shared-memory stages overlap async loads and WGMMA. The wait_group(1)
// retires the preceding WGMMA group before its stage is recycled; wait_group(0)
// retires the last group before reusing shared storage for the output transpose.
template <int BM, int SPLIT>
__global__ __launch_bounds__(128) void QkvWG(const BF *in, const BF *w, int M, float *partial) {
    extern __shared__ __align__(1024) BF wg_storage[];
    BF(*a)[64 * 128] = reinterpret_cast<BF(*)[64 * 128]>(wg_storage);
    BF(*x)[BM * 128] = reinterpret_cast<BF(*)[BM * 128]>(wg_storage + 3 * 64 * 128);
    constexpr int iters = kInput / SPLIT / 128;
    float acc[BM / 2] = {};
    LoadWG<BM>(a[0], x[0], w, in, M, blockIdx.z * (kInput / SPLIT), true);
    LoadWG<BM>(a[1], x[1], w, in, M, blockIdx.z * (kInput / SPLIT) + 128, true);
    for (int it = 0; it < iters; ++it) {
        int stage = it % 3, next = (it + 2) % 3;
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            auto ad = Desc(a[stage] + (k / 64) * 64 * 64 + k % 64),
                 bd = Desc(x[stage] + (k / 64) * BM * 64 + k % 64);
            if constexpr (BM == 16)
                Wgmma16(acc, ad, bd);
            else
                Wgmma32(acc, ad, bd);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 1;" ::: "memory");
        __syncthreads();
        LoadWG<BM>(a[next], x[next], w, in, M,
                   blockIdx.z * (kInput / SPLIT) + (it + 2 < iters ? (it + 2) * 128 : 0), it + 2 < iters);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    float *tile = reinterpret_cast<float *>(wg_storage);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < BM / 2; ++i) {
        int n = warp * 16 + lane / 4 + (i / 2 % 2) * 8, b = (i / 4) * 8 + (lane % 4) * 2 + i % 2;
        tile[b * 64 + n] = acc[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BM * 64; i += 128) {
        int b = blockIdx.y * BM + i / 64, n = blockIdx.x * 64 + i % 64;
        if (b < M) {
            partial[((size_t)blockIdx.z * M + b) * kOutput + n] = tile[i];
        }
    }
}

template <int Batch, int Split> void Matmul(const BF *input, const BF *weight, int rows, float *scratch) {
    constexpr int sharedBytes = 3 * (64 + Batch) * 128 * sizeof(BF);
    QkvWG<Batch, Split><<<dim3(kOutput / 64, (rows + Batch - 1) / Batch, Split), 128, sharedBytes>>>(
        input, weight, rows, scratch);
}
} // namespace

bool Prepare() {
    if (FastllmCudaRuntimeArch() != 90)
        return false;
    static thread_local std::unordered_set<int> initialized;
    int device = FastllmCudaGetDevice();
    if (initialized.count(device))
        return true;
    // First use must warm up before graph capture, as do the projection fallbacks.
    if (FastllmCudaGraphIsCapturingFast())
        return false;
    auto set = [](auto kernel, int batch) {
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    3 * (64 + batch) * 128 * sizeof(BF)) == cudaSuccess;
    };
    if (!set(QkvWG<16, 8>, 16) || !set(QkvWG<32, 8>, 32) || !set(QkvWG<32, 4>, 32) || !set(QkvWG<32, 2>, 32))
        return false;
    initialized.insert(device);
    return true;
}

void Launch(const BF *input, const BF *weight, Params p, int rows, float *scratch) {
    if (rows == 1) {
        // Same compensated FP32 reduction as the existing single-row BF16 GEMV.
        QkvGemv<<<kOutput / 4 + (p.indexKey != nullptr), 128>>>(input, weight, p);
        return;
    }
    int splits = rows <= 32 ? 8 : rows <= 64 ? 4 : 2;
    if (rows <= 16)
        Matmul<16, 8>(input, weight, rows, scratch);
    else if (rows <= 32)
        Matmul<32, 8>(input, weight, rows, scratch);
    else if (rows <= 64)
        Matmul<32, 4>(input, weight, rows, scratch);
    else
        Matmul<32, 2>(input, weight, rows, scratch);
    ReduceQkv<<<dim3((kOutput + 255) / 256 + (p.indexKey != nullptr), rows), 256>>>(scratch, p, rows, splits);
}
} // namespace naive_qkv
