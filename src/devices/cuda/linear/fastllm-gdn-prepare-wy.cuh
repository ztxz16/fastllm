#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
namespace fastllm_gdn_wy {
__device__ __forceinline__ int Index(int r, int c) { return r * 68 + c; }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void
Tf32Product(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> &acc, const float *a, int as,
            const float *b, int bs) {
    using namespace nvcuda;
#pragma unroll
    for (int k = 0; k < 16; k += 8) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> av, ah, al;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bv, bh, bl;
        wmma::load_matrix_sync(av, a + k, as);
        wmma::load_matrix_sync(bv, b + k * bs, bs);
#pragma unroll
        for (int i = 0; i < av.num_elements; i++) {
            ah.x[i] = wmma::__float_to_tf32(av.x[i]);
            al.x[i] = wmma::__float_to_tf32(av.x[i] - ah.x[i]);
        }
#pragma unroll
        for (int i = 0; i < bv.num_elements; i++) {
            bh.x[i] = wmma::__float_to_tf32(bv.x[i]);
            bl.x[i] = wmma::__float_to_tf32(bv.x[i] - bh.x[i]);
        }
        wmma::mma_sync(acc, ah, bh, acc);
        wmma::mma_sync(acc, al, bh, acc);
        wmma::mma_sync(acc, ah, bl, acc);
    }
}
#endif
// One CTA owns a complete 64-token chunk. All source values are consumed
// before an aliased output is written; the V product precedes the K product.
template <int ReduceWidth, bool FromKey = false>
__global__ __launch_bounds__(64 * ReduceWidth) void Prepare(const half *attention, const half *vBeta,
                                                            const half *kBeta, const half *g, half *vOut,
                                                            half *kOut, const half *key = nullptr,
                                                            half *rawG = nullptr, half *decay = nullptr) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int Threads = 64 * ReduceWidth;
    extern __shared__ __align__(32) unsigned char storage[];
    float *matrix = reinterpret_cast<float *>(storage);
    float *row = matrix + 64 * 68;
    int chunk = blockIdx.x, t = threadIdx.x;
    if constexpr (FromKey) {
        constexpr int KktWarps = Threads / 32;
        static_assert(ReduceWidth == 4 || ReduceWidth == 8);
        half *ka = reinterpret_cast<half *>(storage), *kb = ka + 64 * 136;
        half *prefix = reinterpret_cast<half *>(storage + 34816);
        for (int i = t; i < 8192; i += Threads) {
            ka[(i / 128) * 136 + i % 128] = kBeta[size_t(chunk) * 8192 + i];
            kb[(i / 128) * 136 + i % 128] = key[size_t(chunk) * 8192 + i];
        }
        if (t < 64)
            prefix[t] = rawG[size_t(chunk) * 64 + t];
        __syncthreads();
        if (!t)
            for (int i = 1; i < 64; i++)
                prefix[i] = __hadd(prefix[i], prefix[i - 1]);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dots[16 / KktWarps];
#pragma unroll
        for (int p = 0; p < 16 / KktWarps; p++) {
            int tile = t / 32 + p * KktWarps, rm = tile / 4 * 16, cn = tile % 4 * 16;
            wmma::fill_fragment(dots[p], 0.f);
#pragma unroll
            for (int q = 0; q < 128; q += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(af, ka + rm * 136 + q, 136);
                wmma::load_matrix_sync(bf, kb + cn * 136 + q, 136);
                wmma::mma_sync(dots[p], af, bf, dots[p]);
            }
        }
        __syncthreads();
// Reuse input storage only after all tensor-core reads have finished.
#pragma unroll
        for (int p = 0; p < 16 / KktWarps; p++) {
            int tile = t / 32 + p * KktWarps;
            wmma::store_matrix_sync(matrix + Index(tile / 4 * 16, tile % 4 * 16), dots[p], 68,
                                    wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = t; i < 4096; i += Threads) {
            int r = i / 64, c = i % 64;
            half d = __float2half(r >= c ? expf(__half2float(prefix[r]) - __half2float(prefix[c])) : 0.f);
            decay[size_t(chunk) * 4096 + i] = d;
            matrix[Index(r, c)] =
                r > c ? __half2float(__hmul(__float2half_rn(matrix[Index(r, c)]), __hneg(d))) : 0.f;
        }
        if (t < 64)
            rawG[size_t(chunk) * 64 + t] = prefix[t];
    } else {
        for (int i = t; i < 4096; i += Threads)
            matrix[Index(i / 64, i % 64)] = __half2float(attention[size_t(chunk) * 4096 + i]);
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 800
    int diag = t / (16 * ReduceWidth), dt = t % (16 * ReduceWidth), j = dt / ReduceWidth,
        lane = dt % ReduceWidth;
    // Four diagonal inverses are independent. Solve their 16 rows together.
    for (int r = 1; r < 16; r++) {
        if (dt < r)
            row[diag * 16 + dt] = matrix[Index(diag * 16 + r, diag * 16 + dt)];
        __syncthreads();
        if (j < r) {
            float sum = 0;
            for (int k = j + 1 + lane; k < r; k += ReduceWidth)
                sum = fmaf(row[diag * 16 + k], matrix[Index(diag * 16 + k, diag * 16 + j)], sum);
            unsigned mask = __activemask();
#pragma unroll
            for (int d = ReduceWidth / 2; d; d >>= 1)
                sum += __shfl_xor_sync(mask, sum, d, ReduceWidth);
            if (!lane)
                matrix[Index(diag * 16 + r, diag * 16 + j)] = row[diag * 16 + j] + sum;
        }
        __syncthreads();
    }
    if (t < 64)
        matrix[Index(t, t)] = 1.f;
    __syncthreads();
    // Columns in a block row are independent if the original lower blocks
    // are retained. Three warps solve them concurrently, with a CTA barrier
    // between successive block rows.
    float *original = row, *workspace = original + 16 * 64;
#pragma unroll
    for (int r = 1; r < 4; r++) {
        for (int i = t; i < 16 * 64; i += Threads)
            original[i] = matrix[Index(r * 16 + i / 64, i % 64)];
        __syncthreads();
        int c = t / 32;
        if (c < r) {
            float *tmp = workspace + c * 256;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> sum;
            wmma::fill_fragment(sum, 0.f);
            for (int j = c; j < r; j++)
                Tf32Product(sum, original + j * 16, 64, matrix + Index(j * 16, c * 16), 68);
            wmma::store_matrix_sync(tmp, sum, 16, wmma::mem_row_major);
            __syncwarp();
            wmma::fill_fragment(sum, 0.f);
            Tf32Product(sum, matrix + Index(r * 16, r * 16), 68, tmp, 16);
            wmma::store_matrix_sync(matrix + Index(r * 16, c * 16), sum, 68, wmma::mem_row_major);
        }
        __syncthreads();
    }
    __syncthreads();
#else
    int j = t / ReduceWidth, lane = t % ReduceWidth;
    for (int r = 1; r < 64; r++) {
        if (t < r)
            row[t] = matrix[Index(r, t)];
        __syncthreads();
        if (j < r) {
            float sum = 0;
            for (int k = j + 1 + lane; k < r; k += ReduceWidth)
                sum = fmaf(row[k], matrix[Index(k, j)], sum);
            unsigned mask = __activemask();
#pragma unroll
            for (int d = ReduceWidth / 2; d; d >>= 1)
                sum += __shfl_xor_sync(mask, sum, d, ReduceWidth);
            if (lane == 0)
                matrix[Index(r, j)] = row[j] + sum;
        }
        __syncthreads();
    }
#endif
    float held[4096 / Threads];
#pragma unroll
    for (int i = 0; i < 4096 / Threads; i++) {
        int q = t + i * Threads;
        held[i] = matrix[Index(q / 64, q % 64)];
#if __CUDA_ARCH__ < 800
        held[i] += (q / 64 == q % 64 ? 1.f : 0.f);
#endif
    }
    __syncthreads();
    half *a = reinterpret_cast<half *>(storage), *b = a + 64 * 72, *expG = b + 64 * 136;
#pragma unroll
    for (int i = 0; i < 4096 / Threads; i++)
        a[((t + i * Threads) / 64) * 72 + (t + i * Threads) % 64] = __float2half_rn(held[i]);
    if (t < 64)
        expG[t] = hexp(g[size_t(chunk) * 64 + t]);
    __syncthreads();
    constexpr int Warps = Threads / 32;
    int warp = t / 32;
    for (int pass = 0; pass < 2; pass++) {
        const half *source = (pass ? kBeta : vBeta) + size_t(chunk) * 8192;
        half *dest = (pass ? kOut : vOut) + size_t(chunk) * 8192;
        for (int i = t; i < 8192; i += Threads)
            b[(i / 128) * 136 + i % 128] = pass ? __hmul(source[i], expG[i / 128]) : source[i];
        __syncthreads();
        for (int tile = warp; tile < 32; tile += Warps) {
            int rm = tile / 8 * 16, cn = tile % 8 * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
                wmma::load_matrix_sync(af, a + rm * 72 + k, 72);
                wmma::load_matrix_sync(bf, b + k * 136 + cn, 136);
                wmma::mma_sync(acc, af, bf, acc);
            }
            // Store through a documented memory layout: fragment element
            // ordering is architecture-dependent across accumulator types.
            float *tileOutput = reinterpret_cast<float *>(storage + 26752) + warp * 256;
            wmma::store_matrix_sync(tileOutput, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = (t & 31) * 2; i < 256; i += 64) {
                half2 value = __floats2half2_rn(tileOutput[i], tileOutput[i + 1]);
                *reinterpret_cast<half2 *>(dest + (rm + i / 16) * 128 + cn + i % 16) = value;
            }
            __syncwarp();
        }
        __syncthreads();
    }
#endif
}
} // namespace fastllm_gdn_wy
