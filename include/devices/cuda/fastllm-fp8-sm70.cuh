#pragma once
// Adapted from dollarwong/ninfer-v100 (Apache-2.0), commit
// f17d37fc98b78cc025a1037e8adb1f4fb83813fb:
// src/ops/linear/fp8/fp8_volta_qpn_gemm.cuh and ops/common/volta_mma.cuh.
// Modified for FastLLM: FP32 per-row scales, FP16 bias/output, row-major
// non-destructive weights, bounded small-token dispatch and SM70-only guard.
// The Volta fragment mapping in NInfer traces back to llama.cpp (MIT).
// See third_party/ninfer-fp8-sm70/NOTICE and accompanying licenses.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
namespace fastllm { namespace fp8sm70 {
struct Output {
    half *data;
    const half *bias;
    int cols;
    __device__ __forceinline__ void store(int col, int row, float value) const {
        if (bias) value += __half2float(bias[col]);
        data[(int64_t)row * cols + col] = __float2half_rn(value);
    }
};
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
__device__ __forceinline__ void Mma(float (&d)[8], unsigned a0, unsigned a1,
                                               unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

// Four adjacent E4M3 codes -> two half2 of adjacent k, each value carrying a 2^-8 factor.
//
// The permute puts code j at byte 0 and code j+1 at byte 2 of a word, so one shift/mask pair
// builds both fp16 lanes at once. The bytes it leaves in lanes 1 and 3 are garbage on purpose:
// after the shifts they land outside both masks, so zeroing them would be wasted work.
__device__ __forceinline__ void fp8_decode_quad(std::uint32_t word, half2& lo, half2& hi) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x3F803F80u;
    const std::uint32_t p0        = __byte_perm(word, word, 0x0110); // [b0, b1, b1, b0]
    const std::uint32_t p1        = __byte_perm(word, word, 0x2332); // [b2, b3, b3, b2]
    const std::uint32_t v0        = ((p0 << 8) & kSign) | ((p0 << 7) & kExpM);
    const std::uint32_t v1        = ((p1 << 8) & kSign) | ((p1 << 7) & kExpM);
    lo                            = *reinterpret_cast<const half2*>(&v0);
    hi                            = *reinterpret_cast<const half2*>(&v1);
}

// Eight warps split K; each CTA computes up to eight tokens and 32 columns.
// A warp-local transpose coalesces row-major FP8 loads without repacking weights.
static __global__ __launch_bounds__(256, 4)
void RowKernel(const uint8_t *__restrict__ codes, const float *__restrict__ scales,
               const half *__restrict__ x, int n, int k, int t, Output output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int kWarps = 8, kRows = 8, kCols = 32;
    __shared__ float partial[kWarps][kRows * kCols];
    __shared__ uint4 staged[kWarps][8 * 33];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int blocks = k / 128, perWarp = blocks / kWarps;
    const int begin = warp * perWarp;
    const int end = warp == kWarps - 1 ? blocks : begin + perWarp;
    float c[8] = {};
    for (int b = begin; b < end; ++b) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int local = j * 4 + lane / 8;
            const int global = blockIdx.x * kCols + local;
            const int owner = (local / 8) * 4 + (local & 3) + ((local & 4) ? 16 : 0);
            const auto *src = reinterpret_cast<const uint4*>(
                codes + int64_t(global < n ? global : 0) * k + b * 128 + (lane & 7) * 16);
            staged[warp][(lane & 7) * 33 + owner] = __ldg(src);
        }
        __syncwarp();
        uint4 cw[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) cw[e] = staged[warp][e * 33 + lane];
        __syncwarp();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t words[4] = {cw[e].x, cw[e].y, cw[e].z, cw[e].w};
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                half2 b4[4];
                fp8_decode_quad(words[2 * u], b4[0], b4[1]);
                fp8_decode_quad(words[2 * u + 1], b4[2], b4[3]);
                const unsigned *B = reinterpret_cast<const unsigned*>(b4);
                const int offset = b * 128 + e * 16 + u * 8;
                uint4 a = make_uint4(0, 0, 0, 0);
                if (r < t) a = *reinterpret_cast<const uint4*>(x + int64_t(r) * k + offset);
                Mma(c, a.x, a.y, B[0], B[1]);
                Mma(c, a.z, a.w, B[2], B[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[warp][row * kCols + qp * 8 + cl] = c[i];
    }
    __syncthreads();
    const int row = threadIdx.x / kCols, col = threadIdx.x % kCols;
    const int outputCol = blockIdx.x * kCols + col;
    if (row < t && outputCol < n) {
        float sum = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += partial[w][threadIdx.x];
        output.store(outputCol, row, sum * (scales[outputCol] * 256.0f));
    }
#endif
}

// Skinny GDN projection: the four independent Volta quad-pairs split K
// instead of output channels. For each of eight output channels they read
// adjacent 32-byte slices of a 128-byte row, coalescing the original layout
// directly. This removes the shared-memory transpose and its 33 KB staging
// buffer. Only the FP32 partial sums need shared memory (8 KB).
static __global__ __launch_bounds__(256, 4)
void GdnQuadSplitKernel(const uint8_t *__restrict__ codes,
                        const float *__restrict__ scales,
                        const half *__restrict__ x, int n, int k, int t,
                        Output output) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int Warps = 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col = blockIdx.x * 8 + r;
    const int blocks = k / 128, perWarp = blocks / Warps;
    const int begin = warp * perWarp;
    const int end = warp == Warps - 1 ? blocks : begin + perWarp;
    const uint8_t *weights = codes + int64_t(col < n ? col : 0) * k;
    float c[8] = {};
    for (int b = begin; b < end; ++b) {
        uint4 words[2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
            words[e] = __ldg(reinterpret_cast<const uint4*>(weights + b * 128 + qp * 32 + e * 16));
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t w[4] = {words[e].x, words[e].y, words[e].z, words[e].w};
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                half2 b4[4];
                fp8_decode_quad(w[2*u], b4[0], b4[1]);
                fp8_decode_quad(w[2*u+1], b4[2], b4[3]);
                const unsigned *B = reinterpret_cast<const unsigned*>(b4);
                const int offset = b * 128 + qp * 32 + e * 16 + u * 8;
                uint4 a = make_uint4(0,0,0,0);
                if (r < t) a = *reinterpret_cast<const uint4*>(x + int64_t(r) * k + offset);
                Mma(c, a.x, a.y, B[0], B[1]);
                Mma(c, a.z, a.w, B[2], B[3]);
            }
        }
    }
    __shared__ float partial[Warps][8 * 32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[warp][row * 32 + qp * 8 + cl] = c[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64; i += Warps * 32) {
        const int row = i / 8, cl = i % 8, outputCol = blockIdx.x * 8 + cl;
        if (row < t && outputCol < n) {
            float sum = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
#pragma unroll
                for (int q = 0; q < 4; ++q) sum += partial[w][row * 32 + q * 8 + cl];
            }
            output.store(outputCol, row, sum * (scales[outputCol] * 256.0f));
        }
    }
#endif
}

inline bool Supported(int K, int N, int T, int device) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#endif
    if (T < 1 || T > 8 || N <= 0 || K < 1024 || K % 128) return false;
    static thread_local std::map<int, bool> devices;
    auto it = devices.find(device);
    if (it == devices.end()) {
        cudaDeviceProp prop{};
        bool ok = cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
                  prop.major == 7 && prop.minor == 0;
        if (ok) {
            // A binary built only for an older architecture must not select
            // the empty non-SM70 body. Check PTX too: compute_60 PTX may be
            // JIT-compiled to sm_70 while retaining its compile-time guards.
            cudaFuncAttributes attributes{};
            const cudaError_t status = cudaFuncGetAttributes(&attributes, RowKernel);
            ok = status == cudaSuccess && attributes.binaryVersion == 70 &&
                 attributes.ptxVersion == 70;
            if (status != cudaSuccess) cudaGetLastError();
        }
        it = devices.emplace(device, ok).first;
    }
    return it->second;
}
inline void Launch(const half *x, const uint8_t *w, const float *scales,
                   const half *bias, half *out, int K, int N, int T,
                   cudaStream_t stream) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    const Output output{out, bias, N};
    if (K == 5120 && N == 16384) {
        GdnQuadSplitKernel<<<(N + 7) / 8, 256, 0, stream>>>(w, scales, x, N, K, T, output);
    } else {
        RowKernel<<<(N + 31) / 32, 256, 0, stream>>>(w, scales, x, N, K, T, output);
    }
#endif
}
inline bool Try(const half *x, const uint8_t *w, const float *scales,
                const half *bias, half *out, int K, int N, int T,
                int blockM, int blockK, cudaStream_t stream) {
    const char *flag = std::getenv("FASTLLM_CUDA_FP8_SM70");
    if (flag && (!std::strcmp(flag,"0") || !std::strcmp(flag,"false"))) return false;
    if (blockK != 1 || blockM < K || T < 1 || T > 8 || K < 1024 || K % 128 || N <= 0)
        return false;
    if (!x || !w || !scales || !out || reinterpret_cast<uintptr_t>(x) % 16 ||
        reinterpret_cast<uintptr_t>(w) % 16 || reinterpret_cast<uintptr_t>(scales) % 4 ||
        reinterpret_cast<uintptr_t>(out) % 2 || (bias && reinterpret_cast<uintptr_t>(bias) % 2))
        return false;
    auto overlap = [](const void *a, size_t as, const void *b, size_t bs) {
        const uintptr_t ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
        return ap < bp + bs && bp < ap + as;
    };
    const size_t bytes = size_t(N) * T * sizeof(half);
    if (overlap(out, bytes, x, size_t(K) * T * sizeof(half)) ||
        overlap(out, bytes, w, size_t(N) * K) ||
        overlap(out, bytes, scales, size_t(N) * sizeof(float)) ||
        (bias && overlap(out, bytes, bias, size_t(N) * sizeof(half)))) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || !Supported(K,N,T,device)) return false;
    Launch(x,w,scales,bias,out,K,N,T,stream);
    return true;
}
} }
