#pragma once
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <map>
#include <stdint.h>
namespace fastllm {
namespace fp8small {
// Check the selected specialization, including 512-thread variants, instead of
// assuming support from a different kernel or a device architecture whitelist.
template <auto Function, int Threads> inline bool KernelAvailable(int device) {
    static thread_local std::map<int, bool> available;
    auto found = available.find(device);
    if (found == available.end()) {
        cudaFuncAttributes attributes{};
        auto status = cudaFuncGetAttributes(&attributes, Function);
        if (status != cudaSuccess)
            cudaGetLastError();
        found = available.emplace(device, status == cudaSuccess &&
                                  attributes.maxThreadsPerBlock >= Threads).first;
    }
    return found->second;
}

// The wider reduction tile was measured for long-K, wide-output M=8
// matrices on SM120. Cache the architecture query per device.
inline bool UseWideMma16(int device) {
    static thread_local std::map<int, bool> supported;
    auto found = supported.find(device);
    if (found == supported.end()) {
        int major = 0, minor = 0;
        bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess;
        if (!ok) cudaGetLastError();
        found = supported.emplace(device, ok && major == 12 && minor == 0).first;
    }
    return found->second;
}

// Callers validate dense row-scaled FP8, 1..8 rows, K divisible by 256,
// and non-overlapping input/output buffers before dispatch. No global workspace
// or reordered weight copy is allocated by these kernels.
// Reuse each decoded weight across a compile-time token tile. Accumulate
// independently in FP32; round the projection before optional residual
// addition.
template <class T, int Tokens, int Warps = 4, int Rows = 2, int Values = 8, bool Add = false, class BiasT = T>
__device__ __forceinline__ void SimtBody(const T *__restrict__ x, const uint8_t *__restrict__ w,
                                         const float *__restrict__ scales, const BiasT *__restrict__ bias,
                                         T *__restrict__ y, int K, int N) {
    static_assert(Tokens >= 1 && Tokens <= 8);
    static_assert(Rows == 1 || Rows % 2 == 0);
    static_assert(Values == 8 || Values == 16);
    const int lane = threadIdx.x % 32, row = (blockIdx.x * Warps + threadIdx.x / 32) * Rows;
    constexpr int Chains = Add ? 1 : 2;
    float acc[Tokens][Rows][Chains] = {};
#pragma unroll(Tokens <= 5 || Add ? 2 : 1)
    for (int base = lane * Values; base < K; base += 32 * Values) {
        uint32_t codes[Rows][Values / 4];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if constexpr (Values == 16) {
                uint4 p = {};
                if (row + r < N)
                    p = (Tokens >= 6 && !Add
                             ? *reinterpret_cast<const uint4 *>(w + size_t(row + r) * K + base)
                             : __ldcs(reinterpret_cast<const uint4 *>(w + size_t(row + r) * K + base)));
                codes[r][0] = p.x;
                codes[r][1] = p.y;
                codes[r][2] = p.z;
                codes[r][3] = p.w;
            } else if constexpr (Values == 8) {
                uint2 p = {};
                if (row + r < N)
                    p = (Tokens >= 6 && !Add
                             ? *reinterpret_cast<const uint2 *>(w + size_t(row + r) * K + base)
                             : __ldcs(reinterpret_cast<const uint2 *>(w + size_t(row + r) * K + base)));
                codes[r][0] = p.x;
                codes[r][1] = p.y;
            }
        }
#pragma unroll
        for (int pair = 0; pair < Values / 2; ++pair) {
            float2 weight[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                __nv_fp8x2_e4m3 c;
                c.__x = uint16_t(codes[r][pair / 2] >> ((pair % 2) * 16));
                weight[r] = static_cast<float2>(c);
            }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                uint32_t bits =
                    __ldg(reinterpret_cast<const uint32_t *>(x + size_t(t) * K + base + pair * 2));
                float2 a;
                if constexpr (__is_same(T, half))
                    a = __half22float2(*reinterpret_cast<half2 *>(&bits));
                else
                    a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&bits));
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    acc[t][r][0] = fmaf(a.x, weight[r].x, acc[t][r][0]);
                    acc[t][r][Chains - 1] = fmaf(a.y, weight[r].y, acc[t][r][Chains - 1]);
                }
            }
        }
    }
    if constexpr (!Add) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float v = acc[t][r][0] + acc[t][r][1];
#pragma unroll
                for (int d = 16; d; d /= 2)
                    v += __shfl_down_sync(0xffffffff, v, d);
                if (lane == 0 && row + r < N)
                    y[size_t(t) * N + row + r] = T(v * scales[row + r] + (bias ? float(bias[row + r]) : 0.f));
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float v = acc[t][r][0];
#pragma unroll
                for (int d = 16; d; d /= 2)
                    v += __shfl_down_sync(0xffffffff, v, d);
                acc[t][r][0] = v;
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
#pragma unroll
                for (int r = 0; r < Rows; r += 2) {
                    if (row + r >= N)
                        continue;
                    T p0 = T(acc[t][r][0] * scales[row + r] + (bias ? float(bias[row + r]) : 0.f));
                    if constexpr (Rows % 2 == 0) {
                        if (N % 2 == 0 && reinterpret_cast<uintptr_t>(y) % 4 == 0 && row + r + 1 < N) {
                            T p1 = T(acc[t][r + 1][0] * scales[row + r + 1] +
                                     (bias ? float(bias[row + r + 1]) : 0.f));
                            uint32_t bits;
                            if constexpr (__is_same(T, half)) {
                                half2 p = __halves2half2(p0, p1);
                                if constexpr (Add)
                                    p = __hadd2(
                                        p, *reinterpret_cast<const half2 *>(y + size_t(t) * N + row + r));
                                bits = *reinterpret_cast<uint32_t *>(&p);
                            } else {
                                __nv_bfloat162 p = __halves2bfloat162(p0, p1);
                                if constexpr (Add)
                                    p = __hadd2(p, *reinterpret_cast<const __nv_bfloat162 *>(
                                                       y + size_t(t) * N + row + r));
                                bits = *reinterpret_cast<uint32_t *>(&p);
                            }
                            *reinterpret_cast<uint32_t *>(y + size_t(t) * N + row + r) = bits;
                            continue;
                        }
                    }
                    if constexpr (Add)
                        y[size_t(t) * N + row + r] = T(float(y[size_t(t) * N + row + r]) + float(p0));
                    else
                        y[size_t(t) * N + row + r] = p0;
                    if constexpr (Rows % 2 == 0)
                        if (row + r + 1 < N) {
                            T p1 = T(acc[t][r + 1][0] * scales[row + r + 1] +
                                     (bias ? float(bias[row + r + 1]) : 0.f));
                            if constexpr (Add)
                                y[size_t(t) * N + row + r + 1] =
                                    T(float(y[size_t(t) * N + row + r + 1]) + float(p1));
                            else
                                y[size_t(t) * N + row + r + 1] = p1;
                        }
                }
            }
        }
    }
}
template <class T, int Tokens, int Warps = 4, int Rows = 2, int Values = 8, bool Add = false, class BiasT = T>
__global__ __launch_bounds__(Warps * 32, 2) void Kernel(const T *x, const uint8_t *w, const float *scales,
                                                        const BiasT *bias, T *y, int K, int N) {
    SimtBody<T, Tokens, Warps, Rows, Values, Add, BiasT>(x, w, scales, bias, y, K, N);
}
// A 16-output-row tile uses K-split warps and m16n8k16 tensor operations.
// FP8 conversion is exact in FP16/BF16; no activation quantization or packed
// weight cache is required. SM80+ uses MMA; older compiled targets use SIMT.
template <class T, int Tokens, int Warps, bool Add = false, class BiasT = T>
__global__ __launch_bounds__(Warps * 32, 2) void MmaKernel(const T *__restrict__ x,
                                                           const uint8_t *__restrict__ w,
                                                           const float *__restrict__ scales,
                                                           const BiasT *__restrict__ bias, T *__restrict__ y,
                                                           int K, int N) {
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, group = lane / 4, part = lane % 4;
    const int row = blockIdx.x * 16 + group;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int base = warp * 64; base < K; base += Warps * 64) {
        uint4 a = {};
        uint4 b = {};
        if (row < N)
            a = *reinterpret_cast<const uint4 *>(w + size_t(row) * K + base + part * 16);
        if (row + 8 < N)
            b = *reinterpret_cast<const uint4 *>(w + size_t(row + 8) * K + base + part * 16);
        // All lanes must select the same source register before the shuffle.
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            uint32_t ax = __shfl_sync(0xffffffff, a.x, group * 4 + step),
                     ay = __shfl_sync(0xffffffff, a.y, group * 4 + step);
            uint32_t az = __shfl_sync(0xffffffff, a.z, group * 4 + step),
                     aw = __shfl_sync(0xffffffff, a.w, group * 4 + step);
            uint32_t bx = __shfl_sync(0xffffffff, b.x, group * 4 + step),
                     by = __shfl_sync(0xffffffff, b.y, group * 4 + step);
            uint32_t bz = __shfl_sync(0xffffffff, b.z, group * 4 + step),
                     bw = __shfl_sync(0xffffffff, b.w, group * 4 + step);
            uint32_t words[4] = {part < 2 ? ax : ay, part < 2 ? bx : by, part < 2 ? az : aw,
                                 part < 2 ? bz : bw};
            uint32_t av[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_fp8x2_e4m3 q;
                q.__x = uint16_t(words[j] >> ((part % 2) * 16));
                __half2 h = static_cast<__half2>(q);
                if constexpr (__is_same(T, half))
                    av[j] = *reinterpret_cast<uint32_t *>(&h);
                else {
                    float2 f = __half22float2(h);
                    __nv_bfloat162 z = __floats2bfloat162_rn(f.x, f.y);
                    av[j] = *reinterpret_cast<uint32_t *>(&z);
                }
            }
            uint32_t v0 = 0, v1 = 0;
            if (group < Tokens) {
                v0 = __ldg(
                    reinterpret_cast<const uint32_t *>(x + size_t(group) * K + base + step * 16 + part * 2));
                v1 = __ldg(reinterpret_cast<const uint32_t *>(x + size_t(group) * K + base + step * 16 +
                                                              part * 2 + 8));
            }
            if constexpr (__is_same(T, half))
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(v0), "r"(v1));
            else
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(v0), "r"(v1));
        }
    }
    __shared__ float partial[Warps][128];
    partial[warp][group * 8 + part * 2] = d0;
    partial[warp][group * 8 + part * 2 + 1] = d1;
    partial[warp][(group + 8) * 8 + part * 2] = d2;
    partial[warp][(group + 8) * 8 + part * 2 + 1] = d3;
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * Tokens; i += Warps * 32) {
        int r = i % 16, t = i / 16;
        float v = 0;
#pragma unroll
        for (int widx = 0; widx < Warps; ++widx)
            v += partial[widx][r * 8 + t];
        int n = blockIdx.x * 16 + r;
        if (n < N) {
            T p = T(v * scales[n] + (bias ? float(bias[n]) : 0.f));
            if constexpr (Add)
                y[size_t(t) * N + n] = T(float(y[size_t(t) * N + n]) + float(p));
            else
                y[size_t(t) * N + n] = p;
        }
    }
#else
    // Older targets retain a SIMT implementation with the same 16-row CTA.
    SimtBody<T, Tokens, Warps, 16 / Warps, 8, Add, BiasT>(x, w, scales, bias, y, K, N);
#endif
}
template <class T, int Tokens, bool Add = false, class BiasT = T, bool CheckOnly = false>
inline bool Launch(const T *x, const uint8_t *w, const float *scales, const BiasT *bias, T *y, int K, int N,
                   cudaStream_t stream, int device = 0) {
    // Capability queries and launches use exactly the same shape dispatch.
#define FASTLLM_FP8_SMALL_LAUNCH(Blocks, Threads, ...)                                    \
    do {                                                                                \
        if constexpr (CheckOnly)                                                        \
            return KernelAvailable<__VA_ARGS__, Threads>(device);                        \
        else {                                                                          \
            __VA_ARGS__<<<Blocks, Threads, 0, stream>>>(x, w, scales, bias, y, K, N);       \
            return true;                                                                \
        }                                                                               \
    } while (false)
    if constexpr (Tokens >= 5) {
        if (N < 16384 || Tokens == 8) {
            if ((N > 8192 && N < 16384) || (N <= 8192 && Tokens == 8) ||
                (Tokens == 8 && N >= 16384 && N <= 65536 && K >= 4096 && K <= 32768 &&
                 UseWideMma16(device)))
                FASTLLM_FP8_SMALL_LAUNCH((N + 15) / 16, 512, MmaKernel<T, Tokens, 16, Add, BiasT>);
            else
                FASTLLM_FP8_SMALL_LAUNCH((N + 15) / 16, 256, MmaKernel<T, Tokens, 8, Add, BiasT>);
        }
        if constexpr (Tokens == 5)
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 128, Kernel<T, Tokens, 4, 2, 8, Add, BiasT>);
        else
            FASTLLM_FP8_SMALL_LAUNCH((N + 15) / 16, 128, Kernel<T, Tokens, 4, 4, 8, Add, BiasT>);
    } else if constexpr (Tokens <= 2) {
        if ((N <= 8192 || N >= 16384) && K % 512 == 0)
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 128, Kernel<T, Tokens, 4, 2, 16, Add, BiasT>);
        else
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 256, Kernel<T, Tokens, 8, 1, 8, Add, BiasT>);
    } else if constexpr (Tokens == 3) {
        if (N <= 8192)
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 256, Kernel<T, Tokens, 8, 1, 8, Add, BiasT>);
        else if (N >= 16384)
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 128, Kernel<T, Tokens, 4, 2, 8, Add, BiasT>);
        else if (K % 512 == 0)
            FASTLLM_FP8_SMALL_LAUNCH((N + 15) / 16, 128, Kernel<T, Tokens, 4, 4, 16, Add, BiasT>);
        else
            FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 128, Kernel<T, Tokens, 4, 2, 8, Add, BiasT>);
    } else {
        if (N >= 16384) {
            FASTLLM_FP8_SMALL_LAUNCH((N + 15) / 16, 128, Kernel<T, Tokens, 4, 4, 8, Add, BiasT>);
        }
        FASTLLM_FP8_SMALL_LAUNCH((N + 7) / 8, 128, Kernel<T, Tokens, 4, 2, 8, Add, BiasT>);
    }
#undef FASTLLM_FP8_SMALL_LAUNCH
}
template <class T, bool Add = false, class BiasT = T, bool CheckOnly = false>
inline bool Dispatch(const T *x, const uint8_t *w, const float *scales, const BiasT *bias, T *y, int K, int N,
                     int tokens, cudaStream_t stream, int device = 0) {
#define FASTLLM_FP8_SMALL_CASE(M)                                                                            \
    case M:                                                                                                  \
        return Launch<T, M, Add, BiasT, CheckOnly>(x, w, scales, bias, y, K, N, stream, device);
    switch (tokens) {
        FASTLLM_FP8_SMALL_CASE(1)
        FASTLLM_FP8_SMALL_CASE(2)
        FASTLLM_FP8_SMALL_CASE(3) FASTLLM_FP8_SMALL_CASE(4) FASTLLM_FP8_SMALL_CASE(5)
            FASTLLM_FP8_SMALL_CASE(6) FASTLLM_FP8_SMALL_CASE(7) FASTLLM_FP8_SMALL_CASE(8)
    }
#undef FASTLLM_FP8_SMALL_CASE
    return false;
}
template <class T, bool Add = false, class BiasT = T>
inline bool CanRun(int device, int K, int N, int tokens) {
    const char *flag = std::getenv("FASTLLM_CUDA_FP8_SMALL_T");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    return Dispatch<T, Add, BiasT, true>(nullptr, nullptr, nullptr, nullptr, nullptr,
                                       K, N, tokens, nullptr, device);
}
} // namespace fp8small
} // namespace fastllm
