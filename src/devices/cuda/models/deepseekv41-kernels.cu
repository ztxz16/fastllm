//
// DeepSeek-V4.1 专用 CUDA kernel。
//
// 数值语义与 src/devices/cpu/deepseekv41ops.cpp 中的 CPU 参考实现一致；
// 面向 SM86 等无 FP8 tensor core 的设备：稀疏注意力与 indexer 打分在 SM80+ 上用
// BF16 mma（FP32 累加），其余算子为 FP32。SM80 以下或 dtype 不匹配时退回标量实现。
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace {
    using fastllm::Data;
    using fastllm::DataType;
    using fastllm::DataDevice;

    // 在当前 CUDA 设备上准备输出张量（与 deepseekv4-kernels.cu 中的同名逻辑一致）
    bool V41PrepareOutput(Data &output, DataType dataType, const std::vector<int> &dims) {
        output.dataType = dataType;
        output.Resize(dims);
        output.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
        output.Allocate(false);
        return output.cudaData != nullptr;
    }

    // ---------------- 类型转换 ----------------

    template <typename T> __device__ __forceinline__ float V41Load(const T *p, uint64_t i);
    template <> __device__ __forceinline__ float V41Load<float>(const float *p, uint64_t i) { return p[i]; }
    template <> __device__ __forceinline__ float V41Load<__nv_bfloat16>(const __nv_bfloat16 *p, uint64_t i) { return __bfloat162float(p[i]); }
    template <> __device__ __forceinline__ float V41Load<half>(const half *p, uint64_t i) { return __half2float(p[i]); }

    template <typename T> __device__ __forceinline__ void V41Store(T *p, uint64_t i, float v);
    template <> __device__ __forceinline__ void V41Store<float>(float *p, uint64_t i, float v) { p[i] = v; }
    template <> __device__ __forceinline__ void V41Store<__nv_bfloat16>(__nv_bfloat16 *p, uint64_t i, float v) { p[i] = __float2bfloat16_rn(v); }
    template <> __device__ __forceinline__ void V41Store<half>(half *p, uint64_t i, float v) { p[i] = __float2half_rn(v); }

    __device__ __forceinline__ float V41Bf16Round(float v) {
        return __bfloat162float(__float2bfloat16_rn(v));
    }

    __device__ __forceinline__ float V41SigmoidDev(float x) {
        return 1.0f / (1.0f + __expf(-x));
    }

    // 2^ceil(log2(x))
    __device__ __forceinline__ float V41Pow2CeilDev(float x) {
        if (!(x > 0.0f)) {
            return 1.0f;
        }
        unsigned bits = __float_as_uint(x);
        int exponent = (int)((bits >> 23) & 0xFF) - 127 + ((bits & ((1u << 23) - 1)) != 0 ? 1 : 0);
        return ldexpf(1.0f, exponent);
    }

    __device__ __forceinline__ float V41Fp8RoundTripDev(float v) {
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(v);
        return (float)q;
    }

    __device__ __forceinline__ float V41Fp4RoundTripDev(float v) {
        const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float a = fabsf(v);
        if (a >= 6.0f) {
            return copysignf(6.0f, v);
        }
        int lower = 0;
#pragma unroll
        for (int i = 1; i < 8; i++) {
            if (grid[i] <= a) {
                lower = i;
            }
        }
        float lo = grid[lower];
        float hi = grid[lower + 1 < 8 ? lower + 1 : 7];
        float r;
        if (a == lo) {
            r = lo;
        } else {
            float mid = 0.5f * (lo + hi);
            if (a < mid) {
                r = lo;
            } else if (a > mid) {
                r = hi;
            } else {
                r = ((lower & 1) == 0) ? lo : hi;
            }
        }
        return copysignf(r, v);
    }

    __device__ __forceinline__ float V41QuantScaleDev(float amax, int quantMode) {
        if (quantMode == 1) {
            amax = fmaxf(amax, 1e-4f);
            return V41Pow2CeilDev(amax * (1.0f / 448.0f));
        } else if (quantMode == 2) {
            amax = fmaxf(amax, 6.0f * ldexpf(1.0f, -126));
            return V41Pow2CeilDev(amax * (1.0f / 6.0f));
        }
        amax = fmaxf(amax, 6.0f * ldexpf(1.0f, -9));
        float s = V41Fp8RoundTripDev(amax / 6.0f);
        return s > 0.0f ? s : ldexpf(1.0f, -9);
    }

    __device__ __forceinline__ float V41QuantValueDev(float v, float scale, int quantMode) {
        float qmax = quantMode == 1 ? 448.0f : 6.0f;
        float q = fminf(qmax, fmaxf(-qmax, v / scale));
        q = quantMode == 1 ? V41Fp8RoundTripDev(q) : V41Fp4RoundTripDev(q);
        return q * scale;
    }

    // E2M1 编码：bit3 符号，bit2..0 是网格下标。r 必须已在 E2M1 网格上。
    __device__ __forceinline__ uint8_t V41EncodeFp4Dev(float r) {
        const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        uint8_t sign = signbit(r) ? 8 : 0;
        float a = fabsf(r);
        int idx = 0;
#pragma unroll
        for (int i = 1; i < 8; i++) {
            if (grid[i] <= a) {
                idx = i;
            }
        }
        return sign | (uint8_t)idx;
    }

    // E2M1 -> float，纯位运算（不再查表，省掉每个元素一次 local / constant 访存）：
    //   c == 0 -> 0；c == 1 -> 0.5（E2M1 的次正规）；c >= 2 -> (1 + 0.5 * (c & 1)) * 2^((c >> 1) - 1)
    // 与旧的 grid[] 查表逐 bit 相同（含 code == 8 时的 -0.0）。
    __device__ __forceinline__ float V41DecodeFp4Dev(int code) {
        const int c = code & 7;
        uint32_t bits = c == 0 ? 0u
                               : ((((uint32_t)(126 + (c >> 1))) << 23) |
                                  ((c >= 2 ? (uint32_t)(c & 1) : 0u) << 22));
        bits |= ((uint32_t)(code & 8)) << 28;
        return __uint_as_float(bits);
    }

    // 量化缓存行的解码（布局见 cpu/deepseekv41ops.cpp 顶部的说明）：
    //   quantMode 1：[dim 个 E4M3][dim/32 个 UE8M0]
    //   quantMode 3：[dim/2 字节打包 E2M1][dim/16 个 E4M3]
    //   quantMode 2：[dim/2 字节打包 E2M1][dim/32 个 UE8M0]
    __device__ __forceinline__ float V41LoadFp8Row(const uint8_t *row, int dim, int d) {
        __nv_fp8_e4m3 c;
        c.__x = row[d];
        return (float)c * ldexpf(1.0f, (int)row[dim + (d >> 5)] - 127);
    }

    __device__ __forceinline__ float V41LoadKvRow(const uint8_t *row, int dim, int d, int quantMode) {
        if (quantMode == 1) {
            return V41LoadFp8Row(row, dim, d);
        }
        const uint8_t *scales = row + (dim >> 1);
        const uint8_t packed = row[d >> 1];
        const float v = V41DecodeFp4Dev((d & 1) ? (packed >> 4) : (packed & 0xF));
        if (quantMode == 3) {
            __nv_fp8_e4m3 s;
            s.__x = scales[d >> 4];
            return v * (float)s;
        }
        return v * ldexpf(1.0f, (int)scales[d >> 5] - 127);
    }

    __device__ __forceinline__ uint32_t V41Pack2Bf16(float a, float b) {
        __nv_bfloat162 v = __halves2bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
        return *reinterpret_cast<const uint32_t*>(&v);
    }

    // 一次展开 8 个连续元素并直接写成 mma 片段需要的 16 字节 BF16。
    //
    // 要求 (d0 & 7) == 0、dst 16 字节对齐。
    // 8 个元素落在同一个 scale 块内（blockSize 是 16 或 32），所以 scale 只算一次；
    // 打包 FP4 的 4 个字节按一次 uint32 读出（缓存行起始与 4 字节对齐：三种 rowBytes
    // 都是 4 的倍数，d0 >> 1 也是 4 的倍数），FP8 按两次 uint32 读出。
    // 数值与逐元素的 V41LoadKvRow + __float2bfloat16_rn 逐 bit 相同。
    __device__ __forceinline__ void V41LoadKvRow8Bf16(const uint8_t *row, int dim, int d0,
                                                      int quantMode, __nv_bfloat16 *dst) {
        uint32_t o[4];
        if (quantMode == 1) {
            const float scale = ldexpf(1.0f, (int)row[dim + (d0 >> 5)] - 127);
            const uint32_t w0 = *(const uint32_t*)(row + d0);
            const uint32_t w1 = *(const uint32_t*)(row + d0 + 4);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t w = (e < 2) ? w0 : w1;
                const int sh = (e & 1) * 16;
                __nv_fp8_e4m3 c0, c1;
                c0.__x = (unsigned char)((w >> sh) & 0xFF);
                c1.__x = (unsigned char)((w >> (sh + 8)) & 0xFF);
                o[e] = V41Pack2Bf16((float)c0 * scale, (float)c1 * scale);
            }
        } else {
            const uint8_t *scales = row + (dim >> 1);
            float scale;
            if (quantMode == 3) {
                __nv_fp8_e4m3 s;
                s.__x = scales[d0 >> 4];
                scale = (float)s;
            } else {
                scale = ldexpf(1.0f, (int)scales[d0 >> 5] - 127);
            }
            const uint32_t packed = *(const uint32_t*)(row + (d0 >> 1));
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int byte = (int)((packed >> (e * 8)) & 0xFF);
                o[e] = V41Pack2Bf16(V41DecodeFp4Dev(byte & 0xF) * scale,
                                    V41DecodeFp4Dev(byte >> 4) * scale);
            }
        }
        *((float4*)dst) = make_float4(__uint_as_float(o[0]), __uint_as_float(o[1]),
                                      __uint_as_float(o[2]), __uint_as_float(o[3]));
    }

    // 一次展开 16 个连续元素（head_dim = 512 时正好是一个 lane 的份额，整行一趟做完）。
    //
    // 16 个元素一定落在同一个 scale 块内（blockSize 是 16 或 32），所以整行每个 lane
    // 只做一次数据读（FP4 8 字节 / FP8 16 字节）+ 一次 scale 读，再写两个 float4。
    // 要求 (d0 & 15) == 0、dst 16 字节对齐、行首 16 字节对齐（dim = 512 的三种
    // rowBytes 288 / 272 / 528 都是 16 的倍数，配合 256 字节对齐的基址成立；
    // 调用方在主机侧检查了这一点）。
    // 数值与 V41LoadKvRow8Bf16 / 逐元素解包逐 bit 相同。
    __device__ __forceinline__ void V41LoadKvRow16Bf16(const uint8_t *row, int dim, int d0,
                                                       int quantMode, __nv_bfloat16 *dst) {
        uint32_t o[8];
        if (quantMode == 1) {
            const float scale = ldexpf(1.0f, (int)row[dim + (d0 >> 5)] - 127);
            const uint4 w = *(const uint4*)(row + d0);
            const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int g = 0; g < 4; g++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int sh = h * 16;
                    __nv_fp8_e4m3 c0, c1;
                    c0.__x = (unsigned char)((ws[g] >> sh) & 0xFF);
                    c1.__x = (unsigned char)((ws[g] >> (sh + 8)) & 0xFF);
                    o[g * 2 + h] = V41Pack2Bf16((float)c0 * scale, (float)c1 * scale);
                }
            }
        } else {
            const uint8_t *scales = row + (dim >> 1);
            float scale;
            if (quantMode == 3) {
                __nv_fp8_e4m3 sq;
                sq.__x = scales[d0 >> 4];
                scale = (float)sq;
            } else {
                scale = ldexpf(1.0f, (int)scales[d0 >> 5] - 127);
            }
            const uint2 packed = *(const uint2*)(row + (d0 >> 1));
            const uint32_t ps[2] = {packed.x, packed.y};
#pragma unroll
            for (int q = 0; q < 2; q++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int byte = (int)((ps[q] >> (e * 8)) & 0xFF);
                    o[q * 4 + e] = V41Pack2Bf16(V41DecodeFp4Dev(byte & 0xF) * scale,
                                                V41DecodeFp4Dev(byte >> 4) * scale);
                }
            }
        }
        float4 *out4 = (float4*)dst;
        out4[0] = make_float4(__uint_as_float(o[0]), __uint_as_float(o[1]),
                              __uint_as_float(o[2]), __uint_as_float(o[3]));
        out4[1] = make_float4(__uint_as_float(o[4]), __uint_as_float(o[5]),
                              __uint_as_float(o[6]), __uint_as_float(o[7]));
    }

    // (quantMode, blockSize) -> 每行字节数；与 CPU 侧 V41KvRowBytes 一致
    inline int V41KvRowBytesHost(int dim, int quantMode, int blockSize) {
        return (quantMode == 1 ? dim : dim / 2) + dim / blockSize;
    }

    // 由行宽反推布局
    inline bool V41ParseKvRowHost(int dim, int rowBytes, int *quantMode, int *blockSize) {
        if (rowBytes == V41KvRowBytesHost(dim, 1, 32)) {
            *quantMode = 1; *blockSize = 32; return true;
        }
        if (rowBytes == V41KvRowBytesHost(dim, 3, 16)) {
            *quantMode = 3; *blockSize = 16; return true;
        }
        if (rowBytes == V41KvRowBytesHost(dim, 2, 32)) {
            *quantMode = 2; *blockSize = 32; return true;
        }
        return false;
    }

    struct V41RopeTable {
        float invFreq[64];
        int pairs;
    };

    // 块内 warp 归约
    __device__ __forceinline__ float V41WarpSum(float v) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        return v;
    }

    __device__ __forceinline__ float V41WarpMax(float v) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        return v;
    }

    template <int THREADS>
    __device__ __forceinline__ float V41BlockSum(float v, float *shared) {
        v = V41WarpSum(v);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        __syncthreads();
        if (lane == 0) {
            shared[warp] = v;
        }
        __syncthreads();
        float total = 0.0f;
        for (int i = 0; i < THREADS / 32; i++) {
            total += shared[i];
        }
        return total;
    }

    template <int THREADS>
    __device__ __forceinline__ float V41BlockMax(float v, float *shared) {
        v = V41WarpMax(v);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        __syncthreads();
        if (lane == 0) {
            shared[warp] = v;
        }
        __syncthreads();
        float total = -FLT_MAX;
        for (int i = 0; i < THREADS / 32; i++) {
            total = fmaxf(total, shared[i]);
        }
        return total;
    }

    // ---------------- HcMix ----------------

    constexpr int kHcThreads = 256;
    constexpr int kHcMaxMix = 32;   // (2 + hc) * hc, hc <= 4

    template <typename T>
    __global__ void V41HcMixKernel(const T *x, const float *fn, const float *scale, const float *base,
                                   int hcMult, int dim, int sinkhornIters, float eps, float normEps,
                                   float *pre, float *post, float *comb) {
        const int t = blockIdx.x;
        const int flatDim = hcMult * dim;
        const int mixHc = (2 + hcMult) * hcMult;
        __shared__ float sharedPartial[kHcMaxMix + 1][kHcThreads / 32];
        __shared__ float mixes[kHcMaxMix];
        __shared__ float combShared[16];
        float acc[kHcMaxMix + 1];
#pragma unroll
        for (int m = 0; m <= kHcMaxMix; m++) {
            acc[m] = 0.0f;
        }
        const T *xrow = x + (uint64_t)t * flatDim;
        for (int k = threadIdx.x; k < flatDim; k += kHcThreads) {
            float xv = V41Load<T>(xrow, k);
            acc[kHcMaxMix] += xv * xv;
            for (int m = 0; m < mixHc; m++) {
                acc[m] += xv * fn[(uint64_t)m * flatDim + k];
            }
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int m = 0; m <= mixHc; m++) {
            int idx = m == mixHc ? kHcMaxMix : m;
            float v = V41WarpSum(acc[idx]);
            if (lane == 0) {
                sharedPartial[idx][warp] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x <= mixHc) {
            int idx = threadIdx.x == mixHc ? kHcMaxMix : threadIdx.x;
            float total = 0.0f;
            for (int w = 0; w < kHcThreads / 32; w++) {
                total += sharedPartial[idx][w];
            }
            if (threadIdx.x == mixHc) {
                sharedPartial[kHcMaxMix][0] = total;   // sumsq
            } else {
                mixes[idx] = total;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float ss = sharedPartial[kHcMaxMix][0];
            float rsqrtv = rsqrtf(ss / flatDim + normEps);
            float *preOut = pre + (uint64_t)t * hcMult;
            float *postOut = post + (uint64_t)t * hcMult;
            float *combOut = comb + (uint64_t)t * hcMult * hcMult;
            for (int h = 0; h < hcMult; h++) {
                preOut[h] = V41SigmoidDev(mixes[h] * rsqrtv * scale[0] + base[h]) + eps;
                postOut[h] = 2.0f * V41SigmoidDev(mixes[h + hcMult] * rsqrtv * scale[1] + base[h + hcMult]);
            }
            for (int r = 0; r < hcMult; r++) {
                float rowMax = -FLT_MAX;
                for (int c = 0; c < hcMult; c++) {
                    int idx = r * hcMult + c + 2 * hcMult;
                    combShared[r * hcMult + c] = mixes[idx] * rsqrtv * scale[2] + base[idx];
                    rowMax = fmaxf(rowMax, combShared[r * hcMult + c]);
                }
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    float v = __expf(combShared[r * hcMult + c] - rowMax);
                    combShared[r * hcMult + c] = v;
                    rowSum += v;
                }
                for (int c = 0; c < hcMult; c++) {
                    combShared[r * hcMult + c] = combShared[r * hcMult + c] / rowSum + eps;
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combShared[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combShared[r * hcMult + c] /= (colSum + eps);
                }
            }
            for (int it = 1; it < sinkhornIters; it++) {
                for (int r = 0; r < hcMult; r++) {
                    float rowSum = 0.0f;
                    for (int c = 0; c < hcMult; c++) {
                        rowSum += combShared[r * hcMult + c];
                    }
                    for (int c = 0; c < hcMult; c++) {
                        combShared[r * hcMult + c] /= (rowSum + eps);
                    }
                }
                for (int c = 0; c < hcMult; c++) {
                    float colSum = 0.0f;
                    for (int r = 0; r < hcMult; r++) {
                        colSum += combShared[r * hcMult + c];
                    }
                    for (int r = 0; r < hcMult; r++) {
                        combShared[r * hcMult + c] /= (colSum + eps);
                    }
                }
            }
            for (int i = 0; i < hcMult * hcMult; i++) {
                combOut[i] = combShared[i];
            }
        }
    }


    // hcMult 编译期特化 + 一个 block 处理多个 token 的版本。
    //
    // 旧 kernel 的 acc[kHcMaxMix + 1] 被运行时下标访问，ptxas 把它放进 local memory
    // （136 字节栈帧）；而且每个 token 一个 block，混合系数矩阵 fn（[(2+hc)*hc, hc*dim]，
    // 真实模型是 24 x 20480 的 FP32，约 2 MB）要被每个 token 各读一遍。
    // 这里把 mixHc 变成编译期常量（acc 进寄存器），并让一个 block 处理 kHcTokens 个 token，
    // 在 k 的循环里复用同一份 fn。每个 (token, m) 的累加顺序与旧 kernel 完全一致，结果逐 bit 相同。

    constexpr int kHcTokens = 4;

    template <typename T, int HC>
    __global__ void __launch_bounds__(kHcThreads)
    V41HcMixKernelMulti(const T *x, const float *fn, const float *scale, const float *base,
                        int tokens, int dim, int sinkhornIters, float eps, float normEps,
                        float *pre, float *post, float *comb) {
        constexpr int MIXHC = (2 + HC) * HC;
        const int flatDim = HC * dim;
        const int t0 = blockIdx.x * kHcTokens;
        __shared__ float sharedPartial[kHcTokens][MIXHC + 1][kHcThreads / 32];
        __shared__ float mixes[kHcTokens][MIXHC];
        __shared__ float sumsq[kHcTokens];
        __shared__ float combShared[kHcTokens][HC * HC];

        float acc[kHcTokens][MIXHC + 1];
#pragma unroll
        for (int tb = 0; tb < kHcTokens; tb++) {
#pragma unroll
            for (int m = 0; m <= MIXHC; m++) {
                acc[tb][m] = 0.0f;
            }
        }
        for (int k = threadIdx.x; k < flatDim; k += kHcThreads) {
            float xs[kHcTokens];
#pragma unroll
            for (int tb = 0; tb < kHcTokens; tb++) {
                xs[tb] = t0 + tb < tokens ? V41Load<T>(x + (uint64_t)(t0 + tb) * flatDim, k) : 0.0f;
                acc[tb][MIXHC] += xs[tb] * xs[tb];
            }
#pragma unroll
            for (int m = 0; m < MIXHC; m++) {
                const float f = fn[(uint64_t)m * flatDim + k];
#pragma unroll
                for (int tb = 0; tb < kHcTokens; tb++) {
                    acc[tb][m] += xs[tb] * f;
                }
            }
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
        for (int tb = 0; tb < kHcTokens; tb++) {
#pragma unroll
            for (int m = 0; m <= MIXHC; m++) {
                float v = V41WarpSum(acc[tb][m]);
                if (lane == 0) {
                    sharedPartial[tb][m][warp] = v;
                }
            }
        }
        __syncthreads();
        // 每个线程负责一个 (token, m) 对，把 8 个 warp 的部分和加起来
        for (int idx = threadIdx.x; idx < kHcTokens * (MIXHC + 1); idx += kHcThreads) {
            const int tb = idx / (MIXHC + 1), m = idx % (MIXHC + 1);
            float total = 0.0f;
            for (int w = 0; w < kHcThreads / 32; w++) {
                total += sharedPartial[tb][m][w];
            }
            if (m == MIXHC) {
                sumsq[tb] = total;
            } else {
                mixes[tb][m] = total;
            }
        }
        __syncthreads();
        if (threadIdx.x < kHcTokens && t0 + (int)threadIdx.x < tokens) {
            const int tb = threadIdx.x;
            const int t = t0 + tb;
            const float rsqrtv = rsqrtf(sumsq[tb] / flatDim + normEps);
            float *preOut = pre + (uint64_t)t * HC;
            float *postOut = post + (uint64_t)t * HC;
            float *combOut = comb + (uint64_t)t * HC * HC;
            for (int h = 0; h < HC; h++) {
                preOut[h] = V41SigmoidDev(mixes[tb][h] * rsqrtv * scale[0] + base[h]) + eps;
                postOut[h] = 2.0f * V41SigmoidDev(mixes[tb][h + HC] * rsqrtv * scale[1] + base[h + HC]);
            }
            float *cs = combShared[tb];
            for (int r = 0; r < HC; r++) {
                float rowMax = -FLT_MAX;
                for (int c = 0; c < HC; c++) {
                    int idx = r * HC + c + 2 * HC;
                    cs[r * HC + c] = mixes[tb][idx] * rsqrtv * scale[2] + base[idx];
                    rowMax = fmaxf(rowMax, cs[r * HC + c]);
                }
                float rowSum = 0.0f;
                for (int c = 0; c < HC; c++) {
                    float v = __expf(cs[r * HC + c] - rowMax);
                    cs[r * HC + c] = v;
                    rowSum += v;
                }
                for (int c = 0; c < HC; c++) {
                    cs[r * HC + c] = cs[r * HC + c] / rowSum + eps;
                }
            }
            for (int c = 0; c < HC; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < HC; r++) {
                    colSum += cs[r * HC + c];
                }
                for (int r = 0; r < HC; r++) {
                    cs[r * HC + c] /= (colSum + eps);
                }
            }
            for (int it = 1; it < sinkhornIters; it++) {
                for (int r = 0; r < HC; r++) {
                    float rowSum = 0.0f;
                    for (int c = 0; c < HC; c++) {
                        rowSum += cs[r * HC + c];
                    }
                    for (int c = 0; c < HC; c++) {
                        cs[r * HC + c] /= (rowSum + eps);
                    }
                }
                for (int c = 0; c < HC; c++) {
                    float colSum = 0.0f;
                    for (int r = 0; r < HC; r++) {
                        colSum += cs[r * HC + c];
                    }
                    for (int r = 0; r < HC; r++) {
                        cs[r * HC + c] /= (colSum + eps);
                    }
                }
            }
            for (int i = 0; i < HC * HC; i++) {
                combOut[i] = cs[i];
            }
        }
    }

    template <typename T>
    bool V41LaunchHcMixMulti(const T *x, const float *fn, const float *scale, const float *base,
                             int hcMult, int tokens, int dim, int sinkhornIters, float eps, float normEps,
                             float *pre, float *post, float *comb) {
        const int blocks = (tokens + kHcTokens - 1) / kHcTokens;
        switch (hcMult) {
            case 1:
                V41HcMixKernelMulti<T, 1><<<blocks, kHcThreads>>>(x, fn, scale, base, tokens, dim,
                    sinkhornIters, eps, normEps, pre, post, comb);
                return true;
            case 2:
                V41HcMixKernelMulti<T, 2><<<blocks, kHcThreads>>>(x, fn, scale, base, tokens, dim,
                    sinkhornIters, eps, normEps, pre, post, comb);
                return true;
            case 3:
                V41HcMixKernelMulti<T, 3><<<blocks, kHcThreads>>>(x, fn, scale, base, tokens, dim,
                    sinkhornIters, eps, normEps, pre, post, comb);
                return true;
            case 4:
                V41HcMixKernelMulti<T, 4><<<blocks, kHcThreads>>>(x, fn, scale, base, tokens, dim,
                    sinkhornIters, eps, normEps, pre, post, comb);
                return true;
            default:
                return false;
        }
    }

    // ---------------- HcApplyPre ----------------

    template <typename T>
    __global__ void V41HcApplyPreKernel(const T *x, const float *pre, T *y, int tokens, int hcMult, int dim) {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t total = (uint64_t)tokens * dim;
        if (idx >= total) {
            return;
        }
        int t = (int)(idx / dim), d = (int)(idx % dim);
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[(uint64_t)t * hcMult + h] * V41Load<T>(x, ((uint64_t)t * hcMult + h) * dim + d);
        }
        V41Store<T>(y, idx, v);
    }

    // ---------------- HcApplyPre + RMSNorm 融合 ----------------
    //
    // V4.1 每个子层的入口都是「HcApplyPre(curHidden, pre) -> x -> RMSNorm(x) -> 子层输入」，
    // 中间的 x 只被紧接着的 RMSNorm 读一次。融合后省掉 x 的一次写 + 一次读，
    // 以及一次 kernel 启动（40 层 x 2 次）。
    //
    // 数值必须与「HcApplyPre 写 BF16」+「FastllmRMSNormKernelInner1<T> 的 BF16 版本」
    // 逐 bit 相同，因此这里照抄两边的顺序：
    //   * 折叠按 h 从小到大 FP32 累加，写进中间结果前先舍入到 BF16（HcApplyPre 的 V41Store）；
    //   * 平方和按 i = tid, tid + T, ... 的顺序累加，warp shuffle-down 归约树、
    //     跨 warp 的 warp_sums 归约、rsqrtf(val / channels + eps) 与写出的 lo * s * w
    //     都与 RMSNorm kernel 一致。
    // 所以 THREAD_PER_BLOCK 必须与 LaunchFastllmRMSNormBFloat16 对同一个 channels 的选择相同。
    template <int THREAD_PER_BLOCK, int MAX_ITER>
    __global__ void __launch_bounds__(THREAD_PER_BLOCK)
    V41HcPreNormKernel(const __nv_bfloat16 *x, const float *pre, const float *weight,
                       __nv_bfloat16 *out, int hcMult, int channels, float eps) {
        constexpr int WARP_SIZE = 32;
        constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
        __shared__ float warp_sums[NUM_WARPS];
        __shared__ float scale;

        const int o = blockIdx.x;
        const int bf2 = channels / 2;
        const __nv_bfloat162 *xb = reinterpret_cast<const __nv_bfloat162*>(x) + (uint64_t)o * hcMult * bf2;
        const float *preRow = pre + (uint64_t)o * hcMult;
        __nv_bfloat162 *outb = reinterpret_cast<__nv_bfloat162*>(out) + (uint64_t)o * bf2;

        const unsigned tid = threadIdx.x;
        const int warp_id = (int)tid / WARP_SIZE, lane_id = (int)tid % WARP_SIZE;

        float2 cache[MAX_ITER];
        float sum2 = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_ITER; k++) {
            const int i = (int)tid + k * THREAD_PER_BLOCK;
            if (i < bf2) {
                float lo = 0.0f, hi = 0.0f;
                for (int h = 0; h < hcMult; h++) {
                    const __nv_bfloat162 v = xb[(uint64_t)h * bf2 + i];
                    const float p = preRow[h];
                    lo += p * __bfloat162float(v.x);
                    hi += p * __bfloat162float(v.y);
                }
                lo = __bfloat162float(__float2bfloat16_rn(lo));
                hi = __bfloat162float(__float2bfloat16_rn(hi));
                cache[k] = make_float2(lo, hi);
                sum2 += lo * lo + hi * hi;
            }
        }

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        }
        if (lane_id == 0) {
            warp_sums[warp_id] = sum2;
        }
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                scale = rsqrtf(val / channels + eps);
            }
        }
        __syncthreads();

        const float s = scale;
#pragma unroll
        for (int k = 0; k < MAX_ITER; k++) {
            const int i = (int)tid + k * THREAD_PER_BLOCK;
            if (i < bf2) {
                const float w0 = __ldg(&weight[i * 2]);
                const float w1 = __ldg(&weight[i * 2 + 1]);
                __nv_bfloat162 ov;
                ov.x = __float2bfloat16_rn(cache[k].x * s * w0);
                ov.y = __float2bfloat16_rn(cache[k].y * s * w1);
                outb[i] = ov;
            }
        }
    }

    // ---------------- EngramApply ----------------

    constexpr int kEngramThreads = 256;

    template <typename HT, typename KT>
    __global__ void V41EngramApplyKernel(HT *hidden, const KT *kv, const float *qw, const float *kw,
                                         const float *mask, int hcMult, int dim, float eps, float clampValue) {
        const int t = blockIdx.x / hcMult;
        const int c = blockIdx.x % hcMult;
        __shared__ float shared[kEngramThreads / 32];
        HT *h = hidden + ((uint64_t)t * hcMult + c) * dim;
        const KT *key = kv + (uint64_t)t * (hcMult + 1) * dim + (uint64_t)c * dim;
        const KT *value = kv + (uint64_t)t * (hcMult + 1) * dim + (uint64_t)hcMult * dim;
        float hss = 0.0f, kss = 0.0f, dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += kEngramThreads) {
            float hv = V41Load<HT>(h, d);
            float kvv = V41Load<KT>(key, d);
            hss += hv * hv;
            kss += kvv * kvv;
            dot += hv * (qw[c * dim + d] * kw[c * dim + d]) * kvv;
        }
        hss = V41BlockSum<kEngramThreads>(hss, shared);
        kss = V41BlockSum<kEngramThreads>(kss, shared);
        dot = V41BlockSum<kEngramThreads>(dot, shared);
        float rstd = rsqrtf(hss / dim + eps) * rsqrtf(kss / dim + eps);
        float score = dot * rstd * rsqrtf((float)dim);
        float mag = sqrtf(fmaxf(fabsf(score), clampValue));
        float gate = V41SigmoidDev(copysignf(mag, score));
        if (mask != nullptr && mask[t] == 0.0f) {
            gate = 0.0f;
        }
        for (int d = threadIdx.x; d < dim; d += kEngramThreads) {
            float hv = V41Load<HT>(h, d);
            V41Store<HT>(h, d, hv + gate * V41Load<KT>(value, d));
        }
    }

    // ---------------- RotaryQuant ----------------

    // 旧实现：一个 block 处理一行，blockDim = dim（<= 1024）。
    // 保留作为 FASTLLM_DSV41_LEGACY_ROTARY=1 的对比基准。
    template <typename T>
    __global__ void V41RotaryQuantLegacyKernel(T *x, int rowsPerToken, int dim, V41RopeTable rope,
                                               int ropeDim, int startPos, int posStep, int inverse,
                                               int quantMode, int quantDim, int quantBlock) {
        extern __shared__ float row[];
        const int r = blockIdx.x;
        const int token = r / rowsPerToken;
        T *base = x + (uint64_t)r * dim;
        const int d = threadIdx.x;
        row[d] = V41Load<T>(base, d);
        __syncthreads();
        const int off = dim - ropeDim;
        if (d < ropeDim / 2) {
            float pos = (float)(startPos + (long long)token * posStep);
            float ang = pos * rope.invFreq[d];
            float c = cosf(ang), s = sinf(ang);
            if (inverse) {
                s = -s;
            }
            float a = row[off + 2 * d], b = row[off + 2 * d + 1];
            row[off + 2 * d] = a * c - b * s;
            row[off + 2 * d + 1] = a * s + b * c;
        }
        __syncthreads();
        float v = row[d];
        if (quantMode > 0 && d < quantDim) {
            // 组内 amax：组是连续的 quantBlock（16 / 32）个元素，与 warp 对齐
            int group = d / quantBlock;
            float a = fabsf(v);
            unsigned mask = 0xffffffff;
            if (quantBlock == 16) {
                a = fmaxf(a, __shfl_xor_sync(mask, a, 8));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 4));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 2));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 1));
            } else {
                a = V41WarpMax(a);
            }
            (void)group;
            float scale = V41QuantScaleDev(a, quantMode);
            v = V41QuantValueDev(v, scale, quantMode);
        }
        V41Store<T>(base, d, v);
    }

    // quantMode == 0（q / 注意力输出的逆旋转）：只有末尾 ropeDim 个元素会变，
    // 旧 kernel 却把整行 load 到共享内存再原样写回（dim = 512、ropeDim = 128 时白读写 75%）。
    // 这里一个 block 处理多行的若干个旋转对，只碰真正需要旋转的元素。
    template <typename T>
    __global__ void V41RotaryOnlyKernel(T *x, int rowsPerToken, int dim, V41RopeTable rope,
                                        int ropeDim, int startPos, int posStep, int inverse,
                                        uint64_t totalPairs) {
        const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= totalPairs) {
            return;
        }
        const int pairs = ropeDim >> 1;
        const int r = (int)(idx / (uint64_t)pairs);
        const int d = (int)(idx - (uint64_t)r * pairs);
        const int token = r / rowsPerToken;
        T *base = x + (uint64_t)r * dim + (dim - ropeDim) + 2 * d;
        const float pos = (float)(startPos + (long long)token * posStep);
        const float ang = pos * rope.invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        if (inverse) {
            s = -s;
        }
        const float a = V41Load<T>(base, 0), b = V41Load<T>(base, 1);
        V41Store<T>(base, 0, a * c - b * s);
        V41Store<T>(base, 1, a * s + b * c);
    }

    // quantMode > 0：一个线程一个元素，旋转对通过相邻 lane 的 __shfl_xor 交换，
    // 因此不再需要共享内存与两次 __syncthreads，一个 block 可以处理 rowsPerBlock 行
    // （dim = 128 时旧实现只有 128 个线程一个 block）。
    // 块内 amax 的语义不变：仍是 warp 内连续 quantBlock（16 / 32）个 lane 的归约。
    template <typename T>
    __global__ void V41RotaryQuantKernel(T *x, int rowsPerToken, int dim, V41RopeTable rope,
                                         int ropeDim, int startPos, int posStep, int inverse,
                                         int quantMode, int quantDim, int quantBlock,
                                         int rowsPerBlock, int rows) {
        const int sub = threadIdx.x / dim;
        const int d = (int)threadIdx.x - sub * dim;
        const int r = blockIdx.x * rowsPerBlock + sub;
        if (r >= rows) {
            return;     // dim 是 32 的倍数，整个 warp 一起返回
        }
        const int token = r / rowsPerToken;
        T *base = x + (uint64_t)r * dim;
        float v = V41Load<T>(base, d);
        // 旋转对 (off + 2p, off + 2p + 1) 是相邻的两个 lane；off 是偶数，所以配对与
        // lane 的奇偶一致。shuffle 放在分支外，保证整个 warp 都参与。
        const float other = __shfl_xor_sync(0xffffffff, v, 1);
        const int off = dim - ropeDim;
        if (d >= off) {
            const int k = d - off;
            const float pos = (float)(startPos + (long long)token * posStep);
            const float ang = pos * rope.invFreq[k >> 1];
            float c = cosf(ang), s = sinf(ang);
            if (inverse) {
                s = -s;
            }
            v = (k & 1) ? (other * s + v * c) : (v * c - other * s);
        }
        if (d < quantDim) {
            float a = fabsf(v);
            if (quantBlock == 16) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 8));
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 4));
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1));
            } else {
                a = V41WarpMax(a);
            }
            const float scale = V41QuantScaleDev(a, quantMode);
            v = V41QuantValueDev(v, scale, quantMode);
        }
        V41Store<T>(base, d, v);
    }

    // ---------------- Compress ----------------

    template <typename T>
    __global__ void V41CompressKernel(const T *kv, const T *score, const float *normWeight,
                                      int n, int dim, int ratio, float normEps, __nv_bfloat16 *out) {
        extern __shared__ float pooled[];
        __shared__ float shared[32];
        const int idx = blockIdx.x;       // b * blocks + j
        const int blocks = n / ratio;
        const int b = idx / blocks, j = idx % blocks;
        const int d = threadIdx.x;
        float value;
        if (ratio == 1) {
            value = V41Load<T>(kv, ((uint64_t)b * n + j) * dim + d);
        } else {
            float mx = -FLT_MAX;
            for (int r = 0; r < ratio; r++) {
                mx = fmaxf(mx, V41Load<T>(score, ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d));
            }
            float sum = 0.0f, acc = 0.0f;
            for (int r = 0; r < ratio; r++) {
                uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                float e = __expf(V41Load<T>(score, off) - mx);
                sum += e;
                acc += e * V41Load<T>(kv, off);
            }
            value = acc / sum;
        }
        value = V41Bf16Round(value);
        pooled[d] = value;
        float ss = V41BlockSum<512>(value * value, shared);
        float rsqrtv = rsqrtf(ss / dim + normEps);
        out[(uint64_t)idx * dim + d] = __float2bfloat16_rn(normWeight[d] * pooled[d] * rsqrtv);
    }

    // ---------------- IndexerScore ----------------

    constexpr int kIdxThreads = 128;

    // k8 != nullptr 时 key 来自量化缓存行（打包 FP4 或 FP8），逐元素解码后参与点积
    template <typename QT, typename KT>
    __global__ void V41IndexerScoreKernel(const QT *q, const float *weights, const KT *k,
                                          int seqlen, int heads, int dim, int m, float *out,
                                          const uint8_t *k8 = nullptr, int kMode = 1, int kRowBytes = 0) {
        extern __shared__ float qs[];   // heads * dim + heads
        const int t = blockIdx.y;       // b * seqlen + i
        const int b = t / seqlen;
        float *ws = qs + heads * dim;
        for (int i = threadIdx.x; i < heads * dim; i += kIdxThreads) {
            qs[i] = V41Load<QT>(q, (uint64_t)t * heads * dim + i);
        }
        for (int i = threadIdx.x; i < heads; i += kIdxThreads) {
            ws[i] = weights[(uint64_t)t * heads + i];
        }
        __syncthreads();
        const int j = blockIdx.x * kIdxThreads + threadIdx.x;
        if (j >= m) {
            return;
        }
        float kvals[128];
        if (k8 != nullptr) {
            const uint8_t *krow8 = k8 + ((uint64_t)b * m + j) * kRowBytes;
            for (int d = 0; d < dim; d++) {
                kvals[d] = V41LoadKvRow(krow8, dim, d, kMode);
            }
        } else {
            const KT *krow = k + ((uint64_t)b * m + j) * dim;
            for (int d = 0; d < dim; d++) {
                kvals[d] = V41Load<KT>(krow, d);
            }
        }
        float total = 0.0f;
        for (int h = 0; h < heads; h++) {
            const float *qh = qs + h * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += qh[d] * kvals[d];
            }
            total += fmaxf(dot, 0.0f) * ws[h];
        }
        out[(uint64_t)t * m + j] = total;
    }


    // ---------------- mma / ldmatrix 内联汇编封装（SM80+） ----------------

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    __device__ __forceinline__ uint32_t V41SmemAddr(const void *p) {
        return static_cast<uint32_t>(__cvta_generic_to_shared(p));
    }

    __device__ __forceinline__ void V41LdmX4(uint32_t (&r)[4], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
    }

    __device__ __forceinline__ void V41LdmX2(uint32_t (&r)[2], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1]) : "r"(s));
    }

    __device__ __forceinline__ void V41LdmX2T(uint32_t (&r)[2], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1]) : "r"(s));
    }

    __device__ __forceinline__ void V41MmaBf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#endif

    // ---------------- IndexerScore（BF16 mma 版本，SM80+） ----------------
    //
    // score[t][j] = sum_h relu(q[t][h] · k[j]) * w[t][h]
    // k 没有 head 维（32 个 indexer head 共享同一份 key），因此 K tile 只装载一次，
    // 循环 head 时只换 Q tile。一个 block 负责 kIdxBT 个 token x kIdxBJ 个候选。
    // 因果可见范围之外（j >= (startPos + i + 1) / ratio）的整块直接写 -inf 跳过计算，
    // prefill 时省掉一半左右的运算。

    constexpr int kIdxBT = 64;                          // 每个 block 的 token 数
    constexpr int kIdxBJ = 64;                          // 每个 block 的候选数
    constexpr int kIdxDim = 128;                        // index_head_dim
    constexpr int kIdxRowStride = kIdxDim + 8;
    constexpr int kIdxMmaWarps = 8;
    constexpr int kIdxMmaThreads = kIdxMmaWarps * 32;
    constexpr int kIdxNPerWarp = kIdxBJ / 16;           // 每个 warp 负责的 n-tile 数（4）

    struct V41IdxShared {
        __nv_bfloat16 ks[kIdxBJ][kIdxRowStride];
        __nv_bfloat16 qs[kIdxBT][kIdxRowStride];
        float ws[kIdxBT];
    };

    __global__ void __launch_bounds__(kIdxMmaThreads)
    V41IndexerScoreMmaKernel(const __nv_bfloat16 *q, const float *weights, const __nv_bfloat16 *k,
                             int seqlen, int heads, int m, int ratio, int startPos, float *out,
                             const uint8_t *k8, int kMode, int kRowBytes, int vecUnpack) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        __shared__ V41IdxShared sh;
        const int j0 = blockIdx.x * kIdxBJ;
        const int i0 = blockIdx.y * kIdxBT;
        const int b = blockIdx.z;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int mTile = warp & 3;                     // 4 个 16 行的 token tile
        const int nHalf = warp >> 2;                    // 每个 warp 负责 32 列候选

        const int iMax = min(seqlen - 1, i0 + kIdxBT - 1);
        const int visibleMax = ratio > 0 ? min(m, (startPos + iMax + 1) / ratio) : m;
        if (j0 >= visibleMax) {
            // 整块都在可见范围外：写 -inf，后续的 block 打分 / top-k 会按 visible 忽略它
            for (int idx = threadIdx.x; idx < kIdxBT * kIdxBJ; idx += kIdxMmaThreads) {
                int ii = idx / kIdxBJ, jj = idx % kIdxBJ;
                if (i0 + ii < seqlen && j0 + jj < m) {
                    out[((uint64_t)b * seqlen + i0 + ii) * m + j0 + jj] = -INFINITY;
                }
            }
            return;
        }

        for (int v = threadIdx.x; v < kIdxBJ * (kIdxDim / 8); v += kIdxMmaThreads) {
            int jj = v / (kIdxDim / 8), dv = v % (kIdxDim / 8);
            if (j0 + jj >= m) {
                *((float4*)&sh.ks[jj][dv * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (k8 != nullptr) {
                // 量化 key：就地解到 BF16 片段再进 mma（FP4 的值在 BF16 上是精确的）
                const uint8_t *krow8 = k8 + ((uint64_t)b * m + j0 + jj) * kRowBytes;
                if (vecUnpack) {
                    V41LoadKvRow8Bf16(krow8, kIdxDim, dv * 8, kMode, &sh.ks[jj][dv * 8]);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        sh.ks[jj][dv * 8 + e] = __float2bfloat16_rn(V41LoadKvRow(krow8, kIdxDim, dv * 8 + e, kMode));
                    }
                }
            } else {
                *((float4*)&sh.ks[jj][dv * 8]) =
                    *((const float4*)(k + ((uint64_t)b * m + j0 + jj) * kIdxDim) + dv);
            }
        }

        const int aRow = mTile * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int aColBlk = (lane >> 4) * 8;
        const int bRowBase = nHalf * 32 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int bColBlk = (lane >> 4) * 8;

        float acc[kIdxNPerWarp / 2][2][4];
#pragma unroll
        for (int n = 0; n < kIdxNPerWarp / 2; n++) {
#pragma unroll
            for (int u = 0; u < 2; u++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    acc[n][u][e] = 0.0f;
                }
            }
        }

        for (int h = 0; h < heads; h++) {
            __syncthreads();
            for (int v = threadIdx.x; v < kIdxBT * (kIdxDim / 8); v += kIdxMmaThreads) {
                int ii = v / (kIdxDim / 8), dv = v % (kIdxDim / 8);
                if (i0 + ii < seqlen) {
                    *((float4*)&sh.qs[ii][dv * 8]) =
                        *((const float4*)(q + (((uint64_t)b * seqlen + i0 + ii) * heads + h) * kIdxDim) + dv);
                } else {
                    *((float4*)&sh.qs[ii][dv * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            for (int ii = threadIdx.x; ii < kIdxBT; ii += kIdxMmaThreads) {
                sh.ws[ii] = i0 + ii < seqlen ? weights[((uint64_t)b * seqlen + i0 + ii) * heads + h] : 0.0f;
            }
            __syncthreads();

            float s[kIdxNPerWarp / 2][2][4];
#pragma unroll
            for (int n = 0; n < kIdxNPerWarp / 2; n++) {
#pragma unroll
                for (int u = 0; u < 2; u++) {
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[n][u][e] = 0.0f;
                    }
                }
            }
#pragma unroll
            for (int kk = 0; kk < kIdxDim; kk += 16) {
                uint32_t a[4];
                V41LdmX4(a, &sh.qs[aRow][kk + aColBlk]);
#pragma unroll
                for (int n = 0; n < kIdxNPerWarp / 2; n++) {
                    uint32_t rb[4];
                    V41LdmX4(rb, &sh.ks[bRowBase + n * 16][kk + bColBlk]);
                    uint32_t b0[2] = {rb[0], rb[2]};
                    uint32_t b1[2] = {rb[1], rb[3]};
                    V41MmaBf16(s[n][0], a, b0);
                    V41MmaBf16(s[n][1], a, b1);
                }
            }
            const float w0 = sh.ws[mTile * 16 + (lane >> 2)];
            const float w1 = sh.ws[mTile * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int n = 0; n < kIdxNPerWarp / 2; n++) {
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    acc[n][u][0] += fmaxf(s[n][u][0], 0.0f) * w0;
                    acc[n][u][1] += fmaxf(s[n][u][1], 0.0f) * w0;
                    acc[n][u][2] += fmaxf(s[n][u][2], 0.0f) * w1;
                    acc[n][u][3] += fmaxf(s[n][u][3], 0.0f) * w1;
                }
            }
        }

        const int row0 = i0 + mTile * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        const int colBase = j0 + nHalf * 32 + (lane & 3) * 2;
#pragma unroll
        for (int n = 0; n < kIdxNPerWarp / 2; n++) {
#pragma unroll
            for (int u = 0; u < 2; u++) {
                int col = colBase + n * 16 + u * 8;
                if (col < m) {
                    if (row0 < seqlen) {
                        out[((uint64_t)b * seqlen + row0) * m + col] = acc[n][u][0];
                    }
                    if (row1 < seqlen) {
                        out[((uint64_t)b * seqlen + row1) * m + col] = acc[n][u][2];
                    }
                }
                if (col + 1 < m) {
                    if (row0 < seqlen) {
                        out[((uint64_t)b * seqlen + row0) * m + col + 1] = acc[n][u][1];
                    }
                    if (row1 < seqlen) {
                        out[((uint64_t)b * seqlen + row1) * m + col + 1] = acc[n][u][3];
                    }
                }
            }
        }
#endif
    }

    // ---------------- radix select（k-th largest） ----------------

    __device__ __forceinline__ unsigned V41FloatKey(float f) {
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }

    constexpr int kSelThreads = 256;

    // 返回第 k 大的 key（keys 由 getKey(i) 给出，i in [0, n)）。若 n <= k，返回 0。
    template <typename GetKey>
    __device__ unsigned V41RadixSelect(int n, int k, GetKey getKey, unsigned *hist /* shared, 16 */) {
        unsigned prefix = 0;
        int remaining = k;
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (threadIdx.x < 16) {
                hist[threadIdx.x] = 0;
            }
            __syncthreads();
            unsigned mask = shift == 28 ? 0u : (0xffffffffu << (shift + 4));
            for (int i = threadIdx.x; i < n; i += kSelThreads) {
                unsigned key = getKey(i);
                if ((key & mask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 15], 1u);
                }
            }
            __syncthreads();
            // 从高 digit 往低找
            if (threadIdx.x == 0) {
                int cnt = remaining;
                int digit = 15;
                for (; digit >= 0; digit--) {
                    int c = (int)hist[digit];
                    if (cnt <= c) {
                        break;
                    }
                    cnt -= c;
                }
                if (digit < 0) {
                    digit = 0;
                }
                hist[0] = (unsigned)digit;
                hist[1] = (unsigned)cnt;
            }
            __syncthreads();
            unsigned digit = hist[0];
            remaining = (int)hist[1];
            prefix |= digit << shift;
            __syncthreads();
        }
        return prefix;
    }

    // ---------------- CandidateBlocks ----------------

    __global__ void V41BlockScoreKernel(const float *score, int seqlen, int m, int blockSize, int numBlocks,
                                        int ratio, int startPos, float *blockScore) {
        const int t = blockIdx.x;
        const int i = t % seqlen;
        const int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        float *brow = blockScore + (uint64_t)t * numBlocks;
        for (int k = threadIdx.x; k < numBlocks; k += blockDim.x) {
            float mx = -INFINITY;
            int end = min(m, (k + 1) * blockSize);
            for (int j = k * blockSize; j < end; j++) {
                if (j < visible) {
                    mx = fmaxf(mx, row[j]);
                }
            }
            if (visible > 0 && k == (visible - 1) / blockSize) {
                mx = INFINITY;
            }
            brow[k] = mx;
        }
    }

    __global__ void V41CandidateSelectKernel(const float *blockScore, int numBlocks, int topkBlocks, uint8_t *mask) {
        const int t = blockIdx.x;
        const float *brow = blockScore + (uint64_t)t * numBlocks;
        uint8_t *mrow = mask + (uint64_t)t * numBlocks;
        __shared__ unsigned hist[16];
        int keep = min(topkBlocks, numBlocks);
        unsigned threshold = V41RadixSelect(numBlocks, keep,
            [&](int i) { return V41FloatKey(brow[i]); }, hist);
        const unsigned negInf = V41FloatKey(-INFINITY);
        // 严格大于的全部保留，等于阈值的按预算保留（顺序无关紧要，候选掩码只需数量约束）
        for (int i = threadIdx.x; i < numBlocks; i += kSelThreads) {
            unsigned key = V41FloatKey(brow[i]);
            uint8_t v = 0;
            if (key > threshold && key > negInf) {
                v = 1;
            }
            mrow[i] = v;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < numBlocks; i += kSelThreads) {
            unsigned key = V41FloatKey(brow[i]);
            if (key == threshold && key > negInf) {
                mrow[i] = 1;   // 阈值上的块全部保留（可能略多于 topkBlocks，只放宽候选范围）
            }
        }
    }

    // ---------------- IndexerTopK ----------------

    __global__ void V41TopKKernelLegacy(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int topK, int width, int ratio, int startPos,
                                  int32_t *out) {
        typedef cub::BlockScan<int, kSelThreads> BlockScan;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ unsigned hist[16];
        const int t = blockIdx.x;
        const int i = t % seqlen;
        const int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        const uint8_t *cand = candidates == nullptr ? nullptr : candidates + (uint64_t)t * numBlocks;
        int32_t *orow = out + (uint64_t)t * width;

        auto eligible = [&](int j) -> bool {
            if (j >= visible) {
                return false;
            }
            if (cand != nullptr) {
                int blk = j / blockSize;
                if (blk >= numBlocks || cand[blk] == 0) {
                    return false;
                }
            }
            return true;
        };
        const unsigned negInf = V41FloatKey(-INFINITY);
        auto getKey = [&](int j) -> unsigned {
            return eligible(j) ? V41FloatKey(row[j]) : negInf;
        };

        // 统计可用数量
        int local = 0;
        for (int j = threadIdx.x; j < visible; j += kSelThreads) {
            local += eligible(j) ? 1 : 0;
        }
        int total = 0;
        BlockScan(scanStorage).ExclusiveSum(local, local, total);
        __syncthreads();
        int keep = min(width, total);
        unsigned threshold = 0;
        int tieBudget = 0;
        if (total > width) {
            threshold = V41RadixSelect(visible, width, getKey, hist);
            // 严格大于的数量
            int greater = 0;
            for (int j = threadIdx.x; j < visible; j += kSelThreads) {
                unsigned key = getKey(j);
                greater += (key > threshold && key > negInf) ? 1 : 0;
            }
            int greaterTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(greater, greater, greaterTotal);
            __syncthreads();
            tieBudget = width - greaterTotal;
        }
        // 按升序压缩写出
        int runningSel = 0, runningTie = 0;
        for (int st = 0; st < visible; st += kSelThreads) {
            int j = st + threadIdx.x;
            bool valid = j < visible;
            unsigned key = valid ? getKey(j) : negInf;
            int isTie = 0, isSel = 0;
            if (valid && key > negInf) {
                if (total <= width) {
                    isSel = 1;
                } else if (key > threshold) {
                    isSel = 1;
                } else if (key == threshold) {
                    isTie = 1;
                }
            }
            int tieRank = 0, tieTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(isTie, tieRank, tieTotal);
            __syncthreads();
            if (isTie && runningTie + tieRank < tieBudget) {
                isSel = 1;
            }
            int selRank = 0, selTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(isSel, selRank, selTotal);
            __syncthreads();
            if (isSel) {
                int pos = runningSel + selRank;
                if (pos < width) {
                    orow[pos] = j;
                }
            }
            runningSel += selTotal;
            runningTie += tieTotal;
        }
        for (int p = keep + threadIdx.x; p < width; p += kSelThreads) {
            orow[p] = -1;
        }
    }

    // 新 top-k：输出与 V41TopKKernelLegacy 逐字节相同（升序、并列按下标从小到大、
    // 不足 width 补 -1），但把「11 趟全量扫描」降到 3 趟：
    //
    //   1. 有候选掩码时先把候选块下标升序压缩进共享内存，之后所有扫描都只在
    //      候选块内进行（两级 top-k 下 visible 远大于候选集，上下文越长省得越多）；
    //   2. radix select 从 4 bit 8 趟改成 8 bit 4 趟，且只有第 1 趟是全量的：
    //      第 1 趟顺带把落在选中桶里的 key 压缩进共享内存，后 3 趟只扫这几百个 key；
    //   3. 「严格大于阈值的个数」不再单独扫一趟：radix select 结束时的 remaining
    //      恰好是需要从等于阈值的元素里取的个数（tieBudget）；
    //   4. 写出阶段把两次 BlockScan 合成一次（greater / tie 的前缀和打包进一个 int），
    //      输出位置 = 前面 greater 的个数 + min(tieBudget, 前面 tie 的个数)。
    //
    // 共享内存里的候选块下标由 candList（动态共享内存）承载，listCap == 0 表示不压缩。
    constexpr int kTopKSurvivorCap = 1024;

    __global__ void V41TopKKernel(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int blockShift, int width,
                                  int ratio, int startPos, int listCap, int32_t *out) {
        typedef cub::BlockScan<int, kSelThreads> BlockScan;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ unsigned hist[256];
        __shared__ unsigned survivors[kTopKSurvivorCap];
        __shared__ int meta[4];             // 0: digit, 1: remaining, 2: 候选块个数
        extern __shared__ int candList[];

        const int t = blockIdx.x;
        const int i = t % seqlen;
        const int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        const uint8_t *cand = candidates == nullptr ? nullptr : candidates + (uint64_t)t * numBlocks;
        int32_t *orow = out + (uint64_t)t * width;
        const unsigned negInf = V41FloatKey(-INFINITY);

        // ---- 候选块下标的升序压缩 ----
        const bool canCompact = cand != nullptr && listCap > 0 && blockShift >= 0;
        int nCandBlocks = 0;
        if (canCompact) {
            int running = 0;
            for (int st = 0; st < numBlocks; st += kSelThreads) {
                const int blk = st + threadIdx.x;
                const int keepIt = (blk < numBlocks && cand[blk] != 0 && (blk << blockShift) < visible) ? 1 : 0;
                int rank = 0, tot = 0;
                BlockScan(scanStorage).ExclusiveSum(keepIt, rank, tot);
                __syncthreads();
                if (keepIt && running + rank < listCap) {
                    candList[running + rank] = blk;
                }
                running += tot;
            }
            nCandBlocks = running;
        }
        // 候选块个数超出压缩表容量时退回逐 j 查掩码（结果不变，只是多扫一些）
        const bool compact = canCompact && nCandBlocks <= listCap;
        const int nDomain = compact ? (nCandBlocks << blockShift) : visible;

        // p（压缩域下标）-> j（原始候选下标）；compact 时 j 一定落在候选块内
#define V41_TOPK_INDEX(p) (compact ? ((candList[(p) >> blockShift] << blockShift) | ((p) & (blockSize - 1))) : (p))
#define V41_TOPK_OK(j)    ((j) < visible && (compact || cand == nullptr || \
                           ((j) / blockSize < numBlocks && cand[(j) / blockSize] != 0)))

        // ---- 第 1 趟：全量直方图（高 8 bit），同时得到可用元素总数 ----
        hist[threadIdx.x] = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < nDomain; p += kSelThreads) {
            const int j = V41_TOPK_INDEX(p);
            if (V41_TOPK_OK(j)) {
                atomicAdd(&hist[V41FloatKey(row[j]) >> 24], 1u);
            }
        }
        __syncthreads();
        int binCount = (int)hist[threadIdx.x], binPrefix = 0, total = 0;
        BlockScan(scanStorage).ExclusiveSum(binCount, binPrefix, total);
        __syncthreads();

        const int keep = min(width, total);
        unsigned threshold = 0;
        int tieBudget = 0;
        const bool selectAll = total <= width;
        if (!selectAll) {
            if (threadIdx.x == 0) {
                int cnt = width, digit = 255;
                for (; digit >= 0; digit--) {
                    const int c = (int)hist[digit];
                    if (cnt <= c) {
                        break;
                    }
                    cnt -= c;
                }
                if (digit < 0) {
                    digit = 0;
                }
                meta[0] = digit;
                meta[1] = cnt;
                meta[2] = (int)hist[digit];
            }
            __syncthreads();
            int digit = meta[0];
            int remaining = meta[1];
            const int survivorCount = meta[2];
            unsigned prefix = (unsigned)digit << 24;
            const bool useShared = survivorCount <= kTopKSurvivorCap;
            __syncthreads();
            if (useShared) {
                // 第 2 趟：把落在选中桶里的 key 压缩进共享内存
                if (threadIdx.x == 0) {
                    meta[3] = 0;
                }
                __syncthreads();
                for (int p = threadIdx.x; p < nDomain; p += kSelThreads) {
                    const int j = V41_TOPK_INDEX(p);
                    if (V41_TOPK_OK(j)) {
                        const unsigned key = V41FloatKey(row[j]);
                        if ((key >> 24) == (unsigned)digit) {
                            const unsigned pos = atomicAdd((unsigned*)&meta[3], 1u);
                            if (pos < (unsigned)kTopKSurvivorCap) {
                                survivors[pos] = key;
                            }
                        }
                    }
                }
                __syncthreads();
            }
            for (int shift = 16; shift >= 0; shift -= 8) {
                hist[threadIdx.x] = 0;
                __syncthreads();
                const unsigned mask = 0xffffffffu << (shift + 8);
                if (useShared) {
                    for (int s = threadIdx.x; s < survivorCount; s += kSelThreads) {
                        const unsigned key = survivors[s];
                        if ((key & mask) == prefix) {
                            atomicAdd(&hist[(key >> shift) & 255], 1u);
                        }
                    }
                } else {
                    for (int p = threadIdx.x; p < nDomain; p += kSelThreads) {
                        const int j = V41_TOPK_INDEX(p);
                        if (V41_TOPK_OK(j)) {
                            const unsigned key = V41FloatKey(row[j]);
                            if ((key & mask) == prefix) {
                                atomicAdd(&hist[(key >> shift) & 255], 1u);
                            }
                        }
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    int cnt = remaining, d = 255;
                    for (; d >= 0; d--) {
                        const int c = (int)hist[d];
                        if (cnt <= c) {
                            break;
                        }
                        cnt -= c;
                    }
                    if (d < 0) {
                        d = 0;
                    }
                    meta[0] = d;
                    meta[1] = cnt;
                }
                __syncthreads();
                digit = meta[0];
                remaining = meta[1];
                prefix |= (unsigned)digit << shift;
                __syncthreads();
            }
            threshold = prefix;
            // radix select 结束时的 remaining 就是等于阈值的元素里要取的个数
            tieBudget = remaining;
        }

        // ---- 最后一趟：按升序写出 ----
        int runningGreater = 0, runningTie = 0;
        for (int st = 0; st < nDomain; st += kSelThreads) {
            const int p = st + threadIdx.x;
            const int j = p < nDomain ? V41_TOPK_INDEX(p) : -1;
            const bool ok = j >= 0 && V41_TOPK_OK(j);
            const unsigned key = ok ? V41FloatKey(row[j]) : negInf;
            int isGreater = 0, isTie = 0;
            if (ok && key > negInf) {
                if (selectAll || key > threshold) {
                    isGreater = 1;
                } else if (key == threshold) {
                    isTie = 1;
                }
            }
            int packedPrefix = 0, packedTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(isGreater * 1024 + isTie, packedPrefix, packedTotal);
            __syncthreads();
            const int gBefore = runningGreater + (packedPrefix >> 10);
            const int tBefore = runningTie + (packedPrefix & 1023);
            if (isGreater || (isTie && tBefore < tieBudget)) {
                const int pos = gBefore + min(tieBudget, tBefore);
                if (pos < width) {
                    orow[pos] = j;
                }
            }
            runningGreater += packedTotal >> 10;
            runningTie += packedTotal & 1023;
        }
        for (int p = keep + threadIdx.x; p < width; p += kSelThreads) {
            orow[p] = -1;
        }
#undef V41_TOPK_INDEX
#undef V41_TOPK_OK
    }

    // ---------------- SparseAttention ----------------

    constexpr int kAttnHeadsPerBlock = 32;
    constexpr int kAttnLanes = 8;
    constexpr int kAttnThreads = kAttnHeadsPerBlock * kAttnLanes;   // 256


    template <typename QT>
    __global__ void __launch_bounds__(kAttnThreads)
    V41SparseAttentionKernel(const QT *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                             const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                             const int32_t *cmpIdx,
                             const float *sink, int seqlen, int heads, int dim,
                             int windowSize, int cap, int topWidth, int startPos,
                             float scale, __nv_bfloat16 *out,
                             int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes) {
        __shared__ float kvRow[512];
        const int t = blockIdx.x;                  // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        constexpr int lanesPerHead = kAttnLanes;          // 8
        constexpr int segment = 512 / lanesPerHead;       // 64
        const int h = blockIdx.y * kAttnHeadsPerBlock + threadIdx.x / lanesPerHead;
        const int lane = threadIdx.x % lanesPerHead;
        const QT *qrow = q + ((uint64_t)t * heads + h) * dim + lane * segment;
        float qv[64];
        for (int d = 0; d < segment; d++) {
            qv[d] = V41Load<QT>(qrow, d);
        }
        float acc[64];
        for (int d = 0; d < segment; d++) {
            acc[d] = 0.0f;
        }
        float mx = sink[h];
        float l = 1.0f;

        const int winStart = max(0, pos - windowSize + 1);
        const int winCount = pos - winStart + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        for (int c = 0; c < winCount + cmpCount; c++) {
            const __nv_bfloat16 *src = nullptr;
            const uint8_t *src8 = nullptr;
            int src8Mode = 1;
            if (c < winCount) {
                int p = winStart + c;
                if (p >= startPos) {
                    src = chunkKV + ((uint64_t)b * seqlen + (p - startPos)) * dim;
                } else if (ringKV8 != nullptr) {
                    src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                    src8Mode = ringMode;
                } else {
                    src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * dim;
                }
            } else {
                int idx = idxRow[c - winCount];
                if (idx < 0 || idx >= cap) {
                    continue;
                }
                if (compressedKV8 != nullptr) {
                    src8 = compressedKV8 + ((uint64_t)b * cap + idx) * cmpRowBytes;
                    src8Mode = cmpMode;
                } else {
                    src = compressedKV + ((uint64_t)b * cap + idx) * dim;
                }
            }
            __syncthreads();
            if (src8 != nullptr) {
                for (int d = threadIdx.x; d < dim; d += kAttnThreads) {
                    kvRow[d] = V41LoadKvRow(src8, dim, d, src8Mode);
                }
            } else {
                for (int d = threadIdx.x; d < dim; d += kAttnThreads) {
                    kvRow[d] = __bfloat162float(src[d]);
                }
            }
            __syncthreads();
            float dot = 0.0f;
            const float *seg = kvRow + lane * segment;
            for (int d = 0; d < segment; d++) {
                dot += qv[d] * seg[d];
            }
            // 8 lane 归约
            dot += __shfl_xor_sync(0xffffffff, dot, 1);
            dot += __shfl_xor_sync(0xffffffff, dot, 2);
            dot += __shfl_xor_sync(0xffffffff, dot, 4);
            float s = dot * scale;
            float newMx = fmaxf(mx, s);
            float alpha = __expf(mx - newMx);
            float p = __expf(s - newMx);
            l = l * alpha + p;
            for (int d = 0; d < segment; d++) {
                acc[d] = acc[d] * alpha + p * seg[d];
            }
            mx = newMx;
        }
        float inv = 1.0f / l;
        __nv_bfloat16 *orow = out + ((uint64_t)t * heads + h) * dim + lane * segment;
        for (int d = 0; d < segment; d++) {
            orow[d] = __float2bfloat16_rn(acc[d] * inv);
        }
    }


    // ---------------- SparseAttention（BF16 mma 版本，SM80+） ----------------
    //
    // 数值语义与上面的 V41SparseAttentionKernel 一致（在线 softmax + attn_sink，
    // 候选顺序为「滑窗 -> 压缩 top-k」），区别只有：
    //   * QK^T 与 PV 用 mma.sync.m16n8k16（BF16 输入 / FP32 累加）代替 FP32 标量点积；
    //   * 候选按 kMmaNC 个一组分块，一组只要 4 次 __syncthreads（原来每个候选 1 次）；
    //   * 一个 block 内 32 个 head 共享同一份候选 KV，Q 也常驻共享内存；
    //   * 候选维可以再切成 gridDim.z 份（split-K），由 V41SparseMergeKernel 合并部分和，
    //     用于 decode 时提高并行度。
    //
    // 片段布局（PTX ISA 的 m16n8k16 定义）：
    //   A(16x16): lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1} [+8])
    //   B(16x8) : lane l 持有 (k = (l%4)*2 + {0,1} [+8], n = l/4)
    //   C(16x8) : lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1})

    constexpr int kMmaHeads = 32;                       // 每个 block 处理的 head 数
    constexpr int kMmaNC = 32;                          // 每轮处理的候选数
    constexpr int kMmaWarps = 8;
    constexpr int kMmaThreads = kMmaWarps * 32;         // 256
    constexpr int kMmaDim = 512;                        // V4.1 的 head_dim 固定为 512
    constexpr int kMmaKvStride = kMmaDim + 8;           // +8 个半字，消除 ldmatrix 的 bank 冲突
    constexpr int kMmaPStride = kMmaNC + 8;
    constexpr int kMmaDimSlice = kMmaDim / 4;           // PV 阶段每个 warp 负责 128 维
    constexpr int kMmaPvTiles = kMmaDimSlice / 8;       // 16 个 n-tile

    struct V41MmaShared {
        __nv_bfloat16 qs[kMmaHeads][kMmaKvStride];
        __nv_bfloat16 kvs[kMmaNC][kMmaKvStride];
        __nv_bfloat16 ps[kMmaHeads][kMmaPStride];
        float sc[kMmaHeads][kMmaNC];
        float mx[kMmaHeads];
        float lsum[kMmaHeads];
        float alpha[kMmaHeads];
        int valid[kMmaNC];
    };


    __global__ void __launch_bounds__(kMmaThreads)
    V41SparseAttentionMmaKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, const float *sink, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos, float scale,
                                __nv_bfloat16 *out, float *partAcc, float *partMx, float *partL,
                                int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes, int vecUnpack) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        extern __shared__ char v41MmaSharedRaw[];
        V41MmaShared &sh = *reinterpret_cast<V41MmaShared*>(v41MmaSharedRaw);

        const int t = blockIdx.x;                       // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int h0 = blockIdx.y * kMmaHeads;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;

        for (int v = threadIdx.x; v < kMmaHeads * (kMmaDim / 8); v += kMmaThreads) {
            int hh = v / (kMmaDim / 8), dv = v % (kMmaDim / 8);
            *((float4*)&sh.qs[hh][dv * 8]) =
                *((const float4*)(q + ((uint64_t)t * heads + h0 + hh) * kMmaDim) + dv);
        }
        for (int r = threadIdx.x; r < kMmaHeads; r += kMmaThreads) {
            sh.mx[r] = blockIdx.z == 0 ? sink[h0 + r] : -FLT_MAX;
            sh.lsum[r] = blockIdx.z == 0 ? 1.0f : 0.0f;
        }

        const int winStart = max(0, pos - windowSize + 1);
        const int winCount = pos - winStart + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        const int totalCand = winCount + cmpCount;
        const int tilesTotal = (totalCand + kMmaNC - 1) / kMmaNC;
        const int tilesPerSplit = (tilesTotal + (int)gridDim.z - 1) / (int)gridDim.z;
        const int tileBegin = (int)blockIdx.z * tilesPerSplit;
        const int tileEnd = min(tilesTotal, tileBegin + tilesPerSplit);

        const int pvM = warp >> 2, pvSlice = warp & 3;   // PV: (mTile, 128 维切片)
        const int qkM = warp >> 2, qkN = warp & 3;       // QK: (mTile, 8 个候选)
        float acc[kMmaPvTiles][4];
#pragma unroll
        for (int n = 0; n < kMmaPvTiles; n++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[n][e] = 0.0f;
            }
        }
        const int qkARow = qkM * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int qkAColBlk = (lane >> 4) * 8;
        const int qkBRow = qkN * 8 + (lane & 7);
        const int qkBColBlk = ((lane >> 3) & 1) * 8;
        const int pvARow = pvM * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int pvAColBlk = (lane >> 4) * 8;
        const int pvBRow = lane & 15;

        for (int tile = tileBegin; tile < tileEnd; tile++) {
            const int tileStart = tile * kMmaNC;
            __syncthreads();
            for (int slot = warp; slot < kMmaNC; slot += kMmaWarps) {
                const int c = tileStart + slot;
                const __nv_bfloat16 *src = nullptr;
                const uint8_t *src8 = nullptr;
                int src8Mode = 1;
                bool ok = c < totalCand;
                if (ok) {
                    if (c < winCount) {
                        int p = winStart + c;
                        if (p >= startPos) {
                            src = chunkKV + ((uint64_t)b * seqlen + (p - startPos)) * kMmaDim;
                        } else if (ringKV8 != nullptr) {
                            src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                            src8Mode = ringMode;
                        } else {
                            src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                        }
                    } else {
                        int idx = idxRow[c - winCount];
                        if (idx < 0 || idx >= cap) {
                            ok = false;
                        } else if (compressedKV8 != nullptr) {
                            src8 = compressedKV8 + ((uint64_t)b * cap + idx) * cmpRowBytes;
                            src8Mode = cmpMode;
                        } else {
                            src = compressedKV + ((uint64_t)b * cap + idx) * kMmaDim;
                        }
                    }
                }
                if (!ok) {
                    for (int v = lane; v < kMmaDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else if (src8 != nullptr) {
                    // 就地解到 BF16 片段再进 mma；FP4 的值在 BF16 上是精确的
                    if (vecUnpack) {
                        // 一个 lane 负责 16 个值（head_dim 512 / 32 lane），整行一趟：
                        // 一次数据读 + 一次 scale 读 -> 两个 float4 写进 mma 片段
                        V41LoadKvRow16Bf16(src8, kMmaDim, lane * 16, src8Mode, &sh.kvs[slot][lane * 16]);
                    } else {
                        for (int d = lane; d < kMmaDim; d += 32) {
                            sh.kvs[slot][d] = __float2bfloat16_rn(V41LoadKvRow(src8, kMmaDim, d, src8Mode));
                        }
                    }
                } else {
                    for (int v = lane; v < kMmaDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = *((const float4*)src + v);
                    }
                }
                if (lane == 0) {
                    sh.valid[slot] = ok ? 1 : 0;
                }
            }
            __syncthreads();

            {   // QK^T：C[16 head, 8 cand] = Q[16, 512] x KV[8, 512]^T
                float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint32_t a[4], bb[2];
#pragma unroll 2
                for (int k = 0; k < kMmaDim; k += 32) {
                    V41LdmX4(a, &sh.qs[qkARow][k + qkAColBlk]);
                    V41LdmX2(bb, &sh.kvs[qkBRow][k + qkBColBlk]);
                    V41MmaBf16(d0, a, bb);
                    V41LdmX4(a, &sh.qs[qkARow][k + 16 + qkAColBlk]);
                    V41LdmX2(bb, &sh.kvs[qkBRow][k + 16 + qkBColBlk]);
                    V41MmaBf16(d1, a, bb);
                }
                const int r = qkM * 16 + (lane >> 2);
                const int c = qkN * 8 + (lane & 3) * 2;
                sh.sc[r][c] = d0[0] + d1[0];
                sh.sc[r][c + 1] = d0[1] + d1[1];
                sh.sc[r + 8][c] = d0[2] + d1[2];
                sh.sc[r + 8][c + 1] = d0[3] + d1[3];
            }
            __syncthreads();

            {   // 在线 softmax：每个 head 一行，由 8 个线程负责
                constexpr int perThread = kMmaNC / 8;
                const int row = threadIdx.x >> 3;
                const int sub = threadIdx.x & 7;
                float vals[perThread];
                bool oks[perThread];
                float m = -FLT_MAX;
#pragma unroll
                for (int c = 0; c < perThread; c++) {
                    int col = sub * perThread + c;
                    bool ok = sh.valid[col] != 0;
                    float v = ok ? sh.sc[row][col] * scale : -FLT_MAX;
                    oks[c] = ok;
                    vals[c] = v;
                    m = fmaxf(m, v);
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
                const float oldMx = sh.mx[row];
                const float newMx = fmaxf(oldMx, m);
                float s = 0.0f;
#pragma unroll
                for (int c = 0; c < perThread; c++) {
                    float p = oks[c] ? __expf(vals[c] - newMx) : 0.0f;
                    s += p;
                    sh.ps[row][sub * perThread + c] = __float2bfloat16_rn(p);
                }
                s += __shfl_xor_sync(0xffffffff, s, 1);
                s += __shfl_xor_sync(0xffffffff, s, 2);
                s += __shfl_xor_sync(0xffffffff, s, 4);
                if (sub == 0) {
                    float a = __expf(oldMx - newMx);
                    sh.alpha[row] = a;
                    sh.mx[row] = newMx;
                    sh.lsum[row] = sh.lsum[row] * a + s;
                }
            }
            __syncthreads();

            {   // PV：O[16 head, 128 dim] = alpha * O + P[16, 32] x V[32, 128]
                const float a0 = sh.alpha[pvM * 16 + (lane >> 2)];
                const float a1 = sh.alpha[pvM * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int n = 0; n < kMmaPvTiles; n++) {
                    acc[n][0] *= a0;
                    acc[n][1] *= a0;
                    acc[n][2] *= a1;
                    acc[n][3] *= a1;
                }
#pragma unroll
                for (int k = 0; k < kMmaNC; k += 16) {
                    uint32_t a[4];
                    V41LdmX4(a, &sh.ps[pvARow][k + pvAColBlk]);
#pragma unroll
                    for (int n = 0; n < kMmaPvTiles; n++) {
                        uint32_t bb[2];
                        V41LdmX2T(bb, &sh.kvs[k + pvBRow][pvSlice * kMmaDimSlice + n * 8]);
                        V41MmaBf16(acc[n], a, bb);
                    }
                }
            }
        }
        __syncthreads();

        const int outRow0 = pvM * 16 + (lane >> 2);
        const int outRow1 = outRow0 + 8;
        const int outCol = pvSlice * kMmaDimSlice + (lane & 3) * 2;
        if (partAcc == nullptr) {
            const float inv0 = 1.0f / sh.lsum[outRow0];
            const float inv1 = 1.0f / sh.lsum[outRow1];
            __nv_bfloat16 *o0 = out + ((uint64_t)t * heads + h0 + outRow0) * kMmaDim;
            __nv_bfloat16 *o1 = out + ((uint64_t)t * heads + h0 + outRow1) * kMmaDim;
#pragma unroll
            for (int n = 0; n < kMmaPvTiles; n++) {
                int d = outCol + n * 8;
                o0[d] = __float2bfloat16_rn(acc[n][0] * inv0);
                o0[d + 1] = __float2bfloat16_rn(acc[n][1] * inv0);
                o1[d] = __float2bfloat16_rn(acc[n][2] * inv1);
                o1[d + 1] = __float2bfloat16_rn(acc[n][3] * inv1);
            }
        } else {
            const uint64_t base = ((uint64_t)blockIdx.z * gridDim.x + t) * heads;
            float *a0 = partAcc + (base + h0 + outRow0) * kMmaDim;
            float *a1 = partAcc + (base + h0 + outRow1) * kMmaDim;
#pragma unroll
            for (int n = 0; n < kMmaPvTiles; n++) {
                int d = outCol + n * 8;
                a0[d] = acc[n][0];
                a0[d + 1] = acc[n][1];
                a1[d] = acc[n][2];
                a1[d + 1] = acc[n][3];
            }
            if (warp == 0) {
                for (int r = lane; r < kMmaHeads; r += 32) {
                    partMx[base + h0 + r] = sh.mx[r];
                    partL[base + h0 + r] = sh.lsum[r];
                }
            }
        }
#endif
    }

    // split-K 的部分和合并（在线 softmax 的标准合并公式）
    __global__ void V41SparseMergeKernel(const float *partAcc, const float *partMx, const float *partL,
                                         int tokens, int heads, int splits, __nv_bfloat16 *out) {
        const int idx = blockIdx.x;                     // t * heads + h
        const uint64_t stride = (uint64_t)tokens * heads;
        float mx = -FLT_MAX;
        for (int s = 0; s < splits; s++) {
            mx = fmaxf(mx, partMx[(uint64_t)s * stride + idx]);
        }
        float denom = 0.0f;
        for (int s = 0; s < splits; s++) {
            denom += __expf(partMx[(uint64_t)s * stride + idx] - mx) * partL[(uint64_t)s * stride + idx];
        }
        const float inv = 1.0f / denom;
        for (int d = threadIdx.x; d < kMmaDim; d += blockDim.x) {
            float v = 0.0f;
            for (int s = 0; s < splits; s++) {
                v += __expf(partMx[(uint64_t)s * stride + idx] - mx) *
                     partAcc[((uint64_t)s * stride + idx) * kMmaDim + d];
            }
            out[(uint64_t)idx * kMmaDim + d] = __float2bfloat16_rn(v * inv);
        }
    }

    // 设备是否支持 BF16 mma（SM80+），按设备号缓存
    bool V41MmaSupported() {
        static int cache[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        int dev = FastllmCudaGetDevice();
        if (dev < 0 || dev >= 16) {
            return false;
        }
        if (cache[dev] < 0) {
            int major = 0;
            cache[dev] = (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          major >= 8) ? 1 : 0;
        }
        return cache[dev] == 1;
    }

    int V41SmCount() {
        static int cache[16] = {0};
        int dev = FastllmCudaGetDevice();
        if (dev < 0 || dev >= 16) {
            return 1;
        }
        if (cache[dev] == 0) {
            int n = 0;
            cache[dev] = (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0)
                         ? n : 1;
        }
        return cache[dev];
    }

    bool V41EnvOn(const char *name) {
        const char *v = getenv(name);
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }

    // 量化 KV 行的向量化解包（一次 4 字节 / 8 个 FP4 值）；
    // FASTLLM_DSV41_LEGACY_FP4_UNPACK=1 退回逐元素解包做对比。
    bool V41Fp4VecUnpack() {
        return !V41EnvOn("FASTLLM_DSV41_LEGACY_FP4_UNPACK");
    }

    // FASTLLM_DSV41_LEGACY_ROTARY=1 退回「一行一个 block + 共享内存」的旧旋转/量化 kernel
    bool V41LegacyRotary() {
        return V41EnvOn("FASTLLM_DSV41_LEGACY_ROTARY");
    }

    // FASTLLM_DSV41_LEGACY_TOPK=1 退回逐 visible 元素扫描的旧 top-k kernel
    bool V41LegacyTopK() {
        return V41EnvOn("FASTLLM_DSV41_LEGACY_TOPK");
    }

    // ---------------- QuantizeKV ----------------

    constexpr int kKvQuantThreads = 128;

    // 每个 block 处理一行；blockSize 个线程为一组处理一个 scale 块，lane 对应块内元素。
    // scale 的推导（V41QuantScaleDev）与伪量化 (DeepSeekV41FakeQuantRow) 共用，
    // 因此对已伪量化过的行是幂等的，即 FP4 / FP8 存储无损。
    template <typename T>
    __global__ void V41QuantizeKVKernel(const T *input, uint8_t *output, int rows, int dim,
                                        int quantMode, int blockSize, int rowBytes) {
        const int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const int nblocks = dim / blockSize;
        const int groups = kKvQuantThreads / blockSize;
        const int gid = threadIdx.x / blockSize;
        const int lane = threadIdx.x % blockSize;
        const T *src = input + (uint64_t)row * dim;
        uint8_t *dst = output + (uint64_t)row * rowBytes;
        uint8_t *scales = dst + (quantMode == 1 ? dim : dim / 2);
        const float qmax = quantMode == 1 ? 448.0f : 6.0f;
        // 固定迭代次数，保证组内所有 lane 都参与 __shfl（否则归约会读到未定义值）
        const int iters = (nblocks + groups - 1) / groups;
        for (int it = 0; it < iters; it++) {
            const int blk = it * groups + gid;
            const int i = blk * blockSize + lane;
            const bool valid = blk < nblocks;
            float v = valid ? V41Load<T>(src, i) : 0.0f;
            float amax = fabsf(v);
            for (int o = blockSize >> 1; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            }
            const float scale = V41QuantScaleDev(amax, quantMode);
            const float q = fminf(qmax, fmaxf(-qmax, v / scale));
            if (quantMode == 1) {
                __nv_fp8_e4m3 code(q);
                if (valid) {
                    dst[i] = code.__x;
                }
            } else {
                unsigned code = V41EncodeFp4Dev(V41Fp4RoundTripDev(q));
                unsigned hi = __shfl_down_sync(0xffffffff, code, 1);
                if (valid && (lane & 1) == 0) {
                    dst[i >> 1] = (uint8_t)(code | (hi << 4));
                }
            }
            if (valid && lane == 0) {
                if (quantMode == 3) {
                    __nv_fp8_e4m3 sc(scale);            // scale 已在 E4M3 网格上
                    scales[blk] = sc.__x;
                } else {
                    int e = (int)((__float_as_uint(scale) >> 23) & 0xFF);   // scale 为正的 2 的幂
                    scales[blk] = (uint8_t)e;
                }
            }
        }
    }

    // ---------------- WindowStore ----------------

    __global__ void V41WindowStoreKernel(const uint8_t *chunk, uint8_t *ring, int seqlen, int rowBytes,
                                         int startPos, int windowSize, int firstRow) {
        const int b = blockIdx.y;
        const int i = firstRow + blockIdx.x;
        if (i >= seqlen) {
            return;
        }
        const int slot = (startPos + i) % windowSize;
        const uint8_t *src = chunk + ((uint64_t)b * seqlen + i) * rowBytes;
        uint8_t *dst = ring + ((uint64_t)b * windowSize + slot) * rowBytes;
        for (int k = threadIdx.x; k < rowBytes; k += blockDim.x) {
            dst[k] = src[k];
        }
    }

    // ---------------- host 辅助 ----------------

    bool V41OnCuda(const Data &d) {
        return d.dataDevice == DataDevice::CUDA && d.cudaData != nullptr;
    }

    bool V41IsFloatType(DataType t) {
        return t == DataType::FLOAT32 || t == DataType::FLOAT16 || t == DataType::BFLOAT16;
    }

    V41RopeTable V41BuildRope(int ropeDim, float base, int originalSeqLen, float factor, int betaFast, int betaSlow) {
        V41RopeTable table;
        table.pairs = ropeDim / 2;
        std::vector<float> invFreq;
        for (int i = 0; i < ropeDim; i += 2) {
            invFreq.push_back(1.0f / std::pow(base, (float)i / ropeDim));
        }
        if (originalSeqLen > 0) {
            auto correctedDim = [&](float rotations) {
                return ropeDim * std::log((float)originalSeqLen / (rotations * 2.0f * (float)M_PI)) /
                       (2.0f * std::log(base));
            };
            int low = std::max((int)std::floor(correctedDim((float)betaFast)), 0);
            int high = std::min((int)std::ceil(correctedDim((float)betaSlow)), ropeDim - 1);
            float denom = std::max((float)(high - low), 1e-3f);
            for (int i = 0; i < (int)invFreq.size(); i++) {
                float ramp = std::max(0.0f, std::min(1.0f, ((float)i - low) / denom));
                float smooth = 1.0f - ramp;
                invFreq[i] = invFreq[i] / factor * (1.0f - smooth) + invFreq[i] * smooth;
            }
        }
        for (int i = 0; i < 64; i++) {
            table.invFreq[i] = i < (int)invFreq.size() ? invFreq[i] : 0.0f;
        }
        return table;
    }

    bool V41CheckLaunch(const char *name) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[Fastllm] DeepSeekV41 CUDA kernel %s failed: %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
}

// ==================== 导出接口 ====================

extern "C" bool FastllmCudaDeepSeekV41HcMix(const fastllm::Data &x, fastllm::Data &hcFn, fastllm::Data &hcScale,
                                            fastllm::Data &hcBase, int hcMult, int sinkhornIters, float eps,
                                            float normEps, fastllm::Data &pre, fastllm::Data &post,
                                            fastllm::Data &comb) {
    if (!V41OnCuda(x) || x.dims.size() != 4 || hcMult <= 0 || hcMult > 4 || !V41IsFloatType(x.dataType) ||
        hcFn.dataType != DataType::FLOAT32 || hcScale.dataType != DataType::FLOAT32 ||
        hcBase.dataType != DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(pre, DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !V41PrepareOutput(post, DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !V41PrepareOutput(comb, DataType::FLOAT32, {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }
    hcFn.ToDevice(DataDevice::CUDA);
    hcScale.ToDevice(DataDevice::CUDA);
    hcBase.ToDevice(DataDevice::CUDA);
    if (!V41EnvOn("FASTLLM_DSV41_LEGACY_HCMIX")) {
        bool launched = false;
        if (x.dataType == DataType::BFLOAT16) {
            launched = V41LaunchHcMixMulti<__nv_bfloat16>((const __nv_bfloat16*)x.cudaData,
                (const float*)hcFn.cudaData, (const float*)hcScale.cudaData, (const float*)hcBase.cudaData,
                hcMult, tokens, dim, sinkhornIters, eps, normEps,
                (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
        } else if (x.dataType == DataType::FLOAT16) {
            launched = V41LaunchHcMixMulti<half>((const half*)x.cudaData,
                (const float*)hcFn.cudaData, (const float*)hcScale.cudaData, (const float*)hcBase.cudaData,
                hcMult, tokens, dim, sinkhornIters, eps, normEps,
                (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
        } else {
            launched = V41LaunchHcMixMulti<float>((const float*)x.cudaData,
                (const float*)hcFn.cudaData, (const float*)hcScale.cudaData, (const float*)hcBase.cudaData,
                hcMult, tokens, dim, sinkhornIters, eps, normEps,
                (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
        }
        if (launched) {
            return V41CheckLaunch("HcMix");
        }
    }
    if (x.dataType == DataType::BFLOAT16) {
        V41HcMixKernel<__nv_bfloat16><<<tokens, kHcThreads>>>(
            (const __nv_bfloat16*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    } else if (x.dataType == DataType::FLOAT16) {
        V41HcMixKernel<half><<<tokens, kHcThreads>>>(
            (const half*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    } else {
        V41HcMixKernel<float><<<tokens, kHcThreads>>>(
            (const float*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    }
    return V41CheckLaunch("HcMix");
}

extern "C" bool FastllmCudaDeepSeekV41HcApplyPre(const fastllm::Data &x, const fastllm::Data &pre, fastllm::Data &y) {
    if (!V41OnCuda(x) || !V41OnCuda(pre) || x.dims.size() != 4 || pre.dataType != DataType::FLOAT32 ||
        !V41IsFloatType(x.dataType)) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], hcMult = x.dims[2], dim = x.dims[3];
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(y, x.dataType, {bsz, seqlen, dim})) {
        return false;
    }
    uint64_t total = (uint64_t)tokens * dim;
    int threads = 256;
    unsigned blocks = (unsigned)((total + threads - 1) / threads);
    if (x.dataType == DataType::BFLOAT16) {
        V41HcApplyPreKernel<__nv_bfloat16><<<blocks, threads>>>((const __nv_bfloat16*)x.cudaData, (const float*)pre.cudaData,
                                                                (__nv_bfloat16*)y.cudaData, tokens, hcMult, dim);
    } else if (x.dataType == DataType::FLOAT16) {
        V41HcApplyPreKernel<half><<<blocks, threads>>>((const half*)x.cudaData, (const float*)pre.cudaData,
                                                       (half*)y.cudaData, tokens, hcMult, dim);
    } else {
        V41HcApplyPreKernel<float><<<blocks, threads>>>((const float*)x.cudaData, (const float*)pre.cudaData,
                                                        (float*)y.cudaData, tokens, hcMult, dim);
    }
    return V41CheckLaunch("HcApplyPre");
}

extern "C" bool FastllmCudaDeepSeekV41HcPreNorm(const fastllm::Data &x, const fastllm::Data &pre,
                                               fastllm::Data &normWeight, float eps, fastllm::Data &output) {
    if (V41EnvOn("FASTLLM_DSV41_DISABLE_HCPRENORM")) {
        return false;
    }
    if (!V41OnCuda(x) || !V41OnCuda(pre) || x.dims.size() != 4 || x.dataType != DataType::BFLOAT16 ||
        pre.dataType != DataType::FLOAT32 || x.multiDeviceData || pre.multiDeviceData ||
        normWeight.multiDeviceData) {
        return false;
    }
    const int bsz = x.dims[0], seqlen = x.dims[1], hcMult = x.dims[2], channels = x.dims[3];
    const int tokens = bsz * seqlen;
    if (tokens <= 0 || hcMult <= 0 || channels <= 0 || (channels & 1) != 0) {
        return false;
    }
    if (pre.Count(0) != (uint64_t)tokens * hcMult) {
        return false;
    }
    normWeight.ToDevice(DataDevice::CUDA);
    if (normWeight.dataType != DataType::FLOAT32 || normWeight.dims.size() != 1 ||
        normWeight.dims[0] != channels || normWeight.cudaData == nullptr) {
        return false;
    }
    // THREAD_PER_BLOCK 必须与 LaunchFastllmRMSNormBFloat16 的选择一致，否则归约树不同、
    // 结果不再逐 bit 相同。channels == 3072 走的是另一个专用 kernel，这里不接管。
    if (channels == 3072) {
        return false;
    }
    const int threads = channels < 512 ? 64 : (channels < 4096 ? 512 : 1024);
    const int bf2 = channels / 2;
    const int maxIter = (bf2 + threads - 1) / threads;
    if (!V41PrepareOutput(output, DataType::BFLOAT16, {bsz, seqlen, channels})) {
        return false;
    }
    const __nv_bfloat16 *xp = (const __nv_bfloat16*)x.cudaData;
    const float *prep = (const float*)pre.cudaData;
    const float *wp = (const float*)normWeight.cudaData;
    __nv_bfloat16 *op = (__nv_bfloat16*)output.cudaData;
#define V41_HCPRENORM_LAUNCH(T)                                                                     \
    switch (maxIter) {                                                                              \
        case 1: V41HcPreNormKernel<T, 1><<<tokens, T>>>(xp, prep, wp, op, hcMult, channels, eps); break;  \
        case 2: V41HcPreNormKernel<T, 2><<<tokens, T>>>(xp, prep, wp, op, hcMult, channels, eps); break;  \
        case 3: V41HcPreNormKernel<T, 3><<<tokens, T>>>(xp, prep, wp, op, hcMult, channels, eps); break;  \
        case 4: V41HcPreNormKernel<T, 4><<<tokens, T>>>(xp, prep, wp, op, hcMult, channels, eps); break;  \
        default: return false;                                                                      \
    }
    if (threads == 64) {
        V41_HCPRENORM_LAUNCH(64)
    } else if (threads == 512) {
        V41_HCPRENORM_LAUNCH(512)
    } else {
        V41_HCPRENORM_LAUNCH(1024)
    }
#undef V41_HCPRENORM_LAUNCH
    return V41CheckLaunch("HcPreNorm");
}

extern "C" bool FastllmCudaDeepSeekV41EngramApply(fastllm::Data &hidden, const fastllm::Data &kv,
                                                  fastllm::Data &qWeight, fastllm::Data &kWeight,
                                                  const fastllm::Data *mask, float eps, float clampValue) {
    if (!V41OnCuda(hidden) || !V41OnCuda(kv) || hidden.dims.size() != 4 ||
        hidden.dataType != DataType::BFLOAT16 || !V41IsFloatType(kv.dataType) ||
        qWeight.dataType != DataType::FLOAT32 || kWeight.dataType != DataType::FLOAT32) {
        return false;
    }
    int bsz = hidden.dims[0], seqlen = hidden.dims[1], hcMult = hidden.dims[2], dim = hidden.dims[3];
    int tokens = bsz * seqlen;
    qWeight.ToDevice(DataDevice::CUDA);
    kWeight.ToDevice(DataDevice::CUDA);
    const float *maskPtr = nullptr;
    if (mask != nullptr && mask->Count(0) > 0) {
        if (!V41OnCuda(*mask) || mask->dataType != DataType::FLOAT32) {
            return false;
        }
        maskPtr = (const float*)mask->cudaData;
    }
    dim3 grid(tokens * hcMult);
    if (kv.dataType == DataType::BFLOAT16) {
        V41EngramApplyKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const __nv_bfloat16*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    } else if (kv.dataType == DataType::FLOAT16) {
        V41EngramApplyKernel<__nv_bfloat16, half><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const half*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    } else {
        V41EngramApplyKernel<__nv_bfloat16, float><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const float*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    }
    return V41CheckLaunch("EngramApply");
}

extern "C" bool FastllmCudaDeepSeekV41RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                                  int posStep, bool inverse, int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, int quantMode, int quantDim,
                                                  int quantBlock) {
    if (!V41OnCuda(x) || (x.dims.size() != 3 && x.dims.size() != 4) || !V41IsFloatType(x.dataType)) {
        return false;
    }
    int dim = x.dims.back();
    if (quantDim <= 0) {
        quantDim = dim;
    }
    if (dim > 1024 || dim % 32 != 0 || ropeDim <= 0 || ropeDim > 128 || ropeDim > dim || (ropeDim & 1) != 0 ||
        (quantMode > 0 && ((quantBlock != 16 && quantBlock != 32) || quantDim % 32 != 0))) {
        return false;
    }
    int rowsPerToken = x.dims.size() == 4 ? x.dims[2] : 1;
    int rows = (int)(x.Count(0) / dim);
    if (rows <= 0) {
        return true;
    }
    V41RopeTable rope = V41BuildRope(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    if (V41LegacyRotary()) {
        size_t shared = (size_t)dim * sizeof(float);
        if (x.dataType == DataType::BFLOAT16) {
            V41RotaryQuantLegacyKernel<__nv_bfloat16><<<rows, dim, shared>>>((__nv_bfloat16*)x.cudaData, rowsPerToken,
                dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
        } else if (x.dataType == DataType::FLOAT16) {
            V41RotaryQuantLegacyKernel<half><<<rows, dim, shared>>>((half*)x.cudaData, rowsPerToken, dim, rope,
                ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
        } else {
            V41RotaryQuantLegacyKernel<float><<<rows, dim, shared>>>((float*)x.cudaData, rowsPerToken, dim, rope,
                ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
        }
        return V41CheckLaunch("RotaryQuant");
    }
    if (quantMode <= 0) {
        // 只旋转，不量化：按「行 x 旋转对」展开，一个 block 256 个对
        const uint64_t totalPairs = (uint64_t)rows * (uint64_t)(ropeDim >> 1);
        const int threads = 256;
        const uint64_t blocks = (totalPairs + threads - 1) / threads;
        if (x.dataType == DataType::BFLOAT16) {
            V41RotaryOnlyKernel<__nv_bfloat16><<<(unsigned)blocks, threads>>>((__nv_bfloat16*)x.cudaData,
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs);
        } else if (x.dataType == DataType::FLOAT16) {
            V41RotaryOnlyKernel<half><<<(unsigned)blocks, threads>>>((half*)x.cudaData,
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs);
        } else {
            V41RotaryOnlyKernel<float><<<(unsigned)blocks, threads>>>((float*)x.cudaData,
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs);
        }
        return V41CheckLaunch("RotaryQuant");
    }
    // 量化：一个 block 处理 rowsPerBlock 行，目标 blockDim 512
    const int rowsPerBlock = std::max(1, 512 / dim);
    const int threads = rowsPerBlock * dim;
    const int blocks = (rows + rowsPerBlock - 1) / rowsPerBlock;
    if (x.dataType == DataType::BFLOAT16) {
        V41RotaryQuantKernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows);
    } else if (x.dataType == DataType::FLOAT16) {
        V41RotaryQuantKernel<half><<<blocks, threads>>>((half*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows);
    } else {
        V41RotaryQuantKernel<float><<<blocks, threads>>>((float*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows);
    }
    return V41CheckLaunch("RotaryQuant");
}

extern "C" bool FastllmCudaDeepSeekV41Compress(const fastllm::Data &kv, const fastllm::Data *score,
                                               fastllm::Data &normWeight, int ratio, float normEps,
                                               fastllm::Data &output) {
    if (!V41OnCuda(kv) || kv.dims.size() != 3 || kv.dims[2] != 512 || ratio <= 0 || kv.dims[1] % ratio != 0 ||
        !V41IsFloatType(kv.dataType) || (ratio > 1 && (score == nullptr || !V41OnCuda(*score) ||
                                                        score->dataType != kv.dataType))) {
        return false;
    }
    int bsz = kv.dims[0], n = kv.dims[1], dim = kv.dims[2];
    int blocks = n / ratio;
    normWeight.ToDevice(DataDevice::CUDA);
    if (normWeight.dataType != DataType::FLOAT32) {
        return false;
    }
    if (!V41PrepareOutput(output, DataType::BFLOAT16, {bsz, blocks, dim})) {
        return false;
    }
    size_t shared = (size_t)dim * sizeof(float);
    const void *scorePtr = ratio > 1 ? score->cudaData : nullptr;
    if (kv.dataType == DataType::FLOAT32) {
        V41CompressKernel<float><<<bsz * blocks, dim, shared>>>((const float*)kv.cudaData, (const float*)scorePtr,
            (const float*)normWeight.cudaData, n, dim, ratio, normEps, (__nv_bfloat16*)output.cudaData);
    } else if (kv.dataType == DataType::BFLOAT16) {
        V41CompressKernel<__nv_bfloat16><<<bsz * blocks, dim, shared>>>((const __nv_bfloat16*)kv.cudaData,
            (const __nv_bfloat16*)scorePtr, (const float*)normWeight.cudaData, n, dim, ratio, normEps,
            (__nv_bfloat16*)output.cudaData);
    } else {
        V41CompressKernel<half><<<bsz * blocks, dim, shared>>>((const half*)kv.cudaData, (const half*)scorePtr,
            (const float*)normWeight.cudaData, n, dim, ratio, normEps, (__nv_bfloat16*)output.cudaData);
    }
    return V41CheckLaunch("Compress");
}

extern "C" bool FastllmCudaDeepSeekV41IndexerScore(const fastllm::Data &q, const fastllm::Data &weights,
                                                   const fastllm::Data &k, int ratio, int startPos,
                                                   fastllm::Data &output) {
    const bool kQuant = k.dataType == DataType::INT8;
    int kMode = 1, kBlock = 32;
    if (!V41OnCuda(q) || !V41OnCuda(weights) || !V41OnCuda(k) || q.dims.size() != 4 || k.dims.size() != 3 ||
        q.dims[3] != 128 || weights.dataType != DataType::FLOAT32 ||
        (q.dataType != DataType::BFLOAT16 && q.dataType != DataType::FLOAT32) ||
        (kQuant ? !V41ParseKvRowHost(128, k.dims[2], &kMode, &kBlock)
                : (k.dims[2] != 128 ||
                   (k.dataType != DataType::BFLOAT16 && k.dataType != DataType::FLOAT32)))) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int m = k.dims[1];
    const uint8_t *k8 = kQuant ? (const uint8_t*)k.cudaData : nullptr;
    const int kRowBytes = kQuant ? k.dims[2] : 0;
    if (!V41PrepareOutput(output, DataType::FLOAT32, {bsz, seqlen, m})) {
        return false;
    }
    if (m == 0) {
        return true;
    }
    // BF16 mma 快速路径（SM80+）；FASTLLM_DSV41_LEGACY_INDEXER=1 退回下面的标量 kernel
    if (q.dataType == DataType::BFLOAT16 && (kQuant || k.dataType == DataType::BFLOAT16) && dim == kIdxDim &&
        V41MmaSupported() && !V41EnvOn("FASTLLM_DSV41_LEGACY_INDEXER")) {
        dim3 mmaGrid((m + kIdxBJ - 1) / kIdxBJ, (seqlen + kIdxBT - 1) / kIdxBT, bsz);
        V41IndexerScoreMmaKernel<<<mmaGrid, kIdxMmaThreads>>>(
            (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData,
            kQuant ? nullptr : (const __nv_bfloat16*)k.cudaData,
            seqlen, heads, m, ratio, startPos, (float*)output.cudaData, k8, kMode, kRowBytes,
            V41Fp4VecUnpack() ? 1 : 0);
        return V41CheckLaunch("IndexerScoreMma");
    }

    dim3 grid((m + kIdxThreads - 1) / kIdxThreads, bsz * seqlen);
    size_t shared = (size_t)(heads * dim + heads) * sizeof(float);
    if (kQuant) {
        if (q.dataType == DataType::BFLOAT16) {
            V41IndexerScoreKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
                (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData, nullptr,
                seqlen, heads, dim, m, (float*)output.cudaData, k8, kMode, kRowBytes);
        } else {
            V41IndexerScoreKernel<float, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
                (const float*)q.cudaData, (const float*)weights.cudaData, nullptr,
                seqlen, heads, dim, m, (float*)output.cudaData, k8, kMode, kRowBytes);
        }
    } else if (q.dataType == DataType::BFLOAT16 && k.dataType == DataType::BFLOAT16) {
        V41IndexerScoreKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
            (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData, (const __nv_bfloat16*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else if (q.dataType == DataType::FLOAT32 && k.dataType == DataType::BFLOAT16) {
        V41IndexerScoreKernel<float, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
            (const float*)q.cudaData, (const float*)weights.cudaData, (const __nv_bfloat16*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else if (q.dataType == DataType::BFLOAT16 && k.dataType == DataType::FLOAT32) {
        V41IndexerScoreKernel<__nv_bfloat16, float><<<grid, kIdxThreads, shared>>>(
            (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData, (const float*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else {
        V41IndexerScoreKernel<float, float><<<grid, kIdxThreads, shared>>>(
            (const float*)q.cudaData, (const float*)weights.cudaData, (const float*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    }
    return V41CheckLaunch("IndexerScore");
}

extern "C" bool FastllmCudaDeepSeekV41CandidateBlocks(const fastllm::Data &score, int blockSize, int topkBlocks,
                                                      int ratio, int startPos, fastllm::Data &output) {
    if (!V41OnCuda(score) || score.dims.size() != 3 || score.dataType != DataType::FLOAT32 || blockSize <= 0) {
        return false;
    }
    int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
    int numBlocks = (m + blockSize - 1) / blockSize;
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(output, DataType::INT8, {bsz, seqlen, numBlocks})) {
        return false;
    }
    float *blockScore = (float*)FastllmCudaMalloc((size_t)tokens * numBlocks * sizeof(float));
    if (blockScore == nullptr) {
        return false;
    }
    V41BlockScoreKernel<<<tokens, 256>>>((const float*)score.cudaData, seqlen, m, blockSize, numBlocks,
                                         ratio, startPos, blockScore);
    V41CandidateSelectKernel<<<tokens, kSelThreads>>>(blockScore, numBlocks, topkBlocks, (uint8_t*)output.cudaData);
    bool ok = V41CheckLaunch("CandidateBlocks");
    cudaDeviceSynchronize();
    FastllmCudaFree(blockScore);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV41IndexerTopK(const fastllm::Data &score, const fastllm::Data *candidates,
                                                  int topK, int ratio, int startPos, int blockSize,
                                                  fastllm::Data &output) {
    if (!V41OnCuda(score) || score.dims.size() != 3 || score.dataType != DataType::FLOAT32 || topK <= 0) {
        return false;
    }
    int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
    int width = std::min(topK, m);
    const uint8_t *cand = nullptr;
    int numBlocks = 0;
    if (candidates != nullptr && candidates->Count(0) > 0) {
        if (!V41OnCuda(*candidates) || candidates->dataType != DataType::INT8 || candidates->dims.size() != 3) {
            return false;
        }
        cand = (const uint8_t*)candidates->cudaData;
        numBlocks = candidates->dims[2];
    }
    if (!V41PrepareOutput(output, DataType::INT32, {bsz, seqlen, width})) {
        return false;
    }
    if (width == 0) {
        return true;
    }
    const int bs = std::max(1, blockSize);
    if (V41LegacyTopK()) {
        V41TopKKernelLegacy<<<bsz * seqlen, kSelThreads>>>((const float*)score.cudaData, cand, seqlen, m, numBlocks,
                                                           bs, topK, width, ratio, startPos,
                                                           (int32_t*)output.cudaData);
        return V41CheckLaunch("IndexerTopK");
    }
    // blockSize 是 2 的幂时才走候选块压缩（下标换算只用移位）
    int blockShift = -1;
    for (int s = 0; s < 31; s++) {
        if ((1 << s) == bs) {
            blockShift = s;
            break;
        }
    }
    // 候选块下标的压缩表放动态共享内存。容量按 numBlocks 取，但封顶 4096（16 KB）：
    // 真实配置里候选块数是 candidate_topk_blocks（2048）量级，远小于长上下文的 numBlocks，
    // 所以封顶后绝大多数情况仍然能压缩；真的装不下时 kernel 内部会退回逐 j 查掩码。
    const int kListCap = 4096;
    const int listCap = (cand != nullptr && blockShift >= 0 && numBlocks > 0)
                        ? std::min(numBlocks, kListCap) : 0;
    V41TopKKernel<<<bsz * seqlen, kSelThreads, (size_t)listCap * sizeof(int)>>>(
        (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
        ratio, startPos, listCap, (int32_t*)output.cudaData);
    return V41CheckLaunch("IndexerTopK");
}

extern "C" bool FastllmCudaDeepSeekV41SparseAttention(const fastllm::Data &q, const fastllm::Data &chunkKV,
                                                      const fastllm::Data *ringKV, const fastllm::Data *compressedKV,
                                                      const fastllm::Data *cmpIdx, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, float softmaxScale,
                                                      fastllm::Data &output) {
    if (!V41OnCuda(q) || !V41OnCuda(chunkKV) || q.dims.size() != 4 || chunkKV.dims.size() != 3 ||
        q.dims[3] != 512 || q.dims[2] % kAttnHeadsPerBlock != 0 || chunkKV.dataType != DataType::BFLOAT16 ||
        (q.dataType != DataType::BFLOAT16 && q.dataType != DataType::FLOAT32)) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    bool hasRing = ringKV != nullptr && ringKV->dims.size() == 3 && ringKV->Count(0) > 0;
    bool hasCmp = compressedKV != nullptr && cmpIdx != nullptr && compressedKV->dims.size() == 3 &&
                  compressedKV->Count(0) > 0 && cmpIdx->dims.size() == 3;
    bool ringFp8 = hasRing && ringKV->dataType == DataType::INT8;
    bool cmpFp8 = hasCmp && compressedKV->dataType == DataType::INT8;
    int ringMode = 1, ringBlock = 32, cmpMode = 1, cmpBlock = 32;
    if (hasRing && (!V41OnCuda(*ringKV) || ringKV->dims[1] != windowSize ||
                    !(ringFp8 ? V41ParseKvRowHost(dim, ringKV->dims[2], &ringMode, &ringBlock)
                              : (ringKV->dataType == DataType::BFLOAT16 && ringKV->dims[2] == dim)))) {
        return false;
    }
    if (hasCmp && (!V41OnCuda(*compressedKV) || !V41OnCuda(*cmpIdx) || cmpIdx->dataType != DataType::INT32 ||
                   !(cmpFp8 ? V41ParseKvRowHost(dim, compressedKV->dims[2], &cmpMode, &cmpBlock)
                            : (compressedKV->dataType == DataType::BFLOAT16 && compressedKV->dims[2] == dim)))) {
        return false;
    }
    const int ringRowBytes = hasRing && ringFp8 ? ringKV->dims[2] : 0;
    const int cmpRowBytes = hasCmp && cmpFp8 ? compressedKV->dims[2] : 0;
    // 向量化解包按 16 字节一组读缓存行，需要行宽是 16 的倍数（现有三种布局都满足）
    const bool vecUnpack = V41Fp4VecUnpack() && (ringRowBytes % 16 == 0) && (cmpRowBytes % 16 == 0);
    if (!hasRing && startPos > 0) {
        return false;
    }
    attnSink.ToDevice(DataDevice::CUDA);
    if (attnSink.dataType != DataType::FLOAT32) {
        return false;
    }
    if (!V41PrepareOutput(output, DataType::BFLOAT16, q.dims)) {
        return false;
    }
    int cap = hasCmp ? compressedKV->dims[1] : 0;
    int topWidth = hasCmp ? cmpIdx->dims[2] : 0;

    // BF16 mma 快速路径（SM80+，head_dim 512，q 为 BF16）。
    // FASTLLM_DSV41_LEGACY_ATTN=1 可退回下面的 FP32 标量 kernel 做对比 / 排查。
    if (q.dataType == DataType::BFLOAT16 && dim == kMmaDim && heads % kMmaHeads == 0 &&
        V41MmaSupported() && !V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN")) {
        // cudaFuncSetAttribute 是**按设备**生效的：多卡（张量并行 / 按层切分）下
        // 每张卡都要设一次。原来的进程级 static bool 会让第二张卡漏设，
        // 动态共享内存超过默认 48 KB，kernel 启动直接返回 invalid argument。
        // 张量并行时两个 worker 线程并发进入这里，还会退化成偶发失败。
        {
            static std::mutex mmaSharedMutex;
            static bool mmaSharedReady[16] = {false};
            const int mmaDevice = FastllmCudaGetDevice();
            if (mmaDevice >= 0 && mmaDevice < 16) {
                std::lock_guard<std::mutex> lock(mmaSharedMutex);
                if (!mmaSharedReady[mmaDevice]) {
                    cudaFuncSetAttribute(V41SparseAttentionMmaKernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)sizeof(V41MmaShared));
                    mmaSharedReady[mmaDevice] = true;
                }
            }
        }
        const int tokens = bsz * seqlen;
        const int headBlocks = heads / kMmaHeads;
        const int maxCand = std::min(windowSize, startPos + seqlen) + topWidth;
        const int maxTiles = std::max(1, (maxCand + kMmaNC - 1) / kMmaNC);
        int splits = 1;
        const char *splitEnv = getenv("FASTLLM_DSV41_ATTN_SPLITS");
        if (splitEnv != nullptr && splitEnv[0] != '\0') {
            splits = std::max(1, std::min(atoi(splitEnv), maxTiles));
        } else {
            // decode（token 数少）时候选维 split-K，把 block 数补到约 4 倍 SM 数
            const int target = 4 * V41SmCount();
            const long long base = (long long)tokens * headBlocks;
            if (base > 0 && base < target && maxTiles > 1) {
                splits = (int)std::min((long long)maxTiles, (target + base - 1) / base);
                splits = std::max(1, std::min(splits, 32));
            }
        }
        // 部分和缓冲是 [splits, tokens, heads, dim + 2] 的 FP32，prefill 时强行 split 会非常大，
        // 超过预算就把 split 数减半（正确性不受影响，只是并行度降低）
        while (splits > 1 &&
               (size_t)splits * tokens * heads * (kMmaDim + 2) * sizeof(float) > (size_t)512 * 1024 * 1024) {
            splits /= 2;
        }
        float *partAcc = nullptr, *partMx = nullptr, *partL = nullptr;
        size_t partCount = 0;
        if (splits > 1) {
            partCount = (size_t)splits * tokens * heads;
            partAcc = (float*)FastllmCudaMalloc(partCount * (kMmaDim + 2) * sizeof(float));
            if (partAcc == nullptr) {
                splits = 1;
            } else {
                partMx = partAcc + partCount * kMmaDim;
                partL = partMx + partCount;
            }
        }
        dim3 grid(tokens, headBlocks, splits);
        V41SparseAttentionMmaKernel<<<grid, kMmaThreads, sizeof(V41MmaShared)>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData, partAcc, partMx, partL,
            ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
        if (splits > 1) {
            V41SparseMergeKernel<<<tokens * heads, 128>>>(partAcc, partMx, partL, tokens, heads, splits,
                                                          (__nv_bfloat16*)output.cudaData);
            FastllmCudaFree(partAcc);
        }
        return V41CheckLaunch("SparseAttentionMma");
    }

    if (q.dataType == DataType::BFLOAT16) {
        dim3 grid(bsz * seqlen, heads / kAttnHeadsPerBlock);
        V41SparseAttentionKernel<__nv_bfloat16><<<grid, kAttnThreads>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, dim, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData, ringMode, ringRowBytes, cmpMode, cmpRowBytes);
    } else {
        dim3 grid(bsz * seqlen, heads / kAttnHeadsPerBlock);
        V41SparseAttentionKernel<float><<<grid, kAttnThreads>>>(
            (const float*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, dim, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData, ringMode, ringRowBytes, cmpMode, cmpRowBytes);
    }
    return V41CheckLaunch("SparseAttention");
}

extern "C" bool FastllmCudaDeepSeekV41WindowStore(const fastllm::Data &chunk, fastllm::Data &ring, int startPos,
                                                  int windowSize) {
    if (!V41OnCuda(chunk) || chunk.dims.size() != 3) {
        return false;
    }
    int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
    if (ring.dims.size() != 3 || ring.dims[0] != bsz || ring.dims[1] != windowSize || ring.dims[2] != dim ||
        ring.dataType != chunk.dataType || !V41OnCuda(ring)) {
        if (!V41PrepareOutput(ring, chunk.dataType, {bsz, windowSize, dim})) {
            return false;
        }
        cudaMemset(ring.cudaData, 0, (size_t)ring.Count(0) * ring.unitSize / ring.unitSizeDiv);
    }
    int rowBytes = dim * chunk.unitSize;
    int firstRow = std::max(0, seqlen - windowSize);
    dim3 grid(seqlen - firstRow, bsz);
    V41WindowStoreKernel<<<grid, 256>>>((const uint8_t*)chunk.cudaData, (uint8_t*)ring.cudaData, seqlen, rowBytes,
                                        startPos, windowSize, firstRow);
    return V41CheckLaunch("WindowStore");
}

extern "C" bool FastllmCudaDeepSeekV41QuantizeKV(const fastllm::Data &input, fastllm::Data &output,
                                                 int quantMode, int quantBlock) {
    if (!V41OnCuda(input) || input.dims.size() != 3 || !V41IsFloatType(input.dataType) ||
        (quantMode != 1 && quantMode != 2 && quantMode != 3) ||
        (quantBlock != 16 && quantBlock != 32) || input.dims[2] % quantBlock != 0 ||
        input.dims[2] % 2 != 0 || kKvQuantThreads % quantBlock != 0) {
        return false;
    }
    int rows = input.dims[0] * input.dims[1], dim = input.dims[2];
    const int rowBytes = V41KvRowBytesHost(dim, quantMode, quantBlock);
    if (!V41PrepareOutput(output, DataType::INT8, {input.dims[0], input.dims[1], rowBytes})) {
        return false;
    }
    if (rows == 0) {
        return true;
    }
    if (input.dataType == DataType::BFLOAT16) {
        V41QuantizeKVKernel<__nv_bfloat16><<<rows, kKvQuantThreads>>>(
            (const __nv_bfloat16*)input.cudaData, (uint8_t*)output.cudaData, rows, dim,
            quantMode, quantBlock, rowBytes);
    } else if (input.dataType == DataType::FLOAT16) {
        V41QuantizeKVKernel<half><<<rows, kKvQuantThreads>>>(
            (const half*)input.cudaData, (uint8_t*)output.cudaData, rows, dim,
            quantMode, quantBlock, rowBytes);
    } else {
        V41QuantizeKVKernel<float><<<rows, kKvQuantThreads>>>(
            (const float*)input.cudaData, (uint8_t*)output.cudaData, rows, dim,
            quantMode, quantBlock, rowBytes);
    }
    return V41CheckLaunch("QuantizeKV");
}
