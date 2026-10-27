// SM120 NVFP4 fused prefill implementation. Native weights are consumed
// directly: CTA TMA for codes, contiguous bulk copies for scales, warp-specialized
// double buffering, and shared-memory staging for vectorized FP16 output.
// Benchmarked shapes: gate/up N=34816 K=5120; residual down N=5120 K=17408.
// CUDA 13.1; no extra weight copy or persistent layout cache.
// Logical M may be any positive number. A/GX/R/Y contain exactly M rows.
// AS keeps the native 256-row-tiled layout (fresh_as_bytes reports capacity).
// Only valid AS rows are fetched; invalid shared rows are explicitly zeroed.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>
#include <string>
namespace fastllm_native_prefill {
namespace fresh_nvfp4 {
__device__ __forceinline__ unsigned smem(const void *p) { return __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mma(float *c, unsigned *a, unsigned *b, unsigned sa, unsigned sb) {
    const unsigned short zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"(zero), "h"(zero),
          "r"(sb), "h"(zero), "h"(zero));
}
__device__ __forceinline__ size_t bsindex(int row, int group, int K) {
    return (size_t(row / 128) * (K / 64) + group / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 +
           group % 4;
}
__device__ __forceinline__ size_t asindex(int row, int group, int K) {
    return (size_t(row / 256) * (K / 256) + group / 16) * 4096 + (row % 256) * 16 + group % 16;
}
template <int MODE, int WN> __device__ __forceinline__ int weightrow(int local, int start, int N) {
    if constexpr (MODE == 1)
        return start + local / WN * (WN / 2) + local % (WN / 2) + (local % WN / (WN / 2)) * (N / 2);
    else
        return start + local;
}
struct Descriptors {
    alignas(128) CUtensorMap a, b;
};
__device__ __forceinline__ void tmaload(void *dst, const CUtensorMap *desc, int x, int y, uint64_t *bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, "
                 "{%2,%3}], [%4];" ::"r"(smem(dst)),
                 "l"(desc), "r"(x), "r"(y), "r"(smem(bar))
                 : "memory");
}
__device__ __forceinline__ void wait(uint64_t *bar, int phase) {
    unsigned ready;
    do {
        asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 0x989680; "
                     "selp.u32 %0, 1, 0, p;}"
                     : "=r"(ready)
                     : "r"(smem(bar)), "r"(phase)
                     : "memory");
    } while (!ready);
}
__device__ __forceinline__ void bulk(void *dst, const void *src, unsigned bytes, uint64_t *bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     smem(dst)),
                 "l"(src), "r"(bytes), "r"(smem(bar))
                 : "memory");
}
template <int BM, int BN, int WARPS, int STAGES, int MODE, bool TAIL>
__device__ __forceinline__ void loadtile(uint8_t *shared, const Descriptors &desc, uint64_t *bar, int tile,
                                         int K, int N, int startM, int startN, uint8_t *asbase,
                                         const uint8_t *AS, const uint8_t *BS, int validRows) {
    constexpr int blocks = MODE == 1 ? 2 : (BN + 127) / 128;
    constexpr int bytes = (BM + BN) * 64 + BM * 16 + blocks * 1024;
    if (threadIdx.x == 0 && tile < K / 128) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem(bar)),
                     "r"(bytes - BM * 16 + ((tile & 1) ? 0 : (TAIL ? validRows : BM) * 16))
                     : "memory");
        tmaload(shared, &desc.a, tile * 64, startM, bar);
        if constexpr (MODE == 1) {
#pragma unroll
            for (int block = 0; block < 2; block++)
                tmaload(shared + BM * 64 + block * (BN / 2) * 64, &desc.b, tile * 64,
                        startN + block * (N / 2), bar);
        } else
            tmaload(shared + BM * 64, &desc.b, tile * 64, startN, bar);
        uint8_t *bs = shared + (BM + BN) * 64 + BM * 16;
        if ((tile & 1) == 0)
            bulk(asbase, AS + asindex(startM, tile * 8, K), (TAIL ? validRows : BM) * 16, bar);
#pragma unroll
        for (int block = 0; block < blocks; block++) {
            int globalrow = startN + (MODE == 1 ? block * (N / 2) : block * 128);
            bulk(bs + block * 1024, BS + bsindex(globalrow / 128 * 128, tile * 8, K), 1024, bar);
        }
    }
}
template <int BM, int BN, int WARPS, int STAGES, int MODE, bool TAIL>
__global__ __launch_bounds__(WARPS * 32 + 128,
                             1) void kernel(const __grid_constant__ Descriptors desc, const uint8_t *A,
                                            const uint8_t *AS, const uint8_t *B, const uint8_t *BS,
                                            const float *GX, const float *GW, const half *R, half *Y, int M,
                                            int unusedN, int unusedK, int group) {
    constexpr int N = MODE == 1 ? 34816 : 5120, K = MODE == 1 ? 5120 : 17408;
    constexpr int WM = BM / (WARPS / 2), WN = BN / 2, MM = WM / 16, NN = WN / 8,
                  STRIDE = (BM + BN) * 64 + BM * 16 + (MODE == 1 ? 2 : (BN + 127) / 128) * 1024;
    extern __shared__ __align__(128) uint8_t shared[];
    int startM = blockIdx.x * BM, startN = blockIdx.y * (MODE == 1 ? BN / 2 : BN);
    int validRows = TAIL ? min(BM, M - startM) : BM;
    int warp = (int(threadIdx.x) - 128) / 32, lane = threadIdx.x % 32, wm = warp / 2, wn = warp % 2;
    uint64_t *bars = reinterpret_cast<uint64_t *>(
                 shared + ((STRIDE * STAGES > BM * ((MODE == 1 ? BN / 2 : BN) + 8) * 2)
                               ? STRIDE * STAGES
                               : BM * ((MODE == 1 ? BN / 2 : BN) + 8) * 2)),
             *empty = bars + STAGES;
    // AS padding need not be initialized by the caller. No global padding is read.
    if constexpr (TAIL) {
        for (int row = validRows + int(threadIdx.x); row < BM; row += blockDim.x) {
#pragma unroll
            for (int slot = 0; slot < STAGES / 2 + 1; slot++)
                *reinterpret_cast<uint4 *>(shared + slot * STRIDE + (BM + BN) * 64 + row * 16) =
                    make_uint4(0, 0, 0, 0);
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < STAGES; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem(bars + i)) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem(empty + i)), "r"(WARPS * 32)
                         : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0) {
            for (int tile = 0; tile < K / 128; tile++) {
                int stage = tile % STAGES;
                wait(empty + stage, 1 ^ ((tile / STAGES) & 1));
                loadtile<BM, BN, WARPS, STAGES, MODE, TAIL>(
                    shared + stage * STRIDE, desc, bars + stage, tile, K, N, startM, startN,
                    shared + ((tile / 2) % (STAGES / 2 + 1)) * STRIDE + (BM + BN) * 64, AS, BS, validRows);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    float acc[MM][NN][4] = {};
#pragma unroll 1
    for (int tile = 0; tile < K / 128; tile++) {
        wait(bars + tile % STAGES, (tile / STAGES) & 1);
        uint8_t *sa = shared + (tile % STAGES) * STRIDE;
        uint8_t *sb = sa + BM * 64;
#pragma unroll
        for (int part = 0; part < 2; part++) {
            unsigned af[MM][4], bf[NN][2], as[MM], bs[NN];
#pragma unroll
            for (int i = 0; i < MM; i++) {
                int row = wm * WM + i * 16 + (lane & 15), col = part * 32 + (lane / 16) * 16;
                unsigned addr = smem(sa + row * 64 + ((col / 16) ^ ((row >> 1) & 3)) * 16);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                             : "r"(addr));
                int sr = startM + wm * WM + i * 16 + ((lane & 1) * 8 + lane / 4);
                int offset = (sr - startM) * 16 + (tile % 2) * 8 + part * 4;
                as[i] = *reinterpret_cast<const unsigned *>(
                    shared + ((tile / 2) % (STAGES / 2 + 1)) * STRIDE + (BM + BN) * 64 + offset);
            }
#pragma unroll
            for (int j = 0; j < NN; j++) {
                int local = wn * WN + j * 8 + (lane & 7);
                int row = MODE == 1
                              ? local / WN * (WN / 2) + local % (WN / 2) + (local % WN / (WN / 2)) * (BN / 2)
                              : local;
                int col = part * 32 + ((lane / 8) & 1) * 16;
                unsigned addr = smem(sb + row * 64 + ((col / 16) ^ ((row >> 1) & 3)) * 16);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bf[j][0]), "=r"(bf[j][1])
                             : "r"(addr));
                int sr = weightrow<MODE, WN>(wn * WN + j * 8 + lane / 4, startN, N);
                int block = MODE == 1 ? (j / (NN / 2)) : ((wn * WN + j * 8 + lane / 4) / 128);
                bs[j] =
                    *reinterpret_cast<const unsigned *>(sa + (BM + BN) * 64 + BM * 16 + block * 1024 +
                                                        part * 512 + (sr % 32) * 16 + (sr % 128 / 32) * 4);
            }
#pragma unroll
            for (int i = 0; i < MM; i++) {
#pragma unroll
                for (int j = 0; j < NN; j++)
                    mma(acc[i][j], af[i], bf[j], as[i], bs[j]);
            }
        }
        __syncwarp(); // Publish all lanes' shared reads before releasing the slot.
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem(empty + tile % STAGES))
                     : "memory");
    }
    // Reuse the finished input ring for coalesced output staging.
    asm volatile("bar.sync 1, %0;" ::"r"(WARPS * 32) : "memory");
    constexpr int OW = MODE == 1 ? BN / 2 : BN, OS = OW + 8;
    half *out = reinterpret_cast<half *>(shared);
    const float alpha = *GW / 128.f;
#pragma unroll
    for (int i = 0; i < MM; i++) {
        int row0 = wm * WM + i * 16 + lane / 4;
        float scale0 = 0.f, scale1 = 0.f;
        if (!TAIL || row0 < validRows)
            scale0 = GX[startM + row0] * alpha;
        if (!TAIL || row0 + 8 < validRows)
            scale1 = GX[startM + row0 + 8] * alpha;
#pragma unroll
        for (int j = 0; j < (MODE == 1 ? NN / 2 : NN); j++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                float scale = h ? scale1 : scale0;
                int col = wn * (MODE == 1 ? WN / 2 : WN) + j * 8 + 2 * (lane % 4);
                half2 v = __floats2half2_rn(acc[i][j][h * 2] * scale, acc[i][j][h * 2 + 1] * scale);
                if constexpr (MODE == 1) {
                    half2 up = __floats2half2_rn(acc[i][j + NN / 2][h * 2] * scale,
                                                 acc[i][j + NN / 2][h * 2 + 1] * scale);
                    half2 den = __hadd2(__float2half2_rn(1.f), h2exp(__hneg2(v)));
                    float2 num = __half22float2(v), df = __half22float2(den);
                    half2 silu = __floats2half2_rn(__fdividef(num.x, df.x), __fdividef(num.y, df.y));
                    v = __hmul2(silu, up);
                }
                *reinterpret_cast<half2 *>(out + (row0 + h * 8) * OS + col) = v;
            }
        }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(WARPS * 32) : "memory");
    for (int v = int(threadIdx.x) - 128; v < BM * OW / 8; v += WARPS * 32) {
        int row = v / (OW / 8), col = (v % (OW / 8)) * 8;
        if constexpr (TAIL) {
            if (row >= validRows)
                continue;
        }
        uint4 value = *reinterpret_cast<uint4 *>(out + row * OS + col);
        size_t dst = size_t(startM + row) * (MODE == 1 ? N / 2 : N) + startN + col;
        if constexpr (MODE == 2) {
            uint4 rv = *reinterpret_cast<const uint4 *>(R + dst);
            half2 *a = reinterpret_cast<half2 *>(&value), *b = reinterpret_cast<half2 *>(&rv);
#pragma unroll
            for (int i = 0; i < 4; i++)
                a[i] = __hadd2(a[i], b[i]);
        }
        *reinterpret_cast<uint4 *>(Y + dst) = value;
    }
}

static constexpr int SharedBytes(int mode) {
    // Input ring shares storage with the epilogue; barriers sit after both.
    constexpr int bm = 256, bn = 128, stages = 2;
    int inputBytes = ((bm + bn) * 64 + bm * 16 + (mode == 1 ? 2 : 1) * 1024) * stages;
    int outputBytes = bm * ((mode == 1 ? bn / 2 : bn) + 8) * 2;
    return (inputBytes > outputBytes ? inputBytes : outputBytes) + stages * 16;
}
static const void *Function(int mode, bool tail) {
    if (mode == 1)
        return tail ? (const void *)kernel<256, 128, 8, 2, 1, true>
                    : (const void *)kernel<256, 128, 8, 2, 1, false>;
    return tail ? (const void *)kernel<256, 128, 8, 2, 2, true>
                : (const void *)kernel<256, 128, 8, 2, 2, false>;
}
static bool CanRun(int m, int n, int k, int mode) {
    const char *flag = std::getenv("FASTLLM_CUDA_NATIVE_NVFP4_FRESH");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    if (m < 256 || m > 4096 ||
        !((mode == 1 && n == 34816 && k == 5120) || (mode == 2 && n == 5120 && k == 17408)))
        return false;
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        major != 12 || minor != 0)
        return false;
    static thread_local std::map<int, bool> ready;
    auto it = ready.find(device);
    if (it != ready.end())
        return it->second;
    int shared = 0;
    bool valid =
        cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess;
    for (int md : {1, 2})
        for (bool tail : {false, true}) {
            cudaFuncAttributes attr{};
            auto fn = Function(md, tail);
            valid = valid && shared >= SharedBytes(md) && cudaFuncGetAttributes(&attr, fn) == cudaSuccess &&
                    attr.binaryVersion >= 120 && attr.maxThreadsPerBlock >= 384 &&
                    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes(md)) ==
                        cudaSuccess;
        }
    if (!valid)
        cudaGetLastError();
    ready.emplace(device, valid);
    return valid;
}
static bool Run(const uint8_t *x, const uint8_t *xs, const uint8_t *w, const uint8_t *ws, const float *gx,
                const float *gw, half *y, int m, int n, int k, int mode) {
    if (!CanRun(m, n, k, mode))
        return false;
    Descriptors desc{};
    auto encode = [](CUtensorMap *d, const void *p, int width, int height, int rows) {
        uint64_t shape[] = {uint64_t(width), uint64_t(height)}, strides[] = {uint64_t(width)};
        uint32_t box[] = {64, uint32_t(rows)}, elem[] = {1, 1};
        return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(p), shape,
                                      strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    };
    if (!encode(&desc.a, x, k / 2, m, 256) || !encode(&desc.b, w, k / 2, n, mode == 1 ? 64 : 128))
        return false;
    const half *residual = mode == 2 ? y : nullptr;
    int group = 1;
    void *args[] = {&desc, &x, &xs, &w, &ws, &gx, &gw, &residual, &y, &m, &n, &k, &group};
    auto error = cudaLaunchKernel(Function(mode, m % 256 != 0), dim3(1 + (m - 1) / 256, n / 128), dim3(384),
                                  args, SharedBytes(mode), cudaStreamPerThread);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("NVFP4 fresh launch: ") + cudaGetErrorString(error));
    return true;
}
} // namespace fresh_nvfp4
} // namespace fastllm_native_prefill
