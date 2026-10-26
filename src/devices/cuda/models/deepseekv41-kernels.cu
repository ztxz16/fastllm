/*
The Top-K warp-prefix scan below is adapted from Strata (MIT):
Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "devices/cuda/fastllm-v41-full-decode.h"
#include "fastllm-v41-decode.cuh"
//
// DeepSeek-V4.1 专用 CUDA kernel。
//
// 数值语义与 src/devices/cpu/deepseekv41ops.cpp 中的 CPU 参考实现一致；
// 面向 SM86 等无 FP8 tensor core 的设备：稀疏注意力与 indexer 打分在 SM80+ 上用
// BF16 mma（FP32 累加），其余算子为 FP32。SM80 以下或 dtype 不匹配时退回标量实现。
//

#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "devices/cuda/fastllm-cuda-hybrid-dsa.h"
#include "../fastllm-hybrid-prefill.cuh"
#include "deepseekv41-reference.cuh"
#include "fastllm.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <map>
#include <limits>
#include <array>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <type_traits>
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
        // Persistent outputs and borrowed decode rows are already on this
        // rank. Avoid rebuilding metadata and querying capture state per row.
        if (output.cudaData && output.dataDevice == DataDevice::CUDA &&
            output.dataDeviceIds.size() == 1 && output.dataDeviceIds[0] == FastllmCudaGetDevice() &&
            output.dataType == dataType && output.dims == dims &&
            output.Count(0) <= output.expansionSize) return true;
        output.dataType = dataType;
        output.Resize(dims);
        output.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
        output.Allocate(false);
        return output.cudaData != nullptr;
    }

    // Separate slots are required: the indexer invokes the transform path
    // while its own descriptors and scores are still in flight. A worker uses
    // one per-thread CUDA stream, so reuse within each slot is stream ordered.
    struct V41DecodeDeviceScratch {
        void *storage = nullptr;
        size_t capacity = 0;
        int device = -1;
        ~V41DecodeDeviceScratch() {
            if (!storage) return;
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(storage);
            FastllmCudaSetDevice(previous);
        }
    };

    void *V41BorrowDecodeScratch(int slot, size_t bytes) {
        constexpr size_t limit = 128ULL * 1024 * 1024;
        if (!bytes || bytes > limit || FastllmCudaGraphIsCapturingFast()) return nullptr;
        static thread_local std::map<std::pair<int, int>, std::unique_ptr<V41DecodeDeviceScratch>> buffers;
        const int device = FastllmCudaGetDevice();
        auto &buffer = buffers[{device, slot}];
        if (!buffer) {
            buffer = std::make_unique<V41DecodeDeviceScratch>();
            buffer->device = device;
        }
        if (buffer->capacity >= bytes) return buffer->storage;
        size_t capacity = 64 * 1024;
        while (capacity < bytes) capacity *= 2;
        size_t available = 0, total = 0;
        void *replacement = nullptr;
        if (cudaMemGetInfo(&available, &total) != cudaSuccess ||
            available < capacity + 256ULL * 1024 * 1024 ||
            FastllmCudaTryDirectMalloc(&replacement, capacity) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
            return nullptr;
        if (buffer->storage) {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(buffer->storage);
        }
        buffer->storage = replacement;
        buffer->capacity = capacity;
        return replacement;
    }

    // Keep host descriptors alive until the asynchronous upload completes.
    // A bounded ring prevents per-layer pinned allocations and never reuses a
    // slot whose copy is still in flight. Payload lifetime remains with callers.
    struct V41DescriptorUploads {
        struct Slot { void *host = nullptr; size_t capacity = 0; cudaEvent_t copied = nullptr; bool pending = false; };
        std::array<Slot, 16> slots{};
        size_t next = 0;
        int device;
        explicit V41DescriptorUploads(int device) : device(device) {}
        ~V41DescriptorUploads() {
            int previous = 0;
            cudaGetDevice(&previous); cudaSetDevice(device);
            for (auto &slot : slots) {
                if (slot.pending) cudaEventSynchronize(slot.copied);
                if (slot.copied) cudaEventDestroy(slot.copied);
                if (slot.host) cudaFreeHost(slot.host);
            }
            cudaSetDevice(previous);
        }
    };

    bool V41UploadDescriptors(void *dst, const void *src, size_t bytes) {
        if (bytes > 128 * 1024 || FastllmCudaGraphIsCapturingFast())
            return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        static thread_local std::map<int, std::unique_ptr<V41DescriptorUploads>> pools;
        const int device = FastllmCudaGetDevice();
        auto &pool = pools[device];
        if (!pool) pool.reset(new V41DescriptorUploads(device));
        auto &slot = pool->slots[pool->next++ % pool->slots.size()];
        if (slot.pending) {
            const auto ready = cudaEventQuery(slot.copied);
            if (ready == cudaErrorNotReady) {
                if (cudaEventSynchronize(slot.copied) != cudaSuccess) return false;
            } else if (ready != cudaSuccess) return false;
            slot.pending = false;
        }
        if (!slot.copied && cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming) != cudaSuccess)
            return false;
        if (slot.capacity < bytes) {
            void *replacement = nullptr;
            const size_t capacity = (bytes + 4095) & ~size_t(4095);
            if (cudaHostAlloc(&replacement, capacity, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
            }
            if (slot.host) cudaFreeHost(slot.host);
            slot.host = replacement; slot.capacity = capacity;
        }
        std::memcpy(slot.host, src, bytes);
        if (cudaMemcpyAsync(dst, slot.host, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) != cudaSuccess)
            return false;
        if (cudaEventRecord(slot.copied, cudaStreamPerThread) != cudaSuccess) {
            cudaStreamSynchronize(cudaStreamPerThread);
            return false;
        }
        slot.pending = true;
        return true;
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

    template <typename T> __device__ __forceinline__ float V41StorageRound(float v) {
        T rounded;
        V41Store<T>(&rounded, 0, v);
        return V41Load<T>(&rounded, 0);
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
        // Construct the same power of two directly, including the original
        // subnormal-input and overflow behavior, without a general ldexpf.
        return __uint_as_float((bits & 0x7f800000u) +
                               ((bits & 0x007fffffu) ? 0x00800000u : 0u));
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
        const unsigned bits = __float_as_uint(scale);
        // FP8 block scales are powers of two. Their normal reciprocals are
        // exact, so multiplication preserves division's FP32 rounding. Keep
        // division for FP4 and scales whose reciprocal is not normal.
        const float scaled = quantMode == 1 && bits >= 0x00800000u &&
                             bits < 0x7f000000u && !(bits & 0x007fffffu)
            ? v * __uint_as_float(0x7f000000u - bits) : v / scale;
        float q = fminf(qmax, fmaxf(-qmax, scaled));
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

    template <typename T, bool QUANTIZE = false>
    __global__ void V41SharedSwigluKernel(const T *input, __nv_bfloat16 *output,
                                         uint64_t count, int mid, float limit, const float *routeScales) {
        const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const uint64_t offset = (i / mid) * (2 * mid) + i % mid;
        float gate = V41Bf16Round(V41Load(input, offset));
        float up = V41Bf16Round(V41Load(input, offset + mid));
        if (limit > 0.0f) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
        }
        // Keep the BF16 boundary before finding each group-32 FP8 scale.
        float gated = (gate / (1.0f + expf(-gate))) * up;
        if (routeScales != nullptr) gated *= routeScales[i / mid];
        __nv_bfloat16 result = __float2bfloat16_rn(gated);
        if constexpr (QUANTIZE) {
            const float value = __bfloat162float(result);
            float maximum = fabsf(value);
            for (int offset = 16; offset; offset >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
            result = __float2bfloat16_rn(V41QuantValueDev(value, V41QuantScaleDev(maximum, 1), 1));
        }
        output[i] = result;
    }

    // Pair adjacent BF16 values without changing the BF16 -> group-32 FP8
    // rounding boundary. Each half warp owns one complete quantization group.
    template <bool QUANTIZE>
    __global__ void V41SharedSwigluBf16PairKernel(const __nv_bfloat162 *input,
            __nv_bfloat162 *output, int pairs, int midPairs, float limit,
            const float *routeScales) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= pairs) return;
        const int row = i / midPairs;
        const int offset = i + row * midPairs;
        const float2 gates = __bfloat1622float2(input[offset]);
        const float2 ups = __bfloat1622float2(input[offset + midPairs]);
        float gate[2] = {gates.x, gates.y}, up[2] = {ups.x, ups.y};
        __nv_bfloat16 result[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (limit > 0.0f) {
                gate[j] = fminf(gate[j], limit);
                up[j] = fmaxf(-limit, fminf(up[j], limit));
            }
            float value = (gate[j] / (1.0f + expf(-gate[j]))) * up[j];
            if (routeScales != nullptr) value *= routeScales[row];
            result[j] = __float2bfloat16_rn(value);
        }
        if constexpr (QUANTIZE) {
            const float x = __bfloat162float(result[0]), y = __bfloat162float(result[1]);
            float maximum = fmaxf(fabsf(x), fabsf(y));
            const unsigned mask = __activemask();
#pragma unroll
            for (int offset = 8; offset; offset >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset, 16));
            const float scale = V41QuantScaleDev(maximum, 1);
            result[0] = __float2bfloat16_rn(V41QuantValueDev(x, scale, 1));
            result[1] = __float2bfloat16_rn(V41QuantValueDev(y, scale, 1));
        }
        output[i] = __halves2bfloat162(result[0], result[1]);
    }

    template <bool QUANTIZE>
    void V41LaunchBf16Swiglu(const void *input, void *output, uint64_t count,
                           int mid, float limit, const float *routeScales) {
        // Tiny batches benefit more from the scalar kernel's extra active lanes.
        if (count >= 32768 && mid % 2 == 0 && count <= INT_MAX / 2 &&
            (uintptr_t)input % 4 == 0 && (uintptr_t)output % 4 == 0) {
            const int pairs = count / 2;
            V41SharedSwigluBf16PairKernel<QUANTIZE><<<(pairs + 127) / 128, 128>>>(
                (const __nv_bfloat162*)input, (__nv_bfloat162*)output,
                pairs, mid / 2, limit, routeScales);
        } else {
            V41SharedSwigluKernel<__nv_bfloat16, QUANTIZE><<<(count + 255) / 256, 256>>>(
                (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count, mid, limit, routeScales);
        }
    }

    // Ascending reduction tree matches FastllmGemvFp32Fp32Kernel2.
    // BF16 converts exactly to FP32 in registers; no activation cast buffer.
    __global__ void V41RouterScoresKernel(const __nv_bfloat16 *input, const float *weight,
                                          float *scores, int dim, int experts, float invTemp) {
        const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
        const int expert = blockIdx.x, token = blockIdx.y;
        float sum = 0.0f;
#pragma unroll 4
        for (int i = tid; i < dim; i += 256)
            sum += __bfloat162float(input[(size_t)token * dim + i]) * weight[(size_t)expert * dim + i];
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, offset);
            if ((lane & (2 * offset - 1)) == 0) sum += other;
        }
        __shared__ float partial[8];
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 8 ? partial[lane] : 0.0f;
#pragma unroll
            for (int offset = 1; offset < 8; offset *= 2) {
                const float other = __shfl_down_sync(0xffffffff, sum, offset);
                if ((lane & (2 * offset - 1)) == 0) sum += other;
            }
            if (lane == 0) {
                const float value = __fmul_rn(__fadd_rn(sum, 0.0f), invTemp);
                const float softplus = value > 20.0f ? value :
                    (value < -20.0f ? expf(value) : log1pf(expf(value)));
                scores[(size_t)token * experts + expert] = sqrtf(softplus);
            }
        }
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

    template <typename T, int HC, bool DOTS_ONLY = false>
    __global__ void __launch_bounds__(kHcThreads)
    V41HcMixKernelMulti(const T *x, const float *fn, const float *scale, const float *base,
                        int tokens, int dim, int sinkhornIters, float eps, float normEps,
                        float *pre, float *post, float *comb, float *dotOutput = nullptr) {
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
        if constexpr (DOTS_ONLY) {
            // Keep the original accumulation/reduction order, but finish the
            // Sinkhorn matrices in parallel warps rather than four CTA lanes.
            for (int idx = threadIdx.x; idx < kHcTokens * (MIXHC + 1); idx += kHcThreads) {
                const int tb = idx / (MIXHC + 1), m = idx % (MIXHC + 1);
                if (t0 + tb < tokens)
                    dotOutput[(uint64_t)(t0 + tb) * (MIXHC + 1) + m] =
                        m == MIXHC ? sumsq[tb] : mixes[tb][m];
            }
            return;
        }
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

    // Small decode/verify batches need parallelism across dot products as
    // well as tokens: the four-token kernel otherwise uses only one SM.
    // Retain each output's 256-thread strided accumulation and reduction order.
    template <typename T, int TILE = 0>
    __device__ __forceinline__ void V41HcDecodeDotsBody(const T *x, const float *fn, int flatDim, float *mix) {
        const int m = blockIdx.x;
        x += (uint64_t)blockIdx.y * flatDim;
        mix += (uint64_t)blockIdx.y * 25;
        float acc = 0.0f;
        int k = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        // Every launch uses exactly 256 threads. Expose that index range to
        // NVCC without changing any lane's FMA or reduction order.
        k &= kHcThreads - 1;
        if constexpr (TILE > 0) {
            // The SM90 BF16 launcher checks shape eligibility. Preserve each
            // lane's FMA order; partial tiles use the original loop below.
            for (; k + (TILE - 1) * kHcThreads < flatDim; k += TILE * kHcThreads) {
                float values[TILE], weights[TILE];
#pragma unroll
                for (int i = 0; i < TILE; ++i) {
                    values[i] = V41Load(x, k + i * kHcThreads);
                    weights[i] = m == 24 ? values[i] : fn[(uint64_t)m * flatDim + k + i * kHcThreads];
                }
#pragma unroll
                for (int i = 0; i < TILE; ++i) acc += values[i] * weights[i];
            }
        }
#endif
        for (; k < flatDim; k += kHcThreads) {
            const float value = V41Load(x, k);
            acc += value * (m == 24 ? value : fn[(uint64_t)m * flatDim + k]);
        }
        const float sum = V41WarpSum(acc);
        __shared__ float partial[kHcThreads / 32];
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int w = 0; w < kHcThreads / 32; ++w) total += partial[w];
            mix[m] = total;
        }
    }

    template <typename T>
    __global__ void V41HcDecodeDots(const T *x, const float *fn, int flatDim, float *mix) {
        V41HcDecodeDotsBody(x, fn, flatDim, mix);
    }

    // Small SM90 grids benefit from registers for independent loads. Larger
    // batches use a smaller tile to leave room for more resident CTAs.
    template <int TILE>
    __global__ __launch_bounds__(kHcThreads, 1)
    void V41HcDecodeDotsSm90(const __nv_bfloat16 *x, const float *fn, int flatDim, float *mix) {
        V41HcDecodeDotsBody<__nv_bfloat16, TILE>(x, fn, flatDim, mix);
    }

    // Reuse each FP32 coefficient across four tokens while parallelizing
    // the 25 independent dot products. Per-output FMA and reduction order
    // remain the same as the portable HcMix implementation.
    template <typename T>
    __global__ void V41HcBatchDots(const T *x, const float *fn, int flatDim,
                                    int tokens, float *mix) {
        constexpr int tile = 4;
        const int m = blockIdx.x, first = blockIdx.y * tile;
        float acc[tile] = {};
        for (int k = threadIdx.x; k < flatDim; k += kHcThreads) {
            const float weight = m == 24 ? 0.0f : fn[(uint64_t)m * flatDim + k];
#pragma unroll
            for (int t = 0; t < tile; ++t) {
                const float value = first + t < tokens ? V41Load(x, (uint64_t)(first + t) * flatDim + k) : 0.0f;
                acc[t] += value * (m == 24 ? value : weight);
            }
        }
        __shared__ float partial[tile][kHcThreads / 32];
#pragma unroll
        for (int t = 0; t < tile; ++t) {
            const float sum = V41WarpSum(acc[t]);
            if ((threadIdx.x & 31) == 0) partial[t][threadIdx.x / 32] = sum;
        }
        __syncthreads();
        if (threadIdx.x < tile && first + threadIdx.x < tokens) {
            float total = 0.0f;
#pragma unroll
            for (int w = 0; w < kHcThreads / 32; ++w) total += partial[threadIdx.x][w];
            mix[(uint64_t)(first + threadIdx.x) * 25 + m] = total;
        }
    }

    // One lane per matrix entry. Shuffles retain the serial kernel's exact
    // left-to-right sum order, including its initial +0 and division rounding.
    __device__ __forceinline__ void V41HcFinishRow(const float *mix, const float *scale, const float *base,
                                         int flatDim, int sinkhornIters, float eps, float normEps,
                                         float *pre, float *post, float *comb, int t) {
        const int lane = threadIdx.x;
        const int entry = lane & 15, row = entry / 4, col = entry % 4;
        mix += (uint64_t)t * 25;
        const float rsqrtv = rsqrtf(mix[24] / flatDim + normEps);
        if (lane < 4) {
            pre[t * 4 + lane] = V41SigmoidDev(mix[lane] * rsqrtv * scale[0] + base[lane]) + eps;
            post[t * 4 + lane] = 2.0f * V41SigmoidDev(mix[lane + 4] * rsqrtv * scale[1] + base[lane + 4]);
        }
        float value = mix[entry + 8] * rsqrtv * scale[2] + base[entry + 8];
        float maximum = -FLT_MAX;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            maximum = fmaxf(maximum, __shfl_sync(0xffffffff, value, row * 4 + c));
        value = __expf(value - maximum);
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) sum += __shfl_sync(0xffffffff, value, row * 4 + c);
        value = value / sum + eps;
        sum = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) sum += __shfl_sync(0xffffffff, value, r * 4 + col);
        value /= sum + eps;
        for (int it = 1; it < sinkhornIters; ++it) {
            sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) sum += __shfl_sync(0xffffffff, value, row * 4 + c);
            value /= sum + eps;
            sum = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) sum += __shfl_sync(0xffffffff, value, r * 4 + col);
            value /= sum + eps;
        }
        if (lane < 16) comb[t * 16 + entry] = value;
    }

    __global__ void V41HcDecodeFinish(const float *mix, const float *scale, const float *base,
                                      int flatDim, int sinkhornIters, float eps, float normEps,
                                      float *pre, float *post, float *comb) {
        V41HcFinishRow(mix, scale, base, flatDim, sinkhornIters, eps, normEps, pre, post, comb, blockIdx.x);
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

    template <typename T, int HC>
    __global__ void V41HcPostKernel(const T *input, const T *residual, const float *post,
                                    const float *comb, T *output, uint64_t count, int dim) {
        const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count / HC) return;
        const int d = idx % dim;
        const uint64_t t = idx / dim;
        float r[4];
#pragma unroll
        for (int src = 0; src < HC; src++) r[src] = V41Load<T>(residual, (t * HC + src) * dim + d);
        const float x = V41Load<T>(input, idx);
#pragma unroll
        for (int target = 0; target < HC; target++) {
            float sum = 0.0f;
    #pragma unroll
        for (int src = 0; src < HC; src++) {
                float term = __fmul_rn(comb[(t * HC + src) * HC + target], r[src]);
                sum = __fadd_rn(sum, term);
            }
            float scaled = __fmul_rn(post[t * HC + target], x);
            V41Store<T>(output, (t * HC + target) * dim + d, __fadd_rn(scaled, sum));
        }
    }

    template <typename T, int HC>
    __global__ void V41MoeHcPostKernel(const T *input, const __nv_bfloat16 *shared, const __nv_bfloat16 *residual, const float *post,
                                    const float *comb, __nv_bfloat16 *output, uint64_t count, int dim) {
        const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count / HC) return;
        const int d = idx % dim;
        const uint64_t t = idx / dim;
        float r[4];
#pragma unroll
        for (int src = 0; src < HC; src++) r[src] = V41Load<__nv_bfloat16>(residual, (t * HC + src) * dim + d);
        // Match the routed sum + BF16 shared output, then the explicit
        // BF16 rounding before the HC residual transform. Do not fuse this
        // rounding boundary into the following FP32 multiply-adds.
        const float sum = __fadd_rn(V41Load<T>(input, idx), __bfloat162float(shared[idx]));
        const float x = __bfloat162float(__float2bfloat16_rn(sum));
#pragma unroll
        for (int target = 0; target < HC; target++) {
            float sum = 0.0f;
    #pragma unroll
        for (int src = 0; src < HC; src++) {
                float term = __fmul_rn(comb[(t * HC + src) * HC + target], r[src]);
                sum = __fadd_rn(sum, term);
            }
            float scaled = __fmul_rn(post[t * HC + target], x);
            V41Store<__nv_bfloat16>(output, (t * HC + target) * dim + d, __fadd_rn(scaled, sum));
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
            v = __fadd_rn(v, __fmul_rn(pre[(uint64_t)t * hcMult + h], V41Load<T>(x, ((uint64_t)t * hcMult + h) * dim + d)));
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
    template <int THREAD_PER_BLOCK, int MAX_ITER, bool FINISH_MIX = false, bool QUANTIZE_OUTPUT = false>
    __global__ void __launch_bounds__(THREAD_PER_BLOCK)
    V41HcPreNormKernel(const __nv_bfloat16 *x, const float *pre, const float *weight,
                       __nv_bfloat16 *out, int hcMult, int channels, float eps,
                       const float *mix = nullptr, const float *mixScale = nullptr,
                       const float *mixBase = nullptr, int sinkhornIters = 0,
                       float mixEps = 0, float *post = nullptr, float *comb = nullptr, float *nextPre = nullptr,
                       __nv_bfloat16 *quantized = nullptr) {
        // The model consumes the PREVIOUS sublayer's pre coefficients.
        // Finish and normalization are independent, so separate CTAs overlap
        // them in one launch. Keep both original reduction orders.
        if constexpr (FINISH_MIX) {
            if ((blockIdx.x & 1) == 0) {
                if (threadIdx.x < 32)
                    V41HcFinishRow(mix, mixScale, mixBase, hcMult * channels,
                        sinkhornIters, mixEps, eps, nextPre, post, comb, blockIdx.x / 2);
                return;
            }
        }
        constexpr int WARP_SIZE = 32;
        constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
        __shared__ float warp_sums[NUM_WARPS];

        const int o = FINISH_MIX ? blockIdx.x / 2 : blockIdx.x;
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
                    lo = __fadd_rn(lo, __fmul_rn(p, __bfloat162float(v.x)));
                    hi = __fadd_rn(hi, __fmul_rn(p, __bfloat162float(v.y)));
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
        // Each warp repeats the same small reduction, then broadcasts lane 0.
        // This preserves the reduction tree without a second CTA barrier.
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        const float s = rsqrtf(__shfl_sync(0xffffffff, val, 0) / channels + eps);
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
                if constexpr (QUANTIZE_OUTPUT) {
                    // One BF16 pair is one lane here; width-16 covers the
                    // same contiguous block-32 group as the standalone
                    // activation quantizer.
                    float maximum = fmaxf(fabsf(__bfloat162float(ov.x)), fabsf(__bfloat162float(ov.y)));
                    // d is a multiple of 32; a tail may contain only one
                    // complete 16-lane group of the warp.
                    const unsigned active = __activemask();
                    for (int offset = 8; offset; offset >>= 1)
                        maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, offset, 16));
                    const float qs = V41QuantScaleDev(maximum, 1);
                    __nv_bfloat162 qv;
                    qv.x = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(ov.x), qs, 1));
                    qv.y = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(ov.y), qs, 1));
                    reinterpret_cast<__nv_bfloat162 *>(quantized)[(uint64_t)o * bf2 + i] = qv;
                }
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
        // apply_rotary_emb stores back to the input dtype before act_quant.
        float v = V41StorageRound<T>(row[d]);
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
                                        uint64_t totalPairs, const int *devicePosition) {
        const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= totalPairs) {
            return;
        }
        const int pairs = ropeDim >> 1;
        const int r = (int)(idx / (uint64_t)pairs);
        const int d = (int)(idx - (uint64_t)r * pairs);
        const int token = r / rowsPerToken;
        T *base = x + (uint64_t)r * dim + (dim - ropeDim) + 2 * d;
        const float pos = (float)((devicePosition ? *devicePosition : startPos) + (long long)token * posStep);
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
                                         int rowsPerBlock, int rows, const int *devicePosition) {
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
            const float pos = (float)((devicePosition ? *devicePosition : startPos) + (long long)token * posStep);
            const float ang = pos * rope.invFreq[k >> 1];
            float c = cosf(ang), s = sinf(ang);
            if (inverse) {
                s = -s;
            }
            v = (k & 1) ? (other * s + v * c) : (v * c - other * s);
        }
        v = V41StorageRound<T>(v);
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

    struct V41DecodeTransformDescriptor {
        __nv_bfloat16 *input, *window;
        int rows, dim, position, mode, inverse, quantBlock;
    };

    __device__ __forceinline__ void V41DecodeTransformRow(
            const V41DecodeTransformDescriptor &item, int row, int d,
            V41RopeTable rope, int ropeDim, int windowSize) {
        auto *base = item.input + (size_t)row * item.dim;
        if (item.window) {
            item.window[(size_t)(item.position % windowSize) * 512 + d] = base[d];
            return;
        }
        float v = __bfloat162float(base[d]);
        const float other = __shfl_xor_sync(0xffffffff, v, 1);
        const int off = item.dim - ropeDim;
        if (d >= off) {
            const int k = d - off;
            const float angle = (float)item.position * rope.invFreq[k >> 1];
            float c = cosf(angle), s = sinf(angle);
            if (item.inverse) s = -s;
            // The RoPE-only reference evaluates the odd component with
            // sine fused into the first product. Keep that FMA ordering:
            // swapping the fused product can cross a BF16 rounding boundary.
            if (item.mode == 0 && (k & 1))
                v = __fmaf_rn(other, s, __fmul_rn(v, c));
            else
                v = (k & 1) ? (other * s + v * c) : (v * c - other * s);
        }
        v = V41Bf16Round(v);
        if (item.mode) {
            float maximum = fabsf(v);
            if (item.quantBlock == 16) {
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 8));
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 4));
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
            } else maximum = V41WarpMax(maximum);
            const float scale = V41QuantScaleDev(maximum, item.mode);
            v = V41QuantValueDev(v, scale, item.mode);
        }
        base[d] = __float2bfloat16_rn(v);
    }

    template<int STRIDE = 1, int ITEMS = 1>
    __global__ void V41DecodeTransformsKernel(const V41DecodeTransformDescriptor *items,
                                               V41RopeTable rope, int ropeDim, int windowSize) {
        const auto item = items[(blockIdx.y / ITEMS) * STRIDE + blockIdx.y % ITEMS];
        const int rowsPerBlock = 512 / item.dim;
        const int row = blockIdx.x * rowsPerBlock + threadIdx.x / item.dim;
        if (row >= item.rows) return;
        V41DecodeTransformRow(item, row, threadIdx.x % item.dim, rope, ropeDim, windowSize);
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


    // Eight candidates share decoded keys; heads run in parallel. Each dot
    // retains the scalar kernel's 128 FP32 FMAs and ordered head reduction.
    template<typename QT, typename KT>
    __global__ void __launch_bounds__(256)
    V41IndexerScoreSmallBatch(const QT *q, const float *weights, const KT *k,
            int seqlen, int m, float *out, const uint8_t *packed,
            int mode, int rowBytes) {
        __shared__ float queries[128][33], keys[128][9], dots[32][9];
        const int t=blockIdx.y, batch=t/seqlen;
        const int first=blockIdx.x*8, h=threadIdx.x/8, c=threadIdx.x%8;
        for (int i=threadIdx.x; i<32*128; i+=256)
            queries[i%128][i/128]=V41Load<QT>(q, size_t(t)*32*128+i);
        for (int i=threadIdx.x; i<8*128; i+=256) {
            const int j=first+i/128, d=i%128;
            keys[d][i/128]=j<m ? (packed
                ? V41LoadKvRow(packed+(size_t(batch)*m+j)*rowBytes,128,d,mode)
                : V41Load<KT>(k,(size_t(batch)*m+j)*128+d)) : 0.f;
        }
        __syncthreads();
        float dot=0.f;
        #pragma unroll 4
        for (int d=0; d<128; ++d) dot += queries[d][h]*keys[d][c];
        dots[h][c]=dot;
        __syncthreads();
        if (threadIdx.x<8 && first+c<m) {
            float total=0.f;
            #pragma unroll
            for (int head=0; head<32; ++head)
                total += fmaxf(dots[head][c],0.f)*weights[size_t(t)*32+head];
            out[size_t(t)*m+first+c]=total;
        }
    }

    #include "deepseekv41-prefill-indexer.cuh"

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

    struct V41DecodeIndexDescriptor {
        const __nv_bfloat16 *query, *keys;
        const float *weights;
        float *scores, *blockScores;
        uint8_t *candidates;
        int32_t *output;
        int heads, count, width, position, blocks;
        bool writeCandidates, useCandidates;
    };

    // Decode uses heads as the MMA rows instead of padding one token to the
    // prefill kernel's 64 rows. Keep the dot-product and ordered head-sum
    // operations unchanged; the prefill kernel remains the general fallback.
    template <bool BATCHED = false, bool PERSISTENT = false>
    __global__ void __launch_bounds__(256)
    V41IndexerScoreDecodeMmaKernel(const __nv_bfloat16 *q, const float *weights,
                                   const __nv_bfloat16 *k, int heads, int m,
                                   int ratio, int startPos, float *out,
                                   const uint8_t *k8, int kMode, int kRowBytes, int vecUnpack,
                                   const V41DecodeIndexDescriptor *items = nullptr) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        __shared__ __nv_bfloat16 ks[64][136];
        __shared__ __nv_bfloat16 qs[16][136];
        __shared__ float scores[16][64];
        int b = blockIdx.y;
        if (BATCHED) {
            const auto item = items[b];
            q = item.query; k = item.keys; weights = item.weights; out = item.scores;
            heads = item.heads; m = item.count; startPos = item.position; b = 0;
        }
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int visible = ratio > 0 ? min(m, (startPos + 1) / ratio) : m;
        const int end = PERSISTENT ? visible : (int)(blockIdx.x + 1) * 64;
        for (int j0 = blockIdx.x * 64; j0 < end; j0 += gridDim.x * 64) {
        if (j0 >= visible) {
            if (tid < 64 && j0 + tid < m) out[(uint64_t)b * m + j0 + tid] = -INFINITY;
            return;
        }
        for (int v = tid; v < 64 * 16; v += 256) {
            const int j = v / 16, d = (v % 16) * 8;
            if (j0 + j >= m) {
                *((float4*)&ks[j][d]) = make_float4(0, 0, 0, 0);
            } else if (k8 != nullptr) {
                const uint8_t *row = k8 + ((uint64_t)b * m + j0 + j) * kRowBytes;
                if (vecUnpack) V41LoadKvRow8Bf16(row, 128, d, kMode, &ks[j][d]);
                else {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        ks[j][d + e] = __float2bfloat16_rn(V41LoadKvRow(row, 128, d + e, kMode));
                }
            } else {
                *((float4*)&ks[j][d]) = *((const float4*)(k + ((uint64_t)b * m + j0 + j) * 128 + d));
            }
        }
        float total = 0.0f;
        for (int h0 = 0; h0 < heads; h0 += 16) {
            __syncthreads();
            const int h = tid / 16, d = (tid % 16) * 8;
            *((float4*)&qs[h][d]) = h0 + h < heads
                ? *((const float4*)(q + ((uint64_t)b * heads + h0 + h) * 128 + d))
                : make_float4(0, 0, 0, 0);
            __syncthreads();
            float acc[4] = {0, 0, 0, 0};
            const int ar = ((lane >> 3) & 1) * 8 + (lane & 7);
            const int ac = (lane >> 4) * 8;
            const int br = warp * 8 + (lane & 7);
            const int bc = ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                uint32_t a[4], bb[2];
                V41LdmX4(a, &qs[ar][kk + ac]);
                V41LdmX2(bb, &ks[br][kk + bc]);
                V41MmaBf16(acc, a, bb);
            }
            const int row = lane >> 2, col = warp * 8 + (lane & 3) * 2;
            scores[row][col] = acc[0]; scores[row][col + 1] = acc[1];
            scores[row + 8][col] = acc[2]; scores[row + 8][col + 1] = acc[3];
            __syncthreads();
            if (tid < 64) {
                for (int h = 0; h < min(16, heads - h0); ++h)
                    total += fmaxf(scores[h][tid], 0.0f) * weights[(uint64_t)b * heads + h0 + h];
            }
        }
        if (tid < 64 && j0 + tid < m) out[(uint64_t)b * m + j0 + tid] = total;
        if constexpr (PERSISTENT) __syncthreads();
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

    template <bool BATCHED = false>
    __global__ void V41BlockScoreKernel(const float *score, int seqlen, int m, int blockSize, int numBlocks,
                                        int ratio, int startPos, float *blockScore,
                                        const V41DecodeIndexDescriptor *items = nullptr) {
        int t = blockIdx.x;
        if (BATCHED) {
            const auto item = items[t];
            if (!item.writeCandidates) return;
            score = item.scores; blockScore = item.blockScores;
            m = item.count; numBlocks = item.blocks; startPos = item.position;
            seqlen = 1; t = 0;
        }
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

    template <bool BATCHED = false>
    __global__ void V41CandidateSelectKernel(const float *blockScore, int numBlocks, int topkBlocks, uint8_t *mask,
                                               const V41DecodeIndexDescriptor *items = nullptr) {
        int t = blockIdx.x;
        if (BATCHED) {
            const auto item = items[t];
            if (!item.writeCandidates) return;
            blockScore = item.blockScores; numBlocks = item.blocks; mask = item.candidates; t = 0;
        }
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
    //   4. 写出阶段：prefill 使用打包 BlockScan，decode 使用 warp ballot / 前缀和，
    //      输出位置 = 前面 greater 的个数 + min(tieBudget, 前面 tie 的个数)。
    //
    // 共享内存里的候选块下标由 candList（动态共享内存）承载，listCap == 0 表示不压缩。
    constexpr int kTopKSurvivorCap = 1024;

    // Parallel radix-bin selection preserves exact integer counts and ties.
    // The prior path scanned up to 256 shared bins on a single thread/pass.
    template <bool BATCHED = false, bool PARALLEL_SELECT = true, int SELECT_THREADS = (PARALLEL_SELECT ? 1024 : 256)>
    __global__ void __launch_bounds__(SELECT_THREADS) V41TopKKernel(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int blockShift, int width,
                                  int ratio, int startPos, int listCap, int32_t *out,
                                  const V41DecodeIndexDescriptor *items = nullptr) {
        typedef cub::BlockScan<int, SELECT_THREADS> BlockScan;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ unsigned hist[256];
        __shared__ unsigned survivors[kTopKSurvivorCap];
        __shared__ int meta[4];             // 0: digit, 1: remaining, 2: 候选块个数
        extern __shared__ int candList[];

        int t = blockIdx.x;
        if (BATCHED) {
            const auto item = items[t];
            score = item.scores; candidates = item.useCandidates ? item.candidates : nullptr;
            m = item.count; numBlocks = item.blocks; width = item.width;
            startPos = item.position; out = item.output; seqlen = 1; t = 0;
            listCap = candidates ? min(numBlocks, listCap) : 0;
        }
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
            for (int st = 0; st < numBlocks; st += SELECT_THREADS) {
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
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < nDomain; p += SELECT_THREADS) {
            const int j = V41_TOPK_INDEX(p);
            if (V41_TOPK_OK(j)) {
                atomicAdd(&hist[V41FloatKey(row[j]) >> 24], 1u);
            }
        }
        __syncthreads();
        int binCount = threadIdx.x < 256 ? (int)hist[threadIdx.x] : 0, binPrefix = 0, total = 0;
        BlockScan(scanStorage).ExclusiveSum(binCount, binPrefix, total);
        __syncthreads();

        const int keep = min(width, total);
        unsigned threshold = 0;
        int tieBudget = 0;
        const bool selectAll = total <= width;
        if (!selectAll) {
            if constexpr (PARALLEL_SELECT) {
                const int greater = total - binPrefix - binCount;
                if (binCount > 0 && width > greater && width <= greater + binCount) {
                    meta[0] = threadIdx.x;
                    meta[1] = width - greater;
                    meta[2] = binCount;
                }
            } else {
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
                for (int p = threadIdx.x; p < nDomain; p += SELECT_THREADS) {
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
                if (threadIdx.x < 256) hist[threadIdx.x] = 0;
                __syncthreads();
                const unsigned mask = 0xffffffffu << (shift + 8);
                if (useShared) {
                    for (int s = threadIdx.x; s < survivorCount; s += SELECT_THREADS) {
                        const unsigned key = survivors[s];
                        if ((key & mask) == prefix) {
                            atomicAdd(&hist[(key >> shift) & 255], 1u);
                        }
                    }
                } else {
                    for (int p = threadIdx.x; p < nDomain; p += SELECT_THREADS) {
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
                if constexpr (PARALLEL_SELECT) {
                    const int count = threadIdx.x < 256 ? (int)hist[threadIdx.x] : 0;
                    int prefixCount = 0, totalCount = 0;
                    BlockScan(scanStorage).ExclusiveSum(count, prefixCount, totalCount);
                    __syncthreads();
                    const int greater = totalCount - prefixCount - count;
                    if (count > 0 && remaining > greater && remaining <= greater + count) {
                        meta[0] = threadIdx.x;
                        meta[1] = remaining - greater;
                    }
                } else {
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

        // Warp-prefix emission adapted from Strata src/dsv41/ds4_kernels.cu
        // (c33fa7ba6d0d), MIT, Copyright (c) 2026 Niko1221 and contributors.
        // Unlike its greater-then-ties layout, merge both ranks here so that
        // FastLLM's globally ascending selected-index / attention order stays exact.
        if constexpr (PARALLEL_SELECT) {
            __shared__ int warpGreater[32], warpEqual[32];
            __shared__ int cumulativeGreater, cumulativeTie;
            if (threadIdx.x == 0) { cumulativeGreater = 0; cumulativeTie = 0; }
            __syncthreads();
            const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
            constexpr int warps = SELECT_THREADS / 32;
            const unsigned below = (1u << lane) - 1u;
            for (int st = 0; st < nDomain; st += SELECT_THREADS) {
                const int p = st + threadIdx.x;
                const int j = p < nDomain ? V41_TOPK_INDEX(p) : -1;
                const bool ok = j >= 0 && V41_TOPK_OK(j);
                const unsigned key = ok ? V41FloatKey(row[j]) : negInf;
                const bool greater = ok && key > negInf && (selectAll || key > threshold);
                const bool equal = ok && key > negInf && !selectAll && key == threshold;
                const unsigned bg = __ballot_sync(0xffffffffu, greater);
                const unsigned be = __ballot_sync(0xffffffffu, equal);
                if (lane == 0) { warpGreater[warp] = __popc(bg); warpEqual[warp] = __popc(be); }
                __syncthreads();
                if (warp == 0) {
                    const int g = lane < warps ? warpGreater[lane] : 0;
                    const int e = lane < warps ? warpEqual[lane] : 0;
                    int gs = g, es = e;
#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        const int a = __shfl_up_sync(0xffffffffu, gs, offset);
                        const int b = __shfl_up_sync(0xffffffffu, es, offset);
                        if (lane >= offset) { gs += a; es += b; }
                    }
                    if (lane < warps) {
                        warpGreater[lane] = cumulativeGreater + gs - g;
                        warpEqual[lane] = cumulativeTie + es - e;
                    }
                    __syncwarp();
                    if (lane == 31) { cumulativeGreater += gs; cumulativeTie += es; }
                }
                __syncthreads();
                const int gBefore = warpGreater[warp] + __popc(bg & below);
                const int tBefore = warpEqual[warp] + __popc(be & below);
                if (greater || (equal && tBefore < tieBudget)) {
                    const int slot = gBefore + min(tieBudget, tBefore);
                    if (slot < width) orow[slot] = j;
                }
                __syncthreads();
            }
        } else {
            // ---- 最后一趟：按升序写出 ----
            int runningGreater = 0, runningTie = 0;
            for (int st = 0; st < nDomain; st += SELECT_THREADS) {
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
        }
        for (int p = keep + threadIdx.x; p < width; p += SELECT_THREADS) {
            orow[p] = -1;
        }
#undef V41_TOPK_INDEX
#undef V41_TOPK_OK
    }

    // ---------------- SparseAttention ----------------

    constexpr int kAttnHeadsPerBlock = 32;
    constexpr int kAttnLanes = 8;
    constexpr int kAttnThreads = kAttnHeadsPerBlock * kAttnLanes;   // 256


    template <typename QT, int HeadsPerBlock = kAttnHeadsPerBlock, bool Padded = false,
              int ValueSplits = 1, bool LowRegisters = false>
    __global__ void __launch_bounds__(HeadsPerBlock * kAttnLanes, LowRegisters ? 2 : 1)
    V41SparseAttentionKernel(const QT *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                             const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                             const int32_t *cmpIdx,
                             const float *sink, int seqlen, int heads, int dim,
                             int windowSize, int cap, int topWidth, int startPos,
                             float scale, __nv_bfloat16 *out,
                             int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes) {
        static_assert(HeadsPerBlock % 4 == 0, "8-lane heads must fill complete warps");
        static_assert(ValueSplits > 0 && 64 % ValueSplits == 0, "value slices must partition each lane segment");
        static_assert(!LowRegisters || (std::is_same_v<QT, __nv_bfloat16> && Padded && ValueSplits == 1),
                      "packed queries require the padded BF16 multi-token path");
        constexpr int threads = HeadsPerBlock * kAttnLanes;
        // A 65-float stride gives the eight head lanes distinct shared-memory banks.
        __shared__ float kvRow[512 + (Padded ? 8 : 0)];
        const int t = blockIdx.x;                  // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        constexpr int lanesPerHead = kAttnLanes;          // 8
        constexpr int segment = 512 / lanesPerHead;       // 64
        const int h = blockIdx.y * HeadsPerBlock + threadIdx.x / lanesPerHead;
        const int lane = threadIdx.x % lanesPerHead;
        const QT *qrow = q + ((uint64_t)t * heads + h) * dim + lane * segment;
        float qv[LowRegisters ? 1 : 64];
        uint32_t packedQ[LowRegisters ? 32 : 1];
        if constexpr (LowRegisters) {
            // Keep two original BF16 values per register. Every 64-value
            // lane segment is aligned, including partial head blocks.
            for (int d = 0; d < 32; d++) {
                packedQ[d] = h < heads ? reinterpret_cast<const uint32_t *>(qrow)[d] : 0;
            }
        } else {
            for (int d = 0; d < segment; d++) {
                qv[d] = h < heads ? V41Load<QT>(qrow, d) : 0.0f;
            }
        }
        constexpr int valueSegment = segment / ValueSplits;
        const int valueOffset = ValueSplits == 1 ? 0 : blockIdx.z * valueSegment;
        float acc[valueSegment];
        for (int d = 0; d < valueSegment; d++) {
            acc[d] = 0.0f;
        }
        float mx = h < heads ? sink[h] : 0.0f;
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
                for (int d = threadIdx.x; d < dim; d += threads) {
                    kvRow[d + (Padded ? d / 64 : 0)] = V41LoadKvRow(src8, dim, d, src8Mode);
                }
            } else {
                for (int d = threadIdx.x; d < dim; d += threads) {
                    kvRow[d + (Padded ? d / 64 : 0)] = __bfloat162float(src[d]);
                }
            }
            __syncthreads();
            float dot = 0.0f;
            // Reload shared KV for PV instead of retaining 64 values across
            // QK. Together with packed Q this permits two SM75 CTAs per SM.
            const std::conditional_t<LowRegisters, volatile float, float> *seg =
                kvRow + lane * (segment + (Padded ? 1 : 0));
            if constexpr (LowRegisters) {
                #pragma unroll
                for (int d = 0; d < segment; d++) {
                    uint32_t bits;
                    // Volatile PTX keeps expansion at the use site; hoisting
                    // all 64 FP32 queries would defeat the register saving.
                    if (d & 1) asm volatile("and.b32 %0, %1, 0xffff0000;" : "=r"(bits) : "r"(packedQ[d / 2]));
                    else asm volatile("shl.b32 %0, %1, 16;" : "=r"(bits) : "r"(packedQ[d / 2]));
                    dot += __uint_as_float(bits) * seg[d];
                }
            } else {
                for (int d = 0; d < segment; d++) {
                    dot += qv[d] * seg[d];
                }
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
            for (int d = 0; d < valueSegment; d++) {
                acc[d] = acc[d] * alpha + p * seg[valueOffset + d];
            }
            mx = newMx;
        }
        float inv = 1.0f / l;
        __nv_bfloat16 *orow = out + ((uint64_t)t * heads + h) * dim + lane * segment + valueOffset;
        if (h < heads) {
            for (int d = 0; d < valueSegment; d++) {
                orow[d] = __float2bfloat16_rn(acc[d] * inv);
            }
        }
    }


    #include "deepseekv41-sm75-quant-attention.cuh"

    template<bool Windowed>
    __device__ __forceinline__ const __nv_bfloat16 *V41Sm75Candidate(
            const __nv_bfloat16 *kv, const int32_t *indices, int c, int cap,
            const __nv_bfloat16 *ring, const __nv_bfloat16 *chunk, int window, int start) {
        if constexpr (Windowed) {
            const int count = min(window, start + 1);
            if (c < count) {
                const int position = start - count + 1 + c;
                return position == start ? chunk : ring + size_t(position % window) * 512;
            }
            c -= count;
        }
        const int idx = indices[c];
        return idx >= 0 && idx < cap ? kv + size_t(idx) * 512 : nullptr;
    }

    // SM75 compressed BF16 decode: materialize independent QK dots, then
    // retain the original candidate order and FP32 online-softmax recurrence.
    // Splitting only output dimensions requires no cross-CTA reduction.
    template<bool Windowed = false>
    __global__ void __launch_bounds__(128)
    V41Sm75DecodeScores(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
                       const int32_t *indices, int count, int cap, int heads, float *scores,
                       const __nv_bfloat16 *ring = nullptr, const __nv_bfloat16 *chunk = nullptr,
                       int window = 0, int start = 0) {
        const int token = blockIdx.z;
        q += (size_t)token * heads * 512;
        if constexpr (!Windowed) indices += (size_t)token * count;
        scores += (size_t)token * heads * count;
        __shared__ float keys[4][520];
        __shared__ float queries[Windowed ? 4 : 1][520];
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const int c = blockIdx.x * 4 + warp;
        const int h = blockIdx.y * 4 + lane / 8;
        const auto *row = c < count ? V41Sm75Candidate<Windowed>(
            kv, indices, c, cap, ring, chunk, window, start) : nullptr;
        const bool valid = row != nullptr;
        if (valid) {
            for (int d = lane; d < 512; d += 32)
                keys[warp][d + d / 64] = __bfloat162float(row[d]);
        }
        if constexpr (Windowed) {
            // The dot lanes walk 64-value segments. Stage Q coalesced once
            // per four heads instead of issuing strided global loads for
            // every candidate and every scalar FMA.
            for (int i = threadIdx.x; i < 4 * 512; i += 128) {
                const int head = i / 512, d = i % 512;
                queries[head][d + d / 64] = __bfloat162float(q[(blockIdx.y * 4 + head) * 512 + d]);
            }
            __syncthreads();
        } else __syncwarp();
        if (!valid) return;
        const int segment = (lane % 8) * 64;
        float dot = 0.f;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            const float query = Windowed ? queries[lane / 8][segment + segment / 64 + d] :
                __bfloat162float(q[h * 512 + segment + d]);
            dot += query * keys[warp][segment + segment / 64 + d];
        }
        dot += __shfl_xor_sync(0xffffffff, dot, 1);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        if (lane % 8 == 0) scores[h * count + c] = dot;
    }

    template<bool Windowed = false>
    __global__ void __launch_bounds__(32)
    V41Sm75DecodeValues(const __nv_bfloat16 *kv, const int32_t *indices,
                       const float *scores, const float *sink, int count, int cap,
                       int heads, float scale, __nv_bfloat16 *out,
                       const __nv_bfloat16 *ring = nullptr, const __nv_bfloat16 *chunk = nullptr,
                       int window = 0, int start = 0) {
        const int token = blockIdx.z;
        if constexpr (!Windowed) indices += (size_t)token * count;
        scores += (size_t)token * heads * count;
        out += (size_t)token * heads * 512;
        constexpr int width = 32, Tile = 4, Values = Windowed ? 4 : 1;
        const int h = blockIdx.x, first = blockIdx.y * width * Values + threadIdx.x;
        float acc[Values] = {};
        float mx = sink[h], l = 1.f;
        // Prefetch four candidates, then consume them in the original order.
        for (int base = 0; base < count; base += Tile) {
            float sc[Tile], vv[Tile][Values];
            bool valid[Tile];
            #pragma unroll
            for (int j = 0; j < Tile; ++j) {
                const int c = base + j;
                const auto *row = c < count ? V41Sm75Candidate<Windowed>(
                    kv, indices, c, cap, ring, chunk, window, start) : nullptr;
                valid[j] = row != nullptr;
                if (valid[j]) {
                    sc[j] = scores[h * count + c];
                    #pragma unroll
                    for (int v = 0; v < Values; ++v)
                        vv[j][v] = __bfloat162float(row[first + v * width]);
                }
            }
            #pragma unroll
            for (int j = 0; j < Tile; ++j) if (valid[j]) {
                const float s = sc[j] * scale;
                const float next = fmaxf(mx, s), alpha = __expf(mx - next), p = __expf(s - next);
                l = l * alpha + p;
                #pragma unroll
                for (int v = 0; v < Values; ++v) acc[v] = acc[v] * alpha + p * vv[j][v];
                mx = next;
            }
        }
        const float inv = 1.f / l;
        #pragma unroll
        for (int v = 0; v < Values; ++v)
            out[h * 512 + first + v * width] = __float2bfloat16_rn(acc[v] * inv);
    }

    // One thread owns one candidate/head dot. Stage keys transposed so all
    // 128 FMAs retain the scalar order while each warp reads contiguous keys.
    template<bool dynamic = false>
    __global__ void Glm5HybridIndexerDots(const __nv_bfloat16 *query,
            const __nv_bfloat16 *keys, int candidates, float *dots, const int32_t *position = nullptr) {
        const int validCandidates = dynamic ? (*position + 1) / 4 : candidates;
        if (blockIdx.x * 32 >= validCandidates) return;
        const int token = blockIdx.z, head = blockIdx.y * 8 + threadIdx.x / 32;
        const int candidate = blockIdx.x * 32 + threadIdx.x % 32;
        __shared__ float keyTile[128][33], queryTile[8][128];
        for (int i = threadIdx.x; i < 32 * 128; i += 256) {
            const int row = i / 128, d = i % 128, source = blockIdx.x * 32 + row;
            keyTile[d][row] = source < validCandidates ? __bfloat162float(keys[size_t(source) * 128 + d]) : 0.f;
        }
        for (int i = threadIdx.x; i < 8 * 128; i += 256)
            queryTile[i / 128][i % 128] = __bfloat162float(
                query[(size_t(token) * 32 + blockIdx.y * 8) * 128 + i]);
        __syncthreads();
        if (candidate >= validCandidates) return;
        float sum = 0.f;
        #pragma unroll
        for (int d = 0; d < 128; ++d)
            sum += queryTile[threadIdx.x / 32][d] * keyTile[d][threadIdx.x % 32];
        dots[(size_t(token) * 32 + head) * candidates + candidate] = sum;
    }

    template<bool dynamic = false>
    __global__ void Glm5HybridIndexerReduce(const float *dots, const float *weights,
            int candidates, int heads, float *output, const int32_t *position = nullptr) {
        const int candidate = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
        if (candidate >= candidates) return;
        if (dynamic && candidate >= (*position + 1) / 4) {
            output[size_t(row) * candidates + candidate] = -INFINITY;
            return;
        }
        float sum = 0;
        for (int h = 0; h < heads; ++h)
            sum += fmaxf(dots[(size_t(row) * heads + h) * candidates + candidate], 0.f) * weights[row * heads + h];
        output[size_t(row) * candidates + candidate] = sum;
    }

    // Keep each query segment in registers across eight candidate groups.
    // Coalesced shared loads preserve the original 64-FMA/eight-lane dot order.
    __global__ void __launch_bounds__(256)
    Glm5HybridAttentionScores(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const int32_t *indices, int count, int cap, int heads, float *scores) {
        const int token = blockIdx.z, group = threadIdx.x / 64;
        const int local = threadIdx.x % 64, lane = local % 8;
        const int h = blockIdx.y * 8 + local / 8;
        q += size_t(token) * heads * 512;
        indices += size_t(token) * count;
        scores += size_t(token) * heads * count;
        __shared__ float keys[4][520], queries[8][520];
        for (int i = threadIdx.x; i < 8 * 512; i += 256) {
            const int d = i % 512;
            queries[i / 512][d + d / 64] = __bfloat162float(q[(blockIdx.y * 8 + i / 512) * 512 + d]);
        }
        __syncthreads();
        float query[64];
        #pragma unroll
        for (int d = 0; d < 64; ++d) query[d] = queries[local / 8][lane * 65 + d];
        for (int tile = 0; tile < 8; ++tile) {
            const int c = blockIdx.x * 32 + tile * 4 + group;
            const int id = c < count ? indices[c] : -1;
            const bool valid = id >= 0 && id < cap;
            if (valid) for (int d = local; d < 512; d += 64)
                keys[group][d + d / 64] = __bfloat162float(kv[size_t(id) * 512 + d]);
            __syncthreads();
            if (valid) {
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < 64; ++d) dot += query[d] * keys[group][lane * 65 + d];
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);
                dot += __shfl_xor_sync(0xffffffff, dot, 4);
                if (lane == 0) scores[h * count + c] = dot;
            }
            __syncthreads();
        }
    }

    // Prefix maxima are associative. Evaluate the same expf(alpha/p) as the
    // online softmax in parallel, but accumulate its denominator sequentially.
    __global__ void Glm5HybridAttentionWeights(const float *scores, const int32_t *indices,
            int count, int cap, int heads, float scale, float2 *weights, float *inverse) {
        const int h = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
        const int lane = tid % 32, warp = tid / 32;
        scores += (size_t(token) * heads + h) * count;
        weights += (size_t(token) * heads + h) * count;
        indices += size_t(token) * count;
        __shared__ float maximum, denominator, warpMax[8], warpBefore[8];
        __shared__ float2 coefficients[256];
        if (tid == 0) { maximum = -INFINITY; denominator = 1.f; }
        __syncthreads();
        for (int base = 0; base < count; base += 256) {
            const int c = base + tid, id = c < count ? indices[c] : -1;
            const bool valid = id >= 0 && id < cap;
            const float score = valid ? scores[c] * scale : -INFINITY;
            float prefix = score;
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                const float before = __shfl_up_sync(0xffffffff, prefix, offset);
                if (lane >= offset) prefix = fmaxf(prefix, before);
            }
            if (lane == 31) warpMax[warp] = prefix;
            __syncthreads();
            if (tid < 8) {
                float value = maximum;
                for (int w = 0; w < tid; ++w) value = fmaxf(value, warpMax[w]);
                warpBefore[tid] = value;
            }
            __syncthreads();
            const float previousLane = __shfl_up_sync(0xffffffff, prefix, 1);
            const float before = fmaxf(warpBefore[warp], lane ? previousLane : -INFINITY);
            const float next = fmaxf(before, score);
            const float2 pair = valid ? make_float2(__expf(before - next), __expf(score - next)) : make_float2(1.f, 0.f);
            coefficients[tid] = pair;
            if (c < count) weights[c] = pair;
            __syncthreads();
            if (tid == 0) {
                float sum = denominator;
                for (int j = 0; j < min(256, count - base); ++j)
                    sum = sum * coefficients[j].x + coefficients[j].y;
                denominator = sum;
                maximum = fmaxf(warpBefore[7], warpMax[7]);
            }
            __syncthreads();
        }
        if (tid == 0) inverse[size_t(token) * heads + h] = 1.f / denominator;
    }

    template<int Heads, int Tile, int Dimensions = 1, int Threads = 128>
    __global__ void Glm5HybridAttentionValues(const __nv_bfloat16 *kv,
            const int32_t *indices, const float2 *weights, const float *inverse,
            int count, int cap, int heads, __nv_bfloat16 *output) {
        const int token = blockIdx.z, h = blockIdx.x * Heads;
        const int d = (blockIdx.y * Threads + threadIdx.x) * Dimensions;
        indices += size_t(token) * count;
        weights += (size_t(token) * heads + h) * count;
        float acc[Heads][Dimensions] = {};
        for (int base = 0; base < count; base += Tile) {
            float2 pair[Heads][Tile]; float value[Dimensions][Tile]; bool valid[Tile];
            #pragma unroll
            for (int j = 0; j < Tile; ++j) {
                const int c = base + j, id = c < count ? indices[c] : -1;
                valid[j] = id >= 0 && id < cap;
                if (valid[j]) {
                    if constexpr (Dimensions == 2) {
                        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                            kv + size_t(id) * 512 + d));
                        value[0][j] = v.x; value[1][j] = v.y;
                    } else {
                        static_assert(Dimensions == 1, "Hybrid values use one or two channels per thread");
                        value[0][j] = __bfloat162float(kv[size_t(id) * 512 + d]);
                    }
                    #pragma unroll
                    for (int k = 0; k < Heads; ++k) pair[k][j] = weights[k * count + c];
                }
            }
            #pragma unroll
            for (int j = 0; j < Tile; ++j) if (valid[j]) {
                #pragma unroll
                for (int k = 0; k < Heads; ++k) {
                    #pragma unroll
                    for (int channel = 0; channel < Dimensions; ++channel)
                        acc[k][channel] = acc[k][channel] * pair[k][j].x + pair[k][j].y * value[channel][j];
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < Heads; ++k) {
            #pragma unroll
            for (int channel = 0; channel < Dimensions; ++channel)
                output[(size_t(token) * heads + h + k) * 512 + d + channel] =
                    __float2bfloat16_rn(acc[k][channel] * inverse[size_t(token) * heads + h + k]);
        }
    }

    // Stage a candidate tile once for 32 heads. The value recurrence retains
    // the original candidate order and FP32 arithmetic, including skipped IDs.
    __global__ void Glm5HybridPrefillValues(const __nv_bfloat16 *kv,
            const int32_t *indices, const float2 *weights, const float *inverse,
            int count, int cap, int heads, __nv_bfloat16 *output) {
        constexpr int headTile = 32, candidateTile = 32, dimensions = 128;
        const int row = blockIdx.z, head = blockIdx.x * headTile;
        const int column = threadIdx.x % dimensions, group = threadIdx.x / dimensions;
        indices += size_t(row) * count;
        weights += (size_t(row) * heads + head) * count;
        __shared__ float values[candidateTile][dimensions];
        __shared__ float2 pairs[headTile][candidateTile];
        __shared__ int ids[candidateTile];
        float sums[headTile / 2] = {};
        for (int first = 0; first < count; first += candidateTile) {
            if (threadIdx.x < candidateTile) {
                const int c = first + threadIdx.x, id = c < count ? indices[c] : -1;
                ids[threadIdx.x] = id >= 0 && id < cap ? id : -1;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < candidateTile * dimensions; i += 256) {
                const int c = i / dimensions, d = i % dimensions, id = ids[c];
                values[c][d] = id >= 0 ? __bfloat162float(kv[size_t(id) * 512 + blockIdx.y * dimensions + d]) : 0.f;
            }
            for (int i = threadIdx.x; i < headTile * candidateTile; i += 256) {
                const int c = i % candidateTile;
                pairs[i / candidateTile][c] = ids[c] >= 0 ?
                    weights[(i / candidateTile) * count + first + c] : make_float2(1.f, 0.f);
            }
            __syncthreads();
            #pragma unroll
            for (int c = 0; c < candidateTile; ++c) if (ids[c] >= 0) {
                const float value = values[c][column];
                #pragma unroll
                for (int a = 0; a < headTile / 2; ++a) {
                    const float2 pair = pairs[group + 2 * a][c];
                    sums[a] = sums[a] * pair.x + pair.y * value;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int a = 0; a < headTile / 2; ++a) {
            const size_t h = size_t(row) * heads + head + group + 2 * a;
            output[h * 512 + blockIdx.y * dimensions + column] = __float2bfloat16_rn(sums[a] * inverse[h]);
        }
    }

    // ---------------- SparseAttention（BF16 mma 版本，SM80+） ----------------
    //
    // Keep the checkpoint's 64-slot online softmax, BF16 probabilities and
    // final sink denominator. Masked window slots must retain their positions:
    // packing them out changes the probability rounding at tile boundaries.
    //   * QK^T 与 PV 用 mma.sync.m16n8k16（BF16 输入 / FP32 累加）代替 FP32 标量点积；
    //   * 候选按 kMmaNC 个一组分块，一组只要 4 次 __syncthreads（原来每个候选 1 次）；
    //   * 16 heads share KV, keeping the 64-slot tile within SM89 shared memory.
    //
    // 片段布局（PTX ISA 的 m16n8k16 定义）：
    //   A(16x16): lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1} [+8])
    //   B(16x8) : lane l 持有 (k = (l%4)*2 + {0,1} [+8], n = l/4)
    //   C(16x8) : lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1})

    constexpr int kMmaHeads = 16;
    constexpr int kMmaNC = 64;
    constexpr int kMmaWarps = 8;
    constexpr int kMmaThreads = kMmaWarps * 32;         // 256
    constexpr int kMmaDim = 512;                        // V4.1 的 head_dim 固定为 512
    constexpr int kMmaKvStride = kMmaDim + 8;           // +8 个半字，消除 ldmatrix 的 bank 冲突
    constexpr int kMmaPStride = kMmaNC + 8;

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


    // Materialized QK needs only the current output slice of V, not Q or
    // the other value slices. Keep the original online-softmax order.
    template <int VALUE_SPLITS>
    struct V41MmaPvShared {
        __nv_bfloat16 kvs[kMmaNC][kMmaDim / VALUE_SPLITS + 8];
        __nv_bfloat16 ps[kMmaHeads][kMmaPStride];
        float sc[kMmaHeads][kMmaNC];
        float mx[kMmaHeads], lsum[kMmaHeads], alpha[kMmaHeads];
        int valid[kMmaNC];
    };


    struct V41AttentionDescriptor {
        const __nv_bfloat16 *q, *chunkKV, *ringKV, *compressedKV;
        const uint8_t *ringKV8, *compressedKV8;
        const int32_t *indices;
        __nv_bfloat16 *output;
        float *scores;
        int cap, topWidth, startPos, ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack;
        int chunkOffset = 0; // query position relative to the shared verify chunk
    };

    // Independent candidate tiles materialize QK once. The consumer keeps
    // the original 64-slot online softmax and BF16 probability sequence.
    template<bool BATCHED = false>
    __global__ void __launch_bounds__(kMmaThreads)
    V41SparseAttentionScoresKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos,
                                float *scores,
                                int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes, int vecUnpack,
                                const V41AttentionDescriptor *descriptors = nullptr) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        if constexpr (BATCHED) {
            const auto &d = descriptors[blockIdx.x];
            q=d.q; chunkKV=d.chunkKV; ringKV=d.ringKV; compressedKV=d.compressedKV;
            ringKV8=d.ringKV8; compressedKV8=d.compressedKV8; cmpIdx=d.indices;
            cap=d.cap; topWidth=d.topWidth; startPos=d.startPos; scores=d.scores;
            ringMode=d.ringMode; ringRowBytes=d.ringRowBytes; cmpMode=d.cmpMode;
            cmpRowBytes=d.cmpRowBytes; vecUnpack=d.vecUnpack;
            if (blockIdx.z >= (windowSize + topWidth + kMmaNC - 1) / kMmaNC) return;
        }
        extern __shared__ char v41MmaSharedRaw[];
        V41MmaShared &sh = *reinterpret_cast<V41MmaShared*>(v41MmaSharedRaw);

        const int t = BATCHED ? 0 : blockIdx.x;                       // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int chunkStart = BATCHED ? startPos - descriptors[blockIdx.x].chunkOffset : startPos;
        const int h0 = blockIdx.y * kMmaHeads;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;

        for (int v = threadIdx.x; v < kMmaHeads * (kMmaDim / 8); v += kMmaThreads) {
            int hh = v / (kMmaDim / 8), dv = v % (kMmaDim / 8);
            *((float4*)&sh.qs[hh][dv * 8]) = h0 + hh < heads
                ? *((const float4*)(q + ((uint64_t)t * heads + h0 + hh) * kMmaDim) + dv)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        const int winSlots = startPos == 0 ? min(seqlen, windowSize) : windowSize;
        const int winStart = startPos == 0 ? max(0, pos - windowSize + 1) : pos - windowSize + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        const int totalCand = winSlots + cmpCount;
        const int tilesTotal = (totalCand + kMmaNC - 1) / kMmaNC;

        const int qkM = 0, qkN = warp;
        const int qkARow = qkM * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int qkAColBlk = (lane >> 4) * 8;
        const int qkBRow = qkN * 8 + (lane & 7);
        const int qkBColBlk = ((lane >> 3) & 1) * 8;
        const int tile = blockIdx.z;
        const int tileStart = tile * kMmaNC;
        __syncthreads();
        for (int slot = warp; slot < kMmaNC; slot += kMmaWarps) {
            const int c = tileStart + slot;
            const __nv_bfloat16 *src = nullptr;
            const uint8_t *src8 = nullptr;
            int src8Mode = 1;
            bool ok = c < totalCand;
            if (ok) {
                if (c < winSlots) {
                    int p = winStart + c;
                    if (p < 0 || p > pos) {
                        ok = false;
                    } else if (p >= chunkStart) {
                        src = chunkKV + ((uint64_t)b * seqlen + (p - chunkStart)) * kMmaDim;
                    } else if (ringKV8 != nullptr) {
                        src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                        src8Mode = ringMode;
                    } else {
                        src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                    }
                } else {
                    int idx = idxRow[c - winSlots];
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

        for (int v = threadIdx.x; v < kMmaHeads * kMmaNC; v += kMmaThreads) {
            const int row = v / kMmaNC, col = v % kMmaNC;
            if (h0 + row < heads) {
                scores[(((uint64_t)t * heads + h0 + row) * tilesTotal + tile) * kMmaNC + col] = sh.sc[row][col];
            }
        }
#endif
    }

    template <int VALUE_SPLITS, bool MATERIALIZED = false, bool BATCHED = false>
    __global__ void __launch_bounds__(kMmaThreads)
    V41SparseAttentionMmaKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, const float *sink, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos, float scale,
                                __nv_bfloat16 *out,
                                int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes, int vecUnpack,
                                const float *scores = nullptr, const V41AttentionDescriptor *descriptors = nullptr) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        if constexpr (BATCHED) {
            const auto &d = descriptors[blockIdx.x];
            q=d.q; chunkKV=d.chunkKV; ringKV=d.ringKV; compressedKV=d.compressedKV;
            ringKV8=d.ringKV8; compressedKV8=d.compressedKV8; cmpIdx=d.indices; out=d.output;
            cap=d.cap; topWidth=d.topWidth; startPos=d.startPos; scores=d.scores;
            ringMode=d.ringMode; ringRowBytes=d.ringRowBytes; cmpMode=d.cmpMode;
            cmpRowBytes=d.cmpRowBytes; vecUnpack=d.vecUnpack;
        }
        extern __shared__ char v41MmaSharedRaw[];
        using Shared = typename std::conditional<MATERIALIZED, V41MmaPvShared<VALUE_SPLITS>, V41MmaShared>::type;
        Shared &sh = *reinterpret_cast<Shared*>(v41MmaSharedRaw);

        const int t = BATCHED ? 0 : blockIdx.x;                       // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int chunkStart = BATCHED ? startPos - descriptors[blockIdx.x].chunkOffset : startPos;
        const int h0 = blockIdx.y * kMmaHeads;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;

        if constexpr (!MATERIALIZED) {
            for (int v = threadIdx.x; v < kMmaHeads * (kMmaDim / 8); v += kMmaThreads) {
                int hh = v / (kMmaDim / 8), dv = v % (kMmaDim / 8);
                *((float4*)&sh.qs[hh][dv * 8]) = h0 + hh < heads
                    ? *((const float4*)(q + ((uint64_t)t * heads + h0 + hh) * kMmaDim) + dv)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        for (int r = threadIdx.x; r < kMmaHeads; r += kMmaThreads) {
            sh.mx[r] = -1e30f;
            sh.lsum[r] = 0.0f;
        }

        const int winSlots = startPos == 0 ? min(seqlen, windowSize) : windowSize;
        const int winStart = startPos == 0 ? max(0, pos - windowSize + 1) : pos - windowSize + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        const int totalCand = winSlots + cmpCount;
        const int tilesTotal = (totalCand + kMmaNC - 1) / kMmaNC;


        // Partition output dimensions only: every block retains the original
        // QK, online-softmax and BF16 probability sequence for all candidates.
        constexpr int kMmaDimSlice = kMmaDim / (kMmaWarps * VALUE_SPLITS);
        constexpr int kMmaPvTiles = kMmaDimSlice / 8;
        const int valueBase = blockIdx.z * (kMmaDim / VALUE_SPLITS);
        constexpr int loadDim = MATERIALIZED ? kMmaDim / VALUE_SPLITS : kMmaDim;
        const int loadBase = MATERIALIZED ? valueBase : 0;
        const int pvM = 0, pvSlice = warp;
        const int qkM = 0, qkN = warp;
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

        for (int tile = 0; tile < tilesTotal; tile++) {
            const int tileStart = tile * kMmaNC;
            __syncthreads();
            if constexpr (MATERIALIZED) {
                // Distribute short value slices across threads, instead of
                // reserving a mostly idle warp for each candidate. This only
                // changes loads; probability rounding and PV order stay intact.
            for (int item = threadIdx.x; item < kMmaNC * (loadDim / 16); item += kMmaThreads) {
                const int slot = item / (loadDim / 16), vector = item % (loadDim / 16);
                const int c = tileStart + slot;
                const __nv_bfloat16 *src = nullptr;
                const uint8_t *src8 = nullptr;
                int src8Mode = 1;
                bool ok = c < totalCand;
                if (ok) {
                    if (c < winSlots) {
                        int p = winStart + c;
                        if (p < 0 || p > pos) {
                            ok = false;
                        } else if (p >= chunkStart) {
                            src = chunkKV + ((uint64_t)b * seqlen + (p - chunkStart)) * kMmaDim;
                        } else if (ringKV8 != nullptr) {
                            src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                            src8Mode = ringMode;
                        } else {
                            src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                        }
                    } else {
                        int idx = idxRow[c - winSlots];
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
                __nv_bfloat16 *dst = &sh.kvs[slot][vector * 16];
                if (!ok) {
                    ((float4*)dst)[0] = make_float4(0.f,0.f,0.f,0.f);
                    ((float4*)dst)[1] = make_float4(0.f,0.f,0.f,0.f);
                } else if (src8 != nullptr) {
                    if (vecUnpack) {
                        V41LoadKvRow16Bf16(src8, kMmaDim, loadBase + vector * 16, src8Mode, dst);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j)
                            dst[j] = __float2bfloat16_rn(V41LoadKvRow(
                                src8, kMmaDim, loadBase + vector * 16 + j, src8Mode));
                    }
                } else {
                    ((float4*)dst)[0] = ((const float4*)(src+loadBase+vector*16))[0];
                    ((float4*)dst)[1] = ((const float4*)(src+loadBase+vector*16))[1];
                }
                if (vector == 0) sh.valid[slot] = ok ? 1 : 0;
            }
            } else {
            for (int slot = warp; slot < kMmaNC; slot += kMmaWarps) {
                const int c = tileStart + slot;
                const __nv_bfloat16 *src = nullptr;
                const uint8_t *src8 = nullptr;
                int src8Mode = 1;
                bool ok = c < totalCand;
                if (ok) {
                    if (c < winSlots) {
                        int p = winStart + c;
                        if (p < 0 || p > pos) {
                            ok = false;
                        } else if (p >= chunkStart) {
                            src = chunkKV + ((uint64_t)b * seqlen + (p - chunkStart)) * kMmaDim;
                        } else if (ringKV8 != nullptr) {
                            src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                            src8Mode = ringMode;
                        } else {
                            src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                        }
                    } else {
                        int idx = idxRow[c - winSlots];
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
                    for (int v = lane; v < loadDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else if (src8 != nullptr) {
                    // 就地解到 BF16 片段再进 mma；FP4 的值在 BF16 上是精确的
                    if (vecUnpack) {
                        // Decode only the current output slice in 16-value groups.
                        for (int v = lane; v < loadDim / 16; v += 32)
                            V41LoadKvRow16Bf16(src8, kMmaDim, loadBase + v * 16, src8Mode, &sh.kvs[slot][v * 16]);
                    } else {
                        for (int d = lane; d < loadDim; d += 32) {
                            sh.kvs[slot][d] = __float2bfloat16_rn(V41LoadKvRow(src8, kMmaDim, loadBase + d, src8Mode));
                        }
                    }
                } else {
                    for (int v = lane; v < loadDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = *((const float4*)(src + loadBase) + v);
                    }
                }
                if (lane == 0) {
                    sh.valid[slot] = ok ? 1 : 0;
                }
            }
            }
            __syncthreads();

            if constexpr (MATERIALIZED) {
                for (int v = threadIdx.x; v < kMmaHeads * kMmaNC; v += kMmaThreads) {
                    const int row = v / kMmaNC, col = v % kMmaNC;
                    sh.sc[row][col] = h0 + row < heads
                        ? scores[(((uint64_t)t * heads + h0 + row) * tilesTotal + tile) * kMmaNC + col]
                        : 0.0f;
                }
                __syncthreads();
            } else {
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

            }
            {   // 16 threads reduce each head's 64 probability slots.
                constexpr int perThread = kMmaNC / 16;
                const int row = threadIdx.x >> 4;
                const int sub = threadIdx.x & 15;
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
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 8));
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
                s += __shfl_xor_sync(0xffffffff, s, 8);
                // Finish every lane's oldMx read before the owner updates it.
                __syncwarp();
                if (sub == 0) {
                    float a = __expf(oldMx - newMx);
                    sh.alpha[row] = a;
                    sh.mx[row] = newMx;
                    sh.lsum[row] = sh.lsum[row] * a + s;
                }
            }
            __syncthreads();

            {   // PV: O[16 heads, 64 dims] = alpha * O + P[16, 64] x V[64, 64]
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
                        V41LdmX2T(bb, &sh.kvs[k + pvBRow][(MATERIALIZED ? 0 : valueBase) + pvSlice * kMmaDimSlice + n * 8]);
                        V41MmaBf16(acc[n], a, bb);
                    }
                }
            }
        }
        __syncthreads();

        const int outRow0 = pvM * 16 + (lane >> 2);
        const int outRow1 = outRow0 + 8;
        const int outCol = valueBase + pvSlice * kMmaDimSlice + (lane & 3) * 2;
        const bool valid0 = h0 + outRow0 < heads, valid1 = h0 + outRow1 < heads;
        const float inv0 = valid0 ? 1.0f / (sh.lsum[outRow0] + expf(sink[h0 + outRow0] - sh.mx[outRow0])) : 0.0f;
        const float inv1 = valid1 ? 1.0f / (sh.lsum[outRow1] + expf(sink[h0 + outRow1] - sh.mx[outRow1])) : 0.0f;
        __nv_bfloat16 *o0 = out + ((uint64_t)t * heads + h0 + outRow0) * kMmaDim;
        __nv_bfloat16 *o1 = out + ((uint64_t)t * heads + h0 + outRow1) * kMmaDim;
#pragma unroll
        for (int n = 0; n < kMmaPvTiles; n++) {
            int d = outCol + n * 8;
            if (valid0) {
                o0[d] = __float2bfloat16_rn(acc[n][0] * inv0);
                o0[d + 1] = __float2bfloat16_rn(acc[n][1] * inv0);
            }
            if (valid1) {
                o1[d] = __float2bfloat16_rn(acc[n][2] * inv1);
                o1[d + 1] = __float2bfloat16_rn(acc[n][3] * inv1);
            }
        }
#endif
    }

    #include "deepseekv41-sm120-prefill.cuh"

    // Check the compiled image as well as hardware. An SM75 PTX image JITed
    // on a newer GPU still contains the #if-disabled MMA body. Resource or
    // image rejection must happen before launch so the scalar path is safe.
    template <auto Kernel, int Threads, int DynamicShared = 0>
    bool V41MmaKernelReady() {
        static std::mutex mutex;
        static std::atomic<int> ready[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const int device = FastllmCudaGetDevice();
        if (device < 0 || device >= 16) return false;
        const int cached = ready[device].load(std::memory_order_acquire);
        if (cached >= 0) return cached == 1;
        std::lock_guard<std::mutex> lock(mutex);
        if (ready[device] < 0) {
            int result = 0;
            cudaFuncAttributes attr{};
            int major = 0, sharedLimit = 0;
            auto supported = [](cudaError_t status) {
                if (status == cudaSuccess) return true;
                if (status != cudaErrorInvalidDeviceFunction && status != cudaErrorNoKernelImageForDevice &&
                    status != cudaErrorNotSupported && status != cudaErrorInvalidValue) {
                    FastllmCudaSetThreadError();
                    fprintf(stderr, "[Fastllm] V4.1 kernel capability query failed: %s\n", cudaGetErrorString(status));
                }
                cudaGetLastError();
                return false;
            };
            const bool fits = supported(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) &&
                major >= 8 && supported(cudaFuncGetAttributes(&attr, Kernel)) &&
                attr.ptxVersion >= 80 && attr.binaryVersion >= 80 && attr.maxThreadsPerBlock >= Threads &&
                supported(cudaDeviceGetAttribute(&sharedLimit, DynamicShared > 0
                    ? cudaDevAttrMaxSharedMemoryPerBlockOptin : cudaDevAttrMaxSharedMemoryPerBlock, device)) &&
                attr.sharedSizeBytes + DynamicShared <= (size_t)sharedLimit;
            if (fits && (DynamicShared == 0 || supported(cudaFuncSetAttribute(Kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, DynamicShared)))) result = 1;
            ready[device].store(result, std::memory_order_release);
        }
        return ready[device] == 1;
    }

    // Cache the architecture per host thread/device: querying it for every
    // decode layer would add runtime API traffic to an already small launch.
    template <int Major, int Minor>
    bool V41IsSm() {
        static thread_local int cached[16] = {};
        const int device = FastllmCudaGetDevice();
        if (device < 0 || device >= 16) return false;
        if (cached[device] == 0) {
            int major = 0, minor = 0;
            cudaError_t status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            if (status == cudaSuccess) {
                status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            }
            if (status != cudaSuccess) {
                FastllmCudaSetThreadError();
                fprintf(stderr, "[Fastllm] V4.1 decode architecture query failed: %s\n", cudaGetErrorString(status));
                return false;
            }
            cached[device] = major == Major && minor == Minor ? 1 : -1;
        }
        return cached[device] == 1;
    }

    bool V41IsSm75() { return V41IsSm<7, 5>(); }

    void V41LaunchHcDecodeDots(const __nv_bfloat16 *x, const float *fn,
                              int tokens, int flatDim, float *mix) {
        const dim3 grid(25, tokens);
        if (tokens <= 8 && flatDim >= 16384 && flatDim <= 32768 && V41IsSm<9, 0>()) {
            if (tokens <= 3)
                V41HcDecodeDotsSm90<64><<<grid, kHcThreads>>>(x, fn, flatDim, mix);
            else
                V41HcDecodeDotsSm90<16><<<grid, kHcThreads>>>(x, fn, flatDim, mix);
        } else {
            V41HcDecodeDots<<<grid, kHcThreads>>>(x, fn, flatDim, mix);
        }
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

    // Quantized weights still require the checkpoint's FP8 activation grid
    // after loading them into FP16. Each warp owns one independent block-32.
    // Accumulate the 32-element checkpoint blocks in K order. Dequantized
    // FP8 operands are exact binary values, but a full-K reduction can change
    // a later BF16 tie. Keep the block sum separate from the running total.
    template <typename W>
    __global__ void V41LinearBlock32Kernel(const __nv_bfloat16 *input, const W *weight,
                                           __nv_bfloat16 *output, int n, int k) {
        const int row = blockIdx.x * 4 + threadIdx.x / 32;
        const int lane = threadIdx.x % 32, token = blockIdx.y;
        if (row >= n) return;
        float total = 0.0f;
        for (int block = 0; block < k; block += 32) {
            float part = __bfloat162float(input[(uint64_t)token * k + block + lane]) *
                         V41Load<W>(weight, (uint64_t)row * k + block + lane);
            for (int offset = 16; offset > 0; offset /= 2) {
                part += __shfl_down_sync(0xffffffff, part, offset);
            }
            if (lane == 0) total += part;
        }
        if (lane == 0) output[(uint64_t)token * n + row] = __float2bfloat16_rn(total);
    }

    // Eight consecutive BF16 values per lane reduce the number of CTAs on
    // large tensors. Four lanes still reduce exactly one block-32 maximum;
    // conversion, scaling and storage rounding are unchanged.

template<int V>
__global__ void V41QuantizeActivationVectorKernel(const __nv_bfloat16 *input, __nv_bfloat16 *output, uint64_t elements) {
    constexpr int GROUP = 32 / V;
    const uint64_t index = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * V;
    __align__(16) __nv_bfloat16 values[V];
    if (index + V <= elements) {
        if constexpr (V == 8) *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(input + index);
        if constexpr (V == 4) *reinterpret_cast<uint2*>(values) = *reinterpret_cast<const uint2*>(input + index);
        if constexpr (V == 2) *reinterpret_cast<uint32_t*>(values) = *reinterpret_cast<const uint32_t*>(input + index);
    } else {
#pragma unroll
        for (int j = 0; j < V; ++j) values[j] = index+j < elements ? input[index+j] : __float2bfloat16_rn(0.f);
    }
    float maximum = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) maximum = fmaxf(maximum, fabsf(__bfloat162float(values[j])));
#pragma unroll
    for (int offset = GROUP/2; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, GROUP));
    const float scale = V41QuantScaleDev(maximum, 1);
#pragma unroll
    for (int j = 0; j < V; ++j) values[j] = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(values[j]), scale, 1));
    if (index + V <= elements) {
        if constexpr (V == 8) *reinterpret_cast<uint4*>(output + index) = *reinterpret_cast<const uint4*>(values);
        if constexpr (V == 4) *reinterpret_cast<uint2*>(output + index) = *reinterpret_cast<const uint2*>(values);
        if constexpr (V == 2) *reinterpret_cast<uint32_t*>(output + index) = *reinterpret_cast<const uint32_t*>(values);
    } else {
#pragma unroll
        for (int j = 0; j < V; ++j) if (index+j < elements) output[index+j] = values[j];
    }
}
    template<class T>
    __global__ void V41QuantizeActivationKernel(const T *input, T *output, uint64_t elements) {
        uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        float value = index < elements ? V41Load<T>(input, index) : 0.0f;
        float maximum = fabsf(value);
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float scale = V41QuantScaleDev(maximum, 1);
        if (index < elements) V41Store<T>(output, index, V41QuantValueDev(value, scale, 1));
    }

    // Match FastllmRMSNormKernelInner1<512>: pairwise loading, FMA order,
    // shuffle-down tree, and BF16 output rounding. Q quantization consumes that
    // rounded value, never the unrounded FP32 expression.
    template <int CACHE_PAIRS = 0>
    __global__ void V41QKVNormQuantKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *qq,
            __nv_bfloat16 *kn, int qdim, int kdim, int qStride, int kStride, float eps) {
        const bool query = (blockIdx.x & 1) == 0;
        const int token = blockIdx.x / 2, dim = query ? qdim : kdim;
        const auto *input = reinterpret_cast<const __nv_bfloat162 *>(query ? q : kv) +
            (size_t)token * ((query ? qStride : kStride) / 2);
        auto *output = reinterpret_cast<__nv_bfloat162 *>(query ? qn : kn) + (size_t)token * (dim / 2);
        const float *weight = query ? qw : kw;
        const int tid = threadIdx.x, lane = tid % 32;
        __shared__ float warpSums[16];
        __nv_bfloat162 values[CACHE_PAIRS ? CACHE_PAIRS : 1];
        float2 weights[CACHE_PAIRS ? CACHE_PAIRS : 1];
        float sum = 0.0f;
        for (int i = tid; i < dim / 2; i += 512) {
            const __nv_bfloat162 v = input[i];
            if constexpr (CACHE_PAIRS) {
                values[i / 512] = v;
                weights[i / 512] = make_float2(__ldg(weight + 2 * i), __ldg(weight + 2 * i + 1));
            }
            const float lo = __bfloat162float(v.x), hi = __bfloat162float(v.y);
            sum += lo * lo + hi * hi;
        }
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) warpSums[tid / 32] = sum;
        __syncthreads();
        float val = lane < 16 ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1) val += __shfl_down_sync(0xffffffff, val, offset);
        const float s = rsqrtf(__shfl_sync(0xffffffff, val, 0) / dim + eps);
        // Uniform iteration count keeps every lane participating in shuffles.
        for (int base = 0; base < dim / 2; base += 512) {
            const int i = base + tid;
            __nv_bfloat162 v = {};
            if (i < dim / 2) {
                // Small Q/KV rows fit in registers: overlap weight loads with
                // the sum reduction and avoid reloading the BF16 input.
                const float2 w = CACHE_PAIRS ? weights[base / 512] :
                    make_float2(__ldg(weight + 2 * i), __ldg(weight + 2 * i + 1));
                v = CACHE_PAIRS ? values[base / 512] : input[i];
                v.x = __float2bfloat16_rn(__bfloat162float(v.x) * s * w.x);
                v.y = __float2bfloat16_rn(__bfloat162float(v.y) * s * w.y);
                output[i] = v;
            }
            if (query) {
                float maximum = fmaxf(fabsf(__bfloat162float(v.x)), fabsf(__bfloat162float(v.y)));
                // Sixteen lanes hold one contiguous 32-element quantization block.
                for (int offset = 8; offset; offset >>= 1)
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 16));
                const float qs = V41QuantScaleDev(maximum, 1);
                v.x = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(v.x), qs, 1));
                v.y = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(v.y), qs, 1));
                if (i < dim / 2) reinterpret_cast<__nv_bfloat162 *>(qq)[(size_t)token * (dim / 2) + i] = v;
            }
        }
    }

    // ---------------- QuantizeKV ----------------

    constexpr int kKvQuantThreads = 128;

    // 每个 block 处理一行；blockSize 个线程为一组处理一个 scale 块，lane 对应块内元素。
    // Reuse the fake-quantization grid, including the subnormal E4M3 case
    // where the quantized maximum alone would otherwise halve the scale.
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
            float scale = V41QuantScaleDev(amax, quantMode);
            // A 2^-8 E4M3 scale may round its maximum to 4*scale = 2^-6.
            // Recomputing from that maximum gives 2^-9 and clips it to 6*2^-9.
            // Preserve the original grid in storage; leave RotaryQuant intact.
            if (quantMode == 3 && amax == 0x1p-6f) scale = 0x1p-8f;
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
                                         int startPos, int windowSize, int firstRow, const int *devicePosition) {
        const int b = blockIdx.y;
        const int i = firstRow + blockIdx.x;
        if (i >= seqlen) {
            return;
        }
        const int slot = ((devicePosition ? *devicePosition : startPos) + i) % windowSize;
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
    #include "deepseekv41-prefill-attention.cuh"
}

// ==================== 导出接口 ====================

template <bool QUANTIZE>
static bool V41RunSharedSwiglu(const fastllm::Data &input, float limit, fastllm::Data &output, const float *routeScales = nullptr) {
    if (!V41OnCuda(input) || input.multiDeviceData || input.dims.empty() ||
        input.dims.back() <= 0 || input.dims.back() % 2 || !V41IsFloatType(input.dataType)) return false;
    auto dims = input.dims;
    const int mid = dims.back() / 2;
    if (QUANTIZE && mid % 32 != 0) return false;
    dims.back() = mid;
    if (!V41PrepareOutput(output, DataType::BFLOAT16, dims)) return false;
    const uint64_t count = input.Count(0) / 2;
    if (count == 0) return true;
    const int blocks = (count + 255) / 256;
    auto out = (__nv_bfloat16 *)output.cudaData;
    if (input.dataType == DataType::BFLOAT16)
        V41LaunchBf16Swiglu<QUANTIZE>(input.cudaData, out, count, mid, limit, routeScales);
    else if (input.dataType == DataType::FLOAT16)
        V41SharedSwigluKernel<half, QUANTIZE><<<blocks, 256>>>((const half *)input.cudaData, out, count, mid, limit, routeScales);
    else
        V41SharedSwigluKernel<float, QUANTIZE><<<blocks, 256>>>((const float *)input.cudaData, out, count, mid, limit, routeScales);
    return V41CheckLaunch("SharedSwiglu");
}

bool FastllmCudaDeepSeekV41BFloat16SwigluQuantized(const void *input, void *output,
        int rows, int intermediate, float limit, const float *routeScales) {
    if (!input || !output || rows <= 0 || intermediate <= 0 || intermediate % 32) return false;
    const uint64_t count = (uint64_t)rows * intermediate;
    V41LaunchBf16Swiglu<true>(input, output, count, intermediate, limit, routeScales);
    return V41CheckLaunch("BFloat16SwigluQuantized");
}

bool FastllmCudaDeepSeekV41SharedSwiglu(const fastllm::Data &input, float limit, fastllm::Data &output) {
    return V41RunSharedSwiglu<false>(input, limit, output);
}

bool FastllmCudaDeepSeekV41SharedSwigluQuantized(const fastllm::Data &input, float limit, fastllm::Data &output, const float *routeScales) {
    return V41RunSharedSwiglu<true>(input, limit, output, routeScales);
}

extern "C" bool FastllmCudaV41QKVNormQuantPacked(const Data &qkv,
        const Data &qw, const Data &kw, float eps, Data &qn, Data &qq, Data &kn) {
    const int device = FastllmCudaGetDevice();
    auto local = [device](const Data &d) {
        return V41OnCuda(d) && !d.multiDeviceData &&
               d.dataDeviceIds == std::vector<int>{device};
    };
    if (!local(qkv) || !local(qw) || !local(kw) || qkv.dataType != DataType::BFLOAT16 ||
        qw.dataType != DataType::FLOAT32 || kw.dataType != DataType::FLOAT32 ||
        qkv.dims.empty() || !std::isfinite(eps) || eps < 0) return false;
    const uint64_t qCount = qw.Count(0), kCount = kw.Count(0);
    if (qCount < 512 || qCount >= 4096 || qCount % 32 ||
        kCount < 512 || kCount >= 4096 || kCount % 32) return false;
    uint64_t denseStride = 1;
    if (qkv.strides.size() != qkv.dims.size()) return false;
    for (int axis = (int)qkv.dims.size() - 1; axis >= 0; --axis) {
        if (qkv.dims[axis] <= 0 || qkv.strides[axis] != denseStride) return false;
        denseStride *= qkv.dims[axis];
    }
    const int qdim = qCount, kdim = kCount, stride = qdim + kdim;
    if (qkv.dims.back() != stride || (uintptr_t)qkv.cudaData % 4) return false;
    const uint64_t tokens = qkv.Count(0) / stride;
    if (!tokens || tokens > 8 || &qn == &qq || &qn == &kn || &qq == &kn) return false;
    auto overlaps = [](const void *a, size_t na, const void *b, size_t nb) {
        const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
        return a && b && (x <= y ? y - x < na : x - y < nb);
    };
    for (Data *out : {&qn, &qq, &kn}) {
        const size_t bytes = tokens * (out == &kn ? kdim : qdim) * 2;
        if ((uintptr_t)out->cudaData % 4) return false;
        for (const Data *src : {&qkv, &qw, &kw})
            if (out == src || overlaps(out->cudaData, bytes, src->cudaData, src->GetBytes())) return false;
    }
    const size_t qb = tokens * qdim * 2, kb = tokens * kdim * 2;
    if (overlaps(qn.cudaData, qb, qq.cudaData, qb) || overlaps(qn.cudaData, qb, kn.cudaData, kb) ||
        overlaps(qq.cudaData, qb, kn.cudaData, kb)) return false;
    auto qdims = qkv.dims, kdims = qkv.dims;
    qdims.back() = qdim; kdims.back() = kdim;
    if (!V41PrepareOutput(qn, DataType::BFLOAT16, qdims) ||
        !V41PrepareOutput(qq, DataType::BFLOAT16, qdims) ||
        !V41PrepareOutput(kn, DataType::BFLOAT16, kdims)) return false;
#define V41_PACKED_NORM(C) V41QKVNormQuantKernel<C><<<tokens * 2, 512>>>( \
        (const __nv_bfloat16*)qkv.cudaData, (const __nv_bfloat16*)qkv.cudaData + qdim, \
        (const float*)qw.cudaData, (const float*)kw.cudaData, (__nv_bfloat16*)qn.cudaData, \
        (__nv_bfloat16*)qq.cudaData, (__nv_bfloat16*)kn.cudaData, qdim, kdim, stride, stride, eps)
    if (qdim <= 2048 && kdim <= 2048) V41_PACKED_NORM(2);
    else V41_PACKED_NORM(0);
#undef V41_PACKED_NORM
    return V41CheckLaunch("QKVNormQuantPacked");
}

extern "C" bool FastllmCudaV41QKVNormQuant(const Data &q, const Data &kv,
        const Data &qWeight, const Data &kvWeight, float eps, Data &qNorm, Data &qQuant, Data &kvNorm) {
    const int device = FastllmCudaGetDevice();
    auto local = [device](const Data &d) {
        return V41OnCuda(d) && !d.multiDeviceData && d.dataDeviceIds.size() == 1 &&
               d.dataDeviceIds[0] == device;
    };
    if (!local(q) || !local(kv) || !local(qWeight) || !local(kvWeight) ||
        q.dataType != DataType::BFLOAT16 || kv.dataType != DataType::BFLOAT16 ||
        qWeight.dataType != DataType::FLOAT32 || kvWeight.dataType != DataType::FLOAT32 ||
        q.dims.empty() || kv.dims.size() != q.dims.size()) return false;
    const int qdim = q.dims.back(), kdim = kv.dims.back();
    if (qdim < 512 || qdim >= 4096 || qdim % 32 || kdim < 512 || kdim >= 4096 || kdim % 32 ||
        qWeight.Count(0) != (uint64_t)qdim || kvWeight.Count(0) != (uint64_t)kdim ||
        reinterpret_cast<uintptr_t>(q.cudaData) % 4 || reinterpret_cast<uintptr_t>(kv.cudaData) % 4) return false;
    auto kdims = q.dims;
    kdims.back() = kdim;
    const auto tokens = q.Count(0) / qdim;
    if (tokens < 1 || tokens > 8 || kdims != kv.dims || !std::isfinite(eps) || eps < 0) return false;
    if (&qNorm == &qQuant || &qNorm == &kvNorm || &qQuant == &kvNorm || &q == &qNorm ||
        &q == &qQuant || &q == &kvNorm || &kv == &qNorm || &kv == &qQuant) return false;
    auto overlaps = [](const void *a, size_t na, const void *b, size_t nb) {
        const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
        return a && b && (x <= y ? y - x < na : x - y < nb);
    };
    const size_t qbytes = q.GetBytes(), kbytes = kv.GetBytes();
    for (Data *out : {&qNorm, &qQuant, &kvNorm}) {
        if (!out->cudaData) continue;
        if (reinterpret_cast<uintptr_t>(out->cudaData) % 4) return false;
        const size_t bytes = out == &kvNorm ? kbytes : qbytes;
        for (const Data *source : {&q, &kv, &qWeight, &kvWeight}) {
            if (out == &kvNorm && source == &kv && out->cudaData == kv.cudaData) continue;
            if (overlaps(out->cudaData, bytes, source->cudaData, source->GetBytes())) return false;
        }
    }
    if (overlaps(qNorm.cudaData, qbytes, qQuant.cudaData, qbytes) ||
        overlaps(qNorm.cudaData, qbytes, kvNorm.cudaData, kbytes) ||
        overlaps(qQuant.cudaData, qbytes, kvNorm.cudaData, kbytes)) return false;
    if (!V41PrepareOutput(qNorm, DataType::BFLOAT16, q.dims) ||
        !V41PrepareOutput(qQuant, DataType::BFLOAT16, q.dims) ||
        !V41PrepareOutput(kvNorm, DataType::BFLOAT16, kv.dims)) return false;
#define V41_QKV_NORM(C) V41QKVNormQuantKernel<C><<<tokens * 2, 512>>>((const __nv_bfloat16 *)q.cudaData, \
        (const __nv_bfloat16 *)kv.cudaData, (const float *)qWeight.cudaData, (const float *)kvWeight.cudaData, \
        (__nv_bfloat16 *)qNorm.cudaData, (__nv_bfloat16 *)qQuant.cudaData, (__nv_bfloat16 *)kvNorm.cudaData, \
        qdim, kdim, qdim, kdim, eps)
    if (qdim <= 2048 && kdim <= 2048) V41_QKV_NORM(2);
    else V41_QKV_NORM(0);
#undef V41_QKV_NORM
    return V41CheckLaunch("QKVNormQuant");
}

extern "C" bool FastllmCudaV41RouterScores(const Data &input, Data &weight,
                                            float temperature, Data &scores) {
    if (!V41OnCuda(input) || input.multiDeviceData || weight.multiDeviceData ||
        input.dataType != DataType::BFLOAT16 || weight.dataType != DataType::FLOAT32 ||
        input.dims.empty() || input.dims.back() <= 0 || weight.dims.size() != 2 ||
        input.dims.back() != weight.dims[1] ||
        input.Count(0) / input.dims.back() < 1 || input.Count(0) / input.dims.back() > 8 ||
        weight.dims[0] < 1 || weight.dims[0] > 1024 || temperature <= 0) return false;
    weight.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()}, true);
    auto dims = input.dims;
    dims.back() = weight.dims[0];
    if (!V41PrepareOutput(scores, DataType::FLOAT32, dims)) return false;
    const float invTemp = fabsf(temperature - 1.0f) > 1e-6f ? 1.0f / temperature : 1.0f;
    V41RouterScoresKernel<<<dim3(weight.dims[0], input.Count(0) / input.dims.back()), 256>>>(
        (const __nv_bfloat16 *)input.cudaData, (const float *)weight.cudaData,
        (float *)scores.cudaData, weight.dims[1], weight.dims[0], invTemp);
    return V41CheckLaunch("RouterScores");
}

// Small BF16 decode batches retain the original FP32 dot accumulation;
// finish and normalization use independent CTAs in one launch, preserving arithmetic.
extern "C" bool FastllmCudaDeepSeekV41HcMixPreNorm(const Data &x, const Data &previousPre, Data &fn,
        Data &scale, Data &base, Data &normWeight, int sinkhornIters, float mixEps,
        float normEps, Data &pre, Data &post, Data &comb, Data &output, Data *quantizedOutput) {
    if (!V41OnCuda(x) || !V41OnCuda(previousPre) || previousPre.dataType != DataType::FLOAT32 ||
        x.multiDeviceData || previousPre.multiDeviceData || x.dataType != DataType::BFLOAT16 ||
        x.dims.size() != 4 || x.dims[2] != 4 || x.dims[0] * x.dims[1] < 1 ||
        x.dims[0] * x.dims[1] >= 16 || x.dims[3] < 4096 || x.dims[3] > 8192 ||
        (x.dims[3] & 1) || fn.dataType != DataType::FLOAT32 ||
        scale.dataType != DataType::FLOAT32 || base.dataType != DataType::FLOAT32 ||
        normWeight.dataType != DataType::FLOAT32 || normWeight.Count(0) != x.dims[3]) return false;
    const int b = x.dims[0], n = x.dims[1], d = x.dims[3], tokens = b * n;
    if (quantizedOutput != nullptr && (d % 32 != 0 || quantizedOutput == &x || quantizedOutput == &output)) return false;
    if (previousPre.Count(0) != (uint64_t)tokens * 4 || fn.Count(0) != (uint64_t)24 * 4 * d ||
        scale.Count(0) != 3 || base.Count(0) != 24) return false;
    for (Data *w : {&fn, &scale, &base, &normWeight})
        w->ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()}, true);
    Data scratch;
    if (!V41PrepareOutput(scratch, DataType::FLOAT32, {tokens, 25}) ||
        !V41PrepareOutput(pre, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(post, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(comb, DataType::FLOAT32, {b, n, 4, 4}) ||
        !V41PrepareOutput(output, DataType::BFLOAT16, {b, n, d})) return false;
    if (quantizedOutput != nullptr &&
        !V41PrepareOutput(*quantizedOutput, DataType::BFLOAT16, {b, n, d})) return false;
    V41LaunchHcDecodeDots((const __nv_bfloat16 *)x.cudaData, (const float *)fn.cudaData,
                         tokens, 4 * d, (float *)scratch.cudaData);
#define V41_MIX_NORM(I, Q) V41HcPreNormKernel<1024, I, true, Q><<<tokens * 2, 1024>>>( \
        (const __nv_bfloat16 *)x.cudaData, (const float *)previousPre.cudaData, \
        (const float *)normWeight.cudaData, (__nv_bfloat16 *)output.cudaData, 4, d, normEps, \
        (const float *)scratch.cudaData, (const float *)scale.cudaData, (const float *)base.cudaData, \
        sinkhornIters, mixEps, (float *)post.cudaData, (float *)comb.cudaData, (float *)pre.cudaData, \
        Q ? (__nv_bfloat16 *)quantizedOutput->cudaData : nullptr)
    if (d <= 4096) { if (quantizedOutput) V41_MIX_NORM(2, true); else V41_MIX_NORM(2, false); }
    else if (d <= 6144) { if (quantizedOutput) V41_MIX_NORM(3, true); else V41_MIX_NORM(3, false); }
    else { if (quantizedOutput) V41_MIX_NORM(4, true); else V41_MIX_NORM(4, false); }
#undef V41_MIX_NORM
    return V41CheckLaunch("HcMixPreNorm");
}

extern "C" bool FastllmCudaDeepSeekV41HcMix(const fastllm::Data &x, fastllm::Data &hcFn, fastllm::Data &hcScale,
                                            fastllm::Data &hcBase, int hcMult, int sinkhornIters, float eps,
                                            float normEps, fastllm::Data &pre, fastllm::Data &post,
                                            fastllm::Data &comb) {
    if (V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && FastllmCudaV41ReferenceMix(x,hcFn,hcScale,hcBase,hcMult,sinkhornIters,eps,normEps,pre,post,comb)) return true;
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
    if (tokens >= 1 && hcMult == 4 && !V41EnvOn("FASTLLM_DSV41_LEGACY_HCMIX") &&
        !V41EnvOn("FASTLLM_DSV41_LEGACY_HCMIX_DECODE")) {
        Data scratch;
        if (!V41PrepareOutput(scratch, DataType::FLOAT32, {tokens, 25})) return false;
        float *mix = (float *)scratch.cudaData;
        if (tokens > 192) {
            const int blocks = (tokens + kHcTokens - 1) / kHcTokens;
            if (x.dataType == DataType::BFLOAT16)
                V41HcMixKernelMulti<__nv_bfloat16, 4, true><<<blocks, kHcThreads>>>(
                    (const __nv_bfloat16 *)x.cudaData, (const float *)hcFn.cudaData,
                    nullptr, nullptr, tokens, dim, sinkhornIters, eps, normEps,
                    nullptr, nullptr, nullptr, mix);
            else if (x.dataType == DataType::FLOAT16)
                V41HcMixKernelMulti<half, 4, true><<<blocks, kHcThreads>>>(
                    (const half *)x.cudaData, (const float *)hcFn.cudaData,
                    nullptr, nullptr, tokens, dim, sinkhornIters, eps, normEps,
                    nullptr, nullptr, nullptr, mix);
            else
                V41HcMixKernelMulti<float, 4, true><<<blocks, kHcThreads>>>(
                    (const float *)x.cudaData, (const float *)hcFn.cudaData,
                    nullptr, nullptr, tokens, dim, sinkhornIters, eps, normEps,
                    nullptr, nullptr, nullptr, mix);
        } else if (tokens >= 16) {
            if (x.dataType == DataType::BFLOAT16)
                V41HcBatchDots<<<dim3(25, (tokens + 3) / 4), kHcThreads>>>((const __nv_bfloat16 *)x.cudaData, (const float *)hcFn.cudaData, hcMult * dim, tokens, mix);
            else if (x.dataType == DataType::FLOAT16)
                V41HcBatchDots<<<dim3(25, (tokens + 3) / 4), kHcThreads>>>((const half *)x.cudaData, (const float *)hcFn.cudaData, hcMult * dim, tokens, mix);
            else
                V41HcBatchDots<<<dim3(25, (tokens + 3) / 4), kHcThreads>>>((const float *)x.cudaData, (const float *)hcFn.cudaData, hcMult * dim, tokens, mix);
        } else {
            if (x.dataType == DataType::BFLOAT16)
                V41LaunchHcDecodeDots((const __nv_bfloat16 *)x.cudaData,
                                     (const float *)hcFn.cudaData, tokens, hcMult * dim, mix);
            else if (x.dataType == DataType::FLOAT16)
                V41HcDecodeDots<<<dim3(25, tokens), kHcThreads>>>((const half *)x.cudaData, (const float *)hcFn.cudaData, hcMult * dim, mix);
            else
                V41HcDecodeDots<<<dim3(25, tokens), kHcThreads>>>((const float *)x.cudaData, (const float *)hcFn.cudaData, hcMult * dim, mix);
        }
        V41HcDecodeFinish<<<tokens, 32>>>(mix, (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            hcMult * dim, sinkhornIters, eps, normEps, (float *)pre.cudaData, (float *)post.cudaData, (float *)comb.cudaData);
        return V41CheckLaunch("HcMixDecode");
    }
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
    if (V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && FastllmCudaV41ReferencePreNorm(x,pre,normWeight,eps,output)) return true;
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
    // Called directly by TP workers as well as the single-device executor.
    // The default executor device may differ from this worker's device.
    normWeight.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()}, true);
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
    if (V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && FastllmCudaV41ReferenceEngram(hidden,kv,qWeight,kWeight,mask,eps,clampValue)) return true;
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

extern "C" bool FastllmCudaDeepSeekV41RotaryQuantPosition(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                                  int posStep, bool inverse, int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, int quantMode, int quantDim,
                                                  int quantBlock, const int *devicePosition) {
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
    if (devicePosition && (V41LegacyRotary() || V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH"))) return false;
    if (V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && FastllmCudaV41ReferenceRotary(x,ropeDim,ropeBase,startPos,posStep,inverse,originalSeqLen,ropeFactor,betaFast,betaSlow,quantMode,quantDim,quantBlock)) return true;
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
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs, devicePosition);
        } else if (x.dataType == DataType::FLOAT16) {
            V41RotaryOnlyKernel<half><<<(unsigned)blocks, threads>>>((half*)x.cudaData,
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs, devicePosition);
        } else {
            V41RotaryOnlyKernel<float><<<(unsigned)blocks, threads>>>((float*)x.cudaData,
                rowsPerToken, dim, rope, ropeDim, startPos, posStep, inverse ? 1 : 0, totalPairs, devicePosition);
        }
        return V41CheckLaunch("RotaryQuant");
    }
    // 量化：一个 block 处理 rowsPerBlock 行，目标 blockDim 512
    const int rowsPerBlock = std::max(1, 512 / dim);
    const int threads = rowsPerBlock * dim;
    const int blocks = (rows + rowsPerBlock - 1) / rowsPerBlock;
    if (x.dataType == DataType::BFLOAT16) {
        V41RotaryQuantKernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows, devicePosition);
    } else if (x.dataType == DataType::FLOAT16) {
        V41RotaryQuantKernel<half><<<blocks, threads>>>((half*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows, devicePosition);
    } else {
        V41RotaryQuantKernel<float><<<blocks, threads>>>((float*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock, rowsPerBlock, rows, devicePosition);
    }
    return V41CheckLaunch("RotaryQuant");
}

extern "C" bool FastllmCudaDeepSeekV41RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                                  int posStep, bool inverse, int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, int quantMode, int quantDim,
                                                  int quantBlock) {
    return FastllmCudaDeepSeekV41RotaryQuantPosition(x, ropeDim, ropeBase, startPos, posStep, inverse,
        originalSeqLen, ropeFactor, betaFast, betaSlow, quantMode, quantDim, quantBlock, nullptr);
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
    const bool legacyIndexer = V41EnvOn("FASTLLM_DSV41_LEGACY_INDEXER");
    if (!legacyIndexer &&
        V41TryIndexerPrefill(q, weights, k, kMode, ratio, startPos, output)) return true;
    if (FastllmCudaGetThreadError()) return false;
    // BF16 mma 快速路径（SM80+）；FASTLLM_DSV41_LEGACY_INDEXER=1 退回下面的标量 kernel
    if (q.dataType == DataType::BFLOAT16 && (kQuant || k.dataType == DataType::BFLOAT16) && dim == kIdxDim &&
        !legacyIndexer) {
        if (seqlen == 1 && heads > 0 &&
            V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<false>, 256>()) {
            V41IndexerScoreDecodeMmaKernel<<<dim3((m + 63) / 64, bsz), 256>>>(
                (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData,
                kQuant ? nullptr : (const __nv_bfloat16*)k.cudaData, heads, m, ratio, startPos,
                (float*)output.cudaData, k8, kMode, kRowBytes, V41Fp4VecUnpack() ? 1 : 0);
            return V41CheckLaunch("IndexerScoreDecodeMma");
        }
        if (FastllmCudaGetThreadError()) return false;
        if (V41MmaKernelReady<V41IndexerScoreMmaKernel, kIdxMmaThreads>()) {
            dim3 mmaGrid((m + kIdxBJ - 1) / kIdxBJ, (seqlen + kIdxBT - 1) / kIdxBT, bsz);
            V41IndexerScoreMmaKernel<<<mmaGrid, kIdxMmaThreads>>>(
                (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData,
                kQuant ? nullptr : (const __nv_bfloat16*)k.cudaData,
                seqlen, heads, m, ratio, startPos, (float*)output.cudaData, k8, kMode, kRowBytes,
                V41Fp4VecUnpack() ? 1 : 0);
            return V41CheckLaunch("IndexerScoreMma");
        }
    }
    if (FastllmCudaGetThreadError()) return false;

    if (!legacyIndexer && V41IsSm75() && seqlen <= 8 && heads == 32) {
        const dim3 grid((m+7)/8, bsz*seqlen);
#define V41_SMALL_INDEX(Q,K) V41IndexerScoreSmallBatch<Q,K><<<grid,256,0,cudaStreamPerThread>>>( \
            static_cast<const Q *>(q.cudaData), static_cast<const float *>(weights.cudaData), \
            kQuant ? nullptr : static_cast<const K *>(k.cudaData), seqlen,m, \
            static_cast<float *>(output.cudaData),k8,kMode,kRowBytes)
        if (q.dataType == DataType::BFLOAT16) {
            if (kQuant || k.dataType == DataType::BFLOAT16) { V41_SMALL_INDEX(__nv_bfloat16,__nv_bfloat16); }
            else { V41_SMALL_INDEX(__nv_bfloat16,float); }
        } else {
            if (kQuant || k.dataType == DataType::BFLOAT16) { V41_SMALL_INDEX(float,__nv_bfloat16); }
            else { V41_SMALL_INDEX(float,float); }
        }
#undef V41_SMALL_INDEX
        return V41CheckLaunch("IndexerScoreSmallBatch");
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
    // Both producers and the only consumer use this thread's CUDA stream.
    // Delay pool reuse until that stream completes, without draining other
    // streams or stalling the CPU for every request in a decode batch.
    if (!FastllmCudaFreeAfterCurrentThreadStream(blockScore)) {
        FastllmCudaSyncCurrentThreadStream();
        FastllmCudaFree(blockScore);
    }
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
    if (seqlen > 1 || m <= width) {
        V41TopKKernel<false, false><<<bsz * seqlen, kSelThreads, (size_t)listCap * sizeof(int)>>>(
            (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
            ratio, startPos, listCap, (int32_t*)output.cudaData);
    } else {
        V41TopKKernel<false, true, 1024><<<bsz * seqlen, 1024, (size_t)listCap * sizeof(int)>>>(
            (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
            ratio, startPos, listCap, (int32_t*)output.cudaData);
    }
    return V41CheckLaunch("IndexerTopK");
}

// Hybrid workspaces require dense tensors on the selected CUDA device.
static bool Glm5HybridDense(const fastllm::Data &x) {
    if (!V41OnCuda(x) || x.multiDeviceData || x.dims.size() != x.strides.size()) return false;
    uint64_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || (x.dims[i] > 1 && x.strides[i] != stride)) return false;
        stride *= x.dims[i];
    }
    return true;
}

static bool Glm5HybridIndexerScore(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &keys, fastllm::Data &output,
        const fastllm::Data *position) {
    if (!Glm5HybridDense(query) || !Glm5HybridDense(keys) || !Glm5HybridDense(weights) ||
        &output == &query || &output == &keys || &output == &weights ||
        query.dataDeviceIds != keys.dataDeviceIds || query.dataDeviceIds != weights.dataDeviceIds ||
        query.dataType != DataType::BFLOAT16 || keys.dataType != DataType::BFLOAT16 ||
        weights.dataType != DataType::FLOAT32 || query.dims.size() != 4 ||
        query.dims[0] != 1 || query.dims[1] < 1 || query.dims[2] != 32 || query.dims[3] != 128 ||
        keys.dims.size() != 3 || keys.dims[0] != 1 || keys.dims[1] < 1 || keys.dims[2] != 128 ||
        weights.Count(0) != size_t(query.dims[1]) * 32 || !V41IsSm75()) return false;
    if (position && (query.dims[1] != 1 || !Glm5HybridDense(*position) ||
        position->dataType != DataType::INT32 || position->Count(0) != 1 ||
        position->dataDeviceIds != query.dataDeviceIds)) return false;
    const int rows = query.dims[1], candidates = keys.dims[1], heads = 32;
    const int batch = std::min(rows, std::max(1, std::min(32, (1 << 22) / candidates / heads)));
    const size_t scoreElements = size_t(batch) * heads * candidates;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, scoreElements * sizeof(float)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) { cudaStreamSynchronize(cudaStreamPerThread); FastllmCudaFree(ptr); } }
    } scratch{allocation};
    if (!V41PrepareOutput(output, DataType::FLOAT32, {1, rows, candidates})) return false;
    auto *dots = static_cast<float *>(allocation);
    for (int first = 0; first < rows; first += batch) {
        const int current = std::min(batch, rows - first);
        auto dotsKernel = position ? Glm5HybridIndexerDots<true> : Glm5HybridIndexerDots<false>;
        dotsKernel<<<dim3((candidates + 31) / 32, heads / 8, current), 256>>>(
            (const __nv_bfloat16 *)query.cudaData + size_t(first) * heads * 128,
            (const __nv_bfloat16 *)keys.cudaData, candidates, dots,
            position ? (const int32_t *)position->cudaData : nullptr);
        if (!V41CheckLaunch("GlmHybridIndexerDots")) return false;
        auto reduceKernel = position ? Glm5HybridIndexerReduce<true> : Glm5HybridIndexerReduce<false>;
        reduceKernel<<<dim3((candidates + 255) / 256, current), 256>>>(
            dots, (const float *)weights.cudaData + size_t(first) * heads, candidates, heads,
            (float *)output.cudaData + size_t(first) * candidates,
            position ? (const int32_t *)position->cudaData : nullptr);
        if (!V41CheckLaunch("GlmHybridIndexerReduce")) return false;
    }
    return true;
}

bool FastllmCudaGlm5NextHybridIndexerScore(const fastllm::Data &query,
        const fastllm::Data &weights, const fastllm::Data &keys, fastllm::Data &output) {
    return Glm5HybridIndexerScore(query, weights, keys, output, nullptr);
}

bool FastllmCudaHybridDsaScore(const fastllm::Data &query, const fastllm::Data &weights,
        const fastllm::Data &keys, const fastllm::Data &position, fastllm::Data &scores) {
    return Glm5HybridIndexerScore(query, weights, keys, scores, &position);
}

bool FastllmCudaHybridDsaSupported() { return V41IsSm75(); }

bool FastllmCudaGlm5NextHybridSparseAttention(const fastllm::Data &query,
        const fastllm::Data &latent, const fastllm::Data &indices,
        float scale, fastllm::Data &output) {
    if (!Glm5HybridDense(query) || !Glm5HybridDense(latent) || !Glm5HybridDense(indices) ||
        query.dataType != DataType::BFLOAT16 || latent.dataType != DataType::BFLOAT16 ||
        indices.dataType != DataType::INT32 || query.dims.size() != 4 ||
        query.dims[0] != 1 || query.dims[1] < 1 ||
        (query.dims[2] != 32 && query.dims[2] != 64) || query.dims[3] != 512 ||
        latent.dims.size() != 3 || latent.dims[0] != 1 || latent.dims[1] < 1 || latent.dims[2] != 512 ||
        indices.dims.size() != 3 || indices.dims[0] != 1 || indices.dims[1] != query.dims[1] ||
        indices.dims[2] < 32 || indices.dims[2] > 4096 || !std::isfinite(scale) || scale <= 0 ||
        &output == &query || &output == &latent || &output == &indices ||
        query.dataDeviceIds != latent.dataDeviceIds || query.dataDeviceIds != indices.dataDeviceIds ||
        !V41IsSm75()) return false;
    const int tokens = query.dims[1], heads = query.dims[2], count = indices.dims[2];
    if (tokens >= 128 && FastllmCudaSm75PrefillAttention(
            query, latent, indices, scale, output)) return true;
    // Bound scratch independently of prompt length. Keep the reference dot,
    // denominator and PV accumulation order, including every BF16 boundary.
    const int batch = std::min(tokens, 16);
    const size_t scoreElements = size_t(batch) * heads * count;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, (scoreElements * 3 + size_t(batch) * heads) * sizeof(float)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation};
    if (!V41PrepareOutput(output, DataType::BFLOAT16, query.dims)) return false;
    auto *scores = static_cast<float *>(allocation);
    auto *weights = reinterpret_cast<float2 *>(scores + scoreElements);
    auto *inverse = scores + scoreElements * 3;
    for (int first = 0; first < tokens; first += batch) {
        const int rows = std::min(batch, tokens - first);
        const auto *selected = (const int32_t *)indices.cudaData + size_t(first) * count;
        Glm5HybridAttentionScores<<<dim3((count + 31) / 32, heads / 8, rows), 256>>>(
            (const __nv_bfloat16 *)query.cudaData + size_t(first) * heads * 512,
            (const __nv_bfloat16 *)latent.cudaData, selected, count, latent.dims[1], heads, scores);
        if (!V41CheckLaunch("GlmHybridAttentionScores")) return false;
        Glm5HybridAttentionWeights<<<dim3(heads, rows), 256>>>(scores, selected,
            count, latent.dims[1], heads, scale, weights, inverse);
        if (!V41CheckLaunch("GlmHybridAttentionWeights")) return false;
        if (tokens >= 64) {
            Glm5HybridPrefillValues<<<dim3(heads / 32, 4, rows), 256>>>(
                (const __nv_bfloat16 *)latent.cudaData, selected, weights, inverse,
                count, latent.dims[1], heads, (__nv_bfloat16 *)output.cudaData + size_t(first) * heads * 512);
        } else if (rows >= 8) {
            Glm5HybridAttentionValues<4, 8><<<dim3(heads / 4, 4, rows), 128>>>(
                (const __nv_bfloat16 *)latent.cudaData, selected, weights, inverse,
                count, latent.dims[1], heads, (__nv_bfloat16 *)output.cudaData + size_t(first) * heads * 512);
        } else if (tokens == 1 && reinterpret_cast<uintptr_t>(latent.cudaData) % alignof(__nv_bfloat162) == 0) {
            // Two independent channels hide value-load latency. Each channel
            // retains the original candidate order and FP32 recurrence.
            Glm5HybridAttentionValues<1, 16, 2, 32><<<dim3(heads, 8, 1), 32>>>(
                (const __nv_bfloat16 *)latent.cudaData, selected, weights, inverse,
                count, latent.dims[1], heads, (__nv_bfloat16 *)output.cudaData);
        } else {
            Glm5HybridAttentionValues<1, 4><<<dim3(heads, 4, rows), 128>>>(
                (const __nv_bfloat16 *)latent.cudaData, selected, weights, inverse,
                count, latent.dims[1], heads, (__nv_bfloat16 *)output.cudaData + size_t(first) * heads * 512);
        }
        if (!V41CheckLaunch("GlmHybridAttentionValues")) return false;
    }
    return true;
}

template<int splits, bool MATERIALIZED = true>
static bool V41SparseAttentionBatchImpl(
        const std::vector<FastllmCudaV41AttentionRequest> &requests,
        fastllm::Data &attnSink, int windowSize, float softmaxScale) {
    const int device = FastllmCudaGetDevice();
    auto local = [device](const Data &d) {
        return V41OnCuda(d) && !d.multiDeviceData &&
               d.dataDeviceIds.size() == 1 && d.dataDeviceIds[0] == device;
    };
    // PP shares compressed KV across layers on different devices. Unlike
    // the executor fallback, this direct entry does not migrate its inputs.
    // Reject remote buffers before submitting descriptors, even with P2P.
    if (requests.empty() || requests.size() > 1024 || windowSize < 0 ||
        V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") || V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN") ||
        !local(attnSink) || attnSink.dataType != DataType::FLOAT32) return false;
    using PvShared = typename std::conditional<MATERIALIZED,
        V41MmaPvShared<splits>, V41MmaShared>::type;
    if constexpr (MATERIALIZED) {
        if (!V41MmaKernelReady<V41SparseAttentionScoresKernel<true>, kMmaThreads, sizeof(V41MmaShared)>()) return false;
    }
    if (!V41MmaKernelReady<V41SparseAttentionMmaKernel<splits, MATERIALIZED, true>, kMmaThreads, sizeof(PvShared)>()) return false;
    int heads = 0, maxTiles = 0;
    std::vector<V41AttentionDescriptor> descriptors;
    descriptors.reserve(requests.size());
    for (const auto &r : requests) {
        if (!r.q || !r.chunkKV || !r.output || r.startPos <= 0 ||
            !local(*r.q) || !local(*r.chunkKV) ||
            r.q->dataType != DataType::BFLOAT16 || r.chunkKV->dataType != DataType::BFLOAT16 ||
            r.q->dims.size() != 4 || r.q->dims[0] != 1 || r.q->dims[1] != 1 ||
            r.q->dims[2] <= 0 || r.q->dims[3] != kMmaDim ||
            r.chunkKV->dims.size() != 3 || r.chunkKV->dims[0] != 1 ||
            r.chunkKV->dims[1] != 1 || r.chunkKV->dims[2] != kMmaDim) return false;
        if (heads && heads != r.q->dims[2]) return false;
        heads = r.q->dims[2];
        if (attnSink.Count(0) < heads) return false;
        V41AttentionDescriptor d{};
        d.q = (const __nv_bfloat16*)r.q->cudaData;
        d.chunkKV = (const __nv_bfloat16*)r.chunkKV->cudaData;
        d.startPos = r.startPos;
        d.ringMode = d.cmpMode = 1;
        if (windowSize) {
            if (!r.ringKV || !local(*r.ringKV) || r.ringKV->dims.size() != 3 ||
                r.ringKV->dims[0] != 1 || r.ringKV->dims[1] != windowSize) return false;
            if (r.ringKV->dataType == DataType::INT8) {
                int block = 0;
                d.ringRowBytes = r.ringKV->dims[2];
                if (!V41ParseKvRowHost(kMmaDim, d.ringRowBytes, &d.ringMode, &block)) return false;
                d.ringKV8 = (const uint8_t*)r.ringKV->cudaData;
            } else if (r.ringKV->dataType == DataType::BFLOAT16 && r.ringKV->dims[2] == kMmaDim) {
                d.ringKV = (const __nv_bfloat16*)r.ringKV->cudaData;
            } else return false;
        }
        if (r.compressedKV && r.indices && !r.compressedKV->dims.empty() && r.compressedKV->Count(0)) {
            if (!local(*r.compressedKV) || !local(*r.indices) ||
                r.compressedKV->dims.size() != 3 || r.compressedKV->dims[0] != 1 ||
                r.indices->dataType != DataType::INT32 || r.indices->dims.size() != 3 ||
                r.indices->dims[0] != 1 || r.indices->dims[1] != 1) return false;
            d.cap = r.compressedKV->dims[1];
            d.topWidth = r.indices->dims[2];
            d.indices = (const int32_t*)r.indices->cudaData;
            if (r.compressedKV->dataType == DataType::INT8) {
                int block = 0;
                d.cmpRowBytes = r.compressedKV->dims[2];
                if (!V41ParseKvRowHost(kMmaDim, d.cmpRowBytes, &d.cmpMode, &block)) return false;
                d.compressedKV8 = (const uint8_t*)r.compressedKV->cudaData;
            } else if (r.compressedKV->dataType == DataType::BFLOAT16 && r.compressedKV->dims[2] == kMmaDim) {
                d.compressedKV = (const __nv_bfloat16*)r.compressedKV->cudaData;
            } else return false;
        }
        d.vecUnpack = V41Fp4VecUnpack() && d.ringRowBytes % 16 == 0 && d.cmpRowBytes % 16 == 0;
        maxTiles = std::max(maxTiles, (windowSize + d.topWidth + kMmaNC - 1) / kMmaNC);
        descriptors.push_back(d);
    }
    const size_t descriptorBytes = (descriptors.size() * sizeof(V41AttentionDescriptor) + 255) / 256 * 256;
    const size_t scoreStride = MATERIALIZED ? (size_t)heads * maxTiles * kMmaNC : 0;
    const size_t bytes = descriptorBytes + descriptors.size() * scoreStride * sizeof(float);
    if (!maxTiles || bytes > 128 * 1024 * 1024) return false;
    void *allocation = V41BorrowDecodeScratch(0, bytes);
    const bool borrowed = allocation != nullptr;
    if (!allocation && FastllmCudaTryMalloc(&allocation, bytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *ptr;
        bool owned;
        ~Scratch() { if (owned && !FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation, !borrowed};
    auto *deviceDescriptors = (V41AttentionDescriptor*)allocation;
    float *scores = (float*)((uint8_t*)allocation + descriptorBytes);
    for (size_t i = 0; i < requests.size(); ++i) {
        if (!V41PrepareOutput(*requests[i].output, DataType::BFLOAT16, requests[i].q->dims)) return false;
        descriptors[i].output = (__nv_bfloat16*)requests[i].output->cudaData;
        descriptors[i].scores = MATERIALIZED ? scores + i * scoreStride : nullptr;
    }
    if (!V41UploadDescriptors(deviceDescriptors, descriptors.data(),
            descriptors.size() * sizeof(V41AttentionDescriptor))) return false;
    const int headBlocks = (heads + kMmaHeads - 1) / kMmaHeads;
    if constexpr (MATERIALIZED) {
    V41SparseAttentionScoresKernel<true><<<dim3(requests.size(), headBlocks, maxTiles), kMmaThreads, sizeof(V41MmaShared), cudaStreamPerThread>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, heads, windowSize,
        0, 0, 0, nullptr, 0, 0, 0, 0, 0, deviceDescriptors);
    if (!V41CheckLaunch("BatchedSparseAttentionScores")) return false;
    }
    V41SparseAttentionMmaKernel<splits, MATERIALIZED, true><<<dim3(requests.size(), headBlocks, splits), kMmaThreads, sizeof(PvShared), cudaStreamPerThread>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (const float*)attnSink.cudaData, 1, heads, windowSize, 0, 0, 0, softmaxScale,
        nullptr, 0, 0, 0, 0, 0, nullptr, deviceDescriptors);
    return V41CheckLaunch("BatchedSparseAttentionValues");
}

extern "C" bool FastllmCudaDeepSeekV41SparseAttentionBatch(
        const std::vector<FastllmCudaV41AttentionRequest> &requests,
        fastllm::Data &attnSink, int windowSize, float softmaxScale) {
    // Small batches split value dimensions to expose parallelism. Once
    // enough requests are ready, fewer splits avoid repeating online
    // softmax, score loads and barriers. Each output keeps the same
    // 64-slot probability rounding and PV accumulation sequence.
    const size_t batch = requests.size();
    const size_t headBlocks = batch && requests[0].q && requests[0].q->dims.size() == 4
        ? (requests[0].q->dims[2] + kMmaHeads - 1) / kMmaHeads : 1;
    if (batch * headBlocks >= 128) {
        // With enough independent requests, compute QK and PV together and
        // read each selected KV tile once. Keep the materialized implementation
        // when the device cannot accommodate the fused shared-memory layout.
        if (V41SparseAttentionBatchImpl<1, false>(requests, attnSink, windowSize, softmaxScale)) return true;
        if (FastllmCudaGetThreadError()) return false;
        return V41SparseAttentionBatchImpl<1>(requests, attnSink, windowSize, softmaxScale);
    }
    if (batch * headBlocks >= 64)
        return V41SparseAttentionBatchImpl<2>(requests, attnSink, windowSize, softmaxScale);
    return V41SparseAttentionBatchImpl<8>(requests, attnSink, windowSize, softmaxScale);
}

// Opt-in entry for hybrid eager decode. Other modes, architectures, graph
// capture, batches and packed KV layouts retain their established path.
extern "C" bool FastllmCudaDeepSeekV41HybridSparseAttention(const fastllm::Data &q,
        const fastllm::Data &chunkKV, const fastllm::Data *ringKV,
        const fastllm::Data *compressedKV, const fastllm::Data *cmpIdx,
        fastllm::Data &attnSink, int windowSize, int startPos, float softmaxScale,
        fastllm::Data &output) {
    if (!V41OnCuda(q) || q.multiDeviceData || q.dims.size() != 4 ||
        q.dims[0] != 1 || q.dims[1] != 1 || q.dims[3] != 512 ||
        (q.dims[2] != 32 && q.dims[2] != 64) || q.dataType != DataType::BFLOAT16 ||
        !V41OnCuda(chunkKV) || chunkKV.multiDeviceData || chunkKV.dataType != DataType::BFLOAT16 ||
        chunkKV.dims != std::vector<int>({1, 1, 512}) || windowSize <= 0 || startPos < 0 ||
        FastllmCudaGraphIsCapturingFast() || V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") ||
        V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN") || !V41IsSm75()) return false;
    if (startPos > 0 && (!ringKV || !V41OnCuda(*ringKV) || ringKV->multiDeviceData ||
        ringKV->dataType != DataType::BFLOAT16 || ringKV->dims != std::vector<int>({1, windowSize, 512})))
        return false;
    const bool compressed = compressedKV && cmpIdx && compressedKV->Count(0) > 0;
    if (compressed && (!V41OnCuda(*compressedKV) || compressedKV->multiDeviceData ||
        compressedKV->dataType != DataType::BFLOAT16 || compressedKV->dims.size() != 3 ||
        compressedKV->dims[0] != 1 || compressedKV->dims[2] != 512 ||
        !V41OnCuda(*cmpIdx) || cmpIdx->multiDeviceData || cmpIdx->dataType != DataType::INT32 ||
        cmpIdx->dims.size() != 3 || cmpIdx->dims[0] != 1 || cmpIdx->dims[1] != 1)) return false;
    const int heads = q.dims[2], cap = compressed ? compressedKV->dims[1] : 0;
    const int count = std::min(windowSize, startPos + 1) + (compressed ? cmpIdx->dims[2] : 0);
    if (count < 32 || count > 4096 || attnSink.dataType != DataType::FLOAT32 ||
        attnSink.Count(0) < heads || attnSink.multiDeviceData) return false;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, size_t(heads) * count * sizeof(float)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation};
    attnSink.ToDevice(DataDevice::CUDA);
    if (!V41PrepareOutput(output, DataType::BFLOAT16, q.dims)) return false;
    const auto *kv = compressed ? (const __nv_bfloat16 *)compressedKV->cudaData : nullptr;
    const auto *indices = compressed ? (const int32_t *)cmpIdx->cudaData : nullptr;
    const auto *ring = startPos > 0 ? (const __nv_bfloat16 *)ringKV->cudaData : nullptr;
    const auto *chunk = (const __nv_bfloat16 *)chunkKV.cudaData;
    auto *scores = static_cast<float *>(allocation);
    V41Sm75DecodeScores<true><<<dim3((count + 3) / 4, heads / 4), 128>>>(
        (const __nv_bfloat16 *)q.cudaData, kv, indices, count, cap, heads, scores,
        ring, chunk, windowSize, startPos);
    if (!V41CheckLaunch("HybridDecodeScores")) return false;
    V41Sm75DecodeValues<true><<<dim3(heads, 4), 32>>>(kv, indices, scores,
        (const float *)attnSink.cudaData, count, cap, heads, softmaxScale,
        (__nv_bfloat16 *)output.cudaData, ring, chunk, windowSize, startPos);
    return V41CheckLaunch("HybridDecodeValues");
}

extern "C" bool FastllmCudaDeepSeekV41SparseAttention(const fastllm::Data &q, const fastllm::Data &chunkKV,
                                                      const fastllm::Data *ringKV, const fastllm::Data *compressedKV,
                                                      const fastllm::Data *cmpIdx, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, float softmaxScale,
                                                      fastllm::Data &output) {
    const bool referenceMath = V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH");
    if (referenceMath && FastllmCudaV41ReferenceAttention(q,chunkKV,ringKV,compressedKV,cmpIdx,attnSink,windowSize,startPos,softmaxScale,output)) return true;
    if (!V41OnCuda(q) || !V41OnCuda(chunkKV) || q.dims.size() != 4 || chunkKV.dims.size() != 3 ||
        q.dims[3] != 512 || q.dims[2] <= 0 || chunkKV.dataType != DataType::BFLOAT16 ||
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
    const bool legacyAttention = V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN");
    if (!legacyAttention && !referenceMath &&
        V41TrySm75PrefillAttention(q, chunkKV, hasRing ? ringKV : nullptr,
            hasCmp ? compressedKV : nullptr, hasCmp ? cmpIdx : nullptr, attnSink,
            windowSize, startPos, softmaxScale, ringMode, cmpMode, output)) return true;
    if (FastllmCudaGetThreadError()) return false;
    // Packed KV on SM75: unpack candidates once and reuse each QK dot across
    // all output dimensions. Optional scratch follows the current stream / graph.
    if (bsz == 1 && seqlen >= 1 && seqlen <= 8 && q.dataType == DataType::BFLOAT16 &&
        (ringFp8 || cmpFp8) && windowSize >= 0 && startPos >= 0 &&
        int64_t(startPos) + seqlen <= int64_t(std::numeric_limits<int>::max()) &&
        chunkKV.dims[0] == 1 && chunkKV.dims[1] == seqlen && chunkKV.dims[2] == dim &&
        (!hasRing || ringKV->dims[0] == 1) &&
        (!hasCmp || (compressedKV->dims[0] == 1 && cmpIdx->dims[0] == 1 && cmpIdx->dims[1] == seqlen)) &&
        attnSink.Count(0) >= heads && !legacyAttention && V41IsSm75()) {
        const int windowCount = int(std::min<int64_t>(windowSize, int64_t(startPos) + seqlen));
        const int64_t candidates = int64_t(windowCount) + topWidth;
        if (candidates >= 32 && candidates <= 4096 && heads <= 128) {
            const int count = int(candidates);
            const size_t rowsCount = size_t(seqlen) * count;
            // A separate coefficient pass amortizes well for long candidate lists
            // or verification batches; short decode keeps the single PV launch.
            const bool reuseCoefficients = count >= 512 || (seqlen >= 4 && count >= 128);
            const size_t scoreCount = rowsCount * heads;
            const size_t bytes = rowsCount * ((512 + heads) * sizeof(float) + sizeof(int32_t)) +
                (reuseCoefficients ? scoreCount * sizeof(float2) + size_t(seqlen) * heads * sizeof(float) : 0);
            void *allocation = nullptr;
            const auto status = FastllmCudaTryMalloc(&allocation, bytes);
            if (status == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
            if (status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
                struct Scratch {
                    void *ptr;
                    ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
                } scratch{allocation};
                auto *rows = static_cast<float *>(allocation);
                // Put float2 scratch immediately after the 512-wide rows so it
                // stays aligned even for odd head/candidate counts.
                auto *coefficients = reinterpret_cast<float2 *>(rows + rowsCount * 512);
                auto *scores = reuseCoefficients ? reinterpret_cast<float *>(coefficients + scoreCount)
                                                : rows + rowsCount * 512;
                auto *inverseSums = scores + scoreCount;
                auto *valid = reinterpret_cast<int32_t *>(inverseSums + (reuseCoefficients ? size_t(seqlen) * heads : 0));
                V41Sm75QuantizedAttention params{
                    static_cast<const __nv_bfloat16 *>(chunkKV.cudaData),
                    hasRing && !ringFp8 ? static_cast<const __nv_bfloat16 *>(ringKV->cudaData) : nullptr,
                    hasCmp && !cmpFp8 ? static_cast<const __nv_bfloat16 *>(compressedKV->cudaData) : nullptr,
                    ringFp8 ? static_cast<const uint8_t *>(ringKV->cudaData) : nullptr,
                    cmpFp8 ? static_cast<const uint8_t *>(compressedKV->cudaData) : nullptr,
                    hasCmp ? static_cast<const int32_t *>(cmpIdx->cudaData) : nullptr,
                    windowSize, startPos, cap, windowCount, topWidth, ringMode, ringRowBytes, cmpMode, cmpRowBytes};
                auto launch = [&](auto multiToken) {
                    constexpr bool multi = decltype(multiToken)::value;
                    V41Sm75QuantizedAttentionGather<multi><<<dim3(count, 1, seqlen), 128, 0, cudaStreamPerThread>>>(params, rows, valid);
                    if (!V41CheckLaunch("Sm75QuantizedAttentionGather")) return false;
                    const bool reuseQueries = heads >= 32 && count >= 128;
                    const bool reuseHeads = seqlen >= 4 && reuseQueries;
                    if (reuseQueries)
                        V41Sm75QuantizedAttentionScores<multi, 2><<<dim3((count + 7) / 8, (heads + 3) / 4, seqlen), 128, 0, cudaStreamPerThread>>>(
                            static_cast<const __nv_bfloat16 *>(q.cudaData), rows, valid, count, heads, scores);
                    else
                        V41Sm75QuantizedAttentionScores<multi><<<dim3((count + 3) / 4, (heads + 3) / 4, seqlen), 128, 0, cudaStreamPerThread>>>(
                            static_cast<const __nv_bfloat16 *>(q.cudaData), rows, valid, count, heads, scores);
                    if (!V41CheckLaunch("Sm75QuantizedAttentionScores")) return false;
                    if (reuseCoefficients) {
                        V41Sm75QuantizedAttentionCoefficients<<<dim3(heads, seqlen), 32, 0, cudaStreamPerThread>>>(
                            valid, scores, static_cast<const float *>(attnSink.cudaData), count, heads,
                            softmaxScale, coefficients, inverseSums);
                        if (!V41CheckLaunch("Sm75QuantizedAttentionCoefficients")) return false;
                        if (reuseHeads)
                            V41Sm75QuantizedAttentionValues<multi, true, 4><<<dim3((heads + 3) / 4, 8, seqlen), 64, 0, cudaStreamPerThread>>>(
                                rows, valid, scores, static_cast<const float *>(attnSink.cudaData), count, heads, softmaxScale,
                                static_cast<__nv_bfloat16 *>(output.cudaData), coefficients, inverseSums);
                        else if constexpr (!multi)
                            // Decode needs enough independent head/column CTAs.
                            // A larger candidate tile hides loads without merging heads.
                            V41Sm75QuantizedAttentionValues<false, true, 1, 16><<<dim3(heads, 8), 64, 0, cudaStreamPerThread>>>(
                                rows, valid, scores, static_cast<const float *>(attnSink.cudaData), count, heads, softmaxScale,
                                static_cast<__nv_bfloat16 *>(output.cudaData), coefficients, inverseSums);
                        else
                            V41Sm75QuantizedAttentionValues<multi, true><<<dim3(heads, 8, seqlen), 64, 0, cudaStreamPerThread>>>(
                                rows, valid, scores, static_cast<const float *>(attnSink.cudaData), count, heads, softmaxScale,
                                static_cast<__nv_bfloat16 *>(output.cudaData), coefficients, inverseSums);
                    } else {
                        V41Sm75QuantizedAttentionValues<multi><<<dim3(heads, 8, seqlen), 64, 0, cudaStreamPerThread>>>(
                            rows, valid, scores, static_cast<const float *>(attnSink.cudaData), count, heads, softmaxScale,
                            static_cast<__nv_bfloat16 *>(output.cudaData));
                    }
                    return V41CheckLaunch("Sm75QuantizedAttentionValues");
                };
                return seqlen == 1 ? launch(std::false_type{}) : launch(std::true_type{});
            }
            // A graph-pool capacity miss has submitted no work; use the scalar path.
        }
    }
    // TP ranks have 32 MLA heads; the unsharded draft has 64. Parallel candidate dots
    // avoid repeating QK in every value slice; PV keeps the scalar recurrence.
    // Each verification row carries its own already-causal candidate list.
    // Other shapes, KV formats and architectures retain their existing path.
    if (bsz == 1 && seqlen >= 1 && seqlen <= 8 && (heads == 32 || heads == 64) &&
        windowSize == 0 && startPos == 0 &&
        hasCmp && !cmpFp8 && topWidth >= 32 && topWidth <= 4096 &&
        cmpIdx->dims[0] == bsz && cmpIdx->dims[1] == seqlen &&
        compressedKV->dims[0] == bsz && attnSink.Count(0) >= heads &&
        q.dataType == DataType::BFLOAT16 && V41IsSm75()) {
        void *allocation = nullptr;
        const auto status = FastllmCudaTryMalloc(&allocation, (size_t)seqlen * heads * topWidth * sizeof(float));
        if (status == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
        if (status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            struct Scratch {
                void *ptr;
                ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
            } scratch{allocation};
            auto scores = static_cast<float*>(allocation);
            V41Sm75DecodeScores<<<dim3((topWidth + 3) / 4, heads / 4, seqlen), 128>>>(
                (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)compressedKV->cudaData,
                (const int32_t*)cmpIdx->cudaData, topWidth, cap, heads, scores);
            if (!V41CheckLaunch("Sm75DecodeScores")) return false;
            V41Sm75DecodeValues<<<dim3(heads, 16, seqlen), 32>>>(
                (const __nv_bfloat16*)compressedKV->cudaData, (const int32_t*)cmpIdx->cudaData, scores,
                (const float*)attnSink.cudaData, topWidth, cap, heads, softmaxScale, (__nv_bfloat16*)output.cudaData);
            return V41CheckLaunch("Sm75DecodeValues");
        }
    }


    // Reuse the gathered KV across two head groups without changing numerical
    // boundaries. Resource/image checks happen before submission; unsupported
    // builds and shapes retain the generic MMA path below.
    if (!legacyAttention && !referenceMath && q.dataType == DataType::BFLOAT16 &&
        seqlen >= 128 && heads >= 32 && V41IsSm<12, 0>() &&
        !V41EnvOn("FASTLLM_DSV41_DISABLE_SM120_PREFILL") &&
        V41MmaKernelReady<V41SparseAttentionSm120PrefillKernel, 512, sizeof(V41Sm120PrefillShared)>()) {
        V41SparseAttentionSm120PrefillKernel<<<dim3(bsz * seqlen, (heads + 31) / 32), 512, sizeof(V41Sm120PrefillShared)>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData,
            ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
        return V41CheckLaunch("SparseAttentionSm120Prefill");
    }
    if (FastllmCudaGetThreadError()) return false;

    // BF16 mma 快速路径（SM80+，head_dim 512，q 为 BF16）。
    // FASTLLM_DSV41_LEGACY_ATTN=1 可退回下面的 FP32 标量 kernel 做对比 / 排查。
    // Tail heads are zero-padded inside the tile; only real heads read/write
    // global Q, scores, sinks and output. In particular TP8 has eight heads.
    if (q.dataType == DataType::BFLOAT16 && dim == kMmaDim &&
        !legacyAttention) {
        const int tokens = bsz * seqlen;
        const int headBlocks = (heads + kMmaHeads - 1) / kMmaHeads;
        bool launched = false;
        auto launch = [&](auto splitTag) {
            constexpr int splits = decltype(splitTag)::value;
            using PvShared = V41MmaPvShared<splits>;
            const int candidateTiles = ((startPos == 0 ? std::min(seqlen, windowSize) : windowSize) + topWidth + 63) / 64;
            const uint64_t scratchBytes = (uint64_t)tokens * heads * candidateTiles * 64 * sizeof(float);
            if (tokens == 1 && candidateTiles >= 4 && scratchBytes <= 4 * 1024 * 1024 &&
                V41MmaKernelReady<V41SparseAttentionScoresKernel<>, kMmaThreads, sizeof(V41MmaShared)>() &&
                V41MmaKernelReady<V41SparseAttentionMmaKernel<splits, true>, kMmaThreads, sizeof(PvShared)>()) {
                void *allocation = nullptr;
                const auto allocated = FastllmCudaTryMalloc(&allocation, scratchBytes);
                if (allocated == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
                if (allocated == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
                    float *scores = (float*)allocation;
                    // Graph-aware pool ownership prevents another stream or graph
                    // from reusing this temporary before both kernels finish.
                    struct Scratch {
                        void *ptr;
                        ~Scratch() {
                            if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr);
                        }
                    } scratch{scores};
                    launched = true;
                    V41SparseAttentionScoresKernel<><<<dim3(tokens, headBlocks, candidateTiles), kMmaThreads, sizeof(V41MmaShared)>>>(
                        (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
                        hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
                        hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
                        ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
                        cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
                        hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
                        seqlen, heads, windowSize, cap, topWidth, startPos, scores,
                        ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
                    if (!V41CheckLaunch("SparseAttentionScores")) return false;
                    V41SparseAttentionMmaKernel<splits, true><<<dim3(tokens, headBlocks, splits), kMmaThreads, sizeof(PvShared)>>>(
                        (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
                        hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
                        hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
                        ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
                        cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
                        hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
                        (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
                        softmaxScale, (__nv_bfloat16*)output.cudaData,
                        ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0, scores);
                    return V41CheckLaunch("SparseAttentionMaterialized");
                }
                // Optional scratch did not fit (including a graph pool miss).
                // The fused MMA path needs no additional global workspace.
            }
            if (FastllmCudaGetThreadError()) return false;
            if (!V41MmaKernelReady<V41SparseAttentionMmaKernel<splits>, kMmaThreads,
                                   sizeof(V41MmaShared)>()) return false;
            launched = true;
            V41SparseAttentionMmaKernel<splits><<<dim3(tokens, headBlocks, splits), kMmaThreads, sizeof(V41MmaShared)>>>(
                (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
                hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
                hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
                ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
                cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
                hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
                (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
                softmaxScale, (__nv_bfloat16*)output.cudaData,
                ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
            return V41CheckLaunch("SparseAttentionMma");
        };
        const bool ok = tokens == 1 ? launch(std::integral_constant<int, 8>{})
                                    : launch(std::integral_constant<int, 1>{});
        // An unsupported image/resource configuration has submitted nothing.
        // Never retry a failed kernel launch: its execution state is unknown.
        if (launched) return ok;
    }
    if (FastllmCudaGetThreadError()) return false;

    if (q.dataType == DataType::BFLOAT16) {
        const bool sm75 = V41IsSm75();
        const bool smallDecode = bsz == 1 && seqlen == 1 && sm75;
        if (FastllmCudaGetThreadError()) return false;
        auto launchScalar = [&](auto headTag, auto padTag, auto valueTag, auto registerTag) {
            constexpr int headGroup = decltype(headTag)::value;
            dim3 grid(bsz * seqlen, (heads + headGroup - 1) / headGroup, decltype(valueTag)::value);
            V41SparseAttentionKernel<__nv_bfloat16, headGroup, decltype(padTag)::value,
                                     decltype(valueTag)::value, decltype(registerTag)::value><<<grid, headGroup * kAttnLanes>>>(
                (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
                hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
                hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
                ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
                cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
                hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
                (const float*)attnSink.cudaData, seqlen, heads, dim, windowSize, cap, topWidth, startPos,
                softmaxScale, (__nv_bfloat16*)output.cudaData, ringMode, ringRowBytes, cmpMode, cmpRowBytes);
        };
        // SM75 decode: 8 heads x 8 value slices distributes 64 heads over
        // 64 CTAs. Each slice retains the full QK dot and online-softmax order;
        // only independent output dimensions are partitioned (no reduction).
        if (smallDecode) launchScalar(std::integral_constant<int, 8>{}, std::true_type{}, std::integral_constant<int, 8>{}, std::false_type{});
        // SM75 multi-token fallback keeps the same heads, QK reduction and online softmax.
        // Padding separates the eight lanes' shared-memory banks without
        // splitting values or repeating QK. Keep the legacy path for comparison.
        else if (sm75 && seqlen > 1 && !legacyAttention) {
            // Packed Q trades extra shared loads/bit shifts for occupancy.
            // Small batches cannot amortize that work; retain their old path.
            if (seqlen >= 128)
                launchScalar(std::integral_constant<int, kAttnHeadsPerBlock>{}, std::true_type{}, std::integral_constant<int, 1>{}, std::true_type{});
            else
                launchScalar(std::integral_constant<int, kAttnHeadsPerBlock>{}, std::true_type{}, std::integral_constant<int, 1>{}, std::false_type{});
        } else launchScalar(std::integral_constant<int, kAttnHeadsPerBlock>{}, std::false_type{}, std::integral_constant<int, 1>{}, std::false_type{});
    } else {
        dim3 grid(bsz * seqlen, (heads + kAttnHeadsPerBlock - 1) / kAttnHeadsPerBlock);
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

static bool V41WindowStore(const fastllm::Data &chunk, fastllm::Data &ring, int startPos,
                           int windowSize, const int *devicePosition, int rowCount) {
    if (!V41OnCuda(chunk) || chunk.dims.size() != 3 || windowSize <= 0) {
        return false;
    }
    int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
    if (rowCount == -1) rowCount = seqlen;
    if (rowCount < 0 || rowCount > seqlen) return false;
    if (rowCount == 0) return true;
    if (ring.dims.size() != 3 || ring.dims[0] != bsz || ring.dims[1] != windowSize || ring.dims[2] != dim ||
        ring.dataType != chunk.dataType || !V41OnCuda(ring)) {
        if (!V41PrepareOutput(ring, chunk.dataType, {bsz, windowSize, dim})) {
            return false;
        }
        cudaMemset(ring.cudaData, 0, (size_t)ring.Count(0) * ring.unitSize / ring.unitSizeDiv);
    }
    int rowBytes = dim * chunk.unitSize;
    int firstRow = std::max(0, rowCount - windowSize);
    dim3 grid(rowCount - firstRow, bsz);
    V41WindowStoreKernel<<<grid, 256>>>((const uint8_t*)chunk.cudaData, (uint8_t*)ring.cudaData, seqlen, rowBytes,
                                        startPos, windowSize, firstRow, devicePosition);
    return V41CheckLaunch("WindowStore");
}

extern "C" bool FastllmCudaDeepSeekV41WindowStorePosition(const fastllm::Data &chunk, fastllm::Data &ring,
                                                        int startPos, int windowSize, const int *devicePosition) {
    return V41WindowStore(chunk, ring, startPos, windowSize, devicePosition, -1);
}

extern "C" bool FastllmCudaDeepSeekV41WindowStore(const fastllm::Data &chunk, fastllm::Data &ring,
                                                int startPos, int windowSize) {
    return V41WindowStore(chunk, ring, startPos, windowSize, nullptr, -1);
}

extern "C" bool FastllmCudaDeepSeekV41WindowStorePrefix(const fastllm::Data &chunk, fastllm::Data &ring,
                                                      int startPos, int windowSize, int rowCount) {
    return V41WindowStore(chunk, ring, startPos, windowSize, nullptr, rowCount);
}

extern "C" bool FastllmCudaDeepSeekV41QuantizeActivation(const fastllm::Data &input, fastllm::Data &output) {
    if (!V41OnCuda(input) || input.dims.empty() || input.dims.back() % 32 || !V41IsFloatType(input.dataType)) return false;
    output.Allocate();
    uint64_t count = input.Count(0);
    if (!count) return true;
    unsigned int blocks = (unsigned int)((count + 255) / 256);
    if (input.dataType == fastllm::DataType::BFLOAT16 && count >= 256 * 1024 &&
        reinterpret_cast<uintptr_t>(input.cudaData) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(output.cudaData) % 16 == 0) {
        V41QuantizeActivationVectorKernel<8><<<(count + 2047) / 2048, 256>>>(
            (const __nv_bfloat16*)input.cudaData, (__nv_bfloat16*)output.cudaData, count);
    } else if (input.dataType == fastllm::DataType::BFLOAT16)
        V41QuantizeActivationKernel<<<blocks, 256>>>((const __nv_bfloat16*)input.cudaData, (__nv_bfloat16*)output.cudaData, count);
    else if (input.dataType == fastllm::DataType::FLOAT16)
        V41QuantizeActivationKernel<<<blocks, 256>>>((const half*)input.cudaData, (half*)output.cudaData, count);
    else
        V41QuantizeActivationKernel<<<blocks, 256>>>((const float*)input.cudaData, (float*)output.cudaData, count);
    return V41CheckLaunch("QuantizeActivation");
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

extern "C" bool FastllmCudaDeepSeekV41LinearBlock32(const fastllm::Data &input,
                                                    fastllm::Data &weight, fastllm::Data &output) {
    if (!V41OnCuda(input) || input.dataType != DataType::BFLOAT16 || input.dims.empty() ||
        input.multiDeviceData || weight.multiDeviceData || weight.dims.size() != 2 ||
        input.dims.back() != weight.dims[1] || weight.dims[1] <= 0 || weight.dims[1] % 32 ||
        weight.dims[0] <= 0 || !V41IsFloatType(weight.dataType)) return false;
    const int k = weight.dims[1], n = weight.dims[0];
    const uint64_t tokens = input.Count(0) / k;
    if (tokens == 0 || tokens > 65535) return false;
    weight.ToDevice(DataDevice::CUDA);
    auto dims = input.dims;
    dims.back() = n;
    if (!V41PrepareOutput(output, DataType::BFLOAT16, dims)) return false;
    const dim3 grid((n + 3) / 4, tokens);
    const auto *x = (const __nv_bfloat16*)input.cudaData;
    auto *y = (__nv_bfloat16*)output.cudaData;
    if (weight.dataType == DataType::FLOAT16)
        V41LinearBlock32Kernel<<<grid, 128>>>(x, (const half*)weight.cudaData, y, n, k);
    else if (weight.dataType == DataType::BFLOAT16)
        V41LinearBlock32Kernel<<<grid, 128>>>(x, (const __nv_bfloat16*)weight.cudaData, y, n, k);
    else
        V41LinearBlock32Kernel<<<grid, 128>>>(x, (const float*)weight.cudaData, y, n, k);
    return V41CheckLaunch("LinearBlock32");
}

extern "C" bool FastllmCudaDeepSeekV41HcPost(const fastllm::Data &input, const fastllm::Data &residual,
    const fastllm::Data &post, const fastllm::Data &comb, fastllm::Data &output) {
    if (!V41OnCuda(input) || !V41OnCuda(residual) || !V41OnCuda(post) || !V41OnCuda(comb) ||
        input.dataType != residual.dataType || !V41IsFloatType(input.dataType) || residual.dims.size() != 4 ||
        post.dataType != DataType::FLOAT32 || comb.dataType != DataType::FLOAT32 ||
        &output == &input) return false;
    const int hc = residual.dims[2], dim = residual.dims[3];
    const uint64_t count = residual.Count(0);
    if (hc <= 0 || hc > 4 || dim <= 0 || input.Count(0) * hc != count || post.Count(0) * dim != count ||
        comb.Count(0) != post.Count(0) * hc) return false;
    if (!V41PrepareOutput(output, input.dataType, residual.dims)) return false;
    const int blocks = (count / hc + 255) / 256;
#define V41_HCPOST_ONE(T, H) V41HcPostKernel<T, H><<<blocks, 256>>>((const T*)input.cudaData, (const T*)residual.cudaData, \
    (const float*)post.cudaData, (const float*)comb.cudaData, (T*)output.cudaData, count, dim)
#define V41_HCPOST(T) switch (hc) { \
    case 1: V41_HCPOST_ONE(T, 1); break; \
    case 2: V41_HCPOST_ONE(T, 2); break; \
    case 3: V41_HCPOST_ONE(T, 3); break; \
    case 4: V41_HCPOST_ONE(T, 4); break; \
    }
    if (input.dataType == DataType::BFLOAT16) { V41_HCPOST(__nv_bfloat16); }
    else if (input.dataType == DataType::FLOAT16) { V41_HCPOST(half); }
    else { V41_HCPOST(float); }
#undef V41_HCPOST_ONE
#undef V41_HCPOST
    return V41CheckLaunch("HcPost");
}

extern "C" bool FastllmCudaDeepSeekV41MoeHcPost(const fastllm::Data &input, const fastllm::Data &shared, const fastllm::Data &residual,
    const fastllm::Data &post, const fastllm::Data &comb, fastllm::Data &output) {
    if (!V41OnCuda(input) || !V41OnCuda(residual) || !V41OnCuda(post) || !V41OnCuda(comb) ||
        (input.dataType != DataType::FLOAT32 && input.dataType != DataType::BFLOAT16) ||
        residual.dataType != DataType::BFLOAT16 ||
        shared.dataType != DataType::BFLOAT16 || !V41OnCuda(shared) ||
        shared.Count(0) != input.Count(0) || residual.dims.size() != 4 ||
        post.dataType != DataType::FLOAT32 || comb.dataType != DataType::FLOAT32 ||
        &output == &input || &output == &shared) return false;
    const int hc = residual.dims[2], dim = residual.dims[3];
    const uint64_t count = residual.Count(0);
    if (hc <= 0 || hc > 4 || dim <= 0 || input.Count(0) * hc != count || post.Count(0) * dim != count ||
        comb.Count(0) != post.Count(0) * hc) return false;
    if (!V41PrepareOutput(output, residual.dataType, residual.dims)) return false;
    const int blocks = (count / hc + 255) / 256;
#define V41_MOE_HCPOST(T, H) V41MoeHcPostKernel<T, H><<<blocks, 256>>>( \
    (const T*)input.cudaData, (const __nv_bfloat16*)shared.cudaData, \
    (const __nv_bfloat16*)residual.cudaData, (const float*)post.cudaData, \
    (const float*)comb.cudaData, (__nv_bfloat16*)output.cudaData, count, dim)
#define V41_MOE_HCPOST_TYPE(T) switch (hc) { \
    case 1: V41_MOE_HCPOST(T, 1); break; \
    case 2: V41_MOE_HCPOST(T, 2); break; \
    case 3: V41_MOE_HCPOST(T, 3); break; \
    case 4: V41_MOE_HCPOST(T, 4); break; \
    }
    if (input.dataType == DataType::BFLOAT16) { V41_MOE_HCPOST_TYPE(__nv_bfloat16); }
    else { V41_MOE_HCPOST_TYPE(float); }
#undef V41_MOE_HCPOST_TYPE
#undef V41_MOE_HCPOST
    return V41CheckLaunch("MoeHcPost");
}

extern "C" bool FastllmCudaV41DecodeTransformsBatch(
        const std::vector<FastllmCudaV41DecodeTransform> &items,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, int windowSize) {
    if (items.empty() || items.size() > 65535 || ropeDim <= 0 || ropeDim > 128 ||
        (ropeDim & 1) || windowSize <= 0 || V41LegacyRotary() ||
        V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH")) return false;
    std::vector<V41DecodeTransformDescriptor> descriptors;
    int maxRows = 0;
    for (const auto &item : items) {
        if (!item.input || !V41OnCuda(*item.input) || item.input->dataType != DataType::BFLOAT16 ||
            (item.input->dims.size() != 3 && item.input->dims.size() != 4) ||
            item.input->dims[0] != 1 || item.input->dims[1] != 1 ||
            (item.input->dims.back() != 512 && item.input->dims.back() != 128) ||
            item.startPos < 0 || item.quantMode < 0 || item.quantMode > 3 ||
            (item.quantBlock != 16 && item.quantBlock != 32)) return false;
        const int dim = item.input->dims.back();
        const int rows = item.input->Count(0) / dim;
        if (!rows) return false;
        if (item.window && (dim != 512 || rows != 1 || !V41OnCuda(*item.window) ||
            item.window->dataType != DataType::BFLOAT16 ||
            item.window->dims != std::vector<int>({1, windowSize, 512}))) return false;
        descriptors.push_back({(__nv_bfloat16 *)item.input->cudaData,
            item.window ? (__nv_bfloat16 *)item.window->cudaData : nullptr,
            rows, dim, item.startPos, item.quantMode, item.inverse ? 1 : 0, item.quantBlock});
        maxRows = std::max(maxRows, (rows + 512 / dim - 1) / (512 / dim));
    }
    const size_t bytes = descriptors.size() * sizeof(V41DecodeTransformDescriptor);
    void *storage = V41BorrowDecodeScratch(1, bytes);
    const bool borrowed = storage != nullptr;
    if (!storage) storage = FastllmCudaMalloc(bytes);
    auto *device = (V41DecodeTransformDescriptor *)storage;
    if (!device) return false;
    if (!V41UploadDescriptors(device, descriptors.data(), bytes)) {
        if (!borrowed) FastllmCudaFree(device);
        return false;
    }
    auto rope = V41BuildRope(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    V41DecodeTransformsKernel<<<dim3(maxRows, items.size()), 512, 0, cudaStreamPerThread>>>(
        device, rope, ropeDim, windowSize);
    const bool ok = V41CheckLaunch("DecodeTransformsBatch");
    if (!borrowed && !FastllmCudaFreeAfterCurrentThreadStream(device)) {
        cudaStreamSynchronize(cudaStreamPerThread);
        FastllmCudaFree(device);
    }
    if (!ok) throw std::runtime_error("DeepSeekV41 decode batch kernel launch failed");
    return true;
}


extern "C" bool FastllmCudaV41DecodeIndexerBatch(
        const std::vector<FastllmCudaV41DecodeIndexTask> &items,
        int ratio, int topK, int blockSize, int candidateTopK,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, bool quantizeQuery) {
    if (items.empty() || items.size() > 65535 || ratio <= 0 || topK <= 0 || blockSize <= 0 || candidateTopK <= 0 ||
        V41LegacyTopK() || V41LegacyRotary() || V41EnvOn("FASTLLM_DSV41_LEGACY_INDEXER") ||
        V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") ||
        !V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true>, 256>()) return false;
    std::vector<V41DecodeIndexDescriptor> descriptors;
    std::vector<FastllmCudaV41DecodeTransform> transforms;
    size_t scoreCount = 0, blockCount = 0;
    int maxCount = 0, maxBlocks = 0;
    bool candidatesWritten = false;
    for (const auto &item : items) {
        if (!item.query || !item.weights || !item.keys || !item.topk ||
            !V41OnCuda(*item.query) || !V41OnCuda(*item.weights) || !V41OnCuda(*item.keys) ||
            item.query->dataType != DataType::BFLOAT16 || item.keys->dataType != DataType::BFLOAT16 ||
            item.weights->dataType != DataType::FLOAT32 || item.query->dims.size() != 4 ||
            item.query->dims[0] != 1 || item.query->dims[1] != 1 || item.query->dims[3] != 128 ||
            item.keys->dims.size() != 3 || item.keys->dims[0] != 1 || item.keys->dims[2] != 128 ||
            item.keys->dims[1] <= 0 || item.startPos < 0 || ropeDim <= 0 || ropeDim > 128 || (ropeDim & 1)) return false;
        const int m = item.keys->dims[1], heads = item.query->dims[2], blocks = (m + blockSize - 1) / blockSize;
        if (heads <= 0 || item.weights->Count(0) != size_t(heads)) return false;
        if ((item.useCandidates || item.writeCandidates) && (!item.candidates ||
            !V41OnCuda(*item.candidates) || item.candidates->dataType != DataType::INT8 ||
            item.candidates->dims != std::vector<int>({1, 1, blocks}))) return false;
        if (!V41PrepareOutput(*item.topk, DataType::INT32, {1, 1, std::min(topK, m)})) return false;
        descriptors.push_back({(const __nv_bfloat16 *)item.query->cudaData,
            (const __nv_bfloat16 *)item.keys->cudaData, (const float *)item.weights->cudaData,
            nullptr, nullptr, (item.useCandidates || item.writeCandidates) ? (uint8_t *)item.candidates->cudaData : nullptr,
            (int32_t *)item.topk->cudaData, heads, m, std::min(topK, m), item.startPos, blocks,
            item.writeCandidates, item.useCandidates});
        transforms.push_back({item.query, nullptr, item.startPos, quantizeQuery ? 2 : 0, false});
        scoreCount += m; blockCount += blocks;
        maxCount = std::max(maxCount, m); maxBlocks = std::max(maxBlocks, blocks);
        candidatesWritten |= item.writeCandidates;
    }
    const size_t descriptorBytes = descriptors.size() * sizeof(V41DecodeIndexDescriptor);
    if ((scoreCount + blockCount) * sizeof(float) > 128ULL * 1024 * 1024) return false;
    const size_t bytes = descriptorBytes + (scoreCount + blockCount) * sizeof(float);
    auto *storage = (uint8_t *)V41BorrowDecodeScratch(2, bytes);
    const bool borrowed = storage != nullptr;
    if (!storage) storage = (uint8_t *)FastllmCudaMalloc(bytes);
    if (!storage) return false;
    auto *scores = (float *)(storage + descriptorBytes);
    auto *blocks = scores + scoreCount;
    for (auto &item : descriptors) {
        item.scores = scores; scores += item.count;
        item.blockScores = blocks; blocks += item.blocks;
    }
    if (!V41UploadDescriptors(storage, descriptors.data(), descriptorBytes)) {
        if (!borrowed && !FastllmCudaFreeAfterCurrentThreadStream(storage)) {
            cudaStreamSynchronize(cudaStreamPerThread);
            FastllmCudaFree(storage);
        }
        return false;
    }
    if (!FastllmCudaV41DecodeTransformsBatch(transforms, ropeDim, ropeBase, originalSeqLen,
                                           ropeFactor, betaFast, betaSlow, 1)) {
        if (!borrowed && !FastllmCudaFreeAfterCurrentThreadStream(storage)) {
            cudaStreamSynchronize(cudaStreamPerThread);
            FastllmCudaFree(storage);
        }
        return false;
    }
    auto *device = (const V41DecodeIndexDescriptor *)storage;
    V41IndexerScoreDecodeMmaKernel<true><<<dim3((maxCount + 63) / 64, items.size()), 256, 0, cudaStreamPerThread>>>(
        nullptr, nullptr, nullptr, 0, 0, ratio, 0, nullptr, nullptr, 0, 0, 0, device);
    if (candidatesWritten) {
        V41BlockScoreKernel<true><<<items.size(), 256, 0, cudaStreamPerThread>>>(
            nullptr, 1, 0, blockSize, 0, ratio, 0, nullptr, device);
        V41CandidateSelectKernel<true><<<items.size(), kSelThreads, 0, cudaStreamPerThread>>>(
            nullptr, 0, candidateTopK, nullptr, device);
    }
    int shift = -1;
    for (int i = 0; i < 31; ++i) if ((1 << i) == blockSize) { shift = i; break; }
    const int listCap = shift >= 0 ? std::min(maxBlocks, 4096) : 0;
    V41TopKKernel<true, true, 1024><<<items.size(), 1024, (size_t)listCap * sizeof(int), cudaStreamPerThread>>>(
        nullptr, nullptr, 1, 0, 0, blockSize, shift, 0, ratio, 0, listCap, nullptr, device);
    const bool ok = V41CheckLaunch("DecodeIndexerBatch");
    if (!borrowed && !FastllmCudaFreeAfterCurrentThreadStream(storage)) {
        cudaStreamSynchronize(cudaStreamPerThread);
        FastllmCudaFree(storage);
    }
    if (!ok) throw std::runtime_error("DeepSeekV41 decode batch kernel launch failed");
    return true;
}

namespace {
constexpr int kFullDecodeTransforms = 5;

struct V41FullDecodePlan {
    FastllmV41FullDecodeConfig config;
    FastllmV41FullDecodeBuffers buffers;
    int device = 0, tiles = 0, splits = 8, listCap = 0, blockShift = -1;
    bool fused = false, pending = false;
    V41RopeTable rope;
    FastllmV41FullDecodeRequest *host = nullptr, *requests = nullptr;
    V41DecodeTransformDescriptor *transforms = nullptr;
    V41DecodeIndexDescriptor *index = nullptr;
    V41AttentionDescriptor *attention = nullptr;
    float *indexScores = nullptr, *blockScores = nullptr, *attentionScores = nullptr;
    cudaEvent_t copied = nullptr;
    ~V41FullDecodePlan() {
        int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
        // Graph ownership must be released before its operand storage.
        cudaStreamSynchronize(cudaStreamPerThread);
        if (pending) cudaEventSynchronize(copied);
        if (copied) cudaEventDestroy(copied);
        if (host) cudaFreeHost(host);
        if (requests && !buffers.requests) cudaFree(requests);
        for (void *p : {static_cast<void *>(transforms),
                       static_cast<void *>(index), static_cast<void *>(attention),
                       static_cast<void *>(indexScores), static_cast<void *>(blockScores),
                       static_cast<void *>(attentionScores)}) if (p) cudaFree(p);
        cudaSetDevice(old);
    }
};

// Descriptor arrays are initialized on the GPU. Request addresses and lengths
// may change on every replay without changing any graph node or launch shape.
__device__ __forceinline__ void V41FullDecodeSetupRow(
        int i, FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41DecodeTransformDescriptor *transforms,
        V41DecodeIndexDescriptor *index, V41AttentionDescriptor *attention,
        float *indexScores, float *blockScores, float *attentionScores, int tiles) {
    const auto r = requests[i];
    const int end = c.verifying ? requests[0].position + c.batch : r.position + 1;
    const int count = c.ratio ? end / c.ratio : 0;
    const int blocks = c.ratio ? (count + c.blockSize - 1) / c.blockSize : 0;
    const int maxBlocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
    auto *q = static_cast<__nv_bfloat16 *>(b.q) + (size_t)i * c.heads * 512;
    auto *kv = static_cast<__nv_bfloat16 *>(b.kv) + (size_t)i * 512;
    auto *out = static_cast<__nv_bfloat16 *>(b.output) + (size_t)i * c.heads * 512;
    auto *tk = c.ratio ? static_cast<int32_t *>(b.topk) + (size_t)i * c.topK : nullptr;
    auto *cand = c.ratio ? static_cast<uint8_t *>(b.candidates) + (size_t)i * maxBlocks : nullptr;
    auto *t = transforms + i * kFullDecodeTransforms;
    t[0] = {out, nullptr, c.heads, 512, r.position, 1, 1, 32};
    t[1] = {kv, static_cast<__nv_bfloat16 *>(r.windowKV), 1, 512, r.position, 0, 0, 32};
    if (c.kvSource) {
        const int blockPosition = (r.position / c.ratio) * c.ratio;
        t[2] = {static_cast<__nv_bfloat16 *>(b.latents) + (size_t)i * 512,
                nullptr, 1, 512, blockPosition, 3, 0, 16};
        if (c.indexSource)
            t[3] = {static_cast<__nv_bfloat16 *>(b.newIndexKeys) + (size_t)i * 128,
                    nullptr, 1, 128, blockPosition, 2, 0, 32};
    }
    if (c.indexSource) {
        auto *iq = static_cast<__nv_bfloat16 *>(b.indexQ) + (size_t)i * c.indexHeads * 128;
        t[4] = {iq, nullptr, c.indexHeads, 128, r.position, 2, 0, 32};
        index[i] = {iq, static_cast<const __nv_bfloat16 *>(r.indexK),
            static_cast<const float *>(b.indexWeights) + (size_t)i * c.indexHeads,
            indexScores + (size_t)i * c.maxKeys, blockScores + (size_t)i * maxBlocks,
            cand, tk, c.indexHeads, count, min(count, c.topK), r.position, blocks,
            c.writeCandidates, c.useCandidates};
    }
    V41AttentionDescriptor a{};
    a.q = q;
    a.chunkKV = c.verifying ? static_cast<const __nv_bfloat16 *>(b.kv) : kv;
    a.chunkOffset = c.verifying ? i : 0;
    a.ringKV = static_cast<const __nv_bfloat16 *>(r.windowKV);
    a.compressedKV = static_cast<const __nv_bfloat16 *>(r.compressedKV);
    a.indices = tk; a.output = out;
    a.scores = attentionScores + (size_t)i * c.heads * tiles * kMmaNC;
    a.cap = count; a.topWidth = min(count, c.topK); a.startPos = r.position;
    attention[i] = a;
}

// Q/KV transforms read request and buffer arguments directly. They do not
// depend on the descriptors initialized by the first CTA; later kernels see
// both descriptor writes and transformed rows through ordinary stream order.
__global__ void V41FullDecodePrepareKernel(
        FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41DecodeTransformDescriptor *transforms,
        V41DecodeIndexDescriptor *index, V41AttentionDescriptor *attention,
        float *indexScores, float *blockScores, float *attentionScores, int tiles,
        V41RopeTable rope) {
    const int token = blockIdx.y;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        V41FullDecodeSetupRow(token, c, b, requests, transforms, index, attention,
                             indexScores, blockScores, attentionScores, tiles);
    const bool kv = blockIdx.x == c.heads;
    auto *input = kv ? static_cast<__nv_bfloat16 *>(b.kv) + (size_t)token * 512 :
        static_cast<__nv_bfloat16 *>(b.q) + ((size_t)token * c.heads + blockIdx.x) * 512;
    const V41DecodeTransformDescriptor item{
        input, nullptr, 1, 512, requests[token].position, kv ? 1 : 0, 0, 32};
    V41DecodeTransformRow(item, 0, threadIdx.x, rope, c.ropeDim, c.window);
}

// Select a subset of the transform descriptors without uploading or reallocating.
// The arithmetic is shared with the ordinary decode transform kernel below.
__global__ void V41FullDecodeGather(const FastllmV41FullDecodeRequest *requests,
        const float *raw, const float *rawScores, float *joined, float *scores, int ratio) {
    const int row = blockIdx.x, d = threadIdx.x;
    const auto r = requests[row];
    const float value = raw[(size_t)row * 512 + d];
    if (ratio == 1) { joined[(size_t)row * 512 + d] = value; return; }
    const float score = rawScores[(size_t)row * 512 + d];
    const bool ready = (r.position + 1) % ratio == 0;
    joined[(size_t)row * 1024 + d] = ready ? static_cast<const float *>(r.rawTailKV)[d] : value;
    joined[(size_t)row * 1024 + 512 + d] = value;
    scores[(size_t)row * 1024 + d] = ready ? static_cast<const float *>(r.rawTailScore)[d] : score;
    scores[(size_t)row * 1024 + 512 + d] = score;
    if (!ready) {
        static_cast<float *>(r.rawTailKV)[d] = value;
        static_cast<float *>(r.rawTailScore)[d] = score;
    }
}

__global__ void V41FullDecodeAppend(const FastllmV41FullDecodeRequest *requests,
        const __nv_bfloat16 *latent, const __nv_bfloat16 *keys, int ratio, bool indexSource) {
    const auto r = requests[blockIdx.x];
    if ((r.position + 1) % ratio) return;
    const int d = threadIdx.x, row = r.position / ratio;
    static_cast<__nv_bfloat16 *>(r.compressedKV)[(size_t)row * 512 + d] =
        latent[(size_t)blockIdx.x * 512 + d];
    if (indexSource && d < 128)
        static_cast<__nv_bfloat16 *>(r.indexK)[(size_t)row * 128 + d] =
            keys[(size_t)blockIdx.x * 128 + d];
}

// Verification never overwrites the committed ring or compressor tail.
// Save a contiguous [old tail, current chunk] stream for any accepted prefix.
__global__ void V41FullVerifyGather(FastllmV41FullDecodeConfig c,
        FastllmV41FullDecodeBuffers b, const FastllmV41FullDecodeRequest *requests) {
    const int row = blockIdx.x, d = threadIdx.x;
    const auto r = requests[row];
    const int tail = requests[0].position % c.ratio;
    const float value = static_cast<const float *>(b.rawKV)[(size_t)row * 512 + d];
    static_cast<float *>(b.savedRawKV)[(size_t)(tail + row) * 512 + d] = value;
    if (c.ratio == 1) {
        static_cast<float *>(b.joined)[(size_t)row * 512 + d] = value;
        return;
    }
    const float score = static_cast<const float *>(b.rawScore)[(size_t)row * 512 + d];
    const bool ready = (r.position + 1) % c.ratio == 0;
    const float previous = row ? static_cast<const float *>(b.rawKV)[(size_t)(row - 1) * 512 + d]
                               : (tail ? static_cast<const float *>(r.rawTailKV)[d] : value);
    const float previousScore = row ? static_cast<const float *>(b.rawScore)[(size_t)(row - 1) * 512 + d]
                                    : (tail ? static_cast<const float *>(r.rawTailScore)[d] : score);
    static_cast<float *>(b.joined)[(size_t)row * 1024 + d] = ready ? previous : value;
    static_cast<float *>(b.joined)[(size_t)row * 1024 + 512 + d] = value;
    static_cast<float *>(b.scores)[(size_t)row * 1024 + d] = ready ? previousScore : score;
    static_cast<float *>(b.scores)[(size_t)row * 1024 + 512 + d] = score;
    static_cast<float *>(b.savedRawScore)[(size_t)(tail + row) * 512 + d] = score;
    if (row == 0 && tail) {
        static_cast<float *>(b.savedRawKV)[d] = previous;
        static_cast<float *>(b.savedRawScore)[d] = previousScore;
    }
}

__global__ void V41FullVerifySaveKV(const __nv_bfloat16 *kv, __nv_bfloat16 *saved) {
    const size_t i = (size_t)blockIdx.x * 512 + threadIdx.x;
    saved[i] = kv[i];
}

template<int SPLITS, bool MATERIALIZED>
void V41FullDecodeAttention(V41FullDecodePlan &p) {
    const auto &c = p.config;
    const int headBlocks = (c.heads + kMmaHeads - 1) / kMmaHeads;
    using Shared = typename std::conditional<MATERIALIZED, V41MmaPvShared<SPLITS>, V41MmaShared>::type;
    if constexpr (MATERIALIZED)
        V41SparseAttentionScoresKernel<true><<<dim3(c.batch, headBlocks, p.tiles), kMmaThreads,
                sizeof(V41MmaShared), cudaStreamPerThread>>>(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, c.heads, c.window,
            0, 0, 0, nullptr, 0, 0, 0, 0, 0, p.attention);
    V41SparseAttentionMmaKernel<SPLITS, MATERIALIZED, true><<<dim3(c.batch, headBlocks, SPLITS),
            kMmaThreads, sizeof(Shared), cudaStreamPerThread>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        static_cast<const float *>(p.buffers.sink), 1, c.heads, c.window, 0, 0, 0, c.softmaxScale,
        nullptr, 0, 0, 0, 0, 0, nullptr, p.attention);
}
} // namespace

void *FastllmCudaV41CreateFullDecode(const FastllmV41FullDecodeConfig &c,
                                   const FastllmV41FullDecodeBuffers &b) {
    if (FastllmCudaGraphIsCapturingFast() || c.batch <= 0 || c.batch > 128 || c.heads <= 0 ||
        c.indexHeads <= 0 || c.ratio < 0 || c.ratio > 2 || c.window <= 0 ||
        c.maxKeys <= 0 || c.topK <= 0 || c.topK > c.maxKeys || c.blockSize <= 0 ||
        c.ropeDim <= 0 || c.ropeDim > 128 || (c.ropeDim & 1) ||
        !b.q || !b.kv || !b.output || !b.sink ||
        (c.verifying && (!b.savedKV || (c.kvSource &&
            (!b.savedRawKV || (c.ratio == 2 && !b.savedRawScore))))) ||
        (c.ratio && (!b.topk || !b.candidates)) ||
        (c.kvSource && (!c.ratio || !b.rawKV || !b.joined || !b.latents || !b.compressNorm ||
          (c.ratio == 2 && (!b.rawScore || !b.scores)))) ||
        (c.indexSource && (!c.ratio || !b.indexQ || !b.indexWeights || (c.kvSource && !b.newIndexKeys))))
        return nullptr;
    auto p = std::make_unique<V41FullDecodePlan>();
    p->config = c; p->buffers = b; p->device = FastllmCudaGetDevice();
    p->tiles = (c.window + (c.ratio ? c.topK : 0) + kMmaNC - 1) / kMmaNC;
    const int heads = (c.heads + kMmaHeads - 1) / kMmaHeads;
    // Verification rows use the same independent QK tiles and value-dimension
    // partitioning as decode. This keeps the 64-key online-softmax recurrence
    // intact while exposing enough CTAs for small batches (including TP8).
    // Large verification batches already have enough independent CTAs and
    // retain the fused path; DSpark verification chunks contain 2--8 rows.
    p->splits = (c.verifying && c.batch > 8) || c.batch * heads >= 128 ? 1 :
                c.batch >= 64 ? 2 : 8;
    if (!V41MmaKernelReady<V41SparseAttentionScoresKernel<true>, kMmaThreads, sizeof(V41MmaShared)>()) return nullptr;
    if (p->splits == 1) {
        p->fused = V41MmaKernelReady<V41SparseAttentionMmaKernel<1, false, true>, kMmaThreads, sizeof(V41MmaShared)>();
        if (!p->fused && !V41MmaKernelReady<V41SparseAttentionMmaKernel<1, true, true>, kMmaThreads, sizeof(V41MmaPvShared<1>)>()) return nullptr;
    } else if (p->splits == 2) {
        if (!V41MmaKernelReady<V41SparseAttentionMmaKernel<2, true, true>, kMmaThreads, sizeof(V41MmaPvShared<2>)>()) return nullptr;
    } else if (!V41MmaKernelReady<V41SparseAttentionMmaKernel<8, true, true>, kMmaThreads, sizeof(V41MmaPvShared<8>)>()) return nullptr;
    if (c.indexSource && !V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true, true>, 256>()) return nullptr;
    p->rope = V41BuildRope(c.ropeDim, c.ropeBase, c.originalSeqLen, c.ropeFactor, c.betaFast, c.betaSlow);
    const int blocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
    for (int shift = 0; shift < 30; ++shift) if ((1 << shift) == c.blockSize) p->blockShift = shift;
    p->listCap = p->blockShift >= 0 ? std::min(blocks, 4096) : 0;
    auto alloc = [](auto **ptr, size_t count) {
        using T = typename std::remove_pointer<typename std::remove_reference<decltype(*ptr)>::type>::type;
        return cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)) == cudaSuccess;
    };
    p->requests = static_cast<FastllmV41FullDecodeRequest *>(b.requests);
    bool ok = (p->requests || alloc(&p->requests, c.batch)) &&
        alloc(&p->transforms, (size_t)c.batch * kFullDecodeTransforms) &&
        alloc(&p->attention, c.batch) && alloc(&p->attentionScores, (size_t)c.batch * c.heads * p->tiles * kMmaNC);
    if (c.indexSource) ok = ok && alloc(&p->index, c.batch) &&
        alloc(&p->indexScores, (size_t)c.batch * c.maxKeys) && alloc(&p->blockScores, (size_t)c.batch * blocks);
    if (!b.requests)
        ok = ok && cudaHostAlloc(reinterpret_cast<void **>(&p->host), (size_t)c.batch * sizeof(*p->host), cudaHostAllocDefault) == cudaSuccess &&
             cudaEventCreateWithFlags(&p->copied, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) { cudaGetLastError(); return nullptr; }
    return p.release();
}

void FastllmCudaV41DestroyFullDecode(void *plan) { delete static_cast<V41FullDecodePlan *>(plan); }

namespace {
bool V41ValidateFullDecodeRequests(const V41FullDecodePlan *p,
        const FastllmV41FullDecodeRequest *requests, size_t count) {
    if (!p || !requests || count != (size_t)p->config.batch ||
        FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config;
    for (size_t i = 0; i < count; ++i) {
        const auto &r = requests[i];
        if (c.verifying && (r.position != requests[0].position + (int)i ||
            r.windowKV != requests[0].windowKV || r.compressedKV != requests[0].compressedKV ||
            r.indexK != requests[0].indexK || r.rawTailKV != requests[0].rawTailKV ||
            r.rawTailScore != requests[0].rawTailScore)) return false;
        if (r.position < 1 || !r.windowKV || (c.ratio &&
            (!r.compressedKV || (r.position + 1LL) / c.ratio > c.maxKeys)) ||
            (c.indexSource && !r.indexK) ||
            (c.kvSource && c.ratio > 1 && (!r.rawTailKV || !r.rawTailScore))) return false;
    }
    return true;
}
} // namespace

bool FastllmCudaV41UploadFullDecodeBatch(void *storage, void *const *plans, size_t layers,
        const FastllmV41FullDecodeRequest *requests, size_t batch) {
    if (!storage || !plans || !requests || !layers || !batch ||
        layers > SIZE_MAX / batch / sizeof(*requests) || FastllmCudaGraphIsCapturingFast())
        return false;
    auto *deviceRequests = static_cast<FastllmV41FullDecodeRequest *>(storage);
    for (size_t l = 0; l < layers; ++l) {
        auto *p = static_cast<V41FullDecodePlan *>(plans[l]);
        if (!p) continue;
        if (!p->buffers.requests || p->requests != deviceRequests + l * batch ||
            !V41ValidateFullDecodeRequests(p, requests + l * batch, batch)) return false;
    }
    // The bounded pinned staging pool retains each source until its copy event
    // completes. The upload and graph replay use the same worker stream.
    return V41UploadDescriptors(storage, requests, layers * batch * sizeof(*requests));
}

bool FastllmCudaV41UploadFullDecode(void *handle, const FastllmV41FullDecodeRequest *requests, size_t count) {
    auto *p = static_cast<V41FullDecodePlan *>(handle);
    if (FastllmCudaGraphIsCapturingFast() || !V41ValidateFullDecodeRequests(p, requests, count) ||
        p->buffers.requests) return false;
    if (p->pending && cudaEventSynchronize(p->copied) != cudaSuccess) return false;
    std::memcpy(p->host, requests, count * sizeof(*requests));
    if (cudaMemcpyAsync(p->requests, p->host, count * sizeof(*requests), cudaMemcpyHostToDevice,
                        cudaStreamPerThread) != cudaSuccess) return false;
    if (cudaEventRecord(p->copied, cudaStreamPerThread) != cudaSuccess) {
        cudaStreamSynchronize(cudaStreamPerThread); return false;
    }
    p->pending = true;
    return true;
}

bool FastllmCudaV41PrepareFullDecode(void *handle) {
    auto *p = static_cast<V41FullDecodePlan *>(handle);
    if (!p || FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config; const auto &b = p->buffers;
    V41FullDecodePrepareKernel<<<dim3(c.heads + 1, c.batch), 512, 0, cudaStreamPerThread>>>(
        c, b, p->requests, p->transforms, p->index, p->attention,
        p->indexScores, p->blockScores, p->attentionScores, p->tiles, p->rope);
    if (c.kvSource) {
        if (c.verifying)
            V41FullVerifyGather<<<c.batch, 512, 0, cudaStreamPerThread>>>(c, b, p->requests);
        else
            V41FullDecodeGather<<<c.batch, 512, 0, cudaStreamPerThread>>>(p->requests,
                static_cast<const float *>(b.rawKV), static_cast<const float *>(b.rawScore),
                static_cast<float *>(b.joined), static_cast<float *>(b.scores), c.ratio);
        V41CompressKernel<float><<<c.batch, 512, 512 * sizeof(float), cudaStreamPerThread>>>(
            static_cast<const float *>(b.joined), static_cast<const float *>(b.scores),
            static_cast<const float *>(b.compressNorm), c.ratio, 512, c.ratio, c.normEps,
            static_cast<__nv_bfloat16 *>(b.latents));
    }
    return V41CheckLaunch("FullDecodePrepare");
}

bool FastllmCudaV41FinishFullDecode(void *handle) {
    auto *p = static_cast<V41FullDecodePlan *>(handle);
    if (!p || FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config; const auto &b = p->buffers;
    if (c.kvSource) {
        V41DecodeTransformsKernel<kFullDecodeTransforms><<<dim3(1, c.batch), 512, 0, cudaStreamPerThread>>>(p->transforms + 2, p->rope, c.ropeDim, c.window);
        if (c.indexSource)
            V41DecodeTransformsKernel<kFullDecodeTransforms><<<dim3(1, c.batch), 512, 0, cudaStreamPerThread>>>(p->transforms + 3, p->rope, c.ropeDim, c.window);
        V41FullDecodeAppend<<<c.batch, 512, 0, cudaStreamPerThread>>>(p->requests,
            static_cast<const __nv_bfloat16 *>(b.latents), static_cast<const __nv_bfloat16 *>(b.newIndexKeys), c.ratio, c.indexSource);
    }
    if (c.indexSource) {
        V41DecodeTransformsKernel<kFullDecodeTransforms><<<dim3((c.indexHeads + 3) / 4, c.batch), 512, 0, cudaStreamPerThread>>>(p->transforms + 4, p->rope, c.ropeDim, c.window);
        V41IndexerScoreDecodeMmaKernel<true, true><<<dim3(128, c.batch), 256, 0, cudaStreamPerThread>>>(
            nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, p->index);
        if (c.writeCandidates) {
            V41BlockScoreKernel<true><<<c.batch, 256, 0, cudaStreamPerThread>>>(nullptr, 1, 0, c.blockSize, 0, c.ratio, 0, nullptr, p->index);
            V41CandidateSelectKernel<true><<<c.batch, kSelThreads, 0, cudaStreamPerThread>>>(nullptr, 0, c.candidateTopK, nullptr, p->index);
        }
        V41TopKKernel<true, true, 1024><<<c.batch, 1024, (size_t)p->listCap * sizeof(int), cudaStreamPerThread>>>(
            nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0, p->listCap, nullptr, p->index);
    }
    if (p->splits == 8) V41FullDecodeAttention<8, true>(*p);
    else if (p->splits == 2) V41FullDecodeAttention<2, true>(*p);
    else if (p->fused) V41FullDecodeAttention<1, false>(*p);
    else V41FullDecodeAttention<1, true>(*p);
    // Independent descriptors share one launch; each row retains its original math.
    if (c.verifying) {
        V41DecodeTransformsKernel<kFullDecodeTransforms><<<dim3(c.heads, c.batch), 512, 0, cudaStreamPerThread>>>(p->transforms, p->rope, c.ropeDim, c.window);
        V41FullVerifySaveKV<<<c.batch, 512, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(b.kv), static_cast<__nv_bfloat16 *>(b.savedKV));
    } else {
        V41DecodeTransformsKernel<kFullDecodeTransforms, 2><<<dim3(c.heads, c.batch * 2), 512, 0, cudaStreamPerThread>>>(p->transforms, p->rope, c.ropeDim, c.window);
    }
    return V41CheckLaunch("FullDecodeFinish");
}
