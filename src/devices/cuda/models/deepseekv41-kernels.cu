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
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
#include <cooperative_groups.h>
#endif
#include "deepseekv41-router.cuh"
#include "deepseekv41-indexer.cuh"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-v41-route-order.cuh"
#endif
#include "deepseekv41-router-tc.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#if !defined(USE_ROCM) && CUDART_VERSION >= 11070
#include "devices/cuda/fastllm-cuda-driver-entry.h"
#endif
#include "devices/cuda/fastllm-v41-full-decode.h"
#include "fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-dsv41-mega.h"
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
#ifndef USE_ROCM
#include "deepseekv41-flashinfer.cuh"
#include "deepseekv41-verify-merge.cuh"
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
#include "deepseekv41-flashmla.cuh"
#endif
// FlashInfer synchronization primitives require SM70 or newer.
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
#define FASTLLM_V41_FP16_ATTENTION
#include "attention/default_prefill_params.cuh"
#include "attention/prefill.cuh"
#include "attention/variants.cuh"
#endif
#endif
#include "fastllm.h"

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
#include <cuda_fp8.h>
void *FastllmCudaV41PrepareBlock32Kernel();
void *FastllmCudaV41IndexHeadGemvKernel();
void FastllmCudaV41IndexWeightKernels(void *&convert, void *&multiply);
#endif
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

    bool V41UploadCopies(const FastllmV41HostCopy &first,
                         const FastllmV41HostCopy *extra, size_t extraCount) {
        if (extraCount == SIZE_MAX || (extraCount && !extra)) return false;
        const size_t count = extraCount + 1;
        auto copyAt = [&](size_t i) -> const FastllmV41HostCopy & {
            return i == 0 ? first : extra[i - 1];
        };
        size_t bytes = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto &copy = copyAt(i);
            if (!copy.dst || !copy.src || !copy.bytes ||
                copy.bytes > SIZE_MAX - bytes) return false;
            bytes += copy.bytes;
        }
        auto synchronous = [&] {
            for (size_t i = 0; i < count; ++i) {
                const auto &copy = copyAt(i);
                if (cudaMemcpy(copy.dst, copy.src, copy.bytes,
                               cudaMemcpyHostToDevice) != cudaSuccess) return false;
            }
            return true;
        };
        if (bytes > 128 * 1024 || FastllmCudaGraphIsCapturingFast()) return synchronous();
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
                return synchronous();
            }
            if (slot.host) cudaFreeHost(slot.host);
            slot.host = replacement; slot.capacity = capacity;
        }
        size_t offset = 0;
        bool contiguous = true;
        for (size_t i = 0; i < count; ++i) {
            const auto &copy = copyAt(i);
            std::memcpy(static_cast<char *>(slot.host) + offset, copy.src, copy.bytes);
            contiguous &= reinterpret_cast<uintptr_t>(copy.dst) ==
                          reinterpret_cast<uintptr_t>(first.dst) + offset;
            offset += copy.bytes;
        }
        // Adjacent pointers may belong to separate allocations. Only merge
        // when the entire destination range belongs to one device allocation.
        contiguous = contiguous && count > 1 &&
                     FastllmCudaValidatePointerRange(first.dst, bytes, device);
        offset = 0;
        for (size_t i = 0; i < (contiguous ? 1 : count); ++i) {
            const auto &copy = copyAt(i);
            char *host = static_cast<char *>(slot.host) + offset;
            if (cudaMemcpyAsync(copy.dst, host, contiguous ? bytes : copy.bytes,
                    cudaMemcpyHostToDevice, cudaStreamPerThread) != cudaSuccess) {
                // Earlier copies may already read this slot. Do not recycle it
                // before they have finished, including on the failure path.
                cudaStreamSynchronize(cudaStreamPerThread);
                return false;
            }
            offset += copy.bytes;
        }
        if (cudaEventRecord(slot.copied, cudaStreamPerThread) != cudaSuccess) {
            cudaStreamSynchronize(cudaStreamPerThread);
            return false;
        }
        slot.pending = true;
        return true;
    }

    bool V41UploadDescriptors(void *dst, const void *src, size_t bytes) {
        return V41UploadCopies({dst, src, bytes}, nullptr, 0);
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

    __device__ __forceinline__ float V41QuantCodeInput(float v, float scale, int quantMode) {
        float qmax = quantMode == 1 ? 448.0f : 6.0f;
        const unsigned bits = __float_as_uint(scale);
        // FP8 block scales are powers of two. Their normal reciprocals are
        // exact, so multiplication preserves division's FP32 rounding. Keep
        // division for FP4 and scales whose reciprocal is not normal.
        const float scaled = quantMode == 1 && bits >= 0x00800000u &&
                             bits < 0x7f000000u && !(bits & 0x007fffffu)
            ? v * __uint_as_float(0x7f000000u - bits) : v / scale;
        return fminf(qmax, fmaxf(-qmax, scaled));
    }

    __device__ __forceinline__ float V41QuantValueDev(float v, float scale, int quantMode) {
        float q = V41QuantCodeInput(v, scale, quantMode);
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
    template<int Tokens, typename Weight = float>
    __global__ void V41RouterScoresKernel(const __nv_bfloat16 *input, const Weight *weight,
                                          float *scores, int dim, int experts, float invTemp, int rows) {
        const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
        const int expert = blockIdx.x, first = blockIdx.y * Tokens;
        float sums[Tokens] = {};
#pragma unroll 4
        for (int i = tid; i < dim; i += 256) {
            const float coefficient = V41Load(weight, (size_t)expert * dim + i);
#pragma unroll
            for (int t=0;t<Tokens;++t)
                if (first+t<rows) sums[t] += __bfloat162float(input[(size_t)(first+t)*dim+i]) * coefficient;
        }
        __shared__ float partial[Tokens][8];
#pragma unroll
        for (int t=0;t<Tokens;++t) {
            float sum=sums[t];
#pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                const float other = __shfl_down_sync(0xffffffff, sum, offset);
                if ((lane & (2 * offset - 1)) == 0) sum += other;
            }
            if (lane == 0) partial[t][warp] = sum;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int t=0;t<Tokens;++t) {
                float sum = lane < 8 ? partial[t][lane] : 0.0f;
#pragma unroll
                for (int offset = 1; offset < 8; offset *= 2) {
                    const float other = __shfl_down_sync(0xffffffff, sum, offset);
                    if ((lane & (2 * offset - 1)) == 0) sum += other;
                }
                if (lane == 0 && first+t<rows) {
                    const float value = __fmul_rn(__fadd_rn(sum, 0.0f), invTemp);
                    const float softplus = value > 20.0f ? value :
                        (value < -20.0f ? expf(value) : log1pf(expf(value)));
                    scores[(size_t)(first+t)*experts+expert] = sqrtf(softplus);
                }
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030)
        // Exactly 256 threads participate in these dots, including the dot
        // CTAs of the combined SM120 kernel. Expose their range to NVCC.
        if constexpr (TILE > 0 || __CUDA_ARCH__ != 1030) k &= kHcThreads - 1;
        if constexpr (TILE > 0) {
            // Tiled launchers check shape eligibility. Preserve each lane's
            // FMA order; partial tiles use the original loop below.
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

    // Stage independent loads in registers while keeping each lane's FMA
    // order. Callers choose the tile for the target SM and row count.
    template <int TILE>
    __global__ __launch_bounds__(kHcThreads, 1)
    void V41HcDecodeDotsTiled(const __nv_bfloat16 *x, const float *fn, int flatDim, float *mix) {
        V41HcDecodeDotsBody<__nv_bfloat16, TILE>(x, fn, flatDim, mix);
    }

    // Each block computes one original warp's partial dot. At one token this
    // gives SM103 200 independently schedulable blocks instead of 25. The
    // finish kernel adds the eight warp sums in their original order.
    __global__ void V41HcWarpDots5120(const __nv_bfloat16 *x, const float *fn, float *mix) {
        constexpr int flat = 4 * 5120;
        const int dot = blockIdx.x, token = blockIdx.y, warp = blockIdx.z;
        const int first = warp * 32 + threadIdx.x;
        x += size_t(token) * flat;
        fn += size_t(dot == 24 ? 0 : dot) * flat;
        float sum = 0.0f;
#pragma unroll 16
        for (int i = 0; i < flat / 256; ++i) {
            const int k = first + i * 256;
            const float value = __bfloat162float(x[k]);
            sum = fmaf(value, dot == 24 ? value : fn[k], sum);
        }
        sum = V41WarpSum(sum);
        if (threadIdx.x == 0) mix[(token * 8 + warp) * 25 + dot] = sum;
    }

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    // Eight warps fetch one reference warp's 80 input/weight pairs in parallel.
    // The original lanes still perform all FMAs and the same XOR reduction.
    template <int Outputs>
    __global__ void V41HcStagedDots5120(const __nv_bfloat16 *x, const float *weights, float *mix) {
        constexpr int flat = 4 * 5120, parts = flat / 256;
        const int token = blockIdx.y, warp = blockIdx.z, firstDot = blockIdx.x * Outputs;
        const int tid = threadIdx.x, lane = tid & 31;
        __shared__ float values[parts * 32], factors[Outputs][parts * 32];
#pragma unroll
        for (int i = 0; i < parts * 32 / 256; ++i) {
            const int local = i * 256 + tid;
            const int column = (local / 32) * 256 + warp * 32 + lane;
            values[local] = __bfloat162float(x[size_t(token) * flat + column]);
#pragma unroll
            for (int d = 0; d < Outputs; ++d)
                if (firstDot + d < 24) factors[d][local] = weights[(firstDot + d) * flat + column];
        }
        __syncthreads();
        const int output = tid / 32, dot = firstDot + output;
        if (output >= Outputs || dot >= 25) return;
        float sum = 0.f;
#pragma unroll 16
        for (int i = 0; i < parts; ++i) {
            const float value = values[i * 32 + lane];
            sum = fmaf(value, dot == 24 ? value : factors[output][i * 32 + lane], sum);
        }
        sum = V41WarpSum(sum);
        if (!lane) mix[(token * 8 + warp) * 25 + dot] = sum;
    }

    // These launches and CUDA graph rewrites share the same shape contract.
    bool V41SelectStagedDots(int rows, cudaKernelNodeParams &p) {
        if (rows > 8 || !FastllmV41HcEnabledAtRows("FASTLLM_DSV41_STAGED_HC", rows)) return false;
        p.func = rows == 1 ? (void *)V41HcStagedDots5120<1> : (void *)V41HcStagedDots5120<2>;
        p.gridDim = dim3(rows == 1 ? 25 : 13, rows, 8);
        p.blockDim = dim3(256);
        return true;
    }
#endif

    // A CTA owns one of the original dot kernel's eight logical warps.
    // Keep every lane's strided FMA sequence and XOR reduction unchanged; the
    // finish kernel sums the eight warp results in the original left-to-right
    // order. Post uses separate multiply/add and rounds to BF16 before the dot.
    __device__ __forceinline__ void V41DependencyWait() {
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        cudaGridDependencySynchronize();
#endif
    }
    __device__ __forceinline__ void V41DependencyLaunch() {
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    }

    template <int DIM, int OUTPUTS>
    __global__ __launch_bounds__(32, 1)
    void V41HcPostWarpDots(const __nv_bfloat16 *__restrict__ input,
            const __nv_bfloat16 *__restrict__ residual, const float *__restrict__ post,
            const float *__restrict__ comb, const float *__restrict__ fn,
            __nv_bfloat16 *__restrict__ hidden, float *__restrict__ mix) {
        V41DependencyWait();
        V41DependencyLaunch();
        constexpr int iterations = DIM / 256;
        const int token = blockIdx.y, first = blockIdx.x * OUTPUTS;
        const int tid = blockIdx.z * 32 + threadIdx.x;
        input += token * DIM;
        residual += token * 4 * DIM;
        hidden += token * 4 * DIM;
        post += token * 4;
        comb += token * 16;
        mix += (token * 8 + blockIdx.z) * 25;
        float values[4][iterations];
#pragma unroll
        for (int i = 0; i < iterations; ++i) {
            const int col = tid + i * 256;
            float r[4], x = __bfloat162float(input[col]);
#pragma unroll
            for (int source = 0; source < 4; ++source)
                r[source] = __bfloat162float(residual[source * DIM + col]);
#pragma unroll
            for (int target = 0; target < 4; ++target) {
                float sum = 0.0f;
#pragma unroll
                for (int source = 0; source < 4; ++source)
                    sum = __fadd_rn(sum, __fmul_rn(comb[source * 4 + target], r[source]));
                const auto rounded = __float2bfloat16_rn(
                    __fadd_rn(__fmul_rn(post[target], x), sum));
                values[target][i] = __bfloat162float(rounded);
                if (first == 0) hidden[target * DIM + col] = rounded;
            }
        }
        float acc[OUTPUTS] = {};
#pragma unroll
        for (int target = 0; target < 4; ++target) {
#pragma unroll
            for (int i = 0; i < iterations; ++i) {
#pragma unroll
                for (int o = 0; o < OUTPUTS; ++o) {
                    const float value = values[target][i];
                    const float weight = first + o >= 24 ? value :
                        fn[(first + o) * 4 * DIM + target * DIM + tid + i * 256];
                    acc[o] += value * weight;
                }
            }
        }
#pragma unroll
        for (int o = 0; o < OUTPUTS; ++o) {
            const float value = V41WarpSum(acc[o]);
            if (threadIdx.x == 0 && first + o < 25) mix[first + o] = value;
        }
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

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    __global__ void V41HcJoinTensorDots(const __nv_bfloat16 *x, const float *dots, float *mix) {
        const int row = blockIdx.x;
        x += row * 20480; dots += row * 48; mix += row * 25;
        float sum = 0.f;
        for (int k = threadIdx.x; k < 20480; k += kHcThreads) {
            const float value = __bfloat162float(x[k]);
            sum += value * value;
        }
        __shared__ float partial[kHcThreads / 32];
        sum = V41WarpSum(sum);
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.f;
            for (int i = 0; i < kHcThreads / 32; ++i) total += partial[i];
            mix[24] = total;
        }
        if (threadIdx.x < 24) mix[threadIdx.x] = dots[threadIdx.x] + dots[24 + threadIdx.x];
    }
#endif

    bool V41EnvOn(const char *name);

    bool V41HcTensorCandidate(const Data &x, int rows) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        return rows >= 16 && rows <= 192 && x.dataType == DataType::BFLOAT16 &&
            x.dims.back() == 5120 && FastllmCudaRuntimeArch() == 103 && !V41EnvOn("FASTLLM_DSV41_DISABLE_HC_TENSOR");
#else
        return false;
#endif
    }

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    // This optional capture marker gives graph composition an ABI-independent
    // boundary for the cuBLAS projection. Composition removes the marker.
    __global__ void V41HcTensorMarker(const __nv_bfloat16 *hidden, const void *packed) {}

    cublasStatus_t V41HcTensorGemm(cublasHandle_t handle, const void *hidden, const void *packed,
                                   int rows, float *projected) {
        const float alpha = 1.f, beta = 0.f;
        return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 48, rows, 20480,
            &alpha, packed, CUDA_R_16BF, 20480, hidden, CUDA_R_16BF, 20480,
            &beta, projected, CUDA_R_32F, 48, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
#endif

    bool V41HcTensorDots(const Data &x, Data &fn, int rows, float *mix) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        if (!V41HcTensorCandidate(x, rows)) return false;
        auto *packed = FastllmCudaHcSplitWeight(fn);
        if (!packed) return false;
        auto handle = getFastllmCublasFp32ReductionHandle();
        if (!handle) return false;
        // Both views belong to the same graph-pinned allocation. Align cuBLAS C.
        float *projected = mix + (rows * 25 + 3) / 4 * 4;
        if (FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_OVERLAP", rows) && V41EnvOn("FASTLLM_DSV41_HC_TENSOR_OVERLAP")) {
            cudaStreamCaptureStatus capture;
            if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) == cudaSuccess &&
                capture == cudaStreamCaptureStatusActive)
                V41HcTensorMarker<<<1, 1, 0, cudaStreamPerThread>>>(
                    static_cast<const __nv_bfloat16 *>(x.cudaData), packed->cudaData);
        }
        const auto status = V41HcTensorGemm(handle, x.cudaData, packed->cudaData, rows, projected);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH ||
            status == CUBLAS_STATUS_ALLOC_FAILED) return false;
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("mHC cuBLAS failure " + std::to_string(int(status)));
        V41HcJoinTensorDots<<<rows, kHcThreads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)x.cudaData, projected, mix);
        return true;
#else
        return false;
#endif
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
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    __device__ __forceinline__ void V41PackNormalizedPair(__nv_bfloat162 ov, uint8_t *codes,
            uint8_t *scales, int pitch, int token, int bf2, int i) {
        float2 v = __bfloat1622float2(ov);
        float maximum = fmaxf(fabsf(v.x), fabsf(v.y));
        const unsigned active = __activemask();
#pragma unroll
        for (int d = 8; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, d, 16));
        const unsigned bits = __float_as_uint(maximum / 448.f);
        const unsigned e = min(254u, max(1u, (bits >> 23) + unsigned((bits & 0x7fffffu) != 0)));
        const float inverse = __uint_as_float(e == 254 ? 0x00400000u : (254u - e) << 23);
        v.x *= inverse; v.y *= inverse;
        reinterpret_cast<__nv_fp8x2_storage_t *>(codes)[size_t(token) * bf2 + i] =
        __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
        if (!(i & 15)) scales[(size_t(i / 64) * pitch + token) * 4 + (i / 16) % 4] = e;
    }
#endif
    template <int THREAD_PER_BLOCK, int MAX_ITER, bool FINISH_MIX = false, bool QUANTIZE_OUTPUT = false, int MIX_WARPS = 1, bool PACK_MX = false>
    __device__ __forceinline__ void
    V41HcPreNormBody(const __nv_bfloat16 *x, const float *pre, const float *weight,
                       __nv_bfloat16 *out, int hcMult, int channels, float eps, int token,
                       const float *mix = nullptr, const float *mixScale = nullptr,
                       const float *mixBase = nullptr, int sinkhornIters = 0,
                       float mixEps = 0, float *post = nullptr, float *comb = nullptr, float *nextPre = nullptr,
                       __nv_bfloat16 *quantized = nullptr, uint8_t *codes = nullptr,
                       uint8_t *scales = nullptr, int pitch = 0) {
        V41DependencyWait();
        V41DependencyLaunch();
        // The model consumes the PREVIOUS sublayer's pre coefficients.
        // Finish and normalization are independent, so separate CTAs overlap
        // them in one launch. Keep both original reduction orders.
        if constexpr (FINISH_MIX) {
            if ((blockIdx.x & 1) == 0) {
                if constexpr (MIX_WARPS == 1) {
                    if (threadIdx.x < 32)
                        V41HcFinishRow(mix, mixScale, mixBase, hcMult * channels,
                            sinkhornIters, mixEps, eps, nextPre, post, comb, blockIdx.x / 2);
                } else {
                    __shared__ float reduced[25];
                    if (threadIdx.x < 32) {
                        const int lane = threadIdx.x, row = blockIdx.x / 2;
                        if (lane < 25) {
                            float value = 0.0f;
#pragma unroll
                            for (int split = 0; split < MIX_WARPS; ++split)
                                value += mix[((uint64_t)row * MIX_WARPS + split) * 25 + lane];
                            reduced[lane] = value;
                        }
                        __syncwarp();
                        // Preserve the old dot kernel's eight-warp sum order.
                        // Coefficient output uses row zero here, then offsets
                        // the three destinations to this token's actual row.
                        V41HcFinishRow(reduced, mixScale, mixBase, hcMult * channels,
                            sinkhornIters, mixEps, eps, nextPre + row * 4,
                            post + row * 4, comb + row * 16, 0);
                    }
                }
                return;
            }
        }
        constexpr int WARP_SIZE = 32;
        constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
        __shared__ float warp_sums[NUM_WARPS];

        const int bf2 = channels / 2;
        const __nv_bfloat162 *xb = reinterpret_cast<const __nv_bfloat162*>(x) + (uint64_t)token * hcMult * bf2;
        const float *preRow = pre + (uint64_t)token * hcMult;
        __nv_bfloat162 *outb = reinterpret_cast<__nv_bfloat162*>(out) + (uint64_t)token * bf2;

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
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if constexpr (PACK_MX) {
                    V41PackNormalizedPair(ov, codes, scales, pitch, token, bf2, i);
                }
#endif
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
                    reinterpret_cast<__nv_bfloat162 *>(quantized)[(uint64_t)token * bf2 + i] = qv;
                }
            }
        }
    }

    template <int THREAD_PER_BLOCK, int MAX_ITER, bool FINISH_MIX = false, bool QUANTIZE_OUTPUT = false, int MIX_WARPS = 1>
    __global__ void __launch_bounds__(THREAD_PER_BLOCK)
    V41HcPreNormKernel(const __nv_bfloat16 *x, const float *pre, const float *weight,
                       __nv_bfloat16 *out, int hcMult, int channels, float eps,
                       const float *mix = nullptr, const float *mixScale = nullptr,
                       const float *mixBase = nullptr, int sinkhornIters = 0,
                       float mixEps = 0, float *post = nullptr, float *comb = nullptr, float *nextPre = nullptr,
                       __nv_bfloat16 *quantized = nullptr) {
        V41HcPreNormBody<THREAD_PER_BLOCK, MAX_ITER, FINISH_MIX, QUANTIZE_OUTPUT, MIX_WARPS>(
            x, pre, weight, out, hcMult, channels, eps, FINISH_MIX ? blockIdx.x / 2 : blockIdx.x,
            mix, mixScale, mixBase,
            sinkhornIters, mixEps, post, comb, nextPre, quantized);
    }

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    template <bool Quantized>
    __global__ __launch_bounds__(1024) void V41HcPreNormPacked(const __nv_bfloat16 *x, const float *pre,
            const float *weight, __nv_bfloat16 *out, int hc, int width, float eps,
            const float *, const float *, const float *, int, float, float *, float *, float *,
            __nv_bfloat16 *quantized, uint8_t *codes, uint8_t *scales, int pitch) {
        V41HcPreNormBody<1024, 3, false, Quantized, 1, true>(x, pre, weight, out, hc, width, eps,
            blockIdx.x, nullptr, nullptr, nullptr, 0, 0.f, nullptr, nullptr, nullptr, quantized, codes, scales, pitch);
    }
#endif
    // The delayed-pre normalization reads previousPre, not this layer's mix.
    // Run it alongside the 25 independent dots, retaining both reduction trees.
    // Only the small single-token SM120 path uses this combined launch.
    template <bool QUANTIZE>
    __global__ void __launch_bounds__(1024)
    V41HcDotsPreNormKernel(const __nv_bfloat16 *x, const float *fn, float *mix,
            const float *previousPre, const float *normWeight, __nv_bfloat16 *out,
            __nv_bfloat16 *quantized, float eps) {
        if (blockIdx.x == 25) {
            V41HcPreNormBody<1024, 3, false, QUANTIZE>(x, previousPre, normWeight,
                out, 4, 5120, eps, 0, nullptr, nullptr, nullptr, 0, 0.f,
                nullptr, nullptr, nullptr, quantized);
        } else if (threadIdx.x < kHcThreads) {
            V41HcDecodeDotsBody<__nv_bfloat16, 16>(x, fn, 4 * 5120, mix);
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
    template <typename T, bool Batched = false>
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
        const int request = Batched ? blockIdx.y : 0;
        T *base = x + ((uint64_t)request * (totalPairs / pairs) + r) * dim + (dim - ropeDim) + 2 * d;
        const float pos = (float)((devicePosition ? devicePosition[request] : startPos) + (long long)token * posStep);
        const float ang = pos * rope.invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        if (inverse) {
            s = -s;
        }
        const float a = V41Load<T>(base, 0), b = V41Load<T>(base, 1);
        V41Store<T>(base, 0, a * c - b * s);
        V41Store<T>(base, 1, __fmaf_rn(b, c, __fmul_rn(a, s)));
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

    // Independent CTAs rotate Q, rotate/quantize current KV, and copy the
    // immutable window. This preserves the two original rotary expressions
    // and the BF16 storage boundary before KV's block-32 quantization.
    __global__ void V41DraftRotaryTableKernel(__nv_bfloat16 *q, __nv_bfloat16 *kv,
            const __nv_bfloat16 *window, __nv_bfloat16 *table, V41RopeTable rope,
            int heads, int tokens, int windowRows, int ropeDim, int startPos,
            const int *devicePosition, int qBlocks, const int *changed) {
        constexpr int dim = 512;
        const int blocksPerRequest = qBlocks + tokens + (windowRows + 7) / 8;
        const int request = blockIdx.x / blocksPerRequest;
        const int block = blockIdx.x % blocksPerRequest;
        q += size_t(request) * tokens * heads * dim;
        kv += size_t(request) * tokens * dim;
        window += size_t(request) * windowRows * dim;
        table += size_t(request) * (windowRows + tokens) * dim;
        if (devicePosition) devicePosition += request;
        if (block < qBlocks) {
            const int idx = block * blockDim.x + threadIdx.x;
            const int pairs = ropeDim / 2;
            if (idx >= tokens * heads * pairs) return;
            const int row = idx / pairs, d = idx % pairs;
            auto *base = q + (size_t)row * dim + dim - ropeDim + 2 * d;
            const float pos = (float)((devicePosition ? *devicePosition : startPos) + (long long)(row / heads));
            const float ang = pos * rope.invFreq[d];
            const float c = cosf(ang), s = sinf(ang);
            const float a = V41Load<__nv_bfloat16>(base, 0), b = V41Load<__nv_bfloat16>(base, 1);
            V41Store<__nv_bfloat16>(base, 0, a * c - b * s);
            V41Store<__nv_bfloat16>(base, 1, __fmaf_rn(b, c, __fmul_rn(a, s)));
        } else if (block < qBlocks + tokens) {
            const int row = block - qBlocks, d = threadIdx.x;
            auto *base = kv + (size_t)row * dim;
            float v = V41Load<__nv_bfloat16>(base, d);
            const float other = __shfl_xor_sync(0xffffffff, v, 1);
            const int off = dim - ropeDim;
            if (d >= off) {
                const int k = d - off;
                const float pos = (float)((devicePosition ? *devicePosition : startPos) + (long long)row);
                const float ang = pos * rope.invFreq[k >> 1];
                const float c = cosf(ang), s = sinf(ang);
                v = (k & 1) ? (other * s + v * c) : (v * c - other * s);
            }
            v = V41StorageRound<__nv_bfloat16>(v);
            const float scale = V41QuantScaleDev(V41WarpMax(fabsf(v)), 1);
            v = V41QuantValueDev(v, scale, 1);
            const auto result = __float2bfloat16_rn(v);
            base[d] = result;
            table[(size_t)(windowRows + row) * dim + d] = result;
        } else {
            const size_t idx = (size_t)(block - qBlocks - tokens) * blockDim.x + threadIdx.x;
            if (idx < (size_t)windowRows * (dim / 8) &&
                (!changed || changed[size_t(request) * windowRows + idx / (dim / 8)]))
                ((uint4*)table)[idx] = ((const uint4*)window)[idx];
        }
    }

    constexpr int kKvQuantThreads = 128;
    template <typename T>
    __device__ __forceinline__ void V41QuantizeKVRow(const T *src, uint8_t *dst, int dim,
                                                    int quantMode, int blockSize);

    struct V41DecodeTransformDescriptor {
        __nv_bfloat16 *input, *window;
        int rows, dim, position, mode, inverse, quantBlock;
        uint8_t *packedWindow = nullptr;
    };

    template <bool PackForProjection = false>
    __device__ __forceinline__ void V41DecodeTransformRow(
            const V41DecodeTransformDescriptor &item, int row, int d,
            V41RopeTable rope, int ropeDim, int windowSize, uint8_t *codes = nullptr,
            uint8_t *scales = nullptr, int token = 0, int pitch = 0, bool headMajor = false) {
        auto *base = item.input + (size_t)row * item.dim;
        if (item.packedWindow) {
            // Reuse the packed-KV encoder's four-warp layout. Other warps
            // leave the already quantized BF16 input unchanged.
            if (threadIdx.x < kKvQuantThreads)
                V41QuantizeKVRow(base, item.packedWindow +
                    (size_t)(item.position % windowSize) * 528, 512, 1, 32);
            return;
        }
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
            // Match RoPE-only and draft Q: round the sine product before
            // fusing the cosine product. Reversing them can cross a BF16 tie.
            if (item.mode == 0 && (k & 1))
                v = __fmaf_rn(v, c, __fmul_rn(other, s));
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
            if constexpr (PackForProjection) {
                const int group = row / 8;
                const int block = headMajor ? (row % 8) * 16 + d / 32 : d / 32 * 8 + row % 8;
                const __nv_fp8_e4m3 code(V41QuantCodeInput(v, scale, 1));
                codes[(size_t(token * 8 + group) * 128 + block) * 32 + d % 32] = code.__x;
                if (d % 32 == 0)
                    scales[((group * 32 + block / 4) * pitch + token) * 4 + block % 4] = __float_as_uint(scale) >> 23;
                return;
            }
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

    struct V41DecodeIndexDescriptor {
        const __nv_bfloat16 *query, *keys;
        const float *weights;
        float *scores, *blockScores;
        uint8_t *candidates;
        int32_t *output;
        int heads, count, width, position, blocks;
        bool writeCandidates, useCandidates;
        int *candidateBlocks = nullptr, *candidateCount = nullptr;
        int candidateBlockSize = 0;
        const uint8_t *packedKeys = nullptr;
        int keyMode = 0, keyRowBytes = 0, keyVectorUnpack = 0;
    };

    // ---------------- IndexerScore ----------------

    constexpr int kIdxThreads = 128;

    // k8 != nullptr 时 key 来自量化缓存行（打包 FP4 或 FP8），逐元素解码后参与点积
    template <typename QT, typename KT, bool BATCHED = false>
    __global__ void V41IndexerScoreKernel(const QT *q, const float *weights, const KT *k,
                                          int seqlen, int heads, int dim, int m, float *out,
                                          const uint8_t *k8 = nullptr, int kMode = 1, int kRowBytes = 0,
                                          const V41DecodeIndexDescriptor *items = nullptr) {
        if constexpr (BATCHED) {
            const auto &item = items[blockIdx.y];
            q = item.query; weights = item.weights; k = item.keys; out = item.scores;
            heads = item.heads; m = item.count; k8 = item.packedKeys;
            kMode = item.keyMode; kRowBytes = item.keyRowBytes;
            if (blockIdx.x * kIdxThreads >= m) return;
        }
        extern __shared__ float qs[];   // heads * dim + heads
        const int t = BATCHED ? 0 : blockIdx.y; // descriptor pointers already address this query
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
    template<typename QT, typename KT, bool BATCHED = false>
    __global__ void __launch_bounds__(256)
    V41IndexerScoreSmallBatch(const QT *q, const float *weights, const KT *k,
            int seqlen, int m, float *out, const uint8_t *packed,
            int mode, int rowBytes, const V41DecodeIndexDescriptor *items = nullptr) {
        if constexpr (BATCHED) {
            const auto &item = items[blockIdx.y];
            q = item.query; weights = item.weights; k = item.keys; out = item.scores;
            m = item.count; packed = item.packedKeys; mode = item.keyMode; rowBytes = item.keyRowBytes;
            if (blockIdx.x * 8 >= m) return;
        }
        __shared__ float queries[128][33], keys[128][9], dots[32][9];
        const int t=BATCHED ? 0 : blockIdx.y, batch=t/seqlen;
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


    // Preserve every selected block, including threshold ties. Unlike a fixed
    // top-block budget this list cannot silently drop valid candidates.
    // Stable compaction needs one CTA scan when each thread owns an ordered
    // contiguous span, instead of one CTA scan for every 256 mask bytes.
    __global__ void V41CompactIndexCandidates(V41DecodeIndexDescriptor *items, int blockSize, int ratio) {
        using Scan = cub::BlockScan<int, 256>;
        __shared__ typename Scan::TempStorage storage;
        const auto item = items[blockIdx.x];
        const int visible = min(item.count, (item.position + 1) / ratio);
        if (visible <= 16384) {
            if (threadIdx.x == 0) *item.candidateCount = -1;
            return;
        }
        const int blocks = min(item.blocks, (visible + blockSize - 1) / blockSize);
        const int span = ((blocks + 255) / 256 + 3) & ~3;
        const int first = threadIdx.x * span, end = min(blocks, first + span);
        const bool aligned = (reinterpret_cast<uintptr_t>(item.candidates) & 3) == 0;
        int local = 0, j = first;
        if (aligned) {
            for (; j + 4 <= end; j += 4) {
                const unsigned mask = *reinterpret_cast<const unsigned *>(item.candidates + j);
                local += __popc(__vsetne4(mask, 0));
            }
        }
        for (; j < end; ++j) local += item.candidates[j] != 0;
        int before, total;
        Scan(storage).ExclusiveSum(local, before, total);
        if (threadIdx.x == 0) *item.candidateCount = total * blockSize >= visible / 2 ? -1 : total;
        j = first;
        if (aligned) {
            for (; j + 4 <= end; j += 4) {
                const unsigned mask = *reinterpret_cast<const unsigned *>(item.candidates + j);
                if (mask) {
    #pragma unroll
                    for (int d = 0; d < 4; ++d)
                        if ((mask >> (d * 8)) & 255) item.candidateBlocks[before++] = j + d;
                }
            }
        }
        for (; j < end; ++j)
            if (item.candidates[j]) item.candidateBlocks[before++] = j;
    }

    // Decode uses heads as the MMA rows instead of padding one token to the
    // prefill kernel's 64 rows. Keep the dot-product and ordered head-sum
    // operations unchanged; the prefill kernel remains the general fallback.
    __device__ __forceinline__ void V41CopyAsync16(void *dst, const void *src, bool valid);
    template <bool BATCHED = false, bool PERSISTENT = false, bool MULTIROW = false, bool SPARSE = false, bool PREFETCH = false, bool BLOCK_MAX = false>
    __global__ void __launch_bounds__(256)
    V41IndexerScoreDecodeMmaKernel(const __nv_bfloat16 *q, const float *weights,
                                   const __nv_bfloat16 *k, int heads, int m,
                                   int ratio, int startPos, float *out,
                                   const uint8_t *k8, int kMode, int kRowBytes, int vecUnpack,
                                   const V41DecodeIndexDescriptor *items = nullptr, int queryRows = 1) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        __shared__ __nv_bfloat16 ks[PREFETCH ? 2 : 1][64][136];
        __shared__ __nv_bfloat16 qs[PERSISTENT ? 32 : 16][136];
        __shared__ float scores[16][64];
        int b = blockIdx.y;
        const int *candidateBlocks = nullptr;
        int candidateCount = 0, candidateBlockSize = 0, scoreBlocks = 0;
        float *blockScores = nullptr;
        if (BATCHED) {
            const auto item = items[b];
            q = item.query; k = item.keys; weights = item.weights; out = item.scores;
            k8 = item.packedKeys; kMode = item.keyMode; kRowBytes = item.keyRowBytes;
            vecUnpack = item.keyVectorUnpack;
            heads = item.heads; m = item.count; startPos = item.position; b = 0;
            if constexpr (BLOCK_MAX) {
                blockScores = item.blockScores; scoreBlocks = item.blocks;
                candidateBlockSize = item.candidateBlockSize;
            }
            if constexpr (PERSISTENT && SPARSE) {
                if (item.useCandidates && item.candidateBlocks && *item.candidateCount >= 0) {
                    candidateBlocks = item.candidateBlocks;
                    candidateCount = *item.candidateCount;
                    candidateBlockSize = item.candidateBlockSize;
                }
            }
        } else if constexpr (MULTIROW) {
            // Each query keeps decode's head-wise MMA and accumulation order.
            // Keys are shared by rows of the same request, not across requests.
            const int request = b / queryRows, row = b % queryRows;
            q += (size_t)b * heads * 128;
            weights += (size_t)b * heads;
            out += (size_t)b * m;
            if (k) k += (size_t)request * m * 128;
            if (k8) k8 += (size_t)request * m * kRowBytes;
            startPos += row;
            b = 0;
        }
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int visible = ratio > 0 ? min(m, (startPos + 1) / ratio) : m;
        const int end = candidateBlocks ? candidateCount * candidateBlockSize :
            PERSISTENT ? visible : (int)(blockIdx.x + 1) * 64;
        auto keyAt = [&](int offset) {
            return candidateBlocks ? (offset < end ? candidateBlocks[offset / candidateBlockSize] *
                candidateBlockSize + offset % candidateBlockSize : m) : offset;
        };
        if constexpr (BLOCK_MAX) {
            // Verifier rows share the full chunk capacity but have distinct
            // causal ends. Do not expose stale scores from future blocks.
            if (blockIdx.x == 0)
                for (int b = (visible + candidateBlockSize - 1) / candidateBlockSize + tid;
                     b < scoreBlocks; b += 256) blockScores[b] = -INFINITY;
        }
        // Persistent CTAs visit many key tiles with the same query. Retain
        // both 16-head groups in shared memory without changing MMA or the
        // ordered FP32 head accumulation. Larger head counts keep streaming.
        const bool cacheQuery = PERSISTENT && heads <= 32;
        if constexpr (PERSISTENT) {
            if (blockIdx.x * 64 >= end) return;
            if (cacheQuery) {
                for (int v = tid; v < ((heads + 15) / 16) * 256; v += 256)
                    *((float4 *)&qs[v / 16][(v % 16) * 8]) = v / 16 < heads
                        ? *((const float4 *)(q + ((uint64_t)b * heads + v / 16) * 128 + (v % 16) * 8))
                        : make_float4(0, 0, 0, 0);
                __syncthreads();
            }
        }
        const bool pipeline = PREFETCH && k8 == nullptr && end > 16384;
        int stage = 0;
        auto prefetchKeys = [&](int tile, int buffer) {
            for (int v = tid; v < 64 * 16; v += 256) {
                const int j = v / 16, d = (v % 16) * 8, key = keyAt(tile + j);
                const bool valid = key < m;
                V41CopyAsync16(&ks[buffer][j][d],
                    k + ((uint64_t)b * m + (valid ? key : 0)) * 128 + d, valid);
            }
        };
        if constexpr (PREFETCH) {
            if (pipeline) {
                prefetchKeys(blockIdx.x * 64, 0);
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
        }
        for (int j0 = blockIdx.x * 64; j0 < end; j0 += gridDim.x * 64) {
        if (!candidateBlocks && j0 >= visible) {
            if (tid < 64 && j0 + tid < m) out[(uint64_t)b * m + j0 + tid] = -INFINITY;
            return;
        }
        if constexpr (PREFETCH) {
            if (pipeline) {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
                __syncthreads();
                const int next = j0 + gridDim.x * 64;
                if (next < end) {
                    prefetchKeys(next, stage ^ 1);
                    asm volatile("cp.async.commit_group;" ::: "memory");
                }
            }
        }
        if (!pipeline) {
            for (int v = tid; v < 64 * 16; v += 256) {
                const int j = v / 16, d = (v % 16) * 8;
                const int key = keyAt(j0 + j);
                if (key >= m) {
                    *((float4*)&ks[stage][j][d]) = make_float4(0, 0, 0, 0);
                } else if (k8 != nullptr) {
                    const uint8_t *row = k8 + ((uint64_t)b * m + key) * kRowBytes;
                    if (vecUnpack) V41LoadKvRow8Bf16(row, 128, d, kMode, &ks[stage][j][d]);
                    else {
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            ks[stage][j][d + e] = __float2bfloat16_rn(V41LoadKvRow(row, 128, d + e, kMode));
                    }
                } else {
                    *((float4*)&ks[stage][j][d]) = *((const float4*)(k + ((uint64_t)b * m + key) * 128 + d));
                }
            }
        }
        float total = 0.0f;
        for (int h0 = 0; h0 < heads; h0 += 16) {
            __syncthreads();
            const int h = tid / 16, d = (tid % 16) * 8;
            if (!cacheQuery) {
                *((float4*)&qs[h][d]) = h0 + h < heads
                    ? *((const float4*)(q + ((uint64_t)b * heads + h0 + h) * 128 + d))
                    : make_float4(0, 0, 0, 0);
                __syncthreads();
            }
            float acc[4] = {0, 0, 0, 0};
            const int ar = ((lane >> 3) & 1) * 8 + (lane & 7);
            const int ac = (lane >> 4) * 8;
            const int br = warp * 8 + (lane & 7);
            const int bc = ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                uint32_t a[4], bb[2];
                V41LdmX4(a, &qs[(cacheQuery ? h0 : 0) + ar][kk + ac]);
                V41LdmX2(bb, &ks[stage][br][kk + bc]);
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
        if (tid < 64) {
            const int key = keyAt(j0 + tid);
            if (key < m) out[(uint64_t)b * m + key] = total;
        }
        if constexpr (BLOCK_MAX) {
            float maximum = tid < 64 && j0 + tid < visible ? fmaxf(-INFINITY, total) : -INFINITY;
            for (int offset = candidateBlockSize / 2; offset; offset /= 2)
                maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset, candidateBlockSize));
            if (tid < 64 && tid % candidateBlockSize == 0 && j0 + tid < visible) {
                const int block = (j0 + tid) / candidateBlockSize;
                // The newest causal block is pinned in the original algorithm.
                blockScores[block] = block == (visible - 1) / candidateBlockSize ? INFINITY : maximum;
            }
        }
        if constexpr (PERSISTENT) __syncthreads();
        if constexpr (PREFETCH) if (pipeline) stage ^= 1;
        }
#endif
    }

    // BF16 verification queries share old index keys. Keep each row's MMA and
    // ordered FP32 head reduction, while loading each 64-key tile once per pair.
    struct V41PairedIndexShared {
        __nv_bfloat16 keys[64][136];
        __nv_bfloat16 queries[2][32][136];
        float scores[16][64];
        // Reused by every key tile; preserve the FP32 weights exactly.
        float weights[2][32];
    };
    template<bool BLOCK_MAX = false, int THREADS = 256>
    __global__ void __launch_bounds__(THREADS)
    V41IndexerScorePairKernel(const V41DecodeIndexDescriptor *items, int rows, int ratio) {
        static_assert(THREADS == 128 || THREADS == 256, "paired index score supports 128 or 256 threads");
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        constexpr int tilesPerWarp = 64 / (THREADS / 32) / 8;
        extern __shared__ __align__(16) unsigned char storage[];
        auto &sh = *reinterpret_cast<V41PairedIndexShared *>(storage);
        const int first = blockIdx.y * 2, nr = min(2, rows - first);
        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        const auto shared = items[first];
        const auto last = items[first + nr - 1];
        const int end = min(last.count, (last.position + 1) / ratio);
        for (int r = 0; r < nr; ++r) {
            const auto item = items[first + r];
            if constexpr (BLOCK_MAX) {
                const int visible = min(item.count, (item.position + 1) / ratio);
                if (blockIdx.x == 0)
                    for (int b = (visible + item.candidateBlockSize - 1) / item.candidateBlockSize + tid;
                         b < item.blocks; b += THREADS) item.blockScores[b] = -INFINITY;
            }
        }
        if (blockIdx.x * 64 >= end) return;
        for (int r = 0; r < nr; ++r) {
            const auto item = items[first + r];
            if (tid < item.heads) sh.weights[r][tid] = item.weights[tid];
            for (int v = tid; v < ((item.heads + 15) / 16) * 256; v += THREADS)
                *reinterpret_cast<float4 *>(&sh.queries[r][v / 16][v % 16 * 8]) = v / 16 < item.heads
                    ? *reinterpret_cast<const float4 *>(item.query + v / 16 * 128 + v % 16 * 8)
                    : make_float4(0, 0, 0, 0);
        }
        auto loadKeys = [&](int tile) {
            for (int v = tid; v < 64 * 16; v += THREADS) {
                const int key = tile + v / 16, d = v % 16 * 8;
                const bool valid = key < last.count;
                auto *dst = &sh.keys[v / 16][d];
                const auto *src = shared.keys + (size_t)(valid ? key : 0) * 128 + d;
                *reinterpret_cast<float4 *>(dst) = valid
                    ? *reinterpret_cast<const float4 *>(src) : make_float4(0, 0, 0, 0);
            }
        };
        for (int tile = blockIdx.x * 64; tile < end; tile += gridDim.x * 64) {
            loadKeys(tile);
            __syncthreads();
            for (int r = 0; r < nr; ++r) {
                const auto item = items[first + r];
                const int visible = min(item.count, (item.position + 1) / ratio);
                if (tile >= visible) continue;
                float total = 0.f;
                for (int h0 = 0; h0 < item.heads; h0 += 16) {
                    float acc[tilesPerWarp][4] = {};
                    const int ar = ((lane >> 3) & 1) * 8 + (lane & 7), ac = (lane >> 4) * 8;
                    const int bc = ((lane >> 3) & 1) * 8;
    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        uint32_t a[4];
                        V41LdmX4(a, &sh.queries[r][h0 + ar][kk + ac]);
    #pragma unroll
                        for (int u = 0; u < tilesPerWarp; ++u) {
                            uint32_t b[2];
                            V41LdmX2(b, &sh.keys[(warp * tilesPerWarp + u) * 8 + (lane & 7)][kk + bc]);
                            V41MmaBf16(acc[u], a, b);
                        }
                    }
    #pragma unroll
                    for (int u = 0; u < tilesPerWarp; ++u) {
                        const int row = lane >> 2, col = (warp * tilesPerWarp + u) * 8 + (lane & 3) * 2;
                        sh.scores[row][col] = acc[u][0]; sh.scores[row][col + 1] = acc[u][1];
                        sh.scores[row + 8][col] = acc[u][2]; sh.scores[row + 8][col + 1] = acc[u][3];
                    }
                    __syncthreads();
                    if (tid < 64)
                        for (int h = 0; h < min(16, item.heads - h0); ++h)
                            total += fmaxf(sh.scores[h][tid], 0.f) * sh.weights[r][h0 + h];
                    __syncthreads();
                }
                if (tid < 64 && tile + tid < item.count) item.scores[tile + tid] = total;
                if constexpr (BLOCK_MAX) {
                    const int size = item.candidateBlockSize;
                    float maximum = tid < 64 && tile + tid < visible ? fmaxf(-INFINITY, total) : -INFINITY;
                    for (int offset = size / 2; offset; offset /= 2)
                        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset, size));
                    if (tid < 64 && tid % size == 0 && tile + tid < visible) {
                        const int block = (tile + tid) / size;
                        item.blockScores[block] = block == (visible - 1) / size ? INFINITY : maximum;
                    }
                }
            }
            __syncthreads();
        }
    #endif
    }

    template<int THREADS>
    void V41LaunchIndexerScorePair(const V41DecodeIndexDescriptor *items, int rows, int ratio, bool blockMax) {
        const dim3 grid(THREADS == 128 ? 256 : 128, (rows + 1) / 2);
        if (blockMax) {
            V41IndexerScorePairKernel<true, THREADS><<<grid, THREADS,
                sizeof(V41PairedIndexShared), cudaStreamPerThread>>>(items, rows, ratio);
        } else {
            V41IndexerScorePairKernel<false, THREADS><<<grid, THREADS,
                sizeof(V41PairedIndexShared), cudaStreamPerThread>>>(items, rows, ratio);
        }
    }

    // ---------------- radix select（k-th largest） ----------------

    __device__ __forceinline__ unsigned V41FloatKey(float f) {
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }

    constexpr int kSelThreads = 256;

    // Combine equal-bin updates within each warp. Radix passes over positive
    // scores often put an entire warp into one exponent bin; one integer
    // atomic preserves the exact count without serializing all 32 lanes.
    __device__ __forceinline__ void V41HistogramIncrement(unsigned *hist, unsigned bin, bool aggregate = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        if (aggregate) {
            const unsigned peers = __match_any_sync(__activemask(), bin);
            if ((threadIdx.x & 31) == __ffs(peers) - 1)
                atomicAdd(hist + bin, (unsigned)__popc(peers));
        } else {
            atomicAdd(hist + bin, 1u);
        }
#else
        atomicAdd(hist + bin, 1u);
#endif
    }

    // 返回第 k 大的 key（keys 由 getKey(i) 给出，i in [0, n)）。若 n <= k，返回 0。
    template <int THREADS = kSelThreads, typename GetKey>
    __device__ unsigned V41RadixSelect(int n, int k, GetKey getKey, unsigned *hist /* shared, 16 */) {
        unsigned prefix = 0;
        int remaining = k;
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (threadIdx.x < 16) {
                hist[threadIdx.x] = 0;
            }
            __syncthreads();
            unsigned mask = shift == 28 ? 0u : (0xffffffffu << (shift + 4));
            for (int i = threadIdx.x; i < n; i += THREADS) {
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

    // Byte radix and warp-aggregated integer counts preserve the exact FP32
    // threshold, including all boundary ties, without eight full rescans.
    template <int THREADS>
    __device__ unsigned V41CandidateThreshold(const float *values, int n, int keep) {
        using Scan = cub::BlockScan<int, THREADS>;
        __shared__ typename Scan::TempStorage scan;
        __shared__ unsigned bins[256], selected[2];
        unsigned prefix = 0;
        int remaining = keep;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (threadIdx.x < 256) bins[threadIdx.x] = 0;
            __syncthreads();
            const unsigned mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
            for (int i = threadIdx.x; i < n; i += THREADS) {
                const unsigned key = V41FloatKey(values[i]);
                if ((key & mask) == prefix) V41HistogramIncrement(bins, (key >> shift) & 255);
            }
            __syncthreads();
            const int count = threadIdx.x < 256 ? bins[threadIdx.x] : 0;
            int before = 0, total = 0;
            Scan(scan).ExclusiveSum(count, before, total);
            const int greater = total - before - count;
            if (count > 0 && remaining > greater && remaining <= greater + count) {
                selected[0] = threadIdx.x;
                selected[1] = remaining - greater;
            }
            __syncthreads();
            prefix |= selected[0] << shift;
            remaining = selected[1];
            __syncthreads();
        }
        return prefix;
    }

    template <bool BATCHED = false, int THREADS = kSelThreads>
    __global__ void __launch_bounds__(THREADS) V41CandidateSelectKernel(const float *blockScore, int numBlocks, int topkBlocks, uint8_t *mask,
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
        unsigned threshold = numBlocks >= 4096 && keep > 0
            ? V41CandidateThreshold<THREADS>(brow, numBlocks, keep)
            : V41RadixSelect<THREADS>(numBlocks, keep, [&](int i) { return V41FloatKey(brow[i]); }, hist);
        const unsigned negInf = V41FloatKey(-INFINITY);
        // All threshold ties remain eligible, even if more than topkBlocks.
        for (int i = threadIdx.x; i < numBlocks; i += THREADS) {
            const unsigned key = V41FloatKey(brow[i]);
            mrow[i] = key >= threshold && key > negInf;
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
    //   4. 写出阶段：各 warp 独立扫描连续区间，warp 间只做一次前缀和，
    //      输出位置 = 前面 greater 的个数 + min(tieBudget, 前面 tie 的个数)。
    //
    // 共享内存里的候选块下标由 candList（动态共享内存）承载，listCap == 0 表示不压缩。
    constexpr int kTopKSurvivorCap = 1024;

    // Exact two-level selection: each contiguous shard retains at most K
    // indices, then a second selection merges them. Shards and local outputs
    // are index ordered, so equal-score tie breaking remains global-index order.
    constexpr int kTopKShardKeys = 8192;
    constexpr int kTopKParallelMinKeys = 2 * kTopKShardKeys;

    constexpr int kTopKCoarseBins = 1024;
    constexpr int kTopKCoarseStride = kTopKCoarseBins + 1;
    constexpr int kTopKCompactCapacity = 8192;

    struct V41TopKWorkspace {
        int *histograms = nullptr, *counts = nullptr, *offsets = nullptr;
        bool global = false, compactCandidates = false;
        float *scores = nullptr;
        int32_t *indices = nullptr;
        uint8_t *valid = nullptr;
        int parts = 0, width = 0, shardSize = 0;

        size_t Bytes(int rows) const {
            return (size_t)rows * parts * width *
                (sizeof(float) + sizeof(int32_t) + sizeof(uint8_t)) +
                (global ? (size_t)rows * (parts * kTopKCoarseStride + parts + 2) * sizeof(int) :
                 compactCandidates ? (size_t)rows * sizeof(int) : 0);
        }
    };
    V41TopKWorkspace V41TopKLayout(void *memory, int rows, int keys, int width, int blockSize, bool global = false, bool compactCandidates = false) {
        V41TopKWorkspace w;
        w.compactCandidates = compactCandidates;
        w.global = global;
        w.shardSize = ((kTopKShardKeys + blockSize - 1) / blockSize) * blockSize;
        w.parts = (keys + w.shardSize - 1) / w.shardSize;
        w.width = width;
        if (memory) {
            const size_t count = (size_t)rows * w.parts * width;
            w.scores = static_cast<float *>(memory);
            w.indices = reinterpret_cast<int32_t *>(w.scores + count);
            w.valid = reinterpret_cast<uint8_t *>(w.indices + count);
            if (global) {
                w.histograms = reinterpret_cast<int *>(w.valid + count);
                w.counts = w.histograms + (size_t)rows * w.parts * kTopKCoarseStride;
                w.offsets = w.counts + rows;
            } else if (compactCandidates) {
                w.counts = reinterpret_cast<int *>(w.valid + count);
            }
        }
        return w;
    }

    struct V41TopKCounts { int greater, tie; };
    struct V41TopKAddCounts {
        __device__ V41TopKCounts operator()(V41TopKCounts a, V41TopKCounts b) const {
            return {a.greater + b.greater, a.tie + b.tie};
        }
    };

    template <bool BATCHED = false, int LEVEL = 0, int THREADS = kSelThreads>
    __device__ void V41TopKSelect(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int blockShift, int width,
                                  int ratio, int startPos, int listCap, int32_t *out,
                                  const V41DecodeIndexDescriptor *items, V41TopKWorkspace workspace) {
        typedef cub::BlockScan<int, THREADS> BlockScan;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ unsigned hist[256];
        __shared__ unsigned survivors[kTopKSurvivorCap];
        __shared__ int meta[4];             // 0: digit, 1: remaining, 2: 候选块个数
        extern __shared__ int candList[];

        int t = LEVEL == 1 ? blockIdx.y : blockIdx.x;
        const int requestRow = t;
        if (BATCHED) {
            const auto item = items[t];
            score = item.scores; candidates = item.useCandidates ? item.candidates : nullptr;
            m = item.count; numBlocks = item.blocks; width = item.width;
            startPos = item.position; out = item.output; seqlen = 1; t = 0;
            listCap = candidates ? min(numBlocks, listCap) : 0;
        }
        const int i = t % seqlen;
        int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        const uint8_t *cand = candidates == nullptr ? nullptr : candidates + (uint64_t)t * numBlocks;
        int32_t *orow = out + (uint64_t)t * width;
        const unsigned negInf = V41FloatKey(-INFINITY);
        const bool hierarchical = LEVEL && visible > kTopKParallelMinKeys;
        int shardOffset = 0;
        size_t partialBase = (size_t)requestRow * workspace.parts * workspace.width;
        if constexpr (LEVEL == 1) {
            if (!hierarchical) {
                if (blockIdx.x != 0) return;
            } else {
                shardOffset = blockIdx.x * workspace.shardSize;
                if (shardOffset >= visible) return;
                const int firstBlock = shardOffset / blockSize;
                row += shardOffset;
                if (cand) cand += firstBlock;
                visible = min(visible - shardOffset, workspace.shardSize);
                numBlocks = max(0, min(numBlocks - firstBlock,
                    (visible + blockSize - 1) / blockSize));
                listCap = min(listCap, numBlocks);
                partialBase += (size_t)blockIdx.x * workspace.width;
                orow = workspace.indices + partialBase;
            }
        } else if constexpr (LEVEL == 2) {
            if (!hierarchical) return;
            const int parts = (visible + workspace.shardSize - 1) / workspace.shardSize;
            row = workspace.scores + partialBase;
            cand = workspace.valid + partialBase;
            visible = parts * workspace.width;
            if (workspace.counts && workspace.counts[requestRow] >= 0) {
                visible = workspace.counts[requestRow];
                cand = nullptr;
            }
            numBlocks = visible;
            blockSize = 1; blockShift = 0; listCap = 0;
        }

        // ---- 候选块下标的升序压缩 ----
        const bool canCompact = cand != nullptr && listCap > 0 && blockShift >= 0;
        int nCandBlocks = 0;
        if (canCompact) {
            int running = 0;
            for (int st = 0; st < numBlocks; st += THREADS) {
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
        for (int p = threadIdx.x; p < nDomain; p += THREADS) {
            const int j = V41_TOPK_INDEX(p);
            if (V41_TOPK_OK(j)) {
                V41HistogramIncrement(hist, V41FloatKey(row[j]) >> 24, nDomain >= 8192);
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

            // Each thread owns one radix bin. The block prefix already tells
            // us how many keys are greater, so the winning bin needs no serial
            // scan over all 256 bins on thread zero.
            const int greater = total - binPrefix - binCount;
            if (binCount > 0 && width > greater && width <= greater + binCount) {
                meta[0] = threadIdx.x;
                meta[1] = width - greater;
                meta[2] = binCount;
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
                for (int p = threadIdx.x; p < nDomain; p += THREADS) {
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
                    for (int s = threadIdx.x; s < survivorCount; s += THREADS) {
                        const unsigned key = survivors[s];
                        if ((key & mask) == prefix) {
                            V41HistogramIncrement(hist, (key >> shift) & 255, nDomain >= 8192);
                        }
                    }
                } else {
                    for (int p = threadIdx.x; p < nDomain; p += THREADS) {
                        const int j = V41_TOPK_INDEX(p);
                        if (V41_TOPK_OK(j)) {
                            const unsigned key = V41FloatKey(row[j]);
                            if ((key & mask) == prefix) {
                                V41HistogramIncrement(hist, (key >> shift) & 255, nDomain >= 8192);
                            }
                        }
                    }
                }
                __syncthreads();

                const int count = threadIdx.x < 256 ? hist[threadIdx.x] : 0;
                int before = 0, active = 0;
                BlockScan(scanStorage).ExclusiveSum(count, before, active);
                __syncthreads();
                const int greater = active - before - count;
                if (count > 0 && remaining > greater && remaining <= greater + count) {
                    meta[0] = threadIdx.x;
                    meta[1] = remaining - greater;
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

        // Warps own consecutive intervals of the index-ordered domain.
        // Coalesced reads and warp ballots replace one block scan per tile;
        // only the prefix between warps needs a block-wide scan.
        using OutputScan = cub::BlockScan<V41TopKCounts, THREADS>;
        __shared__ typename OutputScan::TempStorage outputStorage;
        const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
        const int span = ((nDomain + THREADS - 1) / THREADS) * 32;
        const int first = warp * span, last = min(nDomain, first + span);
        auto classify = [&](int p) -> V41TopKCounts {
            if (p >= nDomain) return {0, 0};
            const int j = V41_TOPK_INDEX(p);
            if (!V41_TOPK_OK(j)) return {0, 0};
            const unsigned key = V41FloatKey(row[j]);
            if (key <= negInf) return {0, 0};
            if (selectAll || key > threshold) return {1, 0};
            return {0, key == threshold ? 1 : 0};
        };
        V41TopKCounts local{0, 0}, before;
        for (int tile = first; tile < last; tile += 32) {
            const auto flags = classify(tile + lane);
            local.greater += __popc(__ballot_sync(0xffffffffu, flags.greater));
            local.tie += __popc(__ballot_sync(0xffffffffu, flags.tie));
        }
        if (lane != 0) local = {0, 0};
        OutputScan(outputStorage).ExclusiveScan(local, before, V41TopKCounts{0, 0}, V41TopKAddCounts{});
        before.greater = __shfl_sync(0xffffffffu, before.greater, 0);
        before.tie = __shfl_sync(0xffffffffu, before.tie, 0);
        const unsigned lower = (1u << lane) - 1u;
        for (int tile = first; tile < last; tile += 32) {
            const auto flags = classify(tile + lane);
            const unsigned greaterMask = __ballot_sync(0xffffffffu, flags.greater);
            const unsigned tieMask = __ballot_sync(0xffffffffu, flags.tie);
            const int tieBefore = before.tie + __popc(tieMask & lower);
            if (flags.greater || (flags.tie && tieBefore < tieBudget)) {
                const int pos = before.greater + __popc(greaterMask & lower) + min(tieBudget, tieBefore);
                if (pos < width) orow[pos] = V41_TOPK_INDEX(tile + lane);
            }
            before.greater += __popc(greaterMask);
            before.tie += __popc(tieMask);
        }
        for (int p = keep + threadIdx.x; p < width; p += THREADS) {
            orow[p] = -1;
        }
        if constexpr (LEVEL == 1) {
            if (hierarchical) {
                __syncthreads();
                for (int j = threadIdx.x; j < width; j += THREADS) {
                    const int index = orow[j];
                    workspace.valid[partialBase + j] = index >= 0;
                    workspace.scores[partialBase + j] = index >= 0 ? row[index] : 0.f;
                    if (index >= 0) orow[j] = shardOffset + index;
                }
            }
        } else if constexpr (LEVEL == 2) {
            __syncthreads();
            for (int j = threadIdx.x; j < width; j += THREADS) {
                const int index = orow[j];
                if (index >= 0) orow[j] = workspace.indices[partialBase + index];
            }
        }
#undef V41_TOPK_INDEX
#undef V41_TOPK_OK
    }

    // Keep the original launch surface for short rows, uncommon shapes and
    // exact overflow recovery. Global preselection uses the same selector.
    template <bool BATCHED = false, int LEVEL = 0, int THREADS = kSelThreads>
    __global__ void __launch_bounds__(THREADS) V41TopKKernel(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int blockShift, int width,
                                  int ratio, int startPos, int listCap, int32_t *out,
                                  const V41DecodeIndexDescriptor *items, V41TopKWorkspace workspace) {
        V41TopKSelect<BATCHED, LEVEL, THREADS>(score, candidates, seqlen, m, numBlocks, blockSize,
            blockShift, width, ratio, startPos, listCap, out, items, workspace);
    }

    // A monotone FP16 bucket is only a prefilter. Every retained value keeps
    // its FP32 bits; the existing selector resolves the exact boundary and ties.
    // Bound the shortlist and recover with the original hierarchy on overflow.
    struct V41TopKGlobalArgs {
        const float *scores;
        const uint8_t *candidates;
        int seqlen, count, blocks, blockSize, blockShift, width, ratio, position, listCap;
        int32_t *output;
        const V41DecodeIndexDescriptor *items;
    };
    struct V41TopKRow {
        const float *scores;
        int visible, width;
    };
    template <bool BATCHED>
    __device__ V41TopKRow V41GlobalTopKRow(const V41TopKGlobalArgs &a, int row) {
        if constexpr (BATCHED) {
            const auto item = a.items[row];
            return {item.scores, min(item.count, (item.position + 1) / a.ratio), item.width};
        } else {
            return {a.scores + (size_t)row * a.count,
                min(a.count, (a.position + row % a.seqlen + 1) / a.ratio), a.width};
        }
    }
    template <bool BATCHED>
    __device__ bool V41GlobalTopKEligible(const V41TopKGlobalArgs &a, int row, int index) {
        if constexpr (BATCHED) return true; // decode globals do not use candidates
        return !a.candidates || (index / a.blockSize < a.blocks &&
            a.candidates[(size_t)row * a.blocks + index / a.blockSize]);
    }
    __device__ unsigned V41TopKCoarseKey(float value) {
        const unsigned bits = __half_as_ushort(__float2half_rn(value));
        return ((bits & 0x8000) ? (~bits & 65535) : (bits | 0x8000)) >> 6;
    }
    template <bool BATCHED, int LEVEL, int THREADS>
    __device__ void V41GlobalTopKFallback(const V41TopKGlobalArgs &a, V41TopKWorkspace w) {
        V41TopKSelect<BATCHED, LEVEL, THREADS>(a.scores, a.candidates, a.seqlen, a.count,
            a.blocks, a.blockSize, a.blockShift, a.width, a.ratio, a.position,
            a.listCap, a.output, a.items, w);
    }

    template <bool BATCHED>
    __global__ void __launch_bounds__(512) V41TopKGlobalHistogram(V41TopKGlobalArgs a, V41TopKWorkspace w) {
        using Scan = cub::BlockScan<int, 512>;
        __shared__ typename Scan::TempStorage temp;
        __shared__ unsigned hist[kTopKCoarseBins], nonfinite;
        const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
        const auto r = V41GlobalTopKRow<BATCHED>(a, row);
        const int first = part * w.shardSize, end = min(r.visible, first + w.shardSize);
        if (r.visible <= kTopKParallelMinKeys || first >= end) return;
        for (int bin = tid; bin < kTopKCoarseBins; bin += 512) hist[bin] = 0;
        if (tid == 0) nonfinite = 0;
        __syncthreads();
        for (int j = first + tid; j < end; j += 512) {
            if (!V41GlobalTopKEligible<BATCHED>(a, row, j)) continue;
            const float value = r.scores[j];
            if (!isfinite(value)) atomicOr(&nonfinite, 1u);
            atomicAdd(hist + V41TopKCoarseKey(value), 1u);
        }
        __syncthreads();
        int values[2] = {int(hist[tid * 2]), int(hist[tid * 2 + 1])}, before[2], total;
        Scan(temp).ExclusiveSum(values, before, total);
        const size_t base = ((size_t)row * w.parts + part) * kTopKCoarseStride;
        for (int i = 0; i < 2; ++i) w.histograms[base + tid * 2 + i] = total - before[i];
        if (tid == 0) w.histograms[base + kTopKCoarseBins] = nonfinite;
    }

    template <bool BATCHED>
    __global__ void __launch_bounds__(1024) V41TopKGlobalThreshold(V41TopKGlobalArgs a, V41TopKWorkspace w) {
        __shared__ int sums[kTopKCoarseBins + 1], selected, retained, nonfinite;
        using Scan = cub::BlockScan<int, 1024>;
        __shared__ typename Scan::TempStorage temp;
        const int row = blockIdx.x, bin = threadIdx.x;
        const auto r = V41GlobalTopKRow<BATCHED>(a, row);
        if (r.visible <= kTopKParallelMinKeys) {
            if (bin == 0) w.counts[row] = -1;
            return;
        }
        const int parts = (r.visible + w.shardSize - 1) / w.shardSize;
        int count = 0;
        for (int part = 0; part < parts; ++part)
            count += w.histograms[((size_t)row * w.parts + part) * kTopKCoarseStride + bin];
        sums[bin] = count;
        if (bin == 0) { sums[kTopKCoarseBins] = 0; nonfinite = 0; }
        __syncthreads();
        if (bin < parts && w.histograms[((size_t)row * w.parts + bin) * kTopKCoarseStride + kTopKCoarseBins])
            atomicOr(&nonfinite, 1);
        if (sums[0] <= r.width) {
            if (bin == 0) { selected = 0; retained = sums[0]; }
        } else if (count >= r.width && sums[bin + 1] < r.width) {
            selected = bin; retained = count;
        }
        __syncthreads();
        if (bin == 0) w.counts[row] = !nonfinite && retained <= kTopKCompactCapacity &&
            retained <= w.parts * w.width ? retained : -1;
        const int local = bin < parts ?
            w.histograms[((size_t)row * w.parts + bin) * kTopKCoarseStride + selected] : 0;
        int before;
        Scan(temp).ExclusiveSum(local, before);
        if (bin < w.parts) w.offsets[(size_t)row * (w.parts + 1) + bin] = before;
        if (bin == 0) w.offsets[(size_t)row * (w.parts + 1) + w.parts] = selected;
    }

    template <bool BATCHED>
    __global__ void __launch_bounds__(256) V41TopKGlobalCollect(V41TopKGlobalArgs a, V41TopKWorkspace w) {
        const int row = blockIdx.y, part = blockIdx.x;
        const auto r = V41GlobalTopKRow<BATCHED>(a, row);
        if (w.counts[row] < 0) { V41GlobalTopKFallback<BATCHED, 1, 256>(a, w); return; }
        const int first = part * w.shardSize, length = min(r.visible - first, w.shardSize);
        if (length <= 0) return;
        const int threshold = w.offsets[(size_t)row * (w.parts + 1) + w.parts];
        const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
        const int span = ((length + 255) / 256) * 32, start = warp * span, end = min(length, start + span);
        int local = 0;
        for (int tile = start; tile < end; tile += 32) {
            const int j = first + tile + lane;
            const bool keep = tile + lane < length && V41GlobalTopKEligible<BATCHED>(a, row, j) &&
                V41TopKCoarseKey(r.scores[j]) >= threshold;
            local += __popc(__ballot_sync(0xffffffffu, keep));
        }
        using Scan = cub::BlockScan<int, 256>;
        __shared__ typename Scan::TempStorage temp;
        int prefix;
        Scan(temp).ExclusiveSum(lane == 0 ? local : 0, prefix);
        prefix = __shfl_sync(0xffffffffu, prefix, 0);
        const size_t base = (size_t)row * w.parts * w.width + w.offsets[(size_t)row * (w.parts + 1) + part];
        for (int tile = start; tile < end; tile += 32) {
            const int j = first + tile + lane;
            const bool eligible = tile + lane < length && V41GlobalTopKEligible<BATCHED>(a, row, j);
            const float value = eligible ? r.scores[j] : -INFINITY;
            const bool keep = eligible && V41TopKCoarseKey(value) >= threshold;
            const unsigned mask = __ballot_sync(0xffffffffu, keep);
            if (keep) {
                const int pos = prefix + __popc(mask & ((1u << lane) - 1));
                w.scores[base + pos] = value; w.indices[base + pos] = j;
            }
            prefix += __popc(mask);
        }
    }
    template <bool BATCHED>
    __global__ void __launch_bounds__(1024) V41TopKGlobalFinish(V41TopKGlobalArgs a, V41TopKWorkspace w) {
        V41GlobalTopKFallback<BATCHED, 2, 1024>(a, w);
    }
    template <bool BATCHED>
    void V41LaunchGlobalTopK(const V41TopKGlobalArgs &a, V41TopKWorkspace w, int rows) {
        V41TopKGlobalHistogram<BATCHED><<<dim3(w.parts, rows), 512, 0, cudaStreamPerThread>>>(a, w);
        V41TopKGlobalThreshold<BATCHED><<<rows, 1024, 0, cudaStreamPerThread>>>(a, w);
        V41TopKGlobalCollect<BATCHED><<<dim3(w.parts, rows), 256, (size_t)a.listCap * sizeof(int), cudaStreamPerThread>>>(a, w);
        V41TopKGlobalFinish<BATCHED><<<rows, 1024, 0, cudaStreamPerThread>>>(a, w);
    }

    // The score producer already has an index-ordered, causal candidate list.
    // Pack it directly instead of selecting K independently in every shard.
    // Dense masks and oversized tie sets retain the original hierarchy.
    __global__ void __launch_bounds__(256) V41TopKCandidateGather(
        const V41DecodeIndexDescriptor *items, int ratio, int blockSize, int blockShift,
        int listCap, V41TopKWorkspace w) {
        const int row = blockIdx.y, part = blockIdx.x;
        const auto item = items[row];
        const int visible = min(item.count, (item.position + 1) / ratio);
        const int blocks = item.candidateCount ? *item.candidateCount : -1;
        int count = -1;
        if (blocks >= 0 && item.candidateBlocks && visible > kTopKParallelMinKeys) {
            count = blocks * blockSize;
            if (blocks) count -= max(0, (item.candidateBlocks[blocks - 1] + 1) * blockSize - visible);
            if (count > w.parts * w.width) count = -1;
        }
        if (part == 0 && threadIdx.x == 0) w.counts[row] = count;
        if (count < 0) {
            V41TopKSelect<true, 1>(nullptr, nullptr, 1, 0, 0, blockSize, blockShift, 0,
                ratio, 0, listCap, nullptr, items, w);
            return;
        }
        const size_t base = (size_t)row * w.parts * w.width;
        for (int p = part * w.width + threadIdx.x; p < min(count, (part + 1) * w.width); p += 256) {
            const int index = item.candidateBlocks[p / blockSize] * blockSize + p % blockSize;
            w.scores[base + p] = item.scores[index];
            w.indices[base + p] = index;
        }
    }

    struct V41AttentionDescriptor {
        const __nv_bfloat16 *q, *chunkKV, *ringKV, *compressedKV;
        const uint8_t *ringKV8, *compressedKV8;
        const int32_t *indices;
        __nv_bfloat16 *output;
        float *scores;
        int cap, topWidth, startPos, ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack;
        int chunkOffset = 0; // query position relative to the shared verify chunk
    };

    // ---------------- SparseAttention ----------------

    constexpr int kAttnHeadsPerBlock = 32;
    constexpr int kAttnLanes = 8;
    constexpr int kAttnThreads = kAttnHeadsPerBlock * kAttnLanes;   // 256


    template <typename QT, int HeadsPerBlock = kAttnHeadsPerBlock, bool Padded = false,
              int ValueSplits = 1, bool LowRegisters = false, bool BATCHED = false>
    __global__ void __launch_bounds__(HeadsPerBlock * kAttnLanes, LowRegisters ? 2 : 1)
    V41SparseAttentionKernel(const QT *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                             const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                             const int32_t *cmpIdx,
                             const float *sink, int seqlen, int heads, int dim,
                             int windowSize, int cap, int topWidth, int startPos,
                             float scale, __nv_bfloat16 *out,
                             int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes,
                             const V41AttentionDescriptor *descriptors = nullptr,
                             const int *fallbackRows = nullptr) {
        if constexpr (BATCHED) {
            if (fallbackRows && !fallbackRows[blockIdx.x]) return;
            const auto &d = descriptors[blockIdx.x];
            q = d.q; chunkKV = d.chunkKV; ringKV = d.ringKV; compressedKV = d.compressedKV;
            ringKV8 = d.ringKV8; compressedKV8 = d.compressedKV8; cmpIdx = d.indices; out = d.output;
            cap = d.cap; topWidth = d.topWidth; startPos = d.startPos;
            ringMode = d.ringMode; ringRowBytes = d.ringRowBytes; cmpMode = d.cmpMode; cmpRowBytes = d.cmpRowBytes;
        }
        static_assert(HeadsPerBlock % 4 == 0, "8-lane heads must fill complete warps");
        static_assert(ValueSplits > 0 && 64 % ValueSplits == 0, "value slices must partition each lane segment");
        static_assert(!LowRegisters || (std::is_same_v<QT, __nv_bfloat16> && Padded && ValueSplits == 1),
                      "packed queries require the padded BF16 multi-token path");
        constexpr int threads = HeadsPerBlock * kAttnLanes;
        // A 65-float stride gives the eight head lanes distinct shared-memory banks.
        __shared__ float kvRow[512 + (Padded ? 8 : 0)];
        const int t = BATCHED ? 0 : blockIdx.x; // descriptor pointers already address this query
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int chunkStart = BATCHED ? startPos - descriptors[blockIdx.x].chunkOffset : startPos;
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
                if (p >= chunkStart) {
                    src = chunkKV + ((uint64_t)b * seqlen + (p - chunkStart)) * dim;
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



    // Two independent stages; probabilities and online-softmax state stay
    // shared so candidate order, BF16 rounding and MMA accumulation are unchanged.
    template <int VALUE_SPLITS>
    struct V41MmaPvPipelineShared {
        struct Stage {
            __nv_bfloat16 kvs[kMmaNC][kMmaDim / VALUE_SPLITS + 8];
            float sc[kMmaHeads][kMmaNC];
            int valid[kMmaNC];
        } stages[2];
        __nv_bfloat16 ps[kMmaHeads][kMmaPStride];
        float mx[kMmaHeads], lsum[kMmaHeads], alpha[kMmaHeads];
    };

    __device__ __forceinline__ void V41CopyAsync16(void *dst, const void *src, bool valid) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        // Preserve L1 reuse between nearby queries. A zero source size fills
        // masked slots without dereferencing their candidate address.
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
            "r"(shared), "l"(src), "r"(valid ? 16 : 0) : "memory");
    #else
        *static_cast<float4 *>(dst) = valid ? *static_cast<const float4 *>(src) : make_float4(0,0,0,0);
    #endif
    }



    // Independent candidate tiles materialize QK once. The consumer keeps
    // the original 64-slot online softmax and BF16 probability sequence.
    constexpr int kQkPrefetchCandidates = 32;
    struct V41MmaScoresPrefetchShared {
        __nv_bfloat16 qs[kMmaHeads][kMmaKvStride];
        __nv_bfloat16 kvs[kQkPrefetchCandidates][kMmaKvStride];
        float sc[kMmaHeads][kQkPrefetchCandidates];
    };

    // One warp computes eight candidate scores for all eight TP-local draft
    // heads. Keep the two alternating K accumulators used by the generic MMA.
    // Padded score slots are masked by the value kernel.
    template <int Heads>
    __global__ void V41DraftScores(const __nv_bfloat16 *q, const __nv_bfloat16 *current,
            const __nv_bfloat16 *history, const int32_t *indices, float *scores) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int token = blockIdx.x, lane = threadIdx.x;
        const int row = lane / 4, pair = (lane % 4) * 2;
        const int slot = blockIdx.y * 8 + row;
        const int index = slot > 0 && slot < 133 ? indices[token * 132 + slot - 1] : -1;
        const auto *key = slot == 0 ? current + size_t(token) * 512 :
            index >= 0 && index < 133 ? history + (size_t(token / 5) * 133 + index) * 512 : nullptr;
        const int head = blockIdx.z * 16 + row;
        const auto *query = q + (size_t(token) * Heads + head) * 512 + pair;
        float even[4] = {}, odd[4] = {};
        auto step = [&](int k, float (&acc)[4]) {
            uint32_t a[4] = {*reinterpret_cast<const uint32_t *>(query + k), 0,
                *reinterpret_cast<const uint32_t *>(query + k + 8), 0};
            if constexpr (Heads >= 16) {
                a[1] = *reinterpret_cast<const uint32_t *>(query + 8 * 512 + k);
                a[3] = *reinterpret_cast<const uint32_t *>(query + 8 * 512 + k + 8);
            }
            uint32_t b[2] = {key ? *reinterpret_cast<const uint32_t *>(key + k + pair) : 0,
                key ? *reinterpret_cast<const uint32_t *>(key + k + pair + 8) : 0};
            V41MmaBf16(acc, a, b);
        };
#pragma unroll 2
        for (int k = 0; k < 512; k += 32) {
            step(k, even);
            step(k + 16, odd);
        }
        float *out = scores + (size_t(token) * Heads + head) * 192 + blockIdx.y * 8 + pair;
        out[0] = even[0] + odd[0];
        out[1] = even[1] + odd[1];
        if constexpr (Heads >= 16) {
            out[8 * 192] = even[2] + odd[2];
            out[8 * 192 + 1] = even[3] + odd[3];
        }
#endif
    }

    template<bool BATCHED = false, bool PREFETCH = false>
    __global__ void __launch_bounds__(PREFETCH ? kQkPrefetchCandidates * 4 : kMmaThreads)
    V41SparseAttentionScoresKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos,
                                float *scores,
                                int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes, int vecUnpack,
                                const V41AttentionDescriptor *descriptors = nullptr) {
        constexpr int CANDIDATES = PREFETCH ? kQkPrefetchCandidates : kMmaNC;
        constexpr int threads = CANDIDATES * 4, warps = CANDIDATES / 8;
        V41DependencyWait();
        V41DependencyLaunch();
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        if constexpr (BATCHED) {
            const auto &d = descriptors[blockIdx.x];
            q=d.q; chunkKV=d.chunkKV; ringKV=d.ringKV; compressedKV=d.compressedKV;
            cmpIdx=d.indices;
            cap=d.cap; topWidth=d.topWidth; startPos=d.startPos; scores=d.scores;
            if constexpr (!PREFETCH) {
                ringKV8=d.ringKV8; compressedKV8=d.compressedKV8;
                ringMode=d.ringMode; ringRowBytes=d.ringRowBytes; cmpMode=d.cmpMode;
                cmpRowBytes=d.cmpRowBytes; vecUnpack=d.vecUnpack;
            }
            if (blockIdx.z >= ((windowSize + topWidth + kMmaNC - 1) / kMmaNC) * (kMmaNC / CANDIDATES)) return;
        }
        extern __shared__ char v41MmaSharedRaw[];
        using Shared = typename std::conditional<PREFETCH, V41MmaScoresPrefetchShared, V41MmaShared>::type;
        Shared &sh = *reinterpret_cast<Shared*>(v41MmaSharedRaw);

        const int t = BATCHED ? 0 : blockIdx.x;                       // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int chunkStart = BATCHED ? startPos - descriptors[blockIdx.x].chunkOffset : startPos;
        const int h0 = blockIdx.y * kMmaHeads;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;

        for (int v = threadIdx.x; v < kMmaHeads * (kMmaDim / 8); v += threads) {
            int hh = v / (kMmaDim / 8), dv = v % (kMmaDim / 8);
            if constexpr (PREFETCH) {
                const bool valid = h0 + hh < heads;
                V41CopyAsync16(&sh.qs[hh][dv * 8],
                    valid ? q + ((uint64_t)t * heads + h0 + hh) * kMmaDim + dv * 8 : q, valid);
            } else {
                *((float4*)&sh.qs[hh][dv * 8]) = h0 + hh < heads
                    ? *((const float4*)(q + ((uint64_t)t * heads + h0 + hh) * kMmaDim) + dv)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
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
        // The smaller producer tile still writes the original padded
        // 64-candidate score layout; PV's online softmax is unchanged.
        const int tileStart = tile * CANDIDATES;
        if constexpr (!PREFETCH) __syncthreads();
        for (int slot = warp; slot < CANDIDATES; slot += warps) {
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
                    } else if (!PREFETCH && ringKV8 != nullptr) {
                        src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * ringRowBytes;
                        src8Mode = ringMode;
                    } else {
                        src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                    }
                } else {
                    int idx = idxRow[c - winSlots];
                    if (idx < 0 || idx >= cap) {
                        ok = false;
                    } else if (!PREFETCH && compressedKV8 != nullptr) {
                        src8 = compressedKV8 + ((uint64_t)b * cap + idx) * cmpRowBytes;
                        src8Mode = cmpMode;
                    } else {
                        src = compressedKV + ((uint64_t)b * cap + idx) * kMmaDim;
                    }
                }
            }
            if constexpr (PREFETCH) {
                for (int v = lane; v < kMmaDim / 8; v += 32)
                    V41CopyAsync16(&sh.kvs[slot][v * 8], ok ? src + v * 8 : q, ok);
            } else if (!ok) {
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
        // Q and KV use disjoint shared storage until this wait and CTA
        // barrier publish every thread's asynchronous copies to MMA peers.
        if constexpr (PREFETCH) {
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
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

        for (int v = threadIdx.x; v < kMmaHeads * CANDIDATES; v += threads) {
            const int row = v / CANDIDATES, col = v % CANDIDATES;
            if (h0 + row < heads) {
                scores[((uint64_t)t * heads + h0 + row) * tilesTotal * kMmaNC + tileStart + col] = sh.sc[row][col];
            }
        }
#endif
    }

    template <int VALUE_SPLITS, bool MATERIALIZED = false, bool BATCHED = false, bool PIPELINED = false>
    __global__ void __launch_bounds__(kMmaThreads)
    V41SparseAttentionMmaKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, const float *sink, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos, float scale,
                                __nv_bfloat16 *out,
                                int ringMode, int ringRowBytes, int cmpMode, int cmpRowBytes, int vecUnpack,
                                const float *scores = nullptr, const V41AttentionDescriptor *descriptors = nullptr) {
        V41DependencyWait();
        V41DependencyLaunch();
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
        using SerialShared = typename std::conditional<MATERIALIZED, V41MmaPvShared<VALUE_SPLITS>, V41MmaShared>::type;
        using Shared = typename std::conditional<PIPELINED, V41MmaPvPipelineShared<VALUE_SPLITS>, SerialShared>::type;
        static_assert(!PIPELINED || MATERIALIZED, "Pipeline requires materialized QK");
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

        auto preload = [&](int tileIndex) {
            if constexpr (PIPELINED) {
                auto &dst = sh.stages[tileIndex & 1];
                for (int item = threadIdx.x; item < kMmaNC * (loadDim / 16); item += kMmaThreads) {
                    const int slot = item / (loadDim / 16), vector = item % (loadDim / 16);
                    const int c = tileIndex * kMmaNC + slot;
                    const __nv_bfloat16 *src = chunkKV;
                    bool ok = c < totalCand;
                    if (ok) {
                        if (c < winSlots) {
                            const int p = winStart + c;
                            ok = p >= 0 && p <= pos;
                            if (ok) src = p >= chunkStart ?
                                chunkKV + ((uint64_t)b * seqlen + p - chunkStart) * kMmaDim :
                                ringKV + ((uint64_t)b * windowSize + p % windowSize) * kMmaDim;
                        } else {
                            const int idx = idxRow[c - winSlots];
                            ok = idx >= 0 && idx < cap;
                            if (ok) src = compressedKV + ((uint64_t)b * cap + idx) * kMmaDim;
                        }
                    }
                    V41CopyAsync16(&dst.kvs[slot][vector * 16], src + loadBase + vector * 16, ok);
                    V41CopyAsync16(&dst.kvs[slot][vector * 16 + 8], src + loadBase + vector * 16 + 8, ok);
                    if (vector == 0) dst.valid[slot] = ok;
                }
                for (int item = threadIdx.x; item < kMmaHeads * kMmaNC / 4; item += kMmaThreads) {
                    const int row = item / (kMmaNC / 4), col = item % (kMmaNC / 4) * 4;
                    const bool valid = h0 + row < heads;
                    const float *src = valid ?
                        scores + (((uint64_t)t * heads + h0 + row) * tilesTotal + tileIndex) * kMmaNC + col : scores;
                    V41CopyAsync16(&dst.sc[row][col], src, valid);
                }
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
        };
        if constexpr (PIPELINED) preload(0);

        for (int tile = 0; tile < tilesTotal; tile++) {
            auto &tileData = [&]() -> auto & {
                if constexpr (PIPELINED) return sh.stages[tile & 1];
                else return sh;
            }();
            const int tileStart = tile * kMmaNC;
            // wait_group completes each thread's copies; the CTA barrier
            // publishes them to peers and retires reads of the previous stage.
            if constexpr (PIPELINED) asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            if constexpr (PIPELINED) {
                if (tile + 1 < tilesTotal) preload(tile + 1);
            } else if constexpr (MATERIALIZED) {
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
                __nv_bfloat16 *dst = &tileData.kvs[slot][vector * 16];
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
                if (vector == 0) tileData.valid[slot] = ok ? 1 : 0;
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
                        *((float4*)&tileData.kvs[slot][v * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else if (src8 != nullptr) {
                    // 就地解到 BF16 片段再进 mma；FP4 的值在 BF16 上是精确的
                    if (vecUnpack) {
                        // Decode only the current output slice in 16-value groups.
                        for (int v = lane; v < loadDim / 16; v += 32)
                            V41LoadKvRow16Bf16(src8, kMmaDim, loadBase + v * 16, src8Mode, &tileData.kvs[slot][v * 16]);
                    } else {
                        for (int d = lane; d < loadDim; d += 32) {
                            tileData.kvs[slot][d] = __float2bfloat16_rn(V41LoadKvRow(src8, kMmaDim, loadBase + d, src8Mode));
                        }
                    }
                } else {
                    for (int v = lane; v < loadDim / 8; v += 32) {
                        *((float4*)&tileData.kvs[slot][v * 8]) = *((const float4*)(src + loadBase) + v);
                    }
                }
                if (lane == 0) {
                    tileData.valid[slot] = ok ? 1 : 0;
                }
            }
            }
            if constexpr (!PIPELINED) __syncthreads();

            if constexpr (PIPELINED) {
                // Both stage buffers were made visible by the wait + CTA barrier above.
            } else if constexpr (MATERIALIZED) {
                for (int v = threadIdx.x; v < kMmaHeads * kMmaNC; v += kMmaThreads) {
                    const int row = v / kMmaNC, col = v % kMmaNC;
                    tileData.sc[row][col] = h0 + row < heads
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
                        V41LdmX2(bb, &tileData.kvs[qkBRow][k + qkBColBlk]);
                        V41MmaBf16(d0, a, bb);
                        V41LdmX4(a, &sh.qs[qkARow][k + 16 + qkAColBlk]);
                        V41LdmX2(bb, &tileData.kvs[qkBRow][k + 16 + qkBColBlk]);
                        V41MmaBf16(d1, a, bb);
                    }
                    const int r = qkM * 16 + (lane >> 2);
                    const int c = qkN * 8 + (lane & 3) * 2;
                    tileData.sc[r][c] = d0[0] + d1[0];
                    tileData.sc[r][c + 1] = d0[1] + d1[1];
                    tileData.sc[r + 8][c] = d0[2] + d1[2];
                    tileData.sc[r + 8][c + 1] = d0[3] + d1[3];
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
                    bool ok = tileData.valid[col] != 0;
                    float v = ok ? tileData.sc[row][col] * scale : -FLT_MAX;
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
                // Every lane must finish reading the previous maximum before
                // lane 0 overwrites it; shuffle synchronization is not a
                // shared-memory barrier under independent thread scheduling.
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
                        V41LdmX2T(bb, &tileData.kvs[k + pvBRow][(MATERIALIZED ? 0 : valueBase) + pvSlice * kMmaDimSlice + n * 8]);
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

    #include "deepseekv41-prefill-shared-kv.cuh"

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
    template <auto Kernel, typename... Args>
    bool V41LaunchDependent(dim3 grid, dim3 block, size_t shared, Args... args) {
#if CUDART_VERSION >= 12000
        static thread_local int capable[16] = {};
        const int device = FastllmCudaGetDevice();
        bool pdl = false;
        if (device >= 0 && device < 16) {
            if (!capable[device]) {
                int major = 0; cudaFuncAttributes attr{};
                const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                    cudaFuncGetAttributes(&attr, Kernel) == cudaSuccess;
                capable[device] = ok && major >= 9 && attr.binaryVersion >= 90 && attr.ptxVersion >= 90 ? 1 : -1;
            }
            pdl = capable[device] == 1;
        }
        if (!pdl) {
            Kernel<<<grid, block, shared, cudaStreamPerThread>>>(args...);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = block;
        cfg.dynamicSmemBytes = shared; cfg.stream = cudaStreamPerThread;
        cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = &attr; cfg.numAttrs = pdl ? 1 : 0;
        const auto status = cudaLaunchKernelEx(&cfg, Kernel, args...);
        if (status != cudaSuccess) {
            FastllmCudaSetThreadError();
            fprintf(stderr, "[Fastllm] V4.1 dependent launch failed: %s\n", cudaGetErrorString(status));
            return false;
        }
        return true;
#else
        Kernel<<<grid, block, shared, cudaStreamPerThread>>>(args...);
        return cudaPeekAtLastError() == cudaSuccess;
#endif
    }


    void V41LaunchHcDecodeDots(const __nv_bfloat16 *x, const float *fn,
                              int tokens, int flatDim, float *mix) {
        const dim3 grid(25, tokens);
        if (tokens <= 8 && flatDim >= 16384 && flatDim <= 32768 && V41IsSm<9, 0>()) {
            if (tokens <= 3)
                V41HcDecodeDotsTiled<64><<<grid, kHcThreads>>>(x, fn, flatDim, mix);
            else
                V41HcDecodeDotsTiled<16><<<grid, kHcThreads>>>(x, fn, flatDim, mix);
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
    template <int CACHE_PAIRS = 0, bool QUANTIZE_QUERY = true, bool PACK = false, int FixedDim = 0, bool PackedOnly = false>
    __device__ __forceinline__ void V41QKVNormBody(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *qq,
            __nv_bfloat16 *kn, int qdim, int kdim, int qStride, int kStride, float eps, uint8_t *codes = nullptr, uint8_t *scales = nullptr, int pitch = 0, bool useRounded = false, int row = -1) {
        const bool query = FixedDim ? FixedDim == 1280 : (blockIdx.x & 1) == 0;
        const int token = row >= 0 ? row : int(blockIdx.x / 2), dim = FixedDim ? FixedDim : (query ? qdim : kdim);
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
                if (!PackedOnly || !query) output[i] = v;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if constexpr (PACK) {
                    if (query && !useRounded) V41PackNormalizedPair(v, codes, scales, pitch, token, dim / 2, i);
                }
#endif
            }
            if (query && QUANTIZE_QUERY) {
                float maximum = fmaxf(fabsf(__bfloat162float(v.x)), fabsf(__bfloat162float(v.y)));
                // Sixteen lanes hold one contiguous 32-element quantization block.
                for (int offset = 8; offset; offset >>= 1)
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 16));
                const float qs = V41QuantScaleDev(maximum, 1);
                v.x = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(v.x), qs, 1));
                v.y = __float2bfloat16_rn(V41QuantValueDev(__bfloat162float(v.y), qs, 1));
                if (!PackedOnly && i < dim / 2) reinterpret_cast<__nv_bfloat162 *>(qq)[(size_t)token * (dim / 2) + i] = v;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if constexpr (PACK) {
                    if (useRounded && i < dim / 2) V41PackNormalizedPair(v, codes, scales, pitch, token, dim / 2, i);
                }
#endif
            }
        }
    }

    template <int CACHE_PAIRS = 0, bool QUANTIZE_QUERY = true>
    __global__ void V41QKVNormQuantKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *qq,
            __nv_bfloat16 *kn, int qdim, int kdim, int qStride, int kStride, float eps) {
        V41QKVNormBody<CACHE_PAIRS, QUANTIZE_QUERY>(q, kv, qw, kw, qn, qq, kn, qdim, kdim, qStride, kStride, eps);
    }
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    __global__ void V41QKVNormPackForB(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *qq, __nv_bfloat16 *kn,
            int qdim, int kdim, int qStride, int kStride, float eps, uint8_t *codes, uint8_t *scales,
            int pitch, bool useRounded) {
        V41QKVNormBody<2, true, true>(q, kv, qw, kw, qn, qq, kn, qdim, kdim, qStride, kStride, eps,
                                      codes, scales, pitch, useRounded);
    }
    template<bool PackedOnly, bool QueryOnly = false>
    __global__ void V41QKVNormPack1280(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *qq, __nv_bfloat16 *kn,
            int qdim, int kdim, int qStride, int kStride, float eps, uint8_t *codes, uint8_t *scales,
            int pitch, bool useRounded) {
        // The 1280-channel query uses two pair-load rounds; the 512-channel
        // K row uses one. Keep the original sixteen-warp FP32 reduction tree.
        if ((blockIdx.x & 1) == 0)
            V41QKVNormBody<2, true, true, 1280, PackedOnly>(q, kv, qw, kw, qn, qq, kn,
                qdim, kdim, qStride, kStride, eps, codes, scales, pitch, useRounded);
        else if constexpr (!QueryOnly)
            V41QKVNormBody<1, false, false, 512>(q, kv, qw, kw, qn, qq, kn, qdim, kdim, qStride, kStride, eps);
    }
#endif

    // Two physical warps groups reproduce all sixteen logical warp sums in
    // the existing order, with fewer idle threads for the short KV row.
    __global__ void V41QKVNormPackedBf16Kernel(const __nv_bfloat16 *qkv,
            const float *qw, const float *kw, __nv_bfloat16 *qn, __nv_bfloat16 *kn,
            int qdim, int kdim, float eps) {
        V41DependencyWait(); V41DependencyLaunch();
        const bool query = !(blockIdx.x & 1);
        const int token = blockIdx.x / 2, dim = query ? qdim : kdim;
        const auto *input = reinterpret_cast<const __nv_bfloat162*>(qkv + token * (qdim + kdim) + (query ? 0 : qdim));
        auto *output = reinterpret_cast<__nv_bfloat162*>(query ? qn : kn) + token * (dim / 2);
        const float *weight = query ? qw : kw;
        const int tid = threadIdx.x, lane = tid & 31;
        __shared__ float warpSums[16];
        __nv_bfloat162 values[2][2]; float2 weights[2][2];
        float sums[2] = {};
#pragma unroll
        for (int v = 0; v < 2; ++v) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int at = tid + v * 256 + i * 512;
                if (at < dim / 2) {
                    const auto value = input[at]; values[v][i] = value;
                    weights[v][i] = make_float2(weight[2 * at], weight[2 * at + 1]);
                    const float lo = __bfloat162float(value.x), hi = __bfloat162float(value.y);
                    sums[v] += lo * lo + hi * hi;
                }
            }
            for (int offset = 16; offset; offset >>= 1)
                sums[v] += __shfl_down_sync(0xffffffff, sums[v], offset);
            if (!lane) warpSums[v * 8 + tid / 32] = sums[v];
        }
        __syncthreads();
        float sum = lane < 16 ? warpSums[lane] : 0.f;
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        const float scale = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / dim + eps);
#pragma unroll
        for (int v = 0; v < 2; ++v) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int at = tid + v * 256 + i * 512;
                if (at < dim / 2) {
                    auto value = values[v][i]; const auto w = weights[v][i];
                    value.x = __float2bfloat16_rn(__bfloat162float(value.x) * scale * w.x);
                    value.y = __float2bfloat16_rn(__bfloat162float(value.y) * scale * w.y);
                    output[at] = value;
                }
            }
        }
    }

    // ---------------- QuantizeKV ----------------

    // 每个 block 处理一行；blockSize 个线程为一组处理一个 scale 块，lane 对应块内元素。
    // Reuse the fake-quantization grid, including the subnormal E4M3 case
    // where the quantized maximum alone would otherwise halve the scale.
    template <typename T>
    __device__ __forceinline__ void V41QuantizeKVRow(const T *src, uint8_t *dst, int dim,
                                   int quantMode, int blockSize) {
        const int nblocks = dim / blockSize;
        const int groups = kKvQuantThreads / blockSize;
        const int gid = threadIdx.x / blockSize;
        const int lane = threadIdx.x % blockSize;
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

    template <typename T>
    __global__ void V41QuantizeKVKernel(const T *input, uint8_t *output, int rows, int dim,
                                      int quantMode, int blockSize, int rowBytes) {
        const int row = blockIdx.x;
        if (row < rows)
            V41QuantizeKVRow(input + (uint64_t)row * dim,
                            output + (uint64_t)row * rowBytes, dim, quantMode, blockSize);
    }

    struct DsparkAttentionPacket {
        const int *src[4];
        int *dest[4];
        int words[4];
    };

    // A window can be recycled, rewound or replaced at the same position.
    // Exact row comparisons make graph replay independent of request history.
    __global__ void DsparkPackAttention(DsparkAttentionPacket packet, const uint4 *window,
            uint4 *shadow, int *valid, int *changed, int rows) {
        const int row = blockIdx.x, lane = threadIdx.x;
        if (row < rows) {
            constexpr int vectors = 512 * sizeof(__nv_bfloat16) / sizeof(uint4);
            const int idx = row * vectors + lane;
            const bool known = valid[row] != 0;
            bool different = !known;
            uint4 value;
            if (lane < vectors) {
                value = window[idx];
                if (known) {
                    const uint4 old = shadow[idx];
                    different = (value.x != old.x || value.y != old.y || value.z != old.z || value.w != old.w);
                }
            }
            const bool dirty = __syncthreads_or(different);
            if (dirty && lane < vectors) shadow[idx] = value;
            if (lane == 0) {
                valid[row] = 1;
                changed[row] = dirty;
            }
        } else {
            const int block = row - rows, field = block % 4;
            for (int i = (block / 4) * blockDim.x + lane; i < packet.words[field]; i += 8 * blockDim.x)
                packet.dest[field][i] = packet.src[field][i];
        }
    }

    struct DsparkContextStoreArgs {
        const __nv_bfloat162 *kv[3];
        const float2 *norm[3];
        __nv_bfloat162 *ring[3];
    };

    __global__ void DsparkContextStore(DsparkContextStoreArgs a, const int *positions, int batch,
            int rows, int window, int stride, float eps, V41RopeTable rope, int ropeDim) {
        const int row = blockIdx.x, layer = blockIdx.y, b = row / rows, i = row % rows;
        const int count = positions[batch + b];
        if (i >= count || i < count - window) return;
        const int tid = threadIdx.x, lane = tid % 32;
        const auto pair = a.kv[layer][size_t(row) * stride + tid];
        float x = __bfloat162float(pair.x), y = __bfloat162float(pair.y);
        // Match the existing 512-thread RMSNorm: its upper eight warps contribute zero.
        float sum = x * x + y * y;
        for (int k = 16; k; k >>= 1) sum += __shfl_down_sync(0xffffffff, sum, k);
        __shared__ float sums[8];
        if (!lane) sums[tid / 32] = sum;
        __syncthreads();
        sum = lane < 8 ? sums[lane] : 0.f;
        for (int k = 16; k; k >>= 1) sum += __shfl_down_sync(0xffffffff, sum, k);
        const float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / 512.f + eps);
        const auto w = a.norm[layer][tid];
        x = V41StorageRound<__nv_bfloat16>(x * inv * w.x);
        y = V41StorageRound<__nv_bfloat16>(y * inv * w.y);
        const int64_t pos = (int64_t)positions[b] + i;
        if (tid >= (512 - ropeDim) / 2) {
            const float angle = (float)pos * rope.invFreq[tid - (512 - ropeDim) / 2];
            const float cs = cosf(angle), sn = sinf(angle);
            const float re = x * cs - y * sn, im = x * sn + y * cs;
            x = V41StorageRound<__nv_bfloat16>(re);
            y = V41StorageRound<__nv_bfloat16>(im);
        }
        float peak = fmaxf(fabsf(x), fabsf(y));
        for (int k = 8; k; k >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, k, 16));
        const float scale = V41QuantScaleDev(peak, 1);
        a.ring[layer][(size_t(b) * window + pos % window) * 256 + tid] =
            __floats2bfloat162_rn(V41QuantValueDev(x, scale, 1), V41QuantValueDev(y, scale, 1));
    }

    // ---------------- WindowStore ----------------

    __global__ void V41WindowStoreKernel(const uint8_t *chunk, uint8_t *ring, int seqlen, int rowBytes,
                                         int startPos, int windowSize, int firstRow, const int *devicePosition,
                                         const int *deviceCounts, bool batchMetadata) {
        const int b = batchMetadata ? blockIdx.x / seqlen : blockIdx.y;
        const int i = firstRow + (batchMetadata ? blockIdx.x % seqlen : blockIdx.x);
        const int accepted = deviceCounts ? deviceCounts[b] : seqlen;
        if (i >= accepted || (batchMetadata && i < accepted - windowSize)) {
            return;
        }
        const int slot = ((int64_t)(devicePosition ? devicePosition[batchMetadata ? b : 0] : startPos) + i) % windowSize;
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
    #include "deepseekv41-flashinfer-fp16.cuh"
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

extern "C" bool FastllmCudaV41QKVNormPacked(const Data &qkv,
        const Data &qw, const Data &kw, float eps, Data &qn, Data &kn, Data *qq) {
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
    if (!tokens || tokens > uint64_t(std::numeric_limits<int>::max() / 2) ||
        &qn == qq || &qn == &kn || qq == &kn) return false;
    auto overlaps = [](const void *a, size_t na, const void *b, size_t nb) {
        const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
        return a && b && (x <= y ? y - x < na : x - y < nb);
    };
    for (Data *out : {&qn, qq, &kn}) {
        if (!out) continue;
        const size_t bytes = tokens * (out == &kn ? kdim : qdim) * 2;
        if ((uintptr_t)out->cudaData % 4) return false;
        for (const Data *src : {&qkv, &qw, &kw})
            if (out == src || overlaps(out->cudaData, bytes, src->cudaData, src->GetBytes())) return false;
    }
    const size_t qb = tokens * qdim * 2, kb = tokens * kdim * 2;
    if (overlaps(qn.cudaData, qb, kn.cudaData, kb) || (qq &&
        (overlaps(qn.cudaData, qb, qq->cudaData, qb) ||
         overlaps(qq->cudaData, qb, kn.cudaData, kb)))) return false;
    auto qdims = qkv.dims, kdims = qkv.dims;
    qdims.back() = qdim; kdims.back() = kdim;
    if (!V41PrepareOutput(qn, DataType::BFLOAT16, qdims) ||
        (qq && !V41PrepareOutput(*qq, DataType::BFLOAT16, qdims)) ||
        !V41PrepareOutput(kn, DataType::BFLOAT16, kdims)) return false;
    if (!qq && qdim <= 2048 && kdim <= 2048) {
        if (!V41LaunchDependent<V41QKVNormPackedBf16Kernel>(tokens * 2, 256, 0,
            (const __nv_bfloat16*)qkv.cudaData, (const float*)qw.cudaData, (const float*)kw.cudaData,
            (__nv_bfloat16*)qn.cudaData, (__nv_bfloat16*)kn.cudaData, qdim, kdim, eps)) return false;
        return V41CheckLaunch("QKVNormPackedBf16");
    }
#define V41_PACKED_NORM(C, Q) V41QKVNormQuantKernel<C, Q><<<tokens * 2, 512>>>( \
        (const __nv_bfloat16*)qkv.cudaData, (const __nv_bfloat16*)qkv.cudaData + qdim, \
        (const float*)qw.cudaData, (const float*)kw.cudaData, (__nv_bfloat16*)qn.cudaData, \
        Q ? (__nv_bfloat16*)qq->cudaData : nullptr, (__nv_bfloat16*)kn.cudaData, qdim, kdim, stride, stride, eps)
    if (qdim <= 2048 && kdim <= 2048) {
        if (qq) V41_PACKED_NORM(2, true); else V41_PACKED_NORM(2, false);
    } else {
        if (qq) V41_PACKED_NORM(0, true); else V41_PACKED_NORM(0, false);
    }
#undef V41_PACKED_NORM
    return V41CheckLaunch("QKVNormQuantPacked");
}

extern "C" bool FastllmCudaV41QKVNormQuantPacked(const Data &qkv,
        const Data &qw, const Data &kw, float eps, Data &qn, Data &qq, Data &kn) {
    return FastllmCudaV41QKVNormPacked(qkv, qw, kw, eps, qn, kn, &qq);
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
    if (tokens < 1 || tokens > uint64_t(std::numeric_limits<int>::max() / 2) ||
        kdims != kv.dims || !std::isfinite(eps) || eps < 0) return false;
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

static bool V41RouterSelectImpl(const Data &input, Data &weight,
        Data &bias, float temperature, float routeScale, Data &logits,
        Data &indices, Data &scores, int topk, bool ownerDecode, const FastllmDsv41InputPacket *packet = nullptr) {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    if (!V41OnCuda(input) || input.multiDeviceData || weight.multiDeviceData || bias.multiDeviceData ||
        input.dataType != DataType::BFLOAT16 || weight.dataType != DataType::FLOAT32 ||
        bias.dataType != DataType::FLOAT32 || input.dims.empty() || input.dims.back() <= 0 ||
        weight.dims.size() != 2 || !((weight.dims[0] == 384 && topk == 6) || (weight.dims[0] == 128 && topk == 3)) || weight.dims[1] != input.dims.back() ||
        weight.dims[1] % 128 || bias.Count(0) != (uint64_t)weight.dims[0] || !std::isfinite(temperature) || temperature <= 0 ||
        &input == &logits || &input == &indices || &input == &scores || &weight == &logits ||
        &bias == &logits || &weight == &indices || &weight == &scores ||
        &bias == &indices || &bias == &scores || &logits == &indices || &logits == &scores || &indices == &scores) return false;
    const int experts = weight.dims[0];
    const int rows = input.Count(0) / input.dims.back();
    const int arch = FastllmCudaRuntimeArch();
    if (packet && (!ownerDecode || (rows != 6 && rows < 16) || arch != 103 || experts != 384 || weight.dims[1] != 5120))
        return false;
    // Keep tall prefill matrices on GEMM; this split-K pipeline targets
    // batches whose row count is comparable to the small router width.
    if (rows < 5 || rows > 2 * experts || arch < 80) return false;
    const int device = FastllmCudaGetDevice();
    if (input.dataDeviceIds.empty() || input.dataDeviceIds.front() != device) return false;
    weight.ToDevice(DataDevice::CUDA, {device}, true);
    Data *compact = FastllmCudaExactRouterBFloat16Weight(weight);
    if (!compact) return false;
    bias.ToDevice(DataDevice::CUDA, {device}, true);
    auto dims = input.dims; dims.back() = experts;
    if (!V41PrepareOutput(logits, DataType::FLOAT32, dims) ||
        !V41PrepareOutput(indices, DataType::INT32, {rows, topk}) ||
        !V41PrepareOutput(scores, DataType::FLOAT32, {rows, topk})) return false;
    // Inspect the compile target as well as hardware: a compute_75/80 PTX
    // fallback lacks the Tensor Core / SM90 instructions of newer targets.
    static thread_local std::map<int, int> compiledTargets;
    auto target = compiledTargets.find(device);
    if (target == compiledTargets.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        cudaFuncAttributes attrs = {};
        if (cudaFuncGetAttributes(&attrs, V41TcRouterPipeline<8,8,640,1,true>) != cudaSuccess) return false;
        target = compiledTargets.emplace(device, attrs.ptxVersion).first;
    }
    if (target->second < 80) return false;
    const float invTemp = fabsf(temperature - 1.f) > 1e-6f ? 1.f / temperature : 1.f;
#if !defined(USE_ROCM) && CUBLAS_VERSION >= 11000
    // Score all real rows in one FP32-accumulating GEMM. Fuse temperature,
    // softplus/sqrt, bias and top-k in the consumer while retaining logits.
    if (ownerDecode && arch == 103 && rows >= 16 && experts == 384 && weight.dims[1] == 5120) {
        auto handle = getFastllmCublasFp32ReductionHandle();
        const float alpha = 1.f, beta = 0.f;
        if (handle) {
            const auto gemm = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                experts, rows, weight.dims[1], &alpha, compact->cudaData, CUDA_R_16BF, weight.dims[1],
                input.cudaData, CUDA_R_16BF, weight.dims[1], &beta, logits.cudaData, CUDA_R_32F, experts,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (gemm == CUBLAS_STATUS_SUCCESS) {
                if (packet) {
                    V41TcRouterTopK<384, 6, true, true><<<rows, 32, 0, cudaStreamPerThread>>>(
                        (float *)logits.cudaData, (const float *)bias.cudaData, (int *)indices.cudaData,
                        (float *)scores.cudaData, routeScale, invTemp, packet->routeIds, packet->routeScores, packet->routeOrder);
                } else {
                    V41TcRouterTopK<384, 6, true><<<rows, 32, 0, cudaStreamPerThread>>>(
                        (float*)logits.cudaData, (const float*)bias.cudaData, (int*)indices.cudaData,
                        (float*)scores.cudaData, routeScale, invTemp);
                }
                return V41CheckLaunch("RouterGemmTopK");
            }
            if (gemm != CUBLAS_STATUS_NOT_SUPPORTED && gemm != CUBLAS_STATUS_ARCH_MISMATCH &&
                gemm != CUBLAS_STATUS_ALLOC_FAILED)
                throw std::runtime_error("DeepSeek V4.1 router cuBLAS failure " + std::to_string(int(gemm)));
        }
    }
#endif
    const bool verify = ownerDecode && rows == 6 && experts == 384 && weight.dims[1] == 5120 &&
                        arch == 103 && target->second >= 90;
    if (packet && !verify) return false;
    const bool pdl = arch >= 90 && target->second >= 90;
    const bool cluster = pdl && arch < 100 && weight.dims[1] == 5120;
    // Bound persistent split-K scratch, independently of the logical batch.
    // Every graph for this thread/device retains the same stable allocation.
    constexpr size_t partialBudget = 1024 * 1024;
    const int chunkRows = int(partialBudget / (16 * experts * sizeof(float))) / 8 * 8;
    float *partial = nullptr;
    if (!cluster) {
        static thread_local std::map<int, std::unique_ptr<Data>> scratch;
        auto found = scratch.find(device);
        if (found == scratch.end()) {
            if (FastllmCudaGraphIsCapturingFast()) return false;
            auto buffer = std::make_unique<Data>(DataType::FLOAT32, std::vector<int>{int(partialBudget / sizeof(float))});
            buffer->ToDevice(DataDevice::CUDA, {device}, false); buffer->Allocate(false);
            found = scratch.emplace(device, std::move(buffer)).first;
        }
        partial = (float*)found->second->cudaData;
    }
    cudaError_t status;
    for (int first = 0; first < rows; first += chunkRows) {
        const int count = std::min(chunkRows, rows - first);
        const auto *x = (const __nv_bfloat16*)input.cudaData + size_t(first) * weight.dims[1];
        auto *y = (float*)logits.cudaData + size_t(first) * experts;
        if (cluster) status = V41TcLaunchPipeline<8,8,640,1,true>(
            x, (const __nv_bfloat16*)compact->cudaData, partial, y, count, experts,
            weight.dims[1], invTemp, cudaStreamPerThread, pdl);
        else status = V41TcLaunchPipeline<4,16,128,2,false>(
            x, (const __nv_bfloat16*)compact->cudaData, partial, y, count, experts,
            weight.dims[1], invTemp, cudaStreamPerThread, pdl, verify);
        if (status != cudaSuccess) { cudaGetLastError(); return false; }
    }
    auto select = topk == 3 ? V41TcRouterTopK<128, 3> : V41TcRouterTopK<384, 6>;
#if CUDART_VERSION >= 12000
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = rows; cfg.blockDim = 32; cfg.stream = cudaStreamPerThread;
    cudaLaunchAttribute attr = {}; attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1; cfg.attrs = &attr; cfg.numAttrs = pdl ? 1 : 0;
    if (verify) {
        cfg.blockDim = experts;
        auto finish = packet ? V41VerifyRouterFinishTopK<true> : V41VerifyRouterFinishTopK<false>;
        status = cudaLaunchKernelEx(&cfg, finish, (const float *)partial, (float *)logits.cudaData,
            (const float *)bias.cudaData, (int *)indices.cudaData, (float *)scores.cudaData, rows, invTemp,
            routeScale, packet ? packet->routeIds : nullptr, packet ? packet->routeScores : nullptr,
            packet ? packet->routeOrder : nullptr);
        if (status != cudaSuccess) { cudaGetLastError(); return false; }
        return V41CheckLaunch("VerifyRouterFinishTopK");
    }
    status = cudaLaunchKernelEx(&cfg, select, (float*)logits.cudaData,
        (const float*)bias.cudaData, (int*)indices.cudaData, (float*)scores.cudaData, routeScale, invTemp, (int64_t *)nullptr, (float *)nullptr, (int *)nullptr);
    if (status != cudaSuccess) { cudaGetLastError(); return false; }
    return V41CheckLaunch("RouterTopK");
#else
    select<<<rows,32,0,cudaStreamPerThread>>>((float*)logits.cudaData,
        (const float*)bias.cudaData, (int*)indices.cudaData, (float*)scores.cudaData, routeScale, invTemp);
    return V41CheckLaunch("RouterTopK");
#endif
#endif
}

extern "C" bool FastllmCudaV41RouterSelectTopK(const Data &input, Data &weight,
        Data &bias, float temperature, float routeScale, Data &logits,
        Data &indices, Data &scores, int topk) {
    return V41RouterSelectImpl(input, weight, bias, temperature, routeScale, logits, indices, scores, topk, false);
}

bool FastllmCudaV41OwnerRouterPacket(const Data &input, Data &weight, Data &bias, float temperature,
        float routeScale, Data &logits, Data &indices, Data &scores, const FastllmDsv41InputPacket &packet) {
    return packet.routeIds && packet.routeScores &&
        V41RouterSelectImpl(input,weight,bias,temperature,routeScale,logits,indices,scores,6,true,&packet);
}
extern "C" bool FastllmCudaV41OwnerRouterSelect(const Data &input, Data &weight, Data &bias,
        float temperature, float routeScale, Data &logits, Data &indices, Data &scores) {
    return V41RouterSelectImpl(input, weight, bias, temperature, routeScale, logits, indices, scores, 6, true);
}

extern "C" bool FastllmCudaV41RouterSelect(const Data &input, Data &weight,
        Data &bias, float temperature, float routeScale, Data &logits,
        Data &indices, Data &scores) {
    return FastllmCudaV41RouterSelectTopK(input, weight, bias, temperature, routeScale,
                                       logits, indices, scores, 6);
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
    const int rows = input.Count(0) / input.dims.back();
    if (weight.dims[0] == 384 && weight.dims[1] == 5120 && V41IsSm<10, 3>() &&
        FastllmV41HcEnabledAtRows("FASTLLM_DSV41_VECTOR_ROUTER", rows) &&
        !(reinterpret_cast<uintptr_t>(input.cudaData) & 7)) {
        if (Data *compact = FastllmCudaExactRouterBFloat16Weight(weight)) {
            auto *x = static_cast<const __nv_bfloat16 *>(input.cudaData);
            auto *w = static_cast<const __nv_bfloat16 *>(compact->cudaData);
            auto *y = static_cast<float *>(scores.cudaData);
            if (!(reinterpret_cast<uintptr_t>(w) & 7)) {
                if (rows <= 2)
                    V41RouterVector5120<128,1><<<dim3(384, rows), 128, 0, cudaStreamPerThread>>>(x, w, y, 384, invTemp);
                else
                    V41RouterVector5120<64,1><<<dim3(384, rows), 64, 0, cudaStreamPerThread>>>(x, w, y, 384, invTemp);
                return V41CheckLaunch("RouterVector5120");
            }
        }
    }
    if (rows >= 2 && weight.dims[0] >= 256 && weight.dims[1] >= 1024) {
        const int arch = FastllmCudaRuntimeArch();
        const bool streaming = arch >= 90 && arch < 100;
#define V41_ROUTER_ROWS(M) case M: \
        if (streaming) V41RouterScoresRowsKernel<M, true><<<weight.dims[0], 256>>>( \
            (const __nv_bfloat16*)input.cudaData, (const float*)weight.cudaData, \
            (float*)scores.cudaData, weight.dims[0], weight.dims[1], invTemp); \
        else V41RouterScoresRowsKernel<M, false><<<weight.dims[0], 256>>>( \
            (const __nv_bfloat16*)input.cudaData, (const float*)weight.cudaData, \
            (float*)scores.cudaData, weight.dims[0], weight.dims[1], invTemp); \
        break
        switch (rows) {
            V41_ROUTER_ROWS(2); V41_ROUTER_ROWS(3); V41_ROUTER_ROWS(4);
            V41_ROUTER_ROWS(5); V41_ROUTER_ROWS(6); V41_ROUTER_ROWS(7); V41_ROUTER_ROWS(8);
        }
#undef V41_ROUTER_ROWS
        return V41CheckLaunch("RouterScoresRows");
    }
    // Checkpoint router weights are BF16 promoted to FP32. Compact only
    // exactly representable weights; retain every FP32 FMA/reduction step.
    if (rows == 1 && (V41IsSm<12, 0>() || V41IsSm<10, 3>())) {
        if (Data *compact = FastllmCudaExactRouterBFloat16Weight(weight)) {
            V41RouterScoresKernel<1, __nv_bfloat16><<<weight.dims[0], 256>>>(
                (const __nv_bfloat16*)input.cudaData, (const __nv_bfloat16*)compact->cudaData,
                (float*)scores.cudaData, weight.dims[1], weight.dims[0], invTemp, rows);
            return V41CheckLaunch("RouterScoresCompact");
        }
    }
    if (rows == 1)
        V41RouterScoresKernel<1><<<dim3(weight.dims[0],rows),256>>>(
            (const __nv_bfloat16 *)input.cudaData,(const float *)weight.cudaData,
            (float *)scores.cudaData,weight.dims[1],weight.dims[0],invTemp,rows);
    else
        V41RouterScoresKernel<2><<<dim3(weight.dims[0],(rows+1)/2),256>>>(
            (const __nv_bfloat16 *)input.cudaData,(const float *)weight.cudaData,
            (float *)scores.cudaData,weight.dims[1],weight.dims[0],invTemp,rows);
    return V41CheckLaunch("RouterScores");
}

// Small BF16 decode batches retain the original FP32 dot accumulation;
// finish and normalization use independent CTAs in one launch, preserving arithmetic.
static bool V41MixPreNormImpl(const Data &x, const Data &previousPre, Data &fn,
        Data &scale, Data &base, Data &normWeight, int sinkhornIters, float mixEps,
        float normEps, Data &pre, Data &post, Data &comb, Data &output, Data *quantizedOutput, FastllmDsparkHcBranch *branch) {
    if (!V41OnCuda(x) || !V41OnCuda(previousPre) || previousPre.dataType != DataType::FLOAT32 ||
        x.multiDeviceData || previousPre.multiDeviceData || x.dataType != DataType::BFLOAT16 ||
        x.dims.size() != 4 || x.dims[2] != 4 || x.dims[0] * x.dims[1] < 1 ||
        x.dims[0] * x.dims[1] > 192 || x.dims[3] < 4096 || x.dims[3] > 8192 ||
        (x.dims[3] & 1) || fn.dataType != DataType::FLOAT32 ||
        scale.dataType != DataType::FLOAT32 || base.dataType != DataType::FLOAT32 ||
        normWeight.dataType != DataType::FLOAT32 || normWeight.Count(0) != x.dims[3]) return false;
    const int b = x.dims[0], n = x.dims[1], d = x.dims[3], tokens = b * n;
    if (quantizedOutput != nullptr && (d % 32 != 0 || quantizedOutput == &x || quantizedOutput == &output)) return false;
    if (previousPre.Count(0) != (uint64_t)tokens * 4 || fn.Count(0) != (uint64_t)24 * 4 * d ||
        scale.Count(0) != 3 || base.Count(0) != 24) return false;
    for (Data *w : {&fn, &scale, &base, &normWeight})
        w->ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()}, true);
    if (branch && (!quantizedOutput || d != 5120 || tokens < 1 || tokens > 8 || !V41IsSm<10, 3>())) return false;
    Data localScratch;
    Data &scratch = branch ? branch->scratch : localScratch;
    const bool splitDots = tokens <= 4 && d == 5120 && V41IsSm<10, 3>();
    const bool tensor = V41HcTensorCandidate(x, tokens);
    if (tokens >= 16 && !tensor) return false;
    if (!V41PrepareOutput(scratch, DataType::FLOAT32,
            {tensor ? (tokens*25+3)/4*4 + tokens*48 : tokens*(splitDots ? 8*25 : 25)}) ||
        !V41PrepareOutput(pre, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(post, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(comb, DataType::FLOAT32, {b, n, 4, 4}) ||
        !V41PrepareOutput(output, DataType::BFLOAT16, {b, n, d})) return false;
    if (quantizedOutput != nullptr &&
        !V41PrepareOutput(*quantizedOutput, DataType::BFLOAT16, {b, n, d})) return false;
    if (branch) {
        auto stream = static_cast<cudaStream_t>(branch->stream);
        FastllmCudaEventRecordCurrentThread(branch->begin);
        FastllmCudaStreamWaitEvent(stream, branch->begin);
        const auto *input = static_cast<const __nv_bfloat16 *>(x.cudaData);
        auto *mix = static_cast<float *>(scratch.cudaData);
        // Each token keeps the same 256-lane accumulation and warp reduction tree.
        V41HcDecodeDots<<<dim3(25, tokens), kHcThreads, 0, stream>>>(input,
            static_cast<const float *>(fn.cudaData), 4 * d, mix);
        V41HcDecodeFinish<<<tokens, 32, 0, stream>>>(mix, (const float *)scale.cudaData,
            (const float *)base.cudaData, 4*d, sinkhornIters, mixEps, normEps,
            (float *)pre.cudaData, (float *)post.cudaData, (float *)comb.cudaData);
        FastllmCudaEventRecord(branch->done, stream);
        V41HcPreNormKernel<1024,3,false,true><<<tokens,1024>>>(input, (const float *)previousPre.cudaData,
            (const float *)normWeight.cudaData, (__nv_bfloat16 *)output.cudaData, 4, d, normEps,
            nullptr, nullptr, nullptr, 0, 0.f, nullptr, nullptr, nullptr,
            (__nv_bfloat16 *)quantizedOutput->cudaData);
        return V41CheckLaunch("DsparkHcFork");
    }
    if (tokens == 1 && d == 5120 && V41IsSm<12, 0>()) {
        const auto kernel = quantizedOutput ? V41HcDotsPreNormKernel<true> : V41HcDotsPreNormKernel<false>;
        kernel<<<26, 1024>>>((const __nv_bfloat16*)x.cudaData, (const float*)fn.cudaData,
            (float*)scratch.cudaData, (const float*)previousPre.cudaData, (const float*)normWeight.cudaData,
            (__nv_bfloat16*)output.cudaData,
            quantizedOutput ? (__nv_bfloat16*)quantizedOutput->cudaData : nullptr, normEps);
        V41HcDecodeFinish<<<tokens, 32>>>((const float*)scratch.cudaData,
            (const float*)scale.cudaData, (const float*)base.cudaData, 4 * d,
            sinkhornIters, mixEps, normEps, (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
        return V41CheckLaunch("HcDotsPreNorm");
    }
    if (tokens >= 16) {
        if (!V41HcTensorDots(x, fn, tokens, (float *)scratch.cudaData)) return false;
    } else if (splitDots) {
        V41HcWarpDots5120<<<dim3(25, tokens, 8), 32>>>(
            (const __nv_bfloat16 *)x.cudaData, (const float *)fn.cudaData, (float *)scratch.cudaData);
    } else {
        V41LaunchHcDecodeDots((const __nv_bfloat16 *)x.cudaData, (const float *)fn.cudaData,
                             tokens, 4 * d, (float *)scratch.cudaData);
    }
#define V41_MIX_NORM_W(I, Q, W) V41HcPreNormKernel<1024, I, true, Q, W><<<tokens * 2, 1024>>>( \
        (const __nv_bfloat16 *)x.cudaData, (const float *)previousPre.cudaData, \
        (const float *)normWeight.cudaData, (__nv_bfloat16 *)output.cudaData, 4, d, normEps, \
        (const float *)scratch.cudaData, (const float *)scale.cudaData, (const float *)base.cudaData, \
        sinkhornIters, mixEps, (float *)post.cudaData, (float *)comb.cudaData, (float *)pre.cudaData, \
        Q ? (__nv_bfloat16 *)quantizedOutput->cudaData : nullptr)
#define V41_MIX_NORM(I, Q) V41_MIX_NORM_W(I, Q, 1)
    if (splitDots) {
        if (quantizedOutput) V41_MIX_NORM_W(3, true, 8);
        else V41_MIX_NORM_W(3, false, 8);
    } else if (d <= 4096) { if (quantizedOutput) V41_MIX_NORM(2, true); else V41_MIX_NORM(2, false); }
    else if (d <= 6144) { if (quantizedOutput) V41_MIX_NORM(3, true); else V41_MIX_NORM(3, false); }
    else { if (quantizedOutput) V41_MIX_NORM(4, true); else V41_MIX_NORM(4, false); }
#undef V41_MIX_NORM
#undef V41_MIX_NORM_W
    return V41CheckLaunch("HcMixPreNorm");
}

extern "C" bool FastllmCudaDeepSeekV41HcMixPreNorm(const Data &x, const Data &previousPre, Data &fn,
        Data &scale, Data &base, Data &normWeight, int sinkhornIters, float mixEps,
        float normEps, Data &pre, Data &post, Data &comb, Data &output, Data *quantizedOutput) {
    return V41MixPreNormImpl(x, previousPre, fn, scale, base, normWeight, sinkhornIters, mixEps, normEps, pre, post, comb, output, quantizedOutput, nullptr);
}

extern "C" bool FastllmCudaDsparkHcFork(const Data &x, const Data &previousPre, Data &fn,
        Data &scale, Data &base, Data &normWeight, int sinkhornIters, float mixEps,
        float normEps, Data &pre, Data &post, Data &comb, Data &output, Data *quantizedOutput, FastllmDsparkHcBranch &branch) {
    return V41MixPreNormImpl(x, previousPre, fn, scale, base, normWeight, sinkhornIters, mixEps, normEps, pre, post, comb, output, quantizedOutput, &branch);
}

// Fused attention post + next mHC projection, followed by the existing
// delayed-pre normalization. Rejection is side-effect free for input tensors.
static bool V41HcPostMixPreNormImpl(const Data &input, const Data &residual,
        const Data &previousPre, const Data &previousPost, const Data &previousComb,
        const Data &fn, const Data &scale, const Data &base, const Data &normWeight,
        int sinkhornIters, float mixEps, float normEps, Data &hidden,
        Data &pre, Data &post, Data &comb, Data &output, Data *quantizedOutput) {
    if (residual.dataType != DataType::BFLOAT16 || residual.dims.size() != 4 ||
        residual.dims[2] != 4 || residual.dims[0] <= 0 || residual.dims[1] <= 0) return false;
    const int64_t tokens64 = (int64_t)residual.dims[0] * residual.dims[1];
    const int d = residual.dims[3];
    if (tokens64 < 1 || (tokens64 == 1 && (d != 5120 || !V41IsSm<10, 3>())) ||
        tokens64 > 8 || d < 4096 || d > 8192 || d % 1024) return false;
    const int tokens = (int)tokens64, b = residual.dims[0], n = residual.dims[1];
    const Data *inputs[] = {&input, &residual, &previousPre, &previousPost, &previousComb,
                           &fn, &scale, &base, &normWeight};
    const std::vector<int> device{FastllmCudaGetDevice()};
    for (const Data *v : inputs) {
        if (!V41OnCuda(*v) || v->multiDeviceData || v->dataDeviceIds != device ||
            v->dataType != (v == &input || v == &residual ? DataType::BFLOAT16 : DataType::FLOAT32))
            return false;
    }
    if (input.Count(0) != (uint64_t)tokens * d || previousPre.Count(0) != tokens * 4 ||
        previousPost.Count(0) != tokens * 4 || previousComb.Count(0) != tokens * 16 ||
        fn.Count(0) != (uint64_t)24 * 4 * d || scale.Count(0) != 3 ||
        base.Count(0) != 24 || normWeight.Count(0) != d) return false;
    Data *outputs[] = {&hidden, &pre, &post, &comb, &output, quantizedOutput};
    for (int i = 0; i < (quantizedOutput ? 6 : 5); ++i) {
        const Data *out = outputs[i];
        if (out->multiDeviceData || out->isFake) return false;
        for (const Data *in : inputs)
            if (out == in || (out->cudaData && out->cudaData == in->cudaData)) return false;
        for (int j = 0; j < i; ++j)
            if (out == outputs[j] || (out->cudaData && out->cudaData == outputs[j]->cudaData)) return false;
    }
    Data scratch;
    if (!V41PrepareOutput(hidden, DataType::BFLOAT16, residual.dims) ||
        !V41PrepareOutput(scratch, DataType::FLOAT32, {tokens, 8, 25}) ||
        !V41PrepareOutput(pre, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(post, DataType::FLOAT32, {b, n, 4}) ||
        !V41PrepareOutput(comb, DataType::FLOAT32, {b, n, 4, 4}) ||
        !V41PrepareOutput(output, DataType::BFLOAT16, {b, n, d})) return false;
    if (quantizedOutput && !V41PrepareOutput(*quantizedOutput, DataType::BFLOAT16, {b, n, d})) return false;
#define V41_POST_DOTS_ONE(D, O) if (!V41LaunchDependent<V41HcPostWarpDots<D, O>>(dim3((25 + O - 1) / O, tokens, 8), 32, 0, \
        (const __nv_bfloat16*)input.cudaData, (const __nv_bfloat16*)residual.cudaData, \
        (const float*)previousPost.cudaData, (const float*)previousComb.cudaData, \
        (const float*)fn.cudaData, (__nv_bfloat16*)hidden.cudaData, (float*)scratch.cudaData)) return false
#define V41_POST_DOTS(D) case D: \
        if (tokens <= 6) { V41_POST_DOTS_ONE(D, 2); } \
        else { V41_POST_DOTS_ONE(D, 4); } break
    switch (d) {
        V41_POST_DOTS(4096); V41_POST_DOTS(5120); V41_POST_DOTS(6144);
        V41_POST_DOTS(7168); V41_POST_DOTS(8192);
    }
#undef V41_POST_DOTS
#undef V41_POST_DOTS_ONE
#define V41_POST_FINISH(I, Q) if (!V41LaunchDependent<V41HcPreNormKernel<1024, I, true, Q, 8>>(tokens * 2, 1024, 0, \
        (const __nv_bfloat16*)hidden.cudaData, (const float*)previousPre.cudaData, \
        (const float*)normWeight.cudaData, (__nv_bfloat16*)output.cudaData, 4, d, normEps, \
        (const float*)scratch.cudaData, (const float*)scale.cudaData, (const float*)base.cudaData, \
        sinkhornIters, mixEps, (float*)post.cudaData, (float*)comb.cudaData, (float*)pre.cudaData, Q ? (__nv_bfloat16*)quantizedOutput->cudaData : nullptr)) return false
    if (d <= 4096) {
        if (quantizedOutput) { V41_POST_FINISH(2, true); }
        else { V41_POST_FINISH(2, false); }
    } else if (d <= 6144) {
        if (quantizedOutput) { V41_POST_FINISH(3, true); }
        else { V41_POST_FINISH(3, false); }
    } else {
        if (quantizedOutput) { V41_POST_FINISH(4, true); }
        else { V41_POST_FINISH(4, false); }
    }
#undef V41_POST_FINISH
    return V41CheckLaunch("HcPostMixPreNorm");
}

extern "C" bool FastllmCudaDeepSeekV41HcPostMixPreNorm(const Data &input, const Data &residual,
        const Data &previousPre, const Data &previousPost, const Data &previousComb,
        const Data &fn, const Data &scale, const Data &base, const Data &normWeight,
        int sinkhornIters, float mixEps, float normEps, Data &hidden,
        Data &pre, Data &post, Data &comb, Data &output) {
    return V41HcPostMixPreNormImpl(input, residual, previousPre, previousPost, previousComb,
        fn, scale, base, normWeight, sinkhornIters, mixEps, normEps,
        hidden, pre, post, comb, output, nullptr);
}

extern "C" bool FastllmCudaDeepSeekV41HcPostMixPreNormQuantized(const Data &input, const Data &residual,
        const Data &previousPre, const Data &previousPost, const Data &previousComb,
        const Data &fn, const Data &scale, const Data &base, const Data &normWeight,
        int sinkhornIters, float mixEps, float normEps, Data &hidden,
        Data &pre, Data &post, Data &comb, Data &output, Data &quantizedOutput) {
    return V41HcPostMixPreNormImpl(input, residual, previousPre, previousPost, previousComb,
        fn, scale, base, normWeight, sinkhornIters, mixEps, normEps,
        hidden, pre, post, comb, output, &quantizedOutput);
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

extern "C" bool FastllmCudaDsparkInverseRotaryBatch(Data &x, int ropeDim, float ropeBase, int originalSeqLen,
        float ropeFactor, int betaFast, int betaSlow, const int *positions) {
    if (!positions || !V41OnCuda(x) || x.dataType != DataType::BFLOAT16 || x.dims.size() != 4 ||
        x.dims[0] < 2 || x.dims[0] > 65535 || x.dims[1] <= 0 || x.dims[2] <= 0 || x.dims[3] <= 0 ||
        x.Count(1) / x.dims[3] > INT_MAX || ropeDim <= 0 || ropeDim > 128 || ropeDim > x.dims[3] ||
        ropeDim % 2 || V41LegacyRotary() || V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH")) return false;
    const uint64_t pairs = uint64_t(x.dims[1]) * x.dims[2] * (ropeDim / 2);
    const auto rope = V41BuildRope(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    V41RotaryOnlyKernel<__nv_bfloat16, true><<<dim3((pairs + 255) / 256, x.dims[0]), 256>>>(
        (__nv_bfloat16 *)x.cudaData, x.dims[2], x.dims[3], rope, ropeDim, 0, 1, 1, pairs, positions);
    return V41CheckLaunch("DsparkInverseRotaryBatch");
}

extern "C" bool FastllmCudaDeepSeekV41DraftRotaryTable(Data &q, Data &kv, const Data &window,
        int ropeDim, float ropeBase, int startPos, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, const int *devicePosition, Data &table) {
    return FastllmCudaDsparkDraftRotaryTable(q, kv, window, ropeDim, ropeBase, startPos, originalSeqLen,
        ropeFactor, betaFast, betaSlow, devicePosition, table, nullptr);
}

extern "C" bool FastllmCudaDsparkDraftRotaryTable(Data &q, Data &kv, const Data &window,
        int ropeDim, float ropeBase, int startPos, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, const int *devicePosition, Data &table, const int *changed) {
    if (V41LegacyRotary() || V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") ||
        V41EnvOn("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
        q.dims.size() != 4 || kv.dims.size() != 3 || window.dims.size() != 3 ||
        q.dims[0] < 1 || q.dims[1] < 1 || q.dims[2] < 1 || q.dims[2] > 128 ||
        q.dims[3] != 512 || kv.dims != std::vector<int>({q.dims[0], q.dims[1], 512}) ||
        window.dims[0] != q.dims[0] || window.dims[1] < 1 || window.dims[1] > 1024 || window.dims[2] != 512 ||
        ropeDim < 2 || ropeDim > 128 || ropeDim % 2 || table.multiDeviceData || table.isFake ||
        (uintptr_t)window.cudaData % 16 || (uintptr_t)table.cudaData % 16) return false;
    const std::vector<int> device{FastllmCudaGetDevice()};
    const Data *inputs[] = {&q, &kv, &window};
    const uint64_t qTasks = uint64_t(q.dims[1]) * q.dims[2] * (ropeDim / 2);
    const uint64_t qBlocks64 = (qTasks + 511) / 512;
    const uint64_t blocks = uint64_t(q.dims[0]) * (qBlocks64 + q.dims[1] + (window.dims[1] + 7) / 8);
    if (qTasks > INT_MAX || blocks > INT_MAX || q.dims[1] > INT_MAX - window.dims[1]) return false;
    const uint64_t tableBytes = (uint64_t)q.dims[0] * (window.dims[1] + q.dims[1]) * 512 * 2;
    auto overlaps = [](const Data &a, const Data &b, uint64_t bBytes = 0) {
        const uintptr_t x = (uintptr_t)a.cudaData, y = (uintptr_t)b.cudaData;
        return x && y && x < y + (bBytes ? bBytes : b.GetBytes()) && y < x + a.GetBytes();
    };
    for (int i = 0; i < 3; ++i) {
        const Data &x = *inputs[i];
        if (!V41OnCuda(x) || x.multiDeviceData || x.dataType != DataType::BFLOAT16 || ((i != 2 || !changed) && x.dataDeviceIds != device) ||
            &x == &table || overlaps(x, table, tableBytes)) return false;
        uint64_t stride = 1;
        if (x.strides.size() != x.dims.size()) return false;
        for (int j = (int)x.dims.size() - 1; j >= 0; --j) {
            if (x.strides[j] != stride) return false;
            stride *= x.dims[j];
        }
        for (int j = 0; j < i; ++j) if (overlaps(x, *inputs[j])) return false;
    }
    if (!V41PrepareOutput(table, DataType::BFLOAT16, {q.dims[0], window.dims[1] + q.dims[1], 512})) return false;
    const auto rope = V41BuildRope(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    const int qBlocks = int(qBlocks64);
    V41DraftRotaryTableKernel<<<unsigned(blocks), 512>>>(
        (__nv_bfloat16*)q.cudaData, (__nv_bfloat16*)kv.cudaData, (const __nv_bfloat16*)window.cudaData,
        (__nv_bfloat16*)table.cudaData, rope, q.dims[2], q.dims[1], window.dims[1], ropeDim,
        startPos, devicePosition, qBlocks, changed);
    return V41CheckLaunch("DraftRotaryTable");
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
        if (seqlen >= 2 && seqlen <= 8 && heads > 0 && bsz <= 65535 / seqlen &&
            V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<false, false, true>, 256>()) {
            V41IndexerScoreDecodeMmaKernel<false, false, true><<<dim3((m + 63) / 64, bsz * seqlen), 256>>>(
                (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData,
                kQuant ? nullptr : (const __nv_bfloat16*)k.cudaData, heads, m, ratio, startPos,
                (float*)output.cudaData, k8, kMode, kRowBytes, V41Fp4VecUnpack() ? 1 : 0,
                nullptr, seqlen);
            return V41CheckLaunch("IndexerScoreSmallChunkMma");
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

extern "C" bool FastllmCudaDeepSeekV41IndexerScoreFp4Query(const fastllm::Data &q,
        const fastllm::Data &weights, const fastllm::Data &k, int ratio, int startPos,
        fastllm::Data &output) {
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_SM120
    const int arch = FastllmCudaRuntimeArch();
    if ((arch == 120 || arch == 121) && !V41EnvOn("FASTLLM_DSV41_LEGACY_INDEXER") &&
        !FastllmCudaGraphIsCapturingFast() && V41OnCuda(q) && V41OnCuda(weights) && V41OnCuda(k) &&
        q.dataType == DataType::BFLOAT16 && q.dims.size() == 4 &&
        q.dims[0] > 0 && q.dims[0] <= 65535 && q.dims[1] >= 9 && q.dims[1] <= 65535 * 4 &&
        q.dims[2] == 32 && q.dims[3] == 128 &&
        weights.dataType == DataType::FLOAT32 && weights.dims == std::vector<int>({q.dims[0], q.dims[1], 32}) &&
        k.dataType == DataType::INT8 && k.dims.size() == 3 && k.dims[0] == q.dims[0] &&
        k.dims[1] > 0 && k.dims[2] == 68 && ratio > 0 && startPos >= 0 &&
        int64_t(startPos) + q.dims[1] < INT_MAX && int64_t(q.dims[0]) * q.dims[1] <= INT_MAX / 128) {
        // Prepare the throwing allocation before acquiring temporary storage.
        if (!V41PrepareOutput(output, DataType::FLOAT32, {q.dims[0], q.dims[1], k.dims[1]})) return false;
        void *scratch = nullptr;
        const size_t bytes = size_t(q.dims[0]) * q.dims[1] * 32 * 68;
        const auto allocation = FastllmCudaTryMalloc(&scratch, bytes);
        if (allocation == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
        if (allocation == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            const bool ok = fastllm_v41_indexer::Fp4Prefill(q.cudaData,
                static_cast<const float *>(weights.cudaData), static_cast<const uint8_t *>(k.cudaData),
                static_cast<float *>(output.cudaData), scratch, q.dims[0], q.dims[1], k.dims[1], ratio, startPos);
            if (!FastllmCudaFreeAfterCurrentThreadStream(scratch)) {
                FastllmCudaSyncCurrentThreadStream();
                FastllmCudaFree(scratch);
            }
            return ok;
        }
    }
#endif
    return FastllmCudaDeepSeekV41IndexerScore(q, weights, k, ratio, startPos, output);
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
    if (tokens <= 8 && numBlocks >= 4096 &&
        V41MmaKernelReady<V41CandidateSelectKernel<false, 1024>, 1024>()) {
        V41CandidateSelectKernel<false, 1024><<<tokens, 1024>>>(blockScore, numBlocks, topkBlocks, (uint8_t*)output.cudaData);
    } else {
        V41CandidateSelectKernel<<<tokens, kSelThreads>>>(blockScore, numBlocks, topkBlocks, (uint8_t*)output.cudaData);
    }
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
    // Standalone captures keep the zero-workspace route; full decode plans
    // own persistent hierarchical workspaces. Eager scratch is stream ordered.
    if (m > kTopKParallelMinKeys && width <= 512 && bs <= kTopKShardKeys &&
        (int64_t)bsz * seqlen <= 128 && !FastllmCudaGraphIsCapturing()) {
        // Prefill rows also benefit from coarse global filtering. Original FP32
        // scores decide the final exact selection; ties/overflow retain hierarchy.
        // Keep masked decode/verification unchanged. Wide masked prefill
        // batches also favor the shard path because of its lower traffic.
        const bool globalTopK = width == 512 && m >= 65536 &&
            (!cand || (bsz * seqlen > 8 && bsz * seqlen <= 64)) && m <= 1024 * kTopKShardKeys &&
            V41MmaKernelReady<V41TopKGlobalFinish<false>, 1024>();
        auto layout = V41TopKLayout(nullptr, bsz * seqlen, m, width, bs, globalTopK);
        void *memory = nullptr;
        const auto status = FastllmCudaTryMalloc(&memory, layout.Bytes(bsz * seqlen));
        if (status == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
        if (status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            const auto w = V41TopKLayout(memory, bsz * seqlen, m, width, bs, globalTopK);
            if (globalTopK) {
                V41TopKGlobalArgs a{(const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift,
                    width, ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr};
                V41LaunchGlobalTopK<false>(a, w, bsz * seqlen);
            } else {
                V41TopKKernel<false, 1><<<dim3(w.parts, bsz * seqlen), kSelThreads,
                    (size_t)listCap * sizeof(int)>>>(
                    (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
                    ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr, w);
                if (bsz * seqlen <= 8 && V41MmaKernelReady<V41TopKKernel<false, 2, 1024>, 1024>()) {
                    V41TopKKernel<false, 2, 1024><<<bsz * seqlen, 1024>>>(
                        (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
                        ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr, w);
                } else {
                    V41TopKKernel<false, 2><<<bsz * seqlen, kSelThreads>>>(
                        (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
                        ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr, w);
                }
            }
            const bool ok = V41CheckLaunch("IndexerTopKHierarchical");
            if (!FastllmCudaFreeAfterCurrentThreadStream(memory)) {
                FastllmCudaSyncCurrentThreadStream();
                FastllmCudaFree(memory);
            }
            return ok;
        }
    }
    if (seqlen > 1 || m <= width) {
        V41TopKKernel<false, 0, 256><<<bsz * seqlen, kSelThreads, (size_t)listCap * sizeof(int)>>>(
            (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
            ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr, {});
    } else {
        V41TopKKernel<false, 0, 1024><<<bsz * seqlen, 1024, (size_t)listCap * sizeof(int)>>>(
            (const float*)score.cudaData, cand, seqlen, m, numBlocks, bs, blockShift, width,
            ratio, startPos, listCap, (int32_t*)output.cudaData, nullptr, {});
    }
    return V41CheckLaunch("IndexerTopK");
}

// Hybrid workspaces require dense tensors on the selected CUDA device.
static bool V41DenseCuda(const fastllm::Data &x) {
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
    if (!V41DenseCuda(query) || !V41DenseCuda(keys) || !V41DenseCuda(weights) ||
        &output == &query || &output == &keys || &output == &weights ||
        query.dataDeviceIds != keys.dataDeviceIds || query.dataDeviceIds != weights.dataDeviceIds ||
        query.dataType != DataType::BFLOAT16 || keys.dataType != DataType::BFLOAT16 ||
        weights.dataType != DataType::FLOAT32 || query.dims.size() != 4 ||
        query.dims[0] != 1 || query.dims[1] < 1 || query.dims[2] != 32 || query.dims[3] != 128 ||
        keys.dims.size() != 3 || keys.dims[0] != 1 || keys.dims[1] < 1 || keys.dims[2] != 128 ||
        weights.Count(0) != size_t(query.dims[1]) * 32 || !V41IsSm75()) return false;
    if (position && (query.dims[1] != 1 || !V41DenseCuda(*position) ||
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
    if (!V41DenseCuda(query) || !V41DenseCuda(latent) || !V41DenseCuda(indices) ||
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

template<int splits, bool MATERIALIZED = true, bool PIPELINED = false>
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
    using SerialShared = typename std::conditional<MATERIALIZED,
        V41MmaPvShared<splits>, V41MmaShared>::type;
    using Shared = typename std::conditional<PIPELINED,
        V41MmaPvPipelineShared<splits>, SerialShared>::type;
    if constexpr (MATERIALIZED) {
        if (!V41MmaKernelReady<V41SparseAttentionScoresKernel<true>, kMmaThreads, sizeof(V41MmaShared)>()) return false;
    }
    if (!V41MmaKernelReady<V41SparseAttentionMmaKernel<splits, MATERIALIZED, true, PIPELINED>,
                           kMmaThreads, sizeof(Shared)>()) return false;
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
    if (!V41LaunchDependent<V41SparseAttentionScoresKernel<true>>(dim3(requests.size(), headBlocks, maxTiles), kMmaThreads, sizeof(V41MmaShared),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, heads, windowSize,
        0, 0, 0, nullptr, 0, 0, 0, 0, 0, deviceDescriptors)) return false;
    if (!V41CheckLaunch("BatchedSparseAttentionScores")) return false;
    }
    if (!V41LaunchDependent<V41SparseAttentionMmaKernel<splits, MATERIALIZED, true, PIPELINED>>(dim3(requests.size(), headBlocks, splits), kMmaThreads, sizeof(Shared),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (const float*)attnSink.cudaData, 1, heads, windowSize, 0, 0, 0, softmaxScale,
        nullptr, 0, 0, 0, 0, 0, nullptr, deviceDescriptors)) return false;
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
    // Two-stage PV overlaps the next KV tile for the small verification
    // batches. Larger batches already have enough CTAs and use the serial path.
    // Async copies consume BF16 rows directly; packed KV needs dequantization.
    if (batch >= 2 && batch <= 6 && V41IsSm<9, 0>() &&
        std::all_of(requests.begin(), requests.end(), [](const FastllmCudaV41AttentionRequest &r) {
            return (!r.ringKV || r.ringKV->dataType == DataType::BFLOAT16) &&
                   (!r.compressedKV || r.compressedKV->dataType == DataType::BFLOAT16);
        }) && V41MmaKernelReady<
            V41SparseAttentionMmaKernel<8, true, true, true>, kMmaThreads,
            sizeof(V41MmaPvPipelineShared<8>)>())
        return V41SparseAttentionBatchImpl<8, true, true>(requests, attnSink, windowSize, softmaxScale);
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

bool FastllmCudaDsparkDraftSparseAttention(const fastllm::Data &q, const fastllm::Data &kv,
        const fastllm::Data &table, const fastllm::Data &indices, fastllm::Data &sink,
        float scale, fastllm::Data &scratch, fastllm::Data &output) {
    const int batch = q.dims.size() == 4 ? q.dims[0] : 0;
    const int heads = batch > 0 ? q.dims[2] : 0;
    if (batch > 0 && (heads == 8 || heads == 64) && q.dims == std::vector<int>({batch, 5, heads, 512}) &&
        kv.dims == std::vector<int>({batch, 5, 512}) && table.dims == std::vector<int>({batch, 133, 512}) &&
        indices.dims == std::vector<int>({batch, 5, 132}) && q.dataType == DataType::BFLOAT16 &&
        kv.dataType == q.dataType && table.dataType == q.dataType && indices.dataType == DataType::INT32 &&
        sink.dataType == DataType::FLOAT32 && sink.Count(0) >= heads &&
        (scratch.dims.empty() || (scratch.dataType == DataType::FLOAT32 &&
         scratch.Count(0) >= size_t(batch) * 5 * heads * 192 && V41OnCuda(scratch))) && V41OnCuda(q) && V41OnCuda(kv) &&
        V41OnCuda(table) && V41OnCuda(indices) && V41OnCuda(sink) &&
        !V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && !V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN") &&
        V41IsSm<10, 3>() && V41MmaKernelReady<V41SparseAttentionMmaKernel<8, true, false, true>,
            kMmaThreads, sizeof(V41MmaPvPipelineShared<8>)>()) {
        if (scratch.dims.empty() && !V41PrepareOutput(scratch, DataType::FLOAT32, {batch, 5, heads, 192})) return false;
        if (!V41PrepareOutput(output, DataType::BFLOAT16, q.dims)) return false;
        const auto *query = static_cast<const __nv_bfloat16 *>(q.cudaData);
        const auto *current = static_cast<const __nv_bfloat16 *>(kv.cudaData);
        const auto *history = static_cast<const __nv_bfloat16 *>(table.cudaData);
        const auto *idx = static_cast<const int32_t *>(indices.cudaData);
        auto *scores = static_cast<float *>(scratch.cudaData);
        if (heads == 64) V41DraftScores<64><<<dim3(batch * 5, 24, 4), 32>>>(query, current, history, idx, scores);
        else V41DraftScores<8><<<dim3(batch * 5, 24), 32>>>(query, current, history, idx, scores);
        if (!V41CheckLaunch("DraftScores")) return false;
        V41SparseAttentionMmaKernel<8, true, false, true>
            <<<dim3(batch * 5, (heads + 15) / 16, 8), kMmaThreads, sizeof(V41MmaPvPipelineShared<8>)>>>(
                query, current, nullptr, history, nullptr, nullptr, idx, static_cast<const float *>(sink.cudaData),
                5, heads, 1, 133, 132, 0, scale, static_cast<__nv_bfloat16 *>(output.cudaData), 1, 0, 1, 0, 0, scores);
        return V41CheckLaunch("DraftValues");
    }
    if (FastllmCudaGetThreadError()) return false;
    return FastllmCudaDeepSeekV41SparseAttention(q, kv, nullptr, &table, &indices, sink, 1, 0, scale, output);
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
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_SM120
    // Shared FlashInfer backend for ordinary small causal chunks, including
    // per-layer DSpark verification. Preserve native index-list semantics;
    // the model's indexer already masks future compressed rows.
    if (!referenceMath && !legacyAttention && FastllmV41FlashInferEnabled() &&
        q.dataType == DataType::BFLOAT16 && !q.multiDeviceData && !chunkKV.multiDeviceData &&
        chunkKV.dims == std::vector<int>({bsz, seqlen, dim}) && windowSize == 128 &&
        (!hasRing || (ringFp8 && ringMode == 1 && ringRowBytes == 528 &&
            ringKV->dims[0] == bsz && !ringKV->multiDeviceData)) &&
        (!hasCmp || (cmpFp8 && cmpMode == 3 && cmpRowBytes == 288 &&
            compressedKV->dims[0] == bsz && cmpIdx->dims[0] == bsz && cmpIdx->dims[1] == seqlen &&
            !compressedKV->multiDeviceData && !cmpIdx->multiDeviceData)) &&
        !attnSink.multiDeviceData && attnSink.Count(0) >= heads) {
        const FastllmV41FlashInferChunkConfig config{bsz, seqlen, heads, topWidth, cap, startPos, softmaxScale};
        const size_t bytes = FastllmV41FlashInferChunkWorkspace(config);
        void *allocation = nullptr;
        if (bytes) {
            const auto status = FastllmCudaTryMalloc(&allocation, bytes);
            if (status == FASTLLM_CUDA_TRY_MALLOC_ERROR) return false;
            if (status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
                struct Scratch {
                    void *ptr;
                    ~Scratch() {
                        if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr);
                    }
                } scratch{allocation};
                FastllmV41FullDecodeBuffers b;
                b.q = q.cudaData; b.kv = chunkKV.cudaData; b.output = output.cudaData;
                b.sink = attnSink.cudaData; b.topk = hasCmp ? cmpIdx->cudaData : nullptr;
                const auto launched = FastllmV41FlashInferChunkRun(config, b,
                    hasRing ? ringKV->cudaData : nullptr, hasCmp ? compressedKV->cudaData : nullptr,
                    allocation, cudaStreamPerThread);
                return launched == cudaSuccess && V41CheckLaunch("SparseAttentionFlashInfer");
            }
            // Optional workspace did not fit. Native attention remains valid.
        }
    }
#endif

#ifdef FASTLLM_V41_FP16_ATTENTION
    // FP16 Tensor Cores are available on SM75 even though native BF16 MMA is
    // not. Use the same gathered-KV backend for eager chunks and full graphs.
    if (!legacyAttention && !referenceMath && FastllmV41FlashInferEnabled() && V41IsSm75() &&
        q.dataType == DataType::BFLOAT16 && bsz > 0 && seqlen > 0 && seqlen <= 32 &&
        windowSize > 0 && startPos >= 0 && int64_t(startPos) + seqlen <= INT_MAX &&
        chunkKV.dims == std::vector<int>({bsz, seqlen, dim}) &&
        (!hasRing || ringKV->dims[0] == bsz) &&
        (!hasCmp || (compressedKV->dims[0] == bsz && cmpIdx->dims[0] == bsz && cmpIdx->dims[1] == seqlen)) &&
        attnSink.Count(0) >= heads && v41_fp16_attention::Ready()) {
        V41AttentionDescriptor d{static_cast<const __nv_bfloat16 *>(q.cudaData),
            static_cast<const __nv_bfloat16 *>(chunkKV.cudaData),
            hasRing && !ringFp8 ? static_cast<const __nv_bfloat16 *>(ringKV->cudaData) : nullptr,
            hasCmp && !cmpFp8 ? static_cast<const __nv_bfloat16 *>(compressedKV->cudaData) : nullptr,
            ringFp8 ? static_cast<const uint8_t *>(ringKV->cudaData) : nullptr,
            cmpFp8 ? static_cast<const uint8_t *>(compressedKV->cudaData) : nullptr,
            hasCmp ? static_cast<const int32_t *>(cmpIdx->cudaData) : nullptr,
            static_cast<__nv_bfloat16 *>(output.cudaData), nullptr, cap, topWidth, startPos,
            ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack};
        if (v41_fp16_attention::Chunk(d, bsz, seqlen, heads, windowSize, softmaxScale,
                                    static_cast<const float *>(attnSink.cudaData))) return true;
        if (FastllmCudaGetThreadError()) return false;
    }
#endif

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
                        V41Sm75QuantizedAttentionScores<multi, 4, 16><<<dim3((count + 15) / 16, (heads + 3) / 4, seqlen), 128, 0, cudaStreamPerThread>>>(
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
    // builds and shapes retain the generic MMA path below. The kernel only needs
    // SM80 BF16 MMA; V41MmaKernelReady also checks the compiled image, thread
    // limit and opt-in shared memory capacity before enabling it on SM80+.
    if (!legacyAttention && !referenceMath && q.dataType == DataType::BFLOAT16 &&
        seqlen >= 128 && heads >= 32 &&
        V41MmaKernelReady<V41SparseAttentionPrefillSharedKVKernel, 512, sizeof(V41PrefillSharedKV)>()) {
        V41SparseAttentionPrefillSharedKVKernel<<<dim3(bsz * seqlen, (heads + 31) / 32), 512, sizeof(V41PrefillSharedKV)>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData,
            ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
        return V41CheckLaunch("SparseAttentionPrefillSharedKV");
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
        const bool smallVerify = tokens >= 2 && tokens <= 8 && startPos > 0 && !ringFp8 && !cmpFp8;
        const bool draftValues = seqlen == 5 && heads == 8 && startPos == 0 && windowSize == 1 &&
            !ringKV && hasCmp && !cmpFp8 && cap == 133 && topWidth == 132 && V41IsSm<10, 3>();
        const bool splitValues = tokens == 1 || smallVerify || draftValues;
        bool launched = false;
        auto launch = [&](auto splitTag, auto pipelineTag) {
            constexpr int splits = decltype(splitTag)::value;
            constexpr bool pipeline = decltype(pipelineTag)::value;
            using PvShared = typename std::conditional<pipeline,
                V41MmaPvPipelineShared<splits>, V41MmaPvShared<splits>>::type;
            using QkShared = typename std::conditional<pipeline,
                V41MmaScoresPrefetchShared, V41MmaShared>::type;
            constexpr int qkThreads = pipeline ? kQkPrefetchCandidates * 4 : kMmaThreads;
            const int candidateTiles = ((startPos == 0 ? std::min(seqlen, windowSize) : windowSize) + topWidth + 63) / 64;
            const uint64_t scratchBytes = (uint64_t)tokens * heads * candidateTiles * 64 * sizeof(float);
            if (splitValues && (!smallVerify || !FastllmCudaGraphIsCapturing()) &&
                candidateTiles >= 4 && scratchBytes <= 4 * 1024 * 1024 &&
                V41MmaKernelReady<V41SparseAttentionScoresKernel<false, pipeline>, qkThreads, sizeof(QkShared)>() &&
                V41MmaKernelReady<V41SparseAttentionMmaKernel<splits, true, false, pipeline>, kMmaThreads, sizeof(PvShared)>()) {
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
                    V41SparseAttentionScoresKernel<false, pipeline><<<dim3(tokens, headBlocks, candidateTiles * (pipeline ? 2 : 1)), qkThreads, sizeof(QkShared)>>>(
                        (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
                        hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
                        hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
                        ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
                        cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
                        hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
                        seqlen, heads, windowSize, cap, topWidth, startPos, scores,
                        ringMode, ringRowBytes, cmpMode, cmpRowBytes, vecUnpack ? 1 : 0);
                    if (!V41CheckLaunch("SparseAttentionScores")) return false;
                    V41SparseAttentionMmaKernel<splits, true, false, pipeline><<<dim3(tokens, headBlocks, splits), kMmaThreads, sizeof(PvShared)>>>(
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
        const bool pipeline = smallVerify && heads == 8 && V41IsSm<9, 0>() &&
            V41MmaKernelReady<V41SparseAttentionScoresKernel<false, true>, kQkPrefetchCandidates * 4,
                sizeof(V41MmaScoresPrefetchShared)>() &&
            V41MmaKernelReady<V41SparseAttentionMmaKernel<8, true, false, true>, kMmaThreads,
                sizeof(V41MmaPvPipelineShared<8>)>();
        const bool ok = pipeline ? launch(std::integral_constant<int, 8>{}, std::true_type{}) :
            splitValues ? launch(std::integral_constant<int, 8>{}, std::false_type{}) :
                          launch(std::integral_constant<int, 1>{}, std::false_type{});
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
                           int windowSize, const int *devicePosition, int rowCount,
                           const int *deviceCounts = nullptr, bool batchMetadata = false) {
    if (!V41OnCuda(chunk) || chunk.dims.size() != 3 || windowSize <= 0) {
        return false;
    }
    int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
    if (bsz <= 0 || seqlen <= 0 || (batchMetadata &&
        (!devicePosition || !deviceCounts || uint64_t(bsz) * seqlen > INT32_MAX))) return false;
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
    int firstRow = batchMetadata ? 0 : std::max(0, rowCount - windowSize);
    dim3 grid(batchMetadata ? bsz * seqlen : rowCount - firstRow, batchMetadata ? 1 : bsz);
    V41WindowStoreKernel<<<grid, 256>>>((const uint8_t*)chunk.cudaData, (uint8_t*)ring.cudaData, seqlen, rowBytes,
                                        startPos, windowSize, firstRow, devicePosition, deviceCounts, batchMetadata);
    return V41CheckLaunch("WindowStore");
}

extern "C" bool FastllmCudaDeepSeekV41WindowStorePosition(const fastllm::Data &chunk, fastllm::Data &ring,
                                                        int startPos, int windowSize, const int *devicePosition) {
    return V41WindowStore(chunk, ring, startPos, windowSize, devicePosition, -1);
}

extern "C" void FastllmCudaDsparkPackAttention(const Data &query, const Data &kv, const Data &window,
        const Data &indices, const Data &position, Data &shadow, Data &valid, const size_t *offsets,
        Data &packet) {
    DsparkAttentionPacket fields{};
    const Data *src[] = {&query, &kv, &indices, &position};
    char *dst = static_cast<char *>(packet.cudaData);
    for (int i = 0; i < 4; ++i) {
        fields.src[i] = static_cast<const int *>(src[i]->cudaData);
        fields.dest[i] = reinterpret_cast<int *>(dst + offsets[i < 2 ? i : i + 1]);
        fields.words[i] = i == 3 ? 1 : int(src[i]->GetBytes() / sizeof(int));
    }
    DsparkPackAttention<<<window.dims[1] + 32, 128>>>(fields, static_cast<const uint4 *>(window.cudaData),
        static_cast<uint4 *>(shadow.cudaData), static_cast<int *>(valid.cudaData),
        reinterpret_cast<int *>(dst + offsets[2]), window.dims[1]);
    V41CheckLaunch("DsparkPackAttention");
}

extern "C" bool FastllmCudaDsparkContextStore(const Data *const projected[3], const Data *const norms[3],
        Data *const rings[3], bool stacked, int batch, int rows, const int *positions, float eps,
        int ropeDim, float ropeBase, float ropeFactor, int betaFast, int betaSlow) {
    if (!positions || batch <= 0 || rows <= 0 || int64_t(batch) * rows > INT_MAX ||
        ropeDim < 2 || ropeDim > 128 || ropeDim % 2 || V41LegacyRotary() ||
        V41EnvOn("FASTLLM_DSV41_DISABLE_FAKE_QUANT") || V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH")) return false;
    const std::vector<int> device{FastllmCudaGetDevice()};
    DsparkContextStoreArgs args{};
    int window = 0;
    const int stride = stacked ? 1536 : 512;
    for (int l = 0; l < 3; ++l) {
        const auto &kv = *projected[stacked ? 0 : l], &norm = *norms[l], &ring = *rings[l];
        if (!V41DenseCuda(kv) || !V41DenseCuda(norm) || !V41DenseCuda(ring) || kv.dataDeviceIds != device ||
            norm.dataDeviceIds != device || ring.dataDeviceIds != device ||
            (uintptr_t(kv.cudaData) % 4) || (uintptr_t(ring.cudaData) % 4) || (uintptr_t(norm.cudaData) % 8) || kv.dataType != DataType::BFLOAT16 || norm.dataType != DataType::FLOAT32 ||
            ring.dataType != DataType::BFLOAT16 || kv.dims.empty() || kv.dims.back() != stride ||
            kv.Count(0) != uint64_t(batch) * rows * stride || norm.Count(0) != 512 ||
            ring.dims.size() != 3 || ring.dims[0] != batch || ring.dims[1] <= 0 ||
            ring.dims[2] != 512 || (l && ring.dims[1] != window)) return false;
        window = ring.dims[1];
        args.kv[l] = (const __nv_bfloat162 *)kv.cudaData + (stacked ? 256 * l : 0);
        args.norm[l] = (const float2 *)norm.cudaData;
        args.ring[l] = (__nv_bfloat162 *)ring.cudaData;
    }
    auto rope = V41BuildRope(ropeDim, ropeBase, 0, ropeFactor, betaFast, betaSlow);
    DsparkContextStore<<<dim3(batch * rows, 3), 256>>>(args, positions, batch, rows, window,
        stride / 2, eps, rope, ropeDim);
    return V41CheckLaunch("DsparkContextStore");
}

extern "C" bool FastllmCudaDeepSeekV41WindowStoreBatchPrefix(const fastllm::Data &chunk, fastllm::Data &ring,
        int windowSize, const int *devicePositions, const int *deviceCounts) {
    return V41WindowStore(chunk, ring, 0, windowSize, devicePositions, -1, deviceCounts, true);
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

__global__ void V41JoinTokenRows(const float *routed, const __nv_bfloat16 *shared,
                                 __nv_bfloat16 *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(routed[i] + __bfloat162float(shared[i]));
}
extern "C" bool FastllmCudaDeepSeekV41JoinTokenRows(const float *routed, const void *shared,
                                                   void *output, int count) {
    if (!routed || !shared || !output || count < 1) return false;
    V41JoinTokenRows<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        routed, static_cast<const __nv_bfloat16 *>(shared), static_cast<__nv_bfloat16 *>(output), count);
    return V41CheckLaunch("JoinTokenRows");
}

__global__ void V41MoeJoinBFloat16Kernel(const float *routed, const float *shared,
                                        __nv_bfloat16 *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        // Match BF16 branch outputs and the subsequent BF16 tensor addition.
        // Rounding only the FP32 combined sum would have different semantics.
        const float a = __bfloat162float(__float2bfloat16_rn(routed[i]));
        const float b = __bfloat162float(__float2bfloat16_rn(shared[i]));
        output[i] = __float2bfloat16_rn(a + b);
    }
}

extern "C" bool FastllmCudaDeepSeekV41MoeJoinBFloat16(
        const float *routed, const float *shared, void *output, int count) {
    if (!routed || !shared || !output || count <= 0 ||
        output == routed || output == shared) return false;
    V41MoeJoinBFloat16Kernel<<<(count + 255) / 256, 256>>>(
        routed, shared, static_cast<__nv_bfloat16 *>(output), count);
    return V41CheckLaunch("MoeJoinBFloat16");
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
        const bool packedWindow = item.window && item.window->dataType == DataType::INT8;
        if (item.window && (dim != 512 || rows != 1 || !V41OnCuda(*item.window) ||
            (!packedWindow && item.window->dataType != DataType::BFLOAT16) ||
            item.window->dims != std::vector<int>({1, windowSize, packedWindow ? 528 : 512}))) return false;
        descriptors.push_back({(__nv_bfloat16 *)item.input->cudaData,
            item.window && !packedWindow ? (__nv_bfloat16 *)item.window->cudaData : nullptr,
            rows, dim, item.startPos, item.quantMode, item.inverse ? 1 : 0, item.quantBlock});
        if (packedWindow) descriptors.back().packedWindow = (uint8_t *)item.window->cudaData;
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
            item.query->dataType != DataType::BFLOAT16 ||
            (item.keys->dataType != DataType::BFLOAT16 && item.keys->dataType != DataType::INT8) ||
            item.weights->dataType != DataType::FLOAT32 || item.query->dims.size() != 4 ||
            item.query->dims[0] != 1 || item.query->dims[1] != 1 || item.query->dims[3] != 128 ||
            item.keys->dims.size() != 3 || item.keys->dims[0] != 1 ||
            item.keys->dims[1] <= 0 || item.startPos < 0 || ropeDim <= 0 || ropeDim > 128 || (ropeDim & 1)) return false;
        int keyMode = 0, ignoredBlockSize = 0;
        const bool packedKeys = item.keys->dataType == DataType::INT8;
        if (packedKeys ? !V41ParseKvRowHost(128, item.keys->dims[2], &keyMode, &ignoredBlockSize) :
                         item.keys->dims[2] != 128) return false;
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
        if (packedKeys) {
            auto &d = descriptors.back();
            d.keys = nullptr;
            d.packedKeys = (const uint8_t *)item.keys->cudaData;
            d.keyMode = keyMode;
            d.keyRowBytes = item.keys->dims[2];
            d.keyVectorUnpack = V41Fp4VecUnpack();
        }
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
    V41TopKKernel<true, 0, 1024><<<items.size(), 1024, (size_t)listCap * sizeof(int), cudaStreamPerThread>>>(
        nullptr, nullptr, 1, 0, 0, blockSize, shift, 0, ratio, 0, listCap, nullptr, device, {});
    const bool ok = V41CheckLaunch("DecodeIndexerBatch");
    if (!borrowed && !FastllmCudaFreeAfterCurrentThreadStream(storage)) {
        cudaStreamSynchronize(cudaStreamPerThread);
        FastllmCudaFree(storage);
    }
    if (!ok) throw std::runtime_error("DeepSeekV41 decode batch kernel launch failed");
    return true;
}

namespace {
#if defined(FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE) && defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103)
// One CTA merges a (request row, attention head) and emits O-A's rounded
// block-scaled BF16 input. Chunk boundaries come from each request descriptor.
__global__ void V41VerifyMergeRotaryOutput(FastllmV41FullDecodeBuffers b, FastllmV41FlashInferPartials history,
        const FastllmV41FullDecodeRequest *requests, V41RopeTable rope, float scale,
        int heads, int rows, int begin) {
    const int row = begin + blockIdx.y, h = blockIdx.x, d = threadIdx.x * 4;
    uint2 packed = fastllm_v41_verify::MergeValues(static_cast<const __nv_bfloat16 *>(b.q),
        static_cast<const __nv_bfloat16 *>(b.kv), static_cast<const __nv_bfloat16 *>(history.values),
        history.lse, static_cast<const float *>(b.sink), scale, heads, history.heads,
        history.splits, rows, requests, begin);
    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&packed);
    float values[4];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        float2 v = __bfloat1622float2(pairs[pair]);
        if (d >= 448) {
            const float angle = float(requests[row].position) * rope.invFreq[(d - 448) / 2 + pair];
            const float c = cosf(angle), s = -sinf(angle);
            // Same contraction and BF16 rounding as V41DecodeTransformRow (mode 1).
            const float x = v.x * c - v.y * s, y = v.x * s + v.y * c;
            v.x = V41Bf16Round(x); v.y = V41Bf16Round(y);
        }
        values[2 * pair] = v.x; values[2 * pair + 1] = v.y;
    }
    float peak = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
    for (int stride = 4; stride; stride >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, stride));
    const float factor = V41QuantScaleDev(peak, 1);
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
        pairs[pair] = __floats2bfloat162_rn(V41QuantValueDev(values[2 * pair], factor, 1),
                                           V41QuantValueDev(values[2 * pair + 1], factor, 1));
    auto *out = static_cast<__nv_bfloat16 *>(b.output) + (size_t(row) * heads + h) * 512;
    reinterpret_cast<uint2 *>(out)[threadIdx.x] = packed;
}
#endif

constexpr int kFullDecodeTransforms = 5;

struct V41FullDecodePlan {
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE
    void *flashInfer = nullptr;
#endif
#ifdef FASTLLM_V41_FP16_ATTENTION
    std::shared_ptr<void> fp16Workspace;
    size_t fp16WorkspaceBytes = 0;
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    std::shared_ptr<FastllmV41MlaWorkspace> mla;
    bool directMlaOutput = false;
#endif
    bool packVerify = false;
    bool directRing = false, earlyHistory = false, shortPages = false;
    bool scalarAttention = false, scalarIndex = false;
    bool indexPairs = false, indexPairNarrow = false;
    bool qkPrefetch = false, wideTopKMerge = false, indexPrefetch = false;
    bool fuseIndexBlocks = false, wideCandidateSelect = false;
    FastllmV41FullDecodeConfig config;
    FastllmV41FullDecodeBuffers buffers;
    int device = 0, gridY = 0, tiles = 0, splits = 8, listCap = 0, blockShift = -1;
    bool fused = false, pending = false, pvPipeline = false;
    V41RopeTable rope;
    FastllmV41FullDecodeRequest *host = nullptr, *requests = nullptr;
    V41DecodeTransformDescriptor *transforms = nullptr;
    V41DecodeIndexDescriptor *index = nullptr;
    V41AttentionDescriptor *attention = nullptr;
    float *indexScores = nullptr, *blockScores = nullptr, *attentionScores = nullptr;
    uint8_t *topKMemory = nullptr;
    V41TopKWorkspace topKWorkspace;
    int *candidateBlocks = nullptr, *candidateCounts = nullptr;
    cudaEvent_t copied = nullptr;
    ~V41FullDecodePlan() {
        int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
        // Graph ownership must be released before its operand storage.
        cudaStreamSynchronize(cudaStreamPerThread);
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE
        FastllmV41FlashInferDestroy(flashInfer);
#endif
        if (pending) cudaEventSynchronize(copied);
        if (copied) cudaEventDestroy(copied);
        if (host) cudaFreeHost(host);
        if (requests && !buffers.requests) cudaFree(requests);
        for (void *p : {static_cast<void *>(transforms),
                       static_cast<void *>(index), static_cast<void *>(attention),
                       static_cast<void *>(indexScores), static_cast<void *>(blockScores),
                       static_cast<void *>(attentionScores), static_cast<void *>(topKMemory)}) if (p) cudaFree(p);
        if (candidateBlocks && !buffers.candidateBlocks) cudaFree(candidateBlocks);
        if (candidateCounts && !buffers.candidateCounts) cudaFree(candidateCounts);
        cudaSetDevice(old);
    }
};

// Descriptor arrays are initialized on the GPU. Request addresses and lengths
// may change on every replay without changing any graph node or launch shape.
__device__ __forceinline__ void V41FullDecodeSetupRow(
        int i, FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41DecodeTransformDescriptor *transforms,
        V41DecodeIndexDescriptor *index, V41AttentionDescriptor *attention,
        float *indexScores, float *blockScores, float *attentionScores, int tiles,
        int *candidateBlocks, int *candidateCounts) {
    const auto r = requests[i];
    const int begin = c.verifying ? r.chunkBegin : i;
    const int length = r.chunkLength > 0 ? r.chunkLength : c.batch;
    const int end = c.verifying ? requests[begin].position + length : r.position + 1;
    const int count = c.ratio ? end / c.ratio : 0;
    const int blocks = c.ratio ? (count + c.blockSize - 1) / c.blockSize : 0;
    const int maxBlocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
    auto *q = static_cast<__nv_bfloat16 *>(b.q) + (size_t)i * c.heads * 512;
    auto *kv = static_cast<__nv_bfloat16 *>(b.kv) + (size_t)i * 512;
    auto *out = static_cast<__nv_bfloat16 *>(b.output) + (size_t)i * c.heads * 512;
    auto *tk = c.ratio ? static_cast<int32_t *>(b.topk) + (size_t)i * c.topK : nullptr;
    auto *cand = c.ratio ? static_cast<uint8_t *>(b.candidates) + (size_t)i * maxBlocks : nullptr;
    auto *t = transforms + size_t(i) * kFullDecodeTransforms;
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
        if (c.indexMode) {
            index[i].keys = nullptr;
            index[i].packedKeys = static_cast<const uint8_t *>(r.indexK);
            index[i].keyMode = c.indexMode;
            index[i].keyRowBytes = c.indexRowBytes;
            index[i].keyVectorUnpack = c.vectorUnpack;
        }
        index[i].candidateBlockSize = c.blockSize;
        if (candidateBlocks) {
            index[i].candidateBlocks = candidateBlocks + (size_t)i * maxBlocks;
            index[i].candidateCount = candidateCounts + i;
        }
    }
    V41AttentionDescriptor a{};
    a.q = q;
    a.chunkKV = static_cast<const __nv_bfloat16 *>(b.kv) + (size_t)begin * 512;
    a.chunkOffset = i - begin;
    a.ringKV = static_cast<const __nv_bfloat16 *>(r.windowKV);
    a.compressedKV = static_cast<const __nv_bfloat16 *>(r.compressedKV);
    if (c.ringMode) {
        a.ringKV = nullptr; a.compressedKV = nullptr;
        a.ringKV8 = static_cast<const uint8_t *>(r.windowKV);
        a.compressedKV8 = static_cast<const uint8_t *>(r.compressedKV);
        a.ringMode = c.ringMode; a.ringRowBytes = c.ringRowBytes;
        a.cmpMode = c.cmpMode; a.cmpRowBytes = c.cmpRowBytes;
        a.vecUnpack = c.vectorUnpack && c.ringRowBytes % 16 == 0 && c.cmpRowBytes % 16 == 0;
    }
    a.indices = tk; a.output = out;
    a.scores = attentionScores + (size_t)i * c.heads * tiles * kMmaNC;
    a.cap = count; a.topWidth = min(count, c.topK); a.startPos = r.position;
    attention[i] = a;
}

// Q/KV transforms read request and buffer arguments directly. They do not
// depend on the descriptors initialized by the first CTA; later kernels see
// both descriptor writes and transformed rows through ordinary stream order.
struct V41DecodeKeyNorm {
    const __nv_bfloat16 *raw;
    const float *weight;
    int stride;
    float eps;
};

// One half warp owns a 32-channel E4M3 group. The verifier consumes the
// rounded BF16 row now and commits the independently encoded row on acceptance.
__device__ __forceinline__ void V41PrepareVerifyKey(__nv_bfloat16 *kv, uint8_t *saved,
        float *angles, const V41RopeTable &rope, int position) {
    const int pair = threadIdx.x;
    float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(kv)[pair]);
    if (pair >= 224) {
        const int k = pair - 224;
        const float angle = float(position) * rope.invFreq[k];
        const float c = cosf(angle), s = sinf(angle);
        angles[k] = c;
        angles[32 + k] = s;
        v = make_float2(V41Bf16Round(v.x * c - v.y * s), V41Bf16Round(v.x * s + v.y * c));
    }
    auto groupScale = [](float2 values) {
        float peak = fmaxf(fabsf(values.x), fabsf(values.y));
#pragma unroll
        for (int delta = 8; delta; delta >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, delta, 16));
        return V41QuantScaleDev(peak, 1);
    };
    const float scale = groupScale(v);
    const auto current = __floats2bfloat162_rn(V41QuantValueDev(v.x, scale, 1),
                                               V41QuantValueDev(v.y, scale, 1));
    reinterpret_cast<__nv_bfloat162 *>(kv)[pair] = current;
    // The group's peak can round down across a scale boundary. Match the
    // storage encoder's second scale selection using the actual BF16 values.
    v = __bfloat1622float2(current);
    const float savedScale = groupScale(v);
    const __nv_fp8_e4m3 lo(V41QuantCodeInput(v.x, savedScale, 1));
    const __nv_fp8_e4m3 hi(V41QuantCodeInput(v.y, savedScale, 1));
    reinterpret_cast<uchar2 *>(saved)[pair] = make_uchar2(lo.__x, hi.__x);
    if (!(pair & 15)) saved[512 + pair / 16] = __float_as_uint(savedScale) >> 23;
}

template<bool NativeQuery = false, bool NormalizeKey = false, bool PackVerify = false>
__global__ void V41FullDecodePrepareKernel(
        FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41DecodeTransformDescriptor *transforms,
        V41DecodeIndexDescriptor *index, V41AttentionDescriptor *attention,
        float *indexScores, float *blockScores, float *attentionScores, int tiles,
        V41RopeTable rope, int *candidateBlocks, int *candidateCounts, int rowOffset, bool directRing,
        __nv_bfloat16 *mlaQuery, float *mlaAngles, uint8_t *mlaRing, V41DecodeKeyNorm keyNorm) {
    const int token = rowOffset + blockIdx.y;
    if constexpr (NativeQuery && PackVerify && !NormalizeKey) {
        if (blockIdx.x == 0) {
            V41PrepareVerifyKey(static_cast<__nv_bfloat16 *>(b.kv) + size_t(token) * 512,
                static_cast<uint8_t *>(b.savedKV) + size_t(token) * c.ringRowBytes,
                mlaAngles + size_t(token) * 64, rope, requests[token].position);
        } else if (threadIdx.x == 0) {
            V41FullDecodeSetupRow(token, c, b, requests, transforms, index, attention,
                indexScores, blockScores, attentionScores, tiles, candidateBlocks, candidateCounts);
        }
        return;
    }
    const bool metadata = blockIdx.x == (NormalizeKey ? 1 : 0);
    if (metadata && threadIdx.x == 0)
        V41FullDecodeSetupRow(token, c, b, requests, transforms, index, attention,
                             indexScores, blockScores, attentionScores, tiles,
                             candidateBlocks, candidateCounts);
    if constexpr (NativeQuery) {
        if (metadata && threadIdx.x < 32) {
            const float angle = float(requests[token].position) * rope.invFreq[threadIdx.x];
            mlaAngles[token * 64 + threadIdx.x] = cosf(angle);
            mlaAngles[token * 64 + 32 + threadIdx.x] = sinf(angle);
        }
    }
    if constexpr (NormalizeKey) {
        if (metadata) return;
        V41QKVNormBody<1, false, false, 512>(nullptr, keyNorm.raw, nullptr, keyNorm.weight,
            nullptr, nullptr, static_cast<__nv_bfloat16 *>(b.kv), 0, 512, 0, keyNorm.stride,
            keyNorm.eps, nullptr, nullptr, 0, false, token);
        __syncthreads(); // Rotate the normalized BF16 pairs produced by this CTA.
    }
    const bool kv = NativeQuery || blockIdx.x == c.heads;
    auto *input = kv ? static_cast<__nv_bfloat16 *>(b.kv) + (size_t)token * 512 :
        static_cast<__nv_bfloat16 *>(b.q) + ((size_t)token * c.heads + blockIdx.x) * 512;
    if (mlaQuery && !kv) {
        const int d = threadIdx.x;
        mlaQuery[((size_t(token) * 32 + d / 16) * c.heads + blockIdx.x) * 16 + d % 16] = input[d];
        if (blockIdx.x == 0 && d < 32) {
            const float angle = float(requests[token].position) * rope.invFreq[d];
            mlaAngles[token * 64 + d] = cosf(angle);
            mlaAngles[token * 64 + 32 + d] = sinf(angle);
        }
        return;
    }
    const V41DecodeTransformDescriptor item{
        input, nullptr, 1, 512, requests[token].position, kv ? 1 : 0, 0, 32};
    V41DecodeTransformRow(item, 0, threadIdx.x, rope, c.ropeDim, c.window);
    if constexpr (PackVerify) {
        if (kv) {
            __syncthreads();
            if (threadIdx.x < kKvQuantThreads)
                V41QuantizeKVRow(input, static_cast<uint8_t *>(b.savedKV) + size_t(token) * c.ringRowBytes,
                                 512, 1, 32);
        }
    }
    if (kv && directRing) {
        // The preceding 127 positions cannot use position % 128. The
        // current token remains BF16 in the separate attention merge.
        __syncthreads();
        auto *saved = static_cast<uint8_t *>(b.savedKV) + size_t(token)*c.ringRowBytes;
        if (threadIdx.x < kKvQuantThreads) V41QuantizeKVRow(input, saved, 512, 1, 32);
        __syncthreads();
        auto *slot = static_cast<uint8_t *>(requests[token].windowKV) +
            size_t(requests[token].position % c.window)*c.ringRowBytes;
        for (int d = threadIdx.x; d < c.ringRowBytes; d += blockDim.x) {
            const auto value = saved[d];
            slot[d] = value;
            if (mlaRing) mlaRing[(size_t(token) * 128 + 127) * 528 + d] = value;
        }
    }
    // This lane owns the transformed element. Saving it here preserves the
    // exact KV bytes and avoids a separate launch after attention. The saved
    // rows are speculative scratch; committed ring/tail storage is untouched.
    if (kv && c.verifying && !c.ringMode)
        static_cast<__nv_bfloat16 *>(b.savedKV)[(size_t)token * 512 + threadIdx.x] = input[threadIdx.x];
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

__device__ __forceinline__ void V41StoreCompressedPage(const FastllmV41FullDecodeConfig &c,
        const FastllmV41FullDecodeRequest &r, V41RopeTable rope, __nv_bfloat16 latent,
        __nv_bfloat16 *transformed) {
    const int d = threadIdx.x;
    transformed[d] = latent;
    __syncthreads();
    const int pos = (r.position / c.ratio) * c.ratio;
    V41DecodeTransformRow({transformed, nullptr, 1, 512, pos, 3, 0, 16}, 0, d, rope, c.ropeDim, c.window);
    __syncthreads();
    if (d < kKvQuantThreads)
        V41QuantizeKVRow(transformed, static_cast<uint8_t *>(r.compressedKV) +
            size_t(r.position / c.ratio) * c.cmpRowBytes, 512, c.cmpMode, 16);
}

// Keep the pre-RoPE BF16 latent for Wk while writing the independent main KV
// record from shared memory. Each request owns its own C2 tail and output page.
__global__ void V41PublishCompressed(FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41RopeTable rope) {
    const int row = blockIdx.x, d = threadIdx.x;
    const auto r = requests[row];
    const bool ready = (r.position + 1) % c.ratio == 0;
    float value = static_cast<const float *>(b.rawKV)[size_t(row) * 512 + d];
    if (c.ratio == 2) {
        const float score = static_cast<const float *>(b.rawScore)[size_t(row) * 512 + d];
        const float old = ready ? static_cast<const float *>(r.rawTailKV)[d] : value;
        const float prev = ready ? static_cast<const float *>(r.rawTailScore)[d] : score;
        if (!ready) {
            static_cast<float *>(r.rawTailKV)[d] = value;
            static_cast<float *>(r.rawTailScore)[d] = score;
        }
        const float mx = fmaxf(fmaxf(-FLT_MAX, prev), score);
        const float a = __expf(prev - mx), z = __expf(score - mx);
        float acc = 0.f;
        acc += a * old; acc += z * value;
        value = acc / (a + z);
    }
    __shared__ float partial[32];
    __shared__ __nv_bfloat16 transformed[512];
    value = V41Bf16Round(value);
    const float ss = V41BlockSum<512>(value * value, partial);
    const auto latent = __float2bfloat16_rn(static_cast<const float *>(b.compressNorm)[d] * value *
                                            rsqrtf(ss / 512.f + c.normEps));
    static_cast<__nv_bfloat16 *>(b.latents)[size_t(row) * 512 + d] = latent;
    if (ready) V41StoreCompressedPage(c, r, rope, latent, transformed);
}

__global__ void V41PublishIndexK(FastllmV41FullDecodeConfig c, FastllmV41FullDecodeBuffers b,
        const FastllmV41FullDecodeRequest *requests, V41RopeTable rope) {
    const int row = blockIdx.x, d = threadIdx.x;
    const auto r = requests[row];
    if ((r.position + 1) % c.ratio) return;
    const auto *key = static_cast<const __nv_bfloat16 *>(b.newIndexKeys) + row * 128;
    __shared__ float partial[4];
    __shared__ __nv_bfloat16 norm[128];
    float sum = 0.f;
    if (d < 64) {
        const auto v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(key)[d]);
        sum += v.x * v.x + v.y * v.y;
    }
    // Match the 64-thread BF16 RMSNorm tree (two nonzero warp sums).
    const float ss = V41BlockSum<128>(sum, partial);
    norm[d] = __float2bfloat16_rn(__bfloat162float(key[d]) * rsqrtf(ss / 128.f + c.normEps) * b.indexNorm[d]);
    __syncthreads();
    V41DecodeTransformRow({norm, nullptr, 1, 128, (r.position / c.ratio) * c.ratio, 2, 0, 32},
                          0, d, rope, c.ropeDim, c.window);
    __syncthreads();
    V41QuantizeKVRow(norm, static_cast<uint8_t *>(r.indexK) + size_t(r.position / c.ratio) * c.indexRowBytes,
                     128, c.indexMode, 32);
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

// Quantization uses the ordinary storage encoder, including scale edge cases.
// Only complete compressor groups are appended; speculative ring rows remain
// in scratch until acceptance selects the prefix to commit.
__global__ void V41FullVerifyPackWindow(FastllmV41FullDecodeConfig c,
                                      FastllmV41FullDecodeBuffers b) {
    const int row = blockIdx.x;
    V41QuantizeKVRow(static_cast<const __nv_bfloat16 *>(b.kv) + (size_t)row * 512,
        static_cast<uint8_t *>(b.savedKV) + (size_t)row * c.ringRowBytes, 512, 1, 32);
}

__global__ void V41FullDecodeAppendPacked(FastllmV41FullDecodeConfig c,
        FastllmV41FullDecodeBuffers b, const FastllmV41FullDecodeRequest *requests) {
    const int token = blockIdx.x;
    const auto r = requests[token];
    if ((r.position + 1) % c.ratio) return;
    const int row = r.position / c.ratio;
    V41QuantizeKVRow(static_cast<const __nv_bfloat16 *>(b.latents) + (size_t)token * 512,
        static_cast<uint8_t *>(r.compressedKV) + (size_t)row * c.cmpRowBytes,
        512, c.cmpMode, c.cmpMode == 3 ? 16 : 32);
    if (c.indexSource)
        V41QuantizeKVRow(static_cast<const __nv_bfloat16 *>(b.newIndexKeys) + (size_t)token * 128,
            static_cast<uint8_t *>(r.indexK) + (size_t)row * c.indexRowBytes, 128, c.indexMode, 32);
}

// Store after attention has consumed the old ring. The current row remains
// BF16 during attention, matching ordinary decode's quantization boundary.
__global__ void V41FullDecodeStorePackedRing(FastllmV41FullDecodeConfig c,
        FastllmV41FullDecodeBuffers b, const FastllmV41FullDecodeRequest *requests) {
    const int row = blockIdx.x;
    const auto request = requests[row];
    auto *dst = static_cast<uint8_t *>(request.windowKV) +
        size_t(request.position % c.window) * c.ringRowBytes;
    const auto *src = static_cast<const uint8_t *>(b.savedKV) + size_t(row) * c.ringRowBytes;
    for (int i = threadIdx.x; i < c.ringRowBytes; i += blockDim.x) dst[i] = src[i];
}

// Verification never overwrites the committed ring or compressor tail.
// Save a contiguous [old tail, current chunk] stream for any accepted prefix.
template<bool Compress, bool Publish = false>
__global__ void V41FullVerifyGather(FastllmV41FullDecodeConfig c,
        FastllmV41FullDecodeBuffers b, const FastllmV41FullDecodeRequest *requests, V41RopeTable rope) {
    const int row = blockIdx.x, d = threadIdx.x;
    const auto r = requests[row];
    const int begin = r.chunkBegin, localRow = row - begin;
    const int tail = requests[begin].position % c.ratio;
    const size_t savedRow = size_t(2) * begin + tail + localRow;
    float value = static_cast<const float *>(b.rawKV)[(size_t)row * 512 + d];
    static_cast<float *>(b.savedRawKV)[(size_t)savedRow * 512 + d] = value;
    if (c.ratio == 1) {
        if constexpr (!Compress) static_cast<float *>(b.joined)[size_t(row) * 512 + d] = value;
    } else {
        const float score = static_cast<const float *>(b.rawScore)[(size_t)row * 512 + d];
        const bool ready = (r.position + 1) % c.ratio == 0;
        const float previous = localRow ? static_cast<const float *>(b.rawKV)[(size_t)(row - 1) * 512 + d]
                                   : (tail ? static_cast<const float *>(r.rawTailKV)[d] : value);
        const float previousScore = localRow ? static_cast<const float *>(b.rawScore)[(size_t)(row - 1) * 512 + d]
                                        : (tail ? static_cast<const float *>(r.rawTailScore)[d] : score);
        if constexpr (Compress) {
            const float left = ready ? previous : value, leftScore = ready ? previousScore : score;
            const float peak = fmaxf(fmaxf(-FLT_MAX, leftScore), score);
            const float a = __expf(leftScore - peak), z = __expf(score - peak);
            float acc = 0.f;
            acc += a * left;
            acc += z * value;
            value = acc / (a + z);
        } else {
            static_cast<float *>(b.joined)[(size_t)row * 1024 + d] = ready ? previous : value;
            static_cast<float *>(b.joined)[(size_t)row * 1024 + 512 + d] = value;
            static_cast<float *>(b.scores)[(size_t)row * 1024 + d] = ready ? previousScore : score;
            static_cast<float *>(b.scores)[(size_t)row * 1024 + 512 + d] = score;
        }
        static_cast<float *>(b.savedRawScore)[(size_t)savedRow * 512 + d] = score;
        if (localRow == 0 && tail) {
            static_cast<float *>(b.savedRawKV)[size_t(2) * begin * 512 + d] = previous;
            static_cast<float *>(b.savedRawScore)[size_t(2) * begin * 512 + d] = previousScore;
        }
    }
    if constexpr (Compress) {
        __shared__ float partial[32];
        value = V41Bf16Round(value);
        const float ss = V41BlockSum<512>(value * value, partial);
        const auto latent = __float2bfloat16_rn(static_cast<const float *>(b.compressNorm)[d] * value *
                                                rsqrtf(ss / 512.f + c.normEps));
        static_cast<__nv_bfloat16 *>(b.latents)[size_t(row) * 512 + d] = latent;
        if constexpr (Publish) {
            __shared__ __nv_bfloat16 transformed[512];
            if ((r.position + 1) % c.ratio == 0)
                V41StoreCompressedPage(c, r, rope, latent, transformed);
        }
    }
}

template<int SPLITS, bool MATERIALIZED, bool PIPELINED = false, bool QK_PREFETCH = false>
bool V41FullDecodeAttention(V41FullDecodePlan &p) {
    const auto &c = p.config;
    const int headBlocks = (c.heads + kMmaHeads - 1) / kMmaHeads;
    using SerialShared = typename std::conditional<MATERIALIZED, V41MmaPvShared<SPLITS>, V41MmaShared>::type;
    using Shared = typename std::conditional<PIPELINED, V41MmaPvPipelineShared<SPLITS>, SerialShared>::type;
    if constexpr (MATERIALIZED && QK_PREFETCH) {
        if (!V41LaunchDependent<V41SparseAttentionScoresKernel<true, true>>(
            dim3(c.batch, headBlocks, p.tiles * (kMmaNC / kQkPrefetchCandidates)), kQkPrefetchCandidates * 4,
            sizeof(V41MmaScoresPrefetchShared),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, c.heads, c.window,
            0, 0, 0, nullptr, 0, 0, 0, 0, 0, p.attention)) return false;
    } else if constexpr (MATERIALIZED) {
        if (!V41LaunchDependent<V41SparseAttentionScoresKernel<true>>(dim3(c.batch, headBlocks, p.tiles), kMmaThreads, sizeof(V41MmaShared),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, c.heads, c.window,
            0, 0, 0, nullptr, 0, 0, 0, 0, 0, p.attention)) return false;
    }
    return V41LaunchDependent<V41SparseAttentionMmaKernel<SPLITS, MATERIALIZED, true, PIPELINED>>(dim3(c.batch, headBlocks, SPLITS), kMmaThreads, sizeof(Shared),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        static_cast<const float *>(p.buffers.sink), 1, c.heads, c.window, 0, 0, 0, c.softmaxScale,
        nullptr, 0, 0, 0, 0, 0, nullptr, p.attention);
}
} // namespace

void *FastllmCudaV41CreateFullDecode(const FastllmV41FullDecodeConfig &c,
                                   const FastllmV41FullDecodeBuffers &b) {
    if (FastllmCudaGraphIsCapturingFast() || c.batch <= 0 || c.heads <= 0 ||
        c.indexHeads <= 0 || c.ratio < 0 || c.ratio > 2 || c.window <= 0 ||
        c.maxKeys <= 0 || c.topK <= 0 || c.topK > c.maxKeys || c.blockSize <= 0 ||
        c.ropeDim <= 0 || c.ropeDim > 128 || (c.ropeDim & 1) ||
        (c.ringMode && (!b.savedKV || c.ringMode != 1 || c.ringRowBytes != 528 ||
            !((c.cmpMode == 3 && c.cmpRowBytes == 288 && c.indexMode == 2 && c.indexRowBytes == 68) ||
              (c.cmpMode == 1 && c.cmpRowBytes == 528 && c.indexMode == 1 && c.indexRowBytes == 132)))) ||
        !b.q || !b.kv || !b.output || !b.sink ||
        (c.verifying && (!b.savedKV || (c.kvSource &&
            (!b.savedRawKV || (c.ratio == 2 && !b.savedRawScore))))) ||
        (c.ratio && (!b.topk || !b.candidates)) ||
        (c.kvSource && (!c.ratio || !b.rawKV || !b.joined || !b.latents || !b.compressNorm ||
          (c.ratio == 2 && (!b.rawScore || !b.scores)))) ||
        (c.indexSource && (!c.ratio || !b.indexQ || !b.indexWeights || (c.kvSource && !b.newIndexKeys))) ||
        (bool(b.candidateBlocks) != bool(b.candidateCounts)) ||
        (b.reuseCandidates && !b.candidateBlocks) ||
        (b.candidateBlocks && (!c.indexSource || !c.useCandidates || c.writeCandidates)))
        return nullptr;
    auto p = std::make_unique<V41FullDecodePlan>();
    p->config = c; p->config.vectorUnpack = V41Fp4VecUnpack();
    p->buffers = b; p->device = FastllmCudaGetDevice();
    if (cudaDeviceGetAttribute(&p->gridY, cudaDevAttrMaxGridDimY, p->device) != cudaSuccess || p->gridY < 2)
        return nullptr;
    p->tiles = (c.window + (c.ratio ? c.topK : 0) + kMmaNC - 1) / kMmaNC;
    const int heads = (c.heads + kMmaHeads - 1) / kMmaHeads;
    // Verification rows use the same independent QK tiles and value-dimension
    // partitioning as decode. This keeps the 64-key online-softmax recurrence
    // intact while exposing enough CTAs for small batches (including TP8).
    // Large verification batches already have enough independent CTAs and
    // retain the fused path; DSpark verification chunks contain 2--8 rows.
    p->splits = (c.verifying && c.batch > 8) || int64_t(c.batch) * heads >= 128 ? 1 :
                c.batch >= 64 ? 2 : 8;
    // Dynamic descriptors also feed the ordinary scalar kernels. A missing
    // BF16 tensor-core image (e.g. SM75) must not disable the whole graph.
    p->scalarAttention = !V41MmaKernelReady<V41SparseAttentionScoresKernel<true>, kMmaThreads, sizeof(V41MmaShared)>();
    if (!p->scalarAttention && p->splits == 1) {
        p->fused = V41MmaKernelReady<V41SparseAttentionMmaKernel<1, false, true>, kMmaThreads, sizeof(V41MmaShared)>();
        p->scalarAttention = !p->fused && !V41MmaKernelReady<V41SparseAttentionMmaKernel<1, true, true>, kMmaThreads, sizeof(V41MmaPvShared<1>)>();
    } else if (!p->scalarAttention && p->splits == 2) {
        p->scalarAttention = !V41MmaKernelReady<V41SparseAttentionMmaKernel<2, true, true>, kMmaThreads, sizeof(V41MmaPvShared<2>)>();
    } else if (!p->scalarAttention) {
        p->scalarAttention = !V41MmaKernelReady<V41SparseAttentionMmaKernel<8, true, true>, kMmaThreads, sizeof(V41MmaPvShared<8>)>();
    }
    p->scalarIndex = c.indexSource &&
        (!V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true, true>, 256>() ||
         (c.useCandidates && !c.writeCandidates &&
          !V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true, true, false, true>, 256>()));
    if (FastllmCudaGetThreadError()) return nullptr;
    // The full verification plan uses BF16 KV. On SM90, overlap each
    // 64-candidate tile's math with the next KV/score gather. Keep the
    // existing path for other architectures, decode and other head groups.
    p->pvPipeline = !c.ringMode && c.verifying && c.batch >= 2 && c.batch <= 8 && c.heads == 8 &&
        V41IsSm<9, 0>() && V41MmaKernelReady<
            V41SparseAttentionMmaKernel<8, true, true, true>, kMmaThreads,
            sizeof(V41MmaPvPipelineShared<8>)>();
    // Only the SM90 small verification BF16 path uses the smaller QK
    // producer tile. Keep decode, other head counts and precisions unchanged.
    p->qkPrefetch = p->pvPipeline && V41MmaKernelReady<
        V41SparseAttentionScoresKernel<true, true>, kQkPrefetchCandidates * 4,
        sizeof(V41MmaScoresPrefetchShared)>();
    p->wideTopKMerge = c.batch <= 8 &&
        V41MmaKernelReady<V41TopKKernel<true, 2, 1024>, 1024>();
    p->indexPrefetch = !c.indexMode && c.indexSource && !c.useCandidates && c.indexHeads <= 32 &&
        V41IsSm<9, 0>() && V41MmaKernelReady<
            V41IndexerScoreDecodeMmaKernel<true, true, false, false, true>, 256>();
    p->wideCandidateSelect = c.writeCandidates && c.batch <= 8 &&
        V41MmaKernelReady<V41CandidateSelectKernel<true, 1024>, 1024>();
    p->fuseIndexBlocks = c.indexSource && c.writeCandidates && c.blockSize <= 32 &&
        (c.blockSize & (c.blockSize - 1)) == 0 &&
        V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true, true, false, false, false, true>, 256>();
    if (p->fuseIndexBlocks && p->indexPrefetch &&
        !V41MmaKernelReady<V41IndexerScoreDecodeMmaKernel<true, true, false, false, true, true>, 256>()) p->indexPrefetch = false;
    p->indexPairs = !c.indexMode && c.verifying && !c.segmentedVerify && c.batch >= 2 && c.batch <= 8 && c.indexSource &&
        !c.useCandidates && c.indexHeads <= 32 &&
        V41MmaKernelReady<V41IndexerScorePairKernel<false>, 256, sizeof(V41PairedIndexShared)>() &&
        (!p->fuseIndexBlocks ||
         V41MmaKernelReady<V41IndexerScorePairKernel<true>, 256, sizeof(V41PairedIndexShared)>());
    // SM90 can reuse each query MMA operand across two eight-key tiles
    // per warp. Keep the 256-thread implementation as the general path.
    p->indexPairNarrow = p->indexPairs && c.indexHeads == 32 && V41IsSm<9, 0>() &&
        V41MmaKernelReady<V41IndexerScorePairKernel<false, 128>, 128, sizeof(V41PairedIndexShared)>() &&
        (!p->fuseIndexBlocks ||
         V41MmaKernelReady<V41IndexerScorePairKernel<true, 128>, 128, sizeof(V41PairedIndexShared)>());
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
    if (c.indexSource && c.maxKeys > kTopKParallelMinKeys && c.topK <= 512 &&
        c.blockSize <= kTopKShardKeys) {
        const bool globalTopK = !c.useCandidates && c.topK == 512 && c.maxKeys >= 65536 &&
            c.maxKeys <= 1024 * kTopKShardKeys && c.batch <= 8 &&
            V41MmaKernelReady<V41TopKGlobalFinish<true>, 1024>();
        const bool compactTopK = c.useCandidates && !c.writeCandidates && c.topK == 512 &&
            c.batch <= 8 && p->wideTopKMerge;
        auto layout = V41TopKLayout(nullptr, c.batch, c.maxKeys, c.topK, c.blockSize, globalTopK, compactTopK);
        ok = ok && alloc(&p->topKMemory, layout.Bytes(c.batch));
        p->topKWorkspace = V41TopKLayout(p->topKMemory, c.batch, c.maxKeys, c.topK, c.blockSize, globalTopK, compactTopK);
    }
    if (c.indexSource && c.useCandidates && !c.writeCandidates) {
        p->candidateBlocks = static_cast<int *>(b.candidateBlocks);
        p->candidateCounts = static_cast<int *>(b.candidateCounts);
        if (!p->candidateBlocks)
            ok = ok && alloc(&p->candidateBlocks, (size_t)c.batch * blocks) && alloc(&p->candidateCounts, c.batch);
    }
    if (!b.requests)
        ok = ok && cudaHostAlloc(reinterpret_cast<void **>(&p->host), (size_t)c.batch * sizeof(*p->host), cudaHostAllocDefault) == cudaSuccess &&
             cudaEventCreateWithFlags(&p->copied, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) { cudaGetLastError(); return nullptr; }
#ifdef FASTLLM_V41_FP16_ATTENTION
    if (V41IsSm75() && FastllmV41FlashInferEnabled() && v41_fp16_attention::Ready()) {
        p->fp16WorkspaceBytes = v41_fp16_attention::Bytes(c.batch, c.heads, c.window + (c.ratio ? c.topK : 0));
        void *storage = nullptr;
        if (cudaMalloc(&storage, p->fp16WorkspaceBytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        const int device = p->device;
        p->fp16Workspace.reset(storage, [device](void *ptr) {
            int old = 0; cudaGetDevice(&old); cudaSetDevice(device); cudaFree(ptr); cudaSetDevice(old);
        });
    }
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE
    if (FastllmV41FlashInferEnabled())
        p->flashInfer = FastllmV41FlashInferCreate(c);
    p->packVerify = p->flashInfer && c.verifying && c.heads == 64 && c.ropeDim == 64 &&
        c.batch <= 32 && V41IsSm<10,3>();
    p->directRing = p->flashInfer && !c.verifying && c.window == 128 && c.ringMode == 1 &&
        V41IsSm<10,3>() && !V41EnvOn("FASTLLM_DSV41_DISABLE_FUSED_RING");
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    if (p->directRing && V41EnvOn("FASTLLM_DSV41_FLASHMLA")) {
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, p->device) == cudaSuccess && sms == 148)
            p->mla = FastllmV41CreateMla(c, V41EnvOn("FASTLLM_DSV41_MLA_EXACT_ATTENTION"));
        p->earlyHistory = p->mla && p->mla->paged && !p->mla->outputOnly && !c.verifying &&
            !c.kvSource && !c.indexSource && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_EARLY_HISTORY", c.batch);
        p->shortPages = p->mla && p->mla->paged && !p->mla->outputOnly && c.indexSource && !c.verifying &&
            FastllmV41HcEnabledAtRows("FASTLLM_DSV41_SHORT_PAGES", c.batch);
    }
#endif
    return p.release();
}

bool FastllmCudaV41MlaOutput(void *handle, void *&codes, void *&scales, int &pitch, bool &headMajor) {
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    auto *p = static_cast<V41FullDecodePlan *>(handle);
#if defined(FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE) && defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103)
    if (p && p->config.batch == 6 && FastllmV41FlashInferVerifyAngles(p->flashInfer)) {
        // This graph slot normally holds BF16 [rows,64,512]. Its packed codes
        // and K128-major UE8M0 words fit in the same aligned allocation.
        codes = p->buffers.output;
        scales = static_cast<uint8_t *>(codes) + size_t(p->config.batch) * 64 * 512;
        pitch = (p->config.batch + 3) / 4 * 4;
        headMajor = true;
        return true;
    }
#endif
    if (p && p->mla) {
        codes = p->mla->codes; scales = p->mla->scales; pitch = p->mla->scalePitch; headMajor = p->mla->headMajor;
        return true;
    }
#endif
    return false;
}

void FastllmCudaV41UseMlaOutput(void *handle) {
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    static_cast<V41FullDecodePlan *>(handle)->directMlaOutput = true;
#endif
}
void FastllmCudaV41DestroyFullDecode(void *plan) { delete static_cast<V41FullDecodePlan *>(plan); }
bool FastllmCudaV41ShareAttentionWorkspace(void *handle, void *source) {
    auto *p = static_cast<V41FullDecodePlan *>(handle), *q = static_cast<V41FullDecodePlan *>(source);
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    if (p && q && p->device == q->device && p->mla && q->mla &&
        p->mla->batch == q->mla->batch && p->mla->width == q->mla->width &&
        p->mla->outputOnly == q->mla->outputOnly && p->mla->paged == q->mla->paged &&
        p->mla->bf16Rounding == q->mla->bf16Rounding && p->mla->headMajor == q->mla->headMajor)
        p->mla = q->mla;
#endif
#ifdef FASTLLM_V41_FP16_ATTENTION
    if (p && q && p->device == q->device && p->fp16Workspace && q->fp16Workspace &&
        p->fp16WorkspaceBytes <= q->fp16WorkspaceBytes) {
        p->fp16Workspace = q->fp16Workspace;
        return true;
    }
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE
    return p && q && p->device == q->device &&
        FastllmV41FlashInferShareWorkspace(p->flashInfer, q->flashInfer);
#else
    (void)handle; (void)source;
    return false;
#endif
}

namespace {
bool V41ValidateFullDecodeRequests(const V41FullDecodePlan *p,
        const FastllmV41FullDecodeRequest *requests, size_t count) {
    if (!p || !requests || count != (size_t)p->config.batch ||
        FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config;
    for (size_t i = 0; i < count; ++i) {
        const auto &r = requests[i];
        if (c.verifying) {
            const int begin = r.chunkBegin, length = r.chunkLength > 0 ? r.chunkLength : c.batch;
            if (begin < 0 || begin > (int)i || length < 1 || length > c.batch - begin ||
                (int)i >= begin + length || (!c.segmentedVerify && (begin || length != c.batch))) return false;
            const auto &first = requests[begin];
            if (first.chunkBegin != begin || first.chunkLength != r.chunkLength ||
                r.position != first.position + (int)i - begin ||
                r.windowKV != first.windowKV || r.compressedKV != first.compressedKV ||
                r.indexK != first.indexK || r.rawTailKV != first.rawTailKV ||
                r.rawTailScore != first.rawTailScore) return false;
            if ((int)i == begin && begin > 0) {
                const auto &previous = requests[begin - 1];
                if (previous.chunkBegin + previous.chunkLength != begin) return false;
                // Distinct causal chunks cannot write the same ring concurrently.
                for (int j = 0; j < begin; j += requests[j].chunkLength)
                    if (requests[j].chunkLength < 1 || requests[j].windowKV == r.windowKV) return false;
            }
        }
        if (r.position < 1 || !r.windowKV || (c.ratio &&
            (!r.compressedKV || (r.position + 1LL) / c.ratio > c.maxKeys)) ||
            (c.indexSource && !r.indexK) ||
            (c.kvSource && c.ratio > 1 && (!r.rawTailKV || !r.rawTailScore))) return false;
    }
    return true;
}
} // namespace

bool FastllmCudaV41UploadAuxiliaryInputs(const FastllmV41HostCopy *copies, size_t count) {
    if (!count) return true;
    return copies && !FastllmCudaGraphIsCapturingFast() &&
        V41UploadCopies(copies[0], copies + 1, count - 1);
}

bool FastllmCudaV41UploadFullDecodeInputs(void *storage, void *const *plans, size_t layers,
        const FastllmV41FullDecodeRequest *requests, size_t batch,
        const FastllmV41HostCopy *extra, size_t extraCount) {
    if (extraCount == SIZE_MAX || (extraCount && !extra) ||
        !storage || !plans || !requests || !layers || !batch ||
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
    return V41UploadCopies({storage, requests, layers * batch * sizeof(*requests)}, extra, extraCount);
}

namespace {
__global__ void V41AdvanceFixedPositions(FastllmV41FullDecodeRequest *requests, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) ++requests[i].position;
}
}
namespace {
template<class Weight>
__global__ void V41OwnerEmbedding(const Weight *weight, const int *ids, __nv_bfloat16 *hidden,
        float *mix, int dim, int vocab, const int *positions,
        FastllmV41FullDecodeRequest *requests, int layers) {
    const int row = blockIdx.y;
    if (positions && !blockIdx.x) {
        for (int layer = threadIdx.x; layer < layers; layer += blockDim.x)
            requests[size_t(layer) * gridDim.y + row].position = positions[row];
    }
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (!blockIdx.x && threadIdx.x < 4) mix[row * 4 + threadIdx.x] = threadIdx.x == 0 ? 1.f : 0.f;
    if (col >= dim) return;
    // Replay validates IDs on the host. Bound even the capture/warmup input.
    const int id = ids[row];
    const auto *src = weight + size_t(unsigned(id) < unsigned(vocab) ? id : 0) * dim + col;
    alignas(8) __nv_bfloat16 value[4];
    if constexpr (std::is_same_v<Weight, __nv_bfloat16>)
        *reinterpret_cast<uint2 *>(value) = *reinterpret_cast<const uint2 *>(src);
    else {
        const auto x = *reinterpret_cast<const float4 *>(src);
        value[0] = __float2bfloat16_rn(x.x); value[1] = __float2bfloat16_rn(x.y);
        value[2] = __float2bfloat16_rn(x.z); value[3] = __float2bfloat16_rn(x.w);
    }
    for (int channel = 0; channel < 4; ++channel)
        *reinterpret_cast<uint2 *>(hidden + (size_t(row) * 4 + channel) * dim + col) =
            *reinterpret_cast<const uint2 *>(value);
}
}
bool FastllmCudaV41OwnerEmbedding(const void *weight, bool fp32, const int *ids, void *hidden,
                                  float *firstMix, int rows, int dim, int vocab) {
    return FastllmCudaV41OwnerEmbedding(weight, fp32, ids, hidden, firstMix, rows, dim, vocab, nullptr, nullptr, 0);
}
bool FastllmCudaV41OwnerEmbedding(const void *weight, bool fp32, const int *ids, void *hidden,
                                  float *firstMix, int rows, int dim, int vocab,
                                  const int *positions, FastllmV41FullDecodeRequest *requests, int layers) {
    if (positions && (!requests || layers <= 0)) return false;
    if (vocab < 1 || !weight || !ids || !hidden || !firstMix || rows < 1 || dim < 4 || dim % 4) return false;
    dim3 grid((dim + 511) / 512, rows);
    auto *dst = static_cast<__nv_bfloat16 *>(hidden);
    if (fp32) V41OwnerEmbedding<<<grid,128,0,cudaStreamPerThread>>>(static_cast<const float *>(weight),ids,dst,firstMix,dim,vocab,positions,requests,layers);
    else V41OwnerEmbedding<<<grid,128,0,cudaStreamPerThread>>>(static_cast<const __nv_bfloat16 *>(weight),ids,dst,firstMix,dim,vocab,positions,requests,layers);
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaV41AdvancePositions(void *storage, size_t count) {
    if (!storage || !count || count > INT_MAX || FastllmCudaGraphIsCapturingFast()) return false;
    V41AdvanceFixedPositions<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<FastllmV41FullDecodeRequest *>(storage), count);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaV41UploadFullDecodeBatch(void *storage, void *const *plans, size_t layers,
        const FastllmV41FullDecodeRequest *requests, size_t batch) {
    return FastllmCudaV41UploadFullDecodeInputs(storage, plans, layers, requests, batch, nullptr, 0);
}

namespace {
// Both packet formats use the same compressed history, signed modulo and padding rule.
__device__ void V41VerifyEngramRow(const int *ids, int anchor, int row, const int *map,
        int vocab, int pad, int imageToken, const int64_t *parameters, int layer, int ngram, int heads,
        FastllmV41VerifyHistory history, int64_t *buckets, float *mask) {
    const int cols = (ngram - 1) * heads;
    auto compress = [&](int id) { return id >= 0 && id < vocab && id != imageToken ? map[id] : -1; };
    if (!threadIdx.x) *mask = compress(row ? ids[row] : anchor) < 0 ? 0.f : 1.f;
    const int64_t *mul = parameters + layer * (ngram + 2 * cols);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        uint64_t hash = 0;
        bool blocked = false;
        for (int shift = 0; shift <= col / heads + 1; ++shift) {
            const int i = row - shift;
            const int compressed = i < 0 ? history.previous[-1-i] : compress(i ? ids[i] : anchor);
            blocked |= compressed < 0;
            hash ^= uint64_t(blocked ? pad : compressed) * uint64_t(mul[shift]);
        }
        const int64_t prime = mul[ngram + col];
        const int64_t bucket = int64_t(hash) % prime;
        buckets[col] = mul[ngram + cols + col] + (bucket < 0 ? bucket + prime : bucket);
    }
}

__global__ void V41DraftVerifyPacket(const int *ids, int anchor, int rows, const int *map, int vocab,
        int pad, int imageToken, const int64_t *parameters, int layers, int ngram, int heads,
        FastllmV41VerifyHistory history, int64_t *result) {
    const int row = blockIdx.x, layer = blockIdx.y, cols = (ngram - 1) * heads;
    const size_t index = size_t(layer) * rows + row;
    auto *masks = reinterpret_cast<float *>(result + size_t(layers) * rows * cols);
    V41VerifyEngramRow(ids, anchor, row, map, vocab, pad, imageToken, parameters, layer, ngram, heads,
                       history, result + index * cols, masks + index);
    if (!layer && !threadIdx.x) masks[layers * rows + row] = float(row ? ids[row] : anchor);
}

__global__ void V41OwnerVerifyPacket(int *ids, int anchor, bool positions, int start, const int *map,
        int vocab, int pad, int imageToken, const int64_t *parameters, int ngram, int heads,
        FastllmV41VerifyHistory history) {
    constexpr int rows = FastllmV41OwnerVerifyRows;
    const int row = blockIdx.x, layer = blockIdx.y, cols = (ngram - 1) * heads;
    auto *result = reinterpret_cast<int64_t *>(ids + (positions ? 2 * rows : rows));
    auto *layerRows = result + size_t(layer) * (rows * cols + rows / 2);
    auto *masks = reinterpret_cast<float *>(layerRows + rows * cols);
    V41VerifyEngramRow(ids, anchor, row, map, vocab, pad, imageToken, parameters, layer, ngram, heads,
                       history, layerRows + row * cols, masks + row);
    if (!layer && !threadIdx.x) {
        if (!row) ids[0] = anchor;
        if (positions) ids[rows + row] = start + row;
    }
}
}
namespace {
struct V41VerifyPageSlot {
    FastllmV41FullDecodeRequest *layers = nullptr, *expanded = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr, consumed = nullptr;
    int device = -1, count = 0, rows = 0;
    bool published = false;
    ~V41VerifyPageSlot() {
        int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (consumed) { cudaEventSynchronize(consumed); cudaEventDestroy(consumed); }
        if (ready) cudaEventDestroy(ready);
        if (stream) cudaStreamDestroy(stream);
        if (layers) cudaFree(layers);
        cudaSetDevice(old);
    }
};
__global__ void V41StageVerifyPages(const FastllmV41FullDecodeRequest *layers,
        FastllmV41FullDecodeRequest *expanded, int count, int rows, int position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * rows) return;
    auto r = layers[i / rows];
    r.position = position + i % rows; r.chunkBegin = 0; r.chunkLength = rows;
    expanded[i] = r;
}
}
void *FastllmCudaV41CreateVerifyPages(const FastllmV41FullDecodeRequest *layers, int count, int rows) {
    if (!layers || count < 1 || rows < 1 || int64_t(count) * (rows + 1LL) > INT_MAX) return nullptr;
    auto slot = std::make_unique<V41VerifyPageSlot>();
    slot->device = FastllmCudaGetDevice(); slot->count = count; slot->rows = rows;
    if (cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&slot->ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&slot->consumed, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc(&slot->layers, sizeof(*layers)*count*(rows+1)) != cudaSuccess ||
        cudaMemcpy(slot->layers, layers, sizeof(*layers)*count, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    slot->expanded = slot->layers + count;
    return slot.release();
}
void FastllmCudaV41DestroyVerifyPages(void *slot) { delete static_cast<V41VerifyPageSlot *>(slot); }
bool FastllmCudaV41StageVerifyPages(void *handle, int position) {
    auto *s = static_cast<V41VerifyPageSlot *>(handle);
    if (!s || s->device != FastllmCudaGetDevice() || position < 1 || int64_t(position)+s->rows > INT_MAX ||
        cudaStreamWaitEvent(s->stream, s->consumed) != cudaSuccess) return false;
    V41StageVerifyPages<<<(s->rows*s->count+127)/128,128,0,s->stream>>>(s->layers,s->expanded,s->count,s->rows,position);
    s->published = cudaGetLastError() == cudaSuccess && cudaEventRecord(s->ready,s->stream) == cudaSuccess;
    return s->published;
}
bool FastllmCudaV41ConsumeVerifyPages(void *handle, void *destination) {
    auto *s = static_cast<V41VerifyPageSlot *>(handle);
    if (!s || !destination || !s->published || FastllmCudaGetDevice()!=s->device) return false;
    if (cudaStreamWaitEvent(cudaStreamPerThread,s->ready) != cudaSuccess ||
        cudaMemcpyAsync(destination,s->expanded,sizeof(*s->expanded)*s->rows*s->count,
                        cudaMemcpyDeviceToDevice,cudaStreamPerThread) != cudaSuccess) return false;
    if (cudaEventRecord(s->consumed,cudaStreamPerThread) != cudaSuccess) {
        cudaStreamSynchronize(cudaStreamPerThread); return false;
    }
    return true;
}

namespace {
__global__ void V41PublishVerifyInputs(const uint32_t *pages, uint32_t *pageDestination, int pageWords,
        const uint32_t *packet, uint32_t *packetDestination, int packetWords) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pageWords) pageDestination[i] = pages[i];
    if (i < packetWords) packetDestination[i] = packet[i];
}
}

bool FastllmCudaV41PublishVerifyInputs(void *handle, void *pageDestination, void *packetDestination,
        const void *packet, size_t packetBytes, void *packetReady, void *lastPageRead, void *lastPacketRead) {
    auto *s = static_cast<V41VerifyPageSlot *>(handle);
    if (!s || !s->published || !pageDestination || !packetDestination || !packet || !packetReady ||
        !lastPageRead || !lastPacketRead || s->device != FastllmCudaGetDevice() ||
        packetBytes % sizeof(uint32_t) || packetBytes > INT_MAX) return false;
    static_assert(sizeof(*s->expanded) % sizeof(uint32_t) == 0);
    const int pageWords = sizeof(*s->expanded) / sizeof(uint32_t) * s->rows * s->count;
    const int packetWords = int(packetBytes / sizeof(uint32_t));
    // Local attention graph slots can be shared by differently sized parent
    // graphs. Fence both destinations before publishing on the page producer.
    for (auto event : {packetReady, lastPageRead, lastPacketRead})
        if (cudaStreamWaitEvent(s->stream, static_cast<cudaEvent_t>(event)) != cudaSuccess) return false;
    V41PublishVerifyInputs<<<(std::max(pageWords, packetWords) + 127) / 128, 128, 0, s->stream>>>(
        reinterpret_cast<const uint32_t *>(s->expanded), static_cast<uint32_t *>(pageDestination), pageWords,
        static_cast<const uint32_t *>(packet), static_cast<uint32_t *>(packetDestination), packetWords);
    if (cudaGetLastError() != cudaSuccess) return false;
    // A failed event operation must finish the side write before a caller
    // can republish the same destinations via the ordinary ordered path.
    if (cudaEventRecord(s->consumed, s->stream) == cudaSuccess &&
        cudaStreamWaitEvent(cudaStreamPerThread, s->consumed) == cudaSuccess) return true;
    cudaStreamSynchronize(s->stream);
    return false;
}

namespace {
__global__ void V41ResolveDraft(const float *target, int stride, const int *draft, int rows, int position,
        int offset, int maxAccepted, FastllmV41VerifyStops stops, int *packet) {
    int accepted = 0;
    int token = int(target[0] + 1e-3f);
    while (accepted < maxAccepted && token == draft[accepted + 1]) {
        bool stopped = false;
        for (int i = 0; i < stops.size; ++i) stopped |= stops.ids[i] == token;
        if (stopped) break;
        token = int(target[++accepted * stride] + 1e-3f);
    }
    packet[0] = accepted + 1; packet[1] = offset;
    packet[2] = position; packet[3] = accepted + 1 == rows ? rows : 0;
    packet[4] = position + accepted + 1; packet[5] = token;
}
}
bool FastllmCudaV41ResolveDraft(const float *target, int stride, const int *draft, int rows, int position,
        int offset, int maxAccepted, FastllmV41VerifyStops stops, int *packet) {
    if (!target || !draft || !packet || stride < 1 || rows < 1 || maxAccepted < 0 ||
        maxAccepted >= rows || stops.size < 0 || stops.size > 16 || position < 0 || offset < 0) return false;
    V41ResolveDraft<<<1, 1, 0, cudaStreamPerThread>>>(target, stride, draft, rows, position, offset, maxAccepted, stops, packet);
    return V41CheckLaunch("ResolveDraft");
}

bool FastllmCudaV41DraftVerifyPacketOnStream(const int *ids, int anchor, int rows, const int *map,
        int vocab, int pad, int imageToken, const int64_t *parameters, int layers, int ngram, int heads,
        FastllmV41VerifyHistory history, void *packet, void *stream) {
    if (!ids || !map || !parameters || !packet || rows < 1 || layers < 1 || ngram < 2 ||
        ngram > 9 || heads < 1 || anchor < 0 || anchor >= vocab) return false;
    V41DraftVerifyPacket<<<dim3(rows, layers), 64, 0, stream ? (cudaStream_t)stream : cudaStreamPerThread>>>(ids, anchor, rows, map, vocab,
        pad, imageToken, parameters, layers, ngram, heads, history, static_cast<int64_t *>(packet));
    return V41CheckLaunch("DraftVerifyPacket");
}

bool FastllmCudaV41BuildOwnerVerifyPacket(void *packet, bool positions, int start, int anchor,
        const int *map, int vocab, int pad, int imageToken, const int64_t *parameters,
        int layers, int ngram, int heads, FastllmV41VerifyHistory history) {
    if (!packet || !map || !parameters || layers < 1 || ngram < 2 || ngram > 9 || heads < 1 ||
        start < 0 || anchor < 0 || anchor >= vocab) return false;
    V41OwnerVerifyPacket<<<dim3(FastllmV41OwnerVerifyRows, layers), 64, 0, cudaStreamPerThread>>>(
        static_cast<int *>(packet), anchor, positions, start, map, vocab, pad, imageToken,
        parameters, ngram, heads, history);
    return V41CheckLaunch("OwnerVerifyPacket");
}

bool FastllmCudaV41DraftVerifyPacket(const int *ids, int anchor, int rows, const int *map,
        int vocab, int pad, int imageToken, const int64_t *parameters, int layers, int ngram, int heads,
        FastllmV41VerifyHistory history, void *packet) {
    return FastllmCudaV41DraftVerifyPacketOnStream(ids, anchor, rows, map, vocab, pad, imageToken,
        parameters, layers, ngram, heads, history, packet, nullptr);
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

namespace {
// Chunk physical grid.y only. Descriptors and causal chunks keep global row
// offsets; no logical request/verification batch is truncated or padded.
template<class F> void V41FullDecodeRowTiles(int rows, int gridY, F &&run) {
    for (int begin = 0; begin < rows;) {
        const int count = std::min(rows - begin, gridY);
        run(begin, count);
        begin += count;
    }
}
V41TopKWorkspace V41FullDecodeTopKSlice(V41TopKWorkspace w, int begin) {
    const size_t entries = size_t(begin) * w.parts * w.width;
    if (w.scores) w.scores += entries;
    if (w.indices) w.indices += entries;
    if (w.valid) w.valid += entries;
    if (w.histograms) w.histograms += size_t(begin) * w.parts * kTopKCoarseStride;
    if (w.counts) w.counts += begin;
    if (w.offsets) w.offsets += size_t(begin) * (w.parts + 1);
    return w;
}
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
// Four independent 128-channel heads per CTA. The BF16 GEMV result is
// widened exactly as in ToFloat32 before the existing FP32 multiplication.
__global__ void V41VerifyIndexQWeights(const V41DecodeTransformDescriptor *items,
        V41RopeTable rope, int ropeDim, int window, const uint16_t *raw, float *weights, float scale) {
    const auto item = items[blockIdx.y * kFullDecodeTransforms];
    const int head = blockIdx.x * 4 + threadIdx.x / 128, d = threadIdx.x % 128;
    if (head >= item.rows) return;
    V41DecodeTransformRow(item, head, d, rope, ropeDim, window);
    if (!d) {
        const size_t i = size_t(blockIdx.y) * item.rows + head;
        weights[i] = __uint_as_float(uint32_t(raw[i]) << 16) * scale;
    }
}
#endif

template<int COMBINED = 1>
void V41FullDecodeTransforms(const V41FullDecodePlan &p, int slot, int columns) {
    V41FullDecodeRowTiles(p.config.batch, p.gridY / COMBINED, [&](int begin, int rows) {
        V41DecodeTransformsKernel<kFullDecodeTransforms, COMBINED><<<dim3(columns, rows * COMBINED), 512, 0, cudaStreamPerThread>>>(
            p.transforms + size_t(begin) * kFullDecodeTransforms + slot, p.rope, p.config.ropeDim, p.config.window);
    });
}
} // namespace

#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
// Keep the original split reduction and BF16 boundary. The existing inverse
// RoPE transform writes its exact FP8 codes in O-A's interleaved input order.
__global__ void V41PackExactAttention(const V41DecodeTransformDescriptor *transforms,
        V41RopeTable rope, uint8_t *codes, uint8_t *scales, int pitch, int begin, bool headMajor) {
    const int token = begin + blockIdx.y;
    V41DecodeTransformRow<true>(transforms[size_t(token) * kFullDecodeTransforms],
        blockIdx.x, threadIdx.x, rope, 64, 128, codes, scales, token, pitch, headMajor);
}
#endif

bool FastllmCudaV41CanPublishSource(const void *handle) {
    const auto *p = static_cast<const V41FullDecodePlan *>(handle);
    return p && (p->shortPages || (p->packVerify && p->config.cmpMode == 3)) && p->config.kvSource && p->config.indexSource && p->buffers.indexNorm &&
        p->config.indexMode == 2 && (p->config.ratio == 1 || p->config.ratio == 2) &&
        !V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH") && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_SOURCE_PUBLISH", p->config.batch);
}
static bool V41PrepareDecode(void *handle, bool allIndices, bool publish) {
    auto *p = static_cast<V41FullDecodePlan *>(handle);
    if (!p || FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config; const auto &b = p->buffers;
    __nv_bfloat16 *packedQuery = nullptr;
    uint8_t *mlaRing = nullptr;
    float *angles = nullptr;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    if (p->earlyHistory || (allIndices && p->shortPages && !c.kvSource))
        mlaRing = static_cast<uint8_t *>(p->mla->keys);
    if (p->mla && !p->mla->outputOnly) { packedQuery = static_cast<__nv_bfloat16 *>(p->mla->query); angles = p->mla->angles; }
#endif
    bool nativeQuery = false;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    nativeQuery = p->mla && p->mla->nativeQuery;
#endif
#if defined(FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE) && defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103)
    if (auto *verifyAngles = FastllmV41FlashInferVerifyAngles(p->flashInfer)) {
        nativeQuery = true;
        angles = verifyAngles;
    }
#endif
    V41FullDecodeRowTiles(c.batch, p->gridY, [&](int begin, int rows) {
        auto prepare = nativeQuery ? V41FullDecodePrepareKernel<true> : V41FullDecodePrepareKernel<false>;
        if (p->packVerify) prepare = nativeQuery ? V41FullDecodePrepareKernel<true, false, true> :
                                                     V41FullDecodePrepareKernel<false, false, true>;
        const bool nativeVerify = nativeQuery && p->packVerify;
        const int columns = nativeVerify ? 2 : (nativeQuery ? 1 : c.heads + 1);
        prepare<<<dim3(columns, rows), nativeVerify ? 256 : 512, 0, cudaStreamPerThread>>>(
            c, b, p->requests, p->transforms, p->index, p->attention,
            p->indexScores, p->blockScores, p->attentionScores, p->tiles, p->rope,
            p->candidateBlocks, p->candidateCounts, begin, p->directRing, packedQuery, angles, mlaRing, V41DecodeKeyNorm{});
    });
    if (c.ringMode && !p->directRing && !p->packVerify)
        V41FullVerifyPackWindow<<<c.batch, kKvQuantThreads, 0, cudaStreamPerThread>>>(c, b);
    if (publish && c.verifying) {
        V41FullVerifyGather<true, true><<<c.batch, 512, 0, cudaStreamPerThread>>>(c, b, p->requests, p->rope);
    } else if (publish) {
        V41PublishCompressed<<<c.batch, 512, 0, cudaStreamPerThread>>>(c, b, p->requests, p->rope);
    } else if (c.kvSource && p->packVerify) {
        V41FullVerifyGather<true><<<c.batch, 512, 0, cudaStreamPerThread>>>(c, b, p->requests, p->rope);
    } else if (c.kvSource) {
        if (c.verifying)
            V41FullVerifyGather<false><<<c.batch, 512, 0, cudaStreamPerThread>>>(c, b, p->requests, p->rope);
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

bool FastllmCudaV41PrepareFullDecode(void *handle) { return V41PrepareDecode(handle, false, false); }
bool FastllmCudaV41PrepareSource(void *handle, bool allIndices) {
    return FastllmCudaV41CanPublishSource(handle) && V41PrepareDecode(handle, allIndices, true);
}
bool FastllmCudaV41PrepareAllIndices(void *handle, bool publishSource) {
    return (!publishSource || FastllmCudaV41CanPublishSource(handle)) && V41PrepareDecode(handle, true, publishSource);
}

// cuBLAS and other projection implementations can leave several capture
// frontiers. Record node identity without changing the graph's dependencies.
std::vector<void *> FastllmCudaV41CaptureNodes() {
#if defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103) && !defined(USE_ROCM) && CUDART_VERSION >= 12000
    cudaGraph_t graph = nullptr;
    cudaStreamCaptureStatus status;
#if CUDART_VERSION >= 13000
    const auto rc = cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, nullptr, nullptr, nullptr);
#else
    const auto rc = cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, nullptr, nullptr);
#endif
    size_t count = 0;
    if (rc != cudaSuccess || status != cudaStreamCaptureStatusActive || !graph ||
        cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess || !count) {
        cudaGetLastError();
        return {};
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) == cudaSuccess)
        return {nodes.begin(), nodes.end()};
    cudaGetLastError();
#endif
    return {};
}

// Store the actual CUDA stream frontier before the index-only Q projections.
void *FastllmCudaV41CaptureFrontier() {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12000
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t *deps = nullptr;
    size_t count = 0;
#if CUDART_VERSION >= 13000
    const auto rc = cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, nullptr, &deps, nullptr, &count);
#else
    const auto rc = cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, nullptr, &deps, &count);
#endif
    if (rc == cudaSuccess && status == cudaStreamCaptureStatusActive && count == 1) return deps[0];
    if (rc != cudaSuccess) cudaGetLastError();
#endif
    return nullptr;
}
void *FastllmCudaV41GraphPrefix(const void *source, const void *frontier, const void *compressorInput,
                                const void *beforeCompressor, bool trim) {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12000
    if (!source || !frontier) return nullptr;
    cudaGraph_t graph = nullptr;
    if (cudaGraphClone(&graph, (cudaGraph_t)source) != cudaSuccess) return nullptr;
    auto reject = [&]() -> void * { cudaGetLastError(); cudaGraphDestroy(graph); return nullptr; };
    cudaGraphNode_t last = nullptr;
    if (cudaGraphNodeFindInClone(&last, (cudaGraphNode_t)frontier, graph) != cudaSuccess) return reject();
    if (compressorInput && beforeCompressor) {
        cudaGraphNode_t ready = nullptr, query = nullptr, first = nullptr;
        size_t count = 0;
        if (cudaGraphNodeFindInClone(&ready, (cudaGraphNode_t)compressorInput, graph) != cudaSuccess ||
            cudaGraphNodeFindInClone(&query, (cudaGraphNode_t)beforeCompressor, graph) != cudaSuccess ||
            FastllmCudaGraphNodeGetDependentNodes(query, nullptr, &count) != cudaSuccess || count != 1 ||
            FastllmCudaGraphNodeGetDependentNodes(query, &first, &count) != cudaSuccess ||
            FastllmCudaGraphNodeGetDependencies(first, nullptr, &count) != cudaSuccess || count != 1 ||
            FastllmCudaGraphAddDependencies(graph, &ready, &first, 1) != cudaSuccess) return reject();
#if CUDART_VERSION >= 13000
        if (cudaGraphRemoveDependencies(graph, &query, &first, nullptr, 1) != cudaSuccess) return reject();
#else
        if (cudaGraphRemoveDependencies(graph, &query, &first, 1) != cudaSuccess) return reject();
#endif
        // The full path can reuse projection scratch in the subsequent indexer.
        // The short path joins both terminal nodes at its attention child.
        if (!trim) {
            if (FastllmCudaGraphNodeGetDependentNodes(last, nullptr, &count) != cudaSuccess || !count) return reject();
            std::vector<cudaGraphNode_t> next(count);
            if (FastllmCudaGraphNodeGetDependentNodes(last, next.data(), &count) != cudaSuccess) return reject();
            for (auto user : next)
                if (FastllmCudaGraphAddDependencies(graph, &query, &user, 1) != cudaSuccess) return reject();
            return graph;
        }
    } else if (!trim) return reject();
    // Only descendants of the recorded stream boundary belong to the suffix.
    // Unrelated nodes and their original PDL edge attributes survive the clone.
    std::vector<cudaGraphNode_t> todo{last}, suffix;
    while (!todo.empty()) {
        auto node = todo.back(); todo.pop_back();
        size_t n = 0;
        if (FastllmCudaGraphNodeGetDependentNodes(node, nullptr, &n) != cudaSuccess) return reject();
        std::vector<cudaGraphNode_t> next(n);
        if (n && FastllmCudaGraphNodeGetDependentNodes(node, next.data(), &n) != cudaSuccess) return reject();
        for (auto child : next) if (std::find(suffix.begin(), suffix.end(), child) == suffix.end()) {
            suffix.push_back(child); todo.push_back(child);
        }
    }
    if (suffix.empty()) return reject();
    for (auto n : suffix) if (cudaGraphDestroyNode(n) != cudaSuccess) return reject();
    return graph;
#else
    return nullptr;
#endif
}
__global__ void V41DecodeAllIndices(int *indices, int width) {
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        indices[size_t(blockIdx.x) * width + i] = i;
}
// A verify chunk owns one selection per real row. Future compressed keys
// in the same chunk are masked even when the whole chunk fits in Top-K.
__global__ void V41VerifyAllIndices(int *indices, const FastllmV41FullDecodeRequest *requests,
                                    int width, int ratio) {
    const int row = blockIdx.x;
    const int visible = (requests[row].position + 1) / ratio;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        indices[size_t(row) * width + i] = i < visible ? i : -1;
}

static bool V41FinishDecode(void *handle, bool allIndices, bool publish = false) {
    auto *p = static_cast<V41FullDecodePlan *>(handle);
    if (!p || FastllmCudaGetDevice() != p->device) return false;
    const auto &c = p->config; const auto &b = p->buffers;
    if (publish) {
        V41PublishIndexK<<<c.batch, 128, 0, cudaStreamPerThread>>>(c, b, p->requests, p->rope);
    } else if (c.kvSource) {
        V41FullDecodeTransforms(*p, 2, 1);
        if (c.indexSource)
            V41FullDecodeTransforms(*p, 3, 1);
        if (c.ringMode)
            V41FullDecodeAppendPacked<<<c.batch, kKvQuantThreads, 0, cudaStreamPerThread>>>(c, b, p->requests);
        else V41FullDecodeAppend<<<c.batch, 512, 0, cudaStreamPerThread>>>(p->requests,
            static_cast<const __nv_bfloat16 *>(b.latents), static_cast<const __nv_bfloat16 *>(b.newIndexKeys), c.ratio, c.indexSource);
    }
    if (allIndices && c.indexSource) {
        if (c.verifying)
            V41VerifyAllIndices<<<c.batch, 128, 0, cudaStreamPerThread>>>(
                static_cast<int *>(b.topk), p->requests, c.topK, c.ratio);
        else if (!p->shortPages)
            V41DecodeAllIndices<<<c.batch, 128, 0, cudaStreamPerThread>>>(static_cast<int *>(b.topk), c.topK);
    } else if (c.indexSource) {
        V41FullDecodeTransforms(*p, 4, (c.indexHeads + 3) / 4);
        V41FullDecodeRowTiles(c.batch, p->gridY, [&](int begin, int rows) {
            auto *index = p->index + begin;
            const auto topKWorkspace = V41FullDecodeTopKSlice(p->topKWorkspace, begin);
            if (p->scalarIndex) {
                if (V41IsSm75() && c.batch <= 8 && c.indexHeads == 32) {
                    V41IndexerScoreSmallBatch<__nv_bfloat16, __nv_bfloat16, true><<<
                        dim3((c.maxKeys + 7) / 8, rows), 256, 0, cudaStreamPerThread>>>(
                        nullptr, nullptr, nullptr, 1, 0, nullptr, nullptr, 0, 0, index);
                } else {
                    V41IndexerScoreKernel<__nv_bfloat16, __nv_bfloat16, true><<<
                        dim3((c.maxKeys + kIdxThreads - 1) / kIdxThreads, rows), kIdxThreads,
                        size_t(c.indexHeads) * 129 * sizeof(float), cudaStreamPerThread>>>(
                        nullptr, nullptr, nullptr, 1, c.indexHeads, 128, 0, nullptr, nullptr, 0, 0, index);
                }
            } else if (p->indexPairs) {
                if (p->indexPairNarrow)
                    V41LaunchIndexerScorePair<128>(index, rows, c.ratio, p->fuseIndexBlocks);
                else
                    V41LaunchIndexerScorePair<256>(index, rows, c.ratio, p->fuseIndexBlocks);
            } else if (p->fuseIndexBlocks) {
                if (p->indexPrefetch) {
                    V41IndexerScoreDecodeMmaKernel<true, true, false, false, true, true><<<dim3(128, rows), 256, 0, cudaStreamPerThread>>>(
                        nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, index);
                } else {
                    V41IndexerScoreDecodeMmaKernel<true, true, false, false, false, true><<<dim3(128, rows), 256, 0, cudaStreamPerThread>>>(
                        nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, index);
                }
            } else if (p->candidateBlocks) {
                if (!b.reuseCandidates)
                    V41CompactIndexCandidates<<<rows, 256, 0, cudaStreamPerThread>>>(index, c.blockSize, c.ratio);
                V41IndexerScoreDecodeMmaKernel<true, true, false, true><<<dim3(128, rows), 256, 0, cudaStreamPerThread>>>(
                    nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, index);
            } else if (p->indexPrefetch) {
                V41IndexerScoreDecodeMmaKernel<true, true, false, false, true><<<dim3(128, rows), 256, 0, cudaStreamPerThread>>>(
                    nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, index);
            } else {
                V41IndexerScoreDecodeMmaKernel<true, true><<<dim3(128, rows), 256, 0, cudaStreamPerThread>>>(
                    nullptr, nullptr, nullptr, 0, 0, c.ratio, 0, nullptr, nullptr, 0, 0, 0, index);
            }
            if (c.writeCandidates) {
                if (!p->fuseIndexBlocks)
                    V41BlockScoreKernel<true><<<rows, 256, 0, cudaStreamPerThread>>>(nullptr, 1, 0, c.blockSize, 0, c.ratio, 0, nullptr, index);
                if (p->wideCandidateSelect) {
                    V41CandidateSelectKernel<true, 1024><<<rows, 1024, 0, cudaStreamPerThread>>>(nullptr, 0, c.candidateTopK, nullptr, index);
                } else {
                    V41CandidateSelectKernel<true><<<rows, kSelThreads, 0, cudaStreamPerThread>>>(nullptr, 0, c.candidateTopK, nullptr, index);
                }
            }
            if (topKWorkspace.global) {
                V41TopKGlobalArgs a{nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift,
                    0, c.ratio, 0, p->listCap, nullptr, index};
                V41LaunchGlobalTopK<true>(a, topKWorkspace, rows);
            } else if (topKWorkspace.compactCandidates) {
                V41TopKCandidateGather<<<dim3(topKWorkspace.parts, rows), 256,
                    (size_t)p->listCap * sizeof(int), cudaStreamPerThread>>>(
                    index, c.ratio, c.blockSize, p->blockShift, p->listCap, topKWorkspace);
                V41TopKKernel<true, 2, 1024><<<rows, 1024, 0, cudaStreamPerThread>>>(
                    nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0,
                    p->listCap, nullptr, index, topKWorkspace);
            } else if (p->topKMemory) {
                V41TopKKernel<true, 1><<<dim3(topKWorkspace.parts, rows), kSelThreads,
                    (size_t)p->listCap * sizeof(int), cudaStreamPerThread>>>(
                    nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0,
                    p->listCap, nullptr, index, topKWorkspace);
                if (p->wideTopKMerge) {
                    V41TopKKernel<true, 2, 1024><<<rows, 1024, 0, cudaStreamPerThread>>>(
                        nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0,
                        p->listCap, nullptr, index, topKWorkspace);
                } else {
                    V41TopKKernel<true, 2><<<rows, kSelThreads, 0, cudaStreamPerThread>>>(
                        nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0,
                        p->listCap, nullptr, index, topKWorkspace);
                }
            } else {
                V41TopKKernel<true, 0, 1024><<<rows, 1024, (size_t)p->listCap * sizeof(int), cudaStreamPerThread>>>(
                    nullptr, nullptr, 1, 0, 0, c.blockSize, p->blockShift, 0, c.ratio, 0, p->listCap, nullptr, index, {});
            }
        });
    }
    const int *fallbackRows = nullptr;
#ifdef FASTLLM_V41_FP16_ATTENTION
    if (p->fp16Workspace) {
        const int keys = c.window + (c.ratio ? c.topK : 0);
        if (!v41_fp16_attention::Run(p->attention, p->fp16Workspace.get(), c.batch, c.heads,
                keys, c.window, c.softmaxScale, static_cast<const float *>(b.sink))) return false;
        fallbackRows = v41_fp16_attention::Bind(p->fp16Workspace.get(), c.batch, c.heads, keys).fallback;
    }
#endif
    auto scalarAttention = [&] {
        auto launch = [&](auto headTag, auto padTag, auto splitTag) {
            constexpr int heads = decltype(headTag)::value, splits = decltype(splitTag)::value;
            V41SparseAttentionKernel<__nv_bfloat16, heads, decltype(padTag)::value, splits, false, true><<<
                dim3(c.batch, (c.heads + heads - 1) / heads, splits), heads * kAttnLanes, 0, cudaStreamPerThread>>>(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                static_cast<const float *>(b.sink), 1, c.heads, 512, c.window, 0, 0, 0,
                c.softmaxScale, nullptr, 0, 0, 0, 0, p->attention, fallbackRows);
        };
        if (V41IsSm75() && c.batch == 1)
            launch(std::integral_constant<int, 8>{}, std::true_type{}, std::integral_constant<int, 8>{});
        else if (V41IsSm75())
            launch(std::integral_constant<int, kAttnHeadsPerBlock>{}, std::true_type{}, std::integral_constant<int, 1>{});
        else
            launch(std::integral_constant<int, kAttnHeadsPerBlock>{}, std::false_type{}, std::integral_constant<int, 1>{});
        return V41CheckLaunch("FullDecodeScalarAttention");
    };
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    if (p->mla && !p->mla->outputOnly)
        return FastllmV41RunMla(*p->mla, c, b, p->requests, cudaStreamPerThread, !p->directMlaOutput,
                p->earlyHistory || (allIndices && p->shortPages && !c.kvSource), allIndices && p->shortPages) == cudaSuccess;
#endif
#if defined(FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE) && defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103)
    if (p->flashInfer && c.verifying && c.ropeDim == 64 && V41IsSm<10,3>()) {
        if (FastllmV41FlashInferVerifyAngles(p->flashInfer)) {
            bool packed = false;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
            packed = p->directMlaOutput;
#endif
            return FastllmV41FlashInferVerifyFused(p->flashInfer, b, p->requests, cudaStreamPerThread, packed) == cudaSuccess;
        }
        FastllmV41FlashInferPartials history{};
        if (FastllmV41FlashInferRun(p->flashInfer, b, p->requests, cudaStreamPerThread, &history) != cudaSuccess)
            return false;
        V41FullDecodeRowTiles(c.batch, p->gridY, [&](int begin, int rows) {
            V41VerifyMergeRotaryOutput<<<dim3(c.heads, rows), 128, 0, cudaStreamPerThread>>>(
                b, history, p->requests, p->rope, c.softmaxScale, c.heads, c.batch, begin);
        });
        return V41CheckLaunch("FullVerifyMergeRotaryOutput");
    }
#endif
    const bool attentionOk =
#ifdef FASTLLM_ENABLE_DSV41_FLASHINFER_DECODE
        p->flashInfer ? FastllmV41FlashInferRun(p->flashInfer, b, p->requests, cudaStreamPerThread) == cudaSuccess :
#endif
        p->scalarAttention ? scalarAttention() :
        p->qkPrefetch ? V41FullDecodeAttention<8, true, true, true>(*p) :
        p->pvPipeline ? V41FullDecodeAttention<8, true, true>(*p) :
        p->splits == 8 ? V41FullDecodeAttention<8, true>(*p) :
        p->splits == 2 ? V41FullDecodeAttention<2, true>(*p) :
        p->fused ? V41FullDecodeAttention<1, false>(*p) : V41FullDecodeAttention<1, true>(*p);
    if (!attentionOk) return false;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
    if (p->mla && p->mla->outputOnly && p->directMlaOutput) {
        V41FullDecodeRowTiles(c.batch, p->gridY, [&](int begin, int rows) {
            V41PackExactAttention<<<dim3(c.heads, rows),512,0,cudaStreamPerThread>>>(
                p->transforms, p->rope, static_cast<uint8_t *>(p->mla->codes),
                reinterpret_cast<uint8_t *>(p->mla->scales), p->mla->scalePitch, begin, p->mla->headMajor);
        });
        return V41CheckLaunch("FullDecodePackExactAttention");
    }
#endif
    // Independent descriptors share one launch; each row retains its original math.
    if (c.verifying || c.ringMode) {
        V41FullDecodeTransforms(*p, 0, c.heads);
    } else {
        V41FullDecodeTransforms<2>(*p, 0, c.heads);
    }
    if (!c.verifying && c.ringMode && !p->directRing)
        V41FullDecodeStorePackedRing<<<c.batch, 256, 0, cudaStreamPerThread>>>(c, b, p->requests);
    return V41CheckLaunch("FullDecodeFinish");
}


#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
// Owners write disjoint (global row, feature slot) pairs on the draft device.
// The target's existing all-rank completion events publish the buffer to sampling.
static __global__ void V41PublishDraftFeature(const __nv_bfloat16 *hidden, __nv_bfloat16 *packed,
        const int *globalRows, int dim, int slot, int features) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    const int row = blockIdx.y;
    float value = 0.0f;
#pragma unroll
    for (int hc = 0; hc < 4; ++hc)
        value = __fadd_rn(value, __fmul_rn(0.25f, __bfloat162float(hidden[(size_t(row) * 4 + hc) * dim + col])));
    packed[(size_t(globalRows[row]) * features + slot) * dim + col] = __float2bfloat16_rn(value);
}

bool FastllmCudaV41PublishDraftFeature(const void *hidden, void *packed, const int *globalRows,
                                      int rows, int dim, int slot, int features) {
    if (!hidden || !packed || !globalRows || rows < 1 || rows > 32 || dim <= 0 || slot < 0 || slot >= features)
        return false;
    V41PublishDraftFeature<<<dim3((dim + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(hidden), static_cast<__nv_bfloat16 *>(packed),
        globalRows, dim, slot, features);
    return V41CheckLaunch("PublishDraftFeature");
}
#endif

bool FastllmCudaV41FinishFullDecode(void *handle) { return V41FinishDecode(handle, false); }
bool FastllmCudaV41FinishSource(void *handle, bool allIndices) {
    return FastllmCudaV41CanPublishSource(handle) && V41FinishDecode(handle, allIndices, true);
}
bool FastllmCudaV41FinishSource(void *handle) { return FastllmCudaV41FinishSource(handle, true); }
void *FastllmCudaV41ForkSourceStore(const void *source) {
#if defined(FASTLLM_ENABLE_DSV41_FLASHMLA) && CUDART_VERSION >= 12000
    cudaGraph_t g = nullptr;
    if (cudaGraphClone(&g, (cudaGraph_t)source) != cudaSuccess) return nullptr;
    auto reject = [&]() -> void * { cudaGetLastError(); cudaGraphDestroy(g); return nullptr; };
    size_t n = 0;
    if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaGraphGetNodes(g, nodes.data(), &n) != cudaSuccess) return reject();
    cudaGraphNode_t main = nullptr, index = nullptr, pages = nullptr;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        auto rc = cudaGraphKernelNodeGetParams(node, &p);
        if (rc == cudaErrorInvalidDeviceFunction) { cudaGetLastError(); continue; }
        if (rc != cudaSuccess) return reject();
        if (p.func == (void *)V41PublishCompressed || p.func == (void *)V41FullVerifyGather<true, true>) main = node;
        if (p.func == (void *)V41PublishIndexK) index = node;
        if (FastllmV41IsMlaPageKernel(p.func) || p.func == (void *)V41VerifyAllIndices) pages = node;
    }
    cudaGraphNode_t parent = nullptr, flash = nullptr;
    if (!main || !index || !pages ||
        FastllmCudaGraphNodeGetDependencies(pages, nullptr, &n) != cudaSuccess || n != 1 ||
        FastllmCudaGraphNodeGetDependencies(pages, &parent, &n) != cudaSuccess || parent != index ||
        FastllmCudaGraphNodeGetDependentNodes(pages, nullptr, &n) != cudaSuccess || n != 1 ||
        FastllmCudaGraphNodeGetDependentNodes(pages, &flash, &n) != cudaSuccess ||
        FastllmCudaGraphNodeGetDependentNodes(flash, nullptr, &n) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> consumers(n);
    if (n && FastllmCudaGraphNodeGetDependentNodes(flash, consumers.data(), &n) != cudaSuccess) return reject();
    // No following projection may reuse the indexer's scratch until it has retired.
    for (auto node : consumers)
        if (FastllmCudaGraphAddDependencies(g, &index, &node, 1) != cudaSuccess) return reject();
    if (FastllmCudaGraphAddDependencies(g, &main, &pages, 1) != cudaSuccess) return reject();
#if CUDART_VERSION >= 13000
    if (cudaGraphRemoveDependencies(g, &index, &pages, nullptr, 1) != cudaSuccess) return reject();
#else
    if (cudaGraphRemoveDependencies(g, &index, &pages, 1) != cudaSuccess) return reject();
#endif
    return g;
#else
    return nullptr;
#endif
}
// The recorded compressor frontier is a node of the original prelude. Its
// clone still produces rawKV and rawScore after the independent Q branch is cut.
void *FastllmCudaV41MoveSourceToPrelude(void *prefix, const void *compressorReady, const void *core) {
#if defined(FASTLLM_ENABLE_DSV41_FLASHMLA) && CUDART_VERSION >= 12000
    cudaGraphNode_t ready = nullptr;
    if (!prefix || !compressorReady || !core || cudaGraphNodeFindInClone(&ready,
            (cudaGraphNode_t)compressorReady, (cudaGraph_t)prefix) != cudaSuccess) {
        cudaGetLastError(); return nullptr;
    }
    cudaGraph_t g = nullptr;
    if (cudaGraphClone(&g, (cudaGraph_t)core) != cudaSuccess) return nullptr;
    auto reject = [&]() -> void * { cudaGetLastError(); cudaGraphDestroy(g); return nullptr; };
    size_t n = 0;
    if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaGraphGetNodes(g, nodes.data(), &n) != cudaSuccess) return reject();
    cudaGraphNode_t compression = nullptr;
    cudaKernelNodeParams params{};
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        auto rc = cudaGraphKernelNodeGetParams(node, &p);
        if (rc == cudaErrorInvalidDeviceFunction) { cudaGetLastError(); continue; }
        if (rc != cudaSuccess) return reject();
        if (p.func == (void *)V41PublishCompressed || p.func == (void *)V41FullVerifyGather<true, true>) {
            if (compression) return reject();
            compression = node; params = p;
        }
    }
    cudaGraphNode_t parent = nullptr, early = nullptr;
    if (!compression || FastllmCudaGraphNodeGetDependencies(compression, nullptr, &n) != cudaSuccess || n != 1 ||
        FastllmCudaGraphNodeGetDependencies(compression, &parent, &n) != cudaSuccess ||
        FastllmCudaGraphNodeGetDependentNodes(compression, nullptr, &n) != cudaSuccess || !n) return reject();
    std::vector<cudaGraphNode_t> consumers(n);
    if (FastllmCudaGraphNodeGetDependentNodes(compression, consumers.data(), &n) != cudaSuccess) return reject();
    for (auto node : consumers)
        if (FastllmCudaGraphAddDependencies(g, &parent, &node, 1) != cudaSuccess) return reject();
    // Only mutate the prefix after the cloned attention is usable. Its child
    // graph joins every prelude leaf before the first Wk or page consumer.
    if (cudaGraphAddKernelNode(&early, (cudaGraph_t)prefix, &ready, 1, &params) != cudaSuccess) return reject();
    if (cudaGraphDestroyNode(compression) != cudaSuccess) {
        cudaGraphDestroyNode(early);
        return reject();
    }
    return g;
#else
    return nullptr;
#endif
}
bool FastllmCudaV41FinishAllIndices(void *handle) { return V41FinishDecode(handle, true); }

bool FastllmCudaV41CreateHostCopy(void *dst, size_t bytes, void **hostData,
        std::atomic<unsigned> **ready, void **result) {
#if !defined(USE_ROCM) && CUDART_VERSION >= 11070
    if (!dst || !bytes || bytes > SIZE_MAX - 64 || !hostData || !ready || !result) return false;
    *hostData = nullptr; *ready = nullptr; *result = nullptr;
    decltype(&cuCtxGetCurrent) context = nullptr;
    decltype(&cuGraphAddBatchMemOpNode) add = nullptr;
    if (!fastllm::cuda::DriverEntry("cuCtxGetCurrent", 4000, context) ||
        !fastllm::cuda::DriverEntry("cuGraphAddBatchMemOpNode", 11070, add)) {
        cudaGetLastError();
        return false;
    }
    void *host = nullptr, *mapped = nullptr;
    cudaGraph_t graph = nullptr;
    CUcontext ctx = nullptr;
    auto fail = [&] {
        if (graph) cudaGraphDestroy(graph);
        if (host) cudaFreeHost(host);
        cudaGetLastError(); return false;
    };
    if (cudaHostAlloc(&host, bytes + 64, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess ||
        cudaGraphCreate(&graph, 0) != cudaSuccess || context(&ctx) != CUDA_SUCCESS) return fail();
    CUstreamBatchMemOpParams op{};
    op.waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.address = reinterpret_cast<CUdeviceptr>(mapped);
    op.waitValue.value = 1; op.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params{ctx, 1, &op, 0};
    CUgraphNode wait = nullptr;
    if (add(&wait, reinterpret_cast<CUgraph>(graph), nullptr, 0, &params) != CUDA_SUCCESS) return fail();
    auto dependency = reinterpret_cast<cudaGraphNode_t>(wait);
    cudaGraphNode_t copy = nullptr;
    auto *data = static_cast<char *>(host) + 64;
    if (cudaGraphAddMemcpyNode1D(&copy, graph, &dependency, 1, dst, data, bytes,
            cudaMemcpyHostToDevice) != cudaSuccess) return fail();
    static_assert(sizeof(std::atomic<unsigned>) == sizeof(unsigned) &&
                  std::atomic<unsigned>::is_always_lock_free, "mapped flag must be a lock-free word");
    new (host) std::atomic<unsigned>(1u);
    *hostData = data; *ready = static_cast<std::atomic<unsigned> *>(host); *result = graph;
    return true;
#else
    return false;
#endif
}

#ifndef USE_ROCM
#include "deepseekv41-hc-graph.cuh"
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
#include "deepseekv41-owner-mix.cuh"
#endif
#endif
// Only remove work captured inside the index projection itself. Any consumer
// outside that set makes this short graph ineligible for the optimization.
void *FastllmCudaV41WithoutIndexProjections(const void *source, const std::vector<void *> &suffix) {
#if defined(FASTLLM_ENABLE_MXFP8_LINEAR_SM103) && !defined(USE_ROCM) && CUDART_VERSION >= 12000
    cudaGraph_t graph = nullptr;
    if (!source || suffix.empty() || cudaGraphClone(&graph, (cudaGraph_t)source) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    auto reject = [&]() -> void * { cudaGetLastError(); cudaGraphDestroy(graph); return nullptr; };
    std::vector<cudaGraphNode_t> nodes;
    for (auto original : suffix) {
        cudaGraphNode_t node;
        if (cudaGraphNodeFindInClone(&node, (cudaGraphNode_t)original, graph) != cudaSuccess) return reject();
        nodes.push_back(node);
    }
    // cuBLAS may use non-default programmatic edges, even for a single consumer.
    for (auto node : nodes) {
        size_t count = 0;
        if (fastllm_v41_hc_graph::OwnerNeighbors<false>(node, nullptr, &count) != cudaSuccess) return reject();
        std::vector<cudaGraphNode_t> users(count);
        if (count && fastllm_v41_hc_graph::OwnerNeighbors<false>(node, users.data(), &count) != cudaSuccess) return reject();
        for (auto user : users)
            if (std::find(nodes.begin(), nodes.end(), user) == nodes.end()) return reject();
    }
    for (auto node : nodes) if (cudaGraphDestroyNode(node) != cudaSuccess) return reject();
    return graph;
#else
    return nullptr;
#endif
}
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
namespace {
struct BoundaryNorm {
    const void *input = nullptr, *pre = nullptr, *weight = nullptr;
    void *out = nullptr, *rounded = nullptr, *codes = nullptr, *scales = nullptr;
    float eps = 0;
    int pitch = 0;
    bool pack = false, quantize = false;
    bool Matches(const BoundaryNorm &b) const {
        return input == b.input && pre == b.pre && weight == b.weight && out == b.out &&
            rounded == b.rounded && codes == b.codes && scales == b.scales && eps == b.eps &&
            pitch == b.pitch && pack == b.pack && quantize == b.quantize;
    }
};

bool TakeBoundaryNorm(cudaGraph_t graph, int rows, BoundaryNorm &result) {
    size_t count = 0;
    if (cudaGraphGetRootNodes(graph, nullptr, &count) != cudaSuccess || !count) return false;
    std::vector<cudaGraphNode_t> roots(count);
    if (cudaGraphGetRootNodes(graph, roots.data(), &count) != cudaSuccess) return false;
    cudaGraphNode_t found = nullptr;
    for (auto root : roots) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(root, &type) != cudaSuccess) return false;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(root, &p) != cudaSuccess) { cudaGetLastError(); continue; }
        bool pack = p.func == (void *)V41HcPreNormPacked<true> || p.func == (void *)V41HcPreNormPacked<false>;
        bool quant = p.func == (void *)V41HcPreNormPacked<true> || p.func == (void *)V41HcPreNormKernel<1024,3,false,true>;
        if (!(pack || quant || p.func == (void *)V41HcPreNormKernel<1024,3,false,false>)) continue;
        void **a = p.kernelParams;
        if (found || !a || p.extra || p.gridDim.x != rows || p.gridDim.y != 1 ||
            *static_cast<int *>(a[4]) != 4 || *static_cast<int *>(a[5]) != 5120) return false;
        auto ptr = [&](int i) {return *static_cast<void **>(a[i]);};
        result = {ptr(0), ptr(1), ptr(2), ptr(3), quant ? ptr(15) : nullptr,
                  pack ? ptr(16) : nullptr, pack ? ptr(17) : nullptr, *static_cast<float *>(a[6]),
                  pack ? *static_cast<int *>(a[18]) : 0, pack, quant};
        found = root;
    }
    return found && cudaGraphDestroyNode(found) == cudaSuccess;
}
}
template <int Blocks, bool PackedOrder = false>
static void *V41BoundaryKernel(bool pack, bool quantize, bool packOnly) {
    using namespace fastllm_v41_owner_mix;
    if (pack && packOnly) return (void *)ExpertResidualNorm<true,false,false,Blocks,PackedOrder>;
    if (pack) return quantize ? (void *)ExpertResidualNorm<true,true,true,Blocks,PackedOrder> : (void *)ExpertResidualNorm<true,false,true,Blocks,PackedOrder>;
    return quantize ? (void *)ExpertResidualNorm<false,true,true,Blocks,PackedOrder> : (void *)ExpertResidualNorm<false,false,true,Blocks,PackedOrder>;
}
// Rank-local verification graphs contain nested per-layer capture segments.
// Keep the exact device dependencies when emitting a single-level graph.
#if !defined(USE_ROCM) && CUDA_VERSION >= 13000
namespace {
struct V41GraphBoundary { int input, output; };

struct V41GraphFlattener {
    struct Node {
        CUgraphNode source = nullptr, copy = nullptr;
        std::vector<std::pair<int, CUgraphEdgeData>> incoming;
    };
    std::vector<Node> nodes;
    CUcontext context = nullptr;
    CUgraph graph = nullptr;
    ~V41GraphFlattener() { if (graph) cuGraphDestroy(graph); }

    int Append(CUgraphNode source = nullptr) {
        nodes.emplace_back();
        nodes.back().source = source;
        return int(nodes.size()) - 1;
    }
    bool Expand(CUgraph source, V41GraphBoundary &bounds) {
        bounds = {Append(), Append()};
        size_t count = 0, edgeCount = 0;
        if (cuGraphGetNodes(source, nullptr, &count) != CUDA_SUCCESS ||
            cuGraphGetEdges(source, nullptr, nullptr, nullptr, &edgeCount) != CUDA_SUCCESS) return false;
        std::vector<CUgraphNode> originals(count), from(edgeCount), to(edgeCount);
        std::vector<CUgraphEdgeData> edges(edgeCount);
        if ((count && cuGraphGetNodes(source, originals.data(), &count) != CUDA_SUCCESS) ||
            (edgeCount && cuGraphGetEdges(source, from.data(), to.data(), edges.data(), &edgeCount) != CUDA_SUCCESS))
            return false;
        std::map<CUgraphNode,V41GraphBoundary> spans;
        std::set<CUgraphNode> outgoing, incoming;
        for (auto original : originals) {
            CUgraphNodeType type;
            if (cuGraphNodeGetType(original, &type) != CUDA_SUCCESS) return false;
            if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
                CUgraph child;
                if (cuGraphChildGraphNodeGetGraph(original, &child) != CUDA_SUCCESS ||
                    !Expand(child, spans[original])) return false;
            } else {
                int i = Append(type == CU_GRAPH_NODE_TYPE_EMPTY ? nullptr : original);
                spans[original] = {i, i};
            }
        }
        for (size_t i = 0; i < edgeCount; ++i) {
            nodes[spans.at(to[i]).input].incoming.emplace_back(spans.at(from[i]).output, edges[i]);
            outgoing.insert(from[i]); incoming.insert(to[i]);
        }
        for (auto original : originals) {
            auto span = spans.at(original);
            if (!incoming.count(original)) nodes[span.input].incoming.push_back({bounds.input, {}});
            if (!outgoing.count(original)) nodes[bounds.output].incoming.push_back({span.output, {}});
        }
        if (originals.empty()) nodes[bounds.output].incoming.push_back({bounds.input, {}});
        return true;
    }
    bool Copy(Node &node) {
        CUgraphNode n = node.source;
        auto *out = &node.copy;
        CUgraphNodeType type;
        if (cuGraphNodeGetType(n, &type) != CUDA_SUCCESS) return false;
        switch (type) {
            case CU_GRAPH_NODE_TYPE_KERNEL: {
                CUDA_KERNEL_NODE_PARAMS p{};
                return cuGraphKernelNodeGetParams(n, &p) == CUDA_SUCCESS &&
                    cuGraphAddKernelNode(out, graph, nullptr, 0, &p) == CUDA_SUCCESS &&
                    cuGraphKernelNodeCopyAttributes(*out, n) == CUDA_SUCCESS;
            }
            case CU_GRAPH_NODE_TYPE_MEMCPY: {
                CUDA_MEMCPY3D p{};
                return cuGraphMemcpyNodeGetParams(n, &p) == CUDA_SUCCESS &&
                    cuGraphAddMemcpyNode(out, graph, nullptr, 0, &p, context) == CUDA_SUCCESS;
            }
            case CU_GRAPH_NODE_TYPE_MEMSET: {
                CUDA_MEMSET_NODE_PARAMS p{};
                return cuGraphMemsetNodeGetParams(n, &p) == CUDA_SUCCESS &&
                    cuGraphAddMemsetNode(out, graph, nullptr, 0, &p, context) == CUDA_SUCCESS;
            }
            case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            case CU_GRAPH_NODE_TYPE_WAIT_EVENT: {
                CUevent event;
                if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD)
                    return cuGraphEventRecordNodeGetEvent(n, &event) == CUDA_SUCCESS &&
                        cuGraphAddEventRecordNode(out, graph, nullptr, 0, event) == CUDA_SUCCESS;
                return cuGraphEventWaitNodeGetEvent(n, &event) == CUDA_SUCCESS &&
                    cuGraphAddEventWaitNode(out, graph, nullptr, 0, event) == CUDA_SUCCESS;
            }
            default: return false;
        }
    }
    // Ports on structural boundaries cannot be contracted unambiguously.
    bool Resolve(int i, CUgraphEdgeData edge, std::map<int, CUgraphEdgeData> &parents) const {
        if (nodes[i].source) {
            auto result = parents.emplace(i, edge);
            return result.second || std::memcmp(&result.first->second, &edge, sizeof(edge)) == 0;
        }
        if (edge.type || edge.from_port || edge.to_port) return false;
        for (auto &p : nodes[i].incoming) if (!Resolve(p.first, p.second, parents)) return false;
        return true;
    }
    CUgraphExec Build(CUgraph original) {
        V41GraphBoundary top{};
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context || !Expand(original, top) ||
            cuGraphCreate(&graph, 0) != CUDA_SUCCESS) return nullptr;
        for (auto &node : nodes) if (node.source && !Copy(node)) return nullptr;
        std::vector<CUgraphNode> from, to;
        std::vector<CUgraphEdgeData> edges;
        for (const auto &node : nodes) if (node.source) {
            std::map<int, CUgraphEdgeData> parents;
            for (auto &p : node.incoming) if (!Resolve(p.first, p.second, parents)) return nullptr;
            for (auto &p : parents) {
                from.push_back(nodes[p.first].copy); to.push_back(node.copy); edges.push_back(p.second);
            }
        }
        CUgraphExec exec = nullptr;
        if (cuGraphAddDependencies(graph, from.data(), to.data(), edges.data(), edges.size()) != CUDA_SUCCESS ||
            cuGraphInstantiateWithFlags(&exec, graph, 0) != CUDA_SUCCESS) return nullptr;
        return exec;
    }
};
}
#endif

void *FastllmCudaV41FlatVerifyExec(void *source) {
#if !defined(USE_ROCM) && CUDA_VERSION >= 13000
    V41GraphFlattener builder;
    return source ? builder.Build(static_cast<CUgraph>(source)) : nullptr;
#else
    return nullptr;
#endif
}

bool FastllmCudaV41ComposeEngramForks(const std::vector<const void *> &parts,
        const std::vector<std::pair<int, int>> &forks, void **outGraph, void **outExec) {
    using namespace fastllm_v41_hc_graph;
    if (!V41IsSm<10, 3>() || forks.empty()) return false;
    cudaGraph_t graph = nullptr;
    auto reject = [] { cudaGetLastError(); return false; };
    if (cudaGraphCreate(&graph, 0) != cudaSuccess) return reject();
    OwnerGraph result(graph, &cudaGraphDestroy);
    std::vector<cudaGraphNode_t> nodes(parts.size());
    auto fork = forks.begin();
    for (int i = 0; i < int(parts.size()); ++i) {
        auto child = reinterpret_cast<cudaGraph_t>(const_cast<void *>(parts[i]));
        if (fork == forks.end() || fork->first != i) {
            if (cudaGraphAddChildGraphNode(&nodes[i], graph, i ? &nodes[i-1] : nullptr, i ? 1 : 0, child) != cudaSuccess)
                return reject();
            continue;
        }
        // Only the gate updates hidden. Keep every other dependency inside the captured preparation.
        if (fork->second < 0 || fork->second >= i-1) return false;
        size_t count = 0;
        if (cudaGraphGetNodes(child, nullptr, &count) != cudaSuccess || count < 2) return reject();
        std::vector<cudaGraphNode_t> sequence(count);
        if (cudaGraphGetNodes(child, sequence.data(), &count) != cudaSuccess) return reject();
        cudaGraphNode_t gate = nullptr;
        cudaKernelNodeParams gateParams{};
        for (auto node : sequence) {
            size_t users = 0;
            cudaGraphNodeType type;
            if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel ||
                OwnerNeighbors<false>(node, nullptr, &users) != cudaSuccess) return reject();
            if (users) continue;
            if (gate || cudaGraphKernelNodeGetParams(node, &gateParams) != cudaSuccess ||
                gateParams.func != (void *)V41EngramApplyKernel<__nv_bfloat16, __nv_bfloat16>) return reject();
            gate = node;
        }
        cudaGraph_t raw = nullptr;
        if (!gate || cudaGraphClone(&raw, child) != cudaSuccess) return reject();
        OwnerGraph prep(raw, &cudaGraphDestroy);
        cudaGraphNode_t tail = nullptr, branch = nullptr;
        if (cudaGraphNodeFindInClone(&tail, gate, raw) != cudaSuccess || cudaGraphDestroyNode(tail) != cudaSuccess ||
            cudaGraphAddChildGraphNode(&branch, graph, &nodes[fork->second], 1, raw) != cudaSuccess) return reject();
        cudaGraphNode_t inputs[] = {nodes[i-1], branch};
        if (cudaGraphAddKernelNode(&nodes[i], graph, inputs, 2, &gateParams) != cudaSuccess ||
            cudaGraphKernelNodeCopyAttributes(nodes[i], gate) != cudaSuccess) return reject();
        ++fork;
    }
    cudaGraphExec_t exec = nullptr;
    if (fork != forks.end() || cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) return reject();
    *outExec = exec;
    *outGraph = result.release();
    return true;
}

void *FastllmCudaV41ForkIndexProjections(const void *const *children, int count, const void *original,
        const std::vector<void *> &projectionNodes, const void *plan, void *rawWeights,
        float *mixScratch, void **ownerGemm) {
    using namespace fastllm_v41_hc_graph;
    const auto *p = static_cast<const V41FullDecodePlan *>(plan);
    if (!p || !p->packVerify || p->config.batch != 6 || p->config.indexHeads != 32 ||
        !p->config.indexSource || !rawWeights || count < 2 || projectionNodes.size() != 4) return nullptr;
    auto clone = [](const void *source) {
        cudaGraph_t g = nullptr;
        cudaGraphClone(&g, (cudaGraph_t)source);
        return OwnerGraph(g, &cudaGraphDestroy);
    };
    auto pre = clone(children[0]), core = clone(children[1]);
    auto reject = []() -> void * { cudaGetLastError(); return nullptr; };
    if (!pre || !core) return reject();
    cudaGraphNode_t head = nullptr, cast = nullptr, mul = nullptr, query = nullptr;
    cudaKernelNodeParams gemv{}, convert{}, multiply{};
    void *castFn = nullptr, *mulFn = nullptr;
    FastllmCudaV41IndexWeightKernels(castFn, mulFn);
    for (auto recorded : projectionNodes) {
        cudaGraphNode_t source = (cudaGraphNode_t)recorded, node = nullptr;
        if (original != children[0] &&
            cudaGraphNodeFindInClone(&source, (cudaGraphNode_t)recorded, (cudaGraph_t)children[0]) != cudaSuccess)
            return reject();
        if (cudaGraphNodeFindInClone(&node, source, pre.get()) != cudaSuccess) return reject();
        cudaKernelNodeParams k{};
        if (cudaGraphKernelNodeGetParams(node, &k) != cudaSuccess || !k.kernelParams || k.extra) return reject();
        if (k.func == FastllmCudaV41IndexHeadGemvKernel()) { head = node; gemv = k; }
        else if (k.func == castFn) { cast = node; convert = k; }
        else if (k.func == mulFn) { mul = node; multiply = k; }
        else query = node;
    }
    auto singleUser = [](cudaGraphNode_t node, cudaGraphNode_t expected) {
        size_t n = 0;
        cudaGraphNode_t user = nullptr;
        return OwnerNeighbors<false>(node, nullptr, &n) == cudaSuccess && n == (expected ? 1 : 0) &&
            (!n || (OwnerNeighbors<false>(node, &user, &n) == cudaSuccess && user == expected));
    };
    auto address = [](const cudaKernelNodeParams &k, int i) { return *static_cast<void **>(k.kernelParams[i]); };
    if (!head || !cast || !mul || !query || !singleUser(query, head) || !singleUser(head, cast) ||
        !singleUser(cast, mul) || !singleUser(mul, nullptr) || gemv.gridDim.x != 32 ||
        gemv.gridDim.y != 1 || *static_cast<int *>(gemv.kernelParams[5]) != 32 ||
        *static_cast<int *>(convert.kernelParams[2]) != 6 * 32 || *static_cast<int *>(multiply.kernelParams[3]) != 6 * 32 ||
        address(gemv, 2) != address(convert, 0) || address(convert, 1) != address(multiply, 0) ||
        address(multiply, 1) != p->buffers.indexWeights) return reject();
    // Each projection reads a different activation. The new head output is
    // graph-owned since the captured BF16 allocation expires at the old cast.
    size_t n = 0;
    for (auto node : {head, cast, mul})
        if (OwnerNeighbors<true>(node, nullptr, &n) != cudaSuccess || n != 1) return reject();
    if (OwnerNeighbors<true>(query, nullptr, &n) != cudaSuccess || !n) return reject();
    std::vector<cudaGraphNode_t> parents(n);
    if (OwnerNeighbors<true>(query, parents.data(), &n) != cudaSuccess) return reject();
    void *projectionArgs[6];
    std::copy_n(gemv.kernelParams, 6, projectionArgs);
    projectionArgs[2] = &rawWeights;
    gemv.kernelParams = projectionArgs;
    cudaGraphNode_t parallelHead = nullptr;
    if (cudaGraphAddKernelNode(&parallelHead, pre.get(), parents.data(), n, &gemv) != cudaSuccess) return reject();
    const float scale = *static_cast<float *>(multiply.kernelParams[2]);
    for (auto node : {mul, cast, head}) if (cudaGraphDestroyNode(node) != cudaSuccess) return reject();

    if (cudaGraphGetNodes(core.get(), nullptr, &n) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaGraphGetNodes(core.get(), nodes.data(), &n) != cudaSuccess) return reject();
    bool fused = false;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams k{};
        auto rc = cudaGraphKernelNodeGetParams(node, &k);
        if (rc == cudaErrorInvalidDeviceFunction) { cudaGetLastError(); continue; }
        if (rc != cudaSuccess) return reject();
        if (k.func != (void *)V41DecodeTransformsKernel<kFullDecodeTransforms, 1> || k.extra ||
            !k.kernelParams || address(k, 0) != p->transforms + 4) continue;
        if (fused || k.gridDim.x != 8 || k.gridDim.y != 6 || k.blockDim.x != 512) return reject();
        auto *out = static_cast<float *>(p->buffers.indexWeights);
        float factor = scale;
        void *args[] = {k.kernelParams[0], k.kernelParams[1], k.kernelParams[2], k.kernelParams[3],
                        &rawWeights, &out, &factor};
        k.func = (void *)V41VerifyIndexQWeights;
        k.kernelParams = args;
        if (cudaGraphKernelNodeSetParams(node, &k) != cudaSuccess) return reject();
        fused = true;
    }
    if (!fused) return reject();
    std::vector<const void *> scored(children, children + count);
    scored[0] = pre.get();
    scored[1] = core.get();
    return ForkOwnerMix(scored.data(), count, 6, mixScratch, nullptr, nullptr, ownerGemm, true);
}

bool FastllmCudaV41FuseExpertBoundary(const void *previous, const void *attention,
        const void *shortAttention, const void *postKernel, int rows, bool packOnly, void **result) {
    using namespace fastllm_v41_owner_mix;
    if (!V41IsSm<10,3>() || rows < 1 || rows > 32 || !previous || !attention || !postKernel) return false;
    auto clone = [](const void *source) {
        cudaGraph_t graph = nullptr;
        if (source) cudaGraphClone(&graph, (cudaGraph_t)source);
        return fastllm_v41_hc_graph::OwnerGraph(graph, &cudaGraphDestroy);
    };
    auto producer = clone(previous), normal = clone(attention), shortGraph = clone(shortAttention);
    BoundaryNorm a{}, b{};
    if (!producer || !normal || !TakeBoundaryNorm(normal.get(), rows, a) ||
        (shortAttention && (!shortGraph || !TakeBoundaryNorm(shortGraph.get(), rows, b) || !a.Matches(b)))) return false;
    size_t count = 0;
    if (cudaGraphGetNodes(producer.get(), nullptr, &count) != cudaSuccess || !count) return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(producer.get(), nodes.data(), &count) != cudaSuccess) return false;
    cudaGraphNode_t combine = nullptr;
    cudaKernelNodeParams params{};
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return false;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess) { cudaGetLastError(); continue; }
        if (p.func != postKernel) continue;
        if (combine) return false;
        combine = node; params = p;
    }
    auto args = params.kernelParams;
    if (!combine || !args || params.extra || *static_cast<void **>(args[2]) != a.input ||
        *static_cast<int *>(args[3]) != rows || *static_cast<int *>(args[4]) != 0 ||
        fastllm_v41_hc_graph::OwnerNeighbors<false>(combine, nullptr, &count) != cudaSuccess || count > 1) return false;
    VerifyDraftFeature feature{};
    cudaGraphNode_t publish = nullptr;
    if (count == 1) {
        if (rows != 6 || fastllm_v41_hc_graph::OwnerNeighbors<false>(combine, &publish, &count) != cudaSuccess)
            return false;
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(publish, &p) != cudaSuccess || p.func != (void *)V41PublishDraftFeature ||
            p.extra || !p.kernelParams || fastllm_v41_hc_graph::OwnerNeighbors<true>(publish, nullptr, &count) != cudaSuccess ||
            count != 1 || fastllm_v41_hc_graph::OwnerNeighbors<false>(publish, nullptr, &count) != cudaSuccess || count)
            return false;
        auto v = p.kernelParams;
        if (*static_cast<void **>(v[0]) != a.input || *static_cast<int *>(v[3]) != 5120) return false;
        feature = {*static_cast<Pair **>(v[1]), *static_cast<const int **>(v[2]),
                   *static_cast<int *>(v[4]), *static_cast<int *>(v[5])};
        if (!feature.packed || !feature.globalRows || feature.slot < 0 || feature.slot >= feature.features) return false;
    }
    ExpertBoundary ep{*static_cast<const Pair **>(args[0]), *static_cast<const int **>(args[1]),
                      static_cast<Pair *>(a.rounded), static_cast<uint8_t *>(a.codes), static_cast<uint8_t *>(a.scales), a.pitch};
    void *fused[] = {args[5],args[6],args[7],&a.pre,&a.weight,&a.eps,args[2],&a.out,&ep,&feature};
    const bool wide = V41EnvOn("FASTLLM_DSV41_WIDE_CLUSTER");
    const int *order = *static_cast<const int **>(args[8]);
    void *vectorKernel = publish ? (void *)VerifyExpertFeatureNorm : (void *)VerifyExpertNorm;
    const bool vectorized = rows == 6 && order && !wide && !a.pack && !a.quantize &&
        VerifyPostAligned({ep.slots, *static_cast<void **>(args[5]), a.input, a.out, a.weight}) &&
        VerifyPostCluster(vectorKernel);
    if (publish && (!vectorized || cudaGraphDestroyNode(publish) != cudaSuccess)) return false;
    const unsigned blocks = (wide || vectorized) ? 8 : 4;
    if (vectorized) {
        ep.ids = order;
        params.func = vectorKernel;
    } else if (order && !wide) {
        ep.ids = order;
        params.func = V41BoundaryKernel<4,true>(a.pack, a.quantize, packOnly);
    } else {
        params.func = wide ? V41BoundaryKernel<8>(a.pack, a.quantize, packOnly) : V41BoundaryKernel<4>(a.pack, a.quantize, packOnly);
    }
    params.kernelParams = fused; params.gridDim = dim3(blocks, rows); params.blockDim = dim3(vectorized ? 192 : 1024 / blocks);
    cudaKernelNodeAttrValue attr{};attr.clusterDim.x = blocks;attr.clusterDim.y = 1;attr.clusterDim.z = 1;
    if (cudaGraphKernelNodeSetParams(combine, &params) != cudaSuccess ||
        cudaGraphKernelNodeSetAttribute(combine, cudaKernelNodeAttributeClusterDimension, &attr) != cudaSuccess) return false;
    result[0] = producer.release(); result[1] = normal.release(); result[2] = shortGraph.release();
    return true;
}
#endif
#ifndef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
void *FastllmCudaV41CreateOwnerStream(int) { return nullptr; }
void FastllmCudaV41DestroyOwnerStream(void *) {}
void *FastllmCudaV41OwnerClusterPacket(void *plan, int rows, float *scratch,
        const fastllm::Data &attention, const fastllm::Data &residual, const fastllm::Data &oldPre, const fastllm::Data &oldPost, const fastllm::Data &oldComb,
        fastllm::Data &fn, const fastllm::Data &scale, const fastllm::Data &base, const fastllm::Data &weight, int iterations, float mixEps,
        float normEps, fastllm::Data &hidden, fastllm::Data &pre, fastllm::Data &post, fastllm::Data &comb, fastllm::Data &normalized, FastllmDsv41InputPacket *packet) { return nullptr; }
#endif
void *FastllmCudaV41OwnerClusterPost(void *plan, int rows, float *scratch,
        const fastllm::Data &attention, const fastllm::Data &residual, const fastllm::Data &oldPre, const fastllm::Data &oldPost, const fastllm::Data &oldComb,
        fastllm::Data &fn, const fastllm::Data &scale, const fastllm::Data &base, const fastllm::Data &weight, int iterations, float mixEps,
        float normEps, fastllm::Data &hidden, fastllm::Data &pre, fastllm::Data &post, fastllm::Data &comb, fastllm::Data &normalized) {
    return FastllmCudaV41OwnerClusterPacket(plan, rows, scratch, attention, residual, oldPre, oldPost, oldComb,
        fn, scale, base, weight, iterations, mixEps, normEps, hidden, pre, post, comb, normalized, nullptr);
}
extern "C" void *FastllmCudaV41ForkOwnerMixQuery(const void *const *children, int count, int rows, float *scratch, const void *postKernel, const void *prepareKernel, void **ownerGemm, bool keepQuery) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    if (V41IsSm<10, 3>()) return fastllm_v41_hc_graph::ForkOwnerMix(children, count, rows, scratch, postKernel, prepareKernel, ownerGemm, keepQuery);
#endif
    return nullptr;
}
extern "C" void *FastllmCudaV41ForkOwnerMix(const void *const *children, int count, int rows, float *scratch, const void *postKernel, const void *prepareKernel, void **ownerGemm) {
    return FastllmCudaV41ForkOwnerMixQuery(children, count, rows, scratch, postKernel, prepareKernel, ownerGemm, true);
}
extern "C" void FastllmCudaV41DestroyOwnerMixGemm(void *plan) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    delete static_cast<fastllm_v41_hc_graph::OwnerMixGemm *>(plan);
#endif
}
extern "C" void *FastllmCudaV41CloneSingleAttentionMix(void *source, float *scratch) {
#ifndef USE_ROCM
    if (V41IsSm<10, 3>()) return fastllm_v41_hc_graph::CloneSingleAttentionMix(source, scratch);
#endif
    return nullptr;
}
extern "C" bool FastllmCudaV41ComposeHcGraph(const void *const *children, int count,
        const int *start, const int *join, int branches, float *saved,
        void **outGraph, void **outExec) {
#ifndef USE_ROCM
    return fastllm_v41_hc_graph::Compose(children, count, start, join, branches, saved, outGraph, outExec);
#else
    return false;
#endif
}
