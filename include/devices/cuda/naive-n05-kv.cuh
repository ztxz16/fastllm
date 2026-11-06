#pragma once
#include "fastllm.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "devices/cuda/attention/fastllm-fp4-kv.cuh"

// Storage conversion happens at the load/store boundary. Attention still
// accumulates in FP32 and retains its established BF16 rounding points.
struct FastllmNaiveKVRow {
    const void *data = nullptr;
    bool fp8 = false;
    const uint8_t *scales = nullptr;
    size_t column = 0;
    __host__ __device__ FastllmNaiveKVRow(const void *data = nullptr, bool fp8 = false,
            const uint8_t *scales = nullptr, size_t column = 0)
        : data(data), fp8(fp8), scales(scales), column(column) {}
    __device__ __forceinline__ FastllmNaiveKVRow operator+(size_t n) const {
        if (scales) return {data, false, scales, column + n};
        return {(const char *)data + n * (fp8 ? 1 : 2), fp8};
    }
    __device__ __forceinline__ FastllmNaiveKVRow operator-(size_t n) const {
        if (scales) return {data, false, scales, column - n};
        return {(const char *)data - n * (fp8 ? 1 : 2), fp8};
    }
    __device__ __forceinline__ __nv_bfloat16 operator[](size_t n) const {
        if (scales) return __float2bfloat16_rn(LoadFloat(n));
        return fp8 ? __float2bfloat16_rn((float)((const __nv_fp8_e4m3 *)data)[n])
                   : ((const __nv_bfloat16 *)data)[n];
    }
    // Every finite E4M3 value is exactly representable in BF16. Scalar
    // FP32 consumers can skip the redundant BF16 round-trip.
    __device__ __forceinline__ float LoadFloat(size_t n) const {
        if (scales) {
            n += column;
            const uint8_t packed = ((const uint8_t *)data)[n / 2];
            __nv_fp8_e4m3 scale; scale.__x = scales[n / 16];
            return FastllmFP4KVDecode((packed >> ((n & 1) * 4)) & 15) * float(scale);
        }
        return fp8 ? (float)((const __nv_fp8_e4m3 *)data)[n]
                   : (float)((const __nv_bfloat16 *)data)[n];
    }
    // E4M3 is exactly representable in FP16 and BF16. Pair conversion uses
    // the native FP8x2 instruction where available and preserves CUDA's NaNs.
    __device__ __forceinline__ static unsigned ConvertPair(unsigned short bits) {
        __half2 h = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
        float2 f = __half22float2(h);
        __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
        return *reinterpret_cast<unsigned *>(&b);
    }
    __device__ __forceinline__ uint4 Load8() const {
        if (scales) {
            if (!(column & 7) && !((uintptr_t)data & 3)) {
                uint4 values;
                LoadFP4Pairs<4>((unsigned *)&values);
                return values;
            }
            return make_uint4(FP4Pair(0), FP4Pair(2), FP4Pair(4), FP4Pair(6));
        }
        if (!fp8) return *(const uint4 *)data;
        uint2 packed = *(const uint2 *)data;
        return make_uint4(ConvertPair(packed.x), ConvertPair(packed.x >> 16),
                          ConvertPair(packed.y), ConvertPair(packed.y >> 16));
    }
    __device__ __forceinline__ uint2 Load4() const {
        if (scales) {
            if (!(column & 3) && !((uintptr_t)data & 1)) {
                uint2 values;
                LoadFP4Pairs<2>((unsigned *)&values);
                return values;
            }
            return make_uint2(FP4Pair(0), FP4Pair(2));
        }
        if (!fp8) return *(const uint2 *)data;
        unsigned packed = *(const unsigned *)data;
        return make_uint2(ConvertPair(packed), ConvertPair(packed >> 16));
    }
    __device__ __forceinline__ unsigned FP4Pair(int n) const {
        __nv_bfloat162 pair = __halves2bfloat162((*this)[n], (*this)[n + 1]);
        return *reinterpret_cast<unsigned *>(&pair);
    }
    // Caller guarantees 2/4-byte alignment and no crossing of a 16-value block.
    template<int Pairs>
    __device__ __forceinline__ void LoadFP4Pairs(unsigned *out) const {
        const auto *source = (const uint8_t *)data + column / 2;
        static_assert(Pairs == 2 || Pairs == 4 || Pairs == 8);
        const unsigned raw = Pairs >= 4 ? *(const unsigned *)source : *(const unsigned short *)source;
        const unsigned high = Pairs == 8 ? *(const unsigned *)(source + 4) : 0;
        const unsigned sf = scales[column / 16];
        // Keep CUDA's scalar NaN conversion. Every finite E2M1 * E4M3
        // product is exact in FP16 (including the smallest block scale).
        // Decode pairs with a byte permutation and a single packed multiply,
        // then convert to exactly the same BF16 values as scalar LoadFloat.
        if ((sf & 127) == 127) {
#pragma unroll
            for (int i = 0; i < Pairs; ++i) out[i] = FP4Pair(i * 2);
            return;
        }
        const __half2 scale = __nv_cvt_fp8x2_to_halfraw2(sf | (sf << 8), __NV_E4M3);
#pragma unroll
        for (int i = 0; i < Pairs; ++i) {
            const unsigned pair = ((i < 4 ? raw : high) >> ((i % 4) * 8)) & 255;
            const unsigned selector = (pair & 7) | (((pair >> 4) & 7) << 8);
            unsigned bits = (__byte_perm(0x3e3c3800, 0x46444240, selector) & 0x00ff00ff) << 8;
            bits |= ((pair & 8) << 12) | ((pair & 128) << 24);
            const float2 values = __half22float2(__hmul2(*reinterpret_cast<__half2 *>(&bits), scale));
            __nv_bfloat162 result = __floats2bfloat162_rn(values.x, values.y);
            out[i] = *reinterpret_cast<unsigned *>(&result);
        }
    }
};

struct FastllmNaiveKVWriteRow {
    void *data;
    bool fp8;
    struct Reference {
        void *data; size_t index; bool fp8;
        __device__ __forceinline__ void operator=(__nv_bfloat16 value) const {
            if (fp8) ((__nv_fp8_e4m3 *)data)[index] = __nv_fp8_e4m3((float)value);
            else ((__nv_bfloat16 *)data)[index] = value;
        }
    };
    __device__ __forceinline__ Reference operator[](size_t n) const { return {data, n, fp8}; }
    __device__ __forceinline__ FastllmNaiveKVWriteRow operator+(size_t n) const {
        return {(char *)data + n * (fp8 ? 1 : 2), fp8};
    }
};

// One scale per 128-dimensional Indexer key. Padding keeps both FP8 and
// BF16 main-K rows 16-byte aligned for vector loads and TMA attention.
constexpr int FastllmNaiveIndexerBytes = 144;
// All 128 or 256 CTA threads participate; only the first 128 own values.
__device__ __forceinline__ void FastllmNaiveStoreIndexer(float x, void *destination) {
    __shared__ float maxima[8];
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float maximum = t < 128 ? fabsf(x) : 0.f;
    for (int delta = 16; delta; delta >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    if (!lane) maxima[warp] = maximum;
    __syncthreads();
    maximum = 0.f;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i) maximum = fmaxf(maximum, maxima[i]);
    float scale = fmaxf(maximum, 1e-4f) / 448.f;
    if (t < 128) ((__nv_fp8_e4m3 *)destination)[t] =
        __nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, x / scale)));
    if (!t) {
        *(float *)((char *)destination + 128) = scale;
        // Deterministic padding makes raw archive copies reproducible.
        *(unsigned *)((char *)destination + 132) = 0;
        *(unsigned long long *)((char *)destination + 136) = 0;
    }
}

// Logical rows backed either by a dense tensor or a shared paged pool.
// Address translation leaves arithmetic, masks and TopK positions unchanged.
struct FastllmNaiveKV {
    const __nv_bfloat16 *base = nullptr;
    const int *pages = nullptr;
    int pageLen = 0, pageShift = -1;
    size_t offset = 0;
    bool fp8 = false, scaledIndexer = false, fp4 = false;

    __host__ __device__ FastllmNaiveKV(const void *data = nullptr) : base((const __nv_bfloat16 *)data) {}
    __host__ __device__ FastllmNaiveKV(const void *data, const int *table, int length,
            bool fp8 = false, bool fp4 = false)
        : base((const __nv_bfloat16 *)data), pages(table), pageLen(length), fp8(fp8), fp4(fp4) {
        if (length > 0 && (length & (length - 1)) == 0) {
#if defined(__CUDA_ARCH__)
            pageShift = __ffs(length) - 1;
#else
            pageShift = 0;
            for (int n = length; n > 1; n >>= 1) ++pageShift;
#endif
        }
    }
    explicit FastllmNaiveKV(const fastllm::Data &data)
        : FastllmNaiveKV((const __nv_bfloat16 *)(data.isPagedKVCache
            ? data.pagedKVCacheData->cudaData : data.cudaData),
            data.isPagedKVCache ? (const int *)data.cudaData : nullptr,
            data.isPagedKVCache ? data.pageLen : 0, data.dataType == fastllm::FP8_E4M3,
            data.dataType == fastllm::FP4_E2M1) {
        scaledIndexer = data.kvCacheFp8Indexer;
    }
    // Callers retain logical rows instead of reconstructing them from element
    // offsets. One page lookup serves every dimension loaded from the row.
    __device__ __forceinline__ int PhysicalRow(int row) const {
        if (pages) {
            int page = pageShift >= 0 ? row >> pageShift : row / pageLen;
            row += (pages[page] - page) * pageLen;
        }
        return row;
    }
    __device__ __forceinline__ FastllmNaiveKVRow Row(int row, int stride) const {
        row = PhysicalRow(row);
        if (fp4) {
            // Naive sparse KV copies opaque rows, including their scales.
            // Each row is [stride/2 packed bytes][stride/16 E4M3 scales].
            const auto *packed = (const uint8_t *)base + (size_t)row * (stride / 16 * 9);
            return {packed, false, packed + stride / 2, offset};
        }
        return FastllmNaiveKVRow(base, fp8) + ((size_t)row * stride + offset);
    }
    __device__ __forceinline__ FastllmNaiveKVRow IndexRow(int row, int stride) const {
        auto start = Row(row, stride);
        if (!scaledIndexer) return start + stride - 128;
        return {(const char *)start.data + stride * (fp8 ? 1 : 2) - FastllmNaiveIndexerBytes, true};
    }
    __device__ __forceinline__ float IndexScale(int row, int stride) const {
        return scaledIndexer ? *(const float *)((const char *)IndexRow(row, stride).data + 128) : 1.f;
    }
    __host__ __device__ FastllmNaiveKV &operator+=(size_t index) { offset += index; return *this; }
    __device__ __forceinline__ void Bind(const void *const *table, int part = 0) {
        if (table) {
            base = (const __nv_bfloat16 *)table[part];
            if (pageLen) pages = (const int *)table[2 + part];
        }
    }
};
