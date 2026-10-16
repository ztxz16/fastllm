#pragma once
#include "fastllm.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Storage conversion happens at the load/store boundary. Attention still
// accumulates in FP32 and retains its established BF16 rounding points.
struct FastllmNaiveKVRow {
    const void *data = nullptr;
    bool fp8 = false;
    __host__ __device__ FastllmNaiveKVRow(const void *data = nullptr, bool fp8 = false)
        : data(data), fp8(fp8) {}
    __device__ __forceinline__ FastllmNaiveKVRow operator+(size_t n) const {
        return {(const char *)data + n * (fp8 ? 1 : 2), fp8};
    }
    __device__ __forceinline__ FastllmNaiveKVRow operator-(size_t n) const {
        return {(const char *)data - n * (fp8 ? 1 : 2), fp8};
    }
    __device__ __forceinline__ __nv_bfloat16 operator[](size_t n) const {
        return fp8 ? __float2bfloat16_rn((float)((const __nv_fp8_e4m3 *)data)[n])
                   : ((const __nv_bfloat16 *)data)[n];
    }
    // Every finite E4M3 value is exactly representable in BF16. Scalar
    // FP32 consumers can skip the redundant BF16 round-trip.
    __device__ __forceinline__ float LoadFloat(size_t n) const {
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
        if (!fp8) return *(const uint4 *)data;
        uint2 packed = *(const uint2 *)data;
        return make_uint4(ConvertPair(packed.x), ConvertPair(packed.x >> 16),
                          ConvertPair(packed.y), ConvertPair(packed.y >> 16));
    }
    __device__ __forceinline__ uint2 Load4() const {
        if (!fp8) return *(const uint2 *)data;
        unsigned packed = *(const unsigned *)data;
        return make_uint2(ConvertPair(packed), ConvertPair(packed >> 16));
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
    bool fp8 = false, scaledIndexer = false;

    __host__ __device__ FastllmNaiveKV(const void *data = nullptr) : base((const __nv_bfloat16 *)data) {}
    __host__ __device__ FastllmNaiveKV(const void *data, const int *table, int length, bool fp8 = false)
        : base((const __nv_bfloat16 *)data), pages(table), pageLen(length), fp8(fp8) {
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
            data.isPagedKVCache ? data.pageLen : 0, data.dataType == fastllm::FP8_E4M3) {
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
