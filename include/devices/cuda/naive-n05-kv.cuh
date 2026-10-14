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
    __device__ __forceinline__ uint4 Load8() const {
        if (!fp8) return *(const uint4 *)data;
        uint2 packed = *(const uint2 *)data;
        uint4 result;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            __nv_fp8_e4m3 value;
            value.__x = ((const unsigned char *)&packed)[i];
            ((__nv_bfloat16 *)&result)[i] = __float2bfloat16_rn((float)value);
        }
        return result;
    }
    __device__ __forceinline__ uint2 Load4() const {
        if (!fp8) return *(const uint2 *)data;
        unsigned packed = *(const unsigned *)data;
        uint2 result;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __nv_fp8_e4m3 value;
            value.__x = ((const unsigned char *)&packed)[i];
            ((__nv_bfloat16 *)&result)[i] = __float2bfloat16_rn((float)value);
        }
        return result;
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

// Logical rows backed either by a dense tensor or a shared paged pool.
// Address translation leaves arithmetic, masks and TopK positions unchanged.
struct FastllmNaiveKV {
    const __nv_bfloat16 *base = nullptr;
    const int *pages = nullptr;
    int pageLen = 0, pageShift = -1;
    size_t offset = 0;
    bool fp8 = false;

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
            data.isPagedKVCache ? data.pageLen : 0, data.dataType == fastllm::FP8_E4M3) {}
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
    __host__ __device__ FastllmNaiveKV &operator+=(size_t index) { offset += index; return *this; }
    __device__ __forceinline__ void Bind(const void *const *table, int part = 0) {
        if (table) {
            base = (const __nv_bfloat16 *)table[part];
            if (pageLen) pages = (const int *)table[2 + part];
        }
    }
};
