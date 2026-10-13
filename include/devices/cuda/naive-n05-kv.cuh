#pragma once
#include "fastllm.h"
#include <cuda_bf16.h>

// Logical BF16 rows backed either by a dense tensor or a shared paged pool.
// Address translation leaves arithmetic, masks and TopK positions unchanged.
struct FastllmNaiveKV {
    const __nv_bfloat16 *base = nullptr;
    const int *pages = nullptr;
    int pageLen = 0, pageShift = -1;
    size_t offset = 0;

    __host__ __device__ FastllmNaiveKV(const __nv_bfloat16 *data = nullptr) : base(data) {}
    __host__ __device__ FastllmNaiveKV(const __nv_bfloat16 *data, const int *table, int length)
        : base(data), pages(table), pageLen(length) {
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
            data.isPagedKVCache ? data.pageLen : 0) {}
    // Callers retain logical rows instead of reconstructing them from element
    // offsets. One page lookup serves every dimension loaded from the row.
    __device__ __forceinline__ int PhysicalRow(int row) const {
        if (pages) {
            int page = pageShift >= 0 ? row >> pageShift : row / pageLen;
            row += (pages[page] - page) * pageLen;
        }
        return row;
    }
    __device__ __forceinline__ const __nv_bfloat16 *Row(int row, int stride) const {
        row = PhysicalRow(row);
        return base + (size_t)row * stride + offset;
    }
    __host__ __device__ FastllmNaiveKV &operator+=(size_t index) { offset += index; return *this; }
    __device__ __forceinline__ void Bind(const void *const *table, int part = 0) {
        if (table) {
            base = (const __nv_bfloat16 *)table[part];
            if (pageLen) pages = (const int *)table[2 + part];
        }
    }
};
