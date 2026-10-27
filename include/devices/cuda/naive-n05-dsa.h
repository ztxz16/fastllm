#pragma once

// Eager callers use direct addresses; serving graphs can bind a fixed device
// table whose key/value entries are refreshed when the active requests change.
struct FastllmNaiveDSARow {
    const void *key = nullptr, *value = nullptr;
    const int *indices = nullptr, *liveKeys = nullptr;
    int length = 0, capacity = 0, keyStride = 0, selectedCount = 0;
    const void *const *cache = nullptr;
    const void *tensorMaps = nullptr;
    const int *keyPages = nullptr, *valuePages = nullptr;
    int pageLen = 0, poolTokens = 0;
    bool fp8 = false;
};

// Two Hopper tensor maps, stored in a 128-byte-aligned device allocation.
constexpr int FastllmNaiveDSAMapBytes = 256;
bool FastllmCudaNaiveDSAMapsSupported();
bool FastllmCudaNaiveEncodeDSAMaps(void *host, const void *key, const void *value,
                                 int capacity, int keyStride);

constexpr int FastllmNaiveDSABatchLimit = 32;
struct FastllmNaiveDSABatch {
    FastllmNaiveDSARow rows[FastllmNaiveDSABatchLimit];
};

#ifdef __CUDACC__
__device__ __forceinline__ FastllmNaiveDSARow FastllmNaiveResolveDSARow(FastllmNaiveDSARow row) {
    if (row.cache) { row.key = row.cache[0]; row.value = row.cache[1]; }
    if (row.cache && row.pageLen) {
        row.keyPages = (const int *)row.cache[2];
        row.valuePages = (const int *)row.cache[3];
    }
    return row;
}
__device__ __forceinline__ int FastllmNaivePhysicalRow(const FastllmNaiveDSARow &row, int token, bool value = false) {
    const int *pages = value ? row.valuePages : row.keyPages;
    if (!pages) return token;
    const int length = row.pageLen;
    const int page = (length & (length - 1)) == 0
        ? token >> (__ffs(length) - 1) : token / length;
    return token + (pages[page] - page) * length;
}
#endif
