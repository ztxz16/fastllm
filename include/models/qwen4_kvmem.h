#pragma once

#include "fastllm.h"
#include "kvmem.h"

namespace fastllm {
#ifdef USE_CUDA
// Exact QSA storage: logical positions never change. The complete KV is backed
// by pinned host memory; the device cache contains a bounded subset of pages.
// All selection and refill operations use the current CUDA stream, including
// graph replay. This object is owned jointly by a request's key/value tensors.
class Qwen4KvMem final : public KvMemCache {
public:
    static constexpr int pageTokens = 4;
    Qwen4KvMem(const KvMemConfig &config, DataType type, int heads, int dim, int device);
    ~Qwen4KvMem() override;
    Qwen4KvMem(const Qwen4KvMem &) = delete;
    Qwen4KvMem &operator=(const Qwen4KvMem &) = delete;

    void Attach(Data &key, Data &value, const std::shared_ptr<Qwen4KvMem> &owner);
    void Reset();
    void Prepare(const Data &indices, int rowStart, int rows, int length,
                 const int32_t *decodeMeta = nullptr, int appended = 1);
    bool HasResidentPrefix(int length) const { return prefixResident && length <= resident; }
    // Explicit stats reads synchronize; the inference path only updates the
    // logical length and device-local counters.
    const KvMemStats &Stats() const override;
    void SetLength(int tokens);
    void BeginTransaction(int tokens) override;
    void FinishTransaction(int accepted, Data &key, Data &value) override;
    void Append(const Data &, const Data &, const Data &, const Data &, Data &, Data &) override;

    void *hostKey = nullptr, *hostValue = nullptr;
    void *mappedKey = nullptr, *mappedValue = nullptr;
    void *deviceKey = nullptr, *deviceValue = nullptr;
    int *pageSlots = nullptr;
    int capacity = 0, resident = 0, heads = 0, dim = 0;
    size_t gpuBytes = 0;

private:
    void InitializeSlots();
    void Release() noexcept;
    int device, logicalPages, residentPages;
    size_t pageBytes;
    DataType type;
    KvMemConfig config;
    mutable KvMemStats stats;
    int *slotPages = nullptr, *needed = nullptr, *misses = nullptr,
        *available = nullptr, *counts = nullptr;
    int *metadata = nullptr;
    unsigned long long *metrics = nullptr;
    bool prefixResident = true;
};

inline Qwen4KvMem *Qwen4GetKvMem(const Data &cache) {
    return cache.kvMemCache ? dynamic_cast<Qwen4KvMem *>(cache.kvMemCache.get()) : nullptr;
}
#endif
}
