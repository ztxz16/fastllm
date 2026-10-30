#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#if !defined(USE_ROCM) && CUDART_VERSION >= 11070
#include <cuda/atomic>
#endif

namespace fastllm {
namespace cuda {
namespace moe_admission {

#if !defined(USE_ROCM) && CUDART_VERSION >= 11070
// Only readers paired with release publication may overlap admissions.
__device__ inline int32_t LoadAdmission(const int32_t *key) {
    return ::cuda::atomic_ref<int32_t, ::cuda::thread_scope_device>(const_cast<int32_t &>(*key))
        .load(::cuda::memory_order_acquire);
}
#endif

// Update the device mapping and its host-visible mirror together. Call only
// after the admitted payload and all readers of the previous record are done.
__device__ inline void PublishAdmission(int32_t *keys, int32_t *slotKeys, int32_t *mappedKeys,
                                        unsigned long long *lastUsed, int key, int slot,
                                        unsigned long long age) {
    const int old = slotKeys[slot];
    if (old >= 0) {
        keys[old] = -1;
        if (mappedKeys)
            mappedKeys[old] = -1;
    }
    keys[key] = slot;
    slotKeys[slot] = key;
    lastUsed[slot] = age;
    if (mappedKeys)
        mappedKeys[key] = slot;
}

// Evict on the reader stream before a concurrent upload can overwrite a slot.
// The next replay may miss an admission in flight, but cannot read its old key.
static __global__ void InvalidateCacheAdmissions(int32_t *keys, int32_t *slotKeys, int32_t *mappedKeys,
                                                 const int32_t *changes, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const int slot = changes[count + i], old = slotKeys[slot];
    if (old >= 0) {
        keys[old] = -1;
        if (mappedKeys)
            mappedKeys[old] = -1;
    }
    slotKeys[slot] = -1;
}

static __global__ void PublishCacheAdmissions(int32_t *keys, int32_t *slotKeys, int32_t *mappedKeys,
                                              unsigned long long *lastUsed, unsigned long long *step,
                                              const int32_t *changes, int count, bool concurrent = false) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
#if !defined(USE_ROCM) && CUDART_VERSION >= 11070
    if (concurrent) {
        const int key = changes[i], slot = changes[count + i];
        slotKeys[slot] = key;
        lastUsed[slot] = atomicAdd(step, 1ULL) + 1;
        if (mappedKeys)
            mappedKeys[key] = slot;
        // A native route's acquire lookup observes both metadata and payload.
        ::cuda::atomic_ref<int32_t, ::cuda::thread_scope_device>(keys[key]).store(
            slot, ::cuda::memory_order_release);
        return;
    }
#endif
    PublishAdmission(keys, slotKeys, mappedKeys, lastUsed, changes[i], changes[count + i],
                     atomicAdd(step, 1ULL) + 1);
}

} // namespace moe_admission
} // namespace cuda
} // namespace fastllm
